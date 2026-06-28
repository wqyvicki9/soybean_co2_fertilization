# -*- coding: utf-8 -*-
"""
Corn-Soy climate-residual growth rate correlation.
每个 random seed 单独算, 各自存一份 reliable counties CSV。
seed=k 的 CSV 会被后续 seed=k 的 soy 主模型用作 county 筛选集。
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
from scipy import stats


# ============================================================================
# 配置
# ============================================================================
folder             = '.'   # repo root (contains Data/ and Results/); run from here
soy_results_name   = "Results_Soy_ClimateOnly_20seeds"
corn_results_name  = "Results_Corn_ClimateOnly_20seeds"   # 新跑的 corn climate-only 结果
figure_path        = os.path.join(folder, 'Results', 'Figures')
reliable_dir       = os.path.join(folder, 'Results', 'reliable_counties_per_seed')
os.makedirs(figure_path,   exist_ok=True)
os.makedirs(reliable_dir,  exist_ok=True)

MODEL_TYPE      = 'nn'        # 'nn' 或 'linear'
SMOOTH_WIN      = 5           # rolling mean window
LAG_RANGE       = range(-20, 21)
P_THRESHOLD     = 0.01

# 设置画图开关: 每个 seed 都画一张图 → 20 张 figure
MAKE_FIGURE_PER_SEED = True
REPRESENTATIVE_SEED  = 0       # (仅在 MAKE_FIGURE_PER_SEED=False 时使用)

MAP_XLIM = (-105, -67)
MAP_YLIM = (25, 50)
shp_dir  = os.path.join(folder, "Data", "county_shp")   # US county shapefile (supply separately; used only for the maps)


# ============================================================================
# 1. 加载所有 seed 的 predictions
# ============================================================================
def load_predictions_per_seed(results_folder_name, crop_name, model_type):
    pred_file = f'{folder}/Results/{results_folder_name}/{crop_name}_predictions_{model_type}.p'
    preds = pickle.load(open(pred_file, 'rb'))
    print(f"[{crop_name} {model_type}] Loaded {len(preds)} seeds from {pred_file}")

    per_seed = {}
    for p in preds:
        rs = p['rs']
        train = p['train_meta'].copy()
        train['predicted'] = np.asarray(p['train_pred']).ravel()
        test = p['test_meta'].copy()
        test['predicted'] = np.asarray(p['test_pred']).ravel()
        df = pd.concat([train, test], ignore_index=True)
        df['FIPS'] = df['FIPS'].astype(str).str.zfill(5)
        df['residual'] = df['Yield'] - df['predicted']
        per_seed[rs] = df
    return per_seed


print("Loading Soy and Corn predictions ...")
soy_per_seed  = load_predictions_per_seed(soy_results_name,  'Soy',  MODEL_TYPE)
corn_per_seed = load_predictions_per_seed(corn_results_name, 'Corn', MODEL_TYPE)

common_seeds = sorted(set(soy_per_seed.keys()) & set(corn_per_seed.keys()), key=int)
print(f"\nCommon seeds: {common_seeds}")
if len(common_seeds) == 0:
    raise ValueError("No common seeds between Soy and Corn!")


# ============================================================================
# 2. 核心: per-seed per-county correlation
# ============================================================================
def compute_correlation_for_county(group, smooth_win=SMOOTH_WIN, lag_range=LAG_RANGE):
    """对单个 county 的 group, 算 lag=0 r/p 和 best_lag / best r."""
    g = group.sort_values('Year_actual').reset_index(drop=True)
    gr_soy  = g['residual_soy'].diff()
    gr_corn = g['residual_corn'].diff()
    gr_soy_s  = gr_soy.rolling(smooth_win, center=True,
                               min_periods=max(2, smooth_win // 2)).mean()
    gr_corn_s = gr_corn.rolling(smooth_win, center=True,
                                min_periods=max(2, smooth_win // 2)).mean()

    best_lag, best_r, best_p = 0, -np.inf, np.nan
    for lag in lag_range:
        if lag >= 0:
            x = gr_soy_s.iloc[lag:].values
            y = gr_corn_s.iloc[:len(gr_corn_s) - lag].values if lag > 0 else gr_corn_s.values
        else:
            x = gr_soy_s.iloc[:len(gr_soy_s) + lag].values
            y = gr_corn_s.iloc[-lag:].values
        n_min = min(len(x), len(y))
        x = x[:n_min]; y = y[:n_min]
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 5:
            continue
        r, p = stats.pearsonr(x[mask], y[mask])
        if r > best_r:
            best_r, best_p, best_lag = r, p, lag

    mask0 = ~(np.isnan(gr_soy_s) | np.isnan(gr_corn_s))
    if mask0.sum() < 5:
        r_0, p_0 = np.nan, np.nan
    else:
        r_0, p_0 = stats.pearsonr(gr_soy_s[mask0], gr_corn_s[mask0])

    return pd.Series({
        'r_lag0':   r_0,
        'p_lag0':   p_0,
        'best_lag': best_lag,
        'r_best':   best_r if np.isfinite(best_r) else np.nan,
        'p_best':   best_p,
        'n_years':  len(g),
    })


def per_seed_correlation(soy_df, corn_df):
    """给定某 seed 的 soy 和 corn panel, 返回 per-county correlation + n_both_crops."""
    merged = soy_df[['FIPS', 'Year_actual', 'residual']].rename(
        columns={'residual': 'residual_soy'}
    ).merge(
        corn_df[['FIPS', 'Year_actual', 'residual']].rename(
            columns={'residual': 'residual_corn'}),
        on=['FIPS', 'Year_actual'], how='inner',
    )
    per_county = merged.groupby('FIPS').apply(
        compute_correlation_for_county).reset_index()
    return per_county, merged['FIPS'].nunique()


# ============================================================================
# 3. 提前加载一次 shapefile (所有 seed 通用)
# ============================================================================
print("\nLoading shapefile (once, shared across seeds) ...")
shp_path = None
for f in os.listdir(shp_dir):
    if f.endswith('.shp'):
        shp_path = os.path.join(shp_dir, f)
        break
if shp_path is None:
    raise FileNotFoundError(f"No .shp file in {shp_dir}")

counties_gdf = gpd.read_file(shp_path)
contig_states = [
    "01","04","05","06","08","09","10","11","12","13",
    "16","17","18","19","20","21","22","23","24","25",
    "26","27","28","29","30","31","32","33","34","35",
    "36","37","38","39","40","41","42","44","45","46",
    "47","48","49","50","51","53","54","55","56",
]
counties_gdf = counties_gdf[counties_gdf["STATEFP"].isin(contig_states)].copy()
counties_gdf["FIPS"] = counties_gdf["GEOID"].astype(str).str.zfill(5)
counties_gdf = counties_gdf.to_crs("EPSG:4326")
states_plot = counties_gdf.dissolve(by='STATEFP')
gdf_east = counties_gdf[
    (counties_gdf.geometry.centroid.x >= MAP_XLIM[0]) &
    (counties_gdf.geometry.centroid.x <= MAP_XLIM[1]) &
    (counties_gdf.geometry.centroid.y >= MAP_YLIM[0]) &
    (counties_gdf.geometry.centroid.y <= MAP_YLIM[1])
].copy()


# ============================================================================
# 4. 画图函数 (被 per-seed 循环调用)
# ============================================================================
def plot_map_for_seed(rs, per_county, n_sig, n_both, min_r_in_sig,
                      intersection_fips_this):
    """画 (a) 20 seed mean r map + (b) best lag map, outline 这个 seed 的 sig county."""
    gdf_r = gdf_east.merge(
        per_county[['FIPS', 'r_lag0', 'best_lag']],
        on='FIPS', how='left'
    ).rename(columns={'r_lag0': 'r'})

    # Mask best_lag: 只保留 p<P_THRESHOLD 的 county (sig), 其他设为 NaN
    # 这样右图 (b) 只画 sig counties, 其他变成浅灰
    gdf_r['best_lag_sig_only'] = gdf_r['best_lag'].where(
        gdf_r['FIPS'].isin(intersection_fips_this)
    )

    gdf_sig_orig = gdf_east[gdf_east['FIPS'].isin(intersection_fips_this)].copy()
    gdf_sig_dissolved = gdf_sig_orig[['geometry']].dissolve() if len(gdf_sig_orig) > 0 else None

    fig, axes = plt.subplots(1, 2, figsize=(8, 5), facecolor='white',
                             gridspec_kw={'wspace': 0.0})
    norm_r   = mcolors.Normalize(vmin=0, vmax=1)
    norm_lag = mcolors.TwoSlopeNorm(vcenter=0, vmin=-20, vmax=20)

    # (a) r map
    ax = axes[0]
    gdf_r.boundary.plot(ax=ax, linewidth=0.02, edgecolor='0.6', zorder=1)
    gdf_r.plot(column='r', ax=ax, cmap='RdYlGn_r', norm=norm_r,
               linewidth=0, edgecolor='none', zorder=2,
               missing_kwds={'color': 'whitesmoke', 'edgecolor': '0.8', 'linewidth': 0.02})
    states_plot.boundary.plot(ax=ax, edgecolor='0.1', linewidth=0.2, zorder=3)
    if gdf_sig_dissolved is not None:
        gdf_sig_dissolved.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5, zorder=4)
    ax.set_xlim(*MAP_XLIM); ax.set_ylim(*MAP_YLIM); ax.axis('off')
    ax.text(0.01, 0.97, '(a)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

    if np.isfinite(min_r_in_sig):
        label_str = (f'{n_sig} / {n_both} counties\n'
                     f'(p < {P_THRESHOLD}, r ≥ {min_r_in_sig:.2f})')
    else:
        label_str = f'{n_sig} / {n_both} counties'
    ax.text(0.99, 0.03, label_str,
            transform=ax.transAxes,
            fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='black', linewidth=0.5, alpha=0.9))

    sm_r = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm_r)
    sm_r.set_array([])
    cb = fig.colorbar(sm_r, ax=ax, orientation='horizontal',
                      shrink=0.6, pad=0.06, aspect=15)
    cb.ax.set_title('Correlation Coefficient of Corn & Soy\nResidual Yield Growth Rate',
                    fontsize=10)
    cb.ax.tick_params(labelsize=10, direction='in')
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1])

    # (b) lag map
    ax = axes[1]
    gdf_r.boundary.plot(ax=ax, linewidth=0.02, edgecolor='0.6', zorder=1)
    gdf_r.plot(column='best_lag_sig_only', ax=ax, cmap='RdYlBu_r', norm=norm_lag,
               linewidth=0, edgecolor='none', zorder=2,
               missing_kwds={'color': 'whitesmoke', 'edgecolor': '0.8', 'linewidth': 0.02})
    states_plot.boundary.plot(ax=ax, edgecolor='0.1', linewidth=0.2, zorder=3)
    ax.set_xlim(*MAP_XLIM); ax.set_ylim(*MAP_YLIM); ax.axis('off')
    ax.text(0.01, 0.97, '(b)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

    sm_lag = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm_lag)
    sm_lag.set_array([])
    cb2 = fig.colorbar(sm_lag, ax=ax, orientation='horizontal',
                       shrink=0.6, pad=0.06, aspect=15)
    cb2.set_label('Years', fontsize=10, labelpad=2)
    cb2.ax.set_title('Lag of Corn and Soybean Residual Yield\n'
                     'Growth Rateat Maximum Correlation', fontsize=10)
    cb2.ax.tick_params(labelsize=10, direction='in')
    cb2.set_ticks([-20, -10, 0, 10, 20])

    plt.suptitle(
        f'Corn–Soy Residual Growth Rate Correlation (seed={rs})\n'
        f'(residual = observed − climate-only [{MODEL_TYPE.upper()}], '
        f'{SMOOTH_WIN}-yr rolling mean)',
        fontsize=9, y=0.95)

    plt.tight_layout()
    out = f'{figure_path}/Soy_corn_residual_correlation_map_{MODEL_TYPE}_rs{rs}.pdf'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved figure → {out}")
    plt.close(fig)


# ============================================================================
# 5. 主循环: per-seed 独立处理
# ============================================================================
print("\n" + "=" * 60)
print("Per-seed correlation analysis")
print("=" * 60)

summary_rows = []

for rs in common_seeds:
    print(f"\n--- seed = {rs} ---")

    per_county, n_both = per_seed_correlation(soy_per_seed[rs], corn_per_seed[rs])

    # 筛显著 county
    sig = per_county[
        (per_county['p_lag0'] < P_THRESHOLD) &
        (per_county['r_lag0'].notna())
    ].copy()
    sig = sig.sort_values('r_lag0').reset_index(drop=True)

    n_sig = len(sig)
    min_r = sig['r_lag0'].min() if n_sig > 0 else np.nan
    max_r = sig['r_lag0'].max() if n_sig > 0 else np.nan

    print(f"  both-crop counties:       {n_both}")
    print(f"  significant (p<{P_THRESHOLD}):     {n_sig}")
    if n_sig > 0:
        print(f"  → MIN r at p<{P_THRESHOLD}:   {min_r:.4f}   "
              f"(this is the lower bound of 'r' among sig counties)")
        print(f"    max r:   {max_r:.4f}")
        print(f"    mean r:  {sig['r_lag0'].mean():.4f}")
        print(f"    median r:{sig['r_lag0'].median():.4f}")

    # 存 per-seed CSV
    csv_out = f'{reliable_dir}/reliable_counties_NN_rs{rs}.csv'
    sig.to_csv(csv_out, index=False)
    print(f"  ✓ Saved CSV → {csv_out}")

    # 画图 (视配置)
    if MAKE_FIGURE_PER_SEED or (rs == REPRESENTATIVE_SEED):
        intersection_fips_this = set(sig['FIPS'].astype(str).str.zfill(5))
        plot_map_for_seed(rs, per_county, n_sig, n_both, min_r, intersection_fips_this)

    summary_rows.append({
        'rs':     rs,
        'n_both': n_both,
        'n_sig':  n_sig,
        'min_r':  min_r,
        'max_r':  max_r,
        'mean_r': sig['r_lag0'].mean() if n_sig > 0 else np.nan,
        'median_r': sig['r_lag0'].median() if n_sig > 0 else np.nan,
    })

# ============================================================================
# 6. 汇总 table
# ============================================================================
summary_df = pd.DataFrame(summary_rows)
summary_csv = f'{figure_path}/per_seed_correlation_summary_NN.csv'
summary_df.to_csv(summary_csv, index=False)

print("\n" + "=" * 60)
print("PER-SEED SUMMARY")
print("=" * 60)
print(summary_df.to_string(index=False))
print("=" * 60)

print(f"\nMedian across 20 seeds:")
print(f"  n_sig  median = {summary_df['n_sig'].median():.0f}")
print(f"  min r  median = {summary_df['min_r'].median():.4f}")
print(f"  mean r median = {summary_df['mean_r'].median():.4f}")

print(f"\n✓ Per-seed CSVs saved to: {reliable_dir}/")
print(f"✓ Summary saved to: {summary_csv}")
