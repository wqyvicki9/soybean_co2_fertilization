# -*- coding: utf-8 -*-
"""
Compute Corn-derived background trend using FIXED reliable county set (aggregated).

关键改动 vs 旧版:
  1. 所有 seed 都用同一个 fixed county set (reliable_counties_aggregated.csv), 
     不再用 per-seed reliable CSV
  2. 每个 seed 算一份 background trend (方法不变: full - year_fixed)
  3. 最后对每个 county 跨 20 seed 取 median, 保存一份 "median trend" 文件,
     后续 soybean 主模型用这一份 (不再是 per-seed)

流程:
  Step 1: 读 reliable_counties_aggregated.csv -> FIXED_COUNTY_SET
  Step 2: 对每个 seed 0..19:
    a. Load corn model + data + scaler
    b. Predict full + Year-fixed
    c. diff = full - year_fixed
    d. 筛 FIXED_COUNTY_SET
    e. 5-yr rolling smooth
    f. 保存 per-seed pickle
  Step 3: 对每个 (county, year), 跨 20 seed 取 median, 保存一份 median trend
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import gc


# ============================================================================
# 配置
# ============================================================================
folder                = '.'   # repo root (contains Data/ and Results/); run from here
corn_results_name     = "Results_random_seeds_20"
figure_path_root      = os.path.join(folder, 'Results', 'Figures')
output_dir            = os.path.join(folder, 'Results', 'corn_trend_per_seed')
figure_path           = os.path.join(figure_path_root, 'corn_trend_per_seed')
os.makedirs(output_dir,  exist_ok=True)
os.makedirs(figure_path, exist_ok=True)

# Fixed reliable county set (output of 1-Reliable_counties/aggregate_reliable_counties.py)
reliable_csv_aggregated = os.path.join(folder, 'Results', 'reliable_counties_per_seed',
                                       'reliable_counties_aggregated.csv')

crop            = 'Corn'
TAG             = "nn_time"
SMOOTH_WIN      = 5
RANDOM_SEEDS    = list(range(20))

# State abbreviation mapping (for plotting)
fips_to_abbr = {
    '01':'AL','04':'AZ','05':'AR','06':'CA','08':'CO','09':'CT','10':'DE',
    '11':'DC','12':'FL','13':'GA','16':'ID','17':'IL','18':'IN','19':'IA',
    '20':'KS','21':'KY','22':'LA','23':'ME','24':'MD','25':'MA','26':'MI',
    '27':'MN','28':'MS','29':'MO','30':'MT','31':'NE','32':'NV','33':'NH',
    '34':'NJ','35':'NM','36':'NY','37':'NC','38':'ND','39':'OH','40':'OK',
    '41':'OR','42':'PA','44':'RI','45':'SC','46':'SD','47':'TN','48':'TX',
    '49':'UT','50':'VT','51':'VA','53':'WA','54':'WV','55':'WI','56':'WY'
}


# ============================================================================
# Step 1: 读 FIXED county set
# ============================================================================
print("=" * 70)
print("Step 1: Load fixed reliable county set")
print("=" * 70)

reliable_df = pd.read_csv(reliable_csv_aggregated, dtype={'FIPS': str})
reliable_df['FIPS'] = reliable_df['FIPS'].str.zfill(5)
FIXED_COUNTY_SET = set(reliable_df['FIPS'])
print(f"Fixed county set: {len(FIXED_COUNTY_SET)} counties")
print(f"  Source: {reliable_csv_aggregated}")


# ============================================================================
# Load shared feature list
# ============================================================================
nn_time_features = pickle.load(
    open(f'{folder}/Results/{corn_results_name}/{crop}_basetime.p', 'rb'))
print(f"\nLoaded {len(nn_time_features)} features for {TAG}")

if 'Year' not in nn_time_features:
    raise ValueError("'Year' not in feature list! Cannot fix year-trend.")
has_year_squared = 'Year_squared' in nn_time_features


def predict_year_fixed(df, feature_list, model, scaler_X, scaler_Y):
    """Predict with Year fixed at 1979 (and Year_squared if it exists)."""
    df_fixed = df[feature_list].copy()
    year_min = df[feature_list]['Year'].min()
    df_fixed['Year'] = year_min
    if has_year_squared:
        df_fixed['Year_squared'] = year_min ** 2
    X_scaled    = scaler_X.transform(df_fixed)
    pred_scaled = model.predict(X_scaled, verbose=0).ravel()
    return scaler_Y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()


# ============================================================================
# Step 2: Per-seed background trend
# ============================================================================
print("\n" + "=" * 70)
print("Step 2: Compute per-seed background trend (on FIXED county set)")
print("=" * 70)

all_seed_trends = []   # 收集每个 seed 的 df, 最后 concat 算 median

summary_rows = []

for rs in RANDOM_SEEDS:
    print(f"\n--- seed = {rs} ---")

    # Load
    model_path  = f'{folder}/Results/{corn_results_name}/{crop}_model_{TAG}_rs{rs}.p'
    test_path   = f'{folder}/Results/{corn_results_name}/{crop}_run_test_{TAG}_rs{rs}.p'
    train_path  = f'{folder}/Results/{corn_results_name}/{crop}_run_train_{TAG}_rs{rs}.p'
    scaler_path = f'{folder}/Results/{corn_results_name}/{crop}_scaler_{TAG}_rs{rs}.p'

    model              = pickle.load(open(model_path,  'rb'))
    test_df            = pickle.load(open(test_path,   'rb')).reset_index(drop=True)
    train_df           = pickle.load(open(train_path,  'rb')).reset_index(drop=True)
    scaler_X, scaler_Y = pickle.load(open(scaler_path, 'rb'))

    # Full + year-fixed prediction
    def full_predict(df):
        X_scaled = scaler_X.transform(df[nn_time_features])
        pred_scaled = model.predict(X_scaled, verbose=0).ravel()
        return scaler_Y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    pred_orig_test   = full_predict(test_df)
    pred_orig_train  = full_predict(train_df)
    pred_fixed_test  = predict_year_fixed(test_df,  nn_time_features, model, scaler_X, scaler_Y)
    pred_fixed_train = predict_year_fixed(train_df, nn_time_features, model, scaler_X, scaler_Y)

    diff_test  = pred_orig_test  - pred_fixed_test
    diff_train = pred_orig_train - pred_fixed_train

    df_all = pd.concat([
        test_df[['FIPS',  'Year_delta', 'Yield']].assign(diff=diff_test),
        train_df[['FIPS', 'Year_delta', 'Yield']].assign(diff=diff_train),
    ], ignore_index=True)
    df_all['FIPS']        = df_all['FIPS'].astype(str).str.zfill(5)
    df_all['state']       = df_all['FIPS'].str[:2]
    df_all['Year_actual'] = (df_all['Year_delta'] + 1979).astype(int)

    # ── Filter FIXED county set ──
    df_r = df_all[df_all['FIPS'].isin(FIXED_COUNTY_SET)].copy()
    df_r = df_r.sort_values(['FIPS', 'Year_actual']).reset_index(drop=True)

    # 5-yr rolling smooth per county
    df_r['diff_smooth'] = (
        df_r.groupby('FIPS')['diff']
        .transform(lambda x: x.rolling(SMOOTH_WIN, center=True,
                                       min_periods=max(2, SMOOTH_WIN // 2)).mean())
    )

    print(f"  county-year rows: {len(df_r)}")
    print(f"  unique counties with data: {df_r['FIPS'].nunique()}")

    # Save per-seed pickle
    save_df = df_r[['FIPS', 'Year_actual', 'state', 'Yield', 'diff', 'diff_smooth']].copy()
    out_pkl = f'{output_dir}/{crop}_county_year_tech_diff_smooth_rs{rs}.p'
    pickle.dump(save_df, open(out_pkl, 'wb'))
    save_df.to_csv(f'{output_dir}/{crop}_county_year_tech_diff_smooth_rs{rs}.csv', index=False)
    print(f"  ✓ Saved per-seed → {out_pkl}")

    # 保存副本到 all_seed_trends (加 rs 列)
    save_df['rs'] = rs
    all_seed_trends.append(save_df[['FIPS', 'Year_actual', 'state', 'diff', 'diff_smooth', 'rs']])

    # Summary
    summary_rows.append({
        'rs':              rs,
        'n_counties':      df_r['FIPS'].nunique(),
        'median_diff_2023': df_r[df_r['Year_actual'] == 2023]['diff_smooth'].median(),
        'median_diff_1990': df_r[df_r['Year_actual'] == 1990]['diff_smooth'].median(),
    })

    # Cleanup
    del model, test_df, train_df, scaler_X, scaler_Y, df_all, df_r
    tf.keras.backend.clear_session()
    gc.collect()


# ============================================================================
# Step 3: Aggregate -- median trend across 20 seeds per (county, year)
# ============================================================================
print("\n" + "=" * 70)
print("Step 3: Compute median trend across 20 seeds")
print("=" * 70)

combined = pd.concat(all_seed_trends, ignore_index=True)
print(f"Combined: {len(combined)} rows (county × seed × year)")

# Per (county, year): median of diff_smooth across 20 seeds
median_trend = (
    combined.groupby(['FIPS', 'Year_actual'])
    .agg(
        diff_smooth_median=('diff_smooth', 'median'),
        diff_median=('diff', 'median'),
        state=('state', 'first'),   # state 不会变
        n_seeds=('rs', 'nunique'),
    )
    .reset_index()
    .sort_values(['FIPS', 'Year_actual'])
    .reset_index(drop=True)
)

# 注意: diff_smooth_median 是把 "per-seed smoothed" 再取 median;
# 如果你想 "median trend 本身再做一次 smooth", 可以对 median 再 rolling一下
# 这里我选前者 (每个 seed 先 smooth, 再取 median), 因为和 per-seed 结果一致

print(f"Median trend: {len(median_trend)} rows, "
      f"{median_trend['FIPS'].nunique()} counties")

# Save
out_median_pkl = f'{output_dir}/{crop}_county_year_tech_median_trend.p'
out_median_csv = f'{output_dir}/{crop}_county_year_tech_median_trend.csv'
pickle.dump(median_trend, open(out_median_pkl, 'wb'))
median_trend.to_csv(out_median_csv, index=False)
print(f"\n✓ Saved median trend → {out_median_pkl}")
print(f"✓ Saved median trend → {out_median_csv}")


# ============================================================================
# Step 4: 画 median trend 图 (和 per-seed 同风格)
# ============================================================================
print("\nPlotting median trend ...")

fig, ax = plt.subplots(figsize=(8, 5))

states_list = sorted(median_trend['state'].unique())
colors_20   = [cm.tab20(i / 19)  for i in range(20)]
colors_20b  = [cm.tab20b(i / 19) for i in range(20)]
all_colors  = colors_20 + colors_20b
state_colors = {s: all_colors[i % len(all_colors)] for i, s in enumerate(states_list)}

# 县级细线
for fips_id, grp in median_trend.groupby('FIPS'):
    state = grp['state'].iloc[0]
    ax.plot(grp['Year_actual'], grp['diff_smooth_median'],
            color=state_colors[state],
            linewidth=0.3, alpha=0.15, zorder=1)

# 州 median
state_annual = (
    median_trend.groupby(['state', 'Year_actual'])['diff_smooth_median']
    .median().reset_index()
)
for state, grp in state_annual.groupby('state'):
    abbr = fips_to_abbr.get(state, state)
    ax.plot(grp['Year_actual'], grp['diff_smooth_median'],
            color=state_colors[state],
            linewidth=1.2, alpha=0.85, linestyle=':', label=abbr, zorder=3)

# 全美 median
overall = median_trend.groupby('Year_actual')['diff_smooth_median'].median()
ax.plot(overall.index, overall.values,
        color='black', linewidth=2.5, linestyle='-',
        label='U.S. Median', zorder=5)

ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Yield Difference (bu/acre)', fontsize=11)
ax.set_title(
    f'Corn-derived Background Trend (median across 20 seeds, '
    f'fixed reliable county set, n={len(FIXED_COUNTY_SET)})',
    fontsize=10)
ax.grid(True, linestyle=':', alpha=0.4)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.28),
          ncol=9, fontsize=7, columnspacing=0.5, handlelength=1.5,
          frameon=False)
ax.tick_params(axis='both', labelsize=10, direction='in')

plt.tight_layout()
out_fig = f'{figure_path}/{crop}_background_trend_median_20seeds.pdf'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"✓ Saved figure → {out_fig}")


# ============================================================================
# Summary
# ============================================================================
summary_df = pd.DataFrame(summary_rows).sort_values('rs').reset_index(drop=True)
summary_csv = f'{output_dir}/per_seed_trend_summary.csv'
summary_df.to_csv(summary_csv, index=False)

print("\n" + "=" * 70)
print("PER-SEED SUMMARY")
print("=" * 70)
print(summary_df.to_string(index=False))
print("=" * 70)

print(f"\n✓ Done.")
print(f"  Per-seed trends:     {output_dir}/{crop}_county_year_tech_diff_smooth_rs{{0..19}}.p")
print(f"  Median trend (USE):  {out_median_pkl}")
print(f"  Figure:              {out_fig}")
