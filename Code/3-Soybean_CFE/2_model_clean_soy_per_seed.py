# -*- coding: utf-8 -*-
"""
Soy full model, per random seed.

Core idea: the seed=k soy model uses the seed=k corn background trend and the seed=k reliable counties.

For each seed k in 0..19:
  1. Load the seed=k corn trend: Corn_county_year_tech_diff_smooth_rs{k}.p
     Merge the trend into the soy CAMS data, cams['Year'] = Tech_trend (diff_smooth)
  2. Load the seed=k reliable counties: reliable_counties_NN_rs{k}.csv
     Filter source_df to keep only these counties
  3. Run 8 blocks with the same architecture as corn:
       linear {baseline, time, co2, time_co2} + NN {baseline, time, co2, time_co2}
  4. Save a separate per-seed file for each (block, seed)
  5. Free memory immediately after running
"""

import warnings
warnings.filterwarnings('ignore')
import gc
import pandas
import numpy as np
import random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import math
import pickle
import os
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================
random_seed_list    = list(range(20))
folder              = '.'   # repo root (contains Data/ and Results/); run from here
results_folder_name = "Results_Soy_FullModel_20seeds"

# Per-seed input paths
corn_trend_dir      = os.path.join(folder, 'Results', 'corn_trend_per_seed')
reliable_dir        = os.path.join(folder, 'Results', 'reliable_counties_per_seed')

crop = 'Soy'
start_year_base = 1979
end_year        = 2023
year_datapoint_threshold = int((end_year - start_year_base) * 0.8)

out_dir = os.path.join(folder, 'Results', results_folder_name)
os.makedirs(out_dir, exist_ok=True)


# ============================================================================
# Data preprocessing (the part common to all seeds: CSV loading + county filter)
# ============================================================================
print("Loading Soy CAMS data ...")
crop_df = pandas.read_csv(
    os.path.join(folder, 'Data', 'Aggregated_yield_climate_co2', crop + '_CAMS_1979_2023_allcounties.csv'))
crop_df['FIPS'] = crop_df.FIPS.apply(lambda x: f"{int(x):05}")
crop_df = crop_df[(crop_df.Year <= end_year) & (crop_df.Year >= start_year_base)]
fips_count = crop_df.FIPS.value_counts()
fips = crop_df.FIPS.unique()
fips = [i for i in fips if fips_count[i] > year_datapoint_threshold]
crop_df = crop_df[crop_df.FIPS.isin(fips)].reset_index(drop=True)
print(f"Number of counties (before trend/reliable filter): {crop_df['FIPS'].nunique()}")


# ============================================================================
# Feature engineering & utilities
# ============================================================================
def create_features(df):
    df['Precipitation_Early'] = df[['ppt_May', 'ppt_Jun']].mean(axis=1)
    df['Precipitation_Mid']   = df[['ppt_Jul', 'ppt_Aug']].mean(axis=1)
    df['Precipitation_Late']  = df[['ppt_Sep', 'ppt_Oct']].mean(axis=1)
    df['Tmin_Early'] = df[['tmin_May', 'tmin_Jun']].mean(axis=1)
    df['Tmin_Mid']   = df[['tmin_Jul', 'tmin_Aug']].mean(axis=1)
    df['Tmin_Late']  = df[['tmin_Sep', 'tmin_Oct']].mean(axis=1)
    df['Tmax_Early'] = df[['tmax_May', 'tmax_Jun']].mean(axis=1)
    df['Tmax_Mid']   = df[['tmax_Jul', 'tmax_Aug']].mean(axis=1)
    df['Tmax_Late']  = df[['tmax_Sep', 'tmax_Oct']].mean(axis=1)
    df['CO2_Early']  = df[['co2_May', 'co2_Jun']].mean(axis=1)
    df['CO2_Mid']    = df[['co2_Jul', 'co2_Aug']].mean(axis=1)
    df['CO2_Late']   = df[['co2_Sep', 'co2_Oct']].mean(axis=1)
    df['CO2']        = df[['CO2_Early', 'CO2_Mid', 'CO2_Late']].mean(axis=1)
    return df


def alter_input(df, input_cols):
    """Relative-anomaly normalization: x / global_mean(x)."""
    df_norm = df.copy()
    feature_mean = {}
    for c in input_cols:
        mu = df[c].mean()
        feature_mean[c] = mu
        df_norm[c] = df[c] / mu if mu != 0 else 0.0

    county_feature_trends = [{}, {}]
    for county in df.FIPS.unique():
        county_feature_trends[0][county] = {}
        county_feature_trends[1][county] = {}
        for c in input_cols:
            county_feature_trends[0][county][c] = feature_mean[c]
            county_feature_trends[1][county][c] = feature_mean[c]
    return df_norm, county_feature_trends


def encode_and_bind(df, encode_col):
    dummies = pandas.get_dummies(df[[encode_col]], prefix='Spatial', dtype=int)
    return pandas.concat([df, dummies], axis=1)


def train_test_split_county(df):
    train, test = [], []
    county_model_data = dict(list(df.groupby(['FIPS'])))
    for county in df.FIPS.unique():
        county_df = county_model_data[(county,)].sample(frac=1.0).copy()
        county_df = county_df.sort_values('Year_original').copy()   # sort by Year_original
        n = int(max(2, int(county_df.shape[0] * 0.2)))
        early = county_df.iloc[:n, :].copy()
        late  = county_df.iloc[-n:, :].copy()
        mid   = county_df.iloc[n:-n, :].copy()
        edge_n = int(max(1, int(early.shape[0] * 0.2)))
        mid_n  = math.ceil(0.2 * county_df.shape[0]) - (2 * edge_n)
        early = early.sample(frac=1.0).copy()
        late  = late.sample(frac=1.0).copy()
        mid   = mid.sample(frac=1.0).copy()
        train.append(early.iloc[edge_n:, :].copy())
        test.append(early.iloc[:edge_n, :].copy())
        train.append(late.iloc[edge_n:, :].copy())
        test.append(late.iloc[:edge_n, :].copy())
        train.append(mid.iloc[mid_n:, :].copy())
        test.append(mid.iloc[:mid_n, :].copy())
    return pandas.concat(train), pandas.concat(test)


# Feature definitions (consistent with the original soy script: only Year, no Year_squared)
temporal_features          = ['Year']
bimonthly_climate_features = [
    'Precipitation_Early', 'Tmax_Early', 'Tmin_Early',
    'Precipitation_Mid',   'Tmax_Mid',   'Tmin_Mid',
    'Precipitation_Late',  'Tmax_Late',  'Tmin_Late',
]
bimonthly_co2_features = ['CO2_Early', 'CO2_Mid', 'CO2_Late']
output_cols            = ['Yield']
output_deviation_cols  = ['Yield_deviation']


# ============================================================================
# Per-seed data preparation
# ============================================================================
def prepare_source_df_for_seed(rs):
    """
    Per-seed source_df:
      - merge the seed=k corn trend (diff_smooth) → cams['Year'] = Tech_trend
      - filter to the seed=k reliable counties
      - normalize + one-hot encoding
    """
    # ── Load per-seed corn trend ─────────────────────────────
    corn_trend_path = f'{corn_trend_dir}/Corn_county_year_tech_diff_smooth_rs{rs}.p'
    if not os.path.exists(corn_trend_path):
        raise FileNotFoundError(f"Missing corn trend: {corn_trend_path}")
    corn_trend = pickle.load(open(corn_trend_path, 'rb'))
    corn_trend['FIPS'] = corn_trend['FIPS'].astype(str).str.zfill(5)
    trend_col = 'diff_smooth' if 'diff_smooth' in corn_trend.columns else 'diff'
    print(f"  [rs={rs}] corn trend loaded: {len(corn_trend)} rows, "
          f"{corn_trend['FIPS'].nunique()} counties (col: {trend_col})")

    # ── Merge trend into CAMS ────────────────────────────────
    cams = crop_df.copy()
    cams['Yield_deviation']  = cams['Yield'].copy()
    cams['Year_delta']       = cams['Year'] - start_year_base
    cams['Year_original']    = cams['Year'] - start_year_base + 1   # 1..45
    cams['Year_actual']      = cams['Year_delta'] + start_year_base
    cams['FIPS']             = cams['FIPS'].astype(str).str.zfill(5)

    cams = cams.merge(
        corn_trend[['FIPS', 'Year_actual', trend_col]].rename(columns={trend_col: 'Tech_trend'}),
        on=['FIPS', 'Year_actual'], how='left',
    )
    # Drop rows without a trend (county not in the corn reliable set, or NaN from edge-year smoothing)
    cams = cams.dropna(subset=['Tech_trend']).reset_index(drop=True)

    # Replace cams['Year'] with Tech_trend (consistent with the original soy script)
    cams['Year'] = cams['Tech_trend']

    # ── Load per-seed reliable counties ──────────────────────
    reliable_csv = f'{reliable_dir}/reliable_counties_NN_rs{rs}.csv'
    if not os.path.exists(reliable_csv):
        raise FileNotFoundError(f"Missing reliable counties: {reliable_csv}")
    reliable = pd.read_csv(reliable_csv, dtype={'FIPS': str})
    reliable['FIPS'] = reliable['FIPS'].str.zfill(5)
    cams = cams[cams['FIPS'].isin(reliable['FIPS'])].reset_index(drop=True)
    print(f"  [rs={rs}] after reliable filter: {len(cams)} rows, "
          f"{cams['FIPS'].nunique()} counties")

    # ── Feature engineering ──────────────────────────────────
    cams = create_features(cams)
    cams = cams[
        ["FIPS", "Year_delta", "Year_actual", "Year_original"]
        + temporal_features + bimonthly_climate_features + bimonthly_co2_features
        + output_cols + output_deviation_cols
    ]
    cams_ref = cams.groupby("FIPS", as_index=True).mean(numeric_only=True)

    cams, ref_dic = alter_input(
        cams,
        temporal_features + bimonthly_climate_features
        + bimonthly_co2_features + output_deviation_cols,
    )
    cams = encode_and_bind(cams, 'FIPS')

    # ── Feature lists ────────────────────────────────────────
    county_cols = [c for c in cams.columns if c.startswith('Spatial_')]

    linear_baseline    = bimonthly_climate_features + county_cols
    linear_time        = temporal_features + bimonthly_climate_features + county_cols
    linear_co2         = bimonthly_climate_features + bimonthly_co2_features + county_cols
    linear_time_co2    = temporal_features + bimonthly_climate_features + bimonthly_co2_features + county_cols

    feature_lists = {
        'linear_baseline':  linear_baseline,
        'linear_time':      linear_time,
        'linear_co2':       linear_co2,
        'linear_time_co2':  linear_time_co2,
        'nn_baseline':      linear_baseline.copy(),
        'nn_time':          linear_time.copy(),
        'nn_co2':           linear_co2.copy(),
        'nn_time_co2':      linear_time_co2.copy(),
    }

    return cams, ref_dic, feature_lists, cams_ref


# ============================================================================
# Global buffers (cleared per seed)
# ============================================================================
train_loss      = []
train_r2        = []
test_loss       = []
test_r2         = []
models_list     = []
run_train_list  = []
run_test_list   = []
scalers_list    = []


def save_per_seed_results(tag, rs, sensitivity, stage_early, stage_mid, stage_late,
                          is_nn=False, year_sens=None,
                          this_train_loss=None, this_train_r2=None,
                          this_test_loss=None, this_test_r2=None):
    base = f'{out_dir}/{crop}'
    pickle.dump(sensitivity, open(f'{base}_sensitivities_{tag}_rs{rs}.p',       'wb'))
    pickle.dump(stage_early, open(f'{base}_early_sensitivities_{tag}_rs{rs}.p', 'wb'))
    pickle.dump(stage_mid,   open(f'{base}_mid_sensitivities_{tag}_rs{rs}.p',   'wb'))
    pickle.dump(stage_late,  open(f'{base}_late_sensitivities_{tag}_rs{rs}.p',  'wb'))
    pickle.dump(models_list[-1],    open(f'{base}_model_{tag}_rs{rs}.p',     'wb'))
    pickle.dump(run_test_list[-1],  open(f'{base}_run_test_{tag}_rs{rs}.p',  'wb'))
    pickle.dump(run_train_list[-1], open(f'{base}_run_train_{tag}_rs{rs}.p', 'wb'))
    if is_nn:
        pickle.dump(year_sens,        open(f'{base}_year_sensitivities_{tag}_rs{rs}.p', 'wb'))
        pickle.dump(scalers_list[-1], open(f'{base}_scaler_{tag}_rs{rs}.p',             'wb'))
        pickle.dump(this_train_loss,  open(f'{base}_train_loss_{tag}_rs{rs}.p',         'wb'))
        pickle.dump(this_train_r2,    open(f'{base}_train_r2_{tag}_rs{rs}.p',           'wb'))
        pickle.dump(this_test_loss,   open(f'{base}_test_loss_{tag}_rs{rs}.p',          'wb'))
        pickle.dump(this_test_r2,     open(f'{base}_test_r2_{tag}_rs{rs}.p',            'wb'))
    print(f"    ✓ Saved {tag} rs={rs}")


def cleanup_after_seed(is_nn=False):
    models_list.clear()
    run_train_list.clear()
    run_test_list.clear()
    if is_nn:
        scalers_list.clear()
        tf.keras.backend.clear_session()
    gc.collect()


# ============================================================================
# Linear (Ridge)
# ============================================================================
def linear_model(df, input, output, ref, sensitivity_list):
    seed = random_seed
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

    crop_model_run_train, crop_model_run_test = train_test_split_county(df)
    train_X = crop_model_run_train[input]
    train_Y = np.array(crop_model_run_train[output].values.flatten())
    test_X  = crop_model_run_test[input]
    test_Y  = np.array(crop_model_run_test[output].values.flatten())

    model = Ridge(alpha=0.2)
    model.fit(train_X, train_Y)
    coef_map = dict(zip(input, np.asarray(model.coef_).ravel()))

    train_pred = model.predict(train_X).ravel()
    test_pred  = model.predict(test_X).ravel()
    print(f"    [Linear] train r2: {r2_score(train_Y, train_pred):.4f}  "
          f"test r2: {r2_score(test_Y, test_pred):.4f}")

    models_list.append(model)
    run_train_list.append(crop_model_run_train)
    run_test_list.append(crop_model_run_test)

    # CO2 sensitivity via analytic coefficient
    beta_co2_norm = {f: coef_map[f] for f in bimonthly_co2_features if f in coef_map}
    stage_sensitivity = {f: [] for f in bimonthly_co2_features}

    for split_df in [crop_model_run_test, crop_model_run_train]:
        base_pred_split = model.predict(split_df[input]).ravel()
        abs_effects = []
        stage_abs = {f: [] for f in bimonthly_co2_features}

        for _, row in split_df.iterrows():
            fips = str(row["FIPS"]).zfill(5)
            me_abs = 0.0; valid = True; stage_me = {}
            for f in bimonthly_co2_features:
                if f not in beta_co2_norm:
                    stage_me[f] = np.nan; continue
                mu = ref[0][fips][f]
                if (mu is None) or (mu == 0) or np.isnan(mu):
                    valid = False; stage_me[f] = np.nan
                else:
                    effect = beta_co2_norm[f] / mu
                    me_abs += effect; stage_me[f] = effect
            abs_effects.append(me_abs if valid else np.nan)
            for f in bimonthly_co2_features:
                stage_abs[f].append(stage_me.get(f, np.nan))

        # Overall: denominator = per-row baseline prediction
        S_run = []
        for abs_eff, base_y in zip(abs_effects, base_pred_split):
            if (base_y is None) or (base_y == 0) or np.isnan(base_y) or np.isnan(abs_eff):
                S_run.append(np.nan)
            else:
                S_run.append((abs_eff / base_y) * 100.0)
        sensitivity_list.extend(S_run)

        for f in bimonthly_co2_features:
            S_stage = []
            for abs_eff, base_y in zip(stage_abs[f], base_pred_split):
                if (base_y is None) or (base_y == 0) or np.isnan(base_y) or np.isnan(abs_eff):
                    S_stage.append(np.nan)
                else:
                    S_stage.append((abs_eff / base_y) * 100.0)
            stage_sensitivity[f].extend(S_stage)

    return stage_sensitivity


# ============================================================================
# NN
# ============================================================================
def nn_model(df, input, output, ref, sensitivity_list, year_sensitivity_list):
    print(f"    NN input features: {len(input)}")
    seed = random_seed
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

    crop_model_run_train, crop_model_run_test = train_test_split_county(df)
    train_sub, val_sub = train_test_split_county(crop_model_run_train)

    train_X = train_sub[input]; train_Y = train_sub[output]
    val_X   = val_sub[input];   val_Y   = val_sub[output]
    test_X  = crop_model_run_test[input]; test_Y = crop_model_run_test[output]

    scaler_X = StandardScaler(); scaler_Y = StandardScaler()
    train_X_scaled = scaler_X.fit_transform(train_X)
    val_X_scaled   = scaler_X.transform(val_X)
    test_X_scaled  = scaler_X.transform(test_X)
    train_Y_scaled = scaler_Y.fit_transform(train_Y.values.reshape(-1, 1)).ravel()
    val_Y_scaled   = scaler_Y.transform(val_Y.values.reshape(-1, 1)).ravel()

    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(len(input),)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mean_squared_error")

    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20,
                              min_lr=1e-7, verbose=1)
    early_stop = EarlyStopping(monitor="val_loss", patience=100,
                               restore_best_weights=True, verbose=1)

    model.fit(train_X_scaled, train_Y_scaled,
              epochs=2000, batch_size=64, verbose=1,
              validation_data=(val_X_scaled, val_Y_scaled),
              callbacks=[early_stop, rlrop])

    y_tr = np.array(train_Y.values.flatten())
    y_te = np.array(test_Y.values.flatten())
    yhat_tr = scaler_Y.inverse_transform(
        model.predict(train_X_scaled, verbose=0).reshape(-1, 1)).ravel()
    yhat_te = scaler_Y.inverse_transform(
        model.predict(test_X_scaled, verbose=0).reshape(-1, 1)).ravel()
    r2_tr  = r2_score(y_tr, yhat_tr); r2_te = r2_score(y_te, yhat_te)
    mae_tr = mean_absolute_error(y_tr, yhat_tr); mae_te = mean_absolute_error(y_te, yhat_te)
    print(f"    [NN] train r2={r2_tr:.4f}  test r2={r2_te:.4f}")

    models_list.append(model)
    run_train_list.append(crop_model_run_train)
    run_test_list.append(crop_model_run_test)
    train_loss.append(mae_tr); train_r2.append(r2_tr)
    test_loss.append(mae_te); test_r2.append(r2_te)
    scalers_list.append((scaler_X, scaler_Y))

    # CO2 sensitivity via numerical perturbation (+1 ppm)
    stage_sensitivity = {f: [] for f in bimonthly_co2_features}

    for split_df in [crop_model_run_test, crop_model_run_train]:
        split_X_scaled   = scaler_X.transform(split_df[input])
        base_pred_scaled = model.predict(split_X_scaled, verbose=0).ravel()
        base_pred        = scaler_Y.inverse_transform(base_pred_scaled.reshape(-1, 1)).ravel()

        stage_abs = {}
        for stage_feature in bimonthly_co2_features:
            if stage_feature not in input:
                stage_abs[stage_feature] = np.full(len(split_df), np.nan)
                continue

            split_pert = split_df.copy()
            for idx, row in split_pert.iterrows():
                fips_key = str(row["FIPS"]).zfill(5)
                mu = ref[0][fips_key][stage_feature]
                if mu == 0 or np.isnan(mu):
                    split_pert.loc[idx, stage_feature] = np.nan
                else:
                    split_pert.loc[idx, stage_feature] = row[stage_feature] + 1.0 / mu

            split_pert_X_scaled = scaler_X.transform(split_pert[input])
            pert_pred_scaled    = model.predict(split_pert_X_scaled, verbose=0).ravel()
            pert_pred           = scaler_Y.inverse_transform(pert_pred_scaled.reshape(-1, 1)).ravel()
            stage_abs[stage_feature] = pert_pred - base_pred

        combined_abs = np.zeros(len(split_df))
        for stage_feature in bimonthly_co2_features:
            if stage_feature in input:
                combined_abs += np.nan_to_num(stage_abs[stage_feature], nan=0.0)

        # Overall: denominator = per-row baseline prediction
        S_run = []
        for abs_eff, base_y in zip(combined_abs, base_pred):
            if (base_y is None) or (base_y == 0) or np.isnan(base_y) or np.isnan(abs_eff):
                S_run.append(np.nan)
            else:
                S_run.append((abs_eff / base_y) * 100.0)
        sensitivity_list.extend(S_run)

        for stage_feature in bimonthly_co2_features:
            S_stage = []
            for abs_eff, base_y in zip(stage_abs[stage_feature], base_pred):
                if (base_y is None) or (base_y == 0) or np.isnan(base_y) or np.isnan(abs_eff):
                    S_stage.append(np.nan)
                else:
                    S_stage.append((abs_eff / base_y) * 100.0)
            stage_sensitivity[stage_feature].extend(S_stage)

    return stage_sensitivity


# ============================================================================
# Block runner
# ============================================================================
def run_one_block(tag, run_input, source_df, ref_dic, is_nn, rs):
    print(f"\n  >>> {tag}")

    sensitivity = []
    year_sens   = []
    stage_early = []; stage_mid = []; stage_late = []

    if is_nn:
        _tl, _tr, _el, _er = len(train_loss), len(train_r2), len(test_loss), len(test_r2)
        stage_s = nn_model(source_df.copy(), run_input, output_cols, ref_dic,
                           sensitivity, year_sens)
    else:
        stage_s = linear_model(source_df.copy(), run_input, output_cols, ref_dic,
                               sensitivity)

    if stage_s is not None:
        stage_early.extend(stage_s['CO2_Early'])
        stage_mid.extend(stage_s['CO2_Mid'])
        stage_late.extend(stage_s['CO2_Late'])

    snp = np.asarray(sensitivity, dtype=float)
    print(f"    [rs={rs}] median sensitivity = {np.nanmedian(snp):.6f}")

    if is_nn:
        this_train_loss = train_loss[_tl:]
        this_train_r2   = train_r2[_tr:]
        this_test_loss  = test_loss[_el:]
        this_test_r2    = test_r2[_er:]
        save_per_seed_results(
            tag, rs, sensitivity, stage_early, stage_mid, stage_late,
            is_nn=True, year_sens=year_sens,
            this_train_loss=this_train_loss, this_train_r2=this_train_r2,
            this_test_loss=this_test_loss, this_test_r2=this_test_r2,
        )
    else:
        save_per_seed_results(
            tag, rs, sensitivity, stage_early, stage_mid, stage_late,
            is_nn=False,
        )
    cleanup_after_seed(is_nn=is_nn)


# ============================================================================
# Main loop: per-seed
# ============================================================================
BLOCKS = [
    ('linear_baseline',  False),
    ('linear_time',      False),
    ('linear_co2',       False),
    ('linear_time_co2',  False),
    ('nn_baseline',      True),
    ('nn_time',          True),
    ('nn_co2',           True),
    ('nn_time_co2',      True),
]

for rs in random_seed_list:
    random_seed = rs
    print("\n" + "=" * 70)
    print(f"SEED = {rs}")
    print("=" * 70)

    source_df, ref_dic, feature_lists, cams_ref = prepare_source_df_for_seed(rs)

    # On the first seed, save the feature list + source_ref once (same for all seeds)
    if rs == random_seed_list[0]:
        for tag, feats in feature_lists.items():
            pickle.dump(feats, open(f'{out_dir}/{crop}_features_{tag}.p', 'wb'))
        pickle.dump(cams_ref, open(f'{out_dir}/{crop}_source_ref.p', 'wb'))

    for tag, is_nn in BLOCKS:
        run_input = feature_lists[tag]
        run_one_block(tag, run_input, source_df, ref_dic, is_nn, rs)

    del source_df, ref_dic, feature_lists, cams_ref
    gc.collect()


# ============================================================================
# Finally: assemble the predictions files (to make later residual/sensitivity analysis easier)
# ============================================================================
print("\n" + "=" * 70)
print("Assembling predictions files from all per-seed models")
print("=" * 70)

for tag, is_nn in BLOCKS:
    preds_list = []
    feats = pickle.load(open(f'{out_dir}/{crop}_features_{tag}.p', 'rb'))

    for rs in random_seed_list:
        model    = pickle.load(open(f'{out_dir}/{crop}_model_{tag}_rs{rs}.p',     'rb'))
        test_df  = pickle.load(open(f'{out_dir}/{crop}_run_test_{tag}_rs{rs}.p',  'rb'))
        train_df = pickle.load(open(f'{out_dir}/{crop}_run_train_{tag}_rs{rs}.p', 'rb'))

        if is_nn:
            scaler_X, scaler_Y = pickle.load(
                open(f'{out_dir}/{crop}_scaler_{tag}_rs{rs}.p', 'rb'))
            tp = scaler_Y.inverse_transform(
                model.predict(scaler_X.transform(test_df[feats]),  verbose=0)).ravel()
            rp = scaler_Y.inverse_transform(
                model.predict(scaler_X.transform(train_df[feats]), verbose=0)).ravel()
            del scaler_X, scaler_Y
        else:
            tp = model.predict(test_df[feats])
            rp = model.predict(train_df[feats])

        preds_list.append({
            'rs':         rs,
            'test_pred':  np.asarray(tp).ravel(),
            'train_pred': np.asarray(rp).ravel(),
            'test_meta':  test_df[['FIPS', 'Year_actual', 'Yield']].reset_index(drop=True),
            'train_meta': train_df[['FIPS', 'Year_actual', 'Yield']].reset_index(drop=True),
        })

        del model, test_df, train_df
        if is_nn:
            tf.keras.backend.clear_session()
        gc.collect()

    out_pkl = f'{out_dir}/{crop}_predictions_{tag}.p'
    pickle.dump(preds_list, open(out_pkl, 'wb'))
    print(f"  ✓ {tag}: {len(preds_list)} seeds → {out_pkl}")

print("\n✓ All done.")
print(f"  Per-seed files in: {out_dir}/")
print(f"  Predictions: {crop}_predictions_{{tag}}.p (same dict format as corn)")
