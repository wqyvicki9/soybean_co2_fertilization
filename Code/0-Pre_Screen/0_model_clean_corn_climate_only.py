# -*- coding: utf-8 -*-
"""0_Model_Clean_Corn_ClimateOnly_20seeds.py

Runs only Corn's climate-only model (no time feature and no CO2 feature),
used later for growth-rate correlation against soy's same-design residuals to pick reliable counties.

- 20 random seeds (0–19)
- No Year/CO2 feature; uses only climate + county fixed effects
- Two model types: Linear Ridge + NN
- Each seed saves files independently and frees memory immediately after running
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
results_folder_name = "Results_Corn_ClimateOnly_20seeds"

crop = 'Corn'
start_year_base = 1979
end_year        = 2023
year_datapoint_threshold = int((end_year - start_year_base) * 0.8)

out_dir = os.path.join(folder, 'Results', results_folder_name)
os.makedirs(out_dir, exist_ok=True)


# ============================================================================
# Data preprocessing —— keep only the basics; no corn trend merge, no reliable county filtering
# ============================================================================
crop_df = pandas.read_csv(
    os.path.join(folder,'Data', 'Aggregated_yield_climate_co2', crop + '_CAMS_1979_2023_allcounties.csv'))
crop_df['FIPS'] = crop_df.FIPS.apply(lambda x: f"{int(x):05}")
crop_df = crop_df[(crop_df.Year <= end_year) & (crop_df.Year >= start_year_base)]
fips_count = crop_df.FIPS.value_counts()
fips = crop_df.FIPS.unique()
fips = [i for i in fips if fips_count[i] > year_datapoint_threshold]
crop_df = crop_df[crop_df.FIPS.isin(fips)].reset_index(drop=True)

print(f"Number of counties: {crop_df['FIPS'].nunique()}")
print(f"Total rows: {len(crop_df)}")

cams = crop_df.copy()
cams['Yield_deviation'] = cams['Yield'].copy()
cams['Year_delta']      = cams['Year'].copy()
cams['Year_delta']     -= start_year_base
cams['Year']            = cams['Year'] - start_year_base + 1
cams['Year_squared']    = cams['Year'] ** 2
cams['Year_actual']     = cams['Year_delta'] + start_year_base   # actual year 1979..2023

# No reliable county filtering and no Tech_trend merge
full_df = cams.copy()


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
    return df


def alter_input(df, input_cols):
    """Relative-anomaly normalization: x_norm = x / global_mean(x)."""
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
    """Add only spatial dummies (no temporal)."""
    dummies_spatial = pandas.get_dummies(df[[encode_col]], prefix='Spatial', dtype=int)
    return pandas.concat([df, dummies_spatial], axis=1)


def train_test_split_county(df):
    train, test = [], []
    county_model_data = df.groupby(['FIPS'])
    county_model_data = dict(list(county_model_data))
    counties = df.FIPS.unique()
    for county in counties:
        county_df = county_model_data[(county,)].sample(frac=1.0).copy()
        county_df = county_df.sort_values('Year').copy()
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


# ============================================================================
# Feature list construction
# ============================================================================
bimonthly_climate_features = [
    'Precipitation_Early', 'Tmax_Early', 'Tmin_Early',
    'Precipitation_Mid',   'Tmax_Mid',   'Tmin_Mid',
    'Precipitation_Late',  'Tmax_Late',  'Tmin_Late',
]
output_cols           = ['Yield']
output_deviation_cols = ['Yield_deviation']

cams_bck = full_df.copy()
cams_bck['FIPS'] = cams_bck['FIPS'].apply(lambda x: f"{int(x):05}")
cams_bck = create_features(cams_bck)

# Keep only the climate-only-related columns + FIPS + Year info + yield
# Note: the Year column must be kept —— train_test_split_county uses Year ordering for the stratified split
cams_bck = cams_bck[
    ["FIPS", "Year", "Year_delta", "Year_actual"] + bimonthly_climate_features + output_cols + output_deviation_cols
]
cams_ref = cams_bck.groupby("FIPS", as_index=True).mean()

cams_bck, ref_dic = alter_input(cams_bck, bimonthly_climate_features + output_deviation_cols)
cams_bck = encode_and_bind(cams_bck, 'FIPS')

df_cams = cams_bck.copy()
cams_county_cols = [c for c in df_cams.columns if c.startswith('Spatial_')]

# Climate-only feature: climate + county fixed effects only
climate_only_cols = bimonthly_climate_features + cams_county_cols

linear_baseline = climate_only_cols.copy()
nn_baseline     = climate_only_cols.copy()

source_df  = df_cams.copy()
source_ref = cams_ref.copy()

# Save the feature list and reference (stored only once for the whole pipeline)
pickle.dump(source_ref,      open(f'{out_dir}/{crop}_source_ref.p',         'wb'))
pickle.dump(linear_baseline, open(f'{out_dir}/{crop}_climate_only_features.p', 'wb'))

print(f"\nFeature counts:")
print(f"  bimonthly climate features: {len(bimonthly_climate_features)}")
print(f"  county fixed-effect dummies: {len(cams_county_cols)}")
print(f"  total features: {len(climate_only_cols)}")


# ============================================================================
# Global lists —— used as temporary buffers for linear_model / nn_model
# Cleared immediately after each seed finishes
# ============================================================================
train_loss      = []
train_r2        = []
test_loss       = []
test_r2         = []
models_list     = []
run_train_list  = []
run_test_list   = []
scalers_list    = []

num_runs = len(random_seed_list)


# ============================================================================
# Per-seed saving + memory cleanup
# ============================================================================
def save_per_seed_results(tag, rs, is_nn=False,
                          this_train_loss=None, this_train_r2=None,
                          this_test_loss=None, this_test_r2=None):
    base = f'{out_dir}/{crop}'

    # Model + train/test split
    pickle.dump(models_list[-1],    open(f'{base}_model_{tag}_rs{rs}.p',     'wb'))
    pickle.dump(run_test_list[-1],  open(f'{base}_run_test_{tag}_rs{rs}.p',  'wb'))
    pickle.dump(run_train_list[-1], open(f'{base}_run_train_{tag}_rs{rs}.p', 'wb'))

    if is_nn:
        pickle.dump(scalers_list[-1], open(f'{base}_scaler_{tag}_rs{rs}.p',     'wb'))
        pickle.dump(this_train_loss,  open(f'{base}_train_loss_{tag}_rs{rs}.p', 'wb'))
        pickle.dump(this_train_r2,    open(f'{base}_train_r2_{tag}_rs{rs}.p',   'wb'))
        pickle.dump(this_test_loss,   open(f'{base}_test_loss_{tag}_rs{rs}.p',  'wb'))
        pickle.dump(this_test_r2,     open(f'{base}_test_r2_{tag}_rs{rs}.p',    'wb'))

    print(f"  ✓ Saved per-seed files: {tag} rs={rs}")


def cleanup_after_seed(is_nn=False):
    """Free memory immediately after a seed finishes."""
    models_list.clear()
    run_train_list.clear()
    run_test_list.clear()
    if is_nn:
        scalers_list.clear()
        tf.keras.backend.clear_session()
    gc.collect()


# ============================================================================
# Linear Ridge
# ============================================================================
def linear_model(df, input, output, ref):
    seed = random_seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    df_copy = df.copy()
    crop_model_run_train, crop_model_run_test = train_test_split_county(df_copy)

    train_X = crop_model_run_train[input]
    train_Y = np.array(crop_model_run_train[output].values.flatten())
    test_X  = crop_model_run_test[input]
    test_Y  = np.array(crop_model_run_test[output].values.flatten())

    model = Ridge(alpha=0.2)
    model.fit(train_X, train_Y)

    train_pred = model.predict(train_X).ravel()
    test_pred  = model.predict(test_X).ravel()
    print(f"  [Linear] train r2: {r2_score(train_Y, train_pred):.4f}  "
          f"test r2: {r2_score(test_Y, test_pred):.4f}")

    models_list.append(model)
    run_train_list.append(crop_model_run_train)
    run_test_list.append(crop_model_run_test)


# ============================================================================
# NN
# ============================================================================
def nn_model(df, input, output):
    seed = random_seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    df_copy = df.copy()
    crop_model_run_train, crop_model_run_test = train_test_split_county(df_copy)
    train_sub, val_sub = train_test_split_county(crop_model_run_train)

    train_X = train_sub[input]; train_Y = train_sub[output]
    val_X   = val_sub[input];   val_Y   = val_sub[output]
    test_X  = crop_model_run_test[input]; test_Y = crop_model_run_test[output]

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
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
    model.compile(
        optimizer=Adam(learning_rate=5e-4, clipnorm=1.0),  # Corn uses 5e-4 + clipnorm (consistent with the original corn script)
        loss="mean_squared_error",
    )

    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20,
                              min_lr=1e-7, verbose=1)
    early_stop = EarlyStopping(monitor="val_loss", patience=100,
                               restore_best_weights=True, verbose=1)

    model.fit(
        train_X_scaled, train_Y_scaled,
        epochs=2000, batch_size=64, verbose=1,
        validation_data=(val_X_scaled, val_Y_scaled),
        callbacks=[early_stop, rlrop],
    )

    y_tr = np.array(train_Y.values.flatten())
    y_te = np.array(test_Y.values.flatten())

    yhat_tr = scaler_Y.inverse_transform(
        model.predict(train_X_scaled, verbose=0).reshape(-1, 1)).ravel()
    yhat_te = scaler_Y.inverse_transform(
        model.predict(test_X_scaled, verbose=0).reshape(-1, 1)).ravel()

    r2_tr  = r2_score(y_tr, yhat_tr)
    r2_te  = r2_score(y_te, yhat_te)
    mae_tr = mean_absolute_error(y_tr, yhat_tr)
    mae_te = mean_absolute_error(y_te, yhat_te)
    print(f"  [NN] train r2={r2_tr:.4f}  test r2={r2_te:.4f}")

    models_list.append(model)
    run_train_list.append(crop_model_run_train)
    run_test_list.append(crop_model_run_test)
    train_loss.append(mae_tr)
    train_r2.append(r2_tr)
    test_loss.append(mae_te)
    test_r2.append(r2_te)
    scalers_list.append((scaler_X, scaler_Y))


# ============================================================================
# Block runner
# ============================================================================
def run_block_linear_climate_only():
    print("=" * 60)
    print("Linear climate-only")
    print("=" * 60)
    global random_seed

    tag = "linear_baseline"
    run_df     = source_df.copy()
    run_input  = linear_baseline
    run_output = output_cols

    for i in range(num_runs):
        random_seed = random_seed_list[i]
        print(f"\n--- Run {i+1}/{num_runs}  (seed={random_seed}) ---")

        linear_model(run_df, run_input, run_output, ref_dic)

        save_per_seed_results(tag, random_seed, is_nn=False)
        cleanup_after_seed(is_nn=False)


def run_block_nn_climate_only():
    print("=" * 60)
    print("NN climate-only")
    print("=" * 60)
    global random_seed

    tag = "nn_baseline"
    run_df     = source_df.copy()
    run_input  = nn_baseline
    run_output = output_cols

    for i in range(num_runs):
        random_seed = random_seed_list[i]
        print(f"\n--- Run {i+1}/{num_runs}  (seed={random_seed}) ---")

        _tl = len(train_loss)
        _tr = len(train_r2)
        _el = len(test_loss)
        _er = len(test_r2)

        nn_model(run_df, run_input, run_output)

        this_train_loss = train_loss[_tl:]
        this_train_r2   = train_r2[_tr:]
        this_test_loss  = test_loss[_el:]
        this_test_r2    = test_r2[_er:]

        save_per_seed_results(
            tag, random_seed, is_nn=True,
            this_train_loss=this_train_loss,
            this_train_r2=this_train_r2,
            this_test_loss=this_test_loss,
            this_test_r2=this_test_r2,
        )
        cleanup_after_seed(is_nn=True)


# ============================================================================
# Run the two blocks
# ============================================================================
run_block_linear_climate_only()
run_block_nn_climate_only()


# ============================================================================
# Predictions: load each model on demand and compute predicted yield
# ============================================================================
print("\n" + "=" * 60)
print("Computing predictions from all saved per-seed models")
print("=" * 60)

predictions_linear = []   # list of (test_pred, train_pred) per seed
predictions_nn     = []

# ── Linear ──────────────────────────────────────────────────
for rs in random_seed_list:
    tag = "linear_baseline"
    model    = pickle.load(open(f'{out_dir}/{crop}_model_{tag}_rs{rs}.p',     'rb'))
    test_df  = pickle.load(open(f'{out_dir}/{crop}_run_test_{tag}_rs{rs}.p',  'rb'))
    train_df = pickle.load(open(f'{out_dir}/{crop}_run_train_{tag}_rs{rs}.p', 'rb'))

    tp = model.predict(test_df[linear_baseline])
    rp = model.predict(train_df[linear_baseline])

    predictions_linear.append({
        'rs':         rs,
        'test_pred':  tp,
        'train_pred': rp,
        # Keep FIPS + Year_actual + actual Yield to make computing residuals and merging easier later
        'test_meta':  test_df[['FIPS', 'Year_actual', 'Yield']].reset_index(drop=True),
        'train_meta': train_df[['FIPS', 'Year_actual', 'Yield']].reset_index(drop=True),
    })

    del model, test_df, train_df
    gc.collect()

print(f"  Linear: {len(predictions_linear)} seeds done")

# ── NN ──────────────────────────────────────────────────────
for rs in random_seed_list:
    tag = "nn_baseline"
    model    = pickle.load(open(f'{out_dir}/{crop}_model_{tag}_rs{rs}.p',     'rb'))
    test_df  = pickle.load(open(f'{out_dir}/{crop}_run_test_{tag}_rs{rs}.p',  'rb'))
    train_df = pickle.load(open(f'{out_dir}/{crop}_run_train_{tag}_rs{rs}.p', 'rb'))
    scaler_X, scaler_Y = pickle.load(
        open(f'{out_dir}/{crop}_scaler_{tag}_rs{rs}.p', 'rb'))

    test_X_scaled  = scaler_X.transform(test_df[nn_baseline])
    train_X_scaled = scaler_X.transform(train_df[nn_baseline])

    tp = scaler_Y.inverse_transform(model.predict(test_X_scaled,  verbose=0)).ravel()
    rp = scaler_Y.inverse_transform(model.predict(train_X_scaled, verbose=0)).ravel()

    predictions_nn.append({
        'rs':         rs,
        'test_pred':  tp,
        'train_pred': rp,
        'test_meta':  test_df[['FIPS', 'Year_actual', 'Yield']].reset_index(drop=True),
        'train_meta': train_df[['FIPS', 'Year_actual', 'Yield']].reset_index(drop=True),
    })

    del model, test_df, train_df, scaler_X, scaler_Y, test_X_scaled, train_X_scaled
    tf.keras.backend.clear_session()
    gc.collect()

print(f"  NN: {len(predictions_nn)} seeds done")

pickle.dump(predictions_linear,
            open(f'{out_dir}/{crop}_predictions_linear.p', 'wb'))
pickle.dump(predictions_nn,
            open(f'{out_dir}/{crop}_predictions_nn.p', 'wb'))

print("\n✓ All per-seed files saved. Predictions saved as list of dicts.")
print("  Each dict has: rs, test_pred, train_pred, test_meta, train_meta")
print(f"  Meta includes FIPS, Year_actual, Yield → use this to compute residuals.")
