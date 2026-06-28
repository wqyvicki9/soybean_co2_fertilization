# -*- coding: utf-8 -*-
"""0_Model_Clean_Corn_20random_seeds_PER_SEED_MEMFIXED.ipynb

Each random seed's results are saved to separate files + memory is freed immediately after running.
Suitable for SLURM jobs; peak memory of ~2-4 GB is enough (`--mem=16G` is sufficient).
"""

import warnings
warnings.filterwarnings('ignore')
import gc
import os
import math
import random
import pickle
import pandas
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

random_seed_list = list(range(20))
folder = '.'   # repo root (contains Data/ and Results/); run from here
results_folder_name = "Results_random_seeds_20"

"""Pre-Processing"""

crop = 'Corn'
start_year_base = 1979
end_year = 2023
year_datapoint_threshold = int((end_year - start_year_base) * 0.8)

crop_df = pandas.read_csv(os.path.join(folder, 'Data', 'Aggregated_yield_climate_co2', crop + '_CAMS_1979_2023_allcounties.csv'))
crop_df['FIPS'] = crop_df.FIPS.apply(lambda x: f"{int(x):05}")
crop_df = crop_df[(crop_df.Year <= end_year) & (crop_df.Year >= start_year_base)]
fips_count = crop_df.FIPS.value_counts()
fips = crop_df.FIPS.unique()
fips = [i for i in fips if fips_count[i] > year_datapoint_threshold]
crop_df = crop_df[crop_df.FIPS.isin(fips)].reset_index(drop=True)

cams = crop_df.copy()
cams['Yield_deviation'] = cams['Yield'].copy()
cams['Year_delta'] = cams['Year'].copy()
cams['Year_delta'] -= start_year_base
cams['Year'] = cams['Year'] - start_year_base + 1
cams['Year_squared'] = cams['Year'] ** 2


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

    df['Tmean_Early'] = df[['tmean_May', 'tmean_Jun']].mean(axis=1)
    df['Tmean_Mid']   = df[['tmean_Jul', 'tmean_Aug']].mean(axis=1)
    df['Tmean_Late']  = df[['tmean_Sep', 'tmean_Oct']].mean(axis=1)

    df['CO2_Early'] = df[['co2_May', 'co2_Jun']].mean(axis=1)
    df['CO2_Mid']   = df[['co2_Jul', 'co2_Aug']].mean(axis=1)
    df['CO2_Late']  = df[['co2_Sep', 'co2_Oct']].mean(axis=1)

    df['CO2']           = df[['CO2_Early', 'CO2_Mid', 'CO2_Late']].mean(axis=1)
    df['Precipitation'] = df[['Precipitation_Early', 'Precipitation_Mid', 'Precipitation_Late']].mean(axis=1)
    df['Tmean']         = df[['Tmean_Early', 'Tmean_Mid', 'Tmean_Late']].mean(axis=1)
    df['Tmax']          = df[['Tmax_Early', 'Tmax_Mid', 'Tmax_Late']].mean(axis=1)
    df['Tmin']          = df[['Tmin_Early', 'Tmin_Mid', 'Tmin_Late']].mean(axis=1)

    return df


def alter_input(df, input_cols):
    df_norm = df.copy()
    feature_mean = {}
    for c in input_cols:
        mu = df[c].mean()
        feature_mean[c] = mu
        if mu == 0:
            df_norm[c] = 0.0
        else:
            df_norm[c] = df[c] / mu

    county_feature_trends = [{}, {}]
    counties = df.FIPS.unique()
    for county in counties:
        county_feature_trends[0][county] = {}
        county_feature_trends[1][county] = {}
        for c in input_cols:
            county_feature_trends[0][county][c] = feature_mean[c]
            county_feature_trends[1][county][c] = feature_mean[c]
    return df_norm, county_feature_trends


def encode_and_bind(df, encode_col, spatial=1, temporal=1):
    dummies_spatial  = pandas.get_dummies(df[[encode_col]], prefix='Spatial',  dtype=int)
    dummies_temporal = pandas.get_dummies(df[[encode_col]], prefix='Temporal', dtype=int)
    dummies_temporal.values[dummies_temporal != 0] = df['Year']
    df_list = [df]
    if spatial:
        df_list.append(dummies_spatial)
    if temporal:
        df_list.append(dummies_temporal)
    return pandas.concat(df_list, axis=1)


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


temporal_features           = ['Year', 'Year_squared']
bimonthly_climate_features  = ['Precipitation_Early', 'Tmax_Early', 'Tmin_Early',
                               'Precipitation_Mid',   'Tmax_Mid',   'Tmin_Mid',
                               'Precipitation_Late',  'Tmax_Late',  'Tmin_Late']
bimonthly_co2_features      = ['CO2_Early', 'CO2_Mid', 'CO2_Late']
output_cols                 = ['Yield']
output_deviation_cols       = ['Yield_deviation']

cams_bck = cams.copy()
cams_bck['FIPS'] = cams_bck['FIPS'].apply(lambda x: f"{int(x):05}")
cams_bck = create_features(cams_bck)

cams_bck = cams_bck[["FIPS", "Year_delta"] + temporal_features + bimonthly_climate_features
                    + bimonthly_co2_features + output_cols + output_deviation_cols]
cams_ref = cams_bck.groupby("FIPS", as_index=True).mean()

cams_bck, ref_dic = alter_input(cams_bck,
                                temporal_features + bimonthly_climate_features
                                + bimonthly_co2_features + output_deviation_cols)
cams_bck = encode_and_bind(cams_bck, 'FIPS', spatial=1, temporal=0)

df_cams = cams_bck.copy()

cams_county_cols                = [c for c in df_cams.columns if c.startswith('Spatial_')]
linear_baseline_cols            = bimonthly_climate_features + cams_county_cols
linear_baseline_time_cols       = temporal_features + bimonthly_climate_features + cams_county_cols
linear_baseline_co2_cols        = bimonthly_climate_features + bimonthly_co2_features + cams_county_cols
linear_baseline_time_co2_cols   = temporal_features + bimonthly_climate_features + bimonthly_co2_features + cams_county_cols

nn_baseline_cols            = bimonthly_climate_features + cams_county_cols
nn_baseline_time_cols       = temporal_features + bimonthly_climate_features + cams_county_cols
nn_baseline_co2_cols        = bimonthly_climate_features + bimonthly_co2_features + cams_county_cols
nn_baseline_time_co2_cols   = temporal_features + bimonthly_climate_features + bimonthly_co2_features + cams_county_cols

source_df = df_cams.copy()

nn_baseline   = nn_baseline_cols.copy()
nn_time       = nn_baseline_time_cols.copy()
nn_co2        = nn_baseline_co2_cols.copy()
nn_time_co2   = nn_baseline_time_co2_cols.copy()

linear_baseline   = linear_baseline_cols.copy()
linear_time       = linear_baseline_time_cols.copy()
linear_co2        = linear_baseline_co2_cols.copy()
linear_time_co2   = linear_baseline_time_co2_cols.copy()

source_ref = cams_ref.copy()

out_dir = os.path.join(folder, 'Results', results_folder_name)
os.makedirs(out_dir, exist_ok=True)

# These feature lists are saved only once (not per seed)
pickle.dump(source_ref,       open(f'{out_dir}/{crop}_source_ref.p',            'wb'))
pickle.dump(nn_baseline,      open(f'{out_dir}/{crop}_baseline.p',              'wb'))
pickle.dump(nn_time,          open(f'{out_dir}/{crop}_basetime.p',              'wb'))
pickle.dump(nn_co2,           open(f'{out_dir}/{crop}_input.p',                 'wb'))
pickle.dump(nn_time_co2,      open(f'{out_dir}/{crop}_time.p',                  'wb'))
pickle.dump(linear_baseline,  open(f'{out_dir}/{crop}_baseline_linear.p',       'wb'))
pickle.dump(linear_time,      open(f'{out_dir}/{crop}_basetime_linear.p',       'wb'))
pickle.dump(linear_co2,       open(f'{out_dir}/{crop}_input_linear.p',          'wb'))
pickle.dump(linear_time_co2,  open(f'{out_dir}/{crop}_time_linear.p',           'wb'))

"""Modelling"""

# Global lists —— cleared after each seed via clear(); used only as a "temporary buffer"
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
# Helper: after each seed finishes, save all of the current seed's results to separate files
# ============================================================================
def save_per_seed_results(tag, rs, sensitivity, stage_early, stage_mid, stage_late,
                          is_nn=False, year_sens=None,
                          this_train_loss=None, this_train_r2=None,
                          this_test_loss=None, this_test_r2=None):
    """
    Save all of the current seed's results to separate pickle files.

    File name format: {crop}_<what>_{tag}_rs{rs}.p
    E.g.: Corn_sensitivities_nn_time_co2_rs3.p
    """
    base = f'{out_dir}/{crop}'

    # Sensitivity-related
    pickle.dump(sensitivity, open(f'{base}_sensitivities_{tag}_rs{rs}.p',       'wb'))
    pickle.dump(stage_early, open(f'{base}_early_sensitivities_{tag}_rs{rs}.p', 'wb'))
    pickle.dump(stage_mid,   open(f'{base}_mid_sensitivities_{tag}_rs{rs}.p',   'wb'))
    pickle.dump(stage_late,  open(f'{base}_late_sensitivities_{tag}_rs{rs}.p',  'wb'))

    # Model + train/test data (linear_model / nn_model appends one element to the global list each time)
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

    print(f"  ✓ Saved per-seed files: {tag} rs={rs}")


def cleanup_after_seed(is_nn=False):
    """After finishing a seed, immediately clear the global lists and TF session to prevent memory buildup."""
    models_list.clear()
    run_train_list.clear()
    run_test_list.clear()
    if is_nn:
        scalers_list.clear()
        # Clear the internal Keras graph / session, otherwise TF accumulates one per trained model
        tf.keras.backend.clear_session()
    # Prompt the Python GC to reclaim the just-cleared objects
    gc.collect()


# ============================================================================
# Linear model
# ============================================================================
def linear_model(
    df, input, output, ref,
    sensitivity_list, beta_time_list,
    beta_co2_abs_list, beta_co2_rel_list,
    beta_year_abs_list, beta_year_rel_list,
    county_time_trend=0, deltaC=1.0, year_scale=1.0,
):
    seed = random_seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    df_copy = df.copy()
    if county_time_trend:
        for sc in cams_county_cols:
            df_copy[sc] = df_copy[sc] * df_copy["Year"]

    tmp = df_copy[["FIPS", "Yield"]].copy()
    tmp["FIPS_str"] = tmp["FIPS"].astype(str).str.zfill(5)
    real_mean_yield = tmp.groupby("FIPS_str")["Yield"].mean().to_dict()

    crop_model_run_train, crop_model_run_test = train_test_split_county(df_copy)
    train_X = crop_model_run_train[input]
    train_Y = np.array(crop_model_run_train[output].values.flatten())
    test_X  = crop_model_run_test[input]
    test_Y  = np.array(crop_model_run_test[output].values.flatten())

    model = Ridge(alpha=0.2)
    model.fit(train_X, train_Y)
    coef = np.asarray(model.coef_).ravel()
    coef_map = dict(zip(input, coef))

    beta_year_raw = float(coef_map.get("Year", np.nan))
    beta_time_list.append(beta_year_raw)
    beta_year_abs = beta_year_raw / float(year_scale) if (year_scale and year_scale != 0) else np.nan
    beta_year_abs_list.append(beta_year_abs)

    train_pred_fit = model.predict(train_X).ravel()
    test_pred_fit  = model.predict(test_X).ravel()
    train_r2_val   = r2_score(train_Y, train_pred_fit)
    test_r2_val    = r2_score(test_Y,  test_pred_fit)
    print(f"train r2: {train_r2_val:.4f}  test r2: {test_r2_val:.4f}")

    models_list.append(model)
    run_train_list.append(crop_model_run_train)
    run_test_list.append(crop_model_run_test)

    beta_co2_norm = {f: coef_map[f] for f in bimonthly_co2_features if f in coef_map}
    stage_sensitivity = {f: [] for f in bimonthly_co2_features}

    for split_name, split_df in [('test', crop_model_run_test), ('train', crop_model_run_train)]:
        # Baseline prediction for each row —— denominator
        base_pred_split = model.predict(split_df[input]).ravel()

        abs_effects = []
        stage_abs_effects = {f: [] for f in bimonthly_co2_features}

        for _, row in split_df.iterrows():
            fips = str(row["FIPS"]).zfill(5)
            me_abs = 0.0
            valid = True
            stage_me = {}
            for f in bimonthly_co2_features:
                if f not in beta_co2_norm:
                    stage_me[f] = np.nan
                    continue
                mu = ref[0][fips][f]
                if (mu is None) or (mu == 0) or np.isnan(mu):
                    valid = False
                    stage_me[f] = np.nan
                else:
                    effect = beta_co2_norm[f] / mu
                    me_abs += effect
                    stage_me[f] = effect
            abs_effects.append(me_abs if valid else np.nan)
            for f in bimonthly_co2_features:
                stage_abs_effects[f].append(stage_me.get(f, np.nan))

        # Overall relative sensitivity —— denominator uses the current row's baseline prediction
        S_run = []
        for abs_eff, base_y in zip(abs_effects, base_pred_split):
            if (base_y is None) or (base_y == 0) or np.isnan(base_y) or np.isnan(abs_eff):
                S_run.append(np.nan)
            else:
                S_run.append((abs_eff / base_y) * 100.0)
        sensitivity_list.extend(S_run)

        # Per-stage relative sensitivity —— likewise uses the baseline prediction
        for f in bimonthly_co2_features:
            S_stage = []
            for abs_eff, base_y in zip(stage_abs_effects[f], base_pred_split):
                if (base_y is None) or (base_y == 0) or np.isnan(base_y) or np.isnan(abs_eff):
                    S_stage.append(np.nan)
                else:
                    S_stage.append((abs_eff / base_y) * 100.0)
            stage_sensitivity[f].extend(S_stage)

        if split_name == 'test':
            beta_co2_abs_list.append(np.nanmedian(abs_effects))
            beta_co2_rel_list.append(np.nanmedian(S_run))

    return stage_sensitivity


# ============================================================================
# NN model
# ============================================================================
def nn_model(df, input, output, ref, sensitivity_list, year_sensitivity_list, county_time_trend=0):
    print("Number of input features:", len(input))

    seed = random_seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    df_copy = df.copy()
    if county_time_trend:
        for sc in cams_county_cols:
            df_copy[sc] = df_copy[sc] * df_copy["Year"]

    tmp = df_copy[["FIPS", "Yield"]].copy()
    tmp["FIPS_str"] = tmp["FIPS"].astype(str).str.zfill(5)
    real_mean_yield = tmp.groupby("FIPS_str")["Yield"].mean().to_dict()

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
    test_Y_scaled  = scaler_Y.transform(test_Y.values.reshape(-1, 1)).ravel()

    model = Sequential()
    if crop == "Soy":
        model.add(Dense(64, activation="relu", input_shape=(len(input),)))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                      loss="mean_squared_error")
    elif crop == "Corn":
        model.add(Dense(64, activation="relu", input_shape=(len(input),)))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
                      loss="mean_squared_error")
    else:
        model.add(Dense(64, activation="relu", input_shape=(len(input),)))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss="mean_squared_error")

    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=1e-7, verbose=1)
    early_stop = EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True, verbose=1)

    def create_sample_weights(df_subset, emphasis_years={2012: 3.0, 2008: 2.0, 2016: 2.0}):
        weights = np.ones(len(df_subset))
        for year, weight in emphasis_years.items():
            year_mask = df_subset['Year'] == year
            weights[year_mask] = weight
        return weights

    train_weights = create_sample_weights(train_sub)
    val_weights   = create_sample_weights(val_sub)

    history = model.fit(
        train_X_scaled, train_Y_scaled,
        sample_weight=train_weights,
        epochs=2000, batch_size=64, verbose=1,
        validation_data=(val_X_scaled, val_Y_scaled, val_weights),
        callbacks=[early_stop, rlrop],
    )

    y_tr = np.array(train_Y.values.flatten())
    y_va = np.array(val_Y.values.flatten())
    y_te = np.array(test_Y.values.flatten())

    yhat_tr = scaler_Y.inverse_transform(model.predict(train_X_scaled, verbose=0).reshape(-1, 1)).ravel()
    yhat_va = scaler_Y.inverse_transform(model.predict(val_X_scaled,   verbose=0).reshape(-1, 1)).ravel()
    yhat_te = scaler_Y.inverse_transform(model.predict(test_X_scaled,  verbose=0).reshape(-1, 1)).ravel()

    r2_tr  = r2_score(y_tr, yhat_tr)
    r2_te  = r2_score(y_te, yhat_te)
    mae_tr = mean_absolute_error(y_tr, yhat_tr)
    mae_te = mean_absolute_error(y_te, yhat_te)
    print(f"[NN] r2 train={r2_tr:.4f}  test={r2_te:.4f}")

    models_list.append(model)
    run_train_list.append(crop_model_run_train)
    run_test_list.append(crop_model_run_test)
    train_loss.append(mae_tr)
    train_r2.append(r2_tr)
    test_loss.append(mae_te)
    test_r2.append(r2_te)
    scalers_list.append((scaler_X, scaler_Y))

    # ── CO2 sensitivity via numerical perturbation ───────────────
    stage_sensitivity = {f: [] for f in bimonthly_co2_features}

    for split_name, split_df in [('test', crop_model_run_test), ('train', crop_model_run_train)]:
        # Baseline prediction (one per row) —— used as the denominator for relative sensitivity
        split_X_scaled   = scaler_X.transform(split_df[input])
        base_pred_scaled = model.predict(split_X_scaled, verbose=0).ravel()
        base_pred        = scaler_Y.inverse_transform(base_pred_scaled.reshape(-1, 1)).ravel()

        stage_abs_effects = {}
        for stage_feature in bimonthly_co2_features:
            if stage_feature not in input:
                stage_abs_effects[stage_feature] = np.full(len(split_df), np.nan)
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
            stage_abs_effects[stage_feature] = pert_pred - base_pred

        combined_abs = np.zeros(len(split_df))
        for stage_feature in bimonthly_co2_features:
            if stage_feature in input:
                combined_abs += np.nan_to_num(stage_abs_effects[stage_feature], nan=0.0)

        # Overall relative sensitivity —— denominator uses the baseline prediction
        S_run = []
        for abs_eff, base_y in zip(combined_abs, base_pred):
            if (base_y is None) or (base_y == 0) or np.isnan(base_y) or np.isnan(abs_eff):
                S_run.append(np.nan)
            else:
                S_run.append((abs_eff / base_y) * 100.0)
        sensitivity_list.extend(S_run)

        # Per-stage relative sensitivity —— likewise uses the baseline prediction
        for stage_feature in bimonthly_co2_features:
            S_stage = []
            for abs_eff, base_y in zip(stage_abs_effects[stage_feature], base_pred):
                if (base_y is None) or (base_y == 0) or np.isnan(base_y) or np.isnan(abs_eff):
                    S_stage.append(np.nan)
                else:
                    S_stage.append((abs_eff / base_y) * 100.0)
            stage_sensitivity[stage_feature].extend(S_stage)

    return stage_sensitivity


# ============================================================================
# Helper: run logic for a single block (avoids repeating the same code 8 times)
# ============================================================================
def run_block_linear(tag, run_input, year_scale=1.0):
    """Run one linear block: for each seed, train, compute sensitivity, immediately save files + free memory."""
    print("=" * 60)
    print(f"Linear block: {tag}")
    print("=" * 60)
    global random_seed

    run_df     = source_df.copy()
    run_output = output_cols.copy()

    for i in range(num_runs):
        random_seed = random_seed_list[i]
        print(f"\n--- Run {i+1}/{num_runs}  (seed={random_seed}) ---")

        sensitivity          = []
        beta_time_list1      = []
        beta_co2_abs_list1   = []
        beta_co2_rel_list1   = []
        beta_year_abs_list1  = []
        beta_year_rel_list1  = []
        stage_sens_early     = []
        stage_sens_mid       = []
        stage_sens_late      = []

        stage_sens = linear_model(
            run_df, run_input, run_output, ref_dic,
            sensitivity, beta_time_list1,
            beta_co2_abs_list1, beta_co2_rel_list1,
            beta_year_abs_list1, beta_year_rel_list1,
            county_time_trend=0, deltaC=1.0, year_scale=year_scale,
        )
        if stage_sens is not None:
            stage_sens_early.extend(stage_sens['CO2_Early'])
            stage_sens_mid.extend(stage_sens['CO2_Mid'])
            stage_sens_late.extend(stage_sens['CO2_Late'])

        print(f"[seed={random_seed}] median sensitivity = "
              f"{np.nanmedian(np.asarray(sensitivity, dtype=float)):.6f}")

        save_per_seed_results(
            tag, random_seed, sensitivity,
            stage_sens_early, stage_sens_mid, stage_sens_late,
            is_nn=False,
        )

        # Clean up the current seed's memory
        cleanup_after_seed(is_nn=False)


def run_block_nn(tag, run_input):
    """Run one NN block: for each seed, train, compute sensitivity, immediately save files + free memory."""
    print("=" * 60)
    print(f"NN block: {tag}")
    print("=" * 60)
    global random_seed

    run_df     = source_df.copy()
    run_output = output_cols.copy()

    for i in range(num_runs):
        random_seed = random_seed_list[i]
        print(f"\n--- Run {i+1}/{num_runs}  (seed={random_seed}) ---")

        sensitivity            = []
        year_sensitivity_list  = []
        stage_sens_early       = []
        stage_sens_mid         = []
        stage_sens_late        = []

        # Record the lengths of the global train/test lists; after the run, take the elements newly added by this seed
        _tl = len(train_loss)
        _tr = len(train_r2)
        _el = len(test_loss)
        _er = len(test_r2)

        stage_sens = nn_model(
            run_df, run_input, run_output, ref_dic,
            sensitivity, year_sensitivity_list,
            county_time_trend=0,
        )
        if stage_sens is not None:
            stage_sens_early.extend(stage_sens['CO2_Early'])
            stage_sens_mid.extend(stage_sens['CO2_Mid'])
            stage_sens_late.extend(stage_sens['CO2_Late'])

        this_train_loss = train_loss[_tl:]
        this_train_r2   = train_r2[_tr:]
        this_test_loss  = test_loss[_el:]
        this_test_r2    = test_r2[_er:]

        print(f"[seed={random_seed}] median sensitivity = "
              f"{np.nanmedian(np.array(sensitivity)):.6f}")

        save_per_seed_results(
            tag, random_seed, sensitivity,
            stage_sens_early, stage_sens_mid, stage_sens_late,
            is_nn=True,
            year_sens=year_sensitivity_list,
            this_train_loss=this_train_loss,
            this_train_r2=this_train_r2,
            this_test_loss=this_test_loss,
            this_test_r2=this_test_r2,
        )

        # Clean up the current seed's memory (including the TF graph)
        cleanup_after_seed(is_nn=True)


# ============================================================================
# Run 8 blocks
# ============================================================================

# ── 4 Linear blocks ────────────────────────────────────────
run_block_linear("linear_baseline",      linear_baseline)
run_block_linear("linear_time",          linear_time)
run_block_linear("linear_co2",           linear_co2)
run_block_linear("linear_time_co2",      linear_time_co2)

# ── 4 NN blocks ────────────────────────────────────────────
run_block_nn("nn_baseline",      nn_baseline)
run_block_nn("nn_time",          nn_time)
run_block_nn("nn_co2",           nn_co2)
run_block_nn("nn_time_co2",      nn_time_co2)


# ============================================================================
# Final Predictions computation
# Load each model on demand and release it immediately after computing —— keeps peak memory within a few GB
# ============================================================================
print("\n" + "=" * 60)
print("Computing predictions from all saved per-seed models")
print("=" * 60)

MODEL_TAGS_IN_ORDER = [
    'linear_baseline', 'linear_time', 'linear_co2', 'linear_time_co2',
    'nn_baseline',     'nn_time',     'nn_co2',     'nn_time_co2',
]

# Mapping: the feature list corresponding to each tag
TAG_FEATURE_MAP = {
    'linear_baseline': linear_baseline,
    'linear_time':     linear_time,
    'linear_co2':      linear_co2,
    'linear_time_co2': linear_time_co2,
    'nn_baseline':     nn_baseline,
    'nn_time':         nn_time,
    'nn_co2':          nn_co2,
    'nn_time_co2':     nn_time_co2,
}

predictions = []
train_pred  = []

for tag in MODEL_TAGS_IN_ORDER:
    feats = TAG_FEATURE_MAP[tag]
    is_nn_tag = tag.startswith('nn_')

    for rs in random_seed_list:
        # Load the current seed's model on demand
        model    = pickle.load(open(f'{out_dir}/{crop}_model_{tag}_rs{rs}.p',     'rb'))
        test_df  = pickle.load(open(f'{out_dir}/{crop}_run_test_{tag}_rs{rs}.p',  'rb'))
        train_df = pickle.load(open(f'{out_dir}/{crop}_run_train_{tag}_rs{rs}.p', 'rb'))

        if is_nn_tag:
            scaler_X, scaler_Y = pickle.load(
                open(f'{out_dir}/{crop}_scaler_{tag}_rs{rs}.p', 'rb'))
            test_X_scaled  = scaler_X.transform(test_df[feats])
            tp = scaler_Y.inverse_transform(model.predict(test_X_scaled, verbose=0))
            train_X_scaled = scaler_X.transform(train_df[feats])
            rp = scaler_Y.inverse_transform(model.predict(train_X_scaled, verbose=0))
            del scaler_X, scaler_Y, test_X_scaled, train_X_scaled
        else:
            tp = model.predict(test_df[feats])
            rp = model.predict(train_df[feats])

        predictions.append(tp)
        train_pred.append(rp)

        # Release immediately
        del model, test_df, train_df
        if is_nn_tag:
            tf.keras.backend.clear_session()
        gc.collect()

    print(f"  Done {tag}")

print(f"Total predictions: {len(predictions)}")
print(f"Total train_pred: {len(train_pred)}")

pickle.dump(predictions, open(f'{out_dir}/{crop}_predictions.p', 'wb'))
pickle.dump(train_pred,  open(f'{out_dir}/{crop}_train.p',       'wb'))

print("✓ All per-seed files saved and predictions computed.")
