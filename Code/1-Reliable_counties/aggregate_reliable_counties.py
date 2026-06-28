# -*- coding: utf-8 -*-
"""
Aggregate the 20 per-seed corn-soy residual-correlation results into a single
fixed "reliable county" set (manuscript Step 2, retention criterion).

A county is retained ("reliable") if it shows a statistically significant
(p < 0.01) lag-0 correlation between corn and soybean non-climate yield growth
in at least 16 of the 20 random-seed ensemble members (>= 80%).

This script is the missing aggregation step between
  corn_soy_residual_correlation_per_seed.py   (writes per-seed CSVs)
and
  ../3-Corn_trend/compute_corn_trend_fixed_counties.py
  (reads reliable_counties_aggregated.csv as the FIXED county set).

------------------------------------------------------------------------------
Two input modes (auto-detected):

  FULL mode  (preferred, reproduces the published r_median exactly):
    Reads per-seed tables that contain *all* counties (significant or not),
    pattern  ALL_PATTERN  below. Then:
        r_median   = ensemble median of r_lag0 across all 20 seeds
        lag_median = median of best_lag across the significant seeds
        n_sig      = #seeds with p_lag0 < P_THRESHOLD
    To produce these tables, save the full `per_county` (not only `sig`) in
    corn_soy_residual_correlation_per_seed.py -- see the one-line patch noted
    at the bottom of this file.

  SIG mode   (fallback, uses the files already on disk):
    Reads the per-seed *significant-only* CSVs (reliable_counties_NN_rs{rs}.csv).
    The retained county set, n_sig_seeds, pct_sig_seeds and is_reliable are
    reproduced EXACTLY. r_median / lag_median are computed across the
    significant seeds only (so r_median can differ slightly from the published
    value for counties significant in fewer than 20 seeds).
------------------------------------------------------------------------------
"""

import os
import glob
import re
import numpy as np
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================
# Directory holding the per-seed outputs of corn_soy_residual_correlation_per_seed.py.
# Override with env var RELIABLE_DIR; otherwise the first existing candidate is used.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.environ.get('RELIABLE_DIR', ''),
    os.path.join(_HERE, 'reliable_counties_per_seed'),                          # alongside this script
    os.path.join(_HERE, '..', '..', 'Results', 'reliable_counties_per_seed'),   # repo Results/ layout
]
RELIABLE_DIR = next((os.path.abspath(p) for p in _CANDIDATES if p and os.path.isdir(p)),
                    os.path.join(_HERE, 'reliable_counties_per_seed'))

# Where to write the aggregated fixed county set
OUT_CSV = os.path.join(RELIABLE_DIR, 'reliable_counties_aggregated.csv')

N_SEEDS          = 20      # number of random-seed ensemble members
SIG_SEED_MIN     = 16      # retain county if significant in >= this many seeds (>=80%)
P_THRESHOLD      = 0.01    # per-seed significance threshold (FULL mode only)

# File-name patterns (rs index captured by the group)
SIG_PATTERN = 'reliable_counties_NN_rs*.csv'          # significant-only (always present)
ALL_PATTERN = 'corr_all_counties_NN_rs*.csv'          # full per-county (FULL mode, optional)


def _seed_from_name(path):
    m = re.search(r'rs(\d+)\.csv$', os.path.basename(path))
    return int(m.group(1)) if m else None


def _load_per_seed(pattern):
    frames = []
    for f in sorted(glob.glob(os.path.join(RELIABLE_DIR, pattern))):
        rs = _seed_from_name(f)
        if rs is None:
            continue
        d = pd.read_csv(f, dtype={'FIPS': str})
        d['FIPS'] = d['FIPS'].str.zfill(5)
        d['rs'] = rs
        frames.append(d)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ============================================================================
# Aggregation
# ============================================================================
def aggregate():
    all_df = _load_per_seed(ALL_PATTERN)

    if all_df is not None:
        # ---------- FULL mode ----------
        mode = 'FULL'
        all_df['is_sig'] = (all_df['p_lag0'] < P_THRESHOLD) & all_df['r_lag0'].notna()

        # r_median: ensemble median of lag-0 correlation across ALL seeds
        r_med = all_df.groupby('FIPS')['r_lag0'].median().rename('r_median')

        # lag_median: median best_lag across the SIGNIFICANT seeds
        sig_only = all_df[all_df['is_sig']]
        lag_med = sig_only.groupby('FIPS')['best_lag'].median().rename('lag_median')

        # n_sig_seeds: how many seeds the county is significant in
        n_sig = sig_only.groupby('FIPS')['rs'].nunique().rename('n_sig_seeds')

        agg = pd.concat([r_med, lag_med, n_sig], axis=1).reset_index()
        # counties never significant in any seed -> n_sig NaN -> 0
        agg['n_sig_seeds'] = agg['n_sig_seeds'].fillna(0).astype(int)

    else:
        # ---------- SIG mode (fallback) ----------
        sig_df = _load_per_seed(SIG_PATTERN)
        if sig_df is None:
            raise FileNotFoundError(
                f"No per-seed files found in {RELIABLE_DIR} "
                f"(looked for '{ALL_PATTERN}' and '{SIG_PATTERN}').")
        mode = 'SIG'
        # In the significant-only CSVs, a county's presence == significant in that seed.
        agg = (sig_df.groupby('FIPS')
               .agg(r_median=('r_lag0', 'median'),
                    lag_median=('best_lag', 'median'),
                    n_sig_seeds=('rs', 'nunique'))
               .reset_index())

    # ---------- retention criterion (shared) ----------
    agg['pct_sig_seeds'] = 100.0 * agg['n_sig_seeds'] / N_SEEDS
    agg['is_reliable'] = agg['n_sig_seeds'] >= SIG_SEED_MIN

    reliable = (agg[agg['is_reliable']]
                .sort_values('r_median', ascending=False)
                .reset_index(drop=True))

    cols = ['FIPS', 'r_median', 'lag_median', 'n_sig_seeds', 'pct_sig_seeds', 'is_reliable']
    reliable = reliable[cols]
    reliable.to_csv(OUT_CSV, index=False)

    # ---------- report ----------
    print('=' * 60)
    print(f"Aggregation mode : {mode}")
    print(f"Seeds found      : {agg['n_sig_seeds'].max()} max sig / {N_SEEDS} total")
    print(f"Counties scanned : {len(agg)}")
    print(f"Reliable (>= {SIG_SEED_MIN}/{N_SEEDS}) : {len(reliable)}")
    if len(reliable):
        print(f"  r_median range : {reliable['r_median'].min():.4f} .. "
              f"{reliable['r_median'].max():.4f}")
        lag0 = (reliable['lag_median'] == 0).mean() * 100
        print(f"  lag_median == 0: {lag0:.1f}% of reliable counties")
    print(f"Saved -> {OUT_CSV}")
    print('=' * 60)
    return reliable


if __name__ == '__main__':
    aggregate()


# ----------------------------------------------------------------------------
# One-line patch for corn_soy_residual_correlation_per_seed.py to enable FULL
# mode (exact reproduction of the published r_median). Inside the per-seed loop,
# right after `per_county, n_both = per_seed_correlation(...)`, add:
#
#     per_county.assign(FIPS=per_county['FIPS'].astype(str).str.zfill(5)) \
#         .to_csv(f'{reliable_dir}/corr_all_counties_NN_rs{rs}.csv', index=False)
#
# (this saves *all* counties, not just the significant subset)
# ----------------------------------------------------------------------------
