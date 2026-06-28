# 1 — Reliable counties (shared-trend validation, Step 2)

**Purpose:** identify counties where the non-climate yield signal of corn and
soybean move together, so the maize trend can be transferred to soybean.

**Scripts (run in this order)**
1. `corn_soy_residual_correlation_per_seed.py` — for each seed: form
   `residual = observed Yield − climate-only prediction` for both crops, take the
   **annual change** (`.diff()`, 5-yr centered rolling mean), and compute the
   per-county Pearson correlation at lag 0 (also scans lags −20..+20). Saves the
   significant counties (p < 0.01) per seed → `reliable_counties_per_seed/reliable_counties_NN_rs{0..19}.csv`.
2. `aggregate_reliable_counties.py` — keep a county if it is significant in
   **≥ 16 of 20 seeds (≥80%)** → `reliable_counties_aggregated.csv` (705 counties).
   See the FULL/SIG mode note in the script header for exact reproduction of the
   reported `r_median`.

**Input ←** `0-Pre_Screen/` climate-only predictions for both crops.

**Output →** `reliable_counties_aggregated.csv` — the **fixed county set** used by
`2-Corn_trend/` and `3-Soybean_CFE/`.
