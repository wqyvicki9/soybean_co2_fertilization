# 2 — Corn-derived background trend (Step 1)

**Purpose:** extract the county-specific non-climate (agronomy/technology) trend
from maize, to be transferred to soybean.

**Scripts (run in this order)**
1. `0_model_clean_corn_per_seed.py` — corn **full** model (Scenario 2): climate +
   **Year + Year²** + county fixed effects, 20 seeds, linear + NN (8 blocks).
   Produces the `nn_time` model + its train/test split + scaler per seed.
2. `compute_corn_trend_fixed_counties.py` — using `nn_time`, the trend is
   `ΔY = f(year, X, α) − f(1979, X, α)` (climate held fixed). Filtered to the
   fixed reliable county set, 5-yr smoothed, then the **median across 20 seeds**.

**Input ←**
- `Data/Corn_CAMS_1979_2023_allcounties.csv` (script 1)
- corn `nn_time` model from script 1 (`Results/Results_random_seeds_20/`)
- `1-Reliable_counties/reliable_counties_aggregated.csv` (fixed county set)

**Output →** `Corn_county_year_tech_median_trend.p` (column `diff_smooth_median`)
— the maize-derived agronomy proxy used by `3-Soybean_CFE/`.
