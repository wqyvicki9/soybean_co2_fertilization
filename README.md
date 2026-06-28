# Soybean CO₂-fertilization effect from county yields — code & data

Code and processed data to reproduce the soybean **ΔCO₂-sensitivity** estimates:
the soybean CO₂-fertilization response, isolated by transferring a maize-derived
background (agronomy/technology) trend onto soybean so that the soybean CO₂
coefficient captures the soybean-minus-maize CO₂ effect.

> **Paper:** _<title>_, <authors>, <journal/year>. DOI: <doi>  ← fill in before release
> **Contact:** <corresponding author / email>

## Overview

Yields are modelled under two scenarios with 9 bimonthly climate predictors and
county fixed effects:

- **Scenario 1 (climate-only):** yield ~ climate + county fixed effects.
- **Scenario 2:** maize adds `Year + Year²` (long-term trend); soybean replaces
  the time terms with the **maize-derived agronomy proxy + CO₂**.

The pipeline (1) builds the non-climate residual for both crops, (2) keeps the
counties where maize and soybean non-climate yield growth co-move, (3) extracts
the maize background trend, and (4) feeds it into the soybean Scenario-2 model;
the CO₂ coefficient there is the ΔCO₂-sensitivity. Every model is trained with
**20 random seeds**; reported quantities are the median across seeds.

## Repository structure

```
Code_availability/
├── README.md              ← this file
├── requirements.txt
├── Code/                  ← pipeline (run folders 0 → 3 in order; see Code/README.md)
│   ├── 0-Pre_Screen/        climate-only (Scenario 1) yield models, corn & soy
│   ├── 1-Reliable_counties/ residual-growth correlation → fixed 705-county set
│   ├── 2-Corn_trend/        corn full model (Scenario 2) → maize background trend
│   └── 3-Soybean_CFE/       soybean Scenario 2 → ΔCO₂-sensitivity
├── Data/
│   └── Aggregated_yield_climate_co2/
│       ├── Corn_CAMS_1979_2023_allcounties.csv
│       └── Soy_CAMS_1979_2023_allcounties.csv
└── Results/               ← pipeline outputs (per-seed dumps, trend, reliable counties)
```

Each `Code/<step>/` folder has its own README stating that step's inputs and
outputs. A full pipeline diagram is in [`Code/README.md`](Code/README.md).

## Data

Two county-by-year panels (1979–2023), one per crop. Each row is a county-year.

| Column | Description |
|---|---|
| `FIPS` | 5-digit county code |
| `Year` | calendar year (1979–2023) |
| `Yield` | reported yield, bu/acre (USDA NASS) |
| `ppt_May … ppt_Oct` | monthly precipitation |
| `tmax_/tmin_/tmean_May … Oct` | monthly max/min/mean temperature |
| `co2_May … co2_Oct` | solar-radiation-weighted monthly CO₂ concentration (CAMS) |

Monthly variables are averaged into three **bimonthly stages** inside the code:
Early (May+Jun), Mid (Jul+Aug), Late (Sep+Oct). Counties with fewer than ~80% of
years are dropped. Corn: 2,475 counties / 83,802 rows. Soy: 2,212 / 71,112.

> Provenance: yields — USDA NASS; climate — <PRISM/source>; CO₂ — CAMS. Fill in
> exact sources/versions before release.

## Setup

```bash
pip install -r requirements.txt   # numpy, pandas, scipy, scikit-learn, tensorflow, matplotlib, geopandas
```

Tested with Python 3.9. Training all 20 seeds is compute-heavy (designed for a
SLURM cluster; each model script frees memory and writes per-seed files so it
runs in a few GB of RAM).

## How to run

Run everything **from this `Code_availability/` directory** (every script sets
`folder = '.'` and reads `Data/` / writes `Results/`). Execute in pipeline order:

```bash
# 0 — climate-only (Scenario 1) models
python Code/0-Pre_Screen/0_model_clean_corn_climate_only.py
python Code/0-Pre_Screen/2_model_clean_soy_climate_only.py

# 1 — reliable counties (shared-trend validation)
python Code/1-Reliable_counties/corn_soy_residual_correlation_per_seed.py
python Code/1-Reliable_counties/aggregate_reliable_counties.py      # → reliable_counties_aggregated.csv (705 counties)

# 2 — corn full model + maize background trend
python Code/2-Corn_trend/0_model_clean_corn_per_seed.py
python Code/2-Corn_trend/compute_corn_trend_fixed_counties.py       # → Corn_county_year_tech_median_trend.p

# 3 — soybean CO₂-fertilization effect
#     open Code/3-Soybean_CFE/Step2_soybean_model.ipynb  (or the .py variant)
```

The processed intermediate outputs needed downstream (reliable counties, the
maize trend) are already included under `Results/`, so a step can be reproduced
without re-running the heavy upstream model training.

## Key methodological choices

- **Non-climate residual** = `observed USDA yield − climate-only (Scenario 1)
  prediction`, per crop. County retention in step 1 correlates the **annual
  change** of this residual (5-yr rolling mean) between corn and soy.
- **Reliable county set**: a county is kept if it shows a significant (p < 0.01)
  lag-0 correlation in **≥ 16 of 20 seeds** → 705 counties.
- **Maize trend transfer**: `ΔY = f(year) − f(1979)` from the corn `nn_time`
  model, 5-yr smoothed, median across seeds; used in place of soybean time terms.

## Known TODO before public release

- **Two soybean implementations** in `3-Soybean_CFE/`: the notebook (median trend
  + fixed county set, matches the manuscript) and `2_model_clean_soy_per_seed.py`
  (per-seed variant). Keep the one that produced the published numbers; mark or
  remove the other. See `Code/3-Soybean_CFE/README.md`.
- The county **shapefile** for the step-1 maps is not included — add it at
  `Data/county_shp/` or document a download link (it affects figures only).
- Fill in the paper/citation/data-provenance placeholders above.
