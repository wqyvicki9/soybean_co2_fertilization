# 0 — Pre-screen: climate-only (Scenario 1) yield models

**Purpose:** train the climate-only yield model for each crop, so the next step
can form the *non-climate residual* and screen counties.

**Scripts**
- `0_model_clean_corn_climate_only.py` — corn, 9 bimonthly climate features +
  county fixed effects, **no Year, no CO2** (Scenario 1). 20 seeds, linear + NN.
- `2_model_clean_soy_climate_only.py` — same design for soybean.

**The below input and output are all samples here** 

**Input ←** `Data/{Corn,Soy}_CAMS_1979_2023_allcounties.csv`

**Output →** `Results/Results_{Corn,Soy}_ClimateOnly_20seeds/{crop}_predictions_nn.p`
(and `_linear.p`). Each is a per-seed list of dicts holding `test_pred`,
`train_pred`, and `*_meta` (FIPS, Year_actual, **observed Yield**). Consumed by
`1-Reliable_counties/`.

**Residual definition (formed downstream):**
`residual = observed Yield − climate-only prediction` (matches the manuscript,
Step 2). Train+test predictions are concatenated there to rebuild the full
county-year panel before differencing.

> Note: the corn/soy **full** models are not part of pre-screening. The corn full
> model lives in `2-Corn_trend/` (it feeds the trend); the soybean full model
> lives in `3-Soybean_CFE/`.
