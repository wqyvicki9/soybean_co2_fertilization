# Code — pipeline overview

Estimating the soybean CO2-fertilization effect (ΔCO2-sensitivity) from county
maize/soybean yields, climate, and CO2. Run the folders in order; each folder's
README states its inputs and outputs.

```
Data/  (Corn_CAMS, Soy_CAMS : county-year yield + bimonthly climate + CO2)
   │
0-Pre_Screen/        Scenario-1 climate-only yield models (corn & soy)
   │                 → per-seed predictions incl. observed Yield + FIPS/Year meta
   ▼
1-Reliable_counties/ correlation of the annual change of the non-climate residual
   │                 between corn & soy, per county; keep counties significant in
   │                 ≥16/20 seeds  → reliable_counties_aggregated.csv (705 counties)
   ▼
2-Corn_trend/        corn full model (Scenario 2, Year+Year²) → nn_time;
   │                 ΔY = f(year) − f(1979) on the fixed county set, 5-yr smooth,
   │                 20-seed median  → Corn_county_year_tech_median_trend.p
   ▼
3-Soybean_CFE/       soybean Scenario 2: replace Year/Year² with the maize-derived
                     agronomy proxy (Tech_trend) + CO2; the CO2 coefficient is the
                     ΔCO2-sensitivity
```

## Key definitions

- **Non-climate residual** (used in 1-Reliable_counties): `residual = observed
  USDA yield − climate-only (Scenario 1) prediction`, for each crop. The
  correlation is computed on the **annual change** (`.diff()`, 5-yr rolling mean)
  of this residual.
- **Ensemble**: every model is trained with **20 random seeds**; reported results
  are the median across seeds.

## Conventions

- **Run every script (and the notebook) from the `Code_availability/` repo root.**
  Each one sets `folder = '.'`, reads inputs from `Data/`, and writes all outputs
  under `Results/` (model dumps in `Results/Results_*`, the corn trend in
  `Results/corn_trend_per_seed/`, reliable counties in
  `Results/reliable_counties_per_seed/`, figures in `Results/Figures/`).
- `1-Reliable_counties` plots need a US county shapefile at `Data/county_shp/`
  (supply separately; used only for the maps, not for any numeric result).
- File-name prefixes (`0_`, `2_`) are legacy crop tags (`0`=corn, `2`=soy) and do
  not imply run order — follow the folder numbers.
