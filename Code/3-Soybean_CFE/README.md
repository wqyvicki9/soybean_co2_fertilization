# 3 — Soybean CO2-fertilization effect (Step 3)

**Purpose:** estimate soybean **ΔCO2-sensitivity**. In the soybean Scenario-2
model the time terms (Year/Year²) are replaced by the maize-derived agronomy
proxy (`Tech_trend`); the coefficient on CO2 is the ΔCO2-sensitivity.

**Input ←**
- `Data/.../Soy_CAMS_1979_2023_allcounties.csv`
- `2-Corn_trend/Corn_county_year_tech_median_trend.p` (the agronomy proxy)
- `1-Reliable_counties/reliable_counties_aggregated.csv` (fixed county set)

**Output →** soybean `*_sensitivities*.p` (overall + Early/Mid/Late stage).
and all other files for analysis

---

- `Step2_soybean_model.ipynb` — uses the **median** corn trend
  (`diff_smooth_median`) + the **aggregated** 705-county set. 
