# 3 — Soybean CO2-fertilization effect (Step 3)

**Purpose:** estimate soybean **ΔCO2-sensitivity**. In the soybean Scenario-2
model the time terms (Year/Year²) are replaced by the maize-derived agronomy
proxy (`Tech_trend`); the coefficient on CO2 is the ΔCO2-sensitivity.

**Input ←**
- `Data/.../Soy_CAMS_1979_2023_allcounties.csv`
- `2-Corn_trend/Corn_county_year_tech_median_trend.p` (the agronomy proxy)
- `1-Reliable_counties/reliable_counties_aggregated.csv` (fixed county set)

**Output →** soybean `*_sensitivities*.p` (overall + Early/Mid/Late stage).

---

## ⚠️ Two implementations here — pick the canonical one before release

- `Step2_soybean_model.ipynb` — uses the **median** corn trend
  (`diff_smooth_median`) + the **aggregated** 705-county set. **This matches the
  manuscript.** Currently still a Colab notebook running only 2 seeds
  (`random_seed_list = [1,2]`) — needs cleanup to 20 seeds + relative paths.
- `2_model_clean_soy_per_seed.py` — a **per-seed** variant: each seed uses its own
  corn trend (`Corn_..._diff_smooth_rs{k}.p`) and its own per-seed reliable
  counties. 20 seeds, clean `.py`. Does **not** consume the median trend / fixed
  county set.

The model core (NN architecture, CO2 perturbation `+1 ppm`, %-sensitivity) is
identical between the two; they differ only in the trend/county inputs and seed
count. **Confirm which folder produced the published numbers, then keep one and
delete (or mark `supplementary_`) the other.**
