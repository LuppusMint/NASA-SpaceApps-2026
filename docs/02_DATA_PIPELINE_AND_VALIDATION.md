# Data Pipeline and Validation

**Fire Relay South America - reproducible processing from raw FIRMS archives to the validated monthly product.**

This document describes the implemented pipeline, the committed counts, the validation design, and the scientific limits. It is deliberately specific about what is verified and what is estimated.

---

## 1. End-to-end workflow

```text
NASA FIRMS Archive & NRT
  MODIS C6.1 (Terra, Aqua)
  VIIRS S-NPP, NOAA-20/J1, NOAA-21/J2
          |
          v
scripts/download_firms_sa.py
  archive zips + FIRMS_MAP_KEY API gap-fills
          |
          v
scripts/build_canonical_parquet.py
  parse, quality flags, WGS84 -> Albers, 10 km cell ids
  canonical partitioned Parquet: data/parquet/south_america_hotspots/
          |
          v
src/data/clean_and_flag.py
  type 1/2/3 removal, static thermal mask, FRP/brightness repair
  SAMA flags, overpass events, QC summary
          |
          v
src/data/build_spatiotemporal_cubes.py
  10 km, 50 km, Admin-1, country, continent and daily-season cubes
          |
          v
src/models/freeze_validation.py
  pre-drift overlap 2012-02..2020-12
  200 km spatial blocks, 5 folds, temporal splits
  data/processed/overlap_cell50km_2012_2020.parquet
  data/processed/splits.json
          |
          v
src/models/harmonize_pipeline.py
  M0, M1, M2, M3 out-of-fold benchmark
  production fit, Kalman relay, hierarchical reconciliation
  data/processed/benchmarks_m0_m3.json
  data/processed/oof_m0_m3.parquet
  data/processed/experiments.jsonl
          |
          v
src/models/calendar_analytics.py
  frozen 2001-2020 climatology, severity matrix
  Theil-Sen + Hamed-Rao + BH FDR, GEV return levels
  90-day early-onset curves
          |
          v
app/server.py
  offline WSGI API over the processed Parquet cubes
```

Each stage is deterministic: fixed seeds, stable sort keys, and explicit versioned schemas. The pipeline is out-of-core: DuckDB and Polars process the 85M-row record, and the full raw table is never materialised in pandas.

---

## 2. Stage-by-stage implementation

### 2.1 Download (`scripts/download_firms_sa.py`)

- Reads `FIRMS_MAP_KEY` strictly from the environment. No key is stored in the repository.
- Uses the FIRMS area API for 2023-2026 gap-fills and the public archive zips for the historical record.
- Writes the raw archive under `data/raw/`, which is git-ignored.
- Records source, date range, and checksums for provenance.

### 2.2 Canonical Parquet (`scripts/build_canonical_parquet.py`)

- Parses the FIRMS CSVs with DuckDB, normalises column names, and derives a UTC acquisition timestamp.
- Harmonises the confidence representation across MODIS and VIIRS.
- Projects WGS84 coordinates into the Albers Equal-Area grid with `pyproj`, and assigns 10 km cell indices and ids.
- Joins Natural Earth 1:10m Admin-0 and Admin-1 boundaries with `geopandas`/`shapely`.
- Writes partitioned Parquet under `data/parquet/south_america_hotspots/` and a `DATA_SUMMARY.json` provenance record.

### 2.3 Cleaning and flags (`src/data/clean_and_flag.py`)

- Removes `type IN (1,2,3)` where the type flag exists.
- Identifies persistent static 10 km thermal cells with the temporal rule (at least 9 active months per year over at least 5 years) and the type-1/2 dominance rule.
- Repairs negative FRP and zero-brightness rows.
- Flags SAMA containment and suspect nocturnal NRT VIIRS detections; after the nominal+ confidence filter the suspect class is empty, so the flag removes no rows.
- Consolidates detections into overpass events using satellite, date, day/night, and a 2-hour overpass bin.
- Writes `qc_cleaning_summary.json`, `static_thermal_cells.parquet`, and `static_thermal_cells_summary.json`.

### 2.4 Spatiotemporal cubes (`src/data/build_spatiotemporal_cubes.py`)

- Builds the 10 km cell-month cube, the 50 km cell-month cube, Admin-1, country, and continental monthly cubes.
- Builds the Admin-1 daily-season cube used by the 90-day onset curve.
- Preserves exact additive conservation across the hierarchy.
- Handles the partial last month explicitly with `is_partial_month` and a maturity label. November 2000 is a complete month and is not flagged as partial; the year 2000 is a two-month start.

### 2.5 Frozen validation (`src/models/freeze_validation.py`)

- Restricts calibration to **2012-02..2020-12**.
- Builds the overlap table with **930,100** rows = **465,050** cell-months x **2** counterfactual replicas (`TERRA_AQUA`/`TERRA_ONLY`, same S-NPP target) and **7,184** unique 50 km cells.
- Assigns **200 km** spatial blocks to **5** folds. A block is never split across folds, and every cell-month inherits its block's fold.
- Stores the fold assignment, block assignment, fold sizes, temporal folds, dataset SHA-256, and schema fingerprints in `data/processed/splits.json`.
- Verifies that the overlap hash matches the frozen manifest before training.

### 2.6 Harmonization (`src/models/harmonize_pipeline.py`)

- Fits M0-M3 on identical folds and seeds.
- Fits the M2 covariate-shift weights, hurdle threshold, and sigma scale on outer-training data only.
- Fits the M3 simplex blend in temporal order (warm-up fold, then strictly earlier folds).
- Applies the 1D Kalman relay smoothing and the non-negative hierarchical reconciliation.
- Writes out-of-fold predictions, the benchmark JSON, and four experiment records.

### 2.7 Calendar analytics (`src/models/calendar_analytics.py`)

- Builds the frozen **2001-2020** climatology.
- Builds the monthly severity matrix with z-scores and percent-of-median.
- Runs Theil-Sen + Hamed-Rao Mann-Kendall + Benjamini-Hochberg FDR.
- Fits GEV to the annual maximum of the monthly SREHI/FRP series with convergence, KS checks and a sanity bound (10-year level <=10x the observed annual maximum); parametric bootstrap CIs (500 replicates) are reported for every published fit where the bootstrap converges.
- Builds true trailing 90-day early-onset curves on a 5-day step. The window crosses year boundaries; the first 89 days of the record are `INSUFFICIENT_HISTORY`; thresholds use only complete windows with at least 10 years per calendar day, otherwise `NO_THRESHOLD`.

---

## 3. Data provenance

### 3.1 Source counts

The canonical land panel contains **85,002,288** rows. After deterministic quality control, static-source masking, and SAMA treatment, **75,556,207** rows remain in the analysis panel used by the models (referred to in the project brief as the vegetated panel; operationally it is the quality-controlled land panel, not a separate vegetation classification).

| Quantity | Verified value |
|---|---:|
| Raw NASA FIRMS detections, South America | **85,002,288** |
| Clean detections after quality control | **75,556,207** |
| Temporal coverage | **2000-11** to **2026-11** |
| Months | **313** |
| Countries and territories | **14** |
| Country-month rows in cube/CSV | **4,014** (missing rows mean no detections, not missing data) |
| Admin-1 units | **243** |
| Historical active 10 km cells | **143,613** |

### 3.2 Cleaning ledger

| Quantity | Verified value |
|---|---:|
| Removed by `type IN (1,2,3)` | **2,703,746** |
| Removed as persistent static-cell detections | **2,872,511** |
| Clean candidate detections in static cells | **472,710** |
| Negative FRP rows corrected | **138** |
| Zero-brightness rows corrected | **39** |
| Rows intersecting SAMA | **76,388,628** |
| Remaining suspect nocturnal detections | **0** (the suspect class is empty after the nominal+ filter; no rows were removed by the SAMA suspect rule) |

### 3.3 Counts by sensor

| Sensor slot | Clean detections | Raw detections |
|---|---:|---:|
| `VIIRS_SNPP` | **31,965,171** | **36,304,172** |
| `VIIRS_N20` | **19,370,162** | **22,017,351** |
| `MODIS_Aqua` | **10,162,384** | **10,897,858** |
| `VIIRS_N21` | **7,965,108** | **9,038,172** |
| `MODIS_Terra` | **6,093,382** | **6,744,735** |

The archive contains earlier and later partial years; the table covers the full 2000-11..2026-11 window. The 2026 months are NRT/partial and are labelled accordingly.

### 3.4 Portable provenance

`data/DATA_SUMMARY.json` and `data/PROVENANCE.json` record the generated counts and the pipeline stages. The committed copies use relative paths only. The raw archive is not redistributed.

---

## 4. Conservation and integrity checks

### 4.1 Exact integrated FRP

The integrated continental FRP before harmonization is

\[
\sum_{i,t} F_{i,t} = 1{,}453{,}803{,}616.63\ \text{MW}.
\]

The hierarchy is checked after every aggregation:

- 10 km -> 50 km (exact sum);
- country = sum of its assigned 50 km cells;
- Admin-1 -> country (country is the sum of Admin-1; Admin-1 is a proportional downscaling of the country total by raw overpass/FRP shares, not a bottom-up sum of cells);
- country -> South America (exact sum).

The 50 km -> Admin-1 step is **not** a bottom-up aggregation and is not claimed as one. The Admin-1 -> country check is tautological by construction; it verifies the allocation, not an independent cell sum.

The maximum observed differences after M3 reconciliation are:

| Transition | Max SREHI difference | Max FRP difference (MW) |
|---|---:|---:|
| Admin-1 -> Country | **4.37e-11** | **3.73e-09** |
| Country -> South America | **5.82e-11** | **1.86e-09** |

The minimum reconciled value is **>= 0.0**. The reconciliation is proportional and non-negative; the residuals are numerical round-off, not unallocated fire activity.

### 4.2 Frozen split integrity

- **5** spatial folds, **200 km** blocks.
- **930,100** overlap rows = **465,050** cell-months x 2 replicas; **7,184** unique 50 km cells.
- Dataset SHA-256: `ff927d7b27a7124d44cfcb776bb26ae26b13f24cc7c4ed1b786feb3db02ee0d6`.
- No 50 km cell appears in more than one fold.
- No spatial block appears in more than one fold.
- Temporal folds preserve `min_train_time <= min_val_time`.

### 4.3 Climatology isolation

The 2001-2020 climatology is built only from years 2001 through 2020. The test suite copies the processed directory, multiplies the 2024 harmonized values by 3, reruns the analytics, and requires the climatology file to be identical. This is a direct leakage test, not an assertion.

---

## 5. Quantitative validation results

### 5.1 M0-M3 benchmark

| Target | Model | RMSE | R-squared |
|---|---|---:|---:|
| SREHI | M0 raw | 24.6227 | 0.3533 |
| SREHI | M1 physical bridge | 26.7017 | 0.2374 |
| SREHI | M2 hurdle Tweedie + covariate shift | 11.8025 | 0.8512 |
| SREHI | **M3 simplex + Kalman + reconciliation** | **12.8838** | **0.8171** |
| FRP (MW) | M0 raw | 1377.5885 | 0.3973 |
| FRP (MW) | M1 physical bridge | 1385.0089 | 0.3928 |
| FRP (MW) | M2 hurdle Tweedie + covariate shift | 790.2797 | 0.8018 |
| FRP (MW) | **M3 simplex + Kalman + reconciliation** | **833.9594** | **0.7779** |

M2 is the point-error minimum. M3 is the operational product because it satisfies the hierarchy exactly, stays non-negative, and exposes a coherent 90% interval. The benchmark reports both; no result is suppressed.

### 5.2 Relay continuity

The aggregate absolute step discontinuity `|delta_step|` falls **92.84%**, from **2.1892** to **0.1567**. The per-transition SREHI step ratios are:

| Relay year | Raw ratio | M3 ratio | Log-space reduction |
|---:|---:|---:|---:|
| 2002 | 3.414 | 1.523 | 65.7% |
| 2012 | 3.938 | 1.177 | 88.1% |
| 2018 | 1.407 | 0.836 | 47.7% |
| 2021 | 0.738 | 0.754 | 7.2% |
| 2023 | 1.375 | 1.000 | 99.9% |

The 2021 transition is the weakest because it is dominated by orbital drift and a genuine sampling change, not a sensor addition. The 2023 transition is the strongest because the VIIRS backbone is deduplicated.

### 5.3 Cross-regime guard

| M3 metric | Regime RMSE ratio | Pairwise RMSE |
|---|---:|---|
| SREHI | 1.2572 | 11.5281 vs 14.4926 |
| FRP (MW) | 1.3718 | 699.5017 vs 959.5740 |

All checks report `passed: true`. The guard verifies that the reconstruction does not collapse or explode when the sensor mix changes; it does not claim the regimes are physically identical.

### 5.4 Calendar and trend results

- **45 of 242** Admin-1 units are significant in SREHI at q < 0.05: **34 increasing**, **11 decreasing**.
- **Peru** is the only significant country: slope **+903.23/yr**, z **3.34**, q **0.0134**.
- Leading increasing Admin-1 units: **Amazonas (Brazil)** **+785.81/yr** (q **0.0125**), **Ucayali (Peru)** **+186.55/yr** (q **0.0028**), **Loreto (Peru)** **+133.58/yr** (q **0.0028**).
- Critical extreme months, 2024-2026: **2024-02 Suriname** (z **8.28**), **2025-12 Peru** (z **6.51**), **2024-02 Guyana** (z **5.69**), **2024-03 Ecuador** (z **5.05**), **2024-02 Bolivia** (z **4.95**).
- Historical continental SREHI peak: **2007-09 = 293,471.7**, vs **2004-09 = 262,575.2** and **2010-09 = 239,710.3**.
- 90-day early onset: 2024 **5,125** critical / **3,009** high; 2025 **2,891** / **2,448**; 2026 **3,232** / **2,176** (through November). The curve uses a true trailing 90-day window across year boundaries; the first 89 days are `INSUFFICIENT_HISTORY` and thresholds require at least 10 complete years.

---

## 6. Test coverage

Run:

```bash
python -m unittest discover -s tests
```

| Test module | Scope |
|---|---|
| `tests/test_data_cubes.py` | Cleaning flags, static-cell detection, cube conservation, schema |
| `tests/test_harmonization_models.py` | Frozen splits, M0-M3 benchmark, hierarchy, climatology/trends/GEV/onset |
| `tests/test_webapp_api.py` | Catalog, calendar, spatial, early onset, CSV, health, static UI |
| `tests/test_submission_package.py` | Exact CSV contract, 7-slide deck, public documents |

The model tests build a self-contained synthetic cube with a known 2024 extreme and a known relay structure. They run the real modules as subprocesses, so a broken import or CLI contract fails the test rather than being masked by an in-process mock.

The web-app tests use a complete synthetic Parquet fixture and exercise both the WSGI application through `httpx.WSGITransport` and a real threaded socket server. They check the redesigned date-range selector, the single light theme, and the absence of a theme switcher.

---

## 7. Reproduction commands

```bash
# 1. Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Large processed artefacts
python scripts/fetch_processed_data.py

# 3. Tests
python -m unittest discover -s tests

# 4. Web application
python -m app.server --processed-dir data/processed --host 127.0.0.1 --port 8080

# 5. Full rebuild from raw FIRMS archives (requires FIRMS_MAP_KEY)
export FIRMS_MAP_KEY="<your key>"
python scripts/download_firms_sa.py
python scripts/build_canonical_parquet.py
python -m src.data.build_spatiotemporal_cubes --project-root .
python -m src.models.freeze_validation --processed-dir data/processed
python -m src.models.harmonize_pipeline --processed-dir data/processed
python -m src.models.calendar_analytics --processed-dir data/processed
python scripts/export_submission.py --processed-dir data/processed --output submission.csv
```

---

## 8. Scientific limitations

- FIRMS detections are thermal anomalies at overpass time, not burned area. Cloud, smoke, and saturation can hide or distort the signal.
- The overpass event count is a sampling proxy. It is more stable than raw detections but still not a physical fire count.
- The 10 km and 50 km cells are analysis units. Admin-1 assignment near a border carries an epistemic uncertainty of about half a cell width.
- The M3 90% interval is a predictive band from the ensemble and fold-wise sigma calibration, not a full Bayesian posterior.
- The trend test controls FDR but does not establish causality.
- GEV return levels rest on a short annual record and are sensitive to the shape parameter.
- The 2026 record is partial through November and is labelled NRT/partial.
- The pre-2012 record is reconstructed by the bridge and carries larger uncertainty than the 2012-2020 overlap.
- The SAMA suspect flag never activates after the nominal+ confidence filter; it removes no rows. The `sama_suspect_det_count` column is a night-detection count. SAMA masking does not recover the true fire signal in the anomaly region.

These limits are also exposed in the API metadata, the web application, and the submission package. The project does not claim more precision than the underlying sensors provide.

---

## Scientific References

The methods above are based on, or informed by, the following publications.
Citations support scientific attribution and reproducibility; the project does
not reproduce the copyrighted text of these works.

1. Giglio, L., Schroeder, W., & Justice, C. O. (2016). *The collection 6 MODIS active fire detection algorithm and fire products*. Remote Sensing of Environment, 178, 31–41. <https://doi.org/10.1016/j.rse.2016.02.054>

2. Schroeder, W., Oliva, P., Giglio, L., & Csiszar, I. A. (2014). *The New VIIRS 375m active fire detection data product: Algorithm description and initial assessment*. Remote Sensing of Environment, 143, 85–96. <https://doi.org/10.1016/j.rse.2013.12.008>

3. Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). *Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization*. Journal of the American Statistical Association, 114(526), 804–819. <https://doi.org/10.1080/01621459.2018.1448825>

4. Wickramasuriya, S. L., Turlach, B. A., & Hyndman, R. J. (2020). *Optimal non-negative forecast reconciliation*. Statistics and Computing, 30(5), 1167–1182. <https://doi.org/10.1007/s11222-020-09930-0>

5. Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems*. Journal of Basic Engineering, 82(1), 35–45. <https://doi.org/10.1115/1.3662552>

6. Hamed, K. H., & Rao, A. R. (1998). *A modified Mann-Kendall trend test for autocorrelated data*. Journal of Hydrology, 204(1–4), 182–196. <https://doi.org/10.1016/S0022-1694(97)00125-X>

7. Sen, P. K. (1968). *Estimates of the Regression Coefficient Based on Kendall's Tau*. Journal of the American Statistical Association, 63(324), 1379–1389. <https://doi.org/10.1080/01621459.1968.10480934>

8. Theil, H. (1950). *A rank-invariant method of linear and polynomial regression analysis*. Indagationes Mathematicae, 12, 85–91. (No DOI was assigned to the original 1950 record; cited as published.)

9. Benjamini, Y., & Hochberg, Y. (1995). *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*. Journal of the Royal Statistical Society: Series B, 57(1), 289–300. <https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>

10. Hosking, J. R. M. (1990). *L-Moments: Analysis and Estimation of Distributions Using Linear Combinations of Order Statistics*. Journal of the Royal Statistical Society: Series B, 52(1), 105–124. <https://doi.org/10.1111/j.2517-6161.1990.tb01775.x>

11. Roberts, D. R., et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*. Ecography, 40(8), 913–929. <https://doi.org/10.1111/ecog.02881>

12. Pinto, O., et al. (2003). *South Atlantic Magnetic Anomaly: energetic particle precipitation and satellite detector effects*. Journal of Atmospheric and Solar-Terrestrial Physics, 65(16–18), 1393–1404. (No DOI was located for this record; cited as published.)
