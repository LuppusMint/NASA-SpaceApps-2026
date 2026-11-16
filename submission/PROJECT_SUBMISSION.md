# Fire Relay South America

## Cross-Sensor Burning Calendar Across 26 Years of NASA FIRMS Observations

**NASA Space Apps Challenge 2026 — Harmonization of MODIS and VIIRS Hot Spots**

**Team:** KernelCrew · **Authors:** Manuel Guerrero & Jessica Garay · **Repository:** `https://github.com/LuppusMint/NASA-SpaceApps-2026` (`nasa-spaceapps-2026-modis-viirs-harmonization`)

> **Data attribution (text only):** NASA FIRMS Archive & NRT — MODIS C6.1, VIIRS S-NPP, NOAA-20/J1, NOAA-21/J2.
> No NASA logo, emblem, seal, or wordmark is used anywhere in this submission.

---

## 1. Project Title & Elevator Pitch

**Fire Relay South America** turns 26 years of overlapping satellite fire detections into one comparable, leakage-free burning calendar. The project harmonizes **85,002,288** raw NASA FIRMS detections (**2000-11** to **2026-11**, **313** months) into **75,556,207** quality-controlled detections, then learns the sensor-to-sensor relay from a frozen **2012–2020** overlap instead of pretending that five instruments are one instrument.

The result is a single monthly record for **14** South American countries and **243** Admin-1 units, served by an offline bilingual **EN/ES** web application. Emergency responders can compare the current month with a frozen **2001–2020** climatology, scientists can inspect FDR-controlled Theil–Sen trends and 90-day early-onset signals, and land managers can export harmonized analytics in one click. Everything runs on a local CPU with DuckDB and Parquet; there is no Google Cloud, no external tile server, and no runtime CDN.

**One sentence:** *Fire Relay South America makes the satellite identity stop mattering, so the fire signal can be compared across decades.*

---

## 2. Challenge Alignment & Problem Statement

### 2.1 The sensor relay problem

The NASA FIRMS record is not one homogeneous time series. It is a relay of instruments with different footprints, sensitivities, overpass schedules, and quality flags:

- **MODIS Terra** (1 km nadir) and **MODIS Aqua** (1 km nadir) provide the long historical backbone, with Terra and Aqua local overpasses near 10:30 and 13:30.
- **VIIRS S-NPP**, **NOAA-20/J1**, and **NOAA-21/J2** provide a 375 m nadir footprint and additional overpasses, including cooler and nocturnal fires that MODIS may miss.
- The constellation expands over time: Terra-only, Terra+Aqua, the calibrated MODIS/VIIRS overlap, and the deduplicated VIIRS backbone.
- Terra/Aqua orbital drift after 2021 changes swath coverage and introduces sampling gaps.
- The **South Atlantic Magnetic Anomaly (SAMA)** can produce energetic-particle false alarms, especially in nocturnal VIIRS low-confidence data.
- Static gas flares, industrial sources, and volcanoes contaminate the record unless explicitly removed.

A naive sum across sensors therefore turns instrument additions into artificial trends. The biggest distortions sit at the relay boundaries: **2002**, **2012**, **2018**, **2021**, and **2023**. The project treats those boundaries as a scientific calibration problem, not as a plotting detail.

### 2.2 Why a constant multiplier is not enough

A universal MODIS-to-VIIRS multiplier fails because the ratio varies by biome, fuel load, cloud cover, diurnal cycle, and detection confidence. It also fails differently for **detection counts** (pixel fragmentation) and **fire radiative power** (integrated energy). Fire Relay therefore uses a physical bridge, a covariate-shift model, and a hierarchical reconciliation instead of a single scale factor.

### 2.3 What the submission delivers

1. A deterministic quality-control and spatiotemporal cube build.
2. A leakage-free overlap calibration and **M0–M3** reconciliation benchmark.
3. A calendar-analytics layer for seasonal calendars, FDR trends, extremes, and 90-day early onset.
4. An offline bilingual web application with DuckDB-backed queries and CSV export.
5. This English submission package, a 7-slide deck, a reproducible README, and a validated `submission.csv` contract.

---

## 3. Architecture & Pipeline Summary

```text
NASA FIRMS Archive & NRT
        │
        ▼
QC  Deterministic quality control + Albers 10 km cubes
    type 1/2/3 removal · static-cell mask · FRP/brightness repair
    SAMA flags · overpass consolidation · 10/50 km + Admin-1 + country cubes
        │
        ▼
CAL Leakage-free overlap calibration (2012–2020)
    5 frozen spatial folds · M0 raw · M1 physical bridge
    M2 hurdle/Tweedie with covariate-shift weights
    M3 simplex + Kalman + hierarchical reconciliation
        │
        ├──► Calendar analytics
        │    frozen 2001–2020 climatology · seasonal severity
        │    Theil–Sen + Hamed–Rao + FDR · GEV return periods
        │    90-day early-onset curves · 2024–2026 critical months
        │
        ▼
WEB Offline bilingual EN/ES web application
    WSGI server · thread-local DuckDB · /static assets · 1-click CSV
        │
        ▼
SUB Submission package
    submission.csv · submission_spec.json · 7-slide HTML/PDF · README
```

### 3.1 Deterministic quality control and cubes

- Pair MODIS confidence **≥ 30%** with VIIRS **nominal/high** confidence; retain a high-severity track at MODIS **≥ 75%** / VIIRS **high**.
- Remove `type IN (1,2,3)` where the source provides a type flag.
- Mask **358** persistent static 10 km thermal cells, identified by chronic activity (at least 9 active months per year over at least 5 years).
- Repair **138** negative FRP values and **39** zero-brightness rows.
- Flag SAMA containment and consolidate multiple detections into overpass fire events.
- Aggregate to equal-area Albers cells at **10 km** and **50 km**, plus Admin-1, country, and South America monthly cubes.
- Preserve exact sums through the hierarchy.

### 3.2 Leakage-free calibration and reconciliation

- Freeze **5** spatial block folds in the committed split manifest.
- Calibrate only on the pre-drift overlap **2012–2020**: **930,100** rows = **465,050** cell-months x 2 counterfactual replicas across **7,184** unique 50 km cells.
- Benchmark four reconstructions on the same folds:
  - **M0** raw, unharmonized;
  - **M1** physical biweight bridge;
  - **M2** hurdle/Tweedie model with covariate-shift weights;
  - **M3** simplex ensemble + Kalman smoother + non-negative hierarchical reconciliation.
- Enforce exact conservation from Admin-1 to country to South America.
- Report a cross-regime guard between **TERRA_AQUA** and **TERRA_ONLY** regimes and a step-change continuity diagnostic at every relay boundary.

### 3.3 Calendar analytics: trends, extremes, early onset

- Freeze the climatology to **2001–2020** so the reference does not absorb the post-2020 regime change.
- Compute monthly severity tiers against frozen percentiles and z-scores.
- Estimate trends with **Theil–Sen** and **Hamed–Rao modified Mann–Kendall**, then control the false discovery rate with Benjamini–Hochberg **q < 0.05**.
- Fit **GEV** return periods on the annual series.
- Build **90-day early-onset** curves for operational warning.

### 3.4 Offline bilingual web application

- `app/server.py` is a WSGI application; static assets live in `app/static/`.
- DuckDB opens the Parquet cubes with `read_parquet`; connections are thread-local and project only the columns needed.
- The UI supports country, Admin-1, bounding box, point-radius, date range, metric, and harmonized/raw series selection.
- The interface is available in **English** (default) and **Spanish**.
- CSV export is one click and UTF-8.
- No external CDN, map tile, font, or cloud service is required.

---

## 4. Verified Quantitative Results

All values below are taken from the deterministic pipeline artefacts and the frozen benchmark records. They are reported as measured; they are not rounded for presentation except where the decimal places shown are the published precision.

### 4.1 Dataset coverage and quality-control ledger

| Quantity | Verified value |
|---|---:|
| Temporal coverage | **2000-11** to **2026-11** |
| Months | **313** |
| Raw FIRMS detections, South America | **85,002,288** |
| Clean detections | **75,556,207** |
| Excluded by `type IN (1,2,3)` | **2,703,746** |
| Excluded as persistent static-cell detections | **2,872,511** |
| Persistent static 10 km cells | **358** |
| Clean candidate detections in static cells | **472,710** |
| Negative FRP rows corrected | **138** |
| Zero-brightness rows corrected | **39** |
| Rows intersecting SAMA | **76,388,628** |
| Suspect nocturnal detections remaining in clean data | **0** |
| South American countries/territories | **14** |
| Admin-1 units | **243** |
| Historical active 10 km cells | **143,613** |

### 4.2 Clean and raw counts by sensor

| Sensor slot | Clean detections | Raw detections |
|---|---:|---:|
| `VIIRS_SNPP` | **31,965,171** | **36,304,172** |
| `VIIRS_N20` | **19,370,162** | **22,017,351** |
| `MODIS_Aqua` | **10,162,384** | **10,897,858** |
| `VIIRS_N21` | **7,965,108** | **9,038,172** |
| `MODIS_Terra` | **6,093,382** | **6,744,735** |

### 4.3 Frozen validation design

| Item | Value |
|---|---|
| Folds | **5** spatial block folds, no leakage |
| Calibration overlap | **2012–2020** |
| Overlap rows | **930,100** = **465,050** cell-months x 2 replicas |
| Unique 50 km cells | **7,184** |
| Country-month rows | **4,014** (absence = no detections, not missing data) |
| Dataset SHA-256 | `ff927d7b27a7124d44cfcb776bb26ae26b13f24cc7c4ed1b786feb3db02ee0d6` |

### 4.4 Out-of-fold benchmark, 5-fold mean

**Target `target_srehi`**

| Model | RMSE | R² | Other published metrics |
|---|---:|---:|---|
| `M0_raw_unharmonized` | **24.6227** | **0.3533** | — |
| `M1_physical_biweight_bridge` | **26.7017** | **0.2374** | RMSLE **0.8757** |
| `M2_hurdle_tweedie_covshift` | **11.8025** | **0.8512** | — |
| `M3_simplex_kalman_reconciled` | **12.8838** | **0.8171** | MAE **6.1743** |

**Target `target_frp_mw`**

| Model | RMSE | R² | Other published metrics |
|---|---:|---:|---|
| `M0_raw_unharmonized` | **1377.5885** | **0.3973** | — |
| `M1_physical_biweight_bridge` | **1385.0089** | **0.3928** | — |
| `M2_hurdle_tweedie_covshift` | **790.2797** | **0.8018** | — |
| `M3_simplex_kalman_reconciled` | **833.9594** | **0.7779** | MAE **208.7884** |

**Reading the benchmark honestly.** M2 has the lowest point error on both targets. The fitted M3 simplex weights are 0.0 for M1 and 1.0 for M2, so out of fold M3 equals M2 on four of the five folds; its higher mean RMSE comes entirely from the first fold, which the causal blend scores with uniform warm-up weights by construction. The M1/M2 simplex weight is 1.0 on M2. In the MODIS-only backcast the production point is the M2 bridge; in overlap cells with VIIRS it blends the bridge with the observation. The hierarchy is reconciled non-negatively; Admin-1 is a proportional downscaling of the country total, not a bottom-up sum of cells. Both are reported; neither is hidden. A temporal hold-out re-run (fit on 2016–2020, predict 2012–2015) gives Terra-only SREHI R² 0.808 and FRP R² 0.748 (Terra+Aqua 0.920/0.890); the overlap is 930,100 rows = 465,050 cell-months × 2 replicas. Full tables are in [docs/validation/TEMPORAL_HOLDOUT_VALIDATION.md](docs/validation/TEMPORAL_HOLDOUT_VALIDATION.md). The operational calendar uses M3 because a country total that does not equal the sum of its Admin-1 units is not a usable public record.

### 4.5 Cross-regime guard

The cross-regime guard compares **TERRA_AQUA** and **TERRA_ONLY** regimes. Every check reports `passed: True`.

| M3 metric | Regime RMSE ratio | Pairwise RMSE values |
|---|---:|---|
| SREHI | **1.2572** | **11.5281** vs **14.4926** |
| FRP (MW) | **1.3718** | **699.5017** vs **959.5740** |

The guard is a bias check, not a claim that the two regimes are physically identical. It verifies that the harmonized reconstruction does not collapse or explode when the sensor mix changes.

### 4.6 Constellation step-change continuity (SREHI)

The reduction is computed in log space as `1 - |log(harmonized ratio)| / |log(raw ratio)|`. A value near 100% means the relay step almost disappears.

| Relay year | Raw step ratio | Harmonized step ratio | Reduction | Guard |
|---:|---:|---:|---:|---|
| **2002** | **3.414** | **1.523** | **65.7%** | `passed: true` |
| **2012** | **3.938** | **1.177** | **88.1%** | `passed: true` |
| **2018** | **1.407** | **0.836** | **47.7%** | `passed: true` |
| **2021** | **0.738** | **0.754** | **7.2%** | `passed: true` |
| **2023** | **1.375** | **1.000** | **99.9%** | `passed: true` |

### 4.7 Exact hierarchical conservation (M3)

| Level transition | Max SREHI difference | Max FRP difference (MW) |
|---|---:|---:|
| Admin-1 → Country | **4.37e-11** | **3.73e-09** |
| Country → South America | **5.82e-11** | **1.86e-09** |

The minimum reconciled value is **>= 0.0**. No negative fire activity is emitted.

### 4.8 FDR-controlled trends

- **45 of 242** Admin-1 units are significant in SREHI at **q < 0.05**: **34 increasing** and **11 decreasing**.
- One country is significant in SREHI: **Peru**, slope **+903.23/yr**, **z = 3.34**, **q = 0.0134**.
- Leading increasing Admin-1 units:
  - **Amazonas (Brazil)**: **+785.81/yr**, **q = 0.0125**.
  - **Ucayali (Peru)**: **+186.55/yr**, **q = 0.0028**.
  - **Loreto (Peru)**: **+133.58/yr**, **q = 0.0028**.

### 4.9 Critical extremes and continental context

**Critical extreme months, 2024–2026:**

| Month | Geography | z |
|---|---|---:|
| **2024-02** | **Suriname** | **8.28** |
| **2025-12** | **Peru** | **6.51** |
| **2024-02** | **Guyana** | **5.69** |
| **2024-03** | **Ecuador** | **5.05** |
| **2024-02** | **Bolivia** | **4.95** |

**Historical continental SREHI peak:** **2007-09 = 293,471.7**, compared with **2004-09 = 262,575.2** and **2010-09 = 239,710.3**.

### 4.10 90-day early onset

From `calendar_early_onset_90d.parquet`:

| Year | `EARLY_ONSET_CRITICAL` | `EARLY_ONSET_HIGH` |
|---:|---:|---:|
| **2024** | **5,125** | **3,009** |
| **2025** | **2,891** | **2,448** |
| **2026** | **3,232** | **2,176** |

The 2026 record is a partial year through November and is labelled accordingly by the data-maturity logic.

### 4.11 Web application verification

| Item | Verified result |
|---|---|
| Backend | **WSGI** (`app/server.py`) with offline **DuckDB** |
| Warm query latency | **< 1 ms** on real cubes |
| Cold query latency | **9.8–135.9 ms** on real cubes |
| CSV export | **1 click**, UTF-8 |
| Web-app baseline unit tests | **40/40 passing** |
| External CDNs / cloud at runtime | **0** |

---

## 5. Data Provenance & Attribution

**Source:** NASA FIRMS Archive & NRT — MODIS C6.1, VIIRS S-NPP, NOAA-20/J1, NOAA-21/J2.

The project uses openly available NASA FIRMS active-fire detections for South America. It does not redistribute the raw archive as part of this repository; the submitted CSV is a derived, harmonized monthly country product. Source-specific quality flags, confidence classes, and type flags are preserved or explicitly documented before they are transformed.

**Attribution statement (text only):**

> NASA FIRMS Archive & NRT — MODIS C6.1, VIIRS S-NPP, NOAA-20/J1, NOAA-21/J2.

**Logo policy:** this submission uses **zero NASA logos, badges, seals, emblems, or graphical wordmarks** in HTML, SVG, CSS, Markdown, or PDF. The attribution above is textual and does not imply NASA endorsement. NASA does not endorse this independent challenge submission.

### 5.1 License, Data Citation & Operational Disclaimer

**License.** The repository uses the multi-part terms in `LICENSE`: MIT for
source code (`app/`, `src/`, `scripts/`, `tests/`, `Dockerfile`), CC-BY-4.0 for
documentation and presentation assets (`README.md`, `docs/`, `submission/`),
and CC0 1.0 Universal for derived data products (`submission.csv`,
`data/processed/*`). Third-party data, software, and scientific references are
recorded in `THIRD_PARTY_NOTICES.md`.

**NASA LANCE / FIRMS acknowledgment.** We acknowledge the use of data and/or
imagery from NASA's Land, Atmosphere Near real-time Capability for EOSDIS
(LANCE) system ([Earthdata near-real-time](https://earthdata.nasa.gov/earth-observation-data/near-real-time)),
part of NASA's Earth Science Data and Information System (ESDIS), and the Fire
Information for Resource Management System (FIRMS;
<https://firms.modaps.eosdis.nasa.gov/>).

**Dataset DOIs.** MODIS Collection 6.1: `10.5067/FIRMS/MODIS/MCD14ML` and
`10.5067/FIRMS/MODIS/MCD14DL.NRT.0061`. VIIRS 375 m: S-NPP
`10.5067/FIRMS/VIIRS/VNP14IMGT_NRT.002`, NOAA-20
`10.5067/FIRMS/VIIRS/VJ114IMGT_NRT.002`, and NOAA-21
`10.5067/FIRMS/VIIRS/VJ214IMGT_NRT.002`. These products are governed by the
NASA Open Data Policy (CC0 / 17 U.S.C. § 105). Boundaries are derived from
Natural Earth 1:10m Cultural Vectors v5.1.2, public domain (CC0).

**Non-Endorsement & Not-for-Tactical-Fire-Management.** Redistribution and
analysis of derived NASA FIRMS data do not imply sponsorship, endorsement, or
official approval by NASA, LANCE, ESDIS, NOAA, or the U.S. Government. All
data, models, and visualizations are provided "AS IS" without warranty of any
kind, strictly for scientific, retrospective, and climatological research.
Because of orbital revisit intervals, cloud cover, and spatial aggregation on
10 km and 50 km grids, this system must not be used for tactical wildfire
suppression, real-time emergency dispatch, evacuation routing, or
life-and-property safety decisions.

**Key academic references.** Giglio, L., Schroeder, W., & Justice, C. O.
(2016). *The collection 6 MODIS active fire detection algorithm and fire
products*. Remote Sensing of Environment, 178, 31–41.
<https://doi.org/10.1016/j.rse.2016.02.054>. Schroeder, W., Oliva, P., Giglio,
L., & Csiszar, I. A. (2014). *The New VIIRS 375m active fire detection data
product*. Remote Sensing of Environment, 143, 85–96.
<https://doi.org/10.1016/j.rse.2013.12.008>. Wickramasuriya, S. L.,
Athanasopoulos, G., & Hyndman, R. J. (2019). *Optimal Forecast Reconciliation
for Hierarchical and Grouped Time Series Through Trace Minimization*. Journal
of the American Statistical Association, 114(526), 804–819.
<https://doi.org/10.1080/01621459.2018.1448825>. Hamed, K. H., & Rao, A. R.
(1998). *A modified Mann-Kendall trend test for autocorrelated data*. Journal
of Hydrology, 204(1–4), 182–196.
<https://doi.org/10.1016/S0022-1694(97)00125-X>. Kalman, R. E. (1960). *A New
Approach to Linear Filtering and Prediction Problems*. Journal of Basic
Engineering, 82(1), 35–45. <https://doi.org/10.1115/1.3662552>. Benjamini, Y.,
& Hochberg, Y. (1995). *Controlling the False Discovery Rate*. Journal of the
Royal Statistical Society: Series B, 57(1), 289–300.
<https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>. Roberts, D. R., et al.
(2017). *Cross-validation strategies for data with temporal, spatial,
hierarchical, or phylogenetic structure*. Ecography, 40(8), 913–929.
<https://doi.org/10.1111/ecog.02881>.

---

## 6. Reproducibility & Local Execution Commands

All commands run from the repository root with the Python interpreter from your local virtual environment (`python`). The reference runtime is CPU-only.

### 6.1 Fetch the processed cubes

```bash
python scripts/fetch_processed_data.py
```

### 6.2 Build the canonical and quality-controlled cubes

```bash
python scripts/build_canonical_parquet.py
python -m src.data.build_spatiotemporal_cubes --project-root .
```

The quality-control build writes the 10 km, 50 km, Admin-1, country, and South America Parquet cubes under `data/processed/`, plus `qc_cleaning_summary.json` and `geography_catalog.json`.

The data implementation lives in `src/data/`; the calibration, reconciliation, analytics, and web application live in `src/models/` and `app/`. The contract tests live in `tests/`, including `tests/test_submission_package.py`.

### 6.3 Freeze splits and run the benchmark

```bash
python -m src.models.freeze_validation --processed-dir data/processed

python -m src.models.harmonize_pipeline --processed-dir data/processed
```

### 6.4 Build the calendar analytics

```bash
python -m src.models.calendar_analytics --processed-dir data/processed
```

### 6.5 Run the web application

```bash
python -m app.server --processed-dir data/processed --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/` locally. The server serves only local static assets and local Parquet data.

### 6.6 Export and validate the submission CSV

```bash
python scripts/export_submission.py --processed-dir data/processed --output submission.csv

python -m unittest tests/test_submission_package.py -v
```

### 6.7 Resource envelope

- CPU-only; no NVIDIA GPU and no CUDA requirement.
- DuckDB with a bounded memory limit and 4 threads for the reference run.
- Out-of-core Parquet processing; no full 85M-row table is materialized in pandas.
- Offline-capable architecture with optional Google Cloud Run container hosting (`https://fire-relay-south-america-946659834930.us-central1.run.app`), zero external map tile APIs, and zero runtime CDN calls.
- Deterministic seeds and frozen folds for the benchmark.

---

## 7. Submission Artifacts & Validation

| Artifact | Purpose |
|---|---|
| `submission.csv` | 4,014-row public CSV, sorted by `iso_a3, year_month` |
| `submission_spec.json` | Schema, row-count, ID, non-negative, and no-NaN validation rules |
| `scripts/export_submission.py` | Deterministic DuckDB export |
| `submission/PROJECT_SUBMISSION.md` | Full project submission overview |
| `docs/Fire_Relay_South_America_Presentation.pdf` | 7-slide project presentation PDF (source in `docs/presentation.html`) |
| `README.md` | Root project and reproducibility guide |
| `tests/test_submission_package.py` | Automated tests for CSV export, presentation, and documentation |

The CSV schema is exactly:

```text
record_id, iso_a3, country_name, year_month, year, month,
harmonized_srehi, harmonized_frp_mw, raw_srehi, raw_frp_mw,
clean_det_count, raw_all_det_count, data_maturity, constellation_regime
```

`record_id` is `f"{iso_a3}_{year_month}"`. `raw_srehi` is the source `overpass_event_count`. The contract enforces **`row_count = 4014`**, **`exact_columns = true`**, **`id_column = "record_id"`**, **`min = 0.0`**, and **`allow_nan = false`**.

---

## 8. Synthetic Intelligence (SI) / AI Tools Disclosure

**Scientific Authorship (Team KernelCrew — Manuel Guerrero & Jessica Garay).** Research formulation, multi-sensor harmonization (`Hierarchical NB-GAMM M3`), SAMA segmentation, early-onset seasonal curves, and cartographic design are the original intellectual work of Manuel Guerrero and Jessica Garay (Team KernelCrew).

**Technical Assistance.** Synthetic Intelligence (SI) and AI software engineering tools were utilized strictly as productivity accelerators for code implementation, automated test suites, and bilingual interface formatting under direct human supervision.

**Accountability & Verification.** All physical constants, statistical models, and spatial aggregations were independently verified by the team across all `85,002,288` NASA FIRMS archival records (`1,453,803,616.63 MW` exact radiative power conservation).

**Verification record.** The delivery is checked with the deterministic `scripts/export_submission.py` export, the `tests/test_submission_package.py` suite, hierarchy checks (country = sum of Admin-1; Admin-1 downscaling), a brand and external-URL check, and human review of the scientific framing, code, and documentation by Team KernelCrew.

> **Disclaimer & Licensing:** NASA LANCE / ESDIS / FIRMS data are used with textual attribution and without NASA endorsement. The product is intended solely for scientific and climatological analysis, not tactical wildfire suppression. Source code is MIT-licensed (`LICENSE`), documentation CC-BY-4.0, and derived data CC0 1.0 (`THIRD_PARTY_NOTICES.md`).
