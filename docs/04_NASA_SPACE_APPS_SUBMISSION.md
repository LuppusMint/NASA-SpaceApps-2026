# NASA Space Apps Submission Package

**Fire Relay South America - harmonized burning calendar and early-onset observatory from 26 years of NASA FIRMS observations.**

This is the clean English copy for the NASA Space Apps Challenge 2026 portal. It uses only the verified numbers in this repository and contains no internal project jargon, local paths, or third-party tooling references.

---

## Project Title

**Fire Relay South America: Cross-Sensor Harmonization of MODIS and VIIRS Hot Spots (2000-2026)**

---

## Elevator Pitch

Five satellite instruments do not make one instrument. Fire Relay South America harmonizes **85,002,288** raw NASA FIRMS detections from MODIS C6.1 and VIIRS S-NPP / NOAA-20 / NOAA-21 into **75,556,207** quality-controlled detections, then learns the sensor relay on a frozen **2012-2020** overlap instead of summing incompatible records. The result is one comparable monthly burning calendar for **14** South American countries and **243** Admin-1 units, served by an offline bilingual web application. Emergency responders can compare the current month with a frozen **2001-2020** climatology, scientists can inspect FDR-controlled Theil-Sen trends and 90-day early-onset signals, and land managers can export harmonized analytics in one click. Everything runs on a local CPU with DuckDB and Parquet; there is no Google Cloud dependency, no external tile server, and no runtime CDN.

**One sentence:** *Fire Relay South America makes the satellite identity stop mattering, so the fire signal can be compared across decades.*

---

## Challenge Alignment

The challenge asks for a harmonized view of MODIS and VIIRS hot spots. The hard part is not drawing a map; it is preventing the constellation change from becoming a fake trend.

- **MODIS Terra/Aqua** provide the long historical backbone at a 1 km nadir footprint.
- **VIIRS S-NPP, NOAA-20/J1, and NOAA-21/J2** provide a 375 m footprint and additional overpasses, including cooler and nocturnal fires.
- The constellation expands through Terra-only, Terra+Aqua, the calibrated MODIS/VIIRS overlap, and the deduplicated VIIRS backbone.
- Terra/Aqua orbital drift after 2021 changes swath coverage (NASA LDOPE note **HC_MOD_26008**).
- The South Atlantic Magnetic Anomaly (**SAMA, 7°N–55°S, 110°W–11°E**) can produce energetic-particle false alarms in nocturnal VIIRS data.
- Static gas flares, industrial sources, and volcanoes contaminate the record unless removed.

A naive sum turns instrument additions into artificial trends at the relay boundaries **2002, 2012, 2018, 2021, and 2023**. Fire Relay treats those boundaries as a calibration problem: it trains only on the pre-drift overlap, freezes spatial folds, and reconciles the hierarchy exactly.

---

## Architecture

```text
NASA FIRMS Archive & NRT
  MODIS C6.1 · VIIRS S-NPP · NOAA-20/J1 · NOAA-21/J2
          |
          v
Quality control and equal-area cubes
  type 1/2/3 removal · static-cell mask · FRP/brightness repair
  SAMA flags · overpass consolidation
  Albers Equal-Area EPSG:102033 · 10 km and 50 km cells
          |
          v
Leakage-free calibration (2012-02..2020-12)
  200 km spatial blocks · 5 folds · temporal splits
  M0 raw · M1 physical biweight bridge
  M2 hurdle Tweedie + covariate-shift weights
  M3 simplex + 1D Kalman relay + non-negative hierarchical reconciliation
          |
          +--> Calendar analytics
          |    frozen 2001-2020 climatology
          |    severity tiers · Theil-Sen + Hamed-Rao + BH FDR
          |    GEV return levels · 90-day early onset
          |
          v
Offline bilingual EN/ES web application
  WSGI · thread-local DuckDB · local static assets · 1-click CSV
```

The scientific code is self-contained under `src/core/` (`geogrid`, `splits_st`, `harmonize`, `postprocess`, `stacking`, `trends`, `extremes`, `metrics`, `duckdb_helper`). The data pipeline lives in `src/data/`, the models in `src/models/`, and the application in `app/`.

---

## Verified Quantitative Results

### Dataset and quality control

| Quantity | Verified value |
|---|---:|
| Raw FIRMS detections, South America | **85,002,288** |
| Clean detections | **75,556,207** |
| Temporal coverage | **2000-11** to **2026-11** |
| Months | **313** |
| Countries and territories | **14** |
| Admin-1 units | **243** |
| Historical active 10 km cells | **143,613** |
| Persistent static 10 km cells masked | **358** |
| Integrated fire radiative power | **1,453,803,616.63 MW** |

### Frozen validation

| Item | Value |
|---|---:|
| Spatial blocks | **200 km** |
| Folds | **5** |
| Calibration overlap | **2012-02..2020-12** |
| Overlap rows | **930,100** = **465,050** cell-months x 2 replicas |
| Unique 50 km cells | **7,184** |
| Country-month rows | **4,014** (absence = no detections, not missing data) |
| Spatial leakage | **zero** |
| Dataset SHA-256 | `ff927d7b27a7124d44cfcb776bb26ae26b13f24cc7c4ed1b786feb3db02ee0d6` |

### Out-of-fold benchmark, 5-fold mean

| Target | Model | RMSE | R-squared |
|---|---|---:|---:|
| SREHI | M0 raw unharmonized | 24.6227 | 0.3533 |
| SREHI | M1 physical biweight bridge | 26.7017 | 0.2374 |
| SREHI | M2 hurdle Tweedie + covariate shift | 11.8025 | 0.8512 |
| SREHI | **M3 simplex + Kalman + reconciliation** | **12.8838** | **0.8171** |
| FRP (MW) | M0 raw unharmonized | 1377.5885 | 0.3973 |
| FRP (MW) | M1 physical biweight bridge | 1385.0089 | 0.3928 |
| FRP (MW) | M2 hurdle Tweedie + covariate shift | 790.2797 | 0.8018 |
| FRP (MW) | **M3 simplex + Kalman + reconciliation** | **833.9594** | **0.7779** |

M2 has the lowest point error. The fitted M3 simplex weights are 0.0 for M1 and 1.0 for M2, so out of fold M3 equals M2 on four of the five folds; its higher mean RMSE comes entirely from the first, uniform warm-up fold of the causal blend. The M1/M2 simplex weight is 1.0 on M2. In the MODIS-only backcast the production point is the M2 bridge; in overlap cells with VIIRS it blends the bridge with the observation. The hierarchy is reconciled non-negatively; Admin-1 is a proportional downscaling of the country total. A temporal hold-out re-run (fit on 2016–2020, predict 2012–2015) gives Terra-only SREHI R² 0.808 and FRP R² 0.748 (Terra+Aqua 0.920/0.890); the overlap is 930,100 rows = 465,050 cell-months × 2 replicas. The operational calendar uses M3 because a country total must equal the sum of its Admin-1 units.

### Relay continuity

The aggregate absolute step discontinuity `|delta_step|` falls **92.84%**, from **2.1892** to **0.1567**. Per-transition SREHI reductions: **65.7%** (2002), **88.1%** (2012), **47.7%** (2018), **7.2%** (2021), and **99.9%** (2023). The 2021 value is weakest because it reflects physical orbital drift rather than a sensor addition.

### Conservation

| Transition | Max SREHI difference | Max FRP difference (MW) |
|---|---:|---:|
| Admin-1 -> Country | **4.37e-11** | **3.73e-09** |
| Country -> South America | **5.82e-11** | **1.86e-09** |

The minimum reconciled value is **>= 0.0**.

### Calendar science

- **45 of 242** Admin-1 units are significant in SREHI at q < 0.05: **34 increasing**, **11 decreasing**.
- **Peru** is the only significant country: slope **+903.23/yr**, z **3.34**, q **0.0134**.
- Leading increasing Admin-1 units: **Amazonas (Brazil) +785.81/yr** (q **0.0125**), **Ucayali (Peru) +186.55/yr** (q **0.0028**), **Loreto (Peru) +133.58/yr** (q **0.0028**).
- Critical extreme months, 2024-2026: **2024-02 Suriname** (z **8.28**), **2025-12 Peru** (z **6.51**), **2024-02 Guyana** (z **5.69**), **2024-03 Ecuador** (z **5.05**), **2024-02 Bolivia** (z **4.95**).
- 90-day early onset: 2024 **5,125** critical / **3,009** high; 2025 **2,891** / **2,448**; 2026 **3,232** / **2,176** (through November). The window is a true trailing 90 days across year boundaries; the first 89 days are `INSUFFICIENT_HISTORY`.

### Application

- Offline WSGI backend over DuckDB.
- Warm query latency **< 1 ms**; cold query latency **9.8-135.9 ms** on real cubes.
- One-click UTF-8 CSV export.
- Bilingual English/Spanish interface, light-only theme, no external CDN.

---

## Data Provenance

**Source:** NASA FIRMS Archive & NRT - MODIS C6.1, VIIRS S-NPP, NOAA-20/J1, NOAA-21/J2.

The raw archive is not redistributed. The submitted CSV is a derived, harmonized monthly country product. Source-specific quality flags, confidence classes, and type flags are preserved or explicitly documented before transformation. `data/PROVENANCE.json` and `data/DATA_SUMMARY.json` record the portable provenance.

**Attribution statement (text only):**

> NASA FIRMS Archive & NRT - MODIS C6.1, VIIRS S-NPP, NOAA-20/J1, NOAA-21/J2.

**Logo policy:** zero NASA logos, badges, seals, emblems, or graphical wordmarks in HTML, SVG, CSS, Markdown, or PDF. The attribution is textual and does not imply NASA endorsement. NASA does not endorse this independent challenge submission.

---

## Reproducibility

All commands run from the project root.

```bash
# 1. Install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Fetch the five large processed artefacts and verify SHA-256
python scripts/fetch_processed_data.py

# 3. Run the full test suite
python -m unittest discover -s tests

# 4. Run the web application
python -m app.server --processed-dir data/processed --host 127.0.0.1 --port 8080

# 5. Export the submission CSV
python scripts/export_submission.py --processed-dir data/processed --output submission.csv
```

Full pipeline rebuild from raw archives (requires `FIRMS_MAP_KEY` in the environment):

```bash
python scripts/download_firms_sa.py
python scripts/build_canonical_parquet.py
python -m src.data.build_spatiotemporal_cubes --project-root .
python -m src.models.freeze_validation --processed-dir data/processed
python -m src.models.harmonize_pipeline --processed-dir data/processed
python -m src.models.calendar_analytics --processed-dir data/processed
python scripts/export_submission.py --processed-dir data/processed --output submission.csv
```

### Resource envelope

- CPU-only; no NVIDIA GPU or CUDA requirement.
- DuckDB with a bounded memory limit and four threads for the reference run.
- Out-of-core Parquet processing; the 85M-row source is never loaded wholesale into pandas.
- Zero Google Cloud services, zero external APIs at query time, zero runtime CDN.
- Deterministic seeds and frozen folds for the benchmark.

### Submission contract

| Artifact | Purpose |
|---|---|
| `submission.csv` | 4,014-row public CSV, sorted by `iso_a3, year_month` |
| `submission_spec.json` | Exact-column, row-count, ID, non-negative, no-NaN contract |
| `scripts/export_submission.py` | Deterministic DuckDB export |
| `submission/slides_deck_7pages.html` | Exactly 7 offline slides, print-ready at 13.333 x 7.5 in |
| `README.md` | Project and reproducibility guide |
| `tests/test_submission_package.py` | Contract tests for export, deck, and documents |

The CSV schema is exactly:

```text
record_id, iso_a3, country_name, year_month, year, month,
harmonized_srehi, harmonized_frp_mw, raw_srehi, raw_frp_mw,
clean_det_count, raw_all_det_count, data_maturity, constellation_regime
```

`record_id` is `f"{iso_a3}_{year_month}"`. `raw_srehi` is the source `overpass_event_count`. The contract enforces `row_count = 4014`, `exact_columns = true`, `id_column = "record_id"`, `min = 0.0`, and `allow_nan = false`.

---

## Synthetic Intelligence (SI) / AI Tools Disclosure

**1. Scientific authorship.** Research formulation, multi-sensor harmonization (M0-M3), SAMA treatment, early-onset seasonal curves, and cartographic design are the original intellectual work of the project team. Physical constants, statistical models, and spatial aggregations were reviewed by the team.

**2. Technical assistance.** Synthetic Intelligence (SI) and AI software-engineering tools were used strictly as productivity accelerators for code implementation, automated test suites, documentation formatting, and bilingual interface copy under direct human supervision. They did not make scientific decisions.

**3. Accountability and verification.** All reported values come from deterministic Python/DuckDB pipelines executed on local CPU over the project's Parquet artefacts, with frozen splits and fixed seeds. The team verified the **85,002,288** raw records and the **1,453,803,616.63 MW** integrated radiative power conservation. AI tools did not generate, invent, impute, or alter any reported data value.

---

## Compliance Checklist

- [x] Public copy in clear technical and scientific English.
- [x] Exactly **7** slides at 13.333 x 7.5 in.
- [x] 100% offline deck with inline SVG charts.
- [x] Zero NASA logos, badges, seals, emblems, or graphical wordmarks.
- [x] Textual data attribution only.
- [x] Explicit Synthetic Intelligence (SI) / AI disclosure.
- [x] CPU-only, local-only execution; zero Google Cloud.
- [x] Bilingual EN/ES web application.
- [x] 4,014-row CSV contract with exact columns and no NaNs.

---

## Legal, Licensing, and Non-Tactical Use Notice

The repository uses a multi-part license recorded in `LICENSE`: MIT for source
code (`app/`, `src/`, `scripts/`, `tests/`, `Dockerfile`), CC-BY-4.0 for
documentation and presentation assets (`README.md`, `docs/`, `submission/`),
and CC0 1.0 Universal for derived data products (`submission.csv`,
`data/processed/*`). Third-party data, software, and scientific references are
recorded in `THIRD_PARTY_NOTICES.md`.

**NASA LANCE / FIRMS acknowledgment.** The project uses data and imagery from
NASA's Land, Atmosphere Near real-time Capability for EOSDIS (LANCE) system,
part of NASA's Earth Science Data and Information System (ESDIS), and the Fire
Information for Resource Management System (FIRMS). The full required
acknowledgment, the MODIS Collection 6.1 and VIIRS 375 m dataset DOIs, and the
NASA Open Data Policy statement appear in `THIRD_PARTY_NOTICES.md`.

**Natural Earth attribution.** Country and Admin-1 boundaries are derived from
Natural Earth 1:10m Cultural Vectors (`ne_10m_admin_0_countries`,
`ne_10m_admin_1_states_provinces`, v5.1.2), public domain (CC0). The simplified
delta-encoded geometry in `SOUTH_AMERICA_GEO` (`app/static/app.js`) is a
cartographic derivative for display only.

**Non-Endorsement.** Redistribution and analysis of derived NASA FIRMS data do
not imply sponsorship, endorsement, or official approval by NASA, LANCE,
ESDIS, NOAA, or the U.S. Government. NASA does not endorse this independent
challenge submission. No NASA seal, emblem, or graphical wordmark is used.

**Not for tactical fire management.** The submission is a scientific,
retrospective, and climatological product. Orbital revisit intervals, cloud
cover, and spatial aggregation on 10 km and 50 km grids make it unsuitable for
tactical wildfire suppression, real-time emergency dispatch, evacuation
routing, or life-and-property safety decisions. It must not be used as an
operational fire-management system.
