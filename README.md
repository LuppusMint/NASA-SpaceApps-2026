# Fire Relay South America

**A harmonized, leakage-free burning calendar for South America from 26 years of NASA FIRMS observations (2000-11 to 2026-11).**

Fire Relay South America turns five satellite instruments into one comparable monthly active-fire record. It removes the artificial trends created by the MODIS-to-VIIRS constellation relay, reconciles the geographic hierarchy exactly, and serves the result through a fully offline bilingual (English/Spanish) web application.

The project is **CPU-only**, **local-first**, and **cloud-free**. There is no Google Cloud dependency, no runtime CDN, no external font, no remote map tile, and no third-party analytics.

> **Data attribution (text only):** NASA FIRMS Archive & NRT - MODIS C6.1, VIIRS S-NPP, NOAA-20/J1, NOAA-21/J2.
> No NASA logo, emblem, seal, or wordmark is used in this repository.

---

## Quickstart

All commands run from the root of this folder. Python 3.12 is the reference runtime.

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Fetch the large processed cubes

Five derived artefacts exceed GitHub's comfortable file-size limit and are distributed through the `v1.0-processed` release. The script downloads them into `data/processed/` and verifies their SHA-256 checksums:

```bash
python scripts/fetch_processed_data.py
```

The lightweight cubes, calendar tables, provenance files, and benchmark JSON needed by the API and the tests are already tracked in this folder. Without the five release assets, the country/Admin-1 calendars, time series, trends, and CSV export still work; the 10 km and 50 km map layers and bounding-box/radius queries require the fetched assets.

### 3. Run the web application

```bash
python -m app.server --processed-dir data/processed --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/`. The application is offline by design: it serves local static files and queries local Parquet cubes with DuckDB.

### 4. Run the test suite

```bash
python -m unittest discover -s tests
```

The suite covers the deterministic data pipeline, the frozen M0-M3 benchmark, hierarchical conservation, climatology/trends/GEV/early onset, the HTTP API, the submission contract, and the seven-slide deck. It uses synthetic fixtures where multi-hundred-megabyte inputs are not required.

### 5. Optional: rebuild the pipeline from raw FIRMS archives

```bash
export FIRMS_MAP_KEY="<your NASA FIRMS map key>"
python scripts/download_firms_sa.py
python scripts/build_canonical_parquet.py
python -m src.data.build_spatiotemporal_cubes --project-root .
python -m src.models.freeze_validation --processed-dir data/processed
python -m src.models.harmonize_pipeline --processed-dir data/processed
python -m src.models.calendar_analytics --processed-dir data/processed
python scripts/export_submission.py --processed-dir data/processed --output submission.csv
```

`FIRMS_MAP_KEY` is read strictly from the environment; no credential is stored in the repository.

### 6. Docker deployment

```bash
docker build -t fire-relay-south-america .
docker run --rm -p 8080:8080 fire-relay-south-america
```

The image runs `python -m app.server --processed-dir data/processed --host 0.0.0.0 --port 8080`. For a data volume outside the image, mount it and pass a matching `--processed-dir`; the default image already contains the tracked lightweight cubes.

The container is compatible with Google Cloud Run because it reads `$PORT` and binds `0.0.0.0`; this is optional hosting, not a runtime dependency. The application does not call Google Cloud APIs and the scientific product remains offline/local-first. Example:

```bash
gcloud run deploy fire-relay-south-america --source . --region=us-central1 \
  --allow-unauthenticated --memory=1Gi --port=8080
```

The 10 km/50 km map layers need the Release assets; the default image contains the tracked country/Admin-1 cubes only.
### 6.1 Including the Release assets in a container

The five large processed cubes are not Git-tracked. For a deployment that needs the 10 km/50 km map layers, place them in `data/processed/` before `docker build` or `gcloud run deploy --source .`; the `.dockerignore` excludes raw archives and intermediate Parquet, not these processed assets. Without them the container still serves country/Admin-1/calendar, trends, early onset and CSV, but the spatial map endpoints report missing assets.


---

## Repository layout

```text
fire-relay-south-america/
├── app/
│   ├── server.py                  # offline WSGI API + DuckDB + CSV export
│   └── static/                    # light-only bilingual EN/ES front end
├── src/
│   ├── core/                      # self-contained scientific utilities
│   │   ├── duckdb_helper.py
│   │   ├── metrics.py
│   │   ├── geogrid.py
│   │   ├── splits_st.py
│   │   ├── harmonize.py
│   │   ├── postprocess.py
│   │   ├── stacking.py
│   │   ├── trends.py
│   │   └── extremes.py
│   ├── data/
│   │   ├── clean_and_flag.py
│   │   └── build_spatiotemporal_cubes.py
│   └── models/
│       ├── freeze_validation.py
│       ├── harmonize_pipeline.py
│       └── calendar_analytics.py
├── scripts/
│   ├── download_firms_sa.py
│   ├── build_canonical_parquet.py
│   ├── fetch_processed_data.py
│   └── export_submission.py
├── data/
│   ├── DATA_SUMMARY.json
│   ├── PROVENANCE.json
│   └── processed/                 # lightweight cubes + benchmark JSON
├── docs/
│   ├── 01_MATHEMATICAL_AND_STATISTICAL_METHODOLOGY.md
│   ├── 02_DATA_PIPELINE_AND_VALIDATION.md
│   ├── 03_APPLICATION_AND_CARTOGRAPHY.md
│   ├── 04_NASA_SPACE_APPS_SUBMISSION.md
│   └── slides_deck_7pages.html
├── tests/
│   ├── test_data_cubes.py
│   ├── test_harmonization_models.py
│   ├── test_webapp_api.py
│   └── test_submission_package.py
├── submission/
│   ├── SUBMISSION_PACKAGE_EN.md
│   └── slides_deck_7pages.html
├── requirements.txt
├── Dockerfile
├── submission_spec.json
├── submission.csv
└── README.md
```

Data pipeline code lives in `src/data/`; the calibrated models live in `src/models/`; the self-contained scientific helpers live in `src/core/`; the submission contract is checked by `tests/test_submission_package.py`.

---

## Dataset summary

| Item | Value |
|---|---:|
| Temporal coverage | **2000-11** to **2026-11** |
| Months | **313** |
| Raw NASA FIRMS detections, South America | **85,002,288** |
| Clean detections after quality control | **75,556,207** |
| Countries and territories | **14** |
| Admin-1 units | **243** |
| Historical active 10 km cells | **143,613** |
| Persistent static 10 km cells masked | **358** |
| Country-month rows | **4,014** (absence = no detections, not missing data) |
| Frozen validation overlap rows (2012-2020) | **930,100** = **465,050** cell-months × 2 replicas |
| Unique 50 km cells in the overlap | **7,184** |
| Integrated fire radiative power | **1,453,803,616.63 MW** |
| Grid | Albers Equal-Area, **EPSG:102033**, 10 km and 50 km |

The five instruments have different footprints (MODIS 1 km, VIIRS 375 m), overpass schedules, quality flags, and operating periods. The constellation expands through Terra-only, Terra+Aqua, the calibrated MODIS/VIIRS overlap, and the deduplicated VIIRS backbone. Terra/Aqua orbital drift after 2021 and the South Atlantic Magnetic Anomaly (SAMA, 7°N–55°S, 110°W–11°E) add further structure that a naive sum would misread as a fire trend.

The calibration therefore learns the relay only on the pre-drift overlap **2012-02 to 2020-12**, freezes spatial folds, and applies the learned bridge and reconciliation to the full record. The full mathematical treatment is in [docs/01_MATHEMATICAL_AND_STATISTICAL_METHODOLOGY.md](docs/01_MATHEMATICAL_AND_STATISTICAL_METHODOLOGY.md).

---

## Key results: M0 to M3

Five-fold out-of-fold benchmark on the frozen 200 km spatial blocks:

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

M2 has the lowest point error. The fitted M3 simplex weights are 0.0 for M1 and 1.0 for M2, so out of fold M3 equals M2 on four of the five folds; its higher mean RMSE comes entirely from the first fold, which the causal blend scores with uniform warm-up weights by construction. The M1/M2 simplex weight is 1.0 on M2. In the MODIS-only backcast the production point is the M2 bridge; in overlap cells with VIIRS it blends the bridge with the observation. The hierarchy is reconciled non-negatively; Admin-1 is a proportional downscaling of the country total, not a bottom-up sum of cells. A temporal hold-out re-run (fit on 2016–2020, predict 2012–2015) gives Terra-only SREHI R² 0.808 and FRP R² 0.748 (Terra+Aqua 0.920/0.890); the overlap is 930,100 rows = 465,050 cell-months × 2 replicas. The operational calendar uses M3 because a public country total must equal the sum of its Admin-1 units. Temporal and backcast checks are in [docs/validation/](docs/validation/).

**Relay continuity.** Across the five constellation transitions (2002, 2012, 2018, 2021, 2023), the aggregate absolute step discontinuity `|delta_step|` falls **92.84%**, from **2.1892** to **0.1567**. The SREHI step ratio improves from **3.414** to **1.523** at 2002 and from **3.938** to **1.177** at 2012; the 2023 transition is reduced by **99.9%**.

**Exact conservation.** The maximum Admin-1 to country difference is **4.37e-11** in SREHI and **3.73e-09 MW** in FRP. The maximum country to South America difference is **5.82e-11** in SREHI and **1.86e-09 MW** in FRP. The minimum reconciled value is **>= 0.0**.

**Calendar science.** The climatology is frozen to **2001-2020** so the post-2021 regime cannot leak into the baseline. Trends use Theil-Sen slopes, the Hamed-Rao modified Mann-Kendall test, and Benjamini-Hochberg FDR at **q < 0.05**. GEV return levels are fitted on annual block maxima (the largest monthly value per year) by L-moments/PWM with a parametric bootstrap, plus a sanity bound: ordered positive levels, median anchor, shape in [-0.5, 0.5], and the 10-year level ≤10× the observed annual maximum. In the current run 475/510 series pass and 25/26 countries have bootstrap CIs; the GEV is a screening diagnostic, not an operational return-period product. The 90-day early-onset curve is a true trailing window across year boundaries; the first 89 days are `INSUFFICIENT_HISTORY`, and thresholds require complete windows with at least 10 years. **Peru** is the only country with a significant SREHI trend (slope **+903.23/yr**, z **3.34**, q **0.0134**). **45 of 242** Admin-1 units are significant: **34 increasing**, **11 decreasing**.

Detailed derivations, rejected alternatives, and limitations are in [docs/01](docs/01_MATHEMATICAL_AND_STATISTICAL_METHODOLOGY.md) and [docs/02](docs/02_DATA_PIPELINE_AND_VALIDATION.md).

---

## Interactive application

The web app is a single-file WSGI server (`app/server.py`) over DuckDB plus a dependency-free HTML/CSS/JavaScript front end. It provides:

- country, Admin-1, bounding-box, and point-radius selection;
- a month-year range selector with quick presets and a keyboard/pointer calendar popover;
- SREHI and FRP metrics, harmonized M3 versus raw series;
- an annual activity calendar, a monthly series with 90% bands, a 90-day early-onset curve, and a South America cell map;
- an audit table with anomaly scores, severity, data maturity, and constellation regime;
- one-click UTF-8 CSV export.

The interface is light-only, defaults to English, and switches to Spanish without a reload. It reads only local static assets and local Parquet. The API contract and cartographic rationale are in [docs/03_APPLICATION_AND_CARTOGRAPHY.md](docs/03_APPLICATION_AND_CARTOGRAPHY.md).

---

## Data provenance

The raw FIRMS archive is not redistributed here. `submission.csv` is a derived, harmonized monthly country product. Source quality flags, confidence classes, and type flags are preserved or explicitly documented before transformation. `data/PROVENANCE.json` and `data/DATA_SUMMARY.json` record the generated counts and the portable pipeline provenance.

`data/processed/` contains the lightweight derived tables. The five largest artefacts are fetched from the release and verified against committed SHA-256 values. No credentials, local user paths, or machine-specific paths are stored in this folder.

---

## Synthetic Intelligence (SI) / AI transparency disclosure

The project uses a three-pillar disclosure.

1. **Scientific authorship.** The research question, the multi-sensor harmonization design (M0-M3), the SAMA treatment, the seasonal-calendar mathematics, the early-onset definition, and the cartographic design are the original intellectual work of the project team. Physical constants, model choices, and spatial aggregations were reviewed by the team.
2. **Synthetic Intelligence (SI) and AI assistance.** SI/AI software-engineering tools were used as productivity accelerators for code implementation, test scaffolding, documentation formatting, and bilingual interface copy. They operated under direct human supervision and did not make scientific decisions.
3. **Accountability and verification.** All reported numbers come from deterministic Python/DuckDB pipelines executed on local CPU over the project's Parquet artefacts, with frozen splits and fixed seeds. The team verified the **85,002,288** raw records and the **1,453,803,616.63 MW** integrated radiative power conservation. AI tools did not generate, invent, impute, or alter any reported data value.

---

## Documentation index

- [Mathematical and statistical methodology](docs/01_MATHEMATICAL_AND_STATISTICAL_METHODOLOGY.md)
- [Data pipeline and validation](docs/02_DATA_PIPELINE_AND_VALIDATION.md)
- [Application and cartography](docs/03_APPLICATION_AND_CARTOGRAPHY.md)
- [NASA Space Apps submission package](docs/04_NASA_SPACE_APPS_SUBMISSION.md)
- [Seven-slide deck](docs/slides_deck_7pages.html)
- [Public submission copy](submission/SUBMISSION_PACKAGE_EN.md)

---

## Resource envelope and constraints

- **CPU-only.** No GPU, CUDA, or accelerator is required.
- **Local-only.** No Google Cloud, hosted database, or external API at query time.
- **Bounded memory.** DuckDB uses a configurable memory limit (`FIRE_RELAY_DUCKDB_MEM`, default `6GB`) and four threads (`FIRE_RELAY_THREADS`).
- **Out-of-core.** Parquet is the canonical format; the 85M-row source is processed with DuckDB/Polars, never loaded wholesale into pandas.
- **Deterministic.** Fixed seeds, frozen spatial folds, stable sort order, and byte-stable CSV export.
- **Offline UI.** No CDN, remote font, remote tile, tracker, or NASA logo.

---

## License, Data Citation & Operational Disclaimer

**License.** This repository is distributed under the multi-part terms in
[`LICENSE`](LICENSE). Source code (`app/`, `src/`, `scripts/`, `tests/`,
`Dockerfile`) is licensed under the MIT License. Documentation and presentation
assets (`README.md`, `docs/`, `submission/`) are licensed under CC-BY-4.0.
Derived data products (`submission.csv`, `data/processed/*`) are released under
CC0 1.0 Universal. Third-party data, software, and scientific references are
recorded in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

**NASA LANCE / FIRMS acknowledgment.** We acknowledge the use of data and/or
imagery from NASA's Land, Atmosphere Near real-time Capability for EOSDIS
(LANCE) system
(<https://earthdata.nasa.gov/earth-observation-data/near-real-time>), part of
NASA's Earth Science Data and Information System (ESDIS), and the Fire
Information for Resource Management System (FIRMS;
<https://firms.modaps.eosdis.nasa.gov/>).

**Dataset DOIs.** MODIS Collection 6.1: `10.5067/FIRMS/MODIS/MCD14ML` and
`10.5067/FIRMS/MODIS/MCD14DL.NRT.0061`. VIIRS 375 m: S-NPP
`10.5067/FIRMS/VIIRS/VNP14IMGT_NRT.002`, NOAA-20
`10.5067/FIRMS/VIIRS/VJ114IMGT_NRT.002`, and NOAA-21
`10.5067/FIRMS/VIIRS/VJ214IMGT_NRT.002`. These products are governed by the
NASA Open Data Policy (CC0 / 17 U.S.C. § 105). Country and Admin-1 boundaries
are derived from Natural Earth 1:10m Cultural Vectors v5.1.2, public domain
(CC0).

**Non-Endorsement & Not-for-Tactical-Fire-Management.** Redistribution and
analysis of derived NASA FIRMS data do not imply sponsorship, endorsement, or
official approval by NASA, LANCE, ESDIS, NOAA, or the U.S. Government. The
product is provided "AS IS" for scientific, retrospective, and climatological
research only. Because of orbital revisit intervals, cloud cover, and spatial
aggregation on 10 km and 50 km grids, it must not be used for tactical wildfire
suppression, real-time emergency dispatch, evacuation routing, or
life-and-property safety decisions.

**Key references.** Giglio, L., Schroeder, W., & Justice, C. O. (2016).
*The collection 6 MODIS active fire detection algorithm and fire products*.
Remote Sensing of Environment, 178, 31–41.
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
