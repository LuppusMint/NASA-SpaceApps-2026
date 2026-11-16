# Third-Party Notices, Data Attribution, and Methodology References

Fire Relay South America is an independent NASA Space Apps Challenge 2026
submission. This document records the third-party data sources, terms of use,
software licenses, operational-safety caveats, and scientific references that
apply to the project. It is a notice and attribution document; it does not
modify the licenses in `LICENSE`.

---

## 1. Satellite and Geospatial Data Sources and Terms of Use

### 1.1 NASA LANCE / ESDIS / FIRMS acknowledgment

The project uses active-fire detections and associated imagery products from
NASA's Fire Information for Resource Management System (FIRMS), distributed
through NASA's Land, Atmosphere Near real-time Capability for EOSDIS (LANCE).
The required acknowledgment is:

> "We acknowledge the use of data and/or imagery from NASA's Land, Atmosphere Near real-time Capability for EOSDIS (LANCE) system (https://earthdata.nasa.gov/earth-observation-data/near-real-time), part of NASA's Earth Science Data and Information System (ESDIS), and the Fire Information for Resource Management System (FIRMS; https://firms.modaps.eosdis.nasa.gov/)."

The project does not redistribute the raw FIRMS archive. It redistributes
derived, harmonized monthly country products in `submission.csv` and derived
tabular summaries under `data/processed/`.

### 1.2 Dataset DOIs and product identifiers

The following FIRMS products are the upstream sources for the harmonized
record. They are governed by the NASA Open Data Policy (CC0 / 17 U.S.C. § 105)
and the applicable NASA Earthdata terms of use.

| Product | Sensor / collection | Dataset DOI |
|---|---|---|
| MODIS Collection 6.1 active fire product | MODIS Terra/Aqua, MCD14ML | `10.5067/FIRMS/MODIS/MCD14ML` |
| MODIS Collection 6.1 near-real-time active fire product | MODIS Terra/Aqua, MCD14DL NRT | `10.5067/FIRMS/MODIS/MCD14DL.NRT.0061` |
| VIIRS 375 m active fire product, S-NPP | VIIRS S-NPP, VNP14IMGT_NRT | `10.5067/FIRMS/VIIRS/VNP14IMGT_NRT.002` |
| VIIRS 375 m active fire product, NOAA-20 | VIIRS NOAA-20 (J1), VJ114IMGT_NRT | `10.5067/FIRMS/VIIRS/VJ114IMGT_NRT.002` |
| VIIRS 375 m active fire product, NOAA-21 | VIIRS NOAA-21 (J2), VJ214IMGT_NRT | `10.5067/FIRMS/VIIRS/VJ214IMGT_NRT.002` |

The harmonized index is a derived scientific product. It is not an official
NASA product, and NASA has not reviewed or approved the derivation.

### 1.3 Natural Earth 1:10m Cultural Vectors

The project uses **Natural Earth 1:10m Cultural Vectors**:

- `ne_10m_admin_0_countries` (country boundaries);
- `ne_10m_admin_1_states_provinces` (first-order administrative boundaries);
- Natural Earth version **v5.1.2**.

Natural Earth is public domain (**CC0**). The boundaries are used for
administrative clipping, country and Admin-1 assignment, and the simplified
delta-encoded vector boundaries exported as `SOUTH_AMERICA_GEO` in
`app/static/app.js`. The simplified display geometry is a cartographic
derivative and is not a survey-grade boundary product.

Natural Earth attribution: *Natural Earth — Free vector and raster map data at
naturalearthdata.com. Public domain (CC0).* See
<https://www.naturalearthdata.com/about/terms-of-use/>.

### 1.4 NASA Open Data Policy

NASA Earth science data are made available under the NASA Open Data Policy,
which places NASA-produced data in the public domain in the United States
(CC0 / 17 U.S.C. § 105). Users remain responsible for checking the current
FIRMS product documentation, quality flags, and terms of use at
<https://firms.modaps.eosdis.nasa.gov/> and
<https://earthdata.nasa.gov/>.

---

## 2. Mandatory Non-Endorsement & Operational Safety Disclaimer

> "Non-Endorsement & Tactical Use Disclaimer: Redistribution and analysis of derived NASA FIRMS data do not imply sponsorship, endorsement, or official approval by NASA, LANCE, ESDIS, NOAA, or the U.S. Government. All data, models, and visualizations are provided 'AS IS' without warranty of any kind, strictly for scientific, retrospective, and climatological research. Due to orbital revisit intervals, cloud cover, and spatial aggregation (10 km and 50 km grids), this system must NOT be used for tactical wildfire suppression, real-time emergency dispatch, evacuation routing, or life-and-property safety decisions."

This notice is repeated in the application footer and in the public
documentation. The project is not an operational fire-management system, an
emergency-response service, or a source of real-time incident intelligence.

---

## 3. Open-Source Software Dependencies and Transitive Library Notices

### 3.1 Direct Python dependencies

`requirements.txt` lists the following 13 direct dependencies with
OSI-permissive licenses. The version bounds are minimum compatible versions,
not a claim that only those versions work.

| Dependency | Minimum version | License | Role |
|---|---|---|---|
| `duckdb` | `>=1.1.0` | MIT | In-process analytical SQL and Parquet access |
| `geopandas` | `>=1.0.0` | BSD-3-Clause | Vector boundary handling and spatial joins |
| `httpx` | `>=0.27.0` | BSD-3-Clause | HTTP client and test transport |
| `lightgbm` | `>=4.5.0` | MIT | Gradient-boosted tree models |
| `numpy` | `>=1.26.0` | BSD-3-Clause | Numerical arrays and linear algebra |
| `pandas` | `>=2.2.0` | BSD-3-Clause | Tabular data manipulation |
| `polars` | `>=1.0.0` | MIT | Out-of-core DataFrame processing |
| `pyarrow` | `>=16.0.0` | Apache-2.0 | Parquet, Arrow, and columnar I/O |
| `pyproj` | `>=3.6.0` | MIT | CRS transformations and Albers projection |
| `scikit-learn` | `>=1.5.0` | BSD-3-Clause | Calibration, weighting, and validation models |
| `scipy` | `>=1.13.0` | BSD-3-Clause | Statistical distributions and spatial trees |
| `shapely` | `>=2.0.0` | BSD-3-Clause | Planar geometry operations |
| `statsmodels` | `>=0.14.0` | BSD-3-Clause | Statistical models and diagnostics |

All 13 licenses are OSI-permissive: MIT, BSD-3-Clause, or Apache-2.0. The
repository does not vendor or relicense these packages; users obtain them from
their upstream distributions.

### 3.2 Key transitive libraries

The direct dependencies pull in additional libraries. The following are the
most relevant transitive notices for this project:

| Transitive library | License | Notes |
|---|---|---|
| `pyogrio` | MIT | Vector I/O backend used by GeoPandas |
| `joblib` | BSD-3-Clause | Parallelism and model persistence |
| `patsy` | BSD-2-Clause | Formula parsing for statistical models |
| `certifi` | MPL-2.0 | CA certificate bundle used by HTTP clients |

The complete transitive dependency graph is resolved by the package index at
install time. Each transitive package keeps its own copyright notice and
license; nothing in this repository overrides those terms.

### 3.3 Binary wheels, native libraries, and LGPL components

Precompiled binary wheels of `scipy` and `shapely` may dynamically link native
libraries, including `libquadmath` (used by `scipy`) and `GEOS` (used by
`shapely`). Those native libraries are distributed under LGPL-2.1 or their own
compatible terms. This repository distributes only original Python source
code and does not redistribute those native libraries or their binary wheels.
Anyone redistributing a bundled wheel or container image is responsible for
complying with the corresponding LGPL-2.1 obligations, including notice and
source-availability requirements.

### 3.4 Front end and slide deck

The web frontend (`app/static/`) and the slide deck
(`docs/slides_deck_7pages.html`) use zero third-party JavaScript libraries,
zero external fonts, and zero external map tile services. All charts, icons,
and map geometry are generated locally with standard HTML, CSS, JavaScript,
Canvas, and inline SVG. The application is designed to run fully offline with
no runtime CDN, analytics, tracker, or remote asset request.

### 3.5 `src/core/` utility primitives

`src/core/` contains the team's open-source general mathematical,
geostatistical, and numerical utility primitives. These modules are released
under the MIT License, consistent with Part I of `LICENSE`. They implement
reusable operations such as equal-area gridding, robust location estimation,
cross-sensor bridge calibration, Kalman smoothing, spatial and temporal split
construction, trend testing, and extreme-value fitting. They are not
third-party code and do not carry separate upstream notices.

---

## 4. Scientific and Statistical Methodology References

The methods implemented in this project are based on, or informed by, the
following publications. Citations are provided for scientific attribution and
reproducibility; the project does not reproduce the copyrighted text of these
works.

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

---

## 5. Attribution and Compliance Summary

- Textual attribution is used throughout. No NASA seal, emblem, or graphical
  wordmark is included in the application, documentation, or slide deck.
- NASA, LANCE, ESDIS, NOAA, and the U.S. Government do not endorse this
  independent challenge submission.
- Natural Earth boundaries are public domain (CC0) and are used for
  cartographic display and administrative assignment only.
- The project is scientific, retrospective, and climatological. It is not for
  tactical wildfire suppression, emergency dispatch, evacuation routing, or
  life-and-property safety decisions.
- Questions about this notice should be directed to the project maintainers
  through the repository issue tracker.
