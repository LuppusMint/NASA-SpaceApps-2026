# Application and Cartography

**Fire Relay South America - offline WSGI application, API contract, map design, and Docker deployment.**

The application is the public face of the harmonized record. It must be fast on a laptop CPU, work without a network, and never imply more spatial precision than the sensors support. This document describes the architecture, the exact API surface, the cartographic choices, and the deployment path.

---

## 1. Architecture

```text
Browser (HTML5 + CSS + vanilla JS)
  |  same-origin fetch / static files
  v
app/server.py  (WSGI, Python standard library)
  |  thread-local DuckDB connection
  v
data/processed/*.parquet
  harmonized_country_monthly.parquet
  harmonized_admin1_monthly.parquet
  harmonized_south_america_monthly.parquet
  cube_country_monthly.parquet
  cube_admin1_monthly.parquet
  cube_admin1_daily_season.parquet
  calendar_climatology.parquet
  calendar_matrix_*.parquet
  calendar_trends.parquet
  calendar_early_onset_90d.parquet
  harmonized_cell50km_monthly.parquet   (release asset)
  cube_cell10km_monthly.parquet         (release asset)
  benchmarks_m0_m3.json
```

### 1.1 Backend

- `app/server.py` is a WSGI application built on the standard library. It binds to loopback by default and serves only local static files and local Parquet.
- Each worker thread gets its own DuckDB connection (`threading.local`). Connections use two threads and a 1 GB memory limit, so a laptop remains responsive.
- Queries use `read_parquet` directly. They project only the columns they need and push filters into DuckDB; the full cube is never materialised in Python.
- A bounded in-memory response cache (48 entries, 600 s TTL) absorbs repeated month/area requests.
- CSV export is streamed as a UTF-8 attachment with `text/csv; charset=utf-8`.

### 1.2 Front end

- `app/static/index.html` is semantic HTML with explicit labels, a skip link, ARIA live regions, and a keyboard-operable date picker.
- `app/static/styles.css` is a single light theme (`color-scheme: light`) using a mineral-paper palette: page `#f4f3ee`, surface `#ffffff`, ink `#0f172a`, brand `#c2410c`, accent `#0d9488`, info `#0284c7`. There is no dark stylesheet, no `prefers-color-scheme` branch, and no theme switcher.
- `app/static/app.js` is dependency-free. Charts are native SVG; the map is a native Canvas.
- The interface is bilingual (`EN` default, `ES` toggle). The language switch re-renders labels, status text, chart axes, and the date-picker month names without a page reload.

### 1.3 Date-range selector

The two native `<input type="month">` controls were replaced by a conventional month-year range selector that works in every browser:

- hidden canonical inputs `#start-ym` and `#end-ym` store `YYYY-MM` (or an empty string for the full span), so `currentFilterParams()` keeps sending exact `start_ym` / `end_ym`;
- paired visible triggers `#start-date-btn` and `#end-date-btn` show labels such as `Nov 2000` and `Nov 2026`;
- quick presets: `All (2000-2026)`, `Last 12M`, `Last 5Y`, `Baseline (2001-2020)`, `Drift Era (2021-2026)`;
- a floating `#date-picker-popover` with `role="dialog"`, Start/End tabs, a year `<select>` for 2000-2026, previous/next year buttons, and a 12-month grid;
- the grid highlights `is-start`, `is-end`, and the `is-in-range` band, disables months outside the catalog, and swaps reversed endpoints chronologically;
- Escape and outside clicks close the popover; picking a Start auto-advances to End; completing the range or clicking a preset calls `refresh()`.

The selector is keyboard operable and announces its state through `aria-expanded`, `aria-selected`, and `aria-pressed`.

---

## 2. API contract

All responses are JSON unless noted. Errors return a JSON object with `error` and a human-readable `message`, plus an HTTP status of 400, 404, or 500.

### 2.1 `GET /api/health`

Liveness and data-availability report. It never emits internal directory names beyond the processed directory's basename, and it does not expose an operations directory.

Selected fields:

| Field | Meaning |
|---|---|
| `status` | `ok` when all core Parquet files are readable, otherwise `degraded` |
| `version` | Application version |
| `processed_dir` | Basename of the processed directory |
| `parquets` | Per-file availability, size, modification time, query latency |
| `core_available` | Boolean for the required cubes |
| `qc_cleaning_summary` | Copy of the cleaning ledger |
| `benchmarks_m0_m3` | Benchmark summary (models, transitions, dedup, guard) |
| `duckdb` | Version, threads, memory limit, connection mode |

### 2.2 `GET /api/catalog`

Returns the geography hierarchy and the dataset time range:

```json
{
  "countries": [{"iso_a3": "BRA", "country_name": "Brazil", "admin1": [...], "bbox": [...]}],
  "admin1": [...],
  "time_range": {"min_year_month": "2000-11", "max_year_month": "2026-11"},
  "sama": {"lat_min": -55.0, "lat_max": 7.0, "lon_min": -110.0, "lon_max": 11.0}
}
```

The UI uses `time_range` to bound the date picker and the year selectors.

### 2.3 `GET /api/calendar`

Monthly calendar, climatology comparison, trends, and summary indicators.

| Parameter | Type | Notes |
|---|---|---|
| `iso_a3` | string | Three-letter code or `SOUTH_AMERICA` |
| `admin1_name` | string | Optional; empty means all Admin-1 units |
| `metric` | string | `srehi` or `frp_mw` |
| `series_mode` | string | `harmonized` or `raw` |
| `start_ym` | string | Optional `YYYY-MM` |
| `end_ym` | string | Optional `YYYY-MM` |
| `bbox` | string | Optional `min_lat,max_lat,min_lon,max_lon` |
| `lat`, `lon`, `radius_km` | numbers | Optional point AOI |

The response includes:

- `selection`: resolved geography, metric, series mode, date range, and source file;
- `timeseries`: M3 value, 90% band, raw value, anomaly z, percent of median, severity, maturity, and regime;
- `matrix`: the annual month-by-year matrix used by the activity calendar;
- `climatology`: frozen 2001-2020 quantiles;
- `kpis`: latest month, peak month, last-12-month total, year-over-year, selected total, static-cell count, and mean band;
- `badges`: quality, SAMA/static, and relay/uncertainty summaries.

### 2.4 `GET /api/spatial`

Cell-level map data.

| Parameter | Type | Notes |
|---|---|---|
| `year_month` | string | `YYYY-MM`; mutually exclusive with `year` |
| `year` | integer | Annual aggregate; mutually exclusive with `year_month` |
| `iso_a3` | string | Country or `SOUTH_AMERICA` |
| `resolution` | string | `50km` (default) or `10km` |
| `include_static` | boolean | Include the static thermal mask layer |
| `limit` | integer | Page size, default `10000`, range `1..20000` |
| `offset` | integer | Zero-based page offset, range `0..100000` |
| `range_start`, `range_end` | string | Optional `YYYY-MM` reference window for the fixed colour scale |
| `bbox` | string | Optional bounding box |
| `lat`, `lon`, `radius_km` | numbers | Optional point AOI |

The 10 km response is read from `cube_cell10km_monthly.parquet`; the 50 km response prefers `harmonized_cell50km_monthly.parquet` and falls back to `cube_cell50km_monthly.parquet`. The point AOI switches to the 10 km cube when `radius_km <= 50`.

The response includes `cells`, `static_cells`, `available_months`, `reference_scale`, and a `summary` with:

- exact pagination metadata: `total_available_cells`, `returned_cells`, `active_cells`, `applied_limit`, `offset`, `truncated`;
- `month_scale` (`max`, `p99`, `p95`, `ceiling`) computed over every cell of the month before slicing;
- traceability fields: `month_state`, `effective_resolution`, `value_type`, `constellation_regime`, `participating_sensors`, `data_maturity`, `is_partial_month`, and `processing_date_utc`.

The `50 km` layer is always `M3_HARMONIZED`. The `10 km` layer is always `RAW_OVERPASS_COUNT`: it reports orbital overpasses, not the calibrated M3 index, and the UI says so permanently.

### 2.5 `GET /api/early-onset`

90-day early-onset curve.

| Parameter | Type | Notes |
|---|---|---|
| `iso_a3` | string | Country or `SOUTH_AMERICA` |
| `admin1_name` | string | Optional |
| `year` | integer | Season year |

The response includes the daily curve, the frozen p75/p90 thresholds, the first crossing dates, the number of critical days, and the peak signal.

### 2.6 `GET /api/export.csv`

UTF-8 CSV export of the selected monthly record. It accepts the same geography, metric, series, date, bounding-box, and point-radius parameters as `/api/calendar`. The response is an attachment named `srehi_calendar_export.csv`.

### 2.7 Static assets

`GET /` serves `app/static/index.html`. `GET /static/<file>` serves the CSS and JavaScript with path-traversal protection. There is no external asset host.

---

## 3. Cartographic justification

### 3.1 Why equal-area cells, not point markers

A point marker implies a fire at an exact coordinate. FIRMS provides a detection coordinate, but the footprint is **1 km** (MODIS) or **375 m** (VIIRS), and the product is a sample at overpass time. A dense point cloud also overplots and hides the spatial structure at continental scale.

The map therefore draws the analysis cell, not the raw point. The 50 km cell is the public map unit; the 10 km cell is used for point-radius queries up to 50 km. This is the honest unit of the calibrated product.

### 3.2 Why not kernel density estimation

KDE is visually attractive but introduces a bandwidth choice that changes the apparent pattern. It also smears activity across coastlines, borders, and uninhabited regions. A reader cannot tell whether a hotspot is a detection or a smoothing artefact. Equal-area cells preserve the actual support and make the intensity comparison reproducible.

### 3.3 Why not municipal choropleths

Most municipalities are smaller than the 10 km cell, and all are smaller than the 50 km cell. A municipal choropleth would imply that a detection can be attributed to a municipality with survey-grade confidence. The grid cannot support that claim. Admin-1 is the finest public choropleth level because it is large enough to contain the cell footprint, and even there the app states a **±25 km** support at 50 km and **±5 km** at 10 km.

### 3.4 Projection on the canvas

The Canvas map uses a cosine-corrected geographic projection at **18°S**:

\[
x = \lambda \cos(18^\circ), \qquad y = \phi.
\]

This keeps South America's shape close to an atlas while preserving the correct east-west scale at the continent's population-weighted centre. It is a display projection, not the analysis projection. All area calculations and cell indices use the Albers Equal-Area grid.

### 3.5 Boundaries

Country and Admin-1 outlines come from **Natural Earth 1:10m** boundaries, simplified offline and embedded in the JavaScript bundle. No remote tile server is used. The boundary layer is decorative context; it does not override the cell labels. A cell whose centre falls near a border is still assigned to one Admin-1 unit, and the inspector states the epistemic limit.

### 3.6 Layers

- country boundaries;
- Admin-1 internal borders;
- harmonized fire cells;
- static thermal cells (the persistent-source mask);
- SAMA containment;
- selected AOI.

The map supports wheel zoom, drag pan, a fit-to-selection action, a reset action, a zoom badge, a hover tooltip, and a pinned cell inspector. Clicking in point mode sets the AOI.

### 3.7 Temporal navigation, colour scales and scientific transparency

Module `05 · Spatial observatory` is an animated monthly map, not a static snapshot. Its toolbar provides previous/next month, a month selector, play/pause (`1x = 900 ms`, `2x = 450 ms`, `4x = 220 ms`), and a colour-scale mode. Playback stops automatically on the last month of the selected range. Clicking any monthly point on `03 · Harmonized record` jumps the map directly to that month without re-running modules `01–04`; the point receives a visible active ring and the map requests a single `/api/spatial` frame.

The navigable domain is the intersection of the catalogue months with the selected `[start_ym, end_ym]`. If the catalogue does not expose an explicit month list, the UI generates the continuous monthly sequence. Applying a new range in `01` resets the map to the last month of that range.

Two colour scales are offered:

- **Fixed range scale (p99 ref)** — the default. The ceiling is the cached `reference_scale` over the selected range. A p99 cap is applied when the maximum exceeds `1.5 × p99`, so one extreme cell does not wash out the continent. This scale never changes during playback.
- **Adjusted to month** — the ceiling follows `summary.month_scale` for the current month, making intra-month structure visible.

`50 km` cells always use the M3-harmonized values. The `10 km` point-radius layer is explicitly `RAW_OVERPASS_COUNT` and shows a permanent resolution notice; it is never presented as harmonized. The inspector separates the **Harmonized (M3)** block (SREHI, 90% PI, FRP) from the **Observed Raw** block (raw detections, night overpasses) at `50 km`, and replaces those blocks with the raw-count warning at `10 km`.

Scientific states are displayed as a status banner: `ZERO_DETECTIONS`, `PARTIAL_MONTH`, `NRT_PRELIMINARY`, `OUT_OF_CATALOG`, `ASSET_MISSING`, or `QUERY_ERROR`. A failed spatial request is never silent: the map clears and the explicit error state is shown. When a page is truncated, a banner reports `returned / total (limit: applied_limit)` and offers **Load all cells** up to the API maximum of `20,000`. A traceability strip reports month and state, resolution/value type, constellation regime, participating sensors, data maturity, cell counts, and the processing timestamp.

`04 · Early onset` is decoupled from `01` and `02`: changing the calendar year or month no longer overwrites the season year, and a contextual warning appears when the chosen season year falls outside the selected date range.

### 3.8 Dual-folder synchronization

The repository ships two synchronized deliverables: the root project and the standalone `fire-relay-south-america/` folder. The three shared static assets (`app/static/index.html`, `app/static/styles.css`, `app/static/app.js`) and the frozen evidence (`submission.csv`, `submission_spec.json`, `data/processed/benchmarks_m0_m3.json`) must remain byte-for-byte identical. `tests/test_dual_deliverable_parity.py` enforces SHA-256 equality and forbids internal tooling identifiers in the standalone folder. The standalone `app/server.py` exposes the same `/api/spatial` contract but carries no internal operations-directory reference.

---

## 4. Performance

On the reference CPU:

- warm query latency **< 1 ms** for cached requests;
- cold query latency **9.8-135.9 ms** on real cubes;
- no request loads the full 85M-row table; DuckDB pushes the filters and projections into the Parquet scan.

The response cache is bounded, so a long browsing session cannot grow memory without limit.

---

## 5. Docker deployment

The included `Dockerfile` is:

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir duckdb pyarrow numpy pandas

EXPOSE 8080

CMD ["sh", "-c", "python -m app.server --processed-dir data/processed --host 0.0.0.0 --port ${PORT:-8080}"]
```

Build and run:

```bash
docker build -t fire-relay-south-america .
docker run --rm -p 8080:8080 fire-relay-south-america
```

For a cloud platform, set `PORT` and mount a volume that contains the fetched processed cubes:

```bash
docker run --rm -p 8080:8080 \
  -e PORT=8080 \
  -v /srv/fire-relay/data/processed:/app/data/processed:ro \
  fire-relay-south-america
```

The container is stateless. It needs no database service, no message broker, and no external API. The only network access is the inbound HTTP port.

### 5.1 Health check

Point the platform health check at `GET /api/health`. A `200` response with `status: "ok"` means the core Parquet files are readable. A `degraded` response still returns `200` so the platform does not restart a container that is merely missing an optional release asset; the payload lists exactly which file is absent.

### 5.2 Resource guidance

- 1 vCPU and 1 GB RAM are sufficient for the tracked lightweight cubes and country/Admin-1 queries.
- 2 vCPU and 2-4 GB RAM are recommended when serving the 50 km map layer concurrently.
- The 10 km cube is larger; use it for point-radius queries or pre-warm the cache.
- No GPU is required at any stage.

---

## 6. Accessibility and offline guarantees

- Every form control has a visible `<label for=...>`; the date triggers expose `aria-haspopup`, `aria-expanded`, and `aria-controls`.
- The map canvas has a descriptive `role="img"` label, and the inspector provides a textual equivalent for the selected cell.
- The table has a caption, scoped headers, and a keyboard-focusable scroll region.
- Colour is not the only carrier of meaning: severity and maturity use text chips.
- The interface honours `prefers-reduced-motion`.
- No external CDN, font, tile, image, tracker, or analytics script is requested at runtime. The page works from a local server with the network disconnected.
- No NASA logo or emblem appears in the HTML, SVG, CSS, or JavaScript.

---

## Legal, Licensing, and Non-Tactical Use Notice

The application source code, documentation, presentation assets, and derived
data products are distributed under the multi-part terms in `LICENSE`: MIT for
software (`app/`, `src/`, `scripts/`, `tests/`, `Dockerfile`), CC-BY-4.0 for
documentation and slides (`README.md`, `docs/`, `submission/`), and CC0 1.0
Universal for derived data products (`submission.csv`, `data/processed/*`).
Third-party data, software, and scientific references are recorded in
`THIRD_PARTY_NOTICES.md`.

**NASA LANCE / FIRMS acknowledgment.** The project uses data and imagery from
NASA's Land, Atmosphere Near real-time Capability for EOSDIS (LANCE) system,
part of NASA's Earth Science Data and Information System (ESDIS), and the Fire
Information for Resource Management System (FIRMS). The required acknowledgment
is reproduced in full in `THIRD_PARTY_NOTICES.md`, together with the MODIS
Collection 6.1 and VIIRS 375 m dataset DOIs.

**Natural Earth attribution.** Administrative boundaries are derived from
Natural Earth 1:10m Cultural Vectors (`ne_10m_admin_0_countries`,
`ne_10m_admin_1_states_provinces`, v5.1.2), which are public domain (CC0). The
simplified delta-encoded display geometry in `SOUTH_AMERICA_GEO`
(`app/static/app.js`) is a cartographic derivative, not a survey-grade
boundary product.

**Non-Endorsement.** Redistribution and analysis of derived NASA FIRMS data do
not imply sponsorship, endorsement, or official approval by NASA, LANCE,
ESDIS, NOAA, or the U.S. Government. NASA does not endorse this independent
challenge submission. No NASA seal, emblem, or graphical wordmark is used.

**Not for tactical fire management.** The application is a scientific,
retrospective, and climatological visualization. Orbital revisit intervals,
cloud cover, and spatial aggregation on 10 km and 50 km grids make it
unsuitable for tactical wildfire suppression, real-time emergency dispatch,
evacuation routing, or life-and-property safety decisions. It must not be
used as an operational fire-management system.
