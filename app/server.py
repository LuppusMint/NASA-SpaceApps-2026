#!/usr/bin/env python3
"""Bilingual interactive burning-calendar web application.

The module deliberately uses only the standard library plus DuckDB (and
PyArrow, which is a DuckDB dependency here).  It provides:

* :func:`create_app` — a clean WSGI application compatible with
  ``httpx.WSGITransport``, ``wsgiref`` and the bundled multithreaded server.
* :func:`create_server` — a ``ThreadingMixIn``-based WSGI server for the CLI.
* A small set of JSON endpoints backed by the Parquet artefacts produced by
  the preprocessing and modelling pipelines, plus a one-click UTF-8 CSV export.
* Static-file serving for the fully offline front end in ``app/static``.

DuckDB connections are thread-local and read the Parquet files directly
(``read_parquet``), with ``threads=2`` and ``memory_limit='1GB'``.  Queries
project only the columns they need and push ``WHERE`` filters into DuckDB.

Run from the repository root::

    python -m app.server --processed-dir data/processed \
        --host 127.0.0.1 --port 8080
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
import datetime as dt
import io
import json
import math
import mimetypes
import re
import statistics
import threading
import time
import traceback
from http import HTTPStatus
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qs, unquote
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

import duckdb

try:  # pragma: no cover - import is only used for typing/version reporting
    import pyarrow as _pa  # noqa: F401
except Exception:  # noqa: BLE001 - optional metadata only
    _pa = None


APP_VERSION = "3.0.0"
STATIC_ROOT = Path(__file__).resolve().parent / "static"
API_CACHE_CONTROL = "no-store"
DEFAULT_LIMIT = 10000
MAX_LIMIT = 20000
#: Nominal catalogue window documented by the FIRMS archive release.
CATALOG_MIN_YM = "2000-11"
CATALOG_MAX_YM = "2026-11"
LATENCY_BUDGET_MS = 50.0
DEFAULT_YEAR_MIN = 2000
DEFAULT_YEAR_MAX = 2026
YM_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
ISO3_RE = re.compile(r"^[A-Z]{3}$")
SOUTH_AMERICA = "SOUTH_AMERICA"

#: Files the API can consume.  The first group is required for a healthy
#: installation; the calendar files are optional because the API can rebuild
#: their outputs from the harmonized monthly cubes.
PARQUET_FILES: tuple[str, ...] = (
    "harmonized_admin1_monthly.parquet",
    "harmonized_country_monthly.parquet",
    "harmonized_south_america_monthly.parquet",
    "harmonized_cell50km_monthly.parquet",
    "cube_cell10km_monthly.parquet",
    "static_thermal_cells.parquet",
    "calendar_climatology.parquet",
    "calendar_matrix_admin1.parquet",
    "calendar_matrix_country.parquet",
    "calendar_trends.parquet",
    "calendar_early_onset_90d.parquet",
    "cube_country_monthly.parquet",
    "cube_admin1_monthly.parquet",
)

CORE_PARQUETS: tuple[str, ...] = (
    "harmonized_admin1_monthly.parquet",
    "harmonized_country_monthly.parquet",
    "harmonized_south_america_monthly.parquet",
    "harmonized_cell50km_monthly.parquet",
    "static_thermal_cells.parquet",
)

MATURITY_RANK = {
    "": 0,
    "STANDARD_SCIENCE": 1,
    "NRT_PRELIMINARY": 2,
    "PARTIAL_MONTH": 3,
}

SEVERITY_ORDER = {
    "LOW_SAMPLE": 0,
    "LOW_NORMAL": 1,
    "MODERATE": 2,
    "HIGH_UNUSUAL": 3,
    "CRITICAL_EXTREME": 4,
}


# ---------------------------------------------------------------------------
# Errors and small utilities
# ---------------------------------------------------------------------------


class HttpError(Exception):
    """An error that maps directly to an HTTP response."""

    status = HTTPStatus.BAD_REQUEST
    code = "bad_request"

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        if status is not None:
            self.status = int(status)
        if code is not None:
            self.code = str(code)


class BadRequest(HttpError):
    status = HTTPStatus.BAD_REQUEST
    code = "bad_request"


class NotFound(HttpError):
    status = HTTPStatus.NOT_FOUND
    code = "not_found"


class MethodNotAllowed(HttpError):
    status = HTTPStatus.METHOD_NOT_ALLOWED
    code = "method_not_allowed"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _clean_json(value: Any) -> Any:
    """Return a JSON-safe copy of ``value`` (NaN/Inf become null)."""

    if isinstance(value, Mapping):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.name
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _clean_json(value.item())
        except Exception:  # noqa: BLE001 - exotic scalar
            return str(value)
    return value


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        _clean_json(payload),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and not math.isfinite(value):
        return default
    text = str(value)
    return text if text else default


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default convention)."""

    finite = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not finite:
        return 0.0
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * max(0.0, min(1.0, float(q)))
    lower = int(math.floor(position))
    upper = min(lower + 1, len(finite) - 1)
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def _mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else 0.0


def _std(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.pstdev(finite) if len(finite) > 1 else 0.0


def _severity(value: float, p25: float, p75: float, p90: float, z: float) -> str:
    if value > p90 or z > 2.0:
        return "CRITICAL_EXTREME"
    if value > p75:
        return "HIGH_UNUSUAL"
    if value > p25:
        return "MODERATE"
    return "LOW_NORMAL"


def _ym(year: int, month: int) -> str:
    return f"{int(year):04d}-{int(month):02d}"


def _normalize_ym(value: Any, year: int, month: int) -> str:
    """Accept ``YYYY-MM``, ``YYYYMM`` and ``YYYY-M`` spellings."""

    text = _text(value)
    match = re.match(r"^(\d{4})[-/]?(\d{1,2})$", text)
    if match:
        candidate_month = int(match.group(2))
        if 1 <= candidate_month <= 12:
            return f"{int(match.group(1)):04d}-{candidate_month:02d}"
    return _ym(year, month)


def _safe_ym(value: str, *, name: str) -> str:
    if not YM_RE.match(value or ""):
        raise BadRequest(f"'{name}' must use the YYYY-MM format (got {value!r}).")
    return value


def _parse_bbox(raw: str) -> tuple[float, float, float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 4:
        raise BadRequest(
            "bbox must contain four comma-separated values: "
            "min_lat,max_lat,min_lon,max_lon."
        )
    try:
        min_lat, max_lat, min_lon, max_lon = (float(part) for part in parts)
    except ValueError as exc:
        raise BadRequest("bbox values must be numeric.") from exc
    if not all(math.isfinite(item) for item in (min_lat, max_lat, min_lon, max_lon)):
        raise BadRequest("bbox values must be finite.")
    if not (-90.0 <= min_lat < max_lat <= 90.0):
        raise BadRequest("bbox latitude range is invalid or inverted (min_lat < max_lat).")
    if not (-180.0 <= min_lon < max_lon <= 180.0):
        raise BadRequest("bbox longitude range is invalid or inverted (min_lon < max_lon).")
    return min_lat, max_lat, min_lon, max_lon


def _bbox_from_radius(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    dlat = radius_km / 111.32
    cos_lat = max(math.cos(math.radians(lat)), 0.01)
    dlon = radius_km / (111.32 * cos_lat)
    return (
        max(-90.0, lat - dlat),
        min(90.0, lat + dlat),
        max(-180.0, lon - dlon),
        min(180.0, lon + dlon),
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius * math.asin(min(1.0, math.sqrt(max(0.0, a))))


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------


class Request:
    """A tiny immutable view over a WSGI environ."""

    def __init__(self, environ: Mapping[str, Any]) -> None:
        self.environ = environ
        self.method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        self.path = str(environ.get("PATH_INFO") or "/") or "/"
        raw_query = str(environ.get("QUERY_STRING") or "")
        self.query: dict[str, list[str]] = parse_qs(raw_query, keep_blank_values=True)

    def has(self, name: str) -> bool:
        return name in self.query

    def one(self, name: str, default: str | None = None) -> str | None:
        values = self.query.get(name)
        if not values:
            return default
        return values[0]

    def int_param(
        self,
        name: str,
        *,
        default: int | None = None,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int | None:
        raw = self.one(name)
        if raw is None or raw == "":
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise BadRequest(f"'{name}' must be an integer.") from exc
        if minimum is not None and value < minimum:
            raise BadRequest(f"'{name}' must be >= {minimum}.")
        if maximum is not None and value > maximum:
            raise BadRequest(f"'{name}' must be <= {maximum}.")
        return value

    def float_param(
        self,
        name: str,
        *,
        default: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float | None:
        raw = self.one(name)
        if raw is None or raw == "":
            return default
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise BadRequest(f"'{name}' must be numeric.") from exc
        if not math.isfinite(value):
            raise BadRequest(f"'{name}' must be finite.")
        if minimum is not None and value < minimum:
            raise BadRequest(f"'{name}' must be >= {minimum}.")
        if maximum is not None and value > maximum:
            raise BadRequest(f"'{name}' must be <= {maximum}.")
        return value

    def bool_param(self, name: str, *, default: bool = False) -> bool:
        raw = self.one(name)
        if raw is None or raw == "":
            return default
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise BadRequest(f"'{name}' must be a boolean (true/false).")


# ---------------------------------------------------------------------------
# Location filter
# ---------------------------------------------------------------------------


class Location:
    """Validated spatial filter (bbox or lat/lon/radius)."""

    def __init__(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        lat: float | None = None,
        lon: float | None = None,
        radius_km: float | None = None,
    ) -> None:
        self.bbox = bbox
        self.lat = lat
        self.lon = lon
        self.radius_km = radius_km

    @property
    def active(self) -> bool:
        return self.bbox is not None or (
            self.lat is not None and self.lon is not None and self.radius_km is not None
        )

    @property
    def effective_bbox(self) -> tuple[float, float, float, float] | None:
        if self.bbox is not None:
            return self.bbox
        if self.lat is not None and self.lon is not None and self.radius_km is not None:
            return _bbox_from_radius(self.lat, self.lon, self.radius_km)
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox) if self.bbox else None,
            "lat": self.lat,
            "lon": self.lon,
            "radius_km": self.radius_km,
        }


# ---------------------------------------------------------------------------
# WSGI application
# ---------------------------------------------------------------------------


class FireRelayApplication:
    """The Fire Relay South America WSGI application."""

    def __init__(self, processed_dir: Path | str) -> None:
        self.processed = Path(processed_dir).expanduser().resolve()
        self.static_root = STATIC_ROOT.resolve()
        self._local = threading.local()
        self._schema_lock = threading.Lock()
        self._schema_cache: dict[tuple[str, int], list[str]] = {}
        self._cache_lock = threading.Lock()
        self._cache: dict[tuple[Any, ...], Any] = {}
        self._response_lock = threading.Lock()
        self._response_cache: OrderedDict[
            tuple[Any, ...], tuple[float, int, list[tuple[str, str]], bytes]
        ] = OrderedDict()
        self._response_cache_ttl = 600.0
        self._response_cache_size = 48
        self._started_at = _utc_now()
        self._duckdb_version = str(getattr(duckdb, "__version__", "unknown"))
        self.qc_summary = self._read_json(self.processed / "qc_cleaning_summary.json")
        self._country_ranges = self._compute_country_ranges()
        self._admin1_ranges = self._compute_admin1_ranges()
        self.catalog = self._build_catalog()
        self._static_count = self._count_static_cells()

    # ------------------------------------------------------------------
    # DuckDB access
    # ------------------------------------------------------------------

    def _connection(self) -> duckdb.DuckDBPyConnection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = duckdb.connect(database=":memory:")
            connection.execute("SET threads = 2")
            connection.execute("SET memory_limit = '1GB'")
            connection.execute("SET preserve_insertion_order = false")
            self._local.connection = connection
        return connection

    @staticmethod
    def _parquet(path: Path) -> str:
        escaped = str(path).replace("'", "''")
        return f"read_parquet('{escaped}')"

    def _fetch(self, path: Path, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        connection = self._connection()
        try:
            cursor = connection.execute(sql, list(params))
            columns = [description[0] for description in (cursor.description or [])]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except duckdb.Error as exc:
            raise RuntimeError(f"DuckDB query failed for {path.name}: {exc}") from exc

    def _columns(self, path: Path) -> list[str]:
        try:
            stat = path.stat()
            key = (str(path), int(stat.st_mtime_ns))
        except OSError:
            return []
        with self._schema_lock:
            cached = self._schema_cache.get(key)
        if cached is not None:
            return list(cached)
        if not path.is_file():
            return []
        rows = self._fetch(path, f"DESCRIBE SELECT * FROM {self._parquet(path)}")
        columns = [str(row.get("column_name")) for row in rows if row.get("column_name")]
        with self._schema_lock:
            self._schema_cache[key] = columns
        return list(columns)

    def _first_existing(self, *names: str) -> Path | None:
        for name in names:
            candidate = self.processed / name
            if candidate.is_file():
                return candidate
        return None

    def _require(self, *names: str) -> Path:
        path = self._first_existing(*names)
        if path is None:
            raise NotFound(
                "The requested data asset is not available in this installation "
                f"({', '.join(names)}).",
                code="asset_not_found",
            )
        return path

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    # ------------------------------------------------------------------
    # Catalogue
    # ------------------------------------------------------------------

    def _compute_country_ranges(self) -> dict[str, dict[str, Any]]:
        path = self._first_existing(
            "harmonized_country_monthly.parquet",
            "calendar_matrix_country.parquet",
            "cube_country_monthly.parquet",
        )
        if path is None:
            return {}
        try:
            rows = self._fetch(
                path,
                f"""
                SELECT iso_a3,
                       MIN(year_month) AS min_year_month,
                       MAX(year_month) AS max_year_month,
                       COUNT(*) AS n_rows
                FROM {self._parquet(path)}
                WHERE iso_a3 IS NOT NULL
                GROUP BY iso_a3
                """,
            )
        except Exception:  # noqa: BLE001 - catalogue enrichment is best effort
            return {}
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            iso = _text(row.get("iso_a3")).upper()
            if not iso:
                continue
            result[iso] = {
                "min_year_month": _text(row.get("min_year_month")) or None,
                "max_year_month": _text(row.get("max_year_month")) or None,
                "n_rows": int(_num(row.get("n_rows"))),
            }
        return result

    def _compute_admin1_ranges(self) -> dict[tuple[str, str], dict[str, Any]]:
        path = self._first_existing(
            "harmonized_admin1_monthly.parquet",
            "calendar_matrix_admin1.parquet",
            "cube_admin1_monthly.parquet",
        )
        if path is None:
            return {}
        try:
            rows = self._fetch(
                path,
                f"""
                SELECT iso_a3, admin1_name,
                       MIN(year_month) AS min_year_month,
                       MAX(year_month) AS max_year_month
                FROM {self._parquet(path)}
                WHERE iso_a3 IS NOT NULL AND admin1_name IS NOT NULL
                GROUP BY iso_a3, admin1_name
                """,
            )
        except Exception:  # noqa: BLE001 - catalogue enrichment is best effort
            return {}
        result: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            iso = _text(row.get("iso_a3")).upper()
            admin1 = _text(row.get("admin1_name"))
            if not iso or not admin1:
                continue
            result[(iso, admin1)] = {
                "min_year_month": _text(row.get("min_year_month")) or None,
                "max_year_month": _text(row.get("max_year_month")) or None,
            }
        return result

    def _build_catalog(self) -> dict[str, Any]:
        raw = self._read_json(self.processed / "geography_catalog.json")
        countries: list[dict[str, Any]] = []
        admin1_flat: list[dict[str, Any]] = []
        if raw:
            for entry in raw.get("countries") or []:
                if not isinstance(entry, dict):
                    continue
                country = dict(entry)
                country["iso_a3"] = _text(country.get("iso_a3")).upper()
                country["country_name"] = _text(country.get("country_name"))
                children = [
                    dict(child)
                    for child in (country.get("admin1") or [])
                    if isinstance(child, dict)
                ]
                country["admin1"] = children
                country["admin1_count"] = int(
                    _num(country.get("admin1_count"), len(children))
                )
                country["historical_active_cells_10km"] = int(
                    _num(country.get("historical_active_cells_10km"))
                )
                range_info = self._country_ranges.get(country["iso_a3"], {})
                country["min_year_month"] = range_info.get("min_year_month")
                country["max_year_month"] = range_info.get("max_year_month")
                for child in children:
                    child["country_name"] = _text(
                        child.get("country_name"), country["country_name"]
                    )
                    child["iso_a3"] = _text(child.get("iso_a3"), country["iso_a3"]).upper()
                    child["admin1_name"] = _text(child.get("admin1_name"))
                    child["historical_active_cells_10km"] = int(
                        _num(child.get("historical_active_cells_10km"))
                    )
                    child_range = self._admin1_ranges.get(
                        (child["iso_a3"], child["admin1_name"]), {}
                    )
                    child["min_year_month"] = child_range.get("min_year_month")
                    child["max_year_month"] = child_range.get("max_year_month")
                    admin1_flat.append(dict(child))
                countries.append(country)
        else:
            # Fall back to the harmonized country cube so the UI remains usable
            # even when the geography catalogue has not been generated.
            path = self._first_existing(
                "harmonized_country_monthly.parquet",
                "calendar_matrix_country.parquet",
                "cube_country_monthly.parquet",
            )
            if path is not None:
                try:
                    rows = self._fetch(
                        path,
                        f"""
                        SELECT iso_a3, MAX(country_name) AS country_name
                        FROM {self._parquet(path)}
                        WHERE iso_a3 IS NOT NULL
                        GROUP BY iso_a3
                        ORDER BY iso_a3
                        """,
                    )
                except Exception:  # noqa: BLE001
                    rows = []
                for row in rows:
                    iso = _text(row.get("iso_a3")).upper()
                    if not iso:
                        continue
                    range_info = self._country_ranges.get(iso, {})
                    countries.append(
                        {
                            "country_name": _text(row.get("country_name"), iso),
                            "iso_a3": iso,
                            "bbox": None,
                            "intersects_sama": False,
                            "historical_active_cells_10km": 0,
                            "admin1_count": 0,
                            "admin1": [],
                            "min_year_month": range_info.get("min_year_month"),
                            "max_year_month": range_info.get("max_year_month"),
                        }
                    )
        if raw:
            admin1_raw = raw.get("admin1")
            if isinstance(admin1_raw, list) and admin1_raw:
                admin1_flat = []
                for entry in admin1_raw:
                    if not isinstance(entry, dict):
                        continue
                    child = dict(entry)
                    child["iso_a3"] = _text(child.get("iso_a3")).upper()
                    child["admin1_name"] = _text(child.get("admin1_name"))
                    child["country_name"] = _text(child.get("country_name"))
                    child_range = self._admin1_ranges.get(
                        (child["iso_a3"], child["admin1_name"]), {}
                    )
                    child["min_year_month"] = child_range.get("min_year_month")
                    child["max_year_month"] = child_range.get("max_year_month")
                    admin1_flat.append(child)

        # If the catalogue stores Admin-1 only in the flat list, mirror it
        # under each country so the hierarchy is always complete.
        for country in countries:
            if not country.get("admin1"):
                children = [
                    dict(item)
                    for item in admin1_flat
                    if item.get("iso_a3") == country.get("iso_a3")
                ]
                if children:
                    country["admin1"] = children
                    country["admin1_count"] = len(children)
        countries.sort(key=lambda item: (item.get("country_name") or "", item.get("iso_a3") or ""))
        admin1_flat.sort(
            key=lambda item: (
                item.get("country_name") or "",
                item.get("admin1_name") or "",
            )
        )
        totals = dict(raw.get("totals") or {}) if raw else {}
        totals["countries"] = len(countries)
        totals["admin1"] = len(admin1_flat)
        totals["static_cell_count"] = int(
            _num((self.qc_summary or {}).get("static_cell_count"), 0)
        )
        min_values = [c["min_year_month"] for c in countries if c.get("min_year_month")]
        max_values = [c["max_year_month"] for c in countries if c.get("max_year_month")]
        time_range = {
            "min_year_month": min(min_values) if min_values else None,
            "max_year_month": max(max_values) if max_values else None,
        }
        bboxes = [c.get("bbox") for c in countries if isinstance(c.get("bbox"), list)]
        south_bbox = None
        if bboxes:
            south_bbox = [
                min(_num(b[0]) for b in bboxes),
                max(_num(b[1]) for b in bboxes),
                min(_num(b[2]) for b in bboxes),
                max(_num(b[3]) for b in bboxes),
            ]
        return {
            "generated_at_utc": (raw or {}).get("generated_at_utc"),
            "bbox_order": (raw or {}).get(
                "bbox_order", ["min_lat", "max_lat", "min_lon", "max_lon"]
            ),
            "countries": countries,
            "admin1": admin1_flat,
            "totals": totals,
            "south_america_bbox": south_bbox,
            "time_range": time_range,
        }

    def _count_static_cells(self) -> int:
        path = self.processed / "static_thermal_cells.parquet"
        if not path.is_file():
            return int(_num((self.qc_summary or {}).get("static_cell_count"), 0))
        try:
            rows = self._fetch(
                path, f"SELECT COUNT(*) AS n FROM {self._parquet(path)}"
            )
            return int(_num(rows[0].get("n"))) if rows else 0
        except Exception:  # noqa: BLE001
            return int(_num((self.qc_summary or {}).get("static_cell_count"), 0))

    def _catalog_payload(self) -> dict[str, Any]:
        return {
            "generated_at_utc": _utc_now(),
            "source_generated_at_utc": self.catalog.get("generated_at_utc"),
            "bbox_order": self.catalog.get("bbox_order"),
            "countries": self.catalog.get("countries", []),
            "admin1": self.catalog.get("admin1", []),
            "totals": self.catalog.get("totals", {}),
            "south_america_bbox": self.catalog.get("south_america_bbox"),
            "time_range": self.catalog.get("time_range"),
            "min_year_month": (self.catalog.get("time_range") or {}).get("min_year_month"),
            "max_year_month": (self.catalog.get("time_range") or {}).get("max_year_month"),
            "static_cell_count": self._static_count,
            "default_country": SOUTH_AMERICA,
        }

    def _known_country(self, iso: str) -> bool:
        if iso == SOUTH_AMERICA:
            return True
        return any(country.get("iso_a3") == iso for country in self.catalog.get("countries", []))

    def _known_admin1(self, iso: str, admin1: str) -> bool:
        return any(
            entry.get("iso_a3") == iso and entry.get("admin1_name") == admin1
            for entry in self.catalog.get("admin1", [])
        )

    def _catalog_country(self, iso: str) -> dict[str, Any] | None:
        for country in self.catalog.get("countries", []):
            if country.get("iso_a3") == iso:
                return country
        return None

    def _resolve_admin1_name(self, iso_a3: Any, lat: Any, lon: Any) -> str:
        """Resolve the best Admin-1 name for a cell centre.

        When a cube does not carry ``admin1_name`` (for example the 50 km
        harmonized cube), the name is inferred from the Admin-1 bounding boxes
        in ``geography_catalog.json``.  The containing box is preferred; if
        several contain the point the one with the closest bbox centre wins.
        When no box contains the point, the closest bbox centre for that
        country is used as a deterministic fallback.
        """
        iso = _text(iso_a3).upper()
        if not iso:
            return ""
        try:
            latitude = float(lat)
            longitude = float(lon)
        except (TypeError, ValueError):
            return ""
        if not (math.isfinite(latitude) and math.isfinite(longitude)):
            return ""
        candidates: list[tuple[str, float, bool]] = []
        for entry in self.catalog.get("admin1", []):
            if _text(entry.get("iso_a3")).upper() != iso:
                continue
            bbox = entry.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            min_lat, max_lat, min_lon, max_lon = (_num(value) for value in bbox)
            if not all(
                math.isfinite(value)
                for value in (min_lat, max_lat, min_lon, max_lon)
            ):
                continue
            name = _text(entry.get("admin1_name"))
            if not name:
                continue
            center_lat = (min_lat + max_lat) / 2.0
            center_lon = (min_lon + max_lon) / 2.0
            distance = (center_lat - latitude) ** 2 + (center_lon - longitude) ** 2
            contains = min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon
            candidates.append((name, distance, contains))
        if not candidates:
            return ""
        containing = [item for item in candidates if item[2]]
        pool = containing or candidates
        pool.sort(key=lambda item: (item[1], item[0]))
        return pool[0][0]

    # ------------------------------------------------------------------
    # Parameter models
    # ------------------------------------------------------------------

    def _parse_location(self, request: Request) -> Location:
        raw_bbox = request.one("bbox")
        lat = request.float_param("lat", minimum=-90.0, maximum=90.0)
        lon = request.float_param("lon", minimum=-180.0, maximum=180.0)
        radius = request.float_param("radius_km", minimum=0.1, maximum=20000.0)
        bbox = _parse_bbox(raw_bbox) if raw_bbox not in (None, "") else None
        point_requested = any(value is not None for value in (lat, lon, radius))
        if bbox is not None and point_requested:
            raise BadRequest("Use either 'bbox' or 'lat'/'lon'/'radius_km', not both.")
        if point_requested:
            if lat is None or lon is None or radius is None:
                raise BadRequest(
                    "The point filter requires all three parameters: lat, lon and radius_km."
                )
            return Location(lat=lat, lon=lon, radius_km=radius)
        return Location(bbox=bbox)

    def _calendar_params(self, request: Request) -> dict[str, Any]:
        iso_raw = request.one("iso_a3", "")
        iso = (iso_raw or "").strip().upper()
        if iso in {"", "*", "ALL", "SA", SOUTH_AMERICA}:
            iso = SOUTH_AMERICA
        elif not ISO3_RE.match(iso):
            raise BadRequest("'iso_a3' must be a three-letter ISO code or SOUTH_AMERICA.")
        admin1 = (request.one("admin1_name", "") or "").strip()
        metric = (request.one("metric", "srehi") or "srehi").strip().lower()
        if metric not in {"srehi", "frp_mw"}:
            raise BadRequest("'metric' must be 'srehi' or 'frp_mw'.")
        series_mode = (request.one("series_mode", "harmonized") or "harmonized").strip().lower()
        if series_mode not in {"harmonized", "raw"}:
            raise BadRequest("'series_mode' must be 'harmonized' or 'raw'.")
        start_raw = request.one("start_ym")
        end_raw = request.one("end_ym")
        start_ym = _safe_ym(start_raw, name="start_ym") if start_raw else None
        end_ym = _safe_ym(end_raw, name="end_ym") if end_raw else None
        if start_ym and end_ym and start_ym > end_ym:
            raise BadRequest("'start_ym' must be earlier than or equal to 'end_ym'.")
        location = self._parse_location(request)
        return {
            "iso_a3": iso,
            "admin1_name": admin1,
            "metric": metric,
            "series_mode": series_mode,
            "start_ym": start_ym,
            "end_ym": end_ym,
            "range_start": start_ym or "0000-00",
            "range_end": end_ym or "9999-99",
            "location": location,
        }

    # ------------------------------------------------------------------
    # Series normalization and geo queries
    # ------------------------------------------------------------------

    def _series_row(self, raw: Mapping[str, Any]) -> dict[str, Any] | None:
        year_value = raw.get("year")
        month_value = raw.get("month")
        try:
            year = int(year_value)
            month = int(month_value)
        except (TypeError, ValueError):
            return None
        if not (1 <= month <= 12):
            return None
        year_month = _normalize_ym(raw.get("year_month"), year, month)
        h_srehi = _num(raw.get("harmonized_srehi"), _num(raw.get("overpass_event_count")))
        h_frp = _num(raw.get("harmonized_frp_mw"), _num(raw.get("raw_frp_mw")))
        lo_srehi = _num(raw.get("harmonized_srehi_lo90"), h_srehi)
        hi_srehi = _num(raw.get("harmonized_srehi_hi90"), h_srehi)
        lo_frp = _num(raw.get("harmonized_frp_lo90"), h_frp)
        hi_frp = _num(raw.get("harmonized_frp_hi90"), h_frp)
        raw_srehi = _num(
            raw.get("overpass_event_count"),
            _num(raw.get("clean_det_count"), _num(raw.get("raw_all_det_count"))),
        )
        raw_frp = _num(raw.get("raw_frp_mw"), _num(raw.get("frp_sum_mw")))
        maturity = _text(raw.get("data_maturity"), "STANDARD_SCIENCE").upper()
        regime = _text(raw.get("constellation_regime"))
        return {
            "year": year,
            "month": month,
            "year_month": year_month,
            "country_name": _text(raw.get("country_name")),
            "iso_a3": _text(raw.get("iso_a3")).upper(),
            "admin1_name": _text(raw.get("admin1_name")),
            "harmonized_srehi": h_srehi,
            "lo90_srehi": lo_srehi,
            "hi90_srehi": max(hi_srehi, h_srehi),
            "harmonized_frp_mw": h_frp,
            "lo90_frp_mw": lo_frp,
            "hi90_frp_mw": max(hi_frp, h_frp),
            "raw_srehi": raw_srehi,
            "raw_frp_mw": raw_frp,
            "m1_srehi": _num(raw.get("m1_srehi")),
            "m2_srehi": _num(raw.get("m2_srehi")),
            "m1_frp_mw": _num(raw.get("m1_frp_mw")),
            "m2_frp_mw": _num(raw.get("m2_frp_mw")),
            "nrt_det_count": int(_num(raw.get("nrt_det_count"))),
            "sp_det_count": int(_num(raw.get("sp_det_count"))),
            "clean_det_count": int(_num(raw.get("clean_det_count"))),
            "raw_all_det_count": int(_num(raw.get("raw_all_det_count"))),
            "static_excluded_det_count": int(_num(raw.get("static_excluded_det_count"))),
            "sama_suspect_det_count": int(_num(raw.get("sama_suspect_det_count"))),
            "sama_night_det_count": int(_num(raw.get("sama_suspect_det_count"))),
            "is_in_sama": _as_bool(raw.get("is_in_sama"))
            or _as_bool(raw.get("intersects_sama")),
            "intersects_sama": _as_bool(raw.get("intersects_sama"))
            or _as_bool(raw.get("is_in_sama")),
            "is_partial_month": _as_bool(raw.get("is_partial_month")),
            "data_maturity": maturity if maturity in MATURITY_RANK else "STANDARD_SCIENCE",
            "constellation_regime": regime,
            "z_score_srehi": _num(raw.get("z_score_srehi")),
            "pct_of_median_srehi": _num(raw.get("pct_of_median_srehi")),
            "severity_tier": _text(raw.get("severity_tier")),
            "z_score_frp": _num(raw.get("z_score_frp")),
            "pct_of_median_frp": _num(raw.get("pct_of_median_frp")),
            "severity_tier_frp": _text(raw.get("severity_tier_frp")),
        }

    @staticmethod
    def _merge_series(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        sum_keys = (
            "harmonized_srehi",
            "lo90_srehi",
            "hi90_srehi",
            "harmonized_frp_mw",
            "lo90_frp_mw",
            "hi90_frp_mw",
            "raw_srehi",
            "raw_frp_mw",
            "m1_srehi",
            "m2_srehi",
            "m1_frp_mw",
            "m2_frp_mw",
            "nrt_det_count",
            "sp_det_count",
            "clean_det_count",
            "raw_all_det_count",
            "static_excluded_det_count",
            "sama_suspect_det_count",
        )
        for row in rows:
            key = row["year_month"]
            target = merged.get(key)
            if target is None:
                merged[key] = dict(row)
                continue
            for name in sum_keys:
                target[name] = _num(target.get(name)) + _num(row.get(name))
            target["is_in_sama"] = bool(target.get("is_in_sama")) or bool(row.get("is_in_sama"))
            target["intersects_sama"] = bool(target.get("intersects_sama")) or bool(
                row.get("intersects_sama")
            )
            target["is_partial_month"] = bool(target.get("is_partial_month")) or bool(
                row.get("is_partial_month")
            )
            if MATURITY_RANK.get(row.get("data_maturity"), 0) > MATURITY_RANK.get(
                target.get("data_maturity"), 0
            ):
                target["data_maturity"] = row.get("data_maturity")
            if not target.get("country_name") and row.get("country_name"):
                target["country_name"] = row.get("country_name")
            if not target.get("admin1_name") and row.get("admin1_name"):
                target["admin1_name"] = row.get("admin1_name")
            if not target.get("constellation_regime") and row.get("constellation_regime"):
                target["constellation_regime"] = row.get("constellation_regime")
        return sorted(merged.values(), key=lambda row: (row["year"], row["month"]))

    def _load_geo_rows(self, params: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
        iso = str(params["iso_a3"])
        admin1 = str(params["admin1_name"])
        if iso == SOUTH_AMERICA:
            path = self._first_existing(
                "harmonized_south_america_monthly.parquet",
                "harmonized_country_monthly.parquet",
                "calendar_matrix_country.parquet",
            )
            if path is None:
                raise NotFound(
                    "No continental South America series is available.",
                    code="asset_not_found",
                )
            sql = f"SELECT * FROM {self._parquet(path)} ORDER BY year, month"
            rows = self._fetch(path, sql)
            return rows, path.name, "South America"
        if admin1:
            path = self._first_existing(
                "harmonized_admin1_monthly.parquet",
                "calendar_matrix_admin1.parquet",
                "cube_admin1_monthly.parquet",
            )
            if path is None:
                raise NotFound(
                    "No Admin-1 monthly series is available.",
                    code="asset_not_found",
                )
            sql = (
                f"SELECT * FROM {self._parquet(path)} "
                "WHERE iso_a3 = ? AND admin1_name = ? ORDER BY year, month"
            )
            rows = self._fetch(path, sql, [iso, admin1])
            country_name = _text(rows[0].get("country_name")) if rows else ""
            return rows, path.name, country_name
        path = self._first_existing(
            "harmonized_country_monthly.parquet",
            "calendar_matrix_country.parquet",
            "cube_country_monthly.parquet",
        )
        if path is None:
            raise NotFound(
                "No country monthly series is available.",
                code="asset_not_found",
            )
        sql = (
            f"SELECT * FROM {self._parquet(path)} "
            "WHERE iso_a3 = ? ORDER BY year, month"
        )
        rows = self._fetch(path, sql, [iso])
        country_name = _text(rows[0].get("country_name")) if rows else ""
        return rows, path.name, country_name

    def _geo_series(self, params: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        raw_rows, source, country_name = self._load_geo_rows(params)
        normalized = [
            row
            for row in (self._series_row(item) for item in raw_rows)
            if row is not None
        ]
        series = self._merge_series(normalized)
        series = [
            row
            for row in series
            if params["range_start"] <= row["year_month"] <= params["range_end"]
        ]
        if not series:
            iso = str(params["iso_a3"])
            admin1 = str(params["admin1_name"])
            if iso != SOUTH_AMERICA and not self._known_country(iso):
                raise NotFound(f"Unknown country ISO code: {iso}.", code="unknown_country")
            if admin1 and not self._known_admin1(iso, admin1):
                raise NotFound(
                    f"Unknown Admin-1 unit for {iso}: {admin1}.", code="unknown_admin1"
                )
            raise NotFound(
                "No harmonized observations match the requested selection.",
                code="no_rows",
            )
        # Merge z-score/severity metadata from the calendar matrices when
        # available.  This keeps the API faithful to the frozen analytics.
        matrix_path = self._first_existing(
            "calendar_matrix_admin1.parquet" if params["admin1_name"] else "calendar_matrix_country.parquet"
        )
        if matrix_path is not None and params["iso_a3"] != SOUTH_AMERICA:
            try:
                if params["admin1_name"]:
                    matrix_rows = self._fetch(
                        matrix_path,
                        f"SELECT * FROM {self._parquet(matrix_path)} "
                        "WHERE iso_a3 = ? AND admin1_name = ?",
                        [params["iso_a3"], params["admin1_name"]],
                    )
                else:
                    matrix_rows = self._fetch(
                        matrix_path,
                        f"SELECT * FROM {self._parquet(matrix_path)} WHERE iso_a3 = ?",
                        [params["iso_a3"]],
                    )
            except Exception:  # noqa: BLE001 - metadata enrichment is optional
                matrix_rows = []
            lookup = {}
            for row in matrix_rows:
                if row.get("year_month") is None:
                    continue
                lookup[
                    _normalize_ym(
                        row.get("year_month"),
                        int(_num(row.get("year"))),
                        int(_num(row.get("month"))),
                    )
                ] = row
            for row in series:
                meta = lookup.get(row["year_month"])
                if not meta:
                    continue
                for name in (
                    "z_score_srehi",
                    "pct_of_median_srehi",
                    "severity_tier",
                    "z_score_frp",
                    "pct_of_median_frp",
                    "severity_tier_frp",
                ):
                    if meta.get(name) is not None:
                        row[name] = meta.get(name)
        if not country_name:
            country = self._catalog_country(str(params["iso_a3"]))
            country_name = _text((country or {}).get("country_name"), str(params["iso_a3"]))
        return series, {
            "source": source,
            "country_name": country_name,
            "level": "continental"
            if params["iso_a3"] == SOUTH_AMERICA
            else ("admin1" if params["admin1_name"] else "country"),
        }

    # ------------------------------------------------------------------
    # Cell / local-area series
    # ------------------------------------------------------------------

    def _cell_where(
        self,
        *,
        location: Location,
        iso: str,
        year_month: str | None = None,
        year: int | None = None,
        range_start: str | None = None,
        range_end: str | None = None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if year_month:
            clauses.append("year_month = ?")
            params.append(year_month)
        elif year is not None:
            clauses.append("year = ?")
            params.append(int(year))
        else:
            if range_start:
                clauses.append("year_month >= ?")
                params.append(range_start)
            if range_end:
                clauses.append("year_month <= ?")
                params.append(range_end)
        if iso != SOUTH_AMERICA and iso:
            clauses.append("iso_a3 = ?")
            params.append(iso)
        bbox = location.effective_bbox
        if bbox:
            min_lat, max_lat, min_lon, max_lon = bbox
            clauses.append("mean_lat BETWEEN ? AND ?")
            params.extend([min_lat, max_lat])
            clauses.append("mean_lon BETWEEN ? AND ?")
            params.extend([min_lon, max_lon])
        if location.lat is not None and location.lon is not None and location.radius_km is not None:
            haversine = (
                "6371.0088 * 2.0 * asin(least(1.0, sqrt("
                "power(sin(radians(mean_lat - ?) / 2.0), 2) + "
                "cos(radians(?)) * cos(radians(mean_lat)) * "
                "power(sin(radians(mean_lon - ?) / 2.0), 2))))"
            )
            clauses.append(f"({haversine}) <= ?")
            params.extend([location.lat, location.lat, location.lon, location.radius_km])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    def _local_series(self, params: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        location: Location = params["location"]
        resolution = (
            "10km"
            if location.radius_km is not None and location.radius_km <= 50.0
            else "50km"
        )
        if resolution == "10km":
            path = self._require("cube_cell10km_monthly.parquet")
            kind = "cube10"
        else:
            path = self._first_existing(
                "harmonized_cell50km_monthly.parquet",
                "cube_cell50km_monthly.parquet",
            )
            if path is None:
                raise NotFound(
                    "No 50 km cell cube is available for the spatial filter.",
                    code="asset_not_found",
                )
            kind = "cell50" if path.name.startswith("harmonized_") else "cube50"
        columns = set(self._columns(path))
        where, where_params = self._cell_where(
            location=location,
            iso=str(params["iso_a3"]),
            range_start=str(params["range_start"]),
            range_end=str(params["range_end"]),
        )
        has_harmonized = "harmonized_srehi" in columns
        partial_expr = (
            "BOOL_OR(COALESCE(is_partial_month, FALSE))"
            if "is_partial_month" in columns
            else "FALSE"
        )
        if has_harmonized:
            select = f"""
                year, month, year_month,
                SUM(COALESCE(harmonized_srehi, 0)) AS h_srehi,
                SUM(COALESCE(harmonized_srehi_lo90, COALESCE(harmonized_srehi, 0))) AS lo_srehi,
                SUM(COALESCE(harmonized_srehi_hi90, COALESCE(harmonized_srehi, 0))) AS hi_srehi,
                SUM(COALESCE(harmonized_frp_mw, 0)) AS h_frp,
                SUM(COALESCE(harmonized_frp_lo90, COALESCE(harmonized_frp_mw, 0))) AS lo_frp,
                SUM(COALESCE(harmonized_frp_hi90, COALESCE(harmonized_frp_mw, 0))) AS hi_frp,
                SUM(COALESCE(overpass_event_count, 0)) AS raw_srehi,
                SUM(COALESCE(raw_frp_mw, 0)) AS raw_frp,
                SUM(COALESCE(sama_suspect_det_count, 0)) AS sama_suspect_det_count,
                BOOL_OR(COALESCE(is_in_sama, FALSE)) AS is_in_sama,
                MAX(COALESCE(data_maturity, 'STANDARD_SCIENCE')) AS data_maturity,
                MAX(COALESCE(constellation_regime, '')) AS constellation_regime,
                {partial_expr} AS is_partial_month,
                COUNT(*) AS n_cells
            """
        else:
            if "is_nrt_majority" in columns:
                maturity_expr = (
                    "MAX(CASE WHEN is_nrt_majority THEN 'NRT_PRELIMINARY' "
                    "ELSE 'STANDARD_SCIENCE' END)"
                )
            else:
                maturity_expr = "'STANDARD_SCIENCE'"
            select = f"""
                year, month, year_month,
                SUM(COALESCE(overpass_event_count, 0)) AS raw_srehi,
                SUM(COALESCE(frp_sum_mw, 0)) AS raw_frp,
                SUM(COALESCE(clean_det_count, 0)) AS clean_det_count,
                BOOL_OR(COALESCE(is_in_sama, FALSE)) AS is_in_sama,
                {maturity_expr} AS data_maturity,
                '' AS constellation_regime,
                {partial_expr} AS is_partial_month,
                COUNT(*) AS n_cells
            """
        sql = f"""
            SELECT {select}
            FROM {self._parquet(path)}
            {where}
            GROUP BY year, month, year_month
            ORDER BY year, month
        """
        rows = self._fetch(path, sql, where_params)
        if not rows:
            iso = str(params["iso_a3"])
            if iso != SOUTH_AMERICA and not self._known_country(iso):
                raise NotFound(f"Unknown country ISO code: {iso}.", code="unknown_country")
            raise NotFound(
                "No cell observations match the requested spatial filter.",
                code="no_rows",
            )
        series: list[dict[str, Any]] = []
        if has_harmonized:
            for row in rows:
                series.append(
                    {
                        "year": int(_num(row.get("year"))),
                        "month": int(_num(row.get("month"))),
                        "year_month": _text(row.get("year_month")),
                        "country_name": "",
                        "iso_a3": str(params["iso_a3"]),
                        "admin1_name": "",
                        "harmonized_srehi": _num(row.get("h_srehi")),
                        "lo90_srehi": _num(row.get("lo_srehi")),
                        "hi90_srehi": _num(row.get("hi_srehi")),
                        "harmonized_frp_mw": _num(row.get("h_frp")),
                        "lo90_frp_mw": _num(row.get("lo_frp")),
                        "hi90_frp_mw": _num(row.get("hi_frp")),
                        "raw_srehi": _num(row.get("raw_srehi")),
                        "raw_frp_mw": _num(row.get("raw_frp")),
                        "m1_srehi": 0.0,
                        "m2_srehi": 0.0,
                        "m1_frp_mw": 0.0,
                        "m2_frp_mw": 0.0,
                        "nrt_det_count": 0,
                        "sp_det_count": 0,
                        "clean_det_count": 0,
                        "raw_all_det_count": 0,
                        "static_excluded_det_count": 0,
                        "sama_suspect_det_count": int(_num(row.get("sama_suspect_det_count"))),
                        "sama_night_det_count": int(_num(row.get("sama_suspect_det_count"))),
                        "is_in_sama": _as_bool(row.get("is_in_sama")),
                        "intersects_sama": _as_bool(row.get("is_in_sama")),
                        "is_partial_month": _as_bool(row.get("is_partial_month")),
                        "data_maturity": _text(row.get("data_maturity"), "STANDARD_SCIENCE"),
                        "constellation_regime": _text(row.get("constellation_regime")),
                        "z_score_srehi": 0.0,
                        "pct_of_median_srehi": 0.0,
                        "severity_tier": "",
                        "z_score_frp": 0.0,
                        "pct_of_median_frp": 0.0,
                        "severity_tier_frp": "",
                    }
                )
        else:
            # Raw cubes do not carry harmonized predictions.  Derive a local
            # scale factor from the overlapping 50 km harmonized product so
            # the UI still has a defensible M3-equivalent series.
            ratio = {
                "srehi": 1.0,
                "frp": 1.0,
                "lo_srehi": 0.8,
                "hi_srehi": 1.2,
                "lo_frp": 0.8,
                "hi_frp": 1.2,
            }
            anchor = self._first_existing("harmonized_cell50km_monthly.parquet")
            if anchor is not None:
                try:
                    anchor_where, anchor_params = self._cell_where(
                        location=location,
                        iso=str(params["iso_a3"]),
                        range_start=str(params["range_start"]),
                        range_end=str(params["range_end"]),
                    )
                    anchor_rows = self._fetch(
                        anchor,
                        f"""
                        SELECT
                            SUM(COALESCE(harmonized_srehi, 0)) AS h_srehi,
                            SUM(COALESCE(overpass_event_count, 0)) AS r_srehi,
                            SUM(COALESCE(harmonized_srehi_lo90, 0)) AS lo_srehi,
                            SUM(COALESCE(harmonized_srehi_hi90, 0)) AS hi_srehi,
                            SUM(COALESCE(harmonized_frp_mw, 0)) AS h_frp,
                            SUM(COALESCE(raw_frp_mw, 0)) AS r_frp,
                            SUM(COALESCE(harmonized_frp_lo90, 0)) AS lo_frp,
                            SUM(COALESCE(harmonized_frp_hi90, 0)) AS hi_frp
                        FROM {self._parquet(anchor)}
                        {anchor_where}
                        """,
                        anchor_params,
                    )
                    anchor_row = anchor_rows[0] if anchor_rows else {}
                    h_srehi = _num(anchor_row.get("h_srehi"))
                    r_srehi = _num(anchor_row.get("r_srehi"))
                    h_frp = _num(anchor_row.get("h_frp"))
                    r_frp = _num(anchor_row.get("r_frp"))
                    if r_srehi > 0 and h_srehi > 0:
                        ratio["srehi"] = h_srehi / r_srehi
                        ratio["lo_srehi"] = _num(anchor_row.get("lo_srehi")) / h_srehi
                        ratio["hi_srehi"] = _num(anchor_row.get("hi_srehi")) / h_srehi
                    if r_frp > 0 and h_frp > 0:
                        ratio["frp"] = h_frp / r_frp
                        ratio["lo_frp"] = _num(anchor_row.get("lo_frp")) / h_frp
                        ratio["hi_frp"] = _num(anchor_row.get("hi_frp")) / h_frp
                except Exception:  # noqa: BLE001 - optional calibration
                    pass
            for row in rows:
                raw_srehi = _num(row.get("raw_srehi"))
                raw_frp = _num(row.get("raw_frp"))
                h_srehi = raw_srehi * ratio["srehi"]
                h_frp = raw_frp * ratio["frp"]
                series.append(
                    {
                        "year": int(_num(row.get("year"))),
                        "month": int(_num(row.get("month"))),
                        "year_month": _text(row.get("year_month")),
                        "country_name": "",
                        "iso_a3": str(params["iso_a3"]),
                        "admin1_name": "",
                        "harmonized_srehi": h_srehi,
                        "lo90_srehi": h_srehi * ratio["lo_srehi"],
                        "hi90_srehi": h_srehi * ratio["hi_srehi"],
                        "harmonized_frp_mw": h_frp,
                        "lo90_frp_mw": h_frp * ratio["lo_frp"],
                        "hi90_frp_mw": h_frp * ratio["hi_frp"],
                        "raw_srehi": raw_srehi,
                        "raw_frp_mw": raw_frp,
                        "m1_srehi": 0.0,
                        "m2_srehi": 0.0,
                        "m1_frp_mw": 0.0,
                        "m2_frp_mw": 0.0,
                        "nrt_det_count": 0,
                        "sp_det_count": 0,
                        "clean_det_count": int(_num(row.get("clean_det_count"))),
                        "raw_all_det_count": 0,
                        "static_excluded_det_count": 0,
                        "sama_suspect_det_count": 0,
                        "is_in_sama": _as_bool(row.get("is_in_sama")),
                        "intersects_sama": _as_bool(row.get("is_in_sama")),
                        "is_partial_month": _as_bool(row.get("is_partial_month")),
                        "data_maturity": _text(row.get("data_maturity"), "STANDARD_SCIENCE"),
                        "constellation_regime": _text(row.get("constellation_regime")),
                        "z_score_srehi": 0.0,
                        "pct_of_median_srehi": 0.0,
                        "severity_tier": "",
                        "z_score_frp": 0.0,
                        "pct_of_median_frp": 0.0,
                        "severity_tier_frp": "",
                    }
                )
        country_name = "Custom AOI"
        if params["iso_a3"] != SOUTH_AMERICA:
            country = self._catalog_country(str(params["iso_a3"]))
            country_name = _text((country or {}).get("country_name"), "Custom AOI")
        return series, {
            "source": path.name,
            "country_name": country_name,
            "level": "spatial",
            "resolution": resolution,
        }

    # ------------------------------------------------------------------
    # Climatology, matrix, trend and badges
    # ------------------------------------------------------------------

    def _climatology(self, series: Sequence[dict[str, Any]], params: Mapping[str, Any]) -> dict[str, Any]:
        geo_level = None
        if params["iso_a3"] != SOUTH_AMERICA:
            geo_level = "admin1" if params["admin1_name"] else "country"
        clim_path = self._first_existing("calendar_climatology.parquet")
        if clim_path is not None and geo_level is not None and not params["location"].active:
            try:
                rows = self._fetch(
                    clim_path,
                    f"""
                    SELECT * FROM {self._parquet(clim_path)}
                    WHERE geo_level = ? AND iso_a3 = ?
                      AND COALESCE(admin1_name, '') = ?
                    ORDER BY month
                    """,
                    [geo_level, params["iso_a3"], params["admin1_name"]],
                )
            except Exception:  # noqa: BLE001 - fall back to derived values
                rows = []
            if rows:
                entries = []
                for row in rows:
                    entries.append(
                        {
                            "month": int(_num(row.get("month"))),
                            "n_years": int(_num(row.get("n_years"))),
                            "p25": _num(row.get("srehi_p25")),
                            "p50": _num(row.get("srehi_p50")),
                            "p75": _num(row.get("srehi_p75")),
                            "p90": _num(row.get("srehi_p90")),
                            "p95": _num(row.get("srehi_p95")),
                            "iqr": _num(row.get("srehi_iqr")),
                            "mean": _num(row.get("srehi_mean")),
                            "std": _num(row.get("srehi_std")),
                            "srehi_p25": _num(row.get("srehi_p25")),
                            "srehi_p50": _num(row.get("srehi_p50")),
                            "srehi_p75": _num(row.get("srehi_p75")),
                            "srehi_p90": _num(row.get("srehi_p90")),
                            "srehi_p95": _num(row.get("srehi_p95")),
                            "srehi_iqr": _num(row.get("srehi_iqr")),
                            "srehi_mean": _num(row.get("srehi_mean")),
                            "srehi_std": _num(row.get("srehi_std")),
                            "frp_p25": _num(row.get("frp_p25")),
                            "frp_p50": _num(row.get("frp_p50")),
                            "frp_p75": _num(row.get("frp_p75")),
                            "frp_p90": _num(row.get("frp_p90")),
                            "frp_p95": _num(row.get("frp_p95")),
                            "frp_iqr": _num(row.get("frp_iqr")),
                            "frp_mean": _num(row.get("frp_mean")),
                            "frp_std": _num(row.get("frp_std")),
                        }
                    )
                if len(entries) == 12:
                    return {
                        "frozen_period": "2001-2020",
                        "source": clim_path.name,
                        "available": True,
                        "months": entries,
                    }
        # Derived frozen climatology (also used for South America and AOIs).
        by_month_srehi: dict[int, list[float]] = {month: [] for month in range(1, 13)}
        by_month_frp: dict[int, list[float]] = {month: [] for month in range(1, 13)}
        for row in series:
            if 2001 <= int(row["year"]) <= 2020:
                by_month_srehi[int(row["month"])].append(
                    _num(row.get("harmonized_srehi"))
                )
                by_month_frp[int(row["month"])].append(
                    _num(row.get("harmonized_frp_mw"))
                )
        if not any(by_month_srehi.values()):
            for row in series:
                by_month_srehi[int(row["month"])].append(
                    _num(row.get("harmonized_srehi"))
                )
                by_month_frp[int(row["month"])].append(
                    _num(row.get("harmonized_frp_mw"))
                )

        def _stats(values: Sequence[float]) -> dict[str, float]:
            p25 = _percentile(values, 0.25)
            p50 = _percentile(values, 0.50)
            p75 = _percentile(values, 0.75)
            p90 = _percentile(values, 0.90)
            p95 = _percentile(values, 0.95)
            mean = _mean(values)
            return {
                "p25": p25,
                "p50": p50,
                "p75": p75,
                "p90": p90,
                "p95": p95,
                "iqr": p75 - p25,
                "mean": mean,
                "std": _std(values),
            }

        entries = []
        for month in range(1, 13):
            srehi_stats = _stats(by_month_srehi[month])
            frp_stats = _stats(by_month_frp[month])
            entries.append(
                {
                    "month": month,
                    "n_years": len(by_month_srehi[month]),
                    **srehi_stats,
                    "srehi_p25": srehi_stats["p25"],
                    "srehi_p50": srehi_stats["p50"],
                    "srehi_p75": srehi_stats["p75"],
                    "srehi_p90": srehi_stats["p90"],
                    "srehi_p95": srehi_stats["p95"],
                    "srehi_iqr": srehi_stats["iqr"],
                    "srehi_mean": srehi_stats["mean"],
                    "srehi_std": srehi_stats["std"],
                    "frp_p25": frp_stats["p25"],
                    "frp_p50": frp_stats["p50"],
                    "frp_p75": frp_stats["p75"],
                    "frp_p90": frp_stats["p90"],
                    "frp_p95": frp_stats["p95"],
                    "frp_iqr": frp_stats["iqr"],
                    "frp_mean": frp_stats["mean"],
                    "frp_std": frp_stats["std"],
                }
            )
        return {
            "frozen_period": "2001-2020",
            "source": "derived_from_harmonized_series",
            "available": True,
            "months": entries,
        }

    @staticmethod
    def _clim_month(climatology: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
        return {
            int(entry.get("month")): entry
            for entry in climatology.get("months", [])
            if entry.get("month") is not None
        }

    def _enrich_anomalies(
        self,
        series: Sequence[dict[str, Any]],
        climatology: Mapping[str, Any],
        metric: str,
        mode: str,
    ) -> None:
        prefix = "frp" if metric == "frp_mw" else "srehi"
        severity_key = "severity_tier_frp" if prefix == "frp" else "severity_tier"
        clim_by_month = self._clim_month(climatology)
        for row in series:
            clim = clim_by_month.get(int(row["month"]), {})
            value = _num(row.get(f"harmonized_{metric}"))
            if mode == "raw":
                value = _num(row.get(f"raw_{metric}"))
            mean_value = _num(clim.get("mean"))
            std_value = _num(clim.get("std"))
            p50_value = _num(clim.get("p50"))
            z_value = (value - mean_value) / std_value if std_value > 0 else 0.0
            row[f"z_score_{prefix}"] = z_value
            row[f"pct_of_median_{prefix}"] = (
                100.0 * value / p50_value if p50_value > 0 else 0.0
            )
            row[severity_key] = _severity(
                value,
                _num(clim.get("p25")),
                _num(clim.get("p75")),
                _num(clim.get("p90")),
                z_value,
            )

    def _build_matrix(
        self,
        series: Sequence[dict[str, Any]],
        params: Mapping[str, Any],
        climatology: Mapping[str, Any],
    ) -> dict[str, Any]:
        metric = str(params["metric"])
        mode = str(params["series_mode"])
        prefix = "frp" if metric == "frp_mw" else "srehi"
        clim_by_month = self._clim_month(climatology)
        rows: list[dict[str, Any]] = []
        years_present = {int(row["year"]) for row in series}
        if params.get("start_ym") or params.get("end_ym"):
            years = list(range(min(years_present), max(years_present) + 1)) if years_present else []
        else:
            years = list(range(DEFAULT_YEAR_MIN, DEFAULT_YEAR_MAX + 1))
            if years_present:
                years = sorted(set(years) | years_present)
        by_ym = {str(row["year_month"]): row for row in series}
        for year in years:
            for month in range(1, 13):
                ym = _ym(year, month)
                source_row = by_ym.get(ym)
                clim = clim_by_month.get(month, {})
                if source_row is None:
                    value = 0.0
                    raw_value = 0.0
                    has_data = False
                    z_score = 0.0
                    pct_median = 0.0
                    severity = "LOW_NORMAL"
                    maturity = ""
                    regime = ""
                    sama = False
                else:
                    h_value = _num(source_row.get(f"harmonized_{metric}"))
                    raw_value = _num(source_row.get(f"raw_{metric}"))
                    value = h_value if mode == "harmonized" else raw_value
                    has_data = True
                    clim_mean = _num(clim.get(f"{prefix}_mean"), _num(clim.get("mean")))
                    clim_std = _num(clim.get(f"{prefix}_std"), _num(clim.get("std")))
                    clim_p25 = _num(clim.get(f"{prefix}_p25"), _num(clim.get("p25")))
                    clim_p50 = _num(clim.get(f"{prefix}_p50"), _num(clim.get("p50")))
                    clim_p75 = _num(clim.get(f"{prefix}_p75"), _num(clim.get("p75")))
                    clim_p90 = _num(clim.get(f"{prefix}_p90"), _num(clim.get("p90")))
                    computed_z = (
                        (value - clim_mean) / clim_std if clim_std > 0 else 0.0
                    )
                    computed_pct = (
                        100.0 * value / clim_p50 if clim_p50 > 0 else 0.0
                    )
                    z_score = (
                        computed_z
                        if mode == "raw"
                        else _num(
                            source_row.get(f"z_score_{prefix}"),
                            computed_z,
                        )
                    )
                    pct_median = (
                        computed_pct
                        if mode == "raw"
                        else _num(
                            source_row.get(f"pct_of_median_{prefix}"),
                            computed_pct,
                        )
                    )
                    severity = "" if mode == "raw" else _text(
                        source_row.get(f"severity_tier_{prefix}" if prefix == "frp" else "severity_tier")
                    )
                    if not severity:
                        severity = _severity(
                            value,
                            clim_p25,
                            clim_p75,
                            clim_p90,
                            z_score,
                        )
                    maturity = _text(source_row.get("data_maturity"))
                    regime = _text(source_row.get("constellation_regime"))
                    sama = bool(source_row.get("is_in_sama"))
                rows.append(
                    {
                        "year": year,
                        "month": month,
                        "year_month": ym,
                        "value": float(value),
                        "raw_value": float(raw_value),
                        "has_data": has_data,
                        "z_score": float(z_score),
                        "pct_of_median": float(pct_median),
                        "severity_tier": severity,
                        "data_maturity": maturity,
                        "constellation_regime": regime,
                        "is_in_sama": sama,
                    }
                )
        values = [row["value"] for row in rows if row["has_data"]]
        grid = [
            [row["value"] for row in rows if row["year"] == year]
            for year in years
        ]
        return {
            "years": years,
            "months": list(range(1, 13)),
            "cells": rows,
            "values": grid,
            "max_value": max(values) if values else 0.0,
            "min_value": min(values) if values else 0.0,
            "p95_value": _percentile(values, 0.95),
            "metric": metric,
            "series_mode": mode,
        }

    def _trend(self, series: Sequence[dict[str, Any]], params: Mapping[str, Any]) -> dict[str, Any]:
        metric_column = "harmonized_frp_mw" if params["metric"] == "frp_mw" else "harmonized_srehi"
        short_metric = "frp_mw" if params["metric"] == "frp_mw" else "srehi"
        if params["iso_a3"] != SOUTH_AMERICA and not params["location"].active:
            geo_level = "admin1" if params["admin1_name"] else "country"
            path = self._first_existing("calendar_trends.parquet")
            if path is not None:
                try:
                    rows = self._fetch(
                        path,
                        f"""
                        SELECT * FROM {self._parquet(path)}
                        WHERE geo_level = ? AND iso_a3 = ?
                          AND COALESCE(admin1_name, '') = ?
                          AND metric IN (?, ?)
                        LIMIT 1
                        """,
                        [
                            geo_level,
                            params["iso_a3"],
                            params["admin1_name"],
                            metric_column,
                            short_metric,
                        ],
                    )
                except Exception:  # noqa: BLE001
                    rows = []
                if rows:
                    row = rows[0]
                    return {
                        "available": True,
                        "source": path.name,
                        "method": _text(row.get("method"), "hamed_rao_mk"),
                        "n_years": int(_num(row.get("n_years"))),
                        "start_year": int(_num(row.get("start_year"))),
                        "end_year": int(_num(row.get("end_year"))),
                        "slope": _num(row.get("slope")),
                        "slope_ci_low": _num(row.get("slope_ci_low")),
                        "slope_ci_high": _num(row.get("slope_ci_high")),
                        "p_value": _num(row.get("p_value"), 1.0),
                        "q_value": _num(row.get("q_value"), 1.0),
                        "significant_fdr": _as_bool(row.get("significant_fdr")),
                        "trend_direction": _text(row.get("trend_direction"), "no_significant_trend"),
                        "gev_fit_ok": _as_bool(row.get("gev_fit_ok")),
                        "gev_rl10": _num(row.get("gev_rl10")),
                        "gev_rl20": _num(row.get("gev_rl20")),
                        "gev_rl50": _num(row.get("gev_rl50")),
                        "gev_input": _text(row.get("gev_input"), "annual_max_monthly"),
                        "gev_max_observed": _num(row.get("gev_max_observed")),
                        "gev_rl10_ci_low": _num(row.get("gev_rl10_ci_low")),
                        "gev_rl10_ci_high": _num(row.get("gev_rl10_ci_high")),
                        "gev_rl50_ci_low": _num(row.get("gev_rl50_ci_low")),
                        "gev_rl50_ci_high": _num(row.get("gev_rl50_ci_high")),
                        "gev_reject_reason": _text(row.get("gev_reject_reason")),
                    }
        # Deterministic fallback: ordinary least squares on annual totals.
        annual: dict[int, float] = {}
        for row in series:
            year = int(row["year"])
            annual[year] = annual.get(year, 0.0) + _num(row.get(metric_column))
        points = [(year, value) for year, value in sorted(annual.items()) if 2001 <= year <= 2025]
        if len(points) < 3:
            return {
                "available": False,
                "source": None,
                "method": "insufficient_data",
                "n_years": len(points),
                "slope": 0.0,
                "slope_ci_low": 0.0,
                "slope_ci_high": 0.0,
                "p_value": 1.0,
                "q_value": 1.0,
                "significant_fdr": False,
                "trend_direction": "no_significant_trend",
                "gev_fit_ok": False,
                "gev_rl10": 0.0,
                "gev_rl20": 0.0,
                "gev_rl50": 0.0,
            }
        xs = [float(year) for year, _ in points]
        ys = [float(value) for _, value in points]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denominator = sum((x - mean_x) ** 2 for x in xs)
        slope = (
            sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
            if denominator
            else 0.0
        )
        intercept = mean_y - slope * mean_x
        residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        sigma = math.sqrt(sum(r * r for r in residuals) / max(1, len(xs) - 2))
        se = sigma / math.sqrt(denominator) if denominator else 0.0
        direction = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "no_significant_trend")
        return {
            "available": True,
            "source": "derived_ols_annual_totals",
            "method": "ols_on_annual_totals",
            "n_years": len(points),
            "start_year": int(points[0][0]),
            "end_year": int(points[-1][0]),
            "slope": slope,
            "slope_ci_low": slope - 1.96 * se,
            "slope_ci_high": slope + 1.96 * se,
            "p_value": 1.0,
            "q_value": 1.0,
            "significant_fdr": False,
            "trend_direction": direction,
            "gev_fit_ok": False,
            "gev_rl10": 0.0,
            "gev_rl20": 0.0,
            "gev_rl50": 0.0,
        }

    def _badges(
        self,
        series: Sequence[dict[str, Any]],
        params: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        dates = {str(row["year_month"]) for row in series}
        if dates:
            nrt_months = sum(1 for row in series if row.get("data_maturity") == "NRT_PRELIMINARY")
            partial_months = sum(1 for row in series if row.get("is_partial_month"))
            sama_suspect = int(sum(_num(row.get("sama_suspect_det_count")) for row in series))
            sama_hit = any(bool(row.get("is_in_sama")) for row in series)
            regimes = sorted({_text(row.get("constellation_regime")) for row in series} - {""})
            metric = str(params["metric"])
            widths = []
            for row in series:
                high = _num(row.get(f"hi90_{metric}"))
                low = _num(row.get(f"lo90_{metric}"))
                value = _num(row.get(f"harmonized_{metric}"))
                if high > low and value > 0:
                    widths.append((high - low) / value)
            band_ratio = _mean(widths)
            quality_tone = "warn" if (nrt_months or partial_months) else "ok"
            sama_tone = "warn" if (sama_hit or sama_suspect > 0) else "ok"
            relay_tone = "info" if len(regimes) > 1 else "ok"
        else:  # pragma: no cover - defensive
            nrt_months = partial_months = sama_suspect = 0
            sama_hit = False
            regimes = []
            band_ratio = 0.0
            quality_tone = sama_tone = relay_tone = "info"
        badges = [
            {
                "id": "quality_maturity",
                "code": "quality",
                "kind": "quality",
                "tone": quality_tone,
                "icon": "shield",
                "title_en": "Quality & data maturity",
                "title_es": "Calidad y madurez del dato",
                "detail_en": (
                    f"{nrt_months} NRT-preliminary month(s), {partial_months} partial "
                    f"month(s) out of {len(series)} selected."
                ),
                "detail_es": (
                    f"{nrt_months} mes(es) NRT-preliminar, {partial_months} mes(es) "
                    f"parcial(es) de {len(series)} seleccionados."
                ),
                "metrics": {
                    "nrt_preliminary_months": nrt_months,
                    "partial_months": partial_months,
                    "selected_months": len(series),
                },
            },
            {
                "id": "sama_static",
                "code": "sama",
                "kind": "sama",
                "tone": sama_tone,
                "icon": "magnet",
                "title_en": "SAMA & static thermal cells",
                "title_es": "SAMA y celdas térmicas estáticas",
                "detail_en": (
                    f"{sama_suspect} SAMA-suspect night detection(s); "
                    f"{self._static_count} chronic static cells flagged globally."
                ),
                "detail_es": (
                    f"{sama_suspect} detección(es) nocturna(s) sospechosa(s) de SAMA; "
                    f"{self._static_count} celdas estáticas crónicas marcadas globalmente."
                ),
                "metrics": {
                    "sama_suspect_detections": sama_suspect,
                    "sama_night_detections": sama_suspect,
                    "intersects_sama": bool(sama_hit),
                    "static_cells": self._static_count,
                },
            },
            {
                "id": "relay_uncertainty",
                "code": "relay",
                "kind": "relay",
                "tone": relay_tone,
                "icon": "relay",
                "title_en": "Constellation relay & uncertainty",
                "title_es": "Relevo de constelación e incertidumbre",
                "detail_en": (
                    f"Regimes present: {', '.join(regimes) if regimes else 'n/a'}; "
                    f"mean 90% band width = {100.0 * band_ratio:.1f}% of the M3 estimate."
                ),
                "detail_es": (
                    f"Regímenes presentes: {', '.join(regimes) if regimes else 'n/a'}; "
                    f"ancho medio de banda 90% = {100.0 * band_ratio:.1f}% de la estimación M3."
                ),
                "metrics": {
                    "regimes": regimes,
                    "band_width_ratio": band_ratio,
                    "transitions": ["2002", "2012", "2018", "2021", "2023"],
                },
            },
        ]
        for badge in badges:
            badge.setdefault("title", badge.get("title_en"))
            badge.setdefault("detail", badge.get("detail_en"))
        return badges

    # ------------------------------------------------------------------
    # API handlers
    # ------------------------------------------------------------------

    def _calendar_payload(self, request: Request) -> dict[str, Any]:
        params = self._calendar_params(request)
        location: Location = params["location"]
        if location.active:
            series, meta = self._local_series(params)
        else:
            series, meta = self._geo_series(params)
        climatology = self._climatology(series, params)
        climatology_prefix = "frp" if params["metric"] == "frp_mw" else "srehi"
        for entry in climatology.get("months", []):
            for statistic in ("p25", "p50", "p75", "p90", "p95", "iqr", "mean", "std"):
                metric_value = entry.get(f"{climatology_prefix}_{statistic}")
                if metric_value is not None:
                    entry[statistic] = metric_value
        anomaly_prefix = "frp" if params["metric"] == "frp_mw" else "srehi"
        anomaly_severity_key = (
            "severity_tier_frp" if anomaly_prefix == "frp" else "severity_tier"
        )
        has_anomaly_metadata = any(
            _text(row.get(anomaly_severity_key))
            or abs(_num(row.get(f"z_score_{anomaly_prefix}"))) > 1e-12
            for row in series
        )
        if (
            params["location"].active
            or params["series_mode"] == "raw"
            or not has_anomaly_metadata
        ):
            self._enrich_anomalies(
                series,
                climatology,
                str(params["metric"]),
                str(params["series_mode"]),
            )
        matrix = self._build_matrix(series, params, climatology)
        trend = self._trend(series, params)
        badges = self._badges(series, params)
        metric = str(params["metric"])
        prefix = "frp" if metric == "frp_mw" else "srehi"
        timeseries = []
        for row in series:
            harmonized_value = _num(row.get(f"harmonized_{metric}"))
            raw_value = _num(row.get(f"raw_{metric}"))
            timeseries.append(
                {
                    "year": int(row["year"]),
                    "month": int(row["month"]),
                    "year_month": str(row["year_month"]),
                    "value": harmonized_value if params["series_mode"] == "harmonized" else raw_value,
                    "harmonized": harmonized_value,
                    "lo90": _num(row.get(f"lo90_{metric}")),
                    "hi90": _num(row.get(f"hi90_{metric}")),
                    "raw": raw_value,
                    "m1": _num(row.get(f"m1_{metric}")),
                    "m2": _num(row.get(f"m2_{metric}")),
                    "z_score": _num(row.get(f"z_score_{prefix}")),
                    "pct_of_median": _num(row.get(f"pct_of_median_{prefix}")),
                    "severity_tier": _text(
                        row.get("severity_tier" if prefix == "srehi" else "severity_tier_frp")
                    ),
                    "data_maturity": _text(row.get("data_maturity")),
                    "constellation_regime": _text(row.get("constellation_regime")),
                    "is_in_sama": bool(row.get("is_in_sama")),
                    "sama_suspect_det_count": int(_num(row.get("sama_suspect_det_count"))),
                    "sama_night_det_count": int(_num(row.get("sama_suspect_det_count"))),
                    "nrt_det_count": int(_num(row.get("nrt_det_count"))),
                    "sp_det_count": int(_num(row.get("sp_det_count"))),
                }
            )
        latest = timeseries[-1] if timeseries else None
        selected_values = [item["harmonized"] if params["series_mode"] == "harmonized" else item["raw"] for item in timeseries]
        last12 = timeseries[-12:]
        previous12 = timeseries[-24:-12] if len(timeseries) >= 24 else []
        kpis = {
            "selected_months": len(timeseries),
            "selected_total": sum(selected_values),
            "peak_value": max(selected_values) if selected_values else 0.0,
            "peak_year_month": (
                max(timeseries, key=lambda item: item["harmonized"] if params["series_mode"] == "harmonized" else item["raw"])["year_month"]
                if timeseries
                else None
            ),
            "latest_year_month": latest["year_month"] if latest else None,
            "latest_value": (
                latest["harmonized"] if params["series_mode"] == "harmonized" and latest else (latest["raw"] if latest else 0.0)
            ),
            "last12_total": sum(
                item["harmonized"] if params["series_mode"] == "harmonized" else item["raw"]
                for item in last12
            ),
            "previous12_total": sum(
                item["harmonized"] if params["series_mode"] == "harmonized" else item["raw"]
                for item in previous12
            ),
            "yoy_change_pct": (
                100.0
                * (
                    sum(item["harmonized"] if params["series_mode"] == "harmonized" else item["raw"] for item in last12)
                    - sum(item["harmonized"] if params["series_mode"] == "harmonized" else item["raw"] for item in previous12)
                )
                / max(1.0, sum(item["harmonized"] if params["series_mode"] == "harmonized" else item["raw"] for item in previous12))
                if previous12
                else 0.0
            ),
            "static_cells": self._static_count,
        }
        return {
            "generated_at_utc": _utc_now(),
            "filters": {
                "iso_a3": params["iso_a3"],
                "admin1_name": params["admin1_name"],
                "metric": metric,
                "series_mode": params["series_mode"],
                "start_ym": params["start_ym"],
                "end_ym": params["end_ym"],
                **location.describe(),
            },
            "selection": {
                "iso_a3": params["iso_a3"],
                "country_name": meta.get("country_name"),
                "admin1_name": params["admin1_name"],
                "level": meta.get("level"),
                "source": meta.get("source"),
                "resolution": meta.get("resolution"),
            },
            "iso_a3": params["iso_a3"],
            "country_name": meta.get("country_name"),
            "admin1_name": params["admin1_name"],
            "metric": metric,
            "series_mode": params["series_mode"],
            "climatology": climatology,
            "climatology_rows": climatology.get("months", []),
            "matrix": matrix,
            "timeseries": timeseries,
            "trend": trend,
            "badges": badges,
            "quality_badges": badges,
            "kpis": kpis,
            "methodology": {
                "harmonization": "M3 simplex ensemble + Kalman smoother + hierarchical reconciliation",
                "climatology": "frozen 2001-2020 monthly quantiles",
                "transitions": ["2002", "2012", "2018", "2021", "2023"],
                "uncertainty": "90% predictive band from the M3 ensemble",
                "quality": "SP/NRT maturity, SAMA containment and persistent static-cell removal",
            },
        }

    def _health_payload(self) -> dict[str, Any]:
        started = time.perf_counter()
        parquet_status: list[dict[str, Any]] = []
        for name in PARQUET_FILES:
            path = self.processed / name
            item: dict[str, Any] = {
                "name": name,
                "available": path.is_file(),
                "size_bytes": None,
                "modified_utc": None,
                "query_ms": None,
                "error": None,
            }
            if path.is_file():
                try:
                    stat = path.stat()
                    item["size_bytes"] = int(stat.st_size)
                    item["modified_utc"] = dt.datetime.fromtimestamp(
                        stat.st_mtime, tz=dt.timezone.utc
                    ).isoformat()
                    query_started = time.perf_counter()
                    self._fetch(
                        path,
                        f"SELECT * FROM {self._parquet(path)} LIMIT 0",
                    )
                    item["query_ms"] = round(
                        (time.perf_counter() - query_started) * 1000.0, 3
                    )
                except Exception as exc:  # noqa: BLE001 - health reports, never raises
                    item["error"] = str(exc)
            else:
                item["error"] = "missing"
            parquet_status.append(item)
        core_ok = all(
            item["available"] and not item["error"]
            for item in parquet_status
            if item["name"] in CORE_PARQUETS
        )
        qc = self._read_json(self.processed / "qc_cleaning_summary.json")
        benchmarks = self._benchmark_summary()
        status = "ok" if core_ok and qc is not None else "degraded"
        availability = {item["name"]: item for item in parquet_status}
        return {
            "status": status,
            "version": APP_VERSION,
            "generated_at_utc": _utc_now(),
            "started_at_utc": self._started_at,
            "processed_dir": self.processed.name,
            "latency_budget_ms": LATENCY_BUDGET_MS,
            "response_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "duckdb": {
                "version": self._duckdb_version,
                "threads": 2,
                "memory_limit": "1GB",
                "connection_mode": "thread-local read_parquet",
            },
            "pyarrow_version": str(getattr(_pa, "__version__", "unknown")),
            "parquets": parquet_status,
            "parquet_availability": availability,
            "core_available": core_ok,
            "qc_cleaning_summary": qc,
            "qc": qc,
            "benchmarks_m0_m3": benchmarks,
            "benchmarks": benchmarks,
        }

    def _benchmark_summary(self) -> dict[str, Any]:
        benchmark = None
        source: Path | None = None
        candidate = self.processed / "benchmarks_m0_m3.json"
        if candidate.is_file():
            benchmark = self._read_json(candidate)
            source = candidate
        models: dict[str, Any] = {}
        experiments_source: Path | None = None
        if isinstance(benchmark, dict):
            for key in ("models", "metrics", "m0_m3"):
                candidate = benchmark.get(key)
                if isinstance(candidate, dict):
                    models = candidate
                    break
                if isinstance(candidate, list):
                    for item in candidate:
                        if not isinstance(item, dict):
                            continue
                        name = _text(item.get("model") or item.get("id"))
                        if name:
                            models[name] = item
                    if models:
                        break
        if not models:
            experiments = self.processed / "experiments.jsonl"
            if experiments.is_file():
                experiments_source = experiments
                try:
                    with experiments.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                            except ValueError:
                                continue
                            if not isinstance(record, dict):
                                continue
                            name = _text(record.get("model") or record.get("id"))
                            if not name:
                                continue
                            models[name] = {
                                "model": name,
                                "oof_rmse": record.get("oof_score"),
                                "fold_mean_rmse": record.get("mean"),
                                "fold_std_rmse": record.get("std"),
                                "n_folds": record.get("n_folds"),
                                "target": record.get("target"),
                                "metrics": record.get("metrics")
                                or record.get("metrics_by_target"),
                            }
                except OSError:
                    experiments_source = None
        m0_m3 = []
        for name, payload in models.items():
            if not isinstance(payload, dict):
                continue
            fold_mean = payload.get("fold_mean_rmse")
            if fold_mean is None:
                mean_payload = payload.get("mean")
                fold_mean = (
                    _find_metric(mean_payload, "rmse")
                    if isinstance(mean_payload, (Mapping, list))
                    else mean_payload
                )
            m0_m3.append(
                {
                    "model": name,
                    "oof_rmse": _find_metric(payload, "rmse"),
                    "fold_mean_rmse": fold_mean,
                    "n_folds": payload.get("n_folds")
                    or (benchmark or {}).get("folds"),
                }
            )
        m0_m3.sort(key=lambda item: item["model"])
        payload: dict[str, Any] = {
            "available": bool(benchmark or models),
            "source": source.name if source else None,
            "experiments_source": experiments_source.name if experiments_source else None,
            "folds": (benchmark or {}).get("folds"),
            "transitions": (benchmark or {}).get("transitions", {}),
            "dedup": (benchmark or {}).get("dedup", {}),
            "cross_regime_guard": (benchmark or {}).get("cross_regime_guard", {}),
            "models": models,
            "metrics": models,
            "m0_m3": m0_m3,
            "step_change_ratios": (benchmark or {}).get("step_change_ratios", {}),
            "schema": (benchmark or {}).get("schema"),
        }
        return payload

    def _early_onset_payload(self, request: Request) -> dict[str, Any]:
        iso_raw = request.one("iso_a3", "")
        iso = (iso_raw or "").strip().upper()
        if iso in {"", "*", "ALL", "SA", SOUTH_AMERICA}:
            iso = SOUTH_AMERICA
        elif not ISO3_RE.match(iso):
            raise BadRequest("'iso_a3' must be a three-letter ISO code or SOUTH_AMERICA.")
        admin1 = (request.one("admin1_name", "") or "").strip()
        year = request.int_param("year", minimum=1900, maximum=2100)
        path = self._require("calendar_early_onset_90d.parquet")
        where = []
        params: list[Any] = []
        if iso != SOUTH_AMERICA:
            where.append("iso_a3 = ?")
            params.append(iso)
        if admin1:
            where.append("admin1_name = ?")
            params.append(admin1)
        if year is not None:
            where.append("year = ?")
            params.append(year)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        if year is None:
            # Resolve the latest available season before materializing the
            # (much larger) all-years curve.
            latest = self._fetch(
                path,
                f"SELECT MAX(year) AS max_year FROM {self._parquet(path)} {clause}",
                params,
            )
            if latest and latest[0].get("max_year") is not None:
                year = int(_num(latest[0].get("max_year")))
                where.append("year = ?")
                params.append(year)
                clause = f"WHERE {' AND '.join(where)}"
        rows = self._fetch(
            path,
            f"SELECT * FROM {self._parquet(path)} {clause} ORDER BY year, doy",
            params,
        )
        if not rows:
            if iso != SOUTH_AMERICA and not self._known_country(iso):
                raise NotFound(f"Unknown country ISO code: {iso}.", code="unknown_country")
            if admin1 and not self._known_admin1(iso, admin1):
                raise NotFound(
                    f"Unknown Admin-1 unit for {iso}: {admin1}.", code="unknown_admin1"
                )
            raise NotFound("No early-onset curve matches the requested selection.", code="no_rows")
        if year is None:
            year = max(int(_num(row.get("year"))) for row in rows)
        rows = [row for row in rows if int(_num(row.get("year"))) == int(year)]
        grouped: dict[int, dict[str, Any]] = {}
        for row in rows:
            doy = int(_num(row.get("doy")))
            target = grouped.setdefault(
                doy,
                {
                    "doy": doy,
                    "acq_date": row.get("acq_date"),
                    "events": 0.0,
                    "scaled_events": 0.0,
                    "srehi_90d": 0.0,
                    "p75_90d": 0.0,
                    "p90_90d": 0.0,
                    "window_days": 0,
                    "history_status": "OK",
                    "n_years": 0,
                    "early_onset_flag": "NORMAL",
                    "country_name": _text(row.get("country_name")),
                    "iso_a3": _text(row.get("iso_a3")),
                    "admin1_name": _text(row.get("admin1_name")),
                },
            )
            target["events"] += _num(row.get("events"))
            target["scaled_events"] += _num(row.get("scaled_events"))
            target["srehi_90d"] += _num(row.get("srehi_90d"))
            target["p75_90d"] += _num(row.get("p75_90d"))
            target["p90_90d"] += _num(row.get("p90_90d"))
            row_window = int(_num(row.get("window_days"), 90.0))
            target["window_days"] = row_window if target["window_days"] == 0 else min(target["window_days"], row_window)
            target["n_years"] = max(target["n_years"], int(_num(row.get("n_years"), 20.0)))
            if _text(row.get("history_status")) == "INSUFFICIENT_HISTORY":
                target["history_status"] = "INSUFFICIENT_HISTORY"
            if row.get("acq_date") is not None:
                target["acq_date"] = row.get("acq_date")
        curve = []
        for doy in sorted(grouped):
            item = grouped[doy]
            if item["history_status"] == "INSUFFICIENT_HISTORY":
                item["early_onset_flag"] = "INSUFFICIENT_HISTORY"
            elif item["p90_90d"] <= 0 or item["n_years"] < 10:
                item["early_onset_flag"] = "NO_THRESHOLD"
            elif item["srehi_90d"] > item["p90_90d"]:
                item["early_onset_flag"] = "EARLY_ONSET_CRITICAL"
            elif item["srehi_90d"] > item["p75_90d"]:
                item["early_onset_flag"] = "EARLY_ONSET_HIGH"
            else:
                item["early_onset_flag"] = "NORMAL"
            curve.append(item)
        first_p75 = next(
            (item["doy"] for item in curve if item["p75_90d"] > 0 and item["srehi_90d"] > item["p75_90d"]),
            None,
        )
        first_p90 = next(
            (item["doy"] for item in curve if item["p90_90d"] > 0 and item["srehi_90d"] > item["p90_90d"]),
            None,
        )
        critical_days = [item["doy"] for item in curve if item["early_onset_flag"] == "EARLY_ONSET_CRITICAL"]
        high_days = [
            item["doy"]
            for item in curve
            if item["early_onset_flag"] in ("EARLY_ONSET_CRITICAL", "EARLY_ONSET_HIGH")
        ]
        insufficient_days = [item["doy"] for item in curve if item["early_onset_flag"] == "INSUFFICIENT_HISTORY"]
        no_threshold_days = [item["doy"] for item in curve if item["early_onset_flag"] == "NO_THRESHOLD"]
        peak = max(curve, key=lambda item: item["srehi_90d"]) if curve else None
        first_p75_date = _doy_to_date(int(year), first_p75)
        first_p90_date = _doy_to_date(int(year), first_p90)
        summary = {
            "year": int(year),
            "first_crossing_p75_doy": first_p75,
            "first_crossing_p90_doy": first_p90,
            "first_crossing_p75_date": first_p75_date,
            "first_crossing_p90_date": first_p90_date,
            "first_p75_doy": first_p75,
            "first_p90_doy": first_p90,
            "first_p75_date": first_p75_date,
            "first_p90_date": first_p90_date,
            "critical_count": len(critical_days),
            "critical_days": critical_days,
            "above_threshold_count": len(high_days),
            "insufficient_history_count": len(insufficient_days),
            "no_threshold_count": len(no_threshold_days),
            "peak_doy": peak["doy"] if peak else None,
            "peak_srehi_90d": peak["srehi_90d"] if peak else 0.0,
            "peak_ratio_to_p90": (
                peak["srehi_90d"] / peak["p90_90d"]
                if peak and peak["p90_90d"] > 0
                else 0.0
            ),
        }
        return {
            "generated_at_utc": _utc_now(),
            "filters": {"iso_a3": iso, "admin1_name": admin1, "year": int(year)},
            "source": path.name,
            "curve": curve,
            "days": curve,
            "summary": summary,
            "thresholds": {
                "p75": "frozen 2001-2020 trailing 90-day p75",
                "p90": "frozen 2001-2020 trailing 90-day p90",
                "window_days": 90,
                "history_note": "first 89 days of a record are INSUFFICIENT_HISTORY; thresholds require >=10 complete years",
            },
        }

    # ------------------------------------------------------------------
    # Spatial traceability and colour-scale helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_regime_for_ym(ym: str) -> str:
        """Return the constellation regime encoded by a ``YYYY-MM`` month."""

        if ym < "2002-07":
            return "C0_TERRA_ONLY"
        if ym < "2012-01":
            return "C1_TERRA_AQUA"
        if ym < "2018-01":
            return "C2_MODIS_SNPP"
        if ym < "2024-01":
            return "C3_MODIS_SNPP_NOAA20"
        return "C4_MODIS_VIIRS_TRIO"

    @staticmethod
    def _infer_sensors_for_regime(regime: str) -> list[str]:
        mapping = {
            "C0_TERRA_ONLY": ["MODIS_TERRA"],
            "C1_TERRA_AQUA": ["MODIS_AQUA", "MODIS_TERRA"],
            "C2_MODIS_SNPP": ["MODIS_AQUA", "MODIS_TERRA", "VIIRS_SNPP"],
            "C3_MODIS_SNPP_NOAA20": [
                "MODIS_AQUA",
                "MODIS_TERRA",
                "VIIRS_NOAA20",
                "VIIRS_SNPP",
            ],
            "C4_MODIS_VIIRS_TRIO": [
                "MODIS_AQUA",
                "MODIS_TERRA",
                "VIIRS_NOAA20",
                "VIIRS_NOAA21",
                "VIIRS_SNPP",
            ],
        }
        return list(mapping.get(regime, []))

    @staticmethod
    def _scale_from_summary(max_v: float, p99_v: float, p95_v: float) -> dict[str, float]:
        maximum = max(0.0, float(max_v))
        p99 = max(0.0, float(p99_v))
        p95 = max(0.0, float(p95_v))
        if maximum <= 0:
            return {"max": 0.0, "p99": 0.0, "p95": 0.0, "ceiling": 1.0}
        ceiling = p99 if (p99 > 0 and maximum > 1.5 * p99) else maximum
        if ceiling <= 0:
            ceiling = 1.0
        return {
            "max": round(maximum, 4),
            "p99": round(p99, 4),
            "p95": round(p95, 4),
            "ceiling": round(ceiling, 4),
        }

    @classmethod
    def _compute_scale_stats(cls, values: Sequence[float]) -> dict[str, float]:
        finite = [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ]
        if not finite:
            return {"max": 0.0, "p99": 0.0, "p95": 0.0, "ceiling": 1.0}
        maximum = max(finite)
        if maximum <= 0:
            return {"max": 0.0, "p99": 0.0, "p95": 0.0, "ceiling": 1.0}
        return cls._scale_from_summary(
            maximum,
            _percentile(finite, 0.99),
            _percentile(finite, 0.95),
        )

    def _available_months(self) -> list[str]:
        time_range = (self.catalog or {}).get("time_range") or {}
        start = _text(time_range.get("min_year_month")) or CATALOG_MIN_YM
        end = _text(time_range.get("max_year_month")) or CATALOG_MAX_YM
        if not YM_RE.match(start) or not YM_RE.match(end) or start > end:
            return []
        months: list[str] = []
        year, month = int(start[:4]), int(start[5:7])
        end_year, end_month = int(end[:4]), int(end[5:7])
        for _ in range(2400):
            months.append(_ym(year, month))
            if year == end_year and month == end_month:
                break
            month += 1
            if month > 12:
                month = 1
                year += 1
        return months

    def _spatial_meta(
        self,
        resolution: str,
        target_ym: str,
        iso_filter: str,
    ) -> dict[str, Any]:
        """Return sensor participation and maturity for one month (cached)."""

        key = ("spatial_meta", resolution, target_ym, iso_filter)
        with self._cache_lock:
            cached = self._cache.get(key)
        if isinstance(cached, dict):
            return {
                "participating_sensors": list(cached.get("participating_sensors") or []),
                "is_partial_month": bool(cached.get("is_partial_month")),
                "data_maturity": _text(cached.get("data_maturity")),
            }
        result: dict[str, Any] = {
            "participating_sensors": [],
            "is_partial_month": False,
            "data_maturity": "",
        }
        path = self.processed / (
            "cube_cell10km_monthly.parquet"
            if resolution == "10km"
            else "cube_cell50km_monthly.parquet"
        )
        if path.is_file():
            columns = set(self._columns(path))
            clauses = ["year_month = ?"]
            params: list[Any] = [target_ym]
            if iso_filter and iso_filter != SOUTH_AMERICA:
                clauses.append("iso_a3 = ?")
                params.append(iso_filter)
            where = " AND ".join(clauses)
            if "sensor_slot" in columns:
                try:
                    rows = self._fetch(
                        path,
                        f"SELECT DISTINCT sensor_slot FROM {self._parquet(path)} "
                        f"WHERE {where} ORDER BY sensor_slot",
                        params,
                    )
                except Exception:  # noqa: BLE001 - metadata is best effort
                    rows = []
                result["participating_sensors"] = [
                    _text(row.get("sensor_slot"))
                    for row in rows
                    if _text(row.get("sensor_slot"))
                ]
            aggregate_fields: list[str] = []
            if "is_partial_month" in columns:
                aggregate_fields.append(
                    "BOOL_OR(COALESCE(is_partial_month, FALSE)) AS is_partial_month"
                )
            if "data_maturity" in columns:
                aggregate_fields.append(
                    "MAX(COALESCE(data_maturity, '')) AS data_maturity"
                )
            if aggregate_fields:
                try:
                    rows = self._fetch(
                        path,
                        f"SELECT {', '.join(aggregate_fields)} "
                        f"FROM {self._parquet(path)} WHERE {where}",
                        params,
                    )
                except Exception:  # noqa: BLE001 - metadata is best effort
                    rows = []
                if rows:
                    result["is_partial_month"] = _as_bool(
                        rows[0].get("is_partial_month")
                    )
                    result["data_maturity"] = _text(rows[0].get("data_maturity"))
        regime = self._infer_regime_for_ym(target_ym)
        if not result["participating_sensors"]:
            result["participating_sensors"] = self._infer_sensors_for_regime(regime)
        if not result["data_maturity"]:
            if target_ym in ("2000-11", "2026-11"):
                result["data_maturity"] = "PARTIAL_MONTH"
            elif target_ym >= "2026-10":
                result["data_maturity"] = "NRT_PRELIMINARY"
            else:
                result["data_maturity"] = "STANDARD_SCIENCE"
        if target_ym in ("2000-11", "2026-11"):
            result["is_partial_month"] = True
        result["data_maturity"] = _text(result["data_maturity"]).upper()
        if result["data_maturity"] == "PARTIAL_MONTH":
            result["is_partial_month"] = True
        with self._cache_lock:
            self._cache[key] = {
                "participating_sensors": list(result["participating_sensors"]),
                "is_partial_month": bool(result["is_partial_month"]),
                "data_maturity": result["data_maturity"],
            }
            while len(self._cache) > 512:
                self._cache.pop(next(iter(self._cache)))
        return result

    def _reference_scale(
        self,
        resolution: str,
        iso_filter: str,
        effective_start: str,
        effective_end: str,
        fallback: Mapping[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """Reference p99 scale across ``[effective_start, effective_end]`` (cached)."""

        if resolution != "50km":
            # The 10 km layer is a raw overpass layer; its fixed reference is
            # the current month scale so the animation never locks onto an
            # arbitrary first frame.
            return {name: dict(stats) for name, stats in fallback.items()}
        key = (
            "spatial_ref_scale",
            resolution,
            iso_filter,
            effective_start,
            effective_end,
        )
        with self._cache_lock:
            cached = self._cache.get(key)
        if isinstance(cached, dict):
            return {name: dict(stats) for name, stats in cached.items()}
        result = {name: dict(stats) for name, stats in fallback.items()}
        if resolution == "50km":
            path = self._first_existing(
                "harmonized_cell50km_monthly.parquet",
                "cube_cell50km_monthly.parquet",
            )
            if path is not None:
                columns = set(self._columns(path))
                srehi_expr = (
                    "harmonized_srehi"
                    if "harmonized_srehi" in columns
                    else "overpass_event_count"
                )
                frp_expr = (
                    "harmonized_frp_mw"
                    if "harmonized_frp_mw" in columns
                    else "frp_sum_mw"
                )
                clauses = ["year_month BETWEEN ? AND ?"]
                params: list[Any] = [effective_start, effective_end]
                if iso_filter and iso_filter != SOUTH_AMERICA:
                    clauses.append("iso_a3 = ?")
                    params.append(iso_filter)
                where = " AND ".join(clauses)
                try:
                    rows = self._fetch(
                        path,
                        f"""
                        SELECT
                            MAX(COALESCE({srehi_expr}, 0)) AS max_srehi,
                            QUANTILE_CONT(COALESCE({srehi_expr}, 0), 0.99) AS p99_srehi,
                            QUANTILE_CONT(COALESCE({srehi_expr}, 0), 0.95) AS p95_srehi,
                            MAX(COALESCE({frp_expr}, 0)) AS max_frp,
                            QUANTILE_CONT(COALESCE({frp_expr}, 0), 0.99) AS p99_frp,
                            QUANTILE_CONT(COALESCE({frp_expr}, 0), 0.95) AS p95_frp
                        FROM {self._parquet(path)}
                        WHERE {where}
                        """,
                        params,
                    )
                except Exception:  # noqa: BLE001 - scale is best effort
                    rows = []
                if rows:
                    row = rows[0]
                    result = {
                        "srehi": self._scale_from_summary(
                            _num(row.get("max_srehi")),
                            _num(row.get("p99_srehi")),
                            _num(row.get("p95_srehi")),
                        ),
                        "frp_mw": self._scale_from_summary(
                            _num(row.get("max_frp")),
                            _num(row.get("p99_frp")),
                            _num(row.get("p95_frp")),
                        ),
                    }
        with self._cache_lock:
            self._cache[key] = {name: dict(stats) for name, stats in result.items()}
            while len(self._cache) > 512:
                self._cache.pop(next(iter(self._cache)))
        return result

    def _spatial_payload(self, request: Request) -> dict[str, Any]:
        year_month = request.one("year_month")
        year = request.int_param("year", minimum=1900, maximum=2100)
        if year_month and year is not None:
            raise BadRequest("Use either 'year_month' or 'year', not both.")
        if not year_month and year is None:
            raise BadRequest("'year_month' or 'year' is required.")
        if year_month:
            year_month = _safe_ym(year_month, name="year_month")
        iso_raw = request.one("iso_a3", "")
        iso = (iso_raw or "").strip().upper()
        if iso in {"", "*", "ALL", "SA", SOUTH_AMERICA}:
            iso = SOUTH_AMERICA
        elif not ISO3_RE.match(iso):
            raise BadRequest("'iso_a3' must be a three-letter ISO code or SOUTH_AMERICA.")
        resolution = (request.one("resolution", "50km") or "50km").strip().lower()
        if resolution not in {"50km", "10km"}:
            raise BadRequest("'resolution' must be '50km' or '10km'.")
        include_static = request.bool_param("include_static", default=True)
        limit = request.int_param(
            "limit", default=DEFAULT_LIMIT, minimum=1, maximum=MAX_LIMIT
        )
        if limit is None:
            limit = DEFAULT_LIMIT
        offset = request.int_param("offset", default=0, minimum=0, maximum=100000)
        if offset is None:
            offset = 0
        range_start_raw = request.one("range_start")
        range_end_raw = request.one("range_end")
        range_start = (
            _safe_ym(range_start_raw, name="range_start") if range_start_raw else None
        )
        range_end = _safe_ym(range_end_raw, name="range_end") if range_end_raw else None
        if range_start and range_end and range_start > range_end:
            raise BadRequest("'range_start' must be earlier than or equal to 'range_end'.")
        location = self._parse_location(request)
        if resolution == "10km":
            path = self._require("cube_cell10km_monthly.parquet")
        else:
            path = self._first_existing(
                "harmonized_cell50km_monthly.parquet",
                "cube_cell50km_monthly.parquet",
            )
            if path is None:
                raise NotFound("No 50 km cell cube is available.", code="asset_not_found")
        columns = set(self._columns(path))
        cell_id = "cell_id_10km" if resolution == "10km" else "cell_id_50km"
        has_harmonized = "harmonized_srehi" in columns
        where, where_params = self._cell_where(
            location=location,
            iso=iso,
            year_month=year_month,
            year=year,
        )
        srehi_expr = "harmonized_srehi" if has_harmonized else "overpass_event_count"
        frp_expr = "harmonized_frp_mw" if has_harmonized else "frp_sum_mw"
        raw_srehi_expr = "overpass_event_count" if "overpass_event_count" in columns else srehi_expr
        raw_frp_expr = "raw_frp_mw" if "raw_frp_mw" in columns else (
            "frp_sum_mw" if "frp_sum_mw" in columns else frp_expr
        )
        if "data_maturity" in columns:
            maturity_expr = "COALESCE(data_maturity, 'STANDARD_SCIENCE')"
        elif "is_nrt_majority" in columns:
            maturity_expr = (
                "CASE WHEN is_nrt_majority THEN 'NRT_PRELIMINARY' "
                "ELSE 'STANDARD_SCIENCE' END"
            )
        else:
            maturity_expr = "'STANDARD_SCIENCE'"
        regime_expr = (
            "COALESCE(constellation_regime, '')"
            if "constellation_regime" in columns
            else "''"
        )
        aggregate_by_cell = year is not None or "sensor_slot" in columns
        has_admin1 = "admin1_name" in columns
        has_srehi_lo90 = "harmonized_srehi_lo90" in columns
        has_srehi_hi90 = "harmonized_srehi_hi90" in columns
        has_frp_lo90 = "harmonized_frp_lo90" in columns
        has_frp_hi90 = "harmonized_frp_hi90" in columns
        has_static_excluded = "static_excluded_det_count" in columns
        has_active_cells = "active_cells_10km" in columns
        has_night = "night_det_count" in columns
        value_type = (
            "M3_HARMONIZED"
            if resolution == "50km" and has_harmonized
            else "RAW_OVERPASS_COUNT"
        )
        if aggregate_by_cell:
            select_fields = [
                f"{cell_id} AS cell_id",
                "MAX(mean_lat) AS mean_lat",
                "MAX(mean_lon) AS mean_lon",
                "MAX(country_name) AS country_name",
                "MAX(iso_a3) AS iso_a3",
                f"SUM(COALESCE({srehi_expr}, 0)) AS srehi",
                f"SUM(COALESCE({frp_expr}, 0)) AS frp_mw",
                f"SUM(COALESCE({raw_srehi_expr}, 0)) AS raw_srehi",
                f"SUM(COALESCE({raw_frp_expr}, 0)) AS raw_frp_mw",
                "BOOL_OR(COALESCE(is_in_sama, FALSE)) AS is_in_sama",
                f"MAX({maturity_expr}) AS data_maturity",
                f"MAX({regime_expr}) AS constellation_regime",
            ]
            if has_admin1:
                select_fields.append("MAX(admin1_name) AS admin1_name")
            if has_srehi_lo90:
                select_fields.append(
                    "SUM(COALESCE(harmonized_srehi_lo90, 0)) AS srehi_lo90"
                )
            if has_srehi_hi90:
                select_fields.append(
                    "SUM(COALESCE(harmonized_srehi_hi90, 0)) AS srehi_hi90"
                )
            if has_frp_lo90:
                select_fields.append(
                    "SUM(COALESCE(harmonized_frp_lo90, 0)) AS frp_lo90"
                )
            if has_frp_hi90:
                select_fields.append(
                    "SUM(COALESCE(harmonized_frp_hi90, 0)) AS frp_hi90"
                )
            if has_static_excluded:
                select_fields.append(
                    "SUM(COALESCE(static_excluded_det_count, 0)) AS static_excluded_det_count"
                )
            if has_active_cells:
                select_fields.append(
                    "MAX(COALESCE(active_cells_10km, 1)) AS active_cells_10km"
                )
            if has_night:
                select_fields.append(
                    "SUM(COALESCE(night_det_count, 0)) AS night_det_count"
                )
            select = ",\n                ".join(select_fields)
            group_by = f"GROUP BY {cell_id}"
            order_by = "ORDER BY srehi DESC NULLS LAST, cell_id"
        else:
            select_fields = [
                f"{cell_id} AS cell_id",
                "mean_lat",
                "mean_lon",
                "country_name",
                "iso_a3",
                f"COALESCE({srehi_expr}, 0) AS srehi",
                f"COALESCE({frp_expr}, 0) AS frp_mw",
                f"COALESCE({raw_srehi_expr}, 0) AS raw_srehi",
                f"COALESCE({raw_frp_expr}, 0) AS raw_frp_mw",
                "COALESCE(is_in_sama, FALSE) AS is_in_sama",
                f"{maturity_expr} AS data_maturity",
                f"{regime_expr} AS constellation_regime",
            ]
            if has_admin1:
                select_fields.append("admin1_name")
            if has_srehi_lo90:
                select_fields.append(
                    "COALESCE(harmonized_srehi_lo90, 0) AS srehi_lo90"
                )
            if has_srehi_hi90:
                select_fields.append(
                    "COALESCE(harmonized_srehi_hi90, 0) AS srehi_hi90"
                )
            if has_frp_lo90:
                select_fields.append("COALESCE(harmonized_frp_lo90, 0) AS frp_lo90")
            if has_frp_hi90:
                select_fields.append("COALESCE(harmonized_frp_hi90, 0) AS frp_hi90")
            if has_static_excluded:
                select_fields.append(
                    "COALESCE(static_excluded_det_count, 0) AS static_excluded_det_count"
                )
            if has_active_cells:
                select_fields.append(
                    "COALESCE(active_cells_10km, 1) AS active_cells_10km"
                )
            if has_night:
                select_fields.append(
                    "COALESCE(night_det_count, 0) AS night_det_count"
                )
            select = ",\n                ".join(select_fields)
            group_by = ""
            order_by = "ORDER BY srehi DESC NULLS LAST, cell_id"
        row_cap = None if year_month else MAX_LIMIT
        limit_clause = "LIMIT ?" if row_cap is not None else ""
        sql = f"""
            SELECT {select}
            FROM {self._parquet(path)}
            {where}
            {group_by}
            {order_by}
            {limit_clause}
        """
        query_params: list[Any] = [*where_params]
        if row_cap is not None:
            query_params.append(int(row_cap))
        rows = self._fetch(path, sql, query_params)
        if not rows and iso != SOUTH_AMERICA:
            if not self._known_country(iso):
                raise NotFound(f"Unknown country ISO code: {iso}.", code="unknown_country")
        all_cells = []
        for row in rows:
            lat = _num(row.get("mean_lat"))
            lon = _num(row.get("mean_lon"))
            iso_a3 = _text(row.get("iso_a3"))
            if has_admin1:
                admin1_name = _text(row.get("admin1_name"))
            else:
                admin1_name = self._resolve_admin1_name(iso_a3, lat, lon)
            static_excluded = (
                int(round(_num(row.get("static_excluded_det_count"))))
                if has_static_excluded
                else 0
            )
            active_cells = (
                int(round(_num(row.get("active_cells_10km"), 1.0)))
                if has_active_cells
                else 1
            )
            night_det_count = (
                int(round(_num(row.get("night_det_count")))) if has_night else 0
            )
            all_cells.append(
                {
                    "cell_id": _text(row.get("cell_id")),
                    "year_month": year_month,
                    "lat": lat,
                    "lon": lon,
                    "country_name": _text(row.get("country_name")),
                    "iso_a3": iso_a3,
                    "srehi": _num(row.get("srehi")),
                    "frp_mw": _num(row.get("frp_mw")),
                    "raw_srehi": _num(row.get("raw_srehi")),
                    "raw_frp_mw": _num(row.get("raw_frp_mw")),
                    "night_det_count": night_det_count,
                    "night_overpasses": night_det_count,
                    "is_in_sama": _as_bool(row.get("is_in_sama")),
                    "data_maturity": _text(row.get("data_maturity")),
                    "constellation_regime": _text(row.get("constellation_regime")),
                    "admin1_name": admin1_name,
                    "srehi_lo90": _num(row.get("srehi_lo90")) if has_srehi_lo90 else 0.0,
                    "srehi_hi90": _num(row.get("srehi_hi90")) if has_srehi_hi90 else 0.0,
                    "frp_lo90": _num(row.get("frp_lo90")) if has_frp_lo90 else 0.0,
                    "frp_hi90": _num(row.get("frp_hi90")) if has_frp_hi90 else 0.0,
                    "static_excluded_det_count": static_excluded,
                    "active_cells_10km": active_cells,
                    "cell_area_km2": 100.0 if resolution == "10km" else 2500.0,
                    "footprint_radius_km": 5.0 if resolution == "10km" else 25.0,
                    "value_type": value_type,
                }
            )
        total_available_cells = len(all_cells)
        cells = all_cells[offset : offset + limit]
        static_cells: list[dict[str, Any]] = []
        if include_static:
            static_cells = self._static_cells(location=location, iso=iso, limit=limit)
        values = [cell["srehi"] for cell in all_cells]
        frp_values = [cell["frp_mw"] for cell in all_cells]
        month_scale = {
            "srehi": self._compute_scale_stats(values),
            "frp_mw": self._compute_scale_stats(frp_values),
        }
        available_months = self._available_months()
        target_ym = year_month
        if target_ym is None and year is not None:
            target_ym = f"{int(year):04d}-12"
        regime = (
            self._infer_regime_for_ym(year_month)
            if year_month
            else self._infer_regime_for_ym(target_ym or CATALOG_MAX_YM)
        )
        if year_month:
            meta = self._spatial_meta(resolution, year_month, iso)
        else:
            meta = {
                "participating_sensors": self._infer_sensors_for_regime(regime),
                "is_partial_month": False,
                "data_maturity": "STANDARD_SCIENCE",
            }
        data_maturity = _text(meta.get("data_maturity")).upper() or "STANDARD_SCIENCE"
        for cell in all_cells:
            candidate = _text(cell.get("data_maturity")).upper()
            if candidate and MATURITY_RANK.get(candidate, 0) > MATURITY_RANK.get(
                data_maturity, 0
            ):
                data_maturity = candidate
        if data_maturity not in MATURITY_RANK:
            data_maturity = "STANDARD_SCIENCE"
        is_partial_month = bool(meta.get("is_partial_month")) or (
            data_maturity == "PARTIAL_MONTH"
        )
        effective_start = range_start or (
            available_months[0] if available_months else target_ym or CATALOG_MIN_YM
        )
        effective_end = range_end or (
            available_months[-1] if available_months else target_ym or CATALOG_MAX_YM
        )
        if effective_start and effective_end and effective_start > effective_end:
            effective_start, effective_end = effective_end, effective_start
        reference_scale = self._reference_scale(
            resolution,
            iso,
            effective_start,
            effective_end,
            month_scale,
        )
        out_of_catalog = bool(
            year_month
            and (year_month < CATALOG_MIN_YM or year_month > CATALOG_MAX_YM)
        )
        real_assets_ready = bool(path.is_file() and columns)
        if not real_assets_ready:
            month_state = "ASSET_MISSING"
        elif out_of_catalog and total_available_cells == 0:
            month_state = "OUT_OF_CATALOG"
        elif total_available_cells == 0:
            month_state = "ZERO_DETECTIONS"
        elif is_partial_month or data_maturity == "PARTIAL_MONTH":
            month_state = "PARTIAL_MONTH"
        elif data_maturity == "NRT_PRELIMINARY":
            month_state = "NRT_PRELIMINARY"
        else:
            month_state = "DETECTIONS_PRESENT"
        generated_at = _utc_now()
        return {
            "generated_at_utc": generated_at,
            "processing_date_utc": generated_at,
            "filters": {
                "year_month": year_month,
                "year": year,
                "iso_a3": iso,
                "resolution": resolution,
                "include_static": include_static,
                "limit": limit,
                "offset": offset,
                "range_start": range_start,
                "range_end": range_end,
                **location.describe(),
            },
            "source": path.name,
            "cells": cells,
            "static_cells": static_cells,
            "static": static_cells,
            "available_months": available_months,
            "reference_scale": reference_scale,
            "month_state": month_state,
            "effective_resolution": resolution,
            "value_type": value_type,
            "constellation_regime": regime,
            "participating_sensors": list(
                meta.get("participating_sensors")
                or self._infer_sensors_for_regime(regime)
            ),
            "data_maturity": data_maturity,
            "is_partial_month": is_partial_month,
            "summary": {
                "n_cells": len(cells),
                "n_static_cells": len(static_cells),
                "active_cells": len(cells),
                "total_available_cells": int(total_available_cells),
                "returned_cells": len(cells),
                "applied_limit": int(limit),
                "offset": int(offset),
                "truncated": bool(offset + len(cells) < total_available_cells),
                "max_srehi": max((cell["srehi"] for cell in cells), default=0.0),
                "total_srehi": sum(cell["srehi"] for cell in cells),
                "total_frp_mw": sum(cell["frp_mw"] for cell in cells),
                "month_total_srehi": sum(values),
                "month_total_frp_mw": sum(frp_values),
                "returned_total_srehi": sum(cell["srehi"] for cell in cells),
                "resolution": resolution,
                "effective_resolution": resolution,
                "value_type": value_type,
                "month_state": month_state,
                "constellation_regime": regime,
                "participating_sensors": list(
                    meta.get("participating_sensors")
                    or self._infer_sensors_for_regime(regime)
                ),
                "data_maturity": data_maturity,
                "is_partial_month": is_partial_month,
                "processing_date_utc": generated_at,
                "out_of_catalog": out_of_catalog,
                "catalog_min_ym": CATALOG_MIN_YM,
                "catalog_max_ym": CATALOG_MAX_YM,
                "range_start": effective_start,
                "range_end": effective_end,
                "month_scale": month_scale,
            },
        }

    def _static_cells(
        self,
        *,
        location: Location,
        iso: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        path = self._first_existing("static_thermal_cells.parquet")
        if path is None:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        bbox = location.effective_bbox
        if bbox:
            min_lat, max_lat, min_lon, max_lon = bbox
            clauses.append("mean_lat BETWEEN ? AND ?")
            params.extend([min_lat, max_lat])
            clauses.append("mean_lon BETWEEN ? AND ?")
            params.extend([min_lon, max_lon])
        if iso != SOUTH_AMERICA:
            country = self._catalog_country(iso)
            country_bbox = (country or {}).get("bbox")
            if isinstance(country_bbox, list) and len(country_bbox) == 4:
                clauses.append("mean_lat BETWEEN ? AND ?")
                params.extend([_num(country_bbox[0]), _num(country_bbox[1])])
                clauses.append("mean_lon BETWEEN ? AND ?")
                params.extend([_num(country_bbox[2]), _num(country_bbox[3])])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._fetch(
            path,
            f"SELECT * FROM {self._parquet(path)} {where} ORDER BY det_count DESC LIMIT ?",
            [*params, int(min(limit, 5000))],
        )
        cells = []
        for row in rows:
            cells.append(
                {
                    "cell_id": _text(row.get("cell_id_10km")),
                    "lat": _num(row.get("mean_lat")),
                    "lon": _num(row.get("mean_lon")),
                    "det_count": int(_num(row.get("det_count"))),
                    "distinct_ym": int(_num(row.get("distinct_ym"))),
                    "active_years": int(_num(row.get("active_years"))),
                    "avg_months_per_year": _num(row.get("avg_months_per_year")),
                    "type_12_ratio": _num(row.get("type_12_ratio")),
                    "first_date": row.get("first_date"),
                    "last_date": row.get("last_date"),
                    "is_persistent_temporal": _as_bool(row.get("is_persistent_temporal")),
                    "is_type_12_dominant": _as_bool(row.get("is_type_12_dominant")),
                }
            )
        if location.lat is not None and location.lon is not None and location.radius_km is not None:
            cells = [
                cell
                for cell in cells
                if _haversine_km(
                    float(location.lat),
                    float(location.lon),
                    float(cell["lat"]),
                    float(cell["lon"]),
                )
                <= float(location.radius_km)
            ]
        return cells

    def _export_csv(self, request: Request) -> tuple[bytes, dict[str, str]]:
        payload = self._calendar_payload(request)
        timeseries = payload.get("timeseries", [])
        filters = payload.get("filters", {})
        selection = payload.get("selection", {})
        generated = _utc_now()
        bbox = filters.get("bbox")
        bbox_text = ",".join(str(item) for item in bbox) if bbox else ""
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(
            [
                "iso_a3",
                "country_name",
                "admin1_name",
                "year",
                "month",
                "year_month",
                "metric",
                "series_mode",
                "harmonized",
                "lo90",
                "hi90",
                "raw",
                "z_score",
                "pct_of_median",
                "severity_tier",
                "constellation_regime",
                "data_maturity",
                "is_in_sama",
                "bbox",
                "radius_km",
                "generated_at_utc",
                "source",
            ]
        )
        for row in timeseries:
            writer.writerow(
                [
                    selection.get("iso_a3") or filters.get("iso_a3"),
                    selection.get("country_name") or "",
                    selection.get("admin1_name") or filters.get("admin1_name") or "",
                    row.get("year"),
                    row.get("month"),
                    row.get("year_month"),
                    payload.get("metric"),
                    payload.get("series_mode"),
                    row.get("harmonized"),
                    row.get("lo90"),
                    row.get("hi90"),
                    row.get("raw"),
                    row.get("z_score"),
                    row.get("pct_of_median"),
                    row.get("severity_tier"),
                    row.get("constellation_regime"),
                    row.get("data_maturity"),
                    row.get("is_in_sama"),
                    bbox_text,
                    filters.get("radius_km") or "",
                    generated,
                    selection.get("source") or "",
                ]
            )
        body = buffer.getvalue().encode("utf-8")
        headers = {
            "Content-Type": "text/csv; charset=utf-8",
            "Content-Disposition": 'attachment; filename="srehi_calendar_export.csv"',
            "Cache-Control": API_CACHE_CONTROL,
            "Content-Length": str(len(body)),
        }
        return body, headers

    # ------------------------------------------------------------------
    # Static files and dispatch
    # ------------------------------------------------------------------

    def _resolve_static(self, relative: str) -> Path:
        decoded = unquote(relative or "")
        if not decoded or "\x00" in decoded or "\\" in decoded:
            raise NotFound("Static asset not found.")
        parts = Path(decoded).parts
        if not parts or any(part in {"..", ""} for part in parts) or Path(decoded).is_absolute():
            raise NotFound("Static asset not found.")
        target = (self.static_root / Path(decoded)).resolve()
        try:
            target.relative_to(self.static_root)
        except ValueError as exc:
            raise NotFound("Static asset not found.") from exc
        if not target.is_file():
            raise NotFound("Static asset not found.")
        return target

    def _static_response(self, relative: str) -> tuple[int, list[tuple[str, str]], bytes]:
        target = self._resolve_static(relative)
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if target.suffix.lower() == ".js":
            content_type = "text/javascript"
        elif target.suffix.lower() == ".css":
            content_type = "text/css"
        elif target.suffix.lower() == ".svg":
            content_type = "image/svg+xml"
        elif target.suffix.lower() in {".html", ".htm"}:
            content_type = "text/html"
        if content_type.startswith("text/") or content_type in {
            "application/javascript",
            "application/json",
            "image/svg+xml",
        }:
            content_type = f"{content_type}; charset=utf-8"
        body = target.read_bytes()
        headers = [
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "public, max-age=300"),
            ("X-Content-Type-Options", "nosniff"),
            ("Referrer-Policy", "no-referrer"),
        ]
        if target.suffix.lower() == ".html":
            headers.append(
                (
                    "Content-Security-Policy",
                    "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                    "script-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'self'",
                )
            )
        return int(HTTPStatus.OK), headers, body

    def _dispatch(self, request: Request) -> tuple[int, list[tuple[str, str]], bytes]:
        if request.method not in {"GET", "HEAD"}:
            raise MethodNotAllowed(
                f"Method {request.method} is not allowed. Use GET.",
                status=HTTPStatus.METHOD_NOT_ALLOWED,
            )
        path = request.path
        if path in {"/", "/index.html"}:
            return self._static_response("index.html")
        if path.startswith("/static/"):
            return self._static_response(path[len("/static/") :])
        if path == "/api/health":
            return self._json_response(HTTPStatus.OK, self._health_payload())
        if path == "/api/catalog":
            return self._json_response(HTTPStatus.OK, self._catalog_payload())
        if path == "/api/calendar":
            return self._json_response(HTTPStatus.OK, self._calendar_payload(request))
        if path == "/api/early-onset":
            return self._json_response(HTTPStatus.OK, self._early_onset_payload(request))
        if path == "/api/spatial":
            return self._json_response(HTTPStatus.OK, self._spatial_payload(request))
        if path == "/api/export.csv":
            body, headers = self._export_csv(request)
            return int(HTTPStatus.OK), list(headers.items()), body
        if path == "/favicon.ico":
            raise NotFound("Static asset not found.")
        if path.startswith("/api/"):
            raise NotFound(f"Unknown API endpoint: {path}.")
        raise NotFound("Resource not found.")

    @staticmethod
    def _json_response(status: int, payload: Any) -> tuple[int, list[tuple[str, str]], bytes]:
        body = _json_bytes(payload)
        headers = [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("Cache-Control", API_CACHE_CONTROL),
            ("X-Content-Type-Options", "nosniff"),
            ("Referrer-Policy", "no-referrer"),
        ]
        return int(status), headers, body

    @staticmethod
    def _error_response(request: Request, error: HttpError) -> tuple[int, list[tuple[str, str]], bytes]:
        status = int(error.status)
        payload = {
            "error": error.code,
            "message": error.message,
            "status": status,
            "path": request.path,
            "generated_at_utc": _utc_now(),
        }
        body = _json_bytes(payload)
        headers = [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            ("Cache-Control", API_CACHE_CONTROL),
            ("X-Content-Type-Options", "nosniff"),
            ("Referrer-Policy", "no-referrer"),
        ]
        if status == int(HTTPStatus.METHOD_NOT_ALLOWED):
            headers.append(("Allow", "GET, HEAD"))
        return status, headers, body

    def _cache_key(self, request: Request) -> tuple[Any, ...] | None:
        if request.method not in {"GET", "HEAD"}:
            return None
        if not request.path.startswith("/api/") or request.path == "/api/health":
            return None
        fingerprint: list[tuple[str, int, int]] = []
        for name in PARQUET_FILES:
            path = self.processed / name
            try:
                stat = path.stat()
                fingerprint.append((name, int(stat.st_mtime_ns), int(stat.st_size)))
            except OSError:
                fingerprint.append((name, 0, 0))
        key: tuple[Any, ...] = (
            request.path,
            str(request.environ.get("QUERY_STRING") or ""),
            tuple(fingerprint),
        )
        if request.path == "/api/spatial":
            # Keep pagination and reference-window parameters explicit in the
            # cache identity instead of relying only on the raw query string.
            key = key + (
                request.one("offset", "0"),
                request.one("range_start", ""),
                request.one("range_end", ""),
            )
        return key

    def _cache_get(
        self, key: tuple[Any, ...]
    ) -> tuple[int, list[tuple[str, str]], bytes] | None:
        now = time.monotonic()
        with self._response_lock:
            entry = self._response_cache.get(key)
            if entry is None:
                return None
            created, status, headers, body = entry
            if now - created > self._response_cache_ttl:
                self._response_cache.pop(key, None)
                return None
            self._response_cache.move_to_end(key)
            return status, headers, body

    def _cache_put(
        self,
        key: tuple[Any, ...],
        status: int,
        headers: list[tuple[str, str]],
        body: bytes,
    ) -> None:
        if not (200 <= int(status) < 300):
            return
        with self._response_lock:
            self._response_cache[key] = (time.monotonic(), status, headers, body)
            self._response_cache.move_to_end(key)
            while len(self._response_cache) > self._response_cache_size:
                self._response_cache.popitem(last=False)

    def __call__(self, environ: Mapping[str, Any], start_response: Any) -> Iterable[bytes]:
        request = Request(environ)
        cache_key = self._cache_key(request)
        cached = self._cache_get(cache_key) if cache_key is not None else None
        if cached is not None:
            status, headers, body = cached
        else:
            try:
                status, headers, body = self._dispatch(request)
            except HttpError as error:
                status, headers, body = self._error_response(request, error)
            except Exception as exc:  # noqa: BLE001 - final safety net
                traceback.print_exc()
                error = HttpError(
                    f"Internal server error: {type(exc).__name__}: {exc}",
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    code="internal_error",
                )
                status, headers, body = self._error_response(request, error)
            if cache_key is not None:
                self._cache_put(cache_key, status, headers, body)
        if request.method == "HEAD":
            body = b""
        start_response(f"{int(status)} {HTTPStatus(status).phrase}", headers)
        return [body]


def _find_metric(payload: Any, key: str) -> float | None:
    """Find the first finite ``key`` inside a nested metrics payload."""

    if isinstance(payload, Mapping):
        for preferred in ("oof", "mean", "overall", "metrics"):
            nested = payload.get(preferred)
            if isinstance(nested, Mapping):
                value = nested.get(key)
                if value is not None:
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        number = math.nan
                    if math.isfinite(number):
                        return number
        for name, value in payload.items():
            if name == key:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    return number
            if isinstance(value, (Mapping, list)):
                found = _find_metric(value, key)
                if found is not None:
                    return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_metric(item, key)
            if found is not None:
                return found
    return None


def _doy_to_date(year: int, doy: int | None) -> str | None:
    if doy is None:
        return None
    try:
        return (dt.date(year, 1, 1) + dt.timedelta(days=int(doy) - 1)).isoformat()
    except (ValueError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# Threaded server and CLI
# ---------------------------------------------------------------------------


class ThreadingHTTPServer(ThreadingMixIn, WSGIServer):
    """A multithreaded WSGI server used by the CLI and the test-suite."""

    daemon_threads = True
    allow_reuse_address = True


def create_app(processed_dir: Path | str) -> FireRelayApplication:
    """Build the WSGI application.

    Parameters are intentionally strings/Path like so callers can pass either
    the repository-relative defaults or a temporary fixture directory.
    """

    return FireRelayApplication(processed_dir=processed_dir)


def create_server(
    app: FireRelayApplication,
    host: str = "127.0.0.1",
    port: int = 0,
) -> WSGIServer:
    """Create a threaded WSGI server bound to ``host:port``."""

    return make_server(
        host,
        int(port),
        app,
        server_class=ThreadingHTTPServer,
        handler_class=WSGIRequestHandler,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.server",
        description=(
            "Fire Relay South America — harmonized burn calendar and early-onset "
            "observatory."
        ),
    )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    application = create_app(args.processed_dir)
    server = create_server(application, host=args.host, port=int(args.port))
    host, port = server.server_address[:2]
    print(
        f"Fire Relay South America listening on http://{host}:{port} "
        f"(processed={application.processed})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive CLI
        print("\nStopping.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
