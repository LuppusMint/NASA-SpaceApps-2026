"""Contract C3 integration tests for the bilingual burning-calendar web app.

The fixture is intentionally self-contained: it writes a small but complete set
of C1/C2 Parquet cubes and JSON summaries into ``tempfile.TemporaryDirectory``
and exercises the same WSGI application through both ``httpx.WSGITransport``
and a real multithreaded socket server.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.server import create_app, create_server  # noqa: E402

COUNTRIES = {
    "BRA": {
        "name": "Brazil",
        "bbox": [-33.75, 5.27, -73.99, -34.79],
        "weight": 120.0,
        "sama": False,
        "admin1": [
            ("Amazonas", [-9.0, 2.0, -73.0, -60.0], 0.55),
            ("Mato Grosso", [-18.0, -9.0, -60.0, -50.0], 0.45),
        ],
    },
    "ARG": {
        "name": "Argentina",
        "bbox": [-55.25, -21.78, -73.58, -53.64],
        "weight": 55.0,
        "sama": True,
        "admin1": [
            ("Chaco", [-28.0, -24.0, -63.0, -58.0], 0.6),
            ("Patagonia", [-50.0, -40.0, -72.0, -65.0], 0.4),
        ],
    },
    "COL": {
        "name": "Colombia",
        "bbox": [-4.23, 12.46, -79.0, -66.87],
        "weight": 35.0,
        "sama": True,
        "admin1": [
            ("Meta", [2.0, 6.0, -74.0, -71.0], 0.5),
            ("Vichada", [1.0, 6.0, -71.0, -67.0], 0.5),
        ],
    },
}

YEARS = list(range(2000, 2026))
MONTHS = list(range(1, 13))
YEAR_MONTHS = [(year, month) for year in YEARS for month in MONTHS] + [(2026, 10), (2026, 11)]
TRANSITIONS = ["2002", "2012", "2018", "2021", "2023"]


def _regime(year: int) -> str:
    if year <= 2002:
        return "TERRA_SCALED"
    if year <= 2011:
        return "TERRA_AQUA"
    if year <= 2020:
        return "MODIS_VIIRS_OVERLAP"
    return "VIIRS_BACKBONE"


def _season(month: int, phase: float = 0.0) -> float:
    return 0.55 + 1.15 * (0.5 + 0.5 * math.cos((month - 9 + phase) * math.pi / 6.0))


def _write_parquet(path: Path, rows: list[dict], schema: pa.Schema | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _severity(value: float, p25: float, p75: float, p90: float) -> str:
    if value > p90:
        return "CRITICAL_EXTREME"
    if value > p75:
        return "HIGH_UNUSUAL"
    if value > p25:
        return "MODERATE"
    return "LOW_NORMAL"


def _monthly_row(
    iso: str,
    country_name: str,
    year: int,
    month: int,
    *,
    admin1_name: str | None,
    weight: float,
    sama: bool,
    share: float = 1.0,
) -> dict:
    year_month = f"{year:04d}-{month:02d}"
    seasonal = _season(month, phase=0.2 if admin1_name else 0.0)
    trend = 1.0 + 0.012 * (year - 2000)
    pulse = 1.0 + 0.08 * math.sin((year * 12 + month) / 7.0)
    raw_events = max(1, int(round(weight * share * seasonal * trend * pulse)))
    harmonized = round(raw_events * 1.24 + 3.0, 2)
    raw_frp = round(raw_events * 3.4, 2)
    harmonized_frp = round(raw_frp * 1.38 + 5.0, 2)
    maturity = "STANDARD_SCIENCE"
    if (year, month) in {(2025, 12), (2026, 11)}:
        maturity = "PARTIAL_MONTH"
    elif (year, month) in {(2025, 11), (2026, 10)}:
        maturity = "NRT_PRELIMINARY"
    return {
        "year": year,
        "month": month,
        "year_month": year_month,
        "country_name": country_name,
        "iso_a3": iso,
        "admin1_name": admin1_name,
        "raw_all_det_count": raw_events * 5,
        "clean_det_count": raw_events * 4,
        "static_excluded_det_count": max(0, raw_events // 8),
        "high_conf_count": int(raw_events * 0.8),
        "night_det_count": int(raw_events * 0.2),
        "overpass_event_count": raw_events,
        "raw_frp_mw": raw_frp,
        "active_cells_10km": max(1, raw_events // 3),
        "nrt_det_count": 12 if maturity == "NRT_PRELIMINARY" else 0,
        "sp_det_count": 240,
        "intersects_sama": bool(sama),
        "is_partial_month": maturity == "PARTIAL_MONTH",
        "is_in_sama": bool(sama),
        "sama_suspect_det_count": 3 if sama else 0,
        "harmonized_srehi": harmonized,
        "harmonized_srehi_lo90": round(harmonized * 0.80, 2),
        "harmonized_srehi_hi90": round(harmonized * 1.20, 2),
        "harmonized_frp_mw": harmonized_frp,
        "harmonized_frp_lo90": round(harmonized_frp * 0.82, 2),
        "harmonized_frp_hi90": round(harmonized_frp * 1.18, 2),
        "m1_srehi": round(harmonized * 0.96, 2),
        "m2_srehi": round(harmonized * 1.03, 2),
        "m1_frp_mw": round(harmonized_frp * 0.95, 2),
        "m2_frp_mw": round(harmonized_frp * 1.04, 2),
        "constellation_regime": _regime(year),
        "data_maturity": maturity,
    }


def build_fixture(root: Path) -> Path:
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    admin1_rows: list[dict] = []
    for iso, config in COUNTRIES.items():
        for admin1_name, _bbox, share in config["admin1"]:
            for year, month in YEAR_MONTHS:
                    admin1_rows.append(
                        _monthly_row(
                            iso,
                            config["name"],
                            year,
                            month,
                            admin1_name=admin1_name,
                            weight=config["weight"],
                            sama=config["sama"],
                            share=share,
                        )
                    )
    _write_parquet(processed / "harmonized_admin1_monthly.parquet", admin1_rows)

    # Country rows are exact sums of their Admin-1 children.
    country_rows: list[dict] = []
    for iso, config in COUNTRIES.items():
        for year, month in YEAR_MONTHS:
                children = [
                    row
                    for row in admin1_rows
                    if row["iso_a3"] == iso and row["year"] == year and row["month"] == month
                ]
                total = sum(row["harmonized_srehi"] for row in children)
                total_frp = sum(row["harmonized_frp_mw"] for row in children)
                raw_events = sum(row["overpass_event_count"] for row in children)
                raw_frp = sum(row["raw_frp_mw"] for row in children)
                country_rows.append(
                    {
                        "year": year,
                        "month": month,
                        "year_month": f"{year:04d}-{month:02d}",
                        "country_name": config["name"],
                        "iso_a3": iso,
                        "raw_all_det_count": sum(row["raw_all_det_count"] for row in children),
                        "clean_det_count": sum(row["clean_det_count"] for row in children),
                        "static_excluded_det_count": sum(
                            row["static_excluded_det_count"] for row in children
                        ),
                        "high_conf_count": sum(row["high_conf_count"] for row in children),
                        "night_det_count": sum(row["night_det_count"] for row in children),
                        "overpass_event_count": raw_events,
                        "raw_frp_mw": round(raw_frp, 2),
                        "active_cells_10km": sum(row["active_cells_10km"] for row in children),
                        "nrt_det_count": sum(row["nrt_det_count"] for row in children),
                        "sp_det_count": sum(row["sp_det_count"] for row in children),
                        "intersects_sama": config["sama"],
                        "is_partial_month": any(row["is_partial_month"] for row in children),
                        "is_in_sama": config["sama"],
                        "sama_suspect_det_count": sum(
                            row["sama_suspect_det_count"] for row in children
                        ),
                        "harmonized_srehi": round(total, 2),
                        "harmonized_srehi_lo90": round(total * 0.80, 2),
                        "harmonized_srehi_hi90": round(total * 1.20, 2),
                        "harmonized_frp_mw": round(total_frp, 2),
                        "harmonized_frp_lo90": round(total_frp * 0.82, 2),
                        "harmonized_frp_hi90": round(total_frp * 1.18, 2),
                        "m1_srehi": round(total * 0.96, 2),
                        "m2_srehi": round(total * 1.03, 2),
                        "m1_frp_mw": round(total_frp * 0.95, 2),
                        "m2_frp_mw": round(total_frp * 1.04, 2),
                        "constellation_regime": _regime(year),
                        "data_maturity": next(
                            (
                                row["data_maturity"]
                                for row in children
                                if row["data_maturity"] != "STANDARD_SCIENCE"
                            ),
                            "STANDARD_SCIENCE",
                        ),
                    }
                )
    _write_parquet(processed / "harmonized_country_monthly.parquet", country_rows)

    south_rows: list[dict] = []
    for year, month in YEAR_MONTHS:
            children = [
                row for row in country_rows if row["year"] == year and row["month"] == month
            ]
            total = sum(row["harmonized_srehi"] for row in children)
            total_frp = sum(row["harmonized_frp_mw"] for row in children)
            south_rows.append(
                {
                    "year": year,
                    "month": month,
                    "year_month": f"{year:04d}-{month:02d}",
                    "harmonized_srehi": round(total, 2),
                    "harmonized_srehi_lo90": round(total * 0.80, 2),
                    "harmonized_srehi_hi90": round(total * 1.20, 2),
                    "harmonized_frp_mw": round(total_frp, 2),
                    "harmonized_frp_lo90": round(total_frp * 0.82, 2),
                    "harmonized_frp_hi90": round(total_frp * 1.18, 2),
                    "raw_all_det_count": sum(row["raw_all_det_count"] for row in children),
                    "clean_det_count": sum(row["clean_det_count"] for row in children),
                    "overpass_event_count": sum(row["overpass_event_count"] for row in children),
                    "raw_frp_mw": round(sum(row["raw_frp_mw"] for row in children), 2),
                    "nrt_det_count": sum(row["nrt_det_count"] for row in children),
                    "sp_det_count": sum(row["sp_det_count"] for row in children),
                    "is_partial_month": any(row["is_partial_month"] for row in children),
                    "n_countries": len(children),
                    "is_in_sama": True,
                    "constellation_regime": _regime(year),
                    "data_maturity": next(
                        (
                            row["data_maturity"]
                            for row in children
                            if row["data_maturity"] != "STANDARD_SCIENCE"
                        ),
                        "STANDARD_SCIENCE",
                    ),
                }
            )
    _write_parquet(processed / "harmonized_south_america_monthly.parquet", south_rows)

    # C1 raw cubes (a single sensor slot is enough for fallback smoke tests).
    cube_country = []
    for row in country_rows:
        cube_country.append(
            {
                "year": row["year"],
                "month": row["month"],
                "year_month": row["year_month"],
                "country_name": row["country_name"],
                "iso_a3": row["iso_a3"],
                "sensor_slot": "MODIS_Aqua",
                "raw_all_det_count": row["raw_all_det_count"],
                "clean_det_count": row["clean_det_count"],
                "static_excluded_det_count": row["static_excluded_det_count"],
                "high_conf_count": row["high_conf_count"],
                "night_det_count": row["night_det_count"],
                "overpass_event_count": row["overpass_event_count"],
                "frp_sum_mw": row["raw_frp_mw"],
                "unique_active_cells_10km": row["active_cells_10km"],
                "nrt_det_count": row["nrt_det_count"],
                "sp_det_count": row["sp_det_count"],
                "is_nrt_majority": row["data_maturity"] == "NRT_PRELIMINARY",
                "is_partial_month": row["is_partial_month"],
                "intersects_sama": row["intersects_sama"],
            }
        )
    _write_parquet(processed / "cube_country_monthly.parquet", cube_country)
    cube_admin1 = []
    for row in admin1_rows:
        cube_admin1.append(
            {
                "year": row["year"],
                "month": row["month"],
                "year_month": row["year_month"],
                "country_name": row["country_name"],
                "iso_a3": row["iso_a3"],
                "admin1_name": row["admin1_name"],
                "sensor_slot": "MODIS_Aqua",
                "raw_all_det_count": row["raw_all_det_count"],
                "clean_det_count": row["clean_det_count"],
                "static_excluded_det_count": row["static_excluded_det_count"],
                "high_conf_count": row["high_conf_count"],
                "night_det_count": row["night_det_count"],
                "overpass_event_count": row["overpass_event_count"],
                "frp_sum_mw": row["raw_frp_mw"],
                "unique_active_cells_10km": row["active_cells_10km"],
                "nrt_det_count": row["nrt_det_count"],
                "sp_det_count": row["sp_det_count"],
                "is_nrt_majority": row["data_maturity"] == "NRT_PRELIMINARY",
                "is_partial_month": row["is_partial_month"],
                "intersects_sama": row["intersects_sama"],
            }
        )
    _write_parquet(processed / "cube_admin1_monthly.parquet", cube_admin1)

    # 50 km harmonized cell cube: centre + four corners per country.
    cell50_rows: list[dict] = []
    offsets = [(0.0, 0.0), (-0.8, -0.8), (-0.8, 0.8), (0.8, -0.8), (0.8, 0.8)]
    for iso, config in COUNTRIES.items():
        min_lat, max_lat, min_lon, max_lon = config["bbox"]
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0
        for cell_index, (d_lat, d_lon) in enumerate(offsets):
            cell_lat = center_lat + d_lat
            cell_lon = center_lon + d_lon
            for year, month in YEAR_MONTHS:
                    base = next(
                        row
                        for row in country_rows
                        if row["iso_a3"] == iso and row["year"] == year and row["month"] == month
                    )
                    share = 0.2 + 0.02 * cell_index
                    cell50_rows.append(
                        {
                            "year": year,
                            "month": month,
                            "year_month": base["year_month"],
                            "cell_ix_50km": cell_index,
                            "cell_iy_50km": cell_index + 1,
                            "cell_id_50km": f"{iso}_50_{cell_index}",
                            "country_name": config["name"],
                            "iso_a3": iso,
                            "mean_lat": round(cell_lat, 5),
                            "mean_lon": round(cell_lon, 5),
                            "raw_all_det_count": int(base["raw_all_det_count"] * share),
                            "clean_det_count": int(base["clean_det_count"] * share),
                            "static_excluded_det_count": int(
                                base["static_excluded_det_count"] * share
                            ),
                            "night_det_count": int(base["night_det_count"] * share),
                            "overpass_event_count": max(
                                1, int(base["overpass_event_count"] * share)
                            ),
                            "frp_sum_mw": round(base["raw_frp_mw"] * share, 2),
                            "raw_frp_mw": round(base["raw_frp_mw"] * share, 2),
                            "active_cells_10km": max(1, int(base["active_cells_10km"] * share)),
                            "harmonized_srehi": round(base["harmonized_srehi"] * share, 2),
                            "harmonized_srehi_lo90": round(
                                base["harmonized_srehi"] * share * 0.8, 2
                            ),
                            "harmonized_srehi_hi90": round(
                                base["harmonized_srehi"] * share * 1.2, 2
                            ),
                            "harmonized_frp_mw": round(base["harmonized_frp_mw"] * share, 2),
                            "harmonized_frp_lo90": round(
                                base["harmonized_frp_mw"] * share * 0.82, 2
                            ),
                            "harmonized_frp_hi90": round(
                                base["harmonized_frp_mw"] * share * 1.18, 2
                            ),
                            "m1_srehi": round(base["m1_srehi"] * share, 2),
                            "m2_srehi": round(base["m2_srehi"] * share, 2),
                            "m1_frp_mw": round(base["m1_frp_mw"] * share, 2),
                            "m2_frp_mw": round(base["m2_frp_mw"] * share, 2),
                            "constellation_regime": base["constellation_regime"],
                            "data_maturity": base["data_maturity"],
                            "is_in_sama": config["sama"],
                            "sama_suspect_det_count": 1 if config["sama"] else 0,
                        }
                    )
    _write_parquet(processed / "harmonized_cell50km_monthly.parquet", cell50_rows)

    # 10 km raw cube: four cells around the country centre.
    cell10_rows: list[dict] = []
    offsets10 = [(-0.05, -0.05), (-0.05, 0.05), (0.05, -0.05), (0.05, 0.05)]
    for iso, config in COUNTRIES.items():
        min_lat, max_lat, min_lon, max_lon = config["bbox"]
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0
        first_admin1 = config["admin1"][0][0]
        for cell_index, (d_lat, d_lon) in enumerate(offsets10):
            for year, month in YEAR_MONTHS:
                    base = next(
                        row
                        for row in country_rows
                        if row["iso_a3"] == iso and row["year"] == year and row["month"] == month
                    )
                    share = 0.05
                    cell10_rows.append(
                        {
                            "year": year,
                            "month": month,
                            "year_month": base["year_month"],
                            "cell_id_10km": f"{iso}_10_{cell_index}",
                            "cell_ix_10km": cell_index,
                            "cell_iy_10km": cell_index + 10,
                            "country_name": config["name"],
                            "iso_a3": iso,
                            "admin1_name": first_admin1,
                            "sensor_slot": "VIIRS_SNPP",
                            "is_in_sama": config["sama"],
                            "clean_det_count": max(1, int(base["clean_det_count"] * share)),
                            "static_excluded_det_count": 0,
                            "overpass_event_count": max(
                                1, int(base["overpass_event_count"] * share)
                            ),
                            "frp_sum_mw": round(base["raw_frp_mw"] * share, 2),
                            "mean_lat": round(center_lat + d_lat, 5),
                            "mean_lon": round(center_lon + d_lon, 5),
                            "is_nrt_majority": base["data_maturity"] == "NRT_PRELIMINARY",
                            "is_partial_month": base["is_partial_month"],
                        }
                    )
    _write_parquet(processed / "cube_cell10km_monthly.parquet", cell10_rows)

    # Persistent static thermal cells.
    static_rows = [
        {
            "cell_id_10km": "BRA_10_0",
            "det_count": 5400,
            "distinct_ym": 280,
            "active_years": 24,
            "avg_months_per_year": 11.2,
            "type_12_count": 4900,
            "typed_count": 5100,
            "type_12_ratio": 0.96,
            "first_date": date(2001, 1, 4),
            "last_date": date(2025, 12, 1),
            "mean_lat": -14.2,
            "mean_lon": -52.1,
            "is_persistent_temporal": True,
            "is_type_12_dominant": True,
        },
        {
            "cell_id_10km": "BRA_10_1",
            "det_count": 3100,
            "distinct_ym": 230,
            "active_years": 19,
            "avg_months_per_year": 10.1,
            "type_12_count": 2200,
            "typed_count": 2600,
            "type_12_ratio": 0.85,
            "first_date": date(2002, 3, 1),
            "last_date": date(2024, 11, 1),
            "mean_lat": -15.1,
            "mean_lon": -53.4,
            "is_persistent_temporal": True,
            "is_type_12_dominant": True,
        },
        {
            "cell_id_10km": "COL_10_0",
            "det_count": 1800,
            "distinct_ym": 180,
            "active_years": 17,
            "avg_months_per_year": 10.5,
            "type_12_count": 900,
            "typed_count": 1100,
            "type_12_ratio": 0.82,
            "first_date": date(2004, 6, 1),
            "last_date": date(2025, 10, 1),
            "mean_lat": 4.6,
            "mean_lon": -72.3,
            "is_persistent_temporal": True,
            "is_type_12_dominant": True,
        },
        {
            "cell_id_10km": "ARG_10_0",
            "det_count": 900,
            "distinct_ym": 120,
            "active_years": 12,
            "avg_months_per_year": 10.0,
            "type_12_count": 300,
            "typed_count": 700,
            "type_12_ratio": 0.43,
            "first_date": date(2008, 2, 1),
            "last_date": date(2023, 12, 1),
            "mean_lat": -25.4,
            "mean_lon": -60.8,
            "is_persistent_temporal": True,
            "is_type_12_dominant": False,
        },
        {
            "cell_id_10km": "OFF_10_0",
            "det_count": 420,
            "distinct_ym": 90,
            "active_years": 9,
            "avg_months_per_year": 10.0,
            "type_12_count": 0,
            "typed_count": 20,
            "type_12_ratio": 0.0,
            "first_date": date(2010, 1, 1),
            "last_date": date(2020, 1, 1),
            "mean_lat": -22.0,
            "mean_lon": -38.0,
            "is_persistent_temporal": True,
            "is_type_12_dominant": False,
        },
        {
            "cell_id_10km": "BRA_10_2",
            "det_count": 600,
            "distinct_ym": 95,
            "active_years": 10,
            "avg_months_per_year": 9.5,
            "type_12_count": 500,
            "typed_count": 550,
            "type_12_ratio": 0.91,
            "first_date": date(2011, 5, 1),
            "last_date": date(2024, 9, 1),
            "mean_lat": -12.4,
            "mean_lon": -50.2,
            "is_persistent_temporal": True,
            "is_type_12_dominant": True,
        },
    ]
    _write_parquet(processed / "static_thermal_cells.parquet", static_rows)

    # Frozen climatology 2001-2020 for every geo group.
    climatology_rows: list[dict] = []
    for iso, config in COUNTRIES.items():
        for admin1_name, _bbox, share in config["admin1"]:
            weight = config["weight"] * share
            for month in MONTHS:
                base = weight * _season(month, phase=0.2) * 1.1
                p25, p50, p75, p90, p95 = (
                    base * 0.70,
                    base,
                    base * 1.30,
                    base * 1.62,
                    base * 1.90,
                )
                climatology_rows.append(
                    {
                        "geo_level": "admin1",
                        "iso_a3": iso,
                        "country_name": config["name"],
                        "admin1_name": admin1_name,
                        "month": month,
                        "n_years": 20,
                        "srehi_p25": round(p25, 2),
                        "srehi_p50": round(p50, 2),
                        "srehi_p75": round(p75, 2),
                        "srehi_iqr": round(p75 - p25, 2),
                        "srehi_p90": round(p90, 2),
                        "srehi_p95": round(p95, 2),
                        "srehi_mean": round(p50, 2),
                        "srehi_std": round(p50 * 0.22, 2),
                        "frp_p25": round(p25 * 3.4, 2),
                        "frp_p50": round(p50 * 3.4, 2),
                        "frp_p75": round(p75 * 3.4, 2),
                        "frp_iqr": round((p75 - p25) * 3.4, 2),
                        "frp_p90": round(p90 * 3.4, 2),
                        "frp_p95": round(p95 * 3.4, 2),
                        "frp_mean": round(p50 * 3.4, 2),
                        "frp_std": round(p50 * 0.8, 2),
                    }
                )
        for month in MONTHS:
            base = config["weight"] * _season(month) * 1.1
            p25, p50, p75, p90, p95 = (
                base * 0.70,
                base,
                base * 1.30,
                base * 1.62,
                base * 1.90,
            )
            climatology_rows.append(
                {
                    "geo_level": "country",
                    "iso_a3": iso,
                    "country_name": config["name"],
                    "admin1_name": "",
                    "month": month,
                    "n_years": 20,
                    "srehi_p25": round(p25, 2),
                    "srehi_p50": round(p50, 2),
                    "srehi_p75": round(p75, 2),
                    "srehi_iqr": round(p75 - p25, 2),
                    "srehi_p90": round(p90, 2),
                    "srehi_p95": round(p95, 2),
                    "srehi_mean": round(p50, 2),
                    "srehi_std": round(p50 * 0.22, 2),
                    "frp_p25": round(p25 * 3.4, 2),
                    "frp_p50": round(p50 * 3.4, 2),
                    "frp_p75": round(p75 * 3.4, 2),
                    "frp_iqr": round((p75 - p25) * 3.4, 2),
                    "frp_p90": round(p90 * 3.4, 2),
                    "frp_p95": round(p95 * 3.4, 2),
                    "frp_mean": round(p50 * 3.4, 2),
                    "frp_std": round(p50 * 0.8, 2),
                }
            )
    _write_parquet(processed / "calendar_climatology.parquet", climatology_rows)

    # Calendar matrices add z-scores, percent-of-median and severity tiers.
    def add_matrix_columns(rows: list[dict]) -> list[dict]:
        climatology = {
            (row["iso_a3"], row["admin1_name"], row["month"]): row
            for row in climatology_rows
            if row["geo_level"] == ("admin1" if row.get("admin1_name") else "country")
        }
        output = []
        for row in rows:
            key = (row["iso_a3"], row.get("admin1_name", ""), row["month"])
            clim = climatology.get(key)
            if clim is None:
                clim = next(
                    item
                    for item in climatology_rows
                    if item["iso_a3"] == row["iso_a3"]
                    and item["month"] == row["month"]
                    and item["geo_level"] == "country"
                )
            srehi = row["harmonized_srehi"]
            frp = row["harmonized_frp_mw"]
            z_srehi = (srehi - clim["srehi_mean"]) / max(1e-9, clim["srehi_std"])
            z_frp = (frp - clim["frp_mean"]) / max(1e-9, clim["frp_std"])
            output.append(
                {
                    **row,
                    "z_score_srehi": round(z_srehi, 4),
                    "pct_of_median_srehi": round(100.0 * srehi / max(1e-9, clim["srehi_p50"]), 3),
                    "z_score_frp": round(z_frp, 4),
                    "pct_of_median_frp": round(100.0 * frp / max(1e-9, clim["frp_p50"]), 3),
                    "severity_tier": _severity(
                        srehi, clim["srehi_p25"], clim["srehi_p75"], clim["srehi_p90"]
                    ),
                    "severity_tier_frp": _severity(
                        frp, clim["frp_p25"], clim["frp_p75"], clim["frp_p90"]
                    ),
                }
            )
        return output

    _write_parquet(
        processed / "calendar_matrix_admin1.parquet", add_matrix_columns(admin1_rows)
    )
    _write_parquet(
        processed / "calendar_matrix_country.parquet", add_matrix_columns(country_rows)
    )

    trend_rows: list[dict] = []
    for geo_level, source_rows in (("admin1", admin1_rows), ("country", country_rows)):
        groups = {}
        for row in source_rows:
            key = (row["iso_a3"], row["country_name"], row.get("admin1_name", ""))
            groups.setdefault(key, None)
        for (iso, country_name, admin1_name) in groups:
            for metric in ("harmonized_srehi", "harmonized_frp_mw"):
                trend_rows.append(
                    {
                        "geo_level": geo_level,
                        "iso_a3": iso,
                        "country_name": country_name,
                        "admin1_name": admin1_name,
                        "metric": metric,
                        "n_years": 25,
                        "start_year": 2001,
                        "end_year": 2025,
                        "slope": 1.25 if metric == "harmonized_srehi" else 4.2,
                        "intercept": 100.0,
                        "slope_ci_low": 0.44,
                        "slope_ci_high": 2.06,
                        "z": 2.71,
                        "p_value": 0.0067,
                        "method": "hamed_rao_mk",
                        "ess": 18.2,
                        "annual_mean": 140.0,
                        "gev_fit_ok": True,
                        "gev_n": 5,
                        "gev_xi": -0.12,
                        "gev_mu": 120.0,
                        "gev_sigma": 22.0,
                        "gev_rl10": 210.0,
                        "gev_rl20": 260.0,
                        "gev_rl50": 340.0,
                        "rl10": 210.0,
                        "rl20": 260.0,
                        "rl50": 340.0,
                        "gev_ks_p_value": 0.44,
                        "q_value": 0.0134,
                        "significant_fdr": True,
                        "trend_direction": "increasing",
                    }
                )
    _write_parquet(processed / "calendar_trends.parquet", trend_rows)

    # 90-day early-onset curves for every Admin-1 unit and year.
    early_rows: list[dict] = []
    for iso, config in COUNTRIES.items():
        for admin1_name, _bbox, share in config["admin1"]:
            weight = config["weight"] * share
            for year in YEARS:
                doys = list(range(1, 367, 5))
                if doys[-1] != 366:
                    doys.append(366)
                for doy in doys:
                    seasonal = _season(1 + ((doy - 1) // 30) % 12, phase=0.2)
                    p75 = weight * seasonal * 1.15
                    p90 = p75 * 1.30
                    if year == 2024 and 100 <= doy <= 160:
                        signal = p90 * 1.45
                    elif year == 2024 and 70 <= doy < 100:
                        signal = p75 * 1.12
                    else:
                        signal = p75 * 0.72
                    events = max(1.0, signal / 9.0)
                    flag = (
                        "EARLY_ONSET_CRITICAL"
                        if signal > p90
                        else "EARLY_ONSET_HIGH"
                        if signal > p75
                        else "NORMAL"
                    )
                    early_rows.append(
                        {
                            "iso_a3": iso,
                            "country_name": config["name"],
                            "admin1_name": admin1_name,
                            "year": year,
                            "doy": doy,
                            "acq_date": date(year, 1, 1) + timedelta(days=doy - 1),
                            "events": round(events, 3),
                            "scaled_events": round(events * 1.24, 3),
                            "srehi_90d": round(signal, 3),
                            "p75_90d": round(p75, 3),
                            "p90_90d": round(p90, 3),
                            "early_onset_flag": flag,
                        }
                    )
    _write_parquet(processed / "calendar_early_onset_90d.parquet", early_rows)

    # Geography catalogue.
    catalog_countries = []
    catalog_admin1 = []
    for iso, config in COUNTRIES.items():
        min_lat, max_lat, min_lon, max_lon = config["bbox"]
        children = []
        for admin1_name, bbox, _share in config["admin1"]:
            child = {
                "country_name": config["name"],
                "iso_a3": iso,
                "admin1_name": admin1_name,
                "bbox": bbox,
                "intersects_sama": config["sama"],
                "historical_active_cells_10km": 1200,
            }
            children.append(child)
            catalog_admin1.append(child)
        catalog_countries.append(
            {
                "country_name": config["name"],
                "iso_a3": iso,
                "bbox": config["bbox"],
                "intersects_sama": config["sama"],
                "historical_active_cells_10km": 5000,
                "admin1_count": len(children),
                "admin1": children,
            }
        )
    _write_json(
        processed / "geography_catalog.json",
        {
            "generated_at_utc": "2026-11-15T06:22:59+00:00",
            "bbox_order": ["min_lat", "max_lat", "min_lon", "max_lon"],
            "countries": catalog_countries,
            "admin1": catalog_admin1,
            "totals": {"countries": len(catalog_countries), "admin1": len(catalog_admin1)},
        },
    )
    _write_json(
        processed / "qc_cleaning_summary.json",
        {
            "generated_at_utc": "2026-11-15T06:18:18+00:00",
            "raw_all_det_count": 85002288,
            "clean_det_count": 75556207,
            "static_excluded_rows": 2872511,
            "static_cell_count": len(static_rows),
            "sama_rows": 76388628,
            "nrt_rows": 2613221,
            "by_sensor_slot": [
                {
                    "sensor_slot": "MODIS_Aqua",
                    "raw_all_det_count": 10897858,
                    "clean_det_count": 10162384,
                },
                {
                    "sensor_slot": "VIIRS_SNPP",
                    "raw_all_det_count": 36304172,
                    "clean_det_count": 31965171,
                },
            ],
        },
    )
    transitions = {}
    for label in TRANSITIONS:
        transitions[label] = {
            "start_month": f"{label}-01",
            "window_months": 12,
            "srehi": {
                "raw_step_change_ratio": 1.42,
                "harmonized_step_change_ratio": 1.05,
                "ratio_reduction": 0.88,
                "passed": True,
            },
            "frp_mw": {
                "raw_step_change_ratio": 1.31,
                "harmonized_step_change_ratio": 1.03,
                "ratio_reduction": 0.90,
                "passed": True,
            },
        }
    _write_json(
        processed / "benchmarks_m0_m3.json",
        {
            "schema": "c2_benchmarks_v1",
            "generated_at": "2026-11-15T07:00:00+00:00",
            "folds": 5,
            "transitions": transitions,
            "dedup": {
                "n20_marginal_ratio_events": 0.18,
                "n21_marginal_ratio_events": 0.09,
                "n20_marginal_ratio_frp": 0.15,
                "n21_marginal_ratio_frp": 0.07,
            },
            "cross_regime_guard": {"passed": True, "n_checks": 5},
            "models": {
                "M0_raw_unharmonized": {
                    "srehi": {"oof": {"rmse": 0.612}, "mean": {"rmse": 0.640}},
                    "frp_mw": {"oof": {"rmse": 0.820}, "mean": {"rmse": 0.850}},
                },
                "M1_physical_biweight_bridge": {
                    "srehi": {"oof": {"rmse": 0.441}, "mean": {"rmse": 0.460}},
                    "frp_mw": {"oof": {"rmse": 0.620}, "mean": {"rmse": 0.640}},
                },
                "M2_hurdle_tweedie_covshift": {
                    "srehi": {"oof": {"rmse": 0.382}, "mean": {"rmse": 0.400}},
                    "frp_mw": {"oof": {"rmse": 0.540}, "mean": {"rmse": 0.560}},
                },
                "M3_simplex_kalman_reconciled": {
                    "srehi": {"oof": {"rmse": 0.311}, "mean": {"rmse": 0.325}},
                    "frp_mw": {"oof": {"rmse": 0.462}, "mean": {"rmse": 0.480}},
                },
            },
            "step_change_ratios": {
                label: {
                    "raw": 1.42,
                    "harmonized": 1.05,
                    "ratio_reduction": 0.88,
                    "frp_raw": 1.31,
                    "frp_harmonized": 1.03,
                }
                for label in TRANSITIONS
            },
        },
    )
    return processed


class WebAppApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        cls.processed = build_fixture(cls.root)
        cls.app = create_app(cls.processed)
        cls.client = httpx.Client(
            transport=httpx.WSGITransport(app=cls.app),
            base_url="http://testserver",
        )
        # Warm the thread-local DuckDB connection and schema cache.
        for path in (
            "/api/health",
            "/api/catalog",
            "/api/calendar?iso_a3=BRA",
            "/api/calendar?iso_a3=BRA&bbox=-20,5,-75,-45",
            "/api/spatial?year_month=2025-12&iso_a3=BRA",
        ):
            response = cls.client.get(path)
            assert response.status_code == 200, (path, response.status_code, response.text)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        cls.temp_dir.cleanup()

    # -- health and catalog -------------------------------------------------

    def test_health_reports_qc_and_benchmarks(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/json", response.headers["content-type"])
        self.assertEqual(response.headers["cache-control"], "no-store")
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["duckdb"]["threads"], 2)
        self.assertEqual(payload["duckdb"]["memory_limit"], "1GB")
        self.assertEqual(payload["qc_cleaning_summary"]["raw_all_det_count"], 85002288)
        self.assertEqual(payload["qc_cleaning_summary"]["clean_det_count"], 75556207)
        benchmarks = payload["benchmarks_m0_m3"]
        self.assertTrue(benchmarks["available"])
        self.assertIn("M0_raw_unharmonized", benchmarks["models"])
        self.assertIn("M3_simplex_kalman_reconciled", benchmarks["models"])
        self.assertIn("2012", benchmarks["transitions"])
        for model in ("M0", "M1", "M2", "M3"):
            self.assertTrue(any(item["model"].startswith(model) for item in benchmarks["m0_m3"]))
        available = {item["name"]: item for item in payload["parquets"]}
        for name in (
            "harmonized_admin1_monthly.parquet",
            "harmonized_country_monthly.parquet",
            "harmonized_cell50km_monthly.parquet",
            "calendar_early_onset_90d.parquet",
        ):
            self.assertTrue(available[name]["available"], name)
            self.assertIsNotNone(available[name]["query_ms"])
        self.assertLess(payload["response_time_ms"], 5000.0)

    def test_catalog_hierarchy_ranges_and_bboxes(self) -> None:
        response = self.client.get("/api/catalog")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["bbox_order"], ["min_lat", "max_lat", "min_lon", "max_lon"])
        self.assertEqual(len(payload["countries"]), 3)
        brazil = next(item for item in payload["countries"] if item["iso_a3"] == "BRA")
        self.assertEqual(brazil["country_name"], "Brazil")
        self.assertEqual(brazil["min_year_month"], "2000-01")
        self.assertEqual(brazil["max_year_month"], "2026-11")
        self.assertEqual(len(brazil["admin1"]), 2)
        names = {item["admin1_name"] for item in brazil["admin1"]}
        self.assertEqual(names, {"Amazonas", "Mato Grosso"})
        self.assertEqual(len(payload["admin1"]), 6)
        self.assertEqual(payload["totals"]["countries"], 3)
        self.assertEqual(payload["static_cell_count"], 6)
        for country in payload["countries"]:
            self.assertEqual(len(country["bbox"]), 4)
            self.assertLess(country["bbox"][0], country["bbox"][1])
            self.assertLess(country["bbox"][2], country["bbox"][3])

    # -- calendar -----------------------------------------------------------

    def test_calendar_defaults_matrix_climatology_trend_badges(self) -> None:
        response = self.client.get(
            "/api/calendar",
            params={"iso_a3": "BRA", "metric": "srehi", "series_mode": "harmonized"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        payload = response.json()
        self.assertEqual(payload["filters"]["iso_a3"], "BRA")
        self.assertEqual(payload["metric"], "srehi")
        self.assertEqual(payload["series_mode"], "harmonized")
        self.assertEqual(payload["selection"]["country_name"], "Brazil")
        self.assertGreaterEqual(len(payload["matrix"]["cells"]), 26 * 12)
        self.assertEqual(payload["matrix"]["years"][0], 2000)
        self.assertEqual(len(payload["climatology"]["months"]), 12)
        self.assertEqual(payload["climatology"]["frozen_period"], "2001-2020")
        self.assertGreaterEqual(len(payload["timeseries"]), 300)
        first = payload["timeseries"][0]
        for key in (
            "year",
            "month",
            "year_month",
            "harmonized",
            "lo90",
            "hi90",
            "raw",
            "z_score",
            "pct_of_median",
            "severity_tier",
            "data_maturity",
            "constellation_regime",
        ):
            self.assertIn(key, first)
        self.assertNotEqual(first["harmonized"], first["raw"])
        self.assertLessEqual(first["lo90"], first["harmonized"])
        self.assertGreaterEqual(first["hi90"], first["harmonized"])
        self.assertTrue(payload["trend"]["available"])
        self.assertEqual(payload["trend"]["trend_direction"], "increasing")
        self.assertEqual(len(payload["badges"]), 3)
        self.assertEqual(
            {badge["id"] for badge in payload["badges"]},
            {"quality_maturity", "sama_static", "relay_uncertainty"},
        )
        for badge in payload["badges"]:
            self.assertIn("detail_en", badge)
            self.assertIn("detail_es", badge)
        self.assertGreater(payload["kpis"]["selected_total"], 0)

    def test_calendar_raw_vs_harmonized_toggle(self) -> None:
        harmonized = self.client.get(
            "/api/calendar", params={"iso_a3": "BRA", "series_mode": "harmonized"}
        ).json()
        raw = self.client.get(
            "/api/calendar", params={"iso_a3": "BRA", "series_mode": "raw"}
        ).json()
        h_last = harmonized["timeseries"][-1]
        r_last = raw["timeseries"][-1]
        self.assertEqual(h_last["raw"], r_last["raw"])
        self.assertEqual(h_last["harmonized"], r_last["harmonized"])
        self.assertNotEqual(h_last["raw"], h_last["harmonized"])
        self.assertEqual(harmonized["series_mode"], "harmonized")
        self.assertEqual(raw["series_mode"], "raw")
        # In raw mode the selected matrix value follows the raw observation.
        raw_matrix = next(
            cell for cell in raw["matrix"]["cells"] if cell["year_month"] == r_last["year_month"]
        )
        self.assertAlmostEqual(raw_matrix["value"], r_last["raw"], places=4)

    def test_calendar_south_america_and_admin1_levels(self) -> None:
        continental = self.client.get(
            "/api/calendar", params={"iso_a3": "SOUTH_AMERICA"}
        ).json()
        self.assertEqual(continental["filters"]["iso_a3"], "SOUTH_AMERICA")
        self.assertEqual(continental["selection"]["level"], "continental")
        self.assertGreater(len(continental["timeseries"]), 300)
        admin1 = self.client.get(
            "/api/calendar",
            params={"iso_a3": "BRA", "admin1_name": "Amazonas"},
        ).json()
        self.assertEqual(admin1["filters"]["admin1_name"], "Amazonas")
        self.assertEqual(admin1["selection"]["level"], "admin1")
        self.assertEqual(len(admin1["climatology"]["months"]), 12)
        # Admin-1 is a partition of the country in the fixture.
        amazonas_total = sum(row["harmonized"] for row in admin1["timeseries"])
        country_total = sum(row["harmonized"] for row in continental["timeseries"])
        self.assertGreater(amazonas_total, 0)
        self.assertLess(amazonas_total, country_total)

    def test_calendar_bbox_and_point_filters(self) -> None:
        bbox = self.client.get(
            "/api/calendar",
            params={"iso_a3": "BRA", "bbox": "-20,5,-75,-45", "metric": "frp_mw"},
        )
        self.assertEqual(bbox.status_code, 200)
        bbox_payload = bbox.json()
        self.assertEqual(bbox_payload["selection"]["level"], "spatial")
        self.assertEqual(bbox_payload["selection"]["source"], "harmonized_cell50km_monthly.parquet")
        self.assertEqual(bbox_payload["filters"]["bbox"], [-20.0, 5.0, -75.0, -45.0])
        self.assertGreater(len(bbox_payload["timeseries"]), 0)
        self.assertEqual(bbox_payload["metric"], "frp_mw")

        point = self.client.get(
            "/api/calendar",
            params={"iso_a3": "BRA", "lat": -14.24, "lon": -54.39, "radius_km": 40},
        )
        self.assertEqual(point.status_code, 200)
        point_payload = point.json()
        self.assertEqual(point_payload["selection"]["resolution"], "10km")
        self.assertEqual(point_payload["selection"]["source"], "cube_cell10km_monthly.parquet")
        self.assertGreater(len(point_payload["timeseries"]), 0)
        self.assertEqual(point_payload["filters"]["lat"], -14.24)
        self.assertEqual(point_payload["filters"]["radius_km"], 40.0)

    # -- early onset --------------------------------------------------------

    def test_early_onset_curve_and_first_crossings(self) -> None:
        response = self.client.get(
            "/api/early-onset",
            params={"iso_a3": "BRA", "admin1_name": "Amazonas", "year": 2024},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreater(len(payload["curve"]), 50)
        row = payload["curve"][0]
        for key in (
            "doy",
            "acq_date",
            "events",
            "scaled_events",
            "srehi_90d",
            "p75_90d",
            "p90_90d",
            "early_onset_flag",
        ):
            self.assertIn(key, row)
        summary = payload["summary"]
        self.assertEqual(summary["year"], 2024)
        self.assertIsNotNone(summary["first_crossing_p90_doy"])
        self.assertIsNotNone(summary["first_crossing_p75_doy"])
        self.assertGreater(summary["critical_count"], 0)
        self.assertGreater(summary["peak_srehi_90d"], 0)

    # -- spatial ------------------------------------------------------------

    def test_spatial_50km_10km_and_static_cells(self) -> None:
        cell50 = self.client.get(
            "/api/spatial",
            params={"year_month": "2025-12", "iso_a3": "BRA", "resolution": "50km"},
        )
        self.assertEqual(cell50.status_code, 200)
        payload = cell50.json()
        self.assertEqual(payload["filters"]["resolution"], "50km")
        self.assertGreater(len(payload["cells"]), 0)
        self.assertTrue(payload["static_cells"])
        first = payload["cells"][0]
        for key in ("cell_id", "lat", "lon", "srehi", "frp_mw", "is_in_sama"):
            self.assertIn(key, first)
        self.assertAlmostEqual(first["lat"], -14.24, delta=5.0)
        self.assertAlmostEqual(first["lon"], -54.39, delta=8.0)

        no_static = self.client.get(
            "/api/spatial",
            params={
                "year_month": "2025-12",
                "iso_a3": "BRA",
                "include_static": "false",
            },
        ).json()
        self.assertEqual(no_static["static_cells"], [])

        cell10 = self.client.get(
            "/api/spatial",
            params={
                "year_month": "2025-12",
                "iso_a3": "BRA",
                "resolution": "10km",
                "limit": 100,
            },
        )
        self.assertEqual(cell10.status_code, 200)
        payload10 = cell10.json()
        self.assertEqual(payload10["filters"]["resolution"], "10km")
        self.assertGreater(len(payload10["cells"]), 0)
        self.assertTrue(all(cell["cell_id"].startswith("BRA_10_") for cell in payload10["cells"]))

        yearly = self.client.get(
            "/api/spatial",
            params={"year": 2024, "iso_a3": "BRA", "resolution": "50km"},
        )
        self.assertEqual(yearly.status_code, 200)
        self.assertGreater(len(yearly.json()["cells"]), 0)

    def test_spatial_reference_scale_and_month_scale(self) -> None:
        response = self.client.get(
            "/api/spatial",
            params={
                "year_month": "2024-06",
                "iso_a3": "BRA",
                "range_start": "2020-01",
                "range_end": "2024-12",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        summary = payload["summary"]
        self.assertEqual(summary["range_start"], "2020-01")
        self.assertEqual(summary["range_end"], "2024-12")
        self.assertIn("month_scale", summary)
        self.assertIn("reference_scale", payload)
        for metric in ("srehi", "frp_mw"):
            month = summary["month_scale"][metric]
            reference = payload["reference_scale"][metric]
            for stats in (month, reference):
                self.assertGreaterEqual(stats["max"], stats["p99"])
                self.assertGreaterEqual(stats["p99"], stats["p95"])
                self.assertGreater(stats["ceiling"], 0)
                self.assertGreaterEqual(stats["ceiling"], stats["p99"])
            self.assertGreater(reference["max"], 0.0)
        # The reference scale is stable for every animation frame in the range.
        other = self.client.get(
            "/api/spatial",
            params={
                "year_month": "2021-03",
                "iso_a3": "BRA",
                "range_start": "2020-01",
                "range_end": "2024-12",
            },
        ).json()
        self.assertEqual(other["reference_scale"], payload["reference_scale"])
        self.assertNotEqual(
            other["summary"]["month_scale"], summary["month_scale"]
        )

    def test_spatial_truncation_metadata_and_pagination(self) -> None:
        first = self.client.get(
            "/api/spatial",
            params={
                "year_month": "2025-12",
                "iso_a3": "SOUTH_AMERICA",
                "limit": 10,
                "offset": 0,
            },
        ).json()
        second = self.client.get(
            "/api/spatial",
            params={
                "year_month": "2025-12",
                "iso_a3": "SOUTH_AMERICA",
                "limit": 10,
                "offset": 10,
            },
        ).json()
        summary = first["summary"]
        self.assertEqual(summary["total_available_cells"], 15)
        self.assertEqual(summary["returned_cells"], 10)
        self.assertEqual(summary["active_cells"], 10)
        self.assertEqual(summary["applied_limit"], 10)
        self.assertEqual(summary["offset"], 0)
        self.assertTrue(summary["truncated"])
        self.assertEqual(second["summary"]["offset"], 10)
        self.assertEqual(second["summary"]["returned_cells"], 5)
        self.assertFalse(second["summary"]["truncated"])
        first_ids = {cell["cell_id"] for cell in first["cells"]}
        second_ids = {cell["cell_id"] for cell in second["cells"]}
        self.assertEqual(len(first_ids), 10)
        self.assertFalse(first_ids & second_ids)
        # The default limit is 10,000, not the legacy 5,000.
        default_payload = self.client.get(
            "/api/spatial",
            params={"year_month": "2025-12", "iso_a3": "SOUTH_AMERICA"},
        ).json()
        self.assertEqual(default_payload["summary"]["applied_limit"], 10000)

    def test_spatial_scientific_states(self) -> None:
        cases = [
            ("2000-11", {}, "PARTIAL_MONTH"),
            ("2026-11", {}, "PARTIAL_MONTH"),
            ("2026-10", {}, "NRT_PRELIMINARY"),
            (
                "2024-06",
                {"lat": -30.0, "lon": -20.0, "radius_km": 20.0},
                "ZERO_DETECTIONS",
            ),
            ("1999-01", {}, "OUT_OF_CATALOG"),
        ]
        for year_month, extra, expected in cases:
            with self.subTest(year_month=year_month, expected=expected):
                params = {
                    "year_month": year_month,
                    "iso_a3": "SOUTH_AMERICA",
                    **extra,
                }
                response = self.client.get("/api/spatial", params=params)
                self.assertEqual(response.status_code, 200)
                summary = response.json()["summary"]
                self.assertEqual(summary["month_state"], expected)
                if expected == "PARTIAL_MONTH":
                    self.assertTrue(summary["is_partial_month"])
                    self.assertEqual(summary["data_maturity"], "PARTIAL_MONTH")
                if expected == "NRT_PRELIMINARY":
                    self.assertEqual(summary["data_maturity"], "NRT_PRELIMINARY")
                    self.assertFalse(summary["is_partial_month"])
                if expected == "ZERO_DETECTIONS":
                    self.assertEqual(summary["total_available_cells"], 0)
                if expected == "OUT_OF_CATALOG":
                    self.assertTrue(summary["out_of_catalog"])
                    self.assertEqual(summary["total_available_cells"], 0)

    def test_spatial_resolution_value_type_and_regime_helpers(self) -> None:
        cell50 = self.client.get(
            "/api/spatial",
            params={"year_month": "2025-12", "iso_a3": "BRA", "resolution": "50km"},
        ).json()
        self.assertEqual(cell50["summary"]["effective_resolution"], "50km")
        self.assertEqual(cell50["summary"]["value_type"], "M3_HARMONIZED")
        self.assertTrue(cell50["cells"])
        self.assertTrue(
            all(cell["value_type"] == "M3_HARMONIZED" for cell in cell50["cells"])
        )
        cell10 = self.client.get(
            "/api/spatial",
            params={"year_month": "2025-12", "iso_a3": "BRA", "resolution": "10km"},
        ).json()
        self.assertEqual(cell10["summary"]["effective_resolution"], "10km")
        self.assertEqual(cell10["summary"]["value_type"], "RAW_OVERPASS_COUNT")
        self.assertTrue(cell10["cells"])
        self.assertTrue(
            all(
                cell["value_type"] == "RAW_OVERPASS_COUNT"
                for cell in cell10["cells"]
            )
        )
        # Regime/sensor helpers follow the documented constellation epochs.
        self.assertEqual(self.app._infer_regime_for_ym("2001-01"), "C0_TERRA_ONLY")
        self.assertEqual(self.app._infer_regime_for_ym("2005-01"), "C1_TERRA_AQUA")
        self.assertEqual(self.app._infer_regime_for_ym("2015-01"), "C2_MODIS_SNPP")
        self.assertEqual(
            self.app._infer_regime_for_ym("2020-01"), "C3_MODIS_SNPP_NOAA20"
        )
        self.assertEqual(
            self.app._infer_regime_for_ym("2025-01"), "C4_MODIS_VIIRS_TRIO"
        )
        self.assertEqual(
            self.app._infer_sensors_for_regime("C0_TERRA_ONLY"), ["MODIS_TERRA"]
        )
        self.assertIn(
            "VIIRS_NOAA21",
            self.app._infer_sensors_for_regime("C4_MODIS_VIIRS_TRIO"),
        )

    # -- CSV ---------------------------------------------------------------

    def test_csv_export_is_clean_attachment_with_metadata(self) -> None:
        response = self.client.get(
            "/api/export.csv",
            params={"iso_a3": "BRA", "metric": "srehi", "series_mode": "harmonized"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/csv; charset=utf-8")
        self.assertEqual(
            response.headers["content-disposition"],
            'attachment; filename="srehi_calendar_export.csv"',
        )
        self.assertEqual(response.headers["cache-control"], "no-store")
        text = response.content.decode("utf-8")
        self.assertFalse(text.startswith("\ufeff"))
        self.assertNotIn("\r\n", text)
        reader = csv.DictReader(io.StringIO(text))
        self.assertIn("generated_at_utc", reader.fieldnames)
        self.assertIn("source", reader.fieldnames)
        self.assertIn("series_mode", reader.fieldnames)
        self.assertIn("bbox", reader.fieldnames)
        rows = list(reader)
        self.assertGreater(len(rows), 300)
        self.assertEqual(rows[0]["iso_a3"], "BRA")
        self.assertEqual(rows[0]["series_mode"], "harmonized")
        self.assertEqual(rows[0]["metric"], "srehi")
        self.assertNotEqual(rows[0]["harmonized"], rows[0]["raw"])

    # -- validation ---------------------------------------------------------

    def test_invalid_parameters_return_400(self) -> None:
        cases = [
            "/api/calendar?metric=wind",
            "/api/calendar?series_mode=magic",
            "/api/calendar?iso_a3=Brazil",
            "/api/calendar?bbox=10,-10,-70,-50",
            "/api/calendar?bbox=-20,5,-45,-75",
            "/api/calendar?bbox=-20,5,-75",
            "/api/calendar?start_ym=2024-13",
            "/api/calendar?start_ym=2024-12&end_ym=2024-01",
            "/api/calendar?lat=-10&lon=-50",
            "/api/calendar?radius_km=-5",
            "/api/calendar?lat=-10&lon=-50&radius_km=40&bbox=-20,5,-75,-45",
            "/api/spatial?year_month=2024-1&iso_a3=BRA",
            "/api/spatial?resolution=25km&year_month=2024-01",
            "/api/spatial?limit=0&year_month=2024-01",
            "/api/spatial?limit=20001&year_month=2024-01",
            "/api/spatial?offset=-1&year_month=2024-01",
            "/api/spatial?year_month=2024-01&range_start=2024-13",
            "/api/spatial?year_month=2024-01&range_end=2024-1",
            "/api/spatial?year_month=2024-01&range_start=2024-12&range_end=2024-01",
            "/api/spatial?include_static=maybe&year_month=2024-01",
            "/api/spatial?iso_a3=BRA",
        ]
        for path in cases:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 400, path)
                self.assertEqual(response.headers["cache-control"], "no-store")
                payload = response.json()
                self.assertEqual(payload["status"], 400)
                self.assertTrue(payload["message"])

    def test_unknown_country_and_admin1_return_404(self) -> None:
        unknown_country = self.client.get("/api/calendar?iso_a3=ZZZ")
        self.assertEqual(unknown_country.status_code, 404)
        self.assertEqual(unknown_country.json()["error"], "unknown_country")

        unknown_admin1 = self.client.get(
            "/api/calendar", params={"iso_a3": "BRA", "admin1_name": "Nowhere"}
        )
        self.assertEqual(unknown_admin1.status_code, 404)
        self.assertEqual(unknown_admin1.json()["error"], "unknown_admin1")

        unknown_onset = self.client.get(
            "/api/early-onset", params={"iso_a3": "ZZZ", "year": 2024}
        )
        self.assertEqual(unknown_onset.status_code, 404)

    def test_missing_asset_returns_404(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory)
            (empty / "processed").mkdir()
            app = create_app(empty / "processed")
            client = httpx.Client(
                transport=httpx.WSGITransport(app=app),
                base_url="http://empty",
            )
            try:
                self.assertEqual(client.get("/api/health").status_code, 200)
                self.assertEqual(client.get("/api/calendar?iso_a3=BRA").status_code, 404)
                self.assertEqual(client.get("/api/early-onset?iso_a3=BRA&year=2024").status_code, 404)
            finally:
                client.close()

    # -- static UI ----------------------------------------------------------

    def test_html_is_english_by_default_and_has_no_nasa_logo(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        html = response.text
        self.assertIn('lang="en"', html)
        self.assertIn('data-theme="light"', html)
        self.assertIn('<meta name="color-scheme" content="light">', html)
        self.assertIn("Fire Relay South America", html)
        self.assertIn("Harmonized Burn Calendar", html)
        self.assertIn('id="lang-en"', html)
        self.assertIn('aria-pressed="true"', html)
        self.assertNotIn("nasa", html.lower())
        self.assertNotIn("<img", html.lower())
        self.assertNotIn("logo", html.lower())
        self.assertIn('role="region"', html)
        self.assertIn('aria-labelledby="anomaly-table-title"', html)
        self.assertIn('class="table-wrapper"', html)
        self.assertIn('tabindex="0"', html)
        # Single light theme: no theme switcher remains.
        self.assertNotIn("theme-select", html)
        self.assertNotIn("theme-control", html)
        # Every labelled form control has an explicit <label for=...>.
        for control in (
            "country-select",
            "admin1-select",
            "metric-select",
            "series-harmonized",
            "series-raw",
            "onset-year-inline",
            "map-scale-mode",
            "filter-geo",
            "filter-bbox",
            "filter-point",
            "bbox-min-lat",
            "bbox-max-lat",
            "bbox-min-lon",
            "bbox-max-lon",
            "point-lat",
            "point-lon",
            "point-radius",
            "start-date-btn",
            "end-date-btn",
        ):
            self.assertIn(f'for="{control}"', html, control)
        # Hidden canonical date inputs and the redesigned picker contract.
        for element_id in (
            "start-ym",
            "end-ym",
            "date-picker-popover",
            "date-range-clear",
            "picker-year-select",
            "picker-month-grid",
            "picker-tab-start",
            "picker-tab-end",
            "date-range-clear",
        ):
            self.assertIn(f'id="{element_id}"', html, element_id)
        self.assertEqual(html.count('data-preset="'), 5)
        self.assertNotIn('type="month"', html)
        self.assertNotIn("placeholder=", html)
        title_free = html.replace("brand-title", "")
        for allowed_title in (
            "Zoom in",
            "Zoom out",
            "Fit selection",
            "Reset continental view",
            "Previous month",
            "Next month",
        ):
            title_free = title_free.replace(f'title="{allowed_title}"', "")
        self.assertNotIn('title="', title_free)
        for control in (
            "map-zoom-in",
            "map-zoom-out",
            "map-zoom-fit",
            "map-zoom-reset",
            "map-zoom-badge",
            "map-cell-inspector",
        ):
            self.assertIn(f'id="{control}"', html, control)
        # Module 05 temporal navigation, scientific-state banners and traceability.
        for control in (
            "map-month-select",
            "map-month-prev",
            "map-month-next",
            "map-play-btn",
            "map-speed-select",
            "map-scale-mode",
            "map-state-banner",
            "map-resolution-notice",
            "map-truncation-banner",
            "map-traceability-strip",
        ):
            self.assertIn(f'id="{control}"', html, control)
        self.assertIn('id="onset-range-warning"', html)
        self.assertIn('id="onset-year-inline"', html)
        # The duplicate season-year selector was removed from module 01.
        self.assertNotIn('id="onset-year"', html)

    def test_static_assets_and_container_styles(self) -> None:
        css = self.client.get("/static/styles.css")
        self.assertEqual(css.status_code, 200)
        self.assertIn("text/css", css.headers["content-type"])
        self.assertIn("color-scheme: light", css.text)
        self.assertNotIn("color-scheme: light dark", css.text)
        self.assertNotIn("@media (prefers-color-scheme: dark)", css.text)
        self.assertNotIn(':root[data-theme="dark"]', css.text)
        self.assertNotIn(".theme-control", css.text)
        self.assertIn(".date-range-field", css.text)
        self.assertIn(".date-picker-popover", css.text)
        self.assertIn(".date-month-btn", css.text)
        self.assertIn(".is-in-range", css.text)
        self.assertIn(".map-temporal-toolbar", css.text)
        self.assertIn(".map-state-banner", css.text)
        self.assertIn(".map-truncation-banner", css.text)
        self.assertIn(".map-traceability-strip", css.text)
        self.assertIn(".trace-chip", css.text)
        self.assertIn(".onset-range-warning", css.text)
        self.assertIn(".ts-point", css.text)
        self.assertIn("@container (width < 600px)", css.text)
        self.assertIn("position: sticky", css.text)
        self.assertIn("inset-block-start: 0", css.text)
        self.assertIn("inset-inline-start: 0", css.text)

        container_block = css.text.split("@container (width < 600px)", 1)[1]
        self.assertRegex(container_block, r"thead\s*\{\s*display:\s*none")
        self.assertRegex(
            container_block,
            r"tbody\s+tr\s*\{[^}]*display:\s*grid",
            re.S,
        )
        self.assertRegex(
            container_block,
            r"tbody\s+th\s*\{[^}]*position:\s*static",
            re.S,
        )
        self.assertIn("attr(data-label)", container_block)

        js = self.client.get("/static/app.js")
        self.assertEqual(js.status_code, 200)
        self.assertIn("javascript", js.headers["content-type"])
        for token in ("aria-pressed", "SVG_NS", "getContext(\"2d\")", "api/calendar", "api/export.csv"):
            self.assertIn(token, js.text)
        self.assertNotIn("prefers-color-scheme", js.text)
        self.assertNotIn("HEAT_STOPS_DARK", js.text)
        self.assertNotIn("isDarkTheme", js.text)
        for token in (
            "start-date-btn",
            "end-date-btn",
            "date-picker-popover",
            "picker-year-select",
            "date-preset",
            "start_ym",
            "end_ym",
            "map-month-select",
            "map-play-btn",
            "map-scale-mode",
            "loadSpatialMonth",
            "updateOnsetRangeWarning",
            "renderMapBannersAndTraceability",
            "highlightTimeseriesActiveMonth",
        ):
            self.assertIn(token, js.text)
        self.assertIn('button.dataset.lang === "en"', js.text)
        self.assertIn("dataset.label", js.text)

    def test_path_traversal_is_blocked(self) -> None:
        attempts = [
            "/static/../app/server.py",
            "/static/%2e%2e/app/server.py",
            "/static/%2e%2e%2fapp%2fserver.py",
            "/static/..%2fapp%2fserver.py",
            "/static/%2fetc%2fpasswd",
            "/static//etc/passwd",
            "/static/does-not-exist.css",
            "/../app/server.py",
        ]
        for path in attempts:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertIn(response.status_code, (400, 404), path)
                if path.startswith("/static/"):
                    self.assertIn("application/json", response.headers["content-type"])

    def test_head_and_method_handling(self) -> None:
        head = self.client.head("/api/health")
        self.assertEqual(head.status_code, 200)
        self.assertEqual(head.content, b"")
        post = self.client.post("/api/health")
        self.assertEqual(post.status_code, 405)
        self.assertEqual(post.headers.get("allow"), "GET, HEAD")

    # -- real socket --------------------------------------------------------

    def test_threading_http_server_on_real_socket(self) -> None:
        server = create_server(self.app, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        base = f"http://{host}:{port}"
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Cache-Control"], "no-store")
                health = json.loads(response.read().decode("utf-8"))
                self.assertEqual(health["status"], "ok")
            with urllib.request.urlopen(
                f"{base}/api/calendar?iso_a3=BRA&metric=srehi", timeout=5
            ) as response:
                calendar = json.loads(response.read().decode("utf-8"))
                self.assertGreater(len(calendar["timeseries"]), 0)
                self.assertEqual(calendar["filters"]["iso_a3"], "BRA")
            with urllib.request.urlopen(f"{base}/", timeout=5) as response:
                html = response.read().decode("utf-8")
                self.assertIn('lang="en"', html)
            with urllib.request.urlopen(f"{base}/api/catalog", timeout=5) as response:
                catalog = json.loads(response.read().decode("utf-8"))
                self.assertEqual(len(catalog["countries"]), 3)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    # -- latency ------------------------------------------------------------

    def test_average_query_latency_below_budget(self) -> None:
        paths = [
            "/api/calendar?iso_a3=BRA&metric=srehi",
            "/api/calendar?iso_a3=ARG&metric=frp_mw",
            "/api/spatial?year_month=2025-12&iso_a3=COL&resolution=50km",
            "/api/calendar?iso_a3=SOUTH_AMERICA",
        ]
        # Warm every path first; the first call on a thread necessarily opens
        # the thread-local DuckDB connection.
        for path in paths:
            self.assertEqual(self.client.get(path).status_code, 200)
        samples = []
        for index in range(40):
            path = paths[index % len(paths)]
            started = time.perf_counter()
            response = self.client.get(path)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.assertEqual(response.status_code, 200)
            samples.append(elapsed_ms)
        average = sum(samples) / len(samples)
        self.assertLess(
            average,
            50.0,
            f"average query latency {average:.2f} ms exceeds the 50 ms budget: {samples}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
