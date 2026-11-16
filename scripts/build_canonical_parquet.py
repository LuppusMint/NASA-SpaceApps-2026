#!/usr/bin/env python3
"""Build canonical out-of-core ZSTD Parquet for South America NASA FIRMS (2000-present).

Reproducibly unifies country archives, user-downloaded FIRMS ZIPs, and API SP/NRT fills
via DuckDB (6 GB RAM limit, NVMe spill) and Polars, normalizes MODIS/VIIRS schemas,
deduplicates overlapping feeds (Standard Science Product > NRT), clips to exact South
America land boundaries (including French Guiana), attaches equal-area 10 km grid IDs,
and writes audited summary metrics to data/DATA_SUMMARY.json.
"""

from __future__ import annotations

import datetime as dt
import gc
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.prepared import prep

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
COUNTRY_DIR = DATA_DIR / "raw" / "firms_country"
API_DIR = DATA_DIR / "raw" / "firms_api_fill"
BOUNDARY_DIR = DATA_DIR / "boundaries"
OUT_PARQUET_DIR = DATA_DIR / "parquet" / "south_america_hotspots"
SUMMARY_PATH = DATA_DIR / "DATA_SUMMARY.json"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.core.duckdb_helper import connect as duckdb_connect  # noqa: E402
from src.core.duckdb_helper import settings as duckdb_settings  # noqa: E402
from src.core.geogrid import default_equal_area_crs  # noqa: E402


SA_ADMIN0_MAP = {
    "Argentina": ("Argentina", "ARG"),
    "Bolivia": ("Bolivia", "BOL"),
    "Brazil": ("Brazil", "BRA"),
    "Brazilian Island": ("Brazil", "BRA"),
    "Chile": ("Chile", "CHL"),
    "Colombia": ("Colombia", "COL"),
    "Ecuador": ("Ecuador", "ECU"),
    "Falkland Islands": ("Falkland Islands", "FLK"),
    "Guyana": ("Guyana", "GUY"),
    "Paraguay": ("Paraguay", "PRY"),
    "Peru": ("Peru", "PER"),
    "Southern Patagonian Ice Field": ("Argentina/Chile", "ARG"),
    "Suriname": ("Suriname", "SUR"),
    "Uruguay": ("Uruguay", "URY"),
    "Venezuela": ("Venezuela", "VEN"),
}


def _load_south_america_boundaries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load exact South America admin-0 and admin-1 polygons (including French Guiana)."""
    adm0_zip = BOUNDARY_DIR / "ne_10m_admin_0_countries.zip"
    adm1_zip = BOUNDARY_DIR / "ne_10m_admin_1_states_provinces.zip"

    g0 = gpd.read_file(f"zip://{adm0_zip.resolve()}")
    g0_sa = g0[g0["CONTINENT"] == "South America"][["ADMIN", "ISO_A3", "geometry"]].copy()
    g0_sa["country_name"] = g0_sa["ADMIN"].map(lambda x: SA_ADMIN0_MAP.get(x, (x, "-99"))[0])
    g0_sa["iso_a3"] = [
        iso if (iso and iso != "-99") else SA_ADMIN0_MAP.get(adm, (adm, "-99"))[1]
        for adm, iso in zip(g0_sa["ADMIN"], g0_sa["ISO_A3"])
    ]
    g0_sa["admin1_name"] = g0_sa["country_name"]

    g1 = gpd.read_file(f"zip://{adm1_zip.resolve()}")
    sa_isos = {v[1] for v in SA_ADMIN0_MAP.values()}
    g1_sa = g1[(g1["adm0_a3"].isin(sa_isos)) | (g1["iso_3166_2"] == "FR-GF")].copy()
    g1_sa["country_name"] = np.where(
        g1_sa["iso_3166_2"] == "FR-GF",
        "French Guiana",
        g1_sa["admin"].replace(
            {
                "Argentine": "Argentina",
            }
        ),
    )
    g1_sa["iso_a3"] = np.where(g1_sa["iso_3166_2"] == "FR-GF", "GUF", g1_sa["adm0_a3"])
    g1_sa["admin1_name"] = g1_sa["name"].fillna(g1_sa["country_name"])

    fr_gf_adm0 = g1_sa[g1_sa["iso_3166_2"] == "FR-GF"][["country_name", "iso_a3", "admin1_name", "geometry"]].copy()

    adm0_all = pd.concat(
        [
            g0_sa[["country_name", "iso_a3", "admin1_name", "geometry"]],
            fr_gf_adm0,
        ],
        ignore_index=True,
    )
    adm0_gdf = gpd.GeoDataFrame(adm0_all, geometry="geometry", crs="EPSG:4326").reset_index(drop=True)
    adm1_gdf = gpd.GeoDataFrame(
        g1_sa[["country_name", "iso_a3", "admin1_name", "geometry"]],
        geometry="geometry",
        crs="EPSG:4326",
    ).reset_index(drop=True)
    return adm0_gdf, adm1_gdf


def _extract_user_zips(tmp_zip_dir: Path) -> list[str]:
    """Stream-extract CSV members from user FIRMS ZIP archives into NVMe scratch."""
    extracted: list[str] = []
    for zp in sorted(BASE_DIR.glob("DL_FIRE_*.zip")):
        with zipfile.ZipFile(zp, "r") as zf:
            for member in zf.namelist():
                if not member.lower().endswith(".csv"):
                    continue
                out_path = tmp_zip_dir / f"{zp.stem}__{Path(member).name}"
                with zf.open(member, "r") as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                extracted.append(out_path.as_posix())
    return extracted


def _feed_projection_sql(
    glob_or_files: str | list[str],
    source_feed: str,
    source_priority: int,
    *,
    has_bright_ti: bool,
    has_brightness: bool,
    has_type: bool,
) -> str:
    """Build DuckDB SQL SELECT projecting heterogeneous FIRMS CSVs into canonical schema."""
    if isinstance(glob_or_files, list):
        files_sql = "[" + ", ".join(f"'{p}'" for p in glob_or_files) + "]"
    else:
        files_sql = f"'{glob_or_files}'"

    b1_expr = (
        "COALESCE(TRY_CAST(bright_ti4 AS DOUBLE), TRY_CAST(brightness AS DOUBLE))"
        if (has_bright_ti and has_brightness)
        else ("TRY_CAST(bright_ti4 AS DOUBLE)" if has_bright_ti else "TRY_CAST(brightness AS DOUBLE)")
    )
    b2_expr = (
        "COALESCE(TRY_CAST(bright_ti5 AS DOUBLE), TRY_CAST(bright_t31 AS DOUBLE))"
        if (has_bright_ti and has_brightness)
        else ("TRY_CAST(bright_ti5 AS DOUBLE)" if has_bright_ti else "TRY_CAST(bright_t31 AS DOUBLE)")
    )
    type_expr = "TRY_CAST(type AS SMALLINT)" if has_type else "CAST(NULL AS SMALLINT)"

    return f"""
    SELECT
        TRY_CAST(latitude AS DOUBLE) AS latitude,
        TRY_CAST(longitude AS DOUBLE) AS longitude,
        {b1_expr} AS brightness_primary,
        {b2_expr} AS brightness_secondary,
        TRY_CAST(scan AS DOUBLE) AS scan,
        TRY_CAST(track AS DOUBLE) AS track,
        TRY_CAST(acq_date AS DATE) AS acq_date,
        SUBSTR(LPAD(REGEXP_REPLACE(TRIM(COALESCE(acq_time, '0000')), '[^0-9]', '', 'g'), 4, '0'), 1, 4) AS acq_time,
        CASE
            WHEN TRY_CAST(acq_date AS DATE) IS NOT NULL
             AND TRY_CAST(SUBSTR(LPAD(REGEXP_REPLACE(TRIM(COALESCE(acq_time, '0000')), '[^0-9]', '', 'g'), 4, '0'), 1, 2) AS INTEGER) BETWEEN 0 AND 23
             AND TRY_CAST(SUBSTR(LPAD(REGEXP_REPLACE(TRIM(COALESCE(acq_time, '0000')), '[^0-9]', '', 'g'), 4, '0'), 3, 2) AS INTEGER) BETWEEN 0 AND 59
            THEN TRY_CAST(acq_date AS DATE)
               + CAST(
                   TRY_CAST(SUBSTR(LPAD(REGEXP_REPLACE(TRIM(COALESCE(acq_time, '0000')), '[^0-9]', '', 'g'), 4, '0'), 1, 2) AS INTEGER) * 60
                   + TRY_CAST(SUBSTR(LPAD(REGEXP_REPLACE(TRIM(COALESCE(acq_time, '0000')), '[^0-9]', '', 'g'), 4, '0'), 3, 2) AS INTEGER)
                 AS INTEGER) * INTERVAL 1 MINUTE
            ELSE NULL
        END AS acq_datetime_utc,
        CASE
            WHEN UPPER(TRIM(satellite)) IN ('T', 'TERRA') THEN 'Terra'
            WHEN UPPER(TRIM(satellite)) IN ('A', 'AQUA') THEN 'Aqua'
            WHEN UPPER(TRIM(satellite)) IN ('N', 'SNPP', 'S-NPP', 'NPP', 'SUOMI-NPP') THEN 'SNPP'
            WHEN UPPER(TRIM(satellite)) IN ('N20', 'NOAA-20', 'NOAA20', 'J1', '1') THEN 'N20'
            WHEN UPPER(TRIM(satellite)) IN ('N21', 'NOAA-21', 'NOAA21', 'J2', '2') THEN 'N21'
            ELSE TRIM(satellite)
        END AS satellite,
        CASE
            WHEN UPPER(TRIM(instrument)) IN ('SNPP', 'S-NPP', 'SUOMI-NPP', 'N', 'N20', 'N21', 'J1', 'J2', 'NOAA-20', 'NOAA-21', 'VIIRS') THEN 'VIIRS'
            WHEN UPPER(TRIM(instrument)) IN ('MODIS', 'TERRA', 'AQUA') THEN 'MODIS'
            ELSE UPPER(TRIM(instrument))
        END AS instrument,
        LOWER(TRIM(CAST(confidence AS VARCHAR))) AS confidence_raw,
        TRY_CAST(confidence AS DOUBLE) AS confidence_pct,
        CASE
            WHEN LOWER(TRIM(confidence)) IN ('h', 'high') THEN 'high'
            WHEN LOWER(TRIM(confidence)) IN ('n', 'nominal') THEN 'nominal'
            WHEN LOWER(TRIM(confidence)) IN ('l', 'low') THEN 'low'
            WHEN TRY_CAST(confidence AS DOUBLE) >= 80 THEN 'high'
            WHEN TRY_CAST(confidence AS DOUBLE) >= 30 THEN 'nominal'
            WHEN TRY_CAST(confidence AS DOUBLE) IS NOT NULL THEN 'low'
            ELSE 'unknown'
        END AS confidence_tier,
        TRIM(CAST(version AS VARCHAR)) AS version,
        TRY_CAST(frp AS DOUBLE) AS frp,
        UPPER(TRIM(daynight)) AS daynight,
        {type_expr} AS type,
        '{source_feed}' AS source_feed,
        CAST({source_priority} AS TINYINT) AS source_priority
    FROM read_csv_auto({files_sql}, union_by_name=true, all_varchar=true, header=true)
    WHERE TRY_CAST(latitude AS DOUBLE) IS NOT NULL
      AND TRY_CAST(longitude AS DOUBLE) IS NOT NULL
      AND TRY_CAST(acq_date AS DATE) IS NOT NULL
    """


def main() -> int:
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    scratch_base = Path(os.environ.get("TMPDIR") or (Path.home() / "data-scratch" / "tmp")).expanduser()
    scratch_base.mkdir(parents=True, exist_ok=True)
    work_scratch = Path(tempfile.mkdtemp(prefix="sa_firms_build_", dir=str(scratch_base)))

    try:
        print("[1/5] Extracting user FIRMS ZIPs to NVMe scratch...")
        (work_scratch / "zips").mkdir(parents=True, exist_ok=True)
        zip_csvs = _extract_user_zips(work_scratch / "zips")
        print(f"  Extracted {len(zip_csvs)} archive CSVs")

        country_modis_files = sorted(p.as_posix() for p in COUNTRY_DIR.glob("modis/*.csv") if p.stat().st_size > 150)
        country_viirs_files = sorted(p.as_posix() for p in COUNTRY_DIR.glob("viirs-*/*.csv") if p.stat().st_size > 150)
        sp_api_files = sorted(p.as_posix() for p in API_DIR.glob("*_SP/*.csv") if p.stat().st_size > 150)
        nrt_api_files = sorted(p.as_posix() for p in API_DIR.glob("*_NRT/*.csv") if p.stat().st_size > 150)

        con = duckdb_connect(work_scratch / "build.duckdb")
        print("  DuckDB settings:", duckdb_settings(con))

        parts: list[str] = []
        if country_modis_files:
            parts.append(
                _feed_projection_sql(
                    country_modis_files,
                    "sp_country",
                    1,
                    has_bright_ti=False,
                    has_brightness=True,
                    has_type=True,
                )
            )
        if country_viirs_files:
            parts.append(
                _feed_projection_sql(
                    country_viirs_files,
                    "sp_country",
                    1,
                    has_bright_ti=True,
                    has_brightness=False,
                    has_type=True,
                )
            )
        if zip_csvs:
            parts.append(
                _feed_projection_sql(
                    zip_csvs,
                    "sp_zip",
                    2,
                    has_bright_ti=False,
                    has_brightness=True,
                    has_type=True,
                )
            )
        if sp_api_files:
            parts.append(
                _feed_projection_sql(
                    sp_api_files,
                    "sp_api",
                    3,
                    has_bright_ti=True,
                    has_brightness=True,
                    has_type=True,
                )
            )
        if nrt_api_files:
            parts.append(
                _feed_projection_sql(
                    nrt_api_files,
                    "nrt_api",
                    4,
                    has_bright_ti=True,
                    has_brightness=True,
                    has_type=False,
                )
            )
        if not parts:
            raise RuntimeError("No readable FIRMS input CSV files found.")

        union_sql = "\nUNION ALL\n".join(parts)
        stage_dedup = work_scratch / "stage_dedup.parquet"
        print("[2/5] Unifying & deduplicating all feeds -> stage_dedup.parquet (out-of-core)...")
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM ({union_sql}) AS raw_all
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY instrument, satellite, acq_date, acq_time, ROUND(latitude, 4), ROUND(longitude, 4)
                    ORDER BY
                        source_priority ASC,
                        frp DESC NULLS LAST,
                        confidence_pct DESC NULLS LAST,
                        brightness_primary DESC NULLS LAST,
                        source_feed ASC
                ) = 1
            ) TO '{stage_dedup.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        raw_dedup_rows = int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{stage_dedup.as_posix()}')").fetchone()[0])
        print(f"  Deduplicated stage rows: {raw_dedup_rows:,}")
        con.close()
        del con
        gc.collect()

        print("[3/5] Computing exact 2-level South America boundary lookup (admin-0 + admin-1 + French Guiana)...")
        adm0_gdf, adm1_gdf = _load_south_america_boundaries()
        sa_union_prep = prep(unary_union(adm0_gdf.geometry.tolist()))

        con = duckdb_connect(work_scratch / "build.duckdb")
        box_step = 0.05
        boxes = con.execute(
            f"""
            SELECT DISTINCT
                CAST(FLOOR(latitude / {box_step}) AS INTEGER) AS by_idx,
                CAST(FLOOR(longitude / {box_step}) AS INTEGER) AS bx_idx
            FROM read_parquet('{stage_dedup.as_posix()}')
            """
        ).fetchall()

        adm1_geoms = adm1_gdf.geometry.tolist()
        adm1_meta = list(zip(adm1_gdf["country_name"].tolist(), adm1_gdf["iso_a3"].tolist(), adm1_gdf["admin1_name"].tolist()))
        adm1_sindex = adm1_gdf.sindex

        pure_rows: list[tuple[int, int, str, str, str]] = []
        mixed_boxes: list[tuple[int, int]] = []
        for by_idx, bx_idx in boxes:
            lat_lo = by_idx * box_step
            lon_lo = bx_idx * box_step
            cell = box(lon_lo, lat_lo, lon_lo + box_step, lat_lo + box_step)
            if sa_union_prep.disjoint(cell):
                continue
            cand_idx = list(adm1_sindex.intersection(cell.bounds))
            covered_by = None
            for idx in cand_idx:
                if adm1_geoms[idx].covers(cell):
                    covered_by = adm1_meta[idx]
                    break
            if covered_by is not None:
                pure_rows.append((int(by_idx), int(bx_idx), covered_by[0], covered_by[1], covered_by[2]))
            else:
                mixed_boxes.append((int(by_idx), int(bx_idx)))

        print(f"  0.05-deg boxes: {len(boxes):,} total | {len(pure_rows):,} pure land | {len(mixed_boxes):,} coastal/border mixed")

        con.execute("CREATE OR REPLACE TEMP TABLE pure_boxes (by_idx INTEGER, bx_idx INTEGER, country_name VARCHAR, iso_a3 VARCHAR, admin1_name VARCHAR)")
        if pure_rows:
            con.register("_pure_df", pa.Table.from_pylist([
                {"by_idx": r[0], "bx_idx": r[1], "country_name": r[2], "iso_a3": r[3], "admin1_name": r[4]} for r in pure_rows
            ]))
            con.execute("INSERT INTO pure_boxes SELECT * FROM _pure_df")

        con.execute("CREATE OR REPLACE TEMP TABLE mixed_box_keys (by_idx INTEGER, bx_idx INTEGER)")
        if mixed_boxes:
            con.register("_mixed_df", pa.Table.from_pylist([{"by_idx": b[0], "bx_idx": b[1]} for b in mixed_boxes]))
            con.execute("INSERT INTO mixed_box_keys SELECT * FROM _mixed_df")

        border_lookup_pq = work_scratch / "border_lookup.parquet"
        border_schema = pa.schema(
            [
                pa.field("lat5", pa.float64()),
                pa.field("lon5", pa.float64()),
                pa.field("country_name", pa.string()),
                pa.field("iso_a3", pa.string()),
                pa.field("admin1_name", pa.string()),
            ]
        )
        n_mixed_coords = 0
        n_inside_border = 0
        mixed_reader = con.sql(
            f"""
            SELECT DISTINCT
                ROUND(s.latitude, 5) AS lat5,
                ROUND(s.longitude, 5) AS lon5
            FROM read_parquet('{stage_dedup.as_posix()}') s
            INNER JOIN mixed_box_keys m
              ON CAST(FLOOR(s.latitude / {box_step}) AS INTEGER) = m.by_idx
             AND CAST(FLOOR(s.longitude / {box_step}) AS INTEGER) = m.bx_idx
            """
        ).to_arrow_reader(batch_size=250_000)
        with pq.ParquetWriter(border_lookup_pq, border_schema, compression="zstd") as bw:
            for batch in mixed_reader:
                if batch.num_rows == 0:
                    continue
                n_mixed_coords += batch.num_rows
                pts_pdf = pa.Table.from_batches([batch]).to_pandas()
                pts_gdf = gpd.GeoDataFrame(
                    pts_pdf,
                    geometry=gpd.points_from_xy(pts_pdf["lon5"], pts_pdf["lat5"]),
                    crs="EPSG:4326",
                )
                joined1 = gpd.sjoin(pts_gdf, adm1_gdf, how="left", predicate="intersects")
                joined1 = joined1[~joined1.index.duplicated(keep="first")].copy()
                missing_mask = joined1["country_name"].isna()
                if missing_mask.any():
                    pts_miss = pts_gdf.loc[missing_mask[missing_mask].index]
                    joined0 = gpd.sjoin(pts_miss, adm0_gdf, how="left", predicate="intersects")
                    joined0 = joined0[~joined0.index.duplicated(keep="first")]
                    for col in ("country_name", "iso_a3", "admin1_name"):
                        joined1.loc[joined0.index, col] = joined0[col]
                inside_df = joined1[joined1["country_name"].notna()][["lat5", "lon5", "country_name", "iso_a3", "admin1_name"]]
                if not inside_df.empty:
                    n_inside_border += len(inside_df)
                    bw.write_table(pa.Table.from_pandas(inside_df, schema=border_schema, preserve_index=False))
        print(f"  Distinct border/coast coordinates tested: {n_mixed_coords:,} (inside SA land: {n_inside_border:,})")

        del adm0_gdf, adm1_gdf, sa_union_prep, adm1_geoms, adm1_meta, adm1_sindex, pure_rows, mixed_boxes, boxes
        gc.collect()

        print("[4/5] Attaching Albers Equal-Area 10 km grid & writing Hive-partitioned ZSTD Parquet...")
        sa_aea_crs = default_equal_area_crs(-59.5, 13.7, -82.2, -27.4)
        transformer = Transformer.from_crs("EPSG:4326", sa_aea_crs, always_xy=True)

        clipped_stage_pq = work_scratch / "clipped_stage.parquet"
        con.execute(
            f"""
            COPY (
                SELECT
                    s.*,
                    EXTRACT(YEAR FROM s.acq_date)::SMALLINT AS year,
                    COALESCE(pb.country_name, bl.country_name) AS country_name,
                    COALESCE(pb.iso_a3, bl.iso_a3) AS iso_a3,
                    COALESCE(pb.admin1_name, bl.admin1_name) AS admin1_name
                FROM read_parquet('{stage_dedup.as_posix()}') s
                LEFT JOIN pure_boxes pb
                  ON CAST(FLOOR(s.latitude / {box_step}) AS INTEGER) = pb.by_idx
                 AND CAST(FLOOR(s.longitude / {box_step}) AS INTEGER) = pb.bx_idx
                LEFT JOIN read_parquet('{border_lookup_pq.as_posix()}') bl
                  ON pb.country_name IS NULL
                 AND ROUND(s.latitude, 5) = bl.lat5
                 AND ROUND(s.longitude, 5) = bl.lon5
                WHERE COALESCE(pb.country_name, bl.country_name) IS NOT NULL
            ) TO '{clipped_stage_pq.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        clipped_rows = int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{clipped_stage_pq.as_posix()}')").fetchone()[0])
        dropped_outside = raw_dedup_rows - clipped_rows
        print(f"  South America land rows: {clipped_rows:,} (dropped outside/ocean: {dropped_outside:,})")

        if OUT_PARQUET_DIR.exists():
            shutil.rmtree(OUT_PARQUET_DIR)
        OUT_PARQUET_DIR.mkdir(parents=True, exist_ok=True)

        # Stream distinct years cleanly through pyproj vectorized C transform -> ZSTD Hive partitions
        years = [
            int(r[0])
            for r in con.execute(f"SELECT DISTINCT year FROM read_parquet('{clipped_stage_pq.as_posix()}') ORDER BY year").fetchall()
        ]
        cell_size_m = 10_000.0
        for yr in years:
            yr_dir = OUT_PARQUET_DIR / f"year={yr}"
            yr_dir.mkdir(parents=True, exist_ok=True)
            reader = con.sql(
                f"""
                SELECT
                    latitude, longitude, brightness_primary, brightness_secondary,
                    scan, track, acq_date, acq_time, acq_datetime_utc,
                    satellite, instrument, confidence_raw, confidence_pct, confidence_tier,
                    version, frp, daynight, type, source_feed, source_priority,
                    country_name, iso_a3, admin1_name
                FROM read_parquet('{clipped_stage_pq.as_posix()}')
                WHERE year = {yr}
                ORDER BY acq_date, acq_time, latitude, longitude
                """
            ).to_arrow_reader(batch_size=500_000)
            out_schema = pa.schema(
                list(reader.schema)
                + [
                    pa.field("x_m", pa.float64()),
                    pa.field("y_m", pa.float64()),
                    pa.field("cell_ix_10km", pa.int32()),
                    pa.field("cell_iy_10km", pa.int32()),
                    pa.field("cell_id_10km", pa.string()),
                ]
            )
            with pq.ParquetWriter(
                yr_dir / "data_0.parquet",
                out_schema,
                compression="zstd",
                use_dictionary=True,
            ) as writer:
                for batch in reader:
                    if batch.num_rows == 0:
                        continue
                    lon_np = batch.column("longitude").to_numpy(zero_copy_only=False)
                    lat_np = batch.column("latitude").to_numpy(zero_copy_only=False)
                    x_m, y_m = transformer.transform(lon_np, lat_np)
                    cell_ix = np.floor(x_m / cell_size_m).astype(np.int32)
                    cell_iy = np.floor(y_m / cell_size_m).astype(np.int32)
                    cell_id = pa.array([f"{ix}_{iy}" for ix, iy in zip(cell_ix.tolist(), cell_iy.tolist())], type=pa.string())
                    chunk_tbl = pa.Table.from_arrays(
                        list(batch.columns)
                        + [
                            pa.array(x_m, type=pa.float64()),
                            pa.array(y_m, type=pa.float64()),
                            pa.array(cell_ix, type=pa.int32()),
                            pa.array(cell_iy, type=pa.int32()),
                            cell_id,
                        ],
                        schema=out_schema,
                    )
                    writer.write_table(chunk_tbl, row_group_size=250_000)

        con.close()

        print("[5/5] Auditing final Hive-partitioned Parquet with DuckDB & Polars lazy scan...")
        audit_con = duckdb_connect()
        parquet_glob = (OUT_PARQUET_DIR / "*" / "*.parquet").as_posix()

        min_date, max_date, min_lat, max_lat, min_lon, max_lon, total_final = audit_con.execute(
            f"""
            SELECT
                CAST(MIN(acq_date) AS VARCHAR),
                CAST(MAX(acq_date) AS VARCHAR),
                MIN(latitude), MAX(latitude),
                MIN(longitude), MAX(longitude),
                COUNT(*)
            FROM read_parquet('{parquet_glob}', hive_partitioning=true)
            """
        ).fetchone()

        by_inst_yr = audit_con.execute(
            f"""
            SELECT year, instrument, satellite, COUNT(*) AS n_rows, ROUND(COALESCE(SUM(frp), 0), 2) AS sum_frp_mw
            FROM read_parquet('{parquet_glob}', hive_partitioning=true)
            GROUP BY year, instrument, satellite
            ORDER BY year, instrument, satellite
            """
        ).fetchall()

        by_country = audit_con.execute(
            f"""
            SELECT country_name, COUNT(*) AS n_rows
            FROM read_parquet('{parquet_glob}', hive_partitioning=true)
            GROUP BY country_name
            ORDER BY n_rows DESC
            """
        ).fetchall()

        by_source_feed = audit_con.execute(
            f"""
            SELECT source_feed, COUNT(*) AS n_rows
            FROM read_parquet('{parquet_glob}', hive_partitioning=true)
            GROUP BY source_feed
            ORDER BY n_rows DESC
            """
        ).fetchall()

        # Polars lazy verification
        pl_totals = (
            pl.scan_parquet(parquet_glob, hive_partitioning=True)
            .group_by("instrument")
            .agg(
                pl.len().alias("n_rows"),
                pl.col("frp").sum().alias("sum_frp"),
            )
            .sort("instrument")
            .collect()
        ).to_dicts()
        audit_con.close()

        parquet_bytes = sum(p.stat().st_size for p in OUT_PARQUET_DIR.rglob("*.parquet"))
        summary = {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "started_at_utc": started_utc,
            "equal_area_crs": sa_aea_crs,
            "grid_cell_size_m": cell_size_m,
            "grid_cell_area_km2": (cell_size_m / 1000.0) ** 2,
            "parquet_root": str(OUT_PARQUET_DIR.relative_to(BASE_DIR)),
            "parquet_total_bytes": parquet_bytes,
            "parquet_files": len(list(OUT_PARQUET_DIR.rglob("*.parquet"))),
            "raw_dedup_rows_before_sa_clip": raw_dedup_rows,
            "outside_sa_dropped_rows": dropped_outside,
            "final_sa_land_rows": int(total_final),
            "date_range": [min_date, max_date],
            "lat_range": [round(float(min_lat), 6), round(float(max_lat), 6)],
            "lon_range": [round(float(min_lon), 6), round(float(max_lon), 6)],
            "polars_by_instrument": pl_totals,
            "by_source_feed": [{"source_feed": r[0], "rows": int(r[1])} for r in by_source_feed],
            "by_country": [{"country_name": r[0], "rows": int(r[1])} for r in by_country],
            "by_year_instrument_satellite": [
                {"year": int(r[0]), "instrument": r[1], "satellite": r[2], "rows": int(r[3]), "sum_frp_mw": float(r[4])}
                for r in by_inst_yr
            ],
        }
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"DONE -> {SUMMARY_PATH} ({int(total_final):,} rows, {parquet_bytes / 1e9:.2f} GB ZSTD Parquet)")
        return 0
    finally:
        shutil.rmtree(work_scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
