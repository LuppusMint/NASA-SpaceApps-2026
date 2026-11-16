"""Shared out-of-core helpers for the M0-M3 harmonization pipeline.

The package centralises DuckDB/Parquet utilities and the regime constants used
by ``freeze_validation``, ``harmonize_pipeline`` and ``calendar_analytics``.
All scientific helpers now live under :mod:`src.core`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.core.duckdb_helper import configure as _configure_duckdb

#: Pre-drift window used to calibrate the MODIS -> VIIRS bridge.
PRE_DRIFT_START = "2012-02"
PRE_DRIFT_END = "2020-12"

#: MODIS/VIIRS overlap regime labels.
REGIME_REAL = "TERRA_AQUA"
REGIME_TERRA_ONLY = "TERRA_ONLY"

#: Constellation relay labels.
RELAY_REGIMES = (
    "MODIS_TERRA_ONLY",
    "MODIS_TERRA_AQUA",
    "MODIS_VIIRS_OVERLAP",
    "VIIRS_BACKBONE",
)


def sha256_file(path: Path | str) -> str:
    """SHA-256 hexadecimal de un archivo, leído por bloques."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    """JSON canónico (claves ordenadas, sin espacios) usado para huellas."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    """SHA-256 del JSON canónico de ``value``."""
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def write_json(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    """Escribe JSON UTF-8 de forma determinista."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return destination


def connect_duckdb(
    *,
    scratch_dir: Path | str | None = None,
    memory_limit: str | None = None,
    threads: int | None = None,
) -> "duckdb.DuckDBPyConnection":
    """DuckDB connection with bounded memory, threads and workspace spill.

    ``FIRE_RELAY_DUCKDB_MEM`` and ``FIRE_RELAY_THREADS`` are honoured by
    :mod:`src.core.duckdb_helper`; the default spill directory is the supplied
    ``scratch_dir`` (typically ``processed_dir/.duckdb_tmp``).
    """
    import duckdb

    connection = duckdb.connect()
    return _configure_duckdb(
        connection,
        memory_limit=memory_limit,
        threads=threads,
        temp_dir=scratch_dir,
    )


def parquet_sql(path: Path | str) -> str:
    """Expresión DuckDB ``read_parquet`` con la ruta escapada."""
    safe = str(Path(path).resolve()).replace("'", "''")
    return f"read_parquet('{safe}')"


def write_parquet(con: "duckdb.DuckDBPyConnection", query: str, path: Path | str) -> Path:
    """Escribe el resultado de ``query`` como Parquet ZSTD."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe = str(destination.resolve()).replace("'", "''")
    con.execute(
        f"COPY ({query}) TO '{safe}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    return destination


#: Agregación mensual por celda 50 km de todas las señales MODIS/VIIRS.
#: Se materializa en SQL para no arrastrar el cubo completo a pandas.
CELL_MONTH_FEATURES_SQL = """
    WITH base AS (
        SELECT
            "year", "month", year_month, cell_id_50km, cell_ix_50km, cell_iy_50km,
            country_name, iso_a3, is_continental_sa, is_in_sama,
            x_m_center, y_m_center, mean_lat, mean_lon,
            sensor_slot, raw_all_det_count, clean_det_count,
            static_excluded_det_count, high_conf_count, night_det_count,
            day_det_count, active_days, overpass_event_count, frp_sum_mw,
            active_cells_10km, nrt_det_count, sp_det_count, is_partial_month
        FROM {source}
        WHERE year_month BETWEEN '{start}' AND '{end}'
          AND sensor_slot IN (
              'MODIS_Terra', 'MODIS_Aqua',
              'VIIRS_SNPP', 'VIIRS_N20', 'VIIRS_N21'
          )
    ),
    agg AS (
        SELECT
            "year", "month", year_month, cell_id_50km, cell_ix_50km, cell_iy_50km,
            MAX(country_name) AS country_name,
            MAX(iso_a3) AS iso_a3,
            BOOL_OR(is_continental_sa) AS is_continental_sa,
            BOOL_OR(is_in_sama) AS is_in_sama,
            AVG(x_m_center) AS x_m_center,
            AVG(y_m_center) AS y_m_center,
            AVG(mean_lat) AS mean_lat,
            AVG(mean_lon) AS mean_lon,
            SUM(CASE WHEN sensor_slot = 'MODIS_Terra'
                THEN overpass_event_count ELSE 0 END)::BIGINT AS terra_events,
            SUM(CASE WHEN sensor_slot = 'MODIS_Aqua'
                THEN overpass_event_count ELSE 0 END)::BIGINT AS aqua_events,
            SUM(CASE WHEN sensor_slot LIKE 'MODIS%'
                THEN overpass_event_count ELSE 0 END)::BIGINT AS modis_events,
            COALESCE(SUM(CASE WHEN sensor_slot = 'MODIS_Terra'
                THEN frp_sum_mw ELSE 0 END), 0.0) AS terra_frp_mw,
            COALESCE(SUM(CASE WHEN sensor_slot = 'MODIS_Aqua'
                THEN frp_sum_mw ELSE 0 END), 0.0) AS aqua_frp_mw,
            COALESCE(SUM(CASE WHEN sensor_slot LIKE 'MODIS%'
                THEN frp_sum_mw ELSE 0 END), 0.0) AS modis_frp_mw,
            SUM(CASE WHEN sensor_slot = 'MODIS_Terra'
                THEN active_cells_10km ELSE 0 END)::BIGINT AS terra_active_cells,
            SUM(CASE WHEN sensor_slot = 'MODIS_Aqua'
                THEN active_cells_10km ELSE 0 END)::BIGINT AS aqua_active_cells,
            SUM(CASE WHEN sensor_slot LIKE 'MODIS%'
                THEN active_cells_10km ELSE 0 END)::BIGINT AS modis_active_cells,
            SUM(CASE WHEN sensor_slot LIKE 'MODIS%'
                THEN clean_det_count ELSE 0 END)::BIGINT AS modis_clean_det,
            SUM(CASE WHEN sensor_slot = 'MODIS_Terra'
                THEN clean_det_count ELSE 0 END)::BIGINT AS terra_clean_det,
            SUM(CASE WHEN sensor_slot LIKE 'MODIS%'
                THEN night_det_count ELSE 0 END)::BIGINT AS modis_night_det,
            SUM(CASE WHEN sensor_slot = 'MODIS_Terra'
                THEN night_det_count ELSE 0 END)::BIGINT AS terra_night_det,
            SUM(CASE WHEN sensor_slot LIKE 'MODIS%'
                THEN high_conf_count ELSE 0 END)::BIGINT AS modis_high_conf_det,
            SUM(CASE WHEN sensor_slot = 'MODIS_Terra'
                THEN high_conf_count ELSE 0 END)::BIGINT AS terra_high_conf_det,
            SUM(CASE WHEN sensor_slot = 'VIIRS_SNPP'
                THEN overpass_event_count ELSE 0 END)::BIGINT AS snpp_events,
            SUM(CASE WHEN sensor_slot = 'VIIRS_N20'
                THEN overpass_event_count ELSE 0 END)::BIGINT AS n20_events,
            SUM(CASE WHEN sensor_slot = 'VIIRS_N21'
                THEN overpass_event_count ELSE 0 END)::BIGINT AS n21_events,
            SUM(CASE WHEN sensor_slot LIKE 'VIIRS%'
                THEN overpass_event_count ELSE 0 END)::BIGINT AS viirs_events,
            COALESCE(SUM(CASE WHEN sensor_slot = 'VIIRS_SNPP'
                THEN frp_sum_mw ELSE 0 END), 0.0) AS snpp_frp_mw,
            COALESCE(SUM(CASE WHEN sensor_slot = 'VIIRS_N20'
                THEN frp_sum_mw ELSE 0 END), 0.0) AS n20_frp_mw,
            COALESCE(SUM(CASE WHEN sensor_slot = 'VIIRS_N21'
                THEN frp_sum_mw ELSE 0 END), 0.0) AS n21_frp_mw,
            COALESCE(SUM(CASE WHEN sensor_slot LIKE 'VIIRS%'
                THEN frp_sum_mw ELSE 0 END), 0.0) AS viirs_frp_mw,
            SUM(CASE WHEN sensor_slot = 'VIIRS_SNPP'
                THEN active_cells_10km ELSE 0 END)::BIGINT AS snpp_active_cells,
            SUM(CASE WHEN sensor_slot LIKE 'VIIRS%'
                THEN active_cells_10km ELSE 0 END)::BIGINT AS viirs_active_cells,
            MAX(CASE WHEN sensor_slot = 'MODIS_Terra' THEN 1 ELSE 0 END)::INTEGER
                AS has_terra,
            MAX(CASE WHEN sensor_slot = 'MODIS_Aqua' THEN 1 ELSE 0 END)::INTEGER
                AS has_aqua,
            MAX(CASE WHEN sensor_slot = 'VIIRS_SNPP' THEN 1 ELSE 0 END)::INTEGER
                AS has_snpp,
            MAX(CASE WHEN sensor_slot = 'VIIRS_N20' THEN 1 ELSE 0 END)::INTEGER
                AS has_n20,
            MAX(CASE WHEN sensor_slot = 'VIIRS_N21' THEN 1 ELSE 0 END)::INTEGER
                AS has_n21,
            SUM(raw_all_det_count)::BIGINT AS raw_all_det_count,
            SUM(clean_det_count)::BIGINT AS clean_det_count,
            SUM(static_excluded_det_count)::BIGINT AS static_excluded_det_count,
            SUM(high_conf_count)::BIGINT AS high_conf_count,
            SUM(night_det_count)::BIGINT AS night_det_count,
            SUM(day_det_count)::BIGINT AS day_det_count,
            SUM(active_days)::BIGINT AS active_days,
            SUM(overpass_event_count)::BIGINT AS overpass_event_count,
            COALESCE(SUM(frp_sum_mw), 0.0) AS frp_sum_mw,
            MAX(active_cells_10km)::BIGINT AS active_cells_10km,
            SUM(nrt_det_count)::BIGINT AS nrt_det_count,
            SUM(sp_det_count)::BIGINT AS sp_det_count,
            BOOL_OR(is_partial_month) AS is_partial_month
        FROM base
        GROUP BY
            "year", "month", year_month, cell_id_50km, cell_ix_50km, cell_iy_50km
    )
    SELECT
        "year", "month", year_month, cell_id_50km, cell_ix_50km, cell_iy_50km,
        country_name, iso_a3, is_continental_sa, is_in_sama,
        x_m_center, y_m_center, mean_lat, mean_lon,
        terra_events, aqua_events, modis_events,
        terra_frp_mw, aqua_frp_mw, modis_frp_mw,
        terra_active_cells, aqua_active_cells, modis_active_cells,
        modis_clean_det, terra_clean_det, modis_night_det, terra_night_det,
        modis_high_conf_det, terra_high_conf_det,
        CASE WHEN modis_clean_det > 0
            THEN CAST(modis_night_det AS DOUBLE) / modis_clean_det ELSE 0.0 END
            AS modis_night_frac,
        CASE WHEN modis_clean_det > 0
            THEN CAST(modis_high_conf_det AS DOUBLE) / modis_clean_det ELSE 0.0 END
            AS modis_high_conf_frac,
        has_terra, has_aqua, has_snpp, has_n20, has_n21,
        snpp_events, n20_events, n21_events, viirs_events,
        snpp_frp_mw, n20_frp_mw, n21_frp_mw, viirs_frp_mw,
        snpp_active_cells, viirs_active_cells,
        raw_all_det_count, clean_det_count, static_excluded_det_count,
        high_conf_count, night_det_count, day_det_count, active_days,
        overpass_event_count, frp_sum_mw, active_cells_10km,
        nrt_det_count, sp_det_count, is_partial_month,
        SIN(2.0 * PI() * ("month" - 1) / 12.0) AS month_sin,
        COS(2.0 * PI() * ("month" - 1) / 12.0) AS month_cos,
        ABS(COALESCE(mean_lat, 0.0)) AS abs_lat
    FROM agg
"""


def cell_month_features_query(path: Path | str, start: str, end: str) -> str:
    """Devuelve la consulta de features mensuales por celda para una ventana."""
    return CELL_MONTH_FEATURES_SQL.format(
        source=parquet_sql(path), start=start, end=end
    )


def load_cell_month_features(
    con: "duckdb.DuckDBPyConnection",
    path: Path | str,
    *,
    start: str = "2000-01",
    end: str = "2026-12",
    columns: Sequence[str] | None = None,
) -> "pandas.DataFrame":
    """Carga las features mensuales por celda 50 km en un DataFrame."""
    query = cell_month_features_query(path, start, end)
    if columns:
        query = f"SELECT {', '.join(columns)} FROM ({query}) AS features"
    return con.execute(query).fetch_df()


def fold_hashes(assignment: dict[str, int], folds: int) -> dict[str, str]:
    """SHA-256 de los IDs ordenados de cada fold."""
    by_fold: dict[int, list[str]] = {fold: [] for fold in range(int(folds))}
    for identifier, fold in assignment.items():
        by_fold.setdefault(int(fold), []).append(str(identifier))
    return {
        str(fold): hashlib.sha256("\n".join(sorted(by_fold[fold])).encode("utf-8")).hexdigest()
        for fold in range(int(folds))
    }


def fold_sizes(assignment: dict[str, int], folds: int) -> dict[str, int]:
    """Tamaño de cada fold."""
    sizes = {str(fold): 0 for fold in range(int(folds))}
    for fold in assignment.values():
        sizes[str(int(fold))] = sizes.get(str(int(fold)), 0) + 1
    return sizes


def id_hash(identifiers: Iterable[str]) -> str:
    """SHA-256 de un conjunto de identificadores ordenados."""
    return hashlib.sha256("\n".join(sorted(str(value) for value in identifiers)).encode()).hexdigest()


__all__ = [
    "PRE_DRIFT_START",
    "PRE_DRIFT_END",
    "REGIME_REAL",
    "REGIME_TERRA_ONLY",
    "RELAY_REGIMES",
    "sha256_file",
    "sha256_json",
    "stable_json",
    "write_json",
    "connect_duckdb",
    "parquet_sql",
    "write_parquet",
    "cell_month_features_query",
    "load_cell_month_features",
    "fold_hashes",
    "fold_sizes",
    "id_hash",
]
