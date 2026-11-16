"""Homologación de sensores, banderas de calidad y detección de fuentes estáticas.

El módulo trabaja enteramente con SQL de DuckDB. Las expresiones de banderas se
definen una sola vez y se reutilizan mediante CTEs/vistas temporales, de modo
que el dataset crudo nunca se materializa en pandas.
"""

from __future__ import annotations

import glob as globlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import duckdb

from src.core.duckdb_helper import connect as _core_connect
from src.core.geogrid import flag_persistent_static_sources

# Esquema canónico mínimo (se omite la columna de partición ``year`` porque el
# año se deriva de ``acq_date`` y así también funcionan fixtures sin partición).
SOURCE_COLUMNS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "brightness_primary",
    "brightness_secondary",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "acq_datetime_utc",
    "satellite",
    "instrument",
    "confidence_raw",
    "confidence_pct",
    "confidence_tier",
    "version",
    "frp",
    "daynight",
    "type",
    "source_feed",
    "source_priority",
    "country_name",
    "iso_a3",
    "admin1_name",
    "x_m",
    "y_m",
    "cell_ix_10km",
    "cell_iy_10km",
    "cell_id_10km",
)

# Expresiones que dependen solo de columnas del origen.
BASE_FLAG_EXPRESSIONS: dict[str, str] = {
    "valid_brightness": "(brightness_primary IS NOT NULL AND brightness_primary > 0.0)",
    "is_vegetation_type": '("type" IS NULL OR "type" = 0)',
    "frp_clean": (
        "CASE WHEN isnan(frp) OR frp IS NULL OR frp < 0.0 "
        "THEN 0.0 ELSE frp END"
    ),
    "is_negative_frp_raw": "(frp < 0.0)",
    "is_nominal_plus": (
        "((instrument = 'MODIS' AND COALESCE(confidence_pct, -1.0) >= 30.0) "
        "OR (instrument = 'VIIRS' AND confidence_tier IN ('nominal', 'high')) "
        "OR (confidence_tier IN ('nominal', 'high') AND confidence_pct IS NULL))"
    ),
    "is_high_conf": (
        "((instrument = 'MODIS' AND COALESCE(confidence_pct, -1.0) >= 75.0) "
        "OR (instrument = 'VIIRS' AND confidence_tier = 'high'))"
    ),
    "is_in_sama": "(latitude BETWEEN -55.0 AND 7.0 AND longitude BETWEEN -110.0 AND 11.0)",
    "is_continental_sa": "(longitude >= -82.2)",
    "overpass_bucket": (
        "CONCAT(satellite, '|', strftime(acq_date, '%Y-%m-%d'), '|', "
        "COALESCE(daynight, '?'), '|', "
        "CAST(FLOOR(COALESCE(TRY_CAST(SUBSTR(LPAD(acq_time, 4, '0'), 1, 2) AS INT), 0) / 2) AS VARCHAR))"
    ),
}

# Expresiones que pueden referenciar banderas ya calculadas en la capa base.
DEPENDENT_FLAG_EXPRESSIONS: dict[str, str] = {
    "is_sama_suspect_night": (
        "(is_in_sama AND instrument = 'VIIRS' AND daynight = 'N' "
        "AND source_feed = 'nrt_api' AND confidence_tier = 'low' "
        "AND (brightness_primary - COALESCE(brightness_secondary, brightness_primary)) < 15.0)"
    ),
}

DERIVED_COLUMNS: tuple[str, ...] = (
    *BASE_FLAG_EXPRESSIONS,
    *DEPENDENT_FLAG_EXPRESSIONS,
    "sensor_slot",
)


def quote_identifier(name: str) -> str:
    """Entrecomilla un identificador SQL."""
    return '"' + str(name).replace('"', '""') + '"'


def sql_literal(value: str | Path) -> str:
    """Devuelve un literal SQL de cadena seguro."""
    return "'" + str(value).replace("'", "''") + "'"


def _iter_paths(parquet_glob: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(parquet_glob, (list, tuple)):
        paths: list[Path] = []
        for item in parquet_glob:
            paths.extend(_iter_paths(item))
        return paths
    pattern = str(parquet_glob)
    if globlib.has_magic(pattern):
        return [Path(p) for p in sorted(globlib.glob(pattern, recursive=True)) if Path(p).is_file()]
    path = Path(pattern)
    return [path] if path.is_file() else []


def resolve_parquet_glob(parquet_glob: str | Path) -> str:
    """Resuelve el patrón canónico y, si hace falta, el nombre real de partición."""
    pattern = str(parquet_glob)
    if _iter_paths(pattern):
        return pattern
    suffix = "part-*.parquet"
    if pattern.endswith(suffix):
        alternative = pattern[: -len(suffix)] + "data_*.parquet"
        if _iter_paths(alternative):
            return alternative
    return pattern


def validate_parquet_glob(parquet_glob: str | Path | Sequence[str | Path]) -> None:
    """Falla temprano si el patrón no resuelve a ningún Parquet local."""
    if not _iter_paths(parquet_glob):
        raise FileNotFoundError(f"No se encontraron Parquet para: {parquet_glob}")


def parquet_scan_sql(parquet_glob: str | Path | Sequence[str | Path]) -> str:
    """Construye ``read_parquet(...)`` validando primero que existan archivos."""
    if isinstance(parquet_glob, (list, tuple)):
        validate_parquet_glob(parquet_glob)
        literals = ", ".join(sql_literal(p) for p in parquet_glob)
        return f"read_parquet([{literals}])"
    pattern = resolve_parquet_glob(parquet_glob)
    validate_parquet_glob(pattern)
    if globlib.has_magic(pattern):
        return f"read_parquet({sql_literal(pattern)})"
    return f"read_parquet({sql_literal(Path(pattern).resolve())})"


def connect_duckdb(
    db_path: str | Path = ":memory:",
    *,
    memory_limit: str | None = None,
    threads: int | None = None,
    temp_dir: str | Path | None = None,
) -> duckdb.DuckDBPyConnection:
    """Connect DuckDB with explicit, portable resource limits."""
    return _core_connect(
        db_path,
        memory_limit=memory_limit,
        threads=threads,
        temp_dir=temp_dir,
    )


def sensor_slot_expr(table_alias: str | None = None) -> str:
    """Expresión SQL que homologa satélite+instrumento a un slot estable."""
    prefix = f"{table_alias}." if table_alias else ""
    return (
        "CASE "
        f"WHEN {prefix}instrument = 'MODIS' AND {prefix}satellite = 'Terra' THEN 'MODIS_Terra' "
        f"WHEN {prefix}instrument = 'MODIS' AND {prefix}satellite = 'Aqua' THEN 'MODIS_Aqua' "
        f"WHEN {prefix}instrument = 'VIIRS' AND {prefix}satellite = 'SNPP' THEN 'VIIRS_SNPP' "
        f"WHEN {prefix}instrument = 'VIIRS' AND {prefix}satellite = 'N20' THEN 'VIIRS_N20' "
        f"WHEN {prefix}instrument = 'VIIRS' AND {prefix}satellite = 'N21' THEN 'VIIRS_N21' "
        f"ELSE CONCAT({prefix}instrument, '_', {prefix}satellite) "
        "END"
    )


def flag_expressions() -> dict[str, str]:
    """Devuelve las banderas en el orden en que deben exponerse."""
    return {**BASE_FLAG_EXPRESSIONS, **DEPENDENT_FLAG_EXPRESSIONS}


def flagged_columns(columns: Sequence[str] | None = None) -> tuple[str, ...]:
    """Columnas de origen más banderas derivadas (sin ``is_clean``)."""
    return tuple(columns or SOURCE_COLUMNS) + DERIVED_COLUMNS


def clean_filter_flags_cte(
    source_sql: str = "read_parquet(?)",
    *,
    columns: Sequence[str] | None = None,
) -> str:
    """Construye una consulta con todas las banderas de homologación.

    La capa ``clean_base`` calcula las banderas independientes; la capa
    ``flagged`` añade ``is_sama_suspect_night`` sin repetir la expresión de
    ``is_in_sama``.
    """
    source_columns = tuple(columns or SOURCE_COLUMNS)
    source_sql_list = ", ".join(quote_identifier(c) for c in source_columns)
    base_flags = ",\n        ".join(
        f"{expr} AS {quote_identifier(name)}" for name, expr in BASE_FLAG_EXPRESSIONS.items()
    )
    base_sensor = f"{sensor_slot_expr()} AS {quote_identifier('sensor_slot')}"
    final_columns = ", ".join(quote_identifier(c) for c in flagged_columns(source_columns))
    dependent = ",\n        ".join(
        f"{expr} AS {quote_identifier(name)}" for name, expr in DEPENDENT_FLAG_EXPRESSIONS.items()
    )
    base_output = ", ".join(
        [*(quote_identifier(c) for c in source_columns), *(quote_identifier(c) for c in BASE_FLAG_EXPRESSIONS), quote_identifier("sensor_slot")]
    )
    return f"""WITH clean_base AS (
    SELECT
        {source_sql_list},
        {base_flags},
        {base_sensor}
    FROM {source_sql}
),
flagged AS (
    SELECT
        {base_output},
        {dependent}
    FROM clean_base
)
SELECT
    {final_columns}
FROM flagged"""


def _write_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _copy_parquet(con: duckdb.DuckDBPyConnection, query: str, out_parquet: str | Path) -> None:
    target = Path(out_parquet)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    con.execute(
        f"COPY ({query}) TO {sql_literal(target)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def identify_static_thermal_cells(
    con: duckdb.DuckDBPyConnection,
    parquet_glob: str | Path | Sequence[str | Path],
    out_parquet: str | Path,
    summary_json: str | Path | None = None,
    *,
    min_active_months: int = 48,
    min_active_years: int = 5,
    min_months_per_year: float = 9.0,
    min_type12_count: int = 5,
) -> dict[str, Any]:
    """Detecta celdas 10 km con persistencia crónica o dominio de tipo 1/2.

    Criterio temporal: ``distinct_ym >= min_active_months``,
    ``active_years >= min_active_years`` y
    ``avg_months_per_year >= min_months_per_year``. Criterio alternativo:
    al menos ``min_type12_count`` filas con ``type`` 1 o 2 y una proporción
    de esas filas superior a 0.50.

    ``distinct_ym`` se calcula como entero ``YYYYMM`` para evitar agrupar
    millones de cadenas de texto durante la detección.
    """
    if int(min_active_months) < 1 or int(min_active_years) < 1:
        raise ValueError("min_active_months y min_active_years deben ser >= 1.")
    if float(min_months_per_year) < 0.0:
        raise ValueError("min_months_per_year debe ser >= 0.")
    if int(min_type12_count) < 1:
        raise ValueError("min_type12_count debe ser >= 1.")

    source = parquet_scan_sql(parquet_glob)
    con.execute("DROP TABLE IF EXISTS _static_cell_stats")
    con.execute(
        f"""
        CREATE TEMP TABLE _static_cell_stats AS
        WITH base AS (
            SELECT
                cell_id_10km,
                COUNT(*)::BIGINT AS det_count,
                COUNT(DISTINCT (
                    CAST(EXTRACT(year FROM acq_date) AS INTEGER) * 100
                    + CAST(EXTRACT(month FROM acq_date) AS INTEGER)
                ))::BIGINT AS distinct_ym,
                COUNT(DISTINCT CAST(EXTRACT(year FROM acq_date) AS INTEGER))::BIGINT AS active_years,
                COUNT(*) FILTER (WHERE "type" IN (1, 2))::BIGINT AS type_12_count,
                COUNT("type")::BIGINT AS typed_count,
                MIN(acq_date) AS first_date,
                MAX(acq_date) AS last_date,
                AVG(latitude) AS mean_lat,
                AVG(longitude) AS mean_lon
            FROM {source}
            WHERE cell_id_10km IS NOT NULL
            GROUP BY cell_id_10km
        ),
        stats AS (
            SELECT
                cell_id_10km,
                det_count,
                distinct_ym,
                active_years,
                distinct_ym * 1.0 / NULLIF(active_years, 0) AS avg_months_per_year,
                type_12_count,
                typed_count,
                type_12_count * 1.0 / NULLIF(typed_count, 0) AS type_12_ratio,
                first_date,
                last_date,
                mean_lat,
                mean_lon
            FROM base
        )
        SELECT
            cell_id_10km,
            det_count,
            distinct_ym,
            active_years,
            avg_months_per_year,
            type_12_count,
            typed_count,
            type_12_ratio,
            first_date,
            last_date,
            mean_lat,
            mean_lon,
            (
                distinct_ym >= {int(min_active_months)}
                AND active_years >= {int(min_active_years)}
                AND avg_months_per_year >= {float(min_months_per_year)}
            ) AS is_persistent_temporal,
            (type_12_count >= {int(min_type12_count)} AND type_12_ratio > 0.50)
                AS is_type_12_dominant
        FROM stats
        """
    )
    n_observed = int(con.execute("SELECT COUNT(*) FROM _static_cell_stats").fetchone()[0])
    _copy_parquet(
        con,
        """
        SELECT
            cell_id_10km,
            det_count,
            distinct_ym,
            active_years,
            avg_months_per_year,
            type_12_count,
            typed_count,
            type_12_ratio,
            first_date,
            last_date,
            mean_lat,
            mean_lon,
            is_persistent_temporal,
            is_type_12_dominant
        FROM _static_cell_stats
        WHERE is_persistent_temporal OR is_type_12_dominant
        ORDER BY cell_id_10km
        """,
        out_parquet,
    )
    con.execute("DROP TABLE IF EXISTS _static_cell_stats")

    row = con.execute(
        f"""
        SELECT
            COUNT(*)::BIGINT AS static_cell_count,
            COALESCE(SUM(det_count), 0)::BIGINT AS static_det_count,
            COUNT(*) FILTER (WHERE is_persistent_temporal)::BIGINT AS n_persistent_temporal,
            COUNT(*) FILTER (WHERE is_type_12_dominant)::BIGINT AS n_type_12_dominant
        FROM read_parquet({sql_literal(Path(out_parquet).resolve())})
        """
    ).fetchone()
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parquet_glob": str(parquet_glob),
        "out_parquet": str(Path(out_parquet)),
        "min_active_months": int(min_active_months),
        "min_active_years": int(min_active_years),
        "min_months_per_year": float(min_months_per_year),
        "min_type12_count": int(min_type12_count),
        "min_type_12_ratio": 0.50,
        "n_cells_observed": n_observed,
        "static_cell_count": int(row[0]),
        "n_static_cells": int(row[0]),
        "static_det_count": int(row[1]),
        "n_static_rows": int(row[1]),
        "n_persistent_temporal": int(row[2]),
        "n_type_12_dominant": int(row[3]),
    }
    _write_json(summary_json, summary)
    return summary


def compute_qc_summary(
    con: duckdb.DuckDBPyConnection,
    parquet_glob: str | Path | Sequence[str | Path],
    static_cells_parquet: str | Path,
    out_json: str | Path,
) -> dict[str, Any]:
    """Resume la pérdida de detecciones entre crudo y armonizado por sensor."""
    source = parquet_scan_sql(parquet_glob)
    validate_parquet_glob(static_cells_parquet)
    static_source = parquet_scan_sql(static_cells_parquet)
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW _qc_flagged AS {clean_filter_flags_cte(source)}"
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _qc_static_cells AS
        SELECT DISTINCT cell_id_10km
        FROM {static_source}
        WHERE cell_id_10km IS NOT NULL
        """
    )
    metric_names = (
        "raw_all_det_count",
        "valid_veg_nominal_plus",
        "clean_det_count",
        "static_clean_candidate_rows",
        "excluded_type_123",
        "negative_frp_raw",
        "zero_brightness",
        "sama_rows",
        "sama_suspect_night",
        "static_excluded_rows",
        "nrt_rows",
    )
    qc_columns = ",\n                ".join(quote_identifier(c) for c in flagged_columns())
    rows = con.execute(
        f"""
        SELECT
            sensor_slot,
            COUNT(*)::BIGINT AS raw_all_det_count,
            COUNT(*) FILTER (WHERE is_clean_candidate)::BIGINT
                AS valid_veg_nominal_plus,
            COUNT(*) FILTER (
                WHERE is_clean_candidate AND NOT is_static_cell
            )::BIGINT AS clean_det_count,
            COUNT(*) FILTER (
                WHERE is_clean_candidate AND is_static_cell
            )::BIGINT AS static_clean_candidate_rows,
            COUNT(*) FILTER (WHERE "type" IN (1, 2, 3))::BIGINT AS excluded_type_123,
            COUNT(*) FILTER (WHERE is_negative_frp_raw)::BIGINT AS negative_frp_raw,
            COUNT(*) FILTER (
                WHERE brightness_primary IS NULL OR brightness_primary <= 0.0
            )::BIGINT AS zero_brightness,
            COUNT(*) FILTER (WHERE is_in_sama)::BIGINT AS sama_rows,
            COUNT(*) FILTER (WHERE is_sama_suspect_night)::BIGINT AS sama_suspect_night,
            COUNT(*) FILTER (WHERE is_static_cell)::BIGINT AS static_excluded_rows,
            COUNT(*) FILTER (WHERE source_feed = 'nrt_api')::BIGINT AS nrt_rows
        FROM (
            SELECT
                {qc_columns},
                (valid_brightness AND is_vegetation_type AND is_nominal_plus)
                    AS is_clean_candidate,
                COALESCE(
                    cell_id_10km IN (SELECT cell_id_10km FROM _qc_static_cells),
                    FALSE
                ) AS is_static_cell
            FROM _qc_flagged
        )
        GROUP BY sensor_slot
        ORDER BY sensor_slot NULLS LAST
        """
    ).fetchall()

    by_sensor_slot: list[dict[str, Any]] = []
    totals = {name: 0 for name in metric_names}
    for row in rows:
        entry: dict[str, Any] = {"sensor_slot": row[0]}
        for name, value in zip(metric_names, row[1:]):
            entry[name] = int(value or 0)
            totals[name] += entry[name]
        by_sensor_slot.append(entry)

    static_row = con.execute(
        f"""
        SELECT COUNT(*)::BIGINT, COALESCE(SUM(det_count), 0)::BIGINT
        FROM {static_source}
        """
    ).fetchone()
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parquet_glob": str(parquet_glob),
        "static_cells_parquet": str(Path(static_cells_parquet)),
        **totals,
        "total_raw_rows": totals["raw_all_det_count"],
        "total_raw": totals["raw_all_det_count"],
        "total_rows": totals["raw_all_det_count"],
        "static_cell_count": int(static_row[0]),
        "n_static_cells": int(static_row[0]),
        "static_det_count": int(static_row[1]),
        "by_sensor_slot": by_sensor_slot,
    }
    _write_json(out_json, summary)
    con.execute("DROP VIEW IF EXISTS _qc_flagged")
    con.execute("DROP TABLE IF EXISTS _qc_static_cells")
    return summary


def cross_check_static_cells_with_geogrid(
    con: duckdb.DuckDBPyConnection,
    parquet_glob: str | Path | Sequence[str | Path],
    static_cells_parquet: str | Path,
    *,
    sample_rows: int = 200_000,
    min_active_days: int = 60,
    min_distinct_months: int = 6,
) -> dict[str, Any]:
    """Cross-check a sample with :func:`src.core.geogrid.flag_persistent_static_sources`."""
    source = parquet_scan_sql(parquet_glob)
    sample = con.execute(
        f"""
        SELECT cell_id_10km, acq_date, latitude, longitude
        FROM {source}
        WHERE cell_id_10km IS NOT NULL
        USING SAMPLE {max(1, int(sample_rows))} ROWS
        """
    ).fetchdf()
    if sample.empty:
        return {"available": True, "sample_rows": 0, "agreement": None}
    flagged = flag_persistent_static_sources(
        sample,
        cell_col="cell_id_10km",
        date_col="acq_date",
        min_active_days=int(min_active_days),
        min_distinct_months=int(min_distinct_months),
    )
    helper_cells = set(
        flagged.loc[flagged["is_persistent_static"], "cell_id_10km"].astype(str).tolist()
    )
    contract_cells = set(
        con.execute(
            f"SELECT DISTINCT cell_id_10km FROM {parquet_scan_sql(static_cells_parquet)}"
        )
        .fetchdf()["cell_id_10km"]
        .astype(str)
        .tolist()
    )
    observed_cells = set(sample["cell_id_10km"].astype(str).tolist())
    helper_in_sample = helper_cells & observed_cells
    contract_in_sample = contract_cells & observed_cells
    union = helper_in_sample | contract_in_sample
    return {
        "available": True,
        "sample_rows": int(len(sample)),
        "helper_static_cells_in_sample": len(helper_in_sample),
        "contract_static_cells_in_sample": len(contract_in_sample),
        "overlap_cells": len(helper_in_sample & contract_in_sample),
        "agreement": (len(helper_in_sample & contract_in_sample) / len(union)) if union else None,
        "min_active_days": int(min_active_days),
        "min_distinct_months": int(min_distinct_months),
    }
