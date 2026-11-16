#!/usr/bin/env python3
"""Construye los cubos espaciotemporales armonizados con DuckDB out-of-core.

El crudo nunca se materializa en pandas: todas las agregaciones se expresan en
SQL y se escriben directamente a Parquet ZSTD. Los cubos conservan
``raw_all_det_count`` y ``static_excluded_det_count`` para auditar la diferencia
entre el crudo y el conjunto armonizado.
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from src.data.clean_and_flag import (
    clean_filter_flags_cte,
    compute_qc_summary,
    connect_duckdb,
    flagged_columns,
    identify_static_thermal_cells,
    parquet_scan_sql,
    quote_identifier,
    resolve_parquet_glob,
    sql_literal,
)

CUBE_NAMES = {
    "cell10km_monthly": "cube_cell10km_monthly.parquet",
    "cell50km_monthly": "cube_cell50km_monthly.parquet",
    "admin1_monthly": "cube_admin1_monthly.parquet",
    "country_monthly": "cube_country_monthly.parquet",
    "admin1_daily_season": "cube_admin1_daily_season.parquet",
    "static_cells": "static_thermal_cells.parquet",
    "geography_catalog": "geography_catalog.json",
    "qc_summary": "qc_cleaning_summary.json",
}

CELL10KM_MONTHLY_COLUMNS: tuple[str, ...] = (
    "year",
    "month",
    "year_month",
    "cell_id_10km",
    "cell_ix_10km",
    "cell_iy_10km",
    "x_m",
    "y_m",
    "country_name",
    "iso_a3",
    "admin1_name",
    "sensor_slot",
    "is_continental_sa",
    "is_in_sama",
    "raw_all_det_count",
    "clean_det_count",
    "static_excluded_det_count",
    "high_conf_count",
    "night_det_count",
    "day_det_count",
    "active_days",
    "overpass_event_count",
    "frp_sum_mw",
    "frp_p50_mw",
    "frp_p90_mw",
    "frp_max_mw",
    "mean_scan",
    "mean_track",
    "mean_lat",
    "mean_lon",
    "nrt_det_count",
    "sp_det_count",
    "is_nrt_majority",
    "is_partial_month",
)

CELL50KM_MONTHLY_COLUMNS: tuple[str, ...] = (
    "year",
    "month",
    "year_month",
    "cell_ix_50km",
    "cell_iy_50km",
    "cell_id_50km",
    "country_name",
    "iso_a3",
    "sensor_slot",
    "is_continental_sa",
    "is_in_sama",
    "raw_all_det_count",
    "clean_det_count",
    "static_excluded_det_count",
    "high_conf_count",
    "night_det_count",
    "day_det_count",
    "active_days",
    "overpass_event_count",
    "frp_sum_mw",
    "active_cells_10km",
    "mean_lat",
    "mean_lon",
    "x_m_center",
    "y_m_center",
    "nrt_det_count",
    "sp_det_count",
    "is_nrt_majority",
    "is_partial_month",
)

RAW_ADMIN1_MONTHLY_COLUMNS: tuple[str, ...] = (
    "year",
    "month",
    "year_month",
    "country_name",
    "iso_a3",
    "admin1_name",
    "sensor_slot",
    "raw_all_det_count",
    "min_lat",
    "max_lat",
    "min_lon",
    "max_lon",
    "intersects_sama",
)

ADMIN1_DAILY_SEASON_COLUMNS: tuple[str, ...] = (
    "year",
    "acq_date",
    "doy",
    "country_name",
    "iso_a3",
    "admin1_name",
    "sensor_slot",
    "clean_det_count",
    "frp_sum_mw",
    "active_cells_10km",
    "overpass_event_count",
)

_YEAR_PARTITION_RE = re.compile(r"^year=(\d{4})$")

# Un año muy poblado (p. ej. 2024 con 12.6 M de focos) todavía agota la memoria
# al calcular ``approx_quantile`` por celda-mes-sensor. Por encima de este
# umbral se subdivide el año en doce lotes mensuales y se recompone el
# fragmento anual antes de la unión final.
_YEAR_MONTH_SPLIT_ROWS = 4_000_000

_CELL10KM_MONTHLY_QUERY = """
        WITH agg AS (
            SELECT
                CAST(EXTRACT(year FROM acq_date) AS INTEGER) AS "year",
                CAST(EXTRACT(month FROM acq_date) AS INTEGER) AS "month",
                strftime(acq_date, '%Y-%m') AS year_month,
                cell_id_10km,
                cell_ix_10km,
                cell_iy_10km,
                AVG(x_m) AS x_m,
                AVG(y_m) AS y_m,
                country_name,
                iso_a3,
                admin1_name,
                sensor_slot,
                is_continental_sa,
                is_in_sama,
                COUNT(*)::BIGINT AS raw_all_det_count,
                COUNT(*) FILTER (WHERE is_clean)::BIGINT AS clean_det_count,
                COUNT(*) FILTER (
                    WHERE is_clean_candidate AND is_static_cell
                )::BIGINT AS static_excluded_det_count,
                COUNT(*) FILTER (WHERE is_clean AND is_high_conf)::BIGINT
                    AS high_conf_count,
                COUNT(*) FILTER (WHERE is_clean AND daynight = 'N')::BIGINT
                    AS night_det_count,
                COUNT(*) FILTER (WHERE is_clean AND daynight = 'D')::BIGINT
                    AS day_det_count,
                COUNT(DISTINCT CASE WHEN is_clean THEN acq_date END)::BIGINT
                    AS active_days,
                COUNT(DISTINCT CASE WHEN is_clean THEN overpass_bucket END)::BIGINT
                    AS overpass_event_count,
                COALESCE(SUM(CASE WHEN is_clean THEN frp_clean END), 0.0)
                    AS frp_sum_mw,
                approx_quantile(CASE WHEN is_clean THEN frp_clean END, 0.50)
                    AS frp_p50_mw,
                approx_quantile(CASE WHEN is_clean THEN frp_clean END, 0.90)
                    AS frp_p90_mw,
                MAX(CASE WHEN is_clean THEN frp_clean END) AS frp_max_mw,
                AVG(CASE WHEN is_clean THEN scan END) AS mean_scan,
                AVG(CASE WHEN is_clean THEN track END) AS mean_track,
                AVG(CASE WHEN is_clean THEN latitude END) AS mean_lat,
                AVG(CASE WHEN is_clean THEN longitude END) AS mean_lon,
                COUNT(*) FILTER (
                    WHERE is_clean AND source_feed = 'nrt_api'
                )::BIGINT AS nrt_det_count,
                COUNT(*) FILTER (
                    WHERE is_clean AND COALESCE(source_feed, '') <> 'nrt_api'
                )::BIGINT AS sp_det_count
            FROM v_clean
            GROUP BY
                "year", "month", year_month, cell_id_10km, cell_ix_10km, cell_iy_10km,
                country_name, iso_a3, admin1_name, sensor_slot,
                is_continental_sa, is_in_sama
        )
        SELECT
            "year",
            "month",
            year_month,
            cell_id_10km,
            cell_ix_10km,
            cell_iy_10km,
            x_m,
            y_m,
            country_name,
            iso_a3,
            admin1_name,
            sensor_slot,
            is_continental_sa,
            is_in_sama,
            raw_all_det_count,
            clean_det_count,
            static_excluded_det_count,
            high_conf_count,
            night_det_count,
            day_det_count,
            active_days,
            overpass_event_count,
            frp_sum_mw,
            frp_p50_mw,
            frp_p90_mw,
            frp_max_mw,
            mean_scan,
            mean_track,
            mean_lat,
            mean_lon,
            nrt_det_count,
            sp_det_count,
            (nrt_det_count > sp_det_count) AS is_nrt_majority,
            (("year" = 2026 AND "month" = 11)) AS is_partial_month
        FROM agg
    """

_RAW_ADMIN1_MONTHLY_QUERY = """
        SELECT
            CAST(EXTRACT(year FROM acq_date) AS INTEGER) AS "year",
            CAST(EXTRACT(month FROM acq_date) AS INTEGER) AS "month",
            strftime(acq_date, '%Y-%m') AS year_month,
            country_name,
            iso_a3,
            admin1_name,
            sensor_slot,
            COUNT(*)::BIGINT AS raw_all_det_count,
            MIN(latitude) AS min_lat,
            MAX(latitude) AS max_lat,
            MIN(longitude) AS min_lon,
            MAX(longitude) AS max_lon,
            BOOL_OR(is_in_sama) AS intersects_sama
        FROM v_flagged
        GROUP BY
            "year", "month", year_month, country_name, iso_a3, admin1_name, sensor_slot
    """

_ADMIN1_DAILY_SEASON_QUERY = """
        SELECT
            CAST(EXTRACT(year FROM acq_date) AS INTEGER) AS "year",
            acq_date,
            CAST(dayofyear(acq_date) AS INTEGER) AS doy,
            country_name,
            iso_a3,
            admin1_name,
            sensor_slot,
            COUNT(*) FILTER (WHERE is_clean)::BIGINT AS clean_det_count,
            COALESCE(SUM(CASE WHEN is_clean THEN frp_clean END), 0.0) AS frp_sum_mw,
            COUNT(DISTINCT CASE WHEN is_clean
                THEN cell_id_10km END)::BIGINT AS active_cells_10km,
            COUNT(DISTINCT CASE WHEN is_clean
                THEN CONCAT(cell_id_10km, '|', overpass_bucket) END)::BIGINT
                AS overpass_event_count
        FROM v_clean
        GROUP BY
            "year", acq_date, doy, country_name, iso_a3, admin1_name, sensor_slot
        HAVING COUNT(*) FILTER (WHERE is_clean) > 0
    """


def _select_columns(columns: Sequence[str]) -> str:
    """Lista explícita de columnas para los COPY finales."""
    return ", ".join(quote_identifier(column) for column in columns)


def _resolved_parquet_files(
    parquet_glob: str | Path | Sequence[str | Path],
) -> list[Path]:
    """Resuelve el glob a archivos reales, sin duplicados y en orden estable."""
    if isinstance(parquet_glob, (list, tuple)):
        files: list[Path] = []
        for item in parquet_glob:
            files.extend(_resolved_parquet_files(item))
        return sorted(set(files))
    pattern = str(parquet_glob)
    if globlib.has_magic(pattern):
        candidates = globlib.glob(pattern, recursive=True)
    else:
        candidates = [pattern]
    return sorted({Path(candidate) for candidate in candidates if Path(candidate).is_file()})


def _partitioned_year_sources(
    files: Sequence[Path],
) -> list[tuple[int, list[Path]]] | None:
    """Agrupa archivos por carpeta ``year=YYYY``; ``None`` si no es particionado."""
    by_year: dict[int, list[Path]] = {}
    year_dir_by_year: dict[int, Path] = {}
    for file_path in files:
        matches = [
            (index, int(match.group(1)))
            for index, part in enumerate(file_path.parts)
            if (match := _YEAR_PARTITION_RE.match(part))
        ]
        if len(matches) != 1:
            return None
        year_index, year = matches[0]
        year_dir = Path(*file_path.parts[: year_index + 1])
        if year in year_dir_by_year and year_dir_by_year[year] != year_dir:
            return None
        year_dir_by_year[year] = year_dir
        by_year.setdefault(year, []).append(file_path)
    if not by_year:
        return None
    return [(year, sorted(by_year[year])) for year in sorted(by_year)]


def _iter_year_sources(
    con, parquet_glob: str | Path | Sequence[str | Path]
) -> list[tuple[int, list[Path] | str]]:
    """Devuelve ``(year, archivos_o_filtro)`` para procesar un año a la vez.

    Si todos los archivos viven en carpetas ``year=YYYY`` inequívocas se
    devuelve la lista de archivos de cada año, de modo que cada pasada escanea
    solo una partición. En caso contrario se consultan los años presentes en
    ``acq_date`` y se devuelve una predicción SQL para filtrar cada año.
    """
    files = _resolved_parquet_files(parquet_glob)
    if not files:
        return []
    partitioned = _partitioned_year_sources(files)
    if partitioned is not None:
        return partitioned
    source = parquet_scan_sql(parquet_glob)
    rows = con.execute(
        f"""
        SELECT DISTINCT CAST(EXTRACT(year FROM acq_date) AS INTEGER) AS "year"
        FROM {source}
        WHERE acq_date IS NOT NULL
        ORDER BY 1
        """
    ).fetchall()
    return [
        (
            int(row[0]),
            f"CAST(EXTRACT(year FROM acq_date) AS INTEGER) = {int(row[0])}",
        )
        for row in rows
        if row[0] is not None
    ]


def _source_for_year(
    parquet_glob: str | Path | Sequence[str | Path],
    year_source: list[Path] | str,
    month: int | None = None,
) -> str:
    """Construye la relación SQL de un solo año (o de un mes de ese año)."""
    month_filter = (
        None
        if month is None
        else f"CAST(EXTRACT(month FROM acq_date) AS INTEGER) = {int(month)}"
    )
    if isinstance(year_source, (list, tuple)):
        source = parquet_scan_sql(year_source)
        return source if month_filter is None else f"{source} WHERE {month_filter}"
    predicate = str(year_source)
    if month_filter is not None:
        predicate = f"({predicate}) AND {month_filter}"
    return f"{parquet_scan_sql(parquet_glob)} WHERE {predicate}"


def _parquet_row_count(
    con, files: Sequence[str | Path]
) -> int:
    """Cuenta filas usando solo el footer Parquet (sin leer datos)."""
    if not files:
        return 0
    literals = ", ".join(sql_literal(path) for path in files)
    row = con.execute(
        f"SELECT COALESCE(SUM(num_rows), 0) FROM parquet_file_metadata([{literals}])"
    ).fetchone()
    return int(row[0] or 0)


def _year_month_batches(
    con,
    parquet_glob: str | Path | Sequence[str | Path],
    year_source: list[Path] | str,
) -> list[int] | None:
    """Devuelve los meses a procesar si el lote anual es demasiado grande."""
    if isinstance(year_source, (list, tuple)):
        rows = _parquet_row_count(con, year_source)
    else:
        rows = _parquet_row_count(con, _resolved_parquet_files(parquet_glob))
    if rows <= _YEAR_MONTH_SPLIT_ROWS:
        return None
    return list(range(1, 13))


def _create_year_views(
    con,
    parquet_glob: str | Path | Sequence[str | Path],
    static_cells_parquet: Path | None,
    year_source: list[Path] | str,
    month: int | None,
) -> None:
    """Crea las vistas de un lote anual o mensual según haga falta."""
    if static_cells_parquet is None:
        _create_flagged_view(con, parquet_glob, year_source, month=month)
    else:
        _create_views(
            con,
            parquet_glob,
            static_cells_parquet,
            year_source=year_source,
            month=month,
        )


@contextmanager
def _cube_scratch_dir(label: str) -> Iterator[Path]:
    """Directorio scratch en NVMe, con fallback a ``tempfile`` y limpieza final."""
    base = Path.home() / "data-scratch" / "tmp" / f"cubes_{os.getpid()}"
    fallback: Any = None
    work: Path | None = None
    try:
        try:
            base.mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix=f"{label}_", dir=str(base)))
        except OSError:
            fallback = tempfile.TemporaryDirectory(prefix=f"cubes_{label}_")
            work = Path(fallback.name)
        yield work
    finally:
        if fallback is not None:
            fallback.cleanup()
        shutil.rmtree(base, ignore_errors=True)


def _merge_year_fragments(
    con,
    fragments: Sequence[Path],
    columns: Sequence[str],
    out_path: Path,
) -> None:
    """Une los fragmentos anuales con columnas explícitas en un solo Parquet."""
    if not fragments:
        raise ValueError("No hay fragmentos anuales para unir.")
    query = f"SELECT {_select_columns(columns)} FROM {parquet_scan_sql(fragments)}"
    _write_parquet(con, query, out_path)


def _drop_temp_objects(con, *, include_static: bool = False) -> None:
    """Libera vistas/tablas temporales de la pasada anual."""
    for view_name in ("v_clean", "v_scoped", "v_flagged"):
        con.execute(f"DROP VIEW IF EXISTS {view_name}")
    if include_static:
        con.execute("DROP TABLE IF EXISTS static_cells")


def _build_batched_parquet(
    con,
    *,
    parquet_glob: str | Path | Sequence[str | Path],
    static_cells_parquet: Path | None,
    label: str,
    query: str,
    columns: Sequence[str],
    out_path: Path,
) -> None:
    """Escribe un cubo por años (y meses para años muy poblados) y lo une."""
    include_static = static_cells_parquet is not None
    with _cube_scratch_dir(label) as scratch:
        year_fragments: list[Path] = []
        for year, year_source in _iter_year_sources(con, parquet_glob):
            months = _year_month_batches(con, parquet_glob, year_source)
            year_fragment = scratch / f"year={year}.parquet"
            try:
                if months is None:
                    _create_year_views(
                        con,
                        parquet_glob,
                        static_cells_parquet,
                        year_source,
                        None,
                    )
                    _write_parquet(con, query, year_fragment)
                else:
                    month_fragments: list[Path] = []
                    try:
                        for month in months:
                            _create_year_views(
                                con,
                                parquet_glob,
                                static_cells_parquet,
                                year_source,
                                month,
                            )
                            month_fragment = (
                                scratch / f"year={year}_month={month:02d}.parquet"
                            )
                            _write_parquet(con, query, month_fragment)
                            month_fragments.append(month_fragment)
                        _merge_year_fragments(
                            con, month_fragments, columns, year_fragment
                        )
                    finally:
                        for month_fragment in month_fragments:
                            month_fragment.unlink(missing_ok=True)
                year_fragments.append(year_fragment)
            finally:
                _drop_temp_objects(con, include_static=include_static)
        if not year_fragments:
            # Fuente vacía: produce un Parquet con el esquema esperado.
            try:
                _create_year_views(
                    con,
                    parquet_glob,
                    static_cells_parquet,
                    "FALSE",
                    None,
                )
                year_fragment = scratch / "year=empty.parquet"
                _write_parquet(con, query, year_fragment)
                year_fragments.append(year_fragment)
            finally:
                _drop_temp_objects(con, include_static=include_static)
        _merge_year_fragments(con, year_fragments, columns, out_path)


def _write_parquet(con, query: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con.execute(
        f"COPY ({query}) TO {sql_literal(path)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def default_parquet_glob(project_root: Path) -> str:
    """Patrón canónico, con fallback al nombre real de las particiones locales."""
    base = project_root / "data" / "parquet" / "south_america_hotspots"
    for pattern in ("year=*/part-*.parquet", "year=*/data_*.parquet", "year=*/*.parquet"):
        candidate = base / pattern
        if globlib.glob(str(candidate), recursive=True):
            return str(candidate)
    return str(base / "year=*/part-*.parquet")


def _relation(parquet_glob: str | Path | Sequence[str | Path]) -> str:
    return parquet_scan_sql(parquet_glob)


def _create_views(
    con,
    parquet_glob: str | Path | Sequence[str | Path],
    static_cells_parquet: Path,
    *,
    year_source: list[Path] | str | None = None,
    month: int | None = None,
) -> None:
    """Crea las vistas temporales crudas, con banderas y limpias."""
    source = (
        _relation(parquet_glob)
        if year_source is None
        else _source_for_year(parquet_glob, year_source, month=month)
    )
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW v_flagged AS {clean_filter_flags_cte(source)}"
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE static_cells AS
        SELECT DISTINCT cell_id_10km
        FROM {parquet_scan_sql(static_cells_parquet)}
        WHERE cell_id_10km IS NOT NULL
        """
    )
    base_columns = ", ".join(quote_identifier(c) for c in flagged_columns())
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW v_scoped AS
        SELECT
            {base_columns},
            COALESCE(
                cell_id_10km IN (SELECT cell_id_10km FROM static_cells),
                FALSE
            ) AS is_static_cell
        FROM v_flagged
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW v_clean AS
        SELECT
            {base_columns},
            is_static_cell,
            (valid_brightness AND is_vegetation_type AND is_nominal_plus)
                AS is_clean_candidate,
            (
                valid_brightness AND is_vegetation_type AND is_nominal_plus
                AND NOT is_static_cell
            ) AS is_clean
        FROM v_scoped
        """
    )


def _create_flagged_view(
    con,
    parquet_glob: str | Path | Sequence[str | Path],
    year_source: list[Path] | str,
    *,
    month: int | None = None,
) -> None:
    """Crea solo la vista cruda con banderas (sin tabla de celdas estáticas)."""
    source = _source_for_year(parquet_glob, year_source, month=month)
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW v_flagged AS {clean_filter_flags_cte(source)}"
    )


def build_cell10km_monthly(
    con,
    out_path: Path,
    parquet_glob: str | Path | Sequence[str | Path] | None = None,
    static_cells_parquet: Path | None = None,
) -> None:
    """Cubo 10 km mensual con métricas crudas y limpias en la misma pasada.

    Cuando se recibe ``parquet_glob`` se procesa una partición anual a la vez y
    se unen fragmentos al final; así el ``GROUP BY`` con cuantiles aproximados y
    ``COUNT(DISTINCT ...)`` nunca ve los 27 años simultáneamente.
    """
    con.execute("SET preserve_insertion_order=false")
    if parquet_glob is None or static_cells_parquet is None:
        _write_parquet(con, _CELL10KM_MONTHLY_QUERY, out_path)
        return

    _build_batched_parquet(
        con,
        parquet_glob=parquet_glob,
        static_cells_parquet=static_cells_parquet,
        label="cell10km_monthly",
        query=_CELL10KM_MONTHLY_QUERY,
        columns=CELL10KM_MONTHLY_COLUMNS,
        out_path=out_path,
    )


def _cell50km_monthly_query(source_sql: str) -> str:
    """Consulta de reagregación 10 km -> 50 km para una fuente ya filtrada."""
    return f"""
        WITH base AS (
            SELECT
                "year", "month", year_month, cell_id_10km, cell_ix_10km, cell_iy_10km,
                mean_lat, mean_lon, x_m, y_m, country_name, iso_a3, sensor_slot,
                is_continental_sa, is_in_sama, raw_all_det_count, clean_det_count,
                static_excluded_det_count, high_conf_count, night_det_count,
                day_det_count, active_days, overpass_event_count, frp_sum_mw,
                nrt_det_count, sp_det_count
            FROM {source_sql}
        ),
        gridded AS (
            SELECT
                "year", "month", year_month, cell_id_10km, cell_ix_10km, cell_iy_10km,
                mean_lat, mean_lon, x_m, y_m, country_name, iso_a3, sensor_slot,
                is_continental_sa, is_in_sama, raw_all_det_count, clean_det_count,
                static_excluded_det_count, high_conf_count, night_det_count,
                day_det_count, active_days, overpass_event_count, frp_sum_mw,
                nrt_det_count, sp_det_count,
                CAST(FLOOR(cell_ix_10km / 5.0) AS INTEGER) AS cell_ix_50km,
                CAST(FLOOR(cell_iy_10km / 5.0) AS INTEGER) AS cell_iy_50km
            FROM base
        ),
        agg AS (
            SELECT
                "year",
                "month",
                year_month,
                cell_ix_50km,
                cell_iy_50km,
                CONCAT(
                    CAST(cell_ix_50km AS VARCHAR), '_', CAST(cell_iy_50km AS VARCHAR)
                ) AS cell_id_50km,
                country_name,
                iso_a3,
                sensor_slot,
                is_continental_sa,
                is_in_sama,
                SUM(raw_all_det_count)::BIGINT AS raw_all_det_count,
                SUM(clean_det_count)::BIGINT AS clean_det_count,
                SUM(static_excluded_det_count)::BIGINT AS static_excluded_det_count,
                SUM(high_conf_count)::BIGINT AS high_conf_count,
                SUM(night_det_count)::BIGINT AS night_det_count,
                SUM(day_det_count)::BIGINT AS day_det_count,
                SUM(active_days)::BIGINT AS active_days,
                SUM(overpass_event_count)::BIGINT AS overpass_event_count,
                COALESCE(SUM(frp_sum_mw), 0.0) AS frp_sum_mw,
                COUNT(DISTINCT CASE WHEN clean_det_count > 0
                    THEN cell_id_10km END)::BIGINT AS active_cells_10km,
                SUM(CASE WHEN clean_det_count > 0
                    THEN mean_lat * clean_det_count ELSE 0.0 END)
                    / NULLIF(SUM(CASE WHEN clean_det_count > 0
                        THEN clean_det_count ELSE 0 END), 0) AS mean_lat,
                SUM(CASE WHEN clean_det_count > 0
                    THEN mean_lon * clean_det_count ELSE 0.0 END)
                    / NULLIF(SUM(CASE WHEN clean_det_count > 0
                        THEN clean_det_count ELSE 0 END), 0) AS mean_lon,
                SUM(CASE WHEN clean_det_count > 0
                    THEN x_m * clean_det_count ELSE 0.0 END)
                    / NULLIF(SUM(CASE WHEN clean_det_count > 0
                        THEN clean_det_count ELSE 0 END), 0) AS x_m_center,
                SUM(CASE WHEN clean_det_count > 0
                    THEN y_m * clean_det_count ELSE 0.0 END)
                    / NULLIF(SUM(CASE WHEN clean_det_count > 0
                        THEN clean_det_count ELSE 0 END), 0) AS y_m_center,
                SUM(nrt_det_count)::BIGINT AS nrt_det_count,
                SUM(sp_det_count)::BIGINT AS sp_det_count
            FROM gridded
            GROUP BY
                "year", "month", year_month, cell_ix_50km, cell_iy_50km, cell_id_50km,
                country_name, iso_a3, sensor_slot, is_continental_sa, is_in_sama
        )
        SELECT
            "year",
            "month",
            year_month,
            cell_ix_50km,
            cell_iy_50km,
            cell_id_50km,
            country_name,
            iso_a3,
            sensor_slot,
            is_continental_sa,
            is_in_sama,
            raw_all_det_count,
            clean_det_count,
            static_excluded_det_count,
            high_conf_count,
            night_det_count,
            day_det_count,
            active_days,
            overpass_event_count,
            frp_sum_mw,
            active_cells_10km,
            mean_lat,
            mean_lon,
            x_m_center,
            y_m_center,
            nrt_det_count,
            sp_det_count,
            (nrt_det_count > sp_det_count) AS is_nrt_majority,
            (("year" = 2026 AND "month" = 11)) AS is_partial_month
        FROM agg
    """


def build_cell50km_monthly(con, cell10km_path: Path, out_path: Path) -> None:
    """Re-agrega el cubo 10 km a una grilla de 50 km, año por año."""
    con.execute("SET preserve_insertion_order=false")
    source = parquet_scan_sql(cell10km_path)
    year_rows = con.execute(
        f'SELECT DISTINCT "year" FROM {source} WHERE "year" IS NOT NULL ORDER BY 1'
    ).fetchall()
    with _cube_scratch_dir("cell50km_monthly") as scratch:
        fragments: list[Path] = []
        for row in year_rows:
            year = int(row[0])
            fragment = scratch / f"year={year}.parquet"
            _write_parquet(
                con,
                _cell50km_monthly_query(f'{source} WHERE "year" = {year}'),
                fragment,
            )
            fragments.append(fragment)
        if not fragments:
            fragment = scratch / "year=empty.parquet"
            _write_parquet(con, _cell50km_monthly_query(f"{source} WHERE FALSE"), fragment)
            fragments.append(fragment)
        _merge_year_fragments(con, fragments, CELL50KM_MONTHLY_COLUMNS, out_path)


def _build_raw_admin1_monthly(
    con,
    parquet_glob: str | Path | Sequence[str | Path] | None = None,
    out_path: Path | None = None,
) -> None:
    """Pasada cruda ligera para ``raw_all_det_count`` y bboxes de Admin-1."""
    if parquet_glob is None or out_path is None:
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE raw_admin1_monthly AS "
            f"{_RAW_ADMIN1_MONTHLY_QUERY}"
        )
        return

    con.execute("SET preserve_insertion_order=false")
    _build_batched_parquet(
        con,
        parquet_glob=parquet_glob,
        static_cells_parquet=None,
        label="raw_admin1_monthly",
        query=_RAW_ADMIN1_MONTHLY_QUERY,
        columns=RAW_ADMIN1_MONTHLY_COLUMNS,
        out_path=out_path,
    )


def build_admin1_monthly(con, cell10km_path: Path, out_path: Path) -> None:
    """Cubo Admin-1 mensual con crudo y limpio en el mismo grain."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW clean_admin1_monthly AS
        SELECT
            "year",
            "month",
            year_month,
            country_name,
            iso_a3,
            admin1_name,
            sensor_slot,
            SUM(clean_det_count)::BIGINT AS clean_det_count,
            SUM(static_excluded_det_count)::BIGINT AS static_excluded_det_count,
            SUM(high_conf_count)::BIGINT AS high_conf_count,
            SUM(night_det_count)::BIGINT AS night_det_count,
            SUM(day_det_count)::BIGINT AS day_det_count,
            SUM(overpass_event_count)::BIGINT AS overpass_event_count,
            COALESCE(SUM(frp_sum_mw), 0.0) AS frp_sum_mw,
            COUNT(DISTINCT CASE WHEN clean_det_count > 0
                THEN cell_id_10km END)::BIGINT AS unique_active_cells_10km,
            SUM(active_days)::BIGINT AS active_cell_days,
            SUM(nrt_det_count)::BIGINT AS nrt_det_count,
            SUM(sp_det_count)::BIGINT AS sp_det_count,
            BOOL_OR(is_in_sama) AS intersects_sama
        FROM {parquet_scan_sql(cell10km_path)}
        GROUP BY
            "year", "month", year_month, country_name, iso_a3, admin1_name, sensor_slot
        """
    )
    query = """
        WITH combined AS (
            SELECT
                COALESCE(r."year", c."year") AS "year",
                COALESCE(r."month", c."month") AS "month",
                COALESCE(r.year_month, c.year_month) AS year_month,
                COALESCE(r.country_name, c.country_name) AS country_name,
                COALESCE(r.iso_a3, c.iso_a3) AS iso_a3,
                COALESCE(r.admin1_name, c.admin1_name) AS admin1_name,
                COALESCE(r.sensor_slot, c.sensor_slot) AS sensor_slot,
                COALESCE(r.raw_all_det_count, 0)::BIGINT AS raw_all_det_count,
                COALESCE(c.clean_det_count, 0)::BIGINT AS clean_det_count,
                COALESCE(c.static_excluded_det_count, 0)::BIGINT
                    AS static_excluded_det_count,
                COALESCE(c.high_conf_count, 0)::BIGINT AS high_conf_count,
                COALESCE(c.night_det_count, 0)::BIGINT AS night_det_count,
                COALESCE(c.day_det_count, 0)::BIGINT AS day_det_count,
                COALESCE(c.overpass_event_count, 0)::BIGINT AS overpass_event_count,
                COALESCE(c.frp_sum_mw, 0.0) AS frp_sum_mw,
                COALESCE(c.unique_active_cells_10km, 0)::BIGINT
                    AS unique_active_cells_10km,
                COALESCE(c.active_cell_days, 0)::BIGINT AS active_cell_days,
                COALESCE(c.nrt_det_count, 0)::BIGINT AS nrt_det_count,
                COALESCE(c.sp_det_count, 0)::BIGINT AS sp_det_count,
                COALESCE(r.intersects_sama, c.intersects_sama, FALSE)
                    AS intersects_sama
            FROM raw_admin1_monthly r
            FULL OUTER JOIN clean_admin1_monthly c
                ON r."year" IS NOT DISTINCT FROM c."year"
               AND r."month" IS NOT DISTINCT FROM c."month"
               AND r.year_month IS NOT DISTINCT FROM c.year_month
               AND r.country_name IS NOT DISTINCT FROM c.country_name
               AND r.iso_a3 IS NOT DISTINCT FROM c.iso_a3
               AND r.admin1_name IS NOT DISTINCT FROM c.admin1_name
               AND r.sensor_slot IS NOT DISTINCT FROM c.sensor_slot
        )
        SELECT
            "year",
            "month",
            year_month,
            country_name,
            iso_a3,
            admin1_name,
            sensor_slot,
            raw_all_det_count,
            clean_det_count,
            static_excluded_det_count,
            high_conf_count,
            night_det_count,
            day_det_count,
            overpass_event_count,
            frp_sum_mw,
            unique_active_cells_10km,
            active_cell_days,
            nrt_det_count,
            sp_det_count,
            (nrt_det_count > sp_det_count) AS is_nrt_majority,
            (("year" = 2026 AND "month" = 11)) AS is_partial_month,
            intersects_sama
        FROM combined
    """
    _write_parquet(con, query, out_path)


def build_country_monthly(con, admin1_path: Path, cell10km_path: Path, out_path: Path) -> None:
    """Cubo país mensual: sumas exactas de Admin-1 y celdas únicas directas."""
    query = f"""
        WITH admin AS (
            SELECT
                "year", "month", year_month, country_name, iso_a3, sensor_slot,
                clean_det_count, static_excluded_det_count, high_conf_count,
                night_det_count, day_det_count, overpass_event_count, frp_sum_mw,
                active_cell_days, nrt_det_count, sp_det_count, raw_all_det_count,
                intersects_sama
            FROM {parquet_scan_sql(admin1_path)}
        ),
        agg AS (
            SELECT
                "year",
                "month",
                year_month,
                country_name,
                iso_a3,
                sensor_slot,
                SUM(raw_all_det_count)::BIGINT AS raw_all_det_count,
                SUM(clean_det_count)::BIGINT AS clean_det_count,
                SUM(static_excluded_det_count)::BIGINT AS static_excluded_det_count,
                SUM(high_conf_count)::BIGINT AS high_conf_count,
                SUM(night_det_count)::BIGINT AS night_det_count,
                SUM(day_det_count)::BIGINT AS day_det_count,
                SUM(overpass_event_count)::BIGINT AS overpass_event_count,
                COALESCE(SUM(frp_sum_mw), 0.0) AS frp_sum_mw,
                SUM(active_cell_days)::BIGINT AS active_cell_days,
                SUM(nrt_det_count)::BIGINT AS nrt_det_count,
                SUM(sp_det_count)::BIGINT AS sp_det_count,
                BOOL_OR(intersects_sama) AS intersects_sama
            FROM admin
            GROUP BY "year", "month", year_month, country_name, iso_a3, sensor_slot
        ),
        cells AS (
            SELECT
                "year",
                "month",
                year_month,
                country_name,
                iso_a3,
                sensor_slot,
                COUNT(DISTINCT CASE WHEN clean_det_count > 0
                    THEN cell_id_10km END)::BIGINT AS unique_active_cells_10km
            FROM {parquet_scan_sql(cell10km_path)}
            GROUP BY "year", "month", year_month, country_name, iso_a3, sensor_slot
        )
        SELECT
            a."year",
            a."month",
            a.year_month,
            a.country_name,
            a.iso_a3,
            a.sensor_slot,
            a.raw_all_det_count,
            a.clean_det_count,
            a.static_excluded_det_count,
            a.high_conf_count,
            a.night_det_count,
            a.day_det_count,
            a.overpass_event_count,
            a.frp_sum_mw,
            COALESCE(c.unique_active_cells_10km, 0)::BIGINT AS unique_active_cells_10km,
            a.active_cell_days,
            a.nrt_det_count,
            a.sp_det_count,
            (a.nrt_det_count > a.sp_det_count) AS is_nrt_majority,
            ((a."year" = 2026 AND a."month" = 11)) AS is_partial_month,
            a.intersects_sama
        FROM agg a
        LEFT JOIN cells c
            ON a."year" IS NOT DISTINCT FROM c."year"
           AND a."month" IS NOT DISTINCT FROM c."month"
           AND a.year_month IS NOT DISTINCT FROM c.year_month
           AND a.country_name IS NOT DISTINCT FROM c.country_name
           AND a.iso_a3 IS NOT DISTINCT FROM c.iso_a3
           AND a.sensor_slot IS NOT DISTINCT FROM c.sensor_slot
    """
    _write_parquet(con, query, out_path)


def build_admin1_daily_season(
    con,
    out_path: Path,
    parquet_glob: str | Path | Sequence[str | Path] | None = None,
    static_cells_parquet: Path | None = None,
) -> None:
    """Serie diaria Admin-1 para curvas estacionales y detección temprana."""
    con.execute("SET preserve_insertion_order=false")
    if parquet_glob is None or static_cells_parquet is None:
        _write_parquet(con, _ADMIN1_DAILY_SEASON_QUERY, out_path)
        return

    _build_batched_parquet(
        con,
        parquet_glob=parquet_glob,
        static_cells_parquet=static_cells_parquet,
        label="admin1_daily_season",
        query=_ADMIN1_DAILY_SEASON_QUERY,
        columns=ADMIN1_DAILY_SEASON_COLUMNS,
        out_path=out_path,
    )


def build_geography_catalog(
    con,
    cell10km_path: Path,
    out_json: Path,
    *,
    parquet_glob: str,
) -> dict[str, Any]:
    """Catálogo de países/Admin-1 con bbox, SAMA y celdas activas históricas."""
    admin_rows = con.execute(
        """
        SELECT
            country_name,
            iso_a3,
            admin1_name,
            MIN(min_lat) AS min_lat,
            MAX(max_lat) AS max_lat,
            MIN(min_lon) AS min_lon,
            MAX(max_lon) AS max_lon,
            BOOL_OR(intersects_sama) AS intersects_sama
        FROM raw_admin1_monthly
        WHERE country_name IS NOT NULL
          AND min_lat IS NOT NULL AND min_lon IS NOT NULL
        GROUP BY country_name, iso_a3, admin1_name
        """
    ).fetchall()

    admin_cells = {
        (row[0], row[1], row[2]): int(row[3])
        for row in con.execute(
            f"""
            SELECT country_name, iso_a3, admin1_name,
                   COUNT(DISTINCT cell_id_10km)::BIGINT
            FROM {parquet_scan_sql(cell10km_path)}
            WHERE clean_det_count > 0 AND country_name IS NOT NULL
            GROUP BY country_name, iso_a3, admin1_name
            """
        ).fetchall()
    }
    country_cells = {
        (row[0], row[1]): int(row[2])
        for row in con.execute(
            f"""
            SELECT country_name, iso_a3, COUNT(DISTINCT cell_id_10km)::BIGINT
            FROM {parquet_scan_sql(cell10km_path)}
            WHERE clean_det_count > 0 AND country_name IS NOT NULL
            GROUP BY country_name, iso_a3
            """
        ).fetchall()
    }
    global_cells = int(
        con.execute(
            f"""
            SELECT COUNT(DISTINCT cell_id_10km)
            FROM {parquet_scan_sql(cell10km_path)}
            WHERE clean_det_count > 0
            """
        ).fetchone()[0]
        or 0
    )

    countries: dict[tuple[str, str | None], dict[str, Any]] = {}
    admin1_flat: list[dict[str, Any]] = []
    for row in admin_rows:
        country_name, iso_a3, admin1_name = row[0], row[1], row[2]
        bbox = [float(row[3]), float(row[4]), float(row[5]), float(row[6])]
        entry = {
            "country_name": country_name,
            "iso_a3": iso_a3,
            "admin1_name": admin1_name,
            "bbox": bbox,
            "intersects_sama": bool(row[7]),
            "historical_active_cells_10km": admin_cells.get(
                (country_name, iso_a3, admin1_name), 0
            ),
        }
        admin1_flat.append(entry)
        country = countries.setdefault(
            (country_name, iso_a3),
            {
                "country_name": country_name,
                "iso_a3": iso_a3,
                "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                "intersects_sama": False,
                "historical_active_cells_10km": country_cells.get(
                    (country_name, iso_a3), 0
                ),
                "admin1": [],
            },
        )
        country["bbox"][0] = min(country["bbox"][0], bbox[0])
        country["bbox"][1] = max(country["bbox"][1], bbox[1])
        country["bbox"][2] = min(country["bbox"][2], bbox[2])
        country["bbox"][3] = max(country["bbox"][3], bbox[3])
        country["intersects_sama"] = country["intersects_sama"] or bool(row[7])
        country["admin1"].append(entry)

    countries_list = sorted(
        countries.values(), key=lambda item: (item["country_name"], item["iso_a3"] or "")
    )
    for country in countries_list:
        country["admin1"].sort(key=lambda item: (item["admin1_name"] or ""))
        country["admin1_count"] = len(country["admin1"])
    admin1_flat.sort(
        key=lambda item: (
            item["country_name"] or "",
            item["iso_a3"] or "",
            item["admin1_name"] or "",
        )
    )
    catalog: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_parquet_glob": parquet_glob,
        "bbox_order": ["min_lat", "max_lat", "min_lon", "max_lon"],
        "countries": countries_list,
        "admin1": admin1_flat,
        "totals": {
            "countries": len(countries_list),
            "admin1_units": len(admin1_flat),
            "historical_active_cells_10km": global_cells,
        },
    }
    _write_json(out_json, catalog)
    return catalog


def build_cubes(
    project_root: str | Path | None = None,
    parquet_glob: str | Path | None = None,
    out_dir: str | Path | None = None,
    *,
    min_active_months: int = 48,
    min_active_years: int = 5,
    min_months_per_year: float = 9.0,
    min_type12_count: int = 5,
) -> dict[str, Any]:
    """Ejecuta el pipeline completo y devuelve rutas y resúmenes."""
    root = Path(project_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    if parquet_glob is None:
        glob_pattern = default_parquet_glob(root)
    else:
        glob_pattern = resolve_parquet_glob(str(parquet_glob))
    if not globlib.glob(glob_pattern, recursive=True) and not Path(glob_pattern).is_file():
        raise FileNotFoundError(f"No se encontraron Parquet para: {glob_pattern}")
    target_dir = Path(out_dir or (root / "data" / "processed")).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    paths = {name: target_dir / filename for name, filename in CUBE_NAMES.items()}
    raw_admin1_path = target_dir / "_raw_admin1_monthly.parquet"
    con = connect_duckdb()
    try:
        static_summary = identify_static_thermal_cells(
            con,
            glob_pattern,
            paths["static_cells"],
            summary_json=target_dir / "static_thermal_cells_summary.json",
            min_active_months=min_active_months,
            min_active_years=min_active_years,
            min_months_per_year=min_months_per_year,
            min_type12_count=min_type12_count,
        )
        qc_summary = compute_qc_summary(
            con,
            glob_pattern,
            paths["static_cells"],
            paths["qc_summary"],
        )

        build_cell10km_monthly(
            con,
            paths["cell10km_monthly"],
            parquet_glob=glob_pattern,
            static_cells_parquet=paths["static_cells"],
        )
        build_cell50km_monthly(con, paths["cell10km_monthly"], paths["cell50km_monthly"])
        _build_raw_admin1_monthly(con, glob_pattern, raw_admin1_path)
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW raw_admin1_monthly AS
            SELECT {_select_columns(RAW_ADMIN1_MONTHLY_COLUMNS)}
            FROM {parquet_scan_sql(raw_admin1_path)}
            """
        )
        build_admin1_monthly(con, paths["cell10km_monthly"], paths["admin1_monthly"])
        build_country_monthly(
            con,
            paths["admin1_monthly"],
            paths["cell10km_monthly"],
            paths["country_monthly"],
        )
        build_admin1_daily_season(
            con,
            paths["admin1_daily_season"],
            parquet_glob=glob_pattern,
            static_cells_parquet=paths["static_cells"],
        )
        catalog = build_geography_catalog(
            con,
            paths["cell10km_monthly"],
            paths["geography_catalog"],
            parquet_glob=glob_pattern,
        )

        row_counts = {}
        for name in (
            "cell10km_monthly",
            "cell50km_monthly",
            "admin1_monthly",
            "country_monthly",
            "admin1_daily_season",
        ):
            row_counts[name] = int(
                con.execute(
                    f"SELECT COUNT(*) FROM {parquet_scan_sql(paths[name])}"
                ).fetchone()[0]
            )
        return {
            "project_root": str(root),
            "parquet_glob": glob_pattern,
            "out_dir": str(target_dir),
            "static_summary": static_summary,
            "qc_summary": qc_summary,
            "catalog_totals": catalog["totals"],
            "row_counts": row_counts,
            "outputs": {name: str(path) for name, path in paths.items()},
        }
    finally:
        raw_admin1_path.unlink(missing_ok=True)
        con.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Construye cubos espaciotemporales MODIS/VIIRS armonizados."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Raíz del proyecto (default: raíz del repositorio).",
    )
    parser.add_argument(
        "--parquet-glob",
        type=str,
        default=None,
        help="Glob del Parquet canónico (default: data/parquet/south_america_hotspots).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directorio de salida (default: <project-root>/data/processed).",
    )
    parser.add_argument("--min-active-months", type=int, default=48)
    parser.add_argument("--min-active-years", type=int, default=5)
    parser.add_argument("--min-months-per-year", type=float, default=9.0)
    parser.add_argument(
        "--min-type12-count",
        type=int,
        default=5,
        help="Mínimo de filas type 1/2 para considerar dominancia de tipo 12.",
    )
    args = parser.parse_args(argv)

    summary = build_cubes(
        project_root=args.project_root,
        parquet_glob=args.parquet_glob,
        out_dir=args.out_dir,
        min_active_months=args.min_active_months,
        min_active_years=args.min_active_years,
        min_months_per_year=args.min_months_per_year,
        min_type12_count=args.min_type12_count,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
