#!/usr/bin/env python3
"""Export the deterministic NASA Space Apps submission CSV.

The contract reads ``harmonized_country_monthly.parquet`` through DuckDB and
writes one row per ``(iso_a3, year_month)`` with the exact columns declared in
``submission_spec.json``.  DuckDB is used strictly as a read-only query engine:
the Parquet artefact is opened with ``read_parquet`` and the processed
directory is never modified.

Usage from the repository root::

    python scripts/export_submission.py \
        --processed-dir data/processed --output submission.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

#: Exact public CSV contract, in order.
EXPORT_COLUMNS: tuple[str, ...] = (
    "record_id",
    "iso_a3",
    "country_name",
    "year_month",
    "year",
    "month",
    "harmonized_srehi",
    "harmonized_frp_mw",
    "raw_srehi",
    "raw_frp_mw",
    "clean_det_count",
    "raw_all_det_count",
    "data_maturity",
    "constellation_regime",
)

#: Canonical source file produced by the calibration stage.
SOURCE_FILENAME = "harmonized_country_monthly.parquet"


def _sql_literal(value: Path | str) -> str:
    """Return a single-quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def build_query(parquet_path: Path) -> str:
    """Build the deterministic projection and ordering for the public CSV."""
    return f"""
        SELECT
            iso_a3 || '_' || year_month AS record_id,
            iso_a3,
            COALESCE(country_name, 'UNKNOWN') AS country_name,
            year_month,
            CAST(year AS INTEGER) AS year,
            CAST(month AS INTEGER) AS month,
            CAST(harmonized_srehi AS DOUBLE) AS harmonized_srehi,
            CAST(harmonized_frp_mw AS DOUBLE) AS harmonized_frp_mw,
            CAST(overpass_event_count AS BIGINT) AS raw_srehi,
            CAST(raw_frp_mw AS DOUBLE) AS raw_frp_mw,
            CAST(clean_det_count AS BIGINT) AS clean_det_count,
            CAST(raw_all_det_count AS BIGINT) AS raw_all_det_count,
            COALESCE(data_maturity, '') AS data_maturity,
            COALESCE(constellation_regime, '') AS constellation_regime
        FROM read_parquet({_sql_literal(parquet_path)})
        ORDER BY iso_a3 ASC, year_month ASC
    """


def export(processed_dir: Path | str, output: Path | str) -> tuple[Path, int]:
    """Export the submission CSV and return ``(path, row_count)``.

    Raises ``FileNotFoundError`` when the country cube is absent, so callers
    fail before creating a partial delivery file.
    """
    source = (Path(processed_dir).expanduser() / SOURCE_FILENAME).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing harmonized country cube: {source}")

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    query = build_query(source)
    connection = duckdb.connect(database=":memory:")
    try:
        # No INSERT/COPY into the processed directory: the Parquet file is
        # consumed through a read-only scan and only the destination CSV is
        # written.  Explicit ordering makes bytes reproducible across hosts.
        row_count = int(connection.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0])
        copy_sql = (
            f"COPY ({query}) TO {_sql_literal(destination)} "
            "(FORMAT CSV, HEADER TRUE, DELIMITER ',', QUOTE '\"', "
            "ESCAPE '\"', NULL '')"
        )
        connection.execute(copy_sql)
    finally:
        connection.close()
    return destination, row_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing harmonized_country_monthly.parquet.",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Destination CSV path (default: submission.csv).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path, rows = export(args.processed_dir, args.output)
    except FileNotFoundError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    print(f"OK wrote {rows} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
