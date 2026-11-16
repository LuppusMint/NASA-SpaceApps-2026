#!/usr/bin/env python3
"""Congela la ventana de solapamiento MODIS -> VIIRS y sus folds espaciales.

El módulo construye ``overlap_cell50km_2012_2020.parquet`` y ``splits.json``
para el benchmark M0-M3. Ningún bloque espacial de 200 km puede aparecer en dos
folds: la asignación se calcula sobre celdas únicas y se propaga por
``cell_id_50km`` a todos los meses, regímenes y réplicas enmascaradas.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models import (  # noqa: E402
    PRE_DRIFT_END,
    PRE_DRIFT_START,
    REGIME_REAL,
    REGIME_TERRA_ONLY,
    connect_duckdb,
    fold_hashes,
    fold_sizes,
    id_hash,
    load_cell_month_features,
    parquet_sql,
    sha256_file,
    sha256_json,
    write_json,
    write_parquet,
)

BLOCK_SIZE_M = 200_000.0
FOLDS = 5
SEED = 42
OVERLAP_FILENAME = "overlap_cell50km_2012_2020.parquet"

#: Columns preserved in the matched overlap table.
OVERLAP_COLUMNS: tuple[str, ...] = (
    "year",
    "month",
    "year_month",
    "cell_id_50km",
    "cell_ix_50km",
    "cell_iy_50km",
    "country_name",
    "iso_a3",
    "is_continental_sa",
    "is_in_sama",
    "x_m_center",
    "y_m_center",
    "mean_lat",
    "mean_lon",
    "target_srehi",
    "target_frp_mw",
    "target_active_cells",
    "terra_frp_mw",
    "aqua_frp_mw",
    "modis_frp_mw",
    "terra_events",
    "aqua_events",
    "modis_events",
    "terra_active_cells",
    "aqua_active_cells",
    "modis_active_cells",
    "modis_clean_det",
    "modis_night_frac",
    "modis_high_conf_frac",
    "has_terra",
    "has_aqua",
    "month_sin",
    "month_cos",
    "abs_lat",
    "regime_mode",
    "sample_id",
    "fold_id",
)

_SPATIAL_KEYS = (
    "schema",
    "mode",
    "folds",
    "seed",
    "rows",
    "id_hash",
    "fold_sizes",
    "fold_hashes",
    "assignment",
    "block_size_m",
    "crs",
    "block_assignment",
)

#: Claves que ``tools/freeze_split.py`` usa para su huella estándar.
_FREEZE_KEYS = (
    "dataset",
    "dataset_sha256",
    "rows",
    "mode",
    "folds",
    "seed",
    "id_hash",
    "fold_sizes",
    "assignment",
)


def _safe_fraction(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Razón numérica segura (0 cuando el denominador es 0)."""
    values = numerator.astype(float) / denominator.replace(0, np.nan).astype(float)
    return values.fillna(0.0).clip(0.0, 1.0)


def build_overlap_table(
    processed_dir: Path, *, cube_path: Path | None = None
) -> pd.DataFrame:
    """Construye el emparejamiento MODIS/VIIRS y las réplicas ``TERRA_ONLY``.

    La tabla resultante contiene una fila real (``TERRA_AQUA``) y otra con las
    señales de Aqua enmascaradas a cero (``TERRA_ONLY``) por cada celda-mes. El
    objetivo (VIIRS-SNPP) es idéntico en ambas réplicas: así el clasificador
    Hurdle aprende ceros estructurales MODIS y el puente condiciona por
    ``has_aqua`` sin inventar señal de Aqua.
    """
    cube = cube_path or (processed_dir / "cube_cell50km_monthly.parquet")
    if not cube.is_file():
        raise FileNotFoundError(f"No existe el cubo de celdas 50 km: {cube}")

    con = connect_duckdb(scratch_dir=processed_dir / ".duckdb_tmp")
    try:
        features = load_cell_month_features(
            con, cube, start=PRE_DRIFT_START, end=PRE_DRIFT_END
        )
    finally:
        con.close()

    if features.empty:
        raise ValueError(
            f"El cubo {cube.name} no tiene filas entre {PRE_DRIFT_START} y {PRE_DRIFT_END}."
        )

    work = features.copy()
    work["target_srehi"] = work["snpp_events"].astype("int64")
    work["target_frp_mw"] = work["snpp_frp_mw"].astype(float)
    work["target_active_cells"] = work["snpp_active_cells"].astype("int64")

    # Ceros estructurales: conservamos actividad VIIRS o MODIS, nunca una fila vacía.
    work = work[
        (work["target_srehi"] > 0)
        | (work["target_frp_mw"] > 0)
        | (work["modis_events"] > 0)
        | (work["viirs_events"] > 0)
    ].copy()
    if work.empty:
        raise ValueError("La ventana pre-deriva no contiene actividad MODIS/VIIRS.")

    for column in ("country_name", "iso_a3"):
        work[column] = work[column].fillna("UNKNOWN").astype(str)
    work["is_continental_sa"] = work["is_continental_sa"].fillna(False).astype(bool)
    work["is_in_sama"] = work["is_in_sama"].fillna(False).astype(bool)
    for column in ("mean_lat", "mean_lon", "x_m_center", "y_m_center"):
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)

    # Fracciones MODIS calculadas antes de enmascarar Aqua.
    work["modis_night_frac"] = _safe_fraction(
        work["modis_night_det"], work["modis_clean_det"]
    )
    work["modis_high_conf_frac"] = _safe_fraction(
        work["modis_high_conf_det"], work["modis_clean_det"]
    )
    work["terra_night_frac"] = _safe_fraction(
        work["terra_night_det"], work["terra_clean_det"]
    )
    work["terra_high_conf_frac"] = _safe_fraction(
        work["terra_high_conf_det"], work["terra_clean_det"]
    )

    real = work.copy()
    real["regime_mode"] = REGIME_REAL
    real["sample_id"] = (
        real["cell_id_50km"].astype(str)
        + "|"
        + real["year_month"].astype(str)
        + "|"
        + REGIME_REAL
    )

    terra_only = work.copy()
    terra_only["regime_mode"] = REGIME_TERRA_ONLY
    terra_only["aqua_frp_mw"] = 0.0
    terra_only["aqua_events"] = 0
    terra_only["aqua_active_cells"] = 0
    terra_only["has_aqua"] = 0
    terra_only["modis_frp_mw"] = terra_only["terra_frp_mw"]
    terra_only["modis_events"] = terra_only["terra_events"]
    terra_only["modis_active_cells"] = terra_only["terra_active_cells"]
    terra_only["modis_clean_det"] = terra_only["terra_clean_det"]
    terra_only["modis_night_frac"] = terra_only["terra_night_frac"]
    terra_only["modis_high_conf_frac"] = terra_only["terra_high_conf_frac"]
    terra_only["sample_id"] = (
        terra_only["cell_id_50km"].astype(str)
        + "|"
        + terra_only["year_month"].astype(str)
        + "|"
        + REGIME_TERRA_ONLY
    )

    overlap = pd.concat([real, terra_only], ignore_index=True)
    overlap = overlap.drop_duplicates(subset=["sample_id"], keep="first")
    overlap = overlap[list(OVERLAP_COLUMNS[:-1])].copy()
    overlap["fold_id"] = -1

    for column in (
        "year",
        "month",
        "cell_ix_50km",
        "cell_iy_50km",
        "target_srehi",
        "target_active_cells",
        "terra_events",
        "aqua_events",
        "modis_events",
        "terra_active_cells",
        "aqua_active_cells",
        "modis_active_cells",
        "modis_clean_det",
        "has_terra",
        "has_aqua",
    ):
        overlap[column] = overlap[column].astype("int64")

    overlap = overlap.sort_values(["year_month", "cell_id_50km", "regime_mode"]).reset_index(
        drop=True
    )
    return overlap


def assign_folds(overlap: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Asigna folds por bloque espacial de 200 km y los propaga por celda."""
    from src.core.splits_st import assign_spatial_block_folds

    cells = (
        overlap[["cell_id_50km", "mean_lat", "mean_lon"]]
        .drop_duplicates(subset=["cell_id_50km"])
        .reset_index(drop=True)
    )
    cells["mean_lat"] = pd.to_numeric(cells["mean_lat"], errors="coerce").fillna(0.0)
    cells["mean_lon"] = pd.to_numeric(cells["mean_lon"], errors="coerce").fillna(0.0)

    spatial = assign_spatial_block_folds(
        cells,
        id_col="cell_id_50km",
        lat_col="mean_lat",
        lon_col="mean_lon",
        block_size_m=BLOCK_SIZE_M,
        folds=FOLDS,
        seed=SEED,
    )
    fold_by_cell = {
        str(cell): int(fold) for cell, fold in spatial["assignment"].items()
    }
    block_by_cell = {
        str(cell): str(block) for cell, block in spatial["block_assignment"].items()
    }

    overlap = overlap.copy()
    overlap["fold_id"] = overlap["cell_id_50km"].astype(str).map(fold_by_cell)
    if overlap["fold_id"].isna().any():
        missing = int(overlap["fold_id"].isna().sum())
        raise ValueError(f"{missing} filas sin fold espacial asignado.")
    overlap["fold_id"] = overlap["fold_id"].astype("int64")
    return overlap, {
        "fold_by_cell": fold_by_cell,
        "block_by_cell": block_by_cell,
        "crs": spatial.get("crs"),
        "n_cells": int(len(cells)),
    }


def build_temporal_folds(
    overlap: pd.DataFrame, *, max_rows: int = 100_000
) -> tuple[list[dict[str, Any]], bool]:
    """Genera folds espacio-temporales causales sobre el eje cronológico.

    Con tablas grandes se usa una muestra sistemática determinista para que
    ``splits.json`` no crezca de forma desmedida; los identificadores completos
    solo se conservan en tablas pequeñas.
    """
    from src.core.splits_st import assign_spatiotemporal_block_folds

    working = overlap[["sample_id", "year_month", "mean_lat", "mean_lon"]].copy()
    sampled = False
    if len(working) > int(max_rows):
        working = (
            working.sort_values(["year_month", "sample_id"])
            .iloc[:: max(1, len(working) // int(max_rows))]
            .head(int(max_rows))
            .reset_index(drop=True)
        )
        sampled = True

    folds = assign_spatiotemporal_block_folds(
        working,
        id_col="sample_id",
        time_col="year_month",
        lat_col="mean_lat",
        lon_col="mean_lon",
        block_size_m=BLOCK_SIZE_M,
        n_time_bins=4,
        n_space_folds=4,
        gap_days=0.0,
        seed=SEED,
    )
    compact = sampled or len(working) > 50_000
    payload: list[dict[str, Any]] = []
    for fold in folds:
        drop_keys = {"train_ids", "val_ids"}
        if compact:
            drop_keys.add("excluded_space_ids")
        item = {key: value for key, value in fold.items() if key not in drop_keys}
        if compact:
            item["train_ids_hash"] = id_hash(fold["train_ids"])
            item["val_ids_hash"] = id_hash(fold["val_ids"])
            item["excluded_space_ids_hash"] = id_hash(fold.get("excluded_space_ids", []))
        else:
            item["train_ids"] = fold["train_ids"]
            item["val_ids"] = fold["val_ids"]
        payload.append(item)
    return payload, sampled


def build_splits_payload(
    overlap: pd.DataFrame,
    *,
    dataset_path: Path,
    assignment_meta: dict[str, Any],
) -> dict[str, Any]:
    """Construye el ``splits.json`` canónico compatible con ``check_g2``."""
    assignment = {
        str(identifier): int(fold)
        for identifier, fold in zip(
            overlap["sample_id"].astype(str), overlap["fold_id"].astype(int)
        )
    }
    block_assignment = {
        str(identifier): str(assignment_meta["block_by_cell"][str(cell)])
        for identifier, cell in zip(
            overlap["sample_id"].astype(str), overlap["cell_id_50km"].astype(str)
        )
    }
    fold_by_cell = {
        str(cell): int(fold)
        for cell, fold in sorted(assignment_meta["fold_by_cell"].items())
    }
    temporal_folds, temporal_sampled = build_temporal_folds(overlap)

    payload: dict[str, Any] = {
        "schema": "st_split_v1",
        "mode": "spatial_block",
        "folds": int(FOLDS),
        "seed": int(SEED),
        "block_size_m": float(BLOCK_SIZE_M),
        "crs": str(assignment_meta["crs"]),
        "rows": int(len(assignment)),
        "id_hash": id_hash(assignment),
        "fold_sizes": fold_sizes(assignment, FOLDS),
        "fold_hashes": fold_hashes(assignment, FOLDS),
        "assignment": assignment,
        "block_assignment": block_assignment,
        "fold_by_cell": fold_by_cell,
        "n_cells": int(assignment_meta["n_cells"]),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "pre_drift_window": f"{PRE_DRIFT_START}..{PRE_DRIFT_END}",
        "targets": ["target_srehi", "target_frp_mw"],
        "regimes": [REGIME_REAL, REGIME_TERRA_ONLY],
        "temporal_folds": temporal_folds,
        "temporal_folds_sampled": bool(temporal_sampled),
        "leakage_control": {
            "id_col": "cell_id_50km",
            "propagation": "cell_id_50km",
            "block_size_m": float(BLOCK_SIZE_M),
        },
    }
    # Huella estándar de ``freeze_split`` (dataset + particiones) y huella
    # espacial del esquema ``st_split_v1``; ambas quedan auditables.
    payload["fingerprint"] = sha256_json(
        {key: payload[key] for key in _FREEZE_KEYS if key in payload}
    )
    payload["st_fingerprint"] = sha256_json(
        {key: payload[key] for key in _SPATIAL_KEYS if key in payload}
    )
    return payload


def freeze(
    processed_dir: Path | str,
    *,
    cube_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build the frozen overlap table and write ``splits.json`` into ``processed_dir``."""
    processed = Path(processed_dir).expanduser().resolve()
    processed.mkdir(parents=True, exist_ok=True)

    cube = Path(cube_path).expanduser().resolve() if cube_path else None
    overlap = build_overlap_table(processed, cube_path=cube)
    overlap, assignment_meta = assign_folds(overlap)

    dataset_path = processed / OVERLAP_FILENAME
    con = connect_duckdb(scratch_dir=processed / ".duckdb_tmp")
    try:
        con.register("overlap_df", overlap)
        write_parquet(
            con,
            f"SELECT * FROM overlap_df ORDER BY year_month, cell_id_50km, regime_mode",
            dataset_path,
        )
    finally:
        con.close()

    payload = build_splits_payload(
        overlap, dataset_path=dataset_path, assignment_meta=assignment_meta
    )
    write_json(processed / "splits.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing the quality-controlled cubes and destination of the overlap table.",
    )
    parser.add_argument(
        "--cube",
        default=None,
        help="Ruta alternativa a cube_cell50km_monthly.parquet.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = freeze(args.processed_dir, cube_path=args.cube)
    print(
        "splits.json: "
        f"{payload['rows']} filas, {payload['folds']} folds, "
        f"{payload['n_cells']} celdas, fingerprint={payload['fingerprint'][:12]}"
    )
    print(f"dataset_sha256={payload['dataset_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
