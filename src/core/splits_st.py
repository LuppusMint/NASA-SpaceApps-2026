#!/usr/bin/env python3
"""Frozen spatio-temporal partitions with buffering and object grouping, without leakage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.core.geogrid import project_points_equal_area

ST_SCHEMA = "st_split_v1"
_COMMON_KEYS = (
    "schema",
    "mode",
    "folds",
    "seed",
    "rows",
    "id_hash",
    "fold_sizes",
    "fold_hashes",
    "assignment",
)
_SPATIAL_KEYS = _COMMON_KEYS + ("block_size_m", "crs", "block_assignment")
_OBJECT_KEYS = _COMMON_KEYS + ("object_id_col", "group_assignment")
ST_FINGERPRINT_KEYS = tuple(dict.fromkeys(_SPATIAL_KEYS + _OBJECT_KEYS))


def _stable_json(value: Any) -> str:
    """Serializes canonically for hashing."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"Non-serializable type: {type(value)!r}")


def _fingerprint(payload: dict[str, Any], keys: Sequence[str]) -> str:
    subset = {key: payload[key] for key in keys if key in payload}
    return hashlib.sha256(_stable_json(subset).encode("utf-8")).hexdigest()


def _ids(df: pd.DataFrame, id_col: str) -> np.ndarray:
    if id_col not in df.columns:
        raise ValueError(f"Missing ID column {id_col!r}.")
    ids = df[id_col].astype(str).to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise ValueError("The ID column contains duplicates.")
    return ids


def _fold_hashes(assignment: dict[str, int], folds: int) -> dict[str, str]:
    by_fold: dict[int, list[str]] = {fold: [] for fold in range(int(folds))}
    for identifier, fold in assignment.items():
        by_fold.setdefault(int(fold), []).append(str(identifier))
    return {
        str(fold): hashlib.sha256("\n".join(sorted(by_fold.get(fold, []))).encode("utf-8")).hexdigest()
        for fold in range(int(folds))
    }


def _fold_sizes(assignment: dict[str, int], folds: int) -> dict[str, int]:
    sizes = {str(fold): 0 for fold in range(int(folds))}
    for fold in assignment.values():
        sizes[str(int(fold))] = sizes.get(str(int(fold)), 0) + 1
    return sizes


def _ranked(values: Iterable[str], seed: int) -> list[str]:
    return sorted(values, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest())


def _block_ids(x_m: np.ndarray, y_m: np.ndarray, block_size_m: float) -> np.ndarray:
    ix = np.floor(x_m / float(block_size_m)).astype(np.int64)
    iy = np.floor(y_m / float(block_size_m)).astype(np.int64)
    return np.asarray([f"{int(a)}_{int(b)}" for a, b in zip(ix, iy)], dtype=object)


def _assign_blocks(block_ids: np.ndarray, folds: int, seed: int) -> dict[str, int]:
    unique = sorted(set(str(block) for block in block_ids))
    ordered = _ranked(unique, seed)
    return {block: index % int(folds) for index, block in enumerate(ordered)}


def assign_spatial_block_folds(
    df: pd.DataFrame,
    *,
    id_col: str = "sample_id",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    block_size_m: float = 50_000.0,
    folds: int = 5,
    seed: int = 42,
    crs: str | None = None,
) -> dict[str, Any]:
    """Assigns complete equal-area spatial blocks to deterministic folds.

    Blocks are ordered by ``SHA-256(seed:block_id)`` and distributed
    round-robin, so no coordinate from a block falls into two folds.
    """
    if int(folds) < 2:
        raise ValueError("folds must be >= 2.")
    if not np.isfinite(block_size_m) or block_size_m <= 0:
        raise ValueError("block_size_m must be positive.")
    for col in (id_col, lat_col, lon_col):
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r}.")

    ids = _ids(df, id_col)
    x_m, y_m, resolved_crs = project_points_equal_area(
        df[lon_col].to_numpy(), df[lat_col].to_numpy(), crs=crs
    )
    block_ids = _block_ids(x_m, y_m, block_size_m)
    fold_of_block = _assign_blocks(block_ids, int(folds), int(seed))
    assignment = {identifier: int(fold_of_block[block]) for identifier, block in zip(ids, block_ids)}
    block_assignment = {identifier: str(block) for identifier, block in zip(ids, block_ids)}

    ordered_ids = sorted(assignment)
    assignment = {identifier: assignment[identifier] for identifier in ordered_ids}
    block_assignment = {identifier: block_assignment[identifier] for identifier in ordered_ids}
    payload: dict[str, Any] = {
        "schema": ST_SCHEMA,
        "mode": "spatial_block",
        "folds": int(folds),
        "seed": int(seed),
        "block_size_m": float(block_size_m),
        "crs": str(resolved_crs),
        "rows": int(len(df)),
        "id_hash": hashlib.sha256("\n".join(ordered_ids).encode("utf-8")).hexdigest(),
        "fold_sizes": _fold_sizes(assignment, int(folds)),
        "fold_hashes": _fold_hashes(assignment, int(folds)),
        "assignment": assignment,
        "block_assignment": block_assignment,
    }
    payload["fingerprint"] = _fingerprint(payload, _SPATIAL_KEYS)
    return payload


def build_buffered_spatial_folds(
    df: pd.DataFrame,
    *,
    id_col: str = "sample_id",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    block_size_m: float = 50_000.0,
    buffer_m: float = 25_000.0,
    folds: int = 5,
    seed: int = 42,
    crs: str | None = None,
) -> list[dict[str, Any]]:
    """Builds spatial folds with a Euclidean buffer around the validation set."""
    if not np.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("buffer_m must be >= 0.")
    payload = assign_spatial_block_folds(
        df,
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        block_size_m=block_size_m,
        folds=folds,
        seed=seed,
        crs=crs,
    )
    x_m, y_m, _ = project_points_equal_area(
        df[lon_col].to_numpy(), df[lat_col].to_numpy(), crs=payload["crs"]
    )
    ids = np.asarray([str(value) for value in df[id_col].to_numpy()], dtype=object)
    assignment = payload["assignment"]
    fold_values = np.asarray([int(assignment[str(identifier)]) for identifier in ids], dtype=np.int64)

    from scipy.spatial import cKDTree

    result: list[dict[str, Any]] = []
    for fold in range(int(folds)):
        val_mask = fold_values == fold
        train_mask = ~val_mask
        val_ids = ids[val_mask]
        candidate_ids = ids[train_mask]
        candidate_xy = np.column_stack([x_m[train_mask], y_m[train_mask]])
        val_xy = np.column_stack([x_m[val_mask], y_m[val_mask]])

        if len(val_xy) and len(candidate_xy):
            tree = cKDTree(val_xy)
            distances, _ = tree.query(candidate_xy, k=1)
            excluded_mask = distances < float(buffer_m)
            excluded_ids = candidate_ids[excluded_mask]
            kept_mask = ~excluded_mask
            train_ids = candidate_ids[kept_mask]
            min_distance = float(np.min(distances[kept_mask])) if np.any(kept_mask) else float("inf")
        elif len(val_xy) and not len(candidate_xy):
            excluded_ids = np.asarray([], dtype=object)
            train_ids = np.asarray([], dtype=object)
            min_distance = float("inf")
        else:
            excluded_ids = np.asarray([], dtype=object)
            train_ids = candidate_ids
            min_distance = float("inf")

        result.append(
            {
                "fold": int(fold),
                "train_ids": [str(value) for value in train_ids],
                "val_ids": [str(value) for value in val_ids],
                "excluded_buffer_ids": [str(value) for value in excluded_ids],
                "min_train_val_distance_m": float(min_distance),
                "buffer_m": float(buffer_m),
                "block_size_m": float(block_size_m),
                "crs": payload["crs"],
                "n_train": int(len(train_ids)),
                "n_val": int(len(val_ids)),
                "n_excluded_buffer": int(len(excluded_ids)),
            }
        )
    return result


def assign_spatiotemporal_block_folds(
    df: pd.DataFrame,
    *,
    id_col: str = "sample_id",
    time_col: str = "acq_date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    block_size_m: float = 50_000.0,
    n_time_bins: int = 4,
    n_space_folds: int = 4,
    gap_days: float = 0.0,
    crs: str | None = None,
    seed: int = 42,
    exclude_val_space: bool = True,
) -> list[dict[str, Any]]:
    """Generates ``(time block, space block)`` folds with causal history.

    The first time block is reserved as initial history; evaluation traverses
    ``t=1..n_time_bins-1``. Training uses times strictly before the validation
    minimum when ``gap_days == 0``, or ``<= min(val_time) - gap_days``
    otherwise, and optionally excludes the same validation space block.
    """
    for col in (id_col, time_col, lat_col, lon_col):
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r}.")
    if int(n_time_bins) < 2:
        raise ValueError("n_time_bins must be >= 2.")
    if int(n_space_folds) < 1:
        raise ValueError("n_space_folds must be >= 1.")
    if not np.isfinite(gap_days) or gap_days < 0:
        raise ValueError("gap_days must be >= 0.")
    if not np.isfinite(block_size_m) or block_size_m <= 0:
        raise ValueError("block_size_m must be positive.")

    ids = _ids(df, id_col)
    times = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    if bool(times.isna().any()):
        raise ValueError("time_col contains null or unparseable dates.")

    dense_rank = times.rank(method="dense").to_numpy(dtype=np.int64)
    n_unique_times = int(dense_rank.max()) if len(dense_rank) else 0
    time_bins = np.minimum(
        int(n_time_bins) - 1,
        (dense_rank - 1) * int(n_time_bins) // max(n_unique_times, 1),
    )

    x_m, y_m, resolved_crs = project_points_equal_area(
        df[lon_col].to_numpy(), df[lat_col].to_numpy(), crs=crs
    )
    block_ids = _block_ids(x_m, y_m, block_size_m)
    fold_of_block = _assign_blocks(block_ids, int(n_space_folds), int(seed))
    space_folds = np.asarray([int(fold_of_block[str(block)]) for block in block_ids], dtype=np.int64)
    time_values = times.to_numpy()

    folds: list[dict[str, Any]] = []
    fold_index = 0
    for t_idx in range(1, int(n_time_bins)):
        in_time = time_bins == t_idx
        for s_idx in range(int(n_space_folds)):
            val_mask = in_time & (space_folds == s_idx)
            val_times = time_values[val_mask]
            min_val_time = pd.Timestamp(val_times.min()) if val_times.size else None
            max_val_time = pd.Timestamp(val_times.max()) if val_times.size else None
            if min_val_time is None:
                eligible = np.zeros(len(df), dtype=bool)
            elif float(gap_days) == 0.0:
                # Strict zero gap: the minimum validation instant never enters train.
                eligible = times < min_val_time
            else:
                eligible = times <= min_val_time - pd.Timedelta(days=float(gap_days))
            train_mask = eligible & (~val_mask)
            excluded_space_mask = np.zeros(len(df), dtype=bool)
            if exclude_val_space:
                excluded_space_mask = eligible & (space_folds == s_idx) & (~val_mask)
                train_mask = train_mask & (~excluded_space_mask)

            val_ids = ids[val_mask]
            train_ids = ids[train_mask]
            min_train_time = pd.Timestamp(time_values[train_mask].min()) if np.any(train_mask) else None
            max_train_time = pd.Timestamp(time_values[train_mask].max()) if np.any(train_mask) else None
            folds.append(
                {
                    "fold": int(fold_index),
                    "fold_id": f"t{int(t_idx)}_s{int(s_idx)}",
                    "t_idx": int(t_idx),
                    "s_idx": int(s_idx),
                    "train_ids": [str(value) for value in train_ids],
                    "val_ids": [str(value) for value in val_ids],
                    "excluded_space_ids": [
                        str(value) for value in ids[excluded_space_mask]
                    ],
                    "gap_days": float(gap_days),
                    "exclude_val_space": bool(exclude_val_space),
                    "min_val_time": min_val_time.isoformat() if min_val_time is not None else None,
                    "max_val_time": max_val_time.isoformat() if max_val_time is not None else None,
                    "min_train_time": min_train_time.isoformat() if min_train_time is not None else None,
                    "max_train_time": max_train_time.isoformat() if max_train_time is not None else None,
                    "n_train": int(len(train_ids)),
                    "n_val": int(len(val_ids)),
                    "n_time_bins": int(n_time_bins),
                    "n_space_folds": int(n_space_folds),
                    "crs": str(resolved_crs),
                }
            )
            fold_index += 1
    return folds


def propagate_object_group_folds(
    df: pd.DataFrame,
    *,
    id_col: str = "sample_id",
    object_id_col: str = "object_id",
    folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Assigns all records of an object to the same fold."""
    if int(folds) < 2:
        raise ValueError("folds must be >= 2.")
    for col in (id_col, object_id_col):
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r}.")
    ids = _ids(df, id_col)
    object_ids = df[object_id_col].astype(str).to_numpy()
    if bool(pd.isna(df[object_id_col]).any()):
        raise ValueError("object_id_col contains null values.")

    unique_objects = sorted(set(str(value) for value in object_ids))
    ordered = _ranked(unique_objects, int(seed))
    object_fold = {obj: index % int(folds) for index, obj in enumerate(ordered)}
    assignment = {identifier: int(object_fold[obj]) for identifier, obj in zip(ids, object_ids)}
    group_assignment = {obj: int(object_fold[obj]) for obj in unique_objects}

    ordered_ids = sorted(assignment)
    assignment = {identifier: assignment[identifier] for identifier in ordered_ids}
    payload: dict[str, Any] = {
        "schema": ST_SCHEMA,
        "mode": "object_group",
        "folds": int(folds),
        "seed": int(seed),
        "rows": int(len(df)),
        "id_hash": hashlib.sha256("\n".join(ordered_ids).encode("utf-8")).hexdigest(),
        "fold_sizes": _fold_sizes(assignment, int(folds)),
        "fold_hashes": _fold_hashes(assignment, int(folds)),
        "assignment": assignment,
        "object_id_col": str(object_id_col),
        "group_assignment": group_assignment,
    }
    payload["fingerprint"] = _fingerprint(payload, _OBJECT_KEYS)
    return payload


def _as_id_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        return [str(values)]
    if isinstance(values, np.ndarray):
        return [str(value) for value in values.ravel().tolist()]
    return [str(value) for value in values]


def _as_id_set(values: Any) -> set[str]:
    return set(_as_id_list(values))


def _verify_payload(df: pd.DataFrame, payload: dict[str, Any], **kwargs: Any) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != ST_SCHEMA:
        return [f"Invalid schema: {payload.get('schema')!r}; expected {ST_SCHEMA!r}."]
    mode = payload.get("mode")
    if mode not in {"spatial_block", "object_group"}:
        return [f"Unverifiable payload mode: {mode!r}."]

    keys = _SPATIAL_KEYS if mode == "spatial_block" else _OBJECT_KEYS
    missing = [key for key in keys if key not in payload]
    if missing:
        return [f"Missing payload fields: {', '.join(missing)}."]
    expected = _fingerprint(payload, keys)
    if expected != payload.get("fingerprint"):
        errors.append("The payload SHA-256 fingerprint does not match: it was modified.")

    id_col = kwargs.get("id_col") or "sample_id"
    if id_col not in df.columns:
        return errors + [f"Missing ID column {id_col!r}."]
    df_ids = [str(value) for value in df[id_col].to_numpy()]
    if len(df_ids) != len(set(df_ids)):
        errors.append("The DataFrame ID column contains duplicates.")
    assignment = payload.get("assignment") or {}
    if not isinstance(assignment, dict):
        return errors + ["assignment must be an id -> fold dictionary."]
    assigned_ids = set(str(key) for key in assignment)
    expected_ids = set(df_ids)
    if assigned_ids != expected_ids:
        errors.append("The set of assigned IDs does not match the DataFrame.")
    folds = int(payload.get("folds") or 0)
    try:
        numeric_assignment = {str(key): int(value) for key, value in assignment.items()}
    except (TypeError, ValueError):
        return errors + ["assignment contains non-integer folds."]
    invalid_folds = [fold for fold in numeric_assignment.values() if not (0 <= fold < folds)]
    if invalid_folds:
        errors.append("Some folds fall outside the declared range.")
    if _fold_sizes(numeric_assignment, folds) != payload.get("fold_sizes"):
        errors.append("fold_sizes does not match assignment.")
    if _fold_hashes(numeric_assignment, folds) != payload.get("fold_hashes"):
        errors.append("fold_hashes does not match assignment.")
    expected_id_hash = hashlib.sha256("\n".join(sorted(assigned_ids)).encode("utf-8")).hexdigest()
    if expected_id_hash != payload.get("id_hash"):
        errors.append("id_hash does not match the assigned IDs.")

    if mode == "spatial_block":
        block_assignment = payload.get("block_assignment") or {}
        if not isinstance(block_assignment, dict):
            errors.append("block_assignment must be an id -> block_id dictionary.")
        else:
            if set(str(key) for key in block_assignment) != assigned_ids:
                errors.append("block_assignment does not cover exactly the assigned IDs.")
            folds_by_block: dict[str, set[int]] = {}
            for identifier, fold in assignment.items():
                block = block_assignment.get(identifier)
                if block is None:
                    continue
                folds_by_block.setdefault(str(block), set()).add(int(fold))
            leaked = [block for block, block_folds in folds_by_block.items() if len(block_folds) > 1]
            if leaked:
                errors.append(f"Spatial leakage: {len(leaked)} blocks appear in more than one fold.")
    else:
        object_col = kwargs.get("object_id_col") or payload.get("object_id_col") or "object_id"
        if object_col not in df.columns:
            errors.append(f"Missing object column {object_col!r}.")
        else:
            folds_by_object: dict[str, set[int]] = {}
            for identifier, object_id in zip(df[id_col].astype(str), df[object_col].astype(str)):
                fold = assignment.get(identifier)
                if fold is not None:
                    folds_by_object.setdefault(str(object_id), set()).add(int(fold))
            leaked = [obj for obj, obj_folds in folds_by_object.items() if len(obj_folds) > 1]
            if leaked:
                errors.append(f"Object leakage: {len(leaked)} objects appear in more than one fold.")
            group_assignment = payload.get("group_assignment")
            if isinstance(group_assignment, dict):
                object_folds: dict[str, set[int]] = {}
                for identifier, object_id in zip(df[id_col].astype(str), df[object_col].astype(str)):
                    if identifier in numeric_assignment:
                        object_folds.setdefault(str(object_id), set()).add(numeric_assignment[identifier])
                inconsistent = [
                    obj
                    for obj, fold in group_assignment.items()
                    if object_folds.get(str(obj), set()) != {int(fold)}
                ]
                if inconsistent:
                    errors.append("group_assignment is inconsistent with assignment.")
    return errors


def _lookup_times(df: pd.DataFrame, time_col: str | None, ids: set[str], id_col: str) -> pd.Series:
    if time_col is None or time_col not in df.columns:
        return pd.Series(dtype="datetime64[ns, UTC]")
    selected = df[df[id_col].astype(str).isin(ids)]
    return pd.to_datetime(selected[time_col], errors="coerce", utc=True)


def verify_st_splits_payload(
    df: pd.DataFrame,
    payload_or_folds: Any,
    *,
    id_col: str = "sample_id",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    time_col: str | None = None,
    object_id_col: str | None = None,
) -> list[str]:
    """Verifies fingerprints, absence of leakage, and buffer/temporal separation.

    Returns a list of readable errors; an empty list means the split is valid.
    Accepts frozen payloads and in-memory fold lists.
    """
    if isinstance(payload_or_folds, (str, Path)):
        payload_or_folds = load_st_splits(Path(payload_or_folds))
    if isinstance(payload_or_folds, dict):
        return _verify_payload(df, payload_or_folds, id_col=id_col, object_id_col=object_id_col)
    if not isinstance(payload_or_folds, (list, tuple)):
        return ["The payload must be a dictionary, a list of folds, or a JSON path."]
    if id_col not in df.columns:
        return [f"Missing ID column {id_col!r}."]

    errors: list[str] = []
    df_ids = [str(value) for value in df[id_col].to_numpy()]
    known_ids = set(df_ids)
    if len(df_ids) != len(known_ids):
        errors.append("The DataFrame ID column contains duplicates.")

    use_coords = lat_col in df.columns and lon_col in df.columns
    coord_cache: dict[str | None, dict[str, tuple[float, float]]] = {}

    def _coords_for(crs: str | None) -> dict[str, tuple[float, float]]:
        if not use_coords or not len(df):
            return {}
        if crs not in coord_cache:
            x_m, y_m, _ = project_points_equal_area(
                df[lon_col].to_numpy(), df[lat_col].to_numpy(), crs=crs
            )
            coord_cache[crs] = {
                str(identifier): (float(x), float(y))
                for identifier, x, y in zip(df[id_col].to_numpy(), x_m, y_m)
            }
        return coord_cache[crs]

    for position, fold in enumerate(payload_or_folds):
        if not isinstance(fold, dict):
            errors.append(f"Fold {position}: invalid structure.")
            continue
        train_list = _as_id_list(fold.get("train_ids"))
        val_list = _as_id_list(fold.get("val_ids"))
        train = set(train_list)
        val = set(val_list)
        if len(train) != len(train_list):
            errors.append(f"Fold {position}: duplicate train IDs.")
        if len(val) != len(val_list):
            errors.append(f"Fold {position}: duplicate val IDs.")
        unknown = (train | val) - known_ids
        if unknown:
            errors.append(f"Fold {position}: {len(unknown)} IDs do not exist in the DataFrame.")
        overlap = train & val
        if overlap:
            errors.append(f"Fold {position}: {len(overlap)} IDs overlap between train and val.")

        excluded_raw = fold.get("excluded_buffer_ids")
        if excluded_raw is None:
            excluded_raw = fold.get("excluded_space_ids")
        excluded = _as_id_set(excluded_raw)
        if excluded & train:
            errors.append(f"Fold {position}: IDs excluded by buffer/space reappear in train.")
        if excluded & val:
            errors.append(f"Fold {position}: IDs excluded by buffer/space appear in val.")

        buffer_m = fold.get("buffer_m")
        reported_distance = fold.get("min_train_val_distance_m")
        if buffer_m is not None:
            limit = float(buffer_m) - 1e-6
            if reported_distance is not None and np.isfinite(float(reported_distance)):
                if float(reported_distance) < limit:
                    errors.append(
                        f"Fold {position}: min_train_val_distance_m={float(reported_distance):.3f} "
                        f"< buffer_m={float(buffer_m):.3f}."
                    )
            coord_by_id = _coords_for(fold.get("crs"))
            if train and val and coord_by_id and set(train).issubset(coord_by_id) and set(val).issubset(coord_by_id):
                from scipy.spatial import cKDTree

                train_xy = np.asarray([coord_by_id[key] for key in sorted(train)], dtype=float)
                val_xy = np.asarray([coord_by_id[key] for key in sorted(val)], dtype=float)
                actual_min = float(np.min(cKDTree(val_xy).query(train_xy, k=1)[0]))
                if actual_min < limit:
                    errors.append(
                        f"Fold {position}: actual train-val distance {actual_min:.3f} m "
                        f"< buffer_m={float(buffer_m):.3f}."
                    )
        elif reported_distance is not None and np.isfinite(float(reported_distance)):
            # Without a declared buffer there is no threshold to compare against.
            pass

        gap_days = fold.get("gap_days", fold.get("gap"))
        if gap_days is not None:
            gap = pd.Timedelta(days=float(gap_days))
            checked = False
            if time_col and time_col in df.columns and train and val:
                train_times = _lookup_times(df, time_col, train, id_col)
                val_times = _lookup_times(df, time_col, val, id_col)
                train_times = train_times.dropna()
                val_times = val_times.dropna()
                if len(train_times) and len(val_times):
                    checked = True
                    if train_times.max() > val_times.min() - gap:
                        errors.append(
                            f"Fold {position}: temporal leakage; max(train)={train_times.max()} "
                            f"> min(val)-gap={val_times.min() - gap}."
                        )
            if not checked:
                max_train = fold.get("max_train_time")
                min_val = fold.get("min_val_time")
                if max_train is not None and min_val is not None:
                    max_train_ts = pd.Timestamp(max_train)
                    min_val_ts = pd.Timestamp(min_val)
                    if max_train_ts > min_val_ts - gap:
                        errors.append(f"Fold {position}: temporal metadata violates the gap.")
    return errors


def save_st_splits(payload: Any, out_path: Path) -> Path:
    """Writes a payload (or list of folds) as deterministic JSON."""
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def load_st_splits(path: Path) -> Any:
    """Loads a JSON payload of spatio-temporal partitions."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Splits not found: {source}")
    return json.loads(source.read_text(encoding="utf-8"))
