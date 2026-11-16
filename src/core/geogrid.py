#!/usr/bin/env python3
"""Equal-area geospatial grid, polygon clipping, and overpass events."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

VECTOR_SUFFIXES = (".geojson", ".gpkg", ".shp", ".json")


def _finite_vector(values: Iterable[float], name: str) -> np.ndarray:
    """Converts to a 1-D float array and requires finite values."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _check_lat_lon(lat: np.ndarray, lon: np.ndarray) -> None:
    """Validates WGS84 geographic ranges."""
    if lat.size and (np.any(lat < -90.0) or np.any(lat > 90.0)):
        raise ValueError("lat outside the range [-90, 90].")
    if lon.size and (np.any(lon < -180.0) or np.any(lon > 180.0)):
        raise ValueError("lon outside the range [-180, 180].")
    if lat.shape != lon.shape:
        raise ValueError("lat and lon must have the same length.")


def default_equal_area_crs(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> str:
    """Builds a PROJ Albers Equal-Area string fitted to the bounding box.

    The standard parallels are placed inside the latitude range to preserve
    area; if the range is degenerate, a minimum one-degree opening is used.
    """
    vals = np.asarray([lat_min, lat_max, lon_min, lon_max], dtype=float)
    if not np.all(np.isfinite(vals)):
        raise ValueError("The bounding box contains non-finite values.")
    if -90.0 > lat_min or lat_min > 90.0 or -90.0 > lat_max or lat_max > 90.0:
        raise ValueError("Latitudes outside the range [-90, 90].")
    if -180.0 > lon_min or lon_min > 180.0 or -180.0 > lon_max or lon_max > 180.0:
        raise ValueError("Longitudes outside the range [-180, 180].")
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min

    lat_0 = 0.5 * (lat_min + lat_max)
    lon_0 = 0.5 * (lon_min + lon_max)
    span = lat_max - lat_min
    if span < 1e-9:
        # No latitude extent: open one degree around the center without touching the poles.
        half = 1.0
        if lat_0 + half > 89.9:
            half = max(0.25, 89.9 - lat_0)
        if lat_0 - half < -89.9:
            half = max(0.25, lat_0 + 89.9)
        lat_1, lat_2 = lat_0 - half, lat_0 + half
    else:
        lat_1 = lat_min + span / 6.0
        lat_2 = lat_max - span / 6.0
        if abs(lat_2 - lat_1) < 1e-9:
            lat_1 = lat_min + span / 4.0
            lat_2 = lat_min + 3.0 * span / 4.0
    lat_1 = float(np.clip(lat_1, -89.9, 89.9))
    lat_2 = float(np.clip(lat_2, -89.9, 89.9))
    if abs(lat_2 - lat_1) < 1e-9:
        lat_1 = float(np.clip(lat_0 - 0.5, -89.9, 89.9))
        lat_2 = float(np.clip(lat_0 + 0.5, -89.9, 89.9))
    # PROJ requires |lat_1 + lat_2| > 0; avoid parallels symmetric about the equator.
    if abs(lat_1 + lat_2) < 1e-6:
        if lat_0 >= 0.0:
            lat_1 = float(np.clip(min(lat_2 - 0.5, -0.5), -89.9, 89.9))
            lat_2 = float(np.clip(max(lat_2, 0.5), -89.9, 89.9))
        else:
            lat_2 = float(np.clip(max(lat_1 + 0.5, 0.5), -89.9, 89.9))
            lat_1 = float(np.clip(min(lat_1, -0.5), -89.9, 89.9))
    return (
        f"+proj=aea +lat_0={lat_0:.10g} +lon_0={lon_0:.10g} "
        f"+lat_1={lat_1:.10g} +lat_2={lat_2:.10g} "
        "+datum=WGS84 +units=m"
    )


def project_points_equal_area(
    lon: Sequence[float] | np.ndarray,
    lat: Sequence[float] | np.ndarray,
    crs: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Projects WGS84 ``(lon, lat)`` to equal-area metres ``(x_m, y_m)``.

    If ``crs`` is ``None``, an Albers projection fitted to the points'
    bounding box is built. The resolved PROJ string is also returned.
    """
    from pyproj import Transformer

    lon_arr = _finite_vector(lon, "lon")
    lat_arr = _finite_vector(lat, "lat")
    if lon_arr.size != lat_arr.size:
        raise ValueError("lon and lat must have the same length.")
    _check_lat_lon(lat_arr, lon_arr)

    if crs is None:
        if lon_arr.size:
            crs = default_equal_area_crs(
                float(np.min(lat_arr)),
                float(np.max(lat_arr)),
                float(np.min(lon_arr)),
                float(np.max(lon_arr)),
            )
        else:
            crs = default_equal_area_crs(0.0, 0.0, 0.0, 0.0)
    if not isinstance(crs, str) or not crs.strip():
        raise ValueError("crs must be a non-empty PROJ/EPSG string.")

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_m, y_m = transformer.transform(lon_arr, lat_arr)
    x_arr = np.asarray(x_m, dtype=float).ravel()
    y_arr = np.asarray(y_m, dtype=float).ravel()
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        raise ValueError("The projection produced non-finite coordinates.")
    return x_arr, y_arr, crs


def snap_to_equal_area_grid(
    df: pd.DataFrame,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    cell_size_m: float = 10_000.0,
    crs: str | None = None,
    x_origin: float = 0.0,
    y_origin: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Snaps each point to an equal-area cell and adds cell centroids and area.

    The cell is indexed with ``floor((x - origin) / size)`` in the equal-area
    CRS; by construction every cell has the same area.
    """
    for col in (lat_col, lon_col):
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r}.")
    if not np.isfinite(cell_size_m) or cell_size_m <= 0:
        raise ValueError("cell_size_m must be a positive finite number.")
    if not (np.isfinite(x_origin) and np.isfinite(y_origin)):
        raise ValueError("Grid origins must be finite.")

    out = df.copy()
    cell_area_km2 = float((cell_size_m / 1000.0) ** 2)
    n_rows = int(len(out))

    if n_rows:
        x_m, y_m, resolved_crs = project_points_equal_area(
            out[lon_col].to_numpy(), out[lat_col].to_numpy(), crs=crs
        )
        cell_ix = np.floor((x_m - float(x_origin)) / float(cell_size_m)).astype(np.int64)
        cell_iy = np.floor((y_m - float(y_origin)) / float(cell_size_m)).astype(np.int64)
        cell_x_m = float(x_origin) + (cell_ix.astype(float) + 0.5) * float(cell_size_m)
        cell_y_m = float(y_origin) + (cell_iy.astype(float) + 0.5) * float(cell_size_m)

        from pyproj import Transformer

        inverse = Transformer.from_crs(resolved_crs, "EPSG:4326", always_xy=True)
        cell_lon, cell_lat = inverse.transform(cell_x_m, cell_y_m)
        out["x_m"] = x_m
        out["y_m"] = y_m
        out["cell_ix"] = cell_ix
        out["cell_iy"] = cell_iy
        out["cell_id"] = [f"{int(ix)}_{int(iy)}" for ix, iy in zip(cell_ix, cell_iy)]
        out["cell_x_m"] = cell_x_m
        out["cell_y_m"] = cell_y_m
        out["cell_lat"] = np.asarray(cell_lat, dtype=float)
        out["cell_lon"] = np.asarray(cell_lon, dtype=float)
        n_cells = int(pd.Series(out["cell_id"]).nunique())
    else:
        resolved_crs = crs or default_equal_area_crs(0.0, 0.0, 0.0, 0.0)
        out["x_m"] = pd.Series(dtype=float)
        out["y_m"] = pd.Series(dtype=float)
        out["cell_ix"] = pd.Series(dtype="int64")
        out["cell_iy"] = pd.Series(dtype="int64")
        out["cell_id"] = pd.Series(dtype="object")
        out["cell_x_m"] = pd.Series(dtype=float)
        out["cell_y_m"] = pd.Series(dtype=float)
        out["cell_lat"] = pd.Series(dtype=float)
        out["cell_lon"] = pd.Series(dtype=float)
        x_m = y_m = np.asarray([], dtype=float)
        n_cells = 0
    out["cell_area_km2"] = float(cell_area_km2)

    grid_meta = {
        "crs": resolved_crs,
        "cell_size_m": float(cell_size_m),
        "cell_size_km": float(cell_size_m / 1000.0),
        "cell_area_km2": cell_area_km2,
        "x_origin": float(x_origin),
        "y_origin": float(y_origin),
        "rows": n_rows,
        "n_cells": n_cells,
        "x_min": float(np.min(x_m)) if n_rows else None,
        "x_max": float(np.max(x_m)) if n_rows else None,
        "y_min": float(np.min(y_m)) if n_rows else None,
        "y_max": float(np.max(y_m)) if n_rows else None,
    }
    return out, grid_meta


def _spatial_members(zip_path: Path) -> list[str]:
    """Lists the vector members of a ZIP in deterministic order."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(VECTOR_SUFFIXES + (".kml",))]
    return sorted(n for n in names if not n.endswith("/"))


def read_boundary_frame(boundary_path: str | Path):
    """Reads a vector layer (including ZIP files) and returns it in EPSG:4326."""
    import geopandas as gpd

    path = Path(boundary_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Boundary not found: {path}")

    candidates: list[str] = []
    if path.suffix.lower() == ".zip":
        candidates.append(f"zip://{path}")
        for member in _spatial_members(path):
            candidates.append(f"zip://{path}!{member}")
            candidates.append(f"/vsizip/{path}/{member}")
        candidates.append(f"/vsizip/{path}")
    else:
        candidates.append(str(path))

    frame = None
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            frame = gpd.read_file(candidate)
            if frame is not None:
                break
        except Exception as exc:  # noqa: BLE001 - several drivers/URIs are attempted
            last_error = exc
    if frame is None and path.suffix.lower() == ".zip":
        # Last resort: extract into a temporary directory and read the layer.
        with tempfile.TemporaryDirectory(prefix="boundary_zip_") as tmp:
            with zipfile.ZipFile(path) as zf:
                for member in _spatial_members(path):
                    target = Path(tmp) / Path(member).name
                    with zf.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
            for candidate in sorted(Path(tmp).iterdir()):
                try:
                    frame = gpd.read_file(candidate)
                    if frame is not None:
                        break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
    if frame is None:
        raise ValueError(f"Could not read boundary {path}: {last_error}")
    if frame.empty:
        raise ValueError(f"Boundary {path} is empty.")

    frame = frame[frame.geometry.notna()].copy()
    if frame.empty:
        raise ValueError(f"Boundary {path} does not contain valid geometries.")
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    elif frame.crs.to_string() != "EPSG:4326":
        frame = frame.to_crs("EPSG:4326")
    try:
        if not bool(frame.geometry.is_valid.all()):
            frame = frame.set_geometry(frame.geometry.make_valid())
        frame = frame[~frame.geometry.is_empty].copy()
    except Exception:  # noqa: BLE001 - make_valid is unavailable in older versions
        pass
    frame = frame.reset_index(drop=True)
    return frame


def clip_points_to_polygons(
    df: pd.DataFrame,
    boundary_path: str | Path,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    keep_cols: list[str] | None = None,
    drop_outside: bool = True,
) -> pd.DataFrame:
    """Marks (and optionally drops) points outside a polygon layer.

    The first polygon intersecting each point is kept deterministically to
    avoid duplicates on shared borders.
    """
    import geopandas as gpd

    for col in (lat_col, lon_col):
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r}.")
    boundary = read_boundary_frame(boundary_path)
    keep = [c for c in (keep_cols or []) if c in boundary.columns]
    out = df.copy().reset_index(drop=True)

    if out.empty:
        out["is_inside_boundary"] = pd.Series([], dtype=bool)
        for col in keep:
            out[col] = pd.Series([], dtype="object")
        return out

    boundary = boundary.reset_index(drop=True)
    boundary["__b_order"] = np.arange(len(boundary), dtype=np.int64)
    points = gpd.GeoDataFrame(
        out[[]].copy(),
        geometry=gpd.points_from_xy(out[lon_col].to_numpy(), out[lat_col].to_numpy()),
        crs="EPSG:4326",
    )
    points["__p_order"] = np.arange(len(points), dtype=np.int64)
    joined = gpd.sjoin(points, boundary, how="left", predicate="intersects")
    if "__b_order" not in joined.columns:
        joined["__b_order"] = np.nan
    joined = joined.sort_values(["__p_order", "__b_order"], kind="stable")
    joined = joined.drop_duplicates(subset="__p_order", keep="first")

    inside = joined["index_right"].notna().to_numpy() if "index_right" in joined.columns else np.zeros(len(joined), dtype=bool)
    inside_by_pos = pd.Series(inside, index=joined["__p_order"].to_numpy())
    out["is_inside_boundary"] = (
        inside_by_pos.reindex(np.arange(len(out))).fillna(False).astype(bool).to_numpy()
    )
    for col in keep:
        values = joined[col].to_numpy()
        mapped = pd.Series(values, index=joined["__p_order"].to_numpy())
        out[col] = mapped.reindex(np.arange(len(out))).to_numpy()

    if drop_outside:
        out = out[out["is_inside_boundary"]].reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


class _UnionFind:
    """Union-find with path compression and union by rank."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_l, root_r = self.find(left), self.find(right)
        if root_l == root_r:
            return
        if self.rank[root_l] < self.rank[root_r]:
            root_l, root_r = root_r, root_l
        self.parent[root_r] = root_l
        if self.rank[root_l] == self.rank[root_r]:
            self.rank[root_l] += 1


def cluster_overpass_events(
    df: pd.DataFrame,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    time_col: str = "acq_datetime_utc",
    max_space_m: float = 1500.0,
    max_time_minutes: float = 90.0,
    priority_col: str = "frp",
    crs: str | None = None,
) -> pd.DataFrame:
    """Groups detections that are close in space and time into a single event.

    Uses a KD-tree for spatial candidates and union-find to obtain connected
    components; the representative of each event maximizes ``priority_col``.
    """
    for col in (lat_col, lon_col, time_col):
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r}.")
    if not np.isfinite(max_space_m) or max_space_m < 0:
        raise ValueError("max_space_m must be >= 0.")
    if not np.isfinite(max_time_minutes) or max_time_minutes < 0:
        raise ValueError("max_time_minutes must be >= 0.")

    work = df.copy().reset_index(drop=True)
    if work.empty:
        work["event_cluster_id"] = pd.Series([], dtype="int64")
        work["cluster_size"] = pd.Series([], dtype="int64")
        work["is_cluster_representative"] = pd.Series([], dtype="int8")
        return work

    x_m, y_m, _ = project_points_equal_area(work[lon_col].to_numpy(), work[lat_col].to_numpy(), crs=crs)
    times = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    coords = np.column_stack([x_m, y_m])
    n_rows = len(work)
    uf = _UnionFind(n_rows)

    if n_rows > 1:
        from scipy.spatial import cKDTree

        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=float(max_space_m), output_type="ndarray")
        if len(pairs):
            left, right = pairs[:, 0], pairs[:, 1]
            distances = np.hypot(coords[left, 0] - coords[right, 0], coords[left, 1] - coords[right, 1])
            valid = distances <= float(max_space_m) + 1e-9
            time_ns = times.to_numpy(dtype="datetime64[ns]").astype("int64")
            time_valid = times.notna().to_numpy()
            valid &= time_valid[left] & time_valid[right]
            valid_positions = np.flatnonzero(valid)
            if valid_positions.size:
                valid_left = left[valid_positions]
                valid_right = right[valid_positions]
                minutes = np.abs(time_ns[valid_left] - time_ns[valid_right]) / 6.0e10
                keep = minutes <= float(max_time_minutes) + 1e-9
                for i, j in zip(valid_left[keep], valid_right[keep]):
                    uf.union(int(i), int(j))

    roots = [uf.find(i) for i in range(n_rows)]
    members: dict[int, list[int]] = {}
    for pos, root in enumerate(roots):
        members.setdefault(root, []).append(pos)
    ordered_roots = sorted(members, key=lambda root: min(members[root]))
    root_to_cluster = {root: cluster for cluster, root in enumerate(ordered_roots)}
    cluster_ids = np.asarray([root_to_cluster[root] for root in roots], dtype=np.int64)

    work["event_cluster_id"] = cluster_ids
    work["cluster_size"] = pd.Series(cluster_ids).map(pd.Series(cluster_ids).value_counts()).astype("int64").to_numpy()

    if priority_col in work.columns:
        priority = pd.to_numeric(work[priority_col], errors="coerce").fillna(-np.inf).to_numpy(dtype=float)
    else:
        priority = np.full(n_rows, -np.inf, dtype=float)
    candidates = work.assign(__priority=priority, __pos=np.arange(n_rows))
    representatives = (
        candidates.sort_values(["event_cluster_id", "__priority", "__pos"], ascending=[True, False, True])
        .drop_duplicates(subset="event_cluster_id", keep="first")["__pos"]
        .to_numpy(dtype=np.int64)
    )
    is_rep = np.zeros(n_rows, dtype=np.int8)
    is_rep[representatives] = 1
    work["is_cluster_representative"] = is_rep
    return work.reset_index(drop=True)


def flag_persistent_static_sources(
    df: pd.DataFrame,
    *,
    cell_col: str = "cell_id",
    date_col: str = "acq_date",
    min_active_days: int = 60,
    min_distinct_months: int = 6,
    out_col: str = "is_persistent_static",
) -> pd.DataFrame:
    """Marks cells with chronic activity (many distinct days and months).

    A seasonal fire is concentrated in a few weeks; an industrial or
    volcanic source reappears over many days spread across months.
    """
    for col in (cell_col, date_col):
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r}.")
    if int(min_active_days) < 1 or int(min_distinct_months) < 1:
        raise ValueError("Thresholds must be integers >= 1.")

    work = df.copy().reset_index(drop=True)
    dates = pd.to_datetime(work[date_col], errors="coerce", utc=True)
    days = dates.dt.date
    months = dates.dt.strftime("%Y-%m")

    stats = pd.DataFrame(
        {
            "cell": work[cell_col].astype("object"),
            "day": days,
            "month": months,
        }
    ).dropna(subset=["cell"])
    if stats.empty:
        aggregate = pd.DataFrame(columns=["n_days", "n_months"])
    else:
        aggregate = (
            stats.groupby("cell", dropna=False)
            .agg(n_days=("day", "nunique"), n_months=("month", "nunique"))
            .reset_index()
        )
    if aggregate.empty:
        static_cells: set[Any] = set()
    else:
        active = aggregate[
            (aggregate["n_days"] >= int(min_active_days))
            & (aggregate["n_months"] >= int(min_distinct_months))
        ]
        static_cells = set(active["cell"].tolist())

    work[out_col] = work[cell_col].isin(static_cells).fillna(False).astype(bool)
    summary = {
        "min_active_days": int(min_active_days),
        "min_distinct_months": int(min_distinct_months),
        "rows": int(len(work)),
        "n_cells_observed": int(len(aggregate)),
        "n_static_cells": int(len(static_cells)),
        "static_cells": sorted(str(cell) for cell in static_cells),
        "n_rows_flagged": int(work[out_col].sum()),
    }
    work.attrs = dict(df.attrs)
    work.attrs["static_source_summary"] = summary
    return work
