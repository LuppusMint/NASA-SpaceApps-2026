#!/usr/bin/env python3
"""Climate trend detection: Sen's slope + confidence interval, classical Mann-Kendall, Hamed-Rao AR(1) correction, and BH FDR control."""

from __future__ import annotations

import math
from typing import Any


def _clean_xy(y, x=None):
    import numpy as np

    y_arr = np.asarray(y, dtype=float).ravel()
    if x is None:
        x_arr = np.arange(len(y_arr), dtype=float)
    else:
        x_arr = np.asarray(x, dtype=float).ravel()
    if len(x_arr) != len(y_arr):
        raise ValueError(f"Incompatible lengths: len(x)={len(x_arr)} vs len(y)={len(y_arr)}")
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return y_arr[mask], x_arr[mask]


def sens_slope(y, x=None, *, alpha: float = 0.95) -> dict[str, Any]:
    """Theil-Sen median slope and confidence interval at level alpha (default 0.95)."""
    from scipy import stats

    y_v, x_v = _clean_xy(y, x)
    n = int(len(y_v))
    if n < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": n}
    res = stats.theilslopes(y_v, x_v, alpha=float(alpha))
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "ci_low": float(res.low_slope),
        "ci_high": float(res.high_slope),
        "alpha": float(alpha),
        "n": n,
    }


def _mk_s_and_var(y_arr) -> tuple[float, float, int]:
    import numpy as np

    n = int(len(y_arr))
    if n < 4:
        return 0.0, 0.0, n
    s_stat = 0.0
    for k in range(n - 1):
        s_stat += float(np.sum(np.sign(y_arr[k + 1 :] - y_arr[k])))
    _, counts = np.unique(y_arr, return_counts=True)
    ties = counts[counts > 1]
    tp = float(np.sum(ties * (ties - 1.0) * (2.0 * ties + 5.0))) if len(ties) else 0.0
    var_s = (n * (n - 1.0) * (2.0 * n + 5.0) - tp) / 18.0
    return float(s_stat), float(var_s), n


def _mk_z_and_p(s_stat: float, var_s: float) -> tuple[float, float]:
    from scipy import stats

    if var_s <= 0.0:
        return 0.0, 1.0
    sd = math.sqrt(var_s)
    if s_stat > 0.0:
        z_stat = (s_stat - 1.0) / sd
    elif s_stat < 0.0:
        z_stat = (s_stat + 1.0) / sd
    else:
        z_stat = 0.0
    p_val = float(2.0 * stats.norm.sf(abs(z_stat)))
    return float(z_stat), float(min(1.0, max(p_val, 0.0)))


def mann_kendall_test(y, x=None, *, alpha: float = 0.05) -> dict[str, Any]:
    """Classical non-parametric Mann-Kendall test (assumes serial independence) plus Sen's slope."""
    y_v, x_v = _clean_xy(y, x)
    order = x_v.argsort(kind="mergesort")
    y_ord = y_v[order]
    x_ord = x_v[order]
    s_stat, var_s, n = _mk_s_and_var(y_ord)
    z_stat, p_val = _mk_z_and_p(s_stat, var_s)
    sen = sens_slope(y_ord, x_ord, alpha=1.0 - alpha)
    return {
        "method": "mann_kendall",
        "n": n,
        "S": s_stat,
        "var_s": var_s,
        "z": z_stat,
        "p_value": p_val,
        "significant": bool(p_val < alpha),
        "slope": sen["slope"],
        "intercept": sen["intercept"],
        "slope_ci95": [sen["ci_low"], sen["ci_high"]],
    }


def hamed_rao_mk_test(
    y,
    x=None,
    *,
    alpha: float = 0.05,
    nlags: int | None = None,
) -> dict[str, Any]:
    """Modified Mann-Kendall test (Hamed & Rao 1998 / Yue & Wang 2004) for AR autocorrelation."""
    import numpy as np
    from scipy import stats

    y_v, x_v = _clean_xy(y, x)
    order = x_v.argsort(kind="mergesort")
    y_ord = y_v[order]
    x_ord = x_v[order]
    n = int(len(y_ord))
    if n < 6:
        base = mann_kendall_test(y_ord, x_ord, alpha=alpha)
        base["method"] = "hamed_rao_mk"
        base["n_over_n_star"] = 1.0
        base["ess"] = float(n)
        return base

    s_stat, var_s, _ = _mk_s_and_var(y_ord)
    sen = sens_slope(y_ord, x_ord, alpha=1.0 - alpha)
    detrended = y_ord - sen["slope"] * x_ord

    significant_lags = 0
    r1_est = 0.0
    ks = np.arange(1, n, dtype=float)
    weights = (n - ks) * (n - ks - 1.0) * (n - ks - 2.0)

    if nlags is None:
        # Geometric AR(1) model (Yue & Wang 2004) on detrended residuals: avoids negative bias at high lags
        a = detrended[:-1] - float(np.mean(detrended[:-1]))
        b = detrended[1:] - float(np.mean(detrended[1:]))
        den = math.sqrt(float(np.sum(a ** 2)) * float(np.sum(b ** 2)))
        if den > 1e-15:
            r1_est = float(np.sum(a * b) / den)
        r1_pos = max(0.0, min(0.95, r1_est))
        crit = 1.96 / math.sqrt(n)
        significant_lags = int(abs(r1_est) > crit)
        acf_sum = float(np.sum(weights * (r1_pos ** ks)))
    else:
        ranks = stats.rankdata(detrended)
        centered = ranks - np.mean(ranks)
        denom = float(np.sum(centered ** 2))
        max_lag = min(n - 1, int(nlags))
        crit = stats.norm.ppf(1.0 - 0.05 / 2.0) / math.sqrt(n)
        acf_sum = 0.0
        if denom > 0.0:
            for lag in range(1, max_lag + 1):
                r_k = float(np.sum(centered[:-lag] * centered[lag:]) / denom)
                if lag == 1:
                    r1_est = r_k
                acf_sum += (n - lag) * (n - lag - 1.0) * (n - lag - 2.0) * r_k
                if abs(r_k) > crit:
                    significant_lags += 1

    n_over_n_star = 1.0 + (2.0 / (n * (n - 1.0) * (n - 2.0))) * acf_sum
    n_over_n_star = max(float(n_over_n_star), 1.0)
    var_s_corr = var_s * n_over_n_star
    z_stat, p_val = _mk_z_and_p(s_stat, var_s_corr)

    return {
        "method": "hamed_rao_mk",
        "n": n,
        "ess": round(float(n / n_over_n_star), 4),
        "n_over_n_star": round(n_over_n_star, 6),
        "r1": round(r1_est, 6),
        "significant_acf_lags": int(significant_lags),
        "S": s_stat,
        "var_s_uncorrected": var_s,
        "var_s_corrected": var_s_corr,
        "z": z_stat,
        "p_value": p_val,
        "significant": bool(p_val < alpha),
        "slope": sen["slope"],
        "intercept": sen["intercept"],
        "slope_ci95": [sen["ci_low"], sen["ci_high"]],
    }


def benjamini_hochberg_fdr(pvals, *, alpha: float = 0.05) -> dict[str, Any]:
    """Benjamini-Hochberg false discovery rate control (preserves shape and NaNs)."""
    import numpy as np
    from scipy import stats

    arr = np.asarray(pvals, dtype=float)
    q_out = np.full(arr.shape, np.nan, dtype=float)
    rej = np.zeros(arr.shape, dtype=bool)

    valid_mask = np.isfinite(arr)
    valid_p = np.clip(arr[valid_mask], 0.0, 1.0)
    if valid_p.size > 0:
        q_valid = stats.false_discovery_control(valid_p, method="bh")
        q_out[valid_mask] = q_valid
        rej[valid_mask] = q_valid <= float(alpha)

    return {
        "q_values": q_out,
        "rejected": rej,
        "alpha": float(alpha),
        "n_valid": int(valid_p.size),
        "n_rejected": int(np.sum(rej)),
    }


def to_decimal_years(values):
    """Convert datetime-like coordinates to decimal years (365.25-day year)."""
    import numpy as np
    import pandas as pd

    index = pd.DatetimeIndex(pd.to_datetime(np.asarray(values)))
    return np.asarray(index.year + (index.dayofyear - 1) / 365.25, dtype=float)


def grid_trend_field(
    da,
    *,
    time_dim: str = "time",
    time_values=None,
    method: str = "hamed_rao",
    alpha: float = 0.05,
    min_valid: int = 10,
):
    """Computes Sen's slope, p-value (MK or Hamed-Rao), and BH q-value at each grid cell."""
    import numpy as np
    import xarray as xr

    if time_values is None:
        t_vals = to_decimal_years(da.coords[time_dim])
    else:
        t_vals = np.asarray(time_values, dtype=float)

    other_dims = [d for d in da.dims if d != time_dim]
    transposed = da.transpose(time_dim, *other_dims)
    arr = np.asarray(transposed.values, dtype=float)
    spatial_shape = arr.shape[1:]
    flat = arr.reshape(arr.shape[0], -1)
    n_cells = flat.shape[1]

    slope_f = np.full(n_cells, np.nan, dtype=float)
    pval_f = np.full(n_cells, np.nan, dtype=float)
    z_f = np.full(n_cells, np.nan, dtype=float)

    test_fn = hamed_rao_mk_test if method == "hamed_rao" else mann_kendall_test
    for idx in range(n_cells):
        col = flat[:, idx]
        valid = np.isfinite(col) & np.isfinite(t_vals)
        if int(np.sum(valid)) < min_valid:
            continue
        res = test_fn(col[valid], t_vals[valid], alpha=alpha)
        slope_f[idx] = res["slope"]
        pval_f[idx] = res["p_value"]
        z_f[idx] = res["z"]

    p_grid = pval_f.reshape(spatial_shape)
    fdr = benjamini_hochberg_fdr(p_grid, alpha=alpha)
    coords = {d: da.coords[d] for d in other_dims if d in da.coords}

    return xr.Dataset(
        {
            "slope": (other_dims, slope_f.reshape(spatial_shape)),
            "p_value": (other_dims, p_grid),
            "q_value": (other_dims, fdr["q_values"]),
            "significant_fdr": (other_dims, fdr["rejected"]),
            "z_stat": (other_dims, z_f.reshape(spatial_shape)),
        },
        coords=coords,
        attrs={"trend_method": method, "fdr_method": "benjamini_hochberg", "alpha": float(alpha)},
    )
