#!/usr/bin/env python3
"""Climate extremes analysis: block maxima and GEV fitting (explicit SciPy c = -xi convention)."""

from __future__ import annotations

import math
from typing import Any


def block_maxima(
    data,
    *,
    freq: str = "YS",
    time_dim: str = "time",
    min_valid_frac: float = 0.75,
):
    """Extracts block maxima over a time block (e.g. annual 'YS'), discarding incomplete blocks."""
    import numpy as np
    import pandas as pd

    if isinstance(data, pd.Series):
        s = data.dropna()
        if not isinstance(s.index, pd.DatetimeIndex):
            raise ValueError("The pandas Series must have a DatetimeIndex for block_maxima.")
        counts = s.resample(freq).count()
        max_vals = s.resample(freq).max()
        max_count = float(counts.max()) if len(counts) else 1.0
        valid = counts >= max(1.0, min_valid_frac * max_count)
        return max_vals[valid].dropna()

    res = data.resample({time_dim: freq})
    max_da = res.max(skipna=True)
    if min_valid_frac > 0.0:
        counts = res.count()
        max_count = float(counts.max()) if counts.size else 1.0
        min_req = max(1.0, min_valid_frac * max_count)
        max_da = max_da.where(counts >= min_req)
    return max_da.dropna(dim=time_dim) if max_da.ndim == 1 else max_da


def _lmoments_gev(arr):
    """L-moment (PWM) GEV fit. Returns (scipy_c, loc, scale) or None."""
    import math
    import numpy as np
    from scipy import special

    x = np.sort(np.asarray(arr, dtype=float))
    n = int(len(x))
    if n < 10 or not np.all(np.isfinite(x)) or float(np.std(x)) <= 0.0:
        return None
    i = np.arange(n, dtype=float)
    b0 = float(np.mean(x))
    b1 = float(np.sum((i / (n - 1.0)) * x) / n) if n > 1 else float("nan")
    b2 = (
        float(np.sum((i * (i - 1.0) / ((n - 1.0) * (n - 2.0))) * x) / n)
        if n > 2
        else float("nan")
    )
    l1 = b0
    l2 = 2.0 * b1 - b0
    l3 = 6.0 * b2 - 6.0 * b1 + b0
    if not (math.isfinite(l2) and l2 > 0.0):
        return None
    t3 = l3 / l2
    c = 2.0 / (3.0 + t3) - math.log(2.0) / math.log(3.0)
    k = 7.8590 * c + 2.9554 * c * c
    if not math.isfinite(k):
        return None
    if abs(k) < 1e-7:
        alpha = l2 / math.log(2.0)
        loc = l1 - 0.5772156649015329 * alpha
    else:
        if 1.0 + k <= 0.0:
            return None
        g = float(special.gamma(1.0 + k))
        denom = 1.0 - 2.0 ** (-k)
        if not math.isfinite(g) or abs(denom) < 1e-15:
            return None
        alpha = l2 * k / (denom * g)
        loc = l1 - alpha * (1.0 - g) / k
    if not (math.isfinite(alpha) and alpha > 0.0 and math.isfinite(loc)):
        return None
    return float(k), float(loc), float(alpha)


def fit_gev(
    maxima,
    *,
    return_periods: tuple[float, ...] | list[float] = (10, 20, 50, 100),
    n_bootstrap: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fits a GEV with L-moments (PWM) and a parametric bootstrap.

    The returned shape follows the explicit SciPy convention ``c_scipy = -xi``.
    L-moments are used because the free MLE is unstable for 25-year block
    series; if the L-moment fit is invalid, the function falls back to MLE.
    """
    import math
    import numpy as np
    from scipy import stats

    arr = np.asarray(maxima.values if hasattr(maxima, "values") else maxima, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    n = int(len(arr))
    if n < 10:
        raise ValueError(f"At least 10 block maxima are required to fit the GEV distribution (n={n}).")

    estimator = "lmoments"
    fitted = _lmoments_gev(arr)
    if fitted is None:
        estimator = "mle"
        fitted = [float(v) for v in stats.genextreme.fit(arr)]
    c_hat, loc_hat, scale_hat = [float(v) for v in fitted]
    xi_hat = -c_hat
    mu_hat = loc_hat
    sigma_hat = max(scale_hat, 1e-12)

    periods = [float(t) for t in return_periods]
    return_levels: dict[str, float] = {}
    for t_per in periods:
        if t_per <= 1.0:
            raise ValueError(f"Invalid return period (must be > 1): {t_per}")
        rl = float(stats.genextreme.isf(1.0 / t_per, c=c_hat, loc=mu_hat, scale=sigma_hat))
        return_levels[str(int(t_per) if t_per.is_integer() else t_per)] = rl

    ks_res = stats.kstest(arr, "genextreme", args=(c_hat, mu_hat, sigma_hat))

    # Parametric bootstrap from the fitted GEV, refitting with L-moments.
    rng = np.random.default_rng(seed)
    boot_mu: list[float] = []
    boot_sigma: list[float] = []
    boot_xi: list[float] = []
    boot_rls: dict[str, list[float]] = {k: [] for k in return_levels}

    for _ in range(max(int(n_bootstrap), 0)):
        try:
            sample = stats.genextreme.rvs(
                c=c_hat, loc=mu_hat, scale=sigma_hat, size=n, random_state=rng
            )
            fitted_b = _lmoments_gev(sample)
            if fitted_b is None:
                cb, lb, sb = [float(v) for v in stats.genextreme.fit(sample)]
            else:
                cb, lb, sb = [float(v) for v in fitted_b]
        except Exception:
            continue
        if not (math.isfinite(cb) and math.isfinite(lb) and math.isfinite(sb) and sb > 0.0):
            continue
        xi_b = float(-cb)
        if abs(xi_b) > 2.0:
            continue
        boot_mu.append(float(lb))
        boot_sigma.append(float(sb))
        boot_xi.append(xi_b)
        for key, t_per in zip(return_levels.keys(), periods):
            val = float(stats.genextreme.isf(1.0 / t_per, c=cb, loc=float(lb), scale=float(sb)))
            if math.isfinite(val):
                boot_rls[key].append(val)

    lo_q, hi_q = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)

    def _ci(vals: list[float]) -> list[float] | None:
        if len(vals) < 10:
            return None
        lo, hi = np.percentile(vals, [lo_q, hi_q]).tolist()
        return [float(lo), float(hi)]

    return {
        "n_blocks": n,
        "estimator": estimator,
        "scipy_shape_c": c_hat,
        "mu": mu_hat,
        "sigma": sigma_hat,
        "xi": xi_hat,
        "parameter_convention": "Standard GEV (mu=loc, sigma=scale, xi=-c_scipy; xi>0 Frechet heavy tail)",
        "return_levels": return_levels,
        "ci_level": round(1.0 - alpha, 4),
        "param_cis": {
            "mu": _ci(boot_mu),
            "sigma": _ci(boot_sigma),
            "xi": _ci(boot_xi),
        },
        "return_level_cis": {k: _ci(v) for k, v in boot_rls.items()},
        "ks_statistic": float(ks_res.statistic),
        "ks_p_value": float(ks_res.pvalue),
        "n_bootstrap_valid": len(boot_xi),
    }
