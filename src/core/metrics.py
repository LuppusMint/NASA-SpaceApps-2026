"""Regression and classification metrics used by the M0-M3 benchmark.

The public entry point is :func:`evaluate_metric`, whose signature matches the
experiment ledger: ``evaluate_metric(metric, task_type, y_true, y_pred)``.
All metrics are oriented so that lower is better, except ``r2`` (higher is
better, as usual).  Empty or non-finite inputs return ``nan`` rather than
raising, so a degenerate validation fold cannot abort a full benchmark run.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SUPPORTED_METRICS = (
    "rmse",
    "mae",
    "rmsle",
    "r2",
    "tweedie_deviance",
    "poisson_deviance",
    "gaussian_nll",
    "log_loss",
    "brier",
)


def _as_float_1d(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def gaussian_nll(y_true: Any, mu: Any, sigma: Any) -> float:
    """Mean Gaussian negative log-likelihood ``-log N(y | mu, sigma)``.

    ``sigma`` is clipped from below to avoid a division by zero.  The returned
    value follows the usual convention ``0.5 * mean(log(2*pi*sigma**2) +
    ((y - mu) / sigma)**2)``.
    """
    truth = _as_float_1d(y_true)
    mean = _as_float_1d(mu)
    scale = _as_float_1d(sigma)
    if not (truth.size == mean.size == scale.size) or truth.size == 0:
        return float("nan")
    valid = np.isfinite(truth) & np.isfinite(mean) & np.isfinite(scale) & (scale > 0.0)
    if not np.any(valid):
        return float("nan")
    safe_sigma = np.clip(scale[valid], 1e-12, None)
    residual = (truth[valid] - mean[valid]) / safe_sigma
    value = 0.5 * np.mean(np.log(2.0 * math.pi * safe_sigma**2) + residual**2)
    return float(value)


def _r2(truth: np.ndarray, pred: np.ndarray) -> float:
    residual = float(np.sum((truth - pred) ** 2))
    total = float(np.sum((truth - float(np.mean(truth))) ** 2))
    if total == 0.0:
        return 1.0 if residual == 0.0 else 0.0
    return 1.0 - residual / total


def evaluate_metric(metric: str, task_type: str, y_true: Any, y_pred: Any) -> float:
    """Evaluate one metric with a uniform, leakage-neutral interface.

    Parameters
    ----------
    metric:
        One of :data:`SUPPORTED_METRICS`.
    task_type:
        ``"regression"`` or ``"binary"``.  The value only changes how
        probabilistic predictions are interpreted.
    y_true, y_pred:
        Array-like predictions.  For ``gaussian_nll`` a two-column ``y_pred`` is
        read as ``(mu, sigma)``; otherwise ``y_pred`` is treated as the mean and
        the residual standard deviation is estimated from the data.
    """
    name = str(metric or "").strip().lower()
    if name not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: {metric!r}")

    truth = _as_float_1d(y_true)
    raw_pred = np.asarray(y_pred, dtype=float)
    if truth.size == 0:
        return float("nan")

    if name == "gaussian_nll":
        if raw_pred.ndim == 2 and raw_pred.shape[1] == 2 and raw_pred.shape[0] == truth.size:
            return gaussian_nll(truth, raw_pred[:, 0], raw_pred[:, 1])
        pred = raw_pred.reshape(-1)
        if pred.size != truth.size:
            return float("nan")
        valid = np.isfinite(truth) & np.isfinite(pred)
        if not np.any(valid):
            return float("nan")
        residual = truth[valid] - pred[valid]
        sigma = float(np.sqrt(np.mean(residual**2)))
        return gaussian_nll(truth[valid], pred[valid], np.full(pred[valid].size, max(sigma, 1e-12)))

    if name in {"log_loss", "brier"}:
        if raw_pred.ndim == 2 and raw_pred.shape[1] == 2:
            if raw_pred.shape[0] != truth.size:
                return float("nan")
            pred = np.asarray(raw_pred[:, 1], dtype=float).reshape(-1)
        else:
            pred = raw_pred.reshape(-1)
        if pred.size != truth.size:
            return float("nan")
        valid = np.isfinite(truth) & np.isfinite(pred)
        if not np.any(valid):
            return float("nan")
        truth = truth[valid]
        probability = np.clip(pred[valid], 1e-15, 1.0 - 1e-15)
        label = (truth > 0.5).astype(float)
        if name == "brier":
            return float(np.mean((label - probability) ** 2))
        return float(
            -np.mean(label * np.log(probability) + (1.0 - label) * np.log(1.0 - probability))
        )

    pred = raw_pred.reshape(-1)
    if pred.size != truth.size:
        return float("nan")
    valid = np.isfinite(truth) & np.isfinite(pred)
    if not np.any(valid):
        return float("nan")
    truth = truth[valid]
    pred = pred[valid]

    if name == "rmse":
        return float(np.sqrt(np.mean((truth - pred) ** 2)))
    if name == "mae":
        return float(np.mean(np.abs(truth - pred)))
    if name == "rmsle":
        truth_pos = np.clip(truth, 0.0, None)
        pred_pos = np.clip(pred, 0.0, None)
        return float(np.sqrt(np.mean((np.log1p(truth_pos) - np.log1p(pred_pos)) ** 2)))
    if name == "r2":
        return float(_r2(truth, pred))

    # Deviance metrics require non-negative observations/predictions.
    if np.any(truth < 0.0):
        return float("nan")
    non_negative = np.clip(pred, 1e-12, None)
    try:
        if name == "poisson_deviance":
            from sklearn.metrics import mean_poisson_deviance

            return float(mean_poisson_deviance(truth, non_negative))
        if name == "tweedie_deviance":
            from sklearn.metrics import mean_tweedie_deviance

            return float(mean_tweedie_deviance(truth, non_negative, power=1.5))
    except (ValueError, ZeroDivisionError):
        return float("nan")
    return float("nan")


__all__ = ["SUPPORTED_METRICS", "evaluate_metric", "gaussian_nll"]
