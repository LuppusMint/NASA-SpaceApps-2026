"""Out-of-sample post-processing for regression predictions.

Both routines fit their free parameter on the outer-training folds only and
apply it to the held-out fold, so the returned score evaluates an adaptive
procedure rather than a parameter tuned on the validation data.
"""

from __future__ import annotations

import math
from typing import Any

from src.core.metrics import evaluate_metric, gaussian_nll


def calibrate_regression_sigma_foldwise(
    y_true,
    mu_pred,
    sigma_pred,
    fold_ids,
    *,
    min_scale: float = 1e-3,
    max_scale: float = 1e3,
    min_sigma: float = 1e-6,
) -> dict[str, Any]:
    """Calibrate a predictive ``sigma`` scale per fold without look-ahead.

    For each fold ``k`` the MLE scale of the standardised outer-training
    residuals is estimated and applied to the outer-validation residuals::

        s_k = clip(sqrt(mean(((y_tr - mu_tr) / sigma_tr) ** 2)), min_scale, max_scale)
        sigma_cal[k] = clip(sigma_pred[k] * s_k, min_sigma, None)
    """
    import numpy as np

    y_arr = np.asarray(y_true, dtype=float).reshape(-1)
    mu_arr = np.asarray(mu_pred, dtype=float).reshape(-1)
    sigma_arr = np.asarray(sigma_pred, dtype=float)
    if sigma_arr.ndim == 2 and sigma_arr.shape[1] == 2:
        sigma_arr = sigma_arr[:, 1]
    sigma_arr = sigma_arr.reshape(-1)
    f_arr = np.asarray(fold_ids, dtype=int).reshape(-1)

    if y_arr.size == 0:
        raise ValueError("calibrate_regression_sigma_foldwise requires at least one sample.")
    if not (y_arr.size == mu_arr.size == sigma_arr.size == f_arr.size):
        raise ValueError(
            "Incompatible lengths in calibrate_regression_sigma_foldwise: "
            f"y_true={y_arr.size}, mu_pred={mu_arr.size}, sigma_pred={sigma_arr.size}, "
            f"fold_ids={f_arr.size}."
        )
    if min_scale <= 0.0 or max_scale < min_scale or min_sigma <= 0.0:
        raise ValueError("Invalid scales: require 0 < min_scale <= max_scale and min_sigma > 0.")

    sigma_safe = np.clip(sigma_arr, min_sigma, None)
    folds = sorted(np.unique(f_arr).tolist())
    calibrated = np.empty_like(sigma_safe)
    fold_scales: list[float] = []

    for fold in folds:
        val_mask = f_arr == fold
        tr_mask = ~val_mask
        if not np.any(tr_mask):
            scale = 1.0
        else:
            standardized = (y_arr[tr_mask] - mu_arr[tr_mask]) / sigma_safe[tr_mask]
            scale = float(np.clip(np.sqrt(np.mean(standardized**2)), min_scale, max_scale))
        fold_scales.append(round(scale, 8))
        calibrated[val_mask] = np.clip(sigma_safe[val_mask] * scale, min_sigma, None)

    nll_before = gaussian_nll(y_arr, mu_arr, sigma_safe)
    nll_after = gaussian_nll(y_arr, mu_arr, calibrated)
    mean_scale = float(np.mean(fold_scales)) if fold_scales else 1.0

    return {
        "sigma_calibrated": calibrated,
        "fold_scales": fold_scales,
        "mean_scale": round(mean_scale, 8),
        "nll_before": round(float(nll_before), 8),
        "nll_after": round(float(nll_after), 8),
        "evaluation_scope": "foldwise_adaptive_procedure_not_frozen_scale",
    }


def hurdle_zero_threshold_foldwise(
    y_true,
    p_positive_oof,
    pos_intensity_oof,
    fold_ids,
    *,
    metric: str = "rmse",
    thresholds=None,
) -> dict[str, Any]:
    """Fit a hurdle threshold per fold on outer-training data only.

    The hard prediction is ``I(p >= t_k) * max(pos_intensity, 0)``; the smooth
    baseline is ``p * max(pos_intensity, 0)``.  ``score_calibrated`` evaluates a
    fold-adaptive procedure, not a permanently frozen threshold.
    """
    import numpy as np

    supported = {"rmse", "mae", "poisson_deviance", "tweedie_deviance"}
    if metric not in supported:
        raise ValueError(f"Unsupported hurdle metric: {metric}")

    y_arr = np.asarray(y_true, dtype=float).reshape(-1)
    p_arr = np.asarray(p_positive_oof, dtype=float).reshape(-1)
    intensity_arr = np.clip(np.asarray(pos_intensity_oof, dtype=float).reshape(-1), 0.0, None)
    f_arr = np.asarray(fold_ids, dtype=int).reshape(-1)

    if y_arr.size == 0:
        raise ValueError("hurdle_zero_threshold_foldwise requires at least one sample.")
    if not (y_arr.size == p_arr.size == intensity_arr.size == f_arr.size):
        raise ValueError(
            "Incompatible lengths in hurdle_zero_threshold_foldwise: "
            f"y_true={y_arr.size}, p_positive_oof={p_arr.size}, "
            f"pos_intensity_oof={intensity_arr.size}, fold_ids={f_arr.size}."
        )

    if thresholds is None:
        grid = np.linspace(0.0, 1.0, 101, dtype=float)
    else:
        grid = np.asarray(thresholds, dtype=float).reshape(-1)
        if grid.size == 0:
            raise ValueError("thresholds cannot be empty.")
        if not np.all(np.isfinite(grid)):
            raise ValueError("thresholds must contain finite values.")
        grid = np.clip(grid, 0.0, 1.0)
    grid = np.unique(grid)

    folds = sorted(np.unique(f_arr).tolist())
    fold_thresholds: list[float] = []
    oof_pred = np.zeros_like(y_arr)

    for fold in folds:
        val_mask = f_arr == fold
        tr_mask = ~val_mask
        best_threshold = 0.5
        best_score = math.inf
        if np.any(tr_mask):
            for threshold in grid:
                train_pred = (p_arr[tr_mask] >= threshold).astype(float) * intensity_arr[tr_mask]
                score = evaluate_metric(metric, "regression", y_arr[tr_mask], train_pred)
                if np.isfinite(score) and score < best_score - 1e-15:
                    best_score = float(score)
                    best_threshold = float(threshold)
        fold_thresholds.append(round(best_threshold, 8))
        oof_pred[val_mask] = (p_arr[val_mask] >= best_threshold).astype(float) * intensity_arr[val_mask]

    soft_pred = p_arr * intensity_arr
    score_soft = evaluate_metric(metric, "regression", y_arr, soft_pred)
    score_calibrated = evaluate_metric(metric, "regression", y_arr, oof_pred)

    return {
        "oof_pred": oof_pred,
        "fold_thresholds": fold_thresholds,
        "metric": metric,
        "score_soft_baseline": round(float(score_soft), 8),
        "score_calibrated": round(float(score_calibrated), 8),
        "evaluation_scope": "foldwise_adaptive_procedure_not_frozen_threshold",
    }


__all__ = ["calibrate_regression_sigma_foldwise", "hurdle_zero_threshold_foldwise"]
