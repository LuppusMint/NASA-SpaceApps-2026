"""Strict Level-2 simplex stacking over validated out-of-fold predictions.

Only the scientific blending routine and its helpers are kept.  The original
CLI, run-directory bookkeeping and experiment-registry calls are intentionally
absent: the module is a deterministic numerical utility.
"""

from __future__ import annotations

import math
from typing import Any

from src.core.metrics import evaluate_metric

_SIMPLEX_REGRESSION_METRICS = {"rmse", "mae", "poisson_deviance", "tweedie_deviance"}
_SIMPLEX_CLASSIFICATION_METRICS = {"log_loss", "brier"}


def _simplex_project(weights):
    """Proyecta un vector al símplex ``w >= 0, sum(w) == 1`` de forma determinista."""
    import numpy as np

    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, 1.0)
    total = float(w.sum())
    if total <= 0.0:
        return np.full_like(w, 1.0 / max(len(w), 1))
    return w / total


def _simplex_predict(matrix, weights):
    """Predice con un blend convexo de columnas (regresión/binaria) o bloques (multiclase)."""
    import numpy as np

    w = np.asarray(weights, dtype=float).reshape(-1)
    if matrix.ndim == 3:
        return np.tensordot(matrix, w, axes=([1], [0]))
    return np.asarray(matrix, dtype=float) @ w


def _simplex_score(metric: str, task: str, y_true, prediction) -> float:
    """Puntúa una predicción blend con la métrica solicitada (menor es mejor)."""
    import numpy as np

    y_arr = np.asarray(y_true)
    pred = np.asarray(prediction, dtype=float)
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_arr.astype(float) - pred) ** 2)))
    if metric == "mae":
        return float(np.mean(np.abs(y_arr.astype(float) - pred)))
    if metric == "poisson_deviance":
        return evaluate_metric("poisson_deviance", "regression", y_arr, pred)
    if metric == "tweedie_deviance":
        return evaluate_metric("tweedie_deviance", "regression", y_arr, pred)
    if metric == "log_loss":
        if pred.ndim == 1:
            p = np.clip(pred, 1e-15, 1.0 - 1e-15)
            y_bin = y_arr.astype(int)
            return float(-np.mean(y_bin * np.log(p) + (1 - y_bin) * np.log(1.0 - p)))
        probs = np.clip(pred, 1e-15, None)
        probs = probs / probs.sum(axis=1, keepdims=True)
        y_idx = y_arr.astype(int)
        return float(-np.mean(np.log(probs[np.arange(len(y_idx)), y_idx])))
    if metric == "brier":
        if pred.ndim == 1:
            return float(np.mean((y_arr.astype(float) - pred) ** 2))
        probs = np.clip(pred, 1e-15, None)
        probs = probs / probs.sum(axis=1, keepdims=True)
        y_idx = y_arr.astype(int)
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(y_idx)), y_idx] = 1.0
        return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    raise ValueError(f"Métrica no soportada para blending simplex: {metric}")


def _simplex_starts(matrix, y_true, task: str, metric: str, *, max_vertices: int = 5):
    """Inicio uniforme + vértices de los mejores modelos individuales."""
    import numpy as np

    n_models = int(matrix.shape[1])
    starts = [np.full(n_models, 1.0 / n_models)]
    if n_models <= 1:
        return starts

    individual_scores = []
    for idx in range(n_models):
        vertex = np.zeros(n_models)
        vertex[idx] = 1.0
        individual_scores.append(
            _simplex_score(metric, task, y_true, _simplex_predict(matrix, vertex))
        )
    order = np.argsort(np.asarray(individual_scores), kind="stable")
    for idx in order[: min(n_models, max_vertices)]:
        vertex = np.zeros(n_models)
        vertex[int(idx)] = 1.0
        starts.append(vertex)
    return starts


def _simplex_fit_weights(matrix, y_true, task: str, metric: str):
    """Ajusta pesos en el símplex con SLSQP desde varios inicios."""
    import numpy as np
    from scipy.optimize import minimize

    n_models = int(matrix.shape[1])
    if n_models == 1:
        weights = np.ones(1, dtype=float)
        return weights, _simplex_score(metric, task, y_true, _simplex_predict(matrix, weights))

    bounds = [(0.0, 1.0)] * n_models
    constraint = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}

    def objective(w):
        return _simplex_score(metric, task, y_true, _simplex_predict(matrix, w))

    best_weights = None
    best_score = math.inf
    for start in _simplex_starts(matrix, y_true, task, metric):
        try:
            result = minimize(
                objective,
                np.asarray(start, dtype=float),
                method="SLSQP",
                bounds=bounds,
                constraints=constraint,
                options={"maxiter": 500, "ftol": 1e-12},
            )
            candidate = _simplex_project(result.x if result.x is not None else start)
        except Exception:
            candidate = _simplex_project(start)
        score = objective(candidate)
        if score < best_score - 1e-15:
            best_score = float(score)
            best_weights = candidate

    if best_weights is None:
        best_weights = _simplex_project(np.full(n_models, 1.0 / n_models))
        best_score = objective(best_weights)
    return best_weights, float(best_score)


def _simplex_fit_with_pruning(
    matrix,
    y_true,
    task: str,
    metric: str,
    *,
    prune_threshold: float,
    allow_prune: bool,
):
    """Ajusta y, opcionalmente, poda pesos pequeños y reajusta solo los activos."""
    import numpy as np

    n_models = int(matrix.shape[1])
    weights, _ = _simplex_fit_weights(matrix, y_true, task, metric)
    if not allow_prune or prune_threshold <= 0.0 or n_models <= 1:
        return weights

    active = np.where(weights >= float(prune_threshold))[0]
    if active.size == 0:
        active = np.asarray([int(np.argmax(weights))])
    if active.size >= n_models:
        return weights

    submatrix = matrix[:, active] if matrix.ndim == 2 else matrix[:, active, :]
    active_weights, _ = _simplex_fit_weights(submatrix, y_true, task, metric)
    full_weights = np.zeros(n_models, dtype=float)
    full_weights[active] = active_weights
    return _simplex_project(full_weights)


def optimize_simplex_blend_foldwise(
    oof_matrix,
    y_true,
    fold_ids,
    *,
    metric: str = "rmse",
    prune_threshold: float = 0.02,
    allow_prune: bool = True,
    time_mode: bool = False,
) -> dict[str, Any]:
    """Optimiza pesos convexos por fold con ajuste estrictamente fuera de validación.

    Con ``time_mode=False`` cada fold se ajusta en ``fold_ids != k``. Con
    ``time_mode=True`` el fold inicial es warmup (blend uniforme) y cada fold se
    ajusta únicamente en el pasado estricto ``fold_ids < k``, sin usar futuro.

    ``oof_matrix`` puede ser ``(n, n_models)`` para regresión o clasificación binaria
    (probabilidad de la clase positiva por modelo), o ``(n, n_models, n_classes)``
    para clasificación multiclase. En cada fold se prueban el inicio uniforme y
    los mejores vértices individuales; si ``allow_prune`` los pesos menores que
    ``prune_threshold`` se eliminan y se reajustan únicamente los activos.
    """
    import numpy as np

    matrix = np.asarray(oof_matrix, dtype=float)
    if matrix.ndim not in (2, 3):
        raise ValueError("oof_matrix debe ser 2-D (n, modelos) o 3-D (n, modelos, clases).")
    n_samples, n_models = int(matrix.shape[0]), int(matrix.shape[1])
    if n_samples == 0 or n_models == 0:
        raise ValueError("oof_matrix no puede estar vacío.")

    if metric in _SIMPLEX_REGRESSION_METRICS:
        task = "regression"
    elif metric in _SIMPLEX_CLASSIFICATION_METRICS:
        task = "classification"
    else:
        raise ValueError(f"Métrica no soportada para blending simplex: {metric}")
    if matrix.ndim == 3 and task != "classification":
        raise ValueError("Una matriz 3-D solo es válida para clasificación multiclase.")

    y_arr = np.asarray(y_true).reshape(-1)
    f_arr = np.asarray(fold_ids, dtype=int).reshape(-1)
    if not (y_arr.size == f_arr.size == n_samples):
        raise ValueError(
            "Longitudes incompatibles en optimize_simplex_blend_foldwise: "
            f"oof_matrix={n_samples}, y_true={y_arr.size}, fold_ids={f_arr.size}."
        )

    if task == "regression":
        if np.any(~np.isfinite(y_arr.astype(float))):
            raise ValueError("y_true contiene valores no finitos.")
        if metric in {"poisson_deviance", "tweedie_deviance"} and np.any(y_arr.astype(float) < 0):
            raise ValueError(f"{metric} requiere y_true >= 0.")
    else:
        y_int = y_arr.astype(int)
        if matrix.ndim == 3:
            n_classes = int(matrix.shape[2])
            if np.any(y_int < 0) or np.any(y_int >= n_classes):
                raise ValueError("y_true debe contener etiquetas enteras en [0, n_classes).")
        elif np.any((y_int != 0) & (y_int != 1)):
            raise ValueError("La clasificación binaria simplex requiere y_true en {0, 1}.")

    folds = sorted(np.unique(f_arr).tolist())
    is_time = bool(time_mode) and (len(folds) >= 2)
    eval_folds = folds[1:] if is_time else folds
    eval_mask = np.isin(f_arr, eval_folds) if is_time else np.ones(n_samples, dtype=bool)

    uniform_weights = np.full(n_models, 1.0 / n_models)
    oof_pred = (
        np.zeros((n_samples, int(matrix.shape[2])), dtype=float)
        if matrix.ndim == 3
        else np.zeros(n_samples, dtype=float)
    )
    if is_time:
        warmup_mask = f_arr == folds[0]
        oof_pred[warmup_mask] = _simplex_predict(matrix[warmup_mask], uniform_weights)

    fold_weights: list[list[float]] = []
    fold_scores: list[float] = []
    pruned_by_fold: list[list[int]] = []
    for fold in eval_folds:
        val_mask = f_arr == fold
        tr_mask = (f_arr < fold) if is_time else (~val_mask)
        if not np.any(tr_mask):
            weights = _simplex_project(uniform_weights)
        else:
            weights = _simplex_fit_with_pruning(
                matrix[tr_mask],
                y_arr[tr_mask],
                task,
                metric,
                prune_threshold=prune_threshold,
                allow_prune=allow_prune,
            )
        fold_weights.append([float(v) for v in weights])
        pruned_by_fold.append([int(i) for i in np.where(weights == 0.0)[0]])
        val_pred = _simplex_predict(matrix[val_mask], weights)
        oof_pred[val_mask] = val_pred
        fold_scores.append(
            round(float(_simplex_score(metric, task, y_arr[val_mask], val_pred)), 8)
        )

    weights_mean = _simplex_project(np.mean(np.asarray(fold_weights, dtype=float), axis=0))
    y_eval = y_arr[eval_mask]
    matrix_eval = matrix[eval_mask]
    score_uniform = float(
        _simplex_score(metric, task, y_eval, _simplex_predict(matrix_eval, uniform_weights))
    )

    individual_scores = []
    for idx in range(n_models):
        vertex = np.zeros(n_models)
        vertex[idx] = 1.0
        individual_scores.append(
            float(_simplex_score(metric, task, y_eval, _simplex_predict(matrix_eval, vertex)))
        )
    best_single_index = int(np.argmin(individual_scores))
    score_best_single = float(individual_scores[best_single_index])
    score_blended = float(_simplex_score(metric, task, y_eval, oof_pred[eval_mask]))

    zero_in_all_folds = [
        int(idx)
        for idx in range(n_models)
        if all(row[idx] == 0.0 for row in fold_weights)
    ]
    active_models = [int(i) for i in np.where(weights_mean > 0.0)[0]]

    return {
        "task": task,
        "metric": metric,
        "time_mode": is_time,
        "eval_folds": [int(fold) for fold in eval_folds],
        "weights": [float(v) for v in weights_mean],
        "fold_weights": fold_weights,
        "uniform_weights": [float(v) for v in uniform_weights],
        "best_single_index": best_single_index,
        "best_single_score": round(score_best_single, 8),
        "score_uniform": round(score_uniform, 8),
        "score_best_single": round(score_best_single, 8),
        "score_blended": round(score_blended, 8),
        "fold_scores": fold_scores,
        "active_models": active_models,
        "pruned_models": zero_in_all_folds,
        "pruned_models_by_fold": pruned_by_fold,
        "oof_pred": oof_pred,
        "evaluation_scope": "foldwise_adaptive_procedure_not_frozen_weights",
    }

__all__ = ["optimize_simplex_blend_foldwise"]
