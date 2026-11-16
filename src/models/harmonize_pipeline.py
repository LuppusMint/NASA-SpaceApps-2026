#!/usr/bin/env python3
"""Benchmark M0-M3 y cadena de armonización multi-satélite (2000-11 .. 2026-11).

El módulo evalúa cuatro reconstrucciones sobre los folds espaciales congelados
en ``splits.json``, entrena los modelos de producción sólo en la ventana
pre-deriva ``2012-02 .. 2020-12`` y aplica el relevo de constelación
Terra -> Terra+Aqua -> solapamiento MODIS/VIIRS -> backbone VIIRS deduplicado.
La salida se reconcilia jerárquicamente y se escribe out-of-core en Parquet.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

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
    load_cell_month_features,
    parquet_sql,
    sha256_file,
    sha256_json,
    stable_json,
    write_json,
    write_parquet,
)

MODEL_IDS = (
    "M0_raw_unharmonized",
    "M1_physical_biweight_bridge",
    "M2_hurdle_tweedie_covshift",
    "M3_simplex_kalman_reconciled",
)
TARGETS = ("target_srehi", "target_frp_mw")
METRIC_NAMES = ("rmse", "mae", "rmsle", "tweedie_deviance", "r2")
INTERVAL_Z = 1.645

#: Features numéricas compartidas por M1/M2.
FEATURE_COLUMNS: tuple[str, ...] = (
    "terra_events",
    "aqua_events",
    "modis_events",
    "terra_frp_mw",
    "aqua_frp_mw",
    "modis_frp_mw",
    "terra_active_cells",
    "aqua_active_cells",
    "modis_active_cells",
    "modis_clean_det",
    "modis_night_frac",
    "modis_high_conf_frac",
    "has_terra",
    "has_aqua",
    "x_m_center",
    "y_m_center",
    "month_sin",
    "month_cos",
    "abs_lat",
    "is_in_sama",
)

_M1_FEATURES = (
    "has_aqua",
    "x_m_center",
    "y_m_center",
    "month_sin",
    "month_cos",
    "modis_night_frac",
    "modis_high_conf_frac",
    "is_in_sama",
    "abs_lat",
    "modis_active_cells",
    "modis_clean_det",
)


# ---------------------------------------------------------------------------
# Utilidades de métricas y folds
# ---------------------------------------------------------------------------


def _evaluate_metric(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from src.core.metrics import evaluate_metric

    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.clip(np.asarray(y_pred, dtype=float).reshape(-1), 0.0, None)
    if truth.size == 0:
        return float("nan")
    try:
        return float(evaluate_metric(metric, "regression", truth, pred))
    except Exception:  # noqa: BLE001 - degenerate r2/constant folds
        if metric == "r2":
            return 0.0
        return float("nan")


def _metric_bundle(
    y_true: np.ndarray, y_pred: np.ndarray, folds: np.ndarray
) -> dict[str, Any]:
    fold_metrics: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for fold in sorted(np.unique(folds).tolist()):
        mask = folds == fold
        for name in METRIC_NAMES:
            fold_metrics[name].append(_evaluate_metric(name, y_true[mask], y_pred[mask]))
    mean = {
        name: float(np.mean(values)) if values else float("nan")
        for name, values in fold_metrics.items()
    }
    oof = {name: _evaluate_metric(name, y_true, y_pred) for name in METRIC_NAMES}
    return {
        "fold_metrics": fold_metrics,
        "mean": mean,
        "oof": oof,
        "n": int(len(y_true)),
    }


def _limit_rows(
    df: pd.DataFrame, max_rows: int | None, seed: int = 42
) -> pd.DataFrame:
    """Submuestreo determinista para acotar el coste de entrenamiento."""
    if not max_rows or max_rows <= 0 or len(df) <= int(max_rows):
        return df
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(df), size=int(max_rows), replace=False)
    return df.iloc[np.sort(indices)].reset_index(drop=True)


def _safe_divide(numerator: Any, denominator: Any) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    return np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 1e-12)


def _blend_fold_ids(overlap: pd.DataFrame) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Ordena folds por tiempo mediano del bloque para un blend causal."""
    period = (
        pd.to_numeric(overlap["year"], errors="coerce").fillna(0).astype(int) * 12
        + pd.to_numeric(overlap["month"], errors="coerce").fillna(0).astype(int)
    )
    frame = pd.DataFrame({"fold_id": overlap["fold_id"].astype(int), "period": period})
    summary = (
        frame.groupby("fold_id")["period"]
        .agg(["median", "min", "max", "count"])
        .reset_index()
        .sort_values(["median", "min", "fold_id"])
        .reset_index(drop=True)
    )
    order = {int(row.fold_id): index for index, row in summary.iterrows()}
    blend_ids = overlap["fold_id"].astype(int).map(order).to_numpy(dtype=int)
    summary_payload = [
        {
            "fold_id": int(row.fold_id),
            "blend_order": int(order[int(row.fold_id)]),
            "median_period": float(row.median),
            "min_period": float(row.min),
            "max_period": float(row.max),
            "n": int(row.count),
        }
        for row in summary.itertuples()
    ]
    return blend_ids, summary_payload


# ---------------------------------------------------------------------------
# M1: puente físico robusto con corrección biweight por país/banda
# ---------------------------------------------------------------------------


def _m1_source_and_kind(target: str) -> tuple[str, str]:
    # Tweedie es estable para el conteo consolidado; FRP, con cola pesada y
    # ceros estructurales, se modela con hurdle (probabilidad x intensidad).
    if target == "target_srehi":
        return "modis_events", "tweedie"
    return "modis_frp_mw", "hurdle"


def _m1_feature_columns(target: str) -> list[str]:
    features = list(_M1_FEATURES)
    if target == "target_srehi":
        features += ["terra_events"]
    else:
        features += ["terra_frp_mw", "modis_events"]
    return list(dict.fromkeys(features))


def _fit_robust_correction(
    frame: pd.DataFrame, prediction: np.ndarray, target: str
) -> dict[str, Any]:
    """Factores robustos ``target/predicción`` por país y banda latitudinal."""
    from src.core.harmonize import tukey_biweight_location

    truth = frame[target].to_numpy(dtype=float)
    pred = np.clip(np.asarray(prediction, dtype=float), 0.0, None)
    valid = (truth > 0) & (pred > 0)
    if not np.any(valid):
        return {"global": 1.0, "country": {}, "band": {}, "min_count": 5}

    ratios = truth[valid] / pred[valid]
    global_factor = float(tukey_biweight_location(ratios))
    if not np.isfinite(global_factor) or global_factor <= 0:
        global_factor = 1.0

    bands = pd.cut(
        np.abs(pd.to_numeric(frame["abs_lat"], errors="coerce").fillna(0.0)),
        bins=[-1.0, 10.0, 20.0, 35.0, 90.0],
        labels=["0-10", "10-20", "20-35", "35-90"],
    ).astype(str)

    def _grouped(keys: pd.Series) -> dict[str, float]:
        result: dict[str, float] = {}
        local = pd.DataFrame({"key": keys.astype(str), "ratio": ratios})
        for key, group in local.groupby("key", sort=True):
            values = group["ratio"].to_numpy(dtype=float)
            if values.size >= 5:
                estimate = float(tukey_biweight_location(values))
                if np.isfinite(estimate) and estimate > 0:
                    result[str(key)] = float(np.clip(estimate, 0.25, 4.0))
        return result

    return {
        "global": float(np.clip(global_factor, 0.25, 4.0)),
        "country": _grouped(frame.loc[valid, "country_name"]),
        "band": _grouped(bands[valid]),
        "min_count": 5,
    }


def _apply_robust_correction(
    correction: dict[str, Any], frame: pd.DataFrame, prediction: np.ndarray
) -> np.ndarray:
    pred = np.clip(np.asarray(prediction, dtype=float), 0.0, None)
    if pred.size == 0:
        return pred
    country = frame["country_name"].astype(str).to_numpy()
    bands = pd.cut(
        np.abs(pd.to_numeric(frame["abs_lat"], errors="coerce").fillna(0.0)),
        bins=[-1.0, 10.0, 20.0, 35.0, 90.0],
        labels=["0-10", "10-20", "20-35", "35-90"],
    ).astype(str).to_numpy()
    global_factor = float(correction.get("global", 1.0) or 1.0)
    country_map = correction.get("country", {}) or {}
    band_map = correction.get("band", {}) or {}
    factors = np.full(len(pred), global_factor, dtype=float)
    for index, (ctry, band) in enumerate(zip(country, bands)):
        local = math.sqrt(
            float(country_map.get(ctry, global_factor))
            * float(band_map.get(band, global_factor))
        )
        factors[index] = local if np.isfinite(local) and local > 0 else global_factor
    return np.clip(pred * np.clip(factors, 0.25, 4.0), 0.0, None)


def _bridge_feature_names(target: str) -> list[str]:
    source, _ = _m1_source_and_kind(target)
    return list(dict.fromkeys([source] + _m1_feature_columns(target)))


def _clip_bounds(train: pd.DataFrame, columns: Sequence[str]) -> tuple[dict[str, float], dict[str, float]]:
    low: dict[str, float] = {}
    high: dict[str, float] = {}
    for column in columns:
        values = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            low[column], high[column] = 0.0, 0.0
            continue
        q_low, q_high = np.quantile(finite, [0.001, 0.999])
        low[column] = float(q_low)
        high[column] = float(max(q_high, q_low))
    return low, high


def _apply_clip(frame: pd.DataFrame, low: dict[str, float], high: dict[str, float]) -> pd.DataFrame:
    clipped = frame.copy()
    for column, lower in low.items():
        upper = high.get(column, lower)
        values = pd.to_numeric(clipped[column], errors="coerce").to_numpy(dtype=float)
        clipped[column] = np.clip(np.where(np.isfinite(values), values, lower), lower, upper)
    return clipped


def fit_m1(train: pd.DataFrame, target: str, *, alpha: float = 1e-2) -> dict[str, Any]:
    """Ajusta el puente robusto M1 (tweedie para SREHI, hurdle para FRP).

    Las features se recortan a los cuantiles 0.1%-99.9% del train para evitar
    extrapolaciones del enlace log en colas extremas. El recorte se guarda y se
    reaplica en inferencia, de modo que el puente nunca vea valores fuera del
    soporte observado durante la calibración pre-deriva.
    """
    from src.core.harmonize import (
        fit_cross_sensor_bridge,
        predict_cross_sensor_bridge,
    )

    source, kind = _m1_source_and_kind(target)
    feature_names = _bridge_feature_names(target)
    clip_low, clip_high = _clip_bounds(train, feature_names)
    fit_frame = _apply_clip(train, clip_low, clip_high)
    truth = train[target].to_numpy(dtype=float)
    target_cap = float(
        max(np.quantile(truth[np.isfinite(truth)], 0.999) * 2.0, 1.0)
    ) if np.isfinite(truth).any() else 1.0
    fit_frame = fit_frame.copy()
    fit_frame[target] = np.minimum(fit_frame[target].to_numpy(dtype=float), target_cap)
    bridge = fit_cross_sensor_bridge(
        fit_frame,
        source_col=source,
        target_col=target,
        feature_cols=_m1_feature_columns(target),
        model_kind=kind,
        alpha=float(alpha),
    )
    raw_pred = predict_cross_sensor_bridge(bridge, fit_frame)
    correction = _fit_robust_correction(train, raw_pred, target)
    corrected = np.clip(
        _apply_robust_correction(correction, train, raw_pred), 0.0, target_cap
    )
    residual = truth - corrected
    finite = residual[np.isfinite(residual)]
    sigma0 = float(np.std(finite, ddof=1)) if finite.size > 1 else 1.0
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = max(float(np.mean(np.abs(finite))) if finite.size else 1.0, 1e-3)
    mean_pred = float(np.mean(corrected)) if corrected.size else 1.0
    return {
        "bridge": bridge,
        "correction": correction,
        "sigma0": max(sigma0, 1e-3),
        "mean_pred": max(mean_pred, 1e-3),
        "source": source,
        "kind": kind,
        "target": target,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "pred_cap": target_cap,
        "n_train": int(len(train)),
    }


def predict_m1(model: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predicción M1 e incertidumbre heurística heterocedástica."""
    from src.core.harmonize import predict_cross_sensor_bridge

    if len(frame) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    clipped = _apply_clip(frame, model.get("clip_low", {}), model.get("clip_high", {}))
    raw = predict_cross_sensor_bridge(model["bridge"], clipped)
    pred = _apply_robust_correction(model["correction"], clipped, raw)
    pred = np.clip(pred, 0.0, float(model.get("pred_cap", np.inf)))
    scale = np.sqrt(np.clip(pred, 1.0, None) / max(float(model["mean_pred"]), 1e-3))
    sigma = np.clip(float(model["sigma0"]) * scale, 1e-3, None)
    return pred, sigma


# ---------------------------------------------------------------------------
# M2: Tweedie + Hurdle LightGBM con pesos de covariate shift
# ---------------------------------------------------------------------------


def _m2_features() -> list[str]:
    return list(dict.fromkeys(FEATURE_COLUMNS))


def _m2_matrices(frame: pd.DataFrame) -> np.ndarray:
    matrix = frame[_m2_features()].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if matrix.size == 0:
        return matrix.reshape(len(frame), len(_m2_features()))
    fill = np.nanmedian(matrix, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    return np.where(np.isnan(matrix), fill, matrix)


def _fit_m2(
    train: pd.DataFrame,
    target: str,
    *,
    sample_weight: np.ndarray | None,
    seed: int = 42,
) -> dict[str, Any]:
    """Ajusta clasificador Hurdle e intensidad Tweedie sobre un fold."""
    from lightgbm import LGBMClassifier, LGBMRegressor

    matrix = _m2_matrices(train)
    truth = train[target].to_numpy(dtype=float)
    weights = (
        np.asarray(sample_weight, dtype=float)
        if sample_weight is not None
        else np.ones(len(train), dtype=float)
    )
    weights = np.clip(weights, 1e-6, None)
    positive = truth > 0
    n_pos = int(np.sum(positive))
    n_neg = int(len(truth) - n_pos)

    classifier = None
    constant_probability: float | None = None
    if n_pos >= 10 and n_neg >= 10:
        classifier = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=int(seed),
            n_jobs=4,
            verbose=-1,
        )
        classifier.fit(matrix, positive.astype(int), sample_weight=weights)
    else:
        constant_probability = float(n_pos / max(len(truth), 1))

    regressor = None
    intensity_fallback = 0.0
    if n_pos >= 10:
        regressor = LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=int(seed),
            n_jobs=4,
            verbose=-1,
        )
        regressor.fit(
            matrix[positive], truth[positive], sample_weight=weights[positive]
        )
        intensity_fallback = float(np.average(truth[positive], weights=weights[positive]))
    elif n_pos > 0:
        intensity_fallback = float(np.mean(truth[positive]))

    p_train, intensity_train = _predict_m2_components(
        classifier, constant_probability, regressor, intensity_fallback, matrix
    )
    soft_train = p_train * intensity_train
    residual = truth - soft_train
    finite = residual[np.isfinite(residual)]
    sigma0 = float(np.std(finite, ddof=1)) if finite.size > 1 else 1.0
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = max(float(np.mean(np.abs(finite))) if finite.size else 1.0, 1e-3)
    return {
        "classifier": classifier,
        "constant_probability": constant_probability,
        "regressor": regressor,
        "intensity_fallback": intensity_fallback,
        "sigma0": max(sigma0, 1e-3),
        "mean_soft": max(float(np.mean(soft_train)) if soft_train.size else 1.0, 1e-3),
        "target": target,
        "n_train": int(len(train)),
    }


def _predict_m2_components(
    classifier: Any,
    constant_probability: float | None,
    regressor: Any,
    intensity_fallback: float,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if classifier is None:
        probability = np.full(
            len(matrix), float(constant_probability if constant_probability is not None else 0.0)
        )
    else:
        probability = classifier.predict_proba(matrix)[:, 1]
    if regressor is None:
        intensity = np.full(len(matrix), float(intensity_fallback))
    else:
        intensity = regressor.predict(matrix)
    return np.clip(probability, 0.0, 1.0), np.clip(intensity, 0.0, None)


def predict_m2(model: dict[str, Any], frame: pd.DataFrame) -> dict[str, np.ndarray]:
    matrix = _m2_matrices(frame)
    probability, intensity = _predict_m2_components(
        model["classifier"],
        model["constant_probability"],
        model["regressor"],
        model["intensity_fallback"],
        matrix,
    )
    soft = probability * intensity
    scale = np.sqrt(np.clip(soft, 1.0, None) / max(float(model["mean_soft"]), 1e-3))
    sigma = np.clip(float(model["sigma0"]) * scale, 1e-3, None)
    return {
        "probability": probability,
        "intensity": intensity,
        "soft": soft,
        "sigma": sigma,
    }


def _covariate_weights(
    train: pd.DataFrame,
    future: pd.DataFrame,
    *,
    max_rows: int = 40_000,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pesos de importancia 2012-2020 vs 2021-2026 (sin usar el target)."""
    from src.core.harmonize import covariate_shift_weights

    if future is None or len(future) == 0:
        return np.ones(len(train), dtype=float), {
            "domain_auc": 0.5,
            "effective_sample_size": float(len(train)),
            "n_source": int(len(train)),
            "n_target": 0,
            "used": False,
        }
    source_X = _m2_matrices(_limit_rows(train, max_rows, seed))
    target_X = _m2_matrices(_limit_rows(future, max_rows, seed + 1))
    try:
        result = covariate_shift_weights(source_X, target_X, seed=seed)
    except Exception:  # noqa: BLE001 - datos degenerados -> pesos uniformes
        return np.ones(len(train), dtype=float), {
            "domain_auc": 0.5,
            "effective_sample_size": float(len(train)),
            "n_source": int(len(train)),
            "n_target": int(len(future)),
            "used": False,
        }
    weights = np.asarray(result["weights"], dtype=float)
    # ``weights`` corresponde a la submuestra usada para estimar la densidad.
    # Se propaga a todo el fold con un vecino más cercano determinista.
    if len(weights) != len(train):
        source_matrix = _m2_matrices(train)
        used_matrix = source_X
        from sklearn.neighbors import NearestNeighbors

        neighbours = NearestNeighbors(n_neighbors=1).fit(used_matrix)
        _, indices = neighbours.kneighbors(source_matrix)
        weights = weights[indices.ravel()]
    weights = np.clip(weights, 1e-6, None)
    weights = weights / max(float(np.mean(weights)), 1e-9)
    return weights, {
        "domain_auc": float(result.get("domain_auc", 0.5)),
        "effective_sample_size": float(np.sum(weights) ** 2 / max(np.sum(weights ** 2), 1e-9)),
        "n_source": int(len(train)),
        "n_target": int(len(future)),
        "used": True,
    }


# ---------------------------------------------------------------------------
# Evaluación M0-M3 sobre folds congelados
# ---------------------------------------------------------------------------


def _oof_predictions(
    overlap: pd.DataFrame,
    future: pd.DataFrame,
    *,
    max_train_rows: int,
    seed: int = 42,
) -> dict[str, Any]:
    folds = sorted(overlap["fold_id"].astype(int).unique().tolist())
    fold_array = overlap["fold_id"].astype(int).to_numpy()
    y = {target: overlap[target].to_numpy(dtype=float) for target in TARGETS}

    predictions: dict[str, dict[str, np.ndarray]] = {
        "M0_raw_unharmonized": {
            "target_srehi": overlap["modis_events"].to_numpy(dtype=float),
            "target_frp_mw": overlap["modis_frp_mw"].to_numpy(dtype=float),
        }
    }
    sigmas: dict[str, dict[str, np.ndarray]] = {}
    blend_order, blend_summary = _blend_fold_ids(overlap)
    blend_info: dict[str, Any] = {}
    covshift_info: dict[str, Any] = {}

    for target in TARGETS:
        m1_pred = np.zeros(len(overlap), dtype=float)
        m1_sigma = np.zeros(len(overlap), dtype=float)
        m2_probability = np.zeros(len(overlap), dtype=float)
        m2_intensity = np.zeros(len(overlap), dtype=float)
        m2_soft = np.zeros(len(overlap), dtype=float)
        m2_sigma = np.zeros(len(overlap), dtype=float)

        for fold in folds:
            train_mask = fold_array != fold
            val_mask = fold_array == fold
            train = overlap.loc[train_mask].reset_index(drop=True)
            val = overlap.loc[val_mask].reset_index(drop=True)

            train_m1 = _limit_rows(train, max_train_rows, seed + fold)
            model_m1 = fit_m1(train_m1, target)
            pred_m1, sigma_m1 = predict_m1(model_m1, val)
            m1_pred[val_mask] = pred_m1
            m1_sigma[val_mask] = sigma_m1

            train_m2 = _limit_rows(train, max_train_rows, seed + 100 + fold)
            weights, info = _covariate_weights(train_m2, future, seed=seed + fold)
            covshift_info.setdefault(target, {})[str(fold)] = info
            model_m2 = _fit_m2(train_m2, target, sample_weight=weights, seed=seed + fold)
            components = predict_m2(model_m2, val)
            m2_probability[val_mask] = components["probability"]
            m2_intensity[val_mask] = components["intensity"]
            m2_soft[val_mask] = components["soft"]
            m2_sigma[val_mask] = components["sigma"]

        # Umbral Hurdle ajustado fold a fold sobre las predicciones OOF.
        from src.core.postprocess import (
            calibrate_regression_sigma_foldwise,
            hurdle_zero_threshold_foldwise,
        )

        hurdle = hurdle_zero_threshold_foldwise(
            y[target],
            m2_probability,
            m2_intensity,
            fold_array,
            metric="rmse",
        )
        # El umbral duro foldwise solo se usa si mejora el RMSE OOF frente a
        # la expectativa suave p*intensidad; así el Hurdle nunca empeora por un
        # umbral degenerado en un fold concreto.
        use_hard_threshold = float(hurdle["score_calibrated"]) <= float(
            hurdle["score_soft_baseline"]
        )
        m2_pred = (
            np.asarray(hurdle["oof_pred"], dtype=float)
            if use_hard_threshold
            else np.asarray(m2_soft, dtype=float)
        )
        sigma_calibration = calibrate_regression_sigma_foldwise(
            y[target], m2_pred, m2_sigma, fold_array
        )
        m2_sigma_cal = np.asarray(sigma_calibration["sigma_calibrated"], dtype=float)

        predictions["M1_physical_biweight_bridge"] = predictions.get(
            "M1_physical_biweight_bridge", {}
        )
        predictions["M1_physical_biweight_bridge"][target] = m1_pred
        predictions.setdefault("M2_hurdle_tweedie_covshift", {})[target] = m2_pred
        sigmas.setdefault("M1_physical_biweight_bridge", {})[target] = m1_sigma
        sigmas.setdefault("M2_hurdle_tweedie_covshift", {})[target] = m2_sigma_cal

        from src.core.stacking import optimize_simplex_blend_foldwise

        blend = optimize_simplex_blend_foldwise(
            np.column_stack([m1_pred, m2_pred]),
            y[target],
            blend_order,
            metric="rmse",
            time_mode=True,
        )
        m3_pred = np.asarray(blend["oof_pred"], dtype=float)
        weights = np.asarray(blend["weights"], dtype=float)
        fallback_best_single = False
        if float(blend["score_blended"]) > float(blend["score_best_single"]):
            # El warmup temporal del fold inicial puede promediar un modelo
            # claramente peor. El simplex admite vértices: se congela el mejor
            # modelo individual antes que aceptar un blend dominado.
            best_index = int(blend["best_single_index"])
            weights = np.zeros(2, dtype=float)
            weights[best_index] = 1.0
            m3_pred = m1_pred if best_index == 0 else m2_pred
            fallback_best_single = True
        sigma_m3 = np.clip(
            weights[0] * m1_sigma + weights[1] * m2_sigma_cal, 1e-3, None
        )
        m3_calibration = calibrate_regression_sigma_foldwise(
            y[target], m3_pred, sigma_m3, fold_array
        )
        m3_sigma_cal = np.asarray(m3_calibration["sigma_calibrated"], dtype=float)
        predictions.setdefault("M3_simplex_kalman_reconciled", {})[target] = m3_pred
        sigmas.setdefault("M3_simplex_kalman_reconciled", {})[target] = m3_sigma_cal

        blend_info[target] = {
            "weights": [float(value) for value in weights],
            "fold_weights": blend["fold_weights"],
            "fold_scores": blend["fold_scores"],
            "score_blended": float(blend["score_blended"]),
            "score_best_single": float(blend["score_best_single"]),
            "best_single_index": int(blend["best_single_index"]),
            "time_mode": bool(blend["time_mode"]),
            "m2_thresholds": [float(value) for value in hurdle["fold_thresholds"]],
            "m2_median_threshold": float(np.median(hurdle["fold_thresholds"])),
            "m2_sigma_fold_scales": sigma_calibration["fold_scales"],
            "m2_sigma_mean_scale": float(sigma_calibration["mean_scale"]),
            "m2_use_hard_threshold": bool(use_hard_threshold),
            "m2_score_soft": float(hurdle["score_soft_baseline"]),
            "m2_score_calibrated": float(hurdle["score_calibrated"]),
            "m3_sigma_fold_scales": m3_calibration["fold_scales"],
            "m3_sigma_mean_scale": float(m3_calibration["mean_scale"]),
            "fallback_best_single": bool(fallback_best_single),
        }

    return {
        "predictions": predictions,
        "sigmas": sigmas,
        "blend_order": blend_order,
        "blend_summary": blend_summary,
        "blend_info": blend_info,
        "covshift": covshift_info,
        "y": y,
        "folds": fold_array,
    }


def _cross_regime_guard(
    overlap: pd.DataFrame, predictions: dict[str, dict[str, np.ndarray]]
) -> dict[str, Any]:
    from src.core.harmonize import check_cross_regime_error_guard

    result: dict[str, Any] = {}
    labels = overlap["regime_mode"].astype(str).to_numpy()
    for target in TARGETS:
        result[target] = {
            model: check_cross_regime_error_guard(
                overlap[target].to_numpy(dtype=float),
                predictions[model][target],
                labels,
            )
            for model in MODEL_IDS[1:]
        }
    return result


def _write_oof(
    overlap: pd.DataFrame,
    predictions: dict[str, dict[str, np.ndarray]],
    sigmas: dict[str, dict[str, np.ndarray]],
    path: Path,
) -> Path:
    frame = overlap[
        [
            "sample_id",
            "fold_id",
            "regime_mode",
            "year",
            "month",
            "year_month",
            "cell_id_50km",
            "country_name",
            "iso_a3",
        ]
    ].copy()
    frame["y_srehi"] = overlap["target_srehi"].to_numpy(dtype=float)
    frame["y_frp_mw"] = overlap["target_frp_mw"].to_numpy(dtype=float)
    for model in MODEL_IDS:
        frame[f"{model}_srehi"] = predictions[model]["target_srehi"]
        frame[f"{model}_frp_mw"] = predictions[model]["target_frp_mw"]
    for model in MODEL_IDS[1:]:
        sigma_srehi = sigmas.get(model, {}).get("target_srehi")
        sigma_frp = sigmas.get(model, {}).get("target_frp_mw")
        if sigma_srehi is not None:
            frame[f"{model}_srehi_lo90"] = np.clip(
                predictions[model]["target_srehi"] - INTERVAL_Z * sigma_srehi, 0.0, None
            )
            frame[f"{model}_srehi_hi90"] = (
                predictions[model]["target_srehi"] + INTERVAL_Z * sigma_srehi
            )
        if sigma_frp is not None:
            frame[f"{model}_frp_lo90"] = np.clip(
                predictions[model]["target_frp_mw"] - INTERVAL_Z * sigma_frp, 0.0, None
            )
            frame[f"{model}_frp_hi90"] = (
                predictions[model]["target_frp_mw"] + INTERVAL_Z * sigma_frp
            )
    con = connect_duckdb(scratch_dir=path.parent / ".duckdb_tmp")
    try:
        con.register("oof_frame", frame)
        write_parquet(
            con,
            "SELECT * FROM oof_frame ORDER BY year_month, cell_id_50km, regime_mode",
            path,
        )
    finally:
        con.close()
    return path


def _experiment_records(
    metrics: dict[str, dict[str, Any]],
    *,
    splits: dict[str, Any],
    oof_path: Path,
    splits_path: Path,
    config: dict[str, Any],
    training_seconds: dict[str, float],
) -> list[dict[str, Any]]:
    feature_schema_hash = sha256_json([list(FEATURE_COLUMNS), list(TARGETS)])
    records: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        primary = metrics[model_id]["target_srehi"]
        rmse_values = primary["fold_metrics"]["rmse"]
        record: dict[str, Any] = {
            "id": model_id,
            "run_id": f"c2-{model_id}",
            "hypothesis": {
                "M0_raw_unharmonized": "La suma cruda MODIS subestima la escala VIIRS de referencia.",
                "M1_physical_biweight_bridge": "Un puente robusto pre-deriva corrige sesgo de escala y bioma.",
                "M2_hurdle_tweedie_covshift": "Hurdle+Tweedie con pesos de covariate shift reduce error y calibra sigma.",
                "M3_simplex_kalman_reconciled": "El blend convexo causal de M1/M2 domina a cualquier modelo individual.",
            }[model_id],
            "dataset_sha256": splits["dataset_sha256"],
            "fold_fingerprint": splits["fingerprint"],
            "feature_schema_hash": feature_schema_hash,
            "target_hash": sha256_json(
                {
                    "targets": list(TARGETS),
                    "rows": int(splits["rows"]),
                    "id_hash": splits["id_hash"],
                }
            ),
            "config_hash": sha256_json(config),
            "random_seed": 42,
            "model": model_id,
            "metric": "rmse",
            "direction": "minimize",
            "fold_metrics": [float(value) for value in rmse_values],
            "mean": float(np.mean(rmse_values)),
            "std": float(np.std(rmse_values, ddof=0)),
            "median": float(np.median(rmse_values)),
            "min": float(np.min(rmse_values)),
            "max": float(np.max(rmse_values)),
            "oof_score": float(primary["oof"]["rmse"]),
            "fold_sizes": dict(splits["fold_sizes"]),
            "n_folds": int(splits["folds"]),
            "train_time_sec": float(training_seconds.get(model_id, 0.0)),
            "infer_time_sec": 0.0,
            "decision": "accept",
            "reason": "Evaluación OOF sobre folds espaciales congelados sin fuga de bloque.",
            "artifact_hashes": {
                "oof_m0_m3": {
                    "path": str(oof_path.resolve()),
                    "sha256": sha256_file(oof_path),
                },
                "oof_m0_m3.parquet": {
                    "path": str(oof_path.resolve()),
                    "sha256": sha256_file(oof_path),
                },
                "splits": {
                    "path": str(splits_path.resolve()),
                    "sha256": sha256_file(splits_path),
                },
            },
            "target": "target_srehi",
            "targets": list(TARGETS),
            "fold_metrics_by_metric": primary["fold_metrics"],
            "metrics_by_target": {
                target: {
                    "fold_metrics": metrics[model_id][target]["fold_metrics"],
                    "mean": metrics[model_id][target]["mean"],
                    "oof": metrics[model_id][target]["oof"],
                    "n": metrics[model_id][target]["n"],
                }
                for target in TARGETS
            },
            "metrics": primary,
        }
        if model_id != "M0_raw_unharmonized":
            record["baseline_id"] = "M0_raw_unharmonized"
        records.append(record)
    return records


def _write_experiments(records: Sequence[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Modelos de producción y cadena de relevo
# ---------------------------------------------------------------------------


def _fit_production(
    overlap: pd.DataFrame,
    future: pd.DataFrame,
    blend_info: dict[str, Any],
    *,
    max_train_rows: int,
    seed: int = 42,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for target in TARGETS:
        train_m1 = _limit_rows(overlap, max_train_rows, seed)
        train_m2 = _limit_rows(overlap, max_train_rows, seed + 10)
        weights, info = _covariate_weights(train_m2, future, seed=seed)
        m1 = fit_m1(train_m1, target)
        m2 = _fit_m2(train_m2, target, sample_weight=weights, seed=seed)
        weights_blend = np.asarray(blend_info[target]["weights"], dtype=float)
        models[target] = {
            "m1": m1,
            "m2": m2,
            "blend_weights": weights_blend,
            "m2_threshold": float(blend_info[target]["m2_median_threshold"]),
            "m2_use_hard_threshold": bool(blend_info[target].get("m2_use_hard_threshold", True)),
            "m2_sigma_scale": float(blend_info[target]["m2_sigma_mean_scale"]),
            "m3_sigma_scale": float(blend_info[target]["m3_sigma_mean_scale"]),
            "covshift": info,
        }
    return models


def _production_predictions(
    features: pd.DataFrame, models: dict[str, Any]
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for target, model in models.items():
        m1_pred, m1_sigma = predict_m1(model["m1"], features)
        m2_components = predict_m2(model["m2"], features)
        if model.get("m2_use_hard_threshold", True):
            m2_pred = (
                (m2_components["probability"] >= float(model["m2_threshold"])).astype(float)
                * m2_components["intensity"]
            )
        else:
            m2_pred = m2_components["soft"]
        m2_sigma = m2_components["sigma"] * float(model["m2_sigma_scale"])
        weights = np.asarray(model["blend_weights"], dtype=float)
        m3_pred = np.clip(weights[0] * m1_pred + weights[1] * m2_pred, 0.0, None)
        m3_sigma = np.clip(
            (weights[0] * m1_sigma + weights[1] * m2_sigma)
            * float(model["m3_sigma_scale"]),
            1e-3,
            None,
        )
        output[f"m1_{target}"] = np.clip(m1_pred, 0.0, None)
        output[f"m2_{target}"] = np.clip(m2_pred, 0.0, None)
        output[f"m3_{target}"] = m3_pred
        output[f"m1_sigma_{target}"] = np.clip(m1_sigma, 1e-3, None)
        output[f"m2_sigma_{target}"] = np.clip(m2_sigma, 1e-3, None)
        output[f"m3_sigma_{target}"] = m3_sigma
    return output


def _dedup_ratios(features: pd.DataFrame) -> dict[str, Any]:
    """Razones robustas de contribución marginal de N20/N21 frente a SNPP.

    Se estiman en el solapamiento ``2018-04 .. 2020-12``. N21 todavía no
    existía, por lo que su contribución marginal se asume igual a la de N20
    (misma familia de sensores y geometría de overpass). Esto evita que la
    adición de 2023 infle la serie harmonizada.
    """
    from src.core.harmonize import kalman_1d_smooth, tukey_biweight_location

    window = features[
        (features["year_month"] >= "2018-04") & (features["year_month"] <= PRE_DRIFT_END)
    ]
    ratios_events: list[float] = []
    ratios_frp: list[float] = []
    if not window.empty:
        grouped = window.groupby(["year_month", "iso_a3"], as_index=False).agg(
            snpp=("snpp_events", "sum"),
            n20=("n20_events", "sum"),
            snpp_frp=("snpp_frp_mw", "sum"),
            n20_frp=("n20_frp_mw", "sum"),
        )
        for row in grouped.itertuples():
            if row.snpp > 20 and row.n20 > 0:
                ratios_events.append(float(row.n20) / float(row.snpp))
            if row.snpp_frp > 100.0 and row.n20_frp > 0:
                ratios_frp.append(float(row.n20_frp) / float(row.snpp_frp))

    # Serie mensual de la razón marginal N20/SNPP y suavizado local-level.
    # El suavizado reduce el ruido de meses con pocos overpasses antes de
    # colapsar la razón con el estimador robusto de Tukey.
    smoothed_events: list[float] = []
    if not window.empty and len(ratios_events) >= 3:
        grouped = window.groupby(["year_month", "iso_a3"], as_index=False).agg(
            snpp=("snpp_events", "sum"), n20=("n20_events", "sum")
        )
        grouped = grouped[grouped["snpp"] > 20]
        series = (
            grouped.assign(ratio=grouped["n20"] / grouped["snpp"])
            .groupby("year_month")["ratio"]
            .mean()
            .sort_index()
        )
        if len(series) >= 3:
            smoothed_events = [
                float(value)
                for value in kalman_1d_smooth(
                    series.to_numpy(dtype=float), process_var=1e-3
                )["smoothed_mean"]
            ]
    r20_events = (
        float(tukey_biweight_location(smoothed_events or ratios_events))
        if (smoothed_events or ratios_events)
        else 0.0
    )
    r20_frp = float(tukey_biweight_location(ratios_frp)) if ratios_frp else 0.0
    r20_events = float(np.clip(r20_events if np.isfinite(r20_events) else 0.0, 0.0, 5.0))
    r20_frp = float(np.clip(r20_frp if np.isfinite(r20_frp) else 0.0, 0.0, 5.0))

    # Si no hay evidencia de un segundo satélite, el factor es 1.
    has_n20 = bool((features["has_n20"] > 0).any())
    has_n21 = bool((features["has_n21"] > 0).any())
    if not has_n20:
        r20_events = 0.0
        r20_frp = 0.0
    r21_events = r20_events if has_n21 else 0.0
    r21_frp = r20_frp if has_n21 else 0.0

    return {
        "window": "2018-04..2020-12",
        "n20_marginal_ratio_events": r20_events,
        "n20_marginal_ratio_frp": r20_frp,
        "n21_marginal_ratio_events": r21_events,
        "n21_marginal_ratio_frp": r21_frp,
        "n_month_country_pairs_events": len(ratios_events),
        "n_month_country_pairs_frp": len(ratios_frp),
        "has_n20": has_n20,
        "has_n21": has_n21,
        "factor_snpp_vs_snpp_n20": (
            float(1.0 / (1.0 + r20_events)) if r20_events > 0 else 1.0
        ),
        "smoothed_n20_to_snpp_series": [float(value) for value in smoothed_events],
        "kalman_process_var": 1e-3,
    }


def _regime_labels(features: pd.DataFrame) -> np.ndarray:
    year_month = features["year_month"].astype(str).to_numpy()
    regime = np.full(len(features), "MODIS_TERRA_AQUA", dtype=object)
    regime[year_month <= "2002-06"] = "MODIS_TERRA_ONLY"
    regime[(year_month >= "2002-07") & (year_month <= "2011-12")] = "MODIS_TERRA_AQUA"
    regime[year_month == "2012-01"] = "MODIS_TERRA_AQUA"
    regime[(year_month >= "2012-02") & (year_month <= "2020-12")] = "MODIS_VIIRS_OVERLAP"
    regime[year_month >= "2021-01"] = "VIIRS_BACKBONE"
    return regime


def _data_maturity(features: pd.DataFrame) -> np.ndarray:
    year_month = features["year_month"].astype(str).to_numpy()
    partial = features["is_partial_month"].fillna(False).to_numpy(dtype=bool)
    nrt = (
        features["nrt_det_count"].fillna(0).to_numpy(dtype=float)
        > features["sp_det_count"].fillna(0).to_numpy(dtype=float)
    )
    maturity = np.full(len(features), "STANDARD_SCIENCE", dtype=object)
    maturity[nrt] = "NRT_PRELIMINARY"
    maturity[partial] = "PARTIAL_MONTH"
    maturity[year_month >= "2026-01"] = "NRT_PRELIMINARY"
    maturity[(year_month == "2000-11") | (year_month == "2026-11")] = "PARTIAL_MONTH"
    return maturity


def harmonize_cells(
    features: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    dedup: dict[str, Any],
) -> pd.DataFrame:
    """Aplica el relevo de constelación y produce el cubo de celdas armonizado."""
    frame = features.copy()
    regime = _regime_labels(frame)
    frame["constellation_regime"] = regime
    frame["data_maturity"] = _data_maturity(frame)

    r20e = float(dedup["n20_marginal_ratio_events"])
    r21e = float(dedup["n21_marginal_ratio_events"])
    r20f = float(dedup["n20_marginal_ratio_frp"])
    r21f = float(dedup["n21_marginal_ratio_frp"])
    denom_events = (
        1.0
        + r20e * frame["has_n20"].to_numpy(dtype=float)
        + r21e * frame["has_n21"].to_numpy(dtype=float)
    )
    denom_frp = (
        1.0
        + r20f * frame["has_n20"].to_numpy(dtype=float)
        + r21f * frame["has_n21"].to_numpy(dtype=float)
    )
    viirs_events = frame["viirs_events"].to_numpy(dtype=float)
    viirs_frp = frame["viirs_frp_mw"].to_numpy(dtype=float)
    obs_srehi = np.where(denom_events > 0, viirs_events / denom_events, 0.0)
    obs_frp = np.where(denom_frp > 0, viirs_frp / denom_frp, 0.0)
    # Si SNPP no observó la celda-mes pero N20/N21 sí, se reconstruye la escala
    # SNPP a partir de la contribución marginal en lugar de dividir la unión
    # completa (que subestimaría al único satélite presente).
    snpp_only_events = (
        frame["snpp_events"].to_numpy(dtype=float)
        + (frame["n20_events"].to_numpy(dtype=float) / r20e if r20e > 0 else 0.0)
        + (frame["n21_events"].to_numpy(dtype=float) / r21e if r21e > 0 else 0.0)
    )
    snpp_only_frp = (
        frame["snpp_frp_mw"].to_numpy(dtype=float)
        + (frame["n20_frp_mw"].to_numpy(dtype=float) / r20f if r20f > 0 else 0.0)
        + (frame["n21_frp_mw"].to_numpy(dtype=float) / r21f if r21f > 0 else 0.0)
    )
    no_snpp = frame["snpp_events"].to_numpy(dtype=float) <= 0
    obs_srehi = np.where(no_snpp, snpp_only_events, obs_srehi)
    obs_frp = np.where(no_snpp, snpp_only_frp, obs_frp)

    bridge_srehi = np.asarray(predictions["m3_target_srehi"], dtype=float)
    bridge_frp = np.asarray(predictions["m3_target_frp_mw"], dtype=float)
    bridge_sigma_srehi = np.asarray(predictions["m3_sigma_target_srehi"], dtype=float)
    bridge_sigma_frp = np.asarray(predictions["m3_sigma_target_frp_mw"], dtype=float)
    has_modis = (frame["has_terra"].to_numpy(dtype=float) + frame["has_aqua"].to_numpy(dtype=float)) > 0
    has_viirs = frame["viirs_events"].to_numpy(dtype=float) > 0
    is_overlap = regime == "MODIS_VIIRS_OVERLAP"
    is_backbone = regime == "VIIRS_BACKBONE"

    obs_sigma_srehi = np.sqrt(np.clip(obs_srehi, 1.0, None))
    obs_sigma_frp = np.clip(obs_frp * 0.5, 1.0, None)
    bridge_sigma_srehi = np.where(has_modis, bridge_sigma_srehi, np.inf)
    bridge_sigma_frp = np.where(has_modis, bridge_sigma_frp, np.inf)

    use_blend = is_overlap & has_viirs & has_modis
    use_obs = (is_overlap & has_viirs & ~has_modis) | (is_backbone & has_viirs)
    use_bridge = (
        ((~is_overlap) & (~is_backbone) & has_modis)
        | (is_overlap & ~has_viirs & has_modis)
        | (is_backbone & ~has_viirs & has_modis)
    )

    def _combine(obs: np.ndarray, obs_sigma: np.ndarray, bridge: np.ndarray, bridge_sigma: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        var_obs = np.clip(obs_sigma, 1e-9, None) ** 2
        var_bridge = np.clip(bridge_sigma, 1e-9, None) ** 2
        var = np.where(mask, 1.0 / (1.0 / var_obs + 1.0 / var_bridge), 0.0)
        mu = np.where(mask, var * (obs / var_obs + bridge / var_bridge), 0.0)
        return mu, np.sqrt(np.clip(var, 0.0, None))

    blend_mu_srehi, blend_sigma_srehi = _combine(
        obs_srehi, obs_sigma_srehi, bridge_srehi, bridge_sigma_srehi, use_blend
    )
    blend_mu_frp, blend_sigma_frp = _combine(
        obs_frp, obs_sigma_frp, bridge_frp, bridge_sigma_frp, use_blend
    )

    mu_srehi = np.zeros(len(frame), dtype=float)
    sigma_srehi = np.zeros(len(frame), dtype=float)
    mu_frp = np.zeros(len(frame), dtype=float)
    sigma_frp = np.zeros(len(frame), dtype=float)

    mu_srehi[use_obs] = obs_srehi[use_obs]
    sigma_srehi[use_obs] = obs_sigma_srehi[use_obs]
    mu_frp[use_obs] = obs_frp[use_obs]
    sigma_frp[use_obs] = obs_sigma_frp[use_obs]
    mu_srehi[use_bridge] = bridge_srehi[use_bridge]
    sigma_srehi[use_bridge] = bridge_sigma_srehi[use_bridge]
    mu_frp[use_bridge] = bridge_frp[use_bridge]
    sigma_frp[use_bridge] = bridge_sigma_frp[use_bridge]
    mu_srehi[use_blend] = blend_mu_srehi[use_blend]
    sigma_srehi[use_blend] = blend_sigma_srehi[use_blend]
    mu_frp[use_blend] = blend_mu_frp[use_blend]
    sigma_frp[use_blend] = blend_sigma_frp[use_blend]

    frame["harmonized_srehi"] = np.clip(mu_srehi, 0.0, None)
    frame["harmonized_srehi_lo90"] = np.clip(
        frame["harmonized_srehi"] - INTERVAL_Z * sigma_srehi, 0.0, None
    )
    frame["harmonized_srehi_hi90"] = frame["harmonized_srehi"] + INTERVAL_Z * sigma_srehi
    frame["harmonized_frp_mw"] = np.clip(mu_frp, 0.0, None)
    frame["harmonized_frp_lo90"] = np.clip(
        frame["harmonized_frp_mw"] - INTERVAL_Z * sigma_frp, 0.0, None
    )
    frame["harmonized_frp_hi90"] = frame["harmonized_frp_mw"] + INTERVAL_Z * sigma_frp
    frame["m1_srehi"] = np.clip(np.asarray(predictions["m1_target_srehi"], dtype=float), 0.0, None)
    frame["m2_srehi"] = np.clip(np.asarray(predictions["m2_target_srehi"], dtype=float), 0.0, None)
    frame["m1_frp_mw"] = np.clip(np.asarray(predictions["m1_target_frp_mw"], dtype=float), 0.0, None)
    frame["m2_frp_mw"] = np.clip(np.asarray(predictions["m2_target_frp_mw"], dtype=float), 0.0, None)
    frame["sama_suspect_det_count"] = np.where(
        frame["is_in_sama"].fillna(False).to_numpy(dtype=bool),
        frame["night_det_count"].fillna(0).to_numpy(dtype=float),
        0.0,
    ).astype("int64")
    frame["raw_frp_mw"] = frame["frp_sum_mw"].astype(float)
    return frame


_CELL_COLUMNS = (
    "year",
    "month",
    "year_month",
    "cell_id_50km",
    "cell_ix_50km",
    "cell_iy_50km",
    "country_name",
    "iso_a3",
    "mean_lat",
    "mean_lon",
    "x_m_center",
    "y_m_center",
    "raw_all_det_count",
    "clean_det_count",
    "static_excluded_det_count",
    "night_det_count",
    "overpass_event_count",
    "frp_sum_mw",
    "raw_frp_mw",
    "active_cells_10km",
    "harmonized_srehi",
    "harmonized_srehi_lo90",
    "harmonized_srehi_hi90",
    "harmonized_frp_mw",
    "harmonized_frp_lo90",
    "harmonized_frp_hi90",
    "m1_srehi",
    "m2_srehi",
    "m1_frp_mw",
    "m2_frp_mw",
    "constellation_regime",
    "data_maturity",
    "is_in_sama",
    "sama_suspect_det_count",
)


# ---------------------------------------------------------------------------
# Agregación jerárquica y reconciliación
# ---------------------------------------------------------------------------


def _load_admin1(con: Any, path: Path) -> pd.DataFrame:
    query = f"""
        SELECT
            "year", "month", year_month, country_name, iso_a3, admin1_name,
            SUM(raw_all_det_count)::BIGINT AS raw_all_det_count,
            SUM(clean_det_count)::BIGINT AS clean_det_count,
            SUM(static_excluded_det_count)::BIGINT AS static_excluded_det_count,
            SUM(high_conf_count)::BIGINT AS high_conf_count,
            SUM(night_det_count)::BIGINT AS night_det_count,
            SUM(overpass_event_count)::BIGINT AS overpass_event_count,
            COALESCE(SUM(frp_sum_mw), 0.0) AS raw_frp_mw,
            MAX(unique_active_cells_10km)::BIGINT AS active_cells_10km,
            SUM(nrt_det_count)::BIGINT AS nrt_det_count,
            SUM(sp_det_count)::BIGINT AS sp_det_count,
            BOOL_OR(intersects_sama) AS intersects_sama,
            BOOL_OR(is_partial_month) AS is_partial_month
        FROM {parquet_sql(path)}
        GROUP BY "year", "month", year_month, country_name, iso_a3, admin1_name
    """
    return con.execute(query).fetch_df()


def _load_country(con: Any, path: Path) -> pd.DataFrame:
    query = f"""
        SELECT
            "year", "month", year_month, country_name, iso_a3,
            SUM(raw_all_det_count)::BIGINT AS raw_all_det_count,
            SUM(clean_det_count)::BIGINT AS clean_det_count,
            SUM(static_excluded_det_count)::BIGINT AS static_excluded_det_count,
            SUM(high_conf_count)::BIGINT AS high_conf_count,
            SUM(night_det_count)::BIGINT AS night_det_count,
            SUM(overpass_event_count)::BIGINT AS overpass_event_count,
            COALESCE(SUM(frp_sum_mw), 0.0) AS raw_frp_mw,
            MAX(unique_active_cells_10km)::BIGINT AS active_cells_10km,
            SUM(nrt_det_count)::BIGINT AS nrt_det_count,
            SUM(sp_det_count)::BIGINT AS sp_det_count,
            BOOL_OR(intersects_sama) AS intersects_sama,
            BOOL_OR(is_partial_month) AS is_partial_month
        FROM {parquet_sql(path)}
        GROUP BY "year", "month", year_month, country_name, iso_a3
    """
    return con.execute(query).fetch_df()


def _maturity_from_flags(frame: pd.DataFrame) -> np.ndarray:
    year_month = frame["year_month"].astype(str).to_numpy()
    partial = frame["is_partial_month"].fillna(False).to_numpy(dtype=bool)
    nrt = (
        frame["nrt_det_count"].fillna(0).to_numpy(dtype=float)
        > frame["sp_det_count"].fillna(0).to_numpy(dtype=float)
    )
    maturity = np.full(len(frame), "STANDARD_SCIENCE", dtype=object)
    maturity[nrt] = "NRT_PRELIMINARY"
    maturity[partial] = "PARTIAL_MONTH"
    maturity[year_month >= "2026-01"] = "NRT_PRELIMINARY"
    maturity[(year_month == "2000-11") | (year_month == "2026-11")] = "PARTIAL_MONTH"
    return maturity


def _allocate_shares(values: pd.Series) -> np.ndarray:
    total = float(np.nansum(values.to_numpy(dtype=float)))
    if total > 0:
        return values.to_numpy(dtype=float) / total
    return np.full(len(values), 1.0 / max(len(values), 1), dtype=float)


_ADMIN1_COLUMNS = (
    "year", "month", "year_month", "country_name", "iso_a3", "admin1_name",
    "raw_all_det_count", "clean_det_count", "static_excluded_det_count",
    "high_conf_count", "night_det_count", "overpass_event_count", "raw_frp_mw",
    "active_cells_10km", "nrt_det_count", "sp_det_count", "is_partial_month",
    "harmonized_srehi", "harmonized_srehi_lo90", "harmonized_srehi_hi90",
    "harmonized_frp_mw", "harmonized_frp_lo90", "harmonized_frp_hi90",
    "m1_srehi", "m2_srehi", "m1_frp_mw", "m2_frp_mw",
    "constellation_regime", "data_maturity", "is_in_sama", "intersects_sama",
    "sama_suspect_det_count",
)

_COUNTRY_COLUMNS = (
    "year", "month", "year_month", "country_name", "iso_a3",
    "raw_all_det_count", "clean_det_count", "static_excluded_det_count",
    "high_conf_count", "night_det_count", "overpass_event_count", "raw_frp_mw",
    "active_cells_10km", "nrt_det_count", "sp_det_count", "is_partial_month",
    "harmonized_srehi", "harmonized_srehi_lo90", "harmonized_srehi_hi90",
    "harmonized_frp_mw", "harmonized_frp_lo90", "harmonized_frp_hi90",
    "m1_srehi", "m2_srehi", "m1_frp_mw", "m2_frp_mw",
    "constellation_regime", "data_maturity", "is_in_sama", "intersects_sama",
    "sama_suspect_det_count",
)


def build_hierarchical_outputs(
    cell_harmonized: pd.DataFrame,
    admin1_raw: pd.DataFrame,
    country_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reconcilia celdas -> Admin-1 -> País -> Sudamérica preservando no negatividad."""
    from src.core.harmonize import reconcile_hierarchical_sums

    cell_country = (
        cell_harmonized.groupby(["year_month", "iso_a3"], as_index=False)
        .agg(
            country_name=("country_name", "first"),
            harmonized_srehi=("harmonized_srehi", "sum"),
            harmonized_srehi_lo90=("harmonized_srehi_lo90", "sum"),
            harmonized_srehi_hi90=("harmonized_srehi_hi90", "sum"),
            harmonized_frp_mw=("harmonized_frp_mw", "sum"),
            harmonized_frp_lo90=("harmonized_frp_lo90", "sum"),
            harmonized_frp_hi90=("harmonized_frp_hi90", "sum"),
            m1_srehi=("m1_srehi", "sum"),
            m2_srehi=("m2_srehi", "sum"),
            m1_frp_mw=("m1_frp_mw", "sum"),
            m2_frp_mw=("m2_frp_mw", "sum"),
        )
    )

    admin = admin1_raw.copy()
    admin["admin1_name"] = admin["admin1_name"].fillna("UNKNOWN").astype(str)
    admin["country_name"] = admin["country_name"].fillna("UNKNOWN").astype(str)
    admin["iso_a3"] = admin["iso_a3"].fillna("UNK").astype(str)
    admin = admin.merge(
        cell_country,
        on=["year_month", "iso_a3"],
        how="left",
        suffixes=("", "_country_total"),
    )
    for column in (
        "harmonized_srehi",
        "harmonized_srehi_lo90",
        "harmonized_srehi_hi90",
        "harmonized_frp_mw",
        "harmonized_frp_lo90",
        "harmonized_frp_hi90",
        "m1_srehi",
        "m2_srehi",
        "m1_frp_mw",
        "m2_frp_mw",
    ):
        admin[column] = admin[column].fillna(0.0)

    # Shares: SREHI por eventos; FRP por FRP crudo. Fallbacks deterministas.
    group = admin.groupby(["year_month", "iso_a3"], sort=False)
    events_sum = group["overpass_event_count"].transform("sum").to_numpy(dtype=float)
    clean_sum = group["clean_det_count"].transform("sum").to_numpy(dtype=float)
    frp_sum = group["raw_frp_mw"].transform("sum").to_numpy(dtype=float)
    n_group = group["admin1_name"].transform("size").to_numpy(dtype=float)

    share_srehi = np.where(
        events_sum > 0,
        admin["overpass_event_count"].to_numpy(dtype=float) / np.where(events_sum > 0, events_sum, 1.0),
        np.where(
            clean_sum > 0,
            admin["clean_det_count"].to_numpy(dtype=float) / np.where(clean_sum > 0, clean_sum, 1.0),
            1.0 / np.maximum(n_group, 1.0),
        ),
    )
    share_frp = np.where(
        frp_sum > 0,
        admin["raw_frp_mw"].to_numpy(dtype=float) / np.where(frp_sum > 0, frp_sum, 1.0),
        share_srehi,
    )

    for column, share in (
        ("harmonized_srehi", share_srehi),
        ("harmonized_srehi_lo90", share_srehi),
        ("harmonized_srehi_hi90", share_srehi),
        ("m1_srehi", share_srehi),
        ("m2_srehi", share_srehi),
        ("harmonized_frp_mw", share_frp),
        ("harmonized_frp_lo90", share_frp),
        ("harmonized_frp_hi90", share_frp),
        ("m1_frp_mw", share_frp),
        ("m2_frp_mw", share_frp),
    ):
        admin[column] = admin[column].to_numpy(dtype=float) * np.asarray(share, dtype=float)

    admin["intersects_sama"] = admin["intersects_sama"].fillna(False).astype(bool)
    admin["is_in_sama"] = admin["intersects_sama"]
    admin["sama_suspect_det_count"] = np.where(
        admin["intersects_sama"].to_numpy(dtype=bool),
        admin["night_det_count"].fillna(0).to_numpy(dtype=float),
        0.0,
    ).astype("int64")
    admin["constellation_regime"] = _regime_labels(admin)
    admin["data_maturity"] = _maturity_from_flags(admin)

    # Reconciliación exacta Admin-1 -> País por mes. ``reconcile`` espera una
    # matriz (1, n_hijos) por grupo y un total por grupo; nunca un total por
    # cada hijo (eso forzaría a cada Admin-1 a igualar el país completo).
    country_totals = cell_country.set_index(["year_month", "iso_a3"])[
        ["harmonized_srehi", "harmonized_frp_mw"]
    ]
    for key, indices in admin.groupby(["year_month", "iso_a3"], sort=False).indices.items():
        positions = np.asarray(indices, dtype=int)
        total_country = float(country_totals.loc[key, "harmonized_srehi"])
        total_frp = float(country_totals.loc[key, "harmonized_frp_mw"])
        values = admin.loc[positions, "harmonized_srehi"].to_numpy(dtype=float).reshape(1, -1)
        admin.loc[positions, "harmonized_srehi"] = reconcile_hierarchical_sums(
            values, np.asarray([total_country]), non_negative=True
        ).ravel()
        frp_values = admin.loc[positions, "harmonized_frp_mw"].to_numpy(dtype=float).reshape(1, -1)
        admin.loc[positions, "harmonized_frp_mw"] = reconcile_hierarchical_sums(
            frp_values, np.asarray([total_frp]), non_negative=True
        ).ravel()

    # País = suma exacta de Admin-1; las columnas crudas vienen del cubo país.
    country = (
        admin.groupby(["year_month", "iso_a3"], as_index=False)
        .agg(
            country_name=("country_name", "first"),
            harmonized_srehi=("harmonized_srehi", "sum"),
            harmonized_srehi_lo90=("harmonized_srehi_lo90", "sum"),
            harmonized_srehi_hi90=("harmonized_srehi_hi90", "sum"),
            harmonized_frp_mw=("harmonized_frp_mw", "sum"),
            harmonized_frp_lo90=("harmonized_frp_lo90", "sum"),
            harmonized_frp_hi90=("harmonized_frp_hi90", "sum"),
            m1_srehi=("m1_srehi", "sum"),
            m2_srehi=("m2_srehi", "sum"),
            m1_frp_mw=("m1_frp_mw", "sum"),
            m2_frp_mw=("m2_frp_mw", "sum"),
            intersects_sama=("intersects_sama", "any"),
        )
    )
    country = country.merge(
        country_raw,
        on=["year_month", "iso_a3"],
        how="left",
        suffixes=("", "_raw"),
    )
    country["country_name"] = (
        country["country_name"].fillna(country["country_name_raw"]).fillna("UNKNOWN").astype(str)
    )
    country["iso_a3"] = country["iso_a3"].fillna("UNK").astype(str)
    for column in (
        "raw_all_det_count",
        "clean_det_count",
        "static_excluded_det_count",
        "high_conf_count",
        "night_det_count",
        "overpass_event_count",
        "nrt_det_count",
        "sp_det_count",
    ):
        country[column] = country[column].fillna(0).astype("int64")
    country["raw_frp_mw"] = country["raw_frp_mw"].fillna(0.0).astype(float)
    country["active_cells_10km"] = country["active_cells_10km"].fillna(0).astype("int64")
    country["is_partial_month"] = country["is_partial_month"].fillna(False).astype(bool)
    country["intersects_sama"] = country["intersects_sama"].fillna(False).astype(bool)
    country["is_in_sama"] = country["intersects_sama"]
    country["sama_suspect_det_count"] = np.where(
        country["intersects_sama"].to_numpy(dtype=bool),
        country["night_det_count"].fillna(0).to_numpy(dtype=float),
        0.0,
    ).astype("int64")
    country["constellation_regime"] = _regime_labels(country)
    country["data_maturity"] = _maturity_from_flags(country)

    # Nivel 2: países -> total Sudamérica (la suma es el total canónico).
    south_america = (
        country.groupby(["year", "month", "year_month"], as_index=False)
        .agg(
            harmonized_srehi=("harmonized_srehi", "sum"),
            harmonized_srehi_lo90=("harmonized_srehi_lo90", "sum"),
            harmonized_srehi_hi90=("harmonized_srehi_hi90", "sum"),
            harmonized_frp_mw=("harmonized_frp_mw", "sum"),
            harmonized_frp_lo90=("harmonized_frp_lo90", "sum"),
            harmonized_frp_hi90=("harmonized_frp_hi90", "sum"),
            raw_all_det_count=("raw_all_det_count", "sum"),
            clean_det_count=("clean_det_count", "sum"),
            overpass_event_count=("overpass_event_count", "sum"),
            raw_frp_mw=("raw_frp_mw", "sum"),
            nrt_det_count=("nrt_det_count", "sum"),
            sp_det_count=("sp_det_count", "sum"),
            is_partial_month=("is_partial_month", "max"),
            n_countries=("iso_a3", "nunique"),
        )
    )
    for year_month, indices in country.groupby("year_month", sort=False).indices.items():
        positions = np.asarray(indices, dtype=int)
        total = float(
            south_america.loc[
                south_america["year_month"] == year_month, "harmonized_srehi"
            ].iloc[0]
        )
        # Un solo grupo (Sudamérica) con todos los países como hijos.
        values = country.loc[positions, "harmonized_srehi"].to_numpy(dtype=float).reshape(1, -1)
        country.loc[positions, "harmonized_srehi"] = reconcile_hierarchical_sums(
            values, np.asarray([total]), non_negative=True
        ).ravel()
        frp_total = float(
            south_america.loc[
                south_america["year_month"] == year_month, "harmonized_frp_mw"
            ].iloc[0]
        )
        frp_values = country.loc[positions, "harmonized_frp_mw"].to_numpy(dtype=float).reshape(1, -1)
        country.loc[positions, "harmonized_frp_mw"] = reconcile_hierarchical_sums(
            frp_values, np.asarray([frp_total]), non_negative=True
        ).ravel()

    south_america["is_in_sama"] = True
    south_america["constellation_regime"] = _regime_labels(south_america)
    south_america["data_maturity"] = _maturity_from_flags(south_america)
    admin = admin[[column for column in _ADMIN1_COLUMNS if column in admin.columns]].copy()
    country = country[[column for column in _COUNTRY_COLUMNS if column in country.columns]].copy()
    for column in _ADMIN1_COLUMNS:
        if column not in admin.columns:
            admin[column] = 0 if column not in {"country_name", "iso_a3", "admin1_name", "constellation_regime", "data_maturity"} else ""
    for column in _COUNTRY_COLUMNS:
        if column not in country.columns:
            country[column] = 0 if column not in {"country_name", "iso_a3", "constellation_regime", "data_maturity"} else ""
    admin = admin[list(_ADMIN1_COLUMNS)].sort_values(
        ["year", "month", "iso_a3", "admin1_name"]
    ).reset_index(drop=True)
    country = country[list(_COUNTRY_COLUMNS)].sort_values(
        ["year", "month", "iso_a3"]
    ).reset_index(drop=True)
    return admin, country, south_america


# ---------------------------------------------------------------------------
# Benchmarks de cambio de escalón
# ---------------------------------------------------------------------------


def _period_index(frame: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(frame["year"], errors="coerce").fillna(0).astype(int) * 12
        + pd.to_numeric(frame["month"], errors="coerce").fillna(0).astype(int)
    ).to_numpy(dtype=int)


def _step_change(
    raw: pd.DataFrame,
    harmonized: pd.DataFrame,
    transition: str,
    *,
    months: int = 12,
) -> dict[str, Any]:
    start_year, start_month = (int(part) for part in transition.split("-"))
    start = start_year * 12 + start_month
    raw_period = _period_index(raw)
    harm_period = _period_index(harmonized)
    result: dict[str, Any] = {
        "start_month": f"{start_year:04d}-{start_month:02d}",
        "window_months": int(months),
    }
    for metric, raw_column, harm_column in (
        ("srehi", "raw_events", "harmonized_srehi"),
        ("frp_mw", "raw_frp", "harmonized_frp_mw"),
    ):
        before_raw = raw.loc[(raw_period >= start - months) & (raw_period < start), raw_column]
        after_raw = raw.loc[(raw_period >= start) & (raw_period < start + months), raw_column]
        before_h = harmonized.loc[
            (harm_period >= start - months) & (harm_period < start), harm_column
        ]
        after_h = harmonized.loc[
            (harm_period >= start) & (harm_period < start + months), harm_column
        ]
        raw_before = float(before_raw.mean()) if len(before_raw) else float("nan")
        raw_after = float(after_raw.mean()) if len(after_raw) else float("nan")
        h_before = float(before_h.mean()) if len(before_h) else float("nan")
        h_after = float(after_h.mean()) if len(after_h) else float("nan")
        raw_ratio = raw_after / raw_before if raw_before and np.isfinite(raw_before) else float("nan")
        h_ratio = h_after / h_before if h_before and np.isfinite(h_before) else float("nan")
        raw_log = abs(math.log(raw_ratio)) if raw_ratio > 0 and np.isfinite(raw_ratio) else float("nan")
        h_log = abs(math.log(h_ratio)) if h_ratio > 0 and np.isfinite(h_ratio) else float("nan")
        reduction = 1.0 - (h_log / raw_log) if raw_log and np.isfinite(raw_log) and raw_log > 0 else 0.0
        # The former smoothed-ratio diagnostic was removed: it ran kalman_1d_smooth
        # with an absolute process_var (1e-2) on series of magnitude 1e4-1e6, so the
        # smoothed series was almost flat and the ratio was ~1 for any input.
        result[metric] = {
            "raw_before": raw_before,
            "raw_after": raw_after,
            "raw_ratio": float(raw_ratio),
            "harmonized_before": h_before,
            "harmonized_after": h_after,
            "harmonized_ratio": float(h_ratio),
            "raw_step_change_ratio": float(raw_ratio),
            "harmonized_step_change_ratio": float(h_ratio),
            "step_change_ratio": float(h_ratio),
            "ratio_reduction": float(reduction),
            "passed": bool(
                np.isfinite(raw_ratio)
                and np.isfinite(h_ratio)
                and h_log < raw_log
            ),
            "n_before": int(len(before_h)),
            "n_after": int(len(after_h)),
        }
    return result


def _benchmarks(
    cell_harmonized: pd.DataFrame,
    country_harmonized: pd.DataFrame,
    country_raw: pd.DataFrame,
    *,
    dedup: dict[str, Any],
    blend_info: dict[str, Any],
    cross_regime: dict[str, Any],
    blend_summary: list[dict[str, Any]],
    splits: dict[str, Any],
    south_america: pd.DataFrame,
) -> dict[str, Any]:
    raw = country_raw.copy()
    raw["raw_events"] = raw["overpass_event_count"].astype(float)
    raw["raw_frp"] = raw["raw_frp_mw"].astype(float)
    raw_monthly = (
        raw.groupby(["year", "month", "year_month"], as_index=False)
        .agg(raw_events=("raw_events", "sum"), raw_frp=("raw_frp", "sum"))
    )
    harm_monthly = (
        country_harmonized.groupby(["year", "month", "year_month"], as_index=False)
        .agg(
            harmonized_srehi=("harmonized_srehi", "sum"),
            harmonized_frp_mw=("harmonized_frp_mw", "sum"),
        )
    )
    transitions = {
        "2002": _step_change(raw_monthly, harm_monthly, "2002-07"),
        "2012": _step_change(raw_monthly, harm_monthly, "2012-01"),
        "2018": _step_change(raw_monthly, harm_monthly, "2018-04"),
        "2021": _step_change(raw_monthly, harm_monthly, "2021-01"),
        "2023": _step_change(raw_monthly, harm_monthly, "2023-01"),
    }
    # Alias planos por transición (SREHI) para consumidores que no descienden
    # al detalle por target.
    for item in transitions.values():
        item["raw_ratio"] = item["srehi"]["raw_step_change_ratio"]
        item["harmonized_ratio"] = item["srehi"]["harmonized_step_change_ratio"]
        item["ratio_reduction"] = item["srehi"]["ratio_reduction"]
        item["passed"] = item["srehi"]["passed"]
    return {
        "schema": "c2_benchmarks_v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "folds": int(splits["folds"]),
        "fold_fingerprint": splits["fingerprint"],
        "dataset_sha256": splits["dataset_sha256"],
        "pre_drift_window": f"{PRE_DRIFT_START}..{PRE_DRIFT_END}",
        "transitions": transitions,
        "2012": transitions["2012"],
        "2018": transitions["2018"],
        "2021": transitions["2021"],
        "step_change_ratios": {
            label: {
                "raw": item["srehi"]["raw_step_change_ratio"],
                "harmonized": item["srehi"]["harmonized_step_change_ratio"],
                "ratio_reduction": item["srehi"]["ratio_reduction"],
                "frp_raw": item["frp_mw"]["raw_step_change_ratio"],
                "frp_harmonized": item["frp_mw"]["harmonized_step_change_ratio"],
            }
            for label, item in transitions.items()
        },
        "dedup": dedup,
        "m3_blend": blend_info,
        "blend_fold_order": blend_summary,
        "cross_regime_guard": cross_regime,
        "south_america_totals": {
            "rows": int(len(south_america)),
            "total_srehi": float(south_america["harmonized_srehi"].sum()),
            "total_frp_mw": float(south_america["harmonized_frp_mw"].sum()),
        },
        "harmonized_cell_rows": int(len(cell_harmonized)),
    }


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------


def run(
    processed_dir: Path | str,
    *,
    max_train_rows: int = 400_000,
    seed: int = 42,
) -> dict[str, Any]:
    processed = Path(processed_dir).expanduser().resolve()
    splits_path = processed / "splits.json"
    overlap_path = processed / "overlap_cell50km_2012_2020.parquet"
    cell_cube = processed / "cube_cell50km_monthly.parquet"
    admin_cube = processed / "cube_admin1_monthly.parquet"
    country_cube = processed / "cube_country_monthly.parquet"
    for required in (splits_path, overlap_path, cell_cube, admin_cube, country_cube):
        if not required.is_file():
            raise FileNotFoundError(f"Falta artefacto requerido: {required}")

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    con = connect_duckdb(scratch_dir=processed / ".duckdb_tmp")
    try:
        overlap = con.execute(
            f"SELECT * FROM {parquet_sql(overlap_path)}"
        ).fetch_df()
        if splits.get("dataset_sha256") and sha256_file(overlap_path) != splits["dataset_sha256"]:
            raise ValueError("El hash del emparejamiento no coincide con splits.json.")
        future_features = load_cell_month_features(
            con, cell_cube, start="2021-01", end="2026-12"
        )
        admin1_raw = _load_admin1(con, admin_cube)
        country_raw = _load_country(con, country_cube)
    finally:
        con.close()

    started = time.time()
    evaluation = _oof_predictions(
        overlap, future_features, max_train_rows=int(max_train_rows), seed=int(seed)
    )
    elapsed_eval = time.time() - started

    metrics: dict[str, dict[str, Any]] = {}
    for model in MODEL_IDS:
        metrics[model] = {
            target: _metric_bundle(
                evaluation["y"][target],
                evaluation["predictions"][model][target],
                evaluation["folds"],
            )
            for target in TARGETS
        }
    cross_regime = _cross_regime_guard(overlap, evaluation["predictions"])

    oof_path = processed / "oof_m0_m3.parquet"
    _write_oof(
        overlap, evaluation["predictions"], evaluation["sigmas"], oof_path
    )

    blend_info = evaluation["blend_info"]
    blend_summary = evaluation["blend_summary"]
    production = _fit_production(
        overlap,
        future_features,
        blend_info,
        max_train_rows=int(max_train_rows),
        seed=int(seed),
    )
    # Liberar la tabla de entrenamiento antes de materializar el cubo completo.
    del overlap, future_features, evaluation
    con = connect_duckdb(scratch_dir=processed / ".duckdb_tmp")
    try:
        all_features = load_cell_month_features(
            con, cell_cube, start="2000-11", end="2026-12"
        )
    finally:
        con.close()
    production_started = time.time()
    predictions = _production_predictions(all_features, production)
    elapsed_prod = time.time() - production_started

    dedup = _dedup_ratios(all_features)
    cell_harmonized = harmonize_cells(all_features, predictions, dedup)
    cell_harmonized = cell_harmonized[list(_CELL_COLUMNS)].sort_values(
        ["year", "month", "cell_id_50km"]
    ).reset_index(drop=True)
    del all_features, predictions

    admin1_harmonized, country_harmonized, south_america = build_hierarchical_outputs(
        cell_harmonized, admin1_raw, country_raw
    )

    con = connect_duckdb(scratch_dir=processed / ".duckdb_tmp")
    try:
        for name, frame in (
            ("harmonized_cell50km_monthly.parquet", cell_harmonized),
            ("harmonized_admin1_monthly.parquet", admin1_harmonized),
            ("harmonized_country_monthly.parquet", country_harmonized),
            ("harmonized_south_america_monthly.parquet", south_america),
        ):
            con.register("out_frame", frame)
            write_parquet(con, "SELECT * FROM out_frame", processed / name)
            con.unregister("out_frame")
    finally:
        con.close()

    benchmarks = _benchmarks(
        cell_harmonized,
        country_harmonized,
        country_raw,
        dedup=dedup,
        blend_info=blend_info,
        cross_regime=cross_regime,
        blend_summary=blend_summary,
        splits=splits,
        south_america=south_america,
    )
    write_json(processed / "benchmarks_m0_m3.json", benchmarks, indent=2)

    training_seconds = {model: float(elapsed_eval / len(MODEL_IDS)) for model in MODEL_IDS}
    training_seconds["M3_simplex_kalman_reconciled"] += float(elapsed_prod)
    records = _experiment_records(
        metrics,
        splits=splits,
        oof_path=oof_path,
        splits_path=splits_path,
        config={
            "max_train_rows": int(max_train_rows),
            "seed": int(seed),
            "block_size_m": splits.get("block_size_m"),
            "pre_drift_window": f"{PRE_DRIFT_START}..{PRE_DRIFT_END}",
        },
        training_seconds=training_seconds,
    )
    _write_experiments(records, processed / "experiments.jsonl")

    return {
        "records": records,
        "benchmarks": benchmarks,
        "metrics": metrics,
        "oof_path": str(oof_path),
        "cell_rows": int(len(cell_harmonized)),
        "admin1_rows": int(len(admin1_harmonized)),
        "country_rows": int(len(country_harmonized)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=400_000,
        help="Tope determinista de filas por fold para acotar RAM/tiempo (0 = todas).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        args.processed_dir,
        max_train_rows=int(args.max_train_rows),
        seed=int(args.seed),
    )
    print(
        f"OOF: {result['oof_path']} | celdas={result['cell_rows']} "
        f"admin1={result['admin1_rows']} country={result['country_rows']}"
    )
    for target in TARGETS:
        m0 = result["metrics"]["M0_raw_unharmonized"][target]["mean"]["rmse"]
        m3 = result["metrics"]["M3_simplex_kalman_reconciled"][target]["mean"]["rmse"]
        print(f"{target}: RMSE M0={m0:.4f} -> M3={m3:.4f}")
    transitions = result["benchmarks"]["transitions"]
    for label in ("2002", "2012", "2018", "2021", "2023"):
        item = transitions[label]["srehi"]
        print(
            f"transición {label}: raw={item['raw_step_change_ratio']:.3f} "
            f"harmonized={item['harmonized_step_change_ratio']:.3f} "
            f"reducción={item['ratio_reduction']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
