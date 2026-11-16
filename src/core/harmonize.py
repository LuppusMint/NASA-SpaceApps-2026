#!/usr/bin/env python3
"""Robust harmonization, cross-sensor bridges, GP/Kalman smoothing, and reconciliation."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


def _finite_1d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def tukey_biweight_location(
    x: Sequence[float] | np.ndarray,
    *,
    c: float = 6.0,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> float:
    """Tukey/Beers biweight M-estimator of location.

    Ignores ``NaN``; if there are fewer than three values, the vector is
    constant, or the MAD is zero, returns the finite median without dividing
    by zero.
    """
    arr = _finite_1d(x)
    if arr.size == 0:
        return float("nan")
    median = float(np.median(arr))
    if arr.size <= 2 or float(np.max(arr)) == float(np.min(arr)):
        return median
    mad = float(np.median(np.abs(arr - median)))
    if not np.isfinite(mad) or mad <= 0.0:
        return median
    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be positive.")
    if tol <= 0 or int(max_iter) < 1:
        raise ValueError("tol and max_iter must be positive.")

    scale = 1.4826 * mad
    location = median
    for _ in range(int(max_iter)):
        u = (arr - location) / (float(c) * scale)
        weights = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        total = float(np.sum(weights))
        if total <= 0.0:
            return median
        updated = float(np.sum(weights * arr) / total)
        if abs(updated - location) <= float(tol):
            location = updated
            break
        location = updated
    return float(location)


def robust_biweight_baseline(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str,
    c: float = 6.0,
    min_count: int = 3,
) -> pd.DataFrame:
    """Computes a grouped robust baseline: count, median, biweight, and MAD."""
    if not group_cols:
        raise ValueError("group_cols must contain at least one column.")
    for col in list(group_cols) + [value_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r}.")
    if int(min_count) < 1:
        raise ValueError("min_count must be >= 1.")

    columns = list(group_cols) + ["count", "median", "biweight_loc", "mad_scale"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    grouped = df.groupby(list(group_cols), dropna=False, sort=True)
    for key, group in grouped:
        values = _finite_1d(group[value_col].to_numpy())
        count = int(values.size)
        if count == 0:
            median = biweight = mad = float("nan")
        else:
            median = float(np.median(values))
            biweight = tukey_biweight_location(values, c=c) if count >= int(min_count) else median
            mad = float(1.4826 * np.median(np.abs(values - median)))
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {col: value for col, value in zip(group_cols, key_tuple)}
        row.update(
            {
                "count": count,
                "median": median,
                "biweight_loc": float(biweight),
                "mad_scale": float(mad),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _clean_feature_matrix(
    df: pd.DataFrame,
    features: list[str],
    *,
    target: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Extracts finite X/y and imputes feature NaNs with the median."""
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")
    matrix = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    matrix[~np.isfinite(matrix)] = np.nan
    fill = np.nanmedian(matrix, axis=0) if len(matrix) else np.zeros(len(features))
    fill = np.where(np.isfinite(fill), fill, 0.0)
    matrix = np.where(np.isnan(matrix), fill, matrix)
    if target is None:
        return matrix, None, fill
    target_arr = np.asarray(target, dtype=float).ravel()
    valid = np.isfinite(target_arr)
    return matrix[valid], target_arr[valid], fill


def _build_bridge_features(
    df: pd.DataFrame, source_col: str, feature_cols: list[str] | None, target_col: str
) -> list[str]:
    features: list[str] = []
    if feature_cols:
        features.extend(str(col) for col in feature_cols)
    if source_col not in features:
        features.insert(0, source_col)
    features = list(dict.fromkeys(col for col in features if col != target_col))
    return features


def fit_cross_sensor_bridge(
    overlap_df: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    feature_cols: list[str] | None = None,
    model_kind: str = "poisson",
    alpha: float = 1e-3,
) -> dict[str, Any]:
    """Calibrates a source sensor onto a target sensor within their overlap window.

    ``model_kind`` accepts ``linear_ridge``, ``poisson``, ``tweedie``, and
    ``hurdle``. The hurdle model combines the positive probability with the
    conditional intensity ``P(Y>0) * E[Y|Y>0]``.
    """
    from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge, TweedieRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    kind = str(model_kind).lower()
    if kind not in {"linear_ridge", "poisson", "tweedie", "hurdle"}:
        raise ValueError("model_kind must be linear_ridge, poisson, tweedie, or hurdle.")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be >= 0.")
    for col in (source_col, target_col):
        if col not in overlap_df.columns:
            raise ValueError(f"Missing column {col!r}.")
    if overlap_df.empty:
        raise ValueError("The overlap window is empty.")

    features = _build_bridge_features(overlap_df, source_col, feature_cols, target_col)
    matrix, target, fill = _clean_feature_matrix(
        overlap_df, features, target=np.asarray(overlap_df[target_col], dtype=float)
    )
    if len(target) < 2:
        raise ValueError("At least 2 valid rows are required to calibrate the bridge.")
    if kind in {"poisson", "tweedie"} and np.any(target < 0):
        raise ValueError(f"{kind} requires a non-negative target.")

    bridge: dict[str, Any] = {
        "model_kind": kind,
        "source_col": source_col,
        "target_col": target_col,
        "feature_cols": features,
        "feature_fill": fill,
        "alpha": float(alpha),
        "n_train": int(len(target)),
    }
    if kind == "hurdle":
        positive = target > 0
        bridge["positive_rate"] = float(np.mean(positive))
        if np.all(positive) or np.all(~positive):
            bridge["classifier"] = None
            bridge["constant_probability"] = 1.0 if np.all(positive) else 0.0
        else:
            classifier = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "logit",
                        LogisticRegression(
                            max_iter=2000,
                            C=max(1e-3, 1.0 / max(float(alpha), 1e-9)),
                            random_state=0,
                        ),
                    ),
                ]
            )
            classifier.fit(matrix, positive.astype(int))
            bridge["classifier"] = classifier
            bridge["constant_probability"] = None

        if np.any(positive):
            positive_matrix = matrix[positive]
            positive_target = target[positive]
            if len(positive_target) >= 2 and float(np.var(positive_target)) > 0.0:
                intensity = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "poisson",
                            PoissonRegressor(alpha=max(float(alpha), 1e-9), max_iter=2000),
                        ),
                    ]
                )
            else:
                intensity = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("ridge", Ridge(alpha=max(float(alpha), 1e-9))),
                    ]
                )
            if len(positive_target) >= 2:
                try:
                    intensity.fit(positive_matrix, positive_target)
                except Exception:  # noqa: BLE001 - numerical fallback if the estimator does not converge
                    intensity = None
            else:
                intensity = None
            bridge["intensity_model"] = intensity
            bridge["intensity_fallback"] = float(np.mean(positive_target))
        else:
            bridge["intensity_model"] = None
            bridge["intensity_fallback"] = 0.0
    else:
        if kind in {"poisson", "tweedie"} and float(np.max(np.abs(target))) == 0.0:
            # The log link is undefined at zero; the constant predictor avoids NaN.
            bridge["model"] = None
            bridge["constant_prediction"] = 0.0
        else:
            if kind == "linear_ridge":
                estimator = Ridge(alpha=max(float(alpha), 0.0))
                bridge["model"] = Pipeline([("scale", StandardScaler()), ("ridge", estimator)])
            elif kind == "poisson":
                bridge["model"] = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("poisson", PoissonRegressor(alpha=max(float(alpha), 1e-9), max_iter=2000)),
                    ]
                )
            else:
                bridge["model"] = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "tweedie",
                            TweedieRegressor(
                                power=1.5,
                                alpha=max(float(alpha), 1e-9),
                                link="log",
                                max_iter=2000,
                            ),
                        ),
                    ]
                )
            bridge["model"].fit(matrix, target)
    return bridge


def predict_cross_sensor_bridge(bridge: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    """Applies a calibrated bridge to a DataFrame with the same features."""
    if not isinstance(bridge, dict) or "feature_cols" not in bridge:
        raise ValueError("invalid bridge: feature_cols is missing.")
    if len(df) == 0:
        return np.asarray([], dtype=float)
    features = list(bridge["feature_cols"])
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for prediction: {', '.join(missing)}")
    matrix = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    fill = np.asarray(bridge.get("feature_fill", np.zeros(len(features))), dtype=float)
    matrix[~np.isfinite(matrix)] = np.nan
    matrix = np.where(np.isnan(matrix), fill, matrix)
    if len(matrix) == 0:
        return np.asarray([], dtype=float)
    kind = bridge.get("model_kind")
    if kind == "hurdle":
        classifier = bridge.get("classifier")
        constant = bridge.get("constant_probability")
        if classifier is None:
            probability = np.full(len(matrix), float(constant if constant is not None else 0.0))
        else:
            probability = classifier.predict_proba(matrix)[:, 1]
        intensity_model = bridge.get("intensity_model")
        if intensity_model is None:
            intensity = np.full(len(matrix), float(bridge.get("intensity_fallback", 0.0)))
        else:
            intensity = np.asarray(intensity_model.predict(matrix), dtype=float)
        return np.clip(probability * intensity, 0.0, None)
    if bridge.get("model") is None and "constant_prediction" in bridge:
        return np.full(len(matrix), float(bridge["constant_prediction"]), dtype=float)
    model = bridge.get("model")
    if model is None:
        raise ValueError("invalid bridge: the fitted model is missing.")
    predictions = np.asarray(model.predict(matrix), dtype=float).ravel()
    if kind in {"poisson", "tweedie"}:
        predictions = np.clip(predictions, 0.0, None)
    return predictions


def gp_continuous_projection(
    t_obs: Sequence[float] | np.ndarray,
    y_obs: Sequence[float] | np.ndarray,
    t_query: Sequence[float] | np.ndarray,
    *,
    y_err: Sequence[float] | np.ndarray | float | None = None,
    length_scale: float | None = None,
    noise_level: float = 1e-2,
    normalize_y: bool = True,
) -> dict[str, Any]:
    """1-D GP projection with an RBF kernel and a safe fallback for degenerate cases."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

    t = np.asarray(t_obs, dtype=float).ravel()
    y = np.asarray(y_obs, dtype=float).ravel()
    query = np.asarray(t_query, dtype=float).ravel()
    if t.size != y.size:
        raise ValueError("t_obs and y_obs must have the same length.")
    if not np.all(np.isfinite(query)):
        raise ValueError("t_query contains non-finite values.")
    mask = np.isfinite(t) & np.isfinite(y)
    if y_err is not None:
        errors = np.asarray(y_err, dtype=float)
        if errors.ndim == 0:
            errors = np.full(t.size, float(errors))
        errors = errors.ravel()
        if errors.size != t.size:
            raise ValueError("y_err must have the same length as t_obs.")
        mask &= np.isfinite(errors) & (errors >= 0)
        errors = errors[mask]
    else:
        errors = None
    t = t[mask]
    y = y[mask]
    n_obs = int(t.size)

    if len(query) == 0:
        return {
            "mean": np.asarray([], dtype=float),
            "std": np.asarray([], dtype=float),
            "n_obs": n_obs,
            "fallback_used": n_obs < 3,
        }

    fallback = n_obs < 3 or float(np.var(y)) <= 0.0 or float(np.ptp(t)) <= 0.0
    if fallback:
        constant = float(np.median(y)) if n_obs else 0.0
        return {
            "mean": np.full(len(query), constant, dtype=float),
            "std": np.zeros(len(query), dtype=float),
            "n_obs": n_obs,
            "fallback_used": True,
        }

    if length_scale is None:
        spacing = np.diff(np.sort(np.unique(t)))
        median_spacing = float(np.median(spacing)) if spacing.size else 1.0
        length_scale = max(median_spacing * 10.0, float(np.ptp(t)) / 10.0, 1e-6)
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("length_scale must be positive and finite.")
    if not np.isfinite(noise_level) or noise_level < 0:
        raise ValueError("noise_level must be >= 0.")

    if errors is None:
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e4))
            * RBF(length_scale=float(length_scale), length_scale_bounds=(1e-6, 1e6))
            + WhiteKernel(noise_level=max(float(noise_level), 1e-12), noise_level_bounds=(1e-12, 1e2))
        )
        alpha = 1e-10
    else:
        kernel = ConstantKernel(1.0, (1e-3, 1e4)) * RBF(
            length_scale=float(length_scale), length_scale_bounds=(1e-6, 1e6)
        )
        alpha = np.asarray(errors, dtype=float) ** 2 + 1e-6

    try:
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=bool(normalize_y),
            optimizer=None,
            n_restarts_optimizer=0,
            random_state=0,
        )
        model.fit(t.reshape(-1, 1), y)
        mean, std = model.predict(query.reshape(-1, 1), return_std=True)
        return {
            "mean": np.asarray(mean, dtype=float),
            "std": np.sqrt(np.clip(np.asarray(std, dtype=float) ** 2, 0.0, None)),
            "n_obs": n_obs,
            "fallback_used": False,
        }
    except Exception:  # noqa: BLE001 - the projection must degrade without failing
        constant = float(np.median(y))
        return {
            "mean": np.full(len(query), constant, dtype=float),
            "std": np.zeros(len(query), dtype=float),
            "n_obs": n_obs,
            "fallback_used": True,
        }


def _robust_diff_variance(y: np.ndarray) -> float:
    """Estimates observation variance from robust first differences."""
    finite = y[np.isfinite(y)]
    if finite.size < 3:
        return 1e-6
    differences = np.diff(finite)
    if differences.size == 0:
        return 1e-6
    center = float(np.median(differences))
    mad = float(np.median(np.abs(differences - center)))
    variance = (1.4826 * mad) ** 2 / 2.0
    return max(float(variance), 1e-6)


def kalman_1d_smooth(
    y: Sequence[float] | np.ndarray,
    *,
    obs_var: Sequence[float] | np.ndarray | float | None = None,
    process_var: float = 1e-2,
) -> dict[str, np.ndarray]:
    """Local-level Kalman filter and RTS smoother in pure NumPy.

    ``NaN`` values in ``y`` run prediction only: the mean is not updated and
    the variance grows with ``process_var``.
    """
    observations = np.asarray(y, dtype=float).ravel()
    n = int(observations.size)
    if not np.isfinite(process_var) or process_var < 0:
        raise ValueError("process_var must be >= 0.")
    q = float(process_var)
    if n == 0:
        empty = np.asarray([], dtype=float)
        return {"smoothed_mean": empty, "smoothed_var": empty, "smoothed_std": empty}

    if obs_var is None:
        observation_var = np.full(n, _robust_diff_variance(observations), dtype=float)
    else:
        values = np.asarray(obs_var, dtype=float)
        if values.ndim == 0:
            values = np.full(n, float(values))
        values = values.ravel()
        if values.size != n:
            raise ValueError("obs_var must be scalar or have the same length as y.")
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("obs_var must be non-negative and finite.")
        observation_var = np.clip(values, 1e-12, None)

    first_finite = next((value for value in observations if np.isfinite(value)), 0.0)
    prior_mean = np.zeros(n, dtype=float)
    prior_var = np.zeros(n, dtype=float)
    filtered_mean = np.zeros(n, dtype=float)
    filtered_var = np.zeros(n, dtype=float)

    mean = float(first_finite)
    variance = 1.0 + float(observation_var[0])
    for index in range(n):
        if index > 0:
            mean = mean
            variance = variance + q
        prior_mean[index] = mean
        prior_var[index] = max(variance, 1e-15)
        value = observations[index]
        if np.isfinite(value):
            gain = prior_var[index] / (prior_var[index] + observation_var[index])
            mean = prior_mean[index] + gain * (float(value) - prior_mean[index])
            variance = (1.0 - gain) * prior_var[index]
        else:
            mean = prior_mean[index]
            variance = prior_var[index]
        filtered_mean[index] = mean
        filtered_var[index] = max(variance, 1e-15)

    smoothed_mean = filtered_mean.copy()
    smoothed_var = filtered_var.copy()
    for index in range(n - 2, -1, -1):
        gain = filtered_var[index] / max(prior_var[index + 1], 1e-15)
        smoothed_mean[index] = filtered_mean[index] + gain * (
            smoothed_mean[index + 1] - prior_mean[index + 1]
        )
        smoothed_var[index] = filtered_var[index] + gain ** 2 * (
            smoothed_var[index + 1] - prior_var[index + 1]
        )
        smoothed_var[index] = max(float(smoothed_var[index]), 0.0)
    return {
        "smoothed_mean": smoothed_mean,
        "smoothed_var": smoothed_var,
        "smoothed_std": np.sqrt(np.clip(smoothed_var, 0.0, None)),
    }


def covariate_shift_weights(
    X_source: Any,
    X_target: Any,
    *,
    min_bin_count: int = 5,
    max_weight: float = 20.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Importance weights ``P(target|x)/P(source|x) * n_src/n_tgt``.

    A regularized logistic classifier estimates the domain probability;
    probabilities are clipped with ``min_bin_count`` and the final weights are
    capped at ``max_weight`` while keeping the mean exactly 1.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    source = np.asarray(X_source, dtype=float)
    target = np.asarray(X_target, dtype=float)
    if source.ndim == 1:
        source = source.reshape(-1, 1)
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("X_source and X_target must be 1-D or 2-D matrices.")
    if source.shape[1] != target.shape[1]:
        raise ValueError("X_source and X_target must have the same number of columns.")
    if len(source) == 0 or len(target) == 0:
        raise ValueError("X_source and X_target must not be empty.")
    if int(min_bin_count) < 1:
        raise ValueError("min_bin_count must be >= 1.")
    if not np.isfinite(max_weight) or max_weight < 1.0:
        raise ValueError("max_weight must be >= 1.")

    combined = np.vstack([source, target]).astype(float, copy=True)
    combined[~np.isfinite(combined)] = np.nan
    fill = np.nanmedian(combined, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    combined = np.where(np.isnan(combined), fill, combined)
    n_source, n_target = len(source), len(target)
    source = combined[:n_source]
    target = combined[n_source:]
    labels = np.concatenate([np.zeros(n_source, dtype=int), np.ones(n_target, dtype=int)])

    try:
        classifier = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        max_iter=2000,
                        C=1.0,
                        random_state=int(seed),
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        classifier.fit(combined, labels)
        probability = classifier.predict_proba(source)[:, 1]
        domain_auc = float(roc_auc_score(labels, classifier.predict_proba(combined)[:, 1]))
    except Exception:  # noqa: BLE001 - uniform fallback for degenerate data
        probability = np.full(n_source, 0.5, dtype=float)
        domain_auc = 0.5

    epsilon = min(0.499, float(int(min_bin_count)) / max(n_source + n_target, 1))
    probability = np.clip(probability, epsilon, 1.0 - epsilon)
    odds = probability / (1.0 - probability)
    raw_weights = np.clip(odds * (n_source / max(n_target, 1)), 0.0, None)

    if not np.any(raw_weights > 0):
        weights = np.ones(n_source, dtype=float)
    else:
        cap = float(max_weight)
        # Find the scale factor that makes mean(min(raw*s, cap)) == 1.
        low, high = 0.0, 1.0
        for _ in range(200):
            if float(np.mean(np.minimum(raw_weights * high, cap))) >= 1.0:
                break
            high *= 2.0
        else:
            high = 1e12
        for _ in range(200):
            mid = 0.5 * (low + high)
            if float(np.mean(np.minimum(raw_weights * mid, cap))) < 1.0:
                low = mid
            else:
                high = mid
        weights = np.minimum(raw_weights * (0.5 * (low + high)), cap)
        mean = float(np.mean(weights))
        if mean > 0:
            weights = weights / mean
        if float(np.max(weights)) > cap:
            weights = np.minimum(weights, cap)
            weights = weights / max(float(np.mean(weights)), 1e-15)
    weights = np.asarray(weights, dtype=float)
    effective_sample_size = float(np.sum(weights) ** 2 / max(np.sum(weights ** 2), 1e-15))
    return {
        "weights": weights,
        "effective_sample_size": effective_sample_size,
        "domain_auc": domain_auc,
        "n_source": int(n_source),
        "n_target": int(n_target),
        "clip_fraction": float(np.mean(weights >= float(max_weight) - 1e-12)),
    }


def _extract_prediction_vector(reconstructed: Any, clean: pd.DataFrame, value_col: str, time_col: str) -> np.ndarray:
    """Extracts predictions aligned by position or by time."""
    if isinstance(reconstructed, tuple) and reconstructed:
        reconstructed = reconstructed[0]
    if isinstance(reconstructed, dict):
        if value_col not in reconstructed:
            raise ValueError(f"The reconstructed dict does not contain {value_col!r}.")
        return np.asarray(reconstructed[value_col], dtype=float).ravel()
    if isinstance(reconstructed, pd.DataFrame):
        if value_col not in reconstructed.columns:
            raise ValueError(f"The reconstructed DataFrame does not contain {value_col!r}.")
        if time_col in clean.columns and time_col in reconstructed.columns:
            left = clean[[time_col, value_col]].copy()
            right = reconstructed[[time_col, value_col]].copy()
            if not left[time_col].duplicated().any() and not right[time_col].duplicated().any():
                merged = left.merge(right, on=time_col, how="inner", suffixes=("_true", "_pred"))
                return merged[f"{value_col}_pred"].to_numpy(dtype=float)
        if len(reconstructed) == len(clean):
            return reconstructed[value_col].to_numpy(dtype=float)
        if reconstructed.index.isin(clean.index).all():
            return reconstructed[value_col].reindex(clean.index).to_numpy(dtype=float)
        raise ValueError("Could not align the reconstruction with the clean truth.")
    array = np.asarray(reconstructed, dtype=float).ravel()
    if array.size != len(clean):
        raise ValueError("The reconstruction must have the same length as clean_df.")
    return array


def synthetic_degradation_benchmark(
    clean_df: pd.DataFrame,
    *,
    value_col: str,
    time_col: str,
    degrade_fn: Callable[[pd.DataFrame], Any],
    reconstruct_fn: Callable[[Any], Any],
) -> dict[str, Any]:
    """Measures reconstruction fidelity against the clean truth."""
    if value_col not in clean_df.columns:
        raise ValueError(f"Missing value column {value_col!r}.")
    if time_col not in clean_df.columns:
        raise ValueError(f"Missing time column {time_col!r}.")
    clean = clean_df.copy()
    degraded = degrade_fn(clean.copy())
    reconstructed = reconstruct_fn(degraded)
    predictions = _extract_prediction_vector(reconstructed, clean, value_col, time_col)
    truth = pd.to_numeric(clean[value_col], errors="coerce").to_numpy(dtype=float)
    if predictions.size != truth.size:
        raise ValueError("Predictions and truth must have the same length.")
    mask = np.isfinite(truth) & np.isfinite(predictions)
    n_eval = int(np.sum(mask))
    if n_eval == 0:
        metrics = {"rmse": float("nan"), "mae": float("nan"), "bias": float("nan"), "pearson_r": float("nan")}
    else:
        residuals = predictions[mask] - truth[mask]
        metrics = {
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "mae": float(np.mean(np.abs(residuals))),
            "bias": float(np.mean(residuals)),
            "pearson_r": 0.0,
        }
        if n_eval >= 2 and float(np.std(truth[mask])) > 0 and float(np.std(predictions[mask])) > 0:
            metrics["pearson_r"] = float(np.corrcoef(truth[mask], predictions[mask])[0, 1])
    metrics["n_eval"] = n_eval
    metrics["clean_rows"] = int(len(clean))
    metrics["degraded_rows"] = int(len(degraded)) if hasattr(degraded, "__len__") else None
    return metrics


def reconcile_hierarchical_sums(
    sub_preds: Any,
    total_pred: Any,
    *,
    weights: Any | None = None,
    non_negative: bool = True,
) -> np.ndarray:
    """Projects children so that their row sum matches the total.

    With ``non_negative=True``, negative children are clipped to zero. Scaling
    is proportional when ``abs(sum) > 1e-12``; the total is spread uniformly
    (or according to ``weights``) only when the sum is practically zero.
    """
    children = np.asarray(sub_preds, dtype=float)
    totals = np.asarray(total_pred, dtype=float).ravel()
    if children.ndim == 1:
        children = children.reshape(-1, 1)
    if children.ndim != 2:
        raise ValueError("sub_preds must be an (n, m) matrix.")
    n_rows, n_children = children.shape
    if n_children < 1:
        raise ValueError("sub_preds must have at least one child.")
    if totals.size != n_rows:
        raise ValueError("total_pred must have one entry per row.")
    if np.any(~np.isfinite(children)) or np.any(~np.isfinite(totals)):
        raise ValueError("sub_preds and total_pred must be finite.")
    if non_negative and np.any(totals < -1e-12):
        raise ValueError("With non_negative=True the total cannot be negative.")

    if weights is None:
        weight_matrix = np.ones((n_rows, n_children), dtype=float)
    else:
        weight_matrix = np.asarray(weights, dtype=float)
        if weight_matrix.ndim == 1:
            weight_matrix = np.tile(weight_matrix.reshape(1, -1), (n_rows, 1))
        if weight_matrix.shape != (n_rows, n_children):
            raise ValueError("weights must have shape (m,) or (n, m).")
        if np.any(~np.isfinite(weight_matrix)) or np.any(weight_matrix < 0):
            raise ValueError("weights must be non-negative and finite.")

    work = np.clip(children, 0.0, None) if non_negative else children.copy()
    row_sums = np.sum(work, axis=1)
    result = np.zeros_like(work)

    # Preserve proportions for any non-zero sum (including negative when
    # non_negative=False); only near-zero rows use uniform allocation.
    nonzero = np.abs(row_sums) > 1e-12
    if np.any(nonzero):
        result[nonzero] = work[nonzero] * (totals[nonzero] / row_sums[nonzero])[:, None]

    zero = ~nonzero
    if np.any(zero):
        local_weights = weight_matrix[zero]
        local_sums = np.sum(local_weights, axis=1)
        uniform = np.full((int(np.sum(zero)), n_children), 1.0 / n_children, dtype=float)
        valid = local_sums > 0
        if np.any(valid):
            uniform[valid] = local_weights[valid] / local_sums[valid, None]
        result[zero] = totals[zero, None] * uniform

    residual = totals - np.sum(result, axis=1)
    if np.any(np.abs(residual) > 1e-12):
        largest = np.argmax(result, axis=1)
        result[np.arange(n_rows), largest] += residual
    if non_negative:
        result = np.clip(result, 0.0, None)
    return result


def check_cross_regime_error_guard(
    y_true: Any,
    y_pred: Any,
    regime_labels: Any,
    *,
    max_regime_ratio: float = 3.0,
) -> dict[str, Any]:
    """Compares RMSE by regime and detects relative error explosions."""
    truth = np.asarray(y_true, dtype=float).ravel()
    prediction = np.asarray(y_pred, dtype=float).ravel()
    labels = np.asarray(regime_labels).ravel()
    if not (truth.size == prediction.size == labels.size):
        raise ValueError("y_true, y_pred, and regime_labels must have the same length.")
    if not np.isfinite(max_regime_ratio) or max_regime_ratio <= 0:
        raise ValueError("max_regime_ratio must be positive.")

    mask = np.isfinite(truth) & np.isfinite(prediction)
    by_regime: dict[str, dict[str, float]] = {}
    for label in pd.unique(labels[mask]):
        in_regime = mask & (labels == label)
        if not np.any(in_regime):
            continue
        residuals = prediction[in_regime] - truth[in_regime]
        by_regime[str(label)] = {
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "mae": float(np.mean(np.abs(residuals))),
            "n": int(np.sum(in_regime)),
        }
    if not by_regime:
        return {"passed": True, "by_regime": {}, "ratio": 0.0, "max_rmse": None, "min_rmse": None}
    rmse_values = [entry["rmse"] for entry in by_regime.values()]
    max_rmse = float(np.max(rmse_values))
    min_rmse = float(np.min(rmse_values))
    ratio = max_rmse / max(min_rmse, 1e-9)
    return {
        "passed": bool(ratio <= float(max_regime_ratio)),
        "by_regime": by_regime,
        "ratio": float(ratio),
        "max_rmse": max_rmse,
        "min_rmse": min_rmse,
    }
