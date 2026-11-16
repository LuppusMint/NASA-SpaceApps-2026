#!/usr/bin/env python3
"""Analítica del calendario de quemas sobre la serie armonizada 2000-2026.

Genera la climatología congelada ``2001-01 .. 2020-12``, las matrices de
severidad mensual, las tendencias Hamed-Rao/Theil-Sen con control FDR, los
ajustes GEV y las curvas de inicio temprano de 90 días.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models import (  # noqa: E402
    connect_duckdb,
    parquet_sql,
    write_parquet,
)

CLIMATOLOGY_START = 2001
CLIMATOLOGY_END = 2020
TREND_START = 2001
TREND_END = 2025

CLIMATOLOGY_KEYS = ("geo_level", "iso_a3", "country_name", "admin1_name", "month")

CLIMATOLOGY_COLUMNS = (
    "geo_level",
    "iso_a3",
    "country_name",
    "admin1_name",
    "month",
    "n_years",
    "srehi_p50",
    "srehi_p25",
    "srehi_p75",
    "srehi_iqr",
    "srehi_p90",
    "srehi_p95",
    "srehi_mean",
    "srehi_std",
    "frp_p50",
    "frp_p25",
    "frp_p75",
    "frp_iqr",
    "frp_p90",
    "frp_p95",
    "frp_mean",
    "frp_std",
)

_MONTHLY_COLUMNS = (
    "year",
    "month",
    "year_month",
    "country_name",
    "iso_a3",
    "admin1_name",
    "raw_all_det_count",
    "clean_det_count",
    "overpass_event_count",
    "harmonized_srehi",
    "harmonized_frp_mw",
    "constellation_regime",
    "data_maturity",
    "is_in_sama",
    "intersects_sama",
)


def _read_monthly(con: Any, path: Path, geo_level: str) -> pd.DataFrame:
    """Lee sólo las columnas necesarias de un cubo mensual armonizado."""
    has_admin1 = geo_level == "admin1"
    admin_expr = "admin1_name" if has_admin1 else "CAST('' AS VARCHAR) AS admin1_name"
    columns = ", ".join(
        column for column in _MONTHLY_COLUMNS if column != "admin1_name"
    )
    query = f"""
        SELECT {columns}, {admin_expr}
        FROM {parquet_sql(path)}
        ORDER BY year, month, iso_a3, admin1_name
    """
    frame = con.execute(query).fetch_df()
    frame["geo_level"] = geo_level
    frame["country_name"] = frame["country_name"].fillna("UNKNOWN").astype(str)
    frame["iso_a3"] = frame["iso_a3"].fillna("UNK").astype(str)
    frame["admin1_name"] = frame["admin1_name"].fillna("").astype(str)
    frame["is_in_sama"] = frame["is_in_sama"].fillna(False).astype(bool)
    frame["intersects_sama"] = frame["intersects_sama"].fillna(False).astype(bool)
    for column in ("raw_all_det_count", "clean_det_count", "overpass_event_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype("int64")
    for column in ("harmonized_srehi", "harmonized_frp_mw"):
        frame[column] = (
            pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
        )
    return frame


def _quantile_frame(frame: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    grouped = frame.groupby(list(CLIMATOLOGY_KEYS), dropna=False, sort=True)
    stats = grouped[value_col].agg(["count", "mean", "std"]).reset_index()
    stats = stats.rename(
        columns={
            "count": "n_years",
            f"mean": f"{prefix}_mean",
            "std": f"{prefix}_std",
        }
    )
    quantiles = (
        grouped[value_col]
        .quantile([0.25, 0.5, 0.75, 0.9, 0.95])
        .unstack()
        .reset_index()
    )
    quantiles = quantiles.rename(
        columns={
            0.25: f"{prefix}_p25",
            0.5: f"{prefix}_p50",
            0.75: f"{prefix}_p75",
            0.9: f"{prefix}_p90",
            0.95: f"{prefix}_p95",
        }
    )
    merged = stats.merge(quantiles, on=list(CLIMATOLOGY_KEYS), how="left")
    merged[f"{prefix}_iqr"] = merged[f"{prefix}_p75"] - merged[f"{prefix}_p25"]
    return merged


def build_climatology(admin1: pd.DataFrame, country: pd.DataFrame) -> pd.DataFrame:
    """Climatología congelada por mes calendario (excluye 2000 y >=2021)."""
    frames: list[pd.DataFrame] = []
    for frame in (admin1, country):
        frozen = frame[
            (frame["year"] >= CLIMATOLOGY_START)
            & (frame["year"] <= CLIMATOLOGY_END)
        ].copy()
        if frozen.empty:
            continue
        srehi = _quantile_frame(frozen, "harmonized_srehi", "srehi")
        frp = _quantile_frame(frozen, "harmonized_frp_mw", "frp")
        # ``n_years`` se conserva de la serie SREHI (idéntico para FRP).
        frp = frp.drop(columns=["n_years"])
        merged = srehi.merge(frp, on=list(CLIMATOLOGY_KEYS), how="outer")
        frames.append(merged)
    if not frames:
        raise ValueError("No hay filas 2001-2020 para construir la climatología.")
    climatology = pd.concat(frames, ignore_index=True)
    for column in CLIMATOLOGY_COLUMNS:
        if column not in climatology.columns:
            climatology[column] = np.nan
    climatology = climatology[list(CLIMATOLOGY_COLUMNS)].sort_values(
        ["geo_level", "iso_a3", "admin1_name", "month"]
    )
    return climatology.reset_index(drop=True)


def _severity(
    values: np.ndarray,
    p25: np.ndarray,
    p75: np.ndarray,
    p90: np.ndarray,
    z: np.ndarray,
    n_years: np.ndarray,
) -> np.ndarray:
    critical = (values > p90) | (z > 2.0)
    high = values > p75
    moderate = values > p25
    tier = np.full(len(values), "LOW_NORMAL", dtype=object)
    tier[moderate] = "MODERATE"
    tier[high] = "HIGH_UNUSUAL"
    tier[critical] = "CRITICAL_EXTREME"
    # A tier computed from fewer than 10 years is not a distribution; keep the
    # value visible but do not call it critical/high.
    tier = np.where(np.asarray(n_years, dtype=float) < 10.0, "LOW_SAMPLE", tier)
    return tier


def build_calendar_matrix(
    monthly: pd.DataFrame, climatology: pd.DataFrame
) -> pd.DataFrame:
    """Matriz mensual con z-score, porcentaje de la mediana y severidad."""
    frame = monthly.copy()
    clim = climatology[
        [
            "geo_level",
            "iso_a3",
            "admin1_name",
            "month",
            "srehi_p25",
            "srehi_p50",
            "srehi_p75",
            "srehi_p90",
            "srehi_mean",
            "srehi_std",
            "frp_p25",
            "frp_p50",
            "frp_p75",
            "frp_p90",
            "frp_mean",
            "frp_std",
            "n_years",
        ]
    ].copy()
    # El país no distingue Admin-1: la climatología de país tiene admin1_name="".
    frame = frame.merge(
        clim.drop(columns=["geo_level"]),
        on=["iso_a3", "admin1_name", "month"],
        how="left",
    )
    srehi = frame["harmonized_srehi"].to_numpy(dtype=float)
    frp = frame["harmonized_frp_mw"].to_numpy(dtype=float)
    n_years = frame["n_years"].fillna(0).to_numpy(dtype=float)

    def _z(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        safe_std = np.where(np.isfinite(std) & (std > 0), std, np.nan)
        raw = (values - mean) / safe_std
        return np.where(np.isfinite(raw), raw, 0.0)

    def _pct(values: np.ndarray, median: np.ndarray) -> np.ndarray:
        safe = np.where(np.isfinite(median) & (median > 0), median, np.nan)
        raw = 100.0 * values / safe
        return np.where(np.isfinite(raw), raw, 0.0)

    frame["z_score_srehi"] = _z(
        srehi,
        frame["srehi_mean"].to_numpy(dtype=float),
        frame["srehi_std"].to_numpy(dtype=float),
    )
    frame["pct_of_median_srehi"] = _pct(
        srehi, frame["srehi_p50"].to_numpy(dtype=float)
    )
    frame["z_score_frp"] = _z(
        frp,
        frame["frp_mean"].to_numpy(dtype=float),
        frame["frp_std"].to_numpy(dtype=float),
    )
    frame["pct_of_median_frp"] = _pct(frp, frame["frp_p50"].to_numpy(dtype=float))
    frame["severity_tier"] = _severity(
        srehi,
        frame["srehi_p25"].to_numpy(dtype=float),
        frame["srehi_p75"].to_numpy(dtype=float),
        frame["srehi_p90"].to_numpy(dtype=float),
        frame["z_score_srehi"].to_numpy(dtype=float),
        n_years,
    )
    frame["severity_tier_frp"] = _severity(
        frp,
        frame["frp_p25"].to_numpy(dtype=float),
        frame["frp_p75"].to_numpy(dtype=float),
        frame["frp_p90"].to_numpy(dtype=float),
        frame["z_score_frp"].to_numpy(dtype=float),
        n_years,
    )
    ordered = [
        "geo_level",
        "year",
        "month",
        "year_month",
        "country_name",
        "iso_a3",
        "admin1_name",
        "raw_all_det_count",
        "clean_det_count",
        "overpass_event_count",
        "harmonized_srehi",
        "harmonized_frp_mw",
        "srehi_p25",
        "srehi_p50",
        "srehi_p75",
        "srehi_p90",
        "srehi_mean",
        "srehi_std",
        "z_score_srehi",
        "pct_of_median_srehi",
        "frp_p25",
        "frp_p50",
        "frp_p75",
        "frp_p90",
        "frp_mean",
        "frp_std",
        "n_years",
        "z_score_frp",
        "pct_of_median_frp",
        "severity_tier",
        "severity_tier_frp",
        "constellation_regime",
        "data_maturity",
        "is_in_sama",
        "intersects_sama",
    ]
    return frame[ordered].sort_values(
        ["year", "month", "iso_a3", "admin1_name"]
    ).reset_index(drop=True)


def _annual_totals(monthly: pd.DataFrame, value_col: str) -> pd.DataFrame:
    frame = monthly[
        (monthly["year"] >= TREND_START) & (monthly["year"] <= TREND_END)
    ].copy()
    keys = ["geo_level", "iso_a3", "country_name", "admin1_name", "year"]
    return (
        frame.groupby(keys, dropna=False, as_index=False)[value_col]
        .sum()
        .sort_values(["geo_level", "iso_a3", "admin1_name", "year"])
    )


def _annual_maxima(monthly: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Annual block maxima of the monthly series (for the GEV fit)."""
    frame = monthly[
        (monthly["year"] >= TREND_START) & (monthly["year"] <= TREND_END)
    ].copy()
    group_keys = ["geo_level", "iso_a3", "country_name", "admin1_name"]
    keys = group_keys + ["year"]
    observed = (
        frame.groupby(keys, dropna=False, as_index=False)[value_col]
        .max()
        .rename(columns={value_col: "annual_max"})
    )
    # A year with no rows means no detections; its annual block maximum is 0,
    # not a missing block. Complete the 2001-2025 grid for every geography.
    grid = observed[group_keys].drop_duplicates()
    years = pd.DataFrame({"year": list(range(TREND_START, TREND_END + 1))})
    completed = grid.merge(years, how="cross").merge(
        observed, on=keys, how="left"
    )
    completed["annual_max"] = completed["annual_max"].fillna(0.0)
    return completed.sort_values(["geo_level", "iso_a3", "admin1_name", "year"])


def build_trends(
    admin1: pd.DataFrame,
    country: pd.DataFrame,
    *,
    gev_bootstrap: int = 0,
) -> pd.DataFrame:
    """Tendencias Hamed-Rao + Theil-Sen, FDR global y GEV de totales anuales."""
    from src.core.extremes import fit_gev
    from src.core.trends import (
        benjamini_hochberg_fdr,
        hamed_rao_mk_test,
        sens_slope,
    )

    rows: list[dict[str, Any]] = []
    for frame in (admin1, country):
        for metric, column in (
            ("harmonized_srehi", "harmonized_srehi"),
            ("harmonized_frp_mw", "harmonized_frp_mw"),
        ):
            annual = _annual_totals(frame, column)
            maxima = _annual_maxima(frame, column)
            group_keys = ["geo_level", "iso_a3", "country_name", "admin1_name"]
            for keys, group in annual.groupby(group_keys, dropna=False, sort=True):
                group = group.sort_values("year")
                years = group["year"].to_numpy(dtype=float)
                values = group[column].to_numpy(dtype=float)
                finite = np.isfinite(values) & np.isfinite(years)
                years = years[finite]
                values = values[finite]
                max_mask = np.ones(len(maxima), dtype=bool)
                for column_key, value_key in zip(group_keys, keys):
                    max_mask &= maxima[column_key].astype(str).to_numpy() == str(value_key)
                max_values = maxima.loc[max_mask, "annual_max"].to_numpy(dtype=float)
                max_values = max_values[np.isfinite(max_values)]
                if values.size < 3:
                    continue
                mk = hamed_rao_mk_test(values, years)
                sen = sens_slope(values, years)
                gev: dict[str, Any] = {}
                gev_ok = False
                gev_reject_reason = "insufficient_blocks"
                levels: dict[str, Any] = {}
                active_years = int(np.sum(max_values > 0))
                if max_values.size < 10:
                    gev_reject_reason = "insufficient_blocks"
                elif active_years < 10:
                    gev_reject_reason = "insufficient_active_years"
                elif float(np.std(max_values)) <= 0.0:
                    gev_reject_reason = "zero_variance"
                else:
                    try:
                        candidate = fit_gev(
                            max_values,
                            return_periods=(10, 20, 50),
                            n_bootstrap=int(gev_bootstrap),
                            seed=42,
                        )
                        candidate_levels = candidate.get("return_levels", {})
                        max_observed = float(np.max(max_values))
                        median_observed = float(np.median(max_values))
                        rl10 = float(candidate_levels.get("10", float("nan")))
                        rl20 = float(candidate_levels.get("20", float("nan")))
                        rl50 = float(candidate_levels.get("50", float("nan")))
                        xi = float(candidate.get("xi", float("nan")))
                        sane = (
                            np.isfinite(rl10)
                            and np.isfinite(rl20)
                            and np.isfinite(rl50)
                            and np.isfinite(xi)
                            and max_observed > 0
                            and 0.0 < rl10 <= rl20 <= rl50
                            and rl10 >= median_observed
                            and rl10 <= 10.0 * max_observed
                            and rl50 <= 10.0 * max_observed
                            and -0.5 <= xi <= 0.5
                        )
                        if sane:
                            gev = candidate
                            levels = candidate_levels
                            gev_ok = True
                            gev_reject_reason = ""
                        else:
                            gev_reject_reason = "return_level_sanity"
                    except Exception as exc:  # noqa: BLE001 - series degeneradas
                        gev_reject_reason = f"fit_error:{type(exc).__name__}"
                geo_level, iso_a3, country_name, admin1_name = keys
                p_value = float(mk.get("p_value", 1.0))
                slope = float(sen.get("slope", float("nan")))
                if not np.isfinite(slope):
                    slope = float(mk.get("slope", float("nan")))
                max_observed = float(np.max(max_values)) if max_values.size else float("nan")
                ci = gev.get("return_level_cis", {}) if gev else {}
                rl10_ci = (ci.get("10") or [float("nan"), float("nan")]) if gev else [float("nan"), float("nan")]
                rl50_ci = (ci.get("50") or [float("nan"), float("nan")]) if gev else [float("nan"), float("nan")]
                rows.append(
                    {
                        "geo_level": str(geo_level),
                        "iso_a3": str(iso_a3),
                        "country_name": str(country_name),
                        "admin1_name": str(admin1_name),
                        "metric": metric,
                        "n_years": int(values.size),
                        "low_sample": bool(values.size < 10),
                        "gev_low_sample": bool(max_values.size < 10),
                        "start_year": int(years.min()),
                        "end_year": int(years.max()),
                        "slope": slope,
                        "intercept": float(sen.get("intercept", float("nan"))),
                        "slope_ci_low": float(sen.get("ci_low", float("nan"))),
                        "slope_ci_high": float(sen.get("ci_high", float("nan"))),
                        "z": float(mk.get("z", float("nan"))),
                        "p_value": p_value,
                        "method": str(mk.get("method", "hamed_rao_mk")),
                        "ess": float(mk.get("ess", values.size)),
                        "annual_mean": float(np.mean(values)),
                        "gev_input": "annual_max_monthly",
                        "gev_unit": "monthly maximum (same unit as monthly series)",
                        "gev_estimator": str(gev.get("estimator", "")) if gev else "",
                        "gev_fit_ok": bool(gev_ok),
                        "gev_reject_reason": str(gev_reject_reason),
                        "gev_n": int(max_values.size),
                        "gev_active_years": int(active_years),
                        "gev_max_observed": max_observed,
                        "gev_xi": float(gev.get("xi", float("nan"))) if gev else float("nan"),
                        "gev_mu": float(gev.get("mu", float("nan"))) if gev else float("nan"),
                        "gev_sigma": float(gev.get("sigma", float("nan"))) if gev else float("nan"),
                        "gev_rl10": float(levels.get("10", float("nan"))),
                        "gev_rl20": float(levels.get("20", float("nan"))),
                        "gev_rl50": float(levels.get("50", float("nan"))),
                        "gev_rl10_ci_low": float(rl10_ci[0]),
                        "gev_rl10_ci_high": float(rl10_ci[1]),
                        "gev_rl50_ci_low": float(rl50_ci[0]),
                        "gev_rl50_ci_high": float(rl50_ci[1]),
                        "rl10": float(levels.get("10", float("nan"))),
                        "rl20": float(levels.get("20", float("nan"))),
                        "rl50": float(levels.get("50", float("nan"))),
                        "gev_ks_p_value": float(gev.get("ks_p_value", float("nan"))) if gev else float("nan"),
                    }
                )

    if not rows:
        raise ValueError("No hay series anuales suficientes para tendencias.")
    trends = pd.DataFrame(rows)
    fdr = benjamini_hochberg_fdr(trends["p_value"].to_numpy(dtype=float), alpha=0.05)
    trends["q_value"] = np.asarray(fdr["q_values"], dtype=float)
    trends["significant_fdr"] = np.asarray(fdr["rejected"], dtype=bool)
    trends["trend_direction"] = np.where(
        ~trends["significant_fdr"].to_numpy(dtype=bool),
        "no_significant_trend",
        np.where(trends["slope"].to_numpy(dtype=float) > 0, "increasing", "decreasing"),
    )
    return trends.sort_values(["geo_level", "iso_a3", "admin1_name", "metric"]).reset_index(
        drop=True
    )


def build_early_onset(
    con: Any,
    monthly_path: Path,
    daily_path: Path,
    out_path: Path,
    *,
    step_days: int = 5,
    max_bytes: int = 25 * 1024 * 1024,
) -> Path:
    """Curva causal de 90 días y banderas de inicio temprano.

    La serie diaria se escala por la razón mensual armonizada
    ``harmonized_srehi / GREATEST(overpass_event_count, 1)``. La ventana es
    trailing real de 90 días y cruza el cambio de año; los primeros 89 días
    del registro se marcan ``INSUFFICIENT_HISTORY`` y no se usan para los
    umbrales. Los umbrales p75/p90 se congelan con ``2001-2020`` usando solo
    ventanas completas y al menos 10 años por día calendario.
    """
    con.execute("DROP TABLE IF EXISTS daily_agg")
    con.execute(
        f"""
        CREATE TEMP TABLE daily_agg AS
        SELECT
            "year", doy, iso_a3, admin1_name,
            MAX(country_name) AS country_name,
            make_date("year", 1, 1) + CAST(doy - 1 AS INTEGER) AS acq_date,
            SUM(overpass_event_count)::DOUBLE AS events,
            SUM(clean_det_count)::DOUBLE AS clean_det,
            SUM(frp_sum_mw)::DOUBLE AS frp
        FROM {parquet_sql(daily_path)}
        GROUP BY "year", doy, iso_a3, admin1_name
        """
    )
    con.execute("DROP TABLE IF EXISTS monthly_ratio")
    con.execute(
        f"""
        CREATE TEMP TABLE monthly_ratio AS
        SELECT
            "year", "month", iso_a3, admin1_name,
            MAX(country_name) AS country_name,
            MAX(harmonized_srehi) AS harmonized_srehi,
            MAX(overpass_event_count) AS monthly_events,
            MAX(harmonized_srehi)
                / GREATEST(MAX(overpass_event_count), 1.0) AS ratio
        FROM {parquet_sql(monthly_path)}
        GROUP BY "year", "month", iso_a3, admin1_name
        """
    )
    con.execute("DROP TABLE IF EXISTS geo_names")
    con.execute(
        """
        CREATE TEMP TABLE geo_names AS
        SELECT iso_a3, admin1_name, MAX(country_name) AS country_name
        FROM (
            SELECT iso_a3, admin1_name, country_name FROM daily_agg
            UNION ALL
            SELECT iso_a3, admin1_name, country_name FROM monthly_ratio
        )
        GROUP BY iso_a3, admin1_name
        """
    )
    con.execute("DROP TABLE IF EXISTS daily_grid")
    con.execute(
        """
        CREATE TEMP TABLE daily_grid AS
        SELECT
            g.iso_a3,
            g.admin1_name,
            CAST(t.dt AS DATE) AS acq_date,
            CAST(EXTRACT(year FROM t.dt) AS INTEGER) AS "year",
            CAST(EXTRACT(doy FROM t.dt) AS INTEGER) AS doy
        FROM (SELECT DISTINCT iso_a3, admin1_name FROM geo_names) g
        CROSS JOIN (
            SELECT UNNEST(
                generate_series(DATE '2000-11-01', DATE '2026-11-15', INTERVAL 1 DAY)
            ) AS dt
        ) t
        """
    )
    con.execute("DROP TABLE IF EXISTS daily_full")
    con.execute(
        """
        CREATE TEMP TABLE daily_full AS
        SELECT
            g.iso_a3, n.country_name, g.admin1_name, g."year", g.doy, g.acq_date,
            COALESCE(d.events, 0.0) AS events,
            COALESCE(d.clean_det, 0.0) AS clean_det,
            COALESCE(d.events, 0.0) * COALESCE(r.ratio, 1.0) AS scaled_events
        FROM daily_grid g
        LEFT JOIN geo_names n USING (iso_a3, admin1_name)
        LEFT JOIN daily_agg d
            ON d.iso_a3 = g.iso_a3
           AND d.admin1_name = g.admin1_name
           AND d.acq_date = g.acq_date
        LEFT JOIN monthly_ratio r
            ON r.iso_a3 = g.iso_a3
           AND r.admin1_name = g.admin1_name
           AND r."year" = g."year"
           AND r."month" = CAST(EXTRACT(month FROM g.acq_date) AS INTEGER)
        """
    )
    con.execute("DROP TABLE IF EXISTS daily_roll")
    con.execute(
        """
        CREATE TEMP TABLE daily_roll AS
        SELECT
            *,
            COUNT(*) OVER (
                PARTITION BY iso_a3, admin1_name
                ORDER BY acq_date
                ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
            ) AS window_days,
            CASE
                WHEN COUNT(*) OVER (
                    PARTITION BY iso_a3, admin1_name
                    ORDER BY acq_date
                    ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
                ) < 90 THEN 'INSUFFICIENT_HISTORY'
                ELSE 'OK'
            END AS history_status,
            SUM(scaled_events) OVER (
                PARTITION BY iso_a3, admin1_name
                ORDER BY acq_date
                ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
            ) AS srehi_90d
        FROM daily_full
        """
    )
    con.execute("DROP TABLE IF EXISTS thresholds")
    con.execute(
        """
        CREATE TEMP TABLE thresholds AS
        SELECT
            iso_a3, admin1_name, LEAST(doy, 365) AS doy_key,
            COUNT(*) AS n_years,
            CASE WHEN COUNT(*) >= 10 THEN quantile_cont(srehi_90d, 0.75) END AS p75_90d,
            CASE WHEN COUNT(*) >= 10 THEN quantile_cont(srehi_90d, 0.90) END AS p90_90d
        FROM daily_roll
        WHERE "year" BETWEEN 2001 AND 2020
          AND history_status = 'OK'
        GROUP BY iso_a3, admin1_name, LEAST(doy, 365)
        """
    )
    step = max(1, int(step_days))
    query = f"""
        SELECT
            r.iso_a3,
            COALESCE(r.country_name, 'UNKNOWN') AS country_name,
            r.admin1_name,
            r."year" AS year,
            r.doy,
            r.acq_date,
            r.events,
            r.scaled_events,
            r.srehi_90d,
            r.window_days,
            r.history_status,
            t.n_years,
            t.p75_90d,
            t.p90_90d,
            CASE
                WHEN r.history_status = 'INSUFFICIENT_HISTORY' THEN 'INSUFFICIENT_HISTORY'
                WHEN t.p90_90d IS NULL THEN 'NO_THRESHOLD'
                WHEN r.srehi_90d > t.p90_90d THEN 'EARLY_ONSET_CRITICAL'
                WHEN r.srehi_90d > t.p75_90d THEN 'EARLY_ONSET_HIGH'
                ELSE 'NORMAL'
            END AS early_onset_flag
        FROM daily_roll r
        LEFT JOIN thresholds t
            ON t.iso_a3 = r.iso_a3
           AND t.admin1_name = r.admin1_name
           AND t.doy_key = LEAST(r.doy, 365)
        WHERE (r.doy % {step} = 1)
           OR r.doy = date_diff(
                'day', make_date(r."year", 1, 1), make_date(r."year" + 1, 1, 1)
           )
        ORDER BY r.iso_a3, r.admin1_name, r."year", r.doy
    """
    write_parquet(con, query, out_path)
    # Si el archivo excede el presupuesto, se reduce la resolución temporal.
    if out_path.stat().st_size > max_bytes and step < 30:
        return build_early_onset(
            con,
            monthly_path,
            daily_path,
            out_path,
            step_days=step * 2,
            max_bytes=max_bytes,
        )
    return out_path


def run(processed_dir: Path | str, *, gev_bootstrap: int = 0) -> dict[str, Any]:
    processed = Path(processed_dir).expanduser().resolve()
    admin1_path = processed / "harmonized_admin1_monthly.parquet"
    country_path = processed / "harmonized_country_monthly.parquet"
    daily_path = processed / "cube_admin1_daily_season.parquet"
    for required in (admin1_path, country_path, daily_path):
        if not required.is_file():
            raise FileNotFoundError(f"Falta artefacto requerido: {required}")

    con = connect_duckdb(scratch_dir=processed / ".duckdb_tmp")
    try:
        admin1 = _read_monthly(con, admin1_path, "admin1")
        country = _read_monthly(con, country_path, "country")

        climatology = build_climatology(admin1, country)
        matrix_admin1 = build_calendar_matrix(admin1, climatology)
        matrix_country = build_calendar_matrix(country, climatology)
        trends = build_trends(admin1, country, gev_bootstrap=int(gev_bootstrap))

        # DuckDB no conoce los DataFrames hasta registrarlos; se escribe por bloques.
        for name, frame in (
            ("calendar_climatology.parquet", climatology),
            ("calendar_matrix_admin1.parquet", matrix_admin1),
            ("calendar_matrix_country.parquet", matrix_country),
            ("calendar_trends.parquet", trends),
        ):
            con.register("calendar_frame", frame)
            write_parquet(con, "SELECT * FROM calendar_frame", processed / name)
            con.unregister("calendar_frame")

        early_path = build_early_onset(
            con,
            admin1_path,
            daily_path,
            processed / "calendar_early_onset_90d.parquet",
        )
    finally:
        con.close()

    import pyarrow.parquet as pq

    return {
        "climatology_rows": int(len(climatology)),
        "matrix_admin1_rows": int(len(matrix_admin1)),
        "matrix_country_rows": int(len(matrix_country)),
        "trend_rows": int(len(trends)),
        "early_onset_rows": int(pq.read_metadata(early_path).num_rows),
        "early_onset_path": str(early_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument(
        "--gev-bootstrap",
        type=int,
        default=0,
        help="Bootstrap no paramétrico de GEV (0 acelera el pipeline).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args.processed_dir, gev_bootstrap=int(args.gev_bootstrap))
    print(
        f"climatología={result['climatology_rows']} "
        f"matriz_admin1={result['matrix_admin1_rows']} "
        f"matriz_country={result['matrix_country_rows']} "
        f"tendencias={result['trend_rows']}"
    )
    print(f"early onset: {result['early_onset_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
