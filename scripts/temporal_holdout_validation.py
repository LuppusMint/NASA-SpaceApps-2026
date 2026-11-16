#!/usr/bin/env python3
"""Temporal hold-out validation of the M0-M3 cross-sensor bridge.

The published M0-M3 metrics (``data/processed/benchmarks_m0_m3.json``) come from
5-fold spatial-block cross-validation inside 2012-2020. The product, however, is
used mainly to reconstruct 2000-2011, i.e. years the model never saw. This
script re-runs the *unchanged* model code (``_oof_predictions`` and
``_metric_bundle`` from ``src.models.harmonize_pipeline``) under three fold
schemes on the committed overlap table, so the only difference between schemes
is how rows are assigned to folds:

* ``spatial_block``    - the frozen ``fold_id`` column (same scheme as published);
* ``leave_one_year_out`` - each overlap year 2012..2020 held out in turn;
* ``backward_block``   - two blocks, 2012-2015 and 2016-2020; the reported
  "backward" score is the 2012-2015 block predicted from a model fitted on
  2016-2020 only (the direction of the real backcast).

Deviation from the published run (applies to all three schemes equally): the
M2 covariate-shift weights toward 2021-2026 need ``cube_cell50km_monthly.parquet``
(GitHub Release asset), so ``future=None`` is passed and M2 uses uniform weights.
Compare schemes with each other, not with the published numbers.

Usage::

    python scripts/temporal_holdout_validation.py \
        --out docs/validation/temporal_holdout_validation.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import sha256_file  # noqa: E402
from src.models.harmonize_pipeline import (  # noqa: E402
    INTERVAL_Z,
    MODEL_IDS,
    TARGETS,
    _metric_bundle,
    _oof_predictions,
)

EARLY_YEARS = (2012, 2013, 2014, 2015)
#: Bins of the observed target used to stratify interval coverage.
COVERAGE_BINS = ((0.0, 0.0), (0.0, 10.0), (10.0, 100.0), (100.0, float("inf")))


def _coverage(y_true: np.ndarray, pred: np.ndarray, sigma: np.ndarray) -> dict:
    """Empirical coverage of the published 90% interval ``[max(p-z*s,0), p+z*s]``."""
    lo = np.clip(pred - INTERVAL_Z * sigma, 0.0, None)
    hi = pred + INTERVAL_Z * sigma
    inside = (y_true >= lo) & (y_true <= hi)
    out = {
        "nominal": 0.90,
        "all": float(inside.mean()),
        "observed_positive": float(inside[y_true > 0].mean()) if (y_true > 0).any() else float("nan"),
        "share_observed_zero": float((y_true == 0).mean()),
        "by_observed_bin": {},
    }
    for low, high in COVERAGE_BINS:
        if low == high:
            mask, label = y_true == low, "y=0"
        else:
            mask = (y_true > low) & (y_true <= high)
            label = f"{low:g}<y<={high:g}"
        if mask.any():
            out["by_observed_bin"][label] = {"n": int(mask.sum()), "coverage": float(inside[mask].mean())}
    return out


def _coverage_block(evaluation: dict, target: str, mask: np.ndarray | None = None) -> dict:
    y = evaluation["y"][target]
    preds, sigmas = evaluation["predictions"], evaluation["sigmas"]
    m2, m3 = "M2_hurdle_tweedie_covshift", "M3_simplex_kalman_reconciled"
    sel = slice(None) if mask is None else mask
    return {
        "M2_own_sigma": _coverage(y[sel], preds[m2][target][sel], sigmas[m2][target][sel]),
        "M3_as_scored": _coverage(y[sel], preds[m3][target][sel], sigmas[m3][target][sel]),
        # MODIS-only backcast: production point is the M2 bridge and sigma is
        # the M3-calibrated sigma. For overlap cells with VIIRS, production
        # uses the observation/bridge blend, so this is NOT the overlap
        # production interval.
        "production_backcast_equivalent": _coverage(
            y[sel], preds[m2][target][sel], sigmas[m3][target][sel]
        ),
    }


def _fold_scheme(overlap: pd.DataFrame, scheme: str) -> np.ndarray:
    years = overlap["year"].astype(int).to_numpy()
    if scheme == "spatial_block":
        return overlap["fold_id"].astype(int).to_numpy()
    if scheme == "leave_one_year_out":
        return years - years.min()
    if scheme == "backward_block":
        return np.where(np.isin(years, EARLY_YEARS), 0, 1)
    raise ValueError(scheme)


def _annual_bias(frame: pd.DataFrame, pred: np.ndarray, target: str) -> dict[str, float]:
    """Relative error (%) of the South America annual total per held-out year."""
    tmp = pd.DataFrame({"year": frame["year"].astype(int), "y": frame[target], "p": pred})
    totals = tmp.groupby("year")[["y", "p"]].sum()
    return {
        str(int(year)): float(100.0 * (row.p - row.y) / row.y) if row.y > 0 else float("nan")
        for year, row in totals.iterrows()
    }


def run_scheme(overlap: pd.DataFrame, scheme: str, *, max_train_rows: int, seed: int) -> dict:
    frame = overlap.copy()
    frame["fold_id"] = _fold_scheme(frame, scheme)
    started = time.time()
    evaluation = _oof_predictions(frame, None, max_train_rows=max_train_rows, seed=seed)
    elapsed = time.time() - started

    unique_cell_months = int(
        frame[["year_month", "cell_id_50km"]].drop_duplicates().shape[0]
    )
    out: dict = {
        "seconds": round(elapsed, 1),
        "n_folds": int(frame["fold_id"].nunique()),
        "n_rows": int(len(frame)),
        "n_unique_cell_months": unique_cell_months,
        "replication_note": (
            "Each cell-month appears once as TERRA_AQUA and once as TERRA_ONLY "
            "(Aqua signals masked, same target). Rows are not independent "
            "cell-months; metrics are reported per regime."
        ),
        "models": {},
    }
    early_mask = frame["year"].astype(int).isin(EARLY_YEARS).to_numpy()
    regimes = sorted(str(value) for value in frame["regime_mode"].unique())
    for model in MODEL_IDS:
        out["models"][model] = {}
        for target in TARGETS:
            y_true = evaluation["y"][target]
            y_pred = np.clip(evaluation["predictions"][model][target], 0.0, None)
            bundle = _metric_bundle(y_true, y_pred, evaluation["folds"])
            entry = {
                "mean_fold": bundle["mean"],
                "pooled_oof": bundle["oof"],
                "annual_total_bias_pct": _annual_bias(frame, y_pred, target),
                "by_regime": {},
            }
            for regime in regimes:
                regime_mask = (frame["regime_mode"].astype(str) == regime).to_numpy()
                regime_entry = {
                    "n_rows": int(regime_mask.sum()),
                    "n_unique_cell_months": int(
                        frame.loc[regime_mask, ["year_month", "cell_id_50km"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "pooled_oof": _metric_bundle(
                        y_true[regime_mask],
                        y_pred[regime_mask],
                        np.zeros(int(regime_mask.sum())),
                    )["oof"],
                    "annual_total_bias_pct": _annual_bias(
                        frame.loc[regime_mask].reset_index(drop=True),
                        y_pred[regime_mask],
                        target,
                    ),
                }
                if scheme == "backward_block":
                    back_mask = early_mask & regime_mask
                    forward_mask = (~early_mask) & regime_mask
                    regime_entry["backward_2012_2015_from_2016_2020"] = _metric_bundle(
                        y_true[back_mask], y_pred[back_mask], np.zeros(int(back_mask.sum()))
                    )["oof"]
                    regime_entry["forward_2016_2020_from_2012_2015"] = _metric_bundle(
                        y_true[forward_mask],
                        y_pred[forward_mask],
                        np.zeros(int(forward_mask.sum())),
                    )["oof"]
                entry["by_regime"][regime] = regime_entry
            if scheme == "backward_block":
                # Fold 0 = 2012-2015, predicted by a model fitted on 2016-2020 only.
                backward = _metric_bundle(
                    y_true[early_mask], y_pred[early_mask], np.zeros(int(early_mask.sum()))
                )
                forward = _metric_bundle(
                    y_true[~early_mask], y_pred[~early_mask], np.zeros(int((~early_mask).sum()))
                )
                entry["backward_2012_2015_from_2016_2020"] = backward["oof"]
                entry["forward_2016_2020_from_2012_2015"] = forward["oof"]
            out["models"][model][target] = entry
    out["interval_coverage_90"] = {target: _coverage_block(evaluation, target) for target in TARGETS}
    out["interval_coverage_90_note"] = (
        "production_backcast_equivalent = M2 point + M3 sigma on MODIS-only rows. "
        "It is not the overlap production interval, where observation and bridge are blended."
    )
    out["interval_coverage_90_by_regime"] = {
        target: {
            regime: _coverage_block(
                evaluation,
                target,
                (frame["regime_mode"].astype(str) == regime).to_numpy(),
            )
            for regime in regimes
        }
        for target in TARGETS
    }
    if scheme == "backward_block":
        out["interval_coverage_90_backward_2012_2015"] = {
            target: _coverage_block(evaluation, target, early_mask) for target in TARGETS
        }
        out["interval_coverage_90_backward_2012_2015_by_regime"] = {
            target: {
                regime: _coverage_block(
                    evaluation,
                    target,
                    early_mask & (frame["regime_mode"].astype(str) == regime).to_numpy(),
                )
                for regime in regimes
            }
            for target in TARGETS
        }
    out["m3_blend_weights"] = {
        target: evaluation["blend_info"][target]["weights"] for target in TARGETS
    }
    out["m3_fallback_best_single"] = {
        target: evaluation["blend_info"][target]["fallback_best_single"] for target in TARGETS
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out", default="docs/validation/temporal_holdout_validation.json")
    parser.add_argument("--max-train-rows", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["spatial_block", "leave_one_year_out", "backward_block"],
    )
    parser.add_argument("--limit-rows", type=int, default=0, help="smoke-test subsample (0 = all)")
    args = parser.parse_args(argv)

    overlap_path = (ROOT / args.processed_dir / "overlap_cell50km_2012_2020.parquet").resolve()
    overlap = pd.read_parquet(overlap_path)
    if args.limit_rows:
        overlap = overlap.sample(n=min(args.limit_rows, len(overlap)), random_state=args.seed)
    overlap = overlap.sort_values(["year_month", "cell_id_50km", "regime_mode"]).reset_index(drop=True)

    result = {
        "schema": "temporal_holdout_validation_v3",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": {
            "path": "data/processed/overlap_cell50km_2012_2020.parquet",
            "sha256": sha256_file(overlap_path),
            "rows_used": int(len(overlap)),
            "unique_cell_months": int(
                overlap[["year_month", "cell_id_50km"]].drop_duplicates().shape[0]
            ),
            "smoke_test_subsample": bool(args.limit_rows),
        },
        "replication": {
            "rows": int(len(overlap)),
            "unique_cell_months": int(
                overlap[["year_month", "cell_id_50km"]].drop_duplicates().shape[0]
            ),
            "regimes": sorted(str(value) for value in overlap["regime_mode"].unique()),
            "note": (
                "Every cell-month has a real TERRA_AQUA row and a TERRA_ONLY replica "
                "with Aqua signals masked and the same target. Rows are not "
                "independent cell-months; metrics are reported per regime."
            ),
        },
        "settings": {
            "max_train_rows": args.max_train_rows,
            "seed": args.seed,
            "m2_covariate_shift": "disabled (future=None, uniform weights) for all schemes",
            "python": platform.python_version(),
        },
        "schemes": {},
    }
    for scheme in args.schemes:
        print(f"[{dt.datetime.now():%H:%M:%S}] running {scheme} ...", flush=True)
        result["schemes"][scheme] = run_scheme(
            overlap, scheme, max_train_rows=args.max_train_rows, seed=args.seed
        )
        print(f"[{dt.datetime.now():%H:%M:%S}] done {scheme} in {result['schemes'][scheme]['seconds']} s", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
