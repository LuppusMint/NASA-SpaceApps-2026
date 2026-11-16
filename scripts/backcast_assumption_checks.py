#!/usr/bin/env python3
"""Evidence on the two assumptions behind the 2000-2011 backcast.

1. **Ratio stationarity.** Every MODIS-to-VIIRS backcast assumes the VIIRS/MODIS
   relationship of 2012-2020 also held before 2012. That cannot be tested
   directly, but a drift of the ratio *inside* 2012-2020 bounds how wrong it
   could be. For each MODIS regime (Terra+Aqua, Terra-only) and the largest
   fire countries, this script fits a Theil-Sen trend to the log annual ratio
   ``sum(VIIRS-scale target) / sum(MODIS)`` and reports the change that trend
   would imply for the ratio at 2006 (middle of the backcast), relative to the
   2012-2020 centre, if it continued backwards. A positive value means the true
   2006 ratio would be higher than the fitted one, so the backcast would be low.

2. **Relay steps vs natural variability.** The relay guard passes whenever the
   harmonized 12-month step is smaller than the raw one. This script measures
   how large the same 12-month-after / 12-month-before ratio is in the
   harmonized South America series at start months whose 24-month window
   contains no sensor transition, and places each relay step in that
   distribution.

Inputs (all committed): ``overlap_cell50km_2012_2020.parquet``,
``harmonized_south_america_monthly.parquet``, ``benchmarks_m0_m3.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TRANSITIONS = {"2002": "2002-07", "2012": "2012-01", "2018": "2018-04", "2021": "2021-01", "2023": "2023-01"}
BACKCAST_MID_YEAR = 2006
TOP_COUNTRIES = 6


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trend(years: np.ndarray, ratios: np.ndarray) -> dict:
    """Theil-Sen trend of log(ratio) with 95% CI, as %/year and implied 2006 ratio change."""
    log_r = np.log(ratios)
    slope, intercept, lo, hi = stats.theilslopes(log_r, years, alpha=0.95)
    centre = float(np.mean(years))
    years_back = BACKCAST_MID_YEAR - centre

    def pct(value: float) -> float:
        return float(100.0 * (math.exp(value) - 1.0))

    return {
        "annual_ratio": {str(int(y)): float(r) for y, r in zip(years, ratios)},
        "coefficient_of_variation_pct": float(100.0 * np.std(ratios, ddof=1) / np.mean(ratios)),
        "trend_pct_per_year": pct(slope),
        "trend_pct_per_year_ci95": [pct(lo), pct(hi)],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "implied_ratio_change_at_2006_pct": pct(slope * years_back),
        "implied_ratio_change_at_2006_pct_ci95": sorted([pct(lo * years_back), pct(hi * years_back)]),
    }


def ratio_drift(overlap: pd.DataFrame) -> dict:
    out: dict = {"by_regime": {}, "by_country_terra_aqua": {}}
    pairs = (("srehi", "target_srehi", "modis_events"), ("frp_mw", "target_frp_mw", "modis_frp_mw"))
    for regime, frame in overlap.groupby("regime_mode"):
        annual = frame.groupby("year")[["target_srehi", "modis_events", "target_frp_mw", "modis_frp_mw"]].sum()
        out["by_regime"][regime] = {
            name: _trend(annual.index.to_numpy(float), (annual[num] / annual[den]).to_numpy(float))
            for name, num, den in pairs
        }
    ta = overlap[overlap["regime_mode"] == "TERRA_AQUA"]
    top = ta.groupby("country_name")["target_srehi"].sum().nlargest(TOP_COUNTRIES).index
    for country in top:
        annual = ta[ta["country_name"] == country].groupby("year")[["target_srehi", "modis_events"]].sum()
        annual = annual[annual["modis_events"] > 0]
        out["by_country_terra_aqua"][country] = _trend(
            annual.index.to_numpy(float), (annual["target_srehi"] / annual["modis_events"]).to_numpy(float)
        )
    return out


def natural_variability(sa: pd.DataFrame, benchmarks: dict) -> dict:
    sa = sa[~sa["is_partial_month"].astype(bool)].copy()
    sa["period"] = sa["year"].astype(int) * 12 + sa["month"].astype(int)
    sa = sa.sort_values("period")
    starts_t = [int(v[:4]) * 12 + int(v[5:]) for v in TRANSITIONS.values()]
    first, last = int(sa["period"].min()), int(sa["period"].max())
    out: dict = {"window_months": 12}
    for name, column in (("srehi", "harmonized_srehi"), ("frp_mw", "harmonized_frp_mw")):
        series = sa.set_index("period")[column]
        values = []
        for start in range(first + 12, last - 11 + 1):
            if any(start - 12 <= t < start + 12 for t in starts_t):
                continue
            before = series.reindex(range(start - 12, start))
            after = series.reindex(range(start, start + 12))
            if before.isna().any() or after.isna().any() or before.mean() <= 0:
                continue
            values.append(abs(math.log(after.mean() / before.mean())))
        values = np.asarray(values)
        quant = {f"p{q}": float(math.exp(np.quantile(values, q / 100))) for q in (50, 75, 90, 95)}
        steps = {}
        for label in TRANSITIONS:
            item = benchmarks["transitions"][label][name]
            h_abs = abs(math.log(item["harmonized_ratio"]))
            r_abs = abs(math.log(item["raw_ratio"]))
            steps[label] = {
                "raw_ratio": item["raw_ratio"],
                "harmonized_ratio": item["harmonized_ratio"],
                "harmonized_percentile_in_natural": float(100.0 * np.mean(values <= h_abs)),
                "raw_percentile_in_natural": float(100.0 * np.mean(values <= r_abs)),
                "harmonized_within_natural_p90": bool(h_abs <= np.quantile(values, 0.90)),
            }
        out[name] = {
            "n_clean_start_months": int(len(values)),
            "natural_abs_step_as_ratio": quant,
            "relay_steps": steps,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out", default="docs/validation/backcast_assumption_checks.json")
    args = parser.parse_args(argv)
    processed = ROOT / args.processed_dir
    overlap_path = processed / "overlap_cell50km_2012_2020.parquet"
    sa_path = processed / "harmonized_south_america_monthly.parquet"
    bench_path = processed / "benchmarks_m0_m3.json"
    overlap = pd.read_parquet(overlap_path)
    sa = pd.read_parquet(sa_path)
    benchmarks = json.loads(bench_path.read_text(encoding="utf-8"))
    result = {
        "schema": "backcast_assumption_checks_v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": {
            str(p.relative_to(ROOT)).replace("\\", "/"): _sha256(p) for p in (overlap_path, sa_path, bench_path)
        },
        "ratio_drift_2012_2020": ratio_drift(overlap),
        "relay_steps_vs_natural_variability": natural_variability(sa, benchmarks),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
