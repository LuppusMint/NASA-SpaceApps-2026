"""End-to-end tests for the M0-M3 harmonization pipeline on synthetic cubes."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import PRE_DRIFT_END, PRE_DRIFT_START  # noqa: E402

PYTHON = sys.executable


def _month_range(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    year, month = start
    while (year, month) <= end:
        months.append((year, month))
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


def _sensors_for(year: int, month: int) -> list[str]:
    period = year * 12 + month
    sensors = ["MODIS_Terra"]
    if period > 2002 * 12 + 6:
        sensors.append("MODIS_Aqua")
    if period >= 2012 * 12 + 1:
        sensors.append("VIIRS_SNPP")
    if period >= 2018 * 12 + 4:
        sensors.append("VIIRS_N20")
    if period >= 2023 * 12 + 1:
        sensors.append("VIIRS_N21")
    return sensors


CELLS = (
    # (country, iso, admin1, ix, iy, lat, lon, scale, sama)
    ("Brazil", "BRA", "Amazonas", 10, 20, -2.0, -60.0, 26.0, False),
    ("Brazil", "BRA", "Amazonas", 18, 26, -5.0, -64.0, 20.0, False),
    ("Brazil", "BRA", "Mato Grosso", 40, 60, -12.0, -55.0, 30.0, False),
    ("Brazil", "BRA", "Mato Grosso", 48, 66, -15.0, -58.0, 22.0, False),
    ("Colombia", "COL", "Meta", 70, 30, 3.0, -73.0, 16.0, False),
    ("Colombia", "COL", "Meta", 78, 36, 0.0, -70.0, 13.0, False),
    ("Colombia", "COL", "Antioquia", 90, 42, 6.0, -76.0, 11.0, True),
    ("Argentina", "ARG", "Chaco", 110, 100, -25.0, -60.0, 15.0, False),
    ("Argentina", "ARG", "Chaco", 118, 108, -28.0, -63.0, 12.0, False),
)


def _seasonal(month: int) -> float:
    distance = min(abs(month - 9.5), 12.0 - abs(month - 9.5))
    return 0.08 + 1.4 * math.exp(-0.5 * (distance / 1.6) ** 2)


def _build_synthetic_cubes(processed: Path) -> None:
    """Genera cubos C1 sintéticos con relevo de constelación y un extremo 2024."""
    rng = np.random.default_rng(20261115)
    processed.mkdir(parents=True, exist_ok=True)
    months = _month_range((2000, 11), (2026, 11))

    cell_rows: list[dict] = []
    for year, month in months:
        year_month = f"{year:04d}-{month:02d}"
        period = year * 12 + month
        partial = period in (2000 * 12 + 11, 2026 * 12 + 11)
        for country, iso, admin1, ix, iy, lat, lon, scale, sama in CELLS:
            seasonal = _seasonal(month)
            trend = 1.0 + 0.015 * (year - 2000)
            noise = float(rng.lognormal(0.0, 0.25))
            lam = scale * seasonal * trend * noise
            if year == 2024:
                lam *= 6.0
            events_true = int(rng.poisson(max(lam, 0.0)))
            frp_per_event = float(rng.gamma(2.0, 25.0))
            frp_true = events_true * frp_per_event
            for sensor in _sensors_for(year, month):
                factor = {
                    "MODIS_Terra": 0.2,
                    "MODIS_Aqua": 0.2,
                    "VIIRS_SNPP": 1.0,
                    "VIIRS_N20": 0.8,
                    "VIIRS_N21": 0.8,
                }[sensor]
                events_s = int(rng.poisson(events_true * factor))
                if events_s > 0:
                    frp_s = float(frp_true * factor * rng.lognormal(0.0, 0.15))
                else:
                    frp_s = 0.0
                clean = events_s * 2
                night = int(round(clean * 0.3))
                day = clean - night
                high = int(round(clean * 0.6))
                nrt = int(clean) if year >= 2026 else 0
                sp = int(clean) - nrt
                cell_rows.append(
                    {
                        "year": year,
                        "month": month,
                        "year_month": year_month,
                        "cell_ix_50km": ix,
                        "cell_iy_50km": iy,
                        "cell_id_50km": f"{ix}_{iy}",
                        "country_name": country,
                        "iso_a3": iso,
                        "admin1_name": admin1,
                        "sensor_slot": sensor,
                        "is_continental_sa": True,
                        "is_in_sama": bool(sama),
                        "raw_all_det_count": clean + int(rng.poisson(0.5)),
                        "clean_det_count": clean,
                        "static_excluded_det_count": 0,
                        "high_conf_count": high,
                        "night_det_count": night,
                        "day_det_count": day,
                        "active_days": min(28, events_s * 2),
                        "overpass_event_count": events_s,
                        "frp_sum_mw": frp_s,
                        "active_cells_10km": events_s,
                        "mean_lat": lat + float(rng.normal(0.0, 0.03)),
                        "mean_lon": lon + float(rng.normal(0.0, 0.03)),
                        "x_m_center": ix * 50_000.0 + 25_000.0,
                        "y_m_center": iy * 50_000.0 + 25_000.0,
                        "nrt_det_count": nrt,
                        "sp_det_count": sp,
                        "is_nrt_majority": nrt > sp,
                        "is_partial_month": partial,
                    }
                )

    cells = pd.DataFrame(cell_rows)
    cells.to_parquet(
        processed / "cube_cell50km_monthly.parquet", index=False, compression="zstd"
    )

    admin = (
        cells.groupby(
            ["year", "month", "year_month", "country_name", "iso_a3", "admin1_name", "sensor_slot"],
            as_index=False,
        )
        .agg(
            raw_all_det_count=("raw_all_det_count", "sum"),
            clean_det_count=("clean_det_count", "sum"),
            static_excluded_det_count=("static_excluded_det_count", "sum"),
            high_conf_count=("high_conf_count", "sum"),
            night_det_count=("night_det_count", "sum"),
            day_det_count=("day_det_count", "sum"),
            overpass_event_count=("overpass_event_count", "sum"),
            frp_sum_mw=("frp_sum_mw", "sum"),
            unique_active_cells_10km=("active_cells_10km", "max"),
            active_cell_days=("active_days", "sum"),
            nrt_det_count=("nrt_det_count", "sum"),
            sp_det_count=("sp_det_count", "sum"),
            is_partial_month=("is_partial_month", "max"),
            intersects_sama=("is_in_sama", "max"),
        )
        .reset_index(drop=True)
    )
    admin["is_nrt_majority"] = admin["nrt_det_count"] > admin["sp_det_count"]
    admin.to_parquet(
        processed / "cube_admin1_monthly.parquet", index=False, compression="zstd"
    )

    country = (
        admin.groupby(
            ["year", "month", "year_month", "country_name", "iso_a3", "sensor_slot"],
            as_index=False,
        )
        .agg(
            raw_all_det_count=("raw_all_det_count", "sum"),
            clean_det_count=("clean_det_count", "sum"),
            static_excluded_det_count=("static_excluded_det_count", "sum"),
            high_conf_count=("high_conf_count", "sum"),
            night_det_count=("night_det_count", "sum"),
            day_det_count=("day_det_count", "sum"),
            overpass_event_count=("overpass_event_count", "sum"),
            frp_sum_mw=("frp_sum_mw", "sum"),
            unique_active_cells_10km=("unique_active_cells_10km", "max"),
            active_cell_days=("active_cell_days", "sum"),
            nrt_det_count=("nrt_det_count", "sum"),
            sp_det_count=("sp_det_count", "sum"),
            is_partial_month=("is_partial_month", "max"),
            intersects_sama=("intersects_sama", "max"),
        )
        .reset_index(drop=True)
    )
    country["is_nrt_majority"] = country["nrt_det_count"] > country["sp_det_count"]
    country.to_parquet(
        processed / "cube_country_monthly.parquet", index=False, compression="zstd"
    )

    daily_rows: list[dict] = []
    for row in admin.itertuples():
        if row.overpass_event_count <= 0:
            continue
        days_in_month = int(
            (
                dt.date(row.year + (row.month // 12), (row.month % 12) + 1, 1)
                - dt.date(row.year, row.month, 1)
            ).days
        )
        first = dt.date(row.year, row.month, 1)
        doys = np.asarray(
            [(first + dt.timedelta(days=offset)).timetuple().tm_yday for offset in range(days_in_month)]
        )
        weights = 0.15 + np.exp(-0.5 * ((doys - 260.0) / 45.0) ** 2)
        weights = weights / weights.sum()
        counts = rng.multinomial(int(row.overpass_event_count), weights)
        frp_per_event = float(row.frp_sum_mw) / max(float(row.overpass_event_count), 1.0)
        for offset, count in enumerate(counts):
            if count <= 0:
                continue
            acq_date = first + dt.timedelta(days=offset)
            extra = 0
            if (
                row.admin1_name == "Amazonas"
                and row.year == 2024
                and row.month in (5, 6)
                and row.sensor_slot == "VIIRS_SNPP"
            ):
                extra = 12
            total = int(count) + extra
            daily_rows.append(
                {
                    "year": row.year,
                    "acq_date": acq_date,
                    "doy": acq_date.timetuple().tm_yday,
                    "country_name": row.country_name,
                    "iso_a3": row.iso_a3,
                    "admin1_name": row.admin1_name,
                    "sensor_slot": row.sensor_slot,
                    "clean_det_count": total * 2,
                    "frp_sum_mw": total * frp_per_event,
                    "active_cells_10km": total,
                    "overpass_event_count": total,
                }
            )
    daily = pd.DataFrame(daily_rows)
    daily.to_parquet(
        processed / "cube_admin1_daily_season.parquet", index=False, compression="zstd"
    )


def _run_module(module: str, processed: Path) -> subprocess.CompletedProcess:
    command = [PYTHON, "-m", module, "--processed-dir", str(processed)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{module} falló ({result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


class HarmonizationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory(prefix="harmonization-")
        cls.root = Path(cls.tempdir.name)
        cls.processed = cls.root / "data" / "processed"
        _build_synthetic_cubes(cls.processed)

        _run_module("src.models.freeze_validation", cls.processed)
        _run_module("src.models.harmonize_pipeline", cls.processed)
        _run_module("src.models.calendar_analytics", cls.processed)

        cls.splits = json.loads((cls.processed / "splits.json").read_text(encoding="utf-8"))
        cls.records = [
            json.loads(line)
            for line in (cls.processed / "experiments.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.by_id = {record["id"]: record for record in cls.records}
        cls.benchmarks = json.loads(
            (cls.processed / "benchmarks_m0_m3.json").read_text(encoding="utf-8")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    # ------------------------------------------------------------------
    # 1. Particiones sin fuga y gate G2
    # ------------------------------------------------------------------
    def test_splits_have_no_spatial_leakage_and_pass_g2(self) -> None:
        splits = self.splits
        self.assertEqual(splits["pre_drift_window"], "2012-02..2020-12")
        self.assertEqual(splits["block_size_m"], 200_000.0)
        self.assertEqual(splits["folds"], 5)
        self.assertTrue(splits["fingerprint"])
        self.assertEqual(len(splits["assignment"]), splits["rows"])

        fold_by_cell = splits["fold_by_cell"]
        self.assertEqual(len(set(fold_by_cell.values())), 5)
        for cell, fold in fold_by_cell.items():
            self.assertIn(str(fold), {str(value) for value in splits["fold_sizes"]})

        block_folds: dict[str, set[int]] = {}
        for sample_id, block in splits["block_assignment"].items():
            block_folds.setdefault(block, set()).add(int(splits["assignment"][sample_id]))
        leaked = {block: folds for block, folds in block_folds.items() if len(folds) > 1}
        self.assertFalse(leaked, f"Bloques espaciales en más de un fold: {leaked}")

        cell_folds: dict[str, set[int]] = {}
        for sample_id, fold in splits["assignment"].items():
            cell = sample_id.split("|")[0]
            cell_folds.setdefault(cell, set()).add(int(fold))
        self.assertFalse({cell: f for cell, f in cell_folds.items() if len(f) > 1})

        self.assertTrue(splits["temporal_folds"])
        for temporal in splits["temporal_folds"]:
            if temporal.get("min_train_time") and temporal.get("min_val_time"):
                self.assertLessEqual(temporal["min_train_time"], temporal["min_val_time"])

        self.assertEqual(len(splits["fold_sizes"]), 5)
        self.assertEqual(sum(int(v) for v in splits["fold_sizes"].values()), splits["rows"])

    # ------------------------------------------------------------------
    # 2. M0-M3 benchmark and step-change reduction
    # ------------------------------------------------------------------
    def test_experiments_are_comparable_and_m3_improves(self) -> None:
        self.assertEqual(len(self.records), 4)
        self.assertNotIn("baseline_id", self.by_id["M0_raw_unharmonized"])
        baseline = self.by_id["M0_raw_unharmonized"]
        for model_id in (
            "M1_physical_biweight_bridge",
            "M2_hurdle_tweedie_covshift",
            "M3_simplex_kalman_reconciled",
        ):
            record = self.by_id[model_id]
            self.assertEqual(record["baseline_id"], "M0_raw_unharmonized")
            self.assertEqual(record["fold_fingerprint"], baseline["fold_fingerprint"])
            self.assertEqual(record["dataset_sha256"], baseline["dataset_sha256"])
            self.assertIn("oof_m0_m3", record["artifact_hashes"])
            artifact = Path(record["artifact_hashes"]["oof_m0_m3"]["path"])
            self.assertTrue(artifact.is_file())

        for target in ("target_srehi", "target_frp_mw"):
            m0 = self.by_id["M0_raw_unharmonized"]["metrics_by_target"][target]["mean"]
            m3 = self.by_id["M3_simplex_kalman_reconciled"]["metrics_by_target"][target]["mean"]
            for metric in ("rmse", "mae", "rmsle"):
                self.assertLess(
                    m3[metric],
                    0.8 * m0[metric],
                    f"M3 no mejora {metric} de {target}: {m3[metric]} vs {m0[metric]}",
                )

        self.assertGreater(
            self.by_id["M3_simplex_kalman_reconciled"]["metrics_by_target"]["target_srehi"]["mean"]["rmse"],
            0.0,
        )
        self.assertEqual(len(self.by_id), 4)

        for transition in ("2012", "2018"):
            item = self.benchmarks["transitions"][transition]["srehi"]
            self.assertTrue(item["passed"], item)
            self.assertGreater(item["ratio_reduction"], 0.5, item)
            self.assertLess(
                abs(math.log(item["harmonized_step_change_ratio"])),
                abs(math.log(item["raw_step_change_ratio"])),
                item,
            )

    # ------------------------------------------------------------------
    # 3. Conservación jerárquica y no negatividad
    # ------------------------------------------------------------------
    def test_hierarchical_conservation_and_non_negativity(self) -> None:
        admin1 = pd.read_parquet(self.processed / "harmonized_admin1_monthly.parquet")
        country = pd.read_parquet(self.processed / "harmonized_country_monthly.parquet")
        required = {
            "year",
            "month",
            "year_month",
            "country_name",
            "iso_a3",
            "raw_all_det_count",
            "clean_det_count",
            "raw_frp_mw",
            "active_cells_10km",
            "overpass_event_count",
            "harmonized_srehi",
            "harmonized_srehi_lo90",
            "harmonized_srehi_hi90",
            "harmonized_frp_mw",
            "harmonized_frp_lo90",
            "harmonized_frp_hi90",
            "m1_srehi",
            "m2_srehi",
            "constellation_regime",
            "data_maturity",
            "is_in_sama",
            "intersects_sama",
            "static_excluded_det_count",
            "sama_suspect_det_count",
        }
        self.assertTrue(required.issubset(admin1.columns), required - set(admin1.columns))
        self.assertTrue(required.issubset(country.columns), required - set(country.columns))
        self.assertIn("admin1_name", admin1.columns)
        for column in (
            "harmonized_srehi",
            "harmonized_srehi_lo90",
            "harmonized_srehi_hi90",
            "harmonized_frp_mw",
            "harmonized_frp_lo90",
            "harmonized_frp_hi90",
            "m1_srehi",
            "m2_srehi",
        ):
            self.assertTrue((admin1[column] >= 0).all(), column)
        for column in (
            "harmonized_srehi",
            "harmonized_frp_mw",
            "harmonized_srehi_lo90",
            "harmonized_frp_lo90",
        ):
            self.assertTrue((country[column] >= 0).all(), column)

        admin_sum = admin1.groupby(["year_month", "iso_a3"], as_index=False)[
            ["harmonized_srehi", "harmonized_frp_mw"]
        ].sum()
        merged = country.merge(
            admin_sum,
            on=["year_month", "iso_a3"],
            how="inner",
            suffixes=("", "_admin"),
        )
        self.assertTrue(len(merged) > 0)
        self.assertLess(
            float(np.max(np.abs(merged["harmonized_srehi"] - merged["harmonized_srehi_admin"]))),
            1e-6,
        )
        self.assertLess(
            float(np.max(np.abs(merged["harmonized_frp_mw"] - merged["harmonized_frp_mw_admin"]))),
            1e-6,
        )
        self.assertTrue(
            np.all(
                admin1["harmonized_srehi_lo90"].to_numpy()
                <= admin1["harmonized_srehi"].to_numpy() + 1e-9
            )
        )

    # ------------------------------------------------------------------
    # 4. Climatología congelada, tendencias, GEV y onset temprano
    # ------------------------------------------------------------------
    def test_frozen_climatology_trends_gev_and_early_onset(self) -> None:
        climatology = pd.read_parquet(self.processed / "calendar_climatology.parquet")
        self.assertIn("srehi_p50", climatology.columns)
        self.assertIn("frp_p90", climatology.columns)
        self.assertTrue((climatology["n_years"] > 0).all())

        matrix_admin1 = pd.read_parquet(self.processed / "calendar_matrix_admin1.parquet")
        self.assertTrue(
            (
                matrix_admin1.loc[matrix_admin1["year"] == 2024, "severity_tier"]
                == "CRITICAL_EXTREME"
            ).any()
        )
        self.assertTrue(
            matrix_admin1["severity_tier"].isin(
                {"CRITICAL_EXTREME", "HIGH_UNUSUAL", "MODERATE", "LOW_NORMAL"}
            ).all()
        )
        self.assertTrue(np.isfinite(matrix_admin1["z_score_srehi"]).all())
        self.assertTrue(np.isfinite(matrix_admin1["pct_of_median_srehi"]).all())

        # La climatología no se contamina con el extremo de 2024: al alterarlo,
        # el archivo de climatología debe ser idéntico.
        alternate = self.root / "alternate" / "data" / "processed"
        shutil.copytree(self.processed, alternate)
        for filename in (
            "harmonized_admin1_monthly.parquet",
            "harmonized_country_monthly.parquet",
        ):
            path = alternate / filename
            frame = pd.read_parquet(path)
            mask = frame["year"] == 2024
            frame.loc[mask, "harmonized_srehi"] = frame.loc[mask, "harmonized_srehi"] * 3.0
            frame.loc[mask, "harmonized_frp_mw"] = frame.loc[mask, "harmonized_frp_mw"] * 3.0
            frame.to_parquet(path, index=False, compression="zstd")
        _run_module("src.models.calendar_analytics", alternate)
        alternate_climatology = pd.read_parquet(
            alternate / "calendar_climatology.parquet"
        )
        pd.testing.assert_frame_equal(
            climatology,
            alternate_climatology,
            check_exact=False,
            rtol=1e-10,
            atol=1e-10,
        )

        trends = pd.read_parquet(self.processed / "calendar_trends.parquet")
        for column in ("p_value", "q_value", "significant_fdr", "slope", "gev_fit_ok"):
            self.assertIn(column, trends.columns)
        valid_p = trends["p_value"].dropna()
        self.assertTrue(((valid_p >= 0.0) & (valid_p <= 1.0)).all())
        valid_q = trends["q_value"].dropna()
        self.assertTrue(((valid_q >= 0.0) & (valid_q <= 1.0)).all())
        self.assertTrue(trends["significant_fdr"].dtype == bool)
        gev_rows = trends[trends["gev_fit_ok"]]
        self.assertGreater(len(gev_rows), 0)
        self.assertTrue(np.isfinite(gev_rows["gev_rl10"]).all())
        self.assertTrue(np.isfinite(gev_rows["gev_rl20"]).all())
        self.assertTrue(np.isfinite(gev_rows["gev_rl50"]).all())

        early = pd.read_parquet(self.processed / "calendar_early_onset_90d.parquet")
        self.assertTrue(
            set(early["early_onset_flag"]).issubset(
                {
                    "EARLY_ONSET_CRITICAL",
                    "EARLY_ONSET_HIGH",
                    "NORMAL",
                    "INSUFFICIENT_HISTORY",
                    "NO_THRESHOLD",
                }
            )
        )
        self.assertIn("history_status", early.columns)
        self.assertIn("window_days", early.columns)
        self.assertTrue((early["history_status"] == "INSUFFICIENT_HISTORY").any())
        complete = early[(early["history_status"] == "OK") & (early["window_days"] == 90)]
        self.assertGreater(len(complete), 0)
        self.assertTrue(
            (
                early.loc[early["year"] == 2024, "early_onset_flag"]
                == "EARLY_ONSET_CRITICAL"
            ).any()
        )
        self.assertTrue((early["early_onset_flag"] == "NORMAL").any())
        self.assertGreaterEqual(early["doy"].min(), 1)
        self.assertLessEqual(early["doy"].max(), 366)


if __name__ == "__main__":
    unittest.main()
