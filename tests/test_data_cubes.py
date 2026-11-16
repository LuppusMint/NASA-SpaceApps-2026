"""Self-contained data-pipeline tests over synthetic fixtures."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date, datetime, time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.clean_and_flag import (  # noqa: E402
    clean_filter_flags_cte,
    identify_static_thermal_cells,
    sensor_slot_expr,
)
from src.data.build_spatiotemporal_cubes import _iter_year_sources  # noqa: E402

PYTHON = sys.executable

PA_SCHEMA = pa.schema(
    [
        pa.field("latitude", pa.float64()),
        pa.field("longitude", pa.float64()),
        pa.field("brightness_primary", pa.float64()),
        pa.field("brightness_secondary", pa.float64()),
        pa.field("scan", pa.float64()),
        pa.field("track", pa.float64()),
        pa.field("acq_date", pa.date32()),
        pa.field("acq_time", pa.string()),
        pa.field("acq_datetime_utc", pa.timestamp("us")),
        pa.field("satellite", pa.string()),
        pa.field("instrument", pa.string()),
        pa.field("confidence_raw", pa.string()),
        pa.field("confidence_pct", pa.float64()),
        pa.field("confidence_tier", pa.string()),
        pa.field("version", pa.string()),
        pa.field("frp", pa.float64()),
        pa.field("daynight", pa.string()),
        pa.field("type", pa.int16()),
        pa.field("source_feed", pa.string()),
        pa.field("source_priority", pa.int8()),
        pa.field("country_name", pa.string()),
        pa.field("iso_a3", pa.string()),
        pa.field("admin1_name", pa.string()),
        pa.field("x_m", pa.float64()),
        pa.field("y_m", pa.float64()),
        pa.field("cell_ix_10km", pa.int32()),
        pa.field("cell_iy_10km", pa.int32()),
        pa.field("cell_id_10km", pa.string()),
    ]
)


def make_row(**overrides):
    """Construye una fila con el esquema canónico y valores por defecto."""
    acq_date = overrides.get("acq_date", date(2020, 1, 15))
    acq_time = overrides.get("acq_time", "1030")
    cell_ix = int(overrides.get("cell_ix_10km", 100))
    cell_iy = int(overrides.get("cell_iy_10km", 200))
    row = {
        "latitude": -10.0,
        "longitude": -60.0,
        "brightness_primary": 310.0,
        "brightness_secondary": 290.0,
        "scan": 1.0,
        "track": 1.0,
        "acq_date": acq_date,
        "acq_time": acq_time,
        "satellite": "Terra",
        "instrument": "MODIS",
        "confidence_raw": "80",
        "confidence_pct": 80.0,
        "confidence_tier": "high",
        "version": "6.1",
        "frp": 10.0,
        "daynight": "D",
        "type": 0,
        "source_feed": "sp_country",
        "source_priority": 1,
        "country_name": "Brazil",
        "iso_a3": "BRA",
        "admin1_name": "Amazonas",
        "cell_ix_10km": cell_ix,
        "cell_iy_10km": cell_iy,
        "cell_id_10km": f"{cell_ix}_{cell_iy}",
        "x_m": cell_ix * 10_000.0 + 5_000.0,
        "y_m": cell_iy * 10_000.0 + 5_000.0,
    }
    row.update(overrides)
    if "acq_datetime_utc" not in overrides:
        row["acq_datetime_utc"] = datetime.combine(
            acq_date, time(int(acq_time[:2]), int(acq_time[2:4]))
        )
    return row


def build_fixture_rows():
    """Fixture multi-año/multi-sensor con casos límite del contrato."""
    rows = []

    # Celda estática temporal: 12 meses de 2020 (pasa el criterio temporal).
    for month in range(1, 13):
        rows.append(
            make_row(
                acq_date=date(2020, month, 15),
                acq_time="1010",
                confidence_pct=80.0,
                confidence_tier="high",
                frp=5.0,
                cell_ix_10km=900,
                cell_iy_10km=900,
                admin1_name="Amazonas",
            )
        )

    # Celda estática por tipo: 2/3 filas con type 1/2, solo 3 meses.
    rows.append(
        make_row(
            acq_date=date(2020, 1, 10),
            type=1,
            cell_ix_10km=901,
            cell_iy_10km=901,
            admin1_name="Roraima",
        )
    )
    rows.append(
        make_row(
            acq_date=date(2020, 2, 10),
            type=2,
            cell_ix_10km=901,
            cell_iy_10km=901,
            admin1_name="Roraima",
        )
    )
    rows.append(
        make_row(
            acq_date=date(2020, 3, 10),
            type=0,
            cell_ix_10km=901,
            cell_iy_10km=901,
            admin1_name="Roraima",
        )
    )

    # Celda limpia principal: MODIS Terra/Aqua y VIIRS SNPP/N20/N21.
    rows.extend(
        [
            make_row(
                acq_date=date(2020, 1, 15),
                acq_time="1030",
                confidence_pct=80.0,
                confidence_tier="high",
                frp=10.0,
            ),
            make_row(
                acq_date=date(2020, 1, 15),
                acq_time="1030",
                confidence_pct=85.0,
                confidence_tier="high",
                frp=11.0,
            ),
            make_row(
                acq_date=date(2020, 1, 15),
                acq_time="1330",
                satellite="Aqua",
                confidence_pct=40.0,
                confidence_tier="nominal",
                frp=20.0,
                daynight="N",
            ),
            make_row(
                acq_date=date(2020, 2, 10),
                acq_time="1010",
                confidence_pct=30.0,
                confidence_tier="nominal",
                frp=-5.0,
            ),
            make_row(
                acq_date=date(2020, 2, 11),
                acq_time="1010",
                confidence_pct=20.0,
                confidence_tier="low",
                frp=5.0,
            ),
            make_row(
                acq_date=date(2020, 2, 12),
                acq_time="1010",
                confidence_pct=75.0,
                confidence_tier="high",
                frp=7.0,
            ),
            make_row(
                acq_date=date(2020, 2, 13),
                acq_time="1010",
                confidence_pct=None,
                confidence_tier="nominal",
                confidence_raw="nominal",
                frp=4.0,
            ),
            make_row(
                acq_date=date(2020, 2, 14),
                acq_time="1010",
                confidence_pct=None,
                confidence_tier="high",
                confidence_raw="high",
                frp=4.0,
            ),
            make_row(
                acq_date=date(2020, 2, 15),
                acq_time="1010",
                brightness_primary=0.0,
                confidence_pct=80.0,
                confidence_tier="high",
                frp=4.0,
            ),
            make_row(acq_date=date(2020, 2, 16), acq_time="1010", type=1),
            make_row(acq_date=date(2020, 2, 17), acq_time="1010", type=2),
            make_row(acq_date=date(2020, 2, 18), acq_time="1010", type=3),
            make_row(
                acq_date=date(2020, 2, 19),
                acq_time="1010",
                type=None,
                confidence_pct=80.0,
                confidence_tier="high",
                frp=4.0,
            ),
            make_row(
                acq_date=date(2019, 9, 5),
                acq_time="0130",
                satellite="SNPP",
                instrument="VIIRS",
                confidence_pct=None,
                confidence_tier="low",
                confidence_raw="low",
                frp=6.0,
                daynight="N",
            ),
            make_row(
                acq_date=date(2019, 9, 5),
                acq_time="0130",
                satellite="SNPP",
                instrument="VIIRS",
                confidence_pct=None,
                confidence_tier="nominal",
                confidence_raw="nominal",
                frp=6.0,
                daynight="N",
            ),
            make_row(
                acq_date=date(2019, 9, 5),
                acq_time="0130",
                satellite="N20",
                instrument="VIIRS",
                confidence_pct=None,
                confidence_tier="high",
                confidence_raw="high",
                frp=6.0,
                daynight="N",
            ),
            make_row(
                acq_date=date(2021, 1, 20),
                acq_time="0140",
                satellite="N21",
                instrument="VIIRS",
                confidence_pct=None,
                confidence_tier="nominal",
                confidence_raw="nominal",
                frp=9.0,
                daynight="N",
            ),
            make_row(
                acq_date=date(2021, 1, 21),
                acq_time="0140",
                satellite="N20",
                instrument="VIIRS",
                confidence_pct=None,
                confidence_tier="high",
                confidence_raw="high",
                frp=9.0,
                daynight="N",
            ),
            make_row(
                acq_date=date(2000, 11, 15),
                acq_time="1030",
                confidence_pct=80.0,
                confidence_tier="high",
                frp=6.0,
            ),
            make_row(
                acq_date=date(2026, 10, 15),
                acq_time="1030",
                confidence_pct=80.0,
                confidence_tier="high",
                frp=6.0,
            ),
            make_row(
                acq_date=date(2026, 11, 1),
                acq_time="1030",
                confidence_pct=80.0,
                confidence_tier="high",
                frp=6.0,
            ),
        ]
    )

    # Otra jurisdicción (Perú/Loreto) para probar la jerarquía.
    rows.extend(
        [
            make_row(
                acq_date=date(2020, 3, 1),
                acq_time="1030",
                confidence_pct=50.0,
                confidence_tier="nominal",
                frp=12.0,
                latitude=-4.0,
                longitude=-75.0,
                cell_ix_10km=101,
                cell_iy_10km=200,
                country_name="Peru",
                iso_a3="PER",
                admin1_name="Loreto",
            ),
            make_row(
                acq_date=date(2020, 3, 2),
                acq_time="1330",
                satellite="Aqua",
                confidence_pct=90.0,
                confidence_tier="high",
                frp=15.0,
                daynight="N",
                latitude=-4.1,
                longitude=-75.1,
                cell_ix_10km=101,
                cell_iy_10km=200,
                country_name="Peru",
                iso_a3="PER",
                admin1_name="Loreto",
            ),
            make_row(
                acq_date=date(2021, 3, 1),
                acq_time="1030",
                confidence_pct=80.0,
                confidence_tier="high",
                frp=7.0,
                latitude=-4.2,
                longitude=-75.2,
                cell_ix_10km=101,
                cell_iy_10km=200,
                country_name="Peru",
                iso_a3="PER",
                admin1_name="Loreto",
            ),
        ]
    )

    # Chile/Antofagasta.
    rows.append(
        make_row(
            acq_date=date(2020, 3, 1),
            acq_time="1030",
            confidence_pct=40.0,
            confidence_tier="nominal",
            frp=3.0,
            latitude=-24.0,
            longitude=-70.0,
            cell_ix_10km=102,
            cell_iy_10km=200,
            country_name="Chile",
            iso_a3="CHL",
            admin1_name="Antofagasta",
        )
    )

    # SAMA / NRT nocturno: diferentes diferencias de brillo y sensores.
    rows.extend(
        [
            make_row(
                acq_date=date(2020, 4, 1),
                acq_time="0130",
                satellite="SNPP",
                instrument="VIIRS",
                confidence_pct=None,
                confidence_tier="low",
                confidence_raw="low",
                brightness_primary=300.0,
                brightness_secondary=295.0,
                frp=2.0,
                daynight="N",
                source_feed="nrt_api",
                latitude=-20.0,
                longitude=-50.0,
                cell_ix_10km=103,
                cell_iy_10km=200,
                admin1_name="Roraima",
            ),
            make_row(
                acq_date=date(2020, 4, 1),
                acq_time="0130",
                satellite="N20",
                instrument="VIIRS",
                confidence_pct=None,
                confidence_tier="high",
                confidence_raw="high",
                brightness_primary=310.0,
                brightness_secondary=290.0,
                frp=4.0,
                daynight="N",
                source_feed="nrt_api",
                latitude=-20.0,
                longitude=-50.0,
                cell_ix_10km=103,
                cell_iy_10km=200,
                admin1_name="Roraima",
            ),
            make_row(
                acq_date=date(2020, 4, 2),
                acq_time="0130",
                satellite="N21",
                instrument="VIIRS",
                confidence_pct=None,
                confidence_tier="low",
                confidence_raw="low",
                brightness_primary=320.0,
                brightness_secondary=300.0,
                frp=3.0,
                daynight="N",
                source_feed="nrt_api",
                latitude=-20.0,
                longitude=-50.0,
                cell_ix_10km=103,
                cell_iy_10km=200,
                admin1_name="Roraima",
            ),
            make_row(
                acq_date=date(2020, 4, 3),
                acq_time="0130",
                satellite="SNPP",
                instrument="VIIRS",
                confidence_pct=None,
                confidence_tier="low",
                confidence_raw="low",
                brightness_primary=300.0,
                brightness_secondary=295.0,
                frp=2.0,
                daynight="N",
                source_feed="sp_api",
                latitude=-20.0,
                longitude=-50.0,
                cell_ix_10km=103,
                cell_iy_10km=200,
                admin1_name="Roraima",
            ),
            make_row(
                acq_date=date(2020, 4, 4),
                acq_time="0130",
                satellite="SNPP",
                instrument="VIIRS",
                confidence_pct=None,
                confidence_tier="low",
                confidence_raw="low",
                brightness_primary=300.0,
                brightness_secondary=295.0,
                frp=2.0,
                daynight="D",
                source_feed="nrt_api",
                latitude=-20.0,
                longitude=-50.0,
                cell_ix_10km=103,
                cell_iy_10km=200,
                admin1_name="Roraima",
            ),
        ]
    )

    # Ecuador/Esmeraldas: al oeste del límite continental -82.2.
    rows.append(
        make_row(
            acq_date=date(2020, 5, 1),
            acq_time="1030",
            confidence_pct=60.0,
            confidence_tier="nominal",
            frp=4.0,
            latitude=0.5,
            longitude=-85.0,
            cell_ix_10km=300,
            cell_iy_10km=500,
            country_name="Ecuador",
            iso_a3="ECU",
            admin1_name="Esmeraldas",
        )
    )
    return rows


def expected_flags(row):
    """Implementación independiente del contrato de banderas."""
    instrument = row["instrument"]
    pct = row["confidence_pct"]
    tier = row["confidence_tier"]
    brightness = row["brightness_primary"]
    secondary = row["brightness_secondary"]
    valid = brightness is not None and brightness > 0.0
    veg = row["type"] is None or row["type"] == 0
    nominal = (
        (instrument == "MODIS" and (pct if pct is not None else -1.0) >= 30.0)
        or (instrument == "VIIRS" and tier in ("nominal", "high"))
        or (tier in ("nominal", "high") and pct is None)
    )
    high = (instrument == "MODIS" and (pct if pct is not None else -1.0) >= 75.0) or (
        instrument == "VIIRS" and tier == "high"
    )
    sama = -55.0 <= row["latitude"] <= 7.0 and -110.0 <= row["longitude"] <= 11.0
    diff = brightness - (secondary if secondary is not None else brightness)
    suspect = (
        sama
        and instrument == "VIIRS"
        and row["daynight"] == "N"
        and row["source_feed"] == "nrt_api"
        and tier == "low"
        and diff < 15.0
    )
    return {
        "valid": valid,
        "vegetation": veg,
        "nominal": nominal,
        "high": high,
        "sama": sama,
        "suspect": suspect,
        "nrt": row["source_feed"] == "nrt_api",
    }


def expected_metrics(rows, static_cells):
    """Resume las métricas QC esperadas a partir del fixture."""
    totals = {
        "raw_all_det_count": 0,
        "valid_veg_nominal_plus": 0,
        "clean_det_count": 0,
        "static_clean_candidate_rows": 0,
        "excluded_type_123": 0,
        "negative_frp_raw": 0,
        "zero_brightness": 0,
        "sama_rows": 0,
        "sama_suspect_night": 0,
        "static_excluded_rows": 0,
        "nrt_rows": 0,
    }
    for row in rows:
        flags = expected_flags(row)
        totals["raw_all_det_count"] += 1
        if flags["valid"] and flags["vegetation"] and flags["nominal"]:
            totals["valid_veg_nominal_plus"] += 1
            if row["cell_id_10km"] in static_cells:
                totals["static_clean_candidate_rows"] += 1
            else:
                totals["clean_det_count"] += 1
        if row["type"] in (1, 2, 3):
            totals["excluded_type_123"] += 1
        if row["frp"] is not None and row["frp"] < 0.0:
            totals["negative_frp_raw"] += 1
        if row["brightness_primary"] is None or row["brightness_primary"] <= 0.0:
            totals["zero_brightness"] += 1
        if flags["sama"]:
            totals["sama_rows"] += 1
        if flags["suspect"]:
            totals["sama_suspect_night"] += 1
        if row["cell_id_10km"] in static_cells:
            totals["static_excluded_rows"] += 1
        if flags["nrt"]:
            totals["nrt_rows"] += 1
    return totals


class CubePipelineTests(unittest.TestCase):
    """Ejecuta el pipeline una vez y valida todos los artefactos."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="c1_cubes_")
        cls.tmp = Path(cls._tmp.name)
        cls.parquet_dir = cls.tmp / "parquet" / "south_america_hotspots"
        cls.out_dir = cls.tmp / "processed"
        cls.rows = build_fixture_rows()

        by_year: dict[int, list[dict]] = {}
        for row in cls.rows:
            by_year.setdefault(row["acq_date"].year, []).append(row)
        for year, year_rows in by_year.items():
            target = cls.parquet_dir / f"year={year}" / "part-0.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pylist(year_rows, schema=PA_SCHEMA)
            pq.write_table(table, target, compression="zstd")
        cls.glob = str(cls.parquet_dir / "year=*" / "part-*.parquet")

        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "src.data.build_spatiotemporal_cubes",
                "--project-root",
                str(cls.tmp),
                "--parquet-glob",
                cls.glob,
                "--out-dir",
                str(cls.out_dir),
                "--min-active-months",
                "6",
                "--min-active-years",
                "1",
                "--min-months-per-year",
                "2.0",
                "--min-type12-count",
                "2",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                "El pipeline falló:\nSTDOUT:\n"
                + result.stdout
                + "\nSTDERR:\n"
                + result.stderr
            )
        cls.stdout = result.stdout
        cls.con = duckdb.connect()
        cls.cell10 = cls.out_dir / "cube_cell10km_monthly.parquet"
        cls.cell50 = cls.out_dir / "cube_cell50km_monthly.parquet"
        cls.admin1 = cls.out_dir / "cube_admin1_monthly.parquet"
        cls.country = cls.out_dir / "cube_country_monthly.parquet"
        cls.daily = cls.out_dir / "cube_admin1_daily_season.parquet"
        cls.static_cells = cls.out_dir / "static_thermal_cells.parquet"
        cls.qc_path = cls.out_dir / "qc_cleaning_summary.json"
        cls.catalog_path = cls.out_dir / "geography_catalog.json"
        cls.qc = json.loads(cls.qc_path.read_text(encoding="utf-8"))
        cls.catalog = json.loads(cls.catalog_path.read_text(encoding="utf-8"))
        cls.static_ids = {
            row[0]
            for row in cls.con.execute(
                f"SELECT cell_id_10km FROM read_parquet('{cls.static_cells.as_posix()}')"
            ).fetchall()
        }

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls._tmp.cleanup()

    def query(self, sql: str):
        return self.con.execute(sql).fetchall()

    def test_all_outputs_exist(self):
        for path in (
            self.cell10,
            self.cell50,
            self.admin1,
            self.country,
            self.daily,
            self.static_cells,
            self.qc_path,
            self.catalog_path,
        ):
            self.assertTrue(path.is_file(), f"Falta {path}")
            self.assertGreater(path.stat().st_size, 0, f"Archivo vacío: {path}")

    def test_single_file_fallback_by_acq_date_year(self):
        """Sin carpetas ``year=YYYY`` el loteo usa los años de ``acq_date``."""
        single_dir = self.tmp / "single_file_source"
        single_file = single_dir / "all_years.parquet"
        single_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(self.rows, schema=PA_SCHEMA),
            single_file,
            compression="zstd",
        )

        con = duckdb.connect()
        try:
            partitioned = _iter_year_sources(con, self.glob)
            fallback = _iter_year_sources(con, str(single_file))
        finally:
            con.close()
        expected_years = [2000, 2019, 2020, 2021, 2026]
        self.assertEqual([year for year, _ in partitioned], expected_years)
        self.assertEqual([year for year, _ in fallback], expected_years)
        self.assertTrue(
            all(
                isinstance(source, str)
                and "EXTRACT(year FROM acq_date)" in source
                for _, source in fallback
            )
        )

        fallback_out = self.tmp / "processed_single_file"
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "src.data.build_spatiotemporal_cubes",
                "--project-root",
                str(self.tmp),
                "--parquet-glob",
                str(single_file),
                "--out-dir",
                str(fallback_out),
                "--min-active-months",
                "6",
                "--min-active-years",
                "1",
                "--min-months-per-year",
                "2.0",
                "--min-type12-count",
                "2",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                "El pipeline fallback falló:\nSTDOUT:\n"
                + result.stdout
                + "\nSTDERR:\n"
                + result.stderr
            )

        for name in (
            "cube_cell10km_monthly.parquet",
            "cube_cell50km_monthly.parquet",
            "cube_admin1_monthly.parquet",
            "cube_country_monthly.parquet",
            "cube_admin1_daily_season.parquet",
            "static_thermal_cells.parquet",
        ):
            with self.subTest(name=name):
                partitioned_table = pq.read_table(self.out_dir / name)
                fallback_table = pq.read_table(fallback_out / name)
                self.assertEqual(partitioned_table.schema, fallback_table.schema)
                columns = partitioned_table.column_names

                def sort_key(row):
                    return tuple(str(row[column]) for column in columns)

                self.assertEqual(
                    sorted(partitioned_table.to_pylist(), key=sort_key),
                    sorted(fallback_table.to_pylist(), key=sort_key),
                )

    def test_static_cells_detected(self):
        self.assertEqual(self.static_ids, {"900_900", "901_901"})
        self.assertEqual(self.qc["static_cell_count"], 2)
        self.assertEqual(self.qc["static_det_count"], 15)

        temporal = self.query(
            f"""
            SELECT raw_all_det_count, clean_det_count, static_excluded_det_count
            FROM read_parquet('{self.cell10.as_posix()}')
            WHERE cell_id_10km = '900_900'
            """
        )
        self.assertEqual(len(temporal), 12)
        self.assertEqual(sum(row[0] for row in temporal), 12)
        self.assertEqual(sum(row[1] for row in temporal), 0)
        self.assertEqual(sum(row[2] for row in temporal), 12)

        by_type = self.query(
            f"""
            SELECT raw_all_det_count, clean_det_count, static_excluded_det_count
            FROM read_parquet('{self.cell10.as_posix()}')
            WHERE cell_id_10km = '901_901'
            """
        )
        self.assertEqual(sum(row[0] for row in by_type), 3)
        self.assertEqual(sum(row[1] for row in by_type), 0)
        self.assertEqual(sum(row[2] for row in by_type), 1)

    def test_isolated_type12_row_is_not_static_by_default(self):
        """Una sola fila type=1 no alcanza el umbral por defecto de 5."""
        source_dir = self.tmp / "static_threshold"
        source = source_dir / "part-0.parquet"
        source.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    make_row(
                        acq_date=date(2020, 1, 10),
                        type=1,
                        cell_ix_10km=950,
                        cell_iy_10km=950,
                    )
                ],
                schema=PA_SCHEMA,
            ),
            source,
            compression="zstd",
        )
        con = duckdb.connect()
        try:
            default_out = source_dir / "static_default.parquet"
            default_summary = identify_static_thermal_cells(
                con,
                str(source),
                default_out,
                min_active_months=2,
                min_active_years=1,
                min_months_per_year=0.0,
            )
            self.assertEqual(default_summary["static_cell_count"], 0)
            self.assertEqual(default_summary["n_type_12_dominant"], 0)
            self.assertEqual(
                con.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{default_out.as_posix()}')"
                ).fetchone()[0],
                0,
            )

            lowered_out = source_dir / "static_lowered.parquet"
            lowered_summary = identify_static_thermal_cells(
                con,
                str(source),
                lowered_out,
                min_active_months=2,
                min_active_years=1,
                min_months_per_year=0.0,
                min_type12_count=1,
            )
            self.assertEqual(lowered_summary["static_cell_count"], 1)
            self.assertEqual(lowered_summary["n_type_12_dominant"], 1)
        finally:
            con.close()

    def test_qc_counts_match_independent_oracle(self):
        expected = expected_metrics(self.rows, self.static_ids)
        for key, value in expected.items():
            self.assertEqual(self.qc[key], value, f"Métrica QC {key}")
        self.assertEqual(self.qc["by_sensor_slot"][0]["raw_all_det_count"] > 0, True)

        slot_totals: dict[str, int] = {}
        for row in self.rows:
            slot = self.slot_for(row["satellite"], row["instrument"])
            slot_totals[slot] = slot_totals.get(slot, 0) + 1
        observed = {
            entry["sensor_slot"]: entry["raw_all_det_count"]
            for entry in self.qc["by_sensor_slot"]
        }
        self.assertEqual(observed, slot_totals)

    @staticmethod
    def slot_for(satellite: str, instrument: str) -> str:
        mapping = {
            ("Terra", "MODIS"): "MODIS_Terra",
            ("Aqua", "MODIS"): "MODIS_Aqua",
            ("SNPP", "VIIRS"): "VIIRS_SNPP",
            ("N20", "VIIRS"): "VIIRS_N20",
            ("N21", "VIIRS"): "VIIRS_N21",
        }
        return mapping.get((satellite, instrument), f"{instrument}_{satellite}")

    def test_confidence_homologation_and_exclusions(self):
        rows = self.query(
            f"""
            SELECT clean_det_count, high_conf_count, frp_sum_mw, overpass_event_count
            FROM read_parquet('{self.cell10.as_posix()}')
            WHERE cell_id_10km = '100_200' AND "year" = 2020 AND "month" = 2
              AND sensor_slot = 'MODIS_Terra'
            """
        )
        self.assertEqual(len(rows), 1)
        clean, high, frp_sum, events = rows[0]
        self.assertEqual(clean, 5)
        self.assertEqual(high, 2)
        self.assertAlmostEqual(frp_sum, 19.0, places=6)
        self.assertEqual(events, 5)

        january = self.query(
            f"""
            SELECT clean_det_count, overpass_event_count, active_days
            FROM read_parquet('{self.cell10.as_posix()}')
            WHERE cell_id_10km = '100_200' AND "year" = 2020 AND "month" = 1
              AND sensor_slot = 'MODIS_Terra'
            """
        )
        self.assertEqual(january[0][0], 2)
        self.assertEqual(january[0][1], 1)
        self.assertEqual(january[0][2], 1)

        # La fila con brightness=0 y las de type 1/2/3 nunca entran al cubo limpio.
        brazil_2020 = self.query(
            f"""
            SELECT SUM(raw_all_det_count), SUM(clean_det_count)
            FROM read_parquet('{self.cell10.as_posix()}')
            WHERE cell_id_10km IN ('100_200', '900_900', '901_901')
            """
        )[0]
        self.assertGreater(brazil_2020[0], brazil_2020[1])

    def test_frp_clipping(self):
        row = self.query(
            f"""
            SELECT frp_sum_mw, frp_max_mw, clean_det_count
            FROM read_parquet('{self.cell10.as_posix()}')
            WHERE cell_id_10km = '100_200' AND "year" = 2020 AND "month" = 2
              AND sensor_slot = 'MODIS_Terra'
            """
        )[0]
        self.assertEqual(row[2], 5)
        self.assertAlmostEqual(row[0], 19.0, places=6)
        self.assertGreaterEqual(row[1], 0.0)

    def test_partial_month_flag(self):
        flags = self.query(
            f"""
            SELECT "year", "month", is_partial_month
            FROM read_parquet('{self.cell10.as_posix()}')
            WHERE ("year" = 2000 AND "month" = 11)
               OR "year" = 2026
            ORDER BY "year", "month"
            """
        )
        self.assertEqual(
            flags,
            [(2000, 11, False), (2026, 10, False), (2026, 11, True)],
        )

    def test_nrt_majority_flag(self):
        nrt = self.query(
            f"""
            SELECT nrt_det_count, sp_det_count, is_nrt_majority
            FROM read_parquet('{self.cell10.as_posix()}')
            WHERE cell_id_10km = '103_200' AND "year" = 2020 AND "month" = 4
              AND sensor_slot = 'VIIRS_N20'
            """
        )[0]
        self.assertEqual(nrt, (1, 0, True))

        sp = self.query(
            f"""
            SELECT nrt_det_count, sp_det_count, is_nrt_majority
            FROM read_parquet('{self.cell10.as_posix()}')
            WHERE cell_id_10km = '100_200' AND "year" = 2020 AND "month" = 1
              AND sensor_slot = 'MODIS_Terra'
            """
        )[0]
        self.assertEqual(sp, (0, 2, False))

    def test_sama_suspect_flag(self):
        self.assertEqual(self.qc["sama_suspect_night"], 1)
        self.assertGreaterEqual(self.qc["sama_rows"], self.qc["sama_suspect_night"])
        self.assertGreater(self.qc["nrt_rows"], 0)

    def test_continental_sa_flag(self):
        rows = self.query(
            f"""
            SELECT DISTINCT is_continental_sa
            FROM read_parquet('{self.cell10.as_posix()}')
            WHERE country_name = 'Ecuador'
            """
        )
        self.assertEqual(rows, [(False,)])

    def test_hierarchical_mass_conservation(self):
        group = '"year", "month", sensor_slot'
        metrics = (
            "clean_det_count",
            "static_excluded_det_count",
            "frp_sum_mw",
            "overpass_event_count",
            "high_conf_count",
            "night_det_count",
            "day_det_count",
            "raw_all_det_count",
            "active_cell_days",
        )
        select_metrics = ", ".join(f"SUM({m})" for m in metrics)
        admin = self.query(
            f"""
            SELECT {group}, {select_metrics}
            FROM read_parquet('{self.admin1.as_posix()}')
            GROUP BY {group}
            ORDER BY {group}
            """
        )
        country = self.query(
            f"""
            SELECT {group}, {select_metrics}
            FROM read_parquet('{self.country.as_posix()}')
            GROUP BY {group}
            ORDER BY {group}
            """
        )
        self.assertEqual(len(admin), len(country))
        for left, right in zip(admin, country):
            self.assertEqual(left[:3], right[:3])
            for index, name in enumerate(metrics, start=3):
                if name == "frp_sum_mw":
                    self.assertAlmostEqual(left[index], right[index], places=6, msg=name)
                else:
                    self.assertEqual(left[index], right[index], name)

        cell_metrics = (
            "clean_det_count",
            "static_excluded_det_count",
            "frp_sum_mw",
            "overpass_event_count",
            "high_conf_count",
            "night_det_count",
            "day_det_count",
            "raw_all_det_count",
        )
        cell_totals = self.query(
            "SELECT "
            + ", ".join(f"SUM({m})" for m in cell_metrics)
            + f" FROM read_parquet('{self.cell10.as_posix()}')"
        )[0]
        country_totals = self.query(
            "SELECT "
            + ", ".join(f"SUM({m})" for m in cell_metrics)
            + f" FROM read_parquet('{self.country.as_posix()}')"
        )[0]
        for name, cell_value, country_value in zip(cell_metrics, cell_totals, country_totals):
            if name == "frp_sum_mw":
                self.assertAlmostEqual(cell_value, country_value, places=6, msg=name)
            else:
                self.assertEqual(cell_value, country_value, name)

    def test_country_unique_cells_are_direct(self):
        country = {
            row[:6]: row[6]
            for row in self.query(
                f"""
                SELECT "year", "month", year_month, country_name, iso_a3, sensor_slot,
                       unique_active_cells_10km
                FROM read_parquet('{self.country.as_posix()}')
                """
            )
        }
        direct = {
            row[:6]: row[6]
            for row in self.query(
                f"""
                SELECT "year", "month", year_month, country_name, iso_a3, sensor_slot,
                       COUNT(DISTINCT CASE WHEN clean_det_count > 0
                           THEN cell_id_10km END)
                FROM read_parquet('{self.cell10.as_posix()}')
                GROUP BY 1, 2, 3, 4, 5, 6
                """
            )
        }
        for key, value in country.items():
            self.assertEqual(value, direct.get(key, 0), key)

    def test_cell50km_reaggregation(self):
        row = self.query(
            f"""
            SELECT SUM(clean_det_count), SUM(active_cells_10km),
                   SUM(overpass_event_count), SUM(frp_sum_mw)
            FROM read_parquet('{self.cell50.as_posix()}')
            """
        )[0]
        expected = self.query(
            f"""
            SELECT SUM(clean_det_count), COUNT(*) FILTER (WHERE clean_det_count > 0),
                   SUM(overpass_event_count), SUM(frp_sum_mw)
            FROM read_parquet('{self.cell10.as_posix()}')
            """
        )[0]
        self.assertEqual(row[0], expected[0])
        self.assertEqual(row[2], expected[2])
        self.assertAlmostEqual(row[3], expected[3], places=6)
        # Cada fila 10 km limpia es una celda-mes-sensor activa; al reagregar a
        # 50 km se conserva exactamente ese total.
        self.assertEqual(row[1], expected[1])

    def test_daily_season_cube(self):
        rows = self.query(
            f"""
            SELECT "year", acq_date, doy, clean_det_count, frp_sum_mw,
                   active_cells_10km, overpass_event_count
            FROM read_parquet('{self.daily.as_posix()}')
            WHERE country_name = 'Peru'
            ORDER BY acq_date
            """
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual([row[3] for row in rows], [1, 1, 1])
        self.assertEqual([round(row[4], 6) for row in rows], [12.0, 15.0, 7.0])
        self.assertTrue(all(row[5] == 1 for row in rows))
        self.assertTrue(all(1 <= row[2] <= 366 for row in rows))

    def test_geography_catalog(self):
        catalog = self.catalog
        self.assertEqual(catalog["bbox_order"], ["min_lat", "max_lat", "min_lon", "max_lon"])
        countries = {item["country_name"]: item for item in catalog["countries"]}
        self.assertEqual(set(countries), {"Brazil", "Peru", "Chile", "Ecuador"})
        self.assertEqual(catalog["totals"]["countries"], 4)
        brazil = countries["Brazil"]
        self.assertEqual(brazil["admin1_count"], 2)
        self.assertEqual(brazil["historical_active_cells_10km"], 2)
        self.assertTrue(brazil["intersects_sama"])
        admin_names = {item["admin1_name"] for item in brazil["admin1"]}
        self.assertEqual(admin_names, {"Amazonas", "Roraima"})
        for country in countries.values():
            self.assertEqual(len(country["bbox"]), 4)
            self.assertLessEqual(country["bbox"][0], country["bbox"][1])
            self.assertLessEqual(country["bbox"][2], country["bbox"][3])

    def test_sensor_slot_expression(self):
        con = duckdb.connect()
        try:
            con.execute("CREATE TABLE t(satellite VARCHAR, instrument VARCHAR)")
            con.executemany(
                "INSERT INTO t VALUES (?, ?)",
                [
                    ("Terra", "MODIS"),
                    ("Aqua", "MODIS"),
                    ("SNPP", "VIIRS"),
                    ("N20", "VIIRS"),
                    ("N21", "VIIRS"),
                    ("Other", "VIIRS"),
                ],
            )
            rows = con.execute(
                f"SELECT satellite, instrument, {sensor_slot_expr()} AS slot FROM t"
            ).fetchall()
        finally:
            con.close()
        slots = {(satellite, instrument): slot for satellite, instrument, slot in rows}
        self.assertEqual(slots[("Terra", "MODIS")], "MODIS_Terra")
        self.assertEqual(slots[("Aqua", "MODIS")], "MODIS_Aqua")
        self.assertEqual(slots[("SNPP", "VIIRS")], "VIIRS_SNPP")
        self.assertEqual(slots[("N20", "VIIRS")], "VIIRS_N20")
        self.assertEqual(slots[("N21", "VIIRS")], "VIIRS_N21")
        self.assertEqual(slots[("Other", "VIIRS")], "VIIRS_Other")

    def test_clean_filter_flags_cte(self):
        con = duckdb.connect()
        try:
            con.execute(
                "CREATE TEMP VIEW raw AS "
                "SELECT latitude, longitude, brightness_primary, brightness_secondary, "
                "scan, track, acq_date, acq_time, acq_datetime_utc, satellite, instrument, "
                "confidence_raw, confidence_pct, confidence_tier, version, frp, daynight, "
                '"type", source_feed, source_priority, country_name, iso_a3, admin1_name, '
                'x_m, y_m, cell_ix_10km, cell_iy_10km, cell_id_10km '
                f"FROM read_parquet('{self.glob}')"
            )
            rows = con.execute(
                clean_filter_flags_cte("raw")
                + " WHERE cell_id_10km = '103_200' AND source_feed = 'nrt_api' "
                "ORDER BY acq_date"
            ).fetchall()
            columns = [item[0] for item in con.execute(clean_filter_flags_cte("raw") + " LIMIT 0").description]
            index = {name: position for position, name in enumerate(columns)}
            suspect = [row for row in rows if row[index["is_sama_suspect_night"]]]
            self.assertEqual(len(suspect), 1)
            self.assertEqual(suspect[0][index["sensor_slot"]], "VIIRS_SNPP")
            self.assertTrue(all(row[index["is_in_sama"]] for row in rows))
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
