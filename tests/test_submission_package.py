#!/usr/bin/env python3
"""Submission export, 7-slide deck, and public-document tests.

The export test uses the harmonized country cube when it exists under
``data/processed``.  Otherwise it builds a self-contained 4,014-row fixture in
``tempfile.TemporaryDirectory()`` and exercises the same script and validator
contract.  No test writes a root-level ``submission.csv``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "export_submission.py"
SPEC_PATH = REPO_ROOT / "submission_spec.json"
SLIDES_PATH = REPO_ROOT / "docs" / "presentation.html"
SLIDES_PDF_PATH = REPO_ROOT / "docs" / "Fire_Relay_South_America_Presentation.pdf"
PACKAGE_PATH = REPO_ROOT / "submission" / "PROJECT_SUBMISSION.md"
README_PATH = REPO_ROOT / "README.md"

EXPECTED_COLUMNS = [
    "record_id",
    "iso_a3",
    "country_name",
    "year_month",
    "year",
    "month",
    "harmonized_srehi",
    "harmonized_frp_mw",
    "raw_srehi",
    "raw_frp_mw",
    "clean_det_count",
    "raw_all_det_count",
    "data_maturity",
    "constellation_regime",
]

VALUE_COLUMNS = [
    "harmonized_srehi",
    "harmonized_frp_mw",
    "raw_srehi",
    "raw_frp_mw",
    "clean_det_count",
    "raw_all_det_count",
]

COUNTRIES = [
    ("ARG", "Argentina"),
    ("BOL", "Bolivia"),
    ("BRA", "Brazil"),
    ("CHL", "Chile"),
    ("COL", "Colombia"),
    ("ECU", "Ecuador"),
    ("FLK", "Falkland Islands"),
    ("GUF", "French Guiana"),
    ("GUY", "Guyana"),
    ("PER", "Peru"),
    ("PRY", "Paraguay"),
    ("SUR", "Suriname"),
    ("URY", "Uruguay"),
    ("VEN", "Venezuela"),
]

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_local(spec: dict, csv_path: Path, spec_dir: Path) -> list[str]:
    """Validate the submission contract without external imports."""
    errors: list[str] = []
    if not csv_path.is_file():
        return [f"missing submission: {csv_path}"]

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        rows = list(reader)

    required = list(spec.get("required_columns", []))
    for column in required:
        if column not in header:
            errors.append(f"missing required column: {column}")
    if spec.get("exact_columns") and set(header) != set(required):
        errors.append(f"unexpected columns: {sorted(set(header) - set(required))}")

    expected_rows = spec.get("row_count")
    if expected_rows is not None and len(rows) != int(expected_rows):
        errors.append(f"rows: {len(rows)} != {expected_rows}")

    id_column = spec.get("id_column")
    if id_column and id_column in header:
        ids = [row[id_column] for row in rows]
        if len(ids) != len(set(ids)):
            errors.append("duplicate IDs")
        id_file = spec.get("id_file")
        if id_file:
            expected_path = spec_dir / id_file
            if expected_path.suffix.lower() == ".csv":
                with expected_path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    first = (reader.fieldnames or [None])[0]
                    expected = {row[first].strip() for row in reader}
            else:
                expected = {
                    line.strip()
                    for line in expected_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }
            if set(ids) != expected:
                errors.append("IDs do not match id_file")
    elif id_column:
        errors.append(f"missing ID column: {id_column}")

    value_columns = list(spec.get("value_columns") or [])
    value_column = spec.get("value_column")
    if value_column and value_column not in value_columns:
        value_columns.insert(0, value_column)

    if value_columns:
        missing = [column for column in value_columns if column not in header]
        if missing:
            errors.append(f"missing value column(s): {', '.join(missing)}")
            return errors

        low = spec.get("min")
        high = spec.get("max")
        allow_nan = bool(spec.get("allow_nan", False))
        for index, row in enumerate(rows, 2):
            for column in value_columns:
                raw = row[column]
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    errors.append(f"row {index} ({column}): non-numeric {raw!r}")
                    continue
                if math.isnan(value):
                    if not allow_nan:
                        errors.append(f"row {index} ({column}): NaN not allowed")
                    continue
                if low is not None and value < float(low):
                    errors.append(f"row {index} ({column}): {value} < {low}")
                if high is not None and value > float(high):
                    errors.append(f"row {index} ({column}): {value} > {high}")
    return errors


def _month_sequence(count: int) -> list[str]:
    year, month = 2000, 11
    result: list[str] = []
    for _ in range(count):
        result.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1
    return result


def _regime(year: int) -> str:
    if year <= 2002:
        return "MODIS_TERRA_ONLY"
    if year <= 2011:
        return "MODIS_TERRA_AQUA"
    if year <= 2020:
        return "MODIS_VIIRS_OVERLAP"
    return "VIIRS_BACKBONE"


def _write_fixture(processed_dir: Path) -> Path:
    """Write a complete 4,014-row country cube matching the harmonized schema."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    destination = processed_dir / "harmonized_country_monthly.parquet"

    # 10 countries x 287 months + 4 countries x 286 months = 4,014 rows.
    rows: list[dict] = []
    for index, (iso_a3, country_name) in enumerate(COUNTRIES):
        months = _month_sequence(287 if index < 10 else 286)
        for month_index, year_month in enumerate(months):
            year, month = (int(part) for part in year_month.split("-"))
            seed = index * 313 + month_index
            harmonized_srehi = float((seed % 97) + 1) * 1.5
            harmonized_frp_mw = float((seed % 53) + 1) * 10.0
            overpass_event_count = int((seed % 31) + 1)
            raw_frp_mw = float((seed % 41) + 1) * 5.0
            clean_det_count = int((seed % 17) + 1)
            raw_all_det_count = clean_det_count + int((seed % 7) + 1)
            rows.append(
                {
                    "year": year,
                    "month": month,
                    "year_month": year_month,
                    "country_name": country_name,
                    "iso_a3": iso_a3,
                    "raw_all_det_count": raw_all_det_count,
                    "clean_det_count": clean_det_count,
                    "static_excluded_det_count": 0,
                    "high_conf_count": clean_det_count // 2,
                    "night_det_count": clean_det_count // 3,
                    "overpass_event_count": overpass_event_count,
                    "raw_frp_mw": raw_frp_mw,
                    "active_cells_10km": clean_det_count,
                    "nrt_det_count": 0,
                    "sp_det_count": clean_det_count,
                    "is_partial_month": year_month in {"2000-11", "2026-11"},
                    "harmonized_srehi": harmonized_srehi,
                    "harmonized_srehi_lo90": harmonized_srehi * 0.8,
                    "harmonized_srehi_hi90": harmonized_srehi * 1.2,
                    "harmonized_frp_mw": harmonized_frp_mw,
                    "harmonized_frp_lo90": harmonized_frp_mw * 0.8,
                    "harmonized_frp_hi90": harmonized_frp_mw * 1.2,
                    "m1_srehi": harmonized_srehi * 0.95,
                    "m2_srehi": harmonized_srehi * 0.98,
                    "m1_frp_mw": harmonized_frp_mw * 0.95,
                    "m2_frp_mw": harmonized_frp_mw * 0.98,
                    "constellation_regime": _regime(year),
                    "data_maturity": (
                        "PARTIAL_MONTH"
                        if year_month in {"2000-11", "2026-11"}
                        else "STANDARD_SCIENCE"
                    ),
                    "is_in_sama": True,
                    "intersects_sama": True,
                    "sama_suspect_det_count": 0,
                }
            )

    schema = pa.schema(
        [
            ("year", pa.int32()),
            ("month", pa.int32()),
            ("year_month", pa.string()),
            ("country_name", pa.string()),
            ("iso_a3", pa.string()),
            ("raw_all_det_count", pa.int64()),
            ("clean_det_count", pa.int64()),
            ("static_excluded_det_count", pa.int64()),
            ("high_conf_count", pa.int64()),
            ("night_det_count", pa.int64()),
            ("overpass_event_count", pa.int64()),
            ("raw_frp_mw", pa.float64()),
            ("active_cells_10km", pa.int64()),
            ("nrt_det_count", pa.int64()),
            ("sp_det_count", pa.int64()),
            ("is_partial_month", pa.bool_()),
            ("harmonized_srehi", pa.float64()),
            ("harmonized_srehi_lo90", pa.float64()),
            ("harmonized_srehi_hi90", pa.float64()),
            ("harmonized_frp_mw", pa.float64()),
            ("harmonized_frp_lo90", pa.float64()),
            ("harmonized_frp_hi90", pa.float64()),
            ("m1_srehi", pa.float64()),
            ("m2_srehi", pa.float64()),
            ("m1_frp_mw", pa.float64()),
            ("m2_frp_mw", pa.float64()),
            ("constellation_regime", pa.string()),
            ("data_maturity", pa.string()),
            ("is_in_sama", pa.bool_()),
            ("intersects_sama", pa.bool_()),
            ("sama_suspect_det_count", pa.int64()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        destination,
        compression="zstd",
    )
    return destination


class SubmissionContractTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    def _run_export(self, processed_dir: Path, output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(EXPORT_SCRIPT),
                "--processed-dir",
                str(processed_dir),
                "--output",
                str(output),
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_spec_declares_the_exact_public_contract(self) -> None:
        self.assertTrue(SPEC_PATH.is_file())
        self.assertEqual(self.spec["required_columns"], EXPECTED_COLUMNS)
        self.assertTrue(self.spec["exact_columns"])
        self.assertEqual(int(self.spec["row_count"]), 4014)
        self.assertEqual(self.spec["id_column"], "record_id")
        self.assertEqual(self.spec["value_columns"], VALUE_COLUMNS)
        self.assertEqual(float(self.spec["min"]), 0.0)
        self.assertFalse(self.spec["allow_nan"])

    def test_export_is_deterministic_and_passes_the_validator(self) -> None:
        real_cube = REPO_ROOT / "data" / "processed" / "harmonized_country_monthly.parquet"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            if real_cube.is_file():
                processed_dir = REPO_ROOT / "data" / "processed"
                source = real_cube
                source_hash = _sha256(source)
            else:
                processed_dir = tmp_dir / "processed"
                source = _write_fixture(processed_dir)
                source_hash = _sha256(source)

            output = tmp_dir / "submission.csv"
            result = self._run_export(processed_dir, output)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output.is_file())
            self.assertEqual(_sha256(source), source_hash, "export must not mutate the source Parquet")

            errors = _validate_local(self.spec, output, REPO_ROOT)
            self.assertEqual(errors, [], msg="\n".join(errors))

            with output.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                header = reader.fieldnames

            self.assertEqual(header, EXPECTED_COLUMNS)
            self.assertEqual(len(rows), 4014)
            ids = [row["record_id"] for row in rows]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(ids, sorted(ids))
            self.assertEqual(
                [(row["iso_a3"], row["year_month"]) for row in rows],
                sorted((row["iso_a3"], row["year_month"]) for row in rows),
            )

            for row in rows:
                self.assertEqual(row["record_id"], f"{row['iso_a3']}_{row['year_month']}")
                for column in VALUE_COLUMNS:
                    value = float(row[column])
                    self.assertFalse(math.isnan(value), msg=f"NaN in {column}")
                    self.assertGreaterEqual(value, 0.0, msg=f"negative in {column}")

            # The public raw_srehi is exactly the source overpass_event_count.
            import duckdb

            connection = duckdb.connect(database=":memory:")
            try:
                source_rows = connection.execute(
                    f"SELECT iso_a3, year_month, overpass_event_count "
                    f"FROM read_parquet('{str(source).replace(chr(39), chr(39) * 2)}')"
                ).fetchall()
            finally:
                connection.close()
            expected_raw = {
                (str(iso), str(year_month)): int(overpass or 0)
                for iso, year_month, overpass in source_rows
            }
            self.assertEqual(len(expected_raw), 4014)
            for row in rows:
                key = (row["iso_a3"], row["year_month"])
                self.assertEqual(int(row["raw_srehi"]), expected_raw[key])

            second = tmp_dir / "submission_second.csv"
            result_second = self._run_export(processed_dir, second)
            self.assertEqual(result_second.returncode, 0, msg=result_second.stderr)
            self.assertEqual(output.read_bytes(), second.read_bytes(), "CSV bytes must be reproducible")

    def test_export_fails_cleanly_without_the_country_cube(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_export(Path(tmp) / "missing", Path(tmp) / "out.csv")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing harmonized country cube", result.stderr)

    def test_slides_deck_is_seven_pages_offline_and_logo_free(self) -> None:
        self.assertTrue(SLIDES_PATH.is_file())
        self.assertTrue(SLIDES_PDF_PATH.is_file())
        html = SLIDES_PATH.read_text(encoding="utf-8")

        self.assertEqual(html.count('<section class="slide"'), 7)
        self.assertEqual(len(re.findall(r"<section\b[^>]*\bclass=\"[^\"]*\bslide\b", html)), 7)
        self.assertEqual(len(re.findall(r'id="slide-[1-7]"', html)), 7)
        self.assertIn("@page", html)
        self.assertRegex(html, r"@page\s*\{[^}]*size:\s*13\.333in\s+7\.5in")
        self.assertRegex(html, r"@page\s*\{[^}]*margin:\s*0")
        self.assertGreaterEqual(html.count("break-after: page"), 1)
        self.assertIn("break-after: auto", html)

        # Zero external HTTP(S) URLs. An optional SVG xmlns is the only
        # permitted exception, and this deck does not need one.
        urls = re.findall(r"https?://[^\s\"'<>]+", html)
        allowed = {"http://www.w3.org/2000/svg", "https://www.w3.org/2000/svg"}
        external = [url for url in urls if url.rstrip("/") not in {u.rstrip("/") for u in allowed}]
        self.assertEqual(external, [], msg=f"external URLs found: {external}")

        # No raster images, no NASA logo/meatball/worm references in SVG tags
        # or asset sources. Textual attribution is expected and allowed.
        self.assertNotIn("<img", html.lower())
        for tag in re.findall(r"<(?:svg|image|use|object|embed)\b[^>]*>", html, flags=re.IGNORECASE):
            lowered = tag.lower()
            self.assertNotIn("nasa", lowered)
            self.assertNotIn("meatball", lowered)
            self.assertNotIn("worm", lowered)
        for src in re.findall(r"(?:src|href)\s*=\s*[\"']([^\"']+)[\"']", html, flags=re.IGNORECASE):
            lowered = src.lower()
            self.assertNotIn("nasa", lowered)
            self.assertNotIn("meatball", lowered)
            self.assertNotIn("worm", lowered)
            self.assertFalse(lowered.startswith("http"), msg=f"external asset: {src}")

        self.assertIn("Synthetic Intelligence", html)
        self.assertRegex(html, r"\bAI\b")
        self.assertIn("AI Disclosure", html)
        for figure in ("85,002,288", "75,556,207", "12.8838", "Peru"):
            self.assertIn(figure, html)

    def test_public_documents_are_english_disclosed_and_command_complete(self) -> None:
        for path in (PACKAGE_PATH, README_PATH):
            self.assertTrue(path.is_file(), msg=f"missing {path}")
            text = path.read_text(encoding="utf-8")
            self.assertGreater(len(text), 2000)
            lowered = text.lower()
            self.assertIn("synthetic intelligence", lowered)
            self.assertRegex(text, r"\bAI\b")
            self.assertIn("human supervision", lowered)
            self.assertIn("NASA FIRMS Archive & NRT", text)
            self.assertIn("MODIS C6.1", text)
            self.assertIn("VIIRS S-NPP", text)
            self.assertIn("NOAA-20/J1", text)
            self.assertIn("NOAA-21/J2", text)
            self.assertIn("CPU", text)
            self.assertIn("Google Cloud", text)
            self.assertNotIn("<img", lowered)
            self.assertEqual(re.findall(r"!\[[^\]]*\]\([^)]*\)", text), [])

            # English-only public deliverables: reject common Spanish prose
            # markers from the bilingual app copy.
            for spanish in ("Presentación", "Divulgación", "Instrucciones", "Descripción"):
                self.assertNotIn(spanish, text)

            self.assertIn("scripts/export_submission.py", text)
            self.assertIn("app/server.py", text)
            self.assertIn("src/data", text)
            self.assertIn("src/models", text)
            self.assertIn("tests/test_submission_package.py", text)
            self.assertIn("scripts/fetch_processed_data.py", text)

        package = PACKAGE_PATH.read_text(encoding="utf-8")
        self.assertIn("Project Title", package)
        self.assertIn("Elevator Pitch", package)
        self.assertIn("Challenge Alignment", package)
        self.assertIn("Architecture", package)
        self.assertIn("Verified Quantitative Results", package)
        self.assertIn("Data Provenance", package)
        self.assertIn("Reproducibility", package)
        self.assertIn("Synthetic Intelligence (SI) / AI Tools Disclosure", package)

    def test_legal_license_and_third_party_notices_completeness(self) -> None:
        license_path = REPO_ROOT / "LICENSE"
        notices_path = REPO_ROOT / "THIRD_PARTY_NOTICES.md"
        self.assertTrue(license_path.is_file(), license_path)
        self.assertTrue(notices_path.is_file(), notices_path)

        license_text = license_path.read_text(encoding="utf-8")
        notices_text = notices_path.read_text(encoding="utf-8")
        self.assertGreater(len(license_text.strip()), 0)
        self.assertGreater(len(notices_text.strip()), 0)

        for required in (
            "MIT License",
            "Permission is hereby granted, free of charge",
            "AS IS",
            "CC-BY-4.0",
            "CC0",
        ):
            self.assertIn(required, license_text)

        for required in (
            "LANCE",
            "FIRMS",
            "Natural Earth",
            "tactical",
            "endorsement",
            "10.1016/j.rse.2016.02.054",
            "10.1016/j.rse.2013.12.008",
            "10.1080/01621459.2018.1448825",
            "10.1016/S0022-1694(97)00125-X",
        ):
            self.assertIn(required, notices_text)


if __name__ == "__main__":
    unittest.main()
