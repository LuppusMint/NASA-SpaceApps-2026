#!/usr/bin/env python3
"""Reproducible downloader for South America NASA FIRMS (2000-present) and Natural Earth boundaries.

Does not hardcode API keys; reads FIRMS_MAP_KEY from environment for 2023-2026 API gap-fills.
Logs full provenance (URLs, sizes, SHA-256 checksums, dates, licenses, units) to data/PROVENANCE.json.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
COUNTRY_DIR = DATA_DIR / "raw" / "firms_country"
API_DIR = DATA_DIR / "raw" / "firms_api_fill"
BOUNDARY_DIR = DATA_DIR / "boundaries"

SA_COUNTRIES = {
    "Argentina",
    "Bolivia",
    "Brazil",
    "Chile",
    "Colombia",
    "Ecuador",
    "French_Guiana",
    "Guyana",
    "Paraguay",
    "Peru",
    "Suriname",
    "Uruguay",
    "Venezuela",
}

SA_BBOX = "-82.2,-59.5,-27.4,13.7"
COUNTRY_MANIFEST_URL = "https://firms.modaps.eosdis.nasa.gov/data/country/yearly_summary_files.txt"

BOUNDARY_SOURCES = {
    "ne_10m_admin_0_countries.zip": "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip",
    "ne_10m_admin_1_states_provinces.zip": "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_stream(
    url: str,
    dest: Path,
    *,
    expected_size: int | None = None,
    expected_prefix: bytes | None = None,
    retries: int = 4,
    timeout: int = 120,
) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        sz = dest.stat().st_size
        if expected_size is not None and sz == expected_size:
            return {"status": "skipped_exact", "path": str(dest), "size": sz}
        if expected_size is None and sz > 0:
            return {"status": "skipped_exists", "path": str(dest), "size": sz}

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nasa-spaceapps-data-prep/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                first_chunk = resp.read(65536)
                if expected_prefix is not None and not first_chunk.startswith(expected_prefix):
                    snippet = first_chunk[:200].decode("utf-8", "replace")
                    raise ValueError(f"Unexpected payload prefix for {url}: {snippet!r}")
                with tmp.open("wb") as out:
                    out.write(first_chunk)
                    while True:
                        chunk = resp.read(262144)
                        if not chunk:
                            break
                        out.write(chunk)
            actual = tmp.stat().st_size
            if expected_size is not None and actual != expected_size:
                raise IOError(f"Size mismatch for {dest.name}: got {actual}, expected {expected_size}")
            tmp.replace(dest)
            return {"status": "downloaded", "path": str(dest), "size": actual}
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            time.sleep(min(2**attempt, 16))
    raise RuntimeError(f"Failed downloading {url} after {retries} attempts: {last_err}")


def fetch_country_manifest() -> list[tuple[str, int, str, int, str]]:
    req = urllib.request.Request(COUNTRY_MANIFEST_URL, headers={"User-Agent": "nasa-spaceapps-data-prep/1.0"})
    text = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    items: list[tuple[str, int, str, int, str]] = []
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        fn, size_b = parts[0], int(parts[1])
        m = re.match(r"^(modis|viirs-snpp|viirs-jpss1)_(20\d{2})_(.+?)\.csv$", fn)
        if not m:
            continue
        sensor, year_str, country = m.group(1), m.group(2), m.group(3)
        if country in SA_COUNTRIES:
            items.append((sensor, int(year_str), country, size_b, fn))
    return sorted(items)


def fetch_availability(api_key: str) -> dict[str, tuple[dt.date, dt.date]]:
    url = f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{api_key}/all"
    text = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "replace")
    out: dict[str, tuple[dt.date, dt.date]] = {}
    for line in text.splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            out[parts[0]] = (dt.date.fromisoformat(parts[1]), dt.date.fromisoformat(parts[2]))
    return out


def date_chunks(start: dt.date, end: dt.date, step_days: int = 5) -> list[tuple[dt.date, int]]:
    chunks: list[tuple[dt.date, int]] = []
    cur = start
    while cur <= end:
        span = min(step_days, (end - cur).days + 1)
        chunks.append((cur, span))
        cur += dt.timedelta(days=span)
    return chunks


def main() -> int:
    api_key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not api_key:
        raise SystemExit("ERROR: Set FIRMS_MAP_KEY environment variable before running.")

    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
    COUNTRY_DIR.mkdir(parents=True, exist_ok=True)
    API_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Downloading Natural Earth 10m boundaries...")
    boundary_records = []
    for fname, url in BOUNDARY_SOURCES.items():
        dest = BOUNDARY_DIR / fname
        res = download_stream(url, dest, expected_prefix=b"PK")
        boundary_records.append(
            {
                "file": str(dest.relative_to(BASE_DIR)),
                "url": url,
                "license": "Public Domain (Natural Earth)",
                "size_bytes": dest.stat().st_size,
                "sha256": sha256_file(dest),
                "status": res["status"],
            }
        )
        print(f"  {fname}: {res['status']} ({dest.stat().st_size:,} B)")

    print("[2/4] Fetching FIRMS country archive manifest & downloading South America (2000-2025)...")
    manifest_items = fetch_country_manifest()
    total_expected = sum(x[3] for x in manifest_items)
    print(f"  Targets: {len(manifest_items)} CSV files across 13 territories ({total_expected / 1e6:,.1f} MB)")

    def _dl_country(item: tuple[str, int, str, int, str]) -> dict:
        sensor, year, _country, size_b, fn = item
        url = f"https://firms.modaps.eosdis.nasa.gov/data/country/{sensor}/{year}/{fn}"
        dest = COUNTRY_DIR / sensor / fn
        res = download_stream(url, dest, expected_size=size_b, expected_prefix=b"latitude")
        return {"sensor": sensor, "year": year, "file": fn, "size": size_b, "status": res["status"]}

    country_results = []
    done_cnt = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_dl_country, it) for it in manifest_items]
        for fut in concurrent.futures.as_completed(futs):
            country_results.append(fut.result())
            done_cnt += 1
            if done_cnt % 100 == 0 or done_cnt == len(manifest_items):
                print(f"  country progress: {done_cnt}/{len(manifest_items)}")

    print("[3/4] Querying FIRMS API availability & downloading 2026-07-02..present + NOAA-21 pre-2024-07...")
    avail = fetch_availability(api_key)
    api_tasks: list[tuple[str, dt.date, int]] = []

    def _enqueue_range(source: str, start: dt.date, end: dt.date) -> None:
        if source not in avail:
            return
        min_d, max_d = avail[source]
        lo = max(start, min_d)
        hi = min(end, max_d)
        if lo <= hi:
            for sdate, span in date_chunks(lo, hi, 5):
                api_tasks.append((source, sdate, span))

    # Fill July 2, 2026 -> latest available across all 4 sensors
    tail_start = dt.date(2026, 7, 2)
    far_future = dt.date(2027, 1, 1)
    for sp_src, nrt_src in [
        ("MODIS_SP", "MODIS_NRT"),
        ("VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"),
        ("VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT"),
        ("VIIRS_NOAA21_SP", "VIIRS_NOAA21_NRT"),
    ]:
        sp_max = avail.get(sp_src, (dt.date(2000, 1, 1), dt.date(2000, 1, 1)))[1]
        _enqueue_range(sp_src, tail_start, sp_max)
        _enqueue_range(nrt_src, max(tail_start, sp_max + dt.timedelta(days=1)), far_future)

    # Also backfill VIIRS_NOAA21_SP from its mission archive start (2023-03-01) up to 2024-06-30
    _enqueue_range("VIIRS_NOAA21_SP", dt.date(2023, 3, 1), dt.date(2024, 6, 30))
    print(f"  API 5-day batch requests scheduled: {len(api_tasks)}")

    def _dl_api(task: tuple[str, dt.date, int]) -> dict:
        src, sdate, span = task
        fn = f"{src}_{sdate.isoformat()}_{span}d.csv"
        dest = API_DIR / src / fn
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{src}/{SA_BBOX}/{span}/{sdate.isoformat()}"
        res = download_stream(url, dest, expected_prefix=b"latitude", timeout=180)
        return {"source": src, "date": sdate.isoformat(), "span": span, "file": str(dest.relative_to(BASE_DIR)), "size": res["size"], "status": res["status"]}

    api_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(_dl_api, t) for t in api_tasks]
        for idx, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            api_results.append(fut.result())
            if idx % 25 == 0 or idx == len(api_tasks):
                print(f"  API fill progress: {idx}/{len(api_tasks)}")

    print("[4/4] Recording local ZIP provenance & writing data/PROVENANCE.json...")
    local_zips = []
    for zp in sorted(BASE_DIR.glob("DL_FIRE_*.zip")):
        local_zips.append(
            {
                "file": zp.name,
                "size_bytes": zp.stat().st_size,
                "sha256": sha256_file(zp),
                "note": "User-downloaded NASA FIRMS Archive (2024-07-01..2026-07-01, South America bbox)",
            }
        )

    provenance = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "started_at_utc": started_utc,
        "bbox": SA_BBOX,
        "territories": sorted(SA_COUNTRIES),
        "license_firms": "NASA Earthdata / LANCE FIRMS open public-use data (attribute NASA FIRMS)",
        "units": {
            "latitude_longitude": "decimal degrees WGS84 (EPSG:4326)",
            "brightness_primary": "Kelvin (MODIS Channel 21/22 brightness; VIIRS I4 bright_ti4)",
            "brightness_secondary": "Kelvin (MODIS Channel 31 bright_t31; VIIRS I5 bright_ti5)",
            "frp": "MW (Fire Radiative Power)",
            "scan_track": "km nominal pixel footprint",
            "acq_date_acq_time": "UTC date and HHMM",
            "type_archive_only": "0=presumed vegetation fire, 1=active volcano, 2=other static land source, 3=offshore",
        },
        "boundaries": boundary_records,
        "user_archive_zips": local_zips,
        "country_archives": {
            "manifest_url": COUNTRY_MANIFEST_URL,
            "file_count": len(country_results),
            "total_bytes": sum(r["size"] for r in country_results),
            "skipped_exact": sum(1 for r in country_results if r["status"] == "skipped_exact"),
            "downloaded": sum(1 for r in country_results if r["status"] == "downloaded"),
        },
        "api_availability_snapshot": {k: [v[0].isoformat(), v[1].isoformat()] for k, v in sorted(avail.items())},
        "api_gap_fills": {
            "request_count": len(api_results),
            "total_bytes": sum(r["size"] for r in api_results),
            "downloaded": sum(1 for r in api_results if r["status"] == "downloaded"),
            "skipped_exists": sum(1 for r in api_results if r["status"] == "skipped_exists"),
            "ranges": sorted({(r["source"], r["date"]) for r in api_results})[:5],
        },
    }
    prov_path = DATA_DIR / "PROVENANCE.json"
    prov_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"DONE -> {prov_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
