#!/usr/bin/env python3
"""Download and SHA-256 verify the >50 MB processed artefacts from GitHub Release.

Lightweight derived cubes (<30 MB each) and metadata live directly in Git.
The five larger processed files (total ~1.03 GB) are attached to the
GitHub Release ``v1.0-processed`` in ``LuppusMint/Nasa--SpaceApps`` so Git
clones remain fast without Git LFS quota limits.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = "LuppusMint/NASA-SpaceApps-2026"
DEFAULT_TAG = "v1.0-processed"

RELEASE_ASSETS: tuple[tuple[str, str, str], ...] = (
    (
        "cube_cell10km_monthly.parquet",
        "data/processed/cube_cell10km_monthly.parquet",
        "0f3cec0ddb398a13466e2c5c3934d6ed5cbf0300d474cd2467c1d84bb86e5b2e",
    ),
    (
        "harmonized_cell50km_monthly.parquet",
        "data/processed/harmonized_cell50km_monthly.parquet",
        "199135dc5b5958b73a1c7dc8f4af3d5da4a95c30bd4b61ca08b912d134531bbc",
    ),
    (
        "cube_cell50km_monthly.parquet",
        "data/processed/cube_cell50km_monthly.parquet",
        "4352cbce4f9b56c2cdbb1581299421e08c5eb2e722e280d624aecf478a7ba9c5",
    ),
    (
        "oof_m0_m3.parquet",
        "data/processed/oof_m0_m3.parquet",
        "f9661ba3714b4cea156333315a7fad90f39697ae8c38d5d73a7cde1e82adf507",
    ),
    (
        "splits.json",
        "data/processed/splits.json",
        "8966c51fbadcc904f5a4e398ce96b9475facd1d162aa8dcdac96357ebe1dc37c",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_all(root: Path) -> bool:
    ok = True
    for asset_name, rel_dest, expected_sha in RELEASE_ASSETS:
        dest = root / rel_dest
        if not dest.is_file():
            print(f"MISSING  {rel_dest} (asset: {asset_name})")
            ok = False
            continue
        actual = sha256_file(dest)
        if actual != expected_sha:
            print(f"MISMATCH {rel_dest}: got {actual}, expected {expected_sha}")
            ok = False
        else:
            print(f"OK       {rel_dest} ({dest.stat().st_size:,} bytes)")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root directory")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub owner/repo")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Release tag")
    parser.add_argument("--verify-only", action="store_true", help="Only verify SHA-256 checksums")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if args.verify_only:
        return 0 if verify_all(root) else 1

    gh = shutil.which("gh")
    if not gh:
        print(
            "GitHub CLI (`gh`) not found. Install `gh` and run `gh auth login`, then re-run,\n"
            f"or download the 5 assets manually from https://github.com/{args.repo}/releases/tag/{args.tag}.",
            file=sys.stderr,
        )
        return 1

    for asset_name, rel_dest, expected_sha in RELEASE_ASSETS:
        dest = root / rel_dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file() and sha256_file(dest) == expected_sha:
            print(f"SKIP     {rel_dest} (checksum verified)")
            continue
        tmp_dir = dest.parent / ".release-tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            gh,
            "release",
            "download",
            args.tag,
            "--repo",
            args.repo,
            "--pattern",
            asset_name,
            "--dir",
            str(tmp_dir),
            "--clobber",
        ]
        print("DOWNLOAD", asset_name, "->", rel_dest)
        subprocess.run(cmd, check=True)
        downloaded = tmp_dir / asset_name
        if not downloaded.is_file():
            raise RuntimeError(f"Expected asset not downloaded: {downloaded}")
        downloaded.replace(dest)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    return 0 if verify_all(root) else 1


if __name__ == "__main__":
    raise SystemExit(main())
