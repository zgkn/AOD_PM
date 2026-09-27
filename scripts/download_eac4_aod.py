#!/usr/bin/env python3
"""Download CAMS EAC4 reanalysis AOD data for Singapore, one NetCDF file per year.

Assumes ~/.cdsapirc is already configured (url + key) and that `cdsapi`'s
modern client is installed, which submits, polls, and downloads within a
single client.retrieve(...).download(...) call.
"""
import argparse
import json
import sys
from pathlib import Path

import cdsapi

DATASET = "cams-global-reanalysis-eac4"
VARIABLES = [
    "organic_matter_aerosol_optical_depth_550nm",
    "total_aerosol_optical_depth_550nm",
]
TIMES = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]

# Nearest native 0.75x0.75 deg EAC4 grid point to Singapore's centroid
# (1.35N, 103.82E): [North, West, South, East] pinned to a single cell.
AREA = [1.5, 103.5, 1.5, 103.5]

# Task specifies the range starts mid-year for 2014.
FIRST_DATE_OVERRIDE = {2014: "2014-01-04"}


def date_range(year: int) -> str:
    start = FIRST_DATE_OVERRIDE.get(year, f"{year}-01-01")
    end = f"{year}-12-31"
    return f"{start}/{end}"


def download_year(client: cdsapi.Client, year: int, outdir: Path) -> Path:
    target = outdir / f"cams_eac4_aod_singapore_{year}.nc"
    request = {
        "variable": VARIABLES,
        "date": date_range(year),
        "time": TIMES,
        "area": AREA,
        # NOTE: "data_format" matches the current (post-migration) ADS/CDS
        # request schema. If ADS rejects this key for this dataset, swap it
        # for "format" (the legacy key name) instead.
        "data_format": "netcdf",
    }
    client.retrieve(DATASET, request).download(str(target))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years", required=True, help="Comma-separated years to download, e.g. 2014,2015"
    )
    parser.add_argument("--outdir", default="output", help="Directory to write NetCDF files to")
    args = parser.parse_args()

    years = [int(y) for y in args.years.split(",") if y.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()
    results = {}
    for year in years:
        print(f"::group::Downloading {year}")
        try:
            target = download_year(client, year, outdir)
            size = target.stat().st_size
            print(f"[{year}] OK -> {target} ({size} bytes)")
            results[str(year)] = "ok"
        except Exception as exc:
            print(f"[{year}] FAILED: {exc}", file=sys.stderr)
            results[str(year)] = f"failed: {exc}"
        print("::endgroup::")

    (outdir / "summary.json").write_text(json.dumps(results, indent=2))

    print("\n=== Summary ===")
    failed = []
    for year in years:
        status = results.get(str(year), "missing")
        print(f"{year}: {status}")
        if status != "ok":
            failed.append(year)

    if failed:
        print(f"\n{len(failed)} year(s) failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
