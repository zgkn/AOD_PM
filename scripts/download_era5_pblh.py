#!/usr/bin/env python3
"""Download ERA5 boundary layer height for Singapore, one NetCDF file per year.

Uses the same grid cell as the EAC4 pull (1.5N, 103.5E). ERA5's native 0.25x0.25
grid is an exact refinement of EAC4's 0.75x0.75 grid (0.75 = 3 x 0.25), so this
point is a genuine ERA5 grid node, not an interpolated value -- the two series
are spatially co-located and can be joined on timestamp alone.

Assumes ~/.cdsapirc is already configured (url + key) and that `cdsapi`'s
modern client is installed, which submits, polls, and downloads within a
single client.retrieve(...).download(...) call.
"""
import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import cdsapi
import xarray as xr

DATASET = "reanalysis-era5-single-levels"
VARIABLE = "boundary_layer_height"
TIMES = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]

# Same single grid cell as the EAC4 pull: [North, West, South, East].
AREA = [1.5, 103.5, 1.5, 103.5]

ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]

# Task specifies the range starts 2014-01-04, matching EAC4. ERA5's request
# schema takes month/day lists rather than a date range, so 2014 needs two
# sub-requests (Jan 4-31, then Feb-Dec) merged into one yearly file.
FIRST_DAY_OVERRIDE = {2014: 4}


def year_chunks(year: int):
    first_day = FIRST_DAY_OVERRIDE.get(year)
    if first_day is None:
        return [(ALL_MONTHS, ALL_DAYS)]
    jan_days = [f"{d:02d}" for d in range(first_day, 32)]
    return [(["01"], jan_days), (ALL_MONTHS[1:], ALL_DAYS)]


def build_request(year: int, months: list, days: list) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": [VARIABLE],
        "year": [str(year)],
        "month": months,
        "day": days,
        "time": TIMES,
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def unwrap_if_zip(path: Path) -> None:
    # NOTE: CDS has a documented bug where it sometimes returns a zip archive
    # even when download_format=unarchived is requested. Detect and unwrap it
    # so a zip never silently masquerades as a .nc file downstream.
    with open(path, "rb") as f:
        if f.read(2) != b"PK":
            return
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(path) as zf:
            nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
            if not nc_names:
                raise RuntimeError(f"{path} is a zip but contains no .nc file")
            zf.extract(nc_names[0], tmp)
            shutil.move(str(Path(tmp) / nc_names[0]), path)


def download_year(client: cdsapi.Client, year: int, outdir: Path) -> Path:
    target = outdir / f"era5_pblh_singapore_{year}.nc"
    chunks = year_chunks(year)

    if len(chunks) == 1:
        months, days = chunks[0]
        client.retrieve(DATASET, build_request(year, months, days)).download(str(target))
        unwrap_if_zip(target)
        return target

    with tempfile.TemporaryDirectory() as tmp:
        datasets = []
        for i, (months, days) in enumerate(chunks):
            part_path = Path(tmp) / f"part_{i}.nc"
            client.retrieve(DATASET, build_request(year, months, days)).download(str(part_path))
            unwrap_if_zip(part_path)
            datasets.append(xr.open_dataset(part_path).load())
        # CDS has changed the netCDF time dimension's name across versions
        # ("time" vs "valid_time"); detect whichever this download used
        # rather than assuming one.
        time_dim = next(d for d in ("valid_time", "time") if d in datasets[0].dims)
        combined = xr.concat(datasets, dim=time_dim).sortby(time_dim)
        combined.to_netcdf(target)
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
