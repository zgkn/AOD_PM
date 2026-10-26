#!/usr/bin/env python3
"""Download one UTC day of hourly ERA5 pressure-level u/v over Southeast Asia.

One request per day keeps each CDS request small and lets the workflow run
every day as its own parallel job, so queue waits overlap and a failed day can
be re-run alone. The raw NetCDF (all 24 hours, all requested levels) is written
unchanged to --outdir; averaging happens later in plot_era5_wind_mean.py.
"""
import argparse
import datetime as dt
import shutil
import tempfile
import zipfile
from pathlib import Path

import cdsapi
import xarray as xr

DATASET = "reanalysis-era5-pressure-levels"
# [North, West, South, East]
AREA = [25, 90, -11, 145]
ALL_HOURS = [f"{h:02d}:00" for h in range(24)]


def build_request(day: dt.date, levels: list[int]) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": ["u_component_of_wind", "v_component_of_wind"],
        "pressure_level": [str(p) for p in levels],
        "year": [str(day.year)],
        "month": [f"{day.month:02d}"],
        "day": [f"{day.day:02d}"],
        "time": ALL_HOURS,
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def unwrap_if_zip(path: Path) -> None:
    # CDS sometimes returns a zip even when unarchived is requested.
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


def verify(path: Path, levels: list[int]) -> None:
    """Fail loudly if the file is not a complete day for every level."""
    ds = xr.open_dataset(path)
    time_dim = next((d for d in ("valid_time", "time") if d in ds.dims), None)
    level_dim = next((d for d in ("pressure_level", "level") if d in ds.dims), None)
    if time_dim is None or level_dim is None:
        raise RuntimeError(f"unexpected dims {dict(ds.sizes)} in {path}")
    n_times = int(ds.sizes[time_dim])
    got_levels = sorted(int(p) for p in ds[level_dim].values)
    if n_times != 24 or got_levels != sorted(levels):
        raise RuntimeError(f"{path}: {n_times} hours (want 24), levels {got_levels} (want {sorted(levels)})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="UTC day, YYYY-MM-DD")
    parser.add_argument("--levels", default="925,850", help="comma-separated hPa levels")
    parser.add_argument("--outdir", default="raw", type=Path)
    args = parser.parse_args()

    day = dt.date.fromisoformat(args.date)
    levels = [int(p) for p in args.levels.split(",")]
    args.outdir.mkdir(parents=True, exist_ok=True)
    target = args.outdir / f"era5_pl_wind_{day.isoformat()}.nc"

    print(f"Requesting {DATASET} u/v at {levels} hPa for {day.isoformat()}, area {AREA}")
    cdsapi.Client().retrieve(DATASET, build_request(day, levels)).download(str(target))
    unwrap_if_zip(target)
    verify(target, levels)
    print(f"OK -> {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
