#!/usr/bin/env python3
"""Download hourly ERA5 pressure-level u/v over Southeast Asia from ARCO-ERA5.

ARCO-ERA5 is Google's public, analysis-ready Zarr copy of the ERA5 reanalysis
(the same fields as the Copernicus CDS, 0.25 deg, hourly, 37 pressure levels).
Reading it needs no account and has no request queue, unlike the CDS API.

Every hour in the inclusive UTC date range is fetched as-is (no averaging) and
written as one era5_pl_wind_YYYY-MM-DD.nc per day (24 hours, requested levels),
the layout plot_era5_wind_mean.py reads. Averaging happens there.
"""
import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

STORE = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VARIABLES = ["u_component_of_wind", "v_component_of_wind"]
# [North, West, South, East]
AREA = [25, 90, -11, 145]
WORKERS = 8


def fetch_hour(ds: xr.Dataset, hour: np.datetime64, levels: list[int]) -> xr.Dataset:
    north, west, south, east = AREA
    return ds[VARIABLES].sel(
        time=hour, level=levels, latitude=slice(north, south), longitude=slice(west, east)
    ).load()


def fetch_day(ds: xr.Dataset, day: dt.date, levels: list[int], target: Path) -> Path:
    start = np.datetime64(day.isoformat())
    hours = [start + np.timedelta64(h, "h") for h in range(24)]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        parts = list(pool.map(lambda h: fetch_hour(ds, h, levels), hours))
    out = xr.concat(parts, dim="time")
    n_times = int(out.sizes["time"])
    got_levels = sorted(int(p) for p in out["level"].values)
    if n_times != 24 or got_levels != sorted(levels):
        raise RuntimeError(f"{day}: {n_times} hours (want 24), levels {got_levels} (want {sorted(levels)})")
    if any(bool(np.isnan(out[v]).any()) for v in VARIABLES):
        raise RuntimeError(f"{day}: NaNs in the downloaded winds")
    out.to_netcdf(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="first UTC day, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="last UTC day (inclusive), YYYY-MM-DD")
    parser.add_argument("--levels", default="925,850", help="comma-separated hPa levels")
    parser.add_argument("--outdir", default="raw", type=Path)
    args = parser.parse_args()

    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    n_days = (end - start).days + 1
    if not 1 <= n_days <= 31:
        raise SystemExit(f"range must be 1-31 days, got {n_days}")
    levels = [int(p) for p in args.levels.split(",")]
    args.outdir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_zarr(STORE, chunks=None)
    print(f"ARCO-ERA5 u/v at {levels} hPa, area {AREA}, {start} to {end} ({n_days} days)", flush=True)
    for i in range(n_days):
        day = start + dt.timedelta(days=i)
        target = args.outdir / f"era5_pl_wind_{day.isoformat()}.nc"
        fetch_day(ds, day, levels, target)
        print(f"OK {day} -> {target} ({target.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
