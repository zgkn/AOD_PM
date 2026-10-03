#!/usr/bin/env python3
"""Download the latest 3-day CAMS organic matter AOD forecast for Singapore.

Dataset: cams-global-atmospheric-composition-forecasts
Variable: organic_matter_aerosol_optical_depth_550nm
Same single grid cell as the EAC4/ERA5 pulls: 1.5N, 103.5E.
Leadtime: 0..72h step 3 (3-hourly, matching the training data's cadence).

Forecast cycles (00Z) take a few hours to publish after the cycle time, so
today's cycle may not exist yet; tries today's base date first and falls
back to earlier days (up to --max-days-back) until one succeeds.
"""
import argparse
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cdsapi

DATASET = "cams-global-atmospheric-composition-forecasts"
VARIABLE = "organic_matter_aerosol_optical_depth_550nm"

# Same single grid cell as the EAC4/ERA5 pulls: [North, West, South, East].
AREA = [1.5, 103.5, 1.5, 103.5]
BASE_TIME = "00:00"
LEADTIME_HOURS = [str(h) for h in range(0, 73, 3)]


def unwrap_if_zip(path: Path) -> None:
    # This forecast dataset has been observed to package output as a zip
    # even when netcdf is requested; detect and unwrap it defensively,
    # same pattern used for the ERA5 downloads.
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


def build_request(date_str: str) -> dict:
    return {
        "variable": [VARIABLE],
        "date": [f"{date_str}/{date_str}"],
        "time": [BASE_TIME],
        "leadtime_hour": LEADTIME_HOURS,
        "type": ["forecast"],
        "area": AREA,
        "data_format": "netcdf_zip",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=Path("output/cams_forecast_aod_om.nc"), type=Path)
    parser.add_argument(
        "--max-days-back", type=int, default=2,
        help="How many days back to try if today's forecast cycle isn't published yet.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()
    today = datetime.now(timezone.utc).date()

    last_error = None
    for days_back in range(args.max_days_back + 1):
        date_str = (today - timedelta(days=days_back)).isoformat()
        print(f"Trying forecast base date {date_str} {BASE_TIME} UTC...")
        try:
            client.retrieve(DATASET, build_request(date_str)).download(str(args.output))
            unwrap_if_zip(args.output)
            print(f"OK: downloaded forecast base date {date_str} -> {args.output}")
            return 0
        except Exception as exc:
            print(f"  not available: {exc}", file=sys.stderr)
            last_error = exc

    print(
        f"FAILED: no forecast cycle available in the last {args.max_days_back + 1} days. "
        f"Last error: {last_error}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
