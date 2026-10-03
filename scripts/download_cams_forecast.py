#!/usr/bin/env python3
"""Download the latest 5-day CAMS organic matter AOD forecast for Singapore.

Dataset: cams-global-atmospheric-composition-forecasts
Variable: organic_matter_aerosol_optical_depth_550nm
Downloads a 3x3 degree box centered on 1.5N/103.5E (the EAC4/ERA5 grid
point used elsewhere in this project) rather than a single grid cell --
build_forecast_dashboard.py takes the max AOD_om over this box per
timestep, to account for forecast plume-position uncertainty rather than
betting on one exact grid cell.
Leadtime: 0..120h step 3 (3-hourly, matching the training data's cadence).

This dataset issues a forecast cycle every 12h (00Z and 12Z), and each
cycle takes a few hours to publish after its nominal time. To get the
actual latest cycle rather than always assuming 00Z, this script finds
the most recent 00Z/12Z cycle time that has already occurred (relative to
now, in UTC) and tries that first, then walks backwards 12h at a time
(up to --max-cycles-back) until a download succeeds -- so it picks up a
12Z cycle instead of waiting for the next day's 00Z when 12Z is in fact
the newest one published.
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

# 3x3 degree box centered on 1.5N/103.5E, [North, West, South, East].
# On this dataset's native ~0.4 degree grid that's roughly an 8x8 cell
# region (~330km x 330km), covering Singapore, Johor, the Riau Islands,
# and part of Sumatra's east coast -- wide enough to capture a forecast
# smoke plume landing a bit off from the exact Singapore grid cell, while
# still regional rather than picking up unrelated distant fire sources.
# (A degenerate single-point box also doesn't work here: this dataset's
# native grid doesn't land exactly on 1.5N/103.5E, so even the old
# single-cell version needed a small box just to get a hit.)
AREA = [3.0, 102.0, 0.0, 105.0]
CYCLE_HOURS = (0, 12)  # forecast cycles this dataset issues per day, UTC
LEADTIME_HOURS = [str(h) for h in range(0, 121, 3)]


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


def build_request(date_str: str, time_str: str) -> dict:
    return {
        "variable": [VARIABLE],
        "date": [f"{date_str}/{date_str}"],
        "time": [time_str],
        "leadtime_hour": LEADTIME_HOURS,
        "type": ["forecast"],
        "area": AREA,
        "data_format": "netcdf_zip",
    }


def latest_cycles(now: datetime, max_cycles_back: int) -> list[datetime]:
    """Most recent CYCLE_HOURS cycle times <= now, newest first.

    Walking in 12h steps (rather than whole days at a fixed hour) is what
    lets this correctly pick a 12Z cycle as "latest" instead of always
    waiting for the next 00Z, when 12Z is in fact the most recent one
    issued.
    """
    today_cycles = sorted(
        now.replace(hour=h, minute=0, second=0, microsecond=0) for h in CYCLE_HOURS
    )
    cycle = max((c for c in today_cycles if c <= now), default=today_cycles[0] - timedelta(days=1))
    cycles = []
    for _ in range(max_cycles_back + 1):
        cycles.append(cycle)
        cycle -= timedelta(hours=24 // len(CYCLE_HOURS))
    return cycles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=Path("output/cams_forecast_aod_om.nc"), type=Path)
    parser.add_argument(
        "--max-cycles-back", type=int, default=5,
        help="How many 12h cycles back to try if the latest cycle isn't published yet.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()
    now = datetime.now(timezone.utc)

    last_error = None
    for cycle in latest_cycles(now, args.max_cycles_back):
        date_str = cycle.date().isoformat()
        time_str = cycle.strftime("%H:%M")
        print(f"Trying forecast cycle {date_str} {time_str} UTC...")
        try:
            client.retrieve(DATASET, build_request(date_str, time_str)).download(str(args.output))
            unwrap_if_zip(args.output)
            print(f"OK: downloaded forecast cycle {date_str} {time_str} UTC -> {args.output}")
            return 0
        except Exception as exc:
            print(f"  not available: {exc}", file=sys.stderr)
            last_error = exc

    print(
        f"FAILED: no forecast cycle available in the last {args.max_cycles_back + 1} cycles. "
        f"Last error: {last_error}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
