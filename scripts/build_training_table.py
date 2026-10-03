#!/usr/bin/env python3
"""Build the EAC4/ERA5/PM2.5 training table for regression.

Joins, on UTC timestamp (3-hourly: 00/03/06/09/12/15/18/21):
  - NEA 1-hour PM2.5 CSV (downsampled to those 8 timestamps/day, SGT -> UTC)
  - EAC4 AOD (organic matter, total) from the eac4-data-2014-2025 release
  - ERA5 boundary layer height from the era5-pblh-2014-2025 release
  - ERA5 10m wind speed from the era5-wind10m-2014-2025 release

Any row missing a value in any source is dropped entirely -- no fill or
interpolation. Reanalysis NetCDFs are downloaded from the public GitHub
Release assets and cached locally; re-running the script reuses the cache.
"""
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

YEARS = range(2014, 2026)
TARGET_UTC_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]
SGT_OFFSET_HOURS = 8
EXPECTED_LAT = 1.5
EXPECTED_LON = 103.5

RELEASE_URLS = {
    "eac4": "https://github.com/zgkn/AOD_PM/releases/download/eac4-data-2014-2025/cams_eac4_aod_singapore_{year}.nc",
    "pblh": "https://github.com/zgkn/AOD_PM/releases/download/era5-pblh-2014-2025/era5_pblh_singapore_{year}.nc",
    "wind": "https://github.com/zgkn/AOD_PM/releases/download/era5-wind10m-2014-2025/era5_wind10m_singapore_{year}.nc",
}

PM25_STATION_COLS = ["north", "south", "east", "west", "central"]

# One known export glitch in the source CSV: a 3-digit hour field for
# midnight. The date and the surrounding hourly sequence (...3/4/2016 23:00,
# [this row], 4/4/2016 1:00, 2:00...) unambiguously fix this as 00:00 SGT on
# 4 April 2016.
TIMESTAMP_OVERRIDES = {
    "2016-04-04 010:00:00": "4/4/2016 0:00",
}


def ensure_downloaded(url_template: str, cache_dir: Path, year: int) -> Path:
    url = url_template.format(year=year)
    target = cache_dir / url.rsplit("/", 1)[-1]
    if not target.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, target)
    return target


def load_pm25(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={df.columns[0]: "timestamp_sgt"})

    raw = df["timestamp_sgt"].astype(str)
    override_hits = raw.isin(TIMESTAMP_OVERRIDES)
    if override_hits.any():
        for bad, fixed in TIMESTAMP_OVERRIDES.items():
            if (raw == bad).any():
                print(f"PM2.5 input: correcting malformed timestamp {bad!r} -> {fixed!r}.")
        raw = raw.replace(TIMESTAMP_OVERRIDES)

    # Source format is day-first with no zero-padding, e.g. "1/4/2014 1:00".
    # Any row not matching this (beyond the known overrides above) raises
    # rather than silently mis-parsing.
    df["timestamp_sgt"] = pd.to_datetime(raw, format="%d/%m/%Y %H:%M")

    dup_mask = df["timestamp_sgt"].duplicated(keep=False)
    if dup_mask.any():
        dup_timestamps = sorted(df.loc[dup_mask, "timestamp_sgt"].unique())
        print(
            f"PM2.5 input: {int(dup_mask.sum())} rows share {len(dup_timestamps)} duplicate "
            f"timestamp(s) with conflicting readings ({[str(t) for t in dup_timestamps]}) "
            "-- averaging duplicates per timestamp rather than picking one arbitrarily."
        )
    df = df.groupby("timestamp_sgt", as_index=False)[PM25_STATION_COLS].mean()

    df["timestamp"] = df["timestamp_sgt"] - pd.Timedelta(hours=SGT_OFFSET_HOURS)
    df["pm25_max"] = df[PM25_STATION_COLS].max(axis=1)
    df["pm25_mean"] = df[PM25_STATION_COLS].mean(axis=1)

    mask = df["timestamp"].dt.hour.isin(TARGET_UTC_HOURS) & (df["timestamp"].dt.minute == 0)
    print(
        f"PM2.5: kept {int(mask.sum())} of {len(df)} hourly rows after downsampling to "
        f"3-hourly UTC ({len(df) - int(mask.sum())} dropped)."
    )

    return (
        df.loc[mask, ["timestamp", "pm25_max", "pm25_mean"]]
        .set_index("timestamp")
        .sort_index()
    )


def concat_years(source: str, cache_dir: Path) -> xr.Dataset:
    datasets = [
        xr.open_dataset(ensure_downloaded(RELEASE_URLS[source], cache_dir, year))
        for year in YEARS
    ]
    # CDS has used different time dimension names ("time" vs "valid_time")
    # across backend versions; detect whichever this download used.
    time_dim = next(d for d in ("valid_time", "time") if d in datasets[0].dims)
    combined = xr.concat(datasets, dim=time_dim).sortby(time_dim)
    if time_dim != "time":
        combined = combined.rename({time_dim: "time"})
    return combined


def check_grid_cell(name: str, ds: xr.Dataset) -> None:
    lat = float(np.squeeze(ds["latitude"].values))
    lon = float(np.squeeze(ds["longitude"].values))
    if abs(lat - EXPECTED_LAT) > 1e-6 or abs(lon - EXPECTED_LON) > 1e-6:
        raise SystemExit(
            f"Grid cell mismatch in {name}: got ({lat}, {lon}), expected "
            f"({EXPECTED_LAT}, {EXPECTED_LON}). Refusing to silently proceed."
        )


def to_series(ds: xr.Dataset, var: str, name: str) -> pd.Series:
    data = ds[var].squeeze(dim=["latitude", "longitude"], drop=True)
    if data.dims != ("time",):
        raise SystemExit(
            f"Unexpected dims for {var}: {data.dims} (expected only 'time' after "
            "squeezing out the single-point latitude/longitude)."
        )
    return pd.Series(data.values.astype(float), index=pd.to_datetime(ds["time"].values), name=name)


def find_aod_vars(ds: xr.Dataset) -> tuple:
    om = next((n for n in ("omaod550", "organic_matter_aerosol_optical_depth_550nm") if n in ds.data_vars), None)
    total = next((n for n in ("aod550", "total_aerosol_optical_depth_550nm") if n in ds.data_vars), None)
    if om is None or total is None:
        raise SystemExit(f"Could not find AOD variables among: {list(ds.data_vars)}")
    return om, total


def find_pblh_var(ds: xr.Dataset) -> str:
    if "blh" not in ds.data_vars:
        raise SystemExit(f"Could not find PBLH variable among: {list(ds.data_vars)}")
    return "blh"


def find_wind_speed(ds: xr.Dataset) -> pd.Series:
    if "wind_speed" in ds.data_vars:
        return to_series(ds, "wind_speed", "wind10m")
    u = next((n for n in ("u10", "10m_u_component_of_wind") if n in ds.data_vars), None)
    v = next((n for n in ("v10", "10m_v_component_of_wind") if n in ds.data_vars), None)
    if u is None or v is None:
        raise SystemExit(f"Could not find wind_speed or u/v components among: {list(ds.data_vars)}")
    u_series = to_series(ds, u, "u10")
    v_series = to_series(ds, v, "v10")
    speed = np.sqrt(u_series.values ** 2 + v_series.values ** 2)
    return pd.Series(speed, index=u_series.index, name="wind10m")


def check_timestamps_aligned(series_by_name: dict) -> None:
    names = list(series_by_name)
    ref_name = names[0]
    ref = set(series_by_name[ref_name].index)
    for name in names[1:]:
        idx = set(series_by_name[name].index)
        only_ref, only_other = ref - idx, idx - ref
        if only_ref or only_other:
            print(
                f"WARNING: timestamp mismatch between {ref_name} and {name}: "
                f"{len(only_ref)} timestamps only in {ref_name}, {len(only_other)} only in "
                f"{name}. These will be dropped at the join step, not silently filled."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pm25-csv", required=True, type=Path, help="Path to Historical1hrPM2_5.csv")
    parser.add_argument("--cache-dir", default=Path("data_cache"), type=Path)
    parser.add_argument("--output", default=Path("data/training_table.csv"), type=Path)
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    args = parser.parse_args()

    pm25 = load_pm25(args.pm25_csv)

    eac4_ds = concat_years("eac4", args.cache_dir)
    pblh_ds = concat_years("pblh", args.cache_dir)
    wind_ds = concat_years("wind", args.cache_dir)

    for name, ds in [("eac4", eac4_ds), ("pblh", pblh_ds), ("wind", wind_ds)]:
        check_grid_cell(name, ds)

    om_var, total_var = find_aod_vars(eac4_ds)
    aod_om = to_series(eac4_ds, om_var, "aod_om")
    aod_total = to_series(eac4_ds, total_var, "aod_total")
    pblh = to_series(pblh_ds, find_pblh_var(pblh_ds), "pblh")
    wind10m = find_wind_speed(wind_ds)

    check_timestamps_aligned(
        {"eac4": aod_om, "pblh": pblh, "wind": wind10m, "pm25": pm25["pm25_max"]}
    )

    table = pd.concat([aod_om, aod_total, pblh, wind10m, pm25], axis=1, join="outer").sort_index()
    table.index.name = "timestamp"

    n_before = len(table)
    table = table.dropna(how="any")
    print(
        f"Join: {n_before} candidate timestamps -> {len(table)} complete rows after "
        f"dropping any row with a missing source ({n_before - len(table)} dropped)."
    )

    table = table[["aod_om", "aod_total", "pblh", "wind10m", "pm25_max", "pm25_mean"]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "csv":
        table.to_csv(args.output)
    else:
        table.to_parquet(args.output)
    print(f"Wrote {len(table)} rows to {args.output} ({table.index.min()} to {table.index.max()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
