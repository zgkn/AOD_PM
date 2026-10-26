#!/usr/bin/env python3
"""Average ERA5 winds over a UTC date range and plot them over Southeast Asia.

Reads the per-day files written by download_era5_wind_day.py (one
era5_pl_wind_YYYY-MM-DD.nc per day, 24 hourly fields each) from --indir,
averages u/v over every hour in the inclusive range, and draws one panel per
level: speed of the mean wind (shading) with mean-wind arrows. Refuses to run
if any day in the range is missing. Writes a PNG plus the mean-wind NetCDF
into --outdir.
"""
import argparse
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from plot_era5_wind_925 import AREA, U_CANDIDATES, V_CANDIDATES, pick  # noqa: E402

ARROW_STRIDE = 6  # 0.25 deg grid -> one arrow every 1.5 deg


def load_local(indir: Path, start: dt.date, end: dt.date) -> xr.Dataset:
    """Concatenate the daily files for [start, end] along time."""
    parts, missing = [], []
    day = start
    while day <= end:
        path = indir / f"era5_pl_wind_{day.isoformat()}.nc"
        if path.exists():
            parts.append(xr.open_dataset(path).load())
        else:
            missing.append(day.isoformat())
        day += dt.timedelta(days=1)
    if missing:
        raise SystemExit(f"missing daily files in {indir}: {', '.join(missing)}")
    time_dim = next(d for d in ("valid_time", "time") if d in parts[0].dims)
    combined = xr.concat(parts, dim=time_dim) if len(parts) > 1 else parts[0]
    return combined.sortby(time_dim).rename({time_dim: "time"})


def level_dim_name(da: xr.DataArray) -> str:
    return next(d for d in ("pressure_level", "level", "plev") if d in da.dims)


def mean_wind(ds: xr.Dataset, levels: list[int]) -> xr.Dataset:
    """Time-mean u and v per level, plus the speed of the mean wind and the
    mean of the instantaneous speed (the latter shows how steady the flow is)."""
    u, v = pick(ds, U_CANDIDATES), pick(ds, V_CANDIDATES)
    dim = level_dim_name(u)
    out = xr.Dataset()
    out["u_mean"] = u.mean("time")
    out["v_mean"] = v.mean("time")
    out["speed_of_mean"] = np.hypot(out["u_mean"], out["v_mean"])
    out["mean_speed"] = np.hypot(u, v).mean("time")
    out.attrs["n_hours"] = int(u.sizes["time"])
    return out.sel({dim: levels}) if dim in out.dims else out


def render(mean: xr.Dataset, levels: list[int], start: dt.date, end: dt.date,
           n_hours: int, out_png: Path, use_cartopy: bool) -> None:
    dim = level_dim_name(mean["u_mean"])
    lat = mean["u_mean"]["latitude"].values
    lon = mean["u_mean"]["longitude"].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    vmax = float(np.nanpercentile(mean["speed_of_mean"].values, 99))

    kwargs = {}
    geo = {}
    if use_cartopy:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        kwargs["subplot_kw"] = {"projection": ccrs.PlateCarree()}
        geo = {"transform": ccrs.PlateCarree()}
    fig, axes = plt.subplots(1, len(levels), figsize=(8 * len(levels), 7.5),
                             constrained_layout=True, squeeze=False, **kwargs)
    fig.suptitle(
        f"ERA5 mean wind over Southeast Asia, {start.isoformat()} to {end.isoformat()} "
        f"(7-day mean of {n_hours} hourly fields, UTC)",
        fontsize=15,
    )
    mappable = None
    for ax, level in zip(axes[0], levels):
        u = mean["u_mean"].sel({dim: level}).values
        v = mean["v_mean"].sel({dim: level}).values
        speed = np.hypot(u, v)
        mappable = ax.pcolormesh(lon2d, lat2d, speed, shading="auto", cmap="viridis",
                                 vmin=0, vmax=vmax, **geo)
        ax.quiver(
            lon2d[::ARROW_STRIDE, ::ARROW_STRIDE], lat2d[::ARROW_STRIDE, ::ARROW_STRIDE],
            u[::ARROW_STRIDE, ::ARROW_STRIDE], v[::ARROW_STRIDE, ::ARROW_STRIDE],
            color="white", edgecolor="black", linewidth=0.4, scale=250, width=0.0045, **geo,
        )
        if use_cartopy:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
            ax.set_extent([AREA[1], AREA[3], AREA[2], AREA[0]], crs=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            gl.top_labels = gl.right_labels = False
        else:
            ax.set_xlim(AREA[1], AREA[3])
            ax.set_ylim(AREA[2], AREA[0])
            ax.set_xlabel("Longitude (deg E)")
            ax.set_ylabel("Latitude (deg N)")
            ax.grid(alpha=0.3)
        ax.set_title(f"{level} hPa")
    fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.8,
                 label="Speed of the mean wind (m/s)")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="first UTC day, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="last UTC day (inclusive), YYYY-MM-DD")
    parser.add_argument("--levels", default="925,850", help="comma-separated hPa levels")
    parser.add_argument("--indir", default="raw", type=Path, help="directory of per-day NetCDF files")
    parser.add_argument("--outdir", default="output", type=Path)
    args = parser.parse_args()

    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    if end < start:
        raise SystemExit("--end must not be before --start")
    levels = [int(p) for p in args.levels.split(",")]
    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{start.isoformat()}_{end.isoformat()}"
    png_path = args.outdir / f"era5_mean_wind_{tag}.png"
    nc_path = args.outdir / f"era5_mean_wind_{tag}.nc"

    ds = load_local(args.indir, start, end)
    n_hours = int(ds.sizes["time"])
    expected = ((end - start).days + 1) * 24
    print(f"Got {n_hours} hourly fields (expected {expected})")
    if n_hours != expected:
        raise SystemExit(f"hourly field count {n_hours} != expected {expected}; refusing to plot")

    mean = mean_wind(ds, levels)
    mean.to_netcdf(nc_path)
    try:
        render(mean, levels, start, end, n_hours, png_path, use_cartopy=True)
        print("Rendered with cartopy coastlines.")
    except Exception as exc:  # e.g. Natural Earth download blocked on the runner
        print(f"cartopy render failed ({exc.__class__.__name__}: {exc}); using plain lat/lon.")
        render(mean, levels, start, end, n_hours, png_path, use_cartopy=False)
    print(f"Wrote {png_path} and {nc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
