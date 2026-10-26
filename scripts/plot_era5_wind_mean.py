#!/usr/bin/env python3
"""Average ERA5 winds over a UTC date range and plot them over Southeast Asia.

Reads the per-day files written by download_era5_wind_arco.py (one
era5_pl_wind_YYYY-MM-DD.nc per day, 24 hourly fields each) from --indir,
averages u/v over every hour in the inclusive range, and draws one panel per
level: mean wind speed in knots (average of the hourly speeds, shading) with
streamlines of the mean wind vector. Refuses to run if any day in the range
is missing. Writes a PNG plus the mean-wind NetCDF
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

# [North, West, South, East]
AREA = [25, 90, -11, 145]
MS_TO_KNOTS = 1.0 / 0.514444
SINGAPORE = (103.82, 1.35)
STREAM_DENSITY = 4
# 5-knot bands, 0-50 kt, open-ended above
SPEED_BOUNDS = list(range(0, 55, 5))
SPEED_COLORS = ["#ffffff", "#d4ecea", "#a8c9df", "#b1d89d", "#5cc15c",
                "#efe14f", "#f4a259", "#d9534f", "#b01c2a", "#d86fb0"]

U_CANDIDATES = ("u", "u_component_of_wind")
V_CANDIDATES = ("v", "v_component_of_wind")


def pick(ds: xr.Dataset, candidates: tuple) -> xr.DataArray:
    name = next((n for n in candidates if n in ds.data_vars), None)
    if name is None:
        raise RuntimeError(f"none of {candidates} in data variables {list(ds.data_vars)}")
    return ds[name]


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
    # streamplot needs strictly increasing axes; ERA5 latitude runs north to south
    mean = mean.sortby("latitude").sortby("longitude")
    lat = mean["u_mean"]["latitude"].values
    lon = mean["u_mean"]["longitude"].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    kwargs, geo = {}, {}
    if use_cartopy:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

        kwargs["subplot_kw"] = {"projection": ccrs.PlateCarree()}
        geo = {"transform": ccrs.PlateCarree()}
    fig, axes = plt.subplots(1, len(levels), figsize=(8.5 * len(levels), 6.6), squeeze=False, **kwargs)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.2, wspace=0.12)
    fig.text(0.01, 0.955, f"ERA5 reanalysis, mean wind   {start.isoformat()} to {end.isoformat()} "
             f"(average of {n_hours} hourly fields, UTC)", fontsize=15, ha="left")

    contour = None
    for ax, level in zip(axes[0], levels):
        u = mean["u_mean"].sel({dim: level}).values
        v = mean["v_mean"].sel({dim: level}).values
        speed_kt = mean["mean_speed"].sel({dim: level}).values * MS_TO_KNOTS
        contour = ax.contourf(lon2d, lat2d, speed_kt, levels=SPEED_BOUNDS, colors=SPEED_COLORS,
                              extend="max", **geo)
        ax.streamplot(lon, lat, u, v, color="black", linewidth=0.5, density=STREAM_DENSITY,
                      arrowsize=0.8, **geo)
        ax.plot(*SINGAPORE, marker="*", color="yellow", markeredgecolor="black",
                markersize=13, linestyle="none", zorder=5, **geo)
        ax.annotate("Singapore", SINGAPORE, xytext=(4, 5), textcoords="offset points",
                    fontsize=8, fontweight="bold", zorder=6,
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1})
        if use_cartopy:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.set_extent([AREA[1], AREA[3], AREA[2], AREA[0]], crs=ccrs.PlateCarree())
            ax.set_xticks(np.arange(90, 146, 10), crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(-10, 26, 5), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
        else:
            ax.set_xlim(AREA[1], AREA[3])
            ax.set_ylim(AREA[2], AREA[0])
            ax.set_xlabel("Longitude (deg E)")
            ax.set_ylabel("Latitude (deg N)")
        ax.set_title(f"Wind at {level} hPa", loc="left", fontsize=13)
    cax = fig.add_axes([0.25, 0.1, 0.5, 0.035])
    cbar = fig.colorbar(contour, cax=cax, orientation="horizontal", ticks=SPEED_BOUNDS[1:])
    cbar.set_label("knots")
    fig.text(0.01, 0.012, "Data: ERA5 reanalysis (ECMWF/Copernicus, via ARCO-ERA5), 0.25 deg hourly. "
             "Shading: mean of the hourly wind speeds. Streamlines: mean wind vector.", fontsize=8,
             color="dimgray", ha="left")
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
