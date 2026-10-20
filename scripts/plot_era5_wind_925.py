#!/usr/bin/env python3
"""Plot ERA5 925 hPa wind over Southeast Asia for one date.

Downloads u/v at 925 hPa (ERA5 pressure-level reanalysis, hourly) for a single
day and draws wind speed shading with arrows at 00/06/12/18 UTC. Writes the
PNG plus the downloaded NetCDF into --outdir.
"""
import argparse
import datetime as dt
import tempfile
import zipfile
from pathlib import Path

import cdsapi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

DATASET = "reanalysis-era5-pressure-levels"
LEVEL_HPA = 925
# [North, West, South, East]
AREA = [25, 90, -11, 145]
PLOT_HOURS = ["00:00", "06:00", "12:00", "18:00"]
ALL_HOURS = [f"{h:02d}:00" for h in range(24)]
ARROW_STRIDE = 6  # 0.25 deg grid -> one arrow every 1.5 deg

U_CANDIDATES = ("u", "u_component_of_wind")
V_CANDIDATES = ("v", "v_component_of_wind")


def build_request(day: dt.date) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": ["u_component_of_wind", "v_component_of_wind"],
        "pressure_level": [str(LEVEL_HPA)],
        "year": [f"{day.year}"],
        "month": [f"{day.month:02d}"],
        "day": [f"{day.day:02d}"],
        "time": ALL_HOURS,
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def unwrap_if_zip(path: Path) -> None:
    with open(path, "rb") as f:
        if f.read(2) != b"PK":
            return
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(path) as zf:
            nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
            if not nc_names:
                raise RuntimeError(f"{path} is a zip but contains no .nc file")
            zf.extract(nc_names[0], tmp)
            Path(tmp, nc_names[0]).replace(path)


def download(day: dt.date, target: Path) -> Path:
    client = cdsapi.Client()
    client.retrieve(DATASET, build_request(day)).download(str(target))
    unwrap_if_zip(target)
    return target


def pick(ds: xr.Dataset, candidates: tuple) -> xr.DataArray:
    name = next((n for n in candidates if n in ds.data_vars), None)
    if name is None:
        raise RuntimeError(f"none of {candidates} in data variables {list(ds.data_vars)}")
    return ds[name]


def load_wind(path: Path) -> tuple[xr.DataArray, xr.DataArray]:
    ds = xr.open_dataset(path)
    u = pick(ds, U_CANDIDATES)
    v = pick(ds, V_CANDIDATES)
    level_dim = next((d for d in ("pressure_level", "level", "plev") if d in u.dims), None)
    if level_dim is not None:
        u = u.sel({level_dim: LEVEL_HPA}, method="nearest")
        v = v.sel({level_dim: LEVEL_HPA}, method="nearest")
    time_dim = next(d for d in ("valid_time", "time") if d in u.dims)
    u = u.squeeze(drop=True).sortby(time_dim)
    v = v.squeeze(drop=True).sortby(time_dim)
    return u.rename({time_dim: "time"}), v.rename({time_dim: "time"})


def render(u: xr.DataArray, v: xr.DataArray, day: dt.date, out_png: Path, use_cartopy: bool) -> None:
    speed = np.hypot(u, v)
    lat = u["latitude"].values
    lon = u["longitude"].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    vmax = float(np.nanpercentile(speed.values, 99))

    kwargs = {}
    if use_cartopy:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        kwargs["subplot_kw"] = {"projection": ccrs.PlateCarree()}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True, **kwargs)
    fig.suptitle(
        f"ERA5 wind at {LEVEL_HPA} hPa over Southeast Asia, {day.isoformat()} (UTC)",
        fontsize=15,
    )
    mappable = None
    for ax, hour in zip(axes.flat, PLOT_HOURS):
        t = np.datetime64(f"{day.isoformat()}T{hour}")
        su = u.sel(time=t).values
        sv = v.sel(time=t).values
        ss = np.hypot(su, sv)
        geo = {"transform": ccrs.PlateCarree()} if use_cartopy else {}
        mappable = ax.pcolormesh(
            lon2d, lat2d, ss, shading="auto", cmap="viridis", vmin=0, vmax=vmax, **geo
        )
        ax.quiver(
            lon2d[::ARROW_STRIDE, ::ARROW_STRIDE],
            lat2d[::ARROW_STRIDE, ::ARROW_STRIDE],
            su[::ARROW_STRIDE, ::ARROW_STRIDE],
            sv[::ARROW_STRIDE, ::ARROW_STRIDE],
            color="white",
            edgecolor="black",
            linewidth=0.4,
            scale=350,
            width=0.0045,
            **geo,
        )
        if use_cartopy:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
            ax.set_extent([AREA[1], AREA[3], AREA[2], AREA[0]], crs=ccrs.PlateCarree())
        else:
            ax.set_xlim(AREA[1], AREA[3])
            ax.set_ylim(AREA[2], AREA[0])
            ax.set_xlabel("Longitude (deg E)")
            ax.set_ylabel("Latitude (deg N)")
            ax.grid(alpha=0.3)
        ax.set_title(f"{hour} UTC  ({(int(hour[:2]) + 8) % 24:02d}:00 SGT)")

    fig.colorbar(mappable, ax=axes, shrink=0.8, label="Wind speed at 925 hPa (m/s)")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD (UTC day)")
    parser.add_argument("--outdir", default="output", type=Path)
    args = parser.parse_args()

    day = dt.date.fromisoformat(args.date)
    args.outdir.mkdir(parents=True, exist_ok=True)
    nc_path = args.outdir / f"era5_pl_wind_{LEVEL_HPA}hPa_{day.isoformat()}.nc"
    png_path = args.outdir / f"era5_wind_{LEVEL_HPA}hPa_{day.isoformat()}.png"

    print(f"Requesting {DATASET} u/v at {LEVEL_HPA} hPa for {day.isoformat()}, area {AREA}")
    download(day, nc_path)
    u, v = load_wind(nc_path)

    try:
        render(u, v, day, png_path, use_cartopy=True)
        print("Rendered with cartopy coastlines.")
    except Exception as exc:  # cartopy's Natural Earth download can fail on some runners
        print(f"cartopy render failed ({exc.__class__.__name__}: {exc}); falling back to plain lat/lon.")
        render(u, v, day, png_path, use_cartopy=False)

    print(f"Wrote {png_path} and {nc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
