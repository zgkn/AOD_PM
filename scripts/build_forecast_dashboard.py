#!/usr/bin/env python3
"""Build the 5-day PM2.5 forecast dashboard (static HTML) from CAMS AOD_om.

Applies two models fit once (2026-10-03) on the full historical training
table (data/training_table.csv, n=34,346) -- their coefficients are
hardcoded below, so this script needs no training data and does no
fitting at runtime:

  Linear:   pm25_mean = 8.682231 + 35.967544 * aod_om   (R^2 = 0.467)
  Ordinal:  pm25_mean band ~ aod_om, proportional-odds logit
            (params = [aod_om coef, 3 threshold params], McFadden R^2 = 0.513)

The ordinal params are the exact fitted statsmodels OrderedModel.params
array; they are NOT raw cutpoints you can plug into a hand-written formula
(OrderedModel's internal threshold parameterization isn't a simple cascade
of cutpoints), so predictions are reproduced by calling OrderedModel's own
.predict(params=..., exog=...) on a freshly-constructed (never-fitted)
model built from a synthetic 4-row dummy -- this exactly reproduces the
real fitted model's output without needing any real data.

Bands: https://www.haze.gov.sg/resources/1-hr-pm2.5-readings

AOD_om input: the max over a 3x3 degree box centered on Singapore (set by
download_cams_forecast.py), not the single Singapore grid cell -- a
deliberate choice to account for forecast plume-position uncertainty by
taking a regional worst case instead of betting on one exact grid cell.
Note this does NOT match how the hardcoded coefficients above were fit:
the training table's aod_om is a single-point reanalysis time series, not
a regional max, so feeding a regional max into those coefficients is
intentionally conservative/worst-case, not a statistically calibrated
forecast for the single Singapore point.
"""
import argparse
import base64
import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.miscmodels.ordinal_model import OrderedModel

# --- Hardcoded, fit once on the full training table (see module docstring) ---
LINEAR_INTERCEPT = 8.682231165899342
LINEAR_SLOPE = 35.96754405996798
ORDINAL_PARAMS = np.array([4.108760, 6.495276, 1.610290, 0.306059])

BAND_LABELS = ["Normal", "Elevated", "High", "Very High"]
BAND_EDGES = [0, 55, 150, 250, float("inf")]
# Status palette (good/warning/serious/critical) from the dataviz reference
# palette -- the intended use for a severity/state signal, not series identity.
BAND_COLORS = {
    "Normal": "#0ca30c",
    "Elevated": "#fab219",
    "High": "#ec835a",
    "Very High": "#d03b3b",
}

AOD_COLOR = "#2a78d6"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
MUTED = "#898781"
INK = "#0b0b0b"

AOD_VAR_CANDIDATES = ("omaod550", "organic_matter_aerosol_optical_depth_550nm")


def load_forecast(path: Path) -> pd.DataFrame:
    ds = xr.open_dataset(path)

    var = next((n for n in AOD_VAR_CANDIDATES if n in ds.data_vars), None)
    if var is None:
        raise SystemExit(f"Could not find AOD_om variable among: {list(ds.data_vars)}")

    lat_dim = next((d for d in ("latitude", "lat") if d in ds[var].dims), None)
    lon_dim = next((d for d in ("longitude", "lon") if d in ds[var].dims), None)
    spatial_dims = [d for d in (lat_dim, lon_dim) if d is not None]

    data = ds[var]
    if spatial_dims:
        n_cells = 1
        for d in spatial_dims:
            n_cells *= data.sizes[d]
        lat_lo, lat_hi = float(ds[lat_dim].min()), float(ds[lat_dim].max())
        lon_lo, lon_hi = float(ds[lon_dim].min()), float(ds[lon_dim].max())
        print(
            f"Taking max AOD_om over {n_cells} grid cells "
            f"(lat {lat_lo:.1f} to {lat_hi:.1f}, lon {lon_lo:.1f} to {lon_hi:.1f}) "
            "per timestep -- a regional max to account for forecast "
            "plume-position uncertainty, not the single Singapore grid cell."
        )
        data = data.max(dim=spatial_dims)

    # Forecast-type CAMS output carries a reference time + a leadtime
    # timedelta (named "step" or "forecast_period" depending on how it was
    # packaged) rather than one flat "valid_time" coordinate like the
    # reanalysis datasets do; handle both shapes rather than assume one.
    # "valid_time" itself, when present, is 2-D (forecast_reference_time x
    # leadtime) even for a single reference time, so flatten explicitly --
    # pd.to_datetime() on a 2-D array does not do this for you.
    step_dim = next((d for d in ("step", "forecast_period") if d in ds.coords), None)
    if "valid_time" in ds.coords:
        valid_time = pd.to_datetime(np.asarray(ds["valid_time"].values).reshape(-1))
    elif "time" in ds.coords and np.issubdtype(ds["time"].dtype, np.datetime64):
        valid_time = pd.to_datetime(np.asarray(ds["time"].values).reshape(-1))
    elif "forecast_reference_time" in ds.coords and step_dim is not None:
        ref = pd.to_datetime(np.asarray(ds["forecast_reference_time"].values).item())
        valid_time = ref + pd.to_timedelta(np.asarray(ds[step_dim].values).reshape(-1))
    else:
        raise SystemExit(
            f"Could not determine forecast valid time from coords: {list(ds.coords)}"
        )

    values = np.asarray(data.values).reshape(-1).astype(float)
    if len(values) != len(valid_time):
        raise SystemExit(
            f"AOD_om value count ({len(values)}) does not match time count "
            f"({len(valid_time)}) -- refusing to silently misalign them."
        )

    df = pd.DataFrame({"valid_time": pd.DatetimeIndex(valid_time), "aod_om": values})
    return df.sort_values("valid_time").reset_index(drop=True)


def predict_ordinal_probs(aod_om: pd.Series) -> pd.DataFrame:
    dummy_endog = pd.Series(
        pd.Categorical(BAND_LABELS, categories=BAND_LABELS, ordered=True), name="pm25_category"
    )
    dummy_exog = pd.DataFrame({"aod_om": [0.0, 1.0, 2.0, 3.0]})
    model = OrderedModel(dummy_endog, dummy_exog, distr="logit")
    raw = model.predict(params=ORDINAL_PARAMS, exog=pd.DataFrame({"aod_om": aod_om}))
    return pd.DataFrame(np.asarray(raw), columns=BAND_LABELS, index=aod_om.index)


def compute_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pm25_linear"] = LINEAR_INTERCEPT + LINEAR_SLOPE * df["aod_om"]
    probs = predict_ordinal_probs(df["aod_om"])
    df = pd.concat([df, probs], axis=1)
    df["pm25_band"] = probs.idxmax(axis=1)
    return df


def render_chart(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10.8), facecolor=SURFACE, sharex=True)

    def style(ax):
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    x = df["valid_time"]

    ax = axes[0]
    style(ax)
    ax.plot(x, df["aod_om"], color=AOD_COLOR, linewidth=2, marker="o", markersize=3, zorder=2)
    ax.set_ylabel("Organic matter AOD", color=INK, fontsize=9)
    ax.set_title(
        "CAMS forecast: max organic matter AOD over Singapore region, next 5 days",
        color=INK, fontsize=11, loc="left",
    )

    ax = axes[1]
    style(ax)
    for edge, label in zip(BAND_EDGES[1:-1], BAND_LABELS[1:]):
        ax.axhline(edge, color=BAND_COLORS[label], linewidth=1, linestyle="--", alpha=0.6, zorder=1)
    ax.plot(x, df["pm25_linear"], color=INK, linewidth=2, marker="o", markersize=3, zorder=2)
    ax.set_ylabel("Predicted PM2.5 (µg/m³)", color=INK, fontsize=9)
    ax.set_title(
        "Linear regression estimate (dashed lines = haze.gov.sg band boundaries)",
        color=INK, fontsize=11, loc="left",
    )

    ax = axes[2]
    style(ax)
    ax.stackplot(
        x, [df[label] for label in BAND_LABELS],
        colors=[BAND_COLORS[label] for label in BAND_LABELS],
        labels=BAND_LABELS, alpha=0.85, zorder=2,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted band probability", color=INK, fontsize=9)
    ax.set_xlabel("Forecast valid time (UTC)", color=INK, fontsize=9)
    # pad pushes the title up so the legend (anchored just above the axes)
    # sits between the title and the plot, not on top of either.
    ax.set_title(
        "Ordinal logistic regression: predicted PM2.5 band probability",
        color=INK, fontsize=11, loc="left", pad=28,
    )
    legend = ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.1), ncol=4,
        frameon=False, fontsize=9, mode="expand",
    )
    for text in legend.get_texts():
        text.set_color(INK)

    # Hour-level ticks (default date-only locator is too coarse over a
    # 5-day, 3-hourly series); every 12h keeps labels readable at this size.
    locator = mdates.HourLocator(interval=12)
    formatter = mdates.DateFormatter("%m/%d %H:%M")
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_html(df: pd.DataFrame, chart_b64: str, generated_at: datetime) -> str:
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Singapore PM2.5 5-day forecast</title>
<style>
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: {SURFACE}; color: {INK};
          max-width: 900px; margin: 0 auto; padding: 24px 16px 48px; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
  .meta {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 24px; }}
  img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid {GRIDLINE}; }}
  footer {{ color: {MUTED}; font-size: 0.8rem; margin-top: 24px; }}
  a {{ color: {AOD_COLOR}; }}
</style>
</head>
<body>
  <h1>Singapore PM2.5 -- 5-day forecast</h1>
  <div class="meta">
    AOD_om: max over a 3&deg;&times;3&deg; box centered on Singapore (1.5&deg;N, 103.5&deg;E), to account for forecast plume-position uncertainty
    &middot; Generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')}
  </div>

  <img src="data:image/png;base64,{chart_b64}" alt="5-day AOD and PM2.5 forecast chart">

  <footer>
    Models fit once on 34,346 historical observations (2014-03-31 to 2025-12-31) from EAC4/ERA5
    reanalysis and NEA PM2.5 station data; coefficients are hardcoded, not refit per run.
    Linear: R&sup2;=0.467. Ordinal logistic: McFadden pseudo-R&sup2;=0.513.
    Source data and code: <a href="https://github.com/zgkn/AOD_PM">github.com/zgkn/AOD_PM</a>.
  </footer>
</body>
</html>
"""
    return html


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-nc", default=Path("output/cams_forecast_aod_om.nc"), type=Path)
    parser.add_argument("--output", default=Path("site/index.html"), type=Path)
    args = parser.parse_args()

    df = load_forecast(args.forecast_nc)
    print(f"Loaded {len(df)} forecast timesteps: {df['valid_time'].min()} to {df['valid_time'].max()}")

    df = compute_predictions(df)
    print(df[["valid_time", "aod_om", "pm25_linear", "pm25_band"]].to_string(index=False))

    chart_b64 = render_chart(df)
    html = render_html(df, chart_b64, datetime.now(timezone.utc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
