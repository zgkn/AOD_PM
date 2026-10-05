#!/usr/bin/env python3
"""Build the 5-day PM2.5 forecast dashboard (static HTML) from CAMS AOD_om.

Applies four models fit once (2026-10-03/2026-10-05) on the full
historical training table (data/training_table.csv, n=34,346) -- their
coefficients are hardcoded below, so this script needs no training data
and does no fitting at runtime:

  Linear (pm25_mean):   pm25_mean = 8.682231 + 35.967544 * aod_om   (R^2 = 0.467)
  Ordinal (pm25_mean):  pm25_mean band ~ aod_om, proportional-odds logit
                         (params = [aod_om coef, 3 threshold params], McFadden R^2 = 0.513)
  Linear (pm25_max):    pm25_max = 13.656895 + 42.929792 * aod_om   (R^2 = 0.410)
  Ordinal (pm25_max):   pm25_max band ~ aod_om, proportional-odds logit
                         (McFadden R^2 = 0.416)

The dashboard forecasts BOTH targets over the same 5-day horizon,
overlaid in the same panels rather than as separate tracks: pm25_mean
as a solid line, pm25_max as a dashed line of the same color, in both
the linear-estimate panel and each of the 4 per-band probability
panels -- since max >= mean always, this reads as "typical value, with
a dashed ceiling for how bad it could get" at a glance, which is the
whole point of forecasting both.

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

Chart is rendered client-side with Plotly.js (CDN), not a static image:
pan/zoom (box-drag or scroll-wheel), hover tooltips with exact values, and
a synced x-axis across every panel so zooming one zooms all of them.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.miscmodels.ordinal_model import OrderedModel

# --- Hardcoded, fit once on the full training table (see module docstring) ---
LINEAR_INTERCEPT = 8.682231165899342
LINEAR_SLOPE = 35.96754405996798
LINEAR_R2 = 0.467
ORDINAL_PARAMS = np.array([4.108760, 6.495276, 1.610290, 0.306059])
ORDINAL_MCFADDEN_R2 = 0.513

PM25MAX_LINEAR_INTERCEPT = 13.6568951089901
PM25MAX_LINEAR_SLOPE = 42.92979153232956
PM25MAX_LINEAR_R2 = 0.410
PM25MAX_ORDINAL_PARAMS = np.array([4.139879521805435, 5.608515888012341, 1.533790149659414, 0.5983033258719117])
PM25MAX_ORDINAL_MCFADDEN_R2 = 0.416

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

# Historical linear-regression panel only: PM2.5 mean vs max get distinct
# colors (not just solid/dashed), since that panel overlays two full scatter
# clouds where line style alone is hard to trace back to a cloud of points.
PM25_MEAN_COLOR = "#2a78d6"
PM25_MAX_COLOR = "#d03b3b"

AOD_VAR_CANDIDATES = ("omaod550", "organic_matter_aerosol_optical_depth_550nm")

DATE_TICKFORMAT = "%Y-%m-%d %H:%M"  # ISO-style; applies to axis ticks and hover

# Plotly's "matches" only syncs an axis's zoom/pan range with the master
# axis, not its display properties -- every x-axis needs its own copy of
# these, or axes other than the master fall back to Plotly's auto date
# format instead of DATE_TICKFORMAT.
BASE_X_AXIS = {
    "type": "date", "tickformat": DATE_TICKFORMAT, "gridcolor": GRIDLINE, "linecolor": GRIDLINE,
    "tickfont": {"size": 10, "color": MUTED}, "showline": True, "zeroline": False,
}


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


def predict_ordinal_probs(aod_om: pd.Series, params: np.ndarray = ORDINAL_PARAMS) -> pd.DataFrame:
    dummy_endog = pd.Series(
        pd.Categorical(BAND_LABELS, categories=BAND_LABELS, ordered=True), name="pm25_category"
    )
    dummy_exog = pd.DataFrame({"aod_om": [0.0, 1.0, 2.0, 3.0]})
    model = OrderedModel(dummy_endog, dummy_exog, distr="logit")
    raw = model.predict(params=params, exog=pd.DataFrame({"aod_om": aod_om}))
    return pd.DataFrame(np.asarray(raw), columns=BAND_LABELS, index=aod_om.index)


def compute_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["pm25_linear"] = LINEAR_INTERCEPT + LINEAR_SLOPE * df["aod_om"]
    probs = predict_ordinal_probs(df["aod_om"])
    df = pd.concat([df, probs], axis=1)
    df["pm25_band"] = probs.idxmax(axis=1)

    df["pm25_max_linear"] = PM25MAX_LINEAR_INTERCEPT + PM25MAX_LINEAR_SLOPE * df["aod_om"]
    max_probs = predict_ordinal_probs(df["aod_om"], PM25MAX_ORDINAL_PARAMS)
    max_probs.columns = [f"max_{c}" for c in max_probs.columns]
    df = pd.concat([df, max_probs], axis=1)
    df["pm25_max_band"] = max_probs.idxmax(axis=1).str.replace("max_", "", regex=False)

    return df


def load_training_sample(path: Path, n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """A random sample of (aod_om, pm25_mean, pm25_max) from the full
    training table, for the historical-regression scatter panels --
    plotting all 34,346 rows would bloat the page and slow down
    rendering for no visible gain in density."""
    full = pd.read_csv(path, usecols=["aod_om", "pm25_mean", "pm25_max"]).dropna()
    sample = full.sample(n=min(n, len(full)), random_state=seed)
    return sample.reset_index(drop=True)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _row_domains(weights: list[float], gap: float | list[float] = 0.03) -> list[tuple[float, float]]:
    """Top-to-bottom y-domains for stacked subplots, like a matplotlib
    gridspec height_ratios layout. `gap` is either one fraction reused
    between every pair of rows, or a list of len(weights)-1 fractions for
    when one boundary (e.g. between two unrelated axis groups, each with
    its own tick labels/title crowding that gap) needs more room than the
    rest."""
    n = len(weights)
    gaps = [gap] * (n - 1) if isinstance(gap, (int, float)) else list(gap)
    avail = 1 - sum(gaps)
    total = sum(weights)
    domains = []
    y_top = 1.0
    for i, w in enumerate(weights):
        h = avail * w / total
        y_bottom = y_top - h
        domains.append((round(y_bottom, 6), round(y_top, 6)))
        y_top = y_bottom - (gaps[i] if i < len(gaps) else 0)
    return domains


def _linear_panel(
    data: list, layout: dict, aid: str, domain: tuple, times: list,
    mean_values: pd.Series, max_values: pd.Series, panel_title: str,
) -> None:
    y0, y1 = domain
    layout[f"xaxis{aid}"] = {**BASE_X_AXIS, "matches": "x", "domain": [0, 1], "anchor": f"y{aid}", "showticklabels": False}
    layout[f"yaxis{aid}"] = {
        "domain": [y0, y1], "anchor": f"x{aid}", "rangemode": "tozero",
        "title": {"text": "Predicted PM2.5 (µg/m³)", "font": {"size": 11, "color": INK}},
        "gridcolor": GRIDLINE, "tickfont": {"size": 10, "color": MUTED},
    }
    data.append({
        "type": "scatter", "mode": "lines+markers", "x": times, "y": mean_values.round(2).tolist(),
        "line": {"color": INK, "width": 2}, "marker": {"size": 4, "color": INK},
        "xaxis": f"x{aid}", "yaxis": f"y{aid}",
        "hovertemplate": f"%{{x|{DATE_TICKFORMAT}}}<br>PM2.5 mean: %{{y:.1f}} µg/m³<extra></extra>",
    })
    data.append({
        "type": "scatter", "mode": "lines", "x": times, "y": max_values.round(2).tolist(),
        "line": {"color": INK, "width": 1.5, "dash": "dash"},
        "xaxis": f"x{aid}", "yaxis": f"y{aid}",
        "hovertemplate": f"%{{x|{DATE_TICKFORMAT}}}<br>PM2.5 max: %{{y:.1f}} µg/m³<extra></extra>",
    })
    for edge, label in zip(BAND_EDGES[1:-1], BAND_LABELS[1:]):
        layout["shapes"].append({
            "type": "line", "xref": f"x{aid} domain", "yref": f"y{aid}",
            "x0": 0, "x1": 1, "y0": edge, "y1": edge,
            "line": {"color": BAND_COLORS[label], "width": 1, "dash": "dash"}, "opacity": 0.6,
        })
    layout["annotations"].append({
        "text": "— PM2.5 mean", "xref": f"x{aid} domain", "yref": f"y{aid} domain",
        "x": 0.01, "y": 0.97, "xanchor": "left", "yanchor": "top", "showarrow": False,
        "font": {"size": 10, "color": INK, "family": "system-ui, sans-serif"},
    })
    layout["annotations"].append({
        "text": "- - PM2.5 max", "xref": f"x{aid} domain", "yref": f"y{aid} domain",
        "x": 0.01, "y": 0.88, "xanchor": "left", "yanchor": "top", "showarrow": False,
        "font": {"size": 10, "color": INK, "family": "system-ui, sans-serif"},
    })
    layout["annotations"].append({
        "text": panel_title, "xref": "paper", "yref": "paper", "x": 0, "y": y1 + 0.022,
        "xanchor": "left", "yanchor": "bottom", "showarrow": False,
        "font": {"size": 13, "color": INK},
    })


def _band_panels(
    data: list, layout: dict, axis_ids: list, domains: list, times: list, df: pd.DataFrame,
    group_title: str,
) -> None:
    for i, label in enumerate(BAND_LABELS):
        aid = axis_ids[i]
        y0, y1 = domains[i]
        color = BAND_COLORS[label]
        is_last_panel = i == len(BAND_LABELS) - 1
        layout[f"xaxis{aid}"] = {
            **BASE_X_AXIS, "matches": "x", "domain": [0, 1], "anchor": f"y{aid}", "showticklabels": is_last_panel,
            **({"title": {"text": "Forecast valid time (UTC)", "font": {"size": 11, "color": INK}}} if is_last_panel else {}),
        }
        layout[f"yaxis{aid}"] = {
            "domain": [y0, y1], "anchor": f"x{aid}", "rangemode": "tozero",
            "title": {"text": "Prob.", "font": {"size": 9, "color": INK}},
            "gridcolor": GRIDLINE, "tickfont": {"size": 9, "color": MUTED},
        }
        # Mean: solid line, filled. Max: dashed line, no fill (filling both
        # would muddy the overlap) -- same color per band, same convention
        # as the linear panel above.
        data.append({
            "type": "scatter", "mode": "lines", "x": times, "y": df[label].round(4).tolist(),
            "line": {"color": color, "width": 1.5}, "fill": "tozeroy", "fillcolor": _hex_to_rgba(color, 0.3),
            "xaxis": f"x{aid}", "yaxis": f"y{aid}",
            "hovertemplate": f"%{{x|{DATE_TICKFORMAT}}}<br>{label} (mean): %{{y:.1%}}<extra></extra>",
        })
        data.append({
            "type": "scatter", "mode": "lines", "x": times, "y": df[f"max_{label}"].round(4).tolist(),
            "line": {"color": color, "width": 1.5, "dash": "dash"},
            "xaxis": f"x{aid}", "yaxis": f"y{aid}",
            "hovertemplate": f"%{{x|{DATE_TICKFORMAT}}}<br>{label} (max): %{{y:.1%}}<extra></extra>",
        })
        layout["annotations"].append({
            "text": label, "xref": f"x{aid} domain", "yref": f"y{aid} domain",
            "x": 0.01, "y": 0.85, "xanchor": "left", "yanchor": "top", "showarrow": False,
            "font": {"size": 11, "color": color, "family": "system-ui, sans-serif"},
        })
        if i == 0:
            layout["annotations"].append({
                "text": group_title, "xref": "paper", "yref": "paper", "x": 0, "y": y1 + 0.022,
                "xanchor": "left", "yanchor": "bottom", "showarrow": False,
                "font": {"size": 13, "color": INK},
            })


def _regression_scatter_panel(
    data: list, layout: dict, aid: str, domain: tuple, aod_axis: dict,
    training_sample: pd.DataFrame, x_grid: np.ndarray,
) -> None:
    """Historical pm25 ~ aod_om scatter (both targets) + both fit lines,
    colored by target (blue = mean, red = max) rather than the solid/dashed
    convention used elsewhere -- with two full scatter clouds overlaid here,
    line style alone doesn't carry back to which cloud a point belongs to."""
    y0, y1 = domain
    layout[f"xaxis{aid}"] = {**aod_axis, "domain": [0, 1], "anchor": f"y{aid}", "showticklabels": False}
    layout[f"yaxis{aid}"] = {
        "domain": [y0, y1], "anchor": f"x{aid}", "rangemode": "tozero",
        "title": {"text": "Historical PM2.5 (µg/m³)", "font": {"size": 11, "color": INK}},
        "gridcolor": GRIDLINE, "tickfont": {"size": 10, "color": MUTED},
    }
    data.append({
        "type": "scatter", "mode": "markers", "x": training_sample["aod_om"].round(4).tolist(),
        "y": training_sample["pm25_mean"].round(2).tolist(),
        "marker": {"size": 4, "color": _hex_to_rgba(PM25_MEAN_COLOR, 0.35)},
        "xaxis": f"x{aid}", "yaxis": f"y{aid}",
        "hovertemplate": "Historical<br>AOD_om: %{x:.3f}<br>PM2.5 mean: %{y:.1f} µg/m³<extra></extra>",
    })
    data.append({
        "type": "scatter", "mode": "markers", "x": training_sample["aod_om"].round(4).tolist(),
        "y": training_sample["pm25_max"].round(2).tolist(),
        "marker": {"size": 4, "color": _hex_to_rgba(PM25_MAX_COLOR, 0.35)},
        "xaxis": f"x{aid}", "yaxis": f"y{aid}",
        "hovertemplate": "Historical<br>AOD_om: %{x:.3f}<br>PM2.5 max: %{y:.1f} µg/m³<extra></extra>",
    })
    data.append({
        "type": "scatter", "mode": "lines", "x": x_grid.round(4).tolist(),
        "y": (LINEAR_INTERCEPT + LINEAR_SLOPE * x_grid).round(2).tolist(),
        "line": {"color": PM25_MEAN_COLOR, "width": 2.5}, "xaxis": f"x{aid}", "yaxis": f"y{aid}",
        "hovertemplate": "Linear fit<br>AOD_om: %{x:.3f}<br>PM2.5 mean: %{y:.1f} µg/m³<extra></extra>",
    })
    data.append({
        "type": "scatter", "mode": "lines", "x": x_grid.round(4).tolist(),
        "y": (PM25MAX_LINEAR_INTERCEPT + PM25MAX_LINEAR_SLOPE * x_grid).round(2).tolist(),
        "line": {"color": PM25_MAX_COLOR, "width": 2.5}, "xaxis": f"x{aid}", "yaxis": f"y{aid}",
        "hovertemplate": "Linear fit<br>AOD_om: %{x:.3f}<br>PM2.5 max: %{y:.1f} µg/m³<extra></extra>",
    })
    layout["annotations"].append({
        "text": "— PM2.5 mean fit", "xref": f"x{aid} domain", "yref": f"y{aid} domain",
        "x": 0.01, "y": 0.97, "xanchor": "left", "yanchor": "top", "showarrow": False,
        "font": {"size": 10, "color": PM25_MEAN_COLOR, "family": "system-ui, sans-serif"},
    })
    layout["annotations"].append({
        "text": "— PM2.5 max fit", "xref": f"x{aid} domain", "yref": f"y{aid} domain",
        "x": 0.01, "y": 0.88, "xanchor": "left", "yanchor": "top", "showarrow": False,
        "font": {"size": 10, "color": PM25_MAX_COLOR, "family": "system-ui, sans-serif"},
    })
    layout["annotations"].append({
        "text": "Linear regression: historical PM2.5 vs AOD_om (points) with the fitted line",
        "xref": "paper", "yref": "paper", "x": 0, "y": y1 + 0.022,
        "xanchor": "left", "yanchor": "bottom", "showarrow": False,
        "font": {"size": 13, "color": INK},
    })


def _regression_curve_panel(
    data: list, layout: dict, aid: str, master_aid: str, domain: tuple, aod_axis: dict, x_grid: np.ndarray,
) -> None:
    """Ordinal logistic band-probability curves vs aod_om, both targets
    overlaid (mean solid, max dashed) per band color."""
    y0, y1 = domain
    layout[f"xaxis{aid}"] = {
        **aod_axis, "matches": f"x{master_aid}", "domain": [0, 1], "anchor": f"y{aid}", "showticklabels": True,
        "title": {"text": "Organic matter AOD", "font": {"size": 11, "color": INK}},
    }
    layout[f"yaxis{aid}"] = {
        "domain": [y0, y1], "anchor": f"x{aid}", "range": [0, 1],
        "title": {"text": "Predicted probability", "font": {"size": 10, "color": INK}},
        "gridcolor": GRIDLINE, "tickfont": {"size": 9, "color": MUTED},
    }
    mean_probs = predict_ordinal_probs(pd.Series(x_grid))
    max_probs = predict_ordinal_probs(pd.Series(x_grid), PM25MAX_ORDINAL_PARAMS)
    for label in BAND_LABELS:
        color = BAND_COLORS[label]
        data.append({
            "type": "scatter", "mode": "lines", "x": x_grid.round(4).tolist(), "y": mean_probs[label].round(4).tolist(),
            "line": {"color": color, "width": 2},
            "xaxis": f"x{aid}", "yaxis": f"y{aid}",
            "hovertemplate": f"AOD_om: %{{x:.3f}}<br>{label} (mean): %{{y:.1%}}<extra></extra>",
        })
        data.append({
            "type": "scatter", "mode": "lines", "x": x_grid.round(4).tolist(), "y": max_probs[label].round(4).tolist(),
            "line": {"color": color, "width": 2, "dash": "dash"},
            "xaxis": f"x{aid}", "yaxis": f"y{aid}",
            "hovertemplate": f"AOD_om: %{{x:.3f}}<br>{label} (max): %{{y:.1%}}<extra></extra>",
        })
    layout["annotations"].append({
        "text": "Ordinal logistic: band probability vs AOD_om (solid = PM2.5 mean, dashed = PM2.5 max)",
        "xref": "paper", "yref": "paper", "x": 0, "y": y1 + 0.022,
        "xanchor": "left", "yanchor": "bottom", "showarrow": False,
        "font": {"size": 13, "color": INK},
    })


def build_figure(df: pd.DataFrame, training_sample: pd.DataFrame) -> dict:
    """Plotly.js figure spec (data + layout) for the 8-panel dashboard:
    6 forecast panels (AOD, a linear-estimate panel, and 4 per-band
    probability panels -- each of the latter 5 overlays pm25_mean/solid
    and pm25_max/dashed in the same panel) sharing one synced time
    x-axis, plus 2 regression-result panels (historical scatter + fit
    line -- mean/max distinguished by color, blue/red, rather than line
    style, since two full scatter clouds overlap there -- and ordinal
    logistic probability curves, mean solid/max dashed per band color)
    sharing their own synced, independent aod_om x-axis. Every y-axis is
    pinned to start at 0
    (rangemode='tozero' for the forecast panels; [0,1] fixed for the
    probability panels)."""
    times = [t.isoformat() for t in df["valid_time"]]
    # Extra-wide gap between panel 6 (end of the time-axis group, which
    # carries its own rotated date tick labels + x-axis title) and panel 7
    # (start of the aod_om-axis group, which has its own title) -- the
    # standard gap is too tight for both of those to fit without colliding.
    weights = [3, 3, 1, 1, 1, 1, 3, 3]
    gaps = [0.03] * 5 + [0.07] + [0.03]
    domains = _row_domains(weights, gap=gaps)
    axis_ids = [""] + [str(i) for i in range(2, len(weights) + 1)]

    layout = {
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {"color": INK, "family": "system-ui, -apple-system, Segoe UI, sans-serif"},
        "margin": {"l": 60, "r": 20, "t": 70, "b": 60},
        "showlegend": False,
        "dragmode": "zoom",
        "hovermode": "closest",
        "annotations": [],
        "shapes": [],
    }
    data = []

    # Panel 1: AOD_om
    aid = axis_ids[0]
    y0, y1 = domains[0]
    layout[f"xaxis{aid}"] = {
        **BASE_X_AXIS,
        "domain": [0, 1], "anchor": f"y{aid}", "showticklabels": False,
    }
    layout[f"yaxis{aid}"] = {
        "domain": [y0, y1], "anchor": f"x{aid}", "rangemode": "tozero",
        "title": {"text": "Organic matter AOD", "font": {"size": 11, "color": INK}},
        "gridcolor": GRIDLINE, "tickfont": {"size": 10, "color": MUTED},
    }
    data.append({
        "type": "scatter", "mode": "lines+markers", "x": times, "y": df["aod_om"].round(4).tolist(),
        "line": {"color": AOD_COLOR, "width": 2}, "marker": {"size": 4, "color": AOD_COLOR},
        "xaxis": f"x{aid}", "yaxis": f"y{aid}",
        "hovertemplate": f"%{{x|{DATE_TICKFORMAT}}}<br>AOD_om: %{{y:.3f}}<extra></extra>",
    })
    layout["annotations"].append({
        "text": "CAMS forecast: max organic matter AOD over Singapore region, next 5 days",
        "xref": "paper", "yref": "paper", "x": 0, "y": y1 + 0.022,
        "xanchor": "left", "yanchor": "bottom", "showarrow": False,
        "font": {"size": 13, "color": INK},
    })

    # Panel 2: linear regression estimate, PM2.5 mean (solid) + max (dashed)
    _linear_panel(
        data, layout, axis_ids[1], domains[1], times, df["pm25_linear"], df["pm25_max_linear"],
        "Linear regression estimate (dashed lines = haze.gov.sg band boundaries)",
    )

    # Panels 3-6: per-band probability, PM2.5 mean (solid/filled) + max (dashed)
    _band_panels(
        data, layout, axis_ids[2:6], domains[2:6], times, df,
        group_title="Ordinal logistic regression: predicted PM2.5 band probability (per-band detail)",
    )

    # Panels 7-8: regression-result panels (historical scatter+fit line,
    # ordinal probability curves), sharing their own synced aod_om x-axis --
    # independent of the time axis used by panels 1-6. The grid spans
    # whichever is wider, historical aod_om or this forecast's aod_om, so the
    # fit lines/curves cover the forecast's actual range too.
    x_max = float(max(training_sample["aod_om"].max(), df["aod_om"].max())) * 1.05
    x_grid = np.linspace(0, x_max, 200)
    aod_axis = {
        "gridcolor": GRIDLINE, "linecolor": GRIDLINE, "tickfont": {"size": 10, "color": MUTED},
        "showline": True, "zeroline": False,
    }
    master_aid = axis_ids[6]
    _regression_scatter_panel(data, layout, master_aid, domains[6], aod_axis, training_sample, x_grid)
    _regression_curve_panel(data, layout, axis_ids[7], master_aid, domains[7], aod_axis, x_grid)

    return {"data": data, "layout": layout}


def render_model_details() -> str:
    """A table of the actual fitted equations/parameters behind the chart,
    not just a one-line R^2 summary -- the ordinal rows report statsmodels'
    raw OrderedModel.params (aod_om coefficient + 3 threshold params), not
    literal PM2.5 cutpoints; see module docstring for why those can't be
    hand-derived into a formula the way the linear row's can."""
    rows = [
        (
            "PM2.5 mean", "Linear (OLS)",
            f"pm25_mean = {LINEAR_INTERCEPT:.4f} + {LINEAR_SLOPE:.4f} &times; aod_om",
            f"R&sup2; = {LINEAR_R2:.3f}",
        ),
        (
            "PM2.5 mean", "Ordinal logistic",
            "aod_om coef = {:.4f}; thresholds = {:.4f}, {:.4f}, {:.4f}".format(*ORDINAL_PARAMS),
            f"McFadden pseudo-R&sup2; = {ORDINAL_MCFADDEN_R2:.3f}",
        ),
        (
            "PM2.5 max", "Linear (OLS)",
            f"pm25_max = {PM25MAX_LINEAR_INTERCEPT:.4f} + {PM25MAX_LINEAR_SLOPE:.4f} &times; aod_om",
            f"R&sup2; = {PM25MAX_LINEAR_R2:.3f}",
        ),
        (
            "PM2.5 max", "Ordinal logistic",
            "aod_om coef = {:.4f}; thresholds = {:.4f}, {:.4f}, {:.4f}".format(*PM25MAX_ORDINAL_PARAMS),
            f"McFadden pseudo-R&sup2; = {PM25MAX_ORDINAL_MCFADDEN_R2:.3f}",
        ),
    ]
    body_rows = "\n".join(
        f"    <tr><td>{target}</td><td>{model}</td><td><code>{form}</code></td><td>{fit}</td></tr>"
        for target, model, form, fit in rows
    )
    return f"""  <h2>Regression model details</h2>
  <table class="models">
    <thead><tr><th>Target</th><th>Model</th><th>Fitted form</th><th>Fit quality</th></tr></thead>
    <tbody>
{body_rows}
    </tbody>
  </table>
  <p class="note">
    Both targets regressed on aod_om alone. The ordinal logistic rows are statsmodels'
    raw <code>OrderedModel.params</code> (an aod_om coefficient plus 3 internal threshold
    parameters) -- not literal PM2.5 cutpoints, and not safe to hand-derive a formula from;
    the dashboard's predictions call the fitted model's own <code>.predict()</code> instead.
    All four models were fit once (2026-10-03/2026-10-05) on the full historical training
    table (n=34,346, 2014-03-31 to 2025-12-31) and are hardcoded here, not refit per run.
  </p>
"""


def render_html(df: pd.DataFrame, figure: dict, generated_at: datetime) -> str:
    figure_json = json.dumps(figure)
    config_json = json.dumps({
        "responsive": True, "scrollZoom": True, "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    })
    model_details = render_model_details()
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Singapore PM2.5 5-day forecast</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2/plotly.min.js"></script>
<style>
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: {SURFACE}; color: {INK};
          max-width: 1000px; margin: 0 auto; padding: 24px 16px 48px; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.1rem; margin: 32px 0 12px; }}
  .meta {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 24px; }}
  .hint {{ color: {MUTED}; font-size: 0.8rem; margin-bottom: 12px; }}
  #chart {{ width: 100%; height: 1550px; border-radius: 8px; border: 1px solid {GRIDLINE}; background: {SURFACE}; }}
  table.models {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; background: #fff;
                  border: 1px solid {GRIDLINE}; border-radius: 8px; overflow: hidden; }}
  table.models th, table.models td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid {GRIDLINE}; }}
  table.models th {{ color: {MUTED}; font-weight: 600; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.03em; }}
  table.models tr:last-child td {{ border-bottom: none; }}
  table.models code {{ font-size: 0.82rem; }}
  p.note {{ color: {MUTED}; font-size: 0.8rem; margin-top: 10px; }}
  code {{ background: {SURFACE}; border: 1px solid {GRIDLINE}; border-radius: 4px; padding: 1px 5px; }}
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
  <div class="hint">Drag to zoom, scroll to zoom, double-click to reset, hover for exact values.</div>

  <div id="chart"></div>
  <script>
    const FORECAST_FIGURE = {figure_json};
    Plotly.newPlot("chart", FORECAST_FIGURE.data, FORECAST_FIGURE.layout, {config_json});
  </script>

{model_details}
  <footer>
    Models fit once on 34,346 historical observations (2014-03-31 to 2025-12-31) from EAC4/ERA5
    reanalysis and NEA PM2.5 station data; coefficients are hardcoded, not refit per run.
    Source data and code: <a href="https://github.com/zgkn/AOD_PM">github.com/zgkn/AOD_PM</a>.
  </footer>
</body>
</html>
"""
    return html


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-nc", default=Path("output/cams_forecast_aod_om.nc"), type=Path)
    parser.add_argument("--training-table", default=Path("data/training_table.csv"), type=Path)
    parser.add_argument("--output", default=Path("site/index.html"), type=Path)
    args = parser.parse_args()

    df = load_forecast(args.forecast_nc)
    print(f"Loaded {len(df)} forecast timesteps: {df['valid_time'].min()} to {df['valid_time'].max()}")

    df = compute_predictions(df)
    print(df[["valid_time", "aod_om", "pm25_linear", "pm25_band", "pm25_max_linear", "pm25_max_band"]].to_string(index=False))

    training_sample = load_training_sample(args.training_table)
    print(f"Loaded {len(training_sample)} historical samples for the regression-result panels")

    figure = build_figure(df, training_sample)
    html = render_html(df, figure, datetime.now(timezone.utc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
