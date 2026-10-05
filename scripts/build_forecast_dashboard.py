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

The dashboard forecasts BOTH targets over the same 5-day horizon, each
with its own linear-estimate panel and its own 4 per-band probability
panels -- the pm25_max track is laid out exactly like the existing
pm25_mean track, just lower on the page, sharing the same synced,
zoomable time x-axis as everything else.

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
ORDINAL_PARAMS = np.array([4.108760, 6.495276, 1.610290, 0.306059])

PM25MAX_LINEAR_INTERCEPT = 13.6568951089901
PM25MAX_LINEAR_SLOPE = 42.92979153232956
PM25MAX_ORDINAL_PARAMS = np.array([4.139879521805435, 5.608515888012341, 1.533790149659414, 0.5983033258719117])

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


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _row_domains(weights: list[float], gap: float = 0.03) -> list[tuple[float, float]]:
    """Top-to-bottom y-domains for stacked subplots, like a matplotlib
    gridspec height_ratios layout."""
    total = sum(weights)
    n = len(weights)
    avail = 1 - gap * (n - 1)
    domains = []
    y_top = 1.0
    for w in weights:
        h = avail * w / total
        y_bottom = y_top - h
        domains.append((round(y_bottom, 6), round(y_top, 6)))
        y_top = y_bottom - gap
    return domains


def _linear_panel(
    data: list, layout: dict, aid: str, domain: tuple, times: list, values: pd.Series,
    y_title: str, panel_title: str,
) -> None:
    y0, y1 = domain
    layout[f"xaxis{aid}"] = {**BASE_X_AXIS, "matches": "x", "domain": [0, 1], "anchor": f"y{aid}", "showticklabels": False}
    layout[f"yaxis{aid}"] = {
        "domain": [y0, y1], "anchor": f"x{aid}", "rangemode": "tozero",
        "title": {"text": y_title, "font": {"size": 11, "color": INK}},
        "gridcolor": GRIDLINE, "tickfont": {"size": 10, "color": MUTED},
    }
    data.append({
        "type": "scatter", "mode": "lines+markers", "x": times, "y": values.round(2).tolist(),
        "line": {"color": INK, "width": 2}, "marker": {"size": 4, "color": INK},
        "xaxis": f"x{aid}", "yaxis": f"y{aid}",
        "hovertemplate": f"%{{x|{DATE_TICKFORMAT}}}<br>{y_title}: %{{y:.1f}}<extra></extra>",
    })
    for edge, label in zip(BAND_EDGES[1:-1], BAND_LABELS[1:]):
        layout["shapes"].append({
            "type": "line", "xref": f"x{aid} domain", "yref": f"y{aid}",
            "x0": 0, "x1": 1, "y0": edge, "y1": edge,
            "line": {"color": BAND_COLORS[label], "width": 1, "dash": "dash"}, "opacity": 0.6,
        })
    layout["annotations"].append({
        "text": panel_title, "xref": "paper", "yref": "paper", "x": 0, "y": y1 + 0.022,
        "xanchor": "left", "yanchor": "bottom", "showarrow": False,
        "font": {"size": 13, "color": INK},
    })


def _band_panels(
    data: list, layout: dict, axis_ids: list, domains: list, times: list, df: pd.DataFrame,
    band_cols: dict, group_title: str, is_final_group: bool,
) -> None:
    for i, label in enumerate(BAND_LABELS):
        aid = axis_ids[i]
        y0, y1 = domains[i]
        color = BAND_COLORS[label]
        is_last_panel = is_final_group and i == len(BAND_LABELS) - 1
        layout[f"xaxis{aid}"] = {
            **BASE_X_AXIS, "matches": "x", "domain": [0, 1], "anchor": f"y{aid}", "showticklabels": is_last_panel,
            **({"title": {"text": "Forecast valid time (UTC)", "font": {"size": 11, "color": INK}}} if is_last_panel else {}),
        }
        layout[f"yaxis{aid}"] = {
            "domain": [y0, y1], "anchor": f"x{aid}", "rangemode": "tozero",
            "title": {"text": "Prob.", "font": {"size": 9, "color": INK}},
            "gridcolor": GRIDLINE, "tickfont": {"size": 9, "color": MUTED},
        }
        data.append({
            "type": "scatter", "mode": "lines", "x": times, "y": df[band_cols[label]].round(4).tolist(),
            "line": {"color": color, "width": 1.5}, "fill": "tozeroy", "fillcolor": _hex_to_rgba(color, 0.35),
            "xaxis": f"x{aid}", "yaxis": f"y{aid}",
            "hovertemplate": f"%{{x|{DATE_TICKFORMAT}}}<br>{label}: %{{y:.1%}}<extra></extra>",
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


def build_figure(df: pd.DataFrame) -> dict:
    """Plotly.js figure spec (data + layout) for the 11-panel dashboard:
    AOD, then two parallel forecast tracks (pm25_mean, pm25_max), each a
    linear-estimate panel plus 4 per-band probability small multiples.
    Every panel shares one synced, zoomable/pannable time x-axis, and
    every y-axis is pinned to start at 0 (rangemode='tozero')."""
    times = [t.isoformat() for t in df["valid_time"]]
    weights = [3, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1]
    domains = _row_domains(weights)
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

    # Panel 2: PM2.5 mean, linear regression estimate
    _linear_panel(
        data, layout, axis_ids[1], domains[1], times, df["pm25_linear"],
        "Predicted PM2.5 mean (µg/m³)",
        "Linear regression estimate -- PM2.5 mean (dashed lines = haze.gov.sg band boundaries)",
    )

    # Panels 3-6: PM2.5 mean, per-band probability small multiples
    _band_panels(
        data, layout, axis_ids[2:6], domains[2:6], times, df,
        band_cols={label: label for label in BAND_LABELS},
        group_title="Ordinal logistic regression -- PM2.5 mean: predicted band probability (per-band detail)",
        is_final_group=False,
    )

    # Panel 7: PM2.5 max, linear regression estimate
    _linear_panel(
        data, layout, axis_ids[6], domains[6], times, df["pm25_max_linear"],
        "Predicted PM2.5 max (µg/m³)",
        "Linear regression estimate -- PM2.5 max (dashed lines = haze.gov.sg band boundaries)",
    )

    # Panels 8-11: PM2.5 max, per-band probability small multiples
    _band_panels(
        data, layout, axis_ids[7:11], domains[7:11], times, df,
        band_cols={label: f"max_{label}" for label in BAND_LABELS},
        group_title="Ordinal logistic regression -- PM2.5 max: predicted band probability (per-band detail)",
        is_final_group=True,
    )

    return {"data": data, "layout": layout}


def render_html(df: pd.DataFrame, figure: dict, generated_at: datetime) -> str:
    figure_json = json.dumps(figure)
    config_json = json.dumps({
        "responsive": True, "scrollZoom": True, "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    })
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
  .meta {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 24px; }}
  .hint {{ color: {MUTED}; font-size: 0.8rem; margin-bottom: 12px; }}
  #chart {{ width: 100%; height: 1900px; border-radius: 8px; border: 1px solid {GRIDLINE}; background: {SURFACE}; }}
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

  <footer>
    Models fit once on 34,346 historical observations (2014-03-31 to 2025-12-31) from EAC4/ERA5
    reanalysis and NEA PM2.5 station data; coefficients are hardcoded, not refit per run.
    pm25_mean ~ aod_om -- linear: R&sup2;=0.467, ordinal logistic: McFadden pseudo-R&sup2;=0.513.
    pm25_max ~ aod_om -- linear: R&sup2;=0.410, ordinal logistic: McFadden pseudo-R&sup2;=0.416.
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
    print(df[["valid_time", "aod_om", "pm25_linear", "pm25_band", "pm25_max_linear", "pm25_max_band"]].to_string(index=False))

    figure = build_figure(df)
    html = render_html(df, figure, datetime.now(timezone.utc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
