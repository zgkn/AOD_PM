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

BIAS CORRECTION (linear lines only): the two linear forecasts are nudged
toward the latest NEA station readings from data.gov.sg -- an additive
offset (observed minus raw model, averaged over the forecast steps that
already have a reading) added unchanged to every forecast step.
Mean forecast <- mean of the 5 stations; max forecast <- max of the 5. The
ordinal band probabilities are NOT corrected. See bias_correction.py. If
the readings can't be fetched the dashboard falls back to the raw forecast
and says so in the header.

TIME ZONES: the CAMS forecast is UTC, the station readings are SGT. All
matching is done in UTC (bias_correction.py); the chart is *displayed* in
SGT by shifting every plotted timestamp by +8h in build_chart_spec (the
client engine formats with getUTC*, so shifted values read as SGT
wall-clock time).

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

The chart is a hand-rolled SVG engine (no charting library, no CDN
dependency): each panel is its own inline SVG, laid out in normal
document flow, driven by a small JSON spec embedded in the page. Two
independent axis "groups" exist -- "time" (panels 1-6) and "aod" (panels
7-8, the regression-result panels) -- each with its own pan/wheel-zoom/
pinch-zoom state, "nice" round-number tick computation (recomputed on
every zoom/pan for the x-axis, computed once from the full data range for
each panel's own y-axis), and crosshair-following tooltip. See
_ENGINE_JS below for the implementation.
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.miscmodels.ordinal_model import OrderedModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bias_correction import (  # noqa: E402
    SGT_OFFSET,
    apply_bias_correction,
    fetch_observations,
    items_to_observations,
    utc_now_naive,
)

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
RAW_LINE_OPACITY = 0.3  # uncorrected linear lines, shown faintly beneath the corrected ones

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
    training table, for the historical-regression scatter panel --
    plotting all 34,346 rows would bloat the page and slow down
    rendering for no visible gain in density."""
    full = pd.read_csv(path, usecols=["aod_om", "pm25_mean", "pm25_max"]).dropna()
    sample = full.sample(n=min(n, len(full)), random_state=seed)
    return sample.reset_index(drop=True)


def _series(
    x, y, color: str, *, label: str, fmt: str, dash: bool = False, markers: bool = False,
    scatter: bool = False, fill: bool = False, width: float = 1.6, opacity: float = 1.0,
) -> dict:
    """One drawable trace within a panel. `fmt` names a value formatter the
    client-side engine knows (f1 = "12.3 µg/m³", f3 = "0.123", pct1 = "12.3%").
    `scatter` marks an unsorted point cloud (hover finds the nearest point by
    pixel distance); everything else is treated as x-sorted line data (hover
    binary-searches the nearest x)."""
    return {
        "x": [round(float(v), 4) for v in x],
        "y": [round(float(v), 4) for v in y],
        "color": color, "label": label, "fmt": fmt,
        "dash": dash, "markers": markers, "scatter": scatter, "fill": fill,
        "width": width, "opacity": opacity,
    }


def _label(text: str, color: str, x: float = 0.01, y: float = 0.95) -> dict:
    """An inline text annotation at a fraction (x, y) of the panel's plot
    area -- used in place of a legend, consistent with the rest of the page."""
    return {"text": text, "color": color, "x": x, "y": y}


def _panel(
    id_: str, group: str, weight: float, *, series: list, y_mode: str = "tozero",
    y_label: str | None = None, y_tick_fmt: str = "num", title: str | None = None,
    show_x_axis: bool = False, x_label: str | None = None,
    hlines: list | None = None, labels: list | None = None,
) -> dict:
    """A single chart panel. `weight` sets its plot height relative to the
    other panels in the same stack (matplotlib-gridspec style). `y_mode` is
    "tozero" (y-axis autoscaled per-panel from that panel's own data, pinned
    to start at 0) or "fixed01" (fixed [0, 1] range, for probability curves
    spanning the full aod_om grid). `group` names which of the two
    independent pan/zoom axis groups ("time" or "aod") this panel's x-axis
    belongs to."""
    return {
        "id": id_, "group": group, "weight": weight, "series": series,
        "yMode": y_mode, "yLabel": y_label, "yTickFmt": y_tick_fmt, "title": title,
        "showXAxis": show_x_axis, "xLabel": x_label,
        "hlines": hlines or [], "labels": labels or [],
    }


def build_chart_spec(df: pd.DataFrame, training_sample: pd.DataFrame, bias: dict | None = None) -> dict:
    """JSON spec for the 8-panel dashboard, consumed by the client-side SVG
    engine (_ENGINE_JS): 6 forecast panels (AOD, a linear-estimate panel, and
    4 per-band probability panels -- each of the latter 5 overlays
    pm25_mean/solid and pm25_max/dashed in the same panel) sharing one
    synced time axis group, plus 2 regression-result panels (historical
    scatter + fit line -- mean/max distinguished by color, blue/red, since
    two scatter clouds overlap there -- and ordinal logistic probability
    curves, mean solid/max dashed per band color) sharing their own synced,
    independent aod_om axis group.

    `df["valid_time"]` is naive UTC; every time sent to the chart is shifted
    +8h so the (getUTC*-based) client engine displays SGT. `bias` is the info
    dict from apply_bias_correction (None = no correction attempted)."""
    def to_chart_ms(times: pd.Series) -> list:
        return (
            ((times + SGT_OFFSET) - pd.Timestamp("1970-01-01")) / pd.Timedelta(milliseconds=1)
        ).astype("int64").tolist()

    times_ms = to_chart_ms(df["valid_time"])
    corrected = bool(bias) and bias["status"] == "applied"

    panel_aod = _panel(
        "aod", "time", 3,
        series=[_series(times_ms, df["aod_om"], AOD_COLOR, label="AOD_om", fmt="f3", markers=True, width=2)],
        y_label="Organic matter AOD",
        title="CAMS forecast: max organic matter AOD over Singapore region, next 5 days",
    )

    if corrected:
        linear_series = [
            _series(times_ms, df["pm25_linear"], INK, label="PM2.5 mean (raw)", fmt="f1",
                    width=1.6, opacity=RAW_LINE_OPACITY),
            _series(times_ms, df["pm25_max_linear"], INK, label="PM2.5 max (raw)", fmt="f1",
                    dash=True, width=1.4, opacity=RAW_LINE_OPACITY),
            _series(times_ms, df["pm25_linear_corrected"], INK, label="PM2.5 mean (corrected)", fmt="f1",
                    markers=True, width=2),
            _series(times_ms, df["pm25_max_linear_corrected"], INK, label="PM2.5 max (corrected)", fmt="f1",
                    dash=True, width=1.6),
        ]
        obs = bias["obs"]
        obs_ms = to_chart_ms(obs["obs_time"])
        linear_series += [
            _series(obs_ms, obs["obs_mean"], PM25_MEAN_COLOR, label="Observed mean (5 stations)",
                    fmt="f1", scatter=True),
            _series(obs_ms, obs["obs_max"], PM25_MAX_COLOR, label="Observed max (5 stations)",
                    fmt="f1", scatter=True),
        ]
        # Legend sits right of x=0.17: the observed dots cluster at the panel's left edge.
        linear_labels = [
            _label("— PM2.5 mean (bias-corrected)", INK, 0.17, 0.95),
            _label("- - PM2.5 max (bias-corrected)", INK, 0.17, 0.85),
            _label("faint = uncorrected regression", MUTED, 0.17, 0.75),
            _label("● observed mean of stations", PM25_MEAN_COLOR, 0.17, 0.65),
            _label("● observed max of stations", PM25_MAX_COLOR, 0.17, 0.55),
        ]
        linear_title = ("Linear regression estimate, bias-corrected to latest station readings "
                        "(dashed lines = haze.gov.sg band boundaries)")
    else:
        linear_series = [
            _series(times_ms, df["pm25_linear"], INK, label="PM2.5 mean", fmt="f1", markers=True, width=2),
            _series(times_ms, df["pm25_max_linear"], INK, label="PM2.5 max", fmt="f1", dash=True, width=1.6),
        ]
        linear_labels = [_label("— PM2.5 mean", INK, 0.01, 0.95), _label("- - PM2.5 max", INK, 0.01, 0.85)]
        linear_title = "Linear regression estimate (dashed lines = haze.gov.sg band boundaries)"

    panel_linear = _panel(
        "linear", "time", 3,
        series=linear_series,
        y_label="Predicted PM2.5 (µg/m³)",
        title=linear_title,
        hlines=[{"y": edge, "color": BAND_COLORS[label]} for edge, label in zip(BAND_EDGES[1:-1], BAND_LABELS[1:])],
        labels=linear_labels,
    )

    band_panels = []
    for i, label in enumerate(BAND_LABELS):
        color = BAND_COLORS[label]
        is_last = i == len(BAND_LABELS) - 1
        band_panels.append(_panel(
            f"band_{label}", "time", 1,
            series=[
                _series(times_ms, df[label], color, label=f"{label} (mean)", fmt="pct1", fill=True, width=1.4),
                _series(times_ms, df[f"max_{label}"], color, label=f"{label} (max)", fmt="pct1", dash=True, width=1.4),
            ],
            y_label="Prob.", y_tick_fmt="pct",
            title=(
                "Ordinal logistic regression: predicted PM2.5 band probability (per-band detail)"
                if i == 0 else None
            ),
            show_x_axis=is_last, x_label="Forecast valid time (SGT)" if is_last else None,
            labels=[_label(label, color, 0.01, 0.85)],
        ))

    # Grid spans whichever is wider, historical aod_om or this forecast's
    # aod_om, so the fit lines/curves cover the forecast's actual range too.
    x_max = float(max(training_sample["aod_om"].max(), df["aod_om"].max())) * 1.05
    x_grid = np.linspace(0, x_max, 200)

    panel_scatter = _panel(
        "scatter", "aod", 3,
        series=[
            _series(training_sample["aod_om"], training_sample["pm25_mean"], PM25_MEAN_COLOR,
                    label="PM2.5 mean", fmt="f1", scatter=True, opacity=0.35),
            _series(training_sample["aod_om"], training_sample["pm25_max"], PM25_MAX_COLOR,
                    label="PM2.5 max", fmt="f1", scatter=True, opacity=0.35),
            _series(x_grid, LINEAR_INTERCEPT + LINEAR_SLOPE * x_grid, PM25_MEAN_COLOR,
                    label="PM2.5 mean (fit)", fmt="f1", width=2.5),
            _series(x_grid, PM25MAX_LINEAR_INTERCEPT + PM25MAX_LINEAR_SLOPE * x_grid, PM25_MAX_COLOR,
                    label="PM2.5 max (fit)", fmt="f1", width=2.5),
        ],
        y_label="Historical PM2.5 (µg/m³)",
        title="Linear regression: historical PM2.5 vs AOD_om (points) with the fitted line",
        labels=[
            _label("— PM2.5 mean fit", PM25_MEAN_COLOR, 0.01, 0.95),
            _label("— PM2.5 max fit", PM25_MAX_COLOR, 0.01, 0.85),
        ],
    )

    mean_probs = predict_ordinal_probs(pd.Series(x_grid))
    max_probs = predict_ordinal_probs(pd.Series(x_grid), PM25MAX_ORDINAL_PARAMS)
    curve_series = []
    for label in BAND_LABELS:
        color = BAND_COLORS[label]
        curve_series.append(_series(x_grid, mean_probs[label], color, label=f"{label} (mean)", fmt="pct1", width=2))
        curve_series.append(_series(x_grid, max_probs[label], color, label=f"{label} (max)", fmt="pct1", dash=True, width=2))

    panel_curves = _panel(
        "curves", "aod", 3,
        series=curve_series,
        y_mode="fixed01", y_label="Predicted probability", y_tick_fmt="pct",
        title="Ordinal logistic: band probability vs AOD_om (solid = PM2.5 mean, dashed = PM2.5 max)",
        show_x_axis=True, x_label="Organic matter AOD",
    )

    span_ms = max(times_ms) - min(times_ms)
    pad_ms = int(span_ms * 0.02) or 1
    time_extent = [min(times_ms) - pad_ms, max(times_ms) + pad_ms]

    return {
        "groups": {
            "time": {"type": "time", "extent": time_extent},
            "aod": {"type": "linear", "extent": [0, x_max]},
        },
        "panels": [panel_aod, panel_linear, *band_panels, panel_scatter, panel_curves],
    }


def render_model_details(bias: dict | None = None) -> str:
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
    if bias and bias["status"] == "applied":
        rows.append((
            "PM2.5 mean / max", "Bias correction (linear only)",
            "corrected = raw + offset (constant over the horizon)",
            f"offsets: mean {bias['offset_mean']:+.1f}, max {bias['offset_max']:+.1f} &micro;g/m&sup3;",
        ))
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
    The bias correction touches only the two linear lines in the linear-estimate panel; the
    ordinal band probabilities are never corrected.
    All four models were fit once (2026-10-03/2026-10-05) on the full historical training
    table (n=34,346, 2014-03-31 to 2025-12-31) and are hardcoded here, not refit per run.
  </p>
"""


# Hand-rolled SVG charting engine: one <svg> per panel stacked in normal
# document flow (no absolute-positioned mega-canvas), each panel redrawn
# from scratch on every pan/zoom/resize. Two independent axis "groups"
# ("time" and "aod") each hold their own current visible x-domain; zooming
# or panning any panel re-renders every panel that shares its group. Y-axes
# never rescale on x zoom/pan (computed once from each panel's full data
# range) -- only the x ticks are recomputed live, via the same "nice round
# number" step-selection algorithm for both the linear and time axes.
_ENGINE_JS = r"""
(function () {
  'use strict';
  const SPEC = __CHART_SPEC_JSON__;
  const COL = { grid: '__GRIDLINE__', muted: '__MUTED__', ink: '__INK__' };

  const MARGIN_L = 58, MARGIN_R = 14;
  const ROW_UNIT = 86, TITLE_H = 24, XAXIS_H = 42, GROUP_GAP = 18;
  const SVGNS = 'http://www.w3.org/2000/svg';

  function pad2(n) { return String(n).padStart(2, '0'); }
  // Times arrive pre-shifted +8h by build_chart_spec, so UTC getters read as SGT.
  function fmtDate(ms, withTime) {
    const d = new Date(ms);
    const date = `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`;
    return withTime ? `${date} ${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}` : date;
  }

  function formatTick(v) {
    let s = Math.abs(v) >= 10 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2);
    if (s.indexOf('.') >= 0) s = s.replace(/0+$/, '').replace(/\.$/, '');
    return s === '-0' ? '0' : s;
  }
  function formatPercent(v) {
    const pct = v * 100;
    if (Math.abs(pct) < 1e-9) return '0%';
    return (Math.abs(pct) >= 10 ? pct.toFixed(0) : Math.abs(pct) >= 1 ? pct.toFixed(1) : pct.toFixed(2)) + '%';
  }
  function fmtValue(v, fmt) {
    if (fmt === 'f1') return v.toFixed(1) + ' µg/m³';
    if (fmt === 'f3') return v.toFixed(3);
    if (fmt === 'pct1') return (v * 100).toFixed(1) + '%';
    return String(v);
  }

  // "Nice" round-number tick step selection (classic extended-Wilkinson-
  // lite): pick 1/2/5 x 10^k closest to span/targetCount, snap domain out
  // to multiples of that step.
  function niceLinear(dataMin, dataMax, targetCount) {
    let min = dataMin, max = dataMax;
    if (max - min < 1e-12) max = min + 1;
    const rawStep = (max - min) / Math.max(1, targetCount);
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const norm = rawStep / mag;
    const step = norm >= 5 ? 10 * mag : norm >= 2 ? 5 * mag : norm >= 1 ? 2 * mag : mag;
    const niceMin = Math.floor(min / step) * step;
    const niceMax = Math.ceil(max / step) * step;
    const n = Math.round((niceMax - niceMin) / step);
    const ticks = [];
    for (let i = 0; i <= n; i++) ticks.push(+(niceMin + i * step).toFixed(10));
    return { ticks, min: niceMin, max: niceMax, step };
  }

  const TIME_STEPS_MS = [
    3600e3, 2 * 3600e3, 3 * 3600e3, 6 * 3600e3, 12 * 3600e3,
    24 * 3600e3, 2 * 24 * 3600e3, 3 * 24 * 3600e3, 5 * 24 * 3600e3, 10 * 24 * 3600e3, 20 * 24 * 3600e3,
  ];
  function niceTime(minMs, maxMs, targetCount) {
    const span = Math.max(1, maxMs - minMs);
    let step = TIME_STEPS_MS[TIME_STEPS_MS.length - 1];
    for (const s of TIME_STEPS_MS) { if (span / s <= targetCount) { step = s; break; } }
    const start = Math.ceil(minMs / step) * step;
    const ticks = [];
    for (let t = start; t <= maxMs; t += step) ticks.push(t);
    return { ticks, step };
  }

  function makeScale(domain, range) {
    const k = (range[1] - range[0]) / ((domain[1] - domain[0]) || 1);
    return {
      domain, range,
      toPx: v => range[0] + (v - domain[0]) * k,
      toVal: px => domain[0] + (px - range[0]) / k,
    };
  }

  function nearestIndex(arr, val) {
    let lo = 0, hi = arr.length - 1;
    if (hi < 0) return -1;
    if (val <= arr[0]) return 0;
    if (val >= arr[hi]) return hi;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid] < val) lo = mid + 1; else hi = mid;
    }
    const a = arr[lo - 1];
    return a === undefined ? lo : (val - a) < (arr[lo] - val) ? lo - 1 : lo;
  }

  // ---- groups & panels ----
  const groups = {};
  for (const gid in SPEC.groups) {
    const g = SPEC.groups[gid];
    groups[gid] = { type: g.type, full: g.extent.slice(), domain: g.extent.slice(), panels: [] };
  }
  const container = document.getElementById('chart');
  const tooltip = document.getElementById('tooltip');

  const panels = SPEC.panels.map((spec, idx) => {
    const group = groups[spec.group];
    const p = { spec, group, idx };
    group.panels.push(p);
    return p;
  });

  function computeDataMax(p) {
    let m = 0;
    for (const s of p.spec.series) for (const y of s.y) if (isFinite(y) && y > m) m = y;
    return m;
  }

  function computeLayout(p) {
    const plotH = p.spec.weight * ROW_UNIT;
    const topPad = p.spec.title ? TITLE_H : 6;
    const botPad = p.spec.showXAxis ? XAXIS_H : 8;
    p.layout = { topPad, plotH, botPad, totalH: topPad + plotH + botPad };
  }

  function renderGroup(g) { g.panels.forEach(p => renderPanel(p)); }
  function renderAll() { panels.forEach(p => renderPanel(p)); }

  function clampDomain(domain, full) {
    let [a, b] = domain;
    const fullSpan = full[1] - full[0];
    const minSpan = fullSpan / 500;
    let span = b - a;
    if (span < minSpan) { const c = (a + b) / 2; a = c - minSpan / 2; b = c + minSpan / 2; span = minSpan; }
    if (span > fullSpan) { a = full[0]; b = full[1]; span = fullSpan; }
    if (a < full[0]) { b += full[0] - a; a = full[0]; }
    if (b > full[1]) { a -= b - full[1]; b = full[1]; }
    return [a, b];
  }

  function createPanelDom(p, i) {
    computeLayout(p);
    const wrap = document.createElement('div');
    wrap.className = 'panel';
    const isLastOfGroup = (i === panels.length - 1) || (panels[i + 1].spec.group !== p.spec.group);
    wrap.style.marginBottom = (isLastOfGroup && i !== panels.length - 1) ? GROUP_GAP + 'px' : '0px';
    const svg = document.createElementNS(SVGNS, 'svg');
    svg.style.display = 'block';
    svg.style.width = '100%';
    // Allow the browser's native vertical scroll (the page is taller than
    // the viewport); only horizontal gestures are ours to interpret as pan
    // or pinch -- see the touchstart/touchmove direction-lock below, which
    // decides per-gesture whether a single-finger touch pans the chart or
    // is left alone to scroll the page.
    svg.style.touchAction = 'pan-y';
    svg.style.cursor = 'crosshair';
    wrap.appendChild(svg);
    container.appendChild(wrap);
    p.svg = svg;
    p.clipId = 'clip' + p.idx;
    attachInteractions(p);
  }

  function renderPanel(p) {
    const svg = p.svg;
    const width = Math.max(260, container.clientWidth);
    const { topPad, plotH, botPad, totalH } = p.layout;
    const plotW = Math.max(10, width - MARGIN_L - MARGIN_R);
    svg.setAttribute('viewBox', `0 0 ${width} ${totalH}`);
    svg.setAttribute('width', width);
    svg.setAttribute('height', totalH);
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const plotTop = topPad, plotLeft = MARGIN_L;

    const defs = document.createElementNS(SVGNS, 'defs');
    const clip = document.createElementNS(SVGNS, 'clipPath');
    clip.setAttribute('id', p.clipId);
    const clipRect = document.createElementNS(SVGNS, 'rect');
    clipRect.setAttribute('x', plotLeft); clipRect.setAttribute('y', plotTop);
    clipRect.setAttribute('width', plotW); clipRect.setAttribute('height', plotH);
    clip.appendChild(clipRect);
    defs.appendChild(clip);
    svg.appendChild(defs);

    const xScale = makeScale(p.group.domain, [plotLeft, plotLeft + plotW]);
    let yInfo;
    const isPct = p.spec.yTickFmt === 'pct';
    if (p.spec.yMode === 'fixed01') {
      yInfo = { ticks: [0, 0.2, 0.4, 0.6, 0.8, 1.0], min: 0, max: 1 };
    } else {
      if (p._dataMax === undefined) p._dataMax = computeDataMax(p);
      yInfo = niceLinear(0, (p._dataMax * 1.08) || 1, 4);
      // Probabilities have a hard ceiling at 1 (100%) that "nice" step
      // rounding can overshoot (e.g. a step of 0.5 pads 0.9 up to 1.5);
      // clamp back down and drop ticks past the ceiling.
      if (isPct && p._dataMax <= 1) {
        yInfo = { ticks: yInfo.ticks.filter(t => t <= 1 + 1e-9), min: 0, max: Math.min(yInfo.max, 1), step: yInfo.step };
      }
    }
    const yScale = makeScale([yInfo.min, yInfo.max], [plotTop + plotH, plotTop]);
    p._xScale = xScale; p._yScale = yScale;

    // y gridlines + labels
    const yg = document.createElementNS(SVGNS, 'g');
    yInfo.ticks.forEach(t => {
      const py = yScale.toPx(t);
      const line = document.createElementNS(SVGNS, 'line');
      line.setAttribute('x1', plotLeft); line.setAttribute('x2', plotLeft + plotW);
      line.setAttribute('y1', py); line.setAttribute('y2', py);
      line.setAttribute('stroke', COL.grid); line.setAttribute('stroke-width', '1');
      yg.appendChild(line);
      const txt = document.createElementNS(SVGNS, 'text');
      txt.setAttribute('x', plotLeft - 6); txt.setAttribute('y', py + 3);
      txt.setAttribute('text-anchor', 'end'); txt.setAttribute('font-size', '9.5');
      txt.setAttribute('fill', COL.muted);
      txt.textContent = isPct ? formatPercent(t) : formatTick(t);
      yg.appendChild(txt);
    });
    svg.appendChild(yg);

    // x ticks + gridlines. A floor of 4 keeps the time axis from
    // collapsing to 1-2 widely/oddly spaced ticks on a narrow mobile
    // viewport, where plotW/divisor alone would round down to 1 or 2.
    const xTickInfo = p.group.type === 'time'
      ? niceTime(p.group.domain[0], p.group.domain[1], Math.max(4, Math.floor(plotW / 90)))
      : niceLinear(p.group.domain[0], p.group.domain[1], Math.max(3, Math.floor(plotW / 70)));
    const xg = document.createElementNS(SVGNS, 'g');
    xTickInfo.ticks.forEach(t => {
      const px = xScale.toPx(t);
      if (px < plotLeft - 1 || px > plotLeft + plotW + 1) return;
      const line = document.createElementNS(SVGNS, 'line');
      line.setAttribute('x1', px); line.setAttribute('x2', px);
      line.setAttribute('y1', plotTop); line.setAttribute('y2', plotTop + plotH);
      line.setAttribute('stroke', COL.grid); line.setAttribute('stroke-width', '1');
      xg.appendChild(line);
      if (p.spec.showXAxis) {
        const txt = document.createElementNS(SVGNS, 'text');
        txt.setAttribute('font-size', '9.5'); txt.setAttribute('fill', COL.muted);
        txt.setAttribute('transform', `translate(${px},${plotTop + plotH + 14}) rotate(-28)`);
        txt.setAttribute('text-anchor', 'end');
        txt.textContent = p.group.type === 'time' ? fmtDate(t, xTickInfo.step < 86400e3) : formatTick(t);
        xg.appendChild(txt);
      }
    });
    svg.appendChild(xg);

    const axisLine = document.createElementNS(SVGNS, 'line');
    axisLine.setAttribute('x1', plotLeft); axisLine.setAttribute('x2', plotLeft);
    axisLine.setAttribute('y1', plotTop); axisLine.setAttribute('y2', plotTop + plotH);
    axisLine.setAttribute('stroke', COL.grid);
    svg.appendChild(axisLine);

    if (p.spec.yLabel) {
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('font-size', '10'); t.setAttribute('fill', COL.ink);
      t.setAttribute('transform', `translate(14,${plotTop + plotH / 2}) rotate(-90)`);
      t.setAttribute('text-anchor', 'middle');
      t.textContent = p.spec.yLabel;
      svg.appendChild(t);
    }
    if (p.spec.showXAxis && p.spec.xLabel) {
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('font-size', '11'); t.setAttribute('fill', COL.ink);
      t.setAttribute('x', plotLeft + plotW / 2); t.setAttribute('y', totalH - 2);
      t.setAttribute('text-anchor', 'middle');
      t.textContent = p.spec.xLabel;
      svg.appendChild(t);
    }
    if (p.spec.title) {
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('font-size', '13'); t.setAttribute('fill', COL.ink);
      t.setAttribute('x', plotLeft); t.setAttribute('y', 16);
      t.textContent = p.spec.title;
      svg.appendChild(t);
    }

    const plotG = document.createElementNS(SVGNS, 'g');
    plotG.setAttribute('clip-path', `url(#${p.clipId})`);
    (p.spec.hlines || []).forEach(hl => {
      const py = yScale.toPx(hl.y);
      const line = document.createElementNS(SVGNS, 'line');
      line.setAttribute('x1', plotLeft); line.setAttribute('x2', plotLeft + plotW);
      line.setAttribute('y1', py); line.setAttribute('y2', py);
      line.setAttribute('stroke', hl.color); line.setAttribute('stroke-width', '1');
      line.setAttribute('stroke-dasharray', '5,4'); line.setAttribute('opacity', '0.6');
      plotG.appendChild(line);
    });

    p.spec.series.forEach(s => {
      if (s.scatter) {
        plotG.appendChild(renderScatter(s, xScale, yScale));
      } else {
        plotG.appendChild(renderLine(s, xScale, yScale, yScale.toPx(0)));
        if (s.markers) plotG.appendChild(renderMarkers(s, xScale, yScale));
      }
    });
    svg.appendChild(plotG);

    (p.spec.labels || []).forEach(l => {
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('font-size', '10'); t.setAttribute('fill', l.color);
      t.setAttribute('x', plotLeft + l.x * plotW); t.setAttribute('y', plotTop + (1 - l.y) * plotH);
      t.textContent = l.text;
      svg.appendChild(t);
    });

    const cross = document.createElementNS(SVGNS, 'g');
    cross.setAttribute('display', 'none');
    const cl = document.createElementNS(SVGNS, 'line');
    cl.setAttribute('y1', plotTop); cl.setAttribute('y2', plotTop + plotH);
    cl.setAttribute('stroke', COL.ink); cl.setAttribute('stroke-width', '1'); cl.setAttribute('opacity', '0.35');
    cross.appendChild(cl);
    svg.appendChild(cross);
    p.crossG = cross; p.crossLine = cl;
    p.plotRect = { left: plotLeft, top: plotTop, width: plotW, height: plotH };
  }

  function renderLine(s, xScale, yScale, y0px) {
    const g = document.createElementNS(SVGNS, 'g');
    let d = '';
    for (let i = 0; i < s.x.length; i++) {
      const px = xScale.toPx(s.x[i]), py = yScale.toPx(s.y[i]);
      d += (i === 0 ? 'M' : 'L') + px.toFixed(1) + ',' + py.toFixed(1) + ' ';
    }
    if (s.fill) {
      const first = xScale.toPx(s.x[0]), last = xScale.toPx(s.x[s.x.length - 1]);
      const fillPath = document.createElementNS(SVGNS, 'path');
      fillPath.setAttribute('d', d + `L${last.toFixed(1)},${y0px.toFixed(1)} L${first.toFixed(1)},${y0px.toFixed(1)} Z`);
      fillPath.setAttribute('fill', s.color); fillPath.setAttribute('fill-opacity', '0.3'); fillPath.setAttribute('stroke', 'none');
      g.appendChild(fillPath);
    }
    const path = document.createElementNS(SVGNS, 'path');
    path.setAttribute('d', d);
    path.setAttribute('fill', 'none'); path.setAttribute('stroke', s.color);
    path.setAttribute('stroke-width', s.width || 1.5);
    if (s.dash) path.setAttribute('stroke-dasharray', '6,4');
    if (s.opacity != null && s.opacity < 1) path.setAttribute('stroke-opacity', s.opacity);
    g.appendChild(path);
    return g;
  }

  function dotsPath(points, r) {
    let d = '';
    for (const [px, py] of points) {
      d += `M${(px - r).toFixed(1)},${py.toFixed(1)} a${r},${r} 0 1,0 ${(2 * r).toFixed(1)},0 a${r},${r} 0 1,0 ${(-2 * r).toFixed(1)},0 `;
    }
    return d;
  }
  function renderMarkers(s, xScale, yScale) {
    const pts = s.x.map((xv, i) => [xScale.toPx(xv), yScale.toPx(s.y[i])]);
    const path = document.createElementNS(SVGNS, 'path');
    path.setAttribute('d', dotsPath(pts, 2.2));
    path.setAttribute('fill', s.color); path.setAttribute('stroke', 'none');
    return path;
  }
  function renderScatter(s, xScale, yScale) {
    const [d0, d1] = xScale.domain;
    const pad = (d1 - d0) * 0.02;
    const pts = [];
    for (let i = 0; i < s.x.length; i++) {
      const xv = s.x[i];
      if (xv < d0 - pad || xv > d1 + pad) continue;
      pts.push([xScale.toPx(xv), yScale.toPx(s.y[i])]);
    }
    const path = document.createElementNS(SVGNS, 'path');
    path.setAttribute('d', dotsPath(pts, 2.4));
    path.setAttribute('fill', s.color);
    path.setAttribute('fill-opacity', s.opacity != null ? s.opacity : 1);
    path.setAttribute('stroke', 'none');
    return path;
  }

  // ---- interaction: wheel zoom, drag pan, pinch zoom, dbl-click/tap reset, crosshair hover ----
  function attachInteractions(p) {
    const svg = p.svg;
    let dragging = false, dragStartPx = 0, dragStartDomain = null;
    let lastTap = 0, pinchStartDist = null, pinchStartDomain = null;
    let rafPending = false, pendingEvt = null;

    function toLocalPx(clientX, clientY) {
      const rect = svg.getBoundingClientRect();
      const k = svg.viewBox.baseVal.width / rect.width;
      return [(clientX - rect.left) * k, (clientY - rect.top) * k];
    }

    function zoomAt(px, factor) {
      const scale = p._xScale;
      if (!scale) return;
      const val = scale.toVal(px);
      let [a, b] = p.group.domain;
      a = val - (val - a) * factor;
      b = val + (b - val) * factor;
      p.group.domain = clampDomain([a, b], p.group.full);
      renderGroup(p.group);
    }

    svg.addEventListener('wheel', e => {
      e.preventDefault();
      const [px] = toLocalPx(e.clientX, e.clientY);
      zoomAt(px, Math.exp(e.deltaY * 0.0015));
      hideTooltip(p);
    }, { passive: false });

    svg.addEventListener('mousedown', e => {
      if (e.button !== 0) return;
      dragging = true; dragStartPx = e.clientX; dragStartDomain = p.group.domain.slice();
      svg.style.cursor = 'grabbing';
    });
    window.addEventListener('mousemove', e => {
      if (!dragging) return;
      const rect = svg.getBoundingClientRect();
      const k = svg.viewBox.baseVal.width / rect.width;
      const dxPx = (e.clientX - dragStartPx) * k;
      const scale = p._xScale; if (!scale) return;
      const perPx = (scale.domain[1] - scale.domain[0]) / (scale.range[1] - scale.range[0]);
      p.group.domain = clampDomain([dragStartDomain[0] - dxPx * perPx, dragStartDomain[1] - dxPx * perPx], p.group.full);
      renderGroup(p.group);
    });
    window.addEventListener('mouseup', () => { dragging = false; svg.style.cursor = 'crosshair'; });

    svg.addEventListener('dblclick', () => { p.group.domain = p.group.full.slice(); renderGroup(p.group); });

    // A single-finger touch could mean "pan the chart" or "scroll the
    // page" -- they can't both win. Stay undecided for the first ~8px of
    // movement, then lock to whichever axis dominates and stick with that
    // for the rest of the gesture (the same disambiguation a horizontal
    // carousel uses). While undecided/scrolling we touch nothing, so the
    // page's native touch-action:pan-y scroll runs unobstructed.
    let touchMode = null, touchStartX = 0, touchStartY = 0;
    const LOCK_PX = 8;

    svg.addEventListener('touchstart', e => {
      if (e.touches.length === 1) {
        touchMode = null;
        touchStartX = e.touches[0].clientX; touchStartY = e.touches[0].clientY;
        dragStartPx = touchStartX; dragStartDomain = p.group.domain.slice();
        const now = Date.now();
        if (now - lastTap < 320) { p.group.domain = p.group.full.slice(); renderGroup(p.group); }
        lastTap = now;
      } else if (e.touches.length === 2) {
        touchMode = 'pinch';
        pinchStartDist = Math.abs(e.touches[0].clientX - e.touches[1].clientX);
        pinchStartDomain = p.group.domain.slice();
      }
    }, { passive: true });
    svg.addEventListener('touchmove', e => {
      const rect = svg.getBoundingClientRect();
      const k = svg.viewBox.baseVal.width / rect.width;
      if (e.touches.length === 1) {
        const dx = e.touches[0].clientX - touchStartX, dy = e.touches[0].clientY - touchStartY;
        if (touchMode === null) {
          if (Math.hypot(dx, dy) < LOCK_PX) return;
          touchMode = Math.abs(dx) > Math.abs(dy) ? 'pan' : 'scroll';
        }
        if (touchMode !== 'pan') return;
        const dxPx = (e.touches[0].clientX - dragStartPx) * k;
        const scale = p._xScale; if (!scale) return;
        const perPx = (scale.domain[1] - scale.domain[0]) / (scale.range[1] - scale.range[0]);
        p.group.domain = clampDomain([dragStartDomain[0] - dxPx * perPx, dragStartDomain[1] - dxPx * perPx], p.group.full);
        renderGroup(p.group);
      } else if (e.touches.length === 2 && pinchStartDist) {
        const dist = Math.abs(e.touches[0].clientX - e.touches[1].clientX);
        const factor = pinchStartDist / Math.max(1, dist);
        const midPx = ((e.touches[0].clientX + e.touches[1].clientX) / 2 - rect.left) * k;
        const scale = p._xScale; if (!scale) return;
        const val = scale.toVal(midPx);
        const [a0, b0] = pinchStartDomain;
        p.group.domain = clampDomain([val - (val - a0) * factor, val + (b0 - val) * factor], p.group.full);
        renderGroup(p.group);
      }
    }, { passive: true });
    svg.addEventListener('touchend', e => { if (e.touches.length === 0) { touchMode = null; pinchStartDist = null; } });

    svg.addEventListener('mousemove', e => {
      if (dragging) return;
      pendingEvt = e;
      if (rafPending) return;
      rafPending = true;
      requestAnimationFrame(() => {
        rafPending = false;
        const ev = pendingEvt;
        const [px, py] = toLocalPx(ev.clientX, ev.clientY);
        showCrosshair(p, px, py, ev.clientX, ev.clientY);
      });
    });
    svg.addEventListener('mouseleave', () => { hideCrosshair(p); hideTooltip(p); });
  }

  function hideCrosshair(p) { if (p.crossG) p.crossG.setAttribute('display', 'none'); }
  function hideTooltip() { tooltip.style.display = 'none'; }

  function showCrosshair(p, px, py, clientX, clientY) {
    const r = p.plotRect;
    if (!r || px < r.left || px > r.left + r.width || py < r.top || py > r.top + r.height) {
      hideCrosshair(p); hideTooltip(); return;
    }
    const xScale = p._xScale, yScale = p._yScale;
    const xVal = xScale.toVal(px);

    let scatterHit = null, bestD = 20;
    p.spec.series.forEach(s => {
      if (!s.scatter) return;
      for (let i = 0; i < s.x.length; i++) {
        const sx = xScale.toPx(s.x[i]), sy = yScale.toPx(s.y[i]);
        const d = Math.hypot(sx - px, sy - py);
        if (d < bestD) { bestD = d; scatterHit = { s, x: s.x[i], y: s.y[i], px: sx }; }
      }
    });

    const lines = [];
    let headerVal = xVal;
    const crossPx = scatterHit ? scatterHit.px : px;
    p.crossLine.setAttribute('x1', crossPx); p.crossLine.setAttribute('x2', crossPx);
    p.crossG.setAttribute('display', '');

    if (scatterHit) {
      lines.push({ color: scatterHit.s.color, text: `${scatterHit.s.label}: ${fmtValue(scatterHit.y, scatterHit.s.fmt)}` });
      headerVal = scatterHit.x;
    } else {
      p.spec.series.forEach(s => {
        if (s.scatter) return;
        const i = nearestIndex(s.x, xVal);
        if (i < 0) return;
        lines.push({ color: s.color, text: `${s.label}: ${fmtValue(s.y[i], s.fmt)}` });
      });
    }

    const header = p.group.type === 'time' ? fmtDate(headerVal, true) : `AOD_om: ${headerVal.toFixed(3)}`;
    renderTooltip(header, lines, clientX, clientY);
  }

  function renderTooltip(header, lines, clientX, clientY) {
    let html = `<div class="tt-head">${header}</div>`;
    lines.forEach(l => {
      html += `<div class="tt-row"><span class="tt-dot" style="background:${l.color}"></span>${l.text}</div>`;
    });
    tooltip.innerHTML = html;
    tooltip.style.display = 'block';
    const contRect = container.getBoundingClientRect();
    const left = clientX - contRect.left + 14;
    const top = clientY - contRect.top + 14;
    const maxLeft = container.clientWidth - 190;
    tooltip.style.left = Math.min(left, Math.max(0, maxLeft)) + 'px';
    tooltip.style.top = top + 'px';
  }

  panels.forEach(createPanelDom);
  renderAll();

  let resizeRaf = null;
  function scheduleResize() {
    if (resizeRaf) return;
    resizeRaf = requestAnimationFrame(() => { resizeRaf = null; renderAll(); });
  }
  window.addEventListener('resize', scheduleResize);
  if (window.ResizeObserver) new ResizeObserver(scheduleResize).observe(container);
})();
"""


SGT_TZ = timezone(SGT_OFFSET.to_pytimedelta())


def _fmt_sgt(t: pd.Timestamp) -> str:
    """Naive-UTC timestamp -> 'YYYY-MM-DD HH:MM SGT'."""
    return (t + SGT_OFFSET).strftime("%Y-%m-%d %H:%M SGT")


def _fmt_both(t: pd.Timestamp) -> str:
    return f"{t.strftime('%H:%MZ')} ({_fmt_sgt(t)})"


def render_bias_status(bias: dict | None) -> str:
    """Header line saying whether the linear lines are bias-corrected, and how."""
    if bias is None:
        return ""
    if bias["status"] != "applied":
        return (
            '  <div class="meta bias">Bias correction <b>unavailable</b> '
            f'({bias["reason"]}) &mdash; linear lines show the uncorrected regression.</div>\n'
        )
    return (
        '  <div class="meta bias">Bias correction (linear lines only; ordinal bands uncorrected): '
        f'offset <b>{bias["offset_mean"]:+.1f}</b> &micro;g/m&sup3; on the mean and '
        f'<b>{bias["offset_max"]:+.1f}</b> &micro;g/m&sup3; on the max, from {bias["n_points"]} '
        f'forecast step(s) with a station reading, applied unchanged to every step. '
        f'Latest reading {_fmt_sgt(bias["latest_reading"])}.</div>\n'
    )


def render_html(spec: dict, generated_at: datetime, bias: dict | None = None, cycle_ref: pd.Timestamp | None = None) -> str:
    spec_json = json.dumps(spec, separators=(",", ":"))
    engine_js = (
        _ENGINE_JS
        .replace("__CHART_SPEC_JSON__", spec_json)
        .replace("__GRIDLINE__", GRIDLINE)
        .replace("__MUTED__", MUTED)
        .replace("__INK__", INK)
    )
    model_details = render_model_details(bias)
    bias_html = render_bias_status(bias)
    cycle_html = (
        f" &middot; CAMS cycle {_fmt_both(cycle_ref)}" if cycle_ref is not None else ""
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Singapore PM2.5 5-day forecast</title>
<style>
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: {SURFACE}; color: {INK};
          max-width: 1000px; margin: 0 auto; padding: 24px 16px 48px; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.1rem; margin: 32px 0 12px; }}
  .meta {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 24px; }}
  .meta.bias {{ margin-top: -12px; }}
  .hint {{ color: {MUTED}; font-size: 0.8rem; margin-bottom: 12px; }}
  #chart {{ position: relative; width: 100%; border-radius: 8px; border: 1px solid {GRIDLINE};
            background: {SURFACE}; padding: 8px 4px 4px; }}
  #chart .panel svg {{ user-select: none; }}
  #tooltip {{ position: absolute; display: none; pointer-events: none; z-index: 10;
              background: rgba(11,11,11,0.92); color: #fff; font-size: 11px; line-height: 1.5;
              padding: 6px 9px; border-radius: 6px; white-space: nowrap; box-shadow: 0 2px 8px rgba(0,0,0,0.25); }}
  #tooltip .tt-head {{ font-weight: 600; margin-bottom: 2px; }}
  #tooltip .tt-row {{ display: flex; align-items: center; gap: 5px; }}
  #tooltip .tt-dot {{ width: 7px; height: 7px; border-radius: 2px; display: inline-block; flex: none; }}
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
    &middot; Generated {generated_at.astimezone(SGT_TZ).strftime('%Y-%m-%d %H:%M SGT')}{cycle_html}
    &middot; All times shown in SGT (UTC+8)
  </div>
{bias_html}
  <div class="hint">Drag to pan, scroll or pinch to zoom, double-click/double-tap to reset, hover for exact values.</div>

  <div id="chart"><div id="tooltip"></div></div>
  <script>
{engine_js}
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
    parser.add_argument("--no-bias-correction", action="store_true",
                        help="skip fetching station readings; show the raw linear forecast")
    parser.add_argument("--observations-file", type=Path,
                        help="read data.gov.sg /pm25 items from this JSON file instead of calling the API (testing)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = load_forecast(args.forecast_nc)
    print(f"Loaded {len(df)} forecast timesteps: {df['valid_time'].min()} to {df['valid_time'].max()}")

    df = compute_predictions(df)
    print(df[["valid_time", "aod_om", "pm25_linear", "pm25_band", "pm25_max_linear", "pm25_max_band"]].to_string(index=False))

    bias = None
    if not args.no_bias_correction:
        try:
            if args.observations_file:
                obs = items_to_observations(json.loads(args.observations_file.read_text()))
            else:
                obs = fetch_observations(df["valid_time"].min(), utc_now_naive())
        except Exception as e:  # never let a readings problem block the forecast build
            print(f"Could not get station readings ({e.__class__.__name__}: {e}) -- using uncorrected forecast")
            obs = None
            bias = {"status": "unavailable", "reason": f"station readings could not be fetched: {e.__class__.__name__}"}
        if bias is None:
            df, bias = apply_bias_correction(df, obs)
        if bias["status"] == "applied":
            print(
                f"Bias correction: mean {bias['offset_mean']:+.2f}, max {bias['offset_max']:+.2f} ug/m3 from "
                f"{bias['n_points']} matched step(s)"
            )
        else:
            print(f"Bias correction unavailable: {bias['reason']}")

    training_sample = load_training_sample(args.training_table)
    print(f"Loaded {len(training_sample)} historical samples for the regression-result panels")

    spec = build_chart_spec(df, training_sample, bias)
    html = render_html(spec, datetime.now(timezone.utc), bias, cycle_ref=df["valid_time"].min())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
