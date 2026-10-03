#!/usr/bin/env python3
"""Regression of pm25_max on aod_total, fit separately for day and night.

Singapore sits on the equator, so sunrise/sunset barely moves seasonally --
a fixed boundary is a safe approximation. The 8 reanalysis UTC timestamps
split exactly in half by SGT (UTC+8):
    day:   00/03/06/09 UTC = 08:00/11:00/14:00/17:00 SGT
    night: 12/15/18/21 UTC = 20:00/23:00/02:00/05:00 SGT

PBLH/ventilation_rate are deliberately left out -- this checks whether the
plain AOD-PM2.5 relationship itself differs between a well-mixed daytime
boundary layer and a shallow, stable nighttime one, not whether adding
PBLH as a covariate helps.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

COLOR = "#2a78d6"
LINE_COLOR = "#1c5cab"
BAND_FILL = "#2a78d6"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
MUTED = "#898781"
INK = "#0b0b0b"

DAY_UTC_HOURS = {0, 3, 6, 9}
NIGHT_UTC_HOURS = {12, 15, 18, 21}


def categorize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    hour = df["timestamp"].dt.hour
    df["period"] = np.where(hour.isin(DAY_UTC_HOURS), "Day", "Night")
    return df


def fit(df_sub: pd.DataFrame, log: bool):
    if log:
        X = sm.add_constant(np.log(df_sub[["aod_total"]]))
        y = np.log(df_sub["pm25_max"])
    else:
        X = sm.add_constant(df_sub[["aod_total"]])
        y = df_sub["pm25_max"]
    return sm.OLS(y, X).fit()


def plot_facets(df: pd.DataFrame, models: dict, log: bool, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), facecolor=SURFACE)
    x_label = "log(Total AOD)" if log else "Total AOD"
    y_label = "log(PM2.5 max)" if log else "PM2.5 max (µg/m³)"

    for ax, period in zip(axes, ["Day", "Night"]):
        sub = df[df["period"] == period]
        x_vals = np.log(sub["aod_total"]) if log else sub["aod_total"]
        y_vals = np.log(sub["pm25_max"]) if log else sub["pm25_max"]

        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        n = len(sub)
        alpha = max(0.04, min(0.25, 4000 / max(n, 1)))
        ax.scatter(x_vals, y_vals, s=6, c=COLOR, alpha=alpha, linewidths=0, zorder=2)

        model = models[period]
        x_grid = np.linspace(x_vals.min(), x_vals.max(), 100)
        X_grid = sm.add_constant(pd.DataFrame({"aod_total": x_grid}))
        pred = model.get_prediction(X_grid)
        mean = pred.predicted_mean
        ci_low, ci_high = pred.conf_int().T
        ax.fill_between(x_grid, ci_low, ci_high, color=BAND_FILL, alpha=0.15, zorder=3)
        ax.plot(x_grid, mean, color=LINE_COLOR, linewidth=2, zorder=4)

        stats = (
            f"n={n:,}, slope={model.params['aod_total']:.3g}, "
            f"R²={model.rsquared:.3f}, p={model.pvalues['aod_total']:.1g}"
        )
        ax.set_title(f"{period}  ({stats})", color=INK, fontsize=10, loc="left")
        ax.set_xlabel(x_label, color=INK, fontsize=9)
        ax.set_ylabel(y_label, color=INK, fontsize=9)

    fig.suptitle("pm25_max ~ aod_total, fit separately for day and night (SGT)", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    print(f"Wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=Path("data/training_table.csv"), type=Path)
    parser.add_argument("--output", default=Path("output/daynight_regression.png"), type=Path)
    parser.add_argument("--log", action="store_true", help="Fit log(pm25_max) ~ log(aod_total) instead.")
    args = parser.parse_args()

    df = pd.read_csv(args.table)
    df = categorize(df)

    print("Rows per period:")
    print(df["period"].value_counts().reindex(["Day", "Night"]).to_string())
    print()

    models = {period: fit(df[df["period"] == period], args.log) for period in ["Day", "Night"]}

    print("Per-period regression summary:")
    for period, model in models.items():
        print(
            f"  {period:>5}: n={int(model.nobs):,}  intercept={model.params['const']:.3g}  "
            f"slope(aod_total)={model.params['aod_total']:.3g}  R²={model.rsquared:.3f}  "
            f"p={model.pvalues['aod_total']:.1g}"
        )

    pooled = fit(df, args.log)
    print(
        f"\nFor reference, pooled day+night: slope={pooled.params['aod_total']:.3g}, "
        f"R²={pooled.rsquared:.3f}, n={int(pooled.nobs):,}"
    )

    plot_facets(df, models, args.log, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
