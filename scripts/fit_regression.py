#!/usr/bin/env python3
"""OLS regression of pm25_max against one or more predictor columns.

Fits pm25_max ~ predictors (with intercept) on the training table, prints
the full statsmodels summary (coefficients, std errors, p-values, R^2),
and saves a residuals-vs-fitted + Q-Q diagnostic plot so the fit can be
checked rather than taken on faith.

--log fits log(pm25_max) ~ log(predictors) instead: the target and
aod_total/ventilation_rate are all strictly positive and right-skewed
(see the earlier residual diagnostics -- skew 6.2, kurtosis 137, fanning
residuals), which a log-log fit is the standard first move against. The
fitted slope on a logged predictor is then an elasticity: % change in
pm25_max per % change in that predictor.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

COLOR = "#2a78d6"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
MUTED = "#898781"
INK = "#0b0b0b"

DEFAULT_PREDICTORS = ["aod_total", "ventilation_rate"]
TARGET = "pm25_max"


def fit(df: pd.DataFrame, predictors: list, log: bool):
    if log:
        X = sm.add_constant(np.log(df[predictors]))
        y = np.log(df[TARGET])
    else:
        X = sm.add_constant(df[predictors])
        y = df[TARGET]
    return sm.OLS(y, X).fit()


def plot_diagnostics(model, predictors: list, log: bool, output: Path) -> None:
    fitted = model.fittedvalues
    resid = model.resid
    target_label = f"log({TARGET})" if log else TARGET

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=SURFACE)

    ax = axes[0]
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
    ax.scatter(fitted, resid, s=6, c=COLOR, alpha=0.08, linewidths=0, zorder=2)
    ax.set_xlabel(f"Fitted {target_label}", color=INK, fontsize=9)
    ax.set_ylabel("Residual", color=INK, fontsize=9)
    ax.set_title("Residuals vs fitted", color=INK, fontsize=10)

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    sm.qqplot(resid, line="45", ax=ax, markerfacecolor=COLOR, markeredgecolor=COLOR, alpha=0.3, markersize=3)
    ax.get_lines()[1].set_color(MUTED)
    ax.set_title("Q-Q plot of residuals", color=INK, fontsize=10)
    ax.set_xlabel(ax.get_xlabel(), color=INK, fontsize=9)
    ax.set_ylabel(ax.get_ylabel(), color=INK, fontsize=9)

    pred_label = ", ".join(f"log({p})" for p in predictors) if log else " + ".join(predictors)
    fig.suptitle(f"{target_label} ~ {pred_label} -- residual diagnostics", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    print(f"Wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=Path("data/training_table.csv"), type=Path)
    parser.add_argument("--output", default=Path("output/regression_diagnostics.png"), type=Path)
    parser.add_argument(
        "--predictors",
        default=",".join(DEFAULT_PREDICTORS),
        help="Comma-separated predictor column names.",
    )
    parser.add_argument(
        "--log", action="store_true",
        help="Fit log(pm25_max) ~ log(predictors) instead of the raw-value fit.",
    )
    args = parser.parse_args()
    predictors = [c.strip() for c in args.predictors.split(",") if c.strip()]

    df = pd.read_csv(args.table)
    model = fit(df, predictors, args.log)

    print(model.summary())
    print()
    print(f"n = {len(df):,}")
    print(f"Residual skew: {model.resid.skew():.2f}  (0 = symmetric; the raw-value fit's residuals had skew 6.2)")

    plot_diagnostics(model, predictors, args.log, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
