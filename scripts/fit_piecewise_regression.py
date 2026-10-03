#!/usr/bin/env python3
"""Piecewise regression of pm25_max on aod_total, split by haze.gov.sg's
1-hour PM2.5 bands.

Official bands (https://www.haze.gov.sg/resources/1-hr-pm2.5-readings):
    0-55 ug/m3    Normal
    56-150 ug/m3  Elevated
    151-250 ug/m3 High
    >=251 ug/m3   Very High

Fits a separate pm25_max ~ aod_total OLS within each band (rather than one
global slope) to check whether the AOD-PM2.5 relationship holds steady
across severity levels or changes shape in the tail -- the global fit's
residuals (fan-shaped, heavy right tail) suggested it might not.
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

BAND_EDGES = [-0.001, 55, 150, 250, np.inf]
BAND_LABELS = ["Normal", "Elevated", "High", "Very High"]
MIN_N_FOR_FIT = 3  # OLS with an intercept needs >=2; 3 leaves >=1 residual df


def categorize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pm25_category"] = pd.cut(df["pm25_max"], bins=BAND_EDGES, labels=BAND_LABELS)
    return df


def fit_band(df_band: pd.DataFrame):
    X = sm.add_constant(df_band[["aod_total"]])
    y = df_band["pm25_max"]
    return sm.OLS(y, X).fit()


def summarize(models: dict) -> pd.DataFrame:
    rows = []
    for label, model in models.items():
        rows.append(
            {
                "category": label,
                "n": int(model.nobs),
                "intercept": model.params["const"],
                "slope (aod_total)": model.params["aod_total"],
                "slope p-value": model.pvalues["aod_total"],
                "R2": model.rsquared,
            }
        )
    return pd.DataFrame(rows).set_index("category")


def plot_facets(df: pd.DataFrame, models: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), facecolor=SURFACE)

    for ax, label in zip(axes.flat, BAND_LABELS):
        sub = df[df["pm25_category"] == label]
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        n = len(sub)
        alpha = max(0.05, min(0.4, 2000 / max(n, 1)))
        ax.scatter(
            sub["aod_total"], sub["pm25_max"], s=10, c=COLOR, alpha=alpha, linewidths=0, zorder=2
        )

        model = models.get(label)
        if model is not None:
            x_grid = np.linspace(sub["aod_total"].min(), sub["aod_total"].max(), 100)
            X_grid = sm.add_constant(pd.DataFrame({"aod_total": x_grid}))
            pred = model.get_prediction(X_grid)
            mean = pred.predicted_mean
            ci_low, ci_high = pred.conf_int().T
            ax.fill_between(x_grid, ci_low, ci_high, color=BAND_FILL, alpha=0.15, zorder=3)
            ax.plot(x_grid, mean, color=LINE_COLOR, linewidth=2, zorder=4)
            stats = (
                f"n={n}, slope={model.params['aod_total']:.1f}, "
                f"R²={model.rsquared:.2f}, p={model.pvalues['aod_total']:.1g}"
            )
        else:
            stats = f"n={n}, too few points to fit"

        ax.set_title(f"{label}  ({stats})", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Total AOD", color=INK, fontsize=9)
        ax.set_ylabel("PM2.5 max (µg/m³)", color=INK, fontsize=9)

    fig.suptitle(
        "pm25_max ~ aod_total, fit separately per haze.gov.sg PM2.5 band",
        color=INK, fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    print(f"Wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=Path("data/training_table.csv"), type=Path)
    parser.add_argument("--output", default=Path("output/piecewise_regression.png"), type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.table)
    df = categorize(df)

    print("Rows per haze.gov.sg PM2.5 band:")
    print(df["pm25_category"].value_counts().reindex(BAND_LABELS).to_string())
    print()

    models = {}
    for label in BAND_LABELS:
        sub = df[df["pm25_category"] == label]
        if len(sub) < MIN_N_FOR_FIT:
            print(f"WARNING: {label} has only {len(sub)} rows -- skipping fit (need >= {MIN_N_FOR_FIT}).")
            continue
        models[label] = fit_band(sub)

    summary = summarize(models)
    print("Per-band regression summary:")
    print(summary.to_string(float_format=lambda v: f"{v:.4g}"))
    print()

    global_model = fit_band(df)
    print(
        f"For reference, the single global fit (all bands pooled): "
        f"slope={global_model.params['aod_total']:.2f}, R²={global_model.rsquared:.3f}, "
        f"n={int(global_model.nobs)}"
    )
    print()
    for label in BAND_LABELS:
        n = int((df["pm25_category"] == label).sum())
        if n < 30:
            print(f"CAUTION: '{label}' band has only n={n} -- treat its slope/R² as unreliable, not a finding.")

    plot_facets(df, models, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
