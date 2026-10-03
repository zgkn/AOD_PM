#!/usr/bin/env python3
"""Ordinal logistic regression of the haze.gov.sg PM2.5 band on aod_total.

Buckets pm25_mean into the official 1-hour PM2.5 bands (Normal 0-55,
Elevated 56-150, High 151-250, Very High >=251 ug/m3 --
https://www.haze.gov.sg/resources/1-hr-pm2.5-readings) and fits a
proportional-odds ordinal logit: P(band <= k | aod_total). Unlike plain
multinomial logistic regression, this respects the bands' order and
shares the aod_total slope across all category thresholds, which matters
given how few rows fall in High/Very High.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

COLOR = "#2a78d6"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
MUTED = "#898781"
INK = "#0b0b0b"

# Ordinal-ramp steps (light -> dark blue), one per band, in severity order.
BAND_COLORS = {
    "Normal": "#86b6ef",
    "Elevated": "#3987e5",
    "High": "#1c5cab",
    "Very High": "#0d366b",
}

BAND_EDGES = [-0.001, 55, 150, 250, np.inf]
BAND_LABELS = ["Normal", "Elevated", "High", "Very High"]


def categorize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pm25_category"] = pd.Categorical(
        pd.cut(df["pm25_mean"], bins=BAND_EDGES, labels=BAND_LABELS),
        categories=BAND_LABELS, ordered=True,
    )
    return df


def predict_probs(model_result, X: pd.DataFrame) -> pd.DataFrame:
    # OrderedModel.predict() returns integer category codes (0..k-1) as
    # columns, in the categorical's category order -- relabel to BAND_LABELS.
    probs = model_result.predict(X)
    probs.columns = BAND_LABELS
    return probs


def plot_predicted_probabilities(model_result, df: pd.DataFrame, output: Path) -> None:
    x_grid = np.linspace(df["aod_total"].min(), df["aod_total"].max(), 200)
    probs = predict_probs(model_result, pd.DataFrame({"aod_total": x_grid}))

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for label in BAND_LABELS:
        ax.plot(x_grid, probs[label], color=BAND_COLORS[label], linewidth=2.5, label=label, zorder=2)

    ax.set_xlabel("Total AOD", color=INK, fontsize=10)
    ax.set_ylabel("Predicted probability", color=INK, fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        "Predicted PM2.5 band probability vs AOD (ordinal logistic)", color=INK, fontsize=12, loc="left"
    )
    legend = ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=Path("data/training_table.csv"), type=Path)
    parser.add_argument("--output", default=Path("output/ordinal_logistic.png"), type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.table)
    df = categorize(df)

    print("Rows per PM2.5 band (pm25_mean):")
    print(df["pm25_category"].value_counts().reindex(BAND_LABELS).to_string())
    print()

    model = OrderedModel(df["pm25_category"], df[["aod_total"]], distr="logit")
    result = model.fit(method="bfgs", disp=False)
    print(result.summary())
    print()

    # Classification check: predicted band = argmax predicted probability.
    probs = predict_probs(result, df[["aod_total"]])
    predicted = probs.idxmax(axis=1)
    actual = df["pm25_category"].astype(str)
    accuracy = (predicted.values == actual.values).mean()
    print(f"n = {len(df):,}")
    print(f"Argmax classification accuracy: {accuracy:.3f}")
    print(
        "(A naive 'always predict Normal' baseline would score "
        f"{(actual == 'Normal').mean():.3f} -- compare against that, not against 1.0.)"
    )
    print()
    for label in BAND_LABELS:
        n = int((df["pm25_category"] == label).sum())
        if n < 30:
            print(f"CAUTION: '{label}' has only n={n} in the data -- its predicted probabilities are not well constrained.")

    plot_predicted_probabilities(result, df, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
