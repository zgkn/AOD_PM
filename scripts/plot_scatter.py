#!/usr/bin/env python3
"""Single scatter plot of two training-table columns, with Pearson r printed.

Defaults to pm25_mean (y) vs aod_om (x).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

COLOR = "#2a78d6"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
MUTED = "#898781"
INK = "#0b0b0b"

LABELS = {
    "aod_total": "Total AOD",
    "aod_om": "Organic matter AOD",
    "pm25_max": "PM2.5 max (µg/m³)",
    "pm25_mean": "PM2.5 mean (µg/m³)",
    "wind10m": "Wind speed (m/s)",
    "pblh": "PBLH (m)",
    "ventilation_rate": "Ventilation rate (m²/s)",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=Path("data/training_table.csv"), type=Path)
    parser.add_argument("--x", default="aod_om")
    parser.add_argument("--y", default="pm25_mean")
    parser.add_argument("--output", default=Path("output/scatter.png"), type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.table, usecols=[args.x, args.y])
    r, p = stats.pearsonr(df[args.x], df[args.y])
    print(f"n = {len(df):,}")
    print(f"Pearson r ({args.x}, {args.y}) = {r:.4f}  (R² = {r**2:.4f}, p = {p:.3g})")

    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    n_points = len(df)
    alpha = max(0.03, min(0.3, 4000 / n_points))
    ax.scatter(df[args.x], df[args.y], s=8, c=COLOR, alpha=alpha, linewidths=0, zorder=2)

    x_label = LABELS.get(args.x, args.x)
    y_label = LABELS.get(args.y, args.y)
    ax.set_xlabel(x_label, color=INK, fontsize=10)
    ax.set_ylabel(y_label, color=INK, fontsize=10)
    ax.set_title(
        f"{y_label} vs {x_label}  (n={n_points:,}, r={r:.3f}, R²={r**2:.3f})",
        color=INK, fontsize=12, loc="left",
    )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor=SURFACE)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
