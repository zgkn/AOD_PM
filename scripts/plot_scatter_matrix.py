#!/usr/bin/env python3
"""Scatter plot matrix for total AOD, PM2.5 max, wind speed, and PBLH.

Reads the joined training table (data/training_table.csv by default) and
renders a 4x4 scatter matrix: off-diagonal panels are pairwise scatters,
diagonal panels are each variable's own distribution. Also prints the
pairwise Pearson correlations to stdout.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Reference palette (see dataviz skill): sequential default blue, light
# chart surface, hairline gridlines, muted/primary ink.
COLOR = "#2a78d6"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
MUTED = "#898781"
INK = "#0b0b0b"

COLUMNS = ["aod_total", "pm25_max", "wind10m", "pblh"]
LABELS = {
    "aod_total": "Total AOD",
    "pm25_max": "PM2.5 max (µg/m³)",
    "wind10m": "Wind speed (m/s)",
    "pblh": "PBLH (m)",
}


def plot_scatter_matrix(df: pd.DataFrame, output: Path) -> None:
    n = len(COLUMNS)
    fig, axes = plt.subplots(n, n, figsize=(10, 10), facecolor=SURFACE)

    # 34k+ points overplot badly as opaque dots; small markers + low alpha
    # make the point-cloud density legible instead of a solid blue blob.
    n_points = len(df)
    marker_size = 6
    alpha = max(0.03, min(0.25, 4000 / n_points))

    for i, row_col in enumerate(COLUMNS):
        for j, col_col in enumerate(COLUMNS):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=7, length=0)
            ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)

            if i == j:
                ax.hist(df[row_col], bins=40, color=COLOR, alpha=0.5, zorder=2)
            else:
                ax.scatter(
                    df[col_col],
                    df[row_col],
                    s=marker_size,
                    c=COLOR,
                    alpha=alpha,
                    linewidths=0,
                    zorder=2,
                )

            if i == n - 1:
                ax.set_xlabel(LABELS[col_col], color=INK, fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(LABELS[row_col], color=INK, fontsize=9)
            else:
                ax.set_yticklabels([])

    fig.suptitle(
        "Total AOD, PM2.5 max, wind speed, and PBLH -- pairwise relationships",
        color=INK,
        fontsize=12,
    )
    fig.text(
        0.5, 0.955,
        f"n = {n_points:,} rows; diagonal = each variable's own distribution",
        ha="center", color=MUTED, fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    print(f"Wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=Path("data/training_table.csv"), type=Path)
    parser.add_argument("--output", default=Path("output/scatter_matrix.png"), type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.table, usecols=["timestamp"] + COLUMNS)

    print("Pearson correlation matrix:")
    print(df[COLUMNS].rename(columns=LABELS).corr().round(3).to_string())

    plot_scatter_matrix(df, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
