#!/usr/bin/env python3
"""Scatter plot matrix for columns of the training table.

Reads the joined training table (data/training_table.csv by default) and
renders an NxN scatter matrix: off-diagonal panels are pairwise scatters,
diagonal panels are each variable's own distribution. Also prints the
pairwise Pearson correlations to stdout.

`ventilation_rate` (= pblh * wind10m, the standard ventilation coefficient
used in air-quality meteorology: a deep, windy boundary layer disperses
pollutants; a shallow, calm one lets them accumulate) is computed on the
fly and available as a plot column alongside the table's own columns.
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

DEFAULT_COLUMNS = ["aod_total", "pm25_max", "wind10m", "pblh"]
LABELS = {
    "aod_total": "Total AOD",
    "aod_om": "Organic matter AOD",
    "pm25_max": "PM2.5 max (µg/m³)",
    "pm25_mean": "PM2.5 mean (µg/m³)",
    "wind10m": "Wind speed (m/s)",
    "pblh": "PBLH (m)",
    "ventilation_rate": "Ventilation rate (m²/s)",
}


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["ventilation_rate"] = df["pblh"] * df["wind10m"]
    return df


def plot_scatter_matrix(df: pd.DataFrame, columns: list, output: Path) -> None:
    n = len(columns)
    fig, axes = plt.subplots(n, n, figsize=(10, 10), facecolor=SURFACE)

    # 34k+ points overplot badly as opaque dots; small markers + low alpha
    # make the point-cloud density legible instead of a solid blue blob.
    n_points = len(df)
    marker_size = 6
    alpha = max(0.03, min(0.25, 4000 / n_points))

    for i, row_col in enumerate(columns):
        for j, col_col in enumerate(columns):
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

    title = ", ".join(LABELS.get(c, c) for c in columns)
    fig.suptitle(f"{title} -- pairwise relationships", color=INK, fontsize=12)
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
    parser.add_argument(
        "--columns",
        default=",".join(DEFAULT_COLUMNS),
        help="Comma-separated column names to plot (table columns, plus ventilation_rate).",
    )
    args = parser.parse_args()
    columns = [c.strip() for c in args.columns.split(",") if c.strip()]

    df = pd.read_csv(args.table)
    df = add_derived_columns(df)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"Unknown column(s): {missing}. Available: {sorted(df.columns)}")

    print("Pearson correlation matrix:")
    print(df[columns].rename(columns=LABELS).corr().round(3).to_string())

    plot_scatter_matrix(df, columns, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
