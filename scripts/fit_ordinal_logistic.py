#!/usr/bin/env python3
"""Ordinal logistic regression of the haze.gov.sg PM2.5 band on a predictor.

Buckets the target column (pm25_mean by default) into the official
1-hour PM2.5 bands (Normal 0-55, Elevated 56-150, High 151-250, Very High
>=251 ug/m3 -- https://www.haze.gov.sg/resources/1-hr-pm2.5-readings) and
fits a proportional-odds ordinal logit: P(band <= k | predictor). Unlike
plain multinomial logistic regression, this respects the bands' order and
shares the predictor's slope across all category thresholds, which
matters given how few rows fall in High/Very High.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
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
DEFAULT_TARGET = "pm25_mean"
DEFAULT_PREDICTOR = "aod_total"


def categorize(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.copy()
    df["pm25_category"] = pd.Categorical(
        pd.cut(df[target], bins=BAND_EDGES, labels=BAND_LABELS),
        categories=BAND_LABELS, ordered=True,
    )
    return df


def chronological_split(df: pd.DataFrame, train_frac: float):
    df = df.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def roc_auc_per_band(probs: pd.DataFrame, actual: pd.Series) -> tuple:
    """One-vs-rest ROC-AUC per band. A band with only one class present in
    this split (e.g. zero positives) can't have an AUC computed -- reported
    as None rather than silently dropped or faked."""
    per_band = {}
    for label in BAND_LABELS:
        y_true = (actual == label).astype(int)
        if y_true.nunique() < 2:
            per_band[label] = None
            continue
        per_band[label] = roc_auc_score(y_true, probs[label])
    valid = [v for v in per_band.values() if v is not None]
    macro = float(np.mean(valid)) if valid else None
    return per_band, macro


def evaluate(result, df_eval: pd.DataFrame, predictor: str) -> dict:
    probs = predict_probs(result, df_eval[[predictor]])
    predicted = probs.idxmax(axis=1)
    actual = df_eval["pm25_category"].astype(str)
    accuracy = (predicted.values == actual.values).mean()
    # Mean log-likelihood per row of the TRUE category's predicted probability.
    true_probs = probs.to_numpy()[np.arange(len(df_eval)), actual.map(BAND_LABELS.index).to_numpy()]
    mean_ll = np.log(np.clip(true_probs, 1e-12, None)).mean()
    auc_per_band, auc_macro = roc_auc_per_band(probs, actual)
    return {
        "n": len(df_eval),
        "accuracy": accuracy,
        "baseline_accuracy": (actual == "Normal").mean(),
        "mean_log_likelihood": mean_ll,
        "auc_per_band": auc_per_band,
        "auc_macro": auc_macro,
    }


def predict_probs(model_result, X: pd.DataFrame) -> pd.DataFrame:
    # OrderedModel.predict() returns integer category codes (0..k-1) as
    # columns, in the categorical's category order -- relabel to BAND_LABELS.
    probs = model_result.predict(X)
    probs.columns = BAND_LABELS
    return probs


def plot_predicted_probabilities(model_result, df: pd.DataFrame, predictor: str, output: Path) -> None:
    x_grid = np.linspace(df[predictor].min(), df[predictor].max(), 200)
    probs = predict_probs(model_result, pd.DataFrame({predictor: x_grid}))

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for label in BAND_LABELS:
        ax.plot(x_grid, probs[label], color=BAND_COLORS[label], linewidth=2.5, label=label, zorder=2)

    ax.set_xlabel(predictor, color=INK, fontsize=10)
    ax.set_ylabel("Predicted probability", color=INK, fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"Predicted PM2.5 band probability vs {predictor} (ordinal logistic)", color=INK, fontsize=12, loc="left"
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
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Column to bucket into PM2.5 bands.")
    parser.add_argument("--predictor", default=DEFAULT_PREDICTOR, help="Single predictor column.")
    parser.add_argument(
        "--validate", action="store_true",
        help="Fit on the first --train-frac of the timeline, evaluate on the held-out remainder.",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.8,
        help="Fraction of the (chronologically sorted) data used for training when --validate is set.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.table)
    df = categorize(df, args.target)

    print(f"Rows per PM2.5 band ({args.target}):")
    print(df["pm25_category"].value_counts().reindex(BAND_LABELS).to_string())
    print()

    if args.validate:
        train, val = chronological_split(df, args.train_frac)
        print(
            f"Chronological split: train={len(train):,} rows "
            f"({train['timestamp'].min()} to {train['timestamp'].max()}), "
            f"validation={len(val):,} rows ({val['timestamp'].min()} to {val['timestamp'].max()})"
        )
        print("Rows per band, train vs validation:")
        print(
            pd.DataFrame({
                "train": train["pm25_category"].value_counts().reindex(BAND_LABELS),
                "validation": val["pm25_category"].value_counts().reindex(BAND_LABELS),
            }).to_string()
        )
        print()

        model = OrderedModel(train["pm25_category"], train[[args.predictor]], distr="logit")
        result = model.fit(method="bfgs", disp=False)
        print(result.summary())
        print()

        train_metrics = evaluate(result, train, args.predictor)
        val_metrics = evaluate(result, val, args.predictor)
        for name, m in [("Train", train_metrics), ("Validation", val_metrics)]:
            macro_str = f"{m['auc_macro']:.3f}" if m["auc_macro"] is not None else "n/a"
            print(
                f"{name:>10}: n={m['n']:,}  accuracy={m['accuracy']:.3f} "
                f"(baseline={m['baseline_accuracy']:.3f})  macro ROC-AUC={macro_str}"
            )
            for label in BAND_LABELS:
                auc = m["auc_per_band"][label]
                auc_str = f"{auc:.3f}" if auc is not None else "n/a (only one class present in this split)"
                print(f"            ROC-AUC [{label} vs rest] = {auc_str}")
        print()
        if train_metrics["auc_macro"] is not None and val_metrics["auc_macro"] is not None:
            auc_gap = train_metrics["auc_macro"] - val_metrics["auc_macro"]
            print(f"Macro ROC-AUC gap (train - validation) = {auc_gap:.3f}")

        plot_df = df
    else:
        model = OrderedModel(df["pm25_category"], df[[args.predictor]], distr="logit")
        result = model.fit(method="bfgs", disp=False)
        print(result.summary())
        print()

        metrics = evaluate(result, df, args.predictor)
        print(f"n = {len(df):,}")
        print(f"Argmax classification accuracy: {metrics['accuracy']:.3f}")
        print(
            "(A naive 'always predict Normal' baseline would score "
            f"{metrics['baseline_accuracy']:.3f} -- compare against that, not against 1.0.)"
        )
        print()
        plot_df = df

    for label in BAND_LABELS:
        n = int((df["pm25_category"] == label).sum())
        if n < 30:
            print(f"CAUTION: '{label}' has only n={n} in the full data -- its predicted probabilities are not well constrained.")

    plot_predicted_probabilities(result, plot_df, args.predictor, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
