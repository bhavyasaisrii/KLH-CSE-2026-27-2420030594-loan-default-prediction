"""Reproducible exploratory analysis for the loan-risk project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(fig: plt.Figure, destination: Path, name: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_eda(features: pd.DataFrame, target: pd.Series, output: str | Path) -> dict[str, pd.DataFrame]:
    """Save applicable EDA figures and observations derived from the supplied frame."""
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    frame = features.copy()
    frame["default"] = target.to_numpy()
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]

    target_counts = target.value_counts().rename_axis("default").reset_index(name="count")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=frame, x="default", ax=ax, color="#2a9d8f")
    ax.set(title="Target distribution", xlabel="Default", ylabel="Records")
    _save(fig, destination, "target_distribution.png")

    if numeric:
        fig, axes = plt.subplots(len(numeric), 1, figsize=(8, max(3, 2.8 * len(numeric))))
        axes = [axes] if len(numeric) == 1 else axes
        for axis, column in zip(axes, numeric):
            sns.histplot(data=frame, x=column, hue="default", stat="density", common_norm=False,
                         element="step", ax=axis)
            axis.set_title(f"Distribution of {column} by default status")
        _save(fig, destination, "numeric_distributions.png")

        fig, ax = plt.subplots(figsize=(max(6, len(numeric) * 1.1), 5))
        sns.heatmap(frame[numeric + ["default"]].corr(numeric_only=True), cmap="vlag", center=0,
                    annot=len(numeric) <= 10, fmt=".2f", ax=ax)
        ax.set_title("Numeric feature correlation")
        _save(fig, destination, "correlation_heatmap.png")

        fig, axes = plt.subplots(1, min(4, len(numeric)), figsize=(14, 4))
        axes = [axes] if len(numeric) == 1 else axes
        for axis, column in zip(axes, numeric[:4]):
            sns.boxplot(data=frame, x="default", y=column, ax=axis, color="#e9c46a")
            axis.set_title(column)
        _save(fig, destination, "numeric_boxplots.png")

        for column in numeric:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=frame, x="default", y=column, ax=ax, color="#f4a261")
            ax.set_title(f"{column} versus default")
            _save(fig, destination, f"relationship_{column}.png")

    categorical_summary = []
    for column in categorical:
        grouped = frame.groupby(column, dropna=False)["default"].agg(default_rate="mean", records="size")
        grouped = grouped.reset_index().sort_values("records", ascending=False).head(12)
        categorical_summary.append(grouped.assign(feature=column))
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=grouped, x="default_rate", y=column, ax=ax, color="#264653")
        ax.set(title=f"Default rate by {column}", xlabel="Default rate", ylabel=column)
        _save(fig, destination, f"default_rate_by_{column}.png")

    missing = features.isna().sum().rename("missing").reset_index().rename(columns={"index": "feature"})
    observations = [
        f"Records analysed: {len(features)}.",
        f"Features analysed: {features.shape[1]} ({len(numeric)} numeric, {len(categorical)} categorical).",
        f"Default records: {int(target.sum())}; non-default records: {int((target == 0).sum())}.",
        f"Missing cells: {int(features.isna().sum().sum())}; duplicate feature rows: {int(features.duplicated().sum())}.",
    ]
    if numeric:
        rates = frame.groupby("default")[numeric].mean().T
        for column in numeric:
            if 0 in rates and 1 in rates:
                observations.append(f"Mean {column}: default={rates.loc[column, 1]:.3g}, non-default={rates.loc[column, 0]:.3g}.")
    (destination / "eda_observations.txt").write_text("\n".join(observations) + "\n", encoding="utf-8")
    return {"target_distribution": target_counts, "missing_values": missing,
            "categorical_default_rates": pd.concat(categorical_summary, ignore_index=True) if categorical_summary else pd.DataFrame()}