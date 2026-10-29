"""Group fairness diagnostics for audit, not automatic approval decisions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fairness_metrics(frame: pd.DataFrame, group: str, target: str, prediction: str) -> pd.DataFrame:
    if group not in frame or target not in frame or prediction not in frame:
        raise KeyError(f"Expected columns: {group}, {target}, {prediction}")
    rows = []
    for value, subset in frame.groupby(group, dropna=False):
        actual = subset[target].to_numpy()
        predicted = subset[prediction].to_numpy()
        positives = actual == 1
        negatives = actual == 0
        rows.append({"group": value, "n": len(subset),
                     "demographic_parity": float(predicted.mean()),
                     "true_positive_rate": float(predicted[positives].mean()) if positives.any() else np.nan,
                     "false_positive_rate": float(predicted[negatives].mean()) if negatives.any() else np.nan})
    result = pd.DataFrame(rows)
    for metric in ["demographic_parity", "true_positive_rate", "false_positive_rate"]:
        result[f"{metric}_difference_vs_best"] = result[metric] - result[metric].max()
    return result
