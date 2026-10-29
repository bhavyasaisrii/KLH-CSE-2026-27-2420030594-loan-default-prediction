"""Expected-value threshold analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def threshold_economics(y_true: pd.Series, probabilities: np.ndarray, *, lost_interest: float = 1000.0,
                        default_loss: float = 5000.0, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    thresholds = thresholds if thresholds is not None else np.arange(0.05, 1.0, 0.05)
    rows = []
    for threshold in thresholds:
        accept = probabilities < threshold
        false_positive = accept & (y_true.to_numpy() == 1)
        false_negative = (~accept) & (y_true.to_numpy() == 0)
        rows.append({"threshold": threshold, "accepted": int(accept.sum()),
                     "false_negatives_defaulted": int(false_positive.sum()),
                     "false_positives_rejected": int(false_negative.sum()),
                     "expected_net_value": float(-false_positive.sum() * default_loss - false_negative.sum() * lost_interest)})
    return pd.DataFrame(rows)
