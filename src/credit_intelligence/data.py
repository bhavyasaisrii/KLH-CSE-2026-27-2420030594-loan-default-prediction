"""Point-in-time data curation and target engineering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable, Sequence

import numpy as np
import pandas as pd

# These fields are observable only after funding or during collections.
POST_FUNDING_TERMS = (
    "last_payment", "next_payment", "recoveries", "collection", "total_rec_prncp",
    "total_rec_int", "out_prncp", "funded_amnt", "settlement", "hardship",
    "balance", "payment_plan", "debt_settlement",
)
DEFAULT_STATUSES = {"charged off", "default", "does not meet the credit policy"}
PAID_STATUSES = {"fully paid"}


@dataclass(frozen=True)
class CuratedData:
    features: pd.DataFrame
    target: pd.Series
    audit: pd.DataFrame


def vertically_merge(paths: Sequence[str | Path | IO[bytes] | IO[str]]) -> pd.DataFrame:
    """Read and concatenate datasets row-wise, retaining dataset provenance."""
    if not paths:
        raise ValueError("At least one dataset is required for a vertical merge")

    frames = []
    for index, path in enumerate(paths, start=1):
        frame = pd.read_csv(path).copy()
        source = getattr(path, "name", None) or str(path)
        if "source_dataset" not in frame.columns:
            frame.insert(0, "source_dataset", Path(source).name if source else f"dataset_{index}")
        frames.append(frame)

    return pd.concat(frames, axis=0, ignore_index=True, sort=False)


def audit_columns(columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        normalized = column.lower().replace("-", "_").replace(" ", "_")
        post_funding = any(term in normalized for term in POST_FUNDING_TERMS)
        rows.append({"column": column, "normalized": normalized, "post_funding": post_funding,
                     "decision": "DROP" if post_funding else "RETAIN"})
    return pd.DataFrame(rows)


def _synthetic_data(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(10.8, 0.55, rows).round(0)
    dti = np.clip(rng.normal(17, 8, rows), 0, 55)
    fico = np.clip(rng.normal(700, 45, rows), 580, 850).round()
    employment = rng.integers(0, 11, rows)
    purpose = rng.choice(["debt_consolidation", "home_improvement", "major_purchase", "car"], rows)
    state = rng.choice(["CA", "TX", "FL", "NY", "OH"], rows)
    log_odds = -2.5 + 0.065 * dti - 0.018 * (fico - 700) - 0.05 * employment + 0.3 * (income < 45000)
    probability = 1 / (1 + np.exp(-log_odds))
    default = rng.binomial(1, probability)
    return pd.DataFrame({"annual_inc": income, "dti": dti, "fico_score": fico,
                         "emp_length_years": employment, "purpose": purpose, "state": state,
                         "loan_status": np.where(default, "Charged Off", "Fully Paid"),
                         "last_payment_amount": rng.random(rows) * 1000})


def load_and_curate(path: str | Path | IO[bytes] | None = None, *, paths: Sequence[str | Path | IO[bytes] | IO[str]] | None = None,
                    seed: int = 42) -> CuratedData:
    """Load data and retain only fields available at application time.

    The target status itself is excluded from features. Unknown statuses are excluded
    rather than silently treating them as a good or bad outcome.
    """
    if paths is not None:
        frame = vertically_merge(paths)
    elif path:
        frame = pd.read_csv(path)
    else:
        frame = _synthetic_data(seed=seed)
    status_column = next((c for c in frame.columns if c.lower() in {"loan_status", "status", "target"}), None)
    if status_column is None:
        raise ValueError("Dataset must contain loan_status, status, or target")
    status = frame[status_column].astype(str).str.strip().str.lower()
    target = status.isin(DEFAULT_STATUSES).astype(int)
    known = status.isin(DEFAULT_STATUSES | PAID_STATUSES)
    frame = frame.loc[known].copy()
    target = target.loc[known].rename("default")
    audit = audit_columns(frame.columns)
    dropped = set(audit.loc[audit["post_funding"], "column"])
    feature_columns = [c for c in frame.columns if c not in dropped and c not in {status_column, "source_dataset"}]
    features = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    if features.empty or features.shape[1] == 0:
        raise ValueError("No point-in-time features remain after the leakage audit")
    return CuratedData(features=features.reset_index(drop=True), target=target.reset_index(drop=True), audit=audit)
