"""Leakage-safe model training, comparison, and probability calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelResult:
    name: str
    estimator: Any
    probabilities: np.ndarray
    metrics: dict[str, float]


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                   ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])


def _models() -> dict[str, Any]:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=5,
                                                   class_weight="balanced_subsample", random_state=42, n_jobs=1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
                                                                   max_leaf_nodes=15, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                           eval_metric="logloss", random_state=42)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                             verbosity=-1, random_state=42)
    except ImportError:
        pass
    return models


def evaluate(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {"roc_auc": float(roc_auc_score(y_true, probabilities)),
            "pr_auc": float(average_precision_score(y_true, probabilities)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "brier_score": float(brier_score_loss(y_true, probabilities))}


def train_models(features: pd.DataFrame, target: pd.Series, folds: int = 5) -> list[ModelResult]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    results = []
    for name, classifier in _models().items():
        pipeline = Pipeline([("preprocess", build_preprocessor(features)), ("model", classifier)])
        probabilities = cross_val_predict(pipeline, features, target, cv=splitter,
                                          method="predict_proba", n_jobs=None)[:, 1]
        pipeline.fit(features, target)
        results.append(ModelResult(name, pipeline, probabilities, evaluate(target, probabilities)))
    return results


def calibrate(estimator: Any, features: pd.DataFrame, target: pd.Series, method: str = "isotonic") -> Any:
    """Fit calibration on supplied training data; production should use a separate calibration split."""
    calibrated = CalibratedClassifierCV(estimator, method=method, cv=5)
    calibrated.fit(features, target)
    return calibrated


def calibration_points(target: pd.Series, probabilities: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return empirical default rate and mean prediction for a reliability plot."""
    observed, predicted = calibration_curve(target, probabilities, n_bins=bins, strategy="quantile")
    return predicted, observed


def tune_logistic(features: pd.DataFrame, target: pd.Series, folds: int = 5) -> Any:
    """Perform a reproducible grid search with preprocessing inside each fold."""
    pipeline = Pipeline([("preprocess", build_preprocessor(features)),
                         ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    search = GridSearchCV(pipeline, {"model__C": [0.01, 0.1, 1.0, 10.0]},
                          scoring="roc_auc", cv=StratifiedKFold(folds, shuffle=True, random_state=42), n_jobs=None)
    search.fit(features, target)
    return search
