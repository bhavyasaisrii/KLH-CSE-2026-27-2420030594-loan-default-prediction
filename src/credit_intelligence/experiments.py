"""Research experiment runner: baselines, feature engineering, stacking, and artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .data import CuratedData, load_and_curate
from .eda import run_eda
from .models import build_preprocessor


def engineer_features(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = features.copy()
    created: list[str] = []
    names = {column.lower(): column for column in result.columns}
    pairs = [("dti", "annual_inc", "dti_to_income"), ("loan_amnt", "annual_inc", "loan_to_income"),
             ("installment", "annual_inc", "payment_to_income")]
    for numerator, denominator, name in pairs:
        if numerator in names and denominator in names:
            denominator_values = result[names[denominator]].replace(0, np.nan)
            result[name] = result[names[numerator]] / denominator_values
            created.append(name)
    return result.replace([np.inf, -np.inf], np.nan), created


def _models() -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=5,
                                                  class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=180, min_samples_leaf=5,
                                                  class_weight="balanced_subsample", random_state=42, n_jobs=1),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=140, learning_rate=.08,
                                                              max_leaf_nodes=15, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=180, max_depth=4, learning_rate=.05,
                                           eval_metric="logloss", random_state=42, n_jobs=1)
    except ImportError:
        pass
    return models


def _pipeline(features: pd.DataFrame, classifier: Any) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(features)), ("model", classifier)])


def _oof_fit(
    features: pd.DataFrame,
    target: pd.Series,
    classifier: Any,
    splitter: StratifiedKFold,
) -> tuple[Pipeline, np.ndarray]:
    """Fit a pipeline and return out-of-fold probabilities for the same split."""
    model = _pipeline(features, classifier)
    probability = cross_val_predict(model, features, target, cv=splitter, method="predict_proba")[:, 1]
    model.fit(features, target)
    return model, probability


def _metric_row(name: str, target: pd.Series, probabilities: np.ndarray, threshold: float = .5) -> dict[str, float | str]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    return {"model": name, "accuracy": accuracy_score(target, predictions),
            "precision": precision_score(target, predictions, zero_division=0),
            "recall": recall_score(target, predictions, zero_division=0),
            "f1": f1_score(target, predictions, zero_division=0),
            "roc_auc": roc_auc_score(target, probabilities),
            "pr_auc": average_precision_score(target, probabilities),
            "brier_score": brier_score_loss(target, probabilities),
            "false_positive_rate": fp / max(fp + tn, 1), "false_negative_rate": fn / max(fn + tp, 1)}


def run_experiment(curated: CuratedData, output: str | Path, folds: int = 5,
                   source_label: str = "provided dataset") -> dict[str, Any]:
    root = Path(output)
    figures = root / "figures"
    metrics_dir = root / "metrics"
    paper = root / "paper"
    predictions_dir = root / "predictions"
    for path in (figures, metrics_dir, paper, predictions_dir):
        path.mkdir(parents=True, exist_ok=True)
    target = curated.target
    engineered, created = engineer_features(curated.features)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    trained: dict[str, Pipeline] = {}
    probabilities: dict[str, np.ndarray] = {}
    rows = []
    for name, classifier in _models().items():
        model, probability = _oof_fit(engineered, target, classifier, splitter)
        trained[name], probabilities[name] = model, probability
        rows.append(_metric_row(name, target, probability))

    base_estimators = [("lr", _pipeline(engineered, LogisticRegression(max_iter=1000, class_weight="balanced"))),
                       ("rf", _pipeline(engineered, RandomForestClassifier(n_estimators=160, min_samples_leaf=5,
                                                                              class_weight="balanced_subsample", random_state=42, n_jobs=1)))]
    stack = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression(max_iter=1000),
                               stack_method="predict_proba", cv=splitter, n_jobs=None)
    stack_model = stack
    stack_probability = cross_val_predict(stack_model, engineered, target, cv=splitter, method="predict_proba")[:, 1]
    stack_model.fit(engineered, target)
    trained["Proposed Stacking Hybrid"], probabilities["Proposed Stacking Hybrid"] = stack_model, stack_probability
    rows.append(_metric_row("Proposed Stacking Hybrid", target, stack_probability))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_dir / "model_comparison.csv", index=False)
    metrics.to_csv(paper / "model_comparison.csv", index=False)
    _, baseline_probability = _oof_fit(
        curated.features,
        target,
        LogisticRegression(max_iter=1000, class_weight=None),
        splitter,
    )
    _, feature_probability = _oof_fit(
        engineered,
        target,
        LogisticRegression(max_iter=1000, class_weight=None),
        splitter,
    )
    ablation = pd.DataFrame([
        _metric_row("Baseline Logistic Regression", target, baseline_probability),
        _metric_row("Feature Engineering + Logistic Regression", target, feature_probability),
        _metric_row("Class Weights + Logistic Regression", target, probabilities["Logistic Regression"]),
        _metric_row("Proposed Stacking + Class Weights", target, stack_probability),
    ])
    ablation.to_csv(metrics_dir / "ablation_results.csv", index=False)
    ablation.to_csv(paper / "ablation_results.csv", index=False)

    best_name = str(metrics.sort_values("roc_auc", ascending=False).iloc[0]["model"])
    best_probability = probabilities[best_name]
    best_prediction = (best_probability >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, best_prediction, labels=[0, 1]).ravel()
    pd.DataFrame([[tn, fp], [fn, tp]], index=["actual_non_default", "actual_default"],
                 columns=["predicted_non_default", "predicted_default"]).to_csv(paper / "confusion_matrix_final.csv")
    from sklearn.metrics import roc_curve
    for name, probability in probabilities.items():
        fpr, tpr, _ = roc_curve(target, probability)
        plt.plot(fpr, tpr, label=name)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves"); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "roc_curves.png", dpi=180); plt.close()
    for name, probability in probabilities.items():
        precision, recall, _ = precision_recall_curve(target, probability)
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-recall curves"); plt.legend(); plt.tight_layout()
    plt.savefig(figures / "precision_recall_curves.png", dpi=180); plt.close()
    plt.imshow([[tn, fp], [fn, tp]], cmap="Blues")
    plt.colorbar(label="Count"); plt.xticks([0, 1], ["Predicted non-default", "Predicted default"])
    plt.yticks([0, 1], ["Actual non-default", "Actual default"]); plt.title(f"Confusion matrix: {best_name}")
    for row, values in enumerate([[tn, fp], [fn, tp]]):
        for column, value in enumerate(values):
            plt.text(column, row, str(value), ha="center", va="center")
    plt.tight_layout(); plt.savefig(figures / "confusion_matrix_final.png", dpi=180); plt.close()
    metrics.sort_values("roc_auc").plot(x="model", y=["roc_auc", "pr_auc"], kind="barh", figsize=(9, 5))
    plt.xlabel("Score"); plt.title("Model performance comparison"); plt.tight_layout(); plt.savefig(figures / "model_performance.png", dpi=180); plt.close()
    pd.DataFrame({"feature": engineered.columns, "importance": _feature_importance(trained[best_name], engineered, target)}).sort_values("importance", ascending=False).to_csv(paper / "feature_importance.csv", index=False)
    pd.DataFrame({"actual_default": target, "predicted_probability": best_probability,
                  "predicted_default_at_0_5": best_prediction}).head(100).to_csv(
                      paper / "prediction_examples.csv", index=False)
    configurations = {
        "Logistic Regression": "class_weight=balanced; max_iter=1000",
        "Decision Tree": "max_depth=6; min_samples_leaf=5; class_weight=balanced",
        "Random Forest": "n_estimators=180; min_samples_leaf=5; balanced_subsample",
        "Gradient Boosting": "max_iter=140; learning_rate=0.08; max_leaf_nodes=15",
        "XGBoost": "n_estimators=180; max_depth=4; learning_rate=0.05",
    }
    pd.DataFrame({"model": list(_models()), "configuration": [configurations[name] for name in _models()]}).to_csv(
        paper / "model_configuration.csv", index=False)
    pd.DataFrame({"property": ["records", "features", "target", "missing_values", "duplicate_rows", "default_samples", "non_default_samples"],
                  "value": [len(target), curated.features.shape[1], "default", int(curated.features.isna().sum().sum()),
                             int(curated.features.duplicated().sum()), int(target.sum()), int((target == 0).sum())]}).to_csv(paper / "dataset_summary.csv", index=False)
    feature_summary = pd.DataFrame({"feature": engineered.columns,
                                    "dtype": [str(engineered[column].dtype) for column in engineered.columns],
                                    "missing_values": [int(engineered[column].isna().sum()) for column in engineered.columns],
                                    "engineered": [column in created for column in engineered.columns]})
    feature_summary.to_csv(paper / "feature_summary.csv", index=False)
    pd.DataFrame({"observation": [f"Best ROC-AUC model: {best_name} ({metrics.loc[metrics.model == best_name, 'roc_auc'].iloc[0]:.4f}).",
                                   f"Default prevalence: {target.mean():.4f}.",
                                   "All values are generated from the selected input data and cross-validated probabilities."]}).to_csv(paper / "results_summary.csv", index=False)
    summary = {"source": source_label, "records": len(target),
               "features_before_engineering": curated.features.shape[1], "engineered_features": created,
               "best_model_by_roc_auc": best_name, "metrics": rows}
    (metrics_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return {"metrics": metrics, "models": trained, "probabilities": probabilities, "best_model": best_name, "engineered_features": engineered}


def _feature_importance(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> np.ndarray:
    if not hasattr(model, "named_steps") or "preprocess" not in model.named_steps:
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, features, target,
                                        scoring="roc_auc", n_repeats=2, random_state=42)
        return result.importances_mean
    estimator = model.named_steps["model"]
    transformed = model.named_steps["preprocess"].transform(features)
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        values = np.zeros(transformed.shape[1])
    names = model.named_steps["preprocess"].get_feature_names_out()
    grouped = pd.Series(values, index=names).groupby(lambda value: value.split("__", 1)[-1].split("_", 1)[0]).sum()
    return np.array([grouped.get(column, 0.0) for column in features.columns])


def run_research(data: str | Path | None, output: str | Path, folds: int = 5) -> dict[str, Any]:
    curated = load_and_curate(data)
    root = Path(output)
    (root / "paper").mkdir(parents=True, exist_ok=True)
    curated.audit.to_csv(root / "paper" / "feature_audit.csv", index=False)
    eda_tables = run_eda(curated.features, curated.target, root / "eda")
    eda_tables["missing_values"].to_csv(root / "paper" / "eda_summary.csv", index=False)
    source_label = "synthetic development fallback (no dataset supplied)" if data is None else str(data)
    return run_experiment(curated, root, folds=folds, source_label=source_label)