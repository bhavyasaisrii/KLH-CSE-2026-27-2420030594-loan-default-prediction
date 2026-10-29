"""Global and local explanations with optional SHAP/LIME adapters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def reason_codes(model: Any, applicant: pd.DataFrame, feature_names: list[str] | None = None,
                 top_n: int = 3) -> list[dict[str, Any]]:
    """Return model-agnostic local reason codes using one-feature perturbations.

    A production adverse-action implementation should map these codes to approved
    reason-code language and validate them with compliance/legal review.
    """
    baseline = float(model.predict_proba(applicant)[:, 1][0])
    names = feature_names or applicant.columns.tolist()
    impacts = []
    for feature in applicant.columns:
        changed = applicant.copy()
        value = changed.iloc[0][feature]
        if pd.api.types.is_numeric_dtype(changed[feature]):
            changed.at[changed.index[0], feature] = 0
        else:
            changed.at[changed.index[0], feature] = "__MISSING__"
        counterfactual = float(model.predict_proba(changed)[:, 1][0])
        impacts.append({"feature": feature, "impact": baseline - counterfactual, "value": value})
    return sorted(impacts, key=lambda item: abs(item["impact"]), reverse=True)[:top_n]


def shap_global_importance(model: Any, features: pd.DataFrame) -> pd.DataFrame:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies with: pip install -e '.[explainability]'") from exc
    transformed = model.named_steps["preprocess"].transform(features)
    explainer = shap.Explainer(model.named_steps["model"], transformed)
    values = explainer(transformed).values
    names = model.named_steps["preprocess"].get_feature_names_out()
    return pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
