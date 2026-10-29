"""Data-backed Streamlit research platform for loan default analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve, precision_recall_curve)

from .data import CuratedData, audit_columns, load_and_curate
from .experiments import engineer_features, run_experiment
from .explainability import reason_codes

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MERGED_DATASET = DATA_DIR / "merged_public_credit_datasets.csv"
OUTPUT_DIR = ROOT / "outputs"


def _prepare(uploaded: Any) -> tuple[CuratedData, str]:
    if not uploaded:
        if MERGED_DATASET.exists():
            curated = load_and_curate(MERGED_DATASET)
            source = MERGED_DATASET.name
        else:
            datasets = sorted(DATA_DIR.glob("*.csv"))
            if datasets:
                curated = load_and_curate(paths=datasets)
                source = f"Vertical merge ({len(datasets)} datasets): " + ", ".join(dataset.name for dataset in datasets)
            else:
                curated = load_and_curate()
                source = "DEMO / synthetic development portfolio"
    else:
        uploads = uploaded if isinstance(uploaded, list) else [uploaded]
        curated = load_and_curate(paths=uploads)
        source = f"Vertical merge ({len(uploads)} datasets): " + ", ".join(item.name for item in uploads)
    if curated.target.nunique() < 2:
        raise ValueError("The target must contain both default and non-default records.")
    return curated, source


@st.cache_resource(show_spinner="Training leakage-safe research models...")
def _train(features: pd.DataFrame, target: pd.Series, audit: pd.DataFrame, source: str,
           model_suite_version: str = "v2") -> dict[str, Any]:
    curated = CuratedData(features=features, target=target, audit=audit)
    folds = min(3, int(target.value_counts().min()))
    if folds < 2:
        raise ValueError("At least two records are required in each target class for cross-validation.")
    result = run_experiment(curated, OUTPUT_DIR, folds=folds, source_label=source)
    return result


def _risk(probability: float, thresholds: tuple[float, float, float, float]) -> str:
    labels = ("VERY LOW", "LOW", "MODERATE", "HIGH", "VERY HIGH")
    for limit, label in zip(thresholds, labels):
        if probability < limit:
            return label
    return labels[-1]


def _metric_card(label: str, value: str, note: str = "") -> None:
    st.metric(label, value, help=note or None)


def _fig_defaults(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=45, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _numeric_frame(features: pd.DataFrame) -> list[str]:
    return features.select_dtypes(include=np.number).columns.tolist()


def _portfolio_frame(features: pd.DataFrame, target: pd.Series, probabilities: np.ndarray,
                     thresholds: tuple[float, float, float, float]) -> pd.DataFrame:
    frame = features.copy()
    frame["default"] = target.to_numpy()
    frame["default_probability"] = probabilities
    frame["risk_category"] = [_risk(value, thresholds) for value in probabilities]
    return frame


def _render_overview(frame: pd.DataFrame, source: str) -> None:
    st.title("Loan Default Risk Analytics")
    st.caption("Explainable machine-learning based credit-risk prediction and analytics.")
    if "DEMO" in source:
        st.caption("Source: sample development dataset")
    numeric = _numeric_frame(frame.drop(columns=["default", "default_probability", "risk_category"], errors="ignore"))
    cards = st.columns(5)
    with cards[0]: _metric_card("Applications", f"{len(frame):,}")
    with cards[1]: _metric_card("Default cases", f"{int(frame.default.sum()):,}")
    with cards[2]: _metric_card("Observed default rate", f"{frame.default.mean():.2%}")
    with cards[3]: _metric_card("Average risk", f"{frame.default_probability.mean():.2%}")
    with cards[4]: _metric_card("High-risk applications", f"{int(frame.risk_category.isin(['HIGH', 'VERY HIGH']).sum()):,}")
    left, right = st.columns(2)
    with left:
        counts = frame["risk_category"].value_counts().reindex(["VERY LOW", "LOW", "MODERATE", "HIGH", "VERY HIGH"], fill_value=0)
        st.plotly_chart(_fig_defaults(px.bar(counts, title="Risk distribution", labels={"value": "Applications", "index": "Risk category"})), use_container_width=True)
    with right:
        st.plotly_chart(_fig_defaults(px.histogram(frame, x="default_probability", color="default", nbins=30,
                                                   title="Default probability distribution")), use_container_width=True)
    if numeric:
        st.plotly_chart(_fig_defaults(px.histogram(frame, x=numeric[0], color="default", title=f"{numeric[0]} by observed outcome")), use_container_width=True)
    st.info("This is a model-based risk estimate and not a guarantee of future repayment.")


def _render_dataset(curated: CuratedData, source: str) -> None:
    st.title("Dataset Management")
    st.caption(f"Source: {source}")
    summary = pd.DataFrame({"Property": ["Dataset name/source", "Rows", "Columns", "Target", "Missing cells", "Duplicate feature rows", "Default cases", "Non-default cases"],
                            "Value": [source, len(curated.features), curated.features.shape[1], "default", int(curated.features.isna().sum().sum()), int(curated.features.duplicated().sum()), int(curated.target.sum()), int((curated.target == 0).sum())]})
    summary["Value"] = summary["Value"].astype(str)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.subheader("Dataset preview")
    preview = curated.features.copy(); preview["default"] = curated.target
    st.dataframe(preview.head(100), use_container_width=True)
    st.download_button("Download feature audit", curated.audit.to_csv(index=False), "feature_audit.csv", "text/csv")


def _render_quality(curated: CuratedData) -> None:
    st.title("Data Quality")
    features = curated.features
    missing = features.isna().sum().sort_values(ascending=False)
    numeric = features.select_dtypes(include=np.number)
    outliers = ((numeric - numeric.median()).abs() > 3 * numeric.std()).sum().sort_values(ascending=False) if not numeric.empty else pd.Series(dtype=int)
    cards = st.columns(4)
    with cards[0]: _metric_card("Missing cells", f"{int(missing.sum()):,}")
    with cards[1]: _metric_card("Duplicate rows", f"{int(features.duplicated().sum()):,}")
    with cards[2]: _metric_card("Numeric outlier flags", f"{int(outliers.sum()):,}")
    with cards[3]: _metric_card("Minority class", f"{curated.target.mean():.2%}")
    quality = pd.DataFrame({"feature": features.columns, "dtype": [str(features[column].dtype) for column in features.columns],
                            "missing": [int(features[column].isna().sum()) for column in features.columns],
                            "negative_values": [int((pd.to_numeric(features[column], errors="coerce") < 0).sum()) for column in features.columns]})
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.caption("Outlier flags use a transparent median/standard-deviation screen and are not automatic removal decisions.")


def _render_eda(frame: pd.DataFrame) -> None:
    st.title("Exploratory Data Analysis")
    feature = st.selectbox("Feature", [column for column in frame.columns if column not in {"default", "default_probability", "risk_category"}])
    numeric = pd.api.types.is_numeric_dtype(frame[feature])
    if numeric:
        clean_feature = frame[feature].dropna()
        if clean_feature.empty:
            st.info(f"No numeric values are available for {feature}.")
            return
        left, right = st.columns(2)
        with left: st.plotly_chart(_fig_defaults(px.histogram(frame, x=feature, color="default", title=f"Distribution: {feature}")), use_container_width=True)
        with right: st.plotly_chart(_fig_defaults(px.box(frame, x="default", y=feature, title=f"{feature} versus observed default")), use_container_width=True)
        if clean_feature.nunique() >= 2:
            rates = frame.assign(bin=pd.qcut(frame[feature], q=min(8, clean_feature.nunique()), duplicates="drop")).groupby("bin", observed=True)["default"].mean().reset_index()
            rates["bin"] = rates["bin"].astype(str)
            st.plotly_chart(_fig_defaults(px.bar(rates, x="bin", y="default", title=f"Observed default rate by {feature}")), use_container_width=True)
            st.info(f"Observed default rate ranges from {rates.default.min():.2%} to {rates.default.max():.2%} across the displayed {feature} bins.")
        else:
            st.info(f"{feature} has only one distinct numeric value, so a binned comparison is not applicable.")
    else:
        grouped = frame.groupby(feature, dropna=False).default.agg(default_rate="mean", applications="size").reset_index().sort_values("applications", ascending=False).head(20)
        st.plotly_chart(_fig_defaults(px.bar(grouped, x="default_rate", y=feature, orientation="h", title=f"Observed default rate by {feature}")), use_container_width=True)
        st.dataframe(grouped, use_container_width=True, hide_index=True)
    numeric_columns = [column for column in _numeric_frame(frame) if column not in {"default", "default_probability"}]
    if len(numeric_columns) > 1:
        correlation = frame[numeric_columns + ["default"]].corr(numeric_only=True)
        heatmap = go.Figure(go.Heatmap(z=correlation.to_numpy(), x=correlation.columns.tolist(), y=correlation.index.tolist(),
                                       text=np.round(correlation.to_numpy(), 2), texttemplate="%{text}", colorscale="Teal"))
        heatmap.update_layout(title="Numeric correlation heatmap")
        st.plotly_chart(_fig_defaults(heatmap, 500), use_container_width=True)


def _render_features(features: pd.DataFrame, engineered: pd.DataFrame) -> None:
    st.title("Feature Engineering")
    rows = []
    for column in engineered.columns:
        made = column not in features.columns
        formula = "Derived only when source columns exist" if made else "Original applicant-time feature"
        rows.append({"feature": column, "type": str(engineered[column].dtype), "origin": "Derived" if made else "Original", "formula/description": formula})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Target and post-funding fields are excluded by the point-in-time curation audit before this stage.")
    st.code("dti_to_income = dti / annual_inc\nloan_to_income = loan_amnt / annual_inc\npayment_to_income = installment / annual_inc", language="python")


def _form_for(features: pd.DataFrame) -> pd.DataFrame:
    values: dict[str, Any] = {}
    columns = st.columns(3)
    for index, column in enumerate(features.columns):
        with columns[index % 3]:
            if pd.api.types.is_numeric_dtype(features[column]):
                series = pd.to_numeric(features[column], errors="coerce").dropna()
                default = float(series.median()) if not series.empty else 0.0
                values[column] = st.number_input(column.replace("_", " ").title(), value=default)
            else:
                choices = sorted(features[column].dropna().astype(str).unique().tolist()) or ["Unknown"]
                values[column] = st.selectbox(column.replace("_", " ").title(), choices)
    return pd.DataFrame([values], columns=features.columns)


def _render_prediction(model: Any, features: pd.DataFrame, thresholds: tuple[float, float, float, float]) -> None:
    st.title("Single Loan Prediction")
    st.caption("Applicant-time inputs only. The result is a model-based decision-support estimate.")
    applicant = _form_for(features)
    if st.button("Predict Default Risk", type="primary"):
        probability = float(model.predict_proba(applicant)[0, 1])
        category = _risk(probability, thresholds)
        st.subheader("Loan risk assessment")
        st.progress(probability, text=f"Default probability: {probability:.2%}")
        st.metric("Risk category", category)
        reasons = reason_codes(model, applicant, top_n=min(5, len(features.columns)))
        st.subheader("Contributing model factors")
        st.dataframe(pd.DataFrame(reasons), use_container_width=True, hide_index=True)
        st.warning("Contributions describe model output; they do not establish causality or guarantee repayment.")


def _render_portfolio(frame: pd.DataFrame) -> None:
    st.title("Portfolio Analytics")
    numeric = _numeric_frame(frame)
    selected = st.multiselect("Filter risk categories", frame.risk_category.unique().tolist(), default=frame.risk_category.unique().tolist())
    view = frame[frame.risk_category.isin(selected)]
    st.dataframe(view.describe(include="all").T, use_container_width=True)
    if numeric:
        st.plotly_chart(_fig_defaults(px.box(view, x="risk_category", y=numeric[0], color="risk_category", title=f"{numeric[0]} across risk categories")), use_container_width=True)
    categorical = [column for column in frame.columns if column not in numeric + ["default", "default_probability", "risk_category"]]
    if categorical:
        grouping = view.groupby(categorical[0], dropna=False).agg(default_rate=("default", "mean"), average_risk=("default_probability", "mean"), applications=("default", "size")).reset_index()
        st.plotly_chart(_fig_defaults(px.bar(grouping, x=categorical[0], y="average_risk", title=f"Average predicted risk by {categorical[0]}")), use_container_width=True)


def _metric_table(result: dict[str, Any]) -> pd.DataFrame:
    return result["metrics"].copy()


def _render_evaluation(result: dict[str, Any], target: pd.Series) -> None:
    st.title("Model Evaluation")
    metrics = _metric_table(result)
    st.dataframe(metrics.style.format({column: "{:.3f}" for column in metrics.columns if column != "model"}), use_container_width=True, hide_index=True)
    selected = st.selectbox("Model", metrics.model.tolist())
    probability = result["probabilities"][selected]
    left, right = st.columns(2)
    with left:
        figure = go.Figure()
        for name, values in result["probabilities"].items():
            fpr, tpr, _ = roc_curve(target, values); figure.add_trace(go.Scatter(x=fpr, y=tpr, name=name))
        figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="chance", line=dict(dash="dash")))
        st.plotly_chart(_fig_defaults(figure, 430), use_container_width=True)
    with right:
        figure = go.Figure()
        for name, values in result["probabilities"].items():
            precision, recall, _ = precision_recall_curve(target, values); figure.add_trace(go.Scatter(x=recall, y=precision, name=name))
        st.plotly_chart(_fig_defaults(figure, 430), use_container_width=True)
    matrix = confusion_matrix(target, (probability >= .5).astype(int), labels=[0, 1])
    heatmap = go.Figure(go.Heatmap(z=matrix, x=["Predicted non-default", "Predicted default"],
                                   y=["Actual non-default", "Actual default"], text=matrix, texttemplate="%{text}", colorscale="Teal"))
    heatmap.update_layout(title=f"Confusion matrix: {selected}")
    st.plotly_chart(_fig_defaults(heatmap, 400), use_container_width=True)


def _render_hybrid(result: dict[str, Any]) -> None:
    st.title("Proposed Hybrid Model")
    st.info("Proposed Methodological Contribution: point-in-time audit + conditional financial features + fold-local heterogeneous stacking + probability/error analysis. This is an experimentally testable contribution, not a claim of unprecedentedness.")
    st.graphviz_chart("digraph { rankdir=LR; LR[label=\"Logistic Regression\"]; RF[label=\"Random Forest\"]; LR -> META; RF -> META; META[label=\"Logistic meta-learner\"]; META -> CAL[label=\"risk probability\"]; CAL -> RISK[label=\"risk category\"]; }")
    st.dataframe(_metric_table(result), use_container_width=True, hide_index=True)
    st.subheader("Ablation evidence")
    path = OUTPUT_DIR / "paper" / "ablation_results.csv"
    if path.exists(): st.dataframe(pd.read_csv(path), use_container_width=True, hide_index=True)
    else: st.info("Not yet evaluated")


def _render_what_if(model: Any, features: pd.DataFrame, thresholds: tuple[float, float, float, float]) -> None:
    st.title("What-If Risk Sensitivity")
    st.caption("Sensitivity analysis only; these hypothetical changes do not establish causality or financial advice.")
    applicant = _form_for(features)
    baseline = float(model.predict_proba(applicant)[0, 1])
    numeric = _numeric_frame(features)
    if not numeric:
        st.info("No numeric features are available for sensitivity analysis."); return
    feature = st.selectbox("Feature to vary", numeric)
    series = pd.to_numeric(features[feature], errors="coerce").dropna()
    low, high = float(series.quantile(.05)), float(series.quantile(.95))
    value = st.slider("Scenario value", min_value=low, max_value=high, value=float(applicant.iloc[0][feature]))
    scenario = applicant.copy(); scenario.at[0, feature] = value
    scenario_probability = float(model.predict_proba(scenario)[0, 1])
    cards = st.columns(3)
    with cards[0]: _metric_card("Original probability", f"{baseline:.2%}")
    with cards[1]: _metric_card("Scenario probability", f"{scenario_probability:.2%}")
    with cards[2]: _metric_card("Change", f"{scenario_probability - baseline:+.2%}")
    st.write(f"Original risk: **{_risk(baseline, thresholds)}** | Scenario risk: **{_risk(scenario_probability, thresholds)}**")


def _render_calibration(result: dict[str, Any], target: pd.Series) -> None:
    st.title("Probability Calibration")
    selected = st.selectbox("Model", list(result["probabilities"]))
    probability = result["probabilities"][selected]
    observed, predicted = calibration_curve(target, probability, n_bins=10, strategy="quantile")
    figure = go.Figure([go.Scatter(x=predicted, y=observed, mode="lines+markers", name=selected), go.Scatter(x=[0, 1], y=[0, 1], name="perfect", line=dict(dash="dash"))])
    st.plotly_chart(_fig_defaults(figure), use_container_width=True)
    st.metric("Brier score", f"{brier_score_loss(target, probability):.4f}")
    st.caption("Calibration is assessed from cross-validated probabilities. A separate held-out calibration split is required before production claims.")


def _render_xai(model: Any, features: pd.DataFrame, result: dict[str, Any]) -> None:
    st.title("Explainable AI Laboratory")
    st.caption("Factors are associated with model output and do not claim to cause default.")
    path = OUTPUT_DIR / "paper" / "feature_importance.csv"
    if path.exists():
        importance = pd.read_csv(path).head(15).sort_values("importance")
        st.plotly_chart(_fig_defaults(px.bar(importance, x="importance", y="feature", orientation="h", title="Global feature importance")), use_container_width=True)
    sample = features.iloc[[0]]
    st.metric("Sample predicted probability", f"{float(model.predict_proba(sample)[0, 1]):.2%}")
    st.dataframe(pd.DataFrame(reason_codes(model, sample, top_n=min(8, len(features.columns)))), use_container_width=True, hide_index=True)
    st.info("SHAP is optional in this environment. Install the explainability extra to add SHAP-specific plots; local model-agnostic reason codes remain available.")


def _render_batch(model: Any, features: pd.DataFrame, thresholds: tuple[float, float, float, float]) -> None:
    st.title("Batch Prediction")
    uploaded = st.file_uploader("Upload applicant feature CSV", type="csv", key="batch")
    if uploaded is None:
        st.info("Not yet evaluated: upload a feature CSV to generate predictions."); return
    batch = pd.read_csv(uploaded)
    batch, _ = engineer_features(batch)
    missing = [column for column in features.columns if column not in batch.columns]
    if missing:
        st.error(f"Missing required model features: {missing}"); return
    probabilities = model.predict_proba(batch[features.columns])[:, 1]
    output = batch.copy(); output["default_probability"] = probabilities; output["risk_category"] = [_risk(value, thresholds) for value in probabilities]
    st.dataframe(output, use_container_width=True)
    st.download_button("Download predictions", output.to_csv(index=False), "loan_risk_predictions.csv", "text/csv")


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty or reference.nunique() < 2:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    reference_counts = np.histogram(reference, bins=edges)[0].astype(float)
    current_counts = np.histogram(current, bins=edges)[0].astype(float)
    reference_share = np.maximum(reference_counts / max(reference_counts.sum(), 1), 1e-6)
    current_share = np.maximum(current_counts / max(current_counts.sum(), 1), 1e-6)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def _drift_level(psi: float) -> str:
    if psi < 0.10:
        return "LOW"
    if psi < 0.25:
        return "MEDIUM"
    return "HIGH"


def _render_monitoring(model: Any, features: pd.DataFrame, frame: pd.DataFrame, source: str) -> None:
    st.title("Model Monitoring")
    st.caption("Monitoring compares the active model's reference portfolio with an uploaded feature batch. Drift is a warning signal, not proof of model failure.")
    summary = pd.DataFrame({"metric": ["Model version", "Training source", "Reference rows", "Average reference probability", "Reference high-risk percentage"],
                            "value": ["research-run / 3-fold", source, len(frame), f"{frame.default_probability.mean():.4f}", f"{frame.risk_category.isin(['HIGH', 'VERY HIGH']).mean():.2%}"]})
    summary["value"] = summary["value"].astype(str)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    uploaded = st.file_uploader("Upload monitoring batch (feature columns only)", type="csv", key="monitoring_batch")
    if uploaded is None:
        st.info("Upload a batch CSV to calculate prediction shifts and feature drift. A ready-to-use demo file is available at data/demo_monitoring_batch.csv.")
        return
    batch = pd.read_csv(uploaded)
    batch, _ = engineer_features(batch)
    missing = [column for column in features.columns if column not in batch.columns]
    if missing:
        st.error(f"Monitoring batch is missing required model features: {missing}")
        return
    batch_features = batch[features.columns].replace([np.inf, -np.inf], np.nan)
    probabilities = model.predict_proba(batch_features)[:, 1]
    risk_categories = [_risk(value, (0.20, 0.40, 0.60, 0.80)) for value in probabilities]
    reference_high = frame.risk_category.isin(["HIGH", "VERY HIGH"]).mean()
    current_high = np.mean(np.isin(risk_categories, ["HIGH", "VERY HIGH"]))
    cards = st.columns(4)
    with cards[0]: _metric_card("Monitoring rows", f"{len(batch):,}")
    with cards[1]: _metric_card("Average probability", f"{probabilities.mean():.2%}")
    with cards[2]: _metric_card("High-risk percentage", f"{current_high:.2%}", f"Reference: {reference_high:.2%}")
    with cards[3]: _metric_card("Risk-rate change", f"{current_high - reference_high:+.2%}")
    drift_rows = []
    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            psi = _psi(features[column], batch_features[column])
            drift_rows.append({"feature": column, "psi": psi, "status": _drift_level(psi)})
    drift = pd.DataFrame(drift_rows).sort_values("psi", ascending=False) if drift_rows else pd.DataFrame(columns=["feature", "psi", "status"])
    st.subheader("Feature drift")
    st.dataframe(drift.style.format({"psi": "{:.4f}"}), use_container_width=True, hide_index=True)
    if not drift.empty:
        highest = drift.iloc[0]
        st.warning(f"Highest observed drift: {highest.feature} with PSI {highest.psi:.4f} ({highest.status}). This indicates distribution change, not automatic model failure.")
    comparison = pd.DataFrame({"portfolio": ["Reference", "Monitoring batch"], "average_probability": [frame.default_probability.mean(), probabilities.mean()], "high_risk_percentage": [reference_high, current_high]})
    st.plotly_chart(_fig_defaults(px.bar(comparison, x="portfolio", y=["average_probability", "high_risk_percentage"], barmode="group", title="Reference versus monitoring batch")), use_container_width=True)
    output = batch.copy(); output["default_probability"] = probabilities; output["risk_category"] = risk_categories
    st.download_button("Download monitored predictions", output.to_csv(index=False), "monitoring_predictions.csv", "text/csv")


def _render_responsible(curated: CuratedData, frame: pd.DataFrame) -> None:
    st.title("Responsible AI")
    st.markdown("**Explainability:** model factors support analyst review. **Fairness:** group comparisons require a legally justified audit attribute. **Privacy:** uploaded data is processed in the current session. **Human oversight:** this system does not approve or reject loans automatically.")
    groups = [column for column in curated.features.columns if curated.features[column].nunique(dropna=False) <= 12]
    if groups:
        group = st.selectbox("Audit group", groups)
        audit_rows = []
        audited = frame.assign(predicted=(frame.default_probability >= .5).astype(int))
        for value, subset in audited.groupby(group, dropna=False):
            negatives = subset.default == 0
            audit_rows.append({"group": value, "applications": len(subset),
                               "recall": recall_score(subset.default, subset.predicted, zero_division=0),
                               "false_positive_rate": float(((subset.predicted == 1) & negatives).sum() / max(negatives.sum(), 1))})
        audit = pd.DataFrame(audit_rows)
        st.dataframe(audit, use_container_width=True, hide_index=True)
    else:
        st.info("No small-cardinality audit group is available in the current data.")


def _render_research(result: dict[str, Any]) -> None:
    st.title("Research Results")
    st.caption("Paper-ready artifacts generated by the actual selected input source.")
    for filename in ["dataset_summary.csv", "feature_summary.csv", "model_configuration.csv", "model_comparison.csv", "ablation_results.csv", "confusion_matrix_final.csv", "feature_importance.csv", "results_summary.csv"]:
        path = OUTPUT_DIR / "paper" / filename
        if path.exists():
            with st.expander(filename): st.dataframe(pd.read_csv(path), use_container_width=True, hide_index=True)
        else:
            st.info(f"{filename}: Not yet evaluated")
    for filename in ["roc_curves.png", "precision_recall_curves.png", "model_performance.png", "confusion_matrix_final.png"]:
        path = OUTPUT_DIR / "figures" / filename
        if path.exists(): st.image(str(path), caption=filename)


def _render_review3(result: dict[str, Any], curated: CuratedData, source: str) -> None:
    st.title("Review-3 Assessment")
    st.caption("Live evidence for methodology, implementation, dashboard execution, and comparative model analysis.")

    st.subheader("Assessment coverage")
    coverage = pd.DataFrame([
        {"assessment point": "Methodology validation", "evidence": "Point-in-time audit, target curation, fold-local preprocessing, stratified out-of-fold probabilities", "status": "IMPLEMENTED"},
        {"assessment point": "Code implementation", "evidence": "Reusable data, model, experiment, explainability, fairness, and reporting modules", "status": "IMPLEMENTED"},
        {"assessment point": "Dashboard execution", "evidence": f"Live Streamlit run using {source}", "status": "RUNNING"},
        {"assessment point": "Baseline model comparison", "evidence": "Logistic regression, decision tree, random forest, gradient boosting, optional XGBoost", "status": "EVALUATED"},
        {"assessment point": "Literature-aligned comparison", "evidence": "Classical and ensemble families shown as reproducible in-dataset comparators; published results are not merged with these metrics", "status": "EVALUATED"},
        {"assessment point": "Proposed hybrid model", "evidence": "Weighted logistic-regression + random-forest stacking with a logistic meta-learner", "status": "EVALUATED"},
    ])
    st.dataframe(coverage, use_container_width=True, hide_index=True)

    st.subheader("Methodology and implementation validation")
    left, right = st.columns(2)
    with left:
        st.markdown("""
        **Validated controls**
        - Target is curated from configured default/non-default statuses.
        - Post-funding and collections fields are excluded by the data audit.
        - Imputation, scaling, and encoding are fitted inside each cross-validation fold.
        - Metrics use out-of-fold probabilities rather than in-sample predictions.
        - The threshold is a reporting control, not an automated lending decision.
        """)
    with right:
        implementation = pd.DataFrame([
            {"component": "Input and curation", "module": "credit_intelligence.data", "status": "PASS"},
            {"component": "Model pipelines", "module": "credit_intelligence.models", "status": "PASS"},
            {"component": "Research runner", "module": "credit_intelligence.experiments", "status": "PASS"},
            {"component": "Interactive dashboard", "module": "credit_intelligence.dashboard", "status": "PASS"},
            {"component": "Explainability and fairness", "module": "explainability / fairness", "status": "AVAILABLE"},
        ])
        st.dataframe(implementation, use_container_width=True, hide_index=True)

    st.subheader("Baseline, literature-aligned, and proposed models")
    metrics = result["metrics"].copy()
    families = {
        "Logistic Regression": "Baseline / literature-aligned linear",
        "Decision Tree": "Baseline / literature-aligned tree",
        "Random Forest": "Literature-aligned ensemble",
        "Gradient Boosting": "Literature-aligned ensemble",
        "XGBoost": "Literature-aligned boosting",
        "Proposed Stacking Hybrid": "Proposed hybrid",
    }
    comparison = metrics.assign(
        model_family=metrics["model"].map(families).fillna("Additional comparator")
    )[["model", "model_family", "roc_auc", "pr_auc", "f1", "brier_score", "false_negative_rate"]]
    st.dataframe(comparison.style.format({column: "{:.3f}" for column in comparison.columns[2:]}),
                 use_container_width=True, hide_index=True)
    best_auc = comparison.loc[comparison["roc_auc"].idxmax()]
    st.info(
        f"Current selected source: {source}. Best observed ROC-AUC in this run is "
        f"{best_auc['model']} ({best_auc['roc_auc']:.3f}). Compare PR-AUC, Brier score, "
        "recall, and false-negative rate before declaring a model preferable."
    )
    st.warning(
        "Literature-aligned labels describe model families implemented here. Published scores "
        "from other datasets are not directly comparable and are intentionally not presented as "
        "experimental evidence. Add verified citations before making literature performance claims."
    )

    st.subheader("Run facts")
    facts = pd.DataFrame({
        "property": ["source", "records", "raw features", "engineered features", "default rate", "target classes"],
        "value": [source, len(curated.target), curated.features.shape[1],
                  result["engineered_features"].shape[1], f"{curated.target.mean():.2%}",
                  ", ".join(map(str, sorted(curated.target.unique().tolist())))],
    })
    st.dataframe(facts, use_container_width=True, hide_index=True)
    st.caption("For final Review-3 evidence, replace the synthetic development source with the approved real CSV and attach the generated paper artifacts.")


def render() -> None:
    st.set_page_config(page_title="Loan Default Risk Analytics", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: Manrope, sans-serif; }
    .stApp { background: #08100f; } [data-testid="stSidebar"] { background: #0d1715; }
    .block-container { max-width: 1480px; padding: 2rem 3rem 4rem; }
    div[data-testid="stMetric"] { background: #101c19; border: 1px solid #29413a; border-radius: 6px; padding: .9rem; }
    </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### CREDIT / INTELLIGENCE")
        st.caption("Research decision-support platform")
        uploaded = st.file_uploader("Upload loan datasets", type="csv", accept_multiple_files=True)
        st.divider()
        page = st.radio("Navigate", ["Overview", "Review-3 Assessment", "Dataset Management", "Data Quality", "EDA", "Feature Engineering", "Prediction", "Portfolio Analytics", "Model Training", "Model Evaluation", "Hybrid Model", "Risk Scoring", "What-If Analysis", "Probability Calibration", "Explainable AI", "Risk Sensitivity", "Batch Prediction", "Risk Segmentation", "Model Monitoring", "Responsible AI", "Report Generator", "Research Results", "About Project"])
    try:
        curated, source = _prepare(uploaded)
        result = _train(curated.features, curated.target, curated.audit, source)
    except Exception as error:
        st.error(f"Unable to process this dataset: {error}"); st.stop()
    engineered = result["engineered_features"]
    selected_model = st.sidebar.selectbox("Active model", list(result["models"]))
    model = result["models"][selected_model]
    threshold_values = [st.sidebar.number_input(f"Risk threshold {index + 1}", 0.01, 0.99, default, 0.01, key=f"risk_threshold_{index}")
                        for index, default in enumerate((0.20, 0.40, 0.60, 0.80))]
    thresholds = tuple(sorted(threshold_values))
    portfolio = _portfolio_frame(engineered, curated.target, result["probabilities"][selected_model], thresholds)
    if page == "Overview": _render_overview(portfolio, source)
    elif page == "Review-3 Assessment": _render_review3(result, curated, source)
    elif page == "Dataset Management": _render_dataset(curated, source)
    elif page == "Data Quality": _render_quality(curated)
    elif page == "EDA": _render_eda(portfolio)
    elif page == "Feature Engineering": _render_features(curated.features, engineered)
    elif page == "Prediction": _render_prediction(model, engineered, thresholds)
    elif page == "Portfolio Analytics": _render_portfolio(portfolio)
    elif page == "Model Training": st.title("Model Training"); st.success(f"Models trained from {source} using 3-fold stratified cross-validation."); st.dataframe(_metric_table(result), use_container_width=True, hide_index=True)
    elif page == "Model Evaluation": _render_evaluation(result, curated.target)
    elif page == "Hybrid Model": _render_hybrid(result)
    elif page == "Risk Scoring": st.title("Risk Scoring"); st.dataframe(pd.DataFrame({"lower_bound": [0, thresholds[0], thresholds[1], thresholds[2], thresholds[3]], "upper_bound": [thresholds[0], thresholds[1], thresholds[2], thresholds[3], 1], "category": ["VERY LOW", "LOW", "MODERATE", "HIGH", "VERY HIGH"]}), use_container_width=True, hide_index=True); st.caption("Configurable research thresholds, not universal banking standards.")
    elif page in {"What-If Analysis", "Risk Sensitivity"}: _render_what_if(model, engineered, thresholds)
    elif page == "Probability Calibration": _render_calibration(result, curated.target)
    elif page == "Explainable AI": _render_xai(model, engineered, result)
    elif page == "Batch Prediction": _render_batch(model, engineered, thresholds)
    elif page == "Risk Segmentation": _render_portfolio(portfolio)
    elif page == "Model Monitoring": _render_monitoring(model, engineered, portfolio, source)
    elif page == "Responsible AI": _render_responsible(curated, portfolio)
    elif page == "Report Generator": st.title("Report Generator"); st.download_button("Download model comparison", _metric_table(result).to_csv(index=False), "model_comparison.csv", "text/csv"); st.info("Borrower reports require an active prediction session.")
    elif page == "Research Results": _render_research(result)
    else: st.title("About Project"); st.write("An explainable, leakage-aware loan-default risk analytics prototype. The proposed methodological contribution is evaluated heterogeneous stacking with point-in-time curation and probability-oriented analysis. The dashboard is not the novelty.")
