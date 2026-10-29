"""Persist reproducible audit and model artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from .models import ModelResult


def write_report(output: str | Path, audit, results: list[ModelResult]) -> Path:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    audit.to_csv(destination / "feature_audit.csv", index=False)
    summary = [{"model": result.name, **result.metrics} for result in results]
    (destination / "model_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return destination
