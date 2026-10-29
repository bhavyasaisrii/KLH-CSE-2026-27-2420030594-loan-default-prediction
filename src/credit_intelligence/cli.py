"""Command-line entrypoint."""

from __future__ import annotations

import argparse

from .data import load_and_curate
from .experiments import run_research
from .models import train_models
from .report import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe credit intelligence benchmark")
    parser.add_argument("--data", help="CSV containing loan_status/status/target; omit for synthetic development data")
    parser.add_argument("--output", default="outputs", help="Artifact directory")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--research", action="store_true", help="Generate EDA, hybrid experiments, figures, and paper CSVs")
    args = parser.parse_args()
    if args.research:
        result = run_research(args.data, args.output, folds=args.folds)
        print(result["metrics"].to_string(index=False))
        return
    curated = load_and_curate(args.data)
    results = train_models(curated.features, curated.target, folds=args.folds)
    write_report(args.output, curated.audit, results)
    for result in results:
        print(result.name, result.metrics)


if __name__ == "__main__":
    main()
