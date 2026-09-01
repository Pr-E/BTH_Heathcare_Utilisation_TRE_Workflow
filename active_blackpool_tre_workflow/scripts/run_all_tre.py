"""Command-line orchestration entry point for the reviewed real-data TRE workflow.

Examples
--------
Run the complete required analytical chain through clustering::

    python scripts/run_all_tre.py

Resume from an already validated checkpoint::

    python scripts/run_all_tre.py --from-stage descriptive --to-stage clustering

The individual stage scripts remain the preferred first-run route because they
allow each QA gate to be inspected before proceeding.  This orchestrator is most
useful after Ian/analytical review, once the stages have already been validated.
"""
from __future__ import annotations

# ``argparse`` exposes a reproducible command-line interface instead of relying
# on notebook cell state or hard-coded analyst choices.
import argparse

# ``Path`` lets the script find the repository's ``src`` directory regardless of
# the TRE workspace's absolute path.
from pathlib import Path

# ``sys`` is used only to make the local package importable before/without an
# editable installation.
import sys

# The repository root is the parent of ``scripts``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# All importable production source code is stored in ``src``.
SRC = PROJECT_ROOT / "src"

# Prepend the local package path only when necessary.  The recommended TRE setup
# still installs the project using ``python -m pip install -e .``.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import the reviewed stage order and orchestration function from package code.
from bth_analysis.orchestration.tre import STAGE_ORDER, run_tre_workflow


def main() -> None:
    """Parse CLI arguments and execute the requested contiguous stage range."""
    # Create the command-line parser with a concise purpose statement.
    parser = argparse.ArgumentParser(
        description="Run the Active Blackpool/BTH real-data TRE workflow"
    )

    # Keep all configuration paths explicit and version-controllable.
    parser.add_argument("--workflow", default="config/workflow_tre.yaml")
    parser.add_argument("--pipeline", default="config/pipeline_tre.yaml")
    parser.add_argument("--clustering", default="config/clustering_tre.yaml")

    # Restrict resumption points to the reviewed stage names so typographical
    # errors cannot silently skip or reorder analytical stages.
    parser.add_argument(
        "--from-stage", choices=STAGE_ORDER, default="preflight"
    )
    parser.add_argument(
        "--to-stage", choices=STAGE_ORDER, default="clustering"
    )

    # Materialise the validated command-line arguments.
    args = parser.parse_args()

    # Execute the requested workflow range.  The orchestration layer records a
    # run manifest, per-stage PASS/FAIL status and a traceback file on failure.
    run_tre_workflow(
        workflow_path=args.workflow,
        pipeline_path=args.pipeline,
        clustering_path=args.clustering,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
    )


# Do not run the pipeline when this file is imported by tests or documentation.
if __name__ == "__main__":
    main()
