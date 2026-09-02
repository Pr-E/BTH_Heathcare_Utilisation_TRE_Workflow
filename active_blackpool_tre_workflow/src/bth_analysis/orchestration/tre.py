"""Run the real-data TRE workflow in dependency order.

The orchestrator starts from approved TRE extracts and executes preflight,
ingestion, cleaning, preprocessing, linkage, cohort construction, outcomes,
descriptive analysis, comparative modelling and clustering. Analysts can resume
from a validated stage without changing analytical code.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import traceback
from typing import Callable, Any

import pandas as pd

from bth_analysis.analysis.clustering import run_clustering
from bth_analysis.analysis.comparative import run_comparative
from bth_analysis.analysis.descriptive import run_descriptive
from bth_analysis.data_pipeline.cleaning import run_cleaning
from bth_analysis.data_pipeline.cohort import run_cohort_index
from bth_analysis.data_pipeline.ingestion import run_ingestion
from bth_analysis.data_pipeline.linkage import run_linkage
from bth_analysis.data_pipeline.outcomes import run_outcome_features
from bth_analysis.data_pipeline.preprocessing import run_preprocessing
from bth_analysis.orchestration.preflight import run_preflight
from bth_analysis.workflow import (
    build_run_manifest,
    load_workflow_config,
    output_path,
    write_run_manifest,
)


STAGE_ORDER = [
    "preflight",
    "ingestion",
    "cleaning",
    "preprocessing",
    "linkage",
    "cohort",
    "outcomes",
    "descriptive",
    "comparative",
    "clustering",
]


def _stage_log_path(cfg) -> Path:
    """Resolve the run-level stage-status audit CSV path."""
    audit_dir = output_path(cfg, "audit_dir")
    audit_dir.mkdir(parents=True, exist_ok=True)
    return audit_dir / "stage_status.csv"


def _record_stage(rows: list[dict[str, Any]], cfg, **row: Any) -> None:
    """Append one aggregate stage execution status row and persist the run status file."""
    rows.append(row)
    pd.DataFrame(rows).to_csv(_stage_log_path(cfg), index=False)


def run_tre_workflow(
    workflow_path: str | Path = "config/workflow_tre.yaml",
    pipeline_path: str | Path = "config/pipeline_tre.yaml",
    clustering_path: str | Path = "config/clustering_tre.yaml",
    *,
    from_stage: str = "preflight",
    to_stage: str = "clustering",
) -> dict[str, Any]:
    """Run an ordered section of the TRE workflow.

    Parameters
    ----------
    from_stage, to_stage:
        Any names in ``STAGE_ORDER``.  These allow an analyst to resume from a
        validated checkpoint without modifying code.
    """
    if from_stage not in STAGE_ORDER or to_stage not in STAGE_ORDER:
        raise ValueError(f"Stages must be one of {STAGE_ORDER}")
    if STAGE_ORDER.index(from_stage) > STAGE_ORDER.index(to_stage):
        raise ValueError("from_stage must not occur after to_stage")

    cfg = load_workflow_config(workflow_path)
    manifest = build_run_manifest(
        cfg,
        pipeline_config=pipeline_path,
        stage=f"{from_stage}->{to_stage}",
    )
    clustering_resolved = Path(clustering_path).resolve()
    manifest["clustering_config"] = str(clustering_resolved)
    if clustering_resolved.exists():
        from bth_analysis.workflow import config_sha256
        manifest["clustering_config_sha256"] = config_sha256(clustering_resolved)
    manifest["workflow_type"] = "TRE real-data translation"
    write_run_manifest(cfg, manifest, "tre_run_manifest.json")

    stages: dict[str, Callable[[], Any]] = {
        "preflight": lambda: run_preflight(workflow_path, pipeline_path, fail_on_blocker=True),
        "ingestion": lambda: run_ingestion(pipeline_path),
        "cleaning": lambda: run_cleaning(pipeline_path),
        "preprocessing": lambda: run_preprocessing(pipeline_path),
        "linkage": lambda: run_linkage(workflow_path),
        "cohort": lambda: run_cohort_index(workflow_path),
        "outcomes": lambda: run_outcome_features(workflow_path),
        "descriptive": lambda: run_descriptive(workflow_path),
        "comparative": lambda: run_comparative(workflow_path),
        "clustering": lambda: run_clustering(workflow_path, clustering_path),
    }

    selected = STAGE_ORDER[
        STAGE_ORDER.index(from_stage): STAGE_ORDER.index(to_stage) + 1
    ]


    status_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}

    print("ACTIVE BLACKPOOL / BTH — TRE WORKFLOW")
    print("=" * 88)
    print("Stages:", " -> ".join(selected))

    for stage in selected:
        started = datetime.now(timezone.utc)
        print("\n" + "#" * 88)
        print(f"RUNNING STAGE: {stage.upper()}")
        print("#" * 88)
        try:
            results[stage] = stages[stage]()
            ended = datetime.now(timezone.utc)
            _record_stage(
                status_rows,
                cfg,
                stage=stage,
                status="PASS",
                started_utc=started.isoformat(),
                ended_utc=ended.isoformat(),
                elapsed_seconds=(ended - started).total_seconds(),
                error="",
            )
        except Exception as exc:
            ended = datetime.now(timezone.utc)
            _record_stage(
                status_rows,
                cfg,
                stage=stage,
                status="FAIL",
                started_utc=started.isoformat(),
                ended_utc=ended.isoformat(),
                elapsed_seconds=(ended - started).total_seconds(),
                error=f"{type(exc).__name__}: {exc}",
            )
            # Store a traceback inside the TRE audit folder.  This may contain
            # filenames but should never contain patient-level row dumps.
            audit_dir = output_path(cfg, "audit_dir")
            (audit_dir / f"failure_{stage}.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            raise

    print("\n" + "=" * 88)
    print("TRE WORKFLOW COMPLETE")
    print(f"Stage audit: {_stage_log_path(cfg)}")
    print("Reviewer summary: python scripts/review_audit_summary.py")
    print("Release pre-screen (after reviewed analysis): python scripts/run_10_release_audit.py")
    return results
