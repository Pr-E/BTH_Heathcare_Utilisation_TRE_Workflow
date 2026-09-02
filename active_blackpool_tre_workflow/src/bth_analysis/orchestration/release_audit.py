"""Pre-screen candidate outputs before formal TRE disclosure control.

This is *not* a disclosure-control engine.  It simply helps analysts identify
obvious patient-level files and small aggregate cells before submitting outputs
to the TRE's approved disclosure-review process.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from bth_analysis.workflow import load_workflow_config, output_path, resolve_path
from bth_analysis.audit import (
    dataframe_preview,
    metric,
    save_stage_summary,
    section as audit_section,
    stage_footer,
    stage_header,
)


COUNT_NAME_TOKENS = {
    "n", "patients", "patient_n", "count", "events", "event_n", "matched_sets_n",
    "comparison_n", "sports_linked_n", "wider_msk_n", "minimum_cluster_n",
}


def _looks_like_count_column(name: str) -> bool:
    """Identify aggregate columns that should be inspected for small-cell disclosure risk."""
    low = name.lower()
    return (
        low in COUNT_NAME_TOKENS
        or low.endswith("_n")
        or low.endswith("_count")
        or low.endswith("_patients")
        or low.endswith("_events")
    )


def run_release_audit(
    workflow_path: str | Path = "config/workflow_tre.yaml",
    release_config_path: str | Path = "config/release_audit.yaml",
) -> pd.DataFrame:
    """Pre-screen outputs for patient-level/internal files and possible small-cell aggregate risks."""
    workflow = load_workflow_config(workflow_path)
    release_cfg = yaml.safe_load(Path(release_config_path).read_text(encoding="utf-8")) or {}
    section: dict[str, Any] = release_cfg.get("release_audit", {})

    stage_header(
        "10",
        "RELEASE OUTPUT PRE-SCREEN",
        purpose=(
            "Identify obvious patient-level/internal-only files and aggregate CSVs containing non-zero small cells "
            "before submission to the TRE's formal disclosure-control process. This is a pre-screen, not disclosure approval."
        ),
        inputs=section.get("scan_directories", []),
        outputs=[output_path(workflow, "release_audit_dir") / "release_output_prescreen.csv"],
    )

    threshold = int(section.get("minimum_cell_count", 5))
    patterns = list(section.get("patient_level_filename_patterns", []))

    records: list[dict[str, Any]] = []
    for directory in section.get("scan_directories", []):
        base = resolve_path(workflow, directory)
        if not base.exists():
            records.append({
                "file": str(base),
                "status": "NOT_FOUND",
                "patient_level_name_flag": 0,
                "small_cell_flag": 0,
                "small_cell_details": "",
                "note": "Configured scan directory does not exist yet.",
            })
            continue

        for path in sorted(base.rglob("*.csv")):
            patient_level = any(fnmatch.fnmatch(path.name.lower(), pat.lower()) for pat in patterns)
            small_details: list[str] = []

            # Do not read obvious patient-level files for release scanning.  They are
            # automatically classified as internal-only.
            if not patient_level:
                try:
                    df = pd.read_csv(path, low_memory=False)
                    for col in df.columns:
                        if not _looks_like_count_column(col):
                            continue
                        x = pd.to_numeric(df[col], errors="coerce")
                        small = x[(x > 0) & (x < threshold)]
                        if len(small):
                            small_details.append(
                                f"{col}: {len(small)} non-zero cells below {threshold}"
                            )
                except Exception as exc:
                    small_details.append(f"scan_error:{type(exc).__name__}")

            records.append({
                "file": str(path.relative_to(workflow["_project_root"])),
                "status": "INTERNAL_ONLY" if patient_level else "REQUIRES_FORMAL_DISCLOSURE_REVIEW",
                "patient_level_name_flag": int(patient_level),
                "small_cell_flag": int(bool(small_details)),
                "small_cell_details": "; ".join(small_details),
                "note": (
                    "Patient-level/assignment output: do not request egress."
                    if patient_level
                    else "Aggregate pre-screen only; formal TRE disclosure review still required."
                ),
            })

    result = pd.DataFrame(records)
    out_dir = output_path(workflow, "release_audit_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "release_output_prescreen.csv", index=False)

    patient_level_n = int(result["patient_level_name_flag"].sum()) if not result.empty else 0
    small_cell_n = int(result["small_cell_flag"].sum()) if not result.empty else 0
    aggregate_review_n = int(result["status"].eq("REQUIRES_FORMAL_DISCLOSURE_REVIEW").sum()) if not result.empty else 0

    audit_section("STAGE 10 KEY FINDINGS")
    metric("CSV files/directories scanned", len(result))
    metric("patient-level/internal-only files flagged", patient_level_n)
    metric("aggregate files with possible small cells", small_cell_n)
    metric("aggregate files requiring formal review", aggregate_review_n)
    dataframe_preview(result, max_rows=80)

    audit_dir = output_path(workflow, "audit_dir")
    summary_path = save_stage_summary(
        audit_dir,
        stage_key="release_audit",
        stage_code="10",
        title="Release output pre-screen",
        status="PRE_SCREEN_COMPLETE",
        key_findings={
            "records_scanned": len(result),
            "patient_level_internal_only_n": patient_level_n,
            "possible_small_cell_files_n": small_cell_n,
            "aggregate_files_requiring_formal_review_n": aggregate_review_n,
            "configured_minimum_cell_count": threshold,
        },
        qa_files=[out_dir / "release_output_prescreen.csv"],
        warnings=[
            "This pre-screen does not approve disclosure or egress.",
            "Patient-level files remain inside the TRE and must not be requested for release.",
            "All candidate aggregate outputs still require the organisation's formal TRE disclosure-control process."
        ],
        next_command="Submit approved aggregate outputs to the formal TRE disclosure-control / egress review process",
        config_path=release_config_path,
    )
    stage_footer(
        stage_key="release_audit",
        audit_dir=audit_dir,
        summary_path=summary_path,
        qa_files=[out_dir / "release_output_prescreen.csv"],
        warnings=["PRE-SCREEN ONLY - not disclosure approval."],
        next_command="Formal TRE disclosure-control review / approved export workflow",
    )
    return result
