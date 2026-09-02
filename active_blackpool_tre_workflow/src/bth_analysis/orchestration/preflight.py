"""TRE preflight checks.

Preflight is intentionally separate from data analysis.  It verifies that the
approved source files, schemas, time window and interpretation semantics are in
place before any patient-level analytical output is created.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bth_analysis.data_pipeline.config import load_pipeline_config, source_dir
from bth_analysis.data_pipeline.mapping import canonical_header
from bth_analysis.data_pipeline.schemas import TABLE_SCHEMAS
from bth_analysis.workflow import load_workflow_config, output_path
from bth_analysis.audit import (
    dataframe_preview,
    metric,
    save_stage_summary,
    section,
    stage_footer,
    stage_header,
)


def _header(path: Path, table_cfg: dict[str, Any]) -> list[str]:
    """Read and canonicalise only a source-file header for preflight schema checks."""
    fmt = str(table_cfg.get("format", path.suffix.lstrip(".") or "csv")).lower()
    if fmt in {"csv", "txt"}:
        csv_cfg = table_cfg.get("csv", {}) or {}
        return list(pd.read_csv(
            path,
            nrows=0,
            encoding=csv_cfg.get("encoding", "utf-8"),
            sep=csv_cfg.get("sep", ","),
        ).columns)
    if fmt in {"parquet", "pq"}:
        # Reading zero rows from parquet is not consistently supported across
        # engines, so read the schema through pyarrow/pandas metadata when available.
        try:
            import pyarrow.parquet as pq
            return list(pq.ParquetFile(path).schema.names)
        except Exception:
            return list(pd.read_parquet(path).columns)
    raise ValueError(f"Unsupported input format {fmt!r}: {path}")


def run_preflight(
    workflow_path: str | Path = "config/workflow_tre.yaml",
    pipeline_path: str | Path = "config/pipeline_tre.yaml",
    *,
    fail_on_blocker: bool = True,
) -> pd.DataFrame:
    """Run source, schema and methodological-readiness checks."""
    workflow = load_workflow_config(workflow_path)
    pipeline = load_pipeline_config(pipeline_path)
    src = source_dir(pipeline)

    stage_header(
        "00",
        "TRE PREFLIGHT / TRANSLATION READINESS",
        purpose=(
            "Check source-file presence, canonical headers, identifier candidates, approved real-data observation window, "
            "analysis-group/index semantics and prespecified design settings before any patient-level analytical output is created."
        ),
        inputs=[workflow_path, pipeline_path, src],
        outputs=[output_path(workflow, "qa_dir") / "00_tre_preflight.csv"],
    )

    rows: list[dict[str, Any]] = []

    def add(check: str, passed: bool, severity: str, detail: str) -> None:
        """Append one aggregate preflight check row to the readiness audit."""
        rows.append({
            "check": check,
            "passed": int(bool(passed)),
            "severity": severity,
            "detail": detail,
        })

    add("tre_source_directory_exists", src.exists(), "BLOCKER", str(src))

    # Validate file presence and canonical headers without loading patient rows.
    if src.exists():
        for table_key, table_cfg in pipeline.get("tables", {}).items():
            source_path = src / table_cfg["filename"]
            exists = source_path.exists()
            add(f"{table_key}:source_file_exists", exists, "BLOCKER", str(source_path))
            if not exists:
                continue
            try:
                header = _header(source_path, table_cfg)
                canonical = canonical_header(header, table_cfg)
                expected = list(table_cfg.get("columns") or TABLE_SCHEMAS[table_key])
                missing = [c for c in expected if c not in canonical]
                add(
                    f"{table_key}:required_columns_present",
                    not missing,
                    "BLOCKER",
                    "missing=" + repr(missing),
                )

                # At least one configured patient identifier candidate must be
                # physically present.  The final semantic choice is verified by
                # cross-source overlap during cleaning/preprocessing.
                patient_candidates = []
                for c in [table_cfg.get("patient_id"), *(table_cfg.get("patient_id_candidates", []) or [])]:
                    if c and c not in patient_candidates:
                        patient_candidates.append(c)
                patient_present = [c for c in patient_candidates if c in header]
                add(
                    f"{table_key}:patient_identifier_candidate_present",
                    bool(patient_present),
                    "BLOCKER",
                    f"configured={patient_candidates}; present={patient_present}",
                )

                event_candidates = []
                for c in [table_cfg.get("event_id"), *(table_cfg.get("event_id_candidates", []) or [])]:
                    if c and c not in event_candidates:
                        event_candidates.append(c)
                if event_candidates:
                    event_present = [c for c in event_candidates if c in header]
                    add(
                        f"{table_key}:event_identifier_candidate_present",
                        bool(event_present),
                        "BLOCKER",
                        f"configured={event_candidates}; present={event_present}",
                    )
            except Exception as exc:
                add(
                    f"{table_key}:header_readable",
                    False,
                    "BLOCKER",
                    f"{type(exc).__name__}: {exc}",
                )

    project = workflow.get("project", {})
    start_raw = project.get("study_start_date")
    end_raw = project.get("study_end_date")
    if start_raw in (None, "") or end_raw in (None, ""):
        add(
            "real_tre_study_window_confirmed",
            False,
            "BLOCKER",
            "study_start_date/study_end_date are intentionally unset. Fill them from "
            "approved BTH/TRE extract coverage documentation.",
        )
    else:
        try:
            start = pd.Timestamp(start_raw)
            end = pd.Timestamp(end_raw)
            add("real_tre_study_window_confirmed", start < end, "BLOCKER", f"{start} -> {end}")
        except Exception as exc:
            add("real_tre_study_window_confirmed", False, "BLOCKER", f"Invalid dates: {exc}")

    cohort = workflow.get("cohort", {})
    add(
        "analysis_group_semantics_confirmed_for_workflow",
        bool(cohort.get("analysis_group_semantics_confirmed_for_workflow", False)),
        "BLOCKER",
        "Must be confirmed from the real TRE source documentation before comparative modelling.",
    )
    add(
        "analytical_index_semantics_confirmed_for_workflow",
        bool(cohort.get("analytical_index_semantics_confirmed_for_workflow", False)),
        "BLOCKER",
        "FirstMSKDate/source-relative index meaning must be confirmed for the real extract.",
    )
    add(
        "programme_start_not_assumed",
        not bool(cohort.get("index_is_programme_start", False))
        or bool(cohort.get("programme_start_date_available", False)),
        "BLOCKER",
        "Do not label the analytical FirstMSKDate index as programme start unless a confirmed programme-start field exists.",
    )
    add(
        "baseline_rule_explicit",
        int(cohort.get("baseline_days", 0)) > 0 and "require_full_baseline" in cohort,
        "BLOCKER",
        repr({
            "baseline_days": cohort.get("baseline_days"),
            "require_full_baseline": cohort.get("require_full_baseline"),
        }),
    )
    add(
        "followup_rule_explicit",
        int(cohort.get("followup_days", 0)) > 0 and "require_full_followup" in cohort,
        "BLOCKER",
        repr({
            "followup_days": cohort.get("followup_days"),
            "require_full_followup": cohort.get("require_full_followup"),
        }),
    )

    prop = workflow.get("propensity", {})
    covariates = list(prop.get("covariates", []))
    forbidden = [
        c for c in covariates
        if c.startswith("FollowUp") or c in {"ExposureFlag", "AnalysisGroup", "UtilisationCluster"}
    ]
    add(
        "propensity_covariates_are_pre_index",
        not forbidden,
        "BLOCKER",
        "forbidden=" + repr(forbidden),
    )
    add(
        "balance_threshold_configured",
        float(prop.get("balance_abs_smd_threshold", 0.0)) > 0,
        "BLOCKER",
        f"threshold={prop.get('balance_abs_smd_threshold')}",
    )

    comparative = workflow.get("comparative", {})
    add(
        "comparative_outcomes_configured",
        bool(comparative.get("outcomes")),
        "BLOCKER",
        repr(comparative.get("outcomes", [])),
    )

    result = pd.DataFrame(rows)
    qa_dir = output_path(workflow, "qa_dir")
    qa_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(qa_dir / "00_tre_preflight.csv", index=False)

    blockers = result[(result["severity"].eq("BLOCKER")) & result["passed"].eq(0)]
    passed_n = int(result["passed"].sum())
    failed_n = int(len(result) - passed_n)

    section("STAGE 00 KEY FINDINGS")
    metric("checks run", len(result))
    metric("checks passed", passed_n)
    metric("checks failed", failed_n)
    metric("blocking failures", len(blockers))
    dataframe_preview(result, max_rows=80)

    if not blockers.empty:
        print("\nBLOCKERS TO RESOLVE BEFORE INGESTION:")
        dataframe_preview(blockers, max_rows=40)

    audit_dir = output_path(workflow, "audit_dir")
    status = "PASS" if blockers.empty else "BLOCKED"
    next_command = (
        "python scripts/run_01_ingestion.py"
        if blockers.empty
        else "Resolve every BLOCKER in outputs/qa/00_tre_preflight.csv, then rerun python scripts/run_00_preflight.py"
    )
    summary_path = save_stage_summary(
        audit_dir,
        stage_key="preflight",
        stage_code="00",
        title="TRE preflight / translation readiness",
        status=status,
        key_findings={
            "checks_run": len(result),
            "checks_passed": passed_n,
            "checks_failed": failed_n,
            "blockers_n": len(blockers),
        },
        qa_files=[qa_dir / "00_tre_preflight.csv"],
        warnings=[
            "The BTH study window must come from approved extract documentation.",
            "The source-relative FirstMSKDate index remains blocked until its real-data semantics are confirmed."
        ],
        next_command=next_command,
        config_path=workflow_path,
    )
    stage_footer(
        stage_key="preflight",
        audit_dir=audit_dir,
        summary_path=summary_path,
        qa_files=[qa_dir / "00_tre_preflight.csv"],
        warnings=["No patient-level stage should run while any preflight BLOCKER remains."] if not blockers.empty else [],
        next_command=next_command,
    )

    if not blockers.empty and fail_on_blocker:
        raise RuntimeError(
            "TRE preflight failed. Resolve the BLOCKER rows in "
            f"{qa_dir / '00_tre_preflight.csv'} before running the full workflow."
        )
    return result
