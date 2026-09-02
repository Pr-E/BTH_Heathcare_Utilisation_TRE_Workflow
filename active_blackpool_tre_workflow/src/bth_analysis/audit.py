"""Patient-safe audit and terminal-logging helpers for the TRE workflow.

Each stage uses these helpers to print consistent aggregate findings, write a
compact JSON/Markdown stage summary, record QA file locations, and show the next
command. Patient IDs, hashes and row-level clinical records must never be passed
to these summary functions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


# Canonical command order used throughout the package and printed after each run.
NEXT_COMMANDS: dict[str, str] = {
    "preflight": "python scripts/run_01_ingestion.py",
    "ingestion": "python scripts/run_02_cleaning.py",
    "cleaning": "python scripts/run_03_preprocessing.py",
    "preprocessing": "python scripts/run_04_linkage.py",
    "linkage": "python scripts/run_05_cohort.py",
    "cohort": "python scripts/run_06_outcomes.py",
    "outcomes": "python scripts/run_07_descriptive.py",
    "descriptive": "python scripts/run_08_comparative.py",
    "comparative": "python scripts/run_09_clustering.py",
    "clustering": "python scripts/run_10_release_audit.py",
    "release_audit": "Formal TRE disclosure-control review / approved export workflow",
}


def _json_safe(value: Any) -> Any:
    """Convert common pandas/numpy values into JSON-safe aggregate values."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def stage_header(
    stage_code: str,
    title: str,
    *,
    purpose: str,
    inputs: Iterable[str | Path] | None = None,
    outputs: Iterable[str | Path] | None = None,
) -> None:
    """Print a consistent stage header so terminal logs are easy to scan."""
    print("\n" + "=" * 96)
    print(f"{stage_code} | {title}")
    print("=" * 96)
    print(f"PURPOSE: {purpose}")
    if inputs:
        print("INPUTS:")
        for item in inputs:
            print(f"  - {item}")
    if outputs:
        print("OUTPUTS:")
        for item in outputs:
            print(f"  - {item}")


def section(title: str) -> None:
    """Print a visually distinct subsection inside a stage log."""
    print("\n" + "-" * 96)
    print(title)
    print("-" * 96)


def metric(label: str, value: Any, *, suffix: str = "") -> None:
    """Print one aggregate key-value result with aligned labels."""
    text = "NA" if value is None else str(value)
    print(f"  {label:<46} {text}{suffix}")


def dataframe_preview(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    max_rows: int = 12,
    index: bool = False,
) -> None:
    """Print a bounded aggregate table preview; never use with patient-level rows."""
    if df is None or df.empty:
        print("  <no rows>")
        return
    view = df.copy()
    if columns:
        keep = [c for c in columns if c in view.columns]
        if keep:
            view = view[keep]
    if len(view) > max_rows:
        view = view.head(max_rows)
    print(view.to_string(index=index))


def save_stage_summary(
    audit_dir: str | Path,
    *,
    stage_key: str,
    stage_code: str,
    title: str,
    status: str,
    key_findings: Mapping[str, Any],
    qa_files: Iterable[str | Path] = (),
    warnings: Iterable[str] = (),
    next_command: str | None = None,
    config_path: str | Path | None = None,
) -> Path:
    """Write one compact JSON stage summary containing no patient-level records."""
    audit_dir = Path(audit_dir)
    stage_dir = audit_dir / "stage_summaries"
    stage_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage_key": stage_key,
        "stage_code": stage_code,
        "title": title,
        "status": status,
        "config_path": str(config_path) if config_path else None,
        "key_findings": _json_safe(dict(key_findings)),
        "qa_files": [str(x) for x in qa_files],
        "warnings": list(warnings),
        "next_command": next_command or NEXT_COMMANDS.get(stage_key),
        "patient_level_data_in_summary": False,
    }

    path = stage_dir / f"{stage_code}_{stage_key}_summary.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # Write the same aggregate audit information as plain Markdown so a reviewer
    # can read the stage result without opening JSON or patient-level data.
    md_lines = [
        f"# {stage_code} - {title}",
        "",
        f"**Status:** {status}",
        f"**Timestamp (UTC):** {payload['timestamp_utc']}",
        "",
        "## Key findings",
        "",
    ]
    for key, value in payload["key_findings"].items():
        md_lines.append(f"- **{key}:** {value}")
    if payload["warnings"]:
        md_lines.extend(["", "## Warnings / interpretation boundaries", ""])
        md_lines.extend([f"- {item}" for item in payload["warnings"]])
    if payload["qa_files"]:
        md_lines.extend(["", "## QA / output files", ""])
        md_lines.extend([f"- `{item}`" for item in payload["qa_files"]])
    if payload["next_command"]:
        md_lines.extend(["", "## Next step", "", f"`{payload['next_command']}`"] )
    md_path = stage_dir / f"{stage_code}_{stage_key}_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return path


def stage_footer(
    *,
    stage_key: str,
    audit_dir: str | Path,
    summary_path: str | Path | None = None,
    qa_files: Iterable[str | Path] = (),
    warnings: Iterable[str] = (),
    next_command: str | None = None,
) -> None:
    """Print the audit trail and the exact next workflow command."""
    section("AUDIT / HANDOFF")
    if summary_path:
        print(f"  Stage summary: {summary_path}")
    for qa in qa_files:
        print(f"  QA/output:      {qa}")
    warnings = list(warnings)
    if warnings:
        print("  WARNINGS / INTERPRETATION BOUNDARIES:")
        for warning in warnings:
            print(f"    - {warning}")
    command = next_command or NEXT_COMMANDS.get(stage_key)
    if command:
        print("\nNEXT STEP")
        print(f"  {command}")
    print("=" * 96)
