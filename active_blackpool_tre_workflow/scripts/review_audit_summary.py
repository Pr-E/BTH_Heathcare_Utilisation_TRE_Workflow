"""Build one reviewer-facing audit summary from aggregate per-stage JSON files.

Why this script exists
----------------------
Ian or another reviewer should be able to understand the status of a TRE run
without opening patient-level analytical datasets.  Every workflow stage writes
one compact JSON summary under ``outputs/audit/stage_summaries``.  This script
combines those aggregate summaries into a CSV and a readable Markdown handover.

Privacy boundary
----------------
This script reads only stage-summary JSON files created by ``bth_analysis.audit``.
It does not open patient-level patient spine, outcome, propensity, matched or
cluster-assignment files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


# Resolve paths from the repository location so the script works after TRE
# ingress without requiring analyst-specific absolute paths.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"
SUMMARY_DIR = AUDIT_DIR / "stage_summaries"


def _markdown_table(headers: list[str], rows: Iterable[list[object]]) -> list[str]:
    """Return a dependency-free Markdown table.

    Pandas ``DataFrame.to_markdown`` requires the optional ``tabulate`` package.
    We avoid that extra dependency so the reviewer utility works in a restricted
    TRE Python environment after the core requirements are installed.
    """
    def clean(value: object) -> str:
        text = "" if value is None else str(value)
        # Escape Markdown table separators so a value cannot break the layout.
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(v) for v in row) + " |")
    return lines


def main() -> None:
    """Create ``reviewer_summary.csv`` and ``reviewer_summary.md``.

    The summary deliberately contains stage status, aggregate key findings,
    warnings and the next workflow command only.  It is suitable for internal
    review but still remains subject to local TRE governance before export.
    """
    if not SUMMARY_DIR.exists():
        raise FileNotFoundError(
            f"No stage summaries found at {SUMMARY_DIR}. "
            "Run one or more workflow stages first."
        )

    rows: list[dict[str, object]] = []
    payloads: list[dict[str, object]] = []

    # Stage filenames begin with 00, 01, 02 ...; lexical sorting therefore
    # preserves the intended pipeline order for a reviewer.
    for path in sorted(SUMMARY_DIR.glob("*_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads.append(payload)
        rows.append(
            {
                "stage_code": payload.get("stage_code"),
                "stage_key": payload.get("stage_key"),
                "title": payload.get("title"),
                "status": payload.get("status"),
                "timestamp_utc": payload.get("timestamp_utc"),
                "warnings_n": len(payload.get("warnings") or []),
                "qa_files_n": len(payload.get("qa_files") or []),
                "next_command": payload.get("next_command"),
            }
        )

    # CSV gives a compact machine-readable status matrix.
    summary = pd.DataFrame(rows)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = AUDIT_DIR / "reviewer_summary.csv"
    summary.to_csv(csv_path, index=False)

    # Markdown gives the reviewer the same status plus every stage's key
    # findings and interpretation warnings without requiring Python.
    md: list[str] = [
        "# Active Blackpool / BTH - TRE run audit summary",
        "",
        "This report is assembled only from aggregate stage-audit summaries. "
        "It does not open patient-level analytical datasets.",
        "",
        "## Stage status",
        "",
    ]

    if summary.empty:
        md.append("No stage summaries were found.")
    else:
        headers = list(summary.columns)
        md.extend(_markdown_table(headers, summary.itertuples(index=False, name=None)))

    for payload in payloads:
        md.extend(
            [
                "",
                f"## {payload.get('stage_code')} - {payload.get('title')}",
                "",
                f"**Status:** {payload.get('status')}",
                "",
                "### Key findings",
                "",
            ]
        )
        for key, value in (payload.get("key_findings") or {}).items():
            md.append(f"- **{key}:** {value}")

        warnings = payload.get("warnings") or []
        if warnings:
            md.extend(["", "### Warnings / interpretation boundaries", ""])
            md.extend([f"- {warning}" for warning in warnings])

        qa_files = payload.get("qa_files") or []
        if qa_files:
            md.extend(["", "### QA / output files", ""])
            md.extend([f"- `{path}`" for path in qa_files])

        next_command = payload.get("next_command")
        if next_command:
            md.extend(["", "### Next step", "", f"`{next_command}`"])

    md_path = AUDIT_DIR / "reviewer_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("REVIEW AUDIT SUMMARY")
    print("=" * 88)
    if not summary.empty:
        print(summary.to_string(index=False))
    print(f"CSV: {csv_path}")
    print(f"MD:  {md_path}")


if __name__ == "__main__":
    # Keep script execution explicit: importing this module does not create files.
    main()
