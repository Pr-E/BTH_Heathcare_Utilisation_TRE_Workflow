"""Small aggregate QA helpers shared by source-data pipeline stages.

These functions intentionally return or print table-level counts only.  They do
not expose patient hashes or row-level clinical values.  Their role is to make
basic schema/data-quality state visible at every checkpoint and to persist
aggregate QA tables in a consistent format.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Any

import pandas as pd


def save_records(records: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Write a list of aggregate QA dictionaries to CSV.

    Parameters
    ----------
    records:
        Aggregate dictionaries such as row counts, chronology flags or balance
        diagnostics.  Callers must not pass patient-level rows.
    path:
        Destination inside the TRE project; parent directories are created.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def basic_table_summary(
    df: pd.DataFrame,
    table_key: str,
    table_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Return patient-safe table dimensions and identifier completeness metrics.

    Identifier *values* are never included.  We report only uniqueness and
    missingness counts for the configured patient/event/spell/episode columns.
    """
    # Pull the semantic key columns selected for this source table.
    patient_id = table_cfg.get("patient_id")
    event_id = table_cfg.get("event_id")
    spell_id = table_cfg.get("spell_id")
    episode_id = table_cfg.get("episode_id")

    # Compute table-level dimensions.  Total-cell context makes raw missing-cell
    # counts interpretable across narrow MSK and wide hospital extracts.
    rows = int(len(df))
    columns = int(df.shape[1])
    total_cells = rows * columns
    missing_cells = int(df.isna().sum().sum())

    out: dict[str, Any] = {
        "table": table_key,
        "rows": rows,
        "columns": columns,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_cells_pct": (
            100.0 * missing_cells / total_cells if total_cells else 0.0
        ),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "fully_blank_rows": int(df.isna().all(axis=1).sum()),
    }

    # Report completeness/uniqueness only when the configured key is present.
    if patient_id and patient_id in df.columns:
        out["unique_patients"] = int(df[patient_id].nunique(dropna=True))
        out["missing_patient_ids"] = int(df[patient_id].isna().sum())

    if event_id and event_id in df.columns:
        out["unique_events"] = int(df[event_id].nunique(dropna=True))
        out["missing_event_ids"] = int(df[event_id].isna().sum())

    if spell_id and spell_id in df.columns:
        out["unique_spells"] = int(df[spell_id].nunique(dropna=True))

    if episode_id and episode_id in df.columns:
        out["unique_episodes"] = int(df[episode_id].nunique(dropna=True))

    return out


def print_table_summary(summary: Mapping[str, Any], prefix: str = "") -> None:
    """Print the most useful aggregate table QA metrics in a stable format."""
    lead = f"{prefix} " if prefix else ""
    print(
        f"{lead}{summary['table']}: "
        f"rows={summary['rows']:,}, cols={summary['columns']}"
    )

    if "unique_patients" in summary:
        print(
            f"  unique patients={summary['unique_patients']:,}; "
            f"missing patient IDs={summary['missing_patient_ids']:,}"
        )

    if "unique_events" in summary:
        print(
            f"  unique events={summary['unique_events']:,}; "
            f"missing event IDs={summary['missing_event_ids']:,}"
        )

    if "unique_spells" in summary:
        print(f"  unique spells={summary['unique_spells']:,}")

    print(
        f"  missing cells={summary['missing_cells']:,}/"
        f"{summary.get('total_cells', 0):,} "
        f"({float(summary.get('missing_cells_pct', 0.0)):.2f}%); "
        f"exact duplicates={summary['exact_duplicate_rows']:,}; "
        f"blank rows={summary['fully_blank_rows']:,}"
    )
