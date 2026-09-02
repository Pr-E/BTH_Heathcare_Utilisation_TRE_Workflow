"""Resolve patient and event identifiers across the six TRE source tables.

Patient identifiers in hospital extracts are verified against the matching MSK
cohort using aggregate overlap. ED event identifiers are selected separately
using uniqueness and separation from the patient identifier. Hash values are
never written to QA outputs; only selected column names and aggregate diagnostics
are recorded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class IdentifierChoice:
    """Selected identifier columns for one source table."""

    patient_id: str
    event_id: str | None = None


def _dedupe(values: Iterable[str | None]) -> list[str]:
    """Return non-empty values once, preserving configured preference order."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value:
            continue
        value = str(value)
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _candidate_columns(table_cfg: dict[str, Any], role: str) -> list[str]:
    """Return configured preferred identifier followed by fallback candidates."""
    if role == "patient":
        return _dedupe([
            table_cfg.get("patient_id"),
            *(table_cfg.get("patient_id_candidates", []) or []),
        ])
    if role == "event":
        return _dedupe([
            table_cfg.get("event_id"),
            *(table_cfg.get("event_id_candidates", []) or []),
        ])
    raise ValueError(f"Unknown identifier role: {role!r}")


def _as_id_set(series: pd.Series) -> set[str]:
    """Convert an identifier column to a set without materialising missing tokens."""
    return set(series.dropna().astype("string").astype(str))


def _score_candidate(
    series: pd.Series,
    *,
    reference_ids: set[str] | None,
) -> dict[str, float | int]:
    """Create aggregate diagnostics for one candidate identifier column."""
    non_missing = int(series.notna().sum())
    unique_n = int(series.nunique(dropna=True))
    values = _as_id_set(series)
    overlap_n = int(len(values & reference_ids)) if reference_ids is not None else 0
    return {
        "non_missing_n": non_missing,
        "unique_n": unique_n,
        "uniqueness_ratio": (unique_n / non_missing) if non_missing else 0.0,
        "reference_overlap_n": overlap_n,
        "reference_overlap_pct_of_candidate_unique": (
            overlap_n / unique_n * 100.0 if unique_n else 0.0
        ),
        "reference_overlap_pct_of_reference": (
            overlap_n / len(reference_ids) * 100.0 if reference_ids else 0.0
        ),
    }


def choose_patient_identifier(
    df: pd.DataFrame,
    table_key: str,
    table_cfg: dict[str, Any],
    *,
    reference_ids: set[str] | None = None,
    fail_if_no_reference_overlap: bool = True,
    fail_if_ambiguous: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Select the patient identifier, using cohort overlap when available.

    Reference cohort tables (MSK source tables) use their configured preferred
    identifier when present.  Healthcare-event tables are verified against the
    matching MSK cohort and the candidate with the greatest observed overlap is
    selected.  This is safer than relying on a misleading source-column label.
    """
    candidates = _candidate_columns(table_cfg, "patient")
    present = [c for c in candidates if c in df.columns]
    if not present:
        raise ValueError(
            f"{table_key}: none of the configured patient identifier candidates "
            f"are present: {candidates}"
        )

    records: list[dict[str, Any]] = []
    for rank, column in enumerate(present):
        score = _score_candidate(df[column], reference_ids=reference_ids)
        records.append({
            "table": table_key,
            "identifier_role": "patient",
            "candidate": column,
            "configured_preference_rank": rank + 1,
            **score,
        })

    # Reference cohort itself: use configured preference, since there is no
    # external reference set against which to validate it at this stage.
    if reference_ids is None:
        selected = present[0]
    else:
        ordered = sorted(
            records,
            key=lambda r: (
                int(r["reference_overlap_n"]),
                float(r["reference_overlap_pct_of_candidate_unique"]),
                int(r["unique_n"]),
                -int(r["configured_preference_rank"]),
            ),
            reverse=True,
        )
        selected = str(ordered[0]["candidate"])
        best_overlap = int(ordered[0]["reference_overlap_n"])

        if fail_if_no_reference_overlap and best_overlap == 0:
            raise ValueError(
                f"{table_key}: no patient identifier candidate overlaps the configured "
                "reference MSK cohort. Stop and review source semantics before linkage."
            )

        if fail_if_ambiguous and len(ordered) > 1:
            second_overlap = int(ordered[1]["reference_overlap_n"])
            if best_overlap > 0 and second_overlap == best_overlap:
                raise ValueError(
                    f"{table_key}: patient identifier resolution is ambiguous: "
                    f"{ordered[0]['candidate']!r} and {ordered[1]['candidate']!r} "
                    f"both overlap {best_overlap} reference patients."
                )

    for row in records:
        row["selected"] = int(row["candidate"] == selected)
    return selected, records


def choose_event_identifier(
    df: pd.DataFrame,
    table_key: str,
    table_cfg: dict[str, Any],
    *,
    selected_patient_id: str,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Select the event identifier, preferring a field distinct from patient ID.

    For ED sources, the remaining candidate with the greatest uniqueness is a
    practical attendance-ID check.  The result is still written to QA for review.
    Tables without an event identifier return ``None``.
    """
    candidates = _candidate_columns(table_cfg, "event")
    if not candidates:
        return None, []

    present = [c for c in candidates if c in df.columns]
    if not present:
        raise ValueError(
            f"{table_key}: none of the configured event identifier candidates "
            f"are present: {candidates}"
        )

    records: list[dict[str, Any]] = []
    for rank, column in enumerate(present):
        score = _score_candidate(df[column], reference_ids=None)
        records.append({
            "table": table_key,
            "identifier_role": "event",
            "candidate": column,
            "configured_preference_rank": rank + 1,
            **score,
        })

    distinct = [r for r in records if r["candidate"] != selected_patient_id]
    pool = distinct or records
    pool = sorted(
        pool,
        key=lambda r: (
            float(r["uniqueness_ratio"]),
            int(r["unique_n"]),
            -int(r["configured_preference_rank"]),
        ),
        reverse=True,
    )
    selected = str(pool[0]["candidate"])
    for row in records:
        row["selected"] = int(row["candidate"] == selected)
    return selected, records


def resolve_identifier_plan(
    tables: dict[str, pd.DataFrame],
    table_configs: dict[str, dict[str, Any]],
    *,
    fail_if_no_reference_overlap: bool = True,
    fail_if_ambiguous: bool = True,
) -> tuple[dict[str, IdentifierChoice], pd.DataFrame]:
    """Resolve identifiers for all configured real-TRE source tables.

    Resolution order is important: MSK cohort IDs are selected first, then each
    healthcare source is checked against the appropriate MSK reference cohort.
    """
    choices: dict[str, IdentifierChoice] = {}
    audit_rows: list[dict[str, Any]] = []

    # First resolve cohort/reference tables.
    reference_keys = [
        key for key, cfg in table_configs.items()
        if cfg.get("type") == "msk"
    ]
    for key in reference_keys:
        patient_id, rows = choose_patient_identifier(
            tables[key], key, table_configs[key], reference_ids=None,
            fail_if_no_reference_overlap=fail_if_no_reference_overlap,
            fail_if_ambiguous=fail_if_ambiguous,
        )
        event_id, event_rows = choose_event_identifier(
            tables[key], key, table_configs[key], selected_patient_id=patient_id,
        )
        choices[key] = IdentifierChoice(patient_id=patient_id, event_id=event_id)
        audit_rows.extend(rows)
        audit_rows.extend(event_rows)

    # Healthcare sources must name the MSK cohort used to verify their patient ID.
    # Missing reference mappings are treated as configuration errors rather than
    # silently falling back to the preferred hash column.
    for key, cfg in table_configs.items():
        if key in choices:
            continue
        reference_key = cfg.get("reference_cohort")
        if not reference_key:
            raise ValueError(
                f"{key}: reference_cohort is required for non-MSK source identifier verification."
            )
        if reference_key not in table_configs:
            raise ValueError(
                f"{key}: reference_cohort={reference_key!r} is not a configured table."
            )
        if table_configs[reference_key].get("type") != "msk":
            raise ValueError(
                f"{key}: reference_cohort={reference_key!r} must identify an MSK cohort table."
            )
        if reference_key not in choices:
            raise ValueError(
                f"{key}: reference_cohort={reference_key!r} has not been resolved."
            )
        ref_col = choices[reference_key].patient_id
        reference_ids = _as_id_set(tables[reference_key][ref_col])

        patient_id, rows = choose_patient_identifier(
            tables[key], key, cfg, reference_ids=reference_ids,
            fail_if_no_reference_overlap=fail_if_no_reference_overlap,
            fail_if_ambiguous=fail_if_ambiguous,
        )
        event_id, event_rows = choose_event_identifier(
            tables[key], key, cfg, selected_patient_id=patient_id,
        )
        choices[key] = IdentifierChoice(patient_id=patient_id, event_id=event_id)
        audit_rows.extend(rows)
        audit_rows.extend(event_rows)

    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit["reference_cohort"] = audit["table"].map(
            {k: cfg.get("reference_cohort") for k, cfg in table_configs.items()}
        )
    return choices, audit


def apply_identifier_choices_to_config(
    table_configs: dict[str, dict[str, Any]],
    choices: dict[str, IdentifierChoice],
) -> dict[str, dict[str, Any]]:
    """Return a copied table configuration with resolved ID fields inserted."""
    import copy

    resolved = copy.deepcopy(table_configs)
    for key, choice in choices.items():
        resolved[key]["patient_id"] = choice.patient_id
        if choice.event_id is not None:
            resolved[key]["event_id"] = choice.event_id
    return resolved
