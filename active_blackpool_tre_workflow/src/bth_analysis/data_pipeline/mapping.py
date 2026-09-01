"""Map real TRE source aliases to the stable canonical analytical schema.

The analytical code should not be rewritten because a refreshed BTH extract
changes capitalisation or an approved source column alias.  Instead, raw-to-
canonical mappings are declared in ``config/pipeline_tre.yaml`` and applied
before schema validation.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def apply_column_mapping(
    df: pd.DataFrame,
    table_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Rename configured source columns and block ambiguous canonical duplicates."""
    mapping = table_cfg.get("column_mapping", {}) or {}
    if not isinstance(mapping, dict):
        raise TypeError("table column_mapping must be a YAML mapping/dictionary")

    # Only rename raw columns that actually exist in this extract.  This lets a
    # mapping file contain aliases used by another refresh without causing a
    # KeyError, while later schema validation still enforces required fields.
    present_mapping = {
        raw: canonical
        for raw, canonical in mapping.items()
        if raw in df.columns
    }
    out = df.rename(columns=present_mapping).copy()

    # Two raw fields must never silently map to one canonical name; downstream
    # provenance would be impossible to audit.
    duplicated = out.columns[out.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            "Column mapping produced duplicate canonical column names: "
            f"{sorted(set(duplicated))}. Review column_mapping."
        )
    return out


def canonical_header(columns: list[str], table_cfg: dict[str, Any]) -> list[str]:
    """Apply the same rename map to source header names without loading row data."""
    mapping = table_cfg.get("column_mapping", {}) or {}
    return [mapping.get(column, column) for column in columns]
