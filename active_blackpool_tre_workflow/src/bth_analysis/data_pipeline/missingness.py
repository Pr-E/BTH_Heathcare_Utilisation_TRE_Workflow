"""Audit column-level missingness during deterministic TRE cleaning.

The module reports missing counts and percentages, flags analytically critical
fields, applies configured expected/conditional missingness rules, and compares
common fields between Sports-linked and Wider MSK source families. It does not
impute values. Fields without an explicit semantic rule are labelled
``UNCLASSIFIED`` for review against the BTH source documentation.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd


def _cfg(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the optional missingness subsection of pipeline configuration."""
    return config.get("missingness", {}) or {}


def _classification(
    table_key: str,
    column: str,
    table_cfg: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve the configured interpretation of missingness for one field."""
    rules = ((_cfg(config).get("rules", {}) or {}).get(table_key, {}) or {})
    explicit = rules.get(column)

    if isinstance(explicit, str):
        return explicit.upper(), "explicit pipeline configuration"
    if isinstance(explicit, Mapping):
        return (
            str(explicit.get("classification", "UNCLASSIFIED")).upper(),
            str(explicit.get("reason", "explicit pipeline configuration")),
        )

    # Identifier fields are automatically critical because missing IDs can make
    # deterministic linkage, spell construction or event deduplication impossible.
    critical_ids = {
        table_cfg.get("patient_id"),
        table_cfg.get("event_id"),
        table_cfg.get("spell_id"),
        table_cfg.get("episode_id"),
    }
    critical_ids.discard(None)
    if column in critical_ids:
        return "CRITICAL", "configured identifier/key used for linkage or event grain"

    # Additional analytical fields (for example FirstMSKDate or AdmissionDate)
    # are configured explicitly because their criticality depends on study design.
    critical = set(_cfg(config).get("critical_columns", []) or [])
    critical.update(
        ((_cfg(config).get("critical_columns_by_table", {}) or {}).get(table_key, []) or [])
    )
    if column in critical:
        return "CRITICAL", "configured critical analytical/source field"

    return "UNCLASSIFIED", "no explicit missingness classification rule configured"


def column_missingness_table(
    df: pd.DataFrame,
    table_key: str,
    table_cfg: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Return one aggregate missingness row per cleaned source column."""
    n_rows = int(len(df))
    review_threshold = float(_cfg(config).get("review_threshold_pct", 20.0))
    rows: list[dict[str, Any]] = []

    for column in df.columns:
        missing_n = int(df[column].isna().sum())
        missing_pct = (100.0 * missing_n / n_rows) if n_rows else 0.0
        classification, reason = _classification(table_key, column, table_cfg, config)

        if classification == "CRITICAL" and missing_n:
            status = "FAIL"
        elif classification in {"STRUCTURAL_EXPECTED", "CONDITIONAL_EXPECTED", "EXPECTED"}:
            status = "EXPECTED"
        elif missing_pct >= review_threshold:
            status = "REVIEW"
        elif missing_n:
            status = "CHECK"
        else:
            status = "OK"

        rows.append(
            {
                "table": table_key,
                "column": column,
                "rows": n_rows,
                "nonmissing_n": n_rows - missing_n,
                "missing_n": missing_n,
                "missing_pct": round(missing_pct, 4),
                "classification": classification,
                "classification_reason": reason,
                "status": status,
                "dtype_after_cleaning": str(df[column].dtype),
            }
        )

    return pd.DataFrame(rows)


def print_missingness_summary(miss: pd.DataFrame, *, top_n: int = 10) -> None:
    """Print the most decision-relevant column-level missingness findings."""
    if miss.empty:
        print("  Missingness: no columns available")
        return

    rows = int(miss["rows"].iloc[0])
    n_columns = int(len(miss))
    total_cells = rows * n_columns
    missing_cells = int(miss["missing_n"].sum())
    missing_pct = 100.0 * missing_cells / total_cells if total_cells else 0.0
    affected = int(miss["missing_n"].gt(0).sum())
    critical = miss[(miss["classification"].eq("CRITICAL")) & miss["missing_n"].gt(0)]
    review = miss[miss["status"].isin(["FAIL", "REVIEW", "CHECK"]) & miss["missing_n"].gt(0)]

    print(
        f"  missingness = {missing_cells:,}/{total_cells:,} cells "
        f"({missing_pct:.2f}%); affected columns = {affected}/{n_columns}"
    )
    print(
        f"  critical fields with missing values = {len(critical)}; "
        f"fields requiring check/review = {len(review)}"
    )

    top = (
        miss[miss["missing_n"].gt(0)]
        .sort_values(["missing_pct", "missing_n"], ascending=False)
        .head(int(top_n))
    )
    if top.empty:
        print("  top missing fields: none")
    else:
        print(f"  top missing fields (max {top_n}):")
        for row in top.itertuples(index=False):
            print(
                f"    {row.column}: {row.missing_n:,}/{row.rows:,} "
                f"({row.missing_pct:.2f}%) [{row.classification}; {row.status}]"
            )

    if not critical.empty:
        print("  !!! CRITICAL MISSINGNESS - REVIEW BEFORE PREPROCESSING")
        for row in critical.itertuples(index=False):
            print(
                f"    {row.column}: {row.missing_n:,}/{row.rows:,} "
                f"({row.missing_pct:.2f}%)"
            )


def _source_pairs(table_keys: Iterable[str]) -> list[tuple[str, str, str]]:
    """Identify wider/sports pairs such as msk_wider versus msk_sports."""
    keys = set(table_keys)
    pairs: list[tuple[str, str, str]] = []
    for wider_key in sorted(k for k in keys if k.endswith("_wider")):
        family = wider_key[: -len("_wider")]
        sports_key = f"{family}_sports"
        if sports_key in keys:
            pairs.append((family, wider_key, sports_key))
    return pairs


def compare_group_missingness(
    all_missingness: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Compare missing percentages on columns common to matched source families."""
    if all_missingness.empty:
        return pd.DataFrame()

    alert_pp = float(_cfg(config).get("group_difference_alert_pp", 5.0))
    rows: list[dict[str, Any]] = []

    for family, wider_key, sports_key in _source_pairs(all_missingness["table"].unique()):
        wider = all_missingness[all_missingness["table"].eq(wider_key)]
        sports = all_missingness[all_missingness["table"].eq(sports_key)]
        merged = wider.merge(sports, on="column", suffixes=("_wider", "_sports"), how="inner")

        for row in merged.itertuples(index=False):
            wider_pct = float(row.missing_pct_wider)
            sports_pct = float(row.missing_pct_sports)
            diff = sports_pct - wider_pct
            rows.append(
                {
                    "source_family": family,
                    "column": row.column,
                    "wider_table": wider_key,
                    "sports_table": sports_key,
                    "wider_missing_n": int(row.missing_n_wider),
                    "wider_missing_pct": wider_pct,
                    "sports_missing_n": int(row.missing_n_sports),
                    "sports_missing_pct": sports_pct,
                    "sports_minus_wider_missing_pp": round(diff, 4),
                    "abs_difference_pp": round(abs(diff), 4),
                    "classification_wider": row.classification_wider,
                    "classification_sports": row.classification_sports,
                    "flag_group_difference": bool(abs(diff) >= alert_pp),
                }
            )

    return pd.DataFrame(rows)


def print_group_missingness_comparison(
    comparison: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    """Print the largest Sports-linked versus Wider MSK missingness differences."""
    print("\n" + "-" * 96)
    print("MISSINGNESS COMPARISON: SPORTS-LINKED vs WIDER MSK SOURCE FAMILIES")
    print("-" * 96)
    if comparison.empty:
        print("  No common source-family columns available for comparison.")
        return

    alert_pp = float(_cfg(config).get("group_difference_alert_pp", 5.0))
    top_n = int(_cfg(config).get("print_top_group_differences_n", 10))
    flagged_n = int(comparison["flag_group_difference"].sum())
    print(f"  alert threshold = |difference| >= {alert_pp:.1f} percentage points")
    print(f"  flagged common fields = {flagged_n}/{len(comparison)}")

    top = comparison.sort_values("abs_difference_pp", ascending=False).head(top_n)
    for row in top.itertuples(index=False):
        marker = " **FLAG**" if row.flag_group_difference else ""
        print(
            f"  {row.source_family}.{row.column}: "
            f"Wider={row.wider_missing_pct:.2f}% | "
            f"Sports={row.sports_missing_pct:.2f}% | "
            f"diff={row.sports_minus_wider_missing_pp:+.2f} pp{marker}"
        )
