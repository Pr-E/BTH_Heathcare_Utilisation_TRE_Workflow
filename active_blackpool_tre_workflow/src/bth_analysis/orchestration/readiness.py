"""Configuration-level readiness checks for TRE translation and interpretation.

This module does not read patient data.  It separates two questions that should
not be conflated during review:
1) Is the fallback Sports-linked-vs-Wider-MSK comparative workflow configured?
2) Is the evidence sufficient for programme-specific Active Blackpool claims?

The first can be ready while the second remains deliberately not ready.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from bth_analysis.workflow import load_workflow_config


def translation_readiness(
    workflow_config: str | Path = "config/workflow_tre.yaml",
) -> pd.DataFrame:
    """Return one aggregate row per interpretation/configuration readiness check."""
    cfg = load_workflow_config(workflow_config)
    cohort = cfg["cohort"]

    # Each check states its scope and why the reviewer should care.  Keeping
    # programme-specific checks separate prevents pathway membership from being
    # silently promoted to confirmed programme treatment.
    checks = [
        {
            "check": "analysis_group_semantics_confirmed_for_workflow",
            "ready": bool(
                cohort.get("analysis_group_semantics_confirmed_for_workflow", False)
            ),
            "scope": "fallback comparative workflow",
            "required_for": "Sports-linked BTH pathway vs Wider MSK adjusted comparison",
        },
        {
            "check": "analytical_index_semantics_confirmed_for_workflow",
            "ready": bool(
                cohort.get("analytical_index_semantics_confirmed_for_workflow", False)
            ),
            "scope": "fallback comparative workflow",
            "required_for": "baseline/follow-up window construction",
        },
        {
            "check": "index_is_not_mislabelled_as_programme_start",
            "ready": not bool(cohort.get("index_is_programme_start", False)),
            "scope": "fallback comparative workflow",
            "required_for": "non-causal interpretation when programme start is unavailable",
        },
        {
            "check": "programme_exposure_semantics_confirmed",
            "ready": bool(cohort.get("programme_exposure_semantics_confirmed", False)),
            "scope": "programme-specific extension",
            "required_for": "confirmed Active Blackpool treatment/exposure interpretation",
        },
        {
            "check": "programme_start_date_available",
            "ready": bool(cohort.get("programme_start_date_available", False)),
            "scope": "programme-specific extension",
            "required_for": "programme-start indexed analyses and engagement timing",
        },
        {
            "check": "final_real_data_index_semantics_confirmed",
            "ready": bool(cohort.get("final_real_data_index_semantics_confirmed", False)),
            "scope": "TRE translation/final freeze",
            "required_for": "final real-data protocol/index freeze",
        },
        {
            "check": "full_baseline_rule_explicit",
            "ready": "require_full_baseline" in cohort,
            "scope": "fallback comparative workflow",
            "required_for": "baseline comparability and eligibility QA",
        },
        {
            "check": "full_followup_rule_explicit",
            "ready": "require_full_followup" in cohort,
            "scope": "fallback comparative workflow",
            "required_for": "follow-up/censoring strategy",
        },
        {
            "check": "comparative_outcomes_configured",
            "ready": bool(cfg["comparative"].get("outcomes")),
            "scope": "fallback comparative workflow",
            "required_for": "main comparative models",
        },
    ]

    return pd.DataFrame(checks)
