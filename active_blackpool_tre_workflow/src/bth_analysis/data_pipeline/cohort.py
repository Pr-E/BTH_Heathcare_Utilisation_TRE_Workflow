"""Stage 05: freeze the comparative cohort, analytical index and observation windows.

The current fallback real-data design uses source-relative FirstMSKDate anchors
for both groups.  Baseline is the 365 days before index; follow-up begins at index
and is censored by the planned horizon, study end and death.  Eligibility flags
and exclusion counts are written explicitly so the cohort can be reconstructed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bth_analysis.workflow import load_workflow_config, output_path
from bth_analysis.audit import (
    dataframe_preview,
    metric,
    save_stage_summary,
    section,
    stage_footer,
    stage_header,
)


def _to_dt(series: pd.Series) -> pd.Series:
    """Parse canonical pipeline timestamps written as ISO-style strings.

    Raw source date parsing happens upstream during preprocessing. By the time
    cohort construction runs, dates have been serialised as ISO-style values,
    so locale/day-first reinterpretation must not be applied here.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(
        series,
        errors="coerce",
        dayfirst=False,
        format="mixed",
    )


def _bool_flag(value: bool, index: pd.Index) -> pd.Series:
    """Return a pipeline-style Int64 flag repeated for every cohort row."""
    return pd.Series(int(bool(value)), index=index, dtype="Int64")


def _assign_index_date(
    cohort: pd.DataFrame,
    cohort_cfg: dict,
) -> pd.DataFrame:
    """Assign the configured analytical index without claiming programme start.

    Current fallback strategy
    -------------------------
    ``source_relative_first_msk`` uses the same *semantic* pathway anchor in
    both analysis groups:

    * Sports-linked BTH pathway -> SportsAnchorFirstMSKDate
    * Wider MSK comparison      -> WiderAnchorFirstMSKDate

    This is deliberately an analytical time origin. It must not be interpreted
    as the date Active Blackpool treatment began.
    """
    out = cohort.copy()

    strategy = cohort_cfg.get(
        "index_strategy",
        "source_relative_first_msk",
    )
    index_field = cohort_cfg.get(
        "index_date_field",
        "AnchorFirstMSKDate",
    )

    if strategy != "source_relative_first_msk":
        raise ValueError(
            "Unsupported index_strategy. Current workflow supports "
            "'source_relative_first_msk'. Add and review any alternative "
            "strategy explicitly before use."
        )

    if index_field != "AnchorFirstMSKDate":
        raise ValueError(
            "For index_strategy='source_relative_first_msk', "
            "index_date_field must be 'AnchorFirstMSKDate'."
        )

    required = {
        "ExposureFlag",
        "SportsAnchorFirstMSKDate",
        "WiderAnchorFirstMSKDate",
    }
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(
            "Cannot construct analytical index; missing patient-spine columns: "
            + ", ".join(missing)
        )

    out["IndexDate"] = pd.NaT
    out["IndexDateSource"] = pd.Series(
        pd.NA,
        index=out.index,
        dtype="string",
    )

    exposed = out["ExposureFlag"].eq(1)
    comparison = out["ExposureFlag"].eq(0)

    out.loc[exposed, "IndexDate"] = out.loc[
        exposed,
        "SportsAnchorFirstMSKDate",
    ]
    out.loc[exposed, "IndexDateSource"] = (
        "Sports-linked MSK source: FirstMSKDate"
    )

    out.loc[comparison, "IndexDate"] = out.loc[
        comparison,
        "WiderAnchorFirstMSKDate",
    ]
    out.loc[comparison, "IndexDateSource"] = (
        "Wider MSK source: FirstMSKDate"
    )

    out["IndexDateType"] = cohort_cfg.get(
        "analytical_index_label",
        "Source-relative FirstMSKDate analytical index",
    )
    out["IndexStrategy"] = strategy
    out["IndexAnchorSemantic"] = "FirstMSKDate"

    return out


def run_cohort_index(
    config_path: str | Path = "config/workflow_tre.yaml",
) -> dict[str, pd.DataFrame]:
    """Create the comparative cohort, analytical index and observation windows.

    The current fallback design compares:

    * Sports-linked BTH pathway patients; and
    * Wider MSK non-Sports-linked candidate patients.

    ``ExposureFlag`` therefore represents BTH pathway-group membership, not
    confirmed programme treatment. The analytical index is a source-relative
    FirstMSKDate anchor and is explicitly not a programme-start date.
    """
    cfg = load_workflow_config(config_path)
    cohort_cfg = cfg["cohort"]

    analysis_dir = output_path(cfg, "analysis_dir")
    qa_dir = output_path(cfg, "qa_dir")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    stage_header(
        "05",
        "COHORT + ANALYTICAL INDEX + OBSERVATION WINDOWS",
        purpose=(
            "Freeze the working Sports-linked versus Wider MSK comparison population, assign the "
            "configured source-relative analytical index, construct the 365-day baseline and up-to-365-day "
            "follow-up windows, and write explicit eligibility/exclusion flags."
        ),
        inputs=[analysis_dir / "patient_spine.csv"],
        outputs=[analysis_dir / "analysis_index.csv", qa_dir / "05_cohort_flow.csv", qa_dir / "05_cohort_exclusions.csv"],
    )

    spine_path = analysis_dir / "patient_spine.csv"
    if not spine_path.exists():
        raise FileNotFoundError(
            f"{spine_path} not found. Run Stage 04 linkage first."
        )

    spine = pd.read_csv(spine_path, low_memory=False)

    required_spine_cols = {
        "PatientID",
        "SportsLinkedBTHFlag",
        "EligibleWiderNonSportsCandidateFlag",
        "SportsAnchorFirstMSKDate",
        "WiderAnchorFirstMSKDate",
    }
    missing_spine_cols = sorted(required_spine_cols - set(spine.columns))
    if missing_spine_cols:
        raise ValueError(
            "patient_spine.csv is missing required cohort fields: "
            + ", ".join(missing_spine_cols)
        )

    for col in [
        "DateOfBirth",
        "DateOfDeath",
        "WiderAnchorFirstMSKReferralDate",
        "WiderAnchorFirstMSKDate",
        "WiderAnchorLastMSKDate",
        "SportsAnchorFirstMSKReferralDate",
        "SportsAnchorFirstMSKDate",
        "SportsAnchorLastMSKDate",
    ]:
        if col in spine.columns:
            spine[col] = _to_dt(spine[col])

    baseline_days = int(cohort_cfg["baseline_days"])
    followup_days = int(cohort_cfg["followup_days"])

    # Working comparison population only. Other patient-spine members remain
    # outside the current comparative analysis.
    cohort = spine[
        spine["SportsLinkedBTHFlag"].eq(1)
        | spine["EligibleWiderNonSportsCandidateFlag"].eq(1)
    ].copy()

    cohort["ExposureFlag"] = (
        cohort["SportsLinkedBTHFlag"].fillna(0).astype("Int64")
    )
    cohort["AnalysisGroup"] = np.where(
        cohort["ExposureFlag"].eq(1),
        cohort_cfg["working_exposure_label"],
        cohort_cfg["working_comparison_label"],
    )

    # Explicit semantics/provenance fields. These travel with analysis_index.csv
    # so downstream stages cannot silently reinterpret ExposureFlag or IndexDate.
    cohort["ExposureDefinition"] = cohort_cfg.get(
        "analysis_group_definition",
        "sports_linked_bth_vs_wider_non_sports",
    )
    cohort["AnalysisGroupSemanticsConfirmedForWorkflowFlag"] = _bool_flag(
        cohort_cfg.get(
            "analysis_group_semantics_confirmed_for_workflow",
            False,
        ),
        cohort.index,
    )
    cohort["ProgrammeExposureSemanticsConfirmedFlag"] = _bool_flag(
        cohort_cfg.get("programme_exposure_semantics_confirmed", False),
        cohort.index,
    )
    cohort["ProgrammeStartDateAvailableFlag"] = _bool_flag(
        cohort_cfg.get("programme_start_date_available", False),
        cohort.index,
    )
    cohort["IndexIsProgrammeStartFlag"] = _bool_flag(
        cohort_cfg.get("index_is_programme_start", False),
        cohort.index,
    )
    cohort["AnalyticalIndexConfirmedForWorkflowFlag"] = _bool_flag(
        cohort_cfg.get(
            "analytical_index_semantics_confirmed_for_workflow",
            False,
        ),
        cohort.index,
    )
    cohort["FinalRealDataIndexSemanticsConfirmedFlag"] = _bool_flag(
        cohort_cfg.get(
            "final_real_data_index_semantics_confirmed",
            False,
        ),
        cohort.index,
    )

    cohort = _assign_index_date(cohort, cohort_cfg)

    # Age is recalculated at the analytical index whenever DOB is available.
    if "DateOfBirth" in cohort.columns:
        age_at_index = (
            (cohort["IndexDate"] - cohort["DateOfBirth"]).dt.days / 365.25
        )
        cohort["AgeAtIndex"] = age_at_index
        fallback_age = pd.to_numeric(
            cohort.get(
                "Age",
                pd.Series(np.nan, index=cohort.index),
            ),
            errors="coerce",
        )
        cohort["Age"] = fallback_age.fillna(age_at_index)
    else:
        cohort["AgeAtIndex"] = pd.to_numeric(
            cohort.get(
                "Age",
                pd.Series(np.nan, index=cohort.index),
            ),
            errors="coerce",
        )

    # Missing index dates are handled separately by IndexAvailableFlag, so they
    # are not labelled as death-before-index exclusions.
    if "DateOfDeath" not in cohort.columns:
        cohort["DateOfDeath"] = pd.NaT

    cohort["AliveAtIndexFlag"] = (
        cohort["IndexDate"].isna()
        | cohort["DateOfDeath"].isna()
        | cohort["DateOfDeath"].ge(cohort["IndexDate"])
    ).astype("Int64")

    # The configured study window is the authoritative healthcare observation
    # boundary. Event minima/maxima must not redefine patient observability.
    # The production configuration deliberately leaves these values unset until
    # approved real-TRE extract coverage is confirmed.
    start_raw = cfg.get("project", {}).get("study_start_date")
    end_raw = cfg.get("project", {}).get("study_end_date")
    if start_raw in (None, "") or end_raw in (None, ""):
        raise ValueError(
            "Real TRE study_start_date/study_end_date are not configured. "
            "Populate them from the approved BTH extract coverage specification; "
            "use the approved BTH/TRE extract coverage dates."
        )
    observation_start = pd.Timestamp(start_raw)
    observation_end_raw = pd.Timestamp(end_raw)
    observation_end = (
        observation_end_raw
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
        if observation_end_raw == observation_end_raw.normalize()
        else observation_end_raw
    )

    cohort["ObservationStartDate"] = observation_start
    cohort["ObservationEndDate"] = observation_end

    # Baseline is [IndexDate - baseline_days, IndexDate).
    cohort["BaselineStartDate"] = (
        cohort["IndexDate"] - pd.to_timedelta(baseline_days, unit="D")
    )
    cohort["BaselineEndDate"] = cohort["IndexDate"]

    # Follow-up is [IndexDate, observed follow-up end].
    cohort["FollowUpStartDate"] = cohort["IndexDate"]
    cohort["PlannedFollowUpEndDate"] = (
        cohort["IndexDate"] + pd.to_timedelta(followup_days, unit="D")
    )

    censor_date = pd.concat(
        [
            cohort["PlannedFollowUpEndDate"],
            pd.Series(observation_end, index=cohort.index),
            cohort["DateOfDeath"],
        ],
        axis=1,
    ).min(axis=1, skipna=True)
    cohort["FollowUpEndDate"] = censor_date

    available_baseline_start = pd.concat(
        [
            cohort["BaselineStartDate"],
            pd.Series(observation_start, index=cohort.index),
        ],
        axis=1,
    ).max(axis=1)

    cohort["BaselineDaysAvailable"] = (
        (
            cohort["IndexDate"] - available_baseline_start
        ).dt.total_seconds()
        / 86400.0
    ).clip(lower=0, upper=baseline_days)

    cohort["BaselineCompleteFlag"] = (
        cohort["IndexDate"].notna()
        & cohort["BaselineStartDate"].ge(observation_start)
    ).astype("Int64")

    cohort["FollowUpDaysAvailable"] = (
        (
            cohort["FollowUpEndDate"] - cohort["FollowUpStartDate"]
        ).dt.total_seconds()
        / 86400.0
    ).clip(lower=0, upper=followup_days)

    cohort["FollowUpPersonYears"] = (
        cohort["FollowUpDaysAvailable"] / 365.25
    )

    cohort["IndexAvailableFlag"] = (
        cohort["IndexDate"].notna().astype("Int64")
    )
    cohort["IndexWithinStudyWindowFlag"] = (
        cohort["IndexDate"].between(
            observation_start,
            observation_end,
            inclusive="both",
        )
    ).astype("Int64")
    cohort["BaselineWindowAvailableFlag"] = (
        cohort["BaselineStartDate"].notna().astype("Int64")
    )
    cohort["FullFollowUpFlag"] = (
        cohort["FollowUpDaysAvailable"].ge(followup_days).astype("Int64")
    )

    eligible = (
        cohort["IndexAvailableFlag"].eq(1)
        & cohort["IndexWithinStudyWindowFlag"].eq(1)
    )

    if cohort_cfg.get("require_alive_at_index", True):
        eligible &= cohort["AliveAtIndexFlag"].eq(1)

    minimum_age = cohort_cfg.get("minimum_age")
    if minimum_age is not None:
        eligible &= cohort["AgeAtIndex"].ge(float(minimum_age))

    if cohort_cfg.get("require_full_baseline", False):
        eligible &= cohort["BaselineCompleteFlag"].eq(1)

    if cohort_cfg.get("require_full_followup", False):
        eligible &= cohort["FullFollowUpFlag"].eq(1)

    cohort["AnalysisEligibleFlag"] = eligible.astype("Int64")

    # Cohort exclusion QA. Counts are not mutually exclusive; they identify
    # each criterion separately for diagnostic review.
    checks: dict[str, pd.Series] = {
        "missing_index": cohort["IndexAvailableFlag"].eq(0),
        "index_outside_study_window": (
            cohort["IndexWithinStudyWindowFlag"].eq(0)
            & cohort["IndexAvailableFlag"].eq(1)
        ),
        "death_before_index": (
            cohort["AliveAtIndexFlag"].eq(0)
            & cohort["IndexAvailableFlag"].eq(1)
        ),
    }

    if minimum_age is not None:
        checks["below_minimum_age"] = (
            cohort["AgeAtIndex"].lt(float(minimum_age))
            & cohort["AgeAtIndex"].notna()
        )
        checks["missing_age_at_index"] = cohort["AgeAtIndex"].isna()

    checks["incomplete_baseline"] = cohort["BaselineCompleteFlag"].eq(0)
    checks["incomplete_followup"] = cohort["FullFollowUpFlag"].eq(0)

    exclusions = pd.DataFrame(
        [
            {"reason": reason, "n": int(mask.sum())}
            for reason, mask in checks.items()
        ]
    )

    flow = pd.DataFrame(
        [
            {
                "stage": "Working comparative candidate population",
                "n": len(cohort),
            },
            {
                "stage": "Sports-linked BTH pathway",
                "n": int(cohort["ExposureFlag"].eq(1).sum()),
            },
            {
                "stage": "Wider MSK non-Sports-linked candidate",
                "n": int(cohort["ExposureFlag"].eq(0).sum()),
            },
            {
                "stage": "Index available",
                "n": int(cohort["IndexAvailableFlag"].sum()),
            },
            {
                "stage": "Full baseline",
                "n": int(cohort["BaselineCompleteFlag"].sum()),
            },
            {
                "stage": "Analysis eligible",
                "n": int(cohort["AnalysisEligibleFlag"].sum()),
            },
            {
                "stage": "Full planned follow-up",
                "n": int(cohort["FullFollowUpFlag"].sum()),
            },
        ]
    )

    # Index strategy QA is intentionally aggregate. It documents which clock
    # each group uses and whether the index is available before downstream
    # outcomes/propensity analysis is allowed to proceed.
    index_qa = (
        cohort.groupby(
            ["ExposureFlag", "AnalysisGroup", "IndexDateSource"],
            dropna=False,
            as_index=False,
        )
        .agg(
            patients=("PatientID", "nunique"),
            index_available_n=("IndexAvailableFlag", "sum"),
            analysis_eligible_n=("AnalysisEligibleFlag", "sum"),
            full_baseline_n=("BaselineCompleteFlag", "sum"),
            full_followup_n=("FullFollowUpFlag", "sum"),
        )
    )
    index_qa["index_strategy"] = cohort_cfg.get(
        "index_strategy",
        "source_relative_first_msk",
    )
    index_qa["index_is_programme_start"] = bool(
        cohort_cfg.get("index_is_programme_start", False)
    )
    index_qa["programme_start_date_available"] = bool(
        cohort_cfg.get("programme_start_date_available", False)
    )

    cohort.to_csv(
        analysis_dir / "analysis_index.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    exclusions.to_csv(
        qa_dir / "05_cohort_exclusions.csv",
        index=False,
    )
    flow.to_csv(
        qa_dir / "05_cohort_flow.csv",
        index=False,
    )
    index_qa.to_csv(
        qa_dir / "05_index_strategy_qa.csv",
        index=False,
    )

    eligible_mask = cohort["AnalysisEligibleFlag"].eq(1)
    sports_eligible = int((eligible_mask & cohort["ExposureFlag"].eq(1)).sum())
    wider_eligible = int((eligible_mask & cohort["ExposureFlag"].eq(0)).sum())
    full_followup_eligible = int((eligible_mask & cohort["FullFollowUpFlag"].eq(1)).sum())
    followup_days_eligible = pd.to_numeric(
        cohort.loc[eligible_mask, "FollowUpDaysAvailable"], errors="coerce"
    ).dropna()

    section("STAGE 05 KEY FINDINGS")
    metric("working comparative population", f"{len(cohort):,}")
    metric("Sports-linked pathway candidates", f"{int(cohort['ExposureFlag'].eq(1).sum()):,}")
    metric("Wider MSK candidates", f"{int(cohort['ExposureFlag'].eq(0).sum()):,}")
    metric("analysis-eligible patients", f"{int(eligible_mask.sum()):,}")
    metric("eligible Sports-linked", f"{sports_eligible:,}")
    metric("eligible Wider MSK", f"{wider_eligible:,}")
    metric("full baseline available", f"{int(cohort['BaselineCompleteFlag'].sum()):,}")
    metric("full planned follow-up among eligible", f"{full_followup_eligible:,}")
    if not followup_days_eligible.empty:
        metric("eligible follow-up days: median", f"{followup_days_eligible.median():.1f}")
        metric("eligible follow-up days: minimum", f"{followup_days_eligible.min():.1f}")
        metric("eligible follow-up days: maximum", f"{followup_days_eligible.max():.1f}")
    metric("healthcare observation start", observation_start)
    metric("healthcare observation end", observation_end)
    metric("index strategy", cohort_cfg.get("index_strategy", "source_relative_first_msk"))
    metric("index is confirmed programme start", bool(cohort_cfg.get("index_is_programme_start", False)))

    print("\nCohort exclusion diagnostics (criteria are not mutually exclusive):")
    dataframe_preview(exclusions.sort_values("n", ascending=False), max_rows=20)
    print("\nIndex strategy / group availability QA:")
    dataframe_preview(index_qa, max_rows=10)

    audit_dir = output_path(cfg, "audit_dir")
    summary_path = save_stage_summary(
        audit_dir,
        stage_key="cohort",
        stage_code="05",
        title="Cohort + analytical index + observation windows",
        status="PASS",
        key_findings={
            "working_comparative_n": len(cohort),
            "analysis_eligible_n": int(eligible_mask.sum()),
            "sports_eligible_n": sports_eligible,
            "wider_eligible_n": wider_eligible,
            "full_baseline_n": int(cohort["BaselineCompleteFlag"].sum()),
            "full_followup_among_eligible_n": full_followup_eligible,
            "followup_days_median_eligible": float(followup_days_eligible.median()) if not followup_days_eligible.empty else None,
            "observation_start": observation_start,
            "observation_end": observation_end,
            "index_strategy": cohort_cfg.get("index_strategy", "source_relative_first_msk"),
            "index_is_programme_start": bool(cohort_cfg.get("index_is_programme_start", False)),
        },
        qa_files=[
            qa_dir / "05_cohort_flow.csv",
            qa_dir / "05_cohort_exclusions.csv",
            qa_dir / "05_index_strategy_qa.csv",
            analysis_dir / "analysis_index.csv",
        ],
        warnings=[
            "Exclusion counts overlap and should not be summed as mutually exclusive losses.",
            "ExposureFlag denotes Sports-linked pathway membership, not confirmed Active Blackpool treatment.",
            "The source-relative FirstMSKDate anchor must not be called programme start unless the source owner confirms that semantic."
        ],
        config_path=config_path,
    )
    stage_footer(
        stage_key="cohort",
        audit_dir=audit_dir,
        summary_path=summary_path,
        qa_files=[qa_dir / "05_cohort_flow.csv", qa_dir / "05_cohort_exclusions.csv", qa_dir / "05_index_strategy_qa.csv"],
    )

    return {
        "analysis_index": cohort,
        "cohort_exclusions": exclusions,
        "cohort_flow": flow,
        "index_strategy_qa": index_qa,
    }
