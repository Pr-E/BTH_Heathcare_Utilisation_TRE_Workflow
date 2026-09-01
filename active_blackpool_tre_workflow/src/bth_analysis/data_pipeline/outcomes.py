"""Stage 06: construct patient-level healthcare-utilisation outcomes.

ED attendances are counted at attendance level; inpatient admissions are counted
at spell/admission level rather than raw episode level.  Events are assigned to
non-overlapping baseline and follow-up windows, person-time is calculated, and
rates are derived without discarding patients solely for partial follow-up.
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
    """Parse canonical workflow datetimes without locale reinterpretation.

    Processed and analysis-layer CSVs are written as ISO ``YYYY-MM-DD HH:MM:SS``.
    Re-reading these values with ``dayfirst=True`` can silently swap month/day
    for ambiguous dates (for example 2025-05-02 -> 2025-02-05).
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(
        series,
        errors="coerce",
        format="%Y-%m-%d %H:%M:%S",
    )


def _load(path: Path) -> pd.DataFrame:
    """Read a required upstream CSV and fail with the exact missing path."""
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _build_ed_ledger(
    df: pd.DataFrame,
    exposure_flag: int,
    source_name: str,
) -> pd.DataFrame:
    """Map cleaned ED attendances into the common healthcare-event ledger schema."""
    out = pd.DataFrame({
        "PatientID": df["PatientID"].astype("string"),
        "EventID": df["EDAttendanceID"].astype("string"),
        "EventType": "ED",
        "EventDate": _to_dt(df["ArrivalDateTime"]),
        "EmergencyInpatientFlag": 0,
        "SourceDataset": source_name,
        "ExposureFlag": exposure_flag,
    })
    return out


def _build_inpatient_ledger(
    df: pd.DataFrame,
    exposure_flag: int,
    source_name: str,
    emergency_patterns: list[str],
) -> pd.DataFrame:
    """Map one row per inpatient spell into the common event-ledger schema.

    EmergencyInpatientFlag is derived from reviewed MethodOfAdmission text patterns;
    this coding rule must be rechecked against the real BTH code set before final
    clinical interpretation.
    """
    method = df.get(
        "MethodOfAdmission",
        pd.Series(pd.NA, index=df.index, dtype="string"),
    ).astype("string")

    emergency = pd.Series(False, index=df.index)
    lower = method.str.lower()
    for pattern in emergency_patterns:
        emergency |= lower.str.contains(str(pattern).lower(), na=False)

    out = pd.DataFrame({
        "PatientID": df["PatientID"].astype("string"),
        "EventID": df["SpellID"].astype("string"),
        "EventType": "Inpatient",
        "EventDate": _to_dt(df["AdmissionDate"]),
        "EmergencyInpatientFlag": emergency.astype("Int64"),
        "SourceDataset": source_name,
        "ExposureFlag": exposure_flag,
    })
    return out


def run_outcome_features(
    config_path: str | Path = "config/workflow_tre.yaml",
) -> dict[str, pd.DataFrame]:
    """Build the healthcare event ledger and patient-level counts/person-time outcome features."""
    cfg = load_workflow_config(config_path)
    processed_dir = output_path(cfg, "processed_dir")
    analysis_dir = output_path(cfg, "analysis_dir")
    qa_dir = output_path(cfg, "qa_dir")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    stage_header(
        "06",
        "HEALTHCARE EVENT LEDGER + PATIENT OUTCOME FEATURES",
        purpose=(
            "Construct a deduplicated ED/inpatient event ledger, classify events into each patient's "
            "baseline/follow-up window, count inpatient admissions at spell level, derive emergency "
            "inpatient and total-hospital outcomes, and calculate person-time denominators/rates."
        ),
        inputs=[analysis_dir / "analysis_index.csv", processed_dir],
        outputs=[analysis_dir / "healthcare_event_ledger.csv", analysis_dir / "patient_outcomes.csv", qa_dir],
    )

    index_path = analysis_dir / "analysis_index.csv"
    cohort = _load(index_path)

    # Dates produced by the cohort stage are already canonical ISO values.
    # Parse them deterministically here; do not apply locale/day-first rules.
    cohort_date_cols = [
        "ObservationStartDate",
        "ObservationEndDate",
        "IndexDate",
        "BaselineStartDate",
        "BaselineEndDate",
        "FollowUpStartDate",
        "PlannedFollowUpEndDate",
        "FollowUpEndDate",
    ]
    for col in cohort_date_cols:
        if col in cohort.columns:
            cohort[col] = _to_dt(cohort[col])

    # Defensive QA: a record previously flagged as being inside the study
    # window must remain inside it after re-loading the cohort file.
    if {
        "IndexWithinStudyWindowFlag",
        "IndexDate",
        "ObservationStartDate",
        "ObservationEndDate",
    }.issubset(cohort.columns):
        flagged_inside = cohort["IndexWithinStudyWindowFlag"].eq(1)
        invalid_flagged_index = flagged_inside & (
            cohort["IndexDate"].lt(cohort["ObservationStartDate"])
            | cohort["IndexDate"].gt(cohort["ObservationEndDate"])
            | cohort["IndexDate"].isna()
        )
        if invalid_flagged_index.any():
            raise ValueError(
                "Cohort date integrity check failed: an index flagged inside "
                "the study window falls outside the configured observation "
                "period after loading analysis_index.csv."
            )

    sports_ids = set(
        cohort.loc[cohort["ExposureFlag"].eq(1), "PatientID"]
        .dropna()
        .astype(str)
    )
    comparison_ids = set(
        cohort.loc[cohort["ExposureFlag"].eq(0), "PatientID"]
        .dropna()
        .astype(str)
    )

    wider_ed = _load(processed_dir / "ed_wider_attendances.csv")
    sports_ed = _load(processed_dir / "ed_sports_attendances.csv")
    wider_ip = _load(processed_dir / "inpatient_wider_spells.csv")
    sports_ip = _load(processed_dir / "inpatient_sports_spells.csv")

    # Use cohort-specific BTH extracts to avoid double counting across the
    # wider and Sports-linked source families.
    wider_ed = wider_ed[
        wider_ed["PatientID"].astype(str).isin(comparison_ids)
    ].copy()
    sports_ed = sports_ed[
        sports_ed["PatientID"].astype(str).isin(sports_ids)
    ].copy()
    wider_ip = wider_ip[
        wider_ip["PatientID"].astype(str).isin(comparison_ids)
    ].copy()
    sports_ip = sports_ip[
        sports_ip["PatientID"].astype(str).isin(sports_ids)
    ].copy()

    patterns = cfg["outcomes"].get("emergency_method_patterns", [])

    ledgers = [
        _build_ed_ledger(wider_ed, 0, "ed_wider"),
        _build_ed_ledger(sports_ed, 1, "ed_sports"),
        _build_inpatient_ledger(
            wider_ip, 0, "inpatient_wider", patterns
        ),
        _build_inpatient_ledger(
            sports_ip, 1, "inpatient_sports", patterns
        ),
    ]

    ledger = pd.concat(ledgers, ignore_index=True)
    ledger = ledger.dropna(subset=["PatientID", "EventDate"]).copy()
    ledger = ledger.drop_duplicates(
        ["SourceDataset", "PatientID", "EventID", "EventType"]
    ).copy()

    windows = cohort[
        [
            "PatientID",
            "ExposureFlag",
            "IndexDate",
            "BaselineStartDate",
            "BaselineEndDate",
            "FollowUpStartDate",
            "FollowUpEndDate",
        ]
    ].copy()

    ledger = ledger.merge(
        windows,
        on=["PatientID", "ExposureFlag"],
        how="inner",
        validate="many_to_one",
    )

    ledger["RelativeDay"] = (
        ledger["EventDate"] - ledger["IndexDate"]
    ).dt.total_seconds() / 86400.0

    # Continuous-time baseline: include the 365 days before index,
    # but exclude the index instant itself so there is no 24-hour gap and
    # no overlap with follow-up.
    baseline = (
        ledger["EventDate"].ge(ledger["BaselineStartDate"])
        & ledger["EventDate"].lt(ledger["IndexDate"])
    )
    followup = (
        ledger["EventDate"].ge(ledger["FollowUpStartDate"])
        & ledger["EventDate"].le(ledger["FollowUpEndDate"])
    )

    ledger["AnalysisPeriod"] = np.select(
        [baseline, followup],
        ["Baseline", "Follow-up"],
        default="Outside analysis window",
    )

    ledger["EDFlag"] = ledger["EventType"].eq("ED").astype("Int64")
    ledger["InpatientFlag"] = (
        ledger["EventType"].eq("Inpatient").astype("Int64")
    )

    inside = ledger[
        ledger["AnalysisPeriod"].isin(["Baseline", "Follow-up"])
    ].copy()

    summary = (
        inside.groupby(
            ["PatientID", "AnalysisPeriod"],
            as_index=False,
        )
        .agg(
            EDCount=("EDFlag", "sum"),
            InpatientCount=("InpatientFlag", "sum"),
            EmergencyInpatientCount=("EmergencyInpatientFlag", "sum"),
        )
    )
    summary["TotalHospitalCount"] = (
        summary["EDCount"] + summary["InpatientCount"]
    )

    wide = summary.pivot(
        index="PatientID",
        columns="AnalysisPeriod",
        values=[
            "EDCount",
            "InpatientCount",
            "EmergencyInpatientCount",
            "TotalHospitalCount",
        ],
    )
    wide.columns = [
        f"{'Baseline' if period == 'Baseline' else 'FollowUp'}{metric}"
        for metric, period in wide.columns
    ]
    wide = wide.reset_index()

    patient = cohort.merge(
        wide,
        on="PatientID",
        how="left",
        validate="one_to_one",
    )

    count_cols = [
        "BaselineEDCount",
        "BaselineInpatientCount",
        "BaselineEmergencyInpatientCount",
        "BaselineTotalHospitalCount",
        "FollowUpEDCount",
        "FollowUpInpatientCount",
        "FollowUpEmergencyInpatientCount",
        "FollowUpTotalHospitalCount",
    ]
    for col in count_cols:
        if col not in patient:
            patient[col] = 0
        patient[col] = (
            pd.to_numeric(patient[col], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    patient["BaselinePersonYears"] = (
        pd.to_numeric(
            patient.get("BaselineDaysAvailable"),
            errors="coerce",
        ).fillna(0)
        / 365.25
    )
    patient["FollowUpPersonYears"] = pd.to_numeric(
        patient["FollowUpPersonYears"],
        errors="coerce",
    ).fillna(0)

    for prefix in ("Baseline", "FollowUp"):
        py_col = f"{prefix}PersonYears"
        for metric in (
            "ED",
            "Inpatient",
            "EmergencyInpatient",
            "TotalHospital",
        ):
            count_col = f"{prefix}{metric}Count"
            rate_col = f"{prefix}{metric}RatePerPY"
            denom = patient[py_col].replace(0, np.nan)
            patient[rate_col] = patient[count_col] / denom

    for metric in (
        "ED",
        "Inpatient",
        "EmergencyInpatient",
        "TotalHospital",
    ):
        patient[f"AnyFollowUp{metric}Flag"] = (
            patient[f"FollowUp{metric}Count"].gt(0).astype("Int64")
        )

    event_counts = pd.DataFrame([
        {
            "metric": "ledger_rows_all",
            "n": len(ledger),
        },
        {
            "metric": "ledger_rows_inside_windows",
            "n": len(inside),
        },
        {
            "metric": "ed_events_inside_windows",
            "n": int(inside["EDFlag"].sum()),
        },
        {
            "metric": "inpatient_spells_inside_windows",
            "n": int(inside["InpatientFlag"].sum()),
        },
        {
            "metric": "emergency_inpatient_spells_inside_windows",
            "n": int(inside["EmergencyInpatientFlag"].sum()),
        },
    ])

    # Aggregate rate QA provides an immediately auditable bridge between outcome
    # engineering and the descriptive stage.  These are crude rates only; no
    # propensity weighting or causal interpretation is applied here.
    rate_rows = []
    eligible_patient = patient[patient["AnalysisEligibleFlag"].eq(1)].copy()
    for exposure_flag, group_df in eligible_patient.groupby("ExposureFlag"):
        group_label = (
            str(group_df["AnalysisGroup"].iloc[0])
            if "AnalysisGroup" in group_df.columns and len(group_df)
            else str(exposure_flag)
        )
        for period in ("Baseline", "FollowUp"):
            person_years = float(pd.to_numeric(group_df[f"{period}PersonYears"], errors="coerce").sum())
            for metric_name in ("ED", "Inpatient", "EmergencyInpatient", "TotalHospital"):
                events = float(pd.to_numeric(group_df[f"{period}{metric_name}Count"], errors="coerce").sum())
                rate_rows.append({
                    "ExposureFlag": int(exposure_flag),
                    "AnalysisGroup": group_label,
                    "period": period,
                    "outcome": metric_name,
                    "patients": int(group_df["PatientID"].nunique()),
                    "events": events,
                    "person_years": person_years,
                    "rate_per_100_person_years": (events / person_years * 100.0) if person_years > 0 else np.nan,
                    "zero_event_patient_pct": float(group_df[f"{period}{metric_name}Count"].eq(0).mean() * 100.0),
                })
    rate_qa = pd.DataFrame(rate_rows)
    rate_qa.to_csv(qa_dir / "06_outcome_rate_qa.csv", index=False)

    ledger.to_csv(
        analysis_dir / "healthcare_event_ledger.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    patient.to_csv(
        analysis_dir / "patient_outcomes.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    event_counts.to_csv(
        qa_dir / "06_outcome_feature_qa.csv",
        index=False,
    )

    eligible_n = int(patient["AnalysisEligibleFlag"].fillna(0).sum())
    outside_window_n = int(ledger["AnalysisPeriod"].eq("Outside analysis window").sum())

    section("STAGE 06 KEY FINDINGS")
    metric("patient outcome rows", f"{len(patient):,}")
    metric("analysis-eligible patients", f"{eligible_n:,}")
    metric("event ledger rows (all linked events)", f"{len(ledger):,}")
    metric("events inside baseline/follow-up windows", f"{len(inside):,}")
    metric("events outside analysis windows", f"{outside_window_n:,}")
    print("\nEvent construction QA:")
    dataframe_preview(event_counts, max_rows=10)
    print("\nCrude outcome rates generated for downstream descriptive review:")
    dataframe_preview(
        rate_qa,
        columns=[
            "AnalysisGroup", "period", "outcome", "patients", "events",
            "person_years", "rate_per_100_person_years", "zero_event_patient_pct",
        ],
        max_rows=20,
    )

    audit_dir = output_path(cfg, "audit_dir")
    summary_path = save_stage_summary(
        audit_dir,
        stage_key="outcomes",
        stage_code="06",
        title="Healthcare event ledger + patient outcome features",
        status="PASS",
        key_findings={
            "patient_outcome_rows": len(patient),
            "analysis_eligible_n": eligible_n,
            "event_ledger_rows": len(ledger),
            "events_inside_windows_n": len(inside),
            "events_outside_windows_n": outside_window_n,
            "ed_events_inside_windows": int(inside["EDFlag"].sum()),
            "inpatient_spells_inside_windows": int(inside["InpatientFlag"].sum()),
            "emergency_inpatient_spells_inside_windows": int(inside["EmergencyInpatientFlag"].sum()),
        },
        qa_files=[qa_dir / "06_outcome_feature_qa.csv", qa_dir / "06_outcome_rate_qa.csv"],
        warnings=[
            "The event ledger and patient_outcomes.csv are TRE-internal patient-level analytical files.",
            "Zero outcome counts are derived only after verified linkage/window assignment; Stage 02 source missingness is not converted to zero.",
            "Emergency admission patterns must be reviewed against the real MethodOfAdmission coding before final interpretation."
        ],
        config_path=config_path,
    )
    stage_footer(
        stage_key="outcomes",
        audit_dir=audit_dir,
        summary_path=summary_path,
        qa_files=[qa_dir / "06_outcome_feature_qa.csv", qa_dir / "06_outcome_rate_qa.csv"],
    )

    return {
        "healthcare_event_ledger": ledger,
        "patient_outcomes": patient,
        "outcome_qa": event_counts,
    }
