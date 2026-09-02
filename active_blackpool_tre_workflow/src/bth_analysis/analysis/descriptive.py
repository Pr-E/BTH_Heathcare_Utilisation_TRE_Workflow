"""Stage 07: descriptive EDA, cohort characterisation and diagnostic figures.

This layer describes the analytical population, missingness, source coverage,
pathway timing, baseline utilisation and crude baseline/follow-up rates. It is
separate from propensity adjustment and outcome modelling so descriptive facts
are not confused with adjusted associations.
"""
from __future__ import annotations

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


GROUP_COL = "AnalysisGroup"
EXPOSURE_COL = "ExposureFlag"

NUMERIC_BASELINE_VARS = [
    "AgeAtIndex",
    "Index_of_Multiple_Deprivation_IMD_Decile",
    "BaselineEDCount",
    "BaselineInpatientCount",
    "BaselineEmergencyInpatientCount",
    "BaselineTotalHospitalCount",
]

CATEGORICAL_BASELINE_VARS = [
    "Sex",
    "EthnicityNationalCodeDesc",
    "PostcodeLAName",
    "AgeBand",
    "IMDQuintile",
]

MISSINGNESS_VARS = [
    "AgeAtIndex",
    "Sex",
    "EthnicityNationalCodeDesc",
    "Index_of_Multiple_Deprivation_IMD_Decile",
    "PostcodeLAName",
    "IndexDate",
]

OUTCOME_METRICS = (
    "ED",
    "Inpatient",
    "EmergencyInpatient",
    "TotalHospital",
)

EFFECTIVE_MISSING_ETHNICITY = {
    "not stated",
    "not recorded",
    "unknown",
    "missing",
    "",
}

DISPLAY_LABELS = {
    "AgeAtIndex": "Age at index (years)",
    "Index_of_Multiple_Deprivation_IMD_Decile": "IMD decile",
    "BaselineEDCount": "Baseline ED attendances",
    "BaselineInpatientCount": "Baseline inpatient admissions",
    "BaselineEmergencyInpatientCount": "Baseline emergency admissions",
    "BaselineTotalHospitalCount": "Baseline total hospital events",
    "Sex": "Sex",
    "EthnicityNationalCodeDesc": "Ethnicity",
    "PostcodeLAName": "Local authority",
    "AgeBand": "Age band",
    "IMDQuintile": "IMD quintile",
    "ED": "ED attendances",
    "Inpatient": "Inpatient admissions",
    "EmergencyInpatient": "Emergency inpatient admissions",
    "TotalHospital": "Total hospital utilisation",
}


def _to_dt(series: pd.Series) -> pd.Series:
    """Coerce a series to pandas datetime while turning unparseable values into missing timestamps."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def _slug(value: object) -> str:
    """Create a filesystem-safe text label for aggregate figure filenames."""
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return text.lower() or "group"


def _numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric values for robust descriptive calculations."""
    return pd.to_numeric(series, errors="coerce")


def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate a percentage while returning missing when the denominator is zero."""
    return float(numerator / denominator * 100) if denominator else np.nan


def _derive_reporting_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive presentation/EDA-only age bands and IMD quintiles without changing core analysis variables."""
    out = df.copy()

    if "AgeAtIndex" in out:
        age = _numeric(out["AgeAtIndex"])
        out["AgeBand"] = pd.cut(
            age,
            bins=[16, 35, 50, 65, 75, np.inf],
            right=False,
            labels=["16–34", "35–49", "50–64", "65–74", "75+"],
        )

    imd_col = "Index_of_Multiple_Deprivation_IMD_Decile"
    if imd_col in out:
        imd = _numeric(out[imd_col])
        out["IMDQuintile"] = pd.cut(
            imd,
            bins=[0, 2, 4, 6, 8, 10],
            include_lowest=True,
            labels=[
                "Q1 (most deprived)",
                "Q2",
                "Q3",
                "Q4",
                "Q5 (least deprived)",
            ],
        )

    return out


def _group_lookup(df: pd.DataFrame) -> dict[int, str]:
    """Create an ExposureFlag-to-display-label mapping from observed analysis groups."""
    lookup: dict[int, str] = {}
    if EXPOSURE_COL not in df or GROUP_COL not in df:
        return lookup
    x = df[[EXPOSURE_COL, GROUP_COL]].dropna().drop_duplicates()
    for _, row in x.iterrows():
        try:
            lookup[int(row[EXPOSURE_COL])] = str(row[GROUP_COL])
        except (TypeError, ValueError):
            continue
    return lookup


def _summarise_numeric(
    df: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """Calculate grouped descriptive statistics for configured numeric baseline variables."""
    rows: list[dict[str, object]] = []
    for (flag, group), sub in df.groupby([EXPOSURE_COL, GROUP_COL], dropna=False):
        for variable in variables:
            if variable not in sub:
                continue
            values = _numeric(sub[variable]).dropna()
            if values.empty:
                continue
            mean = values.mean()
            variance = values.var(ddof=1) if len(values) > 1 else 0.0
            rows.append(
                {
                    "ExposureFlag": flag,
                    "group": group,
                    "variable": variable,
                    "label": DISPLAY_LABELS.get(variable, variable),
                    "n": int(values.size),
                    "missing_n": int(sub[variable].isna().sum()),
                    "mean": mean,
                    "sd": values.std(ddof=1) if len(values) > 1 else 0.0,
                    "variance": variance,
                    "variance_to_mean_ratio": (
                        variance / mean if mean > 0 else np.nan
                    ),
                    "skewness": values.skew() if len(values) > 2 else np.nan,
                    "min": values.min(),
                    "q1": values.quantile(0.25),
                    "median": values.median(),
                    "q3": values.quantile(0.75),
                    "p95": values.quantile(0.95),
                    "p99": values.quantile(0.99),
                    "max": values.max(),
                    "zero_n": int(values.eq(0).sum()),
                    "zero_pct": _safe_pct(values.eq(0).sum(), len(values)),
                }
            )
    return pd.DataFrame(rows)


def _summarise_categories(
    df: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """Calculate grouped counts/percentages for configured categorical baseline variables."""
    rows: list[pd.DataFrame] = []
    for variable in variables:
        if variable not in df:
            continue
        temp = df[[EXPOSURE_COL, GROUP_COL, variable]].copy()
        temp[variable] = temp[variable].astype("string").fillna("<Missing>")
        counts = (
            temp.groupby([EXPOSURE_COL, GROUP_COL, variable], dropna=False)
            .size()
            .rename("n")
            .reset_index()
        )
        totals = counts.groupby([EXPOSURE_COL, GROUP_COL])["n"].transform("sum")
        counts["pct"] = counts["n"] / totals * 100
        counts["variable"] = variable
        counts["label"] = DISPLAY_LABELS.get(variable, variable)
        counts = counts.rename(columns={variable: "level", GROUP_COL: "group"})
        rows.append(
            counts[
                [
                    "ExposureFlag",
                    "group",
                    "variable",
                    "label",
                    "level",
                    "n",
                    "pct",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _numeric_smd(x0: pd.Series, x1: pd.Series) -> float:
    """Calculate an unweighted SMD for one continuous baseline variable."""
    a = _numeric(x0).dropna()
    b = _numeric(x1).dropna()
    if a.empty or b.empty:
        return np.nan
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0 if np.isclose(a.mean(), b.mean(), equal_nan=True) else np.nan
    return float((b.mean() - a.mean()) / pooled)


def _binary_smd(p0: float, p1: float) -> float:
    """Calculate an unweighted SMD for one binary indicator."""
    denom = np.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / 2)
    if denom == 0:
        return 0.0 if np.isclose(p0, p1) else np.nan
    return float((p1 - p0) / denom)


def _baseline_balance(
    df: pd.DataFrame,
    numeric_vars: list[str],
    categorical_vars: list[str],
) -> pd.DataFrame:
    """Calculate unadjusted baseline SMDs used to show why confounding adjustment is required."""
    g0 = df[df[EXPOSURE_COL].eq(0)]
    g1 = df[df[EXPOSURE_COL].eq(1)]
    group_names = _group_lookup(df)
    rows: list[dict[str, object]] = []

    for variable in numeric_vars:
        if variable not in df:
            continue
        smd = _numeric_smd(g0[variable], g1[variable])
        rows.append(
            {
                "variable": variable,
                "label": DISPLAY_LABELS.get(variable, variable),
                "level": "",
                "comparison_group": group_names.get(0, "ExposureFlag=0"),
                "sports_linked_group": group_names.get(1, "ExposureFlag=1"),
                "comparison_mean": _numeric(g0[variable]).mean(),
                "sports_linked_mean": _numeric(g1[variable]).mean(),
                "comparison_pct": np.nan,
                "sports_linked_pct": np.nan,
                "smd": smd,
                "abs_smd": abs(smd) if pd.notna(smd) else np.nan,
            }
        )

    for variable in categorical_vars:
        if variable not in df:
            continue
        levels = (
            df[variable].astype("string").fillna("<Missing>").drop_duplicates().tolist()
        )
        for level in levels:
            p0 = (g0[variable].astype("string").fillna("<Missing>") == level).mean()
            p1 = (g1[variable].astype("string").fillna("<Missing>") == level).mean()
            smd = _binary_smd(float(p0), float(p1))
            rows.append(
                {
                    "variable": variable,
                    "label": DISPLAY_LABELS.get(variable, variable),
                    "level": str(level),
                    "comparison_group": group_names.get(0, "ExposureFlag=0"),
                    "sports_linked_group": group_names.get(1, "ExposureFlag=1"),
                    "comparison_mean": np.nan,
                    "sports_linked_mean": np.nan,
                    "comparison_pct": p0 * 100,
                    "sports_linked_pct": p1 * 100,
                    "smd": smd,
                    "abs_smd": abs(smd) if pd.notna(smd) else np.nan,
                }
            )

    return pd.DataFrame(rows).sort_values("abs_smd", ascending=False, na_position="last")


def _cohort_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Build aggregate cohort-flow counts for descriptive reporting."""
    rows: list[dict[str, object]] = []
    metrics = [
        ("Working comparative population", None),
        ("Index available", "IndexAvailableFlag"),
        ("Index within study window", "IndexWithinStudyWindowFlag"),
        ("Alive at index", "AliveAtIndexFlag"),
        ("Full baseline", "BaselineCompleteFlag"),
        ("Full follow-up", "FullFollowUpFlag"),
        ("Analysis eligible", "AnalysisEligibleFlag"),
    ]
    for (flag, group), sub in df.groupby([EXPOSURE_COL, GROUP_COL], dropna=False):
        base_n = int(sub["PatientID"].nunique())
        for stage, column in metrics:
            if column is None:
                n = base_n
            elif column in sub:
                n = int(sub.loc[sub[column].eq(1), "PatientID"].nunique())
            else:
                continue
            rows.append(
                {
                    "ExposureFlag": flag,
                    "group": group,
                    "stage": stage,
                    "n": n,
                    "pct_of_working_group": _safe_pct(n, base_n),
                }
            )
    return pd.DataFrame(rows)


def _missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise effective analytical missingness by group for key descriptive variables."""
    rows: list[dict[str, object]] = []
    for (flag, group), sub in df.groupby([EXPOSURE_COL, GROUP_COL], dropna=False):
        for column in MISSINGNESS_VARS:
            if column not in sub:
                continue
            literal_missing = sub[column].isna()
            effective_missing = literal_missing.copy()
            if column == "EthnicityNationalCodeDesc":
                normalised = sub[column].astype("string").fillna("").str.strip().str.lower()
                effective_missing = literal_missing | normalised.isin(EFFECTIVE_MISSING_ETHNICITY)
            rows.append(
                {
                    "ExposureFlag": flag,
                    "group": group,
                    "variable": column,
                    "label": DISPLAY_LABELS.get(column, column),
                    "patients": len(sub),
                    "literal_missing_n": int(literal_missing.sum()),
                    "literal_missing_pct": _safe_pct(literal_missing.sum(), len(sub)),
                    "effective_missing_n": int(effective_missing.sum()),
                    "effective_missing_pct": _safe_pct(effective_missing.sum(), len(sub)),
                }
            )
    return pd.DataFrame(rows)


def _source_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise linked-source availability by analysis group."""
    source_cols = [
        "PresentWiderMSK",
        "PresentSportsLinkedMSK",
        "PresentWiderInpatient",
        "PresentSportsInpatient",
        "PresentWiderED",
        "PresentSportsED",
    ]
    rows: list[dict[str, object]] = []
    for (flag, group), sub in df.groupby([EXPOSURE_COL, GROUP_COL], dropna=False):
        for column in source_cols:
            if column not in sub:
                continue
            values = _numeric(sub[column]).fillna(0).gt(0)
            rows.append(
                {
                    "ExposureFlag": flag,
                    "group": group,
                    "source_presence_flag": column,
                    "patients": len(sub),
                    "present_n": int(values.sum()),
                    "present_pct": _safe_pct(values.sum(), len(sub)),
                }
            )
    return pd.DataFrame(rows)


def _followup_observation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise observed follow-up in a compact table rather than a density plot."""
    rows: list[dict[str, object]] = []
    for (flag, group), sub in df.groupby([EXPOSURE_COL, GROUP_COL], dropna=False):
        days = _numeric(sub.get("FollowUpDaysAvailable", pd.Series(np.nan, index=sub.index)))
        full = _numeric(sub.get("FullFollowUpFlag", pd.Series(0, index=sub.index))).fillna(0).eq(1)
        partial = days.notna() & ~full
        zero_days = days.fillna(np.nan).eq(0)
        rows.append(
            {
                "ExposureFlag": flag,
                "group": group,
                "eligible_patients": int(sub["PatientID"].nunique()),
                "followup_days_available_n": int(days.notna().sum()),
                "median_followup_days": float(days.median()) if days.notna().any() else np.nan,
                "q1_followup_days": float(days.quantile(0.25)) if days.notna().any() else np.nan,
                "q3_followup_days": float(days.quantile(0.75)) if days.notna().any() else np.nan,
                "min_followup_days": float(days.min()) if days.notna().any() else np.nan,
                "max_followup_days": float(days.max()) if days.notna().any() else np.nan,
                "full_365_day_followup_n": int(full.sum()),
                "full_365_day_followup_pct": _safe_pct(full.sum(), len(sub)),
                "partial_followup_n": int(partial.sum()),
                "partial_followup_pct": _safe_pct(partial.sum(), len(sub)),
                "zero_followup_days_n": int(zero_days.sum()),
                "zero_followup_days_pct": _safe_pct(zero_days.sum(), len(sub)),
            }
        )
    return pd.DataFrame(rows)


def _pathway_timing(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise referral/MSK timing intervals and chronology-related descriptive measures."""
    temp = df.copy()
    date_cols = [
        "WiderAnchorFirstMSKReferralDate",
        "WiderAnchorFirstMSKDate",
        "WiderAnchorLastMSKDate",
        "SportsAnchorFirstMSKReferralDate",
        "SportsAnchorFirstMSKDate",
        "SportsAnchorLastMSKDate",
    ]
    for col in date_cols:
        if col in temp:
            temp[col] = _to_dt(temp[col])

    referral = pd.Series(pd.NaT, index=temp.index, dtype="datetime64[ns]")
    first = pd.Series(pd.NaT, index=temp.index, dtype="datetime64[ns]")
    last = pd.Series(pd.NaT, index=temp.index, dtype="datetime64[ns]")

    comparison = temp[EXPOSURE_COL].eq(0)
    sports = temp[EXPOSURE_COL].eq(1)

    if "WiderAnchorFirstMSKReferralDate" in temp:
        referral.loc[comparison] = temp.loc[comparison, "WiderAnchorFirstMSKReferralDate"]
    if "WiderAnchorFirstMSKDate" in temp:
        first.loc[comparison] = temp.loc[comparison, "WiderAnchorFirstMSKDate"]
    if "WiderAnchorLastMSKDate" in temp:
        last.loc[comparison] = temp.loc[comparison, "WiderAnchorLastMSKDate"]

    if "SportsAnchorFirstMSKReferralDate" in temp:
        referral.loc[sports] = temp.loc[sports, "SportsAnchorFirstMSKReferralDate"]
    if "SportsAnchorFirstMSKDate" in temp:
        first.loc[sports] = temp.loc[sports, "SportsAnchorFirstMSKDate"]
    if "SportsAnchorLastMSKDate" in temp:
        last.loc[sports] = temp.loc[sports, "SportsAnchorLastMSKDate"]

    temp["ReferralToFirstMSKDays"] = (first - referral).dt.total_seconds() / 86400
    temp["FirstToLastMSKDays"] = (last - first).dt.total_seconds() / 86400

    summary = _summarise_numeric(
        temp,
        ["ReferralToFirstMSKDays", "FirstToLastMSKDays"],
    )

    qa_rows = []
    for (flag, group), sub in temp.groupby([EXPOSURE_COL, GROUP_COL], dropna=False):
        for variable in ["ReferralToFirstMSKDays", "FirstToLastMSKDays"]:
            values = _numeric(sub[variable])
            qa_rows.append(
                {
                    "ExposureFlag": flag,
                    "group": group,
                    "interval": variable,
                    "available_n": int(values.notna().sum()),
                    "negative_n": int(values.lt(0).sum()),
                    "zero_n": int(values.eq(0).sum()),
                }
            )
    return summary, pd.DataFrame(qa_rows)


def _utilisation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate crude outcome distributions and person-time rates by group/period."""
    rows: list[dict[str, object]] = []
    for (flag, group), sub in df.groupby([EXPOSURE_COL, GROUP_COL], dropna=False):
        for metric in OUTCOME_METRICS:
            for period in ("Baseline", "FollowUp"):
                count_col = f"{period}{metric}Count"
                py_col = f"{period}PersonYears"
                if count_col not in sub or py_col not in sub:
                    continue
                counts = _numeric(sub[count_col]).fillna(0)
                py = _numeric(sub[py_col]).fillna(0)
                total_events = counts.sum()
                total_py = py.sum()
                variance = counts.var(ddof=1) if len(counts) > 1 else 0.0
                mean = counts.mean()
                rows.append(
                    {
                        "ExposureFlag": flag,
                        "group": group,
                        "period": period,
                        "outcome": metric,
                        "outcome_label": DISPLAY_LABELS.get(metric, metric),
                        "patients": len(sub),
                        "total_events": float(total_events),
                        "patients_with_event_n": int(counts.gt(0).sum()),
                        "patients_with_event_pct": _safe_pct(counts.gt(0).sum(), len(sub)),
                        "zero_event_n": int(counts.eq(0).sum()),
                        "zero_event_pct": _safe_pct(counts.eq(0).sum(), len(sub)),
                        "mean_events_per_patient": mean,
                        "sd_events_per_patient": counts.std(ddof=1) if len(counts) > 1 else 0.0,
                        "median_events_per_patient": counts.median(),
                        "q1_events_per_patient": counts.quantile(0.25),
                        "q3_events_per_patient": counts.quantile(0.75),
                        "p95_events_per_patient": counts.quantile(0.95),
                        "p99_events_per_patient": counts.quantile(0.99),
                        "max_events_per_patient": counts.max(),
                        "variance_to_mean_ratio": variance / mean if mean > 0 else np.nan,
                        "total_person_years": total_py,
                        "rate_per_person_year": total_events / total_py if total_py > 0 else np.nan,
                        "rate_per_100_person_years": (
                            total_events / total_py * 100 if total_py > 0 else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _prepost_change(utilisation: pd.DataFrame) -> pd.DataFrame:
    """Calculate crude within-group baseline-to-follow-up changes."""
    if utilisation.empty:
        return pd.DataFrame()
    key = ["ExposureFlag", "group", "outcome", "outcome_label"]
    rate = utilisation.pivot_table(
        index=key,
        columns="period",
        values="rate_per_person_year",
        aggfunc="first",
    ).reset_index()
    mean = utilisation.pivot_table(
        index=key,
        columns="period",
        values="mean_events_per_patient",
        aggfunc="first",
    ).reset_index()
    rate = rate.rename(columns={"Baseline": "baseline_rate_per_py", "FollowUp": "followup_rate_per_py"})
    mean = mean.rename(columns={"Baseline": "baseline_mean_count", "FollowUp": "followup_mean_count"})
    out = rate.merge(mean, on=key, how="outer")
    out["absolute_rate_change"] = out["followup_rate_per_py"] - out["baseline_rate_per_py"]
    out["relative_rate_change_pct"] = np.where(
        out["baseline_rate_per_py"].gt(0),
        out["absolute_rate_change"] / out["baseline_rate_per_py"] * 100,
        np.nan,
    )
    out["absolute_mean_count_change"] = out["followup_mean_count"] - out["baseline_mean_count"]
    return out


def _index_temporal_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise index-date counts over calendar time."""
    temp = df.copy()
    temp["IndexDate"] = _to_dt(temp["IndexDate"])
    temp = temp.dropna(subset=["IndexDate"])
    temp["IndexYear"] = temp["IndexDate"].dt.year.astype("Int64")
    temp["IndexMonth"] = temp["IndexDate"].dt.to_period("M").astype(str)

    annual = (
        temp.groupby([EXPOSURE_COL, GROUP_COL, "IndexYear"], as_index=False)
        .agg(patients=("PatientID", "nunique"))
    )
    annual["pct_within_group"] = annual["patients"] / annual.groupby(EXPOSURE_COL)["patients"].transform("sum") * 100

    monthly = (
        temp.groupby([EXPOSURE_COL, GROUP_COL, "IndexMonth"], as_index=False)
        .agg(patients=("PatientID", "nunique"))
    )
    monthly["pct_within_group"] = monthly["patients"] / monthly.groupby(EXPOSURE_COL)["patients"].transform("sum") * 100
    return annual, monthly


def _event_structure(
    ledger: pd.DataFrame,
    eligible: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise event-ledger composition by event type/source/period."""
    if ledger.empty:
        return pd.DataFrame(), pd.DataFrame()

    cols = ["PatientID", EXPOSURE_COL, GROUP_COL, "AnalysisEligibleFlag"]
    person = eligible[cols].drop_duplicates("PatientID")
    temp = ledger.merge(
        person[["PatientID", GROUP_COL, "AnalysisEligibleFlag"]],
        on="PatientID",
        how="inner",
        validate="many_to_one",
    )
    temp = temp[temp["AnalysisEligibleFlag"].eq(1)].copy()
    temp["EventDate"] = _to_dt(temp["EventDate"])

    structure = (
        temp.groupby(
            [EXPOSURE_COL, GROUP_COL, "EventType", "AnalysisPeriod", "SourceDataset"],
            dropna=False,
            as_index=False,
        )
        .agg(
            events=("EventID", "nunique"),
            patients_with_event=("PatientID", "nunique"),
        )
    )

    temp["CalendarMonth"] = temp["EventDate"].dt.to_period("M").astype(str)
    monthly = (
        temp[temp["AnalysisPeriod"].isin(["Baseline", "Follow-up"])]
        .groupby(
            [EXPOSURE_COL, GROUP_COL, "EventType", "AnalysisPeriod", "CalendarMonth"],
            as_index=False,
        )
        .agg(events=("EventID", "nunique"), patients=("PatientID", "nunique"))
    )
    return structure, monthly










def _table1(
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    balance: pd.DataFrame,
    group_names: dict[int, str],
) -> pd.DataFrame:
    """Assemble the report-facing baseline-characteristics Table 1."""
    comparison_name = group_names.get(0, "ExposureFlag=0")
    sports_name = group_names.get(1, "ExposureFlag=1")
    rows: list[dict[str, object]] = []

    for variable in ["AgeAtIndex", "Index_of_Multiple_Deprivation_IMD_Decile"]:
        x = numeric[numeric["variable"].eq(variable)]
        if x.empty:
            continue
        values = {}
        for flag in (0, 1):
            row = x[x["ExposureFlag"].eq(flag)]
            if row.empty:
                values[flag] = ""
            else:
                r = row.iloc[0]
                values[flag] = (
                    f"{r['mean']:.1f} ({r['sd']:.1f}); "
                    f"{r['median']:.1f} [{r['q1']:.1f}, {r['q3']:.1f}]"
                )
        smd_row = balance[(balance["variable"].eq(variable)) & (balance["level"].eq(""))]
        rows.append(
            {
                "Characteristic": DISPLAY_LABELS.get(variable, variable),
                "Level": "Mean (SD); median [Q1, Q3]",
                comparison_name: values[0],
                sports_name: values[1],
                "SMD": smd_row["smd"].iloc[0] if not smd_row.empty else np.nan,
            }
        )

    for variable in ["Sex", "AgeBand", "EthnicityNationalCodeDesc", "IMDQuintile", "PostcodeLAName"]:
        x = categorical[categorical["variable"].eq(variable)]
        if x.empty:
            continue
        for level in x["level"].drop_duplicates():
            values = {}
            for flag in (0, 1):
                row = x[(x["ExposureFlag"].eq(flag)) & (x["level"].eq(level))]
                values[flag] = (
                    f"{int(row['n'].iloc[0]):,} ({row['pct'].iloc[0]:.1f}%)"
                    if not row.empty
                    else "0 (0.0%)"
                )
            smd_row = balance[(balance["variable"].eq(variable)) & (balance["level"].astype(str).eq(str(level)))]
            rows.append(
                {
                    "Characteristic": DISPLAY_LABELS.get(variable, variable),
                    "Level": str(level),
                    comparison_name: values[0],
                    sports_name: values[1],
                    "SMD": smd_row["smd"].iloc[0] if not smd_row.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _eda_diagnostics(
    eligible: pd.DataFrame,
    balance: pd.DataFrame,
    utilisation: pd.DataFrame,
    missingness: pd.DataFrame,
) -> pd.DataFrame:
    """Create explicit review flags for descriptive data-quality or distributional concerns."""
    rows: list[dict[str, object]] = []

    max_smd = balance["abs_smd"].max() if not balance.empty else np.nan
    rows.append(
        {
            "domain": "baseline_comparability",
            "group": "All",
            "metric": "maximum absolute unweighted SMD",
            "value": max_smd,
            "reference": "0.10 descriptive balance threshold",
            "flag": "REVIEW" if pd.notna(max_smd) and max_smd >= 0.10 else "OK",
            "interpretation": "Large unweighted SMDs identify measured baseline differences requiring adjustment.",
        }
    )

    for (flag, group), sub in eligible.groupby([EXPOSURE_COL, GROUP_COL], dropna=False):
        full_fu = _numeric(sub.get("FullFollowUpFlag", pd.Series(0, index=sub.index))).fillna(0).eq(1)
        pct = _safe_pct(full_fu.sum(), len(sub))
        rows.append(
            {
                "domain": "followup",
                "group": group,
                "metric": "full 365-day follow-up (%)",
                "value": pct,
                "reference": "Describe incomplete follow-up; use person-time where appropriate",
                "flag": "REVIEW" if pd.notna(pct) and pct < 95 else "OK",
                "interpretation": "Incomplete follow-up can differ by group and is handled using observed person-time.",
            }
        )

    for _, row in utilisation.iterrows():
        if row["period"] != "Baseline":
            continue
        vmr = row["variance_to_mean_ratio"]
        zero = row["zero_event_pct"]
        rows.append(
            {
                "domain": "count_distribution",
                "group": row["group"],
                "metric": f"{row['outcome']} baseline variance/mean",
                "value": vmr,
                "reference": ">1 suggests over-dispersion",
                "flag": "REVIEW" if pd.notna(vmr) and vmr > 1.25 else "OK",
                "interpretation": "Supports checking robust Poisson and Negative Binomial specifications.",
            }
        )
        rows.append(
            {
                "domain": "count_distribution",
                "group": row["group"],
                "metric": f"{row['outcome']} baseline zero-event (%)",
                "value": zero,
                "reference": "High values indicate zero-heavy utilisation",
                "flag": "INFO",
                "interpretation": "Zero-heavy counts should be recognised when interpreting means and model fit.",
            }
        )

    eth = missingness[missingness["variable"].eq("EthnicityNationalCodeDesc")]
    for _, row in eth.iterrows():
        rows.append(
            {
                "domain": "data_quality",
                "group": row["group"],
                "metric": "ethnicity effective missing/uninformative (%)",
                "value": row["effective_missing_pct"],
                "reference": "Includes Not Stated/Not recorded/Unknown",
                "flag": "REVIEW" if row["effective_missing_pct"] >= 10 else "OK",
                "interpretation": "Literal null completeness may overstate the usable information content of ethnicity.",
            }
        )

    return pd.DataFrame(rows)



# -----------------------------------------------------------------------------
# Stakeholder-facing visualisation layer
# -----------------------------------------------------------------------------
# These figures are intentionally one-chart-per-file (no subplot grids) so they
# can be used directly in technical reports, slide decks and clinical review.
# The palette is kept consistent across figures: Wider MSK = navy, Sports-linked
# = coral. No background grid is used.
GROUP_COLOURS = {0: "#2F5597", 1: "#E15759"}
PERIOD_COLOURS = {"Baseline": "#4E79A7", "FollowUp": "#F28E2B"}
NEUTRAL_GREY = "#7A7A7A"


def _style_axis(ax: plt.Axes, *, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Apply restrained common formatting to a Matplotlib axis."""
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14, loc="left")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B5B5B5")
    ax.spines["bottom"].set_color("#B5B5B5")
    ax.tick_params(axis="both", labelsize=10)


def _group_colour_for_label(df: pd.DataFrame, label: object) -> str:
    """Return a stable plotting colour choice for one analysis-group label."""
    matches = df.loc[df[GROUP_COL].astype(str).eq(str(label)), EXPOSURE_COL]
    if not matches.empty:
        try:
            return GROUP_COLOURS.get(int(matches.iloc[0]), NEUTRAL_GREY)
        except (TypeError, ValueError):
            pass
    return NEUTRAL_GREY


def _save_grouped_horizontal_bar(
    categorical: pd.DataFrame,
    variable: str,
    figure_path: Path,
    title: str,
    *,
    max_levels: int | None = None,
    sort_by_exposed: bool = False,
) -> None:
    """Persist the corresponding aggregate analytical/QA output in a reproducible format."""
    x = categorical[categorical["variable"].eq(variable)].copy()
    if x.empty:
        return

    pivot = x.pivot(index="level", columns="ExposureFlag", values="pct").fillna(0)
    if max_levels is not None and len(pivot) > max_levels:
        order = pivot.sum(axis=1).sort_values(ascending=False).head(max_levels).index
        pivot = pivot.loc[order]
    elif sort_by_exposed and 1 in pivot.columns:
        pivot = pivot.sort_values(1, ascending=True)

    # Preserve natural ordering for the pre-defined reporting categories.
    if variable == "AgeBand":
        order = [v for v in ["16–34", "35–49", "50–64", "65–74", "75+"] if v in pivot.index]
        pivot = pivot.reindex(order)
    elif variable == "IMDQuintile":
        order = [v for v in ["Q1 (most deprived)", "Q2", "Q3", "Q4", "Q5 (least deprived)"] if v in pivot.index]
        pivot = pivot.reindex(order)

    fig_h = max(4.8, 0.48 * len(pivot) + 2.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    y = np.arange(len(pivot))
    width = 0.36

    group_names = (
        x[["ExposureFlag", "group"]]
        .drop_duplicates()
        .set_index("ExposureFlag")["group"]
        .to_dict()
    )

    flags = [f for f in [0, 1] if f in pivot.columns]
    offsets = {0: -width / 2, 1: width / 2}
    for flag in flags:
        vals = pivot[flag].to_numpy(dtype=float)
        bars = ax.barh(
            y + offsets.get(flag, 0),
            vals,
            height=width,
            color=GROUP_COLOURS.get(flag, NEUTRAL_GREY),
            label=str(group_names.get(flag, f"ExposureFlag={flag}")),
        )
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(
                    val + max(0.25, pivot.to_numpy().max() * 0.008),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    va="center",
                    fontsize=9,
                )

    ax.set_yticks(y)
    ax.set_yticklabels([str(v) for v in pivot.index])
    _style_axis(ax, title=title, xlabel="Patients (%)")
    ax.legend(frameon=False, title="Analysis group", loc="best")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_age_distribution(df: pd.DataFrame, path: Path) -> None:
    """Create and save the corresponding aggregate diagnostic/reporting figure."""
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bins = np.arange(15, 101, 5)
    for flag, sub in df.groupby(EXPOSURE_COL):
        values = _numeric(sub["AgeAtIndex"]).dropna()
        if values.empty:
            continue
        group = str(sub[GROUP_COL].iloc[0])
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.6,
            color=GROUP_COLOURS.get(int(flag), NEUTRAL_GREY),
            label=f"{group} (n={len(values):,})",
        )
        ax.axvline(
            values.median(),
            linestyle="--",
            linewidth=1.4,
            color=GROUP_COLOURS.get(int(flag), NEUTRAL_GREY),
            alpha=0.9,
        )
    _style_axis(ax, title="Age distribution at the analytical index", xlabel="Age at index (years)", ylabel="Density")
    ax.legend(frameon=False, title="Analysis group")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)






def _plot_utilisation_rates(utilisation: pd.DataFrame, figure_dir: Path) -> list[dict[str, str]]:
    """Create and save the corresponding aggregate diagnostic/reporting figure."""
    manifest: list[dict[str, str]] = []
    for outcome in OUTCOME_METRICS:
        x = utilisation[utilisation["outcome"].eq(outcome)].copy()
        if x.empty:
            continue
        groups = list(dict.fromkeys(x["group"].astype(str)))
        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        y = np.arange(len(groups))
        height = 0.34
        for period, offset in [("Baseline", -height/2), ("FollowUp", height/2)]:
            values = []
            for group in groups:
                z = x[(x["group"].astype(str).eq(group)) & (x["period"].eq(period))]
                values.append(float(z["rate_per_person_year"].iloc[0]) if not z.empty else np.nan)
            bars = ax.barh(y + offset, values, height=height, color=PERIOD_COLOURS[period], label=period)
            for bar, val in zip(bars, values):
                if np.isfinite(val):
                    ax.text(val + 0.01, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(groups)
        _style_axis(ax, title=f"{DISPLAY_LABELS.get(outcome, outcome)}: baseline vs follow-up", xlabel="Crude events per person-year")
        ax.legend(frameon=False, title="Period")
        ax.set_xlim(left=0)
        fig.tight_layout()
        filename = f"utilisation_{_slug(outcome)}_baseline_followup.png"
        fig.savefig(figure_dir / filename, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        manifest.append({"file": filename, "purpose": f"Baseline and follow-up crude {outcome} rates by analysis group."})
    return manifest


def _plot_baseline_smd(balance: pd.DataFrame, path: Path, top_n: int = 20) -> None:
    """Create and save the corresponding aggregate diagnostic/reporting figure."""
    x = balance.dropna(subset=["abs_smd"]).sort_values("abs_smd", ascending=False).head(top_n).copy()
    if x.empty:
        return
    x["display"] = np.where(
        x["level"].astype(str).str.len().gt(0),
        x["label"] + ": " + x["level"].astype(str),
        x["label"],
    )
    x = x.sort_values("abs_smd")
    fig, ax = plt.subplots(figsize=(10.5, max(6.2, len(x) * 0.38)))
    colours = np.where(x["abs_smd"].to_numpy() >= 0.10, "#E15759", "#59A14F")
    ax.scatter(x["abs_smd"], x["display"], s=60, c=colours)
    ax.axvline(0.10, linestyle="--", linewidth=1.5, color="#555555")
    ax.text(0.10, len(x)-0.4, " 0.10 review threshold", fontsize=9, va="top")
    _style_axis(ax, title="Largest unadjusted baseline differences", xlabel="Absolute standardised mean difference")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
















def _plot_pathway_timing(pathway: pd.DataFrame, figure_dir: Path) -> list[dict[str, str]]:
    """Create and save the corresponding aggregate diagnostic/reporting figure."""
    manifest: list[dict[str, str]] = []
    if pathway.empty:
        return manifest
    metric_labels = {
        "ReferralToFirstMSKDays": "Time from MSK referral to FirstMSKDate",
        "FirstToLastMSKDays": "Time from FirstMSKDate to LastMSKDate",
    }
    for metric, label in metric_labels.items():
        x = pathway[pathway["variable"].eq(metric)].copy()
        if x.empty:
            continue
        groups = list(x["group"].astype(str))
        medians = x["median"].to_numpy(dtype=float)
        q1 = x["q1"].to_numpy(dtype=float)
        q3 = x["q3"].to_numpy(dtype=float)
        p95 = x["p95"].to_numpy(dtype=float)
        y = np.arange(len(groups))
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        for i, row in x.reset_index(drop=True).iterrows():
            flag = int(row[EXPOSURE_COL]) if pd.notna(row[EXPOSURE_COL]) else i
            c = GROUP_COLOURS.get(flag, NEUTRAL_GREY)
            ax.hlines(i, row["q1"], row["q3"], linewidth=8, color=c, alpha=0.75)
            ax.hlines(i, row["q3"], row["p95"], linewidth=2, color=c, alpha=0.65)
            ax.scatter(row["median"], i, s=85, color=c, edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(row["p95"] + 1, i, f"median {row['median']:.1f}", va="center", fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(groups)
        _style_axis(ax, title=label, xlabel="Days")
        fig.tight_layout()
        filename = f"pathway_timing_{_slug(metric)}.png"
        fig.savefig(figure_dir / filename, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        manifest.append({"file": filename, "purpose": f"Median, IQR and p95 for {label.lower()} by analysis group."})
    return manifest



def _write_manifest(table_dir: Path, figure_manifest: list[dict[str, str]], tables: dict[str, pd.DataFrame]) -> None:
    """Write a descriptive-output manifest for reproducible review."""
    table_purpose = {
        "cohort_flow": "Patient flow, observability and eligibility by analysis group.",
        "baseline_numeric_summary": "Baseline numeric characteristics and utilisation distributions.",
        "baseline_categorical_summary": "Baseline categorical characteristics by analysis group.",
        "table1_baseline_characteristics": "Report-ready baseline characteristics table with unadjusted SMDs.",
        "baseline_balance_smd": "Unadjusted standardised mean differences before propensity adjustment.",
        "missingness_summary": "Literal and effective information missingness by group.",
        "source_coverage_summary": "Cross-source patient presence/coverage by analysis group.",
        "followup_observation_summary": "Observed follow-up completeness and duration by analysis group.",
        "pathway_timing_summary": "Source-defined referral-to-FirstMSKDate and FirstMSKDate-to-LastMSKDate interval distributions.",
        "pathway_timing_qa": "Temporal interval QA including negative/zero intervals.",
        "utilisation_summary": "Baseline/follow-up event burden, zero inflation, dispersion and crude rates.",
        "prepost_change_summary": "Within-group crude baseline-to-follow-up utilisation changes.",
        "index_annual_summary": "Annual analytical-index distribution by group.",
        "index_monthly_summary": "Monthly analytical-index distribution by group.",
        "event_structure_summary": "Event-ledger structure by group, source, event type and analysis period.",
        "event_calendar_month_summary": "Calendar-month event counts among eligible patients.",
        "eda_diagnostics": "High-level diagnostics that inform comparative-model design and stakeholder review.",
    }
    pd.DataFrame(
        [{"file": f"{name}.csv", "purpose": table_purpose.get(name, name)} for name in tables]
    ).to_csv(table_dir / "table_manifest.csv", index=False)
    pd.DataFrame(figure_manifest).to_csv(table_dir / "figure_manifest.csv", index=False)


def run_descriptive(
    config_path: str | Path = "config/workflow_tre.yaml",
) -> dict[str, pd.DataFrame]:
    """Run the complete unadjusted descriptive/EDA layer and emit aggregate key findings."""
    cfg = load_workflow_config(config_path)
    analysis_dir = output_path(cfg, "analysis_dir")
    out_dir = output_path(cfg, "descriptive_dir")
    table_dir = out_dir / "tables"
    figure_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Stage 07 owns these folders. Clear prior Stage 07 artefacts so removed
    # figures/tables cannot remain on disk and be mistaken for current outputs.
    for stale in figure_dir.glob("*.png"):
        stale.unlink()
    for stale in table_dir.glob("*.csv"):
        stale.unlink()

    stage_header(
        "07",
        "DESCRIPTIVE / EXPLORATORY ANALYSIS",
        purpose=(
            "Describe the analytical population before propensity adjustment: cohort composition, "
            "baseline characteristics, missingness, source coverage, pathway timing, follow-up completeness, "
            "crude hospital-utilisation rates, baseline/follow-up change and unadjusted baseline imbalance."
        ),
        inputs=[analysis_dir / "patient_outcomes.csv", analysis_dir / "healthcare_event_ledger.csv"],
        outputs=[table_dir, figure_dir],
    )

    patient_path = analysis_dir / "patient_outcomes.csv"
    ledger_path = analysis_dir / "healthcare_event_ledger.csv"
    if not patient_path.exists():
        raise FileNotFoundError(patient_path)
    if not ledger_path.exists():
        raise FileNotFoundError(ledger_path)

    df = pd.read_csv(patient_path, low_memory=False)
    ledger = pd.read_csv(ledger_path, low_memory=False)

    required = ["PatientID", EXPOSURE_COL, GROUP_COL, "AnalysisEligibleFlag", "IndexDate"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"patient_outcomes.csv is missing required fields: {missing_required}")

    df = _derive_reporting_features(df)
    eligible = df[df["AnalysisEligibleFlag"].eq(1)].copy()
    if eligible.empty:
        raise ValueError("No analysis-eligible patients are available for descriptive analysis.")

    # Safety/interpretation guardrails: the fallback comparison is an analysis-group
    # comparison, not a confirmed Active Blackpool programme-start evaluation.
    programme_confirmed = bool(
        _numeric(df.get("ProgrammeExposureSemanticsConfirmedFlag", pd.Series([0]))).fillna(0).eq(1).all()
    )
    index_is_programme_start = bool(
        _numeric(df.get("IndexIsProgrammeStartFlag", pd.Series([0]))).fillna(0).eq(1).all()
    )

    cohort_flow = _cohort_flow(df)
    numeric = _summarise_numeric(eligible, NUMERIC_BASELINE_VARS + ["FollowUpDaysAvailable", "FollowUpPersonYears"])
    categorical = _summarise_categories(eligible, CATEGORICAL_BASELINE_VARS + ["FullFollowUpFlag"])
    balance = _baseline_balance(eligible, NUMERIC_BASELINE_VARS, CATEGORICAL_BASELINE_VARS)
    missingness = _missingness_summary(eligible)
    source_coverage = _source_coverage(eligible)
    followup_observation = _followup_observation_summary(eligible)
    pathway_timing, pathway_timing_qa = _pathway_timing(eligible)
    utilisation = _utilisation_summary(eligible)
    prepost = _prepost_change(utilisation)
    # Calendar-time summaries are retained as QA tables because index-year
    # comparability matters for the propensity design; no index-timeline plot is produced.
    annual_index, monthly_index = _index_temporal_summary(eligible)
    event_structure, event_monthly = _event_structure(ledger, eligible)
    group_names = _group_lookup(eligible)
    table1 = _table1(numeric, categorical, balance, group_names)
    diagnostics = _eda_diagnostics(eligible, balance, utilisation, missingness)

    outputs: dict[str, pd.DataFrame] = {
        "cohort_flow": cohort_flow,
        "baseline_numeric_summary": numeric,
        "baseline_categorical_summary": categorical,
        "table1_baseline_characteristics": table1,
        "baseline_balance_smd": balance,
        "missingness_summary": missingness,
        "source_coverage_summary": source_coverage,
        "followup_observation_summary": followup_observation,
        "pathway_timing_summary": pathway_timing,
        "pathway_timing_qa": pathway_timing_qa,
        "utilisation_summary": utilisation,
        "prepost_change_summary": prepost,
        "index_annual_summary": annual_index,
        "index_monthly_summary": monthly_index,
        "event_structure_summary": event_structure,
        "event_calendar_month_summary": event_monthly,
        "eda_diagnostics": diagnostics,
    }

    for name, table in outputs.items():
        table.to_csv(table_dir / f"{name}.csv", index=False)

    # Build a compact aggregate key-findings table for fast reviewer audit.  The
    # full descriptive tables remain available separately; this file surfaces
    # the main denominator, missingness, imbalance and crude-utilisation signals.
    key_rows: list[dict[str, object]] = []
    key_rows.append({
        "domain": "cohort",
        "metric": "analysis_eligible_patients",
        "group": "All",
        "value": int(eligible["PatientID"].nunique()),
        "unit": "patients",
    })
    for flag in (0, 1):
        sub = eligible[eligible[EXPOSURE_COL].eq(flag)]
        if not sub.empty:
            key_rows.append({
                "domain": "cohort",
                "metric": "analysis_eligible_patients",
                "group": group_names.get(flag, str(flag)),
                "value": int(sub["PatientID"].nunique()),
                "unit": "patients",
            })

    if not balance.empty:
        max_row = balance.loc[balance["abs_smd"].idxmax()]
        key_rows.append({
            "domain": "baseline_comparability",
            "metric": "maximum_absolute_unweighted_smd",
            "group": "All",
            "value": float(max_row["abs_smd"]),
            "unit": "SMD",
            "detail": str(max_row.get("feature", max_row.get("variable", ""))),
        })

    # Capture the largest effective missingness value within each analysis group.
    if not missingness.empty:
        for group, sub in missingness.groupby("group", dropna=False):
            row = sub.sort_values("effective_missing_pct", ascending=False).iloc[0]
            key_rows.append({
                "domain": "missingness",
                "metric": "largest_effective_missingness",
                "group": group,
                "value": float(row["effective_missing_pct"]),
                "unit": "percent",
                "detail": str(row["variable"]),
            })

    # Capture every crude baseline/follow-up rate so the reviewer can reconcile
    # the terminal summary against utilisation_summary.csv.
    if not utilisation.empty:
        for row in utilisation.itertuples(index=False):
            key_rows.append({
                "domain": "crude_utilisation",
                "metric": f"{row.outcome}_{row.period}_rate",
                "group": row.group,
                "value": float(row.rate_per_100_person_years) if pd.notna(row.rate_per_100_person_years) else np.nan,
                "unit": "events per 100 person-years",
                "detail": f"events={int(row.total_events)}; person_years={row.total_person_years:.3f}",
            })

    key_findings_table = pd.DataFrame(key_rows)
    key_findings_table.to_csv(table_dir / "descriptive_key_findings.csv", index=False)

    figure_manifest: list[dict[str, str]] = []

    _plot_age_distribution(eligible, figure_dir / "age_distribution.png")
    figure_manifest.append({"file": "age_distribution.png", "purpose": "Age distribution at the analytical index by analysis group."})

    demographic_plots = [
        ("Sex", "sex_distribution.png", "Sex distribution by analysis group", None),
        ("EthnicityNationalCodeDesc", "ethnicity_distribution.png", "Recorded ethnicity profile by analysis group", 10),
        ("IMDQuintile", "imd_quintile_distribution.png", "Deprivation profile by analysis group", None),
        ("PostcodeLAName", "geography_distribution.png", "Geographical profile by analysis group", 12),
    ]
    for variable, filename, title, max_levels in demographic_plots:
        if variable in eligible:
            _save_grouped_horizontal_bar(
                categorical, variable, figure_dir / filename, title,
                max_levels=max_levels, sort_by_exposed=variable in {"EthnicityNationalCodeDesc", "PostcodeLAName"},
            )
            figure_manifest.append({"file": filename, "purpose": f"{DISPLAY_LABELS.get(variable, variable)} distribution by analysis group."})

    figure_manifest.extend(_plot_utilisation_rates(utilisation, figure_dir))

    _plot_baseline_smd(balance, figure_dir / "baseline_smd_top20.png")
    figure_manifest.append({"file": "baseline_smd_top20.png", "purpose": "Largest unadjusted baseline SMDs before propensity adjustment."})

    figure_manifest.extend(_plot_pathway_timing(pathway_timing, figure_dir))

    _write_manifest(table_dir, figure_manifest, outputs)

    section("STAGE 07 KEY FINDINGS")
    metric("working comparative patients", f"{df['PatientID'].nunique():,}")
    metric("analysis-eligible patients", f"{eligible['PatientID'].nunique():,}")
    for flag in (0, 1):
        sub = eligible[eligible[EXPOSURE_COL].eq(flag)]
        if not sub.empty:
            label = group_names.get(flag, f"ExposureFlag={flag}")
            full_fu = _numeric(sub.get("FullFollowUpFlag", pd.Series(0, index=sub.index))).fillna(0).sum()
            metric(f"{label}: eligible", f"{sub['PatientID'].nunique():,}")
            metric(f"{label}: full follow-up", f"{int(full_fu):,}")

    max_smd = float(balance["abs_smd"].max()) if not balance.empty else np.nan
    metric("maximum unadjusted |SMD|", f"{max_smd:.4f}" if np.isfinite(max_smd) else "NA")
    if not balance.empty:
        print("\nLargest unadjusted baseline differences:")
        dataframe_preview(
            balance.sort_values("abs_smd", ascending=False),
            columns=["feature", "variable", "level", "abs_smd"],
            max_rows=10,
        )

    if not missingness.empty:
        print("\nLargest effective missingness by analysis group:")
        top_missing = (
            missingness.sort_values("effective_missing_pct", ascending=False)
            .groupby("group", as_index=False, group_keys=False)
            .head(6)
        )
        dataframe_preview(
            top_missing,
            columns=["group", "variable", "patients", "literal_missing_pct", "effective_missing_pct"],
            max_rows=15,
        )

    print("\nCrude baseline/follow-up hospital-utilisation rates:")
    dataframe_preview(
        utilisation,
        columns=[
            "group", "period", "outcome_label", "patients", "total_events",
            "total_person_years", "rate_per_100_person_years", "zero_event_pct",
            "variance_to_mean_ratio",
        ],
        max_rows=20,
    )

    print("\nCrude pre/post change summary:")
    dataframe_preview(
        prepost,
        columns=[
            "group", "outcome_label", "baseline_rate_per_py", "followup_rate_per_py",
            "absolute_rate_change", "relative_rate_change_pct",
        ],
        max_rows=12,
    )

    review_diagnostics = diagnostics[diagnostics["flag"].isin(["REVIEW"])] if not diagnostics.empty else pd.DataFrame()
    metric("EDA diagnostics flagged for review", len(review_diagnostics))
    if not review_diagnostics.empty:
        print("\nEDA decision flags requiring review:")
        dataframe_preview(
            review_diagnostics,
            columns=["domain", "group", "metric", "value", "reference", "interpretation"],
            max_rows=20,
        )

    audit_dir = output_path(cfg, "audit_dir")
    summary_path = save_stage_summary(
        audit_dir,
        stage_key="descriptive",
        stage_code="07",
        title="Descriptive / exploratory analysis",
        status="PASS",
        key_findings={
            "working_comparative_patients": int(df["PatientID"].nunique()),
            "analysis_eligible_patients": int(eligible["PatientID"].nunique()),
            "maximum_unadjusted_abs_smd": max_smd,
            "eda_review_flags_n": len(review_diagnostics),
            "programme_exposure_semantics_confirmed": programme_confirmed,
            "index_is_programme_start": index_is_programme_start,
        },
        qa_files=[
            table_dir / "descriptive_key_findings.csv",
            table_dir / "table1_baseline_characteristics.csv",
            table_dir / "missingness_summary.csv",
            table_dir / "followup_observation_summary.csv",
            table_dir / "baseline_balance_smd.csv",
            table_dir / "utilisation_summary.csv",
            table_dir / "prepost_change_summary.csv",
            table_dir / "eda_diagnostics.csv",
        ],
        warnings=[
            "All Stage 07 group differences are crude/unadjusted and must not be interpreted as programme effects.",
            "High effective ethnicity missingness includes uninformative categories such as Not Stated/Unknown, not only literal NA.",
            "A large unadjusted SMD is a reason to proceed to the propensity design, not a failure of the descriptive stage."
        ],
        config_path=config_path,
    )
    stage_footer(
        stage_key="descriptive",
        audit_dir=audit_dir,
        summary_path=summary_path,
        qa_files=[table_dir / "descriptive_key_findings.csv", table_dir / "eda_diagnostics.csv"],
    )

    outputs["descriptive_key_findings"] = key_findings_table
    return outputs
