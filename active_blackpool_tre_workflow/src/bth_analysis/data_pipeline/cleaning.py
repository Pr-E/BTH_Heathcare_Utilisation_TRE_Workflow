"""Stage 02: deterministic source cleaning and data-quality audit.

The stage standardises blanks, parses configured dates/numerics, removes only
fully blank or duplicate records under explicit rules, checks event/chronology
integrity, verifies identifier roles, and quantifies column-level missingness.
No exposure, analytical-index or outcome-window derivation occurs here.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from .config import load_pipeline_config, output_dir
from .schemas import (
    MSK_DATE_COLUMNS,
    SPORTS_EXTRA_DATE_COLUMNS,
    INPATIENT_DATE_COLUMNS,
    ED_DATE_COLUMNS,
)
from .qa import basic_table_summary, print_table_summary, save_records
from .identifiers import resolve_identifier_plan, apply_identifier_choices_to_config
from .missingness import (
    column_missingness_table,
    compare_group_missingness,
    print_group_missingness_comparison,
    print_missingness_summary,
)
from bth_analysis.audit import (
    metric,
    save_stage_summary,
    section,
    stage_footer,
    stage_header,
)


def normalize_blank_strings(df):
    """Convert whitespace-only text cells to explicit pandas missing values.

    This standardises visually blank strings such as ``"   "`` so downstream
    missingness counts do not treat them as valid categorical values.
    """
    # Work on a copy so the caller's input frame is never mutated in-place.
    out = df.copy()
    object_cols = out.select_dtypes(include=["object"]).columns
    if len(object_cols):
        out[object_cols] = out[object_cols].replace(r"^\s*$", pd.NA, regex=True)
    return out


def parse_dates(df, columns):
    """Parse configured source date/time fields and coerce invalid text to NaT.

    ``errors="coerce"`` is deliberate: an invalid clinical timestamp is safer
    represented as missing and audited than silently guessed or corrected.
    """
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(
                out[col],
                errors="coerce",
                dayfirst=True,
                format="mixed",
            )
    return out


def coerce_numeric(df, columns):
    """Convert configured numeric fields to numbers; malformed values become NA."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def clean_msk(df, table_key, table_cfg, config):
    """Clean one MSK pathway table and return aggregate chronology QA.

    The function removes only fully blank/exact-duplicate rows, standardises
    configured date/numeric types and *counts* impossible pathway ordering.
    Chronology problems are not silently repaired because source-owner review may
    be needed to decide whether the timestamp or its semantic meaning is wrong.
    """
    issues = []
    before = len(df)

    if config["cleaning"].get("normalize_blank_strings", True):
        df = normalize_blank_strings(df)

    blank_n = int(df.isna().all(axis=1).sum())
    if config["cleaning"].get("drop_fully_blank_rows", True):
        df = df.loc[~df.isna().all(axis=1)].copy()

    duplicate_n = int(df.duplicated().sum())
    if config["cleaning"].get("drop_exact_duplicates", True):
        df = df.drop_duplicates().copy()

    date_cols = list(MSK_DATE_COLUMNS)
    if table_key == "msk_sports":
        date_cols += SPORTS_EXTRA_DATE_COLUMNS
    df = parse_dates(df, date_cols)

    numeric_cols = [
        "ReferralObservationId",
        "Age",
        "Index_of_Multiple_Deprivation_IMD_Decile",
        "NewMSKReferralObservationId",
        "NewOtherReferralObservationId",
    ]
    df = coerce_numeric(df, numeric_cols)

    referral_before_first = 0
    last_before_first = 0
    last_before_referral = 0

    if {"FirstMSKReferralDate", "FirstMSKDate"} <= set(df.columns):
        referral_before_first = int(
            (
                df["FirstMSKDate"].notna()
                & df["FirstMSKReferralDate"].notna()
                & (df["FirstMSKDate"] < df["FirstMSKReferralDate"])
            ).sum()
        )

    if {"LastMSKDate", "FirstMSKDate"} <= set(df.columns):
        last_before_first = int(
            (
                df["LastMSKDate"].notna()
                & df["FirstMSKDate"].notna()
                & (df["LastMSKDate"] < df["FirstMSKDate"])
            ).sum()
        )

    if {"LastMSKDate", "FirstMSKReferralDate"} <= set(df.columns):
        last_before_referral = int(
            (
                df["LastMSKDate"].notna()
                & df["FirstMSKReferralDate"].notna()
                & (df["LastMSKDate"] < df["FirstMSKReferralDate"])
            ).sum()
        )

    issues.append({
        "table": table_key,
        "rows_input": before,
        "fully_blank_rows_removed": blank_n,
        "exact_duplicate_rows_removed": duplicate_n,
        "duplicate_patient_event_rows_removed": 0,
        "age_sentinel_values_replaced": 0,
        "invalid_departure_values_replaced": 0,
        "negative_or_invalid_chronology_n": (
            referral_before_first + last_before_first + last_before_referral
        ),
        "msk_first_before_referral_n": referral_before_first,
        "msk_last_before_first_n": last_before_first,
        "msk_last_before_referral_n": last_before_referral,
        "rows_output": len(df),
    })
    return df, issues


def clean_inpatient(df, table_key, table_cfg, config):
    """Clean one inpatient episode table without collapsing episodes to spells.

    Spell/episode structural checks happen here, while the actual episode-to-spell
    aggregation is deferred to Stage 03 so the source grain remains auditable.
    """
    before = len(df)

    if config["cleaning"].get("normalize_blank_strings", True):
        df = normalize_blank_strings(df)

    blank_n = int(df.isna().all(axis=1).sum())
    if config["cleaning"].get("drop_fully_blank_rows", True):
        df = df.loc[~df.isna().all(axis=1)].copy()

    duplicate_n = int(df.duplicated().sum())
    if config["cleaning"].get("drop_exact_duplicates", True):
        df = df.drop_duplicates().copy()

    df = parse_dates(df, INPATIENT_DATE_COLUMNS)
    df = coerce_numeric(
        df,
        [
            "EpisodeId",
            "SpellID",
            "Age",
            "Index_of_Multiple_Deprivation_IMD_Decile",
        ],
    )

    discharge_before_admission = int(
        (
            df["DischargeDate"].notna()
            & df["AdmissionDate"].notna()
            & (df["DischargeDate"] < df["AdmissionDate"])
        ).sum()
    )
    episode_end_before_start = int(
        (
            df["Episodeend"].notna()
            & df["EpisodeStart"].notna()
            & (df["Episodeend"] < df["EpisodeStart"])
        ).sum()
    )
    episode_start_before_admission = int(
        (
            df["EpisodeStart"].notna()
            & df["AdmissionDate"].notna()
            & (df["EpisodeStart"] < df["AdmissionDate"])
        ).sum()
    )
    episode_end_after_discharge = int(
        (
            df["Episodeend"].notna()
            & df["DischargeDate"].notna()
            & (df["Episodeend"] > df["DischargeDate"])
        ).sum()
    )

    pid = table_cfg["patient_id"]
    spell = table_cfg["spell_id"]
    spell_patient_n = (
        df.dropna(subset=[pid, spell])
        .groupby(spell)[pid]
        .nunique()
    )
    spells_multiple_patients = int((spell_patient_n > 1).sum())

    repeated_episode_ids = int(
        df[table_cfg["episode_id"]].dropna().duplicated(keep=False).sum()
    )

    issues = [{
        "table": table_key,
        "rows_input": before,
        "fully_blank_rows_removed": blank_n,
        "exact_duplicate_rows_removed": duplicate_n,
        "duplicate_patient_event_rows_removed": 0,
        "age_sentinel_values_replaced": 0,
        "invalid_departure_values_replaced": 0,
        "negative_or_invalid_chronology_n": (
            discharge_before_admission
            + episode_end_before_start
            + episode_start_before_admission
            + episode_end_after_discharge
        ),
        "discharge_before_admission_n": discharge_before_admission,
        "episode_end_before_start_n": episode_end_before_start,
        "episode_start_before_admission_n": episode_start_before_admission,
        "episode_end_after_discharge_n": episode_end_after_discharge,
        "spells_linked_to_multiple_patients_n": spells_multiple_patients,
        "rows_with_repeated_episode_id_n": repeated_episode_ids,
        "rows_output": len(df),
    }]
    return df, issues


def clean_ed(df, table_key, table_cfg, config):
    """Clean one ED attendance table and enforce the resolved patient/event key.

    Invalid ages/departure timestamps are converted to missing rather than
    guessed.  Duplicate patient-event keys are removed only after the identifier resolver has selected the patient and attendance identifier fields.
    """
    before = len(df)

    if config["cleaning"].get("normalize_blank_strings", True):
        df = normalize_blank_strings(df)

    blank_n = int(df.isna().all(axis=1).sum())
    if config["cleaning"].get("drop_fully_blank_rows", True):
        df = df.loc[~df.isna().all(axis=1)].copy()

    duplicate_n = int(df.duplicated().sum())
    if config["cleaning"].get("drop_exact_duplicates", True):
        df = df.drop_duplicates().copy()

    pid = table_cfg["patient_id"]
    event_id = table_cfg["event_id"]

    duplicate_key_n = 0
    if (
        config["cleaning"].get(
            "ed_drop_duplicate_patient_event_keys", True
        )
        and pid in df.columns
        and event_id in df.columns
    ):
        key_dup = df.duplicated([pid, event_id], keep="first")
        duplicate_key_n = int(key_dup.sum())
        df = df.loc[~key_dup].copy()

    df = parse_dates(df, ED_DATE_COLUMNS)
    df = coerce_numeric(
        df,
        [
            "AgeAttendance",
            "Gender",
            "ArrivalHour",
            "LOS",
            "ModeOfArrival",
            "Index_of_Multiple_Deprivation_IMD_Decile",
        ],
    )

    sentinel_values = set(
        config["cleaning"].get("ed_age_sentinels", [])
    )
    explicit_age_sentinel = df["AgeAttendance"].isin(sentinel_values)

    age_min = config["cleaning"].get("ed_age_valid_min", 0)
    age_max = config["cleaning"].get("ed_age_valid_max", 120)
    implausible_age = (
        df["AgeAttendance"].notna()
        & (
            df["AgeAttendance"].lt(float(age_min))
            | df["AgeAttendance"].gt(float(age_max))
        )
    )

    age_sentinel_mask = explicit_age_sentinel | implausible_age
    age_sentinel_n = int(age_sentinel_mask.sum())
    age_implausible_n = int(implausible_age.sum())
    df.loc[age_sentinel_mask, "AgeAttendance"] = np.nan

    invalid_years = set(
        config["cleaning"].get("ed_invalid_departure_years", [])
    )
    invalid_departure = (
        df["Departure"].notna()
        & df["Departure"].dt.year.isin(invalid_years)
    )

    negative_departure = (
        df["Departure"].notna()
        & df["ArrivalDateTime"].notna()
        & (df["Departure"] < df["ArrivalDateTime"])
    )

    invalid_departure_mask = invalid_departure | negative_departure
    invalid_departure_n = int(invalid_departure_mask.sum())

    if invalid_departure_n:
        df.loc[invalid_departure_mask, "Departure"] = pd.NaT
        df.loc[invalid_departure_mask, "LOS"] = np.nan

    issues = [{
        "table": table_key,
        "rows_input": before,
        "fully_blank_rows_removed": blank_n,
        "exact_duplicate_rows_removed": duplicate_n,
        "duplicate_patient_event_rows_removed": duplicate_key_n,
        "age_sentinel_values_replaced": age_sentinel_n,
        "age_implausible_values_replaced": age_implausible_n,
        "invalid_departure_values_replaced": invalid_departure_n,
        "negative_or_invalid_chronology_n": invalid_departure_n,
        "rows_output": len(df),
    }]
    return df, issues


def save_cleaned(df, path):
    """Persist the cleaned canonical table using an explicit datetime format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )


def run_cleaning(config_path="config/pipeline_tre.yaml"):
    """Clean all ingested TRE tables after resolving source identifiers.

    The real extracts contain several SHA-256-labelled fields whose semantic
    role is verified by cross-source overlap.  Identifier resolution therefore
    happens once across all ingested tables before any patient/event-key based
    cleaning is attempted.
    """
    config = load_pipeline_config(config_path)
    in_dir = output_dir(config, "ingested_dir")
    out_dir = output_dir(config, "cleaned_dir")
    qa_dir = output_dir(config, "qa_dir")

    stage_header(
        "02",
        "DETERMINISTIC SOURCE CLEANING + MISSINGNESS AUDIT",
        purpose=(
            "Standardise source values, remove only clearly invalid duplicate/blank records, "
            "verify chronology/event keys and quantify column-level missingness. No cohort, "
            "exposure, index or outcome-window derivation occurs at this stage."
        ),
        inputs=[in_dir],
        outputs=[out_dir, qa_dir],
    )

    if not in_dir.exists():
        raise FileNotFoundError(
            f"Ingested directory not found: {in_dir}. Run Stage 01 first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    # Load all six real source tables first so healthcare identifiers can be
    # verified against the corresponding MSK cohort rather than trusted solely
    # from a potentially misleading hash column name.
    raw_tables = {}
    for table_key, table_cfg in config["tables"].items():
        source_path = in_dir / table_cfg.get(
            "canonical_filename", f"{Path(table_cfg['filename']).stem}.csv"
        )
        if not source_path.exists():
            raise FileNotFoundError(
                f"{table_key}: ingested file not found: {source_path}"
            )
        raw_tables[table_key] = pd.read_csv(source_path, low_memory=False)

    id_cfg = config.get("identifier_resolution", {}) or {}
    choices, id_audit = resolve_identifier_plan(
        raw_tables,
        config["tables"],
        fail_if_no_reference_overlap=bool(
            id_cfg.get("fail_if_no_reference_overlap", True)
        ),
        fail_if_ambiguous=bool(id_cfg.get("fail_if_ambiguous", True)),
    )
    resolved_table_cfgs = apply_identifier_choices_to_config(
        config["tables"], choices
    )
    id_audit.to_csv(qa_dir / "02_identifier_resolution_qa.csv", index=False)

    # Print the selected patient identifiers with the overlap evidence used to
    # verify them against the matching MSK cohort. Event identifiers are shown
    # separately because their quality criterion is uniqueness, not MSK overlap.
    section("IDENTIFIER VERIFICATION")
    selected = id_audit[id_audit["selected"].eq(1)].copy() if not id_audit.empty else pd.DataFrame()
    selected_patient = selected[selected["identifier_role"].eq("patient")].copy()
    selected_event = selected[selected["identifier_role"].eq("event")].copy()

    if not selected_patient.empty:
        print("Selected patient identifiers:")
        print(selected_patient[[
            "table", "candidate", "unique_n", "reference_overlap_n",
            "reference_overlap_pct_of_candidate_unique",
            "reference_overlap_pct_of_reference",
        ]].to_string(index=False))
    if not selected_event.empty:
        print("\nSelected event identifiers:")
        print(selected_event[[
            "table", "candidate", "non_missing_n", "unique_n", "uniqueness_ratio"
        ]].to_string(index=False))

    # Each list below contains aggregate QA only.  Patient hashes are never written
    # into the cleaning audit summaries.
    cleaning_records = []
    cleaned_summaries = []
    column_missingness_frames = []

    for table_key, table_cfg in resolved_table_cfgs.items():
        target_path = out_dir / table_cfg.get(
            "canonical_filename", f"{Path(table_cfg['filename']).stem}.csv"
        )
        df = raw_tables[table_key]

        if table_cfg["type"] == "msk":
            cleaned, issues = clean_msk(df, table_key, table_cfg, config)
        elif table_cfg["type"] == "inpatient":
            cleaned, issues = clean_inpatient(df, table_key, table_cfg, config)
        elif table_cfg["type"] == "ed":
            cleaned, issues = clean_ed(df, table_key, table_cfg, config)
        else:
            raise ValueError(
                f"{table_key}: unknown table type {table_cfg['type']!r}"
            )

        save_cleaned(cleaned, target_path)

        summary = basic_table_summary(cleaned, table_key, table_cfg)
        summary["resolved_patient_id_column"] = table_cfg.get("patient_id")
        summary["resolved_event_id_column"] = table_cfg.get("event_id")
        cleaned_summaries.append(summary)
        cleaning_records.extend(issues)

        print_table_summary(summary, prefix="CLEAN")
        print(
            f"  resolved IDs: patient={table_cfg.get('patient_id')!r}; "
            f"event={table_cfg.get('event_id')!r}"
        )
        issue = issues[0]
        print(
            "  removed: "
            f"blank={issue['fully_blank_rows_removed']:,}, "
            f"exact duplicates={issue['exact_duplicate_rows_removed']:,}, "
            f"duplicate ED keys={issue['duplicate_patient_event_rows_removed']:,}"
        )

        # Quantify missingness after deterministic cleaning so these values
        # describe the exact table that will enter preprocessing.
        miss = column_missingness_table(cleaned, table_key, table_cfg, config)
        column_missingness_frames.append(miss)
        print_missingness_summary(
            miss,
            top_n=int((config.get("missingness", {}) or {}).get("print_top_n", 10)),
        )

        # Print table-specific integrity checks that would otherwise only be
        # visible by opening the QA CSV.  Zero values are useful positive QA.
        section(f"{table_key}: integrity / chronology findings")
        integrity_fields = [
            "negative_or_invalid_chronology_n",
            "msk_first_before_referral_n",
            "msk_last_before_first_n",
            "msk_last_before_referral_n",
            "discharge_before_admission_n",
            "episode_end_before_start_n",
            "episode_start_before_admission_n",
            "episode_end_after_discharge_n",
            "spells_linked_to_multiple_patients_n",
            "rows_with_repeated_episode_id_n",
            "age_sentinel_values_replaced",
            "age_implausible_values_replaced",
            "invalid_departure_values_replaced",
        ]
        for field in integrity_fields:
            if field in issue:
                metric(field, int(issue[field]))

    # Consolidate the six per-table missingness reports and compare columns shared
    # by the Wider MSK and Sports-linked source families.
    all_missingness = (
        pd.concat(column_missingness_frames, ignore_index=True)
        if column_missingness_frames
        else pd.DataFrame()
    )
    group_missingness = compare_group_missingness(all_missingness, config)
    all_missingness.to_csv(qa_dir / "02_column_missingness.csv", index=False)
    group_missingness.to_csv(
        qa_dir / "02_missingness_group_comparison.csv", index=False
    )
    print_group_missingness_comparison(group_missingness, config)

    # Keep action/issue and table-level summaries as separate files
    # so a reviewer can distinguish cleaning actions from missingness diagnostics.
    save_records(
        cleaning_records,
        qa_dir / "02_cleaning_actions_and_issues.csv",
    )
    save_records(
        cleaned_summaries,
        qa_dir / "02_cleaned_table_summary.csv",
    )

    # Build compact audit metrics for Ian/reviewer handover.  These are aggregate
    # counts only and can be inspected without opening patient-level files.
    total_rows_in = int(sum(x.get("rows_input", 0) for x in cleaning_records))
    total_rows_out = int(sum(x.get("rows_output", 0) for x in cleaning_records))
    total_removed = total_rows_in - total_rows_out
    critical_missing_fields = int(
        (
            all_missingness["classification"].eq("CRITICAL")
            & all_missingness["missing_n"].gt(0)
        ).sum()
    ) if not all_missingness.empty else 0
    flagged_group_missingness = int(
        group_missingness["flag_group_difference"].sum()
    ) if not group_missingness.empty else 0

    section("STAGE 02 KEY FINDINGS")
    metric("tables cleaned", f"{len(cleaned_summaries)}/{len(config['tables'])}")
    metric("rows entering cleaning across sources", f"{total_rows_in:,}")
    metric("rows after cleaning across sources", f"{total_rows_out:,}")
    metric("rows removed across sources", f"{total_removed:,}")
    metric("critical fields with missing values", critical_missing_fields)
    metric("Sports-vs-Wider missingness fields flagged", flagged_group_missingness)

    audit_dir = Path(config.get("_project_root", Path(config_path).resolve().parent.parent)) / "outputs" / "audit"
    summary_path = save_stage_summary(
        audit_dir,
        stage_key="cleaning",
        stage_code="02",
        title="Deterministic source cleaning + missingness audit",
        status="PASS" if critical_missing_fields == 0 else "REVIEW",
        key_findings={
            "tables_cleaned": len(cleaned_summaries),
            "rows_input_across_sources": total_rows_in,
            "rows_output_across_sources": total_rows_out,
            "rows_removed_across_sources": total_removed,
            "critical_missing_fields_n": critical_missing_fields,
            "sports_vs_wider_missingness_fields_flagged_n": flagged_group_missingness,
        },
        qa_files=[
            qa_dir / "02_identifier_resolution_qa.csv",
            qa_dir / "02_cleaning_actions_and_issues.csv",
            qa_dir / "02_cleaned_table_summary.csv",
            qa_dir / "02_column_missingness.csv",
            qa_dir / "02_missingness_group_comparison.csv",
        ],
        warnings=[
            "UNCLASSIFIED fields require review against BTH source documentation; missingness percentage alone does not establish its cause.",
            "No imputation is performed in Stage 02.",
        ],
        config_path=config_path,
    )
    stage_footer(
        stage_key="cleaning",
        audit_dir=audit_dir,
        summary_path=summary_path,
        qa_files=[qa_dir / "02_column_missingness.csv", qa_dir / "02_missingness_group_comparison.csv"],
        warnings=[
            "If any CRITICAL field is missing, review with the source owner before interpreting downstream analysis."
        ] if critical_missing_fields else [],
    )

    return cleaned_summaries
