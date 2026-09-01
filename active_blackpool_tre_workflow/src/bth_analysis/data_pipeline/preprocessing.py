"""Stage 03: preprocessing and source-grain harmonisation.

This stage converts the six cleaned source tables into canonical analytical
views while preserving source lineage. It:

- sequences MSK referral records and creates one source-relative pathway anchor
  per patient;
- verifies the configured cross-source patient identifiers against the matching
  MSK cohort;
- retains inpatient consultant episodes for QA while creating one row per
  admission/spell for downstream utilisation counting;
- retains one row per cleaned ED attendance;
- distinguishes patient-level MSK linkage from completeness of the pathway dates
  needed for timeframe derivation;
- reports patient-level linkage rates, anchor completeness and timeframe
  agreement using aggregate QA outputs.

No exposure assignment, final analytical-index construction, baseline/follow-up
windowing or outcome modelling occurs in this stage.
"""
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd

from .config import load_pipeline_config, output_dir
from .qa import save_records
from bth_analysis.audit import (
    dataframe_preview,
    metric,
    save_stage_summary,
    section,
    stage_footer,
    stage_header,
)
from .identifiers import resolve_identifier_plan, apply_identifier_choices_to_config


# Canonical cross-source reference relationships for this project.
# These mappings are stable analytical roles: hospital records must be checked
# against the MSK cohort from the same pathway population.
PROJECT_REFERENCE_COHORTS = {
    "inpatient_wider": "msk_wider",
    "ed_wider": "msk_wider",
    "inpatient_sports": "msk_sports",
    "ed_sports": "msk_sports",
}


def validate_reference_cohort_config(table_cfgs):
    """Validate the configured hospital-to-MSK reference relationships.

    Hospital patient identifiers are verified against the MSK cohort from the
    same pathway population. A missing or conflicting mapping is a configuration
    error and stops preprocessing before any linkage-dependent transformation.
    """
    effective = deepcopy(table_cfgs)
    records = []

    for table_key, expected_reference in PROJECT_REFERENCE_COHORTS.items():
        if table_key not in effective:
            raise KeyError(f"Required table {table_key!r} is missing from pipeline configuration.")

        configured_reference = effective[table_key].get("reference_cohort")
        if configured_reference != expected_reference:
            raise ValueError(
                f"{table_key}: reference_cohort must be {expected_reference!r}; "
                f"found {configured_reference!r}."
            )

        records.append({
            "table": table_key,
            "configured_reference_cohort": configured_reference,
            "effective_reference_cohort": expected_reference,
            "configuration_source": "CONFIG",
            "status": "OK",
        })

    return effective, pd.DataFrame(records)


def parse_datetime_columns(df, columns):
    """Re-parse canonical cleaned timestamps after CSV serialisation.

    Stage 02 already validated/parses source dates.  Stage 03 reads cleaned CSVs
    back from disk, so datetime dtype information must be restored before date
    arithmetic and pathway-timeframe derivation.
    """
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(
                out[col],
                errors="coerce",
                format="mixed",
                dayfirst=False,
            )
    return out


def days_between(later, earlier):
    """Return elapsed days between two parsed timestamp series as floating days."""
    return (later - earlier).dt.total_seconds() / 86400.0


def make_msk_referral_view(df, table_cfg, sports=False):
    """Create one canonical MSK referral/history view from a cleaned source table.

    PatientID is derived from the resolved source identifier, pathway dates are
    parsed, records are ordered within patient, and repeat/subsequent referral
    timing variables are derived.  No exposure assignment occurs here.
    """
    out = df.copy()
    pid = table_cfg["patient_id"]

    out = parse_datetime_columns(
        out,
        [
            "FirstMSKReferralDate",
            "FirstMSKDate",
            "LastMSKDate",
            "NewMSKReferralDate",
            "NewOtherReferralDate",
            "DateOfBirth",
            "DateOfDeath",
        ],
    )

    out.insert(0, "PatientID", out[pid].astype("string"))
    out["PopulationSource"] = table_cfg["population"]

    out = out.sort_values(
        ["PatientID", "FirstMSKReferralDate", "ReferralObservationId"],
        kind="mergesort",
    ).reset_index(drop=True)

    out["ReferralSequence"] = (
        out.groupby("PatientID", dropna=False).cumcount() + 1
    )
    out["IsRepeatReferral"] = (
        out["ReferralSequence"].gt(1).astype("Int64")
    )
    out["PreviousMSKReferralDate"] = (
        out.groupby("PatientID", dropna=False)["FirstMSKReferralDate"]
        .shift(1)
    )
    out["DaysSincePreviousReferral"] = days_between(
        out["FirstMSKReferralDate"],
        out["PreviousMSKReferralDate"],
    )

    out["ReferralToFirstMSKDays"] = days_between(
        out["FirstMSKDate"],
        out["FirstMSKReferralDate"],
    )
    out["FirstToLastMSKDays"] = days_between(
        out["LastMSKDate"],
        out["FirstMSKDate"],
    )
    out["ReferralToLastMSKDays"] = days_between(
        out["LastMSKDate"],
        out["FirstMSKReferralDate"],
    )

    if sports:
        out["HasNewMSKReferral"] = (
            out["NewMSKReferralDate"].notna().astype("Int64")
        )
        out["HasNewOtherReferral"] = (
            out["NewOtherReferralDate"].notna().astype("Int64")
        )

        conditions = [
            out["HasNewMSKReferral"].eq(0)
            & out["HasNewOtherReferral"].eq(0),
            out["HasNewMSKReferral"].eq(1)
            & out["HasNewOtherReferral"].eq(0),
            out["HasNewMSKReferral"].eq(0)
            & out["HasNewOtherReferral"].eq(1),
            out["HasNewMSKReferral"].eq(1)
            & out["HasNewOtherReferral"].eq(1),
        ]
        choices = [
            "Neither",
            "New MSK only",
            "New other only",
            "Both",
        ]
        out["SubsequentReferralPattern"] = np.select(
            conditions,
            choices,
            default=pd.NA,
        )

    return out


def make_pathway_anchor(referrals):
    """Create one deterministic source-relative pathway anchor per MSK patient.

    The earliest ordered referral row is retained intact. ``drop_duplicates`` is
    used instead of ``groupby.first`` so values from different referral rows
    cannot be combined when some fields are missing.

    The output is a preprocessing reference only; it is not the final analytical
    index and must not be interpreted as programme start.
    """
    ordered = referrals.sort_values(
        ["PatientID", "FirstMSKReferralDate", "ReferralObservationId"],
        kind="mergesort",
        na_position="last",
    )

    # Keep the complete earliest ordered row for each patient.
    anchor = ordered.drop_duplicates(subset=["PatientID"], keep="first").copy()

    keep = [
        "PatientID",
        "ReferralObservationId",
        "FirstMSKReferralDate",
        "FirstMSKDate",
        "LastMSKDate",
    ]
    anchor = anchor[keep].rename(
        columns={
            "ReferralObservationId": "AnchorReferralObservationId",
            "FirstMSKReferralDate": "AnchorFirstMSKReferralDate",
            "FirstMSKDate": "AnchorFirstMSKDate",
            "LastMSKDate": "AnchorLastMSKDate",
        }
    )

    # A non-missing marker allows hospital-source linkage to be distinguished
    # from completeness of the pathway dates required for timeframe derivation.
    anchor["MSKReferencePatientFlag"] = 1

    if anchor["PatientID"].duplicated().any():
        raise ValueError("Pathway anchor must contain exactly one row per PatientID.")

    return anchor

def derive_relative_timeframe(event_date, anchor_referral, anchor_last):
    """Classify an event as before/during/after the source-relative MSK pathway.

    This derived label is used to reconcile the workflow's date logic against any
    source-provided timeframe label; it is not the final baseline/follow-up window.
    """
    result = pd.Series(pd.NA, index=event_date.index, dtype="string")

    anchor_available = (
        event_date.notna()
        & anchor_referral.notna()
        & anchor_last.notna()
    )

    result.loc[
        anchor_available & (event_date < anchor_referral)
    ] = "Before MSK referral"

    result.loc[
        anchor_available
        & (event_date >= anchor_referral)
        & (event_date <= anchor_last)
    ] = "During MSK referral"

    result.loc[
        anchor_available & (event_date > anchor_last)
    ] = "After MSK referral"

    result.loc[
        event_date.notna()
        & (~anchor_referral.notna() | ~anchor_last.notna())
    ] = "No linked MSK pathway anchor"

    return result


def _aggregate_codes_by_spell(episode, group_cols, columns, output_col, count_col):
    """Collapse repeated diagnosis/procedure slots to unique codes per spell.

    The source stores codes across repeated slot columns. The function reshapes
    those slots once, removes repeated codes within each patient/spell and
    aggregates the remaining values in deterministic slot order.
    """
    existing = [col for col in columns if col in episode.columns]
    if not existing:
        return pd.DataFrame(columns=[*group_cols, output_col, count_col])

    long_codes = episode[group_cols + existing].melt(
        id_vars=group_cols,
        value_vars=existing,
        var_name="_CodeSlot",
        value_name="Code",
    )
    long_codes = long_codes.drop(columns="_CodeSlot").dropna(subset=["Code"])
    long_codes["Code"] = long_codes["Code"].astype(str)

    # Keep each code once per spell while preserving its first deterministic appearance.
    long_codes = long_codes.drop_duplicates(
        subset=[*group_cols, "Code"],
        keep="first",
    )

    return (
        long_codes.groupby(group_cols, dropna=False, sort=False)["Code"]
        .agg(["size", "|".join])
        .rename(columns={"size": count_col, "join": output_col})
        .reset_index()
    )


def make_inpatient_views(df, table_cfg, anchor):
    """Build episode-level and spell/admission-level inpatient views.

    The episode table preserves cleaned consultant-episode detail for QA and
    code aggregation. The spell table contains one row per patient/admission and
    is the downstream counting grain for inpatient utilisation.
    """
    episode = df.copy()
    pid = table_cfg["patient_id"]

    episode = parse_datetime_columns(
        episode,
        [
            "AdmissionDate",
            "DischargeDate",
            "EpisodeStart",
            "Episodeend",
            "DateOfBirth",
            "DateOfDeath",
        ],
    )

    episode.insert(0, "PatientID", episode[pid].astype("string"))
    episode["PopulationSource"] = table_cfg["population"]

    # Pathway anchors are preprocessing references only; they are not final analysis indices.
    episode = episode.merge(
        anchor,
        on="PatientID",
        how="left",
        validate="many_to_one",
    )

    # Linkage and anchor completeness are separate concepts.
    # MSKPatientLinkedFlag asks whether the patient exists in the matching MSK
    # cohort. MSKPathwayAnchorAvailable additionally requires the dates needed
    # to classify the healthcare event relative to the source pathway.
    episode["MSKPatientLinkedFlag"] = (
        episode["MSKReferencePatientFlag"].fillna(0).astype("Int64")
    )
    episode["MSKPathwayAnchorAvailable"] = (
        episode["MSKPatientLinkedFlag"].eq(1)
        & episode["AnchorFirstMSKReferralDate"].notna()
        & episode["AnchorLastMSKDate"].notna()
    ).astype("Int64")

    episode["DerivedInpatientTimeframe"] = derive_relative_timeframe(
        episode["AdmissionDate"],
        episode["AnchorFirstMSKReferralDate"],
        episode["AnchorLastMSKDate"],
    )

    source_tf = episode["Inpatient_Timeframe"].astype("string")
    derived_tf = episode["DerivedInpatientTimeframe"].astype("string")
    comparable = (
        episode["MSKPathwayAnchorAvailable"].eq(1)
        & source_tf.notna()
        & derived_tf.notna()
    )
    agreement = pd.Series(pd.NA, index=episode.index, dtype="Int64")
    agreement.loc[comparable] = (
        source_tf.loc[comparable].eq(derived_tf.loc[comparable]).astype(int)
    )
    episode["TimeframeAgreementFlag"] = agreement

    episode["EpisodeDurationHours"] = (
        (episode["Episodeend"] - episode["EpisodeStart"])
        .dt.total_seconds()
        / 3600.0
    )

    group_cols = ["PatientID", "SpellID"]
    grouped = episode.groupby(group_cols, dropna=False, sort=False)

    # Aggregate fields whose spell-level rule is min/max/first/nunique in one grouped pass.
    spell = grouped.agg(
        PopulationSource=("PopulationSource", "first"),
        AdmissionDate=("AdmissionDate", "min"),
        DischargeDate=("DischargeDate", "max"),
        EpisodeStart=("EpisodeStart", "min"),
        EpisodeEnd=("Episodeend", "max"),
        NumberOfEpisodes=("PopulationSource", "size"),
        SourceInpatientTimeframe=("Inpatient_Timeframe", "first"),
        DerivedInpatientTimeframe=("DerivedInpatientTimeframe", "first"),
        MSKPatientLinkedFlag=("MSKPatientLinkedFlag", "max"),
        MSKPathwayAnchorAvailable=("MSKPathwayAnchorAvailable", "max"),
        TimeframeAgreementFlag=("TimeframeAgreementFlag", "first"),
        MethodOfAdmission=("MethodOfAdmission", "first"),
        FirstSpecialty=("Specialty", "first"),
        UniqueSpecialties=("Specialty", "nunique"),
        DateOfBirth=("DateOfBirth", "first"),
        DateOfDeath=("DateOfDeath", "first"),
        Age=("Age", "first"),
        Sex=("Sex", "first"),
        EthnicityNationalCodeDesc=("EthnicityNationalCodeDesc", "first"),
        PostcodeLAName=("PostcodeLAName", "first"),
        Index_of_Multiple_Deprivation_IMD_Decile=(
            "Index_of_Multiple_Deprivation_IMD_Decile",
            "first",
        ),
    ).reset_index()

    # Spell LOS is derived after aggregation so each admission contributes one duration.
    spell["SpellLOSDays"] = (
        (spell["DischargeDate"] - spell["AdmissionDate"])
        .dt.total_seconds()
        / 86400.0
    )

    diagnosis_cols = [f"Diagnosis_{i}" for i in range(1, 8)]
    procedure_cols = [f"Procedure_{i}" for i in range(1, 8)]

    diagnosis = _aggregate_codes_by_spell(
        episode,
        group_cols,
        diagnosis_cols,
        "DiagnosisCodes",
        "DiagnosisCodeCount",
    )
    procedure = _aggregate_codes_by_spell(
        episode,
        group_cols,
        procedure_cols,
        "ProcedureCodes",
        "ProcedureCodeCount",
    )

    spell = spell.merge(diagnosis, on=group_cols, how="left", validate="one_to_one")
    spell = spell.merge(procedure, on=group_cols, how="left", validate="one_to_one")

    # Use an empty code string and zero count when a spell has no recorded codes.
    for code_col, count_col in [
        ("DiagnosisCodes", "DiagnosisCodeCount"),
        ("ProcedureCodes", "ProcedureCodeCount"),
    ]:
        spell[code_col] = spell[code_col].fillna("")
        spell[count_col] = spell[count_col].fillna(0).astype(int)

    # Keep the established spell schema/order so downstream stages remain unchanged.
    spell = spell[[
        "PatientID",
        "SpellID",
        "PopulationSource",
        "AdmissionDate",
        "DischargeDate",
        "SpellLOSDays",
        "EpisodeStart",
        "EpisodeEnd",
        "NumberOfEpisodes",
        "SourceInpatientTimeframe",
        "DerivedInpatientTimeframe",
        "MSKPatientLinkedFlag",
        "MSKPathwayAnchorAvailable",
        "TimeframeAgreementFlag",
        "MethodOfAdmission",
        "FirstSpecialty",
        "UniqueSpecialties",
        "DiagnosisCodes",
        "ProcedureCodes",
        "DateOfBirth",
        "DateOfDeath",
        "Age",
        "Sex",
        "EthnicityNationalCodeDesc",
        "PostcodeLAName",
        "Index_of_Multiple_Deprivation_IMD_Decile",
        "DiagnosisCodeCount",
        "ProcedureCodeCount",
    ]]

    return episode, spell

def make_ed_view(df, table_cfg, anchor):
    """Create one row per cleaned ED attendance with canonical IDs and timing QA."""
    out = df.copy()
    pid = table_cfg["patient_id"]
    event_id = table_cfg["event_id"]

    out = parse_datetime_columns(
        out,
        [
            "ArrivalDate",
            "ArrivalDateTime",
            "Departure",
            "DateOfBirth",
            "DateOfDeath",
        ],
    )

    out.insert(0, "PatientID", out[pid].astype("string"))
    out.insert(1, "EDAttendanceID", out[event_id].astype("string"))
    out["PopulationSource"] = table_cfg["population"]

    out["LOSCalculatedMinutes"] = (
        (out["Departure"] - out["ArrivalDateTime"])
        .dt.total_seconds()
        / 60.0
    )

    out["LOSDifferenceMinutes"] = (
        pd.to_numeric(out["LOS"], errors="coerce")
        - out["LOSCalculatedMinutes"]
    )

    out["ArrivalYear"] = out["ArrivalDateTime"].dt.year.astype("Int64")
    out["ArrivalMonth"] = out["ArrivalDateTime"].dt.to_period("M").astype("string")
    out["ArrivalWeekday"] = out["ArrivalDateTime"].dt.day_name()
    out["IsWeekend"] = (
        out["ArrivalDateTime"].dt.dayofweek.ge(5).astype("Int64")
    )

    out = out.merge(
        anchor,
        on="PatientID",
        how="left",
        validate="many_to_one",
    )

    # Keep patient linkage separate from pathway-date completeness.
    out["MSKPatientLinkedFlag"] = (
        out["MSKReferencePatientFlag"].fillna(0).astype("Int64")
    )
    out["MSKPathwayAnchorAvailable"] = (
        out["MSKPatientLinkedFlag"].eq(1)
        & out["AnchorFirstMSKReferralDate"].notna()
        & out["AnchorLastMSKDate"].notna()
    ).astype("Int64")

    out["DerivedEDTimeframe"] = derive_relative_timeframe(
        out["ArrivalDateTime"],
        out["AnchorFirstMSKReferralDate"],
        out["AnchorLastMSKDate"],
    )

    return out


def processed_summary(name, df):
    """Return aggregate dimensions, patient linkage and timeframe QA metrics."""
    out = {
        "dataset": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
    }

    if "PatientID" in df.columns:
        total_patients = int(df["PatientID"].nunique(dropna=True))
        out["unique_patients"] = total_patients
        out["missing_patient_ids"] = int(df["PatientID"].isna().sum())

        if "MSKPatientLinkedFlag" in df.columns:
            linked_mask = df["MSKPatientLinkedFlag"].fillna(0).eq(1)
            linked_patients = int(
                df.loc[linked_mask, "PatientID"].nunique(dropna=True)
            )
            out["linked_patients"] = linked_patients
            out["unlinked_patients"] = max(total_patients - linked_patients, 0)
            out["patient_linkage_pct"] = (
                round(100.0 * linked_patients / total_patients, 2)
                if total_patients
                else np.nan
            )
            out["linked_rows"] = int(linked_mask.sum())
            out["linked_rows_pct"] = (
                round(100.0 * linked_mask.mean(), 2) if len(df) else np.nan
            )

        if "MSKPathwayAnchorAvailable" in df.columns:
            anchor_mask = df["MSKPathwayAnchorAvailable"].fillna(0).eq(1)
            anchor_patients = int(
                df.loc[anchor_mask, "PatientID"].nunique(dropna=True)
            )
            out["anchor_available_patients"] = anchor_patients
            out["anchor_available_rows"] = int(anchor_mask.sum())
            out["anchor_available_rows_pct"] = (
                round(100.0 * anchor_mask.mean(), 2) if len(df) else np.nan
            )
            out["anchor_available_patient_pct"] = (
                round(100.0 * anchor_patients / total_patients, 2)
                if total_patients
                else np.nan
            )

            linked_patients = out.get("linked_patients")
            out["anchor_complete_pct_of_linked_patients"] = (
                round(100.0 * anchor_patients / linked_patients, 2)
                if linked_patients
                else np.nan
            )

    if "SpellID" in df.columns:
        out["unique_spells"] = int(df["SpellID"].nunique(dropna=True))

    if "EDAttendanceID" in df.columns:
        out["unique_ed_attendances"] = int(
            df["EDAttendanceID"].nunique(dropna=True)
        )

    if "TimeframeAgreementFlag" in df.columns:
        valid = df["TimeframeAgreementFlag"].dropna()
        out["timeframe_comparable_rows"] = int(len(valid))
        out["timeframe_agreement_pct"] = (
            round(float(valid.mean() * 100), 2)
            if len(valid)
            else np.nan
        )

    return out


def patient_linkage_summary(name, df):
    """Summarise patient-level linkage and anchor completeness for one event view."""
    if "PatientID" not in df.columns or "MSKPatientLinkedFlag" not in df.columns:
        raise ValueError(f"{name}: patient-level linkage fields are unavailable.")

    total_patients = int(df["PatientID"].nunique(dropna=True))
    linked = df["MSKPatientLinkedFlag"].fillna(0).eq(1)
    complete_anchor = df["MSKPathwayAnchorAvailable"].fillna(0).eq(1)

    linked_patients = int(df.loc[linked, "PatientID"].nunique(dropna=True))
    anchor_patients = int(
        df.loc[complete_anchor, "PatientID"].nunique(dropna=True)
    )

    return {
        "dataset": name,
        "total_patients": total_patients,
        "linked_patients": linked_patients,
        "unlinked_patients": max(total_patients - linked_patients, 0),
        "patient_linkage_pct": (
            round(100.0 * linked_patients / total_patients, 2)
            if total_patients
            else np.nan
        ),
        "complete_anchor_patients": anchor_patients,
        "anchor_incomplete_among_linked_patients": max(
            linked_patients - anchor_patients, 0
        ),
        "anchor_complete_pct_of_linked_patients": (
            round(100.0 * anchor_patients / linked_patients, 2)
            if linked_patients
            else np.nan
        ),
    }

def save_processed(df, path):
    """Persist a processed analytical-grain table using canonical datetime text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )


def run_preprocessing(config_path="config/pipeline_tre.yaml"):
    """Harmonise cleaned source grain, derive pathway views and collapse inpatient episodes to spells."""
    config = load_pipeline_config(config_path)
    in_dir = output_dir(config, "cleaned_dir")
    out_dir = output_dir(config, "processed_dir")
    qa_dir = output_dir(config, "qa_dir")

    stage_header(
        "03",
        "PREPROCESSING + SOURCE-GRAIN HARMONISATION",
        purpose=(
            "Convert cleaned source tables into canonical analytical views: MSK referral sequences, "
            "source-relative pathway anchors, inpatient episode/spell views and ED attendances. "
            "Exposure assignment and the final analytical index remain deliberately deferred."
        ),
        inputs=[in_dir],
        outputs=[out_dir, qa_dir],
    )

    if not in_dir.exists():
        raise FileNotFoundError(
            f"Cleaned directory not found: {in_dir}. Run Stage 02 first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    tables = {}
    for table_key, cfg in config["tables"].items():
        path = in_dir / cfg.get("canonical_filename", f"{Path(cfg['filename']).stem}.csv")
        if not path.exists():
            raise FileNotFoundError(
                f"{table_key}: cleaned file not found: {path}"
            )
        tables[table_key] = pd.read_csv(path, low_memory=False)

    # Validate the hospital-to-MSK reference relationships before identifier
    # resolution so cross-source patient-ID verification cannot be skipped.
    effective_table_cfgs, reference_cfg_audit = validate_reference_cohort_config(
        config["tables"]
    )
    reference_cfg_path = qa_dir / "03_reference_cohort_configuration.csv"
    reference_cfg_audit.to_csv(reference_cfg_path, index=False)

    section("REFERENCE COHORT CONFIGURATION")
    dataframe_preview(
        reference_cfg_audit,
        columns=[
            "table",
            "configured_reference_cohort",
            "effective_reference_cohort",
            "configuration_source",
            "status",
        ],
        max_rows=10,
    )

    # Re-resolve candidate patient/event identifiers against the matching MSK
    # cohort after cleaning. This checks linkage semantics using observed overlap
    # rather than trusting potentially misleading hash-column names.
    id_cfg = config.get("identifier_resolution", {}) or {}
    choices, id_audit = resolve_identifier_plan(
        tables,
        effective_table_cfgs,
        fail_if_no_reference_overlap=bool(
            id_cfg.get("fail_if_no_reference_overlap", True)
        ),
        fail_if_ambiguous=bool(id_cfg.get("fail_if_ambiguous", True)),
    )
    table_cfgs = apply_identifier_choices_to_config(
        effective_table_cfgs, choices
    )
    id_audit_path = qa_dir / "03_identifier_resolution_qa.csv"
    id_audit.to_csv(id_audit_path, index=False)

    # Patient and event identifiers have different QA criteria. Patient IDs are
    # verified by overlap with the matching MSK cohort; event IDs are assessed
    # separately for non-missingness and uniqueness.
    section("IDENTIFIER RE-VERIFICATION")
    selected_ids = (
        id_audit[id_audit.get("selected", 0).eq(1)].copy()
        if not id_audit.empty
        else pd.DataFrame()
    )
    selected_patient = selected_ids[
        selected_ids.get("identifier_role", pd.Series(dtype="string")).eq("patient")
        & selected_ids.get("reference_cohort", pd.Series(dtype="string")).notna()
    ].copy() if not selected_ids.empty else pd.DataFrame()
    selected_event = selected_ids[
        selected_ids.get("identifier_role", pd.Series(dtype="string")).eq("event")
    ].copy() if not selected_ids.empty else pd.DataFrame()

    if not selected_patient.empty:
        print("Selected hospital patient identifiers:")
        dataframe_preview(
            selected_patient,
            columns=[
                "table", "candidate", "reference_cohort", "unique_n",
                "reference_overlap_n",
                "reference_overlap_pct_of_candidate_unique",
                "reference_overlap_pct_of_reference",
            ],
            max_rows=10,
        )
    if not selected_event.empty:
        print("\nSelected event identifiers:")
        dataframe_preview(
            selected_event,
            columns=["table", "candidate", "non_missing_n", "unique_n", "uniqueness_ratio"],
            max_rows=10,
        )

    wider_referrals = make_msk_referral_view(
        tables["msk_wider"],
        table_cfgs["msk_wider"],
        sports=False,
    )
    sports_referrals = make_msk_referral_view(
        tables["msk_sports"],
        table_cfgs["msk_sports"],
        sports=True,
    )

    wider_anchor = make_pathway_anchor(wider_referrals)
    sports_anchor = make_pathway_anchor(sports_referrals)

    wider_ip_episode, wider_ip_spell = make_inpatient_views(
        tables["inpatient_wider"],
        table_cfgs["inpatient_wider"],
        wider_anchor,
    )
    sports_ip_episode, sports_ip_spell = make_inpatient_views(
        tables["inpatient_sports"],
        table_cfgs["inpatient_sports"],
        sports_anchor,
    )

    wider_ed = make_ed_view(
        tables["ed_wider"],
        table_cfgs["ed_wider"],
        wider_anchor,
    )
    sports_ed = make_ed_view(
        tables["ed_sports"],
        table_cfgs["ed_sports"],
        sports_anchor,
    )

    outputs = {
        "msk_wider_referrals": (
            wider_referrals,
            out_dir / "msk_wider_referrals.csv",
        ),
        "msk_sports_referrals": (
            sports_referrals,
            out_dir / "msk_sports_referrals.csv",
        ),
        "msk_wider_pathway_anchor": (
            wider_anchor,
            out_dir / "msk_wider_pathway_anchor.csv",
        ),
        "msk_sports_pathway_anchor": (
            sports_anchor,
            out_dir / "msk_sports_pathway_anchor.csv",
        ),
        "inpatient_wider_episodes": (
            wider_ip_episode,
            out_dir / "inpatient_wider_episodes.csv",
        ),
        "inpatient_wider_spells": (
            wider_ip_spell,
            out_dir / "inpatient_wider_spells.csv",
        ),
        "inpatient_sports_episodes": (
            sports_ip_episode,
            out_dir / "inpatient_sports_episodes.csv",
        ),
        "inpatient_sports_spells": (
            sports_ip_spell,
            out_dir / "inpatient_sports_spells.csv",
        ),
        "ed_wider_attendances": (
            wider_ed,
            out_dir / "ed_wider_attendances.csv",
        ),
        "ed_sports_attendances": (
            sports_ed,
            out_dir / "ed_sports_attendances.csv",
        ),
    }

    qa = []
    for name, (df, path) in outputs.items():
        save_processed(df, path)
        summary = processed_summary(name, df)
        qa.append(summary)
        print(
            f"{name}: rows={summary['rows']:,}, "
            f"cols={summary['columns']}, "
            f"patients={summary.get('unique_patients', 'n/a')}"
        )

    save_records(
        qa,
        qa_dir / "03_preprocessing_summary.csv",
    )

    # Report linkage at patient level using the downstream event grains.
    # This is deliberately separate from anchor-date completeness: a patient can
    # link correctly to the MSK cohort but still lack one of the dates needed for
    # source-relative timeframe derivation.
    linkage_qa = pd.DataFrame([
        patient_linkage_summary("inpatient_wider_spells", wider_ip_spell),
        patient_linkage_summary("inpatient_sports_spells", sports_ip_spell),
        patient_linkage_summary("ed_wider_attendances", wider_ed),
        patient_linkage_summary("ed_sports_attendances", sports_ed),
    ])
    linkage_qa_path = qa_dir / "03_patient_linkage_coverage.csv"
    linkage_qa.to_csv(linkage_qa_path, index=False)

    section("PATIENT-LEVEL MSK LINKAGE / PATHWAY-ANCHOR COVERAGE")
    dataframe_preview(
        linkage_qa,
        columns=[
            "dataset",
            "total_patients",
            "linked_patients",
            "unlinked_patients",
            "patient_linkage_pct",
            "complete_anchor_patients",
            "anchor_incomplete_among_linked_patients",
            "anchor_complete_pct_of_linked_patients",
        ],
        max_rows=10,
    )

    # Explicitly quantify the inpatient episode-to-spell collapse.  This is one
    # of the most important source-grain decisions because downstream inpatient
    # outcomes count admissions/spells rather than consultant episode rows.
    wider_episode_n = len(wider_ip_episode)
    wider_spell_n = len(wider_ip_spell)
    sports_episode_n = len(sports_ip_episode)
    sports_spell_n = len(sports_ip_spell)
    wider_collapse_pct = (100.0 * (wider_episode_n - wider_spell_n) / wider_episode_n) if wider_episode_n else 0.0
    sports_collapse_pct = (100.0 * (sports_episode_n - sports_spell_n) / sports_episode_n) if sports_episode_n else 0.0

    section("STAGE 03 KEY FINDINGS")
    metric("processed datasets created", len(outputs))
    metric("Wider inpatient episode rows", f"{wider_episode_n:,}")
    metric("Wider inpatient spell rows", f"{wider_spell_n:,}")
    metric("Wider episode rows collapsed", f"{wider_collapse_pct:.2f}%")
    metric("Sports inpatient episode rows", f"{sports_episode_n:,}")
    metric("Sports inpatient spell rows", f"{sports_spell_n:,}")
    metric("Sports episode rows collapsed", f"{sports_collapse_pct:.2f}%")

    # Surface the patient-level linkage result directly in the terminal.
    linkage_index = linkage_qa.set_index("dataset")
    for dataset_name in [
        "inpatient_wider_spells",
        "inpatient_sports_spells",
        "ed_wider_attendances",
        "ed_sports_attendances",
    ]:
        row = linkage_index.loc[dataset_name]
        metric(
            f"{dataset_name} patient linkage",
            (
                f"{int(row['linked_patients']):,}/{int(row['total_patients']):,} "
                f"({float(row['patient_linkage_pct']):.2f}%)"
            ),
        )

    # The detailed processed-table summary remains available for audit, but the
    # log now distinguishes linkage from pathway-date completeness.
    qa_df = pd.DataFrame(qa)
    dataframe_preview(
        qa_df,
        columns=[
            "dataset",
            "rows",
            "unique_patients",
            "unique_spells",
            "unique_ed_attendances",
            "linked_patients",
            "patient_linkage_pct",
            "anchor_available_patients",
            "anchor_complete_pct_of_linked_patients",
            "timeframe_comparable_rows",
            "timeframe_agreement_pct",
        ],
        max_rows=15,
    )

    audit_dir = config["_project_root"] / "outputs" / "audit"
    summary_path = save_stage_summary(
        audit_dir,
        stage_key="preprocessing",
        stage_code="03",
        title="Preprocessing + source-grain harmonisation",
        status="PASS",
        key_findings={
            "processed_datasets_created": len(outputs),
            "wider_inpatient_episode_rows": wider_episode_n,
            "wider_inpatient_spell_rows": wider_spell_n,
            "wider_episode_to_spell_row_reduction_pct": wider_collapse_pct,
            "sports_inpatient_episode_rows": sports_episode_n,
            "sports_inpatient_spell_rows": sports_spell_n,
            "sports_episode_to_spell_row_reduction_pct": sports_collapse_pct,
            "patient_linkage": {
                row["dataset"]: {
                    "total_patients": int(row["total_patients"]),
                    "linked_patients": int(row["linked_patients"]),
                    "unlinked_patients": int(row["unlinked_patients"]),
                    "patient_linkage_pct": (
                        None if pd.isna(row["patient_linkage_pct"])
                        else float(row["patient_linkage_pct"])
                    ),
                    "complete_anchor_patients": int(row["complete_anchor_patients"]),
                    "anchor_complete_pct_of_linked_patients": (
                        None
                        if pd.isna(row["anchor_complete_pct_of_linked_patients"])
                        else float(row["anchor_complete_pct_of_linked_patients"])
                    ),
                }
                for row in linkage_qa.to_dict(orient="records")
            },
        },
        qa_files=[
            reference_cfg_path,
            id_audit_path,
            qa_dir / "03_preprocessing_summary.csv",
            linkage_qa_path,
        ],
        warnings=[
            "Pathway anchors created here are source-relative preprocessing anchors, not the final analytical index.",
            "Patient linkage and pathway-anchor date completeness are reported separately.",
            "Inpatient outcome counting uses spell/admission grain; episode rows are retained only for QA/detail.",
        ],
        config_path=config_path,
    )
    stage_footer(
        stage_key="preprocessing",
        audit_dir=audit_dir,
        summary_path=summary_path,
        qa_files=[
            reference_cfg_path,
            id_audit_path,
            qa_dir / "03_preprocessing_summary.csv",
            linkage_qa_path,
        ],
    )

    return outputs