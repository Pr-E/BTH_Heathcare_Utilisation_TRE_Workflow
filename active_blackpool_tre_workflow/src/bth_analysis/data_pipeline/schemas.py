"""Canonical minimum schemas for the six real Active Blackpool/BTH extracts.

The source mapping layer may rename real TRE columns into these canonical names
before validation.  These lists describe the fields needed by the translated
workflow; they are not intended to redefine the source system or imply that all
fields are clinically complete.

Why keep schemas in one module?
- reviewers can inspect the analytical source contract in one place;
- ingestion can fail early when required fields disappear;
- downstream modules do not need source-specific column aliases;
- refreshed extracts can be mapped in YAML rather than by rewriting analysis.
"""

# -----------------------------------------------------------------------------
# Wider-MSK pathway source: one or more pathway/referral rows per patient.
# -----------------------------------------------------------------------------
MSK_WIDER_COLUMNS = [
    "sha256_hash",  # Pseudonymised patient identifier candidate.
    "ReferralObservationId",  # Source referral observation identifier.
    "FirstMSKReferralDate",  # Earliest/source first referral timing field.
    "FirstMSKDate",  # Source-relative MSK date used as current analytical anchor.
    "FirstMSKSlotSession",  # Descriptive session metadata; not used as treatment.
    "LastMSKDate",  # Source last MSK date for pathway timing QA.
    "LastMSKSlotSession",
    "DischargedOrNot",
    "DateOfBirth",
    "DateOfDeath",
    "Age",
    "Sex",
    "EthnicityNationalCodeDesc",
    "PostcodeLAName",
    "Index_of_Multiple_Deprivation_IMD_Decile",
]

# -----------------------------------------------------------------------------
# Sports-linked MSK source: includes subsequent-referral fields not present in
# the wider-MSK table.  Their presence does NOT prove Active Blackpool treatment.
# -----------------------------------------------------------------------------
MSK_SPORTS_COLUMNS = [
    "sha256_hash_nhs_no",
    "ReferralObservationId",
    "FirstMSKReferralDate",
    "FirstMSKDate",
    "FirstMSKSlotSession",
    "LastMSKDate",
    "LastMSKSlotSession",
    "DischargedOrNot",
    "NewMSKReferralDate",
    "NewMSK_TargetOrganisationName",
    "NewMSKReferralObservationId",
    "NewOtherReferralDate",
    "NewOther_TargetOrganisationName",
    "NewOtherReferralObservationId",
    "DateOfBirth",
    "DateOfDeath",
    "Age",
    "Sex",
    "EthnicityNationalCodeDesc",
    "PostcodeLAName",
    "Index_of_Multiple_Deprivation_IMD_Decile",
]

# -----------------------------------------------------------------------------
# Inpatient source is episode-grain on ingestion.  Preprocessing later collapses
# episodes to SpellID/admission grain so hospital admissions are not inflated.
# -----------------------------------------------------------------------------
INPATIENT_COLUMNS = [
    "sha256_hash",
    "Cohort",
    "Inpatient_Timeframe",  # Source timeframe retained for reconciliation only.
    "EpisodeId",
    "SpellID",
    "AdmissionDate",
    "DischargeDate",
    "EpisodeStart",
    "Episodeend",
    "Specialty",
    "MethodOfAdmission",  # Used to classify emergency admissions after review.
]

# Seven diagnosis pairs are part of the current real extract contract.
for i in range(1, 8):
    INPATIENT_COLUMNS.extend([f"Diagnosis_{i}", f"Diagnosis_{i}_des"])

# Seven procedure pairs are retained for descriptive/extension work.
for i in range(1, 8):
    INPATIENT_COLUMNS.extend([f"Procedure_{i}", f"Procedure_{i}_des"])

INPATIENT_COLUMNS.extend(
    [
        "DateOfBirth",
        "DateOfDeath",
        "Age",
        "Sex",
        "EthnicityNationalCodeDesc",
        "PostcodeLAName",
        "Index_of_Multiple_Deprivation_IMD_Decile",
    ]
)

# -----------------------------------------------------------------------------
# ED source: attendance grain after event-identifier resolution and deduplication.
# The two SHA-256 fields are deliberately not trusted by name alone; Stage 02/03
# selects patient/event roles using cross-source overlap and uniqueness evidence.
# -----------------------------------------------------------------------------
ED_COLUMNS = [
    "sha256_hash_nhs_no",
    "sha256_hash_aeattendno",
    "AgeAttendance",
    "Gender",
    "ArrivalDate",
    "ArrivalHour",
    "ArrivalDateTime",
    "SourceAEReferral",
    "Emergency_Care_Chief_Complaint",
    "PresentingComplaint_explan",
]
ED_COLUMNS.extend([f"Diagnostic_Code_{i}" for i in range(1, 8)])
ED_COLUMNS.extend([f"Investigation_Code_{i}" for i in range(1, 8)])
ED_COLUMNS.extend([f"Treatment_Code_{i}" for i in range(1, 8)])
ED_COLUMNS.extend(
    [
        "Specialty",
        "MovementArea",
        "Discharge_Status",
        "Discharge_Status_DES",
        "Discharge_Destination",
        "Departure",
        "LOS",
        "ModeOfArrival",
        "ModeOfArrivalText",
        "DateOfBirth",
        "DateOfDeath",
        "EthnicityNationalCodeDesc",
        "PostcodeLAName",
        "Index_of_Multiple_Deprivation_IMD_Decile",
    ]
)

# Table key -> minimum canonical field list used by ingestion validation.
TABLE_SCHEMAS = {
    "msk_wider": MSK_WIDER_COLUMNS,
    "inpatient_wider": INPATIENT_COLUMNS,
    "ed_wider": ED_COLUMNS,
    "msk_sports": MSK_SPORTS_COLUMNS,
    "inpatient_sports": INPATIENT_COLUMNS,
    "ed_sports": ED_COLUMNS,
}

# Date groups are centralised so cleaning applies the same parsing contract to
# both source families and refreshed extracts.
MSK_DATE_COLUMNS = [
    "FirstMSKReferralDate",
    "FirstMSKDate",
    "LastMSKDate",
    "DateOfBirth",
    "DateOfDeath",
]
SPORTS_EXTRA_DATE_COLUMNS = ["NewMSKReferralDate", "NewOtherReferralDate"]
INPATIENT_DATE_COLUMNS = [
    "AdmissionDate",
    "DischargeDate",
    "EpisodeStart",
    "Episodeend",
    "DateOfBirth",
    "DateOfDeath",
]
ED_DATE_COLUMNS = [
    "ArrivalDate",
    "ArrivalDateTime",
    "Departure",
    "DateOfBirth",
    "DateOfDeath",
]
