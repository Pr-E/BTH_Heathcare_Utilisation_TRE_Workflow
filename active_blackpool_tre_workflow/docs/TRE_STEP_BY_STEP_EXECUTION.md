# TRE step-by-step execution guide

This guide is the operational sequence for the real-data workflow. Run the
stages individually on the first execution so each aggregate QA gate can be
reviewed before the next transformation.

## A. Environment verification

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m compileall -q src scripts tests
python -m pytest -q
```

Do not proceed to patient-level processing if compilation or toy-data tests fail.

## B. Configuration freeze

### 1. Confirm the approved source directory

`config/pipeline_tre.yaml` should contain:

```yaml
data_source:
  mode: tre
  tre_dir: /project/readonly
```

Change this only if the TRE administrator provides a different approved read-only
source mount.

### 2. Enter approved extract coverage dates

In `config/workflow_tre.yaml`, replace the null `study_start_date` and
`study_end_date` values only from approved BTH/TRE source documentation.

### 3. Confirm analytical index semantics

Set `analytical_index_semantics_confirmed_for_workflow: true` only after the
meaning of the configured index field has been reviewed. Keep
`index_is_programme_start: false` unless genuine programme-start data establish
otherwise.

---

## Stage 00 — Preflight

```bash
python scripts/run_00_preflight.py
```

Checks source files, required fields, study-window configuration, identifier
candidates, cohort/index semantic gates and propensity-variable timing.

Inspect `outputs/qa/00_tre_preflight.csv` and the Stage 00 audit summary.

## Stage 01 — Ingestion

```bash
python scripts/run_01_ingestion.py
```

Reads the six approved extracts, applies canonical column mapping, validates the
minimum source contract and writes canonical ingested copies without analytical
cleaning or cohort derivation.

Inspect `outputs/qa/01_ingestion_summary.csv`.

## Stage 02 — Deterministic cleaning + missingness audit

```bash
python scripts/run_02_cleaning.py
```

Key checks:

- patient identifiers verified against the matching MSK cohort;
- ED event identifiers assessed separately for uniqueness;
- blank/exact duplicate records handled under explicit rules;
- dates/numerics standardised;
- chronology/integrity issues counted;
- missing n/% reported by column;
- critical, expected/conditional and unclassified missingness distinguished;
- Sports-linked versus Wider MSK missingness differences reported.

Inspect:

- `02_identifier_resolution_qa.csv`
- `02_cleaning_actions_and_issues.csv`
- `02_cleaned_table_summary.csv`
- `02_column_missingness.csv`
- `02_missingness_group_comparison.csv`

Critical missingness or ambiguous identifiers require review before Stage 03.
No analytical imputation occurs in Stage 02.

## Stage 03 — Preprocessing + source-grain harmonisation

```bash
python scripts/run_03_preprocessing.py
```

Key transformations:

- sequence MSK referrals and retain one intact earliest source-relative anchor row per patient;
- reverify hospital patient identifiers against the matching MSK cohort;
- distinguish patient linkage from pathway-date completeness;
- retain inpatient consultant episodes for QA and create one admission/spell row for utilisation counting;
- retain one row per ED attendance;
- derive source-relative timeframe QA where the required dates are present.

Key terminal findings include patient linkage n/%, unlinked patients, anchor
completeness, episode-to-spell reduction and timeframe agreement.

Inspect:

- `03_reference_cohort_configuration.csv`
- `03_identifier_resolution_qa.csv`
- `03_patient_linkage_coverage.csv`
- `03_preprocessing_summary.csv`

## Stage 04 — Patient spine + linkage QA

```bash
python scripts/run_04_linkage.py
```

Builds the pseudonymised patient spine, source-presence flags, demographic values
with field-level source provenance, pathway anchors and working pathway-group
labels. It also compares hospital-source coverage within the working groups.

Inspect:

- `04_linkage_overlap_qa.csv`
- `04_group_source_coverage.csv`
- `04_demographic_source_qa.csv`
- `data/04_analysis/patient_spine.csv` (TRE-internal patient-level file)

## Stage 05 — Cohort + analytical index

```bash
python scripts/run_05_cohort.py
```

Constructs the comparative population, derives the configured source-relative
analytical index, applies baseline/follow-up observation rules and records every
eligibility transition. Review all cohort denominators before proceeding.

## Stage 06 — Healthcare-utilisation outcomes

```bash
python scripts/run_06_outcomes.py
```

Counts ED attendances, inpatient admissions/spells, emergency inpatient
admissions and total hospital utilisation within the defined windows. Uses
observed person-time for rates and retains partial follow-up where configured.

## Stage 07 — Descriptive analysis

```bash
python scripts/run_07_descriptive.py
```

Produces cohort profiles, baseline characteristics, crude utilisation rates,
pathway timing, data-quality summaries and pre-adjustment balance diagnostics.

## Stage 08 — Propensity + comparative analysis

```bash
python scripts/run_08_comparative.py
```

Recomputes structural positivity, common support, ATT weights, SMD balance and
1:3 PSM from the real data. Primary comparative models are blocked when the
configured ATT balance gate fails.

## Stage 09 — Exploratory clustering

```bash
python scripts/run_09_clustering.py
```

Builds baseline utilisation phenotypes using only the three configured baseline
rate variables. Candidate K solutions, cluster sizes and stability are recomputed
on the real data before the report-facing solution is selected.

## Stage 10 — Release-output pre-screen

```bash
python scripts/run_10_release_audit.py
```

Flags obvious patient-level/internal-only filenames and possible non-zero small
aggregate cells before formal TRE disclosure review. This is a pre-screen only;
it does not authorise egress.

---

## Reviewer summary

After any partial or complete run:

```bash
python scripts/review_audit_summary.py
```

This consolidates the patient-safe stage summaries under `outputs/audit/`.
