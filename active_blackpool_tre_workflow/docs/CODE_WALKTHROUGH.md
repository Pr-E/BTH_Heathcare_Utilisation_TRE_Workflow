# Code walkthrough for TRE handover

## `src/bth_analysis/data_pipeline/ingestion.py`

Reads each approved TRE extract, applies `raw -> canonical` column mappings, validates the canonical schema and writes schema-aligned copies. No data cleaning or derivation is performed here.

Key functions:
- `run_ingestion()` — stage entry point.
- `validate_schema()` — stops on missing canonical fields.
- `apply_column_mapping()` — source alias resolution.

## `identifiers.py`

Resolves the real TRE pseudonymised identifiers before linkage-sensitive operations. MSK cohorts provide the reference patient hash; inpatient/ED candidate patient IDs are scored by overlap with the matching MSK cohort, while ED event IDs are selected separately using uniqueness diagnostics. Ambiguous or zero-overlap resolution blocks the workflow. Only aggregate QA counts and selected column names are written.

## `cleaning.py`

Loads all six ingested tables, resolves identifiers across source families, then performs conservative source-level cleaning. It parses dates/numerics, removes fully blank/exact duplicate rows when configured and records chronology/duplicate-key issues.

No cohort, exposure, index or outcome variables are created here.

## `preprocessing.py`

Re-verifies identifier resolution on the cleaned tables, then creates stable analytical grains:
- one row per MSK referral;
- one pathway anchor per patient/source family;
- inpatient episode view;
- inpatient spell/admission view;
- one row per ED attendance.

This is where inpatient episodes are collapsed to spell level so downstream utilisation counts represent admissions rather than episode rows.

## `linkage.py`

Builds the canonical patient spine by pseudonymised patient ID, creates source-presence flags, resolves demographics through a documented source-priority rule, and creates the working Sports-linked/Wider MSK candidate labels.

## `cohort.py`

Freezes:
- the analytical group;
- index date;
- baseline/follow-up windows;
- censoring;
- age at index;
- eligibility and cohort-flow flags.

The current index is source-relative `FirstMSKDate`, not programme start.

## `outcomes.py`

Builds a unified event ledger from ED attendances and inpatient spells, assigns each event to baseline/follow-up, aggregates patient-level counts and derives person-time rates.

## `descriptive.py`

Produces the descriptive/EDA layer: cohort flow, missingness, baseline characteristics, source coverage, utilisation summaries, crude pre/post rates, SMDs and diagnostic figures.

## `propensity.py`

This is the measured-confounding **design** layer.

1. Validate that covariates are pre-index.
2. Derive `IndexYear` if requested.
3. Restrict structurally unsupported categorical levels.
4. Encode/impute baseline covariates.
5. Fit binary logistic regression for Sports-linked pathway membership.
6. Calculate propensity scores.
7. Determine empirical common support.
8. Create ATT weights.
9. Create 1:3 PSM matched sets.
10. Calculate SMDs before/after adjustment.
11. Output weight/overlap/ESS diagnostics.

## `comparative.py`

This is the main outcome layer.

### Crude stage
Calculates baseline and follow-up rates per 100 person-years for both groups.

### Primary adjusted stage
Stacks baseline and follow-up as repeated patient-period observations and fits:

```text
Group + Post + Group×Post + log(person-time) offset
```

using Poisson GEE with patient ID as the repeated-measure cluster.

The exponentiated `Group×Post` coefficient is the rate ratio of rate ratios.

### Sensitivities
- Negative Binomial GEE;
- ATT follow-up-only comparison;
- PSM follow-up and comparative pre/post models.

## `clustering.py`

Secondary exploratory baseline-utilisation phenotyping.

Cluster inputs are restricted to baseline healthcare-utilisation rates. The code recomputes K=2..6 diagnostics, checks stability and size, and only retains the prespecified report-facing K=4 candidate if it remains defensible on real data.

A sensitivity also replaces total inpatient with non-emergency inpatient utilisation to check whether the nested emergency-inpatient definition is driving cluster structure.

## `orchestration/preflight.py`

Runs no-patient-row source/header and methodological readiness checks before the production analysis is allowed to proceed.

## `orchestration/tre.py`

Runs stages in dependency order and records PASS/FAIL status plus configuration hashes for reproducibility.

## `orchestration/release_audit.py`

Flags obvious patient-level files and aggregate count cells below the configured pre-screen threshold. It is not a replacement for formal TRE disclosure control.
