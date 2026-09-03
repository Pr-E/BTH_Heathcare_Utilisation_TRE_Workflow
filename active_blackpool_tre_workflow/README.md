# Active Blackpool / BTH TRE Analytical Workflow

## Purpose

This repository provides a reproducible, auditable and TRE-ready analytical
workflow for evaluating healthcare-utilisation trajectories associated with the
Active Blackpool exercise referral pathway. It compares the **Sports-linked BTH
pathway population** with a suitably adjusted **Wider MSK non-Sports-linked
population** using linked MSK, Emergency Department and inpatient data.

The workflow starts from the six approved BTH extracts inside the TRE and
rebuilds every analytical dataset and result from those sources. Synthetic
results are not used as programme findings.

## Analytical workflow

```text
00  Preflight and source-semantic checks
 ↓
01  Ingestion and canonical schema mapping
 ↓
02  Deterministic cleaning, identifier verification and missingness audit
 ↓
03  Preprocessing and source-grain harmonisation
    • MSK referral sequences and source-relative pathway anchors
    • inpatient consultant episodes retained for QA
    • inpatient admissions represented at spell level
    • ED attendances retained at attendance level
    • patient linkage and pathway-anchor completeness audited separately
 ↓
04  Pseudonymised patient spine and cross-source linkage QA
 ↓
05  Comparative cohort, analytical index and observation windows
 ↓
06  Healthcare-utilisation outcomes and person-time
 ↓
07  Descriptive analysis and data-quality summaries
 ↓
08  Propensity adjustment and comparative pre/post modelling
 ↓
09  Exploratory baseline healthcare-utilisation clustering
 ↓
10  Release-output pre-screen before formal TRE disclosure review
```

## Primary comparative design

The primary analysis uses pre-index covariates to estimate propensity scores and
assess measured baseline comparability. The workflow includes structural
positivity checks, empirical common support, ATT weighting, standardised mean
difference diagnostics and 1:3 propensity-score matching as a sensitivity
design.

Healthcare-utilisation outcomes are analysed using Poisson GEE with a
**group × period interaction** and **log person-time offset**. Negative Binomial
GEE is used as a distributional sensitivity analysis.

The interaction estimates an adjusted comparative difference in
baseline-to-follow-up change. It must not be described as a confirmed causal
programme effect unless programme exposure/index semantics and the required
causal assumptions are separately established.

## Clustering

Clustering is secondary, exploratory and descriptive. Cluster construction uses
only baseline utilisation rates:

- `BaselineEDRatePerPY`
- `BaselineInpatientRatePerPY`
- `BaselineEmergencyInpatientRatePerPY`

Pathway group, demographics, deprivation, geography and follow-up outcomes are
not used to create clusters; they are used only afterwards to profile the
identified phenotypes. Candidate K values are reassessed on the real data using
separation, size and stability diagnostics.

## Reproducibility and auditability

Each stage:

- reads explicit version-controlled configuration;
- prints its purpose, inputs, outputs and key aggregate findings;
- writes detailed QA tables;
- writes a patient-safe JSON/Markdown stage summary;
- records interpretation boundaries and decision gates;
- prints the next command in the workflow.

Run-level metadata include configuration hashes, package/environment details and
stage PASS/FAIL status. Patient-level analytical files remain inside the TRE and
are not treated as egress candidates.

## Required configuration before real-data analysis

The six approved source extracts are expected under the TRE read-only mount:

```yaml
data_source:
  mode: tre
  tre_dir: /project/readonly
```

Before cohort construction, enter the approved BTH extract coverage dates and
confirm the analytical semantics of `FirstMSKDate`. `FirstMSKDate` is not treated
as a programme-start date unless genuine programme data establish that meaning.

## First-run sequence

From the project root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m compileall -q src scripts tests
python -m pytest -q

python scripts/run_00_preflight.py
python scripts/run_01_ingestion.py
python scripts/run_02_cleaning.py
python scripts/run_03_preprocessing.py
python scripts/run_04_linkage.py
python scripts/run_05_cohort.py
python scripts/run_06_outcomes.py
python scripts/run_07_descriptive.py
python scripts/run_08_comparative.py
python scripts/run_09_clustering.py
python scripts/run_10_release_audit.py
```

For the first real-data execution, run stages individually and inspect each QA
handoff before proceeding. After validation, `python scripts/run_all_tre.py` can
rerun the analytical chain through clustering.

After any partial or complete run:

```bash
python scripts/review_audit_summary.py
```

See `docs/ANALYST_FINAL_REVIEW_GUIDE.md` and
`docs/TRE_STEP_BY_STEP_EXECUTION.md` for the reviewer and execution guides.
