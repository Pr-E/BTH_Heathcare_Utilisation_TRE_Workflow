# Refinements made for TRE translation

The uploaded `src` was already a strong synthetic-first development workflow. The TRE package keeps its analytical structure but tightens production behaviour.

## Removed from production package

- synthetic data generation;
- calibration compiler/fidelity tooling;
- development-only orchestration;
- `__pycache__` and package build artefacts.

These are useful outside the TRE but are not required to analyse the real extract.

## Added for TRE production use

### 1. Source mapping layer
Real TRE aliases can be mapped to stable canonical fields in YAML. This prevents repeated source-specific edits to analysis functions.

### 2. Mandatory preflight
The workflow checks file presence, source headers, study dates, pre-index propensity covariates and the two key interpretation semantics before patient-level analysis proceeds.

### 3. Explicit production block on synthetic mode
The TRE pipeline rejects `data_source.mode=synthetic`.

### 4. Reproducibility manifest
Each full run records configuration hashes, Python version, timestamp and git commit when available.

### 5. Resumable stage orchestration
All stages can be run individually or with `--from-stage` / `--to-stage`.

### 6. Propensity missingness handling
Numeric design variables use median imputation plus missingness indicators; categorical missingness is preserved as an explicit level. Raw patient fields are unchanged.

### 7. Propensity leakage protection
Post-index, exposure and clustering fields are explicitly blocked from the propensity covariate list.

### 8. ATT balance gate
The production configuration blocks primary adjusted outcome modelling when maximum absolute SMD exceeds 0.10.

### 9. Clustering translation rule
K=2..6 is always recomputed. Development-informed K=4 is treated as a prespecified candidate only and must pass real-data minimum size and stability criteria. Synthetic centroids are not transferred.

### 10. Release-output pre-screen
Patient-level filenames and obvious small aggregate cells are flagged before formal TRE disclosure review. This is a helper, not a substitute for formal disclosure control.

### 11. Deployment smoke tests
Small in-memory tests verify propensity and clustering core functions without any patient data.
### 12. Real six-table source registry
`pipeline_tre.yaml` now uses the actual TRE filenames and source roles already present in the BTH workspace (`msk_without_sports`, `inpatient_msk`, `msk_without_sports_ed`, `only_msk_sports`, `inpatient_msk_sports`, `only_msk_sports_ed`).

### 13. Identifier verification by real cross-source overlap
The production workflow no longer assumes that a SHA-256 field name alone proves its semantic role. Patient-ID candidates in inpatient/ED sources are checked against the corresponding MSK cohort. ED event IDs are selected separately using uniqueness diagnostics. Ambiguous/no-overlap linkage blocks the run and writes aggregate QA only.

### 14. Synthetic study dates removed from production defaults
The synthetic development window is not carried into the real workflow. `study_start_date` and `study_end_date` are deliberately unset until approved real extract coverage dates are supplied.

### 15. Real-source mapping worksheet pre-populated
`docs/source_mapping_tre.csv` records the expected real source fields, filenames and identifier evidence. Fields seen in prior real-data EDA are distinguished from identifiers explicitly configured in the existing TRE notebook.


---

# Final-review v0.4.0 additions

The pre-TRE final-review release adds operational safeguards and auditability without changing the core primary estimand:

- reusable patient-safe stage audit module (`src/bth_analysis/audit.py`);
- aggregate JSON + Markdown summary after every stage;
- exact `NEXT STEP` command printed after each stage;
- explicit Stage 02 per-column missingness audit and Sports-vs-Wider missingness comparison;
- missingness classification driven by source/config semantics rather than percentage alone;
- Stage 03 episode-to-spell transformation findings surfaced in logs;
- Stage 04 linkage/source-coverage findings surfaced in logs;
- Stage 05 cohort/index/follow-up decision findings surfaced in logs;
- Stage 06 crude person-time rate and zero-event QA surfaced in logs;
- Stage 07 compact descriptive key-findings output;
- Stage 08 propensity logistic model-term audit, design diagnostics and a hard ATT balance gate before primary model interpretation;
- Stage 08 compact primary comparative key-findings output;
- Stage 09 compact clustering key-findings output plus sparse Sports-linked phenotype warning;
- Stage 11 explicit reminder that release audit is a pre-screen, not disclosure approval;
- `scripts/review_audit_summary.py` to produce a reviewer-facing audit report from aggregate stage summaries only;
- complete code-review/step-by-step/audit documentation for Ian and TRE handover;
- additional no-real-data tests for missingness classification and stage-audit behaviour.

The analytical hierarchy remains:

1. descriptive/crude analysis;
2. logistic propensity design;
3. structural positivity + empirical common support;
4. ATT weighting + SMD balance gate;
5. 1:3 PSM design sensitivity;
6. ATT-weighted Poisson GEE comparative pre/post primary model with log(person-time) offset;
7. Negative Binomial GEE sensitivity;
8. exploratory baseline utilisation clustering.
