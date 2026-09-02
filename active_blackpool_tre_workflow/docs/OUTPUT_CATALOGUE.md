# Output catalogue

## Internal patient-level artefacts — do not request for egress

Typical examples include:

- `data/04_analysis/patient_spine.csv`
- `data/04_analysis/analysis_index.csv`
- `data/04_analysis/healthcare_event_ledger.csv`
- `data/04_analysis/patient_outcomes.csv`
- `outputs/comparative/propensity_scored_population.csv`
- `outputs/comparative/psm_matched_population.csv`
- `outputs/clustering/tables/cluster_assignments.csv`

These are required for analysis inside the TRE but are not intended as release outputs.

---

## Aggregate QA / analytical design outputs

### Stage 00 — Preflight

- `outputs/qa/00_tre_preflight.csv`

### Stage 01 — Ingestion

- `outputs/qa/01_ingestion_summary.csv`

### Stage 02 — Cleaning + missingness

- `outputs/qa/02_identifier_resolution_qa.csv`
- `outputs/qa/02_cleaning_actions_and_issues.csv`
- `outputs/qa/02_cleaned_table_summary.csv`
- `outputs/qa/02_column_missingness.csv`
- `outputs/qa/02_missingness_group_comparison.csv`

### Stage 03 — Preprocessing

- `outputs/qa/03_reference_cohort_configuration.csv`
- `outputs/qa/03_identifier_resolution_qa.csv`
- `outputs/qa/03_patient_linkage_coverage.csv`
- `outputs/qa/03_preprocessing_summary.csv`

### Stage 04 — Linkage

- `outputs/qa/04_linkage_overlap_qa.csv`
- `outputs/qa/04_group_source_coverage.csv`
- `outputs/qa/04_demographic_source_qa.csv`

### Stage 05 — Cohort/index

- `outputs/qa/05_cohort_exclusions.csv`
- `outputs/qa/05_cohort_flow.csv`
- `outputs/qa/05_index_strategy_qa.csv`

### Stage 06 — Outcomes

- `outputs/qa/06_outcome_feature_qa.csv`
- `outputs/qa/06_outcome_rate_qa.csv`

### Stage 07 — Descriptive

Key compact output:

- `outputs/descriptive/tables/descriptive_key_findings.csv`

The descriptive directory additionally contains Table 1, missingness, baseline balance, crude-rate/change, pathway timing, source-coverage, follow-up, event-distribution and figure outputs.

### Stage 08 — Comparative / propensity

- `outputs/comparative/design_overlap_audit.csv`
- `outputs/comparative/propensity_balance.csv`
- `outputs/comparative/propensity_diagnostics.csv`
- `outputs/comparative/att_weight_diagnostics.csv`
- `outputs/comparative/propensity_score_distribution.csv`
- `outputs/comparative/propensity_logistic_model_terms.csv`
- `outputs/comparative/crude_period_rates.csv`
- `outputs/comparative/crude_comparative_change.csv`
- `outputs/comparative/comparative_results.csv`
- `outputs/comparative/primary_comparative_key_findings.csv`
- diagnostic figures in `outputs/comparative/figures/`

### Stage 09 — Clustering

- `outputs/clustering/tables/clustering_summary.csv`
- `outputs/clustering/tables/clustering_key_findings.csv`
- `outputs/clustering/tables/cluster_selection_metrics.csv`
- `outputs/clustering/tables/cluster_baseline_profiles.csv`
- `outputs/clustering/tables/cluster_centroids_standardised.csv`
- `outputs/clustering/tables/cluster_characterisation.csv`
- `outputs/clustering/tables/cluster_exposure_distribution.csv`
- `outputs/clustering/tables/cluster_exposure_association.csv`
- `outputs/clustering/tables/cluster_change_summary.csv`
- figures in `outputs/clustering/figures/`


### Stage 10 — Release pre-screen

- pre-screen tables under `outputs/release_audit/`

---

## Reproducibility / reviewer audit

- `outputs/audit/tre_run_manifest.json`
- `outputs/audit/stage_status.csv`
- `outputs/audit/stage_summaries/*.json`
- `outputs/audit/stage_summaries/*.md`
- `outputs/audit/reviewer_summary.csv`
- `outputs/audit/reviewer_summary.md`

Run `python scripts/review_audit_summary.py` after one or more stages to build the reviewer summary.

---

## Release reminder

Being aggregate does not automatically mean releasable. All requested outputs remain subject to the TRE's formal disclosure-control policy. Stage 10 is an internal pre-screen only.
