# Code review map

This map lists the executable Python files in the final TRE workflow and their primary role.

| File | Lines | Purpose |
|---|---:|---|
| `scripts/review_audit_summary.py` | 153 | Build one reviewer-facing audit summary from aggregate per-stage JSON files. Why this script exists ---------------------- Ian or another reviewer should be able to understand the  |
| `scripts/run_00_preflight.py` | 37 | Run Stage: Preflight. This wrapper intentionally contains no analytical logic. It only makes the project ``src`` package importable when running from a checked-out TRE folder and t |
| `scripts/run_01_ingestion.py` | 37 | Run Stage: Ingestion. This wrapper intentionally contains no analytical logic. It only makes the project ``src`` package importable when running from a checked-out TRE folder and t |
| `scripts/run_02_cleaning.py` | 37 | Run Stage: Cleaning. This wrapper intentionally contains no analytical logic. It only makes the project ``src`` package importable when running from a checked-out TRE folder and th |
| `scripts/run_03_preprocessing.py` | 37 | Run Stage: Preprocessing. This wrapper intentionally contains no analytical logic. It only makes the project ``src`` package importable when running from a checked-out TRE folder a |
| `scripts/run_04_linkage.py` | 37 | Run Stage: Linkage. This wrapper intentionally contains no analytical logic. It only makes the project ``src`` package importable when running from a checked-out TRE folder and the |
| `scripts/run_05_cohort.py` | 37 | Run Stage: Cohort/index/windows. This wrapper intentionally contains no analytical logic. It only makes the project ``src`` package importable when running from a checked-out TRE f |
| `scripts/run_06_outcomes.py` | 37 | Run Stage: Outcome engineering. This wrapper intentionally contains no analytical logic. It only makes the project ``src`` package importable when running from a checked-out TRE fo |
| `scripts/run_07_descriptive.py` | 37 | Run Stage: Descriptive/EDA. This wrapper intentionally contains no analytical logic. It only makes the project ``src`` package importable when running from a checked-out TRE folder |
| `scripts/run_08_comparative.py` | 37 | Run Stage: Propensity + comparative modelling. This wrapper intentionally contains no analytical logic. It only makes the project ``src`` package importable when running from a che |
| `scripts/run_09_clustering.py` | 37 | Run Stage: Exploratory clustering. This wrapper intentionally contains no analytical logic. It only makes the project ``src`` package importable when running from a checked-out TRE |
| `scripts/run_10_release_audit.py` | 37 | Run Stage: Release pre-screen. This wrapper intentionally contains no analytical logic. It only makes the project ``src`` package importable when running from a checked-out TRE fol |
| `scripts/run_all_tre.py` | 83 | Command-line orchestration entry point for the reviewed real-data TRE workflow. Examples -------- Run the complete required analytical chain through clustering:: python scripts/run |
| `src/bth_analysis/__init__.py` | 9 | Active Blackpool / BTH real-data TRE analysis package. The package contains the real-data analytical workflow and requires approved TRE source configuration before patient-level pr |
| `src/bth_analysis/analysis/__init__.py` | 5 | Statistical/descriptive analysis modules for the Active Blackpool TRE workflow. Modules are intentionally not auto-imported here: each stage is invoked through its explicit runner/ |
| `src/bth_analysis/analysis/clustering.py` | 1164 | Exploratory baseline healthcare-utilisation phenotyping for TRE data. Clusters are formed from pre-index utilisation only. Pathway membership, demographics and follow-up outcomes a |
| `src/bth_analysis/analysis/comparative.py` | 942 | Primary and sensitivity comparative healthcare-utilisation models. Analysis hierarchy ------------------ 1. Crude baseline/follow-up rates are reported first. 2. Logistic-regressio |
| `src/bth_analysis/analysis/descriptive.py` | 1727 | Stage 07: descriptive EDA, cohort characterisation and diagnostic figures. This layer describes who is in the analysis, missingness, source coverage, baseline utilisation, temporal |
| `src/bth_analysis/analysis/propensity.py` | 643 | Propensity-score design for the primary TRE comparative analysis. The propensity layer is a *design* step, not the outcome analysis. Logistic regression estimates each patient's pr |
| `src/bth_analysis/audit.py` | 202 | Patient-safe audit and terminal-logging helpers for the TRE workflow. Each stage uses these helpers to print consistent aggregate findings, write a compact JSON/Markdown stage summ |
| `src/bth_analysis/data_pipeline/__init__.py` | 10 | Public entry points for deterministic real-source data pipeline stages. |
| `src/bth_analysis/data_pipeline/cleaning.py` | 616 | Stage 02: deterministic source cleaning and data-quality audit. The stage standardises blanks, parses configured dates/numerics, removes only fully blank or duplicate records under |
| `src/bth_analysis/data_pipeline/cohort.py` | 602 | Stage 05: freeze the comparative cohort, analytical index and observation windows. The current fallback real-data design uses source-relative FirstMSKDate anchors for both groups.  |
| `src/bth_analysis/data_pipeline/config.py` | 63 | Configuration/path helpers for real-source pipeline stages inside the TRE. Paths are configured in YAML so the workflow can run in an approved TRE workspace without changing analyt |
| `src/bth_analysis/data_pipeline/identifiers.py` | 299 | Resolve patient and event identifiers across the six TRE source tables. Patient identifiers in hospital extracts are verified against the matching MSK cohort using aggregate overla |
| `src/bth_analysis/data_pipeline/ingestion.py` | 239 | Stage 01: schema-aware ingestion of approved TRE extracts. This stage intentionally performs *no analytical cleaning or derivation*. Its responsibilities are limited to: 1. reading |
| `src/bth_analysis/data_pipeline/linkage.py` | 417 | Stage 04: build the pseudonymised patient spine and cross-source linkage QA. The stage combines pathway and hospital source-presence flags by PatientID, resolves baseline demograph |
| `src/bth_analysis/data_pipeline/mapping.py` | 48 | Map real TRE source aliases to the stable canonical analytical schema. The analytical code should not be rewritten because a refreshed BTH extract changes capitalisation or an appr |
| `src/bth_analysis/data_pipeline/missingness.py` | 235 | Audit column-level missingness during deterministic TRE cleaning. The module reports missing counts and percentages, flags analytically critical fields, applies configured expected |
| `src/bth_analysis/data_pipeline/outcomes.py` | 480 | Stage 06: construct patient-level healthcare-utilisation outcomes. ED attendances are counted at attendance level; inpatient admissions are counted at spell/admission level rather  |
| `src/bth_analysis/data_pipeline/preprocessing.py` | 1019 | Stage 03: preprocessing and source-grain harmonisation. This stage converts the six cleaned source tables into canonical analytical views while preserving source lineage. It: - seq |
| `src/bth_analysis/data_pipeline/qa.py` | 115 | Small aggregate QA helpers shared by source-data pipeline stages. These functions intentionally return or print table-level counts only. They do not expose patient hashes or row-le |
| `src/bth_analysis/data_pipeline/schemas.py` | 175 | Canonical minimum schemas for the six real Active Blackpool/BTH extracts. The source mapping layer may rename real TRE columns into these canonical names before validation. These l |
| `src/bth_analysis/orchestration/__init__.py` | 7 | Production orchestration, readiness and release-audit entry points. |
| `src/bth_analysis/orchestration/preflight.py` | 286 | TRE preflight checks. Preflight is intentionally separate from data analysis. It verifies that the approved source files, schemas, time window and interpretation semantics are in p |
| `src/bth_analysis/orchestration/readiness.py` | 90 | Configuration-level readiness checks for TRE translation and interpretation. This module does not read patient data. It separates two questions that should not be conflated during  |
| `src/bth_analysis/orchestration/release_audit.py` | 163 | Pre-screen candidate outputs before formal TRE disclosure control. This is *not* a disclosure-control engine. It simply helps analysts identify obvious patient-level files and smal |
| `src/bth_analysis/orchestration/tre.py` | 165 | Run the real-data TRE workflow in dependency order. The orchestrator starts from approved TRE extracts and executes preflight, ingestion, cleaning, preprocessing, linkage, cohort c |
| `src/bth_analysis/workflow.py` | 143 | Shared configuration, path and reproducibility helpers for the TRE workflow. The TRE package intentionally keeps configuration outside analysis code. The same Python functions can  |

## Functions and classes

### `scripts/review_audit_summary.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 32 | `_markdown_table` | Return a dependency-free Markdown table. Pandas ``DataFrame.to_markdown`` requires the optional ``tabulate`` package. We avoid that extra dependency so the reviewer utility works i |
| 53 | `main` | Create ``reviewer_summary.csv`` and ``reviewer_summary.md``. The summary deliberately contains stage status, aggregate key findings, warnings and the next workflow command only. It |

### `scripts/run_all_tre.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 46 | `main` | Parse CLI arguments and execute the requested contiguous stage range. |

### `src/bth_analysis/analysis/clustering.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 72 | `_load_clustering_config` | Load clustering YAML and retain its resolved path for reproducibility. |
| 80 | `_require_columns` | Fail early if a required clustering input is absent. |
| 87 | `_clean_numeric` | Coerce one clustering feature to finite numeric values, preserving missingness for explicit checking. |
| 92 | `_pairwise_mean_ari` | Calculate mean pairwise Adjusted Rand Index across repeated cluster fits. |
| 103 | `_cramers_v` | Calculate Cramer's V effect size and chi-square p-value for categorical association. |
| 117 | `_prepare_matrix` | Winsorise, log-transform and standardise baseline utilisation inputs for K-means. |
| 153 | `_evaluate_k` | Fit candidate K solutions and calculate separation, size and stability diagnostics. |
| 235 | `_choose_k` | Select K using the configured stability/size rules and clustering selection policy. |
| 261 | `_reorder_clusters` | Relabel raw K-means cluster IDs into a stable descriptive order for reporting. |
| 286 | `_baseline_profiles` | Summarise raw baseline utilisation distributions within each cluster. |
| 310 | `_cluster_characterisation` | Create provisional data-derived phenotype descriptions for later clinical review. |
| 343 | `_centroid_table` | Return standardised cluster centroids by baseline utilisation feature. |
| 361 | `_demographic_numeric` | Profile numeric demographic/follow-up variables after cluster formation. |
| 389 | `_categorical_profiles` | Profile categorical characteristics after clustering without using them to create clusters. |
| 425 | `_exposure_distribution` | Compare pathway-group composition across clusters and calculate descriptive association. |
| 467 | `_trajectory_tables` | Calculate descriptive baseline/follow-up outcome rates within clusters and pathway groups. |
| 520 | `_followup_profiles` | Summarise follow-up outcome distributions within baseline utilisation phenotypes. |
| 545 | `_plot_cluster_selection` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 560 | `_plot_cluster_sizes` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 577 | `_plot_centroid_heatmap` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 598 | `_plot_group_cluster_distribution` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 642 | `_plot_followup_by_cluster` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 672 | `run_clustering` | Run the exploratory real-data clustering layer, diagnostics, sensitivities and phenotype audit. |

### `src/bth_analysis/analysis/comparative.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 51 | `_rate_result` | Convert one fitted model coefficient into an auditable ratio, confidence interval and p-value row. |
| 99 | `_alpha_mom` | Method-of-moments dispersion parameter used for NB-GEE sensitivity. Statsmodels' GEE NegativeBinomial family requires a supplied alpha rather than estimating it jointly in the same |
| 117 | `_family` | Return the configured Poisson or Negative Binomial count-family object. |
| 124 | `_fit_gee_rate` | Fit one follow-up-only GEE count-rate model with person-time offset and sparse-event gates. |
| 220 | `_prepost_stack` | Reshape patient outcomes into repeated baseline/follow-up rows for comparative pre/post modelling. |
| 252 | `_fit_prepost` | Fit the group-by-period GEE model whose interaction is the comparative rate-ratio-of-rate-ratios. |
| 376 | `_crude_period_rates` | Calculate unadjusted baseline/follow-up rates per person-time before propensity adjustment. |
| 405 | `_crude_change_summary` | Summarise the unadjusted difference in baseline-to-follow-up change between groups. |
| 445 | `_plot_love` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 472 | `_plot_ps_overlap` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 495 | `_plot_weight_distribution` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 520 | `_plot_primary_forest` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 549 | `run_comparative` | Run propensity design, balance gates and the primary/sensitivity comparative models. |

### `src/bth_analysis/analysis/descriptive.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 93 | `_to_dt` | Coerce a series to pandas datetime while turning unparseable values into missing timestamps. |
| 100 | `_slug` | Create a filesystem-safe text label for aggregate figure filenames. |
| 106 | `_numeric` | Coerce a series to numeric values for robust descriptive calculations. |
| 111 | `_safe_pct` | Calculate a percentage while returning missing when the denominator is zero. |
| 116 | `_derive_reporting_features` | Derive presentation/EDA-only age bands and IMD quintiles without changing core analysis variables. |
| 148 | `_group_lookup` | Create an ExposureFlag-to-display-label mapping from observed analysis groups. |
| 162 | `_summarise_numeric` | Calculate grouped descriptive statistics for configured numeric baseline variables. |
| 206 | `_summarise_categories` | Calculate grouped counts/percentages for configured categorical baseline variables. |
| 244 | `_numeric_smd` | Calculate an unweighted SMD for one continuous baseline variable. |
| 256 | `_binary_smd` | Calculate an unweighted SMD for one binary indicator. |
| 264 | `_baseline_balance` | Calculate unadjusted baseline SMDs used to show why confounding adjustment is required. |
| 324 | `_cohort_flow` | Build aggregate cohort-flow counts for descriptive reporting. |
| 357 | `_missingness_summary` | Summarise effective analytical missingness by group for key descriptive variables. |
| 385 | `_source_coverage` | Summarise linked-source availability by analysis group. |
| 414 | `_pathway_timing` | Summarise referral/MSK timing intervals and chronology-related descriptive measures. |
| 475 | `_utilisation_summary` | Calculate crude outcome distributions and person-time rates by group/period. |
| 523 | `_prepost_change` | Calculate crude within-group baseline-to-follow-up changes. |
| 553 | `_index_temporal_summary` | Summarise index-date counts over calendar time. |
| 575 | `_event_structure` | Summarise event-ledger composition by event type/source/period. |
| 618 | `_expand_event_outcomes` | Expand event-ledger rows into the configured outcome categories for aggregate summaries. |
| 645 | `_overlap_days` | Calculate the number of observed days overlapping two time intervals. |
| 658 | `_relative_time_rates` | Calculate aggregate utilisation trajectories in bins relative to the analytical index. |
| 724 | `_correlations` | Calculate aggregate baseline numeric correlations for EDA. |
| 744 | `_table1` | Assemble the report-facing baseline-characteristics Table 1. |
| 807 | `_eda_diagnostics` | Create explicit review flags for descriptive data-quality or distributional concerns. |
| 902 | `_style_axis` | Apply restrained common formatting to a Matplotlib axis. |
| 915 | `_group_colour_for_label` | Return a stable plotting colour choice for one analysis-group label. |
| 926 | `_save_grouped_horizontal_bar` | Persist the corresponding aggregate analytical/QA output in a reproducible format. |
| 998 | `_plot_age_distribution` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 1030 | `_plot_followup_days` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 1065 | `_plot_index_timeline` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 1090 | `_plot_utilisation_rates` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 1123 | `_plot_baseline_smd` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 1145 | `_plot_relative_trajectory` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 1177 | `_annotated_heatmap` | Render an annotated aggregate heatmap for EDA output. |
| 1205 | `_plot_correlation_heatmaps` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 1218 | `_cramers_v` | Calculate Cramer's V effect size and chi-square p-value for categorical association. |
| 1236 | `_categorical_associations` | Internal helper for categorical associations; see the surrounding module comments for the analytical rationale. |
| 1253 | `_plot_categorical_associations` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 1266 | `_plot_source_coverage` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 1326 | `_plot_pathway_timing` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 1364 | `_plot_age_vs_baseline_utilisation` | Create and save the corresponding aggregate diagnostic/reporting figure. |
| 1394 | `_write_manifest` | Write a descriptive-output manifest for reproducible review. |
| 1423 | `run_descriptive` | Run the complete unadjusted descriptive/EDA layer and emit aggregate key findings. |

### `src/bth_analysis/analysis/propensity.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 31 | `PropensityResult` | Internal helper for PropensityResult; see the surrounding module comments for the analytical rationale. |
| 43 | `_weighted_mean_var` | Return the weighted mean and variance used by balance diagnostics. |
| 56 | `_smd` | Calculate a standardised mean difference between Sports-linked and comparison groups. |
| 73 | `_derive_design_covariates` | Create design-only covariates without changing upstream pipeline outputs. |
| 83 | `_normalised_level` | Convert categorical values to stable strings while retaining missingness explicitly. |
| 88 | `_apply_overlap_restrictions` | Restrict the propensity-design population to categorical levels represented in both groups. This is a design/positivity restriction, not data cleaning. |
| 149 | `_validate_preindex_covariates` | Block accidental post-index/exposure leakage into the propensity model. |
| 170 | `_feature_matrix` | Build the leakage-safe propensity design matrix and preprocessing pipeline. |
| 231 | `_logit` | Convert propensity probabilities to the log-odds scale used for matching distances. |
| 237 | `_greedy_match` | Create nearest-neighbour propensity matched sets under the configured caliper and ratio. |
| 337 | `_balance_table` | Calculate before/after covariate balance statistics for the propensity design. |
| 380 | `_weight_diagnostics` | Summarise ATT weight distribution and effective sample size without patient-level output. |
| 406 | `_ps_distribution` | Summarise aggregate propensity-score distributions by group and support status. |
| 430 | `fit_propensity` | Execute the complete propensity design: positivity, logistic PS, ATT, SMD and PSM sensitivity. |

### `src/bth_analysis/audit.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 35 | `_json_safe` | Convert common pandas/numpy values into JSON-safe aggregate values. |
| 59 | `stage_header` | Print a consistent stage header so terminal logs are easy to scan. |
| 82 | `section` | Print a visually distinct subsection inside a stage log. |
| 89 | `metric` | Print one aggregate key-value result with aligned labels. |
| 95 | `dataframe_preview` | Print a bounded aggregate table preview; never use with patient-level rows. |
| 116 | `save_stage_summary` | Write one compact JSON stage summary containing no patient-level records. |
| 178 | `stage_footer` | Print the audit trail and the exact next workflow command. |

### `src/bth_analysis/data_pipeline/cleaning.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 36 | `normalize_blank_strings` | Convert whitespace-only text cells to explicit pandas missing values. This standardises visually blank strings such as ``" "`` so downstream missingness counts do not treat them as |
| 50 | `parse_dates` | Parse configured source date/time fields and coerce invalid text to NaT. ``errors="coerce"`` is deliberate: an invalid clinical timestamp is safer represented as missing and audite |
| 68 | `coerce_numeric` | Convert configured numeric fields to numbers; malformed values become NA. |
| 77 | `clean_msk` | Clean one MSK pathway table and return aggregate chronology QA. The function removes only fully blank/exact-duplicate rows, standardises configured date/numeric types and *counts*  |
| 163 | `clean_inpatient` | Clean one inpatient episode table without collapsing episodes to spells. Spell/episode structural checks happen here, while the actual episode-to-spell aggregation is deferred to S |
| 260 | `clean_ed` | Clean one ED attendance table and enforce the resolved patient/event key. Invalid ages/departure timestamps are converted to missing rather than guessed. Duplicate patient-event ke |
| 363 | `save_cleaned` | Persist the cleaned canonical table using an explicit datetime format. |
| 373 | `run_cleaning` | Clean all ingested TRE tables after resolving source identifiers. The real extracts contain several SHA-256-labelled fields whose semantic role is verified by cross-source overlap. |

### `src/bth_analysis/data_pipeline/cohort.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 26 | `_to_dt` | Parse canonical pipeline timestamps written as ISO-style strings. Raw source date parsing happens upstream during preprocessing. By the time cohort construction runs, dates have be |
| 43 | `_bool_flag` | Return a pipeline-style Int64 flag repeated for every cohort row. |
| 48 | `_assign_index_date` | Assign the configured analytical index without claiming programme start. Current fallback strategy ------------------------- ``source_relative_first_msk`` uses the same *semantic*  |
| 137 | `run_cohort_index` | Create the comparative cohort, analytical index and observation windows. The current fallback design compares: * Sports-linked BTH pathway patients; and * Wider MSK non-Sports-link |

### `src/bth_analysis/data_pipeline/config.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 15 | `load_pipeline_config` | Load source-pipeline YAML and attach the resolved repository root. |
| 35 | `resolve_from_project` | Resolve an absolute path or a project-relative configured path. |
| 43 | `source_dir` | Return the approved TRE source directory and reject any non-TRE source mode. |
| 59 | `output_dir` | Resolve one named source-pipeline output directory from configuration. |

### `src/bth_analysis/data_pipeline/identifiers.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 18 | `IdentifierChoice` | Selected identifier columns for one source table. |
| 25 | `_dedupe` | Return non-empty values once, preserving configured preference order. |
| 39 | `_candidate_columns` | Return configured preferred identifier followed by fallback candidates. |
| 54 | `_as_id_set` | Convert an identifier column to a set without materialising missing tokens. |
| 59 | `_score_candidate` | Create aggregate diagnostics for one candidate identifier column. |
| 83 | `choose_patient_identifier` | Select the patient identifier, using cohort overlap when available. Reference cohort tables (MSK source tables) use their configured preferred identifier when present. Healthcare-e |
| 156 | `choose_event_identifier` | Select the event identifier, preferring a field distinct from patient ID. For ED sources, the remaining candidate with the greatest uniqueness is a practical attendance-ID check. T |
| 208 | `resolve_identifier_plan` | Resolve identifiers for all configured real-TRE source tables. Resolution order is important: MSK cohort IDs are selected first, then each healthcare source is checked against the  |
| 287 | `apply_identifier_choices_to_config` | Return a copied table configuration with resolved ID fields inserted. |

### `src/bth_analysis/data_pipeline/ingestion.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 36 | `_expected_schema` | Return the table's canonical expected column order. |
| 46 | `validate_schema` | Validate required columns after source-to-canonical mapping. Missing required columns always stop the workflow. Extra columns can be either blocked (strict mode) or retained/ignore |
| 78 | `align_columns` | Place canonical fields in a deterministic order. |
| 93 | `_read_source` | Read CSV or Parquet according to the table configuration. |
| 109 | `_write_ingested` | Write a canonical CSV consumed by all downstream stages. |
| 115 | `run_ingestion` | Ingest all configured TRE tables into the canonical layer. |

### `src/bth_analysis/data_pipeline/linkage.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 50 | `_read` | Read one named processed dataset and fail clearly if the prior stage is missing. |
| 58 | `_first_demographics` | Return one deterministic demographic record per patient from one source. Rows are kept intact rather than taking the first non-missing value separately by column. The source label  |
| 81 | `_coalesce_demographics` | Resolve demographics by source priority and retain provenance per field. For each demographic variable, the first non-missing value in the documented source-priority order is selec |
| 147 | `run_linkage` | Build the pseudonymised patient spine and report cross-source linkage completeness. |

### `src/bth_analysis/data_pipeline/mapping.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 15 | `apply_column_mapping` | Rename configured source columns and block ambiguous canonical duplicates. |
| 45 | `canonical_header` | Apply the same rename map to source header names without loading row data. |

### `src/bth_analysis/data_pipeline/missingness.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 16 | `_cfg` | Return the optional missingness subsection of pipeline configuration. |
| 21 | `_classification` | Resolve the configured interpretation of missingness for one field. |
| 63 | `column_missingness_table` | Return one aggregate missingness row per cleaned source column. |
| 108 | `print_missingness_summary` | Print the most decision-relevant column-level missingness findings. |
| 156 | `_source_pairs` | Identify wider/sports pairs such as msk_wider versus msk_sports. |
| 168 | `compare_group_missingness` | Compare missing percentages on columns common to matched source families. |
| 209 | `print_group_missingness_comparison` | Print the largest Sports-linked versus Wider MSK missingness differences. |

### `src/bth_analysis/data_pipeline/outcomes.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 26 | `_to_dt` | Parse canonical workflow datetimes without locale reinterpretation. Processed and analysis-layer CSVs are written as ISO ``YYYY-MM-DD HH:MM:SS``. Re-reading these values with ``day |
| 42 | `_load` | Read a required upstream CSV and fail with the exact missing path. |
| 49 | `_build_ed_ledger` | Map cleaned ED attendances into the common healthcare-event ledger schema. |
| 67 | `_build_inpatient_ledger` | Map one row per inpatient spell into the common event-ledger schema. EmergencyInpatientFlag is derived from reviewed MethodOfAdmission text patterns; this coding rule must be reche |
| 101 | `run_outcome_features` | Build the healthcare event ledger and patient-level counts/person-time outcome features. |

### `src/bth_analysis/data_pipeline/preprocessing.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 50 | `validate_reference_cohort_config` | Validate the configured hospital-to-MSK reference relationships. Hospital patient identifiers are verified against the MSK cohort from the same pathway population. A missing or con |
| 82 | `parse_datetime_columns` | Re-parse canonical cleaned timestamps after CSV serialisation. Stage 02 already validated/parses source dates. Stage 03 reads cleaned CSVs back from disk, so datetime dtype informa |
| 101 | `days_between` | Return elapsed days between two parsed timestamp series as floating days. |
| 106 | `make_msk_referral_view` | Create one canonical MSK referral/history view from a cleaned source table. PatientID is derived from the resolved source identifier, pathway dates are parsed, records are ordered  |
| 198 | `make_pathway_anchor` | Create one deterministic source-relative pathway anchor per MSK patient. The earliest ordered referral row is retained intact. ``drop_duplicates`` is used instead of ``groupby.firs |
| 242 | `derive_relative_timeframe` | Classify an event as before/during/after the source-relative MSK pathway. This derived label is used to reconcile the workflow's date logic against any source-provided timeframe la |
| 278 | `_aggregate_codes_by_spell` | Collapse repeated diagnosis/procedure slots to unique codes per spell. The source stores codes across repeated slot columns. The function reshapes those slots once, removes repeate |
| 312 | `make_inpatient_views` | Build episode-level and spell/admission-level inpatient views. The episode table preserves cleaned consultant-episode detail for QA and code aggregation. The spell table contains o |
| 484 | `make_ed_view` | Create one row per cleaned ED attendance with canonical IDs and timing QA. |
| 549 | `processed_summary` | Return aggregate dimensions, patient linkage and timeframe QA metrics. |
| 622 | `patient_linkage_summary` | Summarise patient-level linkage and anchor completeness for one event view. |
| 657 | `save_processed` | Persist a processed analytical-grain table using canonical datetime text. |
| 667 | `run_preprocessing` | Harmonise cleaned source grain, derive pathway views and collapse inpatient episodes to spells. |

### `src/bth_analysis/data_pipeline/qa.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 16 | `save_records` | Write a list of aggregate QA dictionaries to CSV. Parameters ---------- records: Aggregate dictionaries such as row counts, chronology flags or balance diagnostics. Callers must no |
| 32 | `basic_table_summary` | Return patient-safe table dimensions and identifier completeness metrics. Identifier *values* are never included. We report only uniqueness and missingness counts for the configure |
| 86 | `print_table_summary` | Print the most useful aggregate table QA metrics in a stable format. |

### `src/bth_analysis/orchestration/preflight.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 28 | `_header` | Read and canonicalise only a source-file header for preflight schema checks. |
| 50 | `run_preflight` | Run source, schema and methodological-readiness checks. |

### `src/bth_analysis/orchestration/readiness.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 19 | `translation_readiness` | Return one aggregate row per interpretation/configuration readiness check. |

### `src/bth_analysis/orchestration/release_audit.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 33 | `_looks_like_count_column` | Identify aggregate columns that should be inspected for small-cell disclosure risk. |
| 45 | `run_release_audit` | Pre-screen outputs for patient-level/internal files and possible small-cell aggregate risks. |

### `src/bth_analysis/orchestration/tre.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 49 | `_stage_log_path` | Resolve the run-level stage-status audit CSV path. |
| 56 | `_record_stage` | Append one aggregate stage execution status row and persist the run status file. |
| 62 | `run_tre_workflow` | Run an ordered section of the TRE workflow. Parameters ---------- from_stage, to_stage: Any names in ``STAGE_ORDER``. These allow an analyst to resume from a validated checkpoint w |

### `src/bth_analysis/workflow.py`

| Line | Function/class | Purpose |
|---:|---|---|
| 22 | `load_workflow_config` | Load analytical workflow YAML and attach portable project metadata. |
| 40 | `resolve_path` | Resolve absolute paths or project-relative configured paths. |
| 48 | `output_path` | Resolve a named analytical/audit output directory from workflow config. |
| 55 | `config_sha256` | Return a content digest so every run can be tied to exact configuration. |
| 61 | `_safe_git_commit` | Return current Git commit when available without making Git mandatory. |
| 81 | `_dependency_versions` | Return installed package versions for reproducibility without importing data. |
| 103 | `build_run_manifest` | Create non-patient reproducibility metadata for the current run. |
| 133 | `write_run_manifest` | Persist one JSON run manifest in the configured TRE audit directory. |

