# Final-review navigation

For the most explicit command-by-command execution sequence and decision gates, use `docs/TRE_STEP_BY_STEP_EXECUTION.md`. For Ian's pre-ingress review, use `docs/IAN_FINAL_REVIEW_GUIDE.md`.

---

# TRE runbook

## Before touching the data

1. Confirm the approved project workspace and source location.
2. Confirm that patient identifiers are pseudonymised.
3. Review the BTH data dictionary for every configured source table.
4. Review the pre-populated real source register in `docs/REAL_TRE_SOURCE_REGISTER.md`.
5. Review `docs/source_mapping_tre.csv`; only add YAML aliases where the current extract header differs from the expected real-data field name.
6. Confirm the six configured real filenames are present under the approved source directory.
7. Confirm the real linked-extract coverage start/end dates and populate `workflow_tre.yaml`.
8. Retain the confirmed pathway-group meaning: Sports-linked BTH MSK pathway versus wider MSK without Sports; do not relabel this as confirmed programme treatment.
9. Confirm what `FirstMSKDate` means in both pathway source families before changing the index-semantic blocker to true.

## First execution

```bash
python scripts/run_00_preflight.py
```

Resolve every BLOCKER before continuing.

Then run incrementally on the first real extract:

```bash
python scripts/run_01_ingestion.py
python scripts/run_02_cleaning.py
python scripts/run_03_preprocessing.py
python scripts/run_04_linkage.py
python scripts/run_05_cohort.py
python scripts/run_06_outcomes.py
python scripts/run_07_descriptive.py
```

Review QA after every stage rather than running immediately to the outcome models. In particular, inspect `02_identifier_resolution_qa.csv` and `03_identifier_resolution_qa.csv`: healthcare patient hashes must overlap the corresponding MSK cohort and the selected identifier must be unambiguous.

## Design review before modelling

Before Stage 08, verify:

- analysis group counts are plausible;
- index-date distributions are plausible by group/year;
- baseline is complete under the agreed rule;
- follow-up person-time is not unexpectedly differential;
- inpatient events are spell-level;
- emergency admission coding is correct;
- baseline covariates are measured before index;
- no post-index variable appears in propensity design.

Then run:

```bash
python scripts/run_08_comparative.py
```

Review in order:

1. `design_overlap_audit.csv`;
2. `propensity_score_distribution.csv`;
3. overlap figure;
4. `att_weight_diagnostics.csv`;
5. `propensity_balance.csv` and love plot;
6. `propensity_diagnostics.csv`;
7. `crude_period_rates.csv`;
8. `crude_comparative_change.csv`;
9. `comparative_results.csv`;
10. primary forest plot.

Do not interpret adjusted results if ATT balance fails.

## Clustering review

Run:

```bash
python scripts/run_09_clustering.py
```

Review:

- population flow;
- K=2..6 diagnostics;
- K=4 adequacy/stability;
- centroid heatmap;
- cluster prevalence;
- Sports-linked counts per phenotype;
- winsorisation sensitivity ARI;
- clinical interpretability.

If the prespecified K=4 fails real-data criteria, the workflow blocks by default rather than forcing an inadequate cluster structure.

## Output review and egress

Run:

```bash
python scripts/run_10_release_audit.py
```

Then submit only approved aggregate outputs to the local TRE disclosure-control process.
