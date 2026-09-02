# Ian final review guide — Active Blackpool / BTH TRE workflow

## Purpose of this review

This package is the final **real-data translation workflow** intended for review before ingress into the Trusted Research Environment (TRE). It does not carry synthetic results into the real analysis. It carries the validated **analytical architecture**, explicit assumptions, decision gates, code comments, aggregate audit logging and reproducibility controls required to regenerate all findings from the approved BTH extracts.

The review should answer four questions:

1. Are the six real BTH source files and identifier semantics represented correctly?
2. Is the current pathway-group/index interpretation acceptable for the initial real-data comparison?
3. Are the cohort, propensity, comparative and clustering specifications methodologically appropriate?
4. Are the logs and audit outputs sufficient for another analyst to understand exactly what happened at every stage without relying on undocumented notebook state?

---

## Current real-data interpretation boundary

The workflow is configured for:

- **Sports-linked BTH pathway** patients; versus
- **Wider MSK non-Sports-linked candidate** patients.

`FirstMSKDate` remains a **source-relative analytical time origin** until its real-data meaning is confirmed. It must not be described as programme start by default.

The package therefore estimates an **adjusted comparative pre/post association in healthcare-utilisation rates**, not a confirmed causal Active Blackpool treatment effect.

Two preflight items remain intentionally unresolved and should block execution until confirmed inside the TRE:

- the approved real BTH extract coverage dates (`study_start_date`, `study_end_date`); and
- the analytical semantics of `FirstMSKDate` (`analytical_index_semantics_confirmed_for_workflow`).

This is intentional safety behaviour, not an incomplete code implementation.

---

## Real source register

The configured six source files are:

| Analytical source | TRE filename |
|---|---|
| Wider MSK cohort | `active_blackpool_msk_cohort_without_sports.csv` |
| Wider MSK inpatient | `active_blackpool_inpatient_msk_.csv` |
| Wider MSK ED | `active_blackpool_msk_cohort_without_sports_ed.csv` |
| Sports-linked MSK cohort | `active_blackpool_only_msk_sports.csv` |
| Sports-linked inpatient | `active_blackpool_inpatient_msk_sports.csv` |
| Sports-linked ED | `active_blackpool_only_msk_sports_ed.csv` |

The ED hash labels are not trusted by name alone. Stage 02 and Stage 03 re-resolve patient/event identifiers by aggregate overlap/uniqueness evidence against the corresponding MSK cohort and write the decision to QA files.

---

## Audit and review features

### 1. Cleaning audit

Stage 02 reports, for every source table:

- rows/columns after cleaning;
- unique patient/event/spell counts;
- rows removed as blank, exact duplicate or duplicate ED patient-event key;
- chronology anomalies and other integrity flags;
- total missing cells **and percentage**;
- which columns are missing;
- missing `n` and `%` per column;
- configured missingness classification (`CRITICAL`, `STRUCTURAL_EXPECTED`, `CONDITIONAL_EXPECTED`, `UNCLASSIFIED`);
- critical fields with missing values;
- top missing columns requiring review;
- Sports-linked vs Wider-MSK missingness differences in percentage points.

New Stage 02 files:

- `outputs/qa/02_column_missingness.csv`
- `outputs/qa/02_missingness_group_comparison.csv`

Missingness is **not imputed during cleaning**. Stage 02 describes and standardises it; later modelling decisions handle analytical missingness only where justified.

### 2. Stage-specific key findings

Terminal output is deliberately concise and decision-oriented. Detailed tables still go to CSV, while the terminal shows the findings that determine whether the analyst should proceed.

Examples:

- Stage 03: identifier re-verification, patient linkage, pathway-anchor completeness and episode-to-spell collapse;
- Stage 04: patient-spine linkage, group-specific hospital-source coverage and demographic provenance;
- Stage 05: eligible denominators, exclusions, observation window and index QA;
- Stage 06: event-ledger counts, outcome rates, zero-event percentages;
- Stage 07: largest unadjusted SMDs, missingness, crude baseline/follow-up rates;
- Stage 08: positivity exclusions, common support, ATT ESS, SMD pass/fail, PSM balance and primary/sensitivity estimates;
- Stage 09: K diagnostics, phenotype sizes, stability, Cramér's V and sparse Sports-linked phenotype warnings;
- Stage 10: internal patient-level file flags and aggregate small-cell review flags.

### 3. Explicit stage handoff

Every successful stage ends with an exact `NEXT STEP` command. If a decision gate fails, the log instead tells the analyst to **stop, review and rerun the failed stage**.

### 4. Aggregate stage summaries

Each stage writes both JSON and Markdown to:

`outputs/audit/stage_summaries/`

These contain only aggregate status, key findings, warnings, QA paths and next command.

After a partial or complete run:

```bash
python scripts/review_audit_summary.py
```

creates:

- `outputs/audit/reviewer_summary.csv`
- `outputs/audit/reviewer_summary.md`

This gives a reviewer a complete run-status narrative without opening patient-level analytical files.

---

## Main methodological decision gates

### Cohort/index gate

Do not run the full analysis until the approved source coverage period and index semantics are confirmed.

### Positivity/overlap gate

Real-data geography/source support is recalculated. Exclusions are recalculated from the current TRE data.

### ATT balance gate

The primary comparative models are blocked when post-ATT measured balance fails the configured criterion:

`max |SMD| < 0.10`

If balance fails, the workflow writes all design diagnostics first, records `BLOCKED_AT_BALANCE_GATE`, and stops before substantive model interpretation.

### Sparse-event gate

Count-model results are labelled or withheld when configured minimum event counts are not met, particularly for Negative Binomial models.

### Clustering gate

K=4 is a prespecified report-facing candidate, not a forced solution. K=2–6 diagnostics are recomputed and K=4 must satisfy the configured minimum size/stability requirements.

### Disclosure gate

Stage 10 is a **pre-screen only**. Formal local TRE disclosure-control approval remains mandatory before any output leaves the secure environment.

---

## Recommended review order

1. `README.md`
2. `config/pipeline_tre.yaml`
3. `config/workflow_tre.yaml`
4. `config/clustering_tre.yaml`
5. `docs/REAL_TRE_SOURCE_REGISTER.md`
6. `docs/ANALYSIS_SPECIFICATION.md`
7. `docs/CODE_REVIEW_MAP.md`
8. `docs/AUDIT_TRAIL_AND_LOGGING.md`
9. `docs/TRE_STEP_BY_STEP_EXECUTION.md`
10. `src/bth_analysis/data_pipeline/cleaning.py`
11. `src/bth_analysis/analysis/propensity.py`
12. `src/bth_analysis/analysis/comparative.py`
13. `src/bth_analysis/analysis/clustering.py`

---

## Review sign-off checklist

Before TRE execution, confirm:

- [ ] Six real source filenames are correct.
- [ ] Source directory is correct for the approved workspace.
- [ ] Patient/event identifier candidate lists reflect source knowledge.
- [ ] Real extract start/end coverage dates are entered.
- [ ] `FirstMSKDate` analytical semantics are confirmed or revised.
- [ ] Emergency `MethodOfAdmission` patterns are checked against real source values.
- [ ] Propensity covariates remain pre-index only.
- [ ] ATT balance threshold remains appropriate.
- [ ] Primary/sensitivity model hierarchy is accepted.
- [ ] Clustering remains explicitly secondary/exploratory.
- [ ] Missingness classifications in `pipeline_tre.yaml` are source-supported.
- [ ] Formal disclosure-control process is understood and separate from Stage 10.

Once these items are agreed, the code should be ingressed and executed stage-by-stage.
