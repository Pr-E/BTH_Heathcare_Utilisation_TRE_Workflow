# Audit trail and terminal logging specification

## Objective

The workflow produces two complementary audit layers:

1. **Detailed TRE-internal QA/analysis tables** for technical review; and
2. **Compact aggregate terminal + stage-summary outputs** for straightforward handover and run auditing.

The terminal is not the sole evidence source. It surfaces the findings that determine whether it is safe to proceed, while detailed CSVs preserve the supporting diagnostics.

---

## Common terminal structure

Every stage uses the same layout:

```text
================================================================================
<stage code> | <title>
================================================================================
PURPOSE: ...
INPUTS:
  - ...
OUTPUTS:
  - ...

--------------------------------------------------------------------------------
KEY FINDINGS / DECISION GATE
--------------------------------------------------------------------------------
...

--------------------------------------------------------------------------------
AUDIT / HANDOFF
--------------------------------------------------------------------------------
Stage summary: ...
QA/output: ...
WARNINGS / INTERPRETATION BOUNDARIES:
  - ...

NEXT STEP
  python scripts/run_XX_....py
================================================================================
```

This makes terminal logs comparable across stages and easy to retain in internal run documentation.

---

## Stage summary files

Each stage writes:

```text
outputs/audit/stage_summaries/<stage>_summary.json
outputs/audit/stage_summaries/<stage>_summary.md
```

The summaries contain:

- UTC timestamp;
- stage key/code/title;
- status (`PASS`, `BLOCKED`, etc.);
- aggregate key findings;
- QA file paths;
- interpretation warnings;
- exact next command;
- explicit `patient_level_data_in_summary: false` marker.

These summaries are never populated with patient hashes, patient-level rows or free-text clinical content.

---

## Stage-specific key findings

### 00 Preflight

- checks run/pass/fail;
- mandatory blockers;
- configuration readiness table.

### 01 Ingestion

- source rows/table;
- missing required canonical columns;
- extra approved columns;
- schema/order status.

### 02 Cleaning

- rows after cleaning;
- removed blanks/duplicates/duplicate ED keys;
- resolved patient/event identifier column names;
- chronology/integrity anomalies;
- total missing-cell n/%;
- per-column missing n/% and classification;
- critical missingness;
- Sports vs Wider missingness percentage-point differences.

### 03 Preprocessing

- identifier re-verification;
- inpatient episode-to-spell reduction;
- processed row/spell/attendance counts;
- source-vs-derived timeframe agreement.

### 04 Linkage

- patient-spine n;
- pathway-group candidate n;
- source presence;
- MSK-to-hospital overlaps by source family.

### 05 Cohort/index

- eligible n by group;
- baseline/follow-up completeness;
- follow-up day distribution;
- study coverage;
- index strategy and semantics;
- exclusion counts.

### 06 Outcomes

- event ledger n;
- baseline/follow-up/outside event counts;
- crude outcome rates per 100 PY;
- zero-event percentages.

### 07 Descriptive

- largest unadjusted SMDs;
- key missingness differences;
- crude baseline/follow-up rates;
- crude changes;
- EDA review flags.

### 08 Comparative

- structural positivity exclusions;
- propensity common-support limits;
- ATT supported n and ESS;
- max post-ATT SMD and balance status;
- PSM balance;
- primary/sensitivity RRRs, CIs, p-values and sparse-model statuses.

### 09 Clustering

- K=2–6 metrics;
- selected K and selection mode;
- cluster prevalence and Sports/Wider n;
- stability/winsor/mutually-exclusive sensitivity ARIs;
- Cramér's V;
- sparse Sports-linked cluster warnings.


### 10 Release pre-screen

- internal patient-level files flagged;
- possible small-cell aggregate files;
- outputs requiring formal disclosure review.

---

## Run-level reproducibility metadata

The orchestrated workflow writes:

- configuration paths;
- SHA-256 digest of workflow configuration;
- SHA-256 digest of pipeline configuration;
- Python version;
- installed versions of the core analytical dependencies (pandas, NumPy, SciPy, scikit-learn, statsmodels, Matplotlib, PyYAML and joblib);
- platform;
- Git commit when available;
- UTC run timestamp;
- stage execution status and elapsed time.

Files:

```text
outputs/audit/tre_run_manifest.json
outputs/audit/stage_status.csv
```

This means a numerical output can be linked back to the exact configuration and software context that produced it.

---

## Failure behaviour

A failed stage records:

- `FAIL` in `stage_status.csv` when orchestrated;
- exception type/message;
- a traceback under `outputs/audit/failure_<stage>.txt`.

The traceback is internal TRE material. Functions should not print patient rows into exceptions or debug logs.

The most important analytical failure gate is Stage 08 ATT balance. If the configured balance threshold fails, the design diagnostics are saved first and primary outcome interpretation is blocked.

---

## Reviewer summary

After any partial or complete run:

```bash
python scripts/review_audit_summary.py
```

This reads only aggregate stage-summary JSON and builds a single reviewer-facing audit narrative.
