# Ian Final Review Guide — Active Blackpool / BTH TRE Workflow

## Purpose

This package contains the final real-data workflow I am proposing to take into the BTH Trusted Research Environment (TRE).

The synthetic workflow has been used to develop and test the analytical process only. No synthetic results, fitted models or cluster centroids will be carried into the TRE. The real analysis will be rebuilt from the approved BTH extracts.

The main things I would like this review to confirm are:

1. Are the six real BTH source files, fields and identifier assumptions represented correctly?
2. Are the cohort, propensity-adjustment, comparative modelling and clustering methods appropriate?
3. Is the workflow sufficiently clear and auditable for another analyst to reproduce and review?

---

## Current interpretation

The analysis currently compares:

* **Sports-linked BTH pathway patients**
* **Wider MSK non-Sports-linked candidate patients**

Sports-linked pathway membership is **not currently treated as confirmed Active Blackpool programme participation**.

`FirstMSKDate` is being used as a source-relative analytical index and should not be interpreted as programme start unless its operational meaning is confirmed.

The primary analysis should therefore be described as:

> An adjusted comparison of baseline-to-follow-up healthcare-utilisation changes between the Sports-linked BTH pathway and a comparable Wider MSK population.

It should not currently be described as a causal Active Blackpool treatment effect.

Two items deliberately remain unresolved before the TRE run:

* approved extract start and end dates;
* confirmation of the meaning of `FirstMSKDate`.

The preflight stage is designed to stop the workflow if these remain unresolved.

---

## Real BTH source files

| Source                   | Expected TRE file                                   |
| ------------------------ | --------------------------------------------------- |
| Wider MSK cohort         | `active_blackpool_msk_cohort_without_sports.csv`    |
| Wider MSK inpatient      | `active_blackpool_inpatient_msk_.csv`               |
| Wider MSK ED             | `active_blackpool_msk_cohort_without_sports_ed.csv` |
| Sports-linked MSK cohort | `active_blackpool_only_msk_sports.csv`              |
| Sports-linked inpatient  | `active_blackpool_inpatient_msk_sports.csv`         |
| Sports-linked ED         | `active_blackpool_only_msk_sports_ed.csv`           |

Main files to review:

* `config/pipeline_tre.yaml`
* `docs/REAL_TRE_SOURCE_REGISTER.md`

The ED identifier logic is intentionally cautious. Candidate patient and event identifiers are rechecked against the real data using overlap and uniqueness evidence rather than relying only on column names.

---

## Workflow review points

### Data preparation

Stages 02–06 check:

* cleaning and missingness;
* identifier resolution;
* source linkage;
* patient-spine construction;
* cohort eligibility and exclusions;
* baseline/follow-up windows;
* ED, inpatient, emergency inpatient and total hospital-utilisation outcomes.

Missingness is described and classified during cleaning rather than automatically imputed.

### Descriptive analysis

Stage 07 summarises:

* population characteristics;
* missingness;
* baseline imbalance;
* crude baseline and follow-up utilisation.

### Comparative analysis

Stage 08 uses ATT propensity weighting as the primary adjustment approach.

Key requirements:

* propensity covariates remain **pre-index only**;
* structural and empirical overlap are recalculated from the real data;
* primary models proceed only when:

`max |SMD| < 0.10`

The primary model is an **ATT-weighted comparative pre/post Poisson GEE with a log-person-time offset**.

Sensitivity analyses include:

* Negative Binomial GEE;
* 1:3 propensity-score matching;
* follow-up-only supporting models.

The key estimand is the **difference in baseline-to-follow-up change between groups**, not simply the difference in follow-up rates.

### Clustering

Stage 09 remains secondary and exploratory.

Clusters are formed using baseline:

* ED utilisation rate;
* inpatient utilisation rate;
* emergency inpatient utilisation rate.

Demographics, pathway group and follow-up outcomes are used only afterwards to describe the clusters.

K=4 is the preferred reporting metric, while K=2–6 are reassessed on the real data to confirm that K=4 remains sufficiently stable, well separated and appropriately sized.

### Disclosure control

The workflow's release checks are an **internal pre-screen only**.

Formal TRE disclosure-control approval remains required before any output leaves the secure environment.

---

## Audit trail

Each stage produces:

* concise terminal findings;
* detailed CSV QA outputs;
* an aggregate JSON/Markdown stage summary;
* a clear `NEXT STEP` command.

Stage summaries are stored in:

`outputs/audit/stage_summaries/`

A consolidated review summary can be generated with:

```bash
python scripts/review_audit_summary.py
```

This produces:

* `outputs/audit/reviewer_summary.csv`
* `outputs/audit/reviewer_summary.md`

The aim is for the complete analytical run to be understandable without relying on undocumented notebook state.

---

## Suggested review order

1. `README.md`
2. `docs/IAN_FINAL_REVIEW_GUIDE.md`
3. `config/pipeline_tre.yaml`
4. `config/workflow_tre.yaml`
5. `config/clustering_tre.yaml`
6. `docs/REAL_TRE_SOURCE_REGISTER.md`
7. `docs/ANALYSIS_SPECIFICATION.md`
8. `docs/CODE_REVIEW_MAP.md`
9. `docs/AUDIT_TRAIL_AND_LOGGING.md`
10. `src/bth_analysis/analysis/propensity.py`
11. `src/bth_analysis/analysis/comparative.py`
12. `src/bth_analysis/analysis/clustering.py`

---

## Outcome of the review

Once reviewed, the intention is to make any final configuration changes required and then ingress the workflow into the TRE for the real-data run.
