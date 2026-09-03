# Analysis specification — TRE translation

## 1. Evaluation objective

Examine healthcare-utilisation trajectories among patients identified in the **Sports-linked BTH pathway** compared with a propensity-adjusted **Wider MSK non-Sports-linked population**, using linked routinely collected BTH data.

The current real-data fallback design compares 12-month pre-index utilisation with up to 12-month post-index utilisation. The working exposure is pathway membership, not confirmed programme treatment.

## 1A. Real TRE source families

The production workflow is configured to the six real BTH extracts already used in the TRE:

- `active_blackpool_msk_cohort_without_sports.csv`;
- `active_blackpool_inpatient_msk_.csv`;
- `active_blackpool_msk_cohort_without_sports_ed.csv`;
- `active_blackpool_only_msk_sports.csv`;
- `active_blackpool_inpatient_msk_sports.csv`;
- `active_blackpool_only_msk_sports_ed.csv`.

Healthcare patient identifiers are re-verified against the matching MSK cohort before linkage. 

## 2. Analysis population

### Working exposure group
Patients present in the approved Sports-linked MSK source family.

### Working comparison group
Patients present in the Wider MSK source family and not present in the Sports-linked source family.

### Required eligibility
- valid analytical index;
- index inside the configured study window;
- alive at index where required;
- age at index at least 16 years;
- complete 365-day baseline in the primary configuration;
- follow-up may be partial and is represented using observed person-time.

## 3. Analytical index

Default strategy: `source_relative_first_msk`.

- Sports-linked group: `SportsAnchorFirstMSKDate`;
- Wider MSK group: `WiderAnchorFirstMSKDate`.

This is a **source-relative FirstMSKDate analytical time origin**. It's not described as programme start.

## 4. Windows

- Baseline: `[IndexDate - 365 days, IndexDate)`
- Follow-up: `[IndexDate, min(IndexDate + 365 days, study end, death)]`

No event belongs to both windows.

## 5. Outcomes

Primary utilisation outcomes:

1. ED attendances;
2. inpatient admissions at spell/admission level;
3. emergency inpatient admissions;
4. total hospital utilisation = ED + inpatient admissions.

Rates are calculated per person-year and can be reported per 100 person-years.

## 6. Confounding design

### Propensity model
Binary logistic regression for observed Sports-linked pathway membership.

Pre-index covariates:
- AgeAtIndex;
- Sex;
- EthnicityNationalCodeDesc;
- IMD decile;
- PostcodeLAName;
- IndexYear;
- BaselineEDCount;
- BaselineInpatientCount;
- BaselineEmergencyInpatientCount.

### Missingness inside the propensity design
- numeric: median imputation with a missingness indicator;
- categorical: explicit `<Missing>` level;
- original patient fields are not overwritten.

### Positivity/overlap
1. structural level restriction for configured categorical variables;
2. empirical common support based on overlapping propensity-score ranges.

### Primary weighting estimand
ATT.

- Sports-linked weight = 1;
- comparison weight = `PS/(1-PS)` within common support.

### Balance criterion
Absolute SMD `<0.10` on encoded design features.

If ATT balance fails, the primary adjusted models are blocked by default.

### PSM sensitivity
- 1:3 nearest-neighbour matching;
- logit propensity distance;
- caliper = 0.2 pooled SD of logit PS;
- no replacement;
- full ratio required by default.

## 7. Primary comparative model

Patient-period dataset with two rows per eligible supported patient:

- baseline;
- follow-up.

Model terms:

```text
log(expected count) = β0 + β1(Group) + β2(Post) + β3(Group×Post) + log(person-time)
```

GEE cluster: patient ID for ATT pre/post analysis.

Primary estimand:

```text
exp(β3)
```

This is the adjusted **rate ratio of rate ratios**.

Interpretation:
- `<1`: Sports-linked group has a lower relative baseline-to-follow-up rate trajectory than the adjusted comparison;
- `=1`: no differential relative change;
- `>1`: Sports-linked group has a higher relative change.

Primary distribution: Poisson GEE.

Sensitivity: Negative Binomial GEE using a method-of-moments dispersion parameter.

## 8. Secondary follow-up-only model

ATT-weighted follow-up rate comparison. This answers a different question from the primary pre/post interaction and should remain secondary.

## 9. Crude results

Always report crude baseline and follow-up rates before adjusted model estimates. This makes baseline differences visible and prevents follow-up-only interpretation.

## 10. Clustering

Secondary exploratory baseline phenotype analysis.

Inputs only:
- BaselineEDRatePerPY;
- BaselineInpatientRatePerPY;
- BaselineEmergencyInpatientRatePerPY.

Preprocessing:
1. upper winsorisation at 99.5th percentile;
2. `log1p`;
3. StandardScaler;
4. KMeans.

Real-data diagnostics: K=2..6, silhouette, Calinski-Harabasz, Davies-Bouldin, minimum size/share and stability ARI.

Prespecified report-facing K=4 is retained only if real-data minimum size and stability criteria pass. Exposure and follow-up outcomes are used only after clustering for profiling.

## 11. Interpretation limits

The design controls measured baseline differences only. Residual/unmeasured confounding may remain, including clinical severity, functional limitation, motivation, clinician judgement and social factors not captured in the available data.

The current workflow therefore use language such as **adjusted association**, **comparative trajectory**, or **differential change**, not confirmed Active Blackpool causal effect.
