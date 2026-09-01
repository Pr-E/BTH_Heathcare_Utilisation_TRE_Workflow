# Derived variable dictionary

This document records the most important analysis-layer fields created by the TRE workflow.

## Cohort / exposure fields

| Variable | Meaning |
|---|---|
| `PatientID` | Canonical pseudonymised patient identifier used for linkage. |
| `SportsLinkedBTHFlag` | Patient is present in the Sports-linked MSK source family. |
| `EligibleWiderNonSportsCandidateFlag` | Patient is present in Wider MSK and not present in Sports-linked MSK. |
| `ExposureFlag` | Working comparative group: 1 Sports-linked, 0 Wider MSK. Not confirmed programme treatment by default. |
| `AnalysisGroup` | Human-readable label for `ExposureFlag`. |
| `ExposureDefinition` | Configured provenance/semantics label for the working comparison. |

## Index / observation fields

| Variable | Meaning |
|---|---|
| `IndexDate` | Source-relative `FirstMSKDate` analytical time origin under the current fallback strategy. |
| `IndexDateSource` | Which pathway source supplied the index. |
| `IndexDateType` | Human-readable configured index label. |
| `BaselineStartDate` | `IndexDate - 365 days` under default configuration. |
| `BaselineEndDate` | Index boundary; baseline events use `< IndexDate`. |
| `FollowUpStartDate` | Index date; follow-up events use `>= IndexDate`. |
| `PlannedFollowUpEndDate` | Index + 365 days under default configuration. |
| `FollowUpEndDate` | Minimum of planned end, study end and death. |
| `BaselineDaysAvailable` | Baseline days observable inside the configured study window. |
| `FollowUpDaysAvailable` | Follow-up days observable after censoring. |
| `BaselinePersonYears` | Baseline days / 365.25. |
| `FollowUpPersonYears` | Follow-up days / 365.25. |
| `BaselineCompleteFlag` | Full configured baseline is observable. |
| `FullFollowUpFlag` | Full configured follow-up is observable. |
| `AnalysisEligibleFlag` | Meets all configured cohort eligibility criteria. |

## Healthcare-utilisation outcomes

For each metric `ED`, `Inpatient`, `EmergencyInpatient`, `TotalHospital`:

| Pattern | Meaning |
|---|---|
| `Baseline<Metric>Count` | Number of events during baseline. |
| `FollowUp<Metric>Count` | Number of events during observed follow-up. |
| `Baseline<Metric>RatePerPY` | Baseline count / baseline person-years. |
| `FollowUp<Metric>RatePerPY` | Follow-up count / follow-up person-years. |

`TotalHospitalCount` is ED attendances plus inpatient admissions. Emergency inpatient admissions are a subset of inpatient admissions and are not added again to total hospital utilisation.

## Propensity fields

| Variable | Meaning |
|---|---|
| `IndexYear` | Calendar year derived from analytical index for design adjustment. |
| `PropensityScore` | Logistic-regression estimate of observed Sports-linked pathway membership from measured pre-index covariates. |
| `CommonSupportFlag` | Patient's PS lies in the empirical score range observed in both groups after structural overlap restriction. |
| `ATTWeight` | 1 for Sports-linked; `PS/(1-PS)` for supported Wider MSK comparison patients. |
| `MatchSetID` | Identifier for the 1:3 PSM matched set. |
| `PSMWeight` | Weight used within the matched sensitivity population. |
| `MatchDistance` | Absolute distance on the logit-propensity scale for matched controls. |

## Clustering fields

| Variable | Meaning |
|---|---|
| `UtilisationCluster` | Ordered K-means baseline healthcare-utilisation phenotype label. |
| `BaselineNonEmergencyInpatientRatePerPY` | Sensitivity-only derived rate = total inpatient rate - emergency inpatient rate, clipped at zero. |

Cluster labels are data-derived and must be interpreted from real-data centroids/profiles from the observed TRE centroids and profiles.
