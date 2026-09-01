# TRE translation checklist

Use this as the formal handover checklist for the first real-data run.

## A. Source semantics

- [x] Working source contrast identified from real filenames: wider MSK without Sports versus Sports-linked MSK pathway.
- [ ] Operational meaning of the Sports-linked pathway record reviewed with the source owner for final reporting language.
- [ ] Sports-linked source is not being mislabelled as confirmed programme treatment.
- [ ] Approved real extract coverage start/end dates confirmed; the study window is taken from approved extract documentation.
- [ ] `FirstMSKReferralDate` meaning confirmed.
- [ ] `FirstMSKDate` meaning confirmed.
- [ ] `LastMSKDate` meaning confirmed.
- [ ] Sports Centre programme-start/attendance fields reviewed if available.
- [ ] Emergency admission coding reviewed with data dictionary.

## B. Schema/mapping

- [ ] All six configured real TRE filenames are present: wider MSK, wider inpatient, wider ED, Sports-linked MSK, Sports inpatient and Sports ED.
- [ ] `02_identifier_resolution_qa.csv` confirms patient-ID candidate overlap with the appropriate MSK reference cohort.
- [ ] `03_identifier_resolution_qa.csv` reproduces the same identifier selection after cleaning.
- [ ] ED attendance/event identifier is distinct from the selected patient identifier and has plausible uniqueness at attendance grain.
- [ ] inpatient `SpellID` and `EpisodeId` meanings confirmed.
- [ ] Any real-header aliases are entered in `pipeline_tre.yaml`; identity mappings need no code change.
- [ ] All required canonical fields are present; extra approved real-source columns are reviewed in ingestion QA rather than blocked.

## C. Cleaning/preprocessing QA

- [ ] blank rows reviewed.
- [ ] exact duplicates reviewed.
- [ ] duplicate ED keys reviewed.
- [ ] inpatient chronology issues reviewed.
- [ ] MSK chronology issues reviewed.
- [ ] episode-to-spell aggregation validated.
- [ ] diagnosis/procedure slot handling validated.
- [ ] pathway timeframe reconciliation reviewed.

## D. Linkage/cohort

- [ ] patient spine size plausible.
- [ ] cross-source overlap plausible.
- [ ] demographic source-priority rule accepted.
- [ ] index distribution reviewed by group and year.
- [ ] 365-day baseline rule confirmed.
- [ ] partial follow-up strategy confirmed.
- [ ] death censoring confirmed.
- [ ] age eligibility confirmed.
- [ ] final cohort flow documented.

## E. Outcomes

- [ ] ED counted once per attendance.
- [ ] inpatient counted once per spell/admission.
- [ ] emergency inpatient flag validated.
- [ ] no baseline/follow-up window overlap.
- [ ] person-time denominator validated.
- [ ] crude event totals/rates plausible.

## F. Confounding/propensity

- [ ] covariates are pre-index only.
- [ ] geographic positivity restrictions reviewed.
- [ ] empirical PS common support reviewed.
- [ ] supported exposed count acceptable.
- [ ] ATT weight tail acceptable.
- [ ] ATT effective comparison sample size acceptable.
- [ ] all ATT absolute SMDs <0.10 or design revised.
- [ ] PSM matched-set count acceptable.
- [ ] PSM absolute SMDs <0.10 or sensitivity not interpreted.

## G. Comparative models

- [ ] crude baseline/follow-up rates reviewed first.
- [ ] event-count thresholds satisfied.
- [ ] Poisson GEE model status OK.
- [ ] Negative Binomial sensitivity status reviewed.
- [ ] PSM sensitivity reviewed.
- [ ] CIs/p-values interpreted alongside magnitude/precision.
- [ ] wording remains adjusted association/comparative change.

## H. Clustering

- [ ] only baseline utilisation features used for cluster formation.
- [ ] K=2..6 diagnostics rerun on real data.
- [ ] K=4 passes size and stability criteria if retained.
- [ ] centroid labels reviewed clinically.
- [ ] Sports-linked cell sizes adequate for any cluster-specific trajectory commentary.
- [ ] cluster results kept descriptive.

## I. Disclosure/output

- [ ] patient-level files remain internal to TRE.
- [ ] release pre-screen run.
- [ ] small cells reviewed.
- [ ] formal TRE disclosure control completed.
- [ ] final outputs linked to run manifest/config hashes.
