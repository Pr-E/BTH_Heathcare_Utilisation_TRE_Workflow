# Real TRE source register — Active Blackpool / BTH

This register records confirmed TRE source facts and the source semantics that still require confirmation.

## Six real source extracts

| Canonical key | Existing TRE key | Real filename | Role |
|---|---|---|---|
| `msk_wider` | `msk_without_sports` | `active_blackpool_msk_cohort_without_sports.csv` | Wider MSK pathway/referral cohort without Sports |
| `inpatient_wider` | `inpatient_msk` | `active_blackpool_inpatient_msk_.csv` | Wider-MSK inpatient episode extract |
| `ed_wider` | `msk_without_sports_ed` | `active_blackpool_msk_cohort_without_sports_ed.csv` | Wider-MSK ED attendance extract |
| `msk_sports` | `only_msk_sports` | `active_blackpool_only_msk_sports.csv` | Sports-linked BTH MSK pathway cohort |
| `inpatient_sports` | `inpatient_msk_sports` | `active_blackpool_inpatient_msk_sports.csv` | Sports-linked inpatient episode extract |
| `ed_sports` | `only_msk_sports_ed` | `active_blackpool_only_msk_sports_ed.csv` | Sports-linked ED attendance extract |

## Identifier conventions from the existing TRE workflow

The real TRE notebook/source registry used the following preferred identifier fields and candidate lists:

- Wider MSK cohort: preferred `sha256_hash`.
- Sports-linked MSK cohort: preferred `sha256_hash_nhs_no`, with `sha256_hash` as a candidate.
- Inpatient extracts: preferred `sha256_hash`, with `sha256_hash_nhs_no` as a candidate.
- ED extracts: the existing notebook records `sha256_hash_aeattendno` as the preferred patient-link candidate and `sha256_hash_nhs_no` as the preferred event candidate, with alternatives retained.

Because the ED hash labels are not self-validating, the production workflow **does not trust those labels blindly**. It re-checks each patient-ID candidate against the corresponding MSK cohort and writes aggregate overlap diagnostics. The attendance/event identifier is then selected from the remaining candidate(s) using uniqueness diagnostics. If linkage is ambiguous or there is no overlap, the workflow stops.

No raw SHA-256 values are written to QA summaries intended for review; only aggregate counts and selected column names are recorded.

## Real-data fields already seen in prior EDA/workflow

The existing real-data work has used or identified:

- MSK: `ReferralObservationId`, `FirstMSKReferralDate`, `FirstMSKDate`, `LastMSKDate`, pathway/referral fields and demographics.
- Inpatient: patient → `SpellID` → `EpisodeId`, admission/discharge and episode dates, `MethodOfAdmission`, specialty, diagnosis/procedure slots and demographics.
- ED: attendance/date fields, arrival/departure, LOS, referral/presenting-complaint fields, diagnostic/investigation/treatment slots and demographics.
- Demographics/confounding: sex, ethnicity, `PostcodeLAName`, IMD decile and age/DOB where available.

Prior QA also identified duplicate `ReferralObservationId` warnings, structural missingness, incomplete follow-up, and substantial missing/unstated ethnicity. These are therefore treated as QA targets rather than assumed away.

## Interpretation boundary retained for the real data

The Sports-linked MSK cohort supports a **Sports-linked BTH pathway membership** contrast. It does not by itself establish Active Blackpool programme start, attendance, completion, membership uptake or sustained activity.

The available MSK dates can provide a source-relative analytical anchor, but the final meaning of `FirstMSKDate` must be confirmed with the source owner before the comparative model is treated as the final real-data analysis.

## Values that must be generated from the current TRE data

The following are analysis results rather than fixed configuration and must be
recomputed from the current extracts:

- cohort and eligibility counts;
- healthcare-utilisation event rates;
- propensity overlap, weights and balance diagnostics;
- clustering centroids, selected K and phenotype membership;
- study coverage dates, which come from approved source documentation;
- programme-start semantics, which require genuine programme-specific evidence.

