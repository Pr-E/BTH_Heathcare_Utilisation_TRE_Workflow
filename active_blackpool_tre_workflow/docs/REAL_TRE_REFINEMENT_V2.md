# Real-TRE refinement v2

This revision incorporates genuine source-specific information from the existing Active Blackpool/BTH TRE workspace rather than generic placeholder filenames.

## Source changes

The production pipeline is now configured to the six real source files:

1. `active_blackpool_msk_cohort_without_sports.csv`
2. `active_blackpool_inpatient_msk_.csv`
3. `active_blackpool_msk_cohort_without_sports_ed.csv`
4. `active_blackpool_only_msk_sports.csv`
5. `active_blackpool_inpatient_msk_sports.csv`
6. `active_blackpool_only_msk_sports_ed.csv`

The canonical analysis keys remain stable, while `tre_dataset_key` records the names used in the existing TRE notebook.

## Identifier changes

The existing TRE workflow used several candidate SHA-256 fields and explicitly verified ED linkage by overlap with the MSK cohort. This logic is now formalised in `data_pipeline/identifiers.py`.

- MSK cohort identifiers are resolved first.
- inpatient and ED patient-ID candidates are checked against the corresponding MSK cohort.
- ED event IDs are resolved separately using uniqueness after the patient ID is selected.
- no-overlap or ambiguous patient-ID resolution blocks processing.
- only aggregate overlap/uniqueness statistics and column names are written to QA.

Identifier resolution is executed before identifier-dependent cleaning and repeated after cleaning before analytical preprocessing.

## Real versus synthetic boundaries

The synthetic simulator coverage dates have been removed from the production configuration. Real `study_start_date` and `study_end_date` are deliberately null until approved extract coverage dates are entered.

The real source names support the working contrast **Sports-linked BTH pathway versus Wider MSK without Sports**, so that source-group semantic flag is now confirmed for workflow construction. This does not establish programme treatment.

`FirstMSKDate` remains a source-relative analytical candidate only. Its final semantic confirmation remains a blocker.

## Real-data QA carried forward

The handover explicitly retains checks for issues already seen in real-data exploration:

- duplicate `ReferralObservationId` warnings;
- episode-versus-spell inpatient grain;
- ED attendance uniqueness;
- structural missingness;
- incomplete follow-up;
- missing / `Not Stated` ethnicity;
- source-versus-derived pathway timeframe reconciliation.

## Verification performed outside the TRE

- Python compilation: pass.
- Unit tests: 3/3 pass.
- Six-table mock integration smoke for ingestion → identifier resolution → cleaning → preprocessing: pass.

This verifies the translation mechanics only; real counts and analytical results must still be generated inside the TRE.
