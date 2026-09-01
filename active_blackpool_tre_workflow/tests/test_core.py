"""Small no-patient-data tests for TRE deployment verification."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bth_analysis.analysis.propensity import fit_propensity
from bth_analysis.analysis.clustering import _choose_k


def test_propensity_smoke():
    rng = np.random.default_rng(42)
    n = 300
    age = rng.normal(55, 14, n)
    baseline_ip = rng.poisson(0.4, n)
    geography = rng.choice(["Blackpool", "Fylde", "Wyre"], n, p=[0.45, 0.25, 0.30])
    logit = -2.0 + 0.015 * (age - 55) + 0.35 * baseline_ip + 0.20 * (geography == "Blackpool")
    p = 1 / (1 + np.exp(-logit))
    exposure = rng.binomial(1, p)
    # Ensure both groups for a deterministic smoke test.
    exposure[:5] = 1
    exposure[5:10] = 0

    df = pd.DataFrame({
        "PatientID": [f"P{i:04d}" for i in range(n)],
        "AnalysisEligibleFlag": 1,
        "ExposureFlag": exposure,
        "AnalysisGroup": np.where(exposure == 1, "Sports", "Wider"),
        "IndexDate": pd.Timestamp("2025-01-01"),
        "AgeAtIndex": age,
        "Sex": rng.choice(["F", "M"], n),
        "EthnicityNationalCodeDesc": rng.choice(["White British", "Not Stated"], n),
        "Index_of_Multiple_Deprivation_IMD_Decile": rng.integers(1, 11, n),
        "PostcodeLAName": geography,
        "BaselineEDCount": rng.poisson(0.3, n),
        "BaselineInpatientCount": baseline_ip,
        "BaselineEmergencyInpatientCount": rng.binomial(1, 0.08, n),
    })

    result = fit_propensity(
        df,
        covariates=[
            "AgeAtIndex", "Sex", "EthnicityNationalCodeDesc",
            "Index_of_Multiple_Deprivation_IMD_Decile", "PostcodeLAName",
            "IndexYear", "BaselineEDCount", "BaselineInpatientCount",
            "BaselineEmergencyInpatientCount",
        ],
        overlap_restrictions=[{"variable": "PostcodeLAName", "min_per_group": 1}],
        psm_ratio=1,
        require_full_psm_ratio=True,
    )

    assert result.data["PropensityScore"].between(0, 1).all()
    assert (result.data["ATTWeight"] >= 0).all()
    assert "abs_smd_att" in result.balance.columns
    assert not result.diagnostics.empty


def test_choose_k_prefers_best_eligible_silhouette():
    metrics = pd.DataFrame({
        "k": [2, 3, 4],
        "silhouette_score": [0.4, 0.6, 0.55],
        "adequate_cluster_size": [1, 1, 1],
        "mean_pairwise_stability_ari": [0.9, 0.95, 0.99],
    })
    k, _ = _choose_k(metrics, minimum_stability_ari=0.80)
    assert k == 3


def test_real_tre_identifier_resolution_uses_reference_overlap():
    """ED hash labels are resolved by observed overlap, not by field-name intuition."""
    from bth_analysis.data_pipeline.identifiers import resolve_identifier_plan

    tables = {
        "msk_wider": pd.DataFrame({
            "sha256_hash": ["P1", "P2", "P3"],
        }),
        "ed_wider": pd.DataFrame({
            # This candidate overlaps the cohort patient hashes and should be
            # selected by the cross-source verification rule.
            "sha256_hash_aeattendno": ["P1", "P1", "P2"],
            "sha256_hash_nhs_no": ["E1", "E2", "E3"],
        }),
    }
    cfg = {
        "msk_wider": {
            "type": "msk",
            "patient_id": "sha256_hash",
            "patient_id_candidates": ["sha256_hash"],
        },
        "ed_wider": {
            "type": "ed",
            "patient_id": "sha256_hash_aeattendno",
            "patient_id_candidates": ["sha256_hash_aeattendno", "sha256_hash_nhs_no"],
            "event_id": "sha256_hash_nhs_no",
            "event_id_candidates": ["sha256_hash_nhs_no", "sha256_hash_aeattendno"],
            "reference_cohort": "msk_wider",
        },
    }
    choices, audit = resolve_identifier_plan(tables, cfg)
    assert choices["ed_wider"].patient_id == "sha256_hash_aeattendno"
    assert choices["ed_wider"].event_id == "sha256_hash_nhs_no"
    selected = audit[(audit["table"] == "ed_wider") & (audit["selected"] == 1)]
    assert set(selected["identifier_role"]) == {"patient", "event"}


def test_missingness_audit_classifies_critical_and_group_difference():
    """Column QA must expose critical missingness and wider/sports differences."""
    from bth_analysis.data_pipeline.missingness import (
        column_missingness_table,
        compare_group_missingness,
    )

    config = {
        "missingness": {
            "group_difference_alert_pp": 5.0,
            "critical_columns_by_table": {
                "msk_wider": ["FirstMSKDate"],
                "msk_sports": ["FirstMSKDate"],
            },
            "rules": {},
        }
    }
    table_cfg = {"patient_id": "pid"}

    wider = pd.DataFrame(
        {
            "pid": ["W1", "W2", "W3", "W4"],
            "FirstMSKDate": ["2025-01-01", None, "2025-01-03", "2025-01-04"],
            "optional": [None, None, "x", "x"],
        }
    )
    sports = pd.DataFrame(
        {
            "pid": ["S1", "S2", "S3", "S4"],
            "FirstMSKDate": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            "optional": [None, None, None, None],
        }
    )

    w = column_missingness_table(wider, "msk_wider", table_cfg, config)
    s = column_missingness_table(sports, "msk_sports", table_cfg, config)
    all_miss = pd.concat([w, s], ignore_index=True)
    comp = compare_group_missingness(all_miss, config)

    first = w[w["column"] == "FirstMSKDate"].iloc[0]
    assert first["classification"] == "CRITICAL"
    assert first["status"] == "FAIL"
    optional = comp[comp["column"] == "optional"].iloc[0]
    assert optional["flag_group_difference"]
    assert optional["abs_difference_pp"] == 50.0


def test_stage_summary_contains_only_aggregate_payload(tmp_path):
    """Audit writer should record stage findings and next step without row data."""
    from bth_analysis.audit import save_stage_summary

    path = save_stage_summary(
        tmp_path,
        stage_key="cleaning",
        stage_code="02",
        title="TEST",
        status="PASS",
        key_findings={"rows_output": 100, "critical_missing_fields_n": 0},
        qa_files=["outputs/qa/example.csv"],
        warnings=["aggregate test warning"],
    )
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert payload["patient_level_data_in_summary"] is False
    assert payload["key_findings"]["rows_output"] == 100
    assert payload["next_command"] == "python scripts/run_03_preprocessing.py"


def test_non_msk_identifier_requires_reference_cohort():
    """Hospital patient-ID verification must not run without an MSK reference cohort."""
    from bth_analysis.data_pipeline.identifiers import resolve_identifier_plan

    tables = {
        "msk_wider": pd.DataFrame({"pid": ["P1", "P2"]}),
        "ed_wider": pd.DataFrame({"pid": ["P1"], "event": ["E1"]}),
    }
    cfg = {
        "msk_wider": {"type": "msk", "patient_id": "pid"},
        "ed_wider": {
            "type": "ed", "patient_id": "pid", "event_id": "event",
            "patient_id_candidates": ["pid"], "event_id_candidates": ["event"],
        },
    }
    import pytest
    with pytest.raises(ValueError, match="reference_cohort is required"):
        resolve_identifier_plan(tables, cfg)


def test_pathway_anchor_keeps_one_intact_referral_row():
    """Anchor fields must come from the same earliest ordered referral row."""
    from bth_analysis.data_pipeline.preprocessing import make_pathway_anchor

    referrals = pd.DataFrame({
        "PatientID": ["P1", "P1"],
        "ReferralObservationId": [1, 2],
        "FirstMSKReferralDate": pd.to_datetime(["2025-01-01", "2025-02-01"]),
        "FirstMSKDate": pd.to_datetime([None, "2025-02-10"]),
        "LastMSKDate": pd.to_datetime(["2025-01-20", "2025-03-01"]),
    })
    anchor = make_pathway_anchor(referrals)
    row = anchor.iloc[0]
    assert row["AnchorReferralObservationId"] == 1
    assert pd.isna(row["AnchorFirstMSKDate"])
    assert row["AnchorLastMSKDate"] == pd.Timestamp("2025-01-20")


def test_demographic_coalescing_records_field_level_provenance():
    """Resolved demographics should record the source used for each field."""
    from bth_analysis.data_pipeline.linkage import _first_demographics, _coalesce_demographics

    first = pd.DataFrame({"PatientID": ["P1"], "Age": [50], "Sex": [None]})
    second = pd.DataFrame({"PatientID": ["P1"], "Age": [51], "Sex": ["F"]})
    resolved = _coalesce_demographics([
        _first_demographics(first, "msk_sports"),
        _first_demographics(second, "msk_wider"),
    ])
    row = resolved.iloc[0]
    assert row["Age"] == 50
    assert row["AgeSource"] == "msk_sports"
    assert row["Sex"] == "F"
    assert row["SexSource"] == "msk_wider"
    assert row["ResolvedDemographicSource"] == "MULTIPLE_SOURCES"
