"""Primary and sensitivity comparative healthcare-utilisation models.

Analysis hierarchy
------------------
1. Crude baseline/follow-up rates are reported first.
2. Logistic-regression propensity scores define the measured-confounding design.
3. ATT weighting is the primary adjustment; standardised mean differences (SMDs)
   must meet the configured balance threshold before primary outcome modelling.
4. The primary estimand is the ATT-weighted group-by-period interaction from a
   Poisson GEE with log(person-time) offset.  It is exponentiated to a rate ratio
   of rate ratios (RRR).
5. Negative Binomial GEE checks distributional robustness when event counts are
   overdispersed.
6. 1:3 propensity-score matching (PSM) is a separate design sensitivity.

The models estimate adjusted comparative associations.  They are not causal
Active Blackpool treatment effects unless future real-data semantics and causal
assumptions are separately established.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from bth_analysis.analysis.propensity import fit_propensity
from bth_analysis.workflow import load_workflow_config, output_path
from bth_analysis.audit import (
    dataframe_preview,
    metric,
    save_stage_summary,
    section,
    stage_footer,
    stage_header,
)


OUTCOME_LABELS = {
    "FollowUpEDCount": "ED attendances",
    "FollowUpInpatientCount": "Inpatient admissions",
    "FollowUpEmergencyInpatientCount": "Emergency inpatient admissions",
    "FollowUpTotalHospitalCount": "Total hospital utilisation",
}


def _rate_result(
    fit,
    term,
    outcome,
    method,
    family,
    n,
    *,
    effect_scale,
    analysis_role,
):
    """Convert one fitted model coefficient into an auditable ratio, confidence interval and p-value row."""
    if term not in fit.params.index:
        return {
            "outcome": outcome,
            "method": method,
            "family": family,
            "n": n,
            "term": term,
            "estimate": np.nan,
            "ratio": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "status": "term_not_estimable",
            "effect_scale": effect_scale,
            "analysis_role": analysis_role,
        }

    beta = float(fit.params[term])
    se = float(fit.bse[term])
    return {
        "outcome": outcome,
        "method": method,
        "family": family,
        "n": n,
        "term": term,
        "estimate": beta,
        "ratio": float(np.exp(beta)),
        "ci_low": float(np.exp(beta - 1.96 * se)),
        "ci_high": float(np.exp(beta + 1.96 * se)),
        "p_value": float(fit.pvalues[term]),
        "status": "OK",
        "effect_scale": effect_scale,
        "analysis_role": analysis_role,
    }


def _alpha_mom(y):
    """Method-of-moments dispersion parameter used for NB-GEE sensitivity.

    Statsmodels' GEE NegativeBinomial family requires a supplied alpha rather
    than estimating it jointly in the same way as a full NB likelihood model.
    This pragmatic estimate is therefore treated as a *sensitivity formulation*,
    while Poisson GEE remains the primary prespecified count model.
    """
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(y) < 2:
        return 1e-6
    mean = float(y.mean())
    var = float(y.var(ddof=1))
    if mean <= 0:
        return 1e-6
    return max((var - mean) / (mean * mean), 1e-6)


def _family(y, family_name):
    """Return the configured Poisson or Negative Binomial count-family object."""
    if family_name == "NegativeBinomial":
        return sm.families.NegativeBinomial(alpha=_alpha_mom(y))
    return sm.families.Poisson()


def _fit_gee_rate(
    df,
    outcome,
    person_time,
    weight_col,
    group_col,
    method,
    family_name="Poisson",
    minimum_events_per_group=5,
    minimum_events_per_group_nb=20,
    *,
    analysis_role="secondary_followup",
):
    """Fit one follow-up-only GEE count-rate model with person-time offset and sparse-event gates."""
    work = df.copy()
    work = work[
        pd.to_numeric(work[outcome], errors="coerce").notna()
        & pd.to_numeric(work[person_time], errors="coerce").gt(0)
        & pd.to_numeric(work[weight_col], errors="coerce").gt(0)
        & work["ExposureFlag"].isin([0, 1])
    ].copy()

    if work.empty or work["ExposureFlag"].nunique() < 2:
        return {
            "outcome": outcome,
            "method": method,
            "family": family_name,
            "n": len(work),
            "status": "insufficient_data",
            "effect_scale": "follow-up rate ratio",
            "analysis_role": analysis_role,
        }

    event_by_group = work.groupby("ExposureFlag")[outcome].sum().to_dict()
    required_events = (
        max(int(minimum_events_per_group), int(minimum_events_per_group_nb))
        if family_name == "NegativeBinomial"
        else int(minimum_events_per_group)
    )
    if any(
        float(event_by_group.get(group, 0)) < float(required_events)
        for group in (0, 1)
    ):
        return {
            "outcome": outcome,
            "method": method,
            "family": family_name,
            "n": len(work),
            "status": ("sparse_events_for_negative_binomial" if family_name == "NegativeBinomial" else "sparse_events"),
            "events_comparison": float(event_by_group.get(0, 0)),
            "events_exposed": float(event_by_group.get(1, 0)),
            "effect_scale": "follow-up rate ratio",
            "analysis_role": analysis_role,
        }

    work = work.reset_index(drop=True)
    y = pd.to_numeric(work[outcome], errors="coerce").astype(float).reset_index(drop=True)
    X = pd.DataFrame({
        "const": np.ones(len(work), dtype=float),
        "ExposureFlag": work["ExposureFlag"].astype(float).to_numpy(),
    })
    offset = np.log(pd.to_numeric(work[person_time], errors="coerce").astype(float).to_numpy())
    weights = pd.to_numeric(work[weight_col], errors="coerce").astype(float).to_numpy()

    try:
        fit = sm.GEE(
            y,
            X,
            groups=work[group_col].astype(str).to_numpy(),
            family=_family(y, family_name),
            cov_struct=sm.cov_struct.Independence(),
            offset=offset,
            weights=weights,
        ).fit(maxiter=200)
        return _rate_result(
            fit,
            "ExposureFlag",
            outcome,
            method,
            family_name,
            len(work),
            effect_scale="follow-up rate ratio",
            analysis_role=analysis_role,
        )
    except Exception as exc:
        return {
            "outcome": outcome,
            "method": method,
            "family": family_name,
            "n": len(work),
            "status": f"model_failed:{type(exc).__name__}",
            "effect_scale": "follow-up rate ratio",
            "analysis_role": analysis_role,
        }


def _prepost_stack(df, followup_outcome):
    """Reshape patient outcomes into repeated baseline/follow-up rows for comparative pre/post modelling."""
    stem = followup_outcome.removeprefix("FollowUp").removesuffix("Count")
    baseline_col = f"Baseline{stem}Count"
    follow_col = f"FollowUp{stem}Count"

    required = [
        baseline_col,
        follow_col,
        "BaselinePersonYears",
        "FollowUpPersonYears",
    ]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    base = df.copy()
    base["Period"] = "Baseline"
    base["Post"] = 0
    base["Count"] = base[baseline_col]
    base["PersonYears"] = base["BaselinePersonYears"]

    post = df.copy()
    post["Period"] = "Follow-up"
    post["Post"] = 1
    post["Count"] = post[follow_col]
    post["PersonYears"] = post["FollowUpPersonYears"]

    stacked = pd.concat([base, post], ignore_index=True)
    stacked["ExposurePost"] = stacked["ExposureFlag"].astype(float) * stacked["Post"].astype(float)
    return stacked


def _fit_prepost(
    df,
    outcome,
    weight_col,
    cluster_col,
    method,
    family_name="Poisson",
    minimum_events_per_group=5,
    *,
    analysis_role="primary_prepost",
):
    """Fit the group-by-period GEE model whose interaction is the comparative rate-ratio-of-rate-ratios."""
    stacked = _prepost_stack(df, outcome)
    if stacked.empty:
        return {
            "outcome": outcome,
            "method": method,
            "family": family_name,
            "n": 0,
            "status": "missing_columns",
            "effect_scale": "rate ratio of rate ratios (group × period)",
            "analysis_role": analysis_role,
        }

    required = [weight_col, cluster_col]
    if not all(c in stacked.columns for c in required):
        return {
            "outcome": outcome,
            "method": method,
            "family": family_name,
            "n": 0,
            "status": "missing_weight_or_cluster",
            "effect_scale": "rate ratio of rate ratios (group × period)",
            "analysis_role": analysis_role,
        }

    work = stacked[
        pd.to_numeric(stacked["PersonYears"], errors="coerce").gt(0)
        & pd.to_numeric(stacked[weight_col], errors="coerce").gt(0)
        & stacked["ExposureFlag"].isin([0, 1])
    ].copy()

    if "CommonSupportFlag" in work.columns and weight_col == "ATTWeight":
        work = work[work["CommonSupportFlag"].eq(1)].copy()

    if work.empty or work["ExposureFlag"].nunique() < 2:
        return {
            "outcome": outcome,
            "method": method,
            "family": family_name,
            "n": len(work),
            "status": "insufficient_data",
            "effect_scale": "rate ratio of rate ratios (group × period)",
            "analysis_role": analysis_role,
        }

    event_by_group = work.groupby("ExposureFlag")["Count"].sum().to_dict()
    if any(
        float(event_by_group.get(group, 0)) < float(minimum_events_per_group)
        for group in (0, 1)
    ):
        return {
            "outcome": outcome,
            "method": method,
            "family": family_name,
            "n": len(work),
            "status": "sparse_events",
            "events_comparison": float(event_by_group.get(0, 0)),
            "events_exposed": float(event_by_group.get(1, 0)),
            "effect_scale": "rate ratio of rate ratios (group × period)",
            "analysis_role": analysis_role,
        }

    work = work.reset_index(drop=True)
    y = pd.to_numeric(work["Count"], errors="coerce").astype(float).reset_index(drop=True)
    # Model terms:
    #   ExposureFlag = average baseline difference between pathway groups.
    #   Post         = baseline-to-follow-up change in the comparison group.
    #   ExposurePost= additional change in the Sports-linked group.
    # The exponentiated ExposurePost coefficient is the rate ratio of rate ratios.
    X = pd.DataFrame({
        "const": np.ones(len(work), dtype=float),
        "ExposureFlag": work["ExposureFlag"].astype(float).to_numpy(),
        "Post": work["Post"].astype(float).to_numpy(),
        "ExposurePost": work["ExposurePost"].astype(float).to_numpy(),
    })
    # Offset converts count modelling into rate modelling.  A patient observed
    # for 100 days therefore contributes less person-time than a patient observed
    # for a full year rather than being treated as if follow-up were equal.
    offset = np.log(pd.to_numeric(work["PersonYears"], errors="coerce").astype(float).to_numpy())
    weights = pd.to_numeric(work[weight_col], errors="coerce").astype(float).to_numpy()

    try:
        fit = sm.GEE(
            y,
            X,
            groups=work[cluster_col].astype(str).to_numpy(),
            family=_family(y, family_name),
            cov_struct=sm.cov_struct.Independence(),
            offset=offset,
            weights=weights,
        ).fit(maxiter=200)
        return _rate_result(
            fit,
            "ExposurePost",
            outcome,
            method,
            family_name,
            len(work),
            effect_scale="rate ratio of rate ratios (group × period)",
            analysis_role=analysis_role,
        )
    except Exception as exc:
        return {
            "outcome": outcome,
            "method": method,
            "family": family_name,
            "n": len(work),
            "status": f"model_failed:{type(exc).__name__}",
            "effect_scale": "rate ratio of rate ratios (group × period)",
            "analysis_role": analysis_role,
        }


def _crude_period_rates(df, outcomes):
    """Calculate unadjusted baseline/follow-up rates per person-time before propensity adjustment."""
    rows = []
    for outcome in outcomes:
        stem = outcome.removeprefix("FollowUp").removesuffix("Count")
        for exposure, sub in df.groupby("ExposureFlag"):
            group = sub["AnalysisGroup"].iloc[0] if "AnalysisGroup" in sub and len(sub) else str(exposure)
            for period in ("Baseline", "FollowUp"):
                count_col = f"{period}{stem}Count"
                py_col = f"{period}PersonYears"
                if count_col not in sub or py_col not in sub:
                    continue
                events = pd.to_numeric(sub[count_col], errors="coerce").sum()
                py = pd.to_numeric(sub[py_col], errors="coerce").sum()
                rows.append({
                    "outcome": outcome,
                    "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
                    "period": period,
                    "ExposureFlag": exposure,
                    "group": group,
                    "patients": len(sub),
                    "events": float(events),
                    "person_years": float(py),
                    "rate_per_person_year": events / py if py > 0 else np.nan,
                    "rate_per_100_person_years": events / py * 100 if py > 0 else np.nan,
                })
    return pd.DataFrame(rows)


def _crude_change_summary(period_rates: pd.DataFrame) -> pd.DataFrame:
    """Summarise the unadjusted difference in baseline-to-follow-up change between groups."""
    if period_rates.empty:
        return pd.DataFrame()
    rows = []
    for outcome, sub in period_rates.groupby("outcome"):
        values = {}
        for exposure in (0, 1):
            x = sub[sub["ExposureFlag"].eq(exposure)].set_index("period")
            b = float(x.loc["Baseline", "rate_per_100_person_years"]) if "Baseline" in x.index else np.nan
            f = float(x.loc["FollowUp", "rate_per_100_person_years"]) if "FollowUp" in x.index else np.nan
            values[exposure] = {
                "baseline": b,
                "followup": f,
                "change": f - b if np.isfinite(b) and np.isfinite(f) else np.nan,
                "ratio": f / b if np.isfinite(b) and b > 0 and np.isfinite(f) else np.nan,
            }
        rows.append({
            "outcome": outcome,
            "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
            "comparison_baseline_rate_per_100py": values[0]["baseline"],
            "comparison_followup_rate_per_100py": values[0]["followup"],
            "comparison_absolute_change_per_100py": values[0]["change"],
            "sports_baseline_rate_per_100py": values[1]["baseline"],
            "sports_followup_rate_per_100py": values[1]["followup"],
            "sports_absolute_change_per_100py": values[1]["change"],
            "difference_in_absolute_change_per_100py": (
                values[1]["change"] - values[0]["change"]
                if np.isfinite(values[1]["change"]) and np.isfinite(values[0]["change"])
                else np.nan
            ),
            "crude_rate_ratio_of_rate_ratios": (
                values[1]["ratio"] / values[0]["ratio"]
                if np.isfinite(values[1]["ratio"]) and np.isfinite(values[0]["ratio"]) and values[0]["ratio"] > 0
                else np.nan
            ),
        })
    return pd.DataFrame(rows)


def _plot_love(balance: pd.DataFrame, path: Path, threshold: float) -> None:
    """Create and save the corresponding aggregate diagnostic/reporting figure."""
    if balance.empty:
        return
    x = balance.copy()
    x["max_abs"] = x[["abs_smd_unweighted", "abs_smd_att", "abs_smd_psm"]].max(axis=1, skipna=True)
    x = x.sort_values("max_abs").tail(25)
    y = np.arange(len(x))
    fig, ax = plt.subplots(figsize=(10.5, max(6.2, len(x) * 0.34)))
    ax.scatter(x["abs_smd_unweighted"], y, label="Before adjustment", marker="o", s=38)
    ax.scatter(x["abs_smd_att"], y, label="ATT weighted", marker="s", s=38)
    if x["abs_smd_psm"].notna().any():
        ax.scatter(x["abs_smd_psm"], y, label="PS matched", marker="^", s=42)
    ax.axvline(threshold, linestyle="--", linewidth=1.3, color="#555555")
    ax.set_yticks(y)
    ax.set_yticklabels(x["feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False), fontsize=8)
    ax.set_xlabel("Absolute standardised mean difference")
    ax.set_title("Measured baseline balance before and after propensity adjustment", loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_ps_overlap(data: pd.DataFrame, path: Path) -> None:
    """Create and save the corresponding aggregate diagnostic/reporting figure."""
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    for flag, sub in data.groupby("ExposureFlag"):
        values = pd.to_numeric(sub["PropensityScore"], errors="coerce").dropna()
        if values.empty:
            continue
        label = sub["AnalysisGroup"].iloc[0] if "AnalysisGroup" in sub else str(flag)
        ax.hist(values, bins=35, density=True, histtype="step", linewidth=2.2, label=label)
    ax.set_title("Propensity-score overlap in the primary design population", loc="left", fontweight="bold")
    ax.set_xlabel("Estimated probability of Sports-linked pathway membership")
    ax.set_ylabel("Density")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_weight_distribution(data: pd.DataFrame, path: Path) -> None:
    """Create and save the corresponding aggregate diagnostic/reporting figure."""
    controls = data[
        data["ExposureFlag"].eq(0)
        & data["CommonSupportFlag"].eq(1)
        & pd.to_numeric(data["ATTWeight"], errors="coerce").gt(0)
    ].copy()
    if controls.empty:
        return
    w = pd.to_numeric(controls["ATTWeight"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    ax.hist(w, bins=40, histtype="stepfilled", alpha=0.55)
    ax.axvline(w.quantile(0.99), linestyle="--", linewidth=1.3, color="#555555", label=f"99th percentile = {w.quantile(0.99):.3f}")
    ax.set_title("ATT weights among supported Wider MSK comparison patients", loc="left", fontweight="bold")
    ax.set_xlabel("ATT weight")
    ax.set_ylabel("Patients")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_primary_forest(results: pd.DataFrame, path: Path) -> None:
    """Create and save the corresponding aggregate diagnostic/reporting figure."""
    x = results[
        results["method"].eq("ATT comparative pre/post")
        & results["family"].eq("Poisson")
        & results["status"].eq("OK")
    ].copy()
    if x.empty:
        return
    x["label"] = x["outcome"].map(OUTCOME_LABELS).fillna(x["outcome"])
    x = x.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(x))
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    err_low = x["ratio"] - x["ci_low"]
    err_high = x["ci_high"] - x["ratio"]
    ax.errorbar(x["ratio"], y, xerr=[err_low, err_high], fmt="o", capsize=4, linewidth=1.7)
    ax.axvline(1.0, linestyle="--", linewidth=1.3, color="#555555")
    ax.set_yticks(y)
    ax.set_yticklabels(x["label"])
    ax.set_xlabel("Adjusted rate ratio of rate ratios (95% CI)")
    ax.set_title("ATT-adjusted comparative pre/post associations", loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_comparative(
    config_path: str | Path = "config/workflow_tre.yaml",
) -> dict[str, pd.DataFrame]:
    """Run propensity design, balance gates and the primary/sensitivity comparative models."""
    cfg = load_workflow_config(config_path)
    analysis_dir = output_path(cfg, "analysis_dir")
    out_dir = output_path(cfg, "comparative_dir")
    figure_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    stage_header(
        "08",
        "PROPENSITY DESIGN + COMPARATIVE PRE/POST MODELLING",
        purpose=(
            "Estimate measured selection into the Sports-linked pathway using logistic-regression propensity scores; "
            "apply structural positivity/common-support restrictions, ATT weighting and SMD balance checks; "
            "run 1:3 PSM as a design sensitivity; then estimate baseline-to-follow-up rate changes using "
            "Poisson GEE with a log(person-time) offset and Negative Binomial sensitivity."
        ),
        inputs=[analysis_dir / "patient_outcomes.csv"],
        outputs=[out_dir, figure_dir],
    )

    df = pd.read_csv(analysis_dir / "patient_outcomes.csv", low_memory=False)
    eligible = df[df["AnalysisEligibleFlag"].eq(1)].copy()

    group_ready = bool(cfg["cohort"].get("analysis_group_semantics_confirmed_for_workflow", False))
    index_ready = bool(cfg["cohort"].get("analytical_index_semantics_confirmed_for_workflow", False))
    semantic_gate_override = bool(cfg["comparative"].get("allow_unconfirmed_groups", False))

    if not (group_ready and index_ready) and not semantic_gate_override:
        raise RuntimeError(
            "Comparative analysis is blocked because the analysis-group and/or "
            "analytical-index semantics are not confirmed for this workflow."
        )

    ps_cfg = cfg["propensity"]
    ps = fit_propensity(
        eligible,
        covariates=ps_cfg["covariates"],
        psm_ratio=int(ps_cfg.get("psm_ratio", 3)),
        caliper_sd_logit=float(ps_cfg.get("caliper_sd_logit", 0.2)),
        random_seed=int(ps_cfg.get("random_seed", 42)),
        overlap_restrictions=ps_cfg.get("overlap_restrictions", []),
        require_full_psm_ratio=bool(ps_cfg.get("require_full_psm_ratio", True)),
    )

    balance_threshold = float(ps_cfg.get("balance_abs_smd_threshold", 0.10))
    max_att = float(ps.balance["abs_smd_att"].max()) if not ps.balance.empty else np.nan
    max_psm = (
        float(ps.balance["abs_smd_psm"].max())
        if not ps.balance.empty and ps.balance["abs_smd_psm"].notna().any()
        else np.nan
    )
    att_balance_pass = bool(max_att <= balance_threshold) if np.isfinite(max_att) else False
    psm_balance_pass = bool(max_psm <= balance_threshold) if np.isfinite(max_psm) else False

    ps.diagnostics = pd.concat([
        ps.diagnostics,
        pd.DataFrame([
            {"metric": "balance_abs_smd_threshold", "value": balance_threshold},
            {"metric": "att_balance_pass", "value": float(att_balance_pass)},
            {"metric": "psm_balance_pass", "value": float(psm_balance_pass)},
            {"metric": "analysis_group_semantics_ready", "value": float(group_ready)},
            {"metric": "analytical_index_semantics_ready", "value": float(index_ready)},
            {
                "metric": "programme_exposure_semantics_confirmed",
                "value": float(bool(cfg["cohort"].get("programme_exposure_semantics_confirmed", False))),
            },
            {
                "metric": "index_is_programme_start",
                "value": float(bool(cfg["cohort"].get("index_is_programme_start", False))),
            },
        ]),
    ], ignore_index=True)

    if (
        not att_balance_pass
        and cfg["comparative"].get("block_primary_models_if_att_balance_fails", False)
    ):
        # Persist the failed design diagnostics *before* stopping.  This makes a
        # failed real-data design auditable and gives the analyst the evidence
        # needed to revise the covariate/overlap specification rather than seeing
        # only a traceback.
        ps.data.drop(columns=["_DesignRow"], errors="ignore").to_csv(
            out_dir / "propensity_scored_population.csv", index=False
        )
        ps.balance.to_csv(out_dir / "propensity_balance.csv", index=False)
        ps.diagnostics.to_csv(out_dir / "propensity_diagnostics.csv", index=False)
        ps.overlap_audit.to_csv(out_dir / "design_overlap_audit.csv", index=False)
        ps.weight_diagnostics.to_csv(out_dir / "att_weight_diagnostics.csv", index=False)
        ps.ps_distribution.to_csv(out_dir / "propensity_score_distribution.csv", index=False)
        ps.model_terms.to_csv(out_dir / "propensity_logistic_model_terms.csv", index=False)
        ps.matched.drop(columns=["_DesignRow"], errors="ignore").to_csv(
            out_dir / "psm_matched_population.csv", index=False
        )
        _plot_love(ps.balance, figure_dir / "propensity_balance_love_plot.png", balance_threshold)
        _plot_ps_overlap(ps.data, figure_dir / "propensity_score_overlap.png")
        _plot_weight_distribution(ps.data, figure_dir / "att_weight_distribution.png")

        audit_dir = output_path(cfg, "audit_dir")
        fail_summary = save_stage_summary(
            audit_dir,
            stage_key="comparative",
            stage_code="08",
            title="Propensity design + comparative pre/post modelling",
            status="BLOCKED_AT_BALANCE_GATE",
            key_findings={
                "att_balance_pass": False,
                "maximum_abs_smd_att": max_att,
                "balance_threshold": balance_threshold,
                "psm_balance_pass": psm_balance_pass,
                "maximum_abs_smd_psm": max_psm,
            },
            qa_files=[
                out_dir / "propensity_balance.csv",
                out_dir / "propensity_diagnostics.csv",
                out_dir / "design_overlap_audit.csv",
                out_dir / "att_weight_diagnostics.csv",
            ],
            warnings=[
                "Primary outcome models were deliberately NOT fitted because the prespecified ATT balance gate failed.",
                "Review positivity, common support, covariate coding and the propensity specification before rerunning Stage 08."
            ],
            next_command="Review outputs/comparative propensity diagnostics; revise configuration/design; rerun python scripts/run_08_comparative.py",
            config_path=config_path,
        )
        stage_footer(
            stage_key="comparative",
            audit_dir=audit_dir,
            summary_path=fail_summary,
            qa_files=[out_dir / "propensity_balance.csv", out_dir / "design_overlap_audit.csv"],
            warnings=["DO NOT proceed to outcome interpretation until ATT balance passes."],
            next_command="Review propensity design -> rerun python scripts/run_08_comparative.py",
        )
        raise RuntimeError(
            "ATT balance failed the configured absolute SMD threshold. Design QA has been saved; review the propensity design before fitting primary adjusted models."
        )

    outcomes = cfg["comparative"]["outcomes"]
    minimum_events = int(cfg["comparative"].get("minimum_events_per_group", 5))
    minimum_events_nb = int(cfg["comparative"].get("minimum_events_per_group_negative_binomial", 20))
    model_rows = []

    supported = ps.data[
        ps.data["CommonSupportFlag"].eq(1)
        & ps.data["ATTWeight"].gt(0)
    ].copy()

    run_nb = bool(cfg["comparative"].get("run_negative_binomial_sensitivity", True))
    run_psm_nb = bool(cfg["comparative"].get("run_psm_negative_binomial_sensitivity", False))
    run_prepost = bool(cfg["comparative"].get("run_comparative_prepost_primary", True))
    run_followup = bool(cfg["comparative"].get("run_followup_adjusted_secondary", True))
    run_psm_prepost = bool(cfg["comparative"].get("run_psm_prepost_sensitivity", True))

    for outcome in outcomes:
        # PRIMARY: weighted comparative change from baseline to follow-up.
        if run_prepost:
            model_rows.append(_fit_prepost(
                supported, outcome, "ATTWeight", "PatientID",
                "ATT comparative pre/post", "Poisson",
                minimum_events_per_group=minimum_events,
                analysis_role="primary_prepost",
            ))
            if run_nb:
                model_rows.append(_fit_prepost(
                    supported, outcome, "ATTWeight", "PatientID",
                    "ATT comparative pre/post", "NegativeBinomial",
                    minimum_events_per_group=minimum_events,
                    analysis_role="primary_prepost_nb_sensitivity",
                ))

        # COMPLEMENTARY: adjusted follow-up rate comparison.
        if run_followup:
            model_rows.append(_fit_gee_rate(
                supported, outcome, "FollowUpPersonYears", "ATTWeight", "PatientID",
                "ATT weighted follow-up", "Poisson",
                minimum_events_per_group=minimum_events,
                minimum_events_per_group_nb=minimum_events_nb,
                analysis_role="secondary_followup",
            ))
            if run_nb:
                model_rows.append(_fit_gee_rate(
                    supported, outcome, "FollowUpPersonYears", "ATTWeight", "PatientID",
                    "ATT weighted follow-up", "NegativeBinomial",
                    minimum_events_per_group=minimum_events,
                    minimum_events_per_group_nb=minimum_events_nb,
                    analysis_role="secondary_followup_nb_sensitivity",
                ))

        # MATCHING sensitivity: follow-up and optionally pre/post.
        if not ps.matched.empty:
            model_rows.append(_fit_gee_rate(
                ps.matched, outcome, "FollowUpPersonYears", "PSMWeight", "MatchSetID",
                "PSM follow-up sensitivity", "Poisson",
                minimum_events_per_group=minimum_events,
                minimum_events_per_group_nb=minimum_events_nb,
                analysis_role="psm_followup_sensitivity",
            ))
            if run_psm_nb:
                model_rows.append(_fit_gee_rate(
                    ps.matched, outcome, "FollowUpPersonYears", "PSMWeight", "MatchSetID",
                    "PSM follow-up sensitivity", "NegativeBinomial",
                    minimum_events_per_group=minimum_events,
                    minimum_events_per_group_nb=minimum_events_nb,
                    analysis_role="psm_followup_nb_sensitivity",
                ))
            if run_psm_prepost:
                model_rows.append(_fit_prepost(
                    ps.matched, outcome, "PSMWeight", "MatchSetID",
                    "PSM comparative pre/post sensitivity", "Poisson",
                    minimum_events_per_group=minimum_events,
                    analysis_role="psm_prepost_sensitivity",
                ))
                if run_psm_nb:
                    model_rows.append(_fit_prepost(
                        ps.matched, outcome, "PSMWeight", "MatchSetID",
                        "PSM comparative pre/post sensitivity", "NegativeBinomial",
                        minimum_events_per_group=minimum_events,
                        analysis_role="psm_prepost_nb_sensitivity",
                    ))

    results = pd.DataFrame(model_rows)
    results["analysis_group_definition"] = cfg["cohort"].get(
        "analysis_group_definition", "sports_linked_bth_vs_wider_non_sports"
    )
    results["index_strategy"] = cfg["cohort"].get("index_strategy", "source_relative_first_msk")
    results["index_is_programme_start"] = bool(cfg["cohort"].get("index_is_programme_start", False))
    results["programme_exposure_semantics_confirmed"] = bool(
        cfg["cohort"].get("programme_exposure_semantics_confirmed", False)
    )
    results["att_balance_pass"] = att_balance_pass
    results["psm_balance_pass"] = psm_balance_pass
    results["max_abs_smd_att"] = max_att
    results["max_abs_smd_psm"] = max_psm
    results["interpretation"] = (
        "Adjusted association between Sports-linked BTH pathway membership and healthcare utilisation relative to the Wider MSK comparison. "
        "For comparative pre/post models, the interaction estimates whether the baseline-to-follow-up rate change differs between groups. "
        "The analytical index is not a confirmed programme-start date; estimates are not causal Active Blackpool treatment effects."
    )

    period_rates = _crude_period_rates(eligible, outcomes)
    crude_change = _crude_change_summary(period_rates)

    # Core outputs.
    ps.data.drop(columns=["_DesignRow"], errors="ignore").to_csv(out_dir / "propensity_scored_population.csv", index=False)
    ps.balance.to_csv(out_dir / "propensity_balance.csv", index=False)
    ps.diagnostics.to_csv(out_dir / "propensity_diagnostics.csv", index=False)
    ps.overlap_audit.to_csv(out_dir / "design_overlap_audit.csv", index=False)
    ps.weight_diagnostics.to_csv(out_dir / "att_weight_diagnostics.csv", index=False)
    ps.ps_distribution.to_csv(out_dir / "propensity_score_distribution.csv", index=False)
    ps.model_terms.to_csv(out_dir / "propensity_logistic_model_terms.csv", index=False)
    ps.matched.drop(columns=["_DesignRow"], errors="ignore").to_csv(out_dir / "psm_matched_population.csv", index=False)
    results.to_csv(out_dir / "comparative_results.csv", index=False)
    period_rates.to_csv(out_dir / "crude_period_rates.csv", index=False)
    crude_change.to_csv(out_dir / "crude_comparative_change.csv", index=False)

    # Compact aggregate comparison table for rapid reviewer audit.
    primary_results = results[
        results["method"].eq("ATT comparative pre/post")
        & results["family"].eq("Poisson")
    ].copy()
    if not primary_results.empty:
        primary_results["outcome_label"] = primary_results["outcome"].map(OUTCOME_LABELS).fillna(primary_results["outcome"])
    primary_results.to_csv(out_dir / "primary_comparative_key_findings.csv", index=False)

    # Essential diagnostic/report figures.
    _plot_love(ps.balance, figure_dir / "propensity_balance_love_plot.png", balance_threshold)
    _plot_ps_overlap(ps.data, figure_dir / "propensity_score_overlap.png")
    _plot_weight_distribution(ps.data, figure_dir / "att_weight_distribution.png")
    _plot_primary_forest(results, figure_dir / "primary_prepost_forest.png")

    # Convert the diagnostic long table into a dictionary for concise printing.
    diag_lookup = (
        ps.diagnostics.drop_duplicates("metric", keep="last").set_index("metric")["value"].to_dict()
        if not ps.diagnostics.empty
        else {}
    )

    section("STAGE 08 DESIGN FINDINGS")
    metric("eligible patients entering propensity design", f"{int(diag_lookup.get('eligible_input_n', len(eligible))):,}")
    metric("structural-overlap exclusions", f"{int(diag_lookup.get('overlap_excluded_n', 0)):,}")
    metric("structural-overlap excluded Sports-linked", f"{int(diag_lookup.get('overlap_excluded_exposed_n', 0)):,}")
    metric("structural-overlap excluded Wider MSK", f"{int(diag_lookup.get('overlap_excluded_comparison_n', 0)):,}")
    metric("common-support lower propensity", f"{float(diag_lookup.get('common_support_lower', np.nan)):.6f}")
    metric("common-support upper propensity", f"{float(diag_lookup.get('common_support_upper', np.nan)):.6f}")
    metric("supported Sports-linked", f"{int(diag_lookup.get('supported_exposed_n', 0)):,}")
    metric("supported Wider MSK", f"{int(diag_lookup.get('supported_comparison_n', 0)):,}")
    metric("ATT control effective sample size", f"{float(diag_lookup.get('att_control_effective_sample_size', np.nan)):.1f}")
    metric("ATT max |SMD|", f"{max_att:.4f}")
    metric("ATT balance threshold", f"{balance_threshold:.3f}")
    metric("ATT balance PASS", att_balance_pass)
    metric("PSM max |SMD|", f"{max_psm:.4f}" if np.isfinite(max_psm) else "NA")
    metric("PSM balance PASS", psm_balance_pass)
    metric("matched Sports-linked", f"{int(diag_lookup.get('matched_exposed_n', 0)):,}")
    metric("matched Wider MSK", f"{int(diag_lookup.get('matched_comparison_n', 0)):,}")
    metric("matched sets", f"{int(diag_lookup.get('matched_sets_n', 0)):,}")

    if not ps.overlap_audit.empty:
        print("\nStructural positivity / overlap audit:")
        dataframe_preview(
            ps.overlap_audit,
            columns=[
                "variable", "level", "comparison_n", "sports_linked_n",
                "minimum_required_per_group", "retained_for_primary_design", "reason",
            ],
            max_rows=30,
        )

    if not ps.balance.empty:
        print("\nLargest measured imbalances before and after adjustment:")
        balance_view = ps.balance.copy()
        balance_view["max_abs_for_sort"] = balance_view[["abs_smd_unweighted", "abs_smd_att"]].max(axis=1, skipna=True)
        dataframe_preview(
            balance_view.sort_values("max_abs_for_sort", ascending=False),
            columns=["feature", "abs_smd_unweighted", "abs_smd_att", "abs_smd_psm"],
            max_rows=15,
        )

    print("\nCrude baseline-to-follow-up comparison before propensity adjustment:")
    dataframe_preview(crude_change, max_rows=10)

    section("STAGE 08 PRIMARY / SENSITIVITY MODEL FINDINGS")
    dataframe_preview(
        results,
        columns=[
            "outcome", "method", "family", "analysis_role", "ratio",
            "ci_low", "ci_high", "p_value", "status",
        ],
        max_rows=40,
    )

    primary_ok = primary_results[primary_results["status"].eq("OK")].copy() if not primary_results.empty else pd.DataFrame()
    sparse_or_failed = results[~results["status"].eq("OK")].copy() if not results.empty else pd.DataFrame()
    metric("primary Poisson pre/post models estimated", len(primary_ok))
    metric("model rows sparse/failed/not estimable", len(sparse_or_failed))

    audit_dir = output_path(cfg, "audit_dir")
    summary_path = save_stage_summary(
        audit_dir,
        stage_key="comparative",
        stage_code="08",
        title="Propensity design + comparative pre/post modelling",
        status="PASS",
        key_findings={
            "eligible_input_n": int(diag_lookup.get("eligible_input_n", len(eligible))),
            "overlap_excluded_n": int(diag_lookup.get("overlap_excluded_n", 0)),
            "supported_exposed_n": int(diag_lookup.get("supported_exposed_n", 0)),
            "supported_comparison_n": int(diag_lookup.get("supported_comparison_n", 0)),
            "att_control_effective_sample_size": float(diag_lookup.get("att_control_effective_sample_size", np.nan)),
            "maximum_abs_smd_att": max_att,
            "att_balance_pass": att_balance_pass,
            "maximum_abs_smd_psm": max_psm,
            "psm_balance_pass": psm_balance_pass,
            "primary_model_rows_ok": len(primary_ok),
            "model_rows_non_ok": len(sparse_or_failed),
        },
        qa_files=[
            out_dir / "design_overlap_audit.csv",
            out_dir / "propensity_diagnostics.csv",
            out_dir / "propensity_balance.csv",
            out_dir / "att_weight_diagnostics.csv",
            out_dir / "propensity_logistic_model_terms.csv",
            out_dir / "primary_comparative_key_findings.csv",
            out_dir / "comparative_results.csv",
        ],
        warnings=[
            "Propensity methods address measured pre-index differences only; unmeasured confounding remains possible.",
            "The primary estimand is an adjusted comparative pre/post association, not a confirmed Active Blackpool treatment effect.",
            "A confidence interval crossing 1 is interpreted as statistically inconclusive even when the point estimate is below 1."
        ],
        config_path=config_path,
    )
    stage_footer(
        stage_key="comparative",
        audit_dir=audit_dir,
        summary_path=summary_path,
        qa_files=[out_dir / "primary_comparative_key_findings.csv", out_dir / "propensity_balance.csv"],
    )

    return {
        "propensity_population": ps.data,
        "balance": ps.balance,
        "diagnostics": ps.diagnostics,
        "overlap_audit": ps.overlap_audit,
        "weight_diagnostics": ps.weight_diagnostics,
        "ps_distribution": ps.ps_distribution,
        "propensity_model_terms": ps.model_terms,
        "matched": ps.matched,
        "comparative_results": results,
        "crude_period_rates": period_rates,
        "crude_comparative_change": crude_change,
    }
