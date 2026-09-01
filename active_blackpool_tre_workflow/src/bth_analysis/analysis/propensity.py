"""Propensity-score design for the primary TRE comparative analysis.

The propensity layer is a *design* step, not the outcome analysis.  Logistic
regression estimates each patient's probability of observed Sports-linked BTH
pathway membership using measured pre-index covariates only.  The resulting
score is then used for positivity/common-support checks, ATT weighting and a
1:3 propensity-score-matched sensitivity design.

Important interpretation boundary
---------------------------------
These methods address measured baseline differences only.  They do not remove
unmeasured confounding and they do not convert Sports-linked pathway membership
into confirmed Active Blackpool treatment exposure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PropensityResult:
    """Internal helper for PropensityResult; see the surrounding module comments for the analytical rationale."""
    data: pd.DataFrame
    balance: pd.DataFrame
    matched: pd.DataFrame
    diagnostics: pd.DataFrame
    overlap_audit: pd.DataFrame
    weight_diagnostics: pd.DataFrame
    ps_distribution: pd.DataFrame
    model_terms: pd.DataFrame


def _weighted_mean_var(x, w):
    """Return the weighted mean and variance used by balance diagnostics."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x, w = x[mask], w[mask]
    if len(x) == 0 or w.sum() <= 0:
        return np.nan, np.nan
    mean = np.average(x, weights=w)
    var = np.average((x - mean) ** 2, weights=w)
    return mean, var


def _smd(x, exposure, weights=None):
    """Calculate a standardised mean difference between Sports-linked and comparison groups."""
    x = np.asarray(x, dtype=float)
    exposure = np.asarray(exposure, dtype=int)
    if weights is None:
        weights = np.ones(len(x), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    mt, vt = _weighted_mean_var(x[exposure == 1], weights[exposure == 1])
    mc, vc = _weighted_mean_var(x[exposure == 0], weights[exposure == 0])
    denom = np.sqrt((vt + vc) / 2)
    if not np.isfinite(denom) or denom == 0:
        return 0.0 if np.isfinite(mt) and np.isfinite(mc) and np.isclose(mt, mc) else np.nan
    return float((mt - mc) / denom)


def _derive_design_covariates(df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Create design-only covariates without changing upstream pipeline outputs."""
    out = df.copy()
    if "IndexYear" in covariates and "IndexYear" not in out.columns:
        if "IndexDate" not in out.columns:
            raise ValueError("IndexYear was requested but IndexDate is unavailable.")
        out["IndexYear"] = pd.to_datetime(out["IndexDate"], errors="coerce").dt.year.astype(float)
    return out


def _normalised_level(series: pd.Series) -> pd.Series:
    """Convert categorical values to stable strings while retaining missingness explicitly."""
    return series.astype("string").fillna("<Missing>")


def _apply_overlap_restrictions(
    df: pd.DataFrame,
    restrictions: list[dict[str, Any]] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Restrict the propensity-design population to categorical levels represented
    in both groups. This is a design/positivity restriction, not data cleaning.
    """
    work = df.copy()
    audit_rows: list[dict[str, Any]] = []

    for rule in restrictions or []:
        variable = str(rule.get("variable", "")).strip()
        if not variable or variable not in work.columns:
            continue
        min_per_group = int(rule.get("min_per_group", 1))
        if min_per_group < 1:
            min_per_group = 1

        level_key = _normalised_level(work[variable])
        temp = pd.DataFrame({
            "ExposureFlag": work["ExposureFlag"].astype(int),
            "level": level_key,
        })
        counts = (
            temp.groupby(["level", "ExposureFlag"], dropna=False)
            .size()
            .unstack(fill_value=0)
        )
        for group in (0, 1):
            if group not in counts.columns:
                counts[group] = 0
        counts = counts[[0, 1]]
        retained_levels = set(
            counts.index[(counts[0] >= min_per_group) & (counts[1] >= min_per_group)]
        )

        for level, row in counts.iterrows():
            audit_rows.append({
                "variable": variable,
                "level": str(level),
                "comparison_n": int(row[0]),
                "sports_linked_n": int(row[1]),
                "minimum_required_per_group": min_per_group,
                "retained_for_primary_design": int(level in retained_levels),
                "reason": (
                    "adequate representation in both groups"
                    if level in retained_levels
                    else "insufficient representation in at least one group"
                ),
            })

        work = work[level_key.isin(retained_levels)].copy()
        if work.empty or work["ExposureFlag"].nunique() < 2:
            raise ValueError(
                f"Overlap restriction on {variable!r} removed one or both analysis groups."
            )

    return work, pd.DataFrame(audit_rows)


def _validate_preindex_covariates(covariates: list[str]) -> None:
    """Block accidental post-index/exposure leakage into the propensity model."""
    forbidden_exact = {
        "ExposureFlag",
        "AnalysisGroup",
        "UtilisationCluster",
        "ProgrammeStartDate",
    }
    forbidden = [
        c for c in covariates
        if c in forbidden_exact
        or c.startswith("FollowUp")
        or c.startswith("AnyFollowUp")
    ]
    if forbidden:
        raise ValueError(
            "Propensity covariates must be measured before the analytical index. "
            f"Remove post-index/exposure fields: {forbidden}"
        )


def _feature_matrix(df, covariates):
    """Build the leakage-safe propensity design matrix and preprocessing pipeline."""
    available = [c for c in covariates if c in df.columns]
    if not available:
        raise ValueError("None of the configured propensity covariates are available.")

    # IndexYear is deliberately treated as categorical so calendar-time adjustment
    # does not assume a linear change in pathway membership across study years.
    force_categorical = {"IndexYear"}
    numeric = [
        c for c in available
        if c not in force_categorical
        and (
            pd.api.types.is_numeric_dtype(df[c])
            or c.endswith("Count")
            or c in {
                "Age",
                "AgeAtIndex",
                "Index_of_Multiple_Deprivation_IMD_Decile",
            }
        )
    ]
    categorical = [c for c in available if c not in numeric]

    # Numeric missingness is handled transparently with median imputation plus
    # a missingness indicator.  The indicator prevents an imputed median from
    # silently implying that a missing measurement was genuinely observed.
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    # For categorical variables, preserve missingness as its own explicit level
    # rather than assigning the most common category.
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="<Missing>")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                drop=None,
                sparse_output=False,
            ),
        ),
    ])

    transformers = []
    if numeric:
        transformers.append(("num", numeric_pipe, numeric))
    if categorical:
        transformers.append(("cat", categorical_pipe, categorical))

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
    X = transformer.fit_transform(df[available])
    names = transformer.get_feature_names_out()
    return transformer, np.asarray(X, dtype=float), list(names), available


def _logit(p):
    """Convert propensity probabilities to the log-odds scale used for matching distances."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _greedy_match(
    df,
    ratio,
    caliper_sd_logit,
    random_seed,
    require_full_ratio=True,
):
    """Create nearest-neighbour propensity matched sets under the configured caliper and ratio."""
    work = df[df["CommonSupportFlag"].eq(1)].copy()
    treated = work[work["ExposureFlag"].eq(1)].copy()
    controls = work[work["ExposureFlag"].eq(0)].copy()

    if treated.empty or controls.empty:
        return pd.DataFrame()

    treated["_logit_ps"] = _logit(treated["PropensityScore"])
    controls["_logit_ps"] = _logit(controls["PropensityScore"])

    pooled_sd = float(
        np.nanstd(
            np.concatenate([
                treated["_logit_ps"].to_numpy(),
                controls["_logit_ps"].to_numpy(),
            ]),
            ddof=1,
        )
    )
    caliper = float(caliper_sd_logit) * pooled_sd
    if not np.isfinite(caliper) or caliper <= 0:
        caliper = np.inf

    ratio = max(int(ratio), 1)
    n_neighbors = min(len(controls), max(ratio * 20, ratio))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(controls[["_logit_ps"]].to_numpy())
    distances, indices = nn.kneighbors(treated[["_logit_ps"]].to_numpy())

    rng = np.random.default_rng(random_seed)
    treated_order = rng.permutation(len(treated))
    used_controls = set()
    match_rows = []
    match_set = 0

    treated_reset = treated.reset_index(drop=False)
    controls_reset = controls.reset_index(drop=False)

    for ti in treated_order:
        candidates = []
        for distance, ci in zip(distances[ti], indices[ti]):
            if distance > caliper:
                continue
            original_control_index = int(controls_reset.iloc[ci]["index"])
            if original_control_index in used_controls:
                continue
            candidates.append((float(distance), ci, original_control_index))
            if len(candidates) >= ratio:
                break

        if not candidates:
            continue
        if require_full_ratio and len(candidates) < ratio:
            continue

        match_set += 1
        treated_row = treated_reset.iloc[ti]
        match_rows.append({
            "OriginalIndex": int(treated_row["index"]),
            "PatientID": treated_row["PatientID"],
            "ExposureFlag": 1,
            "MatchSetID": match_set,
            "MatchDistance": 0.0,
            "PSMWeight": 1.0,
        })

        control_weight = 1.0 / len(candidates)
        for distance, ci, original_control_index in candidates:
            used_controls.add(original_control_index)
            control_row = controls_reset.iloc[ci]
            match_rows.append({
                "OriginalIndex": original_control_index,
                "PatientID": control_row["PatientID"],
                "ExposureFlag": 0,
                "MatchSetID": match_set,
                "MatchDistance": distance,
                "PSMWeight": control_weight,
            })

    if not match_rows:
        return pd.DataFrame()

    match_map = pd.DataFrame(match_rows)
    matched = match_map.merge(
        df.reset_index().rename(columns={"index": "OriginalIndex"}),
        on=["OriginalIndex", "PatientID", "ExposureFlag"],
        how="left",
        validate="many_to_one",
    )
    return matched


def _balance_table(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    att_weights: np.ndarray,
    matched: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate before/after covariate balance statistics for the propensity design."""
    rows = []

    psm_smd: dict[int, float] = {}
    if not matched.empty and "_DesignRow" in matched.columns:
        design_rows = pd.to_numeric(matched["_DesignRow"], errors="coerce").dropna().astype(int)
        valid = matched.loc[design_rows.index].copy()
        valid["_DesignRow"] = design_rows
        valid = valid[
            valid["_DesignRow"].between(0, len(X) - 1)
            & valid["ExposureFlag"].isin([0, 1])
            & pd.to_numeric(valid["PSMWeight"], errors="coerce").gt(0)
        ].copy()
        if not valid.empty:
            mx = X[valid["_DesignRow"].astype(int).to_numpy()]
            my = valid["ExposureFlag"].astype(int).to_numpy()
            mw = pd.to_numeric(valid["PSMWeight"], errors="coerce").to_numpy(dtype=float)
            for j in range(X.shape[1]):
                psm_smd[j] = _smd(mx[:, j], my, mw)

    for j, feature in enumerate(feature_names):
        before = _smd(X[:, j], y)
        after = _smd(X[:, j], y, att_weights)
        psm = psm_smd.get(j, np.nan)
        rows.append({
            "feature": feature,
            "smd_unweighted": before,
            "abs_smd_unweighted": abs(before) if np.isfinite(before) else np.nan,
            "smd_att": after,
            "abs_smd_att": abs(after) if np.isfinite(after) else np.nan,
            "smd_psm": psm,
            "abs_smd_psm": abs(psm) if np.isfinite(psm) else np.nan,
        })
    return pd.DataFrame(rows)


def _weight_diagnostics(work: pd.DataFrame) -> pd.DataFrame:
    """Summarise ATT weight distribution and effective sample size without patient-level output."""
    supported = work[work["CommonSupportFlag"].eq(1)].copy()
    controls = supported[supported["ExposureFlag"].eq(0)].copy()
    treated = supported[supported["ExposureFlag"].eq(1)].copy()
    rows: list[dict[str, Any]] = []

    for label, sub in [("sports_linked", treated), ("comparison", controls)]:
        w = pd.to_numeric(sub["ATTWeight"], errors="coerce").dropna()
        if w.empty:
            continue
        rows.append({
            "group": label,
            "n": int(len(w)),
            "weight_sum": float(w.sum()),
            "min": float(w.min()),
            "p01": float(w.quantile(0.01)),
            "p05": float(w.quantile(0.05)),
            "median": float(w.median()),
            "p95": float(w.quantile(0.95)),
            "p99": float(w.quantile(0.99)),
            "max": float(w.max()),
        })
    return pd.DataFrame(rows)


def _ps_distribution(work: pd.DataFrame) -> pd.DataFrame:
    """Summarise aggregate propensity-score distributions by group and support status."""
    rows = []
    for exposure, sub in work.groupby("ExposureFlag"):
        ps = pd.to_numeric(sub["PropensityScore"], errors="coerce").dropna()
        supported = sub["CommonSupportFlag"].eq(1)
        if ps.empty:
            continue
        rows.append({
            "ExposureFlag": int(exposure),
            "group": sub["AnalysisGroup"].iloc[0] if "AnalysisGroup" in sub else str(exposure),
            "n": int(len(ps)),
            "supported_n": int(supported.sum()),
            "min": float(ps.min()),
            "p01": float(ps.quantile(0.01)),
            "p05": float(ps.quantile(0.05)),
            "median": float(ps.median()),
            "p95": float(ps.quantile(0.95)),
            "p99": float(ps.quantile(0.99)),
            "max": float(ps.max()),
        })
    return pd.DataFrame(rows)


def fit_propensity(
    df,
    covariates,
    psm_ratio=3,
    caliper_sd_logit=0.2,
    random_seed=42,
    overlap_restrictions=None,
    require_full_psm_ratio=True,
):
    """Execute the complete propensity design: positivity, logistic PS, ATT, SMD and PSM sensitivity."""
    input_work = df.copy()
    input_work = input_work[
        input_work["AnalysisEligibleFlag"].eq(1)
        & input_work["ExposureFlag"].isin([0, 1])
    ].copy()

    if input_work["ExposureFlag"].nunique() != 2:
        raise ValueError("Both exposure groups are required for propensity analysis.")

    # The propensity model must be temporally prior to the outcome period.
    # This explicit guard prevents accidental inclusion of follow-up information.
    _validate_preindex_covariates(list(covariates))
    input_work = _derive_design_covariates(input_work, list(covariates))
    input_n = len(input_work)
    input_exposed = int(input_work["ExposureFlag"].eq(1).sum())
    input_comparison = int(input_work["ExposureFlag"].eq(0).sum())

    work, overlap_audit = _apply_overlap_restrictions(
        input_work,
        overlap_restrictions,
    )
    work = work.copy()
    work["_DesignRow"] = np.arange(len(work), dtype=int)

    _, X, feature_names, available = _feature_matrix(work, covariates)
    y = work["ExposureFlag"].astype(int).to_numpy()

    # Propensity-score model ---------------------------------------------------
    # Logistic regression is used here because the design exposure is binary:
    #   1 = Sports-linked BTH pathway
    #   0 = Wider MSK comparison
    # The fitted probability is *not* an outcome prediction.  It is a compact
    # summary of measured baseline selection into the observed pathway group.
    model = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        random_state=random_seed,
    )
    model.fit(X, y)
    ps = model.predict_proba(X)[:, 1]

    # Store the fitted logistic-regression terms as an audit aid.  These
    # coefficients describe the *selection model* used to construct propensity
    # scores; they are not interpreted as outcome effects or causal estimates.
    model_terms = pd.DataFrame({
        "encoded_feature": feature_names,
        "log_odds_coefficient": model.coef_.reshape(-1),
        "odds_ratio": np.exp(model.coef_.reshape(-1)),
    })
    intercept_row = pd.DataFrame([{
        "encoded_feature": "<Intercept>",
        "log_odds_coefficient": float(model.intercept_[0]),
        "odds_ratio": float(np.exp(model.intercept_[0])),
    }])
    model_terms = pd.concat([intercept_row, model_terms], ignore_index=True)

    work["PropensityScore"] = ps
    treated_ps = work.loc[work["ExposureFlag"].eq(1), "PropensityScore"]
    control_ps = work.loc[work["ExposureFlag"].eq(0), "PropensityScore"]

    # Empirical common support retains only the propensity-score range observed
    # in both groups after the configured structural positivity screen.
    lower = max(float(treated_ps.min()), float(control_ps.min()))
    upper = min(float(treated_ps.max()), float(control_ps.max()))

    work["CommonSupportFlag"] = (
        work["PropensityScore"].between(lower, upper, inclusive="both")
    ).astype("Int64")

    supported = work["CommonSupportFlag"].eq(1)
    # ATT weighting targets the observed Sports-linked population.
    # Sports-linked patients receive weight 1.  Supported comparison patients
    # receive PS/(1-PS), giving more influence to those who resemble the
    # Sports-linked group on measured pre-index characteristics.
    work["ATTWeight"] = 0.0
    work.loc[
        supported & work["ExposureFlag"].eq(1),
        "ATTWeight",
    ] = 1.0
    control_mask = supported & work["ExposureFlag"].eq(0)
    work.loc[control_mask, "ATTWeight"] = (
        work.loc[control_mask, "PropensityScore"]
        / (1 - work.loc[control_mask, "PropensityScore"])
    )

    # PSM is a separate sensitivity design, not a second primary analysis.
    # Matching uses logit-propensity distance within the configured caliper.
    matched = _greedy_match(
        work,
        ratio=psm_ratio,
        caliper_sd_logit=caliper_sd_logit,
        random_seed=random_seed,
        require_full_ratio=require_full_psm_ratio,
    )

    # Standardised mean differences (SMDs) assess whether the design actually
    # balanced measured baseline features.  Propensity-model fit alone is not
    # considered evidence of successful confounding control.
    balance = _balance_table(
        X,
        y,
        feature_names,
        work["ATTWeight"].to_numpy(dtype=float),
        matched,
    )

    control_weights = work.loc[
        work["ExposureFlag"].eq(0) & supported,
        "ATTWeight",
    ]
    ess = (
        (control_weights.sum() ** 2) / np.square(control_weights).sum()
        if len(control_weights) and np.square(control_weights).sum() > 0
        else np.nan
    )

    max_att = float(balance["abs_smd_att"].max()) if not balance.empty else np.nan
    max_psm = (
        float(balance["abs_smd_psm"].max())
        if not balance.empty and balance["abs_smd_psm"].notna().any()
        else np.nan
    )

    # Covariate missingness is recorded as design QA. Imputation is performed only
    # inside the propensity design matrix; raw patient fields are not overwritten.
    missingness_rows = []
    for covariate in available:
        missingness_rows.append({
            "metric": f"covariate_missing_pct::{covariate}",
            "value": float(work[covariate].isna().mean() * 100),
        })

    diagnostics = pd.DataFrame([
        {"metric": "eligible_input_n", "value": input_n},
        {"metric": "eligible_input_exposed_n", "value": input_exposed},
        {"metric": "eligible_input_comparison_n", "value": input_comparison},
        {"metric": "overlap_restricted_n", "value": len(work)},
        {"metric": "overlap_excluded_n", "value": input_n - len(work)},
        {
            "metric": "overlap_excluded_exposed_n",
            "value": input_exposed - int(work["ExposureFlag"].eq(1).sum()),
        },
        {
            "metric": "overlap_excluded_comparison_n",
            "value": input_comparison - int(work["ExposureFlag"].eq(0).sum()),
        },
        {"metric": "propensity_covariates_used_n", "value": len(available)},
        {"metric": "propensity_features_after_encoding_n", "value": X.shape[1]},
        {"metric": "analysis_n", "value": len(work)},
        {"metric": "exposed_n", "value": int(work["ExposureFlag"].sum())},
        {"metric": "comparison_n", "value": int(work["ExposureFlag"].eq(0).sum())},
        {"metric": "common_support_lower", "value": lower},
        {"metric": "common_support_upper", "value": upper},
        {
            "metric": "supported_exposed_n",
            "value": int((supported & work["ExposureFlag"].eq(1)).sum()),
        },
        {
            "metric": "supported_comparison_n",
            "value": int((supported & work["ExposureFlag"].eq(0)).sum()),
        },
        {"metric": "att_control_effective_sample_size", "value": ess},
        {"metric": "max_abs_smd_att", "value": max_att},
        {"metric": "max_abs_smd_psm", "value": max_psm},
        {
            "metric": "matched_exposed_n",
            "value": (
                int(matched.loc[matched["ExposureFlag"].eq(1), "PatientID"].nunique())
                if not matched.empty else 0
            ),
        },
        {
            "metric": "matched_comparison_n",
            "value": (
                int(matched.loc[matched["ExposureFlag"].eq(0), "PatientID"].nunique())
                if not matched.empty else 0
            ),
        },
        {
            "metric": "matched_sets_n",
            "value": int(matched["MatchSetID"].nunique()) if not matched.empty else 0,
        },
        {
            "metric": "max_match_distance_logit_ps",
            "value": (
                float(pd.to_numeric(matched["MatchDistance"], errors="coerce").max())
                if not matched.empty else np.nan
            ),
        },
    ])

    if missingness_rows:
        diagnostics = pd.concat([diagnostics, pd.DataFrame(missingness_rows)], ignore_index=True)

    return PropensityResult(
        data=work,
        balance=balance,
        matched=matched,
        diagnostics=diagnostics,
        overlap_audit=overlap_audit,
        weight_diagnostics=_weight_diagnostics(work),
        ps_distribution=_ps_distribution(work),
        model_terms=model_terms,
    )
