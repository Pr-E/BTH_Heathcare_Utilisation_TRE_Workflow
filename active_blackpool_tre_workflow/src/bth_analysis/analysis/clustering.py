"""Exploratory baseline healthcare-utilisation phenotyping.

Clusters are formed from pre-index utilisation only. Pathway membership,
demographics and follow-up outcomes are deliberately excluded from K-means
construction and are used only afterwards for descriptive profiling.

Candidate cluster solutions are evaluated using separation, minimum-size and
stability diagnostics. A prespecified K=4 is retained only when it satisfies
the configured criteria; otherwise the workflow follows the configured
selection/failure rule.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from bth_analysis.workflow import load_workflow_config, output_path, resolve_path
from bth_analysis.audit import (
    dataframe_preview,
    metric,
    save_stage_summary,
    section as audit_section,
    stage_footer,
    stage_header,
)


DEFAULT_OUTCOME_MAP = {
    "ED": {
        "label": "ED attendances",
        "baseline_count": "BaselineEDCount",
        "followup_count": "FollowUpEDCount",
        "baseline_rate": "BaselineEDRatePerPY",
        "followup_rate": "FollowUpEDRatePerPY",
    },
    "Inpatient": {
        "label": "Inpatient admissions",
        "baseline_count": "BaselineInpatientCount",
        "followup_count": "FollowUpInpatientCount",
        "baseline_rate": "BaselineInpatientRatePerPY",
        "followup_rate": "FollowUpInpatientRatePerPY",
    },
    "EmergencyInpatient": {
        "label": "Emergency inpatient admissions",
        "baseline_count": "BaselineEmergencyInpatientCount",
        "followup_count": "FollowUpEmergencyInpatientCount",
        "baseline_rate": "BaselineEmergencyInpatientRatePerPY",
        "followup_rate": "FollowUpEmergencyInpatientRatePerPY",
    },
}


def _load_clustering_config(path: str | Path) -> dict[str, Any]:
    """Load clustering YAML and retain its resolved path for reproducibility."""
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg["_config_path"] = path
    return cfg


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    """Fail early if a required clustering input is absent."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Coerce one clustering feature to finite numeric values, preserving missingness for explicit checking."""
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _pairwise_mean_ari(label_sets: list[np.ndarray]) -> float:
    """Calculate mean pairwise Adjusted Rand Index across repeated cluster fits."""
    if len(label_sets) < 2:
        return np.nan
    values = [
        adjusted_rand_score(a, b)
        for a, b in combinations(label_sets, 2)
    ]
    return float(np.mean(values)) if values else np.nan


def _cramers_v(table: pd.DataFrame) -> tuple[float, float, int]:
    """Calculate Cramer's V effect size and chi-square p-value for categorical association."""
    if table.empty or min(table.shape) < 2:
        return np.nan, np.nan, 0
    chi2, p_value, dof, _ = chi2_contingency(table, correction=False)
    n = float(table.to_numpy().sum())
    if n <= 0:
        return np.nan, float(p_value), int(dof)
    r, k = table.shape
    denom = n * max(min(r - 1, k - 1), 1)
    value = np.sqrt(float(chi2) / denom)
    return float(value), float(p_value), int(dof)


def _prepare_matrix(
    df: pd.DataFrame,
    features: list[str],
    winsor_upper_quantile: float,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Winsorise, log-transform and standardise baseline utilisation inputs for K-means."""
    raw = df[features].copy()
    for feature in features:
        raw[feature] = _clean_numeric(raw[feature])

    if raw.isna().any().any():
        cols = raw.columns[raw.isna().any()].tolist()
        raise ValueError(
            "Clustering inputs contain missing/non-finite values after eligibility filtering: "
            f"{cols}. Review upstream outcome construction rather than silently imputing them."
        )

    caps = []
    capped = raw.copy()
    for feature in features:
        q = float(raw[feature].quantile(winsor_upper_quantile))
        capped[feature] = raw[feature].clip(upper=q)
        caps.append({
            "feature": feature,
            "winsor_upper_quantile": winsor_upper_quantile,
            "upper_cap": q,
            "raw_max": float(raw[feature].max()),
            "n_values_capped": int((raw[feature] > q).sum()),
        })

    transformed = np.log1p(capped.to_numpy(dtype=float))
    scaler = StandardScaler()
    X = scaler.fit_transform(transformed)
    return X, raw, pd.DataFrame(caps), scaler


def _evaluate_k(
    X: np.ndarray,
    candidate_k: list[int],
    *,
    random_seed: int,
    n_init: int,
    max_iter: int,
    silhouette_sample_size: int,
    stability_sample_size: int,
    min_cluster_n: int,
    min_cluster_pct: float,
    stability_seeds: list[int],
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Fit candidate K solutions and calculate separation, size and stability diagnostics."""
    rng = np.random.default_rng(random_seed)
    if len(X) > stability_sample_size:
        stability_idx = np.sort(
            rng.choice(len(X), size=stability_sample_size, replace=False)
        )
    else:
        stability_idx = np.arange(len(X))
    X_stability = X[stability_idx]

    rows: list[dict[str, Any]] = []
    label_map: dict[int, np.ndarray] = {}

    for k in candidate_k:
        model = KMeans(
            n_clusters=int(k),
            random_state=random_seed,
            n_init=n_init,
            max_iter=max_iter,
            algorithm="lloyd",
        )
        labels = model.fit_predict(X)
        label_map[int(k)] = labels

        counts = pd.Series(labels).value_counts().sort_index()
        min_n = int(counts.min())
        min_pct = float(min_n / len(labels) * 100)

        sil = silhouette_score(
            X,
            labels,
            sample_size=min(int(silhouette_sample_size), len(X)),
            random_state=random_seed,
        )
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)

        stability_labels = []
        for seed in stability_seeds:
            km = KMeans(
                n_clusters=int(k),
                random_state=int(seed),
                n_init=n_init,
                max_iter=max_iter,
                algorithm="lloyd",
            )
            stability_labels.append(km.fit_predict(X_stability))

        stability = _pairwise_mean_ari(stability_labels)
        adequate_size = (
            min_n >= int(min_cluster_n)
            and min_pct >= float(min_cluster_pct)
        )

        rows.append({
            "k": int(k),
            "inertia": float(model.inertia_),
            "silhouette_score": float(sil),
            "calinski_harabasz_score": float(ch),
            "davies_bouldin_score": float(db),
            "minimum_cluster_n": min_n,
            "minimum_cluster_pct": min_pct,
            "mean_pairwise_stability_ari": stability,
            "adequate_cluster_size": int(adequate_size),
        })

    return pd.DataFrame(rows), label_map


def _choose_k(metrics: pd.DataFrame, minimum_stability_ari: float) -> tuple[int, str]:
    """Select K using the configured stability/size rules and clustering selection policy."""
    eligible = metrics[
        metrics["adequate_cluster_size"].eq(1)
        & metrics["mean_pairwise_stability_ari"].ge(minimum_stability_ari)
    ].copy()

    if not eligible.empty:
        chosen = eligible.sort_values(
            ["silhouette_score", "k"], ascending=[False, True]
        ).iloc[0]
        return int(chosen["k"]), "highest silhouette among stable, adequately sized solutions"

    size_ok = metrics[metrics["adequate_cluster_size"].eq(1)].copy()
    if not size_ok.empty:
        chosen = size_ok.sort_values(
            ["silhouette_score", "k"], ascending=[False, True]
        ).iloc[0]
        return int(chosen["k"]), "highest silhouette among adequately sized solutions; stability target not met"

    chosen = metrics.sort_values(
        ["silhouette_score", "k"], ascending=[False, True]
    ).iloc[0]
    return int(chosen["k"]), "highest silhouette; no solution met configured cluster-size constraints"


def _reorder_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    baseline_features: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Relabel raw K-means cluster IDs into a stable descriptive order for reporting."""
    tmp = df[baseline_features].copy()
    tmp["_raw_cluster"] = labels

    intensity = (
        tmp.groupby("_raw_cluster")[baseline_features]
        .mean()
        .sum(axis=1)
        .sort_values()
    )
    mapping = {int(raw): rank + 1 for rank, raw in enumerate(intensity.index)}
    ordered = np.array([mapping[int(x)] for x in labels], dtype=int)

    map_df = pd.DataFrame({
        "raw_cluster": list(mapping.keys()),
        "UtilisationCluster": list(mapping.values()),
    }).sort_values("UtilisationCluster")
    return ordered, map_df


def _baseline_profiles(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Summarise raw baseline utilisation distributions within each cluster."""
    rows = []
    for cluster, sub in df.groupby("UtilisationCluster"):
        row: dict[str, Any] = {
            "UtilisationCluster": int(cluster),
            "patients": int(len(sub)),
            "pct_of_analysis_population": float(len(sub) / len(df) * 100),
        }
        for feature in features:
            x = _clean_numeric(sub[feature]).dropna()
            row[f"{feature}__mean"] = float(x.mean())
            row[f"{feature}__median"] = float(x.median())
            row[f"{feature}__p75"] = float(x.quantile(0.75))
            row[f"{feature}__p95"] = float(x.quantile(0.95))
            row[f"{feature}__zero_pct"] = float((x == 0).mean() * 100)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("UtilisationCluster")



def _cluster_characterisation(baseline_profiles: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Create provisional data-derived phenotype descriptions for later clinical review."""
    service_names = {
        "BaselineEDRatePerPY": "ED attendances",
        "BaselineInpatientRatePerPY": "Inpatient admissions",
        "BaselineEmergencyInpatientRatePerPY": "Emergency inpatient admissions",
    }
    rows = []
    for _, row in baseline_profiles.iterrows():
        means = {feature: float(row.get(f"{feature}__mean", np.nan)) for feature in features}
        finite = {k: v for k, v in means.items() if np.isfinite(v)}
        total = float(sum(finite.values())) if finite else np.nan
        if finite and all(abs(v) < 1e-12 for v in finite.values()):
            dominant = "None"
            provisional = "No recorded baseline hospital utilisation"
        elif finite:
            dominant_feature = max(finite, key=finite.get)
            dominant = service_names.get(dominant_feature, dominant_feature)
            provisional = f"Baseline profile with highest mean utilisation in {dominant.lower()}"
        else:
            dominant = "Not available"
            provisional = "Review required"
        rows.append({
            "UtilisationCluster": int(row["UtilisationCluster"]),
            "patients": int(row["patients"]),
            "pct_of_analysis_population": float(row["pct_of_analysis_population"]),
            "mean_sum_of_input_rates_per_py": total,
            "dominant_baseline_service": dominant,
            "provisional_description": provisional,
            "label_status": "data-derived provisional description; clinician review required",
        })
    return pd.DataFrame(rows).sort_values("UtilisationCluster")

def _phenotype_guide(
    baseline_profiles: pd.DataFrame,
    characterisation: pd.DataFrame,
) -> pd.DataFrame:
    """Create concise, data-derived phenotype labels and a reviewer-facing guide.

    Cluster numbers are arbitrary categorical identifiers. This helper converts
    them into descriptive labels derived only from the baseline utilisation
    profile. The highest-intensity non-zero cluster is explicitly identified
    relative to the other selected-K phenotypes; labels remain provisional until
    clinical/source review.
    """
    guide = characterisation.merge(
        baseline_profiles,
        on=["UtilisationCluster", "patients", "pct_of_analysis_population"],
        how="left",
    ).copy()

    short_service = {
        "ED attendances": "ED",
        "Inpatient admissions": "Inpatient",
        "Emergency inpatient admissions": "Emergency inpatient",
    }

    intensity = pd.to_numeric(
        guide["mean_sum_of_input_rates_per_py"], errors="coerce"
    )
    positive = intensity.fillna(0).gt(0)
    highest_cluster = None
    if positive.any():
        highest_cluster = int(
            guide.loc[positive]
            .sort_values("mean_sum_of_input_rates_per_py")
            .iloc[-1]["UtilisationCluster"]
        )

    labels = []
    axis_labels = []
    for _, row in guide.iterrows():
        cluster = int(row["UtilisationCluster"])
        dominant = str(row.get("dominant_baseline_service", "Not available"))

        if dominant == "None":
            label = "No recorded baseline hospital use"
        elif dominant in short_service:
            base = f"{short_service[dominant]}-dominant"
            if highest_cluster is not None and cluster == highest_cluster:
                label = f"Highest-intensity {base.lower()}"
            else:
                label = base
        else:
            label = "Baseline utilisation phenotype"

        labels.append(label)
        axis_labels.append(f"C{cluster}: {label}")

    guide["phenotype_label"] = labels
    guide["phenotype_axis_label"] = axis_labels
    guide["phenotype_short_label"] = (
        guide["phenotype_label"]
        .str.replace("No recorded baseline hospital use", "No baseline hospital use", regex=False)
        .str.replace("Highest-intensity inpatient-dominant", "High-intensity inpatient", regex=False)
        .str.replace("Highest-intensity emergency inpatient-dominant", "High-intensity emergency inpatient", regex=False)
    )
    return guide.sort_values("UtilisationCluster")


def _centroid_table(
    model: KMeans,
    cluster_mapping: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Return standardised cluster centroids by baseline utilisation feature."""
    rows = []
    lookup = dict(zip(cluster_mapping["raw_cluster"], cluster_mapping["UtilisationCluster"]))
    for raw_cluster, centroid in enumerate(model.cluster_centers_):
        row = {
            "UtilisationCluster": int(lookup[int(raw_cluster)]),
        }
        for feature, value in zip(features, centroid):
            row[feature] = float(value)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("UtilisationCluster")


def _demographic_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Profile numeric demographic/follow-up variables after cluster formation."""
    candidates = [
        "AgeAtIndex",
        "Index_of_Multiple_Deprivation_IMD_Decile",
        "FollowUpDaysAvailable",
    ]
    rows = []
    for cluster, sub in df.groupby("UtilisationCluster"):
        for variable in candidates:
            if variable not in sub.columns:
                continue
            x = _clean_numeric(sub[variable]).dropna()
            if x.empty:
                continue
            rows.append({
                "UtilisationCluster": int(cluster),
                "variable": variable,
                "n": int(len(x)),
                "mean": float(x.mean()),
                "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                "median": float(x.median()),
                "q1": float(x.quantile(0.25)),
                "q3": float(x.quantile(0.75)),
            })
    return pd.DataFrame(rows)


def _categorical_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Profile categorical characteristics after clustering without using them to create clusters."""
    variables = [
        "Sex",
        "EthnicityNationalCodeDesc",
        "PostcodeLAName",
        "AnalysisGroup",
        "FullFollowUpFlag",
    ]
    frames = []
    for variable in variables:
        if variable not in df.columns:
            continue
        counts = (
            df.groupby(["UtilisationCluster", variable], dropna=False)
            .size()
            .rename("n")
            .reset_index()
        )
        counts["pct_within_cluster"] = (
            counts["n"]
            / counts.groupby("UtilisationCluster")["n"].transform("sum")
            * 100
        )
        counts["variable"] = variable
        counts = counts.rename(columns={variable: "level"})
        frames.append(counts[[
            "UtilisationCluster",
            "variable",
            "level",
            "n",
            "pct_within_cluster",
        ]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()



def _top_category(series: pd.Series) -> tuple[str, float]:
    """Return the most common non-missing category and its within-cluster percentage."""
    x = series.astype("string").fillna("<Missing>")
    if x.empty:
        return "Not available", np.nan
    counts = x.value_counts(dropna=False)
    if counts.empty:
        return "Not available", np.nan
    level = str(counts.index[0])
    pct = float(counts.iloc[0] / len(x) * 100)
    return level, pct


def _aggregate_rate_per_100py(
    df: pd.DataFrame,
    count_col: str,
    py_col: str,
) -> float:
    """Calculate one aggregate event rate per 100 person-years."""
    if count_col not in df.columns or py_col not in df.columns:
        return np.nan
    events = float(_clean_numeric(df[count_col]).fillna(0).sum())
    person_years = float(_clean_numeric(df[py_col]).fillna(0).sum())
    return events / person_years * 100 if person_years > 0 else np.nan


def _cluster_patient_profiles(
    df: pd.DataFrame,
    phenotype_guide: pd.DataFrame,
    exposure_dist: pd.DataFrame,
) -> pd.DataFrame:
    """Create one concise reviewer-facing patient/profile row per phenotype.

    Demographics, pathway membership and follow-up outcomes are profiled only
    after cluster assignment; none of these variables enters K-means fitting.
    """
    guide = phenotype_guide.set_index("UtilisationCluster")
    exposure = exposure_dist.set_index("UtilisationCluster")
    rows: list[dict[str, Any]] = []

    def _rate_distribution(sub: pd.DataFrame, rate_col: str, prefix: str, row: dict[str, Any]) -> None:
        if rate_col not in sub.columns:
            return
        x = _clean_numeric(sub[rate_col]).dropna() * 100.0
        if not x.empty:
            row[f"{prefix}_mean_rate_per_100py"] = float(x.mean())
            row[f"{prefix}_median_rate_per_100py"] = float(x.median())

    for cluster, sub in df.groupby("UtilisationCluster", sort=True):
        cluster = int(cluster)
        row: dict[str, Any] = {
            "UtilisationCluster": cluster,
            "phenotype_label": (
                str(guide.loc[cluster, "phenotype_label"])
                if cluster in guide.index else f"Cluster {cluster}"
            ),
            "n": int(len(sub)),
            "pct": float(len(sub) / len(df) * 100),
        }

        if "AgeAtIndex" in sub.columns:
            age = _clean_numeric(sub["AgeAtIndex"]).dropna()
            if not age.empty:
                row.update({
                    "age_median": float(age.median()),
                    "age_q1": float(age.quantile(0.25)),
                    "age_q3": float(age.quantile(0.75)),
                })

        if "Sex" in sub.columns:
            sex = sub["Sex"].astype("string").str.strip().str.lower()
            valid_sex = sex[sex.notna() & ~sex.isin(["", "<missing>", "unknown", "not known"])]
            if not valid_sex.empty:
                female_tokens = {"female", "f", "woman", "women"}
                row["female_pct"] = float(valid_sex.isin(female_tokens).mean() * 100)
            level, pct = _top_category(sub["Sex"])
            row["dominant_sex"] = level
            row["dominant_sex_pct"] = pct

        if "EthnicityNationalCodeDesc" in sub.columns:
            level, pct = _top_category(sub["EthnicityNationalCodeDesc"])
            row["dominant_ethnicity"] = level
            row["dominant_ethnicity_pct"] = pct

        if "Index_of_Multiple_Deprivation_IMD_Decile" in sub.columns:
            imd = _clean_numeric(sub["Index_of_Multiple_Deprivation_IMD_Decile"]).dropna()
            if not imd.empty:
                row.update({
                    "imd_decile_median": float(imd.median()),
                    "imd_decile_q1": float(imd.quantile(0.25)),
                    "imd_decile_q3": float(imd.quantile(0.75)),
                })

        if "PostcodeLAName" in sub.columns:
            level, pct = _top_category(sub["PostcodeLAName"])
            row["main_geography"] = level
            row["main_geography_pct"] = pct

        if cluster in exposure.index:
            row["sports_linked_n"] = int(exposure.loc[cluster, "sports_linked_n"])
            row["sports_linked_pct"] = float(exposure.loc[cluster, "sports_linked_pct_within_cluster"])
            row["wider_msk_n"] = int(exposure.loc[cluster, "wider_msk_n"])
            row["wider_msk_pct"] = float(100.0 - row["sports_linked_pct"])

        # Patient-level baseline/follow-up rate distributions (events per 100 PY).
        for prefix, rate_col in [
            ("baseline_ed", "BaselineEDRatePerPY"),
            ("baseline_inpatient", "BaselineInpatientRatePerPY"),
            ("baseline_emergency_inpatient", "BaselineEmergencyInpatientRatePerPY"),
            ("followup_ed", "FollowUpEDRatePerPY"),
            ("followup_inpatient", "FollowUpInpatientRatePerPY"),
            ("followup_emergency_inpatient", "FollowUpEmergencyInpatientRatePerPY"),
            ("followup_total_hospital", "FollowUpTotalHospitalRatePerPY"),
        ]:
            _rate_distribution(sub, rate_col, prefix, row)

        # Aggregate crude cluster rates preserve the correct event/person-time denominator.
        for outcome, baseline_count, followup_count in [
            ("ed", "BaselineEDCount", "FollowUpEDCount"),
            ("inpatient", "BaselineInpatientCount", "FollowUpInpatientCount"),
            ("emergency_inpatient", "BaselineEmergencyInpatientCount", "FollowUpEmergencyInpatientCount"),
            ("total_hospital", "BaselineTotalHospitalCount", "FollowUpTotalHospitalCount"),
        ]:
            if baseline_count in sub.columns:
                row[f"baseline_{outcome}_aggregate_rate_per_100py"] = _aggregate_rate_per_100py(
                    sub, baseline_count, "BaselinePersonYears"
                )
            elif outcome == "total_hospital" and {"BaselineEDCount", "BaselineInpatientCount"}.issubset(sub.columns):
                events = (_clean_numeric(sub["BaselineEDCount"]).fillna(0) +
                          _clean_numeric(sub["BaselineInpatientCount"]).fillna(0)).sum()
                py = _clean_numeric(sub["BaselinePersonYears"]).fillna(0).sum()
                row[f"baseline_{outcome}_aggregate_rate_per_100py"] = float(events / py * 100) if py > 0 else np.nan

            if followup_count in sub.columns:
                row[f"followup_{outcome}_aggregate_rate_per_100py"] = _aggregate_rate_per_100py(
                    sub, followup_count, "FollowUpPersonYears"
                )
            elif outcome == "total_hospital" and {"FollowUpEDCount", "FollowUpInpatientCount"}.issubset(sub.columns):
                events = (_clean_numeric(sub["FollowUpEDCount"]).fillna(0) +
                          _clean_numeric(sub["FollowUpInpatientCount"]).fillna(0)).sum()
                py = _clean_numeric(sub["FollowUpPersonYears"]).fillna(0).sum()
                row[f"followup_{outcome}_aggregate_rate_per_100py"] = float(events / py * 100) if py > 0 else np.nan

        if "FullFollowUpFlag" in sub.columns:
            row["full_followup_pct"] = float(_clean_numeric(sub["FullFollowUpFlag"]).eq(1).mean() * 100)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("UtilisationCluster")

def _exposure_distribution(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare pathway-group composition across clusters and calculate descriptive association."""
    rows = []
    total_sports = int(df["ExposureFlag"].eq(1).sum())
    total_wider = int(df["ExposureFlag"].eq(0).sum())

    for cluster, sub in df.groupby("UtilisationCluster"):
        sports_n = int(sub["ExposureFlag"].eq(1).sum())
        wider_n = int(sub["ExposureFlag"].eq(0).sum())
        rows.append({
            "UtilisationCluster": int(cluster),
            "patients": int(len(sub)),
            "pct_of_analysis_population": float(len(sub) / len(df) * 100),
            "sports_linked_n": sports_n,
            "wider_msk_n": wider_n,
            "sports_linked_pct_within_cluster": float(sports_n / len(sub) * 100),
            "pct_of_all_sports_linked_in_cluster": (
                float(sports_n / total_sports * 100) if total_sports else np.nan
            ),
            "pct_of_all_wider_msk_in_cluster": (
                float(wider_n / total_wider * 100) if total_wider else np.nan
            ),
        })

    table = pd.crosstab(df["UtilisationCluster"], df["ExposureFlag"])
    cramer_v, p_value, dof = _cramers_v(table)
    association = pd.DataFrame([{
        "analysis": "UtilisationCluster x ExposureFlag",
        "patients": int(len(df)),
        "clusters": int(df["UtilisationCluster"].nunique()),
        "cramers_v": cramer_v,
        "chi_square_p_value": p_value,
        "degrees_of_freedom": dof,
        "interpretation_note": (
            "ExposureFlag was not used to create clusters. Cramer's V describes the "
            "strength of association between baseline utilisation phenotype and observed "
            "Sports-linked pathway membership; it is descriptive, not causal."
        ),
    }])
    return pd.DataFrame(rows).sort_values("UtilisationCluster"), association


def _trajectory_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate descriptive baseline/follow-up outcome rates within clusters and pathway groups."""
    rows = []
    for cluster, cluster_sub in df.groupby("UtilisationCluster"):
        for exposure, sub in cluster_sub.groupby("ExposureFlag"):
            group_name = (
                str(sub["AnalysisGroup"].iloc[0])
                if "AnalysisGroup" in sub.columns and len(sub)
                else str(exposure)
            )
            for outcome_key, spec in DEFAULT_OUTCOME_MAP.items():
                for period in ("Baseline", "Follow-up"):
                    count_col = spec["baseline_count"] if period == "Baseline" else spec["followup_count"]
                    py_col = "BaselinePersonYears" if period == "Baseline" else "FollowUpPersonYears"
                    if count_col not in sub.columns or py_col not in sub.columns:
                        continue
                    events = float(_clean_numeric(sub[count_col]).fillna(0).sum())
                    py = float(_clean_numeric(sub[py_col]).fillna(0).sum())
                    rows.append({
                        "UtilisationCluster": int(cluster),
                        "ExposureFlag": int(exposure),
                        "group": group_name,
                        "outcome": outcome_key,
                        "outcome_label": spec["label"],
                        "period": period,
                        "patients": int(len(sub)),
                        "events": events,
                        "person_years": py,
                        "rate_per_person_year": events / py if py > 0 else np.nan,
                        "rate_per_100_person_years": events / py * 100 if py > 0 else np.nan,
                    })

    trajectory = pd.DataFrame(rows)
    change_rows = []
    if not trajectory.empty:
        keys = ["UtilisationCluster", "ExposureFlag", "group", "outcome", "outcome_label"]
        for key_vals, sub in trajectory.groupby(keys, dropna=False):
            pivot = sub.set_index("period")["rate_per_100_person_years"]
            base = float(pivot.get("Baseline", np.nan))
            follow = float(pivot.get("Follow-up", np.nan))
            row = dict(zip(keys, key_vals))
            row.update({
                "baseline_rate_per_100py": base,
                "followup_rate_per_100py": follow,
                "absolute_change_per_100py": follow - base,
                "followup_to_baseline_rate_ratio": (
                    follow / base if np.isfinite(base) and base > 0 else np.nan
                ),
            })
            change_rows.append(row)
    return trajectory, pd.DataFrame(change_rows)


def _followup_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise follow-up outcome distributions within baseline utilisation phenotypes."""
    features = [
        "FollowUpEDRatePerPY",
        "FollowUpInpatientRatePerPY",
        "FollowUpEmergencyInpatientRatePerPY",
        "FollowUpTotalHospitalRatePerPY",
    ]
    rows = []
    for cluster, sub in df.groupby("UtilisationCluster"):
        row: dict[str, Any] = {"UtilisationCluster": int(cluster), "patients": int(len(sub))}
        for feature in features:
            if feature not in sub.columns:
                continue
            x = _clean_numeric(sub[feature]).dropna()
            if x.empty:
                continue
            row[f"{feature}__mean"] = float(x.mean())
            row[f"{feature}__median"] = float(x.median())
            row[f"{feature}__p95"] = float(x.quantile(0.95))
            row[f"{feature}__zero_pct"] = float((x == 0).mean() * 100)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("UtilisationCluster")


def _plot_cluster_selection(metrics: pd.DataFrame, selected_k: int, path: Path) -> None:
    """Plot silhouette score across candidate K solutions and highlight selected K."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(metrics["k"], metrics["silhouette_score"], marker="o")
    chosen = metrics.loc[metrics["k"].eq(selected_k)].iloc[0]
    ax.scatter([selected_k], [chosen["silhouette_score"]], s=90, zorder=3)
    ax.set_title("K-means cluster selection: silhouette score", loc="left", fontweight="bold")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(metrics["k"].astype(int).tolist())
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def _plot_cluster_sizes(phenotype_guide: pd.DataFrame, path: Path) -> None:
    """Plot the share of the eligible population in each baseline utilisation phenotype."""
    x = phenotype_guide.sort_values("UtilisationCluster").copy()
    labels = x["phenotype_axis_label"].astype(str).tolist()
    values = x["pct_of_analysis_population"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    y = np.arange(len(x))
    bars = ax.barh(y, values)
    for bar, value in zip(bars, values):
        ax.text(
            value + max(values.max() * 0.012, 0.25),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=10,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Distribution of baseline healthcare-utilisation phenotypes", loc="left", fontweight="bold")
    ax.set_xlabel("Share of eligible population (%)")
    ax.set_ylabel("")
    ax.set_xlim(left=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def _plot_centroid_heatmap(
    centroids: pd.DataFrame,
    features: list[str],
    phenotype_guide: pd.DataFrame,
    path: Path,
) -> None:
    """Plot standardised baseline utilisation signatures using descriptive phenotype labels."""
    data = centroids.set_index("UtilisationCluster")[features]
    guide = phenotype_guide.set_index("UtilisationCluster")
    service_labels = {
        "BaselineEDRatePerPY": "ED",
        "BaselineInpatientRatePerPY": "Inpatient",
        "BaselineEmergencyInpatientRatePerPY": "Emergency inpatient",
    }

    fig, ax = plt.subplots(figsize=(10.2, max(5.0, 1.0 + 0.9 * len(data))))
    image = ax.imshow(data.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels([service_labels.get(f, f) for f in features])
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels([
        f"C{i}: {guide.loc[i, 'phenotype_label']}" if i in guide.index else f"Cluster {i}"
        for i in data.index
    ])
    ax.set_title("Baseline utilisation signature of each phenotype", loc="left", fontweight="bold")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{float(data.iloc[i, j]):.2f}", ha="center", va="center")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Standardised centroid (z-score; 0 = cohort average)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def _plot_baseline_profile_rates(
    baseline_profiles: pd.DataFrame,
    phenotype_guide: pd.DataFrame,
    path: Path,
) -> None:
    """Show the raw baseline rates that define each healthcare-utilisation phenotype."""
    profile = baseline_profiles.sort_values("UtilisationCluster").copy()
    guide = phenotype_guide.set_index("UtilisationCluster")
    specs = [
        ("BaselineEDRatePerPY__mean", "ED attendances"),
        ("BaselineInpatientRatePerPY__mean", "Inpatient admissions"),
        ("BaselineEmergencyInpatientRatePerPY__mean", "Emergency inpatient admissions"),
    ]

    y = np.arange(len(profile), dtype=float)
    height = 0.23
    offsets = [-height, 0.0, height]
    fig, ax = plt.subplots(figsize=(11.2, 6.5))

    for offset, (column, label) in zip(offsets, specs):
        if column not in profile.columns:
            continue
        values = pd.to_numeric(profile[column], errors="coerce").to_numpy() * 100
        bars = ax.barh(y + offset, values, height=height, label=label)
        for bar, value in zip(bars, values):
            if np.isfinite(value) and value > 0:
                ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.1f}", va="center", fontsize=9)

    labels = [
        f"C{int(c)}: {guide.loc[int(c), 'phenotype_label']}"
        for c in profile["UtilisationCluster"]
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("What each baseline utilisation phenotype represents", loc="left", fontweight="bold")
    ax.set_xlabel("Mean baseline events per 100 person-years")
    ax.set_ylabel("")
    ax.legend(frameon=False)
    ax.set_xlim(left=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def _plot_group_cluster_distribution(
    exposure_dist: pd.DataFrame,
    phenotype_guide: pd.DataFrame,
    path: Path,
) -> None:
    """Compare where Sports-linked and Wider MSK patients fall across phenotypes."""
    x = exposure_dist.sort_values("UtilisationCluster").copy()
    guide = phenotype_guide.set_index("UtilisationCluster")
    y = np.arange(len(x), dtype=float)
    height = 0.34

    sports = x["pct_of_all_sports_linked_in_cluster"].astype(float).to_numpy()
    wider = x["pct_of_all_wider_msk_in_cluster"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    sports_bars = ax.barh(y - height / 2, sports, height=height, label="Sports-linked BTH pathway")
    wider_bars = ax.barh(y + height / 2, wider, height=height, label="Wider MSK non-Sports-linked candidate")

    for bars, values in [(sports_bars, sports), (wider_bars, wider)]:
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.1f}%", va="center", fontsize=9)

    labels = [
        f"C{int(c)}: {guide.loc[int(c), 'phenotype_label']}"
        for c in x["UtilisationCluster"]
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Pathway groups across baseline utilisation phenotypes", loc="left", fontweight="bold")
    ax.set_xlabel("Share of each analysis group (%)")
    ax.set_ylabel("")
    ax.legend(frameon=False)
    ax.set_xlim(left=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def _plot_followup_by_cluster(
    trajectory: pd.DataFrame,
    outcome_key: str,
    phenotype_guide: pd.DataFrame,
    path: Path,
) -> None:
    """Plot observed crude follow-up rates across clearly labelled baseline phenotypes."""
    sub = trajectory[
        trajectory["outcome"].eq(outcome_key)
        & trajectory["period"].eq("Follow-up")
    ].copy()
    if sub.empty:
        return

    guide = phenotype_guide.set_index("UtilisationCluster")
    clusters = sorted(sub["UtilisationCluster"].astype(int).unique())
    x = np.arange(len(clusters), dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    preferred_order = [
        "Sports-linked BTH pathway",
        "Wider MSK non-Sports-linked candidate",
    ]
    available_groups = sub["group"].astype(str).drop_duplicates().tolist()
    group_order = [g for g in preferred_order if g in available_groups]
    group_order += [g for g in available_groups if g not in group_order]

    for group in group_order:
        values = []
        for cluster in clusters:
            row = sub[
                sub["UtilisationCluster"].eq(cluster)
                & sub["group"].astype(str).eq(group)
            ]
            values.append(
                float(row["rate_per_100_person_years"].iloc[0])
                if not row.empty
                else np.nan
            )
        ax.plot(x, values, marker="o", linewidth=2, label=group)

    labels = []
    for cluster in clusters:
        phenotype = (
            str(guide.loc[cluster, "phenotype_short_label"])
            if cluster in guide.index
            else f"Cluster {cluster}"
        )
        labels.append(f"C{cluster}\n{phenotype}")

    title_map = {
        "ED": "Follow-up ED attendance by baseline utilisation phenotype",
        "Inpatient": "Follow-up inpatient admissions by baseline utilisation phenotype",
        "EmergencyInpatient": "Follow-up emergency inpatient admissions by baseline utilisation phenotype",
    }
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title_map.get(outcome_key, "Follow-up utilisation by baseline utilisation phenotype"), loc="left", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Crude follow-up events per 100 person-years")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def run_clustering(
    workflow_path: str | Path = "config/workflow_tre.yaml",
    clustering_config_path: str | Path = "config/clustering_tre.yaml",
) -> dict[str, pd.DataFrame]:
    """Run the exploratory clustering layer, diagnostics, sensitivities and phenotype audit."""
    workflow = load_workflow_config(workflow_path)
    cfg = _load_clustering_config(clustering_config_path)
    section = cfg.get("clustering", {})

    analysis_dir = output_path(workflow, "analysis_dir")
    if "clustering_dir" in workflow.get("outputs", {}):
        out_dir = output_path(workflow, "clustering_dir")
    else:
        out_dir = resolve_path(workflow, "outputs/clustering")
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    model_dir = out_dir / "model"
    for path in (tables_dir, figures_dir, model_dir):
        path.mkdir(parents=True, exist_ok=True)

    stage_header(
        "09",
        "EXPLORATORY BASELINE HEALTHCARE-UTILISATION CLUSTERING",
        purpose=(
            "Identify data-driven baseline utilisation phenotypes using ED, inpatient and emergency-inpatient "
            "rates only; assess the configured candidate K solutions; retain the prespecified report-facing K only if "
            "the current data meet size/stability criteria; profile pathway membership and follow-up only after clusters are formed."
        ),
        inputs=[analysis_dir / "patient_outcomes.csv", clustering_config_path],
        outputs=[tables_dir, figures_dir, model_dir],
    )

    source = analysis_dir / "patient_outcomes.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    df = pd.read_csv(source, low_memory=False)

    baseline_features = section.get("baseline_features", [
        "BaselineEDRatePerPY",
        "BaselineInpatientRatePerPY",
        "BaselineEmergencyInpatientRatePerPY",
    ])
    baseline_features = list(baseline_features)

    required = [
        "PatientID",
        "ExposureFlag",
        "AnalysisGroup",
        "AnalysisEligibleFlag",
        "BaselineCompleteFlag",
        "BaselinePersonYears",
        "FollowUpPersonYears",
        *baseline_features,
    ]
    _require_columns(df, required, "Healthcare-utilisation clustering")

    eligible = df[
        df["AnalysisEligibleFlag"].eq(1)
        & df["BaselineCompleteFlag"].eq(1)
    ].copy()

    if eligible.empty:
        raise ValueError("No analysis-eligible patients with complete baseline are available for clustering.")

    # Do not allow exposure or post-index outcomes to leak into cluster construction.
    forbidden = [
        c for c in baseline_features
        if c in {"ExposureFlag", "AnalysisGroup"} or c.startswith("FollowUp")
    ]
    if forbidden:
        raise ValueError(
            "Clustering inputs must be pre-index healthcare-utilisation features only. "
            f"Remove: {forbidden}"
        )

    for feature in baseline_features:
        eligible[feature] = _clean_numeric(eligible[feature])

    input_complete = eligible[baseline_features].notna().all(axis=1)
    exclusions = pd.DataFrame([
        {
            "stage": "analysis_eligible_complete_baseline",
            "patients": int(len(eligible)),
        },
        {
            "stage": "complete_clustering_inputs",
            "patients": int(input_complete.sum()),
        },
        {
            "stage": "excluded_missing_clustering_inputs",
            "patients": int((~input_complete).sum()),
        },
    ])
    analysis = eligible.loc[input_complete].copy().reset_index(drop=True)

    winsor_q = float(section.get("winsor_upper_quantile", 0.995))
    X, raw_inputs, caps, scaler = _prepare_matrix(
        analysis,
        baseline_features,
        winsor_upper_quantile=winsor_q,
    )

    candidate_k = [int(k) for k in section.get("candidate_k", [2, 3, 4, 5, 6])]
    random_seed = int(section.get("random_seed", workflow.get("project", {}).get("random_seed", 42)))
    n_init = int(section.get("n_init", 50))
    max_iter = int(section.get("max_iter", 500))
    silhouette_sample_size = int(section.get("silhouette_sample_size", 2000))
    stability_sample_size = int(section.get("stability_sample_size", 8000))
    min_cluster_n = int(section.get("minimum_cluster_n", 50))
    min_cluster_pct = float(section.get("minimum_cluster_pct", 1.0))
    stability_seeds = [int(x) for x in section.get("stability_seeds", [11, 23, 37, 51, 71])]
    minimum_stability_ari = float(section.get("minimum_stability_ari", 0.80))

    metrics, _ = _evaluate_k(
        X,
        candidate_k,
        random_seed=random_seed,
        n_init=n_init,
        max_iter=max_iter,
        silhouette_sample_size=silhouette_sample_size,
        stability_sample_size=stability_sample_size,
        min_cluster_n=min_cluster_n,
        min_cluster_pct=min_cluster_pct,
        stability_seeds=stability_seeds,
    )
    # Always calculate the data-driven solution from the current analysis data.
    auto_selected_k, auto_selection_reason = _choose_k(metrics, minimum_stability_ari)

    # A prespecified report-facing K is a candidate rather than a guaranteed solution.
    # It is retained only when the current data meet the configured size and
    # stability criteria.
    selection_mode = str(section.get("selection_mode", "prespecified_with_diagnostics")).lower()
    prespecified_k = int(section.get("prespecified_k", 4))
    fail_if_prespecified_inadequate = bool(section.get("fail_if_prespecified_inadequate", True))

    if selection_mode == "automatic":
        selected_k = auto_selected_k
        selection_reason = auto_selection_reason
    elif selection_mode == "prespecified_with_diagnostics":
        if prespecified_k not in set(metrics["k"].astype(int)):
            raise ValueError(
                f"prespecified_k={prespecified_k} is not present in candidate_k={candidate_k}."
            )
        row = metrics.loc[metrics["k"].eq(prespecified_k)].iloc[0]
        adequate = bool(
            int(row["adequate_cluster_size"]) == 1
            and float(row["mean_pairwise_stability_ari"]) >= minimum_stability_ari
        )
        if adequate:
            selected_k = prespecified_k
            selection_reason = (
                f"prespecified K={prespecified_k} retained after current-data size/stability checks; "
                f"automatic diagnostic choice was K={auto_selected_k}"
            )
        elif fail_if_prespecified_inadequate:
            raise RuntimeError(
                f"Prespecified K={prespecified_k} failed the current data cluster size/stability criteria. "
                "Do not force a prespecified cluster structure onto data that fail the configured checks. "
                f"Automatic diagnostic choice is K={auto_selected_k}."
            )
        else:
            selected_k = auto_selected_k
            selection_reason = (
                f"prespecified K={prespecified_k} inadequate on the current data; fell back to automatic K={auto_selected_k}"
            )
    else:
        raise ValueError(
            "clustering.selection_mode must be 'automatic' or 'prespecified_with_diagnostics'."
        )

    final_model = KMeans(
        n_clusters=selected_k,
        random_state=random_seed,
        n_init=max(n_init, 100),
        max_iter=max_iter,
        algorithm="lloyd",
    )
    raw_labels = final_model.fit_predict(X)
    ordered_labels, cluster_mapping = _reorder_clusters(
        analysis,
        raw_labels,
        baseline_features,
    )
    analysis["UtilisationCluster"] = ordered_labels

    # Primary summaries.
    centroids = _centroid_table(final_model, cluster_mapping, baseline_features)
    baseline_profiles = _baseline_profiles(analysis, baseline_features)
    characterisation = _cluster_characterisation(baseline_profiles, baseline_features)
    phenotype_guide = _phenotype_guide(baseline_profiles, characterisation)
    followup_profiles = _followup_profiles(analysis)
    demographic_numeric = _demographic_numeric(analysis)
    categorical_profiles = _categorical_profiles(analysis)
    exposure_distribution, exposure_association = _exposure_distribution(analysis)
    patient_profiles = _cluster_patient_profiles(analysis, phenotype_guide, exposure_distribution)
    trajectory, change_summary = _trajectory_tables(analysis)

    # Sensitivity: refit selected K without winsorisation to quantify dependence on the cap.
    uncapped = analysis[baseline_features].copy()
    transformed_uncapped = np.log1p(uncapped.to_numpy(dtype=float))
    scaler_uncapped = StandardScaler()
    X_uncapped = scaler_uncapped.fit_transform(transformed_uncapped)
    km_uncapped = KMeans(
        n_clusters=selected_k,
        random_state=random_seed,
        n_init=max(n_init, 100),
        max_iter=max_iter,
        algorithm="lloyd",
    )
    uncapped_labels = km_uncapped.fit_predict(X_uncapped)
    winsor_sensitivity_ari = float(adjusted_rand_score(raw_labels, uncapped_labels))

    # Optional sensitivity for the nested inpatient definitions.  Emergency
    # inpatient admissions are a subset of total inpatient admissions, so the
    # primary three-feature space effectively represents emergency-heavy patients
    # in both the total-inpatient and emergency-inpatient dimensions.  This
    # sensitivity replaces total inpatient with non-emergency inpatient to make
    # the three dimensions mutually exclusive and reports ARI versus the primary
    # assignments.  It does not replace the prespecified primary clustering.
    exclusive_ari = np.nan
    if bool(section.get("run_mutually_exclusive_inpatient_sensitivity", True)):
        needed = {
            "BaselineEDRatePerPY",
            "BaselineInpatientRatePerPY",
            "BaselineEmergencyInpatientRatePerPY",
        }
        if needed.issubset(analysis.columns):
            exclusive = analysis.copy()
            exclusive["BaselineNonEmergencyInpatientRatePerPY"] = (
                _clean_numeric(exclusive["BaselineInpatientRatePerPY"])
                - _clean_numeric(exclusive["BaselineEmergencyInpatientRatePerPY"])
            ).clip(lower=0)
            exclusive_features = [
                "BaselineEDRatePerPY",
                "BaselineNonEmergencyInpatientRatePerPY",
                "BaselineEmergencyInpatientRatePerPY",
            ]
            X_exclusive, _, _, _ = _prepare_matrix(
                exclusive, exclusive_features, winsor_upper_quantile=winsor_q
            )
            km_exclusive = KMeans(
                n_clusters=selected_k,
                random_state=random_seed,
                n_init=max(n_init, 100),
                max_iter=max_iter,
                algorithm="lloyd",
            )
            exclusive_labels = km_exclusive.fit_predict(X_exclusive)
            exclusive_ari = float(adjusted_rand_score(raw_labels, exclusive_labels))

    chosen_metric = metrics.loc[metrics["k"].eq(selected_k)].iloc[0]
    summary = pd.DataFrame([{
        "analysis_population_n": int(len(analysis)),
        "sports_linked_n": int(analysis["ExposureFlag"].eq(1).sum()),
        "wider_msk_n": int(analysis["ExposureFlag"].eq(0).sum()),
        "clustering_scope": "baseline healthcare utilisation only",
        "selected_k": int(selected_k),
        "selection_mode": selection_mode,
        "prespecified_k": int(prespecified_k),
        "automatic_selected_k": int(auto_selected_k),
        "automatic_selection_reason": auto_selection_reason,
        "selection_reason": selection_reason,
        "silhouette_score": float(chosen_metric["silhouette_score"]),
        "calinski_harabasz_score": float(chosen_metric["calinski_harabasz_score"]),
        "davies_bouldin_score": float(chosen_metric["davies_bouldin_score"]),
        "minimum_cluster_n": int(chosen_metric["minimum_cluster_n"]),
        "minimum_cluster_pct": float(chosen_metric["minimum_cluster_pct"]),
        "mean_pairwise_stability_ari": float(chosen_metric["mean_pairwise_stability_ari"]),
        "winsorisation_sensitivity_ari": winsor_sensitivity_ari,
        "mutually_exclusive_feature_sensitivity_ari": exclusive_ari,
        "winsor_upper_quantile": winsor_q,
        "transform": "log1p then StandardScaler",
        "exposure_used_in_clustering": False,
        "followup_used_in_clustering": False,
        "programme_effect_interpretation_allowed": False,
        "interpretation_note": (
            "Clusters are exploratory baseline healthcare-utilisation phenotypes. "
            "ExposureFlag and post-index outcomes are used only after clustering for profiling. "
            "Cluster differences are descriptive and must not be interpreted as Active Blackpool treatment effects."
        ),
    }])

    sensitivity_rows = [{
        "sensitivity": "same K without winsorisation",
        "selected_k": int(selected_k),
        "adjusted_rand_index_vs_primary": winsor_sensitivity_ari,
        "interpretation": (
            "Values closer to 1 indicate cluster assignments are less sensitive to the configured upper-tail cap."
        ),
    }]
    if np.isfinite(exclusive_ari):
        sensitivity_rows.append({
            "sensitivity": "mutually exclusive inpatient dimensions",
            "selected_k": int(selected_k),
            "adjusted_rand_index_vs_primary": exclusive_ari,
            "interpretation": (
                "Compares the primary total-inpatient/emergency-inpatient feature space with ED, "
                "non-emergency inpatient and emergency inpatient rates. Higher ARI indicates that "
                "phenotypes are not driven mainly by double representation of emergency admissions."
            ),
        })
    sensitivity = pd.DataFrame(sensitivity_rows)

    # Minimal patient-level assignment output for internal analytical use.
    assignment_cols = [
        "PatientID",
        "ExposureFlag",
        "AnalysisGroup",
        "UtilisationCluster",
        *baseline_features,
    ]
    for optional in [
        "AgeAtIndex",
        "Sex",
        "EthnicityNationalCodeDesc",
        "PostcodeLAName",
        "Index_of_Multiple_Deprivation_IMD_Decile",
        "FollowUpEDRatePerPY",
        "FollowUpInpatientRatePerPY",
        "FollowUpEmergencyInpatientRatePerPY",
        "FollowUpTotalHospitalRatePerPY",
        "FullFollowUpFlag",
        "FollowUpDaysAvailable",
    ]:
        if optional in analysis.columns and optional not in assignment_cols:
            assignment_cols.append(optional)
    assignments = analysis[assignment_cols].copy()

    outputs: dict[str, pd.DataFrame] = {
        "clustering_summary": summary,
        "clustering_population_flow": exclusions,
        "cluster_selection_metrics": metrics,
        "clustering_preprocessing_caps": caps,
        "cluster_mapping": cluster_mapping,
        "cluster_centroids_standardised": centroids,
        "cluster_baseline_profiles": baseline_profiles,
        "cluster_characterisation": characterisation,
        "cluster_phenotype_guide": phenotype_guide,
        "cluster_followup_profiles": followup_profiles,
        "cluster_demographic_numeric": demographic_numeric,
        "cluster_categorical_profiles": categorical_profiles,
        "cluster_exposure_distribution": exposure_distribution,
        "cluster_exposure_association": exposure_association,
        "cluster_patient_profile_summary": patient_profiles,
        "cluster_utilisation_trajectory": trajectory,
        "cluster_change_summary": change_summary,
        "clustering_sensitivity": sensitivity,
        "cluster_assignments": assignments,
    }

    for name, table in outputs.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)

    # Merge cluster size/pathway composition with the provisional data-derived
    # descriptions into one compact reviewer-facing table.  Labels remain
    # provisional until clinical/source review.
    clustering_key_findings = exposure_distribution.merge(
        phenotype_guide[
            [
                "UtilisationCluster",
                "patients",
                "pct_of_analysis_population",
                "phenotype_label",
                "dominant_baseline_service",
                "provisional_description",
                "label_status",
            ]
        ],
        on=["UtilisationCluster", "patients", "pct_of_analysis_population"],
        how="left",
    )
    clustering_key_findings.to_csv(
        tables_dir / "clustering_key_findings.csv", index=False
    )
    outputs["clustering_key_findings"] = clustering_key_findings

    # Save fitted artefacts for reproducibility within the development/TRE environment.
    joblib.dump(final_model, model_dir / "kmeans_model.joblib")
    joblib.dump(scaler, model_dir / "standard_scaler.joblib")
    artifact = {
        "baseline_features": baseline_features,
        "winsor_upper_quantile": winsor_q,
        "winsor_caps": dict(zip(caps["feature"], caps["upper_cap"])),
        "selected_k": int(selected_k),
        "automatic_selected_k": int(auto_selected_k),
        "selection_mode": selection_mode,
        "cluster_mapping": dict(zip(cluster_mapping["raw_cluster"], cluster_mapping["UtilisationCluster"])),
        "transform": "winsor upper tail -> log1p -> StandardScaler -> KMeans",
        "note": "Cluster centroids are fitted from the data supplied to this run and must not be reused as fixed centroids for a different dataset.",
    }
    joblib.dump(artifact, model_dir / "clustering_artifact.joblib")
    (model_dir / "clustering_metadata.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )

    _plot_cluster_selection(
        metrics,
        selected_k,
        figures_dir / "cluster_selection_silhouette.png",
    )
    _plot_cluster_sizes(
        phenotype_guide,
        figures_dir / "cluster_size_distribution.png",
    )
    _plot_centroid_heatmap(
        centroids,
        baseline_features,
        phenotype_guide,
        figures_dir / "cluster_centroid_heatmap.png",
    )
    _plot_baseline_profile_rates(
        baseline_profiles,
        phenotype_guide,
        figures_dir / "cluster_baseline_profile_rates.png",
    )
    _plot_group_cluster_distribution(
        exposure_distribution,
        phenotype_guide,
        figures_dir / "analysis_group_cluster_distribution.png",
    )
    for outcome_key in DEFAULT_OUTCOME_MAP:
        _plot_followup_by_cluster(
            trajectory,
            outcome_key,
            phenotype_guide,
            figures_dir / f"followup_{outcome_key.lower()}_by_baseline_cluster.png",
        )

    audit_section("STAGE 09 MODEL-SELECTION FINDINGS")
    metric("eligible complete-baseline patients", f"{len(analysis):,}")
    metric("Sports-linked", f"{analysis['ExposureFlag'].eq(1).sum():,}")
    metric("Wider MSK", f"{analysis['ExposureFlag'].eq(0).sum():,}")
    metric("clustering features", ", ".join(baseline_features))
    metric("selected K", selected_k)
    metric("automatic best K", auto_selected_k)
    metric("selection mode", selection_mode)
    metric("silhouette score", f"{float(chosen_metric['silhouette_score']):.4f}")
    metric("Davies-Bouldin score", f"{float(chosen_metric['davies_bouldin_score']):.4f}")
    metric("minimum cluster size", f"{int(chosen_metric['minimum_cluster_n']):,} ({float(chosen_metric['minimum_cluster_pct']):.2f}%)")
    metric("mean stability ARI", f"{float(chosen_metric['mean_pairwise_stability_ari']):.4f}")
    metric("no-winsor sensitivity ARI", f"{winsor_sensitivity_ari:.4f}")
    if np.isfinite(exclusive_ari):
        metric("mutually-exclusive feature ARI", f"{exclusive_ari:.4f}")

    print("\nCandidate-K diagnostic comparison (" + ", ".join(str(k) for k in candidate_k) + "):")
    dataframe_preview(
        metrics,
        columns=[
            "k", "silhouette_score", "calinski_harabasz_score",
            "davies_bouldin_score", "minimum_cluster_n", "minimum_cluster_pct",
            "mean_pairwise_stability_ari", "adequate_cluster_size",
        ],
        max_rows=10,
    )

    audit_section("STAGE 09 PHENOTYPE FINDINGS")
    dataframe_preview(
        clustering_key_findings,
        columns=[
            "UtilisationCluster", "patients", "pct_of_analysis_population",
            "sports_linked_n", "wider_msk_n", "pct_of_all_sports_linked_in_cluster",
            "pct_of_all_wider_msk_in_cluster", "phenotype_label",
            "dominant_baseline_service", "provisional_description", "label_status",
        ],
        max_rows=20,
    )

    audit_section("STAGE 09 PATIENT PROFILE SNAPSHOT")
    dataframe_preview(
        patient_profiles,
        columns=[
            "UtilisationCluster", "phenotype_label", "patients",
            "age_years_median", "imd_decile_median",
            "most_common_sex", "most_common_ethnicity", "most_common_geography",
            "sports_linked_pct_within_cluster",
            "baseline_ed_rate_per_100py", "baseline_inpatient_rate_per_100py",
            "baseline_emergency_inpatient_rate_per_100py",
            "followup_ed_rate_per_100py", "followup_inpatient_rate_per_100py",
            "followup_emergency_inpatient_rate_per_100py",
        ],
        max_rows=20,
    )

    cramers_v = float(exposure_association["cramers_v"].iloc[0]) if not exposure_association.empty else np.nan
    assoc_p = float(exposure_association["chi_square_p_value"].iloc[0]) if not exposure_association.empty else np.nan
    metric("Cramer's V: cluster x pathway group", f"{cramers_v:.4f}" if np.isfinite(cramers_v) else "NA")
    metric("chi-square p-value", f"{assoc_p:.4g}" if np.isfinite(assoc_p) else "NA")

    # Cluster-specific Sports-linked trajectories become unstable when very few
    # exposed patients occupy a phenotype.  Surface these cells explicitly.
    small_sports_threshold = int(section.get("minimum_sports_linked_for_trajectory_interpretation", 30))
    small_sports = exposure_distribution[exposure_distribution["sports_linked_n"].lt(small_sports_threshold)].copy()
    metric("clusters below Sports-linked trajectory n threshold", len(small_sports))
    if not small_sports.empty:
        print("\nSparse Sports-linked phenotype cells - descriptive trajectories only:")
        dataframe_preview(
            small_sports,
            columns=["UtilisationCluster", "patients", "sports_linked_n", "wider_msk_n"],
            max_rows=20,
        )

    audit_dir = output_path(workflow, "audit_dir")
    summary_path = save_stage_summary(
        audit_dir,
        stage_key="clustering",
        stage_code="09",
        title="Exploratory baseline healthcare-utilisation clustering",
        status="PASS",
        key_findings={
            "analysis_population_n": len(analysis),
            "selected_k": selected_k,
            "automatic_selected_k": auto_selected_k,
            "silhouette_score": float(chosen_metric["silhouette_score"]),
            "davies_bouldin_score": float(chosen_metric["davies_bouldin_score"]),
            "minimum_cluster_n": int(chosen_metric["minimum_cluster_n"]),
            "mean_stability_ari": float(chosen_metric["mean_pairwise_stability_ari"]),
            "winsorisation_sensitivity_ari": winsor_sensitivity_ari,
            "mutually_exclusive_feature_ari": exclusive_ari,
            "cramers_v_cluster_vs_pathway": cramers_v,
            "small_sports_linked_clusters_n": len(small_sports),
        },
        qa_files=[
            tables_dir / "clustering_summary.csv",
            tables_dir / "cluster_selection_metrics.csv",
            tables_dir / "clustering_key_findings.csv",
            tables_dir / "cluster_phenotype_guide.csv",
            tables_dir / "cluster_patient_profile_summary.csv",
            tables_dir / "cluster_exposure_association.csv",
            tables_dir / "clustering_sensitivity.csv",
        ],
        warnings=[
            "ExposureFlag and all follow-up outcomes are excluded from cluster construction.",
            "Cluster labels are data-derived provisional descriptions requiring clinical review.",
            "Cluster-specific follow-up trajectories are descriptive and vulnerable to regression to the mean and sparse Sports-linked cells."
        ],
        config_path=clustering_config_path,
        next_command="python scripts/check_translation_readiness.py",
    )
    stage_footer(
        stage_key="clustering",
        audit_dir=audit_dir,
        summary_path=summary_path,
        qa_files=[
            tables_dir / "clustering_key_findings.csv",
            tables_dir / "cluster_patient_profile_summary.csv",
            tables_dir / "cluster_selection_metrics.csv",
        ],
        warnings=[
            f"{len(small_sports)} cluster(s) have fewer than {small_sports_threshold} Sports-linked patients for stable trajectory interpretation."
        ] if len(small_sports) else [],
        next_command="python scripts/check_translation_readiness.py",
    )

    return outputs
