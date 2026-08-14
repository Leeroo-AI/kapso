# Imports

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# Data structures

@dataclass
class FeatureBundle:
    seeds: pd.DataFrame
    base: pd.DataFrame
    documents: list[str]
    contexts: list[str]
    keys: dict[str, list[list[str]]]
    table_coverage: dict[str, dict[str, float]]


# Utilities

def _text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def _boolean(series: pd.Series) -> pd.Series:
    values = _text(series).str.lower().str.strip()
    return values.isin(["yes", "true", "1", "y"]).astype(np.float32)


def _age_years(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(year|month|week|day)", str(value).lower())
    if match is None:
        return np.nan
    amount = float(match.group(1))
    unit = match.group(2)
    scale = {"year": 1.0, "month": 1.0 / 12.0, "week": 1.0 / 52.1775, "day": 1.0 / 365.25}[unit]
    return amount * scale


def _latest_asof(seeds: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in table.columns if column != "nct_id"]
    merged = seeds[["row_id", "nct_id", "timestamp"]].merge(table, on="nct_id", how="left")
    merged = merged[merged["date"].notna() & (merged["date"] <= merged["timestamp"])]
    merged = merged.sort_values(["row_id", "date"]).drop_duplicates("row_id", keep="last")
    return merged.set_index("row_id")[columns]


def _aligned(latest: pd.DataFrame, column: str, row_ids: pd.Series) -> pd.Series:
    if column not in latest:
        return pd.Series(np.nan, index=np.arange(len(row_ids)))
    return row_ids.map(latest[column]).reset_index(drop=True)


def _list_map(frame: pd.DataFrame, column: str, row_count: int) -> list[list[str]]:
    if len(frame) == 0 or column not in frame:
        return [[] for _ in range(row_count)]
    grouped = frame.dropna(subset=[column]).groupby("row_id")[column].agg(
        lambda values: sorted({str(value) for value in values if str(value)})
    )
    return [grouped.get(index, []) for index in range(row_count)]


def _entropy(values: pd.Series) -> float:
    counts = values.dropna().astype(str).value_counts().to_numpy(dtype=float)
    if len(counts) == 0:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities + 1e-12)).sum())


def _coverage(seeds: pd.DataFrame, counts: pd.Series) -> dict[str, float]:
    result = {}
    aligned = seeds["row_id"].map(counts).fillna(0)
    for split, index in seeds.groupby("split").groups.items():
        result[str(split)] = float((aligned.loc[index] > 0).mean())
    return result


# Study features

def _study_features(db: Any, seeds: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    studies = db.table_dict["studies"].df.copy()
    frame = seeds.merge(studies, on="nct_id", how="left", validate="many_to_one")
    base = pd.DataFrame(index=np.arange(len(frame)))
    categorical = [
        "study_type", "phase", "enrollment_type", "source", "source_class",
        "has_dmc", "is_fda_regulated_drug", "is_fda_regulated_device",
        "is_unapproved_device", "is_us_export", "plan_to_share_ipd",
    ]
    for column in categorical:
        base[f"cat_{column}"] = _text(frame[column]).replace("", "__missing__")
    numeric = ["enrollment", "number_of_arms", "number_of_groups"]
    for column in numeric:
        base[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    base["enrollment_log"] = np.log1p(base["enrollment"].clip(lower=0))
    base["trial_age_days"] = (frame["timestamp"] - frame["start_date"]).dt.total_seconds() / 86400.0
    base["origin_year"] = frame["timestamp"].dt.year.astype(np.float32)
    base["origin_month"] = frame["timestamp"].dt.month.astype(np.float32)
    base["start_year"] = frame["start_date"].dt.year.astype(np.float32)
    base["start_month"] = frame["start_date"].dt.month.astype(np.float32)
    base["study_missing_count"] = frame[studies.columns.difference(["nct_id", "start_date"])].isna().sum(axis=1).astype(np.float32)
    for column in ["brief_title", "official_title", "brief_summaries", "detailed_descriptions"]:
        values = _text(frame[column])
        base[f"{column}_length"] = values.str.len().astype(np.float32)
        base[f"{column}_words"] = values.str.count(r"\b\w+\b").astype(np.float32)
        base[f"{column}_numeric_count"] = values.str.count(r"\b\d+(?:\.\d+)?\b").astype(np.float32)
    for column in [
        "has_dmc", "is_fda_regulated_drug", "is_fda_regulated_device",
        "is_unapproved_device", "is_us_export",
    ]:
        base[f"flag_{column}"] = _boolean(frame[column])
    documents = [
        "\n".join([
            f"Brief title: {brief}",
            f"Official title: {official}",
            f"Brief summary: {summary}",
            f"Detailed description: {description}",
        ])
        for brief, official, summary, description in zip(
            _text(frame["brief_title"]), _text(frame["official_title"]),
            _text(frame["brief_summaries"]), _text(frame["detailed_descriptions"]),
        )
    ]
    return base, documents


# Design and eligibility features

def _design_eligibility_features(
    db: Any, seeds: pd.DataFrame, base: pd.DataFrame, documents: list[str]
) -> tuple[pd.DataFrame, list[str], dict[str, pd.Series]]:
    row_ids = seeds["row_id"]
    design = _latest_asof(seeds, db.table_dict["designs"].df)
    design_categories = [
        "allocation", "intervention_model", "observational_model", "primary_purpose",
        "time_perspective", "masking", "subject_masked", "caregiver_masked",
        "investigator_masked", "outcomes_assessor_masked",
    ]
    context_values: dict[str, pd.Series] = {}
    for column in design_categories:
        values = _text(_aligned(design, column, row_ids)).replace("", "__missing__")
        base[f"cat_design_{column}"] = values
        context_values[column] = values
    for column in ["masking_description", "intervention_model_description"]:
        values = _text(_aligned(design, column, row_ids))
        base[f"design_{column}_length"] = values.str.len().astype(np.float32)
        base[f"design_{column}_words"] = values.str.count(r"\b\w+\b").astype(np.float32)
        base[f"design_{column}_sentences"] = values.str.count(r"[.!?]").astype(np.float32)
    base["design_missing_count"] = pd.DataFrame(
        {column: _aligned(design, column, row_ids) for column in design_categories}
    ).isna().sum(axis=1).astype(np.float32)
    eligibility = _latest_asof(seeds, db.table_dict["eligibilities"].df)
    eligibility_categories = [
        "sampling_method", "gender", "healthy_volunteers", "adult", "child",
        "older_adult", "gender_based",
    ]
    for column in eligibility_categories:
        values = _text(_aligned(eligibility, column, row_ids)).replace("", "__missing__")
        base[f"cat_eligibility_{column}"] = values
        context_values[column] = values
    minimum = _aligned(eligibility, "minimum_age", row_ids)
    maximum = _aligned(eligibility, "maximum_age", row_ids)
    base["minimum_age_years"] = minimum.map(_age_years).astype(np.float32)
    base["maximum_age_years"] = maximum.map(_age_years).astype(np.float32)
    base["age_range_years"] = base["maximum_age_years"] - base["minimum_age_years"]
    criteria = _text(_aligned(eligibility, "criteria", row_ids))
    population = _text(_aligned(eligibility, "population", row_ids))
    base["criteria_length"] = criteria.str.len().astype(np.float32)
    base["criteria_words"] = criteria.str.count(r"\b\w+\b").astype(np.float32)
    base["criteria_lines"] = criteria.str.count(r"\n").add(1).astype(np.float32)
    base["criteria_numeric_thresholds"] = criteria.str.count(r"(?:<=|>=|<|>|≤|≥)\s*\d|\b\d+(?:\.\d+)?\s*(?:mg|kg|mm|cm|years?|months?|days?|%)\b", flags=re.I).astype(np.float32)
    base["criteria_inclusion_markers"] = criteria.str.count(r"\binclusion\b|\bincluded\b|\beligible\b", flags=re.I).astype(np.float32)
    base["criteria_exclusion_markers"] = criteria.str.count(r"\bexclusion\b|\bexcluded\b|\bnot eligible\b", flags=re.I).astype(np.float32)
    patterns = {
        "cancer_stage": r"\bstage\s*(?:[0-4ivx]+)",
        "biomarker": r"\b(?:biomarker|mutation|receptor|her2|egfr|pd-l1|genotype)\b",
        "prior_treatment": r"\b(?:prior|previous|pretreat|treatment-naive|therapy)\b",
        "pregnancy": r"\b(?:pregnan|lactat|breastfeed|contracepti)\w*",
        "comorbidity": r"\b(?:comorbid|renal|hepatic|cardiac|cardiovascular|diabetes)\w*",
        "performance_status": r"\b(?:ecog|karnofsky|performance status)\b",
    }
    for name, pattern in patterns.items():
        base[f"criteria_{name}_count"] = criteria.str.count(pattern, flags=re.I).astype(np.float32)
    base["population_length"] = population.str.len().astype(np.float32)
    documents = [f"{document}\nEligibility population: {pop}\nEligibility criteria: {crit}" for document, pop, crit in zip(documents, population, criteria)]
    return base, documents, context_values


# Event table features

def _event_features(db: Any, seeds: pd.DataFrame, base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    coverage: dict[str, dict[str, float]] = {}
    event_specs = {
        "outcomes": ("outcome_id", "id", ["outcome_type", "title", "description", "time_frame"]),
        "drop_withdrawals": ("withdrawal_id", "id", ["reason", "period"]),
        "reported_event_totals": ("event_id", "id", ["event_type", "classification"]),
    }
    for table_name, (prefix_id, id_column, text_columns) in event_specs.items():
        table = db.table_dict[table_name].df
        merged = seeds[["row_id", "nct_id", "timestamp"]].merge(table, on="nct_id", how="left")
        merged = merged[merged["date"].notna() & (merged["date"] <= merged["timestamp"])]
        grouped = merged.groupby("row_id")
        counts = grouped[id_column].count() if len(merged) else pd.Series(dtype=float)
        coverage[table_name] = _coverage(seeds, counts)
        base[f"{table_name}_count"] = seeds["row_id"].map(counts).fillna(0).astype(np.float32)
        if len(merged):
            recency = grouped.apply(lambda values: (values["timestamp"].iloc[0] - values["date"].max()).days, include_groups=False)
            base[f"{table_name}_recency_days"] = seeds["row_id"].map(recency).astype(np.float32)
            for column in text_columns:
                unique = grouped[column].nunique()
                lengths = merged.assign(_length=_text(merged[column]).str.len()).groupby("row_id")["_length"].agg(["mean", "max"])
                base[f"{table_name}_{column}_unique"] = seeds["row_id"].map(unique).fillna(0).astype(np.float32)
                base[f"{table_name}_{column}_mean_length"] = seeds["row_id"].map(lengths["mean"]).fillna(0).astype(np.float32)
        else:
            base[f"{table_name}_recency_days"] = np.nan
        if table_name == "drop_withdrawals":
            totals = grouped["count"].sum(min_count=1) if len(merged) else pd.Series(dtype=float)
            base["withdrawal_total"] = seeds["row_id"].map(totals).fillna(0).astype(np.float32)
            base["withdrawal_enrollment_ratio"] = base["withdrawal_total"] / base["enrollment"].clip(lower=1)
        if table_name == "reported_event_totals":
            affected = grouped["subjects_affected"].sum(min_count=1) if len(merged) else pd.Series(dtype=float)
            risk = grouped["subjects_at_risk"].sum(min_count=1) if len(merged) else pd.Series(dtype=float)
            base["event_subjects_affected"] = seeds["row_id"].map(affected).fillna(0).astype(np.float32)
            base["event_subjects_at_risk"] = seeds["row_id"].map(risk).fillna(0).astype(np.float32)
            base["event_affected_ratio"] = base["event_subjects_affected"] / base["event_subjects_at_risk"].clip(lower=1)
    outcomes = db.table_dict["outcomes"].df[["id", "nct_id", "outcome_type"]]
    analyses = db.table_dict["outcome_analyses"].df.merge(outcomes, left_on="outcome_id", right_on="id", suffixes=("", "_outcome"))
    valid = analyses[
        (analyses["outcome_type"] == "Primary")
        & (analyses["p_value"].between(0, 1, inclusive="both"))
        & (analyses["p_value_modifier"].isna() | (analyses["p_value_modifier"] != ">"))
    ]
    merged = seeds[["row_id", "nct_id", "timestamp"]].merge(valid, on="nct_id", how="left")
    merged = merged[merged["date"].notna() & (merged["date"] <= merged["timestamp"])]
    grouped = merged.groupby("row_id")
    counts = grouped["id"].count() if len(merged) else pd.Series(dtype=float)
    coverage["outcome_analyses"] = _coverage(seeds, counts)
    base["valid_primary_analysis_count"] = seeds["row_id"].map(counts).fillna(0).astype(np.float32)
    if len(merged):
        summary = grouped["p_value"].agg(["min", "mean", "max", "std"])
        for column in summary:
            base[f"historical_p_value_{column}"] = seeds["row_id"].map(summary[column]).astype(np.float32)
        base["historical_significant"] = (base["historical_p_value_min"] <= 0.05).astype(np.float32)
        base["historical_analysis_recency_days"] = seeds["row_id"].map(
            grouped.apply(lambda values: (values["timestamp"].iloc[0] - values["date"].max()).days, include_groups=False)
        ).astype(np.float32)
        base["historical_analysis_cadence_days"] = seeds["row_id"].map(
            grouped["date"].apply(lambda values: values.sort_values().diff().dt.days.mean())
        ).astype(np.float32)
        for column in ["method", "non_inferiority_type", "param_type", "dispersion_type"]:
            base[f"analysis_{column}_unique"] = seeds["row_id"].map(grouped[column].nunique()).fillna(0).astype(np.float32)
        for column in ["ci_lower_limit", "ci_upper_limit", "ci_percent"]:
            base[f"analysis_{column}_mean"] = seeds["row_id"].map(grouped[column].mean()).astype(np.float32)
        base["analysis_noninferiority_count"] = seeds["row_id"].map(grouped["non_inferiority_type"].count()).fillna(0).astype(np.float32)
    else:
        for column in ["min", "mean", "max", "std"]:
            base[f"historical_p_value_{column}"] = np.nan
        base["historical_significant"] = 0.0
        base["historical_analysis_recency_days"] = np.nan
        base["historical_analysis_cadence_days"] = np.nan
    return base, coverage


# Relational features

def _degree_features(
    base: pd.DataFrame,
    seeds: pd.DataFrame,
    matched: pd.DataFrame,
    full_relation: pd.DataFrame,
    node_column: str,
    prefix: str,
) -> pd.DataFrame:
    if len(matched) == 0:
        for suffix in ["degree_mean", "degree_max", "degree_min", "degree_rarity", "degree_concentration", "novelty_share", "batch_degree_mean", "batch_degree_max"]:
            base[f"{prefix}_{suffix}"] = 0.0
        return base
    first_date = full_relation.groupby(node_column)["date"].min()
    pieces = []
    for timestamp, current in matched.groupby("timestamp", sort=False):
        nodes = pd.Index(current[node_column].dropna().unique())
        historical = full_relation[(full_relation["date"] <= timestamp) & full_relation[node_column].isin(nodes)]
        degrees = historical.groupby(node_column)["nct_id"].nunique()
        piece = current[["row_id", node_column, "timestamp"]].copy()
        piece["degree"] = piece[node_column].map(degrees).fillna(0)
        piece["novel"] = ((timestamp - piece[node_column].map(first_date)).dt.days <= 365).astype(float)
        batch = current.groupby(node_column)["row_id"].nunique()
        piece["batch_degree"] = piece[node_column].map(batch).fillna(0)
        pieces.append(piece)
    records = pd.concat(pieces, ignore_index=True)
    grouped = records.groupby("row_id")
    summary = grouped["degree"].agg(["mean", "max", "min", "sum"])
    base[f"{prefix}_degree_mean"] = seeds["row_id"].map(summary["mean"]).fillna(0).astype(np.float32)
    base[f"{prefix}_degree_max"] = seeds["row_id"].map(summary["max"]).fillna(0).astype(np.float32)
    base[f"{prefix}_degree_min"] = seeds["row_id"].map(summary["min"]).fillna(0).astype(np.float32)
    base[f"{prefix}_degree_rarity"] = seeds["row_id"].map(grouped["degree"].apply(lambda values: float((values <= 2).mean()))).fillna(0).astype(np.float32)
    concentration = summary["max"] / summary["sum"].clip(lower=1)
    base[f"{prefix}_degree_concentration"] = seeds["row_id"].map(concentration).fillna(0).astype(np.float32)
    base[f"{prefix}_novelty_share"] = seeds["row_id"].map(grouped["novel"].mean()).fillna(0).astype(np.float32)
    batch = grouped["batch_degree"].agg(["mean", "max"])
    base[f"{prefix}_batch_degree_mean"] = seeds["row_id"].map(batch["mean"]).fillna(0).astype(np.float32)
    base[f"{prefix}_batch_degree_max"] = seeds["row_id"].map(batch["max"]).fillna(0).astype(np.float32)
    return base


def _relational_features(
    db: Any, seeds: pd.DataFrame, base: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, list[list[str]]], dict[str, dict[str, float]], dict[str, list[str]]]:
    keys: dict[str, list[list[str]]] = {}
    display: dict[str, list[str]] = {}
    coverage: dict[str, dict[str, float]] = {}
    specifications = [
        ("condition", "conditions_studies", "conditions", "condition_id", ["mesh_term"]),
        ("intervention", "interventions_studies", "interventions", "intervention_id", ["mesh_term"]),
        ("sponsor", "sponsors_studies", "sponsors", "sponsor_id", ["name", "agency_class"]),
        ("facility", "facilities_studies", "facilities", "facility_id", ["name", "city", "state", "country"]),
    ]
    for prefix, relation_name, dimension_name, node_column, dimension_columns in specifications:
        relation = db.table_dict[relation_name].df
        dimension = db.table_dict[dimension_name].df
        matched = seeds[["row_id", "nct_id", "timestamp"]].merge(relation, on="nct_id", how="left")
        matched = matched[matched["date"].notna() & (matched["date"] <= matched["timestamp"])]
        matched = matched.merge(dimension, on=node_column, how="left")
        counts = matched.groupby("row_id")[node_column].nunique() if len(matched) else pd.Series(dtype=float)
        coverage[relation_name] = _coverage(seeds, counts)
        base[f"{prefix}_count"] = seeds["row_id"].map(counts).fillna(0).astype(np.float32)
        keys[prefix] = _list_map(matched, node_column, len(seeds))
        for column in dimension_columns:
            unique = matched.groupby("row_id")[column].nunique() if len(matched) else pd.Series(dtype=float)
            base[f"{prefix}_{column}_unique"] = seeds["row_id"].map(unique).fillna(0).astype(np.float32)
            if column in ["mesh_term", "name", "country", "agency_class"]:
                display[f"{prefix}_{column}"] = [", ".join(values[:12]) for values in _list_map(matched, column, len(seeds))]
        if prefix == "facility":
            entropies = matched.groupby("row_id")["country"].apply(_entropy) if len(matched) else pd.Series(dtype=float)
            us_share = matched.groupby("row_id")["country"].apply(lambda values: float(_text(values).str.lower().isin(["united states", "united states of america", "us", "usa"]).mean())) if len(matched) else pd.Series(dtype=float)
            base["facility_country_entropy"] = seeds["row_id"].map(entropies).fillna(0).astype(np.float32)
            base["facility_us_share"] = seeds["row_id"].map(us_share).fillna(0).astype(np.float32)
            keys["country"] = _list_map(matched, "country", len(seeds))
        if prefix == "sponsor":
            lead_mask = _text(matched["lead_or_collaborator"]).str.lower().str.contains("lead") if len(matched) else pd.Series(dtype=bool)
            lead = matched[lead_mask]
            keys["lead_sponsor"] = _list_map(lead, node_column, len(seeds))
            keys["sponsor_class"] = _list_map(matched, "agency_class", len(seeds))
            lead_name = _list_map(lead, "name", len(seeds))
            base["cat_lead_sponsor"] = [values[0] if values else "__missing__" for values in lead_name]
            base["cat_sponsor_class"] = [values[0] if values else "__missing__" for values in keys["sponsor_class"]]
        base = _degree_features(base, seeds, matched, relation, node_column, prefix)
    return base, keys, coverage, display


# Cohort and key features

def _cross(left: list[str], right: list[str], limit: int = 256) -> list[str]:
    values = [f"{a}|{b}" for a in left for b in right]
    return values[:limit]


def _make_prior_keys(base: pd.DataFrame, keys: dict[str, list[list[str]]]) -> dict[str, list[list[str]]]:
    phase = [[str(value)] for value in base["cat_phase"]]
    source = [[str(value)] for value in base["cat_source_class"]]
    keys["phase"] = phase
    keys["source_class"] = source
    keys["phase_source"] = [[f"{a[0]}|{b[0]}"] for a, b in zip(phase, source)]
    keys["condition_phase"] = [_cross(condition, phase_value) for condition, phase_value in zip(keys["condition"], phase)]
    keys["condition_intervention"] = [_cross(condition, intervention) for condition, intervention in zip(keys["condition"], keys["intervention"])]
    keys["sponsor_condition"] = [_cross(sponsor, condition) for sponsor, condition in zip(keys["lead_sponsor"], keys["condition"])]
    keys["sponsor_phase"] = [_cross(sponsor, phase_value) for sponsor, phase_value in zip(keys["lead_sponsor"], phase)]
    return keys


def _within_origin(base: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    informative = [
        "enrollment_log", "trial_age_days", "number_of_arms", "number_of_groups",
        "brief_summaries_length", "detailed_descriptions_length", "criteria_length",
        "minimum_age_years", "maximum_age_years", "age_range_years", "criteria_lines",
        "criteria_numeric_thresholds", "condition_count", "intervention_count",
        "sponsor_count", "facility_count", "facility_country_unique", "condition_degree_mean",
        "intervention_degree_mean", "sponsor_degree_mean", "facility_degree_mean",
        "valid_primary_analysis_count", "outcomes_count", "drop_withdrawals_count",
        "reported_event_totals_count",
    ]
    groups = seeds.groupby("timestamp").groups
    for column in informative:
        if column not in base:
            continue
        values = pd.to_numeric(base[column], errors="coerce")
        rank = pd.Series(np.nan, index=base.index, dtype=float)
        robust = pd.Series(np.nan, index=base.index, dtype=float)
        gap = pd.Series(np.nan, index=base.index, dtype=float)
        for indices in groups.values():
            current = values.loc[indices]
            rank.loc[indices] = current.rank(method="average", pct=True)
            median = current.median()
            scale = (current - median).abs().median()
            robust.loc[indices] = (current - median) / (1.4826 * scale + 1e-6)
            gap.loc[indices] = current.max() - current
        base[f"origin_rank_{column}"] = rank.astype(np.float32)
        base[f"origin_robust_z_{column}"] = robust.clip(-20, 20).astype(np.float32)
        base[f"origin_leader_gap_{column}"] = gap.astype(np.float32)
    base["row_missing_count"] = base.isna().sum(axis=1).astype(np.float32)
    base["origin_missing_rank"] = base["row_missing_count"].groupby(seeds["timestamp"]).rank(pct=True).astype(np.float32)
    base["origin_cohort_size"] = seeds.groupby("timestamp")["row_id"].transform("size").astype(np.float32)
    phase_counts = pd.DataFrame({"timestamp": seeds["timestamp"], "phase": base["cat_phase"]}).groupby(["timestamp", "phase"])["phase"].transform("size")
    base["origin_phase_share"] = (phase_counts / base["origin_cohort_size"]).astype(np.float32)
    base["origin_condition_mean"] = base["condition_count"].groupby(seeds["timestamp"]).transform("mean").astype(np.float32)
    base["origin_intervention_missing_share"] = (base["intervention_count"] == 0).groupby(seeds["timestamp"]).transform("mean").astype(np.float32)
    base["origin_history_zero_share"] = (base["valid_primary_analysis_count"] == 0).groupby(seeds["timestamp"]).transform("mean").astype(np.float32)
    return base


# Public builder

def build_feature_bundle(db: Any, seeds: pd.DataFrame) -> FeatureBundle:
    seeds = seeds.reset_index(drop=True).copy()
    seeds["row_id"] = np.arange(len(seeds), dtype=np.int64)
    base, documents = _study_features(db, seeds)
    base, documents, design_context = _design_eligibility_features(db, seeds, base, documents)
    base, event_coverage = _event_features(db, seeds, base)
    base, keys, relation_coverage, display = _relational_features(db, seeds, base)
    keys = _make_prior_keys(base, keys)
    base = _within_origin(base, seeds)
    contexts = []
    for index in range(len(seeds)):
        fields = [
            f"origin: {seeds.at[index, 'timestamp'].date()}",
            f"phase: {base.at[index, 'cat_phase']}",
            f"study type: {base.at[index, 'cat_study_type']}",
            f"source class: {base.at[index, 'cat_source_class']}",
            f"planned enrollment: {base.at[index, 'enrollment']}",
            f"arms: {base.at[index, 'number_of_arms']}",
            f"groups: {base.at[index, 'number_of_groups']}",
            f"allocation: {design_context['allocation'].iat[index]}",
            f"intervention model: {design_context['intervention_model'].iat[index]}",
            f"primary purpose: {design_context['primary_purpose'].iat[index]}",
            f"masking: {design_context['masking'].iat[index]}",
            f"gender: {design_context['gender'].iat[index]}",
            f"healthy volunteers: {design_context['healthy_volunteers'].iat[index]}",
            f"condition terms: {display.get('condition_mesh_term', [''])[index]}",
            f"intervention terms: {display.get('intervention_mesh_term', [''])[index]}",
            f"sponsor classes: {display.get('sponsor_agency_class', [''])[index]}",
            f"countries: {display.get('facility_country', [''])[index]}",
            f"site count: {base.at[index, 'facility_count']}",
        ]
        contexts.append("\n".join(fields))
    coverage = {**event_coverage, **relation_coverage}
    return FeatureBundle(seeds=seeds, base=base, documents=documents, contexts=contexts, keys=keys, table_coverage=coverage)
