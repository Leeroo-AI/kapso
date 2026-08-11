from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_VERSION = "lane0_censored_all_tables_v5"
HALF_LIFE_DAYS = 1461.0


def _read(root: Path, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(root / "db" / f"{name}.parquet", columns=columns)


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _safe_key(value: object) -> str:
    value = _text(value)
    return value if value else "__MISSING__"


def _age_years(value: object) -> float:
    value = _text(value).lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(year|month|week|day|hour)", value)
    if match is None:
        return np.nan
    amount = float(match.group(1))
    unit = match.group(2)
    factors = {"year": 1.0, "month": 1.0 / 12.0, "week": 1.0 / 52.1775, "day": 1.0 / 365.25, "hour": 1.0 / 8766.0}
    return amount * factors[unit]


def _duration_days(value: object) -> float:
    value = _text(value).lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(year|month|week|day|hour)", value)
    if match is None:
        return np.nan
    amount = float(match.group(1))
    unit = match.group(2)
    factors = {"year": 365.25, "month": 30.4375, "week": 7.0, "day": 1.0, "hour": 1.0 / 24.0}
    return amount * factors[unit]


def _flag(value: object) -> float:
    value = _text(value).lower()
    if value in {"t", "true", "yes", "1", "accepts healthy volunteers"}:
        return 1.0
    if value in {"f", "false", "no", "0"}:
        return 0.0
    return np.nan


def _task_rows(root: Path) -> pd.DataFrame:
    frames = []
    offset = 0
    for split in ["train", "val", "test"]:
        frame = pd.read_parquet(root / "tasks" / "study-outcome" / f"{split}.parquet")
        frame = frame.copy()
        frame["_split"] = split
        frame["_row_id"] = np.arange(len(frame), dtype=np.int64)
        frame["_global_id"] = np.arange(offset, offset + len(frame), dtype=np.int64)
        offset += len(frame)
        if "outcome" not in frame:
            frame["outcome"] = np.nan
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"])
    return rows


def _latest_for_rows(rows: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    joined = rows[["_global_id", "nct_id", "timestamp"]].merge(table, on="nct_id", how="left")
    joined = joined[joined["date"].isna() | (joined["date"] <= joined["timestamp"])]
    joined = joined.sort_values(["_global_id", "date"], kind="stable")
    joined = joined.groupby("_global_id", sort=False, as_index=False).tail(1)
    return joined.drop(columns=["nct_id", "timestamp", "date"], errors="ignore")


def _query_edges(rows: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    joined = rows[["_global_id", "nct_id", "timestamp"]].merge(edges, on="nct_id", how="left")
    joined = joined[joined["date"].notna() & (joined["date"] <= joined["timestamp"])]
    return joined


def _frequency_stats(query: pd.DataFrame, edges: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    parts = []
    for timestamp, block in query.groupby("timestamp", sort=False):
        counts = edges.loc[edges["date"] <= timestamp, key].value_counts()
        current = block[["_global_id", key]].copy()
        current["_frequency"] = current[key].map(counts).astype(float)
        parts.append(current)
    if not parts:
        return pd.DataFrame({"_global_id": []})
    values = pd.concat(parts, ignore_index=True)
    grouped = values.groupby("_global_id")["_frequency"]
    result = grouped.agg(["mean", "max", "min"]).rename(columns={x: f"{prefix}_entity_freq_{x}" for x in ["mean", "max", "min"]})
    result[f"{prefix}_rare_frac_2"] = grouped.apply(lambda x: float(np.mean(x <= 2)))
    result[f"{prefix}_rare_frac_5"] = grouped.apply(lambda x: float(np.mean(x <= 5)))
    return result.reset_index()


def _add_relation_counts(features: pd.DataFrame, query: pd.DataFrame, edges: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    counts = query.groupby("_global_id")[key].agg(["count", "nunique"]).rename(columns={"count": f"{prefix}_count", "nunique": f"{prefix}_unique"}).reset_index()
    frequencies = _frequency_stats(query, edges, key, prefix)
    features = features.merge(counts, on="_global_id", how="left")
    features = features.merge(frequencies, on="_global_id", how="left")
    return features


def _make_study_features(rows: pd.DataFrame, studies: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str], dict[int, str], dict[int, str]]:
    selected = studies[studies["nct_id"].isin(rows["nct_id"])].copy()
    merged = rows.merge(selected, on="nct_id", how="left", suffixes=("", "_study"))
    result = rows[["_global_id", "_split", "_row_id", "nct_id", "timestamp", "outcome"]].copy()
    categorical = ["study_type", "phase", "enrollment_type", "source_class", "source", "plan_to_share_ipd", "biospec_retention"]
    for column in categorical:
        result[f"study_{column}"] = merged[column].map(_safe_key)
    numeric = ["enrollment", "number_of_arms", "number_of_groups"]
    for column in numeric:
        result[f"study_{column}"] = pd.to_numeric(merged[column], errors="coerce")
    result["study_log_enrollment"] = np.log1p(result["study_enrollment"].clip(lower=0))
    result["study_enrollment_per_arm"] = result["study_enrollment"] / result["study_number_of_arms"].clip(lower=1)
    result["study_enrollment_per_group"] = result["study_enrollment"] / result["study_number_of_groups"].clip(lower=1)
    start_date = pd.to_datetime(merged["start_date"])
    result["study_trial_age_days"] = (rows["timestamp"] - start_date).dt.total_seconds() / 86400.0
    result["study_start_year"] = start_date.dt.year.astype(float)
    result["study_start_month"] = start_date.dt.month.astype(float)
    result["study_target_duration_days"] = merged["target_duration"].map(_duration_days)
    flags = ["has_dmc", "is_fda_regulated_drug", "is_fda_regulated_device", "is_unapproved_device", "is_us_export", "fdaaa801_violation"]
    for column in flags:
        result[f"study_{column}"] = merged[column].map(_flag)
    text_columns = ["brief_title", "official_title", "brief_summaries", "detailed_descriptions", "acronym"]
    for column in text_columns:
        values = merged[column].fillna("").astype(str)
        result[f"study_{column}_chars"] = values.str.len().astype(float)
        result[f"study_{column}_words"] = values.str.split().str.len().astype(float)
    missing_columns = ["phase", "enrollment", "number_of_arms", "number_of_groups", "official_title", "detailed_descriptions", "has_dmc", "plan_to_share_ipd"]
    result["study_missing_count"] = merged[missing_columns].isna().sum(axis=1).astype(float)
    titles = dict(zip(merged["nct_id"].astype(int), merged["brief_title"].map(_text)))
    summaries = dict(zip(merged["nct_id"].astype(int), merged["brief_summaries"].map(_text)))
    phases = dict(zip(studies["nct_id"].astype(int), studies["phase"].map(_safe_key)))
    return result, titles, summaries, phases


def _add_design_features(features: pd.DataFrame, rows: pd.DataFrame, designs: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    latest = _latest_for_rows(rows, designs)
    categorical = ["allocation", "intervention_model", "observational_model", "primary_purpose", "time_perspective", "masking"]
    keep = latest[["_global_id"]].copy()
    for column in categorical:
        keep[f"design_{column}"] = latest[column].map(_safe_key)
    roles = ["subject_masked", "caregiver_masked", "investigator_masked", "outcomes_assessor_masked"]
    keep["design_masked_role_count"] = latest[roles].apply(lambda x: sum(_flag(v) == 1.0 for v in x), axis=1).astype(float)
    keep["design_missing_count"] = latest[categorical + roles].isna().sum(axis=1).astype(float)
    descriptions = latest["masking_description"].fillna("").astype(str) + " " + latest["intervention_model_description"].fillna("").astype(str)
    keep["design_description_chars"] = descriptions.str.len().astype(float)
    features = features.merge(keep, on="_global_id", how="left")
    design_text = {}
    all_designs = designs.sort_values(["nct_id", "date"], kind="stable").groupby("nct_id", as_index=False).tail(1)
    for row in all_designs.itertuples(index=False):
        parts = [_text(getattr(row, column)) for column in categorical]
        design_text[int(row.nct_id)] = " | ".join(part for part in parts if part)
    return features, design_text


def _eligibility_chunks(criteria: str) -> list[str]:
    criteria = _text(criteria)
    if not criteria:
        return [""]
    lower = criteria.lower()
    inclusion = lower.find("inclusion")
    exclusion = lower.find("exclusion")
    if inclusion >= 0 and exclusion > inclusion:
        first = criteria[inclusion:exclusion].strip()
        last = criteria[exclusion:].strip()
        middle_source = criteria
    else:
        first = criteria[:2200].strip()
        last = criteria[-2200:].strip()
        middle_source = criteria
    chunks = [first]
    if len(middle_source) > 4400:
        center = len(middle_source) // 2
        chunks.append(middle_source[max(0, center - 1100):center + 1100].strip())
    if last and last != first:
        chunks.append(last)
    return [chunk for chunk in chunks[:3] if chunk] or [""]


def _add_eligibility_features(features: pd.DataFrame, rows: pd.DataFrame, eligibilities: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    latest = _latest_for_rows(rows, eligibilities)
    keep = latest[["_global_id"]].copy()
    categorical = ["sampling_method", "gender", "healthy_volunteers", "adult", "child", "older_adult"]
    for column in categorical:
        keep[f"elig_{column}"] = latest[column].map(_safe_key)
    keep["elig_min_age"] = latest["minimum_age"].map(_age_years)
    keep["elig_max_age"] = latest["maximum_age"].map(_age_years)
    keep["elig_age_span"] = keep["elig_max_age"] - keep["elig_min_age"]
    criteria = latest["criteria"].fillna("").astype(str)
    lower = criteria.str.lower()
    keep["elig_criteria_chars"] = criteria.str.len().astype(float)
    keep["elig_criteria_words"] = criteria.str.split().str.len().astype(float)
    keep["elig_bullet_count"] = criteria.str.count(r"(?m)(^\s*[-*•]|^\s*\d+[.)])").astype(float)
    keep["elig_numeric_threshold_count"] = criteria.str.count(r"(?:<=|>=|<|>|≤|≥)\s*\d|\b\d+(?:\.\d+)?\s*(?:mg|kg|mm|cm|%|years?|months?|days?)\b").astype(float)
    inclusion_pos = lower.str.find("inclusion")
    exclusion_pos = lower.str.find("exclusion")
    keep["elig_has_inclusion"] = (inclusion_pos >= 0).astype(float)
    keep["elig_has_exclusion"] = (exclusion_pos >= 0).astype(float)
    keep["elig_inclusion_chars"] = np.where((inclusion_pos >= 0) & (exclusion_pos > inclusion_pos), exclusion_pos - inclusion_pos, keep["elig_criteria_chars"] * 0.5)
    keep["elig_exclusion_chars"] = np.where(exclusion_pos >= 0, keep["elig_criteria_chars"] - exclusion_pos, keep["elig_criteria_chars"] * 0.5)
    restriction_terms = lower.str.count(r"\b(no|not|must|exclude|excluding|without|prohibited|unable|required)\b").astype(float)
    keep["elig_restriction_density"] = restriction_terms / keep["elig_criteria_words"].clip(lower=1)
    keep["elig_missing_count"] = latest[["minimum_age", "maximum_age", "gender", "healthy_volunteers", "criteria"]].isna().sum(axis=1).astype(float)
    features = features.merge(keep, on="_global_id", how="left")
    all_elig = eligibilities.sort_values(["nct_id", "date"], kind="stable").groupby("nct_id", as_index=False).tail(1)
    chunks = {int(row.nct_id): _eligibility_chunks(row.criteria) for row in all_elig.itertuples(index=False)}
    return features, chunks


def _relation_block(features: pd.DataFrame, rows: pd.DataFrame, root: Path, name: str, key: str, prefix: str, lookup_name: str, lookup_key: str, lookup_text: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, list[str]]]:
    edges = _read(root, name, ["nct_id", key, "date"])
    edges["date"] = pd.to_datetime(edges["date"])
    query = _query_edges(rows, edges)
    features = _add_relation_counts(features, query, edges, key, prefix)
    query_keys = query[["_global_id", "nct_id", key]].drop_duplicates().rename(columns={key: "key"})
    history_keys = edges[["nct_id", key, "date"]].drop_duplicates().rename(columns={key: "key"})
    lookup = _read(root, lookup_name, [lookup_key, lookup_text])
    text_map = lookup.set_index(lookup_key)[lookup_text].map(_text)
    terms = edges[edges["nct_id"].isin(set(rows["nct_id"]))].copy()
    terms["term"] = terms[key].map(text_map).fillna("")
    text_terms = terms.groupby("nct_id")["term"].apply(lambda x: list(dict.fromkeys(v for v in x if v))).to_dict()
    return features, query_keys, history_keys, {int(k): v for k, v in text_terms.items()}


def _sponsor_block(features: pd.DataFrame, rows: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edges = _read(root, "sponsors_studies", ["nct_id", "sponsor_id", "lead_or_collaborator", "date"])
    edges["date"] = pd.to_datetime(edges["date"])
    sponsors = _read(root, "sponsors", ["sponsor_id", "name", "agency_class"])
    query = _query_edges(rows, edges)
    query = query.merge(sponsors, on="sponsor_id", how="left")
    aggregates = query.groupby("_global_id").agg(sponsor_count=("sponsor_id", "count"), sponsor_unique=("sponsor_id", "nunique"), sponsor_collaborators=("lead_or_collaborator", lambda x: int(np.sum(x == "collaborator")))).reset_index()
    lead = query[query["lead_or_collaborator"] == "lead"].sort_values(["_global_id", "date"], kind="stable").groupby("_global_id", as_index=False).tail(1)
    aggregates = aggregates.merge(lead[["_global_id", "sponsor_id", "name", "agency_class"]], on="_global_id", how="left")
    aggregates = aggregates.rename(columns={"sponsor_id": "sponsor_lead_id", "name": "sponsor_lead_name", "agency_class": "sponsor_lead_class"})
    aggregates["sponsor_lead_name"] = aggregates["sponsor_lead_name"].map(_safe_key)
    aggregates["sponsor_lead_class"] = aggregates["sponsor_lead_class"].map(_safe_key)
    features = features.merge(aggregates, on="_global_id", how="left")
    frequencies = _frequency_stats(query, edges, "sponsor_id", "sponsor")
    features = features.merge(frequencies, on="_global_id", how="left")
    query_keys = lead[["_global_id", "nct_id", "sponsor_id"]].drop_duplicates().rename(columns={"sponsor_id": "key"})
    history_keys = edges[edges["lead_or_collaborator"] == "lead"][["nct_id", "sponsor_id", "date"]].drop_duplicates().rename(columns={"sponsor_id": "key"})
    return features, query_keys, history_keys


def _facility_block(features: pd.DataFrame, rows: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edges = _read(root, "facilities_studies", ["nct_id", "facility_id", "date"])
    edges["date"] = pd.to_datetime(edges["date"])
    facilities = _read(root, "facilities", ["facility_id", "city", "state", "country"])
    query = _query_edges(rows, edges).merge(facilities, on="facility_id", how="left")
    aggregates = query.groupby("_global_id").agg(
        site_count=("facility_id", "count"),
        site_unique=("facility_id", "nunique"),
        site_country_count=("country", "nunique"),
        site_city_count=("city", "nunique"),
        site_state_count=("state", "nunique"),
        site_us_fraction=("country", lambda x: float(np.mean(x.fillna("").str.lower().isin(["united states", "united states of america", "usa"])))),
    ).reset_index()
    features = features.merge(aggregates, on="_global_id", how="left")
    frequencies = _frequency_stats(query, edges, "facility_id", "facility")
    features = features.merge(frequencies, on="_global_id", how="left")
    query_facility = query[["_global_id", "nct_id", "facility_id"]].drop_duplicates().rename(columns={"facility_id": "key"})
    history_facility = edges[["nct_id", "facility_id", "date"]].drop_duplicates().rename(columns={"facility_id": "key"})
    query_country = query[["_global_id", "nct_id", "country"]].dropna().drop_duplicates().rename(columns={"country": "key"})
    history_country = edges.merge(facilities[["facility_id", "country"]], on="facility_id", how="left")[["nct_id", "country", "date"]].dropna().drop_duplicates().rename(columns={"country": "key"})
    return features, query_facility, history_facility, query_country, history_country


def _historical_evidence(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = _read(root, "outcomes", ["id", "nct_id", "outcome_type"])
    primary = outcomes[outcomes["outcome_type"] == "Primary"][["id", "nct_id"]].rename(columns={"id": "outcome_id", "nct_id": "outcome_nct_id"})
    analyses = _read(root, "outcome_analyses", ["nct_id", "outcome_id", "p_value_modifier", "p_value", "date"])
    valid = analyses[(analyses["p_value_modifier"].isna()) | (analyses["p_value_modifier"] != ">")]
    valid = valid[valid["p_value"].between(0.0, 1.0, inclusive="both")]
    valid = valid.merge(primary, on="outcome_id", how="inner")
    valid = valid[valid["nct_id"] == valid["outcome_nct_id"]][["nct_id", "p_value", "date"]]
    valid["date"] = pd.to_datetime(valid["date"])
    withdrawals = _read(root, "drop_withdrawals", ["nct_id", "count", "date"])
    withdrawals["date"] = pd.to_datetime(withdrawals["date"])
    serious = _read(root, "reported_event_totals", ["nct_id", "event_type", "subjects_affected", "subjects_at_risk", "date"])
    serious = serious[serious["event_type"] == "serious"].copy()
    serious["date"] = pd.to_datetime(serious["date"])
    return valid, withdrawals, serious


def _snapshot_evidence(timestamp: pd.Timestamp, valid: pd.DataFrame, withdrawals: pd.DataFrame, serious: pd.DataFrame, enrollment: pd.Series) -> pd.DataFrame:
    current = valid[valid["date"] <= timestamp]
    evidence = current.groupby("nct_id").agg(min_p=("p_value", "min"), analysis_count=("p_value", "size"), evidence_date=("date", "max")).reset_index()
    evidence["success"] = (evidence["min_p"] <= 0.05).astype(float)
    drop = withdrawals[withdrawals["date"] <= timestamp].groupby("nct_id")["count"].sum(min_count=1)
    affected = serious[serious["date"] <= timestamp].groupby("nct_id")["subjects_affected"].sum(min_count=1)
    risk = serious[serious["date"] <= timestamp].groupby("nct_id")["subjects_at_risk"].sum(min_count=1)
    denominator = evidence["nct_id"].map(enrollment).clip(lower=1)
    evidence["dropout_burden"] = np.log1p(evidence["nct_id"].map(drop).fillna(0).clip(lower=0) / denominator)
    evidence["serious_burden"] = (evidence["nct_id"].map(affected).fillna(0) / evidence["nct_id"].map(risk).replace(0, np.nan)).clip(0, 1).fillna(0)
    age = np.maximum((timestamp - evidence["evidence_date"]).dt.total_seconds().to_numpy() / 86400.0, 0.0)
    evidence["weight"] = np.power(0.5, age / HALF_LIFE_DAYS)
    return evidence


def _cohort_table(evidence: pd.DataFrame, mapping: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    current = mapping[mapping["date"] <= timestamp][["nct_id", "key"]].dropna().drop_duplicates()
    current = current.merge(evidence, on="nct_id", how="inner")
    if current.empty:
        return pd.DataFrame(columns=["key", "sum_w", "sum_y", "sum_w2", "sum_analysis", "sum_dropout", "sum_serious", "n_trials"])
    current["weighted_y"] = current["weight"] * current["success"]
    current["weight2"] = current["weight"] ** 2
    current["weighted_analysis"] = current["weight"] * current["analysis_count"]
    current["weighted_dropout"] = current["weight"] * current["dropout_burden"]
    current["weighted_serious"] = current["weight"] * current["serious_burden"]
    return current.groupby("key", as_index=False).agg(sum_w=("weight", "sum"), sum_y=("weighted_y", "sum"), sum_w2=("weight2", "sum"), sum_analysis=("weighted_analysis", "sum"), sum_dropout=("weighted_dropout", "sum"), sum_serious=("weighted_serious", "sum"), n_trials=("nct_id", "nunique"))


def _lookup_cohorts(block: pd.DataFrame, query_mapping: pd.DataFrame, cohort: pd.DataFrame, prefix: str, prior: float, strength: float) -> pd.DataFrame:
    query = query_mapping[query_mapping["_global_id"].isin(block["_global_id"])].copy()
    if query.empty:
        result = block[["_global_id"]].copy()
        for suffix in ["prior_mean", "prior_max", "effective_n", "uncertainty", "analysis_mean", "dropout_mean", "serious_mean", "backoff", "key_count"]:
            result[f"{prefix}_{suffix}"] = prior if suffix.startswith("prior") else (1.0 if suffix == "backoff" else 0.0)
        return result
    query = query.merge(cohort, on="key", how="left")
    query["posterior"] = (query["sum_y"].fillna(0) + strength * prior) / (query["sum_w"].fillna(0) + strength)
    query["effective"] = query["sum_w"].fillna(0) ** 2 / query["sum_w2"].replace(0, np.nan)
    query["uncertainty_value"] = np.sqrt((query["posterior"] * (1 - query["posterior"])) / (query["effective"].fillna(0) + strength + 1))
    query["analysis_value"] = query["sum_analysis"] / query["sum_w"].replace(0, np.nan)
    query["dropout_value"] = query["sum_dropout"] / query["sum_w"].replace(0, np.nan)
    query["serious_value"] = query["sum_serious"] / query["sum_w"].replace(0, np.nan)
    query["missing"] = query["sum_w"].isna().astype(float)
    grouped = query.groupby("_global_id")
    result = grouped.agg(prior_mean=("posterior", "mean"), prior_max=("posterior", "max"), effective_n=("effective", "sum"), uncertainty=("uncertainty_value", "mean"), analysis_mean=("analysis_value", "mean"), dropout_mean=("dropout_value", "mean"), serious_mean=("serious_value", "mean"), backoff=("missing", "mean"), key_count=("key", "nunique")).reset_index()
    result = result.rename(columns={column: f"{prefix}_{column}" for column in result.columns if column != "_global_id"})
    return block[["_global_id"]].merge(result, on="_global_id", how="left")


def _add_historical_priors(features: pd.DataFrame, rows: pd.DataFrame, root: Path, query_groups: dict[str, pd.DataFrame], history_groups: dict[str, pd.DataFrame], valid: pd.DataFrame, withdrawals: pd.DataFrame, serious: pd.DataFrame, studies: pd.DataFrame) -> pd.DataFrame:
    enrollment = studies.set_index("nct_id")["enrollment"]
    outputs = []
    for timestamp, block in rows.groupby("timestamp", sort=True):
        evidence = _snapshot_evidence(timestamp, valid, withdrawals, serious, enrollment)
        evidence = evidence[~evidence["nct_id"].isin(block["nct_id"])]
        if evidence.empty:
            prior = 0.5
            global_analysis = 0.0
            global_dropout = 0.0
            global_serious = 0.0
        else:
            prior = float(np.average(evidence["success"], weights=evidence["weight"]))
            global_analysis = float(np.average(evidence["analysis_count"], weights=evidence["weight"]))
            global_dropout = float(np.average(evidence["dropout_burden"], weights=evidence["weight"]))
            global_serious = float(np.average(evidence["serious_burden"], weights=evidence["weight"]))
        current = block[["_global_id"]].copy()
        current["history_global_prior"] = prior
        current["history_global_trials"] = float(len(evidence))
        current["history_global_analysis_mean"] = global_analysis
        current["history_global_dropout_mean"] = global_dropout
        current["history_global_serious_mean"] = global_serious
        for prefix in ["sponsor", "condition", "intervention", "facility", "phase", "country", "sponsor_phase", "phase_condition"]:
            cohort = _cohort_table(evidence, history_groups[prefix], timestamp)
            looked = _lookup_cohorts(block, query_groups[prefix], cohort, prefix, prior, 20.0)
            current = current.merge(looked, on="_global_id", how="left")
            if prefix == "sponsor":
                sensitivity = _lookup_cohorts(block, query_groups[prefix], cohort, "sponsor_s40", prior, 40.0)
                current = current.merge(sensitivity, on="_global_id", how="left")
        outputs.append(current)
    priors = pd.concat(outputs, ignore_index=True)
    return features.merge(priors, on="_global_id", how="left")


def _within_timestamp_features(features: pd.DataFrame) -> pd.DataFrame:
    columns = ["study_enrollment", "study_trial_age_days", "site_count", "elig_criteria_words", "elig_restriction_density", "sponsor_prior_mean", "condition_prior_mean", "intervention_prior_mean", "country_prior_mean"]
    for column in columns:
        if column not in features:
            continue
        grouped = features.groupby("timestamp")[column]
        features[f"rank_{column}"] = grouped.rank(pct=True, method="average")
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        features[f"z_{column}"] = (features[column] - mean) / std
        features[f"gap_{column}"] = grouped.transform("max") - features[column]
    return features


def _documents(studies: pd.DataFrame, relevant_ids: set[int], titles: dict[int, str], summaries: dict[int, str], phases: dict[int, str], design_text: dict[int, str], condition_terms: dict[int, list[str]], intervention_terms: dict[int, list[str]]) -> dict[int, str]:
    documents = {}
    for nct_id in relevant_ids:
        parts = [titles.get(nct_id, ""), summaries.get(nct_id, ""), phases.get(nct_id, ""), design_text.get(nct_id, ""), "Conditions: " + "; ".join(condition_terms.get(nct_id, [])), "Interventions: " + "; ".join(intervention_terms.get(nct_id, []))]
        documents[nct_id] = "\n".join(part for part in parts if part)
    return documents


def _interaction_mapping(left: pd.DataFrame, right: pd.DataFrame, query: bool) -> pd.DataFrame:
    join_columns = ["_global_id", "nct_id"] if query else ["nct_id"]
    a = left.rename(columns={"key": "left_key", "date": "left_date"})
    b = right.rename(columns={"key": "right_key", "date": "right_date"})
    merged = a.merge(b, on=join_columns, how="inner")
    merged["key"] = merged["left_key"].map(_safe_key) + "|" + merged["right_key"].map(_safe_key)
    if query:
        return merged[["_global_id", "nct_id", "key"]].drop_duplicates()
    merged["date"] = merged[["left_date", "right_date"]].max(axis=1)
    return merged[["nct_id", "key", "date"]].drop_duplicates()


def _build_uncached(root: Path) -> dict:
    started = time.time()
    rows = _task_rows(root)
    studies = _read(root, "studies")
    features, titles, summaries, phases = _make_study_features(rows, studies)
    designs = _read(root, "designs")
    features, design_text = _add_design_features(features, rows, designs)
    eligibilities = _read(root, "eligibilities")
    features, eligibility_chunks = _add_eligibility_features(features, rows, eligibilities)
    features, query_condition, history_condition, condition_terms = _relation_block(features, rows, root, "conditions_studies", "condition_id", "condition", "conditions", "condition_id", "mesh_term")
    features, query_intervention, history_intervention, intervention_terms = _relation_block(features, rows, root, "interventions_studies", "intervention_id", "intervention", "interventions", "intervention_id", "mesh_term")
    features, query_sponsor, history_sponsor = _sponsor_block(features, rows, root)
    features, query_facility, history_facility, query_country, history_country = _facility_block(features, rows, root)
    query_phase = rows[["_global_id", "nct_id"]].copy()
    query_phase["key"] = query_phase["nct_id"].map(phases).fillna("__MISSING__")
    history_phase = studies[["nct_id", "start_date", "phase"]].rename(columns={"start_date": "date", "phase": "key"})
    history_phase["key"] = history_phase["key"].map(_safe_key)
    history_phase["date"] = pd.to_datetime(history_phase["date"])
    query_groups = {"sponsor": query_sponsor, "condition": query_condition, "intervention": query_intervention, "facility": query_facility, "phase": query_phase, "country": query_country}
    history_groups = {"sponsor": history_sponsor, "condition": history_condition, "intervention": history_intervention, "facility": history_facility, "phase": history_phase, "country": history_country}
    query_groups["sponsor_phase"] = _interaction_mapping(query_sponsor, query_phase, True)
    history_groups["sponsor_phase"] = _interaction_mapping(history_sponsor, history_phase, False)
    query_groups["phase_condition"] = _interaction_mapping(query_phase, query_condition, True)
    history_groups["phase_condition"] = _interaction_mapping(history_phase, history_condition, False)
    valid, withdrawals, serious = _historical_evidence(root)
    features = _add_historical_priors(features, rows, root, query_groups, history_groups, valid, withdrawals, serious, studies)
    features = _within_timestamp_features(features)
    relevant_ids = set(rows["nct_id"].astype(int)) | set(valid["nct_id"].astype(int))
    all_studies = studies[studies["nct_id"].isin(relevant_ids)].copy()
    all_titles = dict(zip(all_studies["nct_id"].astype(int), all_studies["brief_title"].map(_text)))
    all_summaries = dict(zip(all_studies["nct_id"].astype(int), all_studies["brief_summaries"].map(_text)))
    documents = _documents(studies, relevant_ids, all_titles, all_summaries, phases, design_text, condition_terms, intervention_terms)
    condition_sets = history_condition[history_condition["nct_id"].isin(relevant_ids)].groupby("nct_id")["key"].apply(lambda x: set(x.dropna().tolist())).to_dict()
    result = {"version": FEATURE_VERSION, "rows": rows, "features": features, "protocol_documents": documents, "eligibility_chunks": {nct_id: eligibility_chunks.get(nct_id, [""]) for nct_id in relevant_ids}, "phases": phases, "condition_sets": condition_sets, "valid_analyses": valid, "build_seconds": time.time() - started}
    return result


def _register_artifact(cache_dir: Path, name: str, path: Path, description: str, content_key: str) -> None:
    import fcntl
    lock_path = cache_dir / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        registry = cache_dir / "artifacts.json"
        try:
            records = json.loads(registry.read_text()) if registry.exists() else []
        except Exception:
            records = []
        relative = str(path.relative_to(cache_dir))
        if not any(record.get("path") == relative for record in records):
            records.append({"name": name, "path": relative, "description": description, "content_key": content_key, "rebuild_hint": "Run main.py with the same sanitized rel-trial cache."})
            temporary = registry.with_suffix(".tmp")
            temporary.write_text(json.dumps(records, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def load_feature_bank(cache_dir: Path) -> dict:
    root = Path(os.environ["RELBENCH_CACHE_DIR"]) / "rel-trial"
    target = cache_dir / f"{FEATURE_VERSION}.pkl"
    if target.exists():
        with target.open("rb") as handle:
            value = pickle.load(handle)
        if value.get("version") == FEATURE_VERSION:
            print(f"[features] loaded cached bank {target.name} in {value.get('build_seconds', 0):.1f}s original build")
            return value
    value = _build_uncached(root)
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, target)
    _register_artifact(cache_dir, "lane0 censored all-table feature bank", target, "Registration features, temporal empirical-Bayes priors, documents, and evidence records for rel-trial study-outcome.", FEATURE_VERSION)
    print(f"[features] built all-table bank in {value['build_seconds']:.1f}s")
    return value


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
