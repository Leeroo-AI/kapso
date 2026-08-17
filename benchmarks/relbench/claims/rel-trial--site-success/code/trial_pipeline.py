# Imports

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score


# Constants

VERSION = "target_exact_quarterly_v3"
HALF_LIFE_YEARS = 4.0
REPORTING_ROUNDS = 520
SUCCESS_ROUNDS = 620
FACILITY_ROUNDS = 500
CATEGORICAL_COLUMNS = [
    "study_type",
    "phase",
    "enrollment_type",
    "source_class",
    "has_dmc",
    "is_fda_regulated_drug",
    "is_fda_regulated_device",
    "plan_to_share_ipd",
    "allocation",
    "intervention_model",
    "observational_model",
    "primary_purpose",
    "time_perspective",
    "masking",
    "gender",
    "sampling_method",
    "healthy_volunteers",
    "sponsor_class",
    "registry_overall_status",
    "registry_last_known_status",
    "registry_completion_date_type",
    "registry_primary_completion_date_type",
    "registry_why_stopped",
]


# Structures

@dataclass
class TrialBundle:
    pairs: pd.DataFrame
    features: pd.DataFrame
    documents: list[str]
    contexts: list[str]
    replay: pd.DataFrame
    edges: dict[int, pd.DataFrame]
    event_info: pd.DataFrame
    table_coverage: dict[str, float]


@dataclass
class HeadFit:
    reporting: Any
    success: Any
    reporting_calibrator: Any | None
    success_calibrator: Any | None
    text_vectorizer: Any | None
    text_model: Any | None
    feature_columns: list[str]
    categorical: list[str]


# Utilities

def _seconds(value: pd.Series) -> pd.Series:
    return value.dt.total_seconds() / 86400.0


def _text(value: pd.Series) -> pd.Series:
    return value.fillna("").astype(str)


def _age(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(year|month|week|day)", str(value).lower())
    if match is None:
        return np.nan
    scales = {"year": 1.0, "month": 1.0 / 12.0, "week": 1.0 / 52.1775, "day": 1.0 / 365.25}
    return float(match.group(1)) * scales[match.group(2)]


def _pooled_hash(values: pd.Series, modulus: int = 4093) -> pd.Series:
    return (pd.util.hash_pandas_object(_text(values), index=False).to_numpy(dtype=np.uint64) % modulus).astype(np.float32)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    threshold = 0.5 * ordered_weights.sum()
    return float(ordered_values[min(int(np.searchsorted(np.cumsum(ordered_weights), threshold, side="left")), len(order) - 1)])


def checksum_frame(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(VERSION.encode())
    digest.update(str(frame.shape).encode())
    for column in [value for value in ["timestamp", "facility_id", "nct_id"] if value in frame]:
        digest.update(pd.util.hash_pandas_object(frame[column], index=False).to_numpy().tobytes())
    return digest.hexdigest()[:20]


def add_snapshot_relative_features(bundle: TrialBundle) -> int:
    candidates = [
        column for column in bundle.features
        if column.startswith("registry_")
        and column not in CATEGORICAL_COLUMNS
        and not column.startswith("registry_expert_")
        and pd.api.types.is_numeric_dtype(bundle.features[column])
    ]
    additions: dict[str, np.ndarray] = {}
    origins = bundle.pairs["timestamp"]
    for column in candidates:
        values = pd.to_numeric(bundle.features[column], errors="coerce")
        if values.notna().sum() == 0 or values.nunique(dropna=True) < 2:
            continue
        grouped = values.groupby(origins)
        additions[f"origin_rank_{column}"] = grouped.rank(pct=True).to_numpy(dtype=np.float32)
        median = grouped.transform("median")
        deviation = (values - median).abs().groupby(origins).transform("median")
        additions[f"origin_z_{column}"] = ((values - median) / (1.4826 * deviation + 1e-6)).clip(-20, 20).to_numpy(dtype=np.float32)
        maximum = grouped.transform("max")
        additions[f"origin_gap_{column}"] = (maximum - values).to_numpy(dtype=np.float32)
    if additions:
        bundle.features = pd.concat([bundle.features, pd.DataFrame(additions)], axis=1).copy()
    return len(additions)


# Exact labels

def exact_trial_events(db: Any) -> pd.DataFrame:
    outcomes = db.table_dict["outcomes"].df[["id", "nct_id", "outcome_type"]]
    analyses = db.table_dict["outcome_analyses"].df[
        ["id", "nct_id", "outcome_id", "p_value_modifier", "p_value", "date"]
    ]
    merged = analyses.merge(outcomes, left_on="outcome_id", right_on="id", how="inner", suffixes=("", "_outcome"))
    admissible = merged[
        merged["outcome_type"].eq("Primary")
        & merged["p_value"].between(0.0, 1.0, inclusive="both")
        & (merged["p_value_modifier"].isna() | merged["p_value_modifier"].ne(">"))
    ].copy()
    admissible["significant"] = admissible["p_value"].lt(0.05).astype(np.int8)
    grouped = admissible.groupby(["nct_id", "date"], as_index=False).agg(
        success_all=("significant", "min"),
        analysis_count=("significant", "size"),
        significant_count=("significant", "sum"),
        p_value_min=("p_value", "min"),
        p_value_mean=("p_value", "mean"),
        p_value_max=("p_value", "max"),
    )
    return grouped.sort_values(["date", "nct_id"]).reset_index(drop=True)


# Replay

def replay_origins(debug: bool) -> list[pd.Timestamp]:
    if debug:
        return [pd.Timestamp("2016-01-01"), pd.Timestamp("2017-01-01")]
    return list(pd.date_range("2012-01-01", "2020-01-01", freq="3MS"))


def make_replay(db: Any, events: pd.DataFrame, debug: bool) -> pd.DataFrame:
    studies = db.table_dict["studies"].df[["nct_id", "start_date", "phase"]]
    facilities_studies = db.table_dict["facilities_studies"].df[["nct_id", "facility_id", "date"]]
    first_site = facilities_studies.groupby("nct_id")["date"].min()
    starts = studies.set_index("nct_id")["start_date"]
    phases = studies.set_index("nct_id")["phase"].fillna("__missing__")
    all_trials = pd.Index(studies["nct_id"].unique())
    rows: list[pd.DataFrame] = []
    for origin in replay_origins(debug):
        eligible = all_trials[
            (all_trials.map(starts) <= origin)
            & (all_trials.map(first_site) <= origin)
        ]
        past = set(events.loc[events["date"] <= origin, "nct_id"].unique())
        eligible = eligible[~eligible.isin(past)]
        future = events[(events["date"] > origin) & (events["date"] <= origin + pd.Timedelta(days=365))]
        future = future[future["nct_id"].isin(eligible)].drop_duplicates("nct_id")
        positive_ids = pd.Index(future["nct_id"])
        negatives = pd.DataFrame({"nct_id": eligible[~eligible.isin(positive_ids)]})
        negatives["phase"] = negatives["nct_id"].map(phases)
        negatives["age_bin"] = pd.cut(
            _seconds(pd.Series(origin - negatives["nct_id"].map(starts))),
            [-np.inf, 365, 1095, 2190, 3650, np.inf],
            labels=False,
        ).fillna(-1)
        target = min(len(negatives), max(1, len(positive_ids) * (3 if debug else 10)))
        if target < len(negatives):
            allocation = negatives.groupby(["phase", "age_bin"], dropna=False).size()
            allocation = np.maximum(1, np.rint(target * allocation / allocation.sum()).astype(int))
            pieces = []
            for key, count in allocation.items():
                mask = negatives["phase"].eq(key[0]) & negatives["age_bin"].eq(key[1])
                current = negatives[mask]
                pieces.append(current.sample(min(int(count), len(current)), random_state=1337 + origin.year))
            sampled = pd.concat(pieces, ignore_index=True).drop_duplicates("nct_id")
            if len(sampled) < target:
                remainder = negatives[~negatives["nct_id"].isin(sampled["nct_id"])]
                sampled = pd.concat([sampled, remainder.sample(min(target - len(sampled), len(remainder)), random_state=7331 + origin.year)])
        else:
            sampled = negatives
        negative_weight = len(negatives) / max(len(sampled), 1)
        positive = future[["nct_id", "success_all"]].copy()
        positive["report_label"] = 1
        positive["sampling_weight"] = 1.0
        negative = sampled[["nct_id"]].copy()
        negative["success_all"] = np.nan
        negative["report_label"] = 0
        negative["sampling_weight"] = negative_weight
        current = pd.concat([positive, negative], ignore_index=True)
        current["timestamp"] = origin
        rows.append(current)
    replay = pd.concat(rows, ignore_index=True)
    replay["replay_row"] = np.arange(len(replay), dtype=np.int64)
    return replay


# Rosters

def make_roster_edges(db: Any, seeds: pd.DataFrame) -> pd.DataFrame:
    facilities_studies = db.table_dict["facilities_studies"].df[["facility_id", "nct_id", "date"]]
    studies = db.table_dict["studies"].df[["nct_id", "start_date"]]
    result: list[pd.DataFrame] = []
    seeds = seeds.reset_index(drop=True).copy()
    seeds["seed_row"] = np.arange(len(seeds), dtype=np.int64)
    starts = studies.set_index("nct_id")["start_date"]
    for origin, current in seeds.groupby("timestamp", sort=False):
        subset = facilities_studies[
            facilities_studies["facility_id"].isin(current["facility_id"])
            & facilities_studies["date"].le(origin)
        ].drop_duplicates(["facility_id", "nct_id"])
        subset = subset[subset["nct_id"].map(starts).le(origin)]
        subset = current[["seed_row", "facility_id", "timestamp"]].merge(
            subset[["facility_id", "nct_id", "date"]], on="facility_id", how="inner"
        )
        result.append(subset)
    if not result:
        return pd.DataFrame(columns=["seed_row", "facility_id", "timestamp", "nct_id", "date"])
    return pd.concat(result, ignore_index=True)


def candidate_pairs_from_edges(edges: dict[int, pd.DataFrame]) -> pd.DataFrame:
    pieces = []
    for year, edge in edges.items():
        if len(edge):
            current = edge[["timestamp", "nct_id"]].drop_duplicates().copy()
            current["candidate_year"] = year
            pieces.append(current)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["timestamp", "nct_id", "candidate_year"])


# Trial features

def _latest_rows(pairs: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    selected = table[table["nct_id"].isin(pairs["nct_id"].unique())]
    merged = pairs[["row_id", "nct_id", "timestamp"]].merge(selected, on="nct_id", how="left")
    merged = merged[merged["date"].notna() & merged["date"].le(merged["timestamp"])]
    if not len(merged):
        return merged.set_index("row_id")
    return merged.sort_values(["row_id", "date"]).drop_duplicates("row_id", keep="last").set_index("row_id")


def _dated_rows(pairs: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    selected = table[table["nct_id"].isin(pairs["nct_id"].unique())]
    merged = pairs[["row_id", "nct_id", "timestamp"]].merge(selected, on="nct_id", how="left")
    return merged[merged["date"].notna() & merged["date"].le(merged["timestamp"])]


def _add_numeric_summary(features: dict[str, Any], pairs: pd.DataFrame, rows: pd.DataFrame, prefix: str, columns: list[str]) -> None:
    grouped = rows.groupby("row_id") if len(rows) else None
    counts = grouped.size() if grouped is not None else pd.Series(dtype=float)
    features[f"{prefix}_count"] = pairs["row_id"].map(counts).fillna(0).to_numpy(dtype=np.float32)
    if len(rows):
        recency = (rows["timestamp"] - rows["date"]).dt.days.groupby(rows["row_id"]).min()
        features[f"{prefix}_recency"] = pairs["row_id"].map(recency).to_numpy(dtype=np.float32)
        recent = rows[rows["date"] > rows["timestamp"] - pd.Timedelta(days=365)].groupby("row_id").size()
        features[f"{prefix}_recent_count"] = pairs["row_id"].map(recent).fillna(0).to_numpy(dtype=np.float32)
        for column in columns:
            values = pd.to_numeric(rows[column], errors="coerce")
            summary = values.groupby(rows["row_id"]).agg(["sum", "mean", "max"])
            for statistic in summary:
                features[f"{prefix}_{column}_{statistic}"] = pairs["row_id"].map(summary[statistic]).to_numpy(dtype=np.float32)
    else:
        features[f"{prefix}_recency"] = np.nan
        features[f"{prefix}_recent_count"] = 0.0


def _event_associations(events: pd.DataFrame, relation: pd.DataFrame, node: str) -> pd.DataFrame:
    selected = relation[["nct_id", node, "date"]].merge(
        events[["nct_id", "date", "success_all"]], on="nct_id", suffixes=("_relation", "_event")
    )
    return selected[selected["date_relation"] <= selected["date_event"]]


def _relation_block(
    features: dict[str, Any],
    documents: list[str],
    contexts: list[str],
    pairs: pd.DataFrame,
    relation: pd.DataFrame,
    dimension: pd.DataFrame,
    node: str,
    prefix: str,
    display_column: str,
    events: pd.DataFrame,
) -> float:
    selected_relation = relation[relation["nct_id"].isin(pairs["nct_id"].unique())]
    matched = pairs[["row_id", "nct_id", "timestamp"]].merge(selected_relation, on="nct_id", how="left")
    matched = matched[matched["date"].notna() & matched["date"].le(matched["timestamp"])]
    counts = matched.groupby("row_id")[node].nunique() if len(matched) else pd.Series(dtype=float)
    features[f"{prefix}_count"] = pairs["row_id"].map(counts).fillna(0).to_numpy(dtype=np.float32)
    if len(matched):
        recency = (matched["timestamp"] - matched["date"]).dt.days.groupby(matched["row_id"]).min()
        recent = matched[matched["date"] > matched["timestamp"] - pd.Timedelta(days=365)].groupby("row_id")[node].nunique()
        features[f"{prefix}_recency"] = pairs["row_id"].map(recency).to_numpy(dtype=np.float32)
        features[f"{prefix}_recent_count"] = pairs["row_id"].map(recent).fillna(0).to_numpy(dtype=np.float32)
        first = relation.groupby(node)["date"].min()
        novel = (matched["timestamp"] - matched[node].map(first)).dt.days.le(365).groupby(matched["row_id"]).mean()
        features[f"{prefix}_novel_share"] = pairs["row_id"].map(novel).fillna(0).to_numpy(dtype=np.float32)
        degree_parts = []
        for origin, current in matched.groupby("timestamp", sort=False):
            nodes = current[node].dropna().unique()
            historical = relation[relation["date"].le(origin) & relation[node].isin(nodes)]
            degree = historical.groupby(node)["nct_id"].nunique()
            part = current[["row_id", node]].copy()
            part["degree"] = part[node].map(degree).fillna(0)
            degree_parts.append(part)
        degree_rows = pd.concat(degree_parts, ignore_index=True)
        degree_summary = degree_rows.groupby("row_id")["degree"].agg(["mean", "max", "min", "sum"])
        for statistic in ["mean", "max", "min"]:
            features[f"{prefix}_degree_{statistic}"] = pairs["row_id"].map(degree_summary[statistic]).fillna(0).to_numpy(dtype=np.float32)
        concentration = degree_summary["max"] / degree_summary["sum"].clip(lower=1)
        features[f"{prefix}_degree_concentration"] = pairs["row_id"].map(concentration).fillna(0).to_numpy(dtype=np.float32)
        associations = _event_associations(events, relation, node)
        eb_parts = []
        for origin, current in matched.groupby("timestamp", sort=False):
            history = associations[associations["date_event"] <= origin]
            summary = history.groupby(node).agg(
                successes=("success_all", "sum"),
                reports=("success_all", "size"),
                last_report=("date_event", "max"),
            )
            global_rate = float(history["success_all"].mean()) if len(history) else 0.42
            summary["rate"] = (summary["successes"] + 20.0 * global_rate) / (summary["reports"] + 20.0)
            current_eb = current[["row_id", node, "timestamp"]].copy()
            current_eb["rate"] = current_eb[node].map(summary["rate"])
            current_eb["reports"] = current_eb[node].map(summary["reports"])
            current_eb["recency"] = (origin - current_eb[node].map(summary["last_report"])).dt.days
            eb_parts.append(current_eb)
        eb = pd.concat(eb_parts, ignore_index=True).groupby("row_id").agg(
            rate_mean=("rate", "mean"),
            rate_max=("rate", "max"),
            reports_mean=("reports", "mean"),
            report_recency_min=("recency", "min"),
        )
        for column in eb:
            features[f"{prefix}_neighbor_{column}"] = pairs["row_id"].map(eb[column]).to_numpy(dtype=np.float32)
        displayed = matched.merge(dimension, on=node, how="left")
        if display_column in displayed:
            grouped_display = displayed.groupby("row_id")[display_column].agg(
                lambda values: ", ".join(sorted({str(value) for value in values.dropna()})[:16])
            )
            for row_id, value in grouped_display.items():
                contexts[int(row_id)] += f"\n{prefix}: {value}"
        if prefix == "sponsor" and "agency_class" in displayed:
            classes = displayed.groupby("row_id")["agency_class"].agg(lambda values: next(iter(values.dropna().astype(str)), "__missing__"))
            features["sponsor_class"] = pairs["row_id"].map(classes).fillna("__missing__").to_numpy()
    else:
        for name in ["recency", "recent_count", "novel_share", "degree_mean", "degree_max", "degree_min", "degree_concentration", "neighbor_rate_mean", "neighbor_rate_max", "neighbor_reports_mean", "neighbor_report_recency_min"]:
            features[f"{prefix}_{name}"] = np.nan
    return float((pairs["row_id"].map(counts).fillna(0) > 0).mean())


def build_trial_features(db: Any, pairs: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], dict[str, float]]:
    started = time.time()
    pairs = pairs.reset_index(drop=True).copy()
    pairs["row_id"] = np.arange(len(pairs), dtype=np.int64)
    studies = db.table_dict["studies"].df
    merged = pairs.merge(studies, on="nct_id", how="left", validate="many_to_one")
    features: dict[str, Any] = {}
    numeric_study = ["enrollment", "number_of_arms", "number_of_groups"]
    for column in numeric_study:
        features[column] = pd.to_numeric(merged[column], errors="coerce").to_numpy(dtype=np.float32)
    features["enrollment_log"] = np.log1p(np.maximum(features["enrollment"], 0)).astype(np.float32)
    features["trial_age_days"] = _seconds(merged["timestamp"] - merged["start_date"]).to_numpy(dtype=np.float32)
    features["origin_year"] = merged["timestamp"].dt.year.to_numpy(dtype=np.float32)
    features["origin_month"] = merged["timestamp"].dt.month.to_numpy(dtype=np.float32)
    features["start_year"] = merged["start_date"].dt.year.to_numpy(dtype=np.float32)
    features["start_month"] = merged["start_date"].dt.month.to_numpy(dtype=np.float32)
    for column in CATEGORICAL_COLUMNS[:8]:
        features[column] = _text(merged[column]).replace("", "__missing__").to_numpy()
    documents = []
    contexts = []
    for _, row in merged.iterrows():
        document = "\n".join([
            f"Brief title: {row.get('brief_title', '')}",
            f"Official title: {row.get('official_title', '')}",
            f"Brief summary: {row.get('brief_summaries', '')}",
            f"Detailed description: {row.get('detailed_descriptions', '')}",
        ])
        context = "\n".join([
            f"origin: {row['timestamp'].date()}",
            f"phase: {row.get('phase', '')}",
            f"study type: {row.get('study_type', '')}",
            f"source class: {row.get('source_class', '')}",
            f"planned enrollment: {row.get('enrollment', '')}",
            f"arms: {row.get('number_of_arms', '')}",
            f"groups: {row.get('number_of_groups', '')}",
        ])
        documents.append(document)
        contexts.append(context)
    for column in ["brief_title", "official_title", "brief_summaries", "detailed_descriptions"]:
        values = _text(merged[column])
        features[f"{column}_length"] = values.str.len().to_numpy(dtype=np.float32)
        features[f"{column}_words"] = values.str.count(r"\b\w+\b").to_numpy(dtype=np.float32)
        features[f"{column}_numbers"] = values.str.count(r"\b\d+(?:\.\d+)?\b").to_numpy(dtype=np.float32)
    latest_design = _latest_rows(pairs, db.table_dict["designs"].df)
    for column in CATEGORICAL_COLUMNS[8:14]:
        features[column] = pairs["row_id"].map(latest_design[column] if column in latest_design else pd.Series(dtype=object)).fillna("__missing__").astype(str).to_numpy()
    for column in ["masking_description", "intervention_model_description"]:
        values = _text(pairs["row_id"].map(latest_design[column] if column in latest_design else pd.Series(dtype=object)))
        features[f"{column}_length"] = values.str.len().to_numpy(dtype=np.float32)
    latest_eligibility = _latest_rows(pairs, db.table_dict["eligibilities"].df)
    for column in CATEGORICAL_COLUMNS[14:17]:
        features[column] = pairs["row_id"].map(latest_eligibility[column] if column in latest_eligibility else pd.Series(dtype=object)).fillna("__missing__").astype(str).to_numpy()
    minimum = pairs["row_id"].map(latest_eligibility["minimum_age"] if "minimum_age" in latest_eligibility else pd.Series(dtype=object))
    maximum = pairs["row_id"].map(latest_eligibility["maximum_age"] if "maximum_age" in latest_eligibility else pd.Series(dtype=object))
    features["minimum_age_years"] = minimum.map(_age).to_numpy(dtype=np.float32)
    features["maximum_age_years"] = maximum.map(_age).to_numpy(dtype=np.float32)
    features["age_range_years"] = features["maximum_age_years"] - features["minimum_age_years"]
    criteria = _text(pairs["row_id"].map(latest_eligibility["criteria"] if "criteria" in latest_eligibility else pd.Series(dtype=object)))
    population = _text(pairs["row_id"].map(latest_eligibility["population"] if "population" in latest_eligibility else pd.Series(dtype=object)))
    features["criteria_length"] = criteria.str.len().to_numpy(dtype=np.float32)
    features["criteria_words"] = criteria.str.count(r"\b\w+\b").to_numpy(dtype=np.float32)
    features["criteria_numeric_thresholds"] = criteria.str.count(r"(?:<=|>=|<|>|≤|≥)\s*\d|\b\d+(?:\.\d+)?\s*(?:mg|kg|years?|months?|days?|%)\b", flags=re.I).to_numpy(dtype=np.float32)
    features["criteria_exclusion_markers"] = criteria.str.count(r"\bexclusion\b|\bexcluded\b|\bnot eligible\b", flags=re.I).to_numpy(dtype=np.float32)
    for index in range(len(documents)):
        documents[index] += f"\nEligibility population: {population.iat[index]}\nEligibility criteria: {criteria.iat[index]}"
        contexts[index] += f"\ngender: {features['gender'][index]}\nminimum age years: {features['minimum_age_years'][index]}\nmaximum age years: {features['maximum_age_years'][index]}"
    coverage: dict[str, float] = {}
    outcomes_rows = _dated_rows(pairs, db.table_dict["outcomes"].df)
    coverage["outcomes"] = float(outcomes_rows["row_id"].nunique() / max(len(pairs), 1))
    _add_numeric_summary(features, pairs, outcomes_rows, "outcomes", [])
    if len(outcomes_rows):
        outcome_groups = outcomes_rows.groupby("row_id")
        primary = outcomes_rows["outcome_type"].eq("Primary").groupby(outcomes_rows["row_id"]).sum()
        features["primary_outcome_count"] = pairs["row_id"].map(primary).fillna(0).to_numpy(dtype=np.float32)
        for column in ["title", "description", "time_frame", "population"]:
            lengths = _text(outcomes_rows[column]).str.len().groupby(outcomes_rows["row_id"]).agg(["mean", "max"])
            for statistic in lengths:
                features[f"outcome_{column}_{statistic}_length"] = pairs["row_id"].map(lengths[statistic]).fillna(0).to_numpy(dtype=np.float32)
        rendered = outcome_groups.apply(lambda frame: "\n".join(
            f"Visible endpoint: type={row.outcome_type}; title={row.title}; description={row.description}; time frame={row.time_frame}; population={row.population}"
            for row in frame.itertuples()
        ), include_groups=False)
        for row_id, value in rendered.items():
            documents[int(row_id)] += "\n" + str(value)
    for table_name, numeric in [
        ("drop_withdrawals", ["count"]),
        ("reported_event_totals", ["subjects_affected", "subjects_at_risk"]),
    ]:
        rows = _dated_rows(pairs, db.table_dict[table_name].df)
        coverage[table_name] = float(rows["row_id"].nunique() / max(len(pairs), 1))
        _add_numeric_summary(features, pairs, rows, table_name, numeric)
    event_rows = pairs[["row_id", "nct_id", "timestamp"]].merge(events, on="nct_id", how="left")
    event_rows = event_rows[event_rows["date"].notna() & event_rows["date"].le(event_rows["timestamp"])]
    coverage["outcome_analyses"] = float(event_rows["row_id"].nunique() / max(len(pairs), 1))
    _add_numeric_summary(features, pairs, event_rows, "primary_analysis", ["success_all", "analysis_count", "p_value_min", "p_value_mean", "p_value_max"])
    relation_specs = [
        ("conditions_studies", "conditions", "condition_id", "condition", "mesh_term"),
        ("interventions_studies", "interventions", "intervention_id", "intervention", "mesh_term"),
        ("sponsors_studies", "sponsors", "sponsor_id", "sponsor", "name"),
        ("facilities_studies", "facilities", "facility_id", "site", "country"),
    ]
    for relation_name, dimension_name, node, prefix, display in relation_specs:
        coverage[relation_name] = _relation_block(
            features,
            documents,
            contexts,
            pairs,
            db.table_dict[relation_name].df,
            db.table_dict[dimension_name].df,
            node,
            prefix,
            display,
            events,
        )
    feature_frame = pd.DataFrame(features)
    for column in CATEGORICAL_COLUMNS:
        if column not in feature_frame:
            feature_frame[column] = "__missing__"
        feature_frame[column] = feature_frame[column].fillna("__missing__").astype("category")
    rank_columns = [
        "enrollment_log", "trial_age_days", "number_of_arms", "number_of_groups",
        "criteria_length", "condition_count", "intervention_count", "sponsor_count",
        "site_count", "site_degree_mean", "outcomes_count", "primary_analysis_count",
    ]
    additions: dict[str, Any] = {}
    for column in rank_columns:
        if column not in feature_frame:
            continue
        values = pd.to_numeric(feature_frame[column], errors="coerce")
        additions[f"origin_rank_{column}"] = values.groupby(pairs["timestamp"]).rank(pct=True).to_numpy(dtype=np.float32)
        median = values.groupby(pairs["timestamp"]).transform("median")
        deviation = (values - median).abs().groupby(pairs["timestamp"]).transform("median")
        additions[f"origin_z_{column}"] = ((values - median) / (1.4826 * deviation + 1e-6)).clip(-20, 20).to_numpy(dtype=np.float32)
        maximum = values.groupby(pairs["timestamp"]).transform("max")
        additions[f"origin_gap_{column}"] = (maximum - values).to_numpy(dtype=np.float32)
    additions["row_missing_count"] = feature_frame.isna().sum(axis=1).to_numpy(dtype=np.float32)
    feature_frame = pd.concat([feature_frame, pd.DataFrame(additions)], axis=1).copy()
    coverage["feature_rows_per_minute"] = float(len(pairs) / max(time.time() - started, 1e-6) * 60.0)
    return feature_frame, documents, contexts, coverage


# Bundle cache

def _bundle_path(cache: Path, context: Any, debug: bool) -> Path:
    train = context.train.df
    validation = context.val.df
    test = context.test.df
    database_maximum = max(
        table.df[table.time_col].max()
        for table in context.db.table_dict.values()
        if table.time_col is not None and len(table.df)
    )
    payload = f"{VERSION}|{debug}|{len(train)}|{len(validation)}|{len(test)}|{database_maximum}"
    key = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return cache / "lane0_target_exact_stage1" / key


def _save_bundle(path: Path, bundle: TrialBundle) -> None:
    path.mkdir(parents=True, exist_ok=True)
    bundle.pairs.to_parquet(path / "pairs.parquet", index=False)
    bundle.features.to_parquet(path / "features.parquet", index=False)
    bundle.replay.to_parquet(path / "replay.parquet", index=False)
    bundle.event_info.to_parquet(path / "events.parquet", index=False)
    for year, edge in bundle.edges.items():
        edge.to_parquet(path / f"edges_{year}.parquet", index=False)
    with (path / "documents.jsonl").open("w", encoding="utf-8") as stream:
        for document, context in zip(bundle.documents, bundle.contexts):
            stream.write(json.dumps({"document": document, "context": context}, ensure_ascii=False) + "\n")
    (path / "metadata.json").write_text(json.dumps({"coverage": bundle.table_coverage, "version": VERSION}, indent=2) + "\n")
    (path / "READY").write_text("ready\n")


def _load_bundle(path: Path) -> TrialBundle:
    pairs = pd.read_parquet(path / "pairs.parquet")
    features = pd.read_parquet(path / "features.parquet")
    replay = pd.read_parquet(path / "replay.parquet")
    events = pd.read_parquet(path / "events.parquet")
    edges = {int(value.stem.split("_")[1]): pd.read_parquet(value) for value in path.glob("edges_*.parquet")}
    documents: list[str] = []
    contexts: list[str] = []
    with (path / "documents.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            documents.append(record["document"])
            contexts.append(record["context"])
    metadata = json.loads((path / "metadata.json").read_text())
    return TrialBundle(pairs, features, documents, contexts, replay, edges, events, metadata["coverage"])


def build_bundle(context: Any, cache: Path, debug: bool) -> tuple[TrialBundle, str]:
    path = _bundle_path(cache, context, debug)
    if (path / "READY").exists():
        return _load_bundle(path), "hit"
    events = exact_trial_events(context.db)
    replay = make_replay(context.db, events, debug)
    frames: dict[int, pd.DataFrame] = {}
    train = context.train.df.reset_index(drop=True)
    for year in [2018, 2019]:
        current = train[train["timestamp"].dt.year.eq(year)][["timestamp", "facility_id"]].reset_index(drop=True)
        frames[year] = current
    frames[2020] = context.val.df[["timestamp", "facility_id"]].reset_index(drop=True)
    frames[2021] = context.test.df[["timestamp", "facility_id"]].reset_index(drop=True)
    if debug:
        for year in frames:
            frames[year] = frames[year].head(600).copy()
    edges = {year: make_roster_edges(context.db, frame) for year, frame in frames.items()}
    candidate_pairs = candidate_pairs_from_edges(edges)[["timestamp", "nct_id"]]
    if debug:
        candidate_pairs = candidate_pairs.groupby("timestamp", group_keys=False).head(800)
        for year, edge in edges.items():
            allowed = set(candidate_pairs.loc[candidate_pairs["timestamp"].dt.year.eq(year), "nct_id"])
            edges[year] = edge[edge["nct_id"].isin(allowed)].copy()
    pairs = pd.concat([
        replay[["timestamp", "nct_id"]],
        candidate_pairs,
    ], ignore_index=True).drop_duplicates(["timestamp", "nct_id"]).sort_values(["timestamp", "nct_id"]).reset_index(drop=True)
    pairs["pair_row"] = np.arange(len(pairs), dtype=np.int64)
    replay = replay.merge(pairs[["timestamp", "nct_id", "pair_row"]], on=["timestamp", "nct_id"], how="left", validate="many_to_one")
    features, documents, contexts, coverage = build_trial_features(context.db, pairs, events)
    bundle = TrialBundle(pairs, features, documents, contexts, replay, edges, events, coverage)
    _save_bundle(path, bundle)
    return bundle, "built"


# Head models

def _lgb_parameters(head: str, debug: bool) -> dict[str, Any]:
    reporting = head == "reporting"
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 31 if reporting else 63,
        "max_depth": 7,
        "min_data_in_leaf": 100 if reporting else 40,
        "lambda_l2": 10.0,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 1337 if reporting else 7331,
        "num_threads": 22,
        "force_col_wise": True,
    }


def _fit_binary(matrix: pd.DataFrame, labels: np.ndarray, weights: np.ndarray, categorical: list[str], head: str, debug: bool) -> Any:
    dataset = lgb.Dataset(matrix, label=labels, weight=weights, categorical_feature=categorical, free_raw_data=False)
    rounds = 60 if debug else (REPORTING_ROUNDS if head == "reporting" else SUCCESS_ROUNDS)
    return lgb.train(_lgb_parameters(head, debug), dataset, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])


def _origin_weights(replay: pd.DataFrame, features: pd.DataFrame, cutoff: pd.Timestamp, scheme: str) -> np.ndarray:
    weights = replay["sampling_weight"].to_numpy(dtype=np.float64)
    site_count = features.iloc[replay["pair_row"].to_numpy()]["site_count"].to_numpy(dtype=np.float64)
    if scheme == "sqrt_site":
        weights *= np.sqrt(np.maximum(site_count, 1.0))
    elif scheme == "full_site":
        weights *= np.maximum(site_count, 1.0)
    age_years = (cutoff - replay["timestamp"]).dt.days.to_numpy(dtype=np.float64) / 365.25
    weights *= np.power(0.5, age_years / HALF_LIFE_YEARS)
    origin_totals = pd.Series(weights).groupby(replay["timestamp"].reset_index(drop=True)).transform("sum").to_numpy()
    return weights / np.maximum(origin_totals, 1e-12)


def _fit_text(documents: list[str], labels: np.ndarray, weights: np.ndarray, debug: bool) -> tuple[Any, Any]:
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        max_features=6000 if debug else 100000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(documents)
    model = LogisticRegression(
        C=0.1,
        solver="saga",
        penalty="elasticnet",
        l1_ratio=0.05,
        max_iter=60 if debug else 220,
        tol=2e-3 if debug else 8e-4,
        random_state=451,
        n_jobs=22,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(matrix, labels, sample_weight=weights)
    return vectorizer, model


def fit_heads(bundle: TrialBundle, cutoff: pd.Timestamp, scheme: str, debug: bool) -> HeadFit:
    replay = bundle.replay[bundle.replay["timestamp"] + pd.Timedelta(days=365) <= cutoff].copy()
    if debug and len(replay) > 5000:
        replay = replay.groupby(["timestamp", "report_label"], group_keys=False).head(2500)
    feature_columns = list(bundle.features.columns)
    categorical = [column for column in CATEGORICAL_COLUMNS if column in feature_columns]
    pair_rows = replay["pair_row"].to_numpy(dtype=np.int64)
    matrix = bundle.features.iloc[pair_rows]
    reporting_scheme = "equal" if scheme.startswith("success_") else scheme
    success_scheme = scheme.removeprefix("success_") if scheme.startswith("success_") else scheme
    reporting_weights = _origin_weights(replay, bundle.features, cutoff, reporting_scheme)
    success_all_weights = _origin_weights(replay, bundle.features, cutoff, success_scheme)
    reporting = _fit_binary(matrix, replay["report_label"].to_numpy(dtype=np.int8), reporting_weights, categorical, "reporting", debug)
    success_mask = replay["report_label"].eq(1).to_numpy()
    success_rows = replay.loc[success_mask]
    success_matrix = bundle.features.iloc[success_rows["pair_row"].to_numpy(dtype=np.int64)]
    success_weights = success_all_weights[success_mask]
    success_weights = success_weights / max(success_weights.mean(), 1e-12)
    success = _fit_binary(success_matrix, success_rows["success_all"].to_numpy(dtype=np.int8), success_weights, categorical, "success", debug)
    success_documents = [bundle.documents[index] for index in success_rows["pair_row"].to_numpy(dtype=np.int64)]
    text_vectorizer, text_model = _fit_text(success_documents, success_rows["success_all"].to_numpy(dtype=np.int8), success_weights, debug)
    return HeadFit(reporting, success, None, None, text_vectorizer, text_model, feature_columns, categorical)


def _apply_calibrator(values: np.ndarray, calibrator: Any | None) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    if calibrator is None:
        return clipped
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1].clip(1e-6, 1 - 1e-6)


def predict_heads(head: HeadFit, bundle: TrialBundle, pair_rows: np.ndarray, text_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = bundle.features.iloc[pair_rows]
    raw_q = head.reporting.predict(matrix)
    raw_tabular = head.success.predict(matrix)
    documents = [bundle.documents[index] for index in pair_rows]
    raw_text = head.text_model.predict_proba(head.text_vectorizer.transform(documents))[:, 1]
    raw_p = (1.0 - text_weight) * raw_tabular + text_weight * raw_text
    q = _apply_calibrator(raw_q, head.reporting_calibrator)
    p = _apply_calibrator(raw_p, head.success_calibrator)
    if getattr(bundle, "direct_poles", False) and "registry_expert_probability" in bundle.features:
        expert = pd.to_numeric(bundle.features.iloc[pair_rows]["registry_expert_probability"], errors="coerce").to_numpy(dtype=np.float64)
        covered = np.isfinite(expert)
        p[covered] = expert[covered]
    reported = bundle.features.iloc[pair_rows]["primary_analysis_count"].to_numpy(dtype=np.float64) > 0
    q[reported] = np.minimum(q[reported], 1e-5)
    return q, p, raw_tabular, raw_text


def target_exact_direct_gate(bundle: TrialBundle) -> tuple[bool, dict[str, Any]]:
    if "registry_expert_probability" not in bundle.features:
        return False, {"state": "unavailable"}
    folds: dict[str, Any] = {}
    passed = True
    for year in [2018, 2019]:
        origin = pd.Timestamp(f"{year}-01-01")
        pair_rows = _pair_rows_for_year(bundle, year)
        reporting, success = _candidate_truth(bundle, pair_rows, origin)
        expert = pd.to_numeric(bundle.features.iloc[pair_rows]["registry_expert_probability"], errors="coerce").to_numpy(dtype=np.float64)
        covered = reporting.astype(bool) & np.isfinite(success) & np.isfinite(expert)
        labels = success[reporting.astype(bool) & np.isfinite(success)]
        default = float(np.median(labels)) if len(labels) else 0.0
        baseline_mae = float(np.mean(np.abs(success[covered] - default))) if covered.any() else float("nan")
        expert_mae = float(np.mean(np.abs(success[covered] - expert[covered]))) if covered.any() else float("nan")
        current_passed = bool(covered.sum() >= 10 and expert_mae < baseline_mae)
        passed &= current_passed
        folds[str(year)] = {
            "covered_reporting_trials": int(covered.sum()),
            "coverage": float(covered.sum() / max(reporting.sum(), 1)),
            "expert_mae": expert_mae,
            "default_median_mae": baseline_mae,
            "passed": current_passed,
        }
    return passed, {"state": "measured", "folds": folds, "passed": passed}


def fit_platt(values: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None) -> Any:
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, max_iter=500)
    model.fit(logits, labels, sample_weight=weights)
    return model


# Facility fallback

def _facility_seed_frame(context: Any, debug: bool) -> pd.DataFrame:
    train = context.train.df[context.train.df["timestamp"].dt.year.ge(2012)].copy()
    train["split"] = "train"
    train["source_row"] = np.arange(len(train), dtype=np.int64)
    validation = context.val.df.copy()
    validation["split"] = "val"
    validation["source_row"] = np.arange(len(validation), dtype=np.int64)
    test = context.test.df.copy()
    test["success_rate"] = np.nan
    test["split"] = "test"
    test["source_row"] = np.arange(len(test), dtype=np.int64)
    frame = pd.concat([train, validation, test], ignore_index=True)
    if debug:
        selected = []
        for _, current in frame.groupby(["timestamp", "split"], sort=False):
            selected.append(current.head(600))
        frame = pd.concat(selected, ignore_index=True)
    frame["facility_row"] = np.arange(len(frame), dtype=np.int64)
    return frame


def build_facility_features(context: Any, events: pd.DataFrame, cache: Path, debug: bool) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    root = _bundle_path(cache, context, debug)
    seed_path = root / "facility_seeds.parquet"
    feature_path = root / "facility_features.parquet"
    if seed_path.exists() and feature_path.exists():
        return pd.read_parquet(seed_path), pd.read_parquet(feature_path), "hit"
    seeds = _facility_seed_frame(context, debug)
    facilities = context.db.table_dict["facilities"].df
    attributes = seeds.merge(facilities, on="facility_id", how="left", validate="many_to_one")
    values: dict[str, Any] = {}
    for column in ["country", "state", "city"]:
        values[column] = _text(attributes[column]).replace("", "__missing__").to_numpy()
        values[f"{column}_hash"] = _pooled_hash(attributes[column])
        frequency = facilities[column].fillna("__missing__").astype(str).value_counts()
        values[f"{column}_frequency"] = attributes[column].fillna("__missing__").astype(str).map(frequency).to_numpy(dtype=np.float32)
    fs = context.db.table_dict["facilities_studies"].df[["facility_id", "nct_id", "date"]]
    studies = context.db.table_dict["studies"].df[["nct_id", "start_date", "enrollment", "phase", "study_type"]]
    facility_events = fs.merge(events[["nct_id", "date", "success_all"]], on="nct_id", suffixes=("_relation", "_event"))
    facility_events = facility_events[facility_events["date_relation"] <= facility_events["date_event"]]
    relation_specs = [
        ("conditions_studies", "condition_id", "condition"),
        ("sponsors_studies", "sponsor_id", "sponsor"),
    ]
    node_events = {
        prefix: _event_associations(events, context.db.table_dict[table].df, node)
        for table, node, prefix in relation_specs
    }
    for origin, current in seeds.groupby("timestamp", sort=False):
        row_ids = current["facility_row"]
        facility_ids = current["facility_id"].unique()
        visible = fs[fs["date"].le(origin) & fs["facility_id"].isin(facility_ids)]
        visible_group = visible.groupby("facility_id")
        trial_count = visible_group["nct_id"].nunique()
        relation_recency = (origin - visible_group["date"].max()).dt.days
        recent = visible[visible["date"] > origin - pd.Timedelta(days=365)].groupby("facility_id")["nct_id"].nunique()
        values.setdefault("candidate_count", np.full(len(seeds), np.nan, dtype=np.float32))[row_ids] = current["facility_id"].map(trial_count).fillna(0)
        values.setdefault("facility_relation_recency", np.full(len(seeds), np.nan, dtype=np.float32))[row_ids] = current["facility_id"].map(relation_recency)
        values.setdefault("facility_recent_trial_count", np.full(len(seeds), np.nan, dtype=np.float32))[row_ids] = current["facility_id"].map(recent).fillna(0)
        visible_studies = visible.merge(studies, on="nct_id", how="left")
        grouped_studies = visible_studies.groupby("facility_id")
        enrollment = grouped_studies["enrollment"].agg(["mean", "max"])
        trial_age = (origin - visible_studies["start_date"]).dt.days.groupby(visible_studies["facility_id"]).agg(["mean", "min", "max"])
        phase_count = grouped_studies["phase"].nunique()
        for statistic in enrollment:
            values.setdefault(f"roster_enrollment_{statistic}", np.full(len(seeds), np.nan, dtype=np.float32))[row_ids] = current["facility_id"].map(enrollment[statistic])
        for statistic in trial_age:
            values.setdefault(f"roster_trial_age_{statistic}", np.full(len(seeds), np.nan, dtype=np.float32))[row_ids] = current["facility_id"].map(trial_age[statistic])
        values.setdefault("roster_phase_count", np.full(len(seeds), np.nan, dtype=np.float32))[row_ids] = current["facility_id"].map(phase_count).fillna(0)
        history = facility_events[facility_events["date_event"].le(origin) & facility_events["facility_id"].isin(facility_ids)]
        history_group = history.groupby("facility_id")
        report_count = history_group.size()
        successes = history_group["success_all"].sum()
        global_rate = float(history["success_all"].mean()) if len(history) else 0.42
        eb_rate = (successes + 20.0 * global_rate) / (report_count + 20.0)
        event_recency = (origin - history_group["date_event"].max()).dt.days
        last_year = history[history["date_event"] > origin - pd.Timedelta(days=365)].groupby("facility_id").size()
        last_three = history[history["date_event"] > origin - pd.Timedelta(days=1095)].groupby("facility_id").size()
        values.setdefault("history_report_count", np.full(len(seeds), np.nan, dtype=np.float32))[row_ids] = current["facility_id"].map(report_count).fillna(0)
        values.setdefault("history_success_rate_eb", np.full(len(seeds), np.nan, dtype=np.float32))[row_ids] = current["facility_id"].map(eb_rate).fillna(global_rate)
        values.setdefault("history_event_recency", np.full(len(seeds), np.nan, dtype=np.float32))[row_ids] = current["facility_id"].map(event_recency)
        values.setdefault("history_last_year_count", np.full(len(seeds), np.nan, dtype=np.float32))[row_ids] = current["facility_id"].map(last_year).fillna(0)
        values.setdefault("history_last_three_year_count", np.full(len(seeds), np.nan, dtype=np.float32))[row_ids] = current["facility_id"].map(last_three).fillna(0)
        for table, node, prefix in relation_specs:
            relation = context.db.table_dict[table].df
            current_nodes = visible[["facility_id", "nct_id"]].merge(
                relation[relation["date"].le(origin)][["nct_id", node]], on="nct_id", how="left"
            )
            history_nodes = node_events[prefix][node_events[prefix]["date_event"].le(origin)]
            node_summary = history_nodes.groupby(node).agg(successes=("success_all", "sum"), reports=("success_all", "size"))
            node_global = float(history_nodes["success_all"].mean()) if len(history_nodes) else global_rate
            node_summary["rate"] = (node_summary["successes"] + 30.0 * node_global) / (node_summary["reports"] + 30.0)
            current_nodes["rate"] = current_nodes[node].map(node_summary["rate"])
            current_nodes["reports"] = current_nodes[node].map(node_summary["reports"])
            neighbor = current_nodes.groupby("facility_id").agg(
                rate_mean=("rate", "mean"),
                rate_max=("rate", "max"),
                reports_mean=("reports", "mean"),
                node_count=(node, "nunique"),
            )
            for column in neighbor:
                values.setdefault(f"{prefix}_neighbor_{column}", np.full(len(seeds), np.nan, dtype=np.float32))[row_ids] = current["facility_id"].map(neighbor[column])
    features = pd.DataFrame(values)
    features["origin_year"] = seeds["timestamp"].dt.year.to_numpy(dtype=np.float32)
    for column in ["country", "state", "city"]:
        features[column] = features[column].fillna("__missing__").astype("category")
    for column in ["candidate_count", "history_report_count", "history_event_recency", "history_success_rate_eb", "condition_neighbor_rate_mean", "sponsor_neighbor_rate_mean"]:
        if column in features:
            numeric = pd.to_numeric(features[column], errors="coerce")
            features[f"origin_rank_{column}"] = numeric.groupby(seeds["timestamp"]).rank(pct=True).to_numpy(dtype=np.float32)
    features["missing_count"] = features.isna().sum(axis=1).to_numpy(dtype=np.float32)
    root.mkdir(parents=True, exist_ok=True)
    seeds.to_parquet(seed_path, index=False)
    features.to_parquet(feature_path, index=False)
    return seeds, features, "built"


def fit_facility_model(seeds: pd.DataFrame, features: pd.DataFrame, cutoff: pd.Timestamp, debug: bool) -> Any:
    mask = seeds["success_rate"].notna() & (seeds["timestamp"] + pd.Timedelta(days=365) <= cutoff)
    if debug:
        indices = np.flatnonzero(mask.to_numpy())[-4000:]
    else:
        indices = np.flatnonzero(mask.to_numpy())
    matrix = features.iloc[indices]
    labels = seeds.iloc[indices]["success_rate"].to_numpy(dtype=np.float64)
    age_years = (cutoff - seeds.iloc[indices]["timestamp"]).dt.days.to_numpy(dtype=np.float64) / 365.25
    weights = np.power(0.5, age_years / HALF_LIFE_YEARS)
    categorical = [column for column in ["country", "state", "city"] if column in matrix]
    dataset = lgb.Dataset(matrix, label=labels, weight=weights, categorical_feature=categorical, free_raw_data=False)
    parameters = {
        "objective": "quantile",
        "alpha": 0.5,
        "metric": "l1",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 7,
        "min_data_in_leaf": 100,
        "lambda_l2": 10.0,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 941,
        "num_threads": 22,
        "force_col_wise": True,
    }
    rounds = 70 if debug else FACILITY_ROUNDS
    return lgb.train(parameters, dataset, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])


def facility_predictions(model: Any, seeds: pd.DataFrame, features: pd.DataFrame, split: str, year: int) -> tuple[np.ndarray, np.ndarray]:
    mask = seeds["split"].eq(split) & seeds["timestamp"].dt.year.eq(year)
    indices = np.flatnonzero(mask.to_numpy())
    return indices, np.clip(model.predict(features.iloc[indices]), 0.0, 1.0)


# Ratio decoding

def exact_ratio_decode(q: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    count = len(q)
    if count == 0:
        return np.nan, np.nan, 0.0
    probability = np.zeros((count + 1, count + 1), dtype=np.float64)
    probability[0, 0] = 1.0
    for index, (reporting, success) in enumerate(zip(q, p)):
        current = probability[:index + 1, :index + 1].copy()
        probability[:index + 1, :index + 1] = current * (1.0 - reporting)
        probability[1:index + 2, :index + 1] += current * reporting * (1.0 - success)
        probability[1:index + 2, 1:index + 2] += current * reporting * success
    n_values, k_values = np.indices(probability.shape)
    mask = n_values > 0
    weights = probability[mask]
    ratios = k_values[mask] / n_values[mask]
    presence = float(weights.sum())
    if presence <= 1e-15:
        return float(np.average(p, weights=np.maximum(q, 1e-12))), float(np.average(p, weights=np.maximum(q, 1e-12))), presence
    weights = weights / presence
    return _weighted_median(ratios, weights), float(np.dot(ratios, weights)), presence


def monte_carlo_ratio_decode(q: np.ndarray, p: np.ndarray, seed: int, draws: int = 2048) -> tuple[float, float, float]:
    random = np.random.default_rng(seed)
    reports = np.zeros(draws, dtype=np.int32)
    successes = np.zeros(draws, dtype=np.int32)
    for reporting, success in zip(q, p):
        current = random.random(draws) < reporting
        reports += current
        successes += current & (random.random(draws) < success)
    mask = reports > 0
    presence = float(-np.expm1(np.log1p(-np.clip(q, 0.0, 1.0 - 1e-12)).sum()))
    if not mask.any():
        fallback = float(np.average(p, weights=np.maximum(q, 1e-12)))
        return fallback, fallback, presence
    ratios = successes[mask] / reports[mask]
    return float(np.median(ratios)), float(np.mean(ratios)), presence


def decode_edges(
    edge: pd.DataFrame,
    q_by_pair: dict[int, float],
    p_by_pair: dict[int, float],
    pair_lookup: dict[tuple[pd.Timestamp, int], int],
    fallback: np.ndarray,
    origin_year: int,
) -> dict[str, np.ndarray]:
    size = len(fallback)
    median = fallback.astype(np.float64).copy()
    mean = fallback.astype(np.float64).copy()
    presence = np.zeros(size, dtype=np.float64)
    roster_count = np.zeros(size, dtype=np.int32)
    expected_count = np.zeros(size, dtype=np.float64)
    dominant_share = np.zeros(size, dtype=np.float64)
    if len(edge):
        current = edge.copy()
        current["pair_row"] = [pair_lookup[(pd.Timestamp(timestamp), int(nct))] for timestamp, nct in zip(current["timestamp"], current["nct_id"])]
        current["q"] = current["pair_row"].map(q_by_pair)
        current["p"] = current["pair_row"].map(p_by_pair)
        current = current[current["q"].notna() & current["p"].notna()]
        for seed_row, group in current.groupby("seed_row", sort=False):
            q = group["q"].to_numpy(dtype=np.float64)
            p = group["p"].to_numpy(dtype=np.float64)
            order = np.argsort(q)[::-1]
            q = q[order]
            p = p[order]
            if len(q) <= 32:
                decoded = exact_ratio_decode(q, p)
            else:
                decoded = monte_carlo_ratio_decode(q, p, 1337 + int(seed_row) + 100000 * origin_year)
            median[int(seed_row)], mean[int(seed_row)], presence[int(seed_row)] = decoded
            roster_count[int(seed_row)] = len(q)
            expected_count[int(seed_row)] = q.sum()
            dominant_share[int(seed_row)] = q.max() / max(q.sum(), 1e-12)
    return {
        "median": np.clip(median, 0.0, 1.0),
        "mean": np.clip(mean, 0.0, 1.0),
        "presence": presence,
        "roster_count": roster_count,
        "expected_count": expected_count,
        "dominant_share": dominant_share,
    }


def _candidate_truth(bundle: TrialBundle, pair_rows: np.ndarray, origin: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    current = bundle.pairs.iloc[pair_rows]
    future = bundle.event_info[
        (bundle.event_info["date"] > origin)
        & (bundle.event_info["date"] <= origin + pd.Timedelta(days=365))
    ].drop_duplicates("nct_id").set_index("nct_id")
    report = current["nct_id"].isin(future.index).to_numpy(dtype=np.int8)
    success = current["nct_id"].map(future["success_all"]).to_numpy(dtype=np.float64)
    return report, success


def _pair_rows_for_year(bundle: TrialBundle, year: int) -> np.ndarray:
    needed = set(bundle.edges[year]["nct_id"].unique())
    mask = bundle.pairs["timestamp"].eq(pd.Timestamp(f"{year}-01-01")) & bundle.pairs["nct_id"].isin(needed)
    return np.flatnonzero(mask.to_numpy()).astype(np.int64)


def _fallback_for_year(
    context: Any,
    facility_seeds: pd.DataFrame,
    facility_features: pd.DataFrame,
    year: int,
    debug: bool,
) -> tuple[np.ndarray, np.ndarray]:
    origin = pd.Timestamp(f"{year}-01-01")
    model = fit_facility_model(facility_seeds, facility_features, origin, debug)
    indices, prediction = facility_predictions(model, facility_seeds, facility_features, "train", year)
    labels = facility_seeds.iloc[indices]["success_rate"].to_numpy(dtype=np.float64)
    return prediction, labels


# Forward selection

def run_forward_selection(
    bundle: TrialBundle,
    context: Any,
    facility_seeds: pd.DataFrame,
    facility_features: pd.DataFrame,
    debug: bool,
) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
    years = [2018, 2019]
    schemes = ["equal"]
    text_weights = [0.0] if debug else [0.0, 0.25, 0.5]
    fallbacks = {year: _fallback_for_year(context, facility_seeds, facility_features, year, debug) for year in years}
    facility_slices: dict[int, dict[str, np.ndarray]] = {}
    for year in years:
        mask = facility_seeds["split"].eq("train") & facility_seeds["timestamp"].dt.year.eq(year)
        current_features = facility_features.loc[mask]
        facility_slices[year] = {
            "history_count": current_features["history_report_count"].fillna(0).to_numpy(dtype=np.float64),
            "history_recency": current_features["history_event_recency"].fillna(np.inf).to_numpy(dtype=np.float64),
        }
    records: dict[str, Any] = {}
    candidates: list[tuple[tuple[float, float, float], dict[str, Any], Any, Any, dict[str, Any]]] = []
    for scheme in schemes:
        fold_data: dict[int, dict[str, Any]] = {}
        q_values = []
        q_labels = []
        q_weights = []
        for year in years:
            origin = pd.Timestamp(f"{year}-01-01")
            head = fit_heads(bundle, origin, scheme, debug)
            pair_rows = _pair_rows_for_year(bundle, year)
            q, _, tabular, text = predict_heads(head, bundle, pair_rows, 0.0)
            reporting, success = _candidate_truth(bundle, pair_rows, origin)
            sites = bundle.features.iloc[pair_rows]["site_count"].to_numpy(dtype=np.float64)
            reporting_scheme = "equal" if scheme.startswith("success_") else scheme
            success_scheme = scheme.removeprefix("success_") if scheme.startswith("success_") else scheme
            reporting_weight = np.ones(len(pair_rows), dtype=np.float64)
            if reporting_scheme == "sqrt_site":
                reporting_weight = np.sqrt(np.maximum(sites, 1.0))
            elif reporting_scheme == "full_site":
                reporting_weight = np.maximum(sites, 1.0)
            success_weight = np.ones(len(pair_rows), dtype=np.float64)
            if success_scheme == "sqrt_site":
                success_weight = np.sqrt(np.maximum(sites, 1.0))
            elif success_scheme == "full_site":
                success_weight = np.maximum(sites, 1.0)
            fold_data[year] = {
                "head": head,
                "pair_rows": pair_rows,
                "q": q,
                "tabular": tabular,
                "text": text,
                "reporting": reporting,
                "success": success,
                "reporting_weights": reporting_weight,
                "success_weights": success_weight,
            }
            q_values.append(q)
            q_labels.append(reporting)
            q_weights.append(reporting_weight)
        reporting_calibrator = fit_platt(np.concatenate(q_values), np.concatenate(q_labels), np.concatenate(q_weights))
        for text_weight in text_weights:
            success_values = []
            success_labels = []
            success_weights = []
            for year in years:
                item = fold_data[year]
                mask = item["reporting"].astype(bool)
                raw = (1.0 - text_weight) * item["tabular"] + text_weight * item["text"]
                success_values.append(raw[mask])
                success_labels.append(item["success"][mask].astype(np.int8))
                success_weights.append(item["success_weights"][mask])
            success_calibrator = fit_platt(np.concatenate(success_values), np.concatenate(success_labels), np.concatenate(success_weights))
            fold_decoded: dict[int, dict[str, np.ndarray]] = {}
            metrics: dict[str, Any] = {}
            for year in years:
                item = fold_data[year]
                q = _apply_calibrator(item["q"], reporting_calibrator)
                raw_p = (1.0 - text_weight) * item["tabular"] + text_weight * item["text"]
                p = _apply_calibrator(raw_p, success_calibrator)
                if getattr(bundle, "direct_poles", False) and "registry_expert_probability" in bundle.features:
                    expert = pd.to_numeric(bundle.features.iloc[item["pair_rows"]]["registry_expert_probability"], errors="coerce").to_numpy(dtype=np.float64)
                    covered = np.isfinite(expert)
                    p[covered] = expert[covered]
                q_map = dict(zip(item["pair_rows"], q))
                p_map = dict(zip(item["pair_rows"], p))
                lookup = {
                    (pd.Timestamp(row.timestamp), int(row.nct_id)): int(row.pair_row)
                    for row in bundle.pairs.iloc[item["pair_rows"]].itertuples()
                }
                fallback, labels = fallbacks[year]
                decoded = decode_edges(bundle.edges[year], q_map, p_map, lookup, fallback, year)
                fold_decoded[year] = decoded
                metrics[str(year)] = {
                    "reporting_auc": float(roc_auc_score(item["reporting"], q, sample_weight=item["reporting_weights"])),
                    "reporting_logloss": float(log_loss(item["reporting"], q, sample_weight=item["reporting_weights"])),
                    "success_auc": float(roc_auc_score(item["success"][item["reporting"].astype(bool)], p[item["reporting"].astype(bool)], sample_weight=item["success_weights"][item["reporting"].astype(bool)])),
                }
            modes = {
                "lattice_median": lambda decoded, fallback, year: decoded["median"],
                "conditional_mean": lambda decoded, fallback, year: decoded["mean"],
                "soft_median_15": lambda decoded, fallback, year: 0.85 * decoded["median"] + 0.15 * fallback,
                "soft_median_30": lambda decoded, fallback, year: 0.70 * decoded["median"] + 0.30 * fallback,
                "low_coverage_05": lambda decoded, fallback, year: np.where((decoded["presence"] < 0.05) | (decoded["roster_count"] == 0), fallback, decoded["median"]),
                "low_coverage_10": lambda decoded, fallback, year: np.where((decoded["presence"] < 0.10) | (decoded["roster_count"] == 0), fallback, decoded["median"]),
                "low_coverage_20": lambda decoded, fallback, year: np.where((decoded["presence"] < 0.20) | (decoded["roster_count"] == 0), fallback, decoded["median"]),
                "low_coverage_fallback": lambda decoded, fallback, year: np.where((decoded["presence"] < 0.35) | (decoded["roster_count"] == 0), fallback, decoded["median"]),
                "rich_history_blend_25": lambda decoded, fallback, year: np.where(facility_slices[year]["history_count"] >= 5, 0.75 * decoded["median"] + 0.25 * fallback, decoded["median"]),
                "rich_history_blend_50": lambda decoded, fallback, year: np.where(facility_slices[year]["history_count"] >= 5, 0.50 * decoded["median"] + 0.50 * fallback, decoded["median"]),
                "dormant_history_blend_25": lambda decoded, fallback, year: np.where((facility_slices[year]["history_count"] > 0) & (facility_slices[year]["history_recency"] > 365), 0.75 * decoded["median"] + 0.25 * fallback, decoded["median"]),
            }
            for mode, function in modes.items():
                fold_mae = {}
                predictions = {}
                for year in years:
                    fallback, labels = fallbacks[year]
                    prediction = np.clip(function(fold_decoded[year], fallback, year), 0.0, 1.0)
                    fold_mae[year] = float(mean_absolute_error(labels, prediction))
                    predictions[year] = prediction
                criterion = (fold_mae[2019], float(np.mean(list(fold_mae.values()))), float(np.std(list(fold_mae.values()))))
                selection = {"scheme": scheme, "text_weight": text_weight, "mode": mode, "fold_mae": fold_mae}
                diagnostic = {"head_metrics": metrics, "predictions": predictions, "decoded": fold_decoded}
                candidates.append((criterion, selection, reporting_calibrator, success_calibrator, diagnostic))
                records[f"{scheme}|text={text_weight}|{mode}"] = {"fold_mae": fold_mae, "criterion": criterion}
    recent_best = min(candidates, key=lambda item: item[0][0])
    recent_labels = fallbacks[2019][1]
    recent_noise = connected_bootstrap(
        recent_labels,
        {"recent_best": recent_best[4]["predictions"][2019]},
        bundle.edges[2019],
        draws=100,
    )["bootstrap_standard_error"]
    eligible = [item for item in candidates if item[0][0] <= recent_best[0][0] + recent_noise]
    criterion, selection, reporting_calibrator, success_calibrator, diagnostic = min(
        eligible,
        key=lambda item: (item[0][1], item[0][2], item[0][0], item[1]["scheme"] != "equal"),
    )
    selected_predictions = diagnostic["predictions"]
    deltas = []
    for year in years:
        labels = fallbacks[year][1]
        prediction = selected_predictions[year]
        baseline = fallbacks[year][0]
        deltas.extend(_paired_component_deltas(labels, prediction, baseline, bundle.edges[year], 200, 1337 + year))
    acceptance = {
        "paired_bootstrap_draws": len(deltas),
        "p_delta_mae_below_zero": float(np.mean(np.asarray(deltas) < 0)),
        "delta_mae_mean": float(np.mean(deltas)),
        "delta_mae_se": float(np.std(deltas, ddof=1)),
    }
    diagnostics = {
        "candidates": records,
        "selected": selection,
        "head_metrics": diagnostic["head_metrics"],
        "acceptance": acceptance,
        "recent_window_clustered_se": recent_noise,
        "recent_window_tied_candidates": len(eligible),
    }
    return selection, reporting_calibrator, success_calibrator, diagnostics


# Final inference

def _select_decoding(mode: str, decoded: dict[str, np.ndarray], fallback: np.ndarray, segment: dict[str, np.ndarray]) -> np.ndarray:
    if mode == "lattice_median":
        prediction = decoded["median"]
    elif mode == "conditional_mean":
        prediction = decoded["mean"]
    elif mode == "soft_median_15":
        prediction = 0.85 * decoded["median"] + 0.15 * fallback
    elif mode == "soft_median_30":
        prediction = 0.70 * decoded["median"] + 0.30 * fallback
    elif mode == "low_coverage_05":
        prediction = np.where((decoded["presence"] < 0.05) | (decoded["roster_count"] == 0), fallback, decoded["median"])
    elif mode == "low_coverage_10":
        prediction = np.where((decoded["presence"] < 0.10) | (decoded["roster_count"] == 0), fallback, decoded["median"])
    elif mode == "low_coverage_20":
        prediction = np.where((decoded["presence"] < 0.20) | (decoded["roster_count"] == 0), fallback, decoded["median"])
    elif mode == "low_coverage_fallback":
        prediction = np.where((decoded["presence"] < 0.35) | (decoded["roster_count"] == 0), fallback, decoded["median"])
    elif mode == "rich_history_blend_25":
        prediction = np.where(segment["history_count"] >= 5, 0.75 * decoded["median"] + 0.25 * fallback, decoded["median"])
    elif mode == "rich_history_blend_50":
        prediction = np.where(segment["history_count"] >= 5, 0.50 * decoded["median"] + 0.50 * fallback, decoded["median"])
    elif mode == "dormant_history_blend_25":
        mask = (segment["history_count"] > 0) & (segment["history_recency"] > 365)
        prediction = np.where(mask, 0.75 * decoded["median"] + 0.25 * fallback, decoded["median"])
    else:
        raise ValueError(f"unknown decoding mode {mode}")
    return np.clip(prediction, 0.0, 1.0)


def final_prediction(
    bundle: TrialBundle,
    facility_seeds: pd.DataFrame,
    facility_features: pd.DataFrame,
    year: int,
    split: str,
    selection: dict[str, Any],
    reporting_calibrator: Any,
    success_calibrator: Any,
    debug: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], HeadFit]:
    origin = pd.Timestamp(f"{year}-01-01")
    head = fit_heads(bundle, origin, selection["scheme"], debug)
    head.reporting_calibrator = reporting_calibrator
    head.success_calibrator = success_calibrator
    pair_rows = _pair_rows_for_year(bundle, year)
    q, p, tabular, text = predict_heads(head, bundle, pair_rows, float(selection["text_weight"]))
    q_map = dict(zip(pair_rows, q))
    p_map = dict(zip(pair_rows, p))
    lookup = {
        (pd.Timestamp(row.timestamp), int(row.nct_id)): int(row.pair_row)
        for row in bundle.pairs.iloc[pair_rows].itertuples()
    }
    facility_model = fit_facility_model(facility_seeds, facility_features, origin, debug)
    _, fallback = facility_predictions(facility_model, facility_seeds, facility_features, split, year)
    segment_mask = facility_seeds["split"].eq(split) & facility_seeds["timestamp"].dt.year.eq(year)
    segment_features = facility_features.loc[segment_mask]
    segment = {
        "history_count": segment_features["history_report_count"].fillna(0).to_numpy(dtype=np.float64),
        "history_recency": segment_features["history_event_recency"].fillna(np.inf).to_numpy(dtype=np.float64),
    }
    decoded = decode_edges(bundle.edges[year], q_map, p_map, lookup, fallback, year)
    prediction = _select_decoding(selection["mode"], decoded, fallback, segment)
    decoded["fallback"] = fallback
    decoded["selected"] = prediction
    decoded["trial_q"] = q
    decoded["trial_p"] = p
    decoded["trial_tabular_p"] = tabular
    decoded["trial_text_p"] = text
    decoded["pair_rows"] = pair_rows
    return prediction, decoded, head


# Diagnostics

def _connected_components(size: int, edge: pd.DataFrame) -> np.ndarray:
    parent = np.arange(size, dtype=np.int64)
    rank = np.zeros(size, dtype=np.int8)

    def find(value: int) -> int:
        current = value
        while parent[current] != current:
            parent[current] = parent[parent[current]]
            current = int(parent[current])
        return current

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    if len(edge):
        for _, rows in edge.groupby("nct_id")["seed_row"]:
            values = rows.drop_duplicates().to_numpy(dtype=np.int64)
            if len(values) > 1:
                anchor = int(values[0])
                for value in values[1:]:
                    union(anchor, int(value))
    roots = np.asarray([find(index) for index in range(size)], dtype=np.int64)
    _, components = np.unique(roots, return_inverse=True)
    return components


def _paired_component_deltas(
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    edge: pd.DataFrame,
    draws: int,
    seed: int,
) -> list[float]:
    components = _connected_components(len(labels), edge)
    units = np.unique(components)
    members = {unit: np.flatnonzero(components == unit) for unit in units}
    random = np.random.default_rng(seed)
    result = []
    for _ in range(draws):
        selected = random.choice(units, len(units), replace=True)
        indices = np.concatenate([members[unit] for unit in selected])
        result.append(mean_absolute_error(labels[indices], candidate[indices]) - mean_absolute_error(labels[indices], baseline[indices]))
    return result


def connected_bootstrap(
    labels: np.ndarray,
    candidates: dict[str, np.ndarray],
    edge: pd.DataFrame,
    draws: int = 100,
) -> dict[str, Any]:
    components = _connected_components(len(labels), edge)
    units = np.unique(components)
    random = np.random.default_rng(1337)
    best = min(candidates, key=lambda name: mean_absolute_error(labels, candidates[name]))
    scores = []
    for _ in range(draws):
        selected = random.choice(units, len(units), replace=True)
        indices = np.concatenate([np.flatnonzero(components == unit) for unit in selected])
        scores.append(mean_absolute_error(labels[indices], candidates[best][indices]))
    correlations = []
    names = list(candidates)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            correlation = spearmanr(candidates[left], candidates[right]).statistic
            if np.isfinite(correlation):
                correlations.append(float(correlation))
    return {
        "cluster_count": int(len(units)),
        "best_candidate": best,
        "bootstrap_standard_error": float(np.std(scores, ddof=1)),
        "mean_pairwise_rank_correlation": float(np.mean(correlations)) if correlations else 1.0,
        "candidate_mae": {name: float(mean_absolute_error(labels, prediction)) for name, prediction in candidates.items()},
    }


def slice_diagnostics(
    labels: np.ndarray,
    prediction: np.ndarray,
    seed_features: pd.DataFrame,
    decoded: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    specifications: dict[str, pd.Series] = {}
    history_count = seed_features["history_report_count"].fillna(0)
    specifications["history_depth"] = pd.cut(history_count, [-1, 0, 5, np.inf], labels=["zero", "sparse", "rich"])
    recency = seed_features["history_event_recency"]
    specifications["history_recency"] = pd.cut(recency, [-np.inf, 92, 365, np.inf], labels=["under_92d", "92_to_365d", "over_365d"])
    specifications["visible_roster"] = pd.cut(decoded["roster_count"], [-1, 0, 1, 2, np.inf], labels=["zero", "one", "two", "three_plus"])
    specifications["roster_probability"] = pd.cut(decoded["presence"], [-1, 0.35, 0.75, 0.95, 1.0], labels=["low", "medium", "high", "very_high"], include_lowest=True)
    specifications["likely_reports"] = pd.cut(decoded["expected_count"], [-1, 0.75, 1.5, np.inf], labels=["under_one", "one", "multiple"])
    specifications["country"] = seed_features["country"].astype(str).where(seed_features["country"].astype(str).isin(seed_features["country"].astype(str).value_counts().head(8).index), "other")
    result: dict[str, dict[str, float]] = {}
    for axis, values in specifications.items():
        values = pd.Series(values).reset_index(drop=True)
        for value in values.dropna().unique():
            mask = values.eq(value).to_numpy()
            result[f"{axis}:{value}"] = {
                "count": int(mask.sum()),
                "label_mean": float(labels[mask].mean()),
                "mae": float(mean_absolute_error(labels[mask], prediction[mask])),
            }
    return result
