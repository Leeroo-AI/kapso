from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer


# Configuration

CACHE_VERSION = "study_adverse_lane0_v4"
DAY_NS = 86_400_000_000_000
KEYWORDS = (
    "adverse",
    "serious",
    "death",
    "mortality",
    "safety",
    "toxicity",
    "hospital",
    "survival",
    "cancer",
    "oncology",
    "cardiovascular",
    "infection",
    "pediatric",
    "elderly",
    "placebo",
    "random",
    "double blind",
    "phase 3",
)
PRIOR_RELATIONS = (
    "sponsor",
    "agency",
    "condition",
    "intervention",
    "facility",
    "country",
    "state",
    "city",
    "phase",
    "study_type",
    "phase_type",
    "phase_agency",
    "country_phase",
)
PROFILE_RELATIONS = (
    "sponsor",
    "agency",
    "condition",
    "intervention",
    "facility",
    "country",
    "phase",
    "study_type",
)
PARENT_RELATION = {
    "sponsor": "agency",
    "facility": "country",
    "state": "country",
    "city": "state",
    "phase_agency": "agency",
    "country_phase": "phase",
}


# Storage

def cache_root(shared_cache: Path) -> Path:
    path = shared_cache / CACHE_VERSION
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary = Path(handle.name)
    temporary.replace(path)


def register_artifact(shared_cache: Path, name: str, path: Path, description: str) -> None:
    registry = shared_cache / "artifacts.json"
    lock_path = shared_cache / "artifacts.lock"
    lock_path.touch(exist_ok=True)
    import fcntl

    with lock_path.open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if registry.exists():
                try:
                    entries = json.loads(registry.read_text())
                except Exception:
                    entries = []
            else:
                entries = []
            relative = str(path.relative_to(shared_cache))
            if not any(x.get("name") == name and x.get("content_key") == CACHE_VERSION for x in entries):
                entries.append(
                    {
                        "name": name,
                        "path": relative,
                        "description": description,
                        "content_key": CACHE_VERSION,
                        "rebuild_hint": "Run python main.py; the cache is rebuilt on a version-key miss.",
                    }
                )
                with tempfile.NamedTemporaryFile("w", dir=shared_cache, delete=False) as handle:
                    json.dump(entries, handle, indent=2)
                    handle.write("\n")
                    temporary = Path(handle.name)
                temporary.replace(registry)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


# Task rows

def load_task_rows(task) -> pd.DataFrame:
    target = task.target_col
    frames = []
    offset = 0
    for split in ("train", "val"):
        frame = task.get_table(split, mask_input_cols=False).df[
            ["timestamp", "nct_id", target]
        ].copy()
        frame = frame.rename(columns={target: "target"})
        frame["split"] = split
        frame["_row_id"] = np.arange(len(frame), dtype=np.int32)
        frame["row_idx"] = np.arange(offset, offset + len(frame), dtype=np.int32)
        offset += len(frame)
        frames.append(frame)
    frame = task.get_table("test").df[["timestamp", "nct_id"]].copy()
    frame["target"] = np.nan
    frame["split"] = "test"
    frame["_row_id"] = np.arange(len(frame), dtype=np.int32)
    frame["row_idx"] = np.arange(offset, offset + len(frame), dtype=np.int32)
    frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"])
    rows["available_time"] = rows["timestamp"] + pd.Timedelta(days=365)
    return rows


def rows_fingerprint(rows: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(
        rows[["split", "_row_id", "nct_id", "timestamp"]], index=False
    ).values
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


# Scalar transforms

def _clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def _category(series: pd.Series) -> pd.Series:
    return _clean_text(series).str.lower().replace("", "__missing__")


def _age_years(value) -> float:
    if value is None or pd.isna(value):
        return np.nan
    match = re.search(
        r"([-+]?[0-9]*\.?[0-9]+)\s*(year|month|week|day|hour|minute)",
        str(value).lower(),
    )
    if not match:
        return np.nan
    amount = float(match.group(1))
    unit = match.group(2)
    factor = {
        "year": 1.0,
        "month": 1.0 / 12.0,
        "week": 1.0 / 52.1429,
        "day": 1.0 / 365.0,
        "hour": 1.0 / 8760.0,
        "minute": 1.0 / 525600.0,
    }[unit]
    return amount * factor


def _phase_number(value) -> float:
    text = str(value).lower()
    numbers = re.findall(r"[0-4]", text)
    if not numbers:
        return np.nan
    return float(np.mean([int(x) for x in numbers]))


def _text_statistics(frame: pd.DataFrame, name: str, series: pd.Series) -> None:
    text = _clean_text(series)
    lower = text.str.lower()
    frame[f"{name}_chars"] = text.str.len().astype(np.float32)
    frame[f"{name}_words"] = text.str.count(r"\b\w+\b").astype(np.float32)
    frame[f"{name}_lines"] = series.fillna("").astype(str).str.count(r"\n").astype(np.float32)
    frame[f"{name}_digits"] = text.str.count(r"\d").astype(np.float32)
    frame[f"{name}_missing"] = (text.str.len() == 0).astype(np.float32)
    for keyword in KEYWORDS:
        token = re.sub(r"[^a-z0-9]+", "_", keyword)
        frame[f"{name}_kw_{token}"] = lower.str.count(re.escape(keyword)).astype(np.float32)


def _visible_singleton(rows: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    merged = rows[["row_idx", "nct_id", "timestamp"]].merge(
        source, on="nct_id", how="left", sort=False
    )
    if "date" in merged.columns:
        merged = merged[merged["date"].isna() | (merged["date"] <= merged["timestamp"])]
        merged = merged.sort_values(["row_idx", "date"]).drop_duplicates("row_idx", keep="last")
    else:
        merged = merged.drop_duplicates("row_idx", keep="last")
    return rows[["row_idx"]].merge(merged.drop(columns=["timestamp"], errors="ignore"), on="row_idx", how="left")


def _entropy(frame: pd.DataFrame, row_col: str, entity_col: str) -> pd.Series:
    if len(frame) == 0:
        return pd.Series(dtype=np.float32)
    counts = frame.groupby([row_col, entity_col], observed=True).size().rename("count").reset_index()
    totals = counts.groupby(row_col, observed=True)["count"].transform("sum")
    probabilities = counts["count"] / totals
    counts["part"] = -probabilities * np.log(np.maximum(probabilities, 1e-12))
    return counts.groupby(row_col, observed=True)["part"].sum().astype(np.float32)


# Relation warehouse

def _query_bridge(rows: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    merged = rows[["row_idx", "nct_id", "timestamp"]].merge(
        bridge, on="nct_id", how="inner", sort=False
    )
    merged = merged[merged["date"] <= merged["timestamp"]]
    return merged


def _make_membership(
    frame: pd.DataFrame,
    entity,
    parent=None,
    keep_duplicates: bool = False,
) -> pd.DataFrame:
    output = frame[["row_idx", "timestamp"]].copy()
    output["entity"] = pd.Series(entity, index=frame.index).fillna("").astype(str).values
    if parent is not None:
        output["parent"] = pd.Series(parent, index=frame.index).fillna("").astype(str).values
    output = output[output["entity"].ne("")]
    if not keep_duplicates:
        output = output.drop_duplicates(["row_idx", "entity"])
    return output.reset_index(drop=True)


def _asof_lookup(left: pd.DataFrame, daily: pd.DataFrame, offset_days: int) -> np.ndarray:
    query = left[["entity", "timestamp"]].copy()
    query["_position"] = np.arange(len(query))
    query["lookup_time"] = query["timestamp"] - pd.Timedelta(days=offset_days)
    query = query.sort_values(["lookup_time", "entity"])
    reference = daily.rename(columns={"date": "lookup_time"}).sort_values(
        ["lookup_time", "entity"]
    )
    merged = pd.merge_asof(
        query,
        reference,
        on="lookup_time",
        by="entity",
        direction="backward",
        allow_exact_matches=True,
    )
    values = np.zeros(len(query), dtype=np.float32)
    values[merged["_position"].to_numpy()] = merged["cumulative"].fillna(0).to_numpy(np.float32)
    return values


def _attach_support(query: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    if len(query) == 0:
        output = query.copy()
        for name in ("support", "support_recent", "support_previous", "support_growth"):
            output[name] = np.float32(0)
        return output
    source = universe[["entity", "nct_id", "date"]].dropna(subset=["entity", "date"]).copy()
    source["entity"] = source["entity"].astype(str)
    source = source.drop_duplicates(["entity", "nct_id", "date"])
    daily = (
        source.groupby(["entity", "date"], observed=True)["nct_id"]
        .nunique()
        .rename("increment")
        .reset_index()
        .sort_values(["entity", "date"])
    )
    daily["cumulative"] = daily.groupby("entity", observed=True)["increment"].cumsum()
    daily = daily[["entity", "date", "cumulative"]]
    output = query.copy()
    current = _asof_lookup(output, daily, 0)
    prior = _asof_lookup(output, daily, 730)
    older = _asof_lookup(output, daily, 1460)
    recent = current - prior
    previous = prior - older
    output["support"] = current
    output["support_recent"] = recent
    output["support_previous"] = previous
    output["support_growth"] = np.log1p(recent) - np.log1p(previous)
    return output


def _support_features(features: pd.DataFrame, membership: pd.DataFrame, prefix: str) -> None:
    if len(membership) == 0:
        for stat in (
            "entity_count",
            "support_mean",
            "support_max",
            "support_min",
            "rare_fraction",
            "recent_support_mean",
            "support_growth_mean",
        ):
            features[f"{prefix}_{stat}"] = np.float32(0)
        return
    group = membership.groupby("row_idx", observed=True)
    values = pd.DataFrame(index=features.index)
    values[f"{prefix}_entity_count"] = group["entity"].nunique()
    values[f"{prefix}_support_mean"] = group["support"].mean()
    values[f"{prefix}_support_max"] = group["support"].max()
    values[f"{prefix}_support_min"] = group["support"].min()
    rare = membership.assign(rare=(membership["support"] <= 5).astype(np.float32))
    values[f"{prefix}_rare_fraction"] = rare.groupby("row_idx", observed=True)["rare"].mean()
    values[f"{prefix}_recent_support_mean"] = group["support_recent"].mean()
    values[f"{prefix}_support_growth_mean"] = group["support_growth"].mean()
    for column in values:
        features[column] = values[column].reindex(features.index).fillna(0).astype(np.float32)


def _scalar_membership(
    rows: pd.DataFrame, values: pd.Series, parents: pd.Series | None = None
) -> pd.DataFrame:
    frame = rows[["row_idx", "timestamp"]].copy()
    frame["entity"] = values.fillna("__missing__").astype(str).values
    if parents is not None:
        frame["parent"] = parents.fillna("__missing__").astype(str).values
    return frame


# Result profiles

def build_result_profiles(db, rows: pd.DataFrame, enrollment: pd.Series) -> pd.DataFrame:
    labeled = rows[rows["target"].notna()][
        ["row_idx", "nct_id", "timestamp", "available_time"]
    ].copy()
    profile = pd.DataFrame(index=labeled["row_idx"].to_numpy())
    enrollment_map = enrollment.reindex(profile.index).fillna(0)
    profile["study_enrollment_for_rate"] = enrollment_map

    reported = labeled.merge(db.table_dict["reported_event_totals"].df, on="nct_id", how="left")
    reported = reported[reported["date"].notna() & (reported["date"] <= reported["available_time"])]
    if len(reported):
        event = _category(reported["event_type"])
        affected = reported["subjects_affected"].fillna(0).astype(float)
        risk = reported["subjects_at_risk"].replace(0, np.nan).astype(float)
        reported = reported.assign(
            serious_affected=np.where(event.eq("serious"), affected, 0.0),
            death_affected=np.where(event.eq("deaths"), affected, 0.0),
            affected=affected,
            affected_rate=np.divide(
                affected,
                risk,
                out=np.zeros(len(reported), dtype=float),
                where=risk.notna().to_numpy(),
            ),
        )
        grouped = reported.groupby("row_idx", observed=True)
        profile["result_serious_affected"] = grouped["serious_affected"].sum()
        profile["result_death_affected"] = grouped["death_affected"].sum()
        profile["result_affected_total"] = grouped["affected"].sum()
        profile["result_affected_rate_mean"] = grouped["affected_rate"].mean()
        profile["result_classification_diversity"] = grouped["classification"].nunique()
        profile["result_event_rows"] = grouped.size()

    withdrawal = labeled.merge(db.table_dict["drop_withdrawals"].df, on="nct_id", how="left")
    withdrawal = withdrawal[
        withdrawal["date"].notna() & (withdrawal["date"] <= withdrawal["available_time"])
    ]
    if len(withdrawal):
        reason = _category(withdrawal["reason"])
        withdrawal = withdrawal.assign(
            withdrawal_count=withdrawal["count"].fillna(0).astype(float),
            adverse_flag=reason.str.contains("adverse|tox|side effect", regex=True).astype(float),
            death_flag=reason.str.contains("death|died|mortality", regex=True).astype(float),
        )
        grouped = withdrawal.groupby("row_idx", observed=True)
        profile["result_withdrawal_total"] = grouped["withdrawal_count"].sum()
        profile["result_withdrawal_reason_diversity"] = grouped["reason"].nunique()
        profile["result_withdrawal_adverse_fraction"] = grouped["adverse_flag"].mean()
        profile["result_withdrawal_death_fraction"] = grouped["death_flag"].mean()
        profile["result_withdrawal_rows"] = grouped.size()
        profile["result_withdrawal_norm"] = profile["result_withdrawal_total"] / (
            enrollment_map + 1.0
        )

    outcomes = labeled.merge(db.table_dict["outcomes"].df, on="nct_id", how="left")
    outcomes = outcomes[outcomes["date"].notna() & (outcomes["date"] <= outcomes["available_time"])]
    if len(outcomes):
        outcome_text = (
            _clean_text(outcomes["title"])
            + " "
            + _clean_text(outcomes["description"])
            + " "
            + _clean_text(outcomes["time_frame"])
        ).str.lower()
        outcome_type = _category(outcomes["outcome_type"])
        outcomes = outcomes.assign(
            safety_flag=outcome_text.str.contains(
                "adverse|safety|toxicity|serious event", regex=True
            ).astype(float),
            death_flag=outcome_text.str.contains(
                "death|mortality|survival", regex=True
            ).astype(float),
            primary_flag=outcome_type.str.contains("primary").astype(float),
            timeframe_length=_clean_text(outcomes["time_frame"]).str.len().astype(float),
        )
        grouped = outcomes.groupby("row_idx", observed=True)
        profile["result_outcome_count"] = grouped.size()
        profile["result_outcome_safety_fraction"] = grouped["safety_flag"].mean()
        profile["result_outcome_death_fraction"] = grouped["death_flag"].mean()
        profile["result_outcome_primary_fraction"] = grouped["primary_flag"].mean()
        profile["result_outcome_timeframe_length"] = grouped["timeframe_length"].mean()
        profile["result_outcome_unit_diversity"] = grouped["units"].nunique()
        profile["result_outcome_type_diversity"] = grouped["outcome_type"].nunique()

    analyses = labeled.merge(db.table_dict["outcome_analyses"].df, on="nct_id", how="left")
    analyses = analyses[
        analyses["date"].notna() & (analyses["date"] <= analyses["available_time"])
    ]
    if len(analyses):
        pvalue = analyses["p_value"].astype(float)
        width = analyses["ci_upper_limit"].astype(float) - analyses["ci_lower_limit"].astype(float)
        numeric_columns = [
            "param_value",
            "dispersion_value",
            "p_value",
            "ci_percent",
            "ci_lower_limit",
            "ci_upper_limit",
        ]
        analyses = analyses.assign(
            significant=np.where(pvalue.notna(), (pvalue < 0.05).astype(float), np.nan),
            ci_width=width.abs(),
            effect_abs=analyses["param_value"].astype(float).abs(),
            analysis_missing=analyses[numeric_columns].isna().mean(axis=1),
        )
        grouped = analyses.groupby("row_idx", observed=True)
        profile["result_analysis_count"] = grouped.size()
        profile["result_analysis_significant_fraction"] = grouped["significant"].mean()
        profile["result_analysis_ci_width"] = grouped["ci_width"].median()
        profile["result_analysis_effect_abs"] = grouped["effect_abs"].median()
        profile["result_analysis_missingness"] = grouped["analysis_missing"].mean()

    profile = profile.reindex(labeled["row_idx"]).replace([np.inf, -np.inf], np.nan)
    count_columns = [
        column
        for column in profile
        if column.endswith("_count")
        or column.endswith("_rows")
        or column.endswith("_total")
        or column.endswith("_diversity")
        or column.endswith("_affected")
    ]
    profile[count_columns] = profile[count_columns].fillna(0)
    return profile.astype(np.float32)


# Local warehouse

def build_local_warehouse(db, rows: pd.DataFrame, shared_cache: Path, debug: bool = False):
    key = rows_fingerprint(rows)
    path = cache_root(shared_cache) / f"local_{key}.pkl"
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)

    query = rows.set_index("row_idx", drop=False)
    studies = db.table_dict["studies"].df.copy()
    studies_view = rows[["row_idx", "nct_id", "timestamp"]].merge(
        studies, on="nct_id", how="left", sort=False
    ).set_index("row_idx")
    features = pd.DataFrame(index=query.index)
    enrollment = studies_view["enrollment"].astype(float)
    features["study_enrollment"] = enrollment
    features["study_log_enrollment"] = np.log1p(enrollment.clip(lower=0))
    features["study_sqrt_enrollment"] = np.sqrt(enrollment.clip(lower=0))
    age_days = (
        studies_view["timestamp"] - pd.to_datetime(studies_view["start_date"])
    ).dt.total_seconds() / 86400.0
    features["study_age_days"] = age_days
    features["study_log_age_days"] = np.log1p(age_days.clip(lower=0))
    features["study_start_year"] = pd.to_datetime(studies_view["start_date"]).dt.year
    features["study_start_month"] = pd.to_datetime(studies_view["start_date"]).dt.month
    features["study_number_of_arms"] = studies_view["number_of_arms"].astype(float)
    features["study_number_of_groups"] = studies_view["number_of_groups"].astype(float)
    features["study_phase_number"] = studies_view["phase"].map(_phase_number).astype(float)
    study_columns = [
        "target_duration",
        "study_type",
        "acronym",
        "baseline_population",
        "brief_title",
        "official_title",
        "phase",
        "enrollment",
        "enrollment_type",
        "source",
        "number_of_arms",
        "number_of_groups",
        "has_dmc",
        "is_fda_regulated_drug",
        "is_fda_regulated_device",
        "is_unapproved_device",
        "is_us_export",
        "biospec_retention",
        "biospec_description",
        "source_class",
        "plan_to_share_ipd",
        "detailed_descriptions",
        "brief_summaries",
    ]
    features["study_missing_count"] = studies_view[study_columns].isna().sum(axis=1)
    features["study_missing_fraction"] = studies_view[study_columns].isna().mean(axis=1)
    for name, column in (
        ("brief_title", "brief_title"),
        ("official_title", "official_title"),
        ("brief_summary", "brief_summaries"),
        ("detailed_description", "detailed_descriptions"),
    ):
        _text_statistics(features, name, studies_view[column])

    designs = _visible_singleton(rows, db.table_dict["designs"].df).set_index("row_idx")
    design_object = [
        "allocation",
        "intervention_model",
        "observational_model",
        "primary_purpose",
        "time_perspective",
        "masking",
        "masking_description",
        "intervention_model_description",
        "subject_masked",
        "caregiver_masked",
        "investigator_masked",
        "outcomes_assessor_masked",
    ]
    features["design_missing_count"] = designs[design_object].isna().sum(axis=1)
    masked = [
        "subject_masked",
        "caregiver_masked",
        "investigator_masked",
        "outcomes_assessor_masked",
    ]
    features["design_masking_breadth"] = designs[masked].apply(
        lambda x: x.fillna("").astype(str).str.lower().isin(["yes", "true", "1"])
    ).sum(axis=1)
    _text_statistics(features, "masking_description", designs["masking_description"])
    _text_statistics(
        features,
        "intervention_model_description",
        designs["intervention_model_description"],
    )

    eligibility = _visible_singleton(rows, db.table_dict["eligibilities"].df).set_index("row_idx")
    minimum_age = eligibility["minimum_age"].map(_age_years)
    maximum_age = eligibility["maximum_age"].map(_age_years)
    features["eligibility_minimum_age_years"] = minimum_age
    features["eligibility_maximum_age_years"] = maximum_age
    features["eligibility_age_span_years"] = maximum_age - minimum_age
    features["eligibility_age_midpoint_years"] = (maximum_age + minimum_age) / 2.0
    eligibility_columns = [
        "sampling_method",
        "gender",
        "minimum_age",
        "maximum_age",
        "healthy_volunteers",
        "population",
        "criteria",
        "gender_description",
        "gender_based",
        "adult",
        "child",
        "older_adult",
    ]
    features["eligibility_missing_count"] = eligibility[eligibility_columns].isna().sum(axis=1)
    _text_statistics(features, "eligibility_criteria", eligibility["criteria"])
    _text_statistics(features, "eligibility_population", eligibility["population"])
    criteria = eligibility["criteria"].fillna("").astype(str)
    features["eligibility_inclusion_sections"] = criteria.str.lower().str.count("inclusion")
    features["eligibility_exclusion_sections"] = criteria.str.lower().str.count("exclusion")
    features["eligibility_bullet_count"] = criteria.str.count(r"[\n\r]\s*[-*•]")

    categorical = pd.DataFrame(index=features.index)
    for name, series in (
        ("study_type", studies_view["study_type"]),
        ("phase", studies_view["phase"]),
        ("enrollment_type", studies_view["enrollment_type"]),
        ("source_class", studies_view["source_class"]),
        ("has_dmc", studies_view["has_dmc"]),
        ("fda_drug", studies_view["is_fda_regulated_drug"]),
        ("fda_device", studies_view["is_fda_regulated_device"]),
        ("share_ipd", studies_view["plan_to_share_ipd"]),
        ("allocation", designs["allocation"]),
        ("intervention_model", designs["intervention_model"]),
        ("observational_model", designs["observational_model"]),
        ("primary_purpose", designs["primary_purpose"]),
        ("time_perspective", designs["time_perspective"]),
        ("masking", designs["masking"]),
        ("gender", eligibility["gender"]),
        ("healthy_volunteers", eligibility["healthy_volunteers"]),
        ("adult", eligibility["adult"]),
        ("child", eligibility["child"]),
        ("older_adult", eligibility["older_adult"]),
    ):
        categorical[name] = _category(series)
    dummies = pd.get_dummies(
        categorical,
        prefix=[f"cat_{column}" for column in categorical.columns],
        dtype=np.float32,
    )
    features = pd.concat([features, dummies], axis=1)

    conditions_bridge = _query_bridge(rows, db.table_dict["conditions_studies"].df)
    conditions_bridge = conditions_bridge.merge(
        db.table_dict["conditions"].df, on="condition_id", how="left"
    )
    condition_membership = _make_membership(
        conditions_bridge, conditions_bridge["condition_id"]
    )
    condition_universe = db.table_dict["conditions_studies"].df[
        ["nct_id", "condition_id", "date"]
    ].copy()
    condition_universe["entity"] = condition_universe["condition_id"].astype(str)
    condition_membership = _attach_support(condition_membership, condition_universe)

    interventions_bridge = _query_bridge(rows, db.table_dict["interventions_studies"].df)
    interventions_bridge = interventions_bridge.merge(
        db.table_dict["interventions"].df, on="intervention_id", how="left"
    )
    intervention_membership = _make_membership(
        interventions_bridge, interventions_bridge["intervention_id"]
    )
    intervention_universe = db.table_dict["interventions_studies"].df[
        ["nct_id", "intervention_id", "date"]
    ].copy()
    intervention_universe["entity"] = intervention_universe["intervention_id"].astype(str)
    intervention_membership = _attach_support(intervention_membership, intervention_universe)

    sponsors_bridge = _query_bridge(rows, db.table_dict["sponsors_studies"].df)
    sponsors_bridge = sponsors_bridge.merge(
        db.table_dict["sponsors"].df, on="sponsor_id", how="left"
    )
    sponsor_membership = _make_membership(
        sponsors_bridge,
        sponsors_bridge["sponsor_id"],
        _category(sponsors_bridge["agency_class"]),
    )
    sponsor_universe = db.table_dict["sponsors_studies"].df[
        ["nct_id", "sponsor_id", "date"]
    ].copy()
    sponsor_universe["entity"] = sponsor_universe["sponsor_id"].astype(str)
    sponsor_membership = _attach_support(sponsor_membership, sponsor_universe)
    agency_membership = _make_membership(
        sponsors_bridge, _category(sponsors_bridge["agency_class"])
    )
    agency_universe = db.table_dict["sponsors_studies"].df.merge(
        db.table_dict["sponsors"].df[["sponsor_id", "agency_class"]],
        on="sponsor_id",
        how="left",
    )
    agency_universe["entity"] = _category(agency_universe["agency_class"])
    agency_membership = _attach_support(agency_membership, agency_universe)

    facilities_bridge = _query_bridge(rows, db.table_dict["facilities_studies"].df)
    facilities_bridge = facilities_bridge.merge(
        db.table_dict["facilities"].df, on="facility_id", how="left"
    )
    country_value = _category(facilities_bridge["country"])
    state_value = country_value + "|" + _category(facilities_bridge["state"])
    city_value = state_value + "|" + _category(facilities_bridge["city"])
    facility_membership = _make_membership(
        facilities_bridge, facilities_bridge["facility_id"], country_value
    )
    country_membership = _make_membership(facilities_bridge, country_value)
    state_membership = _make_membership(facilities_bridge, state_value, country_value)
    city_membership = _make_membership(facilities_bridge, city_value, state_value)
    facility_universe = db.table_dict["facilities_studies"].df.merge(
        db.table_dict["facilities"].df[
            ["facility_id", "country", "state", "city"]
        ],
        on="facility_id",
        how="left",
    )
    facility_universe["entity"] = facility_universe["facility_id"].astype(str)
    facility_membership = _attach_support(facility_membership, facility_universe)
    facility_universe["country_entity"] = _category(facility_universe["country"])
    facility_universe["state_entity"] = (
        facility_universe["country_entity"] + "|" + _category(facility_universe["state"])
    )
    facility_universe["city_entity"] = (
        facility_universe["state_entity"] + "|" + _category(facility_universe["city"])
    )
    location_universe = facility_universe[["nct_id", "date"]].copy()
    location_universe["entity"] = facility_universe["country_entity"]
    country_membership = _attach_support(country_membership, location_universe)
    location_universe["entity"] = facility_universe["state_entity"]
    state_membership = _attach_support(state_membership, location_universe)
    location_universe["entity"] = facility_universe["city_entity"]
    city_membership = _attach_support(city_membership, location_universe)

    memberships = {
        "sponsor": sponsor_membership,
        "agency": agency_membership,
        "condition": condition_membership,
        "intervention": intervention_membership,
        "facility": facility_membership,
        "country": country_membership,
        "state": state_membership,
        "city": city_membership,
    }
    for name, membership in memberships.items():
        _support_features(features, membership, name)

    features["facility_count"] = facilities_bridge.groupby("row_idx").size().reindex(
        features.index, fill_value=0
    )
    for name, series in (
        ("facility_country_diversity", facilities_bridge.groupby("row_idx")["country"].nunique()),
        ("facility_state_diversity", facilities_bridge.groupby("row_idx")["state"].nunique()),
        ("facility_city_diversity", facilities_bridge.groupby("row_idx")["city"].nunique()),
        ("condition_term_diversity", conditions_bridge.groupby("row_idx")["mesh_term"].nunique()),
        ("intervention_term_diversity", interventions_bridge.groupby("row_idx")["mesh_term"].nunique()),
        ("sponsor_lead_count", sponsors_bridge.assign(
            lead=_category(sponsors_bridge["lead_or_collaborator"]).str.contains("lead").astype(float)
        ).groupby("row_idx")["lead"].sum()),
    ):
        features[name] = series.reindex(features.index, fill_value=0)
    country_entropy = _entropy(facilities_bridge, "row_idx", "country")
    state_entropy = _entropy(facilities_bridge, "row_idx", "state")
    features["facility_country_entropy"] = country_entropy.reindex(features.index, fill_value=0)
    features["facility_state_entropy"] = state_entropy.reindex(features.index, fill_value=0)
    if len(facilities_bridge):
        site_span = facilities_bridge.groupby("row_idx")["date"].agg(["min", "max"])
        span_days = (site_span["max"] - site_span["min"]).dt.total_seconds() / 86400.0
        features["facility_site_date_span"] = span_days.reindex(features.index, fill_value=0)
    else:
        features["facility_site_date_span"] = 0

    phase_values = _category(studies_view["phase"]).reindex(query.index)
    type_values = _category(studies_view["study_type"]).reindex(query.index)
    memberships["phase"] = _scalar_membership(rows, phase_values.reset_index(drop=True))
    memberships["study_type"] = _scalar_membership(rows, type_values.reset_index(drop=True))
    phase_type = phase_values.astype(str) + "|" + type_values.astype(str)
    memberships["phase_type"] = _scalar_membership(rows, phase_type.reset_index(drop=True))
    phase_lookup = phase_values.to_dict()
    phase_agency = agency_membership[["row_idx", "timestamp", "entity"]].copy()
    phase_agency["parent"] = phase_agency["entity"]
    phase_agency["entity"] = (
        phase_agency["row_idx"].map(phase_lookup).astype(str)
        + "|"
        + phase_agency["entity"].astype(str)
    )
    memberships["phase_agency"] = phase_agency
    country_phase = country_membership[["row_idx", "timestamp", "entity"]].copy()
    country_phase["parent"] = country_phase["row_idx"].map(phase_lookup).astype(str)
    country_phase["entity"] = (
        country_phase["entity"].astype(str)
        + "|"
        + country_phase["parent"].astype(str)
    )
    memberships["country_phase"] = country_phase

    condition_terms = (
        conditions_bridge.groupby("row_idx", observed=True)["mesh_term"]
        .agg(lambda x: " ".join(sorted(set(_clean_text(x)))))
        .reindex(features.index, fill_value="")
    )
    intervention_terms = (
        interventions_bridge.groupby("row_idx", observed=True)["mesh_term"]
        .agg(lambda x: " ".join(sorted(set(_clean_text(x)))))
        .reindex(features.index, fill_value="")
    )
    design_text = (
        _clean_text(designs["masking_description"])
        + " "
        + _clean_text(designs["intervention_model_description"])
    )
    documents = (
        _clean_text(studies_view["brief_title"])
        + " "
        + _clean_text(studies_view["official_title"])
        + " "
        + _clean_text(studies_view["brief_summaries"])
        + " "
        + _clean_text(studies_view["detailed_descriptions"])
        + " "
        + _clean_text(eligibility["criteria"])
        + " "
        + _clean_text(eligibility["population"])
        + " "
        + design_text
        + " "
        + condition_terms
        + " "
        + intervention_terms
    ).str.slice(0, 18000)
    documents = documents.reindex(features.index).fillna("")

    features["interaction_log_enrollment_phase"] = (
        features["study_log_enrollment"] * features["study_phase_number"].fillna(-1)
    )
    features["interaction_sites_log_enrollment"] = (
        np.log1p(features["facility_count"]) * features["study_log_enrollment"]
    )
    features["interaction_condition_intervention_count"] = (
        features["condition_entity_count"] * features["intervention_entity_count"]
    )
    features["interaction_trial_age_enrollment"] = (
        features["study_log_age_days"] * features["study_log_enrollment"]
    )

    features = features.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    result_profiles = build_result_profiles(db, rows, features["study_enrollment"])
    value = {
        "features": features,
        "documents": documents,
        "memberships": memberships,
        "result_profiles": result_profiles,
    }
    _atomic_pickle(path, value)
    register_artifact(
        shared_cache,
        f"{CACHE_VERSION}_local_warehouse",
        path,
        "Temporally censored query-local features, relation maps, documents, and completed-trial result profiles.",
    )
    return value


# Historical priors

def _global_statistics(history: pd.DataFrame, profile_columns: list[str], cutoff: pd.Timestamp):
    if len(history) == 0:
        result = {
            "zero_rate": 0.5,
            "median_y": 2.0,
            "mean_log_y": math.log1p(2.0),
            "q75_y": 15.0,
            "q90_y": 55.0,
            "median_log_rate": -3.0,
            "std_log_y": 1.0,
            "recency_730": math.log1p(2.0),
            "recency_1460": math.log1p(2.0),
        }
        result.update({column: 0.0 for column in profile_columns})
        return result
    age = (cutoff - history["available_time"]).dt.total_seconds() / 86400.0
    log_y = history["log_y"].to_numpy()
    result = {
        "zero_rate": float(history["zero"].mean()),
        "median_y": float(history["target"].median()),
        "mean_log_y": float(history["log_y"].mean()),
        "q75_y": float(history["target"].quantile(0.75)),
        "q90_y": float(history["target"].quantile(0.90)),
        "median_log_rate": float(history["log_rate"].median()),
        "std_log_y": float(history["log_y"].std()) if len(history) > 1 else 1.0,
    }
    for half_life in (730, 1460):
        weights = np.exp(-math.log(2.0) * age.to_numpy() / half_life)
        result[f"recency_{half_life}"] = float(np.sum(weights * log_y) / np.maximum(weights.sum(), 1e-12))
    for column in profile_columns:
        result[column] = float(history[column].mean()) if history[column].notna().any() else 0.0
    return result


def _entity_statistics(
    history_membership: pd.DataFrame,
    cutoff: pd.Timestamp,
    profile_columns: list[str],
) -> pd.DataFrame:
    history = history_membership[history_membership["available_time"] <= cutoff].copy()
    if len(history) == 0:
        return pd.DataFrame(
            columns=[
                "entity",
                "support",
                "zero_rate",
                "median_y",
                "mean_log_y",
                "q75_y",
                "q90_y",
                "median_log_rate",
                "std_log_y",
                "recency_730",
                "recency_1460",
            ]
            + profile_columns
        )
    group = history.groupby("entity", observed=True)
    stats = pd.DataFrame(index=group.size().index)
    stats["support"] = group["row_idx"].nunique()
    stats["zero_rate"] = group["zero"].mean()
    stats["median_y"] = group["target"].median()
    stats["mean_log_y"] = group["log_y"].mean()
    quantiles = group["target"].quantile([0.75, 0.90]).unstack()
    stats["q75_y"] = quantiles.get(0.75)
    stats["q90_y"] = quantiles.get(0.90)
    stats["median_log_rate"] = group["log_rate"].median()
    stats["std_log_y"] = group["log_y"].std()
    age = (cutoff - history["available_time"]).dt.total_seconds() / 86400.0
    for half_life in (730, 1460):
        weight = np.exp(-math.log(2.0) * age / half_life)
        weighted = history["log_y"] * weight
        numerator = weighted.groupby(history["entity"], observed=True).sum()
        denominator = weight.groupby(history["entity"], observed=True).sum()
        stats[f"recency_{half_life}"] = numerator / denominator
    if profile_columns:
        profile_means = group[profile_columns].mean()
        stats = stats.join(profile_means)
    return stats.reset_index()


def _aggregate_prior_members(
    query_membership: pd.DataFrame,
    relation: str,
    entity_stats: pd.DataFrame,
    parent_stats: pd.DataFrame | None,
    global_stats: dict,
    profile_columns: list[str],
) -> pd.DataFrame:
    if len(query_membership) == 0:
        return pd.DataFrame()
    membership_columns = ["row_idx", "entity"]
    if "parent" in query_membership:
        membership_columns.append("parent")
    mapped = query_membership[membership_columns].merge(
        entity_stats, on="entity", how="left"
    )
    base_columns = [
        "zero_rate",
        "median_y",
        "mean_log_y",
        "q75_y",
        "q90_y",
        "median_log_rate",
        "std_log_y",
        "recency_730",
        "recency_1460",
    ]
    if parent_stats is not None and "parent" in mapped:
        renamed = parent_stats.rename(
            columns={
                "entity": "parent",
                **{
                    column: f"parent_{column}"
                    for column in base_columns + profile_columns
                },
            }
        )
        keep = ["parent"] + [
            f"parent_{column}" for column in base_columns + profile_columns
        ]
        mapped = mapped.merge(renamed[keep], on="parent", how="left")
    else:
        for column in base_columns + profile_columns:
            mapped[f"parent_{column}"] = global_stats[column]
    support = mapped["support"].fillna(0).to_numpy(float)
    output = mapped[["row_idx"]].copy()
    output["known"] = (support > 0).astype(float)
    output["support"] = support
    for pseudo_count in (5, 15, 50):
        for column in base_columns:
            parent = mapped[f"parent_{column}"].fillna(global_stats[column]).to_numpy(float)
            entity = mapped[column].fillna(pd.Series(parent, index=mapped.index)).to_numpy(float)
            shrunk = (support * entity + pseudo_count * parent) / (support + pseudo_count)
            output[f"k{pseudo_count}_{column}"] = shrunk
    for column in profile_columns:
        parent = mapped[f"parent_{column}"].fillna(global_stats[column]).to_numpy(float)
        entity = mapped[column].fillna(pd.Series(parent, index=mapped.index)).to_numpy(float)
        output[f"profile_{column}"] = (support * entity + 15.0 * parent) / (support + 15.0)
    group = output.groupby("row_idx", observed=True)
    aggregated = pd.DataFrame(index=group.size().index)
    aggregated[f"prior_{relation}_entity_count"] = group.size()
    aggregated[f"prior_{relation}_known_fraction"] = group["known"].mean()
    aggregated[f"prior_{relation}_support_sum"] = group["support"].sum()
    aggregated[f"prior_{relation}_support_max"] = group["support"].max()
    aggregated[f"prior_{relation}_support_mean"] = group["support"].mean()
    for pseudo_count in (5, 15, 50):
        for column in base_columns:
            source = f"k{pseudo_count}_{column}"
            aggregated[f"prior_{relation}__k{pseudo_count}__{column}_mean"] = group[source].mean()
        aggregated[f"prior_{relation}__k{pseudo_count}__median_y_max"] = group[
            f"k{pseudo_count}_median_y"
        ].max()
        aggregated[f"prior_{relation}__k{pseudo_count}__median_y_dispersion"] = group[
            f"k{pseudo_count}_median_y"
        ].std()
    for column in profile_columns:
        aggregated[f"prior_{relation}_profile_{column}_mean"] = group[
            f"profile_{column}"
        ].mean()
    return aggregated


def build_historical_priors(
    rows: pd.DataFrame,
    memberships: dict[str, pd.DataFrame],
    result_profiles: pd.DataFrame,
    shared_cache: Path,
    debug: bool = False,
) -> pd.DataFrame:
    key = rows_fingerprint(rows)
    path = cache_root(shared_cache) / f"historical_{key}.pkl"
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)

    labeled = rows[rows["target"].notna()][
        ["row_idx", "target", "available_time"]
    ].copy()
    labeled["zero"] = (labeled["target"] == 0).astype(np.float32)
    labeled["log_y"] = np.log1p(labeled["target"].clip(lower=0))
    profile_columns = [
        column
        for column in result_profiles.columns
        if column != "study_enrollment_for_rate"
    ]
    labeled = labeled.merge(
        result_profiles.reset_index().rename(columns={"index": "row_idx"}),
        on="row_idx",
        how="left",
    )
    labeled["log_rate"] = np.log(
        (labeled["target"] + 1.0)
        / (labeled["study_enrollment_for_rate"].fillna(0) + 1.0)
    )
    history_memberships = {}
    for relation in PRIOR_RELATIONS:
        membership = memberships[relation].drop_duplicates(["row_idx", "entity"]).copy()
        history = membership.merge(labeled, on="row_idx", how="inner")
        history_memberships[relation] = history

    structured = pd.DataFrame(index=rows["row_idx"])
    cutoffs = sorted(rows["timestamp"].drop_duplicates())
    central_columns = [
        "zero_rate",
        "median_y",
        "mean_log_y",
        "q75_y",
        "q90_y",
        "median_log_rate",
        "std_log_y",
        "recency_730",
        "recency_1460",
    ]
    all_parts = []
    for cutoff in cutoffs:
        eligible_rows = labeled[labeled["available_time"] <= cutoff].copy()
        globals_for_cutoff = _global_statistics(
            eligible_rows, profile_columns, cutoff
        )
        stats_by_relation = {}
        for relation in PRIOR_RELATIONS:
            history = history_memberships[relation]
            stats_by_relation[relation] = _entity_statistics(
                history,
                cutoff,
                profile_columns if relation in PROFILE_RELATIONS else [],
            )
        query_index = rows.loc[rows["timestamp"] == cutoff, "row_idx"]
        cutoff_frame = pd.DataFrame(index=query_index)
        for relation in PRIOR_RELATIONS:
            query_membership = memberships[relation]
            query_membership = query_membership[
                query_membership["timestamp"] == cutoff
            ].drop_duplicates(["row_idx", "entity"])
            relation_profiles = profile_columns if relation in PROFILE_RELATIONS else []
            parent_name = PARENT_RELATION.get(relation)
            parent_stats = stats_by_relation.get(parent_name)
            aggregated = _aggregate_prior_members(
                query_membership,
                relation,
                stats_by_relation[relation],
                parent_stats,
                globals_for_cutoff,
                relation_profiles,
            )
            cutoff_frame = cutoff_frame.join(aggregated, how="left")
        for column in central_columns:
            cutoff_frame[f"prior_global_{column}"] = globals_for_cutoff[column]
        for column in profile_columns:
            cutoff_frame[f"prior_global_profile_{column}"] = globals_for_cutoff[column]
        all_parts.append(cutoff_frame)
    structured = pd.concat(all_parts).sort_index().reindex(rows["row_idx"])
    structured = structured.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    _atomic_pickle(path, structured)
    register_artifact(
        shared_cache,
        f"{CACHE_VERSION}_historical_priors",
        path,
        "Chronologically recomputed empirical-Bayes label priors and all-table historical result profiles.",
    )
    return structured


# Text features

def build_text_features(
    documents: pd.Series,
    rows: pd.DataFrame,
    shared_cache: Path,
    mode: str,
    debug: bool = False,
) -> np.ndarray:
    key = rows_fingerprint(rows)
    path = cache_root(shared_cache) / f"text_{mode}_{key}.npy"
    if path.exists() and not debug:
        return np.load(path, mmap_mode=None)
    text = documents.reindex(rows["row_idx"]).fillna("").astype(str).tolist()
    if debug:
        vectorizer = HashingVectorizer(
            n_features=96,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            dtype=np.float32,
        )
        return vectorizer.transform(text).toarray().astype(np.float32)
    if mode == "oof":
        fit_mask = (rows["split"] == "train") & (
            rows["available_time"] <= pd.Timestamp("2016-01-02")
        )
    elif mode == "a":
        fit_mask = rows["split"] == "train"
    elif mode == "b":
        fit_mask = rows["split"].isin(["train", "val"])
    else:
        raise ValueError(mode)
    fit_text = [text[i] for i in np.flatnonzero(fit_mask.to_numpy())]
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=5,
        max_features=35000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    character = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=5,
        max_features=25000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    fit_word = word.fit_transform(fit_text)
    fit_character = character.fit_transform(fit_text)
    fit_matrix = sparse.hstack([fit_word, fit_character], format="csr")
    svd = TruncatedSVD(n_components=96, n_iter=5, random_state=17)
    svd.fit(fit_matrix)
    del fit_word, fit_character, fit_matrix
    blocks = []
    block_size = 5000
    for start in range(0, len(text), block_size):
        block = text[start : start + block_size]
        matrix = sparse.hstack(
            [word.transform(block), character.transform(block)], format="csr"
        )
        blocks.append(svd.transform(matrix).astype(np.float32))
    values = np.vstack(blocks)
    np.save(path, values)
    register_artifact(
        shared_cache,
        f"{CACHE_VERSION}_text_{mode}",
        path,
        f"Word and character TF-IDF projected to 96 dimensions for mode {mode}.",
    )
    return values


# Text cohort priors

def _row_weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1)
    ordered_values = np.take_along_axis(values, order, axis=1)
    ordered_weights = np.take_along_axis(weights, order, axis=1)
    cumulative = np.cumsum(ordered_weights, axis=1)
    threshold = ordered_weights.sum(axis=1, keepdims=True) * 0.5
    position = (cumulative < threshold).sum(axis=1)
    position = np.minimum(position, ordered_values.shape[1] - 1)
    return ordered_values[np.arange(len(ordered_values)), position]


def build_text_neighbor_features(
    embeddings: np.ndarray,
    rows: pd.DataFrame,
    enrollment: pd.Series,
    shared_cache: Path,
    mode: str,
    debug: bool = False,
) -> np.ndarray:
    key = rows_fingerprint(rows)
    path = cache_root(shared_cache) / f"text_neighbors_{mode}_{key}.npy"
    if path.exists() and not debug:
        return np.load(path, mmap_mode=None)
    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-8)
    y = rows["target"].to_numpy(float)
    enrolled = enrollment.reindex(rows["row_idx"]).fillna(0).to_numpy(float)
    log_y = np.log1p(np.maximum(np.nan_to_num(y, nan=0.0), 0))
    log_rate = np.log(
        (np.maximum(np.nan_to_num(y, nan=0.0), 0) + 1.0)
        / (np.maximum(enrolled, 0) + 1.0)
    )
    columns = 29
    output = np.full((len(rows), columns), np.nan, dtype=np.float32)
    for cutoff in sorted(rows["timestamp"].drop_duplicates()):
        query_indices = np.flatnonzero((rows["timestamp"] == cutoff).to_numpy())
        history_indices = np.flatnonzero(
            rows["target"].notna().to_numpy()
            & (rows["available_time"] <= cutoff).to_numpy()
        )
        if len(history_indices) == 0:
            output[query_indices, 0:3] = 0
            continue
        history_vectors = vectors[history_indices]
        history_y = np.maximum(y[history_indices], 0)
        history_log_y = log_y[history_indices]
        history_log_rate = log_rate[history_indices]
        history_zero = (history_y == 0).astype(np.float32)
        history_age = (
            cutoff - rows.iloc[history_indices]["available_time"]
        ).dt.total_seconds().to_numpy() / 86400.0
        history_recency = np.exp(-math.log(2.0) * history_age / 1460.0)
        for start in range(0, len(query_indices), 384):
            block_indices = query_indices[start : start + 384]
            similarities = vectors[block_indices] @ history_vectors.T
            k = min(64, similarities.shape[1])
            nearest = np.argpartition(similarities, -k, axis=1)[:, -k:]
            nearest_similarity = np.take_along_axis(similarities, nearest, axis=1)
            order = np.argsort(-nearest_similarity, axis=1)
            nearest = np.take_along_axis(nearest, order, axis=1)
            nearest_similarity = np.take_along_axis(
                nearest_similarity, order, axis=1
            )
            neighbor_y = history_y[nearest]
            neighbor_log_y = history_log_y[nearest]
            neighbor_log_rate = history_log_rate[nearest]
            neighbor_zero = history_zero[nearest]
            neighbor_recency = history_recency[nearest]
            weights = np.exp(
                8.0
                * (
                    nearest_similarity
                    - nearest_similarity[:, :1]
                )
            )
            weights *= neighbor_recency
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
            values = np.zeros((len(block_indices), columns), dtype=np.float32)
            values[:, 0] = len(history_indices)
            values[:, 1] = nearest_similarity[:, 0]
            values[:, 2] = nearest_similarity.mean(axis=1)
            values[:, 3] = nearest_similarity[:, 0] - nearest_similarity[:, -1]
            destination = 4
            for neighbors in (8, 32, 64):
                use = min(neighbors, k)
                y_slice = neighbor_y[:, :use]
                log_y_slice = neighbor_log_y[:, :use]
                rate_slice = neighbor_log_rate[:, :use]
                zero_slice = neighbor_zero[:, :use]
                weight_slice = weights[:, :use]
                weight_slice /= np.maximum(
                    weight_slice.sum(axis=1, keepdims=True), 1e-12
                )
                median_y = np.median(y_slice, axis=1)
                q75_y = np.quantile(y_slice, 0.75, axis=1)
                median_rate = np.median(rate_slice, axis=1)
                weighted_log_y = np.sum(weight_slice * log_y_slice, axis=1)
                weighted_rate = np.sum(weight_slice * rate_slice, axis=1)
                weighted_median_y = _row_weighted_median(y_slice, weight_slice)
                expected_rate_count = (
                    np.exp(median_rate)
                    * (np.maximum(enrolled[block_indices], 0) + 1.0)
                    - 1.0
                )
                block_values = np.column_stack(
                    [
                        median_y,
                        q75_y,
                        zero_slice.mean(axis=1),
                        median_rate,
                        weighted_log_y,
                        weighted_rate,
                        weighted_median_y,
                        expected_rate_count,
                    ]
                )
                values[:, destination : destination + 8] = block_values
                destination += 8
            values[:, 28] = np.std(neighbor_log_rate, axis=1)
            output[block_indices] = values
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    if not debug:
        np.save(path, output)
        register_artifact(
            shared_cache,
            f"{CACHE_VERSION}_text_neighbors_{mode}",
            path,
            f"Temporally legal nearest-completed-trial text cohort priors for mode {mode}.",
        )
    return output
