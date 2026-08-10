from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


SEED = 20260809
MODEL_NAME = "NeuML/bioclinical-modernbert-base-embeddings"
MODEL_REVISION = "048ad4491de0fb4e2695bfe4705da67caf4804b8"
VIEWS = {"w2y": 730, "w5y": 1825, "all": None, "half5y": "half"}


def elapsed(start: float, phase: str) -> None:
    print(f"[pipeline] {phase}: {time.time() - start:.2f}s elapsed", flush=True)


def text_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def flag_value(value) -> float:
    value = text_value(value).lower()
    if value in {"yes", "true", "1", "y"}:
        return 1.0
    if value in {"no", "false", "0", "n"}:
        return 0.0
    return np.nan


def age_years(value) -> float:
    value = text_value(value).lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(year|month|week|day|hour|minute)", value)
    if match is None:
        return np.nan
    amount = float(match.group(1))
    unit = match.group(2)
    factors = {"year": 1.0, "month": 1 / 12, "week": 1 / 52.1429, "day": 1 / 365.25, "hour": 1 / 8766, "minute": 1 / 525960}
    return amount * factors[unit]


def criteria_metrics(value) -> tuple[float, ...]:
    value = text_value(value)
    lower = value.lower()
    lines = value.splitlines()
    words = re.findall(r"\b\w+\b", value)
    bullets = sum(bool(re.match(r"\s*(?:[-*•]|\d+[.)])\s+", line)) for line in lines)
    inclusion = lower.count("inclusion")
    exclusion = lower.count("exclusion")
    split = lower.find("exclusion criteria")
    inclusion_chars = len(value) if split < 0 else split
    exclusion_chars = 0 if split < 0 else len(value) - split
    balance = (inclusion_chars - exclusion_chars) / max(1, len(value))
    return float(len(value)), float(len(words)), float(len(lines)), float(bullets), float(inclusion), float(exclusion), float(balance)


def make_seeds(context) -> pd.DataFrame:
    frames = []
    offset = 0
    for split, table in (("train", context.train), ("val", context.val), ("test", context.test)):
        frame = table.df.copy()
        frame["split"] = split
        frame["split_row_id"] = np.arange(len(frame), dtype=np.int64)
        frame["row_id"] = np.arange(offset, offset + len(frame), dtype=np.int64)
        offset += len(frame)
        if "outcome" not in frame:
            frame["outcome"] = np.nan
        frames.append(frame[["row_id", "split_row_id", "split", "timestamp", "nct_id", "outcome"]])
    seeds = pd.concat(frames, ignore_index=True)
    seeds["timestamp"] = pd.to_datetime(seeds["timestamp"])
    return seeds


def build_episodes(db) -> pd.DataFrame:
    con = duckdb.connect()
    for name in ("outcome_analyses", "outcomes", "studies", "drop_withdrawals", "reported_event_totals"):
        con.register(name, db.table_dict[name].df)
    episodes = con.execute(
        """
        WITH eligible AS (
            SELECT oa.nct_id, oa.date report_date, oa.p_value, oa.ci_percent,
                   oa.non_inferiority_type, oa.method, oa.p_value_description,
                   oa.param_value, oa.dispersion_value, oa.ci_lower_limit,
                   oa.ci_upper_limit
            FROM outcome_analyses oa
            JOIN outcomes o ON oa.outcome_id = o.id
            WHERE (oa.p_value_modifier IS NULL OR oa.p_value_modifier != '>')
              AND oa.p_value BETWEEN 0 AND 1
              AND o.outcome_type = 'Primary'
        ), base AS (
            SELECT nct_id, report_date, count(*)::DOUBLE N,
                   sum((p_value <= 0.05)::INTEGER)::DOUBLE S,
                   min(p_value) min_p, median(p_value) median_p,
                   (min(p_value) <= 0.05)::INTEGER success,
                   avg((ci_percent IS NOT NULL)::INTEGER) ci_present,
                   avg((non_inferiority_type IS NOT NULL)::INTEGER) ni_present,
                   avg((method IS NOT NULL)::INTEGER) method_present,
                   avg((p_value_description IS NULL)::INTEGER) p_desc_missing,
                   avg((param_value IS NULL)::INTEGER) param_missing,
                   avg((dispersion_value IS NULL)::INTEGER) dispersion_missing,
                   avg((ci_lower_limit IS NULL OR ci_upper_limit IS NULL)::INTEGER) ci_limit_missing
            FROM eligible GROUP BY nct_id, report_date
        ), dropout AS (
            SELECT b.nct_id, b.report_date,
                   sum(coalesce(d.count, 0)) dropout_count
            FROM base b LEFT JOIN drop_withdrawals d
              ON b.nct_id = d.nct_id AND d.date <= b.report_date
            GROUP BY b.nct_id, b.report_date
        ), events AS (
            SELECT b.nct_id, b.report_date,
                   sum(CASE WHEN lower(coalesce(e.event_type, '')) LIKE '%serious%'
                            THEN coalesce(e.subjects_affected, 0) ELSE 0 END) serious_affected,
                   sum(CASE WHEN lower(coalesce(e.event_type, '')) LIKE '%serious%'
                            THEN coalesce(e.subjects_at_risk, 0) ELSE 0 END) serious_risk
            FROM base b LEFT JOIN reported_event_totals e
              ON b.nct_id = e.nct_id AND e.date <= b.report_date
            GROUP BY b.nct_id, b.report_date
        )
        SELECT b.*, b.S / greatest(b.N, 1) significant_fraction,
               date_diff('day', s.start_date, b.report_date)::DOUBLE reporting_delay,
               d.dropout_count / greatest(coalesce(s.enrollment, 0), 1) dropout_ratio,
               e.serious_affected / greatest(e.serious_risk, 1) serious_event_ratio
        FROM base b
        LEFT JOIN studies s ON b.nct_id = s.nct_id
        LEFT JOIN dropout d ON b.nct_id = d.nct_id AND b.report_date = d.report_date
        LEFT JOIN events e ON b.nct_id = e.nct_id AND b.report_date = e.report_date
        ORDER BY report_date, nct_id
        """
    ).df()
    episodes["report_date"] = pd.to_datetime(episodes["report_date"])
    episodes["dropout_ratio"] = episodes["dropout_ratio"].clip(0, 20)
    episodes["serious_event_ratio"] = episodes["serious_event_ratio"].clip(0, 5)
    con.close()
    if len(episodes) != 14164 or episodes["nct_id"].nunique() != 14164:
        raise RuntimeError(f"historical corpus mismatch: {len(episodes)} rows, {episodes['nct_id'].nunique()} trials")
    return episodes


def latest_rows(table: pd.DataFrame, key: str, time_col: str) -> pd.DataFrame:
    return table.sort_values(time_col).drop_duplicates(key, keep="last")


def direct_features(db, seeds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    studies = db.table_dict["studies"].df.copy()
    study_cols = [c for c in studies.columns if c not in {"nct_id", "start_date"}]
    study = seeds[["row_id", "nct_id", "timestamp"]].merge(studies, on="nct_id", how="left", validate="many_to_one")
    invalid = study["start_date"] > study["timestamp"]
    if invalid.any():
        study.loc[invalid, study_cols] = np.nan
    numeric = pd.DataFrame(index=seeds["row_id"])
    numeric.index.name = "row_id"
    enrollment = pd.to_numeric(study["enrollment"], errors="coerce")
    arms = pd.to_numeric(study["number_of_arms"], errors="coerce")
    groups = pd.to_numeric(study["number_of_groups"], errors="coerce")
    numeric["study_log_enrollment"] = np.log1p(enrollment.clip(lower=0)).to_numpy()
    numeric["study_enrollment"] = enrollment.to_numpy()
    numeric["study_arms"] = arms.to_numpy()
    numeric["study_groups"] = groups.to_numpy()
    numeric["study_age_days"] = (study["timestamp"] - study["start_date"]).dt.days.to_numpy(dtype=float)
    numeric["study_enrollment_per_arm"] = (enrollment / arms.clip(lower=1)).to_numpy()
    numeric["study_enrollment_per_group"] = (enrollment / groups.clip(lower=1)).to_numpy()
    for col in ["has_dmc", "is_fda_regulated_drug", "is_fda_regulated_device", "is_unapproved_device", "is_ppsd", "is_us_export", "fdaaa801_violation", "plan_to_share_ipd"]:
        numeric[f"study_{col}"] = study[col].map(flag_value).to_numpy()
    numeric["study_missing_count"] = study[study_cols].isna().sum(axis=1).to_numpy(dtype=float)
    numeric["study_title_chars"] = study["brief_title"].map(text_value).str.len().to_numpy(dtype=float)
    numeric["study_summary_chars"] = study["brief_summaries"].map(text_value).str.len().to_numpy(dtype=float)
    numeric["study_description_chars"] = study["detailed_descriptions"].map(text_value).str.len().to_numpy(dtype=float)
    categories = pd.DataFrame(index=seeds["row_id"])
    categories.index.name = "row_id"
    for col in ["study_type", "phase", "source", "source_class", "enrollment_type", "biospec_retention"]:
        categories[f"study_{col}"] = study[col].map(text_value).replace("", "__MISSING__").to_numpy()

    designs = latest_rows(db.table_dict["designs"].df.copy(), "nct_id", "date")
    design = seeds[["row_id", "nct_id", "timestamp"]].merge(designs, on="nct_id", how="left", validate="many_to_one")
    design_cols = [c for c in designs.columns if c not in {"id", "nct_id", "date"}]
    invalid = design["date"] > design["timestamp"]
    if invalid.any():
        design.loc[invalid, design_cols] = np.nan
    numeric["design_missing_count"] = design[design_cols].isna().sum(axis=1).to_numpy(dtype=float)
    for col in ["subject_masked", "caregiver_masked", "investigator_masked", "outcomes_assessor_masked"]:
        numeric[f"design_{col}"] = design[col].map(flag_value).to_numpy()
    for col in ["masking_description", "intervention_model_description", "nonexistent"]:
        if col in design:
            numeric[f"design_{col}_chars"] = design[col].map(text_value).str.len().to_numpy(dtype=float)
    for col in ["allocation", "intervention_model", "observational_model", "primary_purpose", "time_perspective", "masking"]:
        categories[f"design_{col}"] = design[col].map(text_value).replace("", "__MISSING__").to_numpy()

    eligibilities = latest_rows(db.table_dict["eligibilities"].df.copy(), "nct_id", "date")
    eligibility = seeds[["row_id", "nct_id", "timestamp"]].merge(eligibilities, on="nct_id", how="left", validate="many_to_one")
    elig_cols = [c for c in eligibilities.columns if c not in {"id", "nct_id", "date"}]
    invalid = eligibility["date"] > eligibility["timestamp"]
    if invalid.any():
        eligibility.loc[invalid, elig_cols] = np.nan
    numeric["elig_min_age_years"] = eligibility["minimum_age"].map(age_years).to_numpy(dtype=float)
    numeric["elig_max_age_years"] = eligibility["maximum_age"].map(age_years).to_numpy(dtype=float)
    numeric["elig_age_span_years"] = numeric["elig_max_age_years"] - numeric["elig_min_age_years"]
    for col in ["healthy_volunteers", "adult", "child", "older_adult", "gender_based"]:
        numeric[f"elig_{col}"] = eligibility[col].map(flag_value).to_numpy()
    metrics = np.asarray([criteria_metrics(x) for x in eligibility["criteria"]], dtype=float)
    metric_names = ["chars", "words", "lines", "bullets", "inclusion_mentions", "exclusion_mentions", "balance"]
    for j, name in enumerate(metric_names):
        numeric[f"elig_criteria_{name}"] = metrics[:, j]
    numeric["elig_population_chars"] = eligibility["population"].map(text_value).str.len().to_numpy(dtype=float)
    numeric["elig_missing_count"] = eligibility[elig_cols].isna().sum(axis=1).to_numpy(dtype=float)
    for col in ["sampling_method", "gender"]:
        categories[f"elig_{col}"] = eligibility[col].map(text_value).replace("", "__MISSING__").to_numpy()

    seed_maps = {}
    sponsors_rel = db.table_dict["sponsors_studies"].df.copy()
    sponsors = db.table_dict["sponsors"].df.copy()
    sponsor_rows = seeds[["row_id", "nct_id", "timestamp"]].merge(sponsors_rel, on="nct_id", how="left")
    sponsor_rows = sponsor_rows[sponsor_rows["date"].le(sponsor_rows["timestamp"])].merge(sponsors, on="sponsor_id", how="left")
    grouped = sponsor_rows.groupby("row_id", sort=False)
    numeric["rel_sponsor_count"] = grouped["sponsor_id"].nunique().reindex(numeric.index, fill_value=0)
    lead = sponsor_rows["lead_or_collaborator"].map(text_value).str.lower().eq("lead")
    sponsor_rows["is_lead"] = lead.astype(float)
    sponsor_rows["is_industry"] = sponsor_rows["agency_class"].map(text_value).str.lower().eq("industry").astype(float)
    numeric["rel_lead_sponsor_count"] = grouped["is_lead"].sum().reindex(numeric.index, fill_value=0)
    numeric["rel_industry_sponsor_fraction"] = grouped["is_industry"].mean().reindex(numeric.index)
    numeric["rel_sponsor_class_count"] = grouped["agency_class"].nunique().reindex(numeric.index, fill_value=0)
    sponsor_rows["recency"] = (sponsor_rows["timestamp"] - sponsor_rows["date"]).dt.days
    numeric["rel_sponsor_recency_min"] = grouped["recency"].min().reindex(numeric.index)
    numeric["rel_sponsor_recency_mean"] = grouped["recency"].mean().reindex(numeric.index)
    seed_maps["lead_sponsor"] = sponsor_rows[sponsor_rows["is_lead"].eq(1)][["row_id", "sponsor_id"]].rename(columns={"sponsor_id": "key"}).drop_duplicates()
    seed_maps["all_sponsor"] = sponsor_rows[["row_id", "sponsor_id"]].rename(columns={"sponsor_id": "key"}).dropna().drop_duplicates()

    relation_specs = [("condition", "conditions_studies", "condition_id"), ("intervention", "interventions_studies", "intervention_id")]
    for prefix, table_name, key in relation_specs:
        relation = db.table_dict[table_name].df.copy()
        rows = seeds[["row_id", "nct_id", "timestamp"]].merge(relation, on="nct_id", how="left")
        rows = rows[rows["date"].le(rows["timestamp"])]
        rows["recency"] = (rows["timestamp"] - rows["date"]).dt.days
        grouped = rows.groupby("row_id", sort=False)
        numeric[f"rel_{prefix}_count"] = grouped[key].nunique().reindex(numeric.index, fill_value=0)
        numeric[f"rel_{prefix}_recency_min"] = grouped["recency"].min().reindex(numeric.index)
        numeric[f"rel_{prefix}_recency_mean"] = grouped["recency"].mean().reindex(numeric.index)
        seed_maps[prefix] = rows[["row_id", key]].rename(columns={key: "key"}).dropna().drop_duplicates()

    facility_rel = db.table_dict["facilities_studies"].df.copy()
    facilities = db.table_dict["facilities"].df.copy()
    facility_rows = seeds[["row_id", "nct_id", "timestamp"]].merge(facility_rel, on="nct_id", how="left")
    facility_rows = facility_rows[facility_rows["date"].le(facility_rows["timestamp"])].merge(facilities, on="facility_id", how="left")
    facility_rows["country_clean"] = facility_rows["country"].map(text_value).replace("", "__MISSING__")
    facility_rows["is_us"] = facility_rows["country_clean"].str.lower().isin({"united states", "united states of america", "usa"}).astype(float)
    facility_rows["recency"] = (facility_rows["timestamp"] - facility_rows["date"]).dt.days
    grouped = facility_rows.groupby("row_id", sort=False)
    numeric["rel_site_count"] = grouped["facility_id"].nunique().reindex(numeric.index, fill_value=0)
    numeric["rel_country_count"] = grouped["country_clean"].nunique().reindex(numeric.index, fill_value=0)
    numeric["rel_us_fraction"] = grouped["is_us"].mean().reindex(numeric.index)
    numeric["rel_multinational"] = (numeric["rel_country_count"] > 1).astype(float)
    numeric["rel_facility_recency_min"] = grouped["recency"].min().reindex(numeric.index)
    numeric["rel_facility_recency_mean"] = grouped["recency"].mean().reindex(numeric.index)
    country_counts = facility_rows.groupby(["row_id", "country_clean"], sort=False).size().rename("n").reset_index()
    country_counts["total"] = country_counts.groupby("row_id")["n"].transform("sum")
    country_counts["sq"] = (country_counts["n"] / country_counts["total"]) ** 2
    numeric["rel_country_site_hhi"] = country_counts.groupby("row_id")["sq"].sum().reindex(numeric.index)
    numeric["study_enrollment_per_site"] = numeric["study_enrollment"] / numeric["rel_site_count"].clip(lower=1)
    seed_maps["facility"] = facility_rows[["row_id", "facility_id"]].rename(columns={"facility_id": "key"}).dropna().drop_duplicates()
    seed_maps["country"] = facility_rows[["row_id", "country_clean"]].rename(columns={"country_clean": "key"}).dropna().drop_duplicates()

    seed_maps["source"] = pd.DataFrame({"row_id": seeds["row_id"], "key": categories["study_source"].to_numpy()})
    seed_maps["source_class"] = pd.DataFrame({"row_id": seeds["row_id"], "key": categories["study_source_class"].to_numpy()})
    seed_maps["phase_purpose"] = pd.DataFrame({"row_id": seeds["row_id"], "key": categories["study_phase"].to_numpy() + "||" + categories["design_primary_purpose"].to_numpy()})
    return numeric.astype(float), categories, seed_maps


def historical_maps(db, episodes: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    studies = db.table_dict["studies"].df[["nct_id", "start_date", "source", "source_class", "phase"]].copy()
    hist_maps = {}
    all_maps = {}

    sponsors_rel = db.table_dict["sponsors_studies"].df[["nct_id", "sponsor_id", "lead_or_collaborator", "date"]].copy()
    joined = episodes[["nct_id", "report_date"]].merge(sponsors_rel, on="nct_id", how="left")
    joined = joined[joined["date"].le(joined["report_date"])]
    hist_maps["all_sponsor"] = joined[["nct_id", "sponsor_id"]].rename(columns={"sponsor_id": "key"}).dropna().drop_duplicates()
    hist_maps["lead_sponsor"] = joined[joined["lead_or_collaborator"].map(text_value).str.lower().eq("lead")][["nct_id", "sponsor_id"]].rename(columns={"sponsor_id": "key"}).dropna().drop_duplicates()
    sponsor_all = sponsors_rel.merge(studies[["nct_id", "start_date"]], on="nct_id", how="left")
    all_maps["all_sponsor"] = sponsor_all[["nct_id", "sponsor_id", "date", "start_date"]].rename(columns={"sponsor_id": "key"}).dropna(subset=["key"])
    all_maps["lead_sponsor"] = sponsor_all[sponsor_all["lead_or_collaborator"].map(text_value).str.lower().eq("lead")][["nct_id", "sponsor_id", "date", "start_date"]].rename(columns={"sponsor_id": "key"}).dropna(subset=["key"])

    for prefix, table_name, key in [("condition", "conditions_studies", "condition_id"), ("intervention", "interventions_studies", "intervention_id"), ("facility", "facilities_studies", "facility_id")]:
        relation = db.table_dict[table_name].df[["nct_id", key, "date"]].copy()
        joined = episodes[["nct_id", "report_date"]].merge(relation, on="nct_id", how="left")
        joined = joined[joined["date"].le(joined["report_date"])]
        hist_maps[prefix] = joined[["nct_id", key]].rename(columns={key: "key"}).dropna().drop_duplicates()
        all_maps[prefix] = relation.merge(studies[["nct_id", "start_date"]], on="nct_id", how="left")[["nct_id", key, "date", "start_date"]].rename(columns={key: "key"}).dropna(subset=["key"])

    facilities = db.table_dict["facilities"].df[["facility_id", "country"]].copy()
    facilities["key"] = facilities["country"].map(text_value).replace("", "__MISSING__")
    facility_hist = hist_maps["facility"].rename(columns={"key": "facility_id"}).merge(facilities[["facility_id", "key"]], on="facility_id", how="left")
    hist_maps["country"] = facility_hist[["nct_id", "key"]].dropna().drop_duplicates()
    facility_all = all_maps["facility"].rename(columns={"key": "facility_id"}).merge(facilities[["facility_id", "key"]], on="facility_id", how="left")
    all_maps["country"] = facility_all[["nct_id", "key", "date", "start_date"]].dropna(subset=["key"]).drop_duplicates(["nct_id", "key"])

    designs = latest_rows(db.table_dict["designs"].df[["nct_id", "primary_purpose", "date"]].copy(), "nct_id", "date")
    static = studies.merge(designs, on="nct_id", how="left")
    static["source"] = static["source"].map(text_value).replace("", "__MISSING__")
    static["source_class"] = static["source_class"].map(text_value).replace("", "__MISSING__")
    static["phase_purpose"] = static["phase"].map(text_value).replace("", "__MISSING__") + "||" + static["primary_purpose"].map(text_value).replace("", "__MISSING__")
    hist_trials = episodes[["nct_id", "report_date"]].merge(static, on="nct_id", how="left")
    for prefix in ["source", "source_class", "phase_purpose"]:
        hist_maps[prefix] = hist_trials[["nct_id", prefix]].rename(columns={prefix: "key"}).dropna().drop_duplicates()
        all_maps[prefix] = static[["nct_id", prefix, "start_date"]].rename(columns={prefix: "key"})
        all_maps[prefix]["date"] = all_maps[prefix]["start_date"]
    return hist_maps, all_maps


def entity_view_stats(episodes: pd.DataFrame, mapping: pd.DataFrame, registered: pd.DataFrame, cutoff: pd.Timestamp, view, excluded_nct_ids: np.ndarray) -> pd.DataFrame:
    history = episodes[episodes["report_date"].le(cutoff) & ~episodes["nct_id"].isin(excluded_nct_ids)].copy()
    if view == "half":
        history["weight"] = np.exp(-math.log(2) * (cutoff - history["report_date"]).dt.days / 1825.0)
    else:
        if view is not None:
            history = history[(cutoff - history["report_date"]).dt.days.le(view)]
        history["weight"] = 1.0
    if history.empty:
        return pd.DataFrame(columns=["key", "support", "success_rate", "mean_N", "significant_fraction", "reporting_delay", "reporting_rate", "dropout_ratio", "serious_event_ratio", "ci_present", "method_present", "ni_present"])
    global_rate = np.average(history["success"], weights=history["weight"])
    joined = mapping.merge(history, on="nct_id", how="inner")
    if joined.empty:
        return pd.DataFrame(columns=["key", "support", "success_rate", "mean_N", "significant_fraction", "reporting_delay", "reporting_rate", "dropout_ratio", "serious_event_ratio", "ci_present", "method_present", "ni_present"])
    joined = joined.drop_duplicates(["key", "nct_id"])
    joined["wy"] = joined["weight"] * joined["success"]
    measures = ["N", "significant_fraction", "reporting_delay", "dropout_ratio", "serious_event_ratio", "ci_present", "method_present", "ni_present"]
    for col in measures:
        joined[f"w_{col}"] = joined["weight"] * joined[col].fillna(joined[col].median())
    grouped = joined.groupby("key", sort=False, observed=True)
    stats = grouped[["weight", "wy"] + [f"w_{x}" for x in measures]].sum().reset_index()
    stats = stats.rename(columns={"weight": "support"})
    stats["success_rate"] = (stats["wy"] + 20.0 * global_rate) / (stats["support"] + 20.0)
    for col in measures:
        stats[col if col != "N" else "mean_N"] = stats[f"w_{col}"] / stats["support"].clip(lower=1e-9)
    eligible_registered = registered[registered["date"].le(cutoff) & registered["start_date"].le(cutoff)].copy()
    if view == "half":
        eligible_registered["reg_weight"] = np.exp(-math.log(2) * (cutoff - eligible_registered["start_date"]).dt.days.clip(lower=0) / 1825.0)
        denominator = eligible_registered.drop_duplicates(["key", "nct_id"]).groupby("key", observed=True)["reg_weight"].sum()
    else:
        if view is not None:
            eligible_registered = eligible_registered[(cutoff - eligible_registered["start_date"]).dt.days.le(view)]
        denominator = eligible_registered.drop_duplicates(["key", "nct_id"]).groupby("key", observed=True)["nct_id"].size()
    stats["reporting_rate"] = stats["support"] / stats["key"].map(denominator).clip(lower=1).to_numpy()
    keep = ["key", "support", "success_rate", "mean_N", "significant_fraction", "reporting_delay", "reporting_rate", "dropout_ratio", "serious_event_ratio", "ci_present", "method_present", "ni_present"]
    return stats[keep]


def aggregate_member_stats(members: pd.DataFrame, stats: pd.DataFrame, row_ids: np.ndarray, prefix: str) -> pd.DataFrame:
    output = pd.DataFrame(index=row_ids)
    joined = members.merge(stats, on="key", how="left")
    joined = joined[joined["support"].notna()].copy()
    feature_values = ["success_rate", "mean_N", "significant_fraction", "reporting_delay", "reporting_rate", "dropout_ratio", "serious_event_ratio", "ci_present", "method_present", "ni_present"]
    if joined.empty:
        for col in feature_values:
            output[f"{prefix}_{col}_wmean"] = np.nan
        for col in ["success_rate_max", "success_rate_q25", "success_rate_q50", "success_rate_q75", "support_sum", "support_max", "effective_support"]:
            output[f"{prefix}_{col}"] = np.nan
        output[f"{prefix}_cold_start"] = 1.0
        return output
    grouped = joined.groupby("row_id", sort=False)
    support_sum = grouped["support"].sum()
    output[f"{prefix}_support_sum"] = support_sum.reindex(output.index)
    output[f"{prefix}_support_max"] = grouped["support"].max().reindex(output.index)
    joined["support_sq"] = joined["support"] ** 2
    output[f"{prefix}_effective_support"] = (support_sum ** 2 / grouped["support_sq"].sum().clip(lower=1e-9)).reindex(output.index)
    for col in feature_values:
        joined[f"weighted_{col}"] = joined[col] * joined["support"]
        output[f"{prefix}_{col}_wmean"] = (grouped[f"weighted_{col}"].sum() / support_sum).reindex(output.index)
    output[f"{prefix}_success_rate_max"] = grouped["success_rate"].max().reindex(output.index)
    quantiles = grouped["success_rate"].quantile([0.25, 0.5, 0.75]).unstack()
    for q, name in [(0.25, "q25"), (0.5, "q50"), (0.75, "q75")]:
        output[f"{prefix}_success_rate_{name}"] = quantiles.get(q, pd.Series(dtype=float)).reindex(output.index)
    output[f"{prefix}_cold_start"] = output[f"{prefix}_support_sum"].isna().astype(float)
    return output


def empirical_bayes_features(seeds: pd.DataFrame, episodes: pd.DataFrame, seed_maps: dict[str, pd.DataFrame], hist_maps: dict[str, pd.DataFrame], all_maps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    output = pd.DataFrame(index=seeds["row_id"])
    output.index.name = "row_id"
    seed_time = seeds.set_index("row_id")["timestamp"]
    for entity in ["lead_sponsor", "all_sponsor", "condition", "intervention", "facility", "country", "source", "source_class", "phase_purpose"]:
        members = seed_maps[entity].copy()
        members["timestamp"] = members["row_id"].map(seed_time)
        entity_frames = []
        for cutoff in sorted(seeds["timestamp"].unique()):
            cutoff = pd.Timestamp(cutoff)
            current = members[members["timestamp"].eq(cutoff)][["row_id", "key"]]
            cohort = seeds[seeds["timestamp"].eq(cutoff)]
            row_ids = cohort["row_id"].to_numpy()
            excluded_nct_ids = cohort["nct_id"].to_numpy()
            if len(row_ids) == 0:
                continue
            current_output = pd.DataFrame(index=row_ids)
            for view_name, view in VIEWS.items():
                stats = entity_view_stats(episodes, hist_maps[entity], all_maps[entity], cutoff, view, excluded_nct_ids)
                aggregate = aggregate_member_stats(current, stats, row_ids, f"hist_{entity}_{view_name}")
                current_output = current_output.join(aggregate, how="left")
            entity_frames.append(current_output)
        entity_output = pd.concat(entity_frames).sort_index()
        output = output.join(entity_output, how="left")
        print(f"[pipeline] empirical-Bayes entity complete: {entity} ({entity_output.shape[1]} features)", flush=True)
    return output.astype(float)


def build_document_rows(db, rows: pd.DataFrame) -> pd.DataFrame:
    studies = db.table_dict["studies"].df.copy()
    designs = latest_rows(db.table_dict["designs"].df.copy(), "nct_id", "date")
    eligibilities = latest_rows(db.table_dict["eligibilities"].df.copy(), "nct_id", "date")
    docs = rows.merge(studies, on="nct_id", how="left", validate="many_to_one")
    docs = docs.merge(designs, on="nct_id", how="left", suffixes=("", "_design"), validate="many_to_one")
    docs = docs.merge(eligibilities, on="nct_id", how="left", suffixes=("", "_elig"), validate="many_to_one")
    condition_rel = db.table_dict["conditions_studies"].df[["nct_id", "condition_id", "date"]]
    conditions = db.table_dict["conditions"].df
    condition_rows = rows[["doc_id", "nct_id", "cutoff"]].merge(condition_rel, on="nct_id", how="left")
    condition_rows = condition_rows[condition_rows["date"].le(condition_rows["cutoff"])].merge(conditions, on="condition_id", how="left")
    condition_text = condition_rows.groupby("doc_id")["mesh_term"].agg(lambda x: " ; ".join(sorted({text_value(v) for v in x if text_value(v)})))
    intervention_rel = db.table_dict["interventions_studies"].df[["nct_id", "intervention_id", "date"]]
    interventions = db.table_dict["interventions"].df
    intervention_rows = rows[["doc_id", "nct_id", "cutoff"]].merge(intervention_rel, on="nct_id", how="left")
    intervention_rows = intervention_rows[intervention_rows["date"].le(intervention_rows["cutoff"])].merge(interventions, on="intervention_id", how="left")
    intervention_text = intervention_rows.groupby("doc_id")["mesh_term"].agg(lambda x: " ; ".join(sorted({text_value(v) for v in x if text_value(v)})))
    docs["condition_terms"] = docs["doc_id"].map(condition_text).fillna("")
    docs["intervention_terms"] = docs["doc_id"].map(intervention_text).fillna("")
    docs["title_summary"] = docs.apply(lambda x: " [SEP] ".join(filter(None, [text_value(x.get("brief_title")), text_value(x.get("official_title")), text_value(x.get("brief_summaries")), text_value(x.get("detailed_descriptions"))])), axis=1)
    docs["eligibility"] = docs.apply(lambda x: " [SEP] ".join(filter(None, [text_value(x.get("criteria")), text_value(x.get("population")), text_value(x.get("minimum_age")), text_value(x.get("maximum_age")), text_value(x.get("gender"))])), axis=1)
    docs["entity_terms"] = docs.apply(lambda x: " [SEP] ".join(filter(None, [text_value(x.get("phase")), text_value(x.get("study_type")), text_value(x.get("allocation")), text_value(x.get("intervention_model")), text_value(x.get("observational_model")), text_value(x.get("primary_purpose")), text_value(x.get("masking")), text_value(x.get("source_class")), text_value(x.get("condition_terms")), text_value(x.get("intervention_terms"))])), axis=1)
    return docs[["doc_id", "nct_id", "cutoff", "title_summary", "eligibility", "entity_terms", "phase", "source_class"]]


def document_table(db, seeds: pd.DataFrame, episodes: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    seed_rows = seeds[["row_id", "nct_id", "timestamp"]].rename(columns={"timestamp": "cutoff"}).copy()
    seed_rows["doc_id"] = "seed_" + seed_rows["row_id"].astype(str)
    corpus_rows = episodes[["nct_id", "report_date"]].rename(columns={"report_date": "cutoff"}).copy()
    corpus_rows["doc_id"] = "corpus_" + corpus_rows["nct_id"].astype(str)
    rows = pd.concat([seed_rows[["doc_id", "nct_id", "cutoff"]], corpus_rows[["doc_id", "nct_id", "cutoff"]]], ignore_index=True)
    docs = build_document_rows(db, rows)
    seed_indices = np.arange(len(seed_rows), dtype=np.int64)
    corpus_indices = np.arange(len(seed_rows), len(rows), dtype=np.int64)
    return docs, seed_indices, corpus_indices


def content_digest(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8", errors="replace")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()[:20]


def encode_documents(docs: pd.DataFrame, shared_cache: Path, debug: bool) -> dict[str, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = SentenceTransformer(MODEL_NAME, device="cuda", trust_remote_code=True, revision=MODEL_REVISION)
    model.max_seq_length = 2048
    output = {}
    cache_root = shared_cache / "lane0_bioclinical_embeddings_v1"
    cache_root.mkdir(parents=True, exist_ok=True)
    for section in ["title_summary", "eligibility", "entity_terms"]:
        values = docs[section].fillna("").astype(str).tolist()
        digest = content_digest(docs["doc_id"].astype(str).tolist() + values)
        path = cache_root / f"{section}_{MODEL_REVISION[:12]}_{digest}.npy"
        if path.exists() and not debug:
            embeddings = np.load(path, mmap_mode=None)
            if embeddings.shape != (len(docs), 768):
                raise RuntimeError(f"embedding cache shape mismatch for {section}: {embeddings.shape}")
            print(f"[pipeline] loaded embedding cache {path.name}", flush=True)
        else:
            if debug:
                subset = min(384, len(values))
                embeddings = np.zeros((len(values), 768), dtype=np.float32)
                encoded = model.encode(values[:subset], batch_size=16, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
                embeddings[:subset] = encoded.astype(np.float32)
            else:
                encoded = model.encode(values, batch_size=24, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
                embeddings = encoded.astype(np.float32)
                temporary = path.with_suffix(".tmp.npy")
                np.save(temporary, embeddings)
                os.replace(temporary, path)
            completed = subset if debug else len(values)
            print(f"[pipeline] encoded {section}: {completed}/{len(values)} documents", flush=True)
        output[section] = embeddings
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return output


def semantic_features(seeds: pd.DataFrame, episodes: pd.DataFrame, docs: pd.DataFrame, embeddings: dict[str, np.ndarray], seed_indices: np.ndarray, corpus_indices: np.ndarray, debug: bool) -> tuple[pd.DataFrame, np.ndarray]:
    import torch

    combined = sum(embeddings[name] for name in ["title_summary", "eligibility", "entity_terms"])
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = np.divide(combined, norms, out=np.zeros_like(combined), where=norms > 0)
    seed_embedding = np.concatenate([embeddings[name][seed_indices] for name in ["title_summary", "eligibility", "entity_terms"]], axis=1)
    corpus_embedding = combined[corpus_indices]
    corpus_meta = docs.iloc[corpus_indices][["nct_id", "cutoff", "phase", "source_class"]].reset_index(drop=True)
    corpus_meta = corpus_meta.rename(columns={"cutoff": "report_date"})
    corpus_values = episodes.set_index("nct_id").reindex(corpus_meta["nct_id"])[["success", "N", "significant_fraction"]].reset_index(drop=True)
    seed_meta = docs.iloc[seed_indices][["nct_id", "phase", "source_class"]].reset_index(drop=True)
    names = []
    for k in [16, 64, 256]:
        names.extend([f"sem_k{k}_{x}" for x in ["success", "N", "significant_fraction", "sim_q25", "sim_q50", "sim_q75", "sim_max", "effective_neighbors", "phase_success", "phase_N", "phase_support", "source_success", "source_N", "source_support"]])
    section_names = ["title_summary"]
    for section in section_names:
        for k in [16, 64, 256]:
            names.extend([f"sem_{section}_k{k}_{x}" for x in ["success", "N", "significant_fraction", "sim_q25", "sim_q50", "sim_q75", "sim_max", "effective_neighbors"]])
    output = np.full((len(seeds), len(names)), np.nan, dtype=np.float32)
    device = torch.device("cuda")
    corpus_tensor = torch.as_tensor(corpus_embedding, device=device, dtype=torch.float32)
    for timestamp in sorted(seeds["timestamp"].unique()):
        rows = np.flatnonzero(seeds["timestamp"].to_numpy() == timestamp)
        eligible = np.flatnonzero(corpus_meta["report_date"].to_numpy() <= timestamp)
        if debug:
            eligible = eligible[: min(512, len(eligible))]
        if len(eligible) == 0:
            continue
        candidate_tensor = corpus_tensor[torch.as_tensor(eligible, device=device)]
        for start in range(0, len(rows), 512):
            batch_rows = rows[start:start + 512]
            query = torch.as_tensor(combined[seed_indices[batch_rows]], device=device, dtype=torch.float32)
            similarity = query @ candidate_tensor.T
            self_mask = seed_meta.iloc[batch_rows]["nct_id"].to_numpy()[:, None] == corpus_meta.iloc[eligible]["nct_id"].to_numpy()[None, :]
            if self_mask.any():
                similarity[torch.as_tensor(self_mask, device=device)] = -2.0
            top_n = min(256, similarity.shape[1])
            values, positions = torch.topk(similarity, k=top_n, dim=1)
            values = values.cpu().numpy()
            neighbor_indices = eligible[positions.cpu().numpy()]
            for local, row in enumerate(batch_rows):
                cursor = 0
                sims = values[local]
                indices = neighbor_indices[local]
                labels = corpus_values.iloc[indices]
                phase_match = corpus_meta.iloc[indices]["phase"].map(text_value).to_numpy() == text_value(seed_meta.iloc[row]["phase"])
                source_match = corpus_meta.iloc[indices]["source_class"].map(text_value).to_numpy() == text_value(seed_meta.iloc[row]["source_class"])
                for k in [16, 64, 256]:
                    take = min(k, len(indices))
                    sim = sims[:take]
                    weights = np.exp((sim - sim.max()) / 0.07)
                    weight_sum = weights.sum()
                    success = labels["success"].to_numpy(dtype=float)[:take]
                    counts = labels["N"].to_numpy(dtype=float)[:take]
                    fractions = labels["significant_fraction"].to_numpy(dtype=float)[:take]
                    base = [np.dot(weights, success) / weight_sum, np.dot(weights, counts) / weight_sum, np.dot(weights, fractions) / weight_sum, np.quantile(sim, 0.25), np.quantile(sim, 0.5), np.quantile(sim, 0.75), sim.max(), weight_sum ** 2 / np.square(weights).sum()]
                    matched = []
                    for mask in [phase_match[:take], source_match[:take]]:
                        if mask.any():
                            mw = weights[mask]
                            matched.extend([np.dot(mw, success[mask]) / mw.sum(), np.dot(mw, counts[mask]) / mw.sum(), mw.sum() ** 2 / np.square(mw).sum()])
                        else:
                            matched.extend([np.nan, np.nan, 0.0])
                    output[row, cursor:cursor + 14] = np.asarray(base + matched, dtype=np.float32)
                    cursor += 14
        print(f"[pipeline] semantic retrieval cohort {pd.Timestamp(timestamp).date()}: {len(rows)} queries, {len(eligible)} eligible", flush=True)
    section_offset = 42
    for section in section_names:
        section_corpus = torch.as_tensor(embeddings[section][corpus_indices], device=device, dtype=torch.float32)
        for timestamp in sorted(seeds["timestamp"].unique()):
            rows = np.flatnonzero(seeds["timestamp"].to_numpy() == timestamp)
            eligible = np.flatnonzero(corpus_meta["report_date"].to_numpy() <= timestamp)
            if debug:
                eligible = eligible[: min(512, len(eligible))]
            if len(eligible) == 0:
                continue
            candidate_tensor = section_corpus[torch.as_tensor(eligible, device=device)]
            for start in range(0, len(rows), 512):
                batch_rows = rows[start:start + 512]
                query = torch.as_tensor(embeddings[section][seed_indices[batch_rows]], device=device, dtype=torch.float32)
                similarity = query @ candidate_tensor.T
                self_mask = seed_meta.iloc[batch_rows]["nct_id"].to_numpy()[:, None] == corpus_meta.iloc[eligible]["nct_id"].to_numpy()[None, :]
                if self_mask.any():
                    similarity[torch.as_tensor(self_mask, device=device)] = -2.0
                top_n = min(256, similarity.shape[1])
                values, positions = torch.topk(similarity, k=top_n, dim=1)
                values = values.cpu().numpy()
                neighbor_indices = eligible[positions.cpu().numpy()]
                for local, row in enumerate(batch_rows):
                    cursor = section_offset
                    sims = values[local]
                    labels = corpus_values.iloc[neighbor_indices[local]]
                    for k in [16, 64, 256]:
                        take = min(k, len(sims))
                        sim = sims[:take]
                        weights = np.exp((sim - sim.max()) / 0.07)
                        weight_sum = weights.sum()
                        success = labels["success"].to_numpy(dtype=float)[:take]
                        counts = labels["N"].to_numpy(dtype=float)[:take]
                        fractions = labels["significant_fraction"].to_numpy(dtype=float)[:take]
                        output[row, cursor:cursor + 8] = np.asarray([np.dot(weights, success) / weight_sum, np.dot(weights, counts) / weight_sum, np.dot(weights, fractions) / weight_sum, np.quantile(sim, 0.25), np.quantile(sim, 0.5), np.quantile(sim, 0.75), sim.max(), weight_sum ** 2 / np.square(weights).sum()], dtype=np.float32)
                        cursor += 8
        section_offset += 24
        print(f"[pipeline] section-specific semantic retrieval complete: {section}", flush=True)
    frame = pd.DataFrame(output, index=seeds["row_id"], columns=names)
    frame.index.name = "row_id"
    return frame, seed_embedding.astype(np.float32)


def within_timestamp_features(seeds: pd.DataFrame, numeric: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    selected = [col for col in preferred if col in numeric][:30]
    output = pd.DataFrame(index=numeric.index)
    groups = seeds.set_index("row_id")["timestamp"].reindex(numeric.index)
    sizes = groups.map(groups.value_counts())
    output["within_small_cohort"] = (sizes < 20).astype(float)
    for col in selected:
        values = numeric[col]
        rank = values.groupby(groups).rank(method="average", pct=True)
        mean = values.groupby(groups).transform("mean")
        std = values.groupby(groups).transform("std").replace(0, np.nan)
        leader = values.groupby(groups).transform("max")
        output[f"within_{col}_rank"] = rank
        output[f"within_{col}_percentile"] = rank
        output[f"within_{col}_z"] = (values - mean) / std
        output[f"within_{col}_leader_gap"] = leader - values
        small = sizes < 20
        output.loc[small, [f"within_{col}_rank", f"within_{col}_percentile"]] = 0.5
        output.loc[small, [f"within_{col}_z", f"within_{col}_leader_gap"]] = 0.0
    return output.astype(float)


def true_analysis_counts(rows: pd.DataFrame, episodes: pd.DataFrame) -> np.ndarray:
    lookup = episodes.set_index("nct_id")
    joined = rows[["nct_id", "timestamp"]].join(lookup[["report_date", "N"]], on="nct_id")
    valid = joined["report_date"].gt(joined["timestamp"]) & joined["report_date"].le(joined["timestamp"] + pd.Timedelta(days=365))
    if not valid.all():
        raise RuntimeError(f"analysis-count reconstruction failed for {(~valid).sum()} authorized rows")
    return joined["N"].to_numpy(dtype=float)


def register_artifact(shared_cache: Path, name: str, path: Path, description: str, content_key: str, rebuild_hint: str) -> None:
    import fcntl

    registry = shared_cache / "artifacts.json"
    lock_path = shared_cache / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        records = json.loads(registry.read_text()) if registry.exists() else []
        if not any(x.get("name") == name and x.get("content_key") == content_key for x in records):
            records.append({"name": name, "path": str(path.relative_to(shared_cache)), "description": description, "content_key": content_key, "rebuild_hint": rebuild_hint})
            temporary = registry.with_suffix(".tmp")
            temporary.write_text(json.dumps(records, indent=2) + "\n")
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
