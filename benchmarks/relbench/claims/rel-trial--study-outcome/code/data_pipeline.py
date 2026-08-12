from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def _safe_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _latest_asof(seeds: pd.DataFrame, child: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in child.columns]
    merged = seeds[["_row_id", "nct_id", "timestamp"]].merge(
        child[["nct_id", "date", *available]], on="nct_id", how="left"
    )
    merged = merged[merged["date"].isna() | merged["date"].le(merged["timestamp"])]
    merged = merged.sort_values(["_row_id", "date"], na_position="first")
    latest = merged.groupby("_row_id", sort=False).tail(1).set_index("_row_id")
    latest = latest.reindex(seeds["_row_id"])
    return latest


def _relation_values(
    seeds: pd.DataFrame,
    relation: pd.DataFrame,
    dimension: pd.DataFrame,
    key: str,
    value: str,
) -> tuple[dict[int, list], dict[int, list[str]]]:
    pairs = seeds[["_row_id", "nct_id", "timestamp"]].merge(
        relation[["nct_id", key, "date"]], on="nct_id", how="left"
    )
    pairs = pairs[pairs["date"].le(pairs["timestamp"])]
    pairs = pairs.merge(dimension[[key, value]], on=key, how="left")
    ids = pairs.groupby("_row_id")[key].apply(
        lambda series: sorted(set(series.dropna().tolist()))
    ).to_dict()
    values = pairs.groupby("_row_id")[value].apply(
        lambda series: sorted({_safe_text(item) for item in series if _safe_text(item)})
    ).to_dict()
    return ids, values


def _sponsor_values(
    seeds: pd.DataFrame,
    relation: pd.DataFrame,
    sponsors: pd.DataFrame,
) -> tuple[dict[int, list], dict[int, list], dict[int, list[str]]]:
    columns = ["nct_id", "sponsor_id", "lead_or_collaborator", "date"]
    pairs = seeds[["_row_id", "nct_id", "timestamp"]].merge(
        relation[columns], on="nct_id", how="left"
    )
    pairs = pairs[pairs["date"].le(pairs["timestamp"])]
    pairs = pairs.merge(sponsors[["sponsor_id", "agency_class"]], on="sponsor_id", how="left")
    all_ids = pairs.groupby("_row_id")["sponsor_id"].apply(
        lambda series: sorted(set(series.dropna().tolist()))
    ).to_dict()
    lead_pairs = pairs[
        pairs["lead_or_collaborator"].fillna("").astype(str).str.lower().str.contains("lead")
    ]
    lead_ids = lead_pairs.groupby("_row_id")["sponsor_id"].apply(
        lambda series: sorted(set(series.dropna().tolist()))
    ).to_dict()
    classes = pairs.groupby("_row_id")["agency_class"].apply(
        lambda series: sorted({_safe_text(item) for item in series if _safe_text(item)})
    ).to_dict()
    return all_ids, lead_ids, classes


def _relation_counts(seeds: pd.DataFrame, relation: pd.DataFrame) -> dict[int, int]:
    pairs = seeds[["_row_id", "nct_id", "timestamp"]].merge(
        relation[["nct_id", "date"]], on="nct_id", how="left"
    )
    pairs = pairs[pairs["date"].le(pairs["timestamp"])]
    return pairs.groupby("_row_id").size().astype(int).to_dict()


def _relation_ids(
    seeds: pd.DataFrame,
    relation: pd.DataFrame,
    key: str,
) -> dict[int, list]:
    pairs = seeds[["_row_id", "nct_id", "timestamp"]].merge(
        relation[["nct_id", key, "date"]], on="nct_id", how="left"
    )
    pairs = pairs[pairs["date"].le(pairs["timestamp"])]
    return pairs.groupby("_row_id")[key].apply(
        lambda series: sorted(set(series.dropna().tolist()))
    ).to_dict()


def _intervention_class(term: str) -> str:
    value = term.lower()
    groups = [
        ("vaccine_or_immunization", ("vaccine", "immunization", "toxoid")),
        ("cell_gene_or_biologic", ("antibod", "gene", "cell", "biologic", "immunoglobulin")),
        ("device_or_implant", ("device", "implant", "prosthe", "stent", "catheter")),
        ("procedure_or_surgery", ("surg", "procedure", "ablation", "transplant")),
        ("behavioral", ("behavior", "psychotherap", "exercise", "education", "counsel")),
        ("diagnostic_or_imaging", ("diagnos", "imaging", "tomography", "ultrasound", "magnetic resonance")),
        ("nutrition_or_supplement", ("diet", "nutrition", "vitamin", "supplement", "mineral")),
        ("radiation", ("radiation", "radiotherap")),
    ]
    for label, needles in groups:
        if any(needle in value for needle in needles):
            return label
    return "pharmacologic_or_other"


def _phase_number(value: str) -> float:
    text = value.lower()
    if "early phase 1" in text:
        return 0.5
    if "phase 1/phase 2" in text:
        return 1.5
    if "phase 2/phase 3" in text:
        return 2.5
    if "phase 1" in text:
        return 1.0
    if "phase 2" in text:
        return 2.0
    if "phase 3" in text:
        return 3.0
    if "phase 4" in text:
        return 4.0
    if "not applicable" in text:
        return 0.0
    return -1.0


def assemble_protocols(db, seeds: pd.DataFrame, split: str) -> pd.DataFrame:
    seeds = seeds.copy().reset_index(drop=True)
    seeds["_row_id"] = np.arange(len(seeds), dtype=np.int64)
    seeds["timestamp"] = pd.to_datetime(seeds["timestamp"])
    seeds["_key"] = [
        f"{split}:{row}:{int(entity)}:{pd.Timestamp(timestamp).isoformat()}"
        for row, entity, timestamp in zip(seeds["_row_id"], seeds["nct_id"], seeds["timestamp"])
    ]
    studies = db.table_dict["studies"].df
    designs = db.table_dict["designs"].df
    eligibilities = db.table_dict["eligibilities"].df
    merged = seeds.merge(studies, on="nct_id", how="left", validate="many_to_one")
    if merged["start_date"].gt(merged["timestamp"]).any():
        raise RuntimeError("study attributes precede neither seed nor origin")
    design_columns = [
        "allocation",
        "intervention_model",
        "observational_model",
        "primary_purpose",
        "time_perspective",
        "masking",
        "subject_masked",
        "caregiver_masked",
        "investigator_masked",
        "outcomes_assessor_masked",
    ]
    eligibility_columns = [
        "sampling_method",
        "gender",
        "minimum_age",
        "maximum_age",
        "healthy_volunteers",
        "criteria",
        "adult",
        "child",
        "older_adult",
    ]
    design = _latest_asof(seeds, designs, design_columns)
    eligibility = _latest_asof(seeds, eligibilities, eligibility_columns)
    for column in design_columns:
        merged[column] = design[column].to_numpy() if column in design else np.nan
    for column in eligibility_columns:
        merged[column] = eligibility[column].to_numpy() if column in eligibility else np.nan
    condition_ids, condition_terms = _relation_values(
        seeds,
        db.table_dict["conditions_studies"].df,
        db.table_dict["conditions"].df,
        "condition_id",
        "mesh_term",
    )
    intervention_ids, intervention_terms = _relation_values(
        seeds,
        db.table_dict["interventions_studies"].df,
        db.table_dict["interventions"].df,
        "intervention_id",
        "mesh_term",
    )
    sponsor_ids, lead_sponsor_ids, sponsor_classes = _sponsor_values(
        seeds,
        db.table_dict["sponsors_studies"].df,
        db.table_dict["sponsors"].df,
    )
    facility_ids = _relation_ids(
        seeds, db.table_dict["facilities_studies"].df, "facility_id"
    )
    rows = []
    for record in merged.to_dict("records"):
        row_id = int(record["_row_id"])
        conditions = condition_terms.get(row_id, [])
        intervention_classes = sorted(
            {_intervention_class(term) for term in intervention_terms.get(row_id, [])}
        )
        agencies = sponsor_classes.get(row_id, [])
        phase = _safe_text(record.get("phase")) or "Missing"
        study_type = _safe_text(record.get("study_type")) or "Missing"
        design_parts = [
            f"Phase: {phase}",
            f"Study type: {study_type}",
            f"Enrollment: {_safe_text(record.get('enrollment')) or 'Missing'}",
            f"Arms: {_safe_text(record.get('number_of_arms')) or 'Missing'}",
            f"Groups: {_safe_text(record.get('number_of_groups')) or 'Missing'}",
            f"Allocation: {_safe_text(record.get('allocation')) or 'Missing'}",
            f"Intervention model: {_safe_text(record.get('intervention_model')) or 'Missing'}",
            f"Masking: {_safe_text(record.get('masking')) or 'Missing'}",
            f"Primary purpose: {_safe_text(record.get('primary_purpose')) or 'Missing'}",
            f"Gender: {_safe_text(record.get('gender')) or 'Missing'}",
            f"Age range: {_safe_text(record.get('minimum_age')) or 'Missing'} to {_safe_text(record.get('maximum_age')) or 'Missing'}",
            f"Sponsor agency class: {', '.join(agencies) if agencies else 'Missing'}",
            f"Conditions: {', '.join(conditions[:20]) if conditions else 'Unspecified'}",
            f"Intervention classes: {', '.join(intervention_classes) if intervention_classes else 'Unspecified'}",
        ]
        title = " ".join(
            value
            for value in [
                _safe_text(record.get("brief_title")),
                _safe_text(record.get("official_title")),
            ]
            if value
        )
        title_design = f"Title: {title}. " + ". ".join(design_parts)
        criteria = _safe_text(record.get("criteria"))
        summary = _safe_text(record.get("brief_summaries"))
        allocation = _safe_text(record.get("allocation")).lower()
        masking = _safe_text(record.get("masking")).lower()
        intervention_model = _safe_text(record.get("intervention_model")).lower()
        primary_purpose = _safe_text(record.get("primary_purpose")).lower()
        gender = _safe_text(record.get("gender")).lower()
        healthy_volunteers = _safe_text(record.get("healthy_volunteers")).lower()
        agency_lower = " ".join(agencies).lower()
        masked_roles = sum(
            _safe_text(record.get(column)).lower() in {"true", "yes", "1"}
            for column in [
                "subject_masked",
                "caregiver_masked",
                "investigator_masked",
                "outcomes_assessor_masked",
            ]
        )
        groups = record.get("number_of_groups")
        item = {
            "_key": record["_key"],
            "_row_id": row_id,
            "split": split,
            "nct_id": int(record["nct_id"]),
            "timestamp": pd.Timestamp(record["timestamp"]),
            "outcome": float(record["outcome"]) if "outcome" in record and not pd.isna(record["outcome"]) else np.nan,
            "phase": phase,
            "title_design": title_design,
            "summary": summary,
            "eligibility": criteria,
            "conditions_text": " ".join(conditions),
            "phase_number": _phase_number(phase),
            "is_phase_2": float("phase 2" in phase.lower() and "phase 3" not in phase.lower()),
            "is_phase_3": float("phase 3" in phase.lower()),
            "is_phase_4": float("phase 4" in phase.lower()),
            "is_phase_na": float("not applicable" in phase.lower()),
            "is_interventional": float("interventional" in study_type.lower()),
            "log_enrollment": math.log1p(max(0.0, float(record.get("enrollment") or 0.0))) if not pd.isna(record.get("enrollment")) else 0.0,
            "log_arms": math.log1p(max(0.0, float(record.get("number_of_arms") or 0.0))) if not pd.isna(record.get("number_of_arms")) else 0.0,
            "log_groups": math.log1p(max(0.0, float(groups or 0.0))) if not pd.isna(groups) else 0.0,
            "trial_age_years": max(0.0, (pd.Timestamp(record["timestamp"]) - pd.Timestamp(record["start_date"])).days / 365.25),
            "is_randomized": float("random" in allocation and "non-random" not in allocation and "nonrandom" not in allocation),
            "is_blinded": float(bool(masking) and "none" not in masking and "open" not in masking),
            "masked_role_count": float(masked_roles),
            "is_parallel_design": float("parallel" in intervention_model),
            "is_crossover_design": float("crossover" in intervention_model),
            "is_single_group_design": float("single group" in intervention_model),
            "is_factorial_design": float("factorial" in intervention_model),
            "purpose_treatment": float("treatment" in primary_purpose),
            "purpose_prevention": float("prevention" in primary_purpose),
            "purpose_diagnostic": float("diagnostic" in primary_purpose),
            "gender_all": float("all" in gender or not gender),
            "gender_female": float("female" in gender and "all" not in gender),
            "gender_male": float("male" in gender and "female" not in gender and "all" not in gender),
            "healthy_volunteers_allowed": float("accept" in healthy_volunteers or healthy_volunteers in {"yes", "true", "1"}),
            "adult_allowed": float(_safe_text(record.get("adult")).lower() in {"true", "yes", "1"}),
            "child_allowed": float(_safe_text(record.get("child")).lower() in {"true", "yes", "1"}),
            "older_adult_allowed": float(_safe_text(record.get("older_adult")).lower() in {"true", "yes", "1"}),
            "eligibility_log_words": math.log1p(len(criteria.split())),
            "eligibility_truncated": float(len(criteria.split()) > 508),
            "summary_log_words": math.log1p(len(summary.split())),
            "title_log_words": math.log1p(len(title.split())),
            "condition_count": float(len(condition_ids.get(row_id, []))),
            "intervention_count": float(len(intervention_ids.get(row_id, []))),
            "facility_count": float(len(facility_ids.get(row_id, []))),
            "sponsor_industry": float("industry" in agency_lower),
            "sponsor_government": float(any(term in agency_lower for term in ["nih", "federal", "network"])),
            "_condition_ids": condition_ids.get(row_id, []),
            "_intervention_ids": intervention_ids.get(row_id, []),
            "_sponsor_ids": sponsor_ids.get(row_id, []),
            "_lead_sponsor_ids": lead_sponsor_ids.get(row_id) or sponsor_ids.get(row_id, []),
            "_facility_ids": facility_ids.get(row_id, []),
            "K": float(record["K"]) if "K" in record and not pd.isna(record["K"]) else np.nan,
            "S": float(record["S"]) if "S" in record and not pd.isna(record["S"]) else np.nan,
        }
        rows.append(item)
    output = pd.DataFrame(rows).sort_values("_row_id").reset_index(drop=True)
    if output["_row_id"].tolist() != list(range(len(output))):
        raise RuntimeError("protocol assembly changed task order")
    return output


def qualifying_results(db) -> pd.DataFrame:
    analyses = db.table_dict["outcome_analyses"].df[
        [
            "nct_id",
            "outcome_id",
            "p_value_modifier",
            "p_value",
            "non_inferiority_type",
            "method",
            "date",
        ]
    ].copy()
    outcomes = db.table_dict["outcomes"].df[["id", "outcome_type"]].copy()
    rows = analyses.merge(outcomes, left_on="outcome_id", right_on="id", how="inner")
    modifier = rows["p_value_modifier"]
    rows = rows[
        (modifier.isna() | modifier.ne(">"))
        & rows["p_value"].between(0.0, 1.0, inclusive="both")
        & rows["outcome_type"].eq("Primary")
    ]
    method = rows["method"].fillna("").astype(str).str.lower()
    noninferiority = rows["non_inferiority_type"].fillna("").astype(str).str.lower()
    rows["significant_analysis"] = rows["p_value"].le(0.05).astype(float)
    rows["method_known"] = method.ne("").astype(float)
    rows["noninferiority_analysis"] = noninferiority.str.contains(
        "non-inferiority|equivalence", regex=True
    ).astype(float)
    rows["parametric_analysis"] = method.str.contains(
        "anova|ancova|t-test|regression|mixed|mmrm", regex=True
    ).astype(float)
    rows["survival_analysis"] = method.str.contains(
        "log rank|cox|survival|hazard", regex=True
    ).astype(float)
    rows["nonparametric_analysis"] = method.str.contains(
        "wilcoxon|fisher|chi-squared|kruskal|mantel|exact", regex=True
    ).astype(float)
    result = rows.groupby("nct_id", as_index=False).agg(
        result_date=("date", "max"),
        minimum_p=("p_value", "min"),
        primary_analysis_count=("p_value", "size"),
        significant_analysis_count=("significant_analysis", "sum"),
        method_known_count=("method_known", "sum"),
        noninferiority_analysis_count=("noninferiority_analysis", "sum"),
        parametric_analysis_count=("parametric_analysis", "sum"),
        survival_analysis_count=("survival_analysis", "sum"),
        nonparametric_analysis_count=("nonparametric_analysis", "sum"),
    )
    result["historical_outcome"] = result["minimum_p"].le(0.05).astype(float)
    result["multi_analysis_trial"] = result["primary_analysis_count"].gt(1).astype(float)
    result["analysis_count_square"] = np.square(result["primary_analysis_count"].astype(float))
    return result


def exact_window_targets(db, seeds: pd.DataFrame) -> pd.DataFrame:
    seed_rows = seeds[["timestamp", "nct_id"]].copy().reset_index(drop=True)
    seed_rows["_target_row"] = np.arange(len(seed_rows), dtype=np.int64)
    studies = db.table_dict["studies"].df[["nct_id", "start_date"]]
    outcomes = db.table_dict["outcomes"].df[["id", "nct_id", "outcome_type"]]
    analyses = db.table_dict["outcome_analyses"].df[
        ["nct_id", "outcome_id", "p_value_modifier", "p_value", "date"]
    ]
    connection = duckdb.connect()
    connection.register("seed_rows", seed_rows)
    connection.register("studies", studies)
    connection.register("outcomes", outcomes)
    connection.register("outcome_analyses", analyses)
    query = """
        WITH TRIAL_INFO AS (
            SELECT oa.nct_id, oa.p_value, s.start_date, oa.date
            FROM outcome_analyses oa
            LEFT JOIN outcomes o ON oa.outcome_id = o.id
            LEFT JOIN studies s ON s.nct_id = o.nct_id
            WHERE (oa.p_value_modifier IS NULL OR oa.p_value_modifier != '>')
              AND oa.p_value >= 0
              AND oa.p_value <= 1
              AND o.outcome_type = 'Primary'
        )
        SELECT t._target_row,
               COUNT(*) AS K,
               SUM(CASE WHEN tr.p_value <= 0.05 THEN 1 ELSE 0 END) AS S
        FROM seed_rows t
        LEFT JOIN TRIAL_INFO tr
          ON tr.nct_id = t.nct_id
         AND tr.start_date <= t.timestamp
         AND tr.date > t.timestamp
         AND tr.date <= t.timestamp + INTERVAL '365 days'
        WHERE tr.nct_id IS NOT NULL
        GROUP BY t._target_row
    """
    targets = connection.execute(query).df()
    connection.close()
    output = seed_rows.merge(targets, on="_target_row", how="left")
    if output[["K", "S"]].isna().any().any():
        raise RuntimeError("exact target query did not resolve every supplied seed")
    output["K"] = output["K"].astype(float)
    output["S"] = output["S"].astype(float)
    output["outcome"] = output["S"].gt(0).astype(float)
    return output.drop(columns="_target_row")


def make_replay_seeds(
    db,
    cutoff: pd.Timestamp,
    excluded_entities: set[int],
    reference_seeds: pd.DataFrame,
) -> pd.DataFrame:
    end = pd.Timestamp(cutoff) - pd.Timedelta(days=365)
    timestamps = pd.DataFrame(
        {"timestamp": pd.date_range("2000-01-01", end, freq="MS")}
    )
    studies = db.table_dict["studies"].df[["nct_id", "start_date"]]
    outcomes = db.table_dict["outcomes"].df[["id", "nct_id", "outcome_type"]]
    outcome_analyses = db.table_dict["outcome_analyses"].df[
        ["nct_id", "outcome_id", "p_value_modifier", "p_value", "date"]
    ]
    connection = duckdb.connect()
    connection.register("timestamp_df", timestamps)
    connection.register("studies", studies)
    connection.register("outcomes", outcomes)
    connection.register("outcome_analyses", outcome_analyses)
    query = """
        WITH TRIAL_INFO AS (
            SELECT oa.nct_id, oa.p_value, s.start_date, oa.date
            FROM outcome_analyses oa
            LEFT JOIN outcomes o ON oa.outcome_id = o.id
            LEFT JOIN studies s ON s.nct_id = o.nct_id
            WHERE (oa.p_value_modifier IS NULL OR oa.p_value_modifier != '>')
              AND oa.p_value >= 0
              AND oa.p_value <= 1
              AND o.outcome_type = 'Primary'
        )
        SELECT t.timestamp, tr.nct_id,
               COUNT(*) AS K,
               SUM(CASE WHEN tr.p_value <= 0.05 THEN 1 ELSE 0 END) AS S,
               CASE WHEN MIN(tr.p_value) <= 0.05 THEN 1 ELSE 0 END AS outcome
        FROM timestamp_df t
        LEFT JOIN TRIAL_INFO tr
          ON tr.start_date <= t.timestamp
         AND tr.date > t.timestamp
         AND tr.date <= t.timestamp + INTERVAL '365 days'
        WHERE tr.nct_id IS NOT NULL
        GROUP BY t.timestamp, tr.nct_id
    """
    replay = connection.execute(query).df()
    connection.close()
    replay = replay[~replay["nct_id"].isin(excluded_entities)].copy()
    results = qualifying_results(db)[["nct_id", "result_date"]]
    replay = replay.merge(results, on="nct_id", how="left")
    reference = reference_seeds.merge(results, on="nct_id", how="left")
    horizon = (reference["result_date"] - reference["timestamp"]).dt.days
    horizon = horizon[horizon.between(1, 365)]
    target_horizon = float(horizon.median()) if len(horizon) else 182.0
    replay["horizon_distance"] = (
        (replay["result_date"] - replay["timestamp"]).dt.days - target_horizon
    ).abs()
    replay = replay.sort_values(["nct_id", "horizon_distance", "timestamp"])
    replay = replay.groupby("nct_id", as_index=False).head(1)
    replay = replay[["timestamp", "nct_id", "outcome", "K", "S"]].sort_values(
        ["timestamp", "nct_id"]
    ).reset_index(drop=True)
    if replay["nct_id"].duplicated().any():
        raise RuntimeError("replay contains duplicate studies")
    if (replay["timestamp"] + pd.Timedelta(days=365) > pd.Timestamp(cutoff)).any():
        raise RuntimeError("replay contains an incomplete future window")
    return replay


def _history_asof(
    frame: pd.DataFrame,
    relation: pd.DataFrame,
    key_column: str,
    list_column: str,
    results: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    current = frame[["_key", "timestamp", list_column]].copy()
    target = current.explode(list_column).rename(columns={list_column: key_column})
    target[key_column] = pd.to_numeric(target[key_column], errors="coerce")
    target = target[target[key_column].notna()].copy()
    target[key_column] = target[key_column].astype(np.int64)
    visible = relation[["nct_id", key_column, "date"]].merge(
        results, on="nct_id", how="inner"
    )
    visible[key_column] = pd.to_numeric(visible[key_column], errors="coerce")
    visible = visible[visible[key_column].notna()].copy()
    visible[key_column] = visible[key_column].astype(np.int64)
    visible["availability"] = visible[["date", "result_date"]].max(axis=1)
    visible = visible.sort_values("availability").drop_duplicates(
        [key_column, "nct_id"], keep="first"
    )
    value_columns = {
        "history_count": None,
        "history_success": "historical_outcome",
        "history_analyses": "primary_analysis_count",
        "history_significant": "significant_analysis_count",
        "history_method_known": "method_known_count",
        "history_multi": "multi_analysis_trial",
        "history_analysis_square": "analysis_count_square",
        "history_noninferiority": "noninferiority_analysis_count",
        "history_parametric": "parametric_analysis_count",
        "history_survival": "survival_analysis_count",
        "history_nonparametric": "nonparametric_analysis_count",
    }
    visible["history_count"] = 1.0
    for output_column, source_column in value_columns.items():
        if source_column is not None:
            visible[output_column] = visible[source_column].astype(float)
    cumulative_columns = list(value_columns)
    visible = visible.sort_values([key_column, "availability"])
    visible[cumulative_columns] = visible.groupby(key_column)[
        cumulative_columns
    ].cumsum()
    visible["history_max_analyses"] = visible.groupby(key_column)[
        "primary_analysis_count"
    ].cummax()
    if len(target) and len(visible):
        joined = pd.merge_asof(
            target.sort_values(["timestamp", key_column]),
            visible[
                [
                    key_column,
                    "availability",
                    *cumulative_columns,
                    "history_max_analyses",
                ]
            ].sort_values(
                ["availability", key_column]
            ),
            left_on="timestamp",
            right_on="availability",
            by=key_column,
            direction="backward",
            allow_exact_matches=True,
        )
        summed = joined.groupby("_key", as_index=False)[cumulative_columns].sum()
        maximum = joined.groupby("_key", as_index=False)["history_max_analyses"].max()
        summed = summed.merge(maximum, on="_key", how="left")
    else:
        summed = pd.DataFrame(
            columns=["_key", *cumulative_columns, "history_max_analyses"]
        )
    output = current[["_key", "timestamp"]].merge(summed, on="_key", how="left")
    output[[*cumulative_columns, "history_max_analyses"]] = output[
        [*cumulative_columns, "history_max_analyses"]
    ].fillna(0.0)
    global_rows = []
    for timestamp in output["timestamp"].drop_duplicates().sort_values():
        past = results[results["result_date"].le(timestamp)]
        count = float(len(past))
        analyses = float(past["primary_analysis_count"].sum())
        global_rows.append(
            {
                "timestamp": timestamp,
                "global_history_rate": float(past["historical_outcome"].mean()) if count else 0.5,
                "global_history_count": count,
                "global_expected_analyses": analyses / count if count else 1.0,
                "global_multi_analysis_rate": float(past["multi_analysis_trial"].mean()) if count else 0.0,
                "global_analysis_q": float(past["significant_analysis_count"].sum()) / analyses if analyses else 0.5,
                "global_noninferiority_rate": float(past["noninferiority_analysis_count"].sum()) / analyses if analyses else 0.0,
                "global_parametric_rate": float(past["parametric_analysis_count"].sum()) / analyses if analyses else 0.0,
                "global_survival_rate": float(past["survival_analysis_count"].sum()) / analyses if analyses else 0.0,
                "global_nonparametric_rate": float(past["nonparametric_analysis_count"].sum()) / analyses if analyses else 0.0,
            }
        )
    output = output.merge(pd.DataFrame(global_rows), on="timestamp", how="left")
    trial_alpha = 20.0
    analysis_alpha = 50.0
    output[f"{prefix}_hist_log_count"] = np.log1p(output["history_count"])
    output[f"{prefix}_hist_eb_rate"] = (
        output["history_success"] + trial_alpha * output["global_history_rate"]
    ) / (output["history_count"] + trial_alpha)
    output[f"{prefix}_expected_analyses"] = (
        output["history_analyses"] + trial_alpha * output["global_expected_analyses"]
    ) / (output["history_count"] + trial_alpha)
    history_mean = output["history_analyses"] / output["history_count"].clip(lower=1.0)
    history_second = output["history_analysis_square"] / output["history_count"].clip(lower=1.0)
    output[f"{prefix}_analysis_variance"] = np.maximum(
        0.0, history_second - np.square(history_mean)
    )
    output[f"{prefix}_max_analyses_log"] = np.log1p(
        output["history_max_analyses"]
    )
    output[f"{prefix}_multi_analysis_rate"] = (
        output["history_multi"] + trial_alpha * output["global_multi_analysis_rate"]
    ) / (output["history_count"] + trial_alpha)
    output[f"{prefix}_hist_analysis_q"] = (
        output["history_significant"] + analysis_alpha * output["global_analysis_q"]
    ) / (output["history_analyses"] + analysis_alpha)
    for label, raw, global_column in [
        ("noninferiority", "history_noninferiority", "global_noninferiority_rate"),
        ("parametric", "history_parametric", "global_parametric_rate"),
        ("survival", "history_survival", "global_survival_rate"),
        ("nonparametric", "history_nonparametric", "global_nonparametric_rate"),
    ]:
        output[f"{prefix}_hist_{label}_rate"] = (
            output[raw] + analysis_alpha * output[global_column]
        ) / (output["history_analyses"] + analysis_alpha)
    output[f"{prefix}_hist_method_coverage"] = (
        output["history_method_known"] / output["history_analyses"].clip(lower=1.0)
    )
    output[f"{prefix}_history_count_raw"] = output["history_count"]
    return output.drop(
        columns=["timestamp", "history_max_analyses", *cumulative_columns]
    )


def _pair_history_inputs(
    frame: pd.DataFrame,
    left_relation: pd.DataFrame,
    left_key: str,
    left_list: str,
    right_relation: pd.DataFrame,
    right_key: str,
    right_list: str,
) -> tuple[pd.DataFrame, str, str, pd.DataFrame]:
    left = left_relation[["nct_id", left_key, "date"]].copy()
    right = right_relation[["nct_id", right_key, "date"]].copy()
    pairs = left.merge(right, on="nct_id", how="inner", suffixes=("_left", "_right"))
    pairs["date"] = pairs[["date_left", "date_right"]].max(axis=1)
    tuples = list(zip(pairs[left_key].astype(int), pairs[right_key].astype(int)))
    unique = {value: index for index, value in enumerate(sorted(set(tuples)))}
    pairs["pair_id"] = [unique[value] for value in tuples]
    pair_relation = pairs[["nct_id", "pair_id", "date"]].drop_duplicates()
    list_column = f"_{left_key}_{right_key}_pair_ids"
    enriched = frame.copy()
    enriched[list_column] = [
        sorted(
            {
                unique[(int(left_value), int(right_value))]
                for left_value in left_values
                for right_value in right_values
                if (int(left_value), int(right_value)) in unique
            }
        )
        for left_values, right_values in zip(enriched[left_list], enriched[right_list])
    ]
    return enriched, "pair_id", list_column, pair_relation


def build_compact_features(db, frame: pd.DataFrame) -> pd.DataFrame:
    results = qualifying_results(db)
    lead_sponsors = db.table_dict["sponsors_studies"].df.copy()
    lead_mask = lead_sponsors["lead_or_collaborator"].fillna("").astype(str).str.lower().str.contains("lead")
    lead_sponsors = lead_sponsors[lead_mask]
    sponsor_condition_frame, pair_key, pair_values, sponsor_condition_relation = _pair_history_inputs(
        frame,
        lead_sponsors,
        "sponsor_id",
        "_lead_sponsor_ids",
        db.table_dict["conditions_studies"].df,
        "condition_id",
        "_condition_ids",
    )
    relations = [
        (
            db.table_dict["sponsors_studies"].df,
            "sponsor_id",
            "_lead_sponsor_ids",
            "sponsor",
        ),
        (
            db.table_dict["conditions_studies"].df,
            "condition_id",
            "_condition_ids",
            "condition",
        ),
        (
            db.table_dict["interventions_studies"].df,
            "intervention_id",
            "_intervention_ids",
            "intervention",
        ),
        (
            db.table_dict["facilities_studies"].df,
            "facility_id",
            "_facility_ids",
            "facility",
        ),
    ]
    history_blocks = [
        _history_asof(frame, relation, key, values, results, prefix)
        for relation, key, values, prefix in relations
    ]
    history_blocks.append(
        _history_asof(
            sponsor_condition_frame,
            sponsor_condition_relation,
            pair_key,
            pair_values,
            results,
            "sponsor_condition",
        )
    )
    phase_frame = frame.copy()
    phase_frame["_phase_ids"] = phase_frame["phase_number"].map(
        lambda value: [int(round((float(value) + 1.0) * 2.0))]
    )
    phase_relation = db.table_dict["studies"].df[["nct_id", "start_date", "phase"]].copy()
    phase_relation["phase_id"] = phase_relation["phase"].map(_safe_text).map(_phase_number).map(
        lambda value: int(round((float(value) + 1.0) * 2.0))
    )
    phase_relation = phase_relation.rename(columns={"start_date": "date"})
    history_blocks.append(
        _history_asof(
            phase_frame,
            phase_relation,
            "phase_id",
            "_phase_ids",
            results,
            "phase",
        )
    )
    history = history_blocks[0]
    global_columns = [column for column in history if column.startswith("global_")]
    for block in history_blocks[1:]:
        history = history.merge(
            block.drop(columns=global_columns), on="_key", how="outer"
        )
    base_columns = [
        "_key",
        "timestamp",
        "phase_number",
        "is_phase_2",
        "is_phase_3",
        "is_phase_4",
        "is_phase_na",
        "is_interventional",
        "log_enrollment",
        "log_arms",
        "log_groups",
        "trial_age_years",
        "is_randomized",
        "is_blinded",
        "masked_role_count",
        "is_parallel_design",
        "is_crossover_design",
        "is_single_group_design",
        "is_factorial_design",
        "purpose_treatment",
        "purpose_prevention",
        "purpose_diagnostic",
        "gender_all",
        "gender_female",
        "gender_male",
        "healthy_volunteers_allowed",
        "adult_allowed",
        "child_allowed",
        "older_adult_allowed",
        "eligibility_log_words",
        "eligibility_truncated",
        "summary_log_words",
        "title_log_words",
        "condition_count",
        "intervention_count",
        "facility_count",
        "sponsor_industry",
        "sponsor_government",
    ]
    output = frame[base_columns].merge(history, on="_key", how="left")
    output["expected_primary_analysis_count"] = output[
        [
            "sponsor_expected_analyses",
            "condition_expected_analyses",
            "intervention_expected_analyses",
            "facility_expected_analyses",
            "phase_expected_analyses",
        ]
    ].mean(axis=1)
    rank_columns = {
        "log_enrollment": "enrollment",
        "expected_primary_analysis_count": "expected_analyses",
        "sponsor_hist_eb_rate": "sponsor_rate",
        "condition_hist_eb_rate": "condition_rate",
        "intervention_hist_eb_rate": "intervention_rate",
        "facility_hist_eb_rate": "facility_rate",
        "sponsor_hist_analysis_q": "sponsor_analysis_q",
        "condition_hist_analysis_q": "condition_analysis_q",
    }
    for column, stem in rank_columns.items():
        grouped = output.groupby("timestamp")[column]
        output[f"{stem}_origin_percentile"] = grouped.rank(pct=True)
        deviation = output[column] - grouped.transform("mean")
        output[f"{stem}_origin_z"] = deviation / grouped.transform("std").replace(0.0, np.nan)
        output[f"{stem}_origin_gap_to_leader"] = grouped.transform("max") - output[column]
    numeric = output.select_dtypes(include=[np.number]).columns
    output[numeric] = output[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return output


def protocol_digest(frame: pd.DataFrame, version: str) -> str:
    digest = hashlib.sha256(version.encode())
    for row in frame[["_key", "title_design", "summary", "eligibility"]].itertuples(index=False):
        digest.update("\x1f".join(map(str, row)).encode("utf-8"))
    return digest.hexdigest()


def register_artifact(cache_dir: Path, entry: dict) -> None:
    import fcntl

    path = cache_dir / "artifacts.json"
    lock_path = cache_dir / "artifacts.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            try:
                entries = json.loads(path.read_text())
            except json.JSONDecodeError:
                entries = []
        else:
            entries = []
        if not any(item.get("content_key") == entry.get("content_key") for item in entries):
            entries.append(entry)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(entries, indent=2))
            os.replace(temporary, path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
