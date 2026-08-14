from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LINKER_VERSION = "registry_linker_v1"
FEATURE_VERSION = "registry_clock_features_v2"
LITERATURE_FEATURE_VERSION = "registry_literature_v1"

ORIGIN_SNAPSHOTS = {
    pd.Timestamp("2017-07-01"): "2017-06-13",
    pd.Timestamp("2018-01-01"): "2017-12-17",
    pd.Timestamp("2018-07-01"): "2018-06-01",
    pd.Timestamp("2019-01-01"): "2018-12-01",
    pd.Timestamp("2020-01-01"): "2019-12-01",
    pd.Timestamp("2021-01-01"): "2020-12-01",
}

PREVIOUS_SNAPSHOTS = {
    "2017-06-13": None,
    "2017-12-17": "2017-06-13",
    "2018-06-01": "2017-12-17",
    "2018-12-01": "2017-12-17",
    "2019-12-01": "2018-12-01",
    "2020-12-01": "2019-12-01",
}


@dataclass
class RegistryFeatureBundle:
    seeds: pd.DataFrame
    features_by_strength: dict[float, pd.DataFrame]
    linkage: pd.DataFrame
    reports: dict[str, Any]


def normalize_identity_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^\w]+", " ", normalized).strip()


def normalize_title(value: Any) -> str:
    normalized = normalize_identity_text(value)
    suffixes = [
        r"\s+clinical trial$", r"\s+clinical study$", r"\s+an? observational study$",
        r"\s+a randomized controlled trial$", r"\s+study protocol$",
    ]
    for suffix in suffixes:
        normalized = re.sub(suffix, "", normalized).strip()
    return normalized


def _numeric_agreement(left: Any, right: Any, relative: float = 0.05, absolute: float = 5.0) -> bool:
    left_value = pd.to_numeric(pd.Series([left]), errors="coerce").iat[0]
    right_value = pd.to_numeric(pd.Series([right]), errors="coerce").iat[0]
    if pd.isna(left_value) or pd.isna(right_value):
        return False
    return abs(float(left_value) - float(right_value)) <= max(absolute, relative * max(abs(float(left_value)), abs(float(right_value)), 1.0))


def _categorical_agreement(left: Any, right: Any) -> bool:
    left_value = normalize_identity_text(left)
    right_value = normalize_identity_text(right)
    return bool(left_value and right_value and left_value == right_value)


def _audit_count(local: pd.Series, external: pd.Series) -> int:
    checks = [
        _categorical_agreement(local.get("study_type"), external.get("study_type")),
        _categorical_agreement(local.get("phase"), external.get("phase")),
        _categorical_agreement(local.get("source"), external.get("source")),
        _numeric_agreement(local.get("enrollment"), external.get("enrollment")),
        _numeric_agreement(local.get("number_of_arms"), external.get("number_of_arms"), 0.0, 0.0),
        _numeric_agreement(local.get("number_of_groups"), external.get("number_of_groups"), 0.0, 0.0),
    ]
    return int(sum(checks))


def _title_similarity(local: pd.Series, external: pd.Series) -> float:
    local_titles = [normalize_title(local.get("official_title")), normalize_title(local.get("brief_title"))]
    external_titles = [normalize_title(external.get("official_title")), normalize_title(external.get("brief_title"))]
    scores = []
    for left in local_titles:
        for right in external_titles:
            if left and right:
                scores.append(SequenceMatcher(None, left, right, autojunk=False).ratio())
    return max(scores, default=0.0)


def link_snapshot(seeds: pd.DataFrame, local_studies: pd.DataFrame, external_studies: pd.DataFrame, snapshot_date: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    local = seeds[["row_id", "nct_id", "timestamp", "split"]].merge(local_studies, on="nct_id", how="left", suffixes=("", "_local"))
    external = external_studies.copy().reset_index(drop=True)
    external["start_date_parsed"] = pd.to_datetime(external["start_date"], errors="coerce")
    external["official_title_normalized"] = external["official_title"].map(normalize_title)
    external["brief_title_normalized"] = external["brief_title"].map(normalize_title)
    external["source_normalized"] = external["source"].map(normalize_identity_text)
    external["phase_normalized"] = external["phase"].map(normalize_identity_text)
    external["enrollment_numeric"] = pd.to_numeric(external["enrollment"], errors="coerce")
    local["start_date_parsed"] = pd.to_datetime(local["start_date"], errors="coerce")
    title_index: dict[str, set[int]] = {}
    date_index: dict[pd.Timestamp, list[int]] = {}
    for index, official, brief, start_date in zip(external.index, external["official_title_normalized"], external["brief_title_normalized"], external["start_date_parsed"]):
        for key in (official, brief):
            if key:
                title_index.setdefault(key, set()).add(int(index))
        if pd.notna(start_date):
            date_index.setdefault(pd.Timestamp(start_date).normalize(), []).append(int(index))
    proposals = []
    unresolved = []
    for _, row in local.iterrows():
        candidates: set[int] = set()
        for title in (row.get("official_title"), row.get("brief_title")):
            candidates.update(title_index.get(normalize_title(title), set()))
        compatible = []
        for candidate_index in candidates:
            candidate = external.loc[candidate_index]
            if pd.isna(row["start_date_parsed"]) or pd.isna(candidate["start_date_parsed"]):
                continue
            start_delta = abs((row["start_date_parsed"] - candidate["start_date_parsed"]).days)
            if start_delta <= 365:
                compatible.append((candidate_index, start_delta, _audit_count(row, candidate)))
        compatible.sort(key=lambda value: (-value[2], value[1], str(external.at[value[0], "nct_id"])))
        if compatible:
            best = compatible[0]
            runner_score = compatible[1][2] if len(compatible) > 1 else -1
            proposals.append({
                "row_id": int(row["row_id"]), "external_index": best[0], "external_nct_id": external.at[best[0], "nct_id"],
                "match_type": "exact", "link_confidence": 1.0, "runner_margin": float(max(0.0, (best[2] - runner_score) / 6.0)),
                "audit_agreements": int(best[2]), "start_delta_days": int(best[1]),
            })
        else:
            unresolved.append(row)
    for row in unresolved:
        if pd.isna(row["start_date_parsed"]):
            continue
        candidate_indices = []
        for offset in range(-31, 32):
            candidate_indices.extend(date_index.get(pd.Timestamp(row["start_date_parsed"] + pd.Timedelta(days=offset)).normalize(), []))
        candidate_frame = external.loc[list(set(candidate_indices))]
        source_normalized = normalize_identity_text(row.get("source"))
        phase_normalized = normalize_identity_text(row.get("phase"))
        enrollment = pd.to_numeric(pd.Series([row.get("enrollment")]), errors="coerce").iat[0]
        if source_normalized:
            candidate_frame = candidate_frame[candidate_frame["source_normalized"] == source_normalized]
        if phase_normalized:
            candidate_frame = candidate_frame[(candidate_frame["phase_normalized"] == phase_normalized) | (candidate_frame["phase_normalized"] == "")]
        if pd.notna(enrollment):
            tolerance = max(10.0, 0.10 * abs(float(enrollment)))
            candidate_frame = candidate_frame[(candidate_frame["enrollment_numeric"] - float(enrollment)).abs() <= tolerance]
        scored = []
        for candidate_index in candidate_frame.index:
            candidate = external.loc[candidate_index]
            similarity = _title_similarity(row, candidate)
            audit = _audit_count(row, candidate)
            start_delta = abs((row["start_date_parsed"] - candidate["start_date_parsed"]).days)
            scored.append((similarity, audit, -start_delta, candidate_index))
        scored.sort(reverse=True)
        if not scored:
            continue
        best = scored[0]
        runner_similarity = scored[1][0] if len(scored) > 1 else 0.0
        if best[0] >= 0.995 and best[0] - runner_similarity >= 0.05 and best[1] >= 2:
            proposals.append({
                "row_id": int(row["row_id"]), "external_index": int(best[3]), "external_nct_id": external.at[best[3], "nct_id"],
                "match_type": "fuzzy", "link_confidence": float(best[0]), "runner_margin": float(best[0] - runner_similarity),
                "audit_agreements": int(best[1]), "start_delta_days": int(-best[2]),
            })
    proposals.sort(key=lambda value: (-value["link_confidence"], -value["audit_agreements"], value["start_delta_days"], value["row_id"]))
    used_external = set()
    accepted = []
    for proposal in proposals:
        if proposal["external_nct_id"] in used_external:
            continue
        used_external.add(proposal["external_nct_id"])
        accepted.append(proposal)
    linkage = seeds[["row_id", "nct_id", "timestamp", "split"]].copy()
    accepted_frame = pd.DataFrame(accepted)
    if len(accepted_frame):
        linkage = linkage.merge(accepted_frame.drop(columns=["external_index"]), on="row_id", how="left")
    else:
        for column in ["external_nct_id", "match_type", "link_confidence", "runner_margin", "audit_agreements", "start_delta_days"]:
            linkage[column] = np.nan
    linkage["linked"] = linkage["external_nct_id"].notna()
    audit_sample = linkage[linkage["linked"]].sample(min(100, int(linkage["linked"].sum())), random_state=1337) if linkage["linked"].any() else linkage.iloc[:0]
    estimated_precision = float((audit_sample["audit_agreements"] >= 2).mean()) if len(audit_sample) else 0.0
    report = {
        "snapshot_date": snapshot_date,
        "rows": len(linkage),
        "linked": int(linkage["linked"].sum()),
        "coverage": float(linkage["linked"].mean()),
        "exact": int((linkage["match_type"] == "exact").sum()),
        "fuzzy": int((linkage["match_type"] == "fuzzy").sum()),
        "audit_rows": len(audit_sample),
        "estimated_precision": estimated_precision,
        "linker_version": LINKER_VERSION,
    }
    return linkage, report


def _load_snapshot(projected_root: Path, snapshot_date: str, tables: list[str]) -> dict[str, pd.DataFrame]:
    return {table: pd.read_parquet(projected_root / snapshot_date / f"{table}.parquet") for table in tables}


def _days(snapshot_date: str, values: pd.Series) -> pd.Series:
    return (pd.Timestamp(snapshot_date) - pd.to_datetime(values, errors="coerce")).dt.days.astype(float)


def _boolean(values: pd.Series) -> pd.Series:
    normalized = values.fillna("").astype(str).str.casefold()
    return normalized.isin(["t", "true", "1", "yes"]).astype(float)


def _relation_lists(frame: pd.DataFrame, value_column: str, prefix: str, allowed_nct: set[str] | None = None) -> dict[str, list[str]]:
    source = frame[["nct_id", value_column]].dropna().copy()
    if allowed_nct is not None:
        source = source[source["nct_id"].isin(allowed_nct)]
    source["key"] = source[value_column].map(normalize_identity_text).map(lambda value: f"{prefix}:{value}" if value else "")
    source = source[source["key"] != ""]
    return source.groupby("nct_id")["key"].apply(lambda values: sorted(set(values))).to_dict()


def _cross_keys(left: list[str], right: list[str], prefix: str, limit: int = 256) -> list[str]:
    return [f"{prefix}:{a.split(':', 1)[-1]}|{b.split(':', 1)[-1]}" for a in left for b in right][:limit]


def _key_maps(tables: dict[str, pd.DataFrame], allowed_nct: set[str] | None = None) -> dict[str, dict[str, list[str]]]:
    studies = tables["studies"] if allowed_nct is None else tables["studies"][tables["studies"]["nct_id"].isin(allowed_nct)]
    sponsors = tables["sponsors"].copy()
    lead = sponsors[sponsors["lead_or_collaborator"].fillna("").str.casefold().str.contains("lead")]
    maps = {
        "lead_sponsor": _relation_lists(lead, "name", "sponsor", allowed_nct),
        "sponsor_class": _relation_lists(lead, "agency_class", "sponsor_class", allowed_nct),
        "condition": _relation_lists(tables["browse_conditions"], "mesh_term", "condition", allowed_nct),
        "intervention": _relation_lists(tables["browse_interventions"], "mesh_term", "intervention", allowed_nct),
        "facility": _relation_lists(tables["facilities"], "name", "facility", allowed_nct),
        "country": _relation_lists(tables["facilities"], "country", "country", allowed_nct),
    }
    phase = {}
    for nct_id, value in zip(studies["nct_id"], studies["phase"]):
        normalized = normalize_identity_text(value)
        phase[nct_id] = [f"phase:{normalized}"] if normalized else []
    all_nct = set(studies["nct_id"])
    maps["condition_phase"] = {nct: _cross_keys(maps["condition"].get(nct, []), phase.get(nct, []), "condition_phase") for nct in all_nct}
    maps["intervention_condition"] = {nct: _cross_keys(maps["intervention"].get(nct, []), maps["condition"].get(nct, []), "intervention_condition") for nct in all_nct}
    maps["sponsor_condition"] = {nct: _cross_keys(maps["lead_sponsor"].get(nct, []), maps["condition"].get(nct, []), "sponsor_condition") for nct in all_nct}
    return maps


def _public_results(tables: dict[str, pd.DataFrame], snapshot_date: str) -> pd.DataFrame:
    outcomes = tables["outcomes"].copy()
    outcomes["id"] = pd.to_numeric(outcomes["id"], errors="coerce")
    primary = outcomes[outcomes["outcome_type"].fillna("").str.casefold() == "primary"][["id", "nct_id"]]
    analyses = tables["outcome_analyses"].copy()
    analyses["outcome_id"] = pd.to_numeric(analyses["outcome_id"], errors="coerce")
    analyses["p_value"] = pd.to_numeric(analyses["p_value"], errors="coerce")
    analyses = analyses[
        analyses["p_value"].between(0, 1, inclusive="both")
        & (analyses["p_value_modifier"].isna() | (analyses["p_value_modifier"] != ">"))
    ].merge(primary, left_on="outcome_id", right_on="id", how="inner", suffixes=("", "_outcome"))
    studies = tables["studies"][["nct_id", "results_first_submitted_date", "results_first_posted_date", "primary_completion_date"]].copy()
    studies["results_first_submitted_date"] = pd.to_datetime(studies["results_first_submitted_date"], errors="coerce")
    studies["results_first_posted_date"] = pd.to_datetime(studies["results_first_posted_date"], errors="coerce")
    studies["primary_completion_date"] = pd.to_datetime(studies["primary_completion_date"], errors="coerce")
    valid = analyses.groupby("nct_id", as_index=False)["p_value"].min().merge(studies, on="nct_id", how="left")
    valid["public_date"] = valid["results_first_posted_date"].fillna(valid["results_first_submitted_date"]).fillna(pd.Timestamp(snapshot_date) - pd.Timedelta(days=1))
    valid = valid[valid["public_date"] < pd.Timestamp(snapshot_date)]
    valid["label"] = (valid["p_value"] <= 0.05).astype(float)
    valid["recency"] = (pd.Timestamp(snapshot_date) - valid["public_date"]).dt.days.astype(float)
    valid["reporting_lag"] = (valid["public_date"] - valid["primary_completion_date"]).dt.days.astype(float)
    return valid.set_index("nct_id")[["label", "recency", "reporting_lag"]]


def _aggregate_neighbor_set(neighbor_ids: set[str], target_nct: str, result_frame: pd.DataFrame, strength: float, global_rate: float) -> dict[str, float]:
    neighbor_ids.discard(target_nct)
    available = result_frame.reindex(list(neighbor_ids)).dropna(subset=["label"])
    count = float(len(available))
    successes = float(available["label"].sum()) if len(available) else 0.0
    rate = (successes + strength * global_rate) / (count + strength)
    return {
        "count": count,
        "rate": float(rate),
        "uncertainty": float(math.sqrt(max(rate * (1.0 - rate), 0.0) / (count + strength))),
        "recency_min": float(available["recency"].min()) if len(available) else np.nan,
        "recency_median": float(available["recency"].median()) if len(available) else np.nan,
        "reporting_lag_median": float(available["reporting_lag"].median()) if len(available) else np.nan,
    }


def _neighborhood_features(
    tables: dict[str, pd.DataFrame], linked_nct: pd.Series, snapshot_date: str, strength: float,
    key_maps: dict[str, dict[str, list[str]]] | None = None,
    results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    key_maps = _key_maps(tables) if key_maps is None else key_maps
    results = _public_results(tables, snapshot_date) if results is None else results
    global_rate = float(results["label"].mean()) if len(results) else 0.5
    inverse: dict[str, set[str]] = {}
    for group_map in key_maps.values():
        for nct_id, keys in group_map.items():
            if nct_id not in results.index:
                continue
            for key in keys:
                inverse.setdefault(key, set()).add(nct_id)
    rows = []
    direct_groups = [
        "lead_sponsor", "sponsor_class", "condition", "condition_phase", "intervention",
        "intervention_condition", "facility", "country", "sponsor_condition",
    ]
    hop_groups = ["lead_sponsor", "condition", "intervention", "facility"]
    for nct_id_value in linked_nct:
        nct_id = str(nct_id_value) if pd.notna(nct_id_value) else ""
        row: dict[str, float] = {"neighborhood_global_rate": global_rate}
        one_hop: set[str] = set()
        for group in direct_groups:
            neighbor_ids: set[str] = set()
            keys = key_maps[group].get(nct_id, [])
            for key in keys:
                neighbor_ids.update(inverse.get(key, set()))
            aggregate = _aggregate_neighbor_set(neighbor_ids, nct_id, results, strength, global_rate)
            for suffix, value in aggregate.items():
                row[f"neighbor_{group}_{suffix}"] = value
            if group in hop_groups:
                one_hop.update(neighbor_ids)
        one_aggregate = _aggregate_neighbor_set(set(one_hop), nct_id, results, strength, global_rate)
        for suffix, value in one_aggregate.items():
            row[f"neighbor_one_hop_{suffix}"] = value
        two_hop = set(one_hop)
        for neighbor in list(one_hop)[:100]:
            for group in hop_groups:
                for key in key_maps[group].get(neighbor, []):
                    related = inverse.get(key, set())
                    if len(related) <= 200:
                        two_hop.update(related)
        two_aggregate = _aggregate_neighbor_set(two_hop, nct_id, results, strength, global_rate)
        for suffix, value in two_aggregate.items():
            row[f"neighbor_two_hop_{suffix}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _registry_state_features(tables: dict[str, pd.DataFrame], previous_tables: dict[str, pd.DataFrame] | None, linkage: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    features = linkage[["row_id", "linked", "link_confidence", "runner_margin", "audit_agreements", "start_delta_days"]].copy()
    studies = tables["studies"].drop_duplicates("nct_id").set_index("nct_id")
    aligned = studies.reindex(linkage["external_nct_id"].fillna(""))
    aligned.index = features.index
    for column in ["overall_status", "last_known_status", "completion_date_type", "primary_completion_date_type", "enrollment_type", "study_type", "phase"]:
        features[f"cat_registry_{column}"] = aligned[column].fillna("__missing__").astype(str)
    date_columns = [
        "study_first_submitted_date", "results_first_submitted_date", "disposition_first_submitted_date",
        "last_update_submitted_date", "study_first_submitted_qc_date", "study_first_posted_date",
        "results_first_submitted_qc_date", "results_first_posted_date",
        "disposition_first_submitted_qc_date", "disposition_first_posted_date",
        "last_update_submitted_qc_date", "last_update_posted_date", "verification_date",
        "completion_date", "primary_completion_date",
    ]
    for column in date_columns:
        features[f"days_since_{column}"] = _days(snapshot_date, aligned[column]).to_numpy()
        features[f"has_{column}"] = aligned[column].notna().astype(float).to_numpy()
    features["registry_enrollment"] = pd.to_numeric(aligned["enrollment"], errors="coerce").to_numpy()
    features["registry_enrollment_log"] = np.log1p(features["registry_enrollment"].clip(lower=0))
    features["registry_enrollment_actual"] = aligned["enrollment_type"].fillna("").str.casefold().eq("actual").astype(float).to_numpy()
    features["registry_completion_actual"] = aligned["completion_date_type"].fillna("").str.casefold().eq("actual").astype(float).to_numpy()
    features["registry_primary_completion_actual"] = aligned["primary_completion_date_type"].fillna("").str.casefold().eq("actual").astype(float).to_numpy()
    for column in ["number_of_arms", "number_of_groups"]:
        features[f"registry_{column}"] = pd.to_numeric(aligned[column], errors="coerce").to_numpy()
    for column in ["has_dmc", "is_fda_regulated_drug", "is_fda_regulated_device"]:
        features[f"registry_{column}"] = _boolean(aligned[column]).to_numpy()
    features["why_stopped_present"] = aligned["why_stopped"].fillna("").str.len().gt(0).astype(float).to_numpy()
    target_nct = set(linkage["external_nct_id"].dropna().astype(str))
    facilities = tables["facilities"][tables["facilities"]["nct_id"].isin(target_nct)].copy()
    facility_group = facilities.groupby("nct_id")
    facility_count = facility_group.size()
    country_count = facility_group["country"].nunique()
    recruiting_share = facilities.assign(_recruiting=facilities["status"].fillna("").str.casefold().str.contains("recruit").astype(float)).groupby("nct_id")["_recruiting"].mean()
    features["registry_facility_count"] = linkage["external_nct_id"].map(facility_count).fillna(0).to_numpy()
    features["registry_country_count"] = linkage["external_nct_id"].map(country_count).fillna(0).to_numpy()
    features["registry_recruiting_site_share"] = linkage["external_nct_id"].map(recruiting_share).fillna(0).to_numpy()
    pending = tables["pending_results"].copy()
    pending["event_date"] = pd.to_datetime(pending["event_date"], errors="coerce")
    pending["event_norm"] = pending["event"].fillna("").str.casefold()
    pending_group = pending.groupby("nct_id")
    features["pending_event_count"] = linkage["external_nct_id"].map(pending_group.size()).fillna(0).to_numpy()
    features["pending_event_recency"] = linkage["external_nct_id"].map(pending_group["event_date"].max()).pipe(lambda values: _days(snapshot_date, values)).to_numpy()
    for name, pattern in {
        "submission": "submit", "qc_return": "return|quality control|qc", "cancellation": "cancel",
        "extension": "extension", "disposition": "disposition", "posting": "post",
    }.items():
        counts = pending.assign(_mark=pending["event_norm"].str.contains(pattern, regex=True).astype(int)).groupby("nct_id")["_mark"].sum()
        features[f"pending_{name}_count"] = linkage["external_nct_id"].map(counts).fillna(0).to_numpy()
    documents = tables["documents"].copy()
    document_group = documents.groupby("nct_id")
    features["registry_document_count"] = linkage["external_nct_id"].map(document_group.size()).fillna(0).to_numpy()
    result_document = documents.assign(_mark=documents["document_type"].fillna("").str.casefold().str.contains("result").astype(int)).groupby("nct_id")["_mark"].sum()
    features["registry_result_document_count"] = linkage["external_nct_id"].map(result_document).fillna(0).to_numpy()
    calculated = tables["calculated_values"].drop_duplicates("nct_id").set_index("nct_id")
    calculated_aligned = calculated.reindex(linkage["external_nct_id"].fillna(""))
    calculated_aligned.index = features.index
    for column in [
        "number_of_facilities", "actual_duration", "months_to_report_results",
        "number_of_primary_outcomes_to_measure", "number_of_secondary_outcomes_to_measure",
        "number_of_other_outcomes_to_measure",
    ]:
        features[f"calculated_{column}"] = pd.to_numeric(calculated_aligned[column], errors="coerce").to_numpy()
    for column in ["were_results_reported", "has_us_facility", "has_single_facility"]:
        features[f"calculated_{column}"] = _boolean(calculated_aligned[column]).to_numpy()
    primary_outcomes = tables["design_outcomes"][
        tables["design_outcomes"]["nct_id"].isin(target_nct)
        & (tables["design_outcomes"]["outcome_type"].fillna("").str.casefold() == "primary")
    ].copy()
    outcome_group = primary_outcomes.groupby("nct_id")
    features["registry_primary_outcome_count"] = linkage["external_nct_id"].map(outcome_group.size()).fillna(0).to_numpy()
    measure_length = primary_outcomes.assign(_length=primary_outcomes["measure"].fillna("").str.len()).groupby("nct_id")["_length"].mean()
    timeframe_length = primary_outcomes.assign(_length=primary_outcomes["time_frame"].fillna("").str.len()).groupby("nct_id")["_length"].mean()
    features["registry_primary_outcome_measure_length"] = linkage["external_nct_id"].map(measure_length).to_numpy()
    features["registry_primary_outcome_timeframe_length"] = linkage["external_nct_id"].map(timeframe_length).to_numpy()
    references = tables["study_references"].copy()
    reference_group = references.groupby("nct_id")
    features["registry_reference_count"] = linkage["external_nct_id"].map(reference_group.size()).fillna(0).to_numpy()
    result_reference = references.assign(_mark=references["reference_type"].fillna("").str.casefold().str.contains("result").astype(int)).groupby("nct_id")["_mark"].sum()
    features["registry_result_reference_count"] = linkage["external_nct_id"].map(result_reference).fillna(0).to_numpy()
    if previous_tables is not None:
        previous_studies = previous_tables["studies"].drop_duplicates("nct_id").set_index("nct_id").reindex(linkage["external_nct_id"].fillna(""))
        previous_studies.index = features.index
        previous_enrollment = pd.to_numeric(previous_studies["enrollment"], errors="coerce")
        features["enrollment_revision"] = features["registry_enrollment"] - previous_enrollment.to_numpy()
        features["actual_to_planned_enrollment_ratio"] = features["registry_enrollment"] / previous_enrollment.clip(lower=1).to_numpy()
        for column in ["completion_date", "primary_completion_date"]:
            current_date = pd.to_datetime(aligned[column], errors="coerce")
            previous_date = pd.to_datetime(previous_studies[column], errors="coerce")
            features[f"{column}_slippage_days"] = (current_date - previous_date).dt.days.to_numpy()
        previous_facilities = previous_tables["facilities"][previous_tables["facilities"]["nct_id"].isin(target_nct)].groupby("nct_id").size()
        previous_site_count = linkage["external_nct_id"].map(previous_facilities).fillna(0)
        features["site_count_change"] = features["registry_facility_count"] - previous_site_count.to_numpy()
        previous_primary = previous_tables["design_outcomes"][
            previous_tables["design_outcomes"]["nct_id"].isin(target_nct)
            & (previous_tables["design_outcomes"]["outcome_type"].fillna("").str.casefold() == "primary")
        ].copy()
        current_sets = primary_outcomes.assign(_value=primary_outcomes["measure"].map(normalize_identity_text) + "|" + primary_outcomes["time_frame"].map(normalize_identity_text)).groupby("nct_id")["_value"].apply(lambda values: set(values))
        previous_sets = previous_primary.assign(_value=previous_primary["measure"].map(normalize_identity_text) + "|" + previous_primary["time_frame"].map(normalize_identity_text)).groupby("nct_id")["_value"].apply(lambda values: set(values))
        features["primary_outcome_changed"] = [float(current_sets.get(nct, set()) != previous_sets.get(nct, set())) for nct in linkage["external_nct_id"].fillna("")]
    else:
        for column in ["enrollment_revision", "actual_to_planned_enrollment_ratio", "completion_date_slippage_days", "primary_completion_date_slippage_days", "site_count_change", "primary_outcome_changed"]:
            features[column] = np.nan
    features["registry_missing_count"] = features.isna().sum(axis=1).astype(float)
    return features


def _within_origin(features: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    result = features.copy()
    numeric = [column for column in result.select_dtypes(include=[np.number]).columns if column not in ["row_id"]]
    prioritized = [
        column for column in numeric if column.startswith("days_since_") or column.startswith("pending_")
        or column.startswith("neighbor_") or column in [
            "registry_enrollment", "registry_facility_count", "registry_country_count", "registry_recruiting_site_share",
            "enrollment_revision", "actual_to_planned_enrollment_ratio", "site_count_change", "registry_missing_count",
        ]
    ]
    groups = pd.Series(pd.to_datetime(timestamps)).groupby(pd.to_datetime(timestamps)).groups
    for column in prioritized:
        values = pd.to_numeric(result[column], errors="coerce")
        rank = pd.Series(np.nan, index=result.index, dtype=float)
        z_score = pd.Series(np.nan, index=result.index, dtype=float)
        gap = pd.Series(np.nan, index=result.index, dtype=float)
        for indices in groups.values():
            current = values.iloc[list(indices)]
            rank.iloc[list(indices)] = current.rank(pct=True).to_numpy()
            z_score.iloc[list(indices)] = ((current - current.mean()) / (current.std(ddof=0) + 1e-6)).clip(-20, 20).to_numpy()
            gap.iloc[list(indices)] = (current.max() - current).to_numpy()
        result[f"origin_rank_{column}"] = rank.astype(np.float32)
        result[f"origin_z_{column}"] = z_score.astype(np.float32)
        result[f"origin_gap_{column}"] = gap.astype(np.float32)
    return result


def build_registry_features(db: Any, seeds: pd.DataFrame, projected_root: Path, strengths: tuple[float, ...] = (20.0, 50.0, 100.0)) -> RegistryFeatureBundle:
    seeds = seeds.reset_index(drop=True).copy()
    if "row_id" not in seeds:
        seeds["row_id"] = np.arange(len(seeds), dtype=np.int64)
    local_studies = db.table_dict["studies"].df.copy()
    table_names = [
        "studies", "calculated_values", "facilities", "sponsors", "browse_conditions",
        "browse_interventions", "pending_results", "documents", "design_outcomes",
        "study_references", "outcomes", "outcome_analyses",
    ]
    linkage_parts = []
    state_parts = []
    neighborhood_parts = {strength: [] for strength in strengths}
    reports = {}
    for timestamp, snapshot_date in ORIGIN_SNAPSHOTS.items():
        current = seeds[pd.to_datetime(seeds["timestamp"]) == timestamp].copy()
        if current.empty:
            continue
        print(f"[registry-feature] snapshot_start={snapshot_date} rows={len(current)}", flush=True)
        tables = _load_snapshot(projected_root, snapshot_date, table_names)
        previous_date = PREVIOUS_SNAPSHOTS[snapshot_date]
        previous_tables = _load_snapshot(projected_root, previous_date, ["studies", "facilities", "design_outcomes"]) if previous_date else None
        linkage, report = link_snapshot(current, local_studies, tables["studies"], snapshot_date)
        if report["estimated_precision"] < 0.99:
            raise RuntimeError(f"Registry linkage precision gate failed for {snapshot_date}: {report['estimated_precision']}")
        reports[snapshot_date] = report
        print(f"[registry-feature] linkage={snapshot_date} coverage={report['coverage']:.6f} precision={report['estimated_precision']:.6f}", flush=True)
        state = _registry_state_features(tables, previous_tables, linkage, snapshot_date)
        state["timestamp"] = timestamp
        state["snapshot_date"] = snapshot_date
        linkage_parts.append(linkage)
        state_parts.append(state)
        public_results = _public_results(tables, snapshot_date)
        target_nct = set(linkage["external_nct_id"].dropna().astype(str))
        allowed_nct = target_nct | set(public_results.index.astype(str))
        key_maps = _key_maps(tables, allowed_nct)
        for strength in strengths:
            neighborhood = _neighborhood_features(
                tables, linkage["external_nct_id"], snapshot_date, strength,
                key_maps=key_maps, results=public_results,
            )
            neighborhood["row_id"] = linkage["row_id"].to_numpy()
            neighborhood_parts[strength].append(neighborhood)
        print(f"[registry-feature] snapshot_complete={snapshot_date} public_results={len(public_results)}", flush=True)
    linkage_all = pd.concat(linkage_parts, ignore_index=True).sort_values("row_id").reset_index(drop=True)
    state_all = pd.concat(state_parts, ignore_index=True).sort_values("row_id").reset_index(drop=True)
    features_by_strength = {}
    for strength in strengths:
        neighborhood = pd.concat(neighborhood_parts[strength], ignore_index=True)
        features = state_all.merge(neighborhood, on="row_id", how="left").sort_values("row_id").reset_index(drop=True)
        timestamps = seeds.set_index("row_id").loc[features["row_id"], "timestamp"].reset_index(drop=True)
        features = _within_origin(features.drop(columns=["timestamp", "snapshot_date"]), timestamps)
        features_by_strength[strength] = features
    expected = set(seeds["row_id"])
    if set(linkage_all["row_id"]) != expected:
        missing = sorted(expected - set(linkage_all["row_id"]))
        raise RuntimeError(f"Registry feature coverage omitted {len(missing)} mapped-origin rows")
    return RegistryFeatureBundle(seeds=seeds, features_by_strength=features_by_strength, linkage=linkage_all, reports=reports)


def save_registry_bundle(bundle: RegistryFeatureBundle, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    bundle.seeds.to_parquet(destination / "seeds.parquet", index=False)
    bundle.linkage.to_parquet(destination / "linkage.parquet", index=False)
    for strength, features in bundle.features_by_strength.items():
        features.to_parquet(destination / f"features_strength_{int(strength)}.parquet", index=False)
    (destination / "reports.json").write_text(json.dumps({"feature_version": FEATURE_VERSION, "reports": bundle.reports}, indent=2, sort_keys=True) + "\n")


def load_registry_bundle(destination: Path, strengths: tuple[float, ...] = (20.0, 50.0, 100.0)) -> RegistryFeatureBundle:
    metadata = json.loads((destination / "reports.json").read_text())
    if metadata.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError(f"Registry feature cache version {metadata.get('feature_version')} != {FEATURE_VERSION}")
    seeds = pd.read_parquet(destination / "seeds.parquet")
    linkage = pd.read_parquet(destination / "linkage.parquet")
    features = {strength: pd.read_parquet(destination / f"features_strength_{int(strength)}.parquet") for strength in strengths}
    return RegistryFeatureBundle(seeds=seeds, features_by_strength=features, linkage=linkage, reports=metadata["reports"])


def refresh_literature_features(bundle: RegistryFeatureBundle, projected_root: Path) -> bool:
    if bundle.reports.get("literature_feature_version") == LITERATURE_FEATURE_VERSION:
        return False
    evidence = pd.DataFrame({"row_id": bundle.linkage["row_id"].astype(np.int64)})
    feature_names = [
        "registry_result_reference_count", "registry_result_pmid_count",
        "registry_result_reference_final_count", "registry_result_reference_interim_count",
        "registry_result_reference_protocol_count", "registry_result_reference_primary_count",
        "registry_result_reference_randomized_count", "registry_result_reference_citation_length",
        "registry_result_reference_insufficient",
    ]
    for name in feature_names:
        evidence[name] = 0.0
    for timestamp, snapshot_date in ORIGIN_SNAPSHOTS.items():
        mask = pd.to_datetime(bundle.linkage["timestamp"]).eq(timestamp)
        current = bundle.linkage.loc[mask, ["row_id", "external_nct_id"]].copy()
        references = pd.read_parquet(projected_root / snapshot_date / "study_references.parquet")
        references = references[references["reference_type"].fillna("").str.casefold().str.contains("result")].copy()
        citation = references["citation"].fillna("").astype(str).str.casefold()
        references["_pmid"] = references["pmid"].notna().astype(float)
        references["_final"] = citation.str.contains(r"\bfinal\b", regex=True).astype(float)
        references["_interim"] = citation.str.contains(r"\binterim\b", regex=True).astype(float)
        references["_protocol"] = citation.str.contains(r"\bprotocol\b", regex=True).astype(float)
        references["_primary"] = citation.str.contains(r"\bprimary\b", regex=True).astype(float)
        references["_randomized"] = citation.str.contains(r"\brandomi[sz]", regex=True).astype(float)
        references["_length"] = references["citation"].fillna("").astype(str).str.len().astype(float)
        grouped = references.groupby("nct_id").agg(
            registry_result_reference_count=("nct_id", "size"),
            registry_result_pmid_count=("_pmid", "sum"),
            registry_result_reference_final_count=("_final", "sum"),
            registry_result_reference_interim_count=("_interim", "sum"),
            registry_result_reference_protocol_count=("_protocol", "sum"),
            registry_result_reference_primary_count=("_primary", "sum"),
            registry_result_reference_randomized_count=("_randomized", "sum"),
            registry_result_reference_citation_length=("_length", "mean"),
        )
        for name in feature_names[:-1]:
            mapped = current["external_nct_id"].map(grouped[name]).fillna(0.0).to_numpy(dtype=np.float32)
            evidence.loc[mask, name] = mapped
        evidence.loc[mask, "registry_result_reference_insufficient"] = (evidence.loc[mask, "registry_result_reference_count"].to_numpy() == 0).astype(np.float32)
    for strength, features in bundle.features_by_strength.items():
        drop = [name for name in feature_names if name in features.columns]
        bundle.features_by_strength[strength] = features.drop(columns=drop).merge(evidence, on="row_id", how="left").sort_values("row_id").reset_index(drop=True)
    bundle.reports["literature_feature_version"] = LITERATURE_FEATURE_VERSION
    return True
