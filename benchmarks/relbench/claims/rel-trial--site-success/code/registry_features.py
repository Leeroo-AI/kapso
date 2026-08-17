# Imports

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Configuration

VERSION = "site_success_registry_v1"
SNAPSHOTS = {
    pd.Timestamp("2017-07-01"): "2017-06-13",
    pd.Timestamp("2017-10-01"): "2017-06-13",
    pd.Timestamp("2018-01-01"): "2017-12-17",
    pd.Timestamp("2018-04-01"): "2017-12-17",
    pd.Timestamp("2018-07-01"): "2018-06-01",
    pd.Timestamp("2018-10-01"): "2018-06-01",
    pd.Timestamp("2019-01-01"): "2018-12-01",
    pd.Timestamp("2019-04-01"): "2018-12-01",
    pd.Timestamp("2019-07-01"): "2018-12-01",
    pd.Timestamp("2019-10-01"): "2018-12-01",
    pd.Timestamp("2020-01-01"): "2019-12-01",
    pd.Timestamp("2021-01-01"): "2020-12-01",
}
PREVIOUS = {
    "2017-06-13": None,
    "2017-12-17": "2017-06-13",
    "2018-06-01": "2017-12-17",
    "2018-12-01": "2017-12-17",
    "2019-12-01": "2018-12-01",
    "2020-12-01": "2019-12-01",
}
SHA256 = {
    "2017-06-13": "6dff60cfe80684157c2c072238945851c8cfb3f29e4a0a07c1d893c26353969c",
    "2017-12-17": "866a9d38df183788fb20db57bae0ddef83c2963a7bf95b2fd15011189af9a42b",
    "2018-06-01": "f13182b5a8cb42c3e8700dbd8d1e7f59e1b33e0c07c14b896f623ad402523d69",
    "2018-12-01": "8590ca2c0767957b2fc7b1bf8283c18891aa15ef7288c62cf54755ca153a2212",
    "2019-12-01": "c56c06707fa77786229f3c745771bbc747ba304a32601085988bedb9c5442909",
    "2020-12-01": "d44da0ac916022efb876089dc7e8a7bb698e8d0db72211be184bc90426868405",
}
URLS = {
    value: f"https://aact.ctti-clinicaltrials.org/static/static_db_copies/daily/{value}?source=web"
    for value in SHA256
}
CATEGORICAL = [
    "registry_overall_status",
    "registry_last_known_status",
    "registry_completion_date_type",
    "registry_primary_completion_date_type",
    "registry_why_stopped",
]


# Utilities

def _normalize(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^\w]+", " ", normalized).strip()


def _boolean(values: pd.Series) -> np.ndarray:
    return values.fillna("").astype(str).str.casefold().isin(["t", "true", "1", "yes"]).to_numpy(dtype=np.float32)


def _numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32)


def _days(origin: pd.Timestamp, values: pd.Series) -> np.ndarray:
    return (origin - pd.to_datetime(values, errors="coerce")).dt.days.to_numpy(dtype=np.float32)


def _load(root: Path, snapshot: str, tables: list[str]) -> dict[str, pd.DataFrame]:
    return {table: pd.read_parquet(root / snapshot / f"{table}.parquet") for table in tables}


def verify_snapshots(root: Path) -> tuple[bool, dict[str, Any]]:
    records: dict[str, Any] = {}
    for snapshot, expected in SHA256.items():
        metadata_path = root / snapshot / "metadata.json"
        if not metadata_path.exists():
            return False, {"missing": snapshot}
        metadata = json.loads(metadata_path.read_text())
        actual = metadata.get("archive_sha256", "")
        maximum = pd.Timestamp(metadata.get("maximum_usable_timestamp"))
        if actual != expected or maximum > pd.Timestamp(snapshot):
            return False, {"invalid": snapshot, "sha": actual}
        records[snapshot] = {
            "url": URLS[snapshot],
            "sha256": actual,
            "maximum_usable_timestamp": str(maximum),
        }
    return True, records


# Linkage

def _unique_title_map(external: pd.DataFrame) -> dict[str, str]:
    values: dict[str, list[str]] = {}
    for row in external[["nct_id", "official_title", "brief_title"]].itertuples(index=False):
        for title in [row.official_title, row.brief_title]:
            key = _normalize(title)
            if key:
                values.setdefault(key, []).append(str(row.nct_id))
    return {key: items[0] for key, items in values.items() if len(set(items)) == 1}


def link_trials(local: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    title_map = _unique_title_map(external)
    external_index = external.drop_duplicates("nct_id").set_index(external["nct_id"].astype(str))
    result = local[["nct_id", "start_date", "official_title", "brief_title", "phase", "source", "enrollment"]].drop_duplicates("nct_id").copy()
    official = result["official_title"].map(lambda value: title_map.get(_normalize(value)))
    brief = result["brief_title"].map(lambda value: title_map.get(_normalize(value)))
    result["external_nct_id"] = official.fillna(brief)
    linked = result["external_nct_id"].notna()
    aligned = result.loc[linked, "external_nct_id"].map(external_index["start_date"])
    delta = (pd.to_datetime(result.loc[linked, "start_date"]) - pd.to_datetime(aligned, errors="coerce")).dt.days.abs()
    result.loc[linked, "start_delta_days"] = delta.to_numpy()
    result.loc[linked & result["start_delta_days"].gt(365), "external_nct_id"] = np.nan
    result["linked"] = result["external_nct_id"].notna()
    return result


# Evidence

def direct_evidence(tables: dict[str, pd.DataFrame], origin: pd.Timestamp) -> pd.DataFrame:
    outcomes = tables["outcomes"].copy()
    analyses = tables["outcome_analyses"].copy()
    outcomes["id_numeric"] = pd.to_numeric(outcomes["id"], errors="coerce")
    primary = outcomes[outcomes["outcome_type"].fillna("").str.casefold().eq("primary")]
    analyses["outcome_id_numeric"] = pd.to_numeric(analyses["outcome_id"], errors="coerce")
    analyses["p_value_numeric"] = pd.to_numeric(analyses["p_value"], errors="coerce")
    admissible = analyses[
        analyses["p_value_numeric"].between(0.0, 1.0, inclusive="both")
        & (analyses["p_value_modifier"].isna() | analyses["p_value_modifier"].ne(">"))
    ].merge(primary[["id_numeric", "nct_id"]], left_on="outcome_id_numeric", right_on="id_numeric", how="inner", suffixes=("", "_outcome"))
    admissible = admissible[admissible["nct_id"].astype(str).eq(admissible["nct_id_outcome"].astype(str))]
    grouped = admissible.groupby("nct_id").agg(
        qualifying_analysis_count=("p_value_numeric", "size"),
        minimum_p_value=("p_value_numeric", "min"),
        maximum_p_value=("p_value_numeric", "max"),
        analyzed_primary_count=("outcome_id_numeric", "nunique"),
    )
    registered = tables["design_outcomes"][
        tables["design_outcomes"]["outcome_type"].fillna("").str.casefold().eq("primary")
    ].groupby("nct_id").size()
    reported = primary.groupby("nct_id")["id_numeric"].nunique()
    studies = tables["studies"].drop_duplicates("nct_id").set_index("nct_id")
    calculated = tables["calculated_values"].drop_duplicates("nct_id").set_index("nct_id")
    result = grouped.copy()
    result["registered_primary_count"] = result.index.map(registered).fillna(0).to_numpy(dtype=np.float32)
    result["reported_primary_count"] = result.index.map(reported).fillna(0).to_numpy(dtype=np.float32)
    results_reported = result.index.map(calculated["were_results_reported"]).fillna("").astype(str).str.casefold().isin(["t", "true", "1", "yes"])
    posted = pd.to_datetime(result.index.map(studies["results_first_posted_date"]), errors="coerce")
    submitted = pd.to_datetime(result.index.map(studies["results_first_submitted_date"]), errors="coerce")
    completed = np.asarray(results_reported, dtype=bool) & np.asarray(posted.notna() | submitted.notna(), dtype=bool) & np.asarray(posted.isna() | (posted < origin), dtype=bool) & np.asarray(submitted.isna() | (submitted < origin), dtype=bool)
    complete_endpoints = (
        result["registered_primary_count"].gt(0)
        & result["reported_primary_count"].eq(result["registered_primary_count"])
        & result["analyzed_primary_count"].eq(result["registered_primary_count"])
    )
    result["direct_negative"] = result["maximum_p_value"].ge(0.05)
    result["direct_positive"] = result["maximum_p_value"].lt(0.05) & complete_endpoints & completed
    result["direct_abstain"] = ~(result["direct_negative"] | result["direct_positive"])
    result["expert_probability"] = np.nan
    result.loc[result["direct_negative"], "expert_probability"] = 0.005
    result.loc[result["direct_positive"], "expert_probability"] = 0.995
    return result


# Snapshot features

def snapshot_features(
    local_studies: pd.DataFrame,
    current_pairs: pd.DataFrame,
    projected: Path,
    snapshot: str,
    origin: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tables = _load(projected, snapshot, [
        "studies",
        "calculated_values",
        "facilities",
        "pending_results",
        "design_outcomes",
        "outcomes",
        "outcome_analyses",
        "sponsors",
    ])
    external_studies = tables["studies"].copy()
    linkage = link_trials(local_studies[local_studies["nct_id"].isin(current_pairs["nct_id"])], external_studies)
    current = current_pairs.merge(linkage[["nct_id", "external_nct_id", "linked", "start_delta_days"]], on="nct_id", how="left")
    external_studies = external_studies.drop_duplicates("nct_id").set_index(external_studies["nct_id"].astype(str))
    external_id = current["external_nct_id"]
    values: dict[str, Any] = {
        "registry_linked": current["linked"].fillna(False).to_numpy(dtype=np.float32),
        "registry_start_delta_days": pd.to_numeric(current["start_delta_days"], errors="coerce").to_numpy(dtype=np.float32),
    }
    date_columns = [
        "start_date",
        "verification_date",
        "completion_date",
        "primary_completion_date",
        "results_first_submitted_date",
        "results_first_submitted_qc_date",
        "results_first_posted_date",
        "last_update_submitted_date",
        "last_update_posted_date",
    ]
    for column in date_columns:
        aligned = external_id.map(external_studies[column])
        values[f"registry_days_since_{column}"] = _days(origin, aligned)
        values[f"registry_has_{column}"] = pd.to_datetime(aligned, errors="coerce").notna().to_numpy(dtype=np.float32)
    category_sources = {
        "registry_overall_status": "overall_status",
        "registry_last_known_status": "last_known_status",
        "registry_completion_date_type": "completion_date_type",
        "registry_primary_completion_date_type": "primary_completion_date_type",
        "registry_why_stopped": "why_stopped",
    }
    for target, source in category_sources.items():
        values[target] = external_id.map(external_studies[source]).fillna("__missing__").astype(str).to_numpy()
    for column in ["enrollment", "number_of_arms", "number_of_groups"]:
        values[f"registry_{column}"] = _numeric(external_id.map(external_studies[column]))
    calculated = tables["calculated_values"].drop_duplicates("nct_id").set_index(tables["calculated_values"].drop_duplicates("nct_id")["nct_id"].astype(str))
    for column in ["number_of_facilities", "actual_duration", "months_to_report_results", "number_of_primary_outcomes_to_measure", "number_of_secondary_outcomes_to_measure", "number_of_other_outcomes_to_measure"]:
        values[f"registry_{column}"] = _numeric(external_id.map(calculated[column]))
    for column in ["were_results_reported", "has_us_facility", "has_single_facility"]:
        values[f"registry_{column}"] = _boolean(external_id.map(calculated[column]))
    facilities = tables["facilities"].copy()
    facility_count = facilities.groupby("nct_id").size()
    country_count = facilities.groupby("nct_id")["country"].nunique()
    recruiting = facilities["status"].fillna("").str.casefold().str.contains("recruit").groupby(facilities["nct_id"]).mean()
    values["registry_facility_rows"] = _numeric(external_id.map(facility_count))
    values["registry_country_count"] = _numeric(external_id.map(country_count))
    values["registry_recruiting_share"] = _numeric(external_id.map(recruiting))
    pending = tables["pending_results"].copy()
    pending["event_date_parsed"] = pd.to_datetime(pending["event_date"], errors="coerce")
    pending_count = pending.groupby("nct_id").size()
    pending_latest = pending.groupby("nct_id")["event_date_parsed"].max()
    values["registry_pending_count"] = _numeric(external_id.map(pending_count))
    values["registry_pending_recency"] = _days(origin, external_id.map(pending_latest))
    evidence = direct_evidence(tables, origin)
    evidence_columns = [
        "qualifying_analysis_count",
        "minimum_p_value",
        "maximum_p_value",
        "analyzed_primary_count",
        "registered_primary_count",
        "reported_primary_count",
        "direct_negative",
        "direct_positive",
        "direct_abstain",
        "expert_probability",
    ]
    for column in evidence_columns:
        target = "registry_expert_probability" if column == "expert_probability" else f"registry_{column}"
        values[target] = pd.to_numeric(external_id.map(evidence[column]), errors="coerce").to_numpy(dtype=np.float32)
    previous_snapshot = PREVIOUS[snapshot]
    if previous_snapshot is not None:
        previous = _load(projected, previous_snapshot, ["studies", "facilities", "design_outcomes"])
        previous_studies = previous["studies"].drop_duplicates("nct_id").set_index(previous["studies"].drop_duplicates("nct_id")["nct_id"].astype(str))
        previous_facilities = previous["facilities"].groupby("nct_id").size()
        previous_primary = previous["design_outcomes"][previous["design_outcomes"]["outcome_type"].fillna("").str.casefold().eq("primary")].groupby("nct_id").size()
        current_primary = tables["design_outcomes"][tables["design_outcomes"]["outcome_type"].fillna("").str.casefold().eq("primary")].groupby("nct_id").size()
        values["registry_enrollment_revision"] = values["registry_enrollment"] - _numeric(external_id.map(previous_studies["enrollment"]))
        values["registry_facility_count_change"] = values["registry_facility_rows"] - _numeric(external_id.map(previous_facilities))
        values["registry_primary_endpoint_change"] = _numeric(external_id.map(current_primary)) - _numeric(external_id.map(previous_primary))
        for column in ["completion_date", "primary_completion_date"]:
            current_date = pd.to_datetime(external_id.map(external_studies[column]), errors="coerce")
            previous_date = pd.to_datetime(external_id.map(previous_studies[column]), errors="coerce")
            values[f"registry_{column}_revision_days"] = (current_date - previous_date).dt.days.to_numpy(dtype=np.float32)
    sponsors = tables["sponsors"].copy()
    leads = sponsors[sponsors["lead_or_collaborator"].fillna("").str.casefold().str.contains("lead")]
    lead_map = leads.drop_duplicates("nct_id").set_index("nct_id")["name"]
    study_frame = tables["studies"].copy()
    completion = pd.to_datetime(study_frame["primary_completion_date"], errors="coerce")
    posted = pd.to_datetime(study_frame["results_first_posted_date"], errors="coerce")
    study_frame["report_delay"] = (posted - completion).dt.days
    study_frame["lead_name"] = study_frame["nct_id"].map(lead_map)
    sponsor_delay = study_frame.groupby("lead_name")["report_delay"].agg(["mean", "median", "count"])
    current_lead = external_id.map(lead_map)
    for column in sponsor_delay:
        values[f"registry_sponsor_delay_{column}"] = _numeric(current_lead.map(sponsor_delay[column]))
    frame = pd.DataFrame(values, index=current["pair_row"].to_numpy()).sort_index()
    for column in CATEGORICAL:
        frame[column] = frame[column].fillna("__missing__").astype("category")
    report = {
        "snapshot": snapshot,
        "origin": str(origin),
        "rows": int(len(current)),
        "linked": int(current["linked"].fillna(False).sum()),
        "coverage": float(current["linked"].fillna(False).mean()),
        "direct_positive": int(np.nansum(values["registry_direct_positive"])),
        "direct_negative": int(np.nansum(values["registry_direct_negative"])),
    }
    return frame, report


# Public builder

def build_registry_matrix(db: Any, pairs: pd.DataFrame, cache: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    projected = cache / "registry_clock_lane0" / "projected"
    available, audit = verify_snapshots(projected)
    matrix = pd.DataFrame(index=np.arange(len(pairs)))
    if not available:
        matrix["registry_linked"] = np.nan
        matrix["registry_expert_probability"] = np.nan
        return matrix, {"state": "missing", "audit": audit}
    local_studies = db.table_dict["studies"].df
    reports = {}
    pieces = []
    for origin, snapshot in SNAPSHOTS.items():
        current = pairs[pairs["timestamp"].eq(origin)][["pair_row", "nct_id", "timestamp"]]
        if not len(current):
            continue
        frame, report = snapshot_features(local_studies, current, projected, snapshot, origin)
        pieces.append(frame)
        reports[str(origin)] = report
    if pieces:
        observed = pd.concat(pieces).sort_index()
        matrix = matrix.join(observed, how="left")
    content = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    return matrix, {"state": "ready", "audit": audit, "reports": reports, "content_sha256": content, "version": VERSION}


def cached_registry_matrix(db: Any, pairs: pd.DataFrame, cache: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    key_payload = f"{VERSION}|{len(pairs)}|{pairs['timestamp'].min()}|{pairs['timestamp'].max()}"
    key = hashlib.sha256(key_payload.encode()).hexdigest()[:16]
    root = cache / "registry_clock_lane0" / "features" / f"{VERSION}_{key}"
    matrix_path = root / "matrix.parquet"
    report_path = root / "report.json"
    if matrix_path.exists() and report_path.exists():
        return pd.read_parquet(matrix_path), json.loads(report_path.read_text())
    matrix, report = build_registry_matrix(db, pairs, cache)
    if report.get("state") == "ready":
        root.mkdir(parents=True, exist_ok=True)
        matrix.to_parquet(matrix_path, index=False)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return matrix, report
