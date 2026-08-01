from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


MODEL_ID = "ncbi/MedCPT-Article-Encoder"
MODEL_REVISION = "d05a736da4bb84ee4057b7f7999485be6ed85465"
DOCUMENT_VERSION = "medcpt_dmc_v2"


@dataclass
class ClinicalBundle:
    features: pd.DataFrame
    categorical: list[str]
    groups: dict[str, list[list[Any]]]
    signatures: dict[str, tuple[np.ndarray, np.ndarray]]
    documents: dict[str, np.ndarray]
    ids: np.ndarray
    dates: np.ndarray


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _category(series: pd.Series) -> pd.Series:
    return series.fillna("__MISSING__").astype(str)


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(np.float32)


def _age_years(value: Any) -> float:
    text = _text(value).strip().lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(year|month|week|day|hour|minute)", text)
    if not match:
        return np.nan
    value_float = float(match.group(1))
    unit = match.group(2)
    scales = {"year": 1.0, "month": 1 / 12, "week": 1 / 52, "day": 1 / 365, "hour": 1 / 8760, "minute": 1 / 525600}
    return value_float * scales[unit]


def _safe_relation(db, name: str, ids: set[int], seed_dates: dict[int, pd.Timestamp]) -> pd.DataFrame:
    frame = db.table_dict[name].df
    frame = frame.loc[frame["nct_id"].isin(ids)].copy()
    if "date" in frame.columns:
        limits = frame["nct_id"].map(seed_dates)
        frame = frame.loc[frame["date"].isna() | (frame["date"] <= limits)]
    return frame


def _group_lists(frame: pd.DataFrame, key: str, ordered_ids: np.ndarray) -> list[list[Any]]:
    grouped = frame.dropna(subset=[key]).groupby("nct_id", sort=False)[key].agg(lambda x: list(dict.fromkeys(x.tolist())))
    mapping = grouped.to_dict()
    return [mapping.get(int(identifier), []) for identifier in ordered_ids]


def _pair_lists(left: list[list[Any]], right: list[list[Any]], limit: int = 24) -> list[list[Any]]:
    output = []
    for a_values, b_values in zip(left, right):
        pairs = []
        for a_value in a_values[:8]:
            for b_value in b_values[:8]:
                pairs.append((a_value, b_value))
                if len(pairs) >= limit:
                    break
            if len(pairs) >= limit:
                break
        output.append(pairs)
    return output


def _signatures(values: list[list[Any]]) -> tuple[np.ndarray, np.ndarray]:
    first = np.zeros(len(values), dtype=np.uint64)
    second = np.zeros(len(values), dtype=np.uint64)
    for row, keys in enumerate(values):
        a = 0
        b = 0
        for key in keys:
            digest = hashlib.blake2b(repr(key).encode("utf-8", "ignore"), digest_size=16).digest()
            x = int.from_bytes(digest[:8], "little")
            y = int.from_bytes(digest[8:], "little")
            a |= 1 << (x % 64)
            b |= 1 << (y % 64)
        first[row] = a
        second[row] = b
    return first, second


def _visible_result_features(features: pd.DataFrame, db, ids: set[int], seed_dates: dict[int, pd.Timestamp], ordered_ids: np.ndarray) -> None:
    specifications = {
        "outcomes": ["id"],
        "outcome_analyses": ["p_value", "param_value", "ci_percent"],
        "reported_event_totals": ["subjects_affected", "subjects_at_risk"],
        "drop_withdrawals": ["count"],
    }
    for table_name, numeric_columns in specifications.items():
        frame = _safe_relation(db, table_name, ids, seed_dates)
        grouped = frame.groupby("nct_id", sort=False)
        counts = grouped.size().reindex(ordered_ids).fillna(0)
        features[f"{table_name}_visible_count"] = counts.to_numpy(dtype=np.float32)
        for column in numeric_columns:
            if column == "id" or column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            temporary = pd.DataFrame({"nct_id": frame["nct_id"], "value": values})
            aggregates = temporary.groupby("nct_id")["value"].agg(["mean", "sum", "max"])
            for aggregation in ["mean", "sum", "max"]:
                features[f"{table_name}_{column}_{aggregation}"] = aggregates[aggregation].reindex(ordered_ids).to_numpy(dtype=np.float32)


def build_clinical_bundle(db, rows: pd.DataFrame) -> ClinicalBundle:
    started = time.time()
    ordered_ids = rows["nct_id"].astype(np.int64).to_numpy()
    dates = pd.to_datetime(rows["start_date"]).to_numpy()
    ids = set(map(int, ordered_ids))
    seed_dates = dict(zip(map(int, ordered_ids), pd.to_datetime(rows["start_date"])))
    features = pd.DataFrame(index=np.arange(len(rows)))
    categorical: list[str] = []

    studies = db.table_dict["studies"].df.drop_duplicates("nct_id", keep="last").set_index("nct_id").reindex(ordered_ids)
    study_categories = [
        "study_type", "phase", "enrollment_type", "source", "source_class", "is_fda_regulated_drug",
        "is_fda_regulated_device", "is_unapproved_device", "is_us_export", "plan_to_share_ipd",
        "biospec_retention",
    ]
    study_numbers = ["enrollment", "number_of_arms", "number_of_groups"]
    for column in study_categories:
        features[column] = _category(studies[column]).to_numpy()
        categorical.append(column)
    for column in study_numbers:
        features[column] = _number(studies[column]).to_numpy()
        features[f"{column}_missing"] = studies[column].isna().to_numpy(dtype=np.float32)
    features["log_enrollment"] = np.log1p(np.maximum(features["enrollment"].fillna(0), 0)).astype(np.float32)
    timestamp = pd.to_datetime(rows["start_date"])
    features["year"] = timestamp.dt.year.to_numpy(dtype=np.float32)
    features["month"] = timestamp.dt.month.to_numpy(dtype=np.float32)
    features["day_of_year"] = timestamp.dt.dayofyear.to_numpy(dtype=np.float32)
    features["year_fraction"] = features["year"] + (features["day_of_year"] - 1) / 365.25

    designs = _safe_relation(db, "designs", ids, seed_dates).drop_duplicates("nct_id", keep="last").set_index("nct_id").reindex(ordered_ids)
    design_categories = [
        "allocation", "intervention_model", "observational_model", "primary_purpose", "time_perspective",
        "masking", "subject_masked", "caregiver_masked", "investigator_masked", "outcomes_assessor_masked",
    ]
    for column in design_categories:
        name = f"design_{column}"
        features[name] = _category(designs[column]).to_numpy()
        categorical.append(name)
    for column in ["masking_description", "intervention_model_description"]:
        text_values = designs[column].fillna("").astype(str)
        features[f"design_{column}_length"] = text_values.str.len().to_numpy(dtype=np.float32)

    eligibility = _safe_relation(db, "eligibilities", ids, seed_dates).drop_duplicates("nct_id", keep="last").set_index("nct_id").reindex(ordered_ids)
    eligibility_categories = ["sampling_method", "gender", "healthy_volunteers", "gender_based", "adult", "child", "older_adult"]
    for column in eligibility_categories:
        name = f"eligibility_{column}"
        features[name] = _category(eligibility[column]).to_numpy()
        categorical.append(name)
    features["minimum_age_years"] = eligibility["minimum_age"].map(_age_years).to_numpy(dtype=np.float32)
    features["maximum_age_years"] = eligibility["maximum_age"].map(_age_years).to_numpy(dtype=np.float32)
    for column in ["criteria", "population", "gender_description"]:
        values = eligibility[column].fillna("").astype(str)
        features[f"eligibility_{column}_length"] = values.str.len().to_numpy(dtype=np.float32)
        features[f"eligibility_{column}_words"] = values.str.count(r"\S+").to_numpy(dtype=np.float32)
    print(f"[clinical] core rows={len(rows)} elapsed={time.time() - started:.1f}s", flush=True)

    sponsor_links = _safe_relation(db, "sponsors_studies", ids, seed_dates)
    sponsor_frame = db.table_dict["sponsors"].df.drop_duplicates("sponsor_id")
    sponsor_agencies = dict(zip(sponsor_frame["sponsor_id"], sponsor_frame["agency_class"]))
    sponsor_names = dict(zip(sponsor_frame["sponsor_id"], sponsor_frame["name"]))
    lead_links = sponsor_links.loc[sponsor_links["lead_or_collaborator"].astype(str).str.lower().eq("lead")]
    collaborator_links = sponsor_links.loc[~sponsor_links.index.isin(lead_links.index)]
    lead_sponsor = _group_lists(lead_links, "sponsor_id", ordered_ids)
    collaborator_sponsor = _group_lists(collaborator_links, "sponsor_id", ordered_ids)
    sponsor_all = [list(dict.fromkeys(a + b)) for a, b in zip(lead_sponsor, collaborator_sponsor)]
    lead_agency = [[_text(sponsor_agencies.get(values[0]))] if values else [] for values in lead_sponsor]
    lead_names = [_text(sponsor_names.get(values[0])) if values else "" for values in lead_sponsor]

    condition_links = _safe_relation(db, "conditions_studies", ids, seed_dates)
    intervention_links = _safe_relation(db, "interventions_studies", ids, seed_dates)
    facility_links = _safe_relation(db, "facilities_studies", ids, seed_dates)
    conditions = _group_lists(condition_links, "condition_id", ordered_ids)
    interventions = _group_lists(intervention_links, "intervention_id", ordered_ids)
    facilities = _group_lists(facility_links, "facility_id", ordered_ids)

    facility_meta = db.table_dict["facilities"].df.drop_duplicates("facility_id")
    facility_countries = dict(zip(facility_meta["facility_id"], facility_meta["country"]))
    facility_states = dict(zip(facility_meta["facility_id"], facility_meta["state"]))
    countries = []
    states = []
    for values in facilities:
        countries.append(list(dict.fromkeys(text for value in values if (text := _text(facility_countries.get(value))))))
        states.append(list(dict.fromkeys(text for value in values if (text := _text(facility_states.get(value))))))

    condition_frame = db.table_dict["conditions"].df.drop_duplicates("condition_id")
    intervention_frame = db.table_dict["interventions"].df.drop_duplicates("intervention_id")
    condition_meta = dict(zip(condition_frame["condition_id"], condition_frame["mesh_term"]))
    intervention_meta = dict(zip(intervention_frame["intervention_id"], intervention_frame["mesh_term"]))
    condition_terms = [list(dict.fromkeys(text for value in values if (text := _text(condition_meta.get(value))))) for values in conditions]
    intervention_terms = [list(dict.fromkeys(text for value in values if (text := _text(intervention_meta.get(value))))) for values in interventions]
    print(f"[clinical] relations rows={len(rows)} elapsed={time.time() - started:.1f}s", flush=True)

    phase_keys = [[value] for value in features["phase"].tolist()]
    type_keys = [[value] for value in features["study_type"].tolist()]
    source_keys = [[value] for value in features["source_class"].tolist()]
    sponsor_phase = _pair_lists(lead_sponsor, phase_keys)
    sponsor_condition = _pair_lists(lead_sponsor, conditions)
    agency_phase = _pair_lists(lead_agency, phase_keys)
    condition_intervention = _pair_lists(conditions, interventions)

    groups = {
        "lead_sponsor": lead_sponsor,
        "collaborator_sponsor": collaborator_sponsor,
        "agency_class": lead_agency,
        "source_class": source_keys,
        "condition": conditions,
        "intervention": interventions,
        "facility": facilities,
        "country": countries,
        "state": states,
        "sponsor_phase": sponsor_phase,
        "sponsor_condition": sponsor_condition,
        "agency_phase": agency_phase,
        "condition_intervention": condition_intervention,
    }

    for name, values in [
        ("lead_sponsor", lead_sponsor), ("collaborator_sponsor", collaborator_sponsor),
        ("condition", conditions), ("intervention", interventions), ("facility", facilities),
        ("country", countries), ("state", states),
    ]:
        features[f"{name}_count"] = np.asarray([len(x) for x in values], dtype=np.float32)
        primary_name = f"primary_{name}"
        features[primary_name] = [str(x[0]) if x else "__MISSING__" for x in values]
        categorical.append(primary_name)
    features["lead_agency_class"] = [str(x[0]) if x else "__MISSING__" for x in lead_agency]
    categorical.append("lead_agency_class")
    features["site_country_diversity"] = features["country_count"] / np.maximum(features["facility_count"], 1)
    features["has_us_site"] = np.asarray([any(value.lower() in {"united states", "united states of america", "usa"} for value in values) for values in countries], dtype=np.float32)

    title = np.asarray([_text(o) if _text(o) else _text(b) for o, b in zip(studies["official_title"], studies["brief_title"])], dtype=object)
    summary = studies["brief_summaries"].fillna("").astype(str).to_numpy(dtype=object)
    detailed = studies["detailed_descriptions"].fillna("").astype(str).to_numpy(dtype=object)
    criteria = eligibility["criteria"].fillna("").astype(str).to_numpy(dtype=object)
    for name, values in [("title", title), ("summary", summary), ("detailed", detailed), ("criteria", criteria)]:
        lengths = np.asarray([len(value) for value in values], dtype=np.float32)
        features[f"text_{name}_length"] = lengths
        features[f"text_{name}_missing"] = (lengths == 0).astype(np.float32)
    monitoring_tokens = {
        "dmc_acronym": ["dmc", "dsmb", "idmc"],
        "data_monitoring": ["data monitoring", "data-monitoring"],
        "safety_monitoring": ["safety monitoring", "safety-monitoring"],
        "monitoring_committee": ["monitoring committee", "monitoring board", "monitoring panel"],
        "independent_committee": ["independent committee", "independent board", "independent panel"],
        "interim_analysis": ["interim analysis", "interim analyses", "stopping rule", "stopping boundary"],
    }
    for field_name, values in [("title", title), ("summary", summary), ("detailed", detailed), ("criteria", criteria)]:
        lowered = [value.lower() for value in values]
        for signal_name, tokens in monitoring_tokens.items():
            features[f"text_{field_name}_{signal_name}"] = np.asarray([
                sum(value.count(token) for token in tokens) for value in lowered
            ], dtype=np.float32)

    combined = np.asarray([
        " ".join([title_value, summary_value[:1800], detailed_value[:600], criteria_value[:600], " ".join(condition_value), " ".join(intervention_value)]).lower()
        for title_value, summary_value, detailed_value, criteria_value, condition_value, intervention_value in zip(title, summary, detailed, criteria, condition_terms, intervention_terms)
    ], dtype=object)
    token_groups = {
        "dmc_explicit": [" dmc ", "dsmb", "data monitoring committee", "data and safety monitoring", "monitoring board"],
        "monitoring": ["monitoring", "monitored", "oversight", "interim analysis"],
        "randomized": ["randomized", "randomised", "randomization", "randomisation"],
        "placebo": ["placebo"],
        "masked": ["blind", "mask"],
        "safety": ["safety", "adverse event", "toxicity"],
        "mortality": ["mortality", "death", "survival"],
        "cancer": ["cancer", "carcinoma", "tumor", "tumour", "neoplasm", "oncolog", "leukemia", "lymphoma"],
        "cardiovascular": ["cardiac", "cardiovascular", "heart", "coronary", "stroke"],
        "infectious": ["infection", "infectious", "virus", "viral", "bacteria", "covid", "sars-cov"],
        "neurologic": ["neurolog", "alzheimer", "parkinson", "epilep", "dementia"],
        "pediatric": ["pediatric", "paediatric", "child", "infant", "adolescent"],
        "device": ["device", "implant", "surgical", "surgery"],
    }
    for name, tokens in token_groups.items():
        features[f"regex_{name}"] = np.asarray([sum(value.count(token) for token in tokens) for value in combined], dtype=np.float32)
    print(f"[clinical] text rows={len(rows)} elapsed={time.time() - started:.1f}s", flush=True)

    _visible_result_features(features, db, ids, seed_dates, ordered_ids)

    compact = np.asarray([
        f"Phase: {phase}. Study type: {study_type}. Allocation: {allocation}. Masking: {masking}. Purpose: {purpose}. Lead sponsor: {sponsor}. Agency: {agency}. Conditions: {condition}. Interventions: {intervention}."
        for phase, study_type, allocation, masking, purpose, sponsor, agency, condition, intervention in zip(
            features["phase"], features["study_type"], features["design_allocation"], features["design_masking"],
            features["design_primary_purpose"], lead_names, features["lead_agency_class"],
            ["; ".join(x[:8]) for x in condition_terms], ["; ".join(x[:8]) for x in intervention_terms],
        )
    ], dtype=object)
    additional = np.asarray([f"Detailed description: {a[:5000]} Eligibility: {b[:5000]}" for a, b in zip(detailed, criteria)], dtype=object)
    risk = np.asarray([
        f"Clinical trial monitoring profile. {compact_value} Enrollment: {enrollment}. Arms: {arms}. Sites: {sites}. Countries: {country}. Protocol evidence: {evidence[:1800]}"
        for compact_value, enrollment, arms, sites, country, evidence in zip(
            compact, features["enrollment"].fillna(-1), features["number_of_arms"].fillna(-1),
            features["facility_count"], ["; ".join(x[:12]) for x in countries], combined,
        )
    ], dtype=object)
    documents = {
        "title": title,
        "abstract": np.asarray([f"{a} {b}" for a, b in zip(summary, compact)], dtype=object),
        "additional": additional,
        "risk": risk,
    }
    signatures = {
        "lead_sponsor": _signatures(lead_sponsor),
        "condition": _signatures(conditions),
        "intervention": _signatures(interventions),
        "country": _signatures(countries),
        "phase": _signatures(phase_keys),
        "study_type": _signatures(type_keys),
    }
    print(f"[clinical] complete rows={len(rows)} features={features.shape[1]} elapsed={time.time() - started:.1f}s", flush=True)
    return ClinicalBundle(features, categorical, groups, signatures, documents, ordered_ids, dates)
