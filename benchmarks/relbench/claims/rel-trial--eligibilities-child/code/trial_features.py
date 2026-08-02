# Imports

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


# Configuration

FEATURE_VERSION = "lane1_three_slots_v3"
STRUCT_DIM = 96
NUMERIC_DIM = 32
HASH_DIM = STRUCT_DIM - NUMERIC_DIM

ELIGIBILITY_COLUMNS = [
    "id",
    "nct_id",
    "sampling_method",
    "gender",
    "healthy_volunteers",
    "gender_based",
    "date",
]

STUDY_COLUMNS = [
    "nct_id",
    "start_date",
    "target_duration",
    "study_type",
    "acronym",
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
    "is_ppsd",
    "is_us_export",
    "biospec_retention",
    "source_class",
    "plan_to_share_ipd",
    "brief_summaries",
    "detailed_descriptions",
]

DESIGN_COLUMNS = [
    "id",
    "nct_id",
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
    "date",
]


# Text utilities

def clean_text(value: object, limit: int | None = None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if limit is not None:
        text = text[:limit]
    return text


def labeled(label: str, value: object, limit: int | None = None) -> str:
    text = clean_text(value, limit)
    return f"{label}: {text}" if text else ""


def joined(parts: list[str]) -> str:
    return " [SEP] ".join(part for part in parts if part)


def compact_unique(values: list[str]) -> str:
    seen = set()
    kept = []
    for value in values:
        value = clean_text(value)
        if value and value not in seen:
            seen.add(value)
            kept.append(value)
    return " ; ".join(kept)


# Data access

def db_root() -> Path:
    return Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"] / "db"


def read_table(name: str, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(db_root() / f"{name}.parquet", columns=columns)


def censor_relation(frame: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(cutoffs, on="nct_id", how="inner", validate="many_to_one")
    safe = merged.loc[merged["date"] <= merged["seed_date"]].drop(columns="seed_date")
    return safe


def aggregate_items(
    frame: pd.DataFrame,
    sort_columns: list[str],
    item_columns: list[tuple[str, str, int | None]],
    cap: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    ordered = frame.sort_values(["nct_id", *sort_columns], kind="stable")
    counts = ordered.groupby("nct_id", sort=False).size()
    if cap is not None:
        ordered = ordered.loc[ordered.groupby("nct_id", sort=False).cumcount() < cap]
    items = []
    for row in ordered.itertuples(index=False):
        parts = []
        for column, label, limit in item_columns:
            part = labeled(label, getattr(row, column), limit)
            if part:
                parts.append(part)
        items.append(" | ".join(parts))
    item_series = pd.Series(items, index=ordered.index, dtype=object)
    grouped = item_series.groupby(ordered["nct_id"], sort=False).agg(lambda x: compact_unique(x.tolist()))
    return grouped, counts


# Serialization

def build_feature_cache(cache_dir: Path) -> tuple[Path, dict]:
    target = cache_dir / FEATURE_VERSION
    feature_path = target / "features.parquet"
    struct_path = target / "structured.npy"
    metadata_path = target / "metadata.json"
    if feature_path.exists() and struct_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("version") == FEATURE_VERSION and metadata.get("rows") == 273160:
            return target, metadata

    target.mkdir(parents=True, exist_ok=True)
    started = time.time()
    elig = read_table("eligibilities", ELIGIBILITY_COLUMNS).sort_values("id", kind="stable")
    if not np.array_equal(elig["id"].to_numpy(), np.arange(len(elig))):
        raise RuntimeError("Eligibility IDs are not dense and ordered")
    cutoffs = elig[["nct_id", "date"]].rename(columns={"date": "seed_date"})
    studies = read_table("studies", STUDY_COLUMNS)
    base = elig.merge(studies, on="nct_id", how="left", validate="one_to_one")
    future_study = base["start_date"].notna() & (base["start_date"] > base["date"])
    for column in STUDY_COLUMNS:
        if column not in {"nct_id", "start_date"}:
            base.loc[future_study, column] = np.nan

    relation_counts: dict[str, pd.Series] = {}
    concept_parts: dict[str, pd.Series] = {}
    aux_parts: dict[str, pd.Series] = {}

    conditions_links = censor_relation(
        read_table("conditions_studies", ["id", "nct_id", "condition_id", "date"]), cutoffs
    ).merge(read_table("conditions", ["condition_id", "mesh_term"]), on="condition_id", how="left")
    concept_parts["conditions"], relation_counts["conditions"] = aggregate_items(
        conditions_links, ["condition_id", "id"], [("mesh_term", "condition", 180)]
    )

    intervention_links = censor_relation(
        read_table("interventions_studies", ["id", "nct_id", "intervention_id", "date"]), cutoffs
    ).merge(
        read_table("interventions", ["intervention_id", "mesh_term"]),
        on="intervention_id",
        how="left",
    )
    concept_parts["interventions"], relation_counts["interventions"] = aggregate_items(
        intervention_links, ["intervention_id", "id"], [("mesh_term", "intervention", 180)]
    )

    sponsor_links = censor_relation(
        read_table(
            "sponsors_studies", ["id", "nct_id", "sponsor_id", "lead_or_collaborator", "date"]
        ),
        cutoffs,
    ).merge(read_table("sponsors", ["sponsor_id", "name", "agency_class"]), on="sponsor_id", how="left")
    relation_counts["sponsors"] = sponsor_links.groupby("nct_id", sort=False).size()
    sponsor_links = sponsor_links.sort_values(["nct_id", "sponsor_id", "id"], kind="stable")
    lead_mask = sponsor_links["lead_or_collaborator"].fillna("").astype(str).str.lower().str.contains("lead")
    lead_sponsors = sponsor_links.loc[lead_mask].drop_duplicates("nct_id", keep="first")
    missing_leads = sponsor_links.loc[~sponsor_links["nct_id"].isin(lead_sponsors["nct_id"])]
    lead_sponsors = pd.concat([lead_sponsors, missing_leads.drop_duplicates("nct_id", keep="first")])
    lead_sponsors = lead_sponsors.set_index("nct_id")
    concept_parts["sponsors"] = pd.Series(
        [joined([labeled("lead sponsor", n, 240), labeled("agency", a, 80)]) for n, a in zip(lead_sponsors["name"], lead_sponsors["agency_class"])],
        index=lead_sponsors.index,
        dtype=object,
    )

    facility_links = censor_relation(
        read_table("facilities_studies", ["id", "nct_id", "facility_id", "date"]), cutoffs
    ).merge(
        read_table("facilities", ["facility_id", "name", "country"]), on="facility_id", how="left"
    )
    concept_parts["facilities"], relation_counts["facilities"] = aggregate_items(
        facility_links,
        ["facility_id", "id"],
        [("name", "facility", 160), ("country", "country", 80)],
        cap=8,
    )

    designs = censor_relation(read_table("designs", DESIGN_COLUMNS), cutoffs)
    designs = designs.sort_values(["nct_id", "id"], kind="stable").drop_duplicates("nct_id", keep="first")
    designs = designs.drop(columns="date").rename(columns={c: f"design_{c}" for c in DESIGN_COLUMNS if c not in {"nct_id", "date"}})
    base = base.merge(designs, on="nct_id", how="left", validate="one_to_one")

    outcome_columns = ["id", "nct_id", "outcome_type", "title", "date"]
    outcomes = censor_relation(read_table("outcomes", outcome_columns), cutoffs)
    aux_parts["outcomes"], relation_counts["outcomes"] = aggregate_items(
        outcomes, ["id"], [("outcome_type", "type", 60), ("title", "outcome", 240)], cap=10
    )

    analysis_columns = [
        "id",
        "nct_id",
        "non_inferiority_type",
        "param_type",
        "param_value",
        "dispersion_type",
        "dispersion_value",
        "p_value_modifier",
        "p_value",
        "method",
        "date",
    ]
    analyses = censor_relation(read_table("outcome_analyses", analysis_columns), cutoffs)
    aux_parts["analyses"], relation_counts["analyses"] = aggregate_items(
        analyses,
        ["id"],
        [
            ("method", "method", 80),
            ("non_inferiority_type", "noninferiority", 60),
            ("param_type", "parameter", 60),
            ("dispersion_type", "dispersion", 60),
        ],
        cap=4,
    )
    analysis_numeric = analyses.groupby("nct_id", sort=False).agg(
        analysis_param_mean=("param_value", "mean"),
        analysis_p_min=("p_value", "min"),
    )
    base = base.merge(analysis_numeric, left_on="nct_id", right_index=True, how="left")

    withdrawal_columns = ["id", "nct_id", "period", "reason", "count", "date"]
    withdrawals = censor_relation(read_table("drop_withdrawals", withdrawal_columns), cutoffs)
    aux_parts["withdrawals"], relation_counts["withdrawals"] = aggregate_items(
        withdrawals, ["id"], [("period", "period", 60), ("reason", "withdrawal", 160)], cap=6
    )
    withdrawal_numeric = withdrawals.groupby("nct_id", sort=False).agg(withdrawal_total=("count", "sum"))
    base = base.merge(withdrawal_numeric, left_on="nct_id", right_index=True, how="left")

    event_columns = [
        "id",
        "nct_id",
        "event_type",
        "classification",
        "subjects_affected",
        "subjects_at_risk",
        "date",
    ]
    events = censor_relation(read_table("reported_event_totals", event_columns), cutoffs)
    aux_parts["events"], relation_counts["events"] = aggregate_items(
        events, ["id"], [("event_type", "event", 60), ("classification", "class", 100)], cap=6
    )
    event_numeric = events.groupby("nct_id", sort=False).agg(
        event_affected=("subjects_affected", "sum"), event_at_risk=("subjects_at_risk", "sum")
    )
    base = base.merge(event_numeric, left_on="nct_id", right_index=True, how="left")

    nct = base["nct_id"]
    for name, counts in relation_counts.items():
        base[f"count_{name}"] = nct.map(counts).fillna(0).astype(np.float32)
    for name, values in concept_parts.items():
        base[f"concept_{name}"] = nct.map(values).fillna("")
    for name, values in aux_parts.items():
        base[f"aux_{name}"] = nct.map(values).fillna("")

    study_text = []
    concept_text = []
    design_text = []
    design_text_columns = [
        "design_allocation",
        "design_intervention_model",
        "design_observational_model",
        "design_primary_purpose",
        "design_time_perspective",
        "design_masking",
        "design_masking_description",
        "design_intervention_model_description",
        "design_subject_masked",
        "design_caregiver_masked",
        "design_investigator_masked",
        "design_outcomes_assessor_masked",
    ]
    for row in base.itertuples(index=False):
        detail = clean_text(row.detailed_descriptions)
        detail_head = detail[:350]
        detail_tail = detail[-350:] if len(detail) > 350 else ""
        study_text.append(
            joined(
                [
                    labeled("brief title", row.brief_title, 360),
                    labeled("official title", row.official_title, 520),
                    labeled("acronym", row.acronym, 80),
                    labeled("brief summary", row.brief_summaries, 600),
                    labeled("detailed description head", detail_head),
                    labeled("detailed description tail", detail_tail),
                ]
            )
        )
        concept_text.append(
            joined(
                [
                    clean_text(row.concept_conditions),
                    clean_text(row.concept_interventions),
                    clean_text(row.concept_sponsors),
                    clean_text(row.concept_facilities),
                    labeled("study source", row.source, 180),
                ]
            )
        )
        design_parts = [
            labeled("study type", row.study_type, 80),
            labeled("phase", row.phase, 80),
            labeled("enrollment", row.enrollment, 40),
            labeled("sampling method", row.sampling_method, 80),
            labeled("gender", row.gender, 60),
            labeled("healthy volunteers", row.healthy_volunteers, 60),
            labeled("gender based", row.gender_based, 60),
        ]
        for column in design_text_columns:
            design_parts.append(labeled(column.removeprefix("design_").replace("_", " "), getattr(row, column), 240))
        design_parts.extend(
            [
                labeled("analysis count", row.count_analyses, 30),
                clean_text(row.aux_analyses),
                labeled("withdrawal total", row.withdrawal_total, 40),
                clean_text(row.aux_withdrawals),
                labeled("reported event count", row.count_events, 30),
                clean_text(row.aux_events),
                clean_text(row.aux_outcomes),
            ]
        )
        design_text.append(joined(design_parts))

    features = pd.DataFrame(
        {
            "id": base["id"].to_numpy(np.int64),
            "date": base["date"].to_numpy(),
            "study": study_text,
            "concept_org": concept_text,
            "design_aux": design_text,
        }
    )
    structured = build_structured(base, features, relation_counts)
    temporary_features = target / f"features.{os.getpid()}.tmp.parquet"
    temporary_struct = target / f"structured.{os.getpid()}.tmp.npy"
    features.to_parquet(temporary_features, index=False, compression="zstd")
    with temporary_struct.open("wb") as handle:
        np.save(handle, structured)
    os.replace(temporary_features, feature_path)
    os.replace(temporary_struct, struct_path)
    metadata = profile_features(features, base)
    metadata.update(
        {
            "version": FEATURE_VERSION,
            "rows": len(features),
            "elapsed_seconds": round(time.time() - started, 3),
            "temporal_future_study_rows_blank": int(future_study.sum()),
        }
    )
    temporary_metadata = target / f"metadata.{os.getpid()}.tmp.json"
    temporary_metadata.write_text(json.dumps(metadata, indent=2))
    os.replace(temporary_metadata, metadata_path)
    return target, metadata


# Structured fields

def build_structured(base: pd.DataFrame, features: pd.DataFrame, relation_counts: dict[str, pd.Series]) -> np.ndarray:
    rows = len(base)
    matrix = np.zeros((rows, STRUCT_DIM), dtype=np.float32)
    year = base["date"].dt.year.to_numpy(np.float32)
    month = base["date"].dt.month.to_numpy(np.float32)
    matrix[:, 0] = np.clip((year - 2000.0) / 25.0, 0.0, 1.2)
    matrix[:, 1] = np.sin(2.0 * np.pi * month / 12.0)
    matrix[:, 2] = np.cos(2.0 * np.pi * month / 12.0)
    numeric_columns = ["enrollment", "number_of_arms", "number_of_groups"]
    for offset, column in enumerate(numeric_columns, 3):
        matrix[:, offset] = np.log1p(np.maximum(pd.to_numeric(base[column], errors="coerce").fillna(0).to_numpy(np.float32), 0))
    count_names = ["conditions", "interventions", "sponsors", "facilities", "outcomes", "analyses", "withdrawals", "events"]
    for offset, name in enumerate(count_names, 6):
        matrix[:, offset] = np.log1p(base[f"count_{name}"].to_numpy(np.float32))
    extra_numeric = ["analysis_param_mean", "analysis_p_min", "withdrawal_total", "event_affected", "event_at_risk"]
    for offset, column in enumerate(extra_numeric, 14):
        values = pd.to_numeric(base[column], errors="coerce").fillna(0).to_numpy(np.float32)
        matrix[:, offset] = np.sign(values) * np.log1p(np.abs(values))
    matrix[:, 19] = np.log1p(features["study"].str.len().to_numpy(np.float32))
    matrix[:, 20] = np.log1p(features["concept_org"].str.len().to_numpy(np.float32))
    matrix[:, 21] = np.log1p(features["design_aux"].str.len().to_numpy(np.float32))
    matrix[:, 22] = base["sampling_method"].notna().to_numpy(np.float32)
    matrix[:, 23] = base["gender_based"].notna().to_numpy(np.float32)
    matrix[:, 24] = base["detailed_descriptions"].notna().to_numpy(np.float32)
    matrix[:, 25] = base["design_id"].notna().to_numpy(np.float32)
    matrix[:, 26] = (base["count_outcomes"].to_numpy() > 0).astype(np.float32)
    matrix[:, 27] = (base["count_events"].to_numpy() > 0).astype(np.float32)
    risk = pd.to_numeric(base["event_at_risk"], errors="coerce").fillna(0).to_numpy(np.float32)
    affected = pd.to_numeric(base["event_affected"], errors="coerce").fillna(0).to_numpy(np.float32)
    matrix[:, 28] = affected / np.maximum(risk, 1.0)
    matrix[:, 29] = base["official_title"].notna().to_numpy(np.float32)
    matrix[:, 30] = base["phase"].notna().to_numpy(np.float32)
    matrix[:, 31] = 1.0

    categorical_columns = [
        "sampling_method",
        "gender",
        "healthy_volunteers",
        "gender_based",
        "target_duration",
        "study_type",
        "phase",
        "enrollment_type",
        "source_class",
        "has_dmc",
        "is_fda_regulated_drug",
        "is_fda_regulated_device",
        "is_unapproved_device",
        "is_ppsd",
        "is_us_export",
        "biospec_retention",
        "plan_to_share_ipd",
        "design_allocation",
        "design_intervention_model",
        "design_observational_model",
        "design_primary_purpose",
        "design_time_perspective",
        "design_masking",
    ]
    for column in categorical_columns:
        values = base[column].fillna("missing").astype(str).to_numpy()
        for row_index, value in enumerate(values):
            digest = hashlib.blake2b(f"{column}={value}".encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest, "little") % HASH_DIM
            matrix[row_index, NUMERIC_DIM + bucket] += 1.0
    matrix[:, 3:22] = np.clip(matrix[:, 3:22], -12.0, 12.0) / 12.0
    matrix[:, 28] = np.clip(matrix[:, 28], 0.0, 1.0)
    matrix[:, NUMERIC_DIM:] = np.clip(matrix[:, NUMERIC_DIM:], 0.0, 4.0) / 4.0
    return matrix


# Profiling

def profile_features(features: pd.DataFrame, base: pd.DataFrame) -> dict:
    profile: dict[str, object] = {"slot_characters": {}, "relation_presence": {}}
    for column in ["study", "concept_org", "design_aux"]:
        lengths = features[column].str.len()
        profile["slot_characters"][column] = {
            "p50": int(lengths.quantile(0.5)),
            "p90": int(lengths.quantile(0.9)),
            "p95": int(lengths.quantile(0.95)),
            "p99": int(lengths.quantile(0.99)),
            "empty": int((lengths == 0).sum()),
        }
    for name in ["conditions", "interventions", "sponsors", "facilities", "outcomes", "analyses", "withdrawals", "events"]:
        profile["relation_presence"][name] = int((base[f"count_{name}"] > 0).sum())
    return profile


def load_features(cache_dir: Path) -> tuple[pd.DataFrame, np.ndarray, dict, Path]:
    target, metadata = build_feature_cache(cache_dir)
    features = pd.read_parquet(target / "features.parquet")
    structured = np.load(target / "structured.npy", mmap_mode="r")
    if len(features) != structured.shape[0] or structured.shape[1] != STRUCT_DIM:
        raise RuntimeError("Feature cache alignment failure")
    if not np.array_equal(features["id"].to_numpy(), np.arange(len(features))):
        raise RuntimeError("Feature rows are not aligned to eligibility IDs")
    return features, structured, metadata, target
