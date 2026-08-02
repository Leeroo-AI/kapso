from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ID = "thomas-sounack/BioClinical-ModernBERT-base"
MODEL_REVISION = "c3648aa87af95837c809e6f0c5f85d08160db437"
FEATURE_VERSION = "clinical_longctx_relational_v2"
RELATION_NAMES = ("sponsor", "facility", "condition", "agency", "country")
PRIOR_NAMES = tuple(f"prior_{name}" for name in RELATION_NAMES)
FIELD_SPECS = (
    ("TITLE", "title", 300),
    ("OFFICIAL_TITLE", "official_title", 420),
    ("SUMMARY", "summary", 1250),
    ("CONDITIONS", "conditions", 420),
    ("INTERVENTIONS", "interventions", 300),
    ("LEAD_SPONSOR", "lead_sponsor", 240),
    ("FACILITIES", "facilities", 320),
    ("STUDY_TYPE", "study_type", 80),
    ("PHASE", "phase", 100),
    ("DESIGN", "design", 460),
    ("RESULTS", "results", 460),
    ("ELIGIBILITY", "eligibility", 260),
    ("DETAIL", "detail", 1500),
)
CAT_COLUMNS = (
    "phase",
    "study_type",
    "sampling_method",
    "gender",
    "healthy_volunteers",
    "gender_based",
    "agency_class",
    "primary_country",
)
NUMERIC_COLUMNS = (
    "enrollment",
    "log_enrollment",
    "sponsor_count",
    "facility_count",
    "condition_count",
    "country_count",
    "us_facility_count",
    "non_us_facility_count",
)


@dataclass(frozen=True)
class RuntimeConfig:
    debug: bool
    max_length: int
    micro_batch: int
    inference_batch: int
    cat_iterations: int
    training_deadline: float
    namespace: str


@dataclass
class PreparedData:
    seed_frame: pd.DataFrame
    base_features: pd.DataFrame
    relations: dict[str, tuple[np.ndarray, np.ndarray, int]]
    tokens: np.ndarray
    lengths: np.ndarray
    cache_dir: Path
    n_train: int
    n_val: int
    n_test: int


def elapsed(start: float, phase: str) -> None:
    print(f"[time] {phase}: {time.time() - start:.1f}s", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def content_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()[:20]


def atomic_json(path: Path, value: dict | list) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def register_artifact(shared_dir: Path, name: str, path: Path, description: str, content_key: str, rebuild_hint: str) -> None:
    registry = shared_dir / "artifacts.json"
    lock_path = shared_dir / "artifacts.lock"
    shared_dir.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            records = json.loads(registry.read_text()) if registry.exists() else []
            record = {
                "name": name,
                "path": str(path.relative_to(shared_dir)),
                "description": description,
                "content_key": content_key,
                "rebuild_hint": rebuild_hint,
            }
            if not any(item.get("name") == name and item.get("content_key") == content_key for item in records):
                records.append(record)
                atomic_json(registry, records)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def ensure_pretrained_model(shared_dir: Path) -> Path:
    model_dir = shared_dir / "models" / f"bioclinical-modernbert-{MODEL_REVISION[:17]}"
    required = model_dir / "model.safetensors"
    if not required.exists():
        from huggingface_hub import snapshot_download

        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=model_dir)
    register_artifact(
        shared_dir,
        "BioClinical ModernBERT base",
        model_dir,
        "Pinned pretrained encoder snapshot",
        MODEL_REVISION,
        f"snapshot_download {MODEL_ID} at revision {MODEL_REVISION}",
    )
    return model_dir


def build_seed_frame(ctx) -> pd.DataFrame:
    frames = []
    for split, table in (("train", ctx.train), ("val", ctx.val), ("test", ctx.test)):
        frame = table.df[["id", "date"]].copy()
        frame["split"] = split
        frames.append(frame)
    seed = pd.concat(frames, ignore_index=True)
    seed["row_number"] = np.arange(len(seed), dtype=np.int32)
    eligibility = ctx.db.table_dict["eligibilities"].df[
        ["id", "nct_id", "sampling_method", "gender", "healthy_volunteers", "gender_based"]
    ]
    seed = seed.merge(eligibility, on="id", how="left", sort=False, validate="one_to_one")
    seed = seed.sort_values("row_number", kind="stable").reset_index(drop=True)
    if seed["nct_id"].isna().any() or seed["id"].duplicated().any():
        raise RuntimeError("Seed-to-eligibility alignment failed")
    return seed


def safe_links(frame: pd.DataFrame, seed_dates: pd.Series) -> pd.DataFrame:
    dates = frame["nct_id"].map(seed_dates)
    return frame[dates.notna() & frame["date"].le(dates)].copy()


def joined_values(frame: pd.DataFrame, key: str, value: str, limit: int | None = None) -> pd.Series:
    subset = frame[[key, value]].dropna().drop_duplicates()
    if limit is not None:
        subset = subset.groupby(key, sort=False).head(limit)
    return subset.groupby(key, sort=False)[value].agg("; ".join)


def make_csr(frame: pd.DataFrame, entity_col: str, seed_nct: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    subset = frame[["nct_id", entity_col]].dropna().drop_duplicates()
    row_lookup = pd.Series(np.arange(len(seed_nct), dtype=np.int32), index=seed_nct)
    rows = subset["nct_id"].map(row_lookup)
    subset = subset.loc[rows.notna()].copy()
    row_values = rows.loc[rows.notna()].to_numpy(dtype=np.int32)
    codes, uniques = pd.factorize(subset[entity_col], sort=True)
    order = np.argsort(row_values, kind="stable")
    row_values = row_values[order]
    codes = codes.astype(np.int32, copy=False)[order]
    counts = np.bincount(row_values, minlength=len(seed_nct))
    offsets = np.empty(len(seed_nct) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets, codes, int(len(uniques))


def save_relations(path: Path, relations: dict[str, tuple[np.ndarray, np.ndarray, int]]) -> None:
    values = {}
    for name, (offsets, entities, size) in relations.items():
        values[f"{name}_offsets"] = offsets
        values[f"{name}_values"] = entities
        values[f"{name}_size"] = np.asarray([size], dtype=np.int64)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def load_relations(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    archive = np.load(path, allow_pickle=False)
    return {
        name: (
            archive[f"{name}_offsets"],
            archive[f"{name}_values"],
            int(archive[f"{name}_size"][0]),
        )
        for name in RELATION_NAMES
    }


def build_relational_cache(ctx, seed: pd.DataFrame, cache_dir: Path) -> tuple[Path, Path, Path]:
    meta_path = cache_dir / "relational_meta.json"
    base_path = cache_dir / "base_features.parquet"
    fields_path = cache_dir / "document_fields.parquet"
    relation_path = cache_dir / "relations.npz"
    expected = {
        "version": FEATURE_VERSION,
        "rows": len(seed),
        "id_hash": content_hash(seed["id"].to_numpy(dtype=np.int64)),
    }
    if meta_path.exists() and base_path.exists() and fields_path.exists() and relation_path.exists():
        if json.loads(meta_path.read_text()) == expected:
            print("[cache] relational features hit", flush=True)
            return base_path, fields_path, relation_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    db = ctx.db.table_dict
    seed_dates = pd.Series(seed["date"].to_numpy(), index=seed["nct_id"].to_numpy())
    studies = db["studies"].df[
        [
            "nct_id",
            "brief_title",
            "official_title",
            "brief_summaries",
            "detailed_descriptions",
            "study_type",
            "phase",
            "enrollment",
        ]
    ].copy()
    merged = seed.merge(studies, on="nct_id", how="left", sort=False, validate="many_to_one")
    merged = merged.sort_values("row_number", kind="stable").reset_index(drop=True)

    condition_links = safe_links(db["conditions_studies"].df, seed_dates)
    condition_names = db["conditions"].df.set_index("condition_id")["mesh_term"]
    condition_links["term"] = condition_links["condition_id"].map(condition_names)
    condition_text = joined_values(condition_links, "nct_id", "term")
    condition_count = condition_links.groupby("nct_id", sort=False).size()

    intervention_links = safe_links(db["interventions_studies"].df, seed_dates)
    intervention_names = db["interventions"].df.set_index("intervention_id")["mesh_term"]
    intervention_links["term"] = intervention_links["intervention_id"].map(intervention_names)
    intervention_text = joined_values(intervention_links, "nct_id", "term")

    sponsor_links = safe_links(db["sponsors_studies"].df, seed_dates)
    sponsors = db["sponsors"].df.set_index("sponsor_id")
    sponsor_links["sponsor_name"] = sponsor_links["sponsor_id"].map(sponsors["name"])
    sponsor_links["agency"] = sponsor_links["sponsor_id"].map(sponsors["agency_class"])
    lead_mask = sponsor_links["lead_or_collaborator"].fillna("").astype(str).str.lower().str.contains("lead")
    lead_text = joined_values(sponsor_links[lead_mask], "nct_id", "sponsor_name", 4)
    sponsor_count = sponsor_links.groupby("nct_id", sort=False).size()
    agency_text = joined_values(sponsor_links, "nct_id", "agency", 4)

    facility_links = safe_links(db["facilities_studies"].df, seed_dates)
    facilities = db["facilities"].df.set_index("facility_id")
    facility_links["facility_name"] = facility_links["facility_id"].map(facilities["name"])
    facility_links["city"] = facility_links["facility_id"].map(facilities["city"])
    facility_links["country"] = facility_links["facility_id"].map(facilities["country"])
    facility_links["facility_piece"] = (
        facility_links["facility_name"].fillna("").astype(str)
        + ", "
        + facility_links["city"].fillna("").astype(str)
        + ", "
        + facility_links["country"].fillna("").astype(str)
    )
    facility_text = joined_values(facility_links, "nct_id", "facility_piece", 4)
    facility_count = facility_links.groupby("nct_id", sort=False).size()
    country_count = facility_links.groupby("nct_id", sort=False)["country"].nunique()
    primary_country = facility_links.dropna(subset=["country"]).groupby("nct_id", sort=False)["country"].first()
    us_count = facility_links[facility_links["country"].eq("United States")].groupby("nct_id", sort=False).size()

    designs = safe_links(db["designs"].df, seed_dates)
    design_columns = [
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
    design_values = designs[design_columns].fillna("").astype(str)
    designs["design_text"] = design_values.apply(
        lambda row: "; ".join(f"{column}={row[column]}" for column in design_columns if row[column]),
        axis=1,
    )
    design_text = joined_values(designs, "nct_id", "design_text", 2)

    result_parts = []
    outcomes = safe_links(db["outcomes"].df, seed_dates)
    outcome_columns = ["outcome_type", "title", "description", "time_frame", "population", "units"]
    outcome_values = outcomes[outcome_columns].fillna("").astype(str)
    outcomes["result_text"] = outcome_values.apply(
        lambda row: "; ".join(f"{column}={row[column]}" for column in outcome_columns if row[column]),
        axis=1,
    )
    result_parts.append(outcomes[["nct_id", "result_text"]])
    withdrawals = safe_links(db["drop_withdrawals"].df, seed_dates)
    withdrawal_columns = ["period", "reason", "count"]
    withdrawal_values = withdrawals[withdrawal_columns].fillna("").astype(str)
    withdrawals["result_text"] = withdrawal_values.apply(
        lambda row: "withdrawal: " + "; ".join(f"{column}={row[column]}" for column in withdrawal_columns if row[column]),
        axis=1,
    )
    result_parts.append(withdrawals[["nct_id", "result_text"]])
    events = safe_links(db["reported_event_totals"].df, seed_dates)
    event_columns = ["event_type", "classification", "subjects_affected", "subjects_at_risk"]
    event_values = events[event_columns].fillna("").astype(str)
    events["result_text"] = event_values.apply(
        lambda row: "event: " + "; ".join(f"{column}={row[column]}" for column in event_columns if row[column]),
        axis=1,
    )
    result_parts.append(events[["nct_id", "result_text"]])
    analyses = safe_links(db["outcome_analyses"].df, seed_dates)
    analysis_columns = [
        "non_inferiority_type",
        "param_type",
        "param_value",
        "p_value_modifier",
        "p_value",
        "method",
        "method_description",
        "estimate_description",
    ]
    analysis_values = analyses[analysis_columns].fillna("").astype(str)
    analyses["result_text"] = analysis_values.apply(
        lambda row: "analysis: " + "; ".join(f"{column}={row[column]}" for column in analysis_columns if row[column]),
        axis=1,
    )
    result_parts.append(analyses[["nct_id", "result_text"]])
    result_rows = pd.concat(result_parts, ignore_index=True)
    result_text = joined_values(result_rows, "nct_id", "result_text", 4)

    nct = merged["nct_id"]
    eligibility_text = (
        "sampling_method="
        + merged["sampling_method"].fillna("UNKNOWN").astype(str)
        + "; gender="
        + merged["gender"].fillna("UNKNOWN").astype(str)
        + "; healthy_volunteers="
        + merged["healthy_volunteers"].fillna("UNKNOWN").astype(str)
        + "; gender_based="
        + merged["gender_based"].fillna("UNKNOWN").astype(str)
    )
    document_fields = pd.DataFrame(
        {
            "title": merged["brief_title"],
            "official_title": merged["official_title"],
            "summary": merged["brief_summaries"],
            "detail": merged["detailed_descriptions"],
            "conditions": nct.map(condition_text),
            "interventions": nct.map(intervention_text),
            "lead_sponsor": nct.map(lead_text),
            "facilities": nct.map(facility_text),
            "study_type": merged["study_type"],
            "phase": merged["phase"],
            "design": nct.map(design_text),
            "results": nct.map(result_text),
            "eligibility": eligibility_text,
        }
    ).fillna("")

    enrollment = pd.to_numeric(merged["enrollment"], errors="coerce").fillna(0.0).clip(lower=0.0)
    base = pd.DataFrame(
        {
            "enrollment": enrollment,
            "log_enrollment": np.log1p(enrollment),
            "phase": merged["phase"],
            "study_type": merged["study_type"],
            "sampling_method": merged["sampling_method"],
            "gender": merged["gender"],
            "healthy_volunteers": merged["healthy_volunteers"],
            "gender_based": merged["gender_based"],
            "sponsor_count": nct.map(sponsor_count).fillna(0),
            "facility_count": nct.map(facility_count).fillna(0),
            "condition_count": nct.map(condition_count).fillna(0),
            "country_count": nct.map(country_count).fillna(0),
            "us_facility_count": nct.map(us_count).fillna(0),
            "agency_class": nct.map(agency_text),
            "primary_country": nct.map(primary_country),
            "detail_length": document_fields["detail"].str.len(),
        }
    )
    base["non_us_facility_count"] = (base["facility_count"] - base["us_facility_count"]).clip(lower=0)
    pediatric_pattern = r"child|pediatric|paediatric|infant|neonat|adolescent|juvenile|youth|toddler|newborn"
    marker = np.zeros(len(base), dtype=bool)
    for column in ("title", "official_title", "summary", "detail", "conditions"):
        marker |= document_fields[column].str.contains(pediatric_pattern, case=False, regex=True, na=False).to_numpy()
    base["pediatric_marker"] = marker.astype(np.int8)
    for column in CAT_COLUMNS:
        base[column] = base[column].fillna("__MISSING__").astype(str)
    for column in NUMERIC_COLUMNS:
        base[column] = pd.to_numeric(base[column], errors="coerce").fillna(0.0).astype(np.float32)

    relations = {
        "sponsor": make_csr(sponsor_links, "sponsor_id", seed["nct_id"].to_numpy()),
        "facility": make_csr(facility_links, "facility_id", seed["nct_id"].to_numpy()),
        "condition": make_csr(condition_links, "condition_id", seed["nct_id"].to_numpy()),
        "agency": make_csr(sponsor_links, "agency", seed["nct_id"].to_numpy()),
        "country": make_csr(facility_links, "country", seed["nct_id"].to_numpy()),
    }

    for frame, path in ((base, base_path), (document_fields, fields_path)):
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(path)
    save_relations(relation_path, relations)
    atomic_json(meta_path, expected)
    result_coverage = {
        "outcomes": int(outcomes["nct_id"].nunique()),
        "drop_withdrawals": int(withdrawals["nct_id"].nunique()),
        "reported_event_totals": int(events["nct_id"].nunique()),
        "outcome_analyses": int(analyses["nct_id"].nunique()),
    }
    print(f"[data] seed-time result study counts {result_coverage}", flush=True)
    return base_path, fields_path, relation_path


def pack_documents(frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    documents = []
    boundaries = np.zeros((len(frame), len(FIELD_SPECS), 2), dtype=np.int32)
    records = frame.to_dict("records")
    for row_index, record in enumerate(records):
        pieces = []
        cursor = 0
        for field_index, (label, column, quota) in enumerate(FIELD_SPECS):
            value = str(record[column])[:quota].strip()
            if not value:
                value = "UNKNOWN"
            prefix = f"{label}: "
            piece = prefix + value + "\n"
            start = cursor + len(prefix)
            end = start + len(value)
            boundaries[row_index, field_index] = (start, end)
            pieces.append(piece)
            cursor += len(piece)
        documents.append("".join(pieces))
    return documents, boundaries


def build_token_cache(fields_path: Path, cache_dir: Path, model_dir: Path, max_length: int) -> tuple[Path, Path, Path]:
    token_path = cache_dir / f"tokens_{max_length}.npy"
    length_path = cache_dir / f"lengths_{max_length}.npy"
    boundary_path = cache_dir / f"field_boundaries_{max_length}.npy"
    meta_path = cache_dir / f"tokens_{max_length}.json"
    if token_path.exists() and length_path.exists() and boundary_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("revision") == MODEL_REVISION and meta.get("feature_version") == FEATURE_VERSION:
            print(f"[cache] token IDs hit at max_length={max_length}", flush=True)
            return token_path, length_path, boundary_path

    from transformers import AutoTokenizer

    os.environ.setdefault("RAYON_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "11"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=MODEL_REVISION, use_fast=True)
    fields = pd.read_parquet(fields_path)
    row_count = len(fields)
    temporary_token = token_path.with_name(token_path.name + f".tmp.{os.getpid()}")
    temporary_length = length_path.with_name(length_path.name + f".tmp.{os.getpid()}")
    temporary_boundary = boundary_path.with_name(boundary_path.name + f".tmp.{os.getpid()}")
    token_array = np.lib.format.open_memmap(temporary_token, mode="w+", dtype=np.uint16, shape=(row_count, max_length))
    length_array = np.lib.format.open_memmap(temporary_length, mode="w+", dtype=np.uint16, shape=(row_count,))
    boundary_array = np.lib.format.open_memmap(
        temporary_boundary,
        mode="w+",
        dtype=np.uint16,
        shape=(row_count, len(FIELD_SPECS), 2),
    )
    batch_size = 768
    started = time.time()
    for start in range(0, row_count, batch_size):
        end = min(row_count, start + batch_size)
        documents, char_boundaries = pack_documents(fields.iloc[start:end])
        encoded = tokenizer(
            documents,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        if ids.max(initial=0) >= np.iinfo(np.uint16).max:
            raise RuntimeError("Tokenizer vocabulary does not fit uint16 cache")
        masks = np.asarray(encoded["attention_mask"], dtype=np.uint8)
        offsets = np.asarray(encoded["offset_mapping"], dtype=np.int32)
        token_array[start:end] = ids.astype(np.uint16)
        lengths = masks.sum(axis=1).astype(np.uint16)
        length_array[start:end] = lengths
        for local in range(end - start):
            valid = int(lengths[local])
            token_offsets = offsets[local, :valid]
            for field_index in range(len(FIELD_SPECS)):
                left, right = char_boundaries[local, field_index]
                begins = np.flatnonzero(token_offsets[:, 1] > left)
                ends = np.flatnonzero(token_offsets[:, 0] < right)
                first = int(begins[0]) if len(begins) else valid
                last = int(ends[-1] + 1) if len(ends) else first
                boundary_array[start + local, field_index] = (min(first, max_length), min(last, max_length))
        if end == row_count or end % 24576 < batch_size:
            rate = end / max(time.time() - started, 1e-6)
            print(f"[tokenize] {end}/{row_count} rows at {rate:.1f} rows/s", flush=True)
    token_array.flush()
    length_array.flush()
    boundary_array.flush()
    del token_array, length_array, boundary_array, fields
    gc.collect()
    temporary_token.replace(token_path)
    temporary_length.replace(length_path)
    temporary_boundary.replace(boundary_path)
    atomic_json(
        meta_path,
        {
            "revision": MODEL_REVISION,
            "feature_version": FEATURE_VERSION,
            "max_length": max_length,
            "rows": row_count,
            "boundary_unit": "token",
        },
    )
    return token_path, length_path, boundary_path


def prepare_data(ctx, shared_dir: Path, config: RuntimeConfig) -> PreparedData:
    seed = build_seed_frame(ctx)
    cache_dir = shared_dir / config.namespace / FEATURE_VERSION
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_path, fields_path, relation_path = build_relational_cache(ctx, seed, cache_dir)
    model_dir = ensure_pretrained_model(shared_dir)
    token_path, length_path, boundary_path = build_token_cache(
        fields_path, cache_dir, model_dir, config.max_length
    )
    base = pd.read_parquet(base_path)
    relations = load_relations(relation_path)
    tokens = np.load(token_path, mmap_mode="r", allow_pickle=False)
    lengths = np.load(length_path, mmap_mode="r", allow_pickle=False)
    n_train = int((seed["split"] == "train").sum())
    n_val = int((seed["split"] == "val").sum())
    n_test = int((seed["split"] == "test").sum())
    if tokens.shape != (len(seed), config.max_length) or len(base) != len(seed):
        raise RuntimeError("Cached relational artifacts are not aligned with task rows")
    register_artifact(
        shared_dir,
        f"{config.namespace} relational features and {config.max_length}-token IDs",
        cache_dir,
        "Temporally censored documents, token IDs, field boundaries, structured features, and CSR relations",
        f"{FEATURE_VERSION}:{MODEL_REVISION}:{config.max_length}:{content_hash(seed['id'].to_numpy(dtype=np.int64))}",
        "Delete the namespaced cache directory and rerun main.py",
    )
    print(
        f"[data] aligned rows train={n_train} val={n_val} test={n_test} max_length={config.max_length}",
        flush=True,
    )
    print(
        f"[data] token lengths mean={float(lengths.mean()):.1f} p90={float(np.quantile(lengths, 0.9)):.0f} full={float((lengths == config.max_length).mean()):.3f}",
        flush=True,
    )
    if not boundary_path.exists():
        raise RuntimeError("Field boundary cache was not created")
    return PreparedData(seed, base, relations, tokens, lengths, cache_dir, n_train, n_val, n_test)


def _numba_functions():
    from numba import njit

    @njit(cache=True)
    def online(offsets, entities, entity_count, labels, date_days):
        rows = len(labels)
        result = np.empty(rows, dtype=np.float32)
        sums = np.zeros(entity_count, dtype=np.float64)
        counts = np.zeros(entity_count, dtype=np.int64)
        global_sum = 9.0
        global_count = 10.0
        start = 0
        while start < rows:
            end = start + 1
            while end < rows and date_days[end] == date_days[start]:
                end += 1
            global_rate = global_sum / global_count
            for row in range(start, end):
                left = offsets[row]
                right = offsets[row + 1]
                if left == right:
                    result[row] = global_rate
                else:
                    total = 0.0
                    for position in range(left, right):
                        entity = entities[position]
                        total += (sums[entity] + 20.0 * global_rate) / (counts[entity] + 20.0)
                    result[row] = total / (right - left)
            for row in range(start, end):
                target = labels[row]
                global_sum += target
                global_count += 1.0
                for position in range(offsets[row], offsets[row + 1]):
                    entity = entities[position]
                    sums[entity] += target
                    counts[entity] += 1
            start = end
        return result

    @njit(cache=True)
    def frozen(offsets, entities, entity_count, labels, fit_rows, prediction_rows):
        sums = np.zeros(entity_count, dtype=np.float64)
        counts = np.zeros(entity_count, dtype=np.int64)
        global_sum = 9.0
        global_count = 10.0
        for position in range(len(fit_rows)):
            row = fit_rows[position]
            target = labels[position]
            global_sum += target
            global_count += 1.0
            for member_position in range(offsets[row], offsets[row + 1]):
                entity = entities[member_position]
                sums[entity] += target
                counts[entity] += 1
        global_rate = global_sum / global_count
        result = np.empty(len(prediction_rows), dtype=np.float32)
        for output_position in range(len(prediction_rows)):
            row = prediction_rows[output_position]
            left = offsets[row]
            right = offsets[row + 1]
            if left == right:
                result[output_position] = global_rate
            else:
                total = 0.0
                for member_position in range(left, right):
                    entity = entities[member_position]
                    total += (sums[entity] + 20.0 * global_rate) / (counts[entity] + 20.0)
                result[output_position] = total / (right - left)
        return result

    return online, frozen


def online_priors(
    relations: dict[str, tuple[np.ndarray, np.ndarray, int]],
    labels: np.ndarray,
    date_days: np.ndarray,
    cache_path: Path | None = None,
) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if cached.shape == (len(labels), len(RELATION_NAMES)):
            return cached
    online, _ = _numba_functions()
    result = np.empty((len(labels), len(RELATION_NAMES)), dtype=np.float32)
    for column, name in enumerate(RELATION_NAMES):
        offsets, entities, size = relations[name]
        result[:, column] = online(offsets, entities, size, labels.astype(np.float32), date_days)
    if cache_path is not None:
        temporary = cache_path.with_name(cache_path.name + f".tmp.{os.getpid()}")
        with temporary.open("wb") as handle:
            np.save(handle, result)
        temporary.replace(cache_path)
    return result


def frozen_priors(
    relations: dict[str, tuple[np.ndarray, np.ndarray, int]],
    fit_rows: np.ndarray,
    fit_labels: np.ndarray,
    prediction_rows: np.ndarray,
) -> np.ndarray:
    _, frozen = _numba_functions()
    result = np.empty((len(prediction_rows), len(RELATION_NAMES)), dtype=np.float32)
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    prediction_rows = np.asarray(prediction_rows, dtype=np.int64)
    fit_labels = np.asarray(fit_labels, dtype=np.float32)
    for column, name in enumerate(RELATION_NAMES):
        offsets, entities, size = relations[name]
        result[:, column] = frozen(offsets, entities, size, fit_labels, fit_rows, prediction_rows)
    return result


def structured_frame(base: pd.DataFrame, rows: np.ndarray, priors: np.ndarray) -> pd.DataFrame:
    frame = base.iloc[np.asarray(rows, dtype=np.int64)][list(CAT_COLUMNS) + list(NUMERIC_COLUMNS)].reset_index(drop=True).copy()
    for column, name in enumerate(PRIOR_NAMES):
        frame[name] = priors[:, column]
    return frame


def train_catboost(
    features: pd.DataFrame,
    labels: np.ndarray,
    iterations: int,
    seed: int,
    model_path: Path,
):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=0.04,
        l2_leaf_reg=8.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        thread_count=int(os.environ.get("OMP_NUM_THREADS", "11")),
        verbose=False,
        allow_writing_files=False,
        random_strength=1.0,
    )
    model.fit(features, labels.astype(np.int8), cat_features=list(CAT_COLUMNS), verbose=False)
    temporary = model_path.with_name(model_path.name + f".tmp.{os.getpid()}")
    model.save_model(temporary)
    temporary.replace(model_path)
    return model


def load_catboost(model_path: Path):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def catboost_predict(model, features: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)


def fit_internal_catboost(
    data: PreparedData,
    train_labels: np.ndarray,
    train_online: np.ndarray,
    years: np.ndarray,
    fit_pool: np.ndarray,
    config: RuntimeConfig,
) -> dict[int, np.ndarray]:
    predictions = {}
    mode_dir = data.cache_dir / ("debug_models" if config.debug else "full_models")
    mode_dir.mkdir(parents=True, exist_ok=True)
    for year in (2017, 2018, 2019):
        prediction_rows = np.flatnonzero(years == year).astype(np.int64)
        fit_rows = fit_pool[years[fit_pool] < year]
        if config.debug and len(fit_rows) > 18000:
            rng = np.random.default_rng(1337 + year)
            fit_rows = np.sort(rng.choice(fit_rows, 18000, replace=False))
        prediction_path = mode_dir / f"cat_forward_{year}.npy"
        model_path = mode_dir / f"cat_forward_{year}.cbm"
        if prediction_path.exists() and len(np.load(prediction_path, mmap_mode="r")) == len(prediction_rows):
            predictions[year] = np.load(prediction_path, allow_pickle=False)
            print(f"[cache] CatBoost forward {year} predictions hit", flush=True)
            continue
        prediction_priors = frozen_priors(
            data.relations,
            fit_rows,
            train_labels[fit_rows],
            prediction_rows,
        )
        fit_features = structured_frame(data.base_features, fit_rows, train_online[fit_rows])
        prediction_features = structured_frame(data.base_features, prediction_rows, prediction_priors)
        model = train_catboost(
            fit_features,
            train_labels[fit_rows],
            config.cat_iterations,
            1337 + year,
            model_path,
        )
        predictions[year] = catboost_predict(model, prediction_features)
        np.save(prediction_path, predictions[year])
        print(f"[catboost] forward {year} fit={len(fit_rows)} score_rows={len(prediction_rows)}", flush=True)
        del model, fit_features, prediction_features
        gc.collect()
    return predictions


def fit_catboost_model_a(
    data: PreparedData,
    train_labels: np.ndarray,
    train_online: np.ndarray,
    fit_rows: np.ndarray,
    config: RuntimeConfig,
) -> np.ndarray:
    mode_dir = data.cache_dir / ("debug_models" if config.debug else "full_models")
    prediction_path = mode_dir / "cat_model_a_val.npy"
    model_path = mode_dir / "cat_model_a.cbm"
    if prediction_path.exists() and len(np.load(prediction_path, mmap_mode="r")) == data.n_val:
        print("[cache] CatBoost Model A validation predictions hit", flush=True)
        return np.load(prediction_path, allow_pickle=False)
    val_rows = np.arange(data.n_train, data.n_train + data.n_val, dtype=np.int64)
    val_priors = frozen_priors(data.relations, fit_rows, train_labels[fit_rows], val_rows)
    fit_features = structured_frame(data.base_features, fit_rows, train_online[fit_rows])
    val_features = structured_frame(data.base_features, val_rows, val_priors)
    model = train_catboost(fit_features, train_labels[fit_rows], config.cat_iterations, 2337, model_path)
    prediction = catboost_predict(model, val_features)
    np.save(prediction_path, prediction)
    print(f"[catboost] Model A fit={len(fit_rows)} val={len(val_rows)}", flush=True)
    return prediction


def fit_catboost_model_b(
    data: PreparedData,
    combined_labels: np.ndarray,
    combined_online: np.ndarray,
    config: RuntimeConfig,
) -> np.ndarray:
    mode_dir = data.cache_dir / ("debug_models" if config.debug else "full_models")
    prediction_path = mode_dir / "cat_model_b_test.npy"
    model_path = mode_dir / "cat_model_b.cbm"
    if prediction_path.exists() and len(np.load(prediction_path, mmap_mode="r")) == data.n_test:
        print("[cache] CatBoost Model B test predictions hit", flush=True)
        return np.load(prediction_path, allow_pickle=False)
    fit_rows = np.arange(data.n_train + data.n_val, dtype=np.int64)
    if config.debug:
        train_recent = np.arange(max(0, data.n_train - 18000), data.n_train, dtype=np.int64)
        val_rows = np.arange(data.n_train, data.n_train + data.n_val, dtype=np.int64)
        fit_rows = np.concatenate([train_recent, val_rows])
    test_rows = np.arange(data.n_train + data.n_val, len(data.seed_frame), dtype=np.int64)
    test_priors = frozen_priors(data.relations, fit_rows, combined_labels[fit_rows], test_rows)
    fit_features = structured_frame(data.base_features, fit_rows, combined_online[fit_rows])
    test_features = structured_frame(data.base_features, test_rows, test_priors)
    model = train_catboost(fit_features, combined_labels[fit_rows], config.cat_iterations, 3337, model_path)
    prediction = catboost_predict(model, test_features)
    np.save(prediction_path, prediction)
    print(f"[catboost] Model B fit={len(fit_rows)} test={len(test_rows)}", flush=True)
    return prediction


def model_source_ready(path: Path) -> bool:
    return path.is_dir() and (path / "model.safetensors").exists() and (path / "stage.json").exists()


def load_encoder(source: Path, initial: bool, debug: bool):
    import torch
    from transformers import AutoModelForSequenceClassification, logging as transformers_logging

    transformers_logging.set_verbosity_error()
    kwargs = {"attn_implementation": "sdpa"}
    if initial:
        kwargs.update({"num_labels": 1, "ignore_mismatched_sizes": True})
    model = AutoModelForSequenceClassification.from_pretrained(source, **kwargs)
    model.config.problem_type = None
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if debug:
        backbone = getattr(model, "model", None)
        if backbone is not None and hasattr(backbone, "embeddings"):
            for parameter in backbone.embeddings.parameters():
                parameter.requires_grad = False
        layers = getattr(backbone, "layers", []) if backbone is not None else []
        freeze_count = max(0, len(layers) - 6)
        for layer in layers[:freeze_count]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
        print(f"[encoder] debug froze embeddings and {freeze_count}/{len(layers)} layers", flush=True)
    model.to(torch.device("cuda"))
    return model


def save_encoder(model, path: Path, metadata: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True, max_shard_size="2GB")
    atomic_json(temporary / "stage.json", metadata)
    if path.exists():
        shutil.rmtree(path)
    temporary.replace(path)


def make_training_batches(
    rows: np.ndarray,
    lengths: np.ndarray,
    micro_batch: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(rows, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    blocks = []
    block_size = max(1024, micro_batch * 64)
    for start in range(0, len(shuffled), block_size):
        block = shuffled[start : start + block_size]
        order = np.argsort(lengths[block], kind="stable")
        block = block[order]
        if rng.random() < 0.5:
            block = block[::-1]
        blocks.extend(block[offset : offset + micro_batch] for offset in range(0, len(block), micro_batch))
    rng.shuffle(blocks)
    return blocks


def optimizer_for(model, base_lr: float, head_lr: float):
    import torch

    grouped = {("base", "decay"): [], ("base", "nodecay"): [], ("head", "decay"): [], ("head", "nodecay"): []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        section = "head" if name.startswith("head.") or name.startswith("classifier.") else "base"
        decay = "nodecay" if parameter.ndim == 1 or name.endswith(".bias") else "decay"
        grouped[(section, decay)].append(parameter)
    groups = []
    for section in ("base", "head"):
        for decay in ("decay", "nodecay"):
            parameters = grouped[(section, decay)]
            if parameters:
                groups.append(
                    {
                        "params": parameters,
                        "lr": base_lr if section == "base" else head_lr,
                        "weight_decay": 0.01 if decay == "decay" else 0.0,
                    }
                )
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, fused=True)


def stage_scheduler(optimizer, total_steps: int):
    import torch

    warmup_steps = max(1, int(math.ceil(0.05 * total_steps)))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def train_encoder_stage(
    model,
    tokens: np.ndarray,
    lengths: np.ndarray,
    rows: np.ndarray,
    global_labels: np.ndarray,
    config: RuntimeConfig,
    base_lr: float,
    head_lr: float,
    seed: int,
    maximum_steps: int | None,
    stage_name: str,
) -> dict:
    import torch
    import torch.nn.functional as functional

    accumulation = max(1, 32 // config.micro_batch)
    rows = np.asarray(rows, dtype=np.int64)
    if maximum_steps is not None and len(rows) > maximum_steps * 32:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(rows, maximum_steps * 32, replace=False))
    batches = make_training_batches(rows, lengths, config.micro_batch, seed)
    total_steps = max(1, math.ceil(len(batches) / accumulation))
    if maximum_steps is not None:
        total_steps = min(total_steps, maximum_steps)
        batches = batches[: total_steps * accumulation]
    optimizer = optimizer_for(model, base_lr, head_lr)
    scheduler = stage_scheduler(optimizer, total_steps)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.time()
    examples = 0
    token_count = 0
    optimizer_steps = 0
    loss_total = 0.0
    completed = True
    for batch_number, batch_rows in enumerate(batches):
        sequence_length = int(lengths[batch_rows].max())
        sequence_length = min(tokens.shape[1], max(16, int(math.ceil(sequence_length / 16) * 16)))
        input_ids = torch.from_numpy(np.asarray(tokens[batch_rows, :sequence_length], dtype=np.int64)).to("cuda")
        batch_lengths = torch.from_numpy(np.asarray(lengths[batch_rows], dtype=np.int64)).to("cuda")
        attention_mask = torch.arange(sequence_length, device="cuda").unsqueeze(0) < batch_lengths.unsqueeze(1)
        labels = torch.from_numpy(np.asarray(global_labels[batch_rows], dtype=np.float32)).to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            loss = functional.binary_cross_entropy_with_logits(logits.float(), labels)
            scaled_loss = loss / accumulation
        scaled_loss.backward()
        loss_total += float(loss.detach()) * len(batch_rows)
        examples += len(batch_rows)
        token_count += int(batch_lengths.sum().item())
        update = (batch_number + 1) % accumulation == 0 or batch_number + 1 == len(batches)
        if update:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if optimizer_steps == 1 or optimizer_steps % 250 == 0 or optimizer_steps == total_steps:
                duration = max(time.time() - started, 1e-6)
                print(
                    f"[encoder] {stage_name} step={optimizer_steps}/{total_steps} loss={loss_total / max(examples, 1):.5f} examples_per_min={60.0 * examples / duration:.1f} tokens_per_s={token_count / duration:.1f}",
                    flush=True,
                )
            if time.time() >= config.training_deadline:
                completed = optimizer_steps >= total_steps
                print(f"[encoder] {stage_name} reached optimization freeze deadline", flush=True)
                break
        del input_ids, batch_lengths, attention_mask, labels, logits, loss, scaled_loss
    duration = time.time() - started
    del optimizer, scheduler
    torch.cuda.empty_cache()
    return {
        "stage": stage_name,
        "complete": bool(completed and optimizer_steps >= total_steps),
        "optimizer_steps": optimizer_steps,
        "planned_steps": total_steps,
        "examples": examples,
        "seconds": duration,
        "examples_per_minute": 60.0 * examples / max(duration, 1e-6),
        "tokens_per_second": token_count / max(duration, 1e-6),
    }


def predict_encoder(
    model,
    tokens: np.ndarray,
    lengths: np.ndarray,
    rows: np.ndarray,
    batch_size: int,
    label: str,
) -> np.ndarray:
    import torch

    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(lengths[rows], kind="stable")
    sorted_rows = rows[order]
    sorted_predictions = np.empty(len(rows), dtype=np.float64)
    model.eval()
    started = time.time()
    cursor = 0
    active_batch = batch_size
    while cursor < len(sorted_rows):
        batch_rows = sorted_rows[cursor : cursor + active_batch]
        sequence_length = int(lengths[batch_rows].max())
        sequence_length = min(tokens.shape[1], max(16, int(math.ceil(sequence_length / 16) * 16)))
        try:
            input_ids = torch.from_numpy(np.asarray(tokens[batch_rows, :sequence_length], dtype=np.int64)).to("cuda")
            batch_lengths = torch.from_numpy(np.asarray(lengths[batch_rows], dtype=np.int64)).to("cuda")
            attention_mask = torch.arange(sequence_length, device="cuda").unsqueeze(0) < batch_lengths.unsqueeze(1)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                values = torch.sigmoid(logits.float()).cpu().numpy()
            sorted_predictions[cursor : cursor + len(batch_rows)] = values
            cursor += len(batch_rows)
            del input_ids, batch_lengths, attention_mask, logits, values
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if active_batch <= 1:
                raise
            active_batch = max(1, active_batch // 2)
            print(f"[encoder] {label} reduced inference batch to {active_batch} after CUDA OOM", flush=True)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    prediction = sorted_predictions[inverse]
    print(
        f"[encoder] {label} inferred {len(rows)} rows in {time.time() - started:.1f}s batch={active_batch}",
        flush=True,
    )
    return prediction


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def blend_probabilities(encoder: np.ndarray, companion: np.ndarray, encoder_weight: float) -> np.ndarray:
    values = encoder_weight * logit(encoder) + (1.0 - encoder_weight) * logit(companion)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def internal_diagnostics(
    encoder_predictions: dict[int, np.ndarray],
    cat_predictions: dict[int, np.ndarray],
    labels: np.ndarray,
    years: np.ndarray,
    base: pd.DataFrame,
) -> tuple[float, dict]:
    from sklearn.metrics import roc_auc_score

    all_encoder = []
    all_cat = []
    all_labels = []
    all_rows = []
    diagnostics = {"years": {}, "blend_candidates": {}, "strata": {}}
    for year in (2017, 2018, 2019):
        rows = np.flatnonzero(years == year)
        encoder = encoder_predictions[year]
        companion = cat_predictions[year]
        target = labels[rows]
        diagnostics["years"][str(year)] = {
            "count": int(len(rows)),
            "positive_rate": float(target.mean()),
            "encoder_auc": float(roc_auc_score(target, encoder)),
            "catboost_auc": float(roc_auc_score(target, companion)),
        }
        all_encoder.append(encoder)
        all_cat.append(companion)
        all_labels.append(target)
        all_rows.append(rows)
    encoder = np.concatenate(all_encoder)
    companion = np.concatenate(all_cat)
    target = np.concatenate(all_labels)
    rows = np.concatenate(all_rows)
    best_weight = 0.80
    best_score = -np.inf
    for weight in (0.65, 0.80, 0.90):
        blended = blend_probabilities(encoder, companion, weight)
        score = float(roc_auc_score(target, blended))
        diagnostics["blend_candidates"][str(weight)] = score
        if score > best_score:
            best_score = score
            best_weight = weight
    strata = {
        "pediatric_marker": base.iloc[rows]["pediatric_marker"].to_numpy() == 1,
        "no_pediatric_marker": base.iloc[rows]["pediatric_marker"].to_numpy() == 0,
        "detail_missing": base.iloc[rows]["detail_length"].to_numpy() == 0,
        "detail_long": base.iloc[rows]["detail_length"].to_numpy() >= 3000,
    }
    selected = blend_probabilities(encoder, companion, best_weight)
    for name, mask in strata.items():
        stratum_target = target[mask]
        record = {"count": int(mask.sum()), "positive_rate": float(stratum_target.mean()) if mask.any() else None}
        if len(np.unique(stratum_target)) == 2:
            record["roc_auc"] = float(roc_auc_score(stratum_target, selected[mask]))
        diagnostics["strata"][name] = record
    diagnostics["selected_encoder_weight"] = best_weight
    diagnostics["selected_internal_auc"] = best_score
    print(f"[internal] {json.dumps(diagnostics, sort_keys=True)}", flush=True)
    return best_weight, diagnostics


def replay_rows(new_rows: np.ndarray, earlier_rows: np.ndarray, seed: int) -> np.ndarray:
    if len(new_rows) == 0 or len(earlier_rows) == 0:
        return np.asarray(new_rows, dtype=np.int64)
    replay_count = min(len(earlier_rows), int(round(0.20 * len(new_rows))))
    rng = np.random.default_rng(seed)
    replay = rng.choice(earlier_rows, replay_count, replace=False)
    return np.concatenate([np.asarray(new_rows, dtype=np.int64), replay.astype(np.int64)])


def curriculum_pool(years: np.ndarray, debug: bool) -> np.ndarray:
    rows = np.arange(len(years), dtype=np.int64)
    if not debug:
        return rows
    recent = rows[years >= 2015]
    if len(recent) <= 18000:
        return recent
    rng = np.random.default_rng(1337)
    selected = []
    for year in range(2015, 2020):
        year_rows = recent[years[recent] == year]
        quota = min(len(year_rows), 3600)
        selected.append(rng.choice(year_rows, quota, replace=False))
    return np.sort(np.concatenate(selected).astype(np.int64))


def run_encoder_curriculum(
    data: PreparedData,
    train_labels: np.ndarray,
    years: np.ndarray,
    model_dir: Path,
    config: RuntimeConfig,
    fit_pool: np.ndarray,
) -> tuple[dict[int, np.ndarray], Path, list[dict]]:
    mode_dir = data.cache_dir / ("debug_models" if config.debug else "full_models")
    mode_dir.mkdir(parents=True, exist_ok=True)
    stages = (
        ("through_2016", fit_pool[years[fit_pool] <= 2016], 2017, 140 if config.debug else None),
        (
            "continue_2017",
            replay_rows(fit_pool[years[fit_pool] == 2017], fit_pool[years[fit_pool] <= 2016], 2017),
            2018,
            45 if config.debug else None,
        ),
        (
            "continue_2018",
            replay_rows(fit_pool[years[fit_pool] == 2018], fit_pool[years[fit_pool] <= 2017], 2018),
            2019,
            45 if config.debug else None,
        ),
    )
    source = model_dir
    initial = True
    model = None
    predictions = {}
    stage_records = []
    for stage_index, (stage_name, stage_rows, score_year, maximum_steps) in enumerate(stages):
        checkpoint = mode_dir / f"encoder_{stage_name}"
        prediction_path = mode_dir / f"encoder_forward_{score_year}.npy"
        reusable = model_source_ready(checkpoint) and prediction_path.exists()
        if reusable:
            metadata = json.loads((checkpoint / "stage.json").read_text())
            expected_rows = int((years == score_year).sum())
            reusable = bool(metadata.get("complete")) and len(np.load(prediction_path, mmap_mode="r")) == expected_rows
        if reusable:
            source = checkpoint
            initial = False
            predictions[score_year] = np.load(prediction_path, allow_pickle=False)
            stage_records.append(metadata)
            print(f"[cache] encoder {stage_name} and forward {score_year} hit", flush=True)
            continue
        if model is None:
            model = load_encoder(source, initial=initial, debug=config.debug)
        record = train_encoder_stage(
            model,
            data.tokens,
            data.lengths,
            stage_rows,
            train_labels,
            config,
            2e-5,
            1e-4,
            1337 + stage_index,
            maximum_steps,
            stage_name,
        )
        save_encoder(model, checkpoint, record)
        score_rows = np.flatnonzero(years == score_year).astype(np.int64)
        predictions[score_year] = predict_encoder(
            model,
            data.tokens,
            data.lengths,
            score_rows,
            config.inference_batch,
            f"forward_{score_year}",
        )
        np.save(prediction_path, predictions[score_year])
        stage_records.append(record)
        source = checkpoint
        initial = False
        if not record["complete"]:
            break
    if model is not None:
        del model
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    return predictions, source, stage_records


def build_model_a(
    data: PreparedData,
    train_labels: np.ndarray,
    years: np.ndarray,
    model_source: Path,
    model_dir: Path,
    config: RuntimeConfig,
    fit_pool: np.ndarray,
) -> tuple[np.ndarray, Path, dict]:
    mode_dir = data.cache_dir / ("debug_models" if config.debug else "full_models")
    checkpoint = mode_dir / "encoder_model_a"
    prediction_path = mode_dir / "encoder_model_a_val.npy"
    if model_source_ready(checkpoint) and prediction_path.exists():
        metadata = json.loads((checkpoint / "stage.json").read_text())
        if metadata.get("complete") and len(np.load(prediction_path, mmap_mode="r")) == data.n_val:
            print("[cache] encoder Model A validation vector hit", flush=True)
            return np.load(prediction_path, allow_pickle=False), checkpoint, metadata
    model = load_encoder(model_source, initial=model_source == model_dir, debug=config.debug)
    new_rows = fit_pool[years[fit_pool] == 2019]
    earlier = fit_pool[years[fit_pool] <= 2018]
    rows = replay_rows(new_rows, earlier, 2019)
    record = train_encoder_stage(
        model,
        data.tokens,
        data.lengths,
        rows,
        train_labels,
        config,
        2e-5,
        1e-4,
        1437,
        30 if config.debug else None,
        "model_a_2019_replay",
    )
    save_encoder(model, checkpoint, record)
    val_rows = np.arange(data.n_train, data.n_train + data.n_val, dtype=np.int64)
    prediction = predict_encoder(
        model,
        data.tokens,
        data.lengths,
        val_rows,
        config.inference_batch,
        "model_a_validation",
    )
    np.save(prediction_path, prediction)
    print(f"[freeze] Model A validation vector frozen sha={content_hash(prediction)}", flush=True)
    del model
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    return prediction, checkpoint, record


def build_model_b(
    data: PreparedData,
    combined_labels: np.ndarray,
    years: np.ndarray,
    model_a_source: Path,
    config: RuntimeConfig,
) -> tuple[np.ndarray, Path, dict]:
    mode_dir = data.cache_dir / ("debug_models" if config.debug else "full_models")
    checkpoint = mode_dir / "encoder_model_b"
    prediction_path = mode_dir / "encoder_model_b_test.npy"
    if model_source_ready(checkpoint) and prediction_path.exists():
        metadata = json.loads((checkpoint / "stage.json").read_text())
        if metadata.get("complete") and len(np.load(prediction_path, mmap_mode="r")) == data.n_test:
            print("[cache] encoder Model B test vector hit", flush=True)
            return np.load(prediction_path, allow_pickle=False), checkpoint, metadata
    model = load_encoder(model_a_source, initial=False, debug=config.debug)
    recent_train = np.flatnonzero(years >= 2018).astype(np.int64)
    val_rows = np.arange(data.n_train, data.n_train + data.n_val, dtype=np.int64)
    adaptation_pool = np.concatenate([recent_train, val_rows])
    rng = np.random.default_rng(2337)
    adaptation_count = max(32, int(round(0.20 * len(adaptation_pool))))
    adaptation_rows = rng.choice(adaptation_pool, adaptation_count, replace=False)
    if time.time() < config.training_deadline:
        record = train_encoder_stage(
            model,
            data.tokens,
            data.lengths,
            adaptation_rows,
            combined_labels,
            config,
            5e-6,
            2.5e-5,
            2337,
            40 if config.debug else None,
            "model_b_validation_adaptation",
        )
    else:
        record = {
            "stage": "model_b_validation_adaptation",
            "complete": False,
            "optimizer_steps": 0,
            "planned_steps": int(math.ceil(len(adaptation_rows) / 32)),
            "examples": 0,
            "seconds": 0.0,
            "examples_per_minute": 0.0,
            "tokens_per_second": 0.0,
        }
    save_encoder(model, checkpoint, record)
    test_rows = np.arange(data.n_train + data.n_val, len(data.seed_frame), dtype=np.int64)
    prediction = predict_encoder(
        model,
        data.tokens,
        data.lengths,
        test_rows,
        config.inference_batch,
        "model_b_test",
    )
    np.save(prediction_path, prediction)
    del model
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    return prediction, checkpoint, record
