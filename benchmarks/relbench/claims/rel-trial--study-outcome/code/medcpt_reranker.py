# Imports

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from campaign_io import locked_append, register_artifact
from kapso_datasets.common import shared_cache_dir
from publication_evidence import NCT_PATTERN, _section_payloads, build_trial_contexts, prefilter_candidates


# Configuration

START = time.time()
VERSION = "medcpt_endpoint_reranker_v1"
QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
CROSS_MODEL = "ncbi/MedCPT-Cross-Encoder"
RETRIEVAL_CANDIDATES = 24
RERANKED_DOCUMENTS = 8
MAXIMUM_LENGTH = 512
LEARNING_RATE = 2e-5
EPOCHS = 2
BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
ORIGIN_SNAPSHOTS = {
    "2018-01-01": "2017-12-17",
    "2019-01-01": "2018-12-01",
    "2020-01-01": "2019-12-01",
    "2021-01-01": "2020-12-01",
}


# Runtime

def report(name: str, **values: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"[medcpt] {name} elapsed={time.time() - START:.2f}s {payload}".rstrip(), flush=True)


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n")
    os.replace(temporary, path)


# Documents

def _query_text(context: dict[str, Any]) -> str:
    primary = " ; ".join(
        f"measure={item.get('title', '')} | time_frame={item.get('time_frame', '')}"
        for item in context.get("primary_outcomes", [])
    )
    return "\n".join([
        f"registered title: {context.get('official_title') or context.get('brief_title', '')}",
        f"primary outcomes: {primary}",
        f"interventions: {' ; '.join(context.get('interventions', []))}",
        f"conditions: {' ; '.join(context.get('conditions', []))}",
        f"phase: {context.get('phase', '')}",
        f"enrollment: {context.get('enrollment', '')}",
    ])


def _cached_result_windows(row: pd.Series, cache: Path) -> str:
    pmcid = str(row.get("pmcid", "") or "").upper()
    path = cache / "literature_v3" / "raw" / "full_text_xml" / f"{pmcid}.xml.gz"
    if not pmcid or not path.exists() or path.stat().st_size == 0:
        return ""
    try:
        with gzip.open(path, "rb") as stream:
            payload = stream.read()
        windows, _ = _section_payloads(payload, row)
        return "\n\n".join(windows)
    except Exception:
        return ""


def _article_pair(row: pd.Series, cache: Path) -> list[str]:
    abstract = str(row.get("abstract", "") or "")
    results = _cached_result_windows(row, cache)
    body = "\n\n".join(value for value in [abstract, results] if value)
    return [str(row.get("title", "") or ""), body]


def load_origin(cache: Path, origin: str, linkage: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    path = cache / "literature_v3" / "parsed" / f"{origin}.jsonl"
    records = pd.read_json(path, lines=True)
    records = records[records["date_eligible"].astype(bool)].reset_index(drop=True)
    contexts = build_trial_contexts(linkage, pd.Timestamp(origin), cache / "registry_clock_lane0" / "projected")
    records["query_text"] = records["queried_nct_id"].map(lambda value: _query_text(contexts[str(value)]))
    records["article_pair"] = records.apply(lambda row: _article_pair(row, cache), axis=1)
    records["article_text"] = records["article_pair"].map(lambda values: "\n\n".join(values))
    return records, contexts


# Encoders

def _embedding_key(origin: str, records: pd.DataFrame) -> str:
    digest = hashlib.sha256(f"{VERSION}\0{QUERY_MODEL}\0{ARTICLE_MODEL}\0{origin}".encode())
    for row in records[["queried_nct_id", "content_hash", "query_text", "article_text"]].itertuples(index=False):
        digest.update("\0".join(map(str, row)).encode())
    return digest.hexdigest()[:24]


def _encode_texts(model: AutoModel, tokenizer: AutoTokenizer, values: list[Any], maximum_length: int, batch_size: int, device: torch.device) -> np.ndarray:
    outputs = []
    model.eval().to(device)
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            encoded = tokenizer(
                values[start:start + batch_size],
                truncation=True,
                padding=True,
                max_length=maximum_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = model(**encoded).last_hidden_state[:, 0, :]
            outputs.append(hidden.float().cpu().numpy())
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.vstack(outputs).astype(np.float32)


def bi_encoder_scores(cache: Path, origin: str, records: pd.DataFrame) -> np.ndarray:
    root = cache / VERSION / "embeddings"
    key = _embedding_key(origin, records)
    path = root / f"{origin}_{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if np.array_equal(stored["publication_identity"].astype(str), records["publication_identity"].astype(str).to_numpy()):
                report("embeddings", origin=origin, state="hit", rows=len(records))
                return stored["scores"].astype(np.float64)
    device = _device()
    query_tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL)
    query_model = AutoModel.from_pretrained(QUERY_MODEL)
    article_tokenizer = AutoTokenizer.from_pretrained(ARTICLE_MODEL)
    article_model = AutoModel.from_pretrained(ARTICLE_MODEL)
    unique_queries = records[["queried_nct_id", "query_text"]].drop_duplicates("queried_nct_id").reset_index(drop=True)
    query_embeddings = _encode_texts(query_model, query_tokenizer, unique_queries["query_text"].tolist(), 256, BATCH_SIZE, device)
    article_embeddings = _encode_texts(article_model, article_tokenizer, records["article_pair"].tolist(), MAXIMUM_LENGTH, BATCH_SIZE, device)
    query_map = {nct_id: query_embeddings[index] for index, nct_id in enumerate(unique_queries["queried_nct_id"])}
    aligned_queries = np.vstack(records["queried_nct_id"].map(query_map).to_numpy())
    scores = np.sum(aligned_queries * article_embeddings, axis=1).astype(np.float64)
    root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            publication_identity=records["publication_identity"].astype(str).to_numpy(),
            scores=scores,
            query_embeddings=query_embeddings,
            article_embeddings=article_embeddings,
        )
    os.replace(temporary, path)
    report("embeddings", origin=origin, state="built", rows=len(records), path=path)
    return scores


def retrieve(records: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    result = records.copy()
    result["bi_score"] = scores
    result = result.sort_values(
        ["queried_nct_id", "bi_score", "publication_date", "publication_identity"],
        ascending=[True, False, False, True],
    )
    return result.groupby("queried_nct_id", sort=False).head(RETRIEVAL_CANDIDATES).reset_index(drop=True)


# Relevance training

def _reference_sets(cache: Path, snapshot: str) -> tuple[set[tuple[str, str]], set[tuple[str, str]], pd.DataFrame]:
    references = pd.read_parquet(cache / "registry_clock_lane0" / "projected" / snapshot / "study_references.parquet")
    references["nct_id"] = references["nct_id"].astype(str)
    references["pmid"] = references["pmid"].fillna("").astype(str)
    result_mask = references["reference_type"].fillna("").str.casefold().eq("results_reference")
    background_mask = references["reference_type"].fillna("").str.casefold().eq("reference")
    positive = set(map(tuple, references.loc[result_mask & references["pmid"].str.fullmatch(r"\d+"), ["nct_id", "pmid"]].to_numpy()))
    negative = set(map(tuple, references.loc[background_mask & references["pmid"].str.fullmatch(r"\d+"), ["nct_id", "pmid"]].to_numpy()))
    return positive, negative, references.loc[background_mask, ["nct_id", "citation"]]


def relevance_labels(cache: Path, origin: str, records: pd.DataFrame, contexts: dict[str, dict[str, Any]]) -> pd.DataFrame:
    positive_pairs, negative_pairs, background = _reference_sets(cache, ORIGIN_SNAPSHOTS[origin])
    labels = []
    for index, row in records.iterrows():
        key = (str(row["queried_nct_id"]), str(row.get("pmid", "") or ""))
        mentions = set(value.upper() for value in NCT_PATTERN.findall(f"{row.get('title', '')} {row.get('abstract', '')}"))
        if key in positive_pairs or bool(row.get("registry_result_reference", False)):
            labels.append({"query_nct_id": key[0], "query": row["query_text"], "article": row["article_text"], "label": 1.0, "kind": "result_reference", "record_index": int(index)})
        elif key in negative_pairs:
            labels.append({"query_nct_id": key[0], "query": row["query_text"], "article": row["article_text"], "label": 0.0, "kind": "background_reference", "record_index": int(index)})
        elif len(mentions) > 1:
            labels.append({"query_nct_id": key[0], "query": row["query_text"], "article": row["article_text"], "label": 0.0, "kind": "multi_nct_hard_negative", "record_index": int(index)})
    labeled = pd.DataFrame(labels)
    positive_trials = set(labeled.loc[labeled["label"].eq(1.0), "query_nct_id"])
    background = background[background["nct_id"].isin(positive_trials) & background["citation"].notna()].copy()
    for nct_id, current in background.groupby("nct_id"):
        query = _query_text(contexts[str(nct_id)])
        for citation in current["citation"].astype(str).drop_duplicates().head(12):
            labeled.loc[len(labeled)] = {
                "query_nct_id": str(nct_id), "query": query, "article": citation,
                "label": 0.0, "kind": "background_reference_citation", "record_index": -1,
            }
    result_records = records[records["registry_result_reference"].astype(bool)].copy()
    condition_map = {nct_id: set(context.get("conditions", [])) for nct_id, context in contexts.items()}
    for target in sorted(positive_trials):
        target_conditions = condition_map.get(target, set())
        if not target_conditions:
            continue
        candidates = result_records[
            result_records["queried_nct_id"].astype(str).ne(target)
            & result_records["queried_nct_id"].map(lambda value: bool(target_conditions & condition_map.get(str(value), set())))
            & ~result_records["article_text"].str.contains(target, case=False, regex=False)
        ]
        for _, row in candidates.drop_duplicates("publication_identity").head(4).iterrows():
            labeled.loc[len(labeled)] = {
                "query_nct_id": target, "query": _query_text(contexts[target]), "article": row["article_text"],
                "label": 0.0, "kind": "same_condition_result_reference", "record_index": -1,
            }
    positives = labeled[labeled["label"].eq(1.0)]
    negatives = labeled[labeled["label"].eq(0.0)]
    if positives.empty or negatives.empty:
        raise RuntimeError(f"Verified relevance training labels are incomplete: positive={len(positives)} negative={len(negatives)}")
    maximum_negatives = min(len(negatives), 3 * len(positives))
    negatives = negatives.sample(maximum_negatives, random_state=1337)
    return pd.concat([positives, negatives], ignore_index=True).sample(frac=1.0, random_state=1337).reset_index(drop=True)


def score_cross(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, records: pd.DataFrame, device: torch.device) -> np.ndarray:
    scores = []
    model.eval().to(device)
    pairs = list(zip(records["query_text"].astype(str), records["article_text"].astype(str)))
    with torch.no_grad():
        for start in range(0, len(pairs), BATCH_SIZE):
            encoded = tokenizer(
                pairs[start:start + BATCH_SIZE],
                truncation=True,
                padding=True,
                max_length=MAXIMUM_LENGTH,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**encoded).logits.squeeze(-1)
            scores.append(logits.float().cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


def fit_cross_encoder(training: pd.DataFrame, output: Path) -> tuple[AutoModelForSequenceClassification, AutoTokenizer, dict[str, Any]]:
    torch.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(CROSS_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(CROSS_MODEL).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses = []
    model.train()
    for epoch in range(EPOCHS):
        order = np.random.default_rng(1337 + epoch).permutation(len(training))
        for start in range(0, len(order), BATCH_SIZE):
            current = training.iloc[order[start:start + BATCH_SIZE]]
            encoded = tokenizer(
                list(zip(current["query"].astype(str), current["article"].astype(str))),
                truncation=True,
                padding=True,
                max_length=MAXIMUM_LENGTH,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            labels = torch.tensor(current["label"].to_numpy(dtype=np.float32), device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**encoded).logits.squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        report("train_epoch", epoch=epoch + 1, loss=float(np.mean(losses[-max(1, int(np.ceil(len(training) / BATCH_SIZE))):])))
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    diagnostics = {
        "rows": int(len(training)),
        "positive": int(training["label"].sum()),
        "negative": int((training["label"] == 0).sum()),
        "kinds": training["kind"].value_counts().to_dict(),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "mean_loss": float(np.mean(losses)),
    }
    return model, tokenizer, diagnostics


# Ranking gate

def rank_records(records: pd.DataFrame, cross_scores: np.ndarray) -> pd.DataFrame:
    result = records.copy()
    result["cross_score"] = cross_scores
    result = result.sort_values(
        ["queried_nct_id", "cross_score", "bi_score", "publication_identity"],
        ascending=[True, False, False, True],
    )
    result["cross_rank"] = result.groupby("queried_nct_id").cumcount() + 1
    result["selected_top8"] = result["cross_rank"].le(RERANKED_DOCUMENTS)
    return result.reset_index(drop=True)


def retrieval_metrics(ranked: pd.DataFrame, score_column: str, positive_pairs: set[tuple[str, str]]) -> dict[str, Any]:
    ordered = ranked.sort_values(["queried_nct_id", score_column, "publication_identity"], ascending=[True, False, True])
    positive_trials = sorted({nct_id for nct_id, _ in positive_pairs})
    recalls = []
    reciprocal_ranks = []
    hits = 0
    positives = 0
    for nct_id in positive_trials:
        current = ordered[ordered["queried_nct_id"].astype(str).eq(nct_id)]
        relevant = {identity for trial, identity in positive_pairs if trial == nct_id}
        positions = [position for position, identity in enumerate(current["publication_identity"].astype(str), start=1) if identity in relevant]
        top_positions = [position for position in positions if position <= RERANKED_DOCUMENTS]
        recalls.append(len(top_positions) / max(1, len(relevant)))
        reciprocal_ranks.append(1.0 / min(positions) if positions else 0.0)
        hits += len(top_positions)
        positives += len(relevant)
    return {
        "trials": int(len(positive_trials)),
        "positive_documents": int(positives),
        "recall_at_8": float(np.mean(recalls)) if recalls else 0.0,
        "global_recall_at_8": float(hits / max(1, positives)),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }


def _positive_pairs(records: pd.DataFrame) -> set[tuple[str, str]]:
    positive = records[records["registry_result_reference"].astype(bool)]
    return set(zip(positive["queried_nct_id"].astype(str), positive["publication_identity"].astype(str)))


def _baseline_scores(records: pd.DataFrame, contexts: dict[str, dict[str, Any]]) -> np.ndarray:
    baseline = prefilter_candidates(records.drop(columns=["query_text", "article_pair", "article_text"]), contexts, maximum=RETRIEVAL_CANDIDATES)
    score_map = dict(zip(zip(baseline["queried_nct_id"].astype(str), baseline["publication_identity"].astype(str)), baseline["prefilter_score"].astype(float)))
    minimum = min(score_map.values(), default=0.0) - 100.0
    return np.asarray([score_map.get((str(nct_id), str(identity)), minimum) for nct_id, identity in zip(records["queried_nct_id"], records["publication_identity"])], dtype=np.float64)


def run() -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    cache = shared_cache_dir()
    root = cache / VERSION
    diagnostics_path = root / "diagnostics.json"
    linkage = pd.read_parquet(cache / "registry_clock_lane0" / "features" / "registry_clock_features_v2" / "linkage.parquet")
    records_by_origin = {}
    contexts_by_origin = {}
    retrieved_by_origin = {}
    for origin in ORIGIN_SNAPSHOTS:
        records, contexts = load_origin(cache, origin, linkage)
        records["heuristic_score"] = _baseline_scores(records, contexts)
        scores = bi_encoder_scores(cache, origin, records)
        retrieved = retrieve(records, scores)
        records_by_origin[origin] = records
        contexts_by_origin[origin] = contexts
        retrieved_by_origin[origin] = retrieved
        report("retrieve", origin=origin, records=len(records), trials=records["queried_nct_id"].nunique(), top24=len(retrieved))
    training = relevance_labels(cache, "2018-01-01", retrieved_by_origin["2018-01-01"], contexts_by_origin["2018-01-01"])
    device = _device()
    base_tokenizer = AutoTokenizer.from_pretrained(CROSS_MODEL)
    base_model = AutoModelForSequenceClassification.from_pretrained(CROSS_MODEL)
    base_scores = {}
    for origin in ["2018-01-01", "2019-01-01"]:
        base_scores[origin] = score_cross(base_model, base_tokenizer, retrieved_by_origin[origin], device)
    base_model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model_path = root / "fine_tuned_cross_encoder"
    trained_model, trained_tokenizer, training_diagnostics = fit_cross_encoder(training, model_path)
    trained_scores = {}
    for origin in ORIGIN_SNAPSHOTS:
        trained_scores[origin] = score_cross(trained_model, trained_tokenizer, retrieved_by_origin[origin], device)
    trained_model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metrics = {}
    ranked = {}
    for origin in ["2018-01-01", "2019-01-01"]:
        positive_pairs = _positive_pairs(records_by_origin[origin])
        current = retrieved_by_origin[origin]
        current = current.copy()
        current["pretrained_cross_score"] = base_scores[origin]
        current["fine_tuned_cross_score"] = trained_scores[origin]
        metrics[origin] = {
            "heuristic": retrieval_metrics(current, "heuristic_score", positive_pairs),
            "bi_encoder": retrieval_metrics(current, "bi_score", positive_pairs),
            "pretrained_cross": retrieval_metrics(current, "pretrained_cross_score", positive_pairs),
            "fine_tuned_cross": retrieval_metrics(current, "fine_tuned_cross_score", positive_pairs),
        }
    development = metrics["2018-01-01"]
    candidate_names = ["pretrained_cross", "fine_tuned_cross"]
    selected = max(candidate_names, key=lambda name: (development[name]["recall_at_8"] + development[name]["mrr"], development[name]["mrr"]))
    sealed = metrics["2019-01-01"]
    accepted = bool(
        development[selected]["recall_at_8"] >= development["heuristic"]["recall_at_8"]
        and development[selected]["mrr"] >= development["heuristic"]["mrr"]
        and sealed[selected]["recall_at_8"] >= sealed["heuristic"]["recall_at_8"]
        and sealed[selected]["mrr"] >= sealed["heuristic"]["mrr"]
        and (
            development[selected]["recall_at_8"] > development["heuristic"]["recall_at_8"]
            or development[selected]["mrr"] > development["heuristic"]["mrr"]
            or sealed[selected]["recall_at_8"] > sealed["heuristic"]["recall_at_8"]
            or sealed[selected]["mrr"] > sealed["heuristic"]["mrr"]
        )
    )
    score_name = "fine_tuned_cross_score" if selected == "fine_tuned_cross" else "pretrained_cross_score"
    for origin in ORIGIN_SNAPSHOTS:
        current = retrieved_by_origin[origin].copy()
        current["pretrained_cross_score"] = base_scores.get(origin, np.full(len(current), np.nan))
        current["fine_tuned_cross_score"] = trained_scores[origin]
        if selected == "pretrained_cross" and origin not in base_scores:
            current["pretrained_cross_score"] = score_cross(
                AutoModelForSequenceClassification.from_pretrained(CROSS_MODEL),
                AutoTokenizer.from_pretrained(CROSS_MODEL), current, device,
            )
        ranked[origin] = rank_records(current, current[score_name].to_numpy(dtype=np.float64))
        ranking_path = root / "rankings" / f"{origin}.parquet"
        ranking_path.parent.mkdir(parents=True, exist_ok=True)
        ranked[origin].drop(columns=["article_pair"]).to_parquet(ranking_path, index=False)
    diagnostics = {
        "version": VERSION,
        "accepted": accepted,
        "selected": selected,
        "models": {"query": QUERY_MODEL, "article": ARTICLE_MODEL, "cross": CROSS_MODEL},
        "retrieval_candidates": RETRIEVAL_CANDIDATES,
        "reranked_documents": RERANKED_DOCUMENTS,
        "maximum_length": MAXIMUM_LENGTH,
        "training": training_diagnostics,
        "metrics": metrics,
        "elapsed_seconds": time.time() - START,
    }
    _atomic_json(diagnostics_path, diagnostics)
    register_artifact(cache, {
        "name": "generic_exp_2 MedCPT endpoint-to-paper reranker",
        "path": VERSION,
        "description": "Cached MedCPT query/article embeddings, reference-type-only cross-encoder relevance fit, four-origin rankings, and sealed-2019 recall@8/MRR gate.",
        "content_key": f"rel-trial-study-outcome:{VERSION}:medcpt-reference-supervision",
        "rebuild_hint": "Run medcpt_reranker.py from the literature_v3 parsed retrieval and projected snapshot caches.",
    })
    marker = root / "campaign_memory_recorded"
    if not marker.exists():
        locked_append(cache / "features_history.md", f'''\n### MedCPT endpoint-to-paper reranking
- run/experiment: generic_exp_2 lane 0 | status: TESTED-{"KEPT" if accepted else "REJECTED"}
- what: MedCPT query/article top-24 retrieval and cross-encoder top-8 reranking, trained for two epochs only on snapshot RESULT positives and reference-type/background, same-condition, and multi-NCT negatives.
- outcome: selected {selected}; metrics {json.dumps(metrics, sort_keys=True)}; training {json.dumps(training_diagnostics, sort_keys=True)}.
- takeaway: reranker selection used 2018 reference relevance and was confirmed once on sealed 2019; benchmark outcome labels were never relevance targets.
''')
        marker.write_text("recorded\n")
    report("complete", diagnostics=json.dumps(diagnostics, sort_keys=True, allow_nan=True))
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
