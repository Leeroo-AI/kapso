# Imports

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from campaign_io import locked_append, register_artifact


# Configuration

MODEL = "gpt-5.4-mini-2026-03-17"
EMBEDDING_MODEL = "text-embedding-3-large"
PROMPT_VERSION = "clinical-methodology-v1"
JUDGMENT_FIELDS = [
    "power_adequacy", "endpoint_objectivity", "biological_plausibility", "design_coherence",
    "recruitment_difficulty", "population_restrictiveness", "operational_risk", "multiplicity",
]
SCHEMA = {
    "type": "object",
    "properties": {
        **{field: {"type": "integer", "minimum": 0, "maximum": 5} for field in JUDGMENT_FIELDS},
        "micro_summary": {"type": "string", "maxLength": 600},
    },
    "required": JUDGMENT_FIELDS + ["micro_summary"],
    "additionalProperties": False,
}
SYSTEM_PROMPT = """You are a rigorous clinical-trial methodology reviewer. Judge the complete supplied document in interaction with its structured as-of-origin context. Use integer scores 1-5, where 2 is typical for this therapeutic domain; for favorable dimensions 5 is strongest, while recruitment difficulty, operational risk, and multiplicity use 5 for highest difficulty or risk. Use 0 only when evidence is insufficient. Power adequacy means whether the enrollment and design seem adequate for the implied clinical ambition. Endpoint objectivity means resistance to subjective measurement. Biological plausibility means whether intervention, population, and endpoint fit coherently. Design coherence means internal methodological alignment. Population restrictiveness means selectiveness rather than quality. Return a concise methodological micro-summary using controlled clinical language. Do not identify the trial, restate database identifiers, or predict its recorded outcome."""


# Cache

def _key(context: str, document: str) -> str:
    payload = f"{MODEL}\0{PROMPT_VERSION}\0{context}\0{document}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def _cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    root = cache_dir / "hosted_judgments_v1"
    root.mkdir(parents=True, exist_ok=True)
    return root, root / "responses.jsonl"


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
                if item.get("key") and item.get("result"):
                    records[item["key"]] = item["result"]
            except json.JSONDecodeError:
                continue
    return records


# API

def _extract_one(context: str, document: str) -> dict[str, Any]:
    client = OpenAI(timeout=180.0, max_retries=0)
    input_text = f"STRUCTURED CONTEXT\n{context}\n\nCOMPLETE DOCUMENT\n{document}"
    last_error = None
    for attempt in range(6):
        try:
            response = client.responses.create(
                model=MODEL,
                instructions=SYSTEM_PROMPT,
                input=input_text,
                tools=[],
                text={"format": {"type": "json_schema", "name": "trial_judgments", "strict": True, "schema": SCHEMA}},
                reasoning={"effort": "none"},
                max_output_tokens=500,
                store=False,
            )
            result = json.loads(response.output_text)
            for field in JUDGMENT_FIELDS:
                result[field] = int(result[field])
            result["micro_summary"] = str(result["micro_summary"])
            return result
        except Exception as error:
            last_error = error
            if attempt == 5:
                break
            time.sleep(min(16.0, 0.8 * (2 ** attempt)) + random.random())
    raise RuntimeError(f"Hosted extraction failed after retries: {last_error}")


def ensure_judgments(
    contexts: list[str],
    documents: list[str],
    indices: list[int],
    cache_dir: Path,
    concurrency: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, float]]:
    root, response_path = _cache_paths(cache_dir)
    cached = _load_cache(response_path)
    selected = {index: _key(contexts[index], documents[index]) for index in indices}
    missing = [index for index in indices if selected[index] not in cached]
    started = time.time()
    failures = []
    if missing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            future_to_index = {
                executor.submit(_extract_one, contexts[index], documents[index]): index for index in missing
            }
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    cached[selected[index]] = result
                    locked_append(response_path, json.dumps({"key": selected[index], "result": result}, sort_keys=True) + "\n")
                except Exception as error:
                    failures.append((index, str(error)))
    if failures:
        raise RuntimeError(f"Hosted extraction failures: {failures[:5]}; total={len(failures)}")
    register_artifact(
        cache_dir,
        {
            "name": "clinical-hosted-judgments-v1",
            "path": str(root.relative_to(cache_dir)),
            "description": "GPT-5.4 mini structured methodological judgments keyed by model, prompt, origin context, and complete-document hash",
            "content_key": f"{MODEL}:{PROMPT_VERSION}",
            "rebuild_hint": "Run main.py; missing content hashes are appended under a file lock.",
        },
    )
    elapsed = max(time.time() - started, 1e-6)
    results = {index: cached[selected[index]] for index in indices}
    return results, {"requested": float(len(indices)), "new": float(len(missing)), "seconds": elapsed, "new_rows_per_second": float(len(missing) / elapsed)}


def _embedding_key(summary: str) -> str:
    return hashlib.sha256(f"{EMBEDDING_MODEL}\0{summary}".encode()).hexdigest()


def ensure_embeddings(summaries: list[str], cache_dir: Path, dimensions: int = 256) -> np.ndarray:
    root, _ = _cache_paths(cache_dir)
    path = root / f"embeddings_{dimensions}.jsonl"
    cached: dict[str, list[float]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                    cached[item["key"]] = item["embedding"]
                except json.JSONDecodeError:
                    continue
    missing_keys = []
    missing_summaries = []
    for summary in summaries:
        key = _embedding_key(summary)
        if key not in cached and key not in missing_keys:
            missing_keys.append(key)
            missing_summaries.append(summary)
    client = OpenAI(timeout=180.0)
    for start in range(0, len(missing_summaries), 512):
        batch = missing_summaries[start:start + 512]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch, dimensions=dimensions)
        for key, item in zip(missing_keys[start:start + 512], response.data):
            embedding = [float(value) for value in item.embedding]
            cached[key] = embedding
            locked_append(path, json.dumps({"key": key, "embedding": embedding}, separators=(",", ":")) + "\n")
    return np.asarray([cached[_embedding_key(summary)] for summary in summaries], dtype=np.float32)


def records_to_arrays(records: dict[int, dict[str, Any]], ordered_indices: list[int]) -> tuple[np.ndarray, list[str]]:
    judgments = np.asarray([[records[index][field] for field in JUDGMENT_FIELDS] for index in ordered_indices], dtype=np.float32)
    summaries = [str(records[index]["micro_summary"]) for index in ordered_indices]
    return judgments, summaries
