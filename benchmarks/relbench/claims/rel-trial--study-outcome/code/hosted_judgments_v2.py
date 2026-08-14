# Imports

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from campaign_io import locked_append, register_artifact


# Configuration

MODEL = "gpt-5.4-mini-2026-03-17"
PROMPT_VERSION = "clinical-detectability-v2"
JUDGMENT_FIELDS = [
    "effect_detectability", "endpoint_alignment", "control_integrity", "attrition_resilience",
    "analysis_flexibility", "biology_uncertainty", "execution_capacity", "site_heterogeneity",
    "evidence_maturity",
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
SYSTEM_PROMPT = """You are a senior biostatistician and therapeutic-area trial reviewer. Evaluate interactions among the complete protocol text and structured as-of-origin context. Use integer scores 1-5 anchored so 2 is typical for comparable trials and 0 means genuinely insufficient evidence. Higher is favorable for effect detectability, endpoint alignment, control integrity, attrition resilience, execution capacity, and evidence maturity. Higher means more risk for analysis flexibility, biology uncertainty, and site heterogeneity. Distinguish nominal enrollment from analyzable power; judge whether endpoint timing and population match the intervention's plausible causal pathway; penalize diffuse endpoints, fragile subgrouping, demanding follow-up, contamination, and operational heterogeneity. The micro-summary must identify the main effect-detection strength and the main statistical or operational vulnerability in constrained clinical language. Do not identify the trial, expose database identifiers, restate structured facts, or predict its recorded outcome."""


# Cache

def _key(context: str, document: str) -> str:
    payload = f"{MODEL}\0{PROMPT_VERSION}\0{context}\0{document}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    if path.exists():
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
                text={"format": {"type": "json_schema", "name": "trial_detectability", "strict": True, "schema": SCHEMA}},
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
    raise RuntimeError(f"Hosted detectability extraction failed after retries: {last_error}")


def ensure_judgments_v2(
    contexts: list[str], documents: list[str], indices: list[int], cache_dir: Path, concurrency: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, float]]:
    root = cache_dir / "hosted_detectability_v2"
    root.mkdir(parents=True, exist_ok=True)
    response_path = root / "responses.jsonl"
    cached = _load_cache(response_path)
    selected = {index: _key(contexts[index], documents[index]) for index in indices}
    missing = [index for index in indices if selected[index] not in cached]
    started = time.time()
    failures = []
    if missing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            future_to_index = {executor.submit(_extract_one, contexts[index], documents[index]): index for index in missing}
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    cached[selected[index]] = result
                    locked_append(response_path, json.dumps({"key": selected[index], "result": result}, sort_keys=True) + "\n")
                except Exception as error:
                    failures.append((index, str(error)))
    if failures:
        raise RuntimeError(f"Hosted detectability failures: {failures[:5]}; total={len(failures)}")
    register_artifact(
        cache_dir,
        {
            "name": "clinical-hosted-detectability-v2",
            "path": str(root.relative_to(cache_dir)),
            "description": "GPT-5.4 mini structured effect-detectability judgments from a second internally gated prompt",
            "content_key": f"{MODEL}:{PROMPT_VERSION}",
            "rebuild_hint": "Run main.py; missing context-document hashes are appended under a file lock.",
        },
    )
    elapsed = max(time.time() - started, 1e-6)
    return {index: cached[selected[index]] for index in indices}, {
        "requested": float(len(indices)), "new": float(len(missing)), "seconds": elapsed,
        "new_rows_per_second": float(len(missing) / elapsed),
    }


def records_to_arrays_v2(records: dict[int, dict[str, Any]], ordered_indices: list[int]) -> tuple[np.ndarray, list[str]]:
    judgments = np.asarray([[records[index][field] for field in JUDGMENT_FIELDS] for index in ordered_indices], dtype=np.float32)
    summaries = [str(records[index]["micro_summary"]) for index in ordered_indices]
    return judgments, summaries
