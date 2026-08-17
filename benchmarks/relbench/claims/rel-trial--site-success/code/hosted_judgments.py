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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from campaign_io import locked_append, register_artifact


# Configuration

MODEL = "gpt-5.4-nano-2026-03-17"
PROMPT_VERSION = "site-success-exact-methodology-v1"
FIELDS = [
    "methodological_rigor",
    "endpoint_objectivity",
    "design_adequacy",
    "power_adequacy",
    "execution_risk",
    "endpoint_coherence",
]
SCHEMA = {
    "type": "object",
    "properties": {field: {"type": "integer", "minimum": 0, "maximum": 4} for field in FIELDS},
    "required": FIELDS,
    "additionalProperties": False,
}
SYSTEM = "You are a clinical-trial methodology reviewer. Judge the complete protocol and its structured historical context. Do not guess or predict any recorded result, p-value, significance label, or site-success target. Use 2 for typical in this domain, 0 for very weak or insufficient evidence, and 4 for very strong; execution_risk is reversed so 4 means high risk. Judge interactions such as whether enrollment and design are adequate for the endpoint ambition."


# Cache

def _key(origin: str, context: str, document: str) -> str:
    value = f"{MODEL}\0{PROMPT_VERSION}\0{origin}\0{context}\0{document}"
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _load(path: Path) -> dict[str, dict[str, int]]:
    records: dict[str, dict[str, int]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
                if record.get("key") and record.get("result"):
                    records[record["key"]] = record["result"]
            except json.JSONDecodeError:
                continue
    return records


# Extraction

def _extract(context: str, document: str) -> dict[str, int]:
    client = OpenAI(timeout=180.0, max_retries=0)
    prompt = f"STRUCTURED CONTEXT\n{context}\n\nCOMPLETE DOCUMENT\n{document}"
    error: Exception | None = None
    for attempt in range(6):
        try:
            response = client.responses.create(
                model=MODEL,
                reasoning={"effort": "low"},
                input=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                text={"format": {"type": "json_schema", "name": "trial_methodology", "schema": SCHEMA, "strict": True}},
                max_output_tokens=300,
                store=False,
            )
            result = json.loads(response.output_text)
            return {field: int(result[field]) for field in FIELDS}
        except Exception as current:
            error = current
            if attempt < 5:
                time.sleep(min(12.0, 0.75 * 2 ** attempt) + random.random())
    raise RuntimeError(f"hosted extraction failed: {error}")


def ensure_judgments(
    origins: list[str],
    contexts: list[str],
    documents: list[str],
    indices: list[int],
    cache: Path,
    concurrency: int,
) -> tuple[np.ndarray, dict[str, float]]:
    root = cache / "lane0_hosted_judgments_v1"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "responses.jsonl"
    records = _load(path)
    keys = {index: _key(origins[index], contexts[index], documents[index]) for index in indices}
    missing = [index for index in indices if keys[index] not in records]
    started = time.time()
    failures: list[str] = []
    if missing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            jobs = {pool.submit(_extract, contexts[index], documents[index]): index for index in missing}
            for job in concurrent.futures.as_completed(jobs):
                index = jobs[job]
                try:
                    result = job.result()
                    records[keys[index]] = result
                    locked_append(path, json.dumps({"key": keys[index], "result": result}, sort_keys=True) + "\n")
                except Exception as error:
                    failures.append(f"{index}:{error}")
    if failures:
        raise RuntimeError(f"hosted judgment failures={len(failures)} sample={failures[:3]}")
    register_artifact(cache, {
        "name": "lane 0 target-exact hosted methodology judgments",
        "path": str(root.relative_to(cache)),
        "description": "Structured methodology judgments keyed by pinned model, prompt, origin, context, and complete document",
        "content_key": f"{MODEL}:{PROMPT_VERSION}",
        "rebuild_hint": "Run main.py; absent content hashes are appended under a lock.",
    })
    elapsed = max(time.time() - started, 1e-6)
    matrix = np.asarray([[records[keys[index]][field] for field in FIELDS] for index in indices], dtype=np.float32)
    diagnostics = {
        "requested": float(len(indices)),
        "new": float(len(missing)),
        "seconds": elapsed,
        "new_rows_per_second": float(len(missing) / elapsed),
    }
    return matrix, diagnostics


# Measurement

def forward_measurement(
    judgments: np.ndarray,
    structured: np.ndarray,
    labels: np.ndarray,
    years: np.ndarray,
) -> dict[str, Any]:
    unique = sorted(np.unique(years))
    validation_years = unique[-2:]
    prediction = np.full(len(labels), np.nan, dtype=np.float64)
    copied: list[float] = []
    for year in validation_years:
        train = years < year
        validation = years == year
        if train.sum() < 80 or validation.sum() < 20 or len(np.unique(labels[train])) < 2:
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(structured[train])
        x_validation = scaler.transform(structured[validation])
        residual_train = np.zeros_like(judgments[train], dtype=np.float64)
        residual_validation = np.zeros_like(judgments[validation], dtype=np.float64)
        for column in range(judgments.shape[1]):
            ridge = Ridge(alpha=20.0)
            ridge.fit(x_train, judgments[train, column])
            fitted = ridge.predict(x_train)
            residual_train[:, column] = judgments[train, column] - fitted
            residual_validation[:, column] = judgments[validation, column] - ridge.predict(x_validation)
            copied.append(float(r2_score(judgments[train, column], fitted)))
        model = LogisticRegression(C=0.1, max_iter=400, class_weight="balanced")
        model.fit(residual_train, labels[train])
        prediction[validation] = model.predict_proba(residual_validation)[:, 1]
    mask = np.isfinite(prediction)
    if mask.sum() < 20 or len(np.unique(labels[mask])) < 2:
        return {"rows": int(mask.sum()), "auc": None, "mae": None, "mean_dimension_r2": None}
    return {
        "rows": int(mask.sum()),
        "auc": float(roc_auc_score(labels[mask], prediction[mask])),
        "mae": float(mean_absolute_error(labels[mask], prediction[mask])),
        "mean_dimension_r2": float(np.mean(copied)) if copied else None,
    }
