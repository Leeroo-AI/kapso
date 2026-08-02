# Imports

from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers.utils import logging as transformers_logging

from kapso_datasets.common import is_debug, run_data_dir, save_predictions, shared_cache_dir
from lexical_residual import documents, evidence_categories, fit_lexical
from trial_features import FEATURE_VERSION, load_features
from trial_model import (
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    TOKEN_VERSION,
    ensure_token_cache,
    resolve_model_path,
    train_and_predict,
)


# Configuration

PIPELINE_VERSION = "lane1_hierarchical_lora_v4"
STAGE_VERSION = "lane1_transformer_stages_v4"
SCHEDULES = (0.75, 1.0, 1.25)
HARD_STOP_SECONDS = 3 * 3600 + 20 * 60


# Utilities

def elapsed(started: float) -> float:
    return time.time() - started


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


def ids_fingerprint(train_ids: np.ndarray, predict_ids: np.ndarray, labels: np.ndarray, schedules: tuple[float, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(PIPELINE_VERSION.encode())
    digest.update(np.asarray(train_ids, dtype=np.int64).tobytes())
    digest.update(np.asarray(predict_ids, dtype=np.int64).tobytes())
    digest.update(np.asarray(labels, dtype=np.int8).tobytes())
    digest.update(np.asarray(schedules, dtype=np.float32).tobytes())
    return digest.hexdigest()[:20]


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores)))


def auc_standard_error(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels)
    value = auc(labels, scores)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    q1 = value / max(2.0 - value, 1e-9)
    q2 = 2.0 * value * value / max(1.0 + value, 1e-9)
    variance = (
        value * (1.0 - value)
        + (positives - 1) * (q1 - value * value)
        + (negatives - 1) * (q2 - value * value)
    ) / max(positives * negatives, 1)
    return float(math.sqrt(max(variance, 0.0)))


def task_tables():
    from relbench.tasks import get_task

    task = get_task(os.environ["RELBENCH_DATASET"], os.environ["RELBENCH_TASK"], download=False)
    return task, task.get_table("train").df.copy(), task.get_table("val").df.copy(), task.get_table("test").df.copy()


def register_artifact(cache: Path, name: str, path: Path, description: str, content_key: str, rebuild_hint: str) -> None:
    registry = cache / "artifacts.json"
    lock_path = cache / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = json.loads(registry.read_text()) if registry.exists() else []
        relative = str(path.relative_to(cache))
        record = {
            "name": name,
            "path": relative,
            "description": description,
            "content_key": content_key,
            "rebuild_hint": rebuild_hint,
        }
        if not any(item.get("name") == name and item.get("content_key") == content_key for item in records):
            records.append(record)
            temporary = cache / f"artifacts.{os.getpid()}.tmp.json"
            temporary.write_text(json.dumps(records, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock, fcntl.LOCK_UN)


def append_history(cache: Path, marker: str, text: str) -> None:
    path = cache / "features_history.md"
    lock_path = cache / "features_history.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = path.read_text() if path.exists() else ""
        if marker not in existing:
            with path.open("a") as handle:
                handle.write("\n" + text.rstrip() + "\n")
        fcntl.flock(lock, fcntl.LOCK_UN)


# Transformer stages

def transformer_stage(
    store,
    train_ids: np.ndarray,
    labels: np.ndarray,
    predict_ids: np.ndarray,
    schedules: tuple[float, ...],
    stage: str,
    stage_root: Path,
    model_path: Path,
    seed: int,
    train_lora: bool = True,
    deadline: float | None = None,
) -> tuple[dict[float, np.ndarray], dict]:
    key = ids_fingerprint(train_ids, predict_ids, labels, schedules)
    stage_root.mkdir(parents=True, exist_ok=True)
    path = stage_root / f"{stage}_{key}.npz"
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        metadata = json.loads(str(cached["metadata"].item()))
        outputs = {schedule: cached[f"schedule_{schedule:.2f}"].astype(np.float64) for schedule in schedules}
        if all(len(values) == len(predict_ids) for values in outputs.values()):
            metadata["cached"] = True
            return outputs, metadata
    outputs, metadata = train_and_predict(
        store,
        train_ids,
        labels,
        predict_ids,
        schedules,
        seed,
        model_path,
        train_lora=train_lora,
        deadline=deadline,
    )
    missing = [schedule for schedule in schedules if schedule not in outputs]
    if missing:
        raise RuntimeError(f"Transformer stage {stage} missed checkpoints {missing}")
    payload = {f"schedule_{schedule:.2f}": outputs[schedule].astype(np.float32) for schedule in schedules}
    payload["metadata"] = np.asarray(json.dumps(metadata))
    temporary = stage_root / f"{stage}_{key}.{os.getpid()}.tmp.npz"
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)
    metadata["cached"] = False
    return outputs, metadata


# Diagnostics

def stratum_record(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> dict:
    records = {}
    for group in np.unique(groups):
        mask = groups == group
        y = labels[mask]
        record = {"count": int(mask.sum()), "positive_rate": round(float(y.mean()), 6)}
        if len(np.unique(y)) == 2:
            record["roc_auc"] = round(auc(y, scores[mask]), 6)
        records[str(group)] = record
    return records


def validation_strata(features, structured: np.ndarray, ids: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict:
    texts = documents(features, ids)
    evidence = evidence_categories(texts)
    lengths = features.iloc[ids]["study"].str.len().to_numpy()
    length_band = np.where(lengths < 700, "short", np.where(lengths < 1400, "medium", "long"))
    relation_presence = (np.asarray(structured[ids, 6:14]) > 0).sum(axis=1)
    relation_band = np.where(relation_presence <= 3, "sparse", np.where(relation_presence <= 6, "medium", "rich"))
    del texts
    return {
        "lexical_evidence": stratum_record(labels, scores, evidence),
        "study_length": stratum_record(labels, scores, length_band),
        "relational_coverage": stratum_record(labels, scores, relation_band),
    }


# Debug pipeline

def run_debug(
    features,
    structured: np.ndarray,
    store,
    train,
    val,
    test,
    cache: Path,
    model_path: Path,
    started: float,
) -> dict:
    rng = np.random.default_rng(1337)
    positions = rng.choice(len(train), size=min(8000, len(train)), replace=False)
    train_ids = train.iloc[positions]["id"].to_numpy(np.int64)
    train_labels = train.iloc[positions]["child"].to_numpy(np.int8)
    output_ids = np.concatenate([val["id"].to_numpy(np.int64), test["id"].to_numpy(np.int64)])
    transformer, transformer_meta = train_and_predict(
        store,
        train_ids,
        train_labels,
        output_ids,
        (0.75,),
        1337,
        model_path,
        train_lora=False,
    )
    lexical, lexical_meta = fit_lexical(
        features,
        train_ids,
        train_labels,
        output_ids,
        "debug_model_a",
        cache,
        debug=True,
    )
    logits = 0.5 * transformer[0.75] + 0.5 * lexical
    probabilities = np.clip(sigmoid(logits), 1e-6, 1.0 - 1e-6)
    n_val = len(val)
    save_predictions(probabilities[:n_val], probabilities[n_val:])
    metrics = {
        "mode": "debug",
        "elapsed_seconds": round(elapsed(started), 3),
        "transformer": transformer_meta,
        "lexical": lexical_meta,
        "validation_roc_auc_diagnostic": round(auc(val["child"].to_numpy(), probabilities[:n_val]), 6),
    }
    (run_data_dir() / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# Full pipeline

def run_full(
    features,
    structured: np.ndarray,
    store,
    train,
    val,
    test,
    cache: Path,
    model_path: Path,
    started: float,
) -> dict:
    stage_root = cache / STAGE_VERSION
    train_year = train["date"].dt.year.to_numpy()
    fold_specs = [
        ("fold_2018", train_year <= 2017, train_year == 2018, 1337),
        ("fold_2019", train_year <= 2018, train_year == 2019, 1441),
    ]
    fold_records = []
    for name, fit_mask, predict_mask, seed in fold_specs:
        fit = train.loc[fit_mask]
        predict = train.loc[predict_mask]
        fit_ids = fit["id"].to_numpy(np.int64)
        fit_labels = fit["child"].to_numpy(np.int8)
        predict_ids = predict["id"].to_numpy(np.int64)
        predict_labels = predict["child"].to_numpy(np.int8)
        lexical_logits, lexical_meta = fit_lexical(
            features, fit_ids, fit_labels, predict_ids, name, cache, debug=False
        )
        transformer_logits, transformer_meta = transformer_stage(
            store,
            fit_ids,
            fit_labels,
            predict_ids,
            SCHEDULES,
            name,
            stage_root,
            model_path,
            seed,
        )
        record = {
            "name": name,
            "rows": int(len(predict)),
            "positives": int(predict_labels.sum()),
            "labels": predict_labels,
            "lexical_logits": lexical_logits,
            "transformer_logits": transformer_logits,
            "lexical": lexical_meta,
            "transformer": transformer_meta,
        }
        fold_records.append(record)
        print(
            f"[fold] {name} rows={len(predict)} lexical_auc={auc(predict_labels, lexical_logits):.6f} "
            + " ".join(
                f"epoch_{schedule:.2f}_auc={auc(predict_labels, transformer_logits[schedule]):.6f}"
                for schedule in SCHEDULES
            ),
            flush=True,
        )

    schedule_metrics = {}
    for schedule in SCHEDULES:
        fold_aucs = [auc(record["labels"], record["transformer_logits"][schedule]) for record in fold_records]
        fold_ses = [auc_standard_error(record["labels"], record["transformer_logits"][schedule]) for record in fold_records]
        schedule_metrics[schedule] = {
            "fold_roc_auc": fold_aucs,
            "mean_roc_auc": float(np.mean(fold_aucs)),
            "mean_standard_error": float(math.sqrt(sum(value * value for value in fold_ses)) / len(fold_ses)),
        }
    best_schedule = max(SCHEDULES, key=lambda value: schedule_metrics[value]["mean_roc_auc"])
    noise = max(0.0005, schedule_metrics[best_schedule]["mean_standard_error"])
    selected_schedule = min(
        schedule
        for schedule in SCHEDULES
        if schedule_metrics[schedule]["mean_roc_auc"] >= schedule_metrics[best_schedule]["mean_roc_auc"] - noise
    )
    lexical_fold_auc = [auc(record["labels"], record["lexical_logits"]) for record in fold_records]
    consistent_gain = all(
        auc(record["labels"], record["transformer_logits"][selected_schedule]) > auc(record["labels"], record["lexical_logits"])
        for record in fold_records
    )
    print(
        f"[selection] best={best_schedule:.2f} selected={selected_schedule:.2f} "
        f"noise={noise:.6f} consistent_transformer_gain={consistent_gain}",
        flush=True,
    )

    oos_transformer = np.concatenate(
        [record["transformer_logits"][selected_schedule] for record in fold_records]
    )
    oos_lexical = np.concatenate([record["lexical_logits"] for record in fold_records])
    oos_labels = np.concatenate([record["labels"] for record in fold_records])
    blend = LogisticRegression(C=0.2, solver="lbfgs", max_iter=500, random_state=1337)
    blend.fit(np.column_stack([oos_transformer, oos_lexical]), oos_labels)
    oos_blended = blend.predict_proba(np.column_stack([oos_transformer, oos_lexical]))[:, 1]
    print(
        f"[blend] coefs={blend.coef_[0].round(6).tolist()} intercept={float(blend.intercept_[0]):.6f} "
        f"oos_auc={auc(oos_labels, oos_blended):.6f}",
        flush=True,
    )

    train_ids = train["id"].to_numpy(np.int64)
    train_labels = train["child"].to_numpy(np.int8)
    val_ids = val["id"].to_numpy(np.int64)
    val_labels = val["child"].to_numpy(np.int8)
    model_a_lexical, model_a_lexical_meta = fit_lexical(
        features, train_ids, train_labels, val_ids, "model_a", cache, debug=False
    )
    model_a_transformer_dict, model_a_transformer_meta = transformer_stage(
        store,
        train_ids,
        train_labels,
        val_ids,
        (selected_schedule,),
        "model_a",
        stage_root,
        model_path,
        1553,
        deadline=started + HARD_STOP_SECONDS - 3600,
    )
    model_a_transformer = model_a_transformer_dict[selected_schedule]
    model_a_components = np.column_stack([model_a_transformer, model_a_lexical])
    model_a_blend_logits = blend.decision_function(model_a_components)
    validation_predictions = np.clip(sigmoid(model_a_blend_logits), 1e-6, 1.0 - 1e-6)
    print(
        f"[model_a] val_auc_diagnostic={auc(val_labels, validation_predictions):.6f} "
        f"elapsed={elapsed(started):.1f}s",
        flush=True,
    )

    train_val_ids = np.concatenate([train_ids, val_ids])
    train_val_labels = np.concatenate([train_labels, val_labels])
    test_ids = test["id"].to_numpy(np.int64)
    model_b_lexical, model_b_lexical_meta = fit_lexical(
        features, train_val_ids, train_val_labels, test_ids, "model_b", cache, debug=False
    )
    model_b_transformer_dict, model_b_transformer_meta = transformer_stage(
        store,
        train_val_ids,
        train_val_labels,
        test_ids,
        (selected_schedule,),
        "model_b",
        stage_root,
        model_path,
        1667,
        deadline=started + HARD_STOP_SECONDS - 600,
    )
    model_b_transformer = model_b_transformer_dict[selected_schedule]
    model_b_blend_logits = blend.decision_function(
        np.column_stack([model_b_transformer, model_b_lexical])
    )
    model_b_calibration = LogisticRegression(C=0.2, solver="lbfgs", max_iter=500, random_state=1337)
    model_b_calibration.fit(model_a_blend_logits.reshape(-1, 1), val_labels)
    test_predictions = np.clip(
        model_b_calibration.predict_proba(model_b_blend_logits.reshape(-1, 1))[:, 1],
        1e-6,
        1.0 - 1e-6,
    )
    save_predictions(validation_predictions, test_predictions)

    fold_metrics = []
    for record in fold_records:
        fold_metrics.append(
            {
                "name": record["name"],
                "rows": record["rows"],
                "positives": record["positives"],
                "lexical_roc_auc": round(auc(record["labels"], record["lexical_logits"]), 7),
                "transformer_roc_auc": {
                    f"{schedule:.2f}": round(auc(record["labels"], record["transformer_logits"][schedule]), 7)
                    for schedule in SCHEDULES
                },
                "lexical_runtime": record["lexical"],
                "transformer_runtime": record["transformer"],
            }
        )
    metrics = {
        "mode": "full",
        "pipeline_version": PIPELINE_VERSION,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "sha256": MODEL_SHA256},
        "folds": fold_metrics,
        "schedule_metrics": {f"{key:.2f}": value for key, value in schedule_metrics.items()},
        "selected_schedule": selected_schedule,
        "selection_noise": noise,
        "consistent_transformer_gain": consistent_gain,
        "lexical_fold_auc": lexical_fold_auc,
        "blend_coefficients": blend.coef_[0].tolist(),
        "blend_intercept": float(blend.intercept_[0]),
        "oos_blended_roc_auc": auc(oos_labels, oos_blended),
        "model_a": {"lexical": model_a_lexical_meta, "transformer": model_a_transformer_meta},
        "model_b": {"lexical": model_b_lexical_meta, "transformer": model_b_transformer_meta},
        "model_b_calibration_coefficient": float(model_b_calibration.coef_[0, 0]),
        "model_b_calibration_intercept": float(model_b_calibration.intercept_[0]),
        "validation_roc_auc_diagnostic": auc(val_labels, validation_predictions),
        "validation_strata": validation_strata(features, structured, val_ids, val_labels, validation_predictions),
        "elapsed_seconds": round(elapsed(started), 3),
    }
    (run_data_dir() / "metrics.json").write_text(json.dumps(metrics, indent=2))
    history_marker = f"lane1 internal folds outcome {PIPELINE_VERSION}"
    history_text = f"""### Hierarchical LoRA and lexical forward-fold outcome
- run/experiment: generic_exp_1 | status: TESTED-KEPT | marker: {history_marker}
- what: Pinned three-slot rank-8 BiomedBERT, exact lexical residual, fixed 0.75/1.0/1.25 epoch checkpoints, and C=0.2 OOS logit blend.
- outcome: 2018/2019 lexical AUC {lexical_fold_auc[0]:.6f}/{lexical_fold_auc[1]:.6f}; selected transformer AUC {schedule_metrics[selected_schedule]['fold_roc_auc'][0]:.6f}/{schedule_metrics[selected_schedule]['fold_roc_auc'][1]:.6f}; selected {selected_schedule:.2f} epochs within noise {noise:.6f}; OOS blend AUC {auc(oos_labels, oos_blended):.6f}; consistent transformer gain={consistent_gain}.
- takeaway: Schedule and blend were fixed exclusively from forward OOS predictions; Model A remained train-only and Model B started independently before train+validation fitting.
"""
    append_history(cache, history_marker, history_text)
    gc.collect()
    return metrics


# Entrypoint

def main() -> None:
    started = time.time()
    warnings.filterwarnings("ignore")
    transformers_logging.set_verbosity_error()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    debug = is_debug()
    cache = shared_cache_dir()
    task, train, val, test = task_tables()
    if task.target_col != "child" or len(val) != 14470 or len(test) != 23430:
        raise RuntimeError("Unexpected task contract")
    print(
        f"[data] train={len(train)} val={len(val)} test={len(test)} debug={debug} elapsed={elapsed(started):.1f}s",
        flush=True,
    )
    features, structured, feature_meta, feature_cache = load_features(cache)
    print(
        f"[serialization] rows={len(features)} cached={feature_meta.get('elapsed_seconds', 0) != elapsed(started)} "
        f"build_seconds={feature_meta.get('elapsed_seconds')} elapsed={elapsed(started):.1f}s",
        flush=True,
    )
    for frame, name in [(train, "train"), (val, "val"), (test, "test")]:
        ids = frame["id"].to_numpy(np.int64)
        if ids.min() < 0 or ids.max() >= len(features):
            raise RuntimeError(f"{name} IDs are outside serialized feature rows")
        feature_dates = features.iloc[ids]["date"].to_numpy(dtype="datetime64[ns]")
        if not np.array_equal(feature_dates, frame["date"].to_numpy(dtype="datetime64[ns]")):
            raise RuntimeError(f"{name} feature dates are not row-aligned")
    model_path = resolve_model_path()
    store, token_meta = ensure_token_cache(features, structured, feature_cache, model_path)
    print(
        f"[tokenization] revision={MODEL_REVISION} cached={token_meta.get('cached')} "
        f"seconds={token_meta.get('elapsed_seconds')} elapsed={elapsed(started):.1f}s",
        flush=True,
    )
    register_artifact(
        cache,
        "lane1 deterministic three-slot tokens",
        feature_cache,
        "Temporally censored STUDY, CONCEPT_ORG, DESIGN_AUX text, structured fields, and pinned-tokenizer arrays.",
        f"{FEATURE_VERSION}:{TOKEN_VERSION}:{MODEL_REVISION}",
        "Run main.py; the cache is rebuilt from sanitized projected columns when its version metadata is absent.",
    )
    if debug:
        metrics = run_debug(features, structured, store, train, val, test, cache, model_path, started)
    else:
        metrics = run_full(features, structured, store, train, val, test, cache, model_path, started)
    print(
        f"[complete] mode={metrics['mode']} elapsed={elapsed(started):.1f}s "
        f"output={run_data_dir()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
