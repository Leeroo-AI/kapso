from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from features import FEATURE_NAMES, FeatureBuilder, build_common_bundle, build_synthetic_bundle, day_values
from fusion import (
    anchor_predict,
    calibrated_rank_blend,
    fit_calibrator,
    fit_normalizer,
    fusion_predict,
    percentile_rank,
    temporal_weights,
    tokenize_papers,
    train_anchor,
    train_fusion,
)
from kapso_datasets.common import is_debug, load_task, run_data_dir, save_predictions, shared_cache_dir


def register_artifact(shared: Path, name: str, path: Path, description: str, content_key: str):
    registry = shared / "artifacts.json"
    lock_path = shared / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if registry.exists():
            try:
                records = json.loads(registry.read_text())
            except json.JSONDecodeError:
                records = []
        else:
            records = []
        relative = str(path.relative_to(shared))
        record = {
            "name": name,
            "path": relative,
            "description": description,
            "content_key": content_key,
            "rebuild_hint": "Run python main.py; each artifact is check-before-compute.",
        }
        records = [item for item in records if item.get("name") != name]
        records.append(record)
        temporary = registry.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(records, indent=2))
        os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def concatenate_blocks(blocks):
    return np.concatenate([np.asarray(block) for block in blocks], axis=0)


def choose_blend(anchor_prediction, neural_prediction, labels, anchor_auc, neural_auc):
    candidates = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
    if neural_auc < anchor_auc - 0.01:
        candidates = [weight for weight in candidates if weight >= 0.85]
    anchor_rank = percentile_rank(anchor_prediction)
    neural_rank = percentile_rank(neural_prediction)
    scores = {}
    for weight in candidates:
        prediction = weight * anchor_rank + (1.0 - weight) * neural_rank
        scores[weight] = float(roc_auc_score(labels, prediction))
    best_score = max(scores.values())
    tied = [weight for weight, score in scores.items() if score >= best_score - 0.0002]
    selected = max(tied)
    print(f"[selection] blend_scores={json.dumps(scores)} selected_anchor_weight={selected:.2f}")
    return selected, scores[selected]


def stratum_metrics(features, labels, prediction):
    age = np.asarray(features[:, 1]) * 365.25
    citation_count = np.expm1(np.asarray(features[:, 18]))
    segments = {
        "young": age <= 182,
        "old_cold": (age > 182) & (citation_count < 0.5),
        "old_low": (age > 182) & (citation_count >= 0.5) & (citation_count <= 2.5),
        "old_warm": (age > 182) & (citation_count > 2.5),
    }
    result = {}
    for name, mask in segments.items():
        count = int(mask.sum())
        score = None
        if count and len(np.unique(labels[mask])) == 2:
            score = float(roc_auc_score(labels[mask], prediction[mask]))
        result[name] = {"count": count, "roc_auc": score}
    return result


def debug_pipeline(ctx, tokens, common, builder, output_dir: Path):
    train_df = ctx.train.df
    rng = np.random.default_rng(1337)
    latest = train_df["date"].to_numpy() == train_df["date"].max()
    eligible = np.flatnonzero(latest)
    chosen = np.sort(rng.choice(eligible, size=5000, replace=False))
    features = np.asarray(common["train_x"][chosen])
    paper_ids = train_df["Paper_ID"].to_numpy(np.int64)[chosen]
    labels = train_df[ctx.target_col].to_numpy(np.float32)[chosen]
    days = day_values(train_df["date"])[chosen]
    weights = temporal_weights(days, None)
    anchor = train_anchor(features, labels, weights, 180)
    val_anchor = anchor_predict(anchor, common["val_x"])
    test_anchor = anchor_predict(anchor, common["test_x"])
    mean, scale = fit_normalizer(features)
    model, _ = train_fusion(
        tokens[0],
        tokens[1],
        paper_ids,
        features,
        labels,
        weights,
        mean,
        scale,
        epochs=3.0,
        maximum_steps=100,
    )
    val_ids = ctx.val.df["Paper_ID"].to_numpy(np.int64)
    test_ids = ctx.test.df["Paper_ID"].to_numpy(np.int64)
    val_neural = val_anchor.copy()
    test_neural = test_anchor.copy()
    val_limit = min(5000, len(val_ids))
    test_limit = min(5000, len(test_ids))
    val_neural[:val_limit] = fusion_predict(
        model,
        tokens[0],
        tokens[1],
        val_ids[:val_limit],
        common["val_x"][:val_limit],
        mean,
        scale,
    )
    test_neural[:test_limit] = fusion_predict(
        model,
        tokens[0],
        tokens[1],
        test_ids[:test_limit],
        common["test_x"][:test_limit],
        mean,
        scale,
    )
    val_prediction = calibrated_rank_blend(val_anchor, val_neural, 0.85)
    test_prediction = calibrated_rank_blend(test_anchor, test_neural, 0.85)
    save_predictions(val_prediction, test_prediction)
    diagnostics = {
        "mode": "debug",
        "examples": 5000,
        "optimizer_steps": 100,
        "feature_count": len(FEATURE_NAMES),
        "anchor_fallback_val_rows": len(val_ids) - val_limit,
        "anchor_fallback_test_rows": len(test_ids) - test_limit,
    }
    (run_data_dir() / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    (output_dir / "debug_metrics.json").write_text(json.dumps(diagnostics, indent=2))


def full_pipeline(ctx, tokens, common, synthetic, output_dir: Path, started: float):
    import torch

    train_df = ctx.train.df
    train_ids = train_df["Paper_ID"].to_numpy(np.int64)
    train_y = train_df[ctx.target_col].to_numpy(np.float32)
    train_days = day_values(train_df["date"])
    val_ids = ctx.val.df["Paper_ID"].to_numpy(np.int64)
    val_y = ctx.val.df[ctx.target_col].to_numpy(np.float32)
    val_days = day_values(ctx.val.df["date"])
    test_ids = ctx.test.df["Paper_ID"].to_numpy(np.int64)

    hold_day = train_days.max()
    internal_train = train_days < hold_day
    internal_hold = train_days == hold_day
    internal_x = concatenate_blocks((common["train_x"][internal_train], synthetic["internal"]["x"]))
    internal_ids = concatenate_blocks((train_ids[internal_train], synthetic["internal"]["ids"]))
    internal_y = concatenate_blocks((train_y[internal_train], synthetic["internal"]["y"]))
    internal_days = concatenate_blocks((train_days[internal_train], synthetic["internal"]["days"]))
    hold_x = np.asarray(common["train_x"][internal_hold])
    hold_ids = train_ids[internal_hold]
    hold_y = train_y[internal_hold]

    anchor_candidates = {}
    for name, half_life in (("no_decay", None), ("two_year", 730.0)):
        weights = temporal_weights(internal_days, half_life)
        model = train_anchor(internal_x, internal_y, weights, 1200, hold_x, hold_y)
        prediction = anchor_predict(model, hold_x)
        score = float(roc_auc_score(hold_y, prediction))
        anchor_candidates[name] = {
            "half_life": half_life,
            "model": model,
            "prediction": prediction,
            "score": score,
            "rounds": model.best_iteration or model.current_iteration(),
        }
        print(f"[selection] anchor_decay={name} internal_auc={score:.6f}")
    selected_decay = max(anchor_candidates, key=lambda name: anchor_candidates[name]["score"])
    selected_anchor = anchor_candidates[selected_decay]
    half_life = selected_anchor["half_life"]
    anchor_rounds = int(selected_anchor["rounds"])
    internal_anchor_prediction = selected_anchor["prediction"]
    internal_anchor_auc = float(selected_anchor["score"])

    internal_weights = temporal_weights(internal_days, half_life)
    internal_mean, internal_scale = fit_normalizer(internal_x)
    internal_model, neural_checkpoints = train_fusion(
        tokens[0],
        tokens[1],
        internal_ids,
        internal_x,
        internal_y,
        internal_weights,
        internal_mean,
        internal_scale,
        epochs=1.0,
        checkpoints=[0.5, 1.0],
        evaluation_data={"paper_ids": hold_ids, "features": hold_x},
    )
    neural_scores = {
        fraction: float(roc_auc_score(hold_y, prediction))
        for fraction, prediction in neural_checkpoints.items()
    }
    if neural_scores[1.0] >= neural_scores[0.5] - 0.001:
        selected_epoch = 1.0
    else:
        selected_epoch = 0.5
    internal_neural_prediction = neural_checkpoints[selected_epoch]
    internal_neural_auc = neural_scores[selected_epoch]
    print(f"[selection] neural_epoch_auc={json.dumps(neural_scores)} selected_epoch={selected_epoch}")
    anchor_weight, internal_blend_auc = choose_blend(
        internal_anchor_prediction,
        internal_neural_prediction,
        hold_y,
        internal_anchor_auc,
        internal_neural_auc,
    )
    internal_blend_raw = calibrated_rank_blend(
        internal_anchor_prediction,
        internal_neural_prediction,
        anchor_weight,
    )
    calibrator = fit_calibrator(internal_blend_raw, hold_y)
    internal_strata = stratum_metrics(hold_x, hold_y, internal_blend_raw)
    del internal_model, neural_checkpoints, anchor_candidates
    torch.cuda.empty_cache()

    model_a_x = concatenate_blocks(
        (common["train_x"], synthetic["internal"]["x"], synthetic["model_a"]["x"])
    )
    model_a_ids = concatenate_blocks(
        (train_ids, synthetic["internal"]["ids"], synthetic["model_a"]["ids"])
    )
    model_a_y = concatenate_blocks(
        (train_y, synthetic["internal"]["y"], synthetic["model_a"]["y"])
    )
    model_a_days = concatenate_blocks(
        (train_days, synthetic["internal"]["days"], synthetic["model_a"]["days"])
    )
    model_a_weights = temporal_weights(model_a_days, half_life)
    anchor_a = train_anchor(model_a_x, model_a_y, model_a_weights, anchor_rounds)
    anchor_a.save_model(str(output_dir / "anchor_model_a.txt"))
    val_anchor = anchor_predict(anchor_a, common["val_x"])
    np.save(output_dir / "val_anchor.npy", val_anchor)
    model_a_mean, model_a_scale = fit_normalizer(model_a_x)
    fusion_a, _ = train_fusion(
        tokens[0],
        tokens[1],
        model_a_ids,
        model_a_x,
        model_a_y,
        model_a_weights,
        model_a_mean,
        model_a_scale,
        epochs=selected_epoch,
    )
    val_neural = fusion_predict(
        fusion_a,
        tokens[0],
        tokens[1],
        val_ids,
        common["val_x"],
        model_a_mean,
        model_a_scale,
    )
    np.save(output_dir / "val_fusion.npy", val_neural)
    val_prediction = calibrated_rank_blend(val_anchor, val_neural, anchor_weight, calibrator)
    np.save(output_dir / "val_model_a_final.npy", val_prediction)
    np.save(run_data_dir() / "val_predictions.npy", val_prediction)
    print(f"[model_a] banked validation predictions rows={len(val_prediction)} elapsed={time.time() - started:.1f}s")
    del fusion_a, anchor_a, model_a_x
    torch.cuda.empty_cache()

    model_b_x = concatenate_blocks(
        (
            common["train_x"],
            synthetic["internal"]["x"],
            synthetic["model_a"]["x"],
            common["val_x"],
            synthetic["model_b_1"]["x"],
            synthetic["model_b_2"]["x"],
        )
    )
    model_b_ids = concatenate_blocks(
        (
            train_ids,
            synthetic["internal"]["ids"],
            synthetic["model_a"]["ids"],
            val_ids,
            synthetic["model_b_1"]["ids"],
            synthetic["model_b_2"]["ids"],
        )
    )
    model_b_y = concatenate_blocks(
        (
            train_y,
            synthetic["internal"]["y"],
            synthetic["model_a"]["y"],
            val_y,
            synthetic["model_b_1"]["y"],
            synthetic["model_b_2"]["y"],
        )
    )
    model_b_days = concatenate_blocks(
        (
            train_days,
            synthetic["internal"]["days"],
            synthetic["model_a"]["days"],
            val_days,
            synthetic["model_b_1"]["days"],
            synthetic["model_b_2"]["days"],
        )
    )
    model_b_weights = temporal_weights(model_b_days, half_life)
    anchor_b = train_anchor(model_b_x, model_b_y, model_b_weights, anchor_rounds)
    anchor_b.save_model(str(output_dir / "anchor_model_b.txt"))
    test_anchor = anchor_predict(anchor_b, common["test_x"])
    np.save(output_dir / "test_anchor.npy", test_anchor)
    test_fallback = calibrator.predict(percentile_rank(test_anchor))
    np.save(run_data_dir() / "test_predictions.npy", np.clip(test_fallback, 1e-5, 1.0 - 1e-5))
    print(f"[model_b] banked anchor fallback rows={len(test_anchor)} elapsed={time.time() - started:.1f}s")

    full_timeout = int(os.environ.get("RELBENCH_FULL_TIMEOUT", "14400"))
    remaining = full_timeout - (time.time() - started)
    if remaining >= 2700:
        model_b_mean, model_b_scale = fit_normalizer(model_b_x)
        fusion_b, _ = train_fusion(
            tokens[0],
            tokens[1],
            model_b_ids,
            model_b_x,
            model_b_y,
            model_b_weights,
            model_b_mean,
            model_b_scale,
            epochs=0.5,
        )
        test_neural = fusion_predict(
            fusion_b,
            tokens[0],
            tokens[1],
            test_ids,
            common["test_x"],
            model_b_mean,
            model_b_scale,
        )
        np.save(output_dir / "test_fusion.npy", test_neural)
        test_prediction = calibrated_rank_blend(test_anchor, test_neural, anchor_weight, calibrator)
        del fusion_b
    else:
        test_neural = test_anchor
        test_prediction = np.clip(test_fallback, 1e-5, 1.0 - 1e-5)
        print(f"[model_b] fusion update skipped remaining_seconds={remaining:.0f}")

    diagnostics = {
        "mode": "full",
        "feature_count": len(FEATURE_NAMES),
        "internal_hold_origin_day": int(hold_day),
        "internal_anchor_decay_auc": {
            "no_decay": float(selected_anchor["score"]) if selected_decay == "no_decay" else None,
            "selected": selected_decay,
        },
        "selected_half_life_days": half_life,
        "anchor_rounds": anchor_rounds,
        "neural_epoch_scores": neural_scores,
        "selected_model_a_epochs": selected_epoch,
        "internal_anchor_auc": internal_anchor_auc,
        "internal_neural_auc": internal_neural_auc,
        "anchor_blend_weight": anchor_weight,
        "internal_blend_auc": internal_blend_auc,
        "internal_strata": internal_strata,
        "model_a_rows": len(model_a_y),
        "model_b_rows": len(model_b_y),
        "elapsed_seconds": time.time() - started,
    }
    save_predictions(val_prediction, test_prediction)
    (run_data_dir() / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"[pipeline] completed full elapsed={time.time() - started:.1f}s")


def main():
    started = time.time()
    output_dir = Path("output_data_generic_exp_1")
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_task()
    if ctx.dataset_name != "rel-arxiv" or ctx.task_name != "paper-citation":
        raise RuntimeError("This candidate is specialized for rel-arxiv/paper-citation")
    shared = shared_cache_dir()
    cache_dir = shared / "lane1_specter_fusion_v3"
    cache_dir.mkdir(parents=True, exist_ok=True)
    papers = ctx.db.table_dict["papers"].df
    tokens = tokenize_papers(papers, cache_dir, maximum_length=192)
    register_artifact(
        shared,
        "lane1_specter2_tokens_v3",
        cache_dir,
        "SPECTER2 length-192 token IDs, masks, and untruncated token counts for rel-arxiv papers.",
        "rel-arxiv_papers_193696_specter2_base_len192_v3",
    )
    builder = FeatureBuilder(ctx, tokens[2])
    common = build_common_bundle(ctx, builder, cache_dir)
    register_artifact(
        shared,
        "lane1_temporal_features_v3",
        cache_dir,
        "Temporally censored 80-feature common and synthetic matrices for the SPECTER2 fusion lane.",
        "rel-arxiv_paper-citation_temporal_80_v3",
    )
    print(f"[pipeline] common features ready elapsed={time.time() - started:.1f}s")
    if is_debug():
        debug_pipeline(ctx, tokens, common, builder, output_dir)
    else:
        synthetic = build_synthetic_bundle(builder, cache_dir)
        full_pipeline(ctx, tokens, common, synthetic, output_dir, started)


if __name__ == "__main__":
    main()
