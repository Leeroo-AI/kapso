from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from kapso_datasets.common import run_data_dir, save_predictions, shared_cache_dir

from .graph_data import (
    build_graph_view,
    materialize_text_embeddings,
    register_artifact,
    unix_seconds,
    valid_query_mask,
)
from .graph_model import GraphConfig, predict_graph_model, train_graph_model
from .tabular import (
    FeatureMatrices,
    episode_weights,
    fit_gbdt,
    materialize_feature_matrices,
    predict_gbdt,
)


# Utilities

def rank_uniform(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return (ranks + 0.5) / len(values)


def blend_predictions(
    tabular: np.ndarray,
    graph: np.ndarray,
    graph_weight: float,
    mode: str,
) -> np.ndarray:
    if mode == "rank":
        tabular = rank_uniform(tabular)
        graph = rank_uniform(graph)
    return np.clip((1.0 - graph_weight) * tabular + graph_weight * graph, 1e-6, 1 - 1e-6)


def choose_blend(
    labels: np.ndarray,
    origins: np.ndarray,
    tabular: np.ndarray,
    graph: np.ndarray,
) -> tuple[float, str, dict[str, object]]:
    candidates: dict[str, dict[str, float]] = {}
    for mode in ("probability", "rank"):
        actual_mode = "probability" if mode == "probability" else "rank"
        for weight in (0.2, 0.35, 0.5):
            prediction = blend_predictions(tabular, graph, weight, actual_mode)
            scores = {
                str(origin): float(roc_auc_score(labels[origins == origin], prediction[origins == origin]))
                for origin in np.unique(origins)
            }
            candidates[f"{mode}_{weight}"] = scores
    default = candidates["probability_0.35"]
    supported: list[tuple[float, str, float]] = []
    for name, scores in candidates.items():
        if all(scores[origin] >= default[origin] + 0.0005 for origin in default):
            mode, weight_text = name.rsplit("_", 1)
            supported.append((float(weight_text), mode, float(np.mean(list(scores.values())))))
    if supported:
        weight, mode, _ = max(supported, key=lambda item: item[2])
    else:
        weight, mode = 0.35, "probability"
    return weight, "rank" if mode == "rank" else "probability", {
        "candidates": candidates,
        "selected": {"weight": weight, "mode": mode},
    }


def prediction_cache(shared_root: Path) -> Path:
    path = shared_root / "lane3_heterosage_causal_v1"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_complete_prediction_cache(path: Path, n_val: int, n_test: int):
    complete = path / "complete.json"
    if not complete.exists():
        return None
    metadata = json.loads(complete.read_text())
    val = np.load(path / "val_predictions.npy")
    test = np.load(path / "test_predictions.npy")
    if val.shape != (n_val,) or test.shape != (n_test,):
        return None
    return val, test, metadata


def save_component(path: Path, name: str, values: np.ndarray) -> None:
    temporary = path / f"{name}.tmp.npy"
    np.save(temporary, np.asarray(values, dtype=np.float64))
    os.replace(temporary, path / f"{name}.npy")


# Tabular chain

def fit_tabular_chains(
    ctx,
    matrices: FeatureMatrices,
    debug: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    target = ctx.task.target_col
    time_col = ctx.task.time_col
    train_frame = matrices.train_frame
    train_labels = train_frame[target].to_numpy(dtype=np.int64)
    metrics: dict[str, object] = {}
    internal: dict[str, np.ndarray] = {}
    if debug:
        model_a = fit_gbdt(
            matrices.train,
            train_labels,
            episode_weights(train_frame, time_col),
            True,
            2026,
        )
        val_prediction = predict_gbdt(model_a, matrices.val)
        combined_matrix = np.concatenate([matrices.train, matrices.val], axis=0)
        combined_frame = pd.concat([train_frame, ctx.val.df], ignore_index=True)
        combined_labels = combined_frame[target].to_numpy(dtype=np.int64)
        model_b = fit_gbdt(
            combined_matrix,
            combined_labels,
            episode_weights(combined_frame, time_col),
            True,
            3407,
        )
        test_prediction = predict_gbdt(model_b, matrices.test)
        return val_prediction, test_prediction, metrics, internal

    first_origin = pd.Timestamp("2020-04-02")
    second_origin = pd.Timestamp("2020-07-02")
    internal_train = train_frame[time_col] < first_origin
    internal_holdout = train_frame[time_col].isin([first_origin, second_origin])
    internal_model = fit_gbdt(
        matrices.train[internal_train.to_numpy()],
        train_labels[internal_train.to_numpy()],
        episode_weights(train_frame.loc[internal_train], time_col),
        False,
        2026,
    )
    internal_prediction = predict_gbdt(internal_model, matrices.train[internal_holdout.to_numpy()])
    holdout_frame = train_frame.loc[internal_holdout].reset_index(drop=True)
    internal["predictions"] = internal_prediction
    internal["labels"] = holdout_frame[target].to_numpy(dtype=np.int64)
    internal["origins"] = holdout_frame[time_col].to_numpy(dtype="datetime64[s]")
    metrics["internal_auc"] = {
        str(origin): float(
            roc_auc_score(
                internal["labels"][internal["origins"] == origin],
                internal_prediction[internal["origins"] == origin],
            )
        )
        for origin in np.unique(internal["origins"])
    }
    print(f"[gbdt] internal auc={json.dumps(metrics['internal_auc'])}")
    model_a = fit_gbdt(
        matrices.train,
        train_labels,
        episode_weights(train_frame, time_col),
        False,
        2026,
    )
    val_prediction = predict_gbdt(model_a, matrices.val)
    combined_matrix = np.concatenate([matrices.train, matrices.val], axis=0)
    combined_frame = pd.concat([train_frame, ctx.val.df], ignore_index=True)
    combined_labels = combined_frame[target].to_numpy(dtype=np.int64)
    model_b = fit_gbdt(
        combined_matrix,
        combined_labels,
        episode_weights(combined_frame, time_col),
        False,
        3407,
    )
    test_prediction = predict_gbdt(model_b, matrices.test)
    return val_prediction, test_prediction, metrics, internal


# Graph chain

def graph_training_arrays(ctx, frame: pd.DataFrame, user_count: int):
    mask = valid_query_mask(
        ctx.db,
        frame,
        ctx.task.entity_col,
        ctx.task.time_col,
        user_count,
    )
    ids = frame.loc[mask, ctx.task.entity_col].to_numpy(dtype=np.int64)
    timestamps = unix_seconds(frame.loc[mask, ctx.task.time_col])
    targets = frame.loc[mask, ctx.task.target_col].to_numpy(dtype=np.float32)
    weights = episode_weights(frame, ctx.task.time_col)[mask]
    return mask, ids, timestamps, targets, weights


def graph_prediction_rows(ctx, frame: pd.DataFrame, user_count: int, debug: bool):
    valid = valid_query_mask(
        ctx.db,
        frame,
        ctx.task.entity_col,
        ctx.task.time_col,
        user_count,
    )
    rows = np.flatnonzero(valid)
    if debug:
        rows = rows[: min(4096, len(rows))]
    ids = frame.iloc[rows][ctx.task.entity_col].to_numpy(dtype=np.int64)
    timestamps = unix_seconds(frame.iloc[rows][ctx.task.time_col])
    return rows, ids, timestamps


def fit_graph_seed(
    ctx,
    data,
    feature_meta,
    counts,
    train_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    fallback: np.ndarray,
    config: GraphConfig,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, list[float]]:
    _, ids, timestamps, targets, weights = graph_training_arrays(ctx, train_frame, counts["users"])
    model, epoch_times = train_graph_model(
        data,
        feature_meta,
        ctx.task.entity_table,
        ids,
        timestamps,
        targets,
        weights,
        config,
        seed,
        device,
    )
    rows, predict_ids, predict_times = graph_prediction_rows(
        ctx,
        predict_frame,
        counts["users"],
        config.epochs == 1 and config.fanout == (16, 8),
    )
    prediction = np.asarray(fallback, dtype=np.float64).copy()
    prediction[rows] = predict_graph_model(
        model,
        data,
        ctx.task.entity_table,
        predict_ids,
        predict_times,
        config,
        device,
    )
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return prediction, epoch_times


def lock_graph_epochs(
    ctx,
    data,
    feature_meta,
    counts,
    config: GraphConfig,
    device: torch.device,
) -> tuple[int, np.ndarray, dict[str, object], bool]:
    frame = ctx.train.df
    time_col = ctx.task.time_col
    target = ctx.task.target_col
    first_origin = pd.Timestamp("2020-04-02")
    second_origin = pd.Timestamp("2020-07-02")
    train_frame = frame.loc[frame[time_col] < first_origin].reset_index(drop=True)
    holdout_frame = frame.loc[frame[time_col].isin([first_origin, second_origin])].reset_index(drop=True)
    _, ids, timestamps, targets, weights = graph_training_arrays(ctx, train_frame, counts["users"])
    holdout_rows, holdout_ids, holdout_times = graph_prediction_rows(
        ctx,
        holdout_frame,
        counts["users"],
        False,
    )
    labels = holdout_frame[target].to_numpy(dtype=np.int64)[holdout_rows]
    origins = holdout_frame[time_col].to_numpy(dtype="datetime64[s]")[holdout_rows]
    state: dict[str, object] = {
        "best_epoch": 1,
        "best_mean": -np.inf,
        "best_prediction": None,
        "epochs": [],
        "stale": 0,
        "slow": False,
    }

    def callback(epoch, model, loss, elapsed):
        prediction = predict_graph_model(
            model,
            data,
            ctx.task.entity_table,
            holdout_ids,
            holdout_times,
            config,
            device,
        )
        scores = {
            str(origin): float(roc_auc_score(labels[origins == origin], prediction[origins == origin]))
            for origin in np.unique(origins)
        }
        mean_score = float(np.mean(list(scores.values())))
        state["epochs"].append({"epoch": epoch, "loss": loss, "elapsed": elapsed, "auc": scores})
        print(f"[gnn] internal epoch={epoch} auc={json.dumps(scores)} mean={mean_score:.6f}")
        if mean_score > float(state["best_mean"]) + 0.0002:
            state["best_mean"] = mean_score
            state["best_epoch"] = epoch
            state["best_prediction"] = prediction.copy()
            state["stale"] = 0
        else:
            state["stale"] = int(state["stale"]) + 1
        if epoch == 1 and elapsed > 1500:
            state["slow"] = True
            return False
        return int(state["stale"]) < 2

    model, _ = train_graph_model(
        data,
        feature_meta,
        ctx.task.entity_table,
        ids,
        timestamps,
        targets,
        weights,
        config,
        2026,
        device,
        callback,
    )
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return (
        int(state["best_epoch"]),
        np.asarray(state["best_prediction"]),
        {
            "epochs": state["epochs"],
            "labels": labels,
            "origins": origins,
            "rows": holdout_rows,
        },
        bool(state["slow"]),
    )


# Diagnostics

def diagnostic_slices(
    labels: np.ndarray,
    predictions: np.ndarray,
    matrices: FeatureMatrices,
) -> dict[str, object]:
    feature = np.asarray(matrices.val[:, matrices.names.index("own_comments_lifetime")])
    boundaries = np.quantile(feature, [0.0, 0.5, 0.8, 1.0])
    result: dict[str, object] = {}
    for index in range(3):
        if index == 2:
            mask = (feature >= boundaries[index]) & (feature <= boundaries[index + 1])
        else:
            mask = (feature >= boundaries[index]) & (feature < boundaries[index + 1])
        if mask.sum() and np.unique(labels[mask]).size == 2:
            result[f"comment_history_{index}"] = {
                "count": int(mask.sum()),
                "positives": int(labels[mask].sum()),
                "roc_auc": float(roc_auc_score(labels[mask], predictions[mask])),
                "average_precision": float(average_precision_score(labels[mask], predictions[mask])),
            }
    return result


def resolution_diagnostics(
    labels: np.ndarray,
    final_prediction: np.ndarray,
    components: list[np.ndarray],
) -> dict[str, object]:
    generator = np.random.default_rng(1337)
    aucs: list[float] = []
    for _ in range(100):
        indices = generator.integers(0, len(labels), size=len(labels))
        if np.unique(labels[indices]).size == 2:
            aucs.append(float(roc_auc_score(labels[indices], final_prediction[indices])))
    correlations: list[float] = []
    for left in range(len(components)):
        for right in range(left + 1, len(components)):
            correlations.append(float(spearmanr(components[left], components[right]).statistic))
    return {
        "bootstrap_draws": len(aucs),
        "auc_standard_error": float(np.std(aucs, ddof=1)),
        "mean_pairwise_rank_correlation": float(np.mean(correlations)),
        "pairwise_rank_correlations": correlations,
    }


# Orchestration

def run(ctx, debug: bool) -> None:
    started = time.time()
    shared_root = shared_cache_dir()
    cache = prediction_cache(shared_root)
    if not debug:
        cached = load_complete_prediction_cache(cache, len(ctx.val.df), len(ctx.test.df))
        if cached is not None:
            val_prediction, test_prediction, metadata = cached
            save_predictions(val_prediction, test_prediction)
            (run_data_dir() / "metrics.json").write_text(json.dumps(metadata, indent=2))
            print(f"[pipeline] reused complete prediction cache elapsed={time.time() - started:.1f}s")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    text_started = time.time()
    text_resources = materialize_text_embeddings(ctx.db, shared_root, debug, device)
    print(f"[pipeline] text phase elapsed={time.time() - text_started:.1f}s")

    feature_started = time.time()
    matrices = materialize_feature_matrices(ctx, shared_root, debug)
    gbdt_val, gbdt_test, gbdt_metrics, internal_gbdt = fit_tabular_chains(ctx, matrices, debug)
    save_predictions(gbdt_val, gbdt_test)
    print(f"[pipeline] GBDT phase elapsed={time.time() - feature_started:.1f}s")

    graph_config = GraphConfig(
        fanout=(16, 8) if debug else (64, 32),
        epochs=1 if debug else 8,
        seeds=(2026,) if debug else (2026, 3407),
    )
    cutoff_v = pd.Timestamp(ctx.dataset.val_timestamp)
    cutoff_t = pd.Timestamp(ctx.dataset.test_timestamp)
    graph_started = time.time()
    graph_v, meta_v, counts_v = build_graph_view(ctx.db, cutoff_v, text_resources)
    graph_metrics: dict[str, object] = {"configuration": asdict(graph_config)}
    if debug:
        locked_epochs = 1
        blend_weight, blend_mode = 0.35, "probability"
        train_a = matrices.train_frame
    else:
        locked_epochs, internal_graph_prediction, internal_graph, slow = lock_graph_epochs(
            ctx,
            graph_v,
            meta_v,
            counts_v,
            graph_config,
            device,
        )
        if slow:
            graph_config = replace(graph_config, fanout=(32, 16), seeds=(2026,))
            locked_epochs, internal_graph_prediction, internal_graph, _ = lock_graph_epochs(
                ctx,
                graph_v,
                meta_v,
                counts_v,
                graph_config,
                device,
            )
        blend_weight, blend_mode, blend_metrics = choose_blend(
            internal_graph["labels"],
            internal_graph["origins"],
            internal_gbdt["predictions"][internal_graph["rows"]],
            internal_graph_prediction,
        )
        graph_metrics["internal"] = internal_graph["epochs"]
        graph_metrics["blend"] = blend_metrics
        train_a = ctx.train.df
    final_config = replace(graph_config, epochs=locked_epochs)
    graph_metrics["locked_epochs"] = locked_epochs
    graph_metrics["final_configuration"] = asdict(final_config)
    print(
        f"[pipeline] locked epochs={locked_epochs} blend={blend_mode} "
        f"gbdt={1 - blend_weight:.2f} gnn={blend_weight:.2f}"
    )

    val_seed_predictions: list[np.ndarray] = []
    for seed in final_config.seeds:
        component_path = cache / f"gnn_val_seed_{seed}.npy"
        if not debug and component_path.exists():
            prediction = np.load(component_path)
            if prediction.shape == (len(ctx.val.df),):
                val_seed_predictions.append(prediction)
                print(f"[gnn] reused Model-A seed={seed} predictions")
                continue
        prediction, epoch_times = fit_graph_seed(
            ctx,
            graph_v,
            meta_v,
            counts_v,
            train_a,
            ctx.val.df,
            gbdt_val,
            final_config,
            seed,
            device,
        )
        val_seed_predictions.append(prediction)
        graph_metrics[f"model_a_seed_{seed}_epoch_seconds"] = epoch_times
        if not debug:
            save_component(cache, f"gnn_val_seed_{seed}", prediction)
    gnn_val = np.mean(val_seed_predictions, axis=0)
    del graph_v
    gc.collect()

    graph_t, meta_t, counts_t = build_graph_view(ctx.db, cutoff_t, text_resources)
    train_b = pd.concat([train_a, ctx.val.df], ignore_index=True)
    test_seed_predictions: list[np.ndarray] = []
    for seed in final_config.seeds:
        component_path = cache / f"gnn_test_seed_{seed}.npy"
        if not debug and component_path.exists():
            prediction = np.load(component_path)
            if prediction.shape == (len(ctx.test.df),):
                test_seed_predictions.append(prediction)
                print(f"[gnn] reused Model-B seed={seed} predictions")
                continue
        prediction, epoch_times = fit_graph_seed(
            ctx,
            graph_t,
            meta_t,
            counts_t,
            train_b,
            ctx.test.df,
            gbdt_test,
            final_config,
            seed,
            device,
        )
        test_seed_predictions.append(prediction)
        graph_metrics[f"model_b_seed_{seed}_epoch_seconds"] = epoch_times
        if not debug:
            save_component(cache, f"gnn_test_seed_{seed}", prediction)
    gnn_test = np.mean(test_seed_predictions, axis=0)
    del graph_t
    torch.cuda.empty_cache()
    gc.collect()
    print(f"[pipeline] graph phase elapsed={time.time() - graph_started:.1f}s")

    val_prediction = blend_predictions(gbdt_val, gnn_val, blend_weight, blend_mode)
    test_prediction = blend_predictions(gbdt_test, gnn_test, blend_weight, blend_mode)
    labels = ctx.val.df[ctx.task.target_col].to_numpy(dtype=np.int64)
    diagnostics = {
        "gbdt": gbdt_metrics,
        "graph": graph_metrics,
        "blend": {"weight": blend_weight, "mode": blend_mode},
        "slices": diagnostic_slices(labels, val_prediction, matrices),
        "resolution": resolution_diagnostics(labels, val_prediction, [gbdt_val, gnn_val, val_prediction]),
        "prediction_summary": {
            "val_min": float(val_prediction.min()),
            "val_max": float(val_prediction.max()),
            "val_mean": float(val_prediction.mean()),
            "test_min": float(test_prediction.min()),
            "test_max": float(test_prediction.max()),
            "test_mean": float(test_prediction.mean()),
        },
        "elapsed_seconds": time.time() - started,
    }
    save_predictions(val_prediction, test_prediction)
    (run_data_dir() / "metrics.json").write_text(json.dumps(diagnostics, indent=2, default=str))
    if not debug:
        save_component(cache, "val_predictions", val_prediction)
        save_component(cache, "test_predictions", test_prediction)
        (cache / "complete.json").write_text(json.dumps(diagnostics, indent=2, default=str))
        register_artifact(
            shared_root,
            {
                "name": "lane3 heterogeneous GraphSAGE and GBDT scores",
                "path": cache.name,
                "description": "Independent Model-A validation and Model-B test component and final score vectors for two GNN seeds and causal GBDT.",
                "content_key": cache.name,
                "rebuild_hint": "Delete only this content-key directory and rerun the full lane-3 candidate.",
            },
        )
    print(f"[pipeline] complete elapsed={time.time() - started:.1f}s")
