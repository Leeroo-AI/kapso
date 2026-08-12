from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from graph_model import clone_state, extract_embeddings, fit_churn_head, predict_churn_head, pretrain_encoder
from kapso_datasets.common import run_data_dir, save_predictions, shared_cache_dir
from temporal_pipeline import TemporalData, audit_temporal_routes, ensure_temporal_cache, make_hazard_features, seconds


warnings.filterwarnings("ignore")
VALIDATION_CUTOFF = pd.Timestamp("2018-09-01")
TEST_CUTOFF = pd.Timestamp("2020-01-01")
TARGET = "user_churn"


def phase(name: str, started: float) -> float:
    now = time.time()
    print(f"[phase] {name} elapsed={now - started:.1f}s")
    return now


def fit_lightgbm(features, labels, trees, seed):
    model = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=300,
        colsample_bytree=0.65,
        reg_lambda=10.0,
        n_estimators=trees,
        random_state=seed,
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")),
        verbosity=-1,
    )
    model.fit(np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32))
    return model


def predict_lightgbm(model, features):
    return model.predict_proba(np.asarray(features, dtype=np.float32))[:, 1].astype(np.float64)


def load_task_frames():
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    dataset_name = os.environ["RELBENCH_DATASET"]
    task_name = os.environ["RELBENCH_TASK"]
    get_dataset(dataset_name, download=False)
    task = get_task(dataset_name, task_name, download=False)
    return task, task.get_table("train").df.copy(), task.get_table("val").df.copy(), task.get_table("test").df.copy()


def feature_matrix(data, frame, cache_path: Path | None):
    if cache_path is not None and cache_path.exists():
        matrix = np.load(cache_path, mmap_mode="r")
        return matrix, None, {"count": int(len(matrix)), "cached": True}
    matrix, names, profile = make_hazard_features(data, frame.user_id, frame.timestamp)
    if cache_path is not None:
        temporary = cache_path.with_suffix(".tmp.npy")
        np.save(temporary, matrix, allow_pickle=False)
        os.replace(temporary, cache_path)
    return matrix, names, profile


def embedding_matrix(model, data, frame, fanout, cache_path: Path | None):
    if cache_path is not None and cache_path.exists():
        matrix = np.load(cache_path, mmap_mode="r")
        return matrix, 0.0
    matrix, rate = extract_embeddings(model, data, frame.user_id, frame.timestamp, fanout)
    if cache_path is not None:
        temporary = cache_path.with_suffix(".tmp.npy")
        np.save(temporary, matrix, allow_pickle=False)
        os.replace(temporary, cache_path)
    return matrix, rate


def generate_synthetic(database_root: Path, debug: bool, seed: int):
    anchors = pd.DataFrame({"timestamp": pd.date_range("2018-10-01", "2019-10-01", freq="MS")})
    connection = duckdb.connect()
    connection.register("anchors", anchors)
    ratings = str(database_root / "beer_ratings.parquet")
    query = f"""
        WITH eligible AS (
            SELECT a.timestamp, r.user_id
            FROM anchors a
            JOIN read_parquet('{ratings}') r
              ON r.created_at > a.timestamp - INTERVAL '90 day'
             AND r.created_at <= a.timestamp
            GROUP BY a.timestamp, r.user_id
        )
        SELECT e.timestamp, e.user_id,
               CAST(COUNT(r.user_id) = 0 AS INTEGER) AS user_churn
        FROM eligible e
        LEFT JOIN read_parquet('{ratings}') r
          ON r.user_id = e.user_id
         AND r.created_at > e.timestamp
         AND r.created_at <= e.timestamp + INTERVAL '90 day'
        GROUP BY e.timestamp, e.user_id
        ORDER BY e.timestamp, e.user_id
    """
    frame = connection.sql(query).fetchdf()
    connection.close()
    if (frame.timestamp + pd.Timedelta(days=90) > TEST_CUTOFF).any():
        raise RuntimeError("synthetic outcome window exceeds the test cutoff")
    if debug and len(frame) > 5000:
        frame = frame.sample(5000, random_state=seed).sort_values(["timestamp", "user_id"]).reset_index(drop=True)
    return frame


def sampled_training(frame, debug: bool, limit: int, seed: int):
    if not debug or len(frame) <= limit:
        return frame.reset_index(drop=True)
    timestamps = np.sort(frame.timestamp.unique())
    recent = frame[frame.timestamp.isin(timestamps[-12:])]
    if len(recent) > limit:
        recent = recent.sample(limit, random_state=seed)
    elif len(recent) < limit:
        older = frame[~frame.index.isin(recent.index)].sample(limit - len(recent), random_state=seed)
        recent = pd.concat([recent, older])
    return recent.sort_values(["timestamp", "user_id"]).reset_index(drop=True)


def safe_auc(labels, predictions):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, predictions))


def select_graph_method(frame, hazard, embedding, trees, head_epochs, seed):
    timestamps = np.sort(frame.timestamp.unique())
    recent_count = min(10, max(4, len(timestamps) // 3))
    recent = timestamps[-recent_count:]
    middle = max(2, recent_count // 2)
    stack_times = recent[:middle]
    evaluation_times = recent[middle:]
    before_stack = frame.timestamp < stack_times[0]
    stack_mask = frame.timestamp.isin(stack_times).to_numpy()
    before_evaluation = frame.timestamp < evaluation_times[0]
    evaluation_mask = frame.timestamp.isin(evaluation_times).to_numpy()
    labels = frame[TARGET].to_numpy(np.int32)
    head_stack = fit_churn_head(embedding[before_stack], labels[before_stack], head_epochs, seed + 1)
    graph_stack = predict_churn_head(head_stack, embedding[stack_mask])
    head_evaluation = fit_churn_head(embedding[before_evaluation], labels[before_evaluation], head_epochs, seed + 2)
    graph_evaluation = predict_churn_head(head_evaluation, embedding[evaluation_mask])
    baseline_evaluation_model = fit_lightgbm(hazard[before_evaluation], labels[before_evaluation], trees, seed + 3)
    baseline_evaluation = predict_lightgbm(baseline_evaluation_model, hazard[evaluation_mask])
    fusion_stack_features = np.column_stack([hazard[stack_mask], embedding[stack_mask], graph_stack])
    fusion_model = fit_lightgbm(fusion_stack_features, labels[stack_mask], trees, seed + 4)
    fusion_evaluation_features = np.column_stack([hazard[evaluation_mask], embedding[evaluation_mask], graph_evaluation])
    fusion_evaluation = predict_lightgbm(fusion_model, fusion_evaluation_features)
    weights = np.arange(0.0, 0.51, 0.1)
    blend_scores = []
    for weight in weights:
        blend = (1.0 - weight) * baseline_evaluation + weight * graph_evaluation
        per_anchor = []
        for timestamp in evaluation_times:
            mask = frame.loc[evaluation_mask, "timestamp"].to_numpy() == timestamp
            per_anchor.append(safe_auc(labels[evaluation_mask][mask], blend[mask]))
        blend_scores.append(float(np.mean(per_anchor)))
    blend_weight = float(weights[int(np.argmax(blend_scores))])
    candidates = {
        "direct": graph_evaluation,
        "blend": (1.0 - blend_weight) * baseline_evaluation + blend_weight * graph_evaluation,
        "fusion": fusion_evaluation,
    }
    baseline_anchor = []
    candidate_anchor = {name: [] for name in candidates}
    evaluation_frame = frame.loc[evaluation_mask].reset_index(drop=True)
    evaluation_labels = labels[evaluation_mask]
    for timestamp in evaluation_times:
        mask = evaluation_frame.timestamp.to_numpy() == timestamp
        baseline_anchor.append(safe_auc(evaluation_labels[mask], baseline_evaluation[mask]))
        for name, prediction in candidates.items():
            candidate_anchor[name].append(safe_auc(evaluation_labels[mask], prediction[mask]))
    decisions = {}
    for name, scores in candidate_anchor.items():
        difference = np.asarray(scores) - np.asarray(baseline_anchor)
        standard_error = float(np.std(difference, ddof=1) / np.sqrt(len(difference))) if len(difference) > 1 else 1.0
        decisions[name] = {
            "mean_auc": float(np.mean(scores)),
            "mean_delta": float(np.mean(difference)),
            "standard_error": standard_error,
            "positive_anchors": int((difference > 0).sum()),
            "stable": bool(np.mean(difference) > max(standard_error, 0.001) and (difference > 0).sum() >= math.ceil(0.6 * len(difference))),
        }
    stable = [name for name, values in decisions.items() if values["stable"]]
    method = max(stable, key=lambda name: decisions[name]["mean_auc"]) if stable else "fallback"
    graph_oof = np.empty(stack_mask.sum() + evaluation_mask.sum(), dtype=np.float32)
    graph_oof[: stack_mask.sum()] = graph_stack
    graph_oof[stack_mask.sum() :] = graph_evaluation
    recent_mask = stack_mask | evaluation_mask
    final_fusion = fit_lightgbm(
        np.column_stack([hazard[recent_mask], embedding[recent_mask], graph_oof]),
        labels[recent_mask],
        trees,
        seed + 5,
    )
    final_head = fit_churn_head(embedding, labels, head_epochs, seed + 6)
    diagnostics = {
        "evaluation_anchors": [str(pd.Timestamp(value)) for value in evaluation_times],
        "baseline_mean_auc": float(np.mean(baseline_anchor)),
        "candidates": decisions,
        "blend_weight": blend_weight,
        "selected": method,
    }
    return method, blend_weight, final_head, final_fusion, diagnostics


def apply_graph_method(method, blend_weight, baseline, head, fusion, hazard, embedding):
    graph = predict_churn_head(head, embedding)
    if method == "direct":
        prediction = graph
    elif method == "blend":
        prediction = (1.0 - blend_weight) * baseline + blend_weight * graph
    elif method == "fusion":
        prediction = predict_lightgbm(fusion, np.column_stack([hazard, embedding, graph]))
    else:
        prediction = baseline
    return np.asarray(prediction, dtype=np.float64), np.asarray(graph, dtype=np.float64)


def validation_slices(frame, predictions, hazard, feature_names, train_users):
    labels = frame[TARGET].to_numpy(np.int32)
    n90 = np.asarray(hazard[:, feature_names.index("beer_count_90d")])
    seen = frame.user_id.isin(train_users).to_numpy()
    strata = {}
    groups = {
        "count_1": n90 == 1,
        "count_2_5": (n90 >= 2) & (n90 <= 5),
        "count_6_plus": n90 >= 6,
        "seen_user": seen,
        "unseen_user": ~seen,
    }
    for name, mask in groups.items():
        strata[name] = {"count": int(mask.sum()), "roc_auc": safe_auc(labels[mask], predictions[mask])}
    return strata


def main() -> None:
    started = time.time()
    debug = "--debug" in sys.argv
    seed = 1337
    trees = 50 if debug else 300
    fanout = (4, 2) if debug else (16, 8)
    head_epochs = 3 if debug else 5
    task, train_full, validation, test = load_task_frames()
    shared = shared_cache_dir()
    database_root = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"] / "db"
    cache_root = ensure_temporal_cache(shared, database_root)
    data = TemporalData(cache_root)
    checkpoint_root = cache_root / "models"
    checkpoint_root.mkdir(exist_ok=True)
    matrix_root = cache_root / "matrices"
    matrix_root.mkdir(exist_ok=True)
    train = sampled_training(train_full, debug, 20000, seed)
    use_cache = not debug and len(train) == len(train_full)
    hazard_train, feature_names, train_profile = feature_matrix(data, train, matrix_root / "hazard_v5_train.npy" if use_cache else None)
    hazard_validation, validation_names, validation_profile = feature_matrix(data, validation, matrix_root / "hazard_v5_validation.npy" if not debug else None)
    hazard_test, test_names, test_profile = feature_matrix(data, test, matrix_root / "hazard_v5_test.npy" if not debug else None)
    if feature_names is None:
        feature_names = validation_names or test_names
    if feature_names is None:
        _, feature_names, _ = make_hazard_features(data, validation.iloc[:1].user_id, validation.iloc[:1].timestamp)
    fallback_a = fit_lightgbm(hazard_train, train[TARGET], trees, seed)
    validation_fallback = predict_lightgbm(fallback_a, hazard_validation)
    if debug:
        train_b_indices = np.linspace(0, len(train) - 1, 10000, dtype=np.int64)
        validation_b_indices = np.linspace(0, len(validation) - 1, 5000, dtype=np.int64)
        official_b = pd.concat([train.iloc[train_b_indices], validation.iloc[validation_b_indices]], ignore_index=True)
        hazard_official_b = np.vstack([hazard_train[train_b_indices], hazard_validation[validation_b_indices]])
    else:
        official_b = pd.concat([train, validation], ignore_index=True)
        hazard_official_b = np.vstack([hazard_train, hazard_validation])
    fallback_b_early = fit_lightgbm(hazard_official_b, official_b[TARGET], trees, seed + 10)
    test_fallback_early = predict_lightgbm(fallback_b_early, hazard_test)
    save_predictions(validation_fallback, test_fallback_early)
    marker = phase("compact fallback written", started)
    audit_a = audit_temporal_routes(data, validation.user_id, validation.timestamp, fanout, seed)
    print(f"[audit] chain=A {json.dumps(audit_a, sort_keys=True)}")
    encoder_a = pretrain_encoder(
        data,
        int(VALIDATION_CUTOFF.timestamp()),
        checkpoint_root / "encoder_a.pt",
        fanout,
        debug,
        seed,
    )
    marker = phase("encoder A", marker)
    embedding_train, rate_train = embedding_matrix(
        encoder_a, data, train, fanout, matrix_root / "embedding_a_train.npy" if use_cache else None
    )
    embedding_validation, rate_validation = embedding_matrix(
        encoder_a, data, validation, fanout, matrix_root / "embedding_a_validation.npy" if not debug else None
    )
    method, blend_weight, head_a, fusion_a, selection = select_graph_method(
        train, hazard_train, embedding_train, trees, head_epochs, seed
    )
    validation_prediction, validation_graph = apply_graph_method(
        method,
        blend_weight,
        validation_fallback,
        head_a,
        fusion_a,
        hazard_validation,
        embedding_validation,
    )
    marker = phase("encoder A inference and forward selection", marker)
    synthetic = generate_synthetic(database_root, debug, seed)
    remaining = max(0, (20000 if debug else 600000) - len(official_b))
    if len(synthetic) > remaining:
        synthetic = synthetic.sample(remaining, random_state=seed).sort_values(["timestamp", "user_id"]).reset_index(drop=True)
    hazard_synthetic, synthetic_names, synthetic_profile = feature_matrix(
        data, synthetic, matrix_root / "hazard_v5_synthetic.npy" if not debug else None
    )
    supervised_b = pd.concat([official_b, synthetic], ignore_index=True)
    hazard_b = np.vstack([hazard_official_b, hazard_synthetic])
    fallback_b = fit_lightgbm(hazard_b, supervised_b[TARGET], trees, seed + 20)
    test_fallback = predict_lightgbm(fallback_b, hazard_test)
    audit_b = audit_temporal_routes(data, test.user_id, test.timestamp, fanout, seed + 1)
    print(f"[audit] chain=B {json.dumps(audit_b, sort_keys=True)}")
    state_a = clone_state(encoder_a)
    encoder_b = pretrain_encoder(
        data,
        int(TEST_CUTOFF.timestamp()),
        checkpoint_root / "encoder_b.pt",
        fanout,
        debug,
        seed + 100,
        state_a,
    )
    del encoder_a, state_a
    gc.collect()
    marker = phase("encoder B", marker)
    embedding_official_b, rate_official_b = embedding_matrix(
        encoder_b, data, official_b, fanout, matrix_root / "embedding_b_official.npy" if not debug else None
    )
    embedding_synthetic, rate_synthetic = embedding_matrix(
        encoder_b, data, synthetic, fanout, matrix_root / "embedding_b_synthetic.npy" if not debug else None
    )
    embedding_test, rate_test = embedding_matrix(
        encoder_b, data, test, fanout, matrix_root / "embedding_b_test.npy" if not debug else None
    )
    head_b_base = fit_churn_head(embedding_official_b, official_b[TARGET], head_epochs, seed + 21)
    graph_synthetic = predict_churn_head(head_b_base, embedding_synthetic)
    fusion_b = fit_lightgbm(
        np.column_stack([hazard_synthetic, embedding_synthetic, graph_synthetic]),
        synthetic[TARGET],
        trees,
        seed + 22,
    )
    embedding_b = np.vstack([embedding_official_b, embedding_synthetic])
    head_b = fit_churn_head(embedding_b, supervised_b[TARGET], head_epochs, seed + 23)
    test_prediction, test_graph = apply_graph_method(
        method,
        blend_weight,
        test_fallback,
        head_b,
        fusion_b,
        hazard_test,
        embedding_test,
    )
    validation_prediction = np.clip(np.nan_to_num(validation_prediction, nan=0.5), 0.0, 1.0)
    test_prediction = np.clip(np.nan_to_num(test_prediction, nan=0.5), 0.0, 1.0)
    save_predictions(validation_prediction, test_prediction)
    strata = validation_slices(validation, validation_prediction, hazard_validation, feature_names, set(train.user_id))
    metrics = {
        "debug": debug,
        "selected": method,
        "selection": selection,
        "profiles": {"train": train_profile, "validation": validation_profile, "test": test_profile, "synthetic": synthetic_profile},
        "audits": {"encoder_a": audit_a, "encoder_b": audit_b},
        "embedding_rates_rows_per_second": {
            "train": rate_train,
            "validation": rate_validation,
            "synthetic": rate_synthetic,
            "official_b": rate_official_b,
            "test": rate_test,
        },
        "validation_strata": strata,
        "rows": {"train": len(train), "validation": len(validation), "synthetic": len(synthetic), "test": len(test)},
        "elapsed_seconds": time.time() - started,
    }
    (run_data_dir() / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[result] {json.dumps({'selected': method, 'blend_weight': blend_weight, 'strata': strata, 'elapsed_seconds': time.time() - started}, sort_keys=True)}")
    phase("complete", marker)


if __name__ == "__main__":
    main()
