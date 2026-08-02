import fcntl
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from feature_pipeline import BadgeFeatureBuilder, load_or_build_activity
from kapso_datasets.check_predictions import main as check_predictions
from kapso_datasets.common import is_debug, load_task, run_data_dir, save_predictions, shared_cache_dir
from modeling import (
    choose_blend,
    compare_feature_designs,
    forward_catboost,
    train_catboost,
    train_lightgbm_ensemble,
)


warnings.filterwarnings("ignore")


def read_campaign_memory(shared):
    for name in ("features_history.md", "table_information.md"):
        path = shared / name
        if path.exists():
            text = path.read_text()
            print(f"[memory] read {name} bytes={len(text)}")


def append_campaign_result(shared, internal, selected, blend_weight, cache_key):
    if internal:
        compact = internal["compact"]
        full = internal["full"]
        outcome = (
            f"compact objective {compact['objective']:.6f}, full objective {full['objective']:.6f}, "
            f"selected {selected}, LightGBM blend weight {blend_weight:.2f}"
        )
    else:
        outcome = "debug pipeline completed; feature selection deferred to full forward folds"
    entry = (
        "\n### Trigger-aware causal user-state matrix\n"
        "- run/experiment: generic_exp_0 lane0 | status: TESTED-KEPT\n"
        "- what: deployment-shaped badge state, all-table activity, generic vote crossings, tag progression, and award-batch fingerprints\n"
        f"- outcome: {outcome}; cache {cache_key}\n"
        "- takeaway: validation history stayed frozen before 2020-10-01 and test class history before 2021-01-01\n"
    )
    path = shared / "features_history.md"
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        os.write(descriptor, entry.encode())
    finally:
        os.close(descriptor)


def register_artifact(shared, cache_key):
    if not cache_key:
        return
    lock_path = shared / "artifacts.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = shared / "artifacts.json"
        try:
            records = json.loads(path.read_text()) if path.exists() else []
        except json.JSONDecodeError:
            records = []
        if not any(record.get("content_key") == cache_key for record in records):
            records.append({
                "name": "trigger-aware user-state feature matrix lane0",
                "path": cache_key,
                "description": "Exact-time all-table activity, crossing, and tag features aligned to post-2017 badge seeds",
                "content_key": cache_key,
                "rebuild_hint": "run main.py with the same sanitized rel-stack badges-class cache",
            })
            temporary = shared / f"artifacts.{os.getpid()}.tmp.json"
            temporary.write_text(json.dumps(records, indent=2))
            os.replace(temporary, path)
        fcntl.flock(lock, fcntl.LOCK_UN)


def serializable_internal(internal):
    output = {}
    for design, result in internal.items():
        output[design] = {key: value for key, value in result.items() if key != "predictions"}
    return output


def main():
    started = time.time()
    debug = is_debug()
    shared = shared_cache_dir()
    read_campaign_memory(shared)
    context = load_task(upto_test_timestamp=False)
    train = context.train.df[["Date", "Id", context.target_col]].copy()
    validation = context.val.df[["Date", "Id", context.target_col]].copy()
    test = context.test.df[["Date", "Id"]].copy()
    for frame in (train, validation, test):
        frame["row_idx"] = np.arange(len(frame), dtype=np.int64)
    train = train[train["Date"] >= pd.Timestamp("2017-01-01")].copy().reset_index(drop=True)
    if debug:
        train = train.tail(30000).reset_index(drop=True)
    train_cutoffs = train["Date"].dt.to_period("Q").dt.start_time
    validation_cutoff = pd.Timestamp("2020-10-01")
    test_cutoff = pd.Timestamp("2021-01-01")
    known_train = context.train.df[["Date", "Id", context.target_col]].rename(columns={context.target_col: "Class"})
    known_test = pd.concat([
        known_train,
        context.val.df[["Date", "Id", context.target_col]].rename(columns={context.target_col: "Class"}),
    ], ignore_index=True)
    badge_builder = BadgeFeatureBuilder(
        context.db.table_dict["badges"].df,
        context.db.table_dict["users"].df,
    )
    train_compact, train_seeds = badge_builder.transform(train, train_cutoffs, known_train)
    validation_compact, validation_seeds = badge_builder.transform(
        validation,
        pd.Series(validation_cutoff, index=np.arange(len(validation))),
        known_train,
    )
    test_compact, test_seeds = badge_builder.transform(
        test,
        pd.Series(test_cutoff, index=np.arange(len(test))),
        known_test,
    )
    compact_names = list(train_compact.columns)
    if compact_names != list(validation_compact.columns) or compact_names != list(test_compact.columns):
        raise RuntimeError("compact feature schema mismatch")
    combined_seeds = pd.concat([train_seeds, validation_seeds, test_seeds], ignore_index=True)
    compact_matrix = np.vstack([
        train_compact.to_numpy(dtype=np.float32),
        validation_compact.to_numpy(dtype=np.float32),
        test_compact.to_numpy(dtype=np.float32),
    ])
    compact_count = compact_matrix.shape[1]
    feature_names = compact_names.copy()
    cache_key = "debug_compact_only"
    if debug:
        feature_matrix = compact_matrix
    else:
        activity, cache_key = load_or_build_activity(context.db, combined_seeds, shared)
        activity_matrix = activity.to_numpy(dtype=np.float32)
        cohort_source = pd.DataFrame({
            "Date": combined_seeds["Date"].to_numpy(),
            "post_life": activity.get("post_posts_life", pd.Series(np.zeros(len(activity)))).to_numpy(),
            "vote_90": activity.get("vote_votes_received_90d", pd.Series(np.zeros(len(activity)))).to_numpy(),
            "tag_top": activity.get("tag_tag_top_share", pd.Series(np.zeros(len(activity)))).to_numpy(),
            "diversity": activity.get("activity_diversity_90d", pd.Series(np.zeros(len(activity)))).to_numpy(),
        })
        cohort_columns = []
        cohort_blocks = []
        for column in ("post_life", "vote_90", "tag_top", "diversity"):
            mean = cohort_source.groupby("Date", sort=False)[column].transform("mean").to_numpy(dtype=np.float32)
            std = cohort_source.groupby("Date", sort=False)[column].transform("std").fillna(0).to_numpy(dtype=np.float32)
            cohort_blocks.extend([mean, std])
            cohort_columns.extend([f"award_cohort_{column}_mean", f"award_cohort_{column}_std"])
        feature_matrix = np.column_stack([compact_matrix, activity_matrix, *cohort_blocks]).astype(np.float32)
        feature_names.extend(list(activity.columns))
        feature_names.extend(cohort_columns)
        register_artifact(shared, cache_key)
    train_count = len(train)
    validation_count = len(validation)
    train_features = feature_matrix[:train_count]
    validation_features = feature_matrix[train_count:train_count + validation_count]
    test_features = feature_matrix[train_count + validation_count:]
    train_labels = train[context.target_col].to_numpy(dtype=np.int64)
    validation_labels = validation[context.target_col].to_numpy(dtype=np.int64)
    badge_user_map = context.db.table_dict["badges"].df.set_index("Id")["UserId"]
    train_users = train["Id"].map(badge_user_map).to_numpy(dtype=np.int64)
    internal = {}
    cat_internal = None
    blend_candidates = []
    blend_weight = 1.0
    cat_iterations = 0
    if debug:
        selected = "compact"
        feature_indices = np.arange(compact_count, dtype=np.int64)
        iterations = 50
    else:
        selected, feature_indices, selected_result, internal = compare_feature_designs(
            train_features,
            train_labels,
            train["Date"].to_numpy(),
            train_users,
            feature_names,
            compact_count,
        )
        iterations = selected_result["median_iteration"]
        try:
            cat_internal = forward_catboost(
                train_features,
                train_labels,
                train["Date"].to_numpy(),
                train_users,
                train_features[:, feature_names.index("batch_batch_size")],
                train_features[:, feature_names.index("prior_badge_count")],
                feature_indices,
                iterations,
            )
            blend, blend_candidates = choose_blend(selected_result, cat_internal, train_labels, train_users)
            blend_weight = float(blend["lightgbm_weight"])
            cat_iterations = cat_internal["median_iteration"]
        except Exception as error:
            print(f"[internal] catboost unavailable type={type(error).__name__} detail={str(error)[:300]}")
            blend_weight = 1.0
    print(f"[model] selected={selected} features={len(feature_indices)} iterations={iterations} blend_lightgbm={blend_weight:.2f} elapsed={time.time() - started:.1f}s")
    validation_lightgbm, models_a = train_lightgbm_ensemble(
        train_features,
        train_labels,
        validation_features,
        feature_indices,
        iterations,
        debug=debug,
    )
    validation_prediction = validation_lightgbm
    cat_a = None
    if blend_weight < 1.0:
        validation_catboost, cat_a = train_catboost(
            train_features, train_labels, validation_features, feature_indices, cat_iterations
        )
        validation_prediction = blend_weight * validation_lightgbm + (1 - blend_weight) * validation_catboost
    permanent_validation_prediction = validation_prediction.copy()
    if debug:
        allowed_train = max(1, 30000 - len(validation_labels))
        model_b_features = np.vstack([train_features[-allowed_train:], validation_features])
        model_b_labels = np.concatenate([train_labels[-allowed_train:], validation_labels])
    else:
        model_b_features = np.vstack([train_features, validation_features])
        model_b_labels = np.concatenate([train_labels, validation_labels])
    test_lightgbm, models_b = train_lightgbm_ensemble(
        model_b_features,
        model_b_labels,
        test_features,
        feature_indices,
        iterations,
        debug=debug,
    )
    test_prediction = test_lightgbm
    cat_b = None
    if blend_weight < 1.0:
        test_catboost, cat_b = train_catboost(
            model_b_features, model_b_labels, test_features, feature_indices, cat_iterations
        )
        test_prediction = blend_weight * test_lightgbm + (1 - blend_weight) * test_catboost
    if not np.array_equal(permanent_validation_prediction, validation_prediction):
        raise RuntimeError("Model B altered permanent Model A validation predictions")
    validation_prediction = np.asarray(permanent_validation_prediction, dtype=np.float64)
    test_prediction = np.asarray(test_prediction, dtype=np.float64)
    validation_prediction /= validation_prediction.sum(axis=1, keepdims=True)
    test_prediction /= test_prediction.sum(axis=1, keepdims=True)
    save_predictions(validation_prediction, test_prediction)
    diagnostics = {
        "debug": debug,
        "elapsed_seconds": time.time() - started,
        "feature_count": int(len(feature_indices)),
        "compact_feature_count": int(compact_count),
        "selected_design": selected,
        "lightgbm_iterations": int(iterations),
        "catboost_iterations": int(cat_iterations),
        "lightgbm_blend_weight": float(blend_weight),
        "internal": serializable_internal(internal) if internal else {},
        "catboost_internal": {key: value for key, value in cat_internal.items() if key != "predictions"} if cat_internal else {},
        "blend_candidates": blend_candidates,
        "validation_fit": "Model A: train labels only",
        "test_fit": "Model B: train plus validation labels",
        "cache_key": cache_key,
    }
    (run_data_dir() / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    check_predictions()
    append_campaign_result(shared, internal, selected, blend_weight, cache_key)
    print(f"[done] val={validation_prediction.shape} test={test_prediction.shape} elapsed={time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
