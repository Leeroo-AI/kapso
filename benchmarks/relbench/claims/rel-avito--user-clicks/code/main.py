import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from relbench.tasks import get_task
from sklearn.metrics import roc_auc_score

from avito_features import (
    build_core_features,
    build_episodes,
    build_sketch_features,
    load_feature_cache,
    open_connection,
    save_feature_cache,
    verify_official_labels,
)
from avito_models import (
    categorical_columns,
    encode_categorical_matrix,
    fit_meta_from_oof,
    fit_predict_heads,
    predict_meta,
    propensity_columns,
    select_count_cap,
    select_meta,
    summarize_fold,
)


FEATURE_VERSION = "generic_exp_1_lane1_threshold_process_v2"


def output_dir():
    path = Path(os.environ["KAPSO_RUN_DATA_DIR"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_predictions(validation, test):
    path = output_dir()
    np.save(
        path / "val_predictions.npy",
        np.clip(np.asarray(validation, dtype=np.float64), 0.0, 1.0),
    )
    np.save(
        path / "test_predictions.npy",
        np.clip(np.asarray(test, dtype=np.float64), 0.0, 1.0),
    )


def cache_fingerprint():
    cache_root = Path(os.environ["RELBENCH_CACHE_DIR"]) / "rel-avito"
    digest = hashlib.sha256(FEATURE_VERSION.encode())
    for path in sorted(cache_root.rglob("*.parquet")):
        stat = path.stat()
        digest.update(path.name.encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()[:16]


def register_cache_artifact(path, content_key):
    shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    registry = shared / "artifacts.json"
    lock_path = shared / "artifacts.json.lock"
    shared.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if registry.exists():
            try:
                records = json.loads(registry.read_text())
            except json.JSONDecodeError:
                records = []
        else:
            records = []
        relative = path.relative_to(shared).as_posix()
        if not any(record.get("content_key") == content_key for record in records):
            records.append(
                {
                    "name": "rel-avito causal intensity feature matrix lane 1",
                    "path": relative,
                    "description": "Daily causal all-table aggregates and eight fixed 64-dimensional decayed CountSketch channels",
                    "content_key": content_key,
                    "rebuild_hint": "Run main.py with the same sanitized rel-avito cache and feature version",
                }
            )
            temporary = registry.with_suffix(f".tmp.{os.getpid()}")
            temporary.write_text(json.dumps(records, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def sample_by_anchor(frame, rows):
    sampled = []
    for _, group in frame.groupby("timestamp", sort=True):
        group = group.copy()
        group["_sample_hash"] = (
            group["UserID"].astype(np.uint64)
            * np.uint64(11400714819323198485)
        )
        sampled.append(group.nsmallest(min(rows, len(group)), "_sample_hash"))
    return pd.concat(sampled, ignore_index=True).drop(
        columns=["_sample_hash"]
    )


def build_seed_register(episodes, validation, test):
    seed = pd.concat(
        [
            episodes[["UserID", "timestamp"]],
            validation[["UserID", "timestamp"]],
            test[["UserID", "timestamp"]],
        ],
        ignore_index=True,
    )
    seed = seed.drop_duplicates(["timestamp", "UserID"], keep="first")
    seed = seed.sort_values(["timestamp", "UserID"]).reset_index(drop=True)
    seed.insert(0, "row_id", np.arange(len(seed), dtype=np.int64))
    return seed


def row_lookup(seeds, frame):
    index = pd.MultiIndex.from_frame(seeds[["timestamp", "UserID"]])
    keys = pd.MultiIndex.from_frame(frame[["timestamp", "UserID"]])
    rows = index.get_indexer(keys)
    if np.any(rows < 0):
        raise RuntimeError(
            f"feature alignment failed for {int((rows < 0).sum())} rows"
        )
    return rows


def construct_full_features(train, validation, test):
    start = time.time()
    fingerprint = cache_fingerprint()
    shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    cache_path = (
        shared / f"{FEATURE_VERSION}_{fingerprint}"
    )
    content_key = f"{FEATURE_VERSION}:{fingerprint}"
    if (cache_path / "complete.json").exists():
        result = load_feature_cache(cache_path)
        verify_official_labels(result[0], train, validation)
        print(
            f"[cache] loaded feature artifact path={cache_path.name} "
            f"rows={len(result[1])} elapsed={round(time.time() - start, 2)}s",
            flush=True,
        )
        return result
    con = open_connection()
    anchors = pd.date_range("2015-04-26", "2015-05-10", freq="D")
    episodes = build_episodes(con, anchors)
    checked, mismatch = verify_official_labels(
        episodes, train, validation
    )
    print(
        f"[labels] exact audit rows={checked} mismatches={mismatch} "
        f"model_a_rows={int((episodes.timestamp <= pd.Timestamp('2015-05-04')).sum())} "
        f"model_b_rows={len(episodes)} elapsed={round(time.time() - start, 2)}s",
        flush=True,
    )
    seeds = build_seed_register(episodes, validation, test)
    core, core_names = build_core_features(con, seeds, debug=False)
    sketches, sketch_names = build_sketch_features(con, seeds, debug=False)
    temporary = cache_path.with_name(
        f"{cache_path.name}.tmp.{os.getpid()}"
    )
    save_feature_cache(
        temporary,
        episodes,
        seeds,
        core,
        core_names,
        sketches,
        sketch_names,
    )
    os.replace(temporary, cache_path)
    register_cache_artifact(cache_path, content_key)
    print(
        f"[cache] saved feature artifact path={cache_path.name} "
        f"elapsed={round(time.time() - start, 2)}s",
        flush=True,
    )
    return load_feature_cache(cache_path)


def construct_debug_features(train, validation, test):
    start = time.time()
    con = open_connection()
    audit_anchors = pd.to_datetime(
        ["2015-04-26", "2015-04-30", "2015-05-04", "2015-05-08"]
    )
    audited = build_episodes(con, audit_anchors)
    checked, mismatch = verify_official_labels(
        audited, train, validation
    )
    generated = audited[
        audited["timestamp"].isin(
            pd.to_datetime(["2015-04-26", "2015-04-30"])
        )
    ]
    generated = sample_by_anchor(generated, 4000)
    validation_episode = audited[
        audited["timestamp"] == pd.Timestamp("2015-05-08")
    ]
    validation_episode = sample_by_anchor(validation_episode, 4000)
    episodes = pd.concat(
        [generated, validation_episode], ignore_index=True
    ).sort_values(["timestamp", "UserID"]).reset_index(drop=True)
    validation_sample = sample_by_anchor(validation, 4000)
    test_sample = sample_by_anchor(test, 6000)
    seeds = build_seed_register(
        episodes, validation_sample, test_sample
    )
    print(
        f"[labels] debug exact audit rows={checked} mismatches={mismatch} "
        f"training_rows={len(episodes)} feature_rows={len(seeds)}",
        flush=True,
    )
    core, core_names = build_core_features(con, seeds, debug=True)
    sketches, sketch_names = build_sketch_features(con, seeds, debug=True)
    print(
        f"[features] debug bundle elapsed={round(time.time() - start, 2)}s",
        flush=True,
    )
    return (
        episodes,
        seeds,
        core,
        core_names,
        sketches,
        sketch_names,
    )


def make_targets(episodes, mask):
    selected = episodes.loc[mask]
    return {
        "repeat": selected["repeat_click"].to_numpy(dtype=np.int8),
        "any": selected["any_click"].to_numpy(dtype=np.int8),
        "count": selected["click_count"].to_numpy(dtype=np.float32),
        "exposure": selected["future_exposure_count"].to_numpy(
            dtype=np.float32
        ),
    }


def generate_oof(
    matrix,
    names,
    seeds,
    episodes,
    anchors,
    rounds,
    count_caps,
):
    episode_rows = row_lookup(seeds, episodes)
    cold_index = names.index("cold_no_search")
    q_index = propensity_columns(names)
    categorical_index = categorical_columns(names)
    folds = []
    for fold_number, anchor in enumerate(pd.to_datetime(anchors)):
        train_endpoint = anchor - pd.Timedelta(days=4)
        train_mask = episodes["timestamp"] <= train_endpoint
        validation_mask = episodes["timestamp"] == anchor
        if not train_mask.any() or not validation_mask.any():
            continue
        train_rows = episode_rows[train_mask.to_numpy()]
        validation_rows = episode_rows[validation_mask.to_numpy()]
        targets = make_targets(episodes, train_mask)
        predictions, dispersions = fit_predict_heads(
            matrix[train_rows],
            targets,
            matrix[validation_rows],
            q_index,
            categorical_index,
            rounds,
            count_caps,
            1337 + fold_number * 97,
        )
        target = episodes.loc[
            validation_mask, "repeat_click"
        ].to_numpy(dtype=np.int8)
        summarize_fold(anchor, target, predictions, dispersions)
        folds.append(
            {
                "anchor": anchor,
                "target": target,
                "predictions": predictions,
                "dispersions": dispersions,
                "cold": np.nan_to_num(
                    matrix[validation_rows, cold_index],
                    nan=1.0,
                ),
            }
        )
    if not folds:
        raise RuntimeError("no valid purged forward folds were generated")
    return folds


def fit_final_chain(
    matrix,
    names,
    seeds,
    episodes,
    train_endpoint,
    predict_frame,
    rounds,
    count_cap,
    meta_model,
    meta_selection,
    seed,
):
    episode_rows = row_lookup(seeds, episodes)
    train_mask = episodes["timestamp"] <= pd.Timestamp(train_endpoint)
    train_rows = episode_rows[train_mask.to_numpy()]
    predict_rows = row_lookup(seeds, predict_frame)
    q_index = propensity_columns(names)
    categorical_index = categorical_columns(names)
    targets = make_targets(episodes, train_mask)
    predictions, dispersions = fit_predict_heads(
        matrix[train_rows],
        targets,
        matrix[predict_rows],
        q_index,
        categorical_index,
        rounds,
        [count_cap],
        seed,
    )
    cold_index = names.index("cold_no_search")
    cold = np.nan_to_num(matrix[predict_rows, cold_index], nan=1.0)
    final = predict_meta(
        meta_model,
        predictions,
        cold,
        meta_selection["heads"],
    )
    return final, predictions, dispersions, predict_rows


def slice_diagnostics(validation, predictions, matrix, rows, names):
    labels = validation["num_click"].to_numpy(dtype=np.int8)
    search = np.nan_to_num(
        matrix[rows, names.index("search_n_all")], nan=0.0
    )
    visit = np.nan_to_num(
        matrix[rows, names.index("visit_n_all")], nan=0.0
    )
    strata = {
        "search_cold": search == 0,
        "visit_only": (search == 0) & (visit > 0),
        "warm_low_search": (search > 0) & (search <= 20),
        "warm_high_search": search > 20,
    }
    result = {}
    for name, mask in strata.items():
        target = labels[mask]
        if len(target) and np.unique(target).size > 1:
            auc = float(roc_auc_score(target, predictions[mask]))
        else:
            auc = None
        result[name] = {
            "count": int(mask.sum()),
            "positive_rate": float(target.mean()) if len(target) else None,
            "roc_auc": auc,
        }
    print(
        f"[diagnostics] frozen_validation_slices={json.dumps(result)}",
        flush=True,
    )
    return result


def run_rolling(task):
    test = task.get_table("test").df
    con = duckdb.connect()
    root = (
        Path(os.environ["RELBENCH_CACHE_DIR"])
        / os.environ["RELBENCH_DATASET"]
        / "db"
    )
    con.execute(
        f"CREATE VIEW SearchInfo AS SELECT * FROM read_parquet('{(root / 'SearchInfo.parquet').as_posix()}')"
    )
    con.register("rolling_seed", test[["UserID", "timestamp"]])
    features = con.execute(
        """
        SELECT
            s.UserID,
            COUNT(i.SearchID) AS searches,
            MAX(i.SearchDate) AS recent
        FROM rolling_seed s
        LEFT JOIN SearchInfo i
          ON i.UserID = s.UserID
         AND i.SearchDate <= s.timestamp
        GROUP BY s.UserID
        """
    ).df()
    count_map = features.set_index("UserID")["searches"]
    counts = test["UserID"].map(count_map).fillna(0).to_numpy()
    prediction = np.clip(0.01 + 0.04 * np.log1p(counts), 0.001, 0.5)
    np.save(output_dir() / "test_predictions.npy", prediction)
    print(
        f"[rolling] wrote test predictions rows={len(prediction)}",
        flush=True,
    )


def main():
    start = time.time()
    debug = "--debug" in sys.argv
    task = get_task(
        os.environ["RELBENCH_DATASET"],
        os.environ["RELBENCH_TASK"],
        download=False,
    )
    val_path = (
        Path(os.environ["RELBENCH_CACHE_DIR"])
        / os.environ["RELBENCH_DATASET"]
        / "tasks"
        / os.environ["RELBENCH_TASK"]
        / "val.parquet"
    )
    if not val_path.exists() or len(task.get_table("val")) == 0:
        run_rolling(task)
        return
    train = task.get_table("train", mask_input_cols=False).df.copy()
    validation = task.get_table("val", mask_input_cols=False).df.copy()
    test = task.get_table("test").df.copy()
    prior = float(train["num_click"].mean())
    save_predictions(
        np.full(len(validation), prior),
        np.full(len(test), prior),
    )
    print(
        f"[run] mode={'debug' if debug else 'full'} "
        f"train={len(train)} val={len(validation)} test={len(test)} "
        f"prior={prior:.6f}",
        flush=True,
    )
    if debug:
        bundle = construct_debug_features(
            train, validation, test
        )
        rounds = {
            "binary": 25,
            "count": 30,
            "exposure": 25,
            "propensity": 25,
        }
        oof_anchors = ["2015-04-30", "2015-05-08"]
        model_a_fold_count = 1
        model_a_endpoint = "2015-04-30"
        model_b_endpoint = "2015-05-08"
    else:
        bundle = construct_full_features(
            train, validation, test
        )
        rounds = {
            "binary": 550,
            "count": 700,
            "exposure": 500,
            "propensity": 450,
        }
        oof_anchors = [
            "2015-04-30",
            "2015-05-02",
            "2015-05-04",
            "2015-05-06",
            "2015-05-08",
            "2015-05-10",
        ]
        model_a_fold_count = 3
        model_a_endpoint = "2015-05-04"
        model_b_endpoint = "2015-05-10"
    episodes, seeds, core, core_names, sketches, sketch_names = bundle
    matrix = np.concatenate(
        (
            np.asarray(core, dtype=np.float32),
            np.asarray(sketches, dtype=np.float32),
        ),
        axis=1,
    )
    names = list(core_names) + list(sketch_names)
    matrix = encode_categorical_matrix(
        matrix, categorical_columns(names)
    )
    print(
        f"[stage] bank direct hurdle poisson core-plus-sketch "
        f"matrix={matrix.shape} elapsed={round(time.time() - start, 2)}s",
        flush=True,
    )
    count_caps = [8, 16]
    folds = generate_oof(
        matrix,
        names,
        seeds,
        episodes,
        oof_anchors,
        rounds,
        count_caps,
    )
    model_a_folds = folds[:model_a_fold_count]
    model_b_folds = folds
    model_a_cap, model_a_cap_records = select_count_cap(
        model_a_folds, count_caps
    )
    model_b_cap, model_b_cap_records = select_count_cap(
        model_b_folds, count_caps
    )
    model_a_selection, model_a_meta_records = select_meta(
        model_a_folds, model_a_cap
    )
    model_b_selection, model_b_meta_records = select_meta(
        model_b_folds, model_b_cap
    )
    model_a_meta = fit_meta_from_oof(
        model_a_folds, model_a_selection
    )
    model_b_meta = fit_meta_from_oof(
        model_b_folds, model_b_selection
    )
    if debug:
        validation_predict = sample_by_anchor(validation, 4000)
        test_predict = sample_by_anchor(test, 6000)
    else:
        validation_predict = validation
        test_predict = test
    validation_sample_prediction, validation_heads, _, validation_rows = (
        fit_final_chain(
            matrix,
            names,
            seeds,
            episodes,
            model_a_endpoint,
            validation_predict,
            rounds,
            model_a_cap,
            model_a_meta,
            model_a_selection,
            8819,
        )
    )
    validation_prediction = np.full(len(validation), prior, dtype=np.float64)
    if debug:
        validation_positions = pd.MultiIndex.from_frame(
            validation[["timestamp", "UserID"]]
        ).get_indexer(
            pd.MultiIndex.from_frame(
                validation_predict[["timestamp", "UserID"]]
            )
        )
        validation_prediction[validation_positions] = (
            validation_sample_prediction
        )
    else:
        validation_prediction = validation_sample_prediction.copy()
    frozen_validation = validation_prediction.copy()
    print(
        f"[freeze] model_a validation predictions frozen "
        f"rows={len(frozen_validation)} elapsed={round(time.time() - start, 2)}s",
        flush=True,
    )
    test_sample_prediction, test_heads, _, _ = fit_final_chain(
        matrix,
        names,
        seeds,
        episodes,
        model_b_endpoint,
        test_predict,
        rounds,
        model_b_cap,
        model_b_meta,
        model_b_selection,
        9901,
    )
    test_prediction = np.full(len(test), prior, dtype=np.float64)
    if debug:
        test_positions = pd.MultiIndex.from_frame(
            test[["timestamp", "UserID"]]
        ).get_indexer(
            pd.MultiIndex.from_frame(
                test_predict[["timestamp", "UserID"]]
            )
        )
        test_prediction[test_positions] = test_sample_prediction
    else:
        test_prediction = test_sample_prediction.copy()
    save_predictions(frozen_validation, test_prediction)
    if debug:
        diagnostic_rows = row_lookup(seeds, validation_predict)
        diagnostic_prediction = validation_sample_prediction
        diagnostic_frame = validation_predict
    else:
        diagnostic_rows = validation_rows
        diagnostic_prediction = frozen_validation
        diagnostic_frame = validation
    slices = slice_diagnostics(
        diagnostic_frame,
        diagnostic_prediction,
        matrix,
        diagnostic_rows,
        names,
    )
    diagnostics = {
        "mode": "debug" if debug else "full",
        "feature_shape": list(matrix.shape),
        "model_a": {
            "count_cap": model_a_cap,
            "meta": model_a_selection,
            "cap_records": model_a_cap_records,
        },
        "model_b": {
            "count_cap": model_b_cap,
            "meta": model_b_selection,
            "cap_records": model_b_cap_records,
        },
        "validation_slices": slices,
        "validation_head_ranges": {
            name: [float(np.min(value)), float(np.max(value))]
            for name, value in validation_heads.items()
        },
        "test_head_ranges": {
            name: [float(np.min(value)), float(np.max(value))]
            for name, value in test_heads.items()
        },
        "elapsed_seconds": round(time.time() - start, 2),
    }
    (output_dir() / "metrics.json").write_text(
        json.dumps(diagnostics, indent=2)
    )
    print(
        f"[run] completed val={frozen_validation.shape} "
        f"test={test_prediction.shape} "
        f"elapsed={round(time.time() - start, 2)}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
