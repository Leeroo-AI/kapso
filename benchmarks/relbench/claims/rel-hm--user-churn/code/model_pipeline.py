from __future__ import annotations

import gc
import itertools
import json
import math
import os
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


warnings.filterwarnings("ignore")

META_COLUMNS = {"_split", "_row_id", "timestamp", "customer_id", "label"}
CATEGORICAL_NAMES = {
    "membership_code",
    "news_code",
    "postal_bucket",
    "age_bucket",
    "dominant_channel",
    "dominant_product_group",
    "dominant_department",
    "dominant_index_group",
    "dominant_section",
}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in META_COLUMNS]


def matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].replace([np.inf, -np.inf], np.nan)
    return values.to_numpy(dtype=np.float32, copy=True)


def rank_unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (rankdata(values, method="average") - 0.5) / len(values)


def temporal_weights(timestamps: pd.Series, reference: pd.Timestamp, half_life: int | None) -> np.ndarray | None:
    if half_life is None:
        return None
    age = (reference - pd.to_datetime(timestamps)).dt.total_seconds().to_numpy() / 86400.0
    return np.exp(-math.log(2.0) * np.maximum(age, 0) / (7.0 * half_life)).astype(np.float32)


def sampled_indices(frame: pd.DataFrame, eligible: np.ndarray, limit: int, seed: int) -> np.ndarray:
    indices = np.flatnonzero(eligible)
    if len(indices) <= limit:
        return indices
    timestamps = pd.to_datetime(frame.iloc[indices]["timestamp"])
    newest = timestamps.max()
    recent_mask = timestamps >= newest - pd.Timedelta(days=91)
    recent = indices[recent_mask.to_numpy()]
    older = indices[~recent_mask.to_numpy()]
    recent_limit = min(len(recent), int(limit * 0.7))
    older_limit = limit - recent_limit
    rng = np.random.default_rng(seed)
    if len(recent) > recent_limit:
        recent = rng.choice(recent, recent_limit, replace=False)
    if len(older) > older_limit:
        older = rng.choice(older, older_limit, replace=False)
    result = np.concatenate([recent, older])
    result.sort()
    return result


def lgb_fit_predict(
    frame: pd.DataFrame,
    columns: list[str],
    train_indices: np.ndarray,
    hold_indices: np.ndarray,
    leaves: int,
    min_leaf: int,
    half_life: int | None,
    rounds: int,
    debug: bool,
) -> tuple[np.ndarray, int, float]:
    x_train = matrix(frame.iloc[train_indices], columns)
    x_hold = matrix(frame.iloc[hold_indices], columns)
    y_train = frame.iloc[train_indices]["label"].to_numpy(dtype=np.int8)
    y_hold = frame.iloc[hold_indices]["label"].to_numpy(dtype=np.int8)
    reference = pd.Timestamp(frame.iloc[hold_indices]["timestamp"].min()) - pd.Timedelta(days=7)
    weights = temporal_weights(frame.iloc[train_indices]["timestamp"], reference, half_life)
    train_set = lgb.Dataset(x_train, label=y_train, weight=weights, free_raw_data=True)
    hold_set = lgb.Dataset(x_hold, label=y_hold, reference=train_set, free_raw_data=True)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.035,
        "num_leaves": leaves,
        "min_data_in_leaf": min_leaf,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "verbosity": -1,
        "num_threads": max(1, int(os.environ.get("OMP_NUM_THREADS", "11"))),
        "seed": 1337,
        "feature_fraction_seed": 1337,
        "bagging_seed": 1337,
    }
    model = lgb.train(
        params,
        train_set,
        num_boost_round=rounds,
        valid_sets=[hold_set],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    prediction = model.predict(x_hold, num_iteration=model.best_iteration)
    auc = float(roc_auc_score(y_hold, prediction))
    best_iteration = int(model.best_iteration or rounds)
    del x_train, x_hold, train_set, hold_set, model
    gc.collect()
    return prediction, best_iteration, auc


def bank_core_probe(frame: pd.DataFrame, cache_root: Path, debug: bool) -> dict[str, object]:
    path = cache_root / "core_probe.json"
    prediction_path = cache_root / "core_probe_predictions.npz"
    if path.exists() and prediction_path.exists():
        result = json.loads(path.read_text())
        print(f"[core_probe] cached {json.dumps(result, separators=(',', ':'))}", flush=True)
        return result
    start = time.time()
    train = frame[frame["_split"] == "train"].reset_index(drop=True)
    timestamps = np.sort(pd.to_datetime(train["timestamp"].unique()))
    cutoff = pd.Timestamp(timestamps[-1])
    eligible = pd.to_datetime(train["timestamp"]).to_numpy() <= np.datetime64(cutoff - pd.Timedelta(days=7))
    train_indices = sampled_indices(train, eligible, 180_000 if debug else 1_000_000, 1301)
    hold_indices = np.flatnonzero(pd.to_datetime(train["timestamp"]).to_numpy() == np.datetime64(cutoff))
    columns = feature_columns(train)
    prediction, best_iteration, auc = lgb_fit_predict(
        train,
        columns,
        train_indices,
        hold_indices,
        63,
        1000,
        None,
        80 if debug else 700,
        debug,
    )
    result = {
        "cutoff": cutoff.isoformat(),
        "auc": auc,
        "best_iteration": best_iteration,
        "train_rows": int(len(train_indices)),
        "holdout_rows": int(len(hold_indices)),
        "feature_count": len(columns),
        "reported_0_7119_reproduced": bool(auc >= 0.7119),
        "elapsed_seconds": time.time() - start,
    }
    np.savez_compressed(
        prediction_path,
        row_id=train.iloc[hold_indices]["_row_id"].to_numpy(),
        prediction=prediction,
        label=train.iloc[hold_indices]["label"].to_numpy(),
    )
    path.write_text(json.dumps(result, indent=2))
    print(f"[core_probe] {json.dumps(result, separators=(',', ':'))}", flush=True)
    return result


def recent_folds(frame: pd.DataFrame, debug: bool) -> list[pd.Timestamp]:
    timestamps = np.sort(pd.to_datetime(frame.loc[frame["_split"] == "train", "timestamp"].unique()))
    count = min(4, len(timestamps) - 1)
    if debug:
        count = min(2, count)
    return [pd.Timestamp(value) for value in timestamps[-count:]]


def candidate_key(leaves: int, min_leaf: int, half_life: int | None) -> str:
    return f"l{leaves}_m{min_leaf}_h{half_life if half_life is not None else 'none'}"


def choose_stable(scores: dict[str, list[float]], complexity: dict[str, tuple[int, int, int]]) -> str:
    means = {key: float(np.mean(value)) for key, value in scores.items()}
    best_mean = max(means.values())
    tied = [key for key, value in means.items() if best_mean - value < 0.002]
    return min(tied, key=lambda key: (complexity[key], -means[key]))


def bootstrap_auc_delta(
    labels: list[np.ndarray],
    baseline: list[np.ndarray],
    candidate: list[np.ndarray],
    repetitions: int = 128,
) -> dict[str, float]:
    rng = np.random.default_rng(7331)
    deltas = []
    for _ in range(repetitions):
        fold_deltas = []
        for y, base_prediction, candidate_prediction in zip(labels, baseline, candidate):
            indices = rng.integers(0, len(y), size=len(y))
            sampled_y = y[indices]
            fold_deltas.append(
                roc_auc_score(sampled_y, candidate_prediction[indices])
                - roc_auc_score(sampled_y, base_prediction[indices])
            )
        deltas.append(float(np.mean(fold_deltas)))
    values = np.asarray(deltas)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def run_lgb_selection(frame: pd.DataFrame, core_columns: list[str], all_columns: list[str], debug: bool) -> dict[str, object]:
    folds = recent_folds(frame, debug)
    train_frame = frame[frame["_split"] == "train"].reset_index(drop=True)
    fold_records: list[dict[str, object]] = []
    predictions: dict[str, list[np.ndarray]] = {}
    labels: list[np.ndarray] = []
    row_ids: list[np.ndarray] = []
    core_key = "core_l63_m1000_hnone"
    candidates = [(127, 1000, value) for value in (None, 26, 13)]
    sample_limit = 160_000 if debug else 800_000
    rounds = 90 if debug else 800
    for fold_index, cutoff in enumerate(folds):
        timestamps = pd.to_datetime(train_frame["timestamp"])
        eligible = timestamps.to_numpy() <= np.datetime64(cutoff - pd.Timedelta(days=7))
        train_indices = sampled_indices(train_frame, eligible, sample_limit, 2100 + fold_index)
        hold_indices = np.flatnonzero(timestamps.to_numpy() == np.datetime64(cutoff))
        labels.append(train_frame.iloc[hold_indices]["label"].to_numpy(dtype=np.int8))
        row_ids.append(train_frame.iloc[hold_indices]["_row_id"].to_numpy(dtype=np.int64))
        core_prediction, core_iteration, core_auc = lgb_fit_predict(
            train_frame,
            core_columns,
            train_indices,
            hold_indices,
            63,
            1000,
            None,
            rounds,
            debug,
        )
        predictions.setdefault(core_key, []).append(core_prediction)
        fold_record: dict[str, object] = {
            "cutoff": cutoff.isoformat(),
            "train_rows": int(len(train_indices)),
            "holdout_rows": int(len(hold_indices)),
            core_key: {"auc": core_auc, "iteration": core_iteration},
        }
        for leaves, min_leaf, half_life in candidates:
            key = candidate_key(leaves, min_leaf, half_life)
            prediction, iteration, auc = lgb_fit_predict(
                train_frame,
                all_columns,
                train_indices,
                hold_indices,
                leaves,
                min_leaf,
                half_life,
                rounds,
                debug,
            )
            predictions.setdefault(key, []).append(prediction)
            fold_record[key] = {"auc": auc, "iteration": iteration}
        fold_records.append(fold_record)
        print(f"[forward_fold] {json.dumps(fold_record, separators=(',', ':'))}", flush=True)
    weight_scores = {
        candidate_key(127, 1000, half_life): [
            float(record[candidate_key(127, 1000, half_life)]["auc"]) for record in fold_records
        ]
        for half_life in (None, 26, 13)
    }
    weight_complexity = {
        candidate_key(127, 1000, None): (0, 127, 1000),
        candidate_key(127, 1000, 26): (1, 127, 1000),
        candidate_key(127, 1000, 13): (2, 127, 1000),
    }
    selected_weight_key = choose_stable(weight_scores, weight_complexity)
    selected_half_life = None if selected_weight_key.endswith("hnone") else int(selected_weight_key.rsplit("h", 1)[1])
    tree_candidates = [(63, 300), (63, 1000), (127, 300), (127, 1000)]
    for leaves, min_leaf in tree_candidates:
        key = candidate_key(leaves, min_leaf, selected_half_life)
        if key in predictions:
            continue
        predictions[key] = []
        for fold_index, cutoff in enumerate(folds):
            timestamps = pd.to_datetime(train_frame["timestamp"])
            eligible = timestamps.to_numpy() <= np.datetime64(cutoff - pd.Timedelta(days=7))
            train_indices = sampled_indices(train_frame, eligible, sample_limit, 2100 + fold_index)
            hold_indices = np.flatnonzero(timestamps.to_numpy() == np.datetime64(cutoff))
            prediction, iteration, auc = lgb_fit_predict(
                train_frame,
                all_columns,
                train_indices,
                hold_indices,
                leaves,
                min_leaf,
                selected_half_life,
                rounds,
                debug,
            )
            predictions[key].append(prediction)
            fold_records[fold_index][key] = {"auc": auc, "iteration": iteration}
            print(
                f"[tree_gate] {json.dumps({'cutoff': cutoff.isoformat(), 'candidate': key, 'auc': auc, 'iteration': iteration}, separators=(',', ':'))}",
                flush=True,
            )
    tree_scores = {
        candidate_key(leaves, min_leaf, selected_half_life): [
            float(record[candidate_key(leaves, min_leaf, selected_half_life)]["auc"]) for record in fold_records
        ]
        for leaves, min_leaf in tree_candidates
    }
    tree_complexity = {
        candidate_key(leaves, min_leaf, selected_half_life): (leaves, -min_leaf, 0)
        for leaves, min_leaf in tree_candidates
    }
    selected_key = choose_stable(tree_scores, tree_complexity)
    core_scores = [float(record[core_key]["auc"]) for record in fold_records]
    selected_scores = tree_scores[selected_key]
    mean_gain = float(np.mean(selected_scores)) - float(np.mean(core_scores))
    bootstrap = bootstrap_auc_delta(
        labels,
        predictions[core_key],
        predictions[selected_key],
    )
    use_wide = mean_gain >= 0.002 or bootstrap["lower_95"] > 0
    print(
        f"[feature_gate] {json.dumps({'candidate': selected_key, 'mean_gain': mean_gain, 'bootstrap': bootstrap, 'retained': use_wide}, separators=(',', ':'))}",
        flush=True,
    )
    if not use_wide:
        selected_key = core_key
    if selected_key == core_key:
        leaves, min_leaf, selected_half_life = 63, 1000, None
        selected_columns = core_columns
    else:
        pieces = selected_key.split("_")
        leaves = int(pieces[0][1:])
        min_leaf = int(pieces[1][1:])
        selected_columns = all_columns
    selected_iterations = [int(record[selected_key]["iteration"]) for record in fold_records]
    result: dict[str, object] = {
        "folds": fold_records,
        "selected_key": selected_key,
        "use_wide": use_wide,
        "feature_mean_gain": mean_gain,
        "feature_bootstrap": bootstrap,
        "leaves": leaves,
        "min_leaf": min_leaf,
        "half_life": selected_half_life,
        "rounds": int(np.clip(round(np.median(selected_iterations) * 1.15), 100 if debug else 300, 2200)),
        "columns": selected_columns,
        "fold_cutoffs": [value.isoformat() for value in folds],
        "labels": labels,
        "row_ids": row_ids,
        "lgb_predictions": predictions[selected_key],
    }
    return result


def catboost_fold(
    frame: pd.DataFrame,
    columns: list[str],
    train_indices: np.ndarray,
    hold_indices: np.ndarray,
    half_life: int | None,
    debug: bool,
) -> tuple[np.ndarray, int, float]:
    from catboost import CatBoostClassifier

    x_train = frame.iloc[train_indices][columns].replace([np.inf, -np.inf], np.nan).copy()
    x_hold = frame.iloc[hold_indices][columns].replace([np.inf, -np.inf], np.nan).copy()
    cat_names = [name for name in columns if name in CATEGORICAL_NAMES]
    for name in cat_names:
        x_train[name] = x_train[name].fillna(-1).astype(np.int64)
        x_hold[name] = x_hold[name].fillna(-1).astype(np.int64)
    y_train = frame.iloc[train_indices]["label"].to_numpy(dtype=np.int8)
    y_hold = frame.iloc[hold_indices]["label"].to_numpy(dtype=np.int8)
    reference = pd.Timestamp(frame.iloc[hold_indices]["timestamp"].min()) - pd.Timedelta(days=7)
    weights = temporal_weights(frame.iloc[train_indices]["timestamp"], reference, half_life)
    model = CatBoostClassifier(
        iterations=120 if debug else 900,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=10,
        border_count=128,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        has_time=True,
        random_seed=1337,
        allow_writing_files=False,
        verbose=False,
        od_type="Iter",
        od_wait=100,
        thread_count=max(1, int(os.environ.get("OMP_NUM_THREADS", "11"))),
    )
    model.fit(
        x_train,
        y_train,
        cat_features=cat_names,
        sample_weight=weights,
        eval_set=(x_hold, y_hold),
        verbose=False,
    )
    prediction = model.predict_proba(x_hold)[:, 1]
    auc = float(roc_auc_score(y_hold, prediction))
    best = int(model.get_best_iteration())
    if best <= 0:
        best = 120 if debug else 900
    del x_train, x_hold, model
    gc.collect()
    return prediction, best, auc


def xgboost_fold(
    frame: pd.DataFrame,
    columns: list[str],
    train_indices: np.ndarray,
    hold_indices: np.ndarray,
    half_life: int | None,
    debug: bool,
) -> tuple[np.ndarray, int, float]:
    import xgboost as xgb

    x_train = matrix(frame.iloc[train_indices], columns)
    x_hold = matrix(frame.iloc[hold_indices], columns)
    y_train = frame.iloc[train_indices]["label"].to_numpy(dtype=np.int8)
    y_hold = frame.iloc[hold_indices]["label"].to_numpy(dtype=np.int8)
    reference = pd.Timestamp(frame.iloc[hold_indices]["timestamp"].min()) - pd.Timedelta(days=7)
    weights = temporal_weights(frame.iloc[train_indices]["timestamp"], reference, half_life)
    dtrain = xgb.QuantileDMatrix(x_train, label=y_train, weight=weights)
    dhold = xgb.QuantileDMatrix(x_hold, label=y_hold, ref=dtrain)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "device": "cuda",
        "tree_method": "hist",
        "max_depth": 8,
        "eta": 0.04,
        "min_child_weight": 50,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 10,
        "seed": 1337,
        "nthread": max(1, int(os.environ.get("OMP_NUM_THREADS", "11"))),
    }
    rounds = 120 if debug else 900
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
        evals=[(dhold, "hold")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    prediction = model.predict(dhold, iteration_range=(0, model.best_iteration + 1))
    auc = float(roc_auc_score(y_hold, prediction))
    best = int(model.best_iteration + 1)
    del x_train, x_hold, dtrain, dhold, model
    gc.collect()
    return prediction, best, auc


def blend_auc(labels: list[np.ndarray], predictions: list[list[np.ndarray]], weights: list[float]) -> tuple[float, list[float], np.ndarray]:
    blended = []
    fold_scores = []
    for fold_index, y in enumerate(labels):
        values = sum(weight * rank_unit(model[fold_index]) for weight, model in zip(weights, predictions))
        blended.append(values)
        fold_scores.append(float(roc_auc_score(y, values)))
    pooled = np.concatenate(blended)
    pooled_auc = float(roc_auc_score(np.concatenate(labels), pooled))
    return pooled_auc, fold_scores, pooled


def gate_additional_models(frame: pd.DataFrame, selection: dict[str, object], debug: bool) -> dict[str, object]:
    train_frame = frame[frame["_split"] == "train"].reset_index(drop=True)
    columns = list(selection["columns"])
    folds = [pd.Timestamp(value) for value in selection["fold_cutoffs"]]
    labels = list(selection["labels"])
    lgb_predictions = list(selection["lgb_predictions"])
    sample_limit = 120_000 if debug else 650_000
    cat_predictions = []
    xgb_predictions = []
    cat_iterations = []
    xgb_iterations = []
    cat_scores = []
    xgb_scores = []
    for fold_index, cutoff in enumerate(folds):
        timestamps = pd.to_datetime(train_frame["timestamp"])
        eligible = timestamps.to_numpy() <= np.datetime64(cutoff - pd.Timedelta(days=7))
        train_indices = sampled_indices(train_frame, eligible, sample_limit, 3100 + fold_index)
        hold_indices = np.flatnonzero(timestamps.to_numpy() == np.datetime64(cutoff))
        cat_prediction, cat_iteration, cat_auc = catboost_fold(
            train_frame,
            columns,
            train_indices,
            hold_indices,
            selection["half_life"],
            debug,
        )
        cat_predictions.append(cat_prediction)
        cat_iterations.append(cat_iteration)
        cat_scores.append(cat_auc)
        xgb_prediction, xgb_iteration, xgb_auc = xgboost_fold(
            train_frame,
            columns,
            train_indices,
            hold_indices,
            selection["half_life"],
            debug,
        )
        xgb_predictions.append(xgb_prediction)
        xgb_iterations.append(xgb_iteration)
        xgb_scores.append(xgb_auc)
        print(
            f"[model_gate] {json.dumps({'cutoff': cutoff.isoformat(), 'cat_auc': cat_auc, 'cat_iteration': cat_iteration, 'xgb_auc': xgb_auc, 'xgb_iteration': xgb_iteration}, separators=(',', ':'))}",
            flush=True,
        )
    base_auc, base_folds, _ = blend_auc(labels, [lgb_predictions], [1.0])
    cat_candidates = []
    for cat_weight in np.arange(0.05, 0.55, 0.05):
        pooled, fold_scores, pooled_prediction = blend_auc(
            labels,
            [lgb_predictions, cat_predictions],
            [1.0 - float(cat_weight), float(cat_weight)],
        )
        cat_candidates.append((pooled, float(cat_weight), fold_scores, pooled_prediction))
    cat_best = max(cat_candidates, key=lambda value: value[0])
    cat_gain = float(np.mean(cat_best[2]) - np.mean(base_folds))
    use_cat = cat_gain >= 0.002
    if use_cat:
        active_predictions = [lgb_predictions, cat_predictions]
        active_weights = [1.0 - cat_best[1], cat_best[1]]
        active_fold_scores = cat_best[2]
        active_auc = cat_best[0]
    else:
        active_predictions = [lgb_predictions]
        active_weights = [1.0]
        active_fold_scores = base_folds
        active_auc = base_auc
    xgb_candidates = []
    for xgb_weight in np.arange(0.05, 0.45, 0.05):
        scaled = [weight * (1.0 - float(xgb_weight)) for weight in active_weights]
        weights = scaled + [float(xgb_weight)]
        pooled, fold_scores, pooled_prediction = blend_auc(
            labels,
            active_predictions + [xgb_predictions],
            weights,
        )
        xgb_candidates.append((pooled, weights, fold_scores, pooled_prediction))
    xgb_best = max(xgb_candidates, key=lambda value: value[0])
    improving_folds = sum(
        candidate > baseline + 0.0001 for candidate, baseline in zip(xgb_best[2], active_fold_scores)
    )
    xgb_gain = float(np.mean(xgb_best[2]) - np.mean(active_fold_scores))
    use_xgb = improving_folds >= 3 and xgb_gain >= 0.002
    if use_xgb:
        model_names = (["lgb", "cat"] if use_cat else ["lgb"]) + ["xgb"]
        blend_weights = xgb_best[1]
        pooled_prediction = xgb_best[3]
        final_fold_scores = xgb_best[2]
        final_auc = xgb_best[0]
    elif use_cat:
        model_names = ["lgb", "cat"]
        blend_weights = active_weights
        pooled_prediction = cat_best[3]
        final_fold_scores = active_fold_scores
        final_auc = active_auc
    else:
        model_names = ["lgb"]
        blend_weights = [1.0]
        _, final_fold_scores, pooled_prediction = blend_auc(labels, [lgb_predictions], [1.0])
        final_auc = base_auc
    pooled_labels = np.concatenate(labels)
    platt = LogisticRegression(C=10.0, solver="lbfgs", max_iter=500)
    platt.fit(pooled_prediction.reshape(-1, 1), pooled_labels)
    result = {
        "models": model_names,
        "weights": [float(value) for value in blend_weights],
        "cat_rounds": int(np.clip(round(np.median(cat_iterations) * 1.15), 100 if debug else 300, 1800)),
        "xgb_rounds": int(np.clip(round(np.median(xgb_iterations) * 1.15), 100 if debug else 300, 1800)),
        "cat_scores": cat_scores,
        "xgb_scores": xgb_scores,
        "base_auc": base_auc,
        "cat_gain": cat_gain,
        "xgb_gain": xgb_gain,
        "xgb_improving_folds": improving_folds,
        "final_auc": final_auc,
        "final_fold_scores": final_fold_scores,
        "platt_coef": float(platt.coef_[0, 0]),
        "platt_intercept": float(platt.intercept_[0]),
        "pooled_prediction": pooled_prediction,
    }
    print(
        f"[blend_gate] {json.dumps({key: value for key, value in result.items() if key not in {'pooled_prediction'}}, separators=(',', ':'))}",
        flush=True,
    )
    return result


def train_lgb_final(
    train_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    columns: list[str],
    selection: dict[str, object],
) -> np.ndarray:
    x_train = matrix(train_frame, columns)
    x_predict = matrix(predict_frame, columns)
    y_train = train_frame["label"].to_numpy(dtype=np.int8)
    reference = pd.Timestamp(train_frame["timestamp"].max())
    weights = temporal_weights(train_frame["timestamp"], reference, selection["half_life"])
    dataset = lgb.Dataset(x_train, label=y_train, weight=weights, free_raw_data=True)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.035,
        "num_leaves": int(selection["leaves"]),
        "min_data_in_leaf": int(selection["min_leaf"]),
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "verbosity": -1,
        "num_threads": max(1, int(os.environ.get("OMP_NUM_THREADS", "11"))),
        "seed": 1337,
    }
    model = lgb.train(
        params,
        dataset,
        num_boost_round=int(selection["rounds"]),
        callbacks=[lgb.log_evaluation(0)],
    )
    prediction = model.predict(x_predict)
    del x_train, x_predict, dataset, model
    gc.collect()
    return prediction


def train_cat_final(
    train_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    columns: list[str],
    selection: dict[str, object],
    rounds: int,
) -> np.ndarray:
    from catboost import CatBoostClassifier

    x_train = train_frame[columns].replace([np.inf, -np.inf], np.nan).copy()
    x_predict = predict_frame[columns].replace([np.inf, -np.inf], np.nan).copy()
    cat_names = [name for name in columns if name in CATEGORICAL_NAMES]
    for name in cat_names:
        x_train[name] = x_train[name].fillna(-1).astype(np.int64)
        x_predict[name] = x_predict[name].fillna(-1).astype(np.int64)
    reference = pd.Timestamp(train_frame["timestamp"].max())
    weights = temporal_weights(train_frame["timestamp"], reference, selection["half_life"])
    model = CatBoostClassifier(
        iterations=rounds,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=10,
        border_count=128,
        loss_function="Logloss",
        task_type="GPU",
        devices="0",
        has_time=True,
        random_seed=1337,
        allow_writing_files=False,
        verbose=False,
        thread_count=max(1, int(os.environ.get("OMP_NUM_THREADS", "11"))),
    )
    model.fit(
        x_train,
        train_frame["label"].to_numpy(dtype=np.int8),
        cat_features=cat_names,
        sample_weight=weights,
        verbose=False,
    )
    prediction = model.predict_proba(x_predict)[:, 1]
    del x_train, x_predict, model
    gc.collect()
    return prediction


def train_xgb_final(
    train_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    columns: list[str],
    selection: dict[str, object],
    rounds: int,
) -> np.ndarray:
    import xgboost as xgb

    x_train = matrix(train_frame, columns)
    x_predict = matrix(predict_frame, columns)
    reference = pd.Timestamp(train_frame["timestamp"].max())
    weights = temporal_weights(train_frame["timestamp"], reference, selection["half_life"])
    dtrain = xgb.QuantileDMatrix(x_train, label=train_frame["label"].to_numpy(dtype=np.int8), weight=weights)
    dpredict = xgb.QuantileDMatrix(x_predict, ref=dtrain)
    params = {
        "objective": "binary:logistic",
        "device": "cuda",
        "tree_method": "hist",
        "max_depth": 8,
        "eta": 0.04,
        "min_child_weight": 50,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 10,
        "seed": 1337,
        "nthread": max(1, int(os.environ.get("OMP_NUM_THREADS", "11"))),
    }
    model = xgb.train(params, dtrain, num_boost_round=rounds, verbose_eval=False)
    prediction = model.predict(dpredict)
    del x_train, x_predict, dtrain, dpredict, model
    gc.collect()
    return prediction


def final_predictions(
    frame: pd.DataFrame,
    selection: dict[str, object],
    gate: dict[str, object],
    debug: bool,
) -> tuple[np.ndarray, np.ndarray]:
    columns = list(selection["columns"])
    train = frame[frame["_split"] == "train"].sort_values(["timestamp", "_row_id"]).reset_index(drop=True)
    val = frame[frame["_split"] == "val"].sort_values("_row_id").reset_index(drop=True)
    test = frame[frame["_split"] == "test"].sort_values("_row_id").reset_index(drop=True)
    model_a_predictions = []
    model_a_predictions.append(train_lgb_final(train, val, columns, selection))
    if "cat" in gate["models"]:
        model_a_predictions.append(train_cat_final(train, val, columns, selection, int(gate["cat_rounds"])))
    if "xgb" in gate["models"]:
        model_a_predictions.append(train_xgb_final(train, val, columns, selection, int(gate["xgb_rounds"])))
    train_b = pd.concat([train, val], ignore_index=True).sort_values(["timestamp", "_row_id"]).reset_index(drop=True)
    model_b_predictions = []
    model_b_predictions.append(train_lgb_final(train_b, test, columns, selection))
    if "cat" in gate["models"]:
        model_b_predictions.append(train_cat_final(train_b, test, columns, selection, int(gate["cat_rounds"])))
    if "xgb" in gate["models"]:
        model_b_predictions.append(train_xgb_final(train_b, test, columns, selection, int(gate["xgb_rounds"])))
    weights = list(gate["weights"])
    val_rank_blend = sum(weight * rank_unit(prediction) for weight, prediction in zip(weights, model_a_predictions))
    test_rank_blend = sum(weight * rank_unit(prediction) for weight, prediction in zip(weights, model_b_predictions))
    coefficient = float(gate["platt_coef"])
    intercept = float(gate["platt_intercept"])
    val_prediction = 1.0 / (1.0 + np.exp(-(coefficient * val_rank_blend + intercept)))
    test_prediction = 1.0 / (1.0 + np.exp(-(coefficient * test_rank_blend + intercept)))
    return np.clip(val_prediction, 1e-6, 1 - 1e-6), np.clip(test_prediction, 1e-6, 1 - 1e-6)


def stratified_report(frame: pd.DataFrame, selection: dict[str, object], gate: dict[str, object]) -> dict[str, object]:
    train = frame[frame["_split"] == "train"].reset_index(drop=True)
    row_ids = np.concatenate(selection["row_ids"])
    lookup = train.set_index("_row_id")
    oof = lookup.loc[row_ids].reset_index()
    prediction = np.asarray(gate["pooled_prediction"])
    labels = oof["label"].to_numpy(dtype=np.int8)
    axes: dict[str, pd.Series] = {}
    axes["history_depth"] = pd.cut(
        oof["active_week_sequence"],
        bins=[-np.inf, 1, 3, 7, np.inf],
        labels=["1", "2-3", "4-7", "8+"],
    )
    axes["basket_size"] = pd.cut(
        oof["tx_count_7d"],
        bins=[-np.inf, 1, 3, 7, np.inf],
        labels=["1", "2-3", "4-7", "8+"],
    )
    axes["recency"] = pd.cut(
        oof["days_since_last"],
        bins=[-np.inf, 0, 2, 5, np.inf],
        labels=["0", "1-2", "3-5", "6+"],
    )
    axes["age"] = pd.cut(
        oof["age"],
        bins=[-np.inf, 24, 39, 59, np.inf],
        labels=["<=24", "25-39", "40-59", "60+"],
    ).astype(object).fillna("missing")
    result: dict[str, object] = {}
    for axis_name, groups in axes.items():
        values = {}
        for group in pd.unique(groups):
            mask = np.asarray(groups == group)
            if mask.sum() == 0:
                continue
            auc = None
            if len(np.unique(labels[mask])) == 2:
                auc = float(roc_auc_score(labels[mask], prediction[mask]))
            values[str(group)] = {"count": int(mask.sum()), "auc": auc}
        result[axis_name] = values
    print(f"[oof_slices] {json.dumps(result, separators=(',', ':'))}", flush=True)
    return result
