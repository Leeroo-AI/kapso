import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from content_factors import product_metadata, ridge_content_factors
from forward_replay import FEATURE_NAMES, replay_forward
from relational_features import customer_metadata, fit_hierarchical, hierarchical_prediction, safe_review_projection, temporal_pair_repeat, temporal_training_features
from temporal_svdpp import fit_time_svdpp


warnings.filterwarnings("ignore")


def elapsed(started, phase):
    print(f"[pipeline] phase={phase} elapsed={time.time() - started:.1f}s", flush=True)


def date_day(value):
    return int((np.datetime64(value) - np.datetime64("2008-01-01")) / np.timedelta64(1, "D"))


def r2(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    denominator = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - pred) ** 2) / denominator) if denominator > 0 else 0.0


def sample_positions(positions, size, seed):
    if len(positions) <= size:
        return positions
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(positions), size=size, replace=False)
    return positions[np.sort(chosen)]


def chronological_events(chronology, chronology_days, start_exclusive, end_inclusive):
    begin = np.searchsorted(chronology_days, start_exclusive, side="right")
    end = np.searchsorted(chronology_days, end_inclusive, side="right")
    return chronology[begin:end]


def fit_fold(
    name,
    cutoff,
    episode_start,
    episode_end,
    sample_size,
    data,
    label_ids,
    review_arrays,
    chronology,
    chronology_days,
    metadata,
    customer,
    pair_repeat,
    n_users,
    n_items,
    factors,
    epochs,
    debug,
    seed,
    measure_frozen,
):
    fit_positions = np.flatnonzero(data["day"] <= cutoff)
    if debug:
        fit_positions = fit_positions[-min(len(fit_positions), 180000) :]
    episode_positions = np.flatnonzero((data["day"] >= episode_start) & (data["day"] <= episode_end) & (np.arange(len(label_ids)) < data["n_train"]))
    episode_positions = sample_positions(episode_positions, sample_size, seed)
    state = fit_time_svdpp(data, fit_positions, n_users, n_items, factors=factors, epochs=epochs, seed=seed)
    hierarchy = fit_hierarchical(data, fit_positions, metadata, customer, n_users, n_items)
    content = ridge_content_factors(state, metadata, alpha=20.0)
    history_fit_ids = chronology[: np.searchsorted(chronology_days, cutoff, side="right")]
    replay_ids = chronological_events(chronology, chronology_days, cutoff, episode_end)
    result = replay_forward(
        state,
        content,
        hierarchy,
        metadata,
        customer,
        review_arrays,
        history_fit_ids,
        replay_ids,
        label_ids[episode_positions],
        pair_repeat,
        keep_features=True,
        measure_frozen=measure_frozen,
    )
    target = data["rating"][episode_positions].astype(np.float32)
    print(f"[fold] name={name} rows={len(target)} raw_r2={r2(target, result.raw):.6f} prior_r2={r2(target, result.prior):.6f}", flush=True)
    del state, hierarchy, content
    gc.collect()
    return result, target, np.full(len(target), name, dtype=object)


def train_booster(features, target, raw, prior, debug):
    import xgboost as xgb
    from scipy.optimize import nnls

    rng = np.random.default_rng(1337)
    calibration = rng.random(len(target)) < 0.2
    train = ~calibration
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda:0",
        "eta": 0.035,
        "grow_policy": "lossguide",
        "max_depth": 0,
        "max_leaves": 96,
        "min_child_weight": 300,
        "reg_lambda": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": 1337,
        "verbosity": 0,
        "nthread": int(os.environ.get("OMP_NUM_THREADS", "11")),
    }
    rounds = 40 if debug else 1400
    provisional = xgb.train(params, xgb.DMatrix(features[train], label=target[train] - raw[train]), num_boost_round=rounds)
    residual_calibration = raw[calibration] + provisional.inplace_predict(features[calibration]).astype(np.float32)
    candidates = np.column_stack([raw[calibration], residual_calibration, prior[calibration]]).astype(np.float64)
    ridge = 100.0
    design = np.vstack([candidates, np.sqrt(ridge) * np.eye(3)])
    response = np.concatenate([target[calibration].astype(np.float64), np.sqrt(ridge) * np.full(3, 1 / 3)])
    blend, _ = nnls(design, response)
    blend = blend / max(blend.sum(), 1e-12)
    calibration_prediction = candidates @ blend
    print(f"[blend] weights={blend.tolist()} calibration_r2={r2(target[calibration], calibration_prediction):.6f}", flush=True)
    booster = xgb.train(params, xgb.DMatrix(features, label=target - raw), num_boost_round=rounds)
    return booster, blend.astype(np.float32), calibration, residual_calibration


def report_strata(features, target, prediction, fold_names):
    user_counts = np.expm1(features[:, FEATURE_NAMES.index("log_user_count")])
    cold_user = features[:, FEATURE_NAMES.index("cold_user")] > 0.5
    cold_item = features[:, FEATURE_NAMES.index("cold_item")] > 0.5
    buckets = [
        ("history_0", user_counts < 0.5),
        ("history_1_2", (user_counts >= 0.5) & (user_counts < 2.5)),
        ("history_3_10", (user_counts >= 2.5) & (user_counts < 10.5)),
        ("history_11_100", (user_counts >= 10.5) & (user_counts < 100.5)),
        ("history_gt100", user_counts >= 100.5),
        ("cold_user", cold_user),
        ("cold_item", cold_item),
        ("warm_both", ~cold_user & ~cold_item),
        ("long_lag", fold_names == "long"),
        ("short_lag", fold_names == "short"),
    ]
    output = {}
    for name, mask in buckets:
        count = int(mask.sum())
        if count > 2:
            output[name] = {"count": count, "r2": r2(target[mask], prediction[mask])}
    print(f"[internal_strata] {json.dumps(output, separators=(',', ':'))}", flush=True)
    return output


def deploy_chain(
    name,
    fit_positions,
    target_ids,
    cutoff,
    target_end,
    data,
    target_review_arrays,
    target_chronology,
    target_chronology_days,
    history_review_arrays,
    history_chronology,
    history_chronology_days,
    metadata,
    customer,
    pair_repeat,
    n_users,
    n_items,
    factors,
    epochs,
    booster,
    blend,
    debug,
):
    if debug:
        fit_positions = fit_positions[-min(len(fit_positions), 350000) :]
    state = fit_time_svdpp(data, fit_positions, n_users, n_items, factors=factors, epochs=epochs, seed=2027 if name == "A" else 2029)
    hierarchy = fit_hierarchical(data, fit_positions, metadata, customer, n_users, n_items)
    content = ridge_content_factors(state, metadata, alpha=20.0)
    history_fit_ids = history_chronology[: np.searchsorted(history_chronology_days, cutoff, side="right")]
    full_prediction = hierarchical_prediction(
        hierarchy,
        target_review_arrays["user"][target_ids],
        target_review_arrays["item"][target_ids],
        target_review_arrays["day"][target_ids].astype(np.float32),
        target_review_arrays["verified"][target_ids],
        metadata,
        customer,
    )
    if debug:
        target_order = np.argsort(target_review_arrays["day"][target_ids], kind="stable")
        selected_positions = target_order[: min(100000, len(target_ids))]
        collect_ids = target_ids[selected_positions]
        replay_end = int(target_review_arrays["day"][collect_ids].max())
    else:
        selected_positions = np.arange(len(target_ids))
        collect_ids = target_ids
        replay_end = target_end
    replay_ids = chronological_events(target_chronology, target_chronology_days, cutoff, replay_end)
    result = replay_forward(
        state,
        content,
        hierarchy,
        metadata,
        customer,
        target_review_arrays,
        history_fit_ids,
        replay_ids,
        collect_ids,
        pair_repeat,
        booster=booster,
        blend=blend,
        keep_features=False,
        history_review_arrays=history_review_arrays,
    )
    full_prediction[selected_positions] = result.prediction
    full_prediction = np.clip(full_prediction, 1.0, 5.0).astype(np.float32)
    print(f"[chain] name={name} fit_labels={len(fit_positions)} targets={len(target_ids)} mean={float(full_prediction.mean()):.6f}", flush=True)
    del state, hierarchy, content, result
    gc.collect()
    return full_prediction


def main():
    started = time.time()
    debug = "--debug" in sys.argv
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    dataset_name = os.environ["RELBENCH_DATASET"]
    task_name = os.environ["RELBENCH_TASK"]
    dataset = get_dataset(dataset_name, download=False)
    task = get_task(dataset_name, task_name, download=False)
    db_censored = dataset.get_db(upto_test_timestamp=True)
    db_full = dataset.get_db(upto_test_timestamp=False)
    review = safe_review_projection(db_censored)
    full_review = safe_review_projection(db_full)
    product = db_full.table_dict["product"].df.sort_values("product_id", kind="stable")
    customer_table = db_full.table_dict["customer"].df.sort_values("customer_id", kind="stable")
    train = task.get_table("train").df
    val = task.get_table("val").df
    test = task.get_table("test").df
    elapsed(started, "load")
    train_ids = train["primary_key"].to_numpy(dtype=np.int64, copy=False)
    val_ids = val["primary_key"].to_numpy(dtype=np.int64, copy=False)
    test_ids = test["primary_key"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(train["review_time"].to_numpy(), review["review_time"].to_numpy()[train_ids]):
        raise RuntimeError("train primary keys do not align to the censored review snapshot")
    if not np.array_equal(val["review_time"].to_numpy(), review["review_time"].to_numpy()[val_ids]):
        raise RuntimeError("validation primary keys do not align to the censored review snapshot")
    if not np.array_equal(test["review_time"].to_numpy(), full_review["review_time"].to_numpy()[test_ids]):
        raise RuntimeError("test primary keys do not align to the full review snapshot")
    label_ids = np.concatenate([train_ids, val_ids])
    n_train = len(train_ids)
    review_user = review["customer_id"].to_numpy(dtype=np.int32, copy=False)
    review_item = review["product_id"].to_numpy(dtype=np.int32, copy=False)
    review_day = ((review["review_time"].to_numpy(copy=False) - np.datetime64("2008-01-01")) / np.timedelta64(1, "D")).astype(np.int32)
    review_verified = review["verified"].to_numpy(dtype=bool, copy=False)
    review_month = (review["review_time"].to_numpy(copy=False).astype("datetime64[M]").astype(np.int64) % 12 + 1).astype(np.int8)
    full_review_user = full_review["customer_id"].to_numpy(dtype=np.int32, copy=False)
    full_review_item = full_review["product_id"].to_numpy(dtype=np.int32, copy=False)
    full_review_day = ((full_review["review_time"].to_numpy(copy=False) - np.datetime64("2008-01-01")) / np.timedelta64(1, "D")).astype(np.int32)
    full_review_verified = full_review["verified"].to_numpy(dtype=bool, copy=False)
    full_review_month = (full_review["review_time"].to_numpy(copy=False).astype("datetime64[M]").astype(np.int64) % 12 + 1).astype(np.int8)
    ratings = np.concatenate([train[task.target_col].to_numpy(dtype=np.float32), val[task.target_col].to_numpy(dtype=np.float32)])
    cache_dir = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    temporal = temporal_training_features(review, label_ids, len(product), cache_dir, history_width=8)
    data = {
        "user": review_user[label_ids],
        "item": review_item[label_ids],
        "day": review_day[label_ids],
        "verified": review_verified[label_ids],
        "rating": ratings,
        "history": temporal["history"],
        "user_count": temporal["user_count"],
        "user_stale": temporal["user_stale"],
        "item_count": temporal["item_count"],
        "item_stale": temporal["item_stale"],
        "n_train": n_train,
    }
    pair_repeat = temporal["pair_repeat"]
    full_pair_repeat = temporal_pair_repeat(full_review, len(product), cache_dir)
    metadata = product_metadata(product, cache_dir, debug=debug)
    customer = customer_metadata(customer_table)
    chronology = np.argsort(review_day, kind="stable")
    chronology_days = review_day[chronology]
    review_arrays = {"user": review_user, "item": review_item, "day": review_day, "verified": review_verified, "month": review_month}
    full_chronology = np.argsort(full_review_day, kind="stable")
    full_chronology_days = full_review_day[full_chronology]
    full_review_arrays = {"user": full_review_user, "item": full_review_item, "day": full_review_day, "verified": full_review_verified, "month": full_review_month}
    elapsed(started, "features")
    factors = 16 if debug else 64
    epochs = 1 if debug else 3
    sample_size = 30000 if debug else 650000
    long_result, long_target, long_names = fit_fold(
        "long",
        date_day("2012-12-31"),
        date_day("2015-01-01"),
        date_day("2015-09-30"),
        sample_size,
        data,
        label_ids,
        review_arrays,
        chronology,
        chronology_days,
        metadata,
        customer,
        pair_repeat,
        len(customer_table),
        len(product),
        factors,
        epochs,
        debug,
        1401,
        True,
    )
    short_result, short_target, short_names = fit_fold(
        "short",
        date_day("2015-06-30"),
        date_day("2015-07-01"),
        date_day("2015-09-30"),
        sample_size,
        data,
        label_ids,
        review_arrays,
        chronology,
        chronology_days,
        metadata,
        customer,
        pair_repeat,
        len(customer_table),
        len(product),
        factors,
        epochs,
        debug,
        1409,
        False,
    )
    frozen_gain = r2(long_target, long_result.raw) - r2(long_target, long_result.frozen_raw)
    print(f"[implicit_ablation] long_lag_dynamic_minus_frozen_r2={frozen_gain:.6f}", flush=True)
    features = np.concatenate([long_result.features, short_result.features])
    target = np.concatenate([long_target, short_target])
    raw = np.concatenate([long_result.raw, short_result.raw])
    prior = np.concatenate([long_result.prior, short_result.prior])
    fold_names = np.concatenate([long_names, short_names])
    del long_result, short_result
    booster, blend, calibration, residual_calibration = train_booster(features, target, raw, prior, debug)
    calibration_prediction = blend[0] * raw[calibration] + blend[1] * residual_calibration + blend[2] * prior[calibration]
    strata = report_strata(features[calibration], target[calibration], calibration_prediction, fold_names[calibration])
    elapsed(started, "internal_selection")
    chain_a_positions = np.arange(n_train, dtype=np.int64)
    val_prediction = deploy_chain(
        "A",
        chain_a_positions,
        val_ids,
        date_day("2015-09-30"),
        int(review_day[val_ids].max()),
        data,
        review_arrays,
        chronology,
        chronology_days,
        review_arrays,
        chronology,
        chronology_days,
        metadata,
        customer,
        pair_repeat,
        len(customer_table),
        len(product),
        factors,
        epochs,
        booster,
        blend,
        debug,
    )
    out = Path(os.environ["KAPSO_RUN_DATA_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "val_predictions.npy", val_prediction)
    elapsed(started, "chain_a_saved")
    chain_b_positions = np.arange(len(label_ids), dtype=np.int64)
    test_prediction = deploy_chain(
        "B",
        chain_b_positions,
        test_ids,
        date_day("2015-12-31"),
        int(full_review_day[test_ids].max()),
        data,
        full_review_arrays,
        full_chronology,
        full_chronology_days,
        review_arrays,
        chronology,
        chronology_days,
        metadata,
        customer,
        full_pair_repeat,
        len(customer_table),
        len(product),
        factors,
        epochs,
        booster,
        blend,
        debug,
    )
    np.save(out / "test_predictions.npy", test_prediction)
    diagnostics = {
        "debug": debug,
        "factors": factors,
        "epochs": epochs,
        "blend": blend.tolist(),
        "implicit_long_lag_gain": frozen_gain,
        "internal_strata": strata,
        "elapsed_seconds": time.time() - started,
    }
    (out / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    elapsed(started, "complete")


if __name__ == "__main__":
    main()
