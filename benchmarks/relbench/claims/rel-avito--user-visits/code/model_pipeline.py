from __future__ import annotations

import gc
import json
import math
import time
import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


warnings.filterwarnings("ignore")
SEEDS = (17, 37)
FOLD_ORIGINS = pd.to_datetime(("2015-04-30", "2015-05-02", "2015-05-04"))
META_COLUMNS = {"sid", "origin", "UserID", "future_distinct_ads", "label", "source"}


@dataclass
class CrossValidationResult:
    name: str
    predictions: dict[pd.Timestamp, np.ndarray]
    labels: dict[pd.Timestamp, np.ndarray]
    users: dict[pd.Timestamp, np.ndarray]
    indices: dict[pd.Timestamp, np.ndarray]
    seen: dict[pd.Timestamp, np.ndarray]
    scores: dict[str, float]
    best_iterations: list[int]

    def concatenated(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        origins = list(FOLD_ORIGINS)
        return (
            np.concatenate([self.labels[origin] for origin in origins]),
            np.concatenate([self.predictions[origin] for origin in origins]),
            np.concatenate([self.users[origin] for origin in origins]),
        )


def all_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in META_COLUMNS
        and not pd.api.types.is_datetime64_any_dtype(frame[column])
        and frame[column].dtype != object
    ]


def core_feature_columns(frame: pd.DataFrame) -> list[str]:
    direct = {
        "UserAgentID", "UserAgentOSID", "UserDeviceID", "UserAgentFamilyID",
        "missing_agent", "missing_os", "missing_device", "missing_family",
        "origin_day_index", "origin_day_of_week", "origin_day_of_month", "origin_is_weekend",
        "session_count_all", "session_count_d1", "session_count_d7", "last_session_depth",
        "last_session_duration_min", "recent_session_duration_min", "time_since_session_hours",
        "visit_search_recency_gap", "visit_phone_recency_gap", "search_phone_recency_gap",
    }
    prefixes = (
        "visit_count_", "visit_ads_", "visit_ips_", "visit_days_", "visit_recency_", "visit_decay_",
        "search_count_", "search_recency_", "search_decay_",
        "phone_count_", "phone_ads_", "phone_recency_", "phone_decay_",
        "visit_momentum_", "search_momentum_", "phone_momentum_", "visit_distinct_ad_ratio_",
        "visit_to_phone_",
    )
    columns = []
    for column in all_feature_columns(frame):
        base = column
        for suffix in ("_origin_pct", "_origin_z", "_origin_leader_gap"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        if column in direct or base in direct or any(base.startswith(prefix) for prefix in prefixes):
            columns.append(column)
    return columns


def categorical_columns(feature_columns: list[str]) -> list[str]:
    names = {
        "UserAgentID", "UserAgentOSID", "UserDeviceID", "UserAgentFamilyID",
        "visit_mode_category", "visit_mode_parent", "visit_mode_region", "visit_mode_ip",
        "visit_last_category", "visit_last_region", "search_mode_category", "search_mode_parent",
        "search_mode_region", "search_mode_ip", "search_last_category", "search_last_region",
        "impression_mode_category", "impression_mode_region", "phone_mode_category", "phone_mode_region",
    }
    return [column for column in feature_columns if column in names]


def numeric_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    matrix = frame[columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32, na_value=np.nan)
    return matrix


def catboost_matrix(frame: pd.DataFrame, columns: list[str], cats: list[str]) -> pd.DataFrame:
    result = frame[columns].replace([np.inf, -np.inf], np.nan).copy()
    for column in cats:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(-1).round().astype("int64").astype(str)
    for column in set(columns) - set(cats):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(np.float32)
    return result


def balanced_origin_weights(frame: pd.DataFrame) -> np.ndarray:
    keys = frame[["origin", "label"]].copy()
    counts = keys.groupby(["origin", "label"])["label"].transform("size").to_numpy(dtype=np.float64)
    origins = max(1, keys["origin"].nunique())
    return (len(frame) / (2.0 * origins * counts)).astype(np.float32)


def fold_masks(frame: pd.DataFrame, validation_origin: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    train = frame["label"].notna() & (frame["origin"] <= validation_origin - pd.Timedelta(days=4))
    valid = frame["label"].notna() & (frame["origin"] == validation_origin)
    return train.to_numpy(), valid.to_numpy()


def lgb_parameters(seed: int) -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 8.0,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "verbosity": -1,
        "num_threads": 11,
        "force_col_wise": True,
    }


def bootstrap_auc_se(labels: np.ndarray, predictions: np.ndarray, users: np.ndarray, seed: int = 17) -> float:
    unique, inverse = np.unique(users, return_inverse=True)
    generator = np.random.default_rng(seed)
    estimates = []
    for _ in range(100):
        draws = generator.integers(0, len(unique), len(unique))
        cluster_weights = np.bincount(draws, minlength=len(unique))
        weights = cluster_weights[inverse]
        if weights[labels == 0].sum() and weights[labels == 1].sum():
            estimates.append(roc_auc_score(labels, predictions, sample_weight=weights))
    return float(np.std(estimates, ddof=1))


def bootstrap_gain_se(labels: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, users: np.ndarray) -> float:
    unique, inverse = np.unique(users, return_inverse=True)
    generator = np.random.default_rng(37)
    gains = []
    for _ in range(100):
        draws = generator.integers(0, len(unique), len(unique))
        cluster_weights = np.bincount(draws, minlength=len(unique))[inverse]
        if cluster_weights[labels == 0].sum() and cluster_weights[labels == 1].sum():
            gains.append(
                roc_auc_score(labels, candidate, sample_weight=cluster_weights)
                - roc_auc_score(labels, baseline, sample_weight=cluster_weights)
            )
    return float(np.std(gains, ddof=1))


def summarize_cv(name: str, frame: pd.DataFrame, result: CrossValidationResult) -> dict[str, object]:
    labels, predictions, users = result.concatenated()
    means = list(result.scores.values())
    indices = np.concatenate([result.indices[origin] for origin in FOLD_ORIGINS])
    rows = frame.iloc[indices]
    seen = np.concatenate([result.seen[origin] for origin in FOLD_ORIGINS])
    visit = pd.to_numeric(rows.get("visit_count_all", 0), errors="coerce").fillna(0).to_numpy()
    phone = pd.to_numeric(rows.get("phone_count_all", 0), errors="coerce").fillna(0).to_numpy()
    search = pd.to_numeric(rows.get("search_count_all", 0), errors="coerce").fillna(0).to_numpy()
    positive_visits = visit[visit > 0]
    heavy_threshold = float(np.quantile(positive_visits, 0.75)) if len(positive_visits) else 1.0
    masks = {
        "cold_no_visit": visit == 0,
        "sparse_visit": (visit > 0) & (visit <= 5),
        "heavy_visit": visit >= heavy_threshold,
        "seen_user": seen,
        "unseen_user": ~seen,
        "phone": phone > 0,
        "no_phone": phone == 0,
        "search": search > 0,
        "no_search": search == 0,
    }
    slices = {}
    for key, mask in masks.items():
        count = int(mask.sum())
        score = float(roc_auc_score(labels[mask], predictions[mask])) if count and len(np.unique(labels[mask])) == 2 else None
        slices[key] = {"count": count, "roc_auc": score}
    summary = {
        "model": name,
        "mean_roc_auc": float(np.mean(means)),
        "worst_origin_roc_auc": float(np.min(means)),
        "pooled_roc_auc": float(roc_auc_score(labels, predictions)),
        "user_bootstrap_se": bootstrap_auc_se(labels, predictions, users),
        "folds": result.scores,
        "slices": slices,
        "best_iterations": result.best_iterations,
    }
    print("[cv] " + json.dumps(summary, sort_keys=True, separators=(",", ":")), flush=True)
    return summary


def cross_validate_lgb(frame: pd.DataFrame, columns: list[str], cap: int = 1800) -> CrossValidationResult:
    matrix = numeric_matrix(frame, columns)
    predictions = {}
    labels = {}
    users = {}
    indices = {}
    seen = {}
    scores = {}
    best_iterations = []
    for origin in FOLD_ORIGINS:
        train_mask, valid_mask = fold_masks(frame, origin)
        train_rows = frame.loc[train_mask]
        fold_predictions = []
        fold_iterations = []
        for seed in SEEDS:
            train_set = lgb.Dataset(matrix[train_mask], label=train_rows["label"].to_numpy(), weight=balanced_origin_weights(train_rows), free_raw_data=True)
            valid_set = lgb.Dataset(matrix[valid_mask], label=frame.loc[valid_mask, "label"].to_numpy(), reference=train_set, free_raw_data=True)
            model = lgb.train(
                lgb_parameters(seed), train_set, num_boost_round=cap, valid_sets=[valid_set],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
            )
            fold_predictions.append(model.predict(matrix[valid_mask], num_iteration=model.best_iteration))
            fold_iterations.append(int(model.best_iteration))
            del model, train_set, valid_set
            gc.collect()
        pred = np.mean(fold_predictions, axis=0)
        key = origin.strftime("%Y-%m-%d")
        predictions[origin] = pred
        labels[origin] = frame.loc[valid_mask, "label"].to_numpy(dtype=np.int8)
        users[origin] = frame.loc[valid_mask, "UserID"].to_numpy()
        indices[origin] = np.flatnonzero(valid_mask)
        seen[origin] = frame.loc[valid_mask, "UserID"].isin(set(train_rows["UserID"])).to_numpy()
        scores[key] = float(roc_auc_score(labels[origin], pred))
        best_iterations.append(int(np.median(fold_iterations)))
    return CrossValidationResult("lightgbm", predictions, labels, users, indices, seen, scores, best_iterations)


def xgb_parameters(seed: int, iterations: int, early_stopping: bool) -> dict:
    parameters = {
        "objective": "rank:pairwise",
        "max_depth": 6,
        "min_child_weight": 30,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 10.0,
        "n_estimators": iterations,
        "random_state": seed,
        "tree_method": "hist",
        "device": "cuda",
        "n_jobs": 11,
        "verbosity": 0,
        "eval_metric": "auc",
        "lambdarank_pair_method": "mean",
        "lambdarank_num_pair_per_sample": 10,
    }
    if early_stopping:
        parameters["early_stopping_rounds"] = 100
    return parameters


def cross_validate_xgb(frame: pd.DataFrame, columns: list[str], cap: int = 1800) -> CrossValidationResult:
    matrix = numeric_matrix(frame, columns)
    predictions = {}
    labels = {}
    users = {}
    indices = {}
    seen = {}
    scores = {}
    best_iterations = []
    for origin in FOLD_ORIGINS:
        train_mask, valid_mask = fold_masks(frame, origin)
        train_rows = frame.loc[train_mask]
        train_qid = pd.factorize(train_rows["origin"], sort=True)[0].astype(np.int32)
        valid_qid = np.zeros(valid_mask.sum(), dtype=np.int32)
        fold_predictions = []
        fold_iterations = []
        for seed in SEEDS:
            model = xgb.XGBRanker(**xgb_parameters(seed, cap, True))
            model.fit(
                matrix[train_mask], train_rows["label"].to_numpy(dtype=np.int8), qid=train_qid,
                eval_set=[(matrix[valid_mask], frame.loc[valid_mask, "label"].to_numpy(dtype=np.int8))],
                eval_qid=[valid_qid], verbose=False,
            )
            fold_predictions.append(model.predict(matrix[valid_mask]))
            fold_iterations.append(int(model.best_iteration + 1))
            del model
            gc.collect()
        pred = np.mean(fold_predictions, axis=0)
        key = origin.strftime("%Y-%m-%d")
        predictions[origin] = pred
        labels[origin] = frame.loc[valid_mask, "label"].to_numpy(dtype=np.int8)
        users[origin] = frame.loc[valid_mask, "UserID"].to_numpy()
        indices[origin] = np.flatnonzero(valid_mask)
        seen[origin] = frame.loc[valid_mask, "UserID"].isin(set(train_rows["UserID"])).to_numpy()
        scores[key] = float(roc_auc_score(labels[origin], pred))
        best_iterations.append(int(np.median(fold_iterations)))
    return CrossValidationResult("xgboost_ranker", predictions, labels, users, indices, seen, scores, best_iterations)


def cat_parameters(seed: int, iterations: int) -> dict:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "depth": 7,
        "learning_rate": 0.04,
        "l2_leaf_reg": 10.0,
        "iterations": iterations,
        "random_seed": seed,
        "thread_count": 11,
        "verbose": False,
        "allow_writing_files": False,
        "random_strength": 0.5,
    }


def cross_validate_catboost(frame: pd.DataFrame, columns: list[str], cap: int = 1500) -> CrossValidationResult:
    cats = categorical_columns(columns)
    matrix = catboost_matrix(frame, columns, cats)
    cat_indices = [columns.index(column) for column in cats]
    predictions = {}
    labels = {}
    users = {}
    indices = {}
    seen = {}
    scores = {}
    best_iterations = []
    for origin in FOLD_ORIGINS:
        train_mask, valid_mask = fold_masks(frame, origin)
        train_rows = frame.loc[train_mask]
        fold_predictions = []
        fold_iterations = []
        for seed in SEEDS:
            model = CatBoostClassifier(**cat_parameters(seed, cap), od_type="Iter", od_wait=100)
            model.fit(
                matrix.loc[train_mask], train_rows["label"].to_numpy(dtype=np.int8),
                cat_features=cat_indices, sample_weight=balanced_origin_weights(train_rows),
                eval_set=(matrix.loc[valid_mask], frame.loc[valid_mask, "label"].to_numpy(dtype=np.int8)),
                verbose=False,
            )
            fold_predictions.append(model.predict_proba(matrix.loc[valid_mask])[:, 1])
            best = model.get_best_iteration()
            fold_iterations.append(int(best + 1 if best >= 0 else cap))
            del model
            gc.collect()
        pred = np.mean(fold_predictions, axis=0)
        key = origin.strftime("%Y-%m-%d")
        predictions[origin] = pred
        labels[origin] = frame.loc[valid_mask, "label"].to_numpy(dtype=np.int8)
        users[origin] = frame.loc[valid_mask, "UserID"].to_numpy()
        indices[origin] = np.flatnonzero(valid_mask)
        seen[origin] = frame.loc[valid_mask, "UserID"].isin(set(train_rows["UserID"])).to_numpy()
        scores[key] = float(roc_auc_score(labels[origin], pred))
        best_iterations.append(int(np.median(fold_iterations)))
    return CrossValidationResult("catboost", predictions, labels, users, indices, seen, scores, best_iterations)


def fit_sigmoid(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    model = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000)
    model.fit(scores.reshape(-1, 1), labels)
    slope = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    if slope <= 0:
        scale = max(float(np.std(scores)), 1e-6)
        slope = 1.0 / scale
        intercept = -float(np.mean(scores)) / scale
    return slope, intercept


def rank_average(parts: list[np.ndarray]) -> np.ndarray:
    return np.mean([rankdata(part, method="average") / (len(part) + 1.0) for part in parts], axis=0)


def ensemble_result(name: str, foundation: CrossValidationResult, ranker: CrossValidationResult, cat: CrossValidationResult, kind: str) -> CrossValidationResult:
    all_rank_scores = np.concatenate([ranker.predictions[origin] for origin in FOLD_ORIGINS])
    all_labels = np.concatenate([ranker.labels[origin] for origin in FOLD_ORIGINS])
    slope, intercept = fit_sigmoid(all_rank_scores, all_labels)
    predictions = {}
    scores = {}
    for origin in FOLD_ORIGINS:
        lgb_pred = foundation.predictions[origin]
        rank_pred = expit(slope * ranker.predictions[origin] + intercept)
        cat_pred = cat.predictions[origin]
        if kind == "equal_rank":
            pred = rank_average([lgb_pred, rank_pred, cat_pred])
        else:
            pred = 0.60 * lgb_pred + 0.25 * rank_pred + 0.15 * cat_pred
        predictions[origin] = pred
        scores[origin.strftime("%Y-%m-%d")] = float(roc_auc_score(foundation.labels[origin], pred))
    return CrossValidationResult(
        name, predictions, foundation.labels, foundation.users, foundation.indices,
        foundation.seen, scores, foundation.best_iterations,
    )


def choose_feature_set(frame: pd.DataFrame) -> tuple[list[str], CrossValidationResult, dict[str, object]]:
    core_columns = [column for column in core_feature_columns(frame) if "_origin_" not in column]
    full_columns = [column for column in all_feature_columns(frame) if "_origin_" not in column]
    print(f"[design] feature sets core={len(core_columns)} full={len(full_columns)}", flush=True)
    core_result = cross_validate_lgb(frame, core_columns)
    core_summary = summarize_cv("stage1_core_lightgbm", frame, core_result)
    full_result = cross_validate_lgb(frame, full_columns)
    full_summary = summarize_cv("stage2_alltable_lightgbm", frame, full_result)
    labels, full_predictions, users = full_result.concatenated()
    _, core_predictions, _ = core_result.concatenated()
    gain_se = bootstrap_gain_se(labels, full_predictions, core_predictions, users)
    gain = float(full_summary["mean_roc_auc"] - core_summary["mean_roc_auc"])
    stable = gain >= gain_se and float(full_summary["worst_origin_roc_auc"]) >= float(core_summary["worst_origin_roc_auc"]) - gain_se
    selected_columns = full_columns if stable else core_columns
    selected_result = full_result if stable else core_result
    diagnostics = {
        "core": core_summary,
        "full": full_summary,
        "full_minus_core_mean": gain,
        "paired_bootstrap_se": gain_se,
        "selected": "full" if stable else "core",
    }
    print("[design] " + json.dumps({"feature_gate": diagnostics["selected"], "gain": gain, "gain_se": gain_se}, separators=(",", ":")), flush=True)
    return selected_columns, selected_result, diagnostics


def choose_model(frame: pd.DataFrame, columns: list[str], foundation: CrossValidationResult) -> tuple[str, dict, dict[str, object]]:
    ranker = cross_validate_xgb(frame, columns)
    ranker_summary = summarize_cv("origin_grouped_xgboost_ranker", frame, ranker)
    cat = cross_validate_catboost(frame, columns)
    cat_summary = summarize_cv("catboost_categorical", frame, cat)
    equal = ensemble_result("equal_rank_average", foundation, ranker, cat, "equal_rank")
    fixed = ensemble_result("fixed_060_025_015", foundation, ranker, cat, "fixed")
    foundation_summary = summarize_cv("final_lightgbm_alternative", frame, foundation)
    equal_summary = summarize_cv("equal_rank_average", frame, equal)
    fixed_summary = summarize_cv("fixed_060_025_015", frame, fixed)
    alternatives = {"lightgbm": foundation, "equal_rank": equal, "fixed": fixed}
    summaries = {"lightgbm": foundation_summary, "equal_rank": equal_summary, "fixed": fixed_summary}
    best_name = max(alternatives, key=lambda name: (summaries[name]["mean_roc_auc"], summaries[name]["worst_origin_roc_auc"]))
    if best_name != "lightgbm":
        labels, candidate_predictions, users = alternatives[best_name].concatenated()
        _, baseline_predictions, _ = foundation.concatenated()
        gain_se = bootstrap_gain_se(labels, candidate_predictions, baseline_predictions, users)
        gain = float(summaries[best_name]["mean_roc_auc"] - foundation_summary["mean_roc_auc"])
        stable = gain >= gain_se and float(summaries[best_name]["worst_origin_roc_auc"]) >= float(foundation_summary["worst_origin_roc_auc"]) - gain_se
        if not stable:
            best_name = "lightgbm"
    rank_scores = np.concatenate([ranker.predictions[origin] for origin in FOLD_ORIGINS])
    labels = np.concatenate([ranker.labels[origin] for origin in FOLD_ORIGINS])
    calibration = fit_sigmoid(rank_scores, labels)
    iterations = {
        "lightgbm": int(np.clip(foundation.best_iterations[-1], 1, 1800)),
        "ranker": int(np.clip(np.median(ranker.best_iterations), 1, 1800)),
        "catboost": int(np.clip(np.median(cat.best_iterations), 1, 1500)),
        "calibration": calibration,
    }
    diagnostics = {
        "ranker": ranker_summary,
        "catboost": cat_summary,
        "alternatives": summaries,
        "selected": best_name,
        "iterations": iterations,
    }
    print("[design] " + json.dumps({"model_gate": best_name, "iterations": iterations}, separators=(",", ":")), flush=True)
    return best_name, iterations, diagnostics


def fit_lgb_predictions(frame: pd.DataFrame, columns: list[str], train_mask: np.ndarray, predict_mask: np.ndarray, iterations: int) -> np.ndarray:
    matrix = numeric_matrix(frame, columns)
    train_rows = frame.loc[train_mask]
    predictions = []
    for seed in SEEDS:
        train_set = lgb.Dataset(matrix[train_mask], label=train_rows["label"].to_numpy(), weight=balanced_origin_weights(train_rows), free_raw_data=True)
        model = lgb.train(lgb_parameters(seed), train_set, num_boost_round=iterations, callbacks=[lgb.log_evaluation(0)])
        predictions.append(model.predict(matrix[predict_mask]))
        del model, train_set
        gc.collect()
    return np.mean(predictions, axis=0)


def fit_ranker_predictions(frame: pd.DataFrame, columns: list[str], train_mask: np.ndarray, predict_mask: np.ndarray, iterations: int, calibration: tuple[float, float]) -> np.ndarray:
    matrix = numeric_matrix(frame, columns)
    train_rows = frame.loc[train_mask]
    qid = pd.factorize(train_rows["origin"], sort=True)[0].astype(np.int32)
    predictions = []
    for seed in SEEDS:
        model = xgb.XGBRanker(**xgb_parameters(seed, iterations, False))
        model.fit(matrix[train_mask], train_rows["label"].to_numpy(dtype=np.int8), qid=qid, verbose=False)
        predictions.append(model.predict(matrix[predict_mask]))
        del model
        gc.collect()
    raw = np.mean(predictions, axis=0)
    return expit(calibration[0] * raw + calibration[1])


def fit_cat_predictions(frame: pd.DataFrame, columns: list[str], train_mask: np.ndarray, predict_mask: np.ndarray, iterations: int) -> np.ndarray:
    cats = categorical_columns(columns)
    matrix = catboost_matrix(frame, columns, cats)
    cat_indices = [columns.index(column) for column in cats]
    train_rows = frame.loc[train_mask]
    predictions = []
    for seed in SEEDS:
        model = CatBoostClassifier(**cat_parameters(seed, iterations))
        model.fit(
            matrix.loc[train_mask], train_rows["label"].to_numpy(dtype=np.int8),
            cat_features=cat_indices, sample_weight=balanced_origin_weights(train_rows), verbose=False,
        )
        predictions.append(model.predict_proba(matrix.loc[predict_mask])[:, 1])
        del model
        gc.collect()
    return np.mean(predictions, axis=0)


def fit_chain(frame: pd.DataFrame, columns: list[str], model_choice: str, iterations: dict, train_end: pd.Timestamp, predict_origin: pd.Timestamp) -> np.ndarray:
    train_mask = (frame["label"].notna() & (frame["origin"] <= train_end)).to_numpy()
    predict_mask = (frame["origin"] == predict_origin).to_numpy()
    lgb_prediction = fit_lgb_predictions(frame, columns, train_mask, predict_mask, iterations["lightgbm"])
    if model_choice == "lightgbm":
        return lgb_prediction
    rank_prediction = fit_ranker_predictions(frame, columns, train_mask, predict_mask, iterations["ranker"], iterations["calibration"])
    cat_prediction = fit_cat_predictions(frame, columns, train_mask, predict_mask, iterations["catboost"])
    if model_choice == "equal_rank":
        return rank_average([lgb_prediction, rank_prediction, cat_prediction])
    return 0.60 * lgb_prediction + 0.25 * rank_prediction + 0.15 * cat_prediction


def debug_predictions(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    columns = core_feature_columns(frame)
    train_rows = frame[frame["label"].notna()].groupby("origin", group_keys=False).sample(n=10000, random_state=17)
    train_mask = frame.index.isin(train_rows.index)
    validation_mask = (frame["source"] == "validation").to_numpy()
    test_mask = (frame["source"] == "test").to_numpy()
    matrix = numeric_matrix(frame, columns)
    dataset = lgb.Dataset(matrix[train_mask], label=frame.loc[train_mask, "label"].to_numpy(), weight=balanced_origin_weights(frame.loc[train_mask]))
    model = lgb.train(lgb_parameters(17), dataset, num_boost_round=50, callbacks=[lgb.log_evaluation(0)])
    validation = model.predict(matrix[validation_mask])
    test = model.predict(matrix[test_mask])
    diagnostics = {"debug": True, "features": len(columns), "training_rows": int(train_mask.sum()), "trees": 50}
    return validation, test, diagnostics
