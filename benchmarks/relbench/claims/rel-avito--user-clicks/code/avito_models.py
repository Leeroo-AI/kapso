import json
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


warnings.filterwarnings("ignore")


EPSILON = 1e-6


class ConstantModel:
    def __init__(self, value):
        self.value = float(value)

    def predict_proba(self, matrix):
        probability = np.full(len(matrix), self.value, dtype=np.float64)
        return np.column_stack((1.0 - probability, probability))

    def predict(self, matrix):
        return np.full(len(matrix), self.value, dtype=np.float64)


def _clip_probability(values):
    return np.clip(np.asarray(values, dtype=np.float64), EPSILON, 1 - EPSILON)


def _poisson_tail(mean):
    mean = np.maximum(np.asarray(mean, dtype=np.float64), 0.0)
    return _clip_probability(1.0 - np.exp(-mean) * (1.0 + mean))


def _negative_binomial_tail(mean, alpha):
    mean = np.maximum(np.asarray(mean, dtype=np.float64), 0.0)
    alpha = max(float(alpha), 1e-8)
    p_zero = np.power(1.0 + alpha * mean, -1.0 / alpha)
    p_one = mean * p_zero / (1.0 + alpha * mean)
    return _clip_probability(1.0 - p_zero - p_one)


def _lgb_classifier(rounds, seed):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=rounds,
        num_leaves=63,
        min_child_samples=250,
        learning_rate=0.03,
        reg_lambda=15.0,
        max_bin=127,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=seed,
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")),
        verbosity=-1,
        force_col_wise=True,
    )


def _fit_classifier(matrix, target, rounds, seed, categorical_index):
    target = np.asarray(target, dtype=np.int8)
    if np.unique(target).size < 2:
        return ConstantModel(np.mean(target))
    model = _lgb_classifier(rounds, seed)
    model.fit(
        matrix,
        target,
        categorical_feature=categorical_index.tolist(),
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


def _lgb_poisson(rounds, leaves, seed):
    return lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=rounds,
        num_leaves=leaves,
        min_child_samples=250,
        learning_rate=0.03,
        reg_lambda=15.0,
        max_bin=127,
        random_state=seed,
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")),
        verbosity=-1,
        force_col_wise=True,
    )


def _lgb_propensity(rounds, seed):
    return lgb.LGBMRegressor(
        objective="cross_entropy",
        n_estimators=rounds,
        num_leaves=31,
        min_child_samples=250,
        learning_rate=0.03,
        reg_lambda=15.0,
        max_bin=127,
        random_state=seed,
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")),
        verbosity=-1,
        force_col_wise=True,
    )


def _fit_xgb_poisson(matrix, target, rounds, seed):
    parameters = dict(
        objective="count:poisson",
        n_estimators=rounds,
        max_depth=5,
        min_child_weight=30,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=15.0,
        tree_method="hist",
        device="cuda",
        random_state=seed,
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")),
        verbosity=0,
    )
    model = xgb.XGBRegressor(**parameters)
    try:
        model.fit(matrix, target, verbose=False)
    except xgb.core.XGBoostError as error:
        print(
            f"[models] xgboost CUDA unavailable, retrying CPU: "
            f"{str(error).splitlines()[0]}",
            flush=True,
        )
        parameters["device"] = "cpu"
        model = xgb.XGBRegressor(**parameters)
        model.fit(matrix, target, verbose=False)
    return model


def _dispersion(target, fitted_mean):
    target = np.asarray(target, dtype=np.float64)
    fitted_mean = np.maximum(np.asarray(fitted_mean, dtype=np.float64), 1e-5)
    numerator = np.sum((target - fitted_mean) ** 2 - fitted_mean)
    denominator = np.sum(fitted_mean**2)
    return float(np.clip(numerator / max(denominator, 1e-8), 0.0, 20.0))


def propensity_columns(feature_names):
    tokens = [
        "ctr",
        "click",
        "position",
        "context",
        "query",
        "login",
        "device",
        "category",
        "family",
        "os_",
        "exposure_to_click",
        "repeat_ad",
        "recency",
    ]
    selected = [
        index
        for index, name in enumerate(feature_names)
        if any(token in name for token in tokens)
        and "sketch" not in name
    ]
    return np.asarray(selected, dtype=np.int64)


def categorical_columns(feature_names):
    return np.asarray([], dtype=np.int64)


def encode_categorical_matrix(matrix, categorical_index):
    for column in categorical_index:
        values = matrix[:, column]
        finite = np.isfinite(values) & (values >= 0)
        categories = np.unique(values[finite])
        encoded = np.full(len(values), np.nan, dtype=np.float32)
        encoded[finite] = np.searchsorted(
            categories, values[finite]
        ).astype(np.float32)
        matrix[:, column] = encoded
    return matrix


def fit_predict_heads(
    train_matrix,
    targets,
    predict_matrix,
    propensity_index,
    categorical_index,
    rounds,
    count_caps,
    seed,
):
    start = time.time()
    y_repeat = np.asarray(targets["repeat"], dtype=np.int8)
    y_any = np.asarray(targets["any"], dtype=np.int8)
    y_count = np.asarray(targets["count"], dtype=np.float32)
    y_exposure = np.asarray(targets["exposure"], dtype=np.float32)
    direct_model = _fit_classifier(
        train_matrix,
        y_repeat,
        rounds["binary"],
        seed + 1,
        categorical_index,
    )
    any_model = _fit_classifier(
        train_matrix,
        y_any,
        rounds["binary"],
        seed + 2,
        categorical_index,
    )
    conditional_rows = y_any == 1
    conditional_model = _fit_classifier(
        train_matrix[conditional_rows],
        y_repeat[conditional_rows],
        rounds["binary"],
        seed + 3,
        categorical_index,
    )
    direct = _clip_probability(direct_model.predict_proba(predict_matrix)[:, 1])
    any_probability = _clip_probability(
        any_model.predict_proba(predict_matrix)[:, 1]
    )
    conditional_probability = _clip_probability(
        conditional_model.predict_proba(predict_matrix)[:, 1]
    )
    predictions = {
        "direct": direct,
        "hurdle": _clip_probability(
            any_probability * conditional_probability
        ),
    }
    dispersions = {}
    for offset, cap in enumerate(count_caps):
        capped_target = np.minimum(y_count, float(cap))
        fitted_members = []
        predicted_members = []
        for member_offset in [0, 1009, 2017]:
            count_model = _fit_xgb_poisson(
                train_matrix,
                capped_target,
                rounds["count"],
                seed + 20 + offset + member_offset,
            )
            fitted_members.append(
                np.maximum(count_model.predict(train_matrix), 0.0)
            )
            predicted_members.append(
                np.maximum(count_model.predict(predict_matrix), 0.0)
            )
        fitted_mean = np.mean(fitted_members, axis=0)
        dispersion = _dispersion(y_count, fitted_mean)
        mean_upper = max(
            2.0,
            float(np.quantile(capped_target, 0.999)) * 1.5,
        )
        predicted_mean = np.clip(
            np.mean(predicted_members, axis=0), 0.0, mean_upper
        )
        key = f"count_{cap}"
        predictions[key] = _poisson_tail(predicted_mean)
        predictions[f"negative_binomial_{cap}"] = _negative_binomial_tail(
            predicted_mean, dispersion
        )
        dispersions[str(cap)] = dispersion
    exposure_model = _lgb_poisson(
        rounds["exposure"], 31, seed + 40
    )
    exposure_model.fit(
        train_matrix,
        y_exposure,
        categorical_feature=categorical_index.tolist(),
        callbacks=[lgb.log_evaluation(0)],
    )
    exposure_upper = max(
        5.0, float(np.quantile(y_exposure, 0.999)) * 1.25
    )
    predicted_exposure = np.clip(
        exposure_model.predict(predict_matrix), 0.0, exposure_upper
    )
    propensity_target = y_count / np.maximum(y_exposure, 1.0)
    propensity_weight = np.clip(y_exposure, 1.0, 100.0)
    q_train = train_matrix[:, propensity_index]
    q_predict = predict_matrix[:, propensity_index]
    categorical_set = set(categorical_index.tolist())
    q_categorical = [
        position
        for position, source in enumerate(propensity_index.tolist())
        if source in categorical_set
    ]
    propensity_model = _lgb_propensity(rounds["propensity"], seed + 41)
    propensity_model.fit(
        q_train,
        propensity_target,
        sample_weight=propensity_weight,
        categorical_feature=q_categorical,
        callbacks=[lgb.log_evaluation(0)],
    )
    positive_propensity = propensity_target[propensity_target > 0]
    if len(positive_propensity):
        propensity_upper = max(
            0.01, float(np.quantile(positive_propensity, 0.995))
        )
    else:
        propensity_upper = 0.01
    predicted_propensity = np.clip(
        propensity_model.predict(q_predict),
        EPSILON,
        propensity_upper,
    )
    intensity = predicted_exposure * predicted_propensity
    predictions["mechanistic"] = _poisson_tail(intensity)
    predictions["predicted_exposure"] = predicted_exposure
    predictions["predicted_propensity"] = predicted_propensity
    print(
        f"[models] heads fitted train={len(train_matrix)} "
        f"predict={len(predict_matrix)} dispersions={json.dumps(dispersions)} "
        f"elapsed={round(time.time() - start, 2)}s",
        flush=True,
    )
    return predictions, dispersions


def _safe_auc(target, prediction):
    if np.unique(target).size < 2:
        return float("nan")
    return float(roc_auc_score(target, prediction))


def summarize_fold(timestamp, target, predictions, dispersions):
    scores = {
        name: _safe_auc(target, prediction)
        for name, prediction in predictions.items()
        if name
        not in ["predicted_exposure", "predicted_propensity"]
    }
    record = {
        "anchor": str(timestamp.date()),
        "rows": int(len(target)),
        "positive_rate": float(np.mean(target)),
        "roc_auc": scores,
        "dispersion": dispersions,
    }
    print(f"[oof] fold={json.dumps(record, allow_nan=False)}", flush=True)
    return record


def select_count_cap(folds, caps):
    records = []
    for cap in caps:
        scores = [
            _safe_auc(fold["target"], fold["predictions"][f"count_{cap}"])
            for fold in folds
        ]
        records.append(
            {
                "cap": int(cap),
                "mean": float(np.mean(scores)),
                "worst": float(np.min(scores)),
                "scores": scores,
            }
        )
    records.sort(
        key=lambda item: (
            item["mean"] + 0.25 * item["worst"],
            item["worst"],
            -item["cap"],
        ),
        reverse=True,
    )
    best = records[0]
    for record in sorted(records, key=lambda item: item["cap"]):
        if (
            best["mean"] - record["mean"] <= 0.001
            and best["worst"] - record["worst"] <= 0.0015
        ):
            best = record
            break
    print(
        f"[selection] count_cap={best['cap']} "
        f"candidates={json.dumps(records)}",
        flush=True,
    )
    return best["cap"], records


def _meta_matrix(fold, heads):
    columns = []
    for head in heads:
        prediction = _clip_probability(fold["predictions"][head])
        columns.append(np.log(prediction / (1.0 - prediction)))
    columns.append(np.asarray(fold["cold"], dtype=np.float64))
    return np.column_stack(columns)


def _fit_meta(matrix, target, regularization):
    if np.unique(target).size < 2:
        return ConstantModel(np.mean(target))
    model = LogisticRegression(
        C=regularization,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=1337,
    )
    model.fit(matrix, target)
    return model


def _sequential_meta_scores(folds, heads, regularization):
    scores = []
    deltas = []
    for position in range(1, len(folds)):
        train_matrix = np.concatenate(
            [_meta_matrix(fold, heads) for fold in folds[:position]],
            axis=0,
        )
        train_target = np.concatenate(
            [fold["target"] for fold in folds[:position]]
        )
        model = _fit_meta(train_matrix, train_target, regularization)
        validation_matrix = _meta_matrix(folds[position], heads)
        prediction = model.predict_proba(validation_matrix)[:, 1]
        score = _safe_auc(folds[position]["target"], prediction)
        scores.append(score)
        if heads[-1].startswith("negative_binomial"):
            reduced = heads[:-1]
            reduced_train = np.concatenate(
                [_meta_matrix(fold, reduced) for fold in folds[:position]],
                axis=0,
            )
            reduced_model = _fit_meta(
                reduced_train, train_target, regularization
            )
            reduced_prediction = reduced_model.predict_proba(
                _meta_matrix(folds[position], reduced)
            )[:, 1]
            deltas.append(
                score
                - _safe_auc(
                    folds[position]["target"], reduced_prediction
                )
            )
    return scores, deltas


def select_meta(folds, count_cap):
    count_head = f"count_{count_cap}"
    nb_head = f"negative_binomial_{count_cap}"
    candidates = [
        ["direct"],
        ["hurdle"],
        [count_head],
        ["hurdle", count_head],
        ["direct", "hurdle"],
        ["direct", count_head],
        ["direct", "hurdle", count_head],
        ["direct", "hurdle", count_head, "mechanistic"],
    ]
    dispersion_values = [
        float(fold["dispersions"][str(count_cap)]) for fold in folds
    ]
    overdispersed = float(np.mean(dispersion_values)) > 0.5
    if overdispersed:
        candidates.append(
            ["direct", "hurdle", count_head, "mechanistic", nb_head]
        )
    records = []
    for heads in candidates:
        for regularization in [0.05, 0.2, 1.0]:
            scores, deltas = _sequential_meta_scores(
                folds, heads, regularization
            )
            if not scores:
                scores = [
                    _safe_auc(
                        folds[0]["target"],
                        folds[0]["predictions"][heads[0]],
                    )
                ]
            nb_consistent = True
            if heads[-1].startswith("negative_binomial"):
                nb_consistent = bool(
                    deltas
                    and min(deltas) > 0
                    and float(np.mean(deltas)) > 0.0002
                )
            records.append(
                {
                    "heads": heads,
                    "C": regularization,
                    "mean": float(np.mean(scores)),
                    "worst": float(np.min(scores)),
                    "scores": scores,
                    "nb_consistent": nb_consistent,
                }
            )
    eligible = [record for record in records if record["nb_consistent"]]
    eligible.sort(
        key=lambda item: (
            item["mean"] + 0.25 * item["worst"],
            item["worst"],
            -len(item["heads"]),
            -item["C"],
        ),
        reverse=True,
    )
    best = eligible[0]
    for record in sorted(
        eligible, key=lambda item: (len(item["heads"]), item["C"])
    ):
        if (
            best["mean"] - record["mean"] <= 0.001
            and best["worst"] - record["worst"] <= 0.0015
        ):
            best = record
            break
    summary = {
        "heads": best["heads"],
        "C": best["C"],
        "mean": best["mean"],
        "worst": best["worst"],
        "overdispersed": overdispersed,
        "dispersion_mean": float(np.mean(dispersion_values)),
    }
    print(
        f"[selection] meta={json.dumps(summary)}",
        flush=True,
    )
    return summary, records


def fit_meta_from_oof(folds, selection):
    matrix = np.concatenate(
        [_meta_matrix(fold, selection["heads"]) for fold in folds],
        axis=0,
    )
    target = np.concatenate([fold["target"] for fold in folds])
    return _fit_meta(matrix, target, selection["C"])


def predict_meta(model, predictions, cold, heads):
    fold = {"predictions": predictions, "cold": cold}
    matrix = _meta_matrix(fold, heads)
    return _clip_probability(model.predict_proba(matrix)[:, 1])
