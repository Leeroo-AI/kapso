import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


ORIGINS = pd.to_datetime(["2019-10-01", "2020-01-01", "2020-04-01", "2020-07-01"])
SEEDS = (2027, 9137)


def _accuracy_metric(predictions, dataset):
    labels = dataset.get_label().astype(np.int64)
    probabilities = predictions if predictions.ndim == 2 else predictions.reshape(len(labels), 3)
    return "accuracy", float(np.mean(np.argmax(probabilities, axis=1) == labels)), True


def lightgbm_parameters(seed):
    return {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "None",
        "learning_rate": 0.04,
        "num_leaves": 127,
        "min_data_in_leaf": 150,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 6.0,
        "seed": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "verbosity": -1,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "11")),
        "force_col_wise": True,
    }


def bootstrap_standard_error(correct, users, seed=1337, repeats=120):
    data = pd.DataFrame({"user": users, "correct": np.asarray(correct, dtype=np.float64)})
    grouped = data.groupby("user", sort=False)["correct"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy()
    counts = grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    scores = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        sample = rng.integers(0, len(grouped), size=len(grouped))
        scores[i] = sums[sample].sum() / counts[sample].sum()
    return float(scores.std(ddof=1))


def stratum_metrics(labels, probabilities, batch_sizes, prior_badges):
    prediction = np.argmax(probabilities, axis=1)
    records = {}
    batch_bins = pd.cut(batch_sizes, [-1, 1, 2, 5, 20, np.inf], labels=["1", "2", "3-5", "6-20", "21+"])
    maturity_bins = pd.cut(prior_badges, [-1, 0, 2, 10, 50, np.inf], labels=["0", "1-2", "3-10", "11-50", "51+"])
    for axis, groups in (("batch", batch_bins), ("maturity", maturity_bins)):
        for value in groups.categories:
            mask = np.asarray(groups == value)
            if mask.sum():
                records[f"{axis}:{value}"] = {
                    "count": int(mask.sum()),
                    "accuracy": float(np.mean(prediction[mask] == labels[mask])),
                }
    return records


def forward_lightgbm(features, labels, dates, users, batch_sizes, prior_badges, feature_indices, seed=2027, debug=False):
    folds = []
    predictions = {}
    best_iterations = []
    maximum_rounds = 50 if debug else 1800
    for fold_index, origin in enumerate(ORIGINS):
        train_mask = dates < origin
        validation_mask = (dates >= origin) & (dates < origin + pd.Timedelta(days=90))
        if train_mask.sum() == 0 or validation_mask.sum() == 0:
            continue
        train_data = lgb.Dataset(
            features[train_mask][:, feature_indices],
            label=labels[train_mask],
            free_raw_data=False,
        )
        validation_data = lgb.Dataset(
            features[validation_mask][:, feature_indices],
            label=labels[validation_mask],
            reference=train_data,
            free_raw_data=False,
        )
        model = lgb.train(
            lightgbm_parameters(seed),
            train_data,
            num_boost_round=maximum_rounds,
            valid_sets=[validation_data],
            feval=_accuracy_metric,
            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
        )
        probability = model.predict(features[validation_mask][:, feature_indices], num_iteration=model.best_iteration)
        correct = np.argmax(probability, axis=1) == labels[validation_mask]
        score = float(correct.mean())
        standard_error = bootstrap_standard_error(correct, users[validation_mask], seed=1337 + fold_index)
        folds.append({
            "origin": str(origin.date()),
            "count": int(validation_mask.sum()),
            "accuracy": score,
            "bootstrap_se": standard_error,
            "best_iteration": int(model.best_iteration),
            "strata": stratum_metrics(
                labels[validation_mask], probability, batch_sizes[validation_mask], prior_badges[validation_mask]
            ),
        })
        predictions[str(origin.date())] = (np.flatnonzero(validation_mask), probability)
        best_iterations.append(int(model.best_iteration))
        print(f"[internal] lightgbm origin={origin.date()} rows={validation_mask.sum()} accuracy={score:.6f} se={standard_error:.6f} iteration={model.best_iteration}")
        del model, train_data, validation_data
        gc.collect()
    accuracies = np.asarray([fold["accuracy"] for fold in folds])
    objective = float(accuracies.mean() - 0.5 * accuracies.std(ddof=0))
    return {
        "folds": folds,
        "objective": objective,
        "mean_accuracy": float(accuracies.mean()),
        "fold_std": float(accuracies.std(ddof=0)),
        "median_iteration": int(min(1800, np.median(best_iterations))),
        "predictions": predictions,
    }


def forward_catboost(features, labels, dates, users, batch_sizes, prior_badges, feature_indices, iterations):
    from catboost import CatBoostClassifier

    folds = []
    predictions = {}
    best_iterations = []
    for fold_index, origin in enumerate(ORIGINS):
        train_mask = dates < origin
        validation_mask = (dates >= origin) & (dates < origin + pd.Timedelta(days=90))
        model = CatBoostClassifier(
            iterations=min(1800, max(300, int(iterations * 1.35))),
            depth=8,
            learning_rate=0.04,
            l2_leaf_reg=10,
            loss_function="MultiClass",
            eval_metric="Accuracy",
            task_type="GPU",
            devices="0",
            random_seed=2027,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            features[train_mask][:, feature_indices],
            labels[train_mask],
            eval_set=(features[validation_mask][:, feature_indices], labels[validation_mask]),
            use_best_model=True,
            early_stopping_rounds=120,
            verbose=False,
        )
        probability = model.predict_proba(features[validation_mask][:, feature_indices])
        correct = np.argmax(probability, axis=1) == labels[validation_mask]
        score = float(correct.mean())
        standard_error = bootstrap_standard_error(correct, users[validation_mask], seed=9137 + fold_index)
        best_iteration = int(model.get_best_iteration() + 1)
        folds.append({
            "origin": str(origin.date()),
            "count": int(validation_mask.sum()),
            "accuracy": score,
            "bootstrap_se": standard_error,
            "best_iteration": best_iteration,
            "strata": stratum_metrics(
                labels[validation_mask], probability, batch_sizes[validation_mask], prior_badges[validation_mask]
            ),
        })
        predictions[str(origin.date())] = (np.flatnonzero(validation_mask), probability)
        best_iterations.append(best_iteration)
        print(f"[internal] catboost origin={origin.date()} rows={validation_mask.sum()} accuracy={score:.6f} se={standard_error:.6f} iteration={best_iteration}")
        del model
        gc.collect()
    accuracies = np.asarray([fold["accuracy"] for fold in folds])
    return {
        "folds": folds,
        "objective": float(accuracies.mean() - 0.5 * accuracies.std(ddof=0)),
        "mean_accuracy": float(accuracies.mean()),
        "fold_std": float(accuracies.std(ddof=0)),
        "median_iteration": int(min(1800, np.median(best_iterations))),
        "predictions": predictions,
    }


def compare_feature_designs(features, labels, dates, users, feature_names, compact_count):
    compact_indices = np.arange(compact_count, dtype=np.int64)
    full_indices = np.arange(features.shape[1], dtype=np.int64)
    compact = forward_lightgbm(
        features, labels, dates, users,
        features[:, feature_names.index("batch_batch_size")],
        features[:, feature_names.index("prior_badge_count")],
        compact_indices,
    )
    full = forward_lightgbm(
        features, labels, dates, users,
        features[:, feature_names.index("batch_batch_size")],
        features[:, feature_names.index("prior_badge_count")],
        full_indices,
    )
    tie = float(np.mean([fold["bootstrap_se"] for fold in full["folds"]]))
    if full["objective"] > compact["objective"] + tie:
        selected = "full"
        selected_indices = full_indices
        selected_result = full
    else:
        selected = "compact"
        selected_indices = compact_indices
        selected_result = compact
    print(f"[internal] design compact_objective={compact['objective']:.6f} full_objective={full['objective']:.6f} tie_se={tie:.6f} selected={selected}")
    return selected, selected_indices, selected_result, {"compact": compact, "full": full}


def choose_blend(lightgbm_result, catboost_result, labels, users):
    candidates = []
    for light_weight in (1.0, 0.75, 0.5):
        accuracies = []
        standard_errors = []
        for light_fold, cat_fold in zip(lightgbm_result["folds"], catboost_result["folds"]):
            key = light_fold["origin"]
            indices, light_probability = lightgbm_result["predictions"][key]
            cat_indices, cat_probability = catboost_result["predictions"][key]
            if not np.array_equal(indices, cat_indices):
                raise RuntimeError("internal prediction alignment mismatch")
            label = labels[indices]
            fold_users = users[indices]
            probability = light_weight * light_probability + (1 - light_weight) * cat_probability
            correct = np.argmax(probability, axis=1) == label
            accuracies.append(float(correct.mean()))
            standard_errors.append(bootstrap_standard_error(correct, fold_users))
        accuracies = np.asarray(accuracies)
        candidates.append({
            "lightgbm_weight": light_weight,
            "mean_accuracy": float(accuracies.mean()),
            "fold_std": float(accuracies.std(ddof=0)),
            "objective": float(accuracies.mean() - 0.5 * accuracies.std(ddof=0)),
            "tie_se": float(np.mean(standard_errors)),
        })
    baseline = candidates[0]
    winner = baseline
    for candidate in candidates[1:]:
        if candidate["objective"] > winner["objective"] + candidate["tie_se"]:
            winner = candidate
    print(f"[internal] blends={json.dumps(candidates)} selected_lightgbm_weight={winner['lightgbm_weight']}")
    return winner, candidates


def train_lightgbm_ensemble(train_features, labels, predict_features, feature_indices, iterations, debug=False):
    seeds = (2027,) if debug else SEEDS
    probability = np.zeros((len(predict_features), 3), dtype=np.float64)
    models = []
    rounds = 50 if debug else min(1800, max(100, int(iterations)))
    for seed in seeds:
        dataset = lgb.Dataset(train_features[:, feature_indices], label=labels, free_raw_data=True)
        model = lgb.train(
            lightgbm_parameters(seed),
            dataset,
            num_boost_round=rounds,
            callbacks=[lgb.log_evaluation(0)],
        )
        probability += model.predict(predict_features[:, feature_indices]) / len(seeds)
        models.append(model)
    return probability, models


def train_catboost(train_features, labels, predict_features, feature_indices, iterations):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=min(1800, max(100, int(iterations))),
        depth=8,
        learning_rate=0.04,
        l2_leaf_reg=10,
        loss_function="MultiClass",
        task_type="GPU",
        devices="0",
        random_seed=2027,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_features[:, feature_indices], labels, verbose=False)
    return model.predict_proba(predict_features[:, feature_indices]), model
