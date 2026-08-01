from __future__ import annotations

import json
import os
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SelectionResult:
    logistic_c: float
    selected_head: str
    feature_set: str
    tree_iterations: int
    metrics: dict[str, float]
    fold_metrics: dict[str, list[float]]
    oof_indices: np.ndarray
    oof_prediction: np.ndarray
    direct_oof_prediction: np.ndarray


def make_expanding_folds(timestamps: np.ndarray, labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    times = np.asarray(timestamps, dtype="datetime64[ns]")
    boundaries = [0.50, 0.625, 0.75, 0.875, 1.0]
    ordered_times = np.sort(times)
    cutoffs = [
        ordered_times[min(len(ordered_times) - 1, int(round(fraction * (len(ordered_times) - 1))))]
        for fraction in boundaries
    ]
    folds = []
    for fold_number, (lower, upper) in enumerate(zip(cutoffs[:-1], cutoffs[1:])):
        train_index = np.flatnonzero(times < lower)
        upper_mask = times <= upper if fold_number == len(cutoffs) - 2 else times < upper
        validation_index = np.flatnonzero((times >= lower) & upper_mask)
        if len(train_index) and len(validation_index) and np.unique(labels[validation_index]).size == 2:
            folds.append((train_index, validation_index))
    if len(folds) < 3:
        raise RuntimeError("expanding folds do not contain enough two-class validation blocks")
    summary = [
        {
            "train": int(len(train_index)),
            "train_positive": int(labels[train_index].sum()),
            "valid": int(len(validation_index)),
            "valid_positive": int(labels[validation_index].sum()),
        }
        for train_index, validation_index in folds
    ]
    print(f"[selection] expanding_folds={json.dumps(summary)}")
    return folds


def _logistic(c: float):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            C=c,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=17,
        ),
    )


def _tree(seed: int, estimators: int = 800):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=estimators,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        min_child_samples=60,
        reg_lambda=15.0,
        random_state=seed,
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")),
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def _rank_vector(values: np.ndarray) -> np.ndarray:
    return (rankdata(values, method="average") - 0.5) / len(values)


def _fold_auc(prediction: np.ndarray, labels: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], index_map: dict[int, int]) -> list[float]:
    scores = []
    for _, validation_index in folds:
        positions = np.asarray([index_map[int(index)] for index in validation_index], dtype=np.int32)
        scores.append(float(roc_auc_score(labels[validation_index], prediction[positions])))
    return scores


def select_heads(
    direct_matrix: np.ndarray,
    topology_matrix: np.ndarray,
    compact_matrix: np.ndarray,
    full_matrix: np.ndarray,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    fold_compact_validation: list[np.ndarray],
    fold_full_validation: list[np.ndarray],
) -> SelectionResult:
    oof_indices = np.concatenate([validation for _, validation in folds])
    index_map = {int(index): position for position, index in enumerate(oof_indices)}
    logistic_predictions: dict[float, np.ndarray] = {}
    logistic_fold_metrics: dict[float, list[float]] = {}
    for c in [0.05, 0.2, 1.0]:
        prediction = np.zeros(len(oof_indices), dtype=np.float64)
        scores = []
        for fold_number, (train_index, validation_index) in enumerate(folds):
            model = _logistic(c)
            model.fit(full_matrix[train_index], labels[train_index])
            current = model.predict_proba(fold_full_validation[fold_number])[:, 1]
            positions = np.asarray([index_map[int(index)] for index in validation_index], dtype=np.int32)
            prediction[positions] = current
            scores.append(float(roc_auc_score(labels[validation_index], current)))
        logistic_predictions[c] = prediction
        logistic_fold_metrics[c] = scores
    c_order = sorted(
        [0.05, 0.2, 1.0],
        key=lambda c: (
            np.mean(logistic_fold_metrics[c]) - 0.10 * np.std(logistic_fold_metrics[c]),
            -c,
        ),
        reverse=True,
    )
    selected_c = c_order[0]
    logistic_prediction = logistic_predictions[selected_c]
    tree_prediction = np.zeros(len(oof_indices), dtype=np.float64)
    tree_iterations = []
    for fold_number, (train_index, validation_index) in enumerate(folds):
        seed_predictions = []
        for seed in [17, 43]:
            model = _tree(seed)
            model.fit(
                full_matrix[train_index],
                labels[train_index],
                eval_set=[(fold_full_validation[fold_number], labels[validation_index])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
            )
            seed_predictions.append(model.predict_proba(fold_full_validation[fold_number])[:, 1])
            tree_iterations.append(int(model.best_iteration_ or 800))
        current = np.mean(seed_predictions, axis=0)
        positions = np.asarray([index_map[int(index)] for index in validation_index], dtype=np.int32)
        tree_prediction[positions] = current
    direct_logistic_prediction = np.zeros(len(oof_indices), dtype=np.float64)
    direct_tree_prediction = np.zeros(len(oof_indices), dtype=np.float64)
    topology_tree_prediction = np.zeros(len(oof_indices), dtype=np.float64)
    compact_tree_prediction = np.zeros(len(oof_indices), dtype=np.float64)
    fixed_iterations = max(50, int(np.median(tree_iterations)))
    family_iterations = {"direct": [], "topology": [], "compact_graph": []}
    for fold_number, (train_index, validation_index) in enumerate(folds):
        positions = np.asarray([index_map[int(index)] for index in validation_index], dtype=np.int32)
        logistic_model = _logistic(selected_c)
        logistic_model.fit(direct_matrix[train_index], labels[train_index])
        direct_logistic_prediction[positions] = logistic_model.predict_proba(direct_matrix[validation_index])[:, 1]
        tree_model = _tree(17)
        tree_model.fit(
            direct_matrix[train_index],
            labels[train_index],
            eval_set=[(direct_matrix[validation_index], labels[validation_index])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
        )
        direct_tree_prediction[positions] = tree_model.predict_proba(direct_matrix[validation_index])[:, 1]
        family_iterations["direct"].append(int(tree_model.best_iteration_ or 800))
        topology_model = _tree(17)
        topology_model.fit(
            topology_matrix[train_index],
            labels[train_index],
            eval_set=[(topology_matrix[validation_index], labels[validation_index])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
        )
        topology_tree_prediction[positions] = topology_model.predict_proba(topology_matrix[validation_index])[:, 1]
        family_iterations["topology"].append(int(topology_model.best_iteration_ or 800))
        compact_model = _tree(17)
        compact_model.fit(
            compact_matrix[train_index],
            labels[train_index],
            eval_set=[(fold_compact_validation[fold_number], labels[validation_index])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
        )
        compact_tree_prediction[positions] = compact_model.predict_proba(fold_compact_validation[fold_number])[:, 1]
        family_iterations["compact_graph"].append(int(compact_model.best_iteration_ or 800))
    blend_prediction = 0.7 * _rank_vector(tree_prediction) + 0.3 * _rank_vector(logistic_prediction)
    oof_labels = labels[oof_indices]
    metrics = {
        "direct_logistic_auc": float(roc_auc_score(oof_labels, direct_logistic_prediction)),
        "direct_tree_auc": float(roc_auc_score(oof_labels, direct_tree_prediction)),
        "topology_tree_auc": float(roc_auc_score(oof_labels, topology_tree_prediction)),
        "compact_graph_tree_auc": float(roc_auc_score(oof_labels, compact_tree_prediction)),
        "graph_logistic_auc": float(roc_auc_score(oof_labels, logistic_prediction)),
        "graph_tree_auc": float(roc_auc_score(oof_labels, tree_prediction)),
        "graph_blend_auc": float(roc_auc_score(oof_labels, blend_prediction)),
    }
    logistic_folds = _fold_auc(logistic_prediction, labels, folds, index_map)
    tree_folds = _fold_auc(tree_prediction, labels, folds, index_map)
    blend_folds = _fold_auc(blend_prediction, labels, folds, index_map)
    direct_logistic_folds = _fold_auc(direct_logistic_prediction, labels, folds, index_map)
    direct_tree_folds = _fold_auc(direct_tree_prediction, labels, folds, index_map)
    topology_tree_folds = _fold_auc(topology_tree_prediction, labels, folds, index_map)
    compact_tree_folds = _fold_auc(compact_tree_prediction, labels, folds, index_map)
    improvement = np.asarray(blend_folds) - np.maximum(logistic_folds, tree_folds)
    best_single = max(metrics["graph_logistic_auc"], metrics["graph_tree_auc"])
    stable_blend = (
        metrics["graph_blend_auc"] > best_single + 0.0005
        and float(improvement.mean()) > 0
        and float(improvement.min()) > -0.03
    )
    fold_metrics = {
        "direct_logistic": direct_logistic_folds,
        "direct_tree": direct_tree_folds,
        "topology_tree": topology_tree_folds,
        "compact_graph_tree": compact_tree_folds,
        "graph_logistic": logistic_folds,
        "graph_tree": tree_folds,
        "graph_rank_blend": blend_folds,
    }
    graph_head = "rank_blend" if stable_blend else (
        "tree" if metrics["graph_tree_auc"] >= metrics["graph_logistic_auc"] else "logistic"
    )
    candidates = {
        "direct_logistic": ("direct", "logistic", direct_logistic_prediction),
        "direct_tree": ("direct", "tree", direct_tree_prediction),
        "topology_tree": ("topology", "tree", topology_tree_prediction),
        "compact_graph_tree": ("compact_graph", "tree", compact_tree_prediction),
        f"graph_{graph_head}": (
            "graph",
            graph_head,
            blend_prediction if graph_head == "rank_blend" else (
                tree_prediction if graph_head == "tree" else logistic_prediction
            ),
        ),
    }
    candidate_scores = {
        name: float(np.mean(fold_metrics[name]) - 0.10 * np.std(fold_metrics[name]))
        for name in candidates
    }
    selected_name = max(candidate_scores, key=candidate_scores.get)
    feature_set, selected_head, selected_prediction = candidates[selected_name]
    direct_best = max(metrics["direct_logistic_auc"], metrics["direct_tree_auc"])
    graph_best = max(
        metrics["topology_tree_auc"],
        metrics["compact_graph_tree_auc"],
        metrics["graph_logistic_auc"],
        metrics["graph_tree_auc"],
        metrics["graph_blend_auc"],
    )
    metrics["graph_incremental_auc"] = graph_best - direct_best
    metrics["selected_oof_auc"] = float(roc_auc_score(oof_labels, selected_prediction))
    metrics["selected_stability_score"] = candidate_scores[selected_name]
    selected_iterations = (
        int(np.median(family_iterations[feature_set]))
        if feature_set in family_iterations
        else fixed_iterations
    )
    metrics["selected_tree_iterations"] = float(selected_iterations)
    print(
        f"[selection] logistic_c={selected_c} features={feature_set} head={selected_head} "
        f"metrics={json.dumps(metrics, sort_keys=True)} folds={json.dumps(fold_metrics)}"
    )
    return SelectionResult(
        logistic_c=selected_c,
        selected_head=selected_head,
        feature_set=feature_set,
        tree_iterations=max(1, min(800, selected_iterations)),
        metrics=metrics,
        fold_metrics=fold_metrics,
        oof_indices=oof_indices,
        oof_prediction=selected_prediction,
        direct_oof_prediction=(
            direct_tree_prediction
            if metrics["direct_tree_auc"] >= metrics["direct_logistic_auc"]
            else direct_logistic_prediction
        ),
    )


def debug_selection() -> SelectionResult:
    return SelectionResult(
        logistic_c=0.2,
        selected_head="logistic",
        feature_set="graph",
        tree_iterations=200,
        metrics={},
        fold_metrics={},
        oof_indices=np.empty(0, dtype=np.int32),
        oof_prediction=np.empty(0, dtype=np.float64),
        direct_oof_prediction=np.empty(0, dtype=np.float64),
    )


def _rank_calibrate(prediction: np.ndarray, base_rate: float) -> np.ndarray:
    quantile = _rank_vector(prediction)
    return expit(logit(np.clip(base_rate, 1e-4, 1 - 1e-4)) + 3.0 * (quantile - 0.5))


def fit_two_model_chains(
    train_a: np.ndarray,
    labels_a: np.ndarray,
    validation_a: np.ndarray,
    train_b: np.ndarray,
    labels_b: np.ndarray,
    test_b: np.ndarray,
    selection: SelectionResult,
) -> tuple[np.ndarray, np.ndarray]:
    def fit_chain(train: np.ndarray, labels: np.ndarray, target: np.ndarray) -> np.ndarray:
        logistic_prediction = None
        tree_prediction = None
        if selection.selected_head in {"logistic", "rank_blend"}:
            logistic_model = _logistic(selection.logistic_c)
            logistic_model.fit(train, labels)
            logistic_prediction = logistic_model.predict_proba(target)[:, 1]
        if selection.selected_head in {"tree", "rank_blend"}:
            seed_predictions = []
            for seed in [17, 43]:
                tree_model = _tree(seed, estimators=selection.tree_iterations)
                tree_model.fit(train, labels)
                seed_predictions.append(tree_model.predict_proba(target)[:, 1])
            tree_prediction = np.mean(seed_predictions, axis=0)
        if selection.selected_head == "rank_blend":
            raw = 0.7 * _rank_vector(tree_prediction) + 0.3 * _rank_vector(logistic_prediction)
        elif selection.selected_head == "tree":
            raw = tree_prediction
        else:
            raw = logistic_prediction
        return _rank_calibrate(raw, float(labels.mean())).astype(np.float64)

    validation_prediction = fit_chain(train_a, labels_a, validation_a)
    test_prediction = fit_chain(train_b, labels_b, test_b)
    return validation_prediction, test_prediction


def report_slices(
    labels: np.ndarray,
    prediction: np.ndarray,
    indices: np.ndarray,
    slices: dict[str, np.ndarray],
) -> dict[str, dict[str, float | int | None]]:
    report: dict[str, dict[str, float | int | None]] = {}
    for name, mask in slices.items():
        selected = mask[indices]
        current_labels = labels[indices][selected]
        current_prediction = prediction[selected]
        score = None
        if len(current_labels) and np.unique(current_labels).size == 2:
            score = float(roc_auc_score(current_labels, current_prediction))
        report[name] = {
            "count": int(len(current_labels)),
            "positives": int(current_labels.sum()) if len(current_labels) else 0,
            "roc_auc": score,
        }
    print(f"[oof_slices] {json.dumps(report, sort_keys=True)}")
    return report
