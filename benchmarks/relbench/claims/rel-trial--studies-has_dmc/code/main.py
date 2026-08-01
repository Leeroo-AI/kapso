from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from clinical_features import build_clinical_bundle
from embedding_cache import get_medcpt_embeddings
from empirical_bayes import CausalEmpiricalBayes
from kapso_datasets.common import is_debug, load_task, run_data_dir, save_predictions, shared_cache_dir
from retrieval_features import CausalRetrieval


SEED = 23


def elapsed(start: float, phase: str) -> None:
    print(f"[timing] phase={phase} elapsed={time.time() - start:.1f}s", flush=True)


def feature_cache_path(shared: Path, content_key: str, debug: bool, train_ids: np.ndarray, labels: np.ndarray) -> Path:
    digest = hashlib.sha256()
    digest.update(content_key.encode())
    digest.update(np.asarray(train_ids, dtype=np.int64).tobytes())
    digest.update(np.asarray(labels, dtype=np.float32).tobytes())
    digest.update(b"causal_retrieval_v4_empirical_bayes_v2")
    mode = "debug" if debug else "full"
    return shared / f"lane2_features_{mode}_{digest.hexdigest()[:20]}.npz"


def build_label_features(bundle, embeddings, train_indices, val_indices, test_indices, train_labels, val_labels, debug, cache_path):
    if cache_path.exists():
        print(f"[features] cache hit path={cache_path}", flush=True)
        cached = np.load(cache_path, allow_pickle=False)
        return {name: cached[name] for name in cached.files}

    priors = CausalEmpiricalBayes(bundle.groups)
    prior_train = priors.process_causal(train_indices, bundle.dates[train_indices], train_labels)
    prior_val_a = priors.transform(val_indices)
    prior_val_b = priors.process_causal(val_indices, bundle.dates[val_indices], val_labels)
    prior_test = priors.transform(test_indices)

    retrieval = CausalRetrieval(embeddings, bundle.signatures, len(train_indices) + len(val_indices), debug)
    retrieval_train = retrieval.process_causal(train_indices, bundle.dates[train_indices], train_labels, "train")
    retrieval_val_a = retrieval.transform(val_indices, "validation_model_a")
    retrieval_val_b = retrieval.process_causal(val_indices, bundle.dates[val_indices], val_labels, "validation_model_b_history")
    retrieval_test = retrieval.transform(test_indices, "test_model_b")

    arrays = {
        "prior_train": prior_train,
        "prior_val_a": prior_val_a,
        "prior_val_b": prior_val_b,
        "prior_test": prior_test,
        "retrieval_train": retrieval_train,
        "retrieval_val_a": retrieval_val_a,
        "retrieval_val_b": retrieval_val_b,
        "retrieval_test": retrieval_test,
        "prior_names": np.asarray(priors.feature_names),
        "retrieval_names": np.asarray(retrieval.feature_names),
    }
    temporary = cache_path.with_suffix(cache_path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, cache_path)
    return arrays


def assemble_frame(structured: pd.DataFrame, indices: np.ndarray, prior: np.ndarray, retrieval: np.ndarray, prior_names: list[str], retrieval_names: list[str]) -> pd.DataFrame:
    numeric = pd.DataFrame(np.column_stack([prior, retrieval]), columns=prior_names + retrieval_names)
    return pd.concat([structured.iloc[indices].reset_index(drop=True), numeric], axis=1)


def catboost_model(iterations: int, categorical: list[str], debug: bool) -> CatBoostClassifier:
    parameters = {
        "iterations": iterations,
        "depth": 7,
        "learning_rate": 0.05,
        "l2_leaf_reg": 10.0,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": SEED,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))),
        "cat_features": categorical,
    }
    if not debug:
        parameters.update({"task_type": "GPU", "devices": os.environ.get("CUDA_DEVICE", "0")})
    return CatBoostClassifier(**parameters)


def fit_catboost(frame: pd.DataFrame, labels: np.ndarray, iterations: int, categorical: list[str], debug: bool) -> CatBoostClassifier:
    model = catboost_model(iterations, categorical, debug)
    try:
        model.fit(frame, labels)
    except Exception as error:
        if debug:
            raise
        print(f"[model] GPU CatBoost failed; retrying CPU: {type(error).__name__}: {error}", flush=True)
        model = catboost_model(iterations, categorical, True)
        model.fit(frame, labels)
    return model


def linear_pipeline(frame: pd.DataFrame, categorical: list[str]) -> Pipeline:
    allowed_categories = [
        name for name in categorical
        if not name.startswith("primary_") and name in frame.columns
    ]
    numeric = [name for name in frame.columns if name not in categorical and not name.startswith("retrieval_")]
    transformer = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler(with_mean=False)),
        ]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=100), allowed_categories),
    ])
    classifier = LogisticRegression(
        C=0.2,
        solver="saga",
        max_iter=150,
        tol=1e-3,
        random_state=SEED,
        n_jobs=max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))),
    )
    return Pipeline([("features", transformer), ("classifier", classifier)])


def forward_diagnostics(frame: pd.DataFrame, labels: np.ndarray, dates: pd.Series, categorical: list[str]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, dict]:
    rounds = [350, 500, 700]
    fold_records = {}
    predictions_by_round = {value: [] for value in rounds}
    linear_predictions = []
    positions = []
    targets = []
    for year in [2017, 2018, 2019]:
        train_mask = dates.dt.year.to_numpy() < year
        holdout_mask = dates.dt.year.to_numpy() == year
        train_positions = np.flatnonzero(train_mask)
        holdout_positions = np.flatnonzero(holdout_mask)
        model = fit_catboost(frame.iloc[train_positions], labels[train_positions], 700, categorical, False)
        round_scores = {}
        for value in rounds:
            prediction = model.predict_proba(frame.iloc[holdout_positions], ntree_end=value)[:, 1]
            predictions_by_round[value].append(prediction)
            round_scores[str(value)] = float(roc_auc_score(labels[holdout_positions], prediction))
        linear = linear_pipeline(frame.iloc[train_positions], categorical)
        linear.fit(frame.iloc[train_positions], labels[train_positions])
        linear_prediction = linear.predict_proba(frame.iloc[holdout_positions])[:, 1]
        linear_predictions.append(linear_prediction)
        fold_records[str(year)] = {
            "count": int(len(holdout_positions)),
            "catboost_auc_by_round": round_scores,
            "linear_eb_auc": float(roc_auc_score(labels[holdout_positions], linear_prediction)),
        }
        positions.append(holdout_positions)
        targets.append(labels[holdout_positions])
        print(f"[forward] year={year} diagnostics={json.dumps(fold_records[str(year)], sort_keys=True)}", flush=True)
    mean_scores = {
        value: float(np.mean([fold_records[str(year)]["catboost_auc_by_round"][str(value)] for year in [2017, 2018, 2019]]))
        for value in rounds
    }
    selected = max(rounds, key=lambda value: (mean_scores[value], -value))
    diagnostics = {"folds": fold_records, "mean_catboost_auc_by_round": mean_scores, "selected_rounds": selected}
    return (
        selected,
        np.concatenate([np.asarray(value) for value in predictions_by_round[selected]]),
        np.concatenate(linear_predictions),
        np.concatenate(targets),
        diagnostics,
    )


def select_similarity_threshold(frame: pd.DataFrame, labels: np.ndarray, dates: pd.Series) -> tuple[str, dict]:
    candidates = ["0.85", "0.90", "0.95"]
    diagnostics = {}
    for threshold in candidates:
        column = f"retrieval_threshold_{threshold}_weighted_rate"
        fold_scores = {}
        for year in [2017, 2018, 2019]:
            mask = dates.dt.year.to_numpy() == year
            fold_scores[str(year)] = float(roc_auc_score(labels[mask], frame.loc[mask, column]))
        diagnostics[threshold] = {"folds": fold_scores, "mean": float(np.mean(list(fold_scores.values())))}
    selected = max(candidates, key=lambda threshold: diagnostics[threshold]["mean"])
    return selected, diagnostics


def retain_similarity_threshold(frames: list[pd.DataFrame], selected: str) -> None:
    for frame in frames:
        discarded = [
            column for column in frame.columns
            if column.startswith("retrieval_threshold_") and f"retrieval_threshold_{selected}_" not in column
        ]
        frame.drop(columns=discarded, inplace=True)


def fit_stacker(cat_predictions: np.ndarray, linear_predictions: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    epsilon = 1e-5
    cat_logit = np.log(np.clip(cat_predictions, epsilon, 1 - epsilon) / np.clip(1 - cat_predictions, epsilon, 1 - epsilon))
    linear_logit = np.log(np.clip(linear_predictions, epsilon, 1 - epsilon) / np.clip(1 - linear_predictions, epsilon, 1 - epsilon))
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=SEED)
    model.fit(np.column_stack([cat_logit, linear_logit]), labels)
    return model


def stack_predict(model: LogisticRegression, cat_predictions: np.ndarray, linear_predictions: np.ndarray) -> np.ndarray:
    epsilon = 1e-5
    cat_logit = np.log(np.clip(cat_predictions, epsilon, 1 - epsilon) / np.clip(1 - cat_predictions, epsilon, 1 - epsilon))
    linear_logit = np.log(np.clip(linear_predictions, epsilon, 1 - epsilon) / np.clip(1 - linear_predictions, epsilon, 1 - epsilon))
    return model.predict_proba(np.column_stack([cat_logit, linear_logit]))[:, 1]


def slice_auc(labels: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    values = labels[mask]
    if len(values) < 2 or len(np.unique(values)) < 2:
        return {"count": int(len(values)), "roc_auc": None}
    return {"count": int(len(values)), "roc_auc": float(roc_auc_score(values, predictions[mask]))}


def validation_slices(bundle, train_indices, val_indices, labels, predictions, label_features, structured) -> dict:
    output = {}
    val_dates = pd.Series(pd.to_datetime(bundle.dates[val_indices]))
    output["half_1"] = slice_auc(labels, predictions, val_dates.dt.month.to_numpy() <= 6)
    output["half_2"] = slice_auc(labels, predictions, val_dates.dt.month.to_numpy() > 6)
    for month in range(1, 13):
        output[f"month_{month:02d}"] = slice_auc(labels, predictions, val_dates.dt.month.to_numpy() == month)
    for family in ["lead_sponsor", "condition", "facility"]:
        known = set(key for index in train_indices for key in bundle.groups[family][int(index)])
        seen = np.asarray([any(key in known for key in bundle.groups[family][int(index)]) for index in val_indices])
        output[f"{family}_seen"] = slice_auc(labels, predictions, seen)
        output[f"{family}_unseen"] = slice_auc(labels, predictions, ~seen)
    study_type = structured.iloc[val_indices]["study_type"].astype(str).to_numpy()
    output["interventional"] = slice_auc(labels, predictions, study_type == "Interventional")
    output["non_interventional"] = slice_auc(labels, predictions, study_type != "Interventional")
    detailed = structured.iloc[val_indices]["text_detailed_missing"].to_numpy() < 0.5
    output["detailed_present"] = slice_auc(labels, predictions, detailed)
    output["detailed_missing"] = slice_auc(labels, predictions, ~detailed)
    retrieval_names = [str(value) for value in label_features["retrieval_names"]]
    top_name = "retrieval_k16_top_similarity"
    if top_name in retrieval_names:
        top = label_features["retrieval_val_a"][:, retrieval_names.index(top_name)]
        for low, high in [(-2, 0.5), (0.5, 0.7), (0.7, 2)]:
            output[f"top_similarity_{low}_{high}"] = slice_auc(labels, predictions, (top >= low) & (top < high))
    return output


def main() -> None:
    warnings.filterwarnings("ignore")
    np.random.seed(SEED)
    debug = is_debug()
    started = time.time()
    Path("output_data_generic_exp_2").mkdir(parents=True, exist_ok=True)
    context = load_task(upto_test_timestamp=False)
    train = context.train.df.reset_index(drop=True)
    validation = context.val.df.reset_index(drop=True)
    test = context.test.df.reset_index(drop=True)
    target = context.target_col
    all_rows = pd.concat([
        train[["nct_id", "start_date"]],
        validation[["nct_id", "start_date"]],
        test[["nct_id", "start_date"]],
    ], ignore_index=True)
    if all_rows["nct_id"].duplicated().any():
        raise RuntimeError("task entity ids are not unique across splits")
    n_train = len(train)
    n_validation = len(validation)
    train_start = max(0, n_train - 20000) if debug else 0
    train_indices = np.arange(train_start, n_train, dtype=np.int64)
    val_indices = np.arange(n_train, n_train + n_validation, dtype=np.int64)
    test_indices = np.arange(n_train + n_validation, len(all_rows), dtype=np.int64)
    train_labels = train[target].to_numpy(dtype=np.float32)[train_start:]
    validation_labels = validation[target].to_numpy(dtype=np.float32)
    elapsed(started, "load_task")

    bundle = build_clinical_bundle(context.db, all_rows)
    elapsed(started, "relational_documents")
    embeddings, content_key = get_medcpt_embeddings(bundle, shared_cache_dir(), debug)
    elapsed(started, "medcpt_embeddings")

    cache_path = feature_cache_path(shared_cache_dir(), content_key, debug, bundle.ids[train_indices], train_labels)
    label_features = build_label_features(
        bundle, embeddings, train_indices, val_indices, test_indices,
        train_labels, validation_labels, debug, cache_path,
    )
    elapsed(started, "causal_label_features")

    prior_names = [str(value) for value in label_features["prior_names"]]
    retrieval_names = [str(value) for value in label_features["retrieval_names"]]
    structured = bundle.features
    frame_train = assemble_frame(structured, train_indices, label_features["prior_train"], label_features["retrieval_train"], prior_names, retrieval_names)
    frame_val_a = assemble_frame(structured, val_indices, label_features["prior_val_a"], label_features["retrieval_val_a"], prior_names, retrieval_names)
    frame_val_b = assemble_frame(structured, val_indices, label_features["prior_val_b"], label_features["retrieval_val_b"], prior_names, retrieval_names)
    frame_test = assemble_frame(structured, test_indices, label_features["prior_test"], label_features["retrieval_test"], prior_names, retrieval_names)
    categorical = [name for name in bundle.categorical if name in frame_train.columns]

    if debug:
        selected_rounds = 120
        diagnostics = {"debug": True, "selected_rounds": selected_rounds}
        model_a = fit_catboost(frame_train, train_labels, selected_rounds, categorical, True)
        validation_predictions = model_a.predict_proba(frame_val_a)[:, 1]
        model_b_frame = pd.concat([frame_train, frame_val_b], ignore_index=True)
        model_b_labels = np.concatenate([train_labels, validation_labels])
        model_b = fit_catboost(model_b_frame, model_b_labels, selected_rounds, categorical, True)
        test_predictions = model_b.predict_proba(frame_test)[:, 1]
    else:
        train_dates = pd.Series(pd.to_datetime(bundle.dates[train_indices]))
        selected_threshold, threshold_diagnostics = select_similarity_threshold(frame_train, train_labels, train_dates)
        retain_similarity_threshold([frame_train, frame_val_a, frame_val_b, frame_test], selected_threshold)
        print(f"[forward] similarity_threshold={selected_threshold} diagnostics={json.dumps(threshold_diagnostics, sort_keys=True)}", flush=True)
        selected_rounds, oof_cat, oof_linear, oof_labels, diagnostics = forward_diagnostics(
            frame_train, train_labels, train_dates, categorical,
        )
        diagnostics["similarity_threshold"] = selected_threshold
        diagnostics["similarity_threshold_diagnostics"] = threshold_diagnostics
        model_a = fit_catboost(frame_train, train_labels, selected_rounds, categorical, False)
        validation_cat = model_a.predict_proba(frame_val_a)[:, 1]
        linear_a = linear_pipeline(frame_train, categorical)
        linear_a.fit(frame_train, train_labels)
        validation_linear = linear_a.predict_proba(frame_val_a)[:, 1]
        stacker_a = fit_stacker(oof_cat, oof_linear, oof_labels)
        validation_predictions = stack_predict(stacker_a, validation_cat, validation_linear)

        model_b_frame = pd.concat([frame_train, frame_val_b], ignore_index=True)
        model_b_labels = np.concatenate([train_labels, validation_labels])
        model_b = fit_catboost(model_b_frame, model_b_labels, selected_rounds, categorical, False)
        test_cat = model_b.predict_proba(frame_test)[:, 1]
        linear_b = linear_pipeline(model_b_frame, categorical)
        linear_b.fit(model_b_frame, model_b_labels)
        test_linear = linear_b.predict_proba(frame_test)[:, 1]
        stacker_b = fit_stacker(
            np.concatenate([oof_cat, validation_cat]),
            np.concatenate([oof_linear, validation_linear]),
            np.concatenate([oof_labels, validation_labels]),
        )
        test_predictions = stack_predict(stacker_b, test_cat, test_linear)
        diagnostics["stacker_a_coefficients"] = stacker_a.coef_.tolist()
        diagnostics["stacker_b_coefficients"] = stacker_b.coef_.tolist()
        diagnostics["validation_component_auc"] = {
            "catboost": float(roc_auc_score(validation_labels, validation_cat)),
            "linear_eb": float(roc_auc_score(validation_labels, validation_linear)),
            "stacked": float(roc_auc_score(validation_labels, validation_predictions)),
        }
    elapsed(started, "model_a_model_b")

    validation_predictions = np.clip(np.asarray(validation_predictions, dtype=np.float64), 0, 1)
    test_predictions = np.clip(np.asarray(test_predictions, dtype=np.float64), 0, 1)
    if validation_predictions.shape != (len(validation),) or test_predictions.shape != (len(test),):
        raise RuntimeError(f"prediction shape mismatch: {validation_predictions.shape} {test_predictions.shape}")
    save_predictions(validation_predictions, test_predictions)
    slices = validation_slices(bundle, train_indices, val_indices, validation_labels, validation_predictions, label_features, structured)
    metrics = {
        "validation_roc_auc_self_check": float(roc_auc_score(validation_labels, validation_predictions)),
        "validation_slices": slices,
        "forward_diagnostics": diagnostics,
        "fit_contract": {
            "validation_chain": "model A and stacker A fit without validation labels",
            "test_chain": "model B fit on train plus validation after model A predictions were preserved",
        },
        "content_key": content_key,
        "debug": debug,
        "elapsed_seconds": time.time() - started,
    }
    (run_data_dir() / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"[diagnostics] validation_roc_auc={metrics['validation_roc_auc_self_check']:.6f} slices={json.dumps(slices, sort_keys=True)}", flush=True)
    elapsed(started, "complete")


if __name__ == "__main__":
    main()
