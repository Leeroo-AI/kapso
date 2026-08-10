from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from feature_mart import SEED, within_timestamp_features


THREADS = int(os.environ.get("OMP_NUM_THREADS", "40"))
DESIGNS = ["structured", "structured_embeddings", "structured_neighbors", "full"]


@dataclass
class CategoryTransform:
    ordinal: dict[str, dict[str, int]]
    frequency: dict[str, dict[str, float]]


def finite_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def fit_categories(categories: pd.DataFrame, train_ids: np.ndarray) -> CategoryTransform:
    ordinal = {}
    frequency = {}
    train = categories.loc[train_ids]
    for col in categories:
        values = train[col].fillna("__MISSING__").astype(str)
        counts = values.value_counts(dropna=False)
        ordered = sorted(counts.index.tolist(), key=lambda x: (-counts[x], x))
        ordinal[col] = {value: index for index, value in enumerate(ordered)}
        frequency[col] = (counts / len(values)).to_dict()
    return CategoryTransform(ordinal=ordinal, frequency=frequency)


def transform_categories(categories: pd.DataFrame, row_ids: np.ndarray, transform: CategoryTransform) -> np.ndarray:
    frame = categories.loc[row_ids]
    columns = []
    for col in categories:
        values = frame[col].fillna("__MISSING__").astype(str)
        columns.append(values.map(transform.ordinal[col]).fillna(-1).to_numpy(dtype=np.float32))
        columns.append(values.map(transform.frequency[col]).fillna(0).to_numpy(dtype=np.float32))
    return np.column_stack(columns) if columns else np.empty((len(row_ids), 0), dtype=np.float32)


def category_feature_names(categories: pd.DataFrame) -> list[str]:
    names = []
    for col in categories:
        names.extend([f"cat_{col}_ordinal", f"cat_{col}_frequency"])
    return names


def dense_matrices(numeric: pd.DataFrame, categories: pd.DataFrame, train_ids: np.ndarray, target_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    transform = fit_categories(categories, train_ids)
    train_numeric = numeric.loc[train_ids].to_numpy(dtype=np.float32)
    target_numeric = numeric.loc[target_ids].to_numpy(dtype=np.float32)
    train_category = transform_categories(categories, train_ids, transform)
    target_category = transform_categories(categories, target_ids, transform)
    train_matrix = np.column_stack([train_numeric, train_category])
    target_matrix = np.column_stack([target_numeric, target_category])
    return train_matrix, target_matrix, list(numeric.columns) + category_feature_names(categories)


def count_model(rounds: int = 300) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=rounds,
        learning_rate=0.04,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=80,
        reg_alpha=0.0,
        reg_lambda=2.0,
        random_state=SEED,
        n_jobs=THREADS,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def count_predictions(numeric: pd.DataFrame, categories: pd.DataFrame, fit_ids: np.ndarray, target_ids: np.ndarray, counts: np.ndarray, debug: bool) -> tuple[np.ndarray, np.ndarray]:
    splits = 2 if debug else 5
    splits = min(splits, max(2, len(fit_ids) // 100))
    kfold = KFold(n_splits=splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(fit_ids), dtype=np.float64)
    rounds = 80 if debug else 300
    for fold, (train_position, valid_position) in enumerate(kfold.split(fit_ids)):
        train_ids = fit_ids[train_position]
        valid_ids = fit_ids[valid_position]
        x_train, x_valid, _ = dense_matrices(numeric, categories, train_ids, valid_ids)
        model = count_model(rounds)
        model.fit(x_train, np.log1p(counts[train_position]), callbacks=[lgb.log_evaluation(0)])
        oof[valid_position] = np.expm1(model.predict(x_valid))
    x_fit, x_target, _ = dense_matrices(numeric, categories, fit_ids, target_ids)
    model = count_model(rounds)
    model.fit(x_fit, np.log1p(counts), callbacks=[lgb.log_evaluation(0)])
    target = np.expm1(model.predict(x_target))
    return np.clip(oof, 0, 100), np.clip(target, 0, 100)


def classifier(rounds: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=rounds,
        learning_rate=0.02,
        num_leaves=15,
        max_depth=5,
        min_child_samples=100,
        reg_alpha=1.0,
        reg_lambda=10.0,
        random_state=SEED,
        n_jobs=THREADS,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def fit_pca(embeddings: np.ndarray, train_ids: np.ndarray, target_ids: np.ndarray, debug: bool) -> tuple[np.ndarray, np.ndarray]:
    components = min(16 if debug else 64, len(train_ids) - 1, embeddings.shape[1])
    model = PCA(n_components=components, svd_solver="randomized", iterated_power=3, random_state=SEED)
    train = model.fit_transform(embeddings[train_ids])
    target = model.transform(embeddings[target_ids])
    return train.astype(np.float32), target.astype(np.float32)


def compose_numeric(structured: pd.DataFrame, neighbors: pd.DataFrame, row_ids: np.ndarray, design: str, train_pca: np.ndarray | None, target_pca: np.ndarray | None, train_ids: np.ndarray, target_ids: np.ndarray, count_train: np.ndarray | None, count_target: np.ndarray | None, seeds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = structured.loc[train_ids].copy()
    target = structured.loc[target_ids].copy()
    if design in {"structured_neighbors", "full"}:
        train = train.join(neighbors.loc[train_ids])
        target = target.join(neighbors.loc[target_ids])
    if design in {"structured_embeddings", "full"} and train_pca is not None and target_pca is not None:
        names = [f"embedding_pca_{j}" for j in range(train_pca.shape[1])]
        train = train.join(pd.DataFrame(train_pca, index=train_ids, columns=names))
        target = target.join(pd.DataFrame(target_pca, index=target_ids, columns=names))
    if design == "full" and count_train is not None and count_target is not None:
        count_frame = pd.DataFrame(index=np.concatenate([train_ids, target_ids]))
        count_frame["predicted_analysis_count"] = np.concatenate([count_train, count_target])
        count_seeds = seeds[seeds["row_id"].isin(count_frame.index)]
        normalized = within_timestamp_features(count_seeds, count_frame, ["predicted_analysis_count"])
        normalized = normalized.drop(columns=["within_small_cohort"], errors="ignore")
        count_frame = count_frame.join(normalized)
        train = train.join(count_frame.loc[train_ids])
        target = target.join(count_frame.loc[target_ids])
    return finite_frame(train), finite_frame(target)


def add_pca_features(train: pd.DataFrame, target: pd.DataFrame, train_ids: np.ndarray, target_ids: np.ndarray, train_pca: np.ndarray, target_pca: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = [f"embedding_pca_{j}" for j in range(train_pca.shape[1])]
    if names[0] not in train:
        train = train.join(pd.DataFrame(train_pca, index=train_ids, columns=names))
        target = target.join(pd.DataFrame(target_pca, index=target_ids, columns=names))
    return train, target


def fit_lgb_fold(train_numeric: pd.DataFrame, valid_numeric: pd.DataFrame, categories: pd.DataFrame, train_ids: np.ndarray, valid_ids: np.ndarray, labels: np.ndarray, valid_labels: np.ndarray, debug: bool) -> tuple[np.ndarray, int]:
    x_train, x_valid, _ = dense_matrices(pd.concat([train_numeric, valid_numeric]), categories, train_ids, valid_ids)
    maximum = 120 if debug else 1400
    model = classifier(maximum)
    callbacks = [lgb.log_evaluation(0)]
    if not debug:
        callbacks.append(lgb.early_stopping(100, verbose=False))
    model.fit(x_train, labels, eval_set=[(x_valid, valid_labels)], eval_metric="auc", callbacks=callbacks)
    rounds = int(model.best_iteration_ or maximum)
    return model.predict_proba(x_valid, num_iteration=rounds)[:, 1], rounds


def make_text_matrix(texts: list[str]) -> sparse.csr_matrix:
    word = HashingVectorizer(n_features=2 ** 15, alternate_sign=False, norm="l2", lowercase=True, ngram_range=(1, 2), analyzer="word")
    char = HashingVectorizer(n_features=2 ** 15, alternate_sign=False, norm="l2", lowercase=True, ngram_range=(3, 5), analyzer="char_wb")
    return sparse.hstack([word.transform(texts), char.transform(texts)], format="csr", dtype=np.float32)


def logistic_matrices(train_numeric: pd.DataFrame, target_numeric: pd.DataFrame, train_ids: np.ndarray, target_ids: np.ndarray, raw_embeddings: np.ndarray | None, text_matrix: sparse.csr_matrix | None) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    dense_train = imputer.fit_transform(train_numeric.loc[train_ids])
    dense_target = imputer.transform(target_numeric.loc[target_ids])
    dense_train = scaler.fit_transform(dense_train).astype(np.float32)
    dense_target = scaler.transform(dense_target).astype(np.float32)
    pieces_train = [sparse.csr_matrix(dense_train)]
    pieces_target = [sparse.csr_matrix(dense_target)]
    if raw_embeddings is not None:
        embedding_scaler = StandardScaler()
        embed_train = embedding_scaler.fit_transform(raw_embeddings[train_ids]).astype(np.float32)
        embed_target = embedding_scaler.transform(raw_embeddings[target_ids]).astype(np.float32)
        pieces_train.append(sparse.csr_matrix(embed_train))
        pieces_target.append(sparse.csr_matrix(embed_target))
    if text_matrix is not None:
        pieces_train.append(text_matrix[train_ids])
        pieces_target.append(text_matrix[target_ids])
    return sparse.hstack(pieces_train, format="csr"), sparse.hstack(pieces_target, format="csr")


def fit_logistic(train_numeric: pd.DataFrame, target_numeric: pd.DataFrame, train_ids: np.ndarray, target_ids: np.ndarray, labels: np.ndarray, raw_embeddings: np.ndarray | None, text_matrix: sparse.csr_matrix | None, c_value: float, debug: bool) -> np.ndarray:
    x_train, x_target = logistic_matrices(train_numeric, target_numeric, train_ids, target_ids, raw_embeddings, text_matrix)
    model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", dual=True, max_iter=300 if debug else 3000, random_state=SEED, tol=1e-4)
    model.fit(x_train, labels)
    return model.predict_proba(x_target)[:, 1]


def prepare_structured(seeds: pd.DataFrame, base_numeric: pd.DataFrame, neighbor_numeric: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "study_log_enrollment", "study_enrollment", "study_age_days", "study_enrollment_per_arm", "study_enrollment_per_site",
        "rel_site_count", "rel_country_count", "rel_condition_count", "rel_intervention_count", "rel_sponsor_count",
        "elig_criteria_chars", "elig_criteria_words", "elig_criteria_bullets", "study_summary_chars", "study_description_chars",
    ]
    preferred.extend([col for col in base_numeric if col.endswith("success_rate_wmean")][:15])
    normalized = within_timestamp_features(seeds, base_numeric, preferred)
    sem_preferred = [col for col in neighbor_numeric if col.endswith("_success") or col.endswith("_N")][:12]
    sem_normalized = within_timestamp_features(seeds, neighbor_numeric, sem_preferred)
    sem_normalized = sem_normalized.drop(columns=["within_small_cohort"], errors="ignore")
    structured = base_numeric.join(normalized)
    neighbor_numeric.loc[:, sem_normalized.columns] = sem_normalized
    return finite_frame(structured)


def forward_design_selection(seeds: pd.DataFrame, structured: pd.DataFrame, neighbors: pd.DataFrame, categories: pd.DataFrame, embeddings: np.ndarray, texts: list[str], true_counts: np.ndarray, debug: bool) -> dict:
    train_rows = seeds[seeds["split"].eq("train")]
    labels_by_id = train_rows.set_index("row_id")["outcome"]
    years = [2016, 2017, 2018, 2019]
    if debug:
        years = [2019]
    predictions = {design: pd.Series(np.nan, index=train_rows["row_id"], dtype=float) for design in DESIGNS}
    rounds_by_design = {design: [] for design in DESIGNS}
    scores = {design: [] for design in DESIGNS}
    fold_cache = []
    text_matrix = make_text_matrix(texts)
    for year in years:
        fit_ids_all = train_rows.loc[train_rows["timestamp"].dt.year.lt(year), "row_id"].to_numpy(dtype=np.int64)
        valid_ids = train_rows.loc[train_rows["timestamp"].dt.year.eq(year), "row_id"].to_numpy(dtype=np.int64)
        if debug and len(fit_ids_all) > 2000:
            positions = np.linspace(0, len(fit_ids_all) - 1, 2000).astype(int)
            fit_ids = fit_ids_all[positions]
        else:
            fit_ids = fit_ids_all
        count_values = true_counts[fit_ids]
        count_train, count_valid = count_predictions(structured, categories, fit_ids, valid_ids, count_values, debug)
        pca_train, pca_valid = fit_pca(embeddings, fit_ids, valid_ids, debug)
        fold_numeric = {}
        for design in DESIGNS:
            train_numeric, valid_numeric = compose_numeric(structured, neighbors, np.concatenate([fit_ids, valid_ids]), design, pca_train, pca_valid, fit_ids, valid_ids, count_train, count_valid, seeds)
            prediction, rounds = fit_lgb_fold(train_numeric, valid_numeric, categories, fit_ids, valid_ids, labels_by_id.loc[fit_ids].to_numpy(dtype=int), labels_by_id.loc[valid_ids].to_numpy(dtype=int), debug)
            score = roc_auc_score(labels_by_id.loc[valid_ids], prediction)
            predictions[design].loc[valid_ids] = prediction
            scores[design].append(float(score))
            rounds_by_design[design].append(rounds)
            fold_numeric[design] = (train_numeric, valid_numeric)
            print(f"[selection] year={year} design={design} auc={score:.6f} rounds={rounds}", flush=True)
        fold_cache.append((year, fit_ids, valid_ids, fold_numeric, pca_train, pca_valid))
    summary = {}
    for design in DESIGNS:
        values = np.asarray(scores[design], dtype=float)
        summary[design] = {"fold_scores": values.tolist(), "mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0, "rounds": rounds_by_design[design]}
    best_mean = max(value["mean"] for value in summary.values())
    best_design = max(DESIGNS, key=lambda x: summary[x]["mean"])
    threshold = summary[best_design]["se"]
    selected = next(design for design in DESIGNS if best_mean - summary[design]["mean"] <= threshold)
    if debug:
        selected = "full"
    logistic_scores = {0.02: [], 0.05: []}
    logistic_predictions = {0.02: pd.Series(np.nan, index=train_rows["row_id"], dtype=float), 0.05: pd.Series(np.nan, index=train_rows["row_id"], dtype=float)}
    for year, fit_ids, valid_ids, fold_numeric, _, _ in fold_cache:
        train_numeric, valid_numeric = fold_numeric[selected]
        record = next(item for item in fold_cache if item[0] == year)
        train_numeric, valid_numeric = add_pca_features(train_numeric, valid_numeric, fit_ids, valid_ids, record[4], record[5])
        for c_value in [0.02, 0.05]:
            prediction = fit_logistic(train_numeric, valid_numeric, fit_ids, valid_ids, labels_by_id.loc[fit_ids].to_numpy(dtype=int), embeddings, text_matrix, c_value, debug)
            score = roc_auc_score(labels_by_id.loc[valid_ids], prediction)
            logistic_scores[c_value].append(float(score))
            logistic_predictions[c_value].loc[valid_ids] = prediction
            print(f"[selection] year={year} logistic_C={c_value:.2f} auc={score:.6f}", flush=True)
    chosen_c = max([0.02, 0.05], key=lambda x: np.mean(logistic_scores[x]))
    pooled_ids = np.concatenate([item[2] for item in fold_cache])
    pooled_labels = labels_by_id.loc[pooled_ids].to_numpy(dtype=int)
    blend_scores = {}
    for weight in [0.5, 0.7, 0.85]:
        blended = weight * predictions[selected].loc[pooled_ids].to_numpy() + (1 - weight) * logistic_predictions[chosen_c].loc[pooled_ids].to_numpy()
        blend_scores[weight] = float(roc_auc_score(pooled_labels, blended))
    blend_weight = max([0.5, 0.7, 0.85], key=lambda x: blend_scores[x])
    selected_rounds = int(np.median(rounds_by_design[selected]))
    selected_rounds = max(20, min(1400, selected_rounds))
    per_fold = []
    final_oof = blend_weight * predictions[selected] + (1 - blend_weight) * logistic_predictions[chosen_c]
    for year in years:
        ids = train_rows.loc[train_rows["timestamp"].dt.year.eq(year), "row_id"].to_numpy()
        per_fold.append({"year": year, "count": len(ids), "auc": float(roc_auc_score(labels_by_id.loc[ids], final_oof.loc[ids]))})
    slice_metrics = slice_report(train_rows, labels_by_id, final_oof, structured)
    result = {"designs": summary, "selected_design": selected, "selected_rounds": selected_rounds, "logistic_scores": {str(k): v for k, v in logistic_scores.items()}, "selected_c": chosen_c, "blend_scores": {str(k): v for k, v in blend_scores.items()}, "blend_weight": blend_weight, "forward_folds": per_fold, "slices": slice_metrics}
    print(f"[selection] frozen design={selected} rounds={selected_rounds} C={chosen_c} lgb_weight={blend_weight}", flush=True)
    return result


def slice_report(train_rows: pd.DataFrame, labels: pd.Series, predictions: pd.Series, structured: pd.DataFrame) -> list[dict]:
    output = []
    sites = structured.loc[train_rows["row_id"], "rel_site_count"]
    strata = pd.cut(sites, bins=[-np.inf, 4, 19, np.inf], labels=["sites_0_4", "sites_5_19", "sites_20_plus"])
    for stratum in strata.cat.categories:
        ids = train_rows.loc[strata.to_numpy() == stratum, "row_id"].to_numpy()
        available = predictions.loc[ids].notna()
        ids = ids[available.to_numpy()]
        if len(ids) and labels.loc[ids].nunique() == 2:
            output.append({"axis": "site_richness", "stratum": str(stratum), "count": len(ids), "auc": float(roc_auc_score(labels.loc[ids], predictions.loc[ids]))})
    return output


def fit_final_chain(seeds: pd.DataFrame, structured: pd.DataFrame, neighbors: pd.DataFrame, categories: pd.DataFrame, embeddings: np.ndarray, texts: list[str], episodes_counts: np.ndarray, fit_ids: np.ndarray, target_ids: np.ndarray, selection: dict, debug: bool) -> tuple[np.ndarray, dict]:
    count_train, count_target = count_predictions(structured, categories, fit_ids, target_ids, episodes_counts, debug)
    pca_train, pca_target = fit_pca(embeddings, fit_ids, target_ids, debug)
    train_numeric, target_numeric = compose_numeric(structured, neighbors, np.concatenate([fit_ids, target_ids]), selection["selected_design"], pca_train, pca_target, fit_ids, target_ids, count_train, count_target, seeds)
    x_train, x_target, names = dense_matrices(pd.concat([train_numeric, target_numeric]), categories, fit_ids, target_ids)
    model = classifier(120 if debug else selection["selected_rounds"])
    labels = seeds.set_index("row_id").loc[fit_ids, "outcome"].to_numpy(dtype=int)
    model.fit(x_train, labels, callbacks=[lgb.log_evaluation(0)])
    lgb_prediction = model.predict_proba(x_target)[:, 1]
    logistic_train, logistic_target = add_pca_features(train_numeric, target_numeric, fit_ids, target_ids, pca_train, pca_target)
    text_matrix = make_text_matrix(texts)
    logistic_prediction = fit_logistic(logistic_train, logistic_target, fit_ids, target_ids, labels, embeddings, text_matrix, selection["selected_c"], debug)
    prediction = selection["blend_weight"] * lgb_prediction + (1 - selection["blend_weight"]) * logistic_prediction
    lineage = {"fit_rows": int(len(fit_ids)), "target_rows": int(len(target_ids)), "count_oof_rows": int(len(count_train)), "count_inference_rows": int(len(count_target)), "design": selection["selected_design"], "rounds": int(120 if debug else selection["selected_rounds"]), "C": float(selection["selected_c"]), "lgb_weight": float(selection["blend_weight"]), "dense_features": len(names), "logistic_raw_embeddings": True, "logistic_pca_embeddings": True, "logistic_hashed_word_character_text": True}
    return np.clip(prediction, 1e-6, 1 - 1e-6), lineage


def diffusion_classifier(rounds: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=rounds,
        learning_rate=0.02,
        num_leaves=15,
        max_depth=5,
        min_child_samples=120,
        reg_alpha=1.0,
        reg_lambda=15.0,
        random_state=SEED + 101,
        n_jobs=THREADS,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def diffusion_columns(diffusion: pd.DataFrame, candidate: str) -> list[str]:
    if candidate == "base":
        return []
    pieces = candidate.split("_")
    variant = pieces[0]
    depth = int(pieces[1][1:])
    markers = [f"diff_{variant}_"]
    output = []
    for column in diffusion:
        if not any(marker in column for marker in markers):
            continue
        if f"_d{depth}_" in column or f"_d{depth}" in column:
            output.append(column)
        elif column.startswith("within_") and f"diff_{variant}_" in column and f"_d{depth}_" in column:
            output.append(column)
    return output


def candidate_numeric(structured: pd.DataFrame, neighbors: pd.DataFrame, diffusion: pd.DataFrame, candidate: str, pure: bool = False) -> pd.DataFrame:
    columns = diffusion_columns(diffusion, candidate)
    if pure:
        return finite_frame(diffusion[columns].copy())
    base = structured.join(neighbors)
    if columns:
        base = base.join(diffusion[columns])
    return finite_frame(base)


def route_dropout_matrix(matrix: np.ndarray, names: list[str], candidate: str, debug: bool) -> np.ndarray:
    if candidate == "base":
        return matrix
    rng = np.random.default_rng(SEED + 103)
    augmented = matrix.copy()
    for route in ["condition", "intervention", "sponsor", "facility"]:
        probability = 0.30 if route == "facility" else 0.10
        dropped = rng.random(len(matrix)) < probability
        columns = [index for index, name in enumerate(names) if f"_{route}_" in name and (name.startswith("diff_") or name.startswith("within_diff_"))]
        if not columns or not dropped.any():
            continue
        augmented[np.ix_(dropped, columns)] = 0.0
        success_columns = [index for index in columns if names[index].endswith("_success") or "_success_" in names[index]]
        cold_columns = [index for index in columns if "cold_start" in names[index]]
        if success_columns:
            augmented[np.ix_(dropped, success_columns)] = 0.5
        if cold_columns:
            augmented[np.ix_(dropped, cold_columns)] = 1.0
    return np.row_stack([matrix, augmented])


def fit_diffusion_fold(numeric: pd.DataFrame, categories: pd.DataFrame, train_ids: np.ndarray, valid_ids: np.ndarray, labels: np.ndarray, valid_labels: np.ndarray, candidate: str, debug: bool, pure: bool = False) -> tuple[np.ndarray, int]:
    used_categories = pd.DataFrame(index=categories.index) if pure else categories
    x_train, x_valid, names = dense_matrices(numeric, used_categories, train_ids, valid_ids)
    augmented = route_dropout_matrix(x_train, names, candidate, debug)
    augmented_labels = np.tile(labels, 2) if len(augmented) == 2 * len(labels) else labels
    maximum = 100 if debug else 1000
    model = diffusion_classifier(maximum)
    callbacks = [lgb.log_evaluation(0)]
    if not debug:
        callbacks.append(lgb.early_stopping(80, verbose=False))
    model.fit(augmented, augmented_labels, eval_set=[(x_valid, valid_labels)], eval_metric="auc", callbacks=callbacks)
    rounds = int(model.best_iteration_ or maximum)
    return model.predict_proba(x_valid, num_iteration=rounds)[:, 1], rounds


def diffusion_slices(rows: pd.DataFrame, labels: pd.Series, prediction: pd.Series, structured: pd.DataFrame) -> dict[str, float]:
    sites = structured.loc[rows["row_id"], "rel_site_count"].to_numpy(np.float64)
    output = {}
    for name, mask in [("sparse", sites <= 4), ("rich", sites >= 20)]:
        ids = rows.loc[mask, "row_id"].to_numpy(np.int64)
        if len(ids) and labels.loc[ids].nunique() == 2:
            output[name] = float(roc_auc_score(labels.loc[ids], prediction.loc[ids]))
    return output


def forward_diffusion_selection(seeds: pd.DataFrame, structured: pd.DataFrame, neighbors: pd.DataFrame, diffusion: pd.DataFrame, categories: pd.DataFrame, debug: bool) -> dict:
    train_rows = seeds[seeds["split"].eq("train")]
    labels = train_rows.set_index("row_id")["outcome"]
    years = [2019] if debug else [2016, 2017, 2018, 2019]
    depths = [1] if debug else [2, 3]
    candidates = ["base"] + [f"{variant}_d{depth}" for variant in (["loose"] if debug else ["loose", "strict"]) for depth in depths] + [f"pure_loose_d{depths[-1]}"]
    predictions = {candidate: pd.Series(np.nan, index=train_rows["row_id"], dtype=float) for candidate in candidates}
    scores = {candidate: [] for candidate in candidates}
    rounds = {candidate: [] for candidate in candidates}
    slices = {candidate: [] for candidate in candidates}
    for year in years:
        fit_ids = train_rows.loc[train_rows["timestamp"].dt.year.lt(year), "row_id"].to_numpy(np.int64)
        valid_ids = train_rows.loc[train_rows["timestamp"].dt.year.eq(year), "row_id"].to_numpy(np.int64)
        if debug and len(fit_ids) > 2000:
            fit_ids = fit_ids[np.linspace(0, len(fit_ids) - 1, 2000).astype(np.int64)]
        y_fit = labels.loc[fit_ids].to_numpy(np.int64)
        y_valid = labels.loc[valid_ids].to_numpy(np.int64)
        for candidate in candidates:
            pure = candidate.startswith("pure_")
            feature_candidate = candidate.removeprefix("pure_")
            numeric = candidate_numeric(structured, neighbors, diffusion, feature_candidate, pure=pure)
            prediction, used_rounds = fit_diffusion_fold(numeric, categories, fit_ids, valid_ids, y_fit, y_valid, feature_candidate, debug, pure=pure)
            score = float(roc_auc_score(y_valid, prediction))
            predictions[candidate].loc[valid_ids] = prediction
            scores[candidate].append(score)
            rounds[candidate].append(used_rounds)
            fold_rows = train_rows[train_rows["row_id"].isin(valid_ids)]
            fold_prediction = pd.Series(prediction, index=valid_ids)
            slices[candidate].append(diffusion_slices(fold_rows, labels, fold_prediction, structured))
            print(f"[diffusion-selection] year={year} candidate={candidate} auc={score:.6f} rounds={used_rounds}", flush=True)
    base_scores = np.asarray(scores["base"], dtype=np.float64)
    eligible = []
    for candidate in candidates[1:]:
        values = np.asarray(scores[candidate], dtype=np.float64)
        wins = int(np.sum(values > base_scores))
        sparse_gain = float(np.mean([item.get("sparse", 0.5) - base.get("sparse", 0.5) for item, base in zip(slices[candidate], slices["base"])]))
        rich_loss = float(np.mean([base.get("rich", 0.5) - item.get("rich", 0.5) for item, base in zip(slices[candidate], slices["base"])]))
        gate = wins >= (1 if debug else 3) or (wins >= 2 and sparse_gain >= 0.003 and rich_loss <= 0.005)
        if gate:
            eligible.append(candidate)
    selected = max(eligible, key=lambda value: np.mean(scores[value])) if eligible else "base"
    pooled_ids = np.concatenate([train_rows.loc[train_rows["timestamp"].dt.year.eq(year), "row_id"].to_numpy(np.int64) for year in years])
    pooled_labels = labels.loc[pooled_ids].to_numpy(np.int64)
    blend_scores = {}
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 1.0]
    if selected == "base":
        blend_weight = 1.0
        blend_scores[1.0] = float(roc_auc_score(pooled_labels, predictions["base"].loc[pooled_ids]))
    else:
        for weight in weights:
            prediction = weight * predictions["base"].loc[pooled_ids].to_numpy(np.float64) + (1.0 - weight) * predictions[selected].loc[pooled_ids].to_numpy(np.float64)
            blend_scores[weight] = float(roc_auc_score(pooled_labels, prediction))
        blend_weight = max(weights, key=lambda value: blend_scores[value])
    selected_rounds = int(np.median(rounds[selected]))
    result = {
        "candidates": {candidate: {"fold_scores": scores[candidate], "mean": float(np.mean(scores[candidate])), "rounds": rounds[candidate], "slices": slices[candidate]} for candidate in candidates},
        "selected": selected,
        "selected_rounds": selected_rounds,
        "base_blend_weight": float(blend_weight),
        "blend_scores": {str(key): value for key, value in blend_scores.items()},
        "fold_years": years,
        "route_dropout": {"facility": 0.30, "condition": 0.10, "intervention": 0.10, "sponsor": 0.10},
    }
    print(f"[diffusion-selection] frozen selected={selected} rounds={selected_rounds} base_weight={blend_weight}", flush=True)
    return result


def fit_diffusion_chain(seeds: pd.DataFrame, structured: pd.DataFrame, neighbors: pd.DataFrame, diffusion: pd.DataFrame, categories: pd.DataFrame, fit_ids: np.ndarray, target_ids: np.ndarray, base_prediction: np.ndarray, selection: dict, debug: bool) -> tuple[np.ndarray, dict]:
    selected = selection["selected"]
    if selected == "base" or selection["base_blend_weight"] >= 1.0:
        return np.asarray(base_prediction, dtype=np.float64), {"fit_rows": int(len(fit_ids)), "target_rows": int(len(target_ids)), "selected": "base", "base_blend_weight": 1.0, "route_dropout_applied": False}
    pure = selected.startswith("pure_")
    feature_candidate = selected.removeprefix("pure_")
    numeric = candidate_numeric(structured, neighbors, diffusion, feature_candidate, pure=pure)
    used_categories = pd.DataFrame(index=categories.index) if pure else categories
    x_fit, x_target, names = dense_matrices(numeric, used_categories, fit_ids, target_ids)
    augmented = route_dropout_matrix(x_fit, names, feature_candidate, debug)
    labels = seeds.set_index("row_id").loc[fit_ids, "outcome"].to_numpy(np.int64)
    augmented_labels = np.tile(labels, 2) if len(augmented) == 2 * len(labels) else labels
    rounds = 100 if debug else int(selection["selected_rounds"])
    model = diffusion_classifier(rounds)
    model.fit(augmented, augmented_labels, callbacks=[lgb.log_evaluation(0)])
    diffusion_prediction = model.predict_proba(x_target)[:, 1]
    weight = float(selection["base_blend_weight"])
    prediction = weight * np.asarray(base_prediction, dtype=np.float64) + (1.0 - weight) * diffusion_prediction
    lineage = {"fit_rows": int(len(fit_ids)), "target_rows": int(len(target_ids)), "selected": selected, "rounds": rounds, "base_blend_weight": weight, "route_dropout_applied": True, "numeric_features": len(names), "pure_diffusion": pure}
    return np.clip(prediction, 1e-6, 1 - 1e-6), lineage
