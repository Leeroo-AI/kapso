from __future__ import annotations

import math
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


META_COLUMNS = {"_global_id", "_split", "_row_id", "nct_id", "timestamp", "outcome"}
LOW_CARDINALITY = ["study_study_type", "study_phase", "study_enrollment_type", "study_source_class", "study_plan_to_share_ipd", "study_biospec_retention", "design_allocation", "design_intervention_model", "design_observational_model", "design_primary_purpose", "design_time_perspective", "design_masking", "elig_sampling_method", "elig_gender", "elig_healthy_volunteers", "elig_adult", "elig_child", "elig_older_adult", "sponsor_lead_class"]


@dataclass
class CVResult:
    name: str
    predictions: np.ndarray
    fold_scores: list[float]
    fold_best_iterations: list[int]
    fold_indices: list[np.ndarray]

    @property
    def mean(self) -> float:
        return float(np.mean(self.fold_scores))

    @property
    def uncertainty(self) -> float:
        if len(self.fold_scores) < 2:
            return 0.0
        return float(np.std(self.fold_scores, ddof=1) / np.sqrt(len(self.fold_scores)))


def _vectors(rows: pd.DataFrame, mapping: dict[int, np.ndarray]) -> np.ndarray:
    return np.stack([mapping.get(int(nct_id), np.zeros(768, dtype=np.float32)) for nct_id in rows["nct_id"]]).astype(np.float32)


def with_fold_pca(frame: pd.DataFrame, protocol: dict[int, np.ndarray], eligibility: dict[int, np.ndarray], fit_ids: set[int]) -> pd.DataFrame:
    result = frame.copy()
    fit_mask = result["_global_id"].isin(fit_ids).to_numpy()
    protocol_matrix = _vectors(result, protocol)
    eligibility_matrix = _vectors(result, eligibility)
    configurations = [("protocol_pc", protocol_matrix, 48), ("eligibility_pc", eligibility_matrix, 32)]
    for prefix, matrix, components in configurations:
        usable = min(components, int(np.sum(fit_mask)) - 1, matrix.shape[1])
        if usable < 1:
            for index in range(components):
                result[f"{prefix}_{index:02d}"] = 0.0
            continue
        transformer = PCA(n_components=usable, svd_solver="randomized", random_state=1337)
        transformer.fit(matrix[fit_mask])
        transformed = transformer.transform(matrix)
        for index in range(components):
            result[f"{prefix}_{index:02d}"] = transformed[:, index] if index < usable else 0.0
    return result


def _lgb_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    matrix = frame.drop(columns=[column for column in META_COLUMNS if column in frame], errors="ignore").copy()
    categorical = []
    for column in matrix.columns:
        if matrix[column].dtype == object or isinstance(matrix[column].dtype, pd.StringDtype):
            matrix[column] = matrix[column].fillna("__MISSING__").astype("category")
            categorical.append(column)
        else:
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return matrix, categorical


def _lgb_params(debug: bool) -> dict:
    return {"objective": "binary", "metric": "auc", "num_leaves": 31, "max_depth": 5, "min_data_in_leaf": 80, "learning_rate": 0.03, "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbosity": -1, "seed": 1337, "feature_fraction_seed": 1337, "bagging_seed": 1337, "num_threads": 11, "force_col_wise": True}


def forward_folds(rows: pd.DataFrame, debug: bool) -> list[tuple[np.ndarray, np.ndarray, int]]:
    labeled = rows["_split"] == "train"
    years = [2019] if debug else [2016, 2017, 2018, 2019]
    folds = []
    for year in years:
        evaluation = np.flatnonzero((labeled & (rows["timestamp"].dt.year == year)).to_numpy())
        cutoff = rows.iloc[evaluation]["timestamp"].min()
        training = np.flatnonzero((labeled & ((rows["timestamp"] + pd.Timedelta(days=365)) <= cutoff)).to_numpy())
        if debug and len(training) > 2000:
            training = training[-2000:]
        if debug and len(evaluation) > 500:
            evaluation = evaluation[:500]
        folds.append((training, evaluation, year))
    return folds


def lgb_cross_validate(name: str, frame: pd.DataFrame, rows: pd.DataFrame, protocol: dict[int, np.ndarray] | None, eligibility: dict[int, np.ndarray] | None, debug: bool) -> CVResult:
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    scores = []
    iterations = []
    indices = []
    for training, evaluation, year in forward_folds(rows, debug):
        selected = np.concatenate([training, evaluation])
        fold = frame.iloc[selected].copy().reset_index(drop=True)
        training_ids = set(rows.iloc[training]["_global_id"].astype(int))
        if protocol is not None and eligibility is not None:
            fold = with_fold_pca(fold, protocol, eligibility, training_ids)
        matrix, categorical = _lgb_matrix(fold)
        train_length = len(training)
        train_set = lgb.Dataset(matrix.iloc[:train_length], label=rows.iloc[training]["outcome"].to_numpy(), categorical_feature=categorical, free_raw_data=False)
        eval_set = lgb.Dataset(matrix.iloc[train_length:], label=rows.iloc[evaluation]["outcome"].to_numpy(), categorical_feature=categorical, reference=train_set, free_raw_data=False)
        maximum = 100 if debug else 2000
        callbacks = [lgb.early_stopping(20 if debug else 100, verbose=False), lgb.log_evaluation(0)]
        model = lgb.train(_lgb_params(debug), train_set, num_boost_round=maximum, valid_sets=[eval_set], callbacks=callbacks)
        prediction = model.predict(matrix.iloc[train_length:], num_iteration=model.best_iteration)
        predictions[evaluation] = prediction
        score = float(roc_auc_score(rows.iloc[evaluation]["outcome"], prediction))
        scores.append(score)
        iterations.append(int(model.best_iteration or maximum))
        indices.append(evaluation)
        print(f"[cv] {name} year={year} train={len(training)} eval={len(evaluation)} auc={score:.6f} trees={iterations[-1]}")
    return CVResult(name, predictions, scores, iterations, indices)


def _compact_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    available = [column for column in frame.columns if column not in META_COLUMNS]
    categorical = [column for column in LOW_CARDINALITY if column in available]
    numeric = [column for column in available if column not in categorical and frame[column].dtype != object and not isinstance(frame[column].dtype, pd.StringDtype)]
    numeric = [column for column in numeric if column != "sponsor_lead_id"]
    return numeric, categorical


def _logistic_pipeline(frame: pd.DataFrame, c_value: float) -> tuple[Pipeline, list[str]]:
    numeric, categorical = _compact_columns(frame)
    numerical_pipeline = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    categorical_pipeline = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))])
    transformer = ColumnTransformer([("numeric", numerical_pipeline, numeric), ("categorical", categorical_pipeline, categorical)], sparse_threshold=0.3)
    model = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=600, random_state=1337)
    return Pipeline([("transform", transformer), ("model", model)]), numeric + categorical


def logistic_cross_validate(name: str, frame: pd.DataFrame, rows: pd.DataFrame, protocol: dict[int, np.ndarray] | None, eligibility: dict[int, np.ndarray] | None, c_value: float, debug: bool) -> CVResult:
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    scores = []
    indices = []
    for training, evaluation, year in forward_folds(rows, debug):
        selected = np.concatenate([training, evaluation])
        fold = frame.iloc[selected].copy().reset_index(drop=True)
        training_ids = set(rows.iloc[training]["_global_id"].astype(int))
        if protocol is not None and eligibility is not None:
            fold = with_fold_pca(fold, protocol, eligibility, training_ids)
        pipeline, columns = _logistic_pipeline(fold, c_value)
        train_length = len(training)
        pipeline.fit(fold.iloc[:train_length][columns], rows.iloc[training]["outcome"].to_numpy())
        prediction = pipeline.predict_proba(fold.iloc[train_length:][columns])[:, 1]
        predictions[evaluation] = prediction
        score = float(roc_auc_score(rows.iloc[evaluation]["outcome"], prediction))
        scores.append(score)
        indices.append(evaluation)
        print(f"[cv] {name} year={year} train={len(training)} eval={len(evaluation)} auc={score:.6f}")
    return CVResult(name, predictions, scores, [], indices)


def compare_results(candidate: CVResult, baseline: CVResult) -> tuple[float, float, bool]:
    differences = np.asarray(candidate.fold_scores) - np.asarray(baseline.fold_scores)
    improvement = float(np.mean(differences))
    uncertainty = float(np.std(differences, ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else 0.0
    return improvement, uncertainty, bool(improvement > uncertainty and improvement > 0)


def blend_result(lgb_result: CVResult, logistic_result: CVResult, weight: float, rows: pd.DataFrame) -> CVResult:
    predictions = weight * lgb_result.predictions + (1 - weight) * logistic_result.predictions
    scores = []
    for indices in lgb_result.fold_indices:
        scores.append(float(roc_auc_score(rows.iloc[indices]["outcome"], predictions[indices])))
    return CVResult(f"blend_{weight:.2f}", predictions, scores, lgb_result.fold_best_iterations, lgb_result.fold_indices)


def _fit_frame(frame: pd.DataFrame, rows: pd.DataFrame, fit_indices: np.ndarray, predict_indices: np.ndarray, protocol: dict[int, np.ndarray] | None, eligibility: dict[int, np.ndarray] | None) -> tuple[pd.DataFrame, int]:
    selected = np.concatenate([fit_indices, predict_indices])
    combined = frame.iloc[selected].copy().reset_index(drop=True)
    if protocol is not None and eligibility is not None:
        fit_ids = set(rows.iloc[fit_indices]["_global_id"].astype(int))
        combined = with_fold_pca(combined, protocol, eligibility, fit_ids)
    return combined, len(fit_indices)


def fit_lgb_predict(frame: pd.DataFrame, rows: pd.DataFrame, fit_indices: np.ndarray, predict_indices: np.ndarray, trees: int, protocol: dict[int, np.ndarray] | None, eligibility: dict[int, np.ndarray] | None, debug: bool) -> np.ndarray:
    combined, train_length = _fit_frame(frame, rows, fit_indices, predict_indices, protocol, eligibility)
    matrix, categorical = _lgb_matrix(combined)
    train_set = lgb.Dataset(matrix.iloc[:train_length], label=rows.iloc[fit_indices]["outcome"].to_numpy(), categorical_feature=categorical, free_raw_data=False)
    model = lgb.train(_lgb_params(debug), train_set, num_boost_round=max(1, int(trees)), callbacks=[lgb.log_evaluation(0)])
    return model.predict(matrix.iloc[train_length:])


def fit_logistic_predict(frame: pd.DataFrame, rows: pd.DataFrame, fit_indices: np.ndarray, predict_indices: np.ndarray, c_value: float, protocol: dict[int, np.ndarray] | None, eligibility: dict[int, np.ndarray] | None) -> np.ndarray:
    combined, train_length = _fit_frame(frame, rows, fit_indices, predict_indices, protocol, eligibility)
    pipeline, columns = _logistic_pipeline(combined, c_value)
    pipeline.fit(combined.iloc[:train_length][columns], rows.iloc[fit_indices]["outcome"].to_numpy())
    return pipeline.predict_proba(combined.iloc[train_length:][columns])[:, 1]


def final_indices(rows: pd.DataFrame, debug: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.flatnonzero((rows["_split"] == "train").to_numpy())
    validation = np.flatnonzero((rows["_split"] == "val").to_numpy())
    test = np.flatnonzero((rows["_split"] == "test").to_numpy())
    if debug and len(train) > 2000:
        model_a = train[-2000:]
    else:
        model_a = train
    model_b = np.concatenate([model_a, validation])
    return model_a, model_b, validation, test
