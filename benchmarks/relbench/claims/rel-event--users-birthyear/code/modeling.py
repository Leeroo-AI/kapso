from __future__ import annotations

import os
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import nnls


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    denominator = float(np.square(y - y.mean()).sum())
    return 1.0 - float(np.square(y - pred).sum()) / denominator if denominator > 0 else 0.0


@dataclass
class FrameEncoder:
    categorical: list[str]
    levels: dict[str, dict[str, int]]
    columns: list[str]

    @classmethod
    def fit(cls, frame: pd.DataFrame, categorical: list[str]) -> "FrameEncoder":
        columns = [c for c in frame.columns if c != "user_id"]
        cats = [c for c in categorical if c in columns]
        levels = {}
        for col in cats:
            values = frame[col].fillna("__missing__").astype(str)
            levels[col] = {value: i for i, value in enumerate(pd.unique(values))}
        return cls(cats, levels, columns)

    def lightgbm(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.reindex(columns=self.columns).copy()
        for col in self.categorical:
            values = out[col].fillna("__missing__").astype(str)
            out[col] = values.map(self.levels[col]).fillna(-1).astype(np.int32).astype("category")
        for col in out.columns:
            if col not in self.categorical:
                out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)
        return out

    def catboost(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.reindex(columns=self.columns).copy()
        for col in self.categorical:
            out[col] = out[col].fillna("__missing__").astype(str)
        for col in out.columns:
            if col not in self.categorical:
                out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)
        return out


@dataclass
class DualModel:
    lightgbm_encoder: FrameEncoder
    catboost_encoder: FrameEncoder
    lightgbm_models: list[lgb.LGBMRegressor]
    catboost_models: list[CatBoostRegressor]

    def predict_components(self, frame: pd.DataFrame) -> np.ndarray:
        light_frame = self.lightgbm_encoder.lightgbm(frame)
        cat_frame = self.catboost_encoder.catboost(frame)
        light = np.mean([model.predict(light_frame) for model in self.lightgbm_models], axis=0)
        cat = np.mean([model.predict(cat_frame) for model in self.catboost_models], axis=0)
        return np.column_stack([light, cat])


def fit_dual_model(
    train_frame: pd.DataFrame,
    train_y: np.ndarray,
    categorical: list[str],
    iterations: tuple[int, int],
    seeds: list[int],
    eval_frame: pd.DataFrame | None = None,
    eval_y: np.ndarray | None = None,
    catboost_drop_prefixes: tuple[str, ...] = (),
) -> tuple[DualModel, tuple[list[int], list[int]]]:
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    lightgbm_encoder = FrameEncoder.fit(train_frame, categorical)
    catboost_columns = [column for column in train_frame.columns if not column.startswith(catboost_drop_prefixes)]
    catboost_source = train_frame[catboost_columns]
    catboost_encoder = FrameEncoder.fit(catboost_source, categorical)
    light_train = lightgbm_encoder.lightgbm(train_frame)
    cat_train = catboost_encoder.catboost(catboost_source)
    light_eval = lightgbm_encoder.lightgbm(eval_frame) if eval_frame is not None else None
    cat_eval = catboost_encoder.catboost(eval_frame) if eval_frame is not None else None
    light_models = []
    cat_models = []
    light_best = []
    cat_best = []
    for seed in seeds:
        light = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=int(iterations[0]),
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=40,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=threads,
            verbosity=-1,
        )
        fit_args = {}
        if light_eval is not None and eval_y is not None:
            fit_args = {"eval_set": [(light_eval, eval_y)], "callbacks": [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]}
        light.fit(light_train, train_y, categorical_feature=lightgbm_encoder.categorical, **fit_args)
        light_models.append(light)
        light_best.append(int(light.best_iteration_ or iterations[0]))
        cat = CatBoostRegressor(
            iterations=int(iterations[1]),
            depth=7,
            learning_rate=0.035,
            l2_leaf_reg=12,
            loss_function="RMSE",
            random_seed=seed,
            task_type="GPU",
            devices="0",
            thread_count=threads,
            verbose=False,
            allow_writing_files=False,
            od_type="Iter",
            od_wait=100,
            random_strength=0.5,
        )
        cat_args = {}
        if cat_eval is not None and eval_y is not None:
            cat_args = {"eval_set": (cat_eval, eval_y), "use_best_model": True}
        cat.fit(cat_train, train_y, cat_features=catboost_encoder.categorical, verbose=False, **cat_args)
        cat_models.append(cat)
        best = cat.get_best_iteration()
        cat_best.append(int(best + 1 if best is not None and best >= 0 else iterations[1]))
    return DualModel(lightgbm_encoder, catboost_encoder, light_models, cat_models), (light_best, cat_best)


def choose_ensemble(oof_components: np.ndarray, y: np.ndarray, fold_ids: np.ndarray) -> tuple[np.ndarray, dict]:
    valid = np.isfinite(oof_components).all(axis=1) & np.isfinite(y)
    prediction = oof_components[valid]
    target = y[valid]
    if len(target) == 0:
        return np.asarray([0.5, 0.5]), {"reason": "no_oof"}
    prior = float(target.mean())
    design = prediction - prior
    ridge = 1e-3 * max(1.0, float(np.square(design).mean()))
    augmented_x = np.vstack([design, np.sqrt(ridge) * np.eye(2)])
    augmented_y = np.concatenate([target - prior, np.zeros(2)])
    weights, _ = nnls(augmented_x, augmented_y)
    weights = weights / weights.sum() if weights.sum() > 0 else np.asarray([0.5, 0.5])
    error_delta = np.square(target - prediction[:, 0]) - np.square(target - prediction[:, 1])
    variance = float(np.var(target))
    r2_difference = abs(float(error_delta.mean()) / variance) if variance > 0 else 0.0
    standard_error = float(error_delta.std(ddof=1) / np.sqrt(len(error_delta)) / variance) if len(error_delta) > 1 and variance > 0 else np.inf
    tied = r2_difference <= 2.0 * standard_error
    if tied:
        weights = np.asarray([0.5, 0.5])
    fold_scores = {}
    active_folds = fold_ids[valid]
    for fold in np.unique(active_folds):
        mask = active_folds == fold
        fold_scores[str(int(fold))] = {
            "count": int(mask.sum()),
            "lightgbm_r2": r2_score(target[mask], prediction[mask, 0]),
            "catboost_r2": r2_score(target[mask], prediction[mask, 1]),
        }
    return weights, {"weights": weights.tolist(), "tied": bool(tied), "r2_difference": r2_difference, "two_se": 2.0 * standard_error, "folds": fold_scores}


def shallow_residual_fit(features: np.ndarray, residual: np.ndarray, iterations: int = 180) -> lgb.LGBMRegressor:
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=iterations,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=4,
        min_child_samples=60,
        colsample_bytree=0.8,
        reg_lambda=15.0,
        random_state=17,
        n_jobs=threads,
        verbosity=-1,
    )
    model.fit(features, residual, callbacks=[lgb.log_evaluation(0)])
    return model


def improvement_gate(y: np.ndarray, before: np.ndarray, after: np.ndarray, fold_ids: np.ndarray) -> tuple[bool, dict]:
    valid = np.isfinite(y) & np.isfinite(before) & np.isfinite(after)
    yv = y[valid]
    old = before[valid]
    new = after[valid]
    folds = fold_ids[valid]
    variance = float(np.var(yv))
    delta = np.square(yv - old) - np.square(yv - new)
    mean_improvement = float(delta.mean() / variance) if variance > 0 and len(delta) else 0.0
    standard_error = float(delta.std(ddof=1) / np.sqrt(len(delta)) / variance) if variance > 0 and len(delta) > 1 else np.inf
    fold_deltas = {}
    for fold in np.unique(folds):
        mask = folds == fold
        fold_deltas[str(int(fold))] = r2_score(yv[mask], new[mask]) - r2_score(yv[mask], old[mask])
    latest = fold_deltas[str(int(np.max(folds)))] if len(folds) else -np.inf
    accepted = mean_improvement > standard_error and latest > -max(0.005, standard_error)
    return bool(accepted), {"mean_r2_improvement": mean_improvement, "pooled_se": standard_error, "latest_fold_improvement": latest, "fold_improvements": fold_deltas, "accepted": bool(accepted)}
