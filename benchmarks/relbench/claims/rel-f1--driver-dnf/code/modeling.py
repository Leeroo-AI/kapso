from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_VERSION = "lane0_staged_compact_v2"


@dataclass
class BasePredictions:
    m1: np.ndarray
    m2: np.ndarray
    m3: np.ndarray
    m4: np.ndarray


def thread_count() -> int:
    return max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))


def clip(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-4, 1 - 1e-4)


def constant_probability(labels: np.ndarray) -> float:
    return float(np.clip(np.mean(labels) if len(labels) else 0.5, 1e-4, 1 - 1e-4))


def recency_weights(dates: pd.Series, cutoff: pd.Timestamp, half_life_years: float | None) -> np.ndarray:
    if half_life_years is None:
        return np.ones(len(dates), dtype=float)
    age_years = np.maximum(0, (pd.Timestamp(cutoff) - pd.to_datetime(dates)).dt.total_seconds().to_numpy() / (365.25 * 86400))
    return np.exp2(-age_years / half_life_years)


def lgb_parameters(debug: bool) -> dict:
    return {
        "n_estimators": 50 if debug else 400,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 50,
        "colsample_bytree": 0.75,
        "subsample": 0.8,
        "subsample_freq": 1,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
        "random_state": 1337,
        "n_jobs": thread_count(),
        "verbosity": -1,
    }


def select_half_life(x: pd.DataFrame, labels: np.ndarray, dates: pd.Series, origin: pd.Timestamp, cache_root: Path, debug: bool) -> tuple[float | None, dict]:
    mode = "debug" if debug else "full"
    cache_dir = cache_root / MODEL_VERSION / "recency"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{mode}_{pd.Timestamp(origin).year}.json"
    if path.exists():
        try:
            record = json.loads(path.read_text())
            if pd.Timestamp(record["source_cutoff"]) <= pd.Timestamp(origin):
                value = record["selected"]
                return (None if value == "unlimited" else float(value)), record
        except Exception:
            pass
    unique_dates = np.array(sorted(pd.to_datetime(dates).unique()))
    recent_start = pd.Timestamp(origin) - pd.DateOffset(years=10)
    unique_dates = unique_dates[unique_dates >= np.datetime64(recent_start)]
    blocks = [block for block in np.array_split(unique_dates, 5)[1:] if len(block)]
    candidates: list[float | None] = [4.0, 8.0, None]
    scores: dict[str, list[float]] = {"4": [], "8": [], "unlimited": []}
    parameters = lgb_parameters(True)
    parameters["n_estimators"] = 80 if debug else 120
    for block in blocks:
        validation_start = pd.Timestamp(block[0])
        validation_end = pd.Timestamp(block[-1])
        train_mask = pd.to_datetime(dates) < validation_start
        validation_mask = (pd.to_datetime(dates) >= validation_start) & (pd.to_datetime(dates) <= validation_end)
        if train_mask.sum() < 200 or validation_mask.sum() < 20 or np.unique(labels[validation_mask]).size < 2:
            continue
        for candidate in candidates:
            name = "unlimited" if candidate is None else str(int(candidate))
            model = lgb.LGBMClassifier(objective="binary", **parameters)
            weights = recency_weights(pd.to_datetime(dates)[train_mask], validation_start, candidate)
            model.fit(x.loc[train_mask], labels[train_mask], sample_weight=weights, callbacks=[lgb.log_evaluation(0)])
            prediction = model.predict_proba(x.loc[validation_mask])[:, 1]
            scores[name].append(float(roc_auc_score(labels[validation_mask], prediction)))
    summaries = {}
    for name, values in scores.items():
        summaries[name] = {"mean": float(np.mean(values)) if values else -1.0, "worst": float(np.min(values)) if values else -1.0, "folds": len(values)}
    ordered = sorted(summaries, key=lambda name: (summaries[name]["mean"], summaries[name]["worst"], {"4": 0, "8": 1, "unlimited": 2}[name]), reverse=True)
    selected_name = ordered[0] if summaries[ordered[0]]["folds"] else "8"
    record = {"model_version": MODEL_VERSION, "source_cutoff": pd.Timestamp(origin).isoformat(), "selected": selected_name, "scores": summaries}
    temporary = Path(str(path) + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record))
    os.replace(temporary, path)
    return (None if selected_name == "unlimited" else float(selected_name)), record


def fit_m1(x_event: pd.DataFrame, event_labels: np.ndarray, event_dates: pd.Series, x_prediction: pd.DataFrame, khat: np.ndarray, origin: pd.Timestamp, half_life: float | None, debug: bool) -> np.ndarray:
    if len(event_labels) == 0 or np.unique(event_labels).size < 2:
        finish_probability = np.full(len(x_prediction), np.mean(event_labels == 0) if len(event_labels) else 0.2)
    else:
        model = lgb.LGBMClassifier(objective="multiclass", num_class=3, **lgb_parameters(debug))
        weights = recency_weights(pd.to_datetime(event_dates), origin, half_life)
        model.fit(x_event, event_labels, sample_weight=weights, callbacks=[lgb.log_evaluation(0)])
        probabilities = model.predict_proba(x_prediction)
        finish_probability = probabilities[:, int(np.where(model.classes_ == 0)[0][0])] if 0 in model.classes_ else np.zeros(len(x_prediction))
    hazard = clip(1.0 - finish_probability)
    return clip(1.0 - np.power(1.0 - hazard, np.maximum(1.0, np.asarray(khat, dtype=float))))


def fit_m2(x_train: pd.DataFrame, labels: np.ndarray, dates: pd.Series, x_prediction: pd.DataFrame, origin: pd.Timestamp, half_life: float | None, debug: bool) -> np.ndarray:
    if len(labels) == 0 or np.unique(labels).size < 2:
        return np.full(len(x_prediction), constant_probability(labels))
    model = lgb.LGBMClassifier(objective="binary", **lgb_parameters(debug))
    weights = recency_weights(pd.to_datetime(dates), origin, half_life)
    model.fit(x_train, labels, sample_weight=weights, callbacks=[lgb.log_evaluation(0)])
    return clip(model.predict_proba(x_prediction)[:, 1])


def state_columns(columns: list[str]) -> list[str]:
    selected = [column for column in columns if column.startswith("state_") or column.startswith("standing_") or column.startswith("qualifying_")]
    selected += [column for column in columns if column in {"res_count", "res_inactivity_days", "res_dnf_ewma_slow", "res_dnf_ewma_fast", "res_dnf_5", "res_dnf_20", "constructor_dnf_20", "constructor_dnf_365d", "team_tenure_races", "new_team", "rookie", "driver_age", "calendar_khat"}]
    return sorted(set(selected))


def fit_m3(x_train: pd.DataFrame, labels: np.ndarray, dates: pd.Series, x_prediction: pd.DataFrame, origin: pd.Timestamp, half_life: float | None) -> np.ndarray:
    if len(labels) == 0 or np.unique(labels).size < 2:
        return np.full(len(x_prediction), constant_probability(labels))
    columns = state_columns(list(x_train.columns))
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=0.3, penalty="l2", max_iter=1000, solver="lbfgs")),
    ])
    weights = recency_weights(pd.to_datetime(dates), origin, half_life)
    pipeline.fit(x_train[columns], labels, logistic__sample_weight=weights)
    return clip(pipeline.predict_proba(x_prediction[columns])[:, 1])


def fit_m4(x_train: pd.DataFrame, labels: np.ndarray, dates: pd.Series, x_prediction: pd.DataFrame, debug: bool) -> np.ndarray:
    if len(labels) == 0 or np.unique(labels).size < 2:
        return np.full(len(x_prediction), constant_probability(labels))
    order = np.argsort(pd.to_datetime(dates).to_numpy(), kind="stable")
    ordered_dates = pd.to_datetime(dates).iloc[order]
    groups = ordered_dates.groupby(ordered_dates, sort=False).size().to_numpy(dtype=int)
    parameters = lgb_parameters(debug)
    parameters.update({"n_estimators": 50 if debug else 250, "min_child_samples": 40})
    model = lgb.LGBMRanker(objective="lambdarank", label_gain=[0, 1], **parameters)
    model.fit(x_train.iloc[order], labels[order], group=groups, callbacks=[lgb.log_evaluation(0)])
    return clip(expit(model.predict(x_prediction)))


def fit_all_bases(x_task: pd.DataFrame, task_labels: np.ndarray, task_dates: pd.Series, x_event: pd.DataFrame, event_labels: np.ndarray, event_dates: pd.Series, x_prediction: pd.DataFrame, prediction_features: pd.DataFrame, origin: pd.Timestamp, half_life: float | None, debug: bool, stage_b: bool) -> BasePredictions:
    khat = prediction_features["calendar_khat"].to_numpy(dtype=float)
    m1 = fit_m1(x_event, event_labels, event_dates, x_prediction, khat, origin, half_life, debug)
    m2 = fit_m2(x_task, task_labels, task_dates, x_prediction, origin, half_life, debug)
    if stage_b:
        m3 = fit_m3(x_task, task_labels, task_dates, x_prediction, origin, half_life)
        m4 = fit_m4(x_task, task_labels, task_dates, x_prediction, debug)
    else:
        m3 = m2.copy()
        m4 = m2.copy()
    return BasePredictions(m1=m1, m2=m2, m3=m3, m4=m4)


def base_frame(predictions: BasePredictions, features: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame({"m1": predictions.m1, "m2": predictions.m2, "m3": predictions.m3, "m4": predictions.m4})
    output["disagreement_std"] = output[["m1", "m2", "m3", "m4"]].std(axis=1)
    output["disagreement_range"] = output[["m1", "m2", "m3", "m4"]].max(axis=1) - output[["m1", "m2", "m3", "m4"]].min(axis=1)
    output["calendar_khat"] = features["calendar_khat"].to_numpy(dtype=float)
    output["rookie"] = features["rookie"].to_numpy(dtype=float)
    output["new_team"] = features["new_team"].to_numpy(dtype=float)
    return output


META_COLUMNS = ["m1", "m2", "m3", "m4", "disagreement_std", "disagreement_range", "calendar_khat", "rookie", "new_team"]


def fixed_blend(frame: pd.DataFrame) -> np.ndarray:
    return clip(0.4 * frame["m1"].to_numpy(dtype=float) + 0.6 * frame["m2"].to_numpy(dtype=float))


def meta_cross_validation(frame: pd.DataFrame) -> dict:
    origins = np.array(sorted(pd.to_datetime(frame["date"]).unique()))
    blocks = [block for block in np.array_split(origins, 5)[1:] if len(block)]
    meta_scores = []
    fixed_scores = []
    oof_parts = []
    for block in blocks:
        start = pd.Timestamp(block[0])
        end = pd.Timestamp(block[-1])
        train = frame[pd.to_datetime(frame["date"]) < start]
        validation = frame[(pd.to_datetime(frame["date"]) >= start) & (pd.to_datetime(frame["date"]) <= end)]
        if train["date"].nunique() < 4 or len(validation) < 10 or train["target"].nunique() < 2 or validation["target"].nunique() < 2:
            continue
        model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("logistic", LogisticRegression(C=0.3, penalty="l2", max_iter=1000, solver="lbfgs"))])
        model.fit(train[META_COLUMNS], train["target"].to_numpy(dtype=int))
        meta_prediction = model.predict_proba(validation[META_COLUMNS])[:, 1]
        fixed_prediction = fixed_blend(validation)
        meta_scores.append(float(roc_auc_score(validation["target"], meta_prediction)))
        fixed_scores.append(float(roc_auc_score(validation["target"], fixed_prediction)))
        piece = validation[["date", "target"]].copy()
        piece["meta"] = meta_prediction
        piece["fixed"] = fixed_prediction
        oof_parts.append(piece)
    bootstrap_delta = []
    if oof_parts:
        oof = pd.concat(oof_parts, ignore_index=True)
        origins = np.array(sorted(pd.to_datetime(oof["date"]).unique()))
        random = np.random.default_rng(1337)
        for _ in range(100):
            sampled = random.choice(origins, size=len(origins), replace=True)
            pieces = [oof[pd.to_datetime(oof["date"]) == pd.Timestamp(origin)] for origin in sampled]
            draw = pd.concat(pieces, ignore_index=True)
            if draw["target"].nunique() < 2:
                continue
            bootstrap_delta.append(float(roc_auc_score(draw["target"], draw["meta"]) - roc_auc_score(draw["target"], draw["fixed"])))
    enabled = bool(meta_scores and np.mean(meta_scores) > np.mean(fixed_scores) and np.min(meta_scores) >= np.min(fixed_scores) - 0.01 and (not bootstrap_delta or np.mean(bootstrap_delta) > 0))
    return {
        "enabled": enabled,
        "meta_mean": float(np.mean(meta_scores)) if meta_scores else np.nan,
        "fixed_mean": float(np.mean(fixed_scores)) if fixed_scores else np.nan,
        "meta_worst": float(np.min(meta_scores)) if meta_scores else np.nan,
        "fixed_worst": float(np.min(fixed_scores)) if fixed_scores else np.nan,
        "bootstrap_delta_mean": float(np.mean(bootstrap_delta)) if bootstrap_delta else np.nan,
        "bootstrap_delta_std": float(np.std(bootstrap_delta, ddof=1)) if len(bootstrap_delta) > 1 else np.nan,
        "folds": len(meta_scores),
    }


def adaptive_blend(prequential: pd.DataFrame, current: pd.DataFrame) -> tuple[np.ndarray, dict]:
    if len(prequential) == 0 or prequential["date"].nunique() < 8 or prequential["target"].nunique() < 2:
        return fixed_blend(current), {"enabled": False, "reason": "fewer_than_eight_usable_origins"}
    diagnostics = meta_cross_validation(prequential)
    if not diagnostics["enabled"]:
        diagnostics["reason"] = "forward_gate"
        return fixed_blend(current), diagnostics
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("logistic", LogisticRegression(C=0.3, penalty="l2", max_iter=1000, solver="lbfgs"))])
    model.fit(prequential[META_COLUMNS], prequential["target"].to_numpy(dtype=int))
    return clip(model.predict_proba(current[META_COLUMNS])[:, 1]), diagnostics
