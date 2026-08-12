from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class AuxiliaryModels:
    q_model: object
    multiplicity_model: object
    feature_columns: list[str]
    dispersion: float


@dataclass
class MechanisticConfig:
    replay_weight: float
    candidate: str
    residual_c: float
    use_negative_binomial: bool


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "_row_id",
        "outcome",
        "K",
        "S",
        "is_replay",
        "origin_year",
        "encoder_probability",
        "eligibility_wordpieces",
    }
    return sorted(
        column
        for column in frame.select_dtypes(include=[np.number, "bool"]).columns
        if column not in excluded
    )


def _matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.reindex(columns=columns).astype(float).replace([np.inf, -np.inf], np.nan)


def _weights(frame: pd.DataFrame, replay_weight: float) -> np.ndarray:
    replay = frame["is_replay"].to_numpy(dtype=float)
    return np.where(replay > 0.5, replay_weight, 1.0)


def estimate_dispersion(
    values: np.ndarray,
    means: np.ndarray,
    weights: np.ndarray,
) -> float:
    values = np.asarray(values, dtype=float)
    means = np.clip(np.asarray(means, dtype=float), 1e-9, None)
    weights = np.asarray(weights, dtype=float)

    def objective(log_dispersion: float) -> float:
        dispersion = float(np.exp(log_dispersion))
        likelihood = (
            gammaln(values + dispersion)
            - gammaln(dispersion)
            - gammaln(values + 1.0)
            + dispersion * np.log(dispersion / (dispersion + means))
            + values * np.log(means / (dispersion + means))
        )
        return float(-np.sum(weights * likelihood) / np.maximum(weights.sum(), 1e-12))

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.1), np.log(1_000_000.0)),
        method="bounded",
        options={"xatol": 1e-4},
    )
    if not result.success:
        return 0.1
    return float(max(0.1, np.exp(result.x)))


def fit_auxiliary(
    frame: pd.DataFrame,
    feature_columns: list[str],
    replay_weight: float,
    seed: int,
) -> AuxiliaryModels:
    weights = _weights(frame, replay_weight)
    valid = (
        frame["K"].notna().to_numpy()
        & frame["S"].notna().to_numpy()
        & frame["K"].gt(0).to_numpy()
        & (weights > 0.0)
    )
    training = frame.loc[valid]
    row_weights = weights[valid]
    matrix = _matrix(training, feature_columns)
    q_target = training["S"].to_numpy(dtype=float) / training["K"].to_numpy(dtype=float)
    q_weights = row_weights * np.minimum(training["K"].to_numpy(dtype=float), 10.0)
    multiplicity_target = np.maximum(training["K"].to_numpy(dtype=float) - 1.0, 0.0)
    q_model = lgb.LGBMRegressor(
        objective="cross_entropy",
        num_leaves=15,
        min_child_samples=100,
        learning_rate=0.03,
        n_estimators=600,
        reg_lambda=2.0,
        reg_alpha=0.1,
        random_state=seed,
        n_jobs=11,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    multiplicity_model = lgb.LGBMRegressor(
        objective="poisson",
        num_leaves=15,
        min_child_samples=100,
        learning_rate=0.04,
        n_estimators=500,
        reg_lambda=2.0,
        reg_alpha=0.1,
        random_state=seed + 1,
        n_jobs=11,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q_model.fit(matrix, q_target, sample_weight=q_weights)
        multiplicity_model.fit(matrix, multiplicity_target, sample_weight=row_weights)
    fitted_mu = np.clip(
        np.asarray(multiplicity_model.predict(matrix), dtype=float), 0.0, 1000.0
    )
    dispersion = estimate_dispersion(multiplicity_target, fitted_mu, row_weights)
    return AuxiliaryModels(q_model, multiplicity_model, feature_columns, dispersion)


def mechanistic_block(models: AuxiliaryModels, frame: pd.DataFrame) -> pd.DataFrame:
    matrix = _matrix(frame, models.feature_columns)
    q = np.clip(np.asarray(models.q_model.predict(matrix), dtype=float), 1e-5, 1.0 - 1e-5)
    mu = np.clip(
        np.asarray(models.multiplicity_model.predict(matrix), dtype=float), 0.0, 1000.0
    )
    r = float(models.dispersion)
    negative_binomial = 1.0 - (1.0 - q) * np.power(r / (r + mu * q), r)
    poisson = 1.0 - (1.0 - q) * np.exp(-mu * q)
    simple = 1.0 - np.power(1.0 - q, 1.0 + mu)
    output = frame.copy()
    output["mechanistic_q"] = q
    output["mechanistic_log1p_mu"] = np.log1p(mu)
    output["mechanistic_interaction"] = q * np.log1p(mu)
    output["mechanistic_negative_binomial"] = np.clip(negative_binomial, 1e-5, 1.0 - 1e-5)
    output["mechanistic_poisson"] = np.clip(poisson, 1e-5, 1.0 - 1e-5)
    output["mechanistic_simple"] = np.clip(simple, 1e-5, 1.0 - 1e-5)
    for mode in ["negative_binomial", "poisson"]:
        probability = output[f"mechanistic_{mode}"].to_numpy(dtype=float)
        output[f"mechanistic_logit_{mode}"] = np.log(probability / (1.0 - probability))
    return output


def poisson_deviance(target: np.ndarray, mean: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    mean = np.clip(np.asarray(mean, dtype=float), 1e-9, None)
    terms = np.where(target > 0.0, target * np.log(target / mean) - target + mean, mean)
    return float(2.0 * np.mean(terms))


def negative_binomial_deviance(
    target: np.ndarray,
    mean: np.ndarray,
    dispersion: float,
) -> float:
    target = np.asarray(target, dtype=float)
    mean = np.clip(np.asarray(mean, dtype=float), 1e-9, None)
    r = float(dispersion)
    first = np.where(target > 0.0, target * np.log(target / mean), 0.0)
    second = (target + r) * np.log((target + r) / (mean + r))
    return float(2.0 * np.mean(first - second))


def _head_columns(
    feature_columns: list[str],
    candidate: str,
    use_negative_binomial: bool,
) -> list[str]:
    mode = "negative_binomial" if use_negative_binomial else "poisson"
    if candidate == "base":
        return feature_columns
    if candidate == "residual":
        return [f"mechanistic_logit_{mode}", *feature_columns]
    if candidate == "augmented":
        return [
            *feature_columns,
            "mechanistic_q",
            "mechanistic_log1p_mu",
            "mechanistic_interaction",
            f"mechanistic_{mode}",
            "mechanistic_simple",
        ]
    return []


def fit_binary_head(
    frame: pd.DataFrame,
    feature_columns: list[str],
    candidate: str,
    use_negative_binomial: bool,
    residual_c: float,
    replay_weight: float,
    seed: int,
):
    if candidate in {"pure", "simple"}:
        return None
    columns = _head_columns(feature_columns, candidate, use_negative_binomial)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=residual_c,
                    solver="lbfgs",
                    max_iter=1500,
                    random_state=seed,
                ),
            ),
        ]
    )
    weights = _weights(frame, replay_weight)
    valid = frame["outcome"].notna().to_numpy() & (weights > 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            _matrix(frame.loc[valid], columns),
            frame.loc[valid, "outcome"].to_numpy(dtype=float),
            model__sample_weight=weights[valid],
        )
    return model


def predict_candidate(
    model,
    frame: pd.DataFrame,
    feature_columns: list[str],
    candidate: str,
    use_negative_binomial: bool,
) -> np.ndarray:
    mode = "negative_binomial" if use_negative_binomial else "poisson"
    if candidate == "pure":
        return frame[f"mechanistic_{mode}"].to_numpy(dtype=float)
    if candidate == "simple":
        return frame["mechanistic_simple"].to_numpy(dtype=float)
    columns = _head_columns(feature_columns, candidate, use_negative_binomial)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(model.predict_proba(_matrix(frame, columns))[:, 1], dtype=float)


def _auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, predictions))


def _positive_probability(
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    seed: int,
    draws: int = 500,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        indices = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) < 2:
            continue
        differences.append(
            roc_auc_score(labels[indices], candidate[indices])
            - roc_auc_score(labels[indices], reference[indices])
        )
    values = np.asarray(differences, dtype=float)
    improvement = _auc(labels, candidate) - _auc(labels, reference)
    return improvement, float(values.std(ddof=1)), float(np.mean(values > 0.0))


def select_mechanistic(
    official: pd.DataFrame,
    replay: pd.DataFrame,
    seed: int = 1337,
) -> tuple[MechanisticConfig, dict, pd.DataFrame]:
    feature_columns = numeric_feature_columns(pd.concat([official, replay], ignore_index=True))
    replay_weights = [0.0, 0.25, 0.50]
    residual_cs = [0.03, 0.1]
    records: dict[tuple, list[dict]] = {}
    multiplicity: dict[float, list[dict]] = {weight: [] for weight in replay_weights}
    for replay_weight in replay_weights:
        for year in range(2014, 2020):
            validation = official[official["origin_year"].eq(year)].copy()
            origin = pd.Timestamp(validation["timestamp"].min())
            train_official = official[
                official["timestamp"].add(pd.Timedelta(days=365)).le(origin)
            ]
            train_replay = replay[
                replay["timestamp"].add(pd.Timedelta(days=365)).le(origin)
            ]
            training = pd.concat([train_official, train_replay], ignore_index=True)
            auxiliary = fit_auxiliary(
                training, feature_columns, replay_weight, seed + year
            )
            training_block = mechanistic_block(auxiliary, training)
            validation_block = mechanistic_block(auxiliary, validation)
            labels = validation["outcome"].to_numpy(dtype=float)
            k_target = validation["K"].to_numpy(dtype=float) - 1.0
            mu = np.expm1(validation_block["mechanistic_log1p_mu"].to_numpy(dtype=float))
            nb_deviance = negative_binomial_deviance(k_target, mu, auxiliary.dispersion)
            pois_deviance = poisson_deviance(k_target, mu)
            k_rank = float(spearmanr(validation["K"], 1.0 + mu).statistic)
            if not np.isfinite(k_rank):
                k_rank = 0.0
            multiplicity[replay_weight].append(
                {
                    "year": year,
                    "dispersion": auxiliary.dispersion,
                    "negative_binomial_deviance": nb_deviance,
                    "poisson_deviance": pois_deviance,
                    "K_rank_correlation": k_rank,
                }
            )
            base_model = fit_binary_head(
                training_block,
                feature_columns,
                "base",
                True,
                0.1,
                replay_weight,
                seed + year,
            )
            configurations = [("base", 0.1, True)]
            for use_nb in [True, False]:
                configurations.append(("pure", 0.0, use_nb))
                for residual_c in residual_cs:
                    configurations.extend(
                        [
                            ("residual", residual_c, use_nb),
                            ("augmented", residual_c, use_nb),
                        ]
                    )
            configurations.append(("simple", 0.0, True))
            for candidate, residual_c, use_nb in configurations:
                if candidate == "base":
                    model = base_model
                else:
                    model = fit_binary_head(
                        training_block,
                        feature_columns,
                        candidate,
                        use_nb,
                        residual_c,
                        replay_weight,
                        seed + year,
                    )
                predictions = predict_candidate(
                    model,
                    validation_block,
                    feature_columns,
                    candidate,
                    use_nb,
                )
                sparse = validation["sponsor_history_count_raw"].le(20).to_numpy()
                emerging = validation["condition_history_count_raw"].le(0).to_numpy()
                record = {
                    "year": year,
                    "keys": validation["_key"].tolist(),
                    "labels": labels,
                    "predictions": predictions,
                    "auc": _auc(labels, predictions),
                    "sparse_sponsor_auc": _auc(labels[sparse], predictions[sparse]),
                    "emerging_condition_auc": _auc(labels[emerging], predictions[emerging]),
                }
                records.setdefault(
                    (replay_weight, candidate, residual_c, use_nb), []
                ).append(record)
    summaries = {}
    eligible_keys = []
    for key, fold_records in records.items():
        replay_weight, candidate, residual_c, use_nb = key
        if candidate not in {"base", "simple"}:
            deviance_rows = multiplicity[replay_weight]
            nb_better = np.mean(
                [row["negative_binomial_deviance"] for row in deviance_rows]
            ) < np.mean([row["poisson_deviance"] for row in deviance_rows])
            if bool(use_nb) != bool(nb_better):
                continue
        aucs = np.asarray([row["auc"] for row in fold_records], dtype=float)
        worst_two = float(np.mean(np.sort(aucs)[:2]))
        k_rank = float(
            np.mean([row["K_rank_correlation"] for row in multiplicity[replay_weight]])
        )
        chosen_deviance = float(
            np.mean(
                [
                    row[
                        "negative_binomial_deviance"
                        if use_nb
                        else "poisson_deviance"
                    ]
                    for row in multiplicity[replay_weight]
                ]
            )
        )
        selection_value = (
            0.75 * float(np.mean(aucs))
            + 0.25 * worst_two
            - 0.05 * float(np.std(aucs))
            + 0.005 * k_rank
            - 0.0005 * np.log1p(chosen_deviance)
        )
        name = f"w{replay_weight:.2f}_{candidate}_c{residual_c:.2f}_{'nb' if use_nb else 'poisson'}"
        summaries[name] = {
            "replay_weight": replay_weight,
            "candidate": candidate,
            "residual_c": residual_c,
            "use_negative_binomial": use_nb,
            "fold_auc": aucs.tolist(),
            "mean_auc": float(np.mean(aucs)),
            "worst_two_origin_auc": worst_two,
            "standard_deviation": float(np.std(aucs)),
            "K_rank_correlation": k_rank,
            "multiplicity_deviance": chosen_deviance,
            "selection_value": selection_value,
        }
        if candidate != "base":
            eligible_keys.append((selection_value, name, key))
    _, selected_name, selected_key = max(eligible_keys)
    selected_records = records[selected_key]
    replay_weight, candidate, residual_c, use_nb = selected_key
    base_key = (replay_weight, "base", 0.1, True)
    base_records = records[base_key]
    labels = np.concatenate([row["labels"] for row in selected_records])
    predictions = np.concatenate([row["predictions"] for row in selected_records])
    base_predictions = np.concatenate([row["predictions"] for row in base_records])
    improvement, paired_se, probability_positive = _positive_probability(
        labels, predictions, base_predictions, seed
    )
    origin_improvements = [
        selected["auc"] - base["auc"]
        for selected, base in zip(selected_records, base_records)
    ]
    sparse_deltas = [
        selected["sparse_sponsor_auc"] - base["sparse_sponsor_auc"]
        for selected, base in zip(selected_records, base_records)
        if np.isfinite(selected["sparse_sponsor_auc"])
        and np.isfinite(base["sparse_sponsor_auc"])
    ]
    emerging_deltas = [
        selected["emerging_condition_auc"] - base["emerging_condition_auc"]
        for selected, base in zip(selected_records, base_records)
        if np.isfinite(selected["emerging_condition_auc"])
        and np.isfinite(base["emerging_condition_auc"])
    ]
    gate = {
        "improvement": improvement,
        "paired_bootstrap_se": paired_se,
        "probability_positive": probability_positive,
        "improved_origins": int(np.sum(np.asarray(origin_improvements) > 0.0)),
        "origin_improvements": origin_improvements,
        "sparse_sponsor_mean_delta": float(np.mean(sparse_deltas)) if sparse_deltas else 0.0,
        "emerging_condition_mean_delta": float(np.mean(emerging_deltas)) if emerging_deltas else 0.0,
    }
    gate["passed"] = bool(
        probability_positive >= 0.80
        and gate["improved_origins"] >= 4
        and gate["sparse_sponsor_mean_delta"] >= -0.02
        and gate["emerging_condition_mean_delta"] >= -0.02
    )
    oof_rows = []
    for selected, base in zip(selected_records, base_records):
        for key_value, label, prediction, base_prediction in zip(
            selected["keys"],
            selected["labels"],
            selected["predictions"],
            base["predictions"],
        ):
            oof_rows.append(
                {
                    "_key": key_value,
                    "origin_year": selected["year"],
                    "outcome": float(label),
                    "mechanistic_prediction": float(prediction),
                    "base_prediction": float(base_prediction),
                }
            )
    config = MechanisticConfig(replay_weight, candidate, residual_c, use_nb)
    diagnostics = {
        "feature_count": len(feature_columns),
        "selected": selected_name,
        "candidates": summaries,
        "multiplicity": {str(key): value for key, value in multiplicity.items()},
        "standalone_gate": gate,
    }
    print(f"[mechanistic] selection={json.dumps(diagnostics, separators=(',', ':'))}")
    return config, diagnostics, pd.DataFrame(oof_rows)


def fit_final_auxiliary(
    frame: pd.DataFrame,
    config: MechanisticConfig,
    seed: int,
) -> AuxiliaryModels:
    columns = numeric_feature_columns(frame)
    return fit_auxiliary(frame, columns, config.replay_weight, seed)


def fit_final_binary(
    auxiliary: AuxiliaryModels,
    frame: pd.DataFrame,
    config: MechanisticConfig,
    seed: int,
):
    block = mechanistic_block(auxiliary, frame)
    model = fit_binary_head(
        block,
        auxiliary.feature_columns,
        config.candidate,
        config.use_negative_binomial,
        config.residual_c,
        config.replay_weight,
        seed,
    )
    return model


def predict_final(
    auxiliary: AuxiliaryModels,
    binary_model,
    frame: pd.DataFrame,
    config: MechanisticConfig,
) -> np.ndarray:
    block = mechanistic_block(auxiliary, frame)
    return predict_candidate(
        binary_model,
        block,
        auxiliary.feature_columns,
        config.candidate,
        config.use_negative_binomial,
    )
