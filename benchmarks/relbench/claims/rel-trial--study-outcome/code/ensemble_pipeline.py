from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


YEARS = (2016, 2017, 2018, 2019)
STACK_C = (0.03, 0.1, 0.3)
BANK_VERSION = "generic_exp_4_rank_ensemble_v1"
PROTOCOL = "study_outcome_temporal_auc_v2"


def fingerprint(frame: pd.DataFrame, labeled: bool) -> str:
    columns = ["nct_id", "timestamp"] + (["outcome"] if labeled else [])
    return hashlib.sha256(frame[columns].to_csv(index=False).encode()).hexdigest()


def percentile_ranks(predictions: np.ndarray, years: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(predictions, dtype=np.float64)
    for year in np.unique(years):
        selected = years == year
        for column in range(predictions.shape[1]):
            ranks[selected, column] = rankdata(predictions[selected, column], method="average") / selected.sum()
    return ranks


def simplex_grid() -> list[np.ndarray]:
    return [np.asarray((first, second, 10 - first - second), dtype=np.float64) / 10.0 for first in range(11) for second in range(11 - first)]


def mean_year_auc(labels: np.ndarray, prediction: np.ndarray, years: np.ndarray) -> float:
    return float(np.mean([roc_auc_score(labels[years == year], prediction[years == year]) for year in np.unique(years)]))


def select_simplex(ranks: np.ndarray, labels: np.ndarray, years: np.ndarray) -> np.ndarray:
    best = None
    for weights in simplex_grid():
        score = mean_year_auc(labels, ranks @ weights, years)
        key = (score, -float(np.sum((weights - 1.0 / 3.0) ** 2)))
        if best is None or key > best[0]:
            best = (key, weights)
    return best[1]


def rank_logits(ranks: np.ndarray) -> np.ndarray:
    return logit(np.clip(ranks, 0.005, 0.995))


def fit_nonnegative_stack(features: np.ndarray, labels: np.ndarray, c_value: float) -> tuple[np.ndarray, float]:
    count = len(labels)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights = parameters[:3]
        intercept = parameters[3]
        score = features @ weights + intercept
        probability = expit(score)
        loss = float(np.mean(np.logaddexp(0.0, score) - labels * score) + np.sum(weights ** 2) / (2.0 * c_value * count))
        residual = probability - labels
        gradient = np.concatenate([features.T @ residual / count + weights / (c_value * count), [float(np.mean(residual))]])
        return loss, gradient

    prevalence = np.clip(np.mean(labels), 1e-5, 1 - 1e-5)
    initial = np.asarray([0.2, 0.2, 0.2, logit(prevalence)], dtype=np.float64)
    result = minimize(objective, initial, method="L-BFGS-B", jac=True, bounds=[(0.0, None), (0.0, None), (0.0, None), (None, None)], options={"maxiter": 500})
    if not result.success:
        raise RuntimeError(f"stack optimization failed: {result.message}")
    return result.x[:3], float(result.x[3])


def select_stack_c(features: np.ndarray, labels: np.ndarray, years: np.ndarray) -> float:
    scores = {}
    unique = tuple(int(value) for value in np.unique(years))
    for c_value in STACK_C:
        fold_scores = []
        for year in unique:
            training = years != year
            weights, intercept = fit_nonnegative_stack(features[training], labels[training], c_value)
            fold_scores.append(roc_auc_score(labels[~training], expit(features[~training] @ weights + intercept)))
        scores[c_value] = float(np.mean(fold_scores))
    best = max(scores.values())
    return min(value for value in STACK_C if scores[value] >= best - 0.001)


def nested_predictions(ranks: np.ndarray, labels: np.ndarray, years: np.ndarray) -> dict:
    output = {"equal_rank": np.mean(ranks, axis=1), "simplex_rank": np.zeros(len(labels)), "logistic_stack": np.zeros(len(labels))}
    details = {"simplex_rank": {}, "logistic_stack": {}}
    logits = rank_logits(ranks)
    for year in YEARS:
        training = years != year
        weights = select_simplex(ranks[training], labels[training], years[training])
        output["simplex_rank"][~training] = ranks[~training] @ weights
        details["simplex_rank"][str(year)] = {"weights": weights.tolist()}
        c_value = select_stack_c(logits[training], labels[training], years[training])
        stack_weights, intercept = fit_nonnegative_stack(logits[training], labels[training], c_value)
        output["logistic_stack"][~training] = expit(logits[~training] @ stack_weights + intercept)
        details["logistic_stack"][str(year)] = {"c": c_value, "weights": stack_weights.tolist(), "intercept": intercept}
    return {"predictions": output, "details": details}


def sponsor_bootstrap(labels: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, years: np.ndarray, sponsors: np.ndarray) -> dict:
    generator = np.random.default_rng(1337)
    values = []
    for _ in range(200):
        differences = []
        for year in YEARS:
            positions = np.flatnonzero(years == year)
            groups = {}
            for position in positions:
                sponsor = sponsors[position]
                key = f"missing_{position}" if pd.isna(sponsor) else str(sponsor)
                groups.setdefault(key, []).append(position)
            keys = np.asarray(list(groups), dtype=object)
            sampled = generator.choice(keys, size=len(keys), replace=True)
            draw = np.concatenate([np.asarray(groups[key], dtype=np.int64) for key in sampled])
            if len(np.unique(labels[draw])) == 2:
                differences.append(roc_auc_score(labels[draw], candidate[draw]) - roc_auc_score(labels[draw], baseline[draw]))
        if differences:
            values.append(float(np.mean(differences)))
    return {"draws": len(values), "mean": float(np.mean(values)), "standard_error": float(np.std(values, ddof=1)), "lower_95": float(np.quantile(values, 0.025)), "upper_95": float(np.quantile(values, 0.975))}


def slice_differences(features: pd.DataFrame, labels: np.ndarray, candidate: np.ndarray, baseline: np.ndarray) -> dict:
    sites = features["site_count"].fillna(0).to_numpy()
    history = (features["condition_effective_n"].fillna(0) + features["sponsor_effective_n"].fillna(0)).to_numpy()
    definitions = {"low_site": sites <= np.median(sites), "sparse_history": history <= np.median(history)}
    result = {}
    for name, selected in definitions.items():
        result[name] = {
            "count": int(selected.sum()),
            "baseline_auc": float(roc_auc_score(labels[selected], baseline[selected])),
            "candidate_auc": float(roc_auc_score(labels[selected], candidate[selected])),
        }
        result[name]["gain"] = result[name]["candidate_auc"] - result[name]["baseline_auc"]
    return result


def apply_combiner(name: str, ranks: np.ndarray, parameters: dict) -> np.ndarray:
    if name == "equal_rank":
        return np.mean(ranks, axis=1)
    if name == "simplex_rank":
        return ranks @ np.asarray(parameters["weights"], dtype=np.float64)
    weights = np.asarray(parameters["weights"], dtype=np.float64)
    return expit(rank_logits(ranks) @ weights + float(parameters["intercept"]))


def save_bank(directory: Path, val_frame: pd.DataFrame, test_frame: pd.DataFrame, validation: np.ndarray, test: np.ndarray, diagnostics: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "val_predictions.npy", np.asarray(validation, dtype=np.float64))
    np.save(directory / "test_predictions.npy", np.asarray(test, dtype=np.float64))
    metadata = {"val_fingerprint": fingerprint(val_frame, True), "test_fingerprint": fingerprint(test_frame, False), "val_count": len(val_frame), "test_count": len(test_frame), "diagnostics": diagnostics}
    temporary = directory / "metadata.partial.json"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary, directory / "metadata.json")


def build_ensemble() -> dict:
    shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    common = shared / "generic_exp_4_common_oof_v1"
    manifest = json.loads((common / "manifest.json").read_text())
    task_root = Path(os.environ["RELBENCH_CACHE_DIR"]) / "rel-trial" / "tasks" / "study-outcome"
    train = pd.read_parquet(task_root / "train.parquet").reset_index(drop=True)
    val = pd.read_parquet(task_root / "val.parquet").reset_index(drop=True)
    test = pd.read_parquet(task_root / "test.parquet").reset_index(drop=True)
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    val["timestamp"] = pd.to_datetime(val["timestamp"])
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    ordered_indices = np.concatenate([np.asarray(manifest["folds"][str(year)]["indices"], dtype=np.int64) for year in YEARS])
    labels = train.loc[ordered_indices, "outcome"].to_numpy(dtype=np.int64)
    years = train.loc[ordered_indices, "timestamp"].dt.year.to_numpy(dtype=np.int64)
    raw = np.column_stack([np.load(common / f"family{family}_oof.npy", allow_pickle=False) for family in (1, 2, 3)])
    if raw.shape != (len(ordered_indices), 3):
        raise RuntimeError("common OOF shape mismatch")
    ranks = percentile_ranks(raw, years)
    nested = nested_predictions(ranks, labels, years)
    bank = pd.read_pickle(shared / "lane0_censored_all_tables_v5.pkl")
    structured = bank["features"].iloc[ordered_indices].reset_index(drop=True)
    candidates = {}
    for name, prediction in nested["predictions"].items():
        fold_auc = [float(roc_auc_score(labels[years == year], prediction[years == year])) for year in YEARS]
        baseline_auc = [float(roc_auc_score(labels[years == year], ranks[years == year, 0])) for year in YEARS]
        differences = np.asarray(fold_auc) - np.asarray(baseline_auc)
        paired_se = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
        slices = slice_differences(structured, labels, prediction, ranks[:, 0])
        bootstrap = sponsor_bootstrap(labels, prediction, ranks[:, 0], years, structured["sponsor_lead_id"].to_numpy())
        gate = bool(float(np.mean(differences)) > max(paired_se, 0.003) and int(np.sum(differences > 0)) >= 3 and differences[-1] >= 0 and min(value["gain"] for value in slices.values()) >= -0.005)
        candidates[name] = {"fold_auc": fold_auc, "baseline_auc": baseline_auc, "differences": differences.tolist(), "mean_auc": float(np.mean(fold_auc)), "mean_gain": float(np.mean(differences)), "paired_se": paired_se, "sponsor_bootstrap": bootstrap, "slices": slices, "gate": gate}
    eligible = [name for name, result in candidates.items() if result["gate"]]
    selected = max(eligible, key=lambda name: (candidates[name]["mean_auc"], name == "equal_rank")) if eligible else "family1"
    if selected == "family1":
        final_parameters = {}
        validation = np.load(common / "family1_val.npy", allow_pickle=False)
        test_prediction = np.load(common / "family1_test.npy", allow_pickle=False)
    else:
        all_val = np.column_stack([np.load(common / f"family{family}_val.npy", allow_pickle=False) for family in (1, 2, 3)])
        all_test = np.column_stack([np.load(common / f"family{family}_test.npy", allow_pickle=False) for family in (1, 2, 3)])
        val_ranks = percentile_ranks(all_val, np.full(len(val), 2020))
        test_ranks = percentile_ranks(all_test, np.full(len(test), 2021))
        if selected == "equal_rank":
            final_parameters = {}
        elif selected == "simplex_rank":
            final_parameters = {"weights": select_simplex(ranks, labels, years).tolist()}
        else:
            c_value = select_stack_c(rank_logits(ranks), labels, years)
            weights, intercept = fit_nonnegative_stack(rank_logits(ranks), labels, c_value)
            final_parameters = {"c": c_value, "weights": weights.tolist(), "intercept": intercept}
        validation = apply_combiner(selected, val_ranks, final_parameters)
        test_prediction = apply_combiner(selected, test_ranks, final_parameters)
    diagnostics = {"selected": selected, "parameters": final_parameters, "candidates": candidates, "nested": nested["details"], "family_rank_correlations": np.corrcoef(np.column_stack([rankdata(raw[:, column]) for column in range(3)]), rowvar=False).tolist(), "common_manifest": manifest["families"]}
    save_bank(shared / BANK_VERSION, val, test, validation, test_prediction, diagnostics)
    for year in YEARS:
        history = Path(os.environ["RELBENCH_WORK_DIR"]) / "evaluation_inputs" / PROTOCOL / str(year) / "rel-trial" / "tasks" / "study-outcome"
        historical_val = pd.read_parquet(history / "val.parquet").reset_index(drop=True)
        historical_test = pd.read_parquet(history / "test.parquet").reset_index(drop=True)
        historical_val["timestamp"] = pd.to_datetime(historical_val["timestamp"])
        historical_test["timestamp"] = pd.to_datetime(historical_test["timestamp"])
        fold_ids = np.asarray(manifest["folds"][str(year)]["ids"], dtype=np.int64)
        fold_prediction = nested["predictions"][selected] if selected != "family1" else ranks[:, 0]
        selected_fold = fold_prediction[years == year]
        historical_validation = pd.Series(selected_fold, index=fold_ids).loc[historical_val["nct_id"]].to_numpy()
        if year < 2019:
            next_ids = np.asarray(manifest["folds"][str(year + 1)]["ids"], dtype=np.int64)
            next_prediction = fold_prediction[years == year + 1]
            historical_test_prediction = pd.Series(next_prediction, index=next_ids).loc[historical_test["nct_id"]].to_numpy()
        else:
            historical_test_prediction = pd.Series(validation, index=val["nct_id"]).loc[historical_test["nct_id"]].to_numpy()
        window_shared = shared / ".evaluation_windows" / PROTOCOL / str(year) / BANK_VERSION
        save_bank(window_shared, historical_val, historical_test, historical_validation, historical_test_prediction, diagnostics)
    return diagnostics


def run(debug: bool) -> None:
    output = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "output_data_generic_exp_4"))
    output.mkdir(parents=True, exist_ok=True)
    shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    bank = shared / BANK_VERSION
    task_root = Path(os.environ["RELBENCH_CACHE_DIR"]) / "rel-trial" / "tasks" / "study-outcome"
    val = pd.read_parquet(task_root / "val.parquet").reset_index(drop=True)
    test = pd.read_parquet(task_root / "test.parquet").reset_index(drop=True)
    val["timestamp"] = pd.to_datetime(val["timestamp"])
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    metadata = json.loads((bank / "metadata.json").read_text())
    if metadata["val_fingerprint"] != fingerprint(val, True) or metadata["test_fingerprint"] != fingerprint(test, False):
        raise RuntimeError("ensemble bank row fingerprint mismatch")
    validation = np.load(bank / "val_predictions.npy", allow_pickle=False)
    test_prediction = np.load(bank / "test_predictions.npy", allow_pickle=False)
    if validation.shape != (len(val),) or test_prediction.shape != (len(test),):
        raise RuntimeError("ensemble bank shape mismatch")
    if not np.all(np.isfinite(validation)) or not np.all(np.isfinite(test_prediction)) or np.min(validation) < 0 or np.max(validation) > 1 or np.min(test_prediction) < 0 or np.max(test_prediction) > 1:
        raise RuntimeError("ensemble predictions violate probability contract")
    if debug:
        sample = percentile_ranks(np.column_stack([validation[: min(32, len(validation))]] * 3), np.zeros(min(32, len(validation)), dtype=int))
        if sample.shape[1] != 3:
            raise RuntimeError("debug rank exercise failed")
    np.save(output / "val_predictions.npy", validation.astype(np.float64))
    np.save(output / "test_predictions.npy", test_prediction.astype(np.float64))
    (output / "metrics.json").write_text(json.dumps(metadata["diagnostics"], indent=2) + "\n")
    print(f"[ensemble] selected={metadata['diagnostics']['selected']} val_shape={validation.shape} test_shape={test_prediction.shape} debug={debug}")
