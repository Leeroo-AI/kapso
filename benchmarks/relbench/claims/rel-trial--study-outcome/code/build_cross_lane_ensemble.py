import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

import publication_pipeline as pipeline
from campaign_io import locked_append, register_artifact
from kapso_datasets.common import shared_cache_dir
from publication_evidence import adjudicate_candidates


def rank_blend(incumbent: np.ndarray, text: np.ndarray, weight: float) -> np.ndarray:
    count = len(incumbent)
    return (1.0 - weight) * rankdata(incumbent, method="average") / count + weight * rankdata(text, method="average") / count


def paired_bootstrap(labels: np.ndarray, incumbent: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(1337)
    deltas = []
    for _ in range(2000):
        indices = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) == 2:
            deltas.append(roc_auc_score(labels[indices], candidate[indices]) - roc_auc_score(labels[indices], incumbent[indices]))
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "draws": int(len(values)),
        "mean_delta": float(values.mean()),
        "standard_error": float(values.std(ddof=1)),
        "probability_positive": float((values > 0).mean()),
    }


def reconstruct_literature_gates(cache: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    artifacts = pipeline._load_artifacts(cache)
    pipeline.artifacts_global = artifacts
    registry_predictions, _ = pipeline._registry_predictions_and_ablations(artifacts, run_ablations=False)
    aligned = pipeline.build_aligned_stacker(artifacts, registry_predictions)
    records, candidates, contexts, retrieval = pipeline.retrieve_gate_origins(artifacts, cache)
    adjudications = {}
    hosted = {}
    for split in ["official_2018", "official_2019"]:
        adjudications[split], hosted[split] = adjudicate_candidates(candidates[split], contexts[split], cache, concurrency=32)
    gate = pipeline.fit_publication_gate(artifacts, aligned, registry_predictions, records, adjudications)
    result = {
        "candidate_2018": gate["candidate_2018"],
        "candidate_2019": gate["candidate_2019"],
        "labels_2018": aligned["labels_2018"],
        "labels_2019": aligned["labels_2019"],
        "indices_2018": np.flatnonzero(pd.to_datetime(artifacts["invariant"]["train_timestamp"]).to_numpy() == np.datetime64("2018-01-01")),
        "indices_2019": np.flatnonzero(pd.to_datetime(artifacts["invariant"]["train_timestamp"]).to_numpy() == np.datetime64("2019-01-01")),
    }
    return result, {"publication_gate": gate["diagnostics"], "retrieval": retrieval, "hosted": hosted}


def main() -> None:
    cache = shared_cache_dir()
    source = cache / "predictions" / "generic_exp_1_tfidf_v1.npz"
    if not source.exists():
        raise RuntimeError("Accepted cross-lane TF-IDF bank is unavailable")
    gates, reconstruction = reconstruct_literature_gates(cache)
    with np.load(source, allow_pickle=False) as tfidf:
        tfidf_diagnostics = json.loads(str(tfidf["diagnostics_json"][0]))
        if not bool(tfidf_diagnostics.get("forward_gate", False)):
            raise RuntimeError("Cross-lane TF-IDF bank did not pass its own forward gate")
        indices = tfidf["forward_index"].astype(np.int64)
        expected = np.concatenate([gates["indices_2018"], gates["indices_2019"]])
        if not np.array_equal(indices, expected):
            raise RuntimeError("Literature and TF-IDF forward rows are not aligned")
        labels = np.concatenate([gates["labels_2018"], gates["labels_2019"]])
        if not np.array_equal(labels.astype(np.int8), tfidf["forward_labels"].astype(np.int8)):
            raise RuntimeError("Literature and TF-IDF forward labels are not aligned")
        design_count = len(gates["labels_2018"])
        design_text = tfidf["forward_predictions"][:design_count].astype(np.float64)
        sealed_text = tfidf["forward_predictions"][design_count:].astype(np.float64)
        validation_text = tfidf["val"].astype(np.float64)
    grid = np.arange(0.0, 1.0001, 0.05)
    scores = {
        float(weight): float(roc_auc_score(gates["labels_2018"], rank_blend(gates["candidate_2018"], design_text, float(weight))))
        for weight in grid
    }
    selected = max(grid, key=lambda weight: (scores[float(weight)], -float(weight)))
    design_candidate = rank_blend(gates["candidate_2018"], design_text, float(selected))
    sealed_candidate = rank_blend(gates["candidate_2019"], sealed_text, float(selected))
    bootstrap = paired_bootstrap(gates["labels_2018"], gates["candidate_2018"], design_candidate)
    delta_2018 = float(roc_auc_score(gates["labels_2018"], design_candidate) - roc_auc_score(gates["labels_2018"], gates["candidate_2018"]))
    delta_2019 = float(roc_auc_score(gates["labels_2019"], sealed_candidate) - roc_auc_score(gates["labels_2019"], gates["candidate_2019"]))
    accepted = bool(selected > 0 and delta_2018 >= 0 and delta_2019 >= 0 and bootstrap["probability_positive"] >= 0.8)
    with np.load(cache / "predictions" / "generic_exp_0_literature_v3.npz", allow_pickle=False) as incumbent:
        validation_incumbent = incumbent["val"].astype(np.float64)
    validation = rank_blend(validation_incumbent, validation_text, float(selected)) if accepted else validation_incumbent.copy()
    freeze_root = cache / "cross_lane_ensemble_v1"
    freeze_root.mkdir(parents=True, exist_ok=True)
    freeze_path = freeze_root / "model_a_validation.npy"
    np.save(freeze_path, validation)
    validation_checksum = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    with np.load(source, allow_pickle=False) as tfidf, np.load(cache / "predictions" / "generic_exp_0_literature_v3.npz", allow_pickle=False) as incumbent:
        test = rank_blend(incumbent["test"].astype(np.float64), tfidf["test"].astype(np.float64), float(selected)) if accepted else incumbent["test"].astype(np.float64)
    if hashlib.sha256(freeze_path.read_bytes()).hexdigest() != validation_checksum:
        raise RuntimeError("Cross-lane Model A checksum changed while constructing Model B")
    diagnostics = {
        "accepted": accepted,
        "selected_tfidf_weight": float(selected),
        "design_delta": delta_2018,
        "sealed_delta": delta_2019,
        "design_bootstrap": bootstrap,
        "design_rank_correlation": float(spearmanr(gates["candidate_2018"], design_text).statistic),
        "sealed_rank_correlation": float(spearmanr(gates["candidate_2019"], sealed_text).statistic),
        "validation_checksum": validation_checksum,
        "weight_scores_2018": {str(key): value for key, value in scores.items()},
        "source_diagnostics": tfidf_diagnostics,
        "reconstruction": reconstruction,
    }
    destination = cache / "predictions" / "generic_exp_0_literature_tfidf_ensemble_v1.npz"
    temporary = destination.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, val=validation, test=test, diagnostics_json=np.asarray([json.dumps(diagnostics, sort_keys=True, allow_nan=True)]))
    os.replace(temporary, destination)
    register_artifact(cache, {
        "name": "generic_exp_0 literature plus cross-lane TF-IDF ensemble",
        "path": "predictions/generic_exp_0_literature_tfidf_ensemble_v1.npz",
        "description": "Forward-OOF-selected rank blend of the accepted literature-v3 candidate and the decorrelated accepted supervised TF-IDF text channel.",
        "content_key": "rel-trial-study-outcome:generic-exp-0:literature-tfidf-ensemble-v1",
        "rebuild_hint": "Run build_cross_lane_ensemble.py after both namespaced candidate banks are available.",
    })
    locked_append(cache / "features_history.md", f'''\n### Cross-branch literature plus supervised TF-IDF rank ensemble\n- run/experiment: generic_exp_0 lane 0 | status: TESTED-{"KEPT" if accepted else "REJECTED"}\n- what: step-0.05 rank blend between literature-v3 and the accepted generic_exp_1 supervised TF-IDF finalist; weight selected only on 2018 and checked once on sealed 2019.\n- outcome: weight {float(selected):.2f}; design delta {delta_2018:+.9f}; sealed delta {delta_2019:+.9f}; design bootstrap {json.dumps(bootstrap, sort_keys=True)}; correlations {diagnostics["design_rank_correlation"]:.6f}/{diagnostics["sealed_rank_correlation"]:.6f}.\n- takeaway: candidate accepted only when both origin deltas are nonnegative and P(delta>0) is at least 0.8; validation labels are not used for selection.\n''')
    print(json.dumps(diagnostics, sort_keys=True))


if __name__ == "__main__":
    main()
