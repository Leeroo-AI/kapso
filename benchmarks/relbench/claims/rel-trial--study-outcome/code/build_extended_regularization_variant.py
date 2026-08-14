import hashlib
import json
import os

import numpy as np

import publication_pipeline as pipeline
from build_binary_route_variant import freeze, persist
from build_cross_lane_ensemble import paired_bootstrap, rank_blend
from campaign_io import locked_append, register_artifact
from kapso_datasets.common import load_task, shared_cache_dir
from publication_evidence import adjudicate_candidates


def select_literature(features: pipeline.pd.DataFrame, labels: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray, dict[str, float]]:
    values = [0.003, 0.01, 0.03, 0.1, 0.3]
    predictions = {}
    scores = {}
    for _ in range(4):
        for value in values:
            if value not in predictions:
                predictions[value] = pipeline._crossfit_literature(features, labels, value)
                scores[str(value)] = pipeline._auc(labels, predictions[value], mask)
        selected = max(values, key=lambda value: scores[str(value)])
        if selected == values[0] and selected > 0.00003:
            values = [selected / 3.0] + values
            continue
        if selected == values[-1] and selected < 30.0:
            values = values + [selected * 3.0]
            continue
        break
    return float(selected), predictions[selected], scores


def select_route(matrix: np.ndarray, labels: np.ndarray, covered: np.ndarray, incumbent: np.ndarray) -> tuple[float, np.ndarray, dict[str, float]]:
    values = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    predictions = {}
    scores = {}
    strength = np.ones(len(labels), dtype=np.float64)
    for _ in range(4):
        for value in values:
            if value not in predictions:
                predictions[value] = pipeline._crossfit_route(matrix, labels, covered, incumbent, strength, value)
                scores[str(value)] = pipeline._auc(labels, predictions[value])
        selected = max(values, key=lambda value: scores[str(value)])
        if selected == values[0] and selected > 0.0003:
            values = [selected / 3.0] + values
            continue
        if selected == values[-1] and selected < 300.0:
            values = values + [selected * 3.0]
            continue
        break
    return float(selected), predictions[selected], scores


def main() -> None:
    cache = shared_cache_dir()
    artifacts = pipeline._load_artifacts(cache)
    pipeline.artifacts_global = artifacts
    registry_predictions, _ = pipeline._registry_predictions_and_ablations(artifacts, run_ablations=False)
    aligned = pipeline.build_aligned_stacker(artifacts, registry_predictions)
    records, candidates, contexts, retrieval = pipeline.retrieve_gate_origins(artifacts, cache)
    adjudications = {}
    hosted = {}
    for split in ["official_2018", "official_2019"]:
        adjudications[split], hosted[split] = adjudicate_candidates(candidates[split], contexts[split], cache, concurrency=32)
    base = pipeline.fit_publication_gate(artifacts, aligned, registry_predictions, records, adjudications)
    labels_2018 = aligned["labels_2018"]
    labels_2019 = aligned["labels_2019"]
    literature_c, literature_2018, literature_scores = select_literature(base["features_2018"], labels_2018, base["covered_2018"])
    literature_2019 = pipeline._fit_literature(base["features_2018"], labels_2018, base["features_2019"], literature_c)
    matrix_2018 = pipeline._routing_matrix(aligned["baseline_2018"], registry_predictions["official_2018"], literature_2018, base["features_2018"])
    matrix_2019 = pipeline._routing_matrix(aligned["baseline_2019"], registry_predictions["official_2019"], literature_2019, base["features_2019"])
    routing_c, candidate_2018, routing_scores = select_route(matrix_2018, labels_2018, base["covered_2018"], aligned["baseline_2018"])
    candidate_2019, _ = pipeline._fit_route(matrix_2018, labels_2018, base["covered_2018"], matrix_2019, base["covered_2019"], aligned["baseline_2019"], np.ones(len(labels_2019)), routing_c)
    delta_2018 = float(pipeline._auc(labels_2018, candidate_2018) - pipeline._auc(labels_2018, base["binary_2018"]))
    delta_2019 = float(pipeline._auc(labels_2019, candidate_2019) - pipeline._auc(labels_2019, base["binary_2019"]))
    bootstrap = pipeline._bootstrap([labels_2018, labels_2019], [base["binary_2018"], base["binary_2019"]], [candidate_2018, candidate_2019])
    accepted = bool(delta_2018 >= 0 and delta_2019 >= 0 and bootstrap["probability_positive"] >= 0.8)
    diagnostics = {
        "regularization_accepted": accepted,
        "literature_c": literature_c,
        "routing_c": routing_c,
        "literature_scores_2018": literature_scores,
        "routing_scores_2018": routing_scores,
        "delta_2018": delta_2018,
        "delta_2019": delta_2019,
        "bootstrap": bootstrap,
        "retrieval": retrieval,
        "hosted": hosted,
    }
    if not accepted:
        locked_append(cache / "features_history.md", f'''\n### Extended literature and router regularization grid\n- run/experiment: generic_exp_0 lane 0 | status: TESTED-REJECTED\n- what: adaptively extended boundary grids below literature C=0.03 and above router C=1.0, selected only on 2018 and checked on sealed 2019.\n- outcome: selected literature C {literature_c}, router C {routing_c}; deltas {delta_2018:+.9f}/{delta_2019:+.9f}; bootstrap {json.dumps(bootstrap, sort_keys=True)}.\n- takeaway: retained the binary-route TF-IDF finalist because the regularization extension did not pass both temporal gates.\n''')
        print(json.dumps(diagnostics, sort_keys=True))
        return
    train_features = pipeline.pd.concat([base["features_2018"], base["features_2019"]], ignore_index=True)
    train_labels = np.concatenate([labels_2018, labels_2019])
    train_matrix = np.vstack([matrix_2018, matrix_2019])
    train_covered = np.concatenate([base["covered_2018"], base["covered_2019"]])
    validation_records, validation_adjudications, validation_source = pipeline._retrieve_and_adjudicate_split(artifacts, "validation_2020", cache, False)
    validation_ids, _, _ = pipeline._align_invariant(artifacts, "validation_2020")
    validation_features = pipeline._features_for_split(artifacts, "validation_2020", validation_records, validation_adjudications, validation_ids)
    literature_validation = pipeline._fit_literature(train_features, train_labels, validation_features, literature_c)
    validation_covered = validation_features["usable_evidence"].to_numpy(dtype=bool)
    validation_matrix = pipeline._routing_matrix(artifacts["run0009_val"], registry_predictions["validation_2020"], literature_validation, validation_features)
    validation_route, model_a = pipeline._fit_route(train_matrix, train_labels, train_covered, validation_matrix, validation_covered, artifacts["run0009_val"], np.ones(len(validation_features)), routing_c)
    validation_path = cache / "literature_v3" / "extended_regularization_model_a" / "validation_predictions.npy"
    validation_checksum = freeze(validation_path, validation_route)
    context = load_task()
    validation_labels = context.val.df[context.target_col].to_numpy(dtype=np.int32)
    test_records, test_adjudications, test_source = pipeline._retrieve_and_adjudicate_split(artifacts, "test_2021", cache, False)
    test_ids, _, _ = pipeline._align_invariant(artifacts, "test_2021")
    test_features = pipeline._features_for_split(artifacts, "test_2021", test_records, test_adjudications, test_ids)
    model_b_features = pipeline.pd.concat([train_features, validation_features], ignore_index=True)
    model_b_labels = np.concatenate([train_labels, validation_labels])
    literature_test = pipeline._fit_literature(model_b_features, model_b_labels, test_features, literature_c)
    model_b_matrix = np.vstack([train_matrix, validation_matrix])
    model_b_covered = np.concatenate([train_covered, validation_covered])
    test_covered = test_features["usable_evidence"].to_numpy(dtype=bool)
    test_matrix = pipeline._routing_matrix(artifacts["run0009_test"], registry_predictions["test_2021"], literature_test, test_features)
    test_route, model_b = pipeline._fit_route(model_b_matrix, model_b_labels, model_b_covered, test_matrix, test_covered, artifacts["run0009_test"], np.ones(len(test_features)), routing_c)
    if hashlib.sha256(validation_path.read_bytes()).hexdigest() != validation_checksum:
        raise RuntimeError("Extended-regularization Model A checksum changed during Model B")
    tfidf_path = cache / "predictions" / "generic_exp_1_tfidf_v1.npz"
    with np.load(tfidf_path, allow_pickle=False) as tfidf:
        count_2018 = len(labels_2018)
        text_2018 = tfidf["forward_predictions"][:count_2018].astype(np.float64)
        text_2019 = tfidf["forward_predictions"][count_2018:].astype(np.float64)
        validation_text = tfidf["val"].astype(np.float64)
        test_text = tfidf["test"].astype(np.float64)
        tfidf_diagnostics = json.loads(str(tfidf["diagnostics_json"][0]))
    weights = np.arange(0.0, 1.0001, 0.05)
    weight_scores = {float(weight): float(pipeline._auc(labels_2018, rank_blend(candidate_2018, text_2018, float(weight)))) for weight in weights}
    selected_weight = max(weights, key=lambda weight: (weight_scores[float(weight)], -float(weight)))
    ensemble_2018 = rank_blend(candidate_2018, text_2018, float(selected_weight))
    ensemble_2019 = rank_blend(candidate_2019, text_2019, float(selected_weight))
    ensemble_bootstrap = paired_bootstrap(labels_2018, candidate_2018, ensemble_2018)
    ensemble_delta_2018 = float(pipeline._auc(labels_2018, ensemble_2018) - pipeline._auc(labels_2018, candidate_2018))
    ensemble_delta_2019 = float(pipeline._auc(labels_2019, ensemble_2019) - pipeline._auc(labels_2019, candidate_2019))
    ensemble_accepted = bool(selected_weight > 0 and ensemble_delta_2018 >= 0 and ensemble_delta_2019 >= 0 and ensemble_bootstrap["probability_positive"] >= 0.8)
    validation = rank_blend(validation_route, validation_text, float(selected_weight)) if ensemble_accepted else validation_route
    test = rank_blend(test_route, test_text, float(selected_weight)) if ensemble_accepted else test_route
    ensemble_path = cache / "cross_lane_extended_regularization_v1" / "model_a_validation.npy"
    ensemble_checksum = freeze(ensemble_path, validation)
    if hashlib.sha256(ensemble_path.read_bytes()).hexdigest() != ensemble_checksum:
        raise RuntimeError("Extended-regularization ensemble checksum changed")
    diagnostics.update({
        "accepted": True,
        "variant": "extended_regularization_tfidf" if ensemble_accepted else "extended_regularization",
        "validation_checksum": validation_checksum,
        "ensemble_validation_checksum": ensemble_checksum,
        "validation_covered": int(validation_covered.sum()),
        "test_covered": int(test_covered.sum()),
        "model_a_coefficients": model_a[1].tolist(),
        "model_b_coefficients": model_b[1].tolist(),
        "validation_source": validation_source,
        "test_source": test_source,
        "ensemble_accepted": ensemble_accepted,
        "selected_tfidf_weight": float(selected_weight),
        "ensemble_delta_2018": ensemble_delta_2018,
        "ensemble_delta_2019": ensemble_delta_2019,
        "ensemble_bootstrap": ensemble_bootstrap,
        "weight_scores_2018": {str(key): value for key, value in weight_scores.items()},
        "tfidf_source_diagnostics": tfidf_diagnostics,
    })
    destination = cache / "predictions" / "generic_exp_0_literature_extended_regularization_v1.npz"
    persist(destination, validation, test, diagnostics)
    register_artifact(cache, {
        "name": "generic_exp_0 extended publication regularization finalist",
        "path": "predictions/generic_exp_0_literature_extended_regularization_v1.npz",
        "description": "Boundary-extended literature/router regularization variant, optionally rank-blended with the accepted supervised TF-IDF channel.",
        "content_key": "rel-trial-study-outcome:generic-exp-0:literature-extended-regularization-v1",
        "rebuild_hint": "Run build_extended_regularization_variant.py from warm literature and TF-IDF caches.",
    })
    locked_append(cache / "features_history.md", f'''\n### Extended literature and router regularization grid\n- run/experiment: generic_exp_0 lane 0 | status: TESTED-KEPT\n- what: adaptively extended boundary grids below literature C=0.03 and above router C=1.0, then gated an accepted TF-IDF rank blend.\n- outcome: selected literature C {literature_c}, router C {routing_c}; regularization deltas {delta_2018:+.9f}/{delta_2019:+.9f}, bootstrap {json.dumps(bootstrap, sort_keys=True)}; TF-IDF weight {float(selected_weight):.2f}, deltas {ensemble_delta_2018:+.9f}/{ensemble_delta_2019:+.9f}.\n- takeaway: selection used 2018 only and passed sealed 2019 before Model A was frozen.\n''')
    print(json.dumps(diagnostics, sort_keys=True))


if __name__ == "__main__":
    main()
