import hashlib
import json
import os
from pathlib import Path

import numpy as np

import publication_pipeline as pipeline
from build_cross_lane_ensemble import paired_bootstrap, rank_blend
from campaign_io import locked_append, register_artifact
from kapso_datasets.common import load_task, shared_cache_dir
from publication_evidence import adjudicate_candidates


def freeze(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values.astype(np.float64))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def persist(path: Path, validation: np.ndarray, test: np.ndarray, diagnostics: dict[str, object]) -> None:
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, val=validation, test=test, diagnostics_json=np.asarray([json.dumps(diagnostics, sort_keys=True, allow_nan=True)]))
    os.replace(temporary, path)


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
    gate = pipeline.fit_publication_gate(artifacts, aligned, registry_predictions, records, adjudications)
    binary_delta_2018 = float(pipeline._auc(aligned["labels_2018"], gate["binary_2018"]) - pipeline._auc(aligned["labels_2018"], gate["candidate_2018"]))
    binary_delta_2019 = float(pipeline._auc(aligned["labels_2019"], gate["binary_2019"]) - pipeline._auc(aligned["labels_2019"], gate["candidate_2019"]))
    binary_bootstrap = pipeline._bootstrap(
        [aligned["labels_2018"], aligned["labels_2019"]],
        [gate["candidate_2018"], gate["candidate_2019"]],
        [gate["binary_2018"], gate["binary_2019"]],
    )
    binary_accepted = bool(binary_delta_2018 >= 0 and binary_delta_2019 >= 0 and binary_bootstrap["probability_positive"] >= 0.8)
    diagnostics = {
        "binary_accepted": binary_accepted,
        "binary_delta_2018": binary_delta_2018,
        "binary_delta_2019": binary_delta_2019,
        "binary_bootstrap": binary_bootstrap,
        "publication_gate": gate["diagnostics"],
        "retrieval": retrieval,
        "hosted": hosted,
    }
    if not binary_accepted:
        print(json.dumps(diagnostics, sort_keys=True))
        locked_append(cache / "features_history.md", f'''\n### Binary publication routing finalist\n- run/experiment: generic_exp_0 lane 0 | status: TESTED-REJECTED\n- what: original binary evidence routing against accepted confidence-tiered routing, measured on 2018 and sealed 2019.\n- outcome: deltas {binary_delta_2018:+.9f}/{binary_delta_2019:+.9f}; bootstrap {json.dumps(binary_bootstrap, sort_keys=True)}.\n- takeaway: retained the confidence-tiered ensemble because the binary variant did not meet the precommitted acceptance gate.\n''')
        return
    train_features = np.concatenate([np.arange(len(gate["features_2018"])), np.arange(len(gate["features_2019"]))])
    if len(train_features) != len(aligned["labels_2018"]) + len(aligned["labels_2019"]):
        raise RuntimeError("Gate feature rows are inconsistent")
    labels = np.concatenate([aligned["labels_2018"], aligned["labels_2019"]])
    matrix = np.vstack([gate["matrix_2018"], gate["matrix_2019"]])
    covered = np.concatenate([gate["covered_2018"], gate["covered_2019"]])
    validation_records, validation_adjudications, validation_source = pipeline._retrieve_and_adjudicate_split(artifacts, "validation_2020", cache, False)
    validation_ids, _, _ = pipeline._align_invariant(artifacts, "validation_2020")
    validation_features = pipeline._features_for_split(artifacts, "validation_2020", validation_records, validation_adjudications, validation_ids)
    combined_features = pipeline.pd.concat([gate["features_2018"], gate["features_2019"]], ignore_index=True)
    literature_validation = pipeline._fit_literature(combined_features, labels, validation_features, gate["selected_c"])
    validation_covered = validation_features["usable_evidence"].to_numpy(dtype=bool)
    validation_matrix = pipeline._routing_matrix(artifacts["run0009_val"], registry_predictions["validation_2020"], literature_validation, validation_features)
    validation_binary, model_a = pipeline._fit_route(matrix, labels, covered, validation_matrix, validation_covered, artifacts["run0009_val"], np.ones(len(validation_features)), gate["routing_c"])
    freeze_path = cache / "literature_v3" / "binary_model_a" / "validation_predictions.npy"
    validation_checksum = freeze(freeze_path, validation_binary)
    context = load_task()
    validation_labels = context.val.df[context.target_col].to_numpy(dtype=np.int32)
    test_records, test_adjudications, test_source = pipeline._retrieve_and_adjudicate_split(artifacts, "test_2021", cache, False)
    test_ids, _, _ = pipeline._align_invariant(artifacts, "test_2021")
    test_features = pipeline._features_for_split(artifacts, "test_2021", test_records, test_adjudications, test_ids)
    model_b_features = pipeline.pd.concat([combined_features, validation_features], ignore_index=True)
    model_b_labels = np.concatenate([labels, validation_labels])
    literature_test = pipeline._fit_literature(model_b_features, model_b_labels, test_features, gate["selected_c"])
    model_b_matrix = np.vstack([matrix, validation_matrix])
    model_b_covered = np.concatenate([covered, validation_covered])
    test_covered = test_features["usable_evidence"].to_numpy(dtype=bool)
    test_matrix = pipeline._routing_matrix(artifacts["run0009_test"], registry_predictions["test_2021"], literature_test, test_features)
    test_binary, model_b = pipeline._fit_route(model_b_matrix, model_b_labels, model_b_covered, test_matrix, test_covered, artifacts["run0009_test"], np.ones(len(test_features)), gate["routing_c"])
    if hashlib.sha256(freeze_path.read_bytes()).hexdigest() != validation_checksum:
        raise RuntimeError("Binary Model A checksum changed during Model B construction")
    diagnostics.update({
        "validation_checksum": validation_checksum,
        "validation_covered": int(validation_covered.sum()),
        "test_covered": int(test_covered.sum()),
        "model_a_coefficients": model_a[1].tolist(),
        "model_b_coefficients": model_b[1].tolist(),
        "validation_source": validation_source,
        "test_source": test_source,
    })
    binary_path = cache / "predictions" / "generic_exp_0_literature_binary_v1.npz"
    persist(binary_path, validation_binary, test_binary, {**diagnostics, "accepted": True, "variant": "binary"})
    tfidf_path = cache / "predictions" / "generic_exp_1_tfidf_v1.npz"
    with np.load(tfidf_path, allow_pickle=False) as tfidf:
        tfidf_diagnostics = json.loads(str(tfidf["diagnostics_json"][0]))
        if not bool(tfidf_diagnostics.get("forward_gate", False)):
            raise RuntimeError("Cross-lane TF-IDF bank is not forward accepted")
        forward_indices = tfidf["forward_index"].astype(np.int64)
        timestamps = artifacts["invariant"]["train_timestamp"]
        expected = np.concatenate([np.flatnonzero(timestamps == np.datetime64("2018-01-01")), np.flatnonzero(timestamps == np.datetime64("2019-01-01"))])
        if not np.array_equal(forward_indices, expected):
            raise RuntimeError("Binary literature and TF-IDF rows are not aligned")
        design_count = len(aligned["labels_2018"])
        design_text = tfidf["forward_predictions"][:design_count].astype(np.float64)
        sealed_text = tfidf["forward_predictions"][design_count:].astype(np.float64)
        validation_text = tfidf["val"].astype(np.float64)
        test_text = tfidf["test"].astype(np.float64)
    weights = np.arange(0.0, 1.0001, 0.05)
    scores = {float(weight): float(pipeline._auc(aligned["labels_2018"], rank_blend(gate["binary_2018"], design_text, float(weight)))) for weight in weights}
    selected_weight = max(weights, key=lambda weight: (scores[float(weight)], -float(weight)))
    design_ensemble = rank_blend(gate["binary_2018"], design_text, float(selected_weight))
    sealed_ensemble = rank_blend(gate["binary_2019"], sealed_text, float(selected_weight))
    ensemble_bootstrap = paired_bootstrap(aligned["labels_2018"], gate["binary_2018"], design_ensemble)
    ensemble_delta_2018 = float(pipeline._auc(aligned["labels_2018"], design_ensemble) - pipeline._auc(aligned["labels_2018"], gate["binary_2018"]))
    ensemble_delta_2019 = float(pipeline._auc(aligned["labels_2019"], sealed_ensemble) - pipeline._auc(aligned["labels_2019"], gate["binary_2019"]))
    ensemble_accepted = bool(selected_weight > 0 and ensemble_delta_2018 >= 0 and ensemble_delta_2019 >= 0 and ensemble_bootstrap["probability_positive"] >= 0.8)
    validation = rank_blend(validation_binary, validation_text, float(selected_weight)) if ensemble_accepted else validation_binary
    test = rank_blend(test_binary, test_text, float(selected_weight)) if ensemble_accepted else test_binary
    ensemble_freeze_path = cache / "cross_lane_binary_ensemble_v1" / "model_a_validation.npy"
    ensemble_checksum = freeze(ensemble_freeze_path, validation)
    if hashlib.sha256(ensemble_freeze_path.read_bytes()).hexdigest() != ensemble_checksum:
        raise RuntimeError("Binary ensemble Model A checksum changed")
    ensemble_diagnostics = {
        **diagnostics,
        "accepted": True,
        "variant": "binary_tfidf" if ensemble_accepted else "binary",
        "cross_ensemble_accepted": ensemble_accepted,
        "selected_tfidf_weight": float(selected_weight),
        "ensemble_delta_2018": ensemble_delta_2018,
        "ensemble_delta_2019": ensemble_delta_2019,
        "ensemble_bootstrap": ensemble_bootstrap,
        "ensemble_validation_checksum": ensemble_checksum,
        "weight_scores_2018": {str(key): value for key, value in scores.items()},
        "tfidf_source_diagnostics": tfidf_diagnostics,
    }
    ensemble_path = cache / "predictions" / "generic_exp_0_literature_binary_tfidf_v1.npz"
    persist(ensemble_path, validation, test, ensemble_diagnostics)
    register_artifact(cache, {
        "name": "generic_exp_0 binary literature routing plus TF-IDF finalist",
        "path": "predictions/generic_exp_0_literature_binary_tfidf_v1.npz",
        "description": "Independently accepted binary publication-routing variant and its forward-OOF-gated cross-branch TF-IDF rank ensemble.",
        "content_key": "rel-trial-study-outcome:generic-exp-0:literature-binary-tfidf-v1",
        "rebuild_hint": "Run build_binary_route_variant.py from warm literature and text caches.",
    })
    locked_append(cache / "features_history.md", f'''\n### Binary publication routing finalist and TF-IDF ensemble\n- run/experiment: generic_exp_0 lane 0 | status: TESTED-KEPT\n- what: binary literature evidence route measured against tiered routing, followed by a separately gated rank blend with the accepted supervised TF-IDF text channel.\n- outcome: binary deltas {binary_delta_2018:+.9f}/{binary_delta_2019:+.9f}, bootstrap {json.dumps(binary_bootstrap, sort_keys=True)}; TF-IDF weight {float(selected_weight):.2f}, deltas {ensemble_delta_2018:+.9f}/{ensemble_delta_2019:+.9f}, bootstrap {json.dumps(ensemble_bootstrap, sort_keys=True)}.\n- takeaway: both variants were selected on 2018 and sealed once on 2019; Model A was frozen before validation labels were loaded.\n''')
    print(json.dumps(ensemble_diagnostics, sort_keys=True))


if __name__ == "__main__":
    main()
