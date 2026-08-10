from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from entity_diffusion import VERSION as DIFFUSION_VERSION, load_or_build_diffusion
from feature_mart import build_episodes, direct_features, document_table, elapsed, empirical_bayes_features, encode_documents, historical_maps, make_seeds, register_artifact, semantic_features, true_analysis_counts
from graph_fusion import apply_graph_fusion, load_or_build_graph_fusion
from kapso_datasets.common import is_debug, load_task, run_data_dir, save_predictions, shared_cache_dir
from modeling import fit_diffusion_chain, fit_final_chain, forward_design_selection, forward_diffusion_selection, prepare_structured


def atomic_pickle(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_pickle(temporary)
    os.replace(temporary, path)


def load_or_build_structured(context, seeds: pd.DataFrame, episodes: pd.DataFrame, shared: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = shared / "lane0_temporal_evidence_mart_v1"
    root.mkdir(parents=True, exist_ok=True)
    numeric_path = root / "base_numeric.pkl"
    category_path = root / "categories.pkl"
    if numeric_path.exists() and category_path.exists():
        numeric = pd.read_pickle(numeric_path)
        categories = pd.read_pickle(category_path)
        if len(numeric) != len(seeds) or len(categories) != len(seeds):
            raise RuntimeError("structured mart cache row mismatch")
        print(f"[pipeline] loaded structured mart cache: {numeric.shape}", flush=True)
        return numeric, categories
    direct, categories, seed_maps = direct_features(context.db, seeds)
    hist_maps, all_maps = historical_maps(context.db, episodes)
    priors = empirical_bayes_features(seeds, episodes, seed_maps, hist_maps, all_maps)
    numeric = direct.join(priors)
    atomic_pickle(numeric, numeric_path)
    atomic_pickle(categories, category_path)
    register_artifact(shared, "lane0 all-table temporal evidence mart", root, "Timestamp-censored direct features and multi-view empirical-Bayes histories for every seed row.", "study-outcome-lane0-temporal-v1", "Run main.py after deleting this directory to rebuild from the sanitized RelBench database.")
    print(f"[pipeline] cached structured mart: {numeric.shape}", flush=True)
    return numeric, categories


def load_or_build_semantic(context, seeds: pd.DataFrame, episodes: pd.DataFrame, shared: Path, debug: bool) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    docs, seed_indices, corpus_indices = document_table(context.db, seeds, episodes)
    texts = (docs.iloc[seed_indices]["title_summary"].fillna("") + " [SEP] " + docs.iloc[seed_indices]["eligibility"].fillna("") + " [SEP] " + docs.iloc[seed_indices]["entity_terms"].fillna("")).tolist()
    root = shared / "lane0_semantic_retrieval_v4"
    neighbor_path = root / "neighbors.pkl"
    embedding_path = root / "seed_embeddings.npy"
    if not debug and neighbor_path.exists() and embedding_path.exists():
        neighbors = pd.read_pickle(neighbor_path)
        embeddings = np.load(embedding_path)
        if len(neighbors) != len(seeds) or embeddings.shape != (len(seeds), 2304):
            raise RuntimeError("semantic cache row mismatch")
        print(f"[pipeline] loaded semantic retrieval cache: {neighbors.shape}", flush=True)
        return neighbors, embeddings, texts
    embeddings_by_section = encode_documents(docs, shared, debug)
    neighbors, embeddings = semantic_features(seeds, episodes, docs, embeddings_by_section, seed_indices, corpus_indices, debug)
    if not debug:
        root.mkdir(parents=True, exist_ok=True)
        atomic_pickle(neighbors, neighbor_path)
        temporary = embedding_path.with_suffix(".tmp.npy")
        np.save(temporary, embeddings)
        os.replace(temporary, embedding_path)
        register_artifact(shared, "lane0 title-section temporal biomedical retrieval", root, "Combined-document and title-section temporally eligible semantic neighbor summaries from frozen biomedical embeddings.", "study-outcome-lane0-semantic-v4-model-048ad4491de0", "Delete this directory and run main.py; frozen section embeddings are reused.")
    return neighbors, embeddings, texts


def subset_ids(ids: np.ndarray, maximum: int) -> np.ndarray:
    if len(ids) <= maximum:
        return ids
    positions = np.linspace(0, len(ids) - 1, maximum).astype(int)
    return ids[positions]


def main() -> None:
    warnings.filterwarnings("ignore")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    debug = is_debug()
    start = time.time()
    context = load_task()
    seeds = make_seeds(context)
    shared = shared_cache_dir()
    local = Path("output_data_generic_exp_5")
    local.mkdir(parents=True, exist_ok=True)
    elapsed(start, "loaded task and preserved row register")

    episodes = build_episodes(context.db)
    elapsed(start, "built exact historical result corpus")
    numeric, categories = load_or_build_structured(context, seeds, episodes, shared)
    elapsed(start, "completed structured temporal evidence mart")
    neighbors, embeddings, texts = load_or_build_semantic(context, seeds, episodes, shared, debug)
    elapsed(start, "completed frozen embeddings and semantic retrieval")
    structured = prepare_structured(seeds, numeric, neighbors)

    counts = np.full(len(seeds), np.nan, dtype=float)
    train_ids_all = seeds.loc[seeds["split"].eq("train"), "row_id"].to_numpy(dtype=np.int64)
    val_ids = seeds.loc[seeds["split"].eq("val"), "row_id"].to_numpy(dtype=np.int64)
    test_ids = seeds.loc[seeds["split"].eq("test"), "row_id"].to_numpy(dtype=np.int64)
    counts[train_ids_all] = true_analysis_counts(seeds.loc[train_ids_all], episodes)

    selection_path = shared / "lane0_forward_selection_v5.json"
    if selection_path.exists() and not debug:
        selection = json.loads(selection_path.read_text())
        selection["selected_c"] = float(selection["selected_c"])
        selection["blend_weight"] = float(selection["blend_weight"])
        print(f"[selection] loaded frozen training-only selection: {selection['selected_design']}", flush=True)
    else:
        selection = forward_design_selection(seeds, structured, neighbors, categories, embeddings, texts, counts, debug)
        if not debug:
            temporary = selection_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(selection, indent=2) + "\n")
            os.replace(temporary, selection_path)
            register_artifact(shared, "lane0 forward-fold title-section retrieval selection", selection_path, "Training-only 2016-2019 feature-block, rounds, semantic logistic C, and blend selection with the simpler title-section retrieval block.", "study-outcome-lane0-forward-selection-v5", "Delete this JSON and run main.py to repeat training-only forward selection.")
    elapsed(start, "froze forward-fold design")

    train_ids_a = subset_ids(train_ids_all, 2000) if debug else train_ids_all
    val_prediction, lineage_a = fit_final_chain(seeds, structured, neighbors, categories, embeddings, texts, counts[train_ids_a], train_ids_a, val_ids, selection, debug)
    if not seeds.set_index("row_id").loc[train_ids_a, "split"].eq("train").all():
        raise RuntimeError("Model A lineage includes non-training rows")
    elapsed(start, "fit Model A and generated validation predictions")

    counts[val_ids] = true_analysis_counts(seeds.loc[val_ids], episodes)
    train_ids_b_all = np.concatenate([train_ids_all, val_ids])
    train_ids_b = subset_ids(train_ids_b_all, 2000) if debug else train_ids_b_all
    test_prediction, lineage_b = fit_final_chain(seeds, structured, neighbors, categories, embeddings, texts, counts[train_ids_b], train_ids_b, test_ids, selection, debug)
    if seeds.set_index("row_id").loc[train_ids_b, "split"].eq("test").any():
        raise RuntimeError("Model B lineage includes test rows")
    elapsed(start, "fit Model B and generated test predictions")

    save_predictions(val_prediction.astype(np.float64), test_prediction.astype(np.float64))
    np.save(run_data_dir() / "semantic_base_val_predictions.npy", val_prediction.astype(np.float64))
    np.save(run_data_dir() / "semantic_base_test_predictions.npy", test_prediction.astype(np.float64))
    elapsed(start, "wrote semantic-base failure-safe predictions")

    diffusion, diffusion_diagnostics = load_or_build_diffusion(context.db, seeds, episodes, shared, debug)
    elapsed(start, "completed causal entity-semantic diffusion")
    diffusion_selection_path = shared / DIFFUSION_VERSION / "forward_selection_v1.json"
    if diffusion_selection_path.exists() and not debug:
        diffusion_selection = json.loads(diffusion_selection_path.read_text())
        print(f"[diffusion-selection] loaded frozen training-only selection: {diffusion_selection['selected']}", flush=True)
    else:
        diffusion_selection = forward_diffusion_selection(seeds, structured, neighbors, diffusion, categories, debug)
        if not debug:
            temporary = diffusion_selection_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(diffusion_selection, indent=2) + "\n")
            os.replace(temporary, diffusion_selection_path)
            register_artifact(shared, "Lane 1 diffusion forward selection", diffusion_selection_path, "Training-only 2016--2019 threshold, depth, route-dropout, pure-endpoint, and blend selection for causal typed diffusion.", f"{DIFFUSION_VERSION}-selection-v1", "Delete this JSON and run main.py to repeat the fixed forward-fold screen.")
    elapsed(start, "froze diffusion threshold depth and blend")

    val_prediction, diffusion_lineage_a = fit_diffusion_chain(seeds, structured, neighbors, diffusion, categories, train_ids_a, val_ids, val_prediction, diffusion_selection, debug)
    if not seeds.set_index("row_id").loc[train_ids_a, "split"].eq("train").all():
        raise RuntimeError("Diffusion Model A lineage includes non-training rows")
    elapsed(start, "fit diffusion Model A and generated validation predictions")
    test_prediction, diffusion_lineage_b = fit_diffusion_chain(seeds, structured, neighbors, diffusion, categories, train_ids_b, test_ids, test_prediction, diffusion_selection, debug)
    if seeds.set_index("row_id").loc[train_ids_b, "split"].eq("test").any():
        raise RuntimeError("Diffusion Model B lineage includes test rows")
    elapsed(start, "fit diffusion Model B and generated test predictions")

    graph_fusion_selection = load_or_build_graph_fusion(context, seeds, structured, neighbors, categories, embeddings, texts, episodes, selection, shared, debug)
    val_prediction = apply_graph_fusion(val_prediction, "val", graph_fusion_selection, shared)
    test_prediction = apply_graph_fusion(test_prediction, "test", graph_fusion_selection, shared)
    elapsed(start, "applied training-only graph-semantic OOF fusion")

    if val_prediction.shape != (len(val_ids),) or test_prediction.shape != (len(test_ids),):
        raise RuntimeError(f"prediction shape mismatch: {val_prediction.shape}, {test_prediction.shape}")
    diagnostics = {"debug": debug, "elapsed_seconds": time.time() - start, "historical_corpus_rows": len(episodes), "selection": selection, "diffusion_selection": diffusion_selection, "diffusion_diagnostics": diffusion_diagnostics, "graph_fusion_selection": graph_fusion_selection, "semantic_base_model_a": lineage_a, "semantic_base_model_b": lineage_b, "diffusion_model_a": diffusion_lineage_a, "diffusion_model_b": diffusion_lineage_b, "validation_fit_lineage": "train labels only including banked graph Model A", "test_fit_lineage": "train plus validation labels including banked graph Model B", "val_shape": list(val_prediction.shape), "test_shape": list(test_prediction.shape)}
    (run_data_dir() / "metrics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (local / ("debug_metrics.json" if debug else "full_metrics.json")).write_text(json.dumps(diagnostics, indent=2) + "\n")
    save_predictions(val_prediction.astype(np.float64), test_prediction.astype(np.float64))
    elapsed(start, "saved immutable-order predictions")


if __name__ == "__main__":
    main()
