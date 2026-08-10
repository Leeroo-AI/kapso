from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from feature_mart import register_artifact, true_analysis_counts
from modeling import fit_final_chain


VERSION = "lane1_graph_semantic_oof_fusion_v1"
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 1.0]


def load_graph_module(shared: Path):
    code = shared / "lane1_banked_run_0006" / "code"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    return importlib.import_module("relational_model")


def graph_inputs(context, graph, shared: Path):
    train = context.train.df[["timestamp", "nct_id", context.target_col]].copy().reset_index(drop=True)
    val = context.val.df[["timestamp", "nct_id", context.target_col]].copy().reset_index(drop=True)
    test = context.test.df[["timestamp", "nct_id"]].copy().reset_index(drop=True)
    repository = graph.DataRepository(context, shared, False)
    combined = pd.concat([train[["timestamp", "nct_id"]], val[["timestamp", "nct_id"]], test[["timestamp", "nct_id"]]], ignore_index=True)
    matrix, names = graph.TabularBuilder(repository).build(combined)
    return repository, graph.TemporalGraphManager(repository), train, matrix[:len(train)], names


def graph_oof(context, shared: Path) -> tuple[dict[int, np.ndarray], dict]:
    graph = load_graph_module(shared)
    graph.set_seed(graph.SEED)
    repository, manager, train, matrix, names = graph_inputs(context, graph, shared)
    labels = train[context.target_col].to_numpy(np.float32)
    years = [2017, 2018, 2019]
    tabular_scores, tabular_predictions = graph.lightgbm_forward_scores(train, matrix, labels, years)
    auxiliary = repository.auxiliary_count(train)
    row_year = pd.to_datetime(train["timestamp"]).dt.year.to_numpy()
    device_index = int(os.environ.get("CUDA_DEVICE", "0"))
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
    output = {}
    details = {}
    for year in years:
        train_mask = row_year < year
        valid_mask = row_year == year
        train_rows = train.loc[train_mask].reset_index(drop=True)
        valid_rows = train.loc[valid_mask].reset_index(drop=True)
        train_matrix = matrix[train_mask]
        valid_matrix = matrix[valid_mask]
        train_labels = labels[train_mask]
        valid_labels = labels[valid_mask]
        scaler = RobustScaler(quantile_range=(10, 90)).fit(train_matrix)
        train_groups = graph.graph_groups(manager, train_rows, train_matrix, train_labels, auxiliary[train_mask], scaler)
        valid_groups = graph.graph_groups(manager, valid_rows, valid_matrix, valid_labels, auxiliary[valid_mask], scaler)
        model, best_epoch, _, graph_auc = graph.train_graph_model(train_groups, valid_groups, matrix.shape[1], 40, 6, device)
        prediction = graph.predict_graph(model, valid_groups, device, len(valid_rows))
        blended = 0.75 * tabular_predictions[year] + 0.25 * prediction
        output[year] = blended.astype(np.float64)
        details[str(year)] = {"count": int(valid_mask.sum()), "tabular_auc": float(tabular_scores[year]), "graph_auc": float(graph_auc), "frozen_run0006_blend_auc": float(roc_auc_score(valid_labels, blended)), "best_epoch": int(best_epoch)}
        print(f"[graph-fusion] year={year} graph_auc={graph_auc:.6f} run0006_blend_auc={details[str(year)]['frozen_run0006_blend_auc']:.6f}", flush=True)
        del model, train_groups, valid_groups
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output, details


def semantic_oof(seeds: pd.DataFrame, structured: pd.DataFrame, neighbors: pd.DataFrame, categories: pd.DataFrame, embeddings: np.ndarray, texts: list[str], episodes: pd.DataFrame, selection: dict, debug: bool = False) -> dict[int, np.ndarray]:
    train_rows = seeds[seeds["split"].eq("train")]
    output = {}
    for year in [2017, 2018, 2019]:
        fit_ids = train_rows.loc[train_rows["timestamp"].dt.year.lt(year), "row_id"].to_numpy(np.int64)
        valid_ids = train_rows.loc[train_rows["timestamp"].dt.year.eq(year), "row_id"].to_numpy(np.int64)
        counts = true_analysis_counts(seeds.loc[fit_ids], episodes)
        prediction, _ = fit_final_chain(seeds, structured, neighbors, categories, embeddings, texts, counts, fit_ids, valid_ids, selection, debug)
        output[year] = prediction.astype(np.float64)
        print(f"[graph-fusion] year={year} semantic_auc={roc_auc_score(seeds.set_index('row_id').loc[valid_ids, 'outcome'], prediction):.6f}", flush=True)
    return output


def select_fusion(seeds: pd.DataFrame, semantic_predictions: dict[int, np.ndarray], graph_predictions: dict[int, np.ndarray], graph_details: dict) -> dict:
    train_rows = seeds[seeds["split"].eq("train")]
    labels = train_rows.set_index("row_id")["outcome"]
    fold_scores = {weight: {} for weight in WEIGHTS}
    pooled_labels = []
    pooled_semantic = []
    pooled_graph = []
    for year in [2017, 2018, 2019]:
        ids = train_rows.loc[train_rows["timestamp"].dt.year.eq(year), "row_id"].to_numpy(np.int64)
        current = labels.loc[ids].to_numpy(np.int64)
        pooled_labels.append(current)
        pooled_semantic.append(semantic_predictions[year])
        pooled_graph.append(graph_predictions[year])
        for weight in WEIGHTS:
            prediction = (1.0 - weight) * semantic_predictions[year] + weight * graph_predictions[year]
            fold_scores[weight][str(year)] = float(roc_auc_score(current, prediction))
    pooled_labels_array = np.concatenate(pooled_labels)
    pooled_semantic_array = np.concatenate(pooled_semantic)
    pooled_graph_array = np.concatenate(pooled_graph)
    pooled_scores = {}
    mean_scores = {}
    wins = {}
    for weight in WEIGHTS:
        prediction = (1.0 - weight) * pooled_semantic_array + weight * pooled_graph_array
        pooled_scores[weight] = float(roc_auc_score(pooled_labels_array, prediction))
        mean_scores[weight] = float(np.mean(list(fold_scores[weight].values())))
        wins[weight] = int(sum(fold_scores[weight][str(year)] > fold_scores[0.0][str(year)] for year in [2017, 2018, 2019]))
    eligible = [weight for weight in WEIGHTS if weight == 0.0 or wins[weight] >= 2]
    selected = max(eligible, key=lambda weight: (mean_scores[weight], pooled_scores[weight], -weight))
    return {"weights": WEIGHTS, "fold_scores": {str(weight): fold_scores[weight] for weight in WEIGHTS}, "fold_means": {str(weight): mean_scores[weight] for weight in WEIGHTS}, "pooled_scores": {str(weight): pooled_scores[weight] for weight in WEIGHTS}, "fold_wins_over_semantic": {str(weight): wins[weight] for weight in WEIGHTS}, "selected_graph_weight": float(selected), "graph_details": graph_details, "selection_lineage": "2017-2019 forward folds from training labels only"}


def load_or_build_graph_fusion(context, seeds: pd.DataFrame, structured: pd.DataFrame, neighbors: pd.DataFrame, categories: pd.DataFrame, embeddings: np.ndarray, texts: list[str], episodes: pd.DataFrame, base_selection: dict, shared: Path, debug: bool) -> dict:
    root = shared / VERSION
    root.mkdir(parents=True, exist_ok=True)
    path = root / "selection.json"
    if path.exists():
        return json.loads(path.read_text())
    if debug:
        return {"selected_graph_weight": 0.0, "selection_lineage": "debug fallback"}
    graph_predictions, graph_details = graph_oof(context, shared)
    semantic_predictions = semantic_oof(seeds, structured, neighbors, categories, embeddings, texts, episodes, base_selection)
    selection = select_fusion(seeds, semantic_predictions, graph_predictions, graph_details)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(selection, indent=2) + "\n")
    os.replace(temporary, path)
    np.savez_compressed(root / "oof_predictions.npz", years=np.asarray([2017, 2018, 2019]), semantic=np.concatenate([semantic_predictions[year] for year in [2017, 2018, 2019]]), graph=np.concatenate([graph_predictions[year] for year in [2017, 2018, 2019]]))
    register_artifact(shared, "Lane 1 graph-semantic forward OOF fusion", root, "Matching 2017--2019 training-only OOF predictions for the banked run_0006 structural graph and the semantic mart, with endpoint-inclusive fixed blending.", VERSION, "Run main.py after deleting this directory; temporal graph snapshots and semantic marts are reused.")
    print(f"[graph-fusion] selected_graph_weight={selection['selected_graph_weight']:.2f}", flush=True)
    return selection


def apply_graph_fusion(prediction: np.ndarray, split: str, selection: dict, shared: Path) -> np.ndarray:
    weight = float(selection.get("selected_graph_weight", 0.0))
    if weight <= 0:
        return np.asarray(prediction, dtype=np.float64)
    path = shared / "lane1_banked_run_0006" / f"{split}_predictions.npy"
    graph_prediction = np.load(path).astype(np.float64)
    if graph_prediction.shape != np.asarray(prediction).shape:
        raise RuntimeError(f"banked graph prediction shape mismatch for {split}: {graph_prediction.shape}, {np.asarray(prediction).shape}")
    return np.clip((1.0 - weight) * np.asarray(prediction, dtype=np.float64) + weight * graph_prediction, 1e-6, 1 - 1e-6)
