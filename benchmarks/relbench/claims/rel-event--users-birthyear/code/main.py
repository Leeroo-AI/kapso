from __future__ import annotations

import json
import hashlib
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from graph_stages import SOFT_NAMES, correct_and_smooth_features, soft_neighbor_features
from kapso_datasets.common import is_debug, load_task, run_data_dir, save_predictions, shared_cache_dir
from modeling import choose_ensemble, fit_dual_model, improvement_gate, r2_score, shallow_residual_fit
from relational import FeatureBuilder, append_artifact_registry, build_activity_stores, build_graph, user_frame, version_key


warnings.filterwarnings("ignore")


PIPELINE_VERSION = "staged_causal_graph_boost_20260801_v3"
CATBOOST_DROP_PREFIXES = ("friend_gender_", "friend_timezone_", "friend_locale_", "friend_top_", "friend_join_", "friend_degree_", "friend_demo_")


def phase(name: str, started: float, details: str = "") -> None:
    elapsed = time.time() - started
    suffix = f" {details}" if details else ""
    print(f"[pipeline] {name} elapsed={elapsed:.1f}s{suffix}", flush=True)


def ensure_living_documents(cache: Path) -> tuple[Path, Path]:
    table_path = cache / "table_information.md"
    feature_path = cache / "features_history.md"
    if not table_path.exists():
        table_path.write_text("# Table information\n\n")
    if not feature_path.exists():
        feature_path.write_text("# Feature history\n\n")
    table_path.read_text()
    feature_path.read_text()
    return table_path, feature_path


def append_text(path: Path, text: str) -> None:
    with path.open("a") as handle:
        handle.write(text)


def immutable_table(table, include_target: bool) -> pd.DataFrame:
    columns = ["joinedAt", "user_id"] + (["birthyear"] if include_target else [])
    frame = table.df[columns].copy().reset_index(drop=True)
    frame["_row_index"] = np.arange(len(frame), dtype=np.int64)
    frame["user_id"] = frame["user_id"].astype(np.int64)
    frame["joinedAt"] = pd.to_datetime(frame["joinedAt"])
    return frame


def debug_labels(train: pd.DataFrame, count: int = 3000) -> pd.DataFrame:
    ordered = train.sort_values("joinedAt", kind="stable")
    selected = np.linspace(0, len(ordered) - 1, min(count, len(ordered)), dtype=int)
    return ordered.iloc[selected].sort_values("joinedAt", kind="stable").reset_index(drop=True)


def positions_for(rows: pd.DataFrame, node_positions: pd.Series) -> np.ndarray:
    return node_positions.reindex(rows["user_id"].to_numpy()).to_numpy(dtype=np.int64)


def labels_key(labels: pd.DataFrame, extra: str = "") -> str:
    digest = hashlib.sha256(PIPELINE_VERSION.encode())
    digest.update(extra.encode())
    digest.update(labels["user_id"].to_numpy(dtype=np.int64).tobytes())
    digest.update(labels["birthyear"].to_numpy(dtype=np.float64).tobytes())
    digest.update(pd.to_datetime(labels["joinedAt"]).astype("int64").to_numpy().tobytes())
    return digest.hexdigest()[:20]


def cached_features(builder, nodes: pd.DataFrame, labels: pd.DataFrame, cache: Path, tag: str):
    key = labels_key(labels, f"features|{tag}")
    path = cache / f"relational_features_{key}.pkl"
    if path.exists():
        frame, categorical = pd.read_pickle(path)
        frame.attrs["seconds"] = 0.0
        return frame, categorical, path, True
    frame, categorical = builder.build(nodes, labels, include_activity=True)
    temporary = path.with_suffix(".tmp")
    pd.to_pickle((frame, categorical), temporary)
    temporary.replace(path)
    append_artifact_registry(cache, f"relational_features_{tag}", path, "Causally censored demographic, graph, and activity feature matrix.")
    return frame, categorical, path, False


def components_path(cache: Path, pool: pd.DataFrame, held: pd.DataFrame | None, tag: str, iterations: tuple[int, int], seeds: list[int]) -> Path:
    extra = f"components|{tag}|{iterations}|{seeds}"
    if held is not None:
        extra += labels_key(held, "held")
    return cache / f"dual_components_{labels_key(pool, extra)}.npz"


def booster_features(frame: pd.DataFrame) -> pd.DataFrame:
    prefixes = ("own_att_", "friend_att_", "own_interest_", "friend_interest_")
    selected = [column for column in frame.columns if not column.startswith(prefixes)]
    return frame[selected]


def prediction_by_graph(graph, users: pd.DataFrame, prediction: np.ndarray) -> np.ndarray:
    output = np.full(len(graph.ids), float(np.nanmean(prediction)), dtype=np.float64)
    for uid, value in zip(users["user_id"].to_numpy(), prediction):
        pos = graph.id_to_pos.get(int(uid))
        if pos is not None:
            output[pos] = float(value)
    return output


def select_clip(y: np.ndarray, pred: np.ndarray, fold_ids: np.ndarray) -> tuple[bool, dict]:
    valid = np.isfinite(pred)
    lower, upper = np.quantile(y, [0.001, 0.999])
    clipped = np.clip(pred, lower, upper)
    fold_results = {}
    changes = []
    for fold in np.unique(fold_ids[valid]):
        mask = valid & (fold_ids == fold)
        before = r2_score(y[mask], pred[mask])
        after = r2_score(y[mask], clipped[mask])
        fold_results[str(int(fold))] = after - before
        changes.append(after - before)
    accepted = bool(changes and np.mean(changes) > 0 and changes[-1] >= -0.002)
    return accepted, {"accepted": accepted, "lower": float(lower), "upper": float(upper), "fold_improvements": fold_results}


def sequential_soft_oof(
    labels: pd.DataFrame,
    base_oof: np.ndarray,
    soft_oof: np.ndarray,
    fold_ids: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, object | None, bool, dict]:
    candidate = base_oof.copy()
    usable_folds = sorted(int(v) for v in np.unique(fold_ids[fold_ids >= 0]))
    for current in usable_folds[1:]:
        earlier = (fold_ids >= 0) & (fold_ids < current) & np.isfinite(base_oof) & np.isfinite(soft_oof).all(axis=1)
        held = (fold_ids == current) & np.isfinite(base_oof) & np.isfinite(soft_oof).all(axis=1)
        if earlier.sum() < 200 or held.sum() == 0:
            continue
        train_x = np.column_stack([base_oof[earlier], soft_oof[earlier]])
        residual = labels.loc[earlier, "birthyear"].to_numpy(dtype=float) - base_oof[earlier]
        model = shallow_residual_fit(train_x, residual, iterations)
        correction = model.predict(np.column_stack([base_oof[held], soft_oof[held]]))
        gate = soft_oof[held, SOFT_NAMES.index("soft_gate")] * soft_oof[held, SOFT_NAMES.index("soft_any_friend")]
        candidate[held] = base_oof[held] + gate * correction
    gate_rows = (fold_ids >= usable_folds[1] if len(usable_folds) > 1 else False) & np.isfinite(candidate)
    accepted, diagnostics = improvement_gate(
        labels.loc[gate_rows, "birthyear"].to_numpy(dtype=float),
        base_oof[gate_rows],
        candidate[gate_rows],
        fold_ids[gate_rows],
    ) if np.any(gate_rows) else (False, {"accepted": False, "reason": "insufficient_folds"})
    all_rows = (fold_ids >= 0) & np.isfinite(base_oof) & np.isfinite(soft_oof).all(axis=1)
    final_model = None
    if accepted and all_rows.sum() >= 200:
        final_model = shallow_residual_fit(
            np.column_stack([base_oof[all_rows], soft_oof[all_rows]]),
            labels.loc[all_rows, "birthyear"].to_numpy(dtype=float) - base_oof[all_rows],
            iterations,
        )
    if not accepted:
        candidate = base_oof.copy()
    return candidate, final_model, accepted, diagnostics


def apply_soft(model, base: np.ndarray, features: pd.DataFrame, accepted: bool) -> np.ndarray:
    if not accepted or model is None:
        return base.copy()
    matrix = features[SOFT_NAMES].to_numpy(dtype=float)
    correction = model.predict(np.column_stack([base, matrix]))
    gate = matrix[:, SOFT_NAMES.index("soft_gate")] * matrix[:, SOFT_NAMES.index("soft_any_friend")]
    return base + gate * correction


def fit_pool_soft_model(
    labels: pd.DataFrame,
    base_oof: np.ndarray,
    soft_oof: np.ndarray,
    base_extra: np.ndarray | None,
    soft_extra: np.ndarray | None,
    y_extra: np.ndarray | None,
    accepted: bool,
    iterations: int,
):
    if not accepted:
        return None
    valid = np.isfinite(base_oof) & np.isfinite(soft_oof).all(axis=1)
    matrices = [np.column_stack([base_oof[valid], soft_oof[valid]])]
    residuals = [labels.loc[valid, "birthyear"].to_numpy(dtype=float) - base_oof[valid]]
    if base_extra is not None and soft_extra is not None and y_extra is not None:
        matrices.append(np.column_stack([base_extra, soft_extra]))
        residuals.append(y_extra - base_extra)
    return shallow_residual_fit(np.vstack(matrices), np.concatenate(residuals), iterations)


def slice_diagnostics(labels: pd.DataFrame, pred: np.ndarray, feature_frame: pd.DataFrame, fold_ids: np.ndarray | None = None) -> dict:
    y = labels["birthyear"].to_numpy(dtype=float)
    covered = feature_frame["friend_labeled_count"].to_numpy(dtype=float) > 0
    any_friend = feature_frame["eligible_degree"].to_numpy(dtype=float) > 0
    strata = {
        "hard_friend": covered,
        "friend_no_hard": any_friend & ~covered,
        "isolated": ~any_friend,
        "target_pre_1945": y < 1945,
        "target_1945_2000": (y >= 1945) & (y <= 2000),
        "target_post_2000": y > 2000,
    }
    output = {}
    for name, mask in strata.items():
        scored = mask & np.isfinite(pred)
        if scored.sum() >= 2:
            output[name] = {"count": int(scored.sum()), "r2": r2_score(y[scored], pred[scored]), "mae": float(np.abs(y[scored] - pred[scored]).mean())}
        else:
            output[name] = {"count": int(scored.sum())}
    if fold_ids is not None:
        output["oof_total"] = {"count": int(np.isfinite(pred).sum()), "r2": r2_score(y[np.isfinite(pred)], pred[np.isfinite(pred)])}
    return output


def main() -> None:
    started = time.time()
    debug = is_debug()
    cache = shared_cache_dir()
    table_info, feature_history = ensure_living_documents(cache)
    ctx = load_task(upto_test_timestamp=False)
    train = immutable_table(ctx.train, True)
    val = immutable_table(ctx.val, True)
    test = immutable_table(ctx.test, False)
    if len(val) == 0:
        center = float(train["birthyear"].median())
        np.save(run_data_dir() / "test_predictions.npy", np.full(len(test), center, dtype=float))
        phase("rolling-output", started, f"test={len(test)}")
        return
    phase("load", started, f"train={len(train)} val={len(val)} test={len(test)} debug={debug}")
    labels_a = debug_labels(train) if debug else train.copy()
    users = user_frame(ctx.db.table_dict["users"].df)
    all_nodes = users[["user_id", "joinedAt"]].copy().reset_index(drop=True)
    all_nodes["_row_index"] = np.arange(len(all_nodes), dtype=np.int64)
    graph = build_graph(users, ctx.db.table_dict["user_friends"].df, cache)
    graph_path = cache / f"strict_friend_graph_{version_key()}.npz"
    append_artifact_registry(cache, "strict_friend_graph", graph_path, "Deduplicated strict undirected edge list and compact adjacency arrays.")
    phase("strict-graph", started, f"edges={len(graph.edges)}")
    attendee, interest = build_activity_stores(
        ctx.db.table_dict["event_attendees"].df,
        ctx.db.table_dict["event_interest"].df,
        ctx.db.table_dict["events"].df,
    )
    builder = FeatureBuilder(users, graph, attendee, interest, int(pd.to_datetime(train["joinedAt"]).astype("int64").min()))
    phase("activity-index", started, f"attendee_users={len(attendee.times)} interest_users={len(interest.times)}")
    node_positions = pd.Series(np.arange(len(all_nodes), dtype=np.int64), index=all_nodes["user_id"].to_numpy())
    fold_specs = [
        (pd.Timestamp("2012-10-30"), pd.Timestamp("2012-11-06")),
        (pd.Timestamp("2012-11-06"), pd.Timestamp("2012-11-13")),
        (pd.Timestamp("2012-11-13"), pd.Timestamp("2012-11-21")),
    ]
    if debug:
        fold_specs = fold_specs[-1:]
    seeds = [17] if debug else [17, 43]
    max_iterations = (50, 50) if debug else (1200, 1200)
    residual_iterations = 30 if debug else 180
    oof_components = np.full((len(labels_a), 2), np.nan, dtype=np.float64)
    fold_ids = np.full(len(labels_a), -1, dtype=np.int16)
    fold_records = []
    light_best = []
    cat_best = []
    for fold_index, (cutoff, end) in enumerate(fold_specs):
        pool_mask = labels_a["joinedAt"] < cutoff
        held_mask = (labels_a["joinedAt"] >= cutoff) & (labels_a["joinedAt"] < end)
        pool = labels_a.loc[pool_mask].copy().reset_index(drop=True)
        held = labels_a.loc[held_mask].copy().reset_index(drop=True)
        if len(pool) < 500 or len(held) < 20:
            continue
        node_features, categorical, _, feature_cached = cached_features(builder, all_nodes, pool, cache, f"fold_{fold_index}")
        model_features = booster_features(node_features)
        pool_pos = positions_for(pool, node_positions)
        held_pos = positions_for(held, node_positions)
        component_path = components_path(cache, pool, held, f"fold_{fold_index}_hybrid_selector", max_iterations, seeds)
        if component_path.exists():
            cached = np.load(component_path, allow_pickle=False)
            node_components = cached["components"]
            best = (cached["light_best"].astype(int).tolist(), cached["cat_best"].astype(int).tolist())
            component_cached = True
        else:
            model, best = fit_dual_model(
                model_features.iloc[pool_pos],
                pool["birthyear"].to_numpy(dtype=float),
                categorical,
                max_iterations,
                seeds,
                model_features.iloc[held_pos],
                held["birthyear"].to_numpy(dtype=float),
                CATBOOST_DROP_PREFIXES,
            )
            node_components = model.predict_components(model_features)
            temporary = component_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, components=node_components, light_best=np.asarray(best[0]), cat_best=np.asarray(best[1]))
            temporary.replace(component_path)
            append_artifact_registry(cache, f"dual_components_fold_{fold_index}", component_path, "Per-node causal LightGBM and CatBoost forward-fold predictions.")
            component_cached = False
        held_indices = np.flatnonzero(held_mask)
        oof_components[held_indices] = node_components[held_pos]
        fold_ids[held_indices] = fold_index
        light_best.extend(best[0])
        cat_best.extend(best[1])
        fold_records.append({"index": fold_index, "pool": pool, "held": held, "held_indices": held_indices, "node_components": node_components})
        phase("forward-fold", started, f"fold={fold_index + 1} pool={len(pool)} held={len(held)} features={node_features.shape[1]} feature_cache={feature_cached} component_cache={component_cached}")
    ensemble_weights, ensemble_diagnostics = choose_ensemble(oof_components, labels_a["birthyear"].to_numpy(dtype=float), fold_ids)
    base_oof = np.where(np.isfinite(oof_components).all(axis=1), oof_components @ ensemble_weights, np.nan)
    median_iterations = (
        int(np.median(light_best)) if light_best else max_iterations[0],
        int(np.median(cat_best)) if cat_best else max_iterations[1],
    )
    soft_oof = np.full((len(labels_a), len(SOFT_NAMES)), np.nan, dtype=np.float64)
    node_blends = {}
    for record in fold_records:
        node_prediction = record["node_components"] @ ensemble_weights
        node_blends[record["index"]] = node_prediction
        graph_prediction = prediction_by_graph(graph, users, node_prediction)
        soft = soft_neighbor_features(record["held"], graph, graph_prediction, record["pool"])
        soft_oof[record["held_indices"]] = soft.to_numpy(dtype=float)
    soft_candidate_oof, soft_model_a, soft_accepted, soft_diagnostics = sequential_soft_oof(
        labels_a, base_oof, soft_oof, fold_ids, residual_iterations
    )
    phase("oof-soft-stage", started, f"weights={ensemble_weights.tolist()} accepted={soft_accepted} iterations={median_iterations}")
    cs_candidate_oof = soft_candidate_oof.copy()
    cs_rows = []
    cs_values = []
    cs_folds = []
    for record in fold_records[1:]:
        pool = record["pool"]
        held = record["held"]
        pool_lookup = pd.Series(soft_candidate_oof, index=labels_a["user_id"].to_numpy())
        pool_oof = pool_lookup.reindex(pool["user_id"].to_numpy()).to_numpy(dtype=float)
        graph_prediction = prediction_by_graph(graph, users, node_blends[record["index"]])
        cs = correct_and_smooth_features(graph, pool, pool_oof, graph_prediction, held)
        held_idx = record["held_indices"]
        valid = np.isfinite(soft_candidate_oof[held_idx])
        candidate = soft_candidate_oof[held_idx] + cs["cs_correction"].to_numpy(dtype=float)
        cs_candidate_oof[held_idx[valid]] = candidate[valid]
        cs_rows.extend(held_idx[valid].tolist())
        cs_values.extend(candidate[valid].tolist())
        cs_folds.extend([record["index"]] * int(valid.sum()))
    if cs_rows:
        rows = np.asarray(cs_rows, dtype=int)
        cs_accepted, cs_diagnostics = improvement_gate(
            labels_a.loc[rows, "birthyear"].to_numpy(dtype=float),
            soft_candidate_oof[rows],
            np.asarray(cs_values, dtype=float),
            np.asarray(cs_folds, dtype=int),
        )
    else:
        cs_accepted, cs_diagnostics = False, {"accepted": False, "reason": "insufficient_forward_residuals"}
    if not cs_accepted:
        cs_candidate_oof = soft_candidate_oof.copy()
    clip_accepted, clip_diagnostics = select_clip(labels_a["birthyear"].to_numpy(dtype=float), cs_candidate_oof, fold_ids)
    phase("oof-graph-gates", started, f"cs={cs_accepted} clip={clip_accepted}")
    features_a, categorical, _, feature_a_cached = cached_features(builder, all_nodes, labels_a, cache, "model_a")
    model_features_a = booster_features(features_a)
    train_pos_a = positions_for(labels_a, node_positions)
    component_a_path = components_path(cache, labels_a, None, "model_a_hybrid_selector", median_iterations, seeds)
    if component_a_path.exists():
        components_a = np.load(component_a_path, allow_pickle=False)["components"]
        component_a_cached = True
    else:
        final_a, _ = fit_dual_model(
            model_features_a.iloc[train_pos_a],
            labels_a["birthyear"].to_numpy(dtype=float),
            categorical,
            median_iterations,
            seeds,
            catboost_drop_prefixes=CATBOOST_DROP_PREFIXES,
        )
        components_a = final_a.predict_components(model_features_a)
        temporary = component_a_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, components=components_a)
        temporary.replace(component_a_path)
        append_artifact_registry(cache, "dual_components_model_a", component_a_path, "Per-node Model A LightGBM and CatBoost predictions.")
        component_a_cached = False
    node_base_a = components_a @ ensemble_weights
    graph_base_a = prediction_by_graph(graph, users, node_base_a)
    soft_all_a = soft_neighbor_features(all_nodes, graph, graph_base_a, labels_a)
    node_soft_a = apply_soft(soft_model_a, node_base_a, soft_all_a, soft_accepted)
    val_pos = positions_for(val, node_positions)
    val_base = node_soft_a[val_pos]
    cs_val = correct_and_smooth_features(graph, labels_a, soft_candidate_oof, prediction_by_graph(graph, users, node_soft_a), val)
    val_prediction = val_base + cs_val["cs_correction"].to_numpy(dtype=float) if cs_accepted else val_base
    if clip_accepted:
        limits_a = np.quantile(labels_a["birthyear"].to_numpy(dtype=float), [0.001, 0.999])
        val_prediction = np.clip(val_prediction, limits_a[0], limits_a[1])
    val_prediction_frozen = np.asarray(val_prediction, dtype=np.float64).copy()
    phase("model-a-frozen", started, f"val={len(val_prediction_frozen)} feature_cache={feature_a_cached} component_cache={component_a_cached}")
    labels_b = pd.concat([train, val], ignore_index=True).sort_values("joinedAt", kind="stable").reset_index(drop=True)
    if debug:
        labels_b = pd.concat([labels_a, val], ignore_index=True).sort_values("joinedAt", kind="stable").reset_index(drop=True)
    features_b, categorical_b, _, feature_b_cached = cached_features(builder, all_nodes, labels_b, cache, "model_b")
    model_features_b = booster_features(features_b)
    pool_pos_b = positions_for(labels_b, node_positions)
    component_b_path = components_path(cache, labels_b, None, "model_b_hybrid_selector", median_iterations, seeds)
    if component_b_path.exists():
        components_b = np.load(component_b_path, allow_pickle=False)["components"]
        component_b_cached = True
    else:
        final_b, _ = fit_dual_model(
            model_features_b.iloc[pool_pos_b],
            labels_b["birthyear"].to_numpy(dtype=float),
            categorical_b,
            median_iterations,
            seeds,
            catboost_drop_prefixes=CATBOOST_DROP_PREFIXES,
        )
        components_b = final_b.predict_components(model_features_b)
        temporary = component_b_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, components=components_b)
        temporary.replace(component_b_path)
        append_artifact_registry(cache, "dual_components_model_b", component_b_path, "Per-node Model B LightGBM and CatBoost predictions.")
        component_b_cached = False
    node_base_b = components_b @ ensemble_weights
    graph_base_b = prediction_by_graph(graph, users, node_base_b)
    soft_all_b = soft_neighbor_features(all_nodes, graph, graph_base_b, labels_b)
    train_oof_lookup = pd.Series(base_oof, index=labels_a["user_id"].to_numpy())
    train_soft_lookup = {int(uid): soft_oof[i] for i, uid in enumerate(labels_a["user_id"].to_numpy()) if np.isfinite(soft_oof[i]).all()}
    b_oof = np.full(len(labels_b), np.nan, dtype=float)
    b_soft = np.full((len(labels_b), len(SOFT_NAMES)), np.nan, dtype=float)
    for i, uid in enumerate(labels_b["user_id"].to_numpy()):
        if uid in train_oof_lookup.index:
            b_oof[i] = float(train_oof_lookup.loc[uid])
        if int(uid) in train_soft_lookup:
            b_soft[i] = train_soft_lookup[int(uid)]
    val_soft_a = soft_all_a.iloc[val_pos][SOFT_NAMES].to_numpy(dtype=float)
    soft_model_b = fit_pool_soft_model(
        labels_b,
        b_oof,
        b_soft,
        val_base,
        val_soft_a,
        val["birthyear"].to_numpy(dtype=float),
        soft_accepted,
        residual_iterations,
    )
    node_soft_b = apply_soft(soft_model_b, node_base_b, soft_all_b, soft_accepted)
    test_pos = positions_for(test, node_positions)
    test_base = node_soft_b[test_pos]
    val_oof_by_id = pd.Series(val_prediction_frozen, index=val["user_id"].to_numpy())
    b_oof_for_cs = np.full(len(labels_b), np.nan, dtype=float)
    train_soft_candidate_lookup = pd.Series(soft_candidate_oof, index=labels_a["user_id"].to_numpy())
    for i, uid in enumerate(labels_b["user_id"].to_numpy()):
        if uid in train_soft_candidate_lookup.index:
            b_oof_for_cs[i] = float(train_soft_candidate_lookup.loc[uid])
        elif uid in val_oof_by_id.index:
            b_oof_for_cs[i] = float(val_oof_by_id.loc[uid])
    cs_test = correct_and_smooth_features(graph, labels_b, b_oof_for_cs, prediction_by_graph(graph, users, node_soft_b), test)
    test_prediction = test_base + cs_test["cs_correction"].to_numpy(dtype=float) if cs_accepted else test_base
    if clip_accepted:
        limits_b = np.quantile(labels_b["birthyear"].to_numpy(dtype=float), [0.001, 0.999])
        test_prediction = np.clip(test_prediction, limits_b[0], limits_b[1])
    test_prediction = np.asarray(test_prediction, dtype=np.float64)
    diagnostics = {
        "version": version_key(),
        "debug": debug,
        "elapsed_seconds": time.time() - started,
        "ensemble": ensemble_diagnostics,
        "soft_gate": soft_diagnostics,
        "correct_and_smooth_gate": cs_diagnostics,
        "clip_gate": clip_diagnostics,
        "median_iterations": list(median_iterations),
        "graph_edges": int(len(graph.edges)),
        "oof_slices": slice_diagnostics(labels_a, cs_candidate_oof, features_a.iloc[train_pos_a].reset_index(drop=True), fold_ids),
        "validation_reporting_only": slice_diagnostics(val, val_prediction_frozen, features_a.iloc[val_pos].reset_index(drop=True)),
        "validation_reporting_only_r2": r2_score(val["birthyear"].to_numpy(dtype=float), val_prediction_frozen),
    }
    save_predictions(val_prediction_frozen, test_prediction)
    (run_data_dir() / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    append_text(
        feature_history,
        f"\n### staged causal graph boosting {version_key()} ({'debug' if debug else 'full'})\n"
        f"- run/experiment: generic_exp_0 | status: TESTED-KEPT\n"
        f"- what: demographic and causal encodings, strict hard-friend and two-hop labels, friend demographics, own/friend attendance and interest projections, dual boosting, soft neighbors, and gated Correct & Smooth\n"
        f"- outcome: internal soft accepted={soft_accepted}, C&S accepted={cs_accepted}, clipping accepted={clip_accepted}; OOF slices and fold metrics saved in metrics.json\n"
        f"- takeaway: graph stages are admitted only by expanding forward-fold pooled-standard-error gates; validation is reporting-only\n",
    )
    append_text(
        table_info,
        f"\n### generic_exp_0 strict extraction {version_key()}\n"
        f"- Strict complete non-self friendship rows deduplicate to {len(graph.edges):,} undirected edges; eligibility additionally requires both endpoint joinedAt values not later than the seed.\n"
        f"- Complete activity indices cover {len(attendee.times):,} attendee users and {len(interest.times):,} interest users; every prefix query is censored at the exact seed timestamp.\n",
    )
    phase("output", started, f"val_r2_reporting_only={diagnostics['validation_reporting_only_r2']:.6f} model_b_feature_cache={feature_b_cached} model_b_component_cache={component_b_cached}")


if __name__ == "__main__":
    main()
