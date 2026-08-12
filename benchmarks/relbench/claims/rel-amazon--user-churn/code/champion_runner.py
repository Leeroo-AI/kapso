from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def load_modules():
    code = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / "champion" / "code"
    sys.path.insert(0, str(code))
    import renewal
    import temporal_gnn

    return renewal, temporal_gnn


def label_vector(labels: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    return labels.set_index("row_id")["churn"].loc[frame["row_id"]].to_numpy(np.int8)


def blended_oof(renewal_module, y, origins, renewal_values, graph_values, mask, weight):
    result = np.full(len(y), np.nan, dtype=np.float32)
    for fold in renewal_module.OOF_ORIGIN_INDICES:
        selected = mask & (origins == fold)
        if not np.any(selected):
            continue
        result[selected] = (
            (1.0 - weight) * renewal_module.rank_values(renewal_values[selected])
            + weight * renewal_module.rank_values(graph_values[selected])
        ).astype(np.float32)
    return result


def cache_path(debug: bool) -> Path:
    suffix = "debug" if debug else "full"
    root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / "lane2_champion_reproduction_v1"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"predictions_{suffix}.npz"


def run_champion(debug: bool, started: float) -> dict:
    path = cache_path(debug)
    if path.exists():
        values = np.load(path, allow_pickle=False)
        print(f"[lane2] champion_cache=1 path={path.name}", flush=True)
        return {name: values[name] for name in values.files}
    renewal, graph = load_modules()
    train_paths, _ = renewal.ensure_features("train", debug=debug)
    train_frame = renewal.load_feature_frame(train_paths)
    labels = renewal.labels_frame()
    y_train = label_vector(labels, train_frame)
    train_frame, _, _ = renewal.add_causal_relational_targets(train_frame, y_train)
    train_frame = renewal.add_renewal_transforms(train_frame)
    train_frame, = renewal.add_customer_name_features(train_frame)
    columns = renewal.feature_columns(train_frame)
    renewal_oof, renewal_mask, renewal_folds = renewal.expanding_oof(train_frame, labels, columns, debug=debug)
    graph_oof, graph_mask, graph_folds = graph.expanding_graph_oof(train_frame, labels, debug=debug)
    overlap = renewal_mask & graph_mask
    blend_weight, blend_results = renewal.choose_rank_blend(
        y_train,
        train_frame["origin_index"].to_numpy(),
        renewal_oof,
        graph_oof,
        overlap,
    )
    champion_oof = blended_oof(
        renewal,
        y_train,
        train_frame["origin_index"].to_numpy(),
        renewal_oof,
        graph_oof,
        overlap,
        blend_weight,
    )
    if debug:
        val = np.full(409792, 0.624, dtype=np.float32)
        test = np.full(351885, 0.624, dtype=np.float32)
    else:
        val_paths, _ = renewal.ensure_features("val")
        test_paths, _ = renewal.ensure_features("test")
        val_frame = renewal.load_feature_frame(val_paths)
        test_frame = renewal.load_feature_frame(test_paths)
        import duckdb

        val_labels = duckdb.sql(f"SELECT churn FROM read_parquet('{renewal.split_path('val')}')").fetchnumpy()["churn"].astype(np.int8)
        train_frame, val_frame, test_frame = renewal.add_causal_relational_targets(train_frame, y_train, val_frame, val_labels, test_frame)
        train_frame = renewal.add_renewal_transforms(train_frame)
        val_frame = renewal.add_renewal_transforms(val_frame)
        test_frame = renewal.add_renewal_transforms(test_frame)
        train_frame, val_frame, test_frame = renewal.add_customer_name_features(train_frame, val_frame, test_frame)
        columns = renewal.feature_columns(train_frame)
        model_a = renewal.train_model(
            renewal.matrix(train_frame, columns),
            y_train,
            renewal.temporal_weights(train_frame["origin_index"].to_numpy()),
            rounds=520,
        )
        val = model_a.predict_proba(renewal.matrix(val_frame, columns))[:, 1].astype(np.float32)
        val_for_b = val_frame.copy()
        val_for_b["origin_index"] = int(train_frame["origin_index"].max()) + 1
        train_b = pd.concat([train_frame, val_for_b], ignore_index=True)
        y_b = np.concatenate([y_train, val_labels])
        model_b = renewal.train_model(
            renewal.matrix(train_b, columns),
            y_b,
            renewal.temporal_weights(train_b["origin_index"].to_numpy()),
            rounds=520,
        )
        test = model_b.predict_proba(renewal.matrix(test_frame, columns))[:, 1].astype(np.float32)
        if blend_weight > 0:
            val_frame.attrs["timestamp"] = "2015-10-01 00:00:00"
            test_frame.attrs["timestamp"] = "2016-01-01 00:00:00"
            val_graph = graph.fit_graph_final(train_frame, y_train, val_frame)
            test_graph = graph.fit_graph_final(train_b, y_b, test_frame)
            val = ((1.0 - blend_weight) * renewal.rank_values(val) + blend_weight * renewal.rank_values(val_graph)).astype(np.float32)
            test = ((1.0 - blend_weight) * renewal.rank_values(test) + blend_weight * renewal.rank_values(test_graph)).astype(np.float32)
    metadata = np.array(
        [
            blend_weight,
            len(renewal_folds),
            len(graph_folds),
            time.time() - started,
        ],
        dtype=np.float64,
    )
    temporary = path.with_name(f"predictions_{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        oof=champion_oof,
        oof_mask=overlap,
        val=val,
        test=test,
        metadata=metadata,
    )
    os.replace(temporary, path)
    print(f"[lane2] champion_reproduced blend_weight={blend_weight:.2f} val_rows={len(val)} test_rows={len(test)}", flush=True)
    return {"oof": champion_oof, "oof_mask": overlap, "val": val, "test": test, "metadata": metadata}
