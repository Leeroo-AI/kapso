import gc
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .cache import SCHEMA_VERSION, cache_root, content_hash, register_artifact


def _map_rows(predictions, truths):
    values = []
    for prediction, truth in zip(predictions, truths):
        relevant = set(map(int, truth))
        hits = np.array([int(int(x) in relevant) for x in prediction[:10]], dtype=np.float32)
        precision = np.cumsum(hits) / np.arange(1, 11)
        values.append(float((precision * hits).sum() / min(max(len(relevant), 1), 10)))
    return np.asarray(values, dtype=np.float32)


def slice_metrics(predictions, rows, snapshot):
    truths = rows["sponsor_id"].tolist()
    aps = _map_rows(predictions, truths)
    sizes = np.asarray([len(x) for x in truths])
    output = {"overall": {"count": int(len(rows)), "map": float(aps.mean())}}
    cardinality = {
        "size_1": sizes == 1,
        "size_2_4": (sizes >= 2) & (sizes <= 4),
        "size_5_10": (sizes >= 5) & (sizes <= 10),
        "size_11_50": (sizes >= 11) & (sizes <= 50),
        "size_over_50": sizes > 50,
    }
    for name, mask in cardinality.items():
        output[name] = {"count": int(mask.sum()), "map": float(aps[mask].mean()) if mask.any() else 0.0}
    cold = []
    repeat = []
    for condition_id, truth in zip(rows["condition_id"].to_numpy(), truths):
        historical = set(snapshot.pair_count.getrow(int(condition_id)).indices.tolist())
        cold.append(len(historical) == 0)
        repeat.append(bool(historical.intersection(map(int, truth))))
    cold = np.asarray(cold)
    repeat = np.asarray(repeat)
    for name, mask in {
        "cold_condition": cold,
        "warm_condition": ~cold,
        "has_repeat_positive": repeat,
        "unseen_only_positive": ~repeat,
    }.items():
        output[name] = {"count": int(mask.sum()), "map": float(aps[mask].mean()) if mask.any() else 0.0}
    return output


def candidate_recall(records, rows):
    totals = {100: 0, 500: 0, 2000: 0, 4000: 0}
    denominator = 0
    pools = []
    als_hits = 0
    semantic_hits = 0
    for row in rows.itertuples(index=False):
        truth = set(map(int, row.sponsor_id))
        record = records[int(row.condition_id)]
        candidates = record.candidates
        denominator += len(truth)
        pools.append(len(candidates))
        for depth in totals:
            totals[depth] += len(truth.intersection(candidates[:depth].tolist()))
        als_hits += len(truth.intersection(candidates[record.source_features[:, 7] > 0].tolist()))
        semantic_hits += len(truth.intersection(candidates[record.source_features[:, 8] > 0].tolist()))
    return {
        f"recall_{depth}": float(value / max(denominator, 1)) for depth, value in totals.items()
    } | {
        "mean_pool": float(np.mean(pools)),
        "positive_count": int(denominator),
        "als_source_recall": float(als_hits / max(denominator, 1)),
        "semantic_source_recall": float(semantic_hits / max(denominator, 1)),
    }


def _source_selection(snapshot, record, truth, seed):
    truth_set = set(map(int, truth))
    selected = []

    def take(scores, count, require=None):
        order = np.argsort(-scores, kind="stable")
        taken = 0
        for index in order:
            sponsor = int(record.candidates[index])
            if sponsor in truth_set or sponsor in selected:
                continue
            if require is not None and not require(index):
                continue
            selected.append(sponsor)
            taken += 1
            if taken >= count:
                break

    take(record.baseline_components[:, 0], 60, lambda index: record.baseline_components[index, 0] > 0)
    take(record.source_features[:, 0], 60)
    take(record.source_features[:, 5], 45)
    take(record.source_features[:, 6], 60)
    take(record.source_features[:, 4], 45)
    rng = np.random.default_rng(seed)
    random_candidates = rng.integers(0, snapshot.n_sponsors, size=180)
    for sponsor in random_candidates:
        value = int(sponsor)
        if value not in truth_set and value not in selected:
            selected.append(value)
        if len(selected) >= 300:
            break
    for sponsor in record.candidates:
        value = int(sponsor)
        if value not in truth_set and value not in selected:
            selected.append(value)
        if len(selected) >= 300:
            break
    candidates = np.asarray(list(dict.fromkeys(list(map(int, truth)) + selected[:300])), dtype=np.int64)
    positions = {int(value): index for index, value in enumerate(record.candidates)}
    sources = np.zeros((len(candidates), record.source_features.shape[1]), dtype=np.float32)
    for index, sponsor in enumerate(candidates):
        position = positions.get(int(sponsor))
        if position is not None:
            sources[index] = record.source_features[position]
        else:
            sources[index, 6] = snapshot.sponsor_popularity[int(sponsor)]
            if snapshot.portfolio is not None:
                query = np.asarray(snapshot.query_embeddings[record.condition_id], dtype=np.float32)
                sources[index, 5] = float(np.asarray(snapshot.portfolio[int(sponsor)], dtype=np.float32) @ query)
    labels = np.asarray([int(int(x) in truth_set) for x in candidates], dtype=np.uint8)
    return candidates, sources, labels


def build_origin_dataset(snapshot, records, rows, debug=False):
    origin = str(snapshot.cutoff.date())
    key = content_hash([SCHEMA_VERSION, "origin", origin, len(rows), int(debug), len(snapshot.feature_names())])
    root = cache_root() / f"origin_{key}"
    x_path = root / "features.npy"
    y_path = root / "labels.npy"
    group_path = root / "groups.npy"
    meta_path = root / "meta.json"
    if all(path.exists() for path in [x_path, y_path, group_path, meta_path]):
        return {
            "x": np.load(x_path, mmap_mode="r"),
            "y": np.load(y_path, mmap_mode="r"),
            "groups": np.load(group_path, mmap_mode="r"),
            "meta": json.loads(meta_path.read_text()),
        }
    root.mkdir(parents=True, exist_ok=True)
    selections = []
    total = 0
    limit = min(len(rows), 120 if debug else len(rows))
    active_rows = rows.iloc[:limit]
    for row_index, row in enumerate(active_rows.itertuples(index=False)):
        record = records[int(row.condition_id)]
        candidates, sources, labels = _source_selection(
            snapshot, record, row.sponsor_id, 1337 + row_index + snapshot.cutoff.year * 10000
        )
        selections.append((int(row.condition_id), candidates, sources, labels))
        total += len(candidates)
    feature_count = len(snapshot.feature_names())
    features = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float16, shape=(total, feature_count))
    labels = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.uint8, shape=(total,))
    groups = np.lib.format.open_memmap(group_path, mode="w+", dtype=np.int32, shape=(len(selections),))
    offset = 0
    for group_index, (condition_id, candidates, sources, target) in enumerate(selections):
        block = snapshot.feature_block(condition_id, candidates, sources)
        block = np.clip(block, -65000, 65000)
        end = offset + len(block)
        features[offset:end] = block.astype(np.float16)
        labels[offset:end] = target
        groups[group_index] = len(block)
        offset = end
    features.flush()
    labels.flush()
    groups.flush()
    metadata = {
        "origin": origin,
        "groups": int(len(selections)),
        "rows": int(total),
        "features": int(feature_count),
        "feature_names": snapshot.feature_names(),
    }
    meta_path.write_text(json.dumps(metadata))
    del features, labels, groups
    register_artifact(
        f"lane0 training matrix {origin}",
        x_path,
        "Cutoff-causal sampled LambdaMART matrix with positives and 300 hard negatives per group",
        key,
        "Run main.py after deleting this origin directory",
    )
    return {
        "x": np.load(x_path, mmap_mode="r"),
        "y": np.load(y_path, mmap_mode="r"),
        "groups": np.load(group_path, mmap_mode="r"),
        "meta": metadata,
    }


def load_cached_origin(origin, group_count, feature_count, debug=False):
    root = cache_root()
    for meta_path in root.glob("origin_*/meta.json"):
        try:
            metadata = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            metadata.get("origin") != str(origin)
            or metadata.get("groups") != int(group_count)
            or metadata.get("features") != int(feature_count)
        ):
            continue
        directory = meta_path.parent
        x_path = directory / "features.npy"
        y_path = directory / "labels.npy"
        group_path = directory / "groups.npy"
        if all(path.exists() for path in [x_path, y_path, group_path]):
            return {
                "x": np.load(x_path, mmap_mode="r"),
                "y": np.load(y_path, mmap_mode="r"),
                "groups": np.load(group_path, mmap_mode="r"),
                "meta": metadata,
            }
    return None


def combine(datasets):
    return (
        np.concatenate([np.asarray(x["x"]) for x in datasets]),
        np.concatenate([np.asarray(x["y"]) for x in datasets]),
        np.concatenate([np.asarray(x["groups"]) for x in datasets]),
    )


def train_ranker(train_sets, valid_set=None, debug=False, fixed_iterations=None):
    started = time.time()
    x_train, y_train, groups_train = combine(train_sets)
    train_data = lgb.Dataset(x_train, label=y_train, group=groups_train, free_raw_data=True)
    valid_data = None
    if valid_set is not None:
        x_valid = np.asarray(valid_set["x"])
        y_valid = np.asarray(valid_set["y"])
        groups_valid = np.asarray(valid_set["groups"])
        valid_data = lgb.Dataset(x_valid, label=y_valid, group=groups_valid, reference=train_data, free_raw_data=True)
    parameters = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10],
        "lambdarank_truncation_level": 10,
        "num_leaves": 63,
        "learning_rate": 0.04,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "label_gain": [0, 1],
        "verbosity": -1,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "seed": 1337,
        "feature_fraction_seed": 1337,
        "bagging_seed": 1337,
    }
    if fixed_iterations is not None:
        model = lgb.train(
            parameters,
            train_data,
            num_boost_round=int(fixed_iterations),
            callbacks=[lgb.log_evaluation(0)],
        )
    elif valid_data is not None:
        model = lgb.train(
            parameters,
            train_data,
            num_boost_round=50 if debug else 450,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(30 if not debug else 10, verbose=False), lgb.log_evaluation(0)],
        )
    else:
        model = lgb.train(
            parameters,
            train_data,
            num_boost_round=50 if debug else 250,
            callbacks=[lgb.log_evaluation(0)],
        )
    print(
        f"[ranker] groups={len(groups_train)} rows={len(y_train)} trees={model.best_iteration or model.current_iteration()} "
        f"trained in {time.time() - started:.1f}s"
    )
    del x_train, y_train, groups_train, train_data, valid_data
    if valid_set is not None:
        del x_valid, y_valid, groups_valid
    gc.collect()
    return model


def baseline_scores(record, variant):
    values = record.baseline_components
    normalized = np.zeros_like(values, dtype=np.float32)
    for column in range(values.shape[1]):
        scale = max(float(np.max(np.abs(values[:, column]))), 1e-8)
        normalized[:, column] = values[:, column] / scale
    repeated = (values[:, 0] > 0).astype(np.float32)
    if variant == 0:
        return 12 * repeated + 6 * normalized[:, 0] + 0.8 * normalized[:, 1] + 0.25 * normalized[:, 6]
    if variant == 1:
        return 10 * repeated + 5 * normalized[:, 0] + 1.2 * normalized[:, 2] + 0.5 * normalized[:, 3] + 0.3 * normalized[:, 6]
    return (
        10 * repeated
        + 5 * normalized[:, 0]
        + 1.5 * normalized[:, 1]
        + 0.5 * normalized[:, 2]
        + 0.4 * normalized[:, 3]
        + 0.2 * normalized[:, 4]
        + 0.2 * normalized[:, 5]
        + 0.1 * normalized[:, 6]
        + 0.2 * normalized[:, 7]
        + 0.2 * normalized[:, 8]
    )


def baseline_predictions(records, rows, variant, fallback):
    output = np.empty((len(rows), 10), dtype=np.int64)
    for index, condition_id in enumerate(rows["condition_id"].to_numpy()):
        record = records.get(int(condition_id))
        if record is None:
            output[index] = fallback[:10]
        else:
            order = np.argsort(-baseline_scores(record, variant), kind="stable")[:10]
            output[index] = record.candidates[order]
    return output


def rank_predictions(snapshot, records, rows, model, iterations, fallback, batch_groups=12):
    iteration_list = [int(x) for x in iterations]
    outputs = [np.empty((len(rows), 10), dtype=np.int64) for _ in iteration_list]
    for start in range(0, len(rows), batch_groups):
        batch = rows.iloc[start : start + batch_groups]
        blocks = []
        lengths = []
        candidate_rows = []
        active = []
        for local, condition_id in enumerate(batch["condition_id"].to_numpy()):
            record = records.get(int(condition_id))
            if record is None:
                for output in outputs:
                    output[start + local] = fallback[:10]
                continue
            block = snapshot.feature_block(int(condition_id), record.candidates, record.source_features)
            blocks.append(block)
            lengths.append(len(block))
            candidate_rows.append(record.candidates)
            active.append(local)
        if not blocks:
            continue
        matrix = np.vstack(blocks)
        predictions = [
            model.predict(matrix, num_iteration=iteration).astype(np.float32) for iteration in iteration_list
        ]
        offset = 0
        for block_index, length in enumerate(lengths):
            candidates = candidate_rows[block_index]
            for output, scores in zip(outputs, predictions):
                order = np.argsort(-scores[offset : offset + length], kind="stable")[:10]
                output[start + active[block_index]] = candidates[order]
            offset += length
    return outputs


def blended_predictions(
    snapshot,
    records,
    rows,
    model,
    iteration,
    weights,
    baseline_variant,
    fallback,
    batch_groups=12,
):
    outputs = [np.empty((len(rows), 10), dtype=np.int64) for _ in weights]
    for start in range(0, len(rows), batch_groups):
        batch = rows.iloc[start : start + batch_groups]
        blocks = []
        lengths = []
        candidate_rows = []
        record_rows = []
        active = []
        for local, condition_id in enumerate(batch["condition_id"].to_numpy()):
            record = records.get(int(condition_id))
            if record is None:
                for output in outputs:
                    output[start + local] = fallback[:10]
                continue
            blocks.append(snapshot.feature_block(int(condition_id), record.candidates, record.source_features))
            lengths.append(len(record.candidates))
            candidate_rows.append(record.candidates)
            record_rows.append(record)
            active.append(local)
        if not blocks:
            continue
        scores = model.predict(np.vstack(blocks), num_iteration=int(iteration)).astype(np.float32)
        offset = 0
        for block_index, length in enumerate(lengths):
            record = record_rows[block_index]
            baseline = baseline_scores(record, baseline_variant)
            baseline_order = np.argsort(-baseline, kind="stable")
            model_order = np.argsort(-scores[offset : offset + length], kind="stable")
            baseline_rank = np.empty(length, dtype=np.float32)
            model_rank = np.empty(length, dtype=np.float32)
            baseline_rank[baseline_order] = np.linspace(1.0, 0.0, length, dtype=np.float32)
            model_rank[model_order] = np.linspace(1.0, 0.0, length, dtype=np.float32)
            for output, weight in zip(outputs, weights):
                combined = baseline_rank + float(weight) * model_rank
                order = np.argsort(-combined, kind="stable")[:10]
                output[start + active[block_index]] = candidate_rows[block_index][order]
            offset += length
    return outputs


def validate_predictions(values, expected_rows, n_sponsors):
    array = np.asarray(values)
    if array.shape != (expected_rows, 10):
        raise RuntimeError(f"prediction shape {array.shape} != {(expected_rows, 10)}")
    if not np.issubdtype(array.dtype, np.integer):
        raise RuntimeError(f"prediction dtype {array.dtype} is not integer")
    if np.any(array < 0) or np.any(array >= n_sponsors):
        raise RuntimeError("prediction sponsor id is out of range")
    if any(len(set(map(int, row))) != 10 for row in array):
        raise RuntimeError("prediction rows contain duplicate sponsors")
