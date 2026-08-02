from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from relbench.datasets import get_dataset
from relbench.tasks import get_task

warnings.filterwarnings("ignore")

from factor_model import (
    CLASSES,
    FEATURE_NAMES,
    INITIAL_WEIGHTS,
    FeatureStore,
    HistoryModel,
    apply_document_exception_gate,
    build_design,
    build_gate_features,
    document_labels,
    document_exception_classes,
    fit_gate,
    fit_nonnegative_weights,
    probabilities,
    select_gate_c,
    select_gate_training_rows,
    select_regularization,
    stratified_sample,
)


ORIGINS = (
    ("2019-08-01", "2019-10-01"),
    ("2019-10-01", "2019-12-01"),
    ("2019-12-01", "2020-02-01"),
)

EXCEPTION_CANDIDATES = (
    ("item_org_product", 2, 0.8),
    ("item_org_product_recency", 2, 0.8),
    ("item_org_doctype_product_recency", 2, 0.8),
    ("item_ensemble", 2, 1.0),
    ("document_ensemble", 5, 1.0),
)


def load_inputs():
    dataset_name = os.environ["RELBENCH_DATASET"]
    task_name = os.environ["RELBENCH_TASK"]
    dataset = get_dataset(dataset_name, download=False)
    task = get_task(dataset_name, task_name, download=False)
    database = dataset.get_db(upto_test_timestamp=False)
    train = task.get_table("train", mask_input_cols=False).df.copy()
    validation = task.get_table("val", mask_input_cols=False).df.copy()
    test = task.get_table("test").df.copy()
    train["PLANT"] = train["PLANT"].astype(np.int16)
    validation["PLANT"] = validation["PLANT"].astype(np.int16)
    return task, database, train, validation, test


def prefix_arrays(frame: pd.DataFrame, cutoff: str):
    mask = frame["CREATIONTIMESTAMP"].to_numpy(dtype="datetime64[ns]") < np.datetime64(cutoff)
    selected = frame.loc[mask]
    return (
        selected["ID"].to_numpy(dtype=np.int64),
        selected["PLANT"].to_numpy(dtype=np.int64),
        selected["CREATIONTIMESTAMP"].to_numpy(dtype="datetime64[ns]"),
    )


def block_arrays(frame: pd.DataFrame, start: str, end: str):
    timestamps = frame["CREATIONTIMESTAMP"].to_numpy(dtype="datetime64[ns]")
    mask = (timestamps >= np.datetime64(start)) & (timestamps < np.datetime64(end))
    selected = frame.loc[mask]
    return (
        selected["ID"].to_numpy(dtype=np.int64),
        selected["PLANT"].to_numpy(dtype=np.int64),
        selected["CREATIONTIMESTAMP"].to_numpy(dtype="datetime64[ns]"),
    )


def gate_training_data(
    history: HistoryModel,
    ids: np.ndarray,
    labels: np.ndarray,
    limit: int,
):
    selected = select_gate_training_rows(history.store, ids, labels, limit)
    selected_ids = ids[selected]
    selected_labels = labels[selected]
    features = build_gate_features(history, selected_ids)
    documents, labels_by_document = document_labels(history.store, selected_ids, selected_labels)
    if not np.array_equal(features.documents, documents):
        raise RuntimeError("gate document alignment failed")
    return features, labels_by_document


def fit_history_gate(
    history: HistoryModel,
    ids: np.ndarray,
    labels: np.ndarray,
    c_value: float,
    limit: int,
):
    features, labels_by_document = gate_training_data(history, ids, labels, limit)
    gate = fit_gate(features, labels_by_document, c_value)
    print(
        f"[main] gate history={history.name} documents={len(labels_by_document)} "
        f"exceptions={int(labels_by_document.sum())} C={c_value:g}"
    )
    return gate


def select_gate(
    store: FeatureStore,
    train: pd.DataFrame,
    debug: bool,
):
    if debug:
        return 0.1, {"debug": [0.1]}
    datasets = []
    for start, end in ORIGINS:
        history_ids, history_labels, history_times = prefix_arrays(train, start)
        target_ids, target_labels, _ = block_arrays(train, start, end)
        history = HistoryModel(
            store,
            history_ids,
            history_labels,
            history_times,
            np.datetime64(start),
            f"gate_{start}",
        )
        train_features, train_document_labels = gate_training_data(
            history,
            history_ids,
            history_labels,
            250000,
        )
        test_features = build_gate_features(history, target_ids)
        test_documents, test_document_labels = document_labels(store, target_ids, target_labels)
        if not np.array_equal(test_features.documents, test_documents):
            raise RuntimeError("origin gate document alignment failed")
        datasets.append(
            (
                train_features,
                train_document_labels,
                test_features,
                test_document_labels,
                start,
            )
        )
        del history
    return select_gate_c(datasets)


def origin_designs(
    store: FeatureStore,
    train: pd.DataFrame,
    c_value: float,
    debug: bool,
):
    origins = ORIGINS[-1:] if debug else ORIGINS
    sampled = []
    exception_records = {
        f"{mode}:{support}:{purity:g}": []
        for mode, support, purity in EXCEPTION_CANDIDATES
    }
    exception_baselines = []
    maximum_rows = 25000 if debug else 40000
    for index, (start, end) in enumerate(origins):
        history_ids, history_labels, history_times = prefix_arrays(train, start)
        target_ids, target_labels, _ = block_arrays(train, start, end)
        history = HistoryModel(
            store,
            history_ids,
            history_labels,
            history_times,
            np.datetime64(start),
            f"origin_{start}",
        )
        gate = fit_history_gate(
            history,
            history_ids,
            history_labels,
            c_value,
            120000 if debug else 300000,
        )
        if debug and len(target_ids) > 25000:
            rng = np.random.default_rng(1337)
            chosen = np.sort(rng.choice(len(target_ids), 25000, replace=False))
            target_ids = target_ids[chosen]
            target_labels = target_labels[chosen]
        design = build_design(history, target_ids, gate)
        x, y, row_weights = stratified_sample(
            design,
            target_labels,
            maximum_rows,
            1337 + index,
        )
        baseline = np.average(x[:, :, 0].argmax(axis=1) == y, weights=row_weights)
        full_organization = history.organization_posterior(target_ids)[0].argmax(axis=1)
        full_baseline = float(np.mean(full_organization == target_labels))
        exception_baselines.append(full_baseline)
        for mode, support, purity in EXCEPTION_CANDIDATES:
            key = f"{mode}:{support}:{purity:g}"
            exception = document_exception_classes(
                history,
                target_ids,
                support,
                purity,
                mode,
            )
            score = float(np.mean(exception == target_labels))
            exception_records[key].append(score)
            print(
                f"[main] internal_origin={start} exception_mode={mode} "
                f"support={support} purity={purity:g} "
                f"accuracy={score:.6f} delta={score - full_baseline:.6f}"
            )
        print(
            f"[main] internal_origin={start} count={len(target_labels)} "
            f"fit_count={len(y)} organization_accuracy={baseline:.6f}"
        )
        sampled.append((x, y, row_weights, start))
        del history, design
    selected_key = max(
        exception_records,
        key=lambda key: (
            float(np.mean(np.asarray(exception_records[key]) - exception_baselines))
            + float(np.min(np.asarray(exception_records[key]) - exception_baselines)),
            float(np.mean(np.asarray(exception_records[key]) - exception_baselines)),
        ),
    )
    selected_delta = np.asarray(exception_records[selected_key]) - exception_baselines
    exception_enabled = (
        float(np.mean(selected_delta)) >= 0.0
        and float(np.min(selected_delta)) >= 0.0
    )
    selected_mode, selected_support, selected_purity = selected_key.split(":")
    selected_exception = (
        selected_mode,
        int(selected_support),
        float(selected_purity),
    )
    print(
        f"[main] selected_exception={selected_key} enabled={exception_enabled} "
        f"mean_delta={np.mean(selected_delta):.6f} worst_delta={np.min(selected_delta):.6f}"
    )
    return sampled, selected_exception, exception_enabled, exception_records


def full_history(
    store: FeatureStore,
    ids: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    cutoff: str,
    name: str,
):
    return HistoryModel(
        store,
        ids,
        labels,
        timestamps,
        np.datetime64(cutoff),
        name,
    )


def partial_predictions(
    history: HistoryModel,
    ids: np.ndarray,
    gate,
    weights: np.ndarray,
    exception_config: tuple[str, int, float] | None,
    limit: int | None,
):
    if limit is None or len(ids) <= limit:
        design = build_design(history, ids, gate)
        prediction = probabilities(design, weights)
        if exception_config is not None:
            mode, support, purity = exception_config
            prediction = apply_document_exception_gate(
                prediction,
                history,
                ids,
                support,
                purity,
                mode,
            )
        return prediction, design
    organization, _ = history.organization_posterior(ids)
    baseline_weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    baseline_weights[0] = 6.0
    baseline_logits = 6.0 * np.log(np.maximum(organization, 1e-8))
    baseline_logits -= baseline_logits.max(axis=1, keepdims=True)
    prediction = np.exp(baseline_logits)
    prediction /= prediction.sum(axis=1, keepdims=True)
    chosen = np.arange(limit, dtype=np.int64)
    design = build_design(history, ids[chosen], gate)
    prediction[chosen] = probabilities(design, weights)
    if exception_config is not None:
        mode, support, purity = exception_config
        prediction = apply_document_exception_gate(
            prediction,
            history,
            ids,
            support,
            purity,
            mode,
        )
    return prediction.astype(np.float32), design


def write_outputs(validation_prediction: np.ndarray, test_prediction: np.ndarray, diagnostics: dict):
    output = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "./output_data_generic_exp_1"))
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "val_predictions.npy", np.asarray(validation_prediction, dtype=np.float32))
    np.save(output / "test_predictions.npy", np.asarray(test_prediction, dtype=np.float32))
    (output / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    print(
        f"[main] saved val={validation_prediction.shape} test={test_prediction.shape} "
        f"directory={output}"
    )


def main():
    started = time.time()
    debug = "--debug" in sys.argv
    task, database, train, validation, test = load_inputs()
    if int(task.num_classes) != CLASSES:
        raise RuntimeError(f"expected {CLASSES} classes, received {task.num_classes}")
    print(
        f"[main] mode={'debug' if debug else 'full'} train={len(train)} "
        f"val={len(validation)} test={len(test)} elapsed={time.time() - started:.2f}s"
    )
    store = FeatureStore(database)
    gate_c, gate_records = select_gate(store, train, debug)
    origins, exception_config, exception_enabled, exception_records = origin_designs(
        store,
        train,
        gate_c,
        debug,
    )
    if not exception_enabled:
        exception_config = None
    regularization, corrections_enabled, regularization_records = select_regularization(origins)
    origin_x = np.concatenate([part[0] for part in origins])
    origin_y = np.concatenate([part[1] for part in origins])
    origin_row_weights = np.concatenate([part[2] for part in origins])
    if corrections_enabled:
        weights_a = fit_nonnegative_weights(
            origin_x,
            origin_y,
            origin_row_weights,
            regularization,
        )
    else:
        weights_a = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        weights_a[0] = 6.0
    train_ids = train["ID"].to_numpy(dtype=np.int64)
    train_labels = train["PLANT"].to_numpy(dtype=np.int64)
    train_times = train["CREATIONTIMESTAMP"].to_numpy(dtype="datetime64[ns]")
    model_a = full_history(
        store,
        train_ids,
        train_labels,
        train_times,
        "2020-02-01",
        "model_A_train",
    )
    gate_a = fit_history_gate(
        model_a,
        train_ids,
        train_labels,
        gate_c,
        180000 if debug else 600000,
    )
    validation_ids = validation["ID"].to_numpy(dtype=np.int64)
    validation_labels = validation["PLANT"].to_numpy(dtype=np.int64)
    validation_prediction, validation_design = partial_predictions(
        model_a,
        validation_ids,
        gate_a,
        weights_a,
        exception_config,
        25000 if debug else None,
    )
    if debug:
        val_x, val_y, val_row_weights = stratified_sample(
            validation_design,
            validation_labels[: len(validation_design)],
            min(25000, len(validation_design)),
            7331,
        )
    else:
        val_x, val_y, val_row_weights = stratified_sample(
            validation_design,
            validation_labels,
            50000,
            7331,
        )
    del validation_design, model_a
    combined_x = np.concatenate([origin_x, val_x])
    combined_y = np.concatenate([origin_y, val_y])
    combined_row_weights = np.concatenate([origin_row_weights, val_row_weights])
    if corrections_enabled:
        weights_b = fit_nonnegative_weights(
            combined_x,
            combined_y,
            combined_row_weights,
            regularization,
        )
    else:
        weights_b = weights_a.copy()
    combined = pd.concat([train, validation], ignore_index=True)
    combined_ids = combined["ID"].to_numpy(dtype=np.int64)
    combined_labels = combined["PLANT"].to_numpy(dtype=np.int64)
    combined_times = combined["CREATIONTIMESTAMP"].to_numpy(dtype="datetime64[ns]")
    model_b = full_history(
        store,
        combined_ids,
        combined_labels,
        combined_times,
        "2020-07-01",
        "model_B_train_validation",
    )
    gate_b = fit_history_gate(
        model_b,
        combined_ids,
        combined_labels,
        gate_c,
        180000 if debug else 700000,
    )
    test_ids = test["ID"].to_numpy(dtype=np.int64)
    test_prediction, _ = partial_predictions(
        model_b,
        test_ids,
        gate_b,
        weights_b,
        exception_config,
        25000 if debug else None,
    )
    diagnostics = {
        "debug": debug,
        "gate_c": gate_c,
        "gate_records": gate_records,
        "regularization": regularization,
        "regularization_records": regularization_records,
        "corrections_enabled": corrections_enabled,
        "exception_config": exception_config,
        "exception_records": exception_records,
        "feature_names": FEATURE_NAMES,
        "weights_a": weights_a.tolist(),
        "weights_b": weights_b.tolist(),
        "elapsed_seconds": time.time() - started,
    }
    write_outputs(validation_prediction, test_prediction, diagnostics)
    print(f"[main] complete elapsed={time.time() - started:.2f}s")


if __name__ == "__main__":
    main()
