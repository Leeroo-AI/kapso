from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from kapso_datasets.common import is_debug, load_task, run_data_dir, save_predictions, shared_cache_dir
from modeling import final_predictions, select_pipeline
from relational_features import CACHE_VERSION, TemporalFeatureBuilder, add_publication_interactions, cache_registry


TARGET_WINDOW_DAYS = 182
MODEL_A_CUTOFF = pd.Timestamp("2021-07-03")
MODEL_B_CUTOFF = pd.Timestamp("2022-07-03")
VALIDATION_ORIGIN = pd.Timestamp("2022-01-01")
TEST_ORIGIN = pd.Timestamp("2023-01-01")


def elapsed(started: float, phase: str, details: dict | None = None) -> None:
    payload = {"phase": phase, "elapsed_seconds": round(time.time() - started, 2)}
    if details:
        payload.update(details)
    print(f"[phase] {json.dumps(payload, sort_keys=True)}", flush=True)


def derived_origins(first: pd.Timestamp, cutoff: pd.Timestamp, official: list[pd.Timestamp]) -> list[pd.Timestamp]:
    origins = set(pd.Timestamp(origin) for origin in official if pd.Timestamp(origin) <= cutoff)
    current = pd.Timestamp(first)
    while current <= cutoff:
        origins.add(current)
        current += pd.Timedelta(days=91)
    origins.add(cutoff)
    origins = {origin for origin in origins if origin <= cutoff}
    if cutoff == MODEL_B_CUTOFF:
        origins.discard(pd.Timestamp("2022-07-02"))
    return sorted(origins)


def recreate_labels(raw_pa: pd.DataFrame, origins: list[pd.Timestamp]) -> pd.DataFrame:
    timestamps = pd.DataFrame({"timestamp": pd.to_datetime(origins)})
    connection = duckdb.connect()
    connection.register("timestamp_df", timestamps)
    connection.register("paperAuthors", raw_pa)
    labels = connection.sql(
        "WITH author_pubs AS ("
        " SELECT t.timestamp AS date, pa.Author_ID, COUNT(pa.Paper_ID) AS publication_count"
        " FROM timestamp_df t JOIN paperAuthors pa"
        " ON pa.Submission_Date > t.timestamp"
        " AND pa.Submission_Date <= t.timestamp + INTERVAL '182 days'"
        " GROUP BY t.timestamp, pa.Author_ID"
        ") SELECT date, Author_ID, publication_count FROM author_pubs"
    ).df()
    connection.close()
    labels["Author_ID"] = labels["Author_ID"].astype(np.int64)
    labels["publication_count"] = labels["publication_count"].astype(np.float64)
    labels = labels.sort_values(["date", "Author_ID"], kind="stable").reset_index(drop=True)
    labels["_row_id"] = labels.groupby("date", sort=False).cumcount().astype(np.int64)
    return labels


def verify_official_train(labels: pd.DataFrame, official: pd.DataFrame) -> None:
    comparison = official[["date", "Author_ID", "publication_count"]].merge(
        labels[["date", "Author_ID", "publication_count"]],
        on=["date", "Author_ID"],
        how="left",
        suffixes=("_official", "_derived"),
        validate="one_to_one",
    )
    if comparison["publication_count_derived"].isna().any():
        raise RuntimeError("recreated labels omit official training rows")
    if not np.array_equal(
        comparison["publication_count_official"].to_numpy(dtype=np.float64),
        comparison["publication_count_derived"].to_numpy(dtype=np.float64),
    ):
        raise RuntimeError("recreated labels differ from official training labels")


def task_rows(table) -> pd.DataFrame:
    rows = table.df[["date", "Author_ID"]].copy().reset_index(drop=True)
    rows["Author_ID"] = rows["Author_ID"].astype(np.int64)
    rows["_row_id"] = np.arange(len(rows), dtype=np.int64)
    return rows[["_row_id", "date", "Author_ID"]]


def attach_features(builder: TemporalFeatureBuilder, labels: pd.DataFrame, scope: str, started: float) -> pd.DataFrame:
    frames = []
    built_rows = 0
    built_seconds = 0.0
    for origin, batch in labels.groupby("date", sort=True):
        batch = batch.reset_index(drop=True)
        batch["_row_id"] = np.arange(len(batch), dtype=np.int64)
        features, reused, seconds = builder.build(batch[["_row_id", "date", "Author_ID"]], scope)
        features = add_publication_interactions(features)
        if not np.array_equal(features["Author_ID"].to_numpy(), batch["Author_ID"].to_numpy()):
            raise RuntimeError("label and feature author order mismatch")
        features = features.copy()
        features["publication_count"] = batch["publication_count"].to_numpy(dtype=np.float32)
        features["origin"] = pd.Timestamp(origin)
        frames.append(features)
        built_rows += len(batch)
        if not reused:
            built_seconds += seconds
        rate = len(batch) / max(seconds, 1e-6)
        elapsed(started, "origin_features", {"origin": str(pd.Timestamp(origin).date()), "rows": len(batch), "scope": scope, "cache_reused": reused, "seconds": round(seconds, 2), "rows_per_second": round(rate, 1)})
    result = pd.concat(frames, ignore_index=True)
    result.index = np.arange(len(result), dtype=np.int64)
    elapsed(started, "training_features_complete", {"rows": built_rows, "uncached_seconds": round(built_seconds, 2), "uncached_rows_per_minute": round(60 * built_rows / max(built_seconds, 1e-6), 1)})
    return result


def capped_training(data: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if len(data) <= limit:
        return data.reset_index(drop=True)
    portions = []
    sizes = data.groupby("origin", sort=True).size()
    allocation = np.maximum(1, np.floor(limit * sizes / sizes.sum()).astype(int))
    for origin, group in data.groupby("origin", sort=True):
        count = min(len(group), int(allocation.loc[origin]))
        portions.append(group.sample(count, random_state=seed))
    result = pd.concat(portions, ignore_index=True)
    if len(result) > limit:
        result = result.sample(limit, random_state=seed).reset_index(drop=True)
    return result


def prior_matrix(history: pd.DataFrame, rows: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    result = pd.DataFrame(index=rows.index)
    columns = [
        "publication_label_history_count",
        "publication_label_history_mean",
        "publication_label_history_std",
        "publication_label_history_max",
        "publication_label_history_last",
        "publication_label_history_active_rate",
        "publication_label_history_tail_rate",
        "publication_label_history_decay_mean",
        "publication_label_history_trend",
        "category_target_count",
        "category_target_prior",
        "category_target_active_prior",
        "category_target_tail_prior",
    ]
    if len(history) == 0:
        for column in columns:
            result[column] = 0.0 if column.endswith(("_count", "_rate")) else np.nan
        return result
    ordered = history.sort_values(["origin", "Author_ID"], kind="stable").copy()
    ordered["label_active"] = ordered["publication_count"].gt(1).astype(np.float32)
    ordered["label_tail"] = ordered["publication_count"].ge(4).astype(np.float32)
    label_age = (pd.Timestamp(origin) - pd.to_datetime(ordered["origin"])).dt.days.to_numpy(dtype=np.float32)
    ordered["label_decay_weight"] = np.power(2.0, -label_age / 730.0)
    ordered["label_decay_target"] = ordered["publication_count"] * ordered["label_decay_weight"]
    author = ordered.groupby("Author_ID", sort=False).agg(
        publication_label_history_count=("publication_count", "size"),
        publication_label_history_mean=("publication_count", "mean"),
        publication_label_history_std=("publication_count", "std"),
        publication_label_history_max=("publication_count", "max"),
        publication_label_history_last=("publication_count", "last"),
        publication_label_history_active_rate=("label_active", "mean"),
        publication_label_history_tail_rate=("label_tail", "mean"),
        label_decay_target=("label_decay_target", "sum"),
        label_decay_weight=("label_decay_weight", "sum"),
    ).reset_index()
    author["publication_label_history_decay_mean"] = author["label_decay_target"] / author["label_decay_weight"].clip(lower=1e-6)
    author["publication_label_history_trend"] = author["publication_label_history_last"] - author["publication_label_history_mean"]
    author = author.drop(columns=["label_decay_target", "label_decay_weight"])
    mapped = rows[["Author_ID"]].merge(author, on="Author_ID", how="left", sort=False, validate="many_to_one")
    mapped.index = rows.index
    for column in author.columns:
        if column != "Author_ID":
            result[column] = mapped[column]
    result["publication_label_history_count"] = result["publication_label_history_count"].fillna(0)
    global_mean = float(ordered["publication_count"].mean())
    global_active = float(ordered["label_active"].mean())
    global_tail = float(ordered["label_tail"].mean())
    result["category_target_count"] = 0.0
    result["category_target_prior"] = global_mean
    result["category_target_active_prior"] = global_active
    result["category_target_tail_prior"] = global_tail
    if "category_latest" in ordered.columns and "category_latest" in rows.columns:
        categorized = ordered.dropna(subset=["category_latest"])
        category = categorized.groupby("category_latest", sort=False).agg(
            category_target_count=("publication_count", "size"),
            category_target_sum=("publication_count", "sum"),
            category_target_active_sum=("label_active", "sum"),
            category_target_tail_sum=("label_tail", "sum"),
        ).reset_index()
        smoothing = 50.0
        category["category_target_prior"] = (category["category_target_sum"] + smoothing * global_mean) / (category["category_target_count"] + smoothing)
        category["category_target_active_prior"] = (category["category_target_active_sum"] + smoothing * global_active) / (category["category_target_count"] + smoothing)
        category["category_target_tail_prior"] = (category["category_target_tail_sum"] + smoothing * global_tail) / (category["category_target_count"] + smoothing)
        category = category[["category_latest", "category_target_count", "category_target_prior", "category_target_active_prior", "category_target_tail_prior"]]
        category_mapped = rows[["category_latest"]].merge(category, on="category_latest", how="left", sort=False, validate="many_to_one")
        category_mapped.index = rows.index
        known = category_mapped["category_target_count"].notna()
        for column in ("category_target_count", "category_target_prior", "category_target_active_prior", "category_target_tail_prior"):
            result.loc[known, column] = category_mapped.loc[known, column]
    return result[columns].astype(np.float32)


def expanding_supervision_features(training: pd.DataFrame, prediction: pd.DataFrame, prediction_origin: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_training = training.copy()
    enriched_training = training.copy()
    feature_names = None
    for origin in sorted(pd.to_datetime(raw_training["origin"].unique())):
        selected = pd.to_datetime(raw_training["origin"]).eq(origin)
        eligible = pd.to_datetime(raw_training["origin"]) + pd.to_timedelta(TARGET_WINDOW_DAYS, unit="D") < pd.Timestamp(origin)
        values = prior_matrix(raw_training.loc[eligible], raw_training.loc[selected], pd.Timestamp(origin))
        if feature_names is None:
            feature_names = values.columns.tolist()
            for column in feature_names:
                enriched_training[column] = np.nan
        enriched_training.loc[selected, feature_names] = values.to_numpy()
    prediction_eligible = pd.to_datetime(raw_training["origin"]) + pd.to_timedelta(TARGET_WINDOW_DAYS, unit="D") <= pd.Timestamp(prediction_origin)
    prediction_values = prior_matrix(raw_training.loc[prediction_eligible], prediction, pd.Timestamp(prediction_origin))
    enriched_prediction = prediction.copy()
    for column in prediction_values.columns:
        enriched_prediction[column] = prediction_values[column].to_numpy()
    return enriched_training, enriched_prediction


def validate_predictions(values: np.ndarray, expected: int, provenance: str) -> None:
    if values.shape != (expected,):
        raise RuntimeError(f"{provenance} prediction shape {values.shape} != {(expected,)}")
    if not np.issubdtype(values.dtype, np.floating):
        raise RuntimeError(f"{provenance} predictions are not floating point")
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{provenance} predictions contain non-finite values")
    if np.any(values < 1.0):
        raise RuntimeError(f"{provenance} predictions violate floor")


def write_diagnostics(selection: dict, val_prediction: np.ndarray, test_prediction: np.ndarray, started: float, debug: bool) -> None:
    output = Path("output_data_generic_exp_0")
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        "feature_cache_version": CACHE_VERSION,
        "debug": debug,
        "selection": selection,
        "validation_prediction": {
            "producer": "model_a_without_validation_labels",
            "min": float(val_prediction.min()),
            "mean": float(val_prediction.mean()),
            "max": float(val_prediction.max()),
        },
        "test_prediction": {
            "producer": "model_b_with_train_and_validation_supervision",
            "min": float(test_prediction.min()),
            "mean": float(test_prediction.mean()),
            "max": float(test_prediction.max()),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    (output / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    (run_data_dir() / "metrics.json").write_text(json.dumps(diagnostics, indent=2))


def main() -> None:
    started = time.time()
    debug = is_debug()
    context = load_task()
    elapsed(started, "load_data", {"debug": debug})
    cache_root = shared_cache_dir()
    builder = TemporalFeatureBuilder(context.db, cache_root)
    cache_registry(cache_root)
    elapsed(started, "prepare_relations", {"seconds": round(builder.preparation_seconds, 2)})
    official_origins = sorted(pd.to_datetime(context.train.df["date"].unique()).tolist())
    if debug:
        model_a_origins = [pd.Timestamp(origin) for origin in official_origins[-3:]]
        scope = "core"
    else:
        model_a_origins = derived_origins(pd.Timestamp(official_origins[0]), MODEL_A_CUTOFF, [pd.Timestamp(origin) for origin in official_origins])
        scope = "full"
    labels_a = recreate_labels(builder.raw_pa, model_a_origins)
    if not debug:
        verify_official_train(labels_a, context.train.df)
    if (labels_a["date"] + pd.to_timedelta(TARGET_WINDOW_DAYS, unit="D")).max() > VALIDATION_ORIGIN:
        raise RuntimeError("model A includes labels beyond validation feature origin")
    train_a_base = attach_features(builder, labels_a, scope, started)
    if debug:
        train_a_base = capped_training(train_a_base, 50000, 1337)
    val_rows = task_rows(context.val)
    val_features_base, val_reused, val_seconds = builder.build(val_rows, scope)
    val_features_base = add_publication_interactions(val_features_base)
    train_a, val_features = expanding_supervision_features(train_a_base, val_features_base, VALIDATION_ORIGIN)
    elapsed(started, "validation_features", {"rows": len(val_features), "cache_reused": val_reused, "seconds": round(val_seconds, 2)})
    if debug:
        selection = {
            "feature_scope": "core",
            "weight_mode": "uniform",
            "objectives": ["raw"],
            "weights": [1.0],
            "intercept": 0.0,
            "rounds": {"raw": 200},
            "debug_selection": True,
        }
    else:
        folds = [pd.Timestamp("2020-01-04"), pd.Timestamp("2020-07-04"), pd.Timestamp("2021-01-02"), pd.Timestamp("2021-07-03")]
        selection = select_pipeline(train_a, folds, builder.cache_dir)
    val_prediction = final_predictions(train_a, val_features, VALIDATION_ORIGIN, selection, debug)
    val_prediction = np.maximum(val_prediction, 1.0).astype(np.float64)
    validate_predictions(val_prediction, len(context.val), "model_a_validation")
    val_prediction_locked = val_prediction.copy()
    val_prediction_fingerprint = str(hash(val_prediction_locked.tobytes()))
    elapsed(started, "model_a_complete", {"train_rows": len(train_a), "prediction_fingerprint": val_prediction_fingerprint})
    if debug:
        train_b = pd.concat(
            [
                train_a_base,
                val_features_base.assign(publication_count=context.val.df[context.target_col].to_numpy(dtype=np.float32), origin=VALIDATION_ORIGIN),
            ],
            ignore_index=True,
        )
        train_b = capped_training(train_b, 50000, 7331)
    else:
        model_b_origins = derived_origins(pd.Timestamp(official_origins[0]), MODEL_B_CUTOFF, [pd.Timestamp(origin) for origin in official_origins])
        labels_b = recreate_labels(builder.raw_pa, model_b_origins)
        generated_val = labels_b.loc[labels_b["date"].eq(VALIDATION_ORIGIN), ["date", "Author_ID", "publication_count"]]
        official_val = context.val.df[["date", "Author_ID", context.target_col]].rename(columns={context.target_col: "official_publication_count"})
        validation_check = official_val.merge(generated_val, on=["date", "Author_ID"], how="left", validate="one_to_one")
        if validation_check["publication_count"].isna().any() or not np.array_equal(validation_check["official_publication_count"].to_numpy(dtype=np.float64), validation_check["publication_count"].to_numpy(dtype=np.float64)):
            raise RuntimeError("recreated validation labels differ from official validation labels")
        later_labels = labels_b.loc[labels_b["date"].gt(MODEL_A_CUTOFF) & labels_b["date"].ne(VALIDATION_ORIGIN)].copy()
        later_features = attach_features(builder, later_labels, scope, started)
        val_supervision = val_features_base.copy()
        val_supervision["publication_count"] = context.val.df[context.target_col].to_numpy(dtype=np.float32)
        val_supervision["origin"] = VALIDATION_ORIGIN
        train_b = pd.concat([train_a_base, later_features, val_supervision], ignore_index=True)
        train_b.index = np.arange(len(train_b), dtype=np.int64)
        if (pd.to_datetime(train_b["origin"]) + pd.to_timedelta(TARGET_WINDOW_DAYS, unit="D")).max() > TEST_ORIGIN:
            raise RuntimeError("model B includes labels beyond database cutoff")
    if str(hash(val_prediction_locked.tobytes())) != val_prediction_fingerprint:
        raise RuntimeError("validation predictions changed after validation supervision access")
    test_rows = task_rows(context.test)
    test_features_base, test_reused, test_seconds = builder.build(test_rows, scope)
    test_features_base = add_publication_interactions(test_features_base)
    train_b, test_features = expanding_supervision_features(train_b, test_features_base, TEST_ORIGIN)
    elapsed(started, "test_features", {"rows": len(test_features), "cache_reused": test_reused, "seconds": round(test_seconds, 2)})
    test_prediction = final_predictions(train_b, test_features, TEST_ORIGIN, selection, debug)
    test_prediction = np.maximum(test_prediction, 1.0).astype(np.float64)
    validate_predictions(test_prediction, len(context.test), "model_b_test")
    if not np.array_equal(val_rows["Author_ID"].to_numpy(), context.val.df["Author_ID"].to_numpy(dtype=np.int64)):
        raise RuntimeError("validation row alignment assertion failed")
    if not np.array_equal(test_rows["Author_ID"].to_numpy(), context.test.df["Author_ID"].to_numpy(dtype=np.int64)):
        raise RuntimeError("test row alignment assertion failed")
    write_diagnostics(selection, val_prediction_locked, test_prediction, started, debug)
    save_predictions(val_prediction_locked, test_prediction)
    elapsed(started, "saved", {"val_shape": list(val_prediction_locked.shape), "test_shape": list(test_prediction.shape), "val_producer": "model_a", "test_producer": "model_b"})


if __name__ == "__main__":
    main()
