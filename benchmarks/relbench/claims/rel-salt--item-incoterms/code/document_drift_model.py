from __future__ import annotations

import fcntl
import json
import os
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from kapso_datasets.common import load_task, run_data_dir, save_predictions, shared_cache_dir


NUM_CLASSES = 13
VERSION = "lane3_temporal_document_lgb_v1"
FULL_HALF_LIVES = (180, 365, None)
FRESH_MONTHS = (6, 9, 12)
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEGMENT_NAMES = ("<30", "30-90", "90-180", ">180/new")
HISTORY_KEYS = (
    "sold_to",
    "ship_to",
    "bill_to",
    "payer",
    "sold_ship",
    "sold_sales_area",
    "sold_ship_sales_area",
    "product_first",
    "ship_country_sales_area",
)
DEBUG_HISTORY_KEYS = ("sold_to", "sold_ship", "sold_sales_area", "product_first")


def phase(message: str, started: float) -> float:
    now = time.time()
    print(f"[temporal_lgb] {message}: {now - started:.1f}s", flush=True)
    return now


def stable_mode(values: pd.Series):
    counts = values.value_counts(dropna=False)
    return counts.index[0] if len(counts) else np.nan


def encode_column(values: pd.Series) -> np.ndarray:
    codes, _ = pd.factorize(values, sort=True, use_na_sentinel=True)
    return codes.astype(np.int32)


def encode_interaction(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    hashed = pd.util.hash_pandas_object(frame[columns], index=False)
    codes, _ = pd.factorize(hashed, sort=False)
    return codes.astype(np.int32)


def build_document_data() -> dict:
    ctx = load_task(upto_test_timestamp=False)
    item_columns = [
        "ID",
        "SALESDOCUMENT",
        "SALESDOCUMENTITEM",
        "SALESDOCUMENTITEMCATEGORY",
        "PRODUCT",
        "SOLDTOPARTY",
        "SHIPTOPARTY",
        "BILLTOPARTY",
        "PAYERPARTY",
    ]
    header_columns = [
        "SALESDOCUMENT",
        "SALESDOCUMENTTYPE",
        "SALESORGANIZATION",
        "DISTRIBUTIONCHANNEL",
        "ORGANIZATIONDIVISION",
        "BILLINGCOMPANYCODE",
        "TRANSACTIONCURRENCY",
        "CREATIONTIMESTAMP",
    ]
    item = ctx.db.table_dict["salesdocumentitem"].df[item_columns].copy()
    header = ctx.db.table_dict["salesdocument"].df[header_columns].copy()
    customer = ctx.db.table_dict["customer"].df[["CUSTOMER", "ADDRESSID"]].copy()
    address = ctx.db.table_dict["address"].df[["ADDRESSID", "COUNTRY", "REGION"]].copy()
    geography = customer.merge(address, on="ADDRESSID", how="left", validate="many_to_one")
    country_map = geography.set_index("CUSTOMER")["COUNTRY"]
    region_map = geography.set_index("CUSTOMER")["REGION"]
    task_frames = []
    row_maps = {}
    for split, table in (("train", ctx.train), ("val", ctx.val), ("test", ctx.test)):
        frame = table.df.copy()
        frame["_split"] = split
        frame["_row"] = np.arange(len(frame), dtype=np.int32)
        task_frames.append(frame)
    rows = pd.concat(task_frames, ignore_index=True, sort=False)
    rows = rows.merge(item, on="ID", how="left", validate="one_to_one")
    if rows["SALESDOCUMENT"].isna().any():
        raise RuntimeError("task rows are missing salesdocumentitem matches")
    rows["_item_number"] = pd.to_numeric(rows["SALESDOCUMENTITEM"], errors="coerce").fillna(-1)
    rows = rows.sort_values(
        ["_split", "SALESDOCUMENT", "CREATIONTIMESTAMP", "_item_number", "ID"],
        kind="stable",
    )
    group_columns = ["_split", "SALESDOCUMENT", "CREATIONTIMESTAMP"]
    grouped = rows.groupby(group_columns, sort=False, observed=True, dropna=False)
    docs = grouped.agg(
        item_count=("ID", "size"),
        item_number_min=("_item_number", "min"),
        item_number_max=("_item_number", "max"),
        product_first=("PRODUCT", "first"),
        product_last=("PRODUCT", "last"),
        product_unique=("PRODUCT", "nunique"),
        category_first=("SALESDOCUMENTITEMCATEGORY", "first"),
        category_last=("SALESDOCUMENTITEMCATEGORY", "last"),
        category_unique=("SALESDOCUMENTITEMCATEGORY", "nunique"),
        sold_to=("SOLDTOPARTY", "first"),
        sold_unique=("SOLDTOPARTY", "nunique"),
        ship_to=("SHIPTOPARTY", "first"),
        ship_unique=("SHIPTOPARTY", "nunique"),
        bill_to=("BILLTOPARTY", "first"),
        bill_unique=("BILLTOPARTY", "nunique"),
        payer=("PAYERPARTY", "first"),
        payer_unique=("PAYERPARTY", "nunique"),
    ).reset_index()
    label_rows = rows[rows["_split"].isin(["train", "val"])].copy()
    label_rows["_label"] = label_rows[ctx.target_col].astype(np.int8)
    label_counts = (
        label_rows.groupby(["_split", "SALESDOCUMENT", "_label"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    label_counts = label_counts.sort_values(
        ["_split", "SALESDOCUMENT", "count", "_label"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    doc_labels = label_counts.drop_duplicates(["_split", "SALESDOCUMENT"], keep="first")
    doc_weights = (
        label_rows.groupby(["_split", "SALESDOCUMENT"], observed=True)
        .size()
        .rename("doc_weight")
        .reset_index()
    )
    doc_labels = doc_labels.merge(doc_weights, on=["_split", "SALESDOCUMENT"])
    docs = docs.merge(
        doc_labels[["_split", "SALESDOCUMENT", "_label", "doc_weight"]],
        on=["_split", "SALESDOCUMENT"],
        how="left",
        validate="many_to_one",
    )
    docs = docs.merge(header, on="SALESDOCUMENT", how="left", validate="many_to_one", suffixes=("", "_header"))
    docs["header_age_seconds"] = (
        docs["CREATIONTIMESTAMP"] - docs["CREATIONTIMESTAMP_header"]
    ).dt.total_seconds().fillna(0).clip(lower=0)
    docs = docs.drop(columns=["CREATIONTIMESTAMP_header"])
    for role in ("sold", "ship", "bill", "payer"):
        key = f"{role}_to" if role != "payer" else "payer"
        docs[f"{role}_country"] = docs[key].map(country_map)
        docs[f"{role}_region"] = docs[key].map(region_map)
    docs["all_parties_equal"] = (
        (docs["sold_to"] == docs["ship_to"])
        & (docs["sold_to"] == docs["bill_to"])
        & (docs["sold_to"] == docs["payer"])
    ).astype(np.int8)
    docs["sold_ship_equal"] = (docs["sold_to"] == docs["ship_to"]).astype(np.int8)
    docs["sold_bill_equal"] = (docs["sold_to"] == docs["bill_to"]).astype(np.int8)
    docs["sold_payer_equal"] = (docs["sold_to"] == docs["payer"]).astype(np.int8)
    docs["sold_ship_country_equal"] = (docs["sold_country"] == docs["ship_country"]).astype(np.int8)
    docs["year"] = docs["CREATIONTIMESTAMP"].dt.year.astype(np.int16)
    docs["month"] = docs["CREATIONTIMESTAMP"].dt.month.astype(np.int8)
    docs["day"] = docs["CREATIONTIMESTAMP"].dt.day.astype(np.int8)
    docs["day_of_week"] = docs["CREATIONTIMESTAMP"].dt.dayofweek.astype(np.int8)
    docs["day_of_year"] = docs["CREATIONTIMESTAMP"].dt.dayofyear.astype(np.int16)
    docs["hour"] = docs["CREATIONTIMESTAMP"].dt.hour.astype(np.int8)
    docs["time_days"] = (
        (docs["CREATIONTIMESTAMP"] - pd.Timestamp("2018-01-01")) / pd.Timedelta(days=1)
    ).astype(np.float32)
    docs["document_number"] = docs["SALESDOCUMENT"].astype(np.float32)
    docs["item_number_span"] = docs["item_number_max"] - docs["item_number_min"]
    categorical_columns = [
        "product_first",
        "product_last",
        "category_first",
        "category_last",
        "sold_to",
        "ship_to",
        "bill_to",
        "payer",
        "SALESDOCUMENTTYPE",
        "SALESORGANIZATION",
        "DISTRIBUTIONCHANNEL",
        "ORGANIZATIONDIVISION",
        "BILLINGCOMPANYCODE",
        "TRANSACTIONCURRENCY",
        "sold_country",
        "sold_region",
        "ship_country",
        "ship_region",
        "bill_country",
        "bill_region",
        "payer_country",
        "payer_region",
        "month",
        "day_of_week",
        "hour",
    ]
    for column in categorical_columns:
        docs[column] = encode_column(docs[column])
    interactions = {
        "sold_ship": ["sold_to", "ship_to"],
        "sold_sales_area": ["sold_to", "SALESORGANIZATION", "DISTRIBUTIONCHANNEL", "ORGANIZATIONDIVISION"],
        "sold_ship_sales_area": ["sold_to", "ship_to", "SALESORGANIZATION", "DISTRIBUTIONCHANNEL", "ORGANIZATIONDIVISION"],
        "ship_country_sales_area": ["ship_country", "SALESORGANIZATION", "DISTRIBUTIONCHANNEL", "ORGANIZATIONDIVISION"],
        "product_sales_area": ["product_first", "SALESORGANIZATION", "DISTRIBUTIONCHANNEL"],
        "country_pair": ["sold_country", "ship_country"],
    }
    for name, columns in interactions.items():
        docs[name] = encode_interaction(docs, columns)
        categorical_columns.append(name)
    docs["_time_ns"] = docs["CREATIONTIMESTAMP"].astype("int64")
    docs["_fit_row"] = ~docs.duplicated(["_split", "SALESDOCUMENT"], keep="first")
    docs["_unit"] = np.arange(len(docs), dtype=np.int32)
    unit_lookup = rows[group_columns + ["_row"]].merge(
        docs[group_columns + ["_unit"]], on=group_columns, how="left", validate="many_to_one"
    )
    for split in ("train", "val", "test"):
        part = unit_lookup[unit_lookup["_split"] == split].sort_values("_row")
        row_maps[split] = part["_unit"].to_numpy(np.int32)
    numeric_columns = [
        "item_count",
        "item_number_min",
        "item_number_max",
        "item_number_span",
        "product_unique",
        "category_unique",
        "sold_unique",
        "ship_unique",
        "bill_unique",
        "payer_unique",
        "header_age_seconds",
        "all_parties_equal",
        "sold_ship_equal",
        "sold_bill_equal",
        "sold_payer_equal",
        "sold_ship_country_equal",
        "year",
        "day",
        "day_of_year",
        "time_days",
        "document_number",
    ]
    feature_columns = categorical_columns + numeric_columns
    for column in feature_columns:
        docs[column] = pd.to_numeric(docs[column], errors="coerce").fillna(-1).astype(np.float32)
    return {
        "docs": docs,
        "row_maps": row_maps,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
    }


def load_document_data(debug: bool) -> dict:
    suffix = "debug" if debug else "full"
    path = shared_cache_dir() / f"{VERSION}_documents_{suffix}.pkl"
    full_path = shared_cache_dir() / f"{VERSION}_documents_full.pkl"
    if debug and full_path.exists():
        path = full_path
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    data = build_document_data()
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return data


def causal_history_matrix(source: pd.DataFrame, keys: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    matrices = []
    names = []
    source_index = source[["_source_order"]].copy()
    for key in keys:
        values = source[[key, "_time_ns", "_label", "doc_weight"]].copy()
        values["_label"] = values["_label"].astype(np.int8)
        aggregated = values.groupby([key, "_time_ns", "_label"], observed=True, sort=True)["doc_weight"].sum().unstack(fill_value=0)
        aggregated = aggregated.reindex(columns=range(NUM_CLASSES), fill_value=0).astype(np.float64)
        prior = aggregated.groupby(level=0, sort=False).cumsum() - aggregated
        totals = prior.sum(axis=1).to_numpy()
        probabilities = (prior.to_numpy() + 0.2) / (totals[:, None] + 0.2 * NUM_CLASSES)
        blocks = prior.reset_index()[[key, "_time_ns"]]
        blocks["_hist_total"] = np.log1p(totals)
        for class_id in range(NUM_CLASSES):
            blocks[f"_hist_p{class_id}"] = probabilities[:, class_id]
        block_labels = aggregated.to_numpy().argmax(axis=1).astype(np.float32)
        block_times = blocks["_time_ns"].copy()
        blocks["_hist_last_label"] = pd.Series(block_labels, index=blocks.index).groupby(blocks[key], sort=False).shift(1).fillna(-1)
        blocks["_hist_last_time"] = block_times.groupby(blocks[key], sort=False).shift(1)
        aligned = source[["_source_order", key, "_time_ns"]].merge(
            blocks, on=[key, "_time_ns"], how="left", validate="many_to_one"
        ).sort_values("_source_order")
        age = (aligned["_time_ns"] - aligned["_hist_last_time"]) / (86400.0 * 1e9)
        columns = [f"_hist_p{i}" for i in range(NUM_CLASSES)] + ["_hist_total", "_hist_last_label"]
        matrix = aligned[columns].to_numpy(np.float32)
        matrix = np.column_stack([matrix, age.fillna(9999).clip(lower=0, upper=9999).to_numpy(np.float32)])
        matrices.append(matrix)
        names.extend([f"hist_{key}_p{i}" for i in range(NUM_CLASSES)] + [f"hist_{key}_log_count", f"hist_{key}_last_label", f"hist_{key}_age"])
    return np.column_stack(matrices).astype(np.float32), names


def snapshot_history_matrix(source: pd.DataFrame, target: pd.DataFrame, keys: tuple[str, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    matrices = []
    ages = {}
    for key in keys:
        aggregated = source.groupby([key, "_label"], observed=True)["doc_weight"].sum().unstack(fill_value=0)
        aggregated = aggregated.reindex(columns=range(NUM_CLASSES), fill_value=0)
        aligned = aggregated.reindex(target[key].to_numpy()).fillna(0).to_numpy(np.float64)
        totals = aligned.sum(axis=1)
        probabilities = (aligned + 0.2) / (totals[:, None] + 0.2 * NUM_CLASSES)
        latest = source.groupby(key, observed=True)["_time_ns"].max()
        last_time = latest.reindex(target[key].to_numpy()).to_numpy(np.float64)
        age = (target["_time_ns"].to_numpy(np.float64) - last_time) / (86400.0 * 1e9)
        age = np.nan_to_num(age, nan=9999.0, posinf=9999.0, neginf=9999.0)
        age = np.clip(age, 0, 9999).astype(np.float32)
        ages[key] = age
        latest_rows = source.sort_values([key, "_time_ns"], kind="stable").drop_duplicates(key, keep="last").set_index(key)["_label"]
        last_label = latest_rows.reindex(target[key].to_numpy()).fillna(-1).to_numpy(np.float32)
        matrix = np.column_stack([probabilities, np.log1p(totals), last_label, age]).astype(np.float32)
        matrices.append(matrix)
    return np.column_stack(matrices).astype(np.float32), ages


def get_history_data(data: dict, debug: bool) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    docs = data["docs"]
    keys = DEBUG_HISTORY_KEYS if debug else HISTORY_KEYS
    source = docs[(docs["_split"].isin(["train", "val"])) & docs["_fit_row"]].copy()
    source = source.sort_values(["_time_ns", "SALESDOCUMENT"], kind="stable").reset_index(drop=True)
    source["_source_order"] = np.arange(len(source), dtype=np.int32)
    path = shared_cache_dir() / f"{VERSION}_causal_{'debug' if debug else 'full'}.npz"
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        matrix = cached["matrix"]
        names = cached["names"].astype(str).tolist()
        if matrix.shape[0] == len(source):
            return source, matrix, names
    matrix, names = causal_history_matrix(source, keys)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, matrix=matrix, names=np.asarray(names))
    temporary.replace(path)
    return source, matrix, names


def staleness_segments(age: np.ndarray) -> np.ndarray:
    segments = np.full(len(age), 3, dtype=np.int8)
    segments[age < 180] = 2
    segments[age < 90] = 1
    segments[age < 30] = 0
    return segments


def model_parameters(rounds: int) -> dict:
    return {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": "multi_logloss",
        "num_leaves": 127,
        "learning_rate": 0.05,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.85,
        "lambda_l2": 5.0,
        "max_bin": 127,
        "max_cat_threshold": 32,
        "cat_l2": 10.0,
        "cat_smooth": 20.0,
        "verbosity": -1,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "11")),
        "seed": 1337,
        "feature_fraction_seed": 1337,
        "deterministic": True,
        "force_col_wise": True,
        "num_iterations": rounds,
    }


def train_predict(
    base_matrix: np.ndarray,
    causal_matrix: np.ndarray,
    train_positions: np.ndarray,
    target_base: np.ndarray,
    target_history: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    categorical_indices: list[int],
    feature_names: list[str],
    rounds: int,
) -> np.ndarray:
    train_matrix = np.column_stack([base_matrix[train_positions], causal_matrix[train_positions]])
    predict_matrix = np.column_stack([target_base, target_history])
    dataset = lgb.Dataset(
        train_matrix,
        label=labels[train_positions],
        weight=weights,
        feature_name=feature_names,
        categorical_feature=categorical_indices,
        free_raw_data=True,
    )
    model = lgb.train(
        model_parameters(rounds),
        dataset,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    prediction = model.predict(predict_matrix, num_iteration=rounds)
    return np.asarray(prediction, dtype=np.float32)


def component_path(name: str, debug: bool) -> Path:
    mode = "debug" if debug else "full"
    return shared_cache_dir() / f"{VERSION}_{mode}_{name}.npy"


def load_component(name: str, debug: bool, rows: int) -> np.ndarray | None:
    path = component_path(name, debug)
    if not path.exists():
        return None
    prediction = np.load(path, allow_pickle=False)
    if prediction.shape != (rows, NUM_CLASSES) or not np.isfinite(prediction).all():
        return None
    return prediction


def save_component(name: str, debug: bool, prediction: np.ndarray) -> None:
    path = component_path(name, debug)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, prediction.astype(np.float32))
    temporary.replace(path)


def weighted_accuracy(prediction: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.argmax(prediction, axis=1) == labels, weights=weights))


def forecast_prior_ratios(source: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    frame = source[["CREATIONTIMESTAMP", "_label"]].copy()
    frame["month"] = frame["CREATIONTIMESTAMP"].dt.to_period("M")
    counts = frame.groupby(["month", "_label"], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(NUM_CLASSES), fill_value=0).sort_index()
    counts = counts.tail(12)
    priors = (counts.to_numpy(np.float64) + 2.0) / (counts.sum(axis=1).to_numpy()[:, None] + 2.0 * NUM_CLASSES)
    latest_mean = priors[-min(3, len(priors)):].mean(axis=0)
    latest_mean /= latest_mean.sum()
    reference = 0
    log_ratios = np.log(priors / priors[:, [reference]])
    x = np.arange(len(priors), dtype=np.float64)
    centered = x - x.mean()
    ridge = 100.0
    slopes = (centered[:, None] * log_ratios).sum(axis=0) / ((centered * centered).sum() + ridge)
    intercepts = log_ratios.mean(axis=0)
    latest_log = np.log(latest_mean / latest_mean[reference])
    source_last_month = counts.index[-1]
    target_months = target["CREATIONTIMESTAMP"].dt.to_period("M")
    unique_months = pd.Index(target_months.unique()).sort_values()
    ratio_by_month = {}
    for month in unique_months:
        horizon = month.ordinal - source_last_month.ordinal
        trend_log = intercepts + slopes * (x.mean() + horizon)
        forecast_log = 0.35 * trend_log + 0.65 * latest_log
        forecast = np.exp(forecast_log - forecast_log.max())
        forecast /= forecast.sum()
        ratio_by_month[month] = np.clip(forecast / latest_mean, 0.67, 1.5)
    return np.vstack([ratio_by_month[month] for month in target_months]).astype(np.float32)


def apply_prior(prediction: np.ndarray, ratios: np.ndarray) -> np.ndarray:
    adjusted = prediction * ratios
    adjusted /= adjusted.sum(axis=1, keepdims=True).clip(min=1e-12)
    return adjusted.astype(np.float32)


def blend_predictions(full: np.ndarray, fresh: np.ndarray, segments: np.ndarray, weights: tuple[float, ...]) -> np.ndarray:
    blended = np.empty_like(full)
    for segment, weight in enumerate(weights):
        mask = segments == segment
        blended[mask] = (1.0 - weight) * full[mask] + weight * fresh[mask]
    return blended


def internal_folds(source: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return [
        (pd.Timestamp("2019-04-01"), pd.Timestamp("2019-09-01")),
        (pd.Timestamp("2019-09-01"), pd.Timestamp("2020-02-01")),
    ]


def build_internal_predictions(
    data: dict,
    source: pd.DataFrame,
    causal: np.ndarray,
    base_matrix: np.ndarray,
    categorical_indices: list[int],
    feature_names: list[str],
    rounds: int,
    debug: bool,
) -> list[dict]:
    keys = DEBUG_HISTORY_KEYS if debug else HISTORY_KEYS
    outputs = []
    folds = internal_folds(source)
    if debug:
        folds = folds[-1:]
    for fold_index, (cutoff, end) in enumerate(folds):
        source_mask = source["CREATIONTIMESTAMP"] < cutoff
        hold_mask = (source["CREATIONTIMESTAMP"] >= cutoff) & (source["CREATIONTIMESTAMP"] < end)
        source_docs = source[source_mask].copy()
        hold_docs = source[hold_mask].copy()
        hold_history, ages = snapshot_history_matrix(source_docs, hold_docs, keys)
        model_source_mask = source_mask
        if debug:
            model_source_mask = source_mask & (source["CREATIONTIMESTAMP"] >= cutoff - pd.DateOffset(months=12))
        source_positions = np.flatnonzero(model_source_mask.to_numpy())
        target_base = base_matrix[hold_docs["_unit"].to_numpy(np.int64)]
        labels = source["_label"].to_numpy(np.int8)
        item_weights = source["doc_weight"].to_numpy(np.float32)
        full_predictions = {}
        fresh_predictions = {}
        half_lives = (180,) if debug else FULL_HALF_LIVES
        fresh_windows = (3,) if debug else FRESH_MONTHS
        for half_life in half_lives:
            tag = "inf" if half_life is None else str(half_life)
            name = f"fold{fold_index}_full_h{tag}"
            prediction = load_component(name, debug, len(hold_docs))
            if prediction is None:
                ages_train = (cutoff - source.loc[model_source_mask, "CREATIONTIMESTAMP"]).dt.total_seconds().to_numpy() / 86400.0
                decay = np.ones(len(source_positions), dtype=np.float32) if half_life is None else np.power(0.5, ages_train / half_life).astype(np.float32)
                weights = item_weights[source_positions] * decay
                prediction = train_predict(
                    base_matrix[source["_unit"].to_numpy(np.int64)],
                    causal,
                    source_positions,
                    target_base,
                    hold_history,
                    labels,
                    weights,
                    categorical_indices,
                    feature_names,
                    rounds,
                )
                save_component(name, debug, prediction)
            full_predictions[half_life] = prediction
        for months in fresh_windows:
            name = f"fold{fold_index}_fresh_m{months}"
            prediction = load_component(name, debug, len(hold_docs))
            if prediction is None:
                lower = cutoff - pd.DateOffset(months=months)
                fresh_mask = source_mask & (source["CREATIONTIMESTAMP"] >= lower)
                positions = np.flatnonzero(fresh_mask.to_numpy())
                prediction = train_predict(
                    base_matrix[source["_unit"].to_numpy(np.int64)],
                    causal,
                    positions,
                    target_base,
                    hold_history,
                    labels,
                    item_weights[positions],
                    categorical_indices,
                    feature_names,
                    rounds,
                )
                save_component(name, debug, prediction)
            fresh_predictions[months] = prediction
        output = {
            "cutoff": cutoff,
            "source": source_docs,
            "target": hold_docs,
            "labels": hold_docs["_label"].to_numpy(np.int8),
            "weights": hold_docs["doc_weight"].to_numpy(np.float32),
            "segments": staleness_segments(ages["sold_sales_area"]),
            "full": full_predictions,
            "fresh": fresh_predictions,
            "prior_ratios": forecast_prior_ratios(source_docs, hold_docs),
        }
        outputs.append(output)
        print(f"[temporal_lgb] internal fold {fold_index + 1} components complete", flush=True)
    return outputs


def select_design(folds: list[dict], debug: bool) -> dict:
    if debug:
        return {"half_life": 180, "fresh_months": 3, "weights": (0.5, 0.5, 0.5, 0.5), "prior": False}
    candidates = []
    for half_life in FULL_HALF_LIVES:
        for months in FRESH_MONTHS:
            selected_weights = []
            for segment in range(4):
                weight_scores = []
                for blend_weight in BLEND_WEIGHTS:
                    fold_scores = []
                    for fold in folds:
                        mask = fold["segments"] == segment
                        if mask.any():
                            prediction = (1.0 - blend_weight) * fold["full"][half_life][mask] + blend_weight * fold["fresh"][months][mask]
                            fold_scores.append(weighted_accuracy(prediction, fold["labels"][mask], fold["weights"][mask]))
                    score = np.mean(fold_scores) - 0.25 * np.ptp(fold_scores)
                    weight_scores.append(score)
                selected_weights.append(BLEND_WEIGHTS[int(np.argmax(weight_scores))])
            fold_scores = []
            for fold in folds:
                prediction = blend_predictions(fold["full"][half_life], fold["fresh"][months], fold["segments"], tuple(selected_weights))
                fold_scores.append(weighted_accuracy(prediction, fold["labels"], fold["weights"]))
            objective = float(np.mean(fold_scores) - 0.5 * np.ptp(fold_scores))
            candidates.append((objective, float(np.mean(fold_scores)), half_life, months, tuple(selected_weights), tuple(fold_scores)))
            print(
                f"[temporal_lgb] candidate half_life={half_life} fresh={months} weights={selected_weights} folds={[round(x, 6) for x in fold_scores]} objective={objective:.6f}",
                flush=True,
            )
    selected = max(candidates, key=lambda row: (row[0], row[1], -row[3]))
    half_life, months, weights = selected[2], selected[3], selected[4]
    improvements = []
    segment_ok = True
    for fold_index, fold in enumerate(folds):
        raw = blend_predictions(fold["full"][half_life], fold["fresh"][months], fold["segments"], weights)
        corrected = apply_prior(raw, fold["prior_ratios"])
        raw_score = weighted_accuracy(raw, fold["labels"], fold["weights"])
        corrected_score = weighted_accuracy(corrected, fold["labels"], fold["weights"])
        improvements.append(corrected_score - raw_score)
        strata = []
        for segment in range(4):
            mask = fold["segments"] == segment
            count = int(fold["weights"][mask].sum())
            if count == 0:
                continue
            raw_segment = weighted_accuracy(raw[mask], fold["labels"][mask], fold["weights"][mask])
            corrected_segment = weighted_accuracy(corrected[mask], fold["labels"][mask], fold["weights"][mask])
            delta = corrected_segment - raw_segment
            if count >= 0.05 * fold["weights"].sum() and delta < -0.001:
                segment_ok = False
            strata.append(f"{SEGMENT_NAMES[segment]}:{count}/{raw_segment:.4f}/{corrected_segment:.4f}")
        print(
            f"[temporal_lgb] fold {fold_index + 1} prior raw={raw_score:.6f} corrected={corrected_score:.6f} delta={corrected_score - raw_score:.6f} strata={' '.join(strata)}",
            flush=True,
        )
    prior_enabled = all(delta >= 0.002 for delta in improvements) and segment_ok
    print(f"[temporal_lgb] selected half_life={half_life} fresh={months} weights={weights} prior={prior_enabled}", flush=True)
    return {"half_life": half_life, "fresh_months": months, "weights": weights, "prior": prior_enabled}


def final_chain_components(
    chain: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
    causal: np.ndarray,
    base_matrix: np.ndarray,
    categorical_indices: list[int],
    feature_names: list[str],
    design: dict,
    rounds: int,
    debug: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = DEBUG_HISTORY_KEYS if debug else HISTORY_KEYS
    target_history, ages = snapshot_history_matrix(source, target, keys)
    target_base = base_matrix[target["_unit"].to_numpy(np.int64)]
    all_source = pd.concat([source], ignore_index=True)
    source_positions = source["_source_order"].to_numpy(np.int64)
    source_units = source["_unit"].to_numpy(np.int64)
    source_base = base_matrix[source_units]
    labels = source["_label"].to_numpy(np.int8)
    cutoff = target["CREATIONTIMESTAMP"].min().normalize()
    half_life = design["half_life"]
    tag = "inf" if half_life is None else str(half_life)
    full_name = f"{chain}_full_h{tag}"
    full = load_component(full_name, debug, len(target))
    if full is None:
        age_days = (cutoff - source["CREATIONTIMESTAMP"]).dt.total_seconds().to_numpy() / 86400.0
        decay = np.ones(len(source), dtype=np.float32) if half_life is None else np.power(0.5, age_days / half_life).astype(np.float32)
        weights = source["doc_weight"].to_numpy(np.float32) * decay
        full_positions = np.arange(len(source), dtype=np.int64)
        if debug:
            full_positions = np.flatnonzero((source["CREATIONTIMESTAMP"] >= cutoff - pd.DateOffset(months=12)).to_numpy())
        full = train_predict(
            source_base,
            causal,
            full_positions,
            target_base,
            target_history,
            labels,
            weights[full_positions],
            categorical_indices,
            feature_names,
            rounds,
        )
        save_component(full_name, debug, full)
    months = design["fresh_months"]
    fresh_name = f"{chain}_fresh_m{months}"
    fresh = load_component(fresh_name, debug, len(target))
    if fresh is None:
        lower = cutoff - pd.DateOffset(months=months)
        positions = np.flatnonzero((source["CREATIONTIMESTAMP"] >= lower).to_numpy())
        fresh = train_predict(
            source_base,
            causal,
            positions,
            target_base,
            target_history,
            labels,
            source["doc_weight"].to_numpy(np.float32)[positions],
            categorical_indices,
            feature_names,
            rounds,
        )
        save_component(fresh_name, debug, fresh)
    segments = staleness_segments(ages["sold_sales_area"])
    ratios = forecast_prior_ratios(all_source, target)
    return full, fresh, segments, ratios


def append_artifact_registry(paths: list[Path]) -> None:
    registry = shared_cache_dir() / "artifacts.json"
    lock_path = shared_cache_dir() / f"{VERSION}.artifacts.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entries = []
        if registry.exists():
            try:
                entries = json.loads(registry.read_text())
            except Exception:
                entries = []
        known = {entry.get("name") for entry in entries}
        for path in paths:
            name = f"{VERSION}:{path.name}"
            if path.exists() and name not in known:
                entries.append(
                    {
                        "name": name,
                        "path": path.name,
                        "description": "Lane 3 causal document matrix or FULL/FRESH component predictions",
                        "content_key": VERSION,
                        "rebuild_hint": "Run main.py; each component is rebuilt only when absent or shape-invalid",
                    }
                )
        temporary = registry.with_suffix(".lane3.tmp")
        temporary.write_text(json.dumps(entries, indent=2))
        temporary.replace(registry)


def run(debug: bool) -> None:
    started = time.time()
    checkpoint = started
    data = load_document_data(debug)
    checkpoint = phase("document matrix ready", checkpoint)
    source, causal, history_names = get_history_data(data, debug)
    checkpoint = phase("causal label histories ready", checkpoint)
    docs = data["docs"]
    feature_columns = data["feature_columns"]
    categorical_indices = [feature_columns.index(column) for column in data["categorical_columns"]]
    feature_names = feature_columns + history_names
    base_matrix = docs[feature_columns].to_numpy(np.float32)
    rounds = 50 if debug else 800
    folds = build_internal_predictions(
        data,
        source[source["_split"] == "train"].reset_index(drop=True).assign(_source_order=lambda frame: np.arange(len(frame))),
        causal[source["_split"].to_numpy() == "train"],
        base_matrix,
        categorical_indices,
        feature_names,
        rounds,
        debug,
    )
    design = select_design(folds, debug)
    checkpoint = phase("internal forward selection complete", checkpoint)
    train_source = source[source["_split"] == "train"].copy().reset_index(drop=True)
    train_causal = causal[source["_split"].to_numpy() == "train"]
    train_source["_source_order"] = np.arange(len(train_source), dtype=np.int32)
    val_target = docs[docs["_split"] == "val"].copy().reset_index(drop=True)
    full_a, fresh_a, segments_a, ratios_a = final_chain_components(
        "model_a",
        train_source,
        val_target,
        train_causal,
        base_matrix,
        categorical_indices,
        feature_names,
        design,
        rounds,
        debug,
    )
    val_document_prediction = blend_predictions(full_a, fresh_a, segments_a, design["weights"])
    if design["prior"]:
        val_document_prediction = apply_prior(val_document_prediction, ratios_a)
    val_prediction = val_document_prediction[data["row_maps"]["val"] - val_target["_unit"].min()]
    checkpoint = phase("Model A validation predictions frozen", checkpoint)
    combined_source = source.copy().reset_index(drop=True)
    combined_source["_source_order"] = np.arange(len(combined_source), dtype=np.int32)
    test_target = docs[docs["_split"] == "test"].copy().reset_index(drop=True)
    full_b, fresh_b, segments_b, ratios_b = final_chain_components(
        "model_b",
        combined_source,
        test_target,
        causal,
        base_matrix,
        categorical_indices,
        feature_names,
        design,
        rounds,
        debug,
    )
    test_document_prediction = blend_predictions(full_b, fresh_b, segments_b, design["weights"])
    if design["prior"]:
        test_document_prediction = apply_prior(test_document_prediction, ratios_b)
    test_prediction = test_document_prediction[data["row_maps"]["test"] - test_target["_unit"].min()]
    checkpoint = phase("Model B test predictions complete", checkpoint)
    save_predictions(val_prediction.astype(np.float32), test_prediction.astype(np.float32))
    metrics = {
        "version": VERSION,
        "debug": debug,
        "rounds": rounds,
        "design": design,
        "elapsed_seconds": time.time() - started,
        "validation_staleness_items": {
            SEGMENT_NAMES[index]: int((val_target.loc[segments_a == index, "item_count"]).sum()) for index in range(4)
        },
        "test_staleness_items": {
            SEGMENT_NAMES[index]: int((test_target.loc[segments_b == index, "item_count"]).sum()) for index in range(4)
        },
    }
    (run_data_dir() / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    artifacts = list(shared_cache_dir().glob(f"{VERSION}_*"))
    append_artifact_registry(artifacts)
    phase("pipeline finished", checkpoint)
