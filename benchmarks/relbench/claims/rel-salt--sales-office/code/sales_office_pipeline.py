import fcntl
import json
import os
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from kapso_datasets.common import load_task, save_predictions


TARGET = "SALESOFFICE"
TIME = "CREATIONTIMESTAMP"
DOC = "SALESDOCUMENT"
HALF_LIVES = (45, 90, 180)
POSITIVE_WEIGHTS = (1, 2, 4)
ALPHAS = (0.5, 1.0, 2.0)
PRIOR_MULTIPLIERS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
ORIGINS = (
    (pd.Timestamp("2018-11-01"), pd.Timestamp("2019-04-01")),
    (pd.Timestamp("2019-04-01"), pd.Timestamp("2019-09-01")),
    (pd.Timestamp("2019-09-01"), pd.Timestamp("2020-02-01")),
)
HEADER_COLUMNS = (
    DOC,
    "SALESDOCUMENTTYPE",
    "SALESORGANIZATION",
    "DISTRIBUTIONCHANNEL",
    "ORGANIZATIONDIVISION",
    "BILLINGCOMPANYCODE",
    "TRANSACTIONCURRENCY",
    TIME,
)
ITEM_COLUMNS = (
    DOC,
    "SALESDOCUMENTITEM",
    "SALESDOCUMENTITEMCATEGORY",
    "PRODUCT",
    "SOLDTOPARTY",
    "SHIPTOPARTY",
    "BILLTOPARTY",
    "PAYERPARTY",
    TIME,
    "ID",
)
ROLE_NAMES = ("sold", "ship", "bill", "payer")
HEADER_CATEGORICALS = (
    "SALESDOCUMENTTYPE",
    "SALESORGANIZATION",
    "DISTRIBUTIONCHANNEL",
    "ORGANIZATIONDIVISION",
    "BILLINGCOMPANYCODE",
    "TRANSACTIONCURRENCY",
)


def elapsed(start, phase):
    print(f"[pipeline] phase={phase} elapsed_seconds={time.time() - start:.2f}", flush=True)


def factorize(values):
    return pd.factorize(values, sort=False, use_na_sentinel=True)[0].astype(np.int32) + 1


def combine(left, right, right_size):
    return left.astype(np.int64) * np.int64(right_size + 1) + right.astype(np.int64)


def append_once(path, marker, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        current = handle.read()
        if marker not in current:
            handle.seek(0, os.SEEK_END)
            handle.write(text)
            handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def record_database_notes(shared):
    marker = "lane0-hurdle-database-v1"
    text = (
        "\n### lane0-hurdle-database-v1\n"
        "- All 500,907 task documents join one-to-one to allowed salesdocument headers at identical creation timestamps.\n"
        "- All task documents have same-document items; no joined item timestamp is later than its seed.\n"
        "- Party geography uses salesdocumentitem role IDs -> customer.CUSTOMER -> customer.ADDRESSID -> address.ADDRESSID.\n"
        "- Items per document are heavy-tailed: median 1-2 and split-specific 99th percentiles 48-57; product histories therefore aggregate at document-product granularity while the decoder retains item multiplicity.\n"
    )
    append_once(shared / "table_information.md", marker, text)


def record_feature_notes(shared, status, outcome):
    marker = f"lane0-hurdle-features-v1-{status}"
    text = (
        f"\n### Document hurdle cascade ({marker})\n"
        f"- run/experiment: generic_exp_0 lane 0 | status: {status}\n"
        "- what: Header/time, item/product/category, four party roles, four-role geography, causal label-free activity, and causal label-derived party/product/cohort histories feeding a LightGBM hurdle and smoothed histogram decoder.\n"
        f"- outcome: {outcome}\n"
        "- takeaway: Select the conservative flip operating point on expanding five-month origins and keep validation/test label chains separate.\n"
    )
    append_once(shared / "features_history.md", marker, text)


def build_document_data(context, start):
    frames = []
    for split_code, (split_name, table) in enumerate(
        (("train", context.train), ("val", context.val), ("test", context.test))
    ):
        columns = [TIME, DOC] + ([TARGET] if split_name != "test" else [])
        frame = table.df.loc[:, columns].copy()
        frame["_split"] = split_name
        frame["_split_code"] = split_code
        frame["_row_id"] = np.arange(len(frame), dtype=np.int64)
        if split_name == "test":
            frame[TARGET] = -1
        frames.append(frame)
    documents = pd.concat(frames, ignore_index=True)
    documents["_doc_index"] = np.arange(len(documents), dtype=np.int64)
    headers = context.db.table_dict["salesdocument"].df.loc[:, list(HEADER_COLUMNS)].copy()
    headers = headers.rename(columns={TIME: "_header_time"})
    documents = documents.merge(headers, on=DOC, how="left", validate="one_to_one", sort=False)
    if documents["SALESORGANIZATION"].isna().any():
        raise RuntimeError("header join left unmatched task documents")
    if not documents[TIME].equals(documents["_header_time"]):
        raise RuntimeError("header timestamp differs from task timestamp")
    for column in HEADER_CATEGORICALS:
        documents[f"{column}_code"] = factorize(documents[column])
    area_columns = [f"{name}_code" for name in ("SALESORGANIZATION", "DISTRIBUTIONCHANNEL", "ORGANIZATIONDIVISION")]
    documents["area_code"] = factorize(pd.MultiIndex.from_frame(documents[area_columns]))
    documents["header_cohort_code"] = factorize(
        pd.MultiIndex.from_frame(
            documents[
                area_columns
                + ["SALESDOCUMENTTYPE_code", "BILLINGCOMPANYCODE_code", "TRANSACTIONCURRENCY_code"]
            ]
        )
    )
    timestamp = documents[TIME]
    documents["creation_month"] = timestamp.dt.month.astype(np.int16)
    documents["creation_weekday"] = timestamp.dt.weekday.astype(np.int16)
    documents["creation_hour"] = timestamp.dt.hour.astype(np.int16)
    documents["creation_day"] = ((timestamp - pd.Timestamp("2018-01-01")) / pd.Timedelta(days=1)).astype(np.float32)
    documents["recency_index"] = documents[TIME].rank(method="dense").astype(np.float32)
    customer = context.db.table_dict["customer"].df.loc[:, ["CUSTOMER", "ADDRESSID"]].copy().reset_index(drop=True)
    customer["customer_code"] = np.arange(1, len(customer) + 1, dtype=np.int32)
    address_ids = pd.Index(customer["ADDRESSID"].unique())
    address = context.db.table_dict["address"].df.loc[:, ["ADDRESSID", "COUNTRY", "REGION"]]
    address = address[address["ADDRESSID"].isin(address_ids)].copy()
    customer = customer.merge(address, on="ADDRESSID", how="left", validate="one_to_one", sort=False)
    customer["country_code"] = factorize(customer["COUNTRY"])
    customer["region_code"] = factorize(customer["REGION"])
    customer_map = customer.set_index("CUSTOMER")["customer_code"]
    country_by_customer = np.zeros(len(customer) + 1, dtype=np.int32)
    region_by_customer = np.zeros(len(customer) + 1, dtype=np.int32)
    country_by_customer[customer["customer_code"].to_numpy()] = customer["country_code"].to_numpy()
    region_by_customer[customer["customer_code"].to_numpy()] = customer["region_code"].to_numpy()
    items = context.db.table_dict["salesdocumentitem"].df.loc[:, list(ITEM_COLUMNS)].copy()
    items = items[items[DOC].isin(documents[DOC])].copy()
    if items[TIME].isna().any():
        raise RuntimeError("item timestamp is missing")
    items["product_code"] = factorize(items["PRODUCT"])
    items["category_code"] = factorize(items["SALESDOCUMENTITEMCATEGORY"])
    for role, source in zip(ROLE_NAMES, ("SOLDTOPARTY", "SHIPTOPARTY", "BILLTOPARTY", "PAYERPARTY")):
        mapped = items[source].map(customer_map)
        if mapped.isna().any():
            raise RuntimeError(f"customer join failed for {source}")
        items[f"{role}_code"] = mapped.astype(np.int32)
    aggregations = {
        "item_count": ("product_code", "size"),
        "distinct_products": ("product_code", "nunique"),
        "distinct_categories": ("category_code", "nunique"),
    }
    for role in ROLE_NAMES:
        aggregations[f"{role}_code"] = (f"{role}_code", "first")
        aggregations[f"{role}_within_document"] = (f"{role}_code", "nunique")
    item_aggregate = items.groupby(DOC, sort=False).agg(**aggregations).reset_index()
    product_counts = items.groupby([DOC, "product_code"], sort=False).size().rename("item_occurrences").reset_index()
    category_counts = items.groupby([DOC, "category_code"], sort=False).size().rename("category_occurrences").reset_index()
    product_mode = (
        product_counts.sort_values([DOC, "item_occurrences", "product_code"], ascending=[True, False, True])
        .drop_duplicates(DOC)
        .loc[:, [DOC, "product_code"]]
        .rename(columns={"product_code": "product_mode_code"})
    )
    category_mode = (
        category_counts.sort_values([DOC, "category_occurrences", "category_code"], ascending=[True, False, True])
        .drop_duplicates(DOC)
        .loc[:, [DOC, "category_code"]]
        .rename(columns={"category_code": "category_mode_code"})
    )
    documents = documents.merge(item_aggregate, on=DOC, how="left", validate="one_to_one", sort=False)
    documents = documents.merge(product_mode, on=DOC, how="left", validate="one_to_one", sort=False)
    documents = documents.merge(category_mode, on=DOC, how="left", validate="one_to_one", sort=False)
    if documents["item_count"].isna().any():
        raise RuntimeError("item aggregation left unmatched task documents")
    roles = documents[[f"{role}_code" for role in ROLE_NAMES]].to_numpy(dtype=np.int32)
    sorted_roles = np.sort(roles, axis=1)
    documents["distinct_role_count"] = 1 + (sorted_roles[:, 1:] != sorted_roles[:, :-1]).sum(axis=1)
    equality_mask = np.zeros(len(documents), dtype=np.int16)
    bit = 0
    for left in range(4):
        for right in range(left + 1, 4):
            equality_mask += ((roles[:, left] == roles[:, right]).astype(np.int16) << bit)
            bit += 1
    documents["role_equality_mask"] = equality_mask
    max_area = int(documents["area_code"].max())
    max_dtype = int(documents["SALESDOCUMENTTYPE_code"].max())
    max_region = int(customer["region_code"].max())
    for role in ROLE_NAMES:
        customer_code = documents[f"{role}_code"].to_numpy(dtype=np.int32)
        documents[f"{role}_country_code"] = country_by_customer[customer_code]
        documents[f"{role}_region_code"] = region_by_customer[customer_code]
        documents[f"{role}_area_code"] = combine(customer_code, documents["area_code"].to_numpy(), max_area)
        documents[f"{role}_doctype_code"] = combine(customer_code, documents["SALESDOCUMENTTYPE_code"].to_numpy(), max_dtype)
        documents[f"{role}_doctype_area_code"] = combine(
            documents[f"{role}_doctype_code"].to_numpy(), documents["area_code"].to_numpy(), max_area
        )
        documents[f"{role}_region_area_code"] = combine(
            documents[f"{role}_region_code"].to_numpy(), documents["area_code"].to_numpy(), max_area
        )
    documents["product_mode_hash_1021"] = documents["product_mode_code"].astype(np.int64) % 1021
    documents["product_mode_hash_4093"] = documents["product_mode_code"].astype(np.int64) % 4093
    documents["category_mode_hash"] = documents["category_mode_code"].astype(np.int64) % 257
    doc_meta = documents.set_index(DOC)[["_doc_index", TIME, "area_code"]]
    document_products = product_counts.merge(doc_meta, left_on=DOC, right_index=True, how="left", validate="many_to_one")
    document_products["product_area_code"] = combine(
        document_products["product_code"].to_numpy(), document_products["area_code"].to_numpy(), max_area
    )
    item_times = items[[DOC, TIME]].merge(doc_meta[[TIME]], left_on=DOC, right_index=True, suffixes=("_item", "_seed"))
    if (item_times[f"{TIME}_item"] > item_times[f"{TIME}_seed"]).any():
        raise RuntimeError("joined an item after its document seed")
    documents = add_label_free_single_histories(documents)
    documents, document_products = add_label_free_product_histories(documents, document_products)
    categorical = [
        *(f"{column}_code" for column in HEADER_CATEGORICALS),
        "area_code",
        "header_cohort_code",
        "creation_month",
        "creation_weekday",
        "creation_hour",
        "product_mode_code",
        "category_mode_code",
        "product_mode_hash_1021",
        "product_mode_hash_4093",
        "category_mode_hash",
        "role_equality_mask",
    ]
    for role in ROLE_NAMES:
        categorical.extend(
            [
                f"{role}_code",
                f"{role}_country_code",
                f"{role}_region_code",
            ]
        )
    base_numeric = [
        "creation_day",
        "recency_index",
        "item_count",
        "distinct_products",
        "distinct_categories",
        "distinct_role_count",
        "area_prior_count",
        "area_prior_recency",
        "product_prior_count_mean",
        "product_prior_count_max",
        "product_prior_count_sum",
        "product_prior_recency_min",
        "product_area_prior_count_mean",
        "product_area_prior_count_max",
        "product_area_prior_recency_min",
    ]
    for role in ROLE_NAMES:
        base_numeric.extend(
            [
                f"{role}_within_document",
                f"{role}_prior_count",
                f"{role}_prior_recency",
                f"{role}_area_prior_count",
                f"{role}_area_prior_recency",
            ]
        )
    documents = documents.sort_values("_doc_index").reset_index(drop=True)
    elapsed(start, "document_features")
    return documents, document_products, list(dict.fromkeys(categorical)), base_numeric


def add_prior_history(frame, key, prefix):
    order = frame.sort_values([TIME, DOC], kind="mergesort").index
    work = frame.loc[order, [key, TIME]]
    count = work.groupby(key, sort=False).cumcount().astype(np.float32)
    previous = work.groupby(key, sort=False)[TIME].shift()
    recency = ((work[TIME] - previous) / pd.Timedelta(days=1)).astype(np.float32)
    frame.loc[order, f"{prefix}_prior_count"] = count.to_numpy()
    frame.loc[order, f"{prefix}_prior_recency"] = recency.fillna(9999.0).to_numpy()
    return frame


def add_label_free_single_histories(documents):
    documents = add_prior_history(documents, "area_code", "area")
    for role in ROLE_NAMES:
        documents = add_prior_history(documents, f"{role}_code", role)
        documents = add_prior_history(documents, f"{role}_area_code", f"{role}_area")
    return documents


def add_label_free_product_histories(documents, document_products):
    product_rows = document_products.sort_values([TIME, DOC], kind="mergesort").copy()
    for key, prefix in (("product_code", "product"), ("product_area_code", "product_area")):
        product_rows[f"{prefix}_prior_count"] = product_rows.groupby(key, sort=False).cumcount().astype(np.float32)
        previous = product_rows.groupby(key, sort=False)[TIME].shift()
        product_rows[f"{prefix}_prior_recency"] = (
            (product_rows[TIME] - previous) / pd.Timedelta(days=1)
        ).astype(np.float32).fillna(9999.0)
    aggregations = {
        "product_prior_count_mean": ("product_prior_count", "mean"),
        "product_prior_count_max": ("product_prior_count", "max"),
        "product_prior_count_sum": ("product_prior_count", "sum"),
        "product_prior_recency_min": ("product_prior_recency", "min"),
        "product_area_prior_count_mean": ("product_area_prior_count", "mean"),
        "product_area_prior_count_max": ("product_area_prior_count", "max"),
        "product_area_prior_recency_min": ("product_area_prior_recency", "min"),
    }
    aggregate = product_rows.groupby("_doc_index", sort=False).agg(**aggregations)
    for column in aggregate.columns:
        documents[column] = documents["_doc_index"].map(aggregate[column]).astype(np.float32)
    return documents, product_rows


def history_keys():
    keys = []
    for role in ROLE_NAMES:
        keys.extend(
            [
                (f"{role}_code", role),
                (f"{role}_area_code", f"{role}_area"),
                (f"{role}_doctype_code", f"{role}_doctype"),
                (f"{role}_region_area_code", f"{role}_region_area"),
            ]
        )
    keys.extend((("area_code", "area"), ("header_cohort_code", "header_cohort")))
    return keys


def history_stats(reference, key):
    columns = [key, TIME, DOC, TARGET]
    values = reference.loc[:, columns].copy()
    values = values[values[key].notna()].copy()
    values["_positive"] = (values[TARGET] != 0).astype(np.float64)
    day = ((values[TIME] - pd.Timestamp("2018-01-01")) / pd.Timedelta(days=1)).to_numpy()
    aggregation = {"total": ("_positive", "size"), "nonzero": ("_positive", "sum")}
    for half_life in HALF_LIVES:
        weight = np.exp(day / half_life)
        values[f"_weight_{half_life}"] = weight
        values[f"_positive_weight_{half_life}"] = weight * values["_positive"].to_numpy()
        aggregation[f"weight_{half_life}"] = (f"_weight_{half_life}", "sum")
        aggregation[f"positive_weight_{half_life}"] = (f"_positive_weight_{half_life}", "sum")
    stats = values.groupby([key, TIME], sort=False).agg(**aggregation).reset_index()
    stats = stats.sort_values([key, TIME], kind="mergesort")
    cumulative = ["total", "nonzero"]
    for half_life in HALF_LIVES:
        cumulative.extend((f"weight_{half_life}", f"positive_weight_{half_life}"))
    stats[cumulative] = stats.groupby(key, sort=False)[cumulative].cumsum()
    positives = values[values["_positive"] == 1].sort_values([key, TIME, DOC], kind="mergesort").copy()
    if len(positives):
        positives["label_count"] = positives.groupby([key, TARGET], sort=False).cumcount() + 1
        positives["positive_sequence"] = positives.groupby(key, sort=False).cumcount() + 1
        positives["max_label_count"] = positives.groupby(key, sort=False)["label_count"].cummax()
        positives["purity"] = positives["max_label_count"] / positives["positive_sequence"]
        positives = positives.drop_duplicates([key, TIME], keep="last")
        positives = positives[[key, TIME, TARGET, "purity"]].rename(
            columns={TIME: "last_positive_time", TARGET: "last_label"}
        )
        stats = pd.merge_asof(
            stats.sort_values([TIME, key], kind="mergesort"),
            positives.sort_values(["last_positive_time", key], kind="mergesort"),
            left_on=TIME,
            right_on="last_positive_time",
            by=key,
            direction="backward",
            allow_exact_matches=True,
        )
    else:
        stats["last_positive_time"] = pd.NaT
        stats["last_label"] = 0
        stats["purity"] = 0.0
    return stats.sort_values([TIME, key], kind="mergesort")


def query_history(reference, queries, key, prefix):
    stats = history_stats(reference, key)
    query = queries[[key, TIME]].copy()
    query["_query_position"] = np.arange(len(query), dtype=np.int64)
    merged = pd.merge_asof(
        query.sort_values([TIME, key], kind="mergesort"),
        stats,
        on=TIME,
        by=key,
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("_query_position")
    output = pd.DataFrame(index=queries.index)
    total = merged["total"].fillna(0).to_numpy(dtype=np.float32)
    nonzero = merged["nonzero"].fillna(0).to_numpy(dtype=np.float32)
    output[f"{prefix}_label_count"] = total
    output[f"{prefix}_nonzero_count"] = nonzero
    output[f"{prefix}_nonzero_frequency"] = nonzero / np.maximum(total, 1.0)
    for half_life in HALF_LIVES:
        numerator = merged[f"positive_weight_{half_life}"].fillna(0).to_numpy(dtype=np.float64)
        denominator = merged[f"weight_{half_life}"].fillna(0).to_numpy(dtype=np.float64)
        output[f"{prefix}_decayed_frequency_{half_life}"] = (numerator / np.maximum(denominator, 1e-30)).astype(np.float32)
    last_time = merged["last_positive_time"]
    recency = ((merged[TIME] - last_time) / pd.Timedelta(days=1)).astype(np.float32)
    output[f"{prefix}_last_nonzero_recency"] = recency.fillna(9999.0).to_numpy()
    output[f"{prefix}_last_label"] = merged["last_label"].fillna(0).to_numpy(dtype=np.float32)
    output[f"{prefix}_purity"] = merged["purity"].fillna(0).to_numpy(dtype=np.float32)
    return output


def query_product_history(reference, queries, document_products, prefix_key, prefix):
    ref_docs = reference[["_doc_index", TARGET]]
    ref_products = document_products.merge(ref_docs, on="_doc_index", how="inner", validate="many_to_one")
    query_products = document_products[document_products["_doc_index"].isin(queries["_doc_index"])].copy()
    stats = history_stats(ref_products.rename(columns={prefix_key: "_history_key"}), "_history_key")
    query = query_products[["_doc_index", prefix_key, TIME]].rename(columns={prefix_key: "_history_key"})
    query["_query_position"] = np.arange(len(query), dtype=np.int64)
    merged = pd.merge_asof(
        query.sort_values([TIME, "_history_key"], kind="mergesort"),
        stats,
        on=TIME,
        by="_history_key",
        direction="backward",
        allow_exact_matches=False,
    )
    total = merged["total"].fillna(0).astype(np.float32)
    nonzero = merged["nonzero"].fillna(0).astype(np.float32)
    merged[f"{prefix}_label_count"] = total
    merged[f"{prefix}_nonzero_count"] = nonzero
    merged[f"{prefix}_nonzero_frequency"] = nonzero / np.maximum(total, 1.0)
    for half_life in HALF_LIVES:
        merged[f"{prefix}_decayed_frequency_{half_life}"] = (
            merged[f"positive_weight_{half_life}"].fillna(0)
            / np.maximum(merged[f"weight_{half_life}"].fillna(0), 1e-30)
        ).astype(np.float32)
    merged[f"{prefix}_last_nonzero_recency"] = (
        (merged[TIME] - merged["last_positive_time"]) / pd.Timedelta(days=1)
    ).astype(np.float32).fillna(9999.0)
    merged[f"{prefix}_last_label"] = merged["last_label"].fillna(0).astype(np.float32)
    merged[f"{prefix}_purity"] = merged["purity"].fillna(0).astype(np.float32)
    feature_roots = (
        "label_count",
        "nonzero_count",
        "nonzero_frequency",
        "last_nonzero_recency",
        "last_label",
        "purity",
        *(f"decayed_frequency_{half_life}" for half_life in HALF_LIVES),
    )
    output = pd.DataFrame(index=queries.index)
    grouped = merged.groupby("_doc_index", sort=False)
    query_index = queries.set_index("_doc_index")
    for root in feature_roots:
        column = f"{prefix}_{root}"
        if root in ("last_nonzero_recency",):
            aggregate = grouped[column].min()
        else:
            aggregate = grouped[column].max()
        output[column] = queries["_doc_index"].map(aggregate).fillna(0 if root != "last_nonzero_recency" else 9999).to_numpy()
    return output


def build_label_features(reference, queries, document_products, start, phase_name):
    pieces = []
    for key, prefix in history_keys():
        pieces.append(query_history(reference, queries, key, prefix))
    pieces.append(query_product_history(reference, queries, document_products, "product_code", "product"))
    pieces.append(query_product_history(reference, queries, document_products, "product_area_code", "product_area"))
    features = pd.concat(pieces, axis=1)
    features = features.astype(np.float32)
    features.index = queries.index
    elapsed(start, phase_name)
    return features


def selected_columns(base_categorical, base_numeric, label_features, half_life):
    label_columns = [column for column in label_features.columns if "decayed_frequency_" not in column]
    label_columns.extend(column for column in label_features.columns if column.endswith(f"_{half_life}"))
    return list(dict.fromkeys(base_categorical + base_numeric + label_columns))


def feature_matrix(documents, label_features, columns):
    base = documents.loc[label_features.index, [column for column in columns if column in documents.columns]].copy()
    for column in columns:
        if column in label_features.columns:
            base[column] = label_features[column]
    base = base.loc[:, columns]
    for column in base.columns:
        if base[column].dtype == "object":
            raise RuntimeError(f"object feature reached LightGBM: {column}")
    return base.replace([np.inf, -np.inf], np.nan).fillna(0)


def fit_binary(matrix, labels, positive_weight, trees=300):
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=trees,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=30,
        reg_lambda=5.0,
        colsample_bytree=0.85,
        scale_pos_weight=positive_weight,
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")),
        verbosity=-1,
        random_state=1337,
        deterministic=True,
        force_col_wise=True,
    )
    model.fit(matrix, labels, callbacks=[lgb.log_evaluation(0)])
    return model


def histogram_map(keys, labels, weights=None):
    result = {}
    if weights is None:
        weights = np.ones(len(keys), dtype=np.float64)
    for key, label, weight in zip(keys, labels, weights):
        label = int(label)
        if label <= 0:
            continue
        integer_key = int(key)
        histogram = result.get(integer_key)
        if histogram is None:
            histogram = np.zeros(29, dtype=np.float64)
            result[integer_key] = histogram
        histogram[label - 1] += float(weight)
    return result


def lookup_histograms(keys, mapping):
    output = np.zeros((len(keys), 29), dtype=np.float64)
    for position, key in enumerate(keys):
        histogram = mapping.get(int(key))
        if histogram is not None:
            output[position] = histogram
    return output


def smooth_histograms(histograms, prior, alpha):
    return (histograms + alpha * prior[None, :]) / (histograms.sum(axis=1, keepdims=True) + alpha)


def conditional_distribution(reference, queries, document_products, cutoff, half_life, alpha):
    positive = reference[reference[TARGET] != 0].copy()
    age = ((cutoff - positive[TIME]) / pd.Timedelta(days=1)).clip(lower=0).to_numpy()
    recent_weight = np.exp(-age / half_life)
    prior_counts = np.bincount(
        positive[TARGET].to_numpy(dtype=np.int64) - 1, weights=recent_weight, minlength=29
    ).astype(np.float64)
    prior = (prior_counts + 1e-3) / (prior_counts.sum() + 29e-3)
    labels = positive[TARGET].to_numpy(dtype=np.int64)
    mappings = {}
    source_columns = (
        "sold_area_code",
        "ship_area_code",
        "bill_area_code",
        "payer_area_code",
        "sold_doctype_area_code",
        "header_cohort_code",
    )
    for column in source_columns:
        mappings[column] = histogram_map(positive[column].to_numpy(), labels)
    region_keys = np.concatenate([positive[f"{role}_region_area_code"].to_numpy() for role in ROLE_NAMES])
    region_labels = np.tile(labels, len(ROLE_NAMES))
    mappings["region_area"] = histogram_map(region_keys, region_labels)
    positive_products = document_products.merge(
        positive[["_doc_index", TARGET]], on="_doc_index", how="inner", validate="many_to_one"
    )
    mappings["product_area_code"] = histogram_map(
        positive_products["product_area_code"].to_numpy(),
        positive_products[TARGET].to_numpy(),
        positive_products["item_occurrences"].to_numpy(dtype=np.float64),
    )
    product_queries = document_products[document_products["_doc_index"].isin(queries["_doc_index"])].copy()
    product_hist = lookup_histograms(product_queries["product_area_code"].to_numpy(), mappings["product_area_code"])
    product_hist *= product_queries["item_occurrences"].to_numpy(dtype=np.float64)[:, None]
    product_frame = pd.DataFrame(product_hist)
    product_frame["_doc_index"] = product_queries["_doc_index"].to_numpy()
    product_hist = product_frame.groupby("_doc_index", sort=False).sum().reindex(queries["_doc_index"]).fillna(0).to_numpy()
    product_q = smooth_histograms(product_hist, prior, alpha)
    raw_party = {}
    party_q = {}
    for role in ROLE_NAMES:
        column = f"{role}_area_code"
        raw_party[role] = lookup_histograms(queries[column].to_numpy(), mappings[column])
        party_q[role] = smooth_histograms(raw_party[role], prior, alpha)
    sold_doc_hist = lookup_histograms(
        queries["sold_doctype_area_code"].to_numpy(), mappings["sold_doctype_area_code"]
    )
    sold_doc_q = smooth_histograms(sold_doc_hist, prior, alpha)
    region_components = []
    for role in ROLE_NAMES:
        hist = lookup_histograms(queries[f"{role}_region_area_code"].to_numpy(), mappings["region_area"])
        region_components.append(smooth_histograms(hist, prior, alpha))
    region_q = np.mean(region_components, axis=0)
    cohort_hist = lookup_histograms(queries["header_cohort_code"].to_numpy(), mappings["header_cohort_code"])
    cohort_q = smooth_histograms(cohort_hist, prior, alpha)
    bill_payer_q = 0.5 * (party_q["bill"] + party_q["payer"])
    remaining_q = 0.25 * (party_q["ship"] + sold_doc_q + region_q + cohort_q)
    distribution = 0.35 * product_q + 0.35 * party_q["sold"] + 0.15 * bill_payer_q + 0.15 * remaining_q
    best_labels = np.stack([raw_party[role].argmax(axis=1) + 1 for role in ROLE_NAMES], axis=1)
    totals = np.stack([raw_party[role].sum(axis=1) for role in ROLE_NAMES], axis=1)
    best_counts = np.stack([raw_party[role].max(axis=1) for role in ROLE_NAMES], axis=1)
    qualifying = (totals >= 2) & (best_counts / np.maximum(totals, 1) >= 0.8)
    consensus_label = np.zeros(len(queries), dtype=np.int16)
    consensus_mask = np.zeros(len(queries), dtype=bool)
    for position in range(len(queries)):
        eligible = best_labels[position, qualifying[position]]
        if len(eligible) < 2:
            continue
        counts = np.bincount(eligible, minlength=30)
        label = int(counts.argmax())
        if label > 0 and counts[label] >= 2:
            consensus_mask[position] = True
            consensus_label[position] = label
    if consensus_mask.any():
        consensus = np.full((consensus_mask.sum(), 29), 1e-9, dtype=np.float64)
        consensus[np.arange(consensus_mask.sum()), consensus_label[consensus_mask] - 1] = 1.0
        consensus /= consensus.sum(axis=1, keepdims=True)
        distribution[consensus_mask] = 0.75 * distribution[consensus_mask] + 0.25 * consensus
    distribution = np.maximum(distribution, 1e-9)
    tie_break = 1e-12 * np.arange(29, 0, -1, dtype=np.float64)
    distribution += tie_break[None, :]
    distribution /= distribution.sum(axis=1, keepdims=True)
    return distribution


def decoded_predictions(probability, distribution, multiplier):
    adjusted = multiplier * probability / np.maximum(multiplier * probability + 1.0 - probability, 1e-15)
    identity = distribution.argmax(axis=1) + 1
    identity_probability = distribution.max(axis=1)
    flip = adjusted * identity_probability > 1.0 - adjusted
    prediction = np.where(flip, identity, 0)
    return prediction, flip, adjusted


def prediction_scores(probability, distribution, multiplier):
    _, _, adjusted = decoded_predictions(probability, distribution, multiplier)
    scores = np.empty((len(probability), 30), dtype=np.float64)
    scores[:, 0] = 1.0 - adjusted
    scores[:, 1:] = adjusted[:, None] * distribution
    scores = np.maximum(scores, 1e-9)
    scores[:, 0] += 30e-12
    scores[:, 1:] += 1e-12 * np.arange(29, 0, -1, dtype=np.float64)[None, :]
    return scores


def select_parameters(documents, document_products, base_categorical, base_numeric, train, train_history, start):
    records = {}
    for fold_index, (origin, end) in enumerate(ORIGINS):
        fit_rows = train[train[TIME] < origin]
        holdout = train[(train[TIME] >= origin) & (train[TIME] < end)]
        frozen_history = build_label_features(
            fit_rows, holdout, document_products, start, f"origin_{fold_index + 1}_histories"
        )
        probabilities = {}
        for half_life in HALF_LIVES:
            columns = selected_columns(base_categorical, base_numeric, train_history, half_life)
            fit_matrix = feature_matrix(documents, train_history.loc[fit_rows.index], columns)
            holdout_matrix = feature_matrix(documents, frozen_history, columns)
            fit_target = (fit_rows[TARGET].to_numpy() != 0).astype(np.int8)
            for positive_weight in POSITIVE_WEIGHTS:
                model = fit_binary(fit_matrix, fit_target, positive_weight)
                probabilities[(positive_weight, half_life)] = model.predict_proba(holdout_matrix)[:, 1]
        true = holdout[TARGET].to_numpy(dtype=np.int64)
        for half_life in HALF_LIVES:
            for alpha in ALPHAS:
                distribution = conditional_distribution(
                    fit_rows, holdout, document_products, origin, half_life, alpha
                )
                for positive_weight in POSITIVE_WEIGHTS:
                    probability = probabilities[(positive_weight, half_life)]
                    for multiplier in PRIOR_MULTIPLIERS:
                        prediction, flip, _ = decoded_predictions(probability, distribution, multiplier)
                        correct_nonzero = int(((prediction == true) & (true != 0)).sum())
                        false_zero_flip = int((flip & (true == 0)).sum())
                        gain = correct_nonzero - false_zero_flip
                        key = (positive_weight, half_life, alpha, multiplier)
                        records.setdefault(key, []).append(
                            {
                                "gain": gain,
                                "rate": gain / len(holdout),
                                "flips": int(flip.sum()),
                                "correct_nonzero": correct_nonzero,
                                "false_zero_flip": false_zero_flip,
                                "count": len(holdout),
                                "positive_count": int((true != 0).sum()),
                            }
                        )
        print(
            f"[selection] origin={origin.date()} holdout_count={len(holdout)} "
            f"holdout_nonzero={int((true != 0).sum())}",
            flush=True,
        )
    eligible = []
    for key, folds in records.items():
        if all(fold["gain"] >= 0 for fold in folds):
            rates = np.array([fold["rate"] for fold in folds])
            eligible.append(
                {
                    "key": key,
                    "mean": float(rates.mean()),
                    "se": float(rates.std(ddof=1) / np.sqrt(len(rates))),
                    "flips": float(np.mean([fold["flips"] for fold in folds])),
                    "folds": folds,
                }
            )
    if not eligible:
        candidates = []
        for key, folds in records.items():
            rates = np.array([fold["rate"] for fold in folds])
            candidates.append(
                {
                    "key": key,
                    "mean": float(rates.mean()),
                    "se": float(rates.std(ddof=1) / np.sqrt(len(rates))),
                    "flips": float(np.mean([fold["flips"] for fold in folds])),
                    "folds": folds,
                }
            )
        eligible = sorted(candidates, key=lambda record: (min(f["gain"] for f in record["folds"]), record["mean"]), reverse=True)[:1]
    best_mean = max(record["mean"] for record in eligible)
    leader = max(eligible, key=lambda record: record["mean"])
    tolerance = max(1e-7, leader["se"])
    tied = [record for record in eligible if record["mean"] >= best_mean - tolerance]
    selected = min(tied, key=lambda record: (record["flips"], -record["mean"], record["key"]))
    positive_weight, half_life, alpha, multiplier = selected["key"]
    print(
        "[selection] selected="
        + json.dumps(
            {
                "positive_weight": positive_weight,
                "half_life": half_life,
                "alpha": alpha,
                "prior_multiplier": multiplier,
                "mean_gain_rate": selected["mean"],
                "standard_error": selected["se"],
                "mean_flips": selected["flips"],
                "folds": selected["folds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    elapsed(start, "internal_selection")
    return selected


def fit_chain(documents, document_products, base_categorical, base_numeric, reference, queries, reference_history, parameters, cutoff, trees, start, name):
    positive_weight, half_life, alpha, multiplier = parameters
    query_features = build_label_features(reference, queries, document_products, start, f"{name}_frozen_histories")
    columns = selected_columns(base_categorical, base_numeric, reference_history, half_life)
    reference_matrix = feature_matrix(documents, reference_history, columns)
    query_matrix = feature_matrix(documents, query_features, columns)
    target = (reference[TARGET].to_numpy() != 0).astype(np.int8)
    model = fit_binary(reference_matrix, target, positive_weight, trees=trees)
    probability = model.predict_proba(query_matrix)[:, 1]
    distribution = conditional_distribution(reference, queries, document_products, cutoff, half_life, alpha)
    scores = prediction_scores(probability, distribution, multiplier)
    prediction = scores.argmax(axis=1)
    print(
        f"[pipeline] chain={name} fit_rows={len(reference)} query_rows={len(queries)} "
        f"fit_nonzero={int(target.sum())} predicted_nonzero={int((prediction != 0).sum())}",
        flush=True,
    )
    elapsed(start, f"{name}_fit_predict")
    return scores


def validation_strata(validation, scores):
    prediction = scores.argmax(axis=1)
    true = validation[TARGET].to_numpy(dtype=np.int64)
    month = validation[TIME].dt.to_period("M").astype(str)
    for value in sorted(month.unique()):
        mask = month.to_numpy() == value
        print(
            f"[stratum] axis=month value={value} count={int(mask.sum())} "
            f"accuracy={float((prediction[mask] == true[mask]).mean()):.9f}",
            flush=True,
        )
    for name, mask in (("zero", true == 0), ("nonzero", true != 0)):
        print(
            f"[stratum] axis=target value={name} count={int(mask.sum())} "
            f"accuracy={float((prediction[mask] == true[mask]).mean()):.9f}",
            flush=True,
        )


def debug_run(documents, document_products, base_categorical, base_numeric, start):
    train = documents[documents["_split"] == "train"]
    validation = documents[documents["_split"] == "val"]
    test = documents[documents["_split"] == "test"]
    model_a_reference = train.tail(30000)
    model_a_history = build_label_features(
        model_a_reference, model_a_reference, document_products, start, "debug_model_a_causal_histories"
    )
    parameters = (4, 90, 1.0, 0.1)
    validation_scores = fit_chain(
        documents,
        document_products,
        base_categorical,
        base_numeric,
        model_a_reference,
        validation,
        model_a_history,
        parameters,
        pd.Timestamp("2020-02-01"),
        60,
        start,
        "debug_model_a",
    )
    combined = documents[documents["_split"].isin(("train", "val"))].tail(30000)
    model_b_history = build_label_features(
        combined, combined, document_products, start, "debug_model_b_causal_histories"
    )
    test_scores = fit_chain(
        documents,
        document_products,
        base_categorical,
        base_numeric,
        combined,
        test,
        model_b_history,
        parameters,
        pd.Timestamp("2020-07-01"),
        60,
        start,
        "debug_model_b",
    )
    return validation_scores, test_scores, {"key": parameters, "mean": 0.0, "folds": []}


def full_run(documents, document_products, base_categorical, base_numeric, start):
    train = documents[documents["_split"] == "train"]
    validation = documents[documents["_split"] == "val"]
    test = documents[documents["_split"] == "test"]
    train_history = build_label_features(train, train, document_products, start, "model_a_causal_histories")
    selection = select_parameters(
        documents, document_products, base_categorical, base_numeric, train, train_history, start
    )
    parameters = selection["key"]
    validation_scores = fit_chain(
        documents,
        document_products,
        base_categorical,
        base_numeric,
        train,
        validation,
        train_history,
        parameters,
        pd.Timestamp("2020-02-01"),
        300,
        start,
        "model_a",
    )
    combined = documents[documents["_split"].isin(("train", "val"))]
    combined_history = build_label_features(
        combined, combined, document_products, start, "model_b_causal_histories"
    )
    test_scores = fit_chain(
        documents,
        document_products,
        base_categorical,
        base_numeric,
        combined,
        test,
        combined_history,
        parameters,
        pd.Timestamp("2020-07-01"),
        300,
        start,
        "model_b",
    )
    validation_strata(validation, validation_scores)
    return validation_scores, test_scores, selection


def run(debug):
    warnings.filterwarnings("ignore")
    start = time.time()
    print(
        f"[pipeline] mode={'debug' if debug else 'full'} lightgbm={lgb.__version__} "
        f"threads={os.environ.get('OMP_NUM_THREADS', '1')}",
        flush=True,
    )
    context = load_task(upto_test_timestamp=False)
    elapsed(start, "load_task")
    shared = Path(os.environ.get("KAPSO_SHARED_CACHE_DIR", "./shared_cache"))
    record_database_notes(shared)
    record_feature_notes(shared, "PROPOSED", "Implementation entered causal pipeline validation.")
    documents, document_products, base_categorical, base_numeric = build_document_data(context, start)
    if debug:
        validation_scores, test_scores, selection = debug_run(
            documents, document_products, base_categorical, base_numeric, start
        )
    else:
        validation_scores, test_scores, selection = full_run(
            documents, document_products, base_categorical, base_numeric, start
        )
    if validation_scores.shape != (71474, 30) or test_scores.shape != (88942, 30):
        raise RuntimeError(
            f"prediction contract mismatch: validation={validation_scores.shape} test={test_scores.shape}"
        )
    save_predictions(validation_scores, test_scores)
    if not debug:
        record_feature_notes(
            shared,
            "TESTED-KEPT",
            f"Internal expanding-origin mean net-flip rate {selection['mean']:.9f}; selected {selection['key']} with fold diagnostics {selection['folds']}.",
        )
    elapsed(start, "complete")
