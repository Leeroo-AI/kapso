from __future__ import annotations

import json
import os
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from numba import njit
from sklearn.metrics import roc_auc_score


FEATURE_VERSION = "lane3_churn_renewal_graph_v2"
WINDOW_DAYS = 91
OOF_ORIGIN_INDICES = (20, 24, 30)


@njit(cache=True)
def summarize_products(pointer, days, products, customers, seed_day, product_target, product_count, fanout):
    result = np.empty((len(customers), 6), dtype=np.float32)
    result[:] = np.nan
    for row in range(len(customers)):
        customer = customers[row]
        left = pointer[customer]
        right = pointer[customer + 1]
        low = left
        high = right
        while low < high:
            middle = (low + high) // 2
            if days[middle] <= seed_day:
                low = middle + 1
            else:
                high = middle
        end = low
        low = left
        high = end
        boundary = seed_day - 91
        while low < high:
            middle = (low + high) // 2
            if days[middle] < boundary:
                low = middle + 1
            else:
                high = middle
        beginning = max(low, end - fanout)
        count = end - beginning
        if count <= 0:
            continue
        total = 0.0
        square = 0.0
        maximum = -1e9
        minimum = 1e9
        covered = 0.0
        last_two = 0.0
        last_two_count = 0
        for index in range(beginning, end):
            product = products[index]
            value = product_target[product]
            total += value
            square += value * value
            maximum = max(maximum, value)
            minimum = min(minimum, value)
            covered += product_count[product] > 0
            if index >= end - 2:
                last_two += value
                last_two_count += 1
        mean = total / count
        result[row, 0] = mean
        result[row, 1] = maximum
        result[row, 2] = minimum
        result[row, 3] = np.sqrt(max(0.0, square / count - mean * mean))
        result[row, 4] = covered / count
        result[row, 5] = last_two / last_two_count
    return result


@njit(cache=True)
def accumulate_recent_product_labels(pointer, days, products, customers, seed_day, values, product_count, product_sum, category_count, category_sum, brand_count, brand_sum, product_category, product_brand, fanout):
    for row in range(len(customers)):
        customer = customers[row]
        left = pointer[customer]
        right = pointer[customer + 1]
        low = left
        high = right
        while low < high:
            middle = (low + high) // 2
            if days[middle] <= seed_day:
                low = middle + 1
            else:
                high = middle
        end = low
        low = left
        high = end
        boundary = seed_day - 91
        while low < high:
            middle = (low + high) // 2
            if days[middle] < boundary:
                low = middle + 1
            else:
                high = middle
        beginning = max(low, end - fanout)
        for index in range(beginning, end):
            product = products[index]
            duplicate = False
            for previous in range(beginning, index):
                if products[previous] == product:
                    duplicate = True
                    break
            if duplicate:
                continue
            value = values[row]
            category = product_category[product]
            brand = product_brand[product]
            product_count[product] += 1.0
            product_sum[product] += value
            category_count[category] += 1.0
            category_sum[category] += value
            brand_count[brand] += 1.0
            brand_sum[brand] += value


@njit(cache=True)
def summarize_day_history(pointer, days, multiplicity, customers, seed_day):
    result = np.zeros((len(customers), 33), dtype=np.float32)
    windows = np.array([7, 14, 30, 60, 91, 182, 365, 730], dtype=np.int32)
    for row in range(len(customers)):
        customer = customers[row]
        left = pointer[customer]
        right = pointer[customer + 1]
        low = left
        high = right
        while low < high:
            middle = (low + high) // 2
            if days[middle] <= seed_day:
                low = middle + 1
            else:
                high = middle
        end = low
        result[row, 8] = end - left
        starts = np.empty(8, dtype=np.int64)
        for window_index in range(8):
            low = left
            high = end
            boundary = seed_day - windows[window_index]
            while low < high:
                middle = (low + high) // 2
                if days[middle] < boundary:
                    low = middle + 1
                else:
                    high = middle
            starts[window_index] = low
            result[row, window_index] = end - low
        recent_start = max(left, end - 8)
        gap_total = 0.0
        gap_square = 0.0
        gap_count = 0
        for index in range(end - 1, recent_start, -1):
            gap = days[index] - days[index - 1]
            output_index = 9 + gap_count
            result[row, output_index] = gap
            gap_total += gap
            gap_square += gap * gap
            gap_count += 1
        if gap_count:
            gap_mean = gap_total / gap_count
            gap_std = np.sqrt(max(0.0, gap_square / gap_count - gap_mean * gap_mean))
            result[row, 16] = gap_mean
            result[row, 17] = gap_std
            result[row, 18] = gap_std / max(gap_mean, 1.0)
        if end > left:
            result[row, 19] = multiplicity[end - 1]
        day91_start = starts[4]
        count91 = end - day91_start
        if count91:
            total = 0.0
            maximum = 0.0
            multiple = 0.0
            for index in range(day91_start, end):
                value = multiplicity[index]
                total += value
                maximum = max(maximum, value)
                multiple += value > 1
            result[row, 20] = total / count91
            result[row, 21] = maximum
            result[row, 22] = multiple / count91
        result[row, 23] = (result[row, 2] / 30.0) / max(result[row, 6] / 365.0, 1e-4)
        completed = end - left - 1
        if completed > 0:
            all_total = 0.0
            all_square = 0.0
            above_30 = 0.0
            above_91 = 0.0
            above_182 = 0.0
            maximum_gap = 0.0
            conditional_denominator = 0.0
            conditional_numerator = 0.0
            recency = seed_day - days[end - 1]
            for index in range(left + 1, end):
                gap = days[index] - days[index - 1]
                all_total += gap
                all_square += gap * gap
                maximum_gap = max(maximum_gap, gap)
                above_30 += gap > 30
                above_91 += gap > 91
                above_182 += gap > 182
                conditional_denominator += gap > recency
                conditional_numerator += gap > recency + 91
            all_mean = all_total / completed
            all_std = np.sqrt(max(0.0, all_square / completed - all_mean * all_mean))
            result[row, 24] = completed
            result[row, 25] = above_30 / completed
            result[row, 26] = above_91 / completed
            result[row, 27] = above_182 / completed
            result[row, 28] = maximum_gap
            result[row, 29] = conditional_numerator
            result[row, 30] = conditional_denominator
            result[row, 31] = (conditional_numerator + 1.0) / (conditional_denominator + 2.0)
            result[row, 32] = (all_std - all_mean) / max(all_std + all_mean, 1.0)
    return result


class ProductHistoryStore:
    def __init__(self, debug: bool = False):
        events, _ = ensure_event_cache(debug=debug)
        frame = connection().execute(
            f"""
            SELECT customer_id, CAST(epoch(review_time) / 86400 AS INTEGER) AS event_day, product_id
            FROM read_parquet('{events}')
            ORDER BY customer_id, review_time
            """
        ).fetch_df()
        self.customer = frame.pop("customer_id").to_numpy(np.int32)
        self.day = frame.pop("event_day").to_numpy(np.int32)
        self.product = frame.pop("product_id").to_numpy(np.int32)
        counts = np.bincount(self.customer, minlength=1_850_193)
        self.pointer = np.empty(1_850_194, dtype=np.int64)
        self.pointer[0] = 0
        np.cumsum(counts, out=self.pointer[1:])

    def summarize(self, customers: np.ndarray, seed_day: int, product_target: np.ndarray, product_count: np.ndarray, fanout: int = 8) -> np.ndarray:
        return summarize_products(self.pointer, self.day, self.product, customers, seed_day, product_target, product_count, fanout)

    def accumulate(self, customers: np.ndarray, seed_day: int, values: np.ndarray, product_count: np.ndarray, product_sum: np.ndarray, category_count: np.ndarray, category_sum: np.ndarray, brand_count: np.ndarray, brand_sum: np.ndarray, product_category: np.ndarray, product_brand: np.ndarray, fanout: int = 8) -> None:
        accumulate_recent_product_labels(self.pointer, self.day, self.product, customers, seed_day, values, product_count, product_sum, category_count, category_sum, brand_count, brand_sum, product_category, product_brand, fanout)


class DayHistoryStore:
    def __init__(self, debug: bool = False):
        events, _ = ensure_event_cache(debug=debug)
        frame = connection().execute(
            f"""
            SELECT customer_id, CAST(epoch(CAST(review_time AS DATE)) / 86400 AS INTEGER) AS event_day, count(*) AS multiplicity
            FROM read_parquet('{events}')
            GROUP BY customer_id, CAST(review_time AS DATE)
            ORDER BY customer_id, event_day
            """
        ).fetch_df()
        self.customer = frame.pop("customer_id").to_numpy(np.int32)
        self.day = frame.pop("event_day").to_numpy(np.int32)
        self.multiplicity = frame.pop("multiplicity").to_numpy(np.float32)
        counts = np.bincount(self.customer, minlength=1_850_193)
        self.pointer = np.empty(1_850_194, dtype=np.int64)
        self.pointer[0] = 0
        np.cumsum(counts, out=self.pointer[1:])

    def summarize(self, customers: np.ndarray, seed_day: int) -> np.ndarray:
        return summarize_day_history(self.pointer, self.day, self.multiplicity, customers, seed_day)


def cache_root() -> Path:
    root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / FEATURE_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def database_root() -> Path:
    return Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"]


def connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={int(os.environ.get('OMP_NUM_THREADS', '1'))}")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")
    return con


def ensure_event_cache(debug: bool = False) -> tuple[Path, float]:
    root = cache_root()
    path = root / ("events_debug.parquet" if debug else "events.parquet")
    if path.exists():
        return path, 0.0
    db = database_root() / "db"
    review = db / "review.parquet"
    product = db / "product.parquet"
    train = split_path("train")
    temporary = root / f"events_{os.getpid()}.parquet"
    debug_join = ""
    if debug:
        debug_join = f"JOIN (SELECT DISTINCT customer_id FROM (SELECT customer_id, row_number() OVER (PARTITION BY timestamp ORDER BY file_row_number) AS position FROM read_parquet('{train}', file_row_number=true) WHERE timestamp <= TIMESTAMP '2008-04-10') WHERE position <= 20000) d USING (customer_id)"
    start = time.time()
    con = connection()
    con.execute(
        f"""
        COPY (
            SELECT
                r.review_time,
                CAST(r.customer_id AS INTEGER) AS customer_id,
                CAST(r.product_id AS INTEGER) AS product_id,
                CAST(r.rating AS FLOAT) AS rating,
                CAST(r.verified AS UTINYINT) AS verified,
                CAST(length(r.review_text) AS INTEGER) AS text_length,
                CAST(length(r.summary) AS INTEGER) AS summary_length,
                CAST(p.price AS FLOAT) AS price,
                CAST(hash(p.category) % 65521 AS INTEGER) AS category_hash,
                CAST(coalesce(hash(p.category[2]) % 65521, 0) AS INTEGER) AS category_level2_hash,
                CAST(hash(p.brand) % 65521 AS INTEGER) AS brand_hash,
                CAST(length(p.title) AS SMALLINT) AS title_length,
                CAST(coalesce(length(p.description), 0) AS INTEGER) AS description_length,
                CAST(p.category IS NULL AS UTINYINT) AS category_missing,
                CAST(p.description IS NULL AS UTINYINT) AS description_missing,
                CAST(CASE WHEN length(r.review_text) > 0 THEN 1 + length(r.review_text) - length(replace(r.review_text, ' ', '')) ELSE 0 END AS INTEGER) AS review_word_count,
                CAST(length(coalesce(r.review_text, '')) - length(replace(coalesce(r.review_text, ''), '!', '')) AS SMALLINT) AS exclamation_count,
                CAST(length(coalesce(r.review_text, '')) - length(replace(coalesce(r.review_text, ''), '?', '')) AS SMALLINT) AS question_count,
                CAST(length(coalesce(r.summary, '')) > 0 AS UTINYINT) AS summary_present,
                CAST(regexp_matches(lower(coalesce(r.summary, '') || ' ' || coalesce(r.review_text, '')), '(love|great|excellent|amazing|wonderful|best|enjoy|recommend|favorite)') AS UTINYINT) AS positive_language,
                CAST(regexp_matches(lower(coalesce(r.summary, '') || ' ' || coalesce(r.review_text, '')), '(bad|poor|disappoint|boring|waste|terrible|awful|hate|worst|refund)') AS UTINYINT) AS negative_language
            FROM read_parquet('{review}') r
            {debug_join}
            JOIN read_parquet('{product}') p USING (product_id)
        ) TO '{temporary}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
        """
    )
    if path.exists():
        temporary.unlink(missing_ok=True)
    else:
        temporary.replace(path)
    return path, time.time() - start


def split_path(split: str) -> Path:
    return database_root() / "tasks" / os.environ["RELBENCH_TASK"] / f"{split}.parquet"


def read_origins(split: str) -> list[pd.Timestamp]:
    con = connection()
    frame = con.execute(
        f"SELECT DISTINCT timestamp FROM read_parquet('{split_path(split)}') ORDER BY timestamp"
    ).fetch_df()
    return [pd.Timestamp(value) for value in frame["timestamp"]]


def label_relation(split: str) -> str:
    train = split_path("train")
    if split == "test":
        val = split_path("val")
        return f"SELECT timestamp, customer_id, churn FROM read_parquet('{train}') UNION ALL SELECT timestamp, customer_id, churn FROM read_parquet('{val}')"
    return f"SELECT timestamp, customer_id, churn FROM read_parquet('{train}')"


def feature_partition_path(split: str, origin_index: int, debug: bool = False) -> Path:
    suffix = "_debug" if debug else ""
    return cache_root() / "features" / f"{split}_{origin_index:02d}{suffix}.parquet"


def build_feature_partition(
    split: str,
    origin: pd.Timestamp,
    origin_index: int,
    debug: bool = False,
) -> tuple[Path, float, int]:
    output = feature_partition_path(split, origin_index, debug=debug)
    if output.exists():
        rows = connection().execute(f"SELECT count(*) FROM read_parquet('{output}')").fetchone()[0]
        return output, 0.0, int(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    events, _ = ensure_event_cache(debug=debug)
    customer = database_root() / "db" / "customer.parquet"
    seeds = split_path(split)
    labels = label_relation(split)
    timestamp = origin.strftime("%Y-%m-%d %H:%M:%S")
    temporary = output.with_name(f"{output.stem}_{os.getpid()}.parquet")
    seed_limit = "LIMIT 20000" if debug else ""
    start = time.time()
    con = connection()
    con.execute(
        f"""
        COPY (
        WITH
        seed_all AS (
            SELECT file_row_number AS row_id, timestamp, customer_id
            FROM read_parquet('{seeds}', file_row_number=true)
        ),
        seed AS (
            SELECT * FROM seed_all WHERE timestamp = TIMESTAMP '{timestamp}' ORDER BY row_id {seed_limit}
        ),
        customer_stats AS (
            SELECT
                customer_id,
                count(*) AS n_all,
                count(*) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '7 days') AS n_7,
                count(*) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '14 days') AS n_14,
                count(*) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '30 days') AS n_30,
                count(*) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '45 days') AS n_45,
                count(*) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '60 days') AS n_60,
                count(*) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '91 days') AS n_91,
                count(*) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '182 days') AS n_182,
                count(*) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '365 days') AS n_365,
                count(*) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '730 days') AS n_730,
                approx_count_distinct(product_id) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '91 days') AS products_91,
                approx_count_distinct(product_id) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '365 days') AS products_365,
                count(DISTINCT CAST(review_time AS DATE)) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '91 days') AS days_active_91,
                min(review_time) AS first_time,
                max(review_time) AS last_time,
                max_by(review_time, review_time, 8) AS recent_times,
                avg(rating) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '91 days') AS rating_91,
                stddev_pop(rating) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '365 days') AS rating_std_365,
                avg(rating) AS rating_all,
                avg(verified) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '91 days') AS verified_91,
                avg(verified) AS verified_all,
                avg(text_length) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '91 days') AS text_length_91,
                avg(text_length) AS text_length_all,
                avg(summary_length) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '91 days') AS summary_length_91,
                avg(price) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '91 days') AS price_91,
                stddev_pop(price) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '365 days') AS price_std_365,
                approx_count_distinct(category_hash) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '365 days') AS categories_365,
                approx_count_distinct(category_level2_hash) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '365 days') AS category_level2_365,
                avg(review_word_count) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '91 days') AS review_words_91,
                avg(exclamation_count) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '91 days') AS exclamation_91,
                avg(question_count) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '91 days') AS question_91,
                avg(summary_present) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '91 days') AS summary_present_91,
                avg(positive_language) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '91 days') AS positive_language_91,
                avg(negative_language) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '91 days') AS negative_language_91,
                avg(positive_language) AS positive_language_all,
                avg(negative_language) AS negative_language_all,
                max_by(product_id, review_time) AS last_product_id,
                max_by(rating, review_time) AS last_rating,
                max_by(verified, review_time) AS last_verified,
                max_by(text_length, review_time) AS last_text_length,
                max_by(summary_length, review_time) AS last_summary_length,
                max_by(price, review_time) AS last_price,
                max_by(category_hash, review_time) AS last_category_hash,
                max_by(category_level2_hash, review_time) AS last_category_level2_hash,
                max_by(brand_hash, review_time) AS last_brand_hash,
                max_by(title_length, review_time) AS last_title_length,
                max_by(description_length, review_time) AS last_description_length,
                max_by(category_missing, review_time) AS last_category_missing,
                max_by(description_missing, review_time) AS last_description_missing,
                max_by(review_word_count, review_time) AS last_review_word_count,
                max_by(exclamation_count, review_time) AS last_exclamation_count,
                max_by(question_count, review_time) AS last_question_count,
                max_by(summary_present, review_time) AS last_summary_present,
                max_by(positive_language, review_time) AS last_positive_language,
                max_by(negative_language, review_time) AS last_negative_language,
                year(min(review_time)) AS first_review_year
            FROM read_parquet('{events}')
            WHERE review_time <= TIMESTAMP '{timestamp}'
            GROUP BY customer_id
        ),
        needed_product AS (
            SELECT DISTINCT cs.last_product_id AS product_id
            FROM seed s JOIN customer_stats cs USING (customer_id)
        ),
        product_stats AS (
            SELECT
                e.product_id,
                count(*) AS product_n_all,
                count(*) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '91 days') AS product_n_91,
                count(*) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '365 days') AS product_n_365,
                approx_count_distinct(customer_id) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '365 days') AS product_customers_365,
                avg(rating) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '365 days') AS product_rating_365,
                avg(verified) FILTER (WHERE review_time > TIMESTAMP '{timestamp}' - INTERVAL '365 days') AS product_verified_365
            FROM read_parquet('{events}') e
            JOIN needed_product p USING (product_id)
            WHERE review_time <= TIMESTAMP '{timestamp}'
            GROUP BY e.product_id
        ),
        eligible_labels AS (
            SELECT l.*
            FROM ({labels}) l
            WHERE l.timestamp + INTERVAL '91 days' <= TIMESTAMP '{timestamp}'
        ),
        global_label AS (
            SELECT count(*) AS global_n, avg(churn) AS global_mean FROM eligible_labels
        ),
        customer_label AS (
            SELECT
                customer_id,
                count(*) AS history_label_n,
                sum(churn) AS history_label_sum,
                avg(churn) AS history_label_mean,
                arg_max(churn, timestamp) AS previous_churn,
                sum(1 - churn) AS history_return_n
            FROM eligible_labels GROUP BY customer_id
        ),
        cohort_label AS (
            SELECT
                cs.first_review_year,
                count(*) AS cohort_n,
                avg(el.churn) AS cohort_mean
            FROM eligible_labels el
            JOIN customer_stats cs USING (customer_id)
            GROUP BY cs.first_review_year
        ),
        raw AS (
            SELECT
                s.row_id,
                s.customer_id,
                {origin_index} AS origin_index,
                cs.* EXCLUDE (customer_id, recent_times),
                date_diff('second', cs.last_time, s.timestamp) / 86400.0 AS recency_days,
                date_diff('second', cs.first_time, s.timestamp) / 86400.0 AS tenure_days,
                date_diff('second', list_extract(cs.recent_times, 2), list_extract(cs.recent_times, 1)) / 86400.0 AS gap_1,
                date_diff('second', list_extract(cs.recent_times, 3), list_extract(cs.recent_times, 2)) / 86400.0 AS gap_2,
                date_diff('second', list_extract(cs.recent_times, 4), list_extract(cs.recent_times, 3)) / 86400.0 AS gap_3,
                date_diff('second', list_extract(cs.recent_times, 5), list_extract(cs.recent_times, 4)) / 86400.0 AS gap_4,
                date_diff('second', list_extract(cs.recent_times, 6), list_extract(cs.recent_times, 5)) / 86400.0 AS gap_5,
                date_diff('second', list_extract(cs.recent_times, 7), list_extract(cs.recent_times, 6)) / 86400.0 AS gap_6,
                date_diff('second', list_extract(cs.recent_times, 8), list_extract(cs.recent_times, 7)) / 86400.0 AS gap_7,
                ps.* EXCLUDE (product_id),
                length(c.customer_name) AS customer_name_length,
                coalesce(cl.history_label_n, 0) AS history_label_n,
                coalesce(cl.history_label_sum, 0) AS history_label_sum,
                cl.history_label_mean,
                cl.previous_churn,
                coalesce(cl.history_return_n, 0) AS history_return_n,
                gl.global_mean,
                coalesce((col.cohort_n * col.cohort_mean + 64 * gl.global_mean) / (col.cohort_n + 64), gl.global_mean) AS cohort_target,
                coalesce((cl.history_label_sum + 8 * coalesce((col.cohort_n * col.cohort_mean + 64 * gl.global_mean) / (col.cohort_n + 64), gl.global_mean)) / (cl.history_label_n + 8), gl.global_mean) AS customer_target,
                sin(2 * pi() * month(s.timestamp) / 12.0) AS season_sin,
                cos(2 * pi() * month(s.timestamp) / 12.0) AS season_cos
            FROM seed s
            JOIN customer_stats cs USING (customer_id)
            LEFT JOIN product_stats ps ON ps.product_id = cs.last_product_id
            LEFT JOIN read_parquet('{customer}') c USING (customer_id)
            LEFT JOIN customer_label cl USING (customer_id)
            LEFT JOIN cohort_label col USING (first_review_year)
            CROSS JOIN global_label gl
        )
        SELECT
            * EXCLUDE (first_time, last_time),
            n_7 / greatest(n_91, 1) AS share_7_91,
            n_14 / greatest(n_91, 1) AS share_14_91,
            n_30 / greatest(n_91, 1) AS share_30_91,
            n_60 / greatest(n_182, 1) AS share_60_182,
            n_91 / greatest(n_365, 1) AS share_91_365,
            n_182 / greatest(n_730, 1) AS share_182_730,
            (n_30 / 30.0) / greatest(n_365 / 365.0, 1e-4) AS fast_slow_30_365,
            (n_91 / 91.0) / greatest(n_730 / 730.0, 1e-4) AS fast_slow_91_730,
            (n_30 + 0.25 * n_365) / 121.25 AS blended_daily_rate,
            products_91 / greatest(n_91, 1) AS product_diversity_91,
            days_active_91 / greatest(n_91, 1) AS active_day_share_91,
            history_return_n / greatest(history_label_n, 1) AS historical_return_share,
            percent_rank() OVER (ORDER BY recency_days) AS recency_percentile,
            percent_rank() OVER (ORDER BY n_91) AS activity_91_percentile,
            percent_rank() OVER (ORDER BY n_365) AS activity_365_percentile,
            percent_rank() OVER (ORDER BY tenure_days) AS tenure_percentile,
            percent_rank() OVER (ORDER BY product_n_365) AS product_popularity_percentile
        FROM raw
        ORDER BY row_id
        ) TO '{temporary}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
        """
    )
    if output.exists():
        temporary.unlink(missing_ok=True)
    else:
        temporary.replace(output)
    rows = con.execute(f"SELECT count(*) FROM read_parquet('{output}')").fetchone()[0]
    return output, time.time() - start, int(rows)


def ensure_features(split: str, debug: bool = False) -> tuple[list[Path], list[dict]]:
    origins = read_origins(split)
    paths = []
    timings = []
    selected = origins[:2] if debug and split == "train" else origins[:1] if debug else origins
    for index, origin in enumerate(selected):
        path, elapsed, rows = build_feature_partition(split, origin, index, debug=debug)
        paths.append(path)
        timings.append({"split": split, "origin": str(origin), "rows": rows, "seconds": elapsed})
    return paths, timings


def load_feature_frame(paths: list[Path]) -> pd.DataFrame:
    relation = ",".join(f"'{path}'" for path in paths)
    frame = connection().execute(f"SELECT * FROM read_parquet([{relation}]) ORDER BY row_id").fetch_df()
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "row_id", "customer_id", "origin_index", "last_product_id", "last_category_hash",
        "last_category_level2_hash", "last_brand_hash", "global_mean", "completed_day_gaps",
        "day_gap_share_gt30", "day_gap_share_gt91", "day_gap_share_gt182",
        "maximum_completed_day_gap", "conditional_gap_numerator", "conditional_gap_denominator",
        "empirical_conditional_survival", "day_gap_burstiness", "smoothed_conditional_survival",
    }
    return [column for column in frame.columns if column not in excluded]


def labels_frame() -> pd.DataFrame:
    return connection().execute(
        f"SELECT file_row_number AS row_id, timestamp, customer_id, churn FROM read_parquet('{split_path('train')}', file_row_number=true) ORDER BY row_id"
    ).fetch_df()


def validation_labels() -> np.ndarray:
    return connection().execute(f"SELECT churn FROM read_parquet('{split_path('val')}')").fetchnumpy()["churn"].astype(np.int8)


def add_causal_relational_targets(
    train_frame: pd.DataFrame,
    train_y: np.ndarray,
    val_frame: pd.DataFrame | None = None,
    val_y: np.ndarray | None = None,
    test_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    train_frame = train_frame.copy()
    val_frame = None if val_frame is None else val_frame.copy()
    test_frame = None if test_frame is None else test_frame.copy()
    product_count = np.zeros(506012, dtype=np.float64)
    product_sum = np.zeros(506012, dtype=np.float64)
    product_last = np.full(506012, np.nan, dtype=np.float32)
    category_count = np.zeros(65521, dtype=np.float64)
    category_sum = np.zeros(65521, dtype=np.float64)
    category_level2_count = np.zeros(65521, dtype=np.float64)
    category_level2_sum = np.zeros(65521, dtype=np.float64)
    brand_count = np.zeros(65521, dtype=np.float64)
    brand_sum = np.zeros(65521, dtype=np.float64)
    neighbor_product_count = np.zeros(506012, dtype=np.float64)
    neighbor_product_sum = np.zeros(506012, dtype=np.float64)
    neighbor_category_count = np.zeros(65521, dtype=np.float64)
    neighbor_category_sum = np.zeros(65521, dtype=np.float64)
    neighbor_brand_count = np.zeros(65521, dtype=np.float64)
    neighbor_brand_sum = np.zeros(65521, dtype=np.float64)
    customer_count = np.zeros(1_850_193, dtype=np.float32)
    customer_last = np.full(1_850_193, np.nan, dtype=np.float32)
    customer_second_last = np.full(1_850_193, np.nan, dtype=np.float32)
    customer_last_origin = np.full(1_850_193, -1, dtype=np.int16)
    customer_return_streak = np.zeros(1_850_193, dtype=np.float32)
    customer_churn_streak = np.zeros(1_850_193, dtype=np.float32)
    global_count = 0.0
    global_sum = 0.0
    train_origins = read_origins("train")
    train_origin_index = train_frame["origin_index"].to_numpy(np.int16)
    history_store = ProductHistoryStore(debug=len(train_frame) < 100000)
    day_store = DayHistoryStore(debug=len(train_frame) < 100000)
    product_meta = connection().execute(
        f"""
        SELECT product_id, CAST(coalesce(hash(category) % 65521, 0) AS INTEGER) AS category_hash, CAST(coalesce(hash(category[2]) % 65521, 0) AS INTEGER) AS category_level2_hash, CAST(coalesce(hash(brand) % 65521, 0) AS INTEGER) AS brand_hash
        FROM read_parquet('{database_root() / 'db' / 'product.parquet'}') ORDER BY product_id
        """
    ).fetch_df()
    product_category = product_meta["category_hash"].to_numpy(np.int32)
    product_category_level2 = product_meta["category_level2_hash"].to_numpy(np.int32)
    product_brand = product_meta["brand_hash"].to_numpy(np.int32)

    def add_rows(frame: pd.DataFrame, labels: np.ndarray, indices: np.ndarray, label_origin: int, label_timestamp: pd.Timestamp) -> None:
        nonlocal global_count, global_sum
        products = frame["last_product_id"].to_numpy(np.int32)[indices]
        categories = frame["last_category_hash"].fillna(0).to_numpy(np.int32)[indices]
        categories_level2 = frame["last_category_level2_hash"].fillna(0).to_numpy(np.int32)[indices]
        brands = frame["last_brand_hash"].fillna(0).to_numpy(np.int32)[indices]
        customers = frame["customer_id"].to_numpy(np.int32)[indices]
        values = labels[indices].astype(np.float64)
        np.add.at(product_count, products, 1.0)
        np.add.at(product_sum, products, values)
        np.add.at(category_count, categories, 1.0)
        np.add.at(category_sum, categories, values)
        np.add.at(category_level2_count, categories_level2, 1.0)
        np.add.at(category_level2_sum, categories_level2, values)
        np.add.at(brand_count, brands, 1.0)
        np.add.at(brand_sum, brands, values)
        product_last[products] = values.astype(np.float32)
        customer_second_last[customers] = customer_last[customers]
        customer_last[customers] = values.astype(np.float32)
        customer_last_origin[customers] = label_origin
        customer_count[customers] += 1.0
        returned = values == 0
        customer_return_streak[customers] = np.where(returned, customer_return_streak[customers] + 1.0, 0.0)
        customer_churn_streak[customers] = np.where(returned, 0.0, customer_churn_streak[customers] + 1.0)
        history_store.accumulate(
            customers,
            int(label_timestamp.timestamp() // 86400),
            values,
            neighbor_product_count,
            neighbor_product_sum,
            neighbor_category_count,
            neighbor_category_sum,
            neighbor_brand_count,
            neighbor_brand_sum,
            product_category_level2,
            product_brand,
        )
        global_count += float(len(indices))
        global_sum += float(values.sum())

    def assign(frame: pd.DataFrame, current_origin: int, timestamp: pd.Timestamp) -> None:
        products = frame["last_product_id"].to_numpy(np.int32)
        categories = frame["last_category_hash"].fillna(0).to_numpy(np.int32)
        categories_level2 = frame["last_category_level2_hash"].fillna(0).to_numpy(np.int32)
        brands = frame["last_brand_hash"].fillna(0).to_numpy(np.int32)
        customers = frame["customer_id"].to_numpy(np.int32)
        global_mean = global_sum / global_count if global_count else 0.5
        category_mean = (category_sum[categories] + 64.0 * global_mean) / (category_count[categories] + 64.0)
        category_level2_mean = (category_level2_sum[categories_level2] + 64.0 * global_mean) / (category_level2_count[categories_level2] + 64.0)
        brand_mean = (brand_sum[brands] + 64.0 * global_mean) / (brand_count[brands] + 64.0)
        group_mean = 0.5 * (category_mean + brand_mean)
        frame["category_target"] = category_mean.astype(np.float32)
        frame["category_level2_target"] = category_level2_mean.astype(np.float32)
        frame["brand_target"] = brand_mean.astype(np.float32)
        frame["product_target"] = ((product_sum[products] + 12.0 * group_mean) / (product_count[products] + 12.0)).astype(np.float32)
        frame["product_target_level2"] = ((product_sum[products] + 6.0 * (category_level2_mean + brand_mean)) / (product_count[products] + 12.0)).astype(np.float32)
        frame["product_label_n"] = product_count[products].astype(np.float32)
        frame["category_label_n"] = category_count[categories].astype(np.float32)
        frame["category_level2_label_n"] = category_level2_count[categories_level2].astype(np.float32)
        frame["brand_label_n"] = brand_count[brands].astype(np.float32)
        frame["product_previous_churn"] = product_last[products]
        frame["customer_origin_gap"] = np.where(customer_last_origin[customers] >= 0, current_origin - customer_last_origin[customers], current_origin + 1).astype(np.float32)
        frame["customer_participation_rate"] = (customer_count[customers] / max(current_origin, 1)).astype(np.float32)
        frame["customer_second_previous_churn"] = customer_second_last[customers]
        recent = np.stack([customer_last[customers], customer_second_last[customers]], axis=1)
        recent_count = np.isfinite(recent).sum(axis=1)
        frame["customer_recent_two_churn"] = np.divide(np.nansum(recent, axis=1), recent_count, out=np.full(len(frame), np.nan, dtype=np.float32), where=recent_count > 0).astype(np.float32)
        frame["customer_return_streak"] = customer_return_streak[customers]
        frame["customer_churn_streak"] = customer_churn_streak[customers]
        category_all = (category_sum[product_category] + 64.0 * global_mean) / (category_count[product_category] + 64.0)
        category_level2_all = (category_level2_sum[product_category_level2] + 64.0 * global_mean) / (category_level2_count[product_category_level2] + 64.0)
        brand_all = (brand_sum[product_brand] + 64.0 * global_mean) / (brand_count[product_brand] + 64.0)
        product_target_all = (product_sum + 3.0 * category_all + 6.0 * category_level2_all + 3.0 * brand_all) / (product_count + 12.0)
        recent = history_store.summarize(customers, int(timestamp.timestamp() // 86400), product_target_all, product_count)
        frame["recent_product_target_mean"] = recent[:, 0]
        frame["recent_product_target_max"] = recent[:, 1]
        frame["recent_product_target_min"] = recent[:, 2]
        frame["recent_product_target_std"] = recent[:, 3]
        frame["recent_product_target_coverage"] = recent[:, 4]
        frame["recent_product_target_last_two"] = recent[:, 5]
        neighbor_category_all = (neighbor_category_sum[product_category_level2] + 64.0 * global_mean) / (neighbor_category_count[product_category_level2] + 64.0)
        neighbor_brand_all = (neighbor_brand_sum[product_brand] + 64.0 * global_mean) / (neighbor_brand_count[product_brand] + 64.0)
        neighbor_target_all = (neighbor_product_sum + 6.0 * (neighbor_category_all + neighbor_brand_all)) / (neighbor_product_count + 12.0)
        neighbor_recent = history_store.summarize(customers, int(timestamp.timestamp() // 86400), neighbor_target_all, neighbor_product_count)
        frame["neighbor_product_target"] = neighbor_target_all[products].astype(np.float32)
        frame["neighbor_product_label_n"] = neighbor_product_count[products].astype(np.float32)
        frame["recent_neighbor_target_mean"] = neighbor_recent[:, 0]
        frame["recent_neighbor_target_max"] = neighbor_recent[:, 1]
        frame["recent_neighbor_target_min"] = neighbor_recent[:, 2]
        frame["recent_neighbor_target_std"] = neighbor_recent[:, 3]
        frame["recent_neighbor_target_coverage"] = neighbor_recent[:, 4]
        frame["recent_neighbor_target_last_two"] = neighbor_recent[:, 5]
        day_history = day_store.summarize(customers, int(timestamp.timestamp() // 86400))
        day_columns = [
            "active_days_7", "active_days_14", "active_days_30", "active_days_60",
            "active_days_91_store", "active_days_182", "active_days_365", "active_days_730",
            "active_days_all", "day_gap_1", "day_gap_2", "day_gap_3", "day_gap_4",
            "day_gap_5", "day_gap_6", "day_gap_7", "day_gap_mean_7", "day_gap_std_7",
            "day_gap_cv_7", "last_day_multiplicity", "mean_day_multiplicity_91",
            "max_day_multiplicity_91", "multi_review_day_share_91", "day_fast_slow_30_365",
            "completed_day_gaps", "day_gap_share_gt30", "day_gap_share_gt91",
            "day_gap_share_gt182", "maximum_completed_day_gap", "conditional_gap_numerator",
            "conditional_gap_denominator", "empirical_conditional_survival", "day_gap_burstiness",
        ]
        for column_index, column in enumerate(day_columns):
            frame[column] = day_history[:, column_index]

    next_eligible = 0
    for current, timestamp in enumerate(train_origins):
        while next_eligible < current and train_origins[next_eligible] + pd.Timedelta(days=WINDOW_DAYS) <= timestamp:
            indices = np.flatnonzero(train_origin_index == next_eligible)
            add_rows(train_frame, train_y, indices, next_eligible, train_origins[next_eligible])
            next_eligible += 1
        current_indices = np.flatnonzero(train_origin_index == current)
        current_frame = train_frame.iloc[current_indices].copy()
        assign(current_frame, current, timestamp)
        for column in ("category_target", "category_level2_target", "brand_target", "product_target", "product_target_level2", "product_label_n", "category_label_n", "category_level2_label_n", "brand_label_n", "product_previous_churn", "customer_origin_gap", "customer_participation_rate", "customer_second_previous_churn", "customer_recent_two_churn", "customer_return_streak", "customer_churn_streak", "recent_product_target_mean", "recent_product_target_max", "recent_product_target_min", "recent_product_target_std", "recent_product_target_coverage", "recent_product_target_last_two", "neighbor_product_target", "neighbor_product_label_n", "recent_neighbor_target_mean", "recent_neighbor_target_max", "recent_neighbor_target_min", "recent_neighbor_target_std", "recent_neighbor_target_coverage", "recent_neighbor_target_last_two", "active_days_7", "active_days_14", "active_days_30", "active_days_60", "active_days_91_store", "active_days_182", "active_days_365", "active_days_730", "active_days_all", "day_gap_1", "day_gap_2", "day_gap_3", "day_gap_4", "day_gap_5", "day_gap_6", "day_gap_7", "day_gap_mean_7", "day_gap_std_7", "day_gap_cv_7", "last_day_multiplicity", "mean_day_multiplicity_91", "max_day_multiplicity_91", "multi_review_day_share_91", "day_fast_slow_30_365", "completed_day_gaps", "day_gap_share_gt30", "day_gap_share_gt91", "day_gap_share_gt182", "maximum_completed_day_gap", "conditional_gap_numerator", "conditional_gap_denominator", "empirical_conditional_survival", "day_gap_burstiness"):
            train_frame.loc[train_frame.index[current_indices], column] = current_frame[column].to_numpy()
    if val_frame is not None:
        val_timestamp = pd.Timestamp("2015-10-01 00:00:00")
        while next_eligible < len(train_origins) and train_origins[next_eligible] + pd.Timedelta(days=WINDOW_DAYS) <= val_timestamp:
            indices = np.flatnonzero(train_origin_index == next_eligible)
            add_rows(train_frame, train_y, indices, next_eligible, train_origins[next_eligible])
            next_eligible += 1
        assign(val_frame, len(train_origins), val_timestamp)
    if test_frame is not None:
        test_timestamp = pd.Timestamp("2016-01-01 00:00:00")
        while next_eligible < len(train_origins) and train_origins[next_eligible] + pd.Timedelta(days=WINDOW_DAYS) <= test_timestamp:
            indices = np.flatnonzero(train_origin_index == next_eligible)
            add_rows(train_frame, train_y, indices, next_eligible, train_origins[next_eligible])
            next_eligible += 1
        if val_frame is not None and val_y is not None and pd.Timestamp("2015-10-01") + pd.Timedelta(days=WINDOW_DAYS) <= test_timestamp:
            add_rows(val_frame, val_y, np.arange(len(val_frame)), len(train_origins), pd.Timestamp("2015-10-01 00:00:00"))
        assign(test_frame, len(train_origins) + 1, test_timestamp)
    return train_frame, val_frame, test_frame


def add_renewal_transforms(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    gaps = frame[[f"gap_{index}" for index in range(1, 8)]].astype(np.float32)
    frame["gap_mean_3"] = gaps.iloc[:, :3].mean(axis=1)
    frame["gap_mean_7"] = gaps.mean(axis=1)
    frame["gap_std_7"] = gaps.std(axis=1)
    frame["gap_cv_7"] = frame["gap_std_7"] / frame["gap_mean_7"].clip(lower=1.0)
    frame["recency_gap_ratio"] = frame["recency_days"] / frame["gap_mean_7"].clip(lower=1.0)
    frame["recency_minus_gap"] = frame["recency_days"] - frame["gap_mean_7"]
    frame["last_gap_ratio"] = frame["gap_1"] / frame["gap_mean_7"].clip(lower=1.0)
    exposure_365 = frame["tenure_days"].clip(lower=1.0, upper=365.0)
    exposure_730 = frame["tenure_days"].clip(lower=1.0, upper=730.0)
    rate_91 = frame["n_91"] / 91.0
    rate_365 = frame["n_365"] / exposure_365
    rate_730 = frame["n_730"] / exposure_730
    rate_all = frame["n_all"] / frame["tenure_days"].clip(lower=1.0)
    frame["poisson_churn_91"] = np.exp(-91.0 * rate_91.clip(upper=20.0 / 91.0))
    frame["poisson_churn_365"] = np.exp(-91.0 * rate_365.clip(upper=20.0 / 91.0))
    frame["poisson_churn_730"] = np.exp(-91.0 * rate_730.clip(upper=20.0 / 91.0))
    frame["poisson_churn_all"] = np.exp(-91.0 * rate_all.clip(upper=20.0 / 91.0))
    frame["smoothed_conditional_survival"] = (frame["conditional_gap_numerator"] + 4.0 * frame["poisson_churn_365"]) / (frame["conditional_gap_denominator"] + 4.0)
    frame["gamma_poisson_churn_365"] = np.power((exposure_365 + 91.0) / (exposure_365 + 182.0), frame["n_365"] + 1.0)
    frame["activity_acceleration"] = rate_91 / rate_365.clip(lower=1e-4)
    frame["rating_delta"] = frame["rating_91"] - frame["rating_all"]
    frame["verified_delta"] = frame["verified_91"] - frame["verified_all"]
    frame["text_length_delta"] = frame["text_length_91"] - frame["text_length_all"]
    frame["product_recent_share"] = frame["product_n_91"] / frame["product_n_365"].clip(lower=1.0)
    frame["product_audience_repeat"] = frame["product_n_365"] / frame["product_customers_365"].clip(lower=1.0)
    frame["positive_language_delta"] = frame["positive_language_91"] - frame["positive_language_all"]
    frame["negative_language_delta"] = frame["negative_language_91"] - frame["negative_language_all"]
    return frame


def add_customer_name_features(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    customer = database_root() / "db" / "customer.parquet"
    names = connection().execute(
        f"""
        SELECT
            customer_id,
            count(*) OVER (PARTITION BY customer_name) AS name_frequency,
            CASE WHEN length(customer_name) > 0 THEN 1 + length(customer_name) - length(replace(customer_name, ' ', '')) ELSE 0 END AS name_words,
            CAST(lower(customer_name) = 'amazon customer' AS UTINYINT) AS name_amazon_customer,
            CAST(lower(customer_name) = 'kindle customer' AS UTINYINT) AS name_kindle_customer,
            CAST(lower(customer_name) LIKE '%avid reader%' AS UTINYINT) AS name_avid_reader,
            CAST(regexp_matches(coalesce(customer_name, ''), '[0-9]') AS UTINYINT) AS name_has_digit
        FROM read_parquet('{customer}') ORDER BY customer_id
        """
    ).fetch_df()
    columns = [column for column in names.columns if column != "customer_id"]
    arrays = {column: names[column].to_numpy() for column in columns}
    result = []
    for frame in frames:
        frame = frame.copy()
        customers = frame["customer_id"].to_numpy(np.int32)
        for column in columns:
            frame[column] = arrays[column][customers]
        result.append(frame)
    return tuple(result)


def matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].replace([np.inf, -np.inf], np.nan).astype(np.float32).to_numpy()


def train_model(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None, rounds: int = 520) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=rounds,
        learning_rate=0.04,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=500,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.84,
        reg_alpha=0.2,
        reg_lambda=3.0,
        max_bin=127,
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")),
        random_state=1337,
        verbosity=-1,
    )
    model.fit(x, y, sample_weight=weights, callbacks=[lgb.log_evaluation(0)])
    return model


def expanding_oof(frame: pd.DataFrame, labels: pd.DataFrame, columns: list[str], debug: bool = False) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    merged = frame[["row_id", "origin_index"]].merge(labels[["row_id", "churn"]], on="row_id", how="left", validate="one_to_one")
    y = merged["churn"].to_numpy(np.int8)
    x = matrix(frame, columns)
    origin_values = frame["origin_index"].to_numpy()
    sample_weights = temporal_weights(origin_values)
    predictions = np.full(len(frame), np.nan, dtype=np.float32)
    fold_rows = []
    folds = (min(1, int(frame["origin_index"].max())),) if debug else OOF_ORIGIN_INDICES
    for fold in folds:
        train_mask = origin_values < fold
        hold_mask = origin_values == fold
        if train_mask.sum() == 0 or hold_mask.sum() == 0:
            continue
        rounds = 30 if debug else 520
        model = train_model(x[train_mask], y[train_mask], sample_weights[train_mask], rounds=rounds)
        pred = model.predict_proba(x[hold_mask])[:, 1]
        predictions[hold_mask] = pred
        auc = roc_auc_score(y[hold_mask], pred)
        fold_rows.append({"fold": int(fold), "train_n": int(train_mask.sum()), "hold_n": int(hold_mask.sum()), "auc": float(auc)})
        print(f"[lane3] renewal_fold={fold} train={int(train_mask.sum())} hold={int(hold_mask.sum())} auc={auc:.6f}", flush=True)
    mask = np.isfinite(predictions)
    return predictions, mask, fold_rows


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return (ranks + 0.5) / len(values)


def choose_rank_blend(y: np.ndarray, origins: np.ndarray, renewal: np.ndarray, graph: np.ndarray, mask: np.ndarray) -> tuple[float, list[dict]]:
    results = []
    for weight in (0.0, 0.2, 0.4, 0.6):
        fold_scores = []
        for fold in OOF_ORIGIN_INDICES:
            fold_mask = mask & (origins == fold)
            if not np.any(fold_mask):
                continue
            blended = (1.0 - weight) * rank_values(renewal[fold_mask]) + weight * rank_values(graph[fold_mask])
            fold_scores.append(float(roc_auc_score(y[fold_mask], blended)))
        if fold_scores:
            results.append({"weight": weight, "median_auc": float(np.median(fold_scores)), "worst_auc": float(np.min(fold_scores)), "fold_auc": fold_scores})
    results.sort(key=lambda row: (row["median_auc"], row["worst_auc"]), reverse=True)
    best = results[0] if results else {"weight": 0.0, "median_auc": 0.0, "worst_auc": 0.0, "fold_auc": []}
    baseline = next((row for row in results if row["weight"] == 0.0), best)
    admitted = best["weight"] > 0 and best["median_auc"] >= baseline["median_auc"] + 0.002
    return (float(best["weight"]) if admitted else 0.0), results


def temporal_weights(origin_indices: np.ndarray) -> np.ndarray:
    latest = float(np.max(origin_indices))
    return np.clip(np.exp(-0.025 * (latest - origin_indices)), 0.55, 1.0).astype(np.float32)


def write_diagnostics(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
