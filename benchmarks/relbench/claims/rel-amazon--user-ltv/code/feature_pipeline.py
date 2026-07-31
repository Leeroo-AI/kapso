from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


FEATURE_VERSION = "rel_amazon_user_ltv_compound_lane3_v1"
WINDOWS = (30, 91, 182, 365)


def data_paths() -> dict[str, Path]:
    root = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"]
    task = root / "tasks" / os.environ["RELBENCH_TASK"]
    return {
        "review": root / "db" / "review.parquet",
        "product": root / "db" / "product.parquet",
        "customer": root / "db" / "customer.parquet",
        "train": task / "train.parquet",
        "val": task / "val.parquet",
        "test": task / "test.parquet",
    }


def cache_root() -> Path:
    root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / FEATURE_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={int(os.environ.get('OMP_NUM_THREADS', '8'))}")
    con.execute("SET memory_limit='90GB'")
    con.execute(f"SET temp_directory='{cache_root() / 'duckdb_tmp'}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def atomic_copy(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.parquet")
    con.execute(f"COPY ({query}) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)")
    os.replace(tmp, path)


def register_artifact(name: str, path: Path, description: str, rebuild_hint: str) -> None:
    import fcntl

    shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    registry = shared / "artifacts.json"
    lock_path = shared / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            entries = json.loads(registry.read_text()) if registry.exists() else []
            relative = str(path.relative_to(shared))
            if not any(x.get("name") == name and x.get("path") == relative for x in entries):
                entries.append(
                    {
                        "name": name,
                        "path": relative,
                        "description": description,
                        "content_key": FEATURE_VERSION,
                        "rebuild_hint": rebuild_hint,
                    }
                )
                tmp = registry.with_name(f"artifacts.{os.getpid()}.tmp.json")
                tmp.write_text(json.dumps(entries, indent=2))
                os.replace(tmp, registry)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def ensure_event_state() -> Path:
    output = cache_root() / "event_state.parquet"
    if output.exists():
        return output
    paths = data_paths()
    con = connection()
    query = f"""
    WITH raw AS (
        SELECT
            r.review_time,
            r.customer_id,
            r.product_id,
            p.price,
            r.rating::DOUBLE AS rating,
            r.verified::INT AS verified,
            coalesce(length(r.review_text), 0)::DOUBLE AS text_len,
            coalesce(length(r.summary), 0)::DOUBLE AS summary_len,
            length(p.title)::DOUBLE AS title_len,
            coalesce(length(p.description), 0)::DOUBLE AS description_len,
            coalesce(array_length(p.category), 0)::DOUBLE AS category_depth,
            (hash(p.brand) % 1000003)::DOUBLE / 1000003.0 AS brand_code,
            (hash(coalesce(p.category[2], '')) % 1000003)::DOUBLE / 1000003.0 AS category_code,
            row_number() OVER (
                PARTITION BY r.product_id ORDER BY r.review_time, r.customer_id
            ) - 1 AS product_demand_before,
            row_number() OVER (
                PARTITION BY r.customer_id, r.product_id ORDER BY r.review_time
            ) = 1 AS first_customer_product
        FROM read_parquet('{paths['review']}') r
        INNER JOIN read_parquet('{paths['product']}') p USING (product_id)
    ), daily AS (
        SELECT
            customer_id,
            review_time,
            count(*)::DOUBLE AS event_count,
            sum(price)::DOUBLE AS event_spend,
            sum(price * price)::DOUBLE AS event_price_sq,
            sum(rating)::DOUBLE AS event_rating,
            sum(verified)::DOUBLE AS event_verified,
            sum(text_len)::DOUBLE AS event_text_len,
            sum(summary_len)::DOUBLE AS event_summary_len,
            sum(title_len)::DOUBLE AS event_title_len,
            sum(description_len)::DOUBLE AS event_description_len,
            sum(category_depth)::DOUBLE AS event_category_depth,
            sum(brand_code)::DOUBLE AS event_brand_code,
            sum(brand_code * brand_code)::DOUBLE AS event_brand_code_sq,
            sum(category_code)::DOUBLE AS event_category_code,
            sum(category_code * category_code)::DOUBLE AS event_category_code_sq,
            sum(ln(1 + product_demand_before))::DOUBLE AS event_product_demand,
            sum(first_customer_product::INT)::DOUBLE AS event_unique_products
        FROM raw
        GROUP BY customer_id, review_time
    ), gapped AS (
        SELECT
            *,
            date_diff('day', lag(review_time) OVER (
                PARTITION BY customer_id ORDER BY review_time
            ), review_time)::DOUBLE AS gap_days
        FROM daily
    )
    SELECT
        customer_id,
        review_time,
        min(review_time) OVER customer_history AS first_time,
        sum(event_count) OVER customer_history AS c_events,
        sum(event_spend) OVER customer_history AS c_spend,
        sum(event_price_sq) OVER customer_history AS c_price_sq,
        sum(event_rating) OVER customer_history AS c_rating,
        sum(event_verified) OVER customer_history AS c_verified,
        sum(event_text_len) OVER customer_history AS c_text_len,
        sum(event_summary_len) OVER customer_history AS c_summary_len,
        sum(event_title_len) OVER customer_history AS c_title_len,
        sum(event_description_len) OVER customer_history AS c_description_len,
        sum(event_category_depth) OVER customer_history AS c_category_depth,
        sum(event_brand_code) OVER customer_history AS c_brand_code,
        sum(event_brand_code_sq) OVER customer_history AS c_brand_code_sq,
        sum(event_category_code) OVER customer_history AS c_category_code,
        sum(event_category_code_sq) OVER customer_history AS c_category_code_sq,
        sum(event_product_demand) OVER customer_history AS c_product_demand,
        sum(event_unique_products) OVER customer_history AS c_unique_products,
        count(gap_days) OVER customer_history AS c_gap_n,
        sum(gap_days) OVER customer_history AS c_gap_sum,
        sum(gap_days * gap_days) OVER customer_history AS c_gap_sq,
        max(gap_days) OVER customer_history AS c_gap_max,
        gap_days AS last_gap
    FROM gapped
    WINDOW customer_history AS (
        PARTITION BY customer_id ORDER BY review_time
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )
    """
    start = time.time()
    atomic_copy(con, query, output)
    elapsed = time.time() - start
    n = con.execute(f"SELECT count(*) FROM read_parquet('{output}')").fetchone()[0]
    print(f"[features] event state rows={n} elapsed={elapsed:.1f}s rate={12644508 / max(elapsed, 0.1):.0f} reviews/s")
    register_artifact(
        f"{FEATURE_VERSION}-event-state",
        output,
        "Causal cumulative customer review and joined-product event state",
        "Delete the version directory and rerun main.py after changing feature SQL",
    )
    return output


def state_value(alias: str, column: str) -> str:
    return f"coalesce({alias}.{column}, 0)"


def window_expression(days: int, column: str, name: str) -> str:
    return f"({state_value('s0', column)} - {state_value(f's{days}', column)}) AS {name}{days}"


def ensure_raw_features(split: str) -> Path:
    output = cache_root() / f"raw_features_{split}.parquet"
    if output.exists():
        return output
    paths = data_paths()
    state = ensure_event_state()
    con = connection()
    window_columns = []
    for days in WINDOWS:
        for column, name in (
            ("c_events", "n"),
            ("c_spend", "spend"),
            ("c_price_sq", "price_sq"),
            ("c_rating", "rating_sum"),
            ("c_verified", "verified_sum"),
            ("c_text_len", "text_sum"),
            ("c_summary_len", "summary_sum"),
            ("c_product_demand", "demand_sum"),
            ("c_unique_products", "unique_products"),
        ):
            window_columns.append(window_expression(days, column, name))
    joins = "\n".join(
        f"ASOF LEFT JOIN read_parquet('{state}') s{days} ON b.customer_id = s{days}.customer_id AND b.timestamp - INTERVAL '{days} days' >= s{days}.review_time"
        for days in WINDOWS
    )
    query = f"""
    WITH seeds AS (
        SELECT file_row_number::BIGINT AS row_id, timestamp, customer_id
        FROM read_parquet('{paths[split]}', file_row_number=true)
    ), base AS (
        SELECT
            q.*,
            coalesce(length(c.customer_name), 0)::DOUBLE AS customer_name_len,
            coalesce(length(regexp_replace(c.customer_name, '[^ ]', '', 'g')) + 1, 0)::DOUBLE AS customer_name_words,
            coalesce(length(regexp_replace(c.customer_name, '[^0-9]', '', 'g')), 0)::DOUBLE AS customer_name_digits
        FROM (
            SELECT
                b.row_id,
                b.timestamp,
                b.customer_id,
                date_diff('day', s0.review_time, b.timestamp)::DOUBLE AS recency_days,
                date_diff('day', s0.first_time, b.timestamp)::DOUBLE AS age_days,
                {state_value('s0', 'c_events')} AS frequency,
                greatest({state_value('s0', 'c_events')} - 1, 0) AS frequency_after_first,
                {state_value('s0', 'c_spend')} AS spend_lifetime,
                {state_value('s0', 'c_price_sq')} AS price_sq_lifetime,
                {state_value('s0', 'c_rating')} AS rating_sum_lifetime,
                {state_value('s0', 'c_verified')} AS verified_sum_lifetime,
                {state_value('s0', 'c_text_len')} AS text_sum_lifetime,
                {state_value('s0', 'c_summary_len')} AS summary_sum_lifetime,
                {state_value('s0', 'c_title_len')} AS title_sum_lifetime,
                {state_value('s0', 'c_description_len')} AS description_sum_lifetime,
                {state_value('s0', 'c_category_depth')} AS category_depth_sum_lifetime,
                {state_value('s0', 'c_brand_code')} AS brand_code_sum_lifetime,
                {state_value('s0', 'c_brand_code_sq')} AS brand_code_sq_lifetime,
                {state_value('s0', 'c_category_code')} AS category_code_sum_lifetime,
                {state_value('s0', 'c_category_code_sq')} AS category_code_sq_lifetime,
                {state_value('s0', 'c_product_demand')} AS demand_sum_lifetime,
                {state_value('s0', 'c_unique_products')} AS unique_products_lifetime,
                {state_value('s0', 'c_gap_n')} AS gap_n,
                {state_value('s0', 'c_gap_sum')} AS gap_sum,
                {state_value('s0', 'c_gap_sq')} AS gap_sq,
                {state_value('s0', 'c_gap_max')} AS gap_max,
                coalesce(s0.last_gap, 0) AS last_gap,
                {','.join(window_columns)}
            FROM seeds b
            ASOF LEFT JOIN read_parquet('{state}') s0 ON b.customer_id = s0.customer_id AND b.timestamp >= s0.review_time
            {joins}
        ) q
        LEFT JOIN read_parquet('{paths['customer']}') c USING (customer_id)
    ), engineered AS (
        SELECT
            *,
            spend_lifetime / greatest(frequency, 1) AS price_mean_lifetime,
            sqrt(greatest(price_sq_lifetime / greatest(frequency, 1) - pow(spend_lifetime / greatest(frequency, 1), 2), 0)) AS price_sd_lifetime,
            rating_sum_lifetime / greatest(frequency, 1) AS rating_mean_lifetime,
            verified_sum_lifetime / greatest(frequency, 1) AS verified_share_lifetime,
            text_sum_lifetime / greatest(frequency, 1) AS text_mean_lifetime,
            summary_sum_lifetime / greatest(frequency, 1) AS summary_mean_lifetime,
            title_sum_lifetime / greatest(frequency, 1) AS title_mean_lifetime,
            description_sum_lifetime / greatest(frequency, 1) AS description_mean_lifetime,
            category_depth_sum_lifetime / greatest(frequency, 1) AS category_depth_mean_lifetime,
            demand_sum_lifetime / greatest(frequency, 1) AS demand_mean_lifetime,
            unique_products_lifetime / greatest(frequency, 1) AS product_diversity_share,
            sqrt(greatest(brand_code_sq_lifetime / greatest(frequency, 1) - pow(brand_code_sum_lifetime / greatest(frequency, 1), 2), 0)) AS brand_mix_sd,
            sqrt(greatest(category_code_sq_lifetime / greatest(frequency, 1) - pow(category_code_sum_lifetime / greatest(frequency, 1), 2), 0)) AS category_mix_sd,
            gap_sum / greatest(gap_n, 1) AS gap_mean,
            sqrt(greatest(gap_sq / greatest(gap_n, 1) - pow(gap_sum / greatest(gap_n, 1), 2), 0)) AS gap_sd,
            n30 / greatest(n91 * 30.0 / 91.0, 1.0) AS count_fast_slow_30_91,
            n91 / greatest(n365 * 91.0 / 365.0, 1.0) AS count_fast_slow_91_365,
            spend30 / greatest(spend91 * 30.0 / 91.0, 5.0) AS spend_fast_slow_30_91,
            spend91 / greatest(spend365 * 91.0 / 365.0, 5.0) AS spend_fast_slow_91_365,
            spend30 / greatest(n30, 1) AS price_mean30,
            spend91 / greatest(n91, 1) AS price_mean91,
            spend182 / greatest(n182, 1) AS price_mean182,
            spend365 / greatest(n365, 1) AS price_mean365,
            rating_sum91 / greatest(n91, 1) AS rating_mean91,
            verified_sum91 / greatest(n91, 1) AS verified_share91,
            text_sum91 / greatest(n91, 1) AS text_mean91,
            summary_sum91 / greatest(n91, 1) AS summary_mean91,
            demand_sum91 / greatest(n91, 1) AS demand_mean91,
            unique_products91 / greatest(n91, 1) AS product_diversity_share91,
            (frequency_after_first + 0.5) * 91.0 / greatest(age_days + 30.0, 30.0) AS recurrence_rate91_raw,
            exp(-least((frequency_after_first + 0.5) * 91.0 / greatest(age_days + 30.0, 30.0), 30.0)) AS recurrence_p0_raw,
            exp(-recency_days / greatest(age_days + 30.0, 30.0)) AS recurrence_alive_raw,
            sin(2 * pi() * extract(month FROM timestamp) / 12.0) AS month_sin,
            cos(2 * pi() * extract(month FROM timestamp) / 12.0) AS month_cos,
            date_diff('day', TIMESTAMP '2008-01-01', timestamp) / 365.25 AS database_age_years
        FROM base
    )
    SELECT
        *,
        percent_rank() OVER (PARTITION BY timestamp ORDER BY n30) AS n30_percentile,
        percent_rank() OVER (PARTITION BY timestamp ORDER BY n91) AS n91_percentile,
        percent_rank() OVER (PARTITION BY timestamp ORDER BY frequency) AS frequency_percentile,
        percent_rank() OVER (PARTITION BY timestamp ORDER BY recency_days) AS recency_percentile,
        percent_rank() OVER (PARTITION BY timestamp ORDER BY price_mean_lifetime) AS price_affinity_percentile,
        percent_rank() OVER (PARTITION BY timestamp ORDER BY demand_mean91) AS recent_demand_percentile
    FROM engineered
    ORDER BY row_id
    """
    start = time.time()
    atomic_copy(con, query, output)
    elapsed = time.time() - start
    n = con.execute(f"SELECT count(*) FROM read_parquet('{output}')").fetchone()[0]
    print(f"[features] {split} raw rows={n} elapsed={elapsed:.1f}s")
    register_artifact(
        f"{FEATURE_VERSION}-raw-{split}",
        output,
        f"Split-aligned causal RFM, price, product, review-text scalar, and relative activity features for {split}",
        "Delete this file and rerun main.py after changing feature SQL",
    )
    return output


def ensure_outcomes(split: str) -> Path:
    if split not in {"train", "val"}:
        raise ValueError(split)
    output = cache_root() / f"outcomes_{split}.parquet"
    if output.exists():
        return output
    paths = data_paths()
    state = ensure_event_state()
    con = connection()
    query = f"""
    WITH seeds AS (
        SELECT file_row_number::BIGINT AS row_id, timestamp, customer_id, ltv::DOUBLE AS y91
        FROM read_parquet('{paths[split]}', file_row_number=true)
    )
    SELECT
        b.row_id,
        b.y91,
        greatest(coalesce(sf.c_events, 0) - coalesce(s0.c_events, 0), 0)::BIGINT AS n91
    FROM seeds b
    ASOF LEFT JOIN read_parquet('{state}') s0 ON b.customer_id = s0.customer_id AND b.timestamp >= s0.review_time
    ASOF LEFT JOIN read_parquet('{state}') sf ON b.customer_id = sf.customer_id AND b.timestamp + INTERVAL '91 days' >= sf.review_time
    ORDER BY b.row_id
    """
    start = time.time()
    atomic_copy(con, query, output)
    elapsed = time.time() - start
    check = con.execute(
        f"SELECT count(*), sum(n91=0), sum(abs(y91) > 1e-8 AND n91=0), max(n91) FROM read_parquet('{output}')"
    ).fetchone()
    print(f"[features] {split} outcomes rows={check[0]} zero_count={check[1]} inconsistent={check[2]} max_count={check[3]} elapsed={elapsed:.1f}s")
    register_artifact(
        f"{FEATURE_VERSION}-outcomes-{split}",
        output,
        f"Aligned future 91-day review-count and LTV outcomes for legal {split} training use",
        "Delete this file and rerun main.py after changing outcome construction",
    )
    return output


def load_raw_features(split: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = ensure_raw_features(split)
    return pd.read_parquet(path, columns=columns)


def load_outcomes(split: str) -> pd.DataFrame:
    return pd.read_parquet(ensure_outcomes(split)).sort_values("row_id", kind="stable").reset_index(drop=True)


def target_history_features(seeds: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    seed = seeds[["row_id", "timestamp", "customer_id"]].copy()
    src = source[["timestamp", "customer_id", "y91"]].copy()
    src["available_time"] = src["timestamp"] + pd.Timedelta(days=91)
    src = src.sort_values(["customer_id", "available_time"], kind="stable")
    grouped = src.groupby("customer_id", sort=False, observed=True)
    src["target_hist_n"] = grouped.cumcount() + 1
    src["target_hist_sum"] = grouped["y91"].cumsum()
    src["target_hist_sq_sum"] = (src["y91"] * src["y91"]).groupby(src["customer_id"], sort=False).cumsum()
    positive = src["y91"].gt(0).astype(np.int32)
    src["target_hist_positive_n"] = positive.groupby(src["customer_id"], sort=False).cumsum()
    src["target_hist_last"] = src["y91"]
    src["target_hist_zero_share"] = 1.0 - src["target_hist_positive_n"] / src["target_hist_n"]
    src["target_hist_mean"] = src["target_hist_sum"] / src["target_hist_n"]
    src["target_hist_positive_mean"] = src["target_hist_sum"] / src["target_hist_positive_n"].clip(lower=1)
    src["target_hist_sd"] = np.sqrt(
        np.maximum(src["target_hist_sq_sum"] / src["target_hist_n"] - src["target_hist_mean"] ** 2, 0)
    )
    state_columns = [
        "customer_id",
        "available_time",
        "target_hist_n",
        "target_hist_last",
        "target_hist_zero_share",
        "target_hist_mean",
        "target_hist_positive_mean",
        "target_hist_sd",
    ]
    con = duckdb.connect()
    con.register("seed_rows", seed)
    con.register("target_state", src[state_columns])
    result = con.execute(
        """
        SELECT
            s.row_id,
            coalesce(t.target_hist_n, 0)::DOUBLE AS target_hist_n,
            coalesce(t.target_hist_last, 0)::DOUBLE AS target_hist_last,
            coalesce(t.target_hist_zero_share, 1)::DOUBLE AS target_hist_zero_share,
            coalesce(t.target_hist_mean, 0)::DOUBLE AS target_hist_mean,
            coalesce(t.target_hist_positive_mean, 0)::DOUBLE AS target_hist_positive_mean,
            coalesce(t.target_hist_sd, 0)::DOUBLE AS target_hist_sd,
            coalesce(date_diff('day', t.available_time, s.timestamp), 10000)::DOUBLE AS target_hist_recency_days
        FROM seed_rows s
        ASOF LEFT JOIN target_state t
        ON s.customer_id = t.customer_id AND s.timestamp >= t.available_time
        ORDER BY s.row_id
        """
    ).df()
    return result


def add_target_history(seeds: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    history = target_history_features(seeds, source)
    if not np.array_equal(seeds["row_id"].to_numpy(), history["row_id"].to_numpy()):
        raise RuntimeError("target history row order changed")
    result = seeds.copy()
    for column in history.columns:
        if column != "row_id":
            result[column] = history[column].to_numpy(dtype=np.float32)
    return result


def validate_feature_rows(frame: pd.DataFrame, expected: int) -> None:
    if len(frame) != expected:
        raise RuntimeError(f"feature row count {len(frame)} != {expected}")
    row_id = frame["row_id"].to_numpy()
    if not np.array_equal(row_id, np.arange(expected)):
        raise RuntimeError("feature row IDs are not immutable split order")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64, copy=False)
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("non-finite feature value")
