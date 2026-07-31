from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


WINDOWS = (7, 14, 28, 56, 91, 182, 364)
CONTENT_WINDOWS = (7, 28, 91, 364)


def elapsed(start: float, name: str) -> None:
    print(f"[timing] {name}: {time.time() - start:.1f}s", flush=True)


def sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def frame_signature(frames: dict[str, pd.DataFrame], cache_root: Path, debug: bool) -> str:
    digest = hashlib.sha256()
    digest.update(b"causal-hazard-lane0-v6")
    digest.update(str(debug).encode())
    for split in ("train", "val", "test"):
        frame = frames[split]
        digest.update(split.encode())
        digest.update(str(frame.shape).encode())
        values = frame[["timestamp", "customer_id"]]
        digest.update(pd.util.hash_pandas_object(values, index=True).values.tobytes())
    for name in ("transactions", "customer", "article"):
        path = cache_root / "rel-hm" / "db" / f"{name}.parquet"
        stat = path.stat()
        digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:20]


def prepare_seeds(frames: dict[str, pd.DataFrame], debug: bool) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    prepared = []
    selected = {}
    train_max = pd.Timestamp(frames["train"]["timestamp"].max())
    for split in ("train", "val", "test"):
        source = frames[split].reset_index(drop=True)
        keep = np.ones(len(source), dtype=bool)
        if debug:
            hashes = pd.util.hash_pandas_object(source["customer_id"], index=False).to_numpy()
            keep = hashes % 23 == 0
            if split == "train":
                keep &= source["timestamp"].to_numpy() >= np.datetime64(train_max - pd.Timedelta(days=14))
        selected[split] = np.flatnonzero(keep)
        part = source.loc[keep, ["timestamp", "customer_id"]].copy()
        part["_split"] = split
        part["_row_id"] = np.flatnonzero(keep).astype(np.int64)
        part["label"] = source.loc[keep, "churn"].to_numpy() if "churn" in source else np.nan
        prepared.append(part)
    seeds = pd.concat(prepared, ignore_index=True)
    seeds["timestamp"] = pd.to_datetime(seeds["timestamp"])
    seeds["customer_id"] = seeds["customer_id"].astype(np.int64)
    return seeds, selected


class FeatureBuilder:
    def __init__(self, frames: dict[str, pd.DataFrame], debug: bool):
        self.frames = frames
        self.debug = debug
        self.cache_root = Path(os.environ["RELBENCH_CACHE_DIR"])
        self.shared_root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
        self.seeds, self.selected = prepare_seeds(frames, debug)
        self.key = frame_signature(frames, self.cache_root, debug)
        self.root = self.shared_root / "lane0_causal_hazard_v6" / self.key
        self.root.mkdir(parents=True, exist_ok=True)
        self.core_path = self.root / "core_features.parquet"
        self.wide_path = self.root / "wide_features.parquet"
        self.priority_path = self.root / "priority_features_v1.parquet"
        self.priority_two_path = self.root / "priority_features_v2.parquet"
        self.database_path = self.root / "build.duckdb"
        self.con: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self.con is not None:
            return self.con
        self.con = duckdb.connect(str(self.database_path))
        threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "11")))
        temp_root = self.root / "duckdb_temp"
        temp_root.mkdir(exist_ok=True)
        self.con.execute(f"SET threads={threads}")
        self.con.execute("SET preserve_insertion_order=false")
        self.con.execute(f"SET temp_directory='{sql_path(temp_root)}'")
        self.con.register("seed_frame", self.seeds)
        self.con.execute("CREATE OR REPLACE TABLE seeds AS SELECT * FROM seed_frame")
        db_root = self.cache_root / "rel-hm" / "db"
        transaction_path = sql_path(db_root / "transactions.parquet")
        customer_path = sql_path(db_root / "customer.parquet")
        article_path = sql_path(db_root / "article.parquet")
        self.con.execute(f"CREATE OR REPLACE VIEW customer_source AS SELECT * FROM read_parquet('{customer_path}')")
        self.con.execute(f"CREATE OR REPLACE VIEW article_source AS SELECT * FROM read_parquet('{article_path}')")
        if self.debug:
            self.con.execute("CREATE OR REPLACE TABLE selected_customers AS SELECT DISTINCT customer_id FROM seeds")
            self.con.execute(
                f"CREATE OR REPLACE VIEW tx_source AS SELECT t.* FROM read_parquet('{transaction_path}') t INNER JOIN selected_customers c USING(customer_id)"
            )
        else:
            self.con.execute(f"CREATE OR REPLACE VIEW tx_source AS SELECT * FROM read_parquet('{transaction_path}')")
        return self.con

    def build_core(self) -> Path:
        if self.core_path.exists():
            print(f"[cache] core features: {self.core_path}", flush=True)
            return self.core_path
        start = time.time()
        con = self.connect()
        con.execute(
            """
            CREATE OR REPLACE TABLE customer_profile AS
            WITH c AS (
                SELECT *,
                    count(*) OVER (PARTITION BY postal_code) AS postal_frequency
                FROM customer_source
            )
            SELECT
                customer_id,
                age,
                CASE WHEN age IS NULL THEN 1 ELSE 0 END AS age_missing,
                coalesce(FN, 0) AS FN,
                CASE WHEN FN IS NULL THEN 1 ELSE 0 END AS FN_missing,
                coalesce(Active, 0) AS Active,
                CASE WHEN Active IS NULL THEN 1 ELSE 0 END AS Active_missing,
                CAST(hash(coalesce(club_member_status, '__missing__')) % 2147483647 AS BIGINT) AS membership_code,
                CAST(hash(coalesce(fashion_news_frequency, '__missing__')) % 2147483647 AS BIGINT) AS news_code,
                postal_frequency,
                CASE WHEN postal_frequency >= 50 THEN CAST(hash(postal_code) % 2147483647 AS BIGINT) ELSE 0 END AS postal_bucket,
                CASE WHEN age IS NULL THEN -1 ELSE CAST(floor(age / 10) AS BIGINT) END AS age_bucket
            FROM c
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE week_basic AS
            SELECT
                customer_id,
                CAST(date_trunc('week', t_dat - INTERVAL 1 DAY) + INTERVAL 7 DAY AS TIMESTAMP) AS timestamp,
                count(*)::DOUBLE AS tx_count_week,
                count(DISTINCT t_dat)::DOUBLE AS active_days_week,
                count(DISTINCT article_id)::DOUBLE AS distinct_articles_week,
                sum(price)::DOUBLE AS spend_week,
                sum(price * price)::DOUBLE AS price_sq_week,
                avg(price)::DOUBLE AS price_mean_week,
                stddev_samp(price)::DOUBLE AS price_std_week,
                min(price)::DOUBLE AS price_min_week,
                max(price)::DOUBLE AS price_max_week,
                arg_max(price, t_dat)::DOUBLE AS last_price,
                sum(CASE WHEN sales_channel_id = 2 THEN 1 ELSE 0 END)::DOUBLE AS online_count_week,
                avg(CASE WHEN sales_channel_id = 2 THEN 1.0 ELSE 0.0 END)::DOUBLE AS online_share_week,
                mode(sales_channel_id)::BIGINT AS dominant_channel,
                count(DISTINCT sales_channel_id)::DOUBLE AS distinct_channels_week,
                min(t_dat) AS first_day_week,
                max(t_dat) AS last_day_week
            FROM tx_source
            GROUP BY customer_id, timestamp
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE day_zero AS
            SELECT customer_id, t_dat, count(*)::DOUBLE AS tx_count_day, sum(price)::DOUBLE AS spend_day
            FROM tx_source
            GROUP BY customer_id, t_dat
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE day_one AS
            SELECT
                *,
                date_diff('day', lag(t_dat) OVER (PARTITION BY customer_id ORDER BY t_dat), t_dat)::DOUBLE AS gap_days,
                row_number() OVER (PARTITION BY customer_id ORDER BY t_dat)::DOUBLE AS active_day_sequence,
                min(t_dat) OVER (PARTITION BY customer_id ORDER BY t_dat ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS first_active_day,
                lag(t_dat, 1) OVER (PARTITION BY customer_id ORDER BY t_dat) AS previous_active_day,
                lag(t_dat, 2) OVER (PARTITION BY customer_id ORDER BY t_dat) AS third_last_active_day
            FROM day_zero
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE day_features AS
            SELECT
                *,
                avg(gap_days) OVER d AS gap_mean,
                median(gap_days) OVER d AS gap_median,
                stddev_samp(gap_days) OVER d AS gap_std,
                avg(gap_days) OVER (PARTITION BY customer_id ORDER BY t_dat ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS gap_recent_5,
                avg(gap_days) OVER (PARTITION BY customer_id ORDER BY t_dat ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS gap_recent_10,
                0.45 * coalesce(gap_days, 0)
                    + 0.25 * coalesce(lag(gap_days, 1) OVER (PARTITION BY customer_id ORDER BY t_dat), gap_days, 0)
                    + 0.15 * coalesce(lag(gap_days, 2) OVER (PARTITION BY customer_id ORDER BY t_dat), gap_days, 0)
                    + 0.10 * coalesce(lag(gap_days, 3) OVER (PARTITION BY customer_id ORDER BY t_dat), gap_days, 0)
                    + 0.05 * coalesce(lag(gap_days, 4) OVER (PARTITION BY customer_id ORDER BY t_dat), gap_days, 0) AS gap_ewm
            FROM day_one
            WINDOW d AS (PARTITION BY customer_id ORDER BY t_dat ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE week_sequence_zero AS
            SELECT
                *,
                row_number() OVER (PARTITION BY customer_id ORDER BY timestamp) AS active_week_sequence,
                date_diff('week', lag(timestamp) OVER (PARTITION BY customer_id ORDER BY timestamp), timestamp) AS gap_weeks,
                date_diff('week', timestamp, lead(timestamp) OVER (PARTITION BY customer_id ORDER BY timestamp)) AS next_gap_weeks
            FROM week_basic
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE week_sequence_one AS
            SELECT *, timestamp - active_week_sequence * INTERVAL 7 DAY AS island_key
            FROM week_sequence_zero
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE week_state AS
            SELECT
                *,
                row_number() OVER (PARTITION BY customer_id, island_key ORDER BY timestamp)::DOUBLE AS active_week_streak
            FROM week_sequence_one
            """
        )
        roll_select = [
            "w.customer_id",
            "w.timestamp",
            "w.gap_weeks",
            "w.next_gap_weeks",
            "w.active_week_sequence",
            "w.active_week_streak",
            "w.last_day_week",
            "w.first_day_week",
            "w.last_price",
            "w.dominant_channel",
            "w.distinct_channels_week",
        ]
        window_defs = []
        for days in WINDOWS:
            alias = f"r{days}"
            preceding = days - 7
            window_defs.append(
                f"{alias} AS (PARTITION BY customer_id ORDER BY timestamp RANGE BETWEEN INTERVAL {preceding} DAY PRECEDING AND CURRENT ROW)"
            )
            count_expr = f"sum(tx_count_week) OVER {alias}"
            spend_expr = f"sum(spend_week) OVER {alias}"
            sq_expr = f"sum(price_sq_week) OVER {alias}"
            roll_select.extend(
                [
                    f"{count_expr} AS tx_count_{days}d",
                    f"sum(active_days_week) OVER {alias} AS active_days_{days}d",
                    f"(count(*) OVER {alias})::DOUBLE AS active_weeks_{days}d",
                    f"sum(distinct_articles_week) OVER {alias} AS distinct_articles_{days}d",
                    f"{spend_expr} AS spend_{days}d",
                    f"{spend_expr} / greatest({count_expr}, 1) AS price_mean_{days}d",
                    f"sqrt(greatest({sq_expr} / greatest({count_expr}, 1) - pow({spend_expr} / greatest({count_expr}, 1), 2), 0)) AS price_std_{days}d",
                    f"min(price_min_week) OVER {alias} AS price_min_{days}d",
                    f"max(price_max_week) OVER {alias} AS price_max_{days}d",
                    f"sum(online_count_week) OVER {alias} / greatest({count_expr}, 1) AS online_share_{days}d",
                    f"avg(tx_count_week) OVER {alias} AS basket_mean_{days}d",
                ]
            )
        con.execute(
            "CREATE OR REPLACE TABLE week_roll AS SELECT "
            + ",\n".join(roll_select)
            + "\nFROM week_state w WINDOW "
            + ",\n".join(window_defs)
        )
        lag_select = ["c.customer_id", "c.timestamp"]
        for k in range(1, 53):
            lag_select.append(
                f"coalesce(max(CASE WHEN date_diff('week', p.timestamp, c.timestamp) = {k} THEN 1 ELSE 0 END), 0)::DOUBLE AS active_flag_lag_{k}"
            )
            lag_select.append(
                f"coalesce(sum(CASE WHEN date_diff('week', p.timestamp, c.timestamp) = {k} THEN p.tx_count_week ELSE 0 END), 0)::DOUBLE AS tx_count_lag_{k}"
            )
        lag_select.extend(
            [
                "(coalesce(sum(CASE WHEN p.next_gap_weeks = 1 THEN 1 ELSE 0 END), 0) + 2.0) / (count(p.timestamp) + 10.0) AS continuation_rate_lifetime",
                "(coalesce(sum(CASE WHEN p.timestamp >= c.timestamp - INTERVAL 91 DAY AND p.next_gap_weeks = 1 THEN 1 ELSE 0 END), 0) + 2.0) / (count(CASE WHEN p.timestamp >= c.timestamp - INTERVAL 91 DAY THEN 1 END) + 10.0) AS continuation_rate_recent13",
                "(coalesce(sum(CASE WHEN p.timestamp <= c.timestamp - INTERVAL 14 DAY AND p.next_gap_weeks = 2 THEN 1 ELSE 0 END), 0) + 2.0) / (count(CASE WHEN p.timestamp <= c.timestamp - INTERVAL 14 DAY AND coalesce(p.next_gap_weeks, 99) <> 1 THEN 1 END) + 10.0) AS recovery_after_inactive_rate",
                "(coalesce(sum(CASE WHEN p.gap_weeks = 1 AND p.next_gap_weeks = 1 THEN 1 ELSE 0 END), 0) + 2.0) / (count(CASE WHEN p.gap_weeks = 1 THEN 1 END) + 10.0) AS order2_active_active_rate",
                "coalesce(max(greatest(coalesce(p.gap_weeks, 1) - 1, 0)), 0)::DOUBLE AS longest_prior_inactivity",
                "count(p.timestamp)::DOUBLE AS completed_personal_transitions",
            ]
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE week_lag_features AS
            SELECT
            """
            + ",\n".join(lag_select)
            + """
            FROM week_state c
            LEFT JOIN week_state p
                ON c.customer_id = p.customer_id
                AND p.timestamp <= c.timestamp - INTERVAL 7 DAY
                AND p.timestamp >= c.timestamp - INTERVAL 364 DAY
            GROUP BY c.customer_id, c.timestamp
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE core_features AS
            SELECT
                s._split,
                s._row_id,
                s.timestamp,
                s.customer_id,
                s.label,
                date_diff('day', TIMESTAMP '2019-09-01', s.timestamp)::DOUBLE AS calendar_day,
                sin(2 * pi() * date_diff('day', TIMESTAMP '2019-09-01', s.timestamp) / 365.25) AS year_sin,
                cos(2 * pi() * date_diff('day', TIMESTAMP '2019-09-01', s.timestamp) / 365.25) AS year_cos,
                p.* EXCLUDE(customer_id),
                r.* EXCLUDE(customer_id, timestamp, last_day_week, next_gap_weeks),
                date_diff('day', d.t_dat, s.timestamp)::DOUBLE AS days_since_last,
                date_diff('day', d.previous_active_day, s.timestamp)::DOUBLE AS days_since_second_last,
                date_diff('day', d.third_last_active_day, s.timestamp)::DOUBLE AS days_since_third_last,
                date_diff('day', d.first_active_day, s.timestamp)::DOUBLE AS tenure_days,
                d.active_day_sequence,
                d.gap_days AS last_interpurchase_gap,
                d.gap_mean,
                d.gap_median,
                d.gap_std,
                d.gap_recent_5,
                d.gap_recent_10,
                d.gap_ewm,
                date_diff('day', d.t_dat, s.timestamp) / greatest(d.gap_median, 1) AS current_gap_over_median,
                r.tx_count_364d / greatest(d.active_day_sequence, 1) AS transactions_per_active_day,
                l.* EXCLUDE(customer_id, timestamp),
                l.continuation_rate_recent13 - l.continuation_rate_lifetime AS continuation_rate_shift,
                r.active_weeks_28d / 4.0 - (r.active_weeks_56d - r.active_weeks_28d) / 4.0 AS activity_slope_4_8,
                r.active_weeks_56d / 8.0 - (r.active_weeks_91d - r.active_weeks_56d) / 5.0 AS activity_slope_8_13,
                r.tx_count_28d / 4.0 - (r.tx_count_56d - r.tx_count_28d) / 4.0 AS count_slope_4_8,
                r.tx_count_56d / 8.0 - (r.tx_count_91d - r.tx_count_56d) / 5.0 AS count_slope_8_13,
                greatest(l.active_flag_lag_51, l.active_flag_lag_52) AS active_same_week_prior_year,
                CASE WHEN r.dominant_channel <> lag(r.dominant_channel) OVER (PARTITION BY s.customer_id ORDER BY s.timestamp) THEN 1 ELSE 0 END AS recent_channel_switch
            FROM seeds s
            INNER JOIN week_roll r USING(customer_id, timestamp)
            INNER JOIN customer_profile p USING(customer_id)
            LEFT JOIN day_features d ON r.customer_id = d.customer_id AND r.last_day_week = d.t_dat
            LEFT JOIN week_lag_features l USING(customer_id, timestamp)
            ORDER BY s._split, s._row_id
            """
        )
        con.execute(f"COPY core_features TO '{sql_path(self.core_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        elapsed(start, "core feature build")
        return self.core_path

    def build_wide(self) -> Path:
        if self.wide_path.exists():
            print(f"[cache] wide features: {self.wide_path}", flush=True)
            return self.wide_path
        start = time.time()
        con = self.connect()
        con.execute(
            """
            CREATE OR REPLACE TABLE tx_article_zero AS
            SELECT
                t.*,
                CAST(date_trunc('week', t.t_dat - INTERVAL 1 DAY) + INTERVAL 7 DAY AS TIMESTAMP) AS timestamp,
                a.product_code,
                a.product_type_no,
                CAST(hash(a.product_group_name) % 2147483647 AS BIGINT) AS product_group_id,
                a.department_no,
                a.index_group_no,
                a.section_no,
                a.garment_group_no,
                a.colour_group_code,
                a.perceived_colour_master_id
            FROM tx_source t
            INNER JOIN article_source a USING(article_id)
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE tx_article_one AS
            SELECT
                *,
                min(t_dat) OVER (PARTITION BY article_id) AS article_first_market_day,
                min(t_dat) OVER (PARTITION BY customer_id, article_id) AS customer_article_first_day,
                min(t_dat) OVER (PARTITION BY customer_id, product_code) AS customer_product_first_day,
                min(t_dat) OVER (PARTITION BY customer_id, product_group_id) AS customer_group_first_day,
                avg(price) OVER article_history AS article_price_baseline,
                stddev_samp(price) OVER article_history AS article_price_history_std,
                (count(*) OVER article_history)::DOUBLE AS article_popularity_cumulative,
                (count(*) OVER article_recent)::DOUBLE AS article_popularity_28d,
                avg(price) OVER group_history AS group_price_baseline,
                stddev_samp(price) OVER group_history AS group_price_history_std,
                (count(*) OVER customer_history)::DOUBLE AS customer_tx_cumulative,
                (count(*) OVER customer_group_history)::DOUBLE AS customer_group_tx_cumulative,
                (count(*) OVER customer_department_history)::DOUBLE AS customer_department_tx_cumulative
            FROM tx_article_zero
            WINDOW
                article_history AS (PARTITION BY article_id ORDER BY t_dat RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
                article_recent AS (PARTITION BY article_id ORDER BY t_dat RANGE BETWEEN INTERVAL 27 DAY PRECEDING AND CURRENT ROW),
                group_history AS (PARTITION BY product_group_id ORDER BY t_dat RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
                customer_history AS (PARTITION BY customer_id ORDER BY t_dat RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
                customer_group_history AS (PARTITION BY customer_id, product_group_id ORDER BY t_dat RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
                customer_department_history AS (PARTITION BY customer_id, department_no ORDER BY t_dat RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE wide_week AS
            SELECT
                customer_id,
                timestamp,
                count(DISTINCT product_code)::DOUBLE AS distinct_product_codes_week,
                count(DISTINCT product_type_no)::DOUBLE AS distinct_product_types_week,
                count(DISTINCT product_group_id)::DOUBLE AS distinct_product_groups_week,
                count(DISTINCT department_no)::DOUBLE AS distinct_departments_week,
                count(DISTINCT index_group_no)::DOUBLE AS distinct_index_groups_week,
                count(DISTINCT section_no)::DOUBLE AS distinct_sections_week,
                count(DISTINCT garment_group_no)::DOUBLE AS distinct_garment_groups_week,
                count(DISTINCT colour_group_code)::DOUBLE AS distinct_colors_week,
                count(DISTINCT perceived_colour_master_id)::DOUBLE AS distinct_perceived_colors_week,
                entropy(product_group_id)::DOUBLE AS product_group_entropy_week,
                entropy(department_no)::DOUBLE AS department_entropy_week,
                entropy(product_type_no)::DOUBLE AS product_type_entropy_week,
                list_max(map_values(histogram(product_group_id)))::DOUBLE / count(*) AS product_group_dominant_share_week,
                list_max(map_values(histogram(department_no)))::DOUBLE / count(*) AS department_dominant_share_week,
                mode(product_group_id)::BIGINT AS dominant_product_group,
                mode(department_no)::BIGINT AS dominant_department,
                mode(index_group_no)::BIGINT AS dominant_index_group,
                mode(section_no)::BIGINT AS dominant_section,
                avg(CASE WHEN customer_article_first_day <= timestamp - INTERVAL 7 DAY THEN 1.0 ELSE 0.0 END) AS repeated_article_share_week,
                avg(CASE WHEN customer_product_first_day <= timestamp - INTERVAL 7 DAY THEN 1.0 ELSE 0.0 END) AS repeated_product_share_week,
                avg(CASE WHEN customer_group_first_day > timestamp - INTERVAL 7 DAY THEN 1.0 ELSE 0.0 END) AS novel_group_share_week,
                avg(customer_group_tx_cumulative / greatest(customer_tx_cumulative, 1)) AS group_preference_affinity_week,
                max(customer_group_tx_cumulative / greatest(customer_tx_cumulative, 1)) AS top_group_preference_share_week,
                avg(customer_department_tx_cumulative / greatest(customer_tx_cumulative, 1)) AS department_preference_affinity_week,
                max(customer_department_tx_cumulative / greatest(customer_tx_cumulative, 1)) AS top_department_preference_share_week,
                avg(price / greatest(article_price_baseline, 0.000001)) AS article_price_relative_week,
                stddev_samp(price / greatest(article_price_baseline, 0.000001)) AS article_price_relative_std_week,
                avg(CASE WHEN price < 0.8 * article_price_baseline THEN 1.0 ELSE 0.0 END) AS low_price_share_week,
                avg((price - group_price_baseline) / greatest(group_price_history_std, 0.000001)) AS category_price_z_week,
                stddev_samp((price - group_price_baseline) / greatest(group_price_history_std, 0.000001)) AS category_price_z_std_week,
                avg(ln(1 + article_popularity_cumulative)) AS article_popularity_log_week,
                avg(article_popularity_28d / greatest(article_popularity_cumulative, 1)) AS article_popularity_momentum_week,
                avg(ln(1 + article_popularity_cumulative) / greatest(date_diff('day', article_first_market_day, t_dat) + 1, 1)) AS article_popularity_velocity_week
            FROM tx_article_one
            GROUP BY customer_id, timestamp
            """
        )
        wide_select = [
            "customer_id",
            "timestamp",
            "dominant_product_group",
            "dominant_department",
            "dominant_index_group",
            "dominant_section",
        ]
        wide_defs = []
        diversity = (
            "distinct_product_codes",
            "distinct_product_types",
            "distinct_product_groups",
            "distinct_departments",
            "distinct_index_groups",
            "distinct_sections",
            "distinct_garment_groups",
            "distinct_colors",
            "distinct_perceived_colors",
        )
        means = (
            "product_group_entropy",
            "department_entropy",
            "product_type_entropy",
            "product_group_dominant_share",
            "department_dominant_share",
            "repeated_article_share",
            "repeated_product_share",
            "novel_group_share",
            "group_preference_affinity",
            "top_group_preference_share",
            "department_preference_affinity",
            "top_department_preference_share",
            "article_price_relative",
            "article_price_relative_std",
            "low_price_share",
            "category_price_z",
            "category_price_z_std",
            "article_popularity_log",
            "article_popularity_momentum",
            "article_popularity_velocity",
        )
        for days in CONTENT_WINDOWS:
            alias = f"c{days}"
            wide_defs.append(
                f"{alias} AS (PARTITION BY customer_id ORDER BY timestamp RANGE BETWEEN INTERVAL {days - 7} DAY PRECEDING AND CURRENT ROW)"
            )
            for name in diversity:
                wide_select.append(f"sum({name}_week) OVER {alias} AS {name}_{days}d")
            for name in means:
                wide_select.append(f"avg({name}_week) OVER {alias} AS {name}_{days}d")
        con.execute(
            "CREATE OR REPLACE TABLE wide_roll AS SELECT "
            + ",\n".join(wide_select)
            + "\nFROM wide_week WINDOW "
            + ",\n".join(wide_defs)
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE basket_dimension AS
            SELECT customer_id, timestamp, 1::BIGINT AS dimension_id, product_group_id::BIGINT AS category_id, count(*)::DOUBLE AS weight FROM tx_article_one GROUP BY 1,2,4
            UNION ALL
            SELECT customer_id, timestamp, 2, department_no::BIGINT, count(*)::DOUBLE FROM tx_article_one GROUP BY 1,2,4
            UNION ALL
            SELECT customer_id, timestamp, 3, index_group_no::BIGINT, count(*)::DOUBLE FROM tx_article_one GROUP BY 1,2,4
            UNION ALL
            SELECT customer_id, timestamp, 4, section_no::BIGINT, count(*)::DOUBLE FROM tx_article_one GROUP BY 1,2,4
            UNION ALL
            SELECT customer_id, timestamp, 5, sales_channel_id::BIGINT, count(*)::DOUBLE FROM tx_article_one GROUP BY 1,2,4
            UNION ALL
            SELECT w.customer_id, w.timestamp, 6, p.age_bucket, 1.0 FROM week_basic w INNER JOIN customer_profile p USING(customer_id)
            UNION ALL
            SELECT w.customer_id, w.timestamp, 7, p.membership_code, 1.0 FROM week_basic w INNER JOIN customer_profile p USING(customer_id)
            UNION ALL
            SELECT w.customer_id, w.timestamp, 8, p.postal_bucket, 1.0 FROM week_basic w INNER JOIN customer_profile p USING(customer_id)
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE cohort_time AS
            SELECT
                b.dimension_id,
                b.category_id,
                b.timestamp AS source_timestamp,
                sum(b.weight * CASE WHEN w.next_gap_weeks = 1 THEN 1 ELSE 0 END)::DOUBLE AS active_weight,
                sum(b.weight)::DOUBLE AS total_weight
            FROM basket_dimension b
            INNER JOIN week_state w USING(customer_id, timestamp)
            GROUP BY 1,2,3
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE cohort_cumulative AS
            SELECT
                *,
                sum(active_weight) OVER h AS active_cumulative,
                sum(total_weight) OVER h AS total_cumulative
            FROM cohort_time
            WINDOW h AS (PARTITION BY dimension_id, category_id ORDER BY source_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            """
        )
        cohort_select = ["b.customer_id", "b.timestamp"]
        for dim in range(1, 9):
            prior = "(coalesce(c.active_cumulative, 0) + 5.0) / (coalesce(c.total_cumulative, 0) + 25.0)"
            cohort_select.extend(
                [
                    f"sum(CASE WHEN b.dimension_id = {dim} THEN b.weight * {prior} ELSE 0 END) / greatest(sum(CASE WHEN b.dimension_id = {dim} THEN b.weight ELSE 0 END), 1) AS cohort_prior_mean_{dim}",
                    f"min(CASE WHEN b.dimension_id = {dim} THEN {prior} END) AS cohort_prior_min_{dim}",
                    f"max(CASE WHEN b.dimension_id = {dim} THEN {prior} END) AS cohort_prior_max_{dim}",
                ]
            )
        con.execute(
            """
            CREATE OR REPLACE TABLE cohort_features AS
            SELECT
            """
            + ",\n".join(cohort_select)
            + """
            FROM basket_dimension b
            ASOF LEFT JOIN cohort_cumulative c
                ON b.dimension_id = c.dimension_id
                AND b.category_id = c.category_id
                AND b.timestamp - INTERVAL 7 DAY >= c.source_timestamp
            GROUP BY b.customer_id, b.timestamp
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE global_week_zero AS
            SELECT
                timestamp,
                sum(tx_count_week)::DOUBLE AS global_volume,
                count(*)::DOUBLE AS global_active_customers,
                sum(online_count_week) / greatest(sum(tx_count_week), 1) AS global_online_share,
                sum(spend_week) / greatest(sum(tx_count_week), 1) AS global_price_mean
            FROM week_basic
            GROUP BY timestamp
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE global_week AS
            SELECT
                *,
                global_volume / greatest(avg(global_volume) OVER g4, 1) AS global_volume_vs_4w,
                global_volume / greatest(avg(global_volume) OVER g13, 1) AS global_volume_vs_13w,
                global_active_customers / greatest(avg(global_active_customers) OVER g4, 1) AS global_customers_vs_4w,
                global_active_customers / greatest(avg(global_active_customers) OVER g13, 1) AS global_customers_vs_13w,
                global_online_share - avg(global_online_share) OVER g4 AS global_online_shift_4w,
                global_online_share - avg(global_online_share) OVER g13 AS global_online_shift_13w,
                global_price_mean - avg(global_price_mean) OVER g4 AS global_price_shift_4w,
                global_price_mean - avg(global_price_mean) OVER g13 AS global_price_shift_13w,
                global_volume - lag(global_volume) OVER (ORDER BY timestamp) AS global_volume_change_1w,
                global_active_customers - lag(global_active_customers) OVER (ORDER BY timestamp) AS global_customers_change_1w
            FROM global_week_zero
            WINDOW
                g4 AS (ORDER BY timestamp ROWS BETWEEN 3 PRECEDING AND CURRENT ROW),
                g13 AS (ORDER BY timestamp ROWS BETWEEN 12 PRECEDING AND CURRENT ROW)
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE wide_features AS
            SELECT
                s._split,
                s._row_id,
                s.timestamp,
                s.customer_id,
                w.* EXCLUDE(customer_id, timestamp),
                w.product_group_entropy_7d - w.product_group_entropy_364d AS product_group_entropy_drift,
                w.department_entropy_7d - w.department_entropy_364d AS department_entropy_drift,
                w.group_preference_affinity_7d - w.group_preference_affinity_364d AS group_preference_drift,
                w.article_price_relative_7d - w.article_price_relative_91d AS recent_price_shift,
                w.article_popularity_momentum_7d - w.article_popularity_momentum_91d AS popularity_momentum_shift,
                c.* EXCLUDE(customer_id, timestamp),
                g.* EXCLUDE(timestamp)
            FROM seeds s
            INNER JOIN wide_roll w USING(customer_id, timestamp)
            LEFT JOIN cohort_features c USING(customer_id, timestamp)
            LEFT JOIN global_week g USING(timestamp)
            ORDER BY s._split, s._row_id
            """
        )
        con.execute(f"COPY wide_features TO '{sql_path(self.wide_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        elapsed(start, "wide feature build")
        return self.wide_path

    def build_priority(self) -> Path:
        if self.priority_path.exists():
            print(f"[cache] priority features: {self.priority_path}", flush=True)
            return self.priority_path
        start = time.time()
        con = self.connect()
        purchase_select = [
            "customer_id",
            "CAST(date_trunc('week', t_dat - INTERVAL 1 DAY) + INTERVAL 7 DAY AS TIMESTAMP) AS timestamp",
            "date_diff('day', min(t_dat), max(t_dat))::DOUBLE AS basket_day_span",
            "count(*) FILTER (WHERE t_dat = max(t_dat) OVER (PARTITION BY customer_id, date_trunc('week', t_dat - INTERVAL 1 DAY)))::DOUBLE AS last_day_basket_rows",
        ]
        for offset in range(7):
            purchase_select.append(
                f"sum(CASE WHEN date_diff('day', t_dat, date_trunc('week', t_dat - INTERVAL 1 DAY) + INTERVAL 7 DAY) = {offset} THEN 1 ELSE 0 END)::DOUBLE AS transaction_rows_offset_{offset}"
            )
        con.execute(
            """
            CREATE OR REPLACE TABLE purchase_shape_zero AS
            SELECT
                *,
                max(t_dat) OVER (
                    PARTITION BY customer_id, date_trunc('week', t_dat - INTERVAL 1 DAY) + INTERVAL 7 DAY
                ) AS week_last_day
            FROM tx_source
            """
        )
        purchase_select[3] = "sum(CASE WHEN t_dat = week_last_day THEN 1 ELSE 0 END)::DOUBLE AS last_day_basket_rows"
        con.execute(
            "CREATE OR REPLACE TABLE purchase_shape AS SELECT "
            + ",\n".join(purchase_select)
            + """
            ,
                sum(exp(-0.35 * date_diff('day', t_dat, date_trunc('week', t_dat - INTERVAL 1 DAY) + INTERVAL 7 DAY)))::DOUBLE AS recency_weighted_rows,
                sum(price * exp(-0.35 * date_diff('day', t_dat, date_trunc('week', t_dat - INTERVAL 1 DAY) + INTERVAL 7 DAY)))::DOUBLE AS recency_weighted_spend,
                count(DISTINCT CASE WHEN sales_channel_id = 2 THEN t_dat END)::DOUBLE AS online_active_days,
                count(DISTINCT CASE WHEN sales_channel_id = 1 THEN t_dat END)::DOUBLE AS store_active_days
            FROM purchase_shape_zero
            GROUP BY customer_id, timestamp
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE renewal_zero AS
            SELECT
                c.customer_id,
                c.timestamp,
                date_diff('day', c.last_day_week, c.timestamp)::DOUBLE AS elapsed_since_purchase,
                count(p.gap_days)::DOUBLE AS completed_day_gaps,
                count(CASE WHEN p.gap_days > date_diff('day', c.last_day_week, c.timestamp) THEN 1 END)::DOUBLE AS day_gap_at_risk,
                count(CASE
                    WHEN p.gap_days > date_diff('day', c.last_day_week, c.timestamp)
                    AND p.gap_days <= date_diff('day', c.last_day_week, c.timestamp) + 7
                    THEN 1
                END)::DOUBLE AS day_gap_events_7,
                count(CASE
                    WHEN p.gap_days > date_diff('day', c.last_day_week, c.timestamp)
                    AND p.gap_days <= date_diff('day', c.last_day_week, c.timestamp) + 14
                    THEN 1
                END)::DOUBLE AS day_gap_events_14,
                approx_quantile(p.gap_days, 0.25)::DOUBLE AS gap_q25,
                approx_quantile(p.gap_days, 0.75)::DOUBLE AS gap_q75,
                approx_quantile(p.gap_days, 0.90)::DOUBLE AS gap_q90,
                max(p.gap_days)::DOUBLE AS gap_max,
                count(CASE WHEN p.gap_days <= 7 THEN 1 END)::DOUBLE / greatest(count(p.gap_days), 1) AS gap_share_le_7,
                count(CASE WHEN p.gap_days BETWEEN 8 AND 14 THEN 1 END)::DOUBLE / greatest(count(p.gap_days), 1) AS gap_share_8_14,
                count(CASE WHEN p.gap_days > 28 THEN 1 END)::DOUBLE / greatest(count(p.gap_days), 1) AS gap_share_gt_28
            FROM week_state c
            LEFT JOIN day_one p
                ON c.customer_id = p.customer_id
                AND p.t_dat <= c.last_day_week
            GROUP BY c.customer_id, c.timestamp, c.last_day_week
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE renewal_features AS
            SELECT
                *,
                (day_gap_events_7 + 2.0) / (day_gap_at_risk + 10.0) AS empirical_renewal_hazard_7,
                (day_gap_events_14 + 4.0) / (day_gap_at_risk + 10.0) AS empirical_renewal_hazard_14,
                day_gap_at_risk / greatest(completed_day_gaps, 1) AS empirical_gap_survival,
                day_gap_events_7 / greatest(completed_day_gaps, 1) AS empirical_unconditional_7,
                gap_q75 - gap_q25 AS gap_iqr,
                elapsed_since_purchase / greatest(gap_q75, 1) AS elapsed_over_gap_q75,
                elapsed_since_purchase / greatest(gap_q90, 1) AS elapsed_over_gap_q90
            FROM renewal_zero
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE state_source AS
            SELECT
                w.customer_id,
                w.timestamp,
                w.next_gap_weeks,
                date_diff('day', w.last_day_week, w.timestamp)::BIGINT AS state_recency,
                least(CAST(w.tx_count_week AS BIGINT), 10) AS state_basket,
                least(CAST(w.active_days_week AS BIGINT), 4) AS state_days,
                CASE
                    WHEN w.active_week_sequence = 1 THEN 1
                    WHEN w.active_week_sequence <= 3 THEN 2
                    WHEN w.active_week_sequence <= 7 THEN 3
                    ELSE 4
                END::BIGINT AS state_history,
                least(coalesce(w.gap_weeks, 20), 20)::BIGINT AS state_prior_week_gap,
                least(CAST(w.active_week_streak AS BIGINT), 6) AS state_streak,
                w.dominant_channel::BIGINT AS state_channel,
                p.age_bucket::BIGINT AS state_age,
                p.membership_code::BIGINT AS state_membership,
                x.dominant_product_group::BIGINT AS state_product_group,
                x.dominant_department::BIGINT AS state_department
            FROM week_state w
            INNER JOIN customer_profile p USING(customer_id)
            INNER JOIN wide_week x USING(customer_id, timestamp)
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE state_dimension AS
            SELECT customer_id, timestamp, next_gap_weeks, 1::BIGINT AS dimension_id, state_recency AS category_id FROM state_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 2, state_basket FROM state_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 3, state_days FROM state_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 4, state_history FROM state_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 5, state_prior_week_gap FROM state_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 6, state_streak FROM state_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 7, state_channel FROM state_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 8, state_age FROM state_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 9, state_membership FROM state_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 10, state_recency * 32 + state_basket FROM state_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 11, state_recency * 8 + state_history FROM state_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 12, CAST(hash(state_recency, state_product_group) % 2147483647 AS BIGINT) FROM state_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 13, CAST(hash(state_basket, state_product_group) % 2147483647 AS BIGINT) FROM state_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 14, CAST(hash(state_recency, state_department) % 2147483647 AS BIGINT) FROM state_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 15, CAST(hash(state_age, state_recency) % 2147483647 AS BIGINT) FROM state_source
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE state_cohort_time AS
            SELECT
                dimension_id,
                category_id,
                timestamp AS source_timestamp,
                sum(CASE WHEN next_gap_weeks = 1 THEN 1 ELSE 0 END)::DOUBLE AS active_count,
                count(*)::DOUBLE AS total_count
            FROM state_dimension
            GROUP BY 1,2,3
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE state_cohort_cumulative AS
            SELECT
                *,
                sum(active_count) OVER h AS active_cumulative,
                sum(total_count) OVER h AS total_cumulative
            FROM state_cohort_time
            WINDOW h AS (
                PARTITION BY dimension_id, category_id
                ORDER BY source_timestamp
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )
            """
        )
        state_select = ["b.customer_id", "b.timestamp"]
        for dimension in range(1, 16):
            state_select.append(
                f"max(CASE WHEN b.dimension_id = {dimension} THEN (coalesce(c.active_cumulative, 0) + 5.0) / (coalesce(c.total_cumulative, 0) + 25.0) END) AS state_continuation_prior_{dimension}"
            )
        con.execute(
            """
            CREATE OR REPLACE TABLE state_cohort_features AS
            SELECT
            """
            + ",\n".join(state_select)
            + """
            FROM state_dimension b
            ASOF LEFT JOIN state_cohort_cumulative c
                ON b.dimension_id = c.dimension_id
                AND b.category_id = c.category_id
                AND b.timestamp - INTERVAL 7 DAY >= c.source_timestamp
            GROUP BY b.customer_id, b.timestamp
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE priority_features AS
            SELECT
                s._split,
                s._row_id,
                s.timestamp,
                s.customer_id,
                p.* EXCLUDE(customer_id, timestamp),
                p.last_day_basket_rows / greatest(w.tx_count_week, 1) AS last_day_basket_share,
                p.transaction_rows_offset_0 / greatest(w.tx_count_week, 1) AS transaction_share_offset_0,
                p.transaction_rows_offset_1 / greatest(w.tx_count_week, 1) AS transaction_share_offset_1,
                p.transaction_rows_offset_2 / greatest(w.tx_count_week, 1) AS transaction_share_offset_2,
                p.transaction_rows_offset_3 / greatest(w.tx_count_week, 1) AS transaction_share_offset_3,
                p.transaction_rows_offset_4 / greatest(w.tx_count_week, 1) AS transaction_share_offset_4,
                p.transaction_rows_offset_5 / greatest(w.tx_count_week, 1) AS transaction_share_offset_5,
                p.transaction_rows_offset_6 / greatest(w.tx_count_week, 1) AS transaction_share_offset_6,
                r.* EXCLUDE(customer_id, timestamp),
                h.* EXCLUDE(customer_id, timestamp),
                c.cohort_prior_mean_1,
                c.cohort_prior_mean_2,
                c.cohort_prior_mean_3,
                c.cohort_prior_mean_4,
                c.cohort_prior_mean_5,
                c.cohort_prior_mean_6,
                c.cohort_prior_mean_7,
                c.cohort_prior_mean_8,
                x.distinct_product_groups_7d,
                x.distinct_departments_7d,
                x.product_group_entropy_7d,
                x.department_entropy_7d,
                x.product_group_dominant_share_7d,
                x.department_dominant_share_7d,
                x.repeated_article_share_7d,
                x.repeated_product_share_7d,
                x.novel_group_share_7d,
                x.group_preference_affinity_7d,
                x.department_preference_affinity_7d,
                x.article_price_relative_7d,
                x.low_price_share_7d,
                x.category_price_z_7d,
                x.article_popularity_log_7d,
                x.article_popularity_momentum_7d,
                x.distinct_product_groups_91d,
                x.distinct_departments_91d,
                x.product_group_entropy_91d,
                x.department_entropy_91d,
                x.repeated_article_share_91d,
                x.repeated_product_share_91d,
                x.group_preference_affinity_91d,
                x.department_preference_affinity_91d,
                x.article_price_relative_91d,
                x.low_price_share_91d,
                x.article_popularity_momentum_91d,
                x.distinct_product_groups_364d,
                x.distinct_departments_364d,
                x.product_group_entropy_364d,
                x.department_entropy_364d,
                x.group_preference_affinity_364d,
                x.department_preference_affinity_364d,
                x.product_group_entropy_7d - x.product_group_entropy_364d AS product_group_entropy_targeted_drift,
                x.department_entropy_7d - x.department_entropy_364d AS department_entropy_targeted_drift,
                x.group_preference_affinity_7d - x.group_preference_affinity_364d AS preference_targeted_drift,
                x.article_price_relative_7d - x.article_price_relative_91d AS price_targeted_shift,
                x.article_popularity_momentum_7d - x.article_popularity_momentum_91d AS popularity_targeted_shift
            FROM seeds s
            INNER JOIN week_state w USING(customer_id, timestamp)
            INNER JOIN purchase_shape p USING(customer_id, timestamp)
            INNER JOIN renewal_features r USING(customer_id, timestamp)
            INNER JOIN state_cohort_features h USING(customer_id, timestamp)
            LEFT JOIN cohort_features c USING(customer_id, timestamp)
            INNER JOIN wide_roll x USING(customer_id, timestamp)
            ORDER BY s._split, s._row_id
            """
        )
        con.execute(f"COPY priority_features TO '{sql_path(self.priority_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        elapsed(start, "priority feature build")
        return self.priority_path

    def build_priority_two(self) -> Path:
        if self.priority_two_path.exists():
            print(f"[cache] priority features v2: {self.priority_two_path}", flush=True)
            return self.priority_two_path
        start = time.time()
        con = self.connect()
        con.execute(
            """
            CREATE OR REPLACE TABLE fine_basket_dimension AS
            SELECT customer_id, timestamp, 1::BIGINT AS dimension_id, article_id::BIGINT AS category_id, count(*)::DOUBLE AS weight FROM tx_article_one GROUP BY 1,2,4
            UNION ALL
            SELECT customer_id, timestamp, 2, product_code::BIGINT, count(*)::DOUBLE FROM tx_article_one GROUP BY 1,2,4
            UNION ALL
            SELECT customer_id, timestamp, 3, product_type_no::BIGINT, count(*)::DOUBLE FROM tx_article_one GROUP BY 1,2,4
            UNION ALL
            SELECT customer_id, timestamp, 4, garment_group_no::BIGINT, count(*)::DOUBLE FROM tx_article_one GROUP BY 1,2,4
            UNION ALL
            SELECT customer_id, timestamp, 5, colour_group_code::BIGINT, count(*)::DOUBLE FROM tx_article_one GROUP BY 1,2,4
            UNION ALL
            SELECT customer_id, timestamp, 6, perceived_colour_master_id::BIGINT, count(*)::DOUBLE FROM tx_article_one GROUP BY 1,2,4
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE fine_cohort_time AS
            SELECT
                b.dimension_id,
                b.category_id,
                b.timestamp AS source_timestamp,
                sum(b.weight * CASE WHEN w.next_gap_weeks = 1 THEN 1 ELSE 0 END)::DOUBLE AS active_weight,
                sum(b.weight)::DOUBLE AS total_weight
            FROM fine_basket_dimension b
            INNER JOIN week_state w USING(customer_id, timestamp)
            GROUP BY 1,2,3
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE fine_cohort_cumulative AS
            SELECT
                *,
                sum(active_weight) OVER h AS active_cumulative,
                sum(total_weight) OVER h AS total_cumulative
            FROM fine_cohort_time
            WINDOW h AS (
                PARTITION BY dimension_id, category_id
                ORDER BY source_timestamp
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )
            """
        )
        fine_select = ["b.customer_id", "b.timestamp"]
        for dimension in range(1, 7):
            smoothing = 80 if dimension <= 2 else 30
            alpha = smoothing * 0.2
            prior = f"(coalesce(c.active_cumulative, 0) + {alpha}) / (coalesce(c.total_cumulative, 0) + {smoothing})"
            fine_select.extend(
                [
                    f"sum(CASE WHEN b.dimension_id = {dimension} THEN b.weight * {prior} ELSE 0 END) / greatest(sum(CASE WHEN b.dimension_id = {dimension} THEN b.weight ELSE 0 END), 1) AS fine_prior_mean_{dimension}",
                    f"min(CASE WHEN b.dimension_id = {dimension} THEN {prior} END) AS fine_prior_min_{dimension}",
                    f"max(CASE WHEN b.dimension_id = {dimension} THEN {prior} END) AS fine_prior_max_{dimension}",
                ]
            )
        con.execute(
            """
            CREATE OR REPLACE TABLE fine_cohort_features AS
            SELECT
            """
            + ",\n".join(fine_select)
            + """
            FROM fine_basket_dimension b
            ASOF LEFT JOIN fine_cohort_cumulative c
                ON b.dimension_id = c.dimension_id
                AND b.category_id = c.category_id
                AND b.timestamp - INTERVAL 7 DAY >= c.source_timestamp
            GROUP BY b.customer_id, b.timestamp
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE state_extra_source AS
            SELECT
                s.customer_id,
                s.timestamp,
                s.next_gap_weeks,
                least(CAST(floor(5 * coalesce(w.repeated_article_share_week, 0)) AS BIGINT), 5) AS repeat_bin,
                least(CAST(floor(5 * coalesce(w.repeated_product_share_week, 0)) AS BIGINT), 5) AS repeat_product_bin,
                least(CAST(floor(5 * coalesce(w.novel_group_share_week, 0)) AS BIGINT), 5) AS novelty_bin,
                least(greatest(CAST(floor(4 * coalesce(w.article_price_relative_week, 1)) AS BIGINT), 0), 12) AS price_bin,
                least(greatest(CAST(floor(5 * coalesce(w.low_price_share_week, 0)) AS BIGINT), 0), 5) AS low_price_bin,
                least(greatest(CAST(floor(2 * coalesce(w.article_popularity_log_week, 0)) AS BIGINT), 0), 30) AS popularity_bin,
                least(greatest(CAST(floor(20 * coalesce(w.article_popularity_momentum_week, 0)) AS BIGINT), 0), 20) AS momentum_bin,
                CASE
                    WHEN s.active_week_sequence = 1 THEN 1
                    WHEN s.active_week_sequence <= 3 THEN 2
                    WHEN s.active_week_sequence <= 7 THEN 3
                    ELSE 4
                END::BIGINT AS history_bin,
                date_diff('day', s.last_day_week, s.timestamp)::BIGINT AS recency_bin
            FROM week_state s
            INNER JOIN wide_week w USING(customer_id, timestamp)
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE state_extra_dimension AS
            SELECT customer_id, timestamp, next_gap_weeks, 1::BIGINT AS dimension_id, repeat_bin AS category_id FROM state_extra_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 2, repeat_product_bin FROM state_extra_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 3, novelty_bin FROM state_extra_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 4, price_bin FROM state_extra_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 5, low_price_bin FROM state_extra_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 6, popularity_bin FROM state_extra_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 7, momentum_bin FROM state_extra_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 8, history_bin * 8 + repeat_bin FROM state_extra_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 9, recency_bin * 8 + repeat_bin FROM state_extra_source
            UNION ALL SELECT customer_id, timestamp, next_gap_weeks, 10, history_bin * 32 + popularity_bin FROM state_extra_source
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE state_extra_time AS
            SELECT
                dimension_id,
                category_id,
                timestamp AS source_timestamp,
                sum(CASE WHEN next_gap_weeks = 1 THEN 1 ELSE 0 END)::DOUBLE AS active_count,
                count(*)::DOUBLE AS total_count
            FROM state_extra_dimension
            GROUP BY 1,2,3
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE state_extra_cumulative AS
            SELECT
                *,
                sum(active_count) OVER h AS active_cumulative,
                sum(total_count) OVER h AS total_cumulative
            FROM state_extra_time
            WINDOW h AS (
                PARTITION BY dimension_id, category_id
                ORDER BY source_timestamp
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )
            """
        )
        extra_state_select = ["b.customer_id", "b.timestamp"]
        for dimension in range(1, 11):
            extra_state_select.append(
                f"max(CASE WHEN b.dimension_id = {dimension} THEN (coalesce(c.active_cumulative, 0) + 5.0) / (coalesce(c.total_cumulative, 0) + 25.0) END) AS state_extra_prior_{dimension}"
            )
        con.execute(
            """
            CREATE OR REPLACE TABLE state_extra_features AS
            SELECT
            """
            + ",\n".join(extra_state_select)
            + """
            FROM state_extra_dimension b
            ASOF LEFT JOIN state_extra_cumulative c
                ON b.dimension_id = c.dimension_id
                AND b.category_id = c.category_id
                AND b.timestamp - INTERVAL 7 DAY >= c.source_timestamp
            GROUP BY b.customer_id, b.timestamp
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE article_degree_week AS
            SELECT
                customer_id,
                timestamp,
                min(ln(1 + article_popularity_cumulative)) AS article_popularity_log_min,
                max(ln(1 + article_popularity_cumulative)) AS article_popularity_log_max,
                stddev_samp(ln(1 + article_popularity_cumulative)) AS article_popularity_log_std,
                approx_quantile(ln(1 + article_popularity_cumulative), 0.25)::DOUBLE AS article_popularity_log_q25,
                approx_quantile(ln(1 + article_popularity_cumulative), 0.75)::DOUBLE AS article_popularity_log_q75,
                avg(CASE WHEN article_popularity_cumulative < 20 THEN 1.0 ELSE 0.0 END) AS rare_article_share,
                avg(CASE WHEN article_popularity_cumulative >= 500 THEN 1.0 ELSE 0.0 END) AS popular_article_share,
                min(article_popularity_28d / greatest(article_popularity_cumulative, 1)) AS popularity_momentum_min,
                max(article_popularity_28d / greatest(article_popularity_cumulative, 1)) AS popularity_momentum_max,
                stddev_samp(article_popularity_28d / greatest(article_popularity_cumulative, 1)) AS popularity_momentum_std
            FROM tx_article_one
            GROUP BY customer_id, timestamp
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE priority_features_two AS
            SELECT
                s._split,
                s._row_id,
                s.timestamp,
                s.customer_id,
                f.* EXCLUDE(customer_id, timestamp),
                e.* EXCLUDE(customer_id, timestamp),
                d.* EXCLUDE(customer_id, timestamp)
            FROM seeds s
            INNER JOIN fine_cohort_features f USING(customer_id, timestamp)
            INNER JOIN state_extra_features e USING(customer_id, timestamp)
            INNER JOIN article_degree_week d USING(customer_id, timestamp)
            ORDER BY s._split, s._row_id
            """
        )
        con.execute(
            f"COPY priority_features_two TO '{sql_path(self.priority_two_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        elapsed(start, "priority feature v2 build")
        return self.priority_two_path

    def close(self) -> None:
        if self.con is not None:
            self.con.close()
            self.con = None

    def metadata(self) -> dict[str, object]:
        return {
            "name": "lane0 causal hazard features",
            "path": str(self.root.relative_to(self.shared_root)),
            "description": "Point-in-time customer-week core and widened all-table feature matrices with training-only selection artifacts",
            "content_key": f"causal-hazard-lane0-v6-priority-v2-{self.key}",
            "rebuild_hint": "Run main.py with the same sanitized RelBench cache and seed tables",
        }


def load_feature_frames(core_path: Path, wide_path: Path | list[Path] | tuple[Path, ...] | None = None) -> pd.DataFrame:
    core = pd.read_parquet(core_path)
    if wide_path is None:
        return core
    keys = ["_split", "_row_id", "timestamp", "customer_id"]
    paths = list(wide_path) if isinstance(wide_path, (list, tuple)) else [wide_path]
    result = core
    for path in paths:
        extension = pd.read_parquet(path)
        result = result.merge(extension, on=keys, how="left", validate="one_to_one", sort=False)
    return result


def register_artifact(shared_root: Path, metadata: dict[str, object]) -> None:
    import fcntl

    registry = shared_root / "artifacts.json"
    lock_path = shared_root / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if registry.exists():
            try:
                values = json.loads(registry.read_text())
            except json.JSONDecodeError:
                values = []
        else:
            values = []
        if not any(value.get("content_key") == metadata["content_key"] for value in values):
            values.append(metadata)
            temporary = registry.with_suffix(f".lane0.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(values, indent=2))
            temporary.replace(registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
