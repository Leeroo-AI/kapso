import gc
import json
import math
import os
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


# Configuration

PANEL_CATEGORICAL = [
    "sales_channel_id",
    "product_code",
    "product_type_no",
    "graphical_appearance_no",
    "colour_group_code",
    "perceived_colour_value_id",
    "perceived_colour_master_id",
    "department_no",
    "index_group_no",
    "section_no",
    "garment_group_no",
    "index_code_id",
]

META_COLUMNS = [
    "product_code",
    "product_type_no",
    "graphical_appearance_no",
    "colour_group_code",
    "perceived_colour_value_id",
    "perceived_colour_master_id",
    "department_no",
    "index_group_no",
    "section_no",
    "garment_group_no",
    "index_code_id",
]

HIERARCHIES = [
    "product_type_no",
    "department_no",
    "section_no",
    "garment_group_no",
    "index_group_no",
]

PANEL_FEATURES = [
    "last_mean",
    "last_mode",
    "last_min",
    "last_max",
    "last_std",
    "last_count",
    "weeks_since_observation",
    "price_lag1",
    "price_lag2",
    "price_lag3",
    "price_lag4",
    "mode_lag1",
    "mode_lag2",
    "mode_lag3",
    "mode_lag4",
    "count_lag1",
    "count_lag2",
    "count_lag3",
    "count_lag4",
    "std_lag1",
    "std_lag2",
    "std_lag3",
    "std_lag4",
    "current_max_ratio",
    "regime_duration",
    "distinct_state_count",
    "velocity_trend",
    "article_age_weeks",
    "launch_week_price",
    "launch_support",
    "article_recent_state",
    "article_recent_support",
    "other_channel_price",
    "other_channel_support",
    "cross_channel_spread",
    "product_state",
    "product_support",
    "product_channel_state",
    "product_channel_support",
    "product_type_no_state",
    "product_type_no_support",
    "department_no_state",
    "department_no_support",
    "section_no_state",
    "section_no_support",
    "garment_group_no_state",
    "garment_group_no_support",
    "index_group_no_state",
    "index_group_no_support",
    "baseline_state",
    "has_article_history",
    "week_of_year",
    "horizon_mean",
    "horizon_min",
    "horizon_max",
    "weekday_mean",
    "target_count_log",
    "origin_day_panel_count_log",
    "origin_day_article_count_log",
    "origin_day_channel_share",
    "metadata_article_age_weeks",
    "metadata_total_activity_log",
] + META_COLUMNS + ["sales_channel_id"]

LAUNCH_PRIOR_FEATURES = [
    "product_state",
    "product_support",
    "product_channel_state",
    "product_channel_support",
    "product_type_no_state",
    "product_type_no_support",
    "department_no_state",
    "department_no_support",
    "section_no_state",
    "section_no_support",
    "garment_group_no_state",
    "garment_group_no_support",
    "index_group_no_state",
    "index_group_no_support",
]

LAUNCH_HISTORY_FEATURES = [
    "product_launch_state",
    "product_launch_support",
    "product_channel_launch_state",
    "product_channel_launch_support",
    "product_type_no_launch_state",
    "product_type_no_launch_support",
    "department_no_launch_state",
    "department_no_launch_support",
    "section_no_launch_state",
    "section_no_launch_support",
    "garment_group_no_launch_state",
    "garment_group_no_launch_support",
    "index_group_no_launch_state",
    "index_group_no_launch_support",
]


# Data access

class DataAccess:
    def __init__(self, label_splits, threads):
        self.label_splits = tuple(label_splits)
        self.cache_root = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"]
        self.db_root = self.cache_root / "db"
        self.task_root = self.cache_root / "tasks" / os.environ["RELBENCH_TASK"]
        self.connection = duckdb.connect()
        self.connection.execute(f"SET threads={int(threads)}")
        self.connection.execute("SET preserve_insertion_order=false")
        self.connection.execute("SET enable_progress_bar=false")
        transaction_path = self.db_root / "transactions.parquet"
        self.connection.execute(
            f"CREATE OR REPLACE TEMP VIEW transaction_inputs AS "
            f"SELECT file_row_number::BIGINT AS primary_key, t_dat, customer_id, "
            f"article_id, sales_channel_id FROM read_parquet('{transaction_path}', file_row_number=true)"
        )
        parts = []
        for split in self.label_splits:
            path = self.task_root / f"{split}.parquet"
            parts.append(f"SELECT t_dat, primary_key, price FROM read_parquet('{path}')")
        label_union = " UNION ALL ".join(parts)
        self.connection.execute(f"CREATE OR REPLACE TEMP VIEW task_labels AS {label_union}")
        self.connection.execute(
            "CREATE OR REPLACE TEMP VIEW legal_labeled_rows AS "
            "SELECT l.t_dat, l.primary_key, x.customer_id, x.article_id, "
            "x.sales_channel_id, l.price FROM task_labels l "
            "JOIN transaction_inputs x USING(primary_key)"
        )

    def label_range(self):
        row = self.connection.execute(
            "SELECT min(price), max(price), max(t_dat), count(*) FROM legal_labeled_rows"
        ).fetchone()
        return float(row[0]), float(row[1]), pd.Timestamp(row[2]), int(row[3])

    def weekly_panel(self, cache_path):
        expected = {
            "article_id",
            "sales_channel_id",
            "week_start",
            "price_mean",
            "price_mode",
            "price_min",
            "price_max",
            "price_std",
            "transaction_count",
        }
        if cache_path.exists():
            frame = pd.read_parquet(cache_path)
            if expected.issubset(frame.columns):
                return frame
        temporary = cache_path.with_suffix(".building.parquet")
        if temporary.exists():
            temporary.unlink()
        query = (
            "WITH base AS ("
            "SELECT article_id, sales_channel_id, date_trunc('week', t_dat)::DATE AS week_start, "
            "price FROM legal_labeled_rows), "
            "stats AS (SELECT article_id, sales_channel_id, week_start, avg(price) AS price_mean, "
            "min(price) AS price_min, max(price) AS price_max, coalesce(stddev_pop(price), 0) AS price_std, "
            "count(*)::INTEGER AS transaction_count FROM base GROUP BY ALL), "
            "frequencies AS (SELECT article_id, sales_channel_id, week_start, price, count(*) AS frequency "
            "FROM base GROUP BY ALL), "
            "modes AS (SELECT article_id, sales_channel_id, week_start, arg_max(price, frequency) AS price_mode "
            "FROM frequencies GROUP BY article_id, sales_channel_id, week_start) "
            "SELECT s.article_id::INTEGER AS article_id, s.sales_channel_id::TINYINT AS sales_channel_id, "
            "s.week_start, s.price_mean::FLOAT AS price_mean, m.price_mode::FLOAT AS price_mode, "
            "s.price_min::FLOAT AS price_min, s.price_max::FLOAT AS price_max, "
            "s.price_std::FLOAT AS price_std, s.transaction_count "
            "FROM stats s JOIN modes m USING(article_id, sales_channel_id, week_start)"
        )
        self.connection.execute(
            f"COPY ({query}) TO '{temporary}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        os.replace(temporary, cache_path)
        return pd.read_parquet(cache_path)

    def target_panels(self, origin, available_end):
        origin_text = pd.Timestamp(origin).strftime("%Y-%m-%d")
        end = min(pd.Timestamp(origin) + pd.Timedelta(days=8), pd.Timestamp(available_end))
        end_text = end.strftime("%Y-%m-%d")
        query = f"""
            SELECT article_id::INTEGER AS article_id,
                   sales_channel_id::TINYINT AS sales_channel_id,
                   avg(price)::FLOAT AS target_mean,
                   min(price)::FLOAT AS target_min,
                   max(price)::FLOAT AS target_max,
                   coalesce(stddev_pop(price), 0)::FLOAT AS target_std,
                   count(*)::INTEGER AS target_count,
                   avg(date_diff('day', DATE '{origin_text}', t_dat))::FLOAT AS horizon_mean,
                   min(date_diff('day', DATE '{origin_text}', t_dat))::FLOAT AS horizon_min,
                   max(date_diff('day', DATE '{origin_text}', t_dat))::FLOAT AS horizon_max,
                   avg(dayofweek(t_dat))::FLOAT AS weekday_mean
            FROM legal_labeled_rows
            WHERE t_dat > DATE '{origin_text}' AND t_dat <= DATE '{end_text}'
            GROUP BY article_id, sales_channel_id
        """
        return self.connection.execute(query).fetchdf()

    def target_rows(self, origin, available_end):
        origin_text = pd.Timestamp(origin).strftime("%Y-%m-%d")
        end = min(pd.Timestamp(origin) + pd.Timedelta(days=8), pd.Timestamp(available_end))
        end_text = end.strftime("%Y-%m-%d")
        query = f"""
            SELECT t_dat, primary_key, customer_id::INTEGER AS customer_id,
                   article_id::INTEGER AS article_id,
                   sales_channel_id::TINYINT AS sales_channel_id,
                   price::FLOAT AS price,
                   date_diff('day', DATE '{origin_text}', t_dat)::SMALLINT AS horizon,
                   dayofweek(t_dat)::TINYINT AS weekday
            FROM legal_labeled_rows
            WHERE t_dat > DATE '{origin_text}' AND t_dat <= DATE '{end_text}'
        """
        return self.connection.execute(query).fetchdf()

    def customer_history(self, origin):
        origin_text = pd.Timestamp(origin).strftime("%Y-%m-%d")
        start_text = (pd.Timestamp(origin) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
        recent_text = (pd.Timestamp(origin) - pd.Timedelta(days=35)).strftime("%Y-%m-%d")
        query = f"""
            WITH history AS (
                SELECT t_dat::DATE AS day, customer_id, article_id, sales_channel_id, price
                FROM legal_labeled_rows
                WHERE t_dat < DATE '{origin_text}' AND t_dat >= DATE '{start_text}'
            ), article_day AS (
                SELECT article_id, day, avg(price) AS reference_price
                FROM history GROUP BY article_id, day
            )
            SELECT h.customer_id::INTEGER AS customer_id,
                   avg(h.price / nullif(a.reference_price, 0))::FLOAT AS customer_affinity,
                   coalesce(stddev_pop(h.price / nullif(a.reference_price, 0)), 0)::FLOAT AS customer_affinity_std,
                   count(*)::INTEGER AS customer_activity,
                   avg(h.price)::FLOAT AS customer_mean_price,
                   coalesce(stddev_pop(h.price), 0)::FLOAT AS customer_price_std,
                   avg(h.price / nullif(a.reference_price, 0)) FILTER (
                       WHERE h.day >= DATE '{recent_text}'
                   )::FLOAT AS customer_recent_affinity,
                   count(*) FILTER (WHERE h.day >= DATE '{recent_text}')::INTEGER AS customer_recent_activity,
                   avg(h.price / nullif(a.reference_price, 0)) FILTER (
                       WHERE h.sales_channel_id = 1
                   )::FLOAT AS customer_channel1_affinity,
                   count(*) FILTER (WHERE h.sales_channel_id = 1)::INTEGER AS customer_channel1_activity,
                   avg(h.price / nullif(a.reference_price, 0)) FILTER (
                       WHERE h.sales_channel_id = 2
                   )::FLOAT AS customer_channel2_affinity,
                   count(*) FILTER (WHERE h.sales_channel_id = 2)::INTEGER AS customer_channel2_activity,
                   avg((h.sales_channel_id = 2)::INTEGER)::FLOAT AS customer_channel2_share
            FROM history h JOIN article_day a USING(article_id, day)
            GROUP BY h.customer_id
        """
        return self.connection.execute(query).fetchdf()

    def article_day_dispersion(self, origin):
        origin_text = pd.Timestamp(origin).strftime("%Y-%m-%d")
        start_text = (pd.Timestamp(origin) - pd.Timedelta(days=35)).strftime("%Y-%m-%d")
        query = f"""
            WITH daily AS (
                SELECT article_id, t_dat::DATE AS day,
                       coalesce(stddev_pop(price), 0) AS daily_std,
                       max(price) - min(price) AS daily_range,
                       count(DISTINCT round(price, 5)) AS daily_states,
                       count(*) AS daily_count
                FROM legal_labeled_rows
                WHERE t_dat < DATE '{origin_text}' AND t_dat >= DATE '{start_text}'
                GROUP BY article_id, day
            )
            SELECT article_id::INTEGER AS article_id,
                   avg(daily_std)::FLOAT AS article_day_std,
                   avg(daily_range)::FLOAT AS article_day_range,
                   avg(daily_states)::FLOAT AS article_day_states,
                   sum(daily_count)::INTEGER AS article_recent_activity
            FROM daily GROUP BY article_id
        """
        return self.connection.execute(query).fetchdf()

    def launch_examples(self, inference_origin, weeks):
        cutoff_text = pd.Timestamp(inference_origin).strftime("%Y-%m-%d")
        start_text = (pd.Timestamp(inference_origin) - pd.Timedelta(weeks=weeks)).strftime("%Y-%m-%d")
        query = f"""
            WITH first_seen AS (
                SELECT article_id, min(t_dat)::DATE AS first_date
                FROM legal_labeled_rows GROUP BY article_id
            ), launches AS (
                SELECT r.article_id, r.sales_channel_id, f.first_date,
                       date_trunc('week', f.first_date)::DATE AS launch_origin,
                       avg(r.price)::FLOAT AS target_mean,
                       count(*)::INTEGER AS target_count
                FROM legal_labeled_rows r JOIN first_seen f USING(article_id)
                WHERE f.first_date >= DATE '{start_text}' AND f.first_date < DATE '{cutoff_text}'
                  AND r.t_dat >= f.first_date AND r.t_dat <= f.first_date + INTERVAL 6 DAY
                GROUP BY r.article_id, r.sales_channel_id, f.first_date
            )
            SELECT article_id::INTEGER AS article_id,
                   sales_channel_id::TINYINT AS sales_channel_id,
                   first_date, launch_origin, target_mean, target_count
            FROM launches
        """
        return self.connection.execute(query).fetchdf()

    def snapshot_activity(self, origin):
        origin_text = pd.Timestamp(origin).strftime("%Y-%m-%d")
        panel_query = f"""
            SELECT article_id::INTEGER AS article_id,
                   sales_channel_id::TINYINT AS sales_channel_id,
                   count(*)::INTEGER AS origin_day_panel_count
            FROM transaction_inputs
            WHERE t_dat::DATE = DATE '{origin_text}'
            GROUP BY article_id, sales_channel_id
        """
        article_query = f"""
            SELECT article_id::INTEGER AS article_id,
                   min(t_dat)::DATE AS metadata_first_seen,
                   count(*)::INTEGER AS metadata_total_activity,
                   count(*) FILTER (WHERE t_dat::DATE = DATE '{origin_text}')::INTEGER
                       AS origin_day_article_count
            FROM transaction_inputs
            WHERE t_dat::DATE <= DATE '{origin_text}'
            GROUP BY article_id
        """
        return (
            self.connection.execute(panel_query).fetchdf(),
            self.connection.execute(article_query).fetchdf(),
        )

    def close(self):
        self.connection.close()


# Feature construction

class FeatureBuilder:
    def __init__(self, weekly, article):
        self.weekly = weekly.copy()
        self.weekly["week_start"] = pd.to_datetime(self.weekly["week_start"])
        self.article = article.copy()
        self.meta = self.article[["article_id"] + META_COLUMNS].copy()
        self.weekly_meta = self.weekly.merge(self.meta, on="article_id", how="left")

    @staticmethod
    def weighted_state(frame, keys, prefix):
        if len(frame) == 0:
            return pd.DataFrame(columns=keys + [f"{prefix}_state", f"{prefix}_support"])
        data = frame[keys + ["price_mean", "transaction_count"]].copy()
        data["weighted_price"] = data["price_mean"] * data["transaction_count"]
        grouped = data.groupby(keys, observed=True, sort=False).agg(
            weighted_price=("weighted_price", "sum"),
            support=("transaction_count", "sum"),
        ).reset_index()
        grouped[f"{prefix}_state"] = grouped["weighted_price"] / grouped["support"].clip(lower=1)
        grouped[f"{prefix}_support"] = grouped["support"]
        return grouped[keys + [f"{prefix}_state", f"{prefix}_support"]]

    def attach_recent_priors(self, base, origin, weeks=4):
        origin = pd.Timestamp(origin)
        recent = self.weekly_meta[
            (self.weekly_meta["week_start"] < origin)
            & (self.weekly_meta["week_start"] >= origin - pd.Timedelta(weeks=weeks))
        ]
        output = base.copy()
        product = self.weighted_state(recent, ["product_code"], "product")
        output = output.merge(product, on="product_code", how="left")
        product_channel = self.weighted_state(
            recent, ["product_code", "sales_channel_id"], "product_channel"
        )
        output = output.merge(
            product_channel, on=["product_code", "sales_channel_id"], how="left"
        )
        for hierarchy in HIERARCHIES:
            prior = self.weighted_state(recent, [hierarchy], hierarchy)
            output = output.merge(prior, on=hierarchy, how="left")
        global_state = np.average(
            recent["price_mean"], weights=recent["transaction_count"]
        ) if len(recent) else 0.03
        output["global_recent_state"] = float(global_state)
        return output

    def panel_features(self, panels, origin):
        origin = pd.Timestamp(origin)
        output = panels.copy()
        output = output.merge(self.meta, on="article_id", how="left")
        history = self.weekly[self.weekly["week_start"] < origin]
        history_26 = history[history["week_start"] >= origin - pd.Timedelta(weeks=26)]
        keys = ["article_id", "sales_channel_id"]

        last = history.sort_values("week_start").drop_duplicates(keys, keep="last")
        last = last[keys + [
            "week_start", "price_mean", "price_mode", "price_min", "price_max",
            "price_std", "transaction_count",
        ]].rename(columns={
            "week_start": "last_week",
            "price_mean": "last_mean",
            "price_mode": "last_mode",
            "price_min": "last_min",
            "price_max": "last_max",
            "price_std": "last_std",
            "transaction_count": "last_count",
        })
        output = output.merge(last, on=keys, how="left")
        output["weeks_since_observation"] = (
            (origin - output["last_week"]).dt.days / 7.0
        )

        for lag in range(1, 5):
            lag_week = origin - pd.Timedelta(weeks=lag)
            frame = history[history["week_start"] == lag_week][
                keys + ["price_mean", "price_mode", "transaction_count", "price_std"]
            ].rename(columns={
                "price_mean": f"price_lag{lag}",
                "price_mode": f"mode_lag{lag}",
                "transaction_count": f"count_lag{lag}",
                "price_std": f"std_lag{lag}",
            })
            output = output.merge(frame, on=keys, how="left")

        panel_history = history_26.groupby(keys, observed=True, sort=False).agg(
            history_max=("price_mean", "max"),
            distinct_state_count=("price_mean", lambda s: s.round(4).nunique()),
        ).reset_index()
        output = output.merge(panel_history, on=keys, how="left")
        output["current_max_ratio"] = output["last_mean"] / output["history_max"].replace(0, np.nan)
        output["velocity_trend"] = (output["price_lag1"] - output["price_lag4"]) / 3.0

        base_lag = output["price_lag1"].fillna(output["last_mean"])
        duration = np.ones(len(output), dtype=np.float32)
        active = base_lag.notna().to_numpy()
        base_values = base_lag.to_numpy(dtype=float)
        for lag in range(2, 5):
            values = output[f"price_lag{lag}"].to_numpy(dtype=float)
            threshold = np.maximum(0.001, 0.03 * np.abs(base_values))
            same = np.isfinite(values) & np.isfinite(base_values) & (np.abs(values - base_values) <= threshold)
            active &= same
            duration += active.astype(np.float32)
        output["regime_duration"] = duration

        article_weeks = history.groupby("article_id", observed=True, sort=False)["week_start"].agg(
            first_week="min", latest_week="max"
        ).reset_index()
        output = output.merge(article_weeks, on="article_id", how="left")
        output["article_age_weeks"] = (origin - output["first_week"]).dt.days / 7.0

        first_pairs = article_weeks[["article_id", "first_week"]].merge(
            history, left_on=["article_id", "first_week"], right_on=["article_id", "week_start"], how="left"
        )
        launch = self.weighted_state(first_pairs, ["article_id"], "launch")
        launch = launch.rename(columns={"launch_state": "launch_week_price"})
        output = output.merge(launch, on="article_id", how="left")

        latest_pairs = article_weeks[["article_id", "latest_week"]].merge(
            history, left_on=["article_id", "latest_week"], right_on=["article_id", "week_start"], how="left"
        )
        article_recent = self.weighted_state(latest_pairs, ["article_id"], "article_recent")
        output = output.merge(article_recent, on="article_id", how="left")

        other = last[["article_id", "sales_channel_id", "last_mean", "last_count"]].copy()
        other["sales_channel_id"] = 3 - other["sales_channel_id"]
        other = other.rename(columns={
            "last_mean": "other_channel_price",
            "last_count": "other_channel_support",
        })
        output = output.merge(other, on=keys, how="left")
        output["cross_channel_spread"] = output["last_mean"] - output["other_channel_price"]

        output = self.attach_recent_priors(output, origin, weeks=4)
        fallback_order = [
            "last_mean",
            "article_recent_state",
            "product_channel_state",
            "product_state",
            "product_type_no_state",
            "department_no_state",
            "section_no_state",
            "garment_group_no_state",
            "index_group_no_state",
            "global_recent_state",
        ]
        state = pd.Series(np.nan, index=output.index, dtype=float)
        for column in fallback_order:
            state = state.fillna(output[column])
        output["baseline_state"] = state
        output["has_article_history"] = output["article_age_weeks"].notna().astype(np.int8)
        output["week_of_year"] = int(origin.isocalendar().week)
        output["target_count_log"] = np.log1p(output["target_count"].astype(float))
        for column in PANEL_FEATURES:
            if column not in output:
                output[column] = np.nan
        return output


# Models

class ConstantClassifier:
    def __init__(self, value):
        self.value = float(value)

    def predict_proba(self, frame):
        positive = np.full(len(frame), self.value, dtype=float)
        return np.column_stack([1.0 - positive, positive])


class ConstantRegressor:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


def numeric_frame(frame, columns):
    output = frame[columns].copy()
    for column in columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def weighted_mse(y, prediction, weight):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    weight = np.asarray(weight, dtype=float)
    return float(np.average((y - prediction) ** 2, weights=weight))


def fit_panel_heads(frame, debug, seed):
    trees_regression = 100 if debug else 1000
    trees_classifier = 100 if debug else 600
    features = numeric_frame(frame, PANEL_FEATURES)
    warm = frame["has_article_history"].to_numpy() > 0
    last = frame["baseline_state"].to_numpy(dtype=float)
    target = frame["target_mean"].to_numpy(dtype=float)
    threshold = np.maximum(0.001, 0.03 * np.abs(last))
    changed = warm & (np.abs(target - last) > threshold)
    weights = frame["target_count"].to_numpy(dtype=float)
    if changed.sum() >= 20:
        regressor = lgb.LGBMRegressor(
            objective="regression_l2",
            n_estimators=trees_regression,
            learning_rate=0.05,
            num_leaves=127,
            min_child_samples=200,
            colsample_bytree=0.85,
            reg_lambda=10.0,
            random_state=seed,
            n_jobs=int(os.environ.get("OMP_NUM_THREADS", "11")),
            verbosity=-1,
        )
        regressor.fit(features.loc[changed], target[changed], sample_weight=weights[changed])
    else:
        regressor = ConstantRegressor(np.average(target, weights=weights))
    labels = changed[warm].astype(np.int8)
    if len(labels) and np.unique(labels).size == 2:
        classifier = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=trees_classifier,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=200,
            colsample_bytree=0.85,
            reg_lambda=10.0,
            random_state=seed + 1,
            n_jobs=int(os.environ.get("OMP_NUM_THREADS", "11")),
            verbosity=-1,
        )
        classifier.fit(features.loc[warm], labels, sample_weight=weights[warm])
    else:
        classifier = ConstantClassifier(float(labels.mean()) if len(labels) else 0.0)
    return regressor, classifier


def predict_panel_heads(frame, regressor, classifier):
    features = numeric_frame(frame, PANEL_FEATURES)
    changed_state = np.asarray(regressor.predict(features), dtype=float)
    probability = np.asarray(classifier.predict_proba(features)[:, 1], dtype=float)
    last_state = frame["baseline_state"].to_numpy(dtype=float)
    decoded = probability * changed_state + (1.0 - probability) * last_state
    decoded = np.where(frame["has_article_history"].to_numpy() > 0, decoded, last_state)
    uncertainty = probability * (1.0 - probability) * (changed_state - last_state) ** 2
    return decoded, probability, changed_state, uncertainty


def affine_calibration(frame, prediction_column):
    x = frame[prediction_column].to_numpy(dtype=float)
    y = frame["target_mean"].to_numpy(dtype=float)
    w = frame["target_count"].to_numpy(dtype=float)
    x_mean = np.average(x, weights=w)
    y_mean = np.average(y, weights=w)
    denominator = np.average((x - x_mean) ** 2, weights=w)
    slope = np.average((x - x_mean) * (y - y_mean), weights=w) / max(denominator, 1e-12)
    slope = float(np.clip(slope, 0.5, 1.5))
    intercept = float(np.clip(y_mean - slope * x_mean, -0.01, 0.01))
    return slope, intercept


# Pipeline

class PricePanelPipeline:
    def __init__(self, debug=False):
        self.debug = bool(debug)
        self.threads = int(os.environ.get("OMP_NUM_THREADS", "11"))
        self.shared = Path(os.environ.get("KAPSO_SHARED_CACHE_DIR", "output_data_generic_exp_2"))
        self.shared.mkdir(parents=True, exist_ok=True)
        self.output = Path("output_data_generic_exp_2")
        self.output.mkdir(parents=True, exist_ok=True)
        self.cache_root = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"]
        self.task_root = self.cache_root / "tasks" / os.environ["RELBENCH_TASK"]
        self.db_root = self.cache_root / "db"
        self.article = pd.read_parquet(self.db_root / "article.parquet")
        self.article["index_code_id"] = pd.factorize(self.article["index_code"], sort=True)[0].astype(np.int16)
        self.customer = pd.read_parquet(self.db_root / "customer.parquet")
        self.customer["club_status_id"] = pd.factorize(self.customer["club_member_status"], sort=True)[0].astype(np.int8)
        self.customer["news_frequency_id"] = pd.factorize(self.customer["fashion_news_frequency"], sort=True)[0].astype(np.int8)
        self.customer = self.customer[[
            "customer_id", "FN", "Active", "age", "club_status_id", "news_frequency_id"
        ]]
        self.text = self.load_text_features()

    def register_artifact(self, name, path, description, content_key, rebuild_hint):
        registry = self.shared / "artifacts.json"
        lock_path = self.shared / "artifacts.lock"
        import fcntl
        with open(lock_path, "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if registry.exists():
                try:
                    values = json.loads(registry.read_text())
                except Exception:
                    values = []
            else:
                values = []
            relative = os.path.relpath(path, self.shared)
            entry = {
                "name": name,
                "path": relative,
                "description": description,
                "content_key": content_key,
                "rebuild_hint": rebuild_hint,
            }
            values = [item for item in values if item.get("name") != name]
            values.append(entry)
            temporary = registry.with_suffix(".lane2.tmp")
            temporary.write_text(json.dumps(values, indent=2))
            os.replace(temporary, registry)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def load_text_features(self):
        path = self.shared / "lane2_article_text_wordchar_svd32_v1.npy"
        if path.exists():
            matrix = np.load(path, allow_pickle=False)
            if matrix.shape == (len(self.article), 32):
                return matrix
        name = self.article["prod_name"].fillna("").astype(str)
        description = self.article["detail_desc"].fillna("").astype(str)
        text = (name + " " + description).tolist()
        word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_features=12000,
            sublinear_tf=True, dtype=np.float32,
        ).fit_transform(text)
        character = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=20000,
            sublinear_tf=True, dtype=np.float32,
        ).fit_transform(text)
        joined = sparse.hstack([word, character], format="csr", dtype=np.float32)
        matrix = TruncatedSVD(n_components=32, n_iter=5, random_state=1337).fit_transform(joined).astype(np.float32)
        temporary = path.with_suffix(".building.npy")
        np.save(temporary, matrix)
        os.replace(temporary, path)
        self.register_artifact(
            "lane2_article_text_wordchar_svd32_v1",
            path,
            "Static H&M article word and character TF-IDF projected to 32 SVD dimensions",
            "relhm_article_static_wordchar_svd32_v1",
            "Rebuild from sanitized article prod_name and detail_desc with sklearn TF-IDF and TruncatedSVD",
        )
        return matrix

    def load_inference_rows(self, split):
        path = self.task_root / f"{split}.parquet"
        connection = duckdb.connect()
        connection.execute(f"SET threads={self.threads}")
        transaction_path = self.db_root / "transactions.parquet"
        query = f"""
            WITH seeds AS (
                SELECT file_row_number::BIGINT AS row_index, t_dat, primary_key
                FROM read_parquet('{path}', file_row_number=true)
            ), inputs AS (
                SELECT file_row_number::BIGINT AS primary_key, customer_id, article_id, sales_channel_id
                FROM read_parquet('{transaction_path}', file_row_number=true)
            )
            SELECT s.row_index, s.t_dat, s.primary_key,
                   x.customer_id::INTEGER AS customer_id,
                   x.article_id::INTEGER AS article_id,
                   x.sales_channel_id::TINYINT AS sales_channel_id
            FROM seeds s JOIN inputs x USING(primary_key)
            ORDER BY s.row_index
        """
        frame = connection.execute(query).fetchdf()
        connection.close()
        return frame

    @staticmethod
    def inference_panels(rows, origin):
        data = rows.copy()
        data["horizon"] = (pd.to_datetime(data["t_dat"]) - pd.Timestamp(origin)).dt.days
        data["weekday"] = pd.to_datetime(data["t_dat"]).dt.dayofweek + 1
        return data.groupby(["article_id", "sales_channel_id"], observed=True, sort=False).agg(
            target_count=("primary_key", "size"),
            horizon_mean=("horizon", "mean"),
            horizon_min=("horizon", "min"),
            horizon_max=("horizon", "max"),
            weekday_mean=("weekday", "mean"),
        ).reset_index()

    def build_examples(self, access, builder, origins, available_end):
        examples = []
        for index, origin in enumerate(origins):
            targets = access.target_panels(origin, available_end)
            if len(targets) == 0:
                continue
            targets = self.attach_snapshot_activity(targets, access, origin)
            frame = builder.panel_features(targets, origin)
            frame["origin"] = pd.Timestamp(origin)
            frame["origin_index"] = index
            examples.append(frame)
        if not examples:
            raise RuntimeError("no supervised panel examples were constructed")
        return pd.concat(examples, ignore_index=True)

    @staticmethod
    def attach_snapshot_activity(frame, access, origin):
        panel, article = access.snapshot_activity(origin)
        output = frame.merge(
            panel, on=["article_id", "sales_channel_id"], how="left"
        ).merge(article, on="article_id", how="left")
        output["origin_day_panel_count"] = output["origin_day_panel_count"].fillna(0)
        output["origin_day_article_count"] = output["origin_day_article_count"].fillna(0)
        output["origin_day_panel_count_log"] = np.log1p(output["origin_day_panel_count"])
        output["origin_day_article_count_log"] = np.log1p(output["origin_day_article_count"])
        output["origin_day_channel_share"] = (
            output["origin_day_panel_count"]
            / output["origin_day_article_count"].replace(0, np.nan)
        )
        output["metadata_article_age_weeks"] = (
            pd.Timestamp(origin) - pd.to_datetime(output["metadata_first_seen"])
        ).dt.days / 7.0
        output["metadata_total_activity_log"] = np.log1p(
            output["metadata_total_activity"].fillna(0)
        )
        return output

    def forward_oof(self, examples, origin_count):
        if self.debug:
            folds = [([0, 1], [3])]
        else:
            folds = [
                (list(range(0, 5)), [6, 7]),
                (list(range(0, 7)), [8, 9]),
                (list(range(0, 9)), [10, 11]),
            ]
        outputs = []
        fold_metrics = []
        for fold_index, (train_indices, valid_indices) in enumerate(folds):
            train = examples[examples["origin_index"].isin(train_indices)]
            valid = examples[examples["origin_index"].isin(valid_indices)].copy()
            if len(train) == 0 or len(valid) == 0:
                continue
            regressor, classifier = fit_panel_heads(train, self.debug, 1337 + fold_index * 17)
            decoded, probability, changed_state, uncertainty = predict_panel_heads(
                valid, regressor, classifier
            )
            valid["decoded_prediction"] = decoded
            valid["p_change"] = probability
            valid["changed_state_prediction"] = changed_state
            valid["panel_uncertainty"] = uncertainty
            baseline_mse = weighted_mse(
                valid["target_mean"], valid["baseline_state"], valid["target_count"]
            )
            decoded_mse = weighted_mse(
                valid["target_mean"], decoded, valid["target_count"]
            )
            fold_metrics.append({
                "fold": fold_index,
                "train_panels": int(len(train)),
                "valid_panels": int(len(valid)),
                "baseline_mse": baseline_mse,
                "decoded_mse": decoded_mse,
            })
            outputs.append(valid)
            del regressor, classifier
            gc.collect()
        if not outputs:
            raise RuntimeError("no forward OOF panel forecasts were produced")
        oof = pd.concat(outputs, ignore_index=True)
        mean_baseline = weighted_mse(oof["target_mean"], oof["baseline_state"], oof["target_count"])
        mean_decoded = weighted_mse(oof["target_mean"], oof["decoded_prediction"], oof["target_count"])
        worst_baseline = max(item["baseline_mse"] for item in fold_metrics)
        worst_decoded = max(item["decoded_mse"] for item in fold_metrics)
        use_gate = mean_decoded < mean_baseline and worst_decoded <= worst_baseline
        oof["selected_prediction"] = np.where(
            use_gate, oof["decoded_prediction"], oof["baseline_state"]
        )
        slope, intercept = affine_calibration(oof, "selected_prediction")
        calibrated = slope * oof["selected_prediction"].to_numpy(dtype=float) + intercept
        calibrated_mse = weighted_mse(oof["target_mean"], calibrated, oof["target_count"])
        selected_mse = weighted_mse(
            oof["target_mean"], oof["selected_prediction"], oof["target_count"]
        )
        use_calibration = calibrated_mse < selected_mse
        if not use_calibration:
            slope, intercept = 1.0, 0.0
        oof["panel_prediction"] = slope * oof["selected_prediction"] + intercept
        diagnostics = {
            "folds": fold_metrics,
            "mean_baseline_mse": mean_baseline,
            "mean_decoded_mse": mean_decoded,
            "use_gate": bool(use_gate),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "calibrated_mse": weighted_mse(
                oof["target_mean"], oof["panel_prediction"], oof["target_count"]
            ),
        }
        print(f"[forward_oof] {json.dumps(diagnostics)}", flush=True)
        return oof, diagnostics

    def attach_row_history(self, rows, access, origin):
        output = rows.copy()
        customer_history = access.customer_history(origin)
        article_dispersion = access.article_day_dispersion(origin)
        output = output.merge(customer_history, on="customer_id", how="left")
        output = output.merge(self.customer, on="customer_id", how="left")
        output = output.merge(article_dispersion, on="article_id", how="left")
        output["customer_activity_log"] = np.log1p(output["customer_activity"].fillna(0))
        output["customer_recent_activity_log"] = np.log1p(
            output["customer_recent_activity"].fillna(0)
        )
        channel_one = output["sales_channel_id"].to_numpy() == 1
        output["customer_channel_affinity"] = np.where(
            channel_one,
            output["customer_channel1_affinity"],
            output["customer_channel2_affinity"],
        )
        output["customer_channel_activity_log"] = np.log1p(
            np.where(
                channel_one,
                output["customer_channel1_activity"].fillna(0),
                output["customer_channel2_activity"].fillna(0),
            )
        )
        output["customer_affinity_distance"] = (
            output["customer_affinity"] - 1.0
        ).abs()
        output["article_activity_log"] = np.log1p(output["article_recent_activity"].fillna(0))
        return output

    @staticmethod
    def row_feature_columns():
        return [
            "weekday", "horizon", "sales_channel_id", "customer_affinity",
            "customer_affinity_std", "customer_affinity_distance",
            "customer_activity_log", "customer_recent_affinity",
            "customer_recent_activity_log", "customer_channel_affinity",
            "customer_channel_activity_log", "customer_channel2_share",
            "customer_mean_price", "customer_price_std",
            "FN", "Active", "age", "club_status_id", "news_frequency_id",
            "article_day_std", "article_day_range", "article_day_states",
            "article_activity_log", "p_change", "changed_state_gap",
            "panel_uncertainty", "last_std", "last_count", "weeks_since_observation",
            "product_support", "product_channel_support", "article_recent_support",
            "target_count_log", "horizon_mean", "weekday_mean",
        ]

    def build_residual_rows(self, oof, access, available_end):
        pieces = []
        panel_columns = [
            "article_id", "sales_channel_id", "panel_prediction", "p_change",
            "changed_state_prediction", "baseline_state", "panel_uncertainty",
            "last_std", "last_count", "weeks_since_observation", "product_support",
            "product_channel_support", "article_recent_support", "target_count_log",
            "horizon_mean", "weekday_mean", "has_article_history",
        ]
        for origin in sorted(oof["origin"].unique()):
            panels = oof[oof["origin"] == origin][panel_columns]
            rows = access.target_rows(origin, available_end)
            rows = rows.merge(panels, on=["article_id", "sales_channel_id"], how="inner")
            rows = rows[rows["has_article_history"] > 0].copy()
            rows["changed_state_gap"] = (
                rows["changed_state_prediction"] - rows["baseline_state"]
            ).abs()
            rows = self.attach_row_history(rows, access, origin)
            rows["origin"] = pd.Timestamp(origin)
            rows["residual_target"] = rows["price"] - rows["panel_prediction"]
            pieces.append(rows)
        residual = pd.concat(pieces, ignore_index=True)
        limit = 250000 if self.debug else 1800000
        if len(residual) > limit:
            residual = residual.sample(limit, random_state=1337).sort_index().reset_index(drop=True)
        return residual

    def fit_residual(self, residual_rows):
        columns = self.row_feature_columns()
        trees = 100 if self.debug else 500
        origins = sorted(residual_rows["origin"].unique())
        if len(origins) < 2:
            train_mask = np.ones(len(residual_rows), dtype=bool)
            valid_mask = np.ones(len(residual_rows), dtype=bool)
        else:
            holdout = origins[-1]
            train_mask = residual_rows["origin"].to_numpy() < holdout
            valid_mask = residual_rows["origin"].to_numpy() >= holdout
        parameters = dict(
            objective="regression_l2",
            n_estimators=trees,
            learning_rate=0.035,
            num_leaves=63,
            min_child_samples=1000,
            colsample_bytree=0.9,
            reg_lambda=10.0,
            random_state=1441,
            n_jobs=self.threads,
            verbosity=-1,
        )
        diagnostic = lgb.LGBMRegressor(**parameters)
        train_features = numeric_frame(residual_rows.loc[train_mask], columns)
        diagnostic.fit(train_features, residual_rows.loc[train_mask, "residual_target"])
        valid_features = numeric_frame(residual_rows.loc[valid_mask], columns)
        adjustment = np.clip(diagnostic.predict(valid_features), -0.02, 0.02)
        base = residual_rows.loc[valid_mask, "panel_prediction"].to_numpy(dtype=float)
        target = residual_rows.loc[valid_mask, "price"].to_numpy(dtype=float)
        base_mse = float(np.mean((target - base) ** 2))
        adjusted_mse = float(np.mean((target - base - adjustment) ** 2))
        use_residual = adjusted_mse < base_mse
        final_model = lgb.LGBMRegressor(**parameters)
        final_features = numeric_frame(residual_rows, columns)
        final_model.fit(final_features, residual_rows["residual_target"])
        diagnostics = {
            "rows": int(len(residual_rows)),
            "holdout_base_mse": base_mse,
            "holdout_adjusted_mse": adjusted_mse,
            "use_residual": bool(use_residual),
        }
        print(f"[row_residual] {json.dumps(diagnostics)}", flush=True)
        return final_model, use_residual, diagnostics

    def launch_feature_frame(self, frame):
        output = frame.copy()
        cascade_order = [
            "product_channel_state",
            "product_state",
            "product_type_no_state",
            "department_no_state",
            "section_no_state",
            "garment_group_no_state",
            "index_group_no_state",
            "global_recent_state",
        ]
        cascade = pd.Series(np.nan, index=output.index, dtype=float)
        for column in cascade_order:
            cascade = cascade.fillna(output[column])
        output["cascade_state"] = cascade
        launch_cascade_order = [
            "product_channel_launch_state",
            "product_launch_state",
            "product_channel_state",
            "product_state",
            "product_type_no_launch_state",
            "department_no_launch_state",
            "product_type_no_state",
            "department_no_state",
            "section_no_launch_state",
            "garment_group_no_launch_state",
            "index_group_no_launch_state",
            "global_recent_state",
        ]
        launch_cascade = pd.Series(np.nan, index=output.index, dtype=float)
        for column in launch_cascade_order:
            launch_cascade = launch_cascade.fillna(output[column])
        output["launch_cascade_state"] = launch_cascade
        ids = output["article_id"].fillna(-1).astype(int).to_numpy()
        valid = (ids >= 0) & (ids < len(self.text))
        for index in range(32):
            values = np.zeros(len(output), dtype=np.float32)
            values[valid] = self.text[ids[valid], index]
            output[f"text_svd_{index:02d}"] = values
        article_missing = self.article[["article_id", "prod_name", "detail_desc"]].copy()
        article_missing["prod_name_missing"] = article_missing["prod_name"].isna().astype(np.int8)
        article_missing["detail_desc_missing"] = article_missing["detail_desc"].isna().astype(np.int8)
        output = output.merge(
            article_missing[["article_id", "prod_name_missing", "detail_desc_missing"]],
            on="article_id", how="left",
        )
        for prior in LAUNCH_PRIOR_FEATURES + LAUNCH_HISTORY_FEATURES:
            output[f"{prior}_missing"] = output[prior].isna().astype(np.int8)
        return output

    @staticmethod
    def launch_feature_columns():
        missing = [
            f"{column}_missing"
            for column in LAUNCH_PRIOR_FEATURES + LAUNCH_HISTORY_FEATURES
        ]
        return (
            META_COLUMNS
            + [
                "sales_channel_id", "week_of_year", "cascade_state",
                "prod_name_missing", "detail_desc_missing",
            ]
            + LAUNCH_PRIOR_FEATURES
            + LAUNCH_HISTORY_FEATURES
            + missing
            + [f"text_svd_{index:02d}" for index in range(32)]
        )

    @staticmethod
    def attach_launch_history_priors(frame, launch_history, origin):
        output = frame.copy()
        safe = launch_history[
            pd.to_datetime(launch_history["first_date"]) + pd.Timedelta(days=6)
            < pd.Timestamp(origin)
        ].copy()
        definitions = [
            (["product_code"], "product_launch"),
            (["product_code", "sales_channel_id"], "product_channel_launch"),
            (["product_type_no"], "product_type_no_launch"),
            (["department_no"], "department_no_launch"),
            (["section_no"], "section_no_launch"),
            (["garment_group_no"], "garment_group_no_launch"),
            (["index_group_no"], "index_group_no_launch"),
        ]
        safe["weighted_launch_price"] = safe["target_mean"] * safe["target_count"]
        for keys, prefix in definitions:
            grouped = safe.groupby(keys, observed=True, sort=False).agg(
                weighted_launch_price=("weighted_launch_price", "sum"),
                launch_support=("target_count", "sum"),
            ).reset_index()
            grouped[f"{prefix}_state"] = (
                grouped["weighted_launch_price"] / grouped["launch_support"].clip(lower=1)
            )
            grouped = grouped[
                keys + [f"{prefix}_state", "launch_support"]
            ].rename(columns={"launch_support": f"{prefix}_support"})
            output = output.merge(grouped, on=keys, how="left")
        return output

    def fit_launch_model(self, access, builder, inference_origin):
        launch_history = access.launch_examples(inference_origin, 60).merge(
            builder.meta, on="article_id", how="left"
        )
        launches = launch_history[
            pd.to_datetime(launch_history["first_date"])
            >= pd.Timestamp(inference_origin) - pd.Timedelta(weeks=26)
        ].copy()
        pieces = []
        for launch_origin, group in launches.groupby("launch_origin", sort=True):
            base = group.copy()
            base = builder.attach_recent_priors(base, pd.Timestamp(launch_origin), weeks=4)
            base = self.attach_launch_history_priors(
                base, launch_history, pd.Timestamp(launch_origin)
            )
            base["week_of_year"] = int(pd.Timestamp(launch_origin).isocalendar().week)
            pieces.append(base)
        training = pd.concat(pieces, ignore_index=True)
        training = self.launch_feature_frame(training)
        weeks_ago = (
            pd.Timestamp(inference_origin) - pd.to_datetime(training["launch_origin"])
        ).dt.days / 7.0
        weights = training["target_count"].to_numpy(dtype=float) * np.exp(-weeks_ago.to_numpy() / 13.0)
        model = lgb.LGBMRegressor(
            objective="regression_l2",
            n_estimators=100 if self.debug else 800,
            learning_rate=0.05,
            num_leaves=127,
            min_child_samples=30,
            colsample_bytree=0.85,
            reg_lambda=10.0,
            random_state=1559,
            n_jobs=self.threads,
            verbosity=-1,
        )
        features = numeric_frame(training, self.launch_feature_columns())
        model.fit(
            features,
            training["target_mean"] - training["cascade_state"],
            sample_weight=weights,
        )
        return model, int(len(training)), launch_history

    def inference_row_features(self, rows, panel_frame, access, origin):
        panel_columns = [
            "article_id", "sales_channel_id", "panel_prediction", "p_change",
            "changed_state_prediction", "baseline_state", "panel_uncertainty",
            "last_std", "last_count", "weeks_since_observation", "product_support",
            "product_channel_support", "article_recent_support", "target_count_log",
            "horizon_mean", "weekday_mean", "has_article_history",
        ]
        output = rows.copy()
        output["horizon"] = (pd.to_datetime(output["t_dat"]) - pd.Timestamp(origin)).dt.days
        output["weekday"] = pd.to_datetime(output["t_dat"]).dt.dayofweek + 1
        output = output.merge(
            panel_frame[panel_columns], on=["article_id", "sales_channel_id"], how="left"
        )
        output["changed_state_gap"] = (
            output["changed_state_prediction"] - output["baseline_state"]
        ).abs()
        return self.attach_row_history(output, access, origin)

    def cache_oof(self, chain_name, oof):
        suffix = "debug" if self.debug else "full"
        path = self.shared / f"lane2_{chain_name}_panel_oof_{suffix}_v1.parquet"
        columns = [
            "origin", "article_id", "sales_channel_id", "target_mean", "target_count",
            "baseline_state", "decoded_prediction", "panel_prediction", "p_change",
            "changed_state_prediction", "panel_uncertainty",
        ]
        oof[columns].to_parquet(path, index=False, compression="zstd")
        self.register_artifact(
            f"lane2_{chain_name}_panel_oof_{suffix}_v1",
            path,
            "Forward out-of-fold article-channel panel forecasts",
            f"relhm_{chain_name}_panel_oof_{suffix}_v1",
            "Rebuild by running the lane 2 candidate with legal task-label chains",
        )

    def run_chain(self, chain_name, label_splits, inference_rows, inference_origin):
        phase_start = time.time()
        access = DataAccess(label_splits, self.threads)
        label_min, label_max, available_end, label_count = access.label_range()
        cache_key = "train_val" if "val" in label_splits else "train"
        weekly_path = self.shared / f"lane2_weekly_tasklabels_{cache_key}_v1.parquet"
        weekly = access.weekly_panel(weekly_path)
        self.register_artifact(
            f"lane2_weekly_tasklabels_{cache_key}_v1",
            weekly_path,
            "Compact task-label-only article-channel-week price panels",
            f"relhm_weekly_tasklabels_{cache_key}_v1",
            "Join task split labels to projected transaction inputs by primary_key and aggregate",
        )
        builder = FeatureBuilder(weekly, self.article)
        history_origins = 4 if self.debug else 12
        origins = [
            pd.Timestamp(inference_origin) - pd.Timedelta(weeks=index)
            for index in range(history_origins, 0, -1)
        ]
        examples = self.build_examples(access, builder, origins, available_end)
        print(
            f"[{chain_name}] weekly_panels={len(weekly)} supervised_panels={len(examples)} "
            f"labels={label_count} build={time.time() - phase_start:.1f}s",
            flush=True,
        )
        oof, panel_diagnostics = self.forward_oof(examples, history_origins)
        self.cache_oof(chain_name, oof)
        residual_rows = self.build_residual_rows(oof, access, available_end)
        residual_model, use_residual, residual_diagnostics = self.fit_residual(residual_rows)

        panel_regressor, panel_classifier = fit_panel_heads(examples, self.debug, 1711)
        inference_panels = self.inference_panels(inference_rows, inference_origin)
        inference_panels = self.attach_snapshot_activity(
            inference_panels, access, inference_origin
        )
        inference_features = builder.panel_features(inference_panels, inference_origin)
        decoded, probability, changed_state, uncertainty = predict_panel_heads(
            inference_features, panel_regressor, panel_classifier
        )
        selected = decoded if panel_diagnostics["use_gate"] else inference_features["baseline_state"].to_numpy(dtype=float)
        panel_prediction = (
            panel_diagnostics["calibration_slope"] * selected
            + panel_diagnostics["calibration_intercept"]
        )
        inference_features["panel_prediction"] = panel_prediction
        inference_features["p_change"] = probability
        inference_features["changed_state_prediction"] = changed_state
        inference_features["panel_uncertainty"] = uncertainty

        launch_model, launch_rows, launch_history = self.fit_launch_model(
            access, builder, inference_origin
        )
        cold = inference_features["has_article_history"].to_numpy() == 0
        if cold.any():
            cold_features = self.attach_launch_history_priors(
                inference_features.loc[cold].copy(), launch_history, inference_origin
            )
            cold_features = self.launch_feature_frame(cold_features)
            launch_prediction = cold_features["cascade_state"].to_numpy(dtype=float) + launch_model.predict(
                numeric_frame(cold_features, self.launch_feature_columns())
            )
            launch_prediction += 0.20 * (
                cold_features["launch_cascade_state"].to_numpy(dtype=float)
                - cold_features["cascade_state"].to_numpy(dtype=float)
            )
            inference_features.loc[cold, "panel_prediction"] = launch_prediction

        row_features = self.inference_row_features(
            inference_rows, inference_features, access, inference_origin
        )
        predictions = row_features["panel_prediction"].to_numpy(dtype=float)
        warm_rows = row_features["has_article_history"].to_numpy() > 0
        if use_residual and warm_rows.any():
            adjustment = residual_model.predict(
                numeric_frame(row_features.loc[warm_rows], self.row_feature_columns())
            )
            predictions[warm_rows] += np.clip(adjustment, -0.02, 0.02)
        predictions = np.clip(predictions, label_min, label_max)
        ordering = row_features["row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(ordering, np.arange(len(ordering))):
            restored = np.empty(len(predictions), dtype=float)
            restored[ordering] = predictions
            predictions = restored
        diagnostics = {
            "chain": chain_name,
            "label_splits": list(label_splits),
            "label_count": label_count,
            "weekly_panels": int(len(weekly)),
            "supervised_panels": int(len(examples)),
            "oof_panels": int(len(oof)),
            "launch_examples": launch_rows,
            "inference_panels": int(len(inference_features)),
            "cold_panels": int(cold.sum()),
            "cold_rows": int((~warm_rows).sum()),
            "panel": panel_diagnostics,
            "residual": residual_diagnostics,
            "elapsed_seconds": time.time() - phase_start,
        }
        print(f"[{chain_name}] complete {json.dumps(diagnostics)}", flush=True)
        access.close()
        del weekly, builder, examples, oof, residual_rows, residual_model
        gc.collect()
        return predictions, diagnostics

    @staticmethod
    def write_diagnostics(run_dir, diagnostics_a, diagnostics_b):
        payload = {
            "validation_chain": diagnostics_a,
            "test_chain": diagnostics_b,
            "panel_price_source": "task labels only",
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
