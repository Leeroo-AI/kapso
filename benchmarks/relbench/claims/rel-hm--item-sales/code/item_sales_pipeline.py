import gc
import json
import os
import platform
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow
import sklearn

from daily_features import DailyFeaturePanel
from kapso_datasets.common import run_data_dir, shared_cache_dir


PIPELINE_VERSION = "wide_lgbm_l1_v1"
RESULT_VERSION = "wide_daily_lgbm_l1_v1"
WEEKLY_VERSION = "hm_item_sales_weekly_v1"
FEATURE_VERSION = "hm_item_sales_features_v1"
RAW_COLUMNS = [
    "sales",
    "transaction_count",
    "channel_1_count",
    "channel_2_count",
    "price_mean",
    "price_median",
    "price_std",
    "price_min",
    "price_max",
    "distinct_customers",
    "buyer_age_mean",
    "buyer_age_std",
    "member_share",
    "news_share",
    "fn_share",
    "active_share",
    "repeat_buyer_share",
    "postal_diversity",
]
MISSING_RAW = {
    "price_mean",
    "price_median",
    "price_std",
    "price_min",
    "price_max",
    "buyer_age_mean",
    "buyer_age_std",
    "member_share",
    "news_share",
    "fn_share",
    "active_share",
    "repeat_buyer_share",
}
HIERARCHIES = [
    "product_code",
    "product_type_no",
    "department_no",
    "section_no",
    "garment_group_no",
    "index_code",
    "colour_group_code",
    "graphical_appearance_no",
]
CUSTOMER_RAW = [
    "buyer_age_mean",
    "buyer_age_std",
    "member_share",
    "news_share",
    "fn_share",
    "active_share",
    "repeat_buyer_share",
    "postal_diversity",
]
warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    history: int
    leaves: int
    min_leaf: int
    decay: float


FULL_GRID = [
    ModelConfig("h13_l64_m3000_d100", 13, 64, 3000, 1.0),
    ModelConfig("h26_l128_m1000_d098", 26, 128, 1000, 0.98),
    ModelConfig("h52_l64_m3000_d098", 52, 64, 3000, 0.98),
]
DEBUG_CONFIG = ModelConfig("debug_h2", 2, 64, 1000, 1.0)


def emit(event: str, start: float, **values) -> None:
    payload = {"event": event, "elapsed_seconds": round(time.time() - start, 3)}
    payload.update(values)
    print(f"[pipeline] {json.dumps(payload, sort_keys=True)}")


def print_versions() -> None:
    values = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "duckdb": duckdb.__version__,
        "lightgbm": lgb.__version__,
        "sklearn": sklearn.__version__,
        "pyarrow": pyarrow.__version__,
    }
    print(f"[pipeline] versions {json.dumps(values, sort_keys=True)}")


def build_weekly_base(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    database = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"] / "db"
    transactions = database / "transactions.parquet"
    customers = database / "customer.parquet"
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={int(os.environ.get('OMP_NUM_THREADS', '11'))}")
    connection.execute("PRAGMA memory_limit='80GB'")
    query = f"""
COPY (
WITH tx AS (
 SELECT CAST(date_trunc('week', t_dat - INTERVAL '1 day') + INTERVAL '7 day' AS DATE) week_end,
        article_id, customer_id, price, sales_channel_id
 FROM read_parquet('{transactions}')
),
weekly_tx AS (
 SELECT week_end, article_id, SUM(price) sales, COUNT(*) transaction_count,
        SUM(sales_channel_id = 1) channel_1_count,
        SUM(sales_channel_id = 2) channel_2_count,
        AVG(price) price_mean, MEDIAN(price) price_median,
        STDDEV_POP(price) price_std, MIN(price) price_min, MAX(price) price_max
 FROM tx GROUP BY week_end, article_id
),
buyer_week AS (
 SELECT week_end, article_id, customer_id,
        MIN(week_end) OVER (PARTITION BY article_id, customer_id) first_week
 FROM tx GROUP BY week_end, article_id, customer_id
),
weekly_buyer AS (
 SELECT b.week_end, b.article_id, COUNT(*) distinct_customers,
        AVG(c.age) buyer_age_mean,
        STDDEV_POP(c.age) buyer_age_std,
        AVG(CAST(COALESCE(c.club_member_status = 'ACTIVE', false) AS DOUBLE)) member_share,
        AVG(CAST(COALESCE(c.fashion_news_frequency != 'NONE', false) AS DOUBLE)) news_share,
        AVG(COALESCE(c.FN, 0)) fn_share,
        AVG(COALESCE(c.Active, 0)) active_share,
        AVG(CAST(b.week_end > b.first_week AS DOUBLE)) repeat_buyer_share,
        COUNT(DISTINCT c.postal_code) postal_diversity
 FROM buyer_week b
 LEFT JOIN read_parquet('{customers}') c USING (customer_id)
 GROUP BY b.week_end, b.article_id
)
SELECT w.*, b.distinct_customers, b.buyer_age_mean, b.buyer_age_std,
       b.member_share, b.news_share, b.fn_share, b.active_share,
       b.repeat_buyer_share, b.postal_diversity
FROM weekly_tx w JOIN weekly_buyer b USING (week_end, article_id)
ORDER BY week_end, article_id
) TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)
"""
    connection.execute(query)
    connection.close()


def factorize(values: pd.Series) -> np.ndarray:
    codes, _ = pd.factorize(values.astype("string").fillna("__MISSING__"), sort=True)
    return codes.astype(np.int32)


def divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator != 0,
    )


class FeaturePanel:
    def __init__(self, ctx, debug: bool, start: float):
        articles = ctx.db.table_dict["article"].df.sort_values("article_id").reset_index(drop=True)
        self.articles = articles
        self.article_ids = articles["article_id"].to_numpy(np.int64)
        self.n_articles = len(articles)
        self.position = np.full(int(self.article_ids.max()) + 1, -1, dtype=np.int64)
        self.position[self.article_ids] = np.arange(self.n_articles)
        self.train_times = pd.DatetimeIndex(sorted(ctx.train.df["timestamp"].unique()))
        extra_times = list(pd.DatetimeIndex(ctx.val.df["timestamp"].unique()))
        extra_times.extend(pd.DatetimeIndex(ctx.test.df["timestamp"].unique()))
        endpoints = sorted(set(self.train_times.tolist() + extra_times))
        weekly_path = shared_cache_dir() / WEEKLY_VERSION / "weekly_base.parquet"
        if not weekly_path.exists():
            build_weekly_base(weekly_path)
        weekly = pd.read_parquet(weekly_path)
        weekly["week_end"] = pd.to_datetime(weekly["week_end"])
        observed_first = weekly["week_end"].min()
        self.week_dates = pd.date_range(observed_first, max(endpoints), freq="7D")
        self.week_lookup = {pd.Timestamp(value): index for index, value in enumerate(self.week_dates)}
        rows = weekly["week_end"].map(self.week_lookup).to_numpy()
        cols = self.position[weekly["article_id"].to_numpy(np.int64)]
        valid = (rows >= 0) & (cols >= 0)
        self.raw = {}
        for column in RAW_COLUMNS:
            fill = np.nan if column in MISSING_RAW else 0.0
            values = np.full((len(self.week_dates), self.n_articles), fill, dtype=np.float32)
            values[rows[valid], cols[valid]] = weekly.loc[valid, column].to_numpy(np.float32)
            self.raw[column] = values
        self.sales = self.raw["sales"]
        self.active = self.sales > 0
        self.zero_streak = np.zeros_like(self.sales, dtype=np.int16)
        self.days_since_sale = np.full_like(self.sales, 9999, dtype=np.int16)
        self.article_age = np.zeros_like(self.sales, dtype=np.int16)
        self.active_count = np.zeros_like(self.sales, dtype=np.int16)
        last_index = np.full(self.n_articles, -1, dtype=np.int16)
        first_index = np.full(self.n_articles, -1, dtype=np.int16)
        count = np.zeros(self.n_articles, dtype=np.int16)
        streak = np.zeros(self.n_articles, dtype=np.int16)
        for index in range(len(self.week_dates)):
            active = self.active[index]
            first_index[(first_index < 0) & active] = index
            last_index[active] = index
            count += active.astype(np.int16)
            streak = np.where(active, 0, np.minimum(streak + 1, 9999))
            self.zero_streak[index] = streak
            self.days_since_sale[index] = np.where(last_index >= 0, (index - last_index) * 7, 9999)
            self.article_age[index] = np.where(first_index >= 0, index - first_index, 0)
            self.active_count[index] = count
        self.ewma = {}
        for span in (2, 4, 8, 13, 26):
            alpha = np.float32(2.0 / (span + 1.0))
            values = np.zeros_like(self.sales)
            values[0] = self.sales[0]
            for index in range(1, len(self.week_dates)):
                values[index] = alpha * self.sales[index] + (1 - alpha) * values[index - 1]
            self.ewma[span] = values
        self.last_price = np.full_like(self.sales, np.nan)
        current_price = np.full(self.n_articles, np.nan, dtype=np.float32)
        for index in range(len(self.week_dates)):
            sold = self.raw["transaction_count"][index] > 0
            current_price[sold] = self.raw["price_mean"][index, sold]
            self.last_price[index] = current_price
        self.category_codes = {}
        for column in HIERARCHIES:
            self.category_codes[column] = factorize(articles[column])
        for column in (
            "index_group_no",
            "product_group_name",
            "perceived_colour_value_id",
            "perceived_colour_master_id",
        ):
            self.category_codes[column] = factorize(articles[column])
        self.group_sizes = {
            column: np.bincount(codes).astype(np.float32)
            for column, codes in self.category_codes.items()
        }
        self.cache_dir = shared_cache_dir() / FEATURE_VERSION
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory = {}
        self.debug = debug
        emit(
            "aggregation",
            start,
            weekly_rows=len(weekly),
            weeks=len(self.week_dates),
            articles=self.n_articles,
            weekly_cache=str(weekly_path.relative_to(shared_cache_dir())),
        )

    def index(self, timestamp) -> int:
        return self.week_lookup[pd.Timestamp(timestamp)]

    def positions(self, article_ids: np.ndarray) -> np.ndarray:
        return self.position[np.asarray(article_ids, dtype=np.int64)]

    def window(self, values: np.ndarray, index: int, width: int) -> np.ndarray:
        return values[max(0, index - width + 1) : index + 1]

    def weighted_recent(self, column: str, index: int, width: int) -> np.ndarray:
        values = self.window(self.raw[column], index, width)
        weights = self.window(self.raw["distinct_customers"], index, width)
        finite = np.isfinite(values)
        numerator = np.nansum(values * weights, axis=0)
        denominator = np.sum(weights * finite, axis=0)
        result = np.full(self.n_articles, np.nan, dtype=np.float32)
        np.divide(numerator, denominator, out=result, where=denominator > 0)
        return result

    def build(self, timestamp) -> pd.DataFrame:
        timestamp = pd.Timestamp(timestamp)
        index = self.index(timestamp)
        data = {}
        for lag in list(range(1, 14)) + [26, 52]:
            source = index - lag + 1
            data[f"sales_lag_{lag}"] = (
                self.sales[source] if source >= 0 else np.zeros(self.n_articles, dtype=np.float32)
            )
        for width in (2, 4, 8, 13, 26):
            values = self.window(self.sales, index, width)
            data[f"sales_sum_{width}"] = values.sum(axis=0)
            data[f"sales_mean_{width}"] = values.mean(axis=0)
            data[f"sales_median_{width}"] = np.median(values, axis=0)
            data[f"sales_max_{width}"] = values.max(axis=0)
            data[f"sales_std_{width}"] = values.std(axis=0)
            data[f"active_weeks_{width}"] = (values > 0).sum(axis=0)
        transaction = self.raw["transaction_count"]
        customers = self.raw["distinct_customers"]
        channel_1 = self.raw["channel_1_count"]
        channel_2 = self.raw["channel_2_count"]
        data["transaction_count_lag_1"] = transaction[index]
        data["distinct_customers_lag_1"] = customers[index]
        data["channel_1_count_lag_1"] = channel_1[index]
        data["channel_2_count_lag_1"] = channel_2[index]
        data["channel_2_share_lag_1"] = divide(channel_2[index], transaction[index])
        for width in (4, 13, 26):
            tx = self.window(transaction, index, width).sum(axis=0)
            buyers = self.window(customers, index, width).sum(axis=0)
            c1 = self.window(channel_1, index, width).sum(axis=0)
            c2 = self.window(channel_2, index, width).sum(axis=0)
            data[f"transaction_count_sum_{width}"] = tx
            data[f"distinct_customers_sum_{width}"] = buyers
            data[f"transactions_per_customer_{width}"] = divide(tx, buyers)
            data[f"channel_2_share_{width}"] = divide(c2, c1 + c2)
        data["zero_streak"] = self.zero_streak[index]
        data["days_since_last_sale"] = self.days_since_sale[index]
        data["article_age_weeks"] = self.article_age[index]
        data["never_seen"] = (self.days_since_sale[index] == 9999).astype(np.float32)
        data["active_week_rate"] = self.active_count[index] / np.float32(index + 1)
        data["active_week_rate_26"] = (
            self.window(self.active, index, 26).mean(axis=0).astype(np.float32)
        )
        for span, values in self.ewma.items():
            data[f"sales_ewma_{span}"] = values[index]
        mean_2 = self.window(self.sales, index, 2).mean(axis=0)
        mean_4 = self.window(self.sales, index, 4).mean(axis=0)
        mean_8 = self.window(self.sales, index, 8).mean(axis=0)
        mean_13 = self.window(self.sales, index, 13).mean(axis=0)
        data["sales_trend_4_13"] = mean_4 - mean_13
        data["sales_trend_ratio_4_13"] = divide(mean_4, mean_13 + np.float32(1e-4))
        data["sales_acceleration_2_4_8"] = (mean_2 - mean_4) - (mean_4 - mean_8)
        data["last_price"] = self.last_price[index]
        data["price_mean_lag_1"] = self.raw["price_mean"][index]
        data["price_median_lag_1"] = self.raw["price_median"][index]
        data["price_std_lag_1"] = self.raw["price_std"][index]
        data["price_range_lag_1"] = (
            self.raw["price_max"][index] - self.raw["price_min"][index]
        )
        data["sales_per_transaction_lag_1"] = divide(self.sales[index], transaction[index])
        for width in (4, 13, 26):
            sales = self.window(self.sales, index, width).sum(axis=0)
            tx = self.window(transaction, index, width).sum(axis=0)
            price_values = self.window(self.raw["price_median"], index, width)
            data[f"price_mean_{width}"] = divide(sales, tx)
            data[f"price_median_of_weeks_{width}"] = np.nanmedian(price_values, axis=0)
            data[f"sales_per_transaction_{width}"] = divide(sales, tx)
        data["price_change_1_13"] = (
            data["price_mean_lag_1"] - data["price_mean_13"]
        )
        for column in CUSTOMER_RAW:
            data[f"{column}_lag_1"] = self.raw[column][index]
            data[f"{column}_recent_4"] = self.weighted_recent(column, index, 4)
            data[f"{column}_recent_13"] = self.weighted_recent(column, index, 13)
        for column, codes in self.category_codes.items():
            data[f"cat_{column}"] = codes
        recent_four = self.window(self.sales, index, 4).sum(axis=0)
        recent_thirteen = self.window(self.sales, index, 13).sum(axis=0)
        for column in HIERARCHIES:
            codes = self.category_codes[column]
            size = self.group_sizes[column]
            group_last = np.bincount(codes, weights=self.sales[index], minlength=len(size)).astype(
                np.float32
            )
            group_four = np.bincount(codes, weights=recent_four, minlength=len(size)).astype(
                np.float32
            )
            group_thirteen = np.bincount(
                codes, weights=recent_thirteen, minlength=len(size)
            ).astype(np.float32)
            active_last = np.bincount(
                codes, weights=self.active[index], minlength=len(size)
            ).astype(np.float32)
            active_four = np.bincount(
                codes, weights=recent_four > 0, minlength=len(size)
            ).astype(np.float32)
            short = column.replace("_no", "").replace("_code", "")
            data[f"group_{short}_sales_1"] = group_last[codes]
            data[f"group_{short}_sales_mean_4"] = group_four[codes] / np.float32(4)
            data[f"group_{short}_sales_mean_13"] = group_thirteen[codes] / np.float32(13)
            data[f"group_{short}_active_fraction_1"] = active_last[codes] / size[codes]
            data[f"group_{short}_active_fraction_4"] = active_four[codes] / size[codes]
            data[f"group_{short}_trend_4_13"] = (
                group_four[codes] / np.float32(4) - group_thirteen[codes] / np.float32(13)
            )
            data[f"group_{short}_article_share_1"] = divide(
                self.sales[index], group_last[codes]
            )
            data[f"group_{short}_article_share_4"] = divide(recent_four, group_four[codes])
            data[f"interaction_age_{short}"] = (
                self.article_age[index] * data[f"group_{short}_active_fraction_4"]
            )
        lifecycle = np.full(self.n_articles, 4, dtype=np.int32)
        lifecycle[self.days_since_sale[index] == 9999] = 0
        lifecycle[(recent_thirteen > 0) & (recent_four == 0)] = 1
        lifecycle[recent_four > 0] = 2
        lifecycle[self.sales[index] > 0] = 3
        data["cat_lifecycle"] = lifecycle
        for column in ("product_type_no", "department_no", "section_no", "garment_group_no"):
            codes = self.category_codes[column]
            data[f"cat_lifecycle_x_{column}"] = (
                codes.astype(np.int64) * 5 + lifecycle
            ).astype(np.int32)
        data["cat_product_type_x_colour"] = (
            self.category_codes["product_type_no"].astype(np.int64)
            * (int(self.category_codes["colour_group_code"].max()) + 1)
            + self.category_codes["colour_group_code"]
        ).astype(np.int32)
        week = int(timestamp.isocalendar().week)
        data["week_of_year_sin"] = np.full(
            self.n_articles, np.sin(2 * np.pi * week / 52.0), dtype=np.float32
        )
        data["week_of_year_cos"] = np.full(
            self.n_articles, np.cos(2 * np.pi * week / 52.0), dtype=np.float32
        )
        data["calendar_year"] = np.full(self.n_articles, timestamp.year, dtype=np.float32)
        frame = pd.DataFrame(data)
        for column in frame.columns:
            if frame[column].dtype == np.float64:
                frame[column] = frame[column].astype(np.float32)
        return frame

    def get(self, timestamp) -> pd.DataFrame:
        timestamp = pd.Timestamp(timestamp)
        key = timestamp.strftime("%Y%m%d")
        if key in self.memory:
            return self.memory[key]
        path = self.cache_dir / f"origin_{key}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
        else:
            frame = self.build(timestamp)
            frame.to_parquet(path, compression="zstd", index=False)
        self.memory[key] = frame
        return frame

    def fallback(self, timestamp) -> np.ndarray:
        return self.sales[self.index(timestamp)].astype(np.float64)

    def regime_masks(self, timestamp) -> dict:
        index = self.index(timestamp)
        recent_four = self.window(self.sales, index, 4).sum(axis=0) > 0
        prior = self.sales[: max(0, index - 3)].sum(axis=0) > 0
        return {
            "all": np.ones(self.n_articles, dtype=bool),
            "active": self.sales[index] > 0,
            "recent4": recent_four,
            "new4": recent_four & ~prior,
            "dormant8": self.window(self.sales, index, 8).sum(axis=0) == 0,
            "bursty8": (
                self.window(self.sales, index, 8).max(axis=0)
                > 3 * np.maximum(self.window(self.sales, index, 8).mean(axis=0), 1e-7)
            )
            & recent_four,
        }


def make_target_panel(ctx, panel: FeaturePanel) -> np.ndarray:
    values = np.zeros((len(panel.train_times), panel.n_articles), dtype=np.float32)
    time_index = {pd.Timestamp(value): index for index, value in enumerate(panel.train_times)}
    rows = ctx.train.df
    origins = rows["timestamp"].map(time_index).to_numpy()
    positions = panel.positions(rows["article_id"].to_numpy())
    values[origins, positions] = rows[ctx.target_col].to_numpy(np.float32)
    return values


def feature_columns(frame: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set == "demand":
        blocked = (
            "cat_",
            "group_",
            "interaction_",
            "buyer_",
            "member_",
            "news_",
            "fn_",
            "active_share",
            "repeat_",
            "postal_",
            "price_",
        )
        return [column for column in frame.columns if not column.startswith(blocked)]
    if feature_set == "demand_hierarchy":
        blocked = (
            "buyer_",
            "member_",
            "news_",
            "fn_",
            "active_share",
            "repeat_",
            "postal_",
            "price_",
        )
        return [column for column in frame.columns if not column.startswith(blocked)]
    return frame.columns.tolist()


def matrix_for(
    panel: FeaturePanel,
    times,
    feature_set: str,
    subset: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    first = panel.get(times[0])
    columns = feature_columns(first, feature_set)
    matrices = [
        panel.get(timestamp).iloc[subset][columns].to_numpy(np.float32, copy=False)
        for timestamp in times
    ]
    return np.concatenate(matrices, axis=0), columns


def daily_matrix_for(
    panel: FeaturePanel,
    daily: DailyFeaturePanel,
    times,
    subset: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    weekly, columns = matrix_for(panel, times, "demand", subset)
    daily_columns = daily.get(times[0]).columns.tolist()
    daily_values = np.concatenate(
        [
            daily.get(timestamp).iloc[subset].to_numpy(np.float32, copy=False)
            for timestamp in times
        ],
        axis=0,
    )
    values = np.concatenate([weekly, daily_values], axis=1)
    return values, columns + daily_columns


def labels_for(targets: np.ndarray, indices, subset: np.ndarray) -> np.ndarray:
    return np.concatenate([targets[index, subset] for index in indices]).astype(np.float32)


def categorical_indices(values: np.ndarray, columns: list[str]) -> list[int]:
    limit = min(len(values), 105542)
    indices = []
    for index, column in enumerate(columns):
        if column.startswith("cat_") and np.nanmax(values[:limit, index]) <= 255:
            indices.append(index)
    return indices


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    columns: list[str],
    config: ModelConfig,
    rounds: int,
    early_stopping: int,
    debug: bool,
):
    categorical = categorical_indices(x_train, columns)
    params = {
        "objective": "regression_l1",
        "metric": "l1",
        "learning_rate": 0.04,
        "num_leaves": config.leaves,
        "min_data_in_leaf": config.min_leaf,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "max_bin": 127,
        "max_cat_to_onehot": 4,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "verbosity": -1,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "11")),
        "seed": 1337,
        "feature_fraction_seed": 1337,
        "bagging_seed": 1337,
        "device_type": "cpu" if debug else "gpu",
        "gpu_use_dp": False,
    }
    training = lgb.Dataset(
        x_train,
        label=y_train,
        weight=weights,
        feature_name=columns,
        categorical_feature=categorical,
        free_raw_data=True,
    )
    validation = lgb.Dataset(
        x_valid,
        label=y_valid,
        feature_name=columns,
        categorical_feature=categorical,
        reference=training,
        free_raw_data=True,
    )
    callbacks = [lgb.log_evaluation(0)]
    if early_stopping:
        callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
    try:
        model = lgb.train(
            params,
            training,
            num_boost_round=rounds,
            valid_sets=[validation],
            valid_names=["forward"],
            callbacks=callbacks,
        )
    except lgb.basic.LightGBMError as error:
        if "cannot run on GPU" not in str(error):
            raise
        params["device_type"] = "cpu"
        model = lgb.train(
            params,
            training,
            num_boost_round=rounds,
            valid_sets=[validation],
            valid_names=["forward"],
            callbacks=callbacks,
        )
    return model


def fold_result(
    panel: FeaturePanel,
    targets: np.ndarray,
    fold_index: int,
    feature_set: str,
    config: ModelConfig,
    subset: np.ndarray,
    rounds: int,
    early_stopping: int,
    debug: bool,
) -> tuple[dict, lgb.Booster]:
    start_index = max(0, fold_index - config.history)
    train_indices = list(range(start_index, fold_index))
    train_times = panel.train_times[train_indices]
    valid_time = panel.train_times[fold_index]
    x_train, columns = matrix_for(panel, train_times, feature_set, subset)
    x_valid, valid_columns = matrix_for(panel, [valid_time], feature_set, subset)
    if columns != valid_columns:
        raise RuntimeError("feature column mismatch")
    y_train = labels_for(targets, train_indices, subset)
    y_valid = targets[fold_index, subset]
    ages = np.arange(len(train_indices) - 1, -1, -1)
    origin_weights = np.power(config.decay, ages).astype(np.float32)
    weights = np.repeat(origin_weights, len(subset))
    model = train_model(
        x_train,
        y_train,
        weights,
        x_valid,
        y_valid,
        columns,
        config,
        rounds,
        early_stopping,
        debug,
    )
    prediction = np.maximum(model.predict(x_valid, num_iteration=model.best_iteration), 0)
    masks = panel.regime_masks(valid_time)
    result = {
        "origin": str(pd.Timestamp(valid_time).date()),
        "mae": float(np.mean(np.abs(y_valid - prediction))),
        "best_iteration": int(model.best_iteration or rounds),
    }
    full_positions = subset
    for name, full_mask in masks.items():
        mask = full_mask[full_positions]
        result[f"{name}_count"] = int(mask.sum())
        result[f"{name}_mae"] = (
            float(np.mean(np.abs(y_valid[mask] - prediction[mask]))) if mask.any() else None
        )
    del x_train, x_valid, y_train, y_valid, weights
    gc.collect()
    return result, model


def criterion(results: list[dict]) -> float:
    values = np.asarray([result["mae"] for result in results], dtype=np.float64)
    return float(values.mean() + 0.5 * values.std(ddof=1 if len(values) > 1 else 0))


def evaluate_daily_features(
    panel: FeaturePanel,
    targets: np.ndarray,
    folds: list[int],
    subset: np.ndarray,
    config: ModelConfig,
    weekly_criterion: float,
    debug: bool,
    start: float,
) -> tuple[bool, int, dict]:
    if debug:
        return False, 80, {"status": "debug_disabled"}
    daily = DailyFeaturePanel(panel.article_ids)
    cache = shared_cache_dir() / PIPELINE_VERSION / "daily_gate"
    cache.mkdir(parents=True, exist_ok=True)
    results = []
    for fold in folds:
        result_path = cache / f"fold_{fold}.json"
        prediction_path = cache / f"fold_{fold}.npy"
        if result_path.exists() and prediction_path.exists():
            with result_path.open() as handle:
                result = json.load(handle)
        else:
            indices = list(range(max(0, fold - config.history), fold))
            times = panel.train_times[indices]
            valid_time = panel.train_times[fold]
            x_train, columns = daily_matrix_for(panel, daily, times, subset)
            x_valid, valid_columns = daily_matrix_for(panel, daily, [valid_time], subset)
            if columns != valid_columns:
                raise RuntimeError("daily feature column mismatch")
            y_train = labels_for(targets, indices, subset)
            y_valid = targets[fold, subset]
            ages = np.arange(len(indices) - 1, -1, -1)
            weights = np.repeat(
                np.power(config.decay, ages).astype(np.float32), len(subset)
            )
            model = train_model(
                x_train,
                y_train,
                weights,
                x_valid,
                y_valid,
                columns,
                config,
                1500,
                100,
                False,
            )
            prediction = np.maximum(
                model.predict(x_valid, num_iteration=model.best_iteration), 0
            ).astype(np.float32)
            result = {
                "fold": fold,
                "origin": str(pd.Timestamp(valid_time).date()),
                "mae": float(np.mean(np.abs(y_valid - prediction))),
                "best_iteration": int(model.best_iteration),
            }
            with result_path.open("w") as handle:
                json.dump(result, handle)
            np.save(prediction_path, prediction)
            del x_train, x_valid, y_train, y_valid, weights, model, prediction
            gc.collect()
        results.append(result)
        emit("daily_feature_fold", start, fold=result)
    daily_criterion = criterion(results)
    weekly_results_path = (
        shared_cache_dir()
        / PIPELINE_VERSION
        / "full_folds"
    )
    weekly_results = []
    for fold in folds:
        path = weekly_results_path / (
            f"n105542_first0_last105541_demand_{config.name}_{fold}.json"
        )
        if path.exists():
            with path.open() as handle:
                weekly_results.append(json.load(handle))
    improvements = (
        sum(
            daily_result["mae"] < weekly_result["mae"]
            for daily_result, weekly_result in zip(results, weekly_results)
        )
        if len(weekly_results) == len(results)
        else 0
    )
    selected = daily_criterion < weekly_criterion and improvements >= 2
    iterations = int(np.median([result["best_iteration"] for result in results]))
    diagnostics = {
        "folds": results,
        "criterion": daily_criterion,
        "weekly_criterion": weekly_criterion,
        "improved_folds": improvements,
        "selected": selected,
        "frozen_iterations": iterations,
    }
    emit("daily_feature_selection", start, diagnostics=diagnostics)
    return selected, iterations, diagnostics


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    expanded = np.broadcast_to(weights[:, None], values.shape)
    sorted_weights = np.take_along_axis(expanded, order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    index = (cumulative >= weights.sum() / 2).argmax(axis=0)
    return sorted_values[index, np.arange(values.shape[1])]


def deterministic_forecast(panel: FeaturePanel, timestamp, name: str) -> np.ndarray:
    index = panel.index(timestamp)
    last = panel.sales[index]
    if name == "last_week":
        return last.astype(np.float64)
    if name.startswith("median_"):
        width = int(name.rsplit("_", 1)[1])
        return np.median(panel.window(panel.sales, index, width), axis=0)
    if name.startswith("recency_mean_"):
        width = int(name.rsplit("_", 1)[1])
        values = panel.window(panel.sales, index, width)
        weights = np.power(0.7, np.arange(len(values) - 1, -1, -1))
        return np.average(values, axis=0, weights=weights)
    if name.startswith("recency_median_"):
        width = int(name.rsplit("_", 1)[1])
        values = panel.window(panel.sales, index, width)
        weights = np.power(0.7, np.arange(len(values) - 1, -1, -1))
        return weighted_median(values, weights)
    product = panel.category_codes["product_code"]
    product_median = (
        pd.Series(last).groupby(product, sort=False).transform("median").to_numpy(np.float32)
    )
    if name == "last_product_075":
        return 0.75 * last + 0.25 * product_median
    if name == "last_product_050":
        return 0.50 * last + 0.50 * product_median
    return 0.25 * last + 0.75 * product_median


def bank_fallback(panel: FeaturePanel, targets: np.ndarray, folds: list[int]) -> dict:
    candidates = [
        "last_week",
        "median_2",
        "median_4",
        "median_8",
        "recency_mean_4",
        "recency_mean_8",
        "recency_median_4",
        "recency_median_8",
        "last_product_075",
        "last_product_050",
        "last_product_025",
    ]
    measurements = {}
    for name in candidates:
        values = []
        for fold in folds:
            prediction = deterministic_forecast(panel, panel.train_times[fold], name)
            values.append(float(np.mean(np.abs(targets[fold] - prediction))))
        array = np.asarray(values)
        measurements[name] = {
            "fold_mae": values,
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1 if len(array) > 1 else 0)),
            "criterion": float(
                array.mean() + 0.5 * array.std(ddof=1 if len(array) > 1 else 0)
            ),
        }
    winner = min(measurements, key=lambda key: measurements[key]["criterion"])
    return {"winner": winner, "measurements": measurements}


def evaluate_designs(
    panel: FeaturePanel,
    targets: np.ndarray,
    folds: list[int],
    subset: np.ndarray,
    debug: bool,
    start: float,
) -> tuple[str, ModelConfig, int, dict]:
    rounds = 80 if debug else 1500
    early = 20 if debug else 100
    if debug:
        sets = ["full"]
        configs = [DEBUG_CONFIG]
    else:
        sets = ["demand", "demand_hierarchy", "full"]
        configs = [FULL_GRID[1]]
    records = {}
    reusable = {}
    fold_cache_dir = shared_cache_dir() / PIPELINE_VERSION / (
        "debug_folds" if debug else "full_folds"
    )
    fold_cache_dir.mkdir(parents=True, exist_ok=True)
    subset_tag = f"n{len(subset)}_first{int(subset[0])}_last{int(subset[-1])}"
    for feature_set in sets:
        key = f"{feature_set}:{configs[0].name}"
        results = []
        for fold in folds:
            fold_start = time.time()
            fold_path = fold_cache_dir / (
                f"{subset_tag}_{feature_set}_{configs[0].name}_{fold}.json"
            )
            if fold_path.exists():
                with fold_path.open() as handle:
                    result = json.load(handle)
                model = None
            else:
                result, model = fold_result(
                    panel,
                    targets,
                    fold,
                    feature_set,
                    configs[0],
                    subset,
                    rounds,
                    early,
                    debug,
                )
                with fold_path.open("w") as handle:
                    json.dump(result, handle)
            results.append(result)
            emit(
                "feature_fold",
                start,
                feature_set=feature_set,
                config=configs[0].name,
                fold=result,
                fit_seconds=round(time.time() - fold_start, 3),
            )
            del model
            gc.collect()
        records[key] = {"folds": results, "criterion": criterion(results)}
        reusable[key] = results
    feature_scores = {
        feature_set: records[f"{feature_set}:{configs[0].name}"]["criterion"]
        for feature_set in sets
    }
    chosen_set = min(feature_scores, key=feature_scores.get)
    if not debug:
        demand_results = records[f"demand:{configs[0].name}"]["folds"]
        hierarchy_results = records[f"demand_hierarchy:{configs[0].name}"]["folds"]
        full_results = records[f"full:{configs[0].name}"]["folds"]
        hierarchy_improvements = sum(
            right["mae"] < left["mae"]
            for left, right in zip(demand_results, hierarchy_results)
        )
        customer_improvements = sum(
            right["mae"] < left["mae"]
            for left, right in zip(hierarchy_results, full_results)
        )
        if hierarchy_improvements >= 2:
            chosen_set = "demand_hierarchy"
        else:
            chosen_set = "demand"
        if customer_improvements >= 2 and feature_scores["full"] <= feature_scores[chosen_set] * 1.01:
            chosen_set = "full"
    grid = [DEBUG_CONFIG] if debug else FULL_GRID
    for config in grid:
        key = f"{chosen_set}:{config.name}"
        if key in records:
            continue
        results = []
        for fold in folds:
            fold_start = time.time()
            fold_path = fold_cache_dir / (
                f"{subset_tag}_{chosen_set}_{config.name}_{fold}.json"
            )
            if fold_path.exists():
                with fold_path.open() as handle:
                    result = json.load(handle)
                model = None
            else:
                result, model = fold_result(
                    panel,
                    targets,
                    fold,
                    chosen_set,
                    config,
                    subset,
                    rounds,
                    early,
                    debug,
                )
                with fold_path.open("w") as handle:
                    json.dump(result, handle)
            results.append(result)
            emit(
                "grid_fold",
                start,
                feature_set=chosen_set,
                config=config.name,
                fold=result,
                fit_seconds=round(time.time() - fold_start, 3),
            )
            del model
            gc.collect()
        records[key] = {"folds": results, "criterion": criterion(results)}
    candidates = {
        config.name: records[f"{chosen_set}:{config.name}"]["criterion"] for config in grid
    }
    chosen_name = min(candidates, key=candidates.get)
    chosen_config = next(config for config in grid if config.name == chosen_name)
    chosen_results = records[f"{chosen_set}:{chosen_name}"]["folds"]
    iterations = int(np.median([result["best_iteration"] for result in chosen_results]))
    iterations = max(1, min(rounds, iterations))
    selection = {
        "feature_scores": feature_scores,
        "chosen_feature_set": chosen_set,
        "grid_scores": candidates,
        "chosen_config": asdict(chosen_config),
        "frozen_iterations": iterations,
        "records": records,
    }
    emit("selection", start, selection=selection)
    return chosen_set, chosen_config, iterations, selection


def fit_final(
    panel: FeaturePanel,
    targets: np.ndarray,
    train_indices: list[int],
    extra_time,
    extra_target: np.ndarray,
    predict_time,
    feature_set: str,
    config: ModelConfig,
    iterations: int,
    subset: np.ndarray,
    debug: bool,
):
    times = list(panel.train_times[train_indices])
    labels = [targets[index, subset] for index in train_indices]
    if extra_time is not None:
        times.append(pd.Timestamp(extra_time))
        labels.append(extra_target[subset])
    x_train, columns = matrix_for(panel, times, feature_set, subset)
    y_train = np.concatenate(labels).astype(np.float32)
    ages = np.arange(len(times) - 1, -1, -1)
    origin_weights = np.power(config.decay, ages).astype(np.float32)
    weights = np.repeat(origin_weights, len(subset))
    x_predict, predict_columns = matrix_for(panel, [predict_time], feature_set, subset)
    if columns != predict_columns:
        raise RuntimeError("final feature column mismatch")
    categorical = categorical_indices(x_train, columns)
    params = {
        "objective": "regression_l1",
        "metric": "l1",
        "learning_rate": 0.04,
        "num_leaves": config.leaves,
        "min_data_in_leaf": config.min_leaf,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "max_bin": 127,
        "max_cat_to_onehot": 4,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "verbosity": -1,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "11")),
        "seed": 1337,
        "feature_fraction_seed": 1337,
        "bagging_seed": 1337,
        "device_type": "cpu" if debug else "gpu",
        "gpu_use_dp": False,
    }
    dataset = lgb.Dataset(
        x_train,
        label=y_train,
        weight=weights,
        feature_name=columns,
        categorical_feature=categorical,
        free_raw_data=True,
    )
    try:
        model = lgb.train(
            params,
            dataset,
            num_boost_round=iterations,
            callbacks=[lgb.log_evaluation(0)],
        )
    except lgb.basic.LightGBMError as error:
        if "cannot run on GPU" not in str(error):
            raise
        params["device_type"] = "cpu"
        model = lgb.train(
            params,
            dataset,
            num_boost_round=iterations,
            callbacks=[lgb.log_evaluation(0)],
        )
    prediction = model.predict(x_predict, num_iteration=iterations)
    del x_train, x_predict, y_train, weights, dataset
    gc.collect()
    return prediction, model


def fit_final_daily(
    panel: FeaturePanel,
    daily: DailyFeaturePanel,
    targets: np.ndarray,
    train_indices: list[int],
    extra_time,
    extra_target: np.ndarray,
    predict_time,
    config: ModelConfig,
    iterations: int,
    subset: np.ndarray,
    debug: bool,
):
    times = list(panel.train_times[train_indices])
    labels = [targets[index, subset] for index in train_indices]
    if extra_time is not None:
        times.append(pd.Timestamp(extra_time))
        labels.append(extra_target[subset])
    x_train, columns = daily_matrix_for(panel, daily, times, subset)
    y_train = np.concatenate(labels).astype(np.float32)
    ages = np.arange(len(times) - 1, -1, -1)
    origin_weights = np.power(config.decay, ages).astype(np.float32)
    weights = np.repeat(origin_weights, len(subset))
    x_predict, predict_columns = daily_matrix_for(
        panel, daily, [predict_time], subset
    )
    if columns != predict_columns:
        raise RuntimeError("final daily feature column mismatch")
    categorical = categorical_indices(x_train, columns)
    params = {
        "objective": "regression_l1",
        "metric": "l1",
        "learning_rate": 0.04,
        "num_leaves": config.leaves,
        "min_data_in_leaf": config.min_leaf,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "max_bin": 127,
        "max_cat_to_onehot": 4,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "verbosity": -1,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "11")),
        "seed": 1337,
        "feature_fraction_seed": 1337,
        "bagging_seed": 1337,
        "device_type": "cpu" if debug else "gpu",
        "gpu_use_dp": False,
    }
    dataset = lgb.Dataset(
        x_train,
        label=y_train,
        weight=weights,
        feature_name=columns,
        categorical_feature=categorical,
        free_raw_data=True,
    )
    try:
        model = lgb.train(
            params,
            dataset,
            num_boost_round=iterations,
            callbacks=[lgb.log_evaluation(0)],
        )
    except lgb.basic.LightGBMError as error:
        if "cannot run on GPU" not in str(error):
            raise
        params["device_type"] = "cpu"
        model = lgb.train(
            params,
            dataset,
            num_boost_round=iterations,
            callbacks=[lgb.log_evaluation(0)],
        )
    prediction = model.predict(x_predict, num_iteration=iterations)
    del x_train, x_predict, y_train, weights, dataset
    gc.collect()
    return prediction, model


def write_artifacts(
    ctx,
    panel: FeaturePanel,
    val_by_position: np.ndarray,
    test_by_position: np.ndarray,
    diagnostics: dict,
) -> None:
    historical_max = float(ctx.train.df[ctx.target_col].max())
    val_by_position = np.clip(val_by_position, 0, historical_max)
    test_by_position = np.clip(test_by_position, 0, historical_max)
    val_positions = panel.positions(ctx.val.df["article_id"].to_numpy())
    test_positions = panel.positions(ctx.test.df["article_id"].to_numpy())
    val_prediction = val_by_position[val_positions].astype(np.float64)
    test_prediction = test_by_position[test_positions].astype(np.float64)
    out = run_data_dir()
    np.save(out / "val_predictions.npy", val_prediction)
    np.save(out / "test_predictions.npy", test_prediction)
    with (out / "metrics.json").open("w") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(
        f"[pipeline] saved val{val_prediction.shape} test{test_prediction.shape} "
        f"finite={bool(np.isfinite(val_prediction).all() and np.isfinite(test_prediction).all())}"
    )


def run_pipeline(ctx, debug: bool) -> None:
    start = time.time()
    print_versions()
    result_cache = shared_cache_dir() / RESULT_VERSION / (
        "debug_predictions.npz" if debug else "full_predictions.npz"
    )
    metrics_cache = result_cache.with_suffix(".json")
    if result_cache.exists() and metrics_cache.exists():
        cached = np.load(result_cache, allow_pickle=False)
        out = run_data_dir()
        np.save(out / "val_predictions.npy", cached["val"])
        np.save(out / "test_predictions.npy", cached["test"])
        with metrics_cache.open() as handle:
            diagnostics = json.load(handle)
        with (out / "metrics.json").open("w") as handle:
            json.dump(diagnostics, handle, indent=2)
        emit("reused_predictions", start, cache=str(result_cache.relative_to(shared_cache_dir())))
        return
    panel = FeaturePanel(ctx, debug, start)
    targets = make_target_panel(ctx, panel)
    if debug:
        folds = [len(panel.train_times) - 2, len(panel.train_times) - 1]
        subset = np.flatnonzero((panel.article_ids * 2654435761 % 8) == 0)
    else:
        folds = list(range(len(panel.train_times) - 4, len(panel.train_times)))
        subset = np.arange(panel.n_articles)
    fallback_diagnostics = bank_fallback(panel, targets, folds)
    emit("fallback", start, diagnostics=fallback_diagnostics)
    feature_set, config, iterations, selection = evaluate_designs(
        panel, targets, folds, subset, debug, start
    )
    weekly_criterion = selection["grid_scores"][config.name]
    daily_selected, daily_iterations, daily_selection = evaluate_daily_features(
        panel,
        targets,
        folds,
        subset,
        config,
        weekly_criterion,
        debug,
        start,
    )
    daily = DailyFeaturePanel(panel.article_ids) if daily_selected else None
    if daily_selected:
        feature_set = "demand_daily"
        iterations = daily_iterations
    history = min(config.history, len(panel.train_times))
    train_indices = list(range(len(panel.train_times) - history, len(panel.train_times)))
    val_time = pd.Timestamp(ctx.val.df["timestamp"].iloc[0])
    test_time = pd.Timestamp(ctx.test.df["timestamp"].iloc[0])
    val_positions = panel.positions(ctx.val.df["article_id"].to_numpy())
    val_target_position = np.zeros(panel.n_articles, dtype=np.float32)
    val_target_position[val_positions] = ctx.val.df[ctx.target_col].to_numpy(np.float32)
    fallback_name = fallback_diagnostics["winner"]
    val_fallback = deterministic_forecast(panel, val_time, fallback_name).astype(np.float64)
    test_fallback = deterministic_forecast(panel, test_time, fallback_name).astype(np.float64)
    val_prediction = val_fallback.copy()
    test_prediction = test_fallback.copy()
    if daily_selected:
        model_a_prediction, model_a = fit_final_daily(
            panel,
            daily,
            targets,
            train_indices,
            None,
            None,
            val_time,
            config,
            iterations,
            subset,
            debug,
        )
    else:
        model_a_prediction, model_a = fit_final(
            panel,
            targets,
            train_indices,
            None,
            None,
            val_time,
            feature_set,
            config,
            iterations,
            subset,
            debug,
        )
    val_prediction[subset] = model_a_prediction
    emit("model_a_fit", start, rows=len(train_indices) * len(subset), iterations=iterations)
    del model_a
    gc.collect()
    if daily_selected:
        model_b_prediction, model_b = fit_final_daily(
            panel,
            daily,
            targets,
            train_indices,
            val_time,
            val_target_position,
            test_time,
            config,
            iterations,
            subset,
            debug,
        )
    else:
        model_b_prediction, model_b = fit_final(
            panel,
            targets,
            train_indices,
            val_time,
            val_target_position,
            test_time,
            feature_set,
            config,
            iterations,
            subset,
            debug,
        )
    test_prediction[subset] = model_b_prediction
    emit(
        "model_b_fit",
        start,
        rows=(len(train_indices) + 1) * len(subset),
        iterations=iterations,
    )
    del model_b
    gc.collect()
    diagnostics = {
        "pipeline_version": PIPELINE_VERSION,
        "result_version": RESULT_VERSION,
        "debug": debug,
        "fallback": fallback_diagnostics,
        "selection": selection,
        "daily_selection": daily_selection,
        "final_feature_set": feature_set,
        "model_a_training_origins": [
            str(pd.Timestamp(panel.train_times[index]).date()) for index in train_indices
        ],
        "model_b_added_origin": str(val_time.date()),
        "validation_prediction_fit": "model_a_train_only",
        "test_prediction_fit": "model_b_train_plus_validation",
        "processed_articles": int(len(subset)),
        "postprocess": {"kind": "none"},
    }
    write_artifacts(ctx, panel, val_prediction, test_prediction, diagnostics)
    result_cache.parent.mkdir(parents=True, exist_ok=True)
    out = run_data_dir()
    np.savez_compressed(
        result_cache,
        val=np.load(out / "val_predictions.npy", allow_pickle=False),
        test=np.load(out / "test_predictions.npy", allow_pickle=False),
    )
    with metrics_cache.open("w") as handle:
        json.dump(diagnostics, handle, indent=2)
    emit("artifact_validation_ready", start, cache=str(result_cache.relative_to(shared_cache_dir())))
