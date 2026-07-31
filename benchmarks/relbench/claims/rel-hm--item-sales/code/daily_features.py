import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from kapso_datasets.common import shared_cache_dir


DAILY_BASE_VERSION = "hm_item_sales_daily_v1"
DAILY_FEATURE_VERSION = "hm_item_sales_daily_features_v1"
DAILY_COLUMNS = [
    "sales",
    "transaction_count",
    "distinct_customers",
    "channel_1_count",
    "channel_2_count",
    "price_mean",
]


def divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator != 0,
    )


def build_daily_base(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source = (
        Path(os.environ["RELBENCH_CACHE_DIR"])
        / os.environ["RELBENCH_DATASET"]
        / "db"
        / "transactions.parquet"
    )
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={int(os.environ.get('OMP_NUM_THREADS', '11'))}")
    connection.execute("PRAGMA memory_limit='80GB'")
    connection.execute(
        f"""
COPY (
 SELECT CAST(t_dat AS DATE) date, article_id,
        SUM(price) sales, COUNT(*) transaction_count,
        COUNT(DISTINCT customer_id) distinct_customers,
        SUM(sales_channel_id = 1) channel_1_count,
        SUM(sales_channel_id = 2) channel_2_count,
        AVG(price) price_mean
 FROM read_parquet('{source}')
 GROUP BY date, article_id
 ORDER BY date, article_id
) TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)
"""
    )
    connection.close()


class DailyFeaturePanel:
    def __init__(self, article_ids: np.ndarray):
        self.article_ids = np.asarray(article_ids, dtype=np.int64)
        self.n_articles = len(article_ids)
        self.position = np.full(int(self.article_ids.max()) + 1, -1, dtype=np.int64)
        self.position[self.article_ids] = np.arange(self.n_articles)
        base_path = shared_cache_dir() / DAILY_BASE_VERSION / "daily_base.parquet"
        if not base_path.exists():
            build_daily_base(base_path)
        daily = pd.read_parquet(base_path)
        daily["date"] = pd.to_datetime(daily["date"])
        self.dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
        self.date_lookup = {pd.Timestamp(value): index for index, value in enumerate(self.dates)}
        rows = daily["date"].map(self.date_lookup).to_numpy(np.int64)
        columns = self.position[daily["article_id"].to_numpy(np.int64)]
        self.raw = {}
        for name in DAILY_COLUMNS:
            fill = np.nan if name == "price_mean" else 0.0
            values = np.full((len(self.dates), self.n_articles), fill, dtype=np.float32)
            values[rows, columns] = daily[name].to_numpy(np.float32)
            self.raw[name] = values
        self.cache_dir = shared_cache_dir() / DAILY_FEATURE_VERSION
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory = {}

    def window(self, values: np.ndarray, index: int, width: int) -> np.ndarray:
        return values[max(0, index - width + 1) : index + 1]

    def build(self, timestamp) -> pd.DataFrame:
        timestamp = pd.Timestamp(timestamp)
        index = self.date_lookup[timestamp]
        sales = self.raw["sales"]
        transactions = self.raw["transaction_count"]
        customers = self.raw["distinct_customers"]
        channel_1 = self.raw["channel_1_count"]
        channel_2 = self.raw["channel_2_count"]
        data = {}
        for lag in range(14):
            source = index - lag
            data[f"daily_sales_lag_{lag}"] = sales[source]
        for lag in range(7):
            source = index - lag
            data[f"daily_transaction_lag_{lag}"] = transactions[source]
            data[f"daily_customers_lag_{lag}"] = customers[source]
            data[f"daily_price_lag_{lag}"] = self.raw["price_mean"][source]
            data[f"daily_channel_2_share_lag_{lag}"] = divide(
                channel_2[source], transactions[source]
            )
        for width in (3, 7, 14, 28):
            recent_sales = self.window(sales, index, width)
            recent_transactions = self.window(transactions, index, width)
            recent_customers = self.window(customers, index, width)
            recent_channel_1 = self.window(channel_1, index, width)
            recent_channel_2 = self.window(channel_2, index, width)
            data[f"daily_sales_sum_{width}"] = recent_sales.sum(axis=0)
            data[f"daily_sales_mean_{width}"] = recent_sales.mean(axis=0)
            data[f"daily_sales_max_{width}"] = recent_sales.max(axis=0)
            data[f"daily_sales_std_{width}"] = recent_sales.std(axis=0)
            data[f"daily_transaction_sum_{width}"] = recent_transactions.sum(axis=0)
            data[f"daily_customers_sum_{width}"] = recent_customers.sum(axis=0)
            data[f"daily_channel_2_share_{width}"] = divide(
                recent_channel_2.sum(axis=0),
                recent_channel_1.sum(axis=0) + recent_channel_2.sum(axis=0),
            )
            data[f"daily_active_days_{width}"] = (recent_sales > 0).sum(axis=0)
        last_28 = self.window(sales, index, 28)
        reversed_active = (last_28 > 0)[::-1]
        has_sale = reversed_active.any(axis=0)
        recency = reversed_active.argmax(axis=0).astype(np.float32)
        recency[~has_sale] = 99
        data["daily_days_since_sale"] = recency
        data["daily_recent_3_share_14"] = divide(
            self.window(sales, index, 3).sum(axis=0),
            self.window(sales, index, 14).sum(axis=0),
        )
        data["daily_origin_day_share_7"] = divide(
            sales[index], self.window(sales, index, 7).sum(axis=0)
        )
        return pd.DataFrame(data, dtype=np.float32)

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
