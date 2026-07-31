from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


BASE_DAY = pd.Timestamp("2019-09-01")
ARTICLE_COLUMNS = [
    "product_code",
    "product_type_no",
    "product_group_name",
    "graphical_appearance_no",
    "colour_group_code",
    "department_no",
    "index_code",
    "index_group_no",
    "section_no",
    "garment_group_no",
]


def day_number(values) -> np.ndarray:
    return ((pd.to_datetime(values) - BASE_DAY) / pd.Timedelta(days=1)).astype(np.int16)


def factor(values) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(values, sort=True)
    return (codes + 1).astype(np.int32), len(uniques) + 1


def hash_text(values, buckets: int) -> np.ndarray:
    hashed = pd.util.hash_pandas_object(values.fillna("").astype(str), index=False).to_numpy()
    return (hashed % buckets).astype(np.int32)


@dataclass
class HMState:
    transactions: pd.DataFrame
    articles: pd.DataFrame
    customers: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_items: int
    n_customers: int
    item_features: np.ndarray
    item_cardinalities: list[int]
    customer_features: np.ndarray
    customer_cardinalities: list[int]
    origin_days: np.ndarray
    origin_to_index: dict[int, int]
    popularity_buckets: np.ndarray
    price_buckets: np.ndarray
    customer_ptr: np.ndarray
    history_days: np.ndarray
    history_items: np.ndarray
    history_channels: np.ndarray
    history_prices: np.ndarray


def build_dynamic_features(
    tx_day: np.ndarray,
    tx_item: np.ndarray,
    tx_price: np.ndarray,
    origin_days: np.ndarray,
    n_items: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame(
        {
            "day": tx_day.astype(np.int32),
            "item": tx_item.astype(np.int32),
            "price": tx_price.astype(np.float32),
        }
    )
    daily = (
        frame.groupby(["day", "item"], sort=True, observed=True)
        .price.agg(["size", "sum"])
        .reset_index()
    )
    days = daily.day.to_numpy(np.int32)
    items = daily.item.to_numpy(np.int32)
    counts = daily["size"].to_numpy(np.int32)
    sums = daily["sum"].to_numpy(np.float64)
    cumulative_count = np.zeros(n_items, dtype=np.int32)
    cumulative_sum = np.zeros(n_items, dtype=np.float64)
    recent_count = np.zeros(n_items, dtype=np.int32)
    pop = np.zeros((len(origin_days), n_items), dtype=np.uint8)
    price = np.zeros((len(origin_days), n_items), dtype=np.uint8)
    add_at = 0
    remove_at = 0
    for origin_index, origin_day in enumerate(origin_days.astype(np.int32)):
        add_end = np.searchsorted(days, origin_day, side="right")
        if add_end > add_at:
            np.add.at(cumulative_count, items[add_at:add_end], counts[add_at:add_end])
            np.add.at(cumulative_sum, items[add_at:add_end], sums[add_at:add_end])
            np.add.at(recent_count, items[add_at:add_end], counts[add_at:add_end])
            add_at = add_end
        remove_end = np.searchsorted(days, origin_day - 28, side="right")
        if remove_end > remove_at:
            np.add.at(recent_count, items[remove_at:remove_end], -counts[remove_at:remove_end])
            remove_at = remove_end
        pop[origin_index] = np.clip(np.floor(np.log2(recent_count + 1)), 0, 15).astype(np.uint8)
        average = np.divide(
            cumulative_sum,
            cumulative_count,
            out=np.zeros(n_items, dtype=np.float64),
            where=cumulative_count > 0,
        )
        price[origin_index] = np.clip(
            np.floor(np.log1p(average * 100.0) * 10.0), 0, 31
        ).astype(np.uint8)
    return pop, price


def load_state() -> HMState:
    import os

    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    dataset_name = os.environ["RELBENCH_DATASET"]
    task_name = os.environ["RELBENCH_TASK"]
    dataset = get_dataset(dataset_name, download=False)
    task = get_task(dataset_name, task_name, download=False)
    db = dataset.get_db(upto_test_timestamp=True)
    transactions = db.table_dict["transactions"].df
    articles = db.table_dict["article"].df.sort_values("article_id").reset_index(drop=True)
    customers = db.table_dict["customer"].df.sort_values("customer_id").reset_index(drop=True)
    train = task.get_table("train").df
    val = task.get_table("val").df
    test = task.get_table("test").df
    n_items = len(articles)
    n_customers = len(customers)
    item_parts = []
    item_cardinalities = []
    for column in ARTICLE_COLUMNS:
        encoded, cardinality = factor(articles[column])
        item_parts.append(encoded)
        item_cardinalities.append(cardinality)
    item_parts.append(hash_text(articles["prod_name"], 8192))
    item_cardinalities.append(8192)
    item_parts.append(hash_text(articles["detail_desc"], 16384))
    item_cardinalities.append(16384)
    item_features = np.stack(item_parts, axis=1)
    age = np.where(customers.age.notna(), np.clip(customers.age.fillna(0) // 5 + 1, 1, 25), 0)
    fn = np.where(customers.FN.notna(), customers.FN.fillna(0).astype(int) + 1, 0)
    active = np.where(customers.Active.notna(), customers.Active.fillna(0).astype(int) + 1, 0)
    club, club_card = factor(customers.club_member_status)
    news, news_card = factor(customers.fashion_news_frequency)
    postal_frequency = customers.postal_code.map(customers.postal_code.value_counts()).fillna(0)
    postal_bucket = np.clip(np.floor(np.log2(postal_frequency + 1)), 0, 15).astype(np.int32)
    customer_features = np.stack(
        [
            age.astype(np.int32),
            fn.astype(np.int32),
            active.astype(np.int32),
            club,
            news,
            postal_bucket,
        ],
        axis=1,
    )
    customer_cardinalities = [26, 3, 3, club_card, news_card, 16]
    tx_day = day_number(transactions.t_dat).to_numpy(np.int16)
    tx_customer = transactions.customer_id.to_numpy(np.int32)
    tx_item = transactions.article_id.to_numpy(np.int32)
    tx_channel = transactions.sales_channel_id.to_numpy(np.int8)
    tx_price = transactions.price.to_numpy(np.float32)
    origin_days = np.unique(
        np.concatenate(
            [
                day_number(train.timestamp).to_numpy(np.int16),
                day_number(val.timestamp).to_numpy(np.int16),
                day_number(test.timestamp).to_numpy(np.int16),
            ]
        )
    )
    popularity_buckets, price_buckets = build_dynamic_features(
        tx_day, tx_item, tx_price, origin_days, n_items
    )
    order = np.lexsort((tx_day, tx_customer))
    sorted_customer = tx_customer[order]
    customer_counts = np.bincount(sorted_customer, minlength=n_customers)
    customer_ptr = np.empty(n_customers + 1, dtype=np.int64)
    customer_ptr[0] = 0
    np.cumsum(customer_counts, out=customer_ptr[1:])
    return HMState(
        transactions=transactions,
        articles=articles,
        customers=customers,
        train=train,
        val=val,
        test=test,
        n_items=n_items,
        n_customers=n_customers,
        item_features=item_features,
        item_cardinalities=item_cardinalities,
        customer_features=customer_features,
        customer_cardinalities=customer_cardinalities,
        origin_days=origin_days,
        origin_to_index={int(day): i for i, day in enumerate(origin_days)},
        popularity_buckets=popularity_buckets,
        price_buckets=price_buckets,
        customer_ptr=customer_ptr,
        history_days=tx_day[order],
        history_items=tx_item[order],
        history_channels=tx_channel[order],
        history_prices=tx_price[order],
    )


def choose_recent_indices(frame: pd.DataFrame, limit: int, seed: int) -> np.ndarray:
    timestamps = pd.to_datetime(frame.timestamp)
    origins = np.sort(timestamps.unique())
    recent_start = origins[max(0, len(origins) - 14)]
    indices = np.flatnonzero(timestamps.to_numpy() >= recent_start)
    if len(indices) <= limit:
        return indices
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(indices, limit, replace=False))


class EpisodeDataset:
    def __init__(
        self,
        state: HMState,
        frames: Iterable[pd.DataFrame],
        indices: Iterable[np.ndarray] | None = None,
    ):
        selected = []
        frame_list = list(frames)
        if indices is None:
            indices = [np.arange(len(frame), dtype=np.int64) for frame in frame_list]
        for frame, frame_indices in zip(frame_list, indices):
            part = frame.iloc[np.asarray(frame_indices)]
            selected.append(part)
        joined = pd.concat(selected, ignore_index=True)
        self.state = state
        self.customers = joined.customer_id.to_numpy(np.int32)
        self.days = day_number(joined.timestamp).to_numpy(np.int16)
        self.origins = np.asarray(
            [state.origin_to_index[int(day)] for day in self.days], dtype=np.int16
        )
        self.positives = joined.article_id.to_numpy(object)

    def __len__(self):
        return len(self.customers)

    def __getitem__(self, index):
        customer = int(self.customers[index])
        query_day = int(self.days[index])
        start = int(self.state.customer_ptr[customer])
        stop = int(self.state.customer_ptr[customer + 1])
        relative_stop = np.searchsorted(
            self.state.history_days[start:stop], query_day, side="right"
        )
        stop = start + int(relative_stop)
        begin = max(start, stop - 128)
        days = self.state.history_days[begin:stop]
        if len(days):
            unique_days = np.unique(days)
            if len(unique_days) > 32:
                begin += int(np.searchsorted(days, unique_days[-32], side="left"))
                days = self.state.history_days[begin:stop]
        return (
            customer,
            query_day,
            int(self.origins[index]),
            self.state.history_items[begin:stop],
            days,
            self.state.history_channels[begin:stop],
            self.state.history_prices[begin:stop],
            np.asarray(self.positives[index], dtype=np.int32),
        )

    def collate(self, rows):
        import torch

        batch_size = len(rows)
        max_positive = max(len(row[7]) for row in rows)
        history = np.full((batch_size, 128), self.state.n_items, dtype=np.int32)
        basket = np.zeros((batch_size, 128), dtype=np.int16)
        valid = np.zeros((batch_size, 128), dtype=np.bool_)
        channel = np.zeros((batch_size, 128), dtype=np.int8)
        history_price = np.zeros((batch_size, 128), dtype=np.float32)
        history_day = np.zeros((batch_size, 128), dtype=np.int16)
        context_numeric = np.zeros((batch_size, 5), dtype=np.float32)
        positives = np.full((batch_size, max_positive), self.state.n_items, dtype=np.int32)
        positive_valid = np.zeros((batch_size, max_positive), dtype=np.bool_)
        customers = np.empty(batch_size, dtype=np.int32)
        days = np.empty(batch_size, dtype=np.int16)
        origins = np.empty(batch_size, dtype=np.int16)
        for row_index, row in enumerate(rows):
            customer, query_day, origin, items, item_days, channels, prices, targets = row
            length = len(items)
            if length:
                history[row_index, :length] = items
                unique_days, inverse = np.unique(item_days, return_inverse=True)
                basket[row_index, :length] = inverse
                valid[row_index, :length] = True
                channel[row_index, :length] = channels
                history_price[row_index, :length] = prices
                history_day[row_index, :length] = item_days
                category = self.state.item_features[items, 2]
                category_count = np.unique(category, return_counts=True)[1].astype(np.float32)
                probabilities = category_count / category_count.sum()
                entropy = float(-(probabilities * np.log(probabilities + 1e-8)).sum())
                context_numeric[row_index] = [
                    np.log1p(length),
                    min(365, query_day - int(item_days[-1])) / 365.0,
                    entropy,
                    float(np.bincount(channels.astype(np.int32), minlength=3).argmax()) / 2.0,
                    float(np.median(prices)),
                ]
            positive_length = len(targets)
            positives[row_index, :positive_length] = targets
            positive_valid[row_index, :positive_length] = True
            customers[row_index] = customer
            days[row_index] = query_day
            origins[row_index] = origin
        customer_features = self.state.customer_features[customers]
        return {
            "customer": torch.from_numpy(customers),
            "query_day": torch.from_numpy(days),
            "origin": torch.from_numpy(origins),
            "history": torch.from_numpy(history),
            "basket": torch.from_numpy(basket),
            "valid": torch.from_numpy(valid),
            "channel": torch.from_numpy(channel),
            "history_price": torch.from_numpy(history_price),
            "history_day": torch.from_numpy(history_day),
            "context_numeric": torch.from_numpy(context_numeric),
            "customer_features": torch.from_numpy(customer_features),
            "positives": torch.from_numpy(positives),
            "positive_valid": torch.from_numpy(positive_valid),
        }


class InferenceDataset(EpisodeDataset):
    def __init__(self, state: HMState, frame: pd.DataFrame):
        copied = frame.copy()
        copied["article_id"] = [np.empty(0, dtype=np.int32) for _ in range(len(copied))]
        super().__init__(state, [copied])

    def collate(self, rows):
        import torch

        batch_size = len(rows)
        history = np.full((batch_size, 128), self.state.n_items, dtype=np.int32)
        basket = np.zeros((batch_size, 128), dtype=np.int16)
        valid = np.zeros((batch_size, 128), dtype=np.bool_)
        channel = np.zeros((batch_size, 128), dtype=np.int8)
        history_price = np.zeros((batch_size, 128), dtype=np.float32)
        history_day = np.zeros((batch_size, 128), dtype=np.int16)
        context_numeric = np.zeros((batch_size, 5), dtype=np.float32)
        customers = np.empty(batch_size, dtype=np.int32)
        days = np.empty(batch_size, dtype=np.int16)
        origins = np.empty(batch_size, dtype=np.int16)
        for row_index, row in enumerate(rows):
            customer, query_day, origin, items, item_days, channels, prices, _ = row
            length = len(items)
            if length:
                history[row_index, :length] = items
                _, inverse = np.unique(item_days, return_inverse=True)
                basket[row_index, :length] = inverse
                valid[row_index, :length] = True
                channel[row_index, :length] = channels
                history_price[row_index, :length] = prices
                history_day[row_index, :length] = item_days
                category = self.state.item_features[items, 2]
                category_count = np.unique(category, return_counts=True)[1].astype(np.float32)
                probabilities = category_count / category_count.sum()
                entropy = float(-(probabilities * np.log(probabilities + 1e-8)).sum())
                context_numeric[row_index] = [
                    np.log1p(length),
                    min(365, query_day - int(item_days[-1])) / 365.0,
                    entropy,
                    float(np.bincount(channels.astype(np.int32), minlength=3).argmax()) / 2.0,
                    float(np.median(prices)),
                ]
            customers[row_index] = customer
            days[row_index] = query_day
            origins[row_index] = origin
        return {
            "customer": torch.from_numpy(customers),
            "query_day": torch.from_numpy(days),
            "origin": torch.from_numpy(origins),
            "history": torch.from_numpy(history),
            "basket": torch.from_numpy(basket),
            "valid": torch.from_numpy(valid),
            "channel": torch.from_numpy(channel),
            "history_price": torch.from_numpy(history_price),
            "history_day": torch.from_numpy(history_day),
            "context_numeric": torch.from_numpy(context_numeric),
            "customer_features": torch.from_numpy(self.state.customer_features[customers]),
        }
