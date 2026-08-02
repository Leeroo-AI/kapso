import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from content_factors import register_artifact


@dataclass
class HierarchicalState:
    mu: float
    mean_day: float
    global_slope: float
    user_effect: np.ndarray
    item_effect: np.ndarray
    category_effect: np.ndarray
    brand_effect: np.ndarray
    top_category_effect: np.ndarray
    title_effect: np.ndarray
    title_count: np.ndarray
    customer_name_effect: np.ndarray
    customer_name_count: np.ndarray
    customer_morphology_effect: np.ndarray
    user_last_rating: np.ndarray
    item_last_rating: np.ndarray
    user_recent3_rating: np.ndarray
    item_recent3_rating: np.ndarray
    user_recent10_rating: np.ndarray
    item_recent10_rating: np.ndarray
    user_last_label_day: np.ndarray
    item_last_label_day: np.ndarray
    user_recent3_count: np.ndarray
    item_recent3_count: np.ndarray
    user_recent10_count: np.ndarray
    item_recent10_count: np.ndarray
    user_rating_std: np.ndarray
    item_rating_std: np.ndarray
    user_low_rating_rate: np.ndarray
    item_low_rating_rate: np.ndarray
    user_five_rating_rate: np.ndarray
    item_five_rating_rate: np.ndarray
    user_verified_rate: np.ndarray
    item_verified_rate: np.ndarray
    user_count_90d: np.ndarray
    item_count_90d: np.ndarray
    user_count_365d: np.ndarray
    item_count_365d: np.ndarray
    user_count_730d: np.ndarray
    item_count_730d: np.ndarray
    user_first_label_day: np.ndarray
    item_first_label_day: np.ndarray
    verified_effect: np.ndarray
    user_recent_effect: np.ndarray
    item_recent_effect: np.ndarray
    category_recent_effect: np.ndarray
    brand_recent_effect: np.ndarray
    price_effect: np.ndarray
    category_verified_effect: np.ndarray
    user_verified_effect: np.ndarray
    item_verified_effect: np.ndarray
    user_category_key: np.ndarray
    user_category_effect: np.ndarray
    user_category_count: np.ndarray
    user_brand_key: np.ndarray
    user_brand_effect: np.ndarray
    user_brand_count: np.ndarray
    user_item_key: np.ndarray
    user_item_effect: np.ndarray
    user_item_count: np.ndarray
    user_count: np.ndarray
    item_count: np.ndarray
    cutoff_day: float


@dataclass
class CustomerMetadata:
    name_length: np.ndarray
    name_tokens: np.ndarray
    name_frequency: np.ndarray
    digit_flag: np.ndarray
    name_code: np.ndarray
    morphology_code: np.ndarray


def _pair_effect(first, second, second_cardinality, residual, shrinkage):
    key = first.astype(np.uint64) * np.uint64(second_cardinality) + second.astype(np.uint64)
    unique, inverse, count = np.unique(key, return_inverse=True, return_counts=True)
    total = np.bincount(inverse, weights=residual, minlength=len(unique))
    effect = total / (count + shrinkage)
    return unique, effect.astype(np.float32), count.astype(np.int32)


def _pair_lookup(key, effect, count, query):
    position = np.searchsorted(key, query)
    valid = position < len(key)
    valid[valid] &= key[position[valid]] == query[valid]
    output_effect = np.zeros(len(query), dtype=np.float32)
    output_count = np.zeros(len(query), dtype=np.int32)
    output_effect[valid] = effect[position[valid]]
    output_count[valid] = count[position[valid]]
    return output_effect, output_count


def _recent_state(entity, ratings, days, cardinality, fallback):
    order = np.lexsort((days, entity))
    sorted_entity = entity[order]
    sorted_rating = ratings[order]
    sorted_day = days[order]
    boundary = np.r_[0, np.flatnonzero(sorted_entity[1:] != sorted_entity[:-1]) + 1, len(order)]
    group_entity = sorted_entity[boundary[:-1]]
    start = boundary[:-1]
    end = boundary[1:]
    last_rating = np.full(cardinality, fallback, dtype=np.float32)
    recent_rating = np.full(cardinality, fallback, dtype=np.float32)
    recent10_rating = np.full(cardinality, fallback, dtype=np.float32)
    last_day = np.full(cardinality, -1, dtype=np.int32)
    recent_count = np.zeros(cardinality, dtype=np.int8)
    recent10_count = np.zeros(cardinality, dtype=np.int8)
    last_rating[group_entity] = sorted_rating[end - 1]
    last_day[group_entity] = sorted_day[end - 1]
    total = np.zeros(len(group_entity), dtype=np.float64)
    count = np.zeros(len(group_entity), dtype=np.int8)
    for lag in range(3):
        position = end - lag - 1
        valid = position >= start
        total[valid] += sorted_rating[position[valid]]
        count[valid] += 1
    recent_rating[group_entity] = (total / np.maximum(count, 1)).astype(np.float32)
    recent_count[group_entity] = count
    total.fill(0)
    count.fill(0)
    for lag in range(10):
        position = end - lag - 1
        valid = position >= start
        total[valid] += sorted_rating[position[valid]]
        count[valid] += 1
    recent10_rating[group_entity] = (total / np.maximum(count, 1)).astype(np.float32)
    recent10_count[group_entity] = count
    return last_rating, recent_rating, recent10_rating, last_day, recent_count, recent10_count


def safe_review_projection(db):
    review = db.table_dict["review"]
    expected = {"primary_key", "review_time", "customer_id", "product_id", "verified"}
    if set(review.df.columns) != expected or review.pkey_col != "primary_key":
        raise RuntimeError(f"unsafe review projection: {review.df.columns.tolist()}")
    primary = review.df["primary_key"].to_numpy(copy=False)
    if not np.array_equal(primary, np.arange(len(primary), dtype=primary.dtype)):
        raise RuntimeError("review primary keys do not match row positions")
    return review.df[["primary_key", "review_time", "customer_id", "product_id", "verified"]]


def customer_metadata(customer):
    names = customer["customer_name"].fillna("").astype(str)
    codes, _ = names.factorize(sort=True)
    frequency = np.bincount(codes)[codes].astype(np.float32)
    length = names.str.len().to_numpy(dtype=np.float32)
    tokens = names.str.count(r"\S+").to_numpy(dtype=np.float32)
    digits = names.str.contains(r"\d", regex=True).to_numpy(dtype=np.float32)
    morphology = np.clip(length.astype(np.int32), 0, 63) * 12 + np.clip(tokens.astype(np.int32), 0, 5) * 2 + digits.astype(np.int32)
    return CustomerMetadata(
        name_length=length,
        name_tokens=tokens,
        name_frequency=frequency,
        digit_flag=digits,
        name_code=codes.astype(np.int32),
        morphology_code=morphology.astype(np.int32),
    )


def _starts(sorted_group, sorted_day):
    n = len(sorted_group)
    index = np.arange(n, dtype=np.int32)
    group_boundary = np.empty(n, dtype=bool)
    group_boundary[0] = True
    group_boundary[1:] = sorted_group[1:] != sorted_group[:-1]
    time_boundary = group_boundary.copy()
    time_boundary[1:] |= sorted_day[1:] != sorted_day[:-1]
    group_start = np.maximum.accumulate(np.where(group_boundary, index, 0))
    time_start = np.maximum.accumulate(np.where(time_boundary, index, 0))
    return group_start, time_start


def _atomic_save(path, array):
    path = Path(path)
    temp = path.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(temp, array)
    os.replace(temp, path)


def temporal_training_features(review, label_ids, n_items, cache_dir, history_width=8):
    cache_dir = Path(cache_dir)
    prefix = cache_dir / f"rel_amazon_review_temporal_v3_{len(review)}_{len(label_ids)}"
    paths = {
        "history": prefix.with_name(prefix.name + "_history8.npy"),
        "user_count": prefix.with_name(prefix.name + "_user_count.npy"),
        "user_stale": prefix.with_name(prefix.name + "_user_stale.npy"),
        "item_count": prefix.with_name(prefix.name + "_item_count.npy"),
        "item_stale": prefix.with_name(prefix.name + "_item_stale.npy"),
        "pair_repeat": prefix.with_name(prefix.name + "_pair_repeat_all.npy"),
    }
    expected = {
        "history": (len(label_ids), history_width),
        "user_count": (len(label_ids),),
        "user_stale": (len(label_ids),),
        "item_count": (len(label_ids),),
        "item_stale": (len(label_ids),),
        "pair_repeat": (len(review),),
    }
    valid = all(path.exists() and np.load(path, mmap_mode="r").shape == expected[name] for name, path in paths.items())
    lock_path = prefix.with_name(prefix.name + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        valid = all(path.exists() and np.load(path, mmap_mode="r").shape == expected[name] for name, path in paths.items())
        if not valid:
            started = time.time()
            users = review["customer_id"].to_numpy(dtype=np.int32, copy=False)
            items = review["product_id"].to_numpy(dtype=np.int32, copy=False)
            days = ((review["review_time"].to_numpy(copy=False) - np.datetime64("2008-01-01")) / np.timedelta64(1, "D")).astype(np.int32)
            primary = review["primary_key"].to_numpy(dtype=np.int64, copy=False)
            n = len(review)
            order = np.lexsort((primary, days, users))
            sorted_user = users[order]
            sorted_day = days[order]
            sorted_item = items[order]
            group_start, time_start = _starts(sorted_user, sorted_day)
            position = np.empty(n, dtype=np.int32)
            position[order] = np.arange(n, dtype=np.int32)
            target_position = position[label_ids]
            prior_end = time_start[target_position]
            target_group_start = group_start[target_position]
            user_count = (prior_end - target_group_start).astype(np.int32)
            user_stale = np.zeros(len(label_ids), dtype=np.int32)
            warm = user_count > 0
            user_stale[warm] = days[label_ids[warm]] - sorted_day[prior_end[warm] - 1]
            history = np.full((len(label_ids), history_width), n_items, dtype=np.int32)
            for lag in range(history_width):
                source = prior_end - lag - 1
                available = source >= target_group_start
                history[available, lag] = sorted_item[source[available]]
            _atomic_save(paths["history"], history)
            _atomic_save(paths["user_count"], user_count)
            _atomic_save(paths["user_stale"], user_stale)
            del order, sorted_user, sorted_day, sorted_item, group_start, time_start, position, target_position, prior_end, target_group_start, history, user_count, user_stale
            order = np.lexsort((primary, days, items))
            sorted_item = items[order]
            sorted_day = days[order]
            group_start, time_start = _starts(sorted_item, sorted_day)
            position = np.empty(n, dtype=np.int32)
            position[order] = np.arange(n, dtype=np.int32)
            target_position = position[label_ids]
            item_count = (time_start[target_position] - group_start[target_position]).astype(np.int32)
            item_stale = np.zeros(len(label_ids), dtype=np.int32)
            warm = item_count > 0
            item_stale[warm] = days[label_ids[warm]] - sorted_day[time_start[target_position[warm]] - 1]
            _atomic_save(paths["item_count"], item_count)
            _atomic_save(paths["item_stale"], item_stale)
            del order, sorted_item, sorted_day, group_start, time_start, position, target_position, item_count, item_stale
            pair = users.astype(np.uint64) * np.uint64(n_items) + items.astype(np.uint64)
            order = np.lexsort((primary, days, pair))
            sorted_pair = pair[order]
            sorted_day = days[order]
            group_start, time_start = _starts(sorted_pair, sorted_day)
            repeat = np.empty(n, dtype=bool)
            repeat[order] = time_start > group_start
            _atomic_save(paths["pair_repeat"], repeat)
            print(f"[temporal] rows={n} labels={len(label_ids)} elapsed={time.time() - started:.1f}s", flush=True)
            for name, path in paths.items():
                register_artifact(
                    cache_dir,
                    f"Amazon review temporal {name}",
                    path,
                    f"Strictly-prior temporal {name} aligned to sanitized primary keys",
                    f"rel-amazon-review-temporal-v3-{len(review)}-{len(label_ids)}-{name}",
                    "Sort sanitized review identifiers by entity, date, and primary key; exclude all tied-date interactions",
                )
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return {name: np.load(path, mmap_mode="r") for name, path in paths.items()}


def temporal_pair_repeat(review, n_items, cache_dir):
    cache_dir = Path(cache_dir)
    path = cache_dir / f"rel_amazon_pair_repeat_v3_{len(review)}.npy"
    lock_path = cache_dir / f"rel_amazon_pair_repeat_v3_{len(review)}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        valid = path.exists() and np.load(path, mmap_mode="r").shape == (len(review),)
        if not valid:
            started = time.time()
            users = review["customer_id"].to_numpy(dtype=np.int32, copy=False)
            items = review["product_id"].to_numpy(dtype=np.int32, copy=False)
            days = ((review["review_time"].to_numpy(copy=False) - np.datetime64("2008-01-01")) / np.timedelta64(1, "D")).astype(np.int32)
            primary = review["primary_key"].to_numpy(dtype=np.int64, copy=False)
            pair = users.astype(np.uint64) * np.uint64(n_items) + items.astype(np.uint64)
            order = np.lexsort((primary, days, pair))
            group_start, time_start = _starts(pair[order], days[order])
            repeat = np.empty(len(review), dtype=bool)
            repeat[order] = time_start > group_start
            _atomic_save(path, repeat)
            print(f"[temporal] pair_repeat_rows={len(review)} elapsed={time.time() - started:.1f}s", flush=True)
            register_artifact(
                cache_dir,
                "Amazon full review repeat-pair flags",
                path,
                "Strictly-prior repeat-pair flag in full sanitized primary-key order",
                f"rel-amazon-pair-repeat-v3-{len(review)}",
                "Sort the full sanitized review projection by customer-product pair, date, and primary key",
            )
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return np.load(path, mmap_mode="r")


def fit_hierarchical(data, selected, metadata, customer, n_users, n_items):
    users = data["user"][selected]
    items = data["item"][selected]
    ratings = data["rating"][selected].astype(np.float64, copy=False)
    days = data["day"][selected].astype(np.float64, copy=False)
    verified = data["verified"][selected].astype(np.int8, copy=False)
    mu = float(ratings.mean())
    mean_day = float(days.mean())
    centered_day = days - mean_day
    global_slope = float(np.clip(np.dot(centered_day, ratings - mu) / (np.dot(centered_day, centered_day) + 1e7), -0.0005, 0.0005))
    residual = ratings - mu - global_slope * centered_day
    verified_count = np.bincount(verified, minlength=2)
    verified_effect = np.bincount(verified, weights=residual, minlength=2) / (verified_count + 1000.0)
    residual = residual - verified_effect[verified]
    category = metadata.category[items]
    category_count = np.bincount(category, minlength=int(metadata.category.max()) + 1)
    category_effect = np.bincount(category, weights=residual, minlength=len(category_count)) / (category_count + 80.0)
    residual = residual - category_effect[category]
    brand = metadata.brand[items]
    brand_count = np.bincount(brand, minlength=int(metadata.brand.max()) + 1)
    brand_effect = np.bincount(brand, weights=residual, minlength=len(brand_count)) / (brand_count + 60.0)
    residual = residual - brand_effect[brand]
    top_category = metadata.top_category[items]
    top_category_count = np.bincount(top_category, minlength=int(metadata.top_category.max()) + 1)
    top_category_effect = np.bincount(top_category, weights=residual, minlength=len(top_category_count)) / (top_category_count + 300.0)
    residual = residual - top_category_effect[top_category]
    item_count = np.bincount(items, minlength=n_items).astype(np.int32)
    item_effect = np.bincount(items, weights=residual, minlength=n_items) / (item_count + 20.0)
    residual = residual - item_effect[items]
    user_count = np.bincount(users, minlength=n_users).astype(np.int32)
    user_effect = np.bincount(users, weights=residual, minlength=n_users) / (user_count + 12.0)
    base_prediction = (
        mu
        + global_slope * centered_day
        + verified_effect[verified]
        + category_effect[category]
        + brand_effect[brand]
        + top_category_effect[top_category]
        + item_effect[items]
        + user_effect[users]
    )
    residual = ratings - base_prediction
    cutoff_day = float(days.max())
    recent = days >= cutoff_day - 548
    recent_users = users[recent]
    recent_items = items[recent]
    recent_category = category[recent]
    recent_brand = brand[recent]
    recent_residual = residual[recent]
    user_recent_count = np.bincount(recent_users, minlength=n_users)
    item_recent_count = np.bincount(recent_items, minlength=n_items)
    category_recent_count = np.bincount(recent_category, minlength=len(category_count))
    brand_recent_count = np.bincount(recent_brand, minlength=len(brand_count))
    user_recent_effect = np.bincount(recent_users, weights=recent_residual, minlength=n_users) / (user_recent_count + 20.0)
    item_recent_effect = np.bincount(recent_items, weights=recent_residual, minlength=n_items) / (item_recent_count + 30.0)
    category_recent_effect = np.bincount(recent_category, weights=recent_residual, minlength=len(category_count)) / (category_recent_count + 200.0)
    brand_recent_effect = np.bincount(recent_brand, weights=recent_residual, minlength=len(brand_count)) / (brand_recent_count + 100.0)
    price_bin = np.clip((metadata.log_price[items] * 8).astype(np.int32), 0, 63)
    price_count = np.bincount(price_bin, minlength=64)
    price_effect = np.bincount(price_bin, weights=residual, minlength=64) / (price_count + 200.0)
    category_verified = category * 2 + verified
    category_verified_count = np.bincount(category_verified, minlength=len(category_count) * 2)
    category_verified_effect = np.bincount(category_verified, weights=residual, minlength=len(category_count) * 2) / (category_verified_count + 200.0)
    title = metadata.title[items]
    title_count = np.bincount(title, minlength=int(metadata.title.max()) + 1)
    title_effect = np.bincount(title, weights=residual, minlength=len(title_count)) / (title_count + 35.0)
    name = customer.name_code[users]
    name_count = np.bincount(name, minlength=int(customer.name_code.max()) + 1)
    customer_name_effect = np.bincount(name, weights=residual, minlength=len(name_count)) / (name_count + 40.0)
    morphology = customer.morphology_code[users]
    morphology_count = np.bincount(morphology, minlength=int(customer.morphology_code.max()) + 1)
    customer_morphology_effect = np.bincount(morphology, weights=residual, minlength=len(morphology_count)) / (morphology_count + 300.0)
    user_verified_code = users * 2 + verified
    item_verified_code = items * 2 + verified
    user_verified_count = np.bincount(user_verified_code, minlength=n_users * 2)
    item_verified_count = np.bincount(item_verified_code, minlength=n_items * 2)
    user_verified_effect = np.bincount(user_verified_code, weights=residual, minlength=n_users * 2) / (user_verified_count + 20.0)
    item_verified_effect = np.bincount(item_verified_code, weights=residual, minlength=n_items * 2) / (item_verified_count + 30.0)
    user_category_key, user_category_effect, user_category_count = _pair_effect(users, category, len(category_count), residual, 18.0)
    user_brand_key, user_brand_effect, user_brand_count = _pair_effect(users, brand, len(brand_count), residual, 22.0)
    user_item_key, user_item_effect, user_item_count = _pair_effect(users, items, n_items, residual, 5.0)
    user_last_rating, user_recent3_rating, user_recent10_rating, user_last_label_day, user_recent3_count, user_recent10_count = _recent_state(users, ratings, days.astype(np.int32), n_users, mu)
    item_last_rating, item_recent3_rating, item_recent10_rating, item_last_label_day, item_recent3_count, item_recent10_count = _recent_state(items, ratings, days.astype(np.int32), n_items, mu)
    user_rating_mean = np.bincount(users, weights=ratings, minlength=n_users) / np.maximum(user_count, 1)
    item_rating_mean = np.bincount(items, weights=ratings, minlength=n_items) / np.maximum(item_count, 1)
    user_rating_std = np.sqrt(np.bincount(users, weights=(ratings - user_rating_mean[users]) ** 2, minlength=n_users) / np.maximum(user_count, 1))
    item_rating_std = np.sqrt(np.bincount(items, weights=(ratings - item_rating_mean[items]) ** 2, minlength=n_items) / np.maximum(item_count, 1))
    user_low_rating_rate = np.bincount(users, weights=ratings <= 2, minlength=n_users) / np.maximum(user_count, 1)
    item_low_rating_rate = np.bincount(items, weights=ratings <= 2, minlength=n_items) / np.maximum(item_count, 1)
    user_five_rating_rate = np.bincount(users, weights=ratings >= 5, minlength=n_users) / np.maximum(user_count, 1)
    item_five_rating_rate = np.bincount(items, weights=ratings >= 5, minlength=n_items) / np.maximum(item_count, 1)
    user_verified_rate = np.bincount(users, weights=verified, minlength=n_users) / np.maximum(user_count, 1)
    item_verified_rate = np.bincount(items, weights=verified, minlength=n_items) / np.maximum(item_count, 1)
    user_count_90d = np.bincount(users[days >= cutoff_day - 90], minlength=n_users).astype(np.int32)
    item_count_90d = np.bincount(items[days >= cutoff_day - 90], minlength=n_items).astype(np.int32)
    user_count_365d = np.bincount(users[days >= cutoff_day - 365], minlength=n_users).astype(np.int32)
    item_count_365d = np.bincount(items[days >= cutoff_day - 365], minlength=n_items).astype(np.int32)
    user_count_730d = np.bincount(users[days >= cutoff_day - 730], minlength=n_users).astype(np.int32)
    item_count_730d = np.bincount(items[days >= cutoff_day - 730], minlength=n_items).astype(np.int32)
    user_first_label_day = np.full(n_users, int(cutoff_day), dtype=np.int32)
    item_first_label_day = np.full(n_items, int(cutoff_day), dtype=np.int32)
    np.minimum.at(user_first_label_day, users, days.astype(np.int32))
    np.minimum.at(item_first_label_day, items, days.astype(np.int32))
    return HierarchicalState(
        mu=mu,
        mean_day=mean_day,
        global_slope=global_slope,
        user_effect=user_effect.astype(np.float32),
        item_effect=item_effect.astype(np.float32),
        category_effect=category_effect.astype(np.float32),
        brand_effect=brand_effect.astype(np.float32),
        top_category_effect=top_category_effect.astype(np.float32),
        title_effect=title_effect.astype(np.float32),
        title_count=title_count.astype(np.int32),
        customer_name_effect=customer_name_effect.astype(np.float32),
        customer_name_count=name_count.astype(np.int32),
        customer_morphology_effect=customer_morphology_effect.astype(np.float32),
        user_last_rating=user_last_rating,
        item_last_rating=item_last_rating,
        user_recent3_rating=user_recent3_rating,
        item_recent3_rating=item_recent3_rating,
        user_recent10_rating=user_recent10_rating,
        item_recent10_rating=item_recent10_rating,
        user_last_label_day=user_last_label_day,
        item_last_label_day=item_last_label_day,
        user_recent3_count=user_recent3_count,
        item_recent3_count=item_recent3_count,
        user_recent10_count=user_recent10_count,
        item_recent10_count=item_recent10_count,
        user_rating_std=user_rating_std.astype(np.float32),
        item_rating_std=item_rating_std.astype(np.float32),
        user_low_rating_rate=user_low_rating_rate.astype(np.float32),
        item_low_rating_rate=item_low_rating_rate.astype(np.float32),
        user_five_rating_rate=user_five_rating_rate.astype(np.float32),
        item_five_rating_rate=item_five_rating_rate.astype(np.float32),
        user_verified_rate=user_verified_rate.astype(np.float32),
        item_verified_rate=item_verified_rate.astype(np.float32),
        user_count_90d=user_count_90d,
        item_count_90d=item_count_90d,
        user_count_365d=user_count_365d,
        item_count_365d=item_count_365d,
        user_count_730d=user_count_730d,
        item_count_730d=item_count_730d,
        user_first_label_day=user_first_label_day,
        item_first_label_day=item_first_label_day,
        verified_effect=verified_effect.astype(np.float32),
        user_recent_effect=user_recent_effect.astype(np.float32),
        item_recent_effect=item_recent_effect.astype(np.float32),
        category_recent_effect=category_recent_effect.astype(np.float32),
        brand_recent_effect=brand_recent_effect.astype(np.float32),
        price_effect=price_effect.astype(np.float32),
        category_verified_effect=category_verified_effect.astype(np.float32),
        user_verified_effect=user_verified_effect.astype(np.float32),
        item_verified_effect=item_verified_effect.astype(np.float32),
        user_category_key=user_category_key,
        user_category_effect=user_category_effect,
        user_category_count=user_category_count,
        user_brand_key=user_brand_key,
        user_brand_effect=user_brand_effect,
        user_brand_count=user_brand_count,
        user_item_key=user_item_key,
        user_item_effect=user_item_effect,
        user_item_count=user_item_count,
        user_count=user_count,
        item_count=item_count,
        cutoff_day=cutoff_day,
    )


def hierarchical_prediction(hierarchy, users, items, days, verified, metadata, customer=None):
    components = hierarchical_components(hierarchy, users, items, days, verified, metadata, customer)
    recent_weight = np.exp(-np.maximum(days - hierarchy.cutoff_day, 0) / 730.0)
    return (
        components["global"]
        + components["user"]
        + components["item"]
        + components["category"]
        + components["brand"]
        + components["top_category"]
        + components["title"]
        + components["customer_name"]
        + components["customer_morphology"]
        + components["verified"]
        + components["price"]
        + components["category_verified"]
        + components["user_verified"]
        + components["item_verified"]
        + components["user_category"]
        + components["user_brand"]
        + components["user_item"]
        + recent_weight
        * (
            components["user_recent"]
            + components["item_recent"]
            + components["category_recent"]
            + components["brand_recent"]
        )
    ).astype(np.float32)


def hierarchical_components(hierarchy, users, items, days, verified, metadata, customer=None):
    category = metadata.category[items]
    brand = metadata.brand[items]
    verified_code = verified.astype(np.int8)
    price_bin = np.clip((metadata.log_price[items] * 8).astype(np.int32), 0, 63)
    user_category_query = users.astype(np.uint64) * np.uint64(len(hierarchy.category_effect)) + category.astype(np.uint64)
    user_brand_query = users.astype(np.uint64) * np.uint64(len(hierarchy.brand_effect)) + brand.astype(np.uint64)
    user_category_effect, user_category_count = _pair_lookup(
        hierarchy.user_category_key,
        hierarchy.user_category_effect,
        hierarchy.user_category_count,
        user_category_query,
    )
    user_brand_effect, user_brand_count = _pair_lookup(
        hierarchy.user_brand_key,
        hierarchy.user_brand_effect,
        hierarchy.user_brand_count,
        user_brand_query,
    )
    user_item_query = users.astype(np.uint64) * np.uint64(len(hierarchy.item_effect)) + items.astype(np.uint64)
    user_item_effect, user_item_count = _pair_lookup(
        hierarchy.user_item_key,
        hierarchy.user_item_effect,
        hierarchy.user_item_count,
        user_item_query,
    )
    if customer is None:
        customer_name = np.zeros(len(users), dtype=np.float32)
        customer_name_count = np.zeros(len(users), dtype=np.int32)
        customer_morphology = np.zeros(len(users), dtype=np.float32)
    else:
        customer_name_code = customer.name_code[users]
        customer_name = hierarchy.customer_name_effect[customer_name_code]
        customer_name_count = hierarchy.customer_name_count[customer_name_code]
        customer_morphology = hierarchy.customer_morphology_effect[customer.morphology_code[users]]
    return {
        "global": (hierarchy.mu + hierarchy.global_slope * (days - hierarchy.mean_day)).astype(np.float32),
        "user": hierarchy.user_effect[users],
        "item": hierarchy.item_effect[items],
        "category": hierarchy.category_effect[category],
        "brand": hierarchy.brand_effect[brand],
        "top_category": hierarchy.top_category_effect[metadata.top_category[items]],
        "title": hierarchy.title_effect[metadata.title[items]],
        "title_count": hierarchy.title_count[metadata.title[items]],
        "customer_name": customer_name,
        "customer_name_count": customer_name_count,
        "customer_morphology": customer_morphology,
        "user_last_rating": hierarchy.user_last_rating[users],
        "item_last_rating": hierarchy.item_last_rating[items],
        "user_recent3_rating": hierarchy.user_recent3_rating[users],
        "item_recent3_rating": hierarchy.item_recent3_rating[items],
        "user_recent10_rating": hierarchy.user_recent10_rating[users],
        "item_recent10_rating": hierarchy.item_recent10_rating[items],
        "user_label_staleness": np.where(hierarchy.user_last_label_day[users] >= 0, days - hierarchy.user_last_label_day[users], 0),
        "item_label_staleness": np.where(hierarchy.item_last_label_day[items] >= 0, days - hierarchy.item_last_label_day[items], 0),
        "user_recent3_count": hierarchy.user_recent3_count[users],
        "item_recent3_count": hierarchy.item_recent3_count[items],
        "user_recent10_count": hierarchy.user_recent10_count[users],
        "item_recent10_count": hierarchy.item_recent10_count[items],
        "user_rating_std": hierarchy.user_rating_std[users],
        "item_rating_std": hierarchy.item_rating_std[items],
        "user_low_rating_rate": hierarchy.user_low_rating_rate[users],
        "item_low_rating_rate": hierarchy.item_low_rating_rate[items],
        "user_five_rating_rate": hierarchy.user_five_rating_rate[users],
        "item_five_rating_rate": hierarchy.item_five_rating_rate[items],
        "user_verified_rate": hierarchy.user_verified_rate[users],
        "item_verified_rate": hierarchy.item_verified_rate[items],
        "user_count_90d": hierarchy.user_count_90d[users],
        "item_count_90d": hierarchy.item_count_90d[items],
        "user_count_365d": hierarchy.user_count_365d[users],
        "item_count_365d": hierarchy.item_count_365d[items],
        "user_count_730d": hierarchy.user_count_730d[users],
        "item_count_730d": hierarchy.item_count_730d[items],
        "user_label_age": np.where(hierarchy.user_count[users] > 0, days - hierarchy.user_first_label_day[users], 0),
        "item_label_age": np.where(hierarchy.item_count[items] > 0, days - hierarchy.item_first_label_day[items], 0),
        "user_label_span": np.where(hierarchy.user_count[users] > 0, hierarchy.user_last_label_day[users] - hierarchy.user_first_label_day[users], 0),
        "item_label_span": np.where(hierarchy.item_count[items] > 0, hierarchy.item_last_label_day[items] - hierarchy.item_first_label_day[items], 0),
        "verified": hierarchy.verified_effect[verified_code],
        "user_recent": hierarchy.user_recent_effect[users],
        "item_recent": hierarchy.item_recent_effect[items],
        "category_recent": hierarchy.category_recent_effect[category],
        "brand_recent": hierarchy.brand_recent_effect[brand],
        "price": hierarchy.price_effect[price_bin],
        "category_verified": hierarchy.category_verified_effect[category * 2 + verified_code],
        "user_verified": hierarchy.user_verified_effect[users * 2 + verified_code],
        "item_verified": hierarchy.item_verified_effect[items * 2 + verified_code],
        "user_category": user_category_effect,
        "user_category_count": user_category_count,
        "user_brand": user_brand_effect,
        "user_brand_count": user_brand_count,
        "user_item": user_item_effect,
        "user_item_count": user_item_count,
    }
