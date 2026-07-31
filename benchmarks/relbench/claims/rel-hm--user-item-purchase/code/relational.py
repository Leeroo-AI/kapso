from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RelationalBank:
    cutoff_day: int
    window_counts: np.ndarray
    global_score: np.ndarray
    global_top: np.ndarray
    cohort_top: np.ndarray
    family_top: np.ndarray
    transition_top: np.ndarray
    sampling_weights: np.ndarray


def top_indices(score: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(score))
    choice = np.argpartition(score, -count)[-count:]
    return choice[np.argsort(score[choice])[::-1]].astype(np.int32)


def build_bank(state, cutoff_day: int) -> RelationalBank:
    tx = state.transactions
    tx_day = ((tx.t_dat - pd.Timestamp("2019-09-01")) / pd.Timedelta(days=1)).to_numpy(
        np.int32
    )
    item = tx.article_id.to_numpy(np.int32)
    customer = tx.customer_id.to_numpy(np.int32)
    channel = tx.sales_channel_id.to_numpy(np.int8)
    windows = [1, 3, 7, 14, 28, 56]
    window_counts = np.zeros((len(windows), state.n_items), dtype=np.float32)
    for window_index, window in enumerate(windows):
        mask = (tx_day > cutoff_day - window) & (tx_day <= cutoff_day)
        window_counts[window_index] = np.bincount(
            item[mask], minlength=state.n_items
        ).astype(np.float32)
    global_score = (
        2.0 * window_counts[1]
        + window_counts[2]
        + 0.35 * window_counts[3]
        + 0.1 * window_counts[4]
    )
    global_top = top_indices(global_score, 400)
    cohort_top = np.empty((26, 3, 30), dtype=np.int32)
    global_probability = global_score / max(1.0, float(global_score.sum()))
    recent_mask = (tx_day > cutoff_day - 28) & (tx_day <= cutoff_day)
    recent_customer = customer[recent_mask]
    recent_item = item[recent_mask]
    recent_channel = channel[recent_mask]
    recent_age = state.customer_features[recent_customer, 0]
    recent = pd.DataFrame(
        {
            "age": recent_age,
            "channel": recent_channel,
            "item": recent_item,
        }
    )
    age_counts = recent.groupby(["age", "item"], observed=True).size()
    cohort_counts = recent.groupby(["age", "channel", "item"], observed=True).size()
    available_ages = set(age_counts.index.get_level_values(0).unique().tolist())
    available_cohorts = set(cohort_counts.index.droplevel(2).unique().tolist())
    for age in range(26):
        age_vector = np.zeros(state.n_items, dtype=np.float32)
        if age in available_ages:
            values = age_counts.loc[age]
            age_vector[values.index.to_numpy(np.int32)] = values.to_numpy(np.float32)
        age_probability = age_vector / max(1.0, float(age_vector.sum()))
        cohort_top[age, 0] = top_indices(age_probability + global_probability, 30)
        for selected_channel in (1, 2):
            vector = np.zeros(state.n_items, dtype=np.float32)
            key = (age, selected_channel)
            if key in available_cohorts:
                values = cohort_counts.loc[key]
                vector[values.index.to_numpy(np.int32)] = values.to_numpy(np.float32)
            probability = vector / max(1.0, float(vector.sum()))
            cohort_top[age, selected_channel] = top_indices(
                probability + global_probability, 30
            )
    product = state.item_features[:, 0]
    family_top = np.tile(global_top[:20], (state.n_items, 1)).astype(np.int32)
    family_frame = pd.DataFrame(
        {
            "item": np.arange(state.n_items, dtype=np.int32),
            "product": product,
            "score": global_score,
        }
    ).sort_values(["product", "score"], ascending=[True, False])
    for _, group in family_frame.groupby("product", sort=False):
        members = group.item.to_numpy(np.int32)
        ranked = members[:21]
        if len(members) < 2:
            continue
        for member in members:
            siblings = ranked[ranked != member][:20]
            if len(siblings):
                family_top[member, : len(siblings)] = siblings
    transition_top = np.tile(global_top[:25], (state.n_items, 1)).astype(np.int32)
    transition_mask = (tx_day > cutoff_day - 56) & (tx_day <= cutoff_day)
    transition_frame = pd.DataFrame(
        {
            "customer": customer[transition_mask],
            "day": tx_day[transition_mask],
            "item": item[transition_mask],
        }
    ).sort_values(["customer", "day"], kind="stable")
    previous_customer = transition_frame.customer.shift().to_numpy()
    previous_day = transition_frame.day.shift().to_numpy()
    previous_item = transition_frame.item.shift().to_numpy()
    current_customer = transition_frame.customer.to_numpy()
    current_day = transition_frame.day.to_numpy()
    current_item = transition_frame.item.to_numpy()
    valid = (
        (current_customer == previous_customer)
        & ((current_day - previous_day) <= 14)
        & (current_item != previous_item)
    )
    pairs = pd.DataFrame(
        {
            "source": previous_item[valid].astype(np.int32),
            "destination": current_item[valid].astype(np.int32),
        }
    )
    pair_counts = (
        pairs.groupby(["source", "destination"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["source", "count"], ascending=[True, False])
        .groupby("source", sort=False)
        .head(25)
    )
    for source, group in pair_counts.groupby("source", sort=False):
        destinations = group.destination.to_numpy(np.int32)
        transition_top[int(source), : len(destinations)] = destinations
    sampling_weights = np.power(window_counts[-1] + 0.1, 0.75)
    sampling_weights /= sampling_weights.sum()
    return RelationalBank(
        cutoff_day=cutoff_day,
        window_counts=window_counts,
        global_score=global_score.astype(np.float32),
        global_top=global_top,
        cohort_top=cohort_top,
        family_top=family_top,
        transition_top=transition_top,
        sampling_weights=sampling_weights.astype(np.float32),
    )


def unique_ranked(values):
    output = []
    seen = set()
    for value in values:
        value = int(value)
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def relational_channels(state, bank: RelationalBank, row):
    customer, query_day, _, items, item_days, channels, _, _ = row
    frequency = {}
    latest = {}
    for item, day in zip(items, item_days):
        item = int(item)
        frequency[item] = frequency.get(item, 0) + 1
        latest[item] = int(day)
    repeat = sorted(
        frequency,
        key=lambda value: (
            np.log1p(frequency[value]) + np.exp(-(query_day - latest[value]) / 30.0)
        ),
        reverse=True,
    )[:25]
    recent_unique = unique_ranked(reversed(items.tolist()))
    family_values = []
    transition_values = []
    for item in recent_unique:
        family_values.extend(bank.family_top[item].tolist())
        transition_values.extend(bank.transition_top[item].tolist())
    family = unique_ranked(family_values)[:20]
    transition = unique_ranked(transition_values)[:25]
    age = int(state.customer_features[int(customer), 0])
    if len(channels):
        dominant = int(np.bincount(channels.astype(np.int32), minlength=3).argmax())
    else:
        dominant = 0
    cohort = bank.cohort_top[age, dominant].tolist()
    return repeat, family, transition, cohort


def build_candidate_pool(state, bank, dataset, explore: np.ndarray):
    rows = len(dataset)
    width = 250
    pool = np.empty((rows, width), dtype=np.int32)
    ranks = np.full((rows, width, 5), 255, dtype=np.uint8)
    fallback = np.empty((rows, 12), dtype=np.int64)
    for row_index in range(rows):
        repeat, family, transition, cohort = relational_channels(
            state, bank, dataset[row_index]
        )
        sources = [
            explore[row_index].tolist(),
            repeat,
            family,
            transition,
            cohort,
        ]
        rank_maps = [
            {int(item): rank for rank, item in enumerate(source)}
            for source in sources
        ]
        robust = unique_ranked(repeat + family + transition + cohort)
        merged = unique_ranked(robust + sources[0] + bank.global_top.tolist())
        selected = merged[:width]
        if len(selected) < width:
            used = set(selected)
            for item in range(state.n_items):
                if item not in used:
                    selected.append(item)
                    used.add(item)
                    if len(selected) == width:
                        break
        pool[row_index] = selected
        for candidate_index, item in enumerate(selected):
            for source_index, rank_map in enumerate(rank_maps):
                if item in rank_map:
                    ranks[row_index, candidate_index, source_index] = min(
                        254, rank_map[item]
                    )
        rank_float = ranks[row_index].astype(np.float32)
        source_score = (
            0.025 / (rank_float[:, 1] + 1)
            + 0.08 / (rank_float[:, 2] + 1)
            + 0.05 / (rank_float[:, 3] + 1)
            + 0.8 / (rank_float[:, 4] + 1)
            + 0.35 / (rank_float[:, 0] + 1)
        )
        popularity = bank.global_score[pool[row_index]]
        source_score += 0.8 * popularity / max(1.0, float(bank.global_score.max()))
        top = np.argpartition(source_score, -12)[-12:]
        top = top[np.argsort(source_score[top])[::-1]]
        fallback[row_index] = pool[row_index, top]
    return pool, ranks, fallback


def relational_features(state, bank, dataset, pool, ranks):
    rows, width = pool.shape
    output = np.zeros((rows, width, 28), dtype=np.float32)
    output[:, :, :5] = np.where(
        ranks < 255, 1.0 / (ranks.astype(np.float32) + 1.0), 0.0
    )
    selected_counts = bank.window_counts[:, pool]
    output[:, :, 5:11] = np.log1p(selected_counts.transpose(1, 2, 0))
    output[:, :, 11] = np.log1p(
        (selected_counts[1] + 1.0) / (selected_counts[2] + 1.0)
    )
    output[:, :, 12] = np.log1p(
        (selected_counts[2] + 1.0) / (selected_counts[4] + 1.0)
    )
    metadata_columns = [0, 1, 2, 3, 4, 5, 7, 8, 9]
    for row_index in range(rows):
        row = dataset[row_index]
        _, query_day, origin, items, item_days, _, prices, _ = row
        candidates = pool[row_index]
        if len(items):
            unique_items, counts = np.unique(items, return_counts=True)
            frequency = dict(zip(unique_items.tolist(), counts.tolist()))
            latest = {}
            for item, day in zip(items, item_days):
                latest[int(item)] = int(day)
            output[row_index, :, 13] = [
                np.log1p(frequency.get(int(item), 0)) for item in candidates
            ]
            output[row_index, :, 14] = [
                np.exp(-(query_day - latest.get(int(item), query_day - 366)) / 30.0)
                for item in candidates
            ]
            candidate_metadata = state.item_features[candidates]
            history_metadata = state.item_features[items]
            for offset, column in enumerate(metadata_columns):
                output[row_index, :, 15 + offset] = np.isin(
                    candidate_metadata[:, column], history_metadata[:, column]
                )
            candidate_price = state.price_buckets[int(origin), candidates].astype(np.float32)
            history_price = float(
                np.median(state.price_buckets[int(origin), items].astype(np.float32))
            )
            output[row_index, :, 24] = np.abs(candidate_price - history_price) / 31.0
            output[row_index, :, 25] = np.log1p(len(items))
            output[row_index, :, 26] = min(365, query_day - int(item_days[-1])) / 365.0
            output[row_index, :, 27] = float(np.median(prices))
    return output
