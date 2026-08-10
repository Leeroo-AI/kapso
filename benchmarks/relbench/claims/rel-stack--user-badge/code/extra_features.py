from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy.stats import rankdata

from temporal_features import BASE_GROUPS, GROUPS, build_feature_events, cache_root, load_users, register_artifact


# Names

def extra_feature_names() -> list[str]:
    names = [
        "account_years",
        "anniversary_phase_sin",
        "anniversary_phase_cos",
        "days_to_anniversary",
        "anniversary_in_91d",
        "user_id_percentile",
        "creation_month_sin",
        "creation_month_cos",
    ]
    for group in range(BASE_GROUPS):
        names.extend(
            [
                f"acceleration_7_30_g{group}",
                f"acceleration_91_365_g{group}",
                f"lifetime_active_days_g{group}",
                f"active_days_91d_g{group}",
                f"active_days_365d_g{group}",
                f"events_per_active_day_g{group}",
            ]
        )
    names += [f"missing_badge_family_{group}" for group in range(10)]
    names += [
        "authored_lifetime_total",
        "received_lifetime_total",
        "authored_received_lifetime_ratio",
        "authored_91d_total",
        "received_91d_total",
        "authored_received_91d_ratio",
        "vote_per_post_lifetime",
        "response_per_post_lifetime",
        "comment_received_authored_ratio",
        "badge_count_recency_interaction",
        "active_days_91d_total",
        "active_days_365d_total",
    ]
    names += ["rank_days_to_anniversary", "rank_active_days_91d_total", "z_active_days_91d_total"]
    return names


# Build

def build_extra_features(frames: list[pd.DataFrame], compact: np.ndarray, compact_names: list[str], mapped_users: np.ndarray) -> np.memmap:
    path = cache_root() / "extra_v2.npy"
    names = extra_feature_names()
    total_rows = sum(len(frame) for frame in frames)
    if path.exists():
        matrix = np.load(path, mmap_mode="r")
        if matrix.shape == (total_rows, len(names)):
            return matrix
    users = load_users()
    user_ids = users["Id"].to_numpy(dtype=np.int64)
    events = build_feature_events(user_ids, pd.Timestamp(frames[0]["timestamp"].max()))
    event_time = np.asarray(events["time"])
    event_user = np.asarray(events["user"])
    event_group = np.asarray(events["group"])
    primary = event_group < BASE_GROUPS
    size = len(users) * BASE_GROUPS
    event_day = event_time[primary] // 86400
    encoded = event_day * size + event_user[primary].astype(np.int64) * BASE_GROUPS + event_group[primary]
    unique = np.unique(encoded)
    active_time = (unique // size) * 86400
    active_flat = unique % size
    del encoded, unique
    matrix = open_memmap(path, mode="w+", dtype=np.float16, shape=(total_rows, len(names)))
    combined_time = np.concatenate([frame["timestamp"].to_numpy(dtype="datetime64[s]").astype(np.int64) for frame in frames])
    creation = users["CreationDate"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    creation_month = users["CreationDate"].dt.month.to_numpy(dtype=np.float32)
    id_percentile = rankdata(user_ids, method="average").astype(np.float32) / len(user_ids)
    name_to_index = {name: index for index, name in enumerate(compact_names)}
    cumulative_days = np.zeros(size, dtype=np.float32)
    previous = 0
    authored_groups = (0, 1, 3, 5)
    received_groups = (2, 4, 6, 9)
    for cutoff in np.unique(combined_time):
        current = np.searchsorted(active_time, cutoff, side="right")
        cumulative_days += np.bincount(active_flat[previous:current], minlength=size).astype(np.float32)
        previous = current
        lo91 = np.searchsorted(active_time, cutoff - 91 * 86400, side="right")
        lo365 = np.searchsorted(active_time, cutoff - 365 * 86400, side="right")
        days91 = np.bincount(active_flat[lo91:current], minlength=size).reshape(len(users), BASE_GROUPS).astype(np.float32)
        days365 = np.bincount(active_flat[lo365:current], minlength=size).reshape(len(users), BASE_GROUPS).astype(np.float32)
        rows = np.flatnonzero(combined_time == cutoff)
        selected_users = mapped_users[rows]
        age = np.maximum(0, (cutoff - creation[selected_users]) / 86400).astype(np.float32)
        years = age / 365.2425
        phase = np.mod(years, 1.0)
        blocks = [
            np.floor(years),
            np.sin(2 * np.pi * phase),
            np.cos(2 * np.pi * phase),
            (1.0 - phase) * 365.2425,
            ((1.0 - phase) * 365.2425 <= 91).astype(np.float32),
            id_percentile[selected_users],
            np.sin(2 * np.pi * creation_month[selected_users] / 12.0),
            np.cos(2 * np.pi * creation_month[selected_users] / 12.0),
        ]
        lifetime_values = []
        recent_values = []
        for group in range(BASE_GROUPS):
            lifetime = np.expm1(np.asarray(compact[rows, name_to_index[f"lifetime_g{group}"]], dtype=np.float32))
            count7 = np.expm1(np.asarray(compact[rows, name_to_index[f"count_7d_g{group}"]], dtype=np.float32))
            count30 = np.expm1(np.asarray(compact[rows, name_to_index[f"count_30d_g{group}"]], dtype=np.float32))
            count91 = np.expm1(np.asarray(compact[rows, name_to_index[f"count_91d_g{group}"]], dtype=np.float32))
            count365 = np.expm1(np.asarray(compact[rows, name_to_index[f"count_365d_g{group}"]], dtype=np.float32))
            lifetime_days = cumulative_days.reshape(len(users), BASE_GROUPS)[selected_users, group]
            selected_days91 = days91[selected_users, group]
            selected_days365 = days365[selected_users, group]
            blocks.extend(
                [
                    np.log1p(count7) - np.log1p(count30 * 7.0 / 30.0),
                    np.log1p(count91) - np.log1p(count365 * 91.0 / 365.0),
                    np.log1p(lifetime_days),
                    np.log1p(selected_days91),
                    np.log1p(selected_days365),
                    np.log1p(lifetime / np.maximum(1, lifetime_days)),
                ]
            )
            lifetime_values.append(lifetime)
            recent_values.append(count91)
        for family in range(10):
            family_count = np.asarray(compact[rows, name_to_index[f"lifetime_g{23 + family}"]], dtype=np.float32)
            blocks.append((family_count == 0).astype(np.float32))
        authored_lifetime = sum(lifetime_values[group] for group in authored_groups)
        received_lifetime = sum(lifetime_values[group] for group in received_groups)
        authored_recent = sum(recent_values[group] for group in authored_groups)
        received_recent = sum(recent_values[group] for group in received_groups)
        badge_recency = np.asarray(compact[rows, name_to_index["recency_g7"]], dtype=np.float32)
        selected_days91_total = days91[selected_users].sum(1)
        selected_days365_total = days365[selected_users].sum(1)
        blocks.extend(
            [
                np.log1p(authored_lifetime),
                np.log1p(received_lifetime),
                np.log1p(received_lifetime) - np.log1p(authored_lifetime),
                np.log1p(authored_recent),
                np.log1p(received_recent),
                np.log1p(received_recent) - np.log1p(authored_recent),
                np.log1p(lifetime_values[6] / np.maximum(1, lifetime_values[0] + lifetime_values[1])),
                np.log1p(lifetime_values[2] / np.maximum(1, lifetime_values[0])),
                np.log1p(lifetime_values[4]) - np.log1p(lifetime_values[3]),
                np.log1p(lifetime_values[7]) / np.maximum(1, badge_recency),
                np.log1p(selected_days91_total),
                np.log1p(selected_days365_total),
            ]
        )
        values = np.column_stack(blocks).astype(np.float32)
        rank_anniversary = rankdata(values[:, 3], method="average").astype(np.float32) / len(rows)
        rank_active = rankdata(values[:, -2], method="average").astype(np.float32) / len(rows)
        active_column = values[:, -2]
        z_active = (active_column - active_column.mean()) / max(float(active_column.std()), 1e-5)
        matrix[rows] = np.column_stack([values, rank_anniversary, rank_active, z_active]).astype(np.float16)
        matrix.flush()
        print(f"[extra] origin={pd.to_datetime(cutoff, unit='s').date()} rows={len(rows)}", flush=True)
    register_artifact(
        "lane3 cadence and anniversary expansion",
        path,
        "Active-day cadence, anniversary phase, cross-channel ratios, and badge-family eligibility features.",
        "rel-stack-user-badge-lane3-extra-v2",
    )
    return np.load(path, mmap_mode="r")


def combine_expanded(compact: np.ndarray, extra: np.ndarray, memory: np.ndarray | None, name: str) -> np.memmap:
    path = cache_root() / name
    columns = compact.shape[1] + extra.shape[1] + (0 if memory is None else memory.shape[1])
    if path.exists():
        matrix = np.load(path, mmap_mode="r")
        if matrix.shape == (len(compact), columns):
            register_artifact(
                "lane3 expanded compact plus memory matrix",
                path,
                "Anniversary/cadence expansion concatenated with compact statistics and optional graph memory.",
                f"rel-stack-user-badge-lane3-{name}",
            )
            return matrix
    matrix = open_memmap(path, mode="w+", dtype=np.float16, shape=(len(compact), columns))
    for start in range(0, len(compact), 100000):
        stop = min(len(compact), start + 100000)
        matrix[start:stop, : compact.shape[1]] = compact[start:stop]
        matrix[start:stop, compact.shape[1] : compact.shape[1] + extra.shape[1]] = extra[start:stop]
        if memory is not None:
            matrix[start:stop, compact.shape[1] + extra.shape[1] :] = memory[start:stop]
    matrix.flush()
    register_artifact(
        "lane3 expanded compact plus memory matrix",
        path,
        "Anniversary/cadence expansion concatenated with compact statistics and optional graph memory.",
        f"rel-stack-user-badge-lane3-{name}",
    )
    return np.load(path, mmap_mode="r")
