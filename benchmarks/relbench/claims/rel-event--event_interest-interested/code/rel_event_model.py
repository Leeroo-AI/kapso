import fcntl
import gc
import json
import math
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from relbench.datasets import get_dataset
from relbench.tasks import get_task


CACHE_VERSION = "lane1_conditional_choice_v7"
PRIOR_RATE = 0.27


def elapsed(started: float, phase: str) -> None:
    print(f"[{phase}] elapsed={time.time() - started:.1f}s")


def hash_category(values: pd.Series) -> np.ndarray:
    normalized = values.astype("string").fillna("__missing__")
    hashed = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype=np.uint64)
    return (hashed % np.uint64(1000003)).astype(np.float64)


def strict_prior_count(frame: pd.DataFrame, keys: list[str], name: str) -> pd.Series:
    grouped = frame.groupby(keys + ["timestamp"], dropna=False, sort=False).size().rename("_count").reset_index()
    grouped[name] = grouped.groupby(keys, dropna=False, sort=False)["_count"].cumsum() - grouped["_count"]
    lookup = frame[keys + ["timestamp"]].merge(grouped[keys + ["timestamp", name]], on=keys + ["timestamp"], how="left", sort=False)
    return lookup[name].astype(np.float64)


def resolve_split(table: pd.DataFrame, entity: pd.DataFrame, split: str) -> pd.DataFrame:
    identity = table[["timestamp", "primary_key"]].copy()
    identity["_row_id"] = np.arange(len(identity), dtype=np.int64)
    identity = identity.rename(columns={"timestamp": "_task_timestamp"})
    source = entity[["primary_key", "timestamp", "user", "event", "invited"]].copy()
    merged = identity.merge(source, on="primary_key", how="left", validate="one_to_one", sort=False)
    exact = merged["_task_timestamp"].to_numpy(dtype="datetime64[ns]") == merged["timestamp"].to_numpy(dtype="datetime64[ns]")
    if len(merged) != len(identity) or not bool(exact.all()):
        raise RuntimeError(f"{split} identity resolution failed: exact={int(exact.sum())}/{len(exact)}")
    merged["split"] = split
    merged["_source_pk"] = merged["primary_key"]
    merged = merged.drop(columns=["_task_timestamp"])
    return merged


def add_interest_features(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    ordered = rows.sort_values(["timestamp", "user", "_source_pk", "split", "_row_id"], kind="mergesort").copy()
    ordered["f_burst_size"] = ordered.groupby(["user", "timestamp"], dropna=False)["user"].transform("size").astype(np.float64)
    ordered["f_burst_position"] = ordered.groupby(["user", "timestamp"], dropna=False).cumcount().astype(np.float64)
    ordered["f_burst_reverse_position"] = ordered["f_burst_size"] - 1.0 - ordered["f_burst_position"]
    ordered["f_burst_position_fraction"] = (ordered["f_burst_position"] + 0.5) / ordered["f_burst_size"]
    ordered["f_burst_singleton"] = (ordered["f_burst_size"] == 1).astype(np.float64)
    ordered["f_burst_edge_distance"] = np.minimum(ordered["f_burst_position"], ordered["f_burst_reverse_position"])
    ordered["f_burst_center_distance"] = np.abs(ordered["f_burst_position_fraction"] - 0.5)
    ordered["f_burst_first"] = (ordered["f_burst_position"] == 0).astype(np.float64)
    ordered["f_burst_last"] = (ordered["f_burst_reverse_position"] == 0).astype(np.float64)
    for position in range(6):
        ordered[f"f_burst_position_is_{position}"] = (ordered["f_burst_position"] == position).astype(np.float64)
    for size in [1, 2, 4, 5, 6]:
        ordered[f"f_burst_size_is_{size}"] = (ordered["f_burst_size"] == size).astype(np.float64)
    bursts = ordered[["user", "timestamp"]].drop_duplicates().sort_values(["user", "timestamp"], kind="mergesort")
    bursts["f_user_session_number"] = bursts.groupby("user", dropna=False).cumcount().astype(np.float64)
    bursts["f_user_session_gap_days"] = bursts.groupby("user", dropna=False)["timestamp"].diff().dt.total_seconds() / 86400.0
    ordered = ordered.merge(bursts, on=["user", "timestamp"], how="left", sort=False)
    ordered["f_user_prior_rows"] = strict_prior_count(ordered, ["user"], "_user_prior").to_numpy()
    ordered["f_event_prior_rows"] = strict_prior_count(ordered, ["event"], "_event_prior").to_numpy()
    ordered["f_user_event_prior_rows"] = strict_prior_count(ordered, ["user", "event"], "_pair_prior").to_numpy()
    ordered["f_invited"] = pd.to_numeric(ordered["invited"], errors="coerce").astype(np.float64)
    ordered["f_event_missing"] = ordered["event"].isna().astype(np.float64)
    ordered["f_timestamp_days"] = (ordered["timestamp"] - pd.Timestamp("2012-01-01")).dt.total_seconds() / 86400.0
    ordered["f_timestamp_hour_sin"] = np.sin(2.0 * np.pi * (ordered["timestamp"].dt.hour + ordered["timestamp"].dt.minute / 60.0) / 24.0)
    ordered["f_timestamp_hour_cos"] = np.cos(2.0 * np.pi * (ordered["timestamp"].dt.hour + ordered["timestamp"].dt.minute / 60.0) / 24.0)
    ordered["f_timestamp_dow_sin"] = np.sin(2.0 * np.pi * ordered["timestamp"].dt.dayofweek / 7.0)
    ordered["f_timestamp_dow_cos"] = np.cos(2.0 * np.pi * ordered["timestamp"].dt.dayofweek / 7.0)
    feature_columns = [column for column in ordered.columns if column.startswith("f_")]
    result = result.merge(ordered[["split", "_row_id"] + feature_columns], on=["split", "_row_id"], how="left", validate="one_to_one", sort=False)
    return result


def add_user_features(rows: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    selected = users[["user_id", "locale", "birthyear", "gender", "joinedAt", "location", "timezone"]].copy()
    selected = selected.rename(columns={column: f"_user_{column}" for column in selected.columns if column != "user_id"})
    merged = rows.merge(selected, left_on="user", right_on="user_id", how="left", validate="many_to_one", sort=False)
    eligible = merged["_user_joinedAt"].notna() & (merged["_user_joinedAt"] <= merged["timestamp"])
    merged["f_user_missing"] = merged["user_id"].isna().astype(np.float64)
    merged["f_user_eligible"] = eligible.astype(np.float64)
    merged["f_membership_days"] = (merged["timestamp"] - merged["_user_joinedAt"]).dt.total_seconds() / 86400.0
    merged["f_user_age"] = merged["timestamp"].dt.year.astype(np.float64) - pd.to_numeric(merged["_user_birthyear"], errors="coerce")
    merged["f_user_birthyear"] = pd.to_numeric(merged["_user_birthyear"], errors="coerce")
    merged["f_user_timezone"] = pd.to_numeric(merged["_user_timezone"], errors="coerce")
    merged["f_user_locale_hash"] = hash_category(merged["_user_locale"])
    merged["f_user_gender_hash"] = hash_category(merged["_user_gender"])
    merged["f_user_location_hash"] = hash_category(merged["_user_location"])
    utc_minutes = merged["timestamp"].dt.hour * 60.0 + merged["timestamp"].dt.minute
    local_minutes = (utc_minutes + merged["f_user_timezone"].fillna(0.0)) % 1440.0
    merged["f_user_local_time_sin"] = np.sin(2.0 * np.pi * local_minutes / 1440.0)
    merged["f_user_local_time_cos"] = np.cos(2.0 * np.pi * local_minutes / 1440.0)
    mask_columns = [
        "f_membership_days", "f_user_age", "f_user_birthyear", "f_user_timezone",
        "f_user_locale_hash", "f_user_gender_hash", "f_user_location_hash",
        "f_user_local_time_sin", "f_user_local_time_cos",
    ]
    merged.loc[~eligible, mask_columns] = np.nan
    return merged.drop(columns=[column for column in merged.columns if column.startswith("_user_")] + ["user_id"])


def add_event_features(rows: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    content_columns = [f"c_{index}" for index in range(1, 101)] + ["c_other"]
    columns = ["event_id", "user_id", "start_time", "city", "state", "zip", "country", "lat", "lng"] + content_columns
    event_ids = set(pd.to_numeric(rows["event"], errors="coerce").dropna().astype(np.int64).tolist())
    selected = events.loc[events["event_id"].isin(event_ids), columns].copy()
    selected = selected.rename(columns={column: f"_event_{column}" for column in selected.columns if column != "event_id"})
    merged = rows.merge(selected, left_on="event", right_on="event_id", how="left", validate="many_to_one", sort=False)
    eligible = merged["_event_start_time"].notna() & (merged["_event_start_time"] <= merged["timestamp"])
    merged["f_event_metadata_missing"] = merged["event_id"].isna().astype(np.float64)
    merged["f_event_eligible"] = eligible.astype(np.float64)
    merged["f_event_age_days"] = (merged["timestamp"] - merged["_event_start_time"]).dt.total_seconds() / 86400.0
    merged["f_event_city_hash"] = hash_category(merged["_event_city"])
    merged["f_event_state_hash"] = hash_category(merged["_event_state"])
    merged["f_event_zip_hash"] = hash_category(merged["_event_zip"])
    merged["f_event_country_hash"] = hash_category(merged["_event_country"])
    merged["f_event_lat"] = pd.to_numeric(merged["_event_lat"], errors="coerce")
    merged["f_event_lng"] = pd.to_numeric(merged["_event_lng"], errors="coerce")
    merged["f_event_geo_missing"] = (merged["_event_lat"].isna() | merged["_event_lng"].isna()).astype(np.float64)
    content = merged[[f"_event_{column}" for column in content_columns]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    content_log = np.log1p(np.maximum(content, 0.0))
    total = content.sum(axis=1)
    probability = np.divide(content, total[:, None], out=np.zeros_like(content), where=total[:, None] > 0)
    entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum(axis=1)
    merged["f_event_content_total"] = np.log1p(total)
    merged["f_event_content_nnz"] = (content > 0).sum(axis=1).astype(np.float64)
    merged["f_event_content_max"] = np.log1p(content.max(axis=1))
    merged["f_event_content_argmax"] = content.argmax(axis=1).astype(np.float64)
    merged["f_event_content_entropy"] = entropy.astype(np.float64)
    merged["f_event_content_norm"] = np.sqrt((content_log * content_log).sum(axis=1)).astype(np.float64)
    generator = np.random.default_rng(73017)
    projection = generator.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(len(content_columns), 12)) / math.sqrt(len(content_columns))
    projected = content_log @ projection
    for index in range(projected.shape[1]):
        merged[f"f_event_content_factor_{index}"] = projected[:, index].astype(np.float64)
    mask_columns = [column for column in merged.columns if column.startswith("f_event_") and column not in {"f_event_metadata_missing", "f_event_eligible"}]
    merged.loc[~eligible, mask_columns] = np.nan
    drop_columns = [column for column in merged.columns if column.startswith("_event_")] + ["event_id"]
    return merged.drop(columns=drop_columns)


def cumulative_attendance(rows: pd.DataFrame, attendance: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=np.arange(len(rows)))
    result["f_att_user_count"] = 0.0
    result["f_att_user_invited"] = 0.0
    result["f_att_user_yes"] = 0.0
    result["f_att_user_maybe"] = 0.0
    result["f_att_user_no"] = 0.0
    result["f_att_user_gap_days"] = np.nan
    if len(attendance) == 0:
        return result
    source = attendance.sort_values(["user_id", "start_time"], kind="mergesort")
    query_groups = rows.reset_index(names="_query_index").groupby("user", dropna=False, sort=False)
    source_groups = {key: value for key, value in source.groupby("user_id", dropna=False, sort=False)}
    statuses = ["invited", "yes", "maybe", "no"]
    for key, query in query_groups:
        if pd.isna(key) or key not in source_groups:
            continue
        history = source_groups[key]
        times = history["start_time"].to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(times, query["timestamp"].to_numpy(dtype="datetime64[ns]"), side="right")
        indices = query["_query_index"].to_numpy(dtype=np.int64)
        result.loc[indices, "f_att_user_count"] = positions.astype(np.float64)
        status_values = history["status"].astype(str).to_numpy()
        for status in statuses:
            prefix = np.concatenate([[0], np.cumsum(status_values == status)])
            result.loc[indices, f"f_att_user_{status}"] = prefix[positions].astype(np.float64)
        valid = positions > 0
        if valid.any():
            prior_times = times[positions[valid] - 1]
            query_times = query["timestamp"].to_numpy(dtype="datetime64[ns]")[valid]
            gaps = (query_times - prior_times) / np.timedelta64(1, "D")
            result.loc[indices[valid], "f_att_user_gap_days"] = gaps.astype(np.float64)
    return result


def add_attendance_features(rows: pd.DataFrame, attendees: pd.DataFrame) -> pd.DataFrame:
    event_ids = set(pd.to_numeric(rows["event"], errors="coerce").dropna().astype(np.int64).tolist())
    target = attendees.loc[attendees["event"].isin(event_ids), ["event", "status", "start_time"]].copy()
    counts = target.groupby(["event", "status"], dropna=False).size().unstack(fill_value=0)
    for status in ["invited", "yes", "maybe", "no"]:
        if status not in counts.columns:
            counts[status] = 0
    counts = counts[["invited", "yes", "maybe", "no"]].rename(columns={status: f"_att_{status}" for status in ["invited", "yes", "maybe", "no"]})
    times = target.groupby("event", dropna=False)["start_time"].min().rename("_att_start_time")
    aggregate = counts.join(times).reset_index()
    merged = rows.merge(aggregate, on="event", how="left", validate="many_to_one", sort=False)
    eligible = merged["_att_start_time"].notna() & (merged["_att_start_time"] <= merged["timestamp"])
    total = sum(pd.to_numeric(merged[f"_att_{status}"], errors="coerce").fillna(0.0) for status in ["invited", "yes", "maybe", "no"])
    merged["f_att_event_count"] = np.log1p(total)
    for status in ["invited", "yes", "maybe", "no"]:
        values = pd.to_numeric(merged[f"_att_{status}"], errors="coerce").fillna(0.0)
        merged[f"f_att_event_{status}_rate"] = (values + 1.0) / (total + 4.0)
    event_feature_columns = [column for column in merged.columns if column.startswith("f_att_event_")]
    merged.loc[~eligible, event_feature_columns] = np.nan
    query_users = set(pd.to_numeric(rows["user"], errors="coerce").dropna().astype(np.int64).tolist())
    user_attendance = attendees.loc[attendees["user_id"].notna() & attendees["user_id"].isin(query_users), ["user_id", "status", "start_time"]].copy()
    cumulative = cumulative_attendance(rows, user_attendance)
    for column in cumulative.columns:
        merged[column] = cumulative[column].to_numpy()
    return merged.drop(columns=[column for column in merged.columns if column.startswith("_att_")])


def add_friend_features(rows: pd.DataFrame, friendships: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, set[int]]]:
    degree = friendships["user"].value_counts(dropna=True).rename("f_friend_row_degree")
    known = friendships.loc[friendships["user"].notna() & friendships["friend"].notna(), ["user", "friend"]].copy()
    known["user"] = known["user"].astype(np.int64)
    known["friend"] = known["friend"].astype(np.int64)
    known_degree = known.groupby("user").size().rename("f_friend_known_degree")
    friend_degree = known["friend"].map(degree).astype(np.float64)
    known["_friend_degree"] = friend_degree
    factors = pd.DataFrame({"user": known["user"]})
    friend_values = known["friend"].to_numpy(dtype=np.float64)
    primes = [1009.0, 2027.0, 4001.0, 8011.0]
    for index, prime in enumerate(primes):
        factors[f"f_friend_community_sin_{index}"] = np.sin(2.0 * np.pi * (friend_values % prime) / prime)
        factors[f"f_friend_community_cos_{index}"] = np.cos(2.0 * np.pi * (friend_values % prime) / prime)
    factors["f_friend_neighbor_degree_mean"] = known["_friend_degree"].to_numpy()
    community = factors.groupby("user").mean()
    features = pd.concat([degree, known_degree, community], axis=1).reset_index().rename(columns={"index": "user"})
    merged = rows.merge(features, on="user", how="left", validate="many_to_one", sort=False)
    friend_sets = {int(key): set(value["friend"].astype(np.int64).tolist()) for key, value in known.groupby("user", sort=False)}
    return merged, friend_sets


def register_artifact(cache_dir: Path, artifact_path: Path) -> None:
    registry = cache_dir / "artifacts.json"
    lock_path = cache_dir / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            entries = json.loads(registry.read_text()) if registry.exists() else []
        except json.JSONDecodeError:
            entries = []
        relative = artifact_path.relative_to(cache_dir).as_posix()
        if not any(entry.get("content_key") == CACHE_VERSION for entry in entries):
            entries.append({
                "name": "lane1 conditional-choice static features",
                "path": relative,
                "description": "All-table temporally censored static features and target-user known-friend sets",
                "content_key": CACHE_VERSION,
                "rebuild_hint": "Delete the pickle and rerun main.py after changing CACHE_VERSION",
            })
            temporary = cache_dir / f"artifacts.{os.getpid()}.tmp"
            temporary.write_text(json.dumps(entries, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def build_static_features(dataset, task, cache_dir: Path, started: float) -> tuple[pd.DataFrame, dict[int, set[int]]]:
    cache_path = cache_dir / f"{CACHE_VERSION}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            artifact = pickle.load(handle)
        rows = artifact["rows"]
        for split in ["train", "val", "test"]:
            table = task.get_table(split).df
            cached = rows.loc[rows["split"] == split].sort_values("_row_id")
            exact = cached["timestamp"].to_numpy(dtype="datetime64[ns]") == table["timestamp"].to_numpy(dtype="datetime64[ns]")
            if len(cached) != len(table) or not bool(exact.all()):
                raise RuntimeError(f"cached feature identity mismatch for {split}")
        register_artifact(cache_dir, cache_path)
        print(f"[feature construction] reused {cache_path.name} rows={len(rows)} features={sum(column.startswith('f_') for column in rows.columns)}")
        elapsed(started, "feature construction")
        return rows, artifact["friend_sets"]
    cutoff_db = dataset.get_db(upto_test_timestamp=True)
    cutoff_entity = cutoff_db.table_dict["event_interest"].df
    train_rows = resolve_split(task.get_table("train").df, cutoff_entity, "train")
    val_rows = resolve_split(task.get_table("val").df, cutoff_entity, "val")
    del cutoff_entity, cutoff_db
    gc.collect()
    full_db = dataset.get_db(upto_test_timestamp=False)
    test_rows = resolve_split(task.get_table("test").df, full_db.table_dict["event_interest"].df, "test")
    rows = pd.concat([train_rows, val_rows, test_rows], ignore_index=True)
    rows = add_interest_features(rows)
    rows = add_user_features(rows, full_db.table_dict["users"].df)
    rows = add_event_features(rows, full_db.table_dict["events"].df)
    rows = add_attendance_features(rows, full_db.table_dict["event_attendees"].df)
    rows, friend_sets = add_friend_features(rows, full_db.table_dict["user_friends"].df)
    feature_columns = [column for column in rows.columns if column.startswith("f_")]
    rows[feature_columns] = rows[feature_columns].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    artifact = {"rows": rows, "friend_sets": friend_sets}
    temporary = cache_dir / f"{CACHE_VERSION}.{os.getpid()}.tmp"
    with temporary.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, cache_path)
    register_artifact(cache_dir, cache_path)
    del full_db
    gc.collect()
    print(f"[feature construction] built rows={len(rows)} features={len(feature_columns)} cache={cache_path.name}")
    elapsed(started, "feature construction")
    return rows, friend_sets


def attach_labels(rows: pd.DataFrame, task, split: str) -> pd.DataFrame:
    subset = rows.loc[rows["split"] == split].sort_values("_row_id", kind="mergesort").copy()
    table = task.get_table(split).df
    exact = subset["timestamp"].to_numpy(dtype="datetime64[ns]") == table["timestamp"].to_numpy(dtype="datetime64[ns]")
    if len(subset) != len(table) or not bool(exact.all()):
        raise RuntimeError(f"label alignment failed for {split}")
    if "interested" in table.columns:
        subset["label"] = table["interested"].to_numpy(dtype=np.int64)
    return subset


def smoothed_rate(positive: float, count: float) -> float:
    return (positive + 4.0 * PRIOR_RATE) / (count + 4.0)


def history_features(query: pd.DataFrame, history: pd.DataFrame, friend_sets: dict[int, set[int]]) -> pd.DataFrame:
    names = [
        "h_global_count", "h_global_rate", "h_user_count", "h_user_positive", "h_user_rate", "h_user_gap_days",
        "h_event_count", "h_event_positive", "h_event_rate", "h_event_gap_days", "h_pair_count", "h_pair_rate",
        "h_user_invited_count", "h_user_invited_rate", "h_event_invited_count", "h_event_invited_rate",
        "h_user_burst_count", "h_user_burst_any_rate", "h_friend_count", "h_friend_positive", "h_friend_rate",
        "h_position_count", "h_position_rate", "h_size_position_count", "h_size_position_rate",
        "h_size_row_count", "h_size_row_rate", "h_size_burst_count", "h_size_burst_any_rate",
    ]
    positions = {name: index for index, name in enumerate(names)}
    output = np.zeros((len(query), len(names)), dtype=np.float64)
    output[:, positions["h_user_gap_days"]] = np.nan
    output[:, positions["h_event_gap_days"]] = np.nan
    output_positions = {index: position for position, index in enumerate(query.index)}
    ordered_history = history.sort_values(["timestamp", "_row_id"], kind="mergesort")
    history_records = ordered_history[["timestamp", "user", "event", "invited", "label", "f_burst_size", "f_burst_position"]].to_dict("records")
    burst_history = ordered_history.groupby(["user", "timestamp"], dropna=False, sort=False).agg(
        label=("label", "max"), burst_size=("f_burst_size", "first"),
    ).reset_index().sort_values("timestamp", kind="mergesort")
    burst_records = burst_history.to_dict("records")
    user_count = defaultdict(int)
    user_positive = defaultdict(float)
    event_count = defaultdict(int)
    event_positive = defaultdict(float)
    pair_count = defaultdict(int)
    pair_positive = defaultdict(float)
    user_invited_count = defaultdict(int)
    user_invited_positive = defaultdict(float)
    event_invited_count = defaultdict(int)
    event_invited_positive = defaultdict(float)
    user_burst_count = defaultdict(int)
    user_burst_any = defaultdict(float)
    position_count = defaultdict(int)
    position_positive = defaultdict(float)
    size_position_count = defaultdict(int)
    size_position_positive = defaultdict(float)
    size_row_count = defaultdict(int)
    size_row_positive = defaultdict(float)
    size_burst_count = defaultdict(int)
    size_burst_any = defaultdict(float)
    user_last = {}
    event_last = {}
    event_responders = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    global_count = 0
    global_positive = 0.0
    pointer = 0
    burst_pointer = 0
    query_groups = query.sort_values("timestamp", kind="mergesort").groupby("timestamp", sort=True)
    for timestamp, current in query_groups:
        while pointer < len(history_records) and history_records[pointer]["timestamp"] < timestamp:
            record = history_records[pointer]
            user = record["user"]
            event = record["event"]
            label = float(record["label"])
            global_count += 1
            global_positive += label
            user_count[user] += 1
            user_positive[user] += label
            user_last[user] = record["timestamp"]
            if not pd.isna(event):
                event_count[event] += 1
                event_positive[event] += label
                pair_count[(user, event)] += 1
                pair_positive[(user, event)] += label
                event_last[event] = record["timestamp"]
                event_responders[event][user][0] += 1
                event_responders[event][user][1] += label
            if int(record["invited"]) == 1:
                user_invited_count[user] += 1
                user_invited_positive[user] += label
                if not pd.isna(event):
                    event_invited_count[event] += 1
                    event_invited_positive[event] += label
            size = int(record["f_burst_size"])
            position = int(record["f_burst_position"])
            position_count[position] += 1
            position_positive[position] += label
            size_position_count[(size, position)] += 1
            size_position_positive[(size, position)] += label
            size_row_count[size] += 1
            size_row_positive[size] += label
            pointer += 1
        while burst_pointer < len(burst_records) and burst_records[burst_pointer]["timestamp"] < timestamp:
            record = burst_records[burst_pointer]
            user_burst_count[record["user"]] += 1
            user_burst_any[record["user"]] += float(record["label"])
            size = int(record["burst_size"])
            size_burst_count[size] += 1
            size_burst_any[size] += float(record["label"])
            burst_pointer += 1
        for index, record in current.iterrows():
            output_position = output_positions[index]
            user = record["user"]
            event = record["event"]
            size = int(record["f_burst_size"])
            position = int(record["f_burst_position"])
            uc = user_count[user]
            up = user_positive[user]
            ec = event_count[event]
            ep = event_positive[event]
            pc = pair_count[(user, event)]
            pp = pair_positive[(user, event)]
            uic = user_invited_count[user]
            uip = user_invited_positive[user]
            eic = event_invited_count[event]
            eip = event_invited_positive[event]
            ubc = user_burst_count[user]
            uba = user_burst_any[user]
            pcnt = position_count[position]
            ppos = position_positive[position]
            spcnt = size_position_count[(size, position)]
            sppos = size_position_positive[(size, position)]
            srcnt = size_row_count[size]
            srpos = size_row_positive[size]
            sbcnt = size_burst_count[size]
            sbany = size_burst_any[size]
            values = {
                "h_global_count": math.log1p(global_count), "h_global_rate": smoothed_rate(global_positive, global_count),
                "h_user_count": math.log1p(uc), "h_user_positive": math.log1p(up), "h_user_rate": smoothed_rate(up, uc),
                "h_event_count": math.log1p(ec), "h_event_positive": math.log1p(ep), "h_event_rate": smoothed_rate(ep, ec),
                "h_pair_count": math.log1p(pc), "h_pair_rate": smoothed_rate(pp, pc),
                "h_user_invited_count": math.log1p(uic), "h_user_invited_rate": smoothed_rate(uip, uic),
                "h_event_invited_count": math.log1p(eic), "h_event_invited_rate": smoothed_rate(eip, eic),
                "h_user_burst_count": math.log1p(ubc), "h_user_burst_any_rate": smoothed_rate(uba, ubc),
                "h_position_count": math.log1p(pcnt), "h_position_rate": smoothed_rate(ppos, pcnt),
                "h_size_position_count": math.log1p(spcnt), "h_size_position_rate": smoothed_rate(sppos, spcnt),
                "h_size_row_count": math.log1p(srcnt), "h_size_row_rate": smoothed_rate(srpos, srcnt),
                "h_size_burst_count": math.log1p(sbcnt), "h_size_burst_any_rate": smoothed_rate(sbany, sbcnt),
            }
            for name, value in values.items():
                output[output_position, positions[name]] = value
            if user in user_last:
                output[output_position, positions["h_user_gap_days"]] = (timestamp - user_last[user]).total_seconds() / 86400.0
            if event in event_last:
                output[output_position, positions["h_event_gap_days"]] = (timestamp - event_last[event]).total_seconds() / 86400.0
            if not pd.isna(user) and not pd.isna(event):
                friends = friend_sets.get(int(user), set())
                responders = event_responders.get(event, {})
                friend_count = 0
                friend_positive = 0.0
                for friend in friends:
                    values = responders.get(friend)
                    if values is not None:
                        friend_count += values[0]
                        friend_positive += values[1]
                output[output_position, positions["h_friend_count"]] = math.log1p(friend_count)
                output[output_position, positions["h_friend_positive"]] = math.log1p(friend_positive)
                output[output_position, positions["h_friend_rate"]] = smoothed_rate(friend_positive, friend_count)
    return pd.DataFrame(output.astype(np.float32), index=query.index, columns=names)


def make_row_features(query: pd.DataFrame, history: pd.DataFrame, friend_sets: dict[int, set[int]]) -> pd.DataFrame:
    static_columns = [column for column in query.columns if column.startswith("f_")]
    features = query[static_columns].copy()
    dynamic = history_features(query, history, friend_sets)
    features = pd.concat([features, dynamic], axis=1)
    rank_candidates = [
        "f_burst_position", "f_burst_reverse_position", "f_event_prior_rows", "f_user_event_prior_rows",
        "f_event_age_days", "f_event_lat", "f_event_lng", "f_event_content_total", "f_event_content_nnz",
        "f_event_content_max", "f_event_content_entropy", "f_event_content_norm", "f_att_event_count",
        "f_att_event_yes_rate", "f_att_event_maybe_rate", "f_att_event_no_rate", "f_friend_row_degree",
        "f_friend_known_degree", "h_user_rate", "h_event_rate", "h_pair_rate", "h_friend_rate",
        "h_position_rate", "h_size_position_rate",
    ] + [f"f_event_content_factor_{index}" for index in range(12)]
    keys = pd.MultiIndex.from_arrays([query["user"].to_numpy(), query["timestamp"].to_numpy()])
    group_codes = pd.factorize(keys, sort=False)[0]
    grouping = pd.Series(group_codes, index=query.index)
    for column in rank_candidates:
        if column not in features.columns:
            continue
        features[f"r_{column}"] = features[column].groupby(grouping).rank(method="average", pct=True)
        features[f"d_{column}"] = features[column] - features[column].groupby(grouping).transform("mean")
    return features.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def align_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    aligned = frame.reindex(columns=columns)
    return aligned.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def burst_frame(features: pd.DataFrame, rows: pd.DataFrame, labels: np.ndarray | None = None) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray, np.ndarray]:
    keys = pd.MultiIndex.from_arrays([rows["user"].to_numpy(), rows["timestamp"].to_numpy()])
    codes, uniques = pd.factorize(keys, sort=False)
    work = features.copy()
    work["_burst_code"] = codes
    invited = pd.to_numeric(rows["invited"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    missing_event = rows["event"].isna().to_numpy(dtype=np.float64)
    context = pd.DataFrame({"_burst_code": codes, "_invited": invited, "_missing_event": missing_event})
    base = context.groupby("_burst_code", sort=True).agg(
        b_size=("_burst_code", "size"),
        b_invited_any=("_invited", "max"),
        b_invited_count=("_invited", "sum"),
        b_missing_event_fraction=("_missing_event", "mean"),
    )
    base["b_singleton"] = (base["b_size"] == 1).astype(np.float64)
    summary_candidates = [
        "f_user_session_number", "f_user_session_gap_days", "f_user_prior_rows", "f_event_prior_rows",
        "f_event_age_days", "f_user_age", "f_membership_days", "f_friend_row_degree", "f_friend_known_degree",
        "f_event_content_total", "f_event_content_nnz", "f_event_content_entropy", "f_att_event_count",
        "f_att_user_count", "h_user_count", "h_user_rate", "h_event_count", "h_event_rate", "h_friend_count", "h_friend_rate",
        "h_size_row_rate", "h_size_burst_count", "h_size_burst_any_rate",
    ]
    available = [column for column in summary_candidates if column in work.columns]
    summary = work.groupby("_burst_code", sort=True)[available].agg(["mean", "max", "min"])
    summary.columns = [f"b_{column}_{statistic}" for column, statistic in summary.columns]
    output = base.join(summary).reset_index(drop=True).astype(np.float32)
    burst_labels = None
    if labels is not None:
        label_frame = pd.DataFrame({"_burst_code": codes, "label": labels})
        burst_labels = label_frame.groupby("_burst_code", sort=True)["label"].max().to_numpy(dtype=np.int64)
    return output, burst_labels, codes, np.asarray(uniques)


class ConditionalNetwork(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(1)


@dataclass
class ConditionalScorer:
    network: ConditionalNetwork
    fill: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    device: str

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        values = features.to_numpy(dtype=np.float32)
        values = np.where(np.isfinite(values), values, self.fill)
        return ((values - self.mean) / self.scale).astype(np.float32)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        values = torch.from_numpy(self.transform(features)).to(self.device)
        self.network.eval()
        with torch.no_grad():
            return self.network(values).detach().cpu().numpy().astype(np.float64)


def fit_conditional_scorer(features: pd.DataFrame, labels: np.ndarray, groups: np.ndarray, epochs: int, regularization: float) -> ConditionalScorer:
    values = features.to_numpy(dtype=np.float32)
    fill = np.nanmedian(values, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0).astype(np.float32)
    values = np.where(np.isfinite(values), values, fill)
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale = np.where(scale > 1e-4, scale, 1.0).astype(np.float32)
    values = ((values - mean) / scale).astype(np.float32)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337)
    network = ConditionalNetwork(values.shape[1]).to(device)
    x_tensor = torch.from_numpy(values).to(device)
    y_tensor = torch.from_numpy(labels.astype(np.float32)).to(device)
    group_tensor = torch.from_numpy(groups.astype(np.int64)).to(device)
    group_count = int(groups.max()) + 1
    optimizer = torch.optim.AdamW(network.parameters(), lr=5e-4, weight_decay=regularization)
    for _ in range(epochs):
        network.train()
        optimizer.zero_grad(set_to_none=True)
        logits = network(x_tensor)
        maxima = torch.full((group_count,), -torch.inf, device=device)
        maxima.scatter_reduce_(0, group_tensor, logits.detach(), reduce="amax", include_self=True)
        sums = torch.zeros(group_count, device=device)
        sums.scatter_add_(0, group_tensor, torch.exp(logits - maxima[group_tensor]))
        log_normalizers = maxima + torch.log(sums.clamp_min(1e-12))
        positives = torch.zeros(group_count, device=device)
        positives.scatter_add_(0, group_tensor, logits * y_tensor)
        loss = (log_normalizers - positives).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
        optimizer.step()
    return ConditionalScorer(network=network, fill=fill, mean=mean, scale=scale, device=device)


@dataclass
class ChoiceModels:
    stage_a: lgb.LGBMClassifier
    stage_b: lgb.LGBMRanker
    conditional: ConditionalScorer
    guard: lgb.LGBMClassifier
    row_columns: list[str]
    rank_columns: list[str]
    burst_columns: list[str]
    rank_scale: float
    conditional_scale: float


def center_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"value": values, "group": groups})
    return values - frame.groupby("group", sort=False)["value"].transform("mean").to_numpy()


def conditional_columns(features: pd.DataFrame) -> list[str]:
    exact = {
        "f_invited", "f_event_missing", "f_burst_position", "f_burst_reverse_position",
        "f_burst_position_fraction", "f_burst_edge_distance", "f_burst_center_distance",
        "f_burst_first", "f_burst_last", "f_event_prior_rows", "f_user_event_prior_rows",
        "h_position_count", "h_position_rate", "h_size_position_count", "h_size_position_rate",
    }
    prefixes = (
        "f_burst_position_is_", "f_event_", "f_att_event_", "h_event_", "h_pair_", "h_friend_", "r_", "d_",
    )
    return [column for column in features.columns if column in exact or column.startswith(prefixes)]


def fit_choice_models(features: pd.DataFrame, rows: pd.DataFrame, labels: np.ndarray, rank_leaves: int, rounds: int, epochs: int, regularization: float) -> ChoiceModels:
    row_columns = list(features.columns)
    rank_columns = conditional_columns(features)
    burst_features, burst_labels, burst_codes, _ = burst_frame(features, rows, labels)
    burst_columns = list(burst_features.columns)
    stage_a = lgb.LGBMClassifier(
        objective="binary", n_estimators=rounds, learning_rate=0.04, num_leaves=15,
        min_child_samples=20, reg_lambda=5.0, reg_alpha=0.2, subsample=0.9,
        colsample_bytree=0.85, random_state=1337, n_jobs=11, verbosity=-1,
    )
    stage_a.fit(burst_features, burst_labels)
    burst_sizes = np.bincount(burst_codes)
    burst_positive = np.bincount(burst_codes, weights=labels)
    valid_bursts = np.flatnonzero((burst_positive > 0) & (burst_sizes > 1))
    mask = np.isin(burst_codes, valid_bursts)
    order = np.argsort(burst_codes[mask], kind="stable")
    selected_indices = np.flatnonzero(mask)[order]
    selected_features = features.iloc[selected_indices][rank_columns]
    selected_labels = labels[selected_indices]
    selected_original_groups = burst_codes[selected_indices]
    selected_groups = pd.factorize(selected_original_groups, sort=False)[0]
    group_sizes = np.bincount(selected_groups)
    stage_b = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=max(rounds, 30), learning_rate=0.04,
        num_leaves=rank_leaves, min_child_samples=20, reg_lambda=5.0,
        reg_alpha=0.1, subsample=0.9, colsample_bytree=0.9, random_state=1337,
        n_jobs=11, verbosity=-1, label_gain=[0, 1], lambdarank_truncation_level=6,
    )
    stage_b.fit(selected_features, selected_labels, group=group_sizes)
    conditional = fit_conditional_scorer(selected_features, selected_labels, selected_groups, epochs, regularization)
    rank_training = stage_b.predict(selected_features)
    conditional_training = conditional.predict(selected_features)
    rank_scale = max(float(np.std(center_by_group(rank_training, selected_groups))), 1e-3)
    conditional_scale = max(float(np.std(center_by_group(conditional_training, selected_groups))), 1e-3)
    guard = lgb.LGBMClassifier(
        objective="binary", n_estimators=max(30, min(rounds, 300)), learning_rate=0.04,
        num_leaves=15, max_depth=4, min_child_samples=30, reg_lambda=8.0,
        reg_alpha=0.5, subsample=0.9, colsample_bytree=0.8, random_state=7331,
        n_jobs=11, verbosity=-1,
    )
    guard.fit(features, labels)
    return ChoiceModels(
        stage_a=stage_a, stage_b=stage_b, conditional=conditional, guard=guard,
        row_columns=row_columns, rank_columns=rank_columns, burst_columns=burst_columns,
        rank_scale=rank_scale, conditional_scale=conditional_scale,
    )


def raw_predictions(models: ChoiceModels, features: pd.DataFrame, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    aligned = align_features(features, models.row_columns)
    rank_features = align_features(features, models.rank_columns)
    burst_features, _, burst_codes, _ = burst_frame(aligned, rows)
    burst_features = align_features(burst_features, models.burst_columns)
    burst_probability = models.stage_a.predict_proba(burst_features)[:, 1]
    rank_scores = models.stage_b.predict(rank_features)
    conditional_scores = models.conditional.predict(rank_features)
    scores = 0.7 * rank_scores / models.rank_scale + 0.3 * conditional_scores / models.conditional_scale
    guard_probability = models.guard.predict_proba(aligned)[:, 1]
    return burst_probability, scores, guard_probability, burst_codes, burst_features["b_size"].to_numpy(dtype=np.int64)


def compose_predictions(raw: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], temperature: float, guard_weight: float) -> np.ndarray:
    burst_probability, scores, guard_probability, burst_codes, burst_sizes = raw
    conditional = np.empty(len(scores), dtype=np.float64)
    for burst in range(len(burst_probability)):
        indices = np.flatnonzero(burst_codes == burst)
        if len(indices) == 1:
            conditional[indices] = 1.0
        else:
            scaled = scores[indices] / temperature
            scaled = scaled - scaled.max()
            probabilities = np.exp(scaled)
            conditional[indices] = probabilities / probabilities.sum()
    composed = burst_probability[burst_codes] * conditional
    return np.clip((1.0 - guard_weight) * composed + guard_weight * guard_probability, 1e-6, 1.0 - 1e-6)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")


def fold_diagnostics(labels: np.ndarray, predictions: np.ndarray, raw, rows: pd.DataFrame) -> dict:
    burst_probability, scores, _, burst_codes, burst_sizes = raw
    burst_labels = np.bincount(burst_codes, weights=labels) > 0
    multi_correct = []
    for burst in range(len(burst_probability)):
        indices = np.flatnonzero(burst_codes == burst)
        if len(indices) > 1 and labels[indices].sum() == 1:
            multi_correct.append(float(labels[indices[np.argmax(scores[indices])]] == 1))
    strata = {}
    row_sizes = burst_sizes[burst_codes]
    for size in sorted(np.unique(row_sizes)):
        mask = row_sizes == size
        strata[str(int(size))] = {"count": int(mask.sum()), "auc": safe_auc(labels[mask], predictions[mask])}
    return {
        "row_auc": safe_auc(labels, predictions),
        "burst_any_auc": safe_auc(burst_labels.astype(np.int64), burst_probability),
        "within_multi_accuracy": float(np.mean(multi_correct)) if multi_correct else float("nan"),
        "burst_count": int(len(burst_probability)),
        "strata": strata,
    }


def forward_folds(train: pd.DataFrame, debug: bool) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    end = train["timestamp"].max().normalize() + pd.Timedelta(days=1)
    count = 1 if debug else 5
    starts = [end - pd.Timedelta(days=7 * index) for index in range(count, 0, -1)]
    return [(start, start + pd.Timedelta(days=7)) for start in starts]


def internal_validation(train: pd.DataFrame, friend_sets: dict[int, set[int]], debug: bool, started: float) -> tuple[dict, list[dict]]:
    folds = forward_folds(train, debug)
    depths = [15] if debug else [15, 31]
    temperatures = [1.0] if debug else [0.5, 0.75, 1.0, 1.5, 2.0]
    guard_weights = [0.0] if debug else [0.0, 0.15, 0.30]
    regularizations = [1e-4] if debug else [1e-5, 1e-4, 1e-3]
    rounds = 30 if debug else 350
    epochs = 30 if debug else 100
    records = []
    for fold_index, (start, end) in enumerate(folds):
        prefix = train.loc[train["timestamp"] < start].copy()
        validation = train.loc[(train["timestamp"] >= start) & (train["timestamp"] < end)].copy()
        if len(prefix) == 0 or validation["label"].nunique() < 2:
            raise RuntimeError(f"invalid forward fold {fold_index}: prefix={len(prefix)} validation={len(validation)}")
        train_features = make_row_features(prefix, prefix, friend_sets)
        validation_features = make_row_features(validation, prefix, friend_sets)
        labels = prefix["label"].to_numpy(dtype=np.int64)
        validation_labels = validation["label"].to_numpy(dtype=np.int64)
        for rank_leaves in depths:
            for regularization in regularizations:
                models = fit_choice_models(train_features, prefix, labels, rank_leaves, rounds, epochs, regularization)
                raw = raw_predictions(models, validation_features, validation)
                for temperature in temperatures:
                    for guard_weight in guard_weights:
                        predictions = compose_predictions(raw, temperature, guard_weight)
                        diagnostic = fold_diagnostics(validation_labels, predictions, raw, validation)
                        records.append({
                            "fold": fold_index, "start": str(start), "end": str(end), "prefix_rows": len(prefix),
                            "validation_rows": len(validation), "rank_leaves": rank_leaves, "regularization": regularization,
                            "temperature": temperature, "guard_weight": guard_weight, **diagnostic,
                        })
                chosen = compose_predictions(raw, 1.0, 0.0)
                diagnostic = fold_diagnostics(validation_labels, chosen, raw, validation)
                print(f"[forward fold] fold={fold_index} leaves={rank_leaves} regularization={regularization:.0e} rows={len(validation)} row_auc={diagnostic['row_auc']:.6f} burst_auc={diagnostic['burst_any_auc']:.6f} within={diagnostic['within_multi_accuracy']:.6f} strata={json.dumps(diagnostic['strata'], allow_nan=True)}")
                del models
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    base_candidates = []
    for rank_leaves in depths:
        for regularization in regularizations:
            for temperature in temperatures:
                values = [record["row_auc"] for record in records if record["rank_leaves"] == rank_leaves and record["regularization"] == regularization and record["temperature"] == temperature and record["guard_weight"] == 0.0]
                if len(values) == len(folds):
                    base_candidates.append((float(np.mean(values)) + 0.2 * float(np.min(values)), float(np.mean(values)), float(np.min(values)), rank_leaves, regularization, temperature, values))
    best_base = max(base_candidates)
    _, base_mean, base_worst, rank_leaves, regularization, temperature, base_values = best_base
    selected_weight = 0.0
    selected_mean = base_mean
    selected_worst = base_worst
    for weight in guard_weights[1:]:
        values = [record["row_auc"] for record in records if record["rank_leaves"] == rank_leaves and record["regularization"] == regularization and record["temperature"] == temperature and record["guard_weight"] == weight]
        differences = np.asarray(values) - np.asarray(base_values)
        mean_value = float(np.mean(values))
        worst_value = float(np.min(values))
        stable = int((differences > 0).sum()) >= 3 and float(differences.mean()) > 0.0 and worst_value >= base_worst - 0.002
        if stable and mean_value + 0.2 * worst_value > selected_mean + 0.2 * selected_worst:
            selected_weight = weight
            selected_mean = mean_value
            selected_worst = worst_value
    selection = {
        "rank_leaves": int(rank_leaves), "temperature": float(temperature), "guard_weight": float(selected_weight),
        "mean_auc": float(selected_mean), "worst_auc": float(selected_worst),
        "rounds": rounds, "epochs": epochs, "regularization": float(regularization),
    }
    print(f"[forward validation] selection={json.dumps(selection)}")
    elapsed(started, "forward validation")
    return selection, records


def fit_and_predict(training: pd.DataFrame, query: pd.DataFrame, friend_sets: dict[int, set[int]], selection: dict, debug: bool) -> tuple[np.ndarray, dict]:
    training_features = make_row_features(training, training, friend_sets)
    query_features = make_row_features(query, training, friend_sets)
    labels = training["label"].to_numpy(dtype=np.int64)
    rounds = 30 if debug else 400
    epochs = 30 if debug else 120
    models = fit_choice_models(
        training_features, training, labels, selection["rank_leaves"], rounds, epochs, selection["regularization"],
    )
    raw = raw_predictions(models, query_features, query)
    predictions = compose_predictions(raw, selection["temperature"], selection["guard_weight"])
    diagnostics = {
        "rows": int(len(query)), "bursts": int(len(raw[0])), "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()), "prediction_mean": float(predictions.mean()),
    }
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, diagnostics


def run_pipeline(debug: bool, started: float) -> dict:
    dataset_name = os.environ["RELBENCH_DATASET"]
    task_name = os.environ["RELBENCH_TASK"]
    cache_dir = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = get_dataset(dataset_name, download=False)
    task = get_task(dataset_name, task_name, download=False)
    rows, friend_sets = build_static_features(dataset, task, cache_dir, started)
    train = attach_labels(rows, task, "train")
    validation = attach_labels(rows, task, "val")
    test = attach_labels(rows, task, "test")
    selection, fold_records = internal_validation(train, friend_sets, debug, started)
    val_predictions, val_build = fit_and_predict(train, validation, friend_sets, selection, debug)
    elapsed(started, "Model A")
    train_validation = pd.concat([train, validation], ignore_index=True)
    test_predictions, test_build = fit_and_predict(train_validation, test, friend_sets, selection, debug)
    elapsed(started, "Model B")
    diagnostics = {
        "debug": debug, "selection": selection, "folds": fold_records,
        "model_a_validation": val_build, "model_b_test": test_build,
        "feature_count": int(sum(column.startswith("f_") for column in rows.columns)),
        "validation_fit_labels": "train_only", "test_fit_labels": "train_plus_validation",
    }
    return {"val_predictions": val_predictions, "test_predictions": test_predictions, "diagnostics": diagnostics}
