from __future__ import annotations

import hashlib
import json
import os
import time
import fcntl
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "causal_graph_boost_v2"


def version_key() -> str:
    return hashlib.sha256(VERSION.encode()).hexdigest()[:16]


def clean_category(values: pd.Series, missing: str = "__missing__") -> pd.Series:
    return values.fillna(missing).astype(str).str.strip().replace("", missing)


@dataclass
class GraphData:
    ids: np.ndarray
    id_to_pos: dict[int, int]
    joined_ns: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    degree: np.ndarray
    edges: np.ndarray

    def neighbors(self, user_id: int, cutoff_ns: int) -> np.ndarray:
        pos = self.id_to_pos.get(int(user_id))
        if pos is None:
            return np.empty(0, dtype=np.int32)
        out = self.indices[self.indptr[pos] : self.indptr[pos + 1]]
        return out[self.joined_ns[out] <= cutoff_ns]


def build_graph(users: pd.DataFrame, friends: pd.DataFrame, cache_dir: Path) -> GraphData:
    key = version_key()
    path = cache_dir / f"strict_friend_graph_{key}.npz"
    ids = users["user_id"].astype(np.int64).to_numpy()
    joined_values = pd.to_datetime(users["joinedAt"])
    joined_ns = joined_values.astype("int64").to_numpy()
    joined_ns[pd.isna(joined_values).to_numpy()] = np.iinfo(np.int64).max
    order = np.argsort(ids)
    ids = ids[order]
    joined_ns = joined_ns[order]
    id_to_pos = {int(v): i for i, v in enumerate(ids)}
    if path.exists():
        data = np.load(path, allow_pickle=False)
        return GraphData(ids, id_to_pos, joined_ns, data["indptr"], data["indices"], data["degree"], data["edges"])
    valid = friends["user"].notna() & friends["friend"].notna()
    raw = friends.loc[valid, ["user", "friend"]].astype(np.int64).to_numpy()
    raw = raw[raw[:, 0] != raw[:, 1]]
    raw.sort(axis=1)
    raw = np.unique(raw, axis=0)
    src = np.concatenate([raw[:, 0], raw[:, 1]])
    dst = np.concatenate([raw[:, 1], raw[:, 0]])
    src_pos = np.searchsorted(ids, src)
    dst_pos = np.searchsorted(ids, dst)
    keep = (src_pos < len(ids)) & (dst_pos < len(ids)) & (ids[np.minimum(src_pos, len(ids) - 1)] == src) & (ids[np.minimum(dst_pos, len(ids) - 1)] == dst)
    src_pos = src_pos[keep]
    dst_pos = dst_pos[keep]
    sort_idx = np.lexsort((dst_pos, src_pos))
    src_pos = src_pos[sort_idx].astype(np.int32)
    dst_pos = dst_pos[sort_idx].astype(np.int32)
    degree = np.bincount(src_pos, minlength=len(ids)).astype(np.int32)
    indptr = np.concatenate([[0], np.cumsum(degree)]).astype(np.int64)
    edges = np.column_stack([np.searchsorted(ids, raw[:, 0]), np.searchsorted(ids, raw[:, 1])]).astype(np.int32)
    np.savez_compressed(path, indptr=indptr, indices=dst_pos, degree=degree, edges=edges)
    return GraphData(ids, id_to_pos, joined_ns, indptr, dst_pos, degree, edges)


def user_frame(users: pd.DataFrame) -> pd.DataFrame:
    out = users.copy()
    out["user_id"] = out["user_id"].astype(np.int64)
    out["locale"] = clean_category(out["locale"])
    out["gender"] = clean_category(out["gender"])
    out["location"] = clean_category(out["location"])
    parts = out["location"].str.split(",")
    out["city"] = parts.str[0].str.strip().replace("", "__missing__")
    out["country"] = parts.str[-1].str.strip().replace("", "__missing__")
    out.loc[out["location"] == "__missing__", ["city", "country"]] = "__missing__"
    out["locale_gender"] = out["locale"] + "|" + out["gender"]
    out["timezone_bucket"] = out["timezone"].round().fillna(999).astype(int).astype(str)
    return out


def demographic_features(seeds: pd.DataFrame, users: pd.DataFrame, origin_ns: int) -> tuple[pd.DataFrame, list[str]]:
    columns = ["user_id", "locale", "gender", "location", "city", "country", "locale_gender", "timezone_bucket", "timezone", "joinedAt"]
    out = seeds[["user_id", "joinedAt"]].merge(users[columns].drop(columns="joinedAt"), on="user_id", how="left", sort=False)
    joined = pd.to_datetime(out["joinedAt"])
    joined_safe = joined.fillna(pd.Timestamp(origin_ns))
    tz = pd.to_numeric(out["timezone"], errors="coerce")
    location = clean_category(out["location"])
    out["timezone"] = tz
    out["timezone_normalized"] = ((tz + 12.0) % 24.0) - 12.0
    out["timezone_missing"] = tz.isna().astype(np.int8)
    out["location_missing"] = (location == "__missing__").astype(np.int8)
    out["location_token_count"] = location.str.replace(",", " ", regex=False).str.split().str.len().fillna(0).astype(np.int16)
    out["joined_time_missing"] = joined.isna().astype(np.int8)
    out["joined_month"] = joined_safe.dt.month.astype(np.int8)
    out["joined_week"] = joined_safe.dt.isocalendar().week.astype(np.int16)
    out["joined_day"] = joined_safe.dt.day.astype(np.int8)
    out["joined_hour"] = joined_safe.dt.hour.astype(np.int8)
    out["joined_dow"] = joined_safe.dt.dayofweek.astype(np.int8)
    out["joined_dayofyear"] = joined_safe.dt.dayofyear.astype(np.int16)
    out["joined_weekend"] = (joined_safe.dt.dayofweek >= 5).astype(np.int8)
    out["elapsed_days"] = (joined_safe.astype("int64") - origin_ns) / 86_400_000_000_000.0
    uid = out["user_id"].astype(np.int64)
    out["user_id_rank"] = uid.rank(method="dense").astype(float) / max(1, len(users))
    out["user_hash_17"] = ((uid * 1103515245 + 17) & 0x7FFFFFFF) / 2147483647.0
    out["user_hash_43"] = ((uid * 2654435761 + 43) & 0xFFFFFFFF) / 4294967295.0
    categorical = ["locale", "gender", "location", "city", "country", "locale_gender", "timezone_bucket"]
    return out.drop(columns=["joinedAt"]), categorical


def _prefix_encoding(seed_keys: np.ndarray, seed_times: np.ndarray, label_keys: np.ndarray, label_times: np.ndarray, label_values: np.ndarray, strength: float) -> np.ndarray:
    label_order = np.argsort(label_times, kind="stable")
    all_times = label_times[label_order]
    all_values = label_values[label_order]
    all_sum = np.concatenate([[0.0], np.cumsum(all_values)])
    result = np.empty(len(seed_keys), dtype=np.float64)
    grouped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    frame = pd.DataFrame({"key": label_keys, "time": label_times, "value": label_values})
    for key, group in frame.groupby("key", sort=False, observed=True):
        order = np.argsort(group["time"].to_numpy(), kind="stable")
        times = group["time"].to_numpy()[order]
        sums = np.concatenate([[0.0], np.cumsum(group["value"].to_numpy()[order])])
        grouped[str(key)] = (times, sums)
    for i, (key, stamp) in enumerate(zip(seed_keys, seed_times)):
        total_count = int(np.searchsorted(all_times, stamp, side="left"))
        prior = all_sum[total_count] / total_count if total_count else 1988.0
        entry = grouped.get(str(key))
        if entry is None:
            result[i] = prior
            continue
        count = int(np.searchsorted(entry[0], stamp, side="left"))
        result[i] = (entry[1][count] + strength * prior) / (count + strength)
    return result


def target_encoding_features(seeds: pd.DataFrame, users: pd.DataFrame, labels: pd.DataFrame, strength: float = 20.0) -> pd.DataFrame:
    keys = ["locale", "country", "city", "locale_gender", "timezone_bucket"]
    seed = seeds[["user_id", "joinedAt"]].merge(users[["user_id"] + keys], on="user_id", how="left", sort=False)
    lab = labels[["user_id", "joinedAt", "birthyear"]].merge(users[["user_id"] + keys], on="user_id", how="left", sort=False)
    seed_times = pd.to_datetime(seed["joinedAt"]).astype("int64").to_numpy()
    label_times = pd.to_datetime(lab["joinedAt"]).astype("int64").to_numpy()
    label_values = lab["birthyear"].astype(float).to_numpy()
    out = pd.DataFrame(index=np.arange(len(seed)))
    for key in keys:
        out[f"te_{key}"] = _prefix_encoding(clean_category(seed[key]).to_numpy(), seed_times, clean_category(lab[key]).to_numpy(), label_times, label_values, strength)
    return out


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    return float(np.dot(values, weights) / total) if total > 0 else np.nan


def graph_features(seeds: pd.DataFrame, users: pd.DataFrame, graph: GraphData, labels: pd.DataFrame, priors: np.ndarray, strength: float = 20.0) -> pd.DataFrame:
    user_rows = users.set_index("user_id", drop=False)
    label_value = {int(row.user_id): float(row.birthyear) for row in labels.itertuples(index=False)}
    label_time = {int(row.user_id): int(pd.Timestamp(row.joinedAt).value) for row in labels.itertuples(index=False)}
    names = [
        "eligible_degree", "friend_labeled_count", "friend_labeled_fraction", "friend_label_mean", "friend_label_median",
        "friend_label_std", "friend_label_q25", "friend_label_q75", "friend_label_min", "friend_label_max", "friend_label_trimmed",
        "friend_label_invdegree", "friend_label_range", "friend_label_shrunk", "twohop_label_numerator", "twohop_label_mass",
        "twohop_effective_count", "twohop_label_mean", "twohop_disagreement", "friend_gender_female", "friend_gender_male",
        "friend_gender_missing", "friend_timezone_mean", "friend_timezone_std", "friend_locale_match", "friend_top_locale_share",
        "friend_join_gap_mean_days", "friend_join_gap_std_days", "friend_degree_q25", "friend_degree_median", "friend_degree_q75",
        "friend_demo_missing_fraction"
    ]
    values = np.full((len(seeds), len(names)), np.nan, dtype=np.float64)
    for i, seed in enumerate(seeds.itertuples(index=False)):
        uid = int(seed.user_id)
        cutoff = int(pd.Timestamp(seed.joinedAt).value)
        pos = graph.id_to_pos.get(uid)
        if pos is None:
            values[i, 0] = 0
            values[i, 1] = 0
            values[i, 2] = 0
            values[i, 15:18] = 0
            continue
        neighbors = graph.neighbors(uid, cutoff)
        neighbor_ids = graph.ids[neighbors]
        values[i, 0] = len(neighbors)
        eligible_labels = [j for j, v in enumerate(neighbor_ids) if int(v) in label_value and label_time[int(v)] < cutoff]
        if eligible_labels:
            lab_ids = neighbor_ids[np.asarray(eligible_labels, dtype=int)]
            labs = np.asarray([label_value[int(v)] for v in lab_ids], dtype=float)
            weights = 1.0 / np.log2(2.0 + graph.degree[np.asarray(neighbors)[np.asarray(eligible_labels, dtype=int)]])
            sorted_labs = np.sort(labs)
            trim = sorted_labs[1:-1] if len(sorted_labs) >= 5 else sorted_labs
            mean = float(np.mean(labs))
            values[i, 1:14] = [
                len(labs), len(labs) / max(1, len(neighbors)), mean, float(np.median(labs)), float(np.std(labs)),
                float(np.quantile(labs, 0.25)), float(np.quantile(labs, 0.75)), float(np.min(labs)), float(np.max(labs)),
                float(np.mean(trim)), _weighted_mean(labs, weights), float(np.max(labs) - np.min(labs)),
                float((labs.sum() + strength * priors[i]) / (len(labs) + strength))
            ]
        else:
            values[i, 1] = 0
            values[i, 2] = 0
            values[i, 13] = priors[i]
        direct = set(int(v) for v in neighbor_ids)
        twohop: dict[int, float] = {}
        for neighbor in neighbors:
            for node in graph.neighbors(int(graph.ids[neighbor]), cutoff):
                node_id = int(graph.ids[node])
                if node_id == uid or node_id in direct or node_id not in label_value or label_time[node_id] >= cutoff:
                    continue
                path_weight = 1.0 / (np.log2(2.0 + graph.degree[neighbor]) * np.log2(2.0 + graph.degree[node]))
                twohop[node_id] = twohop.get(node_id, 0.0) + path_weight
        if twohop:
            two_ids = np.fromiter(twohop.keys(), dtype=np.int64)
            two_weights = np.fromiter(twohop.values(), dtype=np.float64)
            two_values = np.asarray([label_value[int(v)] for v in two_ids])
            numerator = float(np.dot(two_values, two_weights))
            mass = float(two_weights.sum())
            estimate = numerator / mass
            effective = mass * mass / float(np.dot(two_weights, two_weights))
            onehop = values[i, 3] if np.isfinite(values[i, 3]) else priors[i]
            values[i, 14:19] = [numerator, mass, effective, estimate, estimate - onehop]
        else:
            values[i, 14:17] = 0
            values[i, 17] = priors[i]
            values[i, 18] = 0
        if len(neighbors):
            demo = user_rows.loc[neighbor_ids]
            genders = clean_category(demo["gender"]).str.lower()
            timezones = pd.to_numeric(demo["timezone"], errors="coerce").to_numpy(dtype=float)
            locales = clean_category(demo["locale"])
            seed_locale = str(user_rows.loc[uid, "locale"]) if uid in user_rows.index else "__missing__"
            gaps = (cutoff - graph.joined_ns[neighbors]) / 86_400_000_000_000.0
            degrees = graph.degree[neighbors].astype(float)
            top_share = float(locales.value_counts(normalize=True).iloc[0]) if len(locales) else 0.0
            miss = ((locales == "__missing__").to_numpy() | genders.eq("__missing__").to_numpy() | np.isnan(timezones)).mean()
            values[i, 19:] = [
                float(genders.str.startswith("f").mean()), float(genders.str.startswith("m").mean()), float(genders.eq("__missing__").mean()),
                float(np.nanmean(timezones)) if np.isfinite(timezones).any() else np.nan,
                float(np.nanstd(timezones)) if np.isfinite(timezones).any() else np.nan,
                float((locales == seed_locale).mean()), top_share, float(np.mean(gaps)), float(np.std(gaps)),
                float(np.quantile(degrees, 0.25)), float(np.median(degrees)), float(np.quantile(degrees, 0.75)), float(miss)
            ]
    return pd.DataFrame(values, columns=names)


@dataclass
class ActivityStore:
    times: dict[int, np.ndarray]
    events: dict[int, np.ndarray]
    values: dict[int, np.ndarray]
    names: list[str]

    def aggregate(self, members: np.ndarray, cutoff_ns: int) -> np.ndarray:
        total = np.zeros(len(self.names), dtype=np.float64)
        total_sq = np.zeros(len(self.names), dtype=np.float64)
        count = 0
        event_parts = []
        for member in members:
            uid = int(member)
            times = self.times.get(uid)
            if times is None:
                continue
            end = int(np.searchsorted(times, cutoff_ns, side="right"))
            if end == 0:
                continue
            total += self.values[uid][:end].sum(axis=0)
            total_sq += np.square(self.values[uid][:end]).sum(axis=0)
            count += end
            event_parts.append(self.events[uid][:end])
        out = np.zeros(3 + len(self.names) * 2, dtype=np.float64)
        out[0] = count
        out[1] = len(np.unique(np.concatenate(event_parts))) if event_parts else 0
        out[2] = float(count > 0)
        if count:
            mean = total / count
            out[3 : 3 + len(self.names)] = mean
            out[3 + len(self.names) :] = np.sqrt(np.maximum(0.0, total_sq / count - mean * mean))
        return out


def _activity_store(frame: pd.DataFrame, user_col: str, time_col: str, event_col: str, values: np.ndarray, names: list[str]) -> ActivityStore:
    times: dict[int, np.ndarray] = {}
    events: dict[int, np.ndarray] = {}
    matrices: dict[int, np.ndarray] = {}
    users = frame[user_col].astype(np.int64).to_numpy()
    stamps = pd.to_datetime(frame[time_col]).astype("int64").to_numpy()
    event_ids = frame[event_col].astype(np.int64).to_numpy()
    for uid in np.unique(users):
        idx = np.flatnonzero(users == uid)
        order = np.argsort(stamps[idx], kind="stable")
        selected = idx[order]
        times[int(uid)] = stamps[selected]
        events[int(uid)] = event_ids[selected]
        matrices[int(uid)] = values[selected]
    return ActivityStore(times, events, matrices, names)


def build_activity_stores(attendees: pd.DataFrame, interests: pd.DataFrame, events: pd.DataFrame) -> tuple[ActivityStore, ActivityStore]:
    category_cols = [c for c in events.columns if c.startswith("c_")]
    rng = np.random.default_rng(17)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(len(category_cols), 16)) / np.sqrt(max(1, len(category_cols)))
    event_cols = ["event_id", "start_time", "lat", "lng"] + category_cols
    attendee_mask = attendees["user_id"].notna() & attendees["event"].notna()
    interest_mask = interests["user"].notna() & interests["event"].notna()
    needed = np.unique(np.concatenate([
        attendees.loc[attendee_mask, "event"].astype(np.int64).to_numpy(),
        interests.loc[interest_mask, "event"].astype(np.int64).to_numpy()
    ]))
    event_frame = events.loc[events["event_id"].isin(needed), event_cols].copy()
    categories = event_frame[category_cols].fillna(0).to_numpy(dtype=np.float32)
    projection = categories @ signs
    for j in range(16):
        event_frame[f"rp_{j}"] = projection[:, j]
    attendee = attendees.loc[attendee_mask, ["user_id", "event", "status", "start_time"]].copy()
    attendee["user_id"] = attendee["user_id"].astype(np.int64)
    attendee["event"] = attendee["event"].astype(np.int64)
    event_join = event_frame.rename(columns={"start_time": "event_start_time"})
    attendee = attendee.merge(event_join, left_on="event", right_on="event_id", how="left", sort=False)
    attendee["feature_time"] = pd.concat([pd.to_datetime(attendee["start_time"]), pd.to_datetime(attendee["event_start_time"])], axis=1).max(axis=1)
    stamp = pd.to_datetime(attendee["feature_time"])
    status = clean_category(attendee["status"]).str.lower()
    att_names = ["status_yes", "status_maybe", "status_no", "status_invited", "hour_sin", "hour_cos", "weekend", "geo_valid", "lat_scaled", "lng_scaled"] + [f"rp_{j}_scaled" for j in range(16)]
    att_values = np.column_stack([
        status.isin(["yes", "going", "attending"]).astype(float), status.str.contains("maybe", regex=False).astype(float),
        status.isin(["no", "declined", "not attending"]).astype(float), status.str.contains("invited", regex=False).astype(float),
        np.sin(2 * np.pi * stamp.dt.hour.fillna(0).to_numpy() / 24), np.cos(2 * np.pi * stamp.dt.hour.fillna(0).to_numpy() / 24),
        (stamp.dt.dayofweek >= 5).astype(float), attendee["lat"].notna().astype(float),
        attendee["lat"].fillna(0).to_numpy(dtype=float) / 90.0, attendee["lng"].fillna(0).to_numpy(dtype=float) / 180.0,
        attendee[[f"rp_{j}" for j in range(16)]].fillna(0).to_numpy(dtype=float) / 10.0
    ])
    interest = interests.loc[interest_mask, ["user", "event", "invited", "interested", "not_interested", "timestamp"]].copy()
    interest["user"] = interest["user"].astype(np.int64)
    interest["event"] = interest["event"].astype(np.int64)
    interest = interest.merge(event_join, left_on="event", right_on="event_id", how="left", sort=False)
    interest["feature_time"] = pd.concat([pd.to_datetime(interest["timestamp"]), pd.to_datetime(interest["event_start_time"])], axis=1).max(axis=1)
    int_names = ["invited", "interested", "not_interested", "geo_valid", "lat_scaled", "lng_scaled"] + [f"rp_{j}_scaled" for j in range(16)]
    int_values = np.column_stack([
        interest["invited"].fillna(0).to_numpy(dtype=float), interest["interested"].fillna(0).to_numpy(dtype=float),
        interest["not_interested"].fillna(0).to_numpy(dtype=float), interest["lat"].notna().astype(float),
        interest["lat"].fillna(0).to_numpy(dtype=float) / 90.0, interest["lng"].fillna(0).to_numpy(dtype=float) / 180.0,
        interest[[f"rp_{j}" for j in range(16)]].fillna(0).to_numpy(dtype=float) / 10.0
    ])
    return _activity_store(attendee, "user_id", "feature_time", "event", att_values, att_names), _activity_store(interest, "user", "feature_time", "event", int_values, int_names)


def activity_features(seeds: pd.DataFrame, graph: GraphData, attendee: ActivityStore, interest: ActivityStore) -> pd.DataFrame:
    att_width = 3 + 2 * len(attendee.names)
    int_width = 3 + 2 * len(interest.names)
    data = np.zeros((len(seeds), 2 * att_width + 2 * int_width), dtype=np.float64)
    for i, seed in enumerate(seeds.itertuples(index=False)):
        uid = int(seed.user_id)
        cutoff = int(pd.Timestamp(seed.joinedAt).value)
        neighbors = graph.neighbors(uid, cutoff)
        member_ids = graph.ids[neighbors]
        cursor = 0
        for store, members in ((attendee, np.asarray([uid])), (attendee, member_ids), (interest, np.asarray([uid])), (interest, member_ids)):
            block = store.aggregate(members, cutoff)
            data[i, cursor : cursor + len(block)] = block
            cursor += len(block)
    names = []
    for prefix, store in (("own_att", attendee), ("friend_att", attendee), ("own_interest", interest), ("friend_interest", interest)):
        names.extend([f"{prefix}_count", f"{prefix}_distinct_events", f"{prefix}_coverage"])
        names.extend([f"{prefix}_{name}_mean" for name in store.names])
        names.extend([f"{prefix}_{name}_dispersion" for name in store.names])
    return pd.DataFrame(data, columns=names)


@dataclass
class FeatureBuilder:
    users: pd.DataFrame
    graph: GraphData
    attendee: ActivityStore
    interest: ActivityStore
    origin_ns: int

    def build(self, seeds: pd.DataFrame, labels: pd.DataFrame, include_activity: bool = True) -> tuple[pd.DataFrame, list[str]]:
        start = time.time()
        base, categorical = demographic_features(seeds, self.users, self.origin_ns)
        encoded = target_encoding_features(seeds, self.users, labels)
        graph = graph_features(seeds, self.users, self.graph, labels, encoded["te_locale_gender"].to_numpy())
        blocks = [base.reset_index(drop=True), encoded, graph]
        if include_activity:
            blocks.append(activity_features(seeds, self.graph, self.attendee, self.interest))
        frame = pd.concat(blocks, axis=1)
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
        frame.attrs["seconds"] = time.time() - start
        return frame, categorical


def append_artifact_registry(cache_dir: Path, name: str, path: Path, description: str) -> None:
    registry = cache_dir / "artifacts.json"
    lock_path = cache_dir / ".artifacts.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            records = json.loads(registry.read_text()) if registry.exists() else []
        except json.JSONDecodeError:
            records = []
        relative = os.path.relpath(path, cache_dir)
        record = {"name": name, "path": relative, "description": description, "content_key": version_key(), "rebuild_hint": "Run python main.py to rebuild when the version key changes."}
        if not any(item.get("name") == name and item.get("content_key") == version_key() for item in records):
            records.append(record)
            temporary = registry.with_suffix(".tmp")
            temporary.write_text(json.dumps(records, indent=2))
            temporary.replace(registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
