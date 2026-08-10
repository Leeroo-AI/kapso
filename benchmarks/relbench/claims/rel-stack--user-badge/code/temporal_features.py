from __future__ import annotations

import json
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy.stats import rankdata


# Configuration

BASE_GROUPS = 10
GROUPS = 39
HORIZONS = (7, 30, 91, 365)


# Cache

def cache_root() -> Path:
    root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / "lane3_chrono_graph_v1"
    root.mkdir(parents=True, exist_ok=True)
    return root


def register_artifact(name: str, path: Path, description: str, content_key: str) -> None:
    root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    registry = root / "artifacts.json"
    record = {
        "name": name,
        "path": str(path.relative_to(root)),
        "description": description,
        "content_key": content_key,
        "rebuild_hint": "Run main.py; the candidate extends missing lane3_chrono_graph_v1 artifacts.",
    }
    records = json.loads(registry.read_text()) if registry.exists() else []
    if not any(item.get("content_key") == content_key for item in records):
        records.append(record)
        registry.write_text(json.dumps(records, indent=2))


# Data

def database_root() -> Path:
    return Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"] / "db"


def load_users() -> pd.DataFrame:
    path = database_root() / "users.parquet"
    return duckdb.sql(
        f"select Id, AccountId, DisplayName, Location, WebsiteUrl, AboutMe, CreationDate from read_parquet('{path}') order by Id"
    ).df()


def top_badge_names(train_cutoff: pd.Timestamp) -> list[str]:
    path = database_root() / "badges.parquet"
    cutoff = pd.Timestamp(train_cutoff).isoformat()
    rows = duckdb.sql(
        f"select Name from read_parquet('{path}') where Date <= timestamp '{cutoff}' group by Name order by count(*) desc, Name limit 10"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_feature_events(user_ids: np.ndarray, train_cutoff: pd.Timestamp) -> dict[str, np.ndarray]:
    root = cache_root() / "feature_events"
    paths = {name: root / f"{name}.npy" for name in ("time", "user", "group", "value")}
    if all(path.exists() for path in paths.values()):
        return {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    root.mkdir(parents=True, exist_ok=True)
    db = database_root()
    names = top_badge_names(train_cutoff)
    family_case = "case " + " ".join(
        f"when Name={_quoted(name)} then {23 + index}" for index, name in enumerate(names)
    ) + " else -1 end"
    sql = f"""
    with
    posts as materialized (select * from read_parquet('{db / 'posts.parquet'}')),
    base as materialized (
        select p.CreationDate ts, p.OwnerUserId uid,
               case when p.PostTypeId=1 then 0 else 1 end grp,
               length(coalesce(p.Title,''))+length(coalesce(p.Body,''))+length(coalesce(p.Tags,'')) amount,
               pp.OwnerUserId counterpart, p.Id pk
        from posts p left join posts pp on p.ParentId=pp.Id where p.OwnerUserId is not null
        union all
        select p.CreationDate, pp.OwnerUserId, 2,
               length(coalesce(p.Title,''))+length(coalesce(p.Body,''))+length(coalesce(p.Tags,'')),
               p.OwnerUserId, p.Id
        from posts p join posts pp on p.ParentId=pp.Id where pp.OwnerUserId is not null
        union all
        select c.CreationDate,c.UserId,3,length(coalesce(c.Text,'')),p.OwnerUserId,c.Id
        from read_parquet('{db / 'comments.parquet'}') c left join posts p on c.PostId=p.Id
        where c.UserId is not null
        union all
        select c.CreationDate,p.OwnerUserId,4,length(coalesce(c.Text,'')),c.UserId,c.Id
        from read_parquet('{db / 'comments.parquet'}') c join posts p on c.PostId=p.Id
        where p.OwnerUserId is not null
        union all
        select h.CreationDate,h.UserId,5,length(coalesce(h.Text,''))+length(coalesce(h.Comment,'')),p.OwnerUserId,h.Id
        from read_parquet('{db / 'postHistory.parquet'}') h left join posts p on h.PostId=p.Id
        where h.UserId is not null
        union all
        select v.CreationDate,p.OwnerUserId,6,1,null,v.Id
        from read_parquet('{db / 'votes.parquet'}') v join posts p on v.PostId=p.Id
        where p.OwnerUserId is not null
        union all
        select Date,UserId,7,Class,null,Id from read_parquet('{db / 'badges.parquet'}')
        union all
        select l.CreationDate,p.OwnerUserId,8,1,r.OwnerUserId,l.Id
        from read_parquet('{db / 'postLinks.parquet'}') l join posts p on l.PostId=p.Id
        left join posts r on l.RelatedPostId=r.Id where p.OwnerUserId is not null
        union all
        select l.CreationDate,r.OwnerUserId,9,1,p.OwnerUserId,l.Id
        from read_parquet('{db / 'postLinks.parquet'}') l join posts r on l.RelatedPostId=r.Id
        left join posts p on l.PostId=p.Id where r.OwnerUserId is not null
    ),
    derived as (
        select v.CreationDate ts,p.OwnerUserId uid,
               10 + case v.VoteTypeId when 2 then 0 when 1 then 1 when 3 then 2 when 10 then 3 else 4 end grp,
               1 amount,null counterpart,v.Id pk
        from read_parquet('{db / 'votes.parquet'}') v join posts p on v.PostId=p.Id where p.OwnerUserId is not null
        union all
        select Date,UserId,15+greatest(0,least(2,Class-1)),1,null,Id from read_parquet('{db / 'badges.parquet'}')
        union all
        select CreationDate,UserId,
               case PostHistoryTypeId when 1 then 18 when 2 then 19 when 5 then 20 when 10 then 21 when 24 then 22 else -1 end,
               1,null,Id from read_parquet('{db / 'postHistory.parquet'}') where UserId is not null
        union all
        select Date,UserId,{family_case},1,null,Id from read_parquet('{db / 'badges.parquet'}')
    ),
    diversity as (
        select min(ts) ts,uid,
               33 + case grp when 1 then 0 when 2 then 1 when 3 then 2 when 4 then 3 when 8 then 4 else 5 end grp,
               1 amount,min(pk) pk
        from base where grp in (1,2,3,4,8,9) and counterpart is not null and counterpart != uid
        group by uid,grp,counterpart
    )
    select ts,uid,grp,amount,pk from base
    union all select ts,uid,grp,amount,pk from derived where grp >= 0
    union all select ts,uid,grp,amount,pk from diversity
    order by ts,grp,pk
    """
    frame = duckdb.sql(sql).df()
    mapped = np.searchsorted(user_ids, frame["uid"].to_numpy(dtype=np.int64))
    valid = (mapped >= 0) & (mapped < len(user_ids)) & (user_ids[np.minimum(mapped, len(user_ids) - 1)] == frame["uid"].to_numpy(dtype=np.int64))
    arrays = {
        "time": frame.loc[valid, "ts"].to_numpy(dtype="datetime64[s]").astype(np.int64),
        "user": mapped[valid].astype(np.int32),
        "group": frame.loc[valid, "grp"].to_numpy(dtype=np.int8),
        "value": frame.loc[valid, "amount"].fillna(0).to_numpy(dtype=np.float32),
    }
    del frame
    for name, array in arrays.items():
        np.save(paths[name], array)
    register_artifact(
        "lane3 typed user feature events",
        root,
        "Chronologically sorted all-table user events, category channels, and first-counterpart events.",
        "rel-stack-user-badge-lane3-feature-events-v1",
    )
    return {name: np.load(path, mmap_mode="r") for name, path in paths.items()}


# Feature matrix

def _window_counts(flat: np.ndarray, times: np.ndarray, cutoff: int, days: int, size: int) -> np.ndarray:
    lo = np.searchsorted(times, cutoff - days * 86400, side="right")
    hi = np.searchsorted(times, cutoff, side="right")
    return np.bincount(flat[lo:hi], minlength=size).astype(np.float32)


def _window_values(users: np.ndarray, groups: np.ndarray, values: np.ndarray, times: np.ndarray, cutoff: int, days: int, n_users: int) -> tuple[np.ndarray, np.ndarray]:
    lo = np.searchsorted(times, cutoff - days * 86400, side="right")
    hi = np.searchsorted(times, cutoff, side="right")
    selected = groups[lo:hi] < BASE_GROUPS
    flat = users[lo:hi][selected].astype(np.int64) * BASE_GROUPS + groups[lo:hi][selected]
    count = np.bincount(flat, minlength=n_users * BASE_GROUPS).reshape(n_users, BASE_GROUPS).astype(np.float32)
    total = np.bincount(flat, weights=values[lo:hi][selected], minlength=n_users * BASE_GROUPS).reshape(n_users, BASE_GROUPS).astype(np.float32)
    return count, total


def feature_names() -> list[str]:
    names = ["account_age_days", "log_account_age", "display_name_len", "has_location", "has_website", "has_about", "about_len", "has_account_id"]
    names += [f"lifetime_g{group}" for group in range(GROUPS)]
    names += [f"count_7d_g{group}" for group in range(BASE_GROUPS)]
    for days in (30, 91, 365):
        names += [f"count_{days}d_g{group}" for group in range(GROUPS)]
    names += [f"recency_g{group}" for group in range(GROUPS)]
    names += [f"mean_value_g{group}" for group in range(BASE_GROUPS)]
    names += [f"mean_value_91d_g{group}" for group in range(BASE_GROUPS)]
    names += [f"trend_g{group}" for group in range(BASE_GROUPS)]
    normalized = ["account_age_days"] + [f"lifetime_g{group}" for group in range(BASE_GROUPS)] + [f"count_91d_g{group}" for group in range(BASE_GROUPS)] + [f"recency_g{group}" for group in range(BASE_GROUPS)]
    for name in normalized:
        names.extend((f"rank_{name}", f"z_{name}", f"leader_gap_{name}"))
    return names


def build_compact_features(frames: list[pd.DataFrame], debug: bool = False) -> tuple[np.memmap, list[str], np.ndarray, np.ndarray]:
    users = load_users()
    user_ids = users["Id"].to_numpy(dtype=np.int64)
    train_cutoff = pd.Timestamp(frames[0]["timestamp"].max())
    events = build_feature_events(user_ids, train_cutoff)
    names = feature_names()
    key = "compact_v1.npy"
    path = cache_root() / key
    offsets = np.cumsum([0] + [len(frame) for frame in frames])
    combined_user = np.concatenate([frame["UserId"].to_numpy(dtype=np.int64) for frame in frames])
    combined_time = np.concatenate([frame["timestamp"].to_numpy(dtype="datetime64[s]").astype(np.int64) for frame in frames])
    mapped_user = np.searchsorted(user_ids, combined_user).astype(np.int32)
    if path.exists():
        matrix = np.load(path, mmap_mode="r")
        if matrix.shape == (len(combined_user), len(names)):
            register_artifact(
                "lane3 compact temporal matrix",
                path,
                "All-table temporally censored features with within-origin normalization.",
                "rel-stack-user-badge-lane3-compact-v1",
            )
            return matrix, names, offsets, mapped_user
    matrix = open_memmap(path, mode="w+", dtype=np.float16, shape=(len(combined_user), len(names)))
    times = np.asarray(events["time"])
    event_users = np.asarray(events["user"])
    groups = np.asarray(events["group"])
    values = np.asarray(events["value"])
    flat = event_users.astype(np.int64) * GROUPS + groups
    n_users = len(user_ids)
    state_size = n_users * GROUPS
    cumulative = np.zeros(state_size, dtype=np.float32)
    cumulative_value = np.zeros((n_users, BASE_GROUPS), dtype=np.float32)
    last = np.full(state_size, -1, dtype=np.int64)
    creation = users["CreationDate"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    static = np.column_stack(
        [
            np.zeros(n_users, dtype=np.float32),
            np.zeros(n_users, dtype=np.float32),
            np.log1p(users["DisplayName"].fillna("").str.len().to_numpy(dtype=np.float32)),
            users["Location"].notna().to_numpy(dtype=np.float32),
            users["WebsiteUrl"].notna().to_numpy(dtype=np.float32),
            users["AboutMe"].notna().to_numpy(dtype=np.float32),
            np.log1p(users["AboutMe"].fillna("").str.len().to_numpy(dtype=np.float32)),
            users["AccountId"].notna().to_numpy(dtype=np.float32),
        ]
    )
    name_to_index = {name: index for index, name in enumerate(names)}
    normalized_names = ["account_age_days"] + [f"lifetime_g{group}" for group in range(BASE_GROUPS)] + [f"count_91d_g{group}" for group in range(BASE_GROUPS)] + [f"recency_g{group}" for group in range(BASE_GROUPS)]
    previous = 0
    for cutoff in np.unique(combined_time):
        current = np.searchsorted(times, cutoff, side="right")
        if current > previous:
            cumulative += np.bincount(flat[previous:current], minlength=state_size).astype(np.float32)
            recent_primary = groups[previous:current] < BASE_GROUPS
            primary_flat = event_users[previous:current][recent_primary].astype(np.int64) * BASE_GROUPS + groups[previous:current][recent_primary]
            cumulative_value.ravel()[:] += np.bincount(primary_flat, weights=values[previous:current][recent_primary], minlength=n_users * BASE_GROUPS).astype(np.float32)
            np.maximum.at(last, flat[previous:current], times[previous:current])
        previous = current
        counts_7, values_7 = _window_values(event_users, groups, values, times, int(cutoff), 7, n_users)
        counts_30 = _window_counts(flat, times, int(cutoff), 30, state_size).reshape(n_users, GROUPS)
        counts_91 = _window_counts(flat, times, int(cutoff), 91, state_size).reshape(n_users, GROUPS)
        counts_365 = _window_counts(flat, times, int(cutoff), 365, state_size).reshape(n_users, GROUPS)
        counts_91_primary, values_91 = _window_values(event_users, groups, values, times, int(cutoff), 91, n_users)
        age = np.maximum(0, (cutoff - creation) / 86400).astype(np.float32)
        static[:, 0] = age
        static[:, 1] = np.log1p(age)
        recency = np.where(last.reshape(n_users, GROUPS) >= 0, np.minimum(5000, (cutoff - last.reshape(n_users, GROUPS)) / 86400), 5000)
        mean_value = cumulative_value / np.maximum(1, cumulative.reshape(n_users, GROUPS)[:, :BASE_GROUPS])
        mean_value_91 = values_91 / np.maximum(1, counts_91_primary)
        trend = np.log1p(counts_30[:, :BASE_GROUPS]) - np.log1p(np.maximum(0, counts_91[:, :BASE_GROUPS] - counts_30[:, :BASE_GROUPS]) * (30.0 / 61.0))
        raw = np.column_stack(
            [
                static,
                np.log1p(cumulative.reshape(n_users, GROUPS)),
                np.log1p(counts_7),
                np.log1p(counts_30),
                np.log1p(counts_91),
                np.log1p(counts_365),
                np.log1p(recency),
                np.log1p(mean_value),
                np.log1p(mean_value_91),
                trend,
            ]
        ).astype(np.float32)
        rows = np.flatnonzero(combined_time == cutoff)
        selected = raw[mapped_user[rows]]
        normalized = np.empty((len(rows), len(normalized_names) * 3), dtype=np.float32)
        for index, name in enumerate(normalized_names):
            column = selected[:, name_to_index[name]]
            normalized[:, 3 * index] = rankdata(column, method="average").astype(np.float32) / max(1, len(column))
            scale = max(float(column.std()), 1e-5)
            normalized[:, 3 * index + 1] = (column - float(column.mean())) / scale
            normalized[:, 3 * index + 2] = float(column.max()) - column
        matrix[rows] = np.column_stack([selected, normalized]).astype(np.float16)
        matrix.flush()
        print(f"[features] origin={pd.to_datetime(cutoff, unit='s').date()} rows={len(rows)} events={current}", flush=True)
    register_artifact(
        "lane3 compact temporal matrix",
        path,
        "All-table temporally censored features with within-origin normalization.",
        "rel-stack-user-badge-lane3-compact-v1",
    )
    return np.load(path, mmap_mode="r"), names, offsets, mapped_user


# Hazard labels

def build_hazard_bins(frames: list[pd.DataFrame]) -> np.ndarray:
    db = database_root()
    labeled = pd.concat(frames, ignore_index=True)[["timestamp", "UserId"]]
    labeled["row_id"] = np.arange(len(labeled), dtype=np.int64)
    con = duckdb.connect()
    con.register("seeds", labeled)
    gaps = con.sql(
        f"""
        select s.row_id, date_diff('second',s.timestamp,min(b.Date))/86400.0 gap
        from seeds s left join read_parquet('{db / 'badges.parquet'}') b
        on b.UserId=s.UserId and b.Date>s.timestamp and b.Date<=s.timestamp+interval '91 days'
        group by s.row_id,s.timestamp order by s.row_id
        """
    ).df()["gap"].to_numpy(dtype=np.float32)
    bins = np.full(len(labeled), 13, dtype=np.int8)
    positive = np.isfinite(gaps)
    bins[positive] = np.minimum(12, np.floor(np.maximum(0, gaps[positive]) / 7)).astype(np.int8)
    return bins
