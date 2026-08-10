# Section: imports

from __future__ import annotations

import json
import math
import os
import signal
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score


# Section: constants

VERSION = "lane3_context_eb_v1"
BASE_TIME = pd.Timestamp("2012-07-11")
MODEL_A_CUTOFF = pd.Timestamp("2012-11-14")
MODEL_B_CUTOFF = pd.Timestamp("2012-11-22")
VAL_TIME = pd.Timestamp("2012-11-21")
TEST_TIME = pd.Timestamp("2012-11-29")
PRIOR_STRENGTH = 20.0
RETRIEVAL_TEMPERATURE = 0.10
RETRIEVAL_HALF_LIFE_DAYS = 90.0
CONTEXT_HALF_LIFE_DAYS = 84.0
SOCIAL_DIM = 32
PROJECTION_DIM = 16
MAX_FEATURES = 256
SEED = 1337
LOCAL_TABPFN_MODEL_PATH: str | None = None


# Section: utilities

def elapsed_line(start: float, phase: str) -> None:
    print(f"[lane3] phase={phase} elapsed_seconds={time.time() - start:.3f}", flush=True)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return np.full(len(values), 0.5, dtype=np.float64)
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def clipped_logit(value: float) -> float:
    value = float(np.clip(value, 1e-5, 1.0 - 1e-5))
    return float(math.log(value / (1.0 - value)))


def days_between(later: np.datetime64, earlier: np.ndarray | np.datetime64) -> np.ndarray:
    return (later - earlier).astype("timedelta64[s]").astype(np.float64) / 86400.0


def ensure_float(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].to_numpy(dtype=np.float32, na_value=np.nan)


def standardize_fit(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(train, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        median = np.nanmedian(values, axis=0)
    median[~np.isfinite(median)] = 0.0
    filled = np.where(np.isfinite(values), values, median)
    mean = filled.mean(axis=0)
    scale = filled.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def standardize_apply(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    filled = np.where(finite, array, mean)
    output = (filled - mean) / scale
    return np.clip(output, -12.0, 12.0).astype(np.float32)


def l2_normalize(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1.0
    return (values / norm).astype(np.float32)


def prediction_columns(names: list[str], prefixes: tuple[str, ...]) -> list[int]:
    return [i for i, name in enumerate(names) if name.startswith(prefixes)]


# Section: source loading

def database_paths() -> dict[str, str]:
    root = Path(os.environ["RELBENCH_CACHE_DIR"]) / "rel-event" / "db"
    return {name: str(root / f"{name}.parquet") for name in [
        "users", "user_friends", "events", "event_attendees", "event_interest"
    ]}


def build_replay(attendee_path: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    all_times = pd.date_range(BASE_TIME - pd.Timedelta(days=14), cutoff, freq="D")
    timestamp_df = pd.DataFrame({
        "timestamp": all_times,
        "phase": ((all_times - BASE_TIME).days % 7).astype(np.int8),
    })
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={int(os.environ.get('OMP_NUM_THREADS', '1'))}")
    connection.register("timestamp_df", timestamp_df)
    replay = connection.execute(
        """
        WITH tb AS(
            SELECT
                t.timestamp,
                t.phase,
                ea.user_id AS user,
                MAX(CASE WHEN ea.status IN ('yes', 'maybe') THEN 1 ELSE 0 END)::INTEGER AS target,
                MAX(MAX(CASE WHEN ea.status IN ('yes', 'maybe') THEN 1 ELSE 0 END))
                    OVER(
                        PARTITION BY t.phase, ea.user_id
                        ORDER BY t.timestamp
                        ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING
                    ) AS prev_target
            FROM timestamp_df t
            LEFT JOIN (
                SELECT user_id, status, start_time
                FROM read_parquet(?)
                WHERE user_id IS NOT NULL
            ) ea
            ON ea.start_time > t.timestamp
               AND ea.start_time <= t.timestamp + INTERVAL 7 DAY
            GROUP BY t.timestamp, t.phase, ea.user_id
        )
        SELECT timestamp, phase, user::BIGINT AS user, target
        FROM tb
        WHERE prev_target = 1
          AND user IS NOT NULL
          AND timestamp >= ?
        """,
        [attendee_path, BASE_TIME],
    ).df()
    connection.close()
    replay["timestamp"] = pd.to_datetime(replay["timestamp"])
    replay["user"] = replay["user"].astype(np.int64)
    replay["target"] = replay["target"].astype(np.int8)
    replay["phase"] = replay["phase"].astype(np.int8)
    return replay.sort_values(["timestamp", "user"]).reset_index(drop=True)


def make_model_pools(
    train: pd.DataFrame,
    val: pd.DataFrame,
    attendee_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replay = build_replay(attendee_path, MODEL_B_CUTOFF)
    official = train[["timestamp", "user", "target"]].copy()
    official["phase"] = np.int8(0)
    official["_official"] = True
    check = replay[
        (replay["phase"] == 0) & (replay["timestamp"] <= MODEL_A_CUTOFF)
    ][["timestamp", "user", "target"]].sort_values(["timestamp", "user"]).reset_index(drop=True)
    expected = official[["timestamp", "user", "target"]].sort_values(
        ["timestamp", "user"]
    ).reset_index(drop=True)
    exact = (
        len(check) == len(expected)
        and np.array_equal(
            check["timestamp"].to_numpy(dtype="datetime64[ns]"),
            expected["timestamp"].to_numpy(dtype="datetime64[ns]"),
        )
        and np.array_equal(
            check["user"].to_numpy(dtype=np.int64),
            expected["user"].to_numpy(dtype=np.int64),
        )
        and np.array_equal(
            check["target"].to_numpy(dtype=np.int8),
            expected["target"].to_numpy(dtype=np.int8),
        )
    )
    if not exact:
        raise RuntimeError(
            f"official train replay mismatch: replay={len(check)} official={len(expected)}"
        )
    replay["_official"] = False
    replay_a = replay[replay["timestamp"] <= MODEL_A_CUTOFF].copy()
    pool_a = pd.concat([replay_a, official], ignore_index=True)
    official_val = val[["timestamp", "user", "target"]].copy()
    official_val["phase"] = np.int8(0)
    official_val["_official"] = True
    pool_b = pd.concat([replay, official, official_val], ignore_index=True)
    for pool in (pool_a, pool_b):
        pool.sort_values(
            ["timestamp", "user", "_official"],
            ascending=[True, True, False],
            inplace=True,
        )
        pool.drop_duplicates(["timestamp", "user"], keep="first", inplace=True)
        pool.reset_index(drop=True, inplace=True)
        pool["phase_weight"] = np.where(pool["_official"].to_numpy(), 1.0, 0.5)
        pool["outcome_end"] = pool["timestamp"] + pd.Timedelta(days=7)
    return pool_a, pool_b, replay


def load_sources(paths: dict[str, str]) -> dict[str, pd.DataFrame]:
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={int(os.environ.get('OMP_NUM_THREADS', '1'))}")
    users = connection.execute(
        """
        SELECT user_id::BIGINT AS user_id, locale, birthyear, gender, joinedAt,
               location, timezone
        FROM read_parquet(?)
        ORDER BY user_id
        """,
        [paths["users"]],
    ).df()
    n_users = int(users["user_id"].max()) + 1
    friends = connection.execute(
        """
        SELECT DISTINCT user::BIGINT AS user, friend::BIGINT AS friend
        FROM read_parquet(?)
        WHERE user IS NOT NULL
          AND friend IS NOT NULL
          AND user >= 0
          AND friend >= 0
          AND user < ?
          AND friend < ?
          AND user != friend
        """,
        [paths["user_friends"], n_users, n_users],
    ).df()
    attendees = connection.execute(
        """
        SELECT user_id::BIGINT AS user, event::BIGINT AS event, status, start_time
        FROM read_parquet(?)
        WHERE user_id IS NOT NULL
          AND user_id >= 0
          AND user_id < ?
        ORDER BY user_id, start_time
        """,
        [paths["event_attendees"], n_users],
    ).df()
    interests = connection.execute(
        """
        SELECT user::BIGINT AS user, event::BIGINT AS event, invited::INTEGER AS invited,
               timestamp, interested::INTEGER AS interested,
               not_interested::INTEGER AS not_interested
        FROM read_parquet(?)
        WHERE user IS NOT NULL
          AND user >= 0
          AND user < ?
        ORDER BY user, timestamp
        """,
        [paths["event_interest"], n_users],
    ).df()
    schema = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)", [paths["events"]]
    ).df()
    c_columns = [str(x) for x in schema["column_name"] if str(x).startswith("c_")]
    c_select = ", ".join([f"e.{name}" for name in c_columns])
    events = connection.execute(
        f"""
        WITH relevant AS(
            SELECT DISTINCT event
            FROM read_parquet(?)
            WHERE user_id IS NOT NULL
            UNION
            SELECT DISTINCT event
            FROM read_parquet(?)
            WHERE event IS NOT NULL
        ),
        popularity AS(
            SELECT event,
                   COUNT(*)::DOUBLE AS popularity_total,
                   SUM(CASE WHEN status IN ('yes', 'maybe') THEN 1 ELSE 0 END)::DOUBLE
                       AS popularity_positive
            FROM read_parquet(?)
            GROUP BY event
        )
        SELECT e.event_id::BIGINT AS event_id, e.start_time, e.city, e.state,
               e.country, e.lat, e.lng, p.popularity_total,
               p.popularity_positive, {c_select}
        FROM read_parquet(?) e
        INNER JOIN relevant r ON e.event_id = r.event
        LEFT JOIN popularity p ON e.event_id = p.event
        """,
        [
            paths["event_attendees"],
            paths["event_interest"],
            paths["event_attendees"],
            paths["events"],
        ],
    ).df()
    connection.close()
    users["joinedAt"] = pd.to_datetime(users["joinedAt"])
    attendees["start_time"] = pd.to_datetime(attendees["start_time"])
    interests["timestamp"] = pd.to_datetime(interests["timestamp"])
    events["start_time"] = pd.to_datetime(events["start_time"])
    return {
        "users": users,
        "friends": friends,
        "attendees": attendees,
        "interests": interests,
        "events": events,
        "c_columns": c_columns,
    }


# Section: social representation

def build_adjacency(
    friends: pd.DataFrame,
    n_users: int,
) -> tuple[list[np.ndarray], coo_matrix]:
    left = friends["user"].to_numpy(dtype=np.int64)
    right = friends["friend"].to_numpy(dtype=np.int64)
    rows = np.concatenate([left, right])
    cols = np.concatenate([right, left])
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_users, n_users),
    ).tocsr()
    graph.data[:] = 1.0
    graph.eliminate_zeros()
    adjacency = [
        graph.indices[graph.indptr[i] : graph.indptr[i + 1]].astype(np.int64)
        for i in range(n_users)
    ]
    return adjacency, graph


def social_embedding(
    graph,
    cache_dir: Path,
    debug: bool,
) -> tuple[np.ndarray, str]:
    cache_path = cache_dir / "social_svd32.npy"
    if cache_path.exists():
        values = np.load(cache_path, allow_pickle=False)
        if values.shape == (graph.shape[0], SOCIAL_DIM):
            return values.astype(np.float32), "truncated_svd_cached"
    model = TruncatedSVD(
        n_components=SOCIAL_DIM,
        algorithm="randomized",
        n_iter=3 if debug else 7,
        random_state=SEED,
    )
    values = model.fit_transform(graph).astype(np.float32)
    values = l2_normalize(values)
    np.save(cache_path, values)
    return values, "truncated_svd"


# Section: relational histories

def category_arrays(users: pd.DataFrame, column: str, n_users: int) -> tuple[np.ndarray, np.ndarray]:
    values = users.set_index("user_id")[column].reindex(np.arange(n_users))
    text = values.fillna("__MISSING__").astype(str)
    categories = {value: i for i, value in enumerate(sorted(text.unique()))}
    codes = text.map(categories).to_numpy(dtype=np.float32)
    frequencies = text.map(text.value_counts(normalize=True)).to_numpy(dtype=np.float32)
    return codes, frequencies


def prepare_histories(
    sources: dict[str, pd.DataFrame],
    adjacency: list[np.ndarray],
    embedding: np.ndarray,
) -> dict:
    users = sources["users"].copy()
    attendees = sources["attendees"].copy()
    interests = sources["interests"].copy()
    events = sources["events"].copy()
    n_users = len(users)
    rng = np.random.default_rng(SEED)
    projection = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=(len(sources["c_columns"]), PROJECTION_DIM),
    ) / math.sqrt(max(1, len(sources["c_columns"])))
    c_values = ensure_float(events, sources["c_columns"])
    c_values = np.nan_to_num(c_values, nan=0.0, posinf=0.0, neginf=0.0)
    projected = c_values @ projection
    projection_names = [f"event_projection_{i:02d}" for i in range(PROJECTION_DIM)]
    for i, name in enumerate(projection_names):
        events[name] = projected[:, i]
    event_columns = [
        "event_id",
        "start_time",
        "city",
        "state",
        "country",
        "lat",
        "lng",
        "popularity_total",
        "popularity_positive",
    ] + projection_names
    attendees = attendees.merge(
        events[event_columns],
        left_on="event",
        right_on="event_id",
        how="left",
        suffixes=("", "_meta"),
    )
    event_start = events.set_index("event_id")["start_time"]
    interests["event_start"] = interests["event"].map(event_start)
    positive_pairs = set(
        zip(
            attendees.loc[attendees["status"].isin(["yes", "maybe"]), "user"].astype(int),
            attendees.loc[attendees["status"].isin(["yes", "maybe"]), "event"].astype(int),
        )
    )
    interests["converted"] = [
        float((int(u), int(e)) in positive_pairs) if pd.notna(e) else 0.0
        for u, e in zip(interests["user"], interests["event"])
    ]
    attendance_histories: dict[int, dict[str, np.ndarray]] = {}
    for user, group in attendees.groupby("user", sort=False):
        group = group.sort_values("start_time")
        status = group["status"].astype(str).to_numpy()
        attendance_histories[int(user)] = {
            "times": group["start_time"].to_numpy(dtype="datetime64[ns]"),
            "events": group["event"].fillna(-1).to_numpy(dtype=np.int64),
            "positive": np.isin(status, ["yes", "maybe"]).astype(np.int8),
            "yes": (status == "yes").astype(np.int8),
            "maybe": (status == "maybe").astype(np.int8),
            "no": (status == "no").astype(np.int8),
            "invited": (status == "invited").astype(np.int8),
            "popularity_total": group["popularity_total"].fillna(0).to_numpy(dtype=np.float32),
            "popularity_positive": group["popularity_positive"].fillna(0).to_numpy(dtype=np.float32),
            "lat": group["lat"].to_numpy(dtype=np.float32, na_value=np.nan),
            "lng": group["lng"].to_numpy(dtype=np.float32, na_value=np.nan),
            "city": group["city"].fillna("__MISSING__").astype(str).to_numpy(),
            "state": group["state"].fillna("__MISSING__").astype(str).to_numpy(),
            "country": group["country"].fillna("__MISSING__").astype(str).to_numpy(),
            "projection": group[projection_names].to_numpy(dtype=np.float32, na_value=np.nan),
        }
    interest_histories: dict[int, dict[str, np.ndarray]] = {}
    for user, group in interests.groupby("user", sort=False):
        group = group.sort_values("timestamp")
        interest_histories[int(user)] = {
            "times": group["timestamp"].to_numpy(dtype="datetime64[ns]"),
            "event_start": group["event_start"].to_numpy(dtype="datetime64[ns]"),
            "invited": group["invited"].fillna(0).to_numpy(dtype=np.float32),
            "interested": group["interested"].fillna(0).to_numpy(dtype=np.float32),
            "not_interested": group["not_interested"].fillna(0).to_numpy(dtype=np.float32),
            "converted": group["converted"].to_numpy(dtype=np.float32),
        }
    friend_records: list[list[tuple[np.datetime64, int, int, int]]] = [
        [] for _ in range(n_users)
    ]
    for row in attendees[["user", "event", "start_time", "status"]].itertuples(index=False):
        actor = int(row.user)
        positive = int(row.status in ("yes", "maybe"))
        event = int(row.event) if pd.notna(row.event) else -1
        for user in adjacency[actor]:
            friend_records[int(user)].append(
                (np.datetime64(row.start_time), actor, event, positive)
            )
    friend_histories: dict[int, dict[str, np.ndarray]] = {}
    for user, records in enumerate(friend_records):
        if not records:
            continue
        records.sort(key=lambda value: value[0])
        friend_histories[user] = {
            "times": np.array([x[0] for x in records], dtype="datetime64[ns]"),
            "actors": np.array([x[1] for x in records], dtype=np.int64),
            "events": np.array([x[2] for x in records], dtype=np.int64),
            "positive": np.array([x[3] for x in records], dtype=np.int8),
        }
    indexed = users.set_index("user_id").reindex(np.arange(n_users))
    birthyear = indexed["birthyear"].to_numpy(dtype=np.float32, na_value=np.nan)
    timezone = indexed["timezone"].to_numpy(dtype=np.float32, na_value=np.nan)
    joined = indexed["joinedAt"].to_numpy(dtype="datetime64[ns]")
    locale_code, locale_frequency = category_arrays(users, "locale", n_users)
    location_code, location_frequency = category_arrays(users, "location", n_users)
    gender_code, gender_frequency = category_arrays(users, "gender", n_users)
    location_missing = indexed["location"].isna().to_numpy(dtype=np.float32)
    degree = np.array([len(x) for x in adjacency], dtype=np.float32)
    homophily = np.zeros((n_users, 7), dtype=np.float32)
    for user in range(n_users):
        neighbors = adjacency[user]
        if len(neighbors) == 0:
            homophily[user, 6] = 1.0
            continue
        homophily[user, 0] = np.mean(gender_code[neighbors] == gender_code[user])
        homophily[user, 1] = np.mean(locale_code[neighbors] == locale_code[user])
        homophily[user, 2] = np.mean(location_code[neighbors] == location_code[user])
        valid_age = np.isfinite(birthyear[neighbors])
        homophily[user, 3] = (
            float(np.mean(birthyear[neighbors][valid_age])) if valid_age.any() else np.nan
        )
        valid_tz = np.isfinite(timezone[neighbors])
        homophily[user, 4] = (
            float(np.mean(timezone[neighbors][valid_tz])) if valid_tz.any() else np.nan
        )
        homophily[user, 5] = float(np.mean(~valid_age))
        homophily[user, 6] = 0.0
    return {
        "attendance": attendance_histories,
        "interest": interest_histories,
        "friend_activity": friend_histories,
        "adjacency": adjacency,
        "embedding": embedding,
        "birthyear": birthyear,
        "timezone": timezone,
        "joined": joined,
        "locale_code": locale_code,
        "locale_frequency": locale_frequency,
        "location_code": location_code,
        "location_frequency": location_frequency,
        "location_missing": location_missing,
        "gender_code": gender_code,
        "gender_frequency": gender_frequency,
        "degree": degree,
        "homophily": homophily,
        "projection_names": projection_names,
    }


# Section: behavior feature matrix

def recency(times: np.ndarray, end: int, query_time: np.datetime64, mask: np.ndarray | None = None) -> float:
    if end == 0:
        return 730.0
    candidate = times[:end] if mask is None else times[:end][mask]
    if len(candidate) == 0:
        return 730.0
    return float(np.clip(days_between(query_time, candidate[-1]), 0.0, 730.0))


def window_count(
    times: np.ndarray,
    values: np.ndarray,
    end: int,
    query_time: np.datetime64,
    days: int,
) -> float:
    begin_time = query_time - np.timedelta64(days, "D")
    begin = int(np.searchsorted(times, begin_time, side="right"))
    return float(values[begin:end].sum())


def unique_fraction(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(len(np.unique(values)) / len(values))


def build_behavior_features(states: pd.DataFrame, histories: dict) -> tuple[np.ndarray, list[str]]:
    windows = [7, 14, 28, 56, 90, 180]
    names = [
        "age",
        "birth_missing",
        "birthyear",
        "gender_code",
        "gender_frequency",
        "locale_code",
        "locale_frequency",
        "location_code",
        "location_frequency",
        "location_missing",
        "timezone",
        "timezone_missing",
        "timezone_sin",
        "timezone_cos",
        "tenure_days",
        "joined_after_query",
        "degree",
        "log_degree",
        "isolated",
        "friend_gender_homophily",
        "friend_locale_homophily",
        "friend_location_homophily",
        "friend_mean_birthyear",
        "friend_mean_timezone",
        "friend_birth_missing",
    ]
    for days in windows:
        names.extend([
            f"att_any_{days}",
            f"att_positive_{days}",
            f"att_positive_ratio_{days}",
        ])
    names.extend([
        "att_total",
        "att_unique_events",
        "att_yes_total",
        "att_maybe_total",
        "att_no_total",
        "att_invited_total",
        "att_positive_total",
        "att_positive_ratio",
        "att_yes_ratio",
        "att_maybe_ratio",
        "att_no_ratio",
        "att_invited_ratio",
        "att_recency_any",
        "att_recency_positive",
        "att_recency_yes",
        "att_recency_maybe",
        "att_recency_no",
        "att_gap_mean",
        "att_gap_median",
        "att_gap_std",
        "att_gap_min",
        "att_gap_max",
        "att_gap_last",
        "att_renewal_hazard",
        "att_weekly_streak",
        "att_last7_minus_prev7",
        "att_last14_minus_prev14",
        "att_momentum_ratio",
        "event_popularity_mean",
        "event_popularity_max",
        "event_positive_popularity_mean",
        "event_positive_popularity_max",
        "event_geo_fraction",
        "event_lat_mean",
        "event_lat_std",
        "event_lng_mean",
        "event_lng_std",
        "event_city_diversity",
        "event_state_diversity",
        "event_country_diversity",
    ])
    names.extend(histories["projection_names"])
    for days in windows:
        names.extend([
            f"interest_count_{days}",
            f"interest_positive_{days}",
            f"interest_negative_{days}",
        ])
    names.extend([
        "interest_total",
        "interest_invited_total",
        "interest_positive_total",
        "interest_negative_total",
        "interest_positive_ratio",
        "interest_negative_ratio",
        "interest_recency",
        "interest_positive_recency",
        "interest_negative_recency",
        "interest_slope_28",
        "interest_past_event_count",
        "interest_conversion_count",
        "interest_conversion_rate",
    ])
    for days in windows:
        names.extend([
            f"social_activity_{days}",
            f"social_positive_{days}",
            f"social_active_friends_{days}",
        ])
    names.extend([
        "social_activity_total",
        "social_positive_total",
        "social_positive_ratio",
        "social_activity_recency",
        "social_positive_recency",
        "social_active_friends_total",
        "social_active_friend_fraction",
        "social_coattendance_events",
        "social_coattendance_fraction",
    ])
    names.extend([f"social_embedding_{i:02d}" for i in range(SOCIAL_DIM)])
    rows: list[list[float]] = []
    for row in states[["timestamp", "user"]].itertuples(index=False):
        query_time = np.datetime64(row.timestamp, "ns")
        user = int(row.user)
        birthyear = histories["birthyear"][user]
        timezone = histories["timezone"][user]
        joined = histories["joined"][user]
        year = pd.Timestamp(row.timestamp).year
        age = year - birthyear if np.isfinite(birthyear) else np.nan
        joined_valid = not np.isnat(joined) and joined <= query_time
        tenure = float(days_between(query_time, joined)) if joined_valid else 0.0
        values: list[float] = [
            age,
            float(not np.isfinite(birthyear)),
            birthyear,
            histories["gender_code"][user],
            histories["gender_frequency"][user],
            histories["locale_code"][user],
            histories["locale_frequency"][user],
            histories["location_code"][user],
            histories["location_frequency"][user],
            histories["location_missing"][user],
            timezone,
            float(not np.isfinite(timezone)),
            math.sin(float(timezone) * math.pi / 12.0) if np.isfinite(timezone) else 0.0,
            math.cos(float(timezone) * math.pi / 12.0) if np.isfinite(timezone) else 0.0,
            max(0.0, tenure),
            float(not joined_valid),
            histories["degree"][user],
            math.log1p(histories["degree"][user]),
            float(histories["degree"][user] == 0),
        ]
        values.extend(histories["homophily"][user, :6].tolist())
        attendance = histories["attendance"].get(user)
        user_positive_events: set[int] = set()
        if attendance is None:
            for _ in windows:
                values.extend([0.0, 0.0, 0.5])
            values.extend([
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.5, 0.0, 0.0, 0.0, 0.0,
                730.0, 730.0, 730.0, 730.0, 730.0,
                90.0, 90.0, 0.0, 90.0, 90.0, 90.0, 8.0,
                0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ])
            values.extend([0.0] * PROJECTION_DIM)
        else:
            times = attendance["times"]
            end = int(np.searchsorted(times, query_time, side="right"))
            positive = attendance["positive"][:end]
            for days in windows:
                any_count = window_count(times, np.ones(len(times)), end, query_time, days)
                pos_count = window_count(times, attendance["positive"], end, query_time, days)
                values.extend([any_count, pos_count, (pos_count + 1.0) / (any_count + 2.0)])
            total = float(end)
            yes_total = float(attendance["yes"][:end].sum())
            maybe_total = float(attendance["maybe"][:end].sum())
            no_total = float(attendance["no"][:end].sum())
            invited_total = float(attendance["invited"][:end].sum())
            positive_total = float(positive.sum())
            pos_times = times[:end][positive.astype(bool)]
            pos_events = attendance["events"][:end][positive.astype(bool)]
            user_positive_events = set(pos_events[pos_events >= 0].tolist())
            gaps = (
                np.diff(pos_times).astype("timedelta64[s]").astype(np.float64) / 86400.0
                if len(pos_times) > 1
                else np.array([], dtype=np.float64)
            )
            gap_default = 90.0
            last_gap = float(gaps[-1]) if len(gaps) else gap_default
            pos_recency = recency(times, end, query_time, positive.astype(bool))
            weekly = [
                window_count(times, attendance["positive"], end, query_time, days)
                - (
                    window_count(times, attendance["positive"], end, query_time, days - 7)
                    if days > 7 else 0.0
                )
                for days in [7, 14, 21, 28]
            ]
            streak = 0
            for count in weekly:
                if count > 0:
                    streak += 1
                else:
                    break
            selected = positive.astype(bool)
            pop_total = attendance["popularity_total"][:end][selected]
            pop_positive = attendance["popularity_positive"][:end][selected]
            lat = attendance["lat"][:end][selected]
            lng = attendance["lng"][:end][selected]
            valid_geo = np.isfinite(lat) & np.isfinite(lng)
            projection_values = attendance["projection"][:end][selected]
            projection_mean = (
                np.nanmean(projection_values, axis=0)
                if len(projection_values)
                else np.zeros(PROJECTION_DIM, dtype=np.float32)
            )
            projection_mean = np.nan_to_num(projection_mean, nan=0.0)
            values.extend([
                total,
                float(len(np.unique(attendance["events"][:end]))) if end else 0.0,
                yes_total,
                maybe_total,
                no_total,
                invited_total,
                positive_total,
                (positive_total + 1.0) / (total + 2.0),
                (yes_total + 0.5) / (total + 2.0),
                (maybe_total + 0.5) / (total + 2.0),
                (no_total + 0.5) / (total + 2.0),
                (invited_total + 0.5) / (total + 2.0),
                recency(times, end, query_time),
                pos_recency,
                recency(times, end, query_time, attendance["yes"][:end].astype(bool)),
                recency(times, end, query_time, attendance["maybe"][:end].astype(bool)),
                recency(times, end, query_time, attendance["no"][:end].astype(bool)),
                float(np.mean(gaps)) if len(gaps) else gap_default,
                float(np.median(gaps)) if len(gaps) else gap_default,
                float(np.std(gaps)) if len(gaps) else 0.0,
                float(np.min(gaps)) if len(gaps) else gap_default,
                float(np.max(gaps)) if len(gaps) else gap_default,
                last_gap,
                pos_recency / max(1.0, float(np.mean(gaps)) if len(gaps) else gap_default),
                float(streak),
                weekly[0] - weekly[1],
                sum(weekly[:2]) - sum(weekly[2:]),
                (sum(weekly[:2]) + 1.0) / (sum(weekly[2:]) + 1.0),
                float(np.mean(np.log1p(pop_total))) if len(pop_total) else 0.0,
                float(np.max(np.log1p(pop_total))) if len(pop_total) else 0.0,
                float(np.mean(np.log1p(pop_positive))) if len(pop_positive) else 0.0,
                float(np.max(np.log1p(pop_positive))) if len(pop_positive) else 0.0,
                float(valid_geo.mean()) if len(valid_geo) else 0.0,
                float(np.nanmean(lat)) if valid_geo.any() else 0.0,
                float(np.nanstd(lat)) if valid_geo.any() else 0.0,
                float(np.nanmean(lng)) if valid_geo.any() else 0.0,
                float(np.nanstd(lng)) if valid_geo.any() else 0.0,
                unique_fraction(attendance["city"][:end][selected]),
                unique_fraction(attendance["state"][:end][selected]),
                unique_fraction(attendance["country"][:end][selected]),
            ])
            values.extend(projection_mean.tolist())
        interest = histories["interest"].get(user)
        if interest is None:
            values.extend([0.0, 0.0, 0.0] * len(windows))
            values.extend([
                0.0, 0.0, 0.0, 0.0, 0.5, 0.5,
                730.0, 730.0, 730.0, 0.0, 0.0, 0.0, 0.5,
            ])
        else:
            times = interest["times"]
            end = int(np.searchsorted(times, query_time, side="right"))
            for days in windows:
                values.extend([
                    window_count(times, np.ones(len(times)), end, query_time, days),
                    window_count(times, interest["interested"], end, query_time, days),
                    window_count(times, interest["not_interested"], end, query_time, days),
                ])
            total = float(end)
            positive_total = float(interest["interested"][:end].sum())
            negative_total = float(interest["not_interested"][:end].sum())
            recent_28 = window_count(times, np.ones(len(times)), end, query_time, 28)
            recent_56 = window_count(times, np.ones(len(times)), end, query_time, 56)
            event_start = interest["event_start"][:end]
            past_mask = (~np.isnat(event_start)) & (event_start <= query_time)
            converted = interest["converted"][:end][past_mask]
            values.extend([
                total,
                float(interest["invited"][:end].sum()),
                positive_total,
                negative_total,
                (positive_total + 1.0) / (total + 2.0),
                (negative_total + 1.0) / (total + 2.0),
                recency(times, end, query_time),
                recency(times, end, query_time, interest["interested"][:end].astype(bool)),
                recency(times, end, query_time, interest["not_interested"][:end].astype(bool)),
                recent_28 - (recent_56 - recent_28),
                float(past_mask.sum()),
                float(converted.sum()) if len(converted) else 0.0,
                (float(converted.sum()) + 1.0) / (len(converted) + 2.0),
            ])
        social = histories["friend_activity"].get(user)
        if social is None:
            values.extend([0.0, 0.0, 0.0] * len(windows))
            values.extend([0.0, 0.0, 0.5, 730.0, 730.0, 0.0, 0.0, 0.0, 0.0])
        else:
            times = social["times"]
            end = int(np.searchsorted(times, query_time, side="right"))
            for days in windows:
                begin = int(np.searchsorted(times, query_time - np.timedelta64(days, "D"), side="right"))
                values.extend([
                    float(end - begin),
                    float(social["positive"][begin:end].sum()),
                    float(len(np.unique(social["actors"][begin:end]))),
                ])
            positive_mask = social["positive"][:end].astype(bool)
            social_positive_events = set(
                social["events"][:end][positive_mask & (social["events"][:end] >= 0)].tolist()
            )
            shared_events = user_positive_events.intersection(social_positive_events)
            active_friends = len(np.unique(social["actors"][:end])) if end else 0
            positive_total = float(social["positive"][:end].sum())
            values.extend([
                float(end),
                positive_total,
                (positive_total + 1.0) / (end + 2.0),
                recency(times, end, query_time),
                recency(times, end, query_time, positive_mask),
                float(active_friends),
                active_friends / max(1.0, histories["degree"][user]),
                float(len(shared_events)),
                len(shared_events) / max(1.0, len(user_positive_events)),
            ])
        values.extend(histories["embedding"][user].tolist())
        rows.append(values)
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.shape[1] != len(names):
        raise RuntimeError(f"behavior feature mismatch: {matrix.shape[1]} != {len(names)}")
    return matrix, names


# Section: retrieval and empirical Bayes

def retrieval_feature_names(k_values: list[int]) -> list[str]:
    names: list[str] = []
    for k in k_values:
        names.extend([
            f"retrieval_rate_{k}",
            f"retrieval_logit_{k}",
            f"retrieval_similarity_q10_{k}",
            f"retrieval_similarity_q50_{k}",
            f"retrieval_similarity_q90_{k}",
            f"retrieval_positive_distance_{k}",
            f"retrieval_negative_distance_{k}",
            f"retrieval_distance_gap_{k}",
            f"retrieval_age_{k}",
            f"retrieval_community_{k}",
            f"retrieval_count_{k}",
        ])
    return names


def retrieval_vectors(
    pool_base: np.ndarray,
    query_base: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    preferred = [
        "age",
        "timezone_sin",
        "timezone_cos",
        "tenure_days",
        "log_degree",
        "friend_gender_homophily",
        "friend_locale_homophily",
        "friend_location_homophily",
        "att_positive_7",
        "att_positive_14",
        "att_positive_28",
        "att_positive_56",
        "att_positive_90",
        "att_positive_ratio_14",
        "att_positive_ratio_56",
        "att_recency_positive",
        "att_gap_mean",
        "att_gap_last",
        "att_renewal_hazard",
        "att_weekly_streak",
        "att_momentum_ratio",
        "event_popularity_mean",
        "event_positive_popularity_mean",
        "event_geo_fraction",
        "interest_count_28",
        "interest_positive_56",
        "interest_positive_ratio",
        "interest_conversion_rate",
        "social_positive_14",
        "social_positive_56",
        "social_active_friend_fraction",
        "social_coattendance_fraction",
    ]
    preferred.extend([f"social_embedding_{i:02d}" for i in range(SOCIAL_DIM)])
    indices = [names.index(name) for name in preferred if name in names]
    mean, scale = standardize_fit(pool_base[:, indices])
    pool = standardize_apply(pool_base[:, indices], mean, scale)
    query = standardize_apply(query_base[:, indices], mean, scale)
    return l2_normalize(pool), l2_normalize(query)


def exact_retrieval_features(
    pool: pd.DataFrame,
    query: pd.DataFrame,
    pool_vectors: np.ndarray,
    query_vectors: np.ndarray,
    social_embedding_values: np.ndarray,
    k_values: list[int],
) -> tuple[np.ndarray, list[str]]:
    names = retrieval_feature_names(k_values)
    output = np.zeros((len(query), len(names)), dtype=np.float32)
    pool_users = pool["user"].to_numpy(dtype=np.int64)
    pool_y = pool["target"].to_numpy(dtype=np.float32)
    pool_end = pool["outcome_end"].to_numpy(dtype="datetime64[ns]")
    pool_weight = pool["phase_weight"].to_numpy(dtype=np.float32)
    query_users = query["user"].to_numpy(dtype=np.int64)
    query_times = query["timestamp"].to_numpy(dtype="datetime64[ns]")
    for query_time in np.unique(query_times):
        query_indices = np.flatnonzero(query_times == query_time)
        eligible = np.flatnonzero(pool_end <= query_time)
        if len(eligible) == 0:
            for row_index in query_indices:
                values: list[float] = []
                for _ in k_values:
                    values.extend([0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 365.0, 0.0, 0.0])
                output[row_index] = values
            continue
        similarities = query_vectors[query_indices] @ pool_vectors[eligible].T
        same = query_users[query_indices, None] == pool_users[eligible][None, :]
        similarities[same] = -np.inf
        maximum_k = min(max(k_values), len(eligible))
        partitions = np.argpartition(similarities, -maximum_k, axis=1)[:, -maximum_k:]
        partition_similarity = np.take_along_axis(similarities, partitions, axis=1)
        order = np.argsort(partition_similarity, axis=1)[:, ::-1]
        sorted_local = np.take_along_axis(partitions, order, axis=1)
        sorted_similarity = np.take_along_axis(similarities, sorted_local, axis=1)
        for local_row, row_index in enumerate(query_indices):
            valid = np.isfinite(sorted_similarity[local_row])
            local = sorted_local[local_row][valid]
            similarity = sorted_similarity[local_row][valid]
            values = []
            for k in k_values:
                take = min(k, len(local))
                if take == 0:
                    values.extend([0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 365.0, 0.0, 0.0])
                    continue
                selected = eligible[local[:take]]
                selected_similarity = similarity[:take]
                ages = days_between(query_time, pool_end[selected])
                weights = np.exp(
                    np.clip(
                        (selected_similarity - selected_similarity.max()) / RETRIEVAL_TEMPERATURE,
                        -40.0,
                        0.0,
                    )
                )
                weights *= np.exp(-np.maximum(0.0, ages) * math.log(2.0) / RETRIEVAL_HALF_LIFE_DAYS)
                weights *= pool_weight[selected]
                weight_sum = float(weights.sum())
                if weight_sum <= 1e-9:
                    weights = np.ones(take, dtype=np.float64)
                    weight_sum = float(take)
                labels = pool_y[selected]
                rate = float(np.dot(weights, labels) / weight_sum)
                positive_similarity = selected_similarity[labels > 0.5]
                negative_similarity = selected_similarity[labels <= 0.5]
                positive_distance = (
                    float(1.0 - positive_similarity.max()) if len(positive_similarity) else 2.0
                )
                negative_distance = (
                    float(1.0 - negative_similarity.max()) if len(negative_similarity) else 2.0
                )
                query_social = social_embedding_values[query_users[row_index]]
                neighbor_social = social_embedding_values[pool_users[selected]]
                community = float(np.dot(weights, neighbor_social @ query_social) / weight_sum)
                values.extend([
                    rate,
                    clipped_logit(rate),
                    float(np.quantile(selected_similarity, 0.1)),
                    float(np.quantile(selected_similarity, 0.5)),
                    float(np.quantile(selected_similarity, 0.9)),
                    positive_distance,
                    negative_distance,
                    negative_distance - positive_distance,
                    float(np.dot(weights, ages) / weight_sum),
                    community,
                    float(take),
                ])
            output[row_index] = values
    return output, names


def cohort_keys(base: np.ndarray, names: list[str]) -> list[tuple[int, int, int, int, int]]:
    index = {name: names.index(name) for name in [
        "gender_code", "age", "timezone", "att_positive_56", "degree"
    ]}
    output = []
    for row in base:
        age = row[index["age"]]
        timezone = row[index["timezone"]]
        output.append((
            int(row[index["gender_code"]]) if np.isfinite(row[index["gender_code"]]) else -1,
            int(np.digitize(age, [22, 30, 40, 55])) if np.isfinite(age) else -1,
            int(np.digitize(timezone, [-6, 0, 6])) if np.isfinite(timezone) else -1,
            int(np.digitize(row[index["att_positive_56"]], [1, 2, 4, 8])),
            int(np.digitize(row[index["degree"]], [1, 4, 12, 30])),
        ))
    return output


def eb_feature_names() -> list[str]:
    return [
        "eb_global_rate",
        "eb_user_rate",
        "eb_user_logit",
        "eb_user_count",
        "eb_user_raw_rate",
        "eb_user_last_target",
        "eb_user_age",
        "eb_user_phase_rate",
        "eb_user_phase_logit",
        "eb_user_phase_count",
        "eb_user_phase_last_target",
        "eb_user_recent_rate",
        "eb_social_rate",
        "eb_social_logit",
        "eb_social_count",
        "eb_social_raw_rate",
        "eb_social_recent_rate",
        "eb_cohort_rate",
        "eb_cohort_logit",
        "eb_cohort_count",
        "eb_cohort_raw_rate",
        "eb_cohort_recent_rate",
        "eb_recurrence_blend",
    ]


def empirical_bayes_features(
    pool: pd.DataFrame,
    query: pd.DataFrame,
    pool_base: np.ndarray,
    query_base: np.ndarray,
    base_names: list[str],
    adjacency: list[np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    names = eb_feature_names()
    output = np.zeros((len(query), len(names)), dtype=np.float32)
    pool_users = pool["user"].to_numpy(dtype=np.int64)
    pool_y = pool["target"].to_numpy(dtype=np.float64)
    pool_end = pool["outcome_end"].to_numpy(dtype="datetime64[ns]")
    pool_weight = pool["phase_weight"].to_numpy(dtype=np.float64)
    pool_phase = pool["phase"].to_numpy(dtype=np.int8)
    query_users = query["user"].to_numpy(dtype=np.int64)
    query_times = query["timestamp"].to_numpy(dtype="datetime64[ns]")
    if "phase" in query:
        query_phase = query["phase"].to_numpy(dtype=np.int8)
    else:
        query_phase = (
            (
                query["timestamp"] - BASE_TIME
            ).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
        ).astype(np.int64) % 7
    pool_cohort = cohort_keys(pool_base, base_names)
    query_cohort = cohort_keys(query_base, base_names)
    attendance_ratio_index = base_names.index("att_positive_ratio")
    by_user: dict[int, np.ndarray] = {}
    for user in np.unique(pool_users):
        by_user[int(user)] = np.flatnonzero(pool_users == user)
    by_cohort: dict[tuple[int, int, int, int, int], np.ndarray] = {}
    cohort_groups: dict[tuple[int, int, int, int, int], list[int]] = {}
    for index, key in enumerate(pool_cohort):
        cohort_groups.setdefault(key, []).append(index)
    for key, indices in cohort_groups.items():
        by_cohort[key] = np.asarray(indices, dtype=np.int64)
    for row_index, (query_time, user) in enumerate(zip(query_times, query_users)):
        eligible = pool_end <= query_time
        global_weights = pool_weight[eligible]
        global_rate = (
            float(np.dot(global_weights, pool_y[eligible]) / global_weights.sum())
            if global_weights.sum() > 0 else 0.5
        )
        user_indices = by_user.get(int(user), np.empty(0, dtype=np.int64))
        user_indices = user_indices[pool_end[user_indices] <= query_time]
        user_weights = pool_weight[user_indices]
        user_count = float(user_weights.sum())
        user_positive = float(np.dot(user_weights, pool_y[user_indices])) if len(user_indices) else 0.0
        user_rate = (user_positive + PRIOR_STRENGTH * global_rate) / (
            user_count + PRIOR_STRENGTH
        )
        user_raw = user_positive / user_count if user_count > 0 else global_rate
        if len(user_indices):
            latest = user_indices[np.argmax(pool_end[user_indices])]
            user_last = float(pool_y[latest])
            user_age = float(days_between(query_time, pool_end[latest]))
        else:
            user_last = global_rate
            user_age = 365.0
        phase_indices = user_indices[
            pool_phase[user_indices] == query_phase[row_index]
        ]
        phase_weights = pool_weight[phase_indices]
        phase_count = float(phase_weights.sum())
        phase_positive = (
            float(np.dot(phase_weights, pool_y[phase_indices]))
            if len(phase_indices) else 0.0
        )
        phase_rate = (phase_positive + PRIOR_STRENGTH * global_rate) / (
            phase_count + PRIOR_STRENGTH
        )
        if len(phase_indices):
            phase_latest = phase_indices[np.argmax(pool_end[phase_indices])]
            phase_last = float(pool_y[phase_latest])
        else:
            phase_last = global_rate
        recent_threshold = query_time - np.timedelta64(90, "D")
        user_recent_indices = user_indices[pool_end[user_indices] > recent_threshold]
        user_recent_weights = pool_weight[user_recent_indices]
        user_recent_count = float(user_recent_weights.sum())
        user_recent_positive = (
            float(np.dot(user_recent_weights, pool_y[user_recent_indices]))
            if len(user_recent_indices) else 0.0
        )
        user_recent_rate = (
            user_recent_positive + PRIOR_STRENGTH * global_rate
        ) / (user_recent_count + PRIOR_STRENGTH)
        neighbors = adjacency[int(user)]
        social_parts = [by_user.get(int(friend)) for friend in neighbors]
        social_parts = [part for part in social_parts if part is not None and len(part)]
        if social_parts:
            social_indices = np.concatenate(social_parts)
            social_indices = social_indices[pool_end[social_indices] <= query_time]
        else:
            social_indices = np.empty(0, dtype=np.int64)
        social_weights = pool_weight[social_indices]
        social_count = float(social_weights.sum())
        social_positive = (
            float(np.dot(social_weights, pool_y[social_indices]))
            if len(social_indices) else 0.0
        )
        social_rate = (social_positive + PRIOR_STRENGTH * global_rate) / (
            social_count + PRIOR_STRENGTH
        )
        social_raw = social_positive / social_count if social_count > 0 else global_rate
        social_recent_indices = social_indices[
            pool_end[social_indices] > recent_threshold
        ]
        social_recent_weights = pool_weight[social_recent_indices]
        social_recent_count = float(social_recent_weights.sum())
        social_recent_positive = (
            float(np.dot(social_recent_weights, pool_y[social_recent_indices]))
            if len(social_recent_indices) else 0.0
        )
        social_recent_rate = (
            social_recent_positive + PRIOR_STRENGTH * global_rate
        ) / (social_recent_count + PRIOR_STRENGTH)
        cohort_indices = by_cohort.get(query_cohort[row_index], np.empty(0, dtype=np.int64))
        cohort_indices = cohort_indices[pool_end[cohort_indices] <= query_time]
        cohort_weights = pool_weight[cohort_indices]
        cohort_count = float(cohort_weights.sum())
        cohort_positive = (
            float(np.dot(cohort_weights, pool_y[cohort_indices]))
            if len(cohort_indices) else 0.0
        )
        cohort_rate = (cohort_positive + PRIOR_STRENGTH * global_rate) / (
            cohort_count + PRIOR_STRENGTH
        )
        cohort_raw = cohort_positive / cohort_count if cohort_count > 0 else global_rate
        cohort_recent_indices = cohort_indices[
            pool_end[cohort_indices] > recent_threshold
        ]
        cohort_recent_weights = pool_weight[cohort_recent_indices]
        cohort_recent_count = float(cohort_recent_weights.sum())
        cohort_recent_positive = (
            float(np.dot(cohort_recent_weights, pool_y[cohort_recent_indices]))
            if len(cohort_recent_indices) else 0.0
        )
        cohort_recent_rate = (
            cohort_recent_positive + PRIOR_STRENGTH * global_rate
        ) / (cohort_recent_count + PRIOR_STRENGTH)
        recurrence = (
            0.20 * float(query_base[row_index, attendance_ratio_index])
            + 0.40 * user_recent_rate
            + 0.20 * social_recent_rate
            + 0.20 * cohort_recent_rate
        )
        output[row_index] = [
            global_rate,
            user_rate,
            clipped_logit(user_rate),
            user_count,
            user_raw,
            user_last,
            user_age,
            phase_rate,
            clipped_logit(phase_rate),
            phase_count,
            phase_last,
            user_recent_rate,
            social_rate,
            clipped_logit(social_rate),
            social_count,
            social_raw,
            social_recent_rate,
            cohort_rate,
            clipped_logit(cohort_rate),
            cohort_count,
            cohort_raw,
            cohort_recent_rate,
            recurrence,
        ]
    return output, names


# Section: context and diversity models

def context_weights(pool: pd.DataFrame) -> np.ndarray:
    age = (
        pool["timestamp"].max() - pool["timestamp"]
    ).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
    return (
        pool["phase_weight"].to_numpy(dtype=np.float64)
        * np.exp(-age * math.log(2.0) / CONTEXT_HALF_LIFE_DAYS)
    )


def time_buckets(pool: pd.DataFrame) -> np.ndarray:
    times = pool["timestamp"].astype("int64").to_numpy()
    quantiles = np.quantile(times, [0.25, 0.50, 0.75])
    return np.digitize(times, quantiles, right=True)


def sample_context(
    pool: pd.DataFrame,
    rows: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = pool["target"].to_numpy(dtype=np.int8)
    buckets = time_buckets(pool)
    weights = context_weights(pool)
    rows = min(rows, len(pool))
    selected: list[int] = []
    allocation = max(1, rows // 8)
    for target in [0, 1]:
        for bucket in range(4):
            indices = np.flatnonzero((y == target) & (buckets == bucket))
            if len(indices) == 0:
                continue
            take = min(allocation, len(indices))
            probability = weights[indices]
            probability = probability / probability.sum()
            chosen = rng.choice(indices, size=take, replace=False, p=probability)
            selected.extend(chosen.tolist())
    selected_array = np.unique(np.asarray(selected, dtype=np.int64))
    if len(selected_array) < rows:
        remaining = np.setdiff1d(np.arange(len(pool)), selected_array, assume_unique=False)
        take = min(rows - len(selected_array), len(remaining))
        probability = weights[remaining]
        probability = probability / probability.sum()
        extra = rng.choice(remaining, size=take, replace=False, p=probability)
        selected_array = np.concatenate([selected_array, extra])
    return selected_array[:rows]


class TabPFNTimeout(Exception):
    pass


def _timeout_handler(_signum, _frame) -> None:
    raise TabPFNTimeout("TabPFN checkpoint probe exceeded eight minutes")


def probe_tabpfn(cache_dir: Path) -> tuple[bool, str]:
    global LOCAL_TABPFN_MODEL_PATH
    checkpoint_dir = cache_dir.parent / "lane3_tabpfn_v2_checkpoint"
    checkpoint = checkpoint_dir / "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
    config = checkpoint_dir / "config.json"
    if not checkpoint.exists() or not config.exists():
        try:
            from huggingface_hub import hf_hub_download

            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id="Prior-Labs/TabPFN-v2-clf",
                filename=checkpoint.name,
                local_dir=checkpoint_dir,
                token=os.environ.get("HF_TOKEN"),
            )
            hf_hub_download(
                repo_id="Prior-Labs/TabPFN-v2-clf",
                filename="config.json",
                local_dir=checkpoint_dir,
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception:
            pass
    if checkpoint.exists() and config.exists():
        LOCAL_TABPFN_MODEL_PATH = str(checkpoint)
    marker = cache_dir / "tabpfn_unavailable.txt"
    if marker.exists() and LOCAL_TABPFN_MODEL_PATH is None:
        return False, marker.read_text().strip()
    previous = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(480)
    try:
        from tabpfn import TabPFNClassifier

        values = np.random.default_rng(SEED).normal(size=(24, 8)).astype(np.float32)
        labels = np.array([0, 1] * 12, dtype=np.int8)
        model = TabPFNClassifier(
            n_estimators=1,
            model_path=LOCAL_TABPFN_MODEL_PATH or "auto",
            device="cuda:0",
            ignore_pretraining_limits=True,
            random_state=SEED,
            show_progress_bar=False,
            fit_mode="fit_preprocessors",
            n_preprocessing_jobs=1,
        )
        model.fit(values, labels)
        model.predict_proba(values[:2])
        return True, "tabpfn_v2_local_checkpoint"
    except Exception as error:
        reason = f"{type(error).__name__}: {str(error).splitlines()[0]}"
        marker.write_text(reason)
        return False, reason
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def fit_context_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    sample_weight: np.ndarray,
    seed: int,
    use_tabpfn: bool,
    debug: bool,
) -> np.ndarray:
    if use_tabpfn:
        from tabpfn import TabPFNClassifier

        model = TabPFNClassifier(
            n_estimators=8,
            model_path=LOCAL_TABPFN_MODEL_PATH or "auto",
            device="cuda:0",
            ignore_pretraining_limits=True,
            random_state=seed,
            show_progress_bar=False,
            fit_mode="fit_preprocessors",
            n_preprocessing_jobs=1,
        )
        model.fit(train_x, train_y)
        return model.predict_proba(query_x)[:, 1].astype(np.float64)
    model = CatBoostClassifier(
        iterations=60,
        depth=2,
        learning_rate=0.07,
        l2_leaf_reg=10.0,
        loss_function="Logloss",
        random_seed=seed,
        thread_count=int(os.environ.get("OMP_NUM_THREADS", "1")),
        verbose=False,
        allow_writing_files=False,
        random_strength=2.0,
    )
    model.fit(train_x, train_y, sample_weight=sample_weight, verbose=False)
    return model.predict_proba(query_x)[:, 1].astype(np.float64)


def context_ensemble(
    pool: pd.DataFrame,
    train_x: np.ndarray,
    query_x: np.ndarray,
    use_tabpfn: bool,
    debug: bool,
    cache_dir: Path,
    label: str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    context_count = 1 if debug else 6
    context_rows = 2000 if debug else 8192
    predictions = []
    contexts = []
    y = pool["target"].to_numpy(dtype=np.int8)
    weights = context_weights(pool)
    for context_index in range(context_count):
        indices = sample_context(pool, context_rows, SEED + 101 * context_index)
        contexts.append(indices)
        prediction = fit_context_predict(
            train_x[indices],
            y[indices],
            query_x,
            weights[indices],
            SEED + 101 * context_index,
            use_tabpfn,
            debug,
        )
        predictions.append(percentile_rank(prediction))
    np.savez_compressed(
        cache_dir / f"context_row_ids_{label}.npz",
        **{f"context_{i}": values for i, values in enumerate(contexts)},
    )
    return np.mean(np.vstack(predictions), axis=0), contexts


def lightgbm_model(debug: bool, seed: int, trees: int | None = None) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=trees if trees is not None else (100 if debug else 700),
        learning_rate=0.07 if debug else 0.035,
        num_leaves=31,
        min_child_samples=60,
        reg_lambda=10.0,
        reg_alpha=0.5,
        subsample=0.9,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")),
        verbosity=-1,
    )


def lightgbm_predict(
    pool: pd.DataFrame,
    train_x: np.ndarray,
    query_x: np.ndarray,
    debug: bool,
    seed: int,
    trees: int | None = None,
) -> np.ndarray:
    model = lightgbm_model(debug, seed, trees)
    model.fit(
        train_x,
        pool["target"].to_numpy(dtype=np.int8),
        sample_weight=context_weights(pool),
    )
    return model.predict_proba(query_x)[:, 1].astype(np.float64)


# Section: stacker and diagnostics

def forward_oof(
    pool: pd.DataFrame,
    features: np.ndarray,
    peer_prediction: np.ndarray,
    eb_prediction: np.ndarray,
    use_tabpfn: bool,
    debug: bool,
) -> tuple[np.ndarray, np.ndarray]:
    unique_times = np.sort(pool["timestamp"].unique())
    fractions = [0.35, 0.52, 0.69, 0.84, 1.01]
    boundaries = [
        unique_times[min(len(unique_times) - 1, int(len(unique_times) * value))]
        if value <= 1.0 else unique_times[-1] + np.timedelta64(1, "D")
        for value in fractions
    ]
    rows: list[int] = []
    predictions: list[np.ndarray] = []
    for fold in range(4):
        start_time = boundaries[fold]
        end_time = boundaries[fold + 1]
        valid = np.flatnonzero(
            (pool["timestamp"].to_numpy(dtype="datetime64[ns]") >= start_time)
            & (pool["timestamp"].to_numpy(dtype="datetime64[ns]") < end_time)
        )
        train = np.flatnonzero(
            pool["outcome_end"].to_numpy(dtype="datetime64[ns]") <= start_time
        )
        if use_tabpfn and len(valid) > 2048:
            rng = np.random.default_rng(SEED + 1701 + fold)
            probability = pool["phase_weight"].to_numpy(dtype=np.float64)[valid]
            probability = probability / probability.sum()
            valid = np.sort(
                rng.choice(valid, size=2048, replace=False, p=probability)
            )
        if len(valid) == 0 or len(train) < 200:
            continue
        train_pool = pool.iloc[train].reset_index(drop=True)
        context_predictions = []
        oof_context_count = 1 if debug or use_tabpfn else 6
        for context_index in range(oof_context_count):
            context_seed = SEED + 701 + fold + 101 * context_index
            context_local = sample_context(
                train_pool,
                min(2000 if debug else 8192, len(train_pool)),
                context_seed,
            )
            selected = train[context_local]
            context_predictions.append(
                percentile_rank(
                    fit_context_predict(
                        features[selected],
                        pool["target"].to_numpy(dtype=np.int8)[selected],
                        features[valid],
                        context_weights(train_pool)[context_local],
                        context_seed,
                        use_tabpfn,
                        debug,
                    )
                )
            )
        tab = np.mean(context_predictions, axis=0)
        lgb_pool = pool.iloc[train].reset_index(drop=True)
        lgb = lightgbm_predict(
            lgb_pool,
            features[train],
            features[valid],
            debug,
            SEED + 801 + fold,
            trees=80 if debug else 700,
        )
        meta = np.column_stack([
            tab,
            percentile_rank(lgb),
            percentile_rank(peer_prediction[valid]),
            percentile_rank(eb_prediction[valid]),
        ])
        rows.extend(valid.tolist())
        predictions.append(meta)
    return np.asarray(rows, dtype=np.int64), np.vstack(predictions)


def fit_nonnegative_stacker(
    values: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, float, bool, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.mean()
    c_value = 0.1

    def objective(parameters: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        linear = parameters[0] + x @ parameters[1:]
        probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -30.0, 30.0)))
        loss = -np.average(
            y * np.log(np.clip(probability, 1e-8, 1.0))
            + (1.0 - y) * np.log(np.clip(1.0 - probability, 1e-8, 1.0)),
            weights=w,
        )
        penalty = np.sum(parameters[1:] ** 2) / (2.0 * c_value * len(y))
        return float(loss + penalty)

    initial = np.array([clipped_logit(float(np.average(target, weights=weights))), 1, 1, 1, 1], dtype=np.float64)
    bounds = [(None, None)] + [(0.0, None)] * 4
    fitted = minimize(
        objective,
        initial,
        args=(values, target, weights),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500},
    )
    coefficients = fitted.x[1:]
    intercept = float(fitted.x[0])
    rng = np.random.default_rng(SEED)
    bootstrap = []
    for _ in range(5):
        indices = rng.choice(len(target), size=len(target), replace=True)
        result = minimize(
            objective,
            fitted.x,
            args=(values[indices], target[indices], weights[indices]),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 250},
        )
        bootstrap.append(result.x[1:])
    bootstrap_values = np.vstack(bootstrap)
    spread = bootstrap_values.std(axis=0) / np.maximum(0.05, bootstrap_values.mean(axis=0))
    stable = bool(
        fitted.success
        and np.all(np.isfinite(coefficients))
        and coefficients.sum() > 0.1
        and coefficients.max() / coefficients.sum() < 0.85
        and np.max(spread) < 1.0
    )
    return coefficients, intercept, stable, spread


def blend_predictions(
    meta: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    stable: bool,
) -> np.ndarray:
    if stable:
        linear = intercept + meta @ coefficients
        return 1.0 / (1.0 + np.exp(-np.clip(linear, -30.0, 30.0)))
    return meta @ np.array([0.60, 0.20, 0.10, 0.10], dtype=np.float64)


def print_oof_diagnostics(
    pool: pd.DataFrame,
    rows: np.ndarray,
    meta: np.ndarray,
    fixed: np.ndarray,
    base: np.ndarray,
    names: list[str],
) -> None:
    y = pool["target"].to_numpy(dtype=np.int8)[rows]
    if len(np.unique(y)) == 2:
        print(
            f"[lane3] forward_oof count={len(rows)} "
            f"tab_auc={roc_auc_score(y, meta[:, 0]):.6f} "
            f"lgb_auc={roc_auc_score(y, meta[:, 1]):.6f} "
            f"peer_auc={roc_auc_score(y, meta[:, 2]):.6f} "
            f"eb_auc={roc_auc_score(y, meta[:, 3]):.6f} "
            f"blend_auc={roc_auc_score(y, fixed):.6f}",
            flush=True,
        )
    time_values = pool["timestamp"].astype("int64").to_numpy()[rows]
    time_bins = np.digitize(time_values, np.quantile(time_values, [0.33, 0.66]))
    user_count_index = names.index("eb_user_count")
    degree_index = names.index("degree")
    history_bins = np.digitize(base[rows, user_count_index], [1, 2, 5])
    degree_bins = np.digitize(base[rows, degree_index], [1, 5, 20])
    for axis, strata in [
        ("time", time_bins),
        ("prior_history", history_bins),
        ("social_degree", degree_bins),
    ]:
        pieces = []
        for stratum in np.unique(strata):
            mask = strata == stratum
            score = (
                roc_auc_score(y[mask], fixed[mask])
                if mask.sum() > 1 and len(np.unique(y[mask])) == 2 else float("nan")
            )
            pieces.append(f"{int(stratum)}:{int(mask.sum())}:{score:.6f}")
        print(f"[lane3] oof_strata axis={axis} bin:count:auc={'|'.join(pieces)}", flush=True)


# Section: pipeline assembly

def query_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["timestamp", "user"]].copy().reset_index(drop=True)


def assemble_label_features(
    pool: pd.DataFrame,
    query: pd.DataFrame,
    pool_base: np.ndarray,
    query_base: np.ndarray,
    base_names: list[str],
    histories: dict,
    k_values: list[int],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    pool_vector, query_vector = retrieval_vectors(pool_base, query_base, base_names)
    retrieval, retrieval_names = exact_retrieval_features(
        pool,
        query,
        pool_vector,
        query_vector,
        histories["embedding"],
        k_values,
    )
    eb, eb_names = empirical_bayes_features(
        pool,
        query,
        pool_base,
        query_base,
        base_names,
        histories["adjacency"],
    )
    peer_name = f"retrieval_rate_{max(k_values)}"
    peer = retrieval[:, retrieval_names.index(peer_name)]
    recurrence = eb[:, eb_names.index("eb_recurrence_blend")]
    return np.column_stack([query_base, retrieval, eb]), base_names + retrieval_names + eb_names, peer, recurrence


def fit_scaler_and_apply(
    pool_features: np.ndarray,
    *query_features: np.ndarray,
) -> tuple[np.ndarray, ...]:
    mean, scale = standardize_fit(pool_features)
    return tuple(
        standardize_apply(values, mean, scale)
        for values in (pool_features,) + query_features
    )


def run_solution(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    debug: bool,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    start = time.time()
    paths = database_paths()
    pool_a, pool_b, replay = make_model_pools(train, val, paths["event_attendees"])
    elapsed_line(start, "exact_replay")
    use_tabpfn, tabpfn_status = probe_tabpfn(cache_dir)
    print(
        f"[lane3] pretrained_model={'TabPFN' if use_tabpfn else 'CatBoost_fallback'} "
        f"status={tabpfn_status}",
        flush=True,
    )
    sources = load_sources(paths)
    elapsed_line(start, "source_loading")
    n_users = len(sources["users"])
    adjacency, graph = build_adjacency(sources["friends"], n_users)
    embedding, embedding_method = social_embedding(graph, cache_dir, debug)
    histories = prepare_histories(sources, adjacency, embedding)
    elapsed_line(start, "social_and_histories")
    val_query = query_frame(val)
    test_query = query_frame(test)
    state_union = pd.concat(
        [
            query_frame(pool_b),
            val_query,
            test_query,
        ],
        ignore_index=True,
    ).drop_duplicates(["timestamp", "user"]).sort_values(
        ["timestamp", "user"]
    ).reset_index(drop=True)
    union_base, base_names = build_behavior_features(state_union, histories)
    key_to_index = {
        (pd.Timestamp(timestamp).value, int(user)): index
        for index, (timestamp, user) in enumerate(
            zip(state_union["timestamp"], state_union["user"])
        )
    }

    def select_base(frame: pd.DataFrame) -> np.ndarray:
        indices = [
            key_to_index[(pd.Timestamp(timestamp).value, int(user))]
            for timestamp, user in zip(frame["timestamp"], frame["user"])
        ]
        return union_base[np.asarray(indices, dtype=np.int64)]

    base_a = select_base(pool_a)
    base_b = select_base(pool_b)
    base_val = select_base(val_query)
    base_test = select_base(test_query)
    np.savez_compressed(
        cache_dir / ("behavior_vectors_debug.npz" if debug else "behavior_vectors_full.npz"),
        timestamp=state_union["timestamp"].astype("int64").to_numpy(),
        user=state_union["user"].to_numpy(dtype=np.int64),
        values=union_base,
        names=np.asarray(base_names),
    )
    elapsed_line(start, "behavior_features")
    k_values = [32] if debug else [32, 64, 128]
    feature_a_pool, names_a, peer_a_pool, eb_a_pool = assemble_label_features(
        pool_a,
        pool_a,
        base_a,
        base_a,
        base_names,
        histories,
        k_values,
    )
    feature_a_val, names_a_val, peer_a_val, eb_a_val = assemble_label_features(
        pool_a,
        val_query,
        base_a,
        base_val,
        base_names,
        histories,
        k_values,
    )
    feature_b_pool, names_b, peer_b_pool, eb_b_pool = assemble_label_features(
        pool_b,
        pool_b,
        base_b,
        base_b,
        base_names,
        histories,
        k_values,
    )
    feature_b_test, names_b_test, peer_b_test, eb_b_test = assemble_label_features(
        pool_b,
        test_query,
        base_b,
        base_test,
        base_names,
        histories,
        k_values,
    )
    if names_a != names_a_val or names_b != names_b_test or names_a != names_b:
        raise RuntimeError("feature names differ across legal pools")
    if len(names_a) > MAX_FEATURES:
        raise RuntimeError(f"feature cap exceeded: {len(names_a)} > {MAX_FEATURES}")
    np.savez_compressed(
        cache_dir / ("retrieval_summaries_debug.npz" if debug else "retrieval_summaries_full.npz"),
        a_pool=feature_a_pool[:, len(base_names):],
        a_val=feature_a_val[:, len(base_names):],
        b_pool=feature_b_pool[:, len(base_names):],
        b_test=feature_b_test[:, len(base_names):],
        names=np.asarray(names_a[len(base_names):]),
    )
    scaled_a_pool, scaled_a_val = fit_scaler_and_apply(feature_a_pool, feature_a_val)
    scaled_b_pool, scaled_b_test = fit_scaler_and_apply(feature_b_pool, feature_b_test)
    elapsed_line(start, "causal_retrieval_and_eb")
    oof_rows, oof_meta = forward_oof(
        pool_a,
        scaled_a_pool,
        peer_a_pool,
        eb_a_pool,
        use_tabpfn,
        debug,
    )
    stack_coefficients, stack_intercept, stack_stable, stack_spread = fit_nonnegative_stacker(
        oof_meta,
        pool_a["target"].to_numpy(dtype=np.int8)[oof_rows],
        pool_a["phase_weight"].to_numpy(dtype=np.float64)[oof_rows],
    )
    oof_blend = blend_predictions(
        oof_meta,
        stack_coefficients,
        stack_intercept,
        stack_stable,
    )
    print(
        f"[lane3] stacker coefficients={np.round(stack_coefficients, 5).tolist()} "
        f"spread={np.round(stack_spread, 5).tolist()} stable={stack_stable}",
        flush=True,
    )
    diagnostic_matrix = np.column_stack([
        feature_a_pool,
    ])
    print_oof_diagnostics(
        pool_a,
        oof_rows,
        oof_meta,
        oof_blend,
        diagnostic_matrix,
        names_a,
    )
    elapsed_line(start, "forward_oof_stacker")
    tab_a, context_ids_a = context_ensemble(
        pool_a,
        scaled_a_pool,
        scaled_a_val,
        use_tabpfn,
        debug,
        cache_dir,
        "a_debug" if debug else "a_full",
    )
    lgb_a = lightgbm_predict(pool_a, scaled_a_pool, scaled_a_val, debug, SEED + 4001)
    meta_a = np.column_stack([
        tab_a,
        percentile_rank(lgb_a),
        percentile_rank(peer_a_val),
        percentile_rank(eb_a_val),
    ])
    val_prediction = blend_predictions(
        meta_a,
        stack_coefficients,
        stack_intercept,
        stack_stable,
    )
    elapsed_line(start, "model_a")
    tab_b, context_ids_b = context_ensemble(
        pool_b,
        scaled_b_pool,
        scaled_b_test,
        use_tabpfn,
        debug,
        cache_dir,
        "b_debug" if debug else "b_full",
    )
    lgb_b = lightgbm_predict(pool_b, scaled_b_pool, scaled_b_test, debug, SEED + 5001)
    meta_b = np.column_stack([
        tab_b,
        percentile_rank(lgb_b),
        percentile_rank(peer_b_test),
        percentile_rank(eb_b_test),
    ])
    test_prediction = blend_predictions(
        meta_b,
        stack_coefficients,
        stack_intercept,
        stack_stable,
    )
    elapsed_line(start, "model_b")
    diagnostics = {
        "version": VERSION,
        "debug": debug,
        "feature_count": len(names_a),
        "pool_a_rows": len(pool_a),
        "pool_b_rows": len(pool_b),
        "replay_rows": len(replay),
        "official_replay_exact": True,
        "embedding_method": embedding_method,
        "tabpfn_available": use_tabpfn,
        "tabpfn_status": tabpfn_status,
        "context_count": 1 if debug else 6,
        "context_rows": 2000 if debug else 8192,
        "context_a_sizes": [len(x) for x in context_ids_a],
        "context_b_sizes": [len(x) for x in context_ids_b],
        "stack_coefficients": stack_coefficients.tolist(),
        "stack_intercept": stack_intercept,
        "stack_stable": stack_stable,
        "stack_spread": stack_spread.tolist(),
        "oof_rows": len(oof_rows),
        "oof_auc": float(
            roc_auc_score(
                pool_a["target"].to_numpy(dtype=np.int8)[oof_rows],
                oof_blend,
            )
        ),
        "elapsed_seconds": time.time() - start,
        "model_versions": {
            "catboost": __import__("catboost").__version__,
            "lightgbm": __import__("lightgbm").__version__,
            "tabpfn": __import__("tabpfn").__version__,
        },
    }
    (cache_dir / ("model_versions_debug.json" if debug else "model_versions_full.json")).write_text(
        json.dumps(diagnostics, indent=2)
    )
    return (
        np.clip(np.asarray(val_prediction, dtype=np.float64), 1e-6, 1.0 - 1e-6),
        np.clip(np.asarray(test_prediction, dtype=np.float64), 1e-6, 1.0 - 1e-6),
        diagnostics,
    )
