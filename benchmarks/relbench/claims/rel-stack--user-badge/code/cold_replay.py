from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


# Configuration

DAY = 86400
WINDOW = 91 * DAY
PURGE = 92 * DAY
NAME_BUCKETS = 32
EB_STRENGTHS = (50.0, 250.0, 1000.0)
FINGERPRINTS = {
    "train": "f341b1e9e6a1afa47684219db4b908db570c07819bc9d587a1bfcd207cf1929d",
    "val": "5a85d725ff3a086b7fefb49cea81bde15e57d19ef786716cb331ddf7bebfea0c",
    "test": "20ddbe3c6c208aa59e7dbf4a9ef166a9987bc865d155c53145e774429528bbf2",
    "run0006_val": "7a5ac6eda859f4d4fbd456d0265b15a5c5e64b3ea0c5c85b3922db90f08b494c",
    "run0006_test": "0fd18a83974b397d409615a8b636966f7fc34bdf70e3d1410e5db3434d5862d3",
}


# Data structures

@dataclass
class TargetRows:
    origin: int
    users: np.ndarray
    state: np.ndarray
    features: np.ndarray


@dataclass
class ReplayPanel:
    features: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    users: np.ndarray
    origin: np.ndarray
    close: np.ndarray
    state: np.ndarray
    targets: dict[int, TargetRows]
    feature_names: list[str]
    replay_origins: np.ndarray


# Fingerprints

def cache_root() -> Path:
    root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / "lane0_cold_replay_v1"
    root.mkdir(parents=True, exist_ok=True)
    return root


def archive_root() -> Path:
    return Path(os.environ["RELBENCH_WORK_DIR"]) / "runs" / "run_0006"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def task_key_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(frame["timestamp"].to_numpy(dtype="datetime64[ns]").astype("<i8", copy=False).tobytes())
    digest.update(frame["UserId"].to_numpy(dtype="<i8", copy=False).tobytes())
    return digest.hexdigest()


def verify_fingerprints(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> dict:
    measured = {
        "train": task_key_hash(train),
        "val": task_key_hash(validation),
        "test": task_key_hash(test),
        "run0006_val": sha256_file(archive_root() / "val_predictions.npy"),
        "run0006_test": sha256_file(archive_root() / "test_predictions.npy"),
    }
    for name, expected in FINGERPRINTS.items():
        if measured[name] != expected:
            raise RuntimeError(f"fingerprint mismatch for {name}: {measured[name]} != {expected}")
    inventory = {
        "task_and_archive": measured,
        "champion_oof": "dc495fddcc534733500d11bb8879b19873b7d38e7a1feb1058dc0145d07db5dd",
        "champion_oof_indices": "c926252a93929fff0d1ef701ccd7c8ef67932cd8ce8d66974d1c7a4ce8d7d3b9",
        "champion_feature_matrix": "5804032a08669943a3d263021ae9999b5885f700f17ed8fb21035fd640c76c3c",
        "run0006_fold_matrix_0": "4388d5f98039adf0c114073456901fc852a1190053d55c0dbd5f3eac485a471a",
    }
    (cache_root() / "fingerprints.json").write_text(json.dumps(inventory, indent=2))
    return inventory


# Empirical Bayes

class EBTracker:
    def __init__(self, n_users: int, sizes: list[int]) -> None:
        self.counts = [np.zeros(size, dtype=np.int64) for size in sizes]
        self.positives = [np.zeros(size, dtype=np.float64) for size in sizes]
        self.last = [np.full(n_users, -1, dtype=np.int16) for _ in sizes]
        self.user_counts = [np.zeros(n_users, dtype=np.uint8) for _ in sizes]
        self.user_positives = [np.zeros(n_users, dtype=np.uint8) for _ in sizes]
        self.total = 0
        self.total_positive = 0.0

    def features(self, users: np.ndarray, keys: list[np.ndarray]) -> np.ndarray:
        prior = (self.total_positive + 1.0) / (self.total + 20.0)
        columns = []
        for group, key in enumerate(keys):
            same = self.last[group][users] == key
            own_count = np.where(same, self.user_counts[group][users], 0)
            own_positive = np.where(same, self.user_positives[group][users], 0)
            count = self.counts[group][key] - own_count
            positive = self.positives[group][key] - own_positive
            for strength in EB_STRENGTHS:
                columns.append(((positive + strength * prior) / (count + strength)).astype(np.float32))
        return np.column_stack(columns)

    def add(self, users: np.ndarray, keys: list[np.ndarray], target: np.ndarray) -> None:
        self.total += len(users)
        self.total_positive += float(target.sum())
        for group, key in enumerate(keys):
            np.add.at(self.counts[group], key, 1)
            np.add.at(self.positives[group], key, target)
            changed = self.last[group][users] != key
            changed_users = users[changed]
            self.last[group][changed_users] = key[changed]
            self.user_counts[group][changed_users] = 0
            self.user_positives[group][changed_users] = 0
            self.user_counts[group][users] += 1
            self.user_positives[group][users] += target.astype(np.uint8)


# Replay construction

def stable_name_bucket(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(), "little") % NAME_BUCKETS


def mapped_users(values: pd.Series, user_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = values.fillna(-1).to_numpy(dtype=np.int64)
    mapped = np.searchsorted(user_ids, raw)
    valid = (mapped >= 0) & (mapped < len(user_ids))
    valid &= user_ids[np.minimum(mapped, len(user_ids) - 1)] == raw
    return mapped.astype(np.int32), valid


def first_authored_time(db, user_ids: np.ndarray) -> np.ndarray:
    first = np.full(len(user_ids), np.iinfo(np.int64).max, dtype=np.int64)
    routes = (("posts", "OwnerUserId"), ("comments", "UserId"), ("postHistory", "UserId"))
    for table, column in routes:
        frame = db.table_dict[table].df
        mapped, valid = mapped_users(frame[column], user_ids)
        times = frame["CreationDate"].to_numpy(dtype="datetime64[s]").astype(np.int64)
        np.minimum.at(first, mapped[valid], times[valid])
    return first


def monthly_origins(debug: bool) -> np.ndarray:
    periods = 12 if debug else 84
    dates = pd.date_range(end=pd.Timestamp("2020-10-01"), periods=periods, freq="MS")
    return dates.to_numpy(dtype="datetime64[s]").astype(np.int64)


def group_keys(
    users: np.ndarray,
    creation: np.ndarray,
    origin: int,
    id_percentile: np.ndarray,
    account_percentile: np.ndarray,
    rank_gap: np.ndarray,
    total_badges: np.ndarray,
    class_counts: np.ndarray,
) -> list[np.ndarray]:
    dates = pd.to_datetime(creation[users], unit="s")
    cohort = np.clip((dates.year.to_numpy() - 2008) * 12 + dates.month.to_numpy() - 1, 0, 255).astype(np.int16)
    age_year = np.clip(((origin - creation[users]) / (365.2425 * DAY)).astype(np.int16), 0, 31)
    id_bin = np.minimum((id_percentile * 200).astype(np.int16), 199)
    account_bin = np.minimum((account_percentile * 200).astype(np.int16), 199)
    gap_bin = np.minimum((np.abs(rank_gap) * 100).astype(np.int16), 99)
    signature = np.minimum(total_badges[users], 7).astype(np.int16)
    signature += 8 * (class_counts[users, 0] > 0).astype(np.int16)
    signature += 16 * (class_counts[users, 1] > 0).astype(np.int16)
    signature += 32 * (class_counts[users, 2] > 0).astype(np.int16)
    return [cohort, age_year, id_bin, account_bin, gap_bin, signature]


def system_incidence(times: np.ndarray, classes: np.ndarray, tag_based: np.ndarray, origin: int) -> np.ndarray:
    values = []
    masks = [np.ones(len(times), dtype=bool)]
    masks += [classes == value for value in (1, 2, 3)]
    masks += [tag_based, ~tag_based]
    for mask in masks:
        selected = times[mask]
        for days in (7, 30, 91, 365):
            hi = np.searchsorted(selected, origin, side="right")
            lo = np.searchsorted(selected, origin - days * DAY, side="right")
            values.append(math.log1p(hi - lo))
        hi = np.searchsorted(selected, origin, side="right")
        lo = np.searchsorted(selected, origin - 91 * DAY, side="right")
        prior_hi = np.searchsorted(selected, origin - 365 * DAY, side="right")
        prior_lo = np.searchsorted(selected, origin - 456 * DAY, side="right")
        values.append(math.log((1.0 + hi - lo) / (1.0 + prior_hi - prior_lo)))
    return np.asarray(values, dtype=np.float32)


def normalized_block(raw: np.ndarray, count: int) -> np.ndarray:
    selected = raw[:, :count].astype(np.float64)
    order = np.argsort(selected, axis=0, kind="mergesort")
    ranks = np.empty_like(selected)
    rows = np.arange(len(selected), dtype=np.float64)[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(rows, order.shape), axis=0)
    percentiles = ranks / max(len(selected) - 1, 1)
    mean = selected.mean(axis=0)
    standard = np.maximum(selected.std(axis=0), 1e-6)
    zscore = (selected - mean) / standard
    leader_gap = selected.max(axis=0) - selected
    return np.column_stack((raw, percentiles, zscore, leader_gap)).astype(np.float32)


def build_panel(ctx, debug: bool) -> ReplayPanel:
    started = time.time()
    users_frame = ctx.db.table_dict["users"].df.sort_values("Id").reset_index(drop=True)
    user_ids = users_frame["Id"].to_numpy(dtype=np.int64)
    creation = users_frame["CreationDate"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    account = users_frame["AccountId"].fillna(np.inf).to_numpy(dtype=np.float64)
    first_authored = first_authored_time(ctx.db, user_ids)
    badges = ctx.db.table_dict["badges"].df.sort_values(["Date", "Id"]).reset_index(drop=True)
    badge_users, badge_valid = mapped_users(badges["UserId"], user_ids)
    badges = badges.loc[badge_valid].reset_index(drop=True)
    badge_users = badge_users[badge_valid]
    badge_times = badges["Date"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    badge_classes = badges["Class"].to_numpy(dtype=np.int8)
    badge_tags = badges["TagBased"].to_numpy(dtype=bool)
    names, badge_name_ids = np.unique(badges["Name"].astype(str).to_numpy(), return_inverse=True)
    name_buckets = np.asarray([stable_name_bucket(name) for name in names], dtype=np.int16)
    yearling_ids = set(np.flatnonzero(names == "Yearling").tolist())
    replay = monthly_origins(debug)
    fold_origins = np.unique(ctx.train.df["timestamp"].to_numpy(dtype="datetime64[s]").astype(np.int64))[-6:]
    validation_origin = int(ctx.val.df["timestamp"].iloc[0].to_datetime64().astype("datetime64[s]").astype(np.int64))
    test_origin = int(ctx.test.df["timestamp"].iloc[0].to_datetime64().astype("datetime64[s]").astype(np.int64))
    target_origins = np.unique(np.concatenate((fold_origins, [validation_origin, test_origin])))
    all_origins = np.unique(np.concatenate((replay, target_origins)))
    n_users = len(user_ids)
    total_badges = np.zeros(n_users, dtype=np.int16)
    class_counts = np.zeros((n_users, 3), dtype=np.int16)
    tag_counts = np.zeros((n_users, 2), dtype=np.int16)
    name_tokens = np.zeros((n_users, NAME_BUCKETS), dtype=np.int16)
    distinct_names = np.zeros(n_users, dtype=np.int16)
    entropy_sum = np.zeros(n_users, dtype=np.float32)
    repeated_names = np.zeros(n_users, dtype=np.int16)
    max_repeat = np.zeros(n_users, dtype=np.int16)
    first_badge = np.full(n_users, -1, dtype=np.int64)
    last_badge = np.full(n_users, -1, dtype=np.int64)
    last_day = np.full(n_users, -1, dtype=np.int64)
    day_cluster = np.zeros(n_users, dtype=np.int16)
    same_day_clusters = np.zeros(n_users, dtype=np.int16)
    max_day_cluster = np.zeros(n_users, dtype=np.int16)
    yearling_count = np.zeros(n_users, dtype=np.int16)
    pair_counts: dict[int, int] = {}
    pointer = 0
    eb = EBTracker(n_users, [256, 32, 200, 200, 100, 64])
    pending: list[tuple[int, np.ndarray, list[np.ndarray], np.ndarray]] = []
    replay_x = []
    replay_y = []
    replay_w = []
    replay_u = []
    replay_o = []
    replay_s = []
    replay_episode = []
    targets: dict[int, TargetRows] = {}
    feature_names: list[str] | None = None
    rng = np.random.default_rng(1337)
    for origin in all_origins:
        while pending and pending[0][0] <= origin:
            _, old_users, old_keys, old_target = pending.pop(0)
            eb.add(old_users, old_keys, old_target)
        current = np.searchsorted(badge_times, origin, side="right")
        for index in range(pointer, current):
            user = int(badge_users[index])
            name_id = int(badge_name_ids[index])
            pair = user * len(names) + name_id
            old = pair_counts.get(pair, 0)
            new = old + 1
            pair_counts[pair] = new
            if old == 0:
                distinct_names[user] += 1
            if old == 1:
                repeated_names[user] += 1
            entropy_sum[user] += new * math.log(new) - (old * math.log(old) if old else 0.0)
            max_repeat[user] = max(max_repeat[user], new)
            total_badges[user] += 1
            badge_class = int(np.clip(badge_classes[index], 1, 3)) - 1
            class_counts[user, badge_class] += 1
            tag_counts[user, int(badge_tags[index])] += 1
            name_tokens[user, name_buckets[name_id]] += 1
            if name_id in yearling_ids:
                yearling_count[user] += 1
            event_time = int(badge_times[index])
            if first_badge[user] < 0:
                first_badge[user] = event_time
            event_day = event_time // DAY
            if last_day[user] == event_day:
                day_cluster[user] += 1
                if day_cluster[user] == 2:
                    same_day_clusters[user] += 1
            else:
                day_cluster[user] = 1
                last_day[user] = event_day
            max_day_cluster[user] = max(max_day_cluster[user], day_cluster[user])
            last_badge[user] = event_time
        pointer = current
        eligible = np.flatnonzero((creation <= origin) & (first_authored > origin))
        state = (total_badges[eligible] > 0).astype(np.uint8)
        id_rank = rankdata(user_ids[eligible], method="average") / len(eligible)
        account_missing = np.isinf(account[eligible])
        account_rank = rankdata(account[eligible], method="average") / len(eligible)
        rank_gap = id_rank - account_rank
        keys = group_keys(eligible, creation, int(origin), id_rank, account_rank, rank_gap, total_badges, class_counts)
        future_hi = np.searchsorted(badge_times, origin + WINDOW, side="right")
        future_users = badge_users[current:future_hi]
        target = np.zeros(len(eligible), dtype=np.uint8)
        episode = np.full(n_users, np.iinfo(np.int64).max, dtype=np.int64)
        if len(future_users):
            future_global = np.arange(current, future_hi, dtype=np.int64)
            np.minimum.at(episode, future_users, future_global)
            target = (episode[eligible] < np.iinfo(np.int64).max).astype(np.uint8)
        age_days = (origin - creation[eligible]) / DAY
        age_year = np.floor(age_days / 365.2425)
        phase = age_days - age_year * 365.2425
        next_anniversary = 365.2425 - phase
        registration = pd.to_datetime(creation[eligible], unit="s")
        entropy = np.zeros(len(eligible), dtype=np.float32)
        positive_badges = total_badges[eligible] > 0
        entropy[positive_badges] = np.log(total_badges[eligible][positive_badges]) - entropy_sum[eligible][positive_badges] / total_badges[eligible][positive_badges]
        first_recency = np.where(first_badge[eligible] >= 0, (origin - first_badge[eligible]) / DAY, 10000.0)
        last_recency = np.where(last_badge[eligible] >= 0, (origin - last_badge[eligible]) / DAY, 10000.0)
        cadence = np.where(total_badges[eligible] > 1, (last_badge[eligible] - first_badge[eligible]) / DAY / np.maximum(total_badges[eligible] - 1, 1), 10000.0)
        debt = np.maximum(age_year.astype(np.float32) - yearling_count[eligible], 0)
        raw_columns = [
            np.log1p(age_days), phase / 365.2425, next_anniversary / 365.2425,
            registration.month.to_numpy() / 12.0, (registration.year.to_numpy() - 2008) / 16.0,
            age_year, age_year * phase / 365.2425, id_rank, account_rank,
            account_missing.astype(np.float32), rank_gap, np.abs(rank_gap),
            np.log1p(total_badges[eligible]), np.log1p(class_counts[eligible, 0]),
            np.log1p(class_counts[eligible, 1]), np.log1p(class_counts[eligible, 2]),
            np.log1p(tag_counts[eligible, 0]), np.log1p(tag_counts[eligible, 1]),
            np.log1p(distinct_names[eligible]), entropy, np.log1p(first_recency),
            np.log1p(last_recency), np.log1p(cadence), np.log1p(same_day_clusters[eligible]),
            np.log1p(max_day_cluster[eligible]), np.log1p(max_repeat[eligible]),
            np.log1p(repeated_names[eligible]), np.log1p(yearling_count[eligible]),
            np.log1p(debt), (next_anniversary <= 91).astype(np.float32), state.astype(np.float32),
        ]
        raw = np.column_stack(raw_columns).astype(np.float32)
        token_block = np.log1p(name_tokens[eligible]).astype(np.float32)
        system = system_incidence(badge_times, badge_classes, badge_tags, int(origin))
        system_block = np.broadcast_to(system, (len(eligible), len(system))).copy()
        eb_block = eb.features(eligible, keys)
        raw = np.column_stack((raw, token_block, system_block, eb_block)).astype(np.float32)
        features = normalized_block(raw, 31)
        if feature_names is None:
            base_names = [f"raw_{index}" for index in range(raw.shape[1])]
            feature_names = base_names + [f"percentile_{index}" for index in range(31)] + [f"z_{index}" for index in range(31)] + [f"leader_gap_{index}" for index in range(31)]
        if int(origin) in set(target_origins.tolist()):
            targets[int(origin)] = TargetRows(int(origin), eligible.astype(np.int32), state, features)
        if int(origin) in set(replay.tolist()):
            age_decile = np.minimum((rankdata(age_days, method="average") / len(age_days) * 10).astype(np.int16), 9)
            depth = np.minimum(total_badges[eligible], 4).astype(np.int16)
            stratum = state.astype(np.int16) * 100 + age_decile * 10 + depth
            selected = []
            inverse = []
            for code in np.unique(stratum[target == 1]):
                positive = np.flatnonzero((stratum == code) & (target == 1))
                negative = np.flatnonzero((stratum == code) & (target == 0))
                take = min(len(negative), 8 * len(positive))
                chosen = rng.choice(negative, size=take, replace=False) if take else np.empty(0, dtype=np.int64)
                selected.extend(positive.tolist())
                inverse.extend(np.ones(len(positive), dtype=np.float32).tolist())
                selected.extend(chosen.tolist())
                inverse.extend(np.full(take, len(negative) / max(take, 1), dtype=np.float32).tolist())
            selected_array = np.asarray(selected, dtype=np.int64)
            replay_x.append(features[selected_array])
            replay_y.append(target[selected_array])
            replay_w.append(np.asarray(inverse, dtype=np.float32))
            replay_u.append(eligible[selected_array].astype(np.int32))
            replay_o.append(np.full(len(selected_array), origin, dtype=np.int64))
            replay_s.append(state[selected_array])
            replay_episode.append(episode[eligible[selected_array]])
            pending.append((int(origin + WINDOW), eligible.astype(np.int32), keys, target))
    features = np.concatenate(replay_x)
    target = np.concatenate(replay_y)
    weight = np.concatenate(replay_w)
    panel_users = np.concatenate(replay_u)
    panel_origin = np.concatenate(replay_o)
    panel_state = np.concatenate(replay_s)
    episodes = np.concatenate(replay_episode)
    positive = target == 1
    unique_episode, episode_count = np.unique(episodes[positive], return_counts=True)
    position = np.searchsorted(unique_episode, episodes[positive])
    weight[positive] /= episode_count[position]
    panel = ReplayPanel(features, target, weight, panel_users, panel_origin, panel_origin + WINDOW, panel_state, targets, feature_names or [], replay)
    print(f"[cold-replay] origins={len(replay)} rows={len(target)} positives={int(target.sum())} features={features.shape[1]} seconds={time.time() - started:.1f}", flush=True)
    return panel


# Models

def model_parameters(trees: int) -> dict:
    return {
        "objective": "binary", "n_estimators": trees, "learning_rate": 0.04,
        "num_leaves": 31, "min_child_samples": 1000, "colsample_bytree": 0.85,
        "subsample": 0.9, "subsample_freq": 1, "reg_alpha": 0.15, "reg_lambda": 3.0,
        "n_jobs": int(os.environ.get("OMP_NUM_THREADS", "11")), "verbosity": -1, "random_state": 1337,
    }


def density_weights(train_x: np.ndarray, target_x: np.ndarray, trees: int) -> np.ndarray:
    rng = np.random.default_rng(1337)
    size = min(len(train_x), len(target_x), 120000)
    if size < 2000:
        return np.ones(len(train_x), dtype=np.float32)
    left = rng.choice(len(train_x), size=size, replace=False)
    right = rng.choice(len(target_x), size=size, replace=False)
    x = np.concatenate((train_x[left], target_x[right]))
    y = np.concatenate((np.zeros(size, dtype=np.uint8), np.ones(size, dtype=np.uint8)))
    classifier = lgb.LGBMClassifier(**{**model_parameters(min(trees, 80)), "min_child_samples": 500})
    classifier.fit(x, y, callbacks=[lgb.log_evaluation(0)])
    probability = np.clip(classifier.predict_proba(train_x)[:, 1], 1e-4, 1 - 1e-4)
    return np.clip(probability / (1.0 - probability), 0.25, 4.0).astype(np.float32)


def fit_experts(panel: ReplayPanel, target_rows: TargetRows, boundary: int, debug: bool) -> tuple[np.ndarray, dict]:
    cache_tag = f"expert_{boundary}_{'debug' if debug else 'full'}"
    prediction_path = cache_root() / f"{cache_tag}.npy"
    report_path = cache_root() / f"{cache_tag}.json"
    if prediction_path.exists() and report_path.exists():
        prediction = np.load(prediction_path)
        if prediction.shape == (len(target_rows.users),):
            return prediction, json.loads(report_path.read_text())
    admitted = panel.close <= boundary
    output = np.full(len(target_rows.users), 0.5, dtype=np.float32)
    report = {}
    trees = 50 if debug else 350
    density = density_weights(panel.features[admitted], target_rows.features, trees)
    admitted_indices = np.flatnonzero(admitted)
    for state in (0, 1):
        train_indices = admitted_indices[panel.state[admitted_indices] == state]
        target_indices = np.flatnonzero(target_rows.state == state)
        if len(train_indices) < 2000 or panel.target[train_indices].sum() < 20 or not len(target_indices):
            prior = (panel.target[train_indices].sum() + 1) / (len(train_indices) + 20) if len(train_indices) else 0.01
            output[target_indices] = prior
            report[f"N{state}"] = {"rows": int(len(train_indices)), "positives": int(panel.target[train_indices].sum()), "constant": True}
            continue
        local = np.searchsorted(admitted_indices, train_indices)
        sample_weight = panel.weight[train_indices] * density[local]
        model = lgb.LGBMClassifier(**model_parameters(trees))
        model.fit(panel.features[train_indices], panel.target[train_indices], sample_weight=sample_weight, callbacks=[lgb.log_evaluation(0)])
        output[target_indices] = model.predict_proba(target_rows.features[target_indices])[:, 1]
        report[f"N{state}"] = {"rows": int(len(train_indices)), "positives": int(panel.target[train_indices].sum()), "constant": False}
    output = np.clip(output, 1e-6, 1 - 1e-6)
    np.save(prediction_path, output)
    report_path.write_text(json.dumps(report, indent=2))
    return output, report


def predict_chunks(model: lgb.Booster, matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    output = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), 50000):
        stop = min(start + 50000, len(indices))
        output[start:stop] = model.predict(np.asarray(matrix[indices[start:stop]], dtype=np.float32))
    return output


def champion_fold_predictions(train: pd.DataFrame, fold_origins: np.ndarray, debug: bool) -> dict[int, np.ndarray]:
    root = cache_root() / ("champion_folds_debug" if debug else "champion_folds_full")
    root.mkdir(exist_ok=True)
    times = train["timestamp"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    target = train["WillGetBadge"].to_numpy(dtype=np.uint8)
    cache = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    state = cache / "lane3_dormant_content_v1"
    graph = cache / "lane3_chrono_graph_v1"
    late = {
        int(fold_origins[-3]): (state / "combined_oof_both_1577923200_full_7811ea744f98.npy", graph / "gbdt_both_fold0_full_topic_v1_730.txt"),
        int(fold_origins[-2]): (state / "combined_oof_both_1585785600_full_10505d3f95ef.npy", graph / "gbdt_both_fold1_full_topic_v1_730.txt"),
        int(fold_origins[-1]): (state / "combined_oof_both_1593648000_full_9e07ebad09e0.npy", graph / "gbdt_both_fold2_full_topic_v1_730.txt"),
    }
    base = None
    predictions = {}
    for fold_number, origin in enumerate(fold_origins):
        path = root / f"fold_{fold_number}.npy"
        indices = np.flatnonzero(times == origin)
        if path.exists():
            predictions[int(origin)] = np.load(path)
            continue
        if not debug and int(origin) in late:
            matrix_path, model_path = late[int(origin)]
            matrix = np.load(matrix_path, mmap_mode="r")
            model = lgb.Booster(model_file=str(model_path))
            prediction = predict_chunks(model, matrix, indices)
        else:
            if base is None:
                base = np.load(cache / "lane3_chrono_graph_v1/combined_full_v3.npy", mmap_mode="r")
            admissible_origins = np.unique(times[times + PURGE <= origin])[-16:]
            fit_indices = np.flatnonzero(np.isin(times, admissible_origins))
            if debug:
                fit_indices = fit_indices[-250000:]
            columns = np.arange(0, base.shape[1], max(base.shape[1] // 160, 1))[:160]
            model = lgb.LGBMClassifier(**model_parameters(50 if debug else 350))
            model.fit(np.asarray(base[np.ix_(fit_indices, columns)], dtype=np.float32), target[fit_indices], callbacks=[lgb.log_evaluation(0)])
            prediction = model.predict_proba(np.asarray(base[np.ix_(indices, columns)], dtype=np.float32))[:, 1]
        np.save(path, prediction.astype(np.float32))
        predictions[int(origin)] = prediction.astype(np.float32)
        print(f"[cold-stack] champion_fold={fold_number} origin={origin} auc={roc_auc_score(target[indices], prediction):.6f}", flush=True)
    return predictions


# Stack and gate

def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def stack_matrix(champion: np.ndarray, expert: np.ndarray, state: np.ndarray) -> np.ndarray:
    n0 = state == 0
    n1 = state == 1
    champion_logit = logit(champion)
    expert_logit = logit(expert)
    return np.column_stack((np.ones(len(state)), champion_logit, expert_logit * n0, expert_logit * n1, n0, n1, champion_logit * expert_logit * n0, champion_logit * expert_logit * n1))


def fit_stack(matrix: np.ndarray, target: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
    weight = np.ones(len(target), dtype=np.float64) if sample_weight is None else sample_weight.astype(np.float64)
    scale = weight.sum()
    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        score = matrix @ coefficient
        probability = expit(score)
        loss = np.sum(weight * (np.logaddexp(0, score) - target * score)) / scale
        penalty = np.sum(coefficient[1:4] ** 2) / (2 * 0.1 * scale)
        gradient = matrix.T @ (weight * (probability - target)) / scale
        gradient[1:4] += coefficient[1:4] / (0.1 * scale)
        return float(loss + penalty), gradient
    bounds = [(None, None), (0, None), (0, None), (0, None), (None, None), (None, None), (0, None), (0, None)]
    initial = np.asarray([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    return minimize(objective, initial, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": 300}).x


def paired_bootstrap(target: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, users: np.ndarray, origins: np.ndarray, repeats: int) -> dict:
    unique_users, inverse = np.unique(users, return_inverse=True)
    rng = np.random.default_rng(1337)
    deltas = []
    unique_origins = np.unique(origins)
    for _ in range(repeats):
        sampled = rng.integers(0, len(unique_users), len(unique_users))
        multiplicity = np.bincount(sampled, minlength=len(unique_users))
        row_weight = multiplicity[inverse]
        origin_deltas = []
        for origin in unique_origins:
            selected = (row_weight > 0) & (origins == origin)
            try:
                candidate_auc = roc_auc_score(target[selected], candidate[selected], sample_weight=row_weight[selected])
                baseline_auc = roc_auc_score(target[selected], baseline[selected], sample_weight=row_weight[selected])
                origin_deltas.append(candidate_auc - baseline_auc)
            except ValueError:
                pass
        if len(origin_deltas) == len(unique_origins):
            deltas.append(float(np.mean(origin_deltas)))
    values = np.asarray(deltas)
    return {"repeats": int(len(values)), "mean": float(values.mean()), "se": float(values.std(ddof=1)), "probability_positive": float(np.mean(values > 0))}


def run_gate(ctx, panel: ReplayPanel, debug: bool) -> tuple[bool, dict, dict[int, np.ndarray], dict[int, np.ndarray]]:
    train = ctx.train.df.reset_index(drop=True)
    times = train["timestamp"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    target = train["WillGetBadge"].to_numpy(dtype=np.uint8)
    user_ids = ctx.db.table_dict["users"].df.sort_values("Id")["Id"].to_numpy(dtype=np.int64)
    fold_origins = np.unique(times)[-6:]
    champion = champion_fold_predictions(train, fold_origins, debug)
    experts = {}
    fold_records = []
    stack_history_x = []
    stack_history_y = []
    evaluable_wins = 0
    overall_deltas = []
    never_deltas = []
    segment_deltas = {0: [], 1: []}
    expert_deltas = {0: [], 1: []}
    final_candidate_rows = []
    final_baseline_rows = []
    final_target_rows = []
    final_user_rows = []
    final_origin_rows = []
    for fold_number, origin in enumerate(fold_origins):
        rows = panel.targets[int(origin)]
        expert, expert_report = fit_experts(panel, rows, int(origin), debug)
        experts[int(origin)] = expert
        indices = np.flatnonzero(times == origin)
        mapped = np.searchsorted(user_ids, train.loc[indices, "UserId"].to_numpy(dtype=np.int64))
        cold_position = np.searchsorted(rows.users, mapped)
        valid = (cold_position < len(rows.users)) & (rows.users[np.minimum(cold_position, len(rows.users) - 1)] == mapped)
        baseline = champion[int(origin)]
        expert_full = np.full(len(indices), 0.5, dtype=np.float32)
        state_full = np.full(len(indices), 2, dtype=np.uint8)
        expert_full[valid] = expert[cold_position[valid]]
        state_full[valid] = rows.state[cold_position[valid]]
        x = stack_matrix(baseline, expert_full, state_full)
        y = target[indices]
        if fold_number < 2:
            stack_history_x.append(x)
            stack_history_y.append(y)
            candidate = baseline
            coefficient = np.asarray([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        else:
            coefficient = fit_stack(np.concatenate(stack_history_x), np.concatenate(stack_history_y))
            candidate = expit(x @ coefficient)
            stack_history_x.append(x)
            stack_history_y.append(y)
            evaluable_wins += int(roc_auc_score(y, candidate) > roc_auc_score(y, baseline))
        delta = roc_auc_score(y, candidate) - roc_auc_score(y, baseline)
        never = state_full < 2
        never_delta = roc_auc_score(y[never], candidate[never]) - roc_auc_score(y[never], baseline[never])
        overall_deltas.append(delta)
        never_deltas.append(never_delta)
        for state in (0, 1):
            mask = state_full == state
            if mask.sum() and len(np.unique(y[mask])) == 2:
                segment_deltas[state].append(roc_auc_score(y[mask], candidate[mask]) - roc_auc_score(y[mask], baseline[mask]))
                expert_deltas[state].append(roc_auc_score(y[mask], expert_full[mask]) - roc_auc_score(y[mask], baseline[mask]))
        if fold_number >= 2:
            final_candidate_rows.append(candidate)
            final_baseline_rows.append(baseline)
            final_target_rows.append(y)
            final_user_rows.append(train.loc[indices, "UserId"].to_numpy(dtype=np.int64))
            final_origin_rows.append(np.full(len(indices), origin, dtype=np.int64))
        fold_records.append({"fold": fold_number, "origin": int(origin), "baseline_auc": float(roc_auc_score(y, baseline)), "candidate_auc": float(roc_auc_score(y, candidate)), "delta": float(delta), "never_delta": float(never_delta), "expert_delta_n0": float(expert_deltas[0][-1]), "expert_delta_n1": float(expert_deltas[1][-1]), "coefficient": coefficient.tolist(), "experts": expert_report})
        print(f"[cold-gate] fold={fold_number} delta={delta:+.6f} never_delta={never_delta:+.6f}", flush=True)
    boot = paired_bootstrap(np.concatenate(final_target_rows), np.concatenate(final_candidate_rows), np.concatenate(final_baseline_rows), np.concatenate(final_user_rows), np.concatenate(final_origin_rows), 50 if debug else 500)
    segment_se = {}
    segment_ok = True
    for state in (0, 1):
        values = np.asarray(segment_deltas[state], dtype=np.float64)
        se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("inf")
        mean = float(values.mean()) if len(values) else -float("inf")
        segment_se[f"N{state}"] = {"mean_delta": mean, "paired_se": se}
        segment_ok &= mean >= -se
    expert_positive = all(np.mean(segment_deltas[state]) > 0 for state in (0, 1) if segment_deltas[state])
    accepted = bool(expert_positive and evaluable_wins >= 3 and np.mean(overall_deltas) > 0 and np.mean(never_deltas) > 0 and boot["probability_positive"] >= 0.8 and segment_ok)
    report = {"accepted": accepted, "folds": fold_records, "evaluable_wins": evaluable_wins, "mean_delta": float(np.mean(overall_deltas)), "never_mean_delta": float(np.mean(never_deltas)), "integrated_expert_mean_delta": {f"N{state}": float(np.mean(segment_deltas[state])) for state in (0, 1)}, "standalone_expert_mean_delta": {f"N{state}": float(np.mean(expert_deltas[state])) for state in (0, 1)}, "segment": segment_se, "bootstrap": boot}
    return accepted, report, experts, champion


def final_predictions(ctx, panel: ReplayPanel, accepted: bool, report: dict, fold_experts: dict[int, np.ndarray], champion: dict[int, np.ndarray], debug: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    baseline_validation = np.load(archive_root() / "val_predictions.npy").astype(np.float64)
    baseline_test = np.load(archive_root() / "test_predictions.npy").astype(np.float64)
    if not accepted:
        return baseline_validation, baseline_test, {"fallback": "hash_verified_run_0006", "gate": report}
    validation = ctx.val.df.reset_index(drop=True)
    test = ctx.test.df.reset_index(drop=True)
    validation_origin = int(validation["timestamp"].iloc[0].to_datetime64().astype("datetime64[s]").astype(np.int64))
    test_origin = int(test["timestamp"].iloc[0].to_datetime64().astype("datetime64[s]").astype(np.int64))
    validation_rows = panel.targets[validation_origin]
    test_rows = panel.targets[test_origin]
    validation_expert, validation_report = fit_experts(panel, validation_rows, validation_origin, debug)
    test_expert, test_report = fit_experts(panel, test_rows, test_origin, debug)
    user_ids = ctx.db.table_dict["users"].df.sort_values("Id")["Id"].to_numpy(dtype=np.int64)
    def expand(frame: pd.DataFrame, rows: TargetRows, expert: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mapped = np.searchsorted(user_ids, frame["UserId"].to_numpy(dtype=np.int64))
        position = np.searchsorted(rows.users, mapped)
        valid = (position < len(rows.users)) & (rows.users[np.minimum(position, len(rows.users) - 1)] == mapped)
        output = np.full(len(frame), 0.5, dtype=np.float32)
        state = np.full(len(frame), 2, dtype=np.uint8)
        output[valid] = expert[position[valid]]
        state[valid] = rows.state[position[valid]]
        return output, state
    validation_component, validation_state = expand(validation, validation_rows, validation_expert)
    test_component, test_state = expand(test, test_rows, test_expert)
    train = ctx.train.df.reset_index(drop=True)
    times = train["timestamp"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    target = train["WillGetBadge"].to_numpy(dtype=np.uint8)
    fold_origins = np.unique(times)[-6:]
    stack_x = []
    stack_y = []
    for origin in fold_origins:
        rows = panel.targets[int(origin)]
        component = fold_experts[int(origin)]
        indices = np.flatnonzero(times == origin)
        mapped = np.searchsorted(user_ids, train.loc[indices, "UserId"].to_numpy(dtype=np.int64))
        position = np.searchsorted(rows.users, mapped)
        valid = (position < len(rows.users)) & (rows.users[np.minimum(position, len(rows.users) - 1)] == mapped)
        full_component = np.full(len(indices), 0.5, dtype=np.float32)
        full_state = np.full(len(indices), 2, dtype=np.uint8)
        full_component[valid] = component[position[valid]]
        full_state[valid] = rows.state[position[valid]]
        stack_x.append(stack_matrix(champion[int(origin)], full_component, full_state))
        stack_y.append(target[indices])
    coefficient_a = fit_stack(np.concatenate(stack_x), np.concatenate(stack_y))
    validation_prediction = expit(stack_matrix(baseline_validation, validation_component, validation_state) @ coefficient_a)
    validation_target = validation["WillGetBadge"].to_numpy(dtype=np.uint8)
    coefficient_b = fit_stack(np.concatenate((*stack_x, stack_matrix(baseline_validation, validation_component, validation_state))), np.concatenate((*stack_y, validation_target)))
    test_prediction = expit(stack_matrix(baseline_test, test_component, test_state) @ coefficient_b)
    diagnostics = {"fallback": None, "gate": report, "coefficient_a": coefficient_a.tolist(), "coefficient_b": coefficient_b.tolist(), "model_a": validation_report, "model_b": test_report}
    np.save(cache_root() / "candidate_val.npy", validation_prediction.astype(np.float32))
    np.save(cache_root() / "candidate_test.npy", test_prediction.astype(np.float32))
    return validation_prediction, test_prediction, diagnostics


def register_artifact() -> None:
    root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    registry = root / "artifacts.json"
    records = json.loads(registry.read_text()) if registry.exists() else []
    key = "rel-stack-user-badge-lane0-cold-replay-v1"
    if not any(record.get("content_key") == key for record in records):
        records.append({"name": "lane0 monthly N0/N1 cold replay", "path": "lane0_cold_replay_v1", "description": "Fingerprints, forward-fold champion predictions, and diagnostics for the 84-origin cold-start replay experts.", "content_key": key, "rebuild_hint": "Run main.py to rebuild cutoff-valid replay features and missing fold predictions."})
        registry.write_text(json.dumps(records, indent=2))
