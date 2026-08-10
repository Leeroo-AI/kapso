from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .graph_data import register_artifact, unix_seconds


# Event indices

class EventIndex:
    span = np.int64(4_000_000_000)

    def __init__(self, users: np.ndarray, timestamps: np.ndarray):
        users = np.asarray(users)
        timestamps = np.asarray(timestamps, dtype=np.int64)
        valid = np.isfinite(users) & (users >= 0)
        self.users = users[valid].astype(np.int64)
        self.timestamps = timestamps[valid]
        keys = self.users * self.span + self.timestamps
        order = np.argsort(keys, kind="stable")
        self.keys = keys[order]
        self.event_times = self.timestamps[order]
        self.event_users = self.users[order]

    def query(self, users: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        timestamps = np.asarray(timestamps, dtype=np.int64)
        query_key = users * self.span + timestamps
        right = np.searchsorted(self.keys, query_key, side="right")
        start = np.searchsorted(self.keys, users * self.span, side="left")
        output = np.empty((len(users), 8), dtype=np.float32)
        for column, days in enumerate((30, 91, 365, 730)):
            left_key = users * self.span + timestamps - np.int64(days * 86400)
            left = np.searchsorted(self.keys, left_key, side="right")
            output[:, column] = np.log1p(np.maximum(right - left, 0))
        lifetime = np.maximum(right - start, 0)
        output[:, 4] = np.log1p(lifetime)
        last_index = np.maximum(right - 1, 0)
        has_last = (right > start) & (right > 0)
        last_time = np.where(has_last, self.event_times[last_index], timestamps - 3650 * 86400)
        recency = np.maximum(timestamps - last_time, 0) / 86400.0
        output[:, 5] = np.log1p(np.minimum(recency, 3650.0))
        second_index = np.maximum(right - 2, 0)
        has_second = lifetime >= 2
        second_time = np.where(has_second, self.event_times[second_index], last_time)
        gap = np.maximum(last_time - second_time, 0) / 86400.0
        output[:, 6] = np.log1p(np.minimum(gap, 3650.0))
        first_index = np.minimum(start, max(len(self.event_times) - 1, 0))
        if len(self.event_times):
            first_time = self.event_times[first_index]
            span_years = np.maximum((timestamps - first_time) / (365.25 * 86400), 1 / 365.25)
        else:
            span_years = np.ones(len(users), dtype=np.float64)
        output[:, 7] = np.log1p(lifetime / span_years)
        return output


def mapped_users(frame: pd.DataFrame, column: str, post_owner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(raw) & (raw >= 0) & (raw < len(post_owner))
    post_ids = raw[valid].astype(np.int64)
    users = post_owner[post_ids]
    valid_owner = np.isfinite(users) & (users >= 0)
    return users[valid_owner], np.flatnonzero(valid)[valid_owner]


def make_event_indices(db) -> dict[str, EventIndex]:
    posts = db.table_dict["posts"].df
    comments = db.table_dict["comments"].df
    votes = db.table_dict["votes"].df
    history = db.table_dict["postHistory"].df
    badges = db.table_dict["badges"].df
    links = db.table_dict["postLinks"].df
    post_owner = pd.to_numeric(posts["OwnerUserId"], errors="coerce").to_numpy(dtype=np.float64)

    def direct(frame: pd.DataFrame, user_col: str, time_col: str) -> EventIndex:
        return EventIndex(
            pd.to_numeric(frame[user_col], errors="coerce").to_numpy(dtype=np.float64),
            unix_seconds(frame[time_col]),
        )

    indices: dict[str, EventIndex] = {
        "own_posts": direct(posts, "OwnerUserId", "CreationDate"),
        "own_questions": direct(posts.loc[posts["PostTypeId"].eq(1)], "OwnerUserId", "CreationDate"),
        "own_answers": direct(posts.loc[posts["PostTypeId"].eq(2)], "OwnerUserId", "CreationDate"),
        "own_comments": direct(comments, "UserId", "CreationDate"),
        "own_bounties": direct(votes.loc[votes["UserId"].notna()], "UserId", "CreationDate"),
        "own_history": direct(history, "UserId", "CreationDate"),
        "badges": direct(badges, "UserId", "Date"),
        "voting_badges": direct(
            badges.loc[
                badges["Name"].isin(
                    [
                        "Supporter",
                        "Civic Duty",
                        "Suffrage",
                        "Electorate",
                        "Sportsmanship",
                        "Critic",
                        "Commentator",
                        "Editor",
                    ]
                )
            ],
            "UserId",
            "Date",
        ),
    }
    received_vote_users, received_vote_rows = mapped_users(votes, "PostId", post_owner)
    indices["received_votes"] = EventIndex(
        received_vote_users,
        unix_seconds(votes["CreationDate"])[received_vote_rows],
    )
    upvotes = votes["VoteTypeId"].eq(2).to_numpy()
    upvote_users, upvote_rows_local = mapped_users(votes.loc[upvotes], "PostId", post_owner)
    indices["received_upvotes"] = EventIndex(
        upvote_users,
        unix_seconds(votes.loc[upvotes, "CreationDate"])[upvote_rows_local],
    )
    received_comment_users, received_comment_rows = mapped_users(comments, "PostId", post_owner)
    indices["received_comments"] = EventIndex(
        received_comment_users,
        unix_seconds(comments["CreationDate"])[received_comment_rows],
    )
    answers = posts.loc[posts["PostTypeId"].eq(2) & posts["ParentId"].notna()]
    answer_users, answer_rows = mapped_users(answers, "ParentId", post_owner)
    indices["received_answers"] = EventIndex(
        answer_users,
        unix_seconds(answers["CreationDate"])[answer_rows],
    )
    link_user_parts: list[np.ndarray] = []
    link_time_parts: list[np.ndarray] = []
    link_times = unix_seconds(links["CreationDate"])
    for column in ("PostId", "RelatedPostId"):
        link_users, link_rows = mapped_users(links, column, post_owner)
        link_user_parts.append(link_users)
        link_time_parts.append(link_times[link_rows])
    indices["post_links"] = EventIndex(
        np.concatenate(link_user_parts),
        np.concatenate(link_time_parts),
    )
    return indices


# Feature matrices

@dataclass
class FeatureMatrices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    names: list[str]
    train_frame: pd.DataFrame


def add_group_normalization(
    matrix: np.ndarray,
    timestamps: np.ndarray,
    selected: list[int],
) -> np.ndarray:
    additions = np.empty((len(matrix), len(selected) * 3), dtype=np.float32)
    offset = 0
    for column in selected:
        values = matrix[:, column]
        percentile = np.empty(len(values), dtype=np.float32)
        zscore = np.empty(len(values), dtype=np.float32)
        leader_gap = np.empty(len(values), dtype=np.float32)
        for timestamp in np.unique(timestamps):
            rows = np.flatnonzero(timestamps == timestamp)
            group = values[rows]
            order = np.argsort(group, kind="stable")
            rank = np.empty(len(rows), dtype=np.float32)
            rank[order] = np.arange(len(rows), dtype=np.float32)
            percentile[rows] = rank / max(len(rows) - 1, 1)
            standard = max(float(group.std()), 1e-6)
            zscore[rows] = (group - float(group.mean())) / standard
            leader_gap[rows] = float(group.max()) - group
        additions[:, offset] = percentile
        additions[:, offset + 1] = zscore
        additions[:, offset + 2] = leader_gap
        offset += 3
    return np.concatenate([matrix, additions], axis=1)


def build_features_for_frame(
    frame: pd.DataFrame,
    db,
    indices: dict[str, EventIndex],
    entity_col: str,
    time_col: str,
) -> tuple[np.ndarray, list[str]]:
    users = frame[entity_col].to_numpy(dtype=np.int64)
    timestamps = unix_seconds(frame[time_col])
    blocks: list[np.ndarray] = []
    names: list[str] = []
    suffixes = ["count30", "count91", "count365", "count730", "lifetime", "recency", "last_gap", "rate"]
    selected: list[int] = []
    for stream_name, index in indices.items():
        block = index.query(users, timestamps)
        base = len(names)
        blocks.append(block)
        names.extend([f"{stream_name}_{suffix}" for suffix in suffixes])
        if stream_name in {
            "own_posts",
            "own_comments",
            "own_bounties",
            "own_history",
            "badges",
            "received_votes",
            "received_comments",
            "received_answers",
        }:
            selected.extend([base + 1, base + 4, base + 5])
    users_df = db.table_dict["users"].df
    user_creation_all = unix_seconds(users_df["CreationDate"])
    user_creation = user_creation_all[np.clip(users, 0, len(user_creation_all) - 1)]
    tenure = np.maximum(timestamps - user_creation, 0) / 86400.0
    base = np.column_stack(
        [
            np.log1p(tenure),
            np.log1p(np.maximum(users, 0)),
            pd.to_datetime(frame[time_col]).dt.year.to_numpy(dtype=np.float32) - 2009.0,
            pd.to_datetime(frame[time_col]).dt.quarter.to_numpy(dtype=np.float32),
        ]
    ).astype(np.float32)
    blocks.append(base)
    names.extend(["tenure", "user_index", "origin_year", "origin_quarter"])
    matrix = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    matrix = add_group_normalization(matrix, timestamps, selected)
    for index in selected:
        names.extend([f"{names[index]}_percentile", f"{names[index]}_z", f"{names[index]}_leader_gap"])
    return matrix, names


def materialize_feature_matrices(ctx, shared_root: Path, debug: bool) -> FeatureMatrices:
    content_key = "rel-stack-causal-all-tables-ranks-lane3-v2"
    train_frame = ctx.train.df
    if debug:
        origins = np.sort(train_frame[ctx.task.time_col].unique())[-2:]
        train_frame = train_frame.loc[train_frame[ctx.task.time_col].isin(origins)].reset_index(drop=True)
    root = shared_root / content_key
    complete = root / "complete.json"
    expected = {
        "train": len(ctx.train.df),
        "val": len(ctx.val.df),
        "test": len(ctx.test.df),
    }
    if not debug and complete.exists():
        metadata = json.loads(complete.read_text())
        if metadata.get("rows") == expected:
            print(f"[gbdt] reused feature cache {content_key}")
            return FeatureMatrices(
                np.load(root / "train.npy", mmap_mode="r"),
                np.load(root / "val.npy", mmap_mode="r"),
                np.load(root / "test.npy", mmap_mode="r"),
                metadata["names"],
                ctx.train.df,
            )
    started = time.time()
    indices = make_event_indices(ctx.db)
    matrices: dict[str, np.ndarray] = {}
    names: list[str] = []
    frames = {"train": train_frame, "val": ctx.val.df, "test": ctx.test.df}
    for split, frame in frames.items():
        split_started = time.time()
        matrices[split], split_names = build_features_for_frame(
            frame,
            ctx.db,
            indices,
            ctx.task.entity_col,
            ctx.task.time_col,
        )
        if not names:
            names = split_names
        if split_names != names:
            raise RuntimeError("feature names diverged across splits")
        print(
            f"[gbdt] features {split}: shape={matrices[split].shape} "
            f"elapsed={time.time() - split_started:.1f}s"
        )
    if not debug:
        root.mkdir(parents=True, exist_ok=True)
        for split, matrix in matrices.items():
            np.save(root / f"{split}.npy", matrix)
        complete.write_text(json.dumps({"content_key": content_key, "rows": expected, "names": names}))
        register_artifact(
            shared_root,
            {
                "name": "lane3 causal all-table feature matrices",
                "path": content_key,
                "description": "Query-censored user histories, gaps, attention, badges, links, tenure, and within-origin normalizations.",
                "content_key": content_key,
                "rebuild_hint": "Run the full lane-3 candidate after changing the feature content key.",
            },
        )
    print(f"[gbdt] all feature matrices elapsed={time.time() - started:.1f}s")
    return FeatureMatrices(
        matrices["train"],
        matrices["val"],
        matrices["test"],
        names,
        train_frame,
    )


# LightGBM

def episode_weights(frame: pd.DataFrame, time_col: str, half_life_days: float = 730.0) -> np.ndarray:
    timestamps = pd.to_datetime(frame[time_col])
    counts = timestamps.value_counts()
    age_days = (timestamps.max() - timestamps).dt.total_seconds().to_numpy() / 86400.0
    decay = np.exp2(-age_days / half_life_days)
    normalization = timestamps.map(counts).to_numpy(dtype=np.float64)
    weights = decay / normalization
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def fit_gbdt(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    debug: bool,
    seed: int,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=80 if debug else 650,
        learning_rate=0.05 if debug else 0.035,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=500,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.20,
        reg_lambda=2.0,
        max_bin=127,
        random_state=seed,
        n_jobs=11,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    started = time.time()
    model.fit(matrix, labels, sample_weight=weights, callbacks=[lgb.log_evaluation(0)])
    print(f"[gbdt] fitted rows={len(labels)} trees={model.n_estimators} elapsed={time.time() - started:.1f}s")
    return model


def predict_gbdt(model: lgb.LGBMClassifier, matrix: np.ndarray) -> np.ndarray:
    return model.predict_proba(matrix, num_iteration=model.best_iteration_)[:, 1].astype(np.float64)


def auc_by_origin(
    frame: pd.DataFrame,
    time_col: str,
    target_col: str,
    predictions: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}
    timestamps = frame[time_col].to_numpy()
    labels = frame[target_col].to_numpy(dtype=np.int64)
    for timestamp in np.unique(timestamps):
        mask = timestamps == timestamp
        result[str(timestamp)] = float(roc_auc_score(labels[mask], predictions[mask]))
    return result
