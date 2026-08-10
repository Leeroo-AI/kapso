from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy.stats import rankdata

from temporal_features import cache_root, database_root, load_users, register_artifact


# Names

def mechanism_feature_names() -> list[str]:
    names = [
        "question_score_max",
        "answer_score_max",
        "question_upvote_max",
        "answer_upvote_max",
    ]
    names += [f"question_score_ge_{threshold}" for threshold in (1, 10, 25, 100)]
    names += [f"answer_score_ge_{threshold}" for threshold in (1, 10, 25, 100)]
    names += [
        "accepted_answer_posts",
        "accepted_vote_total",
        "questions_without_answer",
        "questions_with_answer",
        "answer_count_max_question",
        "answer_count_total",
        "comment_count_max_post",
        "commented_post_count",
        "vote_count_max_post",
        "voted_post_count",
        "top_post_vote_fraction",
        "votes_30d_owned_posts",
        "votes_91d_owned_posts",
        "voted_posts_91d",
        "votes_91d_old_posts",
        "question_votes_91d",
        "answer_votes_91d",
        "score_momentum_91d",
    ]
    names += [f"question_gap_to_{threshold}" for threshold in (1, 10, 25, 100)]
    names += [f"answer_gap_to_{threshold}" for threshold in (10, 25, 100)]
    names += [
        "question_near_threshold_with_momentum",
        "answer_near_threshold_with_momentum",
        "old_post_momentum_fraction",
        "accepted_answer_fraction",
        "question_answer_rate",
        "rank_question_score_max",
        "rank_answer_score_max",
        "rank_votes_91d_old_posts",
        "z_votes_91d_old_posts",
    ]
    return names


# Helpers

def _maximum(users: np.ndarray, values: np.ndarray, n_users: int) -> np.ndarray:
    output = np.zeros(n_users, dtype=np.float32)
    np.maximum.at(output, users, values.astype(np.float32))
    return output


def _count(users: np.ndarray, n_users: int, weights: np.ndarray | None = None) -> np.ndarray:
    return np.bincount(users, weights=weights, minlength=n_users).astype(np.float32)


def _threshold_gap(users: np.ndarray, scores: np.ndarray, threshold: int, n_users: int) -> np.ndarray:
    output = np.full(n_users, float(threshold), dtype=np.float32)
    below = scores < threshold
    if np.any(below):
        np.minimum.at(output, users[below], threshold - scores[below])
    return np.log1p(output)


# Build

def build_mechanism_features(frames: list[pd.DataFrame], mapped_users: np.ndarray) -> np.memmap:
    path = cache_root() / "mechanism_v3.npy"
    names = mechanism_feature_names()
    total_rows = sum(len(frame) for frame in frames)
    if path.exists():
        matrix = np.load(path, mmap_mode="r")
        if matrix.shape == (total_rows, len(names)):
            return matrix
    db = database_root()
    users = load_users()
    user_ids = users["Id"].to_numpy(dtype=np.int64)
    posts = duckdb.sql(
        f"select Id,OwnerUserId,PostTypeId,ParentId,CreationDate from read_parquet('{db / 'posts.parquet'}') order by Id"
    ).df()
    post_ids = posts["Id"].to_numpy(dtype=np.int64)
    n_posts = len(posts)
    n_users = len(users)
    owner_id = posts["OwnerUserId"].fillna(-1).to_numpy(dtype=np.int64)
    owner = np.searchsorted(user_ids, np.maximum(owner_id, 0)).astype(np.int32)
    owner[(owner_id < 0) | (owner >= n_users) | (user_ids[np.minimum(owner, n_users - 1)] != owner_id)] = -1
    post_type = posts["PostTypeId"].to_numpy(dtype=np.int8)
    post_creation = posts["CreationDate"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    parent_id = posts["ParentId"].fillna(-1).to_numpy(dtype=np.int64)
    parent = np.searchsorted(post_ids, np.maximum(parent_id, 0)).astype(np.int32)
    parent[(parent_id < 0) | (parent >= n_posts) | (post_ids[np.minimum(parent, n_posts - 1)] != parent_id)] = -1
    votes = duckdb.sql(
        f"select PostId,VoteTypeId,CreationDate from read_parquet('{db / 'votes.parquet'}') where PostId is not null order by CreationDate,Id"
    ).df()
    vote_post_id = votes["PostId"].to_numpy(dtype=np.int64)
    vote_post = np.searchsorted(post_ids, vote_post_id).astype(np.int32)
    vote_valid = (vote_post < n_posts) & (post_ids[np.minimum(vote_post, n_posts - 1)] == vote_post_id)
    vote_post = vote_post[vote_valid]
    vote_time = votes.loc[vote_valid, "CreationDate"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    vote_type_raw = votes.loc[vote_valid, "VoteTypeId"].to_numpy(dtype=np.int16)
    vote_group = np.select(
        [vote_type_raw == 2, vote_type_raw == 3, vote_type_raw == 1, vote_type_raw == 10],
        [0, 1, 2, 3],
        default=4,
    ).astype(np.int8)
    comments = duckdb.sql(
        f"select PostId,CreationDate from read_parquet('{db / 'comments.parquet'}') where PostId is not null order by CreationDate,Id"
    ).df()
    comment_post_id = comments["PostId"].to_numpy(dtype=np.int64)
    comment_post = np.searchsorted(post_ids, comment_post_id).astype(np.int32)
    comment_valid = (comment_post < n_posts) & (post_ids[np.minimum(comment_post, n_posts - 1)] == comment_post_id)
    comment_post = comment_post[comment_valid]
    comment_time = comments.loc[comment_valid, "CreationDate"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    del votes, comments
    matrix = open_memmap(path, mode="w+", dtype=np.float16, shape=(total_rows, len(names)))
    combined_time = np.concatenate([frame["timestamp"].to_numpy(dtype="datetime64[s]").astype(np.int64) for frame in frames])
    vote_state = np.zeros((n_posts, 5), dtype=np.float32)
    comment_state = np.zeros(n_posts, dtype=np.float32)
    previous_vote = 0
    previous_comment = 0
    for cutoff in np.unique(combined_time):
        vote_end = np.searchsorted(vote_time, cutoff, side="right")
        vote_flat = vote_post[previous_vote:vote_end].astype(np.int64) * 5 + vote_group[previous_vote:vote_end]
        vote_state.ravel()[:] += np.bincount(vote_flat, minlength=n_posts * 5).astype(np.float32)
        previous_vote = vote_end
        comment_end = np.searchsorted(comment_time, cutoff, side="right")
        comment_state += np.bincount(comment_post[previous_comment:comment_end], minlength=n_posts).astype(np.float32)
        previous_comment = comment_end
        vote_lo30 = np.searchsorted(vote_time, cutoff - 30 * 86400, side="right")
        vote_lo91 = np.searchsorted(vote_time, cutoff - 91 * 86400, side="right")
        recent30 = np.bincount(vote_post[vote_lo30:vote_end], minlength=n_posts).astype(np.float32)
        recent91 = np.bincount(vote_post[vote_lo91:vote_end], minlength=n_posts).astype(np.float32)
        recent_up = np.bincount(vote_post[vote_lo91:vote_end], weights=(vote_group[vote_lo91:vote_end] == 0), minlength=n_posts).astype(np.float32)
        recent_down = np.bincount(vote_post[vote_lo91:vote_end], weights=(vote_group[vote_lo91:vote_end] == 1), minlength=n_posts).astype(np.float32)
        active_answers = (post_creation <= cutoff) & (parent >= 0)
        answer_count = np.bincount(parent[active_answers], minlength=n_posts).astype(np.float32)
        active = (post_creation <= cutoff) & (owner >= 0) & np.isin(post_type, (1, 2))
        post_index = np.flatnonzero(active)
        selected_owner = owner[post_index]
        question = post_type[post_index] == 1
        answer = post_type[post_index] == 2
        up = vote_state[post_index, 0]
        down = vote_state[post_index, 1]
        accepted = vote_state[post_index, 2]
        score = up - down
        total_votes = vote_state[post_index].sum(1)
        recent30_selected = recent30[post_index]
        recent91_selected = recent91[post_index]
        recent_score = recent_up[post_index] - recent_down[post_index]
        question_owner = selected_owner[question]
        answer_owner = selected_owner[answer]
        question_score = score[question]
        answer_score = score[answer]
        blocks = [
            np.log1p(_maximum(question_owner, np.maximum(0, question_score), n_users)),
            np.log1p(_maximum(answer_owner, np.maximum(0, answer_score), n_users)),
            np.log1p(_maximum(question_owner, up[question], n_users)),
            np.log1p(_maximum(answer_owner, up[answer], n_users)),
        ]
        for threshold in (1, 10, 25, 100):
            blocks.append(np.log1p(_count(question_owner[question_score >= threshold], n_users)))
        for threshold in (1, 10, 25, 100):
            blocks.append(np.log1p(_count(answer_owner[answer_score >= threshold], n_users)))
        blocks.extend(
            [
                np.log1p(_count(answer_owner[accepted[answer] > 0], n_users)),
                np.log1p(_count(selected_owner, n_users, accepted)),
                np.log1p(_count(question_owner[answer_count[post_index][question] == 0], n_users)),
                np.log1p(_count(question_owner[answer_count[post_index][question] > 0], n_users)),
                np.log1p(_maximum(question_owner, answer_count[post_index][question], n_users)),
                np.log1p(_count(question_owner, n_users, answer_count[post_index][question])),
                np.log1p(_maximum(selected_owner, comment_state[post_index], n_users)),
                np.log1p(_count(selected_owner[comment_state[post_index] > 0], n_users)),
                np.log1p(_maximum(selected_owner, total_votes, n_users)),
                np.log1p(_count(selected_owner[total_votes > 0], n_users)),
            ]
        )
        owner_vote_total = _count(selected_owner, n_users, total_votes)
        owner_vote_max = _maximum(selected_owner, total_votes, n_users)
        recent30_user = _count(selected_owner, n_users, recent30_selected)
        recent91_user = _count(selected_owner, n_users, recent91_selected)
        old = cutoff - post_creation[post_index] > 365 * 86400
        old_recent = _count(selected_owner, n_users, recent91_selected * old)
        blocks.extend(
            [
                owner_vote_max / np.maximum(1, owner_vote_total),
                np.log1p(recent30_user),
                np.log1p(recent91_user),
                np.log1p(_count(selected_owner[recent91_selected > 0], n_users)),
                np.log1p(old_recent),
                np.log1p(_count(question_owner, n_users, recent91_selected[question])),
                np.log1p(_count(answer_owner, n_users, recent91_selected[answer])),
                np.sign(_count(selected_owner, n_users, recent_score)) * np.log1p(np.abs(_count(selected_owner, n_users, recent_score))),
            ]
        )
        for threshold in (1, 10, 25, 100):
            blocks.append(_threshold_gap(question_owner, question_score, threshold, n_users))
        for threshold in (10, 25, 100):
            blocks.append(_threshold_gap(answer_owner, answer_score, threshold, n_users))
        question_near = (question_score < 100) & (question_score >= 0) & (recent_score[question] > 0)
        answer_near = (answer_score < 100) & (answer_score >= 0) & (recent_score[answer] > 0)
        question_total = _count(question_owner, n_users)
        answer_total = _count(answer_owner, n_users)
        accepted_total = _count(answer_owner[accepted[answer] > 0], n_users)
        blocks.extend(
            [
                np.log1p(_count(question_owner[question_near], n_users)),
                np.log1p(_count(answer_owner[answer_near], n_users)),
                old_recent / np.maximum(1, recent91_user),
                accepted_total / np.maximum(1, answer_total),
                _count(question_owner, n_users, answer_count[post_index][question]) / np.maximum(1, question_total),
            ]
        )
        user_values = np.column_stack(blocks).astype(np.float32)
        rows = np.flatnonzero(combined_time == cutoff)
        selected = user_values[mapped_users[rows]]
        rank_question = rankdata(selected[:, 0], method="average").astype(np.float32) / len(rows)
        rank_answer = rankdata(selected[:, 1], method="average").astype(np.float32) / len(rows)
        rank_old = rankdata(selected[:, 26], method="average").astype(np.float32) / len(rows)
        old_column = selected[:, 26]
        z_old = (old_column - old_column.mean()) / max(float(old_column.std()), 1e-5)
        matrix[rows] = np.column_stack([selected, rank_question, rank_answer, rank_old, z_old]).astype(np.float16)
        matrix.flush()
        print(f"[mechanism] origin={pd.to_datetime(cutoff, unit='s').date()} rows={len(rows)}", flush=True)
    register_artifact(
        "lane3 post threshold mechanism features",
        path,
        "Temporally reconstructed per-post scores, badge thresholds, responses, and old-post momentum.",
        "rel-stack-user-badge-lane3-mechanism-v3",
    )
    return np.load(path, mmap_mode="r")
