from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy.stats import rankdata
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import Ridge

from topic_data import QuestionData, build_embeddings, register_topic_artifact, state_root, topic_root


# Counts

DAY = 86400


def _counts(times: np.ndarray, questions: np.ndarray, lower: int, upper: int, size: int) -> np.ndarray:
    start = int(np.searchsorted(times, lower, side="right"))
    stop = int(np.searchsorted(times, upper, side="right"))
    return np.bincount(np.asarray(questions[start:stop], dtype=np.int64), minlength=size).astype(np.float32)


def _event_counts(data: QuestionData, name: str, lower: int, upper: int) -> np.ndarray:
    times, questions = data.events[name]
    return _counts(np.asarray(times), np.asarray(questions), lower, upper, len(data.question_ids))


def _question_window(data: QuestionData, cutoff: int, days: int) -> dict[str, np.ndarray]:
    lower = cutoff - days * DAY
    output = {name: _event_counts(data, name, lower, cutoff) for name in data.events}
    output["new"] = ((data.creation > lower) & (data.creation <= cutoff)).astype(np.float32)
    return output


def _future_window(data: QuestionData, cutoff: int) -> np.ndarray:
    return np.column_stack(
        [_event_counts(data, name, cutoff, cutoff + 91 * DAY) for name in ("votes", "comments", "answers", "links")]
    ).astype(np.float32)


def _lifetime(data: QuestionData, cutoff: int) -> dict[str, np.ndarray]:
    lower = int(min(data.creation.min(), min(np.asarray(pair[0])[0] for pair in data.events.values()))) - 1
    return {name: _event_counts(data, name, lower, cutoff) for name in data.events}


# Aggregation

def _sum(groups: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(groups, weights=values, minlength=size).astype(np.float32)


def _maximum(groups: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    output = np.zeros(size, dtype=np.float32)
    np.maximum.at(output, groups, values)
    return output


def _mean(groups: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    count = np.bincount(groups, minlength=size).astype(np.float32)
    return _sum(groups, values, size) / np.maximum(1, count)


def _top_three(groups: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    if not len(groups):
        return np.zeros(size, dtype=np.float32)
    order = np.lexsort((-values, groups))
    ordered_groups = groups[order]
    starts = np.maximum.accumulate(np.where(np.r_[True, ordered_groups[1:] != ordered_groups[:-1]], np.arange(len(order)), 0))
    within = np.arange(len(order)) - starts
    selected = within < 3
    return _sum(ordered_groups[selected], values[order][selected], size)


def _statistics(groups: np.ndarray, values: np.ndarray, size: int) -> list[np.ndarray]:
    return [_sum(groups, values, size), _mean(groups, values, size), _maximum(groups, values, size)]


def _normalizations(values: np.ndarray, columns: list[int]) -> np.ndarray:
    output = []
    for column in columns:
        current = values[:, column]
        output.append(rankdata(current, method="average").astype(np.float32) / max(1, len(current)))
        output.append((current - float(current.mean())) / max(float(current.std()), 1e-5))
        output.append(float(current.max()) - current)
    return np.column_stack(output).astype(np.float32)


# Tag traffic

def tag_feature_names() -> list[str]:
    metrics = ("heat30", "heat91", "heat365", "growth30", "growth91", "growth365", "acceleration", "active_questions", "percentile")
    names = [f"tag_{metric}_{stat}" for metric in metrics for stat in ("sum", "mean", "max")]
    names += [f"tag_{metric}_top3" for metric in ("heat30", "heat91", "heat365")]
    names += [f"old_tag_{metric}_{stat}" for metric in ("heat91", "heat365", "growth91", "percentile") for stat in ("sum", "mean", "max", "top3")]
    names += ["top_decile_tag_count", "old_top_decile_tag_count", "heat_weighted_question_age", "old_heat_weighted_question_age"]
    names += [f"heat_interaction_{signal}_{stat}" for signal in ("vote_lifetime", "vote_recent", "comment_lifetime", "comment_recent") for stat in ("sum", "max")]
    names += ["owned_question_count", "owned_old_question_count", "question_tag_pair_count", "historical_text_missing_count", "historical_tag_missing_count", "non_latin_text_count"]
    names += [f"tag_{signal}_{stat}" for signal in ("new91", "vote91", "comment91", "answer91", "link91") for stat in ("mean", "max")]
    normalized = ("tag_heat91_sum", "tag_heat91_max", "tag_heat365_sum", "tag_growth91_max", "tag_percentile_max", "old_tag_heat91_sum", "top_decile_tag_count", "heat_interaction_vote_lifetime_sum", "heat_interaction_comment_lifetime_sum", "owned_question_count")
    for name in normalized:
        names += [f"rank_{name}", f"z_{name}", f"leader_gap_{name}"]
    return names


def _tag_values(data: QuestionData, origin_index: int, cutoff: int) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    pair_question, pair_tag = data.pairs(origin_index)
    pair_question = np.asarray(pair_question, dtype=np.int32)
    pair_tag = np.asarray(pair_tag, dtype=np.int32)
    size = len(data.tag_names)
    current30 = _question_window(data, cutoff, 30)
    current91 = _question_window(data, cutoff, 91)
    current365 = _question_window(data, cutoff, 365)
    prior30 = _question_window(data, cutoff - 365 * DAY, 30)
    prior91 = _question_window(data, cutoff - 365 * DAY, 91)
    prior365 = _question_window(data, cutoff - 365 * DAY, 365)

    def tag_sum(values: np.ndarray) -> np.ndarray:
        return np.bincount(pair_tag, weights=values[pair_question], minlength=size).astype(np.float32)

    raw = {
        "new30": tag_sum(current30["new"]),
        "new91": tag_sum(current91["new"]),
        "new365": tag_sum(current365["new"]),
        "vote30": tag_sum(current30["votes"]),
        "vote91": tag_sum(current91["votes"]),
        "vote365": tag_sum(current365["votes"]),
        "comment91": tag_sum(current91["comments"]),
        "comment365": tag_sum(current365["comments"]),
        "answer91": tag_sum(current91["answers"]),
        "answer365": tag_sum(current365["answers"]),
        "link91": tag_sum(current91["links"]),
        "link365": tag_sum(current365["links"]),
    }
    prior_raw = {
        "heat30": tag_sum(prior30["new"] + prior30["votes"]),
        "heat91": tag_sum(prior91["new"] + prior91["votes"] + prior91["comments"] + prior91["answers"] + prior91["links"]),
        "heat365": tag_sum(prior365["new"] + prior365["votes"] + prior365["comments"] + prior365["answers"] + prior365["links"]),
    }
    heat30 = raw["new30"] + raw["vote30"]
    heat91 = raw["new91"] + raw["vote91"] + raw["comment91"] + raw["answer91"] + raw["link91"]
    heat365 = raw["new365"] + raw["vote365"] + raw["comment365"] + raw["answer365"] + raw["link365"]
    recent_activity = current91["votes"] + current91["comments"] + current91["answers"] + current91["links"]
    active_questions = tag_sum((recent_activity > 0).astype(np.float32))
    percentile = rankdata(heat91, method="average").astype(np.float32) / max(1, len(heat91))
    metrics = {
        "heat30": np.log1p(heat30),
        "heat91": np.log1p(heat91),
        "heat365": np.log1p(heat365),
        "growth30": np.clip((heat30 + 1) / (prior_raw["heat30"] + 1), 0.1, 10),
        "growth91": np.clip((heat91 + 1) / (prior_raw["heat91"] + 1), 0.1, 10),
        "growth365": np.clip((heat365 + 1) / (prior_raw["heat365"] + 1), 0.1, 10),
        "acceleration": np.log(np.clip((heat30 / 30 + 1) / (np.maximum(0, heat91 - heat30) / 61 + 1), 0.1, 10)),
        "active_questions": np.log1p(active_questions),
        "percentile": percentile,
    }
    question = {
        "vote_lifetime": _lifetime(data, cutoff)["votes"],
        "vote_recent": current91["votes"],
        "comment_lifetime": _lifetime(data, cutoff)["comments"],
        "comment_recent": current91["comments"],
    }
    return np.column_stack([pair_question, pair_tag]), metrics, {**raw, **question}


def build_tag_features(data: QuestionData, frames: list[pd.DataFrame], mapped_users: np.ndarray) -> np.ndarray:
    root = state_root(data.origins)
    path = root / "tag_features_v1.npy"
    names = tag_feature_names()
    total_rows = sum(len(frame) for frame in frames)
    complete = path.with_suffix(".complete")
    if path.exists() and complete.exists():
        matrix = np.load(path, mmap_mode="r")
        if matrix.shape == (total_rows, len(names)):
            return matrix
    matrix = open_memmap(path, mode="w+", dtype=np.float16, shape=(total_rows, len(names)))
    users = int(mapped_users.max()) + 1
    combined_time = np.concatenate([frame["timestamp"].to_numpy(dtype="datetime64[s]").astype(np.int64) for frame in frames])
    started = time.time()
    metric_names = ("heat30", "heat91", "heat365", "growth30", "growth91", "growth365", "acceleration", "active_questions", "percentile")
    for origin_index, cutoff in enumerate(data.origins):
        pairs, metrics, raw = _tag_values(data, origin_index, int(cutoff))
        pair_question = pairs[:, 0].astype(np.int32)
        pair_tag = pairs[:, 1].astype(np.int32)
        pair_owner = data.owners[pair_question]
        valid = pair_owner >= 0
        pair_question = pair_question[valid]
        pair_tag = pair_tag[valid]
        pair_owner = pair_owner[valid]
        age = np.maximum(0, (cutoff - data.creation[pair_question]) / DAY).astype(np.float32)
        old = age > 365
        blocks = []
        for name in metric_names:
            blocks.extend(_statistics(pair_owner, metrics[name][pair_tag], users))
        for name in ("heat30", "heat91", "heat365"):
            blocks.append(_top_three(pair_owner, metrics[name][pair_tag], users))
        for name in ("heat91", "heat365", "growth91", "percentile"):
            values = metrics[name][pair_tag][old]
            groups = pair_owner[old]
            blocks.extend(_statistics(groups, values, users))
            blocks.append(_top_three(groups, values, users))
        decile = metrics["percentile"][pair_tag] >= 0.9
        blocks.append(np.bincount(pair_owner[decile], minlength=users).astype(np.float32))
        blocks.append(np.bincount(pair_owner[decile & old], minlength=users).astype(np.float32))
        weight = np.maximum(metrics["heat91"][pair_tag], 1e-4)
        blocks.append(_sum(pair_owner, weight * np.log1p(age), users) / np.maximum(_sum(pair_owner, weight, users), 1e-4))
        blocks.append(_sum(pair_owner[old], weight[old] * np.log1p(age[old]), users) / np.maximum(_sum(pair_owner[old], weight[old], users), 1e-4))
        for signal in ("vote_lifetime", "vote_recent", "comment_lifetime", "comment_recent"):
            interaction = metrics["heat91"][pair_tag] * np.log1p(raw[signal][pair_question])
            blocks.extend([_sum(pair_owner, interaction, users), _maximum(pair_owner, interaction, users)])
        active_question = (data.creation <= cutoff) & (data.owners >= 0)
        active_owner = data.owners[active_question]
        active_age = (cutoff - data.creation[active_question]) / DAY
        question_count = np.bincount(active_owner, minlength=users).astype(np.float32)
        old_count = np.bincount(active_owner[active_age > 365], minlength=users).astype(np.float32)
        pair_count = np.bincount(pair_owner, minlength=users).astype(np.float32)
        state = np.asarray(data.state_index[origin_index])
        missing = active_question & (state < 0)
        missing_count = np.bincount(data.owners[missing], minlength=users).astype(np.float32)
        tag_question = np.zeros(len(data.question_ids), dtype=np.bool_)
        tag_question[pair_question] = True
        missing_tag = active_question & ~tag_question
        missing_tag_count = np.bincount(data.owners[missing_tag], minlength=users).astype(np.float32)
        non_latin = active_question & (state >= 0) & (np.asarray(data.state_latin[origin_index]) == 0)
        non_latin_count = np.bincount(data.owners[non_latin], minlength=users).astype(np.float32)
        blocks.extend([np.log1p(question_count), np.log1p(old_count), np.log1p(pair_count), np.log1p(missing_count), np.log1p(missing_tag_count), np.log1p(non_latin_count)])
        for signal in ("new91", "vote91", "comment91", "answer91", "link91"):
            values = np.log1p(raw[signal][pair_tag])
            blocks.extend([_mean(pair_owner, values, users), _maximum(pair_owner, values, users)])
        user_values = np.column_stack(blocks).astype(np.float32)
        rows = np.flatnonzero(combined_time == cutoff)
        selected = user_values[mapped_users[rows]]
        normalized = _normalizations(selected, [0, 2, 6, 14, 26, 34, 46, 50, 54, 58])
        matrix[rows] = np.column_stack([selected, normalized]).astype(np.float16)
        matrix.flush()
        print(f"[tag-traffic] origin={pd.to_datetime(cutoff, unit='s').date()} rows={len(rows)} features={len(names)} elapsed_seconds={time.time() - started:.1f}", flush=True)
    complete.write_text("complete\n")
    register_topic_artifact(
        "lane3 tag traffic user features",
        path,
        "Cutoff-valid per-tag activity windows, prior-year heat growth, system percentiles, and old-question owner aggregates.",
        f"rel-stack-user-badge-lane3-tag-features-{root.name}-v1",
    )
    return np.load(path, mmap_mode="r")


# Content traffic

def content_feature_names() -> list[str]:
    names = []
    for scope in ("all", "old"):
        for target in ("votes", "comments", "answers", "links"):
            names += [f"predicted_{scope}_{target}_sum", f"predicted_{scope}_{target}_max"]
        names += [f"predicted_{scope}_traffic_sum", f"predicted_{scope}_traffic_max", f"predicted_{scope}_traffic_top3"]
        names += [f"predicted_{scope}_crossing_sum", f"predicted_{scope}_crossing_max"]
    names += ["content_question_count", "content_old_question_count", "content_missing_count", "content_non_latin_count", "content_coverage_fraction"]
    for name in ("predicted_all_votes_sum", "predicted_all_comments_sum", "predicted_all_answers_sum", "predicted_all_links_sum", "predicted_all_traffic_sum", "predicted_all_traffic_max", "predicted_old_traffic_sum", "content_coverage_fraction"):
        names += [f"rank_{name}", f"z_{name}", f"leader_gap_{name}"]
    return names


def _question_numeric(data: QuestionData, origin_index: int, cutoff: int) -> np.ndarray:
    recent = _question_window(data, cutoff, 91)
    lifetime = _lifetime(data, cutoff)
    age = np.maximum(0, (cutoff - data.creation) / DAY).astype(np.float32)
    state = np.asarray(data.state_index[origin_index])
    latin = np.asarray(data.state_latin[origin_index])
    columns = [np.log1p(age)]
    for name in ("votes", "comments", "answers", "links"):
        columns.extend([np.log1p(lifetime[name]), np.log1p(recent[name])])
    columns.extend([(state < 0).astype(np.float32), (latin == 0).astype(np.float32)])
    return np.column_stack(columns).astype(np.float32)


def _training_rows(data: QuestionData, embeddings: np.ndarray, boundary: int, debug: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible = np.flatnonzero(data.origins + 91 * DAY <= boundary)
    cap = 5000 if debug else 500000
    per_origin = max(1, math.ceil(cap / max(1, len(eligible))))
    rng = np.random.default_rng(8441 + int(boundary // DAY))
    state_parts = []
    numeric_parts = []
    target_parts = []
    for origin_index in eligible:
        cutoff = int(data.origins[origin_index])
        state = np.asarray(data.state_index[origin_index])
        available = np.flatnonzero((state >= 0) & (np.asarray(data.state_latin[origin_index]) > 0))
        if len(available) > per_origin:
            available = np.sort(rng.choice(available, per_origin, replace=False))
        if not len(available):
            continue
        state_parts.append(state[available])
        numeric_parts.append(_question_numeric(data, origin_index, cutoff)[available])
        target_parts.append(_future_window(data, cutoff)[available])
    states = np.concatenate(state_parts)
    numeric = np.concatenate(numeric_parts)
    targets = np.concatenate(target_parts)
    valid_embedding = np.linalg.norm(np.asarray(embeddings[states], dtype=np.float32), axis=1) > 0
    states = states[valid_embedding]
    numeric = numeric[valid_embedding]
    targets = targets[valid_embedding]
    if len(states) > cap:
        selected = np.sort(rng.choice(len(states), cap, replace=False))
        states, numeric, targets = states[selected], numeric[selected], targets[selected]
    return states, numeric, targets


def fit_traffic_auxiliary(data: QuestionData, embeddings: np.ndarray, boundary: int, debug: bool) -> dict:
    mode = "debug" if debug else "full"
    path = state_root(data.origins) / f"traffic_aux_{boundary}_{mode}_v1.joblib"
    if path.exists():
        return joblib.load(path)
    started = time.time()
    states, numeric, targets = _training_rows(data, embeddings, boundary, debug)
    if len(states) < 64:
        raise RuntimeError(f"traffic auxiliary has only {len(states)} eligible historical snapshots")
    embedding_values = np.asarray(embeddings[states], dtype=np.float32)
    pca = IncrementalPCA(n_components=32, batch_size=8192)
    pca.fit(embedding_values)
    reduced = pca.transform(embedding_values).astype(np.float32)
    values = np.column_stack([reduced, numeric]).astype(np.float32)
    ridge = Ridge(alpha=10.0)
    ridge.fit(values, np.log1p(targets))
    crossing_target = (targets.sum(axis=1) > 0).astype(np.uint8)
    crossing = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=30 if debug else 120,
        num_leaves=15,
        min_child_samples=250,
        learning_rate=0.05,
        max_bin=127,
        reg_lambda=5.0,
        reg_alpha=0.25,
        verbosity=-1,
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "11")),
        random_state=4417,
    )
    crossing.fit(values, crossing_target, callbacks=[lgb.log_evaluation(0)])
    model = {"pca": pca, "ridge": ridge, "crossing": crossing, "rows": len(states), "boundary": boundary}
    joblib.dump(model, path, compress=3)
    print(f"[traffic-aux] boundary={pd.to_datetime(boundary, unit='s').date()} rows={len(states)} positive_rate={crossing_target.mean():.6f} elapsed_seconds={time.time() - started:.1f}", flush=True)
    register_topic_artifact(
        f"lane3 content traffic auxiliary {boundary} {mode}",
        path,
        "Fold-safe IncrementalPCA-32, alpha-10 multi-target Ridge, and compact LightGBM traffic crossing model.",
        f"rel-stack-user-badge-lane3-traffic-aux-{boundary}-{mode}-v1",
    )
    return model


def _content_aggregate(owners: np.ndarray, prediction: np.ndarray, old: np.ndarray, users: int) -> np.ndarray:
    blocks = []
    total = prediction[:, :4].sum(axis=1)
    crossing = prediction[:, 4]
    for mask in (np.ones(len(owners), dtype=np.bool_), old):
        groups = owners[mask]
        for target in range(4):
            values = prediction[mask, target]
            blocks.extend([_sum(groups, values, users), _maximum(groups, values, users)])
        total_values = total[mask]
        blocks.extend([_sum(groups, total_values, users), _maximum(groups, total_values, users), _top_three(groups, total_values, users)])
        blocks.extend([_sum(groups, crossing[mask], users), _maximum(groups, crossing[mask], users)])
    return np.column_stack(blocks).astype(np.float32)


def build_content_features(data: QuestionData, frames: list[pd.DataFrame], mapped_users: np.ndarray, boundary: int, debug: bool) -> np.ndarray:
    root = state_root(data.origins)
    mode = "debug" if debug else "full"
    path = root / f"content_features_{boundary}_{mode}_v1.npy"
    names = content_feature_names()
    total_rows = sum(len(frame) for frame in frames)
    complete = path.with_suffix(".complete")
    if path.exists() and complete.exists():
        matrix = np.load(path, mmap_mode="r")
        if matrix.shape == (total_rows, len(names)):
            return matrix
    embeddings = build_embeddings(data, debug)
    model = fit_traffic_auxiliary(data, embeddings, boundary, debug)
    pca = model["pca"]
    state_reduced = np.zeros((len(data.texts), 32), dtype=np.float32)
    available_states = np.flatnonzero(np.linalg.norm(np.asarray(embeddings, dtype=np.float32), axis=1) > 0)
    for start in range(0, len(available_states), 32768):
        selected = available_states[start : start + 32768]
        state_reduced[selected] = pca.transform(np.asarray(embeddings[selected], dtype=np.float32)).astype(np.float32)
    matrix = open_memmap(path, mode="w+", dtype=np.float16, shape=(total_rows, len(names)))
    matrix[:] = 0
    combined_time = np.concatenate([frame["timestamp"].to_numpy(dtype="datetime64[s]").astype(np.int64) for frame in frames])
    users = int(mapped_users.max()) + 1
    started = time.time()
    for origin_index, cutoff in enumerate(data.origins):
        if cutoff > boundary:
            continue
        state = np.asarray(data.state_index[origin_index])
        latin = np.asarray(data.state_latin[origin_index]) > 0
        valid = (data.creation <= cutoff) & (data.owners >= 0) & (state >= 0) & latin & (np.linalg.norm(state_reduced[np.maximum(state, 0)], axis=1) > 0)
        question = np.flatnonzero(valid)
        numeric = _question_numeric(data, origin_index, int(cutoff))[question]
        values = np.column_stack([state_reduced[state[question]], numeric]).astype(np.float32)
        log_prediction = model["ridge"].predict(values)
        count_prediction = np.maximum(0, np.expm1(np.clip(log_prediction, -10, 8))).astype(np.float32)
        crossing = model["crossing"].predict_proba(values)[:, 1].astype(np.float32)
        prediction = np.column_stack([count_prediction, crossing])
        age = (cutoff - data.creation[question]) / DAY
        user_values = _content_aggregate(data.owners[question], prediction, age > 365, users)
        active = (data.creation <= cutoff) & (data.owners >= 0)
        active_owner = data.owners[active]
        active_age = (cutoff - data.creation[active]) / DAY
        question_count = np.bincount(active_owner, minlength=users).astype(np.float32)
        old_count = np.bincount(active_owner[active_age > 365], minlength=users).astype(np.float32)
        valid_count = np.bincount(data.owners[question], minlength=users).astype(np.float32)
        active_state = state[active]
        missing = active_state < 0
        non_latin = (active_state >= 0) & ~latin[active]
        missing_count = np.bincount(active_owner[missing], minlength=users).astype(np.float32)
        non_latin_count = np.bincount(active_owner[non_latin], minlength=users).astype(np.float32)
        user_values = np.column_stack(
            [
                user_values,
                np.log1p(question_count),
                np.log1p(old_count),
                np.log1p(missing_count),
                np.log1p(non_latin_count),
                valid_count / np.maximum(1, question_count),
            ]
        ).astype(np.float32)
        rows = np.flatnonzero(combined_time == cutoff)
        selected = user_values[mapped_users[rows]]
        normalized = _normalizations(selected, [0, 2, 4, 6, 8, 9, 21, 30])
        matrix[rows] = np.column_stack([selected, normalized]).astype(np.float16)
        matrix.flush()
        print(f"[content-traffic] boundary={pd.to_datetime(boundary, unit='s').date()} origin={pd.to_datetime(cutoff, unit='s').date()} questions={len(question)} elapsed_seconds={time.time() - started:.1f}", flush=True)
    complete.write_text("complete\n")
    register_topic_artifact(
        f"lane3 content-predicted traffic features {boundary} {mode}",
        path,
        "Owner sums, maxima, top-three totals, and old-question variants from fold-safe MiniLM traffic predictions.",
        f"rel-stack-user-badge-lane3-content-features-{root.name}-{boundary}-{mode}-v1",
    )
    return np.load(path, mmap_mode="r")


# Matrices

def combine_topic_features(base: np.ndarray, blocks: list[np.ndarray], name: str) -> np.ndarray:
    digest = hashlib.sha256(f"{base.shape}-{[block.shape for block in blocks]}-{name}".encode()).hexdigest()[:12]
    path = topic_root() / f"combined_{name}_{digest}.npy"
    shape = (len(base), base.shape[1] + sum(block.shape[1] for block in blocks))
    complete = path.with_suffix(".complete")
    if path.exists() and complete.exists():
        matrix = np.load(path, mmap_mode="r")
        if matrix.shape == shape:
            return matrix
    matrix = open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    for start in range(0, len(base), 100000):
        stop = min(len(base), start + 100000)
        matrix[start:stop] = np.column_stack([base[start:stop], *[block[start:stop] for block in blocks]]).astype(np.float16)
    matrix.flush()
    complete.write_text("complete\n")
    register_topic_artifact(
        f"lane3 champion plus accepted topic features {name}",
        path,
        "Champion matrix widened with accepted cutoff-valid tag and content-predicted traffic blocks.",
        f"rel-stack-user-badge-lane3-combined-topic-{digest}",
    )
    return np.load(path, mmap_mode="r")
