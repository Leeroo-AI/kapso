from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from graph_pipeline import GraphBundle, map_ids


@dataclass
class DirectBundle:
    matrix: np.ndarray
    names: list[str]
    graph_static: np.ndarray
    graph_names: list[str]
    content_embedding: np.ndarray
    content_cluster: np.ndarray
    event_start: np.ndarray
    attendance_time: np.ndarray
    attendance_user: np.ndarray
    attendance_status: np.ndarray


@dataclass
class DynamicBundle:
    matrix: np.ndarray
    names: list[str]
    cold_user: np.ndarray


def _safe_cosine(vector: np.ndarray, total: np.ndarray, count: float) -> float:
    if count <= 0:
        return 0.0
    centroid = total / count
    denominator = np.linalg.norm(vector) * np.linalg.norm(centroid)
    return float(np.dot(vector, centroid) / denominator) if denominator > 1e-8 else 0.0


def _event_rows(events: pd.DataFrame, event_ids: np.ndarray) -> pd.DataFrame:
    database_ids = events["event_id"].to_numpy(dtype=np.int64, copy=False)
    if np.all(database_ids[:-1] <= database_ids[1:]):
        positions = np.searchsorted(database_ids, event_ids)
        valid = positions < len(database_ids)
        valid &= database_ids[np.minimum(positions, len(database_ids) - 1)] == event_ids
        if not valid.all():
            raise RuntimeError("candidate event IDs are missing from events table")
        return events.iloc[positions].reset_index(drop=True)
    indexed = events.set_index("event_id", drop=False)
    return indexed.loc[event_ids].reset_index(drop=True)


def build_direct_bundle(seeds: pd.DataFrame, full_db, graph: GraphBundle) -> DirectBundle:
    started = time.time()
    seeds["user_idx"] = map_ids(seeds["user"].to_numpy(), graph.user_ids)
    if (seeds["user_idx"] < 0).any():
        raise RuntimeError("seed user missing from full user mapping")
    event_valid = seeds["event"].notna().to_numpy()
    seed_event_ids = seeds["event"].fillna(-1).to_numpy(dtype=np.int64)
    event_keys = seeds["event_key"].to_numpy(dtype=np.int64)
    unique_event_ids = np.unique(seed_event_ids[event_valid])
    events = full_db.table_dict["events"].df
    candidate_events = _event_rows(events, unique_event_ids)
    content_columns = [column for column in events.columns if column.startswith("c_")]
    content_raw = candidate_events[content_columns].fillna(0).to_numpy(dtype=np.float32)
    content_raw = np.log1p(np.maximum(content_raw, 0))
    occurrence = (content_raw > 0).sum(axis=0)
    content_raw *= np.log((len(content_raw) + 1) / (occurrence + 1)).astype(np.float32) + 1.0
    content_normalized = normalize(content_raw, norm="l2", copy=True)
    content_model = TruncatedSVD(n_components=16, n_iter=7, random_state=17)
    unique_content_embedding = content_model.fit_transform(content_normalized).astype(np.float32)
    content_cluster_model = MiniBatchKMeans(
        n_clusters=32, batch_size=2048, n_init=5, random_state=17
    )
    unique_content_cluster = content_cluster_model.fit_predict(unique_content_embedding).astype(np.int16)
    event_positions = np.zeros(len(seeds), dtype=np.int32)
    event_positions[event_valid] = np.searchsorted(unique_event_ids, seed_event_ids[event_valid])
    content_embedding = np.zeros((len(seeds), 16), dtype=np.float32)
    content_embedding[event_valid] = unique_content_embedding[event_positions[event_valid]]
    content_cluster = np.full(len(seeds), 32, dtype=np.int16)
    content_cluster[event_valid] = unique_content_cluster[event_positions[event_valid]]
    event_start_unique = candidate_events["start_time"].to_numpy(dtype="datetime64[ns]")
    event_start = np.full(len(seeds), np.datetime64("NaT"), dtype="datetime64[ns]")
    event_start[event_valid] = event_start_unique[event_positions[event_valid]]
    timestamps = seeds["timestamp"].to_numpy(dtype="datetime64[ns]")
    timestamp_series = pd.to_datetime(seeds["timestamp"])
    users = full_db.table_dict["users"].df.sort_values("user_id").reset_index(drop=True)
    if not np.array_equal(users["user_id"].to_numpy(dtype=np.int64), graph.user_ids):
        raise RuntimeError("user table order does not match graph mapping")
    user_idx = seeds["user_idx"].to_numpy(dtype=np.int32)
    timezone_all = users["timezone"].fillna(0).to_numpy(dtype=np.float32)
    timezone = timezone_all[user_idx]
    utc_hour = (
        timestamp_series.dt.hour.to_numpy(dtype=np.float32)
        + timestamp_series.dt.minute.to_numpy(dtype=np.float32) / 60.0
        + timestamp_series.dt.second.to_numpy(dtype=np.float32) / 3600.0
    )
    local_hour = np.mod(utc_hour + timezone, 24.0)
    weekday = timestamp_series.dt.dayofweek.to_numpy(dtype=np.float32)
    latency = (event_start - timestamps) / np.timedelta64(1, "D")
    joined_all = users["joinedAt"].to_numpy(dtype="datetime64[ns]")
    account_age = (timestamps - joined_all[user_idx]) / np.timedelta64(1, "D")
    birthyear_all = users["birthyear"].to_numpy(dtype=np.float32)
    age = timestamp_series.dt.year.to_numpy(dtype=np.float32) - birthyear_all[user_idx]
    gender = users["gender"].fillna("missing").astype(str).str.lower().to_numpy()[user_idx]
    locale_counts = users["locale"].fillna("missing").value_counts(normalize=True)
    location_counts = users["location"].fillna("missing").value_counts(normalize=True)
    locale_frequency = users["locale"].fillna("missing").map(locale_counts).to_numpy(dtype=np.float32)[user_idx]
    location_frequency = users["location"].fillna("missing").map(location_counts).to_numpy(dtype=np.float32)[user_idx]
    event_country = candidate_events["country"].fillna("missing")
    country_counts = event_country.value_counts(normalize=True)
    country_frequency = np.zeros(len(seeds), dtype=np.float32)
    lat = np.full(len(seeds), np.nan, dtype=np.float32)
    lng = np.full(len(seeds), np.nan, dtype=np.float32)
    content_nnz = np.zeros(len(seeds), dtype=np.float32)
    content_norm = np.zeros(len(seeds), dtype=np.float32)
    country_frequency[event_valid] = event_country.map(country_counts).to_numpy(dtype=np.float32)[event_positions[event_valid]]
    lat[event_valid] = candidate_events["lat"].to_numpy(dtype=np.float32)[event_positions[event_valid]]
    lng[event_valid] = candidate_events["lng"].to_numpy(dtype=np.float32)[event_positions[event_valid]]
    content_nnz[event_valid] = (content_raw > 0).sum(axis=1).astype(np.float32)[event_positions[event_valid]]
    content_norm[event_valid] = np.linalg.norm(content_raw, axis=1).astype(np.float32)[event_positions[event_valid]]
    n = len(seeds)
    event_prior_count = np.zeros(n, dtype=np.float32)
    event_prior_users = np.zeros(n, dtype=np.float32)
    user_prior_count = np.zeros(n, dtype=np.float32)
    time_since_user = np.full(n, np.nan, dtype=np.float32)
    event_counts: dict[int, int] = {}
    event_users: dict[int, set[int]] = {}
    user_counts: dict[int, int] = {}
    last_user_time: dict[int, np.datetime64] = {}
    ordered = seeds.sort_values(["timestamp", "row_id"], kind="mergesort")
    for _, batch in ordered.groupby("timestamp", sort=False):
        rows = batch["row_id"].to_numpy(dtype=np.int32)
        for row in rows:
            event = int(event_keys[row])
            user = int(user_idx[row])
            event_prior_count[row] = event_counts.get(event, 0)
            event_prior_users[row] = len(event_users.get(event, ()))
            user_prior_count[row] = user_counts.get(user, 0)
            if user in last_user_time:
                time_since_user[row] = float((timestamps[row] - last_user_time[user]) / np.timedelta64(1, "h"))
        for row in rows:
            event = int(event_keys[row])
            user = int(user_idx[row])
            event_counts[event] = event_counts.get(event, 0) + 1
            event_users.setdefault(event, set()).add(user)
            user_counts[user] = user_counts.get(user, 0) + 1
            last_user_time[user] = timestamps[row]
    batch_size = seeds.groupby(["user", "timestamp"], sort=False)["row_id"].transform("size").to_numpy(dtype=np.float32)
    sorted_for_position = seeds.sort_values(["user", "timestamp", "primary_key", "row_id"], kind="mergesort")
    positions = sorted_for_position.groupby(["user", "timestamp"], sort=False).cumcount().to_numpy(dtype=np.float32)
    batch_position = np.zeros(n, dtype=np.float32)
    batch_position[sorted_for_position["row_id"].to_numpy(dtype=np.int32)] = positions
    same_event_batch = seeds.groupby(["event_key", "timestamp"], sort=False)["row_id"].transform("size").to_numpy(dtype=np.float32)
    direct_parts = [
        seeds["invited"].to_numpy(dtype=np.float32)[:, None],
        np.sin(2 * np.pi * utc_hour / 24)[:, None],
        np.cos(2 * np.pi * utc_hour / 24)[:, None],
        np.sin(2 * np.pi * local_hour / 24)[:, None],
        np.cos(2 * np.pi * local_hour / 24)[:, None],
        np.sin(2 * np.pi * weekday / 7)[:, None],
        np.cos(2 * np.pi * weekday / 7)[:, None],
        np.sign(latency)[:, None],
        np.log1p(np.abs(latency))[:, None],
        (latency > 0).astype(np.float32)[:, None],
        np.log1p(np.maximum(account_age, 0))[:, None],
        age[:, None],
        np.isnan(age).astype(np.float32)[:, None],
        timezone[:, None],
        (gender == "male").astype(np.float32)[:, None],
        (gender == "female").astype(np.float32)[:, None],
        (~np.isin(gender, ["male", "female"])).astype(np.float32)[:, None],
        locale_frequency[:, None],
        location_frequency[:, None],
        country_frequency[:, None],
        lat[:, None],
        lng[:, None],
        np.isnan(lat).astype(np.float32)[:, None],
        np.isnan(lng).astype(np.float32)[:, None],
        seeds["event_missing"].to_numpy(dtype=np.float32)[:, None],
        content_nnz[:, None],
        np.log1p(content_norm)[:, None],
        content_embedding,
        np.eye(33, dtype=np.float32)[content_cluster],
        np.log1p(event_prior_count)[:, None],
        np.log1p(event_prior_users)[:, None],
        np.log1p(user_prior_count)[:, None],
        np.log1p(np.maximum(time_since_user, 0))[:, None],
        batch_size[:, None],
        batch_position[:, None],
        same_event_batch[:, None],
    ]
    direct_names = [
        "invited",
        "utc_hour_sin",
        "utc_hour_cos",
        "local_hour_sin",
        "local_hour_cos",
        "weekday_sin",
        "weekday_cos",
        "event_latency_sign",
        "event_latency_logabs",
        "future_event",
        "account_age_logdays",
        "age",
        "age_missing",
        "timezone",
        "gender_male",
        "gender_female",
        "gender_other",
        "locale_frequency",
        "location_frequency",
        "event_country_frequency",
        "event_lat",
        "event_lng",
        "event_lat_missing",
        "event_lng_missing",
        "event_id_missing",
        "content_nnz",
        "content_norm",
    ]
    direct_names += [f"event_content_svd_{index:02d}" for index in range(16)]
    direct_names += [f"event_content_cluster_{index:02d}" for index in range(33)]
    direct_names += [
        "event_prior_rows",
        "event_prior_users",
        "user_prior_rows",
        "hours_since_user_response",
        "user_timestamp_batch_size",
        "user_timestamp_batch_position",
        "event_timestamp_concurrency",
    ]
    direct_matrix = np.concatenate(direct_parts, axis=1).astype(np.float32)
    graph_rows = user_idx
    graph_static_parts = [
        np.log1p(graph.full_degree[graph_rows])[:, None],
        np.log1p(graph.out_degree[graph_rows])[:, None],
        np.log1p(graph.in_degree[graph_rows])[:, None],
        np.log1p(graph.und_degree[graph_rows])[:, None],
        np.divide(
            graph.out_degree[graph_rows],
            np.maximum(graph.full_degree[graph_rows], 1),
        )[:, None],
        np.log1p(graph.twohop_reach[graph_rows])[:, None],
        graph.pagerank[graph_rows, None],
        graph.clustering[graph_rows, None],
        np.log1p(graph.component_size[graph_rows])[:, None],
        np.log1p(graph.community_size[graph_rows])[:, None],
        graph.component[graph_rows, None].astype(np.float32),
        graph.community[graph_rows, None].astype(np.float32),
        graph.svd[graph_rows],
        graph.deepwalk[graph_rows],
    ]
    graph_names = [
        "friend_list_size_full",
        "resolved_out_degree",
        "resolved_in_degree",
        "resolved_undirected_degree",
        "resolved_fraction",
        "twohop_reach",
        "pagerank",
        "clustering_proxy",
        "component_size",
        "community_size",
        "component_id",
        "community_id",
    ]
    graph_names += [f"adjacency_svd_{index:02d}" for index in range(32)]
    graph_names += [f"deepwalk_{index:02d}" for index in range(32)]
    graph_static = np.concatenate(graph_static_parts, axis=1).astype(np.float32)
    attendees = full_db.table_dict["event_attendees"].df
    attendance_mask = attendees["user_id"].notna() & attendees["start_time"].notna()
    attendance = attendees.loc[attendance_mask, ["start_time", "user_id", "status"]]
    attendance_user = map_ids(attendance["user_id"].to_numpy(), graph.user_ids)
    status_map = {"invited": 0, "yes": 1, "maybe": 2, "no": 3}
    attendance_status = attendance["status"].map(status_map).fillna(-1).to_numpy(dtype=np.int8)
    attendance_time = attendance["start_time"].to_numpy(dtype="datetime64[ns]")
    valid_attendance = (attendance_user >= 0) & (attendance_status >= 0)
    order = np.argsort(attendance_time[valid_attendance], kind="mergesort")
    attendance_time = attendance_time[valid_attendance][order]
    attendance_user = attendance_user[valid_attendance][order]
    attendance_status = attendance_status[valid_attendance][order]
    print(
        f"[phase] direct_features rows={n} direct={direct_matrix.shape[1]} "
        f"static_graph={graph_static.shape[1]} seconds={time.time() - started:.1f}"
    )
    return DirectBundle(
        matrix=direct_matrix,
        names=direct_names,
        graph_static=graph_static,
        graph_names=graph_names,
        content_embedding=content_embedding,
        content_cluster=content_cluster,
        event_start=event_start,
        attendance_time=attendance_time,
        attendance_user=attendance_user,
        attendance_status=attendance_status,
    )


def _co_features(
    user: int,
    event_users: dict[int, set[int]],
    user_events: dict[int, set[int]],
    probability: np.ndarray,
    component: np.ndarray,
    community: np.ndarray,
) -> tuple[float, float, float, float, float]:
    weights: dict[int, float] = {}
    for event in user_events.get(user, ()):
        participants = sorted(event_users.get(event, ()))
        if len(participants) > 50:
            participants = [user] + [other for other in participants if other != user][:49]
        degree = len(participants)
        if degree <= 1:
            continue
        contribution = 1.0 / max(1, degree - 1)
        for other in participants:
            if other != user:
                weights[other] = weights.get(other, 0.0) + contribution
    if not weights:
        return 0.0, 0.0, 0.0, 0.0, 0.03
    neighbors = np.fromiter(weights.keys(), dtype=np.int32)
    values = np.fromiter(weights.values(), dtype=np.float32)
    degree = float(values.sum())
    same_component = float(values[component[neighbors] == component[user]].sum())
    same_community = float(values[community[neighbors] == community[user]].sum())
    propagated = float(np.dot(values, probability[neighbors]) / max(degree, 1e-8))
    return degree, float(len(neighbors)), same_component, same_community, propagated


def build_dynamic_bundle(
    seeds: pd.DataFrame,
    direct: DirectBundle,
    graph: GraphBundle,
    allowed_labels: np.ndarray,
    debug: bool,
) -> DynamicBundle:
    started = time.time()
    n_rows = len(seeds)
    n_users = len(graph.user_ids)
    n_clusters = 33
    content_dim = direct.content_embedding.shape[1]
    labels = seeds["label"].to_numpy(dtype=np.float32)
    user_idx = seeds["user_idx"].to_numpy(dtype=np.int32)
    event_ids = seeds["event_key"].to_numpy(dtype=np.int64)
    timestamps = seeds["timestamp"].to_numpy(dtype="datetime64[ns]")
    label_count = np.zeros(n_users, dtype=np.float32)
    positive_count = np.zeros(n_users, dtype=np.float32)
    cluster_count = np.zeros((n_users, n_clusters), dtype=np.float32)
    cluster_positive = np.zeros((n_users, n_clusters), dtype=np.float32)
    rejected_content = np.zeros((n_users, content_dim), dtype=np.float32)
    accepted_content = np.zeros((n_users, content_dim), dtype=np.float32)
    rejected_count = np.zeros(n_users, dtype=np.float32)
    accepted_count = np.zeros(n_users, dtype=np.float32)
    attendance_count = np.zeros((n_users, 4), dtype=np.float32)
    event_users: dict[int, set[int]] = {}
    user_events: dict[int, set[int]] = {}
    event_label_count: dict[int, int] = {}
    event_positive_count: dict[int, int] = {}
    names = [
        "self_label_count",
        "self_positive_count",
        "self_smoothed_rate",
        "friend_onehop_rate",
        "friend_twohop_rate",
        "friend_damped_rate",
        "friend_onehop_coverage",
        "friend_twohop_coverage",
        "friend_onehop_denominator",
        "friend_twohop_denominator",
        "friend_onehop_variance",
        "friend_twohop_variance",
        "friends_prior_event_count",
        "friends_prior_event_weight",
        "friends_prior_event_fraction",
        "friend_cluster_rejection_rate",
        "friend_cluster_coverage",
        "friend_cluster_denominator",
        "friend_rejected_content_cosine",
        "friend_accepted_content_cosine",
        "friend_rejected_content_count",
        "friend_accepted_content_count",
        "friend_attendance_invited",
        "friend_attendance_yes",
        "friend_attendance_maybe",
        "friend_attendance_no",
        "self_attendance_invited",
        "self_attendance_yes",
        "self_attendance_maybe",
        "self_attendance_no",
        "event_prior_label_count",
        "event_prior_rejection_rate",
        "coresponse_weighted_degree",
        "coresponse_unique_neighbors",
        "coresponse_same_component",
        "coresponse_same_community",
        "coresponse_rejection_rate",
        "event_prior_responders_snapshot",
        "event_same_component_snapshot",
        "event_same_community_snapshot",
        "event_community_diversity_snapshot",
    ]
    output = np.zeros((n_rows, len(names)), dtype=np.float32)
    cold_user = np.zeros(n_rows, dtype=bool)
    weight_sum = np.asarray(graph.normalized.sum(axis=1)).ravel().astype(np.float32)
    two_weight_sum = np.asarray(graph.normalized @ weight_sum).ravel().astype(np.float32)
    attendance_pointer = 0
    snapshot_ready = False
    ordered = seeds.sort_values(["timestamp", "row_id"], kind="mergesort")
    for day, day_frame in ordered.groupby(ordered["timestamp"].dt.floor("D"), sort=False):
        day_start = np.datetime64(day.to_datetime64(), "ns")
        while (
            attendance_pointer < len(direct.attendance_time)
            and direct.attendance_time[attendance_pointer] < day_start
        ):
            end = attendance_pointer + 1
            while (
                end < len(direct.attendance_time)
                and direct.attendance_time[end] < day_start
                and end - attendance_pointer < 65536
            ):
                end += 1
            np.add.at(
                attendance_count,
                (direct.attendance_user[attendance_pointer:end], direct.attendance_status[attendance_pointer:end]),
                1,
            )
            attendance_pointer = end
        if not snapshot_ready or not debug:
            probability = (positive_count + 50.0 * 0.03) / (label_count + 50.0)
            seen = (label_count > 0).astype(np.float32)
            one_raw = graph.normalized @ probability
            one_rate = np.divide(one_raw, weight_sum, out=np.full(n_users, 0.03, dtype=np.float32), where=weight_sum > 0)
            two_raw = graph.normalized @ one_raw
            two_rate = np.divide(two_raw, two_weight_sum, out=np.full(n_users, 0.03, dtype=np.float32), where=two_weight_sum > 0)
            one_coverage = np.divide(
                graph.normalized @ seen,
                weight_sum,
                out=np.zeros(n_users, dtype=np.float32),
                where=weight_sum > 0,
            )
            two_coverage = np.divide(
                graph.normalized @ (graph.normalized @ seen),
                two_weight_sum,
                out=np.zeros(n_users, dtype=np.float32),
                where=two_weight_sum > 0,
            )
            one_denominator = graph.normalized @ label_count
            two_denominator = graph.normalized @ (graph.normalized @ label_count)
            one_second = np.divide(
                graph.normalized @ (probability * probability),
                weight_sum,
                out=np.zeros(n_users, dtype=np.float32),
                where=weight_sum > 0,
            )
            two_second = np.divide(
                graph.normalized @ (graph.normalized @ (probability * probability)),
                two_weight_sum,
                out=np.zeros(n_users, dtype=np.float32),
                where=two_weight_sum > 0,
            )
            one_variance = np.maximum(one_second - one_rate * one_rate, 0)
            two_variance = np.maximum(two_second - two_rate * two_rate, 0)
            attendance_total = attendance_count.sum(axis=1, keepdims=True)
            attendance_propensity = (attendance_count + 1.0) / (attendance_total + 4.0)
            friend_attendance_raw = graph.normalized @ attendance_propensity
            friend_attendance = np.divide(
                friend_attendance_raw,
                weight_sum[:, None],
                out=np.full_like(friend_attendance_raw, 0.25),
                where=weight_sum[:, None] > 0,
            )
            snapshot_ready = True
        day_rows = day_frame["row_id"].to_numpy(dtype=np.int32)
        unique_users = np.unique(user_idx[day_rows])
        co_by_user = {
            int(user): _co_features(
                int(user), event_users, user_events, probability, graph.component, graph.community
            )
            for user in unique_users
        }
        event_snapshot: dict[int, tuple[float, float, float, float]] = {}
        for row in day_rows:
            user = int(user_idx[row])
            event = int(event_ids[row])
            participants = sorted(event_users.get(event, ()))
            if len(participants) > 50:
                participants = participants[:50]
            if participants:
                participant_array = np.asarray(participants, dtype=np.int32)
                same_component = float((graph.component[participant_array] == graph.component[user]).sum())
                same_community = float((graph.community[participant_array] == graph.community[user]).sum())
                diversity = float(np.unique(graph.community[participant_array]).size)
                event_snapshot[row] = (float(len(participants)), same_component, same_community, diversity)
            else:
                event_snapshot[row] = (0.0, 0.0, 0.0, 0.0)
        for _, batch in day_frame.groupby("timestamp", sort=False):
            rows = batch["row_id"].to_numpy(dtype=np.int32)
            for row in rows:
                user = int(user_idx[row])
                event = int(event_ids[row])
                cluster = int(direct.content_cluster[row])
                content = direct.content_embedding[row]
                cold_user[row] = label_count[user] == 0
                self_rate = (positive_count[user] + 50.0 * 0.03) / (label_count[user] + 50.0)
                begin = graph.normalized.indptr[user]
                end = graph.normalized.indptr[user + 1]
                neighbors = graph.normalized.indices[begin:end]
                weights = graph.normalized.data[begin:end]
                total_weight = float(weights.sum())
                prior_responders = event_users.get(event, set())
                directed_begin = graph.directed.indptr[user]
                directed_end = graph.directed.indptr[user + 1]
                directed_neighbors = graph.directed.indices[directed_begin:directed_end]
                friend_event_count = float(sum(int(neighbor in prior_responders) for neighbor in directed_neighbors))
                friend_event_weight = float(
                    sum(weight for neighbor, weight in zip(neighbors, weights) if neighbor in prior_responders)
                )
                friend_event_fraction = friend_event_count / max(len(directed_neighbors), 1)
                if len(neighbors):
                    cluster_rates = (
                        cluster_positive[neighbors, cluster] + 50.0 * 0.03
                    ) / (cluster_count[neighbors, cluster] + 50.0)
                    cluster_rate = float(np.dot(weights, cluster_rates) / max(total_weight, 1e-8))
                    cluster_coverage = float(
                        np.dot(weights, (cluster_count[neighbors, cluster] > 0).astype(np.float32))
                        / max(total_weight, 1e-8)
                    )
                    cluster_denominator = float(np.dot(weights, cluster_count[neighbors, cluster]))
                    rejected_total = (weights[:, None] * rejected_content[neighbors]).sum(axis=0)
                    accepted_total = (weights[:, None] * accepted_content[neighbors]).sum(axis=0)
                    rejected_weight = float(np.dot(weights, rejected_count[neighbors]))
                    accepted_weight = float(np.dot(weights, accepted_count[neighbors]))
                    rejected_cosine = _safe_cosine(content, rejected_total, rejected_weight)
                    accepted_cosine = _safe_cosine(content, accepted_total, accepted_weight)
                else:
                    cluster_rate = 0.03
                    cluster_coverage = 0.0
                    cluster_denominator = 0.0
                    rejected_weight = 0.0
                    accepted_weight = 0.0
                    rejected_cosine = 0.0
                    accepted_cosine = 0.0
                event_count = event_label_count.get(event, 0)
                event_positive = event_positive_count.get(event, 0)
                event_rate = (event_positive + 50.0 * 0.03) / (event_count + 50.0)
                co = co_by_user[user]
                snapshot = event_snapshot[row]
                output[row] = np.asarray(
                    [
                        np.log1p(label_count[user]),
                        np.log1p(positive_count[user]),
                        self_rate,
                        one_rate[user],
                        two_rate[user],
                        0.5 * one_rate[user] + 0.5 * two_rate[user],
                        one_coverage[user],
                        two_coverage[user],
                        np.log1p(one_denominator[user]),
                        np.log1p(two_denominator[user]),
                        one_variance[user],
                        two_variance[user],
                        np.log1p(friend_event_count),
                        friend_event_weight,
                        friend_event_fraction,
                        cluster_rate,
                        cluster_coverage,
                        np.log1p(cluster_denominator),
                        rejected_cosine,
                        accepted_cosine,
                        np.log1p(rejected_weight),
                        np.log1p(accepted_weight),
                        friend_attendance[user, 0],
                        friend_attendance[user, 1],
                        friend_attendance[user, 2],
                        friend_attendance[user, 3],
                        np.log1p(attendance_count[user, 0]),
                        np.log1p(attendance_count[user, 1]),
                        np.log1p(attendance_count[user, 2]),
                        np.log1p(attendance_count[user, 3]),
                        np.log1p(event_count),
                        event_rate,
                        np.log1p(co[0]),
                        np.log1p(co[1]),
                        co[2] / max(co[0], 1e-8),
                        co[3] / max(co[0], 1e-8),
                        co[4],
                        np.log1p(snapshot[0]),
                        np.log1p(snapshot[1]),
                        np.log1p(snapshot[2]),
                        np.log1p(snapshot[3]),
                    ],
                    dtype=np.float32,
                )
            for row in rows:
                user = int(user_idx[row])
                event = int(event_ids[row])
                event_users.setdefault(event, set()).add(user)
                user_events.setdefault(user, set()).add(event)
            for row in rows:
                if not allowed_labels[row] or not np.isfinite(labels[row]):
                    continue
                user = int(user_idx[row])
                event = int(event_ids[row])
                cluster = int(direct.content_cluster[row])
                label = float(labels[row])
                label_count[user] += 1
                positive_count[user] += label
                cluster_count[user, cluster] += 1
                cluster_positive[user, cluster] += label
                if label > 0.5:
                    rejected_content[user] += direct.content_embedding[row]
                    rejected_count[user] += 1
                else:
                    accepted_content[user] += direct.content_embedding[row]
                    accepted_count[user] += 1
                event_label_count[event] = event_label_count.get(event, 0) + 1
                event_positive_count[event] = event_positive_count.get(event, 0) + int(label > 0.5)
    print(
        f"[phase] dynamic_features allowed={int(allowed_labels.sum())} snapshots={'1' if debug else 'daily'} "
        f"rows={n_rows} features={len(names)} seconds={time.time() - started:.1f}"
    )
    return DynamicBundle(matrix=output, names=names, cold_user=cold_user)
