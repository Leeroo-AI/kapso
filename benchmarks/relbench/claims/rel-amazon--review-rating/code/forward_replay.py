import gc
import time
from dataclasses import dataclass

import numpy as np
import torch

from relational_features import hierarchical_components, hierarchical_prediction


FEATURE_NAMES = [
    "global",
    "user_bias",
    "item_bias",
    "user_drift",
    "item_drift",
    "recency",
    "latent_dot",
    "implicit_dot",
    "raw_cf",
    "user_factor_norm",
    "item_factor_norm",
    "history_factor_norm",
    "content_item_disagreement",
    "content_implicit_disagreement",
    "log_user_count",
    "log_item_count",
    "log_user_staleness",
    "log_item_staleness",
    "cold_user",
    "cold_item",
    "repeat_pair",
    "hierarchical_prior",
    "verified",
    "log_price",
    "log_category_frequency",
    "log_brand_frequency",
    "log_title_frequency",
    "title_length",
    "description_length",
    "category_depth",
    "customer_name_length",
    "customer_name_tokens",
    "customer_name_frequency",
    "customer_name_digit",
    "elapsed_years",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
    "hierarchical_global",
    "hierarchical_user",
    "hierarchical_item",
    "hierarchical_category",
    "hierarchical_brand",
    "hierarchical_verified",
    "hierarchical_user_recent",
    "hierarchical_item_recent",
    "hierarchical_category_recent",
    "hierarchical_brand_recent",
    "hierarchical_price",
    "hierarchical_category_verified",
    "log_labeled_user_count",
    "log_labeled_item_count",
    "mapped_item_bias",
    "content_bias_disagreement",
    "hierarchical_user_verified",
    "hierarchical_item_verified",
    "hierarchical_user_category",
    "log_user_category_count",
    "hierarchical_user_brand",
    "log_user_brand_count",
    "hierarchical_top_category",
    "hierarchical_title",
    "log_title_count",
    "hierarchical_customer_name",
    "log_customer_name_count",
    "hierarchical_customer_morphology",
    "user_last_rating",
    "item_last_rating",
    "user_recent3_rating",
    "item_recent3_rating",
    "log_user_label_staleness",
    "log_item_label_staleness",
    "user_recent3_count",
    "item_recent3_count",
    "user_recent10_rating",
    "item_recent10_rating",
    "user_recent10_count",
    "item_recent10_count",
    "user_rating_std",
    "item_rating_std",
    "user_low_rating_rate",
    "item_low_rating_rate",
    "user_five_rating_rate",
    "item_five_rating_rate",
    "user_verified_rate",
    "item_verified_rate",
    "hierarchical_user_item",
    "log_user_item_count",
    "log_user_count_90d",
    "log_item_count_90d",
    "log_user_count_365d",
    "log_item_count_365d",
    "log_user_count_730d",
    "log_item_count_730d",
    "user_recent_activity_ratio",
    "item_recent_activity_ratio",
    "log_user_label_age",
    "log_item_label_age",
    "log_user_label_span",
    "log_item_label_span",
] + [f"content_pca_{index:02d}" for index in range(32)]


@dataclass
class ReplayResult:
    raw: np.ndarray
    residual: np.ndarray
    prior: np.ndarray
    prediction: np.ndarray
    features: np.ndarray | None
    frozen_raw: np.ndarray | None


def _to_device(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def replay_forward(
    state,
    content,
    hierarchy,
    metadata,
    customer,
    review_arrays,
    fit_event_ids,
    replay_event_ids,
    collect_ids,
    pair_repeat,
    booster=None,
    blend=None,
    keep_features=True,
    measure_frozen=False,
    history_review_arrays=None,
):
    started = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_users = len(state.user_bias)
    n_items = len(state.item_bias)
    factors = state.user_factor.shape[1]
    user_factor = _to_device(state.user_factor, device)
    item_factor = _to_device(content.item_factor, device)
    implicit_factor = _to_device(content.implicit_factor, device)
    mapped_item_factor = _to_device(content.mapped_item_factor, device)
    mapped_implicit_factor = _to_device(content.mapped_implicit_factor, device)
    user_bias = _to_device(state.user_bias, device)
    item_bias = _to_device(content.item_bias, device)
    user_drift = _to_device(state.user_drift, device)
    item_drift = _to_device(state.item_drift, device)
    user_mean_day = _to_device(state.user_mean_day, device)
    item_mean_day = _to_device(state.item_mean_day, device)
    history_state = torch.zeros((n_users, factors), dtype=torch.float32, device=device)
    users_all = review_arrays["user"]
    items_all = review_arrays["item"]
    days_all = review_arrays["day"]
    verified_all = review_arrays["verified"]
    history_review_arrays = review_arrays if history_review_arrays is None else history_review_arrays
    history_users = history_review_arrays["user"]
    history_items = history_review_arrays["item"]
    history_days = history_review_arrays["day"]
    for begin in range(0, len(fit_event_ids), 262144):
        ids = fit_event_ids[begin : begin + 262144]
        u = torch.as_tensor(history_users[ids], dtype=torch.long, device=device)
        i = torch.as_tensor(history_items[ids], dtype=torch.long, device=device)
        history_state.index_add_(0, u, implicit_factor[i])
    frozen_history_state = history_state.clone() if measure_frozen else None
    user_count = np.bincount(history_users[fit_event_ids], minlength=n_users).astype(np.int32)
    item_count = np.bincount(history_items[fit_event_ids], minlength=n_items).astype(np.int32)
    frozen_user_count = user_count.copy() if measure_frozen else None
    last_user_day = np.full(n_users, -1, dtype=np.int32)
    last_item_day = np.full(n_items, -1, dtype=np.int32)
    np.maximum.at(last_user_day, history_users[fit_event_ids], history_days[fit_event_ids])
    np.maximum.at(last_item_day, history_items[fit_event_ids], history_days[fit_event_ids])
    collect_position = np.full(len(users_all), -1, dtype=np.int32)
    collect_position[collect_ids] = np.arange(len(collect_ids), dtype=np.int32)
    raw_output = np.empty(len(collect_ids), dtype=np.float32)
    residual_output = np.empty(len(collect_ids), dtype=np.float32)
    prior_output = np.empty(len(collect_ids), dtype=np.float32)
    prediction_output = np.empty(len(collect_ids), dtype=np.float32)
    feature_output = np.empty((len(collect_ids), len(FEATURE_NAMES)), dtype=np.float32) if keep_features else None
    frozen_output = np.empty(len(collect_ids), dtype=np.float32) if measure_frozen else None
    replay_days = days_all[replay_event_ids]
    boundaries = np.flatnonzero(np.r_[True, replay_days[1:] != replay_days[:-1], True])
    for boundary in range(len(boundaries) - 1):
        ids_all = replay_event_ids[boundaries[boundary] : boundaries[boundary + 1]]
        day_value = int(days_all[ids_all[0]])
        positions_all = collect_position[ids_all]
        collect_mask = positions_all >= 0
        if np.any(collect_mask):
            ids = ids_all[collect_mask]
            positions = positions_all[collect_mask]
            users = users_all[ids]
            items = items_all[ids]
            days = days_all[ids].astype(np.float32)
            verified = verified_all[ids].astype(np.int8)
            u = torch.as_tensor(users, dtype=torch.long, device=device)
            i = torch.as_tensor(items, dtype=torch.long, device=device)
            day = torch.as_tensor(days, dtype=torch.float32, device=device)
            current_user_count = user_count[users]
            current_item_count = item_count[items]
            history = history_state[u] / torch.as_tensor(np.sqrt(np.maximum(current_user_count, 1))[:, None], dtype=torch.float32, device=device)
            p = user_factor[u]
            q = item_factor[i]
            dev_user = torch.sign(day - user_mean_day[u]) * torch.abs(day - user_mean_day[u]).pow(0.4)
            dev_item = torch.sign(day - item_mean_day[i]) * torch.abs(day - item_mean_day[i]).pow(0.4)
            user_term = user_bias[u]
            item_term = item_bias[i]
            user_drift_term = user_drift[u] * dev_user
            item_drift_term = item_drift[i] * dev_item
            user_stale = np.where(current_user_count > 0, day_value - last_user_day[users], 0).astype(np.float32)
            item_stale = np.where(current_item_count > 0, day_value - last_item_day[items], 0).astype(np.float32)
            recency_term = state.recency_coef[0] * np.log1p(user_stale) / 10.0 + state.recency_coef[1] * np.log1p(item_stale) / 10.0
            latent_term = (q * p).sum(1)
            implicit_term = (q * history).sum(1)
            if measure_frozen:
                frozen_history = frozen_history_state[u] / torch.as_tensor(np.sqrt(np.maximum(frozen_user_count[users], 1))[:, None], dtype=torch.float32, device=device)
                frozen_implicit_term = (q * frozen_history).sum(1)
            raw = (
                state.mu
                + user_term
                + item_term
                + user_drift_term
                + item_drift_term
                + torch.as_tensor(recency_term, dtype=torch.float32, device=device)
                + latent_term
                + implicit_term
            )
            components = torch.stack(
                [
                    torch.full_like(raw, state.mu),
                    user_term,
                    item_term,
                    user_drift_term,
                    item_drift_term,
                    torch.as_tensor(recency_term, dtype=torch.float32, device=device),
                    latent_term,
                    implicit_term,
                    raw,
                    torch.linalg.vector_norm(p, dim=1),
                    torch.linalg.vector_norm(q, dim=1),
                    torch.linalg.vector_norm(history, dim=1),
                    torch.linalg.vector_norm(q - mapped_item_factor[i], dim=1),
                    torch.linalg.vector_norm(implicit_factor[i] - mapped_implicit_factor[i], dim=1),
                ],
                dim=1,
            ).detach().cpu().numpy()
            prior = hierarchical_prediction(hierarchy, users, items, days, verified, metadata, customer)
            hierarchy_terms = hierarchical_components(hierarchy, users, items, days, verified, metadata, customer)
            month = review_arrays["month"][ids].astype(np.float32)
            weekday = ((days.astype(np.int32) + 1) % 7).astype(np.float32)
            product_features = np.column_stack(
                [
                    np.log1p(current_user_count),
                    np.log1p(current_item_count),
                    np.log1p(user_stale),
                    np.log1p(item_stale),
                    (state.user_count[users] == 0).astype(np.float32),
                    (state.item_count[items] == 0).astype(np.float32),
                    pair_repeat[ids].astype(np.float32),
                    prior,
                    verified,
                    metadata.log_price[items],
                    np.log1p(metadata.category_frequency[items]),
                    np.log1p(metadata.brand_frequency[items]),
                    np.log1p(metadata.title_frequency[items]),
                    np.log1p(metadata.title_length[items]),
                    np.log1p(metadata.description_length[items]),
                    metadata.category_depth[items],
                    np.log1p(customer.name_length[users]),
                    np.log1p(customer.name_tokens[users]),
                    np.log1p(customer.name_frequency[users]),
                    customer.digit_flag[users],
                    days / 365.25,
                    np.sin(2 * np.pi * month / 12),
                    np.cos(2 * np.pi * month / 12),
                    np.sin(2 * np.pi * weekday / 7),
                    np.cos(2 * np.pi * weekday / 7),
                    hierarchy_terms["global"],
                    hierarchy_terms["user"],
                    hierarchy_terms["item"],
                    hierarchy_terms["category"],
                    hierarchy_terms["brand"],
                    hierarchy_terms["verified"],
                    hierarchy_terms["user_recent"],
                    hierarchy_terms["item_recent"],
                    hierarchy_terms["category_recent"],
                    hierarchy_terms["brand_recent"],
                    hierarchy_terms["price"],
                    hierarchy_terms["category_verified"],
                    np.log1p(state.user_count[users]),
                    np.log1p(state.item_count[items]),
                    content.mapped_item_bias[items],
                    np.abs(content.item_bias[items] - content.mapped_item_bias[items]),
                    hierarchy_terms["user_verified"],
                    hierarchy_terms["item_verified"],
                    hierarchy_terms["user_category"],
                    np.log1p(hierarchy_terms["user_category_count"]),
                    hierarchy_terms["user_brand"],
                    np.log1p(hierarchy_terms["user_brand_count"]),
                    hierarchy_terms["top_category"],
                    hierarchy_terms["title"],
                    np.log1p(hierarchy_terms["title_count"]),
                    hierarchy_terms["customer_name"],
                    np.log1p(hierarchy_terms["customer_name_count"]),
                    hierarchy_terms["customer_morphology"],
                    hierarchy_terms["user_last_rating"],
                    hierarchy_terms["item_last_rating"],
                    hierarchy_terms["user_recent3_rating"],
                    hierarchy_terms["item_recent3_rating"],
                    np.log1p(hierarchy_terms["user_label_staleness"]),
                    np.log1p(hierarchy_terms["item_label_staleness"]),
                    hierarchy_terms["user_recent3_count"],
                    hierarchy_terms["item_recent3_count"],
                    hierarchy_terms["user_recent10_rating"],
                    hierarchy_terms["item_recent10_rating"],
                    hierarchy_terms["user_recent10_count"],
                    hierarchy_terms["item_recent10_count"],
                    hierarchy_terms["user_rating_std"],
                    hierarchy_terms["item_rating_std"],
                    hierarchy_terms["user_low_rating_rate"],
                    hierarchy_terms["item_low_rating_rate"],
                    hierarchy_terms["user_five_rating_rate"],
                    hierarchy_terms["item_five_rating_rate"],
                    hierarchy_terms["user_verified_rate"],
                    hierarchy_terms["item_verified_rate"],
                    hierarchy_terms["user_item"],
                    np.log1p(hierarchy_terms["user_item_count"]),
                    np.log1p(hierarchy_terms["user_count_90d"]),
                    np.log1p(hierarchy_terms["item_count_90d"]),
                    np.log1p(hierarchy_terms["user_count_365d"]),
                    np.log1p(hierarchy_terms["item_count_365d"]),
                    np.log1p(hierarchy_terms["user_count_730d"]),
                    np.log1p(hierarchy_terms["item_count_730d"]),
                    hierarchy_terms["user_count_90d"] / np.maximum(state.user_count[users], 1),
                    hierarchy_terms["item_count_90d"] / np.maximum(state.item_count[items], 1),
                    np.log1p(hierarchy_terms["user_label_age"]),
                    np.log1p(hierarchy_terms["item_label_age"]),
                    np.log1p(hierarchy_terms["user_label_span"]),
                    np.log1p(hierarchy_terms["item_label_span"]),
                    metadata.matrix[items, :32],
                ]
            ).astype(np.float32)
            features = np.column_stack([components, product_features]).astype(np.float32)
            if features.shape[1] != len(FEATURE_NAMES):
                raise RuntimeError(f"feature width {features.shape[1]} != {len(FEATURE_NAMES)}")
            raw_np = components[:, 8]
            residual_np = raw_np.copy() if booster is None else raw_np + booster.inplace_predict(features).astype(np.float32)
            if blend is None:
                prediction = residual_np
            else:
                prediction = blend[0] * raw_np + blend[1] * residual_np + blend[2] * prior
            raw_output[positions] = raw_np
            residual_output[positions] = residual_np
            prior_output[positions] = prior
            prediction_output[positions] = prediction
            if keep_features:
                feature_output[positions] = features
            if measure_frozen:
                frozen_output[positions] = raw_np - components[:, 7] + frozen_implicit_term.detach().cpu().numpy()
        u_all = torch.as_tensor(users_all[ids_all], dtype=torch.long, device=device)
        i_all = torch.as_tensor(items_all[ids_all], dtype=torch.long, device=device)
        history_state.index_add_(0, u_all, implicit_factor[i_all])
        np.add.at(user_count, users_all[ids_all], 1)
        np.add.at(item_count, items_all[ids_all], 1)
        last_user_day[users_all[ids_all]] = day_value
        last_item_day[items_all[ids_all]] = day_value
    print(f"[replay] fit={len(fit_event_ids)} replay={len(replay_event_ids)} collect={len(collect_ids)} elapsed={time.time() - started:.1f}s", flush=True)
    del user_factor, item_factor, implicit_factor, mapped_item_factor, mapped_implicit_factor, history_state
    if frozen_history_state is not None:
        del frozen_history_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ReplayResult(raw_output, residual_output, prior_output, prediction_output, feature_output, frozen_output)
