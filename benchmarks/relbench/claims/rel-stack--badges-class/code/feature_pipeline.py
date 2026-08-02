import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit


DAY_NS = 86_400_000_000_000
WINDOWS = (7, 30, 90, 365)


@njit(cache=True)
def _positions(event_times, starts, ends, query_users, query_times):
    result = np.full(len(query_users), -1, dtype=np.int64)
    for i in range(len(query_users)):
        user = query_users[i]
        if user < 0 or user >= len(starts):
            continue
        left = starts[user]
        right = ends[user]
        if left < 0:
            continue
        target = query_times[i]
        lo = left
        hi = right
        while lo < hi:
            mid = (lo + hi) // 2
            if event_times[mid] <= target:
                lo = mid + 1
            else:
                hi = mid
        if lo > left:
            result[i] = lo - 1
    return result


def _segments(users):
    maximum = int(users.max()) if len(users) else 0
    starts = np.full(maximum + 1, -1, dtype=np.int64)
    ends = np.full(maximum + 1, -1, dtype=np.int64)
    unique, first, counts = np.unique(users, return_index=True, return_counts=True)
    starts[unique] = first
    ends[unique] = first + counts
    return starts, ends


def _take(values, positions):
    output = np.zeros((len(positions), values.shape[1]), dtype=np.float32)
    valid = positions >= 0
    output[valid] = values[positions[valid]]
    return output


def _ns(series):
    return series.to_numpy(dtype="datetime64[ns]").astype(np.int64)


def _safe_text_length(series):
    return series.fillna("").astype(str).str.len().to_numpy(dtype=np.float32)


def _nearest_grid(values, grid):
    array = np.asarray(values, dtype=np.float32)
    points = np.asarray(grid, dtype=np.float32)
    distances = np.abs(array[:, None] - points[None, :])
    nearest = points[np.argmin(distances, axis=1)]
    return nearest - array, np.min(distances, axis=1)


class EventState:
    def __init__(self, frame, feature_columns):
        data = frame[["UserId", "Time", *feature_columns]].copy()
        data = data[data["UserId"].notna() & data["Time"].notna()]
        data["UserId"] = data["UserId"].astype(np.int64)
        data = data[data["UserId"] >= 0]
        data = data.groupby(["UserId", "Time"], as_index=False, sort=False)[feature_columns].sum()
        data.sort_values(["UserId", "Time"], inplace=True, kind="mergesort")
        data[feature_columns] = data.groupby("UserId", sort=False)[feature_columns].cumsum()
        self.users = data["UserId"].to_numpy(dtype=np.int64)
        self.times = _ns(data["Time"])
        self.values = data[feature_columns].to_numpy(dtype=np.float32)
        self.names = list(feature_columns)
        self.starts, self.ends = _segments(self.users)

    def query(self, seed_users, seed_times, prefix, windows=WINDOWS):
        users = np.asarray(seed_users, dtype=np.int64)
        times = np.asarray(seed_times, dtype=np.int64)
        current_positions = _positions(self.times, self.starts, self.ends, users, times)
        current = _take(self.values, current_positions)
        blocks = [current]
        names = [f"{prefix}_{name}_life" for name in self.names]
        for window in windows:
            prior_positions = _positions(
                self.times,
                self.starts,
                self.ends,
                users,
                times - window * DAY_NS,
            )
            blocks.append(current - _take(self.values, prior_positions))
            names.extend(f"{prefix}_{name}_{window}d" for name in self.names)
        return np.column_stack(blocks).astype(np.float32), names

    def last_ages(self, seed_users, seed_times, prefix, raw_frame, feature_columns):
        outputs = []
        names = []
        users = np.asarray(seed_users, dtype=np.int64)
        times = np.asarray(seed_times, dtype=np.int64)
        for column in feature_columns:
            subset = raw_frame.loc[raw_frame[column] > 0, ["UserId", "Time"]].dropna()
            subset = subset.sort_values(["UserId", "Time"], kind="mergesort")
            event_users = subset["UserId"].to_numpy(dtype=np.int64)
            event_times = _ns(subset["Time"])
            starts, ends = _segments(event_users)
            positions = _positions(event_times, starts, ends, users, times)
            age = np.full(len(users), 99999.0, dtype=np.float32)
            valid = positions >= 0
            age[valid] = np.maximum(0, times[valid] - event_times[positions[valid]]) / DAY_NS
            outputs.append(age)
            names.append(f"{prefix}_{column}_last_days")
        return np.column_stack(outputs).astype(np.float32), names


class SnapshotState:
    def __init__(self, frame, feature_columns):
        data = frame[["UserId", "Time", *feature_columns]].dropna(subset=["UserId", "Time"]).copy()
        data["UserId"] = data["UserId"].astype(np.int64)
        data.sort_values(["UserId", "Time"], inplace=True, kind="mergesort")
        data = data.groupby(["UserId", "Time"], as_index=False, sort=False).last()
        self.users = data["UserId"].to_numpy(dtype=np.int64)
        self.times = _ns(data["Time"])
        self.values = data[feature_columns].to_numpy(dtype=np.float32)
        self.names = list(feature_columns)
        self.starts, self.ends = _segments(self.users)

    def query(self, seed_users, seed_times, prefix):
        users = np.asarray(seed_users, dtype=np.int64)
        times = np.asarray(seed_times, dtype=np.int64)
        positions = _positions(self.times, self.starts, self.ends, users, times)
        return _take(self.values, positions), [f"{prefix}_{name}" for name in self.names]


class BadgeFeatureBuilder:
    def __init__(self, badges, users):
        self.badges = badges[["Id", "UserId", "Date"]].copy()
        self.badges["Id"] = self.badges["Id"].astype(np.int64)
        self.badges["UserId"] = self.badges["UserId"].astype(np.int64)
        self.badges["Date"] = pd.to_datetime(self.badges["Date"])
        self.users_table = users.copy()
        self._prepare_sequence()
        self._prepare_batches()

    def _prepare_sequence(self):
        sequence = self.badges.sort_values(["UserId", "Date", "Id"], kind="mergesort").reset_index(drop=True)
        self.sequence_ids = sequence["Id"].to_numpy(dtype=np.int64)
        self.sequence_users = sequence["UserId"].to_numpy(dtype=np.int64)
        self.sequence_times = _ns(sequence["Date"])
        self.sequence_starts, self.sequence_ends = _segments(self.sequence_users)
        self.sequence_order = sequence.groupby("UserId", sort=False).cumcount().to_numpy(dtype=np.int64)
        maximum_id = int(self.badges["Id"].max())
        self.sequence_position_by_id = np.full(maximum_id + 1, -1, dtype=np.int64)
        self.sequence_position_by_id[self.sequence_ids] = np.arange(len(sequence), dtype=np.int64)
        self.sequence_order_by_id = np.full(maximum_id + 1, -1, dtype=np.int64)
        self.sequence_order_by_id[self.sequence_ids] = self.sequence_order

    def _prepare_batches(self):
        frame = self.badges.copy()
        registration = self.users_table.set_index("Id")["CreationDate"]
        frame["registration"] = frame["UserId"].map(registration)
        frame["tenure_days"] = (
            (frame["Date"] - frame["registration"]).dt.total_seconds().div(86400).clip(lower=0).fillna(0)
        )
        frame["prior_order"] = self.sequence_order_by_id[frame["Id"].to_numpy(dtype=np.int64)]
        exact = frame.groupby("Date", sort=False).agg(
            batch_size=("Id", "size"),
            batch_users=("UserId", "nunique"),
            batch_min_id=("Id", "min"),
            batch_max_id=("Id", "max"),
            cohort_tenure_mean=("tenure_days", "mean"),
            cohort_tenure_std=("tenure_days", "std"),
            cohort_prior_mean=("prior_order", "mean"),
            cohort_prior_std=("prior_order", "std"),
        ).reset_index()
        exact["cohort_tenure_std"] = exact["cohort_tenure_std"].fillna(0)
        exact["cohort_prior_std"] = exact["cohort_prior_std"].fillna(0)
        exact["size_bin"] = np.minimum(12, np.floor(np.log2(exact["batch_size"]))).astype(np.int16)
        exact["hour"] = exact["Date"].dt.hour.astype(np.int16)
        exact["minute"] = exact["Date"].dt.minute.astype(np.int16)
        exact["second"] = exact["Date"].dt.second.astype(np.int16)
        exact["tod_key"] = exact["hour"].astype(np.int32) * 60 + exact["minute"].astype(np.int32)
        exact["job_key"] = exact["tod_key"].astype(np.int32) * 16 + exact["size_bin"].astype(np.int32)
        exact.sort_values("Date", inplace=True, kind="mergesort")
        exact["previous_batch_gap_minutes"] = exact["Date"].diff().dt.total_seconds().div(60).fillna(99999).clip(upper=99999)
        exact["previous_job_gap_days"] = (
            exact.groupby("job_key", sort=False)["Date"].diff().dt.total_seconds().div(86400).fillna(99999).clip(upper=99999)
        )
        exact["day"] = exact["Date"].dt.normalize()
        exact["site_badges_day_through"] = exact.groupby("day", sort=False)["batch_size"].cumsum()
        exact["batch_repeat_rate"] = 1.0 - exact["batch_users"] / exact["batch_size"]
        frame = frame.merge(exact, on="Date", how="left", validate="many_to_one")
        frame.sort_values(["Date", "Id"], inplace=True, kind="mergesort")
        frame["batch_rank"] = frame.groupby("Date", sort=False).cumcount()
        frame["batch_rank_fraction"] = frame["batch_rank"] / np.maximum(1, frame["batch_size"] - 1)
        frame["batch_user_multiplicity"] = frame.groupby(["Date", "UserId"], sort=False)["Id"].transform("size")
        frame["batch_user_rank"] = frame.groupby(["Date", "UserId"], sort=False).cumcount()
        frame["id_gap_previous"] = frame.groupby("Date", sort=False)["Id"].diff().fillna(0).clip(upper=100000)
        frame["id_gap_next"] = (-frame.groupby("Date", sort=False)["Id"].diff(-1)).fillna(0).clip(upper=100000)
        frame["day"] = frame["Date"].dt.normalize()
        user_exact = frame.groupby(["UserId", "Date", "day"], sort=False).size().rename("n").reset_index()
        user_exact.sort_values(["UserId", "Date"], inplace=True, kind="mergesort")
        user_exact["user_badges_day_through"] = user_exact.groupby(["UserId", "day"], sort=False)["n"].cumsum()
        frame = frame.merge(
            user_exact[["UserId", "Date", "user_badges_day_through"]],
            on=["UserId", "Date"],
            how="left",
            validate="many_to_one",
        )
        columns = [
            "Id", "batch_size", "batch_users", "batch_min_id", "batch_max_id",
            "cohort_tenure_mean", "cohort_tenure_std", "cohort_prior_mean", "cohort_prior_std",
            "size_bin", "tod_key", "job_key", "previous_batch_gap_minutes", "previous_job_gap_days",
            "site_badges_day_through", "batch_repeat_rate", "batch_rank", "batch_rank_fraction",
            "batch_user_multiplicity", "batch_user_rank", "id_gap_previous", "id_gap_next",
            "user_badges_day_through",
        ]
        self.batch_by_id = frame[columns].set_index("Id").sort_index()

    def _known_features(self, seeds, cutoffs, known_labels):
        known = known_labels[["Id", "Date", "Class"]].merge(
            self.badges[["Id", "UserId"]], on="Id", how="left", validate="one_to_one"
        )
        known.sort_values(["UserId", "Date", "Id"], inplace=True, kind="mergesort")
        known_users = known["UserId"].to_numpy(dtype=np.int64)
        known_times = _ns(known["Date"])
        known_classes = known["Class"].to_numpy(dtype=np.int8)
        starts, ends = _segments(known_users)
        query_users = seeds["UserId"].to_numpy(dtype=np.int64)
        cutoff_times = _ns(pd.Series(pd.to_datetime(cutoffs))) - 1
        positions = _positions(known_times, starts, ends, query_users, cutoff_times)
        output = {}
        valid = positions >= 0
        counts = np.zeros(len(seeds), dtype=np.float32)
        counts[valid] = positions[valid] - starts[query_users[valid]] + 1
        output["known_badge_count"] = counts
        for cls in range(3):
            class_frame = known.loc[known["Class"] == cls]
            class_users = class_frame["UserId"].to_numpy(dtype=np.int64)
            class_times = _ns(class_frame["Date"])
            class_starts, class_ends = _segments(class_users)
            class_positions = _positions(class_times, class_starts, class_ends, query_users, cutoff_times)
            class_valid = class_positions >= 0
            class_counts = np.zeros(len(seeds), dtype=np.float32)
            class_counts[class_valid] = class_positions[class_valid] - class_starts[query_users[class_valid]] + 1
            last_age = np.full(len(seeds), 99999.0, dtype=np.float32)
            last_age[class_valid] = (cutoff_times[class_valid] - class_times[class_positions[class_valid]]) / DAY_NS
            output[f"known_class_{cls}_count"] = class_counts
            output[f"known_class_{cls}_share"] = class_counts / np.maximum(1, counts)
            output[f"known_class_{cls}_last_days"] = np.maximum(0, last_age)
        for lag in (1, 2, 3):
            values = np.full(len(seeds), -1, dtype=np.float32)
            source = positions - (lag - 1)
            valid_source = valid & (source >= starts[np.clip(query_users, 0, len(starts) - 1)])
            values[valid_source] = known_classes[source[valid_source]]
            output[f"last_known_class_{lag}"] = values
            for cls in range(3):
                output[f"last_known_class_{lag}_is_{cls}"] = (values == cls).astype(np.float32)
        return pd.DataFrame(output)

    def _historical_jobs(self, seeds, cutoffs, known_labels):
        batch = self.batch_by_id.reset_index()[["Id", "job_key", "tod_key"]]
        known = known_labels[["Id", "Date", "Class"]].merge(batch, on="Id", how="left", validate="one_to_one")
        seed_batch = self.batch_by_id.reindex(seeds["Id"].to_numpy())
        result = pd.DataFrame(index=np.arange(len(seeds)))
        cutoff_series = pd.Series(pd.to_datetime(cutoffs)).reset_index(drop=True)
        for cutoff in cutoff_series.unique():
            mask = cutoff_series == cutoff
            history = known[known["Date"] < cutoff]
            for key in ("job_key", "tod_key"):
                table = history.groupby([key, "Class"], observed=True).size().unstack(fill_value=0)
                for cls in range(3):
                    if cls not in table:
                        table[cls] = 0
                total = table[[0, 1, 2]].sum(axis=1)
                keys = seed_batch.loc[mask.to_numpy(), key]
                for cls in range(3):
                    shares = (table[cls] / np.maximum(1, total)).reindex(keys.to_numpy()).fillna(0).to_numpy()
                    result.loc[mask.to_numpy(), f"historical_{key}_class_{cls}_share"] = shares
                result.loc[mask.to_numpy(), f"historical_{key}_count"] = total.reindex(keys.to_numpy()).fillna(0).to_numpy()
        return result.fillna(0).reset_index(drop=True)

    def transform(self, seed_frame, cutoffs, known_labels):
        seeds = seed_frame[["Id", "Date", "row_idx"]].copy().reset_index(drop=True)
        badge_users = self.badges.set_index("Id")["UserId"]
        seeds["UserId"] = seeds["Id"].map(badge_users).astype(np.int64)
        ids = seeds["Id"].to_numpy(dtype=np.int64)
        sequence_positions = self.sequence_position_by_id[ids]
        sequence_orders = self.sequence_order_by_id[ids]
        seed_times = _ns(seeds["Date"])
        users = seeds["UserId"].to_numpy(dtype=np.int64)
        current_right = _positions(
            self.sequence_times,
            self.sequence_starts,
            self.sequence_ends,
            users,
            seed_times,
        )
        start = self.sequence_starts[users]
        first_times = self.sequence_times[start]
        features = pd.DataFrame(index=np.arange(len(seeds)))
        features["prior_badge_count"] = sequence_orders.astype(np.float32)
        features["all_badges_through_time"] = (current_right - start + 1).astype(np.float32)
        features["days_since_first_badge"] = np.maximum(0, seed_times - first_times) / DAY_NS
        for lag in (1, 2, 5):
            source = sequence_positions - lag
            valid = source >= start
            values = np.full(len(seeds), 99999.0, dtype=np.float32)
            values[valid] = np.maximum(0, seed_times[valid] - self.sequence_times[source[valid]]) / DAY_NS
            features[f"badge_lag_{lag}_days"] = values
        for left_lag, right_lag in ((1, 2), (2, 3), (5, 6)):
            left_pos = sequence_positions - left_lag
            right_pos = sequence_positions - right_lag
            valid = right_pos >= start
            values = np.full(len(seeds), 99999.0, dtype=np.float32)
            values[valid] = np.maximum(0, self.sequence_times[left_pos[valid]] - self.sequence_times[right_pos[valid]]) / DAY_NS
            features[f"prior_badge_gap_{left_lag}_{right_lag}"] = values
        for window in (1, 7, 30, 90, 365):
            before = _positions(
                self.sequence_times,
                self.sequence_starts,
                self.sequence_ends,
                users,
                seed_times - window * DAY_NS,
            )
            count = current_right - np.maximum(before, start - 1) - 1
            features[f"badge_count_{window}d"] = np.maximum(0, count).astype(np.float32)
        known = self._known_features(seeds, cutoffs, known_labels)
        for column in known:
            features[column] = known[column].to_numpy(dtype=np.float32)
        features["unlabeled_prior_count"] = np.maximum(
            0,
            features["all_badges_through_time"] - 1 - features["known_badge_count"],
        )
        batch = self.batch_by_id.reindex(ids).reset_index(drop=True)
        for column in batch:
            if column not in {"job_key", "tod_key", "batch_min_id", "batch_max_id"}:
                features[f"batch_{column}"] = batch[column].to_numpy(dtype=np.float32)
        features["batch_id_span"] = (batch["batch_max_id"] - batch["batch_min_id"]).to_numpy(dtype=np.float32)
        historical = self._historical_jobs(seeds, cutoffs, known_labels)
        for column in historical:
            features[column] = historical[column].to_numpy(dtype=np.float32)
        users_table = self.users_table.set_index("Id").reindex(users)
        registration_times = _ns(users_table["CreationDate"].reset_index(drop=True))
        tenure_days = np.maximum(0, seed_times - registration_times) / DAY_NS
        completed_anniversaries = np.floor(tenure_days / 365.2425)
        anniversary_phase = tenure_days - completed_anniversaries * 365.2425
        features["account_tenure_days"] = tenure_days.astype(np.float32)
        features["completed_anniversaries"] = completed_anniversaries.astype(np.float32)
        features["distance_nearest_anniversary"] = np.minimum(anniversary_phase, 365.2425 - anniversary_phase).astype(np.float32)
        features["distance_next_anniversary"] = (365.2425 - anniversary_phase).astype(np.float32)
        for column in ("Location", "WebsiteUrl", "AboutMe", "DisplayName"):
            text = users_table[column].reset_index(drop=True)
            features[f"profile_{column}_present"] = text.notna().to_numpy(dtype=np.float32)
            features[f"profile_{column}_length"] = _safe_text_length(text)
        features["profile_account_present"] = users_table["AccountId"].notna().to_numpy(dtype=np.float32)
        dates = seeds["Date"]
        hour = dates.dt.hour.to_numpy() + dates.dt.minute.to_numpy() / 60 + dates.dt.second.to_numpy() / 3600
        weekday = dates.dt.weekday.to_numpy()
        month = dates.dt.month.to_numpy() - 1
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
        features["weekday_sin"] = np.sin(2 * np.pi * weekday / 7).astype(np.float32)
        features["weekday_cos"] = np.cos(2 * np.pi * weekday / 7).astype(np.float32)
        features["month_sin"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
        features["month_cos"] = np.cos(2 * np.pi * month / 12).astype(np.float32)
        features["hour_03_utc"] = (dates.dt.hour.to_numpy() == 3).astype(np.float32)
        features["exact_hour"] = ((dates.dt.minute.to_numpy() == 0) & (dates.dt.second.to_numpy() == 0)).astype(np.float32)
        features.replace([np.inf, -np.inf], 0, inplace=True)
        features.fillna(0, inplace=True)
        return features.astype(np.float32), seeds


class ActivityFeatureBuilder:
    def __init__(self, db):
        self.db = db
        self.posts = db.table_dict["posts"].df.copy()
        self.post_owner = self.posts.set_index("Id")["OwnerUserId"]
        self.post_type = self.posts.set_index("Id")["PostTypeId"]
        self.post_created = self.posts.set_index("Id")["CreationDate"]

    def _append_state(self, blocks, names, frame, columns, prefix, seed_users, seed_times, last_columns=()):
        raw = frame[["UserId", "Time", *columns]].copy()
        state = EventState(raw, columns)
        block, block_names = state.query(seed_users, seed_times, prefix)
        blocks.append(block)
        names.extend(block_names)
        if last_columns:
            last, last_names = state.last_ages(seed_users, seed_times, prefix, raw, last_columns)
            blocks.append(last)
            names.extend(last_names)
        del state, raw
        gc.collect()

    def _post_events(self):
        posts = self.posts
        event = pd.DataFrame({"UserId": posts["OwnerUserId"], "Time": posts["CreationDate"]})
        post_type = posts["PostTypeId"].to_numpy()
        event["posts"] = 1.0
        event["questions"] = (post_type == 1).astype(np.float32)
        event["answers"] = (post_type == 2).astype(np.float32)
        event["other_posts"] = ((post_type != 1) & (post_type != 2)).astype(np.float32)
        event["body_chars"] = _safe_text_length(posts["Body"])
        event["title_chars"] = _safe_text_length(posts["Title"])
        event["tag_chars"] = _safe_text_length(posts["Tags"])
        event["tag_mentions"] = posts["Tags"].fillna("").astype(str).str.count("<").to_numpy(dtype=np.float32)
        event["has_parent"] = posts["ParentId"].notna().to_numpy(dtype=np.float32)
        ordered = event[["UserId", "Time"]].copy()
        ordered["day"] = ordered["Time"].dt.normalize()
        event["active_post_days"] = (~ordered.duplicated(["UserId", "day"])).to_numpy(dtype=np.float32)
        question_times = posts.loc[posts["PostTypeId"] == 1, ["Id", "CreationDate"]].rename(
            columns={"Id": "ParentId", "CreationDate": "QuestionDate"}
        )
        answer = posts.loc[posts["PostTypeId"] == 2, ["Id", "ParentId", "CreationDate"]].merge(
            question_times, on="ParentId", how="left", validate="many_to_one"
        )
        age_hours = (answer["CreationDate"] - answer["QuestionDate"]).dt.total_seconds().div(3600)
        answer["answer_age_hours"] = age_hours.clip(lower=0, upper=24 * 365).fillna(0)
        answer["fast_answer_1h"] = (age_hours.between(0, 1)).astype(np.float32)
        answer["fast_answer_24h"] = (age_hours.between(0, 24)).astype(np.float32)
        answer.sort_values(["ParentId", "CreationDate", "Id"], inplace=True, kind="mergesort")
        answer["first_answer"] = (answer.groupby("ParentId", sort=False).cumcount() == 0).astype(np.float32)
        answer = answer.set_index("Id")
        for column in ("answer_age_hours", "fast_answer_1h", "fast_answer_24h", "first_answer"):
            event[column] = posts["Id"].map(answer[column]).fillna(0).to_numpy(dtype=np.float32)
        return event

    def _comment_events(self):
        comments = self.db.table_dict["comments"].df
        length = _safe_text_length(comments["Text"])
        authored = pd.DataFrame({
            "UserId": comments["UserId"], "Time": comments["CreationDate"],
            "comments_authored": 1.0, "comment_authored_chars": length,
            "comments_received": 0.0, "comment_received_chars": 0.0,
        })
        received = pd.DataFrame({
            "UserId": comments["PostId"].map(self.post_owner), "Time": comments["CreationDate"],
            "comments_authored": 0.0, "comment_authored_chars": 0.0,
            "comments_received": 1.0, "comment_received_chars": length,
        })
        return pd.concat([authored, received], ignore_index=True)

    def _history_events(self):
        history = self.db.table_dict["postHistory"].df
        types = history["PostHistoryTypeId"].to_numpy()
        base = {
            "history_total": np.ones(len(history), dtype=np.float32),
            "history_initial": np.isin(types, [1, 2, 3]).astype(np.float32),
            "history_edit": np.isin(types, [4, 5, 6, 7, 8, 9]).astype(np.float32),
            "history_moderation": np.isin(types, np.arange(10, 23)).astype(np.float32),
            "history_type_2": (types == 2).astype(np.float32),
            "history_type_3": (types == 3).astype(np.float32),
            "history_type_5": (types == 5).astype(np.float32),
            "history_type_6": (types == 6).astype(np.float32),
            "history_type_10": (types == 10).astype(np.float32),
        }
        authored = pd.DataFrame({"UserId": history["UserId"], "Time": history["CreationDate"]})
        received = pd.DataFrame({"UserId": history["PostId"].map(self.post_owner), "Time": history["CreationDate"]})
        for name, values in base.items():
            authored[f"authored_{name}"] = values
            authored[f"received_{name}"] = 0.0
            received[f"authored_{name}"] = 0.0
            received[f"received_{name}"] = values
        return pd.concat([authored, received], ignore_index=True)

    def _link_events(self):
        links = self.db.table_dict["postLinks"].df
        types = links["LinkTypeId"].to_numpy()
        outgoing = pd.DataFrame({"UserId": links["PostId"].map(self.post_owner), "Time": links["CreationDate"]})
        incoming = pd.DataFrame({"UserId": links["RelatedPostId"].map(self.post_owner), "Time": links["CreationDate"]})
        for direction, frame in (("out", outgoing), ("in", incoming)):
            frame[f"links_{direction}"] = 1.0
            frame[f"links_{direction}_type_1"] = (types == 1).astype(np.float32)
            frame[f"links_{direction}_type_3"] = (types == 3).astype(np.float32)
            other = incoming if direction == "out" else outgoing
            other[f"links_{direction}"] = 0.0
            other[f"links_{direction}_type_1"] = 0.0
            other[f"links_{direction}_type_3"] = 0.0
        return pd.concat([outgoing, incoming], ignore_index=True)

    def _vote_events(self):
        votes = self.db.table_dict["votes"].df
        post_ids = votes["PostId"]
        owners = post_ids.map(self.post_owner)
        types = votes["VoteTypeId"].to_numpy()
        ptypes = post_ids.map(self.post_type).fillna(0).to_numpy()
        created = post_ids.map(self.post_created)
        ages = (votes["CreationDate"] - created).dt.total_seconds().div(86400).fillna(99999).to_numpy()
        event = pd.DataFrame({"UserId": owners, "Time": votes["CreationDate"]})
        event["votes_received"] = 1.0
        for vote_type in (1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 15, 16):
            event[f"vote_type_{vote_type}"] = (types == vote_type).astype(np.float32)
        event["question_votes"] = (ptypes == 1).astype(np.float32)
        event["answer_votes"] = (ptypes == 2).astype(np.float32)
        event["question_upvotes"] = ((ptypes == 1) & (types == 2)).astype(np.float32)
        event["answer_upvotes"] = ((ptypes == 2) & (types == 2)).astype(np.float32)
        event["question_downvotes"] = ((ptypes == 1) & (types == 3)).astype(np.float32)
        event["answer_downvotes"] = ((ptypes == 2) & (types == 3)).astype(np.float32)
        event["score_proxy_delta"] = (types == 2).astype(np.float32) - (types == 3).astype(np.float32)
        event["young_post_vote_1d"] = ((ages >= 0) & (ages <= 1)).astype(np.float32)
        event["young_post_vote_7d"] = ((ages >= 0) & (ages <= 7)).astype(np.float32)
        event["young_post_vote_30d"] = ((ages >= 0) & (ages <= 30)).astype(np.float32)
        return event

    def _crossing_events(self):
        votes = self.db.table_dict["votes"].df[["Id", "PostId", "VoteTypeId", "CreationDate"]].copy()
        votes.sort_values(["CreationDate", "Id"], inplace=True, kind="mergesort")
        valid_posts = self.posts[["Id", "OwnerUserId", "PostTypeId", "ParentId"]].copy()
        post_codes = pd.Series(np.arange(len(valid_posts), dtype=np.int64), index=valid_posts["Id"])
        codes = votes["PostId"].map(post_codes).fillna(-1).to_numpy(dtype=np.int64)
        types = votes["VoteTypeId"].to_numpy(dtype=np.int16)
        grids = np.asarray([1, 2, 3, 5, 10, 20, 25, 40, 100, 400, 1000], dtype=np.int32)
        save_grids = np.asarray([25, 100], dtype=np.int32)
        crossed = _vote_crossings(codes, types, len(valid_posts), grids, save_grids)
        keep = crossed.sum(axis=1) > 0
        event = pd.DataFrame({
            "UserId": votes.loc[keep, "PostId"].map(self.post_owner).to_numpy(),
            "Time": votes.loc[keep, "CreationDate"].to_numpy(),
            "PostId": votes.loc[keep, "PostId"].to_numpy(),
        })
        selected = crossed[keep]
        columns = [f"score_cross_{value}" for value in grids] + [f"save_cross_{value}" for value in save_grids] + ["accepted_cross"]
        for i, column in enumerate(columns):
            event[column] = selected[:, i].astype(np.float32)
        event["score_cross"] = event[[f"score_cross_{value}" for value in grids[:9]]].sum(axis=1)
        event["save_cross"] = event[[f"save_cross_{value}" for value in save_grids]].sum(axis=1)
        event["cross_any"] = 1.0
        family = event.groupby(["UserId", "Time"], dropna=True)[["score_cross", "save_cross", "accepted_cross"]].transform("sum")
        event["cross_family_cofire"] = ((family > 0).sum(axis=1) >= 2).astype(np.float32)
        return event, columns

    def _tag_states(self, crossing_events):
        questions = self.posts.loc[self.posts["PostTypeId"] == 1, ["Id", "Tags"]].rename(
            columns={"Id": "ParentId", "Tags": "ParentTags"}
        )
        answers = self.posts.loc[
            (self.posts["PostTypeId"] == 2) & self.posts["OwnerUserId"].notna(),
            ["Id", "OwnerUserId", "ParentId", "CreationDate"],
        ].merge(questions, on="ParentId", how="left", validate="many_to_one")
        answers["tag"] = answers["ParentTags"].fillna("").astype(str).str.findall(r"<([^>]+)>")
        tags = answers.explode("tag").dropna(subset=["tag"]).copy()
        tags.rename(columns={"OwnerUserId": "UserId", "CreationDate": "Time"}, inplace=True)
        tags.sort_values(["UserId", "Time", "Id", "tag"], inplace=True, kind="mergesort")
        tags["tag_answer_count"] = tags.groupby(["UserId", "tag"], sort=False).cumcount() + 1
        count = tags["tag_answer_count"].to_numpy(dtype=np.float64)
        prior = count - 1
        delta = np.where(count > 0, count * np.log(count), 0) - np.where(prior > 0, prior * np.log(prior), 0)
        tags["sum_c_log_c"] = pd.Series(delta, index=tags.index).groupby(tags["UserId"], sort=False).cumsum()
        tags["tag_total"] = tags.groupby("UserId", sort=False).cumcount() + 1
        tags["tag_unique"] = (tags["tag_answer_count"] == 1).groupby(tags["UserId"], sort=False).cumsum()
        tags["tag_max_answers"] = tags.groupby("UserId", sort=False)["tag_answer_count"].cummax()
        tags["tag_entropy"] = np.log(tags["tag_total"]) - tags["sum_c_log_c"] / tags["tag_total"]
        tags["tag_top_share"] = tags["tag_max_answers"] / tags["tag_total"]
        tags["tag_answer_grid_cross"] = tags["tag_answer_count"].isin([20, 80, 200]).astype(np.float32)
        snapshot_columns = ["tag_total", "tag_unique", "tag_max_answers", "tag_entropy", "tag_top_share"]
        snapshot = tags.groupby(["UserId", "Time"], as_index=False, sort=False)[snapshot_columns].last()
        progress = tags.groupby(["UserId", "Time"], as_index=False, sort=False)[["tag_answer_grid_cross"]].sum()
        high_crosses = crossing_events[
            (crossing_events["score_cross_100"] > 0)
            | (crossing_events["score_cross_400"] > 0)
            | (crossing_events["score_cross_1000"] > 0)
        ].merge(
            tags[["Id", "UserId", "tag"]].drop_duplicates(),
            left_on=["PostId", "UserId"], right_on=["Id", "UserId"], how="inner",
        )
        score_progress_parts = []
        for threshold in (100, 400, 1000):
            column = f"score_cross_{threshold}"
            subset = high_crosses[high_crosses[column] > 0].sort_values("Time")
            subset = subset.drop_duplicates(["UserId", "tag"])
            if len(subset):
                score_progress_parts.append(pd.DataFrame({
                    "UserId": subset["UserId"], "Time": subset["Time"],
                    "tag_score_grid_cross": np.ones(len(subset), dtype=np.float32),
                }))
        if score_progress_parts:
            score_progress = pd.concat(score_progress_parts, ignore_index=True)
        else:
            score_progress = pd.DataFrame(columns=["UserId", "Time", "tag_score_grid_cross"])
        return snapshot, progress, score_progress

    def build(self, seed_frame):
        start = time.time()
        seeds = seed_frame.reset_index(drop=True)
        badge_users = self.db.table_dict["badges"].df.set_index("Id")["UserId"]
        seed_users = seeds["Id"].map(badge_users).to_numpy(dtype=np.int64)
        seed_times = _ns(seeds["Date"])
        blocks = []
        names = []
        post_events = self._post_events()
        self._append_state(blocks, names, post_events, list(post_events.columns[2:]), "post", seed_users, seed_times)
        print(f"[features] posts complete elapsed={time.time() - start:.1f}s")
        del post_events
        comment_events = self._comment_events()
        self._append_state(blocks, names, comment_events, list(comment_events.columns[2:]), "comment", seed_users, seed_times)
        print(f"[features] comments complete elapsed={time.time() - start:.1f}s")
        del comment_events
        history_events = self._history_events()
        self._append_state(blocks, names, history_events, list(history_events.columns[2:]), "history", seed_users, seed_times)
        print(f"[features] postHistory complete elapsed={time.time() - start:.1f}s")
        del history_events
        link_events = self._link_events()
        self._append_state(blocks, names, link_events, list(link_events.columns[2:]), "link", seed_users, seed_times)
        print(f"[features] postLinks complete elapsed={time.time() - start:.1f}s")
        del link_events
        vote_events = self._vote_events()
        self._append_state(blocks, names, vote_events, list(vote_events.columns[2:]), "vote", seed_users, seed_times)
        print(f"[features] votes complete elapsed={time.time() - start:.1f}s")
        del vote_events
        crossing_events, crossing_columns = self._crossing_events()
        cross_features = [*crossing_columns, "score_cross", "save_cross", "cross_any", "cross_family_cofire"]
        self._append_state(
            blocks, names, crossing_events, cross_features, "trigger", seed_users, seed_times,
            last_columns=("score_cross", "save_cross", "accepted_cross", "cross_any"),
        )
        print(f"[features] crossings complete elapsed={time.time() - start:.1f}s")
        snapshot, tag_progress, tag_score_progress = self._tag_states(crossing_events)
        snapshot_state = SnapshotState(snapshot, ["tag_total", "tag_unique", "tag_max_answers", "tag_entropy", "tag_top_share"])
        tag_block, tag_names = snapshot_state.query(seed_users, seed_times, "tag")
        blocks.append(tag_block)
        names.extend(tag_names)
        if len(tag_progress):
            self._append_state(blocks, names, tag_progress, ["tag_answer_grid_cross"], "tag", seed_users, seed_times)
        if len(tag_score_progress):
            self._append_state(blocks, names, tag_score_progress, ["tag_score_grid_cross"], "tag", seed_users, seed_times)
        matrix = np.column_stack(blocks).astype(np.float32)
        frame = pd.DataFrame(matrix, columns=names)
        for prefix, count_column in (
            ("post", "post_posts_life"),
            ("comment", "comment_comments_authored_life"),
            ("history", "history_authored_history_total_life"),
            ("link", "link_links_out_life"),
            ("vote", "vote_votes_received_life"),
        ):
            if count_column in frame:
                life = frame[count_column].to_numpy()
                recent_name = count_column.replace("_life", "_90d")
                if recent_name in frame:
                    ratio = frame[recent_name].to_numpy() / np.maximum(1, life / 4)
                    frame[f"{prefix}_recent_lifetime_rate_ratio"] = np.clip(ratio, 0, 20).astype(np.float32)
        if "vote_answer_upvotes_life" in frame and "post_answers_life" in frame:
            frame["ratio_answer_upvotes_per_answer"] = (
                frame["vote_answer_upvotes_life"] / np.maximum(1, frame["post_answers_life"])
            ).astype(np.float32)
        if "comment_comments_received_life" in frame and "post_posts_life" in frame:
            frame["ratio_comments_received_per_post"] = (
                frame["comment_comments_received_life"] / np.maximum(1, frame["post_posts_life"])
            ).astype(np.float32)
        diversity_sources = [
            column for column in (
                "post_posts_90d", "comment_comments_authored_90d", "comment_comments_received_90d",
                "history_authored_history_total_90d", "link_links_out_90d", "vote_votes_received_90d",
            ) if column in frame
        ]
        frame["activity_diversity_90d"] = (frame[diversity_sources].to_numpy() > 0).sum(axis=1).astype(np.float32)
        for column, grid in (
            ("post_posts_life", [1, 5, 10, 20, 50, 100, 200, 500, 1000]),
            ("post_answers_life", [1, 10, 20, 50, 80, 100, 200, 500]),
            ("comment_comments_authored_life", [1, 10, 50, 100, 500, 1000]),
            ("history_authored_history_edit_life", [1, 10, 50, 100, 500]),
            ("tag_tag_max_answers", [20, 80, 200]),
        ):
            if column in frame:
                signed, absolute = _nearest_grid(frame[column].to_numpy(), grid)
                frame[f"milestone_{column}_signed"] = signed
                frame[f"milestone_{column}_absolute"] = absolute
        frame.replace([np.inf, -np.inf], 0, inplace=True)
        frame.fillna(0, inplace=True)
        print(f"[features] activity matrix rows={len(frame)} cols={len(frame.columns)} elapsed={time.time() - start:.1f}s")
        return frame.astype(np.float32)


@njit(cache=True)
def _vote_crossings(post_codes, vote_types, number_posts, score_grid, save_grid):
    width = len(score_grid) + len(save_grid) + 1
    output = np.zeros((len(post_codes), width), dtype=np.uint8)
    scores = np.zeros(number_posts, dtype=np.int32)
    saves = np.zeros(number_posts, dtype=np.int32)
    score_masks = np.zeros(number_posts, dtype=np.uint16)
    save_masks = np.zeros(number_posts, dtype=np.uint8)
    accepted = np.zeros(number_posts, dtype=np.uint8)
    for i in range(len(post_codes)):
        post = post_codes[i]
        if post < 0:
            continue
        vote_type = vote_types[i]
        if vote_type == 2:
            scores[post] += 1
            for j in range(len(score_grid)):
                bit = np.uint16(1 << j)
                if scores[post] >= score_grid[j] and score_masks[post] & bit == 0:
                    output[i, j] = 1
                    score_masks[post] |= bit
        elif vote_type == 3:
            scores[post] -= 1
        elif vote_type == 5:
            saves[post] += 1
            for j in range(len(save_grid)):
                bit = np.uint8(1 << j)
                if saves[post] >= save_grid[j] and save_masks[post] & bit == 0:
                    output[i, len(score_grid) + j] = 1
                    save_masks[post] |= bit
        elif vote_type == 1 and accepted[post] == 0:
            output[i, width - 1] = 1
            accepted[post] = 1
    return output


def feature_cache_key(seed_frame, version="trigger_user_state_lane0_v7"):
    digest = hashlib.sha256()
    digest.update(np.asarray(seed_frame["Id"], dtype=np.int64).tobytes())
    digest.update(_ns(seed_frame["Date"]).tobytes())
    return f"{version}_{digest.hexdigest()[:16]}"


def load_or_build_activity(db, seed_frame, shared_dir):
    key = feature_cache_key(seed_frame)
    cache_dir = Path(shared_dir) / key
    matrix_path = cache_dir / "activity_features.npy"
    names_path = cache_dir / "feature_names.json"
    ids_path = cache_dir / "seed_ids.npy"
    expected_ids = seed_frame["Id"].to_numpy(dtype=np.int64)
    if matrix_path.exists() and names_path.exists() and ids_path.exists():
        cached_ids = np.load(ids_path, allow_pickle=False)
        if np.array_equal(cached_ids, expected_ids):
            matrix = np.load(matrix_path, allow_pickle=False)
            names = json.loads(names_path.read_text())
            print(f"[cache] loaded activity matrix key={key} shape={matrix.shape}")
            return pd.DataFrame(matrix, columns=names), key
    builder = ActivityFeatureBuilder(db)
    frame = builder.build(seed_frame)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_matrix = cache_dir / f"activity_features.{os.getpid()}.tmp.npy"
    temporary_ids = cache_dir / f"seed_ids.{os.getpid()}.tmp.npy"
    np.save(temporary_matrix, frame.to_numpy(dtype=np.float32))
    np.save(temporary_ids, expected_ids)
    os.replace(temporary_matrix, matrix_path)
    os.replace(temporary_ids, ids_path)
    names_path.write_text(json.dumps(list(frame.columns)))
    return frame, key

