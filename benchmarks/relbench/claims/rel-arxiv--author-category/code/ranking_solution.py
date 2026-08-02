import gc
import math
import time
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


def as_days(values):
    return values.to_numpy(dtype="datetime64[D]").astype(np.int32)


def normalized(matrix):
    denominator = matrix.sum(axis=1, keepdims=True)
    return matrix / np.maximum(denominator, 1e-7)


def softmax(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)
    values = np.exp(np.clip(shifted, -50.0, 0.0))
    return values / values.sum(axis=1, keepdims=True)


def class_distribution(labels, classes):
    counts = np.bincount(labels.astype(np.int64), minlength=classes).astype(np.float64) + 1.0
    return counts / counts.sum()


class RelationalStore:
    def __init__(self, context):
        self.context = context
        self.classes = int(context.task.num_classes)
        authors = context.db.table_dict["authors"].df
        papers = context.db.table_dict["papers"].df
        paper_authors = context.db.table_dict["paperAuthors"].df
        categories = context.db.table_dict["categories"].df
        self.authors = int(authors["Author_ID"].max()) + 1
        self.papers = int(papers["Paper_ID"].max()) + 1
        category_map = np.zeros(int(categories["Category_ID"].max()) + 1, dtype=np.int16)
        category_map[categories["Category_ID"].to_numpy(np.int64)] = categories["Category"].to_numpy(np.int16)
        self.paper_category = np.zeros(self.papers, dtype=np.int16)
        self.paper_category[papers["Paper_ID"].to_numpy(np.int64)] = category_map[papers["Primary_Category_ID"].to_numpy(np.int64)]
        self.paper_day = np.zeros(self.papers, dtype=np.int32)
        self.paper_day[papers["Paper_ID"].to_numpy(np.int64)] = as_days(papers["Submission_Date"])
        pa_author = paper_authors["Author_ID"].to_numpy(np.int32)
        pa_paper = paper_authors["Paper_ID"].to_numpy(np.int32)
        pa_day = as_days(paper_authors["Submission_Date"])
        order = np.argsort(pa_day, kind="stable")
        self.own_author = pa_author[order]
        self.own_paper = pa_paper[order]
        self.own_day = pa_day[order]
        self.own_category = self.paper_category[self.own_paper]
        self.team_size = np.bincount(pa_paper, minlength=self.papers).astype(np.float32)
        self.own_team = self.team_size[self.own_paper]
        self.orcid = authors.set_index("Author_ID")["ORCID"].notna().reindex(np.arange(self.authors), fill_value=False).to_numpy(np.float32)
        names = authors.set_index("Author_ID")["Name"].fillna("").reindex(np.arange(self.authors), fill_value="").astype(str)
        self.names = names.to_numpy()
        self.name_bucket, self.initial_bucket = self._name_buckets(names.to_numpy())
        self.name_matrix = None
        self.first_day = np.full(self.authors, np.iinfo(np.int32).max, dtype=np.int32)
        np.minimum.at(self.first_day, self.own_author, self.own_day)
        self.first_category = np.zeros(self.authors, dtype=np.int16)
        first_position = np.full(self.authors, len(self.own_author), dtype=np.int64)
        np.minimum.at(first_position, self.own_author, np.arange(len(self.own_author), dtype=np.int64))
        present = first_position < len(self.own_author)
        self.first_category[present] = self.own_category[first_position[present]]
        author_order = np.lexsort((self.own_day, self.own_author))
        self.author_history_day = self.own_day[author_order]
        self.author_history_category = self.own_category[author_order]
        history_counts = np.bincount(self.own_author, minlength=self.authors)
        self.author_offsets = np.concatenate(([0], np.cumsum(history_counts))).astype(np.int64)
        self.secondary_author, self.secondary_category, self.secondary_day = self._secondary_events(paper_authors, context, category_map)
        self.cited_author, self.cited_category, self.cited_day = self._citation_events(paper_authors, context, incoming=False)
        self.citing_author, self.citing_category, self.citing_day = self._citation_events(paper_authors, context, incoming=True)
        self.co_source, self.co_target, self.co_day = self._coauthor_events(paper_authors)
        self.pair_primary, self.pair_secondary, self.pair_day = self._category_pairs(context, category_map)
        self.paper_category_by_day = self.paper_category[np.argsort(self.paper_day, kind="stable")]
        self.paper_days_sorted = np.sort(self.paper_day, kind="stable")
        self.event_age_bin = np.minimum(np.searchsorted(np.array([183, 731, 1461]), self.own_day - self.first_day[self.own_author]), 3).astype(np.int8)

    def _name_buckets(self, names):
        full = np.empty(len(names), dtype=np.int32)
        initial = np.empty(len(names), dtype=np.int16)
        for index, value in enumerate(names):
            cleaned = "".join(character.lower() if character.isalpha() else " " for character in value)
            tokens = cleaned.split()
            token = tokens[0] if tokens else ""
            edge = token[:4] + token[-4:]
            initials = "".join(item[:1] for item in tokens[:4])
            full[index] = zlib.crc32(edge.encode()) % 8192
            initial[index] = zlib.crc32(initials.encode()) % 512
        return full, initial

    def _sorted_events(self, author, category, day):
        order = np.argsort(day, kind="stable")
        return author[order].astype(np.int32), category[order].astype(np.int16), day[order].astype(np.int32)

    def _secondary_events(self, paper_authors, context, category_map):
        secondary = context.db.table_dict["paperCategories"].df[["Paper_ID", "Category_ID", "Submission_Date"]]
        joined = paper_authors[["Paper_ID", "Author_ID"]].merge(secondary, on="Paper_ID", how="inner", sort=False)
        author = joined["Author_ID"].to_numpy(np.int32)
        category = category_map[joined["Category_ID"].to_numpy(np.int64)]
        day = as_days(joined["Submission_Date"])
        del joined
        return self._sorted_events(author, category, day)

    def _citation_events(self, paper_authors, context, incoming):
        citations = context.db.table_dict["citations"].df
        if incoming:
            left = paper_authors[["Paper_ID", "Author_ID"]].rename(columns={"Paper_ID": "References_Paper_ID"})
            joined = left.merge(citations[["Paper_ID", "References_Paper_ID", "Submission_Date"]], on="References_Paper_ID", how="inner", sort=False)
            joined = joined[joined["Paper_ID"].notna()]
            category = self.paper_category[joined["Paper_ID"].to_numpy(np.int64)]
        else:
            left = paper_authors[["Paper_ID", "Author_ID"]]
            joined = left.merge(citations[["Paper_ID", "References_Paper_ID", "Submission_Date"]], on="Paper_ID", how="inner", sort=False)
            joined = joined[joined["References_Paper_ID"].notna()]
            category = self.paper_category[joined["References_Paper_ID"].to_numpy(np.int64)]
        author = joined["Author_ID"].to_numpy(np.int32)
        day = as_days(joined["Submission_Date"])
        del joined
        return self._sorted_events(author, category, day)

    def _coauthor_events(self, paper_authors):
        frame = paper_authors[["Paper_ID", "Author_ID", "Submission_Date"]].sort_values(["Paper_ID", "Author_ID"])
        source_parts = []
        target_parts = []
        day_parts = []
        for _, group in frame.groupby("Paper_ID", sort=False):
            values = group["Author_ID"].to_numpy(np.int32)
            size = len(values)
            if size < 2:
                continue
            width = min(size - 1, 8)
            day = as_days(group["Submission_Date"].iloc[:1])[0]
            for offset in range(1, width + 1):
                source_parts.append(values)
                target_parts.append(np.roll(values, -offset))
                day_parts.append(np.full(size, day, dtype=np.int32))
        source = np.concatenate(source_parts)
        target = np.concatenate(target_parts)
        days = np.concatenate(day_parts)
        order = np.argsort(days, kind="stable")
        return source[order].astype(np.int32), target[order].astype(np.int32), days[order].astype(np.int32)

    def _category_pairs(self, context, category_map):
        secondary = context.db.table_dict["paperCategories"].df
        paper = secondary["Paper_ID"].to_numpy(np.int64)
        primary = self.paper_category[paper]
        other = category_map[secondary["Category_ID"].to_numpy(np.int64)]
        day = as_days(secondary["Submission_Date"])
        order = np.argsort(day, kind="stable")
        return primary[order], other[order], day[order]

    def auxiliary_labels(self, dates):
        outputs = []
        for timestamp in dates:
            day = np.datetime64(timestamp, "D").astype(np.int32)
            left = np.searchsorted(self.own_day, day, side="right")
            right = np.searchsorted(self.own_day, day + 182, side="right")
            ids = self.own_author[left:right].astype(np.int64) * self.classes + self.own_category[left:right]
            counts = np.bincount(ids, minlength=self.authors * self.classes).reshape(self.authors, self.classes)
            active = np.flatnonzero(counts.sum(axis=1))
            labels = counts[active].argmax(axis=1).astype(np.int16)
            outputs.append(pd.DataFrame({"date": pd.Timestamp(timestamp), "Author_ID": active, "primary_category": labels}))
        return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=["date", "Author_ID", "primary_category"])

    def cold_name_features(self):
        if self.name_matrix is None:
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=120000, sublinear_tf=True, dtype=np.float32)
            self.name_matrix = vectorizer.fit_transform(self.names)
        return self.name_matrix


class FeatureBuilder:
    def __init__(self, store):
        self.store = store
        self.classes = store.classes
        self.feature_names = []

    def _dense(self, author, category, day, timestamp, window=0, weights=None):
        right = np.searchsorted(day, timestamp, side="right")
        left = np.searchsorted(day, timestamp - window, side="right") if window else 0
        selected_weight = None if weights is None else weights[left:right]
        index = author[left:right].astype(np.int64) * self.classes + category[left:right]
        return np.bincount(index, weights=selected_weight, minlength=self.store.authors * self.classes).reshape(self.store.authors, self.classes).astype(np.float32)

    def _decayed(self, author, category, day, timestamp, half_life):
        right = np.searchsorted(day, timestamp, side="right")
        weights = np.exp2(-(timestamp - day[:right]).astype(np.float32) / half_life)
        index = author[:right].astype(np.int64) * self.classes + category[:right]
        return np.bincount(index, weights=weights, minlength=self.store.authors * self.classes).reshape(self.store.authors, self.classes).astype(np.float32)

    def _latest(self, authors, timestamp):
        recent = np.full(len(authors), -1, dtype=np.int16)
        mode3 = np.full(len(authors), -1, dtype=np.int16)
        gap = np.full(len(authors), 4000.0, dtype=np.float32)
        for row, author in enumerate(authors):
            left = self.store.author_offsets[author]
            right = self.store.author_offsets[author + 1]
            days = self.store.author_history_day[left:right]
            position = np.searchsorted(days, timestamp, side="right")
            if position:
                categories = self.store.author_history_category[left:left + position]
                recent[row] = categories[-1]
                mode3[row] = np.bincount(categories[-3:], minlength=self.classes).argmax()
                gap[row] = timestamp - days[position - 1]
        return recent, mode3, gap

    def state(self, authors, timestamp, outcome_vectors=None):
        authors = np.asarray(authors, dtype=np.int32)
        store = self.store
        life_full = self._dense(store.own_author, store.own_category, store.own_day, timestamp)
        matrices = {}
        own_counts = []
        for window in (91, 182, 365, 730):
            values = self._dense(store.own_author, store.own_category, store.own_day, timestamp, window)[authors]
            own_counts.append(values)
        own_counts.append(life_full[authors])
        for name, values in zip(("91", "182", "365", "730", "life"), own_counts):
            matrices[f"own_log_{name}"] = np.log1p(values)
            matrices[f"own_share_{name}"] = normalized(values)
        for name, recent_values, wider_values in (
            ("91", own_counts[0], own_counts[1]),
            ("182", own_counts[1], own_counts[2]),
            ("365", own_counts[2], own_counts[3]),
        ):
            previous_values = np.maximum(wider_values - recent_values, 0.0)
            matrices[f"previous_log_{name}"] = np.log1p(previous_values)
            matrices[f"previous_share_{name}"] = normalized(previous_values)
            matrices[f"momentum_{name}"] = normalized(recent_values) - normalized(previous_values)
        decay_values = []
        for half_life in (90, 365):
            values = self._decayed(store.own_author, store.own_category, store.own_day, timestamp, half_life)[authors]
            decay_values.append(values)
            matrices[f"decay_log_{half_life}"] = np.log1p(values)
            matrices[f"decay_share_{half_life}"] = normalized(values)
        collaboration_full = self._dense(store.own_author, store.own_category, store.own_day, timestamp, weights=np.clip(store.own_team - 1.0, 0.0, 20.0))
        collaboration = collaboration_full[authors]
        matrices["collaboration_log"] = np.log1p(collaboration)
        matrices["collaboration_share"] = normalized(collaboration)
        del collaboration_full
        right = np.searchsorted(store.co_day, timestamp, side="right")
        adjacency = sparse.csr_matrix((np.ones(right, dtype=np.float32), (store.co_source[:right], store.co_target[:right])), shape=(store.authors, store.authors))
        own_life_share = normalized(life_full)
        coauthor = np.asarray(adjacency[authors].dot(own_life_share), dtype=np.float32)
        coauthor = normalized(coauthor)
        matrices["coauthor_affinity"] = coauthor
        own_recent_full = self._dense(store.own_author, store.own_category, store.own_day, timestamp, 365)
        coauthor_recent = np.asarray(adjacency[authors].dot(normalized(own_recent_full)), dtype=np.float32)
        matrices["coauthor_recent_affinity"] = normalized(coauthor_recent)
        del own_recent_full, coauthor_recent
        del adjacency, own_life_share
        channel_values = {}
        for name, event_author, event_category, event_day in (
            ("cited", store.cited_author, store.cited_category, store.cited_day),
            ("citing", store.citing_author, store.citing_category, store.citing_day),
            ("secondary", store.secondary_author, store.secondary_category, store.secondary_day),
        ):
            values = self._decayed(event_author, event_category, event_day, timestamp, 365)[authors]
            channel_values[name] = values
            matrices[f"{name}_log"] = np.log1p(values)
            matrices[f"{name}_share"] = normalized(values)
        recent, mode3, gap = self._latest(authors, timestamp)
        life_top = own_counts[-1].argmax(axis=1).astype(np.int16)
        recent_top = own_counts[1].argmax(axis=1).astype(np.int16)
        matrices["is_most_recent"] = np.equal(np.arange(self.classes)[None, :], recent[:, None]).astype(np.float32)
        matrices["is_last_three_mode"] = np.equal(np.arange(self.classes)[None, :], mode3[:, None]).astype(np.float32)
        matrices["is_previous_window_dominant"] = np.equal(np.arange(self.classes)[None, :], recent_top[:, None]).astype(np.float32)
        matrices["is_career_mode"] = np.equal(np.arange(self.classes)[None, :], life_top[:, None]).astype(np.float32)
        rank_sources = {
            "rank_91": own_counts[0], "rank_182": own_counts[1], "rank_365": own_counts[2],
            "rank_730": own_counts[3], "rank_life": own_counts[4],
            "rank_decay90": decay_values[0], "rank_decay365": decay_values[1],
        }
        for name, values in rank_sources.items():
            order = np.argsort(-values, axis=1, kind="stable")
            rank = np.empty_like(order, dtype=np.float32)
            rows = np.arange(len(authors))[:, None]
            rank[rows, order] = np.arange(self.classes, dtype=np.float32)[None, :]
            matrices[name] = rank / (self.classes - 1)
        global_recent = self._global_counts(timestamp, 182)
        global_previous = self._global_counts(timestamp - 182, 182)
        global_life = self._global_counts(timestamp, 0)
        global_recent = global_recent / max(global_recent.sum(), 1.0)
        global_previous = global_previous / max(global_previous.sum(), 1.0)
        global_life = global_life / max(global_life.sum(), 1.0)
        global_vectors = {
            "global_recent": global_recent,
            "global_lifetime": global_life,
            "global_slope": global_recent - global_previous,
        }
        if outcome_vectors is not None:
            global_vectors.update(outcome_vectors)
        debut_mask = (store.first_day <= timestamp) & (store.first_day > timestamp - 730)
        debut = np.bincount(store.first_category[debut_mask], minlength=self.classes).astype(np.float32) + 1.0
        global_vectors["debut_prevalence"] = debut / debut.sum()
        event_right = np.searchsorted(store.own_day, timestamp, side="right")
        age_index = store.event_age_bin[:event_right].astype(np.int64) * self.classes + store.own_category[:event_right]
        age_counts = np.bincount(age_index, minlength=4 * self.classes).reshape(4, self.classes).astype(np.float32) + 1.0
        age_counts = normalized(age_counts)
        author_age = timestamp - store.first_day[authors]
        author_age = np.where(author_age < 0, 0, author_age)
        author_age_bin = np.minimum(np.searchsorted(np.array([183, 731, 1461]), author_age), 3)
        matrices["age_specific_prior"] = age_counts[author_age_bin]
        pair_right = np.searchsorted(store.pair_day, timestamp, side="right")
        pair_index = store.pair_primary[:pair_right].astype(np.int64) * self.classes + store.pair_secondary[:pair_right]
        similarity = np.bincount(pair_index, minlength=self.classes * self.classes).reshape(self.classes, self.classes).astype(np.float32)
        similarity = similarity + similarity.T + np.eye(self.classes, dtype=np.float32)
        similarity = similarity / np.sqrt(np.maximum(similarity.sum(axis=1, keepdims=True), 1.0) * np.maximum(similarity.sum(axis=1, keepdims=True).T, 1.0))
        matrices["top_category_similarity"] = similarity[:, life_top].T
        bucket_index = store.name_bucket[store.own_author[:event_right]].astype(np.int64) * self.classes + store.own_category[:event_right]
        bucket_counts = np.bincount(bucket_index, minlength=8192 * self.classes).reshape(8192, self.classes).astype(np.float32) + 0.25
        initial_index = store.initial_bucket[store.own_author[:event_right]].astype(np.int64) * self.classes + store.own_category[:event_right]
        initial_counts = np.bincount(initial_index, minlength=512 * self.classes).reshape(512, self.classes).astype(np.float32) + 1.0
        name_probability = normalized(bucket_counts[store.name_bucket[authors]]) * 0.7 + normalized(initial_counts[store.initial_bucket[authors]]) * 0.3
        matrices["name_model_logit"] = np.log(np.maximum(name_probability, 1e-7))
        name_order = np.argsort(-name_probability, axis=1, kind="stable")
        name_rank = np.empty_like(name_order, dtype=np.float32)
        name_rank[np.arange(len(authors))[:, None], name_order] = np.arange(self.classes, dtype=np.float32)[None, :]
        matrices["name_model_rank"] = name_rank / (self.classes - 1)
        observed_authors = np.flatnonzero(store.first_day <= timestamp)
        global_smoothing = global_recent * 16.0
        for width in (128, 512, 2048):
            bucket_total = (store.authors + width - 1) // width
            ids = (observed_authors // width).astype(np.int64) * self.classes + store.first_category[observed_authors]
            counts = np.bincount(ids, minlength=bucket_total * self.classes).reshape(bucket_total, self.classes).astype(np.float32)
            cumulative = np.vstack([np.zeros((1, self.classes), dtype=np.float32), np.cumsum(counts, axis=0)])
            author_bucket = np.minimum(authors // width, bucket_total - 1)
            lower = np.maximum(author_bucket - 2, 0)
            local = cumulative[author_bucket + 1] - cumulative[lower] + global_smoothing[None, :]
            matrices[f"author_id_prior_{width}"] = normalized(local)
        channel_top = [recent_top, life_top, coauthor.argmax(axis=1)]
        for name in ("cited", "citing", "secondary"):
            channel_top.append(channel_values[name].argmax(axis=1))
        agreement = np.zeros((len(authors), self.classes), dtype=np.float32)
        for index, values in enumerate(channel_top):
            flag = np.equal(np.arange(self.classes)[None, :], values[:, None]).astype(np.float32)
            matrices[f"channel_top_{index}"] = flag
            agreement += flag
        matrices["channel_agreement"] = agreement
        life_total = own_counts[-1].sum(axis=1)
        life_share = normalized(own_counts[-1])
        entropy = -(life_share * np.log(np.maximum(life_share, 1e-8))).sum(axis=1) / math.log(self.classes)
        context = {
            "activity_91": np.log1p(own_counts[0].sum(axis=1)),
            "activity_182": np.log1p(own_counts[1].sum(axis=1)),
            "activity_365": np.log1p(own_counts[2].sum(axis=1)),
            "activity_730": np.log1p(own_counts[3].sum(axis=1)),
            "activity_life": np.log1p(life_total),
            "entropy": entropy.astype(np.float32),
            "time_since_last": np.log1p(np.maximum(gap, 0.0)),
            "history_age": np.log1p(np.maximum(author_age, 0)).astype(np.float32),
            "cold": (life_total == 0).astype(np.float32),
            "orcid": store.orcid[authors],
            "author_id_scaled": authors.astype(np.float32) / max(store.authors - 1, 1),
            "author_id_cohort": np.minimum(authors // 5000, 31).astype(np.float32),
            "frontier_distance": (authors - (int(observed_authors.max()) if len(observed_authors) else 0)).astype(np.float32) / 5000.0,
            "team_mean": self._team_mean(authors, timestamp, 0),
            "team_recent_mean": self._team_mean(authors, timestamp, 365),
            "team_max": self._team_max(authors, timestamp),
            "own_coverage": (life_total > 0).astype(np.float32),
            "coauthor_coverage": (coauthor.sum(axis=1) > 0).astype(np.float32),
            "cited_coverage": (channel_values["cited"].sum(axis=1) > 0).astype(np.float32),
            "citing_coverage": (channel_values["citing"].sum(axis=1) > 0).astype(np.float32),
            "secondary_coverage": (channel_values["secondary"].sum(axis=1) > 0).astype(np.float32),
            "seed_year": np.full(len(authors), (timestamp - 17532) / 365.25, dtype=np.float32),
        }
        cold_prior = global_vectors.get("cold_outcome_prior", global_recent)
        cheap = normalized(decay_values[1]) + global_recent[None, :] * 0.18 + name_probability * 0.08 + cold_prior[None, :] * 0.2
        cheap[life_total == 0] = cold_prior
        del life_full, own_counts, decay_values, channel_values, bucket_counts, initial_counts
        gc.collect()
        return {"authors": authors, "matrices": matrices, "vectors": global_vectors, "context": context, "cheap": cheap.astype(np.float32)}

    def _global_counts(self, timestamp, window):
        right = np.searchsorted(self.store.paper_days_sorted, timestamp, side="right")
        left = np.searchsorted(self.store.paper_days_sorted, timestamp - window, side="right") if window else 0
        return np.bincount(self.store.paper_category_by_day[left:right], minlength=self.classes).astype(np.float32) + 1.0

    def _team_mean(self, authors, timestamp, window):
        right = np.searchsorted(self.store.own_day, timestamp, side="right")
        left = np.searchsorted(self.store.own_day, timestamp - window, side="right") if window else 0
        sums = np.bincount(self.store.own_author[left:right], weights=self.store.own_team[left:right], minlength=self.store.authors)
        counts = np.bincount(self.store.own_author[left:right], minlength=self.store.authors)
        return (sums[authors] / np.maximum(counts[authors], 1)).astype(np.float32)

    def _team_max(self, authors, timestamp):
        right = np.searchsorted(self.store.own_day, timestamp, side="right")
        values = np.zeros(self.store.authors, dtype=np.float32)
        np.maximum.at(values, self.store.own_author[:right], self.store.own_team[:right])
        return values[authors]

    def sample_candidates(self, labels, cheap, hard, random_count, rng):
        rows = len(labels)
        width = 1 + hard + random_count
        candidates = np.empty((rows, width), dtype=np.int16)
        order = np.argsort(-cheap, axis=1, kind="stable")
        for row in range(rows):
            chosen = [int(labels[row])]
            chosen_set = {chosen[0]}
            for category in order[row]:
                value = int(category)
                if value not in chosen_set:
                    chosen.append(value)
                    chosen_set.add(value)
                    if len(chosen) == 1 + hard:
                        break
            while len(chosen) < width:
                value = int(rng.integers(self.classes))
                if value not in chosen_set:
                    chosen.append(value)
                    chosen_set.add(value)
            candidates[row] = chosen
        return candidates

    def assemble(self, state, candidates, row_indices=None):
        if row_indices is None:
            row_indices = np.arange(len(candidates), dtype=np.int64)
        candidates = np.asarray(candidates, dtype=np.int16)
        width = candidates.shape[1]
        row_grid = row_indices[:, None]
        columns = [candidates.reshape(-1).astype(np.float32)]
        names = ["candidate_category"]
        for name, matrix in state["matrices"].items():
            columns.append(matrix[row_grid, candidates].reshape(-1).astype(np.float32))
            names.append(name)
        for name, vector in state["vectors"].items():
            columns.append(np.broadcast_to(vector[candidates], candidates.shape).reshape(-1).astype(np.float32))
            names.append(name)
        for name, values in state["context"].items():
            columns.append(np.repeat(values[row_indices], width).astype(np.float32))
            names.append(name)
        self.feature_names = names
        return np.column_stack(columns).astype(np.float32, copy=False)

    def _outcome_vectors(self, label_history, timestamp):
        available = label_history[label_history["date"] <= pd.Timestamp(timestamp - 182, unit="D")]
        if len(available) == 0:
            uniform = np.full(self.classes, 1.0 / self.classes, dtype=np.float32)
            return {"outcome_recent_prior": uniform, "outcome_prior_slope": np.zeros(self.classes, dtype=np.float32), "cold_outcome_prior": uniform}
        dates = sorted(available["date"].unique())
        recent = class_distribution(available.loc[available["date"] == dates[-1], "primary_category"].to_numpy(), self.classes).astype(np.float32)
        if len(dates) > 1:
            previous = class_distribution(available.loc[available["date"] == dates[-2], "primary_category"].to_numpy(), self.classes).astype(np.float32)
        else:
            previous = class_distribution(available["primary_category"].to_numpy(), self.classes).astype(np.float32)
        author = available["Author_ID"].to_numpy(np.int32)
        seed_day = np.asarray(available["date"].values, dtype="datetime64[D]").astype(np.int32)
        cold = self.store.first_day[author] > seed_day
        cold_available = available.loc[cold]
        if len(cold_available):
            cold_prior = class_distribution(cold_available["primary_category"].to_numpy(), self.classes).astype(np.float32)
        else:
            cold_prior = recent
        return {"outcome_recent_prior": recent, "outcome_prior_slope": recent - previous, "cold_outcome_prior": cold_prior}

    def build_sampled(self, seeds, hard, random_count, seed, label_history=None):
        rng = np.random.default_rng(seed)
        if label_history is None:
            label_history = seeds
        parts = []
        labels = []
        dates = []
        candidates_all = []
        for timestamp, group in seeds.groupby("date", sort=True):
            day = np.datetime64(timestamp, "D").astype(np.int32)
            authors = group["Author_ID"].to_numpy(np.int32)
            target = group["primary_category"].to_numpy(np.int16)
            state = self.state(authors, day, self._outcome_vectors(label_history, day))
            candidates = self.sample_candidates(target, state["cheap"], hard, random_count, rng)
            parts.append(self.assemble(state, candidates))
            labels.append((candidates == target[:, None]).reshape(-1).astype(np.int8))
            dates.append(np.full(len(group), day, dtype=np.int32))
            candidates_all.append(candidates)
            print(f"[features] sampled {timestamp.date()} seeds={len(group)} candidates={candidates.shape[1]}")
            del state
            gc.collect()
        return np.vstack(parts), np.concatenate(labels), np.concatenate(dates), np.vstack(candidates_all)

    def build_full(self, seeds, label_history=None):
        parts = []
        cheap_parts = []
        for timestamp, group in seeds.groupby("date", sort=False):
            day = np.datetime64(timestamp, "D").astype(np.int32)
            authors = group["Author_ID"].to_numpy(np.int32)
            outcome_vectors = self._outcome_vectors(label_history, day) if label_history is not None else None
            state = self.state(authors, day, outcome_vectors)
            candidates = np.broadcast_to(np.arange(self.classes, dtype=np.int16), (len(group), self.classes))
            parts.append(self.assemble(state, candidates))
            cheap_parts.append(state["cheap"])
            del state
            gc.collect()
        return np.vstack(parts), np.vstack(cheap_parts)


class AffinityPipeline:
    def __init__(self, context, debug, seed):
        self.context = context
        self.debug = debug
        self.seed = seed
        self.store = RelationalStore(context)
        self.builder = FeatureBuilder(self.store)
        self.diagnostics = {"debug": debug}

    def _training_frames(self):
        train = self.context.train.df[["date", "Author_ID", "primary_category"]].copy()
        train_dates = sorted(train["date"].unique())
        auxiliary_a_dates = [pd.Timestamp(train_dates[index] + (train_dates[index + 1] - train_dates[index]) / 2).normalize() for index in range(len(train_dates) - 1)]
        auxiliary_a = self.store.auxiliary_labels(auxiliary_a_dates)
        model_a = pd.concat([train, auxiliary_a], ignore_index=True).sort_values(["date", "Author_ID"], kind="stable").reset_index(drop=True)
        validation = self.context.val.df[["date", "Author_ID", "primary_category"]].copy()
        validation_date = pd.Timestamp(validation["date"].iloc[0])
        test_date = pd.Timestamp(self.context.test.df["date"].iloc[0])
        extra_dates = []
        cursor = pd.Timestamp(train_dates[-1]) + pd.Timedelta(days=91)
        occupied = set(model_a["date"].tolist()) | {validation_date}
        while cursor + pd.Timedelta(days=182) <= test_date:
            if cursor not in occupied:
                extra_dates.append(cursor)
            cursor += pd.Timedelta(days=91)
        auxiliary_b = self.store.auxiliary_labels(extra_dates)
        model_b = pd.concat([model_a, validation, auxiliary_b], ignore_index=True).drop_duplicates(["date", "Author_ID"], keep="first").sort_values(["date", "Author_ID"], kind="stable").reset_index(drop=True)
        self.diagnostics.update({
            "model_a_seeds": int(len(model_a)), "model_b_seeds": int(len(model_b)),
            "auxiliary_a_seeds": int(len(auxiliary_a)), "auxiliary_b_seeds": int(len(auxiliary_b)),
            "candidate_recall": 1.0,
        })
        return model_a, model_b

    def _params(self, objective):
        values = {
            "objective": objective,
            "learning_rate": 0.04,
            "num_leaves": 127,
            "min_data_in_leaf": 300,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 2.0,
            "max_bin": 127,
            "verbosity": -1,
            "num_threads": 11,
            "seed": self.seed,
            "feature_fraction_seed": self.seed,
            "bagging_seed": self.seed,
        }
        if objective == "lambdarank":
            values.update({"metric": "ndcg", "ndcg_eval_at": [1, 5], "lambdarank_truncation_level": 5, "label_gain": [0, 1]})
        else:
            values.update({"metric": "binary", "is_unbalance": True})
        return values

    def _train(self, features, labels, group_size, objective, rounds, group_dates=None, recency_half_life=0):
        groups = np.full(len(labels) // group_size, group_size, dtype=np.int32)
        weights = None
        if group_dates is not None and recency_half_life:
            group_weights = np.exp2(-(np.max(group_dates) - group_dates).astype(np.float32) / recency_half_life)
            weights = np.repeat(group_weights, group_size)
        dataset = lgb.Dataset(features, label=labels, weight=weights, group=groups, categorical_feature=[0], free_raw_data=False)
        return lgb.train(self._params(objective), dataset, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)], keep_training_booster=True)

    def _origin_prior_ratio(self, seeds, origin):
        earlier = seeds[seeds["date"] <= pd.Timestamp(origin - 182, unit="D")]
        latest_date = earlier["date"].max()
        overall = class_distribution(earlier["primary_category"].to_numpy(), self.store.classes)
        recent = class_distribution(earlier.loc[earlier["date"] == latest_date, "primary_category"].to_numpy(), self.store.classes)
        return np.clip(recent / overall, 0.2, 5.0)

    def _score_checkpoints(self, booster, features, labels, ratio, checkpoints, cold=None, fallback=None):
        results = {}
        for rounds in checkpoints:
            scores = booster.predict(features, num_iteration=rounds, raw_score=True).reshape(len(labels), self.store.classes)
            row = {}
            for exponent in (0.0, 0.25, 0.5):
                adjusted = scores + exponent * np.log(ratio)[None, :]
                row[str(exponent)] = float(np.mean(adjusted.argmax(axis=1) == labels))
            if cold is not None and fallback is not None:
                model_prediction = scores.argmax(axis=1)
                fallback_prediction = fallback.argmax(axis=1)
                combined = model_prediction.copy()
                combined[cold] = fallback_prediction[cold]
                row["cold_model_accuracy"] = float(np.mean(model_prediction[cold] == labels[cold]))
                row["cold_fallback_accuracy"] = float(np.mean(fallback_prediction[cold] == labels[cold]))
                row["with_cold_fallback"] = float(np.mean(combined == labels))
                warm = ~cold
                row["warm_model_accuracy"] = float(np.mean(model_prediction[warm] == labels[warm]))
                mode_bonus = np.equal(np.arange(self.store.classes)[None, :], fallback_prediction[:, None])
                for strength in (0.05, 0.1, 0.2, 0.4):
                    blended = (scores + strength * mode_bonus).argmax(axis=1)
                    row[f"warm_blend_{strength}"] = float(np.mean(blended[warm] == labels[warm]))
            results[str(rounds)] = row
        return results

    def _cold_subset(self, seeds):
        authors = seeds["Author_ID"].to_numpy(np.int32)
        seed_days = np.asarray(seeds["date"].values, dtype="datetime64[D]").astype(np.int32)
        return self.store.first_day[authors] > seed_days

    def _fit_cold_name(self, seeds):
        cold = self._cold_subset(seeds)
        selected = seeds.loc[cold]
        model = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=30, tol=1e-3, average=True, random_state=self.seed, n_jobs=11)
        model.fit(self.store.cold_name_features()[selected["Author_ID"].to_numpy(np.int32)], selected["primary_category"].to_numpy(np.int16))
        prior = class_distribution(selected["primary_category"].to_numpy(np.int16), self.store.classes)
        return model, prior

    def _cold_name_internal(self, model_a, origins):
        scores = []
        for origin in origins:
            timestamp = pd.Timestamp(origin, unit="D")
            available = model_a[model_a["date"] <= timestamp - pd.Timedelta(days=182)]
            model, prior = self._fit_cold_name(available)
            target = model_a[model_a["date"] == timestamp]
            cold = self._cold_subset(target)
            selected = target.loc[cold]
            raw = model.decision_function(self.store.cold_name_features()[selected["Author_ID"].to_numpy(np.int32)])
            full = np.full((len(selected), self.store.classes), -20.0, dtype=np.float32)
            full[:, model.classes_.astype(np.int64)] = raw
            prediction = (full + np.log(prior)[None, :]).argmax(axis=1)
            scores.append(float(np.mean(prediction == selected["primary_category"].to_numpy(np.int16))))
        return scores

    def _internal_selection(self, features, labels, seed_dates, candidates, model_a, group_size):
        origins = np.unique(seed_dates)[-2:].tolist()
        objective_results = {"binary": [], "lambdarank": []}
        best_models = {}
        pilot_rounds = 200
        for objective in ("binary", "lambdarank"):
            for origin in origins:
                train_groups = int(np.searchsorted(seed_dates, origin - 182, side="right"))
                fold_rows = model_a[np.asarray(model_a["date"].values, dtype="datetime64[D]").astype(np.int32) == origin]
                full_features, fallback = self.builder.build_full(fold_rows, model_a)
                fold_authors = fold_rows["Author_ID"].to_numpy(np.int32)
                cold = self.store.first_day[fold_authors] > origin
                start = time.time()
                booster = self._train(features[:train_groups * group_size], labels[:train_groups * group_size], group_size, objective, pilot_rounds, seed_dates[:train_groups], 730)
                rate = pilot_rounds / max((time.time() - start) / 60.0, 1e-4)
                ratio = self._origin_prior_ratio(model_a, origin)
                fold_labels = fold_rows["primary_category"].to_numpy(np.int16)
                scores = self._score_checkpoints(booster, full_features, fold_labels, ratio, (50, 100, 150, 200), cold, fallback)
                objective_results[objective].append({"origin": str(pd.Timestamp(origin, unit="D").date()), "rounds_per_minute": rate, "scores": scores})
                if origin == origins[-1]:
                    best_models[objective] = (booster, full_features, fold_labels, ratio, cold, fallback)
                else:
                    del booster, full_features
                gc.collect()
        means = {}
        for objective, folds in objective_results.items():
            means[objective] = float(np.mean([max(value for point in fold["scores"].values() for value in [point["0.0"]]) for fold in folds]))
        objective = max(means, key=means.get)
        selected_booster, full_features, fold_labels, ratio, cold, fallback = best_models[objective]
        latest_origin = origins[-1]
        train_groups = int(np.searchsorted(seed_dates, latest_origin - 182, side="right"))
        recency_results = {"730": self._score_checkpoints(selected_booster, full_features, fold_labels, ratio, (200,))["200"]["0.0"]}
        recency_models = {730: selected_booster}
        for half_life in (0, 1460):
            booster = self._train(features[:train_groups * group_size], labels[:train_groups * group_size], group_size, objective, pilot_rounds, seed_dates[:train_groups], half_life)
            recency_results[str(half_life)] = self._score_checkpoints(booster, full_features, fold_labels, ratio, (200,))["200"]["0.0"]
            recency_models[half_life] = booster
        half_life = max((0, 1460, 730), key=lambda value: recency_results[str(value)])
        selected_booster = recency_models[half_life]
        extension = 200
        if extension > pilot_rounds:
            groups = np.full(train_groups, group_size, dtype=np.int32)
            weights = None
            if half_life:
                group_weights = np.exp2(-(np.max(seed_dates[:train_groups]) - seed_dates[:train_groups]).astype(np.float32) / half_life)
                weights = np.repeat(group_weights, group_size)
            dataset = lgb.Dataset(features[:train_groups * group_size], label=labels[:train_groups * group_size], weight=weights, group=groups, categorical_feature=[0], free_raw_data=False)
            selected_booster = lgb.train(self._params(objective), dataset, num_boost_round=extension - pilot_rounds, init_model=selected_booster, callbacks=[lgb.log_evaluation(0)], keep_training_booster=True)
        extended_scores = self._score_checkpoints(selected_booster, full_features, fold_labels, ratio, (50, 100, 150, 200), cold, fallback)
        objective_results[objective][-1]["extended_scores"] = extended_scores
        round_means = {}
        for rounds in (50, 100, 150, 200):
            values = []
            for fold in objective_results[objective][:-1]:
                if str(rounds) in fold["scores"]:
                    values.append(fold["scores"][str(rounds)]["0.0"])
            values.append(extended_scores[str(rounds)]["0.0"])
            round_means[rounds] = float(np.mean(values))
        selected_rounds = max(round_means, key=round_means.get)
        exponent_scores = {}
        for exponent in (0.0, 0.25, 0.5):
            per_fold = []
            for fold in objective_results[objective][:-1]:
                checkpoint = str(min(selected_rounds, 200))
                per_fold.append(fold["scores"][checkpoint][str(exponent)])
            per_fold.append(extended_scores[str(selected_rounds)][str(exponent)])
            exponent_scores[exponent] = per_fold
        eligible = [exponent for exponent, values in exponent_scores.items() if exponent == 0.0 or all(value > base for value, base in zip(values, exponent_scores[0.0]))]
        exponent = max(eligible, key=lambda item: np.mean(exponent_scores[item]))
        fallback_values = []
        base_values = []
        for fold in objective_results[objective][:-1]:
            checkpoint = str(selected_rounds)
            fallback_values.append(fold["scores"][checkpoint]["with_cold_fallback"])
            base_values.append(fold["scores"][checkpoint]["0.0"])
        fallback_values.append(extended_scores[str(selected_rounds)]["with_cold_fallback"])
        base_values.append(extended_scores[str(selected_rounds)]["0.0"])
        use_cold_fallback = all(value > base for value, base in zip(fallback_values, base_values))
        fallback_cold_scores = [objective_results[objective][0]["scores"][str(selected_rounds)]["cold_fallback_accuracy"], extended_scores[str(selected_rounds)]["cold_fallback_accuracy"]]
        model_cold_scores = [objective_results[objective][0]["scores"][str(selected_rounds)]["cold_model_accuracy"], extended_scores[str(selected_rounds)]["cold_model_accuracy"]]
        cold_name_scores = self._cold_name_internal(model_a, origins)
        use_cold_name = all(value > max(fallback_value, model_value) for value, fallback_value, model_value in zip(cold_name_scores, fallback_cold_scores, model_cold_scores))
        cold_strategy = "name" if use_cold_name else ("prior" if use_cold_fallback else "ranker")
        warm_blend_results = {}
        for strength in (0.0, 0.05, 0.1, 0.2, 0.4):
            if strength == 0.0:
                values = [objective_results[objective][0]["scores"][str(selected_rounds)]["warm_model_accuracy"], extended_scores[str(selected_rounds)]["warm_model_accuracy"]]
            else:
                values = [objective_results[objective][0]["scores"][str(selected_rounds)][f"warm_blend_{strength}"], extended_scores[str(selected_rounds)][f"warm_blend_{strength}"]]
            warm_blend_results[str(strength)] = values
        eligible_blends = [0.0] + [strength for strength in (0.05, 0.1, 0.2, 0.4) if all(value > base for value, base in zip(warm_blend_results[str(strength)], warm_blend_results["0.0"]))]
        warm_blend = max(eligible_blends, key=lambda value: np.mean(warm_blend_results[str(value)]))
        selected_raw = selected_booster.predict(full_features, num_iteration=selected_rounds, raw_score=True).reshape(len(fold_labels), self.store.classes)
        selected_prediction = (selected_raw + exponent * np.log(ratio)[None, :]).argmax(axis=1)
        fold_authors = model_a.loc[np.asarray(model_a["date"].values, dtype="datetime64[D]").astype(np.int32) == latest_origin, "Author_ID"].to_numpy(np.int32)
        cold = self.store.first_day[fold_authors] > latest_origin
        slices = {
            "cold": {"count": int(cold.sum()), "accuracy": float(np.mean(selected_prediction[cold] == fold_labels[cold]))},
            "warm": {"count": int((~cold).sum()), "accuracy": float(np.mean(selected_prediction[~cold] == fold_labels[~cold]))},
        }
        self.diagnostics.update({"internal_folds": objective_results, "internal_latest_slices": slices, "cold_fallback_scores": fallback_values, "cold_name_scores": cold_name_scores, "cold_strategy": cold_strategy, "warm_blend_results": warm_blend_results, "warm_blend": warm_blend, "objective_means": means, "recency_results": recency_results, "selected_objective": objective, "selected_rounds": int(selected_rounds), "recency_half_life": int(half_life), "prior_exponent": float(exponent)})
        del selected_booster, full_features
        for value in best_models.values():
            del value
        gc.collect()
        return objective, selected_rounds, half_life, exponent, cold_strategy, warm_blend

    def _inference_prior_ratio(self, seeds):
        latest = seeds["date"].max()
        overall = class_distribution(seeds["primary_category"].to_numpy(), self.store.classes)
        recent = class_distribution(seeds.loc[seeds["date"] == latest, "primary_category"].to_numpy(), self.store.classes)
        return np.clip(recent / overall, 0.2, 5.0)

    def _predict(self, booster, seeds, training_seeds, exponent, cold_strategy, warm_blend, cold_name_model=None, limit=None):
        if limit is None or limit >= len(seeds):
            features, cheap = self.builder.build_full(seeds, training_seeds)
            scores = booster.predict(features, raw_score=True).reshape(len(seeds), self.store.classes)
        else:
            processed = seeds.iloc[:limit]
            features, cheap_processed = self.builder.build_full(processed, training_seeds)
            scores_processed = booster.predict(features, raw_score=True).reshape(limit, self.store.classes)
            _, cheap = self.builder.build_full(seeds, training_seeds)
            scores = np.log(np.maximum(cheap, 1e-7))
            scores[:limit] = scores_processed
            del cheap_processed
        ratio = self._inference_prior_ratio(training_seeds)
        scores += exponent * np.log(ratio)[None, :]
        if warm_blend:
            mode = cheap.argmax(axis=1)
            scores[np.arange(len(scores)), mode] += warm_blend
        if cold_strategy != "ranker":
            seed_days = np.asarray(seeds["date"].values, dtype="datetime64[D]").astype(np.int32)
            authors = seeds["Author_ID"].to_numpy(np.int32)
            cold = self.store.first_day[authors] > seed_days
            if cold_strategy == "name":
                model, prior = cold_name_model
                raw = model.decision_function(self.store.cold_name_features()[authors[cold]])
                name_scores = np.full((int(cold.sum()), self.store.classes), -20.0, dtype=np.float32)
                name_scores[:, model.classes_.astype(np.int64)] = raw
                scores[cold] = name_scores + np.log(prior)[None, :]
            else:
                scores[cold] = np.log(np.maximum(cheap[cold], 1e-7))
        probabilities = softmax(scores).astype(np.float32)
        del features, cheap, scores
        gc.collect()
        return probabilities

    def run(self):
        start = time.time()
        model_a, model_b = self._training_frames()
        hard = 8 if self.debug else 15
        random_count = 0 if self.debug else 5
        group_size = 1 + hard + random_count
        if self.debug:
            model_a_build = model_a.iloc[-10000:].reset_index(drop=True)
        else:
            model_a_build = model_a
        features_a, labels_a, dates_a, candidates_a = self.builder.build_sampled(model_a_build, hard, random_count, self.seed, model_a)
        self.diagnostics.update({"candidate_rows_a": int(len(labels_a)), "feature_count": int(features_a.shape[1]), "feature_build_seconds": time.time() - start})
        if not np.all(labels_a.reshape(-1, group_size).sum(axis=1) == 1):
            raise RuntimeError("candidate positive recall is not exactly one per group")
        if self.debug:
            objective, rounds, half_life, exponent, cold_strategy, warm_blend = "lambdarank", 50, 730, 0.0, "name", 0.0
        else:
            objective, rounds, half_life, exponent, cold_strategy, warm_blend = self._internal_selection(features_a, labels_a, dates_a, candidates_a, model_a_build, group_size)
        train_start = time.time()
        booster_a = self._train(features_a, labels_a, group_size, objective, rounds, dates_a, half_life)
        self.diagnostics["model_a_train_seconds"] = time.time() - train_start
        validation_seeds = self.context.val.df[["date", "Author_ID"]].copy()
        prediction_limit = 5000 if self.debug else None
        cold_name_a = self._fit_cold_name(model_a_build) if cold_strategy == "name" else None
        validation_predictions = self._predict(booster_a, validation_seeds, model_a_build, exponent, cold_strategy, warm_blend, cold_name_a, prediction_limit)
        del booster_a
        if self.debug:
            features_b, labels_b = features_a, labels_a
            model_b_build = model_a_build
        else:
            extra = model_b[model_b["date"] > model_a["date"].max()].reset_index(drop=True)
            features_extra, labels_extra, _, _ = self.builder.build_sampled(extra, hard, random_count, self.seed + 1, model_b)
            features_b = np.vstack([features_a, features_extra])
            labels_b = np.concatenate([labels_a, labels_extra])
            model_b_build = model_b
            del features_extra, labels_extra
        train_start = time.time()
        dates_b = np.asarray(model_b_build["date"].values, dtype="datetime64[D]").astype(np.int32)
        booster_b = self._train(features_b, labels_b, group_size, objective, rounds, dates_b, half_life)
        self.diagnostics["model_b_train_seconds"] = time.time() - train_start
        test_seeds = self.context.test.df[["date", "Author_ID"]].copy()
        cold_name_b = self._fit_cold_name(model_b_build) if cold_strategy == "name" else None
        test_predictions = self._predict(booster_b, test_seeds, model_b_build, exponent, cold_strategy, warm_blend, cold_name_b, prediction_limit)
        self.diagnostics.update({
            "total_seconds": time.time() - start,
            "validation_cold": float(np.mean(self.store.first_day[validation_seeds["Author_ID"].to_numpy(np.int32)] > np.datetime64(validation_seeds["date"].iloc[0], "D").astype(np.int32))),
            "test_cold": float(np.mean(self.store.first_day[test_seeds["Author_ID"].to_numpy(np.int32)] > np.datetime64(test_seeds["date"].iloc[0], "D").astype(np.int32))),
        })
        return {"val_predictions": validation_predictions, "test_predictions": test_predictions, **self.diagnostics}
