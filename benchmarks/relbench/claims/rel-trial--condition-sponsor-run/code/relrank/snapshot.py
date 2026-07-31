import os
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .text import portfolio_embeddings, semantic_topk


HALF_LIVES = (90, 365, 1095)


def _row_normalize(matrix):
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    return sparse.diags(1.0 / np.maximum(norms, 1e-8)) @ matrix


def _top(values: np.ndarray, k: int):
    if len(values) <= k:
        return np.argsort(-values, kind="stable")
    ids = np.argpartition(-values, k - 1)[:k]
    return ids[np.argsort(-values[ids], kind="stable")]


def _matrix_from_frame(frame, row, col, value, shape):
    return sparse.coo_matrix(
        (frame[value].to_numpy(dtype=np.float32), (frame[row].to_numpy(), frame[col].to_numpy())),
        shape=shape,
    ).tocsr()


def _extract(matrix, row: int, columns: np.ndarray):
    return matrix.getrow(row)[:, columns].toarray().ravel().astype(np.float32)


def _normalize_component(values):
    return values / max(float(np.max(np.abs(values))), 1e-8)


@dataclass
class RetrievalRecord:
    condition_id: int
    candidates: np.ndarray
    source_features: np.ndarray
    baseline_components: np.ndarray


class Snapshot:
    def __init__(self, assets, cutoff, text_assets=None, debug=False):
        started = time.time()
        self.assets = assets
        self.cutoff = pd.Timestamp(cutoff)
        self.n_conditions = assets.n_conditions
        self.n_sponsors = assets.n_sponsors
        self.n_studies = assets.n_studies
        self.debug = debug
        self.condition_rel = assets.condition_rel[assets.condition_rel["date"] <= self.cutoff].copy()
        self.sponsor_rel = assets.sponsor_rel[assets.sponsor_rel["date"] <= self.cutoff].copy()
        self.events = assets.events[assets.events["visible"] <= self.cutoff].copy()
        self.condition_study = sparse.coo_matrix(
            (
                np.ones(len(self.condition_rel), dtype=np.float32),
                (self.condition_rel["condition_id"].to_numpy(), self.condition_rel["nct_id"].to_numpy()),
            ),
            shape=(self.n_conditions, self.n_studies),
        ).tocsr()
        self.sponsor_study = sparse.coo_matrix(
            (
                np.ones(len(self.sponsor_rel), dtype=np.float32),
                (self.sponsor_rel["sponsor_id"].to_numpy(), self.sponsor_rel["nct_id"].to_numpy()),
            ),
            shape=(self.n_sponsors, self.n_studies),
        ).tocsr()
        self._build_pair_features()
        self._build_entity_features()
        self._build_profiles()
        self._build_interventions()
        self._build_als(32 if debug else 64)
        self.portfolio = portfolio_embeddings(self, text_assets, str(self.cutoff.date())) if text_assets is not None else None
        self.query_embeddings = text_assets.query if text_assets is not None else None
        print(
            f"[snapshot] {self.cutoff.date()} events={len(self.events)} pairs={self.pair_count.nnz} "
            f"built in {time.time() - started:.1f}s"
        )

    def _build_pair_features(self):
        shape = (self.n_conditions, self.n_sponsors)
        age = (self.cutoff - self.events["visible"]).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
        self.pair_decay = {}
        for half_life in HALF_LIVES:
            values = np.exp2(-age / half_life).astype(np.float32)
            self.pair_decay[half_life] = sparse.coo_matrix(
                (values, (self.events["condition_id"].to_numpy(), self.events["sponsor_id"].to_numpy())),
                shape=shape,
            ).tocsr()
        stats = self.events.groupby(["condition_id", "sponsor_id"], sort=False).agg(
            pair_count=("nct_id", "size"),
            pair_trials=("nct_id", "nunique"),
            first_visible=("visible", "min"),
            last_visible=("visible", "max"),
        )
        stats = stats.reset_index()
        stats["days_since_plus"] = (
            (self.cutoff - stats["last_visible"]).dt.total_seconds() / 86400.0 + 1.0
        ).astype(np.float32)
        stats["active_span_plus"] = (
            (stats["last_visible"] - stats["first_visible"]).dt.total_seconds() / 86400.0 + 1.0
        ).astype(np.float32)
        self.pair_count = _matrix_from_frame(stats, "condition_id", "sponsor_id", "pair_count", shape)
        self.pair_trials = _matrix_from_frame(stats, "condition_id", "sponsor_id", "pair_trials", shape)
        self.pair_days_since = _matrix_from_frame(stats, "condition_id", "sponsor_id", "days_since_plus", shape)
        self.pair_active_span = _matrix_from_frame(stats, "condition_id", "sponsor_id", "active_span_plus", shape)
        role = self.events["lead_or_collaborator"].fillna("").astype(str).str.lower()
        for name, mask in {
            "lead": role.str.contains("lead"),
            "collaborator": role.str.contains("collaborator"),
        }.items():
            frame = self.events.loc[mask, ["condition_id", "sponsor_id"]].copy()
            frame["value"] = 1.0
            setattr(self, f"pair_{name}", _matrix_from_frame(frame, "condition_id", "sponsor_id", "value", shape))
        lead_values = np.exp2(-age / 365.0).astype(np.float32)
        lead_frame = self.events.loc[role.str.contains("lead"), ["condition_id", "sponsor_id"]].copy()
        lead_frame["value"] = lead_values[role.str.contains("lead").to_numpy()]
        self.pair_lead_decay = _matrix_from_frame(lead_frame, "condition_id", "sponsor_id", "value", shape)
        self.first_pairs = stats[["condition_id", "sponsor_id", "first_visible"]]

    def _activity(self, frame, entity_column):
        date_column = "date"
        output = []
        names = []
        for days in (180, 365, 730, 1825):
            mask = frame[date_column] > self.cutoff - pd.Timedelta(days=days)
            values = np.bincount(
                frame.loc[mask, entity_column].to_numpy(),
                minlength=self.n_conditions if entity_column == "condition_id" else self.n_sponsors,
            ).astype(np.float32)
            output.append(values)
            names.append(f"activity_{days}")
        current = frame[date_column] > self.cutoff - pd.Timedelta(days=180)
        previous = (frame[date_column] <= self.cutoff - pd.Timedelta(days=180)) & (
            frame[date_column] > self.cutoff - pd.Timedelta(days=360)
        )
        size = self.n_conditions if entity_column == "condition_id" else self.n_sponsors
        current_values = np.bincount(frame.loc[current, entity_column].to_numpy(), minlength=size).astype(np.float32)
        previous_values = np.bincount(frame.loc[previous, entity_column].to_numpy(), minlength=size).astype(np.float32)
        output.append(current_values - previous_values)
        names.append("acceleration_180")
        first = frame.groupby(entity_column, sort=False)[date_column].min()
        age = np.full(size, np.nan, dtype=np.float32)
        age[first.index.to_numpy()] = ((self.cutoff - first).dt.total_seconds() / 86400.0).to_numpy(dtype=np.float32)
        output.append(age)
        names.append("entity_age_days")
        return np.column_stack(output), names

    def _build_entity_features(self):
        condition_frame = self.condition_rel.rename(columns={"date": "date"})
        sponsor_frame = self.sponsor_rel.rename(columns={"date": "date"})
        condition_values, names = self._activity(condition_frame, "condition_id")
        sponsor_values, sponsor_names = self._activity(sponsor_frame, "sponsor_id")
        pair_unique = self.events[["condition_id", "sponsor_id"]].drop_duplicates()
        condition_breadth = pair_unique.groupby("condition_id", sort=False)["sponsor_id"].nunique()
        sponsor_breadth = pair_unique.groupby("sponsor_id", sort=False)["condition_id"].nunique()
        cb = np.zeros(self.n_conditions, dtype=np.float32)
        sb = np.zeros(self.n_sponsors, dtype=np.float32)
        cb[condition_breadth.index.to_numpy()] = condition_breadth.to_numpy(dtype=np.float32)
        sb[sponsor_breadth.index.to_numpy()] = sponsor_breadth.to_numpy(dtype=np.float32)
        recent_first = self.first_pairs["first_visible"] > self.cutoff - pd.Timedelta(days=365)
        condition_new = self.first_pairs.loc[recent_first].groupby("condition_id", sort=False).size()
        sponsor_new = self.first_pairs.loc[recent_first].groupby("sponsor_id", sort=False).size()
        cn = np.zeros(self.n_conditions, dtype=np.float32)
        sn = np.zeros(self.n_sponsors, dtype=np.float32)
        cn[condition_new.index.to_numpy()] = condition_new.to_numpy(dtype=np.float32)
        sn[sponsor_new.index.to_numpy()] = sponsor_new.to_numpy(dtype=np.float32)
        condition_values = np.column_stack([condition_values, cb, cn / np.maximum(cb, 1)])
        sponsor_values = np.column_stack([sponsor_values, sb, sn / np.maximum(sb, 1)])
        names += ["sponsor_breadth", "new_sponsor_rate"]
        sponsor_names += ["condition_breadth", "new_condition_rate"]
        roles = self.sponsor_rel["lead_or_collaborator"].fillna("").astype(str).str.lower().str.contains("lead")
        lead = np.bincount(
            self.sponsor_rel.loc[roles, "sponsor_id"].to_numpy(), minlength=self.n_sponsors
        ).astype(np.float32)
        all_roles = np.bincount(self.sponsor_rel["sponsor_id"].to_numpy(), minlength=self.n_sponsors).astype(np.float32)
        sponsor_values = np.column_stack([sponsor_values, lead / np.maximum(all_roles, 1)])
        sponsor_names += ["lead_share"]
        self.condition_entity = condition_values.astype(np.float32)
        self.sponsor_entity = sponsor_values.astype(np.float32)
        self.condition_entity_names = [f"condition_{x}" for x in names]
        self.sponsor_entity_names = [f"sponsor_{x}" for x in sponsor_names]
        d90 = np.asarray(self.pair_decay[90].sum(axis=0)).ravel()
        d365 = np.asarray(self.pair_decay[365].sum(axis=0)).ravel()
        acceleration = self.sponsor_entity[:, self.sponsor_entity_names.index("sponsor_acceleration_180")]
        self.sponsor_popularity = (d90 + 0.35 * d365 + 0.15 * np.maximum(acceleration, 0)).astype(np.float32)
        self.global_sponsors = _top(self.sponsor_popularity, 300)
        self.class_sponsors = []
        for code in np.unique(self.assets.agency_code):
            ids = np.flatnonzero(self.assets.agency_code == code)
            top = ids[_top(self.sponsor_popularity[ids], min(100, len(ids)))]
            self.class_sponsors.append(top)
        self.class_sponsors = np.unique(np.concatenate(self.class_sponsors))

    def _build_profiles(self):
        result_values, result_names = self.assets.result_features(self.cutoff)
        study_values = np.column_stack([self.assets.study_features, result_values]).astype(np.float32)
        self.profile_names = self.assets.study_feature_names + result_names

        def aggregate(matrix):
            counts = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
            values = matrix @ study_values
            values = np.asarray(values, dtype=np.float32) / np.maximum(counts[:, None], 1.0)
            values[counts == 0] = np.nan
            return values

        self.condition_profile = aggregate(self.condition_study)
        self.sponsor_profile = aggregate(self.sponsor_study)
        valid = self.assets.study_top_country >= 0
        country_count = int(self.assets.study_top_country[valid].max()) + 1 if valid.any() else 1
        study_country = sparse.coo_matrix(
            (
                np.ones(valid.sum(), dtype=np.float32),
                (np.flatnonzero(valid), self.assets.study_top_country[valid]),
            ),
            shape=(self.n_studies, country_count),
        ).tocsr()

        def top_country(matrix):
            counts = matrix @ study_country
            totals = np.asarray(counts.sum(axis=1)).ravel()
            values = np.asarray(counts.argmax(axis=1)).ravel().astype(np.int16)
            values[totals == 0] = -1
            return values

        self.condition_country = top_country(self.condition_study)
        self.sponsor_country = top_country(self.sponsor_study)

    def _build_interventions(self):
        relation = self.assets.tables["interventions_studies"]
        relation = relation[relation["date"] <= self.cutoff]
        study_intervention = sparse.coo_matrix(
            (
                np.ones(len(relation), dtype=np.float32),
                (relation["nct_id"].to_numpy(), relation["intervention_id"].to_numpy()),
            ),
            shape=(self.n_studies, len(self.assets.tables["interventions"])),
        ).tocsr()
        self.condition_intervention = _row_normalize(self.condition_study @ study_intervention)
        self.sponsor_intervention = _row_normalize(self.sponsor_study @ study_intervention)
        self.normalized_sponsor_study = _row_normalize(self.sponsor_study)

    def _build_als(self, factors):
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError:
            self.als = None
            self.user_factors = np.zeros((self.n_conditions, factors), dtype=np.float32)
            self.item_factors = np.zeros((self.n_sponsors, factors), dtype=np.float32)
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AlternatingLeastSquares(
                factors=factors,
                iterations=15 if not self.debug else 5,
                regularization=0.03,
                use_gpu=False,
                num_threads=int(os.environ.get("OMP_NUM_THREADS", "1")),
                random_state=1337,
            )
            model.fit((self.pair_decay[365] * 10.0).astype(np.float32), show_progress=False)
        self.als = model
        self.user_factors = np.nan_to_num(model.user_factors.astype(np.float32))
        self.item_factors = np.nan_to_num(model.item_factors.astype(np.float32))

    def retrieve(self, condition_ids, text_assets, cap=4000):
        unique = np.asarray(pd.unique(np.asarray(condition_ids, dtype=np.int64)), dtype=np.int64)
        similarity = (self.condition_study @ self.condition_study.T).tocsr().astype(np.float32)
        condition_norms = np.sqrt(np.asarray(self.condition_study.multiply(self.condition_study).sum(axis=1)).ravel())
        intervention_sponsor_t = self.sponsor_intervention.T.tocsr()
        sponsor_study_t = self.normalized_sponsor_study.T.tocsr()
        if self.als is not None:
            als_ids, als_scores = self.als.recommend(
                unique,
                self.pair_decay[365][unique],
                N=150,
                filter_already_liked_items=False,
            )
        else:
            als_ids = np.tile(self.global_sponsors[:150], (len(unique), 1))
            als_scores = np.zeros_like(als_ids, dtype=np.float32)
        semantic_ids, semantic_scores = semantic_topk(
            unique, self.query_embeddings, self.portfolio, 150
        ) if self.query_embeddings is not None else (None, None)
        records = {}
        popular_union = np.unique(np.concatenate([self.global_sponsors, self.class_sponsors]))
        for position, condition_id in enumerate(unique):
            row = similarity.getrow(int(condition_id))
            neighbor_ids = row.indices
            denominator = condition_norms[int(condition_id)] * np.maximum(condition_norms[neighbor_ids], 1e-8)
            cosine = row.data / np.maximum(denominator, 1e-8)
            selected = _top(cosine, min(41, len(cosine)))
            neighbor_ids = neighbor_ids[selected]
            cosine = cosine[selected]
            mask = neighbor_ids != condition_id
            neighbor_ids = neighbor_ids[mask][:40]
            cosine = cosine[mask][:40]
            if len(neighbor_ids):
                collab = (sparse.csr_matrix(cosine.reshape(1, -1)) @ self.pair_decay[365][neighbor_ids]).toarray().ravel()
                raw_weights = similarity.getrow(int(condition_id))[:, neighbor_ids].toarray().ravel()
                raw_collab = (sparse.csr_matrix(raw_weights.reshape(1, -1)) @ self.pair_count[neighbor_ids]).toarray().ravel()
            else:
                collab = np.zeros(self.n_sponsors, dtype=np.float32)
                raw_collab = np.zeros(self.n_sponsors, dtype=np.float32)
            bridge = (self.condition_intervention.getrow(int(condition_id)) @ intervention_sponsor_t).toarray().ravel()
            own = self.pair_decay[365].getrow(int(condition_id))
            if own.nnz:
                seed = sparse.csr_matrix(
                    (own.data / np.maximum(own.data.sum(), 1e-8), own.indices, [0, len(own.indices)]),
                    shape=(1, self.n_sponsors),
                )
                co_sponsor = ((seed @ self.normalized_sponsor_study) @ sponsor_study_t).toarray().ravel()
            else:
                co_sponsor = np.zeros(self.n_sponsors, dtype=np.float32)
            source_sets = [
                own.indices,
                _top(collab, min(2800, len(collab))),
                _top(raw_collab, min(800, len(raw_collab))),
                _top(bridge, min(500, len(bridge))),
                _top(co_sponsor, min(500, len(co_sponsor))),
                np.asarray(als_ids[position], dtype=np.int64),
                popular_union,
            ]
            if semantic_ids is not None:
                source_sets.append(np.asarray(semantic_ids[position], dtype=np.int64))
            candidates = np.unique(np.concatenate(source_sets)).astype(np.int64)
            direct_values = _extract(self.pair_decay[365], int(condition_id), candidates)
            priorities = (
                20.0 * (direct_values > 0)
                + 3.0 * collab[candidates] / max(float(collab.max()), 1e-8)
                + raw_collab[candidates] / max(float(raw_collab.max()), 1e-8)
                + bridge[candidates] / max(float(bridge.max()), 1e-8)
                + co_sponsor[candidates] / max(float(co_sponsor.max()), 1e-8)
                + 0.2 * self.sponsor_popularity[candidates] / max(float(self.sponsor_popularity.max()), 1e-8)
            )
            als_map = dict(zip(np.asarray(als_ids[position], dtype=np.int64), np.asarray(als_scores[position], dtype=np.float32)))
            if semantic_ids is not None:
                semantic_map = dict(zip(np.asarray(semantic_ids[position], dtype=np.int64), np.asarray(semantic_scores[position], dtype=np.float32)))
                priorities += np.array([0.8 if x in semantic_map else 0.0 for x in candidates], dtype=np.float32)
            else:
                semantic_map = {}
            priorities += np.array([0.8 if x in als_map else 0.0 for x in candidates], dtype=np.float32)
            if len(candidates) > cap:
                keep = _top(priorities, cap)
                candidates = candidates[keep]
                priorities = priorities[keep]
            order = np.argsort(-priorities, kind="stable")
            candidates = candidates[order]
            source_features = np.column_stack(
                [
                    collab[candidates],
                    raw_collab[candidates],
                    bridge[candidates],
                    co_sponsor[candidates],
                    np.array([als_map.get(int(x), 0.0) for x in candidates], dtype=np.float32),
                    np.array([semantic_map.get(int(x), 0.0) for x in candidates], dtype=np.float32),
                    self.sponsor_popularity[candidates],
                    np.array([1.0 if x in als_map else 0.0 for x in candidates], dtype=np.float32),
                    np.array([1.0 if x in semantic_map else 0.0 for x in candidates], dtype=np.float32),
                ]
            ).astype(np.float32)
            d90 = _extract(self.pair_decay[90], int(condition_id), candidates)
            d365 = _extract(self.pair_decay[365], int(condition_id), candidates)
            d1095 = _extract(self.pair_decay[1095], int(condition_id), candidates)
            direct_mix = 0.5 * d90 + 0.3 * d365 + 0.2 * d1095
            activity = self.sponsor_entity[
                candidates, self.sponsor_entity_names.index("sponsor_activity_365")
            ]
            condition_profile = np.nan_to_num(self.condition_profile[int(condition_id), 6:45])
            sponsor_profiles = np.nan_to_num(self.sponsor_profile[candidates, 6:45])
            profile_similarity = (sponsor_profiles @ condition_profile) / np.maximum(
                np.linalg.norm(sponsor_profiles, axis=1) * np.linalg.norm(condition_profile),
                1e-8,
            )
            components = np.column_stack(
                [
                    direct_mix,
                    source_features[:, 0],
                    source_features[:, 3],
                    source_features[:, 2],
                    source_features[:, 4],
                    source_features[:, 5],
                    source_features[:, 6],
                    activity,
                    profile_similarity,
                ]
            ).astype(np.float32)
            if self.portfolio is not None:
                query = np.asarray(self.query_embeddings[int(condition_id)], dtype=np.float32)
                components[:, 5] = np.asarray(self.portfolio[candidates], dtype=np.float32) @ query
                source_features[:, 5] = components[:, 5]
            records[int(condition_id)] = RetrievalRecord(int(condition_id), candidates, source_features, components)
        return records

    def feature_block(self, condition_id: int, candidates: np.ndarray, source_features: np.ndarray):
        pair = [
            _extract(self.pair_decay[90], condition_id, candidates),
            _extract(self.pair_decay[365], condition_id, candidates),
            _extract(self.pair_decay[1095], condition_id, candidates),
            _extract(self.pair_count, condition_id, candidates),
            _extract(self.pair_trials, condition_id, candidates),
            _extract(self.pair_days_since, condition_id, candidates) - 1.0,
            _extract(self.pair_active_span, condition_id, candidates) - 1.0,
            _extract(self.pair_lead, condition_id, candidates),
            _extract(self.pair_collaborator, condition_id, candidates),
            _extract(self.pair_lead_decay, condition_id, candidates),
        ]
        pair_values = np.column_stack(pair).astype(np.float32)
        absent = pair_values[:, 3] == 0
        pair_values[absent, 5:7] = np.nan
        sponsor_values = self.sponsor_entity[candidates]
        condition_values = np.repeat(self.condition_entity[condition_id][None, :], len(candidates), axis=0)
        sponsor_profile = self.sponsor_profile[candidates]
        condition_profile = np.repeat(self.condition_profile[condition_id][None, :], len(candidates), axis=0)
        profile_difference = np.abs(sponsor_profile - condition_profile)
        agency = self.assets.agency_code[candidates].astype(np.float32)[:, None]
        geography = (
            (self.sponsor_country[candidates] == self.condition_country[condition_id])
            & (self.sponsor_country[candidates] >= 0)
        ).astype(np.float32)[:, None]
        user = self.user_factors[condition_id]
        items = self.item_factors[candidates]
        dots = items @ user
        item_norms = np.linalg.norm(items, axis=1)
        user_norm = float(np.linalg.norm(user))
        als_values = np.column_stack(
            [dots, dots / np.maximum(item_norms * user_norm, 1e-8), item_norms, np.full(len(candidates), user_norm)]
        ).astype(np.float32)
        direct_mix = 0.5 * pair_values[:, 0] + 0.3 * pair_values[:, 1] + 0.2 * pair_values[:, 2]
        profile_slice = slice(6, 45)
        sponsor_profile_values = np.nan_to_num(sponsor_profile[:, profile_slice])
        condition_profile_values = np.nan_to_num(condition_profile[0, profile_slice])
        profile_similarity = (
            sponsor_profile_values @ condition_profile_values
        ) / np.maximum(
            np.linalg.norm(sponsor_profile_values, axis=1)
            * np.linalg.norm(condition_profile_values),
            1e-8,
        )
        prior = (
            10.0 * (pair_values[:, 3] > 0)
            + 5.0 * _normalize_component(direct_mix)
            + 1.5 * _normalize_component(source_features[:, 0])
            + 0.5 * _normalize_component(source_features[:, 3])
            + 0.4 * _normalize_component(source_features[:, 2])
            + 0.2 * _normalize_component(source_features[:, 4])
            + 0.2 * _normalize_component(source_features[:, 5])
            + 0.1 * _normalize_component(source_features[:, 6])
            + 0.2 * _normalize_component(sponsor_values[:, 1])
            + 0.2 * _normalize_component(profile_similarity)
        ).astype(np.float32)
        prior_order = np.argsort(-prior, kind="stable")
        prior_rank = np.empty(len(prior), dtype=np.float32)
        prior_rank[prior_order] = np.linspace(1.0, 0.0, len(prior), dtype=np.float32)
        values = np.column_stack(
            [
                pair_values,
                source_features,
                sponsor_values,
                condition_values,
                sponsor_profile,
                condition_profile,
                profile_difference,
                agency,
                geography,
                als_values,
                prior,
                prior_rank,
            ]
        ).astype(np.float32)
        missing = np.isnan(values).astype(np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return np.column_stack([values, missing]).astype(np.float32)

    def feature_names(self):
        base = [
            "pair_decay_90",
            "pair_decay_365",
            "pair_decay_1095",
            "pair_count",
            "pair_distinct_trials",
            "pair_days_since",
            "pair_active_span",
            "pair_lead_count",
            "pair_collaborator_count",
            "pair_lead_decay",
            "similar_condition_score",
            "similar_condition_raw_score",
            "intervention_bridge",
            "co_sponsor_score",
            "als_retrieval_score",
            "medcpt_similarity",
            "sponsor_popularity",
            "als_retrieved",
            "medcpt_retrieved",
        ]
        base += self.sponsor_entity_names
        base += self.condition_entity_names
        base += [f"sponsor_{x}" for x in self.profile_names]
        base += [f"condition_{x}" for x in self.profile_names]
        base += [f"profile_difference_{x}" for x in self.profile_names]
        base += [
            "agency_class",
            "geography_overlap",
            "als_dot",
            "als_cosine",
            "als_item_norm",
            "als_user_norm",
            "baseline_prior_score",
            "baseline_prior_rank",
        ]
        return base + [f"{x}_missing" for x in base]
