import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker


NUM_CLASSES = 154
SHRINKAGES = (2.0, 5.0, 12.0)
BLENDS = (0.15, 0.30, 0.45)
HALF_LIVES = np.asarray([30.0, 90.0, 180.0], dtype=np.float32)
NS_DAY = 86_400_000_000_000

LEVELS = [
    ("payer_org_type", ("PAYERPARTY", "SALESORGANIZATION", "SALESDOCUMENTTYPE")),
    ("payer_org_channel", ("PAYERPARTY", "SALESORGANIZATION", "DISTRIBUTIONCHANNEL")),
    ("payer_org", ("PAYERPARTY", "SALESORGANIZATION")),
    ("payer_type", ("PAYERPARTY", "SALESDOCUMENTTYPE")),
    ("payer", ("PAYERPARTY",)),
    ("bill_org_type", ("BILLTOPARTY", "SALESORGANIZATION", "SALESDOCUMENTTYPE")),
    ("bill_org", ("BILLTOPARTY", "SALESORGANIZATION")),
    ("bill", ("BILLTOPARTY",)),
    ("sold_org_type", ("SOLDTOPARTY", "SALESORGANIZATION", "SALESDOCUMENTTYPE")),
    ("sold_org", ("SOLDTOPARTY", "SALESORGANIZATION")),
    ("sold", ("SOLDTOPARTY",)),
    ("ship_org_type", ("SHIPTOPARTY", "SALESORGANIZATION", "SALESDOCUMENTTYPE")),
    ("ship_org", ("SHIPTOPARTY", "SALESORGANIZATION")),
    ("ship", ("SHIPTOPARTY",)),
    ("payer_geo_org", ("PAYERPARTY_COUNTRY", "PAYERPARTY_REGION", "SALESORGANIZATION")),
    ("country_org_type", ("PAYERPARTY_COUNTRY", "SALESORGANIZATION", "SALESDOCUMENTTYPE")),
    ("global", ()),
]

PAYER_LEVEL = 4
COHORT_LEVEL = 15
GLOBAL_LEVEL = 16
POSTERIOR_LEVELS = (GLOBAL_LEVEL, COHORT_LEVEL, PAYER_LEVEL, 3, 2, 1, 0)
CONTEXT_COLUMNS = [
    "SALESDOCUMENTTYPE",
    "SALESORGANIZATION",
    "DISTRIBUTIONCHANNEL",
    "TRANSACTIONCURRENCY",
    "BILLINGCOMPANYCODE",
]

SEED_FEATURE_COLUMNS = [
    "SALESDOCUMENTTYPE",
    "SALESORGANIZATION",
    "DISTRIBUTIONCHANNEL",
    "ORGANIZATIONDIVISION",
    "BILLINGCOMPANYCODE",
    "TRANSACTIONCURRENCY",
    "ITEM_COUNT",
    "PRODUCT_DISTINCT",
    "CATEGORY_DISTINCT",
    "PAYERPARTY_DISTINCT",
    "BILLTOPARTY_DISTINCT",
    "SOLDTOPARTY_DISTINCT",
    "SHIPTOPARTY_DISTINCT",
    "PRODUCT_SET_HASH",
    "CATEGORY_SET_HASH",
    "CATEGORY_TOP1",
    "CATEGORY_TOP2",
    "CATEGORY_TOP3",
    "PRODUCT_TOP1",
    "PRODUCT_TOP2",
    "PRODUCT_TOP3",
    "CATEGORY_TOP_COUNT",
    "PRODUCT_TOP_COUNT",
    "CATEGORY_ENTROPY",
    "PRODUCT_ENTROPY",
    "PAYERPARTY_COUNTRY",
    "PAYERPARTY_REGION",
    "BILLTOPARTY_COUNTRY",
    "BILLTOPARTY_REGION",
    "SOLDTOPARTY_COUNTRY",
    "SOLDTOPARTY_REGION",
    "SHIPTOPARTY_COUNTRY",
    "SHIPTOPARTY_REGION",
    "PAYERPARTY_AMBIGUOUS",
    "BILLTOPARTY_AMBIGUOUS",
    "SOLDTOPARTY_AMBIGUOUS",
    "SHIPTOPARTY_AMBIGUOUS",
    "ROLE_COUNTRY_DISTINCT",
    "ROLE_REGION_DISTINCT",
    "CALENDAR_YEAR",
    "CALENDAR_MONTH",
    "CALENDAR_DAY",
    "CALENDAR_DOW",
    "CALENDAR_HOUR",
    "DAYS_FROM_2018",
    "CATEGORY_TOP_SHARE",
    "PRODUCT_TOP_SHARE",
]

COLD_CATEGORICAL = [
    "PAYERPARTY_COUNTRY",
    "PAYERPARTY_REGION",
    "SALESORGANIZATION",
    "SALESDOCUMENTTYPE",
    "DISTRIBUTIONCHANNEL",
    "BILLINGCOMPANYCODE",
    "TRANSACTIONCURRENCY",
    "ORGANIZATIONDIVISION",
    "CATEGORY_TOP1",
    "CATEGORY_TOP2",
    "CATEGORY_TOP3",
]

COLD_NUMERIC = [
    "ITEM_COUNT",
    "PRODUCT_DISTINCT",
    "CATEGORY_DISTINCT",
    "CATEGORY_TOP_COUNT",
    "PRODUCT_TOP_COUNT",
    "CATEGORY_ENTROPY",
    "PRODUCT_ENTROPY",
    "CATEGORY_TOP_SHARE",
    "PRODUCT_TOP_SHARE",
    "ROLE_COUNTRY_DISTINCT",
    "ROLE_REGION_DISTINCT",
    "CALENDAR_MONTH",
    "CALENDAR_DOW",
]


def level_keys(frame):
    n = len(frame)
    output = np.empty((n, len(LEVELS)), dtype=np.uint64)
    prime = np.uint64(1099511628211)
    offset = np.uint64(1469598103934665603)
    with np.errstate(over="ignore"):
        for level_index, (_, columns) in enumerate(LEVELS):
            if not columns:
                output[:, level_index] = 0
                continue
            values = np.full(n, offset, dtype=np.uint64)
            for column in columns:
                component = frame[column].to_numpy(dtype=np.int64, copy=False).view(np.uint64)
                values = (values ^ (component + np.uint64(11400714819323198485))) * prime
            output[:, level_index] = values
    return output


class FrozenHistory:
    def __init__(self, labeled):
        ordered = labeled.sort_values(["CREATIONTIMESTAMP", "SALESDOCUMENT"]).reset_index(drop=True)
        self.frame = ordered
        self.days = (
            ordered["CREATIONTIMESTAMP"].to_numpy(dtype="datetime64[ns]").astype(np.int64) // NS_DAY
        ).astype(np.int32)
        self.labels = ordered["CUSTOMERPAYMENTTERMS"].to_numpy(dtype=np.int16)
        self.keys = level_keys(ordered)
        self.context = ordered[CONTEXT_COLUMNS].to_numpy(dtype=np.int32)
        self.maps = []
        for level_index in range(len(LEVELS)):
            keys = self.keys[:, level_index]
            order = np.argsort(keys, kind="stable")
            sorted_keys = keys[order]
            starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
            ends = np.r_[starts[1:], len(order)]
            mapping = {
                int(sorted_keys[start]): order[start:end].astype(np.int32, copy=False)
                for start, end in zip(starts, ends)
            }
            self.maps.append(mapping)
        self.cache = [dict() for _ in LEVELS]
        self.cache_cutoff = None

    def clear_cache(self, cutoff_day):
        cutoff_day = int(cutoff_day)
        if cutoff_day != self.cache_cutoff:
            self.cache = [dict() for _ in LEVELS]
            self.cache_cutoff = cutoff_day

    def state(self, level_index, key, cutoff_day):
        cache = self.cache[level_index]
        key = int(key)
        existing = cache.get(key)
        if existing is not None:
            return existing
        indices = self.maps[level_index].get(key)
        if indices is None:
            state = (None, None, None, None, None, None, 0, -100000, None)
            cache[key] = state
            return state
        group_days = self.days[indices]
        position = int(np.searchsorted(group_days, int(cutoff_day), side="left"))
        if position == 0:
            state = (None, None, None, None, None, None, 0, -100000, indices[:0])
            cache[key] = state
            return state
        prefix = indices[:position]
        labels = self.labels[prefix]
        days = self.days[prefix]
        unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        last_days = np.full(len(unique), -100000, dtype=np.int32)
        np.maximum.at(last_days, inverse, days)
        ordering = np.argsort(-last_days, kind="stable")
        ranks = np.empty(len(unique), dtype=np.int16)
        ranks[ordering] = np.arange(1, len(unique) + 1, dtype=np.int16)
        ages = (int(cutoff_day) - days).astype(np.float32)
        decays = np.empty((len(unique), len(HALF_LIVES)), dtype=np.float32)
        for decay_index, half_life in enumerate(HALF_LIVES):
            weights = np.exp2(-ages / half_life)
            decays[:, decay_index] = np.bincount(inverse, weights=weights, minlength=len(unique))
        top_order = np.lexsort((-last_days, -counts))
        state = (
            unique.astype(np.int16, copy=False),
            counts.astype(np.int32, copy=False),
            last_days,
            decays,
            ranks,
            unique[ordering].astype(np.int16, copy=False),
            position,
            int(days[-1]),
            prefix,
            unique[top_order].astype(np.int16, copy=False),
        )
        cache[key] = state
        return state

    def states(self, keys, cutoff_day):
        return [self.state(level_index, keys[level_index], cutoff_day) for level_index in range(len(LEVELS))]

    def payer_dynamics(self, payer_state, candidate, cutoff_day, reference_day, seed_context):
        if payer_state[6] == 0:
            return [0.0] * 11
        indices = payer_state[8]
        labels = self.labels[indices]
        days = self.days[indices]
        last_label = int(labels[-1])
        run_length = 1
        while run_length < len(labels) and labels[-run_length - 1] == labels[-1]:
            run_length += 1
        changes = labels[1:] != labels[:-1]
        switch_count = int(changes.sum())
        switch_rate = switch_count / max(1, len(labels) - 1)
        probabilities = payer_state[1].astype(np.float64) / payer_state[6]
        entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
        if switch_count:
            last_switch_day = int(days[1:][changes][-1])
            days_since_switch = int(cutoff_day) - last_switch_day
        else:
            days_since_switch = int(cutoff_day) - int(days[0])
        source = labels[:-1] == last_label
        transition_total = int(source.sum())
        transition_probability = (
            float(((labels[1:] == candidate) & source).sum()) / transition_total
            if transition_total
            else 0.0
        )
        candidate_mask = labels == candidate
        if candidate_mask.any():
            contexts = self.context[indices[candidate_mask]]
            match_rates = (contexts == seed_context).mean(axis=0).astype(np.float32).tolist()
        else:
            match_rates = [0.0] * len(CONTEXT_COLUMNS)
        return [
            float(run_length),
            float(switch_rate),
            entropy,
            float(len(labels)),
            float(days_since_switch),
            float(transition_probability),
            *match_rates,
        ]


def state_candidate_features(state, candidate, reference_day, cutoff_day):
    if state[6] == 0:
        return [0.0, 99.0, 2000.0, 2000.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    labels = state[0]
    location = int(np.searchsorted(labels, candidate))
    cutoff_days_level = min(2000.0, float(int(cutoff_day) - state[7]))
    if location >= len(labels) or int(labels[location]) != candidate:
        return [0.0, 99.0, 2000.0, cutoff_days_level, 0.0, 0.0, 0.0, 0.0, 0.0]
    count = float(state[1][location])
    return [
        float(state[4][location] == 1),
        float(min(99, int(state[4][location]))),
        min(2000.0, float(int(cutoff_day) - int(state[2][location]))),
        cutoff_days_level,
        count,
        count / state[6],
        float(state[3][location, 0]),
        float(state[3][location, 1]),
        float(state[3][location, 2]),
    ]


def candidates_from_states(states):
    candidates = []
    seen = set()
    for level_index, state in enumerate(states):
        if state[6] == 0:
            continue
        source = state[9] if level_index in (COHORT_LEVEL, GLOBAL_LEVEL) else state[5]
        for label in source[:3]:
            value = int(label)
            if value not in seen:
                candidates.append(value)
                seen.add(value)
                if len(candidates) == 16:
                    return candidates
    return candidates


def posterior_from_states(states, shrinkage):
    posterior = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES, dtype=np.float32)
    for level_index in POSTERIOR_LEVELS:
        state = states[level_index]
        if state[6] == 0:
            continue
        denominator = float(state[6] + shrinkage)
        posterior *= shrinkage / denominator
        posterior[state[0].astype(np.intp)] += state[1].astype(np.float32) / denominator
    return posterior


def row_feature_vector(
    history,
    states,
    candidate,
    cutoff_day,
    seed_features,
    seed_context,
    seed_month,
    seed_day,
    posterior_values,
):
    features = [
        float(candidate) / (NUM_CLASSES - 1),
        float(seed_month),
        float(cutoff_day - states[GLOBAL_LEVEL][7]),
    ]
    for state in states:
        features.extend(state_candidate_features(state, candidate, seed_day, cutoff_day))
    features.extend(history.payer_dynamics(states[PAYER_LEVEL], candidate, cutoff_day, seed_day, seed_context))
    for posterior in posterior_values:
        features.append(float(posterior[candidate]))
    features.extend([float(seed_day - cutoff_day) / 30.4375, float(seed_day - states[GLOBAL_LEVEL][7])])
    features.extend(seed_features)
    return np.asarray(features, dtype=np.float32)


@dataclass
class RankerMatrix:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    seed_days: np.ndarray
    candidates: np.ndarray


@dataclass
class ServingBundle:
    posteriors: dict
    x: np.ndarray
    groups: np.ndarray
    group_seed_indices: np.ndarray
    candidates: np.ndarray
    payer_seen: np.ndarray
    labels: np.ndarray | None
    months: np.ndarray
    staleness: np.ndarray


def artificial_cutoffs(frame, debug):
    seed_months = frame["CREATIONTIMESTAMP"].to_numpy(dtype="datetime64[M]").astype(np.int64)
    if debug:
        cutoffs = np.full(len(frame), np.datetime64("2019-08-01", "D").astype(np.int64), dtype=np.int64)
    else:
        documents = frame["SALESDOCUMENT"].to_numpy(dtype=np.int64)
        gaps = ((documents * 1103515245 + 12345) % 6).astype(np.int64)
        cutoffs = (seed_months - gaps).astype("datetime64[M]").astype("datetime64[D]").astype(np.int64)
    return cutoffs.astype(np.int32)


def build_ranker_matrix(history, seeds, debug=False, debug_limit=20000):
    started = time.time()
    frame = seeds.copy()
    if debug:
        eligible = frame[
            (frame["CREATIONTIMESTAMP"] >= pd.Timestamp("2019-08-01"))
            & (frame["CREATIONTIMESTAMP"] < pd.Timestamp("2020-02-01"))
        ]
        if len(eligible) < debug_limit:
            eligible = frame
        positions = np.linspace(0, len(eligible) - 1, min(debug_limit, len(eligible)), dtype=np.int64)
        frame = eligible.iloc[positions].copy()
    frame = frame.sort_values(["CREATIONTIMESTAMP", "SALESDOCUMENT"]).reset_index(drop=True)
    keys = level_keys(frame)
    cutoffs = artificial_cutoffs(frame, debug)
    seed_days_all = (
        frame["CREATIONTIMESTAMP"].to_numpy(dtype="datetime64[ns]").astype(np.int64) // NS_DAY
    ).astype(np.int32)
    seed_features_all = frame[SEED_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    seed_context_all = frame[CONTEXT_COLUMNS].to_numpy(dtype=np.int32)
    seed_month_all = frame["CREATIONTIMESTAMP"].dt.month.to_numpy(dtype=np.int16)
    labels_all = frame["CUSTOMERPAYMENTTERMS"].to_numpy(dtype=np.int16)
    blocks = []
    targets = []
    candidates_flat = []
    groups = []
    seed_days = []
    for cutoff_day in np.unique(cutoffs):
        history.clear_cache(int(cutoff_day))
        row_indices = np.flatnonzero(cutoffs == cutoff_day)
        rows = []
        row_targets = []
        row_candidates = []
        for row_index in row_indices:
            states = history.states(keys[row_index], int(cutoff_day))
            candidates = candidates_from_states(states)
            target = int(labels_all[row_index])
            if target not in candidates:
                continue
            posterior_values = [posterior_from_states(states, shrinkage) for shrinkage in SHRINKAGES]
            for candidate in candidates:
                rows.append(
                    row_feature_vector(
                        history,
                        states,
                        candidate,
                        int(cutoff_day),
                        seed_features_all[row_index],
                        seed_context_all[row_index],
                        int(seed_month_all[row_index]),
                        int(seed_days_all[row_index]),
                        posterior_values,
                    )
                )
                row_targets.append(int(candidate == target))
                row_candidates.append(candidate)
            groups.append(len(candidates))
            seed_days.append(seed_days_all[row_index])
        if rows:
            blocks.append(np.vstack(rows))
            targets.append(np.asarray(row_targets, dtype=np.int8))
            candidates_flat.append(np.asarray(row_candidates, dtype=np.int16))
    if not blocks:
        raise RuntimeError("no covered synthetic ranker episodes")
    matrix = RankerMatrix(
        np.vstack(blocks),
        np.concatenate(targets),
        np.asarray(groups, dtype=np.int32),
        np.asarray(seed_days, dtype=np.int32),
        np.concatenate(candidates_flat),
    )
    print(
        f"[phase] episodes seeds={len(frame)} covered={len(matrix.groups)} rows={len(matrix.y)} "
        f"features={matrix.x.shape[1]} seconds={time.time() - started:.2f}"
    )
    return matrix


def subset_ranker_matrix(matrix, before):
    cutoff_day = int(np.datetime64(before, "D").astype(np.int64))
    group_mask = matrix.seed_days < cutoff_day
    row_mask = np.repeat(group_mask, matrix.groups)
    return RankerMatrix(
        matrix.x[row_mask],
        matrix.y[row_mask],
        matrix.groups[group_mask],
        matrix.seed_days[group_mask],
        matrix.candidates[row_mask],
    )


def bundle_eval_matrix(bundle):
    row_masks = []
    groups = []
    targets = []
    seed_days = []
    offsets = np.r_[0, np.cumsum(bundle.groups)]
    for group_index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        seed_index = int(bundle.group_seed_indices[group_index])
        target = int(bundle.labels[seed_index])
        candidates = bundle.candidates[start:end]
        if target not in candidates:
            continue
        row_masks.extend(range(start, end))
        targets.extend((candidates == target).astype(np.int8).tolist())
        groups.append(end - start)
        seed_days.append(0)
    row_masks = np.asarray(row_masks, dtype=np.int64)
    return RankerMatrix(
        bundle.x[row_masks],
        np.asarray(targets, dtype=np.int8),
        np.asarray(groups, dtype=np.int32),
        np.asarray(seed_days, dtype=np.int32),
        bundle.candidates[row_masks],
    )


def combine_ranker_matrices(first, second):
    return RankerMatrix(
        np.vstack([first.x, second.x]),
        np.concatenate([first.y, second.y]),
        np.concatenate([first.groups, second.groups]),
        np.concatenate([first.seed_days, second.seed_days]),
        np.concatenate([first.candidates, second.candidates]),
    )


def save_ranker_matrix(matrix, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "x.npy", matrix.x)
    np.save(directory / "y.npy", matrix.y)
    np.save(directory / "groups.npy", matrix.groups)
    np.save(directory / "seed_days.npy", matrix.seed_days)
    np.save(directory / "candidates.npy", matrix.candidates)
    (directory / "complete").write_text("complete\n")


def load_ranker_matrix(directory):
    directory = Path(directory)
    if not (directory / "complete").exists():
        return None
    return RankerMatrix(
        np.load(directory / "x.npy", allow_pickle=False),
        np.load(directory / "y.npy", allow_pickle=False),
        np.load(directory / "groups.npy", allow_pickle=False),
        np.load(directory / "seed_days.npy", allow_pickle=False),
        np.load(directory / "candidates.npy", allow_pickle=False),
    )


def group_top1(y_true, y_pred, weight, group):
    offsets = np.r_[0, np.cumsum(group)]
    correct = 0
    for start, end in zip(offsets[:-1], offsets[1:]):
        correct += int(y_true[start + int(np.argmax(y_pred[start:end]))] > 0.5)
    return "group_top1_accuracy", correct / len(group), True


def fit_ranker(train_matrix, eval_matrix=None, debug=False, num_boost_round=None):
    started = time.time()
    trees = 80 if debug else 1800
    if num_boost_round is not None:
        trees = max(20, min(trees, int(num_boost_round)))
    model = LGBMRanker(
        objective="lambdarank",
        metric="None",
        num_leaves=127,
        learning_rate=0.04,
        n_estimators=trees,
        min_child_samples=100,
        reg_lambda=10.0,
        reg_alpha=0.5,
        feature_fraction=0.85,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_jobs=11,
        random_state=1337,
        verbosity=-1,
    )
    kwargs = {}
    if eval_matrix is not None and len(eval_matrix.groups):
        kwargs = {
            "eval_set": [(eval_matrix.x, eval_matrix.y)],
            "eval_group": [eval_matrix.groups],
            "eval_metric": group_top1,
            "callbacks": [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        }
    model.fit(train_matrix.x, train_matrix.y, group=train_matrix.groups, **kwargs)
    best = int(model.best_iteration_ or trees)
    print(
        f"[phase] ranker_fit groups={len(train_matrix.groups)} rows={len(train_matrix.y)} "
        f"trees={best} seconds={time.time() - started:.2f}"
    )
    return model, best


def build_serving_bundle(history, seeds, cutoff, labels_available=True):
    started = time.time()
    frame = seeds.sort_values("_row_index").reset_index(drop=True).copy()
    cutoff_day = int(np.datetime64(cutoff, "D").astype(np.int64))
    history.clear_cache(cutoff_day)
    keys = level_keys(frame)
    seed_features = frame[SEED_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    seed_context = frame[CONTEXT_COLUMNS].to_numpy(dtype=np.int32)
    seed_month = frame["CREATIONTIMESTAMP"].dt.month.to_numpy(dtype=np.int16)
    seed_days = (
        frame["CREATIONTIMESTAMP"].to_numpy(dtype="datetime64[ns]").astype(np.int64) // NS_DAY
    ).astype(np.int32)
    posteriors = {shrinkage: np.empty((len(frame), NUM_CLASSES), dtype=np.float32) for shrinkage in SHRINKAGES}
    rows = []
    groups = []
    group_seed_indices = []
    candidates_flat = []
    payer_seen = np.zeros(len(frame), dtype=bool)
    staleness = np.full(len(frame), 2000, dtype=np.int32)
    for row_index in range(len(frame)):
        states = history.states(keys[row_index], cutoff_day)
        posterior_values = []
        for shrinkage in SHRINKAGES:
            posterior = posterior_from_states(states, shrinkage)
            posteriors[shrinkage][row_index] = posterior
            posterior_values.append(posterior)
        payer_seen[row_index] = states[PAYER_LEVEL][6] > 0
        if states[PAYER_LEVEL][6] > 0:
            staleness[row_index] = seed_days[row_index] - states[PAYER_LEVEL][7]
        if not payer_seen[row_index]:
            continue
        candidates = candidates_from_states(states)
        if not candidates:
            continue
        for candidate in candidates:
            rows.append(
                row_feature_vector(
                    history,
                    states,
                    candidate,
                    cutoff_day,
                    seed_features[row_index],
                    seed_context[row_index],
                    int(seed_month[row_index]),
                    int(seed_days[row_index]),
                    posterior_values,
                )
            )
            candidates_flat.append(candidate)
        groups.append(len(candidates))
        group_seed_indices.append(row_index)
    x = np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)
    labels = (
        frame["CUSTOMERPAYMENTTERMS"].to_numpy(dtype=np.int16)
        if labels_available and "CUSTOMERPAYMENTTERMS" in frame
        else None
    )
    bundle = ServingBundle(
        posteriors,
        x,
        np.asarray(groups, dtype=np.int32),
        np.asarray(group_seed_indices, dtype=np.int32),
        np.asarray(candidates_flat, dtype=np.int16),
        payer_seen,
        labels,
        seed_month,
        staleness,
    )
    print(
        f"[phase] serving cutoff={cutoff} seeds={len(frame)} candidate_rows={len(candidates_flat)} "
        f"payer_seen={int(payer_seen.sum())} seconds={time.time() - started:.2f}"
    )
    return bundle


def cold_frame(frame):
    output = frame[COLD_CATEGORICAL + COLD_NUMERIC].copy()
    for column in COLD_CATEGORICAL:
        output[column] = output[column].astype("category")
    return output


def fit_cold_classifier(labeled, debug=False, trees=None):
    started = time.time()
    first = labeled.sort_values(["CREATIONTIMESTAMP", "SALESDOCUMENT"]).drop_duplicates("PAYERPARTY", keep="first")
    model = LGBMClassifier(
        objective="multiclass",
        num_leaves=63,
        learning_rate=0.06,
        n_estimators=int(trees or (80 if debug else 240)),
        min_child_samples=20,
        reg_lambda=5.0,
        n_jobs=11,
        random_state=1337,
        verbosity=-1,
    )
    model.fit(cold_frame(first), first["CUSTOMERPAYMENTTERMS"].astype(np.int16))
    print(
        f"[phase] cold_fit first_orders={len(first)} classes={len(model.classes_)} "
        f"seconds={time.time() - started:.2f}"
    )
    return model


def cold_probabilities(model, frame):
    raw = model.predict_proba(cold_frame(frame))
    output = np.zeros((len(frame), NUM_CLASSES), dtype=np.float32)
    output[:, model.classes_.astype(np.intp)] = raw.astype(np.float32)
    return output


def ranker_predictions(model, bundle):
    if len(bundle.x) == 0:
        return np.empty(0, dtype=np.float32)
    return model.predict(bundle.x).astype(np.float32)


def assemble_scores(bundle, rank_scores, cold_scores, shrinkage, blend):
    scores = bundle.posteriors[float(shrinkage)].copy()
    offsets = np.r_[0, np.cumsum(bundle.groups)]
    for group_index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        seed_index = int(bundle.group_seed_indices[group_index])
        values = rank_scores[start:end]
        values = np.exp(values - values.max())
        values /= values.sum()
        scores[seed_index] *= blend
        labels = bundle.candidates[start:end].astype(np.intp)
        scores[seed_index, labels] += (1.0 - blend) * values
    cold_mask = ~bundle.payer_seen
    if cold_scores is not None and cold_mask.any():
        scores[cold_mask] = cold_scores[cold_mask]
    return scores.astype(np.float32, copy=False)


def evaluate_bundle(bundle, scores, name):
    labels = bundle.labels
    predictions = scores.argmax(axis=1)
    accuracy = float((predictions == labels).mean())
    print(f"[internal] {name} count={len(labels)} accuracy={accuracy:.6f}")
    for month in np.unique(bundle.months):
        mask = bundle.months == month
        value = float((predictions[mask] == labels[mask]).mean())
        print(f"[internal] {name} month={int(month)} count={int(mask.sum())} accuracy={value:.6f}")
    for seen in (False, True):
        mask = bundle.payer_seen == seen
        if mask.any():
            value = float((predictions[mask] == labels[mask]).mean())
            print(f"[internal] {name} payer_seen={int(seen)} count={int(mask.sum())} accuracy={value:.6f}")
    buckets = [(-1, 7, "lt7"), (7, 30, "7_30"), (30, 90, "30_90"), (90, 180, "90_180"), (180, 100000, "ge180")]
    monthly = []
    for lower, upper, label in buckets:
        mask = bundle.payer_seen & (bundle.staleness >= lower) & (bundle.staleness < upper)
        if mask.any():
            value = float((predictions[mask] == labels[mask]).mean())
            print(f"[internal] {name} staleness={label} count={int(mask.sum())} accuracy={value:.6f}")
    for month in np.unique(bundle.months):
        mask = bundle.months == month
        monthly.append(float((predictions[mask] == labels[mask]).mean()))
    return accuracy, monthly


def select_design(fold_results):
    candidates = []
    baseline_candidates = []
    for shrinkage in SHRINKAGES:
        fold_accuracies = []
        months = []
        for bundle, rank_scores, cold_scores in fold_results:
            scores = assemble_scores(bundle, rank_scores, cold_scores, shrinkage, 1.0)
            predictions = scores.argmax(axis=1)
            fold_accuracies.append(float((predictions == bundle.labels).mean()))
            for month in np.unique(bundle.months):
                mask = bundle.months == month
                months.append(float((predictions[mask] == bundle.labels[mask]).mean()))
        baseline_candidates.append((float(np.mean(fold_accuracies)), float(np.min(months)), -shrinkage, shrinkage, fold_accuracies))
        print(
            f"[internal] posterior shrinkage={shrinkage:g} mean={np.mean(fold_accuracies):.6f} "
            f"worst_month={np.min(months):.6f}"
        )
    baseline = max(baseline_candidates)
    for shrinkage in SHRINKAGES:
        for blend in BLENDS:
            accuracies = []
            months = []
            for bundle, rank_scores, cold_scores in fold_results:
                scores = assemble_scores(bundle, rank_scores, cold_scores, shrinkage, blend)
                predictions = scores.argmax(axis=1)
                accuracies.append(float((predictions == bundle.labels).mean()))
                for month in np.unique(bundle.months):
                    mask = bundle.months == month
                    months.append(float((predictions[mask] == bundle.labels[mask]).mean()))
            mean_accuracy = float(np.mean(accuracies))
            worst_month = float(np.min(months))
            candidates.append((mean_accuracy, worst_month, -blend, -shrinkage, shrinkage, blend, accuracies))
            print(
                f"[internal] grid shrinkage={shrinkage:g} blend={blend:.2f} "
                f"mean={mean_accuracy:.6f} worst_month={worst_month:.6f}"
            )
    best = max(candidates)
    stable = all(value >= reference for value, reference in zip(best[6], baseline[4]))
    if best[0] <= baseline[0] or not stable:
        print(f"[internal] ranker gate rejected stable={int(stable)}; selected posterior shrinkage={baseline[3]:g}")
        return float(baseline[3]), 1.0
    print(f"[internal] selected shrinkage={best[4]:g} blend={best[5]:.2f}")
    return float(best[4]), float(best[5])


def release(*objects):
    for item in objects:
        del item
    gc.collect()
