from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PriorSpecification:
    strength: float
    minimum_support: int = 0


SPECIFICATIONS = {
    "lead_sponsor": PriorSpecification(20),
    "collaborator_sponsor": PriorSpecification(20),
    "agency_class": PriorSpecification(20),
    "source_class": PriorSpecification(20),
    "condition": PriorSpecification(40),
    "intervention": PriorSpecification(40),
    "facility": PriorSpecification(80),
    "country": PriorSpecification(80),
    "state": PriorSpecification(80),
    "sponsor_phase": PriorSpecification(50, 3),
    "sponsor_condition": PriorSpecification(50, 3),
    "agency_phase": PriorSpecification(50, 3),
    "condition_intervention": PriorSpecification(50, 3),
}


class CausalEmpiricalBayes:
    def __init__(self, groups: dict[str, list[list[Any]]]):
        self.groups = groups
        self.counts = {name: defaultdict(int) for name in SPECIFICATIONS}
        self.positives = {name: defaultdict(float) for name in SPECIFICATIONS}
        self.total = 0
        self.positive_total = 0.0
        self.feature_names = ["eb_global_rate", "eb_global_support"]
        for name in SPECIFICATIONS:
            self.feature_names.extend([
                f"eb_{name}_mean", f"eb_{name}_max", f"eb_{name}_min",
                f"eb_{name}_pooled", f"eb_{name}_support_sum", f"eb_{name}_support_max",
                f"eb_{name}_seen_fraction",
            ])

    def _row(self, index: int) -> np.ndarray:
        global_rate = self.positive_total / self.total if self.total else 0.5
        values = [global_rate, np.log1p(self.total)]
        for name, specification in SPECIFICATIONS.items():
            keys = list(dict.fromkeys(self.groups[name][index]))
            supports = np.asarray([self.counts[name][key] for key in keys], dtype=np.float64)
            positives = np.asarray([self.positives[name][key] for key in keys], dtype=np.float64)
            if specification.minimum_support:
                admitted = supports >= specification.minimum_support
                supports = supports[admitted]
                positives = positives[admitted]
            if len(supports) == 0:
                values.extend([global_rate, global_rate, global_rate, global_rate, 0.0, 0.0, 0.0])
                continue
            rates = (positives + specification.strength * global_rate) / (supports + specification.strength)
            pooled = (positives.sum() + specification.strength * global_rate) / (supports.sum() + specification.strength)
            seen_fraction = float(np.count_nonzero(supports) / max(len(keys), 1))
            values.extend([
                float(rates.mean()), float(rates.max()), float(rates.min()), float(pooled),
                float(np.log1p(supports.sum())), float(np.log1p(supports.max())), seen_fraction,
            ])
        return np.asarray(values, dtype=np.float32)

    def _update(self, index: int, label: float) -> None:
        for name in SPECIFICATIONS:
            for key in dict.fromkeys(self.groups[name][index]):
                self.counts[name][key] += 1
                self.positives[name][key] += label
        self.total += 1
        self.positive_total += label

    def process_causal(self, indices: np.ndarray, dates: np.ndarray, labels: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        dates = np.asarray(dates)
        labels = np.asarray(labels, dtype=np.float32)
        order = np.argsort(dates, kind="stable")
        output = np.empty((len(indices), len(self.feature_names)), dtype=np.float32)
        cursor = 0
        while cursor < len(order):
            end = cursor + 1
            while end < len(order) and dates[order[end]] == dates[order[cursor]]:
                end += 1
            batch = order[cursor:end]
            for position in batch:
                output[position] = self._row(int(indices[position]))
            for position in batch:
                self._update(int(indices[position]), float(labels[position]))
            cursor = end
        return output

    def transform(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        return np.vstack([self._row(int(index)) for index in indices]).astype(np.float32)
