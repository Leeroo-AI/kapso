from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numba
import numpy as np
import torch


EPOCH_DAY = np.datetime64("1970-01-01", "D")


def origin_day(origin: str) -> int:
    return int((np.datetime64(origin, "D") - EPOCH_DAY).astype(np.int64))


@numba.njit(parallel=True, cache=True)
def locate_ends(offsets: np.ndarray, days: np.ndarray, cutoff: int) -> np.ndarray:
    customer_count = len(offsets) - 1
    ends = np.empty(customer_count, dtype=np.int64)
    for customer in numba.prange(customer_count):
        left = offsets[customer]
        right = offsets[customer + 1]
        while left < right:
            middle = (left + right) // 2
            if days[middle] <= cutoff:
                left = middle + 1
            else:
                right = middle
        ends[customer] = left
    return ends


def log_survival(logits: torch.Tensor, locations: torch.Tensor, scales: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    values = torch.clamp(values.float(), min=1e-4)
    z = (torch.log(values).unsqueeze(-1) - locations.float()) / scales.float()
    component = torch.clamp(0.5 * torch.erfc(z / math.sqrt(2.0)), min=1e-30)
    return torch.logsumexp(torch.log_softmax(logits.float(), dim=-1) + torch.log(component), dim=-1)


def log_density(logits: torch.Tensor, locations: torch.Tensor, scales: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    values = torch.clamp(values.float(), min=1e-4)
    log_values = torch.log(values).unsqueeze(-1)
    normal = -0.5 * torch.square((log_values - locations.float()) / scales.float())
    normal = normal - torch.log(scales.float()) - 0.5 * math.log(2.0 * math.pi) - log_values
    return torch.logsumexp(torch.log_softmax(logits.float(), dim=-1) + normal, dim=-1)


def conditional_intervals(
    logits: torch.Tensor,
    locations: torch.Tensor,
    scales: torch.Tensor,
    age: torch.Tensor,
) -> torch.Tensor:
    bounds = torch.tensor([0.0, 7.0, 30.0, 60.0, 91.0], device=age.device)
    survival = []
    for bound in bounds:
        survival.append(torch.exp(log_survival(logits, locations, scales, age + bound)))
    values = torch.stack(survival, dim=1)
    denominator = torch.clamp(values[:, :1], min=1e-12)
    probabilities = torch.cat((values[:, :-1] - values[:, 1:], values[:, -1:]), dim=1) / denominator
    probabilities = torch.clamp(probabilities, min=1e-8)
    return probabilities / probabilities.sum(dim=1, keepdim=True)


class MarkedIntensityFree(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.category = torch.nn.Embedding(1024, 8)
        self.brand_frequency = torch.nn.Embedding(32, 8)
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(64 + 32 + 11 + 8 + 8, 192),
            torch.nn.LayerNorm(192),
            torch.nn.GELU(),
        )
        self.encoder = torch.nn.GRU(
            192,
            192,
            num_layers=2,
            dropout=0.15,
            batch_first=True,
        )
        self.mixture_head = torch.nn.Linear(192, 48)
        with torch.no_grad():
            locations = torch.linspace(math.log(1.5), math.log(1400.0), 16)
            self.mixture_head.bias[16:32].copy_(locations)
            self.mixture_head.bias[32:48].fill_(0.25)

    def mixture(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = self.mixture_head(states)
        logits, locations, raw_scales = values.chunk(3, dim=-1)
        scales = torch.nn.functional.softplus(raw_scales.float()) + 0.05
        return logits.float(), locations.float(), scales

    def forward(
        self,
        review: torch.Tensor,
        product: torch.Tensor,
        numeric: torch.Tensor,
        category: torch.Tensor,
        brand_frequency: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        marks = torch.cat(
            (
                review,
                product,
                numeric,
                self.category(category),
                self.brand_frequency(brand_frequency),
            ),
            dim=-1,
        )
        encoded = self.projection(marks)
        output, final_hidden = self.encoder(encoded, hidden)
        return self.mixture(output), final_hidden


@dataclass
class AdvanceResult:
    origin: str
    events: int
    completed: int
    terminals: int
    loss: float
    seconds: float
    transitions_per_second: float


class SurvivalTrajectory:
    def __init__(
        self,
        events: dict[str, np.ndarray],
        source_index: dict[str, np.ndarray],
        sequence_length: int = 128,
    ) -> None:
        torch.manual_seed(1337)
        self.events = events
        self.source_index = source_index
        self.offsets = np.asarray(events["offsets"])
        self.days = np.asarray(events["day"])
        self.model = MarkedIntensityFree().cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=2e-4)
        self.hidden = np.zeros((2, len(self.offsets) - 1, 192), dtype=np.float16)
        self.ends = self.offsets[:-1].copy()
        self.sequence_length = sequence_length
        total_completed = np.maximum(np.diff(self.offsets) - 1, 1)
        self.customer_weight = np.minimum(1.0, 16.0 / total_completed).astype(np.float32)
        self.probed = False

    def _origin_index(self, cutoff: int) -> int:
        cutoffs = np.asarray(self.source_index["cutoffs"])
        position = int(np.searchsorted(cutoffs, cutoff))
        if position >= len(cutoffs) or int(cutoffs[position]) != cutoff:
            raise RuntimeError(f"No product-popularity snapshot for cutoff {cutoff}")
        return position

    def _batch_marks(
        self,
        positions: np.ndarray,
        valid: np.ndarray,
        cutoff: int,
        origin_index: int,
    ) -> tuple[torch.Tensor, ...]:
        products = np.asarray(self.events["product"])[positions]
        event_era = np.asarray(self.events["day"])[positions] - origin_day("2008-01-01")
        popularity = np.asarray(self.source_index["product_popularity"])[origin_index, products]
        numeric = np.stack(
            (
                np.asarray(self.events["rating_mean"])[positions] / 5.0,
                np.asarray(self.events["rating_std"])[positions] / 2.0,
                np.asarray(self.events["verified_share"])[positions],
                np.asarray(self.events["text_missing_share"])[positions],
                np.asarray(self.events["product_missing_share"])[positions],
                np.log1p(np.asarray(self.events["multiplicity"])[positions].astype(np.float32)) / 5.0,
                np.log1p(np.asarray(self.events["distinct_products"])[positions].astype(np.float32)) / 4.0,
                np.log1p(np.asarray(self.events["gap"])[positions].astype(np.float32)) / 8.0,
                np.log1p(popularity.astype(np.float32)) / 14.0,
                np.log1p(np.maximum(event_era, 0).astype(np.float32)) / 9.0,
                valid.astype(np.float32),
            ),
            axis=-1,
        ).astype(np.float32)
        return (
            torch.from_numpy(np.asarray(self.events["review_mark"])[positions].astype(np.float32)).cuda(non_blocking=True),
            torch.from_numpy(np.asarray(self.events["product_mark"])[positions].astype(np.float32)).cuda(non_blocking=True),
            torch.from_numpy(numeric).cuda(non_blocking=True),
            torch.from_numpy(np.asarray(self.events["category"])[positions].astype(np.int64)).cuda(non_blocking=True),
            torch.from_numpy(np.asarray(self.events["brand_frequency_bucket"])[positions].astype(np.int64)).cuda(non_blocking=True),
        )

    def _segment_batches(
        self,
        customers: np.ndarray,
        starts: np.ndarray,
        lengths: np.ndarray,
    ):
        for max_length in np.unique(lengths):
            selected = np.flatnonzero(lengths == max_length)
            max_length = int(max_length)
            batch_size = max(256, min(16_384, 65_536 // max_length))
            for start in range(0, len(selected), batch_size):
                rows = selected[start : start + batch_size]
                yield customers[rows], starts[rows], lengths[rows], max_length

    def advance(
        self,
        origin: str,
        completed_limit: int | None = None,
        terminal_limit: int | None = None,
    ) -> AdvanceResult:
        cutoff = origin_day(origin)
        origin_index = self._origin_index(cutoff)
        target_ends = locate_ends(self.offsets, self.days, cutoff)
        new_counts = target_ends - self.ends
        previous_counts = self.ends - self.offsets[:-1]
        completed_counts = new_counts - ((previous_counts == 0) & (new_counts > 0)).astype(np.int64)
        completed_counts = np.maximum(completed_counts, 0)
        customers = np.flatnonzero(new_counts > 0).astype(np.int64)
        if completed_limit is not None and completed_counts[customers].sum() > completed_limit:
            cumulative = np.cumsum(completed_counts[customers])
            keep = max(1, int(np.searchsorted(cumulative, completed_limit, side="left") + 1))
            customers = customers[:keep]
            selected_mask = np.zeros(len(new_counts), dtype=np.bool_)
            selected_mask[customers] = True
            target_ends = np.where(selected_mask, target_ends, self.ends)
            new_counts = target_ends - self.ends
            completed_counts = new_counts - ((previous_counts == 0) & (new_counts > 0)).astype(np.int64)
            completed_counts = np.maximum(completed_counts, 0)
        active = np.flatnonzero(target_ends > self.offsets[:-1]).astype(np.int64)
        if terminal_limit is not None:
            active = active[:terminal_limit]
        completed_mass = float(np.sum(completed_counts * self.customer_weight))
        terminal_mass = float(len(active))
        event_total = int(new_counts[customers].sum())
        completed_total = int(completed_counts[customers].sum())
        step_estimate = max(1, math.ceil(max(event_total, 1) / 65_536) + math.ceil(max(len(active), 1) / 65_536))
        started = time.time()
        loss_sum = 0.0
        processed_completed = 0
        self.model.train()
        current = self.ends[customers].copy()
        remaining = target_ends[customers] - current
        round_index = 0
        while np.any(remaining > 0):
            selected = np.flatnonzero(remaining > 0)
            round_customers = customers[selected]
            round_starts = current[selected]
            round_lengths = np.minimum(remaining[selected], self.sequence_length).astype(np.int64)
            for batch_customers, batch_starts, batch_lengths, max_length in self._segment_batches(
                round_customers,
                round_starts,
                round_lengths,
            ):
                positions = batch_starts[:, None] + np.arange(max_length, dtype=np.int64)[None, :]
                valid = positions < (batch_starts + batch_lengths)[:, None]
                positions = np.minimum(positions, (batch_starts + batch_lengths - 1)[:, None])
                initial = torch.from_numpy(
                    np.ascontiguousarray(self.hidden[:, batch_customers], dtype=np.float32)
                ).cuda(non_blocking=True)
                marks = self._batch_marks(positions, valid, cutoff, origin_index)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    mixture, final_hidden = self.model(*marks, initial)
                logits, locations, scales = mixture
                density_values = []
                density_weights = []
                if round_index == 0:
                    prior_exists = previous_counts[batch_customers] > 0
                    if np.any(prior_exists):
                        prior_rows = np.flatnonzero(prior_exists)
                        first_positions = batch_starts[prior_rows]
                        prior_positions = self.ends[batch_customers[prior_rows]] - 1
                        gaps = self.days[first_positions] - self.days[prior_positions]
                        prior_mixture = self.model.mixture(initial[-1, prior_rows])
                        density_values.append(-log_density(*prior_mixture, torch.from_numpy(gaps.astype(np.float32)).cuda()))
                        density_weights.append(torch.from_numpy(self.customer_weight[batch_customers[prior_rows]]).cuda())
                next_valid = valid & ((positions + 1) < target_ends[batch_customers, None])
                if np.any(next_valid):
                    gaps = self.days[positions[next_valid] + 1] - self.days[positions[next_valid]]
                    density_values.append(-log_density(
                        logits[next_valid],
                        locations[next_valid],
                        scales[next_valid],
                        torch.from_numpy(gaps.astype(np.float32)).cuda(),
                    ))
                    repeated_customers = np.broadcast_to(batch_customers[:, None], positions.shape)[next_valid]
                    density_weights.append(torch.from_numpy(self.customer_weight[repeated_customers]).cuda())
                if density_values:
                    values = torch.cat(density_values)
                    weights = torch.cat(density_weights)
                    loss = 0.5 * step_estimate * torch.sum(values * weights) / max(completed_mass, 1e-8)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    value = float(loss.detach())
                    loss_sum += value
                    processed_completed += len(values)
                self.hidden[:, batch_customers] = final_hidden.detach().float().cpu().numpy().astype(np.float16)
                if not self.probed and processed_completed >= 100_000:
                    rate = processed_completed / max(time.time() - started, 1e-6)
                    if rate < 25_000:
                        self.sequence_length = 64
                    self.probed = True
                    print(
                        f"[throughput_probe] transitions={processed_completed:,} rate={rate:.1f}/s required=25000 sequence_length={self.sequence_length} likelihood_passes=1",
                        flush=True,
                    )
            current[selected] += round_lengths
            remaining[selected] -= round_lengths
            round_index += 1
        terminal_batch = 65_536
        for start in range(0, len(active), terminal_batch):
            batch_customers = active[start : start + terminal_batch]
            final_positions = target_ends[batch_customers] - 1
            ages = cutoff - self.days[final_positions]
            states = torch.from_numpy(self.hidden[-1, batch_customers].astype(np.float32)).cuda(non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            mixture = self.model.mixture(states)
            losses = -log_survival(*mixture, torch.from_numpy(ages.astype(np.float32)).cuda())
            loss = 0.5 * step_estimate * losses.sum() / max(terminal_mass, 1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            loss_sum += float(loss.detach())
        self.ends = target_ends
        elapsed = time.time() - started
        rate = completed_total / max(elapsed, 1e-6)
        result = AdvanceResult(origin, event_total, completed_total, len(active), loss_sum / step_estimate, elapsed, rate)
        print(
            f"[likelihood] origin={origin} new_events={event_total:,} completed={completed_total:,} terminals={len(active):,} balanced_loss={result.loss:.6f} transitions_per_second={rate:.1f} elapsed={elapsed:.1f}s",
            flush=True,
        )
        return result

    def score(self, customer_ids: np.ndarray, origin: str, batch_size: int = 65_536) -> tuple[np.ndarray, np.ndarray]:
        cutoff = origin_day(origin)
        probabilities = np.empty((len(customer_ids), 5), dtype=np.float32)
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(customer_ids), batch_size):
                customers = customer_ids[start : start + batch_size].astype(np.int64)
                positions = self.ends[customers] - 1
                if np.any(positions < self.offsets[customers]):
                    raise RuntimeError(f"Missing event state while scoring {origin}")
                age = cutoff - self.days[positions]
                states = torch.from_numpy(self.hidden[-1, customers].astype(np.float32)).cuda(non_blocking=True)
                mixture = self.model.mixture(states)
                values = conditional_intervals(
                    *mixture,
                    torch.from_numpy(age.astype(np.float32)).cuda(non_blocking=True),
                )
                probabilities[start : start + len(customers)] = values.cpu().numpy()
        return probabilities[:, 4].copy(), probabilities
