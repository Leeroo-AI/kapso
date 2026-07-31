from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


class ArticleEncoder(nn.Module):
    def __init__(
        self,
        n_items: int,
        item_features: np.ndarray,
        item_cardinalities: list[int],
        popularity_buckets: np.ndarray,
        price_buckets: np.ndarray,
        dimension: int,
    ):
        super().__init__()
        self.n_items = n_items
        self.id_embedding = nn.Embedding(n_items + 1, dimension, padding_idx=n_items)
        self.feature_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, dimension) for cardinality in item_cardinalities]
        )
        self.popularity_embedding = nn.Embedding(16, dimension)
        self.price_embedding = nn.Embedding(32, dimension)
        padded = np.vstack(
            [item_features.astype(np.int32), np.zeros((1, item_features.shape[1]), dtype=np.int32)]
        )
        self.register_buffer(
            "item_features", torch.from_numpy(padded), persistent=False
        )
        self.register_buffer(
            "popularity_buckets",
            torch.from_numpy(popularity_buckets),
            persistent=False,
        )
        self.register_buffer(
            "price_buckets", torch.from_numpy(price_buckets), persistent=False
        )
        self.normalization = nn.LayerNorm(dimension)

    def forward(self, item_ids: torch.Tensor, origin: torch.Tensor | int):
        valid = item_ids != self.n_items
        safe_ids = item_ids.clamp_max(self.n_items - 1)
        vector = self.id_embedding(item_ids)
        metadata = self.item_features[item_ids]
        for column, embedding in enumerate(self.feature_embeddings):
            vector = vector + embedding(metadata[..., column])
        if isinstance(origin, int):
            popularity = self.popularity_buckets[origin, safe_ids].long()
            price = self.price_buckets[origin, safe_ids].long()
        elif origin.ndim == 0:
            popularity = self.popularity_buckets[origin, safe_ids].long()
            price = self.price_buckets[origin, safe_ids].long()
        else:
            index = origin
            while index.ndim < safe_ids.ndim:
                index = index.unsqueeze(-1)
            popularity = self.popularity_buckets[index, safe_ids].long()
            price = self.price_buckets[index, safe_ids].long()
        vector = vector + self.popularity_embedding(popularity)
        vector = vector + self.price_embedding(price)
        vector = self.normalization(vector / math.sqrt(len(self.feature_embeddings) + 3))
        return vector * valid.unsqueeze(-1)


class BasketTransformer(nn.Module):
    def __init__(
        self,
        n_items: int,
        item_features: np.ndarray,
        item_cardinalities: list[int],
        customer_cardinalities: list[int],
        popularity_buckets: np.ndarray,
        price_buckets: np.ndarray,
        dimension: int = 128,
        layers: int = 3,
        heads: int = 4,
        feedforward: int = 512,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_items = n_items
        self.dimension = dimension
        self.article_encoder = ArticleEncoder(
            n_items,
            item_features,
            item_cardinalities,
            popularity_buckets,
            price_buckets,
            dimension,
        )
        self.customer_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, dimension) for cardinality in customer_cardinalities]
        )
        self.channel_embedding = nn.Embedding(3, dimension)
        self.position_embedding = nn.Embedding(32, dimension)
        self.gap_embedding = nn.Embedding(64, dimension)
        self.numeric_projection = nn.Sequential(
            nn.Linear(5, dimension),
            nn.GELU(),
            nn.LayerNorm(dimension),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, dimension))
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.sequence_normalization = nn.LayerNorm(dimension)
        self.repeat_head = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(dimension + 5, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def encode(self, batch):
        history = batch["history"]
        origin = batch["origin"].long()
        valid = batch["valid"]
        item_vector = self.article_encoder(history, origin)
        item_vector = item_vector + self.channel_embedding(batch["channel"].long()) * valid.unsqueeze(-1)
        basket_index = batch["basket"].long()
        shape = (history.shape[0], 32, self.dimension)
        basket_vector = item_vector.new_zeros(shape)
        basket_vector.scatter_add_(
            1, basket_index.unsqueeze(-1).expand(-1, -1, self.dimension), item_vector
        )
        basket_count = item_vector.new_zeros((history.shape[0], 32))
        basket_count.scatter_add_(1, basket_index, valid.to(item_vector.dtype))
        basket_vector = basket_vector / basket_count.clamp_min(1).unsqueeze(-1)
        gap = (
            batch["query_day"].long().unsqueeze(1) - batch["history_day"].long()
        ).clamp(0, 63)
        basket_gap = item_vector.new_zeros((history.shape[0], 32))
        basket_gap.scatter_add_(1, basket_index, gap.to(item_vector.dtype) * valid)
        basket_gap = (basket_gap / basket_count.clamp_min(1)).long().clamp(0, 63)
        positions = torch.arange(32, device=history.device)
        basket_vector = (
            basket_vector
            + self.position_embedding(positions).unsqueeze(0)
            + self.gap_embedding(basket_gap)
        )
        customer_vector = basket_vector.new_zeros((history.shape[0], self.dimension))
        for column, embedding in enumerate(self.customer_embeddings):
            customer_vector = customer_vector + embedding(
                batch["customer_features"][:, column].long()
            )
        numeric = batch["context_numeric"].to(basket_vector.dtype)
        customer_vector = customer_vector / math.sqrt(len(self.customer_embeddings))
        cls = self.cls.expand(history.shape[0], -1, -1)
        cls = cls + customer_vector.unsqueeze(1) + self.numeric_projection(numeric).unsqueeze(1)
        sequence_input = torch.cat([cls, basket_vector], dim=1)
        basket_valid = basket_count > 0
        padding_mask = torch.cat(
            [
                torch.zeros((history.shape[0], 1), dtype=torch.bool, device=history.device),
                ~basket_valid,
            ],
            dim=1,
        )
        encoded = self.transformer(sequence_input, src_key_padding_mask=padding_mask)
        sequence = self.sequence_normalization(encoded[:, 0])
        gate = torch.sigmoid(self.gate(torch.cat([sequence, numeric], dim=1)))
        return sequence, encoded[:, 1:], gate

    def score_candidates(self, batch, sequence, baskets, gate, candidates):
        origin = batch["origin"].long()
        candidate_vector = self.article_encoder(candidates, origin)
        explore = torch.einsum("bd,bcd->bc", sequence, candidate_vector) / math.sqrt(
            self.dimension
        )
        valid = batch["valid"].unsqueeze(1)
        history = batch["history"].unsqueeze(1)
        match = (candidates.unsqueeze(-1) == history) & valid
        frequency = match.sum(-1).to(explore.dtype)
        history_meta = self.article_encoder.item_features[batch["history"]][..., 0]
        candidate_meta = self.article_encoder.item_features[candidates][..., 0]
        product_match = (
            candidate_meta.unsqueeze(-1) == history_meta.unsqueeze(1)
        ) & valid
        product_frequency = product_match.sum(-1).to(explore.dtype)
        day_gap = (
            batch["query_day"].long().unsqueeze(1) - batch["history_day"].long()
        ).clamp(0, 365)
        gap_values = day_gap.unsqueeze(1).expand_as(match)
        recency = torch.where(match, gap_values, torch.full_like(gap_values, 366)).amin(-1)
        occurrence_day = batch["history_day"].long().unsqueeze(1).expand_as(match)
        minimum_day = torch.where(
            match, occurrence_day, torch.full_like(occurrence_day, 32767)
        ).amin(-1)
        maximum_day = torch.where(
            match, occurrence_day, torch.full_like(occurrence_day, -32768)
        ).amax(-1)
        interval = torch.where(
            frequency > 1,
            (maximum_day - minimum_day).to(explore.dtype) / (frequency - 1).clamp_min(1),
            torch.zeros_like(frequency),
        )
        attention = torch.einsum("bcd,bsd->bcs", candidate_vector, baskets)
        basket_valid = batch["valid"].new_zeros((batch["valid"].shape[0], 32))
        basket_valid.scatter_(1, batch["basket"].long(), batch["valid"])
        attention = attention.masked_fill(~basket_valid.unsqueeze(1), -20.0)
        attention = attention.amax(-1) / math.sqrt(self.dimension)
        features = torch.stack(
            [
                attention,
                torch.log1p(frequency),
                torch.exp(-recency.to(explore.dtype) / 30.0),
                torch.log1p(product_frequency),
                torch.log1p(interval) / 5.0,
            ],
            dim=-1,
        )
        repeat = self.repeat_head(features).squeeze(-1)
        eligible = (frequency > 0) | (product_frequency > 0)
        repeat = repeat.masked_fill(~eligible, -20.0)
        mixed = torch.logaddexp(
            torch.log(gate.clamp_min(1e-5)) + repeat,
            torch.log1p(-gate.clamp_max(1 - 1e-5)) + explore,
        )
        return mixed, explore, repeat, features

    def training_loss(
        self,
        batch,
        popularity_weights,
        family_hard,
        transition_hard,
        repeat_weight: float,
        sampled_negatives: int = 256,
    ):
        sequence, baskets, gate = self.encode(batch)
        positives = batch["positives"]
        positive_valid = batch["positive_valid"]
        batch_size = positives.shape[0]
        sampled = torch.multinomial(
            popularity_weights, sampled_negatives, replacement=True
        )
        first_positive = positives[:, 0]
        common = torch.cat([first_positive, sampled], dim=0)
        common = common.unsqueeze(0).expand(batch_size, -1)
        lengths = batch["valid"].sum(1).long().clamp_min(1)
        latest = batch["history"].gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        latest = torch.where(batch["valid"].any(1), latest, first_positive)
        hard = torch.cat(
            [family_hard[latest, :8], transition_hard[latest, :8]], dim=1
        )
        candidates = torch.cat([common, positives, hard], dim=1)
        mixed, _, _, _ = self.score_candidates(
            batch, sequence, baskets, gate, candidates
        )
        target = torch.zeros_like(mixed)
        target[torch.arange(batch_size, device=mixed.device), torch.arange(batch_size, device=mixed.device)] = 1
        positive_start = common.shape[1]
        target[:, positive_start : positive_start + positives.shape[1]] = positive_valid
        hard_target = (
            hard.unsqueeze(-1) == positives.unsqueeze(1)
        ) & positive_valid.unsqueeze(1)
        target[:, -hard.shape[1] :] = hard_target.any(-1)
        candidate_valid = candidates != self.n_items
        positive_loss = -F.logsigmoid(mixed)
        negative_loss = -F.logsigmoid(-mixed)
        positive_mask = (target > 0) & candidate_valid
        negative_mask = (target == 0) & candidate_valid
        sigmoid_loss = (
            (positive_loss * positive_mask).sum() / positive_mask.sum().clamp_min(1)
            + (negative_loss * negative_mask).sum() / negative_mask.sum().clamp_min(1)
        )
        contrastive = F.cross_entropy(mixed[:, :batch_size], torch.arange(batch_size, device=mixed.device))
        history_meta = self.article_encoder.item_features[batch["history"]][..., 0]
        positive_meta = self.article_encoder.item_features[positives][..., 0]
        repeat_target = (
            (
                (positives.unsqueeze(-1) == batch["history"].unsqueeze(1))
                | (positive_meta.unsqueeze(-1) == history_meta.unsqueeze(1))
            )
            & batch["valid"].unsqueeze(1)
            & positive_valid.unsqueeze(-1)
        ).any((1, 2)).to(gate.dtype)
        gate_loss = F.binary_cross_entropy_with_logits(
            torch.logit(gate.squeeze(1).clamp(1e-5, 1 - 1e-5)), repeat_target
        )
        loss = sigmoid_loss + 0.25 * contrastive + repeat_weight * gate_loss
        return loss, {
            "sigmoid": sigmoid_loss.detach(),
            "contrastive": contrastive.detach(),
            "gate": gate_loss.detach(),
            "gate_mean": gate.mean().detach(),
        }


@torch.no_grad()
def retrieve_explore(model, batch, top_k: int = 200, item_chunk: int = 8192):
    sequence, baskets, gate = model.encode(batch)
    origin = int(batch["origin"][0])
    best_score = None
    best_item = None
    for start in range(0, model.n_items, item_chunk):
        stop = min(model.n_items, start + item_chunk)
        item_ids = torch.arange(start, stop, device=sequence.device)
        item_vector = model.article_encoder(item_ids, origin)
        score = sequence @ item_vector.T / math.sqrt(model.dimension)
        count = min(top_k, stop - start)
        chunk_score, chunk_offset = score.topk(count, dim=1)
        chunk_item = chunk_offset + start
        if best_score is None:
            best_score = chunk_score
            best_item = chunk_item
        else:
            merged_score = torch.cat([best_score, chunk_score], dim=1)
            merged_item = torch.cat([best_item, chunk_item], dim=1)
            best_score, choice = merged_score.topk(top_k, dim=1)
            best_item = merged_item.gather(1, choice)
    return (
        best_item,
        best_score,
        sequence,
        baskets,
        gate,
    )
