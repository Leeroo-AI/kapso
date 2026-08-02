import gc
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class FactorState:
    mu: float
    user_bias: np.ndarray
    item_bias: np.ndarray
    user_drift: np.ndarray
    item_drift: np.ndarray
    user_factor: np.ndarray
    item_factor: np.ndarray
    implicit_factor: np.ndarray
    user_mean_day: np.ndarray
    item_mean_day: np.ndarray
    user_count: np.ndarray
    item_count: np.ndarray
    recency_coef: np.ndarray


class TimeSVDPP(nn.Module):
    def __init__(self, n_users, n_items, factors, mu, init):
        super().__init__()
        self.n_items = n_items
        self.mu = nn.Parameter(torch.tensor(float(mu), dtype=torch.float32))
        self.user_bias = nn.Embedding(n_users, 1, sparse=True)
        self.item_bias = nn.Embedding(n_items, 1, sparse=True)
        self.user_drift = nn.Embedding(n_users, 1, sparse=True)
        self.item_drift = nn.Embedding(n_items, 1, sparse=True)
        self.user_factor = nn.Embedding(n_users, factors, sparse=True)
        self.item_factor = nn.Embedding(n_items, factors, sparse=True)
        self.implicit_factor = nn.EmbeddingBag(
            n_items + 1,
            factors,
            mode="sum",
            sparse=True,
            padding_idx=n_items,
        )
        self.recency_coef = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        with torch.no_grad():
            self.user_bias.weight.copy_(torch.from_numpy(init["user_bias"][:, None]))
            self.item_bias.weight.copy_(torch.from_numpy(init["item_bias"][:, None]))
            self.user_drift.weight.copy_(torch.from_numpy(init["user_drift"][:, None]))
            self.item_drift.weight.copy_(torch.from_numpy(init["item_drift"][:, None]))
            self.user_factor.weight.normal_(0.0, 0.025)
            self.item_factor.weight.normal_(0.0, 0.025)
            self.implicit_factor.weight.zero_()
            self.user_factor.weight[torch.from_numpy(init["user_count"] == 0)] = 0
            self.item_factor.weight[torch.from_numpy(init["item_count"] == 0)] = 0

    def forward(self, user, item, day, user_mean, item_mean, user_stale, item_stale, history, history_count, implicit_enabled=True):
        width = history.shape[1]
        real = history != self.n_items
        available = real.sum(1).clamp_min(1)
        scale = history_count.float().clamp_min(1).sqrt() / available.float()
        weights = real.float() * scale[:, None]
        offsets = torch.arange(0, history.numel(), width, device=history.device)
        implicit = self.implicit_factor(
            history.reshape(-1),
            offsets,
            per_sample_weights=weights.reshape(-1),
        )
        if not implicit_enabled:
            implicit = implicit.detach() * 0
        dev_user = torch.sign(day - user_mean) * torch.abs(day - user_mean).pow(0.4)
        dev_item = torch.sign(day - item_mean) * torch.abs(day - item_mean).pow(0.4)
        global_term = self.mu.expand_as(day)
        user_term = self.user_bias(user).squeeze(1)
        item_term = self.item_bias(item).squeeze(1)
        user_drift_term = self.user_drift(user).squeeze(1) * dev_user
        item_drift_term = self.item_drift(item).squeeze(1) * dev_item
        recency_term = self.recency_coef[0] * user_stale + self.recency_coef[1] * item_stale
        explicit = self.user_factor(user)
        item_factor = self.item_factor(item)
        latent_term = (item_factor * explicit).sum(1)
        implicit_term = (item_factor * implicit).sum(1)
        pred = global_term + user_term + item_term + user_drift_term + item_drift_term + recency_term + latent_term + implicit_term
        return pred, (user_term, item_term, user_drift_term, item_drift_term, latent_term, implicit_term, explicit, item_factor, implicit)


def initialize_biases(users, items, ratings, days, n_users, n_items):
    mu = float(np.mean(ratings, dtype=np.float64))
    user_count = np.bincount(users, minlength=n_users).astype(np.int32)
    item_count = np.bincount(items, minlength=n_items).astype(np.int32)
    item_bias = np.bincount(items, weights=ratings - mu, minlength=n_items) / (item_count + 25.0)
    user_bias = np.bincount(users, weights=ratings - mu - item_bias[items], minlength=n_users) / (user_count + 15.0)
    item_bias = np.bincount(items, weights=ratings - mu - user_bias[users], minlength=n_items) / (item_count + 20.0)
    user_mean_day = np.bincount(users, weights=days, minlength=n_users) / np.maximum(user_count, 1)
    item_mean_day = np.bincount(items, weights=days, minlength=n_items) / np.maximum(item_count, 1)
    residual = ratings - mu - user_bias[users] - item_bias[items]
    user_dev = np.sign(days - user_mean_day[users]) * np.abs(days - user_mean_day[users]) ** 0.4
    item_dev = np.sign(days - item_mean_day[items]) * np.abs(days - item_mean_day[items]) ** 0.4
    user_num = np.bincount(users, weights=user_dev * residual, minlength=n_users)
    user_den = np.bincount(users, weights=user_dev * user_dev, minlength=n_users) + 80.0
    user_drift = np.clip(user_num / user_den, -0.08, 0.08)
    residual = residual - user_drift[users] * user_dev
    item_num = np.bincount(items, weights=item_dev * residual, minlength=n_items)
    item_den = np.bincount(items, weights=item_dev * item_dev, minlength=n_items) + 120.0
    item_drift = np.clip(item_num / item_den, -0.08, 0.08)
    return {
        "mu": mu,
        "user_bias": user_bias.astype(np.float32),
        "item_bias": item_bias.astype(np.float32),
        "user_drift": user_drift.astype(np.float32),
        "item_drift": item_drift.astype(np.float32),
        "user_mean_day": user_mean_day.astype(np.float32),
        "item_mean_day": item_mean_day.astype(np.float32),
        "user_count": user_count,
        "item_count": item_count,
    }


def fit_time_svdpp(data, selected, n_users, n_items, factors=64, epochs=3, seed=1337):
    started = time.time()
    users = data["user"][selected]
    items = data["item"][selected]
    ratings = data["rating"][selected].astype(np.float32, copy=False)
    days = data["day"][selected].astype(np.float32, copy=False)
    init = initialize_biases(users, items, ratings, days, n_users, n_items)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = TimeSVDPP(n_users, n_items, factors, init["mu"], init).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.035)
    rng = np.random.default_rng(seed)
    batch_size = 131072
    for epoch in range(epochs):
        order = rng.permutation(len(selected))
        loss_sum = 0.0
        seen = 0
        for begin in range(0, len(order), batch_size):
            local = order[begin : begin + batch_size]
            pos = selected[local]
            u = torch.as_tensor(data["user"][pos], dtype=torch.long, device=device)
            i = torch.as_tensor(data["item"][pos], dtype=torch.long, device=device)
            target = torch.as_tensor(data["rating"][pos], dtype=torch.float32, device=device)
            day = torch.as_tensor(data["day"][pos], dtype=torch.float32, device=device)
            user_mean = torch.as_tensor(init["user_mean_day"][data["user"][pos]], dtype=torch.float32, device=device)
            item_mean = torch.as_tensor(init["item_mean_day"][data["item"][pos]], dtype=torch.float32, device=device)
            user_stale = torch.as_tensor(np.log1p(data["user_stale"][pos]) / 10.0, dtype=torch.float32, device=device)
            item_stale = torch.as_tensor(np.log1p(data["item_stale"][pos]) / 10.0, dtype=torch.float32, device=device)
            history = torch.as_tensor(data["history"][pos], dtype=torch.long, device=device)
            history_count = torch.as_tensor(data["user_count"][pos], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred, terms = model(u, i, day, user_mean, item_mean, user_stale, item_stale, history, history_count, implicit_enabled=(epochs == 1 or epoch > 0))
            base_loss = torch.mean((pred - target) ** 2)
            factor_penalty = terms[6].square().mean() + terms[7].square().mean() + terms[8].square().mean()
            bias_penalty = terms[0].square().mean() + terms[1].square().mean() + model.user_drift(u).square().mean() + model.item_drift(i).square().mean()
            loss = base_loss + 8e-5 * factor_penalty + 2e-5 * bias_penalty
            loss.backward()
            optimizer.step()
            count = len(pos)
            loss_sum += float(base_loss.detach()) * count
            seen += count
        print(f"[svdpp] epoch={epoch + 1}/{epochs} rows={seen} mse={loss_sum / seen:.6f} elapsed={time.time() - started:.1f}s", flush=True)
    with torch.no_grad():
        state = FactorState(
            mu=float(model.mu.detach().cpu()),
            user_bias=model.user_bias.weight.detach().cpu().numpy().reshape(-1).astype(np.float32),
            item_bias=model.item_bias.weight.detach().cpu().numpy().reshape(-1).astype(np.float32),
            user_drift=model.user_drift.weight.detach().cpu().numpy().reshape(-1).astype(np.float32),
            item_drift=model.item_drift.weight.detach().cpu().numpy().reshape(-1).astype(np.float32),
            user_factor=model.user_factor.weight.detach().cpu().numpy().astype(np.float32),
            item_factor=model.item_factor.weight.detach().cpu().numpy().astype(np.float32),
            implicit_factor=model.implicit_factor.weight[:-1].detach().cpu().numpy().astype(np.float32),
            user_mean_day=init["user_mean_day"],
            item_mean_day=init["item_mean_day"],
            user_count=init["user_count"],
            item_count=init["item_count"],
            recency_coef=model.recency_coef.detach().cpu().numpy().astype(np.float32),
        )
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state
