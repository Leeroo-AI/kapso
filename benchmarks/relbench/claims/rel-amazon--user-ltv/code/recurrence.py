from __future__ import annotations

import contextlib
import inspect
import io
import os
import signal
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from feature_pipeline import FEATURE_VERSION, cache_root, register_artifact


BG_SAMPLE_CAP = 250000
BG_TIMEOUT_SECONDS = 600


class FitTimeout(Exception):
    pass


def timeout_handler(signum, frame):
    raise FitTimeout("BG/NBD MAP exceeded ten-minute cutoff gate")


def raw_recurrence(frame: pd.DataFrame) -> np.ndarray:
    frequency = frame["frequency_after_first"].to_numpy(dtype=np.float64)
    age = frame["age_days"].to_numpy(dtype=np.float64)
    recency = frame["recency_days"].to_numpy(dtype=np.float64)
    expected = (frequency + 0.5) * 91.0 / np.maximum(age + 30.0, 30.0)
    alive = np.exp(-recency / np.maximum(age + 30.0, 30.0))
    no_purchase = np.exp(-np.minimum(expected, 30.0))
    return np.column_stack([expected, alive, no_purchase]).astype(np.float32)


def hash_stratified_sample(data: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(data) <= cap:
        return data.copy()
    work = data.copy()
    work["stratum"] = np.minimum(work["frequency"].to_numpy(dtype=np.int64), 10)
    work["sample_hash"] = pd.util.hash_pandas_object(work["customer_id"], index=False).to_numpy(dtype=np.uint64)
    counts = work["stratum"].value_counts().sort_index()
    pieces = []
    for stratum, count in counts.items():
        take = max(1, int(round(cap * int(count) / len(work))))
        part = work.loc[work["stratum"] == stratum].nsmallest(take, "sample_hash")
        pieces.append(part)
    sampled = pd.concat(pieces, ignore_index=True)
    if len(sampled) > cap:
        sampled = sampled.nsmallest(cap, "sample_hash")
    return sampled.drop(columns=["stratum", "sample_hash"])


def prediction_vector(value, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == size:
        return array.reshape(size)
    axes = [axis for axis, length in enumerate(array.shape) if length == size]
    if not axes:
        raise RuntimeError(f"BG/NBD prediction shape {array.shape} lacks customer axis {size}")
    array = np.moveaxis(array, axes[-1], 0).reshape(size, -1)
    return array.mean(axis=1)


def fit_cutoff(frame: pd.DataFrame) -> np.ndarray:
    from pymc_marketing.clv import BetaGeoModel

    model_data = pd.DataFrame(
        {
            "customer_id": frame["customer_id"].to_numpy(dtype=np.int64),
            "frequency": frame["frequency_after_first"].to_numpy(dtype=np.int64),
            "recency": np.maximum(frame["age_days"].to_numpy(dtype=np.float64) - frame["recency_days"].to_numpy(dtype=np.float64), 0.0) / 7.0,
            "T": np.maximum(frame["age_days"].to_numpy(dtype=np.float64), 0.0) / 7.0,
        }
    )
    model_data["recency"] = np.minimum(model_data["recency"], model_data["T"])
    sample = hash_stratified_sample(model_data, BG_SAMPLE_CAP)
    model = BetaGeoModel(sample)
    fit_signature = inspect.signature(model.fit)
    kwargs = {"progressbar": False, "maxeval": 3000}
    if "method" in fit_signature.parameters:
        kwargs["method"] = "map"
    else:
        kwargs["fit_method"] = "map"
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(BG_TIMEOUT_SECONDS)
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            model.fit(**kwargs)
            expected = prediction_vector(model.expected_purchases(data=model_data, future_t=13), len(frame))
            alive = prediction_vector(model.expected_probability_alive(data=model_data), len(frame))
            no_purchase = prediction_vector(model.expected_probability_no_purchase(t=13, data=model_data), len(frame))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    result = np.column_stack([expected, alive, no_purchase])
    if not np.all(np.isfinite(result)) or np.any(result < -1e-8) or np.any(alive > 1.0001) or np.any(no_purchase > 1.0001):
        raise RuntimeError("BG/NBD MAP produced unstable predictions")
    result[:, 0] = np.clip(result[:, 0], 0, 100)
    result[:, 1:] = np.clip(result[:, 1:], 0, 1)
    return result.astype(np.float32)


def recurrence_features(frame: pd.DataFrame, split: str, debug: bool = False) -> np.ndarray:
    output = cache_root() / f"bgnbd_{split}.npy"
    if output.exists():
        values = np.load(output, allow_pickle=False)
        if values.shape == (len(frame), 3) and np.all(np.isfinite(values)):
            print(f"[bgnbd] loaded {split} cache shape={values.shape}")
            return values
    raw = raw_recurrence(frame)
    if debug:
        print(f"[bgnbd] debug fallback for {split}")
        return raw
    result = raw.copy()
    timestamps = pd.Index(frame["timestamp"].drop_duplicates()).sort_values()
    for index, timestamp in enumerate(timestamps, start=1):
        mask = frame["timestamp"].eq(timestamp).to_numpy()
        started = time.time()
        status = "map"
        try:
            result[mask] = fit_cutoff(frame.loc[mask])
        except Exception as error:
            status = f"fallback:{type(error).__name__}:{str(error)[:120]}"
        print(f"[bgnbd] {split} cutoff={timestamp} rows={int(mask.sum())} mode={status} elapsed={time.time() - started:.1f}s ({index}/{len(timestamps)})")
    tmp = output.with_name(f"{output.name}.{os.getpid()}.tmp.npy")
    np.save(tmp, result)
    os.replace(tmp, output)
    register_artifact(
        f"{FEATURE_VERSION}-bgnbd-{split}",
        output,
        f"Cutoff-specific BG/NBD expected count, alive probability, and no-review probability for {split}",
        "Delete this file and rerun main.py to refit cutoff-specific MAP models",
    )
    return result
