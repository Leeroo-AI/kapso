from __future__ import annotations

import os
import sys
from pathlib import Path


RUNTIME_ROOT = Path("output_data_generic_exp_2/runtime")
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_ROOT))
os.environ.setdefault("TORCH_HOME", str(RUNTIME_ROOT / "torch"))
os.environ.setdefault("TRITON_CACHE_DIR", str(RUNTIME_ROOT / "triton"))
os.environ.setdefault("HF_HOME", str(RUNTIME_ROOT / "huggingface"))

import gc
import json
import time

import numpy as np
import pandas as pd
import torch

from hm_data import EpisodeDataset, InferenceDataset, choose_recent_indices, load_state
from relational import build_bank, build_candidate_pool
from train_pipeline import (
    fit_model_reranker,
    forward_study,
    log_phase,
    make_model,
    predict_with_reranker,
    sample_reranker_dataset,
    save_metrics,
    slice_report,
    train_epochs,
    validate_predictions,
)


CACHE_VERSION = "basket_transformer_repeat_explore_lane2_v2"


def cuda_smoke(device):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is not supported")
    layer = torch.nn.TransformerEncoderLayer(
        64, 4, 128, batch_first=True, norm_first=True
    ).to(device)
    encoder = torch.nn.TransformerEncoder(layer, 2).to(device)
    value = torch.randn(8, 16, 64, device=device, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = encoder(value).square().mean()
    loss.backward()
    print(
        f"[basket] cuda_device={torch.cuda.get_device_name(0)} "
        f"visible={os.environ.get('CUDA_VISIBLE_DEVICES')} bf16=true "
        f"smoke_loss={float(loss.detach()):.6f}"
    )
    del layer, encoder, value, loss
    torch.cuda.empty_cache()


def cutoff_day(timestamp):
    return int(
        (pd.Timestamp(timestamp) - pd.Timestamp("2019-09-01"))
        / pd.Timedelta(days=1)
    )


def bank_fallback(state, out):
    start = time.time()
    val_bank = build_bank(state, cutoff_day("2020-09-07"))
    test_bank = build_bank(state, cutoff_day("2020-09-14"))
    val_dataset = InferenceDataset(state, state.val)
    test_dataset = InferenceDataset(state, state.test)
    val_explore = np.tile(val_bank.global_top[:200], (len(val_dataset), 1))
    val_pool, val_ranks, val_prediction = build_candidate_pool(
        state, val_bank, val_dataset, val_explore
    )
    del val_pool, val_ranks, val_explore
    test_explore = np.tile(test_bank.global_top[:200], (len(test_dataset), 1))
    test_pool, test_ranks, test_prediction = build_candidate_pool(
        state, test_bank, test_dataset, test_explore
    )
    del test_pool, test_ranks, test_explore
    validate_predictions(val_prediction, len(state.val), state.n_items)
    validate_predictions(test_prediction, len(state.test), state.n_items)
    np.save(out / "val_predictions.npy", val_prediction)
    np.save(out / "test_predictions.npy", test_prediction)
    log_phase("relational_fallback_banked", start)
    return val_bank, test_bank, val_dataset, test_dataset, val_prediction, test_prediction


def complete_cache_path():
    shared = Path(os.environ.get("KAPSO_SHARED_CACHE_DIR", "shared_cache"))
    path = shared / CACHE_VERSION
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_complete_cache(out):
    cache = complete_cache_path()
    marker = cache / "complete.json"
    val_path = cache / "val_predictions.npy"
    test_path = cache / "test_predictions.npy"
    if not marker.exists() or not val_path.exists() or not test_path.exists():
        return False
    val = np.load(val_path, allow_pickle=False)
    test = np.load(test_path, allow_pickle=False)
    if val.shape != (74575, 12) or test.shape != (67144, 12):
        return False
    np.save(out / "val_predictions.npy", val)
    np.save(out / "test_predictions.npy", test)
    print(f"[basket] reused_complete_cache={cache}")
    return True


def register_cache(cache):
    import fcntl

    shared = cache.parent
    registry = shared / "artifacts.json"
    lock_path = shared / "artifacts.lock"
    entry = {
        "name": CACHE_VERSION,
        "path": CACHE_VERSION,
        "description": "Model A/B checkpoints and complete validation/test predictions for the basket Transformer repeat/explore pipeline",
        "content_key": CACHE_VERSION,
        "rebuild_hint": "Run python main.py with the registered rel-hm user-item-purchase environment",
    }
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if registry.exists():
            try:
                entries = json.loads(registry.read_text())
            except json.JSONDecodeError:
                entries = []
        else:
            entries = []
        if not any(item.get("content_key") == CACHE_VERSION for item in entries):
            entries.append(entry)
            registry.write_text(json.dumps(entries, indent=2, sort_keys=True))
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def run_debug(
    state,
    out,
    device,
    val_bank,
    test_bank,
    val_fallback,
    test_fallback,
):
    train_indices = choose_recent_indices(state.train, 4096, 1337)
    training = EpisodeDataset(state, [state.train], [train_indices])
    training_bank = build_bank(state, int(training.days.min()))
    model = make_model(state).to(device)
    optimizer = train_epochs(
        model,
        training,
        training_bank,
        1,
        0.20,
        device,
        1337,
        batch_size=512,
    )
    rerank_training = sample_reranker_dataset(state, 256, False, 1337)
    rank_bank = build_bank(state, cutoff_day("2020-08-31"))
    ranker = fit_model_reranker(
        model, state, rank_bank, rerank_training, device, trees=30
    )
    val_frame = state.val.iloc[:512]
    val_dataset = InferenceDataset(state, val_frame)
    val_small, _ = predict_with_reranker(
        model, ranker, state, val_bank, val_dataset, device
    )
    val_prediction = val_fallback.copy()
    val_prediction[: len(val_small)] = val_small
    np.save(out / "val_predictions.npy", val_prediction)
    continuation_indices = choose_recent_indices(state.train, 4096, 1338)
    validation_indices = np.arange(min(1024, len(state.val)), dtype=np.int64)
    continuation = EpisodeDataset(
        state,
        [state.train, state.val],
        [continuation_indices, validation_indices],
    )
    train_epochs(
        model,
        continuation,
        training_bank,
        1,
        0.20,
        device,
        1338,
        optimizer,
        batch_size=512,
    )
    rerank_validation = sample_reranker_dataset(state, 256, True, 1338)
    ranker_b = fit_model_reranker(
        model, state, val_bank, rerank_validation, device, trees=30
    )
    test_frame = state.test.iloc[:512]
    test_dataset = InferenceDataset(state, test_frame)
    test_small, _ = predict_with_reranker(
        model, ranker_b, state, test_bank, test_dataset, device
    )
    test_prediction = test_fallback.copy()
    test_prediction[: len(test_small)] = test_small
    validate_predictions(val_prediction, len(state.val), state.n_items)
    validate_predictions(test_prediction, len(state.test), state.n_items)
    np.save(out / "test_predictions.npy", test_prediction)
    save_metrics(
        out,
        {
            "mode": "debug",
            "neural_rows_val": len(val_small),
            "neural_rows_test": len(test_small),
        },
    )


def run_full(
    state,
    out,
    device,
    val_bank,
    test_bank,
    val_dataset,
    test_dataset,
):
    cache = complete_cache_path()
    repeat_weight, epochs = forward_study(
        state, device, cache / "forward_study.json"
    )
    train_indices = choose_recent_indices(state.train, 1000000, 1337)
    training = EpisodeDataset(state, [state.train], [train_indices])
    training_bank = build_bank(state, int(training.days.min()))
    model = make_model(state).to(device)
    optimizer = train_epochs(
        model,
        training,
        training_bank,
        epochs,
        repeat_weight,
        device,
        1337,
        batch_size=2048,
    )
    torch.save(model.state_dict(), cache / "model_a.pt")
    rerank_training = sample_reranker_dataset(state, 12000, False, 1337)
    rank_bank = build_bank(state, cutoff_day("2020-08-31"))
    ranker_a = fit_model_reranker(
        model, state, rank_bank, rerank_training, device, trees=300
    )
    ranker_a.save_model(str(cache / "reranker_a.txt"))
    val_prediction, _ = predict_with_reranker(
        model, ranker_a, state, val_bank, val_dataset, device
    )
    validate_predictions(val_prediction, len(state.val), state.n_items)
    np.save(out / "val_predictions.npy", val_prediction)
    np.save(cache / "val_predictions.npy", val_prediction)
    slices = slice_report(state, val_prediction)
    del ranker_a
    gc.collect()
    continuation = EpisodeDataset(
        state,
        [state.train, state.val],
        [train_indices, np.arange(len(state.val), dtype=np.int64)],
    )
    train_epochs(
        model,
        continuation,
        training_bank,
        1,
        repeat_weight,
        device,
        2337,
        optimizer,
        batch_size=2048,
    )
    torch.save(model.state_dict(), cache / "model_b.pt")
    rerank_validation = sample_reranker_dataset(state, 12000, True, 2337)
    ranker_b = fit_model_reranker(
        model, state, val_bank, rerank_validation, device, trees=300
    )
    ranker_b.save_model(str(cache / "reranker_b.txt"))
    test_prediction, _ = predict_with_reranker(
        model, ranker_b, state, test_bank, test_dataset, device
    )
    validate_predictions(test_prediction, len(state.test), state.n_items)
    np.save(out / "test_predictions.npy", test_prediction)
    np.save(cache / "test_predictions.npy", test_prediction)
    metrics = {
        "mode": "full",
        "model_a_epochs": epochs,
        "repeat_weight": repeat_weight,
        "slice_metrics": slices,
    }
    save_metrics(out, metrics)
    (cache / "complete.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    register_cache(cache)


def main():
    overall = time.time()
    debug = "--debug" in sys.argv
    out = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "output_data_generic_exp_2"))
    out.mkdir(parents=True, exist_ok=True)
    if not debug and load_complete_cache(out):
        return
    device = torch.device("cuda")
    cuda_smoke(device)
    start = time.time()
    state = load_state()
    log_phase("state_preprocessing", start)
    (
        val_bank,
        test_bank,
        val_dataset,
        test_dataset,
        val_fallback,
        test_fallback,
    ) = bank_fallback(state, out)
    try:
        if debug:
            run_debug(
                state,
                out,
                device,
                val_bank,
                test_bank,
                val_fallback,
                test_fallback,
            )
        else:
            run_full(
                state,
                out,
                device,
                val_bank,
                test_bank,
                val_dataset,
                test_dataset,
            )
    except Exception as error:
        np.save(out / "val_predictions.npy", val_fallback)
        np.save(out / "test_predictions.npy", test_fallback)
        save_metrics(
            out,
            {
                "mode": "fallback_after_error",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        print(
            f"[basket] neural_pipeline_error={type(error).__name__}:"
            f"{str(error).replace(chr(10), ' ')} fallback_preserved=true"
        )
    log_phase("run_complete", overall, f"debug={debug}")


if __name__ == "__main__":
    main()
