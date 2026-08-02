from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np

from clinical_pipeline import (
    FEATURE_VERSION,
    MODEL_REVISION,
    RuntimeConfig,
    blend_probabilities,
    build_model_a,
    build_model_b,
    curriculum_pool,
    elapsed,
    ensure_pretrained_model,
    fit_catboost_model_a,
    fit_catboost_model_b,
    fit_internal_catboost,
    internal_diagnostics,
    online_priors,
    prepare_data,
    register_artifact,
    run_encoder_curriculum,
    seed_everything,
)
from kapso_datasets.common import is_debug, load_task, run_data_dir, save_predictions, shared_cache_dir


def runtime_configuration(started: float) -> RuntimeConfig:
    debug = is_debug()
    return RuntimeConfig(
        debug=debug,
        max_length=512 if debug else 1024,
        micro_batch=8 if debug else 16,
        inference_batch=64,
        cat_iterations=150 if debug else 1200,
        training_deadline=started + (16 * 60 if debug else 210 * 60),
        namespace="generic_exp_1_bioclinical_longctx",
    )


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    started = time.time()
    seed_everything(1337)
    config = runtime_configuration(started)
    shared = shared_cache_dir()
    model_dir = ensure_pretrained_model(shared)
    print(
        f"[config] debug={config.debug} model_revision={MODEL_REVISION} max_length={config.max_length} effective_batch=32 bf16=true flash_attention_2=false",
        flush=True,
    )

    ctx = load_task(upto_test_timestamp=False)
    train_labels = ctx.train.df[ctx.target_col].to_numpy(dtype=np.float32, copy=True)
    validation_source = ctx.val.df
    data = prepare_data(ctx, shared, config)
    years = data.seed_frame.iloc[: data.n_train]["date"].dt.year.to_numpy(dtype=np.int16)
    train_days = data.seed_frame.iloc[: data.n_train]["date"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    del ctx
    gc.collect()
    elapsed(started, "data and token preparation")

    train_prior_path = data.cache_dir / "priors_train_strictly_earlier.npy"
    train_online = online_priors(data.relations, train_labels, train_days, train_prior_path)
    fit_pool = curriculum_pool(years, config.debug)
    print(f"[curriculum] fit_pool={len(fit_pool)} years={years[fit_pool].min()}-{years[fit_pool].max()}", flush=True)

    encoder_forward, latest_source, stage_records = run_encoder_curriculum(
        data,
        train_labels,
        years,
        model_dir,
        config,
        fit_pool,
    )
    elapsed(started, "encoder forward curriculum")

    diagnostics = {"stages": stage_records}
    blend_weight = 0.80
    cat_forward = {}
    if all(year in encoder_forward for year in (2017, 2018, 2019)):
        cat_forward = fit_internal_catboost(
            data,
            train_labels,
            train_online,
            years,
            fit_pool,
            config,
        )
        blend_weight, forward_diagnostics = internal_diagnostics(
            encoder_forward,
            cat_forward,
            train_labels,
            years,
            data.base_features,
        )
        diagnostics["forward"] = forward_diagnostics
    else:
        diagnostics["forward"] = {"status": "deadline fallback", "selected_encoder_weight": blend_weight}
        print("[internal] incomplete forward chain; using predeclared encoder weight 0.80", flush=True)
    elapsed(started, "internal temporal selection")

    encoder_val, model_a_source, model_a_record = build_model_a(
        data,
        train_labels,
        years,
        latest_source,
        model_dir,
        config,
        fit_pool,
    )
    cat_val = fit_catboost_model_a(data, train_labels, train_online, fit_pool, config)
    frozen_dir = data.cache_dir / ("debug_models" if config.debug else "full_models")
    np.save(frozen_dir / "frozen_model_a_encoder_val.npy", encoder_val)
    np.save(frozen_dir / "frozen_model_a_cat_val.npy", cat_val)
    diagnostics["model_a"] = model_a_record
    print("[freeze] both Model A validation components persisted before validation labels are accessed", flush=True)
    elapsed(started, "Model A frozen validation inference")

    validation_labels = validation_source["adult"].to_numpy(dtype=np.float32, copy=True)
    del validation_source
    combined_labels = np.concatenate([train_labels, validation_labels])
    combined_days = data.seed_frame.iloc[: data.n_train + data.n_val]["date"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    combined_prior_path = data.cache_dir / "priors_model_b_train_val_strictly_earlier.npy"
    combined_online = online_priors(data.relations, combined_labels, combined_days, combined_prior_path)

    encoder_test, model_b_source, model_b_record = build_model_b(
        data,
        combined_labels,
        years,
        model_a_source,
        config,
    )
    cat_test = fit_catboost_model_b(data, combined_labels, combined_online, config)
    diagnostics["model_b"] = model_b_record
    elapsed(started, "Model B adaptation and test inference")

    val_predictions = blend_probabilities(encoder_val, cat_val, blend_weight).astype(np.float32)
    test_predictions = blend_probabilities(encoder_test, cat_test, blend_weight).astype(np.float32)
    val_predictions = np.clip(val_predictions, 0.0, 1.0)
    test_predictions = np.clip(test_predictions, 0.0, 1.0)
    if val_predictions.shape != (data.n_val,) or test_predictions.shape != (data.n_test,):
        raise RuntimeError(f"Prediction alignment failure: {val_predictions.shape} {test_predictions.shape}")
    if not np.all(np.isfinite(val_predictions)) or not np.all(np.isfinite(test_predictions)):
        raise RuntimeError("Non-finite final predictions")

    diagnostics.update(
        {
            "feature_version": FEATURE_VERSION,
            "model_revision": MODEL_REVISION,
            "debug": config.debug,
            "encoder_weight": blend_weight,
            "validation_fit": "Model A: train labels only",
            "test_fit": "Model B: train plus validation labels",
            "elapsed_seconds": time.time() - started,
            "val_probability_summary": {
                "mean": float(val_predictions.mean()),
                "standard_deviation": float(val_predictions.std()),
                "minimum": float(val_predictions.min()),
                "maximum": float(val_predictions.max()),
            },
            "test_probability_summary": {
                "mean": float(test_predictions.mean()),
                "standard_deviation": float(test_predictions.std()),
                "minimum": float(test_predictions.min()),
                "maximum": float(test_predictions.max()),
            },
        }
    )
    save_predictions(val_predictions, test_predictions)
    metrics_path = run_data_dir() / "metrics.json"
    metrics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    output_dir = Path("output_data_generic_exp_1")
    output_dir.mkdir(parents=True, exist_ok=True)
    local_metrics = output_dir / ("metrics_debug.json" if config.debug else "metrics_full.json")
    local_metrics.write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    register_artifact(
        shared,
        f"generic_exp_1 {'debug' if config.debug else 'full'} predictions",
        frozen_dir,
        "Frozen Model A validation components, Model B test components, checkpoints, and internal forward predictions",
        f"{FEATURE_VERSION}:{MODEL_REVISION}:{config.max_length}",
        "Rerun main.py at the matching fidelity",
    )
    print(
        f"[complete] elapsed={time.time() - started:.1f}s encoder_weight={blend_weight:.2f} model_a={model_a_source.name} model_b={model_b_source.name}",
        flush=True,
    )


if __name__ == "__main__":
    main()
