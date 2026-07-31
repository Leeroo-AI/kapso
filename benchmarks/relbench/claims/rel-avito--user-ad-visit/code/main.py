from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kapso_datasets.common import is_debug, load_task, run_data_dir
from rank_pipeline import (
    OriginData,
    append_artifact,
    candidate_diagnostics,
    debug_predictions,
    elapsed,
    fallback_predictions,
    fuse_model_ranks,
    hybrid_predictions,
    load_static,
    map_at_12,
    origin_from_table,
    repeat_first_predictions,
    repeat_training_subset,
    scored_predictions,
    train_ranker,
    validate_predictions,
)


def concatenate_training(
    origins: list[OriginData],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices = [origin.features[origin.train_indices] for origin in origins]
    labels = [origin.train_labels for origin in origins]
    groups = [origin.train_groups for origin in origins]
    return np.concatenate(matrices), np.concatenate(labels), np.concatenate(groups)


def concatenate_repeat_training(
    origins: list[OriginData],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subsets = [repeat_training_subset(origin) for origin in origins]
    matrices = [
        origin.features[subset[0]]
        for origin, subset in zip(origins, subsets)
    ]
    labels = [subset[1] for subset in subsets]
    groups = [subset[2] for subset in subsets]
    return np.concatenate(matrices), np.concatenate(labels), np.concatenate(groups)


def save_outputs(val_predictions: np.ndarray, test_predictions: np.ndarray) -> None:
    output = run_data_dir()
    np.save(output / "val_predictions.npy", val_predictions)
    np.save(output / "test_predictions.npy", test_predictions)
    print(
        f"[lane0] saved val={val_predictions.shape} test={test_predictions.shape}",
        flush=True,
    )


def main() -> None:
    start = time.time()
    np.random.seed(2026)
    ctx = load_task()
    elapsed(start, "load_task")
    static = load_static(ctx, start)
    if is_debug():
        val_predictions, test_predictions = debug_predictions(ctx, static, start)
        validate_predictions(
            val_predictions,
            test_predictions,
            len(ctx.val.df),
            len(ctx.test.df),
            static.n_ads,
        )
        save_outputs(val_predictions, test_predictions)
        elapsed(start, "debug_complete")
        return
    train_frame = ctx.train.df
    val_frame = ctx.val.df
    test_frame = ctx.test.df
    train_timestamps = sorted(pd.to_datetime(train_frame["timestamp"].unique()))
    diagnostics: list[dict[str, object]] = []
    train_origins: list[OriginData] = []
    train_states = []
    for timestamp in train_timestamps:
        origin, state = origin_from_table(
            static,
            train_frame,
            pd.Timestamp(timestamp),
            start,
            True,
        )
        diagnostics.append(
            candidate_diagnostics(origin, state.user_features["user_unique_ads"])
        )
        print(
            f"[lane0] candidate_diagnostics={json.dumps(diagnostics[-1], sort_keys=True)}",
            flush=True,
        )
        train_origins.append(origin)
        train_states.append(state)
        gc.collect()
    val_timestamp = pd.Timestamp(val_frame["timestamp"].iloc[0])
    val_origin, val_state = origin_from_table(
        static,
        val_frame,
        val_timestamp,
        start,
        True,
    )
    test_timestamp = pd.Timestamp(test_frame["timestamp"].iloc[0])
    test_origin, test_state = origin_from_table(
        static,
        test_frame,
        test_timestamp,
        start,
        False,
    )
    fallback_val = fallback_predictions(val_origin.candidates, val_state.pop_global)
    fallback_test = fallback_predictions(test_origin.candidates, test_state.pop_global)
    validate_predictions(
        fallback_val,
        fallback_test,
        len(val_frame),
        len(test_frame),
        static.n_ads,
    )
    save_outputs(fallback_val, fallback_test)
    fallback_internal = [
        map_at_12(
            fallback_predictions(origin.candidates, state.pop_global),
            origin.truths,
        )
        for origin, state in zip(train_origins, train_states)
    ]
    print(f"[lane0] internal_rrf_map12={fallback_internal}", flush=True)
    repeat_internal = [
        map_at_12(
            repeat_first_predictions(origin, state.pop_global),
            origin.truths,
        )
        for origin, state in zip(train_origins, train_states)
    ]
    print(f"[lane0] internal_repeat_first_map12={repeat_internal}", flush=True)
    fixed_iterations: list[int] = []
    internal_ranker_scores: list[float] = []
    blend_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    blend_scores = {weight: [] for weight in blend_weights}
    specialist_blend_scores = {weight: [] for weight in blend_weights}
    pointwise_specialist_blend_scores = {
        weight: [] for weight in blend_weights
    }
    dual_weights = [0.25, 0.5, 0.75]
    dual_specialist_scores = {weight: [] for weight in dual_weights}
    try:
        first_model = train_ranker(
            train_origins[0].features[train_origins[0].train_indices],
            train_origins[0].train_labels,
            train_origins[0].train_groups,
            1200,
            train_origins[1].features[train_origins[1].train_indices],
            train_origins[1].train_labels,
            train_origins[1].train_groups,
        )
        first_predictions = scored_predictions(
            first_model, train_origins[1], train_states[1].pop_global
        )
        first_score = map_at_12(first_predictions, train_origins[1].truths)
        first_raw_scores = first_model.predict(
            train_origins[1].features,
            num_iteration=first_model.best_iteration or first_model.current_iteration(),
        )
        for weight in blend_weights:
            prediction = hybrid_predictions(
                first_model,
                train_origins[1],
                train_states[1].pop_global,
                weight,
                first_raw_scores,
            )
            blend_scores[weight].append(
                map_at_12(prediction, train_origins[1].truths)
            )
        fixed_iterations.append(first_model.best_iteration or first_model.current_iteration())
        internal_ranker_scores.append(first_score)
        print(
            f"[lane0] forward_fold=2015-04-30 lambda_map12={first_score:.8f} "
            f"rrf_map12={fallback_internal[1]:.8f} rounds={fixed_iterations[-1]}",
            flush=True,
        )
        del first_model, first_predictions
        gc.collect()
        x_first_two, y_first_two, g_first_two = concatenate_training(train_origins[:2])
        second_model = train_ranker(
            x_first_two,
            y_first_two,
            g_first_two,
            1200,
            train_origins[2].features[train_origins[2].train_indices],
            train_origins[2].train_labels,
            train_origins[2].train_groups,
        )
        second_predictions = scored_predictions(
            second_model, train_origins[2], train_states[2].pop_global
        )
        second_score = map_at_12(second_predictions, train_origins[2].truths)
        second_raw_scores = second_model.predict(
            train_origins[2].features,
            num_iteration=second_model.best_iteration or second_model.current_iteration(),
        )
        for weight in blend_weights:
            prediction = hybrid_predictions(
                second_model,
                train_origins[2],
                train_states[2].pop_global,
                weight,
                second_raw_scores,
            )
            blend_scores[weight].append(
                map_at_12(prediction, train_origins[2].truths)
            )
        fixed_iterations.append(second_model.best_iteration or second_model.current_iteration())
        internal_ranker_scores.append(second_score)
        print(
            f"[lane0] forward_fold=2015-05-04 lambda_map12={second_score:.8f} "
            f"rrf_map12={fallback_internal[2]:.8f} rounds={fixed_iterations[-1]}",
            flush=True,
        )
        pointwise_model = train_ranker(
            x_first_two,
            y_first_two,
            g_first_two,
            300,
            train_origins[2].features[train_origins[2].train_indices],
            train_origins[2].train_labels,
            train_origins[2].train_groups,
            objective="binary",
        )
        pointwise_predictions = scored_predictions(
            pointwise_model, train_origins[2], train_states[2].pop_global
        )
        pointwise_score = map_at_12(pointwise_predictions, train_origins[2].truths)
        print(
            f"[lane0] forward_fold=2015-05-04 pointwise_map12={pointwise_score:.8f}",
            flush=True,
        )
        repeat_x0, repeat_y0, repeat_g0 = concatenate_repeat_training(
            train_origins[:1]
        )
        repeat_i1, repeat_l1, repeat_vg1 = repeat_training_subset(train_origins[1])
        specialist_first = train_ranker(
            repeat_x0,
            repeat_y0,
            repeat_g0,
            1200,
            train_origins[1].features[repeat_i1],
            repeat_l1,
            repeat_vg1,
        )
        specialist_first_scores = specialist_first.predict(
            train_origins[1].features,
            num_iteration=specialist_first.best_iteration
            or specialist_first.current_iteration(),
        )
        for weight in blend_weights:
            prediction = hybrid_predictions(
                specialist_first,
                train_origins[1],
                train_states[1].pop_global,
                weight,
                specialist_first_scores,
                False,
            )
            specialist_blend_scores[weight].append(
                map_at_12(prediction, train_origins[1].truths)
            )
        specialist_iterations = [
            specialist_first.best_iteration or specialist_first.current_iteration()
        ]
        specialist_first_scores_for_dual = specialist_first_scores
        del specialist_first
        repeat_x01, repeat_y01, repeat_g01 = concatenate_repeat_training(
            train_origins[:2]
        )
        repeat_i2, repeat_l2, repeat_vg2 = repeat_training_subset(train_origins[2])
        specialist_second = train_ranker(
            repeat_x01,
            repeat_y01,
            repeat_g01,
            1200,
            train_origins[2].features[repeat_i2],
            repeat_l2,
            repeat_vg2,
        )
        specialist_second_scores = specialist_second.predict(
            train_origins[2].features,
            num_iteration=specialist_second.best_iteration
            or specialist_second.current_iteration(),
        )
        for weight in blend_weights:
            prediction = hybrid_predictions(
                specialist_second,
                train_origins[2],
                train_states[2].pop_global,
                weight,
                specialist_second_scores,
                False,
            )
            specialist_blend_scores[weight].append(
                map_at_12(prediction, train_origins[2].truths)
            )
        specialist_iterations.append(
            specialist_second.best_iteration or specialist_second.current_iteration()
        )
        specialist_second_scores_for_dual = specialist_second_scores
        del specialist_second
        pointwise_first = train_ranker(
            repeat_x0,
            repeat_y0,
            repeat_g0,
            300,
            train_origins[1].features[repeat_i1],
            repeat_l1,
            repeat_vg1,
            objective="binary",
        )
        pointwise_first_scores = pointwise_first.predict(
            train_origins[1].features,
            num_iteration=pointwise_first.best_iteration
            or pointwise_first.current_iteration(),
        )
        for weight in blend_weights:
            prediction = hybrid_predictions(
                pointwise_first,
                train_origins[1],
                train_states[1].pop_global,
                weight,
                pointwise_first_scores,
                False,
            )
            pointwise_specialist_blend_scores[weight].append(
                map_at_12(prediction, train_origins[1].truths)
            )
        for weight in dual_weights:
            fused_scores = fuse_model_ranks(
                train_origins[1],
                specialist_first_scores_for_dual,
                pointwise_first_scores,
                weight,
            )
            prediction = hybrid_predictions(
                pointwise_first,
                train_origins[1],
                train_states[1].pop_global,
                1.0,
                fused_scores,
                False,
            )
            dual_specialist_scores[weight].append(
                map_at_12(prediction, train_origins[1].truths)
            )
        pointwise_iterations = [
            pointwise_first.best_iteration or pointwise_first.current_iteration()
        ]
        del pointwise_first, pointwise_first_scores
        del specialist_first_scores_for_dual
        pointwise_second = train_ranker(
            repeat_x01,
            repeat_y01,
            repeat_g01,
            300,
            train_origins[2].features[repeat_i2],
            repeat_l2,
            repeat_vg2,
            objective="binary",
        )
        pointwise_second_scores = pointwise_second.predict(
            train_origins[2].features,
            num_iteration=pointwise_second.best_iteration
            or pointwise_second.current_iteration(),
        )
        for weight in blend_weights:
            prediction = hybrid_predictions(
                pointwise_second,
                train_origins[2],
                train_states[2].pop_global,
                weight,
                pointwise_second_scores,
                False,
            )
            pointwise_specialist_blend_scores[weight].append(
                map_at_12(prediction, train_origins[2].truths)
            )
        for weight in dual_weights:
            fused_scores = fuse_model_ranks(
                train_origins[2],
                specialist_second_scores_for_dual,
                pointwise_second_scores,
                weight,
            )
            prediction = hybrid_predictions(
                pointwise_second,
                train_origins[2],
                train_states[2].pop_global,
                1.0,
                fused_scores,
                False,
            )
            dual_specialist_scores[weight].append(
                map_at_12(prediction, train_origins[2].truths)
            )
        pointwise_iterations.append(
            pointwise_second.best_iteration or pointwise_second.current_iteration()
        )
        del pointwise_second, pointwise_second_scores
        del specialist_second_scores_for_dual
        frozen_rounds = int(np.clip(np.median(fixed_iterations), 50, 1200))
        specialist_rounds = int(
            np.clip(np.median(specialist_iterations), 50, 1200)
        )
        pointwise_rounds = int(
            np.clip(np.median(pointwise_iterations), 30, 300)
        )
        stability_scores = {
            weight: float(np.mean(scores) - 0.5 * np.std(scores))
            for weight, scores in blend_scores.items()
        }
        specialist_stability = {
            weight: float(np.mean(scores) - 0.5 * np.std(scores))
            for weight, scores in specialist_blend_scores.items()
        }
        pointwise_stability = {
            weight: float(np.mean(scores) - 0.5 * np.std(scores))
            for weight, scores in pointwise_specialist_blend_scores.items()
        }
        dual_stability = {
            weight: float(np.mean(scores) - 0.5 * np.std(scores))
            for weight, scores in dual_specialist_scores.items()
        }
        choices = [
            ("general", weight, stability_scores[weight])
            for weight in blend_weights
        ] + [
            ("specialist", weight, specialist_stability[weight])
            for weight in blend_weights
        ] + [
            ("pointwise_specialist", weight, pointwise_stability[weight])
            for weight in blend_weights
        ] + [
            (f"dual_{weight}", 1.0, dual_stability[weight])
            for weight in dual_weights
        ]
        selected_source, selected_weight, _ = max(
            choices,
            key=lambda choice: (choice[2], -choice[1], choice[0] == "general"),
        )
        print(
            f"[lane0] frozen_recipe rounds={frozen_rounds} "
            f"forward_mean={np.mean(internal_ranker_scores):.8f} "
            f"blend_scores={json.dumps(blend_scores)} "
            f"specialist_blend_scores={json.dumps(specialist_blend_scores)} "
            f"specialist_rounds={specialist_rounds} "
            f"pointwise_specialist_blend_scores="
            f"{json.dumps(pointwise_specialist_blend_scores)} "
            f"pointwise_rounds={pointwise_rounds} "
            f"dual_specialist_scores={json.dumps(dual_specialist_scores)} "
            f"selected_source={selected_source} "
            f"selected_model_weight={selected_weight}",
            flush=True,
        )
        del second_model, second_predictions, pointwise_model, pointwise_predictions
        del x_first_two, y_first_two, g_first_two
        gc.collect()
        x_model_a, y_model_a, g_model_a = concatenate_training(train_origins)
        model_a = train_ranker(
            x_model_a,
            y_model_a,
            g_model_a,
            frozen_rounds,
        )
        if selected_source.startswith("dual_"):
            repeat_x_a, repeat_y_a, repeat_g_a = concatenate_repeat_training(
                train_origins
            )
            specialist_a = train_ranker(
                repeat_x_a,
                repeat_y_a,
                repeat_g_a,
                specialist_rounds,
            )
            pointwise_a = train_ranker(
                repeat_x_a,
                repeat_y_a,
                repeat_g_a,
                pointwise_rounds,
                objective="binary",
            )
            lambda_scores_a = specialist_a.predict(
                val_origin.features,
                num_iteration=specialist_a.best_iteration
                or specialist_a.current_iteration(),
            )
            pointwise_scores_a = pointwise_a.predict(
                val_origin.features,
                num_iteration=pointwise_a.best_iteration
                or pointwise_a.current_iteration(),
            )
            fused_scores_a = fuse_model_ranks(
                val_origin,
                lambda_scores_a,
                pointwise_scores_a,
                float(selected_source.split("_")[1]),
            )
            model_a_val = hybrid_predictions(
                specialist_a,
                val_origin,
                val_state.pop_global,
                1.0,
                fused_scores_a,
                False,
            )
            del specialist_a, pointwise_a
            del lambda_scores_a, pointwise_scores_a, fused_scores_a
            del repeat_x_a, repeat_y_a, repeat_g_a
        elif selected_source in {"specialist", "pointwise_specialist"}:
            repeat_x_a, repeat_y_a, repeat_g_a = concatenate_repeat_training(
                train_origins
            )
            specialist_a = train_ranker(
                repeat_x_a,
                repeat_y_a,
                repeat_g_a,
                specialist_rounds
                if selected_source == "specialist"
                else pointwise_rounds,
                objective=(
                    "lambdarank"
                    if selected_source == "specialist"
                    else "binary"
                ),
            )
            model_a_val = hybrid_predictions(
                specialist_a,
                val_origin,
                val_state.pop_global,
                selected_weight,
                rerank_nonrepeat=False,
            )
            del specialist_a, repeat_x_a, repeat_y_a, repeat_g_a
        else:
            model_a_val = hybrid_predictions(
                model_a,
                val_origin,
                val_state.pop_global,
                selected_weight,
            )
        validate_predictions(
            model_a_val,
            fallback_test,
            len(val_frame),
            len(test_frame),
            static.n_ads,
        )
        save_outputs(model_a_val, fallback_test)
        print("[lane0] model_a_validation_predictions_preserved", flush=True)
        del model_a, x_model_a, y_model_a, g_model_a
        gc.collect()
        model_b_origins = train_origins + [val_origin]
        x_model_b, y_model_b, g_model_b = concatenate_training(model_b_origins)
        model_b = train_ranker(
            x_model_b,
            y_model_b,
            g_model_b,
            frozen_rounds,
        )
        if selected_source.startswith("dual_"):
            repeat_x_b, repeat_y_b, repeat_g_b = concatenate_repeat_training(
                model_b_origins
            )
            specialist_b = train_ranker(
                repeat_x_b,
                repeat_y_b,
                repeat_g_b,
                specialist_rounds,
            )
            pointwise_b = train_ranker(
                repeat_x_b,
                repeat_y_b,
                repeat_g_b,
                pointwise_rounds,
                objective="binary",
            )
            lambda_scores_b = specialist_b.predict(
                test_origin.features,
                num_iteration=specialist_b.best_iteration
                or specialist_b.current_iteration(),
            )
            pointwise_scores_b = pointwise_b.predict(
                test_origin.features,
                num_iteration=pointwise_b.best_iteration
                or pointwise_b.current_iteration(),
            )
            fused_scores_b = fuse_model_ranks(
                test_origin,
                lambda_scores_b,
                pointwise_scores_b,
                float(selected_source.split("_")[1]),
            )
            model_b_test = hybrid_predictions(
                specialist_b,
                test_origin,
                test_state.pop_global,
                1.0,
                fused_scores_b,
                False,
            )
            del specialist_b, pointwise_b
            del lambda_scores_b, pointwise_scores_b, fused_scores_b
            del repeat_x_b, repeat_y_b, repeat_g_b
        elif selected_source in {"specialist", "pointwise_specialist"}:
            repeat_x_b, repeat_y_b, repeat_g_b = concatenate_repeat_training(
                model_b_origins
            )
            specialist_b = train_ranker(
                repeat_x_b,
                repeat_y_b,
                repeat_g_b,
                specialist_rounds
                if selected_source == "specialist"
                else pointwise_rounds,
                objective=(
                    "lambdarank"
                    if selected_source == "specialist"
                    else "binary"
                ),
            )
            model_b_test = hybrid_predictions(
                specialist_b,
                test_origin,
                test_state.pop_global,
                selected_weight,
                rerank_nonrepeat=False,
            )
            del specialist_b, repeat_x_b, repeat_y_b, repeat_g_b
        else:
            model_b_test = hybrid_predictions(
                model_b,
                test_origin,
                test_state.pop_global,
                selected_weight,
            )
        validate_predictions(
            model_a_val,
            model_b_test,
            len(val_frame),
            len(test_frame),
            static.n_ads,
        )
        save_outputs(model_a_val, model_b_test)
        print("[lane0] model_b_test_predictions_saved", flush=True)
    except Exception as error:
        print(
            f"[lane0] ranker_failure_using_banked_predictions "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        if "model_a_val" in locals():
            validate_predictions(
                model_a_val,
                fallback_test,
                len(val_frame),
                len(test_frame),
                static.n_ads,
            )
            save_outputs(model_a_val, fallback_test)
        else:
            save_outputs(fallback_val, fallback_test)
    metrics = {
        "candidate_diagnostics": diagnostics,
        "internal_rrf_map12": fallback_internal,
        "internal_repeat_first_map12": repeat_internal,
        "internal_lambdarank_map12": internal_ranker_scores,
        "internal_blend_map12": blend_scores,
        "internal_specialist_blend_map12": specialist_blend_scores,
        "internal_pointwise_specialist_blend_map12": pointwise_specialist_blend_scores,
        "internal_dual_specialist_map12": dual_specialist_scores,
        "frozen_rounds": fixed_iterations,
        "elapsed_seconds": time.time() - start,
        "validation_fit": "model_A_train_labels_only",
        "test_fit": "model_B_train_plus_validation_labels",
    }
    output = run_data_dir()
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    diagnostic_path = shared / "lane0_lambdarank_v1_diagnostics.json"
    diagnostic_path.write_text(json.dumps(metrics, indent=2))
    append_artifact(
        shared,
        {
            "name": "lane0 cutoff candidate and ranker diagnostics",
            "path": diagnostic_path.name,
            "description": "Forward-origin candidate recall and internal ranker measurements",
            "content_key": "rel-avito-user-ad-visit-lane0-lambdarank-v1",
            "rebuild_hint": "Run main.py with the sanitized rel-avito cache",
        },
    )
    elapsed(start, "complete")


if __name__ == "__main__":
    main()
