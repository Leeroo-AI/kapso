import gc
import json
import sys
import time

import numpy as np
import pandas as pd

from hierarchical_ranker import (
    SHRINKAGES,
    assemble_scores,
    build_ranker_matrix,
    build_serving_bundle,
    bundle_eval_matrix,
    cold_probabilities,
    combine_ranker_matrices,
    evaluate_bundle,
    fit_cold_classifier,
    fit_ranker,
    load_ranker_matrix,
    ranker_predictions,
    save_ranker_matrix,
    select_design,
    subset_ranker_matrix,
    FrozenHistory,
)
from kapso_datasets.common import is_debug, load_task, run_data_dir, save_predictions, shared_cache_dir
from sales_features import attach_split, extract_documents


def labeled_frame(frame, target):
    output = frame.copy()
    output[target] = output[target].astype(np.int16)
    if target != "CUSTOMERPAYMENTTERMS":
        output = output.rename(columns={target: "CUSTOMERPAYMENTTERMS"})
    return output


def run():
    started = time.time()
    debug = is_debug()
    np.random.seed(1337)
    ctx = load_task(upto_test_timestamp=False)
    documents = extract_documents(ctx)
    train = labeled_frame(attach_split(ctx.train, documents, ctx.target_col), ctx.target_col)
    val = labeled_frame(attach_split(ctx.val, documents, ctx.target_col), ctx.target_col)
    test = attach_split(ctx.test, documents)
    del documents
    gc.collect()
    print(
        f"[phase] splits train={len(train)} val={len(val)} test={len(test)} "
        f"seconds={time.time() - started:.2f}"
    )

    history_a_started = time.time()
    history_a = FrozenHistory(train)
    print(f"[phase] history_A seconds={time.time() - history_a_started:.2f}")
    matrix_cache = shared_cache_dir() / ("lane0_ranker_horizon_v4_debug" if debug else "lane0_ranker_horizon_v4_full")
    matrix_a = load_ranker_matrix(matrix_cache)
    if matrix_a is None:
        matrix_a = build_ranker_matrix(history_a, train, debug=debug)
        save_ranker_matrix(matrix_a, matrix_cache)
        print(f"[phase] episode_cache_saved path={matrix_cache.name}")
    else:
        print(
            f"[phase] episode_cache_loaded groups={len(matrix_a.groups)} rows={len(matrix_a.y)} "
            f"path={matrix_cache.name}"
        )

    if debug:
        shrinkage = 5.0
        blend = 0.30
        selected_trees = 80
        print(f"[internal] debug fixed shrinkage={shrinkage:g} blend={blend:.2f} trees={selected_trees}")
    else:
        folds = [
            ("fold_2019_03", pd.Timestamp("2019-03-01"), pd.Timestamp("2019-08-01")),
            ("fold_2019_08", pd.Timestamp("2019-08-01"), pd.Timestamp("2020-02-01")),
        ]
        fold_results = []
        best_iterations = []
        for name, origin, end in folds:
            fold_train_matrix = subset_ranker_matrix(matrix_a, origin.strftime("%Y-%m-%d"))
            fold_eval_frame = train[
                (train["CREATIONTIMESTAMP"] >= origin) & (train["CREATIONTIMESTAMP"] < end)
            ].copy()
            fold_bundle = build_serving_bundle(
                history_a,
                fold_eval_frame,
                origin.strftime("%Y-%m-%d"),
                labels_available=True,
            )
            fold_eval_matrix = bundle_eval_matrix(fold_bundle)
            fold_model, best_iteration = fit_ranker(
                fold_train_matrix,
                fold_eval_matrix,
                debug=False,
            )
            best_iterations.append(best_iteration)
            fold_rank_scores = ranker_predictions(fold_model, fold_bundle)
            earlier = train[train["CREATIONTIMESTAMP"] < origin].copy()
            fold_cold_model = fit_cold_classifier(earlier, debug=False, trees=180)
            ordered_eval = fold_eval_frame.sort_values("_row_index").reset_index(drop=True)
            fold_cold_scores = cold_probabilities(fold_cold_model, ordered_eval)
            fold_results.append((fold_bundle, fold_rank_scores, fold_cold_scores))
            print(
                f"[internal] {name} ranker_oracle_groups={len(fold_eval_matrix.groups)} "
                f"all_seeds={len(fold_eval_frame)}"
            )
            del fold_train_matrix, fold_eval_matrix, fold_model, fold_cold_model, earlier, ordered_eval
            gc.collect()
        shrinkage, blend = select_design(fold_results)
        for fold_index, (bundle, rank_scores, cold_scores) in enumerate(fold_results):
            fold_scores = assemble_scores(bundle, rank_scores, cold_scores, shrinkage, blend)
            evaluate_bundle(bundle, fold_scores, folds[fold_index][0])
        selected_trees = int(np.median(best_iterations))
        selected_trees = max(40, min(1800, selected_trees))
        print(f"[internal] selected_trees={selected_trees} from={best_iterations}")
        del fold_results
        gc.collect()

    if "--internal-only" in sys.argv:
        print(f"[phase] internal_only_complete seconds={time.time() - started:.2f}")
        return

    model_a, _ = fit_ranker(matrix_a, eval_matrix=None, debug=debug, num_boost_round=selected_trees)
    cold_a = fit_cold_classifier(train, debug=debug)
    val_bundle = build_serving_bundle(
        history_a,
        val,
        "2020-02-01",
        labels_available=True,
    )
    val_rank_scores = ranker_predictions(model_a, val_bundle)
    val_cold_scores = cold_probabilities(cold_a, val.sort_values("_row_index").reset_index(drop=True))
    val_scores = assemble_scores(val_bundle, val_rank_scores, val_cold_scores, shrinkage, blend)
    internal_val_accuracy, _ = evaluate_bundle(val_bundle, val_scores, "chain_A_validation_diagnostic")
    np.save(run_data_dir() / "val_predictions.npy", val_scores)
    print(f"[phase] validation_saved seconds={time.time() - started:.2f}")
    del model_a, cold_a, val_bundle, val_rank_scores, val_cold_scores, history_a
    gc.collect()

    combined = pd.concat([train, val], ignore_index=True)
    history_b_started = time.time()
    history_b = FrozenHistory(combined)
    print(f"[phase] history_B seconds={time.time() - history_b_started:.2f}")
    matrix_val = build_ranker_matrix(history_b, val, debug=debug)
    matrix_b = combine_ranker_matrices(matrix_a, matrix_val)
    del matrix_a, matrix_val
    gc.collect()
    model_b, _ = fit_ranker(matrix_b, eval_matrix=None, debug=debug, num_boost_round=selected_trees)
    del matrix_b
    gc.collect()
    cold_b = fit_cold_classifier(combined, debug=debug)
    test_bundle = build_serving_bundle(
        history_b,
        test,
        "2020-07-01",
        labels_available=False,
    )
    test_rank_scores = ranker_predictions(model_b, test_bundle)
    test_cold_scores = cold_probabilities(cold_b, test.sort_values("_row_index").reset_index(drop=True))
    test_scores = assemble_scores(test_bundle, test_rank_scores, test_cold_scores, shrinkage, blend)
    save_predictions(val_scores, test_scores)
    diagnostics = {
        "debug": debug,
        "internal_val_accuracy": internal_val_accuracy,
        "shrinkage": shrinkage,
        "blend": blend,
        "ranker_trees": selected_trees,
        "elapsed_seconds": time.time() - started,
        "validation_fit": "chain_A_train_labels_only",
        "test_fit": "chain_B_train_plus_validation_labels",
    }
    (run_data_dir() / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"[phase] serialization seconds={time.time() - started:.2f}")


if __name__ == "__main__":
    run()
