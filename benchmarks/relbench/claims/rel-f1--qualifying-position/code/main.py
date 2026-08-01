# Imports

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from f1_pipeline import (
    SEEDS,
    add_rating_features,
    cross_validate_residual,
    expanding_folds,
    fit_ensemble,
    load_or_build_features,
    project_races,
    select_rating,
)
from kapso_datasets.common import load_task, run_data_dir, save_predictions, shared_cache_dir


# Execution

def main() -> None:
    started = time.time()
    debug = "--debug" in sys.argv
    context = load_task(upto_test_timestamp=False)
    print(f"[pace] loaded task in {time.time() - started:.2f}s")
    feature_started = time.time()
    features, source_max, cache_hit = load_or_build_features(context, shared_cache_dir())
    print(f"[pace] features rows={len(features)} columns={len(features.columns)} cache_hit={cache_hit} elapsed={time.time() - feature_started:.2f}s")
    print(f"[pace] maximum contributing timestamps {json.dumps(source_max, sort_keys=True)}")
    train_y = context.train.df[context.target_col].to_numpy(dtype=np.float64)
    val_y = context.val.df[context.target_col].to_numpy(dtype=np.float64) if len(context.val.df) else np.empty(0, dtype=np.float64)
    train_base = features[features["split"] == "train"].reset_index(drop=True)
    folds = expanding_folds(train_base)
    prior, temperature, rating_records = select_rating(features, train_y, folds)
    rated = add_rating_features(features, prior, temperature)
    train_frame = rated[rated["split"] == "train"].reset_index(drop=True)
    val_frame = rated[rated["split"] == "val"].reset_index(drop=True)
    test_frame = rated[rated["split"] == "test"].reset_index(drop=True)
    rating_winner = max(rating_records, key=lambda record: (record["stability_score"], record["pooled_r2"]))
    print(f"[pace] rating prior={prior} temperature={temperature} pooled_r2={rating_winner['pooled_r2']:.6f} folds={rating_winner['fold_r2']}")
    selection, selection_records, cv_elapsed = cross_validate_residual(train_frame, train_y, folds, debug)
    tree_rate = float(selection["trees"] * len(SEEDS) * len(folds) * 2 / max(cv_elapsed, 1e-9))
    print(f"[pace] residual selection group={selection['feature_group']} trees={selection['trees']} blend={selection['blend']} pooled_r2={selection['pooled_r2']:.6f} folds={selection['fold_r2']} tree_rate={tree_rate:.1f}_trees_per_second elapsed={cv_elapsed:.2f}s")
    chain_a_started = time.time()
    validation_residual = fit_ensemble(train_frame, train_y, val_frame, selection["feature_group"], int(selection["trees"]))
    validation_prediction = project_races(val_frame, val_frame["r_rating"].to_numpy() + float(selection["blend"]) * validation_residual)
    output_dir = run_data_dir()
    np.save(output_dir / "val_predictions.npy", validation_prediction.astype(np.float64))
    print(f"[pace] Chain A validation predictions frozen before refit elapsed={time.time() - chain_a_started:.2f}s")
    chain_b_started = time.time()
    combined_frame = pd.concat([train_frame, val_frame], ignore_index=True)
    combined_y = np.concatenate([train_y, val_y])
    test_residual = fit_ensemble(combined_frame, combined_y, test_frame, selection["feature_group"], int(selection["trees"]))
    test_prediction = project_races(test_frame, test_frame["r_rating"].to_numpy() + float(selection["blend"]) * test_residual)
    save_predictions(validation_prediction.astype(np.float64), test_prediction.astype(np.float64))
    validation_union = np.concatenate([validation for _, validation in folds])
    chosen_records = [record for record in selection_records if record["feature_group"] == selection["feature_group"] and record["trees"] == selection["trees"] and record["blend"] == selection["blend"]]
    diagnostics = {
        "debug": debug,
        "source_max": source_max,
        "rating_selection": rating_records,
        "residual_selection": selection_records,
        "winner": selection,
        "rating_winner": rating_winner,
        "tree_rate": tree_rate,
        "feature_shape": list(features.shape),
        "seeds": list(SEEDS),
        "elapsed_seconds": time.time() - started,
        "chain_b_seconds": time.time() - chain_b_started,
    }
    if chosen_records:
        diagnostics["selected_internal"] = chosen_records[0]
    (output_dir / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"[pace] Chain B test predictions complete elapsed={time.time() - chain_b_started:.2f}s total={time.time() - started:.2f}s")


if __name__ == "__main__":
    main()
