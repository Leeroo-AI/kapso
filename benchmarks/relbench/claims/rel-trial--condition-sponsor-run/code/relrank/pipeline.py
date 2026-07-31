import gc
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from kapso_datasets.common import load_task

from .data import DataAssets
from .model import (
    baseline_predictions,
    blended_predictions,
    build_origin_dataset,
    candidate_recall,
    load_cached_origin,
    rank_predictions,
    slice_metrics,
    train_ranker,
    validate_predictions,
)
from .snapshot import Snapshot
from .text import TextAssets


def _rows_at(frame, year, limit=None):
    rows = frame[frame["timestamp"].dt.year == year]
    return rows.iloc[:limit] if limit is not None else rows


def _save(path, values):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.int64))
    os.replace(temporary, path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _select_baseline(assets, debug, metrics):
    years = [2018, 2019] if debug else [2017, 2018, 2019]
    scores = {0: [], 1: [], 2: []}
    for year in years:
        rows = _rows_at(assets.train, year, 200 if debug else None)
        snapshot = Snapshot(assets, rows["timestamp"].iloc[0], None, debug)
        records = snapshot.retrieve(rows["condition_id"].to_numpy(), None)
        fallback = snapshot.global_sponsors
        origin = {"candidate": candidate_recall(records, rows), "variants": {}}
        for variant in scores:
            predictions = baseline_predictions(records, rows, variant, fallback)
            value = slice_metrics(predictions, rows, snapshot)["overall"]["map"]
            scores[variant].append(value)
            origin["variants"][str(variant)] = value
        metrics["baseline_forward"][str(year)] = origin
        del records, snapshot
        gc.collect()
    means = {variant: float(np.mean(values)) for variant, values in scores.items()}
    selected = max(means, key=means.get)
    metrics["baseline_selection"] = {"means": means, "selected": int(selected)}
    print(f"[baseline] forward means={json.dumps(means)} selected={selected}")
    return selected


def _bank_baseline(assets, variant, output_dir, debug, metrics):
    predictions = {}
    for split, rows in [("val", assets.val), ("test", assets.test)]:
        snapshot = Snapshot(assets, rows["timestamp"].iloc[0], None, debug)
        active = rows.iloc[:200] if debug else rows
        records = snapshot.retrieve(active["condition_id"].to_numpy(), None)
        predictions[split] = baseline_predictions(records, rows, variant, snapshot.global_sponsors)
        metrics[f"banked_{split}_pool"] = {
            "groups_retrieved": int(len(active)),
            "mean_pool": float(np.mean([len(x.candidates) for x in records.values()])),
        }
        del records, snapshot
        gc.collect()
    validate_predictions(predictions["val"], len(assets.val), assets.n_sponsors)
    validate_predictions(predictions["test"], len(assets.test), assets.n_sponsors)
    _save(output_dir / "val_predictions.npy", predictions["val"])
    _save(output_dir / "test_predictions.npy", predictions["test"])
    print(f"[baseline] banked complete val{predictions['val'].shape} test{predictions['test'].shape}")


def _build_datasets(assets, text_assets, years, debug, metrics):
    datasets = {}
    expected_features = 550
    for year in years:
        source = assets.val if year == 2020 else assets.train
        rows = _rows_at(source, year, 120 if debug else None)
        cached = load_cached_origin(
            str(rows["timestamp"].iloc[0].date()), len(rows), expected_features, debug
        )
        if cached is not None:
            datasets[year] = cached
            print(
                f"[matrix] origin={year} groups={cached['meta']['groups']} "
                f"rows={cached['meta']['rows']} features={cached['meta']['features']} cached"
            )
            continue
        snapshot = Snapshot(assets, rows["timestamp"].iloc[0], text_assets, debug)
        records = snapshot.retrieve(rows["condition_id"].to_numpy(), text_assets)
        datasets[year] = build_origin_dataset(snapshot, records, rows, debug)
        if year in (2017, 2018, 2019):
            metrics["advanced_candidate_forward"][str(year)] = candidate_recall(records, rows)
        print(
            f"[matrix] origin={year} groups={datasets[year]['meta']['groups']} "
            f"rows={datasets[year]['meta']['rows']} features={datasets[year]['meta']['features']}"
        )
        del records, snapshot
        gc.collect()
    return datasets


def _forward_selection(assets, text_assets, datasets, debug, metrics, baseline_variant):
    folds = [(2018, [2017])] if debug else [
        (2017, [2014, 2015, 2016]),
        (2018, [2015, 2016, 2017]),
        (2019, [2016, 2017, 2018]),
    ]
    selected_iterations = []
    blend_weights = [0.0, 0.05, 0.1, 0.25, 0.5]
    blend_scores = {weight: [] for weight in blend_weights}
    metrics["ranker_forward"] = {}
    for holdout, train_years in folds:
        model = train_ranker(
            [datasets[year] for year in train_years],
            datasets[holdout],
            debug=debug,
        )
        best = int(model.best_iteration or model.current_iteration())
        choices = sorted(set([max(20 if debug else 50, int(round(best * 0.65))), best]))
        rows = _rows_at(assets.train, holdout, 150 if debug else None)
        snapshot = Snapshot(assets, rows["timestamp"].iloc[0], text_assets, debug)
        records = snapshot.retrieve(rows["condition_id"].to_numpy(), text_assets)
        predictions = rank_predictions(
            snapshot,
            records,
            rows,
            model,
            choices,
            snapshot.global_sponsors,
            batch_groups=8 if not debug else 16,
        )
        choice_metrics = []
        for iteration, values in zip(choices, predictions):
            slices = slice_metrics(values, rows, snapshot)
            choice_metrics.append({"iteration": int(iteration), "slices": slices})
        chosen = max(choice_metrics, key=lambda x: x["slices"]["overall"]["map"])
        blends = blended_predictions(
            snapshot,
            records,
            rows,
            model,
            chosen["iteration"],
            blend_weights,
            baseline_variant,
            snapshot.global_sponsors,
            batch_groups=8 if not debug else 16,
        )
        blend_results = {}
        for weight, values in zip(blend_weights, blends):
            value = slice_metrics(values, rows, snapshot)["overall"]["map"]
            blend_scores[weight].append(value)
            blend_results[str(weight)] = value
        selected_iterations.append(chosen["iteration"])
        metrics["ranker_forward"][str(holdout)] = {
            "training_origins": train_years,
            "early_stop_iteration": best,
            "candidate": candidate_recall(records, rows),
            "choices": choice_metrics,
            "selected_iteration": chosen["iteration"],
            "baseline_blends": blend_results,
        }
        print(
            f"[forward] holdout={holdout} choices="
            f"{[(x['iteration'], round(x['slices']['overall']['map'], 6)) for x in choice_metrics]} "
            f"selected={chosen['iteration']}"
        )
        del predictions, blends, records, snapshot, model
        gc.collect()
    fixed = int(np.median(selected_iterations))
    blend_means = {weight: float(np.mean(values)) for weight, values in blend_scores.items()}
    best_mean = max(blend_means.values())
    eligible = [weight for weight, value in blend_means.items() if value >= best_mean - 1e-6]
    selected_blend = min(eligible)
    metrics["fixed_tree_count"] = fixed
    metrics["blend_selection"] = {"means": blend_means, "selected": selected_blend}
    print(f"[forward] fixed median tree count={fixed} blend_means={blend_means} selected_blend={selected_blend}")
    return fixed, selected_blend


def _debug_pipeline(assets, text_assets, datasets, tree_count, blend_weight, baseline_variant, output_dir, metrics):
    if blend_weight == 0:
        metrics["final_chain"] = "banked_baseline"
        return
    train_years = [2018, 2019]
    model_a = train_ranker([datasets[year] for year in train_years], debug=True, fixed_iterations=tree_count)
    val_rows = assets.val
    val_snapshot = Snapshot(assets, val_rows["timestamp"].iloc[0], text_assets, True)
    val_active = val_rows.iloc[:150]
    val_records = val_snapshot.retrieve(val_active["condition_id"].to_numpy(), text_assets)
    val_predictions = blended_predictions(
        val_snapshot, val_records, val_rows, model_a, tree_count, [blend_weight],
        baseline_variant, val_snapshot.global_sponsors
    )[0]
    validate_predictions(val_predictions, len(val_rows), assets.n_sponsors)
    _save(output_dir / "val_predictions.npy", val_predictions)
    frozen_hash = _sha256(output_dir / "val_predictions.npy")
    del model_a, val_records, val_snapshot
    gc.collect()
    model_b = train_ranker(
        [datasets[year] for year in train_years] + [datasets[2020]],
        debug=True,
        fixed_iterations=tree_count,
    )
    test_rows = assets.test
    test_snapshot = Snapshot(assets, test_rows["timestamp"].iloc[0], text_assets, True)
    test_active = test_rows.iloc[:150]
    test_records = test_snapshot.retrieve(test_active["condition_id"].to_numpy(), text_assets)
    test_predictions = blended_predictions(
        test_snapshot, test_records, test_rows, model_b, tree_count, [blend_weight],
        baseline_variant, test_snapshot.global_sponsors
    )[0]
    validate_predictions(test_predictions, len(test_rows), assets.n_sponsors)
    if _sha256(output_dir / "val_predictions.npy") != frozen_hash:
        raise RuntimeError("frozen validation predictions changed during Model B")
    _save(output_dir / "test_predictions.npy", test_predictions)
    metrics["validation_prediction_hash"] = frozen_hash


def _full_pipeline(assets, text_assets, datasets, tree_count, blend_weight, baseline_variant, output_dir, metrics):
    if blend_weight == 0:
        frozen_hash = _sha256(output_dir / "val_predictions.npy")
        metrics["validation_prediction_hash"] = frozen_hash
        metrics["final_chain"] = "banked_baseline"
        print(f"[contract] forward selection retained banked Model A/Model B baseline sha256={frozen_hash[:16]}")
        return
    model_a_years = [2015, 2016, 2017, 2018, 2019]
    model_a = train_ranker(
        [datasets[year] for year in model_a_years],
        fixed_iterations=tree_count,
    )
    val_rows = assets.val
    val_snapshot = Snapshot(assets, val_rows["timestamp"].iloc[0], text_assets, False)
    val_records = val_snapshot.retrieve(val_rows["condition_id"].to_numpy(), text_assets)
    val_predictions = blended_predictions(
        val_snapshot,
        val_records,
        val_rows,
        model_a,
        tree_count,
        [blend_weight],
        baseline_variant,
        val_snapshot.global_sponsors,
        batch_groups=8,
    )[0]
    validate_predictions(val_predictions, len(val_rows), assets.n_sponsors)
    _save(output_dir / "val_predictions.npy", val_predictions)
    frozen_hash = _sha256(output_dir / "val_predictions.npy")
    metrics["model_a_val_candidate"] = {
        "mean_pool": float(np.mean([len(x.candidates) for x in val_records.values()])),
        "groups": int(len(val_records)),
    }
    print(f"[contract] Model A validation predictions frozen sha256={frozen_hash[:16]}")
    del model_a, val_records, val_snapshot
    gc.collect()

    model_b = train_ranker(
        [datasets[year] for year in model_a_years] + [datasets[2020]],
        fixed_iterations=tree_count,
    )
    test_rows = assets.test
    test_snapshot = Snapshot(assets, test_rows["timestamp"].iloc[0], text_assets, False)
    test_records = test_snapshot.retrieve(test_rows["condition_id"].to_numpy(), text_assets)
    test_predictions = blended_predictions(
        test_snapshot,
        test_records,
        test_rows,
        model_b,
        tree_count,
        [blend_weight],
        baseline_variant,
        test_snapshot.global_sponsors,
        batch_groups=8,
    )[0]
    validate_predictions(test_predictions, len(test_rows), assets.n_sponsors)
    if _sha256(output_dir / "val_predictions.npy") != frozen_hash:
        raise RuntimeError("frozen validation predictions changed during Model B")
    _save(output_dir / "test_predictions.npy", test_predictions)
    metrics["validation_prediction_hash"] = frozen_hash
    metrics["model_b_test_candidate"] = {
        "mean_pool": float(np.mean([len(x.candidates) for x in test_records.values()])),
        "groups": int(len(test_records)),
    }
    del model_b, test_records, test_snapshot
    gc.collect()


def run():
    warnings.filterwarnings("ignore")
    started = time.time()
    debug = "--debug" in sys.argv
    output_dir = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "./output_data_generic_exp_0"))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "debug": debug,
        "baseline_forward": {},
        "advanced_candidate_forward": {},
    }
    context = load_task()
    assets = DataAssets(context)
    variant = _select_baseline(assets, debug, metrics)
    _bank_baseline(assets, variant, output_dir, debug, metrics)
    text_assets = TextAssets(assets, debug)
    years = [2017, 2018, 2019, 2020] if debug else [2014, 2015, 2016, 2017, 2018, 2019, 2020]
    datasets = _build_datasets(assets, text_assets, years, debug, metrics)
    tree_count, blend_weight = _forward_selection(assets, text_assets, datasets, debug, metrics, variant)
    if debug:
        _debug_pipeline(assets, text_assets, datasets, tree_count, blend_weight, variant, output_dir, metrics)
    else:
        _full_pipeline(assets, text_assets, datasets, tree_count, blend_weight, variant, output_dir, metrics)
    val_predictions = np.load(output_dir / "val_predictions.npy", allow_pickle=False)
    test_predictions = np.load(output_dir / "test_predictions.npy", allow_pickle=False)
    validate_predictions(val_predictions, len(assets.val), assets.n_sponsors)
    validate_predictions(test_predictions, len(assets.test), assets.n_sponsors)
    metrics["elapsed_seconds"] = time.time() - started
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"[done] val{val_predictions.shape} test{test_predictions.shape} "
        f"elapsed={metrics['elapsed_seconds']:.1f}s"
    )
