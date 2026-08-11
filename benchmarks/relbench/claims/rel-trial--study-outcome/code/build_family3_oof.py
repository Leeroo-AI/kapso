from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


YEARS = (2016, 2017, 2018, 2019)
SEEDS = (17, 41, 89)
EPOCH = 2
LM_SHARE = 0.40
SIDECAR_TREES = 317


def _save(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.npy")
    np.save(temporary, np.asarray(values, dtype=np.float64))
    os.replace(temporary, path)


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def main() -> None:
    started = time.time()
    shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    output = shared / "generic_exp_4_common_oof_v1"
    source = Path(os.environ["RELBENCH_WORK_DIR"]) / "runs" / "run_0005" / "code"
    sys.path.insert(0, str(source))
    from biomed_model import build_token_bank, train_model
    from feature_pipeline import build_prepared_data
    from sidecar_model import make_sidecar_frame, train_fixed
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()

    prepared = build_prepared_data(shared, use_cache=True)
    bank = build_token_bank(prepared.frame, shared, use_cache=True)
    frame = prepared.frame.reset_index(drop=True)
    labels = frame["outcome"].fillna(0.0).to_numpy(dtype=np.float32)
    years = frame["seed_year"].to_numpy(dtype=np.float32)
    numeric = frame[prepared.lm_numeric_columns].to_numpy(dtype=np.float32)
    sidecar = make_sidecar_frame(frame, prepared.numeric_columns, prepared.categorical_columns)
    train_indices = np.flatnonzero(frame["split"].eq("train").to_numpy())
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_ids = []
    predictions = []
    source_predictions = shared / "generic_exp_1_lane1_dual_view" / "dual_view_biomedbert_sidecar_v4_tokens_320_384_pair_v2_training_v5"
    for year in YEARS:
        holdout = np.flatnonzero(frame["split"].eq("train").to_numpy() & (years == year))
        fit = np.flatnonzero(frame["split"].eq("train").to_numpy() & ((frame["timestamp"] + pd.Timedelta(days=365)) <= frame.loc[holdout, "timestamp"].min()))
        ids = frame.loc[holdout, "nct_id"].to_numpy(dtype=np.int64)
        if ids.tolist() != manifest["folds"][str(year)]["ids"]:
            raise RuntimeError(f"family3 row mismatch for {year}")
        expected_ids.extend(ids.tolist())
        sidecar_path = output / f"family3_sidecar_{year}.npy"
        if year in (2018, 2019):
            tabular = np.load(source_predictions / f"tabular_fold_{year}.npy", allow_pickle=False)
        elif sidecar_path.exists():
            tabular = np.load(sidecar_path, allow_pickle=False)
        else:
            tabular = train_fixed(sidecar, labels, years, fit, holdout, SIDECAR_TREES)
            _save(sidecar_path, tabular)
        if len(tabular) != len(holdout):
            raise RuntimeError(f"family3 sidecar length mismatch for {year}")
        members = []
        for seed in SEEDS:
            local_path = output / f"family3_lm_seed_{seed}_{year}.npy"
            source_path = source_predictions / f"lm_seed_{seed}_fold_{year}_epoch_{EPOCH}.npy"
            if year in (2018, 2019):
                member = np.load(source_path, allow_pickle=False)
            elif local_path.exists():
                member = np.load(local_path, allow_pickle=False)
            else:
                result = train_model(bank, numeric, labels, years, fit, holdout, EPOCH, seed, {EPOCH})
                member = result.predictions[EPOCH]
                _save(local_path, member)
                print(f"[family3] year={year} seed={seed} steps={result.steps} seconds={result.seconds:.1f}", flush=True)
            if len(member) != len(holdout):
                raise RuntimeError(f"family3 LM length mismatch for {year} seed {seed}")
            members.append(member)
        language = np.mean(np.column_stack(members), axis=1)
        combined = LM_SHARE * language + (1.0 - LM_SHARE) * tabular
        predictions.append(combined)
        score = roc_auc_score(labels[holdout], combined)
        print(f"[family3] year={year} rows={len(holdout)} auc={score:.6f} elapsed={time.time() - started:.1f}s", flush=True)
    oof = np.concatenate(predictions)
    if expected_ids != sum((manifest["folds"][str(year)]["ids"] for year in YEARS), []):
        raise RuntimeError("family3 common OOF order mismatch")
    _save(output / "family3_oof.npy", oof)
    run = Path(os.environ["RELBENCH_WORK_DIR"]) / "runs" / "run_0005"
    _save(output / "family3_val.npy", np.load(run / "val_predictions.npy", allow_pickle=False))
    _save(output / "family3_test.npy", np.load(run / "test_predictions.npy", allow_pickle=False))
    reproduction_path = output / "family3_reproduction_2018_seed17.npy"
    cached = np.load(source_predictions / "lm_seed_17_fold_2018_epoch_2.npy", allow_pickle=False)
    if reproduction_path.exists():
        reproduced = np.load(reproduction_path, allow_pickle=False)
    else:
        holdout = np.flatnonzero(frame["split"].eq("train").to_numpy() & (years == 2018))
        fit = np.flatnonzero(frame["split"].eq("train").to_numpy() & ((frame["timestamp"] + pd.Timedelta(days=365)) <= frame.loc[holdout, "timestamp"].min()))
        result = train_model(bank, numeric, labels, years, fit, holdout, EPOCH, 17, {EPOCH})
        reproduced = result.predictions[EPOCH]
        _save(reproduction_path, reproduced)
    reproduction = {
        "year": 2018,
        "seed": 17,
        "expected_length": int(len(cached)),
        "cached_hash": _hash(cached),
        "reproduced_hash": _hash(reproduced),
        "max_absolute_difference": float(np.max(np.abs(cached - reproduced))),
        "correlation": float(np.corrcoef(cached, reproduced)[0, 1]),
    }
    manifest["families"]["family3"] = {
        "config": {"seeds": list(SEEDS), "epoch": EPOCH, "lm_share": LM_SHARE, "sidecar_trees": SIDECAR_TREES},
        "source_run": "run_0005",
        "reproduction": reproduction,
    }
    temporary = output / "manifest.family3.partial.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, manifest_path)
    print(f"[family3] complete rows={len(oof)} elapsed={time.time() - started:.1f}s reproduction={json.dumps(reproduction)}", flush=True)


if __name__ == "__main__":
    main()
