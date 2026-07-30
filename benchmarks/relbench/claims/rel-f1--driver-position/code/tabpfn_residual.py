from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from rel_f1_model import _compact_features, fit_predict


CHECKPOINT_NAME = "tabpfn-v2-regressor.ckpt"
CHECKPOINT_URL = "https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor.ckpt"
CHECKPOINT_SHA256 = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
OOF_CACHE_VERSION = "lane2_tabpfn_v2_exact_origin_oof_v1"
SELECTION_VERSION = "lane2_tabpfn_v2_gain_selection_v1"
PROMOTION_VERSION = "lane2_tabpfn_v2_promotion_v1"
ALPHA_BASE = 0.20
ALPHA_OFFSEASON_INCREMENT = 0.05
PREDICTION_COLUMNS = ["champion_oof", "lgb_centered_oof", "catboost_oof", "recent_oof"]
RAW_EXCLUSIONS = {
    "driver_id",
    "constructor_id",
    "driver_nationality",
    "constructor_nationality",
    "last_status_id",
}


def configure_environment(cache_directory: Path, run_directory: Path) -> Path:
    root = cache_directory / "tabpfn_v2"
    paths = {
        "TMPDIR": root / "tmp",
        "TABPFN_MODEL_CACHE_DIR": root / "models",
        "MPLCONFIGDIR": root / "mpl",
        "HF_HOME": root / "hf",
        "XDG_CACHE_HOME": root / "xdg",
    }
    for name, path in paths.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            path = run_directory / name.lower()
            path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["WANDB_DISABLED"] = "true"
    return Path(os.environ["TABPFN_MODEL_CACHE_DIR"]) / CHECKPOINT_NAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ensure_checkpoint(path: Path) -> Path:
    if path.exists() and _sha256(path) == CHECKPOINT_SHA256:
        return path
    if path.exists():
        path.unlink()
    temporary = path.with_name(f"{path.name}.{os.getpid()}.download")
    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, temporary)
        digest = _sha256(temporary)
        if digest != CHECKPOINT_SHA256:
            raise RuntimeError(f"TabPFN checkpoint checksum mismatch: {digest}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def offseason_gate(features: pd.DataFrame) -> np.ndarray:
    recent_races = features["races_d60_count"].to_numpy(dtype=np.float64)
    standings = features["driver_standing_season_available"].to_numpy(dtype=np.float64)
    months = features["seed_month"].to_numpy(dtype=np.float64)
    stale = (np.nan_to_num(recent_races, nan=0.0) == 0.0).astype(np.float64)
    preseason = ((months <= 3.0) & (np.nan_to_num(standings, nan=0.0) == 0.0)).astype(np.float64)
    return np.maximum(stale, preseason)


def _context_base(features: pd.DataFrame) -> pd.DataFrame:
    compact = _compact_features(features).copy()
    result = compact.select_dtypes(include=[np.number]).drop(columns=list(RAW_EXCLUSIONS), errors="ignore")
    result["context_missing_fraction"] = result.isna().mean(axis=1).astype(np.float32)
    result["g_offseason"] = offseason_gate(features).astype(np.float32)
    return result.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def _gain_model(features: pd.DataFrame, target: np.ndarray):
    from lightgbm import LGBMRegressor

    center = (features["seed_cohort_size"].to_numpy(dtype=np.float64) + 1.0) / 2.0
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=420,
        learning_rate=0.035,
        num_leaves=23,
        max_depth=-1,
        min_child_samples=28,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.82,
        reg_alpha=0.15,
        reg_lambda=2.5,
        random_state=1337,
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")),
        verbosity=-1,
    )
    model.fit(features, target - center)
    return model


def select_context_columns(
    features: pd.DataFrame,
    target: np.ndarray,
    dates: pd.Series,
    cache_directory: Path,
) -> list[str]:
    path = cache_directory / "tabpfn_v2" / f"{SELECTION_VERSION}.json"
    if path.exists():
        values = json.loads(path.read_text())
        columns = [name for name in values["columns"] if name in _context_base(features).columns]
        if columns:
            return columns
    base = _context_base(features)
    candidates = [
        name
        for name in base.columns
        if name not in {"context_missing_fraction", "g_offseason"} and base[name].notna().mean() >= 0.05
    ]
    years = pd.to_datetime(dates).dt.year.to_numpy(dtype=np.int64)
    available = [year for year in range(1995, 2005) if np.any(years == year) and np.sum(years < year) >= 500]
    if len(available) < 10:
        distinct = sorted(set(years))
        available = [year for year in distinct if np.sum(years < year) >= 500][-10:]
    gains = []
    selections = []
    raw_slots = 90
    for year in available:
        train_mask = years < year
        model = _gain_model(base.loc[train_mask, candidates], target[train_mask])
        current = model.booster_.feature_importance(importance_type="gain").astype(np.float64)
        current = current / max(float(current.sum()), 1.0)
        gains.append(current)
        top = np.argsort(-current, kind="stable")[:raw_slots]
        chosen = np.zeros(len(candidates), dtype=np.int64)
        chosen[top] = 1
        selections.append(chosen)
    if not gains:
        raise RuntimeError("no valid inner forward folds for TabPFN feature selection")
    gain_matrix = np.vstack(gains)
    selection_matrix = np.vstack(selections)
    frequency = selection_matrix.mean(axis=0)
    median_gain = np.median(gain_matrix, axis=0)
    eligible = np.flatnonzero(frequency >= 0.70)
    order = eligible[np.lexsort((np.asarray(candidates, dtype=object)[eligible], -median_gain[eligible]))]
    selected = [candidates[index] for index in order[:raw_slots]]
    selected.extend(["context_missing_fraction", "g_offseason"])
    payload = {
        "version": SELECTION_VERSION,
        "fold_years": available,
        "fold_count": len(available),
        "selection_frequency": 0.70,
        "feature_cap": 96,
        "columns": selected,
        "raw_feature_count": len(selected) - 2,
    }
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)
    return selected


def _empty_oof() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__date_ns": pd.Series(dtype=np.int64),
            "__driver_id": pd.Series(dtype=np.int64),
            "champion_oof": pd.Series(dtype=np.float64),
            "lgb_centered_oof": pd.Series(dtype=np.float64),
            "catboost_oof": pd.Series(dtype=np.float64),
            "recent_oof": pd.Series(dtype=np.float64),
        }
    )


def build_forward_oof(
    samples: pd.DataFrame,
    features: pd.DataFrame,
    target: np.ndarray,
    cache_directory: Path,
    debug: bool,
) -> pd.DataFrame:
    mode = "debug" if debug else "full"
    source_mode = "full"
    path = cache_directory / "tabpfn_v2" / f"{OOF_CACHE_VERSION}_{source_mode}.parquet"
    keys = pd.DataFrame(
        {
            "__date_ns": pd.to_datetime(samples["date"]).astype("int64").to_numpy(dtype=np.int64),
            "__driver_id": samples["driverId"].to_numpy(dtype=np.int64),
        }
    )
    if path.exists():
        try:
            cache = pd.read_parquet(path)
        except Exception:
            cache = _empty_oof()
    else:
        cache = _empty_oof()
    cache = cache.drop_duplicates(["__date_ns", "__driver_id"], keep="last")
    cache_index = pd.MultiIndex.from_frame(cache[["__date_ns", "__driver_id"]])
    key_index = pd.MultiIndex.from_frame(keys)
    present = key_index.isin(cache_index)
    date_values = keys["__date_ns"].to_numpy(dtype=np.int64)
    minimum_rows = 500
    missing_origins = np.array([], dtype=np.int64) if debug else np.unique(date_values[~present])
    completed = 0
    for origin in missing_origins:
        train_mask = date_values <= origin - 60 * 86_400_000_000_000
        predict_mask = date_values == origin
        if int(train_mask.sum()) < minimum_rows:
            continue
        prediction, parts = fit_predict(
            features.loc[train_mask].reset_index(drop=True),
            target[train_mask],
            features.loc[predict_mask].reset_index(drop=True),
            debug=debug,
        )
        additions = keys.loc[predict_mask].copy()
        additions["champion_oof"] = prediction
        additions["lgb_centered_oof"] = parts["lgb_centered"]
        additions["catboost_oof"] = parts["catboost"]
        additions["recent_oof"] = parts["recent"]
        cache = pd.concat([cache, additions], ignore_index=True)
        cache = cache.drop_duplicates(["__date_ns", "__driver_id"], keep="last")
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        cache.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        completed += 1
        if completed % 20 == 0:
            print(f"[oof] completed_origins={completed}/{len(missing_origins)}", flush=True)
    indexed = cache.set_index(["__date_ns", "__driver_id"])
    result = indexed.reindex(key_index).reset_index(drop=True)
    result.insert(0, "__driver_id", keys["__driver_id"].to_numpy())
    result.insert(0, "__date_ns", keys["__date_ns"].to_numpy())
    usable = int(result["champion_oof"].notna().sum())
    print(
        f"[oof] mode={mode} source={source_mode} cached_rows={usable}/{len(result)} "
        f"new_origins={completed}",
        flush=True,
    )
    return result


def append_crossfit_predictions(
    base_oof: pd.DataFrame,
    samples: pd.DataFrame,
    prediction: np.ndarray,
    parts: dict[str, np.ndarray],
) -> pd.DataFrame:
    additions = pd.DataFrame(
        {
            "__date_ns": pd.to_datetime(samples["date"]).astype("int64").to_numpy(dtype=np.int64),
            "__driver_id": samples["driverId"].to_numpy(dtype=np.int64),
            "champion_oof": prediction,
            "lgb_centered_oof": parts["lgb_centered"],
            "catboost_oof": parts["catboost"],
            "recent_oof": parts["recent"],
        }
    )
    return pd.concat([base_oof, additions], ignore_index=True)


def _assemble_context(
    features: pd.DataFrame,
    predictions: pd.DataFrame | dict[str, np.ndarray],
    selected: list[str],
) -> pd.DataFrame:
    base = _context_base(features)
    context = pd.DataFrame(index=np.arange(len(features)))
    for name in PREDICTION_COLUMNS:
        if isinstance(predictions, pd.DataFrame):
            context[name] = predictions[name].to_numpy(dtype=np.float32)
        else:
            source = {
                "champion_oof": "champion",
                "lgb_centered_oof": "lgb_centered",
                "catboost_oof": "catboost",
                "recent_oof": "recent",
            }[name]
            context[name] = np.asarray(predictions[source], dtype=np.float32)
    for name in selected:
        context[name] = base[name].to_numpy(dtype=np.float32)
    return context.iloc[:, :96].replace([np.inf, -np.inf], np.nan).astype(np.float32)


def _sample_context(
    context: pd.DataFrame,
    residual: np.ndarray,
    samples: pd.DataFrame,
    features: pd.DataFrame,
    cap: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(context) <= cap:
        return context.reset_index(drop=True), residual
    dates = pd.to_datetime(samples["date"]).reset_index(drop=True)
    latest = dates.max()
    recent_mask = dates >= latest - pd.DateOffset(years=8)
    recent_indices = np.flatnonzero(recent_mask.to_numpy())
    recent_limit = min(len(recent_indices), cap // 2)
    recent_indices = recent_indices[-recent_limit:]
    older_indices = np.flatnonzero(~recent_mask.to_numpy())
    metadata = pd.DataFrame(
        {
            "index": older_indices,
            "year": dates.iloc[older_indices].dt.year.to_numpy(),
            "phase": offseason_gate(features.iloc[older_indices]).astype(np.int64),
            "history": np.floor(
                np.log2(
                    1.0
                    + np.nan_to_num(
                        features.iloc[older_indices]["label_history_count"].to_numpy(dtype=np.float64),
                        nan=0.0,
                    )
                )
            ).astype(np.int64),
            "origin": dates.iloc[older_indices].astype("int64").to_numpy(),
            "driver": samples.iloc[older_indices]["driverId"].to_numpy(dtype=np.int64),
        }
    )
    group = ["year", "phase", "history", "origin"]
    metadata["driver_rank"] = metadata.sort_values("driver").groupby(group, observed=True).cumcount()
    metadata = metadata.sort_values(["driver_rank", *group, "driver"], kind="stable")
    old_limit = cap - len(recent_indices)
    selected = np.concatenate([metadata["index"].to_numpy(dtype=np.int64)[:old_limit], recent_indices])
    selected.sort()
    return context.iloc[selected].reset_index(drop=True), residual[selected]


def _regressor(checkpoint: Path, estimators: int):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    visible_device = os.environ.get("CUDA_DEVICE", "0")
    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V2,
        n_estimators=estimators,
        auto_scale_n_estimators=False,
        model_path=checkpoint,
        device=f"cuda:{visible_device}",
        fit_mode="fit_with_cache",
        ignore_pretraining_limits=True,
        n_preprocessing_jobs=1,
        show_progress_bar=False,
        random_state=1337,
    )


def apply_residual_specialist(
    train_samples: pd.DataFrame,
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    train_oof: pd.DataFrame,
    predict_features: pd.DataFrame,
    champion: np.ndarray,
    components: dict[str, np.ndarray],
    selected: list[str],
    checkpoint: Path,
    debug: bool,
    enabled: bool = True,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    if not enabled:
        return champion.copy(), {"enabled": False, "reason": "promotion_gate"}
    residual = train_target - train_oof["champion_oof"].to_numpy(dtype=np.float64)
    usable = np.isfinite(residual)
    for name in PREDICTION_COLUMNS:
        usable &= np.isfinite(train_oof[name].to_numpy(dtype=np.float64))
    residual = np.clip(residual[usable], -8.0, 8.0)
    context = _assemble_context(
        train_features.loc[usable].reset_index(drop=True),
        train_oof.loc[usable].reset_index(drop=True),
        selected,
    )
    samples = train_samples.loc[usable].reset_index(drop=True)
    usable_features = train_features.loc[usable].reset_index(drop=True)
    cap = 2_000 if debug else 10_000
    context, residual = _sample_context(context, residual, samples, usable_features, cap)
    estimators = 1 if debug else 4
    model = _regressor(checkpoint, estimators)
    started = time.perf_counter()
    model.fit(context, residual)
    predict_parts = {
        "champion": champion,
        "lgb_centered": components["lgb_centered"],
        "catboost": components["catboost"],
        "recent": components["recent"],
    }
    predict_context = _assemble_context(predict_features, predict_parts, selected)
    correction = np.asarray(model.predict(predict_context), dtype=np.float64)
    correction = np.nan_to_num(correction, nan=0.0, posinf=6.0, neginf=-6.0)
    correction = np.clip(correction, -6.0, 6.0)
    gate = offseason_gate(predict_features)
    alpha = ALPHA_BASE + ALPHA_OFFSEASON_INCREMENT * gate
    prediction = champion + alpha * correction
    upper = np.maximum(
        18.0,
        np.nan_to_num(predict_features["races_d365_field_mean"].to_numpy(dtype=np.float64), nan=26.0) + 2.0,
    )
    prediction = np.clip(prediction, 1.0, upper)
    diagnostics = {
        "enabled": True,
        "context_rows": len(context),
        "context_columns": context.shape[1],
        "estimators": estimators,
        "fit_predict_seconds": time.perf_counter() - started,
        "residual_mean": float(correction.mean()),
        "residual_std": float(correction.std()),
        "offseason_rows": int(gate.sum()),
    }
    return prediction.astype(np.float64), diagnostics


def promotion_enabled(cache_directory: Path) -> bool:
    path = cache_directory / "tabpfn_v2" / f"{PROMOTION_VERSION}.json"
    if not path.exists():
        return True
    return bool(json.loads(path.read_text()).get("promoted", False))


def evaluate_forward_gate(
    samples: pd.DataFrame,
    features: pd.DataFrame,
    target: np.ndarray,
    oof: pd.DataFrame,
    selected: list[str],
    checkpoint: Path,
    cache_directory: Path,
) -> dict:
    dates = pd.to_datetime(samples["date"]).reset_index(drop=True)
    years = dates.dt.year.to_numpy(dtype=np.int64)
    usable = oof["champion_oof"].notna().to_numpy()
    predictions = np.full(len(samples), np.nan, dtype=np.float64)
    corrections = np.full(len(samples), np.nan, dtype=np.float64)
    held_years = list(range(1995, 2005))
    for year in held_years:
        train_mask = usable & (years < year)
        predict_mask = usable & (years == year)
        if train_mask.sum() < 500 or predict_mask.sum() == 0:
            continue
        residual = np.clip(
            target[train_mask] - oof.loc[train_mask, "champion_oof"].to_numpy(dtype=np.float64),
            -8.0,
            8.0,
        )
        context = _assemble_context(
            features.loc[train_mask].reset_index(drop=True),
            oof.loc[train_mask].reset_index(drop=True),
            selected,
        )
        context, residual = _sample_context(
            context,
            residual,
            samples.loc[train_mask].reset_index(drop=True),
            features.loc[train_mask].reset_index(drop=True),
            10_000,
        )
        model = _regressor(checkpoint, 4)
        model.fit(context, residual)
        held_context = _assemble_context(
            features.loc[predict_mask].reset_index(drop=True),
            oof.loc[predict_mask].reset_index(drop=True),
            selected,
        )
        correction = np.clip(np.asarray(model.predict(held_context), dtype=np.float64), -6.0, 6.0)
        held_features = features.loc[predict_mask].reset_index(drop=True)
        alpha = ALPHA_BASE + ALPHA_OFFSEASON_INCREMENT * offseason_gate(held_features)
        held_prediction = oof.loc[predict_mask, "champion_oof"].to_numpy(dtype=np.float64) + alpha * correction
        upper = np.maximum(
            18.0,
            np.nan_to_num(
                held_features["races_d365_field_mean"].to_numpy(dtype=np.float64),
                nan=26.0,
            )
            + 2.0,
        )
        predictions[predict_mask] = np.clip(held_prediction, 1.0, upper)
        corrections[predict_mask] = correction
        print(f"[gate] held_year={year} train={int(train_mask.sum())} predict={int(predict_mask.sum())}", flush=True)
    evaluated = np.isfinite(predictions)
    base = oof.loc[evaluated, "champion_oof"].to_numpy(dtype=np.float64)
    candidate = predictions[evaluated]
    truth = target[evaluated]
    evaluated_dates = dates.loc[evaluated].reset_index(drop=True)
    evaluated_years = evaluated_dates.dt.year.to_numpy(dtype=np.int64)
    evaluated_features = features.loc[evaluated].reset_index(drop=True)
    row_delta = np.abs(candidate - truth) - np.abs(base - truth)
    origin_frame = pd.DataFrame({"date": evaluated_dates, "delta": row_delta})
    origin_delta = origin_frame.groupby("date", observed=True)["delta"].mean().to_numpy(dtype=np.float64)
    standard_error = float(origin_delta.std(ddof=1) / math.sqrt(len(origin_delta)))
    random = np.random.default_rng(1337)
    samples_index = random.integers(0, len(origin_delta), size=(2_000, len(origin_delta)))
    bootstrap = origin_delta[samples_index].mean(axis=1)
    year_rows = []
    season_wins = 0
    for year in held_years:
        mask = evaluated_years == year
        base_mae = float(np.mean(np.abs(base[mask] - truth[mask])))
        candidate_mae = float(np.mean(np.abs(candidate[mask] - truth[mask])))
        season_wins += int(candidate_mae < base_mae)
        year_rows.append(
            {
                "year": year,
                "count": int(mask.sum()),
                "champion_mae": base_mae,
                "specialist_mae": candidate_mae,
                "delta": candidate_mae - base_mae,
            }
        )
    preseason = offseason_gate(evaluated_features) > 0.5
    champion_mae = float(np.mean(np.abs(base - truth)))
    specialist_mae = float(np.mean(np.abs(candidate - truth)))
    preseason_base = float(np.mean(np.abs(base[preseason] - truth[preseason])))
    preseason_specialist = float(np.mean(np.abs(candidate[preseason] - truth[preseason])))
    worst_base = max(row["champion_mae"] for row in year_rows)
    worst_specialist = max(row["specialist_mae"] for row in year_rows)
    mean_delta = float(origin_delta.mean())
    gates = {
        "two_standard_errors": bool(mean_delta + 2.0 * standard_error < 0.0),
        "preseason": bool(preseason_specialist < preseason_base),
        "worst_year": bool(worst_specialist < worst_base),
        "seven_of_ten_seasons": bool(season_wins >= 7),
    }
    result = {
        "version": PROMOTION_VERSION,
        "alpha_base": ALPHA_BASE,
        "alpha_offseason_increment": ALPHA_OFFSEASON_INCREMENT,
        "promoted": bool(all(gates.values())),
        "gates": gates,
        "rows": int(evaluated.sum()),
        "origins": len(origin_delta),
        "champion_mae": champion_mae,
        "specialist_mae": specialist_mae,
        "mean_origin_delta": mean_delta,
        "origin_standard_error": standard_error,
        "bootstrap_delta_q025": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_delta_q975": float(np.quantile(bootstrap, 0.975)),
        "preseason_count": int(preseason.sum()),
        "preseason_champion_mae": preseason_base,
        "preseason_specialist_mae": preseason_specialist,
        "worst_year_champion_mae": worst_base,
        "worst_year_specialist_mae": worst_specialist,
        "season_wins": season_wins,
        "years": year_rows,
    }
    gate_rows = pd.DataFrame(
        {
            "__date_ns": pd.to_datetime(samples.loc[evaluated, "date"]).astype("int64").to_numpy(dtype=np.int64),
            "__driver_id": samples.loc[evaluated, "driverId"].to_numpy(dtype=np.int64),
            "target": truth,
            "champion": base,
            "raw_correction": corrections[evaluated],
            "g_offseason": offseason_gate(evaluated_features),
            "upper": np.maximum(
                18.0,
                np.nan_to_num(
                    evaluated_features["races_d365_field_mean"].to_numpy(dtype=np.float64),
                    nan=26.0,
                )
                + 2.0,
            ),
        }
    )
    gate_path = cache_directory / "tabpfn_v2" / f"{PROMOTION_VERSION}_rows.parquet"
    gate_temporary = gate_path.with_name(f"{gate_path.name}.{os.getpid()}.tmp")
    gate_rows.to_parquet(gate_temporary, index=False)
    os.replace(gate_temporary, gate_path)
    path = cache_directory / "tabpfn_v2" / f"{PROMOTION_VERSION}.json"
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True))
    os.replace(temporary, path)
    return result
