from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from kapso_datasets.common import save_predictions, is_debug
from calendar_forecast import (
    NUM_DST,
    build_season_representations,
    build_presence_training,
    fit_presence_model,
    fit_round_model,
)
from driver_attendance import (
    build_participation,
    build_attendance_training,
    fit_attendance_model,
)
from scoring import DEFAULT_META, rank_cohort, rolling_origin_score_loo

SEED = 13
np.random.seed(SEED)

K = 10
FIRST_TARGET_YEAR = 1985
DEBUG_MIN_YEAR_OFFSET = 6
SELECTION_WINDOW = 10
MEAN_WORST_WEIGHT = (0.7, 0.3)
PARSIMONY_GAMMA = 0.006


def build_year_driver_circuits(results, races):
    merged = results[["driverId", "raceId", "date"]].merge(
        races[["raceId", "circuitId", "year"]], on="raceId", how="left"
    )
    merged = merged.dropna(subset=["driverId", "circuitId", "year"])
    merged["driverId"] = merged["driverId"].astype(int)
    merged["circuitId"] = merged["circuitId"].astype(int)
    merged["year"] = merged["year"].astype(int)
    out = {}
    for (year, did), grp in merged.groupby(["year", "driverId"]):
        out.setdefault(int(year), {})[int(did)] = set(grp["circuitId"].tolist())
    return out


def meta_grid():
    grid = []
    for decay in (0.82, 0.9):
        for blend in ((0.7, 0.2, 0.1), (0.85, 0.1, 0.05), (1.0, 0.0, 0.0)):
            for affinity_weight in (0.0, 0.1, 0.2):
                for gamma in (0.3, 0.6, 1.0):
                    m = dict(DEFAULT_META)
                    m["decay"] = decay
                    m["blend"] = blend
                    m["affinity_weight"] = affinity_weight
                    m["incumbent_gamma"] = gamma
                    grid.append(m)
    return grid


DECAY_OPTIONS = (0.82, 0.9)


def build_loo_models(season_circuits, season_round_pos, driver_events, driver_year_rounds,
                     year_cohort, dob, season_n_rounds, target_years, score_years, decay, debug):
    loo = {}
    for ty in score_years:
        yrs = [y for y in target_years if y != ty]
        loo[ty] = fit_models(
            season_circuits, season_round_pos, driver_events, driver_year_rounds,
            year_cohort, dob, season_n_rounds, yrs, decay, debug,
        )
    return loo


def fit_models(season_circuits, season_round_pos, driver_events, driver_year_rounds,
               year_cohort, dob, season_n_rounds, target_years, decay, debug):
    Xp, yp, Xr, yr = build_presence_training(season_circuits, season_round_pos, target_years, decay)
    presence_model = fit_presence_model(Xp, yp, debug)
    round_model = fit_round_model(Xr, yr, debug)
    Xa, ya = build_attendance_training(
        driver_events, driver_year_rounds, year_cohort, dob, season_n_rounds, target_years
    )
    attend_model = fit_attendance_model(Xa, ya, debug)
    return presence_model, round_model, attend_model


def select_meta(driver_events, driver_year_rounds, year_cohort, dob, season_circuits,
                season_round_pos, season_n_rounds, year_driver_circuits, target_years,
                debug, refit, build_loo):
    if debug:
        pm, rm, am = refit(DEFAULT_META["decay"])
        return DEFAULT_META, pm, rm, am
    score_years = [y for y in target_years if y >= max(target_years) - SELECTION_WINDOW + 1]
    loo_cache = {d: build_loo(d, score_years) for d in DECAY_OPTIONS}
    best, best_obj, best_score, best_worst = None, -1.0, -1.0, -1.0
    wm, ww = MEAN_WORST_WEIGHT
    for m in meta_grid():
        loo = loo_cache[m["decay"]]
        mean_map, worst_map = rolling_origin_score_loo(
            driver_events, driver_year_rounds, year_cohort, dob, season_circuits,
            season_round_pos, season_n_rounds, year_driver_circuits, loo, score_years, m, K,
        )
        combined = wm * mean_map + ww * worst_map
        objective = combined + PARSIMONY_GAMMA * m["incumbent_gamma"]
        if objective > best_obj:
            best_obj, best_score, best, best_worst = objective, combined, m, worst_map
    pm, rm, am = refit(best["decay"])
    print(f"[grid] LOO best combined={best_score:.4f} worst={best_worst:.4f} "
          f"decay={best['decay']} blend={best['blend']} aff={best['affinity_weight']} "
          f"gamma={best['incumbent_gamma']} score_years={min(score_years)}..{max(score_years)}")
    return best, pm, rm, am


def predict_split(split_df, cohort_ids, ty, latest_year, driver_events, driver_year_rounds,
                  dob, season_circuits, season_round_pos, season_n_rounds,
                  presence_model, round_model, attend_model, meta):
    seed_time = split_df["date"].iloc[0]
    preds = rank_cohort(
        driver_events, driver_year_rounds, dob, cohort_ids, seed_time, ty, latest_year,
        season_circuits, season_round_pos, season_n_rounds, presence_model, round_model,
        attend_model, meta, K,
    )
    out = np.zeros((len(split_df), K), dtype=np.int64)
    for i, row in enumerate(split_df.itertuples(index=False)):
        out[i] = preds[int(getattr(row, "driverId"))]
    return out


def validate_preds(pred, name):
    assert pred.shape == (27, K), f"{name} shape {pred.shape}"
    assert np.issubdtype(pred.dtype, np.integer), f"{name} dtype"
    for r in pred:
        assert len(set(r.tolist())) == K, f"{name} duplicate ids"
        assert r.min() >= 0 and r.max() < NUM_DST, f"{name} id out of range"


def main():
    t0 = time.time()
    debug = is_debug()
    print(f"[main] mode={'debug' if debug else 'full'} seed={SEED}")

    dataset = get_dataset("rel-f1", download=False)
    task = get_task("rel-f1", "driver-circuit-compete", download=False)
    db = dataset.get_db()

    val = task.get_table("val").df
    test = task.get_table("test").df

    races = db.table_dict["races"].df
    results = db.table_dict["results"].df
    drivers = db.table_dict["drivers"].df

    season_circuits, season_round_pos, season_bounds, season_n_rounds = build_season_representations(races)
    driver_events, driver_year_rounds, year_cohort, dob = build_participation(results, races, drivers)
    year_driver_circuits = build_year_driver_circuits(results, races)
    print(f"[main] seasons={min(season_circuits)}..{max(season_circuits)} "
          f"drivers={len(driver_events)} elapsed={time.time()-t0:.1f}s")

    val_seed_time = val["date"].iloc[0]
    test_seed_time = test["date"].iloc[0]
    val_ty = int(pd.Timestamp(val_seed_time).year)
    test_ty = int(pd.Timestamp(test_seed_time).year)

    def target_years_upto(seed_time):
        ys = [y for y in sorted(season_circuits)
              if pd.Timestamp(f"{y+1}-01-01") <= seed_time and y >= FIRST_TARGET_YEAR]
        if debug:
            ys = ys[-DEBUG_MIN_YEAR_OFFSET:]
        return ys

    def latest_year_for(seed_time):
        return max(y for y in season_circuits if season_bounds[y][1] <= seed_time)

    val_cohort = sorted(int(d) for d in val["driverId"].tolist())
    test_cohort = sorted(int(d) for d in test["driverId"].tolist())

    # Model A -> val predictions (never sees val labels / val target year)
    val_targets = target_years_upto(val_seed_time)
    print(f"[main] model A target years {min(val_targets)}..{max(val_targets)} n={len(val_targets)}")

    def refit_A(decay):
        return fit_models(season_circuits, season_round_pos, driver_events, driver_year_rounds,
                          year_cohort, dob, season_n_rounds, val_targets, decay, debug)

    def build_loo_A(decay, score_years):
        return build_loo_models(season_circuits, season_round_pos, driver_events,
                                driver_year_rounds, year_cohort, dob, season_n_rounds,
                                val_targets, score_years, decay, debug)

    meta_A, pmA, rmA, amA = select_meta(
        driver_events, driver_year_rounds, year_cohort, dob, season_circuits,
        season_round_pos, season_n_rounds, year_driver_circuits, val_targets, debug,
        refit_A, build_loo_A,
    )
    val_latest = latest_year_for(val_seed_time)
    val_pred = predict_split(
        val, val_cohort, val_ty, val_latest, driver_events, driver_year_rounds, dob,
        season_circuits, season_round_pos, season_n_rounds, pmA, rmA, amA, meta_A,
    )
    print(f"[main] val predictions done elapsed={time.time()-t0:.1f}s")

    # Model B -> test predictions (all legal history incl. val year)
    test_targets = target_years_upto(test_seed_time)
    print(f"[main] model B target years {min(test_targets)}..{max(test_targets)} n={len(test_targets)}")

    def refit_B(decay):
        return fit_models(season_circuits, season_round_pos, driver_events, driver_year_rounds,
                          year_cohort, dob, season_n_rounds, test_targets, decay, debug)

    def build_loo_B(decay, score_years):
        return build_loo_models(season_circuits, season_round_pos, driver_events,
                                driver_year_rounds, year_cohort, dob, season_n_rounds,
                                test_targets, score_years, decay, debug)

    meta_B, pmB, rmB, amB = select_meta(
        driver_events, driver_year_rounds, year_cohort, dob, season_circuits,
        season_round_pos, season_n_rounds, year_driver_circuits, test_targets, debug,
        refit_B, build_loo_B,
    )
    test_latest = latest_year_for(test_seed_time)
    test_pred = predict_split(
        test, test_cohort, test_ty, test_latest, driver_events, driver_year_rounds, dob,
        season_circuits, season_round_pos, season_n_rounds, pmB, rmB, amB, meta_B,
    )
    print(f"[main] test predictions done elapsed={time.time()-t0:.1f}s")

    validate_preds(val_pred, "val")
    validate_preds(test_pred, "test")
    save_predictions(val_pred, test_pred)
    print(f"[main] finished elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
