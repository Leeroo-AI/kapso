from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

INCUMBENT_FRAC = 0.5

LGB_ATTEND = dict(
    objective="binary",
    num_leaves=15,
    max_depth=4,
    learning_rate=0.05,
    min_child_samples=40,
    reg_lambda=8.0,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
    random_state=13,
    n_jobs=1,
    n_estimators=200,
)


# Participation events and per-driver / per-(driver,year) structures
def build_participation(results: pd.DataFrame, races: pd.DataFrame, drivers: pd.DataFrame):
    merged = results[["driverId", "raceId", "constructorId", "date"]].merge(
        races[["raceId", "circuitId", "year", "round"]], on="raceId", how="left"
    )
    merged = merged.dropna(subset=["driverId", "circuitId", "year", "round", "date"])
    merged["driverId"] = merged["driverId"].astype(int)
    merged["circuitId"] = merged["circuitId"].astype(int)
    merged["year"] = merged["year"].astype(int)
    merged["round"] = merged["round"].astype(int)
    merged = merged.sort_values("date")

    driver_events = {}
    for did, grp in merged.groupby("driverId"):
        driver_events[int(did)] = dict(
            date=grp["date"].to_numpy(),
            year=grp["year"].to_numpy(dtype=np.int64),
            round=grp["round"].to_numpy(dtype=np.int64),
            circuitId=grp["circuitId"].to_numpy(dtype=np.int64),
            constructorId=grp["constructorId"].fillna(-1).to_numpy(dtype=np.int64),
        )

    driver_year_rounds = {}
    year_cohort = {}
    for (did, year), grp in merged.groupby(["driverId", "year"]):
        driver_year_rounds[(int(did), int(year))] = set(grp["round"].astype(int).tolist())
        year_cohort.setdefault(int(year), set()).add(int(did))

    dob = {}
    for _, r in drivers[["driverId", "dob"]].dropna(subset=["driverId"]).iterrows():
        dob[int(r["driverId"])] = r["dob"]
    return driver_events, driver_year_rounds, year_cohort, dob


ATTEND_DRIVER_FEATURES = [
    "days_since_last", "starts_90", "starts_180", "starts_365", "starts_730",
    "prev_first_round", "prev_last_round", "frac_prev_rounds", "num_constructors_365",
    "career_starts", "age", "debut_recency", "prior_gap",
    "incumbent", "rookie", "returnee",
]
ATTEND_COHORT_FEATURES = [
    "cohort_rank_recent", "cohort_rank_recency", "cohort_n_incumbent", "cohort_n_rookie",
]
ATTEND_ROUND_FEATURES = [
    "round_idx", "round_norm", "inc_x_round", "rookie_x_round",
]
ATTEND_FEATURE_NAMES = ATTEND_DRIVER_FEATURES + ATTEND_COHORT_FEATURES + ATTEND_ROUND_FEATURES


def _driver_features(driver_events, dob, driver_id, seed_time, ty, latest_year, n_rounds_prev):
    seed64 = np.datetime64(seed_time)
    e = driver_events.get(driver_id)
    if e is None:
        return dict(
            days_since_last=3650.0, starts_90=0, starts_180=0, starts_365=0, starts_730=0,
            prev_first_round=0.0, prev_last_round=0.0, frac_prev_rounds=0.0,
            num_constructors_365=0, career_starts=0, age=25.0, debut_recency=0.0,
            prior_gap=0.0, incumbent=0.0, rookie=1.0, returnee=0.0,
        )
    mask = e["date"] <= seed64
    if not mask.any():
        return dict(
            days_since_last=3650.0, starts_90=0, starts_180=0, starts_365=0, starts_730=0,
            prev_first_round=0.0, prev_last_round=0.0, frac_prev_rounds=0.0,
            num_constructors_365=0, career_starts=0, age=25.0, debut_recency=0.0,
            prior_gap=0.0, incumbent=0.0, rookie=1.0, returnee=0.0,
        )
    dates = e["date"][mask]
    years = e["year"][mask]
    rounds = e["round"][mask]
    ctors = e["constructorId"][mask]

    days = (seed64 - dates) / np.timedelta64(1, "D")
    days_since_last = float(days.min())
    starts_90 = int((days <= 90).sum())
    starts_180 = int((days <= 180).sum())
    starts_365 = int((days <= 365).sum())
    starts_730 = int((days <= 730).sum())
    num_constructors_365 = int(len(np.unique(ctors[days <= 365]))) if starts_365 else 0
    career_starts = int(mask.sum())

    prev_mask = years == latest_year
    if prev_mask.any():
        pr = rounds[prev_mask]
        prev_first_round = float(pr.min())
        prev_last_round = float(pr.max())
        frac_prev_rounds = float(len(np.unique(pr)) / n_rounds_prev) if n_rounds_prev else 0.0
    else:
        prev_first_round = 0.0
        prev_last_round = 0.0
        frac_prev_rounds = 0.0

    first_year = int(years.min())
    debut_recency = float(ty - first_year)
    seen_years = np.unique(years)
    prior_gap = 0.0
    if len(seen_years) >= 1:
        prior_gap = float(latest_year - seen_years.max())

    age = 25.0
    if driver_id in dob and pd.notna(dob[driver_id]):
        age = float(ty - pd.Timestamp(dob[driver_id]).year)

    incumbent = 1.0 if frac_prev_rounds >= INCUMBENT_FRAC else 0.0
    rookie = 1.0 if (debut_recency <= 1.0 and career_starts <= n_rounds_prev) else 0.0
    returnee = 1.0 if prior_gap >= 2.0 else 0.0

    return dict(
        days_since_last=days_since_last, starts_90=starts_90, starts_180=starts_180,
        starts_365=starts_365, starts_730=starts_730, prev_first_round=prev_first_round,
        prev_last_round=prev_last_round, frac_prev_rounds=frac_prev_rounds,
        num_constructors_365=num_constructors_365, career_starts=career_starts, age=age,
        debut_recency=debut_recency, prior_gap=prior_gap, incumbent=incumbent,
        rookie=rookie, returnee=returnee,
    )


def _cohort_features(driver_feats):
    dids = list(driver_feats.keys())
    recent = np.array([driver_feats[d]["starts_365"] for d in dids], dtype=float)
    recency = np.array([driver_feats[d]["days_since_last"] for d in dids], dtype=float)
    rank_recent = recent.argsort().argsort() / max(1, len(dids) - 1)
    rank_recency = (-recency).argsort().argsort() / max(1, len(dids) - 1)
    n_inc = int(sum(driver_feats[d]["incumbent"] for d in dids))
    n_rook = int(sum(driver_feats[d]["rookie"] for d in dids))
    cohort = {}
    for i, d in enumerate(dids):
        cohort[d] = dict(
            cohort_rank_recent=float(rank_recent[i]),
            cohort_rank_recency=float(rank_recency[i]),
            cohort_n_incumbent=float(n_inc),
            cohort_n_rookie=float(n_rook),
        )
    return cohort


def _row_vector(df, cf, r, n_rounds):
    round_norm = r / n_rounds if n_rounds else 0.0
    row = [df[k] for k in ATTEND_DRIVER_FEATURES]
    row += [cf[k] for k in ATTEND_COHORT_FEATURES]
    row += [float(r), float(round_norm),
            df["incumbent"] * round_norm, df["rookie"] * round_norm]
    return row


# Assemble attendance feature vectors + optional labels for a cohort
def cohort_matrix(driver_events, driver_year_rounds, dob, cohort_ids, seed_time, ty,
                  latest_year, n_rounds_target, n_rounds_prev, with_labels):
    driver_feats = {
        d: _driver_features(driver_events, dob, d, seed_time, ty, latest_year, n_rounds_prev)
        for d in cohort_ids
    }
    cohort_feats = _cohort_features(driver_feats)
    rows, labels, index = [], [], []
    for d in cohort_ids:
        df = driver_feats[d]
        cf = cohort_feats[d]
        for r in range(1, n_rounds_target + 1):
            rows.append(_row_vector(df, cf, r, n_rounds_target))
            index.append((d, r))
            if with_labels:
                attended = driver_year_rounds.get((d, ty), set())
                labels.append(1 if r in attended else 0)
    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int) if with_labels else None
    return X, y, index


def build_attendance_training(driver_events, driver_year_rounds, year_cohort, dob,
                              season_n_rounds, target_years):
    Xs, ys = [], []
    for ty in target_years:
        latest_year = ty - 1
        cohort = sorted(year_cohort.get(ty, set()))
        if not cohort or ty not in season_n_rounds:
            continue
        seed_time = pd.Timestamp(f"{ty}-01-01")
        n_rounds_target = season_n_rounds[ty]
        n_rounds_prev = season_n_rounds.get(latest_year, n_rounds_target)
        X, y, _ = cohort_matrix(
            driver_events, driver_year_rounds, dob, cohort, seed_time, ty,
            latest_year, n_rounds_target, n_rounds_prev, with_labels=True,
        )
        Xs.append(X)
        ys.append(y)
    if not Xs:
        return np.empty((0, len(ATTEND_FEATURE_NAMES))), np.empty((0,), dtype=int)
    return np.vstack(Xs), np.concatenate(ys)


def fit_attendance_model(X, y, debug):
    if debug or len(np.unique(y)) < 2:
        model = LogisticRegression(C=0.5, max_iter=500)
        model.fit(X, y)
        return ("logreg", model)
    model = lgb.LGBMClassifier(**LGB_ATTEND)
    model.fit(X, y)
    return ("lgb", model)


def predict_attendance(model, X):
    kind, m = model
    return m.predict_proba(X)[:, 1]
