from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, Ridge

NUM_DST = 77
MAX_LOOKBACK = 8

LGB_PRESENCE = dict(
    objective="binary",
    num_leaves=7,
    max_depth=3,
    learning_rate=0.05,
    min_child_samples=20,
    reg_lambda=5.0,
    verbose=-1,
    random_state=13,
    n_jobs=1,
    n_estimators=150,
)
LGB_ROUND = dict(
    objective="regression",
    num_leaves=7,
    max_depth=3,
    learning_rate=0.05,
    min_child_samples=15,
    reg_lambda=5.0,
    verbose=-1,
    random_state=13,
    n_jobs=1,
    n_estimators=120,
)


# Season representations
def build_season_representations(races: pd.DataFrame):
    races = races[["raceId", "year", "round", "circuitId", "date"]].dropna(
        subset=["circuitId", "year", "date"]
    )
    season_circuits = {}
    season_round_pos = {}
    season_bounds = {}
    season_n_rounds = {}
    for year, grp in races.groupby("year"):
        grp = grp.sort_values(["round", "date"])
        circuits = list(dict.fromkeys(grp["circuitId"].astype(int).tolist()))
        season_circuits[int(year)] = set(circuits)
        n_rounds = int(grp["round"].max())
        season_n_rounds[int(year)] = n_rounds
        pos = {}
        for _, row in grp.iterrows():
            c = int(row["circuitId"])
            frac = float(row["round"]) / float(n_rounds) if n_rounds else 0.0
            pos.setdefault(c, []).append(frac)
        season_round_pos[int(year)] = {c: float(np.mean(v)) for c, v in pos.items()}
        season_bounds[int(year)] = (grp["date"].min(), grp["date"].max())
    return season_circuits, season_round_pos, season_bounds, season_n_rounds


# Persistence prior term of the calendar blend
def persistence_prior(season_circuits, latest_year, circuit, decay):
    present_last = 1.0 if circuit in season_circuits.get(latest_year, set()) else 0.0
    present_two = 1.0 if circuit in season_circuits.get(latest_year - 1, set()) else 0.0
    recent = 0.5 * present_last + 0.3 * present_two
    past_years = [y for y in season_circuits if y <= latest_year]
    long_run = 0.0
    norm = 0.0
    for y in past_years:
        w = decay ** (latest_year - y)
        norm += w
        if circuit in season_circuits[y]:
            long_run += w
    long_run = long_run / norm if norm > 0 else 0.0
    return recent, long_run


# Circuit-year presence features
def circuit_features(season_circuits, season_round_pos, latest_year, circuit, decay):
    past_years = sorted(y for y in season_circuits if y <= latest_year)
    feats = []
    for lag in range(1, MAX_LOOKBACK + 1):
        feats.append(1.0 if circuit in season_circuits.get(latest_year - lag + 1, set()) else 0.0)

    streak = 0
    y = latest_year
    while y in season_circuits and circuit in season_circuits[y]:
        streak += 1
        y -= 1

    last_seen = None
    decayed = 0.0
    appearances = 0
    round_fracs = []
    for y in past_years:
        if circuit in season_circuits[y]:
            decayed += decay ** (latest_year - y)
            last_seen = y
            appearances += 1
            if circuit in season_round_pos.get(y, {}):
                round_fracs.append((y, season_round_pos[y][circuit]))
    years_since = float(latest_year - last_seen) if last_seen is not None else 30.0

    recent_fracs = [f for (yy, f) in round_fracs if yy >= latest_year - 4]
    last_round_frac = round_fracs[-1][1] if round_fracs else 0.5
    median_round_frac = float(np.median([f for (_, f) in round_fracs])) if round_fracs else 0.5
    round_frac_std = float(np.std(recent_fracs)) if len(recent_fracs) > 1 else 0.25

    returned = 0.0
    seen_years = [yy for yy in past_years if circuit in season_circuits[yy]]
    if len(seen_years) >= 2:
        gaps = np.diff(seen_years)
        if (gaps > 1).any() and circuit in season_circuits.get(latest_year, set()):
            returned = 1.0

    feats.extend([
        float(streak),
        years_since,
        decayed,
        float(appearances),
        last_round_frac,
        median_round_frac,
        round_frac_std,
        returned,
    ])
    return feats


PRESENCE_FEATURE_NAMES = (
    [f"present_lag{lag}" for lag in range(1, MAX_LOOKBACK + 1)]
    + ["streak", "years_since", "decayed", "appearances",
       "last_round_frac", "median_round_frac", "round_frac_std", "returned"]
)


# Build training matrix for the presence and round models over rolling origins
def build_presence_training(season_circuits, season_round_pos, target_years, decay):
    X, y_present, X_round, y_round = [], [], [], []
    for ty in target_years:
        latest_year = ty - 1
        if ty not in season_circuits:
            continue
        truth = season_circuits[ty]
        for c in range(NUM_DST):
            feats = circuit_features(season_circuits, season_round_pos, latest_year, c, decay)
            X.append(feats)
            label = 1 if c in truth else 0
            y_present.append(label)
            if label == 1 and c in season_round_pos.get(ty, {}):
                X_round.append(feats)
                y_round.append(season_round_pos[ty][c])
    return (np.array(X, dtype=float), np.array(y_present, dtype=int),
            np.array(X_round, dtype=float), np.array(y_round, dtype=float))


def fit_presence_model(X, y, debug):
    if debug or len(np.unique(y)) < 2:
        model = LogisticRegression(C=1.0, max_iter=500)
        model.fit(X, y)
        return ("logreg", model)
    model = lgb.LGBMClassifier(**LGB_PRESENCE)
    model.fit(X, y)
    return ("lgb", model)


def fit_round_model(X, y, debug):
    if len(X) < 20:
        return None
    if debug:
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        return ("ridge", model)
    model = lgb.LGBMRegressor(**LGB_ROUND)
    model.fit(X, y)
    return ("lgb", model)


def predict_presence(model, X):
    kind, m = model
    if kind == "logreg":
        return m.predict_proba(X)[:, 1]
    return m.predict_proba(X)[:, 1]


def predict_round(model, X):
    if model is None:
        return None
    kind, m = model
    return m.predict(X)


# Produce calendar scores and round ordering for one target year
def calendar_forecast(season_circuits, season_round_pos, latest_year,
                      presence_model, round_model, decay, blend):
    feats = np.array(
        [circuit_features(season_circuits, season_round_pos, latest_year, c, decay)
         for c in range(NUM_DST)],
        dtype=float,
    )
    learned = predict_presence(presence_model, feats)
    p_cal = np.zeros(NUM_DST)
    for c in range(NUM_DST):
        recent, long_run = persistence_prior(season_circuits, latest_year, c, decay)
        p_cal[c] = blend[0] * learned[c] + blend[1] * recent + blend[2] * long_run

    round_pred = predict_round(round_model, feats)
    if round_pred is None:
        round_pred = feats[:, PRESENCE_FEATURE_NAMES.index("median_round_frac")]
    round_pred = np.clip(round_pred, 0.0, 1.0)
    return p_cal, round_pred
