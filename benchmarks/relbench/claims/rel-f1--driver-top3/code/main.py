import bisect
import hashlib
import json
import math
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Configuration

FEATURE_VERSION = "winter_constructor_v1"
MODEL_VERSION = "prequential_prefix_v1"
ELO_INITIAL = 1500.0
SEASON_REGRESSION = 0.90
PREQUENTIAL_YEARS = tuple(range(2001, 2005))
CONSTRUCTOR_RETENTION = 0.70
ERA_REGRESSION = 0.30
TEAMMATE_KAPPA = 8.0
RECENCY_HALF_LIFE_YEARS = 8.0
MAX_OPENER_MASS = 0.20
BOOTSTRAP_DRAWS = 2000
RANDOM_SEED = 1337
RELATIVE_FEATURES = {
    "driver_elo": 1.0,
    "package_elo": 1.0,
    "teammate_elo": 1.0,
    "constructor_elo": 1.0,
    "q_ewma": -1.0,
    "grid_ewma": -1.0,
    "last_q_position": -1.0,
    "driver_standing_position": -1.0,
    "constructor_standing_position": -1.0,
    "two_car_form": -1.0,
    "q_top3_share": 1.0,
    "constructor_points_share": 1.0,
}
PREFIX_FEATURES = {
    1: (
        "global_days_since_race",
        "month",
        "prior_year_expected_races_30d",
        "opener_flag",
        "prior_driver_standing_position",
        "prior_driver_standing_points",
        "prior_driver_points_share",
        "prior_driver_wins",
        "prior_constructor_standing_position",
        "prior_constructor_standing_points",
        "prior_constructor_points_share",
        "prior_constructor_wins",
        "prior_q5_median",
        "prior_q5_top3_share",
    ),
    2: (
        "constructor_season_strength",
        "constructor_offseason_age",
        "constructor_lineup_continuity",
        "constructor_season_uncertainty",
        "transfer_tenure_days",
        "constructor_change_observed",
        "transfer_observed",
        "expected_package_shock",
        "transfer_uncertainty",
    ),
    3: (
        "teammate_residual_mean",
        "teammate_residual_effective_n",
        "teammate_residual_uncertainty",
        "teammate_residual_contrast",
        "constructor_reshuffle_agreement",
        "constructor_reshuffle_volatility",
        "neighbor_constructor_elo",
        "neighbor_constructor_form",
        "neighbor_constructor_result_ewma",
        "neighbor_active_driver_elo_mean",
        "neighbor_active_driver_q_mean",
        "neighbor_active_driver_top3_mean",
        "neighbor_active_count",
    ),
}
NEW_RELATIVE_FEATURES = {
    "prior_driver_standing_position": -1.0,
    "prior_driver_points_share": 1.0,
    "prior_q5_median": -1.0,
    "prior_q5_top3_share": 1.0,
    "prior_constructor_standing_position": -1.0,
    "prior_constructor_points_share": 1.0,
    "constructor_season_strength": 1.0,
    "constructor_lineup_continuity": 1.0,
    "expected_package_shock": 1.0,
    "teammate_residual_mean": 1.0,
    "teammate_residual_contrast": 1.0,
    "constructor_reshuffle_agreement": 1.0,
    "neighbor_constructor_elo": 1.0,
    "neighbor_constructor_form": -1.0,
    "neighbor_constructor_result_ewma": 1.0,
    "neighbor_active_driver_elo_mean": 1.0,
    "neighbor_active_driver_q_mean": -1.0,
    "neighbor_active_driver_top3_mean": 1.0,
}
MODEL_EXCLUDED_FEATURES = {
    "recent_circuit_cluster",
    "driver_nationality_top3_share",
    "constructor_nationality_top3_share",
}


# Utilities

def elapsed(start: float, phase: str) -> None:
    print(f"[dual-elo] {phase}: {time.time() - start:.2f}s", flush=True)


def scalar(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def clipped_days(current: pd.Timestamp, previous, default: float = 730.0) -> float:
    if previous is None or pd.isna(previous):
        return default
    return float(np.clip((current - previous).total_seconds() / 86400.0, 0.0, 730.0))


def ewma(previous, value: float, alpha: float) -> float:
    return value if previous is None else alpha * value + (1.0 - alpha) * previous


def smoothed_rate(successes: float, games: float, prior: float = 0.15, strength: float = 6.0) -> float:
    return (successes + prior * strength) / (games + strength)


def logistic_pipeline(c_value: float):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=c_value, penalty="l2", max_iter=600, solver="lbfgs"),
    )


def predict_binary(model, features: pd.DataFrame) -> np.ndarray:
    if isinstance(model, float):
        return np.full(len(features), model, dtype=np.float64)
    return model.predict_proba(features)[:, 1].astype(np.float64)


def fit_binary(features: pd.DataFrame, labels: np.ndarray, weights: np.ndarray | None = None):
    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return float(np.mean(labels))
    model = logistic_pipeline(0.05)
    if weights is None:
        model.fit(features, labels)
    else:
        model.fit(features, labels, logisticregression__sample_weight=np.asarray(weights, dtype=np.float64))
    return model


def origin_standardize(values: np.ndarray, dates: pd.Series) -> np.ndarray:
    frame = pd.DataFrame({"value": np.asarray(values, dtype=np.float64), "date": pd.to_datetime(dates).to_numpy()})
    means = frame.groupby("date")["value"].transform("mean").to_numpy()
    deviations = frame.groupby("date")["value"].transform("std").fillna(0.0).to_numpy()
    return (frame["value"].to_numpy() - means) / np.where(deviations > 1e-8, deviations, 1.0)


def safe_logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


# State

class RaceState:
    def __init__(self, drivers: pd.DataFrame, constructors: pd.DataFrame, circuit_by_race: dict, cluster_by_race: dict):
        self.driver_info = drivers.set_index("driverId").to_dict("index")
        self.constructor_info = constructors.set_index("constructorId").to_dict("index")
        self.circuit_by_race = circuit_by_race
        self.cluster_by_race = cluster_by_race
        self.year = None
        self.driver_elo = defaultdict(lambda: ELO_INITIAL)
        self.package_elo = defaultdict(lambda: ELO_INITIAL)
        self.teammate_elo = defaultdict(lambda: ELO_INITIAL)
        self.constructor_elo = defaultdict(lambda: ELO_INITIAL)
        self.driver_games = defaultdict(int)
        self.package_games = defaultdict(int)
        self.teammate_games = defaultdict(int)
        self.constructor_games = defaultdict(int)
        self.q_ewma = {}
        self.grid_ewma = {}
        self.finish_ewma = {}
        self.q_games = defaultdict(int)
        self.q_top3 = defaultdict(int)
        self.q_top6 = defaultdict(int)
        self.q_top10 = defaultdict(int)
        self.session_games = defaultdict(int)
        self.session_top3 = defaultdict(int)
        self.result_games = defaultdict(int)
        self.last_q_position = {}
        self.last_q_date = {}
        self.last_result_date = {}
        self.last_event_date = {}
        self.last_constructor = {}
        self.last_cluster = {}
        self.driver_cluster_games = defaultdict(int)
        self.driver_cluster_top3 = defaultdict(int)
        self.constructor_form = {}
        self.constructor_session_games = defaultdict(int)
        self.constructor_session_top3 = defaultdict(int)
        self.constructor_result_ewma = {}
        self.driver_standing = {}
        self.constructor_standing = {}
        self.nationality_games = defaultdict(int)
        self.nationality_top3 = defaultdict(int)
        self.constructor_nationality_games = defaultdict(int)
        self.constructor_nationality_top3 = defaultdict(int)
        self.calendar = defaultdict(list)
        self.last_global_race_date = None
        self.constructor_last_event_date = {}
        self.constructor_since_date = {}
        self.previous_constructor = {}
        self.transfer_observed_year = {}
        self.season_lineups = defaultdict(lambda: defaultdict(set))
        self.season_constructor_games = defaultdict(lambda: defaultdict(int))
        self.season_driver_q = defaultdict(lambda: defaultdict(list))
        self.prior_driver_state = {}
        self.prior_constructor_state = {}
        self.prior_driver_q = {}
        self.prior_lineups = defaultdict(set)
        self.prior_constructor_strength = {}
        self.prior_driver_constructor = {}
        self.prior_constructor_last_date = {}
        self.prior_era_strength = ELO_INITIAL
        self.prior_destination_strength = ELO_INITIAL
        self.teammate_residual_sum = defaultdict(float)
        self.teammate_residual_count = defaultdict(int)
        self.constructor_era_residual_sum = defaultdict(float)
        self.constructor_era_residual_count = defaultdict(int)
        self.first_two_constructor_results = defaultdict(list)
        self.constructor_reshuffle_history = []

    def finalize_season(self, year: int) -> None:
        driver_rows = {
            driver: value for driver, value in self.driver_standing.items() if int(value[3]) == year
        }
        constructor_rows = {
            constructor: value for constructor, value in self.constructor_standing.items() if int(value[3]) == year
        }
        driver_total = sum(max(scalar(value[0]), 0.0) for value in driver_rows.values())
        constructor_total = sum(max(scalar(value[0]), 0.0) for value in constructor_rows.values())
        self.prior_driver_state = {
            driver: (
                scalar(value[0]),
                scalar(value[1], 30.0),
                scalar(value[2]),
                scalar(value[0]) / max(driver_total, 1.0),
            )
            for driver, value in driver_rows.items()
        }
        self.prior_constructor_state = {
            constructor: (
                scalar(value[0]),
                scalar(value[1], 20.0),
                scalar(value[2]),
                scalar(value[0]) / max(constructor_total, 1.0),
            )
            for constructor, value in constructor_rows.items()
        }
        self.prior_driver_q = {}
        for driver, positions in self.season_driver_q.get(year, {}).items():
            recent = np.asarray(positions[-5:], dtype=np.float64)
            self.prior_driver_q[int(driver)] = (
                float(np.median(recent)),
                float(np.mean(recent <= 3.0)),
            )
        self.prior_lineups = defaultdict(set)
        for constructor, lineup in self.season_lineups.get(year, {}).items():
            self.prior_lineups[int(constructor)] = set(int(driver) for driver in lineup)
        active_constructors = set(self.prior_lineups) | set(constructor_rows)
        raw_strengths = {
            constructor: float(self.constructor_elo[constructor]) for constructor in active_constructors
        }
        era_mean = float(np.mean(list(raw_strengths.values()))) if raw_strengths else ELO_INITIAL
        self.prior_era_strength = era_mean
        self.prior_constructor_strength = {
            constructor: CONSTRUCTOR_RETENTION * strength + ERA_REGRESSION * era_mean
            for constructor, strength in raw_strengths.items()
        }
        historical_strengths = list(self.prior_constructor_strength.values())
        self.prior_destination_strength = (
            float(np.mean(historical_strengths)) if historical_strengths else era_mean
        )
        self.prior_driver_constructor = dict(self.last_constructor)
        self.prior_constructor_last_date = dict(self.constructor_last_event_date)

    @staticmethod
    def spearman(values_a: list[float], values_b: list[float]) -> float:
        if len(values_a) < 3:
            return 0.0
        ranks_a = pd.Series(values_a).rank(method="average").to_numpy(dtype=np.float64)
        ranks_b = pd.Series(values_b).rank(method="average").to_numpy(dtype=np.float64)
        if np.std(ranks_a) < 1e-12 or np.std(ranks_b) < 1e-12:
            return 0.0
        return float(np.corrcoef(ranks_a, ranks_b)[0, 1])

    def regress_to_year(self, year: int) -> None:
        if self.year is None:
            self.year = year
            return
        while self.year < year:
            self.finalize_season(self.year)
            for ratings in (self.driver_elo, self.package_elo, self.teammate_elo, self.constructor_elo):
                for key in list(ratings.keys()):
                    ratings[key] = ELO_INITIAL + SEASON_REGRESSION * (ratings[key] - ELO_INITIAL)
            self.year += 1

    def expected_sessions(self, timestamp: pd.Timestamp) -> float:
        counts = []
        for year in range(max(1950, timestamp.year - 6), timestamp.year):
            dates = self.calendar.get(year, ())
            if not dates:
                continue
            day = min(timestamp.day, pd.Timestamp(year=year, month=timestamp.month, day=1).days_in_month)
            start = pd.Timestamp(year=year, month=timestamp.month, day=day)
            end = start + pd.Timedelta(days=30)
            left = bisect.bisect_right(dates, start)
            right = bisect.bisect_right(dates, end)
            counts.append(right - left)
        return float(np.mean(counts)) if counts else 1.0

    def context(self, timestamp: pd.Timestamp) -> dict:
        current_constructor_points = [
            value[0] for value in self.constructor_standing.values() if value[3] == timestamp.year
        ]
        previous_lengths = [len(self.calendar[year]) for year in range(timestamp.year - 6, timestamp.year) if self.calendar.get(year)]
        season_total = float(np.mean(previous_lengths)) if previous_lengths else 16.0
        season_sessions = len(self.calendar.get(timestamp.year, ()))
        day_of_year = timestamp.dayofyear
        expected = self.expected_sessions(timestamp)
        completed_agreements = [
            value for season, value in self.constructor_reshuffle_history if season <= timestamp.year
        ][-5:]
        reshuffle_agreement = float(np.mean(completed_agreements)) if completed_agreements else 0.5
        return {
            "expected_sessions": expected,
            "season_sessions": float(season_sessions),
            "season_phase": float(season_sessions / max(season_total, 1.0)),
            "month_sin": math.sin(2.0 * math.pi * day_of_year / 365.25),
            "month_cos": math.cos(2.0 * math.pi * day_of_year / 365.25),
            "constructor_points_total": float(sum(current_constructor_points)),
            "global_days_since_race": clipped_days(timestamp, self.last_global_race_date),
            "month": float(timestamp.month),
            "opener_flag": float(season_sessions == 0 and expected > 0.0),
            "constructor_reshuffle_agreement": reshuffle_agreement,
            "constructor_reshuffle_volatility": 1.0 - reshuffle_agreement,
        }

    def observe_constructor(self, driver: int, constructor: int, timestamp: pd.Timestamp) -> None:
        previous = self.last_constructor.get(driver)
        if previous is None:
            self.constructor_since_date[driver] = timestamp
        elif previous != constructor:
            self.previous_constructor[driver] = previous
            self.constructor_since_date[driver] = timestamp
            self.transfer_observed_year[driver] = timestamp.year
        self.last_constructor[driver] = constructor
        self.constructor_last_event_date[constructor] = timestamp
        self.season_lineups[timestamp.year][constructor].add(driver)

    def teammate_residual(self, driver: int, constructor: int) -> tuple[float, float, float]:
        count = float(self.teammate_residual_count[driver])
        raw_mean = self.teammate_residual_sum[driver] / max(count, 1.0)
        era_key = (constructor, (self.year or 1950) // 5)
        era_count = float(self.constructor_era_residual_count[era_key])
        era_mean = self.constructor_era_residual_sum[era_key] / max(era_count, 1.0)
        reliability = count / (count + TEAMMATE_KAPPA)
        posterior = reliability * raw_mean + (1.0 - reliability) * era_mean
        return float(posterior), float(reliability), float(1.0 / math.sqrt(count + TEAMMATE_KAPPA))

    def teammate_context(self, driver: int, constructor: int, timestamp: pd.Timestamp) -> tuple[float, float, float, float, float]:
        own, effective_n, uncertainty = self.teammate_residual(driver, constructor)
        current = set(self.season_lineups.get(timestamp.year, {}).get(constructor, set()))
        lineup = current if current else set(self.prior_lineups.get(constructor, set()))
        peers = [peer for peer in lineup if peer != driver]
        peer_values = [self.teammate_residual(peer, constructor)[0] for peer in peers]
        contrast = own - float(np.mean(peer_values)) if peer_values else 0.0
        return own, effective_n, uncertainty, contrast, float(bool(peers))

    def constructor_season_context(self, driver: int, constructor: int, timestamp: pd.Timestamp) -> dict:
        prior_constructor = int(self.prior_driver_constructor.get(driver, constructor))
        prior_strength = self.prior_constructor_strength.get(prior_constructor, self.prior_era_strength)
        current_observed = (
            self.last_event_date.get(driver) is not None
            and pd.Timestamp(self.last_event_date[driver]).year == timestamp.year
        )
        observed_constructor = int(self.last_constructor.get(driver, prior_constructor))
        if current_observed:
            effective_constructor = observed_constructor
            strength = self.prior_constructor_strength.get(effective_constructor, self.prior_era_strength)
            shock = strength - prior_strength
            transfer_uncertainty = 0.0
        else:
            effective_constructor = prior_constructor
            retained = self.prior_constructor_strength.get(prior_constructor, self.prior_era_strength)
            strength = CONSTRUCTOR_RETENTION * retained + (1.0 - CONSTRUCTOR_RETENTION) * self.prior_destination_strength
            shock = strength - retained
            transfer_uncertainty = (1.0 - CONSTRUCTOR_RETENTION) * abs(
                self.prior_destination_strength - retained
            )
        prior_lineup = set(self.prior_lineups.get(effective_constructor, set()))
        current_lineup = set(self.season_lineups.get(timestamp.year, {}).get(effective_constructor, set()))
        continuity = (
            len(prior_lineup & current_lineup) / max(len(prior_lineup), 1)
            if current_lineup
            else CONSTRUCTOR_RETENTION
        )
        prior_games = float(self.season_constructor_games.get(timestamp.year - 1, {}).get(effective_constructor, 0))
        last_constructor_date = self.prior_constructor_last_date.get(effective_constructor)
        since = self.constructor_since_date.get(driver)
        return {
            "effective_constructor": effective_constructor,
            "constructor_season_strength": float(strength),
            "constructor_offseason_age": clipped_days(timestamp, last_constructor_date),
            "constructor_lineup_continuity": float(continuity),
            "constructor_season_uncertainty": float(1.0 / math.sqrt(prior_games + 1.0)),
            "transfer_tenure_days": clipped_days(timestamp, since),
            "constructor_change_observed": float(current_observed and observed_constructor != prior_constructor),
            "transfer_observed": float(current_observed),
            "expected_package_shock": float(shock),
            "transfer_uncertainty": float(transfer_uncertainty),
        }

    def neighbour_context(self, driver: int, constructor: int, timestamp: pd.Timestamp) -> dict:
        candidates = [
            peer
            for peer, peer_constructor in self.last_constructor.items()
            if peer != driver
            and int(peer_constructor) == constructor
            and clipped_days(timestamp, self.last_event_date.get(peer)) <= 365.0
        ]
        if not candidates:
            candidates = [peer for peer in self.prior_lineups.get(constructor, set()) if peer != driver]
        driver_elos = [self.driver_elo[peer] for peer in candidates]
        q_values = [self.q_ewma.get(peer, 15.0) for peer in candidates]
        top3_values = [
            smoothed_rate(self.q_top3[peer], self.q_games[peer]) for peer in candidates
        ]
        return {
            "neighbor_constructor_elo": float(self.constructor_elo[constructor]),
            "neighbor_constructor_form": float(self.constructor_form.get(constructor, 15.0)),
            "neighbor_constructor_result_ewma": float(self.constructor_result_ewma.get(constructor, 0.0)),
            "neighbor_active_driver_elo_mean": float(np.mean(driver_elos)) if driver_elos else ELO_INITIAL,
            "neighbor_active_driver_q_mean": float(np.mean(q_values)) if q_values else 15.0,
            "neighbor_active_driver_top3_mean": float(np.mean(top3_values)) if top3_values else 0.15,
            "neighbor_active_count": float(len(candidates)),
        }

    def features(self, driver: int, constructor: int, timestamp: pd.Timestamp, context: dict) -> dict:
        package = (driver, constructor)
        info = self.driver_info.get(driver, {})
        constructor_info = self.constructor_info.get(constructor, {})
        dob = info.get("dob")
        age = 30.0 if dob is None or pd.isna(dob) else float(np.clip((timestamp - pd.Timestamp(dob)).days / 365.25, 16.0, 75.0))
        nationality = str(info.get("nationality", "unknown"))
        constructor_nationality = str(constructor_info.get("nationality", "unknown"))
        q_games = self.q_games[driver]
        session_games = self.session_games[driver]
        result_games = self.result_games[driver]
        constructor_games = self.constructor_session_games[constructor]
        recent_cluster = self.last_cluster.get(driver, -1)
        cluster_key = (driver, recent_cluster)
        standing = self.driver_standing.get(driver, (0.0, 30.0, 0.0, -1))
        constructor_standing = self.constructor_standing.get(constructor, (0.0, 20.0, 0.0, -1))
        current_driver_points = standing[0] if standing[3] == timestamp.year else 0.0
        current_driver_wins = standing[2] if standing[3] == timestamp.year else 0.0
        current_constructor_points = constructor_standing[0] if constructor_standing[3] == timestamp.year else 0.0
        current_constructor_wins = constructor_standing[2] if constructor_standing[3] == timestamp.year else 0.0
        q_position = self.q_ewma.get(driver, 15.0)
        grid_position = self.grid_ewma.get(driver, 15.0)
        last_q_position = self.last_q_position.get(driver, 20.0)
        days_q = clipped_days(timestamp, self.last_q_date.get(driver))
        days_result = clipped_days(timestamp, self.last_result_date.get(driver))
        days_event = clipped_days(timestamp, self.last_event_date.get(driver))
        q_recency = math.exp(-days_q / 60.0)
        driver_rating = self.driver_elo[driver]
        teammate_rating = self.teammate_elo[driver]
        package_rating = self.package_elo[package]
        constructor_rating = self.constructor_elo[constructor]
        constructor_total = context["constructor_points_total"]
        prior_driver = self.prior_driver_state.get(driver)
        prior_constructor = self.prior_constructor_state.get(constructor)
        prior_q = self.prior_driver_q.get(driver)
        season_context = self.constructor_season_context(driver, constructor, timestamp)
        effective_constructor = int(season_context.pop("effective_constructor"))
        teammate_mean, teammate_n, teammate_uncertainty, teammate_contrast, teammate_known = self.teammate_context(
            driver, effective_constructor, timestamp
        )
        neighbour = self.neighbour_context(driver, effective_constructor, timestamp)
        values = {
            "driver_elo": driver_rating,
            "package_elo": package_rating,
            "teammate_elo": teammate_rating,
            "constructor_elo": constructor_rating,
            "driver_games": math.log1p(self.driver_games[driver]),
            "package_games": math.log1p(self.package_games[package]),
            "teammate_games": math.log1p(self.teammate_games[driver]),
            "constructor_games": math.log1p(self.constructor_games[constructor]),
            "driver_uncertainty": 1.0 / math.sqrt(self.driver_games[driver] + 1.0),
            "package_uncertainty": 1.0 / math.sqrt(self.package_games[package] + 1.0),
            "teammate_uncertainty": 1.0 / math.sqrt(self.teammate_games[driver] + 1.0),
            "constructor_uncertainty": 1.0 / math.sqrt(self.constructor_games[constructor] + 1.0),
            "driver_minus_teammate": driver_rating - teammate_rating,
            "implied_car_contribution": package_rating - teammate_rating,
            "constructor_driver_gap": driver_rating - constructor_rating,
            "q_ewma": q_position,
            "grid_ewma": grid_position,
            "finish_ewma": self.finish_ewma.get(driver, 15.0),
            "q_top3_share": smoothed_rate(self.q_top3[driver], q_games),
            "q_top6_share": smoothed_rate(self.q_top6[driver], q_games, 0.30),
            "q_top10_share": smoothed_rate(self.q_top10[driver], q_games, 0.50),
            "session_top3_share": smoothed_rate(self.session_top3[driver], session_games),
            "q_experience": math.log1p(q_games),
            "race_experience": math.log1p(result_games),
            "last_q_position": last_q_position,
            "last_q_top3": float(last_q_position <= 3.0),
            "last_q_top6": float(last_q_position <= 6.0),
            "days_since_q": days_q,
            "days_since_result": days_result,
            "days_since_event": days_event,
            "q_recency": q_recency,
            "recent_q_strength": q_recency / max(last_q_position, 1.0),
            "driver_standing_points": current_driver_points,
            "driver_standing_position": scalar(standing[1], 30.0),
            "driver_standing_wins": current_driver_wins,
            "constructor_standing_points": current_constructor_points,
            "constructor_standing_position": scalar(constructor_standing[1], 20.0),
            "constructor_standing_wins": current_constructor_wins,
            "constructor_points_share": current_constructor_points / max(constructor_total, 1.0),
            "constructor_result_ewma": self.constructor_result_ewma.get(constructor, 0.0),
            "two_car_form": self.constructor_form.get(constructor, 15.0),
            "constructor_top3_share": smoothed_rate(self.constructor_session_top3[constructor], max(2 * constructor_games, 0)),
            "age": age,
            "age_squared": age * age / 100.0,
            "expected_sessions": context["expected_sessions"],
            "season_sessions": context["season_sessions"],
            "season_phase": context["season_phase"],
            "month_sin": context["month_sin"],
            "month_cos": context["month_cos"],
            "recent_circuit_cluster": float(recent_cluster),
            "recent_cluster_top3_share": smoothed_rate(self.driver_cluster_top3[cluster_key], self.driver_cluster_games[cluster_key]),
            "driver_nationality_top3_share": smoothed_rate(self.nationality_top3[nationality], self.nationality_games[nationality]),
            "constructor_nationality_top3_share": smoothed_rate(
                self.constructor_nationality_top3[constructor_nationality],
                self.constructor_nationality_games[constructor_nationality],
            ),
            "constructor_known": float(constructor >= 0),
            "constructor_continuity": float(self.last_constructor.get(driver, constructor) == constructor),
            "history_none": float(session_games == 0),
            "history_sparse": float(0 < session_games <= 5),
            "history_rich": float(session_games > 5),
            "active_92d": float(days_event < 92.0),
            "active_365d": float(days_event < 365.0),
            "global_days_since_race": context["global_days_since_race"],
            "month": context["month"],
            "prior_year_expected_races_30d": context["expected_sessions"],
            "opener_flag": context["opener_flag"],
            "prior_driver_standing_position": scalar(prior_driver[1], 30.0) if prior_driver else 30.0,
            "prior_driver_standing_points": scalar(prior_driver[0]) if prior_driver else 0.0,
            "prior_driver_points_share": scalar(prior_driver[3]) if prior_driver else 0.0,
            "prior_driver_wins": scalar(prior_driver[2]) if prior_driver else 0.0,
            "prior_constructor_standing_position": scalar(prior_constructor[1], 20.0) if prior_constructor else 20.0,
            "prior_constructor_standing_points": scalar(prior_constructor[0]) if prior_constructor else 0.0,
            "prior_constructor_points_share": scalar(prior_constructor[3]) if prior_constructor else 0.0,
            "prior_constructor_wins": scalar(prior_constructor[2]) if prior_constructor else 0.0,
            "prior_q5_median": scalar(prior_q[0], 15.0) if prior_q else 15.0,
            "prior_q5_top3_share": scalar(prior_q[1], 0.15) if prior_q else 0.15,
            "teammate_residual_mean": teammate_mean,
            "teammate_residual_effective_n": teammate_n,
            "teammate_residual_uncertainty": teammate_uncertainty,
            "teammate_residual_contrast": teammate_contrast,
            "constructor_reshuffle_agreement": context["constructor_reshuffle_agreement"],
            "constructor_reshuffle_volatility": context["constructor_reshuffle_volatility"],
        }
        values.update(season_context)
        values.update(neighbour)
        missing = {
            "global_days_since_race": self.last_global_race_date is None,
            "prior_driver_standing_position": prior_driver is None,
            "prior_driver_standing_points": prior_driver is None,
            "prior_driver_points_share": prior_driver is None,
            "prior_driver_wins": prior_driver is None,
            "prior_constructor_standing_position": prior_constructor is None,
            "prior_constructor_standing_points": prior_constructor is None,
            "prior_constructor_points_share": prior_constructor is None,
            "prior_constructor_wins": prior_constructor is None,
            "prior_q5_median": prior_q is None,
            "prior_q5_top3_share": prior_q is None,
            "constructor_season_strength": effective_constructor not in self.prior_constructor_strength,
            "constructor_offseason_age": effective_constructor not in self.prior_constructor_last_date,
            "constructor_lineup_continuity": not bool(self.prior_lineups.get(effective_constructor)),
            "constructor_season_uncertainty": effective_constructor not in self.season_constructor_games.get(timestamp.year - 1, {}),
            "transfer_tenure_days": driver not in self.constructor_since_date,
            "teammate_residual_mean": self.teammate_residual_count[driver] == 0,
            "teammate_residual_contrast": not bool(teammate_known),
            "neighbor_active_driver_elo_mean": neighbour["neighbor_active_count"] == 0,
            "neighbor_active_driver_q_mean": neighbour["neighbor_active_count"] == 0,
            "neighbor_active_driver_top3_mean": neighbour["neighbor_active_count"] == 0,
        }
        for name in sum((list(group) for group in PREFIX_FEATURES.values()), []):
            values[f"{name}_missing"] = float(missing.get(name, False))
        return values

    @staticmethod
    def pairwise_update(keys: list, positions: np.ndarray, ratings: defaultdict, k_value: float) -> None:
        if len(keys) < 2:
            return
        before = np.array([ratings[key] for key in keys], dtype=np.float64)
        delta = np.zeros(len(keys), dtype=np.float64)
        for index in range(len(keys)):
            opponents = np.arange(len(keys)) != index
            actual = np.where(positions[index] < positions[opponents], 1.0, np.where(positions[index] == positions[opponents], 0.5, 0.0))
            expected = 1.0 / (1.0 + np.power(10.0, (before[opponents] - before[index]) / 400.0))
            delta[index] = k_value * float(np.mean(actual - expected))
        for key, value, change in zip(keys, before, delta):
            ratings[key] = float(value + change)

    def update_session(self, group: pd.DataFrame, timestamp: pd.Timestamp) -> None:
        drivers = group["driverId"].astype(int).tolist()
        constructors = group["constructorId"].fillna(-1).astype(int).tolist()
        positions = group["target_position"].to_numpy(dtype=np.float64)
        packages = list(zip(drivers, constructors))
        field_k = 32.0 / math.sqrt(max(len(drivers), 1))
        self.pairwise_update(drivers, positions, self.driver_elo, field_k)
        self.pairwise_update(packages, positions, self.package_elo, field_k)
        for constructor in sorted(set(constructors)):
            indices = [index for index, value in enumerate(constructors) if value == constructor]
            if len(indices) > 1:
                self.pairwise_update(
                    [drivers[index] for index in indices], positions[indices], self.teammate_elo, 24.0
                )
                for index in indices:
                    self.teammate_games[drivers[index]] += 1
        constructor_positions = {}
        for constructor in sorted(set(constructors)):
            values = sorted(positions[np.array(constructors) == constructor])[:2]
            constructor_positions[constructor] = float(np.mean(values))
        constructor_keys = list(constructor_positions)
        self.pairwise_update(
            constructor_keys,
            np.array([constructor_positions[key] for key in constructor_keys]),
            self.constructor_elo,
            32.0 / math.sqrt(max(len(constructor_keys), 1)),
        )
        race_id = int(group["raceId"].iloc[0])
        cluster = self.cluster_by_race.get(race_id, -1)
        qualifying_rows = group[group["has_qualifying"]].copy()
        if len(qualifying_rows) > 1:
            field_size = len(qualifying_rows)
            qualifying_rows["field_percentile"] = 1.0 - (
                qualifying_rows["target_position"].astype(float) - 1.0
            ) / max(field_size - 1.0, 1.0)
            qualifying_rows["constructor_mean"] = qualifying_rows.groupby("constructorId")["field_percentile"].transform("mean")
            qualifying_rows["residual"] = qualifying_rows["field_percentile"] - qualifying_rows["constructor_mean"]
            for residual_row in qualifying_rows.itertuples():
                residual_driver = int(residual_row.driverId)
                residual_constructor = int(residual_row.constructorId)
                residual = float(residual_row.residual)
                era_key = (residual_constructor, timestamp.year // 5)
                self.teammate_residual_sum[residual_driver] += residual
                self.teammate_residual_count[residual_driver] += 1
                self.constructor_era_residual_sum[era_key] += residual
                self.constructor_era_residual_count[era_key] += 1
        for row in group.itertuples():
            driver = int(row.driverId)
            constructor = int(row.constructorId) if not pd.isna(row.constructorId) else -1
            position = float(row.target_position)
            package = (driver, constructor)
            self.driver_games[driver] += 1
            self.package_games[package] += 1
            self.session_games[driver] += 1
            self.session_top3[driver] += int(position <= 3.0)
            self.last_event_date[driver] = timestamp
            self.observe_constructor(driver, constructor, timestamp)
            self.last_cluster[driver] = cluster
            self.driver_cluster_games[(driver, cluster)] += 1
            self.driver_cluster_top3[(driver, cluster)] += int(position <= 3.0)
            nationality = str(self.driver_info.get(driver, {}).get("nationality", "unknown"))
            constructor_nationality = str(self.constructor_info.get(constructor, {}).get("nationality", "unknown"))
            self.nationality_games[nationality] += 1
            self.nationality_top3[nationality] += int(position <= 3.0)
            self.constructor_nationality_games[constructor_nationality] += 1
            self.constructor_nationality_top3[constructor_nationality] += int(position <= 3.0)
            if bool(row.has_qualifying):
                self.q_ewma[driver] = ewma(self.q_ewma.get(driver), position, 0.35)
                self.q_games[driver] += 1
                self.q_top3[driver] += int(position <= 3.0)
                self.q_top6[driver] += int(position <= 6.0)
                self.q_top10[driver] += int(position <= 10.0)
                self.last_q_position[driver] = position
                self.last_q_date[driver] = timestamp
                self.season_driver_q[timestamp.year][driver].append(position)
        for constructor, position in constructor_positions.items():
            self.constructor_games[constructor] += 1
            self.constructor_form[constructor] = ewma(self.constructor_form.get(constructor), position, 0.35)
            self.constructor_session_games[constructor] += 1
            top3_cars = int(((group["constructorId"] == constructor) & (group["target_position"] <= 3)).sum())
            self.constructor_session_top3[constructor] += top3_cars
            self.season_constructor_games[timestamp.year][constructor] += 1
        bisect.insort(self.calendar[timestamp.year], timestamp)

    def update_results(
        self,
        group: pd.DataFrame,
        timestamp: pd.Timestamp,
        driver_standings: dict,
        constructor_standings: dict,
        constructor_results: dict,
    ) -> None:
        field_size = len(group)
        for row in group.itertuples():
            driver = int(row.driverId)
            constructor = int(row.constructorId) if not pd.isna(row.constructorId) else -1
            grid = scalar(row.grid, field_size + 1.0)
            if grid <= 0:
                grid = field_size + 1.0
            finish = scalar(row.positionOrder, field_size + 1.0)
            self.grid_ewma[driver] = ewma(self.grid_ewma.get(driver), grid, 0.30)
            self.finish_ewma[driver] = ewma(self.finish_ewma.get(driver), finish, 0.30)
            self.result_games[driver] += 1
            self.last_result_date[driver] = timestamp
            self.observe_constructor(driver, constructor, timestamp)
        race_id = int(group["raceId"].iloc[0])
        for row in driver_standings.get(race_id, ()):
            driver = int(row.driverId)
            self.driver_standing[driver] = (
                scalar(row.points), scalar(row.position, 30.0), scalar(row.wins), timestamp.year
            )
        for row in constructor_standings.get(race_id, ()):
            constructor = int(row.constructorId)
            self.constructor_standing[constructor] = (
                scalar(row.points), scalar(row.position, 20.0), scalar(row.wins), timestamp.year
            )
        for row in constructor_results.get(race_id, ()):
            constructor = int(row.constructorId)
            points = scalar(row.points)
            self.constructor_result_ewma[constructor] = ewma(
                self.constructor_result_ewma.get(constructor), points, 0.30
            )
        self.last_global_race_date = timestamp
        season_results = self.first_two_constructor_results[timestamp.year]
        if len(season_results) < 2:
            constructor_finishes = (
                group.assign(_finish=pd.to_numeric(group["positionOrder"], errors="coerce"))
                .groupby("constructorId")["_finish"]
                .mean()
                .to_dict()
            )
            season_results.append({int(key): float(value) for key, value in constructor_finishes.items()})
            if len(season_results) == 2:
                combined = defaultdict(list)
                for race_values in season_results:
                    for constructor, finish in race_values.items():
                        combined[constructor].append(finish)
                constructors = sorted(set(combined) & set(self.prior_constructor_state))
                prior_ranks = [self.prior_constructor_state[key][1] for key in constructors]
                early_ranks = [float(np.mean(combined[key])) for key in constructors]
                agreement = self.spearman(prior_ranks, early_ranks)
                self.constructor_reshuffle_history.append((timestamp.year, agreement))


# Feature construction

def add_relative_features(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["field_size"] = frame.groupby(group_column)[group_column].transform("size").astype(float)
    directions = dict(RELATIVE_FEATURES)
    directions.update(NEW_RELATIVE_FEATURES)
    for name, direction in directions.items():
        adjusted = frame[name].astype(float) * direction
        grouped = adjusted.groupby(frame[group_column])
        frame[f"{name}_rank"] = grouped.rank(method="average", pct=True)
        means = grouped.transform("mean")
        standard = grouped.transform("std").fillna(0.0)
        frame[f"{name}_z"] = (adjusted - means) / np.where(standard > 1e-8, standard, 1.0)
        frame[f"{name}_gap"] = grouped.transform("max") - adjusted
        if name in NEW_RELATIVE_FEATURES:
            frame[f"{name}_ordinal_rank"] = grouped.rank(method="average", ascending=False)
            frame[f"{name}_percentile"] = grouped.rank(method="average", pct=True)
    return frame


def grouped_records(frame: pd.DataFrame) -> dict:
    return {int(key): list(value.itertuples(index=False)) for key, value in frame.groupby("raceId", sort=False)}


def prepare_entries(data: dict) -> pd.DataFrame:
    results = data["results"].copy()
    qualifying = data["qualifying"][["raceId", "driverId", "constructorId", "position", "date"]].copy()
    qualifying = qualifying.rename(
        columns={"constructorId": "q_constructorId", "position": "q_position", "date": "q_date"}
    )
    entries = results.merge(qualifying, on=["raceId", "driverId"], how="outer")
    races = data["races"][["raceId", "date", "circuitId", "year", "round"]].rename(columns={"date": "race_date"})
    entries = entries.merge(races, on="raceId", how="left")
    entries["constructorId"] = entries["q_constructorId"].combine_first(entries["constructorId"]).fillna(-1).astype(int)
    valid_q = entries["q_position"].notna() & (entries["q_position"] > 0)
    valid_grid = entries["grid"].notna() & (entries["grid"] > 0)
    fallback = entries["grid"].where(valid_grid, entries["positionOrder"])
    entries["target_position"] = entries["q_position"].where(valid_q, fallback).fillna(99).astype(float)
    entries["has_qualifying"] = valid_q
    entries["event_time"] = pd.to_datetime(entries["q_date"].where(valid_q, entries["race_date"]))
    entries = entries[entries["event_time"].notna() & entries["driverId"].notna()].copy()
    entries["driverId"] = entries["driverId"].astype(int)
    session_q_share = entries.groupby("raceId")["has_qualifying"].transform("mean")
    entries["session_weight"] = np.where(session_q_share >= 0.5, 1.0, 0.35)
    return entries.sort_values(["event_time", "raceId", "target_position", "driverId"])


def build_feature_frames(data: dict, queries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    races = data["races"].copy()
    circuits = data["circuits"][["circuitId", "lat", "lng"]].copy()
    circuits["lat_bin"] = np.floor((circuits["lat"].fillna(0.0) + 90.0) / 30.0).astype(int)
    circuits["lng_bin"] = np.floor((circuits["lng"].fillna(0.0) + 180.0) / 30.0).astype(int)
    circuits["cluster"] = circuits["lat_bin"] * 12 + circuits["lng_bin"]
    race_context = races[["raceId", "circuitId"]].merge(circuits[["circuitId", "cluster"]], on="circuitId", how="left")
    circuit_by_race = dict(zip(race_context["raceId"].astype(int), race_context["circuitId"].fillna(-1).astype(int)))
    cluster_by_race = dict(zip(race_context["raceId"].astype(int), race_context["cluster"].fillna(-1).astype(int)))
    entries = prepare_entries(data)
    session_groups = [(pd.Timestamp(group["event_time"].iloc[0]), int(race_id), group) for race_id, group in entries.groupby("raceId", sort=False)]
    result_groups = [(pd.Timestamp(group["date"].iloc[0]), int(race_id), group) for race_id, group in data["results"].sort_values(["date", "raceId"]).groupby("raceId", sort=False)]
    driver_standings = grouped_records(data["standings"])
    constructor_standings = grouped_records(data["constructor_standings"])
    constructor_results = grouped_records(data["constructor_results"])
    query_groups = [(pd.Timestamp(date), group) for date, group in queries.groupby("date", sort=False)]
    events = []
    for timestamp, race_id, group in session_groups:
        events.append((timestamp, 0, race_id, "session", group))
    for timestamp, race_id, group in result_groups:
        events.append((timestamp, 1, race_id, "result", group))
    for timestamp, group in query_groups:
        events.append((timestamp, 2, int(group["query_id"].min()), "query", group))
    events.sort(key=lambda item: (item[0], item[1], item[2]))
    state = RaceState(data["drivers"], data["constructors"], circuit_by_race, cluster_by_race)
    rank_rows = []
    query_rows = []
    for timestamp, _, event_id, kind, group in events:
        state.regress_to_year(timestamp.year)
        context = state.context(timestamp)
        if kind == "session":
            for row in group.itertuples():
                driver = int(row.driverId)
                constructor = int(row.constructorId)
                values = state.features(driver, constructor, timestamp, context)
                values.update(
                    {
                        "session_id": int(row.raceId),
                        "event_time": timestamp,
                        "session_year": timestamp.year,
                        "relevance": 3 if row.target_position <= 3 else 2 if row.target_position <= 6 else 1 if row.target_position <= 10 else 0,
                        "source_weight": float(row.session_weight),
                    }
                )
                rank_rows.append(values)
            state.update_session(group, timestamp)
        elif kind == "result":
            state.update_results(group, timestamp, driver_standings, constructor_standings, constructor_results)
        else:
            for row in group.itertuples():
                driver = int(row.driverId)
                constructor = int(state.last_constructor.get(driver, -1))
                values = state.features(driver, constructor, timestamp, context)
                values.update(
                    {
                        "query_id": int(row.query_id),
                        "query_group": int(timestamp.value),
                        "date": timestamp,
                        "split": str(row.split),
                        "row_index": int(row.row_index),
                        "target_label": scalar(row.target_label, np.nan),
                        "is_synthetic": scalar(row.is_synthetic),
                        "window_close": pd.Timestamp(row.window_close),
                    }
                )
                query_rows.append(values)
    rank_frame = add_relative_features(pd.DataFrame(rank_rows), "session_id")
    query_frame = add_relative_features(pd.DataFrame(query_rows), "query_group")
    return rank_frame, query_frame


def load_snapshot() -> tuple[dict, dict]:
    cache = Path(os.environ["RELBENCH_CACHE_DIR"])
    dataset_name = os.environ["RELBENCH_DATASET"]
    task_name = os.environ["RELBENCH_TASK"]
    base = cache / dataset_name
    database = base / "db"
    task_dir = base / "tasks" / task_name
    table_names = (
        "circuits",
        "constructor_results",
        "constructor_standings",
        "constructors",
        "drivers",
        "qualifying",
        "races",
        "results",
        "standings",
    )
    data = {name: pd.read_parquet(database / f"{name}.parquet") for name in table_names}
    splits = {name: pd.read_parquet(task_dir / f"{name}.parquet") for name in ("train", "val", "test")}
    return data, splits


def synthetic_opener_rows(data: dict, splits: dict, snapshot_cutoff: pd.Timestamp) -> pd.DataFrame:
    qualifying = data["qualifying"][["date", "driverId", "position"]].copy()
    qualifying["date"] = pd.to_datetime(qualifying["date"])
    qualifying = qualifying[qualifying["date"].notna() & qualifying["driverId"].notna()].copy()
    qualifying["driverId"] = qualifying["driverId"].astype(int)
    qualifying["year"] = qualifying["date"].dt.year
    official_keys = set()
    for split in ("train", "val", "test"):
        official = splits[split]
        official_keys.update(
            zip(pd.to_datetime(official["date"]).astype("int64"), official["driverId"].astype(int))
        )
    rows = []
    for year, first_date in qualifying.groupby("year")["date"].min().items():
        if int(year) < 1994:
            continue
        for offset in (28, 14):
            origin = pd.Timestamp(first_date) - pd.Timedelta(days=offset)
            window_close = origin + pd.Timedelta(days=30)
            if window_close > snapshot_cutoff:
                continue
            recent = set(
                qualifying.loc[
                    (qualifying["date"] > origin - pd.Timedelta(days=365))
                    & (qualifying["date"] <= origin),
                    "driverId",
                ].astype(int)
            )
            future = qualifying[
                (qualifying["date"] > origin) & (qualifying["date"] <= window_close)
            ]
            if not recent or future.empty:
                continue
            future = future[future["driverId"].isin(recent)]
            for driver, group in future.groupby("driverId"):
                key = (int(origin.value), int(driver))
                if key in official_keys:
                    continue
                rows.append(
                    {
                        "date": origin,
                        "driverId": int(driver),
                        "split": "synthetic",
                        "target_label": int(pd.to_numeric(group["position"], errors="coerce").min() <= 3),
                        "is_synthetic": 1.0,
                        "window_close": window_close,
                    }
                )
    return pd.DataFrame(rows)


def query_table(data: dict, splits: dict, rolling: bool) -> pd.DataFrame:
    frames = []
    next_id = 0
    for split in ("train", "val", "test"):
        frame = splits[split][["date", "driverId"]].copy()
        frame["split"] = split
        frame["target_label"] = (
            splits[split]["qualifying"].to_numpy(dtype=np.float64)
            if "qualifying" in splits[split].columns
            else np.nan
        )
        frame["is_synthetic"] = 0.0
        frame["window_close"] = pd.to_datetime(frame["date"]) + pd.Timedelta(days=30)
        frame["row_index"] = np.arange(len(frame), dtype=np.int64)
        frame["query_id"] = np.arange(next_id, next_id + len(frame), dtype=np.int64)
        next_id += len(frame)
        frames.append(frame)
    table_dates = []
    for name in ("qualifying", "results", "standings", "constructor_standings", "constructor_results", "races"):
        if "date" in data[name].columns and len(data[name]):
            table_dates.append(pd.Timestamp(data[name]["date"].max()))
    if rolling and len(splits["test"]):
        snapshot_cutoff = pd.Timestamp(splits["test"]["date"].max())
    else:
        snapshot_cutoff = max(table_dates)
    synthetic = synthetic_opener_rows(data, splits, snapshot_cutoff)
    if len(synthetic):
        synthetic["row_index"] = np.arange(len(synthetic), dtype=np.int64)
        synthetic["query_id"] = np.arange(next_id, next_id + len(synthetic), dtype=np.int64)
        frames.append(synthetic)
    return pd.concat(frames, ignore_index=True)


def feature_cache_key(data: dict, queries: pd.DataFrame) -> str:
    digest = hashlib.sha256(FEATURE_VERSION.encode())
    for name in ("qualifying", "results", "standings", "constructor_standings", "constructor_results"):
        frame = data[name]
        digest.update(f"{name}:{len(frame)}".encode())
        if "date" in frame and len(frame):
            digest.update(str(pd.Timestamp(frame["date"].max()).value).encode())
    hashed_queries = pd.util.hash_pandas_object(queries[["date", "driverId", "split", "row_index"]], index=False)
    digest.update(hashed_queries.to_numpy().tobytes())
    return digest.hexdigest()[:24]


def register_cache(cache_root: Path) -> None:
    registry = cache_root.parent / "artifacts.json"
    entry = {
        "name": FEATURE_VERSION,
        "path": FEATURE_VERSION,
        "description": "Causal dual-Elo race and exact-window feature frames keyed by snapshot content",
        "content_key": FEATURE_VERSION,
        "rebuild_hint": "Run main.py on the desired rolling origin; exact matching snapshots are reused",
    }
    try:
        records = json.loads(registry.read_text()) if registry.exists() else []
        if not any(record.get("content_key") == FEATURE_VERSION for record in records):
            records.append(entry)
            temporary = registry.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(records, indent=2))
            os.replace(temporary, registry)
    except (OSError, json.JSONDecodeError):
        pass


def get_feature_frames(data: dict, queries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    shared = Path(os.environ.get("KAPSO_SHARED_CACHE_DIR", "./shared_cache"))
    cache_root = shared / FEATURE_VERSION
    cache_root.mkdir(parents=True, exist_ok=True)
    register_cache(cache_root)
    key = feature_cache_key(data, queries)
    path = cache_root / f"features_{key}.pkl"
    if path.exists():
        try:
            with path.open("rb") as handle:
                bundle = pickle.load(handle)
            return bundle["rank"], bundle["query"], True
        except (OSError, EOFError, pickle.UnpicklingError, KeyError):
            pass
    rank_frame, query_frame = build_feature_frames(data, queries)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump({"rank": rank_frame, "query": query_frame}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    return rank_frame, query_frame, False


# Models

def feature_columns(frame: pd.DataFrame, prefix: int) -> list[str]:
    excluded = {
        "session_id",
        "event_time",
        "session_year",
        "relevance",
        "source_weight",
        "query_id",
        "query_group",
        "date",
        "split",
        "row_index",
        "target_label",
        "is_synthetic",
        "window_close",
    }
    available = [
        column
        for column in frame.columns
        if column not in excluded and column not in MODEL_EXCLUDED_FEATURES
    ]
    new_names = sum((list(group) for group in PREFIX_FEATURES.values()), [])
    base = [
        column for column in available
        if not any(column == name or column.startswith(f"{name}_") for name in new_names)
    ]
    selected = list(base)
    for level in range(1, prefix + 1):
        for name in PREFIX_FEATURES[level]:
            selected.extend(
                column
                for column in available
                if column == name or column.startswith(f"{name}_")
            )
    return list(dict.fromkeys(selected))


def fit_ranker(rank_frame: pd.DataFrame, cutoff: pd.Timestamp, columns: list[str], debug: bool):
    eligible = rank_frame[rank_frame["event_time"] <= cutoff].copy()
    eligible = eligible.sort_values(["event_time", "session_id"])
    if debug:
        sessions = eligible[["event_time", "session_id"]].drop_duplicates().tail(60)["session_id"]
        eligible = eligible[eligible["session_id"].isin(set(sessions))]
    latest = eligible["event_time"].max()
    age_days = (latest - eligible["event_time"]).dt.total_seconds() / 86400.0
    weights = eligible["source_weight"].to_numpy(dtype=np.float64) * np.power(0.5, age_days / (6.0 * 365.25))
    groups = eligible.groupby("session_id", sort=False).size().to_numpy(dtype=np.int32)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=50 if debug else 300,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=30,
        lambdarank_truncation_level=8,
        label_gain=[0, 1, 3, 7],
        reg_lambda=1.0,
        verbosity=-1,
        random_state=1337,
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")),
    )
    model.fit(
        eligible[columns],
        eligible["relevance"].astype(int),
        group=groups,
        sample_weight=weights,
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


def split_features(query_frame: pd.DataFrame, split: str) -> pd.DataFrame:
    return query_frame[query_frame["split"] == split].sort_values("row_index").reset_index(drop=True)


def fit_meta(features: np.ndarray, labels: np.ndarray):
    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return float(np.mean(labels))
    model = logistic_pipeline(0.03)
    model.fit(features, labels)
    return model


def predict_meta(model, features: np.ndarray) -> np.ndarray:
    if isinstance(model, float):
        return np.full(len(features), model, dtype=np.float64)
    return model.predict_proba(features)[:, 1].astype(np.float64)


def build_oof_bundle(
    rank_frame: pd.DataFrame,
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    rank_columns: list[str],
    binary_columns: list[str],
    debug: bool,
) -> dict:
    years = train_features["date"].dt.year.to_numpy()
    binary_oof = np.full(len(train_features), np.nan)
    rank_oof = np.full(len(train_features), np.nan)
    for year in FORWARD_YEARS:
        test_mask = years == year
        fit_mask = years < year
        if test_mask.sum() == 0 or fit_mask.sum() == 0 or len(np.unique(train_labels[fit_mask])) < 2:
            continue
        binary_model = fit_binary(train_features.loc[fit_mask, binary_columns], train_labels[fit_mask])
        binary_oof[test_mask] = predict_binary(binary_model, train_features.loc[test_mask, binary_columns])
        cutoff = pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(microseconds=1)
        rank_model = fit_ranker(rank_frame, cutoff, rank_columns, debug)
        raw_rank = rank_model.predict(train_features.loc[test_mask, rank_columns])
        rank_oof[test_mask] = origin_standardize(raw_rank, train_features.loc[test_mask, "date"])
    valid = np.isfinite(binary_oof) & np.isfinite(rank_oof)
    meta_values = np.column_stack(
        [safe_logit(binary_oof[valid]), rank_oof[valid], train_features.loc[valid, "expected_sessions"].to_numpy()]
    )
    meta_labels = train_labels[valid]
    meta_model = fit_meta(meta_values, meta_labels)
    fold_metrics = []
    for year in FORWARD_YEARS[1:]:
        meta_train = valid & (years < year)
        meta_test = valid & (years == year)
        if meta_train.sum() == 0 or meta_test.sum() == 0:
            continue
        if len(np.unique(train_labels[meta_train])) < 2 or len(np.unique(train_labels[meta_test])) < 2:
            continue
        nested_model = fit_meta(
            np.column_stack(
                [safe_logit(binary_oof[meta_train]), rank_oof[meta_train], train_features.loc[meta_train, "expected_sessions"].to_numpy()]
            ),
            train_labels[meta_train],
        )
        stacked = predict_meta(
            nested_model,
            np.column_stack(
                [safe_logit(binary_oof[meta_test]), rank_oof[meta_test], train_features.loc[meta_test, "expected_sessions"].to_numpy()]
            ),
        )
        binary_auc = float(roc_auc_score(train_labels[meta_test], binary_oof[meta_test]))
        stacked_auc = float(roc_auc_score(train_labels[meta_test], stacked))
        fold_metrics.append({"year": year, "n": int(meta_test.sum()), "binary_auc": binary_auc, "stacked_auc": stacked_auc})
    if fold_metrics:
        binary_mean = float(np.mean([item["binary_auc"] for item in fold_metrics]))
        stacked_mean = float(np.mean([item["stacked_auc"] for item in fold_metrics]))
        binary_worst = float(np.min([item["binary_auc"] for item in fold_metrics]))
        stacked_worst = float(np.min([item["stacked_auc"] for item in fold_metrics]))
        ship_stack = stacked_mean >= binary_mean + 0.005 and stacked_worst >= binary_worst
    else:
        binary_mean = stacked_mean = binary_worst = stacked_worst = float("nan")
        ship_stack = False
    return {
        "meta_model": meta_model,
        "ship_stack": bool(ship_stack),
        "fold_metrics": fold_metrics,
        "binary_mean": binary_mean,
        "stacked_mean": stacked_mean,
        "binary_worst": binary_worst,
        "stacked_worst": stacked_worst,
        "oof_count": int(valid.sum()),
    }


def get_oof_bundle(
    rank_frame: pd.DataFrame,
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    rank_columns: list[str],
    binary_columns: list[str],
    debug: bool,
) -> tuple[dict, bool]:
    shared = Path(os.environ.get("KAPSO_SHARED_CACHE_DIR", "./shared_cache")) / FEATURE_VERSION
    shared.mkdir(parents=True, exist_ok=True)
    path = shared / f"meta_{MODEL_VERSION}_{'debug' if debug else 'full'}.pkl"
    if path.exists():
        try:
            with path.open("rb") as handle:
                return pickle.load(handle), True
        except (OSError, EOFError, pickle.UnpicklingError):
            pass
    bundle = build_oof_bundle(
        rank_frame, train_features, train_labels, rank_columns, binary_columns, debug
    )
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    return bundle, False


def predict_chain(
    rank_frame: pd.DataFrame,
    fit_features: pd.DataFrame,
    fit_labels: np.ndarray,
    prediction_features: pd.DataFrame,
    cutoff: pd.Timestamp,
    rank_columns: list[str],
    binary_columns: list[str],
    oof_bundle: dict,
    debug: bool,
) -> tuple[np.ndarray, dict]:
    binary_model = fit_binary(fit_features[binary_columns], fit_labels)
    binary_probability = predict_binary(binary_model, prediction_features[binary_columns])
    rank_model = fit_ranker(rank_frame, cutoff, rank_columns, debug)
    rank_raw = rank_model.predict(prediction_features[rank_columns])
    rank_z = origin_standardize(rank_raw, prediction_features["date"])
    meta_values = np.column_stack(
        [safe_logit(binary_probability), rank_z, prediction_features["expected_sessions"].to_numpy()]
    )
    stacked_probability = predict_meta(oof_bundle["meta_model"], meta_values)
    selected = stacked_probability if oof_bundle["ship_stack"] else binary_probability
    diagnostics = {
        "binary_min": float(np.min(binary_probability)),
        "binary_max": float(np.max(binary_probability)),
        "rank_std": float(np.std(rank_z)),
        "stacked_min": float(np.min(stacked_probability)),
        "stacked_max": float(np.max(stacked_probability)),
        "selected": "stacked" if oof_bundle["ship_stack"] else "binary",
    }
    return np.clip(selected, 1e-6, 1.0 - 1e-6), diagnostics


# Prequential selection

def origin_weights(features: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    dates = pd.to_datetime(features["date"])
    counts = dates.groupby(dates).transform("size").to_numpy(dtype=np.float64)
    age_days = np.maximum(
        (pd.Timestamp(cutoff) - dates).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0,
        0.0,
    )
    weights = np.power(0.5, age_days / (RECENCY_HALF_LIFE_YEARS * 365.25)) / np.maximum(counts, 1.0)
    opener = (
        (features["opener_flag"].to_numpy(dtype=np.float64) > 0.5)
        | (features["is_synthetic"].to_numpy(dtype=np.float64) > 0.5)
    )
    opener_mass = float(weights[opener].sum())
    regular_mass = float(weights[~opener].sum())
    if opener_mass > 0.0 and opener_mass / max(opener_mass + regular_mass, 1e-12) > MAX_OPENER_MASS:
        weights[opener] *= (MAX_OPENER_MASS * regular_mass) / (
            (1.0 - MAX_OPENER_MASS) * opener_mass
        )
    return weights / max(float(weights.mean()), 1e-12)


def safe_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, predictions))


def clustered_bootstrap_support(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    dates: np.ndarray,
    draws: int,
) -> tuple[float, float]:
    unique_dates = np.unique(dates)
    grouped = [np.flatnonzero(dates == date) for date in unique_dates]
    rng = np.random.default_rng(RANDOM_SEED)
    deltas = []
    for _ in range(draws):
        sampled = rng.integers(0, len(grouped), len(grouped))
        indices = np.concatenate([grouped[index] for index in sampled])
        if len(np.unique(labels[indices])) < 2:
            continue
        deltas.append(
            roc_auc_score(labels[indices], candidate[indices])
            - roc_auc_score(labels[indices], baseline[indices])
        )
    if not deltas:
        return 0.0, float("nan")
    values = np.asarray(deltas, dtype=np.float64)
    return float(np.mean(values > 0.0)), float(np.std(values, ddof=1))


def prequential_gate(train_features: pd.DataFrame, debug: bool) -> dict:
    frame = train_features[
        train_features["date"].dt.year.isin(PREQUENTIAL_YEARS)
    ].copy()
    if debug:
        retained_dates = []
        for year in PREQUENTIAL_YEARS:
            retained_dates.extend(sorted(frame.loc[frame["date"].dt.year == year, "date"].unique())[-4:])
        frame = frame[frame["date"].isin(retained_dates)].copy()
    prediction_parts = []
    all_columns = {prefix: feature_columns(train_features, prefix) for prefix in range(4)}
    for year in PREQUENTIAL_YEARS:
        yearly = frame[frame["date"].dt.year == year]
        base_close = pd.Timestamp(year=year - 3, month=12, day=31, hour=23, minute=59, second=59)
        for origin in sorted(yearly["date"].unique()):
            origin = pd.Timestamp(origin)
            test_mask = train_features["date"] == origin
            fit_mask = (train_features["window_close"] <= base_close) | (
                (train_features["date"].dt.year == year)
                & (train_features["window_close"] < origin)
            )
            fit = train_features[fit_mask]
            test = train_features[test_mask]
            labels = fit["target_label"].to_numpy(dtype=np.int64)
            if len(fit) == 0 or len(test) == 0 or len(np.unique(labels)) < 2:
                continue
            weights = origin_weights(fit, origin)
            part = test[["date", "target_label", "days_since_q", "history_none", "opener_flag", "is_synthetic"]].copy()
            for prefix in range(4):
                columns = all_columns[prefix]
                model = fit_binary(fit[columns], labels, weights)
                part[f"prediction_{prefix}"] = predict_binary(model, test[columns])
            prediction_parts.append(part)
    if not prediction_parts:
        return {"selected_prefix": 0, "error": "no eligible prequential rows"}
    predictions = pd.concat(prediction_parts, ignore_index=True)
    labels = predictions["target_label"].to_numpy(dtype=np.int64)
    dates = predictions["date"].astype("int64").to_numpy()
    recency = predictions["days_since_q"].to_numpy(dtype=np.float64)
    no_history = predictions["history_none"].to_numpy(dtype=np.float64) > 0.5
    slice_masks = {
        "lt92d": (recency < 92.0) & ~no_history,
        "92_365d": (recency >= 92.0) & (recency <= 365.0) & ~no_history,
        "gt365d_or_no_history": (recency > 365.0) | no_history,
    }
    draws = 200 if debug else BOOTSTRAP_DRAWS
    metrics = {}
    for prefix in range(4):
        values = predictions[f"prediction_{prefix}"].to_numpy(dtype=np.float64)
        yearly_aucs = {
            str(year): safe_auc(
                labels[predictions["date"].dt.year.to_numpy() == year],
                values[predictions["date"].dt.year.to_numpy() == year],
            )
            for year in PREQUENTIAL_YEARS
        }
        finite_years = [value for value in yearly_aucs.values() if np.isfinite(value)]
        slices = {
            name: {
                "n": int(mask.sum()),
                "auc": safe_auc(labels[mask], values[mask]),
            }
            for name, mask in slice_masks.items()
        }
        metrics[str(prefix)] = {
            "pooled_auc": safe_auc(labels, values),
            "mean_yearly_auc": float(np.mean(finite_years)) if finite_years else float("nan"),
            "worst_year_auc": float(np.min(finite_years)) if finite_years else float("nan"),
            "yearly_auc": yearly_aucs,
            "slices": slices,
        }
        if prefix > 0:
            support, bootstrap_sd = clustered_bootstrap_support(
                labels,
                predictions["prediction_0"].to_numpy(dtype=np.float64),
                values,
                dates,
                draws,
            )
            metrics[str(prefix)]["bootstrap_positive_support"] = support
            metrics[str(prefix)]["bootstrap_delta_sd"] = bootstrap_sd
    baseline = metrics["0"]
    passing = []
    for prefix in range(1, 4):
        candidate = metrics[str(prefix)]
        mean_delta = candidate["mean_yearly_auc"] - baseline["mean_yearly_auc"]
        worst_delta = candidate["worst_year_auc"] - baseline["worst_year_auc"]
        stale_delta = (
            candidate["slices"]["92_365d"]["auc"]
            - baseline["slices"]["92_365d"]["auc"]
        )
        support = candidate["bootstrap_positive_support"]
        overall_gate = (
            mean_delta >= 0.005
            and worst_delta >= -0.005
            and stale_delta >= -0.005
            and support >= 0.80
        )
        stale_gate = (
            stale_delta >= 0.02
            and mean_delta >= -0.001
            and worst_delta >= -0.005
            and support >= 0.80
        )
        candidate["mean_delta"] = mean_delta
        candidate["worst_delta"] = worst_delta
        candidate["stale_delta"] = stale_delta
        candidate["passed"] = bool(overall_gate or stale_gate)
        candidate["gate_path"] = "overall" if overall_gate else "stale" if stale_gate else "none"
        if candidate["passed"]:
            passing.append(prefix)
    weak = slice_masks["92_365d"]
    weak_opener = (
        (predictions["opener_flag"].to_numpy(dtype=np.float64) > 0.5)
        | (predictions["is_synthetic"].to_numpy(dtype=np.float64) > 0.5)
    ) & weak
    return {
        "selected_prefix": max(passing) if passing else 0,
        "rows": int(len(predictions)),
        "origins": int(predictions["date"].nunique()),
        "synthetic_rows": int((predictions["is_synthetic"] > 0.5).sum()),
        "weak_band_rows": int(weak.sum()),
        "weak_band_opener_rows": int(weak_opener.sum()),
        "weak_band_opener_fraction": float(weak_opener.sum() / max(weak.sum(), 1)),
        "bootstrap_draws": draws,
        "metrics": metrics,
    }


def get_prequential_gate(train_features: pd.DataFrame, debug: bool) -> tuple[dict, bool]:
    shared = Path(os.environ.get("KAPSO_SHARED_CACHE_DIR", "./shared_cache")) / FEATURE_VERSION
    shared.mkdir(parents=True, exist_ok=True)
    path = shared / f"gate_{MODEL_VERSION}_{'debug' if debug else 'full'}.json"
    if path.exists():
        try:
            return json.loads(path.read_text()), True
        except (OSError, json.JSONDecodeError):
            pass
    gate = prequential_gate(train_features, debug)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(gate, allow_nan=True))
    os.replace(temporary, path)
    return gate, False


def fit_predict_logistic(
    fit_features: pd.DataFrame,
    prediction_features: pd.DataFrame,
    prefix: int,
    cutoff: pd.Timestamp,
) -> tuple[np.ndarray, dict]:
    columns = feature_columns(fit_features, prefix)
    labels = fit_features["target_label"].to_numpy(dtype=np.int64)
    weights = origin_weights(fit_features, cutoff)
    model = fit_binary(fit_features[columns], labels, weights)
    predictions = np.clip(
        predict_binary(model, prediction_features[columns]),
        1e-6,
        1.0 - 1e-6,
    )
    return predictions, {
        "prefix": int(prefix),
        "feature_count": int(len(columns)),
        "fit_rows": int(len(fit_features)),
        "fit_origins": int(fit_features["date"].nunique()),
        "synthetic_rows": int((fit_features["is_synthetic"] > 0.5).sum()),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
    }


# Orchestration

def main() -> None:
    warnings.filterwarnings("ignore")
    start = time.time()
    debug = "--debug" in sys.argv
    data, splits = load_snapshot()
    rolling = len(splits["val"]) == 0
    queries = query_table(data, splits, rolling)
    elapsed(start, "loaded direct parquet snapshot")
    rank_frame, query_frame, feature_cache_hit = get_feature_frames(data, queries)
    elapsed(start, f"features ready cache_hit={feature_cache_hit} race_rows={len(rank_frame)} query_rows={len(query_frame)}")
    train_features = split_features(query_frame, "train")
    val_features = split_features(query_frame, "val")
    test_features = split_features(query_frame, "test")
    synthetic_features = split_features(query_frame, "synthetic")
    training_pool = pd.concat([train_features, synthetic_features], ignore_index=True)
    training_pool["date"] = pd.to_datetime(training_pool["date"])
    training_pool["window_close"] = pd.to_datetime(training_pool["window_close"])
    training_pool = training_pool[np.isfinite(training_pool["target_label"])].reset_index(drop=True)
    if rolling:
        gate_pool = training_pool
    else:
        validation_start = pd.Timestamp(val_features["date"].min())
        gate_pool = training_pool[
            (training_pool["split"] == "train")
            | (training_pool["window_close"] < validation_start)
        ].reset_index(drop=True)
    gate, gate_cache_hit = get_prequential_gate(gate_pool, debug)
    selected_prefix = int(gate.get("selected_prefix", 0))
    gate_log = {
        "cache_hit": gate_cache_hit,
        "selected_prefix": selected_prefix,
        "rows": gate.get("rows"),
        "origins": gate.get("origins"),
        "synthetic_rows": gate.get("synthetic_rows"),
        "weak_band_rows": gate.get("weak_band_rows"),
        "weak_band_opener_fraction": gate.get("weak_band_opener_fraction"),
    }
    if not gate_cache_hit:
        gate_log["metrics"] = gate.get("metrics")
    print("[dual-elo] prequential gate " + json.dumps(gate_log), flush=True)
    elapsed(start, "prequential prefixes ready")
    output = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "./output_data_generic_exp_4"))
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        "debug": debug,
        "rolling": rolling,
        "feature_cache_hit": feature_cache_hit,
        "gate_cache_hit": gate_cache_hit,
        "gate": gate,
    }
    if rolling:
        cutoff = pd.Timestamp(test_features["date"].max())
        closed_training = training_pool[training_pool["window_close"] <= cutoff].reset_index(drop=True)
        test_predictions, test_diagnostics = fit_predict_logistic(
            closed_training,
            test_features,
            selected_prefix,
            cutoff,
        )
        np.save(output / "test_predictions.npy", test_predictions.astype(np.float64))
        diagnostics["test"] = test_diagnostics
    else:
        val_cutoff = pd.Timestamp(val_features["date"].min()) - pd.Timedelta(microseconds=1)
        model_a_training = training_pool[
            (training_pool["split"] == "train")
            | (training_pool["window_close"] <= val_cutoff)
        ].reset_index(drop=True)
        val_predictions, val_diagnostics = fit_predict_logistic(
            model_a_training,
            val_features,
            selected_prefix,
            val_cutoff,
        )
        test_cutoff = pd.Timestamp(test_features["date"].min()) - pd.Timedelta(microseconds=1)
        combined_features = pd.concat([train_features, val_features, synthetic_features], ignore_index=True)
        combined_features["window_close"] = pd.to_datetime(combined_features["window_close"])
        combined_features = combined_features[
            np.isfinite(combined_features["target_label"])
            & (combined_features["window_close"] <= test_cutoff)
        ].reset_index(drop=True)
        test_predictions, test_diagnostics = fit_predict_logistic(
            combined_features,
            test_features,
            selected_prefix,
            test_cutoff,
        )
        np.save(output / "val_predictions.npy", val_predictions.astype(np.float64))
        np.save(output / "test_predictions.npy", test_predictions.astype(np.float64))
        diagnostics["val"] = val_diagnostics
        diagnostics["test"] = test_diagnostics
    diagnostics["elapsed_seconds"] = time.time() - start
    (output / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    elapsed(start, f"predictions written test_shape={test_predictions.shape}")


if __name__ == "__main__":
    main()
