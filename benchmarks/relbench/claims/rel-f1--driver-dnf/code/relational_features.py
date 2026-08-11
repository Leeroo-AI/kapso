from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_VERSION = "lane1_compact_rel_v2"
BASE_FEATURE_NAMES = [
    "driver_dnf_rate",
    "driver_status_11_19_rate",
    "driver_status_3_4_20_rate",
    "driver_other_dnf_rate",
    "driver_ewma_90",
    "driver_ewma_365",
    "driver_ewma_1095",
    "constructor_dnf_rate",
    "constructor_status_11_19_rate",
    "constructor_status_3_4_20_rate",
    "constructor_other_dnf_rate",
    "constructor_ewma_90",
    "constructor_ewma_365",
    "constructor_ewma_1095",
    "driver_constructor_dnf_rate",
    "career_count_log",
    "driver_age_years",
    "inactivity_days_log",
    "starts_365",
    "constructor_tenure_days_log",
    "team_switch",
    "grid_mean_5",
    "position_order_mean_5",
    "points_mean_5",
    "laps_mean_5",
    "recent_finish_rate",
    "qualifying_position_mean_5",
    "qualifying_count_365",
    "driver_standing_points",
    "driver_standing_position",
    "driver_standing_points_trend",
    "driver_constructor_nationality_match",
    "constructor_standing_points",
    "constructor_standing_position",
    "constructor_standing_points_trend",
    "constructor_results_momentum",
    "last_circuit_attrition",
    "driver_last_circuit_dnf_rate",
    "circuit_attrition_transition",
    "recent_circuit_travel_1000km",
    "era_dnf_rate_365",
    "season_round",
    "season_phase",
    "calendar_races_30d_mean",
    "calendar_races_30d_p0",
    "calendar_races_30d_p1",
    "calendar_races_30d_p2",
    "calendar_races_30d_p3plus",
]
RELATIVE_FEATURES = [
    "driver_dnf_rate",
    "driver_ewma_365",
    "constructor_dnf_rate",
    "driver_constructor_dnf_rate",
    "position_order_mean_5",
    "driver_standing_position",
]
FEATURE_NAMES = BASE_FEATURE_NAMES + [
    f"{name}_{kind}" for name in RELATIVE_FEATURES for kind in ("origin_pct", "origin_z")
]


def read_snapshot() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache = Path(os.environ["RELBENCH_CACHE_DIR"])
    dataset = os.environ["RELBENCH_DATASET"]
    task = os.environ["RELBENCH_TASK"]
    database_root = cache / dataset / "db"
    task_root = cache / dataset / "tasks" / task
    names = [
        "circuits",
        "constructor_results",
        "constructor_standings",
        "constructors",
        "drivers",
        "qualifying",
        "races",
        "results",
        "standings",
    ]
    tables = {name: pd.read_parquet(database_root / f"{name}.parquet") for name in names}
    train = pd.read_parquet(task_root / "train.parquet")
    val = pd.read_parquet(task_root / "val.parquet")
    test = pd.read_parquet(task_root / "test.parquet")
    for frame in (train, val, test):
        frame["date"] = pd.to_datetime(frame["date"])
        frame["_row_id"] = np.arange(len(frame), dtype=np.int64)
    return tables, train, val, test


def _sorted_groups(frame: pd.DataFrame, key: str) -> dict[int, pd.DataFrame]:
    if len(frame) == 0:
        return {}
    ordered = frame.sort_values("date", kind="stable")
    return {int(value): group.reset_index(drop=True) for value, group in ordered.groupby(key, sort=False)}


def _slice(group: pd.DataFrame | None, timestamp: pd.Timestamp) -> pd.DataFrame:
    if group is None or len(group) == 0:
        return pd.DataFrame(columns=[] if group is None else group.columns)
    dates = group["date"].to_numpy(dtype="datetime64[ns]")
    end = int(np.searchsorted(dates, np.datetime64(timestamp), side="right"))
    return group.iloc[:end]


def _smoothed_rate(values: np.ndarray, prior: float, alpha: float) -> float:
    count = len(values)
    return float((np.asarray(values, dtype=np.float64).sum() + alpha * prior) / (count + alpha))


def _ewma(values: np.ndarray, dates: pd.Series, timestamp: pd.Timestamp, horizon: float, prior: float) -> float:
    if len(values) == 0:
        return float(prior)
    age = (timestamp.to_datetime64() - dates.to_numpy(dtype="datetime64[ns]")) / np.timedelta64(1, "D")
    weights = np.exp(-np.maximum(np.asarray(age, dtype=np.float64), 0.0) / horizon)
    return float((np.dot(weights, np.asarray(values, dtype=np.float64)) + 3.0 * prior) / (weights.sum() + 3.0))


def _mean(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if len(array) and np.isfinite(array).any() else np.nan


def _standing_features(history: pd.DataFrame) -> tuple[float, float, float, float]:
    if len(history) == 0:
        return np.nan, np.nan, 0.0, 0.0
    latest = history.iloc[-1]
    previous = history.iloc[-2] if len(history) > 1 else latest
    return (
        float(latest["points"]),
        float(latest["position"]),
        float(latest["points"] - previous["points"]),
        float(previous["position"] - latest["position"]),
    )


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if not np.all(np.isfinite([lat1, lon1, lat2, lon2])):
        return np.nan
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    value = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return float(2.0 * 6371.0 * math.asin(math.sqrt(value)) / 1000.0)


def _calendar_features(races: pd.DataFrame, timestamp: pd.Timestamp) -> tuple[float, float, float, float, float]:
    years = sorted(int(year) for year in races.loc[races["year"] < timestamp.year, "year"].unique())[-10:]
    counts: list[int] = []
    for year in years:
        day = min(timestamp.day, 28 if timestamp.month == 2 else timestamp.day)
        anchor = pd.Timestamp(year=year, month=timestamp.month, day=day)
        count = int(((races["date"] > anchor) & (races["date"] <= anchor + pd.Timedelta(days=30))).sum())
        counts.append(count)
    if not counts:
        return 1.5, 0.0, 0.5, 0.5, 0.0
    array = np.asarray(counts)
    return (
        float(array.mean()),
        float(np.mean(array == 0)),
        float(np.mean(array == 1)),
        float(np.mean(array == 2)),
        float(np.mean(array >= 3)),
    )


class RelationalFeatureBuilder:
    def __init__(self, tables: dict[str, pd.DataFrame], cache_root: Path | None = None):
        self.tables = tables
        self.cache_root = cache_root
        self.results = tables["results"].merge(
            tables["races"][["raceId", "circuitId", "year", "round"]], on="raceId", how="left", validate="many_to_one"
        ).sort_values("date", kind="stable").reset_index(drop=True)
        self.results["dnf"] = (self.results["statusId"] != 1).astype(np.float64)
        self.results["severe"] = self.results["statusId"].between(11, 19).astype(np.float64)
        self.results["crash"] = self.results["statusId"].isin([3, 4, 20]).astype(np.float64)
        self.results["other"] = (
            (self.results["statusId"] != 1)
            & ~self.results["statusId"].between(11, 19)
            & ~self.results["statusId"].isin([3, 4, 20])
        ).astype(np.float64)
        self.races = tables["races"].sort_values("date", kind="stable").reset_index(drop=True)
        self.driver_results = _sorted_groups(self.results, "driverId")
        self.constructor_results_rows = _sorted_groups(self.results, "constructorId")
        self.circuit_results = _sorted_groups(self.results, "circuitId")
        self.qualifying = _sorted_groups(tables["qualifying"], "driverId")
        self.driver_standings = _sorted_groups(tables["standings"], "driverId")
        self.constructor_standings = _sorted_groups(tables["constructor_standings"], "constructorId")
        self.constructor_results = _sorted_groups(tables["constructor_results"], "constructorId")
        self.drivers = tables["drivers"].set_index("driverId", drop=False)
        self.constructors = tables["constructors"].set_index("constructorId", drop=False)
        self.circuits = tables["circuits"].set_index("circuitId", drop=False)
        self.result_dates = self.results["date"].to_numpy(dtype="datetime64[ns]")
        self.race_dates = self.races["date"].to_numpy(dtype="datetime64[ns]")
        if cache_root is not None:
            self.feature_cache = cache_root / "lane1_tabpfn_v2" / "features" / FEATURE_VERSION
            self.feature_cache.mkdir(parents=True, exist_ok=True)
        else:
            self.feature_cache = None

    def _origin_cache_path(self, timestamp: pd.Timestamp, driver_ids: np.ndarray) -> Path | None:
        if self.feature_cache is None or len(self.results) == 0 or self.results["date"].max() < timestamp:
            return None
        signature = int(np.dot(driver_ids.astype(np.int64), np.arange(1, len(driver_ids) + 1, dtype=np.int64)) % 1_000_000_007)
        return self.feature_cache / f"{timestamp.strftime('%Y%m%dT%H%M%S')}_{len(driver_ids)}_{signature}.npz"

    def _origin_base(self, frame: pd.DataFrame, timestamp: pd.Timestamp) -> np.ndarray:
        result_end = int(np.searchsorted(self.result_dates, np.datetime64(timestamp), side="right"))
        race_end = int(np.searchsorted(self.race_dates, np.datetime64(timestamp), side="right"))
        result_history = self.results.iloc[:result_end]
        race_history = self.races.iloc[:race_end]
        if len(result_history):
            prior_dnf = float(result_history["dnf"].mean())
            prior_severe = float(result_history["severe"].mean())
            prior_crash = float(result_history["crash"].mean())
            prior_other = float(result_history["other"].mean())
        else:
            prior_dnf, prior_severe, prior_crash, prior_other = 0.75, 0.15, 0.10, 0.50
        recent_era = result_history[result_history["date"] > timestamp - pd.Timedelta(days=365)]
        era_dnf = float(recent_era["dnf"].mean()) if len(recent_era) else prior_dnf
        if len(race_history):
            latest_race = race_history.iloc[-1]
            season_round = float(latest_race["round"])
            prior_seasons = race_history[race_history["year"] < timestamp.year].groupby("year")["round"].max().tail(10)
            estimated_rounds = float(prior_seasons.median()) if len(prior_seasons) else max(season_round, 10.0)
            season_phase = season_round / max(estimated_rounds, 1.0)
        else:
            season_round, season_phase = 0.0, 0.0
        calendar = _calendar_features(race_history, timestamp)
        recent_races = race_history.drop_duplicates("raceId").tail(2)
        current_circuit = int(recent_races.iloc[-1]["circuitId"]) if len(recent_races) else -1
        previous_circuit = int(recent_races.iloc[-2]["circuitId"]) if len(recent_races) > 1 else current_circuit
        current_circuit_history = _slice(self.circuit_results.get(current_circuit), timestamp)
        previous_circuit_history = _slice(self.circuit_results.get(previous_circuit), timestamp)
        current_attrition = _smoothed_rate(current_circuit_history.get("dnf", pd.Series(dtype=float)).to_numpy(), prior_dnf, 12.0)
        previous_attrition = _smoothed_rate(previous_circuit_history.get("dnf", pd.Series(dtype=float)).to_numpy(), prior_dnf, 12.0)
        if current_circuit in self.circuits.index and previous_circuit in self.circuits.index:
            current_location = self.circuits.loc[current_circuit]
            previous_location = self.circuits.loc[previous_circuit]
            travel = _haversine(
                float(previous_location["lat"]),
                float(previous_location["lng"]),
                float(current_location["lat"]),
                float(current_location["lng"]),
            )
        else:
            travel = np.nan
        rows: list[list[float]] = []
        for seed in frame.itertuples(index=False):
            driver_id = int(seed.driverId)
            driver_history = _slice(self.driver_results.get(driver_id), timestamp)
            if len(driver_history):
                constructor_id = int(driver_history.iloc[-1]["constructorId"])
                constructor_history = _slice(self.constructor_results_rows.get(constructor_id), timestamp)
                pair_history = driver_history[driver_history["constructorId"] == constructor_id]
                inactivity = max(float((timestamp - driver_history.iloc[-1]["date"]).days), 0.0)
                career_age = max(float((timestamp - driver_history.iloc[0]["date"]).days), 0.0)
                constructors = driver_history["constructorId"].to_numpy()
                if len(constructors) > 1:
                    team_switch = float(constructors[-1] != constructors[-2])
                else:
                    team_switch = 0.0
                reverse_change = np.flatnonzero(constructors[::-1] != constructor_id)
                tenure_start = len(constructors) - int(reverse_change[0]) if len(reverse_change) else 0
                tenure_date = driver_history.iloc[min(tenure_start, len(driver_history) - 1)]["date"]
                tenure = max(float((timestamp - tenure_date).days), 0.0)
            else:
                constructor_id = -1
                constructor_history = pd.DataFrame(columns=self.results.columns)
                pair_history = pd.DataFrame(columns=self.results.columns)
                inactivity, career_age, team_switch, tenure = 3650.0, 0.0, 0.0, 0.0
            driver_dnf = driver_history.get("dnf", pd.Series(dtype=float)).to_numpy()
            constructor_dnf = constructor_history.get("dnf", pd.Series(dtype=float)).to_numpy()
            pair_dnf = pair_history.get("dnf", pd.Series(dtype=float)).to_numpy()
            recent_driver = driver_history.tail(5)
            qualifying_history = _slice(self.qualifying.get(driver_id), timestamp)
            recent_qualifying = qualifying_history.tail(5)
            qualifying_365 = qualifying_history[qualifying_history.get("date", pd.Series(dtype="datetime64[ns]")) > timestamp - pd.Timedelta(days=365)] if len(qualifying_history) else qualifying_history
            driver_standing = _slice(self.driver_standings.get(driver_id), timestamp)
            constructor_standing = _slice(self.constructor_standings.get(constructor_id), timestamp)
            driver_points, driver_position, driver_points_trend, driver_position_trend = _standing_features(driver_standing)
            constructor_points, constructor_position, constructor_points_trend, _ = _standing_features(constructor_standing)
            constructor_result_history = _slice(self.constructor_results.get(constructor_id), timestamp)
            if len(constructor_result_history):
                latest_constructor_points = float(constructor_result_history.iloc[-1]["points"])
                earlier_constructor_points = _mean(constructor_result_history.iloc[-4:-1]["points"])
                constructor_momentum = latest_constructor_points - earlier_constructor_points if np.isfinite(earlier_constructor_points) else 0.0
            else:
                constructor_momentum = np.nan
            if len(driver_history) and current_circuit >= 0:
                driver_circuit = driver_history[driver_history["circuitId"] == current_circuit]
                driver_circuit_rate = _smoothed_rate(driver_circuit["dnf"].to_numpy(), current_attrition, 4.0)
            else:
                driver_circuit_rate = current_attrition
            if driver_id in self.drivers.index:
                dob = pd.Timestamp(self.drivers.loc[driver_id]["dob"])
                age = float((timestamp - dob).days / 365.25) if not pd.isna(dob) else np.nan
                driver_nationality = str(self.drivers.loc[driver_id]["nationality"]).strip().lower()
            else:
                age = np.nan
                driver_nationality = ""
            if constructor_id in self.constructors.index:
                constructor_nationality = str(self.constructors.loc[constructor_id]["nationality"]).strip().lower()
            else:
                constructor_nationality = ""
            nationality_match = float(bool(driver_nationality) and driver_nationality == constructor_nationality)
            starts_365 = int((driver_history.get("date", pd.Series(dtype="datetime64[ns]")) > timestamp - pd.Timedelta(days=365)).sum()) if len(driver_history) else 0
            row = [
                _smoothed_rate(driver_dnf, prior_dnf, 6.0),
                _smoothed_rate(driver_history.get("severe", pd.Series(dtype=float)).to_numpy(), prior_severe, 6.0),
                _smoothed_rate(driver_history.get("crash", pd.Series(dtype=float)).to_numpy(), prior_crash, 6.0),
                _smoothed_rate(driver_history.get("other", pd.Series(dtype=float)).to_numpy(), prior_other, 6.0),
                _ewma(driver_dnf, driver_history.get("date", pd.Series(dtype="datetime64[ns]")), timestamp, 90.0, prior_dnf),
                _ewma(driver_dnf, driver_history.get("date", pd.Series(dtype="datetime64[ns]")), timestamp, 365.0, prior_dnf),
                _ewma(driver_dnf, driver_history.get("date", pd.Series(dtype="datetime64[ns]")), timestamp, 1095.0, prior_dnf),
                _smoothed_rate(constructor_dnf, prior_dnf, 12.0),
                _smoothed_rate(constructor_history.get("severe", pd.Series(dtype=float)).to_numpy(), prior_severe, 12.0),
                _smoothed_rate(constructor_history.get("crash", pd.Series(dtype=float)).to_numpy(), prior_crash, 12.0),
                _smoothed_rate(constructor_history.get("other", pd.Series(dtype=float)).to_numpy(), prior_other, 12.0),
                _ewma(constructor_dnf, constructor_history.get("date", pd.Series(dtype="datetime64[ns]")), timestamp, 90.0, prior_dnf),
                _ewma(constructor_dnf, constructor_history.get("date", pd.Series(dtype="datetime64[ns]")), timestamp, 365.0, prior_dnf),
                _ewma(constructor_dnf, constructor_history.get("date", pd.Series(dtype="datetime64[ns]")), timestamp, 1095.0, prior_dnf),
                _smoothed_rate(pair_dnf, prior_dnf, 8.0),
                float(np.log1p(len(driver_history))),
                age,
                float(np.log1p(inactivity)),
                float(starts_365),
                float(np.log1p(tenure)),
                team_switch,
                _mean(recent_driver.get("grid", pd.Series(dtype=float))),
                _mean(recent_driver.get("positionOrder", pd.Series(dtype=float))),
                _mean(recent_driver.get("points", pd.Series(dtype=float))),
                _mean(recent_driver.get("laps", pd.Series(dtype=float))),
                float(1.0 - _mean(recent_driver.get("dnf", pd.Series(dtype=float)))) if len(recent_driver) else np.nan,
                _mean(recent_qualifying.get("position", pd.Series(dtype=float))),
                float(len(qualifying_365)),
                driver_points,
                driver_position,
                driver_points_trend,
                nationality_match,
                constructor_points,
                constructor_position,
                constructor_points_trend,
                constructor_momentum,
                current_attrition,
                driver_circuit_rate,
                current_attrition - previous_attrition,
                travel,
                era_dnf,
                season_round,
                season_phase,
                *calendar,
            ]
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)

    def _add_relative(self, base: np.ndarray) -> np.ndarray:
        columns: list[np.ndarray] = []
        for name in RELATIVE_FEATURES:
            values = base[:, BASE_FEATURE_NAMES.index(name)].astype(np.float64)
            finite = np.isfinite(values)
            fill = float(np.nanmedian(values)) if finite.any() else 0.0
            clean = np.where(finite, values, fill)
            order = pd.Series(clean).rank(method="average", pct=True).to_numpy(dtype=np.float64)
            scale = float(clean.std())
            zscore = (clean - float(clean.mean())) / scale if scale > 1e-8 else np.zeros(len(clean), dtype=np.float64)
            columns.extend([order.astype(np.float32), zscore.astype(np.float32)])
        return np.column_stack([base, *columns]).astype(np.float32)

    def transform(self, seeds: pd.DataFrame) -> np.ndarray:
        output = np.empty((len(seeds), len(FEATURE_NAMES)), dtype=np.float32)
        for timestamp, frame in seeds.groupby("date", sort=True):
            canonical = frame.sort_values(["driverId", "_row_id"], kind="stable")
            positions = canonical.index.to_numpy(dtype=np.int64)
            drivers = canonical["driverId"].to_numpy(dtype=np.int64)
            path = self._origin_cache_path(pd.Timestamp(timestamp), drivers)
            matrix: np.ndarray | None = None
            if path is not None and path.exists():
                try:
                    stored = np.load(path, allow_pickle=False)
                    if np.array_equal(stored["driver_ids"], drivers):
                        matrix = stored["features"].astype(np.float32)
                except Exception:
                    matrix = None
            if matrix is None:
                matrix = self._add_relative(self._origin_base(canonical, pd.Timestamp(timestamp)))
                if path is not None:
                    temporary = path.with_suffix(f".{os.getpid()}.npz")
                    np.savez_compressed(temporary, driver_ids=drivers, features=matrix)
                    os.replace(temporary, path)
            output[positions] = matrix
        return output


def build_all_features(
    tables: dict[str, pd.DataFrame],
    frames: list[pd.DataFrame],
    cache_root: Path | None,
) -> list[np.ndarray]:
    builder = RelationalFeatureBuilder(tables, cache_root)
    outputs: list[np.ndarray] = []
    for frame in frames:
        indexed = frame.reset_index(drop=True).copy()
        outputs.append(builder.transform(indexed))
    return outputs
