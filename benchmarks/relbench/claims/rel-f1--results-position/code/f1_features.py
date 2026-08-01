from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


def _rename(frame: pd.DataFrame, prefix: str, keep: list[str]) -> pd.DataFrame:
    return frame.rename(columns={c: f"{prefix}_{c}" for c in frame.columns if c not in keep})


def _prior_numeric(frame: pd.DataFrame, key: str, columns: list[str], prefix: str) -> pd.DataFrame:
    ordered = frame.sort_values([key, "date", "raceId"], kind="stable").copy()
    grouped = ordered.groupby(key, sort=False)
    ordered[f"{prefix}_prior_count"] = grouped.cumcount().astype(float)
    count = ordered[f"{prefix}_prior_count"].replace(0, np.nan)
    for column in columns:
        values = pd.to_numeric(ordered[column], errors="coerce")
        valid = values.notna().astype(float)
        total = values.fillna(0).groupby(ordered[key], sort=False).cumsum() - values.fillna(0)
        seen = valid.groupby(ordered[key], sort=False).cumsum() - valid
        ordered[f"{prefix}_prior_{column}_mean"] = total / seen.replace(0, np.nan)
        ordered[f"{prefix}_prior_{column}_last"] = grouped[column].shift(1)
    return ordered


def _lag_table(
    frame: pd.DataFrame,
    key: str,
    columns: list[str],
    prefix: str,
    season_reset: bool = False,
) -> pd.DataFrame:
    ordered = frame.sort_values([key, "date", "raceId"], kind="stable").copy()
    grouped = ordered.groupby(key, sort=False)
    selected = ordered[["raceId", key, "date", *columns]].copy()
    previous_year = grouped["date"].shift(1).dt.year
    current_year = ordered["date"].dt.year
    for column in columns:
        selected[f"lag_{column}"] = grouped[column].shift(1)
        if season_reset:
            reset = previous_year != current_year
            selected.loc[reset, f"lag_{column}"] = 0.0 if column in ["points", "wins"] else np.nan
    return _rename(selected, prefix, ["raceId", key])


def _current_group_features(base: pd.DataFrame) -> pd.DataFrame:
    result = base.copy()
    race = result.groupby("raceId", sort=False)
    result["task_field_size"] = race["resultId"].transform("size").astype(float)
    rank_columns = {
        "grid": True,
        "qual_position": True,
        "driver_stand_position": True,
        "driver_stand_points": False,
        "driver_stand_delta_points": False,
        "constructor_stand_position": True,
        "constructor_stand_points": False,
        "constructor_result_points": False,
    }
    for column, ascending in rank_columns.items():
        if column not in result:
            continue
        values = result[column].copy()
        if column == "grid":
            values = values.where(values > 0, result["task_field_size"] + 1)
        result[f"race_rank_{column}"] = values.groupby(result["raceId"]).rank()
        result[f"race_fraction_{column}"] = (
            result[f"race_rank_{column}"] - 1
        ) / (result["task_field_size"] - 1).clip(lower=1)
        if not ascending:
            result[f"race_fraction_{column}"] = 1 - result[f"race_fraction_{column}"]
    team = result.groupby(["raceId", "constructorId"], sort=False)
    for column in [
        "grid",
        "qual_position",
        "driver_stand_position",
        "driver_stand_points",
        "driver_stand_delta_points",
    ]:
        if column not in result:
            continue
        result[f"team_{column}_mean"] = team[column].transform("mean")
        result[f"team_{column}_min"] = team[column].transform("min")
        result[f"team_{column}_max"] = team[column].transform("max")
        result[f"team_{column}_delta"] = result[column] - result[f"team_{column}_mean"]
        team_count = team[column].transform("count")
        teammate_mean = (team[column].transform("sum") - result[column]) / (team_count - 1)
        result[f"teammate_{column}_mean"] = teammate_mean.where(team_count > 1)
        result[f"teammate_{column}_delta"] = result[column] - result[f"teammate_{column}_mean"]
    result["task_team_size"] = team["resultId"].transform("size").astype(float)
    return result


def build_base_features(query: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    results = tables["results"].copy()
    races = tables["races"].copy()
    qualifying = tables["qualifying"].copy()
    standings = tables["standings"].copy()
    constructor_standings = tables["constructor_standings"].copy()
    constructor_results = tables["constructor_results"].copy()

    result_prior = _prior_numeric(results, "driverId", ["grid", "number"], "driver_result")
    result_columns = [
        "resultId",
        "raceId",
        "driverId",
        "constructorId",
        "number",
        "grid",
        "date",
        "driver_result_prior_count",
        "driver_result_prior_grid_mean",
        "driver_result_prior_grid_last",
        "driver_result_prior_number_mean",
        "driver_result_prior_number_last",
    ]
    result_prior = result_prior[result_columns].rename(columns={"date": "result_date"})

    race_result = results.groupby("raceId", sort=False).agg(
        result_field_size=("resultId", "size"),
        result_grid_mean=("grid", "mean"),
        result_grid_std=("grid", "std"),
        result_grid_max=("grid", "max"),
        result_number_mean=("number", "mean"),
    ).reset_index()
    team_result = results.groupby(["raceId", "constructorId"], sort=False).agg(
        result_team_size=("resultId", "size"),
        result_team_grid_mean=("grid", "mean"),
        result_team_grid_min=("grid", "min"),
        result_team_grid_max=("grid", "max"),
        result_team_number_mean=("number", "mean"),
    ).reset_index()
    constructor_races = team_result.merge(
        results[["raceId", "constructorId", "date"]].drop_duplicates(),
        on=["raceId", "constructorId"],
        how="left",
        validate="one_to_one",
    )
    constructor_prior = _prior_numeric(
        constructor_races,
        "constructorId",
        ["result_team_grid_mean", "result_team_size"],
        "constructor_result_history",
    )
    constructor_prior = constructor_prior[[
        "raceId",
        "constructorId",
        "constructor_result_history_prior_count",
        "constructor_result_history_prior_result_team_grid_mean_mean",
        "constructor_result_history_prior_result_team_grid_mean_last",
        "constructor_result_history_prior_result_team_size_mean",
        "constructor_result_history_prior_result_team_size_last",
    ]]

    qualifying_prior = _prior_numeric(
        qualifying,
        "driverId",
        ["position", "number"],
        "driver_qualifying",
    )
    qualifying_current = qualifying_prior[[
        "raceId",
        "driverId",
        "constructorId",
        "number",
        "position",
        "date",
        "driver_qualifying_prior_count",
        "driver_qualifying_prior_position_mean",
        "driver_qualifying_prior_position_last",
        "driver_qualifying_prior_number_mean",
        "driver_qualifying_prior_number_last",
    ]].rename(columns={
        "constructorId": "qual_constructorId",
        "number": "qual_number",
        "position": "qual_position",
        "date": "qual_date",
    })
    race_qualifying = qualifying.groupby("raceId", sort=False).agg(
        qualifying_field_size=("qualifyId", "size"),
        qualifying_position_max=("position", "max"),
    ).reset_index()
    team_qualifying = qualifying.groupby(["raceId", "constructorId"], sort=False).agg(
        qualifying_team_size=("qualifyId", "size"),
        qualifying_team_position_mean=("position", "mean"),
        qualifying_team_position_min=("position", "min"),
        qualifying_team_position_max=("position", "max"),
    ).reset_index()

    driver_stand = _lag_table(
        standings,
        "driverId",
        ["points", "position", "wins"],
        "driver_stand",
        True,
    )
    constructor_stand = _lag_table(
        constructor_standings,
        "constructorId",
        ["points", "position", "wins"],
        "constructor_stand",
        True,
    )
    constructor_result = _lag_table(
        constructor_results,
        "constructorId",
        ["points"],
        "constructor_result",
    )

    base = query.merge(result_prior, on="resultId", how="left", validate="one_to_one")
    base = base.merge(race_result, on="raceId", how="left", validate="many_to_one")
    base = base.merge(team_result, on=["raceId", "constructorId"], how="left", validate="many_to_one")
    base = base.merge(constructor_prior, on=["raceId", "constructorId"], how="left", validate="many_to_one")
    race_frame = _rename(races, "race", ["raceId", "circuitId"])
    base = base.merge(race_frame, on="raceId", how="left", validate="many_to_one")
    base = base.merge(tables["circuits"], on="circuitId", how="left", validate="many_to_one")
    base = base.merge(
        _rename(tables["drivers"], "driver", ["driverId"]),
        on="driverId",
        how="left",
        validate="many_to_one",
    )
    base = base.merge(
        _rename(tables["constructors"], "constructor", ["constructorId"]),
        on="constructorId",
        how="left",
        validate="many_to_one",
    )
    base = base.merge(qualifying_current, on=["raceId", "driverId"], how="left", validate="many_to_one")
    base = base.merge(race_qualifying, on="raceId", how="left", validate="many_to_one")
    base = base.merge(
        team_qualifying,
        on=["raceId", "constructorId"],
        how="left",
        validate="many_to_one",
    )
    base = base.merge(driver_stand, on=["raceId", "driverId"], how="left", validate="many_to_one")
    base = base.merge(
        constructor_stand,
        on=["raceId", "constructorId"],
        how="left",
        validate="many_to_one",
    )
    base = base.merge(
        constructor_result,
        on=["raceId", "constructorId"],
        how="left",
        validate="many_to_one",
    )

    temporal_columns = [
        "result_date",
        "race_date",
        "qual_date",
        "driver_stand_date",
        "constructor_stand_date",
        "constructor_result_date",
    ]
    for column in temporal_columns:
        observed = base[column].notna()
        if (base.loc[observed, column] > base.loc[observed, "date"]).any():
            raise RuntimeError(f"temporal feature violation in {column}")

    for prefix in ["driver_stand", "constructor_stand", "constructor_result"]:
        for column in ["points", "position", "wins"]:
            current = f"{prefix}_{column}"
            lag = f"{prefix}_lag_{column}"
            if current in base and lag in base:
                base[f"{prefix}_delta_{column}"] = base[current] - base[lag]

    base["seed_year"] = base["date"].dt.year.astype(float)
    base["seed_month"] = base["date"].dt.month.astype(float)
    base["seed_dayofyear"] = base["date"].dt.dayofyear.astype(float)
    base["seed_days"] = (base["date"] - pd.Timestamp("1950-01-01")).dt.days.astype(float)
    base["driver_age_days"] = (base["date"] - base["driver_dob"]).dt.days.astype(float)
    base["constructor_grid_delta"] = base["grid"] - base["result_team_grid_mean"]
    base["constructor_qualifying_delta"] = base["qual_position"] - base["qualifying_team_position_mean"]
    base["grid_qualifying_delta"] = base["grid"] - base["qual_position"]
    base["grid_all_result_fraction"] = base["grid"].where(
        base["grid"] > 0,
        base["result_field_size"] + 1,
    ) / base["result_field_size"].clip(lower=1)
    base["qualifying_field_fraction"] = base["qual_position"] / base["qualifying_field_size"].clip(lower=1)
    base["driver_constructor_nationality_match"] = (
        base["driver_nationality"] == base["constructor_nationality"]
    ).astype(float)
    base["circuit_constructor_nationality_match"] = (
        base["country"] == base["constructor_nationality"]
    ).astype(float)
    base["driver_constructor_pair"] = base["driverId"].astype(str) + "_" + base["constructorId"].astype(str)
    base["driver_circuit_pair"] = base["driverId"].astype(str) + "_" + base["circuitId"].astype(str)
    base["constructor_circuit_pair"] = base["constructorId"].astype(str) + "_" + base["circuitId"].astype(str)
    return _current_group_features(base)


def _state_features(state: dict | None, current_date: pd.Timestamp) -> tuple[float, ...]:
    if state is None:
        return (0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    count = state["count"]
    mean = state["sum"] / count
    variance = max(state["square"] / count - mean * mean, 0.0)
    recent = state["recent"]
    mean3 = float(np.mean(list(recent)[-3:]))
    mean5 = float(np.mean(list(recent)[-5:]))
    mean10 = float(np.mean(recent))
    return (
        float(count),
        mean,
        variance ** 0.5,
        state["minimum"],
        state["maximum"],
        state["last"],
        mean3,
        mean5,
        mean10,
        mean3 - mean10,
        state["wins"] / count,
        state["top3"] / count,
        state["norm_sum"] / count,
        state["norm_last"],
        float((current_date - state["last_date"]).days),
    )


def _history_for_key(
    base: pd.DataFrame,
    labels: pd.DataFrame,
    key_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    names = [
        "count",
        "mean",
        "std",
        "min",
        "max",
        "last",
        "mean3",
        "mean5",
        "mean10",
        "recent_delta",
        "win_rate",
        "top3_rate",
        "normalized_mean",
        "normalized_last",
        "days_since",
    ]
    values = np.full((len(base), len(names)), np.nan, dtype=float)
    values[:, 0] = 0.0
    states: dict[object, dict] = {}
    query_groups = {date: rows for date, rows in base.groupby("date", sort=True).groups.items()}
    label_groups = {date: rows for date, rows in labels.groupby("date", sort=True).groups.items()}
    dates = sorted(set(query_groups) | set(label_groups))
    for date in dates:
        for index in query_groups.get(date, []):
            key_values = tuple(base.loc[index, column] for column in key_columns)
            key = key_values[0] if len(key_values) == 1 else key_values
            values[index] = _state_features(states.get(key), date)
        for index in label_groups.get(date, []):
            key_values = tuple(labels.loc[index, column] for column in key_columns)
            key = key_values[0] if len(key_values) == 1 else key_values
            position = float(labels.loc[index, "position"])
            size = float(labels.loc[index, "label_field_size"])
            normalized = (position - 1) / max(size - 1, 1)
            if key not in states:
                states[key] = {
                    "count": 0,
                    "sum": 0.0,
                    "square": 0.0,
                    "minimum": position,
                    "maximum": position,
                    "last": position,
                    "recent": deque(maxlen=10),
                    "wins": 0,
                    "top3": 0,
                    "norm_sum": 0.0,
                    "norm_last": normalized,
                    "last_date": date,
                }
            state = states[key]
            state["count"] += 1
            state["sum"] += position
            state["square"] += position * position
            state["minimum"] = min(state["minimum"], position)
            state["maximum"] = max(state["maximum"], position)
            state["last"] = position
            state["recent"].append(position)
            state["wins"] += int(position == 1)
            state["top3"] += int(position <= 3)
            state["norm_sum"] += normalized
            state["norm_last"] = normalized
            state["last_date"] = date
    return pd.DataFrame(values, columns=[f"{prefix}_{name}" for name in names], index=base.index)


def build_label_history(base: pd.DataFrame, allowed_labels: pd.DataFrame) -> pd.DataFrame:
    label_keys = base[[
        "resultId",
        "raceId",
        "driverId",
        "constructorId",
        "circuitId",
        "grid",
    ]].drop_duplicates("resultId")
    labels = allowed_labels.merge(label_keys, on="resultId", how="left", validate="one_to_one")
    labels["label_field_size"] = labels.groupby("raceId")["resultId"].transform("size").astype(float)
    groups = [
        (["driverId"], "driver_target_history"),
        (["constructorId"], "constructor_target_history"),
        (["driverId", "circuitId"], "driver_circuit_target_history"),
        (["constructorId", "circuitId"], "constructor_circuit_target_history"),
        (["circuitId"], "circuit_target_history"),
        (["driverId", "constructorId"], "driver_constructor_target_history"),
        (["grid"], "grid_target_history"),
    ]
    frames = [_history_for_key(base, labels, keys, prefix) for keys, prefix in groups]
    return pd.concat(frames, axis=1)


def make_model_frame(base: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.concat([base, history], axis=1)
    excluded = {
        "_query_id",
        "_row_idx",
        "_split",
        "position",
        "date",
        "resultId",
        "raceId",
        "result_date",
        "race_date",
        "qual_date",
        "driver_stand_date",
        "constructor_stand_date",
        "constructor_result_date",
        "driver_dob",
    }
    frame = frame[[column for column in frame.columns if column not in excluded]].copy()
    categorical = [
        "driverId",
        "constructorId",
        "circuitId",
        "driver_driverRef",
        "driver_code",
        "driver_forename",
        "driver_surname",
        "driver_nationality",
        "constructor_constructorRef",
        "constructor_name",
        "constructor_nationality",
        "circuitRef",
        "name",
        "location",
        "country",
        "race_name",
        "race_time",
        "driver_constructor_pair",
        "driver_circuit_pair",
        "constructor_circuit_pair",
    ]
    categorical = [column for column in categorical if column in frame]
    for column in categorical:
        frame[column] = frame[column].astype("string").fillna("__missing__").astype("category")
    for column in frame.columns:
        if column in categorical:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    missing_columns = [column for column in frame.columns if column not in categorical and frame[column].isna().any()]
    missing = frame[missing_columns].isna().astype(np.int8).rename(
        columns={column: f"missing_{column}" for column in missing_columns}
    )
    frame[missing_columns] = frame[missing_columns].fillna(-999.0)
    frame = pd.concat([frame, missing], axis=1).copy()
    return frame, categorical
