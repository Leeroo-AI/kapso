# Step 1: build driver->circuit interactions and race history (cached)
from __future__ import annotations

from pathlib import Path

import pandas as pd

INTERACTIONS_VERSION = "v1"


def build_interactions(db, cache_dir: Path, version: str) -> pd.DataFrame:
    path = Path(cache_dir) / f"driver_circuit_interactions_{version}.pkl"
    if path.exists():
        return pd.read_pickle(path)

    results = db.table_dict["results"].df
    races = db.table_dict["races"].df
    inter = results[["raceId", "driverId"]].merge(
        races[["raceId", "circuitId", "date"]], on="raceId", how="left"
    )
    inter = inter.dropna(subset=["driverId", "circuitId", "date"])
    inter = inter[["driverId", "circuitId", "date"]].astype(
        {"driverId": "int64", "circuitId": "int64"}
    )
    inter = inter.reset_index(drop=True)
    inter.to_pickle(path)
    return inter


def build_races_history(db) -> pd.DataFrame:
    races = db.table_dict["races"].df
    rh = races[["circuitId", "year", "date"]].dropna(subset=["circuitId", "date"])
    return rh.astype({"circuitId": "int64", "year": "int64"}).reset_index(drop=True)
