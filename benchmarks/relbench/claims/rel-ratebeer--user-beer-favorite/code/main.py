from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def paths() -> tuple[Path, Path]:
    base = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"]
    return base / "db", base / "tasks" / os.environ["RELBENCH_TASK"]


def connect() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={int(os.environ.get('OMP_NUM_THREADS', '1'))}")
    connection.execute("PRAGMA enable_progress_bar=false")
    connection.execute("PRAGMA preserve_insertion_order=false")
    return connection


def popularity(connection: duckdb.DuckDBPyConnection, db_dir: Path, origin: pd.Timestamp, limit: int = 5000) -> pd.DataFrame:
    ratings = db_dir / "beer_ratings.parquet"
    favorites = db_dir / "favorites.parquet"
    query = f"""
        WITH r AS (
            SELECT beer_id,
                sum(exp(-ln(2) * epoch(TIMESTAMP '{origin}' - created_at) / 86400 / 14)) r14,
                sum(exp(-ln(2) * epoch(TIMESTAMP '{origin}' - created_at) / 86400 / 45)) r45,
                sum(exp(-ln(2) * epoch(TIMESTAMP '{origin}' - created_at) / 86400 / 120)) r120
            FROM read_parquet('{ratings}')
            WHERE created_at <= TIMESTAMP '{origin}' AND beer_id IS NOT NULL
            GROUP BY beer_id
        ), f AS (
            SELECT beer_id,
                sum(exp(-ln(2) * epoch(TIMESTAMP '{origin}' - created_at) / 86400 / 14)) f14,
                sum(exp(-ln(2) * epoch(TIMESTAMP '{origin}' - created_at) / 86400 / 45)) f45,
                sum(exp(-ln(2) * epoch(TIMESTAMP '{origin}' - created_at) / 86400 / 120)) f120
            FROM read_parquet('{favorites}')
            WHERE created_at <= TIMESTAMP '{origin}' AND beer_id IS NOT NULL
            GROUP BY beer_id
        )
        SELECT coalesce(r.beer_id, f.beer_id) beer_id,
            coalesce(r14, 0) r14, coalesce(r45, 0) r45, coalesce(r120, 0) r120,
            coalesce(f14, 0) f14, coalesce(f45, 0) f45, coalesce(f120, 0) f120,
            2.5 * ln(1 + coalesce(f14, 0)) + 1.5 * ln(1 + coalesce(f45, 0)) +
            ln(1 + coalesce(f120, 0)) + 0.25 * ln(1 + coalesce(r14, 0)) +
            0.15 * ln(1 + coalesce(r45, 0)) + 0.1 * ln(1 + coalesce(r120, 0)) score
        FROM r FULL OUTER JOIN f USING (beer_id)
        ORDER BY score DESC, beer_id
        LIMIT {limit}
    """
    return connection.execute(query).fetchdf()


def user_histories(connection: duckdb.DuckDBPyConnection, db_dir: Path, users: np.ndarray, origin: pd.Timestamp) -> tuple[dict[int, list[tuple[int, float, float, int]]], dict[int, set[int]]]:
    user_frame = pd.DataFrame({"user_id": np.asarray(users, dtype=np.int64)})
    connection.register("seed_users", user_frame)
    ratings = db_dir / "beer_ratings.parquet"
    favorites = db_dir / "favorites.parquet"
    rating_frame = connection.execute(f"""
        SELECT user_id, beer_id, max(total_score) own_best, avg(total_score) own_mean,
            max(created_at) own_last, count(*) own_count
        FROM read_parquet('{ratings}')
        SEMI JOIN seed_users USING (user_id)
        WHERE created_at <= TIMESTAMP '{origin}' AND beer_id IS NOT NULL
        GROUP BY user_id, beer_id
        QUALIFY row_number() OVER (
            PARTITION BY user_id
            ORDER BY max(total_score) DESC, max(created_at) DESC, beer_id
        ) <= 500
    """).fetchdf()
    favorite_frame = connection.execute(f"""
        SELECT user_id, beer_id
        FROM read_parquet('{favorites}')
        SEMI JOIN seed_users USING (user_id)
        WHERE created_at <= TIMESTAMP '{origin}' AND beer_id IS NOT NULL
    """).fetchdf()
    connection.unregister("seed_users")
    rating_map: dict[int, list[tuple[int, float, float, int]]] = defaultdict(list)
    origin_value = pd.Timestamp(origin).value
    for row in rating_frame.itertuples(index=False):
        age = max(0, int((origin_value - pd.Timestamp(row.own_last).value) / 86_400_000_000_000))
        rating_map[int(row.user_id)].append((int(row.beer_id), float(row.own_best), float(row.own_mean), age))
    favorite_map: dict[int, set[int]] = defaultdict(set)
    for row in favorite_frame.itertuples(index=False):
        favorite_map[int(row.user_id)].add(int(row.beer_id))
    return rating_map, favorite_map


def floor_predictions(connection: duckdb.DuckDBPyConnection, db_dir: Path, table: pd.DataFrame) -> np.ndarray:
    origin = pd.Timestamp(table["timestamp"].iloc[0])
    users = table["user_id"].to_numpy(np.int64)
    popular = popularity(connection, db_dir, origin).dropna(subset=["beer_id"])
    popular_ids = popular["beer_id"].to_numpy(np.int64)
    popular_scores = popular["score"].to_numpy(np.float64)
    scale = max(float(popular_scores[0] - popular_scores[min(len(popular_scores) - 1, 1999)]), 1e-6)
    pop_map = {int(beer): float((score - popular_scores[min(len(popular_scores) - 1, 1999)]) / scale) for beer, score in zip(popular_ids, popular_scores)}
    rating_map, favorite_map = user_histories(connection, db_dir, users, origin)
    result = np.empty((len(table), 10), dtype=np.int64)
    for index, user in enumerate(users):
        excluded = favorite_map[int(user)]
        scores: dict[int, float] = {}
        for beer in popular_ids[:2500]:
            item = int(beer)
            if item not in excluded:
                scores[item] = pop_map[item]
        for beer, best, mean, age in rating_map[int(user)]:
            if beer in excluded:
                continue
            own = 0.9 + 0.55 * max(0.0, best - 3.5) + 0.2 * max(0.0, mean - 3.5) + 0.35 * math.exp(-math.log(2) * age / 45)
            scores[beer] = scores.get(beer, 0.0) + own
        ranked = sorted(scores, key=lambda beer: (-scores[beer], beer))
        if len(ranked) < 10:
            for beer in range(751_524):
                if beer not in excluded and beer not in scores:
                    ranked.append(beer)
                    if len(ranked) == 10:
                        break
        result[index] = ranked[:10]
    return result


def save_predictions(val: np.ndarray, test: np.ndarray, started: float, diagnostics: dict) -> None:
    output = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "output_data_generic_exp_0"))
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "val_predictions.npy", np.asarray(val, dtype=np.int64))
    np.save(output / "test_predictions.npy", np.asarray(test, dtype=np.int64))
    diagnostics["elapsed_seconds"] = time.time() - started
    (output / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"[pipeline] saved val{val.shape} test{test.shape} elapsed={time.time() - started:.1f}s")


def main() -> None:
    started = time.time()
    db_dir, task_dir = paths()
    val = pd.read_parquet(task_dir / "val.parquet", columns=["timestamp", "user_id"])
    test = pd.read_parquet(task_dir / "test.parquet", columns=["timestamp", "user_id"])
    connection = connect()
    val_predictions = floor_predictions(connection, db_dir, val)
    print(f"[pipeline] deterministic validation floor elapsed={time.time() - started:.1f}s")
    test_predictions = floor_predictions(connection, db_dir, test)
    print(f"[pipeline] deterministic test floor elapsed={time.time() - started:.1f}s")
    save_predictions(val_predictions, test_predictions, started, {"mode": "deterministic_floor"})
    from temporal_ranker import run_temporal_ranker
    debug = "--debug" in sys.argv
    val_predictions, test_predictions, diagnostics = run_temporal_ranker(
        connection, db_dir, task_dir, val, test, val_predictions, test_predictions, debug
    )
    diagnostics["mode"] = "debug" if debug else "full"
    save_predictions(val_predictions, test_predictions, started, diagnostics)


if __name__ == "__main__":
    main()
