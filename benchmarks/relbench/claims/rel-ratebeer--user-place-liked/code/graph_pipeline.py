from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")


SEED = 20260801
HALF_LIFE = 730.0
RESTART = 0.15
EARTH_RADIUS = 6371.0088


@dataclass
class BuildConfig:
    debug: bool
    factors: int
    als_iterations: int
    ppr_iterations: int
    ranker_rounds: int
    train_per_origin: int
    candidate_cap: int
    edge_fraction: float

    @classmethod
    def create(cls, debug: bool) -> "BuildConfig":
        if debug:
            return cls(True, 32, 2, 5, 30, 20, 600, 0.05)
        return cls(False, 96, 15, 15, 900, 120, 600, 1.0)


@dataclass
class Factors:
    users: np.ndarray
    items: np.ndarray


@dataclass
class PPRResult:
    ids: np.ndarray
    scores: np.ndarray


@dataclass
class Snapshot:
    cutoff: pd.Timestamp
    key: str
    graph: sp.csr_matrix
    taste_graph: sp.csr_matrix
    up: sp.csr_matrix
    ub: sp.csr_matrix
    us: sp.csr_matrix
    place_life_count: np.ndarray
    place_365_count: np.ndarray
    place_90_count: np.ndarray
    place_life_liked: np.ndarray
    place_365_liked: np.ndarray
    place_score_sum: np.ndarray
    place_score_count: np.ndarray
    user_place_count: np.ndarray
    user_365_place_count: np.ndarray
    user_90_place_count: np.ndarray
    user_beer_count: np.ndarray
    user_365_beer_count: np.ndarray
    user_90_beer_count: np.ndarray
    user_centroid_lat: np.ndarray
    user_centroid_lon: np.ndarray
    user_last_state: np.ndarray
    user_last_city: np.ndarray
    user_last_h3: np.ndarray
    user_last_place: np.ndarray
    place_factors: Factors
    brewer_factors: Factors


def elapsed(start: float) -> str:
    return f"{time.time() - start:.1f}s"


def normalize_name(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def timestamp_key(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def top_indices(values: np.ndarray, k: int) -> np.ndarray:
    if values.size <= k:
        return np.argsort(-values, kind="stable")
    part = np.argpartition(values, -k)[-k:]
    return part[np.argsort(-values[part], kind="stable")]


def haversine(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    if not np.isfinite(lat1) or not np.isfinite(lon1):
        return np.full(lat2.shape, 20000.0, dtype=np.float32)
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return (2.0 * EARTH_RADIUS * np.arcsin(np.minimum(1.0, np.sqrt(a)))).astype(np.float32)


class StaticData:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.db_dir = Path(os.environ["RELBENCH_CACHE_DIR"]) / "rel-ratebeer" / "db"
        self.task_dir = Path(os.environ["RELBENCH_CACHE_DIR"]) / "rel-ratebeer" / "tasks" / "user-place-liked"
        self.con = duckdb.connect()
        self.con.execute(f"PRAGMA threads={int(os.environ.get('OMP_NUM_THREADS', '11'))}")
        self.con.execute("PRAGMA enable_progress_bar=false")
        self.n_users = 218680
        self.n_places = 86061
        self.n_brewers = 50013
        self.n_styles = 157
        self.n_states = 622
        self.n_countries = 251
        self.n_types = 8
        self._load_places()
        self._load_brewers()
        self._load_brewer_styles()
        self._make_nodes()

    def parquet(self, name: str) -> str:
        return str(self.db_dir / f"{name}.parquet")

    def _load_places(self) -> None:
        query = f"""
            SELECT place_id, state_id, country_id, type_id, city_plain, name_plain,
                   latitude, longitude, try_cast(created_at AS TIMESTAMP) created_at
            FROM read_parquet('{self.parquet('places')}') ORDER BY place_id
        """
        frame = self.con.execute(query).fetchdf()
        self.place_state = frame.state_id.fillna(-1).to_numpy(np.int32)
        self.place_country = frame.country_id.fillna(-1).to_numpy(np.int32)
        self.place_type = frame.type_id.fillna(-1).to_numpy(np.int16)
        self.place_city_text = frame.city_plain.fillna("").astype(str).str.lower().to_numpy()
        self.place_name = np.array([normalize_name(x) for x in frame.name_plain], dtype=object)
        self.place_lat = frame.latitude.to_numpy(np.float32)
        self.place_lon = frame.longitude.to_numpy(np.float32)
        self.place_created = frame.created_at.to_numpy(dtype="datetime64[ns]")
        try:
            import h3

            cells = [h3.latlng_to_cell(float(a), float(b), 4) if np.isfinite(a) and np.isfinite(b) else "" for a, b in zip(self.place_lat, self.place_lon)]
        except Exception:
            cells = [f"{int((float(a) + 90) // 2)}:{int((float(b) + 180) // 2)}" if np.isfinite(a) and np.isfinite(b) else "" for a, b in zip(self.place_lat, self.place_lon)]
        unique = {cell: i for i, cell in enumerate(sorted(set(cells) - {""}))}
        self.place_h3 = np.array([unique.get(cell, -1) for cell in cells], dtype=np.int32)
        self.n_h3 = len(unique)
        self.name_to_places: dict[str, list[int]] = {}
        for place, name in enumerate(self.place_name):
            if name:
                self.name_to_places.setdefault(name, []).append(place)

    def _load_brewers(self) -> None:
        query = f"""
            SELECT brewer_id, state_id, country_id, name_plain, city, score,
                   view_count, is_out_of_business, is_retired
            FROM read_parquet('{self.parquet('brewers')}') ORDER BY brewer_id
        """
        frame = self.con.execute(query).fetchdf()
        self.brewer_state = frame.state_id.fillna(-1).to_numpy(np.int32)
        self.brewer_country = frame.country_id.fillna(-1).to_numpy(np.int32)
        self.brewer_name = np.array([normalize_name(x) for x in frame.name_plain], dtype=object)
        self.brewer_score = frame.score.fillna(0).to_numpy(np.float32)
        self.brewer_views = frame.view_count.fillna(0).to_numpy(np.float32)
        self.brewer_closed = (frame.is_out_of_business.fillna(False) | frame.is_retired.fillna(False)).to_numpy(np.int8)
        self.name_to_brewers: dict[str, list[int]] = {}
        for brewer, name in enumerate(self.brewer_name):
            if name:
                self.name_to_brewers.setdefault(name, []).append(brewer)
        pairs_b: list[int] = []
        pairs_p: list[int] = []
        for name, brewers in self.name_to_brewers.items():
            places = self.name_to_places.get(name)
            if places:
                for brewer in brewers:
                    for place in places:
                        pairs_b.append(brewer)
                        pairs_p.append(place)
        self.name_link_brewer = np.asarray(pairs_b, dtype=np.int32)
        self.name_link_place = np.asarray(pairs_p, dtype=np.int32)
        self.place_name_brewers: list[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(self.n_places)]
        for place in np.unique(self.name_link_place):
            self.place_name_brewers[int(place)] = self.name_link_brewer[self.name_link_place == place]

    def _load_brewer_styles(self) -> None:
        frame = self.con.execute(
            f"SELECT DISTINCT brewer_id, style_id FROM read_parquet('{self.parquet('beers')}') WHERE brewer_id IS NOT NULL AND style_id IS NOT NULL"
        ).fetchdf()
        self.brewer_style_b = frame.brewer_id.to_numpy(np.int32)
        self.brewer_style_s = frame.style_id.to_numpy(np.int32)

    def _make_nodes(self) -> None:
        self.user_offset = 0
        self.place_offset = self.n_users
        self.brewer_offset = self.place_offset + self.n_places
        self.style_offset = self.brewer_offset + self.n_brewers
        self.state_offset = self.style_offset + self.n_styles
        self.country_offset = self.state_offset + self.n_states
        self.type_offset = self.country_offset + self.n_countries
        self.h3_offset = self.type_offset + self.n_types
        self.n_nodes = self.h3_offset + self.n_h3

    def load_tasks(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = self.con.execute(f"SELECT * FROM read_parquet('{self.task_dir / 'train.parquet'}') ORDER BY timestamp").fetchdf()
        val = self.con.execute(f"SELECT * FROM read_parquet('{self.task_dir / 'val.parquet'}')").fetchdf()
        test = self.con.execute(f"SELECT * FROM read_parquet('{self.task_dir / 'test.parquet'}')").fetchdf()
        return train, val, test


class SnapshotBuilder:
    def __init__(self, data: StaticData, config: BuildConfig):
        self.data = data
        self.config = config
        self.cache = data.cache_dir / ("lane2_graphdiff_debug_v3" if config.debug else "lane2_graphdiff_full_v3")
        self.cache.mkdir(parents=True, exist_ok=True)

    def _edge_filter(self, id_col: str) -> str:
        if self.config.edge_fraction >= 1.0:
            return ""
        modulus = max(1, int(round(1.0 / self.config.edge_fraction)))
        return f" AND {id_col} % {modulus} = 0"

    def _load_interactions(self, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        d = self.data
        t = cutoff.strftime("%Y-%m-%d %H:%M:%S")
        pr = d.con.execute(f"""
            SELECT rating_id, user_id, place_id, total_score, created_at,
                   exp(-greatest(0, date_diff('day', created_at, timestamp '{t}')) * ln(2) / {HALF_LIFE})
                   * (1 + cast(coalesce(total_score >= 80, false) AS INTEGER)) weight
            FROM read_parquet('{d.parquet('place_ratings')}')
            WHERE created_at <= timestamp '{t}' AND place_id IS NOT NULL
            {self._edge_filter('rating_id')}
        """).fetchdf()
        ub = d.con.execute(f"""
            WITH edges AS (
                SELECT br.user_id, b.brewer_id item_id,
                       exp(-greatest(0, date_diff('day', br.created_at, timestamp '{t}')) * ln(2) / {HALF_LIFE})
                       * (1 + cast(coalesce(br.total_score >= 80, false) AS INTEGER)) weight
                FROM read_parquet('{d.parquet('beer_ratings')}') br
                JOIN read_parquet('{d.parquet('beers')}') b USING (beer_id)
                WHERE br.created_at IS NOT NULL AND br.created_at <= timestamp '{t}' AND b.brewer_id IS NOT NULL
                {self._edge_filter('br.rating_id')}
                UNION ALL
                SELECT f.user_id, b.brewer_id item_id,
                       2.0 * exp(-greatest(0, date_diff('day', f.created_at, timestamp '{t}')) * ln(2) / {HALF_LIFE}) weight
                FROM read_parquet('{d.parquet('favorites')}') f
                JOIN read_parquet('{d.parquet('beers')}') b USING (beer_id)
                WHERE f.created_at IS NOT NULL AND f.created_at <= timestamp '{t}' AND b.brewer_id IS NOT NULL
                {self._edge_filter('f.favorite_id')}
            )
            SELECT user_id, item_id, ln(1 + sum(weight)) weight FROM edges GROUP BY user_id, item_id
        """).fetchdf()
        us = d.con.execute(f"""
            WITH edges AS (
                SELECT br.user_id, b.style_id item_id,
                       exp(-greatest(0, date_diff('day', br.created_at, timestamp '{t}')) * ln(2) / {HALF_LIFE})
                       * (1 + cast(coalesce(br.total_score >= 80, false) AS INTEGER)) weight
                FROM read_parquet('{d.parquet('beer_ratings')}') br
                JOIN read_parquet('{d.parquet('beers')}') b USING (beer_id)
                WHERE br.created_at IS NOT NULL AND br.created_at <= timestamp '{t}' AND b.style_id IS NOT NULL
                {self._edge_filter('br.rating_id')}
                UNION ALL
                SELECT f.user_id, b.style_id item_id,
                       2.0 * exp(-greatest(0, date_diff('day', f.created_at, timestamp '{t}')) * ln(2) / {HALF_LIFE}) weight
                FROM read_parquet('{d.parquet('favorites')}') f
                JOIN read_parquet('{d.parquet('beers')}') b USING (beer_id)
                WHERE f.created_at IS NOT NULL AND f.created_at <= timestamp '{t}' AND b.style_id IS NOT NULL
                {self._edge_filter('f.favorite_id')}
            )
            SELECT user_id, item_id, ln(1 + sum(weight)) weight FROM edges GROUP BY user_id, item_id
        """).fetchdf()
        beer_counts = d.con.execute(f"""
            SELECT user_id, count(*) life,
                   sum(created_at > timestamp '{t}' - interval '365 days') d365,
                   sum(created_at > timestamp '{t}' - interval '90 days') d90
            FROM read_parquet('{d.parquet('beer_ratings')}')
            WHERE created_at <= timestamp '{t}' {self._edge_filter('rating_id')}
            GROUP BY user_id
        """).fetchdf()
        return pr, ub, us, beer_counts

    def _interaction_matrices(self, pr: pd.DataFrame, ub: pd.DataFrame, us: pd.DataFrame) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        d = self.data
        up = sp.csr_matrix(
            ((1.0 + (pr.total_score.to_numpy() >= 80)).astype(np.float32), (pr.user_id.to_numpy(), pr.place_id.to_numpy())),
            shape=(d.n_users, d.n_places),
            dtype=np.float32,
        )
        ubm = sp.csr_matrix(
            (ub.weight.to_numpy(np.float32), (ub.user_id.to_numpy(), ub.item_id.to_numpy())),
            shape=(d.n_users, d.n_brewers),
            dtype=np.float32,
        )
        usm = sp.csr_matrix(
            (us.weight.to_numpy(np.float32), (us.user_id.to_numpy(), us.item_id.to_numpy())),
            shape=(d.n_users, d.n_styles),
            dtype=np.float32,
        )
        return up, ubm, usm

    def _append_edges(self, rows: list[np.ndarray], cols: list[np.ndarray], vals: list[np.ndarray], left: np.ndarray, right: np.ndarray, weight: np.ndarray | float) -> None:
        if left.size == 0:
            return
        w = np.full(left.size, weight, dtype=np.float32) if np.isscalar(weight) else np.asarray(weight, dtype=np.float32)
        rows.extend((left.astype(np.int32, copy=False), right.astype(np.int32, copy=False)))
        cols.extend((right.astype(np.int32, copy=False), left.astype(np.int32, copy=False)))
        vals.extend((w, w))

    def _graphs(self, cutoff: pd.Timestamp, pr: pd.DataFrame, ub: pd.DataFrame, us: pd.DataFrame) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        d = self.data
        full_r: list[np.ndarray] = []
        full_c: list[np.ndarray] = []
        full_v: list[np.ndarray] = []
        taste_r: list[np.ndarray] = []
        taste_c: list[np.ndarray] = []
        taste_v: list[np.ndarray] = []
        u = pr.user_id.to_numpy(np.int32) + d.user_offset
        p = pr.place_id.to_numpy(np.int32) + d.place_offset
        self._append_edges(full_r, full_c, full_v, u, p, pr.weight.to_numpy(np.float32))
        u = ub.user_id.to_numpy(np.int32) + d.user_offset
        b = ub.item_id.to_numpy(np.int32) + d.brewer_offset
        self._append_edges(full_r, full_c, full_v, u, b, ub.weight.to_numpy(np.float32) * 0.75)
        self._append_edges(taste_r, taste_c, taste_v, u, b, ub.weight.to_numpy(np.float32))
        u = us.user_id.to_numpy(np.int32) + d.user_offset
        s = us.item_id.to_numpy(np.int32) + d.style_offset
        self._append_edges(full_r, full_c, full_v, u, s, us.weight.to_numpy(np.float32) * 0.35)
        self._append_edges(taste_r, taste_c, taste_v, u, s, us.weight.to_numpy(np.float32) * 0.8)
        eligible = np.flatnonzero(d.place_created <= np.datetime64(cutoff))
        place_nodes = eligible.astype(np.int32) + d.place_offset
        for offset, values, weight in (
            (d.state_offset, d.place_state, 1.4),
            (d.country_offset, d.place_country, 0.35),
            (d.type_offset, d.place_type, 0.25),
            (d.h3_offset, d.place_h3, 1.8),
        ):
            chosen = eligible[values[eligible] >= 0]
            left = chosen.astype(np.int32) + d.place_offset
            right = values[chosen].astype(np.int32) + offset
            self._append_edges(full_r, full_c, full_v, left, right, weight)
            if offset != d.type_offset:
                self._append_edges(taste_r, taste_c, taste_v, left, right, weight)
        brewers = np.arange(d.n_brewers, dtype=np.int32)
        for offset, values, weight in (
            (d.state_offset, d.brewer_state, 1.3),
            (d.country_offset, d.brewer_country, 0.3),
        ):
            chosen = brewers[values >= 0]
            left = chosen + d.brewer_offset
            right = values[chosen].astype(np.int32) + offset
            self._append_edges(full_r, full_c, full_v, left, right, weight)
            self._append_edges(taste_r, taste_c, taste_v, left, right, weight)
        left = d.brewer_style_b + d.brewer_offset
        right = d.brewer_style_s + d.style_offset
        self._append_edges(full_r, full_c, full_v, left, right, 0.4)
        self._append_edges(taste_r, taste_c, taste_v, left, right, 0.7)
        mask = d.place_created[d.name_link_place] <= np.datetime64(cutoff)
        left = d.name_link_brewer[mask] + d.brewer_offset
        right = d.name_link_place[mask] + d.place_offset
        self._append_edges(full_r, full_c, full_v, left, right, 3.0)
        self._append_edges(taste_r, taste_c, taste_v, left, right, 4.0)
        full = self._normalize_graph(full_r, full_c, full_v, d.n_nodes)
        taste = self._normalize_graph(taste_r, taste_c, taste_v, d.n_nodes)
        return full, taste

    def _normalize_graph(self, rows: list[np.ndarray], cols: list[np.ndarray], vals: list[np.ndarray], shape: int) -> sp.csr_matrix:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        val = np.concatenate(vals).astype(np.float32, copy=False)
        graph = sp.csr_matrix((val, (row, col)), shape=(shape, shape), dtype=np.float32)
        degree = np.asarray(graph.sum(axis=1)).ravel()
        inv = np.zeros_like(degree, dtype=np.float32)
        np.divide(1.0, degree, out=inv, where=degree > 0)
        graph = sp.diags(inv, format="csr") @ graph
        graph.sort_indices()
        return graph.astype(np.float32)

    def _dynamic_features(self, cutoff: pd.Timestamp, pr: pd.DataFrame, beer_counts: pd.DataFrame) -> dict[str, np.ndarray]:
        d = self.data
        now = np.datetime64(cutoff)
        dates = pr.created_at.to_numpy(dtype="datetime64[ns]")
        place = pr.place_id.to_numpy(np.int32)
        user = pr.user_id.to_numpy(np.int32)
        liked = (pr.total_score.to_numpy() >= 80).astype(np.float32)
        scores = pr.total_score.to_numpy(np.float32)
        d90 = dates > now - np.timedelta64(90, "D")
        d365 = dates > now - np.timedelta64(365, "D")
        result: dict[str, np.ndarray] = {}
        result["place_life_count"] = np.bincount(place, minlength=d.n_places).astype(np.float32)
        result["place_365_count"] = np.bincount(place[d365], minlength=d.n_places).astype(np.float32)
        result["place_90_count"] = np.bincount(place[d90], minlength=d.n_places).astype(np.float32)
        result["place_life_liked"] = np.bincount(place, weights=liked, minlength=d.n_places).astype(np.float32)
        result["place_365_liked"] = np.bincount(place[d365], weights=liked[d365], minlength=d.n_places).astype(np.float32)
        result["place_score_sum"] = np.bincount(place, weights=scores, minlength=d.n_places).astype(np.float32)
        result["place_score_count"] = result["place_life_count"].copy()
        result["user_place_count"] = np.bincount(user, minlength=d.n_users).astype(np.float32)
        result["user_365_place_count"] = np.bincount(user[d365], minlength=d.n_users).astype(np.float32)
        result["user_90_place_count"] = np.bincount(user[d90], minlength=d.n_users).astype(np.float32)
        for name in ("user_beer_count", "user_365_beer_count", "user_90_beer_count"):
            result[name] = np.zeros(d.n_users, dtype=np.float32)
        if len(beer_counts):
            ids = beer_counts.user_id.to_numpy(np.int32)
            result["user_beer_count"][ids] = beer_counts.life.to_numpy(np.float32)
            result["user_365_beer_count"][ids] = beer_counts.d365.to_numpy(np.float32)
            result["user_90_beer_count"][ids] = beer_counts.d90.to_numpy(np.float32)
        lat = d.place_lat[place]
        lon = d.place_lon[place]
        valid = np.isfinite(lat) & np.isfinite(lon)
        count = np.bincount(user[valid], minlength=d.n_users).astype(np.float32)
        lat_sum = np.bincount(user[valid], weights=lat[valid], minlength=d.n_users)
        lon_sum = np.bincount(user[valid], weights=lon[valid], minlength=d.n_users)
        centroid_lat = np.full(d.n_users, np.nan, dtype=np.float32)
        centroid_lon = np.full(d.n_users, np.nan, dtype=np.float32)
        np.divide(lat_sum, count, out=centroid_lat, where=count > 0)
        np.divide(lon_sum, count, out=centroid_lon, where=count > 0)
        result["user_centroid_lat"] = centroid_lat
        result["user_centroid_lon"] = centroid_lon
        order = np.argsort(dates, kind="stable")
        reverse = order[::-1]
        _, first = np.unique(user[reverse], return_index=True)
        last_rows = reverse[first]
        last_user = user[last_rows]
        for name, values in (
            ("user_last_state", d.place_state),
            ("user_last_city", self._city_codes()),
            ("user_last_h3", d.place_h3),
        ):
            array = np.full(d.n_users, -1, dtype=np.int32)
            array[last_user] = values[place[last_rows]]
            result[name] = array
        last_place = np.full(d.n_users, -1, dtype=np.int32)
        last_place[last_user] = place[last_rows]
        result["user_last_place"] = last_place
        return result

    def _city_codes(self) -> np.ndarray:
        if not hasattr(self, "city_codes"):
            names = sorted(set(self.data.place_city_text) - {""})
            mapping = {x: i for i, x in enumerate(names)}
            self.city_codes = np.array([mapping.get(x, -1) for x in self.data.place_city_text], dtype=np.int32)
        return self.city_codes

    def _factor_paths(self, key: str) -> tuple[Path, Path]:
        return self.cache / f"als_up_{key}.npz", self.cache / f"als_ub_{key}.npz"

    def _fit_factors(self, matrix: sp.csr_matrix, factors: int, path: Path) -> Factors:
        if path.exists():
            stored = np.load(path, allow_pickle=False)
            return Factors(stored["users"], stored["items"])
        from implicit.als import AlternatingLeastSquares

        model = AlternatingLeastSquares(
            factors=factors,
            regularization=0.03,
            iterations=self.config.als_iterations,
            random_state=SEED,
            use_gpu=False,
            num_threads=int(os.environ.get("OMP_NUM_THREADS", "11")),
        )
        model.fit(matrix, show_progress=False)
        result = Factors(np.asarray(model.user_factors, dtype=np.float32), np.asarray(model.item_factors, dtype=np.float32))
        np.savez_compressed(path, users=result.users, items=result.items)
        return result

    def build(self, cutoff: pd.Timestamp, build_graph: bool = True) -> Snapshot:
        start = time.time()
        key = timestamp_key(cutoff)
        pr, ub, us, beer_counts = self._load_interactions(cutoff)
        print(f"[snapshot {key}] interactions place={len(pr)} brewer={len(ub)} style={len(us)} {elapsed(start)}")
        up, ubm, usm = self._interaction_matrices(pr, ub, us)
        up_path, ub_path = self._factor_paths(key)
        place_factors = self._fit_factors(up, self.config.factors, up_path)
        brewer_factors = self._fit_factors(ubm, self.config.factors, ub_path)
        print(f"[snapshot {key}] ALS factors={self.config.factors} {elapsed(start)}")
        if build_graph:
            full_path = self.cache / f"graph_{key}.npz"
            taste_path = self.cache / f"taste_{key}.npz"
            if full_path.exists() and taste_path.exists():
                graph = sp.load_npz(full_path)
                taste = sp.load_npz(taste_path)
            else:
                graph, taste = self._graphs(cutoff, pr, ub, us)
                sp.save_npz(full_path, graph, compressed=False)
                sp.save_npz(taste_path, taste, compressed=False)
        else:
            graph = sp.csr_matrix((self.data.n_nodes, self.data.n_nodes), dtype=np.float32)
            taste = graph
        dynamic = self._dynamic_features(cutoff, pr, beer_counts)
        snapshot = Snapshot(cutoff, key, graph, taste, up, ubm, usm, place_factors=place_factors, brewer_factors=brewer_factors, **dynamic)
        print(f"[snapshot {key}] ready nodes={self.data.n_nodes} edges={graph.nnz} {elapsed(start)}")
        return snapshot


def ppr_top_places(data: StaticData, graph: sp.csr_matrix, users: np.ndarray, k: int, iterations: int, batch_size: int | None = None) -> PPRResult:
    import torch

    users = np.asarray(users, dtype=np.int64)
    if len(users) == 0:
        return PPRResult(np.empty((0, k), dtype=np.int32), np.empty((0, k), dtype=np.float32))
    if torch.cuda.is_available():
        device = torch.device("cuda")
        free, _ = torch.cuda.mem_get_info()
        safe = int(free * 0.25 / max(1, 2 * data.n_nodes * 4))
        measured = int(np.clip(safe, 256, 1024))
    else:
        device = torch.device("cpu")
        measured = 64
    batch = measured if batch_size is None else min(batch_size, measured)
    transposed = graph.transpose().tocsr()
    crow = torch.from_numpy(transposed.indptr.astype(np.int64, copy=False)).to(device)
    col = torch.from_numpy(transposed.indices.astype(np.int64, copy=False)).to(device)
    values = torch.from_numpy(transposed.data.astype(np.float32, copy=False)).to(device)
    tensor = torch.sparse_csr_tensor(crow, col, values, size=transposed.shape, device=device)
    all_ids: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    print(f"[ppr] device={device.type} batch={batch} users={len(users)} iterations={iterations} edges={graph.nnz}")
    for begin in range(0, len(users), batch):
        current_users = users[begin:begin + batch]
        width = len(current_users)
        seeds = torch.as_tensor(current_users + data.user_offset, dtype=torch.long, device=device)
        columns = torch.arange(width, dtype=torch.long, device=device)
        state = torch.zeros((data.n_nodes, width), dtype=torch.float32, device=device)
        state[seeds, columns] = 1.0
        for _ in range(iterations):
            state = (1.0 - RESTART) * torch.sparse.mm(tensor, state)
            state[seeds, columns] += RESTART
        place_mass = state[data.place_offset:data.place_offset + data.n_places].T
        scores, ids = torch.topk(place_mass, k=k, dim=1, largest=True, sorted=True)
        all_ids.append(ids.cpu().numpy().astype(np.int32))
        all_scores.append(scores.cpu().numpy().astype(np.float32))
        del state, place_mass, scores, ids
    del tensor, crow, col, values
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return PPRResult(np.vstack(all_ids), np.vstack(all_scores))


class CandidateGenerator:
    def __init__(self, data: StaticData, config: BuildConfig):
        self.data = data
        self.config = config
        mode = "debug" if config.debug else "full"
        self.cache = data.cache_dir / f"lane2_graphdiff_{mode}_v3" / "candidates_v6"
        self.cache.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, snapshot: Snapshot, users: np.ndarray) -> Path:
        digest = hashlib.sha256(np.asarray(users, dtype=np.int32).tobytes()).hexdigest()[:16]
        return self.cache / f"{snapshot.key}_{digest}_i{self.config.ppr_iterations}.npz"

    def _load_candidates(self, path: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
        stored = np.load(path, allow_pickle=False)
        lengths = stored["lengths"]
        candidates = [stored["candidates"][i, :int(lengths[i])].copy() for i in range(len(lengths))]
        components = {}
        for name in ("full", "taste", "place_als", "brewer", "geo"):
            components[name] = PPRResult(stored[f"{name}_ids"], stored[f"{name}_scores"])
        return candidates, components

    def _save_candidates(self, path: Path, candidates: list[np.ndarray], components: dict[str, Any]) -> None:
        width = max(len(x) for x in candidates)
        padded = np.full((len(candidates), width), -1, dtype=np.int32)
        lengths = np.asarray([len(x) for x in candidates], dtype=np.int16)
        for row, values in enumerate(candidates):
            padded[row, :len(values)] = values
        payload: dict[str, np.ndarray] = {"candidates": padded, "lengths": lengths}
        for name, values in components.items():
            payload[f"{name}_ids"] = values.ids
            payload[f"{name}_scores"] = values.scores
        np.savez(path, **payload)

    def _als_top(self, factors: Factors, users: np.ndarray, k: int, blocked: sp.csr_matrix | None = None) -> PPRResult:
        ids_out = np.empty((len(users), k), dtype=np.int32)
        scores_out = np.empty((len(users), k), dtype=np.float32)
        item_t = factors.items.T
        for begin in range(0, len(users), 128):
            current = users[begin:begin + 128]
            values = factors.users[current] @ item_t
            if blocked is not None:
                for local, user in enumerate(current):
                    values[local, blocked[int(user)].indices] = -np.inf
            part = np.argpartition(values, -k, axis=1)[:, -k:]
            part_values = np.take_along_axis(values, part, axis=1)
            order = np.argsort(-part_values, axis=1, kind="stable")
            idx = np.take_along_axis(part, order, axis=1)
            ids_out[begin:begin + len(current)] = idx.astype(np.int32)
            scores_out[begin:begin + len(current)] = np.take_along_axis(values, idx, axis=1)
        return PPRResult(ids_out, scores_out)

    def _projected_brewer(self, snapshot: Snapshot, users: np.ndarray, brewer_top: PPRResult, k: int) -> PPRResult:
        d = self.data
        ids_out = np.zeros((len(users), k), dtype=np.int32)
        scores_out = np.full((len(users), k), -np.inf, dtype=np.float32)
        quality = self._quality(snapshot)
        eligible = d.place_created <= np.datetime64(snapshot.cutoff)
        state_top: dict[int, np.ndarray] = {}
        country_top: dict[int, np.ndarray] = {}
        for state in np.unique(d.brewer_state[brewer_top.ids]):
            if state < 0:
                continue
            candidates = np.flatnonzero(eligible & (d.place_state == state))
            state_top[int(state)] = candidates[top_indices(quality[candidates], min(30, len(candidates)))]
        for country in np.unique(d.brewer_country[brewer_top.ids]):
            if country < 0:
                continue
            candidates = np.flatnonzero(eligible & (d.place_country == country))
            country_top[int(country)] = candidates[top_indices(quality[candidates], min(15, len(candidates)))]
        for row in range(len(users)):
            scores: dict[int, float] = {}
            for brewer, value in zip(brewer_top.ids[row], brewer_top.scores[row]):
                brewer = int(brewer)
                value = float(value)
                name = d.brewer_name[brewer]
                for place in d.name_to_places.get(name, []):
                    if eligible[place]:
                        scores[place] = max(scores.get(place, -np.inf), value + 1.5)
                state = int(d.brewer_state[brewer])
                for place in state_top.get(state, []):
                    scores[int(place)] = max(scores.get(int(place), -np.inf), value * 0.65 + 0.02 * float(quality[place]))
                country = int(d.brewer_country[brewer])
                for place in country_top.get(country, []):
                    scores[int(place)] = max(scores.get(int(place), -np.inf), value * 0.25 + 0.01 * float(quality[place]))
            ranked = sorted(scores, key=lambda x: (-scores[x], x))[:k]
            if len(ranked) < k:
                fallback = top_indices(quality, k)
                ranked.extend([int(x) for x in fallback if int(x) not in scores][:k - len(ranked)])
            ids_out[row] = np.asarray(ranked[:k], dtype=np.int32)
            scores_out[row] = np.asarray([scores.get(int(x), float(quality[x]) * 0.01) for x in ranked[:k]], dtype=np.float32)
        return PPRResult(ids_out, scores_out)

    def _quality(self, snapshot: Snapshot) -> np.ndarray:
        count = snapshot.place_score_count
        mean = np.divide(snapshot.place_score_sum, np.maximum(count, 1.0))
        global_mean = float(snapshot.place_score_sum.sum() / max(1.0, count.sum()))
        shrunk = (snapshot.place_score_sum + 8.0 * global_mean) / (count + 8.0)
        return (shrunk + 1.2 * np.log1p(snapshot.place_365_liked) + 0.35 * np.log1p(snapshot.place_life_liked)).astype(np.float32)

    def _geo(self, snapshot: Snapshot, users: np.ndarray, k: int) -> PPRResult:
        d = self.data
        quality = self._quality(snapshot)
        eligible = d.place_created <= np.datetime64(snapshot.cutoff)
        ids_out = np.empty((len(users), k), dtype=np.int32)
        scores_out = np.empty((len(users), k), dtype=np.float32)
        global_pool = np.flatnonzero(eligible)
        global_rank = global_pool[top_indices(quality[global_pool], min(1000, len(global_pool)))]
        binary = snapshot.up.copy()
        binary.data[:] = 1.0
        for row, user in enumerate(users):
            history = binary[int(user)].indices
            states, state_counts = np.unique(d.place_state[history][d.place_state[history] >= 0], return_counts=True)
            cities, city_counts = np.unique(self._city_codes()[history][self._city_codes()[history] >= 0], return_counts=True)
            cells, cell_counts = np.unique(d.place_h3[history][d.place_h3[history] >= 0], return_counts=True)
            top_states = states[np.argsort(-state_counts)[:3]] if len(states) else np.empty(0, dtype=np.int32)
            top_cities = cities[np.argsort(-city_counts)[:6]] if len(cities) else np.empty(0, dtype=np.int32)
            top_cells = cells[np.argsort(-cell_counts)[:6]] if len(cells) else np.empty(0, dtype=np.int32)
            mask = eligible & (
                np.isin(d.place_state, top_states)
                | np.isin(self._city_codes(), top_cities)
                | np.isin(d.place_h3, top_cells)
            )
            pool = np.flatnonzero(mask)
            if len(pool) < k:
                pool = np.unique(np.concatenate((pool, global_rank)))
            distance = haversine(snapshot.user_centroid_lat[user], snapshot.user_centroid_lon[user], d.place_lat[pool], d.place_lon[pool])
            value = quality[pool] - 0.035 * np.log1p(distance)
            if len(top_states):
                value += 0.55 * np.isin(d.place_state[pool], top_states)
            if len(top_cells):
                value += 0.9 * np.isin(d.place_h3[pool], top_cells)
            if len(top_cities):
                value += 1.25 * np.isin(self._city_codes()[pool], top_cities)
            age = (np.datetime64(snapshot.cutoff) - d.place_created[pool]).astype("timedelta64[D]").astype(np.float32)
            value += 0.7 * ((age >= 0) & (age <= 730))
            idx = top_indices(value, min(k, len(pool)))
            geo_selected = pool[idx].tolist()
            geo_scores = {int(x): float(y) for x, y in zip(pool[idx], value[idx])}
            if len(history):
                peers = binary[int(user)] @ binary.T
                if peers.nnz:
                    peers.data = np.minimum(peers.data, 10.0)
                co = (peers @ binary).tocsr()
                co_places = co.indices
                co_values = co.data.astype(np.float32)
                valid = eligible[co_places] & ~np.isin(co_places, history)
                co_places = co_places[valid]
                co_values = co_values[valid]
                co_idx = top_indices(co_values, min(k, len(co_values)))
                co_selected = co_places[co_idx].tolist()
                co_scores = {int(x): float(y) for x, y in zip(co_places[co_idx], co_values[co_idx])}
            else:
                co_selected = []
                co_scores = {}
            selected = []
            seen = set()
            for offset in range(k):
                for source in (geo_selected, co_selected):
                    if offset < len(source) and int(source[offset]) not in seen:
                        seen.add(int(source[offset]))
                        selected.append(int(source[offset]))
                        if len(selected) >= k:
                            break
                if len(selected) >= k:
                    break
            if len(selected) < k:
                selected.extend([int(x) for x in global_rank if int(x) not in selected][:k - len(selected)])
            ids_out[row] = np.asarray(selected[:k], dtype=np.int32)
            scores_out[row] = np.asarray([max(geo_scores.get(int(x), -np.inf), np.log1p(co_scores.get(int(x), 0.0))) for x in selected[:k]], dtype=np.float32)
            missing = ~np.isfinite(scores_out[row])
            scores_out[row, missing] = quality[ids_out[row, missing]]
        return PPRResult(ids_out, scores_out)

    def _city_codes(self) -> np.ndarray:
        if not hasattr(self, "city_codes"):
            names = sorted(set(self.data.place_city_text) - {""})
            mapping = {x: i for i, x in enumerate(names)}
            self.city_codes = np.array([mapping.get(x, -1) for x in self.data.place_city_text], dtype=np.int32)
        return self.city_codes

    def generate(self, snapshot: Snapshot, users: np.ndarray) -> tuple[list[np.ndarray], dict[str, Any]]:
        start = time.time()
        users = np.asarray(users, dtype=np.int32)
        path = self._cache_path(snapshot, users)
        if path.exists():
            result = self._load_candidates(path)
            print(f"[candidates {snapshot.key}] cache rows={len(users)} {elapsed(start)}")
            return result
        full = ppr_top_places(self.data, snapshot.graph, users, 700, self.config.ppr_iterations)
        taste = ppr_top_places(self.data, snapshot.taste_graph, users, 250, self.config.ppr_iterations)
        place_als = self._als_top(snapshot.place_factors, users, 400, snapshot.up)
        brewer_top = self._als_top(snapshot.brewer_factors, users, 150, snapshot.ub)
        brewer_projected = self._projected_brewer(snapshot, users, brewer_top, 300)
        geo = self._geo(snapshot, users, 100)
        candidates: list[np.ndarray] = []
        eligible = self.data.place_created <= np.datetime64(snapshot.cutoff)
        for row, user in enumerate(users):
            prior = set(snapshot.up[int(user)].indices.tolist())
            clean: dict[str, list[int]] = {}
            for name, source, quota in (
                ("full", full.ids[row], 450),
                ("taste", taste.ids[row], 100),
                ("geo", geo.ids[row], 100),
            ):
                values = []
                for place in source:
                    place = int(place)
                    if place not in prior and eligible[place] and place not in values:
                        values.append(place)
                        if len(values) >= quota:
                            break
                clean[name] = values
            clean["place_als"] = [int(x) for x in place_als.ids[row] if int(x) not in prior and eligible[int(x)]][:250]
            clean["brewer"] = [int(x) for x in brewer_projected.ids[row] if int(x) not in prior and eligible[int(x)]][:150]
            cap_quotas = {"full": 400, "taste": 50, "place_als": 100, "brewer": 25, "geo": 25}
            union = []
            seen = set()
            for name in ("full", "taste", "place_als", "brewer", "geo"):
                added = 0
                for place in clean[name]:
                    if place in seen:
                        continue
                    seen.add(place)
                    union.append(place)
                    added += 1
                    if added >= cap_quotas[name]:
                        break
            if len(union) < self.config.candidate_cap:
                for offset in range(700):
                    for source in (full.ids[row], place_als.ids[row], taste.ids[row], brewer_projected.ids[row]):
                        if offset >= len(source):
                            continue
                        place = int(source[offset])
                        if place in seen or place in prior or not eligible[place]:
                            continue
                        seen.add(place)
                        union.append(place)
                        if len(union) >= self.config.candidate_cap:
                            break
                    if len(union) >= self.config.candidate_cap:
                        break
            candidates.append(np.asarray(union, dtype=np.int32))
        components = {
            "full": full,
            "taste": taste,
            "place_als": place_als,
            "brewer": brewer_projected,
            "geo": geo,
        }
        self._save_candidates(path, candidates, components)
        print(f"[candidates {snapshot.key}] rows={len(users)} mean={np.mean([len(x) for x in candidates]):.1f} {elapsed(start)}")
        return candidates, components


class FeatureBuilder:
    names = [
        "full_ppr", "full_rank", "taste_ppr", "taste_rank", "place_als", "place_als_rank",
        "brewer_als", "brewer_rank", "source_count", "agreement_ppr_als", "co_rater",
        "place_90", "place_365", "place_life", "place_like_365", "place_like_life",
        "place_mean_score", "place_shrunk_quality", "place_age_days", "new_place", "place_cold",
        "same_state", "same_city", "same_h3", "distance_km", "exact_brewer_name",
        "flag_full", "flag_taste", "flag_place_als", "flag_brewer", "flag_geo",
        "user_place_life", "user_place_365", "user_place_90", "user_beer_life", "user_beer_365",
        "user_beer_90", "place_type", "place_country",
        "state_history_count", "city_history_count", "h3_history_count", "same_country",
        "last_distance_km", "place_trend_90_365", "place_like_fraction", "type_history_count",
        "same_last_state", "same_last_city", "same_last_h3",
    ]

    def __init__(self, data: StaticData):
        self.data = data
        self.binary_up: dict[str, sp.csr_matrix] = {}
        cities = sorted(set(data.place_city_text) - {""})
        mapping = {x: i for i, x in enumerate(cities)}
        self.city_codes = np.array([mapping.get(x, -1) for x in data.place_city_text], dtype=np.int32)

    def _component_maps(self, components: dict[str, PPRResult], row: int) -> dict[str, tuple[dict[int, float], dict[int, int]]]:
        result = {}
        for name, values in components.items():
            result[name] = (
                {int(x): float(y) for x, y in zip(values.ids[row], values.scores[row])},
                {int(x): i + 1 for i, x in enumerate(values.ids[row])},
            )
        return result

    def _co_rater(self, snapshot: Snapshot, user: int, candidates: np.ndarray) -> np.ndarray:
        if snapshot.key not in self.binary_up:
            binary = snapshot.up.copy()
            binary.data[:] = 1.0
            self.binary_up = {snapshot.key: binary}
        binary = self.binary_up[snapshot.key]
        profile = binary[user]
        if profile.nnz == 0:
            return np.zeros(len(candidates), dtype=np.float32)
        peers = profile @ binary.T
        if peers.nnz:
            peers.data = np.minimum(peers.data, 10.0)
        values = peers @ binary[:, candidates]
        return np.asarray(values.toarray()).ravel().astype(np.float32)

    def transform(self, snapshot: Snapshot, users: np.ndarray, candidates: list[np.ndarray], components: dict[str, PPRResult]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = self.data
        blocks: list[np.ndarray] = []
        row_users: list[np.ndarray] = []
        row_places: list[np.ndarray] = []
        global_mean = float(snapshot.place_score_sum.sum() / max(1.0, snapshot.place_score_count.sum()))
        shrunk = (snapshot.place_score_sum + 8.0 * global_mean) / (snapshot.place_score_count + 8.0)
        mean_score = np.divide(snapshot.place_score_sum, np.maximum(snapshot.place_score_count, 1.0))
        for row, (user, places) in enumerate(zip(users, candidates)):
            user = int(user)
            maps = self._component_maps(components, row)
            length = len(places)
            x = np.zeros((length, len(self.names)), dtype=np.float32)
            full_s, full_r = maps["full"]
            taste_s, taste_r = maps["taste"]
            up_s, up_r = maps["place_als"]
            ub_s, ub_r = maps["brewer"]
            geo_s, geo_r = maps["geo"]
            score_maps = (full_s, taste_s, up_s, ub_s, geo_s)
            rank_maps = (full_r, taste_r, up_r, ub_r, geo_r)
            score_arrays = [np.array([m.get(int(p), 0.0) for p in places], dtype=np.float32) for m in score_maps]
            rank_arrays = [np.array([m.get(int(p), 1000) for p in places], dtype=np.float32) for m in rank_maps]
            flags = [(v > 0).astype(np.float32) for v in score_arrays]
            age = (np.datetime64(snapshot.cutoff) - d.place_created[places]).astype("timedelta64[D]").astype(np.float32)
            distance = haversine(snapshot.user_centroid_lat[user], snapshot.user_centroid_lon[user], d.place_lat[places], d.place_lon[places])
            name_match = np.array([1.0 if len(d.place_name_brewers[int(p)]) and snapshot.ub[user, d.place_name_brewers[int(p)]].nnz else 0.0 for p in places], dtype=np.float32)
            co = self._co_rater(snapshot, user, places)
            history = snapshot.up[user].indices
            history_states = d.place_state[history]
            history_cities = self.city_codes[history]
            history_h3 = d.place_h3[history]
            history_countries = d.place_country[history]
            history_types = d.place_type[history]
            state_count = np.array([np.sum(history_states == d.place_state[p]) for p in places], dtype=np.float32)
            city_count = np.array([np.sum(history_cities == self.city_codes[p]) for p in places], dtype=np.float32)
            h3_count = np.array([np.sum(history_h3 == d.place_h3[p]) for p in places], dtype=np.float32)
            type_count = np.array([np.sum(history_types == d.place_type[p]) for p in places], dtype=np.float32)
            same_country = np.isin(d.place_country[places], history_countries).astype(np.float32)
            last_place = int(snapshot.user_last_place[user])
            if last_place >= 0:
                last_distance = haversine(d.place_lat[last_place], d.place_lon[last_place], d.place_lat[places], d.place_lon[places])
            else:
                last_distance = np.full(length, 20000.0, dtype=np.float32)
            values = [
                score_arrays[0], rank_arrays[0], score_arrays[1], rank_arrays[1], score_arrays[2], rank_arrays[2],
                score_arrays[3], rank_arrays[3], np.sum(flags, axis=0), flags[0] * flags[2], co,
                np.log1p(snapshot.place_90_count[places]), np.log1p(snapshot.place_365_count[places]), np.log1p(snapshot.place_life_count[places]),
                np.log1p(snapshot.place_365_liked[places]), np.log1p(snapshot.place_life_liked[places]), mean_score[places], shrunk[places],
                age, ((age >= 0) & (age <= 730)).astype(np.float32), (snapshot.place_life_count[places] == 0).astype(np.float32),
                (state_count > 0).astype(np.float32), (city_count > 0).astype(np.float32),
                (h3_count > 0).astype(np.float32), distance, name_match,
                flags[0], flags[1], flags[2], flags[3], flags[4],
                np.full(length, np.log1p(snapshot.user_place_count[user]), dtype=np.float32),
                np.full(length, np.log1p(snapshot.user_365_place_count[user]), dtype=np.float32),
                np.full(length, np.log1p(snapshot.user_90_place_count[user]), dtype=np.float32),
                np.full(length, np.log1p(snapshot.user_beer_count[user]), dtype=np.float32),
                np.full(length, np.log1p(snapshot.user_365_beer_count[user]), dtype=np.float32),
                np.full(length, np.log1p(snapshot.user_90_beer_count[user]), dtype=np.float32),
                (d.place_type[places] + 1).astype(np.float32), (d.place_country[places] + 1).astype(np.float32),
                np.log1p(state_count), np.log1p(city_count), np.log1p(h3_count), same_country, last_distance,
                snapshot.place_90_count[places] / (snapshot.place_365_count[places] + 1.0),
                snapshot.place_life_liked[places] / (snapshot.place_life_count[places] + 1.0), np.log1p(type_count),
                (d.place_state[places] == snapshot.user_last_state[user]).astype(np.float32),
                (self.city_codes[places] == snapshot.user_last_city[user]).astype(np.float32),
                (d.place_h3[places] == snapshot.user_last_h3[user]).astype(np.float32),
            ]
            x[:] = np.column_stack(values)
            blocks.append(x)
            row_users.append(np.full(length, user, dtype=np.int32))
            row_places.append(places)
        return np.vstack(blocks), np.concatenate(row_users), np.concatenate(row_places)


def add_training_positives(data: StaticData, snapshot: Snapshot, users: np.ndarray, labels: list[np.ndarray], candidates: list[np.ndarray], components: dict[str, PPRResult], cap: int) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    eligible = data.place_created <= np.datetime64(snapshot.cutoff)
    for user, truth, pool in zip(users, labels, candidates):
        prior = set(snapshot.up[int(user)].indices.tolist())
        positives = [int(x) for x in truth if x is not None and int(x) >= 0 and int(x) < data.n_places and eligible[int(x)] and int(x) not in prior]
        ordered: list[int] = []
        seen: set[int] = set()
        for place in positives + pool.tolist():
            if place not in seen:
                seen.add(place)
                ordered.append(place)
        result.append(np.asarray(ordered[:cap], dtype=np.int32))
    return result


def labels_for_rows(row_users: np.ndarray, row_places: np.ndarray, users: np.ndarray, labels: list[np.ndarray]) -> np.ndarray:
    lookup = {int(u): set(int(x) for x in target if x is not None) for u, target in zip(users, labels)}
    return np.asarray([1 if int(p) in lookup[int(u)] else 0 for u, p in zip(row_users, row_places)], dtype=np.int8)


def fit_ranker(x: np.ndarray, y: np.ndarray, groups: np.ndarray, rounds: int, validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> tuple[lgb.Booster, int]:
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "lambdarank_truncation_level": 13,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "lambda_l2": 6.0,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "11")),
        "seed": SEED,
        "feature_pre_filter": False,
    }
    categorical = ["place_type", "place_country"]
    train_set = lgb.Dataset(
        x, label=y, group=groups, feature_name=FeatureBuilder.names,
        categorical_feature=categorical, free_raw_data=False,
    )
    valid_sets = None
    callbacks = [lgb.log_evaluation(0)]
    if validation is not None:
        vx, vy, vg = validation
        valid_sets = [lgb.Dataset(
            vx, label=vy, group=vg, reference=train_set, feature_name=FeatureBuilder.names,
            categorical_feature=categorical,
        )]
        callbacks.append(lgb.early_stopping(80, verbose=False))
    model = lgb.train(params, train_set, num_boost_round=rounds, valid_sets=valid_sets, callbacks=callbacks)
    best = model.best_iteration if model.best_iteration else rounds
    return model, int(best)


def rank_predictions(model: lgb.Booster, x: np.ndarray, candidates: list[np.ndarray], blend: float = 0.0) -> np.ndarray:
    scores = model.predict(x, num_iteration=model.best_iteration)
    result = np.empty((len(candidates), 10), dtype=np.int64)
    cursor = 0
    rank_columns = [
        FeatureBuilder.names.index("full_rank"), FeatureBuilder.names.index("taste_rank"),
        FeatureBuilder.names.index("place_als_rank"), FeatureBuilder.names.index("brewer_rank"),
    ]
    rank_weights = np.asarray([0.55, 0.15, 0.25, 0.05], dtype=np.float32)
    for row, places in enumerate(candidates):
        values = scores[cursor:cursor + len(places)]
        block = x[cursor:cursor + len(places)]
        cursor += len(places)
        if blend > 0:
            heuristic = np.sum(-np.log1p(block[:, rank_columns]) * rank_weights, axis=1)
            model_scale = (values - np.mean(values)) / (np.std(values) + 1e-6)
            heuristic_scale = (heuristic - np.mean(heuristic)) / (np.std(heuristic) + 1e-6)
            values = (1.0 - blend) * model_scale + blend * heuristic_scale
        order = np.lexsort((places, -values))[:10]
        result[row] = places[order]
    return result


def map_at_10(predictions: np.ndarray, labels: list[np.ndarray]) -> float:
    values = []
    for pred, truth in zip(predictions, labels):
        target = set(int(x) for x in truth if x is not None)
        hits = np.array([int(x) in target for x in pred], dtype=np.float32)
        precision = np.cumsum(hits) / np.arange(1, 11)
        values.append(float(np.sum(precision * hits) / min(max(len(target), 1), 10)))
    return float(np.mean(values))


def candidate_recall(candidates: list[np.ndarray], labels: list[np.ndarray]) -> tuple[float, float]:
    hits = 0
    total = 0
    rows = []
    for pool, truth in zip(candidates, labels):
        target = set(int(x) for x in truth if x is not None)
        found = len(target.intersection(int(x) for x in pool))
        hits += found
        total += len(target)
        rows.append(found / max(1, len(target)))
    return hits / max(1, total), float(np.mean(rows))


def exact_episodes(data: StaticData, origins: list[pd.Timestamp]) -> pd.DataFrame:
    timestamps = ", ".join(f"(timestamp '{pd.Timestamp(t).strftime('%Y-%m-%d %H:%M:%S')}')" for t in origins)
    query = f"""
        WITH timestamp_df(timestamp) AS (VALUES {timestamps})
        SELECT t.timestamp, pr.user_id, list(DISTINCT pr.place_id) place_id
        FROM timestamp_df t
        LEFT JOIN read_parquet('{data.parquet('place_ratings')}') pr
          ON pr.created_at > t.timestamp AND pr.created_at <= t.timestamp + interval '90 days'
        WHERE
          pr.user_id IS NOT NULL AND pr.place_id IS NOT NULL AND pr.total_score >= 80
          AND EXISTS (
            SELECT 1 FROM read_parquet('{data.parquet('place_ratings')}') pr2
            WHERE pr2.user_id = pr.user_id
              AND pr2.created_at > t.timestamp - interval '90 days'
              AND pr2.created_at <= t.timestamp
          )
          OR EXISTS (
            SELECT 1 FROM read_parquet('{data.parquet('beer_ratings')}') br
            WHERE br.user_id = pr.user_id
              AND br.created_at > t.timestamp - interval '90 days'
              AND br.created_at <= t.timestamp
          )
        GROUP BY t.timestamp, pr.user_id
        ORDER BY t.timestamp, pr.user_id
    """
    return data.con.execute(query).fetchdf()


def content_key(config: BuildConfig) -> str:
    raw = json.dumps(config.__dict__, sort_keys=True) + "graphdiff_v13"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def release_snapshot(snapshot: Snapshot) -> None:
    del snapshot
    gc.collect()
