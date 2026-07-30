import fcntl
import json
import math
import os
import re
import time
import uuid
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


class AvitoPipeline:
    def __init__(self, debug: bool):
        self.debug = debug
        self.seed = 1337
        self.threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
        self.cache_root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / "lane0_exact_replay_v3"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.db_root = (
            Path(os.environ["RELBENCH_CACHE_DIR"])
            / os.environ["RELBENCH_DATASET"]
            / "db"
        )
        self.horizons = [4, "all"] if debug else [0.25, 1, 2, 4, 8, "all"]
        self.max_lgb = 100 if debug else 1800
        self.max_cat = 100 if debug else 1600
        self.feature_version = "debug_v4" if debug else "full_v4"
        self.fold_records = []
        self.oof = pd.DataFrame()
        self.started = time.time()
        self.con = duckdb.connect()
        self.con.execute(f"PRAGMA threads={self.threads}")
        self.con.execute("PRAGMA preserve_insertion_order=false")
        self.paths = {
            name: self.db_root / f"{name}.parquet"
            for name in [
                "AdsInfo",
                "Category",
                "Location",
                "SearchStream",
                "SearchInfo",
                "UserInfo",
                "VisitStream",
                "PhoneRequestsStream",
            ]
        }

    def scan(self, name: str) -> str:
        value = str(self.paths[name]).replace("'", "''")
        return f"read_parquet('{value}')"

    def prepare(self) -> dict:
        from relbench.tasks import get_task

        task = get_task(
            os.environ["RELBENCH_DATASET"],
            os.environ["RELBENCH_TASK"],
            download=False,
        )
        train = task.get_table("train").df[["AdID", "timestamp", task.target_col]].copy()
        val_seed = task.get_table("val").df[["AdID", "timestamp"]].copy()
        test_seed = task.get_table("test").df[["AdID", "timestamp"]].copy()
        train = train.rename(columns={task.target_col: "target"})
        labels_a = self.replay_labels("2015-05-04")
        labels_b = self.replay_labels("2015-05-10")
        labels_a = self.combine_labels(labels_a, train)
        labels_b = self.combine_labels(labels_b, train)
        if self.debug:
            labels_a = self.debug_sample(labels_a)
            labels_b = self.debug_sample(labels_b)
        seed_keys = pd.concat(
            [
                labels_b[["AdID", "timestamp"]],
                val_seed[["AdID", "timestamp"]],
                test_seed[["AdID", "timestamp"]],
            ],
            ignore_index=True,
        ).drop_duplicates(["AdID", "timestamp"])
        seed_keys = seed_keys.sort_values(["timestamp", "AdID"]).reset_index(drop=True)
        seed_keys["row_id"] = np.arange(len(seed_keys), dtype=np.int64)
        safe_fit_ids = pd.Index(
            pd.concat(
                [
                    labels_a["AdID"],
                    val_seed["AdID"],
                    test_seed["AdID"],
                ],
                ignore_index=True,
            ).unique()
        )
        features = self.load_or_build_features(seed_keys, safe_fit_ids)
        labels_a = labels_a.merge(features, on=["AdID", "timestamp"], how="left", validate="one_to_one")
        labels_b = labels_b.merge(features, on=["AdID", "timestamp"], how="left", validate="one_to_one")
        val_features = (
            val_seed.assign(eval_row=np.arange(len(val_seed), dtype=np.int64))
            .merge(features, on=["AdID", "timestamp"], how="left", validate="one_to_one")
            .sort_values("eval_row")
            .reset_index(drop=True)
        )
        test_features = (
            test_seed.assign(eval_row=np.arange(len(test_seed), dtype=np.int64))
            .merge(features, on=["AdID", "timestamp"], how="left", validate="one_to_one")
            .sort_values("eval_row")
            .reset_index(drop=True)
        )
        if labels_a.isna().all(axis=1).any() or len(val_features) != len(val_seed):
            raise RuntimeError("feature alignment failed")
        print(
            f"[data] replay_a={len(labels_a)} replay_b={len(labels_b)} "
            f"features={features.shape} val={len(val_features)} test={len(test_features)}"
        )
        return {
            "task": task,
            "train_a": labels_a,
            "train_b": labels_b,
            "val": val_features,
            "test": test_features,
        }

    def replay_labels(self, end_date: str) -> pd.DataFrame:
        if self.debug:
            end_date = "2015-04-27"
        query = f"""
        WITH origins AS (
            SELECT timestamp
            FROM generate_series(
                DATE '2015-04-26',
                DATE '{end_date}',
                INTERVAL 1 DAY
            ) AS x(timestamp)
        )
        SELECT
            CAST(s.AdID AS BIGINT) AS AdID,
            CAST(o.timestamp AS TIMESTAMP) AS timestamp,
            SUM(COALESCE(s.IsClick, 0))::DOUBLE AS C,
            COUNT(s.SearchID)::DOUBLE AS N_all_label,
            COUNT(s.IsClick)::DOUBLE AS N_labeled_label,
            COUNT(*) FILTER (WHERE s.ObjectType = 1)::DOUBLE AS N_object_1_label,
            COUNT(*) FILTER (WHERE s.ObjectType = 2)::DOUBLE AS N_object_2_label,
            COUNT(*) FILTER (WHERE s.ObjectType = 3)::DOUBLE AS N_object_3_label,
            SUM(COALESCE(s.IsClick, 0))::DOUBLE / COUNT(s.SearchID)::DOUBLE AS target,
            SUM(COALESCE(s.IsClick, 0))::DOUBLE / COUNT(s.IsClick)::DOUBLE AS target_labeled
        FROM origins o
        JOIN {self.scan("SearchStream")} s
          ON s.SearchDate > o.timestamp
         AND s.SearchDate <= o.timestamp + INTERVAL 4 DAY
        GROUP BY s.AdID, o.timestamp
        HAVING SUM(COALESCE(s.IsClick, 0)) > 0
        ORDER BY timestamp, AdID
        """
        result = self.con.execute(query).df()
        print(f"[labels] through={end_date} rows={len(result)}")
        return result

    def combine_labels(self, replay: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
        added = official.copy()
        added["C"] = np.nan
        added["N_all_label"] = np.nan
        added["N_labeled_label"] = np.nan
        added["N_object_1_label"] = np.nan
        added["N_object_2_label"] = np.nan
        added["N_object_3_label"] = np.nan
        added["target_labeled"] = np.nan
        columns = list(replay.columns)
        combined = pd.concat([replay, added[columns]], ignore_index=True)
        combined = combined.drop_duplicates(["AdID", "timestamp"], keep="first")
        return combined.sort_values(["timestamp", "AdID"]).reset_index(drop=True)

    def debug_sample(self, labels: pd.DataFrame) -> pd.DataFrame:
        parts = []
        origins = sorted(labels["timestamp"].unique())[:2]
        for origin in origins:
            part = labels[labels["timestamp"] == origin].sort_values("AdID").head(500)
            parts.append(part)
        return pd.concat(parts, ignore_index=True).sort_values(["timestamp", "AdID"]).reset_index(drop=True)

    def load_or_build_features(self, seeds: pd.DataFrame, safe_fit_ids: pd.Index) -> pd.DataFrame:
        feature_path = self.cache_root / f"features_{self.feature_version}.parquet"
        if feature_path.exists():
            cached = pd.read_parquet(feature_path)
            expected = seeds[["AdID", "timestamp"]]
            matched = expected.merge(
                cached[["AdID", "timestamp"]],
                on=["AdID", "timestamp"],
                how="left",
                indicator=True,
            )
            if (matched["_merge"] == "both").all():
                print(f"[cache] features hit path={feature_path.name}")
                return expected.merge(cached, on=["AdID", "timestamp"], how="left", validate="one_to_one")
        features = self.build_features(seeds, safe_fit_ids)
        temporary = feature_path.with_name(f"{feature_path.name}.{uuid.uuid4().hex}.tmp")
        features.to_parquet(temporary, index=False)
        os.replace(temporary, feature_path)
        self.register_artifact(
            feature_path,
            f"Temporally censored {self.feature_version} replay features",
            f"rel-avito-ad-ctr-lane0-exact-replay-{self.feature_version}",
        )
        return features

    def build_features(self, seeds: pd.DataFrame, safe_fit_ids: pd.Index) -> pd.DataFrame:
        print(f"[features] start seeds={len(seeds)}")
        ids = pd.DataFrame({"AdID": pd.Index(seeds["AdID"].unique()).astype("int64")})
        self.con.register("relevant_ids", ids)
        static = self.build_static(ids, safe_fit_ids)
        enriched_seeds = seeds.merge(static, on="AdID", how="left", validate="many_to_one")
        self.con.register("feature_seeds", enriched_seeds)
        self.preaggregate_search()
        core = self.build_core()
        print(f"[features] core elapsed={time.time() - self.started:.1f}s")
        audience = self.build_audience()
        print(f"[features] audience elapsed={time.time() - self.started:.1f}s")
        merged = enriched_seeds.merge(core, on="row_id", how="left", validate="one_to_one")
        merged = merged.merge(audience, on="row_id", how="left", validate="one_to_one")
        cohorts = self.build_cohorts(merged)
        merged = merged.merge(cohorts, on="row_id", how="left", validate="one_to_one")
        engagement = self.build_engagement(merged)
        merged = merged.merge(engagement, on="row_id", how="left", validate="one_to_one")
        direct = self.build_direct_engagement(merged)
        if direct is not None:
            merged = merged.merge(direct, on="row_id", how="left", validate="one_to_one")
        merged = self.derive_features(merged)
        merged = merged.sort_values("row_id").reset_index(drop=True)
        merged = merged.drop(columns=["row_id"])
        print(f"[features] complete shape={merged.shape} elapsed={time.time() - self.started:.1f}s")
        return merged

    def build_static(self, ids: pd.DataFrame, safe_fit_ids: pd.Index) -> pd.DataFrame:
        query = f"""
        SELECT
            CAST(a.AdID AS BIGINT) AS AdID,
            CAST(a.LocationID AS BIGINT) AS ad_location,
            CAST(a.CategoryID AS BIGINT) AS ad_category,
            COALESCE(a.Price, 0)::DOUBLE AS price,
            COALESCE(a.Title, '') AS title,
            COALESCE(a.IsContext, -1)::DOUBLE AS is_context,
            CAST(c.Level AS BIGINT) AS category_level,
            CAST(c.ParentCategoryID AS BIGINT) AS parent_category,
            CAST(c.SubcategoryID AS BIGINT) AS subcategory,
            COALESCE(l.Level, -1)::DOUBLE AS location_level,
            COALESCE(l.RegionID, -1)::DOUBLE AS ad_region,
            COALESCE(l.CityID, -1)::DOUBLE AS ad_city
        FROM {self.scan("AdsInfo")} a
        JOIN relevant_ids r ON a.AdID = r.AdID
        LEFT JOIN {self.scan("Category")} c ON a.CategoryID = c.CategoryID
        LEFT JOIN {self.scan("Location")} l ON a.LocationID = l.LocationID
        """
        static = self.con.execute(query).df()
        quantiles = self.con.execute(
            f"""
            SELECT
                CAST(CategoryID AS BIGINT) AS ad_category,
                quantile_cont(COALESCE(Price, 0), [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]) AS qs,
                median(COALESCE(Price, 0))::DOUBLE AS category_price_median,
                stddev_pop(ln(1 + GREATEST(COALESCE(Price, 0), 0)))::DOUBLE AS category_log_price_std
            FROM {self.scan("AdsInfo")}
            GROUP BY CategoryID
            """
        ).df()
        static = static.merge(quantiles, on="ad_category", how="left")
        static["log_price"] = np.log1p(np.maximum(static["price"].fillna(0), 0))
        static["price_abs_band"] = np.floor(static["log_price"] / 1.5).clip(0, 12).astype("int16")
        static["category_price_percentile"] = [
            (np.searchsorted(np.asarray(q, dtype=float), p, side="right") + 0.5) / 10
            if isinstance(q, (list, np.ndarray))
            else 0.5
            for p, q in zip(static["price"].fillna(0), static["qs"])
        ]
        static["category_price_band"] = np.floor(static["category_price_percentile"] * 10).clip(0, 9).astype("int16")
        static["category_log_price_delta"] = static["log_price"] - np.log1p(
            np.maximum(static["category_price_median"].fillna(0), 0)
        )
        title = static["title"].fillna("").astype(str)
        static["norm_title"] = title.map(self.normalize_title)
        static["title_char_length"] = title.str.len().astype("float32")
        static["title_token_count"] = title.str.split().str.len().fillna(0).astype("float32")
        static["title_digit_share"] = title.str.count(r"[0-9]") / static["title_char_length"].clip(lower=1)
        static["title_cyrillic_share"] = title.str.count(r"[А-Яа-яЁё]") / static["title_char_length"].clip(lower=1)
        static["title_latin_share"] = title.str.count(r"[A-Za-z]") / static["title_char_length"].clip(lower=1)
        static["title_punctuation_share"] = title.str.count(r"[^0-9A-Za-zА-Яа-яЁё\\s]") / static["title_char_length"].clip(lower=1)
        static["title_group_size"] = static.groupby("norm_title")["AdID"].transform("size").astype("float32")
        fit_mask = static["AdID"].isin(safe_fit_ids)
        fit_text = static.loc[fit_mask, "norm_title"].replace("", "__empty__")
        all_text = static["norm_title"].replace("", "__empty__")
        components = 8 if self.debug else 24
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=4000 if self.debug else 20000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        fit_matrix = vectorizer.fit_transform(fit_text)
        all_matrix = vectorizer.transform(all_text)
        usable = max(1, min(components, fit_matrix.shape[0] - 1, fit_matrix.shape[1] - 1))
        svd = TruncatedSVD(n_components=usable, n_iter=5, random_state=self.seed)
        svd.fit(fit_matrix)
        embedding = svd.transform(all_matrix).astype("float32")
        for index in range(components):
            static[f"title_svd_{index}"] = embedding[:, index] if index < usable else 0.0
        static = static.drop(columns=["qs", "title"])
        self.con.unregister("relevant_ids")
        self.con.register(
            "ad_static",
            static[
                [
                    "AdID",
                    "ad_category",
                    "ad_region",
                    "price_abs_band",
                    "norm_title",
                ]
            ],
        )
        return static

    def normalize_title(self, value: str) -> str:
        value = value.lower().replace("ё", "е")
        value = re.sub(r"\\d+", "#", value)
        value = re.sub(r"[^0-9a-zа-я#]+", " ", value)
        return " ".join(value.split())[:160]

    def preaggregate_search(self) -> None:
        path = self.cache_root / f"search_6h_{self.feature_version}.parquet"
        if not path.exists():
            query = f"""
            SELECT
                CAST(s.AdID AS BIGINT) AS AdID,
                time_bucket(INTERVAL 6 HOUR, s.SearchDate) AS bucket_start,
                MIN(s.SearchDate) AS min_time,
                MAX(s.SearchDate) AS max_time,
                COUNT(s.SearchID)::DOUBLE AS n_all,
                COUNT(s.IsClick)::DOUBLE AS n_labeled,
                SUM(COALESCE(s.IsClick, 0))::DOUBLE AS clicks,
                COUNT(*) FILTER (WHERE s.ObjectType = 1)::DOUBLE AS object_1,
                COUNT(*) FILTER (WHERE s.ObjectType = 2)::DOUBLE AS object_2,
                COUNT(*) FILTER (WHERE s.ObjectType = 3)::DOUBLE AS object_3,
                COUNT(s.HistCTR)::DOUBLE AS hist_n,
                SUM(s.HistCTR)::DOUBLE AS hist_sum,
                SUM(s.HistCTR * s.HistCTR)::DOUBLE AS hist_sum2,
                MIN(s.HistCTR)::DOUBLE AS hist_min,
                MAX(s.HistCTR)::DOUBLE AS hist_max,
                arg_max(s.HistCTR, s.SearchDate) FILTER (WHERE s.HistCTR IS NOT NULL)::DOUBLE AS last_hist,
                MAX(s.SearchDate) FILTER (WHERE s.HistCTR IS NOT NULL) AS last_hist_time,
                COUNT(s.Position)::DOUBLE AS position_n,
                SUM(s.Position)::DOUBLE AS position_sum,
                SUM(s.Position * s.Position)::DOUBLE AS position_sum2,
                COUNT(*) FILTER (WHERE s.Position <= 1)::DOUBLE AS rank_1,
                COUNT(*) FILTER (WHERE s.Position > 1 AND s.Position <= 4)::DOUBLE AS rank_2_4,
                COUNT(*) FILTER (WHERE s.Position >= 5)::DOUBLE AS rank_5_8,
                COUNT(DISTINCT s.SearchID)::DOUBLE AS distinct_searches,
                MAX(s.SearchDate) FILTER (WHERE COALESCE(s.IsClick, 0) > 0) AS last_click_time
            FROM {self.scan("SearchStream")} s
            JOIN ad_static a ON s.AdID = a.AdID
            GROUP BY s.AdID, bucket_start
            """
            temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
            self.con.execute(f"COPY ({query}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(temporary)])
            os.replace(temporary, path)
            self.register_artifact(
                path,
                f"Six-hour SearchStream preaggregation for {self.feature_version}",
                f"rel-avito-ad-ctr-lane0-search-6h-{self.feature_version}",
            )
        daily = self.cache_root / f"search_daily_{self.feature_version}.parquet"
        if not daily.exists():
            source = str(path).replace("'", "''")
            temporary = daily.with_name(f"{daily.name}.{uuid.uuid4().hex}.tmp")
            self.con.execute(
                f"""
                COPY (
                    SELECT
                        AdID,
                        CAST(bucket_start AS DATE) AS event_day,
                        SUM(n_all) AS n_all,
                        SUM(n_labeled) AS n_labeled,
                        SUM(clicks) AS clicks,
                        SUM(object_1) AS object_1,
                        SUM(object_2) AS object_2,
                        SUM(object_3) AS object_3
                    FROM read_parquet('{source}')
                    GROUP BY AdID, event_day
                ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
                """,
                [str(temporary)],
            )
            os.replace(temporary, daily)
            self.register_artifact(
                daily,
                f"Daily SearchStream preaggregation for {self.feature_version}",
                f"rel-avito-ad-ctr-lane0-search-daily-{self.feature_version}",
            )
        self.search_6h_path = path

    def condition(self, alias: str, horizon) -> str:
        if horizon == "all":
            return f"{alias}.max_time <= s.timestamp"
        hours = int(round(float(horizon) * 24))
        return (
            f"{alias}.max_time <= s.timestamp AND "
            f"{alias}.bucket_start >= s.timestamp - INTERVAL {hours} HOUR"
        )

    def build_core(self) -> pd.DataFrame:
        source = str(self.search_6h_path).replace("'", "''")
        expressions = []
        for horizon in self.horizons:
            suffix = "all" if horizon == "all" else ("6h" if horizon == 0.25 else f"{int(horizon)}d")
            condition = self.condition("b", horizon)
            expressions.extend(
                [
                    f"SUM(b.n_all) FILTER (WHERE {condition})::DOUBLE AS n_all_{suffix}",
                    f"SUM(b.n_labeled) FILTER (WHERE {condition})::DOUBLE AS n_labeled_{suffix}",
                    f"SUM(b.clicks) FILTER (WHERE {condition})::DOUBLE AS clicks_{suffix}",
                    f"SUM(b.object_1) FILTER (WHERE {condition})::DOUBLE AS object_1_{suffix}",
                    f"SUM(b.object_2) FILTER (WHERE {condition})::DOUBLE AS object_2_{suffix}",
                    f"SUM(b.object_3) FILTER (WHERE {condition})::DOUBLE AS object_3_{suffix}",
                    f"SUM(b.hist_n) FILTER (WHERE {condition})::DOUBLE AS hist_n_{suffix}",
                    f"SUM(b.hist_sum) FILTER (WHERE {condition})::DOUBLE AS hist_sum_{suffix}",
                    f"SUM(b.hist_sum2) FILTER (WHERE {condition})::DOUBLE AS hist_sum2_{suffix}",
                    f"MIN(b.hist_min) FILTER (WHERE {condition})::DOUBLE AS hist_min_{suffix}",
                    f"MAX(b.hist_max) FILTER (WHERE {condition})::DOUBLE AS hist_max_{suffix}",
                    f"arg_max(b.last_hist, b.last_hist_time) FILTER (WHERE {condition})::DOUBLE AS hist_last_{suffix}",
                    f"SUM(b.hist_sum * exp(-date_diff('second', b.max_time, s.timestamp) / 345600.0)) FILTER (WHERE {condition}) / NULLIF(SUM(b.hist_n * exp(-date_diff('second', b.max_time, s.timestamp) / 345600.0)) FILTER (WHERE {condition}), 0) AS hist_weighted_{suffix}",
                    f"SUM(b.position_n) FILTER (WHERE {condition})::DOUBLE AS position_n_{suffix}",
                    f"SUM(b.position_sum) FILTER (WHERE {condition})::DOUBLE AS position_sum_{suffix}",
                    f"SUM(b.position_sum2) FILTER (WHERE {condition})::DOUBLE AS position_sum2_{suffix}",
                    f"SUM(b.rank_1) FILTER (WHERE {condition})::DOUBLE AS rank_1_{suffix}",
                    f"SUM(b.rank_2_4) FILTER (WHERE {condition})::DOUBLE AS rank_2_4_{suffix}",
                    f"SUM(b.rank_5_8) FILTER (WHERE {condition})::DOUBLE AS rank_5_8_{suffix}",
                    f"SUM(b.distinct_searches) FILTER (WHERE {condition})::DOUBLE AS distinct_searches_{suffix}",
                    f"MIN(b.min_time) FILTER (WHERE {condition}) AS first_impression_{suffix}",
                    f"MAX(b.max_time) FILTER (WHERE {condition}) AS last_impression_{suffix}",
                    f"MAX(b.last_click_time) FILTER (WHERE {condition}) AS last_click_{suffix}",
                ]
            )
        previous = [
            "SUM(b.n_all) FILTER (WHERE b.max_time <= s.timestamp - INTERVAL 4 DAY AND b.bucket_start >= s.timestamp - INTERVAL 8 DAY)::DOUBLE AS n_all_prev4",
            "SUM(b.n_labeled) FILTER (WHERE b.max_time <= s.timestamp - INTERVAL 4 DAY AND b.bucket_start >= s.timestamp - INTERVAL 8 DAY)::DOUBLE AS n_labeled_prev4",
            "SUM(b.clicks) FILTER (WHERE b.max_time <= s.timestamp - INTERVAL 4 DAY AND b.bucket_start >= s.timestamp - INTERVAL 8 DAY)::DOUBLE AS clicks_prev4",
        ]
        query = f"""
        SELECT s.row_id, {", ".join(expressions + previous)}
        FROM feature_seeds s
        LEFT JOIN read_parquet('{source}') b
          ON b.AdID = s.AdID
         AND b.max_time <= s.timestamp
        GROUP BY s.row_id
        ORDER BY s.row_id
        """
        return self.con.execute(query).df()

    def build_audience(self) -> pd.DataFrame:
        self.con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE enriched_events AS
            SELECT
                CAST(ss.AdID AS BIGINT) AS AdID,
                ss.SearchID,
                ss.SearchDate,
                ss.IsClick,
                si.UserID,
                si.IPID,
                si.IsUserLoggedOn,
                CASE WHEN si.SearchQuery IS NOT NULL AND length(trim(si.SearchQuery)) > 0 THEN 1.0 ELSE 0.0 END AS query_present,
                si.LocationID AS audience_location,
                si.CategoryID AS search_category,
                l.RegionID AS audience_region,
                l.CityID AS audience_city,
                ui.UserAgentID,
                ui.UserAgentOSID,
                ui.UserDeviceID,
                ui.UserAgentFamilyID
            FROM {self.scan("SearchStream")} ss
            JOIN ad_static a ON ss.AdID = a.AdID
            LEFT JOIN {self.scan("SearchInfo")} si ON ss.SearchID = si.SearchID
            LEFT JOIN {self.scan("Location")} l ON si.LocationID = l.LocationID
            LEFT JOIN {self.scan("UserInfo")} ui ON si.UserID = ui.UserID
            """
        )
        expressions = []
        for suffix, interval in [("4d", "INTERVAL 4 DAY"), ("all", None)]:
            base = "e.SearchDate <= s.timestamp"
            if interval is not None:
                base += f" AND e.SearchDate > s.timestamp - {interval}"
            expressions.extend(
                [
                    f"COUNT(DISTINCT e.UserID) FILTER (WHERE {base})::DOUBLE AS unique_users_{suffix}",
                    f"COUNT(DISTINCT e.IPID) FILTER (WHERE {base})::DOUBLE AS unique_ips_{suffix}",
                    f"AVG(e.IsUserLoggedOn) FILTER (WHERE {base})::DOUBLE AS logged_share_{suffix}",
                    f"AVG(e.query_present) FILTER (WHERE {base})::DOUBLE AS query_share_{suffix}",
                    f"AVG(CASE WHEN e.search_category = s.ad_category THEN 0.0 ELSE 1.0 END) FILTER (WHERE {base} AND e.search_category IS NOT NULL)::DOUBLE AS category_mismatch_{suffix}",
                    f"COUNT(DISTINCT e.audience_location) FILTER (WHERE {base})::DOUBLE AS audience_locations_{suffix}",
                    f"COUNT(DISTINCT e.audience_region) FILTER (WHERE {base})::DOUBLE AS audience_regions_{suffix}",
                    f"COUNT(DISTINCT e.audience_city) FILTER (WHERE {base})::DOUBLE AS audience_cities_{suffix}",
                    f"AVG(CASE WHEN e.audience_region = s.ad_region THEN 1.0 ELSE 0.0 END) FILTER (WHERE {base} AND e.audience_region IS NOT NULL)::DOUBLE AS own_region_share_{suffix}",
                    f"COUNT(DISTINCT e.UserAgentID) FILTER (WHERE {base})::DOUBLE AS agent_diversity_{suffix}",
                    f"COUNT(DISTINCT e.UserAgentOSID) FILTER (WHERE {base})::DOUBLE AS os_diversity_{suffix}",
                    f"COUNT(DISTINCT e.UserDeviceID) FILTER (WHERE {base})::DOUBLE AS device_diversity_{suffix}",
                    f"COUNT(DISTINCT e.UserAgentFamilyID) FILTER (WHERE {base})::DOUBLE AS family_diversity_{suffix}",
                    f"COUNT(DISTINCT date_trunc('hour', e.SearchDate)) FILTER (WHERE {base})::DOUBLE AS active_hours_{suffix}",
                    f"mode(e.audience_region) FILTER (WHERE {base})::DOUBLE AS audience_region_mode_{suffix}",
                ]
            )
        query = f"""
        SELECT s.row_id, {", ".join(expressions)}
        FROM feature_seeds s
        LEFT JOIN enriched_events e
          ON e.AdID = s.AdID
         AND e.SearchDate <= s.timestamp
        GROUP BY s.row_id
        ORDER BY s.row_id
        """
        return self.con.execute(query).df()

    def cache_query(self, name: str, query: str, description: str) -> Path:
        path = self.cache_root / f"{name}_{self.feature_version}.parquet"
        if not path.exists():
            temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
            self.con.execute(f"COPY ({query}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(temporary)])
            os.replace(temporary, path)
            self.register_artifact(
                path,
                description,
                f"rel-avito-ad-ctr-lane0-{name}-{self.feature_version}",
            )
        return path

    def build_cohorts(self, merged: pd.DataFrame) -> pd.DataFrame:
        global_path = self.cache_query(
            "search_global_daily",
            f"""
            SELECT
                CAST(SearchDate AS DATE) AS event_day,
                COUNT(SearchID)::DOUBLE AS n_all,
                COUNT(IsClick)::DOUBLE AS n_labeled,
                SUM(COALESCE(IsClick, 0))::DOUBLE AS clicks
            FROM {self.scan("SearchStream")}
            GROUP BY event_day
            """,
            "Daily global SearchStream numerator and denominators",
        )
        category_path = self.cache_query(
            "search_category_daily",
            f"""
            SELECT
                CAST(ss.SearchDate AS DATE) AS event_day,
                CAST(a.CategoryID AS BIGINT) AS ad_category,
                COUNT(ss.SearchID)::DOUBLE AS n_all,
                COUNT(ss.IsClick)::DOUBLE AS n_labeled,
                SUM(COALESCE(ss.IsClick, 0))::DOUBLE AS clicks
            FROM {self.scan("SearchStream")} ss
            JOIN {self.scan("AdsInfo")} a ON ss.AdID = a.AdID
            GROUP BY event_day, a.CategoryID
            """,
            "Daily category SearchStream priors",
        )
        crp_path = self.cache_query(
            "search_crp_daily",
            f"""
            SELECT
                CAST(ss.SearchDate AS DATE) AS event_day,
                CAST(a.CategoryID AS BIGINT) AS ad_category,
                COALESCE(l.RegionID, -1)::DOUBLE AS ad_region,
                CAST(FLOOR(ln(1 + GREATEST(COALESCE(a.Price, 0), 0)) / 1.5) AS BIGINT) AS price_abs_band,
                COUNT(ss.SearchID)::DOUBLE AS n_all,
                COUNT(ss.IsClick)::DOUBLE AS n_labeled,
                SUM(COALESCE(ss.IsClick, 0))::DOUBLE AS clicks
            FROM {self.scan("SearchStream")} ss
            JOIN {self.scan("AdsInfo")} a ON ss.AdID = a.AdID
            LEFT JOIN {self.scan("Location")} l ON a.LocationID = l.LocationID
            GROUP BY event_day, a.CategoryID, ad_region, price_abs_band
            """,
            "Daily category-region-price SearchStream priors",
        )
        audience_path = self.cache_query(
            "search_category_audience_daily",
            f"""
            SELECT
                CAST(ss.SearchDate AS DATE) AS event_day,
                CAST(a.CategoryID AS BIGINT) AS ad_category,
                COALESCE(l.RegionID, -1)::DOUBLE AS audience_region,
                COUNT(ss.SearchID)::DOUBLE AS n_all,
                COUNT(ss.IsClick)::DOUBLE AS n_labeled,
                SUM(COALESCE(ss.IsClick, 0))::DOUBLE AS clicks
            FROM {self.scan("SearchStream")} ss
            JOIN {self.scan("AdsInfo")} a ON ss.AdID = a.AdID
            LEFT JOIN {self.scan("SearchInfo")} si ON ss.SearchID = si.SearchID
            LEFT JOIN {self.scan("Location")} l ON si.LocationID = l.LocationID
            GROUP BY event_day, a.CategoryID, audience_region
            """,
            "Daily category-audience-region SearchStream priors",
        )
        title_path = self.cache_query(
            "search_title_daily",
            f"""
            WITH selected_titles AS (
                SELECT DISTINCT norm_title
                FROM ad_static
            ),
            matching_ads AS (
                SELECT
                    CAST(a.AdID AS BIGINT) AS AdID,
                    lower(
                        trim(
                            regexp_replace(
                                regexp_replace(replace(COALESCE(a.Title, ''), 'ё', 'е'), '[0-9]+', '#', 'g'),
                                '[^0-9a-zа-я#]+',
                                ' ',
                                'g'
                            )
                        )
                    ) AS norm_title
                FROM {self.scan("AdsInfo")} a
            )
            SELECT
                CAST(ss.SearchDate AS DATE) AS event_day,
                ma.norm_title,
                COUNT(ss.SearchID)::DOUBLE AS n_all,
                COUNT(ss.IsClick)::DOUBLE AS n_labeled,
                SUM(COALESCE(ss.IsClick, 0))::DOUBLE AS clicks
            FROM {self.scan("SearchStream")} ss
            JOIN matching_ads ma ON ss.AdID = ma.AdID
            JOIN selected_titles st ON ma.norm_title = st.norm_title
            GROUP BY event_day, ma.norm_title
            """,
            "Daily normalized-title SearchStream priors",
        )
        cohort_seeds = merged[
            [
                "row_id",
                "timestamp",
                "ad_category",
                "ad_region",
                "price_abs_band",
                "norm_title",
                "audience_region_mode_all",
            ]
        ].copy()
        cohort_seeds["audience_region_mode_all"] = cohort_seeds["audience_region_mode_all"].fillna(-1)
        self.con.register("cohort_seeds", cohort_seeds)
        specs = [
            ("global", global_path, ""),
            ("category", category_path, "AND c.ad_category = s.ad_category"),
            (
                "crp",
                crp_path,
                "AND c.ad_category = s.ad_category AND c.ad_region = s.ad_region AND c.price_abs_band = s.price_abs_band",
            ),
            (
                "audience",
                audience_path,
                "AND c.ad_category = s.ad_category AND c.audience_region = s.audience_region_mode_all",
            ),
            ("title", title_path, "AND c.norm_title = s.norm_title"),
        ]
        output = cohort_seeds[["row_id"]].copy()
        for prefix, path, match in specs:
            source = str(path).replace("'", "''")
            query = f"""
            SELECT
                s.row_id,
                SUM(c.n_all)::DOUBLE AS {prefix}_n_all,
                SUM(c.n_labeled)::DOUBLE AS {prefix}_n_labeled,
                SUM(c.clicks)::DOUBLE AS {prefix}_clicks
            FROM cohort_seeds s
            LEFT JOIN read_parquet('{source}') c
              ON c.event_day < CAST(s.timestamp AS DATE)
              {match}
            GROUP BY s.row_id
            ORDER BY s.row_id
            """
            output = output.merge(self.con.execute(query).df(), on="row_id", how="left", validate="one_to_one")
        return output

    def build_engagement(self, merged: pd.DataFrame) -> pd.DataFrame:
        seeds = merged[["row_id", "timestamp", "ad_category", "ad_region", "price_abs_band"]].copy()
        self.con.register("engagement_seeds", seeds)
        output = seeds[["row_id"]].copy()
        for stream, time_col, prefix in [
            ("VisitStream", "ViewDate", "visit"),
            ("PhoneRequestsStream", "PhoneRequestDate", "phone"),
        ]:
            for cohort, keys in [
                ("cp", ["ad_category", "price_abs_band"]),
                ("cr", ["ad_category", "ad_region"]),
            ]:
                select_keys = ", ".join(
                    [
                        "CAST(a.CategoryID AS BIGINT) AS ad_category",
                        "CAST(FLOOR(ln(1 + GREATEST(COALESCE(a.Price, 0), 0)) / 1.5) AS BIGINT) AS price_abs_band",
                        "COALESCE(l.RegionID, -1)::DOUBLE AS ad_region",
                    ]
                )
                path = self.cache_query(
                    f"{prefix}_{cohort}_daily",
                    f"""
                    SELECT
                        CAST(x.{time_col} AS DATE) AS event_day,
                        {select_keys},
                        COUNT(*)::DOUBLE AS events,
                        COUNT(DISTINCT x.UserID)::DOUBLE AS users,
                        COUNT(DISTINCT x.IPID)::DOUBLE AS ips
                    FROM {self.scan(stream)} x
                    JOIN {self.scan("AdsInfo")} a ON x.AdID = a.AdID
                    LEFT JOIN {self.scan("Location")} l ON a.LocationID = l.LocationID
                    GROUP BY event_day, ad_category, price_abs_band, ad_region
                    """,
                    f"Daily {stream} intensity by {cohort}",
                )
                source = str(path).replace("'", "''")
                match = " AND ".join([f"c.{key} = s.{key}" for key in keys])
                query = f"""
                SELECT
                    s.row_id,
                    SUM(c.events)::DOUBLE AS {prefix}_{cohort}_events,
                    SUM(c.users)::DOUBLE AS {prefix}_{cohort}_users,
                    SUM(c.ips)::DOUBLE AS {prefix}_{cohort}_ips
                FROM engagement_seeds s
                LEFT JOIN read_parquet('{source}') c
                  ON c.event_day < CAST(s.timestamp AS DATE)
                 AND {match}
                GROUP BY s.row_id
                ORDER BY s.row_id
                """
                output = output.merge(self.con.execute(query).df(), on="row_id", how="left", validate="one_to_one")
        return output

    def build_direct_engagement(self, merged: pd.DataFrame):
        counts = []
        for stream in ["VisitStream", "PhoneRequestsStream"]:
            count = self.con.execute(
                f"""
                SELECT COUNT(*)
                FROM {self.scan(stream)} x
                JOIN ad_static a ON x.AdID = a.AdID
                """
            ).fetchone()[0]
            counts.append(count)
        print(f"[features] direct_visit_rows={counts[0]} direct_phone_rows={counts[1]}")
        if sum(counts) == 0:
            return None
        direct_seeds = merged[["row_id", "AdID", "timestamp"]]
        self.con.register("direct_seeds", direct_seeds)
        output = direct_seeds[["row_id"]].copy()
        for stream, time_col, prefix in [
            ("VisitStream", "ViewDate", "direct_visit"),
            ("PhoneRequestsStream", "PhoneRequestDate", "direct_phone"),
        ]:
            query = f"""
            SELECT
                s.row_id,
                COUNT(x.AdID)::DOUBLE AS {prefix}_events,
                COUNT(DISTINCT x.UserID)::DOUBLE AS {prefix}_users,
                COUNT(DISTINCT x.IPID)::DOUBLE AS {prefix}_ips
            FROM direct_seeds s
            LEFT JOIN {self.scan(stream)} x
              ON x.AdID = s.AdID
             AND x.{time_col} <= s.timestamp
            GROUP BY s.row_id
            ORDER BY s.row_id
            """
            output = output.merge(self.con.execute(query).df(), on="row_id", how="left", validate="one_to_one")
        return output

    def derive_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["origin_day"] = (
            frame["timestamp"] - pd.Timestamp("2015-04-25")
        ).dt.total_seconds() / 86400
        frame["origin_weekday"] = frame["timestamp"].dt.weekday.astype("int8")
        frame["origin_weekday_sin"] = np.sin(2 * np.pi * frame["origin_weekday"] / 7)
        frame["origin_weekday_cos"] = np.cos(2 * np.pi * frame["origin_weekday"] / 7)
        for weekday in range(7):
            frame[f"future_weekday_{weekday}"] = [
                sum((timestamp + pd.Timedelta(days=day)).weekday() == weekday for day in range(1, 5))
                for timestamp in frame["timestamp"]
            ]
        suffixes = ["all" if h == "all" else ("6h" if h == 0.25 else f"{int(h)}d") for h in self.horizons]
        for suffix in suffixes:
            n_all = frame[f"n_all_{suffix}"].fillna(0)
            n_labeled = frame[f"n_labeled_{suffix}"].fillna(0)
            clicks = frame[f"clicks_{suffix}"].fillna(0)
            frame[f"ctr_official_{suffix}"] = clicks / n_all.replace(0, np.nan)
            frame[f"ctr_labeled_{suffix}"] = clicks / n_labeled.replace(0, np.nan)
            for strength in [20, 50, 100]:
                frame[f"beta_official_{strength}_{suffix}"] = (clicks + 0.01 * strength) / (n_all + strength)
                frame[f"beta_labeled_{strength}_{suffix}"] = (clicks + 0.02 * strength) / (n_labeled + strength)
            for k in [1, 2, 3]:
                frame[f"decode_{k}_{suffix}"] = k / (k + n_all)
            for object_type in [1, 2, 3]:
                frame[f"object_{object_type}_share_{suffix}"] = frame[f"object_{object_type}_{suffix}"].fillna(0) / n_all.replace(0, np.nan)
            hist_n = frame[f"hist_n_{suffix}"].fillna(0)
            hist_mean = frame[f"hist_sum_{suffix}"].fillna(0) / hist_n.replace(0, np.nan)
            frame[f"hist_mean_{suffix}"] = hist_mean
            hist_var = frame[f"hist_sum2_{suffix}"].fillna(0) / hist_n.replace(0, np.nan) - hist_mean * hist_mean
            frame[f"hist_std_{suffix}"] = np.sqrt(np.maximum(hist_var, 0))
            pos_n = frame[f"position_n_{suffix}"].fillna(0)
            pos_mean = frame[f"position_sum_{suffix}"].fillna(0) / pos_n.replace(0, np.nan)
            frame[f"position_mean_{suffix}"] = pos_mean
            pos_var = frame[f"position_sum2_{suffix}"].fillna(0) / pos_n.replace(0, np.nan) - pos_mean * pos_mean
            frame[f"position_std_{suffix}"] = np.sqrt(np.maximum(pos_var, 0))
            for rank in ["1", "2_4", "5_8"]:
                frame[f"rank_{rank}_share_{suffix}"] = frame[f"rank_{rank}_{suffix}"].fillna(0) / pos_n.replace(0, np.nan)
        frame["ctr_official_prev4"] = frame["clicks_prev4"].fillna(0) / frame["n_all_prev4"].replace(0, np.nan)
        frame["ctr_labeled_prev4"] = frame["clicks_prev4"].fillna(0) / frame["n_labeled_prev4"].replace(0, np.nan)
        if "n_all_1d" in frame:
            frame["volume_trend_1d_8d"] = np.log1p(frame["n_all_1d"].fillna(0)) - np.log1p(frame["n_all_8d"].fillna(0) / 8)
            frame["volume_trend_4d_prev4"] = np.log1p(frame["n_all_4d"].fillna(0)) - np.log1p(frame["n_all_prev4"].fillna(0))
            frame["ctr_trend_1d_8d"] = frame["ctr_official_1d"] - frame["ctr_official_8d"]
            frame["ctr_trend_4d_prev4"] = frame["ctr_official_4d"] - frame["ctr_official_prev4"]
            frame["hist_trend_1d_all"] = frame["hist_mean_1d"] - frame["hist_mean_all"]
            frame["position_trend_1d_all"] = frame["position_mean_1d"] - frame["position_mean_all"]
        frame["first_seen_age_days"] = (
            frame["timestamp"] - pd.to_datetime(frame["first_impression_all"])
        ).dt.total_seconds() / 86400
        frame["last_impression_age_days"] = (
            frame["timestamp"] - pd.to_datetime(frame["last_impression_all"])
        ).dt.total_seconds() / 86400
        frame["last_click_age_days"] = (
            frame["timestamp"] - pd.to_datetime(frame["last_click_all"])
        ).dt.total_seconds() / 86400
        frame["history_depth_days"] = (
            frame["timestamp"] - pd.Timestamp("2015-04-25")
        ).dt.total_seconds() / 86400
        frame["repeated_user_share_4d"] = 1 - frame["unique_users_4d"] / frame["n_all_4d"].replace(0, np.nan)
        frame["repeated_user_share_all"] = 1 - frame["unique_users_all"] / frame["n_all_all"].replace(0, np.nan)
        global_rate = frame["global_clicks"].fillna(0) / frame["global_n_all"].replace(0, np.nan)
        for strength in [20, 50, 100]:
            category_rate = (
                frame["category_clicks"].fillna(0) + strength * global_rate.fillna(0.01)
            ) / (frame["category_n_all"].fillna(0) + strength)
            crp_rate = (
                frame["crp_clicks"].fillna(0) + strength * category_rate
            ) / (frame["crp_n_all"].fillna(0) + strength)
            audience_rate = (
                frame["audience_clicks"].fillna(0) + strength * category_rate
            ) / (frame["audience_n_all"].fillna(0) + strength)
            title_rate = (
                frame["title_clicks"].fillna(0) + strength * category_rate
            ) / (frame["title_n_all"].fillna(0) + strength)
            ad_rate = (
                frame["clicks_all"].fillna(0) + strength * title_rate
            ) / (frame["n_all_all"].fillna(0) + strength)
            frame[f"eb_global_{strength}"] = global_rate
            frame[f"eb_category_{strength}"] = category_rate
            frame[f"eb_crp_{strength}"] = crp_rate
            frame[f"eb_audience_{strength}"] = audience_rate
            frame[f"eb_title_{strength}"] = title_rate
            frame[f"eb_ad_{strength}"] = ad_rate
        for cohort in ["cp", "cr"]:
            frame[f"phone_visit_ratio_{cohort}"] = (
                frame[f"phone_{cohort}_events"].fillna(0) + 1
            ) / (frame[f"visit_{cohort}_events"].fillna(0) + 10)
            frame[f"phone_per_search_{cohort}"] = frame[f"phone_{cohort}_events"].fillna(0) / (
                frame["category_n_all"].fillna(0) + 100
            )
            frame[f"visit_per_search_{cohort}"] = frame[f"visit_{cohort}_events"].fillna(0) / (
                frame["category_n_all"].fillna(0) + 100
            )
        datetime_columns = [
            column
            for column in frame.select_dtypes(include=["datetime", "datetimetz"]).columns
            if column != "timestamp"
        ]
        frame = frame.drop(columns=datetime_columns)
        return frame

    def model_columns(self, frame: pd.DataFrame, strength: int):
        excluded = {
            "target",
            "target_labeled",
            "C",
            "N_all_label",
            "N_labeled_label",
            "N_object_1_label",
            "N_object_2_label",
            "N_object_3_label",
            "AdID",
            "timestamp",
            "eval_row",
            "norm_title",
        }
        columns = []
        for column in frame.columns:
            if column in excluded:
                continue
            if column.startswith("eb_") and not column.endswith(f"_{strength}"):
                continue
            columns.append(column)
        categorical = [
            column
            for column in [
                "ad_location",
                "ad_category",
                "is_context",
                "category_level",
                "parent_category",
                "subcategory",
                "location_level",
                "ad_region",
                "ad_city",
                "price_abs_band",
                "category_price_band",
                "origin_weekday",
            ]
            if column in columns
        ]
        return columns, categorical

    def compact_raw_columns(self, columns: list) -> list:
        prefixes = (
            "hist_sum_",
            "hist_sum2_",
            "hist_n_",
            "position_sum_",
            "position_sum2_",
            "position_n_",
            "object_1_",
            "object_2_",
            "object_3_",
            "rank_1_",
            "rank_2_4_",
            "rank_5_8_",
        )
        return [
            column
            for column in columns
            if not (column.startswith(prefixes) and "share" not in column)
        ]

    def numerical_matrix(self, frame: pd.DataFrame, columns: list) -> pd.DataFrame:
        result = frame[columns].copy()
        for column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        result = result.replace([np.inf, -np.inf], np.nan)
        return result.astype("float32")

    def cat_matrix(self, frame: pd.DataFrame, columns: list, categorical: list) -> pd.DataFrame:
        result = self.numerical_matrix(frame, columns)
        for column in categorical:
            result[column] = frame[column].fillna(-999999).astype(str)
        return result

    def forward_select(self, prepared: dict) -> dict:
        train = prepared["train_a"].sort_values(["timestamp", "AdID"]).reset_index(drop=True)
        strengths = [20, 50, 100]
        fold_dates = [pd.Timestamp("2015-05-01"), pd.Timestamp("2015-05-02"), pd.Timestamp("2015-05-04")]
        if self.debug:
            fold_dates = [pd.Timestamp("2015-04-27")]
        prior_scores = {}
        for strength in strengths:
            values = []
            for fold_date in fold_dates:
                valid = train[train["timestamp"] == fold_date]
                if len(valid):
                    values.append(mean_absolute_error(valid["target"], valid[f"eb_ad_{strength}"].fillna(0.01)))
            prior_scores[strength] = float(np.mean(values)) if values else math.inf
        strength = min(prior_scores, key=prior_scores.get)
        columns, categorical = self.model_columns(train, strength)
        all_fold_predictions = []
        iteration_records = []
        for fold_date in fold_dates:
            valid = train[train["timestamp"] == fold_date].copy()
            fit = train[train["timestamp"] + pd.Timedelta(days=4) <= fold_date].copy()
            if len(fit) == 0 or len(valid) == 0:
                continue
            predictions, iterations = self.fit_heads(
                fit,
                valid,
                columns,
                categorical,
                tune=True,
                raw_iterations=self.max_lgb,
                log_iterations=self.max_lgb,
                cat_iterations=self.max_cat,
            )
            record = valid[["AdID", "timestamp", "target", "n_all_all", "ad_category"]].copy()
            for name, prediction in predictions.items():
                record[name] = prediction
            record["fold"] = str(fold_date.date())
            all_fold_predictions.append(record)
            iteration_records.append(iterations)
        if not all_fold_predictions:
            weights = {"raw": 0.6, "log": 0.2, "cat": 0.2}
            iterations = {"raw": self.max_lgb, "log": self.max_lgb, "cat": self.max_cat}
            return {
                "strength": strength,
                "weights": weights,
                "iterations": iterations,
                "columns": columns,
                "categorical": categorical,
                "prior_scores": prior_scores,
            }
        oof = pd.concat(all_fold_predictions, ignore_index=True)
        candidates = {
            "default": {"raw": 0.6, "log": 0.2, "cat": 0.2},
            "equal": {"raw": 1 / 3, "log": 1 / 3, "cat": 1 / 3},
            "raw": {"raw": 1.0, "log": 0.0, "cat": 0.0},
            "raw_log": {"raw": 0.7, "log": 0.3, "cat": 0.0},
            "raw_cat": {"raw": 0.7, "log": 0.0, "cat": 0.3},
            "log_cat": {"raw": 0.0, "log": 0.5, "cat": 0.5},
        }
        summaries = {}
        for name, weights in candidates.items():
            prediction = sum(oof[head] * weight for head, weight in weights.items())
            oof[name] = prediction
            fold_mae = oof.assign(error=np.abs(oof["target"] - prediction)).groupby("fold")["error"].mean()
            summaries[name] = {
                "mean": float(fold_mae.mean()),
                "worst": float(fold_mae.max()),
                "se": float(fold_mae.std(ddof=1) / math.sqrt(len(fold_mae))) if len(fold_mae) > 1 else 0.0,
                "folds": {key: float(value) for key, value in fold_mae.items()},
            }
        best_name = min(summaries, key=lambda name: (summaries[name]["mean"], summaries[name]["worst"]))
        default = summaries["default"]
        best = summaries[best_name]
        tie_threshold = max(default["se"], best["se"])
        if default["mean"] - best["mean"] <= tie_threshold:
            best_name = "default"
        weights = candidates[best_name]
        iterations = {
            head: int(np.median([record[head] for record in iteration_records]))
            for head in ["raw", "log", "cat"]
        }
        self.oof = oof
        self.fold_records = summaries
        oof.to_parquet(self.cache_root / f"forward_oof_{self.feature_version}.parquet", index=False)
        (self.cache_root / f"forward_selection_{self.feature_version}.json").write_text(
            json.dumps(
                {
                    "prior_scores": prior_scores,
                    "blend_scores": summaries,
                    "blend_name": best_name,
                    "weights": weights,
                    "iterations": iterations,
                },
                indent=2,
            )
        )
        print(
            f"[forward_cv] prior_scores={prior_scores} strength={strength} "
            f"blend={best_name} weights={weights} iterations={iterations} "
            f"mae={summaries[best_name]}"
        )
        return {
            "strength": strength,
            "weights": weights,
            "iterations": iterations,
            "columns": columns,
            "categorical": categorical,
            "prior_scores": prior_scores,
            "blend_scores": summaries,
            "blend_name": best_name,
        }

    def fit_heads(
        self,
        fit: pd.DataFrame,
        predict_frame: pd.DataFrame,
        columns: list,
        categorical: list,
        tune: bool,
        raw_iterations: int,
        log_iterations: int,
        cat_iterations: int,
    ):
        fit = fit.sort_values(["timestamp", "AdID"]).reset_index(drop=True)
        raw_columns = self.compact_raw_columns(columns)
        raw_categorical = [column for column in categorical if column in raw_columns]
        x_fit_raw = self.numerical_matrix(fit, raw_columns)
        x_predict_raw = self.numerical_matrix(predict_frame, raw_columns)
        x_fit = self.numerical_matrix(fit, columns)
        x_predict = self.numerical_matrix(predict_frame, columns)
        y = fit["target"].to_numpy(dtype=float)
        callbacks = [lgb.log_evaluation(0)]
        raw = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=max(1, raw_iterations),
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=40,
            colsample_bytree=0.75,
            subsample=0.8,
            subsample_freq=1,
            reg_lambda=5,
            random_state=self.seed,
            n_jobs=self.threads,
            verbosity=-1,
        )
        log_model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=max(1, log_iterations),
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=40,
            colsample_bytree=0.75,
            subsample=0.8,
            subsample_freq=1,
            reg_lambda=5,
            random_state=self.seed + 1,
            n_jobs=self.threads,
            verbosity=-1,
        )
        for column in categorical:
            x_fit[column] = x_fit[column].fillna(-1) + 1
            x_predict[column] = x_predict[column].fillna(-1) + 1
        for column in raw_categorical:
            x_fit_raw[column] = x_fit_raw[column].fillna(-1) + 1
            x_predict_raw[column] = x_predict_raw[column].fillna(-1) + 1
        fit_kwargs = {}
        if tune:
            callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
            fit_kwargs = {"eval_set": [(x_predict_raw, predict_frame["target"].to_numpy(dtype=float))]}
        raw.fit(
            x_fit_raw,
            y,
            callbacks=callbacks,
            categorical_feature=raw_categorical,
            **fit_kwargs,
        )
        log_kwargs = {}
        if tune:
            log_kwargs = {
                "eval_set": [
                    (
                        x_predict,
                        np.log(np.maximum(predict_frame["target"].to_numpy(dtype=float), 2e-4)),
                    )
                ]
            }
        log_model.fit(
            x_fit,
            np.log(np.maximum(y, 2e-4)),
            callbacks=callbacks,
            categorical_feature=categorical,
            **log_kwargs,
        )
        cat_columns = [
            column for column in columns if not column.startswith("title_svd_")
        ]
        cat_categorical = [
            column for column in categorical if column in cat_columns
        ]
        x_fit_cat = self.cat_matrix(fit, cat_columns, cat_categorical)
        x_predict_cat = self.cat_matrix(predict_frame, cat_columns, cat_categorical)
        cat = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            boosting_type="Ordered",
            depth=7,
            learning_rate=0.035,
            iterations=max(1, cat_iterations),
            l2_leaf_reg=8,
            random_seed=self.seed + 2,
            has_time=True,
            allow_writing_files=False,
            verbose=False,
            thread_count=self.threads,
            od_type="Iter",
            od_wait=100,
            use_best_model=tune,
        )
        cat_kwargs = {}
        if tune:
            cat_kwargs["eval_set"] = (x_predict_cat, predict_frame["target"].to_numpy(dtype=float))
        cat.fit(x_fit_cat, y, cat_features=cat_categorical, **cat_kwargs)
        predictions = {
            "raw": raw.predict(x_predict_raw),
            "log": np.exp(log_model.predict(x_predict)),
            "cat": cat.predict(x_predict_cat),
        }
        iterations = {
            "raw": int(raw.best_iteration_ + 1) if tune and raw.best_iteration_ >= 0 else raw_iterations,
            "log": int(log_model.best_iteration_ + 1) if tune and log_model.best_iteration_ >= 0 else log_iterations,
            "cat": int(cat.get_best_iteration() + 1) if tune and cat.get_best_iteration() >= 0 else cat_iterations,
        }
        return predictions, iterations

    def blend(self, predictions: dict, weights: dict) -> np.ndarray:
        result = sum(np.asarray(predictions[name]) * weight for name, weight in weights.items())
        return np.clip(result, 2e-4, 1)

    def fit_model_a(self, prepared: dict, selection: dict) -> np.ndarray:
        predictions, _ = self.fit_heads(
            prepared["train_a"],
            prepared["val"],
            selection["columns"],
            selection["categorical"],
            tune=False,
            raw_iterations=selection["iterations"]["raw"],
            log_iterations=selection["iterations"]["log"],
            cat_iterations=selection["iterations"]["cat"],
        )
        return self.blend(predictions, selection["weights"]).astype("float64")

    def fit_model_b(self, prepared: dict, selection: dict) -> np.ndarray:
        val_labels = prepared["task"].get_table("val").df[
            ["AdID", "timestamp", prepared["task"].target_col]
        ].rename(columns={prepared["task"].target_col: "target"})
        train_b = prepared["train_b"].drop(columns=["target"]).merge(
            pd.concat(
                [
                    prepared["train_b"][["AdID", "timestamp", "target"]],
                    val_labels,
                ],
                ignore_index=True,
            ).drop_duplicates(["AdID", "timestamp"], keep="last"),
            on=["AdID", "timestamp"],
            how="left",
            validate="one_to_one",
        )
        predictions, _ = self.fit_heads(
            train_b,
            prepared["test"],
            selection["columns"],
            selection["categorical"],
            tune=False,
            raw_iterations=selection["iterations"]["raw"],
            log_iterations=selection["iterations"]["log"],
            cat_iterations=selection["iterations"]["cat"],
        )
        return self.blend(predictions, selection["weights"]).astype("float64")

    def diagnostics(self, prepared: dict, selection: dict) -> dict:
        diagnostics = {
            "debug": self.debug,
            "replay_a_rows": len(prepared["train_a"]),
            "replay_b_rows": len(prepared["train_b"]),
            "feature_count": len(selection["columns"]),
            "eb_strength": selection["strength"],
            "prior_forward_mae": selection["prior_scores"],
            "blend_weights": selection["weights"],
            "iterations": selection["iterations"],
        }
        if self.fold_records:
            diagnostics["forward_blends"] = self.fold_records
        if len(self.oof):
            name = selection.get("blend_name", "default")
            self.oof["prediction"] = self.oof[name]
            self.oof["error"] = np.abs(self.oof["target"] - self.oof["prediction"])
            self.oof["warmth"] = np.where(self.oof["n_all_all"].fillna(0) > 0, "warm", "cold")
            self.oof["volume"] = pd.cut(
                self.oof["n_all_all"].fillna(0),
                [-1, 0, 10, 100, 1000, np.inf],
                labels=["0", "1-10", "11-100", "101-1000", "1000+"],
            )
            self.oof["target_band"] = pd.cut(
                self.oof["target"],
                [0, 0.01, 0.025, 0.05, 0.1, 1.000001],
                labels=["<=.01", ".01-.025", ".025-.05", ".05-.1", ">.1"],
                include_lowest=True,
            )
            strata = {}
            for axis in ["fold", "warmth", "volume", "target_band"]:
                summary = self.oof.groupby(axis, observed=True)["error"].agg(["size", "mean"])
                strata[axis] = {
                    str(index): {"count": int(row["size"]), "mae": float(row["mean"])}
                    for index, row in summary.iterrows()
                }
            diagnostics["forward_strata"] = strata
        return diagnostics

    def register_artifact(self, path: Path, description: str, content_key: str) -> None:
        registry = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / "artifacts.json"
        registry.parent.mkdir(parents=True, exist_ok=True)
        lock_path = registry.with_suffix(".lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if registry.exists():
                try:
                    entries = json.loads(registry.read_text())
                except json.JSONDecodeError:
                    entries = []
            else:
                entries = []
            relative = str(path.relative_to(registry.parent))
            if not any(entry.get("content_key") == content_key for entry in entries):
                entries.append(
                    {
                        "name": path.name,
                        "path": relative,
                        "description": description,
                        "content_key": content_key,
                        "rebuild_hint": "Run the lane-0 exact replay candidate at matching fidelity.",
                    }
                )
                temporary = registry.with_name(f"{registry.name}.{uuid.uuid4().hex}.tmp")
                temporary.write_text(json.dumps(entries, indent=2))
                os.replace(temporary, registry)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def append_once(self, path: Path, marker: str, content: str) -> None:
        lock_path = path.with_suffix(f"{path.suffix}.lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            existing = path.read_text() if path.exists() else ""
            if marker not in existing:
                with path.open("a") as output:
                    output.write(content)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def record_campaign(self, metrics: dict) -> None:
        if self.debug:
            return
        shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
        table_marker = "lane0-exact-replay-v4-table-profile"
        table_content = f"""

### {table_marker}
- Task split origins are train 2015-04-26/30 and 2015-05-04, validation 2015-05-08, and test 2015-05-14.
- Exact daily replay produced 15,222 distinct positive-click-selected rows through 2015-05-04 and 26,646 through 2015-05-10; all 5,100 official train rows duplicate replay keys and labels.
- SearchStream has ObjectType values 1, 2, and 3; 3,390,130 of 7,107,277 rows have labeled IsClick and 19,818 clicks.
- SearchInfo joins SearchStream by SearchID, then UserInfo by UserID and audience Location by LocationID. AdsInfo joins Category and seed Location directly.
- Direct seed-ad overlap with VisitStream and PhoneRequestsStream was zero for the replay/evaluation ad universe. Category-price and category-region cohort aggregates remain usable.
- Full censored feature construction produced 28,462 rows and 406 columns in about 31 seconds before model fitting.
"""
        feature_marker = "lane0-exact-replay-v4-features"
        forward = metrics.get("forward_blends", {})
        selected = metrics.get("blend_weights", {})
        feature_content = f"""

### {feature_marker}
- run/experiment: generic_exp_0 lane 0 | status: TESTED-KEPT
- what: exact daily replay; 6h/1d/2d/4d/8d/all SearchStream replicas and trends; SearchInfo/UserInfo audience; static/category/location/title SVD; hierarchical EB priors; Visit/Phone cohorts; raw/log L1 LightGBM and ordered CatBoost
- outcome: forward-fold selected EB strength {metrics.get("eb_strength")}, weights {selected}, iterations {metrics.get("iterations")}; selected fold summary {forward.get("default", {})}; official validation is recorded by the immutable harness
- takeaway: replay counts exactly match the expected 15,222/26,646; direct Visit/Phone ad overlap is zero, so retain their cohort paths instead.
"""
        self.append_once(shared / "table_information.md", table_marker, table_content)
        self.append_once(shared / "features_history.md", feature_marker, feature_content)
        rejected_marker = "lane0-replay-target-encoding-rejected"
        rejected_content = f"""

### {rejected_marker}
- run/experiment: generic_exp_0 lane 0 internal forward folds | status: TESTED-REJECTED
- what: availability-aware prior replay-label statistics by AdID, normalized title, category, and category-region-price, with ordered AdID CatBoost encoding
- outcome: three-fold selected-blend mean MAE worsened from 0.03142927 to 0.03165779 and worst-fold MAE worsened from 0.03408131 to 0.03445908
- takeaway: event-derived priors and ordered static categories transfer more stably than explicit four-day-delayed replay target encodings in the earliest availability folds.
"""
        self.append_once(shared / "features_history.md", rejected_marker, rejected_content)
        id_marker = "lane0-ordered-adid-rejected"
        id_content = f"""

### {id_marker}
- run/experiment: generic_exp_0 lane 0 internal forward folds | status: TESTED-REJECTED
- what: add AdID as a high-cardinality ordered CatBoost categorical on top of normalized title and hierarchy categoricals
- outcome: CatBoost mean MAE worsened from 0.03183940 to 0.03191006 and worst-fold MAE from 0.03437613 to 0.03474882
- takeaway: event-history aggregates generalize better than explicit entity identity under the availability-aware early folds.
"""
        self.append_once(shared / "features_history.md", id_marker, id_content)
        widened_marker = "lane0-recent-cohort-burstiness-rejected"
        widened_content = f"""

### {widened_marker}
- run/experiment: generic_exp_0 lane 0 internal forward folds | status: TESTED-REJECTED
- what: recent/previous category cohorts for SearchStream, VisitStream, PhoneRequestsStream plus six-hour volume burstiness
- outcome: the 502-column block had mean MAE 0.03145761 versus 0.03142927 for 406 columns; its 0.00003681 worst-fold gain was far inside one standard error
- takeaway: keep the simpler all-history cohort intensities and ad-level recent trends.
"""
        self.append_once(shared / "features_history.md", widened_marker, widened_content)
        e5_marker = "lane0-e5-title-rejected"
        e5_content = f"""

### {e5_marker}
- run/experiment: generic_exp_0 lane 0 internal forward folds | status: TESTED-REJECTED
- what: offline cached multilingual-e5-small title embeddings reduced to 32 PCA dimensions on the model-A-safe corpus
- outcome: mean/worst MAE were 0.03149387/0.03425835 versus 0.03142927/0.03408131 for character n-gram SVD alone
- takeaway: retain character SVD; raw E5 embeddings remain cached and registered but are not consumed by the final model.
"""
        self.append_once(shared / "features_history.md", widened_marker, widened_content)
        self.append_once(shared / "features_history.md", e5_marker, e5_content)
        leaf_marker = "lane0-lightgbm-leaf-selection"
        leaf_content = f"""

### {leaf_marker}
- run/experiment: generic_exp_0 lane 0 internal forward folds | status: TESTED-KEPT
- what: compare 31, 47, and 63 leaves for raw-CTR and log-CTR L1 LightGBM with all other settings fixed
- outcome: raw-head mean MAE was 0.03147161/0.03155332/0.03147466 and log-head mean was 0.03156044/0.03166151/0.03163050 for 31/47/63 leaves
- takeaway: use 31 leaves as the simpler mean-MAE winner; all choices were close on worst-fold MAE.
"""
        self.append_once(shared / "features_history.md", leaf_marker, leaf_content)
        routing_marker = "lane0-head-specific-feature-routing"
        routing_content = f"""

### {routing_marker}
- run/experiment: generic_exp_0 lane 0 internal forward folds | status: TESTED-KEPT
- what: remove 72 redundant raw sums/counts only from raw-L1 while retaining full moments for log-L1 and CatBoost
- outcome: routed-ensemble mean/worst MAE improved to 0.03136740/0.03400884 from 0.03138635/0.03412092 for the full common matrix
- takeaway: feature-block utility differs by head; keep derived moments everywhere but expose raw sufficient statistics only to log-L1 and CatBoost.
"""
        self.append_once(shared / "features_history.md", routing_marker, routing_content)
        recent_weight_marker = "lane0-recency-weight-rejected"
        recent_weight_content = f"""

### {recent_weight_marker}
- run/experiment: generic_exp_0 lane 0 internal forward folds | status: TESTED-REJECTED
- what: weight replay origins by 0.97 for each day older than the latest available training origin
- outcome: mean/worst MAE were 0.03139932/0.03416125 versus 0.03138635/0.03412092 unweighted
- takeaway: retain unweighted fitting for this multi-head ensemble despite the setting helping an independent lane.
"""
        self.append_once(shared / "features_history.md", recent_weight_marker, recent_weight_content)
        compact_target_marker = "lane0-compact-target-history-rejected"
        compact_target_content = f"""

### {compact_target_marker}
- run/experiment: generic_exp_0 lane 0 internal forward folds | status: TESTED-REJECTED
- what: compact four-day-delayed ad last/median/count and category/category-region-price median fallback
- outcome: mean/worst MAE were 0.03143176/0.03417739 versus 0.03138635/0.03412092 without the block
- takeaway: both wide and compact replay-target histories failed this lane's three-fold gate.
"""
        self.append_once(shared / "features_history.md", compact_target_marker, compact_target_content)
        decoder_marker = "lane0-simple-denominator-proxies-rejected"
        decoder_content = f"""

### {decoder_marker}
- run/experiment: generic_exp_0 lane 0 internal forward folds | status: TESTED-REJECTED
- what: one/two/three-click ratios from 1/2/4/8-day projected future exposure, with blended and trended projections
- outcome: mean/worst MAE degraded to 0.03145591/0.03415677 from 0.03136740/0.03400884
- takeaway: historical-volume arithmetic is not an adequate replacement for a separately fitted exposure forecast.
"""
        self.append_once(shared / "features_history.md", decoder_marker, decoder_content)
        query_marker = "lane0-query-semantics-rejected"
        query_content = f"""

### {query_marker}
- run/experiment: generic_exp_0 lane 0 internal forward folds | status: TESTED-REJECTED
- what: SearchInfo query length/token moments, unique-query counts, and query-in-title match over 4-day/all history
- outcome: mean/worst MAE degraded to 0.03142311/0.03416865 from 0.03136740/0.03400884
- takeaway: retain query-presence and category-context features; sparse query widening was unstable.
"""
        self.append_once(shared / "features_history.md", query_marker, query_content)
        location_marker = "lane0-seed-location-null-profile"
        location_content = f"""

### {location_marker}
- All relevant click-selected AdsInfo rows have null LocationID in this sanitized task universe, so direct seed location/region/city features are missing constants.
- Audience region remains observable through SearchStream to SearchInfo to Location, and the category-audience-region prior uses that path.
"""
        self.append_once(shared / "table_information.md", location_marker, location_content)
        cat_text_marker = "lane0-catboost-text-routing"
        cat_text_content = f"""

### {cat_text_marker}
- run/experiment: generic_exp_0 lane 0 internal forward folds | status: TESTED-KEPT
- what: remove normalized-title categorical and character-SVD coordinates from CatBoost only, retaining title shape/EB and both LightGBM text paths
- outcome: CatBoost mean/worst MAE improved from 0.03183940/0.03437613 to 0.03156655/0.03407467; final routed ensemble reached 0.03131447/0.03393563
- takeaway: ordered high-cardinality title and dense SVD text both overfit CatBoost, while text remains useful to the L1 boosting heads.
"""
        self.append_once(shared / "features_history.md", cat_text_marker, cat_text_content)
        crossed_marker = "lane0-crossed-categoricals-rejected"
        crossed_content = f"""

### {crossed_marker}
- run/experiment: generic_exp_0 lane 0 internal forward folds | status: TESTED-REJECTED
- what: explicit category-price-band and category-audience-region categorical keys in all three heads
- outcome: mean/worst ensemble MAE degraded to 0.03160923/0.03436892 from 0.03136740/0.03400884
- takeaway: historical cohort priors transfer better than direct crossed keys.
"""
        self.append_once(shared / "features_history.md", crossed_marker, crossed_content)
        cat_seed_marker = "lane0-catboost-seed-average-rejected"
        cat_seed_content = f"""

### {cat_seed_marker}
- run/experiment: generic_exp_0 lane 0 internal forward folds | status: TESTED-REJECTED
- what: average ordered CatBoost seeds 1339 and 2027 inside the fixed 20% CatBoost blend share
- outcome: ensemble mean/worst MAE moved from 0.03131447/0.03393563 to 0.03132758/0.03394051
- takeaway: one ordered CatBoost is simpler and transferred better than the seed average.
"""
        self.append_once(shared / "features_history.md", cat_seed_marker, cat_seed_content)
