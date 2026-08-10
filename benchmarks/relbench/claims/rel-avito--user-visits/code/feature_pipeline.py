from __future__ import annotations

import json
import math
import os
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


WINDOWS = ((1, "d1"), (2, "d2"), (4, "d4"), (7, "d7"), (14, "d14"), (None, "all"))
HALF_LIVES = ((0.25, "h6"), (1.0, "d1"), (3.0, "d3"), (7.0, "d7"))
FEATURE_VERSION = "lane0_alltable_v15"
PREVIOUS_FEATURE_VERSION = "lane0_alltable_v13"
warnings.filterwarnings("ignore")


def database_paths() -> dict[str, str]:
    root = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"]
    paths = {p.stem: str(p) for p in (root / "db").glob("*.parquet")}
    paths.update({f"task_{p.stem}": str(p) for p in (root / "tasks" / os.environ["RELBENCH_TASK"]).glob("*.parquet")})
    return paths


def open_database() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute(f"SET threads={int(os.environ.get('OMP_NUM_THREADS', '11'))}")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute("SET memory_limit='72GB'")
    return connection


def build_seed_frame(connection: duckdb.DuckDBPyConnection, paths: dict[str, str], debug: bool) -> pd.DataFrame:
    if debug:
        origin_values = "(timestamp '2015-04-26'), (timestamp '2015-04-30')"
    else:
        origin_values = "SELECT * FROM generate_series(timestamp '2015-04-26', timestamp '2015-05-10', interval 1 day)"
    labels = connection.execute(
        f"""
        WITH origins(origin) AS ({'VALUES ' + origin_values if debug else origin_values})
        SELECT origins.origin, visits.UserID,
               COUNT(DISTINCT visits.AdID)::INTEGER AS future_distinct_ads,
               (COUNT(DISTINCT visits.AdID) > 1)::INTEGER AS label
        FROM origins
        JOIN read_parquet('{paths['VisitStream']}') visits
          ON visits.UserID IS NOT NULL
         AND visits.ViewDate > origins.origin
         AND visits.ViewDate <= origins.origin + INTERVAL 4 DAY
        GROUP BY origins.origin, visits.UserID
        ORDER BY origins.origin, visits.UserID
        """
    ).fetchdf()
    official = connection.execute(
        f"""
        SELECT timestamp AS origin, UserID, num_click::INTEGER AS label
        FROM read_parquet('{paths['task_train']}')
        ORDER BY timestamp, UserID
        """
    ).fetchdf()
    for origin in pd.to_datetime(["2015-04-26", "2015-04-30", "2015-05-04"]):
        expected = official.loc[official["origin"] == origin, ["UserID", "label"]].reset_index(drop=True)
        if origin not in set(labels["origin"]):
            reconstructed = connection.execute(
                f"""
                SELECT UserID, (COUNT(DISTINCT AdID) > 1)::INTEGER AS label
                FROM read_parquet('{paths['VisitStream']}')
                WHERE UserID IS NOT NULL
                  AND ViewDate > TIMESTAMP '{origin:%Y-%m-%d}'
                  AND ViewDate <= TIMESTAMP '{origin:%Y-%m-%d}' + INTERVAL 4 DAY
                GROUP BY UserID ORDER BY UserID
                """
            ).fetchdf()
        else:
            reconstructed = labels.loc[labels["origin"] == origin, ["UserID", "label"]].reset_index(drop=True)
        if not expected.equals(reconstructed):
            raise RuntimeError(f"Exact rolling-label verification failed at {origin:%Y-%m-%d}")
    test = connection.execute(
        f"""
        SELECT timestamp AS origin, UserID
        FROM read_parquet('{paths['task_test']}')
        """
    ).fetchdf()
    test["future_distinct_ads"] = np.nan
    test["label"] = np.nan
    test["source"] = "test"
    labels["source"] = "daily"
    additions = [labels]
    if debug:
        validation = connection.execute(
            f"SELECT timestamp AS origin, UserID FROM read_parquet('{paths['task_val']}')"
        ).fetchdf()
        validation["future_distinct_ads"] = np.nan
        validation["label"] = np.nan
        validation["source"] = "validation"
        additions.append(validation)
    additions.append(test)
    seeds = pd.concat(additions, ignore_index=True)
    seeds.insert(0, "sid", np.arange(len(seeds), dtype=np.int64))
    return seeds


def merge_block(frame: pd.DataFrame, block: pd.DataFrame) -> pd.DataFrame:
    if block.empty:
        return frame
    return frame.merge(block, on="sid", how="left", sort=False, validate="one_to_one")


def condition(days: int | None, time_col: str, event_alias: str = "e") -> str:
    if days is None:
        return "TRUE"
    return f"{event_alias}.{time_col} > s.origin - INTERVAL {days} DAY"


def decay_expressions(prefix: str, time_col: str, event_alias: str = "e") -> list[str]:
    expressions = []
    for half_life, suffix in HALF_LIVES:
        seconds = half_life * 86400.0
        expressions.append(
            f"SUM(EXP(-0.6931471805599453 * date_diff('second', {event_alias}.{time_col}, s.origin) / {seconds})) AS {prefix}_decay_{suffix}"
        )
    return expressions


def build_core_features(connection: duckdb.DuckDBPyConnection, seeds: pd.DataFrame, paths: dict[str, str]) -> pd.DataFrame:
    connection.register("seed_input", seeds[["sid", "origin", "UserID"]])
    connection.execute("CREATE OR REPLACE TEMP TABLE seeds AS SELECT * FROM seed_input")
    base = connection.execute(
        f"""
        SELECT s.sid, s.origin, s.UserID,
               u.UserAgentID, u.UserAgentOSID, u.UserDeviceID, u.UserAgentFamilyID,
               u.UserAgentID IS NULL AS missing_agent,
               u.UserAgentOSID IS NULL AS missing_os,
               u.UserDeviceID IS NULL AS missing_device,
               u.UserAgentFamilyID IS NULL AS missing_family
        FROM seeds s
        LEFT JOIN read_parquet('{paths['UserInfo']}') u USING (UserID)
        ORDER BY s.sid
        """
    ).fetchdf()
    visit_expressions = []
    for days, suffix in WINDOWS:
        filt = condition(days, "ViewDate", "v")
        visit_expressions.extend(
            [
                f"COUNT(v.UserID) FILTER (WHERE {filt}) AS visit_count_{suffix}",
                f"COUNT(DISTINCT v.AdID) FILTER (WHERE {filt}) AS visit_ads_{suffix}",
                f"COUNT(DISTINCT v.IPID) FILTER (WHERE {filt}) AS visit_ips_{suffix}",
                f"COUNT(DISTINCT CAST(v.ViewDate AS DATE)) FILTER (WHERE {filt}) AS visit_days_{suffix}",
                f"date_diff('second', MAX(v.ViewDate) FILTER (WHERE {filt}), s.origin) / 86400.0 AS visit_recency_{suffix}",
            ]
        )
    visit_expressions.extend(decay_expressions("visit", "ViewDate", "v"))
    visits = connection.execute(
        f"""
        SELECT s.sid, {', '.join(visit_expressions)}
        FROM seeds s
        LEFT JOIN read_parquet('{paths['VisitStream']}') v
          ON v.UserID = s.UserID AND v.ViewDate <= s.origin
        GROUP BY s.sid, s.origin
        ORDER BY s.sid
        """
    ).fetchdf()
    session = connection.execute(
        f"""
        WITH lagged AS (
          SELECT UserID, ViewDate,
                 CASE WHEN date_diff('second', LAG(ViewDate) OVER (PARTITION BY UserID ORDER BY ViewDate, AdID), ViewDate) > 1800
                           OR LAG(ViewDate) OVER (PARTITION BY UserID ORDER BY ViewDate, AdID) IS NULL
                      THEN 1 ELSE 0 END AS new_session
          FROM read_parquet('{paths['VisitStream']}') WHERE UserID IS NOT NULL
        ), numbered AS (
          SELECT UserID, ViewDate,
                 SUM(new_session) OVER (PARTITION BY UserID ORDER BY ViewDate ROWS UNBOUNDED PRECEDING) AS session_id
          FROM lagged
        ), per_session AS (
          SELECT s.sid, s.origin, n.session_id, COUNT(*) AS depth,
                 MIN(n.ViewDate) AS session_start, MAX(n.ViewDate) AS session_end
          FROM seeds s JOIN numbered n ON n.UserID=s.UserID AND n.ViewDate<=s.origin
          GROUP BY s.sid, s.origin, n.session_id
        )
        SELECT sid,
               COUNT(*) AS session_count_all,
               COUNT(*) FILTER (WHERE session_end > origin - INTERVAL 1 DAY) AS session_count_d1,
               COUNT(*) FILTER (WHERE session_end > origin - INTERVAL 7 DAY) AS session_count_d7,
               arg_max(depth, session_end) AS last_session_depth,
               arg_max(date_diff('second', session_start, session_end), session_end) / 60.0 AS last_session_duration_min,
               AVG(date_diff('second', session_start, session_end) / 60.0) FILTER (WHERE session_end > origin - INTERVAL 1 DAY) AS recent_session_duration_min,
               date_diff('second', MAX(session_end), origin) / 3600.0 AS time_since_session_hours
        FROM per_session GROUP BY sid, origin ORDER BY sid
        """
    ).fetchdf()
    search_expressions = []
    phone_expressions = []
    for days, suffix in WINDOWS:
        sfilt = condition(days, "SearchDate", "q")
        pfilt = condition(days, "PhoneRequestDate", "p")
        search_expressions.extend(
            [
                f"COUNT(q.SearchID) FILTER (WHERE {sfilt}) AS search_count_{suffix}",
                f"date_diff('second', MAX(q.SearchDate) FILTER (WHERE {sfilt}), s.origin) / 86400.0 AS search_recency_{suffix}",
            ]
        )
        phone_expressions.extend(
            [
                f"COUNT(p.UserID) FILTER (WHERE {pfilt}) AS phone_count_{suffix}",
                f"COUNT(DISTINCT p.AdID) FILTER (WHERE {pfilt}) AS phone_ads_{suffix}",
                f"date_diff('second', MAX(p.PhoneRequestDate) FILTER (WHERE {pfilt}), s.origin) / 86400.0 AS phone_recency_{suffix}",
            ]
        )
    search_expressions.extend(decay_expressions("search", "SearchDate", "q"))
    phone_expressions.extend(decay_expressions("phone", "PhoneRequestDate", "p"))
    searches = connection.execute(
        f"""
        SELECT s.sid, {', '.join(search_expressions)}
        FROM seeds s LEFT JOIN read_parquet('{paths['SearchInfo']}') q
          ON q.UserID=s.UserID AND q.SearchDate<=s.origin
        GROUP BY s.sid, s.origin ORDER BY s.sid
        """
    ).fetchdf()
    phones = connection.execute(
        f"""
        SELECT s.sid, {', '.join(phone_expressions)}
        FROM seeds s LEFT JOIN read_parquet('{paths['PhoneRequestsStream']}') p
          ON p.UserID=s.UserID AND p.PhoneRequestDate<=s.origin
        GROUP BY s.sid, s.origin ORDER BY s.sid
        """
    ).fetchdf()
    frame = base
    for block in (visits, session, searches, phones):
        frame = merge_block(frame, block)
    frame["origin_day_index"] = (frame["origin"] - pd.Timestamp("2015-04-25")).dt.total_seconds() / 86400.0
    frame["origin_day_of_week"] = frame["origin"].dt.dayofweek
    frame["origin_day_of_month"] = frame["origin"].dt.day
    frame["origin_is_weekend"] = (frame["origin"].dt.dayofweek >= 5).astype(np.int8)
    return frame


def create_projections(connection: duckdb.DuckDBPyConnection, paths: dict[str, str]) -> None:
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW ads_meta AS
        SELECT a.AdID, a.LocationID, a.CategoryID, a.Price, a.IsContext,
               LENGTH(COALESCE(a.Title, '')) AS title_len,
               LENGTH(regexp_replace(COALESCE(a.Title, ''), '[^0-9]', '', 'g')) AS title_digits,
               LENGTH(regexp_replace(COALESCE(a.Title, ''), '[^[:alpha:]]', '', 'g')) AS title_alpha,
               LENGTH(regexp_replace(COALESCE(a.Title, ''), '[^[:space:]]', '', 'g')) AS title_spaces,
               c.ParentCategoryID, c.SubcategoryID,
               l.RegionID, l.CityID
        FROM read_parquet('{paths['AdsInfo']}') a
        LEFT JOIN read_parquet('{paths['Category']}') c USING (CategoryID)
        LEFT JOIN read_parquet('{paths['Location']}') l USING (LocationID)
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW visit_events AS
        SELECT v.UserID, v.ViewDate, v.IPID, v.AdID,
               a.Price, a.IsContext, a.title_len, a.title_digits, a.title_alpha, a.title_spaces,
               a.CategoryID, a.ParentCategoryID, a.SubcategoryID,
               a.LocationID, a.RegionID, a.CityID,
               CASE WHEN v.IPID IS DISTINCT FROM LAG(v.IPID) OVER (PARTITION BY v.UserID ORDER BY v.ViewDate, v.AdID) THEN 1 ELSE 0 END AS ip_changed,
               CASE WHEN a.CategoryID IS DISTINCT FROM LAG(a.CategoryID) OVER (PARTITION BY v.UserID ORDER BY v.ViewDate, v.AdID) THEN 1 ELSE 0 END AS category_changed,
               CASE WHEN a.LocationID IS DISTINCT FROM LAG(a.LocationID) OVER (PARTITION BY v.UserID ORDER BY v.ViewDate, v.AdID) THEN 1 ELSE 0 END AS location_changed
        FROM read_parquet('{paths['VisitStream']}') v
        LEFT JOIN ads_meta a USING (AdID)
        WHERE v.UserID IS NOT NULL
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW search_events AS
        SELECT q.UserID, q.SearchID, q.SearchDate, q.IPID, q.IsUserLoggedOn,
               q.SearchQuery IS NOT NULL AS query_present,
               LENGTH(COALESCE(q.SearchQuery, '')) AS query_len,
               LENGTH(regexp_replace(COALESCE(q.SearchQuery, ''), '[^0-9]', '', 'g')) AS query_digits,
               LENGTH(regexp_replace(COALESCE(q.SearchQuery, ''), '[^[:alpha:]]', '', 'g')) AS query_alpha,
               LENGTH(regexp_replace(COALESCE(q.SearchQuery, ''), '[^[:space:]]', '', 'g')) AS query_spaces,
               q.CategoryID, c.ParentCategoryID, c.SubcategoryID,
               q.LocationID, l.RegionID, l.CityID,
               CASE WHEN q.CategoryID IS DISTINCT FROM LAG(q.CategoryID) OVER (PARTITION BY q.UserID ORDER BY q.SearchDate, q.SearchID) THEN 1 ELSE 0 END AS category_changed,
               CASE WHEN q.LocationID IS DISTINCT FROM LAG(q.LocationID) OVER (PARTITION BY q.UserID ORDER BY q.SearchDate, q.SearchID) THEN 1 ELSE 0 END AS location_changed,
               CASE WHEN q.IPID IS DISTINCT FROM LAG(q.IPID) OVER (PARTITION BY q.UserID ORDER BY q.SearchDate, q.SearchID) THEN 1 ELSE 0 END AS ip_changed
        FROM read_parquet('{paths['SearchInfo']}') q
        LEFT JOIN read_parquet('{paths['Category']}') c USING (CategoryID)
        LEFT JOIN read_parquet('{paths['Location']}') l USING (LocationID)
        WHERE q.UserID IS NOT NULL
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW impression_events AS
        SELECT q.UserID, x.SearchDate, x.SearchID, x.AdID, x.Position, x.ObjectType,
               x.HistCTR, x.IsClick, a.Price, a.IsContext,
               a.title_len, a.title_digits, a.title_alpha, a.title_spaces,
               a.CategoryID, a.ParentCategoryID, a.SubcategoryID,
               a.LocationID, a.RegionID, a.CityID
        FROM read_parquet('{paths['SearchStream']}') x
        JOIN read_parquet('{paths['SearchInfo']}') q USING (SearchID)
        LEFT JOIN ads_meta a USING (AdID)
        WHERE q.UserID IS NOT NULL
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW phone_events AS
        SELECT p.UserID, p.PhoneRequestDate, p.IPID, p.AdID,
               a.Price, a.IsContext, a.title_len, a.title_digits, a.title_alpha, a.title_spaces,
               a.CategoryID, a.ParentCategoryID, a.SubcategoryID,
               a.LocationID, a.RegionID, a.CityID
        FROM read_parquet('{paths['PhoneRequestsStream']}') p
        LEFT JOIN ads_meta a USING (AdID)
        WHERE p.UserID IS NOT NULL
        """
    )


def build_visit_widening(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    expressions = []
    for days, suffix in ((4, "d4"), (7, "d7"), (14, "d14"), (None, "all")):
        filt = condition(days, "ViewDate")
        for col, name in (
            ("CategoryID", "categories"), ("ParentCategoryID", "parents"), ("SubcategoryID", "subcategories"),
            ("LocationID", "locations"), ("RegionID", "regions"), ("CityID", "cities"),
        ):
            expressions.append(f"COUNT(DISTINCT e.{col}) FILTER (WHERE {filt}) AS visit_{name}_{suffix}")
        expressions.extend(
            [
                f"AVG(e.Price) FILTER (WHERE {filt}) AS visit_price_mean_{suffix}",
                f"STDDEV_POP(e.Price) FILTER (WHERE {filt}) AS visit_price_std_{suffix}",
                f"MAX(e.Price) FILTER (WHERE {filt}) AS visit_price_max_{suffix}",
                f"AVG(e.IsContext) FILTER (WHERE {filt}) AS visit_context_share_{suffix}",
                f"AVG(e.title_len) FILTER (WHERE {filt}) AS visit_title_len_mean_{suffix}",
                f"AVG(e.title_digits) FILTER (WHERE {filt}) AS visit_title_digits_mean_{suffix}",
                f"AVG(e.title_alpha) FILTER (WHERE {filt}) AS visit_title_alpha_mean_{suffix}",
                f"AVG(e.title_spaces) FILTER (WHERE {filt}) AS visit_title_spaces_mean_{suffix}",
                f"SUM(e.category_changed) FILTER (WHERE {filt}) AS visit_category_transitions_{suffix}",
                f"SUM(e.location_changed) FILTER (WHERE {filt}) AS visit_location_transitions_{suffix}",
                f"SUM(e.ip_changed) FILTER (WHERE {filt}) AS visit_ip_changes_{suffix}",
            ]
        )
    expressions.extend(
        [
            "mode(e.CategoryID) AS visit_mode_category",
            "mode(e.ParentCategoryID) AS visit_mode_parent",
            "mode(e.RegionID) AS visit_mode_region",
            "mode(e.IPID) AS visit_mode_ip",
            "arg_max(e.CategoryID, e.ViewDate) AS visit_last_category",
            "arg_max(e.RegionID, e.ViewDate) AS visit_last_region",
        ]
    )
    return connection.execute(
        f"""
        SELECT s.sid, {', '.join(expressions)}
        FROM seeds s LEFT JOIN visit_events e
          ON e.UserID=s.UserID AND e.ViewDate<=s.origin
        GROUP BY s.sid, s.origin ORDER BY s.sid
        """
    ).fetchdf()


def build_search_widening(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    expressions = []
    for days, suffix in ((4, "d4"), (7, "d7"), (14, "d14"), (None, "all")):
        filt = condition(days, "SearchDate")
        for col, name in (
            ("IPID", "ips"), ("CategoryID", "categories"), ("ParentCategoryID", "parents"),
            ("SubcategoryID", "subcategories"), ("LocationID", "locations"),
            ("RegionID", "regions"), ("CityID", "cities"),
        ):
            expressions.append(f"COUNT(DISTINCT e.{col}) FILTER (WHERE {filt}) AS search_{name}_{suffix}")
        expressions.extend(
            [
                f"AVG(e.IsUserLoggedOn) FILTER (WHERE {filt}) AS search_logged_share_{suffix}",
                f"AVG(e.query_present::INTEGER) FILTER (WHERE {filt}) AS search_query_share_{suffix}",
                f"AVG(e.query_len) FILTER (WHERE {filt} AND e.query_present) AS search_query_len_mean_{suffix}",
                f"MAX(e.query_len) FILTER (WHERE {filt}) AS search_query_len_max_{suffix}",
                f"AVG(e.query_digits) FILTER (WHERE {filt} AND e.query_present) AS search_query_digits_mean_{suffix}",
                f"AVG(e.query_alpha) FILTER (WHERE {filt} AND e.query_present) AS search_query_alpha_mean_{suffix}",
                f"AVG(e.query_spaces) FILTER (WHERE {filt} AND e.query_present) AS search_query_spaces_mean_{suffix}",
                f"SUM(e.category_changed) FILTER (WHERE {filt}) AS search_category_transitions_{suffix}",
                f"SUM(e.location_changed) FILTER (WHERE {filt}) AS search_location_transitions_{suffix}",
                f"SUM(e.ip_changed) FILTER (WHERE {filt}) AS search_ip_changes_{suffix}",
            ]
        )
    expressions.extend(
        [
            "mode(e.CategoryID) AS search_mode_category",
            "mode(e.ParentCategoryID) AS search_mode_parent",
            "mode(e.RegionID) AS search_mode_region",
            "mode(e.IPID) AS search_mode_ip",
            "arg_max(e.CategoryID, e.SearchDate) AS search_last_category",
            "arg_max(e.RegionID, e.SearchDate) AS search_last_region",
        ]
    )
    return connection.execute(
        f"""
        SELECT s.sid, {', '.join(expressions)}
        FROM seeds s LEFT JOIN search_events e
          ON e.UserID=s.UserID AND e.SearchDate<=s.origin
        GROUP BY s.sid, s.origin ORDER BY s.sid
        """
    ).fetchdf()


def build_impression_widening(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    expressions = []
    for days, suffix in ((1, "d1"), (4, "d4"), (7, "d7"), (14, "d14"), (None, "all")):
        filt = condition(days, "SearchDate")
        expressions.extend(
            [
                f"COUNT(e.AdID) FILTER (WHERE {filt}) AS impression_count_{suffix}",
                f"COUNT(DISTINCT e.AdID) FILTER (WHERE {filt}) AS impression_ads_{suffix}",
                f"AVG(e.Position) FILTER (WHERE {filt}) AS impression_position_mean_{suffix}",
                f"STDDEV_POP(e.Position) FILTER (WHERE {filt}) AS impression_position_std_{suffix}",
                f"MIN(e.Position) FILTER (WHERE {filt}) AS impression_position_min_{suffix}",
                f"COUNT(e.IsClick) FILTER (WHERE {filt}) AS impression_click_observed_{suffix}",
                f"SUM(e.IsClick) FILTER (WHERE {filt}) AS impression_clicks_{suffix}",
                f"AVG(e.IsClick) FILTER (WHERE {filt}) AS impression_click_rate_{suffix}",
                f"AVG(e.HistCTR) FILTER (WHERE {filt}) AS impression_histctr_mean_{suffix}",
                f"STDDEV_POP(e.HistCTR) FILTER (WHERE {filt}) AS impression_histctr_std_{suffix}",
                f"AVG((e.HistCTR IS NULL)::INTEGER) FILTER (WHERE {filt}) AS impression_histctr_missing_{suffix}",
                f"AVG(e.Price) FILTER (WHERE {filt}) AS impression_price_mean_{suffix}",
                f"STDDEV_POP(e.Price) FILTER (WHERE {filt}) AS impression_price_std_{suffix}",
                f"AVG(e.IsContext) FILTER (WHERE {filt}) AS impression_context_share_{suffix}",
                f"AVG(e.title_len) FILTER (WHERE {filt}) AS impression_title_len_mean_{suffix}",
                f"AVG(e.title_digits) FILTER (WHERE {filt}) AS impression_title_digits_mean_{suffix}",
                f"COUNT(DISTINCT e.CategoryID) FILTER (WHERE {filt}) AS impression_categories_{suffix}",
                f"COUNT(DISTINCT e.ParentCategoryID) FILTER (WHERE {filt}) AS impression_parents_{suffix}",
                f"COUNT(DISTINCT e.SubcategoryID) FILTER (WHERE {filt}) AS impression_subcategories_{suffix}",
                f"COUNT(DISTINCT e.LocationID) FILTER (WHERE {filt}) AS impression_locations_{suffix}",
                f"COUNT(DISTINCT e.RegionID) FILTER (WHERE {filt}) AS impression_regions_{suffix}",
                f"COUNT(DISTINCT e.CityID) FILTER (WHERE {filt}) AS impression_cities_{suffix}",
            ]
        )
    expressions.extend(decay_expressions("impression", "SearchDate"))
    expressions.extend(
        [
            "date_diff('second', MAX(e.SearchDate), s.origin) / 86400.0 AS impression_recency_all",
            "mode(e.CategoryID) AS impression_mode_category",
            "mode(e.RegionID) AS impression_mode_region",
        ]
    )
    return connection.execute(
        f"""
        SELECT s.sid, {', '.join(expressions)}
        FROM seeds s LEFT JOIN impression_events e
          ON e.UserID=s.UserID AND e.SearchDate<=s.origin
        GROUP BY s.sid, s.origin ORDER BY s.sid
        """
    ).fetchdf()


def build_phone_widening(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    expressions = []
    for days, suffix in ((4, "d4"), (7, "d7"), (14, "d14"), (None, "all")):
        filt = condition(days, "PhoneRequestDate")
        expressions.extend(
            [
                f"COUNT(DISTINCT e.IPID) FILTER (WHERE {filt}) AS phone_ips_{suffix}",
                f"AVG(e.Price) FILTER (WHERE {filt}) AS phone_price_mean_{suffix}",
                f"AVG(e.IsContext) FILTER (WHERE {filt}) AS phone_context_share_{suffix}",
                f"AVG(e.title_len) FILTER (WHERE {filt}) AS phone_title_len_mean_{suffix}",
                f"AVG(e.title_digits) FILTER (WHERE {filt}) AS phone_title_digits_mean_{suffix}",
                f"COUNT(DISTINCT e.CategoryID) FILTER (WHERE {filt}) AS phone_categories_{suffix}",
                f"COUNT(DISTINCT e.ParentCategoryID) FILTER (WHERE {filt}) AS phone_parents_{suffix}",
                f"COUNT(DISTINCT e.SubcategoryID) FILTER (WHERE {filt}) AS phone_subcategories_{suffix}",
                f"COUNT(DISTINCT e.LocationID) FILTER (WHERE {filt}) AS phone_locations_{suffix}",
                f"COUNT(DISTINCT e.RegionID) FILTER (WHERE {filt}) AS phone_regions_{suffix}",
                f"COUNT(DISTINCT e.CityID) FILTER (WHERE {filt}) AS phone_cities_{suffix}",
            ]
        )
    expressions.extend(
        [
            "mode(e.CategoryID) AS phone_mode_category",
            "mode(e.RegionID) AS phone_mode_region",
        ]
    )
    return connection.execute(
        f"""
        SELECT s.sid, {', '.join(expressions)}
        FROM seeds s LEFT JOIN phone_events e
          ON e.UserID=s.UserID AND e.PhoneRequestDate<=s.origin
        GROUP BY s.sid, s.origin ORDER BY s.sid
        """
    ).fetchdf()


def build_concentrations(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    blocks = []
    specifications = (
        ("visit", "visit_events", "ViewDate", ("CategoryID", "LocationID", "IPID")),
        ("search", "search_events", "SearchDate", ("CategoryID", "LocationID", "IPID")),
    )
    for prefix, table, time_col, columns in specifications:
        for column in columns:
            name = {"CategoryID": "category", "LocationID": "location", "IPID": "ip"}[column]
            block = connection.execute(
                f"""
                WITH counts AS (
                  SELECT s.sid, e.{column} AS value, COUNT(*)::DOUBLE AS n
                  FROM seeds s JOIN {table} e
                    ON e.UserID=s.UserID AND e.{time_col}<=s.origin
                  WHERE e.{column} IS NOT NULL
                  GROUP BY s.sid, e.{column}
                )
                SELECT sid, MAX(n) / SUM(n) AS {prefix}_{name}_dominance,
                       SUM(n*n) / (SUM(n)*SUM(n)) AS {prefix}_{name}_hhi
                FROM counts GROUP BY sid ORDER BY sid
                """
            ).fetchdf()
            blocks.append(block)
    result = pd.DataFrame({"sid": connection.execute("SELECT sid FROM seeds ORDER BY sid").fetchnumpy()["sid"]})
    for block in blocks:
        result = merge_block(result, block)
    return result


def build_ip_degrees(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.execute(
        """
        WITH origins AS (SELECT DISTINCT origin FROM seeds),
        ip_stats AS (
          SELECT o.origin, v.IPID, COUNT(DISTINCT v.UserID) AS recent_ip_user_degree,
                 COUNT(DISTINCT v.AdID) AS recent_ip_ad_degree
          FROM origins o JOIN visit_events v
            ON v.ViewDate<=o.origin AND v.ViewDate>o.origin-INTERVAL 7 DAY
          WHERE v.IPID IS NOT NULL GROUP BY o.origin, v.IPID
        ), dominant AS (
          SELECT s.sid, s.origin, mode(v.IPID) AS IPID
          FROM seeds s LEFT JOIN visit_events v
            ON v.UserID=s.UserID AND v.ViewDate<=s.origin
          GROUP BY s.sid, s.origin
        )
        SELECT d.sid, i.recent_ip_user_degree, i.recent_ip_ad_degree
        FROM dominant d LEFT JOIN ip_stats i USING (origin, IPID)
        ORDER BY d.sid
        """
    ).fetchdf()


def build_channel_overlap(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.execute(
        """
        WITH visits AS (
          SELECT UserID, AdID, MIN(ViewDate) AS first_visit
          FROM visit_events GROUP BY UserID, AdID
        ), impressions AS (
          SELECT UserID, AdID, MIN(SearchDate) AS first_impression
          FROM impression_events GROUP BY UserID, AdID
        ), overlap AS (
          SELECT v.UserID, v.AdID, v.first_visit, i.first_impression
          FROM visits v JOIN impressions i USING (UserID, AdID)
        )
        SELECT s.sid, COUNT(DISTINCT o.AdID) AS search_visit_ad_overlap
        FROM seeds s LEFT JOIN overlap o
          ON o.UserID=s.UserID AND o.first_visit<=s.origin AND o.first_impression<=s.origin
        GROUP BY s.sid ORDER BY s.sid
        """
    ).fetchdf()


def build_distinct_visit_decays(connection: duckdb.DuckDBPyConnection, paths: dict[str, str]) -> pd.DataFrame:
    expressions = []
    for half_life, suffix in HALF_LIVES:
        seconds = half_life * 86400.0
        expressions.append(
            f"SUM(EXP(-0.6931471805599453 * date_diff('second', last_visit, origin) / {seconds})) AS visit_distinct_ad_decay_{suffix}"
        )
    return connection.execute(
        f"""
        WITH last_ads AS (
          SELECT s.sid, s.origin, v.AdID, MAX(v.ViewDate) AS last_visit
          FROM seeds s JOIN read_parquet('{paths['VisitStream']}') v
            ON v.UserID=s.UserID AND v.ViewDate<=s.origin
          GROUP BY s.sid, s.origin, v.AdID
        )
        SELECT sid, {', '.join(expressions)}
        FROM last_ads GROUP BY sid, origin ORDER BY sid
        """
    ).fetchdf()


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    for prefix in ("visit", "search", "phone"):
        count_name = "count"
        for left, right in (("d1", "d2"), ("d2", "d4"), ("d4", "d7"), ("d7", "d14"), ("d14", "all")):
            a = f"{prefix}_{count_name}_{left}"
            b = f"{prefix}_{count_name}_{right}"
            if a in frame and b in frame:
                frame[f"{prefix}_momentum_{left}_{right}"] = (frame[a] + 0.5) / (frame[b] - frame[a] + 0.5)
    for prefix in ("visit", "phone", "impression"):
        for suffix in ("d4", "d7", "d14", "all"):
            count_col = f"{prefix}_count_{suffix}"
            ads_col = f"{prefix}_ads_{suffix}"
            if count_col in frame and ads_col in frame:
                frame[f"{prefix}_distinct_ad_ratio_{suffix}"] = frame[ads_col] / (frame[count_col] + 1.0)
    for suffix in ("d1", "d2", "d4", "d7", "d14", "all"):
        if f"visit_count_{suffix}" in frame and f"phone_count_{suffix}" in frame:
            frame[f"visit_to_phone_{suffix}"] = frame[f"phone_count_{suffix}"] / (frame[f"visit_count_{suffix}"] + 1.0)
        if f"impression_count_{suffix}" in frame and f"phone_count_{suffix}" in frame:
            frame[f"impression_to_phone_{suffix}"] = frame[f"phone_count_{suffix}"] / (frame[f"impression_count_{suffix}"] + 1.0)
    for left, right, name in (
        ("visit_recency_all", "search_recency_all", "visit_search_recency_gap"),
        ("visit_recency_all", "phone_recency_all", "visit_phone_recency_gap"),
        ("search_recency_all", "phone_recency_all", "search_phone_recency_gap"),
    ):
        if left in frame and right in frame:
            frame[name] = frame[left] - frame[right]
    for left, right, name in (
        ("visit_mode_category", "search_mode_category", "search_visit_category_match"),
        ("visit_mode_region", "search_mode_region", "search_visit_region_match"),
    ):
        if left in frame and right in frame:
            frame[name] = ((frame[left] == frame[right]) & frame[left].notna()).fillna(False).astype(np.int8)
    high_value = [
        "visit_count_d1", "visit_count_d4", "visit_count_d14", "visit_ads_d4", "visit_ads_all",
        "visit_recency_all", "last_session_depth", "time_since_session_hours",
        "search_count_d4", "search_count_all", "phone_count_all", "impression_count_d4",
        "impression_ads_all", "visit_categories_all", "visit_regions_all",
        "search_categories_all", "search_regions_all", "visit_to_phone_all",
        "impression_to_phone_all", "recent_ip_user_degree",
    ]
    grouped = frame.groupby("origin", sort=False)
    for column in high_value:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        frame[f"{column}_origin_pct"] = grouped[column].rank(method="average", pct=True)
        mean = grouped[column].transform("mean")
        std = grouped[column].transform("std").replace(0, np.nan)
        frame[f"{column}_origin_z"] = ((values - mean) / std).fillna(0.0)
        if "recency" in column or "time_since" in column:
            frame[f"{column}_origin_leader_gap"] = values - grouped[column].transform("min")
        else:
            frame[f"{column}_origin_leader_gap"] = grouped[column].transform("max") - values
    frame = add_history_rates(frame)
    frame = add_breadth_momentum(frame)
    return add_rate_cohort_features(frame)


def add_history_rates(frame: pd.DataFrame) -> pd.DataFrame:
    exposure = pd.to_numeric(frame["origin_day_index"], errors="coerce").clip(lower=1.0)
    bases = (
        "visit_count_all", "visit_ads_all", "visit_ips_all", "visit_days_all",
        "visit_categories_all", "visit_parents_all", "visit_subcategories_all",
        "visit_locations_all", "visit_regions_all", "visit_cities_all",
        "search_count_all", "search_ips_all", "search_categories_all", "search_parents_all",
        "search_subcategories_all", "search_locations_all", "search_regions_all", "search_cities_all",
        "impression_count_all", "impression_ads_all", "impression_click_observed_all",
        "impression_clicks_all", "impression_categories_all", "impression_parents_all",
        "impression_subcategories_all", "impression_locations_all", "impression_regions_all", "impression_cities_all",
        "phone_count_all", "phone_ads_all", "phone_ips_all", "phone_categories_all", "phone_parents_all",
        "phone_subcategories_all", "phone_locations_all", "phone_regions_all", "phone_cities_all",
    )
    for column in bases:
        if column in frame:
            frame[f"{column}_per_history_day"] = pd.to_numeric(frame[column], errors="coerce") / exposure
    return frame


def add_breadth_momentum(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "visit_count", "visit_ads", "visit_ips", "visit_categories", "visit_locations",
        "search_count", "search_ips", "search_categories", "search_locations",
        "impression_count", "impression_ads", "impression_click_observed", "impression_clicks",
        "impression_categories", "impression_locations",
        "phone_count", "phone_ads", "phone_ips", "phone_categories", "phone_locations",
    )
    for metric in metrics:
        for left, right, left_days, right_days in (
            ("d1", "d2", 1.0, 2.0),
            ("d2", "d4", 2.0, 4.0),
            ("d4", "d7", 4.0, 7.0),
            ("d7", "d14", 7.0, 14.0),
        ):
            recent = f"{metric}_{left}"
            broad = f"{metric}_{right}"
            if recent in frame and broad in frame:
                current_rate = pd.to_numeric(frame[recent], errors="coerce") / left_days
                previous_rate = (pd.to_numeric(frame[broad], errors="coerce") - pd.to_numeric(frame[recent], errors="coerce")) / (right_days - left_days)
                frame[f"{metric}_rate_momentum_{left}_{right}"] = (current_rate + 0.1) / (previous_rate + 0.1)
    return frame


def add_rate_cohort_features(frame: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "visit_ads_all_per_history_day",
        "visit_count_all_per_history_day",
        "visit_days_all_per_history_day",
    )
    grouped = frame.groupby("origin", sort=False)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        frame[f"{column}_cohort_pct"] = grouped[column].rank(method="average", pct=True)
        mean = grouped[column].transform("mean")
        std = grouped[column].transform("std").replace(0, np.nan)
        frame[f"{column}_cohort_z"] = ((values - mean) / std).fillna(0.0)
        frame[f"{column}_cohort_leader_gap"] = grouped[column].transform("max") - values
    return frame


def add_target_priors(frame: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    keys = (
        "UserAgentID", "UserAgentOSID", "UserDeviceID", "UserAgentFamilyID",
        "visit_mode_category", "visit_mode_region",
    )
    labels = seeds.loc[seeds["label"].notna(), ["sid", "origin", "label"]].merge(
        frame[["sid", *keys]], on="sid", how="left"
    )
    for key in keys:
        values = np.full(len(frame), 0.9, dtype=np.float32)
        counts = np.zeros(len(frame), dtype=np.float32)
        for origin in sorted(frame["origin"].unique()):
            eligible = labels["origin"] <= origin - pd.Timedelta(days=4)
            history = labels.loc[eligible & labels[key].notna()]
            current = frame["origin"] == origin
            if history.empty:
                continue
            prior_mean = float(labels.loc[eligible, "label"].mean())
            stats = history.groupby(key)["label"].agg(["sum", "count"])
            mapped_count = frame.loc[current, key].map(stats["count"]).fillna(0).to_numpy()
            mapped_sum = frame.loc[current, key].map(stats["sum"]).fillna(0).to_numpy()
            values[current.to_numpy()] = ((mapped_sum + 20.0 * prior_mean) / (mapped_count + 20.0)).astype(np.float32)
            counts[current.to_numpy()] = mapped_count.astype(np.float32)
        frame[f"target_prior_{key}"] = values
        frame[f"target_prior_count_{key}"] = counts
    return frame


def build_feature_frame(debug: bool) -> tuple[pd.DataFrame, dict[str, float]]:
    started = time.time()
    paths = database_paths()
    connection = open_database()
    seeds = build_seed_frame(connection, paths, debug)
    label_seconds = time.time() - started
    core_started = time.time()
    core = build_core_features(connection, seeds, paths)
    core_columns = list(core.columns)
    core_seconds = time.time() - core_started
    if not debug:
        projection_started = time.time()
        create_projections(connection, paths)
        projection_seconds = time.time() - projection_started
        widening_started = time.time()
        for builder in (
            build_visit_widening,
            build_search_widening,
            build_impression_widening,
            build_phone_widening,
            build_concentrations,
            build_ip_degrees,
            build_channel_overlap,
        ):
            core = merge_block(core, builder(connection))
        core = merge_block(core, build_distinct_visit_decays(connection, paths))
        widening_seconds = time.time() - widening_started
    else:
        projection_seconds = 0.0
        widening_seconds = 0.0
    frame = seeds.merge(core, on=["sid", "origin", "UserID"], how="left", sort=False, validate="one_to_one")
    frame = add_derived_features(frame)
    if not debug:
        frame = add_target_priors(frame, seeds)
    frame = frame.sort_values("sid").reset_index(drop=True)
    frame.attrs["core_columns"] = core_columns
    timings = {
        "labels_seconds": label_seconds,
        "core_seconds": core_seconds,
        "projection_seconds": projection_seconds,
        "widening_seconds": widening_seconds,
        "total_feature_seconds": time.time() - started,
        "rows": float(len(frame)),
        "columns": float(len(frame.columns)),
    }
    connection.close()
    return frame, timings


def register_artifact(
    cache: Path,
    path: Path,
    name: str = FEATURE_VERSION,
    description: str = "Lane 0 exact daily labels and temporally censored all-table feature matrix",
    content_key: str = FEATURE_VERSION,
) -> None:
    import fcntl

    registry = cache / "artifacts.json"
    lock_path = cache / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            items = json.loads(registry.read_text()) if registry.exists() else []
        except json.JSONDecodeError:
            items = []
        relative = str(path.relative_to(cache))
        if not any(item.get("path") == relative for item in items):
            items.append(
                {
                    "name": name,
                    "path": relative,
                    "description": description,
                    "content_key": content_key,
                    "rebuild_hint": "Run main.py full; source is the sanitized rel-avito parquet cache",
                }
            )
            temporary = registry.with_suffix(".lane0.tmp")
            temporary.write_text(json.dumps(items, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def load_or_build_feature_frame(debug: bool, cache: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    path = cache / f"{FEATURE_VERSION}.parquet"
    metadata = cache / f"{FEATURE_VERSION}.json"
    if not debug and path.exists() and metadata.exists():
        started = time.time()
        frame = pd.read_parquet(path)
        info = json.loads(metadata.read_text())
        info["cache_load_seconds"] = time.time() - started
        return frame, info
    previous_path = cache / f"{PREVIOUS_FEATURE_VERSION}.parquet"
    if not debug and previous_path.exists():
        started = time.time()
        frame = pd.read_parquet(previous_path)
        connection = open_database()
        connection.register("seed_input", frame[["sid", "origin", "UserID"]])
        connection.execute("CREATE OR REPLACE TEMP TABLE seeds AS SELECT * FROM seed_input")
        frame = merge_block(frame, build_distinct_visit_decays(connection, database_paths()))
        connection.close()
        frame = frame.drop(columns=[column for column in frame if column.startswith("target_prior_")])
        frame = add_history_rates(frame)
        frame = add_breadth_momentum(frame)
        frame = add_rate_cohort_features(frame)
        frame = add_target_priors(frame, frame[["sid", "origin", "label"]])
        info = {
            "extended_from": PREVIOUS_FEATURE_VERSION,
            "total_feature_seconds": time.time() - started,
            "rows": float(len(frame)),
            "columns": float(len(frame.columns)),
        }
        temporary = cache / f"{FEATURE_VERSION}.{os.getpid()}.tmp.parquet"
        frame.to_parquet(temporary, compression="zstd", index=False)
        os.replace(temporary, path)
        metadata.write_text(json.dumps(info, sort_keys=True))
        register_artifact(cache, path)
        return frame, info
    frame, info = build_feature_frame(debug)
    if not debug:
        temporary = cache / f"{FEATURE_VERSION}.{os.getpid()}.tmp.parquet"
        frame.to_parquet(temporary, compression="zstd", index=False)
        os.replace(temporary, path)
        metadata.write_text(json.dumps(info, sort_keys=True))
        register_artifact(cache, path)
    return frame, info
