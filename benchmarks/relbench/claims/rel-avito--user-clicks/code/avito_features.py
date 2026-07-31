import json
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


TABLES = [
    "AdsInfo",
    "Category",
    "Location",
    "PhoneRequestsStream",
    "SearchInfo",
    "SearchStream",
    "UserInfo",
    "VisitStream",
]


def elapsed(start):
    return round(time.time() - start, 2)


def open_connection():
    root = Path(os.environ["RELBENCH_CACHE_DIR"]) / "rel-avito" / "db"
    con = duckdb.connect()
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='85GB'")
    for table in TABLES:
        path = root / f"{table}.parquet"
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path.as_posix()}')"
        )
    return con


def build_episodes(con, anchors):
    anchor_df = pd.DataFrame({"anchor": pd.to_datetime(anchors)})
    con.register("episode_anchors", anchor_df)
    episodes = con.execute(
        """
        SELECT
            si.UserID,
            a.anchor AS timestamp,
            COUNT(ss.AdID) FILTER (WHERE ss.IsClick = 1.0) AS click_count,
            COUNT(DISTINCT si.SearchID) AS future_search_count,
            COUNT(ss.AdID) AS future_exposure_count
        FROM episode_anchors a
        JOIN SearchInfo si
          ON si.SearchDate > a.anchor
         AND si.SearchDate <= a.anchor + INTERVAL '4 days'
         AND si.UserID IS NOT NULL
        LEFT JOIN SearchStream ss
          ON ss.SearchID = si.SearchID
        GROUP BY si.UserID, a.anchor
        ORDER BY a.anchor, si.UserID
        """
    ).df()
    episodes["click_count"] = episodes["click_count"].astype(np.float32)
    episodes["future_search_count"] = episodes["future_search_count"].astype(
        np.float32
    )
    episodes["future_exposure_count"] = episodes[
        "future_exposure_count"
    ].astype(np.float32)
    episodes["any_click"] = (episodes["click_count"] >= 1).astype(np.int8)
    episodes["repeat_click"] = (episodes["click_count"] >= 2).astype(np.int8)
    return episodes


def verify_official_labels(episodes, train, val):
    official = pd.concat(
        [
            train[["UserID", "timestamp", "num_click"]],
            val[["UserID", "timestamp", "num_click"]],
        ],
        ignore_index=True,
    )
    derived = episodes[
        ["UserID", "timestamp", "repeat_click", "click_count"]
    ]
    checked = official.merge(
        derived,
        on=["UserID", "timestamp"],
        how="left",
        validate="one_to_one",
    )
    if checked["repeat_click"].isna().any():
        raise RuntimeError("exact-label audit failed: missing official rows")
    mismatch = (
        checked["num_click"].astype(np.int8)
        != checked["repeat_click"].astype(np.int8)
    ).sum()
    if mismatch:
        raise RuntimeError(
            f"exact-label audit failed: {int(mismatch)} label mismatches"
        )
    val_map = val.set_index(["timestamp", "UserID"])["num_click"]
    ep_index = pd.MultiIndex.from_frame(episodes[["timestamp", "UserID"]])
    replacement = val_map.reindex(ep_index)
    present = replacement.notna().to_numpy()
    episodes.loc[present, "repeat_click"] = (
        replacement[present].astype(np.int8).to_numpy()
    )
    return len(checked), int(mismatch)


def prepare_event_tables(con):
    con.execute(
        """
        CREATE TEMP TABLE search_events AS
        SELECT
            UserID,
            SearchID,
            SearchDate AS event_time,
            IPID,
            IsUserLoggedOn,
            (SearchQuery IS NOT NULL) AS has_query,
            LocationID,
            CategoryID,
            DATE_DIFF(
                'second',
                LAG(SearchDate) OVER (
                    PARTITION BY UserID ORDER BY SearchDate, SearchID
                ),
                SearchDate
            ) AS gap_seconds
        FROM SearchInfo
        WHERE UserID IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE impression_events AS
        SELECT
            si.UserID,
            ss.SearchID,
            ss.AdID,
            ss.SearchDate AS event_time,
            ss.Position,
            ss.ObjectType,
            ss.HistCTR,
            COALESCE(ss.IsClick, 0.0) AS IsClick,
            si.IsUserLoggedOn,
            (si.SearchQuery IS NOT NULL) AS has_query,
            si.CategoryID AS SearchCategoryID,
            si.LocationID AS SearchLocationID,
            ai.Price,
            ai.IsContext,
            ai.CategoryID AS AdCategoryID,
            ai.LocationID AS AdLocationID,
            c.ParentCategoryID,
            c.SubcategoryID,
            c.Level AS CategoryLevel,
            l.RegionID,
            l.CityID,
            l.Level AS LocationLevel
        FROM SearchStream ss
        JOIN SearchInfo si
          ON si.SearchID = ss.SearchID
         AND si.UserID IS NOT NULL
        LEFT JOIN AdsInfo ai
          ON ai.AdID = ss.AdID
        LEFT JOIN Category c
          ON c.CategoryID = ai.CategoryID
        LEFT JOIN Location l
          ON l.LocationID = ai.LocationID
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE visit_events AS
        SELECT
            v.UserID,
            v.AdID,
            v.ViewDate AS event_time,
            v.IPID,
            ai.Price,
            ai.IsContext,
            ai.CategoryID AS AdCategoryID,
            c.ParentCategoryID,
            c.SubcategoryID,
            c.Level AS CategoryLevel,
            l.RegionID,
            l.CityID,
            l.Level AS LocationLevel
        FROM VisitStream v
        LEFT JOIN AdsInfo ai
          ON ai.AdID = v.AdID
        LEFT JOIN Category c
          ON c.CategoryID = ai.CategoryID
        LEFT JOIN Location l
          ON l.LocationID = ai.LocationID
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE phone_events AS
        SELECT
            p.UserID,
            p.AdID,
            p.PhoneRequestDate AS event_time,
            p.IPID,
            ai.Price,
            ai.IsContext,
            ai.CategoryID AS AdCategoryID,
            c.ParentCategoryID,
            c.SubcategoryID,
            c.Level AS CategoryLevel,
            l.RegionID,
            l.CityID,
            l.Level AS LocationLevel
        FROM PhoneRequestsStream p
        LEFT JOIN AdsInfo ai
          ON ai.AdID = p.AdID
        LEFT JOIN Category c
          ON c.CategoryID = ai.CategoryID
        LEFT JOIN Location l
          ON l.LocationID = ai.LocationID
        """
    )


def _append_query_block(con, blocks, names, sql, n_rows):
    frame = con.execute(sql).df()
    columns = [column for column in frame.columns if column != "row_id"]
    block = np.full((n_rows, len(columns)), np.nan, dtype=np.float32)
    if len(frame):
        rows = frame["row_id"].to_numpy(dtype=np.int64)
        values = frame[columns].to_numpy(dtype=np.float32)
        block[rows] = values
    blocks.append(block)
    names.extend(columns)


def _history_guard(column, debug):
    if debug:
        return f" AND {column} > s.timestamp - INTERVAL '7 days'"
    return ""


def build_core_features(con, seeds, debug=False):
    start = time.time()
    seed_frame = seeds[["row_id", "UserID", "timestamp"]].copy()
    con.register("feature_seeds", seed_frame)
    prepare_event_tables(con)
    print(
        f"[features] prepared relational event tables elapsed={elapsed(start)}s",
        flush=True,
    )
    blocks = []
    names = []
    n_rows = len(seeds)
    static_sql = """
        SELECT
            s.row_id,
            s.UserID AS user_id,
            u.UserAgentID AS user_agent_id,
            u.UserAgentOSID AS user_os_id,
            u.UserDeviceID AS user_device_id,
            u.UserAgentFamilyID AS user_family_id,
            (u.UserAgentID IS NULL)::INTEGER AS user_agent_cold,
            (u.UserAgentOSID IS NULL)::INTEGER AS user_os_cold,
            (u.UserDeviceID IS NULL)::INTEGER AS user_device_cold,
            (u.UserAgentFamilyID IS NULL)::INTEGER AS user_family_cold,
            DATE_DIFF('second', TIMESTAMP '2015-04-25', s.timestamp)
                / 86400.0 AS history_days,
            EXTRACT(DOW FROM s.timestamp) AS anchor_dow,
            SIN(2 * PI() * EXTRACT(DOW FROM s.timestamp) / 7.0)
                AS anchor_dow_sin,
            COS(2 * PI() * EXTRACT(DOW FROM s.timestamp) / 7.0)
                AS anchor_dow_cos
        FROM feature_seeds s
        LEFT JOIN UserInfo u USING (UserID)
    """
    _append_query_block(con, blocks, names, static_sql, n_rows)
    search_guard = _history_guard("e.event_time", debug)
    search_sql = f"""
        SELECT
            s.row_id,
            COUNT(e.SearchID) AS search_n_all,
            COUNT(e.SearchID) FILTER (
                WHERE e.event_time > s.timestamp - INTERVAL '1 day'
            ) AS search_n_1d,
            COUNT(e.SearchID) FILTER (
                WHERE e.event_time > s.timestamp - INTERVAL '2 days'
            ) AS search_n_2d,
            COUNT(e.SearchID) FILTER (
                WHERE e.event_time > s.timestamp - INTERVAL '4 days'
            ) AS search_n_4d,
            COUNT(e.SearchID) FILTER (
                WHERE e.event_time > s.timestamp - INTERVAL '7 days'
            ) AS search_n_7d,
            COUNT(e.SearchID) FILTER (
                WHERE e.event_time > s.timestamp - INTERVAL '14 days'
            ) AS search_n_14d,
            SUM(EXP(-DATE_DIFF('second', e.event_time, s.timestamp)
                / 86400.0)) AS search_decay_1d,
            SUM(EXP(-DATE_DIFF('second', e.event_time, s.timestamp)
                / 259200.0)) AS search_decay_3d,
            SUM(EXP(-DATE_DIFF('second', e.event_time, s.timestamp)
                / 604800.0)) AS search_decay_7d,
            MIN(DATE_DIFF('second', e.event_time, s.timestamp))
                / 3600.0 AS search_recency_hours,
            MAX(DATE_DIFF('second', e.event_time, s.timestamp))
                / 86400.0 AS search_history_span_days,
            COUNT(DISTINCT CAST(e.event_time AS DATE)) AS search_active_days,
            COUNT(DISTINCT DATE_TRUNC('hour', e.event_time))
                AS search_hour_sessions,
            COUNT(DISTINCT e.IPID) AS search_distinct_ip,
            COUNT(DISTINCT e.CategoryID) AS search_distinct_category,
            COUNT(DISTINCT e.LocationID) AS search_distinct_location,
            AVG(e.IsUserLoggedOn) AS search_login_rate,
            AVG(e.has_query::INTEGER) AS search_query_rate,
            AVG(e.gap_seconds) FILTER (
                WHERE e.event_time > s.timestamp - INTERVAL '7 days'
                  AND e.gap_seconds BETWEEN 0 AND 604800
            ) / 3600.0 AS search_gap_mean_hours_7d,
            STDDEV_SAMP(e.gap_seconds) FILTER (
                WHERE e.event_time > s.timestamp - INTERVAL '7 days'
                  AND e.gap_seconds BETWEEN 0 AND 604800
            ) / 3600.0 AS search_gap_std_hours_7d,
            MODE(e.CategoryID) AS search_mode_category,
            MODE(e.LocationID) AS search_mode_location,
            ARG_MAX(e.CategoryID, e.event_time) AS search_last_category,
            ARG_MAX(e.LocationID, e.event_time) AS search_last_location
        FROM feature_seeds s
        LEFT JOIN search_events e
          ON e.UserID = s.UserID
         AND e.event_time <= s.timestamp
         {search_guard}
        GROUP BY s.row_id
    """
    _append_query_block(con, blocks, names, search_sql, n_rows)
    burst_sql = f"""
        WITH hourly AS (
            SELECT
                s.row_id,
                DATE_TRUNC('hour', e.event_time) AS event_hour,
                COUNT(*) AS hourly_count
            FROM feature_seeds s
            JOIN search_events e
              ON e.UserID = s.UserID
             AND e.event_time <= s.timestamp
             AND e.event_time > s.timestamp - INTERVAL '7 days'
             {search_guard}
            GROUP BY s.row_id, event_hour
        )
        SELECT row_id, MAX(hourly_count) AS search_burst_7d
        FROM hourly
        GROUP BY row_id
    """
    _append_query_block(con, blocks, names, burst_sql, n_rows)
    impression_guard = _history_guard("e.event_time", debug)
    impression_sql = f"""
        SELECT
            s.row_id,
            COUNT(e.AdID) AS impression_n_all,
            COUNT(e.AdID) FILTER (
                WHERE e.event_time > s.timestamp - INTERVAL '1 day'
            ) AS impression_n_1d,
            COUNT(e.AdID) FILTER (
                WHERE e.event_time > s.timestamp - INTERVAL '2 days'
            ) AS impression_n_2d,
            COUNT(e.AdID) FILTER (
                WHERE e.event_time > s.timestamp - INTERVAL '4 days'
            ) AS impression_n_4d,
            COUNT(e.AdID) FILTER (
                WHERE e.event_time > s.timestamp - INTERVAL '7 days'
            ) AS impression_n_7d,
            COUNT(e.AdID) FILTER (
                WHERE e.event_time > s.timestamp - INTERVAL '14 days'
            ) AS impression_n_14d,
            SUM((e.IsClick = 1.0)::INTEGER) AS click_n_all,
            SUM((e.IsClick = 1.0)::INTEGER) FILTER (
                WHERE e.event_time > s.timestamp - INTERVAL '1 day'
            ) AS click_n_1d,
            SUM((e.IsClick = 1.0)::INTEGER) FILTER (
                WHERE e.event_time > s.timestamp - INTERVAL '2 days'
            ) AS click_n_2d,
            SUM((e.IsClick = 1.0)::INTEGER) FILTER (
                WHERE e.event_time > s.timestamp - INTERVAL '4 days'
            ) AS click_n_4d,
            SUM((e.IsClick = 1.0)::INTEGER) FILTER (
                WHERE e.event_time > s.timestamp - INTERVAL '7 days'
            ) AS click_n_7d,
            SUM((e.IsClick = 1.0)::INTEGER) FILTER (
                WHERE e.event_time > s.timestamp - INTERVAL '14 days'
            ) AS click_n_14d,
            SUM(EXP(-DATE_DIFF('second', e.event_time, s.timestamp)
                / 86400.0)) AS impression_decay_1d,
            SUM(EXP(-DATE_DIFF('second', e.event_time, s.timestamp)
                / 259200.0)) AS impression_decay_3d,
            SUM(EXP(-DATE_DIFF('second', e.event_time, s.timestamp)
                / 604800.0)) AS impression_decay_7d,
            SUM((e.IsClick = 1.0)::INTEGER
                * EXP(-DATE_DIFF('second', e.event_time, s.timestamp)
                / 86400.0)) AS click_decay_1d,
            SUM((e.IsClick = 1.0)::INTEGER
                * EXP(-DATE_DIFF('second', e.event_time, s.timestamp)
                / 259200.0)) AS click_decay_3d,
            SUM((e.IsClick = 1.0)::INTEGER
                * EXP(-DATE_DIFF('second', e.event_time, s.timestamp)
                / 604800.0)) AS click_decay_7d,
            MIN(DATE_DIFF('second', e.event_time, s.timestamp))
                / 3600.0 AS impression_recency_hours,
            MIN(DATE_DIFF('second', e.event_time, s.timestamp)) FILTER (
                WHERE e.IsClick = 1.0
            ) / 3600.0 AS click_recency_hours,
            COUNT(DISTINCT e.AdID) AS impression_distinct_ads,
            COUNT(DISTINCT e.SearchID) AS impression_distinct_searches,
            COUNT(DISTINCT CAST(e.event_time AS DATE))
                AS impression_active_days,
            AVG(e.Position) AS position_mean,
            STDDEV_SAMP(e.Position) AS position_std,
            MIN(e.Position) AS position_min,
            AVG(e.Position) FILTER (
                WHERE e.IsClick = 1.0
            ) AS clicked_position_mean,
            MIN(e.Position) FILTER (
                WHERE e.IsClick = 1.0
            ) AS clicked_position_min,
            AVG(e.HistCTR) AS histctr_mean,
            STDDEV_SAMP(e.HistCTR) AS histctr_std,
            MAX(e.HistCTR) AS histctr_max,
            AVG(e.HistCTR) FILTER (
                WHERE e.IsClick = 1.0
            ) AS clicked_histctr_mean,
            AVG(e.ObjectType) AS object_type_mean,
            AVG(e.IsUserLoggedOn) AS impression_login_rate,
            AVG(e.has_query::INTEGER) AS impression_query_rate,
            AVG(LN(GREATEST(COALESCE(e.Price, 0), 0) + 1))
                AS impression_log_price_mean,
            STDDEV_SAMP(LN(GREATEST(COALESCE(e.Price, 0), 0) + 1))
                AS impression_log_price_std,
            MEDIAN(LN(GREATEST(COALESCE(e.Price, 0), 0) + 1))
                AS impression_log_price_median,
            MAX(LN(GREATEST(COALESCE(e.Price, 0), 0) + 1))
                AS impression_log_price_max,
            AVG(e.IsContext) AS impression_context_rate,
            AVG(e.CategoryLevel) AS impression_category_level,
            AVG(e.LocationLevel) AS impression_location_level,
            COUNT(DISTINCT e.AdCategoryID) AS impression_distinct_category,
            COUNT(DISTINCT e.ParentCategoryID) AS impression_distinct_parent,
            COUNT(DISTINCT e.CityID) AS impression_distinct_city,
            AVG(LN(GREATEST(COALESCE(e.Price, 0), 0) + 1)) FILTER (
                WHERE e.IsClick = 1.0
            ) AS click_log_price_mean,
            AVG(e.IsContext) FILTER (
                WHERE e.IsClick = 1.0
            ) AS click_context_rate,
            COUNT(DISTINCT e.AdID) FILTER (
                WHERE e.IsClick = 1.0
            ) AS click_distinct_ads,
            COUNT(DISTINCT e.AdCategoryID) FILTER (
                WHERE e.IsClick = 1.0
            ) AS click_distinct_category,
            MODE(e.AdCategoryID) FILTER (
                WHERE e.IsClick = 1.0
            ) AS click_mode_category,
            MODE(e.CityID) FILTER (
                WHERE e.IsClick = 1.0
            ) AS click_mode_city,
            ARG_MAX(e.AdCategoryID, e.event_time) FILTER (
                WHERE e.IsClick = 1.0
            ) AS click_last_category,
            ARG_MAX(e.CityID, e.event_time) FILTER (
                WHERE e.IsClick = 1.0
            ) AS click_last_city
        FROM feature_seeds s
        LEFT JOIN impression_events e
          ON e.UserID = s.UserID
         AND e.event_time <= s.timestamp
         {impression_guard}
        GROUP BY s.row_id
    """
    _append_query_block(con, blocks, names, impression_sql, n_rows)
    search_click_sql = f"""
        WITH per_search AS (
            SELECT
                s.row_id,
                s.timestamp AS anchor,
                e.SearchID,
                MAX(e.event_time) AS event_time,
                COUNT(e.AdID) AS exposures,
                SUM((e.IsClick = 1.0)::INTEGER) AS clicks
            FROM feature_seeds s
            JOIN impression_events e
              ON e.UserID = s.UserID
             AND e.event_time <= s.timestamp
             {impression_guard}
            GROUP BY s.row_id, s.timestamp, e.SearchID
        ),
        per_day AS (
            SELECT
                row_id,
                CAST(event_time AS DATE) AS event_day,
                SUM(exposures) AS exposures,
                SUM(clicks) AS clicks
            FROM per_search
            GROUP BY row_id, event_day
        ),
        search_stats AS (
            SELECT
                row_id,
                COUNT(*) AS historical_searches_with_stream,
                SUM((clicks >= 1)::INTEGER) AS historical_clicked_searches,
                SUM((clicks >= 2)::INTEGER)
                    AS historical_repeat_clicked_searches,
                SUM((clicks >= 1)::INTEGER) FILTER (
                    WHERE event_time > anchor - INTERVAL '4 days'
                ) AS clicked_searches_4d,
                SUM((clicks >= 2)::INTEGER) FILTER (
                    WHERE event_time > anchor - INTERVAL '4 days'
                ) AS repeat_clicked_searches_4d,
                AVG(clicks) AS clicks_per_search_mean,
                STDDEV_SAMP(clicks) AS clicks_per_search_std,
                MAX(clicks) AS clicks_per_search_max,
                AVG(exposures) AS exposures_per_search_mean,
                MAX(exposures) AS exposures_per_search_max
            FROM per_search
            GROUP BY row_id, anchor
        ),
        day_stats AS (
            SELECT
                row_id,
                MAX(clicks) AS click_burst_day,
                MAX(exposures) AS impression_burst_day,
                STDDEV_SAMP(clicks) AS click_daily_std,
                STDDEV_SAMP(exposures) AS impression_daily_std,
                SUM((clicks > 0)::INTEGER) AS click_active_days
            FROM per_day
            GROUP BY row_id
        )
        SELECT search_stats.*, day_stats.* EXCLUDE (row_id)
        FROM search_stats
        JOIN day_stats USING (row_id)
    """
    _append_query_block(con, blocks, names, search_click_sql, n_rows)
    for table, prefix in [
        ("visit_events", "visit"),
        ("phone_events", "phone"),
    ]:
        guard = _history_guard("e.event_time", debug)
        event_sql = f"""
            SELECT
                s.row_id,
                COUNT(e.AdID) AS {prefix}_n_all,
                COUNT(e.AdID) FILTER (
                    WHERE e.event_time > s.timestamp - INTERVAL '1 day'
                ) AS {prefix}_n_1d,
                COUNT(e.AdID) FILTER (
                    WHERE e.event_time > s.timestamp - INTERVAL '2 days'
                ) AS {prefix}_n_2d,
                COUNT(e.AdID) FILTER (
                    WHERE e.event_time > s.timestamp - INTERVAL '4 days'
                ) AS {prefix}_n_4d,
                COUNT(e.AdID) FILTER (
                    WHERE e.event_time > s.timestamp - INTERVAL '7 days'
                ) AS {prefix}_n_7d,
                COUNT(e.AdID) FILTER (
                    WHERE e.event_time > s.timestamp - INTERVAL '14 days'
                ) AS {prefix}_n_14d,
                SUM(EXP(-DATE_DIFF('second', e.event_time, s.timestamp)
                    / 86400.0)) AS {prefix}_decay_1d,
                SUM(EXP(-DATE_DIFF('second', e.event_time, s.timestamp)
                    / 259200.0)) AS {prefix}_decay_3d,
                SUM(EXP(-DATE_DIFF('second', e.event_time, s.timestamp)
                    / 604800.0)) AS {prefix}_decay_7d,
                MIN(DATE_DIFF('second', e.event_time, s.timestamp))
                    / 3600.0 AS {prefix}_recency_hours,
                COUNT(DISTINCT CAST(e.event_time AS DATE))
                    AS {prefix}_active_days,
                COUNT(DISTINCT e.AdID) AS {prefix}_distinct_ads,
                COUNT(DISTINCT e.IPID) AS {prefix}_distinct_ip,
                AVG(LN(GREATEST(COALESCE(e.Price, 0), 0) + 1))
                    AS {prefix}_log_price_mean,
                STDDEV_SAMP(LN(GREATEST(COALESCE(e.Price, 0), 0) + 1))
                    AS {prefix}_log_price_std,
                AVG(e.IsContext) AS {prefix}_context_rate,
                AVG(e.CategoryLevel) AS {prefix}_category_level,
                AVG(e.LocationLevel) AS {prefix}_location_level,
                COUNT(DISTINCT e.AdCategoryID)
                    AS {prefix}_distinct_category,
                COUNT(DISTINCT e.ParentCategoryID)
                    AS {prefix}_distinct_parent,
                COUNT(DISTINCT e.CityID) AS {prefix}_distinct_city,
                MODE(e.AdCategoryID) AS {prefix}_mode_category,
                MODE(e.CityID) AS {prefix}_mode_city,
                ARG_MAX(e.AdCategoryID, e.event_time)
                    AS {prefix}_last_category,
                ARG_MAX(e.CityID, e.event_time) AS {prefix}_last_city
            FROM feature_seeds s
            LEFT JOIN {table} e
              ON e.UserID = s.UserID
             AND e.event_time <= s.timestamp
             {guard}
            GROUP BY s.row_id
        """
        _append_query_block(con, blocks, names, event_sql, n_rows)
    core = np.concatenate(blocks, axis=1)
    name_to_index = {name: index for index, name in enumerate(names)}
    context = pd.DataFrame(
        {
            "row_id": seeds["row_id"].to_numpy(),
            "UserID": seeds["UserID"].to_numpy(),
            "timestamp": seeds["timestamp"].to_numpy(),
            "agent": core[:, name_to_index["user_agent_id"]],
            "device": core[:, name_to_index["user_device_id"]],
            "os": core[:, name_to_index["user_os_id"]],
            "family": core[:, name_to_index["user_family_id"]],
            "category": core[:, name_to_index["search_last_category"]],
            "location": core[:, name_to_index["search_last_location"]],
        }
    )
    con.register("seed_context", context)
    cohort_guard = (
        " AND e.event_time > a.timestamp - INTERVAL '7 days'" if debug else ""
    )
    cohort_sql = f"""
        WITH anchors AS (
            SELECT DISTINCT timestamp FROM feature_seeds
        ),
        base AS (
            SELECT
                a.timestamp,
                e.IsClick,
                u.UserAgentID AS agent,
                u.UserDeviceID AS device,
                u.UserAgentOSID AS os,
                u.UserAgentFamilyID AS family,
                e.SearchCategoryID AS category,
                e.SearchLocationID AS location
            FROM anchors a
            JOIN impression_events e
              ON e.event_time <= a.timestamp
             {cohort_guard}
            LEFT JOIN UserInfo u USING (UserID)
        ),
        global_stats AS (
            SELECT
                timestamp,
                SUM((IsClick = 1.0)::INTEGER) AS clicks,
                COUNT(*) AS exposures
            FROM base
            GROUP BY timestamp
        ),
        device_stats AS (
            SELECT timestamp, device AS key,
                   SUM((IsClick = 1.0)::INTEGER) AS clicks,
                   COUNT(*) AS exposures
            FROM base GROUP BY timestamp, device
        ),
        agent_stats AS (
            SELECT timestamp, agent AS key,
                   SUM((IsClick = 1.0)::INTEGER) AS clicks,
                   COUNT(*) AS exposures
            FROM base GROUP BY timestamp, agent
        ),
        os_stats AS (
            SELECT timestamp, os AS key,
                   SUM((IsClick = 1.0)::INTEGER) AS clicks,
                   COUNT(*) AS exposures
            FROM base GROUP BY timestamp, os
        ),
        family_stats AS (
            SELECT timestamp, family AS key,
                   SUM((IsClick = 1.0)::INTEGER) AS clicks,
                   COUNT(*) AS exposures
            FROM base GROUP BY timestamp, family
        ),
        category_stats AS (
            SELECT timestamp, category AS key,
                   SUM((IsClick = 1.0)::INTEGER) AS clicks,
                   COUNT(*) AS exposures
            FROM base GROUP BY timestamp, category
        ),
        location_stats AS (
            SELECT timestamp, location AS key,
                   SUM((IsClick = 1.0)::INTEGER) AS clicks,
                   COUNT(*) AS exposures
            FROM base GROUP BY timestamp, location
        )
        SELECT
            c.row_id,
            (g.clicks + 1.0) / (g.exposures + 200.0)
                AS global_historical_ctr,
            (d.clicks + 20.0 * (g.clicks + 1.0)
                / (g.exposures + 200.0)) / (d.exposures + 20.0)
                AS device_smoothed_ctr,
            LN(d.exposures + 1.0) AS device_cohort_log_exposure,
            (ag.clicks + 20.0 * (g.clicks + 1.0)
                / (g.exposures + 200.0)) / (ag.exposures + 20.0)
                AS agent_smoothed_ctr,
            LN(ag.exposures + 1.0) AS agent_cohort_log_exposure,
            (o.clicks + 20.0 * (g.clicks + 1.0)
                / (g.exposures + 200.0)) / (o.exposures + 20.0)
                AS os_smoothed_ctr,
            LN(o.exposures + 1.0) AS os_cohort_log_exposure,
            (f.clicks + 20.0 * (g.clicks + 1.0)
                / (g.exposures + 200.0)) / (f.exposures + 20.0)
                AS family_smoothed_ctr,
            LN(f.exposures + 1.0) AS family_cohort_log_exposure,
            (k.clicks + 30.0 * (g.clicks + 1.0)
                / (g.exposures + 200.0)) / (k.exposures + 30.0)
                AS category_smoothed_ctr,
            LN(k.exposures + 1.0) AS category_cohort_log_exposure,
            (l.clicks + 30.0 * (g.clicks + 1.0)
                / (g.exposures + 200.0)) / (l.exposures + 30.0)
                AS location_smoothed_ctr,
            LN(l.exposures + 1.0) AS location_cohort_log_exposure
        FROM seed_context c
        LEFT JOIN global_stats g USING (timestamp)
        LEFT JOIN device_stats d
          ON d.timestamp = c.timestamp AND d.key = c.device
        LEFT JOIN agent_stats ag
          ON ag.timestamp = c.timestamp AND ag.key = c.agent
        LEFT JOIN os_stats o
          ON o.timestamp = c.timestamp AND o.key = c.os
        LEFT JOIN family_stats f
          ON f.timestamp = c.timestamp AND f.key = c.family
        LEFT JOIN category_stats k
          ON k.timestamp = c.timestamp AND k.key = c.category
        LEFT JOIN location_stats l
          ON l.timestamp = c.timestamp AND l.key = c.location
    """
    _append_query_block(con, blocks, names, cohort_sql, n_rows)
    core = np.concatenate(blocks, axis=1)
    index = {name: column for column, name in enumerate(names)}

    def add_feature(name, value):
        nonlocal core
        core = np.column_stack((core, np.asarray(value, dtype=np.float32)))
        names.append(name)

    def col(name):
        return np.nan_to_num(core[:, index[name]], nan=0.0)

    def ratio(numerator, denominator, smooth=1.0):
        return (numerator + smooth) / (denominator + smooth)

    add_feature(
        "personal_smoothed_ctr",
        (col("click_n_all") + 1.0) / (col("impression_n_all") + 100.0),
    )
    add_feature(
        "personal_smoothed_ctr_7d",
        (col("click_n_7d") + 0.2) / (col("impression_n_7d") + 20.0),
    )
    add_feature(
        "search_trend_1d_7d",
        ratio(col("search_n_1d"), col("search_n_7d") / 7.0),
    )
    add_feature(
        "impression_trend_1d_7d",
        ratio(col("impression_n_1d"), col("impression_n_7d") / 7.0),
    )
    add_feature(
        "click_trend_1d_7d",
        ratio(col("click_n_1d"), col("click_n_7d") / 7.0),
    )
    add_feature(
        "visit_trend_1d_7d",
        ratio(col("visit_n_1d"), col("visit_n_7d") / 7.0),
    )
    add_feature(
        "phone_trend_1d_7d",
        ratio(col("phone_n_1d"), col("phone_n_7d") / 7.0),
    )
    add_feature(
        "ads_per_search",
        ratio(col("impression_n_all"), col("impression_distinct_searches")),
    )
    add_feature(
        "exposure_to_click",
        ratio(col("click_n_all"), col("impression_n_all"), 2.0),
    )
    add_feature(
        "search_to_view",
        ratio(col("visit_n_all"), col("search_n_all"), 2.0),
    )
    add_feature(
        "view_to_phone",
        ratio(col("phone_n_all"), col("visit_n_all"), 2.0),
    )
    add_feature(
        "search_to_phone",
        ratio(col("phone_n_all"), col("search_n_all"), 2.0),
    )
    add_feature(
        "impression_repeat_ad_ratio",
        1.0
        - ratio(
            col("impression_distinct_ads"),
            col("impression_n_all"),
            1.0,
        ),
    )
    add_feature(
        "click_repeat_ad_ratio",
        1.0
        - ratio(col("click_distinct_ads"), col("click_n_all"), 1.0),
    )
    add_feature(
        "visit_repeat_ad_ratio",
        1.0 - ratio(col("visit_distinct_ads"), col("visit_n_all"), 1.0),
    )
    add_feature(
        "phone_repeat_ad_ratio",
        1.0 - ratio(col("phone_distinct_ads"), col("phone_n_all"), 1.0),
    )
    add_feature("cold_no_search", (col("search_n_all") == 0).astype(np.float32))
    add_feature(
        "cold_no_search_visit",
        (
            (col("search_n_all") == 0)
            & (col("visit_n_all") == 0)
        ).astype(np.float32),
    )
    add_feature(
        "cold_no_interaction",
        (
            (col("search_n_all") == 0)
            & (col("visit_n_all") == 0)
            & (col("phone_n_all") == 0)
        ).astype(np.float32),
    )
    core = np.asarray(core, dtype=np.float32)
    core[np.isinf(core)] = np.nan
    print(
        f"[features] core matrix rows={len(core)} cols={core.shape[1]} "
        f"elapsed={elapsed(start)}s",
        flush=True,
    )
    return core, names


def _hash_projection(values, seed):
    values = np.asarray(values, dtype=np.int64).view(np.uint64)
    hashed = values * np.uint64(11400714819323198485)
    hashed ^= np.uint64(seed)
    hashed *= np.uint64(14029467366897019727)
    bins = (hashed & np.uint64(63)).astype(np.int64)
    signs = np.where(
        ((hashed >> np.uint64(6)) & np.uint64(1)) == 0,
        1.0,
        -1.0,
    ).astype(np.float32)
    return bins, signs


def _update_sketch(state, channel, users, values, weights, seed):
    users = np.asarray(users)
    values = np.asarray(values)
    valid = (
        pd.notna(users)
        & pd.notna(values)
        & (users >= 0)
        & (users < state.shape[1])
    )
    if not valid.any():
        return
    user_values = users[valid].astype(np.int64)
    item_values = values[valid].astype(np.int64)
    bins, signs = _hash_projection(item_values, seed)
    contribution = np.asarray(weights)[valid].astype(np.float32) * signs
    np.add.at(state[channel], (user_values, bins), contribution)


def _event_weights(frame, anchor, multiplier):
    age = (
        anchor.to_datetime64()
        - frame["event_time"].to_numpy(dtype="datetime64[ns]")
    ) / np.timedelta64(1, "D")
    return (
        np.exp(-np.maximum(age.astype(np.float32), 0.0) / 4.0)
        * np.float32(multiplier)
    )


def build_sketch_features(con, seeds, debug=False):
    start = time.time()
    dimensions = 64
    channels = [
        "searched_ad",
        "clicked_ad",
        "viewed_ad",
        "phoned_ad",
        "taste_category",
        "taste_parent_category",
        "taste_city",
        "taste_price_band",
    ]
    seeds_by_time = {
        pd.Timestamp(timestamp): frame
        for timestamp, frame in seeds.groupby("timestamp", sort=True)
    }
    state = np.zeros((len(channels), 98250, dimensions), dtype=np.float32)
    output = np.zeros(
        (len(seeds), len(channels) * dimensions), dtype=np.float32
    )
    allowed = None
    if debug:
        allowed = np.zeros(98250, dtype=bool)
        allowed[
            seeds["UserID"].dropna().astype(np.int64).unique()
        ] = True
    previous = None
    hash_seeds = [
        104729,
        130363,
        155921,
        181081,
        205759,
        229937,
        256019,
        279481,
    ]
    for anchor, anchor_seeds in seeds_by_time.items():
        if previous is not None:
            delta = (anchor - previous).total_seconds() / 86400.0
            state *= np.float32(np.exp(-delta / 4.0))
            lower = f"event_time > TIMESTAMP '{previous}' AND "
        else:
            lower = ""
        upper = f"event_time <= TIMESTAMP '{anchor}'"
        impression = con.execute(
            f"""
            SELECT
                UserID, AdID, event_time, IsClick, AdCategoryID,
                ParentCategoryID, CityID,
                FLOOR(LN(GREATEST(COALESCE(Price, 0), 0) + 1)
                    / LN(2)) AS PriceBand
            FROM impression_events
            WHERE {lower}{upper}
            """
        ).df()
        visit = con.execute(
            f"""
            SELECT
                UserID, AdID, event_time, AdCategoryID,
                ParentCategoryID, CityID,
                FLOOR(LN(GREATEST(COALESCE(Price, 0), 0) + 1)
                    / LN(2)) AS PriceBand
            FROM visit_events
            WHERE {lower}{upper}
            """
        ).df()
        phone = con.execute(
            f"""
            SELECT
                UserID, AdID, event_time, AdCategoryID,
                ParentCategoryID, CityID,
                FLOOR(LN(GREATEST(COALESCE(Price, 0), 0) + 1)
                    / LN(2)) AS PriceBand
            FROM phone_events
            WHERE {lower}{upper}
            """
        ).df()
        for frame in [impression, visit, phone]:
            if len(frame):
                valid_users = (
                    frame["UserID"].notna()
                    & (frame["UserID"] >= 0)
                    & (frame["UserID"] < 98250)
                )
                frame.drop(frame.index[~valid_users], inplace=True)
        if allowed is not None:
            for frame in [impression, visit, phone]:
                if len(frame):
                    users = frame["UserID"].to_numpy(dtype=np.int64)
                    frame.drop(frame.index[~allowed[users]], inplace=True)
        if len(impression):
            weights = _event_weights(impression, anchor, 1.0)
            _update_sketch(
                state,
                0,
                impression["UserID"],
                impression["AdID"],
                weights,
                hash_seeds[0],
            )
            clicked = impression["IsClick"].to_numpy() == 1.0
            if clicked.any():
                _update_sketch(
                    state,
                    1,
                    impression.loc[clicked, "UserID"],
                    impression.loc[clicked, "AdID"],
                    weights[clicked],
                    hash_seeds[1],
                )
            profile_weights = weights * (
                0.2 + impression["IsClick"].to_numpy(dtype=np.float32)
            )
            for channel, column in [
                (4, "AdCategoryID"),
                (5, "ParentCategoryID"),
                (6, "CityID"),
                (7, "PriceBand"),
            ]:
                _update_sketch(
                    state,
                    channel,
                    impression["UserID"],
                    impression[column],
                    profile_weights,
                    hash_seeds[channel],
                )
        for frame, ad_channel, multiplier in [
            (visit, 2, 0.7),
            (phone, 3, 1.5),
        ]:
            if not len(frame):
                continue
            weights = _event_weights(frame, anchor, multiplier)
            _update_sketch(
                state,
                ad_channel,
                frame["UserID"],
                frame["AdID"],
                weights,
                hash_seeds[ad_channel],
            )
            for channel, column in [
                (4, "AdCategoryID"),
                (5, "ParentCategoryID"),
                (6, "CityID"),
                (7, "PriceBand"),
            ]:
                _update_sketch(
                    state,
                    channel,
                    frame["UserID"],
                    frame[column],
                    weights,
                    hash_seeds[channel],
                )
        rows = anchor_seeds["row_id"].to_numpy(dtype=np.int64)
        users = anchor_seeds["UserID"].to_numpy(dtype=np.int64)
        output[rows] = state[:, users, :].transpose(1, 0, 2).reshape(
            len(rows), -1
        )
        previous = anchor
        print(
            f"[features] sketch snapshot anchor={anchor.date()} "
            f"rows={len(rows)} elapsed={elapsed(start)}s",
            flush=True,
        )
    names = [
        f"{channel}_sketch_{dimension:02d}"
        for channel in channels
        for dimension in range(dimensions)
    ]
    return output, names


def save_feature_cache(
    cache_dir,
    episodes,
    seeds,
    core,
    core_names,
    sketches,
    sketch_names,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "core.npy", core)
    np.save(cache_dir / "sketches.npy", sketches)
    episodes.to_parquet(cache_dir / "episodes.parquet", index=False)
    seeds.to_parquet(cache_dir / "seeds.parquet", index=False)
    (cache_dir / "feature_names.json").write_text(
        json.dumps({"core": core_names, "sketches": sketch_names})
    )
    (cache_dir / "complete.json").write_text(
        json.dumps(
            {
                "rows": len(seeds),
                "core_columns": len(core_names),
                "sketch_columns": len(sketch_names),
            }
        )
    )


def load_feature_cache(cache_dir):
    names = json.loads((cache_dir / "feature_names.json").read_text())
    episodes = pd.read_parquet(cache_dir / "episodes.parquet")
    seeds = pd.read_parquet(cache_dir / "seeds.parquet")
    core = np.load(cache_dir / "core.npy", mmap_mode="r")
    sketches = np.load(cache_dir / "sketches.npy", mmap_mode="r")
    return (
        episodes,
        seeds,
        core,
        names["core"],
        sketches,
        names["sketches"],
    )
