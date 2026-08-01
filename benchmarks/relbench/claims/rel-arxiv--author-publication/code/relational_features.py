from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


CACHE_VERSION = "alltable_temporal_v5"
HORIZONS = (30, 91, 182, 365, 730, 1095)
HALF_LIVES = (30, 91, 182, 365)


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.columns:
        if column not in {"_row_id", "Author_ID", "date"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    return frame


def _add(frame: pd.DataFrame, values: pd.DataFrame, key: str = "Author_ID") -> pd.DataFrame:
    columns = [column for column in values.columns if column != key]
    overlap = set(columns).intersection(frame.columns)
    if overlap:
        raise RuntimeError(f"duplicate feature columns: {sorted(overlap)}")
    return frame.merge(values, on=key, how="left", sort=False, validate="many_to_one")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.clip(lower=1e-3)


class TemporalFeatureBuilder:
    def __init__(self, db, cache_root: Path):
        started = time.time()
        self.cache_dir = cache_root / f"generic_exp_0_{CACHE_VERSION}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        raw_pa = db.table_dict["paperAuthors"].df[["Paper_ID", "Author_ID", "Submission_Date"]].copy()
        raw_pa["Paper_ID"] = raw_pa["Paper_ID"].astype(np.int64)
        raw_pa["Author_ID"] = raw_pa["Author_ID"].astype(np.int64)
        self.raw_pa = raw_pa.sort_values("Submission_Date", kind="stable").reset_index(drop=True)
        pa = raw_pa.drop_duplicates(["Paper_ID", "Author_ID"], keep="first").copy()
        date_spread = pa.groupby("Paper_ID", sort=False)["Submission_Date"].nunique().max()
        if int(date_spread) != 1:
            raise RuntimeError("paper roster contains inconsistent submission dates")
        team = pa.groupby("Paper_ID", sort=False).agg(team_size=("Author_ID", "size")).reset_index()
        pa = pa.merge(team, on="Paper_ID", how="left", validate="many_to_one")
        self.pa = pa.sort_values("Submission_Date", kind="stable").reset_index(drop=True)
        papers = db.table_dict["papers"].df.copy()
        papers["Paper_ID"] = papers["Paper_ID"].astype(np.int64)
        papers["Primary_Category_ID"] = papers["Primary_Category_ID"].astype(np.int64)
        for column in ("Title", "Abstract", "DOI", "arXiv_Code"):
            text = papers[column].fillna("").astype(str)
            papers[f"paper_{column.lower()}_length"] = text.str.len().astype(np.int32)
            papers[f"paper_{column.lower()}_missing"] = text.str.strip().eq("").astype(np.int8)
        papers["paper_title_tokens"] = papers["Title"].fillna("").astype(str).str.split().str.len().astype(np.int16)
        papers["paper_abstract_tokens"] = papers["Abstract"].fillna("").astype(str).str.split().str.len().astype(np.int32)
        papers["paper_has_doi"] = papers["DOI"].fillna("").astype(str).str.strip().ne("").astype(np.int8)
        self.papers = papers.merge(team, on="Paper_ID", how="left", validate="one_to_one")
        authors = db.table_dict["authors"].df.copy()
        authors["Author_ID"] = authors["Author_ID"].astype(np.int64)
        names = authors["Name"].fillna("").astype(str)
        orcid = authors["ORCID"].fillna("").astype(str)
        authors["author_name_length"] = names.str.len().astype(np.int16)
        authors["author_name_tokens"] = names.str.split().str.len().astype(np.int8)
        authors["author_name_punctuation"] = names.str.count(r"[^A-Za-z0-9 ]").astype(np.int8)
        authors["author_duplicate_name_frequency"] = names.map(names.value_counts()).astype(np.int32)
        authors["author_has_orcid"] = orcid.str.strip().ne("").astype(np.int8)
        authors["author_id_rank"] = authors["Author_ID"].rank(method="dense", pct=True).astype(np.float32)
        self.authors = authors[["Author_ID", "author_name_length", "author_name_tokens", "author_name_punctuation", "author_duplicate_name_frequency", "author_has_orcid", "author_id_rank"]]
        citations = db.table_dict["citations"].df[["Paper_ID", "References_Paper_ID", "Submission_Date"]].copy()
        citations["Paper_ID"] = citations["Paper_ID"].astype(np.int64)
        citations["References_Paper_ID"] = citations["References_Paper_ID"].fillna(-1).astype(np.int64)
        self.citations = citations.sort_values("Submission_Date", kind="stable").reset_index(drop=True)
        categories = db.table_dict["paperCategories"].df[["Paper_ID", "Category_ID", "Submission_Date"]].copy()
        categories["Paper_ID"] = categories["Paper_ID"].astype(np.int64)
        categories["Category_ID"] = categories["Category_ID"].astype(np.int64)
        self.paper_categories = categories.sort_values("Submission_Date", kind="stable").reset_index(drop=True)
        primary = self.papers[["Paper_ID", "Primary_Category_ID"]]
        agreement = categories.merge(primary, on="Paper_ID", how="left")
        agreement["match"] = agreement["Category_ID"].eq(agreement["Primary_Category_ID"])
        agreement = agreement.groupby("Paper_ID", sort=False)["match"].max().rename("paper_primary_in_categories").reset_index()
        self.papers = self.papers.merge(agreement, on="Paper_ID", how="left", validate="one_to_one")
        self.papers["paper_primary_in_categories"] = self.papers["paper_primary_in_categories"].eq(True).astype(np.int8)
        self.paper_masks = self._build_paper_masks(pa)
        self.preparation_seconds = time.time() - started

    def _build_paper_masks(self, pa: pd.DataFrame) -> pd.DataFrame:
        values = pa[["Paper_ID", "Author_ID"]].copy()
        author = values["Author_ID"].to_numpy(dtype=np.uint64)
        hashed = author * np.uint64(11400714819323198485)
        hashed ^= hashed >> np.uint64(33)
        hashed *= np.uint64(14029467366897019727)
        bucket = (hashed & np.uint64(255)).astype(np.uint16)
        for block in range(4):
            within = (bucket % 64).astype(np.uint64)
            selected = bucket // 64 == block
            mask = np.zeros(len(values), dtype=np.uint64)
            mask[selected] = np.left_shift(np.uint64(1), within[selected])
            values[f"mask_{block}"] = mask
        connection = duckdb.connect()
        connection.register("membership", values)
        result = connection.sql(
            "SELECT Paper_ID, bit_or(mask_0) AS mask_0, bit_or(mask_1) AS mask_1, "
            "bit_or(mask_2) AS mask_2, bit_or(mask_3) AS mask_3 "
            "FROM membership GROUP BY Paper_ID"
        ).df()
        connection.close()
        return result

    def _cache_path(self, rows: pd.DataFrame, origin: pd.Timestamp, feature_scope: str) -> Path:
        author_bytes = np.asarray(rows["Author_ID"], dtype=np.int64).tobytes()
        digest = hashlib.sha1(author_bytes).hexdigest()[:16]
        stamp = pd.Timestamp(origin).strftime("%Y%m%d")
        return self.cache_dir / f"{stamp}_{feature_scope}_{len(rows)}_{digest}.pkl"

    def build(self, rows: pd.DataFrame, feature_scope: str = "full") -> tuple[pd.DataFrame, bool, float]:
        started = time.time()
        if rows["date"].nunique() != 1:
            raise RuntimeError("feature batch must contain exactly one origin")
        origin = pd.Timestamp(rows["date"].iloc[0])
        cache_path = self._cache_path(rows, origin, feature_scope)
        if cache_path.exists():
            cached = pd.read_pickle(cache_path)
            if not np.array_equal(cached["_row_id"].to_numpy(), rows["_row_id"].to_numpy()):
                raise RuntimeError("cached feature row alignment mismatch")
            return cached, True, time.time() - started
        base = rows[["_row_id", "date", "Author_ID"]].copy()
        base["Author_ID"] = base["Author_ID"].astype(np.int64)
        if base["_row_id"].duplicated().any() or base["Author_ID"].duplicated().any():
            raise RuntimeError("origin rows require unique row and author identifiers")
        cohort = set(base["Author_ID"].tolist())
        hist = self.pa.loc[self.pa["Submission_Date"].le(origin)].copy()
        focus = hist.loc[hist["Author_ID"].isin(cohort)].copy()
        focus["age_days"] = (origin - focus["Submission_Date"]).dt.days.astype(np.float32)
        frame = self._publication_features(base, focus)
        frame, paper_state, author_state = self._roster_coauthor_features(frame, hist, focus, cohort, origin)
        frame = self._citation_features(frame, focus, origin)
        if feature_scope == "full":
            frame = self._category_features(frame, focus, origin)
            frame = self._paper_metadata_features(frame, focus, origin)
            frame = _add(frame, self.authors)
        frame = self._global_features(frame, hist, focus, paper_state, author_state, origin)
        frame = frame.sort_values("_row_id", kind="stable").reset_index(drop=True)
        if not np.array_equal(frame["_row_id"].to_numpy(), rows["_row_id"].to_numpy()):
            raise RuntimeError("feature row order changed")
        frame = _numeric_frame(frame)
        frame.to_pickle(cache_path)
        return frame, False, time.time() - started

    def _publication_features(self, base: pd.DataFrame, focus: pd.DataFrame) -> pd.DataFrame:
        values = focus[["Author_ID", "Paper_ID", "Submission_Date", "team_size", "age_days"]].copy()
        for horizon in HORIZONS:
            within = values["age_days"].le(horizon).astype(np.float32)
            values[f"pub_count_{horizon}d"] = within
            values[f"frac_output_{horizon}d"] = within / values["team_size"].clip(lower=1)
        values["pub_count_lifetime"] = 1.0
        values["frac_output_lifetime"] = 1.0 / values["team_size"].clip(lower=1)
        for lag in range(6):
            lower = lag * 182
            upper = (lag + 1) * 182
            values[f"pub_halfyear_lag_{lag}"] = values["age_days"].gt(lower).mul(values["age_days"].le(upper)).astype(np.float32)
        for half_life in HALF_LIVES:
            values[f"pub_decay_hl_{half_life}"] = np.exp(-np.log(2.0) * values["age_days"] / half_life)
        values["team_above_10"] = values["team_size"].gt(10).astype(np.float32)
        values["team_above_20"] = values["team_size"].gt(20).astype(np.float32)
        values["team_above_50"] = values["team_size"].gt(50).astype(np.float32)
        values["team_above_100"] = values["team_size"].gt(100).astype(np.float32)
        values["team_above_500"] = values["team_size"].gt(500).astype(np.float32)
        values["team_recent_size"] = values["team_size"].where(values["age_days"].le(182))
        values["team_older_size"] = values["team_size"].where(values["age_days"].gt(182))
        aggregations = {column: "sum" for column in values.columns if column.startswith(("pub_count_", "frac_output_", "pub_halfyear_", "pub_decay_"))}
        aggregations.update(
            {
                "age_days": ["min", "max"],
                "team_size": ["mean", "max", "std"],
                "team_above_10": "mean",
                "team_above_20": "mean",
                "team_above_50": "mean",
                "team_above_100": "mean",
                "team_above_500": "mean",
                "team_recent_size": "mean",
                "team_older_size": "mean",
            }
        )
        grouped = values.groupby("Author_ID", sort=False).agg(aggregations)
        grouped.columns = ["_".join([str(part) for part in column if part]) if isinstance(column, tuple) else column for column in grouped.columns]
        grouped = grouped.rename(
            columns={
                "age_days_min": "pub_last_age_days",
                "age_days_max": "pub_first_age_days",
                "team_size_mean": "team_size_mean",
                "team_size_max": "team_size_max",
                "team_size_std": "team_size_std",
                "team_above_10_mean": "team_fraction_above_10",
                "team_above_20_mean": "team_fraction_above_20",
                "team_above_50_mean": "team_fraction_above_50",
                "team_above_100_mean": "team_fraction_above_100",
                "team_above_500_mean": "team_fraction_above_500",
                "team_recent_size_mean": "team_size_recent_mean",
                "team_older_size_mean": "team_size_older_mean",
            }
        ).reset_index()
        grouped.columns = [column.removesuffix("_sum") for column in grouped.columns]
        quantiles = values.groupby("Author_ID", sort=False)["team_size"].quantile([0.5, 0.75, 0.9]).unstack()
        quantiles.columns = ["team_size_median", "team_size_q75", "team_size_q90"]
        grouped = grouped.merge(quantiles.reset_index(), on="Author_ID", how="left")
        active = values.assign(active_halfyear=(values["age_days"] // 182).astype(np.int16)).groupby("Author_ID", sort=False)["active_halfyear"].nunique().rename("pub_active_halfyears").reset_index()
        grouped = grouped.merge(active, on="Author_ID", how="left")
        ordered = values.sort_values(["Author_ID", "Submission_Date", "Paper_ID"], kind="stable")
        ordered["interarrival_days"] = ordered.groupby("Author_ID", sort=False)["Submission_Date"].diff().dt.days
        intervals = ordered.groupby("Author_ID", sort=False)["interarrival_days"].agg(["mean", "std", "median", "min", "max"]).add_prefix("pub_interarrival_").reset_index()
        grouped = grouped.merge(intervals, on="Author_ID", how="left")
        frame = _add(base, grouped)
        for column in ("pub_count_30d", "pub_count_91d", "pub_count_182d", "pub_count_365d", "pub_count_730d", "pub_count_lifetime"):
            frame[column] = frame[column].fillna(0)
        frame["pub_acceleration_182"] = _safe_ratio(frame["pub_count_182d"] + 0.25, frame["pub_count_365d"] - frame["pub_count_182d"] + 0.25)
        frame["pub_acceleration_91"] = _safe_ratio(frame["pub_count_91d"] + 0.25, frame["pub_count_182d"] - frame["pub_count_91d"] + 0.25)
        frame["pub_seasonal_ratio"] = _safe_ratio(frame["pub_halfyear_lag_0"].fillna(0) + 0.25, frame["pub_halfyear_lag_2"].fillna(0) + 0.25)
        frame["pub_recent_lifetime_share"] = _safe_ratio(frame["pub_count_365d"], frame["pub_count_lifetime"] + 0.5)
        frame["frac_recent_lifetime_share"] = _safe_ratio(frame["frac_output_365d"].fillna(0), frame["frac_output_lifetime"].fillna(0) + 0.1)
        frame["team_size_trend"] = _safe_ratio(frame["team_size_recent_mean"].fillna(0) + 1, frame["team_size_older_mean"].fillna(0) + 1)
        frame["author_has_history"] = frame["pub_count_lifetime"].gt(0).astype(np.float32)
        frame["author_cold_start"] = 1.0 - frame["author_has_history"]
        return frame

    def _roster_coauthor_features(self, frame: pd.DataFrame, hist: pd.DataFrame, focus: pd.DataFrame, cohort: set[int], origin: pd.Timestamp):
        history = hist[["Paper_ID", "Author_ID", "Submission_Date", "team_size"]].copy()
        history["in_cohort"] = history["Author_ID"].isin(cohort).astype(np.float32)
        history["all_pub_182"] = history["Submission_Date"].gt(origin - pd.Timedelta(days=182)).astype(np.float32)
        history["all_pub_365"] = history["Submission_Date"].gt(origin - pd.Timedelta(days=365)).astype(np.float32)
        author_state = history.groupby("Author_ID", sort=False).agg(
            all_pub_lifetime=("Paper_ID", "size"),
            all_pub_182=("all_pub_182", "sum"),
            all_pub_365=("all_pub_365", "sum"),
            all_first_date=("Submission_Date", "min"),
            all_last_date=("Submission_Date", "max"),
            in_cohort=("in_cohort", "max"),
        ).reset_index()
        author_state["all_career_age"] = (origin - author_state["all_first_date"]).dt.days.astype(np.float32)
        paper_state = history.groupby("Paper_ID", sort=False).agg(
            roster_intersection_count=("in_cohort", "sum"),
            team_size=("team_size", "max"),
            Submission_Date=("Submission_Date", "max"),
        ).reset_index()
        paper_state["roster_intersection_fraction"] = paper_state["roster_intersection_count"] / paper_state["team_size"].clip(lower=1)
        focal = focus[["Author_ID", "Paper_ID", "age_days", "team_size"]].merge(
            paper_state[["Paper_ID", "roster_intersection_count", "roster_intersection_fraction"]],
            on="Paper_ID",
            how="left",
            validate="many_to_one",
        )
        focal["roster_repeat_2"] = focal["roster_intersection_count"].ge(2).astype(np.float32)
        focal["roster_repeat_5"] = focal["roster_intersection_count"].ge(5).astype(np.float32)
        focal["roster_repeat_10"] = focal["roster_intersection_count"].ge(10).astype(np.float32)
        focal["roster_repeat_50"] = focal["roster_intersection_count"].ge(50).astype(np.float32)
        focal["roster_recent_count"] = focal["roster_intersection_count"].where(focal["age_days"].le(365))
        focal["roster_recent_fraction"] = focal["roster_intersection_fraction"].where(focal["age_days"].le(365))
        focal["roster_repeat_age"] = focal["age_days"].where(focal["roster_intersection_count"].ge(2))
        focal["roster_band_small"] = focal["roster_intersection_count"].where(focal["team_size"].le(10))
        focal["roster_band_medium"] = focal["roster_intersection_count"].where(focal["team_size"].between(11, 50))
        focal["roster_band_large"] = focal["roster_intersection_count"].where(focal["team_size"].between(51, 500))
        focal["roster_band_huge"] = focal["roster_intersection_count"].where(focal["team_size"].gt(500))
        roster = focal.groupby("Author_ID", sort=False).agg(
            roster_intersection_max=("roster_intersection_count", "max"),
            roster_intersection_mean=("roster_intersection_count", "mean"),
            roster_fraction_max=("roster_intersection_fraction", "max"),
            roster_fraction_mean=("roster_intersection_fraction", "mean"),
            roster_recent_count_mean=("roster_recent_count", "mean"),
            roster_recent_fraction_mean=("roster_recent_fraction", "mean"),
            roster_repeat_2_count=("roster_repeat_2", "sum"),
            roster_repeat_5_count=("roster_repeat_5", "sum"),
            roster_repeat_10_count=("roster_repeat_10", "sum"),
            roster_repeat_50_count=("roster_repeat_50", "sum"),
            roster_repeat_recency=("roster_repeat_age", "min"),
            roster_band_small_max=("roster_band_small", "max"),
            roster_band_medium_max=("roster_band_medium", "max"),
            roster_band_large_max=("roster_band_large", "max"),
            roster_band_huge_max=("roster_band_huge", "max"),
            coauthor_exposures=("team_size", lambda series: float(np.maximum(series.to_numpy() - 1, 0).sum())),
        ).reset_index()
        frame = _add(frame, roster)
        augmented = history[["Paper_ID", "Author_ID", "team_size"]].merge(
            author_state[["Author_ID", "all_pub_lifetime", "all_pub_182", "all_pub_365", "all_career_age"]],
            on="Author_ID",
            how="left",
            validate="many_to_one",
        )
        paper_productivity = augmented.groupby("Paper_ID", sort=False).agg(
            paper_author_pub182_sum=("all_pub_182", "sum"),
            paper_author_pub182_max=("all_pub_182", "max"),
            paper_author_pub365_sum=("all_pub_365", "sum"),
            paper_author_pub365_max=("all_pub_365", "max"),
            paper_author_lifetime_sum=("all_pub_lifetime", "sum"),
            paper_author_lifetime_max=("all_pub_lifetime", "max"),
            paper_author_career_mean=("all_career_age", "mean"),
            paper_author_career_max=("all_career_age", "max"),
            team_size=("team_size", "max"),
        ).reset_index()
        own = focus[["Author_ID", "Paper_ID", "age_days"]].merge(
            author_state[["Author_ID", "all_pub_lifetime", "all_pub_182", "all_pub_365"]],
            on="Author_ID",
            how="left",
            validate="many_to_one",
        ).merge(paper_productivity, on="Paper_ID", how="left", validate="many_to_one")
        peers = (own["team_size"] - 1).clip(lower=1)
        own["coauthor_pub182_mean"] = (own["paper_author_pub182_sum"] - own["all_pub_182"]) / peers
        own["coauthor_pub365_mean"] = (own["paper_author_pub365_sum"] - own["all_pub_365"]) / peers
        own["coauthor_lifetime_mean"] = (own["paper_author_lifetime_sum"] - own["all_pub_lifetime"]) / peers
        own["coauthor_recent_weight"] = own["age_days"].le(365).astype(np.float32)
        productivity = own.groupby("Author_ID", sort=False).agg(
            coauthor_pub182_mean=("coauthor_pub182_mean", "mean"),
            coauthor_pub182_max=("paper_author_pub182_max", "max"),
            coauthor_pub365_mean=("coauthor_pub365_mean", "mean"),
            coauthor_pub365_max=("paper_author_pub365_max", "max"),
            coauthor_lifetime_mean=("coauthor_lifetime_mean", "mean"),
            coauthor_lifetime_max=("paper_author_lifetime_max", "max"),
            coauthor_career_mean=("paper_author_career_mean", "mean"),
            coauthor_career_max=("paper_author_career_max", "max"),
        ).reset_index()
        frame = _add(frame, productivity)
        masks = focus[["Author_ID", "Paper_ID"]].merge(self.paper_masks, on="Paper_ID", how="left", validate="many_to_one")
        connection = duckdb.connect()
        connection.register("author_papers", masks)
        distinct = connection.sql(
            "SELECT Author_ID, CAST(bit_count(bit_or(mask_0)) AS INTEGER) + CAST(bit_count(bit_or(mask_1)) AS INTEGER) + "
            "CAST(bit_count(bit_or(mask_2)) AS INTEGER) + CAST(bit_count(bit_or(mask_3)) AS INTEGER) AS occupied_hash_buckets "
            "FROM author_papers GROUP BY Author_ID"
        ).df()
        connection.close()
        occupied = distinct["occupied_hash_buckets"].to_numpy(dtype=np.float64)
        zeros = np.maximum(256.0 - occupied, 0.5)
        distinct["coauthor_unique_estimate"] = np.maximum(-256.0 * np.log(zeros / 256.0) - 1.0, 0.0)
        frame = _add(frame, distinct[["Author_ID", "coauthor_unique_estimate"]])
        frame["coauthor_repeat_estimate"] = (frame["coauthor_exposures"].fillna(0) - frame["coauthor_unique_estimate"].fillna(0)).clip(lower=0)
        frame["cohort_coauthor_max"] = (frame["roster_intersection_max"].fillna(0) - frame["author_has_history"]).clip(lower=0)
        return frame, paper_state, author_state

    def _citation_features(self, frame: pd.DataFrame, focus: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        citations = self.citations.loc[self.citations["Submission_Date"].le(origin)].copy()
        citations["recent182"] = citations["Submission_Date"].gt(origin - pd.Timedelta(days=182)).astype(np.float32)
        citations["recent365"] = citations["Submission_Date"].gt(origin - pd.Timedelta(days=365)).astype(np.float32)
        outgoing = citations.groupby("Paper_ID", sort=False).agg(
            citation_out_count=("References_Paper_ID", "size"),
            citation_out_unique=("References_Paper_ID", "nunique"),
            citation_out_recent182=("recent182", "sum"),
            citation_out_recent365=("recent365", "sum"),
        ).reset_index()
        incoming = citations.loc[citations["References_Paper_ID"].ge(0)].groupby("References_Paper_ID", sort=False).agg(
            citation_in_count=("Paper_ID", "size"),
            citation_in_unique=("Paper_ID", "nunique"),
            citation_in_recent182=("recent182", "sum"),
            citation_in_recent365=("recent365", "sum"),
        ).reset_index().rename(columns={"References_Paper_ID": "Paper_ID"})
        authored = focus[["Author_ID", "Paper_ID", "age_days"]].merge(outgoing, on="Paper_ID", how="left", validate="many_to_one").merge(incoming, on="Paper_ID", how="left", validate="many_to_one")
        citation_columns = [column for column in authored.columns if column.startswith("citation_")]
        authored[citation_columns] = authored[citation_columns].fillna(0)
        authored["citation_recent_paper_in"] = authored["citation_in_count"].where(authored["age_days"].le(365))
        aggregations = {column: ["sum", "max", "mean"] for column in citation_columns}
        aggregations["citation_recent_paper_in"] = ["max", "mean"]
        grouped = authored.groupby("Author_ID", sort=False).agg(aggregations)
        grouped.columns = [f"{left}_{right}" for left, right in grouped.columns]
        grouped = grouped.reset_index()
        frame = _add(frame, grouped)
        frame["citation_in_per_paper"] = _safe_ratio(frame["citation_in_count_sum"].fillna(0), frame["pub_count_lifetime"].fillna(0) + 0.5)
        frame["citation_out_per_paper"] = _safe_ratio(frame["citation_out_count_sum"].fillna(0), frame["pub_count_lifetime"].fillna(0) + 0.5)
        frame["citation_in_acceleration"] = _safe_ratio(frame["citation_in_recent182_sum"].fillna(0) + 0.25, frame["citation_in_recent365_sum"].fillna(0) - frame["citation_in_recent182_sum"].fillna(0) + 0.25)
        return frame

    def _category_features(self, frame: pd.DataFrame, focus: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        categories = self.paper_categories.loc[self.paper_categories["Submission_Date"].le(origin)]
        authored = focus[["Author_ID", "Paper_ID", "Submission_Date", "age_days"]].merge(
            categories[["Paper_ID", "Category_ID"]], on="Paper_ID", how="left", validate="many_to_many"
        ).dropna(subset=["Category_ID"])
        if len(authored):
            counts = authored.groupby(["Author_ID", "Category_ID"], sort=False).size().rename("count").reset_index()
            totals = counts.groupby("Author_ID", sort=False)["count"].sum().rename("category_tag_count")
            counts = counts.merge(totals, on="Author_ID", how="left")
            counts["share"] = counts["count"] / counts["category_tag_count"]
            counts["entropy_piece"] = -counts["share"] * np.log(counts["share"].clip(lower=1e-12))
            diversity = counts.groupby("Author_ID", sort=False).agg(
                category_unique=("Category_ID", "nunique"),
                category_tag_count=("category_tag_count", "max"),
                category_concentration=("share", "max"),
                category_entropy=("entropy_piece", "sum"),
            ).reset_index()
            recent = authored.assign(recent=authored["age_days"].le(365)).groupby("Author_ID", sort=False)["recent"].agg(["sum", "mean"]).reset_index().rename(columns={"sum": "category_recent_tags", "mean": "category_recent_share"})
            diversity = diversity.merge(recent, on="Author_ID", how="left")
            frame = _add(frame, diversity)
        primary = focus[["Author_ID", "Paper_ID", "Submission_Date", "age_days"]].merge(
            self.papers[["Paper_ID", "Primary_Category_ID", "paper_primary_in_categories"]], on="Paper_ID", how="left", validate="many_to_one"
        ).sort_values(["Author_ID", "Submission_Date", "Paper_ID"], kind="stable")
        primary["category_switch"] = primary.groupby("Author_ID", sort=False)["Primary_Category_ID"].diff().ne(0).astype(np.float32)
        first_rows = primary.groupby("Author_ID", sort=False).cumcount().eq(0)
        primary.loc[first_rows, "category_switch"] = 0
        switches = primary.groupby("Author_ID", sort=False).agg(
            category_switch_count=("category_switch", "sum"),
            category_switch_rate=("category_switch", "mean"),
            category_primary_agreement=("paper_primary_in_categories", "mean"),
            category_primary_unique=("Primary_Category_ID", "nunique"),
            category_latest=("Primary_Category_ID", "last"),
        ).reset_index()
        frame = _add(frame, switches)
        paper_hist = self.papers.loc[self.papers["Submission_Date"].le(origin), ["Paper_ID", "Primary_Category_ID", "Submission_Date"]].copy()
        paper_hist["recent182"] = paper_hist["Submission_Date"].gt(origin - pd.Timedelta(days=182)).astype(np.float32)
        paper_hist["recent365"] = paper_hist["Submission_Date"].gt(origin - pd.Timedelta(days=365)).astype(np.float32)
        volume = paper_hist.groupby("Primary_Category_ID", sort=False).agg(
            category_global_lifetime=("Paper_ID", "size"),
            category_global_recent182=("recent182", "sum"),
            category_global_recent365=("recent365", "sum"),
        ).reset_index()
        primary = primary.merge(volume, on="Primary_Category_ID", how="left", validate="many_to_one")
        volume_features = primary.groupby("Author_ID", sort=False).agg(
            category_volume_mean=("category_global_lifetime", "mean"),
            category_volume_max=("category_global_lifetime", "max"),
            category_recent_volume_mean=("category_global_recent182", "mean"),
            category_recent_volume_max=("category_global_recent182", "max"),
        ).reset_index()
        frame = _add(frame, volume_features)
        frame["category_volume_momentum"] = _safe_ratio(frame["category_recent_volume_mean"].fillna(0) + 1, frame["category_volume_mean"].fillna(0) + 1)
        return frame

    def _paper_metadata_features(self, frame: pd.DataFrame, focus: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        columns = [
            "Paper_ID",
            "paper_title_length",
            "paper_abstract_length",
            "paper_doi_length",
            "paper_arxiv_code_length",
            "paper_title_missing",
            "paper_abstract_missing",
            "paper_doi_missing",
            "paper_title_tokens",
            "paper_abstract_tokens",
            "paper_has_doi",
        ]
        authored = focus[["Author_ID", "Paper_ID", "age_days"]].merge(self.papers[columns], on="Paper_ID", how="left", validate="many_to_one")
        metadata_columns = [column for column in columns if column != "Paper_ID"]
        aggregations = {column: ["mean", "max", "min"] for column in metadata_columns}
        grouped = authored.groupby("Author_ID", sort=False).agg(aggregations)
        grouped.columns = [f"{left}_{right}" for left, right in grouped.columns]
        grouped = grouped.reset_index()
        recent = authored.loc[authored["age_days"].le(365)].groupby("Author_ID", sort=False).agg(
            paper_recent_title_length=("paper_title_length", "mean"),
            paper_recent_abstract_length=("paper_abstract_length", "mean"),
            paper_recent_doi_fraction=("paper_has_doi", "mean"),
        ).reset_index()
        return _add(_add(frame, grouped), recent)

    def _global_features(self, frame: pd.DataFrame, hist: pd.DataFrame, focus: pd.DataFrame, paper_state: pd.DataFrame, author_state: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        recent30 = paper_state["Submission_Date"].gt(origin - pd.Timedelta(days=30))
        recent182 = paper_state["Submission_Date"].gt(origin - pd.Timedelta(days=182))
        recent365 = paper_state["Submission_Date"].gt(origin - pd.Timedelta(days=365))
        context = {
            "global_origin_days": float((origin - pd.Timestamp("2018-01-01")).days),
            "global_origin_ordinal": float((origin - pd.Timestamp("2018-01-01")).days / 182.0),
            "global_paper_count_lifetime": float(len(paper_state)),
            "global_paper_count_30": float(recent30.sum()),
            "global_paper_count_182": float(recent182.sum()),
            "global_paper_count_365": float(recent365.sum()),
            "global_recent_paper_acceleration": float((recent182.sum() + 1) / (recent365.sum() - recent182.sum() + 1)),
            "global_team_size_mean": float(paper_state["team_size"].mean()),
            "global_team_size_q90": float(paper_state["team_size"].quantile(0.9)),
            "global_team_size_q99": float(paper_state["team_size"].quantile(0.99)),
            "global_team_size_max": float(paper_state["team_size"].max()),
            "global_seed_cohort_size": float(len(frame)),
            "global_seed_cold_share": float(frame["author_cold_start"].mean()),
            "global_seed_history_papers": float(len(focus)),
            "global_active_author_count": float(len(author_state)),
        }
        for key, value in context.items():
            frame[key] = value
        frame["global_seed_history_density"] = len(focus) / max(len(frame), 1)
        return frame


def feature_columns(frame: pd.DataFrame, scope: str) -> list[str]:
    excluded = {"_row_id", "date", "Author_ID", "publication_count", "origin"}
    columns = [column for column in frame.columns if column not in excluded]
    if scope == "core":
        prefixes = (
            "pub_",
            "frac_",
            "team_",
            "roster_",
            "coauthor_",
            "cohort_",
            "citation_",
            "publication_label_",
            "category_target_",
            "author_cold_start",
            "author_has_history",
            "global_",
        )
        columns = [column for column in columns if column.startswith(prefixes)]
    return sorted(columns)


def add_publication_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["pub_career_yearly_rate"] = frame["pub_count_lifetime"] / (frame["pub_first_age_days"] / 365.0 + 0.25)
    frame["pub_active_halfyear_rate"] = frame["pub_count_lifetime"] / (frame["pub_active_halfyears"] + 0.25)
    frame["pub_interarrival_cv"] = frame["pub_interarrival_std"] / (frame["pub_interarrival_mean"] + 1.0)
    frame["pub_interarrival_burstiness"] = (frame["pub_interarrival_std"] - frame["pub_interarrival_mean"]) / (frame["pub_interarrival_std"] + frame["pub_interarrival_mean"] + 1.0)
    frame["pub_decay_acceleration"] = frame["pub_decay_hl_91"] / (frame["pub_decay_hl_365"] + 0.25)
    frame["frac_output_per_publication"] = frame["frac_output_lifetime"] / (frame["pub_count_lifetime"] + 0.25)
    frame["roster_productivity_strength"] = np.log1p(frame["roster_intersection_max"]) * frame["pub_career_yearly_rate"]
    frame["roster_team_density"] = frame["roster_intersection_max"] / (frame["team_size_max"] + 1.0)
    frame["roster_recent_productivity"] = frame["roster_recent_count_mean"] * frame["pub_decay_hl_365"]
    frame["coauthor_relative_momentum"] = frame["coauthor_pub182_mean"] / (frame["pub_count_182d"] + 1.0)
    frame["team_productivity_strength"] = np.log1p(frame["team_size_max"]) * frame["pub_career_yearly_rate"]
    frame["team_fractional_amplification"] = frame["pub_count_lifetime"] / (frame["frac_output_lifetime"] + 0.25)
    return frame


def cache_registry(cache_root: Path) -> None:
    import fcntl

    registry = cache_root / "artifacts.json"
    lock_path = cache_root / "artifacts.lock"
    entry = {
        "name": f"generic_exp_0_{CACHE_VERSION}",
        "path": f"generic_exp_0_{CACHE_VERSION}",
        "description": "Temporally censored per-origin all-table author-publication feature matrices",
        "content_key": CACHE_VERSION,
        "rebuild_hint": "Run main.py; matrices are extended check-before-compute by origin and cohort fingerprint",
    }
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        records = json.loads(registry.read_text()) if registry.exists() else []
        if not any(record.get("name") == entry["name"] for record in records):
            records.append(entry)
            temporary = registry.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(records, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
