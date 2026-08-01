from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "age_days_log",
    "age_years",
    "seed_year",
    "seed_month_sin",
    "seed_month_cos",
    "submission_year",
    "submission_month_sin",
    "submission_month_cos",
    "primary_category_id",
    "primary_category_frequency_log",
    "doi_family_id",
    "arxiv_age_days_log",
    "doi_family_citation_mean_log",
    "arxiv_submission_delay_log",
    "token_count_log",
    "token_truncated",
    "doi_family_frequency_log",
    "doi_family_history_rate",
    "in_total_log",
    "in_30_log",
    "in_90_log",
    "in_182_log",
    "in_365_log",
    "in_730_log",
    "in_previous_182_log",
    "in_previous_365_log",
    "in_share_30",
    "in_share_90",
    "in_share_182",
    "in_share_365",
    "in_acceleration_182",
    "in_acceleration_365",
    "in_days_since_last_log",
    "in_days_since_first_log",
    "in_span_log",
    "in_cited",
    "citer_in_mean_log",
    "citer_in_max_log",
    "out_total_log",
    "out_missing_log",
    "out_unique_log",
    "out_known_fraction",
    "reference_in_sum_log",
    "reference_in_mean_log",
    "reference_in_max_log",
    "reference_in_cited_fraction",
    "reference_recent_182_mean_log",
    "reference_age_mean_log",
    "author_count_log",
    "author_name_length_mean",
    "author_name_ambiguity_mean_log",
    "author_papers_mean_log",
    "author_papers_max_log",
    "author_papers_min_log",
    "author_recent_365_mean_log",
    "author_recent_365_max_log",
    "author_citations_mean_log",
    "author_citations_max_log",
    "author_cited_papers_mean_log",
    "author_cited_papers_max_log",
    "author_best_paper_mean_log",
    "author_best_paper_max_log",
    "author_career_mean_log",
    "author_history_rate_mean",
    "category_count_log",
    "category_papers_mean_log",
    "category_papers_max_log",
    "category_recent_365_mean_log",
    "category_recent_365_max_log",
    "category_citations_mean_log",
    "category_citations_max_log",
    "category_cited_fraction_mean",
    "category_history_rate_mean",
    "primary_category_citation_mean_log",
    "paper_history_count_log",
    "paper_history_rate",
    "paper_history_latest",
    "author_history_count_mean_log",
    "category_history_count_mean_log",
    "no_history",
]


def day_values(values) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[D]").astype(np.int64)


def day_value(value) -> int:
    return int(np.datetime64(value, "D").astype(np.int64))


def group_max(groups: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    result = np.zeros(size, dtype=np.float64)
    if len(groups):
        np.maximum.at(result, groups, values)
    return result


def group_min(groups: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    result = np.full(size, np.inf, dtype=np.float64)
    if len(groups):
        np.minimum.at(result, groups, values)
    result[~np.isfinite(result)] = 0.0
    return result


def safe_rate(total: np.ndarray, count: np.ndarray, default: float = 0.0) -> np.ndarray:
    result = np.full(len(count), default, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def expand_links(ids: np.ndarray, indptr: np.ndarray, values: np.ndarray):
    starts = indptr[ids]
    lengths = indptr[ids + 1] - starts
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=values.dtype), np.empty(0, dtype=np.int64)
    rows = np.repeat(np.arange(len(ids), dtype=np.int64), lengths)
    bases = np.repeat(np.cumsum(lengths) - lengths, lengths)
    positions = np.repeat(starts, lengths) + np.arange(total, dtype=np.int64) - bases
    return rows, values[positions], positions


def doi_family(value: str):
    pieces = value.lower().split("/", 1)
    prefix = pieces[0]
    suffix = pieces[1] if len(pieces) > 1 else ""
    first = suffix.split("/")[0]
    if re.fullmatch(r"\d{4}-\d{3}[\dx]", first):
        family = first
    else:
        family = re.sub(r"([._-]?\d.*)$", "", first).strip("._-")
    return prefix + ("/" + family if family else "")


class FeatureBuilder:
    def __init__(self, ctx, token_counts: np.ndarray):
        self.n_papers = len(ctx.db.table_dict["papers"].df)
        papers = ctx.db.table_dict["papers"].df.sort_values("Paper_ID")
        paper_ids = papers["Paper_ID"].to_numpy(np.int64)
        if not np.array_equal(paper_ids, np.arange(self.n_papers)):
            raise RuntimeError("Paper_ID must be contiguous for the feature pipeline")
        self.paper_sub_day = day_values(papers["Submission_Date"])
        self.primary_category = papers["Primary_Category_ID"].to_numpy(np.int64)
        self.token_counts = np.asarray(token_counts, dtype=np.float64)
        paper_dates = pd.DatetimeIndex(papers["Submission_Date"])
        self.paper_year = paper_dates.year.to_numpy(np.float64) - 2018.0
        self.paper_month = paper_dates.month.to_numpy(np.float64)
        arxiv_parts = papers["arXiv_Code"].astype(str).str.extract(r"(?:arXiv:|/)(\d{2})(\d{2})")
        arxiv_year = arxiv_parts[0].fillna("18").astype(np.int64).to_numpy()
        arxiv_month = arxiv_parts[1].fillna("01").astype(np.int64).clip(1, 12).to_numpy()
        arxiv_year = np.where(arxiv_year >= 90, 1900 + arxiv_year, 2000 + arxiv_year)
        arxiv_dates = pd.to_datetime({"year": arxiv_year, "month": arxiv_month, "day": 1})
        self.arxiv_day = day_values(arxiv_dates)
        invalid_arxiv = arxiv_parts.isna().any(axis=1).to_numpy()
        self.arxiv_day[invalid_arxiv] = self.paper_sub_day[invalid_arxiv]
        self.arxiv_delay = np.maximum(self.paper_sub_day - self.arxiv_day, 0).astype(np.float64)
        doi_families = papers["DOI"].astype(str).map(doi_family)
        self.paper_doi_family, doi_values = pd.factorize(doi_families, sort=True)
        self.paper_doi_family = self.paper_doi_family.astype(np.int64)
        self.n_doi_families = len(doi_values)

        categories = ctx.db.table_dict["categories"].df.sort_values("Category_ID")
        self.category_code = categories["Category"].to_numpy(np.float64)
        self.n_categories = len(categories)

        citations = ctx.db.table_dict["citations"].df
        self.citation_source = citations["Paper_ID"].to_numpy(np.int64)
        reference_float = citations["References_Paper_ID"].to_numpy(dtype=np.float64, na_value=np.nan)
        self.citation_known = np.isfinite(reference_float)
        self.citation_reference = np.where(self.citation_known, reference_float, -1).astype(np.int64)
        self.citation_day = day_values(citations["Submission_Date"])

        paper_authors = ctx.db.table_dict["paperAuthors"].df.sort_values(["Paper_ID", "Author_ID"])
        self.pa_paper = paper_authors["Paper_ID"].to_numpy(np.int64)
        self.pa_author = paper_authors["Author_ID"].to_numpy(np.int64)
        self.pa_day = day_values(paper_authors["Submission_Date"])
        self.n_authors = len(ctx.db.table_dict["authors"].df)
        self.pa_indptr = np.zeros(self.n_papers + 1, dtype=np.int64)
        np.add.at(self.pa_indptr, self.pa_paper + 1, 1)
        self.pa_indptr = np.cumsum(self.pa_indptr)
        authors = ctx.db.table_dict["authors"].df.sort_values("Author_ID")
        names = authors["Name"].astype(str)
        self.author_name_length = names.str.len().to_numpy(np.float64)
        name_codes, _ = pd.factorize(names, sort=False)
        name_counts = np.bincount(name_codes)
        self.author_name_ambiguity = name_counts[name_codes].astype(np.float64)

        extra_categories = ctx.db.table_dict["paperCategories"].df
        pc_paper = np.concatenate((paper_ids, extra_categories["Paper_ID"].to_numpy(np.int64)))
        pc_category = np.concatenate((self.primary_category, extra_categories["Category_ID"].to_numpy(np.int64)))
        pc_day = np.concatenate((self.paper_sub_day, day_values(extra_categories["Submission_Date"])))
        keys = pc_paper * self.n_categories + pc_category
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        keep = np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
        starts = np.flatnonzero(keep)
        self.pc_paper = pc_paper[order][starts]
        self.pc_category = pc_category[order][starts]
        self.pc_day = np.minimum.reduceat(pc_day[order], starts)
        pc_order = np.argsort(self.pc_paper, kind="stable")
        self.pc_paper = self.pc_paper[pc_order]
        self.pc_category = self.pc_category[pc_order]
        self.pc_day = self.pc_day[pc_order]
        self.pc_indptr = np.zeros(self.n_papers + 1, dtype=np.int64)
        np.add.at(self.pc_indptr, self.pc_paper + 1, 1)
        self.pc_indptr = np.cumsum(self.pc_indptr)

        train = ctx.train.df
        val = ctx.val.df
        self.history_train_ids = train["Paper_ID"].to_numpy(np.int64)
        self.history_train_day = day_values(train["date"])
        self.history_train_y = train[ctx.target_col].to_numpy(np.float64)
        self.history_val_ids = val["Paper_ID"].to_numpy(np.int64)
        self.history_val_day = day_values(val["date"])
        self.history_val_y = val[ctx.target_col].to_numpy(np.float64)

    def _history(self, origin_day: int, allow_validation: bool):
        ids = self.history_train_ids
        days = self.history_train_day
        labels = self.history_train_y
        if allow_validation:
            ids = np.concatenate((ids, self.history_val_ids))
            days = np.concatenate((days, self.history_val_day))
            labels = np.concatenate((labels, self.history_val_y))
        legal = days + 182 <= origin_day
        return ids[legal], days[legal], labels[legal]

    def build(self, seed_ids: np.ndarray, origin, allow_validation_history: bool = False) -> np.ndarray:
        started = time.time()
        seed_ids = np.asarray(seed_ids, dtype=np.int64)
        origin_day = day_value(origin)
        origin_ts = pd.Timestamp(origin)
        size = self.n_papers
        age = np.maximum(origin_day - self.paper_sub_day, 0).astype(np.float64)

        available = self.citation_day <= origin_day
        known = available & self.citation_known
        source = self.citation_source[available]
        reference = self.citation_reference[known]
        known_source = self.citation_source[known]
        known_day = self.citation_day[known]
        in_total = np.bincount(reference, minlength=size).astype(np.float64)

        recent = {}
        for window in (30, 90, 182, 365, 730):
            mask = known_day > origin_day - window
            recent[window] = np.bincount(reference[mask], minlength=size).astype(np.float64)

        last_day = np.full(size, -10**9, dtype=np.int64)
        first_day = np.full(size, 10**9, dtype=np.int64)
        if len(reference):
            np.maximum.at(last_day, reference, known_day)
            np.minimum.at(first_day, reference, known_day)
        cited = in_total > 0
        days_since_last = np.where(cited, origin_day - last_day, age)
        days_since_first = np.where(cited, origin_day - first_day, age)
        citation_span = np.where(cited, last_day - first_day, 0)

        citer_values = in_total[known_source]
        citer_sum = np.bincount(reference, weights=citer_values, minlength=size)
        citer_mean = safe_rate(citer_sum, in_total)
        citer_maximum = group_max(reference, citer_values, size)

        out_total = np.bincount(source, minlength=size).astype(np.float64)
        out_known = np.bincount(known_source, minlength=size).astype(np.float64)
        out_missing = out_total - out_known
        if len(reference):
            unique_keys = np.unique(known_source.astype(np.int64) * size + reference)
            out_unique = np.bincount(unique_keys // size, minlength=size).astype(np.float64)
        else:
            out_unique = np.zeros(size, dtype=np.float64)
        reference_values = in_total[reference]
        reference_sum = np.bincount(known_source, weights=reference_values, minlength=size)
        reference_mean = safe_rate(reference_sum, out_known)
        reference_maximum = group_max(known_source, reference_values, size)
        reference_cited = safe_rate(
            np.bincount(known_source, weights=(reference_values > 0), minlength=size),
            out_known,
        )
        reference_recent = safe_rate(
            np.bincount(known_source, weights=recent[182][reference], minlength=size),
            out_known,
        )
        reference_age = np.maximum(origin_day - self.paper_sub_day[reference], 0)
        reference_age_mean = safe_rate(
            np.bincount(known_source, weights=reference_age, minlength=size),
            out_known,
        )

        active_pa = self.pa_day <= origin_day
        pa_paper = self.pa_paper[active_pa]
        pa_author = self.pa_author[active_pa]
        pa_day = self.pa_day[active_pa]
        paper_author_count = np.bincount(pa_paper, minlength=size).astype(np.float64)
        author_papers = np.bincount(pa_author, minlength=self.n_authors).astype(np.float64)
        author_recent = np.bincount(pa_author[pa_day > origin_day - 365], minlength=self.n_authors).astype(np.float64)
        author_citations = np.bincount(pa_author, weights=in_total[pa_paper], minlength=self.n_authors)
        author_cited_papers = np.bincount(pa_author, weights=(in_total[pa_paper] > 0), minlength=self.n_authors)
        author_best_paper = group_max(pa_author, in_total[pa_paper], self.n_authors)
        author_first_day = np.full(self.n_authors, origin_day, dtype=np.float64)
        if len(pa_author):
            np.minimum.at(author_first_day, pa_author, pa_day)
        author_career = np.maximum(origin_day - author_first_day, 0)

        history_ids, history_days, history_y = self._history(origin_day, allow_validation_history)
        history_default = float(history_y.mean()) if len(history_y) else 0.5
        paper_history_count = np.bincount(history_ids, minlength=size).astype(np.float64)
        paper_history_sum = np.bincount(history_ids, weights=history_y, minlength=size)
        paper_history_rate = safe_rate(paper_history_sum, paper_history_count, history_default)
        paper_history_latest = np.full(size, history_default, dtype=np.float64)
        if len(history_ids):
            for history_origin in np.unique(history_days):
                selected = history_days == history_origin
                paper_history_latest[history_ids[selected]] = history_y[selected]

        author_history_count = np.zeros(self.n_authors, dtype=np.float64)
        author_history_sum = np.zeros(self.n_authors, dtype=np.float64)
        category_history_count = np.zeros(self.n_categories, dtype=np.float64)
        category_history_sum = np.zeros(self.n_categories, dtype=np.float64)
        doi_history_count = np.zeros(self.n_doi_families, dtype=np.float64)
        doi_history_sum = np.zeros(self.n_doi_families, dtype=np.float64)
        if len(history_ids):
            doi_history_count = np.bincount(
                self.paper_doi_family[history_ids],
                minlength=self.n_doi_families,
            ).astype(np.float64)
            doi_history_sum = np.bincount(
                self.paper_doi_family[history_ids],
                weights=history_y,
                minlength=self.n_doi_families,
            )
            history_rows, history_authors, history_positions = expand_links(history_ids, self.pa_indptr, self.pa_author)
            if len(history_rows):
                legal_history_authors = self.pa_day[history_positions] <= origin_day
                history_rows = history_rows[legal_history_authors]
                history_authors = history_authors[legal_history_authors]
                author_history_count = np.bincount(history_authors, minlength=self.n_authors).astype(np.float64)
                author_history_sum = np.bincount(history_authors, weights=history_y[history_rows], minlength=self.n_authors)
            history_rows, history_categories, history_positions = expand_links(history_ids, self.pc_indptr, self.pc_category)
            if len(history_rows):
                legal_history_categories = self.pc_day[history_positions] <= origin_day
                history_rows = history_rows[legal_history_categories]
                history_categories = history_categories[legal_history_categories]
                category_history_count = np.bincount(history_categories, minlength=self.n_categories).astype(np.float64)
                category_history_sum = np.bincount(history_categories, weights=history_y[history_rows], minlength=self.n_categories)
        author_history_rate = safe_rate(author_history_sum, author_history_count, history_default)
        category_history_rate = safe_rate(category_history_sum, category_history_count, history_default)
        doi_history_rate = safe_rate(doi_history_sum, doi_history_count, history_default)

        def author_aggregate(values: np.ndarray):
            linked = values[pa_author]
            total = np.bincount(pa_paper, weights=linked, minlength=size)
            return safe_rate(total, paper_author_count), group_max(pa_paper, linked, size), group_min(pa_paper, linked, size)

        author_name_length_mean, _, _ = author_aggregate(self.author_name_length)
        author_name_ambiguity_mean, _, _ = author_aggregate(self.author_name_ambiguity)
        author_papers_mean, author_papers_maximum, author_papers_minimum = author_aggregate(author_papers)
        author_recent_mean, author_recent_maximum, _ = author_aggregate(author_recent)
        author_citations_mean, author_citations_maximum, _ = author_aggregate(author_citations)
        author_cited_mean, author_cited_maximum, _ = author_aggregate(author_cited_papers)
        author_best_mean, author_best_maximum, _ = author_aggregate(author_best_paper)
        author_career_mean, _, _ = author_aggregate(author_career)
        author_history_rate_mean, _, _ = author_aggregate(author_history_rate)
        author_history_count_mean, _, _ = author_aggregate(author_history_count)

        active_pc = self.pc_day <= origin_day
        pc_paper = self.pc_paper[active_pc]
        pc_category = self.pc_category[active_pc]
        pc_day = self.pc_day[active_pc]
        paper_category_count = np.bincount(pc_paper, minlength=size).astype(np.float64)
        category_papers = np.bincount(pc_category, minlength=self.n_categories).astype(np.float64)
        category_recent = np.bincount(pc_category[pc_day > origin_day - 365], minlength=self.n_categories).astype(np.float64)
        category_citation_sum = np.bincount(pc_category, weights=in_total[pc_paper], minlength=self.n_categories)
        category_cited_sum = np.bincount(pc_category, weights=(in_total[pc_paper] > 0), minlength=self.n_categories)
        category_citation_mean = safe_rate(category_citation_sum, category_papers)
        category_cited_rate = safe_rate(category_cited_sum, category_papers)

        def category_aggregate(values: np.ndarray):
            linked = values[pc_category]
            total = np.bincount(pc_paper, weights=linked, minlength=size)
            return safe_rate(total, paper_category_count), group_max(pc_paper, linked, size)

        category_papers_mean, category_papers_maximum = category_aggregate(category_papers)
        category_recent_mean, category_recent_maximum = category_aggregate(category_recent)
        category_citations_mean, category_citations_maximum = category_aggregate(category_citation_mean)
        category_cited_fraction_mean, _ = category_aggregate(category_cited_rate)
        category_history_rate_mean, _ = category_aggregate(category_history_rate)
        category_history_count_mean, _ = category_aggregate(category_history_count)

        primary_frequency = np.bincount(
            self.primary_category[self.paper_sub_day <= origin_day],
            minlength=self.n_categories,
        ).astype(np.float64)
        doi_family_frequency = np.bincount(
            self.paper_doi_family[self.paper_sub_day <= origin_day],
            minlength=self.n_doi_families,
        ).astype(np.float64)
        doi_family_citation_sum = np.bincount(
            self.paper_doi_family[self.paper_sub_day <= origin_day],
            weights=in_total[self.paper_sub_day <= origin_day],
            minlength=self.n_doi_families,
        )
        doi_family_citation_mean = safe_rate(doi_family_citation_sum, doi_family_frequency)
        primary_category_citation_mean = category_citation_mean[self.primary_category]
        month_angle = 2.0 * math.pi * origin_ts.month / 12.0
        sub_month_angle = 2.0 * math.pi * self.paper_month / 12.0
        previous_182 = np.maximum(recent[365] - recent[182], 0)
        previous_365 = np.maximum(recent[730] - recent[365], 0)
        denominator = in_total + 1.0

        columns = [
            np.log1p(age),
            age / 365.25,
            np.full(size, origin_ts.year - 2018, dtype=np.float64),
            np.full(size, math.sin(month_angle), dtype=np.float64),
            np.full(size, math.cos(month_angle), dtype=np.float64),
            self.paper_year,
            np.sin(sub_month_angle),
            np.cos(sub_month_angle),
            self.category_code[self.primary_category],
            np.log1p(primary_frequency[self.primary_category]),
            self.paper_doi_family.astype(np.float64),
            np.log1p(np.maximum(origin_day - self.arxiv_day, 0)),
            np.log1p(doi_family_citation_mean[self.paper_doi_family]),
            np.log1p(self.arxiv_delay),
            np.log1p(self.token_counts),
            (self.token_counts > 192).astype(np.float64),
            np.log1p(doi_family_frequency[self.paper_doi_family]),
            doi_history_rate[self.paper_doi_family],
            np.log1p(in_total),
            np.log1p(recent[30]),
            np.log1p(recent[90]),
            np.log1p(recent[182]),
            np.log1p(recent[365]),
            np.log1p(recent[730]),
            np.log1p(previous_182),
            np.log1p(previous_365),
            recent[30] / denominator,
            recent[90] / denominator,
            recent[182] / denominator,
            recent[365] / denominator,
            np.log1p(recent[182]) - np.log1p(previous_182),
            np.log1p(recent[365]) - np.log1p(previous_365),
            np.log1p(np.maximum(days_since_last, 0)),
            np.log1p(np.maximum(days_since_first, 0)),
            np.log1p(np.maximum(citation_span, 0)),
            cited.astype(np.float64),
            np.log1p(citer_mean),
            np.log1p(citer_maximum),
            np.log1p(out_total),
            np.log1p(out_missing),
            np.log1p(out_unique),
            safe_rate(out_known, out_total),
            np.log1p(reference_sum),
            np.log1p(reference_mean),
            np.log1p(reference_maximum),
            reference_cited,
            np.log1p(reference_recent),
            np.log1p(reference_age_mean),
            np.log1p(paper_author_count),
            author_name_length_mean,
            np.log1p(author_name_ambiguity_mean),
            np.log1p(author_papers_mean),
            np.log1p(author_papers_maximum),
            np.log1p(author_papers_minimum),
            np.log1p(author_recent_mean),
            np.log1p(author_recent_maximum),
            np.log1p(author_citations_mean),
            np.log1p(author_citations_maximum),
            np.log1p(author_cited_mean),
            np.log1p(author_cited_maximum),
            np.log1p(author_best_mean),
            np.log1p(author_best_maximum),
            np.log1p(author_career_mean),
            author_history_rate_mean,
            np.log1p(paper_category_count),
            np.log1p(category_papers_mean),
            np.log1p(category_papers_maximum),
            np.log1p(category_recent_mean),
            np.log1p(category_recent_maximum),
            np.log1p(category_citations_mean),
            np.log1p(category_citations_maximum),
            category_cited_fraction_mean,
            category_history_rate_mean,
            np.log1p(primary_category_citation_mean),
            np.log1p(paper_history_count),
            paper_history_rate,
            paper_history_latest,
            np.log1p(author_history_count_mean),
            np.log1p(category_history_count_mean),
            (paper_history_count == 0).astype(np.float64),
        ]
        if len(columns) != 80 or len(FEATURE_NAMES) != 80:
            raise RuntimeError("Feature contract requires exactly 80 columns")
        matrix = np.column_stack([column[seed_ids] for column in columns]).astype(np.float32)
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError("Non-finite relational features")
        print(f"[features] origin={origin_ts.date()} rows={len(seed_ids)} elapsed={time.time() - started:.1f}s")
        return matrix


def synthetic_rows(builder: FeatureBuilder, origin):
    origin_day = day_value(origin)
    ids = np.flatnonzero(builder.paper_sub_day <= origin_day).astype(np.int64)
    horizon = (
        builder.citation_known
        & (builder.citation_day > origin_day)
        & (builder.citation_day <= origin_day + 182)
    )
    positives = np.zeros(builder.n_papers, dtype=np.float32)
    positives[builder.citation_reference[horizon]] = 1.0
    labels = positives[ids]
    dates = np.full(len(ids), origin_day, dtype=np.int64)
    return ids, labels, dates


def save_array(path: Path, value: np.ndarray):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def build_common_bundle(ctx, builder: FeatureBuilder, cache_dir: Path):
    version = "rel_arxiv_paper_citation_features_v5_80"
    manifest_path = cache_dir / "common_manifest.json"
    paths = {name: cache_dir / f"{name}.npy" for name in ("train_x", "val_x", "test_x")}
    if manifest_path.exists() and all(path.exists() for path in paths.values()):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("version") == version:
            print("[features] loaded cached common matrices")
            return {name: np.load(path, mmap_mode="r") for name, path in paths.items()}

    train_df = ctx.train.df
    train_x = np.empty((len(train_df), 80), dtype=np.float32)
    for origin in sorted(train_df["date"].unique()):
        positions = np.flatnonzero(train_df["date"].to_numpy() == origin)
        ids = train_df["Paper_ID"].to_numpy(np.int64)[positions]
        train_x[positions] = builder.build(ids, origin, False)
    val_ids = ctx.val.df["Paper_ID"].to_numpy(np.int64)
    test_ids = ctx.test.df["Paper_ID"].to_numpy(np.int64)
    val_x = builder.build(val_ids, ctx.val.df["date"].iloc[0], False)
    test_x = builder.build(test_ids, ctx.test.df["date"].iloc[0], True)
    for name, value in (("train_x", train_x), ("val_x", val_x), ("test_x", test_x)):
        save_array(paths[name], value)
    manifest_path.write_text(json.dumps({"version": version, "features": FEATURE_NAMES}, indent=2))
    print("[features] cached common matrices")
    return {name: np.load(path, mmap_mode="r") for name, path in paths.items()}


def build_synthetic_bundle(builder: FeatureBuilder, cache_dir: Path):
    version = "rel_arxiv_paper_citation_synthetic_v5"
    specifications = {
        "internal": ("2020-10-03", False),
        "model_a": ("2021-04-03", False),
        "model_b_1": ("2022-04-02", True),
        "model_b_2": ("2022-07-02", True),
    }
    manifest_path = cache_dir / "synthetic_manifest.json"
    expected = []
    for name in specifications:
        expected.extend((cache_dir / f"{name}_{suffix}.npy" for suffix in ("x", "ids", "y", "days")))
    if manifest_path.exists() and all(path.exists() for path in expected):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("version") == version:
            print("[features] loaded cached synthetic matrices")
            return {
                name: {
                    suffix: np.load(cache_dir / f"{name}_{suffix}.npy", mmap_mode="r")
                    for suffix in ("x", "ids", "y", "days")
                }
                for name in specifications
            }
    result = {}
    for name, (origin, allow_validation) in specifications.items():
        ids, labels, days = synthetic_rows(builder, origin)
        matrix = builder.build(ids, origin, allow_validation)
        result[name] = {"x": matrix, "ids": ids, "y": labels, "days": days}
        for suffix, value in result[name].items():
            save_array(cache_dir / f"{name}_{suffix}.npy", np.asarray(value))
    manifest_path.write_text(json.dumps({"version": version, "specifications": specifications}, indent=2))
    print("[features] cached synthetic matrices")
    return {
        name: {
            suffix: np.load(cache_dir / f"{name}_{suffix}.npy", mmap_mode="r")
            for suffix in ("x", "ids", "y", "days")
        }
        for name in specifications
    }
