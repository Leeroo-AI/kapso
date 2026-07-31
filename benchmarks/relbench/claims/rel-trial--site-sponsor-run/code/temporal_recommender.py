from __future__ import annotations

import json
import math
import os
import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


# Configuration

@dataclass
class RecommenderConfig:
    seed: int = 1337
    memory_dim: int = 128
    time_dim: int = 32
    message_dim: int = 192
    retrieval_dim: int = 128
    temporal_neighbors: int = 20
    attention_heads: int = 2
    dropout: float = 0.15
    query_batch: int = 256
    event_batch: int = 2048
    negatives: int = 192
    temperature: float = 0.07
    smooth_ap_weight: float = 0.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    model_a_epochs: int = 2
    model_b_epochs: int = 1
    neural_retrieval: int = 512
    deterministic_union: int = 128
    pool_cap: int = 640
    eval_k: int = 10
    decay_days: float = 730.0
    debug: bool = False
    debug_events: int = 100000
    debug_queries: int = 10000
    version: str = "lane3_horizon_tgn_v1"

    def apply_debug(self) -> None:
        if not self.debug:
            return
        self.memory_dim = 64
        self.query_batch = 256
        self.negatives = 64
        self.model_a_epochs = 1
        self.model_b_epochs = 0
        self.neural_retrieval = 128
        self.deterministic_union = 64
        self.pool_cap = 192


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.2f}s"


# Data

class TemporalData:
    def __init__(self, config: RecommenderConfig, cache_root: Path, database_root: Path):
        self.config = config
        self.cache_dir = cache_root / config.version
        self.database_root = database_root
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.cache_dir / "events_v1.npz"
        self.static_path = self.cache_dir / "static_v7.npz"
        self.late_path = self.cache_dir / "late_v2.npz"
        self.index_path = self.cache_dir / "indices_v7.npz"
        self.co_path = self.cache_dir / "cosponsor_v1.npz"
        self.alias_path = self.cache_dir / "sponsor_alias_v1.npz"
        self._prepare()
        self._load()

    def _prepare(self) -> None:
        if not self.static_path.exists():
            self._build_static()
        if not self.events_path.exists():
            self._build_events()
        if not self.late_path.exists():
            self._build_late()
        if not self.index_path.exists():
            self._build_indices()
        if not self.co_path.exists():
            self._build_cosponsors()
        if not self.alias_path.exists():
            self._build_aliases()

    def _build_static(self) -> None:
        r = str(self.database_root)
        con = duckdb.connect()
        facilities = con.sql(
            f"""
            SELECT
                facility_id,
                hash(lower(coalesce(name, ''))) name_hash,
                hash(concat_ws('|', coalesce(country, ''), coalesce(state, ''), coalesce(city, ''))) city_hash,
                hash(concat_ws('|', coalesce(country, ''), coalesce(state, ''))) state_hash,
                hash(coalesce(country, '')) country_hash,
                hash(concat_ws('|', coalesce(country, ''),
                     regexp_replace(
                         regexp_replace(lower(coalesce(name, '')), '[0-9]+', '#', 'g'),
                         '[^[:alnum:]#]+', ' ', 'g'
                     ))) template_hash,
                hash(
                     regexp_replace(
                         regexp_replace(lower(coalesce(name, '')), '[0-9]+', '#', 'g'),
                         '[^[:alnum:]#]+', ' ', 'g'
                     )) global_template_hash,
                CASE
                    WHEN zip IS NULL OR trim(zip) = ''
                    THEN hash(concat('missing|', facility_id::VARCHAR))
                    ELSE hash(concat_ws('|', coalesce(country, ''),
                              regexp_replace(lower(zip), '[^[:alnum:]]+', '', 'g')))
                END zip_hash,
                CASE
                    WHEN regexp_extract(coalesce(name, ''), '[0-9]+', 0) = ''
                    THEN hash(concat('missing-digits|', facility_id::VARCHAR))
                    ELSE hash(concat_ws('|', coalesce(country, ''),
                         regexp_replace(
                             regexp_replace(lower(coalesce(name, '')), '[0-9]+', '#', 'g'),
                             '[^[:alnum:]#]+', ' ', 'g'
                         ),
                         length(regexp_extract(coalesce(name, ''), '[0-9]+', 0))::VARCHAR
                    ))
                END digit_template_hash
            FROM read_parquet('{r}/facilities.parquet')
            ORDER BY facility_id
            """
        ).df()
        sponsors = con.sql(
            f"""
            SELECT
                sponsor_id,
                hash(lower(coalesce(name, ''))) name_hash,
                hash(coalesce(agency_class, '')) class_hash
            FROM read_parquet('{r}/sponsors.parquet')
            ORDER BY sponsor_id
            """
        ).df()
        n_facilities = int(facilities.facility_id.max()) + 1
        n_sponsors = int(sponsors.sponsor_id.max()) + 1
        fac_name = np.zeros(n_facilities, dtype=np.uint64)
        city_hash = np.zeros(n_facilities, dtype=np.uint64)
        state_hash = np.zeros(n_facilities, dtype=np.uint64)
        country_hash = np.zeros(n_facilities, dtype=np.uint64)
        template_hash = np.zeros(n_facilities, dtype=np.uint64)
        global_template_hash = np.zeros(n_facilities, dtype=np.uint64)
        zip_hash = np.zeros(n_facilities, dtype=np.uint64)
        digit_template_hash = np.zeros(n_facilities, dtype=np.uint64)
        fi = facilities.facility_id.to_numpy(np.int64)
        fac_name[fi] = facilities.name_hash.to_numpy(np.uint64)
        city_hash[fi] = facilities.city_hash.to_numpy(np.uint64)
        state_hash[fi] = facilities.state_hash.to_numpy(np.uint64)
        country_hash[fi] = facilities.country_hash.to_numpy(np.uint64)
        template_hash[fi] = facilities.template_hash.to_numpy(np.uint64)
        global_template_hash[fi] = facilities.global_template_hash.to_numpy(np.uint64)
        zip_hash[fi] = facilities.zip_hash.to_numpy(np.uint64)
        digit_template_hash[fi] = facilities.digit_template_hash.to_numpy(np.uint64)
        _, city_group = np.unique(city_hash, return_inverse=True)
        _, state_group = np.unique(state_hash, return_inverse=True)
        _, country_group = np.unique(country_hash, return_inverse=True)
        _, template_group = np.unique(template_hash, return_inverse=True)
        _, global_template_group = np.unique(
            global_template_hash, return_inverse=True
        )
        _, zip_group = np.unique(zip_hash, return_inverse=True)
        _, digit_template_group = np.unique(
            digit_template_hash, return_inverse=True
        )
        sponsor_name = np.zeros(n_sponsors, dtype=np.uint64)
        sponsor_class = np.zeros(n_sponsors, dtype=np.uint64)
        si = sponsors.sponsor_id.to_numpy(np.int64)
        sponsor_name[si] = sponsors.name_hash.to_numpy(np.uint64)
        sponsor_class[si] = sponsors.class_hash.to_numpy(np.uint64)
        np.savez(
            self.static_path,
            n_facilities=np.int64(n_facilities),
            n_sponsors=np.int64(n_sponsors),
            fac_name=(fac_name % 65536).astype(np.int32),
            fac_city=(city_hash % 65536).astype(np.int32),
            fac_state=(state_hash % 16384).astype(np.int32),
            fac_country=(country_hash % 1024).astype(np.int32),
            city_group=city_group.astype(np.int32),
            state_group=state_group.astype(np.int32),
            country_group=country_group.astype(np.int32),
            template_group=template_group.astype(np.int32),
            global_template_group=global_template_group.astype(np.int32),
            zip_group=zip_group.astype(np.int32),
            digit_template_group=digit_template_group.astype(np.int32),
            sponsor_name=(sponsor_name % 65536).astype(np.int32),
            sponsor_class=(sponsor_class % 1024).astype(np.int32),
        )

    def _build_events(self) -> None:
        r = str(self.database_root)
        con = duckdb.connect()
        frame = con.sql(
            f"""
            WITH condition_hash AS (
                SELECT nct_id, bit_xor(hash(condition_id)) hash_value
                FROM read_parquet('{r}/conditions_studies.parquet')
                GROUP BY nct_id
            ),
            intervention_hash AS (
                SELECT nct_id, bit_xor(hash(intervention_id)) hash_value
                FROM read_parquet('{r}/interventions_studies.parquet')
                GROUP BY nct_id
            ),
            design_hash AS (
                SELECT
                    nct_id,
                    hash(concat_ws('|', coalesce(allocation, ''), coalesce(intervention_model, ''),
                         coalesce(observational_model, ''), coalesce(primary_purpose, ''),
                         coalesce(masking, ''))) hash_value
                FROM read_parquet('{r}/designs.parquet')
            ),
            eligibility_hash AS (
                SELECT nct_id, hash(concat_ws('|', coalesce(gender, ''), coalesce(criteria, ''),
                       coalesce(minimum_age, ''), coalesce(maximum_age, ''))) hash_value
                FROM read_parquet('{r}/eligibilities.parquet')
            )
            SELECT
                fs.facility_id,
                ss.sponsor_id,
                fs.nct_id,
                CAST(epoch(fs.date::TIMESTAMP) / 86400 AS INTEGER) date_day,
                CASE WHEN lower(coalesce(ss.lead_or_collaborator, '')) LIKE 'lead%' THEN 1 ELSE 0 END lead,
                hash(coalesce(st.study_type, '')) type_hash,
                hash(coalesce(st.phase, '')) phase_hash,
                hash(CAST(floor(log2(coalesce(st.enrollment, 0) + 1)) AS VARCHAR)) enrollment_hash,
                hash(coalesce(f.city, '')) city_hash,
                hash(coalesce(f.state, '')) state_hash,
                hash(coalesce(f.country, '')) country_hash,
                hash(coalesce(sp.agency_class, '')) class_hash,
                coalesce(condition_hash.hash_value, 0) condition_hash,
                coalesce(intervention_hash.hash_value, 0) intervention_hash,
                coalesce(design_hash.hash_value, 0) design_hash,
                coalesce(eligibility_hash.hash_value, 0) eligibility_hash,
                hash(coalesce(st.brief_title, '')) title_hash
            FROM read_parquet('{r}/facilities_studies.parquet') fs
            JOIN read_parquet('{r}/sponsors_studies.parquet') ss USING (nct_id)
            JOIN read_parquet('{r}/studies.parquet') st USING (nct_id)
            JOIN read_parquet('{r}/facilities.parquet') f USING (facility_id)
            JOIN read_parquet('{r}/sponsors.parquet') sp USING (sponsor_id)
            LEFT JOIN condition_hash USING (nct_id)
            LEFT JOIN intervention_hash USING (nct_id)
            LEFT JOIN design_hash USING (nct_id)
            LEFT JOIN eligibility_hash USING (nct_id)
            QUALIFY row_number() OVER (
                PARTITION BY fs.facility_id, ss.sponsor_id, fs.nct_id
                ORDER BY fs.id, ss.id
            ) = 1
            ORDER BY date_day, facility_id, sponsor_id
            """
        ).df()
        hash_columns = [
            "type_hash",
            "phase_hash",
            "enrollment_hash",
            "city_hash",
            "state_hash",
            "country_hash",
            "class_hash",
            "condition_hash",
            "intervention_hash",
            "design_hash",
            "eligibility_hash",
            "title_hash",
        ]
        raw = frame[hash_columns].to_numpy(np.uint64, copy=True)
        seeds = np.arange(raw.shape[1], dtype=np.uint64)[None, :] * np.uint64(0x9E3779B185EBCA87)
        mixed = raw ^ seeds
        message_index = (mixed % np.uint64(self.config.message_dim)).astype(np.uint8)
        message_sign = (((mixed >> np.uint64(23)) & np.uint64(1)).astype(np.int8) * 2 - 1)
        first_day = frame.groupby("facility_id", sort=False).date_day.transform("min").to_numpy(np.int32)
        new_site = (frame.date_day.to_numpy(np.int32) == first_day).astype(np.uint8)
        np.savez(
            self.events_path,
            facility_id=frame.facility_id.to_numpy(np.int32),
            sponsor_id=frame.sponsor_id.to_numpy(np.int32),
            date_day=frame.date_day.to_numpy(np.int32),
            lead=frame.lead.to_numpy(np.uint8),
            new_site=new_site,
            message_index=message_index,
            message_sign=message_sign,
        )

    def _build_late(self) -> None:
        r = str(self.database_root)
        con = duckdb.connect()
        aux = f"""
            SELECT nct_id, date, 0 kind FROM read_parquet('{r}/outcomes.parquet')
            UNION ALL
            SELECT nct_id, date, 1 kind FROM read_parquet('{r}/outcome_analyses.parquet')
            UNION ALL
            SELECT nct_id, date, 2 kind FROM read_parquet('{r}/drop_withdrawals.parquet')
            UNION ALL
            SELECT nct_id, date, 3 kind FROM read_parquet('{r}/reported_event_totals.parquet')
        """
        facility = con.sql(
            f"""
            WITH aux AS ({aux}),
            links AS (
                SELECT DISTINCT nct_id, facility_id node_id
                FROM read_parquet('{r}/facilities_studies.parquet')
            )
            SELECT
                node_id,
                CAST(epoch(date::TIMESTAMP) / 86400 AS INTEGER) date_day,
                kind,
                count(*) amount
            FROM aux
            JOIN links USING (nct_id)
            GROUP BY node_id, date_day, kind
            ORDER BY date_day, node_id, kind
            """
        ).df()
        sponsor = con.sql(
            f"""
            WITH aux AS ({aux}),
            links AS (
                SELECT DISTINCT nct_id, sponsor_id node_id
                FROM read_parquet('{r}/sponsors_studies.parquet')
            )
            SELECT
                node_id,
                CAST(epoch(date::TIMESTAMP) / 86400 AS INTEGER) date_day,
                kind,
                count(*) amount
            FROM aux
            JOIN links USING (nct_id)
            GROUP BY node_id, date_day, kind
            ORDER BY date_day, node_id, kind
            """
        ).df()
        np.savez(
            self.late_path,
            fac_id=facility.node_id.to_numpy(np.int32),
            fac_day=facility.date_day.to_numpy(np.int32),
            fac_kind=facility.kind.to_numpy(np.uint8),
            fac_amount=facility.amount.to_numpy(np.float32),
            late_sponsor_id=sponsor.node_id.to_numpy(np.int32),
            late_sponsor_day=sponsor.date_day.to_numpy(np.int32),
            late_sponsor_kind=sponsor.kind.to_numpy(np.uint8),
            late_sponsor_amount=sponsor.amount.to_numpy(np.float32),
        )

    def _build_indices(self) -> None:
        static = np.load(self.static_path, allow_pickle=False)
        events = np.load(self.events_path, allow_pickle=False)
        sponsor_count = int(static["n_sponsors"])
        facility = events["facility_id"].astype(np.int64)
        sponsor = events["sponsor_id"].astype(np.int64)
        city = static["city_group"][facility].astype(np.int64)
        state = static["state_group"][facility].astype(np.int64)
        country = static["country_group"][facility].astype(np.int64)
        template = static["template_group"][facility].astype(np.int64)
        global_template = static["global_template_group"][facility].astype(np.int64)
        zip_group = static["zip_group"][facility].astype(np.int64)
        digit_template = static["digit_template_group"][facility].astype(np.int64)
        pair_key = facility * sponsor_count + sponsor
        city_key = city * sponsor_count + sponsor
        state_key = state * sponsor_count + sponsor
        country_key = country * sponsor_count + sponsor
        template_key = template * sponsor_count + sponsor
        global_template_key = global_template * sponsor_count + sponsor
        zip_key = zip_group * sponsor_count + sponsor
        digit_template_key = digit_template * sponsor_count + sponsor
        pair_unique, pair_inverse = np.unique(pair_key, return_inverse=True)
        city_unique, city_inverse = np.unique(city_key, return_inverse=True)
        state_unique, state_inverse = np.unique(state_key, return_inverse=True)
        country_unique, country_inverse = np.unique(country_key, return_inverse=True)
        template_unique, template_inverse = np.unique(template_key, return_inverse=True)
        global_template_unique, global_template_inverse = np.unique(
            global_template_key, return_inverse=True
        )
        zip_unique, zip_inverse = np.unique(zip_key, return_inverse=True)
        digit_template_unique, digit_template_inverse = np.unique(
            digit_template_key, return_inverse=True
        )
        np.savez(
            self.index_path,
            pair_unique=pair_unique,
            pair_inverse=pair_inverse.astype(np.int32),
            city_unique=city_unique,
            city_inverse=city_inverse.astype(np.int32),
            state_unique=state_unique,
            state_inverse=state_inverse.astype(np.int32),
            country_unique=country_unique,
            country_inverse=country_inverse.astype(np.int32),
            template_unique=template_unique,
            template_inverse=template_inverse.astype(np.int32),
            global_template_unique=global_template_unique,
            global_template_inverse=global_template_inverse.astype(np.int32),
            zip_unique=zip_unique,
            zip_inverse=zip_inverse.astype(np.int32),
            digit_template_unique=digit_template_unique,
            digit_template_inverse=digit_template_inverse.astype(np.int32),
        )

    def _build_cosponsors(self) -> None:
        r = str(self.database_root)
        con = duckdb.connect()
        frame = con.sql(
            f"""
            SELECT
                first.sponsor_id source_id,
                second.sponsor_id destination_id,
                CAST(epoch(first.date::TIMESTAMP) / 86400 AS INTEGER) date_day
            FROM read_parquet('{r}/sponsors_studies.parquet') first
            JOIN read_parquet('{r}/sponsors_studies.parquet') second USING (nct_id)
            WHERE first.sponsor_id <> second.sponsor_id
            QUALIFY row_number() OVER (
                PARTITION BY first.sponsor_id, second.sponsor_id, first.nct_id
                ORDER BY first.id, second.id
            ) = 1
            ORDER BY date_day, source_id, destination_id
            """
        ).df()
        keys = (
            frame.source_id.to_numpy(np.int64) * 53241
            + frame.destination_id.to_numpy(np.int64)
        )
        unique, inverse = np.unique(keys, return_inverse=True)
        np.savez(
            self.co_path,
            co_source=frame.source_id.to_numpy(np.int32),
            co_destination=frame.destination_id.to_numpy(np.int32),
            co_day=frame.date_day.to_numpy(np.int32),
            co_unique=unique,
            co_inverse=inverse.astype(np.int32),
        )

    def _build_aliases(self) -> None:
        facilities = pd.read_parquet(
            self.database_root / "facilities.parquet",
            columns=["facility_id", "name"],
        ).sort_values("facility_id")
        sponsors = pd.read_parquet(
            self.database_root / "sponsors.parquet",
            columns=["sponsor_id", "name"],
        )
        expression = re.compile(r"[^a-z0-9]+")
        suffixes = {
            "inc", "incorporated", "ltd", "limited", "llc", "gmbh", "corporation",
            "corp", "company", "co", "plc", "sa", "sas", "ag", "bv", "nv", "spa",
            "srl", "pty", "holdings", "group",
        }
        blocked = {
            "hospital", "university", "medical", "health", "clinic", "clinical",
            "research", "center", "centre", "institute", "foundation", "association",
            "services", "pharma", "pharmaceuticals", "laboratories", "government",
            "department", "national", "international", "regional", "community",
        }

        def tokens(value: object) -> list[str]:
            return [
                token
                for token in expression.sub(" ", str(value).lower()).strip().split()
                if token
            ]

        aliases: dict[str, list[int]] = {}
        lengths: set[int] = set()
        for sponsor_id, name in zip(sponsors.sponsor_id, sponsors.name):
            full = tokens(name)
            trimmed = list(full)
            while trimmed and trimmed[-1] in suffixes:
                trimmed.pop()
            for variant in (full, trimmed):
                key = " ".join(variant)
                if not variant or len(key) < 7:
                    continue
                if len(variant) == 1 and (
                    variant[0] in blocked or len(variant[0]) < 7
                ):
                    continue
                aliases.setdefault(key, []).append(int(sponsor_id))
                lengths.add(len(variant))
        pointer = np.zeros(int(facilities.facility_id.max()) + 2, dtype=np.int64)
        match_ids: list[int] = []
        match_quality: list[float] = []
        for facility_id, name in zip(facilities.facility_id, facilities.name.fillna("")):
            name_tokens = tokens(name)
            found: dict[int, float] = {}
            for length in lengths:
                if length > len(name_tokens):
                    continue
                for start in range(len(name_tokens) - length + 1):
                    key = " ".join(name_tokens[start : start + length])
                    values = aliases.get(key)
                    if values is None:
                        continue
                    quality = length + min(len(key), 50) / 100.0
                    for sponsor_id in values:
                        found[sponsor_id] = max(
                            quality, found.get(sponsor_id, 0.0)
                        )
            for sponsor_id, quality in sorted(
                found.items(), key=lambda item: item[1], reverse=True
            ):
                match_ids.append(sponsor_id)
                match_quality.append(quality)
            pointer[int(facility_id) + 1] = len(match_ids)
        np.maximum.accumulate(pointer, out=pointer)
        np.savez(
            self.alias_path,
            alias_pointer=pointer,
            alias_sponsor=np.asarray(match_ids, dtype=np.int32),
            alias_quality=np.asarray(match_quality, dtype=np.float32),
        )

    def _load(self) -> None:
        static = np.load(self.static_path, allow_pickle=False)
        events = np.load(self.events_path, allow_pickle=False)
        late = np.load(self.late_path, allow_pickle=False)
        indices = np.load(self.index_path, allow_pickle=False)
        cosponsors = np.load(self.co_path, allow_pickle=False)
        aliases = np.load(self.alias_path, allow_pickle=False)
        for key in static.files:
            setattr(self, key, static[key])
        for key in events.files:
            setattr(self, key, events[key])
        for key in late.files:
            setattr(self, key, late[key])
        for key in indices.files:
            setattr(self, key, indices[key])
        for key in cosponsors.files:
            setattr(self, key, cosponsors[key])
        for key in aliases.files:
            setattr(self, key, aliases[key])
        self.n_facilities = int(self.n_facilities)
        self.n_sponsors = int(self.n_sponsors)
        if self.config.debug:
            self.event_limit = min(self.config.debug_events, len(self.date_day))
        else:
            self.event_limit = len(self.date_day)


# Temporal state

class HorizonTemporalMemory:
    def __init__(self, data: TemporalData, config: RecommenderConfig, device: torch.device):
        self.data = data
        self.config = config
        self.device = device
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + 17)
        projection = torch.randn(
            config.message_dim, config.memory_dim, generator=generator, dtype=torch.float32
        )
        projection = F.normalize(projection, dim=0)
        self.projection = projection.to(device)
        self.memory = torch.zeros(
            data.n_facilities + data.n_sponsors,
            config.memory_dim,
            dtype=torch.float32,
            device=device,
        )
        self.pointer = 0
        self.cutoff = None

    def reset(self) -> None:
        self.memory.zero_()
        self.pointer = 0
        self.cutoff = None

    def advance(self, cutoff: int) -> None:
        if self.cutoff is not None:
            self.memory.mul_(math.exp(-(cutoff - self.cutoff) / self.config.decay_days))
        end = min(
            int(np.searchsorted(self.data.date_day, cutoff, side="right")),
            self.data.event_limit,
        )
        for lo in range(self.pointer, end, self.config.event_batch):
            hi = min(lo + self.config.event_batch, end)
            indices = torch.as_tensor(
                self.data.message_index[lo:hi].astype(np.int64), device=self.device
            )
            signs = torch.as_tensor(
                self.data.message_sign[lo:hi].astype(np.float32), device=self.device
            )
            message = torch.zeros(
                hi - lo, self.config.message_dim, device=self.device, dtype=torch.float32
            )
            message.scatter_add_(1, indices, signs)
            encoded = message @ self.projection
            days = torch.as_tensor(
                self.data.date_day[lo:hi].astype(np.float32), device=self.device
            )
            encoded.mul_(torch.exp(-(cutoff - days) / self.config.decay_days).unsqueeze(1))
            facility = torch.as_tensor(
                self.data.facility_id[lo:hi].astype(np.int64), device=self.device
            )
            sponsor = torch.as_tensor(
                self.data.sponsor_id[lo:hi].astype(np.int64), device=self.device
            )
            self.memory.index_add_(0, facility, encoded)
            self.memory.index_add_(0, sponsor + self.data.n_facilities, encoded)
        self.pointer = end
        self.cutoff = cutoff

    def facility(self, ids: torch.Tensor) -> torch.Tensor:
        return self.memory[ids]

    def sponsor(self, ids: torch.Tensor) -> torch.Tensor:
        return self.memory[ids + self.data.n_facilities]


# Temporal features

class TemporalFeatureIndex:
    def __init__(self, data: TemporalData, config: RecommenderConfig):
        self.data = data
        self.config = config
        self._arrays = [
            ("pair", data.pair_unique),
            ("city", data.city_unique),
            ("state", data.state_unique),
            ("country", data.country_unique),
            ("template", data.template_unique),
            ("global_template", data.global_template_unique),
            ("zip", data.zip_unique),
            ("digit_template", data.digit_template_unique),
        ]
        self.reset()

    def reset(self) -> None:
        for name, keys in self._arrays:
            setattr(self, f"{name}_decay", np.zeros(len(keys), dtype=np.float32))
            setattr(self, f"{name}_total", np.zeros(len(keys), dtype=np.float32))
            setattr(self, f"{name}_last", np.full(len(keys), -100000, dtype=np.int32))
        self.pair_lead = np.zeros(len(self.data.pair_unique), dtype=np.float32)
        self.pair_first = np.full(len(self.data.pair_unique), 100000, dtype=np.int32)
        self.fac_decay = np.zeros(self.data.n_facilities, dtype=np.float32)
        self.fac_total = np.zeros(self.data.n_facilities, dtype=np.float32)
        self.fac_last = np.full(self.data.n_facilities, -100000, dtype=np.int32)
        self.sponsor_decay = np.zeros(self.data.n_sponsors, dtype=np.float32)
        self.sponsor_total = np.zeros(self.data.n_sponsors, dtype=np.float32)
        self.sponsor_last = np.full(self.data.n_sponsors, -100000, dtype=np.int32)
        self.sponsor_new_site = np.zeros(self.data.n_sponsors, dtype=np.float32)
        self.fac_late = np.zeros((self.data.n_facilities, 4), dtype=np.float32)
        self.sponsor_late = np.zeros((self.data.n_sponsors, 4), dtype=np.float32)
        self.co_decay = np.zeros(len(self.data.co_unique), dtype=np.float32)
        self.co_total = np.zeros(len(self.data.co_unique), dtype=np.float32)
        self.co_last = np.full(len(self.data.co_unique), -100000, dtype=np.int32)
        self.pointer = 0
        self.fac_late_pointer = 0
        self.sponsor_late_pointer = 0
        self.co_pointer = 0
        self.cutoff = None
        self.generation = 0
        self.group_cache: dict[tuple[str, int, int], np.ndarray] = {}
        self.co_cache: dict[int, dict[int, float]] = {}

    def advance(self, cutoff: int) -> None:
        if self.cutoff is not None:
            factor = np.float32(math.exp(-(cutoff - self.cutoff) / self.config.decay_days))
            for name, _ in self._arrays:
                getattr(self, f"{name}_decay")[:] *= factor
            self.fac_decay *= factor
            self.sponsor_decay *= factor
            self.co_decay *= factor
        end = min(
            int(np.searchsorted(self.data.date_day, cutoff, side="right")),
            self.data.event_limit,
        )
        for lo in range(self.pointer, end, 200000):
            hi = min(lo + 200000, end)
            day = self.data.date_day[lo:hi]
            weights = np.exp(-(cutoff - day) / self.config.decay_days).astype(np.float32)
            for name, _ in self._arrays:
                inverse = getattr(self.data, f"{name}_inverse")[lo:hi]
                np.add.at(getattr(self, f"{name}_decay"), inverse, weights)
                np.add.at(getattr(self, f"{name}_total"), inverse, 1)
                np.maximum.at(getattr(self, f"{name}_last"), inverse, day)
            pair_inverse = self.data.pair_inverse[lo:hi]
            np.add.at(self.pair_lead, pair_inverse, self.data.lead[lo:hi])
            np.minimum.at(self.pair_first, pair_inverse, day)
            facility = self.data.facility_id[lo:hi]
            sponsor = self.data.sponsor_id[lo:hi]
            np.add.at(self.fac_decay, facility, weights)
            np.add.at(self.fac_total, facility, 1)
            np.maximum.at(self.fac_last, facility, day)
            np.add.at(self.sponsor_decay, sponsor, weights)
            np.add.at(self.sponsor_total, sponsor, 1)
            np.maximum.at(self.sponsor_last, sponsor, day)
            np.add.at(
                self.sponsor_new_site,
                sponsor,
                self.data.new_site[lo:hi].astype(np.float32),
            )
        self.pointer = end
        co_end = int(np.searchsorted(self.data.co_day, cutoff, side="right"))
        for lo in range(self.co_pointer, co_end, 200000):
            hi = min(lo + 200000, co_end)
            day = self.data.co_day[lo:hi]
            weights = np.exp(-(cutoff - day) / self.config.decay_days).astype(np.float32)
            inverse = self.data.co_inverse[lo:hi]
            np.add.at(self.co_decay, inverse, weights)
            np.add.at(self.co_total, inverse, 1)
            np.maximum.at(self.co_last, inverse, day)
        self.co_pointer = co_end
        self.fac_late_pointer = self._advance_late(
            self.data.fac_day,
            self.data.fac_id,
            self.data.fac_kind,
            self.data.fac_amount,
            self.fac_late,
            self.fac_late_pointer,
            cutoff,
        )
        self.sponsor_late_pointer = self._advance_late(
            self.data.late_sponsor_day,
            self.data.late_sponsor_id,
            self.data.late_sponsor_kind,
            self.data.late_sponsor_amount,
            self.sponsor_late,
            self.sponsor_late_pointer,
            cutoff,
        )
        self.cutoff = cutoff
        self.generation += 1
        self.group_cache.clear()
        self.co_cache.clear()

    def _advance_late(
        self,
        days: np.ndarray,
        ids: np.ndarray,
        kinds: np.ndarray,
        amounts: np.ndarray,
        target: np.ndarray,
        pointer: int,
        cutoff: int,
    ) -> int:
        end = int(np.searchsorted(days, cutoff, side="right"))
        for lo in range(pointer, end, 200000):
            hi = min(lo + 200000, end)
            np.add.at(target, (ids[lo:hi], kinds[lo:hi]), amounts[lo:hi])
        return end

    def _lookup(
        self,
        keys: np.ndarray,
        unique: np.ndarray,
        values: np.ndarray,
        default: float,
    ) -> np.ndarray:
        flat = keys.reshape(-1)
        positions = np.searchsorted(unique, flat)
        valid = positions < len(unique)
        matched = np.zeros(len(flat), dtype=bool)
        matched[valid] = unique[positions[valid]] == flat[valid]
        output = np.full(len(flat), default, dtype=values.dtype)
        output[matched] = values[positions[matched]]
        return output.reshape(keys.shape)

    def facility_node_features(self, facility_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(facility_ids, dtype=np.int64)
        recent = np.clip(self.cutoff - self.fac_last[ids], 0, 3650) / 3650.0
        recent[self.fac_total[ids] == 0] = 1.0
        late = np.log1p(self.fac_late[ids])
        return np.column_stack(
            [
                np.log1p(self.fac_decay[ids]),
                np.log1p(self.fac_total[ids]),
                recent,
                (self.fac_total[ids] == 0).astype(np.float32),
                late,
            ]
        ).astype(np.float32)

    def sponsor_node_features(self, sponsor_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(sponsor_ids, dtype=np.int64)
        recent = np.clip(self.cutoff - self.sponsor_last[ids], 0, 3650) / 3650.0
        recent[self.sponsor_total[ids] == 0] = 1.0
        new_rate = self.sponsor_new_site[ids] / np.maximum(self.sponsor_total[ids], 1)
        late = np.log1p(self.sponsor_late[ids])
        return np.column_stack(
            [
                np.log1p(self.sponsor_decay[ids]),
                np.log1p(self.sponsor_total[ids]),
                recent,
                new_rate,
                late,
            ]
        ).astype(np.float32)

    def pair_features(self, facility_ids: np.ndarray, sponsor_ids: np.ndarray) -> np.ndarray:
        facility = np.asarray(facility_ids, dtype=np.int64).reshape(-1, 1)
        sponsor = np.asarray(sponsor_ids, dtype=np.int64)
        if sponsor.ndim == 1:
            sponsor = sponsor.reshape(len(facility), -1)
        facility = np.broadcast_to(facility, sponsor.shape)
        n_sponsors = self.data.n_sponsors
        direct_key = facility * n_sponsors + sponsor
        city_key = (
            self.data.city_group[facility].astype(np.int64) * n_sponsors + sponsor
        )
        state_key = (
            self.data.state_group[facility].astype(np.int64) * n_sponsors + sponsor
        )
        country_key = (
            self.data.country_group[facility].astype(np.int64) * n_sponsors + sponsor
        )
        template_key = (
            self.data.template_group[facility].astype(np.int64) * n_sponsors + sponsor
        )
        global_template_key = (
            self.data.global_template_group[facility].astype(np.int64)
            * n_sponsors
            + sponsor
        )
        zip_key = (
            self.data.zip_group[facility].astype(np.int64) * n_sponsors + sponsor
        )
        digit_template_key = (
            self.data.digit_template_group[facility].astype(np.int64)
            * n_sponsors
            + sponsor
        )
        direct_decay = self._lookup(
            direct_key, self.data.pair_unique, self.pair_decay, 0.0
        ).astype(np.float32)
        direct_total = self._lookup(
            direct_key, self.data.pair_unique, self.pair_total, 0.0
        ).astype(np.float32)
        direct_last = self._lookup(
            direct_key, self.data.pair_unique, self.pair_last, -100000
        ).astype(np.int32)
        direct_lead = self._lookup(
            direct_key, self.data.pair_unique, self.pair_lead, 0.0
        ).astype(np.float32)
        city_decay = self._lookup(
            city_key, self.data.city_unique, self.city_decay, 0.0
        ).astype(np.float32)
        state_decay = self._lookup(
            state_key, self.data.state_unique, self.state_decay, 0.0
        ).astype(np.float32)
        country_decay = self._lookup(
            country_key, self.data.country_unique, self.country_decay, 0.0
        ).astype(np.float32)
        template_decay = self._lookup(
            template_key, self.data.template_unique, self.template_decay, 0.0
        ).astype(np.float32)
        global_template_decay = self._lookup(
            global_template_key,
            self.data.global_template_unique,
            self.global_template_decay,
            0.0,
        ).astype(np.float32)
        zip_decay = self._lookup(
            zip_key, self.data.zip_unique, self.zip_decay, 0.0
        ).astype(np.float32)
        digit_template_decay = self._lookup(
            digit_template_key,
            self.data.digit_template_unique,
            self.digit_template_decay,
            0.0,
        ).astype(np.float32)
        recency = np.exp(
            -np.clip(self.cutoff - direct_last, 0, 3650).astype(np.float32) / 730.0
        )
        recency[direct_total == 0] = 0
        lead_share = direct_lead / np.maximum(direct_total, 1)
        sponsor_decay = self.sponsor_decay[sponsor]
        sponsor_total = self.sponsor_total[sponsor]
        sponsor_momentum = sponsor_decay / np.sqrt(np.maximum(sponsor_total, 1))
        new_rate = self.sponsor_new_site[sponsor] / np.maximum(sponsor_total, 1)
        facility_decay = self.fac_decay[facility]
        facility_total = self.fac_total[facility]
        facility_recent = np.exp(
            -np.clip(self.cutoff - self.fac_last[facility], 0, 3650).astype(np.float32)
            / 730.0
        )
        facility_recent[facility_total == 0] = 0
        cosponsor = self.cosponsor_scores(facility[:, 0], sponsor)
        cadence = self.cadence_scores(facility[:, 0], sponsor)
        alias = self.alias_scores(facility[:, 0], sponsor)
        heuristic = (
            50.00 * (direct_total > 0).astype(np.float32)
            + 7.00 * np.log1p(direct_decay)
            + 3.00 * recency
            + 0.25 * lead_share
            + 5.00 * np.log1p(city_decay)
            + 0.20 * np.log1p(state_decay)
            + 0.01 * np.log1p(country_decay)
            + 0.00 * np.log1p(sponsor_decay)
            + 0.00 * sponsor_momentum
            + 0.00 * new_rate
            + 20.00 * np.log1p(template_decay)
            + 10.00 * np.log1p(global_template_decay)
            + 2.00 * np.log1p(zip_decay)
            + 20.00 * np.log1p(digit_template_decay)
            + 0.50 * cosponsor
            + 2.00 * cadence
            + 5.00 * alias
        )
        output = np.stack(
            [
                np.log1p(direct_decay),
                np.log1p(direct_total),
                recency,
                lead_share,
                (direct_total > 0).astype(np.float32),
                np.log1p(city_decay),
                np.log1p(state_decay),
                np.log1p(country_decay),
                np.log1p(sponsor_decay),
                np.log1p(sponsor_total),
                sponsor_momentum,
                new_rate,
                np.log1p(facility_decay),
                np.log1p(facility_total),
                facility_recent,
                (facility_total == 0).astype(np.float32),
                heuristic,
            ],
            axis=-1,
        )
        return output.astype(np.float32)

    def alias_scores(
        self,
        facility_ids: np.ndarray,
        sponsor_ids: np.ndarray,
    ) -> np.ndarray:
        facilities = np.asarray(facility_ids, dtype=np.int64)
        sponsors = np.asarray(sponsor_ids, dtype=np.int64)
        output = np.zeros(sponsors.shape, dtype=np.float32)
        for row, facility in enumerate(facilities):
            lo = int(self.data.alias_pointer[facility])
            hi = int(self.data.alias_pointer[facility + 1])
            mapping = {
                int(sponsor): float(quality)
                for sponsor, quality in zip(
                    self.data.alias_sponsor[lo:hi],
                    self.data.alias_quality[lo:hi],
                )
            }
            output[row] = np.fromiter(
                (mapping.get(int(sponsor), 0.0) for sponsor in sponsors[row]),
                dtype=np.float32,
                count=sponsors.shape[1],
            )
        return output

    def cadence_scores(
        self,
        facility_ids: np.ndarray,
        sponsor_ids: np.ndarray,
    ) -> np.ndarray:
        facilities = np.asarray(facility_ids, dtype=np.int64).reshape(-1, 1)
        sponsors = np.asarray(sponsor_ids, dtype=np.int64)
        keys = facilities * self.data.n_sponsors + sponsors
        total = self._lookup(
            keys, self.data.pair_unique, self.pair_total, 0.0
        ).astype(np.float32)
        first = self._lookup(
            keys, self.data.pair_unique, self.pair_first, 100000
        ).astype(np.float32)
        last = self._lookup(
            keys, self.data.pair_unique, self.pair_last, -100000
        ).astype(np.float32)
        valid = (total >= 2) & (last > first)
        mean_gap = np.maximum((last - first) / np.maximum(total - 1, 1), 30.0)
        since = np.maximum(self.cutoff - last, 0)
        phase = np.mod(since, mean_gap)
        wait = np.where(phase == 0, mean_gap, mean_gap - phase)
        score = np.exp(-wait / 365.0) * np.minimum(np.log1p(total), 3.0)
        score[~valid] = 0
        return score.astype(np.float32)

    def _cosponsor_map(self, facility: int) -> dict[int, float]:
        facility = int(facility)
        if facility in self.co_cache:
            return self.co_cache[facility]
        sources = self._top_group("pair", facility, 24)
        if len(sources) == 0:
            self.co_cache[facility] = {}
            return self.co_cache[facility]
        source_keys = facility * self.data.n_sponsors + sources.astype(np.int64)
        source_weight = self._lookup(
            source_keys, self.data.pair_unique, self.pair_decay, 0.0
        )
        scores: dict[int, float] = {}
        for source, weight in zip(sources, source_weight):
            lo = int(
                np.searchsorted(
                    self.data.co_unique, int(source) * self.data.n_sponsors
                )
            )
            hi = int(
                np.searchsorted(
                    self.data.co_unique, (int(source) + 1) * self.data.n_sponsors
                )
            )
            if hi <= lo:
                continue
            values = self.co_decay[lo:hi]
            active = np.flatnonzero(values > 0)
            if len(active) > 32:
                active = active[np.argpartition(values[active], -32)[-32:]]
            destinations = (
                self.data.co_unique[lo:hi][active] % self.data.n_sponsors
            ).astype(np.int32)
            contribution = np.log1p(float(weight)) * np.log1p(values[active])
            for destination, value in zip(destinations, contribution):
                key = int(destination)
                scores[key] = scores.get(key, 0.0) + float(value)
        if len(scores) > 128:
            best = sorted(scores, key=scores.get, reverse=True)[:128]
            scores = {key: scores[key] for key in best}
        self.co_cache[facility] = scores
        return scores

    def cosponsor_scores(
        self,
        facility_ids: np.ndarray,
        sponsor_ids: np.ndarray,
    ) -> np.ndarray:
        facilities = np.asarray(facility_ids, dtype=np.int64)
        sponsors = np.asarray(sponsor_ids, dtype=np.int64)
        output = np.zeros(sponsors.shape, dtype=np.float32)
        for row, facility in enumerate(facilities):
            mapping = self._cosponsor_map(int(facility))
            output[row] = np.fromiter(
                (mapping.get(int(sponsor), 0.0) for sponsor in sponsors[row]),
                dtype=np.float32,
                count=sponsors.shape[1],
            )
        return output

    def _top_group(self, name: str, group: int, k: int) -> np.ndarray:
        cache_key = (name, int(group), int(k))
        if cache_key in self.group_cache:
            return self.group_cache[cache_key]
        unique = getattr(self.data, f"{name}_unique")
        decay = getattr(self, f"{name}_decay")
        lo = int(np.searchsorted(unique, int(group) * self.data.n_sponsors))
        hi = int(np.searchsorted(unique, (int(group) + 1) * self.data.n_sponsors))
        if hi <= lo:
            result = np.empty(0, dtype=np.int32)
        else:
            scores = decay[lo:hi]
            active = np.flatnonzero(scores > 0)
            if len(active) > k:
                chosen = active[np.argpartition(scores[active], -k)[-k:]]
            else:
                chosen = active
            chosen = chosen[np.argsort(scores[chosen])[::-1]]
            result = (unique[lo:hi][chosen] % self.data.n_sponsors).astype(np.int32)
        self.group_cache[cache_key] = result
        return result

    def deterministic_candidates(self, facility_ids: np.ndarray, k: int) -> np.ndarray:
        facilities = np.asarray(facility_ids, dtype=np.int64)
        result = np.empty((len(facilities), k), dtype=np.int32)
        global_k = min(160, self.data.n_sponsors)
        active = np.flatnonzero(self.sponsor_decay > 0)
        if len(active) > global_k:
            chosen = active[np.argpartition(self.sponsor_decay[active], -global_k)[-global_k:]]
        else:
            chosen = active
        global_top = chosen[np.argsort(self.sponsor_decay[chosen])[::-1]].astype(np.int32)
        if len(global_top) < global_k:
            fill = np.setdiff1d(np.arange(self.data.n_sponsors, dtype=np.int32), global_top)
            global_top = np.concatenate([global_top, fill[: global_k - len(global_top)]])
        for row, facility in enumerate(facilities):
            pieces = [
                self._top_group("pair", int(facility), 96),
                self._top_group("city", int(self.data.city_group[facility]), 80),
                self._top_group("state", int(self.data.state_group[facility]), 48),
                self._top_group("country", int(self.data.country_group[facility]), 48),
                self._top_group("template", int(self.data.template_group[facility]), 96),
                self._top_group(
                    "global_template",
                    int(self.data.global_template_group[facility]),
                    96,
                ),
                self._top_group("zip", int(self.data.zip_group[facility]), 64),
                self._top_group(
                    "digit_template",
                    int(self.data.digit_template_group[facility]),
                    96,
                ),
                np.asarray(
                    list(self._cosponsor_map(int(facility)).keys()), dtype=np.int32
                ),
                self.data.alias_sponsor[
                    self.data.alias_pointer[facility] : self.data.alias_pointer[facility + 1]
                ],
                global_top,
            ]
            union = np.unique(np.concatenate(pieces))
            if len(union) < k:
                union = np.unique(np.concatenate([union, np.arange(k, dtype=np.int32)]))
            features = self.pair_features(
                np.array([facility], dtype=np.int64), union.reshape(1, -1)
            )[0, :, -1]
            if len(union) > k:
                take = np.argpartition(features, -k)[-k:]
                take = take[np.argsort(features[take])[::-1]]
            else:
                take = np.argsort(features)[::-1]
            values = union[take].tolist()
            used = set(values)
            for sponsor in global_top:
                if len(values) >= k:
                    break
                if int(sponsor) not in used:
                    values.append(int(sponsor))
                    used.add(int(sponsor))
            fill = 0
            while len(values) < k:
                if fill not in used:
                    values.append(fill)
                    used.add(fill)
                fill += 1
            result[row] = np.asarray(values[:k], dtype=np.int32)
        return result

    def hard_negative_candidates(self, facility_ids: np.ndarray, k: int) -> np.ndarray:
        facilities = np.asarray(facility_ids, dtype=np.int64)
        result = np.empty((len(facilities), k), dtype=np.int32)
        global_order = np.argsort(self.sponsor_decay)[::-1].astype(np.int32)
        for row, facility in enumerate(facilities):
            pieces = [
                self._top_group("pair", int(facility), min(32, k)),
                self._top_group("city", int(self.data.city_group[facility]), min(32, k)),
                self._top_group("state", int(self.data.state_group[facility]), min(16, k)),
                self._top_group("country", int(self.data.country_group[facility]), min(16, k)),
                self._top_group("template", int(self.data.template_group[facility]), min(24, k)),
                self._top_group(
                    "global_template",
                    int(self.data.global_template_group[facility]),
                    min(24, k),
                ),
                self._top_group(
                    "zip", int(self.data.zip_group[facility]), min(16, k)
                ),
                self._top_group(
                    "digit_template",
                    int(self.data.digit_template_group[facility]),
                    min(24, k),
                ),
                np.asarray(
                    list(self._cosponsor_map(int(facility)).keys())[: min(24, k)],
                    dtype=np.int32,
                ),
                self.data.alias_sponsor[
                    self.data.alias_pointer[facility] : self.data.alias_pointer[facility + 1]
                ],
                global_order[:k],
            ]
            values = []
            used = set()
            for piece in pieces:
                for value in piece:
                    value = int(value)
                    if value not in used:
                        values.append(value)
                        used.add(value)
                    if len(values) >= k:
                        break
                if len(values) >= k:
                    break
            fill = 0
            while len(values) < k:
                if fill not in used:
                    values.append(fill)
                    used.add(fill)
                fill += 1
            result[row] = np.asarray(values[:k], dtype=np.int32)
        return result


# Model

class TimeEncoder(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        frequency = 1.0 / (10.0 ** torch.linspace(0, 4, dimension))
        self.frequency = nn.Parameter(frequency)
        self.phase = nn.Parameter(torch.zeros(dimension))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.cos(value.unsqueeze(-1) * self.frequency + self.phase)


class HorizonGraphRecommender(nn.Module):
    pair_feature_dim = 17
    node_feature_dim = 8

    def __init__(self, data: TemporalData, config: RecommenderConfig):
        super().__init__()
        self.config = config
        self.facility_id = nn.Embedding(data.n_facilities, 32)
        self.facility_name = nn.Embedding(65536, 24)
        self.facility_city = nn.Embedding(65536, 24)
        self.facility_state = nn.Embedding(16384, 12)
        self.facility_country = nn.Embedding(1024, 12)
        self.sponsor_id = nn.Embedding(data.n_sponsors, 48)
        self.sponsor_name = nn.Embedding(65536, 24)
        self.sponsor_class = nn.Embedding(1024, 12)
        self.time_encoder = TimeEncoder(config.time_dim)
        fac_input = config.memory_dim + 32 + 24 + 24 + 12 + 12 + self.node_feature_dim + config.time_dim
        sponsor_input = config.memory_dim + 48 + 24 + 12 + self.node_feature_dim + config.time_dim
        self.facility_projection = nn.Sequential(
            nn.Linear(fac_input, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.retrieval_dim),
        )
        self.sponsor_projection = nn.Sequential(
            nn.Linear(sponsor_input, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.retrieval_dim),
        )
        scorer_input = self.pair_feature_dim + config.retrieval_dim
        self.recurrence = nn.Sequential(
            nn.Linear(scorer_input, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Linear(96, 1),
        )
        self.gate_residual = nn.Sequential(
            nn.Linear(self.pair_feature_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.compatibility = nn.Sequential(
            nn.Linear(config.retrieval_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.register_buffer("fac_name_code", torch.as_tensor(data.fac_name, dtype=torch.long))
        self.register_buffer("fac_city_code", torch.as_tensor(data.fac_city, dtype=torch.long))
        self.register_buffer("fac_state_code", torch.as_tensor(data.fac_state, dtype=torch.long))
        self.register_buffer("fac_country_code", torch.as_tensor(data.fac_country, dtype=torch.long))
        self.register_buffer("sponsor_name_code", torch.as_tensor(data.sponsor_name, dtype=torch.long))
        self.register_buffer("sponsor_class_code", torch.as_tensor(data.sponsor_class, dtype=torch.long))
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.025)
        nn.init.zeros_(self.recurrence[-1].weight)
        nn.init.zeros_(self.recurrence[-1].bias)
        nn.init.zeros_(self.gate_residual[-1].weight)
        nn.init.zeros_(self.gate_residual[-1].bias)
        nn.init.zeros_(self.compatibility[-1].weight)
        nn.init.zeros_(self.compatibility[-1].bias)

    def facility_vectors(
        self,
        ids: torch.Tensor,
        memory: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        encoded_time = self.time_encoder(numeric[:, 2] * 3650.0)
        values = torch.cat(
            [
                memory,
                self.facility_id(ids),
                self.facility_name(self.fac_name_code[ids]),
                self.facility_city(self.fac_city_code[ids]),
                self.facility_state(self.fac_state_code[ids]),
                self.facility_country(self.fac_country_code[ids]),
                numeric,
                encoded_time,
            ],
            dim=-1,
        )
        return F.normalize(self.facility_projection(values), dim=-1)

    def sponsor_vectors(
        self,
        ids: torch.Tensor,
        memory: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        encoded_time = self.time_encoder(numeric[:, 2] * 3650.0)
        values = torch.cat(
            [
                memory,
                self.sponsor_id(ids),
                self.sponsor_name(self.sponsor_name_code[ids]),
                self.sponsor_class(self.sponsor_class_code[ids]),
                numeric,
                encoded_time,
            ],
            dim=-1,
        )
        return F.normalize(self.sponsor_projection(values), dim=-1)

    def score(
        self,
        facility_vectors: torch.Tensor,
        sponsor_vectors: torch.Tensor,
        pair_features: torch.Tensor,
        conservative: bool = False,
    ) -> torch.Tensor:
        interaction = facility_vectors.unsqueeze(1) * sponsor_vectors
        exploration = interaction.sum(dim=-1) / self.config.temperature
        recurrence = pair_features[..., -1] + self.recurrence(
            torch.cat([pair_features, interaction], dim=-1)
        ).squeeze(-1)
        fixed_gate = 3.0 * (pair_features[..., 4] - 0.5) + 0.4 * pair_features[..., 13]
        gate = torch.sigmoid(
            fixed_gate + self.gate_residual(pair_features).squeeze(-1)
        )
        compatibility = self.compatibility(interaction).squeeze(-1)
        neural_score = (
            gate * recurrence
            + (1.0 - gate) * (exploration + 0.20 * pair_features[..., -1])
            + 0.10 * compatibility
        )
        if conservative:
            return pair_features[..., -1] + 0.02 * neural_score
        return neural_score


# Training and inference

def timestamp_day(values: pd.Series | np.ndarray) -> np.ndarray:
    return (
        pd.to_datetime(values).to_numpy(dtype="datetime64[D]").astype(np.int64)
    ).astype(np.int32)


def list_matrix(values: Iterable, maximum: int = 32) -> tuple[np.ndarray, np.ndarray]:
    rows = [np.asarray(value, dtype=np.int64)[:maximum] for value in values]
    width = max(1, max(len(row) for row in rows))
    matrix = np.zeros((len(rows), width), dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=bool)
    for index, row in enumerate(rows):
        matrix[index, : len(row)] = row
        mask[index, : len(row)] = True
    return matrix, mask


def sponsor_vector_table(
    model: HorizonGraphRecommender,
    memory: HorizonTemporalMemory,
    index: TemporalFeatureIndex,
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    output = []
    model.eval()
    with torch.no_grad():
        for lo in range(0, index.data.n_sponsors, batch_size):
            hi = min(lo + batch_size, index.data.n_sponsors)
            ids_np = np.arange(lo, hi, dtype=np.int64)
            ids = torch.as_tensor(ids_np, device=device)
            numeric = torch.as_tensor(
                index.sponsor_node_features(ids_np), device=device
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output.append(model.sponsor_vectors(ids, memory.sponsor(ids), numeric))
    return torch.cat(output, dim=0)


def build_negative_batch(
    facility_ids: np.ndarray,
    positives: np.ndarray,
    positive_mask: np.ndarray,
    nearest: np.ndarray,
    index: TemporalFeatureIndex,
    probability: np.ndarray,
    config: RecommenderConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    batch = len(facility_ids)
    count = config.negatives
    negatives = rng.choice(
        index.data.n_sponsors,
        size=(batch, count),
        replace=True,
        p=probability,
    ).astype(np.int64)
    hard_count = min(64, count // 2)
    geo_count = min(48, count - hard_count)
    negatives[:, :hard_count] = nearest[:, :hard_count]
    deterministic = index.hard_negative_candidates(facility_ids, max(geo_count, 1))
    negatives[:, hard_count : hard_count + geo_count] = deterministic[:, :geo_count]
    log_probability = np.log(
        np.maximum(probability[negatives] * float(config.negatives), 1e-12)
    ).astype(np.float32)
    replacement = rng.choice(
        index.data.n_sponsors,
        size=max(batch * count * 2, 1024),
        replace=True,
        p=probability,
    ).astype(np.int64)
    replacement_pointer = 0
    for row in range(batch):
        positive_set = set(positives[row, positive_mask[row]].tolist())
        used = set()
        for col in range(count):
            value = int(negatives[row, col])
            attempts = 0
            while (value in positive_set or value in used) and attempts < 16:
                if replacement_pointer < len(replacement):
                    value = int(replacement[replacement_pointer])
                    replacement_pointer += 1
                else:
                    value = (value + attempts + 1) % index.data.n_sponsors
                attempts += 1
            if value in positive_set or value in used:
                value = 0
                while value in positive_set or value in used:
                    value += 1
            negatives[row, col] = value
            used.add(value)
        log_probability[row] = np.log(
            np.maximum(probability[negatives[row]] * float(config.negatives), 1e-12)
        )
    return negatives, log_probability


def multi_positive_loss(
    positive_scores: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_scores: torch.Tensor,
    negative_log_probability: torch.Tensor,
) -> torch.Tensor:
    positive_scores = positive_scores.masked_fill(~positive_mask, -1e4)
    corrected_negative = negative_scores - negative_log_probability
    numerator = torch.logsumexp(positive_scores, dim=1)
    denominator = torch.logsumexp(
        torch.cat([positive_scores, corrected_negative], dim=1), dim=1
    )
    return (denominator - numerator).mean()


def smooth_ap_loss(
    positive_scores: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_scores: torch.Tensor,
) -> torch.Tensor:
    positive = positive_scores.masked_fill(~positive_mask, -1e4)
    differences = negative_scores.unsqueeze(1) - positive.unsqueeze(2)
    ranks = 1.0 + torch.sigmoid(differences / 0.1).sum(dim=2)
    precision = 1.0 / ranks
    return 1.0 - (precision * positive_mask).sum() / positive_mask.sum().clamp_min(1)


def train_query_group(
    model: HorizonGraphRecommender,
    memory: HorizonTemporalMemory,
    index: TemporalFeatureIndex,
    frame: pd.DataFrame,
    optimizer: torch.optim.Optimizer,
    config: RecommenderConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> float:
    model.train()
    sponsor_vectors = sponsor_vector_table(model, memory, index, device)
    model.train()
    popularity = np.power(index.sponsor_decay + 1e-3, 0.75)
    popularity[index.sponsor_total == 0] *= 0.05
    probability = popularity / popularity.sum()
    order = rng.permutation(len(frame))
    total_loss = 0.0
    total_rows = 0
    for lo in range(0, len(order), config.query_batch):
        take = order[lo : lo + config.query_batch]
        rows = frame.iloc[take]
        facility_np = rows.facility_id.to_numpy(np.int64)
        positive_np, positive_mask_np = list_matrix(rows.sponsor_id.values)
        facility_ids = torch.as_tensor(facility_np, device=device)
        facility_numeric = torch.as_tensor(
            index.facility_node_features(facility_np), device=device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            facility_vectors = model.facility_vectors(
                facility_ids, memory.facility(facility_ids), facility_numeric
            )
        with torch.no_grad():
            nearest_scores = facility_vectors.detach() @ sponsor_vectors.T
            unseen = torch.as_tensor(index.sponsor_total == 0, device=device)
            nearest_scores[:, unseen] = -1e4
            nearest_np = torch.topk(
                nearest_scores, min(96, index.data.n_sponsors), dim=1
            ).indices.cpu().numpy()
        negative_np, negative_log_np = build_negative_batch(
            facility_np,
            positive_np,
            positive_mask_np,
            nearest_np,
            index,
            probability,
            config,
            rng,
        )
        positive_ids = torch.as_tensor(positive_np, device=device)
        negative_ids = torch.as_tensor(negative_np, device=device)
        combined_np = np.concatenate([positive_np, negative_np], axis=1)
        combined_ids = torch.cat([positive_ids, negative_ids], dim=1)
        sponsor_numeric = torch.as_tensor(
            index.sponsor_node_features(combined_np.reshape(-1)),
            device=device,
        )
        sponsor_memory = memory.sponsor(combined_ids.reshape(-1))
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            sponsor_emb = model.sponsor_vectors(
                combined_ids.reshape(-1), sponsor_memory, sponsor_numeric
            ).reshape(len(rows), combined_ids.shape[1], -1)
        pair_features = torch.as_tensor(
            index.pair_features(facility_np, combined_np), device=device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            scores = model.score(facility_vectors, sponsor_emb, pair_features)
        width = positive_np.shape[1]
        positive_scores = scores[:, :width]
        negative_scores = scores[:, width:]
        positive_mask = torch.as_tensor(positive_mask_np, device=device)
        negative_log = torch.as_tensor(negative_log_np, device=device)
        loss = multi_positive_loss(
            positive_scores, positive_mask, negative_scores, negative_log
        )
        if config.smooth_ap_weight > 0:
            loss = loss + config.smooth_ap_weight * smooth_ap_loss(
                positive_scores, positive_mask, negative_scores
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        total_loss += float(loss.detach()) * len(rows)
        total_rows += len(rows)
    return total_loss / max(total_rows, 1)


def infer_predictions(
    model: HorizonGraphRecommender,
    memory: HorizonTemporalMemory,
    index: TemporalFeatureIndex,
    facility_ids: np.ndarray,
    config: RecommenderConfig,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    sponsor_vectors = sponsor_vector_table(model, memory, index, device)
    seen_mask = torch.as_tensor(index.sponsor_total > 0, device=device)
    output = np.empty((len(facility_ids), config.eval_k), dtype=np.int64)
    with torch.no_grad():
        for lo in range(0, len(facility_ids), config.query_batch):
            hi = min(lo + config.query_batch, len(facility_ids))
            facility_np = np.asarray(facility_ids[lo:hi], dtype=np.int64)
            facility = torch.as_tensor(facility_np, device=device)
            facility_numeric = torch.as_tensor(
                index.facility_node_features(facility_np), device=device
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                facility_vectors = model.facility_vectors(
                    facility, memory.facility(facility), facility_numeric
                )
            retrieval_scores = facility_vectors @ sponsor_vectors.T
            retrieval_scores[:, ~seen_mask] = -1e4
            retrieval = torch.topk(
                retrieval_scores,
                min(config.neural_retrieval, index.data.n_sponsors),
                dim=1,
            ).indices.cpu().numpy()
            deterministic = index.deterministic_candidates(
                facility_np, config.deterministic_union
            )
            pools = np.empty((len(facility_np), config.pool_cap), dtype=np.int64)
            for row in range(len(facility_np)):
                values = []
                used = set()
                for source in (retrieval[row], deterministic[row]):
                    for value in source:
                        value = int(value)
                        if value not in used:
                            values.append(value)
                            used.add(value)
                        if len(values) >= config.pool_cap:
                            break
                    if len(values) >= config.pool_cap:
                        break
                fill = 0
                while len(values) < config.pool_cap:
                    if fill not in used:
                        values.append(fill)
                        used.add(fill)
                    fill += 1
                pools[row] = np.asarray(values[: config.pool_cap], dtype=np.int64)
            pool_ids = torch.as_tensor(pools, device=device)
            sponsor_numeric = torch.as_tensor(
                index.sponsor_node_features(pools.reshape(-1)), device=device
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                pool_vectors = model.sponsor_vectors(
                    pool_ids.reshape(-1),
                    memory.sponsor(pool_ids.reshape(-1)),
                    sponsor_numeric,
                ).reshape(len(facility_np), config.pool_cap, -1)
            pair_features = torch.as_tensor(
                index.pair_features(facility_np, pools), device=device
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                scores = model.score(
                    facility_vectors,
                    pool_vectors,
                    pair_features,
                    conservative=True,
                )
            facility_state = F.normalize(memory.facility(facility), dim=-1)
            sponsor_state = F.normalize(
                memory.sponsor(pool_ids.reshape(-1)), dim=-1
            ).reshape(len(facility_np), config.pool_cap, -1)
            state_compatibility = torch.einsum(
                "bd,bkd->bk", facility_state, sponsor_state
            )
            scores = scores + 3.00 * state_compatibility
            top = torch.topk(scores, config.eval_k, dim=1).indices.cpu().numpy()
            output[lo:hi] = np.take_along_axis(pools, top, axis=1)
    return output


def average_precision_rows(predictions: np.ndarray, labels: Iterable) -> np.ndarray:
    output = np.zeros(len(predictions), dtype=np.float64)
    for row, (prediction, truth_values) in enumerate(zip(predictions, labels)):
        truth = set(np.asarray(truth_values, dtype=np.int64).tolist())
        if not truth:
            continue
        hits = 0
        score = 0.0
        for rank, sponsor in enumerate(prediction, 1):
            if int(sponsor) in truth:
                hits += 1
                score += hits / rank
        output[row] = score / min(len(truth), predictions.shape[1])
    return output


def bootstrap_standard_error(
    values: np.ndarray,
    clusters: np.ndarray,
    seed: int,
    repetitions: int = 200,
) -> float:
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(clusters, return_inverse=True)
    cluster_sum = np.bincount(inverse, weights=values)
    cluster_count = np.bincount(inverse)
    results = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = rng.integers(0, len(unique), size=len(unique))
        results[index] = cluster_sum[sample].sum() / cluster_count[sample].sum()
    return float(results.std(ddof=1))


def checkpoint_payload(
    model: HorizonGraphRecommender,
    memory: HorizonTemporalMemory,
    cutoff: int,
    config: RecommenderConfig,
) -> dict:
    return {
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "memory": memory.memory.detach().to(dtype=torch.float16, device="cpu"),
        "cutoff": int(cutoff),
        "config": asdict(config),
    }


def register_artifact(
    cache_root: Path,
    name: str,
    path: Path,
    description: str,
    content_key: str,
    rebuild_hint: str,
) -> None:
    import fcntl

    registry = cache_root / "artifacts.json"
    lock_path = cache_root / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if registry.exists():
            try:
                entries = json.loads(registry.read_text())
            except json.JSONDecodeError:
                entries = []
        else:
            entries = []
        if not any(entry.get("content_key") == content_key for entry in entries):
            entries.append(
                {
                    "name": name,
                    "path": str(path.relative_to(cache_root)),
                    "description": description,
                    "content_key": content_key,
                    "rebuild_hint": rebuild_hint,
                }
            )
            temporary = registry.with_suffix(".tmp")
            temporary.write_text(json.dumps(entries, indent=2))
            temporary.replace(registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def validate_predictions(
    predictions: np.ndarray,
    rows: int,
    sponsors: int,
    k: int,
) -> None:
    if predictions.shape != (rows, k):
        raise RuntimeError(f"prediction shape {predictions.shape} != {(rows, k)}")
    if not np.issubdtype(predictions.dtype, np.integer):
        raise RuntimeError(f"prediction dtype {predictions.dtype} is not integer")
    if predictions.min() < 0 or predictions.max() >= sponsors:
        raise RuntimeError("prediction sponsor IDs are out of range")
    if any(len(set(row.tolist())) != k for row in predictions):
        raise RuntimeError("prediction rows contain duplicate sponsor IDs")


def quiet_runtime() -> None:
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")
