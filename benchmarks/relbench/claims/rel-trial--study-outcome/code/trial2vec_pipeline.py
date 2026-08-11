import fcntl
import hashlib
import json
import math
import os
import random
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from transformers import logging as transformers_logging


warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()


@dataclass
class Config:
    debug: bool
    model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    field_count: int = 5
    max_length: int = 256
    embedding_dim: int = 256
    retrieval_neighbors: int = 32
    contrastive_temperature: float = 0.07
    self_supervised_batch: int = 128
    supervised_batch: int = 32
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_learning_rate: float = 2e-4
    head_learning_rate: float = 6e-4
    supervised_epochs: int = 5
    ranking_weight: float = 0.15
    modality_dropout: float = 0.2
    weight_decay: float = 0.02
    gradient_clip: float = 1.0
    seed: int = 1337

    @property
    def corpus_limit(self):
        return 2000 if self.debug else None

    @property
    def neighbor_count(self):
        return 4 if self.debug else self.retrieval_neighbors


class Clock:
    def __init__(self):
        self.start = time.time()
        self.last = self.start

    def mark(self, phase):
        now = time.time()
        print(f"[trial2vec] phase={phase} phase_seconds={now-self.last:.1f} elapsed_seconds={now-self.start:.1f}", flush=True)
        self.last = now


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid(values):
    values = np.clip(np.asarray(values, dtype=np.float64), -30, 30)
    return 1.0 / (1.0 + np.exp(-values))


def safe_auc(labels, predictions):
    labels = np.asarray(labels)
    return float(roc_auc_score(labels, predictions)) if np.unique(labels).size > 1 else 0.5


def cache_root():
    root = Path(os.environ.get("KAPSO_SHARED_CACHE_DIR", "./shared_cache"))
    root.mkdir(parents=True, exist_ok=True)
    lane = root / "trial2vec_streamed_lane3_v2"
    lane.mkdir(parents=True, exist_ok=True)
    return root, lane


def data_root():
    configured = os.environ.get("RELBENCH_CACHE_DIR")
    if configured:
        return Path(configured) / os.environ.get("RELBENCH_DATASET", "rel-trial")
    shared, _ = cache_root()
    return shared.parent / "sanitized_cache" / "rel-trial"


def register_artifact(path, name, description, content_key, rebuild_hint):
    shared, _ = cache_root()
    registry = shared / "artifacts.json"
    lock = shared / "artifacts.lock"
    with lock.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            entries = json.loads(registry.read_text()) if registry.exists() else []
        except Exception:
            entries = []
        relative = str(path.relative_to(shared))
        if not any(item.get("path") == relative for item in entries):
            entries.append({"name": name, "path": relative, "description": description, "content_key": content_key, "rebuild_hint": rebuild_hint})
            temporary = registry.with_suffix(".tmp_lane3")
            temporary.write_text(json.dumps(entries, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class TrialData:
    def __init__(self, root):
        self.root = Path(root)
        self.db = self.root / "db"
        self.tasks = self.root / "tasks" / "study-outcome"
        self.connection = duckdb.connect()
        self._evidence = {}

    def parquet(self, table):
        return str(self.db / f"{table}.parquet")

    def splits(self):
        train = pd.read_parquet(self.tasks / "train.parquet")
        val = pd.read_parquet(self.tasks / "val.parquet")
        test = pd.read_parquet(self.tasks / "test.parquet")
        for frame in (train, val, test):
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            frame["nct_id"] = frame["nct_id"].astype(np.int64)
        return train, val, test

    def corpus_entities(self, cutoff, limit=None):
        studies = self.parquet("studies")
        clause = f"limit {int(limit)}" if limit else ""
        return self.connection.sql(f"select row_number() over(order by start_date,nct_id)-1 as row_index,nct_id,timestamp '{cutoff}' as timestamp from read_parquet('{studies}') where start_date < timestamp '{cutoff}' order by start_date,nct_id {clause}").df()

    def completed_entities(self, cutoff="2021-01-01"):
        analyses = self.parquet("outcome_analyses")
        outcomes = self.parquet("outcomes")
        return self.connection.sql(f"""
            select row_number() over(order by completion_date,nct_id)-1 as row_index,nct_id,completion_date as timestamp
            from (
                select oa.nct_id,min(oa.date) as completion_date
                from read_parquet('{analyses}') oa
                join read_parquet('{outcomes}') o on oa.outcome_id=o.id and oa.nct_id=o.nct_id
                where (oa.p_value_modifier is null or oa.p_value_modifier!='>')
                  and oa.p_value between 0 and 1 and o.outcome_type='Primary'
                  and oa.date <= timestamp '{cutoff}'
                group by oa.nct_id
            ) x order by completion_date,nct_id
        """).df()

    def _register_entities(self, entities):
        frame = entities[["row_index", "nct_id", "timestamp"]].copy()
        frame["row_index"] = frame["row_index"].astype(np.int64)
        frame["nct_id"] = frame["nct_id"].astype(np.int64)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        self.connection.register("requested_entities", frame)

    def records(self, entities):
        self._register_entities(entities)
        p = self.parquet
        base = self.connection.sql(f"""
            select e.row_index,e.nct_id,e.timestamp,s.start_date,s.phase,s.study_type,s.enrollment,s.enrollment_type,
                   s.number_of_arms,s.number_of_groups,s.brief_title,s.official_title,s.acronym,
                   s.brief_summaries,s.detailed_descriptions,s.source,s.source_class,s.has_dmc,
                   s.is_fda_regulated_drug,s.is_fda_regulated_device,s.plan_to_share_ipd,
                   d.allocation,d.intervention_model,d.observational_model,d.primary_purpose,d.time_perspective,
                   d.masking,d.masking_description,d.intervention_model_description,d.subject_masked,
                   d.caregiver_masked,d.investigator_masked,d.outcomes_assessor_masked,
                   g.sampling_method,g.gender,g.minimum_age,g.maximum_age,g.healthy_volunteers,
                   g.population,g.criteria,g.gender_description,g.gender_based,g.adult,g.child,g.older_adult
            from requested_entities e
            join read_parquet('{p('studies')}') s on e.nct_id=s.nct_id and s.start_date<=e.timestamp
            left join read_parquet('{p('designs')}') d on e.nct_id=d.nct_id and d.date<=e.timestamp
            left join read_parquet('{p('eligibilities')}') g on e.nct_id=g.nct_id and g.date<=e.timestamp
            order by e.row_index
        """).df()
        relations = self.connection.sql(f"""
            with concepts as (
                select e.row_index,'condition' kind,c.mesh_term relation_value
                from requested_entities e join read_parquet('{p('conditions_studies')}') x on e.nct_id=x.nct_id and x.date<=e.timestamp
                join read_parquet('{p('conditions')}') c using(condition_id)
                union all
                select e.row_index,'intervention',i.mesh_term
                from requested_entities e join read_parquet('{p('interventions_studies')}') x on e.nct_id=x.nct_id and x.date<=e.timestamp
                join read_parquet('{p('interventions')}') i using(intervention_id)
                union all
                select e.row_index,'sponsor',coalesce(s.agency_class,'') || ' ' || coalesce(s.name,'')
                from requested_entities e join read_parquet('{p('sponsors_studies')}') x on e.nct_id=x.nct_id and x.date<=e.timestamp
                join read_parquet('{p('sponsors')}') s using(sponsor_id)
                union all
                select e.row_index,'site',coalesce(f.country,'') || ' ' || coalesce(f.state,'') || ' ' || coalesce(f.city,'')
                from requested_entities e join read_parquet('{p('facilities_studies')}') x on e.nct_id=x.nct_id and x.date<=e.timestamp
                join read_parquet('{p('facilities')}') f using(facility_id)
            )
            select row_index,kind,string_agg(distinct relation_value,'; ' order by relation_value) relation_value from concepts group by row_index,kind
        """).df()
        history = self.connection.sql(f"""
            select e.row_index,
                   string_agg(distinct coalesce(o.title,'') || ' ' || coalesce(o.description,'') || ' ' || coalesce(o.time_frame,''),' ; ') outcome_text,
                   string_agg(distinct coalesce(oa.method,'') || ' ' || coalesce(oa.p_value_description,'') || ' ' || coalesce(oa.method_description,''),' ; ') analysis_text
            from requested_entities e
            join read_parquet('{p('outcome_analyses')}') oa on e.nct_id=oa.nct_id and oa.date<=e.timestamp
            join read_parquet('{p('outcomes')}') o on oa.outcome_id=o.id and oa.nct_id=o.nct_id and o.outcome_type='Primary'
            where (oa.p_value_modifier is null or oa.p_value_modifier!='>') and oa.p_value between 0 and 1
            group by e.row_index
        """).df()
        relation_map = relations.pivot(index="row_index", columns="kind", values="relation_value") if len(relations) else pd.DataFrame()
        base = base.join(relation_map, on="row_index")
        base = base.merge(history, how="left", on="row_index", sort=False)
        base = base.sort_values("row_index")
        def combine(row, names):
            values = []
            for name in names:
                value = row.get(name)
                if pd.notna(value) and str(value).strip() not in ("", "nan", "None"):
                    values.append(f"{name.replace('_',' ')}: {value}")
            return " ".join(values)
        fields = np.empty((len(base), 5), dtype=object)
        groups = [
            ["brief_title", "official_title", "acronym", "source"],
            ["brief_summaries", "detailed_descriptions"],
            ["criteria", "population", "gender", "minimum_age", "maximum_age", "healthy_volunteers", "sampling_method", "adult", "child", "older_adult"],
            ["phase", "study_type", "enrollment", "number_of_arms", "allocation", "intervention_model", "observational_model", "primary_purpose", "time_perspective", "masking", "masking_description", "intervention_model_description", "outcome_text", "analysis_text"],
            ["condition", "intervention", "sponsor", "site"],
        ]
        rows = base.to_dict("records")
        for j, names in enumerate(groups):
            fields[:, j] = [combine(row, names) for row in rows]
        return fields

    def current_memberships(self, entities):
        self._register_entities(entities)
        p = self.parquet
        return self.connection.sql(f"""
            select e.row_index,'sponsor' kind,cast(x.sponsor_id as varchar) member_key
            from requested_entities e join read_parquet('{p('sponsors_studies')}') x on e.nct_id=x.nct_id and x.date<=e.timestamp
            union all
            select e.row_index,'condition',cast(x.condition_id as varchar) member_key
            from requested_entities e join read_parquet('{p('conditions_studies')}') x on e.nct_id=x.nct_id and x.date<=e.timestamp
            union all
            select e.row_index,'intervention',cast(x.intervention_id as varchar) member_key
            from requested_entities e join read_parquet('{p('interventions_studies')}') x on e.nct_id=x.nct_id and x.date<=e.timestamp
            union all
            select e.row_index,'country',coalesce(f.country,'unknown') member_key
            from requested_entities e join read_parquet('{p('facilities_studies')}') x on e.nct_id=x.nct_id and x.date<=e.timestamp
            join read_parquet('{p('facilities')}') f using(facility_id)
        """).df()

    def evidence(self, timestamp):
        stamp = pd.Timestamp(timestamp)
        key = stamp.isoformat()
        if key in self._evidence:
            return self._evidence[key]
        p = self.parquet
        frame = self.connection.sql(f"""
            with valid as (
                select oa.nct_id,oa.date,oa.p_value
                from read_parquet('{p('outcome_analyses')}') oa
                join read_parquet('{p('outcomes')}') o on oa.outcome_id=o.id and oa.nct_id=o.nct_id
                where (oa.p_value_modifier is null or oa.p_value_modifier!='>') and oa.p_value between 0 and 1
                  and o.outcome_type='Primary' and oa.date<=timestamp '{stamp}'
            ), analyses as (
                select nct_id,min(date) completion_date,count(*) analysis_count,min(p_value) min_p from valid group by nct_id
            ), dropout as (
                select nct_id,sum(coalesce(count,0)) dropout_count from read_parquet('{p('drop_withdrawals')}')
                where date<=timestamp '{stamp}' group by nct_id
            ), serious as (
                select nct_id,sum(coalesce(subjects_affected,0)) serious_affected,
                       max(coalesce(subjects_at_risk,0)) serious_at_risk
                from read_parquet('{p('reported_event_totals')}')
                where date<=timestamp '{stamp}' and event_type='serious' group by nct_id
            )
            select a.nct_id,a.completion_date,a.analysis_count,a.min_p,cast(a.min_p<=0.05 as double) success,
                   least(coalesce(d.dropout_count,0)/greatest(coalesce(s.enrollment,1),1),10) dropout_burden,
                   least(coalesce(v.serious_affected,0)/greatest(coalesce(v.serious_at_risk,s.enrollment,1),1),10) serious_burden
            from analyses a join read_parquet('{p('studies')}') s using(nct_id)
            left join dropout d using(nct_id) left join serious v using(nct_id)
            order by a.nct_id
        """).df()
        self._evidence[key] = frame
        return frame

    def historical_prior_table(self, timestamp, evidence):
        self.connection.register("available_evidence", evidence[["nct_id", "success", "analysis_count", "min_p"]])
        p = self.parquet
        stamp = pd.Timestamp(timestamp)
        return self.connection.sql(f"""
            with memberships as (
                select a.nct_id,'sponsor' kind,cast(x.sponsor_id as varchar) member_key
                from available_evidence a join read_parquet('{p('sponsors_studies')}') x on a.nct_id=x.nct_id and x.date<=timestamp '{stamp}'
                union all
                select a.nct_id,'condition',cast(x.condition_id as varchar) member_key
                from available_evidence a join read_parquet('{p('conditions_studies')}') x on a.nct_id=x.nct_id and x.date<=timestamp '{stamp}'
                union all
                select a.nct_id,'intervention',cast(x.intervention_id as varchar) member_key
                from available_evidence a join read_parquet('{p('interventions_studies')}') x on a.nct_id=x.nct_id and x.date<=timestamp '{stamp}'
                union all
                select a.nct_id,'country',coalesce(f.country,'unknown') member_key
                from available_evidence a join read_parquet('{p('facilities_studies')}') x on a.nct_id=x.nct_id and x.date<=timestamp '{stamp}'
                join read_parquet('{p('facilities')}') f using(facility_id)
            )
            select m.kind,m.member_key,count(distinct m.nct_id) history_count,avg(a.success) history_success,
                   avg(a.analysis_count) history_analyses,avg(a.min_p) history_min_p
            from memberships m join available_evidence a using(nct_id)
            group by m.kind,m.member_key
        """).df()

    def structured(self, entities):
        self._register_entities(entities)
        p = self.parquet
        frame = self.connection.sql(f"""
            with sponsor as (
                select e.row_index,count(*) sponsor_count,
                       max(case when lower(x.lead_or_collaborator)='lead' then s.agency_class end) lead_agency_class,
                       max(case when lower(x.lead_or_collaborator)='lead' then s.name end) lead_sponsor
                from requested_entities e join read_parquet('{p('sponsors_studies')}') x on e.nct_id=x.nct_id and x.date<=e.timestamp
                join read_parquet('{p('sponsors')}') s using(sponsor_id) group by e.row_index
            ), condition as (
                select e.row_index,count(distinct x.condition_id) condition_count
                from requested_entities e join read_parquet('{p('conditions_studies')}') x on e.nct_id=x.nct_id and x.date<=e.timestamp group by e.row_index
            ), intervention as (
                select e.row_index,count(distinct x.intervention_id) intervention_count
                from requested_entities e join read_parquet('{p('interventions_studies')}') x on e.nct_id=x.nct_id and x.date<=e.timestamp group by e.row_index
            ), site as (
                select e.row_index,count(*) site_count,count(distinct f.country) country_count,
                       count(distinct f.state) state_count,count(distinct f.city) city_count
                from requested_entities e join read_parquet('{p('facilities_studies')}') x on e.nct_id=x.nct_id and x.date<=e.timestamp
                join read_parquet('{p('facilities')}') f using(facility_id) group by e.row_index
            )
            select e.row_index,e.nct_id,e.timestamp,
                   date_diff('day',s.start_date,e.timestamp) trial_age_days,year(s.start_date) start_year,month(s.start_date) start_month,
                   year(e.timestamp) seed_year,coalesce(s.enrollment,0) enrollment,coalesce(s.number_of_arms,0) number_of_arms,
                   coalesce(s.number_of_groups,0) number_of_groups,length(coalesce(s.brief_title,'')) title_length,
                   length(coalesce(s.brief_summaries,'')) summary_length,length(coalesce(s.detailed_descriptions,'')) detail_length,
                   length(coalesce(g.criteria,'')) criteria_length,coalesce(c.condition_count,0) condition_count,
                   coalesce(i.intervention_count,0) intervention_count,coalesce(r.sponsor_count,0) sponsor_count,
                   coalesce(v.site_count,0) site_count,coalesce(v.country_count,0) country_count,
                   coalesce(v.state_count,0) state_count,coalesce(v.city_count,0) city_count,
                   s.phase,s.study_type,s.enrollment_type,s.source_class,s.has_dmc,s.is_fda_regulated_drug,
                   s.is_fda_regulated_device,s.plan_to_share_ipd,d.allocation,d.intervention_model,
                   d.observational_model,d.primary_purpose,d.time_perspective,d.masking,d.subject_masked,
                   d.caregiver_masked,d.investigator_masked,d.outcomes_assessor_masked,g.sampling_method,
                   g.gender,g.minimum_age,g.maximum_age,g.healthy_volunteers,g.gender_based,g.adult,g.child,g.older_adult,
                   r.lead_agency_class,r.lead_sponsor
            from requested_entities e join read_parquet('{p('studies')}') s on e.nct_id=s.nct_id and s.start_date<=e.timestamp
            left join read_parquet('{p('designs')}') d on e.nct_id=d.nct_id and d.date<=e.timestamp
            left join read_parquet('{p('eligibilities')}') g on e.nct_id=g.nct_id and g.date<=e.timestamp
            left join sponsor r using(row_index) left join condition c using(row_index)
            left join intervention i using(row_index) left join site v using(row_index)
            order by e.row_index
        """).df()
        frame["minimum_age_years"] = frame["minimum_age"].map(parse_age)
        frame["maximum_age_years"] = frame["maximum_age"].map(parse_age)
        frame["log_enrollment"] = np.log1p(np.maximum(frame["enrollment"].astype(float), 0))
        frame["enrollment_per_arm"] = frame["enrollment"].astype(float) / np.maximum(frame["number_of_arms"].astype(float), 1)
        frame["trial_age_years"] = frame["trial_age_days"].astype(float) / 365.25
        frame["trial_age_log"] = np.log1p(np.maximum(frame["trial_age_days"].astype(float), 0))
        frame["trial_age_square"] = np.minimum(frame["trial_age_years"], 10) ** 2
        frame["trial_age_bin"] = np.minimum(np.floor(np.maximum(frame["trial_age_days"].astype(float), 0) / 365), 8).astype(int).astype(str)
        frame["enrollment_log_square"] = frame["log_enrollment"] ** 2
        frame["enrollment_bin"] = np.minimum(np.floor(frame["log_enrollment"] / 0.75), 12).astype(int).astype(str)
        frame["site_log"] = np.log1p(np.maximum(frame["site_count"].astype(float), 0))
        frame["site_bin"] = np.minimum(np.floor(frame["site_log"] / 0.75), 8).astype(int).astype(str)
        frame["criteria_log"] = np.log1p(np.maximum(frame["criteria_length"].astype(float), 0))
        frame["arms_category"] = np.minimum(frame["number_of_arms"].astype(int), 6).astype(str)
        phase = frame["phase"].fillna("unknown").astype(str)
        agency = frame["lead_agency_class"].fillna("unknown").astype(str)
        allocation = frame["allocation"].fillna("unknown").astype(str)
        purpose = frame["primary_purpose"].fillna("unknown").astype(str)
        masking = frame["masking"].fillna("unknown").astype(str)
        frame["phase_age_interaction"] = phase + "|" + frame["trial_age_bin"]
        frame["phase_agency_interaction"] = phase + "|" + agency
        frame["phase_allocation_interaction"] = phase + "|" + allocation
        frame["phase_purpose_interaction"] = phase + "|" + purpose
        frame["allocation_masking_interaction"] = allocation + "|" + masking
        frame["study_phase_interaction"] = frame["study_type"].fillna("unknown").astype(str) + "|" + phase
        centered_year = frame["seed_year"].astype(float) - 2015.0
        for phase_name in ("Phase 1", "Phase 2", "Phase 3", "Phase 4", "Not Applicable"):
            safe_name = phase_name.lower().replace(" ", "_")
            frame[f"{safe_name}_year_trend"] = centered_year * (phase == phase_name).astype(float)
        memberships = self.current_memberships(entities)
        numeric_rank = ["enrollment", "trial_age_days", "number_of_arms", "site_count", "country_count", "condition_count", "intervention_count"]
        for column in numeric_rank:
            grouped = frame.groupby("timestamp")[column]
            frame[f"{column}_percentile"] = grouped.rank(pct=True).astype(float)
            mean = grouped.transform("mean")
            std = grouped.transform("std").replace(0, 1).fillna(1)
            frame[f"{column}_z"] = (frame[column] - mean) / std
            frame[f"{column}_leader_gap"] = grouped.transform("max") - frame[column]
        for kind in ("sponsor", "condition", "intervention", "country"):
            for suffix in ("count_mean", "count_max", "rate_mean", "rate_max", "analyses_mean", "p_mean", "coverage"):
                frame[f"{kind}_prior_{suffix}"] = 0.0
        for timestamp, positions in frame.groupby("timestamp").groups.items():
            evidence = self.evidence(timestamp)
            global_rate = float(evidence.success.mean()) if len(evidence) else 0.64
            priors = self.historical_prior_table(timestamp, evidence)
            selected = memberships[memberships.row_index.isin(frame.loc[positions, "row_index"])]
            for kind in ("sponsor", "condition", "intervention", "country"):
                keys = selected[selected.kind == kind]
                stats = priors[priors.kind == kind].copy()
                if len(keys) == 0 or len(stats) == 0:
                    continue
                stats["rate"] = (stats.history_success * stats.history_count + 12.0 * global_rate) / (stats.history_count + 12.0)
                merged = keys.merge(stats, how="left", on=["kind", "member_key"])
                aggregated = merged.groupby("row_index").agg(
                    count_mean=("history_count", "mean"), count_max=("history_count", "max"),
                    rate_mean=("rate", "mean"), rate_max=("rate", "max"),
                    analyses_mean=("history_analyses", "mean"), p_mean=("history_min_p", "mean"),
                    coverage=("history_count", lambda x: float(x.notna().mean())),
                )
                target = frame.set_index("row_index").index
                for suffix in aggregated.columns:
                    values = frame["row_index"].map(aggregated[suffix])
                    frame[f"{kind}_prior_{suffix}"] = frame[f"{kind}_prior_{suffix}"].where(values.isna(), values)
        frame = frame.sort_values("row_index").reset_index(drop=True)
        return frame

    def corpus_metadata(self, entities):
        self._register_entities(entities)
        p = self.parquet
        return self.connection.sql(f"""
            select e.row_index,coalesce(s.phase,'unknown') phase,year(s.start_date) start_year,
                   min(c.condition_id) condition_id,min(i.intervention_id) intervention_id
            from requested_entities e join read_parquet('{p('studies')}') s using(nct_id)
            left join read_parquet('{p('conditions_studies')}') c on e.nct_id=c.nct_id and c.date<=e.timestamp
            left join read_parquet('{p('interventions_studies')}') i on e.nct_id=i.nct_id and i.date<=e.timestamp
            group by e.row_index,s.phase,s.start_date order by e.row_index
        """).df()


def parse_age(value):
    if value is None or pd.isna(value):
        return np.nan
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(year|month|week|day|hour|minute)", str(value).lower())
    if not match:
        return np.nan
    number = float(match.group(1))
    unit = match.group(2)
    factors = {"year": 1.0, "month": 1.0 / 12, "week": 1.0 / 52.1429, "day": 1.0 / 365.25, "hour": 1.0 / 8766, "minute": 1.0 / 525960}
    return number * factors[unit]


class FrozenFieldEncoder:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(config.model_name).to(self.device, dtype=dtype).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, fields, path, batch_size=256):
        count = len(fields)
        output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(count, self.config.field_count, 768))
        flat = fields.reshape(-1).tolist()
        started = time.time()
        for offset in range(0, len(flat), batch_size):
            text = flat[offset:offset + batch_size]
            tokens = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.config.max_length, return_tensors="pt")
            tokens = {name: tensor.to(self.device, non_blocking=True) for name, tensor in tokens.items()}
            with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                vectors = self.model(**tokens).last_hidden_state[:, 0].float().cpu().numpy()
            end = min(offset + len(vectors), len(flat))
            trial = np.arange(offset, end) // self.config.field_count
            field = np.arange(offset, end) % self.config.field_count
            output[trial, field] = vectors.astype(np.float16)
        output.flush()
        rate = len(flat) / max(time.time() - started, 1e-6)
        print(f"[trial2vec] encoded_chunks={len(flat)} chunks_per_second={rate:.1f}", flush=True)
        return np.load(path, mmap_mode="r")


class FieldSetPooler(nn.Module):
    def __init__(self, input_dim=768, embedding_dim=256, field_count=5):
        super().__init__()
        self.field_count = field_count
        self.projections = nn.ModuleList([nn.Linear(input_dim, embedding_dim) for _ in range(field_count)])
        self.field_embedding = nn.Parameter(torch.randn(field_count, embedding_dim) * 0.02)
        self.mask_embedding = nn.Parameter(torch.randn(field_count, embedding_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(embedding_dim, 4, embedding_dim * 2, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.query = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)
        self.attention = nn.MultiheadAttention(embedding_dim, 4, 0.1, batch_first=True)
        self.reconstruction = nn.ModuleList([nn.Linear(embedding_dim, input_dim) for _ in range(field_count)])
        self.normalization = nn.LayerNorm(embedding_dim)

    def project(self, raw):
        return torch.stack([self.projections[index](raw[:, index]) for index in range(self.field_count)], dim=1)

    def forward_projected(self, projected, masked=None):
        values = projected + self.field_embedding.unsqueeze(0)
        if masked is not None:
            values = torch.where(masked.unsqueeze(-1), self.mask_embedding.unsqueeze(0).expand_as(values), values)
        tokens = self.transformer(values)
        query = self.query.expand(len(tokens), -1, -1)
        pooled = self.attention(query, tokens, tokens, need_weights=False)[0][:, 0]
        return self.normalization(pooled), tokens

    def forward(self, raw, masked=None):
        return self.forward_projected(self.project(raw), masked)


def linked_partners(metadata, seed):
    rng = np.random.default_rng(seed)
    count = len(metadata)
    partner = np.full(count, -1, dtype=np.int64)
    for column in ("condition_id", "intervention_id"):
        valid = metadata.dropna(subset=[column]).groupby(column).row_index.apply(list)
        for values in valid:
            if len(values) < 2:
                continue
            values = np.asarray(values, dtype=np.int64)
            shifted = np.roll(values, -1)
            missing = partner[values] < 0
            partner[values[missing]] = shifted[missing]
    absent = np.flatnonzero(partner < 0)
    partner[absent] = rng.integers(0, count, len(absent))
    return partner


def train_pooler(bank, metadata, config, checkpoint, initial=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pooler = FieldSetPooler(embedding_dim=config.embedding_dim, field_count=config.field_count).to(device)
    if initial is not None and Path(initial).exists():
        pooler.load_state_dict(torch.load(initial, map_location=device, weights_only=True))
    if Path(checkpoint).exists():
        pooler.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        return pooler
    optimizer = torch.optim.AdamW(pooler.parameters(), lr=config.head_learning_rate, weight_decay=config.weight_decay)
    partner = linked_partners(metadata, config.seed)
    rng = np.random.default_rng(config.seed)
    epochs = 1 if config.debug else (3 if initial is None else 1)
    batch = config.self_supervised_batch
    indices = np.arange(len(bank))
    for epoch in range(epochs):
        rng.shuffle(indices)
        for offset in range(0, len(indices), batch):
            anchor = indices[offset:offset + batch // 2]
            if len(anchor) < 2:
                continue
            chosen = np.concatenate([anchor, partner[anchor]])
            raw = torch.from_numpy(np.asarray(bank[chosen], dtype=np.float32)).to(device)
            mask1 = torch.rand(len(raw), config.field_count, device=device) < 0.2
            mask2 = torch.rand(len(raw), config.field_count, device=device) < 0.2
            mask1[:, 0] = False
            mask2[:, 1] = False
            z1, token1 = pooler(raw, mask1)
            z2, _ = pooler(raw, mask2)
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
            labels = torch.arange(len(raw), device=device)
            contrast = (F.cross_entropy(z1 @ z2.T / config.contrastive_temperature, labels) + F.cross_entropy(z2 @ z1.T / config.contrastive_temperature, labels)) * 0.5
            target = F.normalize(raw, dim=-1)
            reconstruction = torch.zeros((), device=device)
            for field in range(config.field_count):
                active = mask1[:, field]
                if active.any():
                    prediction = F.normalize(pooler.reconstruction[field](token1[active, field]), dim=-1)
                    reconstruction = reconstruction + F.mse_loss(prediction, target[active, field])
            half = len(anchor)
            linked = 1.0 - F.cosine_similarity(z1[:half], z1[half:half * 2]).mean()
            loss = contrast + 0.5 * reconstruction + 0.5 * linked
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(pooler.parameters(), config.gradient_clip)
            optimizer.step()
    torch.save(pooler.state_dict(), checkpoint)
    register_artifact(Path(checkpoint), Path(checkpoint).stem, "Stage-A field-aware contrastive pooler", Path(checkpoint).stem, "Rebuild from the matching frozen field bank")
    return pooler


def pool_bank(bank, pooler, batch_size=2048):
    device = next(pooler.parameters()).device
    pooled = np.empty((len(bank), 256), dtype=np.float32)
    tokens = np.empty((len(bank), 5, 256), dtype=np.float16)
    pooler.eval()
    for offset in range(0, len(bank), batch_size):
        raw = torch.from_numpy(np.asarray(bank[offset:offset + batch_size], dtype=np.float32)).to(device)
        with torch.inference_mode():
            values, fields = pooler(raw)
        pooled[offset:offset + len(values)] = F.normalize(values, dim=-1).cpu().numpy()
        tokens[offset:offset + len(values)] = fields.cpu().numpy().astype(np.float16)
    return pooled, tokens


def bank_fields(data, encoder, entities, name, config):
    _, lane = cache_root()
    key = hashlib.sha256(f"{name}|{len(entities)}|{config.model_name}|{config.max_length}|fields-v2".encode()).hexdigest()[:16]
    path = lane / f"{name}_{key}.npy"
    if path.exists():
        bank = np.load(path, mmap_mode="r")
        if bank.shape == (len(entities), config.field_count, 768):
            return bank
    fields = data.records(entities)
    bank = encoder.encode(fields, path)
    register_artifact(path, name, f"Frozen BiomedBERT five-field vectors for {name}", key, "Run main.py full to rebuild")
    return bank


def retrieval_evidence(queries, query_pooled, candidate_entities, candidate_pooled, data, count):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_ids = candidate_entities.nct_id.to_numpy(np.int64)
    candidate_times = pd.to_datetime(candidate_entities.timestamp).to_numpy(dtype="datetime64[ns]")
    id_to_position = {int(value): index for index, value in enumerate(candidate_ids)}
    tokens = np.zeros((len(queries), count, 7), dtype=np.float32)
    mask = np.zeros((len(queries), count), dtype=bool)
    for timestamp, group in queries.groupby("timestamp", sort=True):
        row_positions = group.index.to_numpy(np.int64)
        available = np.flatnonzero(candidate_times <= np.datetime64(timestamp))
        if len(available) == 0:
            continue
        candidates = torch.from_numpy(candidate_pooled[available]).to(device)
        current = torch.from_numpy(query_pooled[row_positions]).to(device)
        similarities = current @ candidates.T
        for local, nct_id in enumerate(group.nct_id.to_numpy(np.int64)):
            candidate_position = id_to_position.get(int(nct_id))
            if candidate_position is not None:
                found = np.flatnonzero(available == candidate_position)
                if len(found):
                    similarities[local, int(found[0])] = -2.0
        k = min(count, len(available))
        scores, local_indices = torch.topk(similarities, k=k, dim=1)
        selected = available[local_indices.cpu().numpy()]
        evidence = data.evidence(timestamp).set_index("nct_id")
        for local, row_position in enumerate(row_positions):
            ids = candidate_ids[selected[local]]
            values = evidence.reindex(ids)
            age = (pd.Timestamp(timestamp) - pd.to_datetime(values.completion_date)).dt.days.to_numpy(float) / 3650.0
            tokens[row_position, :k, 0] = scores[local].float().cpu().numpy()
            tokens[row_position, :k, 1] = np.clip(age, 0, 10)
            tokens[row_position, :k, 2] = values.success.fillna(0).to_numpy(float)
            tokens[row_position, :k, 3] = np.log1p(values.analysis_count.fillna(0).to_numpy(float))
            tokens[row_position, :k, 4] = np.minimum(-np.log10(np.maximum(values.min_p.fillna(1).to_numpy(float), 1e-8)), 8) / 8
            tokens[row_position, :k, 5] = np.log1p(values.dropout_burden.fillna(0).to_numpy(float))
            tokens[row_position, :k, 6] = np.log1p(values.serious_burden.fillna(0).to_numpy(float))
            mask[row_position, :k] = True
    aggregates = aggregate_evidence(tokens, mask)
    return tokens, mask, aggregates


def aggregate_evidence(tokens, mask):
    count = np.maximum(mask.sum(axis=1), 1)
    valid = mask[..., None]
    total = (tokens * valid).sum(axis=1) / count[:, None]
    maximum = np.where(valid, tokens, -np.inf).max(axis=1)
    minimum = np.where(valid, tokens, np.inf).min(axis=1)
    maximum[~np.isfinite(maximum)] = 0
    minimum[~np.isfinite(minimum)] = 0
    weights = np.exp(np.clip(tokens[:, :, 0], -1, 1) * 5) * mask
    weighted_success = (weights * tokens[:, :, 2]).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-8)
    top_success = tokens[:, 0, 2]
    prefix = []
    for width in (3, 5, 8, 16):
        width = min(width, tokens.shape[1])
        active = mask[:, :width]
        denominator = np.maximum(active.sum(axis=1), 1)
        prefix.extend([
            ((tokens[:, :width, 2] * active).sum(axis=1) / denominator)[:, None],
            ((tokens[:, :width, 3] * active).sum(axis=1) / denominator)[:, None],
            ((tokens[:, :width, 4] * active).sum(axis=1) / denominator)[:, None],
        ])
    weighted = []
    for temperature in (10.0, 20.0):
        local_weights = np.exp(np.clip(tokens[:, :, 0], -1, 1) * temperature) * mask
        weighted.append(((local_weights * tokens[:, :, 2]).sum(axis=1) / np.maximum(local_weights.sum(axis=1), 1e-8))[:, None])
    fifth = min(4, tokens.shape[1] - 1)
    last = tokens.shape[1] - 1
    margins = np.stack([tokens[:, 0, 0] - tokens[:, fifth, 0], tokens[:, 0, 0] - tokens[:, last, 0]], axis=1)
    return np.concatenate([total, maximum, minimum, weighted_success[:, None], top_success[:, None], count[:, None], *prefix, *weighted, margins], axis=1).astype(np.float32)


def fieldwise_retrieval_aggregates(queries, query_bank, candidate_entities, candidate_bank, data, count):
    outputs = []
    for field in range(query_bank.shape[1]):
        query = np.asarray(query_bank[:, field], dtype=np.float32)
        candidate = np.asarray(candidate_bank[:, field], dtype=np.float32)
        query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-8)
        candidate /= np.maximum(np.linalg.norm(candidate, axis=1, keepdims=True), 1e-8)
        _, _, aggregate = retrieval_evidence(queries, query, candidate_entities, candidate, data, count)
        outputs.append(aggregate)
    return np.concatenate(outputs, axis=1).astype(np.float32)


def linked_retrieval_aggregates(queries, candidate_entities, data, count):
    query_memberships = data.current_memberships(queries)
    candidate_memberships = data.current_memberships(candidate_entities)
    kinds = {"sponsor": 2.0, "condition": 1.0, "intervention": 1.5}
    query_memberships = query_memberships[query_memberships.kind.isin(kinds)]
    candidate_memberships = candidate_memberships[candidate_memberships.kind.isin(kinds)]
    candidate_ids = candidate_entities.nct_id.to_numpy(np.int64)
    candidate_times = pd.to_datetime(candidate_entities.timestamp).to_numpy(dtype="datetime64[ns]")
    tokens = np.zeros((len(queries), count, 7), dtype=np.float32)
    mask = np.zeros((len(queries), count), dtype=bool)
    for timestamp, group in queries.groupby("timestamp", sort=True):
        row_positions = group.index.to_numpy(np.int64)
        available = np.flatnonzero(candidate_times <= np.datetime64(timestamp))
        if len(available) == 0:
            continue
        available_set = set(available.tolist())
        local_memberships = candidate_memberships[candidate_memberships.row_index.isin(available)]
        mapping = local_memberships.groupby(["kind", "member_key"]).row_index.apply(lambda values: values.to_numpy(np.int64)).to_dict()
        evidence = data.evidence(timestamp).set_index("nct_id")
        for row_position, nct_id in zip(row_positions, group.nct_id.to_numpy(np.int64)):
            memberships = query_memberships[query_memberships.row_index == row_position]
            scores = {}
            for item in memberships.itertuples(index=False):
                for candidate_position in mapping.get((item.kind, item.member_key), ()):
                    if candidate_position in available_set and candidate_ids[candidate_position] != nct_id:
                        scores[candidate_position] = scores.get(candidate_position, 0.0) + kinds[item.kind]
            if not scores:
                continue
            selected = sorted(scores, key=lambda position: (scores[position], candidate_times[position]), reverse=True)[:count]
            k = len(selected)
            ids = candidate_ids[selected]
            values = evidence.reindex(ids)
            age = (pd.Timestamp(timestamp) - pd.to_datetime(values.completion_date)).dt.days.to_numpy(float) / 3650.0
            scale = max(max(scores[position] for position in selected), 1.0)
            tokens[row_position, :k, 0] = np.asarray([scores[position] / scale for position in selected], dtype=np.float32)
            tokens[row_position, :k, 1] = np.clip(age, 0, 10)
            tokens[row_position, :k, 2] = values.success.fillna(0).to_numpy(float)
            tokens[row_position, :k, 3] = np.log1p(values.analysis_count.fillna(0).to_numpy(float))
            tokens[row_position, :k, 4] = np.minimum(-np.log10(np.maximum(values.min_p.fillna(1).to_numpy(float), 1e-8)), 8) / 8
            tokens[row_position, :k, 5] = np.log1p(values.dropout_burden.fillna(0).to_numpy(float))
            tokens[row_position, :k, 6] = np.log1p(values.serious_burden.fillna(0).to_numpy(float))
            mask[row_position, :k] = True
    return aggregate_evidence(tokens, mask)


def make_preprocessor(frame):
    ignored = {"row_index", "nct_id", "timestamp"}
    numeric = [column for column in frame.columns if column not in ignored and pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame.columns if column not in ignored and column not in numeric]
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, max_categories=64, sparse_output=True))])
    return ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)], sparse_threshold=1.0), numeric, categorical


def ridge_matrices(train_structured, predict_structured, train_pooled, predict_pooled, train_retrieval, predict_retrieval):
    preprocessor, _, _ = make_preprocessor(train_structured)
    train_tabular = preprocessor.fit_transform(train_structured)
    predict_tabular = preprocessor.transform(predict_structured)
    train_dense = np.concatenate([train_pooled, train_retrieval], axis=1)
    predict_dense = np.concatenate([predict_pooled, predict_retrieval], axis=1)
    train_matrix = sp.hstack([train_tabular, sp.csr_matrix(train_dense)], format="csr")
    predict_matrix = sp.hstack([predict_tabular, sp.csr_matrix(predict_dense)], format="csr")
    return train_matrix, predict_matrix


def fit_ridge(train_structured, predict_structured, train_pooled, predict_pooled, train_retrieval, predict_retrieval, labels, regularization):
    if regularization[0] == "blend":
        weight = regularization[1]
        logistic = fit_ridge(train_structured, predict_structured, train_pooled, predict_pooled, train_retrieval, predict_retrieval, labels, ("logistic", 0.02))
        ridge = fit_ridge(train_structured, predict_structured, train_pooled, predict_pooled, train_retrieval, predict_retrieval, labels, ("ridge", 1000.0))
        return (1.0 - weight) * logistic + weight * ridge
    train_matrix, predict_matrix = ridge_matrices(train_structured, predict_structured, train_pooled, predict_pooled, train_retrieval, predict_retrieval)
    family, strength = regularization
    if family == "logistic":
        classifier = LogisticRegression(C=strength, penalty="l2", solver="liblinear", max_iter=1000, random_state=1337)
    else:
        classifier = RidgeClassifier(alpha=strength, solver="lsqr")
    classifier.fit(train_matrix, labels)
    if family == "logistic":
        return classifier.predict_proba(predict_matrix)[:, 1]
    return sigmoid(classifier.decision_function(predict_matrix))


def select_ridge_regularization(structured, pooled, retrieval, labels):
    years = structured.seed_year.to_numpy(int)
    oof = {}
    fold_scores = {}
    candidates = [("logistic", 0.02), ("logistic", 0.1), ("logistic", 0.5), ("ridge", 1.0), ("ridge", 10.0), ("ridge", 100.0), ("ridge", 1000.0)]
    for regularization in candidates:
        prediction = np.full(len(labels), np.nan)
        scores = []
        for year in (2017, 2018, 2019):
            fit = years < year
            holdout = years == year
            if fit.sum() < 100 or holdout.sum() < 20:
                continue
            prediction[holdout] = fit_ridge(structured.loc[fit], structured.loc[holdout], pooled[fit], pooled[holdout], retrieval[fit], retrieval[holdout], labels[fit], regularization)
            scores.append(safe_auc(labels[holdout], prediction[holdout]))
        valid = np.isfinite(prediction)
        oof[regularization] = prediction
        fold_scores[regularization] = (float(np.mean(scores)), float(np.std(scores)), safe_auc(labels[valid], prediction[valid]))
    for weight in (0.25, 0.5, 0.75):
        regularization = ("blend", weight)
        prediction = (1.0 - weight) * oof[("logistic", 0.02)] + weight * oof[("ridge", 1000.0)]
        valid = np.isfinite(prediction)
        scores = [safe_auc(labels[years == year], prediction[years == year]) for year in (2017, 2018, 2019)]
        oof[regularization] = prediction
        fold_scores[regularization] = (float(np.mean(scores)), float(np.std(scores)), safe_auc(labels[valid], prediction[valid]))
    criteria = {value: fold_scores[value][0] - 0.25 * fold_scores[value][1] for value in fold_scores}
    best_criterion = max(criteria.values())
    best = next(value for value in fold_scores if criteria[value] >= best_criterion - 0.001)
    print(f"[trial2vec] ridge_forward_folds={json.dumps({str(k):v for k,v in fold_scores.items()})} selected_head={best}", flush=True)
    return best, oof[best]


def normalized_probe_values(field_bank, field):
    if field < field_bank.shape[1]:
        values = np.asarray(field_bank[:, field], dtype=np.float32)
    else:
        values = np.asarray(field_bank, dtype=np.float32).mean(axis=1)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    return values


def fit_field_probe(train_bank, predict_bank, labels, specification):
    field, alpha = specification
    train_values = normalized_probe_values(train_bank, field)
    predict_values = normalized_probe_values(predict_bank, field)
    classifier = RidgeClassifier(alpha=alpha, solver="lsqr")
    classifier.fit(train_values, labels)
    return sigmoid(classifier.decision_function(predict_values))


def select_field_probe(field_bank, labels, years):
    predictions = {}
    scores = {}
    for field in range(field_bank.shape[1] + 1):
        values = normalized_probe_values(field_bank, field)
        for alpha in (1.0, 10.0, 100.0):
            specification = (field, alpha)
            oof = np.full(len(labels), np.nan)
            fold = []
            for year in (2017, 2018, 2019):
                fit = years < year
                holdout = years == year
                classifier = RidgeClassifier(alpha=alpha, solver="lsqr")
                classifier.fit(values[fit], labels[fit])
                oof[holdout] = sigmoid(classifier.decision_function(values[holdout]))
                fold.append(safe_auc(labels[holdout], oof[holdout]))
            valid = np.isfinite(oof)
            predictions[specification] = oof
            scores[specification] = (float(np.mean(fold)), float(np.std(fold)), safe_auc(labels[valid], oof[valid]))
    best = max(scores, key=lambda value: scores[value][0] - 0.25 * scores[value][1])
    print(f"[trial2vec] field_probe_selected={best} scores={scores[best]}", flush=True)
    return best, predictions[best], scores[best]


def select_probe_blend(labels, years, baseline, probe):
    valid = np.isfinite(baseline) & np.isfinite(probe)
    choices = []
    for weight in (0.0, 0.1, 0.2, 0.3, 0.4):
        prediction = (1.0 - weight) * baseline + weight * probe
        fold = [safe_auc(labels[(years == year) & valid], prediction[(years == year) & valid]) for year in (2017, 2018, 2019)]
        criterion = float(np.mean(fold) - 0.25 * np.std(fold))
        choices.append((criterion, weight, float(np.mean(fold)), float(np.std(fold)), safe_auc(labels[valid], prediction[valid])))
    best = max(choices)
    baseline_criterion = choices[0][0]
    weight = best[1] if best[0] > baseline_criterion + 0.001 else 0.0
    print(f"[trial2vec] field_probe_blend={json.dumps(choices)} selected_weight={weight}", flush=True)
    return float(weight), choices


def oof_slice_metrics(structured, labels, predictions):
    valid = np.isfinite(predictions)
    frame = structured.iloc[:len(labels)].copy()
    report = {}
    strata = {
        "phase": frame.phase.fillna("missing").astype(str),
        "trial_age": pd.cut(frame.trial_age_years, [-np.inf, 1, 3, np.inf], labels=["under_1y", "1_to_3y", "over_3y"]).astype(str),
        "sponsor_history": np.where(frame.sponsor_prior_coverage.to_numpy(float) > 0, "supported", "sparse"),
        "site_density": pd.cut(frame.site_count, [-np.inf, 1, 20, np.inf], labels=["0_to_1", "2_to_20", "over_20"]).astype(str),
    }
    for axis, groups in strata.items():
        report[axis] = {}
        groups = np.asarray(groups)
        for group in np.unique(groups[valid]):
            selected = valid & (groups == group)
            report[axis][str(group)] = {
                "count": int(selected.sum()),
                "positive_rate": float(labels[selected].mean()),
                "roc_auc": safe_auc(labels[selected], predictions[selected]),
            }
    return report


def bootstrap_auc_uncertainty(labels, predictions, draws=100, seed=1337):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(draws):
        sample = rng.integers(0, len(labels), len(labels))
        if np.unique(labels[sample]).size == 2:
            scores.append(safe_auc(labels[sample], predictions[sample]))
    return float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, values):
        delta = F.linear(F.linear(self.dropout(values), self.lora_a), self.lora_b) * self.scale
        return self.base(values) + delta


class LoRABiomedEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.backbone = AutoModel.from_pretrained(config.model_name, dtype=dtype)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for layer in self.backbone.encoder.layer[-2:]:
            layer.attention.self.query = LoRALinear(layer.attention.self.query, config.lora_rank, config.lora_alpha, config.lora_dropout)
            layer.attention.self.value = LoRALinear(layer.attention.self.value, config.lora_rank, config.lora_alpha, config.lora_dropout)

    def forward(self, input_ids, attention_mask):
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0].float()

    def lora_state(self):
        return {name: value.detach().cpu() for name, value in self.state_dict().items() if "lora_" in name}

    def load_lora_state(self, state):
        current = self.state_dict()
        current.update(state)
        self.load_state_dict(current)


def tokenize_fields(fields, tokenizer, config, batch_size=512):
    count = len(fields)
    ids = np.empty((count, config.field_count, config.max_length), dtype=np.uint16)
    masks = np.empty((count, config.field_count, config.max_length), dtype=np.uint8)
    flat = fields.reshape(-1).tolist()
    for offset in range(0, len(flat), batch_size):
        batch = tokenizer(flat[offset:offset + batch_size], padding="max_length", truncation=True, max_length=config.max_length, return_tensors="np")
        end = min(offset + len(batch["input_ids"]), len(flat))
        trial = np.arange(offset, end) // config.field_count
        field = np.arange(offset, end) % config.field_count
        ids[trial, field] = batch["input_ids"].astype(np.uint16)
        masks[trial, field] = batch["attention_mask"].astype(np.uint8)
    return ids, masks


def field_projection(pooler, vectors, field_indices):
    output = torch.empty((len(vectors), pooler.field_embedding.shape[1]), device=vectors.device, dtype=vectors.dtype)
    for field in range(pooler.field_count):
        active = field_indices == field
        if active.any():
            output[active] = pooler.projections[field](vectors[active])
    return output


def save_stage_b_checkpoint(path, encoder, pooler, cutoff):
    torch.save({"lora": encoder.lora_state(), "pooler": pooler.state_dict(), "cutoff": str(cutoff)}, path)
    register_artifact(Path(path), Path(path).stem, f"Chronological BiomedBERT LoRA and pooler checkpoint through {cutoff}", Path(path).stem, "Run full Stage-B chronological adaptation")


def load_stage_b_checkpoint(path, config, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    encoder = LoRABiomedEncoder(config).to(device)
    pooler = FieldSetPooler(embedding_dim=config.embedding_dim, field_count=config.field_count).to(device)
    encoder.load_lora_state(state["lora"])
    pooler.load_state_dict(state["pooler"])
    return encoder, pooler


def hard_negative_order(metadata, seed):
    rng = np.random.default_rng(seed)
    groups = []
    for _, group in metadata.groupby(["phase", "start_year"], dropna=False):
        values = group.row_index.to_numpy(np.int64)
        rng.shuffle(values)
        groups.append(values)
    rng.shuffle(groups)
    return np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)


def lora_contrastive_epoch(encoder, pooler, fields, metadata, tokenizer, config, optimizer, seed):
    device = next(pooler.parameters()).device
    order = hard_negative_order(metadata, seed)
    rng = np.random.default_rng(seed)
    encoder.train()
    pooler.train()
    for offset in range(0, len(order), config.self_supervised_batch):
        indices = order[offset:offset + config.self_supervised_batch]
        if len(indices) < 8:
            continue
        first = rng.integers(0, config.field_count, len(indices))
        second = rng.integers(0, config.field_count - 1, len(indices))
        second = second + (second >= first)
        text = [fields[index, field] for index, field in zip(indices, first)] + [fields[index, field] for index, field in zip(indices, second)]
        tokens = tokenizer(text, padding="max_length", truncation=True, max_length=config.max_length, return_tensors="pt")
        input_ids = tokens["input_ids"].to(device, non_blocking=True)
        attention_mask = tokens["attention_mask"].to(device, non_blocking=True)
        vectors = encoder(input_ids, attention_mask)
        fields_index = torch.from_numpy(np.concatenate([first, second])).to(device)
        projected = field_projection(pooler, vectors, fields_index)
        first_z = F.normalize(projected[:len(indices)], dim=-1)
        second_z = F.normalize(projected[len(indices):], dim=-1)
        labels = torch.arange(len(indices), device=device)
        logits = first_z @ second_z.T / config.contrastive_temperature
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([parameter for parameter in list(encoder.parameters()) + list(pooler.parameters()) if parameter.requires_grad], config.gradient_clip)
        optimizer.step()


def stage_entities_for_period(data, lower, upper, maximum, seed):
    studies = data.parquet("studies")
    frame = data.connection.sql(f"select nct_id,start_date from read_parquet('{studies}') where start_date>=timestamp '{lower}' and start_date<timestamp '{upper}' order by start_date,nct_id").df()
    if len(frame) > maximum:
        frame = frame.sample(maximum, random_state=seed).sort_values(["start_date", "nct_id"])
    frame = frame[["nct_id"]].reset_index(drop=True)
    frame.insert(0, "row_index", np.arange(len(frame), dtype=np.int64))
    frame["timestamp"] = pd.Timestamp(upper)
    return frame


def train_chronological_stage_b(data, frozen_encoder, config):
    _, lane = cache_root()
    cutoffs = [2017, 2018, 2019, 2020, 2021]
    paths = {year: lane / f"stage_b_cutoff_{year}_v2.pt" for year in cutoffs}
    if all(path.exists() for path in paths.values()):
        return paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_entities = stage_entities_for_period(data, "2000-01-01", "2017-01-01", 24000, config.seed)
    initial_fields = data.records(initial_entities)
    initial_bank_path = lane / "stage_b_initial_frozen_v2.npy"
    if initial_bank_path.exists():
        initial_bank = np.load(initial_bank_path, mmap_mode="r")
    else:
        initial_bank = frozen_encoder.encode(initial_fields, initial_bank_path)
        register_artifact(initial_bank_path, "stage_b_initial_frozen", "Pre-2017 frozen field vectors for chronological Stage-B initialization", "stage-b-initial-v2", "Encode the sampled pre-2017 records")
    initial_meta = data.corpus_metadata(initial_entities)
    initial_pool_path = lane / "stage_b_initial_pooler_v2.pt"
    pooler = train_pooler(initial_bank, initial_meta, config, initial_pool_path)
    encoder = LoRABiomedEncoder(config).to(device)
    pooler = pooler.to(device)
    parameter_groups = [
        {"params": [p for n, p in encoder.named_parameters() if p.requires_grad], "lr": config.lora_learning_rate},
        {"params": [p for p in pooler.parameters() if p.requires_grad], "lr": config.head_learning_rate},
    ]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config.weight_decay)
    lora_contrastive_epoch(encoder, pooler, initial_fields, initial_meta, frozen_encoder.tokenizer, config, optimizer, config.seed)
    save_stage_b_checkpoint(paths[2017], encoder, pooler, "2017-01-01")
    previous = 2017
    for year in (2018, 2019, 2020, 2021):
        entities = stage_entities_for_period(data, f"{previous}-01-01", f"{year}-01-01", 10000, config.seed + year)
        fields = data.records(entities)
        metadata = data.corpus_metadata(entities)
        lora_contrastive_epoch(encoder, pooler, fields, metadata, frozen_encoder.tokenizer, config, optimizer, config.seed + year)
        save_stage_b_checkpoint(paths[year], encoder, pooler, f"{year}-01-01")
        previous = year
    del encoder, pooler, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return paths


def encode_adapted(tokenized, encoder, pooler, batch_size=64):
    ids, masks = tokenized
    device = next(pooler.parameters()).device
    pooled = np.empty((len(ids), 256), dtype=np.float32)
    fields = np.empty((len(ids), 5, 256), dtype=np.float16)
    encoder.eval()
    pooler.eval()
    for offset in range(0, len(ids), batch_size):
        batch_ids = torch.from_numpy(ids[offset:offset + batch_size].astype(np.int64)).to(device)
        batch_masks = torch.from_numpy(masks[offset:offset + batch_size].astype(np.int64)).to(device)
        flat_ids = batch_ids.flatten(0, 1)
        flat_masks = batch_masks.flatten(0, 1)
        with torch.inference_mode():
            raw = encoder(flat_ids, flat_masks).reshape(len(batch_ids), 5, 768)
            values, field_values = pooler(raw)
        pooled[offset:offset + len(values)] = F.normalize(values, dim=-1).cpu().numpy()
        fields[offset:offset + len(values)] = field_values.cpu().numpy().astype(np.float16)
    return pooled, fields


class GatedEvidenceFusion(nn.Module):
    def __init__(self, encoder, pooler, structured_dim, config):
        super().__init__()
        self.encoder = encoder
        self.pooler = pooler
        self.config = config
        self.evidence_projection = nn.Sequential(nn.Linear(7, 128), nn.GELU(), nn.Linear(128, 256), nn.LayerNorm(256))
        self.evidence_attention = nn.MultiheadAttention(256, 4, 0.1, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(512, 256), nn.Sigmoid())
        self.structured = nn.Sequential(nn.Linear(structured_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.LayerNorm(128))
        self.classifier = nn.Sequential(nn.Linear(384, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, input_ids, attention_mask, evidence, evidence_mask, structured):
        batch = len(input_ids)
        raw = self.encoder(input_ids.flatten(0, 1), attention_mask.flatten(0, 1)).reshape(batch, 5, 768)
        if self.training and self.config.modality_dropout > 0:
            drop = torch.rand(batch, device=raw.device) < self.config.modality_dropout
            raw[drop, 4] = 0
        current, _ = self.pooler(raw)
        evidence_values = self.evidence_projection(evidence)
        safe_mask = evidence_mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True
        evidence_values[empty, 0] = 0
        attended = self.evidence_attention(current.unsqueeze(1), evidence_values, evidence_values, key_padding_mask=~safe_mask, need_weights=False)[0][:, 0]
        gate = self.gate(torch.cat([current, attended], dim=-1))
        fused = gate * current + (1.0 - gate) * attended
        residual = self.structured(structured)
        return self.classifier(torch.cat([fused, residual], dim=-1))[:, 0]


def dense_structured(train_frame, predict_frame):
    preprocessor, _, _ = make_preprocessor(train_frame)
    train_values = preprocessor.fit_transform(train_frame)
    predict_values = preprocessor.transform(predict_frame)
    if sp.issparse(train_values):
        train_values = train_values.toarray()
        predict_values = predict_values.toarray()
    return np.asarray(train_values, dtype=np.float32), np.asarray(predict_values, dtype=np.float32)


def supervised_batches(indices, labels, times, batch_size, seed):
    rng = np.random.default_rng(seed)
    batches = []
    local_labels = labels[indices]
    local_times = times[indices]
    for timestamp in np.unique(local_times):
        group = indices[local_times == timestamp]
        positive = group[labels[group] == 1]
        negative = group[labels[group] == 0]
        if len(positive) == 0 or len(negative) == 0:
            continue
        target = max(len(positive), len(negative))
        positive = rng.choice(positive, target, replace=len(positive) < target)
        negative = rng.choice(negative, target, replace=len(negative) < target)
        rng.shuffle(positive)
        rng.shuffle(negative)
        half = batch_size // 2
        for offset in range(0, target, half):
            part_positive = positive[offset:offset + half]
            part_negative = negative[offset:offset + half]
            if len(part_positive) < half:
                part_positive = rng.choice(positive, half, replace=True)
                part_negative = rng.choice(negative, half, replace=True)
            batch = np.concatenate([part_positive, part_negative])
            rng.shuffle(batch)
            batches.append(batch)
    rng.shuffle(batches)
    return batches


def fusion_forward(model, tokenized, evidence_tokens, evidence_mask, structured, indices, device):
    ids, masks = tokenized
    index = np.asarray(indices, dtype=np.int64)
    input_ids = torch.from_numpy(ids[index].astype(np.int64)).to(device, non_blocking=True)
    attention_mask = torch.from_numpy(masks[index].astype(np.int64)).to(device, non_blocking=True)
    evidence = torch.from_numpy(evidence_tokens[index]).to(device, non_blocking=True)
    retrieval_mask = torch.from_numpy(evidence_mask[index]).to(device, non_blocking=True)
    tabular = torch.from_numpy(structured[index]).to(device, non_blocking=True)
    return model(input_ids, attention_mask, evidence, retrieval_mask, tabular)


def fit_fusion(encoder, pooler, tokenized, evidence_tokens, evidence_mask, structured, labels, times, fit_indices, predict_indices, config):
    device = next(pooler.parameters()).device
    model = GatedEvidenceFusion(encoder, pooler, structured.shape[1], config).to(device)
    lora = [parameter for name, parameter in model.named_parameters() if "lora_" in name and parameter.requires_grad]
    head = [parameter for name, parameter in model.named_parameters() if "lora_" not in name and parameter.requires_grad]
    optimizer = torch.optim.AdamW([{"params": lora, "lr": config.lora_learning_rate}, {"params": head, "lr": config.head_learning_rate}], weight_decay=config.weight_decay)
    labels_tensor = torch.from_numpy(labels.astype(np.float32)).to(device)
    for epoch in range(config.supervised_epochs):
        model.train()
        for batch in supervised_batches(np.asarray(fit_indices), labels, times, config.supervised_batch, config.seed + epoch):
            logits = fusion_forward(model, tokenized, evidence_tokens, evidence_mask, structured, batch, device)
            target = labels_tensor[torch.from_numpy(batch).to(device)]
            bce = F.binary_cross_entropy_with_logits(logits, target)
            positive = logits[target > 0.5]
            negative = logits[target < 0.5]
            pairs = min(len(positive), len(negative))
            ranking = F.softplus(-(positive[:pairs] - negative[:pairs])).mean() if pairs else torch.zeros((), device=device)
            loss = bce + config.ranking_weight * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], config.gradient_clip)
            optimizer.step()
    predictions = np.empty(len(predict_indices), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(predict_indices), 64):
            selection = np.asarray(predict_indices[offset:offset + 64], dtype=np.int64)
            logits = fusion_forward(model, tokenized, evidence_tokens, evidence_mask, structured, selection, device)
            predictions[offset:offset + len(selection)] = torch.sigmoid(logits).cpu().numpy()
    return predictions


def stage_b_fold_prediction(checkpoint, config, data, tokenized_queries, tokenized_candidates, all_entities, completed, structured, labels, fit_indices, predict_indices):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, pooler = load_stage_b_checkpoint(checkpoint, config, device)
    query_pooled, _ = encode_adapted(tokenized_queries, encoder, pooler)
    candidate_pooled, _ = encode_adapted(tokenized_candidates, encoder, pooler)
    evidence_tokens, evidence_mask, _ = retrieval_evidence(all_entities, query_pooled, completed, candidate_pooled, data, config.retrieval_neighbors)
    fit_structured, predict_structured = dense_structured(structured.iloc[fit_indices], structured.iloc[predict_indices])
    dimension = fit_structured.shape[1]
    all_structured = np.zeros((len(structured), dimension), dtype=np.float32)
    all_structured[fit_indices] = fit_structured
    all_structured[predict_indices] = predict_structured
    times = pd.to_datetime(all_entities.timestamp).to_numpy(dtype="datetime64[ns]")
    prediction = fit_fusion(encoder, pooler, tokenized_queries, evidence_tokens, evidence_mask, all_structured, labels, times, fit_indices, predict_indices, config)
    del encoder, pooler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction


def stage_b_oof(paths, config, data, tokenized_queries, tokenized_candidates, all_entities, completed, structured, labels, n_train):
    years = structured.seed_year.to_numpy(int)
    _, lane = cache_root()
    prediction_path = lane / "stage_b_training_oof_v3.npy"
    if prediction_path.exists():
        predictions = np.load(prediction_path, allow_pickle=False)
        if predictions.shape == (n_train,):
            fold_scores = {year: safe_auc(labels[years[:n_train] == year], predictions[years[:n_train] == year]) for year in (2017, 2018, 2019)}
            print(f"[trial2vec] stage_b_forward_folds={json.dumps(fold_scores)} source=cache", flush=True)
            return predictions, fold_scores
    predictions = np.full(n_train, np.nan)
    full_labels = np.zeros(len(all_entities), dtype=np.int64)
    full_labels[:n_train] = labels
    fold_scores = {}
    for year in (2017, 2018, 2019):
        fit_indices = np.flatnonzero((years < year) & (np.arange(len(years)) < n_train))
        predict_indices = np.flatnonzero((years == year) & (np.arange(len(years)) < n_train))
        predictions[predict_indices] = stage_b_fold_prediction(paths[year], config, data, tokenized_queries, tokenized_candidates, all_entities, completed, structured, full_labels, fit_indices, predict_indices)
        fold_scores[year] = safe_auc(labels[predict_indices], predictions[predict_indices])
    print(f"[trial2vec] stage_b_forward_folds={json.dumps(fold_scores)}", flush=True)
    np.save(prediction_path, predictions)
    register_artifact(prediction_path, "stage_b_training_oof", "Training-only chronological LoRA fusion forward-fold predictions", "stage-b-oof-v3", "Run the 2017-2019 Stage-B forward folds")
    return predictions, fold_scores


def paired_auc_uncertainty(labels, first, second, draws=100, seed=1337):
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        sample = rng.integers(0, len(labels), len(labels))
        if np.unique(labels[sample]).size == 2:
            differences.append(safe_auc(labels[sample], second[sample]) - safe_auc(labels[sample], first[sample]))
    return float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0


def select_blend(labels, ridge_predictions, neural_predictions):
    choices = []
    for neural_weight in np.linspace(0, 1, 11):
        prediction = (1 - neural_weight) * ridge_predictions + neural_weight * neural_predictions
        choices.append((safe_auc(labels, prediction), neural_weight))
    best_score = max(score for score, _ in choices)
    eligible = [weight for score, weight in choices if score >= best_score - 0.001]
    weight = min(eligible)
    print(f"[trial2vec] oof_blend={json.dumps(choices)} selected_neural_weight={weight}", flush=True)
    return float(weight)


def write_outputs(val_predictions, test_predictions, metrics):
    output = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "./output_data_generic_exp_3"))
    output.mkdir(parents=True, exist_ok=True)
    val_predictions = np.clip(np.asarray(val_predictions, dtype=np.float64), 0, 1)
    test_predictions = np.clip(np.asarray(test_predictions, dtype=np.float64), 0, 1)
    np.save(output / "val_predictions.npy", val_predictions)
    np.save(output / "test_predictions.npy", test_predictions)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[trial2vec] wrote val{val_predictions.shape} test{test_predictions.shape}", flush=True)


def run(debug=False):
    config = Config(debug=debug)
    seed_everything(config.seed)
    clock = Clock()
    data = TrialData(data_root())
    train, val, test = data.splits()
    train_entities = train[["nct_id", "timestamp"]].copy()
    val_entities = val[["nct_id", "timestamp"]].copy()
    test_entities = test[["nct_id", "timestamp"]].copy()
    all_entities = pd.concat([train_entities, val_entities, test_entities], ignore_index=True)
    all_entities.insert(0, "row_index", np.arange(len(all_entities), dtype=np.int64))
    corpus_a = data.corpus_entities("2020-01-01", config.corpus_limit)
    corpus_b = data.corpus_entities("2021-01-01", config.corpus_limit)
    completed = data.completed_entities("2021-01-01")
    if debug:
        completed = completed.tail(min(2000, len(completed))).copy().reset_index(drop=True)
        completed["row_index"] = np.arange(len(completed), dtype=np.int64)
    clock.mark("data_loaded")
    encoder = FrozenFieldEncoder(config)
    bank_a = bank_fields(data, encoder, corpus_a, "corpus_a_debug" if debug else "corpus_a", config)
    bank_b = bank_fields(data, encoder, corpus_b, "corpus_b_debug" if debug else "corpus_b", config)
    query_bank = bank_fields(data, encoder, all_entities, "queries", config)
    candidate_bank = bank_fields(data, encoder, completed, "completed_debug" if debug else "completed", config)
    clock.mark("frozen_field_banks")
    _, lane = cache_root()
    meta_a = data.corpus_metadata(corpus_a)
    meta_b = data.corpus_metadata(corpus_b)
    pool_a_path = lane / ("pooler_a_debug.pt" if debug else "pooler_a.pt")
    pool_b_path = lane / ("pooler_b_debug.pt" if debug else "pooler_b.pt")
    pooler_a = train_pooler(bank_a, meta_a, config, pool_a_path)
    pooler_b = train_pooler(bank_b, meta_b, config, pool_b_path, initial=pool_a_path)
    clock.mark("stage_a_poolers")
    query_a, _ = pool_bank(query_bank, pooler_a)
    query_b, _ = pool_bank(query_bank, pooler_b)
    candidate_a, _ = pool_bank(candidate_bank, pooler_a)
    candidate_b, _ = pool_bank(candidate_bank, pooler_b)
    _, _, retrieval_a = retrieval_evidence(all_entities, query_a, completed, candidate_a, data, config.neighbor_count)
    _, _, retrieval_b = retrieval_evidence(all_entities, query_b, completed, candidate_b, data, config.neighbor_count)
    field_retrieval = fieldwise_retrieval_aggregates(all_entities, query_bank, completed, candidate_bank, data, config.neighbor_count)
    linked_retrieval = linked_retrieval_aggregates(all_entities, completed, data, config.neighbor_count)
    retrieval_a = np.concatenate([retrieval_a, field_retrieval, linked_retrieval], axis=1)
    retrieval_b = np.concatenate([retrieval_b, field_retrieval, linked_retrieval], axis=1)
    clock.mark("retrieval")
    structured = data.structured(all_entities)
    clock.mark("structured_features")
    n_train = len(train)
    n_val = len(val)
    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, len(all_entities))
    labels_train = train.outcome.to_numpy(np.int64)
    labels_train_val = np.concatenate([labels_train, val.outcome.to_numpy(np.int64)])
    regularization, ridge_oof = select_ridge_regularization(structured.iloc[train_slice].reset_index(drop=True), query_a[train_slice], retrieval_a[train_slice], labels_train)
    val_predictions = fit_ridge(structured.iloc[train_slice], structured.iloc[val_slice], query_a[train_slice], query_a[val_slice], retrieval_a[train_slice], retrieval_a[val_slice], labels_train, regularization)
    test_predictions = fit_ridge(structured.iloc[:n_train+n_val], structured.iloc[test_slice], query_b[:n_train+n_val], query_b[test_slice], retrieval_b[:n_train+n_val], retrieval_b[test_slice], labels_train_val, regularization)
    probe_specification, probe_oof, probe_scores = select_field_probe(query_bank[train_slice], labels_train, structured.seed_year.to_numpy(int)[:n_train])
    probe_weight, probe_blend_diagnostics = select_probe_blend(labels_train, structured.seed_year.to_numpy(int)[:n_train], ridge_oof, probe_oof)
    stage_a_oof = (1.0 - probe_weight) * ridge_oof + probe_weight * probe_oof
    if probe_weight > 0:
        val_probe = fit_field_probe(query_bank[train_slice], query_bank[val_slice], labels_train, probe_specification)
        test_probe = fit_field_probe(query_bank[:n_train+n_val], query_bank[test_slice], labels_train_val, probe_specification)
        val_predictions = (1.0 - probe_weight) * val_predictions + probe_weight * val_probe
        test_predictions = (1.0 - probe_weight) * test_predictions + probe_weight * test_probe
    valid = np.isfinite(stage_a_oof)
    stage_b_retained = False
    stage_b_oof_auc = None
    stage_b_delta = None
    stage_b_uncertainty = None
    blend_weight = 0.0
    prediction_correlation = None
    if not debug and os.environ.get("KAPSO_STAGE_A_ONLY") != "1" and time.time() - clock.start < 10800:
        paths = train_chronological_stage_b(data, encoder, config)
        clock.mark("stage_b_chronological")
        query_fields = data.records(all_entities)
        candidate_fields = data.records(completed)
        tokenized_queries = tokenize_fields(query_fields, encoder.tokenizer, config)
        tokenized_candidates = tokenize_fields(candidate_fields, encoder.tokenizer, config)
        del query_fields, candidate_fields
        clock.mark("stage_b_tokenization")
        neural_oof, neural_fold_scores = stage_b_oof(paths, config, data, tokenized_queries, tokenized_candidates, all_entities, completed, structured, labels_train, n_train)
        common = np.isfinite(stage_a_oof) & np.isfinite(neural_oof)
        stage_b_oof_auc = safe_auc(labels_train[common], neural_oof[common])
        ridge_common_auc = safe_auc(labels_train[common], stage_a_oof[common])
        stage_b_delta = stage_b_oof_auc - ridge_common_auc
        stage_b_uncertainty = paired_auc_uncertainty(labels_train[common], stage_a_oof[common], neural_oof[common])
        prediction_correlation = float(pd.Series(stage_a_oof[common]).corr(pd.Series(neural_oof[common]), method="spearman"))
        stage_b_retained = bool(stage_b_delta > stage_b_uncertainty)
        print(f"[trial2vec] stage_b_gate delta={stage_b_delta:.6f} uncertainty={stage_b_uncertainty:.6f} retained={stage_b_retained}", flush=True)
        if stage_b_retained and time.time() - clock.start < 10800:
            blend_weight = select_blend(labels_train[common], stage_a_oof[common], neural_oof[common])
            full_labels_a = np.zeros(len(all_entities), dtype=np.int64)
            full_labels_a[:n_train] = labels_train
            fit_a = np.arange(n_train, dtype=np.int64)
            predict_a = np.arange(n_train, n_train + n_val, dtype=np.int64)
            neural_val = stage_b_fold_prediction(paths[2020], config, data, tokenized_queries, tokenized_candidates, all_entities, completed, structured, full_labels_a, fit_a, predict_a)
            full_labels_b = np.zeros(len(all_entities), dtype=np.int64)
            full_labels_b[:n_train+n_val] = labels_train_val
            fit_b = np.arange(n_train + n_val, dtype=np.int64)
            predict_b = np.arange(n_train + n_val, len(all_entities), dtype=np.int64)
            neural_test = stage_b_fold_prediction(paths[2021], config, data, tokenized_queries, tokenized_candidates, all_entities, completed, structured, full_labels_b, fit_b, predict_b)
            val_predictions = (1.0 - blend_weight) * val_predictions + blend_weight * neural_val
            test_predictions = (1.0 - blend_weight) * test_predictions + blend_weight * neural_test
            clock.mark("stage_b_supervised_final")
    metrics = {
        "debug": debug,
        "stage_a_oof_auc": safe_auc(labels_train[valid], stage_a_oof[valid]),
        "stage_a_oof_bootstrap_se": bootstrap_auc_uncertainty(labels_train[valid], stage_a_oof[valid]),
        "selected_head": regularization,
        "field_probe_specification": probe_specification,
        "field_probe_oof": probe_scores,
        "field_probe_weight": probe_weight,
        "stage_b_retained": stage_b_retained,
        "stage_b_oof_auc": stage_b_oof_auc,
        "stage_b_delta": stage_b_delta,
        "stage_b_uncertainty": stage_b_uncertainty,
        "oof_prediction_spearman": prediction_correlation,
        "neural_blend_weight": blend_weight,
        "neighbors": config.neighbor_count,
        "oof_slices": oof_slice_metrics(structured, labels_train, stage_a_oof),
        "elapsed_seconds": time.time() - clock.start,
    }
    write_outputs(val_predictions, test_predictions, metrics)
    clock.mark("outputs")
