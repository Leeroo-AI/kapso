# Imports

from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI

from campaign_io import register_artifact
from hosted_judgments_v2 import MODEL as HOSTED_MODEL


# Configuration

RETRIEVAL_VERSION = "literature-retrieval-v2"
PROMPT_VERSION = "primary-result-adjudication-v2"
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
NCT_PATTERN = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)
RESULT_WORDS = re.compile(r"\b(result|results|efficacy|effect|endpoint|outcome|survival|response|remission|improv|difference|randomi[sz]ed)\b", re.IGNORECASE)
PROTOCOL_WORDS = re.compile(r"\b(protocol|design|rationale|methods paper|baseline characteristics)\b", re.IGNORECASE)
REVIEW_WORDS = re.compile(r"\b(review|meta-analysis|systematic review|guideline)\b", re.IGNORECASE)
MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7,
    "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9,
    "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
ADJUDICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_this_trial": {"type": "boolean"},
        "report_type": {"type": "string", "enum": ["primary-results", "secondary-results", "interim-results", "protocol", "review", "other"]},
        "primary_endpoint_met": {"type": "string", "enum": ["yes", "no", "mixed", "not-reported"]},
        "explicit_p_value": {"type": ["string", "null"], "maxLength": 80},
        "final_status": {"type": "string", "enum": ["final", "interim", "unclear"]},
        "endpoint_match": {"type": "integer", "minimum": 0, "maximum": 5},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 5},
        "insufficient_evidence": {"type": "boolean"},
    },
    "required": ["is_this_trial", "report_type", "primary_endpoint_met", "explicit_p_value", "final_status", "endpoint_match", "confidence", "insufficient_evidence"],
    "additionalProperties": False,
}
SYSTEM_PROMPT = """You are a senior clinical-trial results reviewer. Decide whether the complete publication abstract reports the supplied registered trial and whether it reports the registered primary endpoint as met. An accession match is strong identity evidence but reviews and papers mentioning many trials are not trial reports. Judge the registered primary outcome, not any favorable secondary endpoint. Use mixed when co-primary endpoints conflict. Use not-reported when the primary endpoint cannot be adjudicated from the complete abstract. Record an explicit p-value only when the abstract states it for the matching primary endpoint. Confidence and endpoint match use 0 for no evidence and 5 for explicit, unambiguous evidence. Mark insufficient evidence whenever a verdict would require guessing. Use only the supplied pre-origin registry context and publication abstract."""


# Utilities

def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _gzip_write(path: Path, payload: bytes) -> None:
    _atomic_bytes(path, gzip.compress(payload, compresslevel=6))


def _gzip_read(path: Path) -> bytes:
    return gzip.decompress(path.read_bytes())


def _request(url: str, attempts: int = 6) -> bytes:
    last_error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "kapso-relbench-literature/2.0 noreply@example.com"})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(min(20.0, 0.8 * (2 ** attempt)) + random.random())
    raise RuntimeError(f"Literature request failed after {attempts} attempts: {url}: {last_error}")


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _normalize_identifier(value: Any) -> str:
    match = NCT_PATTERN.search(str(value or ""))
    return match.group(0).upper() if match else ""


def _publication_identity(record: dict[str, Any]) -> str:
    if record.get("pmid"):
        return f"pmid:{record['pmid']}"
    if record.get("doi"):
        return f"doi:{str(record['doi']).casefold()}"
    payload = f"{record.get('title', '')}\0{record.get('abstract', '')}".encode("utf-8", errors="replace")
    return f"content:{hashlib.sha256(payload).hexdigest()}"


def _date_decision(full_dates: list[str], years: list[int], origin: pd.Timestamp) -> tuple[bool, str, str, str]:
    parsed = []
    for value in full_dates:
        try:
            parsed.append(pd.Timestamp(value).normalize())
        except Exception:
            continue
    if parsed:
        selected = min(parsed)
        accepted = bool(selected < origin)
        return accepted, selected.strftime("%Y-%m-%d"), "complete", "complete_date_pre_origin" if accepted else "complete_date_not_pre_origin"
    valid_years = [int(value) for value in years if 1500 <= int(value) <= 2200]
    if valid_years:
        selected_year = min(valid_years)
        accepted = bool(selected_year < origin.year)
        return accepted, str(selected_year), "year", "year_strictly_pre_origin" if accepted else "year_not_strictly_pre_origin"
    return False, "", "missing", "missing_admissible_date"


def _xml_date(element: ET.Element | None) -> tuple[str, int | None]:
    if element is None:
        return "", None
    year_text = _text(element.find("Year"))
    if not year_text.isdigit():
        medline = _text(element.find("MedlineDate"))
        match = re.search(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b", medline)
        return "", int(match.group(1)) if match else None
    year = int(year_text)
    month_text = _text(element.find("Month")).casefold()
    day_text = _text(element.find("Day"))
    month = int(month_text) if month_text.isdigit() else MONTHS.get(month_text)
    day = int(day_text) if day_text.isdigit() else None
    if month and day:
        try:
            return date(year, month, day).isoformat(), year
        except ValueError:
            pass
    return "", year


def _token_similarity(left: str, right: str) -> float:
    normalize = lambda value: set(re.findall(r"[a-z0-9]+", str(value).casefold()))
    a = normalize(left)
    b = normalize(right)
    overlap = len(a & b) / max(1, len(a | b))
    sequence = SequenceMatcher(None, str(left).casefold(), str(right).casefold(), autojunk=False).ratio()
    return max(overlap, sequence)


# PubMed

def _parse_pubmed(payload: bytes, origin: pd.Timestamp) -> list[dict[str, Any]]:
    root = ET.fromstring(payload)
    records = []
    for article in root.findall(".//PubmedArticle"):
        citation = article.find("MedlineCitation")
        journal_article = citation.find("Article") if citation is not None else None
        pmid = _text(citation.find("PMID")) if citation is not None else ""
        title = _text(journal_article.find("ArticleTitle")) if journal_article is not None else ""
        abstract_parts = []
        if journal_article is not None:
            for part in journal_article.findall("Abstract/AbstractText"):
                label = part.attrib.get("Label", "")
                value = _text(part)
                abstract_parts.append(f"{label}: {value}" if label else value)
        publication_types = [_text(value) for value in journal_article.findall("PublicationTypeList/PublicationType")] if journal_article is not None else []
        identifiers = {}
        for value in article.findall("PubmedData/ArticleIdList/ArticleId"):
            identifiers[value.attrib.get("IdType", "")] = _text(value)
        accessions = sorted({_normalize_identifier(value.text) for value in article.findall(".//DataBank[DataBankName='ClinicalTrials.gov']/AccessionNumberList/AccessionNumber") if _normalize_identifier(value.text)})
        full_dates = []
        years = []
        if journal_article is not None:
            for element in journal_article.findall("ArticleDate") + journal_article.findall("Journal/JournalIssue/PubDate"):
                full, year = _xml_date(element)
                if full:
                    full_dates.append(full)
                if year:
                    years.append(year)
        accepted, publication_date, date_resolution, date_reason = _date_decision(full_dates, years, origin)
        record = {
            "pmid": pmid,
            "pmcid": identifiers.get("pmc", ""),
            "doi": identifiers.get("doi", ""),
            "title": title,
            "abstract": "\n".join(abstract_parts),
            "publication_types": publication_types,
            "accessions": accessions,
            "publication_date": publication_date,
            "date_resolution": date_resolution,
            "date_reason": date_reason,
            "date_eligible": accepted,
            "source": "pubmed",
            "exact_si": True,
        }
        record["publication_identity"] = _publication_identity(record)
        records.append(record)
    return records


def _retrieve_pubmed(accessions: list[str], origin: pd.Timestamp, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    calls = 0
    cache_hits = 0
    for start in range(0, len(accessions), 40):
        batch = sorted(accessions[start:start + 40])
        key = hashlib.sha256(f"{origin.date()}\0{'|'.join(batch)}".encode()).hexdigest()[:20]
        batch_root = root / "raw" / "pubmed" / origin.strftime("%Y-%m-%d")
        search_path = batch_root / f"{key}.search.json.gz"
        fetch_path = batch_root / f"{key}.fetch.xml.gz"
        maximum = (origin - pd.Timedelta(days=1)).strftime("%Y/%m/%d")
        term = f"({' OR '.join(f'{value}[si]' for value in batch)}) AND (\"1800/01/01\"[pdat] : \"{maximum}\"[pdat])"
        search_url = f"{PUBMED_BASE}/esearch.fcgi?" + urllib.parse.urlencode({"db": "pubmed", "term": term, "retmode": "json", "retmax": 10000, "sort": "pub_date", "tool": "kapso_relbench", "email": "noreply@example.com"})
        if search_path.exists():
            search_payload = _gzip_read(search_path)
            cache_hits += 1
        else:
            search_payload = _request(search_url)
            _gzip_write(search_path, search_payload)
            calls += 1
            time.sleep(0.34)
        identifiers = json.loads(search_payload)["esearchresult"]["idlist"]
        if not identifiers:
            continue
        fetch_url = f"{PUBMED_BASE}/efetch.fcgi?" + urllib.parse.urlencode({"db": "pubmed", "id": ",".join(identifiers), "retmode": "xml", "tool": "kapso_relbench", "email": "noreply@example.com"})
        if fetch_path.exists():
            fetch_payload = _gzip_read(fetch_path)
            cache_hits += 1
        else:
            fetch_payload = _request(fetch_url)
            _gzip_write(fetch_path, fetch_payload)
            calls += 1
            time.sleep(0.34)
        records.extend(_parse_pubmed(fetch_payload, origin))
    return records, {"calls": calls, "cache_hits": cache_hits, "query_template": "(<NCT>[si] OR ...) AND (1800/01/01[pdat] : <origin-1d>[pdat])"}


# Europe PMC

class _RateLimiter:
    def __init__(self, interval: float):
        self.interval = interval
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_time - now)
            self.next_time = max(now, self.next_time) + self.interval
        if delay:
            time.sleep(delay)


def _parse_europe_pmc(payload: bytes, queried_accession: str, origin: pd.Timestamp) -> list[dict[str, Any]]:
    data = json.loads(payload)
    records = []
    for item in data.get("resultList", {}).get("result", []):
        full_dates = [value for value in [item.get("firstPublicationDate"), item.get("electronicPublicationDate")] if value and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value))]
        years = [int(item["pubYear"])] if str(item.get("pubYear", "")).isdigit() else []
        accepted, publication_date, date_resolution, date_reason = _date_decision(full_dates, years, origin)
        record = {
            "pmid": str(item.get("pmid") or ""),
            "pmcid": str(item.get("pmcid") or ""),
            "doi": str(item.get("doi") or ""),
            "title": str(item.get("title") or ""),
            "abstract": str(item.get("abstractText") or ""),
            "publication_types": list(item.get("pubTypeList", {}).get("pubType", [])),
            "accessions": [queried_accession],
            "publication_date": publication_date,
            "date_resolution": date_resolution,
            "date_reason": date_reason,
            "date_eligible": accepted,
            "source": "europe_pmc",
            "exact_si": False,
        }
        record["publication_identity"] = _publication_identity(record)
        records.append(record)
    return records


def _retrieve_europe_pmc(accessions: list[str], origin: pd.Timestamp, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    limiter = _RateLimiter(0.205)
    calls = 0
    cache_hits = 0
    counter_lock = threading.Lock()
    maximum = (origin - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    def retrieve_one(accession: str) -> list[dict[str, Any]]:
        nonlocal calls, cache_hits
        path = root / "raw" / "europe_pmc" / origin.strftime("%Y-%m-%d") / f"{accession}.json.gz"
        query = f"ACCESSION_ID:{accession} AND FIRST_PDATE:[1800-01-01 TO {maximum}]"
        url = EUROPE_PMC_BASE + "?" + urllib.parse.urlencode({"query": query, "resultType": "core", "format": "json", "pageSize": 1000})
        if path.exists():
            payload = _gzip_read(path)
            with counter_lock:
                cache_hits += 1
        else:
            limiter.wait()
            payload = _request(url)
            _gzip_write(path, payload)
            with counter_lock:
                calls += 1
        return _parse_europe_pmc(payload, accession, origin)

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for current in executor.map(retrieve_one, accessions):
            records.extend(current)
    return records, {"calls": calls, "cache_hits": cache_hits, "query_template": "ACCESSION_ID:<NCT> AND FIRST_PDATE:[1800-01-01 TO <origin-1d>]"}


# Retrieval assembly

def _merge_records(pubmed: list[dict[str, Any]], europe: list[dict[str, Any]], accessions: list[str], origin: pd.Timestamp) -> list[dict[str, Any]]:
    relations: dict[tuple[str, str], list[dict[str, Any]]] = {}
    allowed = set(accessions)
    for record in pubmed:
        for accession in record["accessions"]:
            if accession in allowed:
                relations.setdefault((accession, record["publication_identity"]), []).append(record)
    for record in europe:
        accession = record["accessions"][0]
        if accession in allowed:
            relations.setdefault((accession, record["publication_identity"]), []).append(record)
    merged = []
    for (accession, identity), values in sorted(relations.items()):
        abstracts = [value["abstract"] for value in values if value["abstract"]]
        titles = [value["title"] for value in values if value["title"]]
        eligible = [value for value in values if value["date_eligible"]]
        date_source = min(eligible, key=lambda value: (len(value["publication_date"]), value["publication_date"])) if eligible else values[0]
        merged.append({
            "origin": origin.strftime("%Y-%m-%d"),
            "queried_nct_id": accession,
            "publication_identity": identity,
            "pmid": next((value["pmid"] for value in values if value["pmid"]), ""),
            "pmcid": next((value["pmcid"] for value in values if value["pmcid"]), ""),
            "doi": next((value["doi"] for value in values if value["doi"]), ""),
            "title": max(titles, key=len) if titles else "",
            "abstract": max(abstracts, key=len) if abstracts else "",
            "publication_types": sorted({item for value in values for item in value["publication_types"]}),
            "sources": sorted({value["source"] for value in values}),
            "exact_si": any(value["exact_si"] for value in values),
            "publication_date": date_source["publication_date"],
            "date_resolution": date_source["date_resolution"],
            "date_reason": date_source["date_reason"],
            "date_eligible": bool(eligible),
            "content_hash": hashlib.sha256(f"{max(titles, key=len) if titles else ''}\0{max(abstracts, key=len) if abstracts else ''}".encode("utf-8", errors="replace")).hexdigest(),
        })
    return merged


def retrieve_origin(linkage: pd.DataFrame, origin: pd.Timestamp, cache_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = cache_root / "literature_v2"
    origin = pd.Timestamp(origin).normalize()
    eligible_linkage = linkage[
        linkage["linked"].astype(bool)
        & linkage["external_nct_id"].notna()
        & (pd.to_numeric(linkage["audit_agreements"], errors="coerce").fillna(0) >= 2)
        & pd.to_datetime(linkage["timestamp"]).eq(origin)
    ]
    accessions = sorted({_normalize_identifier(value) for value in eligible_linkage["external_nct_id"] if _normalize_identifier(value)})
    parsed_path = root / "parsed" / f"{origin.strftime('%Y-%m-%d')}.jsonl"
    diagnostics_path = root / "parsed" / f"{origin.strftime('%Y-%m-%d')}.diagnostics.json"
    if parsed_path.exists() and diagnostics_path.exists():
        frame = pd.read_json(parsed_path, lines=True) if parsed_path.stat().st_size else pd.DataFrame()
        diagnostics = json.loads(diagnostics_path.read_text())
        diagnostics["parsed_cache_hit"] = True
        return frame, diagnostics
    started = time.time()
    pubmed, pubmed_diagnostics = _retrieve_pubmed(accessions, origin, root)
    europe, europe_diagnostics = _retrieve_europe_pmc(accessions, origin, root)
    merged = _merge_records(pubmed, europe, accessions, origin)
    frame = pd.DataFrame(merged)
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = parsed_path.with_suffix(".jsonl.part")
    if len(frame):
        frame.to_json(temporary, orient="records", lines=True)
    else:
        temporary.write_text("")
    os.replace(temporary, parsed_path)
    evidence_accessions = set(frame.loc[frame["date_eligible"].astype(bool), "queried_nct_id"]) if len(frame) else set()
    diagnostics = {
        "retrieval_version": RETRIEVAL_VERSION,
        "origin": origin.strftime("%Y-%m-%d"),
        "linked_rows": int(len(eligible_linkage)),
        "queried_trials": int(len(accessions)),
        "trials_with_admissible_evidence": int(len(evidence_accessions)),
        "coverage": float(len(evidence_accessions) / max(1, len(accessions))),
        "records": int(len(frame)),
        "admissible_records": int(frame["date_eligible"].sum()) if len(frame) else 0,
        "complete_date_share": float((frame["date_resolution"] == "complete").mean()) if len(frame) else 0.0,
        "sources": {"pubmed": pubmed_diagnostics, "europe_pmc": europe_diagnostics},
        "elapsed_seconds": float(time.time() - started),
        "trials_per_minute": float(len(accessions) / max((time.time() - started) / 60.0, 1e-6)),
        "retrieved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "date_policy": "accept a complete electronic/first-publication date only when strictly earlier than origin; year-only dates only when year is strictly earlier than origin year",
        "parsed_cache_hit": False,
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    register_artifact(cache_root, {
        "name": "generic_exp_4 pre-origin publication cache",
        "path": "literature_v2",
        "description": "Raw PubMed and Europe PMC responses, origin-specific date decisions, deterministic candidates, hosted adjudications, and publication-expert vectors.",
        "content_key": "rel-trial-study-outcome:literature-retrieval-v2:primary-result-adjudication-v2",
        "rebuild_hint": "Run publication_pipeline.py; retrieval resumes from origin/accession-addressed raw responses.",
    })
    return frame, diagnostics


# Trial context and prefilter

def build_trial_contexts(linkage: pd.DataFrame, origin: pd.Timestamp, projected_root: Path) -> dict[str, dict[str, Any]]:
    from registry_clock import ORIGIN_SNAPSHOTS

    snapshot = ORIGIN_SNAPSHOTS[pd.Timestamp(origin).normalize()]
    root = projected_root / snapshot
    accessions = set(linkage.loc[pd.to_datetime(linkage["timestamp"]).eq(pd.Timestamp(origin)), "external_nct_id"].dropna().astype(str))
    studies = pd.read_parquet(root / "studies.parquet")
    studies = studies[studies["nct_id"].isin(accessions)].set_index("nct_id")
    conditions = pd.read_parquet(root / "browse_conditions.parquet")
    interventions = pd.read_parquet(root / "browse_interventions.parquet")
    designs = pd.read_parquet(root / "designs.parquet").set_index("nct_id")
    outcomes = pd.read_parquet(root / "design_outcomes.parquet")
    outcomes = outcomes[outcomes["outcome_type"].fillna("").str.casefold().eq("primary")]
    condition_map = conditions[conditions["nct_id"].isin(accessions)].groupby("nct_id")["mesh_term"].apply(lambda values: sorted(set(values.dropna().astype(str)))[:20]).to_dict()
    intervention_map = interventions[interventions["nct_id"].isin(accessions)].groupby("nct_id")["mesh_term"].apply(lambda values: sorted(set(values.dropna().astype(str)))[:20]).to_dict()
    outcome_map = outcomes[outcomes["nct_id"].isin(accessions)].groupby("nct_id").apply(lambda frame: [{"title": str(row["measure"] or ""), "time_frame": str(row["time_frame"] or "")} for _, row in frame.head(12).iterrows()], include_groups=False).to_dict()
    contexts = {}
    for accession in accessions:
        study = studies.loc[accession] if accession in studies.index else pd.Series(dtype=object)
        design = designs.loc[accession] if accession in designs.index else pd.Series(dtype=object)
        if isinstance(study, pd.DataFrame):
            study = study.iloc[0]
        if isinstance(design, pd.DataFrame):
            design = design.iloc[0]
        contexts[accession] = {
            "nct_id": accession,
            "origin": pd.Timestamp(origin).strftime("%Y-%m-%d"),
            "brief_title": str(study.get("brief_title") or ""),
            "official_title": str(study.get("official_title") or ""),
            "conditions": condition_map.get(accession, []),
            "interventions": intervention_map.get(accession, []),
            "phase": str(study.get("phase") or ""),
            "enrollment": str(study.get("enrollment") or ""),
            "arms": str(study.get("number_of_arms") or ""),
            "allocation": str(design.get("allocation") or ""),
            "masking": str(design.get("masking") or ""),
            "primary_outcomes": outcome_map.get(accession, []),
        }
    return contexts


def prefilter_candidates(records: pd.DataFrame, contexts: dict[str, dict[str, Any]], maximum: int = 3) -> pd.DataFrame:
    if records.empty:
        return records.copy()
    scored = []
    for _, row in records[records["date_eligible"].astype(bool)].iterrows():
        context = contexts.get(str(row["queried_nct_id"]), {})
        title = str(row["title"])
        abstract = str(row["abstract"])
        publication_types = " ".join(row["publication_types"] if isinstance(row["publication_types"], list) else [])
        trial_title = f"{context.get('brief_title', '')} {context.get('official_title', '')}"
        primary = " ".join(f"{item.get('title', '')} {item.get('time_frame', '')}" for item in context.get("primary_outcomes", []))
        score = 10.0 if bool(row["exact_si"]) else 8.0
        score += 2.0 if RESULT_WORDS.search(title) else 0.0
        score += 2.0 if RESULT_WORDS.search(abstract) else 0.0
        score += 4.0 * _token_similarity(title, trial_title)
        score += 3.0 * _token_similarity(f"{title} {abstract[:4000]}", primary)
        score += 2.0 if re.search(r"randomized controlled trial|clinical trial", publication_types, re.IGNORECASE) else 0.0
        score -= 5.0 if PROTOCOL_WORDS.search(f"{title} {publication_types}") else 0.0
        score -= 5.0 if REVIEW_WORDS.search(f"{title} {publication_types}") else 0.0
        score -= 3.0 if len(abstract.strip()) < 100 else 0.0
        item = row.to_dict()
        item["prefilter_score"] = float(score)
        item["title_similarity"] = float(_token_similarity(title, trial_title))
        item["primary_outcome_similarity"] = float(_token_similarity(f"{title} {abstract[:4000]}", primary))
        scored.append(item)
    frame = pd.DataFrame(scored)
    if frame.empty:
        return frame
    frame = frame.sort_values(["queried_nct_id", "prefilter_score", "publication_date", "publication_identity"], ascending=[True, False, False, True])
    return frame.groupby("queried_nct_id", sort=False).head(maximum).reset_index(drop=True)


# Hosted adjudication

def _adjudication_key(row: pd.Series, context: dict[str, Any]) -> str:
    payload = json.dumps({"model": HOSTED_MODEL, "prompt": PROMPT_VERSION, "origin": row["origin"], "publication_identity": row["publication_identity"], "content_hash": row["content_hash"], "context": context}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _adjudicate_one(row: pd.Series, context: dict[str, Any]) -> dict[str, Any]:
    client = OpenAI(timeout=180.0, max_retries=0)
    input_text = f"PRE-ORIGIN REGISTRY CONTEXT\n{json.dumps(context, sort_keys=True)}\n\nPUBLICATION TITLE\n{row['title']}\n\nCOMPLETE ABSTRACT\n{row['abstract']}"
    last_error = None
    for attempt in range(6):
        try:
            response = client.responses.create(
                model=HOSTED_MODEL,
                instructions=SYSTEM_PROMPT,
                input=input_text,
                tools=[],
                text={"format": {"type": "json_schema", "name": "primary_result_adjudication", "strict": True, "schema": ADJUDICATION_SCHEMA}},
                reasoning={"effort": "none"},
                max_output_tokens=400,
                temperature=0,
                store=False,
            )
            return json.loads(response.output_text)
        except Exception as error:
            last_error = error
            if attempt + 1 < 6:
                time.sleep(min(20.0, 0.8 * (2 ** attempt)) + random.random())
    raise RuntimeError(f"Publication adjudication failed: {last_error}")


def adjudicate_candidates(candidates: pd.DataFrame, contexts: dict[str, dict[str, Any]], cache_root: Path, concurrency: int = 32, probe_only: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = cache_root / "literature_v2" / "adjudications"
    root.mkdir(parents=True, exist_ok=True)
    if candidates.empty:
        return candidates.copy(), {"calls": 0, "cache_hits": 0, "failures": 0, "usable_rows": 0, "probe_passed": False}
    selected = candidates.head(1).copy() if probe_only else candidates.copy()
    keys = {int(index): _adjudication_key(row, contexts[str(row["queried_nct_id"])]) for index, row in selected.iterrows()}
    results: dict[int, dict[str, Any]] = {}
    missing = []
    for index, key in keys.items():
        path = root / f"{key}.json"
        if path.exists():
            results[index] = json.loads(path.read_text())["result"]
        else:
            missing.append(index)
    failures = []
    if missing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            future_to_index = {executor.submit(_adjudicate_one, selected.loc[index], contexts[str(selected.loc[index, "queried_nct_id"])]): index for index in missing}
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                    key = keys[index]
                    payload = {"key": key, "model": HOSTED_MODEL, "prompt_version": PROMPT_VERSION, "origin": selected.loc[index, "origin"], "publication_identity": selected.loc[index, "publication_identity"], "content_hash": selected.loc[index, "content_hash"], "result": result}
                    temporary = root / f"{key}.{os.getpid()}.part"
                    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
                    os.replace(temporary, root / f"{key}.json")
                except Exception as error:
                    failures.append({"index": int(index), "error": str(error)})
    if failures:
        raise RuntimeError(f"Hosted publication adjudication failures: {failures[:5]}; total={len(failures)}")
    rows = []
    for index, row in selected.iterrows():
        item = row.to_dict()
        item.update({f"judgment_{key}": value for key, value in results[index].items()})
        rows.append(item)
    frame = pd.DataFrame(rows)
    usable = frame["judgment_is_this_trial"].astype(bool) & ~frame["judgment_insufficient_evidence"].astype(bool) & frame["judgment_primary_endpoint_met"].isin(["yes", "no", "mixed"])
    return frame, {"calls": len(missing), "cache_hits": len(selected) - len(missing), "failures": 0, "usable_rows": int(usable.sum()), "probe_passed": True, "model": HOSTED_MODEL, "prompt_version": PROMPT_VERSION}


# Feature assembly

def publication_features(linkage: pd.DataFrame, records: pd.DataFrame, adjudications: pd.DataFrame) -> pd.DataFrame:
    result = linkage[["row_id", "nct_id", "external_nct_id", "timestamp", "split", "linked", "link_confidence"]].copy()
    numeric = [
        "publication_count", "primary_report_count", "endpoint_match_confidence", "met_count",
        "not_met_count", "mixed_count", "explicit_p_significant_count", "explicit_p_nonsignificant_count",
        "final_count", "interim_count", "months_since_newest", "source_agreement", "evidence_confidence",
        "exact_si_count", "usable_evidence",
    ]
    for column in numeric:
        result[column] = 0.0
    if records.empty:
        return result
    eligible = records[records["date_eligible"].astype(bool)].copy()
    record_group = eligible.groupby("queried_nct_id")
    record_counts = record_group.size()
    exact_counts = record_group["exact_si"].sum()
    agreement = record_group["sources"].apply(lambda values: float(any(len(value) > 1 for value in values)))
    for index, row in result.iterrows():
        accession = str(row["external_nct_id"])
        if accession in record_counts:
            result.at[index, "publication_count"] = float(record_counts[accession])
            result.at[index, "exact_si_count"] = float(exact_counts[accession])
            result.at[index, "source_agreement"] = float(agreement[accession])
            dates = pd.to_datetime(eligible.loc[eligible["queried_nct_id"] == accession, "publication_date"], errors="coerce")
            if dates.notna().any():
                result.at[index, "months_since_newest"] = max(0.0, float((pd.Timestamp(row["timestamp"]) - dates.max()).days / 30.4375))
    if adjudications.empty:
        return result
    frame = adjudications.copy()
    frame["usable"] = frame["judgment_is_this_trial"].astype(bool) & ~frame["judgment_insufficient_evidence"].astype(bool) & frame["judgment_primary_endpoint_met"].isin(["yes", "no", "mixed"])
    frame["primary"] = frame["judgment_report_type"].eq("primary-results").astype(float)
    frame["met"] = frame["judgment_primary_endpoint_met"].eq("yes").astype(float)
    frame["not_met"] = frame["judgment_primary_endpoint_met"].eq("no").astype(float)
    frame["mixed"] = frame["judgment_primary_endpoint_met"].eq("mixed").astype(float)
    frame["final"] = frame["judgment_final_status"].eq("final").astype(float)
    frame["interim"] = frame["judgment_final_status"].eq("interim").astype(float)
    parsed_p = frame["judgment_explicit_p_value"].fillna("").astype(str).str.extract(r"([01]?(?:\.\d+))", expand=False)
    parsed_p = pd.to_numeric(parsed_p, errors="coerce")
    frame["p_significant"] = (parsed_p <= 0.05).astype(float)
    frame["p_nonsignificant"] = (parsed_p > 0.05).astype(float)
    frame["match_confidence"] = frame["judgment_endpoint_match"].astype(float) * frame["judgment_confidence"].astype(float) / 25.0
    grouped = frame.groupby("queried_nct_id").agg(
        primary_report_count=("primary", "sum"),
        endpoint_match_confidence=("match_confidence", "max"),
        met_count=("met", "sum"),
        not_met_count=("not_met", "sum"),
        mixed_count=("mixed", "sum"),
        explicit_p_significant_count=("p_significant", "sum"),
        explicit_p_nonsignificant_count=("p_nonsignificant", "sum"),
        final_count=("final", "sum"),
        interim_count=("interim", "sum"),
        evidence_confidence=("judgment_confidence", "max"),
        usable_evidence=("usable", "max"),
    )
    for column in grouped.columns:
        result[column] = result["external_nct_id"].map(grouped[column]).fillna(result[column]).astype(float)
    return result
