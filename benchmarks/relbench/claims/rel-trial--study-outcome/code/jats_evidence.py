# Imports

from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lxml import etree


# Configuration

EXTRACTOR_VERSION = "jats-endpoint-facts-v1"
LEXICAL_THRESHOLD = 0.70
TIME_TOLERANCE = 0.25
P_VALUE_PATTERN = re.compile(
    r"(?i)(?<![a-z])p(?:\s*[-_ ]?\s*value)?\s*(<=|>=|=|<|>|\u2264|\u2265)\s*"
    r"(\d*\.\d+(?:\s*[eE]\s*[+-]?\s*\d+)?|\d+(?:\.\d+)?\s*[eE]\s*[+-]?\s*\d+)"
)
CI_PATTERN = re.compile(
    r"(?i)(?:(\d{2,3}(?:\.\d+)?)\s*%\s*)?(?:CI|confidence\s+interval)\s*[:=]?\s*"
    r"[\[(]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:,|to|[-\u2013\u2014])\s*([-+]?\d+(?:\.\d+)?)"
)
ESTIMATE_PATTERN = re.compile(
    r"(?i)\b(HR|hazard ratio|OR|odds ratio|RR|risk ratio|risk difference|mean difference)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)"
)
SAMPLE_PATTERN = re.compile(r"(?i)(?:\bn\s*=\s*|\bnumber\s+(?:of\s+)?(?:participants|patients|subjects)\s*[:=]?\s*)(\d[\d,]*)")
NS_PATTERN = re.compile(r"(?i)(?:\bnot\s+(?:statistically\s+)?significant\b|(?<![a-z])N\.?S\.?(?![a-z]))")
RESULT_TITLE_PATTERN = re.compile(r"(?i)\b(result|analysis|finding|efficacy|outcome|endpoint|conclusion|discussion)\b")
INTERIM_PATTERN = re.compile(r"(?i)\b(interim|preliminary|protocol|rationale|baseline\s+characteristics)\b")
PROHIBITED_PATTERN = re.compile(r"(?i)\b(retraction|retracted|correction|erratum|expression\s+of\s+concern)\b")
TOKEN_STOP = {
    "a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with",
    "primary", "endpoint", "outcome", "measure", "change", "baseline", "assessed", "assessment",
}
ABBREVIATIONS = {
    "overall survival": "os",
    "progression free survival": "pfs",
    "disease free survival": "dfs",
    "event free survival": "efs",
    "objective response rate": "orr",
    "complete response": "cr",
    "quality of life": "qol",
    "adverse event": "ae",
    "dose limiting toxicity": "dlt",
}


# Records

@dataclass(frozen=True)
class Fact:
    origin: str
    nct_id: str
    publication_identity: str
    pmcid: str
    source_document: str
    node_path: str
    evidence_location: str
    endpoint_name: str
    arm: str
    time_point: str
    analysis_population: str
    p_modifier: str
    p_value: float | None
    confidence_level: float | None
    ci_lower: float | None
    ci_upper: float | None
    effect_type: str
    effect_value: float | None
    sample_size: int | None
    textual_ns: bool
    surrounding_text: str
    extraction_rule: str
    content_hash: str


# Text normalization

def normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("\u2264", " <= ").replace("\u2265", " >= ").replace("\u00b1", " plusminus ")
    text = re.sub(r"(?<=\d)\s*(months?|mos?|years?|yrs?|weeks?|wks?|days?|hrs?|hours?)\b", r" \1", text)
    text = re.sub(r"\bper\s+cent\b", "percent", text)
    for phrase, short in ABBREVIATIONS.items():
        text = text.replace(phrase, f" {short} ")
    text = re.sub(r"[^a-z0-9<>.=+%-]+", " ", text)
    return " ".join(text.split())


def tokens(value: Any) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", normalized_text(value)) if token not in TOKEN_STOP and len(token) > 1}


def parsed_days(value: Any) -> tuple[float, float] | None:
    text = normalized_text(value)
    values = []
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(days?|weeks?|wks?|months?|mos?|years?|yrs?)\b", text):
        multiplier = 1.0
        if unit.startswith(("week", "wk")):
            multiplier = 7.0
        elif unit.startswith(("month", "mo")):
            multiplier = 30.4375
        elif unit.startswith(("year", "yr")):
            multiplier = 365.25
        values.append(float(number) * multiplier)
    if not values:
        return None
    return min(values), max(values)


def timeframe_compatibility(registered: Any, reported: Any) -> float:
    left = parsed_days(registered)
    right = parsed_days(reported)
    if left is None or right is None:
        return 0.5
    distances = [abs(a - b) / max(a, b, 1.0) for a in left for b in right]
    distance = min(distances)
    if distance <= TIME_TOLERANCE:
        return 1.0
    return max(0.0, 1.0 - distance)


def lexical_similarity(left: Any, right: Any) -> tuple[float, float, float]:
    a = normalized_text(left)
    b = normalized_text(right)
    if not a or not b:
        return 0.0, 0.0, 0.0
    if a == b or a in b or b in a:
        exact = 1.0
    else:
        exact = 0.0
    ta = tokens(a)
    tb = tokens(b)
    overlap = len(ta & tb)
    token_score = 2.0 * overlap / max(1, len(ta) + len(tb))
    character_score = SequenceMatcher(None, a, b, autojunk=False).ratio()
    return exact, character_score, token_score


# XML safety

def element_text(element: etree._Element | None) -> str:
    if element is None:
        return ""
    return " ".join(" ".join(element.itertext()).split())


def xml_date(element: etree._Element) -> pd.Timestamp | None:
    def child(name: str) -> str:
        values = element.xpath(f'./*[local-name()="{name}"]//text()')
        return str(values[0]).strip() if values else ""
    year = child("year") or child("Year")
    month = child("month") or child("Month")
    day = child("day") or child("Day")
    if not year.isdigit() or not month.isdigit() or not day.isdigit():
        return None
    try:
        return pd.Timestamp(year=int(year), month=int(month), day=int(day))
    except Exception:
        return None


def safe_xml_tree(payload: bytes, record: dict[str, Any]) -> tuple[etree._Element | None, str]:
    try:
        root = etree.fromstring(payload, parser=etree.XMLParser(resolve_entities=False, no_network=True, recover=False, huge_tree=True))
    except Exception as error:
        return None, f"xml_parse_failure:{type(error).__name__}"
    title_values = root.xpath('.//*[local-name()="article-title"]//text()')
    title = " ".join(str(value) for value in title_values)
    if PROHIBITED_PATTERN.search(f"{root.attrib.get('article-type', '')} {title}"):
        return None, "correction_or_retraction_rejected"
    article_ids = []
    for element in root.xpath('.//*[local-name()="article-id"]'):
        if str(element.attrib.get("pub-id-type", "")).casefold() == "pmc":
            article_ids.append(element_text(element).upper().replace("PMC", ""))
    expected = str(record.get("pmcid", "")).upper().replace("PMC", "")
    if article_ids and expected and expected not in article_ids:
        return None, "pmcid_mismatch"
    origin = pd.Timestamp(record["origin"]).normalize()
    if str(record.get("date_resolution", "")) != "complete":
        return None, "full_text_requires_complete_pre_origin_date"
    try:
        if pd.Timestamp(record["publication_date"]).normalize() >= origin:
            return None, "full_text_not_pre_origin"
    except Exception:
        return None, "full_text_missing_publication_date"
    for element in root.xpath('.//*[local-name()="date"]'):
        date_type = str(element.attrib.get("date-type", "")).casefold()
        if any(token in date_type for token in ["updated", "corrected", "revision", "rev-recd"]):
            value = xml_date(element)
            if value is None:
                return None, "unverified_version_marker"
            if value.normalize() >= origin:
                return None, "post_origin_document_version"
    return root, "pmcid_specific_complete_pre_origin_verified_xml"


# Table structure

def _span(value: Any) -> int:
    try:
        return max(1, int(value))
    except Exception:
        return 1


def _table_grid(table: etree._Element) -> tuple[list[list[str]], list[bool]]:
    rows = table.xpath('.//*[local-name()="tr"]')
    pending: dict[int, tuple[int, str]] = {}
    grid: list[list[str]] = []
    header_flags: list[bool] = []
    for row in rows:
        values: list[str] = []
        column = 0
        cells = row.xpath('./*[local-name()="th" or local-name()="td"]')
        while cells or any(remaining > 0 for remaining, _ in pending.values()):
            if column in pending and pending[column][0] > 0:
                remaining, value = pending[column]
                while len(values) <= column:
                    values.append("")
                values[column] = value
                if remaining == 1:
                    del pending[column]
                else:
                    pending[column] = (remaining - 1, value)
                column += 1
                continue
            if not cells:
                if column > max(pending, default=-1):
                    break
                while len(values) <= column:
                    values.append("")
                column += 1
                continue
            cell = cells.pop(0)
            value = element_text(cell)
            colspan = _span(cell.attrib.get("colspan", 1))
            rowspan = _span(cell.attrib.get("rowspan", 1))
            for offset in range(colspan):
                while len(values) <= column + offset:
                    values.append("")
                values[column + offset] = value
                if rowspan > 1:
                    pending[column + offset] = (rowspan - 1, value)
            column += colspan
        grid.append(values)
        header_flags.append(bool(cells) or all(str(cell.tag).split("}")[-1] == "th" for cell in row.xpath('./*[local-name()="th" or local-name()="td"]')))
    width = max((len(row) for row in grid), default=0)
    return [row + [""] * (width - len(row)) for row in grid], header_flags


def structured_tables(root: etree._Element) -> list[dict[str, Any]]:
    result = []
    tree = root.getroottree()
    for wrap in root.xpath('.//*[local-name()="table-wrap"]'):
        table_values = wrap.xpath('.//*[local-name()="table"]')
        if not table_values:
            continue
        table = table_values[0]
        grid, inferred_headers = _table_grid(table)
        if not grid:
            continue
        thead_rows = table.xpath('./*[local-name()="thead"]/*[local-name()="tr"]')
        header_count = len(thead_rows)
        if not header_count:
            header_count = 0
            for flag in inferred_headers:
                if not flag:
                    break
                header_count += 1
        header_count = min(header_count, max(0, len(grid) - 1))
        width = len(grid[0])
        headers = []
        for column in range(width):
            parts = []
            for row in grid[:header_count]:
                value = row[column]
                if value and value not in parts:
                    parts.append(value)
            headers.append(" | ".join(parts))
        caption_nodes = wrap.xpath('./*[local-name()="caption"]')
        label_nodes = wrap.xpath('./*[local-name()="label"]')
        foot_nodes = wrap.xpath('.//*[local-name()="table-wrap-foot" or local-name()="tfoot"]')
        caption = element_text(caption_nodes[0]) if caption_nodes else ""
        label = element_text(label_nodes[0]) if label_nodes else ""
        footnotes = " | ".join(element_text(value) for value in foot_nodes if element_text(value))
        body = grid[header_count:] if header_count else grid
        rows = []
        for row_index, row in enumerate(body):
            nonempty = [value for value in row if value]
            row_label = nonempty[0] if nonempty else ""
            cells = []
            for column, value in enumerate(row):
                if not value:
                    continue
                cells.append({"column_index": column, "column_header": headers[column], "text": value})
            rows.append({"row_index": row_index, "row_label": row_label, "cells": cells})
        result.append({
            "label": label,
            "caption": caption,
            "headers": headers,
            "rows": rows,
            "footnotes": footnotes,
            "node_path": tree.getpath(wrap),
            "xml": etree.tostring(table, encoding="unicode"),
        })
    return result


# Fact extraction

def _first_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    return next(pattern.finditer(text), None)


def _context_fields(text: str) -> tuple[str, str, str]:
    arm = ""
    time_point = ""
    population = ""
    arm_match = re.search(r"(?i)\b(?:arm|group|cohort)\s+[A-Za-z0-9+._-]+(?:\s+[A-Za-z0-9+._-]+){0,4}", text)
    if arm_match:
        arm = arm_match.group(0)
    time_match = re.search(r"(?i)\b(?:at|after|through|week|month|day|year)\s*[-:]?\s*\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|years?)?", text)
    if time_match:
        time_point = time_match.group(0)
    population_match = re.search(r"(?i)\b(?:intention[- ]to[- ]treat|ITT|per[- ]protocol|modified\s+ITT|safety\s+population|evaluable\s+(?:patients|population))\b", text)
    if population_match:
        population = population_match.group(0)
    return arm, time_point, population


def facts_from_text(
    text: str,
    endpoint_name: str,
    location: str,
    node_path: str,
    record: dict[str, Any],
    rule: str,
) -> list[Fact]:
    normalized = unicodedata.normalize("NFKC", text)
    p_matches = list(P_VALUE_PATTERN.finditer(normalized))
    ns = bool(NS_PATTERN.search(normalized))
    if not p_matches and not ns:
        return []
    ci = _first_match(CI_PATTERN, normalized)
    estimate = _first_match(ESTIMATE_PATTERN, normalized)
    sample = _first_match(SAMPLE_PATTERN, normalized)
    arm, time_point, population = _context_fields(normalized)
    base = {
        "origin": str(record["origin"]),
        "nct_id": str(record["queried_nct_id"]),
        "publication_identity": str(record["publication_identity"]),
        "pmcid": str(record.get("pmcid", "")),
        "source_document": str(record.get("document_version_id") or record.get("pmcid") or record.get("publication_identity")),
        "node_path": node_path,
        "evidence_location": location,
        "endpoint_name": endpoint_name,
        "arm": arm,
        "time_point": time_point,
        "analysis_population": population,
        "confidence_level": float(ci.group(1)) if ci and ci.group(1) else None,
        "ci_lower": float(ci.group(2)) if ci else None,
        "ci_upper": float(ci.group(3)) if ci else None,
        "effect_type": estimate.group(1) if estimate else "",
        "effect_value": float(estimate.group(2)) if estimate else None,
        "sample_size": int(sample.group(1).replace(",", "")) if sample else None,
        "textual_ns": ns,
        "surrounding_text": normalized[:4000],
        "extraction_rule": rule,
    }
    facts = []
    for match in p_matches:
        modifier = match.group(1).replace("\u2264", "<=").replace("\u2265", ">=")
        raw_value = re.sub(r"\s+", "", match.group(2))
        try:
            value = float(raw_value)
        except Exception:
            continue
        payload = f"{base['source_document']}\0{node_path}\0{match.group(0)}\0{normalized}".encode("utf-8", errors="replace")
        facts.append(Fact(**base, p_modifier=modifier, p_value=value, content_hash=hashlib.sha256(payload).hexdigest()))
    if ns and not p_matches:
        payload = f"{base['source_document']}\0{node_path}\0NS\0{normalized}".encode("utf-8", errors="replace")
        facts.append(Fact(**base, p_modifier="NS", p_value=None, content_hash=hashlib.sha256(payload).hexdigest()))
    return facts


def extract_document(record: dict[str, Any], xml_path: Path) -> tuple[list[Fact], dict[str, Any], list[dict[str, Any]]]:
    payload = gzip.decompress(xml_path.read_bytes())
    root, reason = safe_xml_tree(payload, record)
    if root is None:
        return [], {"safe": False, "reason": reason, "tables": 0, "result_sections": 0, "facts": 0}, []
    tables = structured_tables(root)
    facts: list[Fact] = []
    for table in tables:
        prefix = " | ".join(value for value in [table["label"], table["caption"]] if value)
        for row in table["rows"]:
            for cell in row["cells"]:
                context = " | ".join(value for value in [prefix, row["row_label"], cell["column_header"], cell["text"], table["footnotes"]] if value)
                endpoint = " | ".join(value for value in [table["caption"], row["row_label"], cell["column_header"]] if value)
                node_path = f"{table['node_path']}/row[{row['row_index'] + 1}]/cell[{cell['column_index'] + 1}]"
                facts.extend(facts_from_text(context, endpoint, "table", node_path, record, "jats_structured_table_cell_v1"))
    result_sections = []
    tree = root.getroottree()
    for section in root.xpath('.//*[local-name()="sec"]'):
        title_nodes = section.xpath('./*[local-name()="title"]')
        title = element_text(title_nodes[0]) if title_nodes else ""
        if not RESULT_TITLE_PATTERN.search(title):
            continue
        result_sections.append(title)
        for paragraph in section.xpath('./*[local-name()="p"]'):
            value = element_text(paragraph)
            if not value:
                continue
            fragments = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", value)
            for fragment_index, fragment in enumerate(fragments):
                if not P_VALUE_PATTERN.search(fragment) and not NS_PATTERN.search(fragment):
                    continue
                start = max(0, fragment_index - 1)
                end = min(len(fragments), fragment_index + 2)
                surrounding = " ".join(fragments[start:end])
                path = f"{tree.getpath(paragraph)}/sentence[{fragment_index + 1}]"
                facts.extend(facts_from_text(surrounding, surrounding, "results_sentence", path, record, "jats_result_sentence_v1"))
    title_values = root.xpath('.//*[local-name()="article-title"]//text()')
    document_title = " ".join(str(value) for value in title_values)
    interim = bool(INTERIM_PATTERN.search(document_title))
    final_report = bool(result_sections and not interim)
    diagnostics = {
        "safe": True,
        "reason": reason,
        "tables": len(tables),
        "result_sections": len(result_sections),
        "facts": len(facts),
        "document_length": len(payload),
        "final_report": final_report,
        "interim": interim,
        "pubmed_parser_tables": _helper_table_count(payload),
    }
    return facts, diagnostics, tables


def _helper_table_count(payload: bytes) -> int | None:
    try:
        from pubmed_parser import parse_pubmed_table
        parsed = parse_pubmed_table(payload, return_xml=False)
        return len(parsed or [])
    except Exception:
        return None


# Endpoint matching

def _semantic_auxiliary(record: dict[str, Any], rankings: dict[tuple[str, str, str], float]) -> float:
    return float(rankings.get((str(record["origin"]), str(record["queried_nct_id"]), str(record["publication_identity"])), 0.0))


def match_fact(fact: Fact, endpoints: list[dict[str, Any]], semantic: float) -> dict[str, Any]:
    matches = []
    for index, endpoint in enumerate(endpoints):
        measure = str(endpoint.get("title", ""))
        time_frame = str(endpoint.get("time_frame", ""))
        exact, character, token = lexical_similarity(measure, fact.endpoint_name)
        time_score = timeframe_compatibility(time_frame, f"{fact.endpoint_name} {fact.time_point}")
        lexical = max(exact, 0.55 * character + 0.45 * token, token)
        score = min(1.0, 0.80 * lexical + 0.15 * time_score + 0.05 * semantic)
        matches.append({
            "endpoint_index": index,
            "measure": measure,
            "time_frame": time_frame,
            "exact": exact,
            "character": character,
            "token": token,
            "lexical": lexical,
            "semantic_auxiliary": semantic,
            "time_compatibility": time_score,
            "score": score,
        })
    matches.sort(key=lambda value: value["score"], reverse=True)
    best = matches[0] if matches else {"endpoint_index": -1, "score": 0.0, "lexical": 0.0, "time_compatibility": 0.0}
    runner = matches[1]["score"] if len(matches) > 1 else 0.0
    margin = float(best["score"] - runner)
    accepted = bool(
        best["endpoint_index"] >= 0
        and best["lexical"] >= LEXICAL_THRESHOLD
        and best["time_compatibility"] >= 0.5
        and (len(matches) == 1 or margin >= 0.08 or best.get("exact", 0.0) == 1.0)
    )
    return {**best, "runner_score": float(runner), "margin": margin, "accepted": accepted}


def deterministic_verdict(
    record: dict[str, Any],
    facts: list[Fact],
    document_diagnostics: dict[str, Any],
    endpoints: list[dict[str, Any]],
    semantic: float,
) -> dict[str, Any]:
    matched = []
    unmatched_significant = 0
    for fact in facts:
        match = match_fact(fact, endpoints, semantic)
        item = {**asdict(fact), "match": match}
        if match["accepted"]:
            matched.append(item)
        elif fact.p_value is not None and fact.p_modifier in ["=", "<", "<="] and fact.p_value <= 0.05:
            unmatched_significant += 1
    qualifying = [item for item in matched if item["p_value"] is not None and item["p_modifier"] not in [">", ">="] and 0.0 <= item["p_value"] <= 1.0]
    positive = any(item["p_modifier"] in ["=", "<", "<="] and item["p_value"] <= 0.05 for item in qualifying)
    covered = {int(item["match"]["endpoint_index"]) for item in qualifying}
    negative_facts = [item for item in qualifying if item["p_modifier"] in ["=", ">", ">="] and item["p_value"] > 0.05]
    complete_negative = bool(
        endpoints
        and document_diagnostics.get("final_report")
        and len(covered) == len(endpoints)
        and all(index in {int(item["match"]["endpoint_index"]) for item in negative_facts} for index in range(len(endpoints)))
        and unmatched_significant == 0
        and not positive
    )
    if document_diagnostics.get("interim"):
        positive = False
        complete_negative = False
    return {
        "positive": bool(positive),
        "complete_negative": bool(complete_negative),
        "abstain": bool(not positive and not complete_negative),
        "matched_fact_count": len(matched),
        "qualifying_fact_count": len(qualifying),
        "covered_endpoint_count": len(covered),
        "registered_endpoint_count": len(endpoints),
        "unmatched_significant_count": unmatched_significant,
        "facts": matched,
    }


# Origin extraction

def load_semantic_rankings(cache: Path) -> dict[tuple[str, str, str], float]:
    root = cache / "medcpt_endpoint_reranker_v1" / "rankings"
    values: dict[tuple[str, str, str], float] = {}
    for path in root.glob("*.parquet"):
        frame = pd.read_parquet(path, columns=["origin", "queried_nct_id", "publication_identity", "bi_score"])
        scores = pd.to_numeric(frame["bi_score"], errors="coerce").fillna(0.0)
        ranks = scores.groupby(frame["queried_nct_id"]).rank(method="average", pct=True)
        for origin, nct_id, identity, score in zip(frame["origin"], frame["queried_nct_id"], frame["publication_identity"], ranks):
            values[(str(origin), str(nct_id), str(identity))] = float(score)
    return values


def extract_origin(
    records: pd.DataFrame,
    contexts: dict[str, dict[str, Any]],
    cache: Path,
    workers: int = 36,
    maximum_documents: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    xml_root = cache / "literature_v3" / "raw" / "full_text_xml"
    eligible = records[records["date_eligible"].astype(bool) & records["pmcid"].fillna("").astype(str).ne("")].copy()
    eligible = eligible.drop_duplicates(["queried_nct_id", "publication_identity", "pmcid"], keep="first")
    eligible["xml_path"] = eligible["pmcid"].astype(str).str.upper().map(lambda value: xml_root / f"{value}.xml.gz")
    eligible = eligible[eligible["xml_path"].map(Path.exists)].reset_index(drop=True)
    if maximum_documents is not None:
        eligible = eligible.head(maximum_documents)
    semantic = load_semantic_rankings(cache)
    completed = []
    def work(position: int, row: pd.Series) -> tuple[int, dict[str, Any], list[Fact], dict[str, Any], list[dict[str, Any]]]:
        record = row.drop(labels=["xml_path"]).to_dict()
        facts, diagnostics, tables = extract_document(record, Path(row["xml_path"]))
        endpoints = contexts.get(str(row["queried_nct_id"]), {}).get("primary_outcomes", [])
        score = _semantic_auxiliary(record, semantic)
        verdict = deterministic_verdict(record, facts, diagnostics, endpoints, score)
        return position, record, facts, {**diagnostics, **verdict}, tables
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(work, position, row) for position, (_, row) in enumerate(eligible.iterrows())]
        for future in concurrent.futures.as_completed(futures):
            completed.append(future.result())
    completed.sort(key=lambda value: value[0])
    fact_rows = []
    document_rows = []
    fixture_candidates = []
    for _, record, facts, verdict, tables in completed:
        fact_rows.extend(asdict(value) for value in facts)
        document_rows.append({
            "origin": record["origin"],
            "queried_nct_id": record["queried_nct_id"],
            "publication_identity": record["publication_identity"],
            "pmcid": record.get("pmcid", ""),
            "title": record.get("title", ""),
            "publication_date": record.get("publication_date", ""),
            "content_hash": record.get("content_hash", ""),
            "positive": verdict["positive"],
            "complete_negative": verdict["complete_negative"],
            "abstain": verdict["abstain"],
            "matched_fact_count": verdict["matched_fact_count"],
            "qualifying_fact_count": verdict["qualifying_fact_count"],
            "covered_endpoint_count": verdict["covered_endpoint_count"],
            "registered_endpoint_count": verdict["registered_endpoint_count"],
            "unmatched_significant_count": verdict["unmatched_significant_count"],
            "safe": verdict["safe"],
            "safe_reason": verdict["reason"],
            "table_count": verdict["tables"],
            "result_section_count": verdict["result_sections"],
            "fact_count": verdict["facts"] if isinstance(verdict["facts"], int) else len(facts),
            "document_length": verdict.get("document_length", 0),
            "final_report": verdict.get("final_report", False),
            "interim": verdict.get("interim", False),
            "pubmed_parser_tables": verdict.get("pubmed_parser_tables"),
            "matched_facts_json": json.dumps(verdict["facts"], sort_keys=True, allow_nan=False),
        })
        if tables:
            fixture_candidates.append({"record": record, "tables": tables, "verdict": verdict})
    documents = pd.DataFrame(document_rows)
    facts = pd.DataFrame(fact_rows)
    diagnostics = {
        "eligible_records_with_pmcid": int(len(records[records["date_eligible"].astype(bool) & records["pmcid"].fillna("").astype(str).ne("")])),
        "available_safe_xml_candidates": int(len(eligible)),
        "documents_parsed": int(len(documents)),
        "safe_documents": int(documents["safe"].sum()) if len(documents) else 0,
        "tables": int(documents["table_count"].sum()) if len(documents) else 0,
        "documents_with_tables": int((documents["table_count"] > 0).sum()) if len(documents) else 0,
        "result_sections": int(documents["result_section_count"].sum()) if len(documents) else 0,
        "facts": int(len(facts)),
        "documents_with_facts": int((documents["fact_count"] > 0).sum()) if len(documents) else 0,
        "positive_documents": int(documents["positive"].sum()) if len(documents) else 0,
        "complete_negative_documents": int(documents["complete_negative"].sum()) if len(documents) else 0,
        "trials_with_verdict": int(documents.loc[documents["positive"] | documents["complete_negative"], "queried_nct_id"].nunique()) if len(documents) else 0,
        "workers": workers,
        "extractor_version": EXTRACTOR_VERSION,
    }
    return documents, facts, diagnostics, fixture_candidates


def aggregate_trial_verdicts(documents: pd.DataFrame, existing: pd.DataFrame | None = None) -> pd.DataFrame:
    if documents.empty:
        return pd.DataFrame(columns=["external_nct_id", "jats_positive", "jats_complete_negative", "jats_abstain", "jats_conflict", "jats_document_count"])
    rows = []
    for nct_id, frame in documents.groupby("queried_nct_id", sort=False):
        positive = bool(frame["positive"].any())
        negative = bool(frame["complete_negative"].any())
        conflict = positive and negative
        rows.append({
            "external_nct_id": str(nct_id),
            "jats_positive": bool(positive and not conflict),
            "jats_complete_negative": bool(negative and not conflict),
            "jats_abstain": bool(not positive and not negative or conflict),
            "jats_conflict": conflict,
            "jats_document_count": int(len(frame)),
        })
    result = pd.DataFrame(rows)
    if existing is not None and len(existing):
        prior = existing[["external_nct_id", "direct_positive", "direct_complete_negative", "direct_abstain"]].copy()
        result = result.merge(prior, on="external_nct_id", how="left")
        source_conflict = (
            (result["jats_positive"] & result["direct_complete_negative"].fillna(False))
            | (result["jats_complete_negative"] & result["direct_positive"].fillna(False))
        )
        result.loc[source_conflict, ["jats_positive", "jats_complete_negative"]] = False
        result.loc[source_conflict, ["jats_abstain", "jats_conflict"]] = True
    return result


# Fixtures

def save_recon_fixtures(candidates: list[dict[str, Any]], destination: Path, maximum: int = 50) -> dict[str, Any]:
    selected = []
    signatures = set()
    for candidate in candidates:
        for table in candidate["tables"]:
            xml = table["xml"]
            signature = (
                "rowspan" if "rowspan=" in xml else "",
                "colspan" if "colspan=" in xml else "",
                "footnote" if table["footnotes"] else "",
                "unicode_inequality" if any(value in xml for value in ["\u2264", "\u2265"]) else "",
                "missing_cell" if any(not cell for row in table["rows"] for cell in table["headers"]) else "",
                "multi_arm" if re.search(r"(?i)\b(?:arm|group)\s+[A-Za-z0-9]", xml) else "",
                "nested_header" if len([value for value in table["headers"] if " | " in value]) else "",
            )
            if signature in signatures and len(selected) >= 12:
                continue
            signatures.add(signature)
            selected.append({
                "origin": candidate["record"]["origin"],
                "pmcid": candidate["record"].get("pmcid", ""),
                "publication_identity": candidate["record"]["publication_identity"],
                "content_hash": candidate["record"].get("content_hash", ""),
                "signature": [value for value in signature if value],
                "label": table["label"],
                "caption": table["caption"],
                "headers": table["headers"],
                "rows": table["rows"][:20],
                "footnotes": table["footnotes"],
                "node_path": table["node_path"],
            })
            if len(selected) >= maximum:
                break
        if len(selected) >= maximum:
            break
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(selected, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    coverage = {name: sum(name in item["signature"] for item in selected) for name in ["rowspan", "colspan", "footnote", "unicode_inequality", "missing_cell", "multi_arm", "nested_header"]}
    return {"fixtures": len(selected), "coverage": coverage, "path": str(destination)}
