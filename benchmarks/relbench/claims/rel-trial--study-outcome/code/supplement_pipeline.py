# Imports

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from campaign_io import locked_append, register_artifact
from kapso_datasets.common import shared_cache_dir


# Configuration

START = time.time()
VERSION = "historical_supplements_v1"
EUROPE_PMC_REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"
ENDPOINT_MATCH_THRESHOLD = 0.70
ORIGINS = ["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01"]
P_VALUE_PATTERN = re.compile(r"\bp\s*(?:-?value)?\s*(<=|<|=|>|>=)\s*(0?\.\d+|1(?:\.0+)?)", re.IGNORECASE)
CI_PATTERN = re.compile(r"(?:95\s*%\s*)?(?:confidence\s+interval|CI)\s*[:=]?\s*[\[(]?\s*(-?\d+(?:\.\d+)?)\s*[,;\-]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
DATE_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b")


# Runtime

def report(name: str, **values: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"[supplements] {name} elapsed={time.time() - START:.2f}s {payload}".rstrip(), flush=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n")
    os.replace(temporary, path)


# Provenance

def _xml_payload(cache: Path, pmcid: str) -> bytes:
    path = cache / "literature_v3" / "raw" / "full_text_xml" / f"{pmcid}.xml.gz"
    if not path.exists() or path.stat().st_size == 0:
        return b""
    try:
        with gzip.open(path, "rb") as stream:
            return stream.read()
    except Exception:
        return b""


def _supplement_entries(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8", errors="replace")
    entries = []
    pattern = re.compile(r"<supplementary-material\b(?P<open>[^>]*)>(?P<body>.*?)</supplementary-material>", re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(text):
        block = match.group(0)
        href_match = re.search(r"(?:xlink:href|href)=[\"']([^\"']+)[\"']", block, re.IGNORECASE)
        marker_text = " ".join([
            match.group("open"),
            " ".join(re.findall(r"<date\b[^>]*>.*?</date>", block, re.IGNORECASE | re.DOTALL)),
            " ".join(re.findall(r"<version\b[^>]*>.*?</version>", block, re.IGNORECASE | re.DOTALL)),
            " ".join(re.findall(r"<\?(?:suppdata-)?(?:date|created|version)[^?]*\?>", block, re.IGNORECASE)),
        ])
        complete_dates = ["-".join(values) for values in DATE_PATTERN.findall(marker_text)]
        entries.append({
            "href": href_match.group(1) if href_match else "",
            "marker_text": marker_text,
            "complete_dates": "|".join(complete_dates),
            "content_hash": hashlib.sha256(block.encode()).hexdigest(),
        })
    return entries


def scan_provenance(cache: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    diagnostics = {}
    for origin_text in ORIGINS:
        origin = pd.Timestamp(origin_text)
        records = pd.read_json(cache / "literature_v3" / "parsed" / f"{origin_text}.jsonl", lines=True)
        records = records[
            records["date_eligible"].astype(bool)
            & records["pmcid"].fillna("").astype(str).ne("")
        ].drop_duplicates(["queried_nct_id", "pmcid", "publication_identity"])
        articles_with_cached_xml = 0
        articles_with_supplements = 0
        packages = 0
        provable = 0
        provable_trials = set()
        for _, record in records.iterrows():
            pmcid = str(record["pmcid"]).upper()
            payload = _xml_payload(cache, pmcid)
            if not payload:
                continue
            articles_with_cached_xml += 1
            entries = _supplement_entries(payload)
            if entries:
                articles_with_supplements += 1
            for entry in entries:
                packages += 1
                dates = [pd.Timestamp(value) for value in entry["complete_dates"].split("|") if value]
                publication_date = pd.to_datetime(record.get("publication_date"), errors="coerce")
                document_version_date = pd.to_datetime(record.get("document_version_date"), errors="coerce")
                proven = bool(
                    dates
                    and all(value < origin for value in dates)
                    and pd.notna(publication_date)
                    and publication_date < origin
                    and (pd.isna(document_version_date) or document_version_date < origin)
                )
                if proven:
                    provable += 1
                    provable_trials.add(str(record["queried_nct_id"]))
                rows.append({
                    "origin": origin_text,
                    "queried_nct_id": str(record["queried_nct_id"]),
                    "pmcid": pmcid,
                    "publication_identity": str(record["publication_identity"]),
                    "publication_date": str(record.get("publication_date", "")),
                    "document_version_date": str(record.get("document_version_date", "")),
                    "href": entry["href"],
                    "package_content_hash": entry["content_hash"],
                    "package_version_marker": entry["marker_text"],
                    "package_complete_dates": entry["complete_dates"],
                    "historical_availability_proven": proven,
                })
        diagnostics[origin_text] = {
            "candidate_articles": int(len(records)),
            "cached_full_text_articles": articles_with_cached_xml,
            "articles_declaring_supplements": articles_with_supplements,
            "declared_packages": packages,
            "historically_provable_packages": provable,
            "historically_provable_trials": len(provable_trials),
        }
        report("provenance", origin=origin_text, diagnostics=json.dumps(diagnostics[origin_text], sort_keys=True))
    return pd.DataFrame(rows), diagnostics


# Retrieval

def retrieve_package(cache: Path, row: pd.Series) -> tuple[Path, dict[str, Any]]:
    if not bool(row["historical_availability_proven"]):
        raise RuntimeError("Unproven current supplement attachment cannot be retrieved")
    origin = str(row["origin"])
    pmcid = str(row["pmcid"])
    marker_hash = hashlib.sha256(str(row["package_version_marker"]).encode()).hexdigest()
    key = hashlib.sha256(f"{VERSION}\0{pmcid}\0{marker_hash}\0{row['package_content_hash']}\0{origin}".encode()).hexdigest()
    root = cache / VERSION / "packages" / pmcid
    path = root / f"{key}.bin"
    metadata_path = root / f"{key}.json"
    if path.exists() and metadata_path.exists():
        return path, json.loads(metadata_path.read_text())
    request = urllib.request.Request(
        f"{EUROPE_PMC_REST}/{pmcid}/supplementaryFiles",
        headers={"User-Agent": "kapso-relbench-historical-supplements/1.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = response.read()
        headers = dict(response.headers.items())
    root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".bin.part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    metadata = {
        "url": request.full_url,
        "origin": origin,
        "pmcid": pmcid,
        "package_version_marker": str(row["package_version_marker"]),
        "article_package_content_hash": str(row["package_content_hash"]),
        "retrieved_content_sha256": hashlib.sha256(payload).hexdigest(),
        "headers": headers,
        "historical_availability_proven": True,
    }
    _atomic_json(metadata_path, metadata)
    return path, metadata


# Parsing

def _flatten_table(frame: pd.DataFrame, source: str) -> list[dict[str, str]]:
    frame = frame.fillna("").astype(str)
    return [{"source": source, "text": " | ".join(row)} for row in frame.to_numpy().tolist() if any(row)]


def _parse_xml(payload: bytes, source: str) -> list[dict[str, str]]:
    root = ElementTree.fromstring(payload)
    rows = []
    for table in root.findall(".//table"):
        for row in table.findall(".//tr"):
            cells = [" ".join(cell.itertext()).strip() for cell in list(row)]
            if any(cells):
                rows.append({"source": source, "text": " | ".join(cells)})
    return rows


def _parse_html(payload: bytes, source: str) -> list[dict[str, str]]:
    return [row for frame in pd.read_html(io.BytesIO(payload)) for row in _flatten_table(frame, source)]


def _parse_csv(payload: bytes, source: str) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig", errors="replace")
    return [{"source": source, "text": " | ".join(row)} for row in csv.reader(io.StringIO(text)) if any(row)]


def _parse_xlsx(payload: bytes, source: str) -> list[dict[str, str]]:
    book = pd.ExcelFile(io.BytesIO(payload))
    return [row for sheet in book.sheet_names for row in _flatten_table(pd.read_excel(book, sheet_name=sheet), f"{source}:{sheet}")]


def _parse_docx(payload: bytes, source: str) -> list[dict[str, str]]:
    from docx import Document

    document = Document(io.BytesIO(payload))
    rows = []
    for table in document.tables:
        for row in table.rows:
            values = [cell.text.strip() for cell in row.cells]
            if any(values):
                rows.append({"source": source, "text": " | ".join(values)})
    return rows


def _parse_pdf(path: Path, source: str) -> list[dict[str, str]]:
    import camelot

    return [row for table in camelot.read_pdf(str(path), pages="all") for row in _flatten_table(table.df, source)]


def parse_package(path: Path, destination: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = path.read_bytes()
    files = []
    if zipfile.is_zipfile(io.BytesIO(payload)):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for name in archive.namelist():
                if not name.endswith("/"):
                    files.append((name, archive.read(name)))
    else:
        files.append((path.name, payload))
    rows = []
    failures = []
    parsed_files = 0
    for name, content in files:
        suffix = Path(name).suffix.casefold()
        try:
            if suffix in [".xml", ".jats"]:
                current = _parse_xml(content, name)
            elif suffix in [".html", ".htm"]:
                current = _parse_html(content, name)
            elif suffix in [".csv", ".tsv"]:
                current = _parse_csv(content, name)
            elif suffix in [".xlsx", ".xls"]:
                current = _parse_xlsx(content, name)
            elif suffix == ".docx":
                current = _parse_docx(content, name)
            elif suffix == ".pdf":
                temporary = destination / f"{hashlib.sha256(content).hexdigest()}.pdf"
                temporary.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_bytes(content)
                current = _parse_pdf(temporary, name)
            else:
                continue
            parsed_files += 1
            rows.extend(current)
        except Exception as error:
            failures.append({"file": name, "error": f"{type(error).__name__}:{error}"})
    frame = pd.DataFrame(rows, columns=["source", "text"])
    diagnostics = {"files": len(files), "parsed_files": parsed_files, "table_rows": len(frame), "failures": failures}
    return frame, diagnostics


# Endpoint matching

def normalize_endpoint(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def phrase_overlap(endpoint: str, row: str) -> float:
    endpoint_tokens = set(normalize_endpoint(endpoint).split())
    row_tokens = set(normalize_endpoint(row).split())
    return float(len(endpoint_tokens & row_tokens) / max(1, len(endpoint_tokens)))


def extract_statistics(text: str) -> dict[str, Any]:
    p_values = [{"operator": match.group(1), "value": float(match.group(2))} for match in P_VALUE_PATTERN.finditer(text)]
    intervals = [{"lower": float(match.group(1)), "upper": float(match.group(2))} for match in CI_PATTERN.finditer(text)]
    return {"p_values": p_values, "confidence_intervals": intervals, "explicit_non_significance": bool(re.search(r"\b(?:not significant|non-significant|nonsignificant|did not reach significance)\b", text, re.IGNORECASE))}


def endpoint_similarities(primary_endpoints: list[str], table_rows: pd.DataFrame) -> np.ndarray:
    if not primary_endpoints or table_rows.empty:
        return np.empty((len(primary_endpoints), len(table_rows)), dtype=np.float32)
    import torch
    from transformers import AutoModel, AutoTokenizer
    from medcpt_reranker import ARTICLE_MODEL, BATCH_SIZE, QUERY_MODEL, _device, _encode_texts

    device = _device()
    query_tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL)
    query_model = AutoModel.from_pretrained(QUERY_MODEL)
    article_tokenizer = AutoTokenizer.from_pretrained(ARTICLE_MODEL)
    article_model = AutoModel.from_pretrained(ARTICLE_MODEL)
    query_embeddings = _encode_texts(query_model, query_tokenizer, primary_endpoints, 256, BATCH_SIZE, device)
    article_pairs = [[str(value), ""] for value in table_rows["text"].astype(str)]
    article_embeddings = _encode_texts(article_model, article_tokenizer, article_pairs, 512, BATCH_SIZE, device)
    query_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True).clip(min=1e-8)
    article_norm = np.linalg.norm(article_embeddings, axis=1, keepdims=True).clip(min=1e-8)
    similarities = (query_embeddings / query_norm) @ (article_embeddings / article_norm).T
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return similarities.astype(np.float32)


def deterministic_verdict(table_rows: pd.DataFrame, primary_endpoints: list[str], similarities: np.ndarray | None = None) -> dict[str, Any]:
    matched = []
    for endpoint_index, endpoint in enumerate(primary_endpoints):
        candidates = []
        for row_index, text in enumerate(table_rows.get("text", pd.Series(dtype=str)).astype(str)):
            lexical = phrase_overlap(endpoint, text)
            semantic = float(similarities[endpoint_index, row_index]) if similarities is not None else 0.0
            if lexical > 0 and semantic >= ENDPOINT_MATCH_THRESHOLD:
                candidates.append((max(lexical, semantic), row_index, extract_statistics(text)))
        candidates.sort(reverse=True, key=lambda value: value[0])
        if candidates:
            matched.append(candidates[0])
    positive = any(any(value["operator"] in ["<", "<=", "="] and value["value"] <= 0.05 for value in statistics["p_values"]) for _, _, statistics in matched)
    negative = bool(
        primary_endpoints
        and len(matched) == len(primary_endpoints)
        and all(
            statistics["explicit_non_significance"]
            or bool(statistics["p_values"] and all(value["value"] > 0.05 for value in statistics["p_values"]))
            for _, _, statistics in matched
        )
    )
    return {
        "positive": positive,
        "complete_negative": negative and not positive,
        "abstain": not positive and not negative,
        "matched_endpoints": len(matched),
        "registered_endpoints": len(primary_endpoints),
    }


# Orchestration

def run() -> dict[str, Any]:
    cache = shared_cache_dir()
    root = cache / VERSION
    provenance, by_origin = scan_provenance(cache)
    eligible = provenance[provenance["historical_availability_proven"].astype(bool)] if len(provenance) else provenance
    linkage = pd.read_parquet(cache / "registry_clock_lane0" / "features" / "registry_clock_features_v2" / "linkage.parquet")
    from publication_evidence import build_trial_contexts

    contexts = {
        origin: build_trial_contexts(linkage, pd.Timestamp(origin), cache / "registry_clock_lane0" / "projected")
        for origin in sorted(set(eligible["origin"]))
    } if len(eligible) else {}
    parsed_table_count = 0
    parsed_package_count = 0
    verdicts = []
    retrieval_failures = []
    for _, row in eligible.iterrows():
        try:
            package_path, package_metadata = retrieve_package(cache, row)
            table_rows, parsing = parse_package(package_path, root / "temporary_pdf")
            parsed_table_count += int(parsing["table_rows"])
            parsed_package_count += int(parsing["parsed_files"] > 0)
            context = contexts[str(row["origin"])][str(row["queried_nct_id"])]
            endpoints = [f"{item.get('title', '')} {item.get('time_frame', '')}".strip() for item in context.get("primary_outcomes", [])]
            similarities = endpoint_similarities(endpoints, table_rows)
            verdict = deterministic_verdict(table_rows, endpoints, similarities)
            verdicts.append({"origin": row["origin"], "queried_nct_id": row["queried_nct_id"], "pmcid": row["pmcid"], "package": package_metadata, "parsing": parsing, "verdict": verdict})
        except Exception as error:
            retrieval_failures.append({"pmcid": row["pmcid"], "origin": row["origin"], "error": f"{type(error).__name__}:{error}"})
    unique_new_verdicts = len({(item["origin"], item["queried_nct_id"]) for item in verdicts if not item["verdict"]["abstain"]})
    diagnostics = {
        "version": VERSION,
        "by_origin": by_origin,
        "declared_package_rows": int(len(provenance)),
        "eligible_package_count": int(len(eligible)),
        "parsed_package_count": parsed_package_count,
        "parsed_table_count": parsed_table_count,
        "unique_new_verdicts": unique_new_verdicts,
        "retrieval_failures": retrieval_failures,
        "funded": bool(len(eligible) > 0),
        "accepted": False,
        "reason": "no_package_specific_pre_origin_date_or_version_marker" if len(eligible) == 0 else "no_gated_unique_verdicts",
        "endpoint_match_threshold": ENDPOINT_MATCH_THRESHOLD,
        "elapsed_seconds": time.time() - START,
    }
    root.mkdir(parents=True, exist_ok=True)
    provenance.to_parquet(root / "provenance_recon.parquet", index=False)
    _atomic_json(root / "diagnostics.json", diagnostics)
    register_artifact(cache, {
        "name": "generic_exp_2 historical supplementary-table recon",
        "path": VERSION,
        "description": "Europe PMC supplementaryFiles provenance gate plus deterministic JATS/XML/HTML/CSV/XLSX/DOCX/PDF table parsers; current attachments without package-specific pre-origin proof are rejected.",
        "content_key": f"rel-trial-study-outcome:{VERSION}:package-date-required",
        "rebuild_hint": "Run supplement_pipeline.py after literature retrieval; only rows with a package-specific complete pre-origin marker trigger network retrieval.",
    })
    marker = root / "campaign_memory_recorded"
    if not marker.exists():
        locked_append(cache / "features_history.md", f'''\n### Historically provable supplementary tables
- run/experiment: generic_exp_2 lane 0 | status: TESTED-REJECTED
- what: Europe PMC supplementaryFiles packages admitted only with package-specific complete pre-origin provenance, followed by deterministic structured-table parsing and endpoint matching.
- outcome: {json.dumps(diagnostics, sort_keys=True)}.
- takeaway: cached JATS declared supplements but exposed no package-specific date/version marker; current attachments are unverifiable and were rejected before retrieval, so the ASSUMED supplement gain was not funded.
''')
        locked_append(cache / "table_information.md", f'''\n### 2026-08-14 supplement provenance recon
- Europe PMC JATS records can declare supplementary-material hrefs and content hashes without a package-specific creation/version date. Article publication age alone does not prove a current attachment existed at the seed origin.
- Provenance funnel: {json.dumps(by_origin, sort_keys=True)}. Unverifiable attachments were rejected before bytes or endpoint statistics entered a feature.
''')
        marker.write_text("recorded\n")
    report("complete", diagnostics=json.dumps(diagnostics, sort_keys=True))
    return diagnostics


if __name__ == "__main__":
    run()
