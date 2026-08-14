from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PARSER_VERSION = "registry_snapshot_v2"

SNAPSHOT_URLS = {
    "2017-06-13": "https://aact.ctti-clinicaltrials.org/static/static_db_copies/daily/2017-06-13?source=web",
    "2017-12-17": "https://aact.ctti-clinicaltrials.org/static/static_db_copies/daily/2017-12-17?source=web",
    "2018-06-01": "https://aact.ctti-clinicaltrials.org/static/static_db_copies/daily/2018-06-01?source=web",
    "2018-12-01": "https://aact.ctti-clinicaltrials.org/static/static_db_copies/daily/2018-12-01?source=web",
    "2019-12-01": "https://aact.ctti-clinicaltrials.org/static/static_db_copies/daily/2019-12-01?source=web",
    "2020-12-01": "https://aact.ctti-clinicaltrials.org/static/static_db_copies/daily/2020-12-01?source=web",
}

SNAPSHOT_SHA256 = {
    "2017-06-13": "6dff60cfe80684157c2c072238945851c8cfb3f29e4a0a07c1d893c26353969c",
    "2017-12-17": "866a9d38df183788fb20db57bae0ddef83c2963a7bf95b2fd15011189af9a42b",
    "2018-06-01": "f13182b5a8cb42c3e8700dbd8d1e7f59e1b33e0c07c14b896f623ad402523d69",
    "2018-12-01": "8590ca2c0767957b2fc7b1bf8283c18891aa15ef7288c62cf54755ca153a2212",
    "2019-12-01": "c56c06707fa77786229f3c745771bbc747ba304a32601085988bedb9c5442909",
    "2020-12-01": "d44da0ac916022efb876089dc7e8a7bb698e8d0db72211be184bc90426868405",
}

TABLE_COLUMNS = {
    "studies": [
        "nct_id", "study_first_submitted_date", "results_first_submitted_date",
        "disposition_first_submitted_date", "last_update_submitted_date",
        "study_first_submitted_qc_date", "study_first_posted_date",
        "results_first_submitted_qc_date", "results_first_posted_date",
        "disposition_first_submitted_qc_date", "disposition_first_posted_date",
        "last_update_submitted_qc_date", "last_update_posted_date", "start_date",
        "start_date_type", "verification_date", "completion_date", "completion_date_type",
        "primary_completion_date", "primary_completion_date_type", "study_type",
        "brief_title", "official_title", "overall_status", "last_known_status", "phase",
        "enrollment", "enrollment_type", "source", "number_of_arms", "number_of_groups",
        "why_stopped", "has_dmc", "is_fda_regulated_drug", "is_fda_regulated_device",
        "created_at", "updated_at",
    ],
    "calculated_values": [
        "nct_id", "number_of_facilities", "registered_in_calendar_year", "nlm_download_date",
        "actual_duration", "were_results_reported", "months_to_report_results",
        "has_us_facility", "has_single_facility", "number_of_primary_outcomes_to_measure",
        "number_of_secondary_outcomes_to_measure", "number_of_other_outcomes_to_measure",
    ],
    "designs": [
        "nct_id", "allocation", "intervention_model", "observational_model", "primary_purpose",
        "time_perspective", "masking", "subject_masked", "caregiver_masked",
        "investigator_masked", "outcomes_assessor_masked",
    ],
    "eligibilities": [
        "nct_id", "sampling_method", "gender", "minimum_age", "maximum_age",
        "healthy_volunteers", "gender_based",
    ],
    "facilities": ["nct_id", "status", "name", "city", "state", "country"],
    "countries": ["nct_id", "name", "removed"],
    "sponsors": ["nct_id", "agency_class", "lead_or_collaborator", "name"],
    "browse_conditions": ["nct_id", "mesh_term", "downcase_mesh_term"],
    "browse_interventions": ["nct_id", "mesh_term", "downcase_mesh_term"],
    "pending_results": ["nct_id", "event", "event_date_description", "event_date"],
    "documents": ["nct_id", "document_id", "document_type", "url", "comment"],
    "design_outcomes": ["nct_id", "outcome_type", "measure", "time_frame", "population"],
    "study_references": ["nct_id", "pmid", "reference_type", "citation"],
    "outcomes": ["id", "nct_id", "outcome_type", "title", "time_frame", "anticipated_posting_date"],
    "outcome_analyses": ["nct_id", "outcome_id", "p_value_modifier", "p_value", "method"],
}

COLUMN_ALIASES = {
    "studies": {
        "study_first_submitted_date": "first_received_date",
        "results_first_submitted_date": "first_received_results_date",
        "disposition_first_submitted_date": "received_results_disposit_date",
        "last_update_submitted_date": "last_changed_date",
    }
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_copy_value(value: str) -> str | None:
    if value == r"\N":
        return None
    replacements = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v", "\\": "\\"}
    result = []
    index = 0
    while index < len(value):
        if value[index] != "\\" or index + 1 >= len(value):
            result.append(value[index])
            index += 1
            continue
        following = value[index + 1]
        if following in replacements:
            result.append(replacements[following])
            index += 2
            continue
        octal = re.match(r"[0-7]{1,3}", value[index + 1:])
        if octal:
            result.append(chr(int(octal.group(0), 8)))
            index += 1 + len(octal.group(0))
            continue
        result.append(following)
        index += 2
    return "".join(result)


def project_table(dump_path: Path, table_name: str, destination: Path, chunk_size: int = 100000) -> int:
    selected_columns = TABLE_COLUMNS[table_name]
    process = subprocess.Popen(
        ["pg_restore", "-a", "-t", table_name, "-f", "-", str(dump_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        bufsize=1024 * 1024,
    )
    copy_columns = None
    selected_indices = None
    rows = {column: [] for column in selected_columns}
    writer = None
    row_count = 0

    def flush() -> None:
        nonlocal writer
        if not rows[selected_columns[0]]:
            return
        table = pa.Table.from_pydict(rows, schema=pa.schema([(column, pa.string()) for column in selected_columns]))
        if writer is None:
            writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
        writer.write_table(table)
        for column in selected_columns:
            rows[column].clear()

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        if copy_columns is None:
            match = re.match(rf"COPY\s+(?:[^.\s]+\.)?{re.escape(table_name)}\s+\((.+)\)\s+FROM stdin;", line)
            if match:
                copy_columns = [value.strip().strip('"') for value in match.group(1).split(",")]
                aliases = COLUMN_ALIASES.get(table_name, {})
                selected_indices = [
                    copy_columns.index(column) if column in copy_columns
                    else copy_columns.index(aliases[column]) if aliases.get(column) in copy_columns
                    else None
                    for column in selected_columns
                ]
            continue
        if line == r"\.":
            break
        values = line.split("\t")
        if len(values) != len(copy_columns):
            raise RuntimeError(f"{table_name} COPY width {len(values)} != {len(copy_columns)}")
        assert selected_indices is not None
        for column, column_index in zip(selected_columns, selected_indices):
            rows[column].append(decode_copy_value(values[column_index]) if column_index is not None else None)
        row_count += 1
        if len(rows[selected_columns[0]]) >= chunk_size:
            flush()
    flush()
    if writer is not None:
        writer.close()
    else:
        empty = pa.Table.from_pydict({column: [] for column in selected_columns}, schema=pa.schema([(column, pa.string()) for column in selected_columns]))
        pq.write_table(empty, destination, compression="zstd")
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"pg_restore failed for {table_name}: {stderr[-1000:]}")
    return row_count


def project_snapshot(dump_path: Path, snapshot_date: str, output_root: Path) -> dict[str, object]:
    destination = output_root / snapshot_date
    metadata_path = destination / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("parser_version") == PARSER_VERSION and metadata.get("dump_sha256") == file_sha256(dump_path):
            return metadata
    destination.mkdir(parents=True, exist_ok=True)
    started = time.time()
    table_rows = {}
    for table_name in TABLE_COLUMNS:
        temporary = destination / f"{table_name}.parquet.part"
        final_path = destination / f"{table_name}.parquet"
        table_rows[table_name] = project_table(dump_path, table_name, temporary)
        os.replace(temporary, final_path)
    metadata = {
        "snapshot_date": snapshot_date,
        "maximum_usable_timestamp": f"{snapshot_date}T00:00:00",
        "url": SNAPSHOT_URLS[snapshot_date],
        "dump_sha256": file_sha256(dump_path),
        "parser_version": PARSER_VERSION,
        "table_rows": table_rows,
        "elapsed_seconds": time.time() - started,
    }
    temporary_metadata = metadata_path.with_suffix(".json.part")
    temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_metadata, metadata_path)
    return metadata


# Snapshot materialization

def ensure_snapshots(output_root: Path, download_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)
    diagnostics = {}
    for snapshot_date, url in SNAPSHOT_URLS.items():
        destination = output_root / snapshot_date
        metadata_path = destination / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("archive_sha256") == SNAPSHOT_SHA256[snapshot_date] and metadata.get("parser_version") == PARSER_VERSION:
                diagnostics[snapshot_date] = {"state": "projected_cache", "sha256": metadata["archive_sha256"]}
                continue
        dump_path = download_root / f"{snapshot_date}.dmp"
        current_hash = file_sha256(dump_path) if dump_path.exists() else ""
        if current_hash != SNAPSHOT_SHA256[snapshot_date]:
            if dump_path.exists() and dump_path.stat().st_size > 0:
                command = ["curl", "--location", "--silent", "--show-error", "--fail", "--retry", "6", "--continue-at", "-", "--output", str(dump_path), url]
            else:
                command = ["curl", "--location", "--silent", "--show-error", "--fail", "--retry", "6", "--output", str(dump_path), url]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                if dump_path.exists():
                    dump_path.unlink()
                completed = subprocess.run(["curl", "--location", "--silent", "--show-error", "--fail", "--retry", "6", "--output", str(dump_path), url], capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(f"Snapshot download failed for {snapshot_date}: {completed.stderr[-1000:]}")
            current_hash = file_sha256(dump_path)
        if current_hash != SNAPSHOT_SHA256[snapshot_date]:
            raise RuntimeError(f"Snapshot SHA-256 mismatch for {snapshot_date}: {current_hash}")
        projection_source = dump_path
        if zipfile.is_zipfile(dump_path):
            projection_source = download_root / f"{snapshot_date}.postgres_data.dmp"
            if not projection_source.exists():
                temporary_projection = projection_source.with_suffix(".dmp.part")
                with zipfile.ZipFile(dump_path) as archive:
                    with archive.open("postgres_data.dmp") as source, temporary_projection.open("wb") as destination_stream:
                        while True:
                            block = source.read(16 * 1024 * 1024)
                            if not block:
                                break
                            destination_stream.write(block)
                os.replace(temporary_projection, projection_source)
        metadata = project_snapshot(projection_source, snapshot_date, output_root)
        metadata["archive_sha256"] = current_hash
        metadata["archive_url"] = url
        metadata_path = output_root / snapshot_date / "metadata.json"
        temporary_metadata = metadata_path.with_suffix(".json.part")
        temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_metadata, metadata_path)
        diagnostics[snapshot_date] = {
            "state": "projected",
            "sha256": current_hash,
            "rows": metadata["table_rows"],
            "elapsed_seconds": metadata["elapsed_seconds"],
        }
    return diagnostics
