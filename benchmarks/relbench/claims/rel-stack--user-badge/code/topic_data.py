from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from temporal_features import database_root, load_users

duckdb.execute(f"set threads={int(os.environ.get('OMP_NUM_THREADS', '11'))}")


# Cache

def topic_root() -> Path:
    root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / "lane3_dormant_content_v1"
    root.mkdir(parents=True, exist_ok=True)
    return root


def register_topic_artifact(name: str, path: Path, description: str, content_key: str) -> None:
    shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    registry = shared / "artifacts.json"
    lock_path = shared / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        records = json.loads(registry.read_text()) if registry.exists() else []
        if not any(record.get("content_key") == content_key for record in records):
            records.append(
                {
                    "name": name,
                    "path": str(path.relative_to(shared)),
                    "description": description,
                    "content_key": content_key,
                    "rebuild_hint": "Run main.py to extend lane3_dormant_content_v1.",
                }
            )
            temporary = registry.with_suffix(".tmp.lane3")
            temporary.write_text(json.dumps(records, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def state_root(origins: np.ndarray) -> Path:
    digest = hashlib.sha256(np.asarray(origins, dtype=np.int64).tobytes()).hexdigest()[:16]
    root = topic_root() / f"states_{digest}"
    root.mkdir(parents=True, exist_ok=True)
    return root


# Structures

@dataclass
class QuestionData:
    origins: np.ndarray
    question_ids: np.ndarray
    owners: np.ndarray
    creation: np.ndarray
    state_index: np.ndarray
    state_latin: np.ndarray
    texts: list[str]
    hashes: np.ndarray
    tag_names: list[str]
    pairs_root: Path
    events: dict[str, tuple[np.ndarray, np.ndarray]]

    def pairs(self, origin_index: int) -> tuple[np.ndarray, np.ndarray]:
        question = np.load(self.pairs_root / f"question_{origin_index:02d}.npy", mmap_mode="r")
        tag = np.load(self.pairs_root / f"tag_{origin_index:02d}.npy", mmap_mode="r")
        return question, tag


# Mapping

def _map_sorted(values: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = np.searchsorted(keys, values)
    clipped = np.minimum(position, max(0, len(keys) - 1))
    valid = (position < len(keys)) & (keys[clipped] == values)
    return position.astype(np.int32), valid


def _question_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    db = database_root()
    posts = duckdb.sql(
        f"select Id,OwnerUserId,PostTypeId,ParentId,CreationDate from read_parquet('{db / 'posts.parquet'}') order by Id"
    ).df()
    post_ids = posts["Id"].to_numpy(dtype=np.int64)
    post_type = posts["PostTypeId"].to_numpy(dtype=np.int16)
    question_mask = post_type == 1
    question_ids = post_ids[question_mask]
    question_creation = posts.loc[question_mask, "CreationDate"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    users = load_users()
    user_ids = users["Id"].to_numpy(dtype=np.int64)
    owner_raw = posts.loc[question_mask, "OwnerUserId"].fillna(-1).to_numpy(dtype=np.int64)
    owners, owner_valid = _map_sorted(np.maximum(owner_raw, 0), user_ids)
    owners[~owner_valid | (owner_raw < 0)] = -1
    root_question = np.full(len(posts), -1, dtype=np.int32)
    root_question[question_mask] = np.arange(len(question_ids), dtype=np.int32)
    answer_mask = (post_type == 2) & posts["ParentId"].notna().to_numpy()
    parent_raw = posts.loc[answer_mask, "ParentId"].to_numpy(dtype=np.int64)
    parent, parent_valid = _map_sorted(parent_raw, question_ids)
    answer_rows = np.flatnonzero(answer_mask)
    root_question[answer_rows[parent_valid]] = parent[parent_valid]
    return question_ids, owners, question_creation, post_ids, root_question


# Events

def _save_event(root: Path, name: str, times: np.ndarray, questions: np.ndarray) -> None:
    order = np.argsort(times, kind="stable")
    np.save(root / f"{name}_time.npy", np.asarray(times[order], dtype=np.int64))
    np.save(root / f"{name}_question.npy", np.asarray(questions[order], dtype=np.int32))


def build_question_events(post_ids: np.ndarray, root_question: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    root = topic_root() / "question_events_v1"
    names = ("votes", "comments", "answers", "links")
    paths = [root / f"{name}_{kind}.npy" for name in names for kind in ("time", "question")]
    if not all(path.exists() for path in paths):
        root.mkdir(parents=True, exist_ok=True)
        db = database_root()
        for name, table, time_column in (
            ("votes", "votes", "CreationDate"),
            ("comments", "comments", "CreationDate"),
        ):
            frame = duckdb.sql(
                f"select PostId,{time_column} ts from read_parquet('{db / f'{table}.parquet'}') where PostId is not null"
            ).df()
            mapped, valid = _map_sorted(frame["PostId"].to_numpy(dtype=np.int64), post_ids)
            questions = np.full(len(frame), -1, dtype=np.int32)
            questions[valid] = root_question[mapped[valid]]
            keep = questions >= 0
            times = frame.loc[keep, "ts"].to_numpy(dtype="datetime64[s]").astype(np.int64)
            _save_event(root, name, times, questions[keep])
        answers = duckdb.sql(
            f"select Id,CreationDate ts from read_parquet('{db / 'posts.parquet'}') where PostTypeId=2 order by Id"
        ).df()
        mapped, valid = _map_sorted(answers["Id"].to_numpy(dtype=np.int64), post_ids)
        questions = np.full(len(answers), -1, dtype=np.int32)
        questions[valid] = root_question[mapped[valid]]
        keep = questions >= 0
        _save_event(
            root,
            "answers",
            answers.loc[keep, "ts"].to_numpy(dtype="datetime64[s]").astype(np.int64),
            questions[keep],
        )
        links = duckdb.sql(
            f"select PostId,RelatedPostId,CreationDate ts from read_parquet('{db / 'postLinks.parquet'}')"
        ).df()
        link_times = links["ts"].to_numpy(dtype="datetime64[s]").astype(np.int64)
        all_questions = []
        all_times = []
        for column in ("PostId", "RelatedPostId"):
            present = links[column].notna().to_numpy()
            raw = links.loc[present, column].to_numpy(dtype=np.int64)
            mapped, valid = _map_sorted(raw, post_ids)
            questions = np.full(len(raw), -1, dtype=np.int32)
            questions[valid] = root_question[mapped[valid]]
            keep = questions >= 0
            all_questions.append(questions[keep])
            all_times.append(link_times[present][keep])
        _save_event(root, "links", np.concatenate(all_times), np.concatenate(all_questions))
        register_topic_artifact(
            "lane3 root-question inbound events",
            root,
            "Votes, comments, answers, and both link endpoints routed to root questions with event timestamps.",
            "rel-stack-user-badge-lane3-question-events-v1",
        )
    return {
        name: (
            np.load(root / f"{name}_time.npy", mmap_mode="r"),
            np.load(root / f"{name}_question.npy", mmap_mode="r"),
        )
        for name in names
    }


# Historical state

def _normalize_text(title: str, tags: str) -> str:
    cleaned_title = " ".join(str(title).split())
    parsed_tags = " ".join(re.findall(r"<([^<>]+)>", str(tags)))
    return f"{cleaned_title} [SEP] {parsed_tags}".strip()


def _latin_state(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    latin = sum(ord(character) < 768 for character in letters)
    return latin / len(letters) >= 0.8


def _load_state(root: Path, question_ids: np.ndarray, owners: np.ndarray, creation: np.ndarray, events: dict[str, tuple[np.ndarray, np.ndarray]], origins: np.ndarray) -> QuestionData:
    texts = json.loads((root / "texts.json").read_text())
    tag_names = json.loads((root / "tags.json").read_text())
    return QuestionData(
        origins=origins,
        question_ids=question_ids,
        owners=owners,
        creation=creation,
        state_index=np.load(root / "state_index.npy", mmap_mode="r"),
        state_latin=np.load(root / "state_latin.npy", mmap_mode="r"),
        texts=texts,
        hashes=np.load(root / "hashes.npy", mmap_mode="r"),
        tag_names=tag_names,
        pairs_root=root / "pairs",
        events=events,
    )


def build_historical_states(origins: np.ndarray) -> QuestionData:
    origins = np.asarray(origins, dtype=np.int64)
    root = state_root(origins)
    question_ids, owners, creation, post_ids, root_question = _question_tables()
    events = build_question_events(post_ids, root_question)
    expected = [root / "state_index.npy", root / "state_latin.npy", root / "texts.json", root / "tags.json", root / "hashes.npy"]
    expected += [root / "pairs" / f"question_{index:02d}.npy" for index in range(len(origins))]
    expected += [root / "pairs" / f"tag_{index:02d}.npy" for index in range(len(origins))]
    if all(path.exists() for path in expected):
        return _load_state(root, question_ids, owners, creation, events, origins)
    started = time.time()
    db = database_root()
    history = duckdb.sql(
        f"""
        select h.PostId,h.PostHistoryTypeId,h.Text,h.CreationDate,h.Id
        from read_parquet('{db / 'postHistory.parquet'}') h
        join read_parquet('{db / 'posts.parquet'}') p on p.Id=h.PostId
        where p.PostTypeId=1 and h.PostHistoryTypeId in (1,3,4,6)
        order by h.CreationDate,h.Id
        """
    ).df()
    mapped, valid = _map_sorted(history["PostId"].to_numpy(dtype=np.int64), question_ids)
    history = history.loc[valid].reset_index(drop=True)
    history_question = mapped[valid]
    history_time = history["CreationDate"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    history_type = history["PostHistoryTypeId"].to_numpy(dtype=np.int8)
    history_text = history["Text"].fillna("").astype(str).to_numpy()
    pairs = root / "pairs"
    pairs.mkdir(parents=True, exist_ok=True)
    state_matrix = open_memmap(root / "state_index.npy", mode="w+", dtype=np.int32, shape=(len(origins), len(question_ids)))
    state_matrix[:] = -1
    latin_matrix = open_memmap(root / "state_latin.npy", mode="w+", dtype=np.uint8, shape=(len(origins), len(question_ids)))
    latin_matrix[:] = 0
    title = np.full(len(question_ids), None, dtype=object)
    tags = np.full(len(question_ids), None, dtype=object)
    current_state = np.full(len(question_ids), -1, dtype=np.int32)
    current_latin = np.zeros(len(question_ids), dtype=np.uint8)
    texts: list[str] = []
    hashes: list[str] = []
    hash_to_index: dict[str, int] = {}
    tag_names: list[str] = []
    tag_to_index: dict[str, int] = {}
    previous = 0
    for origin_index, cutoff in enumerate(origins):
        current = int(np.searchsorted(history_time, cutoff, side="right"))
        changed = np.unique(history_question[previous:current])
        for row in range(previous, current):
            question = int(history_question[row])
            if history_type[row] in (1, 4):
                title[question] = history_text[row]
            else:
                tags[question] = history_text[row]
        for question in changed:
            if title[question] is None or tags[question] is None:
                current_state[question] = -1
                current_latin[question] = 0
                continue
            normalized = _normalize_text(str(title[question]), str(tags[question]))
            digest = hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()
            if digest not in hash_to_index:
                hash_to_index[digest] = len(texts)
                hashes.append(digest)
                texts.append(normalized)
            current_state[question] = hash_to_index[digest]
            current_latin[question] = int(_latin_state(normalized))
        previous = current
        active = (creation <= cutoff) & (owners >= 0)
        state_matrix[origin_index] = np.where(active, current_state, -1)
        latin_matrix[origin_index] = np.where(active, current_latin, 0)
        active_questions = np.flatnonzero(active & np.fromiter((value is not None for value in tags), dtype=np.bool_, count=len(tags)))
        question_parts = []
        tag_parts = []
        for question in active_questions:
            values = sorted(set(re.findall(r"<([^<>]+)>", str(tags[question]).lower())))
            if not values:
                continue
            indices = []
            for value in values:
                if value not in tag_to_index:
                    tag_to_index[value] = len(tag_names)
                    tag_names.append(value)
                indices.append(tag_to_index[value])
            question_parts.append(np.full(len(indices), question, dtype=np.int32))
            tag_parts.append(np.asarray(indices, dtype=np.int32))
        pair_question = np.concatenate(question_parts) if question_parts else np.empty(0, dtype=np.int32)
        pair_tag = np.concatenate(tag_parts) if tag_parts else np.empty(0, dtype=np.int32)
        np.save(pairs / f"question_{origin_index:02d}.npy", pair_question)
        np.save(pairs / f"tag_{origin_index:02d}.npy", pair_tag)
        state_matrix.flush()
        latin_matrix.flush()
        print(
            f"[historical-state] origin={pd.to_datetime(cutoff, unit='s').date()} questions={int(active.sum())} pairs={len(pair_question)} states={len(texts)} elapsed_seconds={time.time() - started:.1f}",
            flush=True,
        )
    (root / "texts.json").write_text(json.dumps(texts, ensure_ascii=False))
    (root / "tags.json").write_text(json.dumps(tag_names, ensure_ascii=False))
    np.save(root / "hashes.npy", np.asarray(hashes, dtype="S64"))
    register_topic_artifact(
        "lane3 cutoff-valid question title and tag states",
        root,
        "Post-history reconstructed title/tag snapshots, normalized hashes, language route, and question-tag pairs for every origin.",
        f"rel-stack-user-badge-lane3-historical-state-{root.name}",
    )
    return _load_state(root, question_ids, owners, creation, events, origins)


# Embeddings

def build_embeddings(data: QuestionData, debug: bool) -> np.ndarray:
    mode = "debug" if debug else "full"
    path = state_root(data.origins) / f"minilm_embeddings_{mode}.npy"
    complete = path.with_suffix(".complete")
    if path.exists() and complete.exists():
        matrix = np.load(path, mmap_mode="r")
        if matrix.shape == (len(data.texts), 384):
            return matrix
    limit = min(len(data.texts), 5000) if debug else len(data.texts)
    matrix = open_memmap(path, mode="w+", dtype=np.float16, shape=(len(data.texts), 384))
    matrix[:] = 0
    if limit:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
        local_model = topic_root() / "models" / "all-MiniLM-L6-v2"
        model_source = str(local_model) if (local_model / "config.json").exists() else "sentence-transformers/all-MiniLM-L6-v2"
        if (local_model / "config.json").exists():
            register_topic_artifact(
                "all-MiniLM-L6-v2 pretrained encoder",
                local_model,
                "Public 384-dimensional Sentence Transformers checkpoint used for cutoff-valid question text.",
                "sentence-transformers-all-MiniLM-L6-v2-snapshot-v1",
            )
        model = SentenceTransformer(
            model_source,
            device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            cache_folder=str(topic_root() / "models"),
        )
        model.max_seq_length = 128
        if __import__("torch").cuda.is_available():
            model.half()
        states = np.asarray(data.state_index)
        latin = np.asarray(data.state_latin) > 0
        selected = np.unique(states[(states >= 0) & latin])
        selected = selected[selected < limit]
        started = time.time()
        encoded = model.encode(
            [data.texts[index] for index in selected],
            batch_size=512,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        matrix[selected] = encoded.astype(np.float16)
        print(f"[embedding] states={len(selected)} rate={len(selected) / max(time.time() - started, 1e-6):.1f}_states_per_s", flush=True)
    matrix.flush()
    complete.write_text("complete\n")
    register_topic_artifact(
        f"lane3 MiniLM historical text embeddings {mode}",
        path,
        "FP16 384-dimensional all-MiniLM-L6-v2 embeddings keyed by normalized cutoff-valid title/tag hashes.",
        f"rel-stack-user-badge-lane3-minilm-{state_root(data.origins).name}-{mode}",
    )
    return np.load(path, mmap_mode="r")
