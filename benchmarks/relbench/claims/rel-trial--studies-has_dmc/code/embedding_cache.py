from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from clinical_features import DOCUMENT_VERSION, MODEL_ID, MODEL_REVISION, ClinicalBundle


def _cache_root(shared_root: Path, ids: np.ndarray) -> tuple[Path, str]:
    digest = hashlib.sha256()
    digest.update(np.asarray(ids, dtype=np.int64).tobytes())
    digest.update(MODEL_REVISION.encode())
    digest.update(DOCUMENT_VERSION.encode())
    content_key = digest.hexdigest()[:20]
    root = shared_root / f"lane2_medcpt_{content_key}"
    root.mkdir(parents=True, exist_ok=True)
    return root, content_key


def _register(shared_root: Path, root: Path, content_key: str) -> None:
    registry = shared_root / "artifacts.json"
    lock_path = shared_root / "artifacts.json.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            entries = json.loads(registry.read_text()) if registry.exists() else []
        except json.JSONDecodeError:
            entries = []
        name = f"lane2 MedCPT embeddings {content_key}"
        if not any(entry.get("name") == name for entry in entries):
            entries.append({
                "name": name,
                "path": str(root.relative_to(shared_root)),
                "description": "Pinned frozen MedCPT passage and mean embeddings in official task-row order",
                "content_key": content_key,
                "rebuild_hint": "Run main.py; missing passage memmaps are encoded and atomically promoted",
            })
            temporary = registry.with_suffix(".json.tmp.lane2")
            temporary.write_text(json.dumps(entries, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _encode_passages(bundle: ClinicalBundle, path: Path, passage: int) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import AutoModel, AutoTokenizer, logging

    logging.set_verbosity_error()
    logging.disable_progress_bar()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16).eval().cuda()
    dimensions = int(model.config.hidden_size)
    temporary = path.with_suffix(path.suffix + ".partial")
    matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=(len(bundle.ids), dimensions))
    second_key = ["abstract", "additional", "risk"][passage]
    batch_size = [128, 64, 128][passage]
    start = time.time()
    checkpoint = start
    with torch.inference_mode():
        for begin in range(0, len(bundle.ids), batch_size):
            end = min(begin + batch_size, len(bundle.ids))
            encoded = tokenizer(
                bundle.documents["title"][begin:end].tolist(),
                bundle.documents[second_key][begin:end].tolist(),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.cuda(non_blocking=True) for key, value in encoded.items()}
            output = model(**encoded).last_hidden_state[:, 0, :]
            output = torch.nn.functional.normalize(output.float(), dim=1)
            matrix[begin:end] = output.cpu().numpy().astype(np.float16)
            now = time.time()
            if now - checkpoint >= 60 or end == len(bundle.ids):
                rate = end / max(now - start, 1e-6)
                print(f"[embedding] passage={passage + 1} rows={end}/{len(bundle.ids)} rate={rate:.1f}/s elapsed={now - start:.1f}s", flush=True)
                matrix.flush()
                checkpoint = now
    del matrix
    os.replace(temporary, path)
    del model
    torch.cuda.empty_cache()


def get_medcpt_embeddings(bundle: ClinicalBundle, shared_root: Path, debug: bool) -> tuple[np.ndarray, str]:
    root, content_key = _cache_root(shared_root, bundle.ids)
    ids_path = root / "row_ids.npy"
    if ids_path.exists():
        cached_ids = np.load(ids_path, allow_pickle=False)
        if not np.array_equal(cached_ids, bundle.ids):
            raise RuntimeError("MedCPT cache row-order mismatch")
    else:
        np.save(ids_path, bundle.ids)
    passages = 1 if debug else 3
    paths = [root / f"passage_{index + 1}.npy" for index in range(passages)]
    for index, path in enumerate(paths):
        if not path.exists():
            _encode_passages(bundle, path, index)
        else:
            print(f"[embedding] cache hit passage={index + 1} path={path}", flush=True)
    mean_path = root / f"mean_{passages}.npy"
    if not mean_path.exists():
        shape = np.load(paths[0], mmap_mode="r", allow_pickle=False).shape
        temporary = mean_path.with_suffix(mean_path.suffix + ".partial")
        mean = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=shape)
        block = 8192
        for begin in range(0, shape[0], block):
            end = min(begin + block, shape[0])
            values = np.zeros((end - begin, shape[1]), dtype=np.float32)
            for path in paths:
                values += np.asarray(np.load(path, mmap_mode="r", allow_pickle=False)[begin:end], dtype=np.float32)
            values /= passages
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
            mean[begin:end] = values.astype(np.float16)
        mean.flush()
        del mean
        os.replace(temporary, mean_path)
    _register(shared_root, root, content_key)
    return np.load(mean_path, mmap_mode="r", allow_pickle=False), content_key
