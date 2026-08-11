from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from model_pipeline import clean_matrix, context_indices


PACKAGE_VERSION = "8.2.0"
CHECKPOINT_REVISION = "e06f7a4f38db4fb00d6259da01ed80cca61721de"
CHECKPOINT_NAME = "tabpfn-v2-classifier.ckpt"
CHECKPOINT_SHA256 = "f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f"
CHECKPOINT_URL = f"https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/{CHECKPOINT_REVISION}/{CHECKPOINT_NAME}?download=true"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_tabpfn(shared: Path) -> tuple[Path, float]:
    start = time.monotonic()
    os.environ["TABPFN_NO_BROWSER"] = "1"
    try:
        import tabpfn
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", f"tabpfn=={PACKAGE_VERSION}"],
            check=True,
            timeout=300,
        )
        import tabpfn
    checkpoint_root = shared / "tabpfn_v2"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_root / CHECKPOINT_NAME
    if not checkpoint.exists() or _sha256(checkpoint) != CHECKPOINT_SHA256:
        temporary = checkpoint.with_suffix(f".{os.getpid()}.download")
        with urllib.request.urlopen(CHECKPOINT_URL, timeout=240) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        if _sha256(temporary) != CHECKPOINT_SHA256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("pinned TabPFN-v2 checkpoint hash verification failed")
        os.replace(temporary, checkpoint)
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("cached TabPFN-v2 checkpoint hash verification failed")
    return checkpoint, time.monotonic() - start


class TabPFNRunner:
    def __init__(self, checkpoint: Path, estimators: int, seed: int):
        from tabpfn import TabPFNClassifier

        self.classifier = TabPFNClassifier(
            model_path=checkpoint,
            device="cuda",
            n_estimators=estimators,
            auto_scale_n_estimators=False,
            ignore_pretraining_limits=True,
            random_state=seed,
            show_progress_bar=False,
            n_preprocessing_jobs=1,
        )

    def predict(
        self,
        train_matrix: np.ndarray,
        labels: np.ndarray,
        dates: pd.Series,
        query_matrix: np.ndarray,
        policy: str,
        maximum: int,
        seed: int,
        debug: bool,
    ) -> np.ndarray:
        indices = context_indices(labels, dates, maximum, policy, seed, debug)
        context = clean_matrix(train_matrix[indices])
        query = clean_matrix(query_matrix)
        median = np.nanmedian(context, axis=0)
        median[~np.isfinite(median)] = 0.0
        context = np.where(np.isnan(context), median, context).astype(np.float32)
        query = np.where(np.isnan(query), median, query).astype(np.float32)
        self.classifier.fit(context, np.asarray(labels)[indices])
        return self.classifier.predict_proba(query)[:, 1].astype(np.float64)
