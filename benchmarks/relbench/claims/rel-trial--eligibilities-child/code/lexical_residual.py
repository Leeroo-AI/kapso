# Imports

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


# Configuration

LEXICAL_VERSION = "lane1_lexical_v3"

PEDIATRIC_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"\bpediatri(?:c|cs|cian|cians)\b|\bpaediatri(?:c|cs|cian|cians)\b",
        r"\bchild(?:ren|hood)?\b|\bminors?\b",
        r"\badolescen(?:t|ts|ce)\b|\bteen(?:ager|agers|age|s)?\b|\byouth\b",
        r"\binfants?\b|\bnewborns?\b|\bneonat(?:al|e|es)\b",
        r"\bjuvenile\b|\bschool[- ]age(?:d)?\b",
    ]
]

ADULT_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"\badults?\b|\badulthood\b",
        r"\belderly\b|\bgeriatri(?:c|cs)\b|\bseniors?\b",
        r"\bolder (?:people|persons|patients|subjects|men|women|adults)\b",
        r"\bpostmenopausal\b|\bmenopausal\b",
    ]
]

AGE_PATTERNS = [
    re.compile(r"\b(?:aged?|ages?)\s*(?:of\s*)?(\d{1,3})\b", re.I),
    re.compile(r"\b(\d{1,3})\s*(?:years?|months?)\s*(?:old|of age)\b", re.I),
]


# Documents

def documents(features, ids: np.ndarray) -> list[str]:
    selected = features.iloc[np.asarray(ids, dtype=np.int64)]
    return [
        f"{study} [SEP] {concept} [SEP] {aux}"
        for study, concept, aux in zip(selected["study"], selected["concept_org"], selected["design_aux"])
    ]


def regex_features(texts: list[str]) -> np.ndarray:
    matrix = np.zeros((len(texts), 16), dtype=np.float32)
    for row, text in enumerate(texts):
        ped_counts = [len(pattern.findall(text)) for pattern in PEDIATRIC_PATTERNS]
        adult_counts = [len(pattern.findall(text)) for pattern in ADULT_PATTERNS]
        ages = []
        for pattern in AGE_PATTERNS:
            ages.extend(int(value) for value in pattern.findall(text))
        matrix[row, 0:5] = np.minimum(ped_counts, 4)
        matrix[row, 5:9] = np.minimum(adult_counts, 4)
        matrix[row, 9] = float(sum(ped_counts) > 0)
        matrix[row, 10] = float(sum(adult_counts) > 0)
        matrix[row, 11] = float(bool(ages) and min(ages) < 18)
        matrix[row, 12] = float(bool(ages) and max(ages) >= 18)
        matrix[row, 13] = float(bool(ages) and min(ages) <= 12)
        matrix[row, 14] = float(bool(ages) and min(ages) <= 2)
        matrix[row, 15] = np.tanh((sum(ped_counts) - sum(adult_counts)) / 3.0)
    return matrix


def evidence_categories(texts: list[str]) -> np.ndarray:
    matrix = regex_features(texts)
    pediatric = matrix[:, 9] > 0
    adult = matrix[:, 10] > 0
    return np.where(pediatric & adult, 3, np.where(pediatric, 1, np.where(adult, 2, 0))).astype(np.int8)


# Training

def fingerprint(train_ids: np.ndarray, predict_ids: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(LEXICAL_VERSION.encode())
    digest.update(np.asarray(train_ids, dtype=np.int64).tobytes())
    digest.update(np.asarray(predict_ids, dtype=np.int64).tobytes())
    digest.update(np.asarray(labels, dtype=np.int8).tobytes())
    return digest.hexdigest()[:20]


def fit_lexical(
    features,
    train_ids: np.ndarray,
    labels: np.ndarray,
    predict_ids: np.ndarray,
    stage: str,
    cache_root: Path,
    debug: bool = False,
) -> tuple[np.ndarray, dict]:
    key = fingerprint(train_ids, predict_ids, labels)
    stage_dir = cache_root / LEXICAL_VERSION
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / f"{stage}_{key}.npz"
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        logits = cached["logits"].astype(np.float64)
        metadata = json.loads(str(cached["metadata"].item()))
        if len(logits) == len(predict_ids):
            metadata["cached"] = True
            return logits, metadata

    started = time.time()
    train_texts = documents(features, train_ids)
    predict_texts = documents(features, predict_ids)
    min_df = 2 if debug else 4
    word_features = 18000 if debug else 50000
    char_features = 22000 if debug else 50000
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        min_df=min_df,
        max_df=0.998,
        max_features=word_features,
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        min_df=min_df,
        max_df=0.998,
        max_features=char_features,
        dtype=np.float32,
    )
    train_word = word.fit_transform(train_texts)
    predict_word = word.transform(predict_texts)
    train_char = char.fit_transform(train_texts)
    predict_char = char.transform(predict_texts)
    train_regex = sparse.csr_matrix(regex_features(train_texts) * 0.35, dtype=np.float32)
    predict_regex = sparse.csr_matrix(regex_features(predict_texts) * 0.35, dtype=np.float32)
    train_matrix = sparse.hstack([train_word, train_char, train_regex], format="csr", dtype=np.float32)
    predict_matrix = sparse.hstack([predict_word, predict_char, predict_regex], format="csr", dtype=np.float32)
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1.5e-5,
        max_iter=18 if not debug else 8,
        tol=1e-4,
        class_weight={0: 1.0, 1: 2.0},
        average=True,
        random_state=1337,
        shuffle=True,
    )
    classifier.fit(train_matrix, np.asarray(labels, dtype=np.int8))
    logits = np.asarray(classifier.decision_function(predict_matrix), dtype=np.float64)
    metadata = {
        "stage": stage,
        "key": key,
        "train_rows": int(len(train_ids)),
        "predict_rows": int(len(predict_ids)),
        "word_vocabulary": int(len(word.vocabulary_)),
        "char_vocabulary": int(len(char.vocabulary_)),
        "matrix_columns": int(train_matrix.shape[1]),
        "matrix_nonzeros": int(train_matrix.nnz),
        "elapsed_seconds": round(time.time() - started, 3),
        "cached": False,
    }
    temporary = stage_dir / f"{stage}_{key}.{os.getpid()}.tmp.npz"
    np.savez_compressed(temporary, logits=logits.astype(np.float32), metadata=json.dumps(metadata))
    os.replace(temporary, path)
    del train_texts, predict_texts, train_word, predict_word, train_char, predict_char
    del train_regex, predict_regex, train_matrix, predict_matrix, classifier, word, char
    gc.collect()
    return logits, metadata
