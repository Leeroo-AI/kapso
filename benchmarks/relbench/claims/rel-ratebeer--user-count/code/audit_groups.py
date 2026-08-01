from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from feature_pipeline import ensure_episode_cache, ensure_event_cache
from kapso_datasets.common import shared_cache_dir
from main import build_labeled_frame, create_feature_data, load_official_frames
from model_pipeline import DAY_MS, _candidate_oof


def main() -> None:
    shared = shared_cache_dir()
    train, validation, test = load_official_frames()
    events = ensure_event_cache(shared)
    episodes = ensure_episode_cache(shared)
    labeled = build_labeled_frame(train, validation, episodes, False)
    matrix_all, timestamps, users, target, kinds, names, groups = create_feature_data(labeled, test, events, shared, False)
    matrix = matrix_all[: len(labeled)]
    cutoff = np.datetime64("2018-09-01", "ms").astype(np.int64)
    allowed = (kinds != 2) & (timestamps + 90 * DAY_MS <= cutoff)
    folds = [
        np.datetime64("2017-09-01", "ms").astype(np.int64),
        np.datetime64("2018-03-01", "ms").astype(np.int64),
        np.datetime64("2018-06-01", "ms").astype(np.int64),
    ]
    addition_groups = ("site_drift_addition", "user_trajectory_addition", "cadence_distribution_addition", "weekly_trajectory_addition")
    selected = np.asarray([index for index, group in enumerate(groups) if not group.startswith("wide_") and group not in addition_groups], dtype=np.int32)
    baseline = _candidate_oof(np.ascontiguousarray(matrix[:, selected]), target, timestamps, allowed, folds, 63, 1096.0, "regression", 2000, 1337)
    records = {"baseline": baseline["scores"].tolist(), "groups": {}}
    current = baseline
    for group_name in addition_groups:
        additions = [index for index, group in enumerate(groups) if group == group_name]
        candidate_indices = np.asarray(list(selected) + additions, dtype=np.int32)
        candidate = _candidate_oof(np.ascontiguousarray(matrix[:, candidate_indices]), target, timestamps, allowed, folds, 63, 1096.0, "regression", 2000, 1337)
        differences = candidate["scores"] - current["scores"]
        uncertainty = float(np.std(differences) / np.sqrt(len(differences)))
        retained = float(np.mean(differences)) > max(0.0, uncertainty)
        records["groups"][group_name] = {
            "scores": candidate["scores"].tolist(),
            "delta_mean": float(np.mean(differences)),
            "uncertainty": uncertainty,
            "retained": retained,
        }
        if retained:
            selected = candidate_indices
            current = candidate
    removals = {
        "favorite_aux": lambda index: names[index].startswith("favorite_") or names[index].startswith("availability_"),
        "place_aux": lambda index: names[index].startswith("place_rating_"),
        "language": lambda index: names[index].startswith("rating_language_"),
        "score_comment": lambda index: names[index].startswith("rating_score_") or names[index].startswith("rating_comment_"),
        "platform": lambda index: groups[index] == "core_platform",
        "calendar": lambda index: groups[index] == "core_calendar",
    }
    records["ablations"] = {}
    for removal_name, predicate in removals.items():
        candidate_indices = np.asarray([index for index in selected if not predicate(int(index))], dtype=np.int32)
        candidate = _candidate_oof(np.ascontiguousarray(matrix[:, candidate_indices]), target, timestamps, allowed, folds, 63, 1096.0, "regression", 2000, 1337)
        differences = candidate["scores"] - current["scores"]
        uncertainty = float(np.std(differences) / np.sqrt(len(differences)))
        retained = float(np.mean(differences)) > max(0.0, uncertainty)
        records["ablations"][removal_name] = {
            "scores": candidate["scores"].tolist(),
            "delta_mean": float(np.mean(differences)),
            "uncertainty": uncertainty,
            "remove": retained,
        }
        if retained:
            selected = candidate_indices
            current = candidate
    records["selected_features"] = int(len(selected))
    output = Path("output_data_generic_exp_0")
    output.mkdir(parents=True, exist_ok=True)
    (output / "group_audit_v5.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records))


if __name__ == "__main__":
    main()
