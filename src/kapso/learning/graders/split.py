# Split manifest — the authoritative learn/held-out partition of the corpus.
#
# Design: learn-from-trajectories-grader-scoring.md §3. One split.yaml per
# exam version, versioned with the harness; scorecards stamp split_version and
# paired comparisons are valid only within one version. The frame checks at
# load: every store trajectory appears exactly once, no family (grouping key)
# on both sides, and every version carries its rationale.

from typing import Any, Dict, List

import yaml

SPLIT_SIDES = ("learn", "held_out")


def load_split(split_path: str) -> Dict[str, Any]:
    """Parse and shape-check a split manifest; malformed raises."""
    with open(split_path) as handle:
        split = yaml.safe_load(handle)
    if not isinstance(split, dict):
        raise ValueError(f"split manifest {split_path!r} is not a mapping")
    for key in ("version", "rule", "rationale"):
        if not split.get(key):
            raise ValueError(f"split manifest {split_path!r} missing `{key}`")
    for side in SPLIT_SIDES:
        entries = split.get(side)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"split manifest {split_path!r} has no `{side}:` list")
        for entry in entries:
            if not isinstance(entry, dict) or not all(
                k in entry for k in ("id", "family", "date")
            ):
                raise ValueError(
                    f"split entry {entry!r} must carry id, family, date"
                )
    return split


def validate_split(
    split: Dict[str, Any], store_manifests: List[Dict[str, Any]]
) -> List[str]:
    """The frame checks (§3) against the resident corpus; findings, not raises
    — the caller decides whether a mismatch is fatal (it is, for grading)."""
    findings: List[str] = []
    listed: Dict[str, str] = {}
    families: Dict[str, set] = {side: set() for side in SPLIT_SIDES}
    for side in SPLIT_SIDES:
        for entry in split[side]:
            if entry["id"] in listed:
                findings.append(
                    f"trajectory {entry['id']} appears on both `{listed[entry['id']]}` "
                    f"and `{side}`"
                    if listed[entry["id"]] != side
                    else f"trajectory {entry['id']} is listed twice under `{side}`"
                )
            listed[entry["id"]] = side
            families[side].add(entry["family"])

    for family in sorted(families["learn"] & families["held_out"]):
        findings.append(f"family {family!r} appears on both sides of the split")

    store_ids = {m["id"]: m for m in store_manifests}
    for trajectory_id in sorted(set(store_ids) - set(listed)):
        findings.append(f"store trajectory {trajectory_id} is not in the split")
    for trajectory_id in sorted(set(listed) - set(store_ids)):
        findings.append(f"split lists {trajectory_id} which is not in the store")
    for trajectory_id, side in listed.items():
        manifest = store_ids.get(trajectory_id)
        if manifest is None:
            continue
        entry = next(e for e in split[side] if e["id"] == trajectory_id)
        if manifest.get("dataset") and entry["family"] != manifest["dataset"]:
            findings.append(
                f"{trajectory_id}: split family {entry['family']!r} does not "
                f"match the manifest dataset {manifest['dataset']!r}"
            )
    return findings


def held_out_ids(split: Dict[str, Any]) -> List[str]:
    """Full-mode grading grades exactly this list — nothing else is the exam."""
    return [entry["id"] for entry in split["held_out"]]


def assert_batch_disjoint(split: Dict[str, Any], batch_ids: List[str]) -> None:
    """The update-run twin check: a development batch may never contain a
    held-out trajectory (fails loud before any session exists)."""
    overlap = sorted(set(batch_ids) & set(held_out_ids(split)))
    if overlap:
        raise ValueError(
            f"held-out trajectories in a development batch: {overlap}"
        )
