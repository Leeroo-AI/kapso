# Gauntlet — the two-trap harness: substance diff + verdict assembly.
#
# Design: learn-from-trajectories-grader-scoring.md §2.3 (minimal battery).
# The traps buy an answer key by construction: run the learning step on
# controlled input and diff the result. This module owns the mechanical half —
# the duplicate-trap fixture (a clone under a fresh identity), the stability
# substance-diff (touched-card set, states, transitions, scores within
# tolerance; prose free to differ), and gauntlet.md assembly with per-trap
# {verdict, rationale} (no naked tags). The trap RUNNERS — invoking the
# update-crew CLI black-box on sandbox checkouts — wire in at P4, when a crew
# exists to trap.

import shutil
from pathlib import Path
from typing import Any, Dict, List

import yaml

from kapso.learning.bank import Bank

TRAPS = ("duplicate", "stability")
VERDICTS = ("PASS", "FAIL")


def build_duplicate_fixture(
    mined_view_dir: str, fixture_dir: str, clone_trajectory_id: str
) -> Path:
    """Clone a mined view under a fresh trajectory identity (same run ids,
    same numbers — the disguise rewording is the P4 runner's agent step; the
    identity clone is the mechanical base every disguise starts from)."""
    source = Path(mined_view_dir).expanduser()
    if not (source / "index.md").is_file():
        raise FileNotFoundError(f"{mined_view_dir} is not a mined view")
    target = Path(fixture_dir).expanduser() / clone_trajectory_id / "mined"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target


def substance_diff(
    bank_a_dir: str, bank_b_dir: str, tolerance: float
) -> List[str]:
    """The stability comparator: two banks must agree in SUBSTANCE — same
    card set, same states, versions, and scores within tolerance. Prose may
    differ freely (§2.3). Every disagreement is a named difference."""
    bank_a, bank_b = Bank(bank_a_dir), Bank(bank_b_dir)
    differences: List[str] = []
    only_a = sorted(set(bank_a.cards) - set(bank_b.cards))
    only_b = sorted(set(bank_b.cards) - set(bank_a.cards))
    for name in only_a:
        differences.append(f"card {name} exists only in run A")
    for name in only_b:
        differences.append(f"card {name} exists only in run B")
    for name in sorted(set(bank_a.cards) & set(bank_b.cards)):
        card_a, card_b = bank_a.cards[name], bank_b.cards[name]
        if card_a.state != card_b.state:
            differences.append(
                f"{name}: state {card_a.state!r} (A) vs {card_b.state!r} (B)"
            )
        version_a = (card_a.frontmatter.get("provenance") or {}).get("version")
        version_b = (card_b.frontmatter.get("provenance") or {}).get("version")
        if version_a != version_b:
            differences.append(f"{name}: version {version_a} (A) vs {version_b} (B)")
        score_a, score_b = card_a.score, card_b.score
        if (score_a is None) != (score_b is None):
            differences.append(f"{name}: score present in one run only")
        elif score_a is not None and abs(score_a - score_b) > tolerance:
            differences.append(
                f"{name}: score {score_a} (A) vs {score_b} (B) exceeds "
                f"tolerance {tolerance}"
            )
    return differences


def assemble_gauntlet(
    trap_results: Dict[str, Dict[str, str]],
    context: Dict[str, Any],
) -> str:
    """gauntlet.md: frontmatter with per-trap {verdict, rationale} and the
    rolled verdict (any FAIL ⇒ FAIL); body sections carry construction +
    proof refs. Rationales are mandatory — no naked tags."""
    for trap, result in trap_results.items():
        if trap not in TRAPS:
            raise ValueError(f"unknown trap {trap!r}: expected one of {TRAPS}")
        if result.get("verdict") not in VERDICTS:
            raise ValueError(f"trap {trap!r} verdict must be PASS or FAIL")
        if not str(result.get("rationale", "")).strip():
            raise ValueError(f"trap {trap!r} has no rationale — no naked tags")
    rolled = "FAIL" if any(
        r["verdict"] == "FAIL" for r in trap_results.values()
    ) else "PASS"
    rolled_rationale = context.get("rolled_rationale", "")
    if not str(rolled_rationale).strip():
        raise ValueError("rolled verdict has no rationale — no naked tags")

    frontmatter = {
        "learner_version": context["learner_version"],
        "bank_head": context["bank_head"],
        "batch": context.get("batch", []),
        "gauntlet": trap_results,
        "verdict": rolled,
        "rationale": rolled_rationale,
    }
    body_sections = [
        f"## {trap} — construction + proof\n{context.get(f'{trap}_proof', 'n/a')}"
        for trap in trap_results
    ]
    return (
        "---\n" + yaml.safe_dump(frontmatter, sort_keys=False)
        + "---\n\n" + "\n\n".join(body_sections) + "\n"
    )
