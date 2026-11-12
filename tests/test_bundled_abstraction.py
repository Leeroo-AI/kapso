"""The abstraction scan over every bundled harness: no benchmark, task,
dataset, competition or problem name, no data-library name on a page, no
number pinned beside a quota, deadline, score or hardware word (the score
words in a harness written from a benchmark; one written from research
cites its studies' results) — on the pages and the skills' bodies, outside
fenced code. The gate never skips:
the vocabulary is the benchmarks' own dependency-free data, and a missing
data file is a failure, not a skip (docs/plans/redesign/wp8-harness-ports.md
change 16). The scan's own rules are pinned on a fixture harness."""

from pathlib import Path

import pytest

from benchmarks.harness_scan import NEVER, SKILL_ONLY, Finding, scan, scanned_files, vocabulary
from kapso.harness import BUNDLED_ROOT
from kapso.harness.manifest import MANIFEST_FILE

BUNDLED = sorted(path for path in BUNDLED_ROOT.iterdir() if (path / MANIFEST_FILE).is_file())
# The harnesses written from the benchmarks the engine was measured on: a
# number beside a score word in one of them is a benchmark's result. The
# others were written from published research and cite its results as
# evidence, which their reviews read and this rule does not.
PORTED = ("predictive-modelling", "competitive-programming")


@pytest.mark.parametrize("root", BUNDLED, ids=[path.name for path in BUNDLED])
def test_a_bundled_harness_names_no_benchmark_and_pins_no_number(root):
    # Regression: a benchmark's task names, a data library's name on a page
    # or a contest's quota as a figure leaking into a shipped harness — the
    # abstraction the ports were made for. A hit is a finding with its file,
    # line and word; the scan reads every page and skill body.
    findings = scan(root, vocabulary(), score_numbers=root.name in PORTED)
    assert findings == [], "\n".join(str(finding) for finding in findings)
    assert len(scanned_files(root)) >= 4           # the module pages at least


def test_the_vocabulary_is_the_benchmarks_own_data():
    # Regression: the words come from the benchmarks' files, not a list
    # typed here — a RelBench dataset, an ALE problem, an MLE competition
    # and an IOAI slug each resolve to `never`; the data library resolves
    # to `skill-only`; a platform's name is not a word at all.
    words = vocabulary()
    assert words["rel-amazon"] == NEVER and words["ahc008"] == NEVER
    assert words["aerial-cactus-identification"] == NEVER
    assert words["ioai-2026-ai-models-track-practice-task-1-timed-deps-12"] == NEVER
    assert words["relbench"] == SKILL_ONLY and words["mle-bench"] == NEVER
    assert "kaggle" not in words and "atcoder" not in words


def write_harness(root: Path, page: str, skill: str) -> Path:
    (root / "instructions").mkdir(parents=True)
    (root / "skills" / "relbench-api").mkdir(parents=True)
    (root / "instructions" / "setup.md").write_text(page)
    (root / "skills" / "relbench-api" / "SKILL.md").write_text(skill)
    return root


def test_the_rules_on_a_fixture(tmp_path):
    # Regression: the scan's four decisions — a library name is a page's
    # constant and a skill's vocabulary; the harness's own skill name is
    # never read as the library; fenced code is not prose; a number counts
    # only beside a quota, deadline, score or hardware word (a unit attached
    # to it is read apart), and a version string or a statistical threshold
    # beside its own name never does; the threshold beside a score word does.
    root = write_harness(
        tmp_path / "h",
        page=(
            "# Setup\n\nRead the relbench-api skill when the data's library is relbench.\n"
            "Budget 50 submissions a day and 6 hours per session on one GPU with 16 GB.\n"
            "```\nrel-amazon has 24 GB of parquet\n```\n"
            "Version 0.3.1 of the harness; a p-value under 0.05 is the gate; two GPUs.\n"
            "The ahc rounds and the rel-f1 database; mle-bench says so.\n"
            "A score of 0.05 is the floor; the box has 16GB and a 24gb card.\n"
        ),
        skill=(
            "---\nname: relbench-api\n---\n\nThe relbench API loads a database; set RELBENCH_CACHE_DIR.\n"
            "The score of record is the public leaderboard's, read after 3 attempts.\n"
            "A kaggle kernel is a script.\n"
        ),
    )
    findings = scan(root, vocabulary())
    page = root / "instructions" / "setup.md"
    skill = root / "skills" / "relbench-api" / "SKILL.md"
    assert findings == [
        Finding(page, 3, "relbench", SKILL_ONLY),
        Finding(page, 4, "50 submissions", "number"),
        Finding(page, 4, "6 hours", "number"),
        Finding(page, 4, "16 gb", "number"),
        Finding(page, 9, "rel-f1", NEVER),
        Finding(page, 9, "mle-bench", NEVER),
        Finding(page, 9, "ahc", NEVER),
        Finding(page, 10, "0.05 score", "number"),
        Finding(page, 10, "16 gb", "number"),
        Finding(page, 10, "24 gb", "number"),
        Finding(skill, 6, "3 attempts", "number"),
    ]
    # Regression: a harness written from research cites a study's result
    # beside a score word; with the score numbers off that one is no finding,
    # and the quota, the deadline, the hardware and every name still are.
    assert scan(root, vocabulary(), score_numbers=False) == [
        finding for finding in findings if finding.word != "0.05 score"
    ]
