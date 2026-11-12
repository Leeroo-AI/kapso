# The abstraction scan: a bundled harness names no benchmark, task, dataset,
# competition or problem, and pins no quota, deadline, score or hardware
# number on its pages and skills — the domain stays the domain's and a task's
# facts stay in the task statement. Benchmark tooling, not a core module: the
# vocabulary is the benchmarks' own data, dependency-free (RelBench's databases
# and task names, MLE's competition ids and IOAI's competition slugs as
# committed data files, ALE's reported problems from problems.py), plus the names
# of the benchmarks and libraries themselves with a decision per word: a
# platform's name is domain vocabulary everywhere, a data library's name is
# permitted on a skill (its API is what the skill teaches) and a constant on
# a page, a benchmark's name is a constant everywhere. A zero-hit scan is
# evidence, not proof; the reviews of each harness read for the rest.
# docs/plans/redesign/wp8-harness-ports.md change 16.
#
#   PYTHONPATH=src:. python benchmarks/harness_scan.py <harness root> [...]

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from benchmarks.ale.problems import REPORTED_PROBLEMS

BENCHMARKS_ROOT = Path(__file__).resolve().parent
RELBENCH_NAMES = BENCHMARKS_ROOT / "relbench" / "data" / "datasets.json"
MLE_COMPETITIONS = BENCHMARKS_ROOT / "mle" / "data" / "competitions.json"
IOAI_COMPETITIONS = BENCHMARKS_ROOT / "ioai2026" / "data" / "competitions.json"

NEVER = "never"            # a constant anywhere in a harness
SKILL_ONLY = "skill-only"  # a library's API: a skill's vocabulary, a page's constant
# The benchmarks and libraries by name. Platforms (kaggle, atcoder) are
# domain vocabulary and absent from this list.
NAMED_WORDS: Dict[str, str] = {
    "relbench": SKILL_ONLY,
    "relbench_cache_dir": SKILL_ONLY,
    "mle-bench": NEVER,
    "mlebench": NEVER,
    "mle_bench": NEVER,
    "ale-bench": NEVER,
    "ale_bench": NEVER,
    "alebench": NEVER,
    "ahc": NEVER,
    "ioai": NEVER,
    "kumo": NEVER,
    "kumorfm": NEVER,
}
# The number rule: a number within two tokens of a quota, deadline, score or
# hardware word is a pinned constant, the nearest such word named; a
# statistical threshold beside its own name (a p-value, a z), a count of
# standard errors and version strings are allowed. A unit attached to its
# number (16GB) is read as two tokens. The score words are a benchmark's
# results leaking into the harness written from it; a harness written from
# published research cites its studies' results as evidence, so the scan of
# one is run with `score_numbers` off, and quotas, deadlines and hardware
# are read all the same.
QUOTA_WORDS = ("submission", "submissions", "attempt", "attempts", "quota", "daily", "limit", "cap")
DEADLINE_WORDS = ("hour", "hours", "hr", "hrs", "minute", "minutes", "min", "mins", "second", "seconds",
                  "sec", "secs", "day", "days", "week", "weeks", "deadline", "session", "sessions")
SCORE_WORDS = ("score", "scores", "points", "pts", "auroc", "auc", "accuracy", "f1", "mae", "rmse",
               "leaderboard", "lb", "rank", "place")
HARDWARE_WORDS = ("gb", "gib", "mb", "vram", "gpu", "gpus", "cpu", "cpus", "core", "cores", "ram",
                  "memory", "a100", "h100", "t4", "p100", "v100", "l4")
NUMBER_NEIGHBOURS = QUOTA_WORDS + DEADLINE_WORDS + SCORE_WORDS + HARDWARE_WORDS
STATISTICAL_THRESHOLDS = ("0.05", "0.01", "0.001", "1.96")
STATISTICAL_WORDS = ("p", "p-value", "p-values", "pvalue", "alpha", "significance", "significant", "confidence",
                     "level", "z", "t")
SPREAD_UNITS = ("se", "sd", "sigma", "std", "standard")   # "2 SE" is a threshold, not a quota
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_NUMBER = re.compile(r"^\d+(\.\d+)?%?$")
_TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.%-]*")
_ATTACHED = re.compile(r"^(\d+(?:\.\d+)?)([a-z]+)$")
# The manifest (its description is prose `kapso harness list` prints) and a
# skill's references are read as well; scripts and the environment file are not.
SCANNED = ("harness.yaml", "instructions/*.md", "agents/*.md", "skills/*/SKILL.md", "skills/*/references/*.md")


@dataclass(frozen=True)
class Finding:
    file: Path
    line: int
    word: str
    rule: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: {self.word!r} ({self.rule})"


def _names_in(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"{path}: the vocabulary data file is missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(v, list) for v in data.values()):
        raise ValueError(f"{path}: expected an object of name lists")
    return [name for names in data.values() for name in names]


def vocabulary() -> Dict[str, str]:
    """Every word with its decision: the benchmarks' task, dataset,
    competition and problem names (never), the named benchmarks and
    libraries (their decision)."""
    words: Dict[str, str] = {}
    for name in _names_in(RELBENCH_NAMES) + list(REPORTED_PROBLEMS) + _names_in(MLE_COMPETITIONS) + _names_in(IOAI_COMPETITIONS):
        words[name.lower()] = NEVER
    words.update(NAMED_WORDS)
    return words


def scanned_files(root: Path) -> List[Path]:
    """The manifest, the pages, the skills' bodies and their references:
    not scripts, not the environment file."""
    return sorted(path for pattern in SCANNED for path in root.glob(pattern))


def own_names(root: Path) -> List[str]:
    """The harness's own names — skills, hooks, agents, servers, files —
    which a page must use literally and which the scan never reads as a
    vocabulary word (`relbench-api` is a skill, not the library)."""
    names = set()
    for path in root.rglob("*"):
        if path.name.startswith("."):
            continue
        names.add(path.name)
        if path.is_file():
            names.add(path.stem)
    return sorted((name for name in names if len(name) > 2), key=len, reverse=True)


def prose_lines(text: str) -> Iterator[Tuple[int, str]]:
    """The lines outside fenced code, numbered from one."""
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield number, line


def _blank_own_names(line: str, names: List[str]) -> str:
    for name in names:
        line = re.sub(rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?![A-Za-z0-9_-])", " " * len(name), line)
    return line


def _tokens(line: str) -> Tuple[List[str], List[str]]:
    """The line's raw tokens and their lowered forms, a unit attached to
    its number (16GB, 2h) split into the number and the unit."""
    raw = _TOKEN.findall(line)
    tokens: List[str] = []
    for token in raw:
        word = token.lower().strip(".,;:()")
        attached = _ATTACHED.match(word)
        if attached and attached.group(2) in NUMBER_NEIGHBOURS:
            tokens.extend(attached.groups())
        else:
            tokens.append(word)
    return raw, tokens


def _number_findings(path: Path, number: int, line: str, score_numbers: bool) -> List[Finding]:
    raw, tokens = _tokens(line)
    pinning = NUMBER_NEIGHBOURS if score_numbers else QUOTA_WORDS + DEADLINE_WORDS + HARDWARE_WORDS
    findings = []
    for index, token in enumerate(tokens):
        if not _NUMBER.match(token) or _VERSION.match(token):
            continue
        if index == 0 and raw[0].endswith(".") and line.lstrip().startswith(raw[0]):
            continue                                   # an ordered list's marker
        if index + 1 < len(tokens) and tokens[index + 1] in SPREAD_UNITS:
            continue
        nearest = (index + 1, index - 1, index + 2, index - 2)
        window = [tokens[at] for at in nearest if 0 <= at < len(tokens)]
        if token in STATISTICAL_THRESHOLDS and any(word in STATISTICAL_WORDS for word in window):
            continue                                   # a threshold beside its own name
        neighbours = [word for word in window if word in pinning]
        if neighbours:
            findings.append(Finding(path, number, f"{token} {neighbours[0]}", "number"))
    return findings


def scan(root: Path, words: Dict[str, str], score_numbers: bool = True) -> List[Finding]:
    """Every vocabulary word and pinned number on the harness's pages and
    skills, outside fenced code: the file, the line and the word. With
    `score_numbers` off, a number beside a score word is cited evidence and
    not a finding (a harness written from research, not from a benchmark)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root}: no harness directory")
    names = own_names(root)
    patterns = {
        word: re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(word)}(?![A-Za-z0-9_-])", re.IGNORECASE)
        for word in words
    }
    findings: List[Finding] = []
    for path in scanned_files(root):
        on_page = "skills" not in path.relative_to(root).parts
        for number, raw in prose_lines(path.read_text(encoding="utf-8")):
            # The number rule reads the raw line (a file named after a score
            # word must not blind it); the vocabulary reads the line with the
            # harness's own names blanked.
            findings += _number_findings(path, number, raw, score_numbers)
            line = _blank_own_names(raw, names)
            for word, decision in words.items():
                if decision == SKILL_ONLY and not on_page:
                    continue
                if patterns[word].search(line):
                    findings.append(Finding(path, number, word, decision))
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a harness for benchmark names and pinned numbers.")
    parser.add_argument("roots", nargs="+", type=Path, help="harness directories")
    parser.add_argument("--cited-scores", action="store_true",
                        help="the harnesses were written from research: a number beside a score word is cited evidence")
    args = parser.parse_args()
    words = vocabulary()
    total = 0
    for root in args.roots:
        findings = scan(root, words, score_numbers=not args.cited_scores)
        total += len(findings)
        print(f"{root}: {len(findings)} finding(s) over {len(scanned_files(root))} files, {len(words)} words")
        for finding in findings:
            print(f"  {finding}")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
