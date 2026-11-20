"""Static leakage scan for a prediction task.

Two modes over one rule table. `--diff` reads a unified diff on stdin and
scans the added lines (an attempt's own change, before its evaluation);
`--tree DIR` scans every Python file the candidate owns under a directory
(a delivered candidate, whole, for the final audit), leaving out what the
engine staged beside it (the hosts' skill and settings directories, the
evaluation directory, the data directory, version control) and what a
candidate installed (virtual environments, site packages). Every rule
that matches a line is reported, one JSON finding per match on stdout:
the file, the line, the rule, the concern and the line itself. A
violation exits 1; an advisory finding alone exits 0 and is prefixed
`verify:` because the pattern is legal in one context and a leak in
another (the test-predicting model against the one that produces
validation predictions; rows that repeat no unit against rows that do),
so it is traced, not condemned. The scan reads one line at
a time: a call split over several lines is a line of its own, which is
why verification also reads the code (the leakage-auditor delegate).

The shape-neutral rules apply to every task; the library rules are the
relational benchmark library's own traps and say nothing on a flat table.

    git diff main...HEAD | python3 scan.py --diff --regime frozen|seed-time|autocomplete
    python3 scan.py --tree . --regime frozen
"""
import argparse
import json
import re
import sys
from pathlib import Path

REGIMES = ("frozen", "seed-time", "autocomplete")
STAGED_DIRECTORIES = {
    ".agents", ".claude", ".codex", ".git", ".kapso", "kapso_evaluation", "kapso_datasets", "prepared",
    "__pycache__", ".venv", "venv", "site-packages", "node_modules",
}

# (rule, pattern, concern, advisory)
RULES = [
    # Shape-neutral: any prediction task.
    ("test-label-read",
     r"test[^\n]{0,40}\[\s*[\"'](label|labels|target|targets|y)[\"']\s*\]|sample_submission[^\n]{0,60}\[\s*[\"'](label|labels|target|targets|y)[\"']\s*\]",
     "reads a label column from the test data or a sample submission", False),
    ("fit-on-train-and-test",
     r"\.fit(_transform)?\(\s*(pd\.)?concat\([^\n]*test|\.fit(_transform)?\([^\n]*(train[^\n]{0,30}test|test[^\n]{0,30}train)",
     "fits a transform on training and test rows together", False),
    ("train-val-mixing",
     r"(concatenate|concat|hstack|vstack|np\.r_)[^\n]{0,80}(train[^\n]{0,30}val|val[^\n]{0,30}train)|train_?val_?(idx|index|indices|rows)",
     "train+val mixing: confirm the validation-prediction chain is pre-refit (the two-model contract)", True),
    ("fit-on-validation",
     r"\.fit\([^\n]{0,60}(val_|_val\b|valid_|_valid\b)(y|label|labels|target|targets)?",
     "a fit that names validation data: confirm it is the test-predicting chain", True),
    ("ungrouped-split",
     r"\b(KFold|StratifiedKFold|RepeatedKFold|RepeatedStratifiedKFold|ShuffleSplit|StratifiedShuffleSplit|train_test_split)\s*\("
     r"|\bcv\s*=\s*\d|\bnfold\s*="
     # A cross-validation call closed on its line with no cv= takes the
     # default folds, which are ungrouped.
     r"|\b(cross_val_score|cross_val_predict|cross_validate|GridSearchCV|RandomizedSearchCV)\((?![^\n]*\bcv\s*=)[^\n]*\)",
     "a split without groups: confirm the rows repeat no unit, or fold by the repeated unit or by time (the cv-protocol skill)",
     True),
    # The relational benchmark library's traps.
    ("unmasked-test-table", r"mask_input_cols\s*=\s*False[^\n]*test|test[^\n]*mask_input_cols\s*=\s*False",
     "reads the test table with its input masks off (test labels)", False),
    ("unmasked-table", r"mask_input_cols\s*=\s*False",
     "reads a task table with its input masks off: legal on train and val, confirm which split", True),
    ("removed-columns", r"removed_cols", "reads the columns the sanitizer removed (the target and what derives it)", False),
    ("cache-location", r"RELBENCH_CACHE_DIR[\"']?\s*[\]\)]?\s*=[^=]|environ\.setdefault\(\s*[\"']RELBENCH_CACHE_DIR",
     "overrides the library's cache location: the sanitized copy is the only cache", False),
    ("outside-snapshot", r"(^|[/\"'\s])\.\.[/\\](?:\.\.[/\\])*(?:[A-Za-z0-9_-]*[/\\])*(val|test)_\d{3}_|[/\\](val|test)_\d{3}_[0-9-]+[/\\]",
     "reaches into another tick's snapshot: each invocation sees its own", False),
    ("task-stats", r"task\.stats\s*\(", "probes the task tables' statistics on a sanitized copy", False),
    ("download", r"download\s*=\s*True", "downloads the dataset: the sanitized copy is complete and read-only", False),
    ("direct-fetch", r"relbench\.stanford\.edu|\bpooch\b", "fetches the benchmark's files directly", False),
    ("private-builder", r"_get_table\s*\(", "calls a private table builder", False),
    ("pristine-cache", r"\.cache[/\\]relbench", "touches the pristine library cache path", False),
]
REGIME_RULES = {
    "frozen": [("untruncated-load", r"upto_test_timestamp\s*=\s*False",
                "requests the untruncated database under the frozen regime", False)],
    "seed-time": [("re-truncated-load", r"upto_test_timestamp\s*=\s*True",
                   "truncates the per-tick snapshot at the static cutoff: the trap the temporal protocol names", False),
                  ("default-load", r"get_db\s*\(\s*\)",
                   "loads the database with the loader's default, which truncates at the static cutoff under per-row seed time", False)],
    "autocomplete": [("truncated-load", r"upto_test_timestamp\s*=\s*True|get_db\s*\(\s*\)",
                      "truncates the database whose post-cutoff rows are the rows to predict", False)],
}
FILE_HEADER = re.compile(r"^\+\+\+ (?:b/)?(.+)$")
HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def added_lines(diff):
    """Yield (file, line number, text) for every added line of a unified diff."""
    current = None
    number = 0
    for raw in diff.splitlines():
        header = FILE_HEADER.match(raw)
        if header:
            current = header.group(1)
            continue
        hunk = HUNK_HEADER.match(raw)
        if hunk:
            number = int(hunk.group(1))
            continue
        if current is None or raw.startswith("---"):
            continue
        if raw.startswith("+"):
            yield current, number, raw[1:]
            number += 1
        elif not raw.startswith("-"):
            number += 1


def candidate_files(root: Path):
    """Every Python file the candidate owns under `root`: the engine's staged
    directories, version control and installed packages are left out. A root
    that does not exist is an error, not an empty scan."""
    if not root.is_dir():
        raise SystemExit(f"no directory to scan at {root}")
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if STAGED_DIRECTORIES.intersection(relative.parts[:-1]):
            continue
        yield path, relative


def tree_lines(root: Path):
    """Yield (file, line number, text) for every line of every candidate file; a file that is not UTF-8 is an error."""
    for path, relative in candidate_files(root):
        for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            yield str(relative), number, text


def compiled_rules(regime):
    return [(rule, re.compile(pattern), concern, advisory)
            for rule, pattern, concern, advisory in RULES + REGIME_RULES[regime]]


def scan(lines, regime):
    """The findings over (file, number, text) triples: every rule that
    matches a line is reported, except that a test-table unmasking is
    reported once, as the violation and not also as the advisory."""
    rules = compiled_rules(regime)
    findings = []
    for file, number, text in lines:
        matched = [(rule, concern, advisory) for rule, pattern, concern, advisory in rules if pattern.search(text)]
        if any(rule == "unmasked-test-table" for rule, _, _ in matched):
            matched = [entry for entry in matched if entry[0] != "unmasked-table"]
        for rule, concern, advisory in matched:
            findings.append({
                "file": file, "line": number, "rule": rule,
                "concern": f"verify: {concern}" if advisory else concern,
                "advisory": advisory, "text": text.strip(),
            })
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regime", required=True, choices=REGIMES, help="the regime the domain statement records")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--diff", action="store_true", help="scan the added lines of the unified diff on stdin")
    mode.add_argument("--tree", metavar="DIR", help="scan every Python file the candidate owns under this directory")
    args = parser.parse_args()
    lines = added_lines(sys.stdin.read()) if args.diff else tree_lines(Path(args.tree))
    findings = scan(lines, args.regime)
    for finding in findings:
        print(json.dumps(finding))
    sys.exit(1 if any(not finding["advisory"] for finding in findings) else 0)


if __name__ == "__main__":
    main()
