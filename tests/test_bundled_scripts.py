"""The bundled harnesses' scripts on constructed inputs
(docs/plans/redesign/wp8-harness-ports.md, change 4): the leakage scan
reports one finding per rule, in diff mode on the added lines and in tree
mode over a directory, exits 1 on a violation and 0 on advisory findings
alone, and applies the regime's rule; the sanitizer refuses what it must
and coerces a text boolean target."""

import importlib.util
import json
import subprocess
import sys

import pytest

from kapso.harness import BUNDLED_ROOT

SKILLS = BUNDLED_ROOT / "predictive-modelling" / "skills"
SCAN = SKILLS / "leakage-scan" / "scripts" / "scan.py"


def diff_of(*lines, file="main.py"):
    body = "\n".join(f"+{line}" for line in lines)
    return f"--- a/{file}\n+++ b/{file}\n@@ -0,0 +1,{len(lines)} @@\n{body}\n"


def run_scan(regime, *, diff=None, tree=None):
    argv = [sys.executable, str(SCAN), "--regime", regime] + (["--diff"] if diff is not None else ["--tree", str(tree)])
    completed = subprocess.run(argv, input=diff or "", capture_output=True, text=True)
    return completed.returncode, [json.loads(line) for line in completed.stdout.splitlines()]


VIOLATIONS = [
    ("unmasked-test-table", "test = task.get_table('test', mask_input_cols=False)"),
    ("removed-columns", "cols = db.table_dict['users'].removed_cols"),
    ("cache-location", "os.environ['RELBENCH_CACHE_DIR'] = '/tmp/mine'"),
    ("cache-location", "os.environ.setdefault('RELBENCH_CACHE_DIR', '/tmp/mine')"),
    ("outside-snapshot", "other = pd.read_parquet('../test_001_2010-01-01/rel-x/db/users.parquet')"),
    ("test-label-read", "y_test = test_df['label']"),
    ("fit-on-train-and-test", "encoder.fit_transform(pd.concat([train_df, test_df]))"),
    ("task-stats", "print(task.stats())"),
    ("download", "dataset = get_dataset(name, download=True)"),
    ("direct-fetch", "url = 'https://relbench.stanford.edu/data/rel-x.zip'"),
    ("private-builder", "table = task._get_table('test')"),
    ("pristine-cache", "path = Path.home() / '.cache/relbench'"),
]


@pytest.mark.parametrize("rule,line", VIOLATIONS, ids=[f"{rule}-{i}" for i, (rule, _) in enumerate(VIOLATIONS)])
def test_each_violation_is_one_finding_that_fails_the_scan(rule, line):
    code, findings = run_scan("frozen", diff=diff_of("import os", line))
    assert code == 1
    assert [(f["rule"], f["line"], f["advisory"]) for f in findings] == [(rule, 2, False)]
    assert findings[0]["file"] == "main.py" and findings[0]["text"] == line


def test_reading_the_cache_variable_is_not_overriding_it():
    assert run_scan("frozen", diff=diff_of("cache = os.environ['RELBENCH_CACHE_DIR']", "if a == b: pass")) == (0, [])


def test_every_matching_rule_is_reported_on_one_line():
    line = "train = task.get_table('train', mask_input_cols=False); os.environ['RELBENCH_CACHE_DIR'] = '/x'"
    code, findings = run_scan("frozen", diff=diff_of(line))
    assert code == 1 and [(f["rule"], f["advisory"]) for f in findings] == [("unmasked-table", True), ("cache-location", False)]


def test_the_loaders_default_is_the_violation_under_per_row_seed_time_and_the_post_cutoff_row():
    default = diff_of("db = dataset.get_db()")
    assert run_scan("seed-time", diff=default)[0] == 1 and run_scan("seed-time", diff=default)[1][0]["rule"] == "default-load"
    assert run_scan("frozen", diff=default) == (0, [])
    assert run_scan("autocomplete", diff=default)[1][0]["rule"] == "truncated-load"


def test_advisory_findings_alone_pass_with_the_verify_prefix():
    code, findings = run_scan("frozen", diff=diff_of(
        "train = task.get_table('train', mask_input_cols=False)",
        "X = np.concatenate([train_x, val_x])",
        "model.fit(X_val, val_y)",
    ))
    assert code == 0
    assert [f["rule"] for f in findings] == ["unmasked-table", "train-val-mixing", "fit-on-validation"]
    assert all(f["advisory"] and f["concern"].startswith("verify: ") for f in findings)


def test_a_split_without_groups_is_advisory_and_grouped_or_time_splits_are_not():
    code, findings = run_scan("frozen", diff=diff_of(
        "folds = KFold(n_splits=5, shuffle=True)",
        "folds = StratifiedKFold(5)",
        "folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)",
        "splitter = ShuffleSplit(test_size=0.2)",
        "X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2)",
        "scores = cross_val_score(model, X, y, cv=5)",
        "search = GridSearchCV(model, grid, cv = 3)",
        "history = lgb.cv(params, train_set, nfold=5)",
        "scores = cross_val_score(model, X, y)",
        "search = RandomizedSearchCV(model, grid, n_iter=20)",
        "from sklearn.model_selection import KFold, GroupKFold, train_test_split",
        "folds = GroupKFold(n_splits=5)",
        "folds = StratifiedGroupKFold(n_splits=5)",
        "splitter = GroupShuffleSplit(test_size=0.2)",
        "folds = TimeSeriesSplit(n_splits=5)",
        "scores = cross_val_score(model, X, y, groups=g, cv=GroupKFold(5))",
        "n_cv=5",
        "search = GridSearchCV(",
    ))
    assert code == 0
    assert [(f["rule"], f["line"]) for f in findings] == [("ungrouped-split", n) for n in range(1, 11)]
    assert all(f["advisory"] and f["concern"].startswith("verify: ") for f in findings)


def test_a_clean_diff_has_no_findings_and_removed_lines_are_not_read():
    diff = "--- a/main.py\n+++ b/main.py\n@@ -1,2 +1,2 @@\n-x = task._get_table('test')\n+x = task.get_table('val')\n y = 1\n"
    assert run_scan("seed-time", diff=diff) == (0, [])


def test_the_regime_decides_which_truncation_argument_is_the_violation():
    truncated = diff_of("db = dataset.get_db(upto_test_timestamp=True)")
    untruncated = diff_of("db = dataset.get_db(upto_test_timestamp=False)")
    assert run_scan("seed-time", diff=truncated)[0] == 1 and run_scan("frozen", diff=truncated)[0] == 0
    assert run_scan("frozen", diff=untruncated)[0] == 1 and run_scan("seed-time", diff=untruncated)[0] == 0
    # The post-cutoff-row regime keeps its later rows: the untruncated load is the required one.
    assert run_scan("autocomplete", diff=untruncated) == (0, [])
    assert run_scan("autocomplete", diff=truncated)[1][0]["rule"] == "truncated-load"


def test_a_validation_fit_in_the_common_spelling_is_advisory():
    code, findings = run_scan("frozen", diff=diff_of("model.fit(X_val, y_val)", "stacker.fit(valid_preds, valid_y)"))
    assert code == 0 and [f["rule"] for f in findings] == ["fit-on-validation", "fit-on-validation"]


def test_line_numbers_follow_the_hunk_through_context_lines():
    diff = (
        "--- a/features.py\n+++ b/features.py\n@@ -10,3 +10,5 @@\n context\n+a = 1\n context\n"
        "+table = task._get_table('x')\n+b = 2\n"
    )
    code, findings = run_scan("frozen", diff=diff)
    assert code == 1 and [(f["file"], f["line"]) for f in findings] == [("features.py", 13)]


def test_tree_mode_scans_the_candidates_files_and_not_what_the_engine_staged(tmp_path):
    (tmp_path / "main.py").write_text("import os\nx = 1\n")
    (tmp_path / "features").mkdir()
    (tmp_path / "features" / "build.py").write_text("a = 1\ntable = task._get_table('x')\n")
    (tmp_path / "notes.md").write_text("task._get_table('x') in prose is not code\n")
    for staged in (".claude/skills/leakage-scan/scripts", ".agents/skills/x", "kapso_evaluation", "kapso_datasets", "prepared/cache"):
        (tmp_path / staged).mkdir(parents=True)
        (tmp_path / staged / "tool.py").write_text("PATTERN = 'task._get_table('\nos.environ['RELBENCH_CACHE_DIR'] = '/x'\n")
    code, findings = run_scan("frozen", tree=tmp_path)
    assert code == 1 and [(f["file"], f["line"], f["rule"]) for f in findings] == [("features/build.py", 2, "private-builder")]


def test_tree_mode_refuses_a_missing_root_and_a_file_that_is_not_text(tmp_path):
    completed = subprocess.run([sys.executable, str(SCAN), "--regime", "frozen", "--tree", str(tmp_path / "nowhere")],
                               capture_output=True, text=True)
    assert completed.returncode != 0 and "no directory to scan" in completed.stderr
    (tmp_path / "bad.py").write_bytes(b"\xff\xfe not utf-8")
    completed = subprocess.run([sys.executable, str(SCAN), "--regime", "frozen", "--tree", str(tmp_path)], capture_output=True, text=True)
    assert completed.returncode != 0 and "UnicodeDecodeError" in completed.stderr
