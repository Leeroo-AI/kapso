"""Fingerprint every dbt model and snapshot a selector lists (row count, rounded
numeric sums, distinct counts of the other columns) through `dbt show`, save
the result with its measurement context, compare two saved results, or count
the rows of one relation absent from another in both directions.
Standard library only; dbt on PATH.

Usage:
  fingerprint.py --select "<model>+" [--project-dir DIR] [--profiles-dir DIR] [--target NAME] --save out.json
  fingerprint.py --compare before.json after.json [--json]
  fingerprint.py --except "<relation a>" "<relation b>" [--project-dir DIR] [--profiles-dir DIR] [--target NAME] [--save out.json]

A fingerprint is a screen: unequal fingerprints show a change, equal ones do
not show equality (two rows may swap values). A dataset whose measurement
failed is recorded with its error and makes a comparison an error, never
"unchanged"; one that has no relation on the target is recorded as absent
only when the adapter itself says so, and then compares as unchanged against
another absence and as a change against a relation; counts are compared
exactly and a non-finite sum always differs.

`--except` is a set comparison: it counts the rows of each relation that the
other does not hold, with both row counts, and exits 1 when either direction
has rows or the row counts differ. It shows row equality only when the rows
are unique (a declared grain whose uniqueness check passes on both sides);
duplicates on one side are not seen by it.

The datasets of a selection are measured together: one statement reads every
dataset's columns, then one statement per group returns each fingerprint as a
JSON text, because a dbt start on a large project costs more than the queries.
The JSON form is written for Snowflake and DuckDB, the engines this script was
run on; on another adapter, or when a group's statement fails (a column type
that cannot be counted), every dataset is measured on its own with plain
aggregates, so the dataset at fault is the one recorded with an error.
"""
import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

NUMERIC_MARKERS = ("int", "decimal", "numeric", "double", "float", "real", "number", "hugeint")
FLOAT_TYPE = "{{ dbt.type_float() }}"   # the adapter's own name for a float
ABSENT = "no relation of this name on the target"
GROUP_SIZE = 20
COLUMN_ROWS_LIMIT = 200000
RELATIVE_TOLERANCE = 1e-9
ANSI = re.compile(r"\x1b\[[0-9;]*m")
INTERESTING = re.compile(r"error|not found|does not exist|Binder|Catalog|Compilation|Runtime|Database|Parser", re.IGNORECASE)


def error_tail(output):
    """dbt's own error lines, without its warnings and colour codes."""
    lines = [ANSI.sub("", line).strip() for line in output.splitlines()]
    hits = [line for line in lines if line and INTERESTING.search(line) and "deprecat" not in line.lower()]
    tail = hits[-6:] if hits else [line for line in lines if line][-4:]
    return " | ".join(tail)[:700]


def run_dbt(args, project_dir, profiles_dir, target):
    project = str(Path(project_dir).resolve())
    command = ["dbt", "--no-use-colors", *args, "--project-dir", project]
    if profiles_dir:
        command += ["--profiles-dir", str(Path(profiles_dir).resolve())]
    if target:
        command += ["--target", target]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=project)
    return completed.returncode, completed.stdout + completed.stderr


def show(sql, project_dir, profiles_dir, target, limit=1000):
    """(rows, error): the rows of one query with lower-cased keys, or dbt's own last lines."""
    code, output = run_dbt(["show", "--inline", sql, "--output", "json", "--limit", str(limit)], project_dir, profiles_dir, target)
    if code != 0:
        return None, f"dbt show exited {code}: {error_tail(output)}"
    index = output.find('"show"')
    if index < 0:
        return None, f"dbt show printed no result block: {error_tail(output)}"
    start = output.rfind("{", 0, index)
    rows = json.JSONDecoder().raw_decode(output[start:])[0]["show"]
    return [{str(key).lower(): value for key, value in row.items()} for row in rows], None


def list_models(selector, project_dir, profiles_dir, target):
    """(name, database, schema, relation name) of every model and snapshot the selector lists."""
    code, output = run_dbt(["ls", "--select", *selector.split(), "--resource-type", "model", "snapshot", "--output", "json",
                            "--output-keys", "name", "database", "schema", "config", "unique_id"], project_dir, profiles_dir, target)
    if code != 0:
        raise SystemExit(f"dbt ls failed for `{selector}`: {error_tail(output)}")
    rows = [json.loads(line.strip()) for line in output.splitlines() if line.strip().startswith("{")]
    if not rows:
        raise SystemExit(f"dbt ls lists no model or snapshot for `{selector}`:\n{output[-800:]}")
    names = [row["name"] for row in rows]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SystemExit(f"ambiguous names in the selection ({', '.join(duplicates)}); resolve the package or version")
    return [(row["name"], row.get("database"), row.get("schema"), (row.get("config") or {}).get("alias") or row["name"]) for row in rows]


def columns_of(database, schema, table, project_dir, profiles_dir, target):
    """(columns, error): (name, lower-cased type) in ordinal order from the information schema."""
    predicates = [f"lower(table_schema) = '{schema.lower()}'", f"lower(table_name) = '{table.lower()}'"]
    if database:
        predicates.insert(0, f"lower(table_catalog) = '{database.lower()}'")
    rows, error = show("select column_name, data_type from information_schema.columns where " + " and ".join(predicates)
                       + " order by ordinal_position", project_dir, profiles_dir, target)
    if error:
        return None, error
    return [(row["column_name"], str(row.get("data_type") or "").lower()) for row in rows], None


def without_metadata(models, project_dir, profiles_dir, target):
    """What to record for datasets whose columns the information schema did not list: absent when the adapter
    itself says no such relation exists, an error otherwise. Empty metadata alone proves nothing: the
    relation may sit in a catalog this connection cannot list."""
    parts = ["select '%s' as model_name, '{{ adapter.get_relation(database=%s, schema='%s', identifier='%s') is not none }}' as present"
             % (name, f"'{database}'" if database else "target.database", schema, table) for name, database, schema, table in models]
    rows, error = show(" union all ".join(parts), project_dir, profiles_dir, target, limit=len(parts) + 5)
    found = {} if error else {row["model_name"]: str(row["present"]).strip().lower() for row in rows}
    report = {}
    for name, database, schema, table in models:
        where = f"{database}.{schema}.{table}"
        if found.get(name) == "false":
            report[name] = {"rows": None, "columns": {}, "absent": True, "error": f"{ABSENT}: {where}"}
        elif found.get(name) == "true":
            report[name] = {"rows": None, "columns": {}, "error": f"{where} exists and its columns could not be listed (another catalog, or no privilege)"}
        else:
            report[name] = {"rows": None, "columns": {}, "error": f"whether {where} exists could not be established: {(error or 'no answer')[:300]}"}
    return report


def fingerprint(model, database, schema, table, project_dir, profiles_dir, target):
    columns, error = columns_of(database, schema, table, project_dir, profiles_dir, target)
    if error:
        return {"rows": None, "columns": {}, "error": error[:400]}
    if not columns:
        return without_metadata([(model, database, schema, table)], project_dir, profiles_dir, target)[model]
    parts = [f"{expression} as {quoted(key)}" for key, expression in aggregates(columns)]
    rows, error = show("select " + ", ".join(parts) + " from {{ ref('%s') }}" % model, project_dir, profiles_dir, target, limit=5)
    if error:
        return {"rows": None, "columns": {}, "error": error[:400]}
    if len(rows) != 1 or rows[0].get("row_count") is None:
        return {"rows": None, "columns": {}, "error": f"the fingerprint query returned {len(rows)} row(s) without a row count"}
    row = rows[0]
    return {"rows": row["row_count"], "columns": {k: v for k, v in row.items() if k != "row_count"}}


def columns_of_all(models, project_dir, profiles_dir, target):
    """({model: [(column, lower-cased type), ...]}, error): every listed model's columns from one statement."""
    clauses = []
    for _, database, schema, table in models:
        clause = f"(lower(table_schema) = '{schema.lower()}' and lower(table_name) = '{table.lower()}'"
        clauses.append(clause + (f" and lower(table_catalog) = '{database.lower()}')" if database else ")"))
    rows, error = show("select lower(table_catalog) as catalog_name, lower(table_schema) as schema_name, lower(table_name) as relation_name, "
                       "column_name, data_type from information_schema.columns where " + " or ".join(clauses) + " order by 1, 2, 3, ordinal_position",
                       project_dir, profiles_dir, target, limit=COLUMN_ROWS_LIMIT)
    if error:
        return None, error
    found = {}
    for row in rows:
        found.setdefault((row["catalog_name"], row["schema_name"], row["relation_name"]), []).append((row["column_name"], str(row.get("data_type") or "").lower()))
    listed = {}
    for name, database, schema, table in models:
        # A relation is its catalog, schema and name; without a catalog it is the one the connection lists.
        matches = [columns for (catalog, schema_name, relation_name), columns in found.items()
                   if schema_name == schema.lower() and relation_name == table.lower() and (not database or catalog == database.lower())]
        listed[name] = matches[0] if len(matches) == 1 else []
    return listed, None


def quoted(name):
    """A name quoted the adapter's way. A column name read from the engine's own metadata is the stored name,
    so quoted it is the column it is, whatever its case and even when it is a reserved word."""
    return '{{ adapter.quote("%s") }}' % name.replace("\\", "\\\\").replace('"', '\\"')


def aggregates(columns):
    """(key, SQL expression) for every measure of one model's fingerprint."""
    pairs = [("row_count", "count(*)")]
    for name, data_type in columns:
        if any(marker in data_type for marker in NUMERIC_MARKERS):
            pairs.append((f"sum__{name}".lower(), f"round(sum(cast({quoted(name)} as {FLOAT_TYPE})), 4)"))
        else:
            pairs.append((f"nd__{name}".lower(), f"count(distinct {quoted(name)})"))
    return pairs


def as_json_text(pairs):
    """Jinja that renders, on the engine of the target, one JSON text holding the named values; an engine
    without a form here raises, and the group is then measured dataset by dataset."""
    listed = ", ".join("'%s', %s" % (key.replace("'", "''"), expression) for key, expression in pairs)
    return ("{% if target.type == 'snowflake' %}to_json(object_construct_keep_null(" + listed + "))"
            "{% elif target.type == 'duckdb' %}cast(json_object(" + listed + ") as varchar)"
            "{% else %}{{ exceptions.raise_compiler_error('no one-statement fingerprint for ' ~ target.type) }}{% endif %}")


def fingerprint_group(group, columns, project_dir, profiles_dir, target):
    """({model: measurement}, error): the fingerprints of a group of models from one statement."""
    parts = ["select '%s' as model_name, %s as fingerprint from {{ ref('%s') }}" % (name, as_json_text(aggregates(columns[name])), name)
             for name, _, _, _ in group]
    rows, error = show(" union all ".join(parts), project_dir, profiles_dir, target, limit=len(parts) + 5)
    if error:
        return None, error
    measured = {}
    for row in rows:
        values = {str(key).lower(): value for key, value in json.loads(row["fingerprint"]).items()}
        measured[row["model_name"]] = {"rows": values.get("row_count"), "columns": {k: v for k, v in values.items() if k != "row_count"}}
    if set(measured) != {name for name, _, _, _ in group}:
        return None, f"the statement returned {len(measured)} of the {len(group)} models asked for"
    return measured, None


def fingerprint_all(models, project_dir, profiles_dir, target):
    """Every model's measurement, groups first and model by model where a group's statement fails."""
    columns, error = columns_of_all(models, project_dir, profiles_dir, target)
    if error:
        return {name: fingerprint(name, database, schema, table, project_dir, profiles_dir, target) for name, database, schema, table in models}
    known = [model for model in models if columns[model[0]]]
    unlisted = [model for model in models if not columns[model[0]]]
    report = without_metadata(unlisted, project_dir, profiles_dir, target) if unlisted else {}
    for start in range(0, len(known), GROUP_SIZE):
        group = known[start:start + GROUP_SIZE]
        measured, error = fingerprint_group(group, columns, project_dir, profiles_dir, target)
        if error:
            measured = {name: fingerprint(name, database, schema, table, project_dir, profiles_dir, target) for name, database, schema, table in group}
        report.update(measured)
    return {name: report[name] for name, _, _, _ in models}


def differs(a, b):
    if a is None or b is None:
        return a is not b
    if isinstance(a, bool) or isinstance(b, bool):
        return a != b
    for value in (a, b):
        if isinstance(value, float) and not math.isfinite(value):
            return True
    if isinstance(a, int) and isinstance(b, int):
        return a != b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) > RELATIVE_TOLERANCE * max(abs(a), abs(b), 1.0)
    return a != b


def compare(before, after):
    for state, report in (("before", before), ("after", after)):
        for model, measurement in report["models"].items():
            if not measurement.get("absent") and (measurement.get("error") or measurement.get("rows") is None):
                raise SystemExit(f"the {state} measurement of {model} is unavailable ({measurement.get('error') or 'no row count'}); "
                                 "an unmeasured dataset is never \"unchanged\"")
    changed = []
    for model in sorted(set(before["models"]) & set(after["models"])):
        left, right = before["models"][model], after["models"][model]
        if left.get("absent") or right.get("absent"):   # shown absent on both sides is unchanged; on one side, a change
            if bool(left.get("absent")) != bool(right.get("absent")):
                changed.append({"model": model, "rows_before": "no relation" if left.get("absent") else left["rows"],
                                "rows_after": "no relation" if right.get("absent") else right["rows"], "columns": []})
            continue
        left_columns, right_columns = left.get("columns", {}), right.get("columns", {})
        moved = sorted(key for key in set(left_columns) | set(right_columns)
                       if key not in left_columns or key not in right_columns or differs(left_columns[key], right_columns[key]))
        if moved or differs(left["rows"], right["rows"]):
            changed.append({"model": model, "rows_before": left["rows"], "rows_after": right["rows"], "columns": moved})
    return {
        "changed": changed,
        "unchanged": sorted(m for m in set(before["models"]) & set(after["models"]) if m not in {c["model"] for c in changed}),
        "added": sorted(set(after["models"]) - set(before["models"])),
        "removed": sorted(set(before["models"]) - set(after["models"])),
        "context_before": before.get("context"),
        "context_after": after.get("context"),
    }


def except_counts(relation_a, relation_b, project_dir, profiles_dir, target):
    """Rows of a absent from b and of b absent from a by whole-row comparison, with both row counts. A count
    that did not come back as one non-negative integer stops the script: an unknown is never zero."""
    statements = (("only_in_a", f"select count(*) as n from (select * from {relation_a} except select * from {relation_b}) d"),
                  ("only_in_b", f"select count(*) as n from (select * from {relation_b} except select * from {relation_a}) d"),
                  ("rows_a", f"select count(*) as n from {relation_a}"), ("rows_b", f"select count(*) as n from {relation_b}"),
                  ("distinct_a", f"select count(*) as n from (select distinct * from {relation_a}) d"),
                  ("distinct_b", f"select count(*) as n from (select distinct * from {relation_b}) d"))
    counts = {}
    for label, sql in statements:
        rows, error = show(sql, project_dir, profiles_dir, target, limit=5)
        if error:
            raise SystemExit(f"{label}: {error}")
        value = rows[0].get("n") if len(rows) == 1 else None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SystemExit(f"{label}: the count query returned {len(rows)} row(s) and no count; the comparison was not taken")
        counts[label] = value
    return counts


def measurement_context(project_dir, target):
    project = Path(project_dir).resolve()
    head = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"], capture_output=True, text=True)
    return {"project_dir": str(project), "target": target or "profile default",
            "git_head": head.stdout.strip() if head.returncode == 0 else None,
            "taken_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--select", default=None)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--profiles-dir", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), default=None)
    parser.add_argument("--except", dest="except_pair", nargs=2, metavar=("A", "B"), default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.compare:
        with open(args.compare[0], encoding="utf-8") as handle:
            before = json.load(handle)
        with open(args.compare[1], encoding="utf-8") as handle:
            after = json.load(handle)
        result = compare(before, after)
        if args.json:
            json.dump(result, sys.stdout, indent=2)
            print()
            return
        print(f"changed {len(result['changed'])}, unchanged {len(result['unchanged'])}, added {len(result['added'])}, removed {len(result['removed'])}")
        for item in result["changed"]:
            print(f"  {item['model']}: rows {item['rows_before']} -> {item['rows_after']}; columns moved: {', '.join(item['columns']) or '-'}")
        for model in result["added"]:
            print(f"  added {model}")
        for model in result["removed"]:
            print(f"  removed {model}")
        return

    if args.except_pair:
        counts = except_counts(args.except_pair[0], args.except_pair[1], args.project_dir, args.profiles_dir, args.target)
        print(f"rows in a: {counts['rows_a']}, in b: {counts['rows_b']}; only in a: {counts['only_in_a']}; only in b: {counts['only_in_b']}")
        if counts["rows_a"] != counts["distinct_a"] or counts["rows_b"] != counts["distinct_b"]:
            print(f"duplicate rows: a holds {counts['rows_a'] - counts['distinct_a']}, b {counts['rows_b'] - counts['distinct_b']}; "
                  "a set comparison does not count them, so equal sets here do not show equal rows")
        if args.save:
            with open(args.save, "w", encoding="utf-8") as handle:
                json.dump({"tool": "fingerprint.py except", "a": args.except_pair[0], "b": args.except_pair[1], **counts,
                           "context": measurement_context(args.project_dir, args.target)}, handle, indent=2)
                handle.write("\n")
            print(f"written {args.save}")
        sys.exit(1 if (counts["only_in_a"] or counts["only_in_b"] or counts["rows_a"] != counts["rows_b"]) else 0)

    if not args.select or not args.save:
        raise SystemExit("one of: --compare BEFORE AFTER; --except A B; --select SELECTOR with --save PATH")
    models = list_models(args.select, args.project_dir, args.profiles_dir, args.target)
    report = {"tool": "fingerprint.py", "selector": args.select, "context": measurement_context(args.project_dir, args.target),
              "models": fingerprint_all(models, args.project_dir, args.profiles_dir, args.target)}
    for name, measurement in report["models"].items():
        print(f"  {name:<48} " + (f"error: {measurement['error'][:120]}" if measurement.get("error") else f"rows {measurement['rows']}"))
    with open(args.save, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    errors = sum(1 for m in report["models"].values() if m.get("error"))
    print(f"written {args.save} ({len(models)} models, {errors} unmeasured)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
