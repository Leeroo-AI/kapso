"""Run the invariants of a spec against a dbt model through `dbt show`.

Usage:
  invariants.py <spec.json> [--project-dir DIR] [--profiles-dir DIR] [--target NAME] [--out results.json]

Every check becomes one SQL statement that returns a count of violating rows,
or two scalars for a reconciliation; the statements go through dbt's Jinja, so
`ref`, `source` and `target.type` work. A query must return exactly one row
with one non-null value; a violation count must be a finite non-negative
integer; reconciliations compare exact decimals. Check names are unique within
a spec, since a report is read by name. Exit status 1 when any check fails or
errors; an error when the spec yields no check. The report carries the
measurement context (project, target, code revision) and the SHA-256 of the
spec file, so two runs can be shown to be of one spec.

All the scalars of a spec are asked for in one statement, since a dbt start
on a large project costs more than the queries; when that statement fails
(one check's SQL is wrong) every scalar is asked for on its own, so the check
at fault is named and the others are still measured.

Date arithmetic goes through dbt's own cross-database macro. The test for a
non-finite float has no such macro: it is written for Snowflake and DuckDB,
the engines this script was run on, and any other adapter is refused by name
rather than given SQL that may not mean the same. Column names go into the SQL
as the spec writes them, so a quoted mixed-case column is quoted in the spec.
Standard library only; dbt on PATH.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

NUMERIC_MARKERS = ("int", "decimal", "numeric", "double", "float", "real", "number", "hugeint")
FLOAT_MARKERS = ("double", "float", "real")
NUMBER = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")
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
    rows = json.JSONDecoder(parse_float=Decimal).raw_decode(output[start:])[0]["show"]   # a float would round a large scalar
    return [{str(key).lower(): value for key, value in row.items()} for row in rows], None


def scalar(sql, project_dir, profiles_dir, target):
    """(value, error): exactly one row with one non-null value, or why not."""
    rows, error = show(sql, project_dir, profiles_dir, target, limit=5)
    if error:
        return None, error
    if len(rows) != 1 or len(rows[0]) != 1:
        return None, f"expected one row with one value, got {len(rows)} row(s) with {len(rows[0]) if rows else 0} column(s)"
    value = next(iter(rows[0].values()))
    if value is None:
        return None, "the query returned NULL"
    return value, None


def as_decimal(value):
    """A finite Decimal from a scalar, or None when it is not a plain number."""
    if isinstance(value, bool):
        return None
    text = str(value).strip()
    if not NUMBER.match(text):
        return None
    return Decimal(text)


def relation(model):
    return "{{ ref('%s') }}" % model


def model_relation(model, project_dir, profiles_dir, target):
    """(database, schema, identifier) of the one model with this name."""
    code, output = run_dbt(["ls", "--select", model, "--resource-type", "model", "snapshot", "--output", "json",
                            "--output-keys", "name", "database", "schema", "config", "unique_id"], project_dir, profiles_dir, target)
    if code != 0:
        raise SystemExit(f"dbt ls failed for {model!r}: {error_tail(output)}")
    rows = [json.loads(line.strip()) for line in output.splitlines() if line.strip().startswith("{")]
    matches = [row for row in rows if row.get("name") == model]
    if not matches:
        raise SystemExit(f"dbt ls does not list a model or snapshot named {model!r}")
    if len(matches) > 1:
        raise SystemExit(f"{model!r} is ambiguous ({', '.join(m['unique_id'] for m in matches)}); resolve the package or version")
    row = matches[0]
    return row.get("database"), row.get("schema"), (row.get("config") or {}).get("alias") or model


def numeric_columns(model, project_dir, profiles_dir, target):
    database, schema, table = model_relation(model, project_dir, profiles_dir, target)
    predicates = [f"lower(table_schema) = '{schema.lower()}'", f"lower(table_name) = '{table.lower()}'"]
    if database:
        predicates.insert(0, f"lower(table_catalog) = '{database.lower()}'")
    rows, error = show("select column_name, data_type from information_schema.columns where " + " and ".join(predicates),
                       project_dir, profiles_dir, target, limit=1000)
    if error:
        raise SystemExit(f"column metadata unavailable for {model}: {error}. Name the finite columns in the spec instead of \"auto\".")
    if not rows:
        raise SystemExit(f"no column metadata for {model} ({database}.{schema}.{table}): not built, or another catalog. "
                         "Name the finite columns in the spec instead of \"auto\".")
    numeric = []
    for row in rows:
        data_type = str(row.get("data_type") or "").lower()
        if any(marker in data_type for marker in NUMERIC_MARKERS):
            numeric.append((row["column_name"], any(marker in data_type for marker in FLOAT_MARKERS)))
    return numeric


def finite_predicate(column):
    return (
        "{% if target.type == 'snowflake' %}"
        f"cast({column} as varchar) in ('NaN', 'inf', '-inf', 'Infinity', '-Infinity')"
        "{% elif target.type == 'duckdb' %}"
        f"(isnan({column}) or isinf({column}))"
        "{% else %}"
        "{{ exceptions.raise_compiler_error('the finite check is written for snowflake and duckdb, not for ' ~ target.type"
        " ~ ': state it as a custom check in the SQL of this engine') }}"
        "{% endif %}"
    )


def checks_from(spec, project_dir, profiles_dir, target):
    model = spec["model"]
    rel = relation(model)
    checks = []
    grain = spec.get("grain") or []
    if grain:
        keys = ", ".join(grain)
        checks.append(("grain unique (%s)" % keys, "violations",
                       f"select count(*) as n from (select {keys}, count(*) as c from {rel} group by {keys} having count(*) > 1) d"))
    for column in spec.get("not_null") or []:
        checks.append((f"not null {column}", "violations", f"select count(*) as n from {rel} where {column} is null"))
    for column in spec.get("non_negative") or []:
        checks.append((f"non-negative {column}", "violations", f"select count(*) as n from {rel} where {column} < 0"))
    for column, (low, high) in (spec.get("bounds") or {}).items():
        checks.append((f"bounds {column} in [{low}, {high}]", "violations",
                       f"select count(*) as n from {rel} where {column} < {low} or {column} > {high}"))
    finite = spec.get("finite")
    if finite == "auto":
        # A name read from the engine's metadata is the stored name: quoted the adapter's way it is that
        # column, whatever its case. A name the spec writes goes into the SQL as written.
        columns = [(c, '{{ adapter.quote("%s") }}' % c.replace("\\", "\\\\").replace('"', '\\"'))
                   for c, is_float in numeric_columns(model, project_dir, profiles_dir, target) if is_float]
    else:
        columns = [(c, c) for c in finite or []]
    for column, written in columns:
        checks.append((f"finite {column}", "violations", f"select count(*) as n from {rel} where {finite_predicate(written)}"))
    for item in spec.get("monotonic") or []:
        partition = ", ".join(item.get("partition_by") or [])
        over = (f"partition by {partition} " if partition else "") + f"order by {item['order_by']}"
        direction = item.get("direction", "increasing")
        comparison = "<" if direction == "increasing" else ">"
        checks.append((f"monotonic {item['column']} ({direction} by {item['order_by']})", "violations",
                       f"select count(*) as n from (select {item['column']} as v, lag({item['column']}) over ({over}) as prev from {rel}) d "
                       f"where prev is not null and v {comparison} prev"))
    for item in spec.get("no_gaps") or []:
        partition = ", ".join(item.get("partition_by") or [])
        keys = f"{item['column']}" + (f", {partition}" if partition else "")
        over = (f"partition by {partition} " if partition else "") + f"order by {item['column']}"
        checks.append((f"no interior daily gaps in {item['column']}", "violations",
                       f"select count(*) as n from (select {item['column']} as d, lead({item['column']}) over ({over}) as nxt "
                       f"from (select distinct {keys} from {rel}) u) g where nxt is not null and "
                       "{{ dbt.datediff('d', 'nxt', 'day') }} > 1"))
    for item in spec.get("completeness") or []:
        model_key, source_key = item["model_key"], item["source_key"]
        where = f" where {item['source_filter']}" if item.get("source_filter") else ""
        checks.append((item.get("name", f"completeness {model_key}"), "violations",
                       f"select count(*) as n from (select distinct {source_key} as k from {item['source']}{where}) s "
                       f"left join (select distinct {model_key} as k from {rel}) m on s.k = m.k where m.k is null"))
    for item in spec.get("custom") or []:
        checks.append((item["name"], "violations", item["violations"]))
    for item in spec.get("reconcile") or []:
        checks.append((item["name"], "reconcile", item))
    return checks


def judge_violations(name, sql, value, error):
    """One violation-count check from its scalar, or from why there is none."""
    if error:
        return {"name": name, "status": "error", "detail": error[:600], "sql": sql}
    count = as_decimal(value)
    if count is None or not count.is_finite() or count < 0 or count != count.to_integral_value():
        return {"name": name, "status": "error", "detail": f"violation count must be a finite non-negative integer, got {value!r}", "sql": sql}
    count = int(count)
    return {"name": name, "status": "pass" if count == 0 else "fail", "violations": count, "sql": sql}


def judge_reconcile(name, payload, left, left_error, right, right_error):
    """One reconciliation from its two scalars, or from why one is missing."""
    sql = [payload["left"], payload["right"]]
    if left_error:
        return {"name": name, "status": "error", "detail": f"left: {left_error[:500]}", "sql": sql}
    if right_error:
        return {"name": name, "status": "error", "detail": f"right: {right_error[:500]}", "sql": sql}
    values = [as_decimal(left), as_decimal(right), as_decimal(payload.get("tolerance", 0)), as_decimal(payload.get("relative_tolerance", 0))]
    if any(v is None or not v.is_finite() for v in values) or values[2] < 0 or values[3] < 0:
        return {"name": name, "status": "error", "detail": f"non-numeric or non-finite values: left {left!r}, right {right!r}, "
                                                            f"tolerance {payload.get('tolerance', 0)!r}, relative {payload.get('relative_tolerance', 0)!r}", "sql": sql}
    # Fractions hold every finite decimal exactly, whatever its size: no context precision can round a difference.
    left_value, right_value, tolerance, relative = (Fraction(value) for value in values)
    difference = abs(left_value - right_value)
    tolerance = max(tolerance, relative * max(abs(left_value), abs(right_value)))
    return {"name": name, "status": "pass" if difference <= tolerance else "fail",
            "detail": f"left {values[0]}, right {values[1]}, difference {shown(difference)}, tolerance {shown(tolerance)}", "sql": sql}


def shown(value):
    """A fraction for a reader: its decimal expansion to forty significant digits."""
    with localcontext() as wide:
        wide.prec = 40
        return str((Decimal(value.numerator) / Decimal(value.denominator)).normalize())


NULL_SCALAR = "the query returned NULL"


def scalars_in_one_statement(statements, project_dir, profiles_dir, target):
    """(values by key, error): every scalar of the spec from one statement. A NULL comes back as None."""
    parts = [f"select '{key}' as scalar_key, cast(({sql.strip().rstrip(';')}) as varchar) as scalar_value" for key, sql in statements]
    rows, error = show(" union all ".join(parts), project_dir, profiles_dir, target, limit=len(parts) + 5)
    if error:
        return None, error
    values = {str(row.get("scalar_key")): row.get("scalar_value") for row in rows}
    if set(values) != {key for key, _ in statements}:
        return None, f"the statement returned {len(values)} of the {len(statements)} scalars asked for"
    return values, None


def run_checks(checks, project_dir, profiles_dir, target):
    """(results, how they were run): one statement for all, or every scalar alone when that statement fails."""
    statements = []
    for index, (_, kind, payload) in enumerate(checks):
        if kind == "violations":
            statements.append((f"c{index}", payload))
        else:
            statements += [(f"c{index}l", payload["left"]), (f"c{index}r", payload["right"])]
    started = time.time()
    together, statement_error = scalars_in_one_statement(statements, project_dir, profiles_dir, target)
    results = []
    for index, (name, kind, payload) in enumerate(checks):
        if together is None and kind == "violations":
            value, error = scalar(payload, project_dir, profiles_dir, target)
            result = judge_violations(name, payload, value, error)
        elif together is None:
            left, left_error = scalar(payload["left"], project_dir, profiles_dir, target)
            right, right_error = (None, None) if left_error else scalar(payload["right"], project_dir, profiles_dir, target)
            result = judge_reconcile(name, payload, left, left_error, right, right_error)
        elif kind == "violations":
            value = together[f"c{index}"]
            result = judge_violations(name, payload, value, NULL_SCALAR if value is None else None)
        else:
            left, right = together[f"c{index}l"], together[f"c{index}r"]
            result = judge_reconcile(name, payload, left, NULL_SCALAR if left is None else None,
                                     right, NULL_SCALAR if right is None else None)
        results.append(result)
    how = {"statements": 1 if together is not None else len(statements), "seconds": round(time.time() - started, 1)}
    if together is None:
        how["one_statement_error"] = statement_error[:300]
    return results, how


def measurement_context(project_dir, target):
    project = Path(project_dir).resolve()
    head = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"], capture_output=True, text=True)
    return {"project_dir": str(project), "target": target or "profile default",
            "git_head": head.stdout.strip() if head.returncode == 0 else None,
            "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec")
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--profiles-dir", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    with open(args.spec, encoding="utf-8") as handle:
        spec = json.load(handle, parse_float=Decimal)   # a tolerance is not rounded through a float
    if not isinstance(spec, dict) or "model" not in spec:
        raise SystemExit("the spec needs a `model`")
    checks = checks_from(spec, args.project_dir, args.profiles_dir, args.target)
    if not checks:
        raise SystemExit("the spec yields no check; a dataset without a declared grain or a reconciliation is unmeasured")
    names = [name for name, _, _ in checks]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise SystemExit("every check needs a non-empty name")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise SystemExit(f"check names must be unique within a spec, since a report is read by name: {', '.join(repeated)}")
    results, how = run_checks(checks, args.project_dir, args.profiles_dir, args.target)
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    errored = sum(1 for r in results if r["status"] == "error")
    print(f"model {spec['model']}: {passed} pass, {failed} fail, {errored} error")
    for result in results:
        evidence = result.get("detail") if "detail" in result else f"{result.get('violations')} violating row(s)"
        print(f"  {result['status']:<5} {result['name']:<48} {evidence}")
    with open(args.spec, "rb") as handle:
        spec_sha256 = hashlib.sha256(handle.read()).hexdigest()
    report = {"tool": "invariants.py", "model": spec["model"], "spec": args.spec, "spec_sha256": spec_sha256,
              "context": measurement_context(args.project_dir, args.target), "run": how,
              "passed": passed, "failed": failed, "errored": errored, "results": results}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"written {args.out}")
    sys.exit(1 if failed or errored else 0)


if __name__ == "__main__":
    main()
