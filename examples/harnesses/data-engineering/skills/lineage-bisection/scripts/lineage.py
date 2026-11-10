"""The upstream (or downstream) tree of a dbt model from `dbt ls`, with the
materialization and file of each node and, on request, the row counts of the
model and its direct parents through `dbt show`. Standard library only; dbt
on PATH; run from the project directory or pass --project-dir.

Usage:
  lineage.py <model> [--project-dir DIR] [--profiles-dir DIR] [--target NAME] [--counts] [--downstream] [--json]

An ambiguous model name (two packages or versions) is refused. Counts are
taken only for materialized relations: models, sources, seeds and snapshots
that are not ephemeral; tests and ephemeral nodes stay graph entries.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

LS_KEYS = ["unique_id", "name", "resource_type", "depends_on", "config", "schema", "original_file_path", "source_name"]
COUNTABLE = ("model", "source", "seed", "snapshot")
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


def ls_json(selector, project_dir, profiles_dir, target):
    code, output = run_dbt(["ls", "--select", selector, "--output", "json", "--output-keys", *LS_KEYS], project_dir, profiles_dir, target)
    rows = [json.loads(line.strip()) for line in output.splitlines() if line.strip().startswith("{")]
    if code != 0:
        raise SystemExit(f"dbt ls failed (exit {code}): {error_tail(output)}")
    return rows


def show_scalar(sql, project_dir, profiles_dir, target):
    code, output = run_dbt(["show", "--inline", sql, "--output", "json", "--limit", "10"], project_dir, profiles_dir, target)
    index = output.find('"show"')
    if code != 0 or index < 0:
        return f"error: {error_tail(output)}"
    start = output.rfind("{", 0, index)
    rows = json.JSONDecoder().raw_decode(output[start:])[0]["show"]
    return next(iter(rows[0].values())) if rows else None


def relation(node):
    if node["resource_type"] == "source":
        return "{{ source('%s', '%s') }}" % (node.get("source_name"), node["name"])
    return "{{ ref('%s') }}" % node["name"]


def countable(node):
    return node["resource_type"] in COUNTABLE and (node.get("config") or {}).get("materialized") != "ephemeral"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--profiles-dir", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--counts", action="store_true")
    parser.add_argument("--downstream", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    selector = f"{args.model}+" if args.downstream else f"+{args.model}"
    nodes = {row["unique_id"]: row for row in ls_json(selector, args.project_dir, args.profiles_dir, args.target)}
    roots = [uid for uid, row in nodes.items() if row["name"] == args.model and row["resource_type"] in ("model", "snapshot", "seed")]
    if not roots:
        raise SystemExit(f"no model named {args.model!r} in the selection `{selector}`")
    if len(roots) > 1:
        raise SystemExit(f"{args.model!r} is ambiguous ({', '.join(roots)}); resolve the package or version")
    root = roots[0]

    children = {uid: [] for uid in nodes}
    for uid, row in nodes.items():
        for parent in (row.get("depends_on") or {}).get("nodes", []):
            if parent in nodes:
                children[parent].append(uid)

    # Upstream: walk depends_on from the root; downstream: walk the children.
    def neighbours(uid):
        if args.downstream:
            return sorted(children.get(uid, []))
        return sorted(p for p in (nodes[uid].get("depends_on") or {}).get("nodes", []) if p in nodes)

    lines = []
    tree = []
    seen = set()

    def walk(uid, depth):
        node = nodes[uid]
        materialized = (node.get("config") or {}).get("materialized") or ""
        entry = {"depth": depth, "name": node["name"], "type": node["resource_type"], "materialized": materialized,
                 "schema": node.get("schema"), "file": node.get("original_file_path")}
        tree.append(entry)
        marker = " (seen)" if uid in seen else ""
        lines.append(f"{'  ' * depth}{node['name']}  [{node['resource_type']}{', ' + materialized if materialized else ''}]  {node.get('original_file_path') or ''}{marker}")
        if uid in seen:
            return
        seen.add(uid)
        for nxt in neighbours(uid):
            walk(nxt, depth + 1)

    walk(root, 0)

    counts = {}
    if args.counts:
        for uid in [root] + neighbours(root):
            node = nodes[uid]
            if countable(node):
                counts[node["name"]] = show_scalar(f"select count(*) as n from {relation(node)}", args.project_dir, args.profiles_dir, args.target)

    if args.json:
        json.dump({"model": args.model, "direction": "downstream" if args.downstream else "upstream", "nodes": len(nodes), "tree": tree, "counts": counts}, sys.stdout, indent=2)
        print()
        return
    print(f"{'downstream' if args.downstream else 'upstream'} of {args.model}: {len(nodes)} nodes")
    print("\n".join(lines))
    if counts:
        print("row counts (the model and its direct " + ("children" if args.downstream else "parents") + ", materialized relations only):")
        for name, value in counts.items():
            print(f"  {name:<48} {value}")


if __name__ == "__main__":
    main()
