"""Render a pull-request title and body from one ticket's proof and the evidence it names.

Usage: render_proof.py <proof.json> [--verification FILE] [--branch-check pass|fail|not_run]
                       [--title-out PATH] [--body-out PATH] [--label-out PATH]

The title is the change in a reviewer's words (`<ticket>: <the cause in one sentence>`) and carries no tag. The
verdict is the first thing the body says, in a sentence: verified, or the points that are still open, each
named. It is computed here, never copied from the proof. When verification measured the ticket again (the
report `evidence.py verify` wrote, given with --verification, names that measurement), the verdict and every
number in the body come from verification's measurement of the delivered code, computed now against the
files as they are here: the build session's own measurement is history by then. A change is stated as
verified only when that measurement earns `safe to merge`, nothing verification recorded lowers it (a
reviewer's finding does), and the branch the pull request carries passed its own check (--branch-check,
which the pull-request target passes; without it a preview carries that as an open point). Without the
report, or for a ticket verification did not measure again, the build session's measurement is shown and
the change is never stated as verified. --label-out writes the label itself and the open points as JSON,
for the deployment's own record; the pull request's text uses neither label word.

The body is the proof in a reviewer's order: the verdict and why, what was measured (the build against what
production holds, the checks that moved and the ones preserved, what reads the changed datasets), the
validation list, the cause, the runbook of a changed load, the files and the open questions. Run it from the
workspace root, where results/ is. Standard library only.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HARNESS = Path(os.environ["KAPSO_HARNESS_ROOT"]) if os.environ.get("KAPSO_HARNESS_ROOT") else Path(__file__).resolve().parents[3]
EVIDENCE = HARNESS / "skills" / "validation-report" / "scripts" / "evidence.py"
KINDS = ("fix", "refactor", "performance", "new", "migration", "no_change")
KIND_WORDS = {"fix": "fix", "refactor": "refactor", "performance": "performance change", "new": "new dataset", "migration": "migration"}
SAFE, UNSURE = "safe to merge", "unsure"
ORDER = ("won't do", UNSURE, SAFE)
SINGLE_TICKET = "change"
TITLE_LENGTH = 110
GROUP_SHOWN = 6


def proof_location(path):
    """(results directory, ticket) of a proof file: results/proof.json is the single ticket, and
    results/proofs/<ticket>.json one ticket of a backlog."""
    path = path.resolve()
    if path.name == "proof.json":
        return path.parent, SINGLE_TICKET
    if path.parent.name == "proofs":
        return path.parent.parent, path.stem
    raise SystemExit(f"{path} is neither results/proof.json nor results/proofs/<ticket>.json")


def computed(ticket, results, evidence_file=None):
    """The label, its reasons, the obligations and the measured checks as evidence.py computes them now, from
    the attempt's own measurement or from the one given (verification's)."""
    command = [sys.executable, str(EVIDENCE), "label", "--ticket", ticket, "--json", "--results-dir", str(results)]
    completed = subprocess.run(command + (["--evidence", str(evidence_file)] if evidence_file else []),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"the label of {ticket} could not be computed: {completed.stderr.strip()[-300:]}")
    return json.loads(completed.stdout)[ticket]


def check_rows(evidence, model):
    """(name, on the reference before, on the code as it stood, after) for every check of one dataset, from
    the reports the evidence names."""
    reports = {Path(path).name: Path(path) for path in evidence.get("artifacts", {})}
    runs = {}
    for column, name in (("reference", f"invariants_reference_{model}.json"), ("code", f"invariants_before_code_{model}.json"),
                         ("after", f"invariants_after_{model}.json")):
        path = reports.get(name)
        runs[column] = ({r["name"]: r for r in json.loads(path.read_text(encoding="utf-8")).get("results", [])}
                        if path is not None and path.is_file() else None)

    def cell(run, name):
        if run is None:
            return "not measured"
        result = run.get(name)
        if result is None:
            return "not run"
        if result["status"] == "pass":
            return "pass"
        if result["status"] == "fail":
            return f"fail ({result.get('violations', result.get('detail', ''))})"
        return "error"

    return [(name, cell(runs["reference"], name), cell(runs["code"], name), cell(runs["after"], name)) for name in (runs["after"] or {})]


def measure_name(key):
    """A fingerprint key in words: nd__x is the distinct count of x, sum__x its sum."""
    if key.startswith("nd__"):
        return f"the distinct values of `{key[4:]}`"
    if key.startswith("sum__"):
        return f"the sum of `{key[5:]}`"
    return f"`{key}`"


def reason_for(model, explained):
    """The proof's reason for one changed downstream dataset: by its name, or by a name prefix ending in `*`."""
    lowered = model.lower()
    for name, reason in explained.items():
        if reason and (name.lower() == lowered or (name.endswith("*") and len(name) > 3 and lowered.startswith(name[:-1].lower()))):
            return reason
    return None


def shorten(text, length):
    """A title cut at a word, for display; the full sentence is the first line of the body."""
    if len(text) <= length:
        return text
    return text[:length].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."


def final_label(label, reasons, verified, verification_given, measured_again, branch_check):
    """The label a pull request may carry. `label` is what evidence.py computes now: from verification's own
    measurement of the delivered code when it took one (`measured_again`), from the attempt's otherwise. It
    is never above what verification recorded (a reviewer's finding lowers it there), safe only on
    verification's measurement, and only when the branch that is pushed passed its own check. A won't-do
    outcome a reviewer disputed becomes unsure."""
    earned = (verified or {}).get("earned")
    if verified and earned in ORDER and (ORDER.index(earned) < ORDER.index(label) or (label == ORDER[0] and earned == UNSURE)):
        return earned, list(verified.get("reasons") or []) or ["verification recorded this change as not established, without a reason"], True
    if label == SAFE and not (measured_again and earned == SAFE):
        return UNSURE, ["verification's report does not show this change measured again and holding"
                        + ("" if verification_given else ": no verification report was given to the renderer")], False
    if label == SAFE and branch_check != "pass":
        return UNSURE, [f"the branch this pull request carries was not checked on its own ({branch_check})"], False
    return label, reasons, False


def render(proof, proof_path, verification, branch_check):
    kind = proof.get("change_kind")
    if kind not in KINDS:
        raise SystemExit(f"change_kind must be one of {KINDS}, not {kind!r}")
    results, ticket = proof_location(proof_path)
    if proof.get("ticket") not in (None, ticket):
        raise SystemExit(f"{proof_path} is the proof of `{ticket}` by its place and says `ticket`: {proof.get('ticket')!r}")
    validated = proof.get("validated")
    if not isinstance(validated, dict):
        raise SystemExit("`validated` must be an object (commands, outcomes, not_inspected, skipped, assumptions)")
    for item in validated.get("skipped") or []:
        if not isinstance(item, dict) or not item.get("check") or not item.get("reason"):
            raise SystemExit("every `validated.skipped` entry is an object with `check` and `reason`")
    cause = proof.get("root_cause") or {}
    if kind == "no_change" and not proof.get("decision_evidence"):
        raise SystemExit("a no_change outcome lists its decision_evidence")

    verified = ((verification or {}).get("tickets") or {}).get(ticket)
    # Verification's measurement of the delivered code, when it took one, is what the pull request is
    # published on: the label is computed from it now, against the files as they are here, and the numbers
    # below are its numbers. Without one, the attempt's own measurement is shown and cannot read safe.
    fresh = Path(verified["evidence"]) if verified and verified.get("measured_again") is True and verified.get("evidence") else None
    fresh = fresh if fresh is not None and fresh.is_file() else None
    measured = computed(ticket, results, fresh)
    label, reasons, lowered = final_label(measured["label"], list(measured["reasons"]), verified, verification is not None,
                                          fresh is not None, branch_check)

    summary = str(cause.get("summary") or proof.get("summary") or "").strip().rstrip(".")
    # The title is the change in a reviewer's words and carries no tag. The verdict is the first thing the
    # body says, in a sentence: verified, or every point that is still open, each named.
    title = shorten(f"{ticket}: {summary}" if summary else ticket, TITLE_LENGTH)
    lines = ([f"{summary}.", ""] if summary else []) + ["## Verification"]
    if kind == "no_change":
        lines.append("No change is proposed. The evidence for that decision is below." if label != UNSURE
                     else "No change is proposed, and the evidence for that decision is not complete:")
    elif label == SAFE:
        lines.append(f"Verified. Verification measured this {KIND_WORDS[kind]} again on the delivered code, every obligation of its kind "
                     "is met by a measurement (listed below), and this branch passes its check on its own.")
    else:
        # Says only that points are open: an open point may be a measurement that failed, not only one not taken.
        lines.append(("One point is open" if len(reasons) == 1 else f"{len(reasons)} points are open")
                     + (", as verification found. Not established:" if lowered else ". Not established:"))
    for reason in reasons if label != SAFE else []:
        lines.append(f"- {reason}")

    evidence_path = fresh or results / ticket / "evidence.json"
    if kind != "no_change" and evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        context = evidence.get("context", {})
        lines += ["", "## What was measured",
                  ("Verification's own measurement of the delivered code. " if fresh else "The build session's measurement; verification took none. ")
                  + f"Built on `{context.get('build_target')}` at {context.get('taken_at')}, compared with `{context.get('reference_target')}` "
                  f"(what production holds); the baseline was taken at {context.get('baseline_taken_at')}, before the change."]
        if context.get("baseline_retaken"):
            lines.append(f"- The baseline was taken again {context['baseline_retaken']} time(s): the spec was corrected, and the "
                         "before-state is read from production alone when the code had moved.")
        stood = evidence.get("code_as_it_stood", {})
        if stood.get("status") == "does_not_build":
            lines.append("- The code as it stood did not build: " + " | ".join(stood.get("errors", [])[:2]))
        for model, entry in evidence.get("datasets", {}).items():
            compared = entry.get("against_reference")
            if compared:
                moved = ", ".join(measure_name(name) for name in compared["columns_that_differ"][:8]) or "the row count"
                lines.append(f"- `{model}`: {compared['rows_built']} rows built, {compared['rows_reference']} in production; "
                             + (f"differs in {moved}" if compared["changed"] else "identical fingerprint"))
            elif entry.get("reference") == "absent":
                lines.append(f"- `{model}`: {entry.get('built_rows')} rows built; production has no such dataset")
            else:
                lines.append(f"- `{model}`: {entry.get('built_rows')} rows built; production {entry.get('reference')}")
            checks = measured["checks"].get(model)
            if checks:
                lines.append(f"  - checks moved from failing to passing: {', '.join(checks['moved']) or 'none'}")
                lines.append(f"  - checks preserved: {len(checks['preserved'])}; regressed: {', '.join(checks['regressed']) or 'none'}")
                rows = check_rows(evidence, model)
                if rows:
                    lines += ["", f"| Check on `{model}` | Production before | Code as it stood | After |", "| --- | --- | --- | --- |"]
                    lines += [f"| {name} | {reference} | {code} | {after} |" for name, reference, code, after in rows]
                    lines.append("")
        accepted = evidence.get("acceptance")
        if accepted:
            lines.append(f"- The task's acceptance check on this ticket: {accepted.get('verdict')} ({accepted.get('said')}).")
        down = evidence.get("downstream", {})
        explained = {str(item.get("model")): str(item.get("reason") or "").strip() for item in ((proof.get("blast_radius") or {}).get("changed_models") or [])
                     if isinstance(item, dict)} if isinstance(proof.get("blast_radius"), dict) else {}
        if not down.get("names"):
            lines.append("- Downstream: nothing reads these datasets.")
        elif not down.get("rebuilt"):
            lines.append(f"- Downstream: {len(down['names'])} dataset(s) read these and were not rebuilt or compared.")
        else:
            lines.append(f"- Downstream: {len(down['names'])} dataset(s) rebuilt and their fingerprints compared with production's "
                         f"(a screen, not a row comparison): {len(down['changed'])} changed, {len(down.get('same_fingerprint', []))} with the same "
                         f"fingerprint, {len(down['unmeasured'])} not measurable.")
            # One reason may cover a family of datasets by a name prefix; each reason is printed once with what it covers.
            covered = {}
            for item in down["changed"]:
                covered.setdefault(reason_for(item["model"], explained) or "no reason given", []).append(item["model"])
            for reason, models in covered.items():
                shown = ", ".join(f"`{name}`" for name in models[:GROUP_SHOWN]) + (f" and {len(models) - GROUP_SHOWN} more" if len(models) > GROUP_SHOWN else "")
                lines.append(f"  - {len(models)} dataset(s), {shown}: {reason}")
            for item in down["unmeasured"][:GROUP_SHOWN]:
                stood = item.get("on_the_code_as_it_stood") if isinstance(item.get("on_the_code_as_it_stood"), dict) else None
                lines.append(f"  - not measurable: `{item['model']}` ({item['why']}" + (f": {item['error'].strip().splitlines()[0][:160]}" if (item.get("error") or "").strip() else "")
                             + (f"; built from the code as it stood before this change, on production's state, dbt records: {stood['status']}"
                                + (f", {stood['error'].strip().splitlines()[0][:160]}" if (stood.get("error") or "").strip() else "") if stood else "") + ")")
    elif kind != "no_change":
        lines += ["", "## What was measured", "Nothing: the ticket has no evidence file."]

    lines += ["", "## Validation"]
    for command in validated.get("commands") or []:
        lines.append(f"- Command: `{command}`")
    lines.append(f"- Outcomes: {validated.get('outcomes') or 'not stated'}")
    for item in validated.get("not_inspected") or []:
        lines.append(f"- Not inspected: {item}")
    for item in validated.get("skipped") or []:
        lines.append(f"- Skipped: {item['check']}; reason: {item['reason']}")
    for item in validated.get("assumptions") or []:
        lines.append(f"- Assumption to confirm: {item}")

    if kind == "no_change":
        lines += ["", "## Decision evidence", *(f"- {item}" for item in proof["decision_evidence"])]
    else:
        lines += ["", "## Root cause" if kind == "fix" else "## The change",
                  f"`{cause.get('file') or '-'}` lines {cause.get('lines') or '-'}: {cause.get('summary', '')}"]
        if cause.get("reproducing_query"):
            lines += ["", "Reproduce:", "", "```sql", cause["reproducing_query"].strip(), "```"]
    runbook = proof.get("runbook")
    if isinstance(runbook, dict):
        lines += ["", "## Runbook"]
        for key, heading in (("one_time_actions", "Run once"), ("order", "Order"), ("verification", "How to see it worked"), ("recovery", "Recovery")):
            for item in runbook.get(key) or []:
                lines.append(f"- {heading}: {item}")
    for field, heading in (("rerun", "## Replay"), ("performance", "## Performance")):
        if field in proof:
            lines += ["", heading, "", "```json", json.dumps(proof[field], indent=2), "```"]
    portability = proof.get("portability")
    if portability:
        lines += ["", "## Portability"]
        lines += [f"- {item.get('engine')}: {item.get('evidence')}" for item in portability] if isinstance(portability, list) else [str(portability)]
    lines += ["", "## Files", *(f"- `{f}`" for f in proof.get("changed_files") or [])]
    unknowns = proof.get("unknowns") or []
    lines += ["", "## Open questions" if unknowns else "## Open questions: none"]
    for item in unknowns:
        lines.append(f"- {item}")
    return title, "\n".join(lines).rstrip() + "\n", label, (reasons if label != SAFE else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("proof")
    parser.add_argument("--verification", default=None)
    parser.add_argument("--branch-check", choices=("pass", "fail", "not_run"), default="not_run",
                        help="what the acceptance check said on the branch this pull request carries, alone")
    parser.add_argument("--title-out", default=None)
    parser.add_argument("--body-out", default=None)
    parser.add_argument("--label-out", default=None, help="write the label and its open points here as JSON, for the deployment's own record")
    args = parser.parse_args()
    proof_path = Path(args.proof)
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    verification = json.loads(Path(args.verification).read_text(encoding="utf-8")) if args.verification else None
    title, body, label, open_points = render(proof, proof_path, verification, args.branch_check)
    if args.label_out:
        Path(args.label_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.label_out).write_text(json.dumps({"label": label, "open_points": open_points}, indent=2) + "\n", encoding="utf-8")
    if args.title_out:
        Path(args.title_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.title_out).write_text(title + "\n", encoding="utf-8")
    if args.body_out:
        Path(args.body_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.body_out).write_text(body, encoding="utf-8")
    sys.stdout.write(title + "\n\n" + body)


if __name__ == "__main__":
    main()
