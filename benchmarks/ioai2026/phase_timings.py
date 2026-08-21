#!/usr/bin/env python3
"""Where a Kaggle campaign spent its wall clock.

Reads a run log (and, when reachable, the Kaggle API) and reports how long each
phase took, so optimisation targets the phase that actually costs time rather
than the one that feels slow.

Three independent timing sources, in decreasing resolution:

1. Stamped log lines (`HH:MM:SS ...`) — every line, if the run went through
   infra/run_competition.sh, which pipes both stages through a timestamper.
   Phase boundaries are then real deltas.
2. Self-reported durations — components that print their own elapsed time
   (ensemble members, codex sessions). Available even in unstamped logs.
3. The Kaggle API — kernel run times and submission times, giving the cloud
   round trip that no local log can see.

Usage:
    python -m benchmarks.ioai2026.phase_timings --log ~/driver.log
    python -m benchmarks.ioai2026.phase_timings --log ~/driver.log \
        --competition ioai-2026-ai-models-track-practice-task-1 --json out.json
"""

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime, timedelta

# Ordered phase markers: (label, regex matching the line that STARTS the phase).
# The next marker's first occurrence ends the previous phase.
PHASE_MARKERS = [
    ("box setup", re.compile(r"STAGE 1: setup_box")),
    ("preflight", re.compile(r"STAGE 4: run_competition|=== preflight:")),
    ("campaign start", re.compile(r"=== runner: campaign|root=.* competition=")),
    ("lens planning", re.compile(r"Lens planner starting")),
    ("ensemble ideation", re.compile(r"Ensemble ideation member starting")),
    ("selection", re.compile(r"Ensemble ideation pooled")),
    ("implementation", re.compile(r"Node expansion:|Running \w+ implementation")),
    ("feedback", re.compile(r"Generating feedback")),
    ("final eval", re.compile(r"=== campaign done|\[harvest\]")),
]

STAMP = re.compile(r"^(\d{2}:\d{2}:\d{2}) ")
# Stage banners carry their own clock in trailing parens, and the label itself
# may contain parens ("run_competition.sh (preflight + 2h campaign) (18:43:42)")
# — so match the LAST parenthesised time on the line, not the first.
STAGE_STAMP = re.compile(r"===== STAGE \d+:.*\((\d{2}:\d{2}:\d{2})\)")
MEMBER_DURATION = re.compile(
    r"member (?P<label>[\w.:\-/]+): candidates=(?P<kept>\d+)/(?P<asked>\d+)"
    r"[^,]*, (?P<seconds>\d+)s"
)
CODEX_ELAPSED = re.compile(r"killed by its deadline after (\d+)s")


def parse_clock(value: str) -> datetime:
    """HH:MM:SS to a datetime on a nominal day (date is irrelevant to deltas)."""
    return datetime.strptime(value, "%H:%M:%S")


def stamped_phase_spans(lines: list) -> list:
    """(label, start, end) for each phase seen in a timestamped log."""
    hits = []
    for line in lines:
        stamp = STAMP.match(line) or STAGE_STAMP.search(line)
        if not stamp:
            continue
        clock = parse_clock(stamp.group(1))
        for label, pattern in PHASE_MARKERS:
            if pattern.search(line) and (not hits or hits[-1][0] != label):
                hits.append((label, clock))
                break
    spans = []
    for index, (label, start) in enumerate(hits):
        end = hits[index + 1][1] if index + 1 < len(hits) else None
        if end is not None and end < start:      # crossed midnight
            end += timedelta(days=1)
        spans.append((label, start, end))
    return spans


def member_durations(lines: list) -> list:
    """Self-reported ensemble-member timings; independent of log stamping."""
    found = []
    for line in lines:
        match = MEMBER_DURATION.search(line)
        if match:
            found.append({
                "member": match.group("label"),
                "seconds": int(match.group("seconds")),
                "candidates": f"{match.group('kept')}/{match.group('asked')}",
            })
    return found


def kaggle_submission_times(competition: str, timeout_seconds: int) -> list:
    """Submission timestamps from the API — the cloud half of the round trip."""
    kaggle_bin = shutil.which("kaggle")
    if not kaggle_bin:
        return []
    proc = subprocess.run(
        [kaggle_bin, "competitions", "submissions", competition,
         "--format", "json", "-q"],
        capture_output=True, text=True, timeout=timeout_seconds,
    )
    if proc.returncode != 0:
        return []
    start = min(i for i in (proc.stdout.find("["), proc.stdout.find("{")) if i >= 0)
    payload = json.loads(proc.stdout[start:])
    return [
        {"date": str(entry.get("date", ""))[:19],
         "score": str(entry.get("publicScore", "") or ""),
         "description": str(entry.get("description", ""))[:60]}
        for entry in payload
    ]


def build_report(log_path: str, competition: str, timeout_seconds: int) -> dict:
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()

    spans = stamped_phase_spans(lines)
    phases = []
    for label, start, end in spans:
        phases.append({
            "phase": label,
            "start": start.strftime("%H:%M:%S"),
            "minutes": round((end - start).total_seconds() / 60, 1) if end else None,
        })
    report = {
        "log": log_path,
        "log_is_stamped": bool(STAMP.match(lines[0])) if lines else False,
        "phases": phases,
        "ensemble_members": member_durations(lines),
    }
    if competition:
        report["submissions"] = kaggle_submission_times(competition, timeout_seconds)
    return report


def print_report(report: dict) -> None:
    print(f"log: {report['log']}  (per-line stamps: "
          f"{'yes' if report['log_is_stamped'] else 'NO — coarse phases only'})")
    print("\n=== phases ===")
    for entry in report["phases"]:
        span = f"{entry['minutes']:>6.1f} min" if entry["minutes"] is not None \
            else "     (open)"
        print(f"  {entry['start']}  {span}  {entry['phase']}")
    if report["ensemble_members"]:
        print("\n=== ensemble members (self-reported, parallel) ===")
        for member in report["ensemble_members"]:
            print(f"  {member['seconds']:>5}s  candidates={member['candidates']:>5}"
                  f"  {member['member']}")
    for submission in report.get("submissions", []):
        if submission is report["submissions"][0]:
            print("\n=== submissions (Kaggle clock) ===")
        print(f"  {submission['date']}  {submission['score']:>9}  "
              f"{submission['description']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="driver.log or runner.log")
    parser.add_argument("--competition", default="",
                        help="slug; adds submission timings from the Kaggle API")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--json", default="", help="also write the report here")
    args = parser.parse_args()

    report = build_report(args.log, args.competition, args.timeout_seconds)
    print_report(report)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
