"""S5 of the semantic suite (docs/plans/redesign/wp9-semantic-suite.md,
change 8): the failure paths are as smooth as the happy path.

Six injections on the churn fixture, each a campaign of its own with the
operating skill on Claude Code in the loop and a scripted user of at most
two turns ("what happened?" and the fix the skill names):

- provider-limit: a PATH shim before `claude` that passes every call through
  except a session turn, arms after the status reaches implementation, and
  answers the CLI's documented failure shape N times (a reconstruction: no
  captured limit exists in the archives, said in the record), then execs
  the CLI; handled means the run entry errors with the limit's text, the
  attempt fails, the campaign goes on or stops truthfully, the skill names
  the class and the fix.
- evaluator-crash: the hold-out labels are moved away from where the
  protected evaluator reads them while the first attempt is being
  implemented, so its evaluation crashes, the engine's own included; the
  crash is read where it surfaced (the attempt's trace row or its own
  session) and the judge's rejection names it.
- missing-key: a local enrichment service answers the table to the key
  alone; the brief requires the table before training; the session asks
  through the inbox; the driver writes the key where the request's fix
  line says, the user says so, the reply resumes the campaign, and the
  service's log shows the resumed attempt's authorised fetch.
- killed-session: an implementation CLI killed mid-run; the run entry ends
  in error, the attempt fails, the campaign goes on by itself.
- killed-writer: the campaign's writer killed while it implements; the
  heartbeat stops at once, a reader sees DEAD (the writer gone), the user
  resumes with one flag, the attempts are kept.
- full-disk: a small tmpfs holding the campaign directory, filled while
  the writer implements; the writer dies with ENOSPC in the launch's log
  (kept outside the full disk; no try/except), the user frees the filler
  and resumes.
- tampering: the evaluator edited inside an attempt's workspace while it
  runs; the integrity check voids the attempt with the hash evidence; this
  campaign is S4's.

Every injection fires early, while the campaign is demonstrably working
(the implementation phase): the churn fixture's first attempt can meet
its goal and end the campaign, so an injection staged after a checkpoint
raced that end, and an unreachable target was tried and withdrawn — the
engineer saw through it (DECISIONS, 2026-10-27). The timeline sampler
runs beside every campaign at the engine config's heartbeat cadence;
every campaign is served a seeded bank (the fixture
`tests/fixtures/bank_seed`, three cards of a real learn, as one commit in
a bare repo under the output directory) whose head and card count are
compared before and after each injection.

    python3 s5_injections.py --clone <clone> --out <dir> [--bank <bank-dir>] [--steps a,b] [--advisory <reason>]
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plugin_proof  # noqa: E402
import status_timeline  # noqa: E402
from plugin_proof import CAMPAIGN_GRACE_SECONDS, SUITE, ClaudeHost, CodexHost, Context, UserMode, codex_session_note, record_step, shell_step, status_of, wait_for  # noqa: E402
from status_timeline import heartbeat_seconds  # noqa: E402

BUDGET = SUITE["builds"]["s5"]
INJECTIONS = SUITE["injections"]
ENRICHMENT_KEY = "s5-enrichment-key-0427"
RECONSTRUCTED_LIMIT = {
    "note": "the CLI's documented shape of a limit it cannot wait out: a rate_limit_event whose status is rejected, then an error result; a reconstruction, no captured limit being in the archives",
    "stdout": [
        {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "rateLimitType": "five_hour", "resetsAt": 0, "unifiedWindows": {"five_hour": {"utilization": 1.0, "status": "rejected"}}}},
        {"type": "result", "subtype": "error_during_execution", "is_error": True, "result": "API Error: 429 {\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\",\"message\":\"This request would exceed your account's rate limit.\"}}", "duration_ms": 1200, "num_turns": 0},
    ],
    "stderr": "API Error: 429 rate_limit_error: This request would exceed your account's rate limit.",
    "exit": 1,
}
SHIM = '''#!/usr/bin/env bash
# S5's provider-limit shim: passes every call through except an armed
# session turn, which answers the reconstructed limit N times.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL="$(cat "$HERE/real")"
turn=0
for arg in "$@"; do [ "$arg" = "-p" ] && turn=1; done
if [ "$turn" = "1" ] && [ -f "$HERE/armed" ]; then
  count=$(cat "$HERE/count" 2>/dev/null || echo 0)
  if [ "$count" -lt "$(cat "$HERE/limit")" ]; then
    echo $((count + 1)) > "$HERE/count"
    date -u +%Y-%m-%dT%H:%M:%SZ >> "$HERE/fired"
    cat "$HERE/stdout.jsonl"
    cat "$HERE/stderr.txt" >&2
    exit "$(cat "$HERE/exit")"
  fi
fi
exec "$REAL" "$@"
'''
# The diagnosis names the class and the fix: each class's words are what
# only that failure is called, and its fix's words are disjoint from them,
# so one word never satisfies both halves ("you can resume it" named the
# killed writer's class and fix at once under a looser table).
CLASS_WORDS = {
    "provider-limit": re.compile(r"rate.?limit|\b429\b|too many requests|quota", re.I),
    "evaluator-crash": re.compile(r"evaluat\w*\s+(crash|fail|error|broke|could not)|labels?\s+(file\s+)?(is\s+|was\s+|were\s+)?(missing|gone|not found|moved|absent)|holdout_labels|hold-?out labels|FileNotFound", re.I),
    "missing-key": re.compile(r"DATA_API_KEY|missing (the )?(api )?key|api key|waiting (on|for) (the|a|your) (key|reply|answer)|inbox request|asked (you )?for the key", re.I),
    "killed-session": re.compile(r"session\s+(was\s+)?(killed|died|terminated|ended|crashed)|(killed|died|terminated)\s+(mid|during)|SIGKILL|exit(ed)?\s+(code\s+)?(137|-9)|run entry (ended|ends) in error|session (ended|exited) (abnormally|in error)", re.I),
    # "the writer process died", "PID 16983 was gone", "the campaign died …
    # PID 16763 disappeared, leaving a stale 'running' status" (rel-s5-codex)
    # all name the writer gone as plainly as "the writer died".
    "killed-writer": re.compile(r"\bdead\b|(writer|campaign|main)(\s+process)?\s+(is\s+|was\s+)?(gone|dead|killed|died)|pid \d+ (is |was )?(gone|dead|disappeared)|no longer running|not running|process\s+(is\s+|was\s+)?(gone|died|killed)|heartbeat\s+(stopped|stale|is old)|stale\s+.{0,12}status", re.I),
    "full-disk": re.compile(r"no space left|ENOSPC|disk\s+(is\s+|was\s+)?full|out of (disk )?space|tmpfs|ran out of space", re.I),
    "tampering": re.compile(r"integrity|tamper|modified evaluator|evaluat\w*\s+(was\s+)?(edited|changed|modified|altered)|hash\s+(mismatch|check)|\bvoided\b", re.I),
}
FIX_WORDS = {
    # A campaign asked about after it finished (the sixth S5 Codex launch:
    # the rate-limited attempt counted against its two experiments and it
    # stopped at max_iterations) is continued with a larger allowance, not
    # waited for: "continuing requires increasing the experiment allowance".
    "provider-limit": re.compile(r"\bwait\b|retry|retried|rode it out|window|resume|continu(e|es|ing)|experiment allowance|more experiments|--budget-experiments|another (attempt|experiment)", re.I),
    "evaluator-crash": re.compile(r"restore|put (it|them|the labels) back|move (it|them) back|then resume|resume", re.I),
    "missing-key": re.compile(r"kapso inbox reply|\breply\b|\.env|provide the key|set (the )?key", re.I),
    # "recovered from a killed implementation session and finished", "Second
    # attempt: delivered" (rel-s5-codex) name the fix — the campaign went on
    # by itself and finished — as plainly as "it carried on".
    "killed-session": re.compile(r"went on|carried on|continued|recovered|\bfinished\b|next attempt|moved on|nothing to (do|resume|stop)|no action|on its own|finished normally|completed normally|(the |a )?(next|second) (iteration|attempt)|iteration 2", re.I),
    # A resume by another name is the fix named: the second S5 of the release
    # gate found the tmpfs with room again (the driver had removed its filler
    # before asking) and said "relaunch with the same command", which is it.
    "killed-writer": re.compile(r"--resume|\bresum(e|ed|ing)\b|relaunch|restart|re-?run|run (it )?again", re.I),
    "full-disk": re.compile(r"free (up )?(the )?(disk|space)|clear (up )?space|delete|remove .*(file|filler)|resume|relaunch|restart|re-?run|run (it )?again", re.I),
    # "requesting restoration of the evaluator", "not as a valid scored
    # result" (rel-s5-codex) name the fix as "restore the evaluator" does.
    "tampering": re.compile(r"not (counted|as a valid|a valid)|discarded|restor\w*\s+(of\s+)?(the\s+)?evaluator|leave the evaluator|do not (edit|modify)|unedited|revert|unchanged", re.I),
}
STEP_NAMES = ["provider-limit", "evaluator-crash", "missing-key", "killed-session", "killed-writer", "full-disk", "tampering", "bank-compare"]
PAUSE_RULE = (
    "before training, the enrichment table must be fetched from http://127.0.0.1:{port}/table with the DATA_API_KEY in .env as the bearer token "
    "(Authorization: Bearer <DATA_API_KEY>); a session that lacks the key asks for it through the inbox and does not train without the table"
)


# -------------------------------------------------------------- readings
DEAD_SHOWN = ("DEAD", "STALLED", "FAILED", "DONE")


def row_truthful(row: Dict[str, Any], cadence: float) -> Optional[bool]:
    """Whether one sample's shown verdict matches what the sampler observed
    beside it: DEAD when the pid is gone; STALLED when the pid lives and the
    heartbeat is older than a cadence; DONE or FAILED when the stored state
    is terminal; RUNNING, STARTING or WAITING when the pid lives and the
    heartbeat is fresh. None when the sample lacks the observation."""
    shown, alive, age = row.get("shown"), row.get("pid_alive"), row.get("heartbeat_age_s")
    stored = (row.get("stored") or {}).get("state")
    if shown is None or alive is None:
        return None
    if shown == "DEAD":
        return not alive
    if shown == "STALLED":
        return bool(alive) and (age is None or age > cadence)
    if shown in ("DONE", "FAILED"):
        return stored in ("done", "failed")
    if shown in ("RUNNING", "STARTING", "WAITING"):
        return bool(alive) and (age is None or age <= 3 * cadence)
    return None


def truthful_within(timeline: Sequence[Dict[str, Any]], injected_at: datetime, cadence: float, expect_terminal: bool) -> Optional[bool]:
    """Whether `kapso watch` SHOWED the truth within one heartbeat of the
    injection — the verdict the user reads, held against the liveness the
    sampler observed beside it, sample by sample. The first heartbeat after
    the injection may still be in flight, so the samples judged are those
    between one and two cadences after it; each must be truthful
    (`row_truthful`), and a writer the injection killed must have been
    shown gone (DEAD, STALLED, FAILED or DONE) by then. None when no sample
    falls in the judged window: unobserved, which the run's bars count."""
    rows = [(row, (datetime.fromisoformat(row["ts"]) - injected_at).total_seconds()) for row in timeline if row.get("ts")]
    judged = [row for row, seconds in rows if cadence < seconds <= 2 * cadence + 5]
    if not judged:
        return None
    verdicts = [row_truthful(row, cadence) for row in judged]
    if any(verdict is None for verdict in verdicts):
        return None
    shown_gone = any(row.get("shown") in DEAD_SHOWN for row, seconds in rows if 0 <= seconds <= 2 * cadence + 5)
    return all(verdicts) and (shown_gone or not expect_terminal)


def fixture_row_count(project: Path) -> int:
    """The id range the enrichment table must cover: the churn fixture's
    train and hold-out rows together (ids are contiguous C000001…C0{total}),
    read from its MANIFEST, so the table the evaluator joins against covers
    the hold-out it scores, not an arbitrary 3000."""
    manifest = json.loads((project / "MANIFEST.json").read_text(encoding="utf-8"))
    rows = manifest["rows"]
    return int(rows["train"]) + int(rows["holdout"])


def names_class_and_fix(text: str, injection: str) -> Dict[str, bool]:
    return {"names_class": bool(CLASS_WORDS[injection].search(text)), "names_fix": bool(FIX_WORDS[injection].search(text))}


def launched_after(campaign: Path, since: datetime) -> bool:
    """Whether the campaign was launched again after ``since``, from its
    records: the status file's `started_at` (a resume writes the
    relaunch's time) or the launch receipt's. The release gate's second
    S5 read three resumes as none by looking for `running` in the minutes
    after the turn — the plugin had waited inside its turn until the
    resumed campaign was done."""
    status = status_of(campaign) or {}
    resolved = Path(campaign).resolve()
    receipt_path = resolved.with_name(resolved.name + ".launch.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
    times = [status.get("started_at"), receipt.get("started_at")]
    return any(stamp and datetime.fromisoformat(stamp) > since for stamp in times)


def resumed_after(campaign: Path, since: datetime, seconds: int = 300) -> bool:
    return wait_for(lambda: launched_after(campaign, since), seconds, every=10)


def scored_after(scores_before: Dict[Any, Any], attempts: Sequence[Dict[str, Any]]) -> bool:
    """Whether some attempt carries a score it did not carry when the reply
    went: a new node's score, or the paused node's own first score (s5-
    claude-8: the resumed node 0 scored after the reply and a rule on node
    identity alone read it as nothing new)."""
    return any((attempt.get("evaluation") or {}).get("score") is not None and scores_before.get(attempt.get("node")) is None for attempt in attempts)


def request_names_no_value(inbox: Path, key_value: str) -> Optional[bool]:
    """The inbox's request names the variable and never a value."""
    if not inbox.is_file():
        return None
    text = inbox.read_text(encoding="utf-8")
    return "DATA_API_KEY" in text and key_value not in text


# The seed commit's author and committer date: a fixture's commit is the same
# commit every time it is seeded, so the comparison can hold a bank's head
# against its seed (a clock-dated commit differed across a second boundary).
SEED_COMMIT_DATE = "2026-01-01T00:00:00Z"


def seed_bank(fixture: Path, target: Path) -> Dict[str, Any]:
    """A bare bank at `target` holding the fixture's tree as one commit —
    the seeded bank every S5 campaign is served and the comparison reads
    before and after each injection (a bank with no card compares
    vacuously). Returns its state. Re-seeding reproduces the commit: the
    tree, the message and the dates are fixed."""
    if not fixture.is_dir():
        raise FileNotFoundError(f"the bank fixture is missing: {fixture}")
    source = target.with_name(target.name + ".src")
    for path in (source, target):
        if path.exists():
            shutil.rmtree(path)
    shutil.copytree(fixture, source)
    dated = {**os.environ, "GIT_AUTHOR_DATE": SEED_COMMIT_DATE, "GIT_COMMITTER_DATE": SEED_COMMIT_DATE}
    git = lambda *args: subprocess.run(["git", "-C", str(source), "-c", "user.name=kapso-semantic", "-c", "user.email=semantic@kapso.invalid", *args], check=True, capture_output=True, text=True, env=dated)
    # On `main`: the engine reads a bank's main branch, whatever git's default is.
    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "the seeded bank")
    subprocess.run(["git", "clone", "-q", "--bare", str(source), str(target)], check=True, capture_output=True, text=True)
    shutil.rmtree(source)
    return bank_state(target)


def session_mentions(campaign: Path, attempt: Dict[str, Any], needle: Any) -> bool:
    """Whether the attempt's own sessions (their stored stream and record
    under the campaign's `.kapso/sessions/<label>/`) carry the needle (a
    string, or a compiled pattern) — where a crash of the evaluator the
    session ran surfaces when the attempt's trace row carries only the
    score it did not get."""
    for session in attempt.get("sessions", []):
        folder = campaign / ".kapso" / "sessions" / str(session.get("label", ""))
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix in (".jsonl", ".json", ".md", ".txt", ".log"):
                text = path.read_text(encoding="utf-8", errors="replace")
                if (needle.search(text) if hasattr(needle, "search") else needle in text):
                    return True
    return False


# The evaluator's own failure on the labels: the file not found, in a
# traceback or an error line — never the file's name in a prompt or a
# command, which any session that read the brief carries.
CRASH_ON_LABELS = re.compile(r"(FileNotFoundError|No such file|Traceback|Errno 2)[^\n]{0,200}holdout_labels\.csv|holdout_labels\.csv[^\n]{0,120}(not found|No such file|does not exist|is missing|raised `?FileNotFoundError|FileNotFoundError)")


def evaluator_crashed(campaign: Path, attempt: Dict[str, Any]) -> bool:
    """The attempt could not score because the evaluator crashed on the
    labels: no score, and the evaluator's failure line in the attempt's
    evaluation output or integrity error, or in its sessions' records."""
    evaluation = attempt.get("evaluation") or {}
    if evaluation.get("score") is not None:
        return False
    if CRASH_ON_LABELS.search(f"{evaluation.get('output') or ''}\n{evaluation.get('integrity_error') or ''}"):
        return True
    return session_mentions(campaign, attempt, CRASH_ON_LABELS)


def crash_occurred(campaign: Path, trace: Dict[str, Any]) -> bool:
    """Whether the evaluator crashed on the missing labels at all this
    campaign — in any attempt's evaluation output, integrity error or
    sessions — even when a capable agent then restored the labels from a
    workspace copy (WP7's readability finding) and the attempt went on to
    score. The injection bit and the plugin had a crash to diagnose; the
    recovery is the finding, not a miss (rel-s5-codex: the agent restored
    a hash-matching fixture itself and scored 0.8481, and no attempt was
    left scoreless, though the crash was in the record and the plugin
    named it)."""
    for attempt in trace.get("attempts", []):
        evaluation = attempt.get("evaluation") or {}
        if CRASH_ON_LABELS.search(f"{evaluation.get('output') or ''}\n{evaluation.get('integrity_error') or ''}"):
            return True
        if session_mentions(campaign, attempt, CRASH_ON_LABELS):
            return True
    # The engine's own record of each node's difficulties (the fourth S5
    # Codex launch: "The evaluator crashed before scoring anything … raised
    # FileNotFoundError", which the plugin cited, in run_state.json and in
    # no session stream, the session having restored the file first).
    run_state = campaign / ".kapso" / "run_state.json"
    if run_state.is_file():
        state = json.loads(run_state.read_text(encoding="utf-8"))
        history = (state.get("strategy_state") or {}).get("node_history") or []
        if any(CRASH_ON_LABELS.search(str(node.get("technical_difficulties") or "")) for node in history if isinstance(node, dict)):
            return True
    return False


def bank_state(bank: Path) -> Dict[str, Any]:
    """A bank's head and card count, read with git so a BARE repo (Kapso's
    own bank home) works: an insight is `insights/<name>.md` but not the
    index, a procedure is `procedures/<name>/card.md`. None/`-1` when the
    repository cannot be read, so an unreadable comparison fails rather
    than looking intact."""
    if not (bank / "HEAD").exists() and not (bank / ".git").exists():
        return {"head": None, "cards": None, "readable": False}
    head = subprocess.run(["git", "-C", str(bank), "rev-parse", "HEAD"], capture_output=True, text=True)
    tree = subprocess.run(["git", "-C", str(bank), "ls-tree", "-r", "--name-only", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0 or tree.returncode != 0:
        return {"head": None, "cards": -1, "readable": False}
    cards = [path for path in tree.stdout.splitlines()
             if (path.startswith("insights/") and path.endswith(".md") and path != "insights/index.md")
             or (path.startswith("procedures/") and path.endswith("/card.md"))]
    return {"head": head.stdout.strip(), "cards": len(cards), "readable": True}


# ---------------------------------------------------------------- driver
class S5:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.clone = Path(args.clone).resolve()
        self.out = Path(args.out).resolve()
        self.work = self.out / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        # The plugin's host is the run's (`--host`): the release gate's second
        # S5 ran the plugin on Claude Code under a machine named for Codex,
        # since the driver took no host. The wheel is pinned in the record
        # as S2 pins it (`--wheel`), where the first gate runs named the
        # version alone.
        host = CodexHost() if args.host == "codex" else ClaudeHost()
        wheel = Path(args.wheel).resolve() if args.wheel else None
        self.ctx = Context(host, self.clone, self.work, self.out, wheel, args.machine, user=UserMode("scripted", args.host))
        self.cadence = heartbeat_seconds() if args.cadence is None else args.cadence
        # The bank every campaign is served: the one given, else the seeded
        # fixture (three cards of a real learn) as a bare repo under out/.
        self.bank = Path(args.bank).resolve() if args.bank else self.out / "bank-seed.git"
        self.bank_before = bank_state(self.bank) if args.bank else seed_bank(self.clone / "tests" / "fixtures" / "bank_seed", self.bank)
        self.samplers: Dict[str, subprocess.Popen] = {}

    # -- machinery --
    def project(self, name: str) -> Path:
        project = self.ctx.fixture("churn-first-build", f"s5-{name}-project")
        # The fixture's seeded generator is tracked in the campaign's repo,
        # and a session that met a crashed evaluator rebuilt the hold-out
        # labels from it and wrote them into the protected directory
        # (s5-claude-5, 2026-10-28). A real repository has no generator for
        # its hold-out; without it the injected failures surface as designed.
        (project / "generate.py").unlink()
        self.ctx.user.interventions.append(f"{name}: the fixture's generate.py removed after the data was generated (a real repository has no generator for its hold-out)")
        return project

    def launch(self, name: str, project: Path, extra_env: str = "", budget: Optional[Dict[str, int]] = None, campaign: Optional[Path] = None, log: Optional[Path] = None) -> Path:
        """The campaign launched as the operating skill launches it (the WP7
        round's invocation), detached, from the project directory."""
        campaign = campaign or self.work / f"s5-{name}"
        if campaign.exists():
            self.ctx.shell(f"kapso build --output {campaign} --stop", check=False)
            shutil.rmtree(campaign)
        budget = budget or BUDGET
        experience = f" --experience file://{self.bank}"
        where = f" --log {log}" if log else ""
        # No --config: the churn fixture ships none, and passing any config
        # changes the engine's defaults (the known layering gotcha), so the
        # packaged defaults are the truer run.
        self.ctx.shell(
            f"{extra_env}kapso build --brief kapso-brief.md --repo . --evaluation evaluation --data data --output {campaign}{experience}{where} "
            f"--budget-minutes {budget['minutes']} --budget-experiments {budget['experiments']} --detach", cwd=project,
        )
        wait_for(lambda: status_of(campaign) is not None, 300, every=5)
        timeline = self.out / "artifacts" / f"{name}-timeline.jsonl"
        # Detach the sampler's stdio from this process: it outlives a step only
        # on a crash, and if it inherited the ssh channel's pipes an orphaned
        # sampler would hold the channel open and hang the whole run.
        self.samplers[name] = subprocess.Popen([sys.executable, str(Path(__file__).parent / "status_timeline.py"), str(campaign), str(timeline), "--every", str(self.cadence)],
                                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return campaign

    def finish(self, name: str, campaign: Path, grace: int = CAMPAIGN_GRACE_SECONDS) -> Dict[str, Any]:
        wait_for(lambda: (status_of(campaign) or {}).get("state") in ("done", "failed"), BUDGET["minutes"] * 60 + grace, every=20)
        sampler = self.samplers.pop(name, None)
        if sampler:
            sampler.terminate()
            sampler.wait()
        return status_of(campaign) or {}

    def timeline(self, name: str) -> List[Dict[str, Any]]:
        path = self.out / "artifacts" / f"{name}-timeline.jsonl"
        return status_timeline.rows(path) if path.is_file() else []

    def trace(self, campaign: Path) -> Dict[str, Any]:
        run = self.ctx.shell(f"kapso trace {campaign} --json", check=False)
        return json.loads(run.stdout) if run.returncode == 0 and run.stdout.strip() else {}

    def ask(self, name: str, project: Path, prompt: str):
        """One user turn through the operating skill."""
        return self.ctx.converse(f"{name}-user", project, prompt, lambda text, index: None, max_turns=1)[0]

    def wait_phase(self, campaign: Path, phase: str, seconds: int) -> bool:
        return wait_for(lambda: (status_of(campaign) or {}).get("phase") == phase or (status_of(campaign) or {}).get("state") in ("done", "failed"), seconds, every=10)

    def record_injection(self, name: str, turns, metrics: Dict[str, Any], artifacts: List[str], notes: List[str], completed: bool, started: float) -> Dict[str, Any]:
        step = record_step(name, turns, completed, artifacts, notes, extra_minutes=(time.monotonic() - started) / 60 - sum(turn.seconds for turn in turns) / 60, kind="injection")
        step["metrics"] = {**metrics, "user_messages": len(turns), "bank_intact": bank_state(self.bank) == self.bank_before if self.bank else None}
        return step

    # -- the injections --
    def step_provider_limit(self) -> Dict[str, Any]:
        started = time.monotonic()
        name = "provider-limit"
        project = self.project(name)
        shim = self.work / "shim"
        if shim.exists():
            shutil.rmtree(shim)
        shim.mkdir()
        (shim / "claude").write_text(SHIM, encoding="utf-8")
        (shim / "claude").chmod(0o755)
        (shim / "real").write_text(shutil.which("claude") or "claude", encoding="utf-8")
        (shim / "limit").write_text(str(INJECTIONS["provider_limit_count"]), encoding="utf-8")
        (shim / "count").write_text("0", encoding="utf-8")
        (shim / "exit").write_text(str(RECONSTRUCTED_LIMIT["exit"]), encoding="utf-8")
        (shim / "stdout.jsonl").write_text("\n".join(json.dumps(row) for row in RECONSTRUCTED_LIMIT["stdout"]) + "\n", encoding="utf-8")
        (shim / "stderr.txt").write_text(RECONSTRUCTED_LIMIT["stderr"] + "\n", encoding="utf-8")
        # Arm before the first session runs: kapso drives each coding session as
        # `claude -p`, the shim fires on an armed `claude -p` call, and the churn
        # task is solved in a single node -- so arming only after the
        # implementation phase races a campaign that may make no further call
        # (observed: the shim never fired). Arming at launch injects the limit on
        # the campaign's first session, reliably.
        (shim / "armed").write_text("", encoding="utf-8")
        campaign = self.launch(name, project, extra_env=f"PATH={shim}:$PATH ")
        wait_for(lambda: (shim / "fired").is_file(), 1800, every=10)
        injected_at = datetime.now(timezone.utc)
        status = self.finish(name, campaign)
        (shim / "armed").unlink(missing_ok=True)
        trace = self.trace(campaign)
        failures = [f for f in trace.get("failures", []) if "429" in json.dumps(f) or "rate_limit" in json.dumps(f)]
        turn_1 = self.ask(name, project, f"what happened to the kapso campaign at ../s5-{name}? it does not look right")
        asked_at = datetime.now(timezone.utc)
        turn_2 = self.ask(name + "-2", project, f"the window is over; continue the campaign at ../s5-{name}")
        resumed = resumed_after(campaign, asked_at)
        diagnosis = names_class_and_fix(turn_1.text, name)
        metrics = {"shim_fired": (shim / "fired").read_text(encoding="utf-8").count("\n") if (shim / "fired").is_file() else 0, "failures_with_the_limit": len(failures),
                   "truthful": truthful_within(self.timeline(name), injected_at, self.cadence, expect_terminal=False), **diagnosis, "resumed": resumed}
        artifacts = [self.ctx.artifact(f"{name}-answers.txt", turn_1.text + "\n\n---\n\n" + turn_2.text), self.ctx.artifact(f"{name}-status.json", json.dumps(status, indent=2)),
                     self.ctx.artifact(f"{name}-trace.json", json.dumps(trace, indent=2)), self.ctx.artifact(f"{name}-shim.json", json.dumps(RECONSTRUCTED_LIMIT, indent=2))]
        self.ctx.shell(f"kapso build --output {campaign} --stop", check=False)
        # Success = the limit was injected and the engineer handled it
        # correctly, whether the campaign rode it out on its own (retried the
        # 429s and finished: status done) or was blocked and the engineer
        # resumed it. Demanding `resumed` alone wrongly fails the first, correct
        # outcome -- the engineer rightly declines to resume a campaign that is
        # not stuck (observed, and named by the engineer itself).
        return self.record_injection(name, [turn_1, turn_2], metrics, artifacts, [RECONSTRUCTED_LIMIT["note"]],
                                     bool(failures) and diagnosis["names_class"] and diagnosis["names_fix"] and (resumed or status.get("state") == "done"), started)

    def step_evaluator_crash(self) -> Dict[str, Any]:
        started = time.monotonic()
        name = "evaluator-crash"
        project = self.project(name)
        campaign = self.launch(name, project)
        # Move the labels while the campaign is implementing node 0, before its
        # first evaluation reads them: waiting for a checkpoint instead races a
        # node 0 that meets its goal and finishes before the labels are moved
        # (observed: the labels moved 16 s after the campaign had finished;
        # an unreachable target was tried and withdrawn — the engineer saw
        # through it, DECISIONS 2026-10-27).
        self.wait_phase(campaign, "implementation", BUDGET["minutes"] * 60)
        # The evaluator reads the labels through the ABSOLUTE path in
        # evaluation/private_dir.txt, the project's own private directory
        # outside the campaign; moving it there breaks every evaluation,
        # the engine's own included. The seed copies the project directory
        # whole, so the campaign root holds a second copy under private/
        # (ignored by git, on disk, its digest in the project's manifest):
        # the gate's S5 on Codex saw the implementation session copy it back
        # to the path the pointer names and score 0.8505 — the readability
        # limit WP7 named, used. Every copy the campaign can reach moves.
        # … and every sibling's: the earlier injections' projects and
        # campaigns keep their own copies of the same labels under the work
        # directory, and the fourth S5 Codex launch saw the session find one
        # by `sha256sum` across the siblings and `cp` it into place before
        # scoring. The rig's own layout must not hand the labels back, so
        # every copy under the work directory moves away for the step.
        # … and the repository clone's fixtures: the fifth S5 Codex launch's
        # session searched the whole machine and copied the labels from
        # examples/churn-first-build/private in the test checkout. The only
        # copies on a fresh machine are the rig's, and all of them move.
        copies = sorted(set(
            [project / "private" / "holdout_labels.csv", campaign / "private" / "holdout_labels.csv"]
            + list(self.work.glob("*/private/holdout_labels.csv"))
            + list(self.clone.glob("examples/*/private/holdout_labels.csv"))
            + list(self.clone.glob("tests/fixtures/*/private/holdout_labels.csv"))
        ))
        moved = {path: self.work / f"s5-{name}-labels-{index}.away" for index, path in enumerate(copies) if path.is_file()}
        for path, away in moved.items():
            path.rename(away)
        injected_at = datetime.now(timezone.utc)
        status = self.finish(name, campaign)
        trace = self.trace(campaign)
        # The attempt that could not score because the labels were gone: the
        # evaluator's error surfaces where it ran — in the attempt's trace row
        # when the engine ran evaluate.py, in the attempt's own session when
        # the session did — so both are read. Paired with a null score it
        # stays specific to the crash (a healthy attempt carries a score).
        crashed = [attempt for attempt in trace.get("attempts", []) if evaluator_crashed(campaign, attempt)]
        crash_seen = bool(crashed) or crash_occurred(campaign, trace)
        turn_1 = self.ask(name, project, f"why did the later attempt of the kapso campaign at ../s5-{name} fail?")
        # A labels file back in a protected directory before the driver
        # restores it was written by the campaign — the thing the record
        # must say (observed: a session rebuilt it from the generator; a
        # session copied the campaign root's own copy back).
        rebuilt_by_campaign = any(path.exists() for path in copies)
        for path, away in moved.items():
            if path.exists():
                path.unlink()
            away.rename(path)
        asked_at = datetime.now(timezone.utc)
        turn_2 = self.ask(name + "-2", project, f"the labels file is back where it was; continue the campaign at ../s5-{name}")
        resumed = resumed_after(campaign, asked_at)
        diagnosis = names_class_and_fix(turn_1.text, name)
        metrics = {"attempts_with_the_crash": len(crashed), "crash_occurred": crash_seen, "labels_rebuilt_by_campaign": rebuilt_by_campaign, "truthful": truthful_within(self.timeline(name), injected_at, self.cadence, expect_terminal=False), **diagnosis, "resumed": resumed}
        artifacts = [self.ctx.artifact(f"{name}-answers.txt", turn_1.text + "\n\n---\n\n" + turn_2.text), self.ctx.artifact(f"{name}-status.json", json.dumps(status, indent=2)), self.ctx.artifact(f"{name}-trace.json", json.dumps(trace, indent=2))]
        self.ctx.shell(f"kapso build --output {campaign} --stop", check=False)
        # The crash is what the plugin had to diagnose; whether an agent then
        # restored the labels from a workspace copy and re-scored (WP7's
        # readability finding) does not unmake it. The pass is the crash seen
        # in the record, the plugin naming the class and the fix, and a resume.
        return self.record_injection(name, [turn_1, turn_2], metrics, artifacts, ["the hold-out labels moved away before the first evaluation: the protected evaluator crashes on every invocation, the engine's own included"],
                                     crash_seen and diagnosis["names_class"] and diagnosis["names_fix"] and resumed, started)

    def step_missing_key(self) -> Dict[str, Any]:
        started = time.monotonic()
        name = "missing-key"
        project = self.project(name)
        port = INJECTIONS["enrichment_port"]
        log = self.out / "artifacts" / f"{name}-service.log"
        log.unlink(missing_ok=True)
        # Detached stdio (see the sampler): the service logs to --log, and an
        # orphaned server must not hold this process's ssh channel open.
        service = subprocess.Popen([sys.executable, str(Path(__file__).parent / "enrichment_service.py"), "--port", str(port), "--key", ENRICHMENT_KEY, "--log", str(log), "--rows", str(fixture_row_count(project))],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        brief = project / "kapso-brief.md"
        brief.write_text(brief.read_text(encoding="utf-8").replace("Rules\n", "Rules\n  - " + PAUSE_RULE.format(port=port) + "\n"), encoding="utf-8")
        campaign = self.launch(name, project, budget={"minutes": BUDGET["minutes"], "experiments": BUDGET["experiments"]})
        inbox = campaign / ".kapso" / "inbox.jsonl"
        paused = wait_for(lambda: inbox.is_file(), BUDGET["minutes"] * 60, every=15)
        injected_at = datetime.now(timezone.utc)
        turn_1 = self.ask(name, project, f"/kapso:inbox ../s5-{name}")
        for env in (campaign / ".env", project / ".env"):
            with open(env, "a", encoding="utf-8") as out:
                out.write(f"DATA_API_KEY={ENRICHMENT_KEY}\n")
        self.ctx.user.interventions.append(f"{name}: the key written into the campaign's and the project's .env before the person says so")
        replied_at = datetime.now(timezone.utc)
        scores_before_reply = {attempt.get("node"): (attempt.get("evaluation") or {}).get("score") for attempt in self.trace(campaign).get("attempts", [])}
        turn_2 = self.ask(name + "-2", project, f"DATA_API_KEY is in both .env files now; reply to the request of ../s5-{name} that it is in place and resume")
        resumed = wait_for(lambda: inbox.is_file() and '"continued"' in inbox.read_text(encoding="utf-8"), 900, every=15)
        status = self.finish(name, campaign, grace=BUDGET["minutes"] * 60)
        service.terminate()
        service.wait()
        fetches = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.is_file() else []
        # The authorised fetch must come AFTER the reply (the resumed
        # attempt's), not just anywhere; the table fetch for the hold-out,
        # not merely one row.
        authorised_after = any(entry["authorized"] and entry.get("ts", "") >= replied_at.isoformat(timespec="seconds") and entry.get("path", "").startswith("/table") for entry in fetches)
        trace = self.trace(campaign)
        # A score that landed after the reply — on a node the record did not
        # hold, or on the paused node itself, which had none before (the
        # session asked first thing) — not any score anywhere.
        scored_after_resume = scored_after(scores_before_reply, trace.get("attempts", []))
        metrics = {"paused": paused, "request_names_no_value": request_names_no_value(inbox, ENRICHMENT_KEY), "authorised_table_fetch_after_reply": authorised_after,
                   "unauthorised_before": any(not entry["authorized"] for entry in fetches), "scored_after_resume": scored_after_resume, "resumed": resumed,
                   "truthful": truthful_within(self.timeline(name), injected_at, self.cadence, expect_terminal=False), **names_class_and_fix(turn_1.text, name)}
        artifacts = [self.ctx.artifact(f"{name}-answers.txt", turn_1.text + "\n\n---\n\n" + turn_2.text), self.ctx.artifact(f"{name}-status.json", json.dumps(status, indent=2)), self.ctx.artifact(f"{name}-trace.json", json.dumps(trace, indent=2)), f"artifacts/{name}-service.log"]
        if inbox.is_file():
            artifacts.append(self.ctx.copy_artifact(inbox, f"{name}-inbox.jsonl"))
        self.ctx.shell(f"kapso build --output {campaign} --stop", check=False)
        return self.record_injection(name, [turn_1, turn_2], metrics, artifacts, ["the enrichment service answers the table to the key alone and logs every request with its time; the requirement is never lifted; the authorised table fetch and the score must come after the reply"],
                                     paused and bool(metrics["request_names_no_value"]) and authorised_after and scored_after_resume and resumed, started)

    def _session_cli_pid(self, campaign: Path) -> Optional[int]:
        """The implementation session's CLI process: a child of the writer
        whose command line runs claude or codex."""
        writer = (status_of(campaign) or {}).get("pid")
        if not writer:
            return None
        table = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True).stdout.splitlines()
        for line in table:
            parts = line.split(None, 2)
            if len(parts) == 3 and parts[1] == str(writer) and re.search(r"\b(claude|codex)\b", parts[2]):
                return int(parts[0])
        return None

    def step_killed_session(self) -> Dict[str, Any]:
        started = time.monotonic()
        name = "killed-session"
        project = self.project(name)
        campaign = self.launch(name, project)
        self.wait_phase(campaign, "implementation", 1800)
        pid = None
        wait_for(lambda: (pid := self._session_cli_pid(campaign)) is not None, 600, every=10)
        pid = self._session_cli_pid(campaign)
        session_killed = isinstance(pid, int) and Path(f"/proc/{pid}").exists()
        if session_killed:
            os.kill(pid, signal.SIGKILL)
        injected_at = datetime.now(timezone.utc)
        status = self.finish(name, campaign)
        trace = self.trace(campaign)
        errored = [f for f in trace.get("failures", [])]
        turn_1 = self.ask(name, project, f"what happened to the kapso campaign at ../s5-{name}?")
        diagnosis = names_class_and_fix(turn_1.text, name)
        metrics = {"session_killed": session_killed, "failures": len(errored), "attempts": len(trace.get("attempts", [])), "truthful": truthful_within(self.timeline(name), injected_at, self.cadence, expect_terminal=False), **diagnosis}
        artifacts = [self.ctx.artifact(f"{name}-answers.txt", turn_1.text), self.ctx.artifact(f"{name}-status.json", json.dumps(status, indent=2)), self.ctx.artifact(f"{name}-trace.json", json.dumps(trace, indent=2))]
        self.ctx.shell(f"kapso build --output {campaign} --stop", check=False)
        return self.record_injection(name, [turn_1], metrics, artifacts, ["an implementation CLI killed mid-run: the run entry ends in error, the attempt fails, the campaign goes on by itself"],
                                     session_killed and bool(errored) and status.get("state") in ("done", "failed") and diagnosis["names_class"] and diagnosis["names_fix"], started)

    def _resume_after_writer_gone(self, name: str, project: Path, campaign: Path, injected_at: datetime, second_prompt: str, note: str, started: float, extra: Dict[str, Any], injected: bool) -> Dict[str, Any]:
        # The attempts in the record before the writer went, by identity
        # (node and branch), so "kept" means the same attempts are there
        # after the resume — a count could hide a replaced one. With no
        # attempt landed before the kill (the injection is staged early)
        # retention is not exercised, said rather than passed.
        attempts_before = sorted((attempt.get("node"), attempt.get("branch")) for attempt in self.trace(campaign).get("attempts", []))
        # The writer gone, said truthfully: DEAD when it vanished without a
        # word, FAILED when it recorded its own end first (ENOSPC: the
        # engine wrote `failed` before it died, s5-claude-6) — a reader sees
        # the writer is gone either way; a RUNNING with the pid gone is the lie.
        shown_dead = wait_for(lambda: (self.timeline(name) or [{}])[-1].get("shown") in ("DEAD", "FAILED"), int(3 * self.cadence) + 60, every=5)
        shown_after = (self.timeline(name) or [{}])[-1].get("shown")
        turn_1 = self.ask(name, project, f"what happened to the kapso campaign at ../s5-{name}? kapso watch says something is wrong")
        turn_2 = self.ask(name + "-2", project, second_prompt)
        # The resume is any relaunch after the writer went: a plugin that
        # diagnoses and resumes in its first answer (rel-s5-codex's full-disk:
        # "the filesystem now has 59 MB free; I resumed the campaign") has
        # done the thing, and the second turn's "already resumed" is right.
        resumed = resumed_after(campaign, injected_at)
        # The resumed campaign is watched for what is left of the step's own
        # budget, not to its end: the claim is that it continues from its
        # checkpoint with the attempts kept, read from the trace, and a
        # forty-minute campaign run out in full put both resume classes over
        # their forty-five minutes (s5-claude-6: 48.7 and 60.6). It is
        # stopped below either way.
        remaining = SUITE["steps"][name]["minutes"] * 60 - (time.monotonic() - started) - SUITE["waits"]["trace_and_stop_seconds"]
        wait_for(lambda: (status_of(campaign) or {}).get("state") in ("done", "failed"), max(int(remaining), 60), every=20)
        sampler = self.samplers.pop(name, None)
        if sampler:
            sampler.terminate()
            sampler.wait()
        status = status_of(campaign) or {}
        trace = self.trace(campaign)
        attempts_after = sorted((attempt.get("node"), attempt.get("branch")) for attempt in trace.get("attempts", []))
        attempts_kept = (set(attempts_before) <= set(attempts_after)) if attempts_before else None
        diagnosis = names_class_and_fix(turn_1.text, name)
        metrics = {**extra, "shown_dead": shown_dead, "shown_after": shown_after, "truthful": truthful_within(self.timeline(name), injected_at, self.cadence, expect_terminal=True), **diagnosis, "resumed": resumed,
                   "injected": injected, "attempts_kept": attempts_kept, "attempts_before": len(attempts_before), "attempts_after": len(attempts_after)}
        artifacts = [self.ctx.artifact(f"{name}-answers.txt", turn_1.text + "\n\n---\n\n" + turn_2.text), self.ctx.artifact(f"{name}-status.json", json.dumps(status, indent=2)), self.ctx.artifact(f"{name}-trace.json", json.dumps(trace, indent=2))]
        self.ctx.shell(f"kapso build --output {campaign} --stop", check=False)
        return self.record_injection(name, [turn_1, turn_2], metrics, artifacts, [note],
                                     injected and shown_dead and diagnosis["names_class"] and diagnosis["names_fix"] and resumed and attempts_kept is not False, started)

    def step_killed_writer(self) -> Dict[str, Any]:
        started = time.monotonic()
        name = "killed-writer"
        project = self.project(name)
        campaign = self.launch(name, project)
        # Kill the writer while it is demonstrably running (the implementation
        # phase), not at the first checkpoint: a node-0 that reaches the goal
        # flips the campaign to done in the same instant the checkpoint lands,
        # so a post-checkpoint kill would race the writer's own exit. The guard
        # still covers the edge where the writer has already gone -- os.kill on
        # a dead pid would raise, and a writer already absent satisfies the
        # injection's premise either way.
        self.wait_phase(campaign, "implementation", BUDGET["minutes"] * 60)
        pid = (status_of(campaign) or {}).get("pid")
        writer_killed = isinstance(pid, int) and Path(f"/proc/{pid}").exists()
        if writer_killed:
            os.kill(pid, signal.SIGKILL)
        injected_at = datetime.now(timezone.utc)
        return self._resume_after_writer_gone(name, project, campaign, injected_at, f"continue the campaign at ../s5-{name} from where it stopped",
                                              "the writer killed with SIGKILL during the implementation phase; the heartbeat stops at once; a reader sees DEAD as soon as the writer is gone", started, {"writer_killed": writer_killed},
                                              injected=writer_killed)

    def step_full_disk(self) -> Dict[str, Any]:
        started = time.monotonic()
        name = "full-disk"
        project = self.project(name)
        mount = self.work / f"s5-{name}-disk"
        campaign = mount / "campaign"
        if mount.exists():
            self.ctx.shell(f"kapso build --output {campaign} --stop", check=False)
            subprocess.run(["sudo", "umount", "-l", str(mount)], capture_output=True)
            shutil.rmtree(mount)
        mount.mkdir()
        mounted = subprocess.run(["sudo", "mount", "-t", "tmpfs", "-o", f"size={INJECTIONS['tmpfs_mb']}m,uid={os.getuid()},gid={os.getgid()}", "tmpfs", str(mount)], capture_output=True, text=True)
        if mounted.returncode != 0:
            return shell_step(name, False, [self.ctx.artifact(f"{name}-mount.txt", mounted.stderr)], (time.monotonic() - started) / 60, ["the tmpfs could not be mounted (root on the machine is the requirement)"], kind="injection")
        log = self.out / "artifacts" / f"{name}-launch.log"
        campaign = self.launch(name, project, campaign=campaign, log=log)
        # Fill the disk while the writer is implementing (actively writing to
        # the campaign dir), not at the first checkpoint: a node-0 that reaches
        # the goal finishes before a post-checkpoint fill, and ENOSPC can only
        # kill a writer that is still writing.
        self.wait_phase(campaign, "implementation", BUDGET["minutes"] * 60)
        filler = mount / "filler"
        fill = subprocess.run(["dd", "if=/dev/zero", f"of={filler}", "bs=1M", f"count={INJECTIONS['tmpfs_mb'] + 8}", "status=none"], capture_output=True, text=True)
        injected_at = datetime.now(timezone.utc)
        wait_for(lambda: (self.timeline(name) or [{}])[-1].get("shown") == "DEAD", int(10 * self.cadence) + 120, every=5)
        log_text = log.read_text(encoding="utf-8") if log.is_file() else ""
        filler.unlink(missing_ok=True)
        step = self._resume_after_writer_gone(name, project, campaign, injected_at, f"the disk has space again; continue the campaign at ../s5-{name}-disk/campaign",
                                              f"a {INJECTIONS['tmpfs_mb']} MB tmpfs holding the campaign directory, filled while the writer is implementing; the writer dies with ENOSPC in the launch's log (kept outside the full disk; no try/except); the truth lives in the log and the receipt", started,
                                              {"fill_said": fill.stderr.strip()[-200:], "enospc_in_log": bool(re.search(r"No space left on device|ENOSPC", log_text))},
                                              injected=bool(re.search(r"No space left on device|ENOSPC", log_text)))
        step["artifacts"].append(f"artifacts/{name}-launch.log")
        # Copy the whole campaign off the tmpfs and verify it BEFORE
        # unmounting, so the checkpoint, sessions and history survive for
        # the fetch (the unmount would otherwise take them with it).
        saved = self.out / "artifacts" / f"{name}-campaign"
        if saved.exists():
            shutil.rmtree(saved)
        shutil.copytree(campaign, saved, symlinks=True)
        step["metrics"]["campaign_saved"] = (saved / ".kapso" / "run_state.json").is_file() or (saved / ".kapso").is_dir()
        step["artifacts"].append(f"artifacts/{name}-campaign")
        subprocess.run(["sudo", "umount", "-l", str(mount)], capture_output=True)
        return step

    def step_tampering(self) -> Dict[str, Any]:
        started = time.monotonic()
        name = "tampering"
        project = self.project(name)
        campaign = self.launch(name, project)
        # The edit lands in the campaign's own `main` as a commit during the
        # first ideation: after the harness setup session, which works in
        # the campaign root, and before the attempt's session clone, which
        # is made from main's tip — so the clone inherits the edited
        # evaluator with no diff against its parent for the writer to
        # notice. (The release gate's first S5 edited the running clone's
        # working tree; the writer diffed against the parent, restored the
        # pristine file, and nothing was voided.) The integrity check
        # materialises the committed branch and reads its evaluator against
        # the sealed manifest: the void, with the hash evidence.
        self.wait_phase(campaign, "ideation", 1800)
        evaluator = campaign / "kapso_evaluation" / "evaluate.py"
        with open(evaluator, "a", encoding="utf-8") as out:
            out.write("\n# S5: an edit to the protected evaluator in the campaign's main branch, before the attempt's clone\n")
        self.ctx.shell(f"git -C {campaign} add kapso_evaluation/evaluate.py && git -C {campaign} -c user.name=S5 -c user.email=s5@kapso.invalid commit -q -m 'S5: the protected evaluator edited in main'")
        edited = [evaluator]
        injected_at = datetime.now(timezone.utc)
        status = self.finish(name, campaign)
        trace = self.trace(campaign)
        voided = trace.get("voided", [])
        # The void must be the integrity one, naming the evaluator, not any
        # judge rejection: that is the check this injection fires.
        integrity_voids = [v for v in voided if "evaluate.py" in str(v.get("reason", "")) or "integrity" in str(v.get("reason", "")).lower()]
        turn_1 = self.ask(name, project, f"what happened to the first attempt of the kapso campaign at ../s5-{name}? it is not counted")
        diagnosis = names_class_and_fix(turn_1.text, name)
        metrics = {"evaluator_edited": len(edited), "voided": len(voided), "integrity_voids": len(integrity_voids), "truthful": truthful_within(self.timeline(name), injected_at, self.cadence, expect_terminal=False), **diagnosis}
        artifacts = [self.ctx.artifact(f"{name}-answers.txt", turn_1.text), self.ctx.artifact(f"{name}-status.json", json.dumps(status, indent=2)), self.ctx.artifact(f"{name}-trace.json", json.dumps(trace, indent=2))]
        self.ctx.state["tampering_campaign"] = str(campaign)
        self.ctx.save()
        return self.record_injection(name, [turn_1], metrics, artifacts, ["the protected evaluator edited in the campaign's main during the first ideation, before the attempt's clone, so the writer sees no diff to restore; the integrity check voids the attempt with the hash evidence; the campaign is left running to its end, never stopped"],
                                     bool(edited) and bool(integrity_voids) and diagnosis["names_class"], started)

    def step_bank_compare(self) -> Dict[str, Any]:
        started = time.monotonic()
        after = bank_state(self.bank) if self.bank else {"head": None, "cards": None, "readable": None}
        readable = bool(self.bank) and self.bank_before.get("readable") and after.get("readable")
        intact = readable and after == self.bank_before and after.get("cards", 0) >= 1
        note = (f"the bank {self.bank}: {self.bank_before} before, {after} after"
                if self.bank else "no bank was loaded: the comparison has nothing to compare (said, a weakness of this run)")
        if self.bank and not readable:
            note += " — UNREADABLE before or after: the comparison fails rather than looking intact"
        step = shell_step("bank-compare", bool(intact), [self.ctx.artifact("bank-compare.json", json.dumps({"before": self.bank_before, "after": after}, indent=2))], (time.monotonic() - started) / 60, [note])
        step["metrics"] = {"bank_intact": intact if self.bank else None}
        return step


def class_completed(name: str, metrics: Dict[str, Any], state: Optional[str]) -> Optional[bool]:
    """Each class's completion from its recorded metrics and the campaign's
    final state — the predicate the step applies live, read again over the
    archive; None when a metric it needs is not in the record (an earlier
    driver's), in which case the completion stays as recorded."""
    needed = {
        "provider-limit": ("failures_with_the_limit", "names_class", "names_fix", "resumed"),
        "evaluator-crash": ("crash_occurred", "names_class", "names_fix", "resumed"),
        "missing-key": ("paused", "request_names_no_value", "authorised_table_fetch_after_reply", "scored_after_resume", "resumed"),
        "killed-session": ("session_killed", "failures", "names_class", "names_fix"),
        "killed-writer": ("injected", "shown_dead", "names_class", "names_fix", "resumed", "attempts_kept"),
        "full-disk": ("injected", "shown_dead", "names_class", "names_fix", "resumed", "attempts_kept"),
        "tampering": ("evaluator_edited", "integrity_voids", "names_class"),
    }
    if name not in needed or any(key not in metrics for key in needed[name]):
        return None
    m = metrics
    if name == "provider-limit":
        return bool(m["failures_with_the_limit"] and m["names_class"] and m["names_fix"] and (m["resumed"] or state == "done"))
    if name == "evaluator-crash":
        return bool(m["crash_occurred"] and m["names_class"] and m["names_fix"] and m["resumed"])
    if name == "missing-key":
        return bool(m["paused"] and m["request_names_no_value"] and m["authorised_table_fetch_after_reply"] and m["scored_after_resume"] and m["resumed"])
    if name == "killed-session":
        return bool(m["session_killed"] and m["failures"] and state in ("done", "failed") and m["names_class"] and m["names_fix"])
    if name in ("killed-writer", "full-disk"):
        return bool(m["injected"] and m["shown_dead"] and m["names_class"] and m["names_fix"] and m["resumed"] and m["attempts_kept"] is not False)
    return bool(m["evaluator_edited"] and m["integrity_voids"] and m["names_class"])


def rederive(record: Dict[str, Any], out: Path) -> Dict[str, Any]:
    """An archived run read again under the current rules where the archive
    allows: the diagnosis from the answers artifact (the vocabularies
    changed), the crash evidence from the trace and the campaign's sessions
    (the evaluator's own failure line, not a mention), the retention claim
    from the recorded attempt counts (not exercised when none had landed),
    the run metrics with the unobserved count. The truthfulness verdicts
    and each step's completion stay as recorded: the injection's moment is
    not in the archive, so the sample-by-sample rule cannot be re-applied,
    and the record says so."""
    for step in record["steps"]:
        if step.get("kind") != "injection" or step.get("deferred"):
            continue
        name, metrics = step["name"], step.setdefault("metrics", {})
        answers = out / "artifacts" / f"{name}-answers.txt"
        if answers.is_file():
            metrics.update(names_class_and_fix(answers.read_text(encoding="utf-8").split("\n\n---\n\n")[0], name))
        if name == "evaluator-crash":
            trace_file = out / "artifacts" / f"{name}-trace.json"
            if trace_file.is_file():
                trace = json.loads(trace_file.read_text(encoding="utf-8"))
                campaign = out / "work" / f"s5-{name}"
                metrics["attempts_with_the_crash"] = sum(1 for attempt in trace.get("attempts", []) if evaluator_crashed(campaign, attempt))
                metrics["crash_occurred"] = bool(metrics["attempts_with_the_crash"]) or crash_occurred(campaign, trace)
        if "attempts_kept" in metrics and not metrics.get("attempts_before"):
            metrics["attempts_kept"] = None
        if name == "missing-key" and metrics.get("paused"):
            # The pause came before any score by the rule's own construction
            # (the session asks for the key before it reads the data), so a
            # scored attempt in the final trace scored after the reply.
            trace_file = out / "artifacts" / f"{name}-trace.json"
            if trace_file.is_file():
                trace = json.loads(trace_file.read_text(encoding="utf-8"))
                metrics["scored_after_resume"] = any((attempt.get("evaluation") or {}).get("score") is not None for attempt in trace.get("attempts", []))
        status_file = out / "artifacts" / f"{name}-status.json"
        state = json.loads(status_file.read_text(encoding="utf-8")).get("state") if status_file.is_file() else None
        completed = class_completed(name, metrics, state)
        if completed is not None:
            step["completed"] = completed
        step["notes"] = [*[note for note in step.get("notes", []) if not note.startswith("re-read")],
                         f"re-read {datetime.now(timezone.utc).date().isoformat()} from the archive: the diagnosis under the disjoint vocabularies, the crash evidence from the evaluator's own failure line, retention not exercised where no attempt had landed, a paused campaign's score after the reply from its trace, the completion from the class's predicate over the record where every metric it needs is there; the truthfulness verdict as recorded (the injection's moment is not in the archive)"]
    record["metrics"] = run_metrics(record)
    record["notes"] = [*[note for note in record.get("notes", []) if not note.startswith("re-derived")],
                       f"re-derived {datetime.now(timezone.utc).date().isoformat()} under the current rules where the archive allows (each injection's note says what)"]
    return record


def run_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    injections = [step for step in record["steps"] if step.get("kind") == "injection" and not step.get("deferred")]
    truthful = [step["metrics"].get("truthful") for step in injections if step["metrics"].get("truthful") is not None]
    requests = [step["metrics"].get("request_names_no_value") for step in injections if step["metrics"].get("request_names_no_value") is not None]
    bank = next((step["metrics"].get("bank_intact") for step in record["steps"] if step["name"] == "bank-compare"), None)
    return {
        # The driver sends at most two messages by construction, so the
        # count is information, not a bar; a class not handled within them
        # reads false on its own predicate.
        "user_messages_max": max((step["metrics"]["user_messages"] for step in injections), default=None),
        "truthful_within_heartbeat": (sum(1 for value in truthful if value) / len(truthful)) if truthful else None,
        # An injection whose judged window held no sample is unobserved: it
        # leaves the ratio and is counted here, where the bar wants zero.
        "truthfulness_unobserved": sum(1 for step in injections if step["metrics"].get("truthful") is None),
        "requests_name_no_value": (sum(1 for value in requests if value) / len(requests)) if requests else None,
        "bank_intact": 1.0 if bank else (0.0 if bank is False else None),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clone", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--host", choices=["claude_code", "codex"], default="claude_code", help="the host the plugin's conversations run on")
    parser.add_argument("--wheel", default=None, help="the wheel the machine installed, pinned in the record by its sha256")
    parser.add_argument("--rederive", action="store_true", help="read an archived run again under the current rules where the archive allows (on the development box); runs nothing")
    parser.add_argument("--bank", default=None, help="a bank with at least one card, loaded into every campaign and compared before and after each injection")
    parser.add_argument("--cadence", type=float, default=None, help="the sampler's cadence; default: the engine config's budget.checkpoint_heartbeat_seconds")
    parser.add_argument("--machine", default=os.uname().nodename)
    parser.add_argument("--advisory", default=None)
    parser.add_argument("--steps", default=None)
    parser.add_argument("--from", dest="start", default=None)
    args = parser.parse_args(argv)
    if args.rederive:
        out = Path(args.out).resolve()
        record = rederive(json.loads((out / "record.json").read_text(encoding="utf-8")), out)
        (out / "record.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(record["metrics"]))
        return 0
    if not args.clone:
        parser.error("--clone is the checkout the injections run from (omitted only with --rederive)")
    driver = S5(args)
    record_file = driver.out / "record.json"
    started = datetime.now(timezone.utc)
    wheel_sha256 = hashlib.sha256(driver.ctx.wheel.read_bytes()).hexdigest() if driver.ctx.wheel else "-"
    record = json.loads(record_file.read_text()) if record_file.exists() else {
        "test": "S5", "host": args.host, "machine": args.machine, "run_id": f"S5-{args.host}-{started.strftime('%Y%m%dT%H%M%SZ')}", "advisory": args.advisory,
        "package": {"version": subprocess.run(["kapso", "--version"], capture_output=True, text=True).stdout.strip().split()[-1], "wheel_sha256": wheel_sha256},
        "pins": {"wheel_sha256": wheel_sha256, "suite_sha256": hashlib.sha256(plugin_proof.SUITE_PATH.read_bytes()).hexdigest(), "cadence_seconds": driver.cadence, "bank": str(driver.bank) if driver.bank else None},
        "user": {"kind": "scripted", "card": None, "interventions": [], "events": [], "pushback": None},
        "started": started.isoformat(timespec="seconds"), "finished": None, "steps": [], "metrics": {}, "notes": [RECONSTRUCTED_LIMIT["note"], *[note for note in [codex_session_note()] if note]], "transcripts": [],
    }
    steps = {"provider-limit": driver.step_provider_limit, "evaluator-crash": driver.step_evaluator_crash, "missing-key": driver.step_missing_key,
             "killed-session": driver.step_killed_session, "killed-writer": driver.step_killed_writer, "full-disk": driver.step_full_disk,
             "tampering": driver.step_tampering, "bank-compare": driver.step_bank_compare}
    assert list(steps) == STEP_NAMES
    selected = args.steps.split(",") if args.steps else list(STEP_NAMES)
    if args.start:
        selected = selected[selected.index(args.start):]
    for name in selected:
        print(f"== {name}", flush=True)
        step = steps[name]()
        record["steps"] = [existing for existing in record["steps"] if existing["name"] != name] + [step]
        record["metrics"] = run_metrics(record)
        record["user"]["interventions"] = sorted(set(record["user"]["interventions"]) | set(driver.ctx.user.interventions))
        record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        record["transcripts"] = sorted(f"transcripts/{path.name}" for path in (driver.out / "transcripts").glob("*.jsonl"))
        record_file.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in step.items() if key != "notes"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
