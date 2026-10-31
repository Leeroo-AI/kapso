#!/usr/bin/env python3
"""The plugin round (docs/plans/redesign/wp7-plugin.md, change 11): the
acceptance matrix's steps driven on one host by a scripted person (S2
without the simulated user and the judge), each step recorded in the
semantic record shape (tests/semantic/README.md, test `S2-mechanical`)
and rendered by tests/semantic/report.py against tests/semantic/suite.json.

    python3 scripts/redesign/plugin_proof.py --host claude_code|codex --clone <repo> --out <dir>
        [--work <dir>] [--steps a,b,...] [--from <step>] [--wheel <path>] [--machine <name>]

Runs where the host's CLI is logged in and the package is installed with
`kapso` and `kapso-mcp` on PATH: the Claude Code half on a fresh machine,
the Codex half on the development box. The record is written after every
step, so a crash keeps what ran and `--from` continues. Every step the
matrix marks deferred for the host is recorded as such. The driver uses
the standard library and PyYAML (the examples' python3 has it).
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import select
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

import yaml
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

# The persona driver sits beside this script; a run from anywhere finds it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulated_user  # noqa: E402
import judge as judge_module  # noqa: E402  (final_texts: the host's text on both vendors; litellm rides along)

SUITE_PATH = Path(__file__).resolve().parent.parent.parent / "tests" / "semantic" / "suite.json"
SUITE = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
CLAUDE_MODEL = SUITE["hosts"]["claude_code"]["model"]
CODEX_MODEL = SUITE["hosts"]["codex"]["model"]
TURN_TIMEOUT = SUITE["turn_timeout"]
_CHURN_B, _FRAUD_B, _PAUSE_B, _PROBE_B = SUITE["builds"]["churn"], SUITE["builds"]["fraud"], SUITE["builds"]["pause"], SUITE["builds"]["probe"]
# How long a step waits for a campaign or a learn: the step's own minutes
# from the suite, or the campaign's budget plus the suite's grace.
CAMPAIGN_GRACE_SECONDS = SUITE["waits"]["campaign_grace_minutes"] * 60


def step_seconds(name: str) -> int:
    return SUITE["steps"][name]["minutes"] * 60


def machine_hours(test: str, host: str) -> str:
    """A fresh machine's auto-delete cap for one test on one host: the
    host's required steps' budgets end to end, the campaign grace and an
    hour of provisioning, rounded up to the hour — a safety net past the
    whole run, never inside it (s2-claude-2 was deleted at its twenty-third
    step of twenty-four under a six-hour literal, its evidence with it)."""
    minutes = sum(SUITE["steps"][name]["minutes"] for name in SUITE["tests"][test]["required_steps"][host])
    return f"{math.ceil((minutes + SUITE['waits']['campaign_grace_minutes'] + 60) / 60)}h"
# A turn that waits for a word without a question mark: "say go and I'll launch".
# An approval request, matched only against the FINAL line: a plain
# courtesy ("let me know if you want the full trace") is not a question,
# so those weak phrases are gone and a real request has to end the turn.
APPROVAL_ASKED = re.compile(
    r"\b(say|reply|type)\b[^\n]{0,30}\b(go|yes)\b|\b(confirm|approve|approval)\b|\bshall i\b|\bwant me to\b|\bif you say\b|\bsay the word\b",
    re.I,
)
# The approval words without a request: a negation, or the host checking
# something itself.
NO_ASK = re.compile(r"\bno (approval|confirmation|go-ahead|input|answer) (is |was )?(needed|required)\b|\bwithout (your )?(approval|confirmation)\b", re.I)
SELF_CHECK = re.compile(r"^(let me|i'?ll|i will|i) (confirm|approve|verify|check)\b", re.I)
# A campaign that misses its goal delivers no branch; a review of it must say
# so rather than dress a near miss as something to merge.
NOTHING_DELIVERED = re.compile(
    r"\b(goal|target) (was )?not (met|reached)\b|\b(goal|target) (remains |is |still )?unmet\b|\bnothing (was )?delivered\b"
    r"|\bno (approved )?(winner|branch|result|delivery|candidate) (was )?delivered\b|\bdelivered no (approved )?(winner|branch|result|candidate)\b|\bdid not deliver\b"
    r"|\b(below|short of|under) the (target|threshold|goal)\b|\bmissed the (target|goal)\b",
    re.I,
)


def names_delivery(answer: str, branch: Optional[str]) -> bool:
    """The review names what the campaign delivered: its branch, when there
    is one; that nothing was delivered, when there is none — a statement
    of the fact (no winner, the goal unmet), never a bare score beside a
    number ("0.8459, below 0.85" says what scored, not what was delivered)."""
    if branch:
        return branch.lower() in answer.lower()
    return bool(NOTHING_DELIVERED.search(answer))


# The review skill's verdict comes as "is it safe to merge" answered, or as
# the one-line next action it prescribes (merge, cherry-pick, resume with
# more budget, learn from it) — a review that said "merging the branch
# whole carries the labels" and "resume with more time" had given it.
MERGE_VERDICT = re.compile(r"\bmerg(e|ing|ed)\b|\b(not )?safe\b|\bcherry-pick|\bresume with\b|\blearn from\b", re.I)
# A line that carries the words without a decision: the question itself,
# or a hedge ("I can't tell whether it's safe to merge").
NOT_A_VERDICT = re.compile(r"\bwhether\b|\bcan(no|')t (tell|say|judge|determine)\b|\bnot sure\b|\bunsure\b|\bunclear\b|\bunable to (tell|say|judge)\b", re.I)


def merge_verdict(answer: str) -> bool:
    """A merge verdict is a statement — merge, do not merge, not safe,
    cherry-pick, resume with more budget, learn from it — on some line of
    the answer; a line that asks the question or hedges it gives none."""
    for raw in answer.splitlines():
        line = raw.strip().rstrip("*_` ")
        if MERGE_VERDICT.search(line) and not line.endswith("?") and not NOT_A_VERDICT.search(line):
            return True
    return False


CHURN_WISH = (
    "get the churn model's ROC AUC above 0.85 without changing the prediction contract; "
    f"the campaign goes to ../churn-campaign with a budget of {_CHURN_B['minutes']} minutes and {_CHURN_B['experiments']} experiments"
)
PAUSE_WISH = (
    "get the churn model's ROC AUC above 0.85; one rule for the brief: the first thing every session does, before "
    "it reads the data or plans anything, is fetch the enrichment table from the enrichment service with the "
    "DATA_API_KEY in .env, and a session that lacks the key asks for it through the inbox at once and does nothing "
    "else until it has it. Launch now without asking me for the key: the campaign's own session asks "
    f"when it needs it. The campaign goes to ../churn-pause-campaign with a budget of {{minutes}} minutes and {_PAUSE_B['experiments']} experiment"
)
FRAUD_WISH = (
    "build a fraud scorer on data/transactions.csv that beats the rule-based one in score.py, and prove it; "
    f"the campaign goes to ../fraud-campaign with a budget of {_FRAUD_B['minutes']} minutes and {_FRAUD_B['experiments']} experiment"
)
# The order the round runs in; the budgets file bounds every name. The
# trigger probe runs last, on an otherwise idle machine: on Codex it is a
# hundred minutes of sessions one at a time, and when it followed
# build-churn the churn campaign lived its whole hour under that load — its
# ideation session, sixteen minutes alone, passed its 2250-second deadline
# and was killed, and the attempt voided (rel-s2-codex, fifth launch). The
# campaigns and the probe are each measured on their own. The MCP tool steps
# (campaign-tools, then session-brief) run after the learn, when no campaign
# loads the machine: the seventh Codex launch's campaign-tools session had no
# kapso tools though the server was registered, the sixth's had them, and a
# server Codex drops at startup leaves no diagnostic — an idle machine is the
# one lever the rig has.
STEP_NAMES = [
    "install", "setup", "cold-install", "build-churn", "build-pause", "watch", "harness-export",
    "inbox", "build-fraud", "build-data-only", "build-empty-wish", "harness-new", "harness-probe", "review",
    "deploy", "learn", "learn-complete", "campaign-tools", "session-brief", "experience-share", "second-engineer-build", "harness-improve",
    "campaign-session-isolation", "triggers", "codex-rules-probe", "cold-start",
]
# Codex's explicit skill for each Claude Code command.
CODEX_SKILL = {"setup": "kapso-setup", "build": "kapso-build", "review": "kapso-review", "harness": "kapso-harness"}
# Claude Code's launcher path, by design; every other step runs on Codex
# too (advisory on the development box until the machine's Codex key has
# credit: docs/plans/redesign/wp9-semantic-suite.md).
CODEX_DEFERRED = {
    "cold-install": "one machine: the cold wheel install is the fresh machine's step",
    "cold-start": "the launcher is Claude Code's path; Codex runs kapso-mcp from PATH",
}
# What each step is for, told to the persona with the card (the scripted
# policies' knowledge, as a person would hold it).
# The card a step is answered with when it differs from the run's card: the
# trading author knows the harness interview's module answers (patience 8);
# the second engineer owns the served build. The rest fall back to the run's card.
STEP_CARDS = {
    "harness-new": "trading-author",
    "harness-probe": "trading-author",
    "harness-improve": "trading-author",
}
STEP_INTENTS = {
    "setup": "set Kapso up on this machine to a green doctor; you have no Leeroopedia or OpenAI key and want none added; no harness for now",
    "build-churn": "get one proposal for the churn build and have it launched detached at the budget the brief names; you do not hold the data API key a rule may mention, the campaign's own session asks through the inbox when it needs it",
    "build-pause": "have the campaign launched now without being asked for the key: its own session asks through the inbox when it needs it",
    "watch": "how the campaign is going, in a few lines", "watch-2": "how the campaign is going, in a few lines",
    # The inbox step is two exchanges: the host reads the request and says what
    # the person must do (the key is NOT in .env yet; the driver writes it
    # after), then the host replies. A want that spans both can never be met
    # in the first, so each exchange has its own.
    "inbox": "hear what the campaign is waiting on and what I must do before it can continue; I will do it myself and come back",
    "inbox-reply": "answer every open request of the campaign: the key is now in both .env files; there is no enrichment service, endpoint or table; the campaign must reach 0.85 without it; nothing left waiting",
    "build-fraud": "a fraud scorer that beats the rule-based one, with an evaluator proposed from the protocol and approved by you, then launched",
    "build-data-only": "see the candidate criteria and the recommended one; take the first; do not launch anything",
    "build-empty-wish": "see what could be built here and the candidate criteria; take the first; do not launch anything",
    "harness-new": "a proposal first, then say go so the harness is written and checked green — a proposal alone is not done, and DONE comes only once the harness is on disk and its check is green; answer every numbered question in one reply, in order",
    "review": "the review of the finished churn campaign, then whether its branch is safe to merge",
    "review-2": "whether the delivered branch is safe to merge",
    "deploy": "deploy the delivered churn campaign to the local target and call it once with the inputs its proof used",
    "deploy-2": "call the deployment once with the inputs its proof used",
    "learn": "launch the learn from the churn campaign, detached, and hear how it will be served",
    "session-brief": "what this session is served and one read of the bank",
    "experience-share": "share your bank to the remote given, so a teammate can read it",
    "harness-improve": "one improvement proposed from the probe's trace, applied on your go, the check green again",
    "campaign-tools": "launch a tiny campaign through the session's tools rather than the shell, read its status, stop it",
    "harness-export": "what timestamp discipline the backtesting skill asks for, in three lines",
}
CLAUDE_DEFERRED = {
    "codex-rules-probe": "Codex's mechanism; Claude Code's is the plugin switch, which the isolation step reads",
}


# ----------------------------------------------------------------- hosts
class Turn:
    def __init__(self, text: str, seconds: float, session: str, tools: List[Dict[str, Any]], asked: bool):
        self.text, self.seconds, self.session, self.tools, self.asked = text, seconds, session, tools, asked


def question_asked(text: str, tools: Sequence[Dict[str, Any]]) -> bool:
    """A turn that ends asking: the AskUserQuestion tool, or a question
    among its closing lines (an approval request is often followed by a
    sentence on why it is needed)."""
    if any(tool["name"] == "AskUserQuestion" for tool in tools):
        return True
    # A fence line is nothing once its backticks are stripped; it is dropped
    # so the closing lines are the closing prose and commands.
    lines = [line for line in (raw.strip().rstrip("*_`)\"' ") for raw in text.splitlines()) if line]
    # The command the ask refers to, shown after it (a fenced block, a
    # backtick-wrapped line, a `kapso ...` line), is not the ask's last
    # word: three steps on the Claude host ended "Say go and I'll run:"
    # and the command, and read as never asking (2026-10-28).
    while lines and (lines[-1].startswith("`") or lines[-1].startswith("kapso ") or lines[-1].startswith("$ ")):
        lines.pop()
    if not lines:
        return False
    # A question mark or an approval request in the closing lines: the
    # regex names real requests only (a courtesy sign-off never matches),
    # and a host closes "Say go and I'll run:", the command, "or edit a
    # line first" — the request three lines up, its continuation last.
    closing = lines[-3:]
    if any("?" in line for line in closing):
        return True
    # An approval request is a request: "no approval is needed" and "let
    # me confirm the data" carry the word without asking for anything.
    return any(APPROVAL_ASKED.search(line) and not NO_ASK.search(line) and not SELF_CHECK.match(line) for line in closing)


def lines_outside_fences(text: str) -> int:
    count, fenced = 0, False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced and line.strip():
            count += 1
    return count


def parse_claude_stream(stdout: str, sid: str, elapsed: float):
    """A Claude Code stream-json turn: the final text, the tool calls with
    whether each one's result was an error (the result rides in a user
    row's tool_result block; None when no result row came, the stream
    having ended first), the turn's seconds and the session id."""
    text, tools, seconds = "", [], elapsed
    results: Dict[str, bool] = {}   # a tool call's id → whether its result was an error
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        if event.get("type") == "assistant":
            for block in event["message"].get("content", []):
                if block.get("type") == "tool_use":
                    tools.append({"id": block.get("id"), "name": block["name"], "input": block.get("input", {})})
        elif event.get("type") == "user":
            for block in event.get("message", {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    results[block.get("tool_use_id")] = bool(block.get("is_error"))
        elif event.get("type") == "result":
            text = event.get("result") or ""
            seconds = event.get("duration_ms", seconds * 1000) / 1000
            sid = event.get("session_id", sid)
    for tool in tools:
        tool["is_error"] = results.get(tool.pop("id"))
    return text, tools, seconds, sid


def parse_codex_stream(stdout: str, sid: str):
    """A Codex exec --json turn: the tool calls (a command with its exit
    code, an MCP call with its completed item's status and error) and the
    thread id."""
    tools: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        if event.get("type") == "thread.started":
            sid = event["thread_id"]
        item = event.get("item") or {}
        if event.get("type") == "item.completed" and item.get("type") == "command_execution":
            tools.append({"name": "Bash", "input": {"command": item.get("command", "")}, "is_error": item.get("exit_code", 0) != 0})
        elif event.get("type") == "item.completed" and item.get("type") == "mcp_tool_call":
            tools.append({"name": f"mcp__{item.get('server')}__{item.get('tool')}", "input": item.get("arguments", {}),
                          "is_error": item.get("status") != "completed" or bool(item.get("error"))})
    return tools, sid


class ClaudeHost:
    name = "claude_code"

    def __init__(self, extra_argv: Sequence[str] = (), model: Optional[str] = None):
        # What a driver adds to every turn: the cold engineer's session runs
        # plugin-free, server-free and web-free through these. The model is
        # the suite's host model unless a role names its own (S4's readers).
        self.extra_argv = list(extra_argv)
        self.model = model or CLAUDE_MODEL

    def command(self, verb: str, args: str = "") -> str:
        return f"/kapso:{verb} {args}".strip()

    def turn(self, cwd: Path, prompt: str, session: Optional[str], transcript: Path) -> Turn:
        sid = session or str(uuid.uuid4())
        # The turn's deadline is the `timeout` command's: a turn that
        # overstays ends with exit 124 and its stream so far, and the round
        # goes on with what it has.
        argv = ["timeout", str(TURN_TIMEOUT), "claude", "-p", "--dangerously-skip-permissions", "--output-format", "stream-json", "--verbose", "--model", self.model, *self.extra_argv]
        argv += ["--resume", sid] if session else ["--session-id", sid]
        started = time.monotonic()
        run = subprocess.run(argv, input=prompt, capture_output=True, text=True, cwd=str(cwd))
        with open(transcript, "a", encoding="utf-8") as out:
            out.write(json.dumps({"proof_prompt": prompt}) + "\n" + run.stdout + ("\n" if not run.stdout.endswith("\n") else ""))
            if run.stderr.strip():
                out.write(json.dumps({"proof_stderr": run.stderr}) + "\n")
        text, tools, seconds, sid = parse_claude_stream(run.stdout, sid, time.monotonic() - started)
        return Turn(text, seconds, sid, tools, question_asked(text, tools))


class CodexHost:
    name = "codex"

    def __init__(self, extra_argv: Sequence[str] = (), model: Optional[str] = None, effort: Optional[str] = None):
        self.extra_argv = list(extra_argv)
        self.model = model or CODEX_MODEL
        self.effort = effort   # a role's reasoning effort (S4's reader); None leaves the CLI's own

    def command(self, verb: str, args: str = "") -> str:
        return f"${CODEX_SKILL.get(verb, 'kapso')} {verb if verb not in CODEX_SKILL else ''} {args}".replace("  ", " ").strip()

    def turn(self, cwd: Path, prompt: str, session: Optional[str], transcript: Path) -> Turn:
        last = cwd / ".proof-last.txt"
        argv = ["timeout", str(TURN_TIMEOUT), "codex", "exec", "--json", "--skip-git-repo-check", "--sandbox", "danger-full-access", "-C", str(cwd), "-m", self.model, "-o", str(last), *self.extra_argv]
        if self.effort:
            argv += ["-c", f'model_reasoning_effort="{self.effort}"']
        if session:
            argv += ["resume", session, "-"]
        else:
            argv.append(prompt)
        started = time.monotonic()
        run = subprocess.run(argv, input=prompt if session else None, capture_output=True, text=True, cwd=str(cwd))
        seconds = time.monotonic() - started
        with open(transcript, "a", encoding="utf-8") as out:
            out.write(json.dumps({"proof_prompt": prompt}) + "\n" + run.stdout + ("\n" if not run.stdout.endswith("\n") else ""))
            if run.stderr.strip():
                out.write(json.dumps({"proof_stderr": run.stderr}) + "\n")
        tools, sid = parse_codex_stream(run.stdout, session or "")
        text = last.read_text(encoding="utf-8") if last.exists() else ""
        if last.exists():
            last.unlink()
        return Turn(text, seconds, sid, tools, question_asked(text, tools))


# --------------------------------------------------------------- context
class UserMode:
    """Who answers the host: the scripted policies of the mechanical round,
    or a persona card through the other vendor's CLI. The driver's own
    acts are named as interventions either way; a persona run records
    the pins the review asked for, every DONE and every turn cap."""

    def __init__(self, spec: str, host: str):
        if spec != "scripted" and not spec.startswith("persona:"):
            raise ValueError(f"--user is scripted or persona:<card>, not {spec!r}")
        self.kind = "scripted" if spec == "scripted" else "persona"
        self.host = host
        self.card = None
        self.cards: Dict[str, Any] = {}
        self.settings: Dict[str, Any] = {}
        self.turn_cap = SUITE["user"]["turn_cap"]
        self.prompt_sha256 = None
        if self.kind == "persona":
            name = spec.split(":", 1)[1]
            self.card = simulated_user.load_card(simulated_user.PERSONAS / f"{name}.yaml")
            self.settings = SUITE["user"]["persona"][host]
            self.prompt_sha256 = simulated_user.template_sha256()
        self.interventions: List[str] = []
        self.events: List[str] = []
        self.capped: set = set()
        self.pushback: Optional[bool] = None

    def card_for(self, step: str):
        """The card a step is answered with: a per-step card (the trading
        author answers the harness interview, the second engineer the
        served build) falls back to the run's card. Loaded once and cached."""
        name = STEP_CARDS.get(step)
        if not name or self.kind != "persona":
            return self.card
        if name not in self.cards:
            self.cards[name] = simulated_user.load_card(simulated_user.PERSONAS / f"{name}.yaml")
        return self.cards[name]

    def ended_well(self, step: str, predicate: bool) -> bool:
        """A persona step is complete only when its evidence holds AND the
        exchange resolved (the persona said DONE), never when the turn cap
        cut a user still asking for help off."""
        return bool(predicate) and (self.kind != "persona" or step not in self.capped)

    def record(self) -> Dict[str, Any]:
        block: Dict[str, Any] = {"kind": self.kind, "card": self.card.name if self.card else None, "interventions": self.interventions, "events": self.events, "pushback": self.pushback}
        if self.card:
            block.update({"card_sha256": self.card.sha256, "vendor": self.settings["vendor"], "model": self.settings["model"], "effort": self.settings.get("effort"), "prompt_sha256": self.prompt_sha256})
        return block


class Context:
    def __init__(self, host, clone: Path, work: Path, out: Path, wheel: Optional[Path], machine: str, learn_model: Optional[str] = None, pause_minutes: int = _PAUSE_B["minutes"], user: Optional[UserMode] = None,
                 triggers_sandbox: Optional[str] = None):
        self.host, self.clone, self.work, self.out, self.wheel, self.machine = host, clone, work, out, wheel, machine
        self.learn_model = learn_model
        self.pause_minutes = pause_minutes
        self.triggers_sandbox = triggers_sandbox   # the runner's word for a machine whose codex sandbox cannot start (None: the suite's)
        self.user = user or UserMode("scripted", host.name)
        self.state_file = out / "state.json"
        self.state: Dict[str, Any] = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        (out / "transcripts").mkdir(parents=True, exist_ok=True)
        (out / "artifacts").mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        self.state_file.write_text(json.dumps(self.state, indent=1) + "\n", encoding="utf-8")

    def shell(self, command: str, cwd: Optional[Path] = None, timeout: int = 1800, check: bool = True) -> subprocess.CompletedProcess:
        run = subprocess.run(["bash", "-c", command], capture_output=True, text=True, cwd=str(cwd or self.work), timeout=timeout)
        with open(self.out / "shell.log", "a", encoding="utf-8") as log:
            log.write(f"$ {command}\n{run.stdout}{run.stderr}[exit {run.returncode}]\n")
        if check and run.returncode != 0:
            raise RuntimeError(f"{command!r} exited {run.returncode}: {run.stderr.strip()[:500]}")
        return run

    def artifact(self, name: str, text: str) -> str:
        path = self.out / "artifacts" / name
        path.write_text(text, encoding="utf-8")
        return f"artifacts/{name}"

    def copy_artifact(self, source: Path, name: str) -> str:
        target = self.out / "artifacts" / name
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        return f"artifacts/{name}"

    def converse(self, step: str, cwd: Path, opening: str, answer: Callable[[str, int], Optional[str]], max_turns: int = 8) -> List[Turn]:
        """The exchange of one step. Scripted: the policy answers a turn that
        asks, and a turn that does not ask ends the step. Persona: every
        host turn is offered to the persona, who answers or says DONE; the
        turn cap ends the step as an event, never a pass."""
        transcript = self.out / "transcripts" / f"{step}.jsonl"
        turns: List[Turn] = []
        session, prompt = None, opening
        if self.user.kind == "scripted":
            for index in range(max_turns):
                turn = self.host.turn(cwd, prompt, session, transcript)
                turns.append(turn)
                session = turn.session
                if not turn.asked:
                    break
                reply = answer(turn.text, index)
                if reply is None:
                    break
                prompt = reply
            return turns
        history: List[Dict[str, Any]] = []
        intent = STEP_INTENTS.get(step, step)
        card = self.user.card_for(step)
        if card is not self.user.card:
            self.user.interventions.append(f"{step}: answered by the {card.name} card (its knowledge the step needs), not the run's {self.user.card.name}")
        # The exchange's cap is the step's own message budget (a sub-exchange
        # such as review-2 takes its parent's), so a legitimate interview is
        # never cut short of what the suite allows and a step over it fails
        # on its count; the persona's turn cap stands only where the suite
        # has no entry. `max_turns` is the scripted loop's.
        cap = SUITE["steps"].get(parent_step(step), {}).get("messages") or self.user.turn_cap
        for index in range(cap):
            turn = self.host.turn(cwd, prompt, session, transcript)
            turns.append(turn)
            session = turn.session
            history.append({"role": "user", "text": prompt})
            history.append({"role": "assistant", "text": turn.text, "asked": turn.asked})
            reply = simulated_user.next_message(card, cwd, intent, history, turn.text, self.host.name, self.user.settings, TURN_TIMEOUT)
            with open(transcript, "a", encoding="utf-8") as out:
                out.write(json.dumps({"persona_reply": reply.text, "persona_session": reply.session, "done": reply.done}) + "\n")
            if reply.done:
                self.user.events.append(f"{step}: the persona said DONE after {index + 1} turn(s)")
                return turns
            prompt = reply.text
        self.user.events.append(f"{step}: the turn cap ({cap}) ended the exchange without DONE: the step is incomplete")
        self.user.capped.add(parent_step(step))   # a capped follow-up (review-2) makes its step incomplete, not a name nothing reads
        return turns

    def task_view(self, project: Path, planted: Optional[str] = None) -> None:
        """What the persona reads: the repository's own README (never the
        example's, which teaches the Kapso lines and quotes the recorded
        result), with a planted wrong fact when the step checks pushback."""
        own = project / "repo" / "README.md"
        text = own.read_text(encoding="utf-8") if own.is_file() else f"# {project.name}\n\nA repository with no README of its own.\n"
        if planted:
            text = text.rstrip() + f"\n\n{planted}\n"
        (project / "README.md").write_text(text, encoding="utf-8")

    def fixture(self, name: str, target: str, generate: bool = True) -> Path:
        """A copy of an example under the work directory, its data generated,
        the session's credentials in its `.env`."""
        path = self.work / target
        if path.exists():
            shutil.rmtree(path)
        # A copy as a clone would be: no generated data, no private hold-out,
        # no evaluator result left by an earlier run on the source checkout.
        shutil.copytree(self.clone / "examples" / name, path, ignore=shutil.ignore_patterns("campaign", "__pycache__", "data", "private", "result.json"))
        if generate:
            self.shell("python3 generate.py", cwd=path, timeout=600)
        self.seed_env(path)
        if self.user.kind == "persona":
            self.task_view(path)
            self.user.interventions.append(f"{target}: the example's README replaced by the repository's own for the persona (no Kapso line, no recorded result)")
        return path

    def seed_env(self, project: Path) -> None:
        """A host session's shell carries no `CLAUDE_CODE_OAUTH_TOKEN` (the
        host keeps its own token from its tools), so a machine logged in
        by token puts it in the project's `.env`, the doctor's fix line;
        a machine with a stored login has nothing to seed."""
        token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        if token:
            with open(project / ".env", "a", encoding="utf-8") as env:
                env.write(f"CLAUDE_CODE_OAUTH_TOKEN={token}\n")


def project_config(project: Path, blocks: Dict[str, Any]) -> Path:
    """The project's kapso-config.yaml with `blocks` merged over what it
    holds; the skills pass it on every line."""
    path = project / "kapso-config.yaml"
    current = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    current = current or {}
    for key, value in blocks.items():
        current[key] = {**current.get(key, {}), **value} if isinstance(value, dict) and isinstance(current.get(key), dict) else value
    path.write_text(yaml.safe_dump(current, sort_keys=False), encoding="utf-8")
    return path


def effective_config(ctx: "Context", project: Path) -> Dict[str, Any]:
    return yaml.safe_load(ctx.shell("kapso doctor --show-config" + (" --config kapso-config.yaml" if (project / "kapso-config.yaml").is_file() else ""), cwd=project).stdout)


def learning_on(block: Any, model: str, replaced: str) -> Any:
    """The learning block with every crew on `model` instead of `replaced`."""
    if isinstance(block, dict):
        return {key: learning_on(value, model, replaced) for key, value in block.items()}
    if isinstance(block, list):
        return [learning_on(item, model, replaced) for item in block]
    return model if block == replaced else block


def parent_step(name: str) -> str:
    """The suite step an exchange belongs to: the step itself; a numbered
    follow-up's base (`review-2` → `review`); a named sub-exchange's head
    (`inbox-reply` → `inbox`); else the name as given."""
    if name in SUITE["steps"]:
        return name
    base = re.sub(r"-\d+$", "", name)
    if base in SUITE["steps"]:
        return base
    head = name.split("-", 1)[0]
    return head if head in SUITE["steps"] else name


def codex_session_note() -> Optional[str]:
    """What the machine's Codex login is, when the runner marked it
    (fresh_machine.sh writes ~/.codex/SOURCE beside the session it pushed):
    a record made on a borrowed session says so in its notes."""
    marker = Path.home() / ".codex" / "SOURCE"
    return f"codex login: {marker.read_text(encoding='utf-8').strip()}" if marker.is_file() else None


def wait_for(predicate: Callable[[], bool], seconds: int, every: int = 10) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(every)
    return predicate()


def status_of(campaign: Path) -> Optional[Dict[str, Any]]:
    path = campaign / ".kapso" / "status.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def record_step(name: str, turns: Sequence[Turn], completed: bool, artifacts: Sequence[str], notes: Sequence[str] = (), extra_minutes: float = 0.0, kind: str = "conversation") -> Dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "messages": len(turns),
        "questions": sum(1 for turn in turns if turn.asked),
        "minutes": round(sum(turn.seconds for turn in turns) / 60 + extra_minutes, 1),
        "longest_message_lines": max((lines_outside_fences(turn.text) for turn in turns), default=0),
        "completed": completed,
        "artifacts": list(artifacts),
        "notes": list(notes),
    }


def shell_step(name: str, completed: bool, artifacts: Sequence[str], minutes: float, notes: Sequence[str] = (), kind: str = "shell") -> Dict[str, Any]:
    return record_step(name, [], completed, artifacts, notes, extra_minutes=round(minutes, 1), kind=kind)


# --------------------------------------------------------- answer policies
def go_policy(text: str, index: int) -> Optional[str]:
    lower = text.lower()
    if "key" in lower and ("api" in lower or ".env" in lower or "token" in lower):
        return "Launch anyway: the campaign's own session asks through the inbox when it needs the key. Go."
    return "go"


def setup_policy(text: str, index: int) -> Optional[str]:
    lower = text.lower()
    if "leeroopedia" in lower or "openai" in lower or "key" in lower:
        return "I have neither key; continue."
    if "harness" in lower:
        return "No harness for now."
    return "yes"


def stop_here_policy(text: str, index: int) -> Optional[str]:
    return "The first one. Stop here: do not launch anything." if index == 0 else None


HARNESS_ANSWERS = [
    ("ideation", "Vary the signal family (trend, mean reversion, volatility targeting), the lookback, and the sizing rule; one axis per attempt."),
    ("verification", "Before a candidate counts: the look-ahead scan passes, the backtest covers 2021 to 2023, and the risk line (max drawdown, annual turnover) is printed."),
    ("feedback", "Prefer robust over spectacular: a consistent Sharpe across years beats one great year; say which axis to try next."),
    ("deployment", "Paper trading only; the target is the deploy call's."),
    ("learning", "Nothing to add; the defaults."),
    ("setup", "Pin the data snapshot once and record its hash; every session reads the snapshot, never a live fetch."),
]


def harness_policy(text: str, index: int) -> Optional[str]:
    """The proposal gets "go"; a turn that asks about modules gets every
    answer its questions call for, in one reply, as a person answering a
    numbered list would."""
    lower = text.lower()
    if index == 0 and ("confirm" in lower or "go" in lower or "proposal" in lower or "write" in lower):
        return "go"
    answers = [f"{key}: {answer}" for key, answer in HARNESS_ANSWERS if key in lower]
    if answers:
        return "\n".join(answers) + "\nNothing else to add; write them in the cheapest form and run the check."
    return "Use your best judgement from the repository; nothing else to add."


# ------------------------------------------------------------------ steps
def step_install(ctx: Context) -> Dict[str, Any]:
    started = time.monotonic()
    artifacts, notes = [], []
    if ctx.host.name == "claude_code":
        ctx.shell(f"claude plugin marketplace add {ctx.clone}")
        ctx.shell("claude plugin install kapso@kapso")
        validate = ctx.shell(f"claude plugin validate {ctx.clone / 'plugin'} --strict --json", check=False)
        artifacts.append(ctx.artifact("validate.json", validate.stdout))
        details = ctx.shell("claude plugin details kapso@kapso", check=False)
        artifacts.append(ctx.artifact("details.txt", details.stdout + details.stderr))
        listing = ctx.shell("claude plugin list", check=False)
        artifacts.append(ctx.artifact("plugin-list.txt", listing.stdout))
        completed = validate.returncode == 0 and "kapso" in listing.stdout
    else:
        ctx.shell(f"codex plugin marketplace add {ctx.clone}")
        ctx.shell("codex plugin add kapso@kapso")
        probe = ctx.host.turn(ctx.work, "$kapso Do not run any command: in one sentence, say which skill you loaded and stop.", None, ctx.out / "transcripts" / "install.jsonl")
        artifacts.append(ctx.artifact("install-probe.txt", probe.text))
        completed = "kapso" in probe.text.lower()
        notes.append("Codex: the plugin's skills load by name; its MCP tools reach a session through the `codex mcp add` line the setup skill runs")
    return shell_step("install", completed, artifacts, (time.monotonic() - started) / 60, notes)


def step_setup(ctx: Context) -> Dict[str, Any]:
    project = ctx.fixture("churn-first-build", "churn")
    ctx.state["churn_project"] = str(project)
    ctx.save()
    turns = ctx.converse("setup", project, ctx.host.command("setup"), setup_policy)
    doctor = ctx.shell("kapso doctor build --json", cwd=project, check=False)
    artifacts = [ctx.artifact("setup-doctor.json", doctor.stdout)]
    rows = json.loads(doctor.stdout) if doctor.stdout.strip().startswith(("{", "[")) else {}
    checks = rows.get("checks", rows) if isinstance(rows, dict) else rows
    required_ok = all(row.get("ok") for row in checks if isinstance(row, dict) and row.get("required")) if isinstance(checks, list) else doctor.returncode == 0
    notes = ["link clicks: judged by hand from the transcript (the logins were present, so none were needed)"]
    if ctx.host.name == "codex":
        # The registration is the setup skill's step 4, and every later
        # step's tools hang on it: a setup session that skipped the line
        # beside a green doctor (rel-s2-codex, fifth launch: no `codex mcp
        # add` ran, the list came back empty, and campaign-tools and
        # session-brief had no tools to use) is a setup not done, failed
        # here where it belongs rather than three steps later.
        servers = ctx.shell("codex mcp list", check=False)
        artifacts.append(ctx.artifact("codex-mcp-list.txt", servers.stdout))
        registered = "kapso" in servers.stdout
        required_ok = required_ok and registered
        notes.append("the `codex mcp add kapso -- kapso-mcp` line's effect is the mcp list artifact" + ("" if registered else "; the list shows no kapso server: the setup session did not register it, and no later session has the tools"))
    return record_step("setup", turns, required_ok, artifacts, notes)


SECOND_USER = "proof2"


def second_user_install_script(wheel_name: str) -> str:
    """The second engineer's own install, from its own home: `kapso` reads
    `.env` from the working directory, and a directory the user cannot
    see is no place to run."""
    return (
        "set -e; cd ~; curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; export PATH=$HOME/.local/bin:$PATH; "
        f"uv tool install --python 3.12 /tmp/{wheel_name} >/dev/null 2>&1; echo kapso $(kapso --version); echo kapso-mcp $(kapso-mcp --version); "
        "echo which $(which kapso) $(which kapso-mcp)"
    )


def stage_wheel_for_second_user(ctx: Context) -> Path:
    """The wheel where another user can read it."""
    wheel = ctx.wheel
    if wheel.resolve() != Path("/tmp") / wheel.name:
        ctx.shell(f"sudo cp {wheel} /tmp/{wheel.name}")
    ctx.shell(f"sudo chmod 644 /tmp/{wheel.name}")
    return wheel


def ensure_second_user(ctx: Context) -> None:
    """The second engineer exists with the wheel on its PATH. The Claude
    host's cold-install step measures that install; the Codex step list
    has no cold-install, so the steps that act as the second user make
    sure of it here (a no-op once it is there). A fresh Codex machine
    (2026-10-28) had no such user and the ninth step died on it."""
    ctx.shell(f"id -u {SECOND_USER} >/dev/null 2>&1 || sudo useradd -m -s /bin/bash {SECOND_USER}")
    # A second engineer has a git identity, as any person running kapso
    # does; without one the engine's first commit in the campaign dies with
    # git's "empty ident name" after the launch receipt (s2-claude-second-2),
    # the finding below — the suite gives the user one and keeps the finding.
    # From that user's own home: git stats the working directory first, and
    # the driver's is not readable to the second user (s2-claude-second-3
    # died here, on this line's first form).
    ctx.shell(f"sudo -u {SECOND_USER} -H bash -lc 'cd ~ && git config --global user.name \"Second Engineer\" && git config --global user.email second-engineer@kapso.invalid'")
    # The engine's build preflight wants a Codex login (two of its roles run
    # on gpt-5.6-sol), so the second engineer gets the machine's device-auth
    # session too — s2-claude-3's served build died at preflight without it.
    ctx.shell(f"sudo -u {SECOND_USER} -H bash -lc 'umask 077; mkdir -p ~/.codex' && sudo cp /home/ubuntu/.codex/auth.json /home/{SECOND_USER}/.codex/auth.json "
              f"&& sudo chown {SECOND_USER}:{SECOND_USER} /home/{SECOND_USER}/.codex/auth.json && sudo chmod 600 /home/{SECOND_USER}/.codex/auth.json")
    present = ctx.shell(f"sudo -u {SECOND_USER} -H bash -lc 'export PATH=$HOME/.local/bin:$PATH; command -v kapso'", check=False)
    if present.returncode == 0:
        return
    wheel = stage_wheel_for_second_user(ctx)
    ctx.shell(f"sudo -u {SECOND_USER} -H bash -lc {shlex.quote(second_user_install_script(wheel.name))}", timeout=1200)


def step_cold_install(ctx: Context) -> Dict[str, Any]:
    started = time.monotonic()
    user = SECOND_USER
    ctx.shell(f"id -u {user} >/dev/null 2>&1 || sudo useradd -m -s /bin/bash {user}")
    wheel = stage_wheel_for_second_user(ctx)
    run = ctx.shell(f"sudo -u {user} -H bash -lc {shlex.quote(second_user_install_script(wheel.name))}", timeout=1200, check=False)
    text = run.stdout + run.stderr
    # The lines read `kapso kapso 0.8.0.dev1`: the label, then what --version printed.
    versions = re.findall(r"^(kapso|kapso-mcp) (?:\1 )?(\S+)$", run.stdout, re.M)
    completed = run.returncode == 0 and len({version for _, version in versions}) == 1 and len(versions) == 2
    return shell_step("cold-install", completed, [ctx.artifact("cold-install.txt", text)], (time.monotonic() - started) / 60)


def _launch_and_wait_status(ctx: Context, project: Path, campaign: Path, wish: str, step: str, policy=go_policy) -> Dict[str, Any]:
    turns = ctx.converse(step, project, ctx.host.command("build", wish), policy, max_turns=4)
    brief = project / "kapso-brief.md"
    # A LIVE launch: the campaign reaches running (or a non-failed terminal),
    # not merely a status file a child wrote before failing on start.
    launched = wait_for(lambda: (status_of(campaign) or {}).get("state") in ("starting", "running", "done"), 300)
    state = (status_of(campaign) or {}).get("state")
    # A proposal was offered before the launch (the host asked), the step's
    # shape; the brief the host wrote names the objective.
    proposed = bool(turns) and any(turn.asked for turn in turns)
    artifacts = []
    if brief.is_file():
        artifacts.append(ctx.copy_artifact(brief, f"{step}-brief.md"))
    if status_of(campaign) is not None:
        artifacts.append(ctx.copy_artifact(campaign / ".kapso" / "status.json", f"{step}-status.json"))
    return {"turns": turns, "launched": launched, "proposed": proposed, "brief": brief.is_file(), "state": state, "artifacts": artifacts,
            "notes": [f"the launch reached: {state}; a proposal was offered: {proposed}"]}


PLANTED_BASELINE = "The current model scores about 0.90 ROC AUC on the held-out rows."
PLANTED_TRUTH = "the current churn model's ROC AUC is about 0.80 (0.797 on the held-out rows), not 0.90"


def plant_baseline(ctx: Context, project: Path) -> None:
    """The pushback check: the task view and the brief state the baseline
    as 0.90 where the card knows about 0.80; a proposal carries the wrong
    figure and a persona that reads corrects it."""
    ctx.task_view(project, PLANTED_BASELINE)
    brief = project / "kapso-brief.md"
    lines = brief.read_text(encoding="utf-8").splitlines(keepends=True)
    brief.write_text("".join("Baseline     0.90 (the shipped logistic regression)\n" if line.startswith("Baseline") else line for line in lines), encoding="utf-8")
    ctx.user.card.planted = PLANTED_TRUTH
    ctx.user.interventions.append("build-churn: the task view's README and the brief's Baseline line state 0.90 (planted); the card is certain of about 0.80")


def pushed_back(transcript: Path) -> Optional[bool]:
    """Whether the persona corrected the planted figure: None when no host
    turn carried it (nothing to push back on), else whether a persona reply
    named the true one. The host's text is read from BOTH vendors' row
    shapes (judge.final_texts), so the Codex arm's check is not blind."""
    rows = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines() if line.startswith("{")]
    stated = any("0.90" in turn["text"] for turn in judge_module.final_texts(rows) if turn["role"] == "assistant")
    corrected = any("persona_reply" in row and any(figure in row["persona_reply"] for figure in ("0.80", "0.797", "0.8 ")) for row in rows)
    return corrected if stated else None


def step_build_churn(ctx: Context) -> Dict[str, Any]:
    # The fixture again, clean: the setup step's session may have touched it.
    project = ctx.fixture("churn-first-build", "churn")
    campaign = ctx.work / "churn-campaign"
    ctx.state["churn_campaign"] = str(campaign)
    ctx.state["churn_launched_at"] = time.time()
    ctx.save()
    if ctx.user.kind == "persona":
        plant_baseline(ctx, project)
    result = _launch_and_wait_status(ctx, project, campaign, CHURN_WISH, "build-churn")
    if ctx.user.kind == "persona":
        ctx.user.card.planted = None
        ctx.user.pushback = pushed_back(ctx.out / "transcripts" / "build-churn.jsonl")
        result["notes"].append(f"the planted baseline: {'corrected by the persona' if ctx.user.pushback else 'not corrected' if ctx.user.pushback is False else 'never stated by the host'}")
    completed = ctx.user.ended_well("build-churn", result["launched"] and result["proposed"] and result["brief"])
    return record_step("build-churn", result["turns"], completed, result["artifacts"], result["notes"])


def step_build_pause(ctx: Context) -> Dict[str, Any]:
    """The injected pause for the inbox step: a brief rule naming a key
    `.env` lacks, launched beside the churn campaign at a tiny budget."""
    project = ctx.fixture("churn-first-build", "churn-pause")
    campaign = ctx.work / "churn-pause-campaign"
    if campaign.exists():
        ctx.shell(f"kapso build --output {campaign} --stop", check=False)
        shutil.rmtree(campaign)
    ctx.state["pause_campaign"] = str(campaign)
    ctx.save()
    result = _launch_and_wait_status(ctx, project, campaign, PAUSE_WISH.format(minutes=ctx.pause_minutes), "build-pause")
    return record_step("build-pause", result["turns"], result["launched"] and result["brief"], result["artifacts"])


def step_watch(ctx: Context) -> Dict[str, Any]:
    project, campaign = Path(ctx.state["churn_project"]), ctx.state["churn_campaign"]
    wait_for(lambda: time.time() - ctx.state["churn_launched_at"] > 120, 130)
    first = ctx.converse("watch", project, ctx.host.command("watch", "../churn-campaign"), lambda text, index: None, max_turns=1)
    second = ctx.converse("watch-2", project, "how is the kapso campaign in ../churn-campaign going?", lambda text, index: None, max_turns=1)
    turns = first + second
    answer = "\n".join(turn.text for turn in turns).lower()
    status = status_of(Path(campaign)) or {}
    state, phase = (status.get("state") or "").lower(), (status.get("phase") or "").lower()
    # The answer must name the real state (or phase), so a host that
    # reported a dead campaign as running would fail here.
    reflects = bool(state) and (state in answer or (phase and phase in answer))
    completed = ctx.user.ended_well("watch", all(not turn.asked for turn in turns) and reflects)
    return record_step("watch", turns, completed, [ctx.artifact("watch-answers.txt", "\n\n---\n\n".join(turn.text for turn in turns))], [f"the status at the read: state={state or '-'} phase={phase or '-'}; the answer reflected it: {reflects}"])


CASE_NAME = re.compile(r"^(?P<skill>[a-z]+)-(?P<kind>[eicn])\d+$|^(?P<holdout>holdout)-\d+$")
TRIGGER_BAR = SUITE["triggers"]["bar"]   # eighteen of twenty per skill


def trigger_bar(out_dir: Path, host: str) -> Dict[str, Any]:
    """The suite's verdict per skill from its result file: the share of
    the skill's cases that passed every run, and the near misses that
    invoked the skill under test (a cross-skill false positive)."""
    passed: Dict[str, List[bool]] = {}
    false_positives: List[str] = []
    if host == "claude_code":
        document = json.loads((out_dir / "triggers.json").read_text(encoding="utf-8"))
        for case in document["cases"]:
            match = CASE_NAME.match(case["name"])
            if match is None:
                continue
            passed.setdefault(match.group("skill") or "holdout", []).append(case["aggregates"]["score"] >= 1.0)
            for run in case["arms"]["with"]:
                if any(g["name"].startswith("not-") and not g["passed"] for g in run.get("graders", [])):
                    false_positives.append(case["name"])
    else:
        document = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
        by_case: Dict[str, List[Dict[str, Any]]] = {}
        for result in document["results"]:
            by_case.setdefault(result["case"], []).append(result)
        for name, runs in by_case.items():
            match = CASE_NAME.match(name)
            if match is None:
                continue
            passed.setdefault(match.group("skill") or "holdout", []).append(all(run["passed"] for run in runs))
            if any(g["name"].startswith("not-") and not g["passed"] for run in runs for g in run["graders"]):
                false_positives.append(name)
    per_skill = {skill: f"{sum(ok)}/{len(ok)}" for skill, ok in passed.items()}
    # A Codex run whose sandbox never started is the machine's failure
    # (codex_triggers.sandbox_failure): the bar is not established on it.
    environment = sum(1 for result in document.get("results", []) if result.get("environment")) if host != "claude_code" else 0
    met = bool(passed) and all(sum(ok) >= TRIGGER_BAR * len(ok) for ok in passed.values()) and not false_positives and environment == 0
    return {"per_skill": per_skill, "false_positives": sorted(set(false_positives)), "environment": environment, "met": met}


def trigger_verdict(ctx: Context) -> Dict[str, Any]:
    """The suite's verdict from the step's artifacts: both runs read
    against the bar; a run without a result file fails."""
    notes, verdicts = [], []
    for label in ("main", "holdout"):
        out_dir = ctx.out / "artifacts" / f"triggers-{label}"
        result_file = out_dir / ("triggers.json" if ctx.host.name == "claude_code" else "results.json")
        if result_file.is_file():
            verdict = trigger_bar(out_dir, ctx.host.name)
            verdicts.append(verdict["met"])
            notes.append(f"{label}: {verdict['per_skill']}; cross-skill false positives: {verdict['false_positives'] or 'none'}; runs whose sandbox never started (not established): {verdict['environment']}")
            # The probe's own decision: a machine whose bubblewrap fails the
            # probes runs the suite under the fallback sandbox, said here.
            document = json.loads(result_file.read_text(encoding="utf-8")) if ctx.host.name == "codex" else {}
            if document.get("sandbox") and document.get("sandbox") != document.get("sandbox_requested"):
                notes.append(f"{label}: the Codex sessions ran under --sandbox {document['sandbox']}; the suite's {document['sandbox_requested']} failed the probe on this machine (the probe's decision, in sandbox.json beside the results)")
        else:
            verdicts.append(False)
            notes.append(f"{label}: the suite wrote no result file")
    notes.append("the bar: nine of ten of each skill's cases passing every run, no near miss invoking the skill under test; the holdout is read the same way")
    return {"completed": bool(verdicts) and all(verdicts), "notes": notes}


def step_triggers(ctx: Context) -> Dict[str, Any]:
    started = time.monotonic()
    which = "claude-triggers" if ctx.host.name == "claude_code" else "codex-triggers"
    # Codex's sessions run under the suite's sandbox unless the runner says
    # the machine needs another (GCP's image cannot start bwrap); the step
    # notes it, so a verdict on an unsandboxed session is read as such.
    sandbox = f" --sandbox {ctx.triggers_sandbox}" if ctx.host.name == "codex" and ctx.triggers_sandbox else ""
    # The two suites share the step's own budget: the main cases take most
    # of it, the ten held-out cases a quarter. The Codex probe keeps that
    # clock itself (a run not started by the deadline is recorded as not
    # reached, and the result is written all the same); the shell's own
    # timeout stands above it as the last resort. The release gate's S2 on
    # Codex ran the probe one session at a time past the step's two hours,
    # and the shell timeout alone ended the driver with no result at all.
    budget = step_seconds("triggers")
    margin = int(SUITE["triggers"]["deadline_margin_seconds"])
    main_seconds, holdout_seconds = budget - margin, budget // 4 - margin
    deadline = (lambda seconds: f" --deadline-seconds {seconds}") if ctx.host.name == "codex" else (lambda seconds: "")
    main = ctx.shell(f"bash scripts/redesign/plugin_round.sh {which} --tag main{sandbox}{deadline(main_seconds)}", cwd=ctx.clone, timeout=budget, check=False)
    holdout = ctx.shell(f"bash scripts/redesign/plugin_round.sh {which} --tag holdout --runs 1{sandbox}{deadline(holdout_seconds)}", cwd=ctx.clone, timeout=budget // 4, check=False)
    artifacts = []
    for label, run in (("main", main), ("holdout", holdout)):
        out_dir = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else ""
        if out_dir and Path(out_dir).is_dir():
            artifacts.append(ctx.copy_artifact(Path(out_dir), f"triggers-{label}"))
        artifacts.append(ctx.artifact(f"triggers-{label}.log", run.stdout + run.stderr))
    verdict = trigger_verdict(ctx)
    notes = [*verdict["notes"], *([f"the Codex sessions ran under --sandbox {ctx.triggers_sandbox}, the runner's word for this machine"] if sandbox else [])]
    return shell_step("triggers", verdict["completed"], artifacts, (time.monotonic() - started) / 60, notes)


def step_harness_export(ctx: Context) -> Dict[str, Any]:
    started = time.monotonic()
    harness = ctx.clone / "examples" / "harnesses" / "trading"
    artifacts = []
    for host in ("claude_code", "codex"):
        target = ctx.work / f"export-{host}"
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        run = ctx.shell(f"kapso harness export {harness} --host {host} --to {target}", check=False)
        artifacts.append(ctx.artifact(f"export-{host}.txt", run.stdout + run.stderr))
        tree = sorted(str(path.relative_to(target)) for path in target.rglob("*") if path.is_file())
        artifacts.append(ctx.artifact(f"export-{host}-files.txt", "\n".join(tree) + "\n"))
    refused = ctx.shell(f"kapso harness export {harness} --host claude_code --to {ctx.work / 'export-claude_code'}", check=False)
    artifacts.append(ctx.artifact("export-refusal.txt", refused.stdout + refused.stderr))
    # The exported skill's session runs on this host, in its own export:
    # Claude Code loads the skill through its Skill tool; Codex loads one by
    # reading its file (the plugin's trigger runner reads the same event),
    # so the host's commands are read for a SKILL.md read that succeeded.
    session_dir = ctx.work / f"export-{ctx.host.name}"
    turns = ctx.converse("harness-export", session_dir, "Using the backtesting skill in this project, tell me in three lines what timestamp discipline it asks for. Load the skill rather than guessing.", lambda text, index: None, max_turns=1)
    if ctx.host.name == "claude_code":
        skill_used = any(tool["name"] == "Skill" and "backtesting" in json.dumps(tool["input"]) for turn in turns for tool in turn.tools)
    else:
        skill_used = any("backtesting/SKILL.md" in json.dumps(tool["input"]) and tool.get("is_error") is False for turn in turns for tool in turn.tools)
    completed = refused.returncode != 0 and (ctx.work / "export-codex" / ".codex" / "agents" / "lookahead-auditor.toml").is_file() and skill_used
    step = record_step("harness-export", turns, completed, artifacts, [f"the export and its refusal on both hosts; the exported skill's session on {ctx.host.name} loaded the skill: {skill_used}"])
    step["minutes"] = round(step["minutes"] + (time.monotonic() - started) / 60, 1)
    return step


CAMPAIGN_TOOLS = ("kapso_build_start", "kapso_status", "kapso_stop")


def mcp_server_probe(project: Path, seconds: float) -> Dict[str, Any]:
    """A standalone handshake with the plugin's MCP server as a host would
    make it: `kapso-mcp` started from the project (a login shell, the PATH
    a tool install uses), `initialize`, the initialized notification, then
    `tools/list`, over the stdio transport's newline-delimited JSON-RPC.
    What it returns is the one fact the session's own account cannot give:
    whether the server starts on this machine, in how many seconds, and
    which tools it lists (the eighth S2 Codex launch's campaign-tools
    session reported the three tools unavailable, the session-brief one
    minutes later read a card through the same server, and Codex leaves no
    line for a server it drops at startup)."""
    started = time.monotonic()
    child = subprocess.Popen(["bash", "-lc", "exec kapso-mcp"], cwd=str(project), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "kapso-gate-probe", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    assert child.stdin is not None and child.stdout is not None
    for message in messages:
        child.stdin.write(json.dumps(message) + "\n")
    child.stdin.flush()
    tools: List[str] = []
    error = ""
    listed = False
    deadline = started + seconds
    while time.monotonic() < deadline and not listed:
        if select.select([child.stdout], [], [], 0.5)[0] == []:
            if child.poll() is not None:
                error = f"the server exited with {child.returncode}"
                break
            continue
        line = child.stdout.readline()
        if not line:
            error = "the server closed its output"
            break
        if not line.startswith("{"):
            continue
        reply = json.loads(line)
        if reply.get("id") == 2:
            tools = [tool["name"] for tool in (reply.get("result") or {}).get("tools", [])]
            listed = True
        elif reply.get("id") == 1 and "error" in reply:
            error = json.dumps(reply["error"])
            break
    elapsed = time.monotonic() - started
    if not listed and not error:
        error = f"no tools/list reply within {seconds:.0f} s"
    child.kill()
    stderr = child.stderr.read() if child.stderr is not None else ""
    return {"started": listed, "seconds": round(elapsed, 1), "tools": tools, "error": error, "stderr_tail": stderr[-600:]}


def step_campaign_tools(ctx: Context) -> Dict[str, Any]:
    project = Path(ctx.state["churn_project"])
    prompt = (
        "Use the `kapso` MCP server's kapso_build_start tool (not the shell) to start a campaign of this repository: brief kapso-brief.md, repo ., "
        "output ../tools-campaign, evaluation evaluation, data data, budget 10 minutes and 1 experiment. Then read it with the server's "
        "kapso_status, then end it with its kapso_stop, and report the three answers verbatim."
    )
    # The server probed as a host would start it, before the session: the
    # fact the session's own account cannot give.
    probe = mcp_server_probe(project, float(SUITE["waits"]["mcp_probe_seconds"]))
    artifacts = [ctx.artifact("campaign-tools-server-probe.json", json.dumps(probe, indent=1))]
    serves = all(any(name.endswith(tool) for name in probe["tools"]) for tool in CAMPAIGN_TOOLS)
    notes = [f"the server, started as a host starts it: {'listed ' + str(len(probe['tools'])) + ' tools in ' + str(probe['seconds']) + ' s' if probe['started'] else 'did not answer (' + probe['error'] + ')'}"
             + ("" if serves or not probe["started"] else "; the three campaign tools are not among them")]
    turns = ctx.converse("campaign-tools", project, prompt, lambda text, index: None, max_turns=1)
    names = [tool["name"] for turn in turns for tool in turn.tools]
    used = {tool for tool in CAMPAIGN_TOOLS if any(name.endswith(tool) for name in names)}
    if not used and serves:
        # The server lists the tools and the session saw none: a server
        # Codex dropped at that session's start, which a fresh session
        # starts again. Said here, and in both answers kept.
        notes.append("the first session used no kapso tool though the server lists them; a second session ran")
        turns = [*turns, *ctx.converse("campaign-tools-2", project, prompt, lambda text, index: None, max_turns=1)]
        names = [tool["name"] for turn in turns for tool in turn.tools]
        used = {tool for tool in CAMPAIGN_TOOLS if any(name.endswith(tool) for name in names)}
    artifacts.append(ctx.artifact("campaign-tools.txt", "\n".join(names) + "\n\n" + "\n\n---\n\n".join(turn.text for turn in turns)))
    receipt = ctx.work / "tools-campaign.launch.json"
    if receipt.is_file():
        artifacts.append(ctx.copy_artifact(receipt, "tools-campaign.launch.json"))
    ctx.shell(f"kapso build --output {ctx.work / 'tools-campaign'} --stop", check=False)
    return record_step("campaign-tools", turns, used == set(CAMPAIGN_TOOLS), artifacts, notes)


def step_inbox(ctx: Context) -> Dict[str, Any]:
    project = ctx.work / "churn-pause"
    campaign = Path(ctx.state["pause_campaign"])
    paused = wait_for(lambda: (campaign / ".kapso" / "inbox.jsonl").is_file(), ctx.pause_minutes * 60 + 300, every=15)
    notes = [f"the injected pause arrived: {paused}"]
    turns = ctx.converse("inbox", project, ctx.host.command("inbox", "../churn-pause-campaign"), lambda text, index: None, max_turns=1)
    ctx.user.interventions.append("inbox: the key written where the request's fix line says before the person says so")
    # The person does what the request's fix line says before saying so:
    # the key into the campaign's .env (and the project's), never the reply.
    for env in (campaign / ".env", project / ".env"):
        with open(env, "a", encoding="utf-8") as out:
            out.write("DATA_API_KEY=proof-placeholder-key\n")
    turns += ctx.converse(
        "inbox-reply", project,
        "DATA_API_KEY is in .env now, in the campaign's and the project's. There is no enrichment service, endpoint "
        "or document to give: reply to every open request of the campaign — the key is in place, there is no service, "
        "and it must reach 0.85 without the enrichment table — so that nothing is left waiting on me.",
        lambda text, index: None, max_turns=1,
    )
    inbox = campaign / ".kapso" / "inbox.jsonl"
    replied = inbox.is_file() and any('"replied"' in line or '"reply"' in line for line in inbox.read_text(encoding="utf-8").splitlines())
    # The requirement is kept: the suspended work must actually resume (the
    # orchestrator writes a `continued` event), not just be answered.
    continued = wait_for(lambda: inbox.is_file() and '"continued"' in inbox.read_text(encoding="utf-8"), 600, every=15)
    artifacts = [ctx.artifact("inbox-answers.txt", "\n\n---\n\n".join(turn.text for turn in turns))]
    if inbox.is_file():
        artifacts.append(ctx.copy_artifact(inbox, "inbox.jsonl"))
    ctx.shell(f"kapso build --output {campaign} --stop", check=False)
    notes.append(f"the request was replied: {replied}; the campaign resumed (a continued event): {continued}")
    return record_step("inbox", turns, ctx.user.ended_well("inbox", paused and replied and continued), artifacts, notes)


def step_build_fraud(ctx: Context) -> Dict[str, Any]:
    project = ctx.fixture("fraud-no-evaluator", "fraud")
    campaign = ctx.work / "fraud-campaign"
    result = _launch_and_wait_status(ctx, project, campaign, FRAUD_WISH, "build-fraud")
    evaluators = [str(path.relative_to(project)) for path in project.rglob("evaluate.py") if "campaign" not in path.parts]
    artifacts = result["artifacts"] + [ctx.artifact("fraud-evaluators.txt", "\n".join(evaluators) + "\n")]
    ctx.shell(f"kapso build --output {campaign} --stop", check=False)
    contract = evaluator_contract(ctx, project, evaluators[0]) if evaluators else "no evaluator written"
    artifacts.append(ctx.artifact("fraud-contract.txt", contract + "\n"))
    return record_step("build-fraud", result["turns"], result["brief"] and result["launched"] and contract == "ok", artifacts,
                       [f"evaluator files written by the session: {evaluators}", f"the evaluator's contract: {contract}"])


def evaluator_contract(ctx: Context, project: Path, evaluator: str) -> str:
    """The contract every fixture meets, on the evaluator a session wrote:
    one command from a workspace root, the metric on its own line last as
    `name: value`, result.json beside the script with the engine's keys.
    Returns "ok" or the first breach."""
    workspace = ctx.work / f"{project.name}-contract"
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(project / "repo", workspace, ignore=shutil.ignore_patterns("__pycache__"))
    if (project / "data").is_dir():
        shutil.copytree(project / "data", workspace / "kapso_datasets")
    source = (project / evaluator).parent
    shutil.copytree(source, workspace / "kapso_evaluation", ignore=shutil.ignore_patterns("result.json", "__pycache__"))
    run = ctx.shell(f"python3 kapso_evaluation/{Path(evaluator).name}", cwd=workspace, check=False, timeout=900)
    if run.returncode != 0:
        return f"exit {run.returncode}: {(run.stdout + run.stderr).strip().splitlines()[-1:] or ['no output']}"
    last = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else ""
    if not re.match(r"^[a-z_]+: -?[0-9.]+$", last):
        return f"the last line is not `name: value`: {last!r}"
    result = workspace / "kapso_evaluation" / "result.json"
    if not result.is_file():
        return "no result.json beside the script"
    payload = json.loads(result.read_text(encoding="utf-8"))
    keys = set(payload)
    if not {"evaluation_script_path", "evaluation_output", "score"} <= keys:
        return f"result.json lacks {sorted({'evaluation_script_path', 'evaluation_output', 'score'} - keys)}"
    score = payload.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)):
        return f"the score is not a finite number: {score!r}"
    # A malformed submission must be REJECTED (a scorer that accepts anything
    # is not a protected evaluator): a predict.py that writes the wrong
    # columns, re-run, must exit non-zero with a printed reason.
    predict = workspace / "predict.py"
    if predict.is_file():
        predict.write_text("import argparse, csv, sys\np = argparse.ArgumentParser(); p.add_argument('--data'); p.add_argument('--out'); a = p.parse_args()\nw = csv.writer(open(a.out, 'w')); w.writerow(['wrong', 'columns']); w.writerow(['x', 'y'])\n", encoding="utf-8")
        (workspace / "predictions.csv").unlink(missing_ok=True)
        malformed = ctx.shell(f"python3 kapso_evaluation/{Path(evaluator).name}", cwd=workspace, check=False, timeout=900)
        if malformed.returncode == 0:
            return "a malformed submission (wrong prediction columns) was accepted, not rejected"
    return "ok"


def _criteria_step(ctx: Context, name: str, project: Path, wish: str) -> Dict[str, Any]:
    # Any campaign that appears under the work tree during the step is a
    # launch — the one behaviour "do not launch anything" must catch.
    before = {str(path) for path in ctx.work.rglob(".kapso/status.json")}
    turns = ctx.converse(name, project, ctx.host.command("build", wish), stop_here_policy, max_turns=3)
    launched = bool({str(path) for path in ctx.work.rglob(".kapso/status.json")} - before)
    proposed = turns and turns[0].asked
    artifacts = [ctx.artifact(f"{name}-answers.txt", "\n\n---\n\n".join(turn.text for turn in turns))]
    return record_step(name, turns, ctx.user.ended_well(name, bool(proposed) and not launched), artifacts, [f"candidate criteria judged by hand from the answers artifact; a campaign was launched: {launched} (the step forbids it)"])


def step_build_data_only(ctx: Context) -> Dict[str, Any]:
    project = ctx.fixture("fraud-no-evaluator", "fraud-data")
    (project / "kapso-brief.md").unlink()
    readme = project / "README.md"
    readme.write_text("# Fraud scoring\n\nLabelled transactions in data/transactions.csv and a rule-based scorer in repo/score.py.\n", encoding="utf-8")
    return _criteria_step(ctx, "build-data-only", project, "make the fraud scorer in this repository better")


def step_build_empty_wish(ctx: Context) -> Dict[str, Any]:
    project = ctx.work / "empty"
    if project.exists():
        shutil.rmtree(project)
    project.mkdir(parents=True)
    (project / "README.md").write_text("# churn-work\n\nOur churn modelling work will live here.\n", encoding="utf-8")
    ctx.seed_env(project)
    return _criteria_step(ctx, "build-empty-wish", project, "build something useful here")


def step_harness_new(ctx: Context) -> Dict[str, Any]:
    project = ctx.fixture("trading", "trading-author")
    brief = project / "kapso-brief.md"
    brief.write_text("".join(line for line in brief.read_text(encoding="utf-8").splitlines(keepends=True) if not line.startswith("Harness")), encoding="utf-8")
    ctx.state["trading_project"] = str(project)
    ctx.save()
    if ctx.user.kind == "scripted":
        ctx.user.interventions.append("harness-new: the harness's module answers from the driver's policy")
    turns = ctx.converse("harness-new", project, ctx.host.command("harness", "new"), harness_policy, max_turns=12)
    # The session writes where the repository points (`../harnesses/trading`
    # in the README) or under the project: both are the person's.
    manifests = [
        path for path in (*project.rglob("harness.yaml"), *(ctx.work / "harnesses").rglob("harness.yaml"))
        if "campaign" not in path.parts
    ]
    artifacts = [ctx.artifact("harness-new-answers.txt", "\n\n---\n\n".join(turn.text for turn in turns))]
    green = False
    if manifests:
        harness = manifests[0].parent
        ctx.state["new_harness"] = str(harness)
        ctx.save()
        check = ctx.shell(f"kapso harness check {harness}", cwd=project, check=False)
        artifacts.append(ctx.artifact("harness-new-check.txt", check.stdout + check.stderr))
        green = check.returncode == 0 and "warn" not in check.stdout
        tree = sorted(str(path.relative_to(harness)) for path in harness.rglob("*") if path.is_file())
        shipped = ctx.clone / "examples" / "harnesses" / "trading"
        shipped_tree = sorted(str(path.relative_to(shipped)) for path in shipped.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
        artifacts.append(ctx.artifact("harness-new-tree.txt", "written:\n" + "\n".join(tree) + "\n\nshipped:\n" + "\n".join(shipped_tree) + "\n"))
    return record_step("harness-new", turns, bool(manifests) and green, artifacts, ["compared with the shipped harness by hand from harness-new-tree.txt"])


def step_harness_probe(ctx: Context) -> Dict[str, Any]:
    started = time.monotonic()
    project = Path(ctx.state["trading_project"])
    harness = ctx.state.get("new_harness")
    if not harness:
        return shell_step("harness-probe", False, [], 0, ["no harness was written"])
    campaign = ctx.work / "trading-probe"
    # A previous probe is stopped and cleared: the launch and the wait
    # below are the new campaign's, never a finished one's status.
    if campaign.exists():
        ctx.shell(f"kapso build --output {campaign} --stop", check=False)
        shutil.rmtree(campaign)
    ctx.state["probe_campaign"] = str(campaign)
    ctx.save()
    # The interview's session may have run the evaluator while testing the
    # harness and left its result beside it, which a launch refuses
    # ("remove it and launch again"); the person does.
    leftover = project / "evaluation" / "result.json"
    if leftover.is_file():
        leftover.unlink()
    run = ctx.shell(
        f"kapso build --harness {harness} --brief kapso-brief.md --repo repo --evaluation evaluation --data data "
        f"--output {campaign} --budget-minutes {_PROBE_B['minutes']} --budget-experiments {_PROBE_B['experiments']} --detach", cwd=project, check=False,
    )
    done = wait_for(lambda: (status_of(campaign) or {}).get("state") in ("done", "failed"), step_seconds("harness-probe"), every=20)
    artifacts = [ctx.artifact("harness-probe-launch.txt", run.stdout + run.stderr)]
    if status_of(campaign) is not None:
        artifacts.append(ctx.copy_artifact(campaign / ".kapso" / "status.json", "harness-probe-status.json"))
    return shell_step("harness-probe", run.returncode == 0 and done, artifacts, (time.monotonic() - started) / 60)


def step_review(ctx: Context) -> Dict[str, Any]:
    project, campaign = Path(ctx.state["churn_project"]), Path(ctx.state["churn_campaign"])
    finished = wait_for(lambda: (status_of(campaign) or {}).get("state") in ("done", "failed"), _CHURN_B["minutes"] * 60 + CAMPAIGN_GRACE_SECONDS, every=20)
    turns = ctx.converse("review", project, ctx.host.command("review", "../churn-campaign"), lambda text, index: "is it safe to merge?" if index == 0 else None, max_turns=3)   # WP7's round: the review's two-turn headroom was thin for a persona
    # The merge question is the step's second want; a persona that spent
    # its turn elsewhere (s2-claude-2 asked about the planted baseline and
    # was done) leaves it to be asked once, as the scripted user would.
    if len(turns) == 1 or not merge_verdict("\n".join(turn.text for turn in turns)):
        turns += ctx.converse("review-2", project, "is the branch the campaign delivered in ../churn-campaign safe to merge?", lambda text, index: None, max_turns=1)
    artifacts = [ctx.artifact("review-answers.txt", "\n\n---\n\n".join(turn.text for turn in turns))]
    solution = campaign / ".kapso" / "solution.json"
    branch = None
    if solution.is_file():
        artifacts.append(ctx.copy_artifact(solution, "churn-solution.json"))
        # The solution record's own key (handles.py writes `branch` at the top;
        # a run on the Claude host read an older shape's `delivered.branch`
        # and named nothing). Null when no node scored.
        branch = json.loads(solution.read_text(encoding="utf-8")).get("branch")
    answer = "\n".join(turn.text for turn in turns).lower()
    # The answer must name what was delivered (the branch, or that nothing
    # was) and give a merge verdict, so "I can't read that campaign" does
    # not pass review.
    named = names_delivery(answer, branch)
    verdict = merge_verdict(answer)
    completed = ctx.user.ended_well("review", finished and solution.is_file() and named and verdict)
    return record_step("review", turns, completed, artifacts, [f"the churn campaign ended: {(status_of(campaign) or {}).get('state')}; the answer named the delivered branch {branch!r}: {named}; a merge verdict: {verdict}"])


def step_deploy(ctx: Context) -> Dict[str, Any]:
    project, campaign = Path(ctx.state["churn_project"]), Path(ctx.state["churn_campaign"])
    turns = ctx.converse("deploy", project, ctx.host.command("deploy", "../churn-campaign --target local"), lambda text, index: "call it once with the inputs the proof used and show me the answer" if index == 0 else None, max_turns=2)
    if len(turns) == 1:
        turns += ctx.converse("deploy-2", project, "call the deployment once with the inputs its proof used and show me the answer", lambda text, index: None, max_turns=1)
    deployment = campaign / ".kapso" / "deployments" / "1"
    artifacts = [ctx.artifact("deploy-answers.txt", "\n\n---\n\n".join(turn.text for turn in turns))]
    if (deployment / "deployment.json").is_file():
        artifacts.append(ctx.copy_artifact(deployment / "deployment.json", "deploy-deployment.json"))
    parity = deploy_parity(deployment, campaign)
    ctx.shell(f"kapso deploy {campaign} --stop --number 1", cwd=project, check=False)
    # Parity must be "same": a "differs (...)" string or a "not established"
    # is not a proven deployment (the old `!= "differs"` let both pass).
    return record_step("deploy", turns, (deployment / "deployment.json").is_file() and parity == "same", artifacts, [f"the deployed call against the delivered predictions: {parity}"])


def deploy_parity(deployment: Path, campaign: Path) -> str:
    """Deployment parity: the deployed call's observed answer for its proof
    input against the delivered branch's own prediction for that row.
    "same", "differs", or why it is not established."""
    record = deployment / "deployment.json"
    if not record.is_file():
        return "not established: no deployment record"
    observed = json.loads(record.read_text(encoding="utf-8")).get("observed") or {}
    predictions = campaign / "predictions.csv"
    if "id" not in observed or not predictions.is_file():
        return "not established: no observed row or no predictions.csv on the delivered branch"
    for line in predictions.read_text(encoding="utf-8").splitlines()[1:]:
        row_id, _, value = line.partition(",")
        if row_id == str(observed["id"]):
            delivered = float(value)
            return "same" if abs(delivered - float(observed["churn_probability"])) <= 1e-6 else f"differs (deployed {observed['churn_probability']}, delivered {delivered})"
    return "not established: the proof's id is not in predictions.csv"


def step_learn(ctx: Context) -> Dict[str, Any]:
    project = Path(ctx.state["churn_project"])
    config = effective_config(ctx, project)
    notes = []
    if ctx.learn_model:
        # The round's accommodation: the crews on another model when the
        # token's window for the shipped one is spent; said in the record.
        project_config(project, {"learning": learning_on(config["learning"], ctx.learn_model, "claude-fable-5")})
        notes.append(f"the learning crews ran on {ctx.learn_model} for this round (--learn-model): the token's window for the shipped model was spent")
    status_dir = Path(config["learning"]["status_dir"]).expanduser()
    if not status_dir.is_absolute():
        status_dir = project / status_dir
    before = set(glob.glob(str(status_dir / "*.json"))) if status_dir.is_dir() else set()
    turns = ctx.converse("learn", project, ctx.host.command("learn", "../churn-campaign"), lambda text, index: None, max_turns=1)
    after = lambda: set(glob.glob(str(status_dir / "*.json"))) - before
    launched = wait_for(lambda: bool(after()), 180)
    ctx.state["learn_status"] = sorted(after())[-1] if after() else None
    ctx.save()
    artifacts = [ctx.artifact("learn-answers.txt", "\n".join(turn.text for turn in turns))]
    offered = any("session_sources" in turn.text for turn in turns)
    return record_step("learn", turns, launched, artifacts, [f"the serving offer (session_sources) was made: {offered}", *notes])


def step_learn_complete(ctx: Context) -> Dict[str, Any]:
    started = time.monotonic()
    status = ctx.state.get("learn_status")
    if not status:
        return shell_step("learn-complete", False, [], 0, ["no learn status appeared"])
    done = wait_for(lambda: json.loads(Path(status).read_text()).get("state") in ("done", "failed"), step_seconds("learn-complete"), every=60)
    final = json.loads(Path(status).read_text())
    listing = ctx.shell("kapso experience list", cwd=Path(ctx.state["churn_project"]), check=False)
    artifacts = [ctx.artifact("learn-status.json", json.dumps(final, indent=1)), ctx.artifact("experience-list.txt", listing.stdout + listing.stderr)]
    cards = re.search(r"(\d+) cards", listing.stdout)
    # The learn ran end to end: every phase, and a bank commit (the head
    # advanced, whether or not the exam admitted a card). Whether a card is
    # banked is a learning-quality outcome that varies with the campaign and
    # the crew's model — the owner's call (2026-10-31): the step passes on a
    # clean end-to-end learn, the cards banked a recorded fact, not a gate.
    ran: set = set()
    for line in final.get("recent") or []:
        ran |= {phase for phase in ("harvest", "harvested", "mine", "mined", "exam", "lesson", "push", "pushed") if phase in line}
    advanced = bool(final.get("bank_head_after")) and final.get("bank_head_after") != final.get("bank_head_before")
    banked = int(cards.group(1)) if cards else None
    created = len((final.get("cards") or {}).get("created") or [])
    updated = len((final.get("cards") or {}).get("updated") or [])
    note = f"state {final.get('state')}, bank {str(final.get('bank_head_before'))[:8]} -> {str(final.get('bank_head_after'))[:8]}; lesson landed {created} created, {updated} updated; the bank lists {banked if banked is not None else '?'} cards (a quality outcome, not a bar)"
    return shell_step("learn-complete", done and final.get("state") == "done" and {"harvested", "mined", "lesson"} <= ran and advanced, artifacts, (time.monotonic() - started) / 60, [note])


def step_session_brief(ctx: Context) -> Dict[str, Any]:
    project = Path(ctx.state["churn_project"])
    project_config(project, {"experience": {"session_sources": ["mine:all"]}})
    prompt = "What experience is this session served, according to your MCP server's instructions? Then read one card from the bank with the bank tools and quote its title."
    turns = ctx.converse("session-brief", project, prompt, lambda text, index: None, max_turns=1)
    bank_tools = [tool for turn in turns for tool in turn.tools if "bank" in tool["name"]]
    # A successful bank read: a bank tool's result came back and was not
    # an error (both hosts' turns carry `is_error` from the result; a call
    # with no result, or a failed one, is not a card read).
    bank_read = any(tool.get("is_error") is False for tool in bank_tools)
    names = [tool["name"] for turn in turns for tool in turn.tools]
    artifacts = [ctx.artifact("session-brief.txt", "\n".join(names) + "\n\n" + "\n".join(turn.text for turn in turns))]
    return record_step("session-brief", turns, ctx.user.ended_well("session-brief", bank_read), artifacts, [f"a bank tool succeeded (a card read): {bank_read}; the sources recognised judged by hand from the answer"])


def step_experience_share(ctx: Context) -> Dict[str, Any]:
    project = Path(ctx.state["churn_project"])
    ensure_second_user(ctx)
    # The bare remote where the second user can read it: the first user's
    # home is its own.
    remote = Path("/tmp") / "bank-remote.git"
    # A bare remote whose HEAD is `main`, as a git host's would be: the
    # bank pushes `main`, and a clone's HEAD must land on it.
    ctx.shell(f"rm -rf {remote} && git init -q --bare --initial-branch=main {remote} && chmod -R a+rX {remote}")
    turns = ctx.converse("experience-share", project, ctx.host.command("experience", f"share file://{remote}"), go_policy, max_turns=3)
    # The second user, from its own home, loads the shared bank by its
    # reference (a bank needs no `add`: loading fetches it into the store)
    # through the session server, and lists what it reads.
    # A local bare remote owned by another user is "dubious" to that
    # user's git; a team shares over a git host, so the second user adds
    # the one exception the simulation needs.
    second = ctx.shell(
        f"sudo -u proof2 -H bash -lc 'cd ~; export PATH=$HOME/.local/bin:$PATH; git config --global --add safe.directory {remote}; "
        f"timeout 120 kapso-mcp --experience file://{remote} --root ~ < /dev/null 2>&1 | head -20; kapso experience list'",
        check=False, timeout=300,
    )
    artifacts = [ctx.artifact("experience-share.txt", "\n".join(turn.text for turn in turns)), ctx.artifact("experience-second-user.txt", second.stdout + second.stderr)]
    shared = ctx.shell(f"git -C {remote} log --oneline -1", check=False).returncode == 0
    loaded = "banks you read" in second.stdout and "serving" in second.stdout
    return record_step("experience-share", turns, shared and loaded, artifacts, ["the second user loads the shared bank by reference through kapso-mcp and lists it under the banks it reads; a served build at a tiny budget is left to WP9's round"])


def step_second_engineer_build(ctx: Context) -> Dict[str, Any]:
    """The design's ninth step: the second engineer, on its own install,
    builds at a tiny budget served the shared bank by its reference, and
    the record names the bank among what the build was served."""
    started = time.monotonic()
    project = Path(ctx.state["churn_project"])
    budget = SUITE["builds"]["second_engineer"]
    ensure_second_user(ctx)
    ctx.shell(f"sudo rm -rf /home/proof2/churn-second /home/proof2/second-campaign /home/proof2/second-campaign.log /home/proof2/env; sudo cp -r {project} /home/proof2/churn-second; sudo chown -R proof2:proof2 /home/proof2/churn-second")
    # The copied project's evaluator reads the labels through the absolute
    # path in evaluation/private_dir.txt, which the first user's view wrote
    # into its own home — unreadable to the second user (rel-s2-claude: the
    # evaluator died at line 53 on every attempt, `drwxr-x--- ubuntu`, and
    # the build ended unscored). The second engineer's copy points at its own.
    ctx.shell("sudo -u proof2 -H bash -lc 'printf %s /home/proof2/churn-second/private > ~/churn-second/evaluation/private_dir.txt'")
    # The second engineer's project environment, as the docs install it: a
    # venv holding the wheel and the project's requirements, first on PATH,
    # so `python3` — the interpreter the harness's commands name — is it
    # (the uv tool install above keeps the tool's interpreter off PATH, and
    # the system's python3 refuses pip as externally managed). A launch the
    # harness refuses for its requirements prints the remedy, `python3 -m
    # pip install -r <the harness's requirements>`; the engineer follows
    # it as printed and launches again (the bundled harness's requirements
    # are not the package's, by the lean-install decision of 2026-08-26).
    wheel = stage_wheel_for_second_user(ctx)
    ctx.shell(f"sudo -u proof2 -H bash -lc 'cd ~; python3 -m venv ~/env && ~/env/bin/pip install -q /tmp/{wheel.name} -r ~/churn-second/requirements.txt'", timeout=1200)
    build_line = (
        "cd ~/churn-second; export PATH=$HOME/env/bin:$PATH; "
        "kapso build --brief kapso-brief.md --evaluation evaluation --data data --repo repo --output ~/second-campaign "
        f"--experience file:///tmp/bank-remote.git --budget-minutes {budget['minutes']} --budget-experiments {budget['experiments']} --detach"
    )
    launch = ctx.shell(f"sudo -u proof2 -H bash -lc {shlex.quote(build_line)}", check=False, timeout=600)
    remedies = sorted(set(re.findall(r"python3 -m pip install -r (\S+)", launch.stdout + launch.stderr)))
    launches = [launch.stdout + launch.stderr]
    if launch.returncode != 0 and remedies:
        for requirements in remedies:
            ctx.shell(f"sudo -u proof2 -H bash -lc 'export PATH=$HOME/env/bin:$PATH; python3 -m pip install -q -r {requirements}'", timeout=1200)
        launch = ctx.shell(f"sudo -u proof2 -H bash -lc {shlex.quote(build_line)}", check=False, timeout=600)
        launches.append(f"--- after the printed remedy ({', '.join(remedies)}) ---\n" + launch.stdout + launch.stderr)
    campaign = Path("/home/proof2/second-campaign")
    read_status = lambda: json.loads(ctx.shell(f"sudo cat {campaign}/.kapso/status.json", check=False).stdout or "null")
    done = wait_for(lambda: (read_status() or {}).get("state") in ("done", "failed"), budget["minutes"] * 60 + 600, every=30)
    solution_text = ctx.shell(f"sudo cat {campaign}/.kapso/solution.json", check=False).stdout
    solution = json.loads(solution_text) if solution_text.strip() else {}
    served = "bank-remote" in solution_text
    final_state = (read_status() or {}).get("state")
    # Scored: the solution record's own `score` (handles.py writes it at
    # the top; a `delivered.score` key does not exist, and reading it
    # could never be true); and the build read a bank card (a record under
    # the campaign names the shared bank).
    scored = solution.get("score") is not None
    read_bank = "bank-remote" in ctx.shell(f"sudo cat {campaign}/.kapso/experiment_history.json 2>/dev/null; sudo cat {campaign}/.kapso/solution.json", check=False).stdout
    artifacts = [ctx.artifact("second-engineer-launch.txt", "\n".join(launches))]
    if solution_text.strip():
        artifacts.append(ctx.artifact("second-engineer-solution.json", solution_text))
    # The second user's campaign log and status, so a failure has an
    # inspectable cause in the archive (the preflight death of s2-claude-3
    # had none).
    artifacts.append(ctx.artifact("second-engineer-campaign.log", ctx.shell("sudo cat /home/proof2/second-campaign.log", check=False).stdout))
    artifacts.append(ctx.artifact("second-engineer-status.json", ctx.shell(f"sudo cat {campaign}/.kapso/status.json", check=False).stdout))
    ctx.user.interventions.append(
        "second-engineer-build: the churn fixture copied to the second user's home; a venv with the wheel and the project's requirements first on its PATH; "
        + (f"the launch refused for the harness's requirements and the printed remedy followed ({', '.join(remedies)}); " if remedies else "")
        + "the build run as that user with the bank by reference"
    )
    return shell_step("second-engineer-build", launch.returncode == 0 and done and final_state == "done" and served and scored, artifacts, (time.monotonic() - started) / 60,
                      [f"served the shared bank: {served}; a scored delivery: {scored}; the bank in the build's records: {read_bank}; final state: {final_state}"], kind="build")


def step_harness_improve(ctx: Context) -> Dict[str, Any]:
    project = Path(ctx.state["trading_project"])
    harness = ctx.state.get("new_harness")
    if not harness or not ctx.state.get("probe_campaign"):
        return record_step("harness-improve", [], False, [], ["no harness or no probe campaign"])
    manifest = Path(harness) / "harness.yaml"
    before = manifest.read_text(encoding="utf-8")
    turns = ctx.converse("harness-improve", project, ctx.host.command("harness", f"improve {ctx.state['probe_campaign']}"), lambda text, index: "go" if index == 0 else None, max_turns=3)
    check = ctx.shell(f"kapso harness check {harness}", cwd=project, check=False)
    artifacts = [ctx.artifact("harness-improve-answers.txt", "\n".join(turn.text for turn in turns)), ctx.artifact("harness-improve-check.txt", check.stdout + check.stderr)]
    bumped = manifest.read_text(encoding="utf-8") != before
    return record_step("harness-improve", turns, bumped and check.returncode == 0, artifacts)


def step_campaign_session_isolation(ctx: Context) -> Dict[str, Any]:
    """A campaign of its own at a tiny budget, run on the installed
    wheel with the plugin at user scope: its Claude Code sessions list
    no plugin tool or command, and its Codex sessions run (the isolation
    lines must not break them); a rules file is reported where one is
    found, not required (a session's temporary workspace is gone with it)."""
    started = time.monotonic()
    project = Path(ctx.state["churn_project"])
    campaign = ctx.work / "isolation-campaign"
    if campaign.exists():
        ctx.shell(f"kapso build --output {campaign} --stop", check=False)
        shutil.rmtree(campaign)
    launch = ctx.shell(
        f"kapso build --brief kapso-brief.md --repo repo --evaluation evaluation --data data --output {campaign} "
        f"--budget-minutes {_PROBE_B['minutes']} --budget-experiments {_PROBE_B['experiments']} --detach", cwd=project, check=False,
    )
    wait_for(lambda: (status_of(campaign) or {}).get("state") in ("done", "failed"), step_seconds("campaign-session-isolation"), every=20)
    findings = []
    # The implementation sessions' streams are `.kapso/sessions/<branch>/stream.jsonl`,
    # the ideation sessions' `.kapso/ideation/iter<n>/<host>_<model>.stream.txt`.
    for stream in sorted([*campaign.rglob("stream.jsonl"), *campaign.rglob("*.stream.txt")]):
        for line in stream.read_text(encoding="utf-8").splitlines():
            if not line.startswith("{"):
                continue
            event = json.loads(line)
            if event.get("type") == "system" and event.get("subtype") == "init":
                tools = event.get("tools", [])
                commands = event.get("slash_commands", [])
                findings.append({
                    "session": str(stream.relative_to(campaign)),
                    "plugin_tools": [tool for tool in tools if "plugin_kapso" in tool],
                    "plugin_commands": [command for command in commands if command.startswith("kapso:")],
                    "servers": [server.get("name") for server in event.get("mcp_servers", [])],
                })
                break
    # A Codex session in the campaign carries the execpolicy rule in its
    # workspace instead; the file's presence and the session's exit are
    # the evidence.
    codex_sessions = []
    for record in sorted(campaign.rglob("session.json")):
        data = json.loads(record.read_text(encoding="utf-8"))
        if data.get("host") == "codex":
            codex_sessions.append({"session": str(record.parent.relative_to(campaign)), "exits": [run.get("exit") for run in data.get("runs", [])],
                                   "errors": [run.get("error") for run in data.get("runs", []) if run.get("error")]})
    # A Codex session the machine's own quota refused never ran: said, not counted.
    quota = [s for s in codex_sessions if any("Quota exceeded" in error for error in s["errors"])]
    ran = [s for s in codex_sessions if s not in quota]
    codex_ok = all(all(exit == "success" for exit in s["exits"]) for s in ran)
    rules_files = [str(path.relative_to(campaign)) for path in campaign.rglob("kapso-campaign.rules")]
    report = {
        "launch": launch.stdout.strip().splitlines()[:3], "status": (status_of(campaign) or {}).get("state"),
        "claude_sessions": findings, "codex_sessions": codex_sessions, "codex_quota_refused": len(quota), "codex_rules_files": rules_files,
    }
    # A Codex engine session runs in a temporary workspace that is gone
    # with it, rules file included; the adapter's test pins the file, the
    # live evidence is the sessions running with the isolation lines.
    # On the Codex host a Codex session must have run: no session is no evidence.
    completed = bool(findings) and all(not f["plugin_tools"] and not f["plugin_commands"] for f in findings) and codex_ok and (bool(ran) or ctx.host.name != "codex")
    return shell_step("campaign-session-isolation", completed, [ctx.artifact("isolation.json", json.dumps(report, indent=1))], (time.monotonic() - started) / 60)


def step_codex_rules_probe(ctx: Context) -> Dict[str, Any]:
    """Codex's isolation, live: a session whose working directory is a
    gitignored subdirectory of a repository (an implementation session's
    shape), with the installed adapter's rules file and command-line
    lines, asked to run `kapso build --help`. The refusal is the evidence;
    the command's usage line would be the failure."""
    started = time.monotonic()
    root = ctx.work / "rules-probe"
    if root.exists():
        shutil.rmtree(root)
    session = root / "sessions" / "probe"
    session.mkdir(parents=True)
    ctx.shell(
        f"git init -q {root} && printf 'sessions/\\n' > {root}/.gitignore && git -C {root} add .gitignore "
        f"&& git -C {root} -c user.name=probe -c user.email=probe@example.com commit -qm seed"
    )
    # The rules file as the installed adapter writes it, from its own constants.
    write = (
        "from pathlib import Path; from kapso.execution.coding_agents.adapters.codex_agent import CAMPAIGN_RULES, CAMPAIGN_RULES_FILE; "
        f"target = Path({str(session)!r}) / CAMPAIGN_RULES_FILE; target.parent.mkdir(parents=True, exist_ok=True); "
        "target.write_text(CAMPAIGN_RULES, encoding='utf-8'); print(target)"
    )
    rules = ctx.shell(f"python3 -c {shlex.quote(write)}").stdout.strip()
    prompt = "Run exactly this shell command and report its first output line verbatim, or the refusal you get: kapso build --help"
    trust = f'projects.{json.dumps(str(session))}.trust_level="trusted"'
    run = ctx.shell(
        f"cd {session} && timeout 300 codex exec --skip-git-repo-check --sandbox workspace-write --json "
        f"-c {shlex.quote(trust)} -c {shlex.quote('mcp_servers.kapso.command=\"kapso-mcp\"')} -c mcp_servers.kapso.enabled=false "
        f"{shlex.quote(prompt)}", check=False, timeout=360,
    )
    final = ""
    for line in run.stdout.splitlines():
        if line.startswith("{"):
            event = json.loads(line)
            if event.get("type") == "item.completed" and event["item"].get("type") == "agent_message":
                final = event["item"].get("text") or final
    refused = "never inside a Kapso campaign session" in final and "usage: kapso build" not in final.lower()
    artifacts = [ctx.artifact("codex-rules-probe.txt", f"rules file: {rules}\n\nfinal message:\n{final}\n\nstderr:\n{run.stderr[-2000:]}\n")]
    return shell_step("codex-rules-probe", refused and Path(rules).is_file(), artifacts, (time.monotonic() - started) / 60,
                      ["the session's cwd is a gitignored subdirectory of a repository, as an implementation session's is; the rules file sits in it"])


def step_cold_start(ctx: Context) -> Dict[str, Any]:
    started = time.monotonic()
    launcher = ctx.clone / "plugin" / "scripts" / "kapso-mcp"
    timing = ctx.shell(f"TIMEFORMAT=%R; time bash {launcher} --version", check=False)
    details = ctx.shell("claude plugin details kapso@kapso", check=False)
    text = f"launcher --version: {timing.stdout.strip()} in {timing.stderr.strip()} s\n\n{details.stdout}"
    return shell_step("cold-start", timing.returncode == 0, [ctx.artifact("cold-start.txt", text)], (time.monotonic() - started) / 60, ["partial: the uvx and pipx branches resolve only once the pin is on PyPI"])


STEPS: Dict[str, Callable[[Context], Dict[str, Any]]] = {
    "install": step_install, "setup": step_setup, "cold-install": step_cold_install, "build-churn": step_build_churn,
    "build-pause": step_build_pause,
    "watch": step_watch, "harness-export": step_harness_export,
    "inbox": step_inbox, "build-fraud": step_build_fraud, "build-data-only": step_build_data_only,
    "build-empty-wish": step_build_empty_wish, "harness-new": step_harness_new, "harness-probe": step_harness_probe,
    "review": step_review, "deploy": step_deploy, "learn": step_learn, "learn-complete": step_learn_complete,
    "campaign-tools": step_campaign_tools, "session-brief": step_session_brief, "experience-share": step_experience_share, "second-engineer-build": step_second_engineer_build,
    "harness-improve": step_harness_improve,
    "campaign-session-isolation": step_campaign_session_isolation, "triggers": step_triggers,
    "codex-rules-probe": step_codex_rules_probe, "cold-start": step_cold_start,
}
assert list(STEPS) == STEP_NAMES


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", choices=["claude_code", "codex"], required=True)
    parser.add_argument("--clone", required=True, help="the repository clone (the local marketplace)")
    parser.add_argument("--out", required=True, help="where the record, transcripts and artifacts go")
    parser.add_argument("--work", default=None, help="the scratch root for the fixtures and campaigns (default: <out>/work)")
    parser.add_argument("--steps", default=None, help="a comma-separated subset, in the round's order")
    parser.add_argument("--from", dest="start", default=None, help="continue from this step")
    parser.add_argument("--wheel", default=None, help="the wheel the cold install uses")
    parser.add_argument("--machine", default=os.uname().nodename)
    parser.add_argument("--learn-model", default=None, help="run the learning crews on this model instead of the shipped one (said in the record)")
    parser.add_argument("--pause-minutes", type=int, default=_PAUSE_B["minutes"], help="the injected pause campaign's budget (the suite's builds.pause.minutes); a slow box needs more than the default for its session to reach the ask")
    parser.add_argument("--recompute-triggers", action="store_true", help="read the trigger step's verdict again from its artifacts, run nothing")
    parser.add_argument("--triggers-sandbox", default=None, help="codex's sandbox for the trigger sessions on a machine whose own cannot start (the runner's word; default: the suite's triggers.codex_sandbox)")
    parser.add_argument("--advisory", default=None, help="the reason this arm runs on the development box; the record is marked and never a fresh-machine acceptance")
    parser.add_argument("--user", default="scripted", help="who answers the host: scripted (the mechanical round) or persona:<card> (a card under tests/semantic/personas through the other vendor's CLI)")
    args = parser.parse_args(argv)
    host = ClaudeHost() if args.host == "claude_code" else CodexHost()
    out = Path(args.out).resolve()
    work = Path(args.work).resolve() if args.work else out / "work"
    work.mkdir(parents=True, exist_ok=True)
    user = UserMode(args.user, host.name)
    ctx = Context(host, Path(args.clone).resolve(), work, out, Path(args.wheel).resolve() if args.wheel else None, args.machine, args.learn_model, args.pause_minutes, user,
                  triggers_sandbox=args.triggers_sandbox)
    test_name = "S2" if user.kind == "persona" else "S2-mechanical"
    record_file = out / "record.json"
    started = datetime.now(timezone.utc)
    wheel_sha256 = subprocess.run(["sha256sum", str(ctx.wheel)], capture_output=True, text=True).stdout.split()[0] if ctx.wheel else "-"
    record = json.loads(record_file.read_text()) if record_file.exists() else {
        "test": test_name, "host": host.name, "machine": args.machine,
        "run_id": f"{test_name}-{host.name}-{started.strftime('%Y%m%dT%H%M%SZ')}", "advisory": args.advisory,
        "package": {"version": subprocess.run(["kapso", "--version"], capture_output=True, text=True).stdout.strip().split()[-1], "wheel_sha256": wheel_sha256},
        "pins": {"wheel_sha256": wheel_sha256, "suite_sha256": hashlib.sha256(SUITE_PATH.read_bytes()).hexdigest()},
        "user": user.record(),
        "started": started.isoformat(timespec="seconds"), "finished": None, "steps": [], "metrics": {}, "notes": [note for note in [codex_session_note()] if note], "transcripts": [],
    }
    if args.recompute_triggers:
        # The verdict's rule changed after the suite ran: the same evidence
        # read again, the record saying so.
        verdict = trigger_verdict(ctx)
        for step in record["steps"]:
            if step["name"] == "triggers":
                step["completed"] = verdict["completed"]
                step["notes"] = [*verdict["notes"], "verdict recomputed from the artifacts after the bar's rule read the held-out cases as their own group"]
        record_file.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(verdict))
        return 0
    selected = args.steps.split(",") if args.steps else list(STEP_NAMES)
    if args.start:
        selected = selected[selected.index(args.start):]
    done = {step["name"] for step in record["steps"]}
    for name in selected:
        if name in done and not args.start:
            continue
        print(f"== {name}", flush=True)
        if host.name == "codex" and name in CODEX_DEFERRED:
            step = {"name": name, "deferred": CODEX_DEFERRED[name]}
        elif host.name == "claude_code" and name in CLAUDE_DEFERRED:
            step = {"name": name, "deferred": CLAUDE_DEFERRED[name]}
        else:
            step = STEPS[name](ctx)
        record["steps"] = [existing for existing in record["steps"] if existing["name"] != name] + [step]
        record["user"] = {**record.get("user", {}), **user.record(), "interventions": sorted(set(record.get("user", {}).get("interventions", [])) | set(user.interventions))}
        # A persona that never corrected the planted baseline nodded: the
        # engaging and not-annoying dimensions are reported, not binding,
        # for this arm (not earned against a user who nods through a wrong
        # figure). The arm itself stays what it is — `advisory` is the
        # development box's mark alone, never the persona's.
        if user.pushback is False:
            record["advisory_dimensions"] = {"dimensions": ["engaging", "not annoying"], "reason": "the persona did not correct the planted baseline"}
        record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        record["transcripts"] = sorted(f"transcripts/{path.name}" for path in (out / "transcripts").glob("*.jsonl"))
        record_file.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in step.items() if key != "notes"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
