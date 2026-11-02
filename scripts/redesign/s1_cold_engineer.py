"""S1 of the semantic suite (docs/plans/redesign/wp9-semantic-suite.md,
change 6): the cold engineer on the Python and command-line flow.

A model with no Kapso knowledge, in a sandbox holding the public docs (the
pages docs.json navigates to, nothing else), a task view (the repository's
own README, never the example's), the package installed into a venv that
is the Python its sessions use, attempts six tasks in Python and then on
the command line, each a fresh host session that is plugin-free, server-
free and web-free (proved from the session's init event). Per attempt:
`first_try` (the counted verb's first execution succeeds, exit 0 with the
task's evidence), `wrong_calls` (errored executions of the counted verb
before success; `--help`, `doctor`, reads and searches never count),
`minutes_to_launch`, `error_led_to_fix` (an errored execution followed by
the next execution of the same verb succeeding). Then the comprehension
questions and the two counts the design names (the Python flow in twelve
lines, the CLI flow in six commands). `--package released` runs it on the
released wheel and the tag's docs, the retrospective baseline.

    python3 s1_cold_engineer.py --host claude_code --clone <clone> --out <dir> --wheel <whl> [--package released] [--steps a,b] [--advisory <reason>]
"""

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tokenize
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plugin_proof  # noqa: E402
from plugin_proof import SUITE, ClaudeHost, CodexHost, record_step, shell_step, status_of, wait_for  # noqa: E402
import judge  # noqa: E402  (the shared judge-block resolver; litellm rides along, present where S1 runs)

RELEASED = SUITE["released"]["version"]
PREREQUISITE = SUITE["builds"]["prerequisite"]
# The task table: the counted verb's shapes on each surface, and the evidence.
TASKS = [
    {"number": 1, "goal": "start a campaign for a better churn model from the repository under ./task (its evaluator under task/evaluation, its data under task/data, the brief task/kapso-brief.md), at a budget of 10 minutes and 1 experiment, into ./campaign-1; the campaign runs on its own, and the task is done once it is running",
     "verb": "build", "evidence": "campaign-1"},
    {"number": 2, "goal": "start a campaign for a fraud scorer that beats the rule-based one from the repository under ./task-fraud, which has data and a brief with the protocol but no evaluator, at a budget of 10 minutes and 1 experiment, into ./campaign-2; the campaign runs on its own, and the task is done once it is running",
     "verb": "build", "evidence": "campaign-2"},
    {"number": 3, "goal": "start the churn campaign again, under the harness at ./harness, at 10 minutes and 1 experiment, into ./campaign-3 (the task's deviation: the harness is named by its path, since no bundled harness ships yet); the campaign runs on its own, and the task is done once it is running",
     "verb": "build", "evidence": "campaign-3", "harness": True},
    {"number": 4, "restore": "campaign-prereq", "goal": "the delivered result of the campaign at ./campaign-prereq deployed to the local target, then called once", "verb": "deploy", "evidence": "deployment"},
    # The learn runs on the campaign as task 4 left it, deployed: a restore
    # here wiped the deployment record, and the comprehension turn — a fresh
    # session reading the sandbox — could not say what the deploy returned
    # (rel-s1-claude, 17 of 17 steps and the comprehension 0).
    {"number": 5, "clear_learning": True, "goal": "lessons learned from the campaign at ./campaign-prereq: start the learn so that it keeps running on its own after your command returns (it takes hours); the task is done once it is running", "verb": "learn", "evidence": "learn-status"},
    {"number": 6, "restore": "campaign-killed", "goal": "the campaign at ./campaign-killed was interrupted before it finished; continue it from where it stopped, on its own: the task is done once it is running again", "verb": "resume", "evidence": "resumed"},
]
# The counted verb's shapes on each surface, per package: the branch's
# `build` is 0.7.10's verb `evolve` (its resume `evolve --resume`); deploy   # 0.7.10's verb
# and learn keep their names. The released arm is counted in its own verbs
# — a table that knew only the new names counted nothing the baseline ran
# and published a first-try figure that was the detector's, not the product's.
VERB_PATTERNS = {
    "branch": {
        "build": {"cli": re.compile(r"^kapso\s+build\b(?!.*--resume)"), "python": re.compile(r"\.build\(")},
        "deploy": {"cli": re.compile(r"^kapso\s+deploy\b"), "python": re.compile(r"\.deploy\(")},
        "learn": {"cli": re.compile(r"^kapso\s+learn\b"), "python": re.compile(r"\.learn\(")},
        "resume": {"cli": re.compile(r"^kapso\s+build\b.*--resume"), "python": re.compile(r"\.build\(.*resume\s*=\s*True")},
    },
    "released": {
        "build": {"cli": re.compile(r"^kapso\s+evolve\b(?!.*--resume)"), "python": re.compile(r"\.evolve\(")},   # 0.7.10's verb
        "deploy": {"cli": re.compile(r"^kapso\s+deploy\b"), "python": re.compile(r"\.deploy\(")},
        "learn": {"cli": re.compile(r"^kapso\s+learn\b"), "python": re.compile(r"\.learn\(")},
        "resume": {"cli": re.compile(r"^kapso\s+evolve\b.*--resume"), "python": re.compile(r"\.evolve\(.*resume")},   # 0.7.10's verb
    },
}
NOT_COUNTED = re.compile(r"--help|^kapso\s+doctor\b|^kapso\s+--version\b|^kapso\s+watch\b|^kapso\s+trace\b")
# A command's segments: what runs between `&&`, `||`, `;`, `|` and
# newlines (a `bash -lc '...'` wrapper opened first), each stripped of
# leading variable assignments and wrappers, so a segment's first word is
# the program it invokes — a `sed` over the docs that mentions `kapso
# build` is a search, never an execution.
WRAPPERS = ("nohup", "sudo", "env", "exec")
SHELL_WRAPPER = re.compile(r"^(?:/bin/)?(?:ba)?sh\s+-lc\s+(['\"])(.*)\1\s*$", re.S)


HEREDOC = re.compile(r"<<-?\s*(?:'([^']*)'|\"([^\"]*)\"|([^\s;|&<]+))")


def _split_outside_quotes(command: str) -> List[str]:
    """The command cut at `&&`, `||`, `;`, `|` and newlines that stand
    outside quotes, read the shell's way (a `|` inside a quoted pattern
    splits nothing — the splitter that cut `rg 'config|kapso build'` into
    an invocation counted a search as a wrong call). A heredoc's body is
    not shell syntax: the lines after a `<<WORD` command up to the
    terminator stay with that command, whatever quotes they hold (an
    apostrophe in a script written by heredoc swallowed the launch that
    followed it, and a Codex flow counted no execution)."""
    segments, current, quote, heredocs, index = [], [], None, [], 0
    while index < len(command):
        char = command[index]
        if quote:
            if char == "\\" and quote == '"' and index + 1 < len(command):
                current.append(command[index:index + 2])
                index += 2
                continue
            if char == quote:
                quote = None
            current.append(char)
            index += 1
            continue
        if char in ("'", '"'):
            quote = char
            current.append(char)
            index += 1
            continue
        if char == "\\" and index + 1 < len(command):
            current.append(command[index:index + 2])
            index += 2
            continue
        if command.startswith("<<", index) and not command.startswith("<<<", index):
            opened = HEREDOC.match(command, index)
            if opened:
                heredocs.append(next(word for word in opened.groups() if word is not None))
                current.append(opened.group(0))
                index = opened.end()
                continue
        if command[index:index + 2] in ("&&", "||"):
            segments.append("".join(current))
            current = []
            index += 2
            continue
        if char == "\n" and heredocs:
            # The bodies follow this line, one per `<<WORD` in order, each
            # ending at its terminator's line; they are the command's.
            index += 1
            for word in heredocs:
                while index < len(command):
                    end = command.find("\n", index)
                    line = command[index:] if end < 0 else command[index:end]
                    current.append("\n" + line)
                    index = len(command) if end < 0 else end + 1
                    if line.strip() == word:
                        break
            heredocs = []
            segments.append("".join(current))
            current = []
            continue
        if char in ";|\n":
            segments.append("".join(current))
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    segments.append("".join(current))
    return segments


def command_segments(command: str) -> List[str]:
    wrapped = SHELL_WRAPPER.match(command.strip())
    if wrapped:
        command = wrapped.group(2)
    segments = []
    for raw in _split_outside_quotes(command):
        # A subshell's parentheses and a trailing `&` are the shell's, not
        # the program's: `(kapso build … &)` invokes kapso.
        words = raw.strip().lstrip("(").rstrip(")& ").split()
        while words and (re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]) or words[0] in WRAPPERS or (words[0] == "timeout" and len(words) > 1)):
            words = words[2:] if words[0] == "timeout" else words[1:]
        if words:
            segments.append(" ".join(words))
    return segments


def invocations(command: str, program: str) -> List[str]:
    """The segments of a command that invoke `program` (by its name or by a
    path ending in it), each as `<program> <args>`."""
    found = []
    for segment in command_segments(command):
        words = segment.split()
        if words[0] == program or words[0].endswith("/" + program):
            found.append(" ".join([program, *words[1:]]))
    return found


def counted(command: str, verb: str, surface: str, sandbox: Path, package: str) -> bool:
    """Whether a shell command executes the counted verb on the surface: on
    the command line a kapso invocation carrying the verb (help, doctor,
    watch and trace never count); in Python a python invocation running a
    script whose text carries the call."""
    pattern = VERB_PATTERNS[package][verb][surface]
    if surface == "cli":
        return any(pattern.search(call) and not NOT_COUNTED.search(call) for call in invocations(command, "kapso"))
    for call in (*invocations(command, "python"), *invocations(command, "python3")):
        for script in re.findall(r"(\S+\.py)\b", call):
            path = sandbox / script
            if path.is_file() and pattern.search(path.read_text(encoding="utf-8", errors="replace")):
                return True
    return False
COMPREHENSION = (
    "You have used this package for six tasks in this directory. Answer in one short paragraph each, from what you ran and saw: "
    "(1) what did the build call do and what did it return? (2) what did the deploy call do and what did it return? "
    "(3) what did the learn call do and what did it return? (4) what did the resume do? (5) what did the harness change? "
    "(6) what is the evaluator and who runs it? (7) where is the record of a campaign on disk?"
)
CLAUDE_COLD = ["--settings", json.dumps({"enabledPlugins": {"kapso@kapso": False}}), "--strict-mcp-config", "--disallowedTools", "WebFetch,WebSearch"]
CODEX_COLD = ["-c", 'mcp_servers.kapso.command="kapso-mcp"', "-c", "mcp_servers.kapso.enabled=false", "-c", 'web_search="disabled"']


# ------------------------------------------------------------ the sandbox
def navigated_pages(clone: Path) -> List[str]:
    """The public docs: every page docs.json navigates to, nothing else."""
    pages: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            pages.append(node)
        elif isinstance(node, dict):
            for key in ("tabs", "groups", "pages", "anchors", "dropdowns"):
                if key in node:
                    walk(node[key])
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(json.loads((clone / "docs.json").read_text(encoding="utf-8"))["navigation"])
    # The navigation names pages from the repository root (`docs/index`);
    # the snapshot resolves them under the docs directory.
    return [page[len("docs/"):] if page.startswith("docs/") else page for page in pages]


def snapshot_docs(source: Path, pages: Sequence[str], into: Path) -> int:
    """The allow-listed snapshot: the navigated pages as `.mdx`, copied with
    their paths; the images beside them. Returns the count."""
    if into.exists():
        shutil.rmtree(into)
    count = 0
    for page in pages:
        file = source / f"{page}.mdx"
        if file.is_file():
            target = into / f"{page}.mdx"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
            count += 1
    if (source / "images").is_dir():
        shutil.copytree(source / "images", into / "images")
    return count


def task_view(example: Path, into: Path, generate: bool = True) -> None:
    """The repository as an engineer brought it: the example without its
    README (the repository's own README stands in), its data generated."""
    if into.exists():
        shutil.rmtree(into)
    shutil.copytree(example, into, ignore=shutil.ignore_patterns("campaign", "__pycache__", "data", "private", "result.json", "README.md", "run.py", "baseline.py", "baseline.json"))
    own = example / "repo" / "README.md"
    (into / "README.md").write_text(own.read_text(encoding="utf-8") if own.is_file() else f"# {example.name}\n", encoding="utf-8")
    if generate and (into / "generate.py").is_file():
        subprocess.run([sys.executable, "generate.py"], cwd=into, check=True, capture_output=True, text=True, timeout=600)


def session_init(rows: Sequence[dict]) -> Dict[str, Any]:
    """What a Claude Code session started with: the proof that the sandbox
    is cold (no plugin, no server, no Kapso command, no web tool)."""
    for row in rows:
        if row.get("type") == "system" and row.get("subtype") == "init":
            return {
                "plugins": [p.get("source") for p in row.get("plugins", []) if p.get("source", "").startswith("kapso")],
                "mcp_servers": [s.get("name") for s in row.get("mcp_servers", [])],
                "kapso_commands": [c for c in row.get("slash_commands", []) if "kapso" in c],
                "web_tools": [t for t in row.get("tools", []) if t in ("WebFetch", "WebSearch")],
            }
    return {"plugins": None, "mcp_servers": None, "kapso_commands": None, "web_tools": None}


def is_cold(init: Dict[str, Any]) -> Optional[bool]:
    """True when the session proved it is cold (no plugin, server, Kapso
    command or web tool); None when no init row carried the proof (a Codex
    session), which the sandbox step reports as not proved, never a pass."""
    if init["plugins"] is None:
        return None
    return init["plugins"] == [] and init["mcp_servers"] == [] and init["kapso_commands"] == [] and init["web_tools"] == []


# ----------------------------------------------------------- the metrics
def executions(rows: Sequence[dict], verb: str, surface: str, sandbox: Path, package: str = "branch") -> List[Dict[str, Any]]:
    """The counted verb's executions in a session, in order, each with
    whether it errored: on the command line a Bash call that invokes kapso
    with the verb (the package's own: 0.7.10's `evolve` for the released
    arm); in Python a Bash call running a script whose text carries the
    call. Help, doctor, watch, trace, reads and searches are never counted."""
    found: List[Dict[str, Any]] = []
    pending: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("type") == "assistant":
            for block in row["message"].get("content", []):
                if block.get("type") != "tool_use" or block.get("name") != "Bash":
                    continue
                command = block.get("input", {}).get("command", "")
                if counted(command, verb, surface, sandbox, package):
                    entry = {"command": command, "errored": None}
                    pending[block["id"]] = entry
                    found.append(entry)
        elif row.get("type") == "user":
            for block in row["message"].get("content", []):
                if block.get("type") == "tool_result" and block.get("tool_use_id") in pending:
                    # The host's own verdict on the command (a non-zero exit),
                    # never its output's words: a deploy that streams its
                    # agent's session log prints that agent's probes and
                    # tracebacks and exits 0.
                    pending.pop(block["tool_use_id"])["errored"] = bool(block.get("is_error"))
        elif row.get("type") in ("item.started", "item.completed") and (row.get("item") or {}).get("type") == "command_execution":
            # Codex's shape: the command at its start and again, with its exit
            # status, at its end. A command still running when the turn ended
            # (a foreground resume) has a start and no end: counted, its
            # outcome unknown.
            item = row["item"]
            command = item.get("command", "")
            if not counted(command, verb, surface, sandbox, package):
                continue
            if row["type"] == "item.started":
                entry = {"command": command, "errored": None}
                pending[item.get("id")] = entry
                found.append(entry)
                continue
            exit_code = item.get("exit_code")
            errored = (exit_code != 0) if exit_code is not None else _looks_errored(item.get("aggregated_output") or "")
            if item.get("id") in pending:
                pending.pop(item["id"])["errored"] = errored
            else:
                found.append({"command": command, "errored": errored})
    return found


def _looks_errored(text: str) -> bool:
    """The output's words, read only when the host gave no exit status."""
    return bool(re.search(r"Traceback \(most recent call last\)|^Error:|error: ", text, re.M))


class EvidenceWatcher:
    """Samples a predicate while a session runs and keeps the first time
    it held; `stop` ends the sampling and takes one last look, so the
    launch clock stops at the evidence, not after the turn."""

    def __init__(self, predicate: Callable[[], bool], every: float = 2.0):
        self.predicate, self.every = predicate, every
        self.held_at: Optional[float] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.predicate():
                self.held_at = time.time()
                return
            self._stop.wait(self.every)

    def stop(self, grace_seconds: float = 0.0) -> Optional[float]:
        """Ends the sampling: one last look, and up to `grace_seconds` of
        looking after the turn — a detached launch's child writes its
        status seconds after the command returns, and a host whose turn ends
        at the launch (Codex: two launches at 0.5 and 0.9 minutes,
        rel-s1-codex) left the evidence to land just after the look. The
        launch clock still stops at the evidence."""
        self._stop.set()
        self._thread.join()
        deadline = time.time() + grace_seconds
        while self.held_at is None:
            if self.predicate():
                self.held_at = time.time()
            elif time.time() >= deadline:
                break
            else:
                time.sleep(self.every)
        return self.held_at


def attempt_metrics(found: Sequence[Dict[str, Any]], evidence: bool, started: float, evidence_at: Optional[float]) -> Dict[str, Any]:
    """`first_try`, `wrong_calls`, `minutes_to_launch`, `error_led_to_fix`
    from the executions and the evidence."""
    successes = [index for index, entry in enumerate(found) if entry["errored"] is False]
    first_success = successes[0] if successes else None
    wrong = len([entry for entry in found[: first_success if first_success is not None else len(found)] if entry["errored"]])
    led_to_fix = None
    if wrong:
        # One step: exactly one errored call, and it was followed by the success.
        led_to_fix = first_success is not None and wrong == 1
    return {
        "first_try": bool(found) and found[0]["errored"] is False and evidence,
        "wrong_calls": wrong,
        "minutes_to_launch": round((evidence_at - started) / 60, 1) if evidence_at else None,
        "error_led_to_fix": led_to_fix,
        "executions": len(found),
        "evidence": evidence,
    }


def logical_lines(text: str) -> int:
    """A script's length as the design counts a flow: its logical lines
    (the tokenizer's NEWLINE tokens — a call written across six physical
    lines is one line of the flow), comments and blank lines aside."""
    return sum(1 for token in tokenize.generate_tokens(io.StringIO(text).readline) if token.type == tokenize.NEWLINE)


def python_lines(sandbox: Path, scripts: Sequence[str]) -> int:
    count = 0
    for script in scripts:
        path = sandbox / script
        if path.is_file():
            count += logical_lines(path.read_text(encoding="utf-8", errors="replace"))
    return count


# ------------------------------------------------------------- evidence
def pid_alive(pid: Any) -> bool:
    """Whether a process exists, from the process table (the machines are
    Linux); no signal is sent and nothing raises."""
    return isinstance(pid, int) and Path(f"/proc/{pid}").exists()


def seed_env(project: Path) -> None:
    """The token the sessions need, from the environment the round runs in
    (the machine's login is a token in its secrets file), into the
    sandbox's `.env` as the doctor's fix line says; nothing when the
    machine has a stored login."""
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if token:
        with open(project / ".env", "a", encoding="utf-8") as env:
            env.write(f"CLAUDE_CODE_OAUTH_TOKEN={token}\n")


def evidence_of(task: Dict[str, Any], sandbox: Path, prereq: Path) -> bool:
    kind = task["evidence"]
    if kind.startswith("campaign-"):
        status = status_of(sandbox / kind)
        if status is None:
            return False
        return not task.get("harness") or bool(status.get("harness"))
    if kind == "deployment":
        return (prereq / ".kapso" / "deployments" / "1" / "deployment.json").is_file()
    if kind == "learn-status":
        return any(sandbox.glob("**/learning/status/*.json"))
    if kind == "resumed":
        # A resume that is still in flight (running, live pid) or that already
        # carried the killed campaign through to the end (done) are both proof
        # it continued from the checkpoint; the killed campaign on its own stays
        # stopped at the kill phase, so neither state arises without a resume.
        status = status_of(sandbox / "campaign-killed") or {}
        state = status.get("state")
        return state == "done" or (state == "running" and pid_alive(status.get("pid")))
    raise ValueError(kind)


# ---------------------------------------------------------------- driver
def prerequisite_launch(package: str, campaign: Path, minutes: int, experiments: int) -> str:
    """The command that builds a prerequisite campaign from a task view, in
    the package's own verb: the branch's `build` detaches itself; 0.7.10's
    `evolve` has no detach, so it is backgrounded (its status.json carries
    the pid the killed variant and `_stop` signal)."""
    if package == "released":
        return (f"nohup kapso evolve -f kapso-brief.md --eval-dir evaluation --data-dir data --initial-repo repo -o {campaign} "
                f"-i {experiments} --time-budget-minutes {minutes} > {campaign}.log 2>&1 &")
    return (f"kapso build --brief kapso-brief.md --evaluation evaluation --data data --repo repo --output {campaign} "
            f"--budget-minutes {minutes} --budget-experiments {experiments} --detach")


class S1:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.clone = Path(args.clone).resolve()
        self.out = Path(args.out).resolve()
        self.sandbox = Path(args.sandbox).resolve() if args.sandbox else self.out / "sandbox"
        (self.out / "transcripts").mkdir(parents=True, exist_ok=True)
        (self.out / "artifacts").mkdir(parents=True, exist_ok=True)
        extra = CLAUDE_COLD if args.host == "claude_code" else CODEX_COLD
        self.host = ClaudeHost(extra) if args.host == "claude_code" else CodexHost(extra)
        self.venv = self.sandbox / "venv"
        self.env = {**os.environ, "PATH": f"{self.venv / 'bin'}{os.pathsep}{os.environ['PATH']}"}

    def shell(self, command: str, cwd: Optional[Path] = None, check: bool = True, timeout: int = 1800) -> subprocess.CompletedProcess:
        run = subprocess.run(["bash", "-c", command], capture_output=True, text=True, cwd=str(cwd or self.sandbox), timeout=timeout, env=self.env)
        with open(self.out / "shell.log", "a", encoding="utf-8") as log:
            log.write(f"$ {command}\n{run.stdout}{run.stderr}[exit {run.returncode}]\n")
        if check and run.returncode != 0:
            raise RuntimeError(f"{command}: exit {run.returncode}: {run.stderr[-2000:]}")
        return run

    def artifact(self, name: str, text: str) -> str:
        (self.out / "artifacts" / name).write_text(text, encoding="utf-8")
        return f"artifacts/{name}"

    def turn(self, step: str, prompt: str):
        transcript = self.out / "transcripts" / f"{step}.jsonl"
        os.environ["PATH"] = self.env["PATH"]
        return self.host.turn(self.sandbox, prompt, None, transcript), transcript

    # -- steps --
    def step_sandbox(self) -> Dict[str, Any]:
        started = time.monotonic()
        self.sandbox.mkdir(parents=True, exist_ok=True)
        if self.args.package == "released":
            # The tag's docs and docs.json, staged into the pack by semantic.sh
            # (the clone carries no .git to archive them from).
            docs_source = self.clone / "released-docs"
            if not (docs_source / "docs.json").is_file():
                raise RuntimeError(f"no released docs at {docs_source}: semantic.sh stages the v{RELEASED} tag's docs into the harness pack")
            pages = navigated_pages(docs_source)
            count = snapshot_docs(docs_source / "docs", pages, self.sandbox / "docs")
            self.shell(f"python3 -m venv {self.venv} && {self.venv}/bin/pip install -q 'leeroo-kapso=={RELEASED}'", cwd=self.out, timeout=900)
        else:
            pages = navigated_pages(self.clone)
            count = snapshot_docs(self.clone / "docs", pages, self.sandbox / "docs")
            self.shell(f"python3 -m venv {self.venv} && {self.venv}/bin/pip install -q {self.args.wheel}", cwd=self.out, timeout=900)
        # The engineer's own environment holds what the task views say to
        # install (the examples' requirements: numpy, pandas, scikit-learn;
        # the harness's pandas and pyarrow) — with no web, a venv holding the
        # wheel alone met every build as a refusal or an in-session install.
        requirements = sorted((self.clone / "examples").glob("*/requirements.txt")) + sorted((self.clone / "examples" / "harnesses").glob("*/requirements.txt"))
        self.shell(f"{self.venv}/bin/pip install -q " + " ".join(f"-r {path}" for path in requirements), cwd=self.out, timeout=900)
        # Codex runs each command in a login shell, whose profile puts the
        # machine's own ~/.local/bin (the gate's uv tool install of the
        # package) ahead of the venv the driver put first on PATH: the
        # engineer's deploy ran the machine's kapso, without the task's
        # requirements, and failed until the engineer put the venv first by
        # hand. The profile ends with the sandbox's venv first, as the prompt
        # says it is.
        profile = Path.home() / ".profile"
        export = f'export PATH="{self.venv / "bin"}:$PATH"   # the S1 sandbox\n'
        if not (profile.is_file() and export in profile.read_text(encoding="utf-8")):
            with open(profile, "a", encoding="utf-8") as out:
                out.write("\n" + export)
        task_view(self.clone / "examples" / "churn-first-build", self.sandbox / "task")
        task_view(self.clone / "examples" / "fraud-no-evaluator", self.sandbox / "task-fraud")
        if (self.sandbox / "harness").exists():
            shutil.rmtree(self.sandbox / "harness")
        shutil.copytree(self.clone / "examples" / "harnesses" / "trading", self.sandbox / "harness", ignore=shutil.ignore_patterns("__pycache__"))
        (self.sandbox / ".env").write_text("", encoding="utf-8")
        seed_env(self.sandbox)
        version = self.shell(f"{self.venv}/bin/python -c 'import kapso; print(kapso.__version__)'", check=False)
        # The CLI's provenance is its path (the sandbox's venv) and that it
        # runs; the version is the import's, since 0.7.10 has no --version.
        cli = self.shell("command -v kapso && kapso --help >/dev/null", check=False)
        probe, transcript = self.turn("sandbox", "Say only the word ready.")
        init = session_init(plugin_proof_rows(transcript))
        cold = is_cold(init)
        isolated = (cold is True) if self.args.host == "claude_code" else True
        proof = {True: "proved cold from the session's init", False: "NOT cold (the init carried a plugin, server, command or web tool)", None: "not provable from the stream; the CODEX_COLD flags set the isolation"}[cold]
        artifacts = [self.artifact("sandbox.json", json.dumps({"pages": count, "import": version.stdout.strip(), "cli": cli.stdout.strip(), "init": init, "cold": cold}, indent=2))]
        return shell_step("sandbox", version.returncode == 0 and cli.returncode == 0 and isolated, artifacts, (time.monotonic() - started) / 60,
                          [f"{count} public pages; the package {version.stdout.strip()} in the sandbox's venv; isolation: {proof}"])

    def _stop(self, campaign: Path) -> None:
        """A live campaign stopped before its directory is replaced: the
        branch's `build --stop`; 0.7.10 has no stop verb, so the pid its
        status.json records is signalled."""
        if self.args.package == "released":
            pid = (status_of(campaign) or {}).get("pid")
            if pid_alive(pid):
                os.kill(int(pid), signal.SIGTERM)
            return
        self.shell(f"kapso build --output {campaign} --stop", check=False)

    def _stop_learns(self) -> List[int]:
        """Every learn still running from an earlier flow — the engineer's
        own detached script or `kapso learn --detach`, with the crews in
        its process group — stopped before the next flow: the CLI flow of
        task 5 found the python flow's learn still mining and rightly
        declined to start a second (rel-s1-claude). A process is the
        sandbox's when its command line names the sandbox or it runs from
        inside it, and a learn's when its command line says `learn` (the
        verb, or the engineer's own `learn_prereq.py`)."""
        stopped = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid() or entry.stat().st_uid != os.getuid():
                continue
            words = [word.decode("utf-8", "replace") for word in (entry / "cmdline").read_bytes().split(b"\0") if word]
            if not any("learn" in word for word in words):
                continue
            cwd = os.readlink(entry / "cwd")
            if not (any(str(self.sandbox) in word for word in words) or cwd == str(self.sandbox) or cwd.startswith(f"{self.sandbox}/")):
                continue
            os.killpg(os.getpgid(int(entry.name)), signal.SIGTERM)
            stopped.append(int(entry.name))
        wait_for(lambda: not any(pid_alive(pid) for pid in stopped), 20, every=1)
        for pid in stopped:
            if pid_alive(pid):
                os.killpg(os.getpgid(pid), signal.SIGKILL)
        return stopped

    def _build(self, name: str, example: str, minutes: int, experiments: int, kill_after: Optional[int] = None) -> Dict[str, Any]:
        started = time.monotonic()
        project = self.sandbox / f"prereq-{name}"
        task_view(self.clone / "examples" / example, project)
        campaign = self.sandbox / name
        if campaign.exists():
            self._stop(campaign)
            shutil.rmtree(campaign)
        launch = self.shell(prerequisite_launch(self.args.package, campaign, minutes, experiments), cwd=project, check=False)
        if kill_after:
            wait_for(lambda: (status_of(campaign) or {}).get("phase") not in (None, "setup"), kill_after * 60, every=15)
            pid = (status_of(campaign) or {}).get("pid")
            if pid_alive(pid):
                os.kill(int(pid), signal.SIGKILL)
            done = True
        else:
            done = wait_for(lambda: (status_of(campaign) or {}).get("state") in ("done", "failed"), minutes * 60 + 900, every=30)
        status = status_of(campaign) or {}
        return shell_step(f"prerequisite-{'campaign' if not kill_after else 'killed'}", launch.returncode == 0 and done, [self.artifact(f"{name}-status.json", json.dumps(status, indent=2))],
                          (time.monotonic() - started) / 60, [f"delivered: {(status.get('best') or {}).get('score')}" if not kill_after else f"the writer killed at {kill_after} minutes, phase {status.get('phase')}"], kind="build")

    def _snapshot(self, name: str) -> None:
        snap = self.sandbox / ".snap" / name
        if snap.exists():
            shutil.rmtree(snap)
        snap.parent.mkdir(parents=True, exist_ok=True)
        if (self.sandbox / name).exists():
            shutil.copytree(self.sandbox / name, snap, symlinks=True)

    def _restore(self, name: str) -> None:
        snap = self.sandbox / ".snap" / name
        live = self.sandbox / name
        if not snap.exists():
            return
        if live.exists():
            self._stop(live)
            shutil.rmtree(live)
        shutil.copytree(snap, live, symlinks=True)

    def step_prerequisite_campaign(self) -> Dict[str, Any]:
        step = self._build("campaign-prereq", "churn-first-build", PREREQUISITE["minutes"], PREREQUISITE["experiments"])
        self._snapshot("campaign-prereq")
        return step

    def step_prerequisite_killed(self) -> Dict[str, Any]:
        step = self._build("campaign-killed", "churn-first-build", PREREQUISITE["minutes"], PREREQUISITE["experiments"], kill_after=5)
        self._snapshot("campaign-killed")
        return step

    def step_attempt(self, task: Dict[str, Any], surface: str) -> Dict[str, Any]:
        name = f"task{task['number']}-{surface}"
        # The Python flow is asked for as a ticket would ask: a short script.
        # The bar measures how much the package makes an engineer write to
        # drive a task, which a script padded by style (a 39-line launcher
        # with its own polling loop) does not show.
        how = "in Python: write a short script of at most a dozen lines in this directory and run it with ./venv/bin/python" if surface == "python" else "on the command line, with the kapso command on PATH"
        prompt = (
            "You are an engineer who has never used Kapso. The package is installed in this directory's venv (./venv/bin/python; the kapso command is on PATH); "
            "its documentation is under ./docs (start at docs/index.mdx and docs/installation.mdx). Read only what is in this directory; you have no web access "
            "(the campaign's own settings are not yours to change). "
            f"The task: {task['goal']}. Do it {how}. When it is done, say in one line what you ran and what came back."
        )
        for leftover in (self.sandbox / task["evidence"],) if task["evidence"].startswith("campaign-") else ():
            if leftover.exists():
                self.shell(f"kapso build --output {leftover} --stop", check=False)
                shutil.rmtree(leftover)
        if task.get("restore"):
            self._restore(task["restore"])
        if task.get("clear_learning"):
            self._stop_learns()
            for status in self.sandbox.glob("**/learning/status"):
                if status.is_dir():
                    shutil.rmtree(status)
        # The clock starts after the driver's own setup and stops at the
        # first moment the evidence holds, sampled while the session runs
        # (the Interface's "from the session's first event to the evidence");
        # the step's minutes are the turn's wall clock, which the attempt's
        # budget bounds (the host's own duration figure is not it).
        watcher = EvidenceWatcher(lambda: evidence_of(task, self.sandbox, self.sandbox / "campaign-prereq"))
        started = time.time()
        watcher.start()
        turn, transcript = self.turn(name, prompt)
        turn_minutes = (time.time() - started) / 60
        evidence_at = watcher.stop(float(SUITE["waits"]["evidence_grace_seconds"]))
        rows = plugin_proof_rows(transcript)
        found = executions(rows, task["verb"], surface, self.sandbox, self.args.package)
        evidence = evidence_at is not None
        metrics = attempt_metrics(found, evidence, started, evidence_at)
        scripts = sorted({script for entry in found for script in re.findall(r"(\S+\.py)\b", entry["command"]) if (self.sandbox / script).is_file()}) if surface == "python" else []
        metrics["scripts"] = scripts
        metrics["commands"] = [entry["command"] for entry in found] if surface == "cli" else []
        artifacts = [self.artifact(f"{name}-answer.txt", turn.text), self.artifact(f"{name}-executions.json", json.dumps(found, indent=2))]
        step = record_step(name, [turn], bool(evidence), artifacts, kind="attempt")
        step["minutes"] = round(turn_minutes, 1)
        step["metrics"] = metrics
        return step

    def step_comprehension(self) -> Dict[str, Any]:
        turn, transcript = self.turn("comprehension", COMPREHENSION)
        facts = {
            "build": "launches a campaign (detached on the command line: a receipt and a status file; in Python it returns a Solution when the campaign ends)",
            "deploy": "ships the delivered branch to a target and proves one call; the deployment's record under the campaign's .kapso/deployments/",
            "learn": "starts the learning crews over the campaign's record, detached, with a status file under the project's learning/status/",
            "resume": "continues a campaign from its checkpoint with --resume (one flag; the attempts kept)",
            "harness": "the domain layer a build loads: pages, skills, delegates, hooks staged into every session; named in the status",
            "evaluator": "the script the campaign protects and runs on every candidate; the metric on its last line; result.json beside it",
            "record": "the campaign directory's .kapso/: solution.json, status.json, the sessions and attempts; Solution.open reads it; kapso trace renders it",
        }
        key = "\n".join(f"- {verb}: {fact}" for verb, fact in facts.items())
        self.artifact("comprehension-facts.json", json.dumps(facts, indent=2))
        answers = f"# The engineer's answers\n\n{turn.text}\n\n# The key (what each call does; the engineer never saw it)\n\n{key}\n"
        artifacts = [self.artifact("comprehension-answers.txt", answers), "artifacts/comprehension-facts.json"]
        return record_step("comprehension", [turn], bool(turn.text.strip()), artifacts, ["the answers and the key handed to the judge (the s1 rubric); --grade reads the comprehension dimension into comprehension_correct"], kind="reading")

    def step_flow_counts(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return flow_counts(record, self.sandbox, self.artifact)


def flows_text(record: Dict[str, Any], sandbox: Path) -> str:
    """The engineer's flows as the judge reads them: per attempt, the
    script(s) the counted executions ran (their text, with the code-line
    count) or the commands run, beside the attempt's own figures — the
    material for "the flows fit the design's lengths" and "nothing needed
    the docs do not say", which the comprehension answers alone could not
    carry."""
    lines = ["# The engineer's flows (what the counted executions ran)", ""]
    for step in record["steps"]:
        if step.get("kind") != "attempt":
            continue
        metrics = step.get("metrics", {})
        lines.append(f"## {step['name']}: first try {metrics.get('first_try')}, wrong calls {metrics.get('wrong_calls')}, executions {metrics.get('executions')}, evidence {metrics.get('evidence')}")
        for script in metrics.get("scripts", []):
            path = sandbox / script
            text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else "(the script is not in the archive)"
            lines += [f"script {script} ({python_lines(sandbox, [script])} code lines):", "", *["    " + line for line in text.splitlines()], ""]
        for command in metrics.get("commands", []):
            lines.append(f"command: {command}")
        lines.append("")
    return "\n".join(lines) + "\n"


def flow_counts(record: Dict[str, Any], sandbox: Path, artifact: Callable[[str, str], str]) -> Dict[str, Any]:
    """The two counts the design names, per task (a filename reused across
    tasks is not conflated; the bar is per flow, so the longest task's
    figure is the metric), with the flows artifact the judge reads."""
    started = time.monotonic()
    per_task_lines, per_task_commands, detail = {}, {}, {}
    for step in record["steps"]:
        if step.get("kind") != "attempt":
            continue
        metrics = step.get("metrics", {})
        if metrics.get("scripts"):
            per_task_lines[step["name"]] = python_lines(sandbox, metrics["scripts"])
        if metrics.get("commands"):
            per_task_commands[step["name"]] = len(metrics["commands"])
        detail[step["name"]] = {"scripts": metrics.get("scripts", []), "commands": metrics.get("commands", [])}
    metrics = {"python_lines": max(per_task_lines.values(), default=0), "cli_commands": max(per_task_commands.values(), default=0)}
    step = shell_step("flow-counts", True, [artifact("flow-counts.json", json.dumps({**metrics, "per_task_python_lines": per_task_lines, "per_task_cli_commands": per_task_commands, "detail": detail}, indent=2)),
                                           artifact("flows.md", flows_text(record, sandbox))], (time.monotonic() - started) / 60,
                      ["python_lines and cli_commands are the longest single task's flow (the bar is per flow); the per-task figures are kept; a script read in its final state is a known limit when a filename is reused within a task"])
    step["metrics"] = metrics
    return step


def regrade(record: Dict[str, Any], out: Path) -> Dict[str, Any]:
    """Every attempt's executions and metrics read again from the archived
    transcript and sandbox — the patterns or the parsing changed (the
    released arm had been counted under the branch's verbs, and a search
    over the docs had counted as a run) — with the observed facts kept as
    recorded: the evidence and the launch clock were measured then, under
    that driver's clock (its own start to after the turn), and cannot be
    read from the archive. The flow counts and the flows artifact are
    rewritten, the run metrics recomputed, and the record says so."""
    package = "released" if record.get("package", {}).get("released") else "branch"
    sandbox = out / "sandbox"
    for step in record["steps"]:
        if step.get("kind") != "attempt":
            continue
        number, surface = int(step["name"][4]), step["name"].split("-", 1)[1]
        task = next(entry for entry in TASKS if entry["number"] == number)
        rows = plugin_proof_rows(out / "transcripts" / f"{step['name']}.jsonl")
        found = executions(rows, task["verb"], surface, sandbox, package)
        old = step.get("metrics", {})
        metrics = attempt_metrics(found, bool(old.get("evidence")), 0.0, None)
        metrics["minutes_to_launch"] = old.get("minutes_to_launch")
        metrics["scripts"] = sorted({script for entry in found for script in re.findall(r"(\S+\.py)\b", entry["command"]) if (sandbox / script).is_file()}) if surface == "python" else []
        metrics["commands"] = [entry["command"] for entry in found] if surface == "cli" else []
        step["metrics"] = metrics
        (out / "artifacts" / f"{step['name']}-executions.json").write_text(json.dumps(found, indent=2), encoding="utf-8")
        step["notes"] = [*[note for note in step.get("notes", []) if not note.startswith("re-read")],
                         "re-read 2026-10-29 from the archived transcript under the package's own verbs and true invocations; the evidence and the launch clock as observed then (that driver's clock: its start to after the turn)"]

    def artifact(name: str, text: str) -> str:
        (out / "artifacts" / name).write_text(text, encoding="utf-8")
        return f"artifacts/{name}"

    record["steps"] = [step for step in record["steps"] if step["name"] != "flow-counts"] + [flow_counts(record, sandbox, artifact)]
    record["metrics"] = {**record.get("metrics", {}), **run_metrics(record)}
    record["notes"] = [*[note for note in record.get("notes", []) if not note.startswith("re-graded")],
                       "re-graded 2026-10-29: the executions, flow counts and run metrics re-read from the archive (each attempt's note says how); the judge run again over the comprehension answers and the flows"]
    return record


def plugin_proof_rows(transcript: Path) -> List[dict]:
    return [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines() if line.startswith("{")]


def run_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    """The run-level metrics the suite's bars read."""
    attempts = [step for step in record["steps"] if step.get("kind") == "attempt"]
    errored = [step for step in attempts if step["metrics"]["wrong_calls"]]
    counts = next((step["metrics"] for step in record["steps"] if step["name"] == "flow-counts"), {})
    return {
        "first_try_count": sum(1 for step in attempts if step["metrics"]["first_try"]),
        # The one-step rule over the attempts that errored; None when none
        # did (the bar names `errored_calls` as what exercises it, and the
        # report says "not exercised" rather than printing a vacuous pass).
        "errored_calls": len(errored),
        "errors_led_to_fix": (sum(1 for step in errored if step["metrics"]["error_led_to_fix"]) / len(errored)) if errored else None,
        "wrong_calls_max": max((step["metrics"]["wrong_calls"] for step in attempts), default=0),
        "comprehension_correct": record.get("metrics", {}).get("comprehension_correct"),
        "python_lines": counts.get("python_lines"),
        "cli_commands": counts.get("cli_commands"),
    }


COMPREHENSION_DIMENSION = "names say what the calls do"


def grade(record: Dict[str, Any]) -> Dict[str, Any]:
    """The judge's block into the record: `comprehension_correct` is 1.0
    when the judge resolved the comprehension dimension (the answers named
    every call correctly against the key) to the bar, else 0.0."""
    block = record.get("judge") or {}
    if block.get("failure") or not block.get("passes"):
        return record
    spec = SUITE["tests"]["S1"]["judge"]
    labels = sorted({label for a_pass in block["passes"] for label in a_pass.get("items", {})})
    eligible = {label: judge.eligible_dimensions(SUITE, spec, label, spec["dimensions"]) for label in labels}
    final = judge.resolve(block, spec["dimensions"], spec["bar"], eligible).get(COMPREHENSION_DIMENSION)
    record.setdefault("metrics", {})["comprehension_correct"] = 1.0 if final is not None and final >= spec["bar"] else 0.0
    record["metrics"] = {**record["metrics"], **run_metrics(record)}
    return record


STEP_NAMES = ["sandbox", "prerequisite-campaign", "prerequisite-killed", *[f"task{task['number']}-{surface}" for task in TASKS for surface in ("python", "cli")], "comprehension", "flow-counts"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", choices=["claude_code", "codex"])
    parser.add_argument("--clone")
    parser.add_argument("--out", required=True)
    parser.add_argument("--wheel", default=None, help="the wheel built from the branch (the branch's package)")
    parser.add_argument("--package", choices=["branch", "released"], default="branch", help="released: the retrospective baseline on the released wheel and the tag's docs")
    parser.add_argument("--sandbox", default=None)
    parser.add_argument("--machine", default=os.uname().nodename)
    parser.add_argument("--advisory", default=None)
    parser.add_argument("--steps", default=None)
    parser.add_argument("--from", dest="start", default=None)
    parser.add_argument("--grade", action="store_true", help="read the judge's block into comprehension_correct (on the development box)")
    parser.add_argument("--regrade", action="store_true", help="read every attempt's executions again from the archived transcripts and sandbox (the patterns changed), rewrite the counts and the run metrics")
    args = parser.parse_args(argv)
    record_file = Path(args.out).resolve() / "record.json"
    if args.regrade:
        record = regrade(json.loads(record_file.read_text()), Path(args.out).resolve())
        record_file.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(record["metrics"]))
        return 0
    if args.grade:
        record = grade(json.loads(record_file.read_text()))
        record_file.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(record["metrics"]))
        return 0
    if not args.host or not args.clone:
        parser.error("--host and --clone are required to run the cold engineer (omitted only with --grade)")
    if args.package == "branch" and not args.wheel:
        parser.error("--wheel is the branch's package; --package released needs none")
    driver = S1(args)
    record_file = driver.out / "record.json"
    started = datetime.now(timezone.utc)
    record = json.loads(record_file.read_text()) if record_file.exists() else {
        "test": "S1", "host": args.host, "machine": args.machine, "run_id": f"S1-{args.host}-{started.strftime('%Y%m%dT%H%M%SZ')}", "advisory": args.advisory,
        "package": {"version": RELEASED if args.package == "released" else "branch", "wheel_sha256": hashlib.sha256(Path(args.wheel).read_bytes()).hexdigest() if args.wheel else "-", "released": args.package == "released"},
        "pins": {"wheel_sha256": hashlib.sha256(Path(args.wheel).read_bytes()).hexdigest() if args.wheel else "-", "suite_sha256": hashlib.sha256(plugin_proof.SUITE_PATH.read_bytes()).hexdigest()},
        "user": {"kind": "scripted", "card": None, "interventions": ["the task views: the repositories' own READMEs; the prerequisite campaigns built by the driver; the second one killed at its writer"], "events": [], "pushback": None},
        "started": started.isoformat(timespec="seconds"), "finished": None, "steps": [], "metrics": {}, "notes": [note for note in [plugin_proof.codex_session_note()] if note], "transcripts": [],
    }
    selected = args.steps.split(",") if args.steps else list(STEP_NAMES)
    if args.start:
        selected = selected[selected.index(args.start):]
    done = {step["name"] for step in record["steps"]}
    for name in selected:
        if name in done and not args.start:
            continue
        print(f"== {name}", flush=True)
        if name == "sandbox":
            step = driver.step_sandbox()
        elif name == "prerequisite-campaign":
            step = driver.step_prerequisite_campaign()
        elif name == "prerequisite-killed":
            step = driver.step_prerequisite_killed()
        elif name == "comprehension":
            step = driver.step_comprehension()
        elif name == "flow-counts":
            step = driver.step_flow_counts(record)
        else:
            number, surface = int(name[4]), name.split("-", 1)[1]
            step = driver.step_attempt(TASKS[number - 1], surface)
        record["steps"] = [existing for existing in record["steps"] if existing["name"] != name] + [step]
        record["metrics"] = {**record.get("metrics", {}), **run_metrics(record)}
        record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        record["transcripts"] = sorted(f"transcripts/{path.name}" for path in (driver.out / "transcripts").glob("*.jsonl"))
        record_file.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in step.items() if key not in ("notes",)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
