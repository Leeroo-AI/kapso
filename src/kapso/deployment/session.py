"""The deployment session (docs/plans/redesign/wp5-deploy.md, "The
session", "The engine's acceptance", "The workspace"): one session on a
copy of the solution, staged with the harness's deployment module and
the engine's target skills, that adapts the copy, deploys it with the
target's scripts, proves one call and writes the result file; then the
engine's own checks, one resume with the finding, and the record's
completion. `run_deployment` is what `Kapso.deploy()` calls."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import git

from kapso.core.prompt_loader import load_prompt, optional_block, render_prompt
from kapso.core.roles import deployment_role
from kapso.deployment.errors import DeployError
from kapso.deployment.handle import Software
from kapso.deployment.record import FAILED, RUNNING, DeploymentRecord, now, start_record
from kapso.deployment.targets import (
    AUTO, BUNDLED, FUNCTION, HTTP, RUN_KINDS, RUN_SCRIPT, SCRIPT, SCRIPTS_DIR, TargetSkill, readiness, resolve_target,
)
from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.harness.staging import check_harness_unchanged, host_for, open_skill_instruction, skills_dir, stage_harness

RESULT_FILE = "kapso_deployment.json"
ENV_FILE = ".env"
DEPLOYMENT_MODULE = "deployment"
DEPLOYMENT_PROMPT = "deployment/deployment_prompt.md"
RESUME_REASON = "deployment result"
# What the copy leaves behind: the campaign's repository and the engine's
# and the hosts' own directories; the staging mounts the statement and
# the prepared resources again for a harness build.
COPY_IGNORE = (
    ".git", ".kapso", ".claude", ".agents", ".codex", "kapso_task.md", "prepared",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv", "node_modules",
)
ALLOWED_TOOLS = ["Edit", "Read", "Write", "Bash", "Glob", "Grep"]
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# What no runtime's env file carries in a value: a line break, a
# carriage return, a NUL.
ENV_VALUE_FORBIDDEN = ("\n", "\r", "\0")
RUN_KEYS = {FUNCTION: ("module", "callable"), HTTP: ("url", "path"), SCRIPT: ()}
COMMITTER = git.Actor("Kapso", "kapso@leeroo.com")
# The result file's parse, checked at a process boundary: the interpreter
# exits non-zero with the parser's message on a file that is not JSON,
# and the engine's own read never raises on a session's sloppy output.
PARSE_CHECK = "import json, sys\njson.load(open(sys.argv[1], encoding='utf-8'))"


def check_env(env: Dict[str, str]) -> None:
    for name, value in env.items():
        if not ENV_NAME.fullmatch(str(name)):
            raise DeployError(f"env: {name!r} is not a variable name (letters, digits and underscores, not starting with a digit)")
        if any(character in str(value) for character in ENV_VALUE_FORBIDDEN):
            raise DeployError(f"env: the value of {name} holds a line break or a control character, which no runtime's env file carries")


def write_env_file(workspace: Path, env: Dict[str, str]) -> Optional[Path]:
    """The caller's variables in `.env` at the workspace root, one
    `NAME=VALUE` per line with the value verbatim (the format docker's
    `--env-file` and the engine's worker read): an existing file's other
    lines kept, the call's keys winning. The file, owner-only (it holds
    secrets), when the workspace holds one afterwards, the solution's own
    included; None when it holds none."""
    path = workspace / ENV_FILE
    if env:
        kept: List[str] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                key = line.split("=", 1)[0].strip()
                if not line.strip() or line.lstrip().startswith("#") or key not in env:
                    kept.append(line)
        path.write_text("\n".join(kept + [f"{key}={value}" for key, value in env.items()]) + "\n", encoding="utf-8")
    if not path.is_file():
        return None
    os.chmod(path, 0o600)
    return path


def refuse_dirty(campaign: Path) -> None:
    """What was evaluated is the delivered commit: tracked files that
    differ from it are refused by name."""
    status = git.Repo(campaign).git.status("--porcelain", "--untracked-files=no").strip()
    if status:
        names = sorted(line.split(maxsplit=1)[-1] for line in status.splitlines())
        raise DeployError(
            f"{campaign}: tracked files differ from the delivered commit ({', '.join(names)}); "
            f"restore the delivered tree (`git -C {campaign} checkout -- {' '.join(names)}`), "
            f"or build again and deploy what that delivers"
        )


def place_commit(campaign: Path, commit: str, workspace: Path) -> None:
    """The campaign's tracked files taken out of the copy and the commit's
    put in, under the same exclusions; the untracked data stays. Read
    through a private index, so the campaign's head, index and files are
    not touched: a campaign that delivered something stays at it."""
    repo = git.Repo(campaign)
    for name in repo.git.ls_files("-z").split("\0"):
        tracked = workspace / name
        if name and (tracked.is_file() or tracked.is_symlink()):
            tracked.unlink()
            for parent in tracked.parents:
                if parent == workspace or any(parent.iterdir()):
                    break
                parent.rmdir()
    with tempfile.TemporaryDirectory() as scratch:
        tree = Path(scratch) / "tree"
        with repo.git.custom_environment(GIT_INDEX_FILE=str(Path(scratch) / "index")):
            repo.git.read_tree(commit)
            repo.git.checkout_index("--all", f"--prefix={tree}/")
        for source in tree.rglob("*"):
            placed = workspace / source.relative_to(tree)
            if placed.is_symlink() or (placed.exists() and placed.is_dir() != (source.is_dir() and not source.is_symlink())):
                raise DeployError(
                    f"the campaign's untracked {placed.relative_to(workspace)} stands where "
                    f"commit {commit[:12]} has its own; move it and deploy again"
                )
        shutil.copytree(tree, workspace, ignore=shutil.ignore_patterns(*COPY_IGNORE), symlinks=True, dirs_exist_ok=True)


def make_workspace(campaign: Path, record: DeploymentRecord) -> Path:
    """The solution's working tree copied without the engine's and the
    hosts' directories, then its own repository with one commit, so a
    session's git commands stop here. A chosen attempt's tracked files
    then replace the campaign's, which is left where it was."""
    workspace = record.workspace
    shutil.copytree(campaign, workspace, ignore=shutil.ignore_patterns(*COPY_IGNORE), symlinks=True)
    if record.data["chosen_attempt"] is not None:
        place_commit(campaign, record.data["commit"], workspace)
    repo = git.Repo.init(workspace)
    repo.git.add("--all")
    repo.index.commit(
        f"deployment {record.number} of {campaign.name} at {record.data['commit'][:12]}",
        author=COMMITTER, committer=COMMITTER,
    )
    return workspace


def record_target(record: DeploymentRecord, target: TargetSkill) -> None:
    """The target's skill directory copied beside the record and its name
    written: from here `Software.open(campaign, n)` runs the target's
    scripts, after a failed session too."""
    if record.target_dir.exists():
        shutil.rmtree(record.target_dir)
    shutil.copytree(target.path, record.target_dir)
    record.update(target=target.name)


def read_result(workspace: Path) -> Tuple[Any, Optional[str]]:
    """The session's result file, or the finding to send back: missing,
    or not JSON (the parser's message from the process boundary)."""
    path = workspace / RESULT_FILE
    if not path.is_file():
        return None, f"no {RESULT_FILE} at the workspace root; write it as the prompt says"
    checked = subprocess.run([sys.executable, "-c", PARSE_CHECK, str(path)], capture_output=True, text=True)
    if checked.returncode != 0:
        lines = checked.stderr.strip().splitlines()
        return None, f"{RESULT_FILE} is not valid JSON ({lines[-1] if lines else f'exit {checked.returncode}'}); write it again as one JSON object"
    return json.loads(path.read_text(encoding="utf-8")), None


def _target_block(target: Optional[TargetSkill], offered: List[TargetSkill], host: str, staged_skills: Path,
                  offered_lines: List[Dict[str, str]], result_path: Path) -> str:
    if target is not None:
        return (
            f"The target is `{target.name}`. Open its skill with `{open_skill_instruction(host, target.name)}` and follow "
            f"it; its scripts are under $KAPSO_TARGET_DIR ({staged_skills / target.name})."
        )
    lines = ["Choose the target first. The targets available, each with how to open its skill and what this machine has for it:"]
    for candidate, line in zip(offered, offered_lines):
        lines.append(
            f"- {candidate.name} — {candidate.description} — open with `{open_skill_instruction(host, candidate.name)}` — {line['ready']}"
        )
    lines.append("")
    lines.append(
        f"Decide before changing anything: write `{result_path}` at once with `target` and `reasoning` alone (the "
        f"engine reads the choice even when this session fails, to reach what the target's deploy script started), "
        f"then `export KAPSO_TARGET_DIR={staged_skills}/<the target's name>` and follow its skill; complete the file at the end."
    )
    return "\n".join(lines)


def _prompt(solution: Any, record: DeploymentRecord, target: Optional[TargetSkill], offered: List[TargetSkill],
            host: str, workspace: Path, env: Dict[str, str], env_file: Optional[Path], options: Dict[str, str],
            guidance: str, offered_lines: List[Dict[str, str]]) -> str:
    if env:
        env_block = (
            f"The caller's variables, set at runtime by the engine and in `{ENV_FILE}` at the workspace root "
            f"(`$KAPSO_ENV_FILE`): {', '.join(sorted(env))}. Never copy their values into code, images or logs."
        )
    elif env_file is not None:
        env_block = (
            f"The caller passed no runtime variables; the solution's own `{ENV_FILE}` is at the workspace root "
            f"(`$KAPSO_ENV_FILE`) and is loaded on every runtime the engine manages. Never copy its values into code, images or logs."
        )
    else:
        env_block = "The caller passed no runtime variables."
    options_block = (
        "The call's options, for the target's workflow (each also in your environment as `KAPSO_OPTION_<NAME>`): "
        + "; ".join(f"{name} = {value}" for name, value in options.items())
        if options else "The call passed no options."
    )
    result_path = workspace / RESULT_FILE
    return render_prompt(load_prompt(DEPLOYMENT_PROMPT), {
        "objective": solution.brief.objective,
        "origin": "Delivered" if solution.chosen_attempt is None else "Chosen by the person, not delivered by the engine",
        "branch": solution.branch,
        "commit": solution.commit,
        "score": "unscored" if solution.score is None else f"{solution.score}",
        "workspace": str(workspace),
        "deployment_name": record.data["deployment_name"],
        "port": str(record.data["port"]),
        "number": str(record.number),
        "record_dir": str(record.directory),
        "interpreter": record.data["interpreter"],
        "result_path": str(result_path),
        "env_block": env_block,
        "options_block": options_block,
        "target_block": _target_block(target, offered, host, skills_dir(workspace, host), offered_lines, result_path),
        "domain_guidance": optional_block(guidance),
    })


def _variables(record: DeploymentRecord, workspace: Path, env_file: Optional[Path], options: Dict[str, str],
               target_dir: Optional[Path]) -> Dict[str, str]:
    variables = {
        "KAPSO_DEPLOYMENT_NAME": record.data["deployment_name"],
        "KAPSO_PORT": str(record.data["port"]),
        "KAPSO_NUMBER": str(record.number),
        "KAPSO_WORKSPACE": str(workspace),
        "KAPSO_RECORD_DIR": str(record.directory),
        "KAPSO_INTERPRETER": record.data["interpreter"],
    }
    if env_file is not None:
        variables["KAPSO_ENV_FILE"] = str(env_file)
    if target_dir is not None:
        variables["KAPSO_TARGET_DIR"] = str(target_dir)
    for name, value in options.items():
        variables[f"KAPSO_OPTION_{name.upper()}"] = str(value)
    return variables


def _ended_badly(result: Any) -> Optional[str]:
    """Why the session is a failed deployment, before any result is read:
    cut at its deadline, or failed to run."""
    metadata = result.metadata
    if metadata.get("deadline_exceeded") and not metadata.get("completed_before_kill"):
        return "the session was cut at its deadline"
    if not result.success:
        return f"the session failed: {result.error or 'no output'}"
    return None


def _shape_finding(data: Any, target: Optional[TargetSkill], offered: List[TargetSkill]) -> Optional[str]:
    """What is wrong with the result file's shape, in the session's terms."""
    if not isinstance(data, dict):
        return f"{RESULT_FILE} must hold a JSON object"
    for key in ("target", "summary"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            return f"{RESULT_FILE}: `{key}` must be a non-empty string"
    run = data.get("run")
    if not isinstance(run, dict) or run.get("kind") not in RUN_KINDS:
        return f"{RESULT_FILE}: `run.kind` must be one of {', '.join(RUN_KINDS)}"
    for key in RUN_KEYS[run["kind"]]:
        if not isinstance(run.get(key), str) or not run[key]:
            return f"{RESULT_FILE}: `run.{key}` must be a non-empty string for the {run['kind']} kind"
    if data.get("endpoint") is not None and not isinstance(data["endpoint"], str):
        return f"{RESULT_FILE}: `endpoint` must be a string or null"
    values = data.get("values", {})
    if not isinstance(values, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in values.items()):
        return f"{RESULT_FILE}: `values` must be a mapping of strings"
    proof = data.get("proof")
    if not isinstance(proof, dict) or "inputs" not in proof or "output" not in proof:
        return f"{RESULT_FILE}: `proof` must hold the `inputs` of the call you proved and its `output`"
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list) or not all(isinstance(a, str) for a in artifacts):
        return f"{RESULT_FILE}: `artifacts` must list the files you wrote or changed"
    if target is not None and data["target"] != target.name:
        return f"{RESULT_FILE} names target {data['target']}; the request was {target.name}"
    if target is None:
        if data["target"] not in {candidate.name for candidate in offered}:
            return f"{RESULT_FILE} names target {data['target']}, which was not offered"
        if not isinstance(data.get("reasoning"), str) or not data["reasoning"].strip():
            return f"{RESULT_FILE}: `reasoning` must say why you chose {data['target']}"
    return None


def _contract_finding(data: Dict[str, Any], target: TargetSkill, workspace: Path) -> Optional[str]:
    """What the result claims that the target or the workspace does not
    bear out: a run kind the target's skill does not end in, a function
    module outside the workspace, the script kind on a target with no
    `run.py`, an artifact outside the workspace or missing."""
    kind = data["run"]["kind"]
    if target.run_kinds is not None and kind not in target.run_kinds:
        return f"the {target.name} target's run kind is {' or '.join(target.run_kinds)}; the result names {kind}"
    if kind == FUNCTION:
        relative = Path(*data["run"]["module"].split("."))
        if not (workspace / relative).with_suffix(".py").is_file() and not (workspace / relative / "__init__.py").is_file():
            return f"run.module {data['run']['module']} is not a module of the workspace (no {relative}.py or {relative}/__init__.py there)"
    if kind == SCRIPT and not target.has_script(RUN_SCRIPT):
        return f"the {target.name} target has no {SCRIPTS_DIR}/{RUN_SCRIPT}, so the script kind cannot be called; name the kind its skill says"
    for name in data["artifacts"]:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            return f"artifact {name} must be a path inside the workspace, relative to it"
        if not (workspace / name).exists():
            return f"artifact {name} is not in the workspace"
    return None


def _check(record: DeploymentRecord, workspace: Path, target: Optional[TargetSkill], offered: List[TargetSkill],
           software: Software) -> Optional[str]:
    """The engine's acceptance after a completed session: the finding to
    send back, or None once the deployment is `running`. The state stays
    `deploying` until every check passed, so no other process calls a
    deployment that is still being proved."""
    data, finding = read_result(workspace)
    if finding is not None:
        return finding
    finding = _shape_finding(data, target, offered)
    if finding is not None:
        return finding
    chosen = target if target is not None else next(candidate for candidate in offered if candidate.name == data["target"])
    if record.data["target"] != chosen.name:
        record_target(record, chosen)   # under auto: known now, before the checks that may still fail
    finding = _contract_finding(data, chosen, workspace)
    if finding is not None:
        return finding
    record.update(
        reasoning=data.get("reasoning"), result=data, endpoint=data.get("endpoint"), values=dict(data.get("values", {})),
    )
    if not software._answers():
        return "the deployment does not answer its health check (scripts/health.sh does not exit 0, or the function does not import)"
    answer = software._call(data["proof"]["inputs"])
    if answer["status"] != "success":
        return f"the proof call failed: {answer['error']}"
    record.update(observed=answer["output"])
    record.set_state(RUNNING)
    return None


def _record_partial_target(record: DeploymentRecord, workspace: Path, offered: List[TargetSkill]) -> None:
    """A failed session under `auto`: the target its result names, if the
    file parses and names one offered, so `--stop` reaches what the
    target's deploy script started."""
    data, finding = read_result(workspace)
    if finding is not None or not isinstance(data, dict):
        return
    for candidate in offered:
        if candidate.name == data.get("target"):
            record_target(record, candidate)
            return


def run_deployment(
    solution: Any, *, config: Dict[str, Any], target: Optional[str], env: Optional[Dict[str, str]],
    options: Optional[Dict[str, str]], coding_agent: Optional[str], harness: Any,
    agent_factory: Any = CodingAgentFactory,
) -> Software:
    """One deployment of the solution, start to finish: the record, the
    workspace, the session, the acceptance; the running `Software` (the
    handle that proved it, its worker alive for the next call), or
    `DeployError` with the record's path."""
    campaign = Path(solution.campaign).resolve()
    block = config["deployment"]
    role = deployment_role(config, coding_agent)
    host = host_for(role.cli)
    chosen, offered = resolve_target(target, block["target"], harness)
    env = dict(env or {})
    options = {str(k): str(v) for k, v in (options or {}).items()}
    check_env(env)
    if solution.chosen_attempt is None:
        refuse_dirty(campaign)   # a chosen attempt's tracked files come from its commit
    if harness is not None:
        check_harness_unchanged(harness)   # before any record: a changed harness deploys nothing
    record = start_record(
        campaign, target_requested=target if target is not None else block["target"],
        host=f"{role.cli}:{role.model}", interpreter=sys.executable, env_names=sorted(env), options=options,
        commit=solution.commit, branch=solution.branch, chosen_attempt=solution.chosen_attempt,
        harness_built=None if solution.harness is None else {"name": solution.harness.name, "version": solution.harness.version, "hash": solution.harness.hash, "source": solution.harness.source},
        harness=None if harness is None else harness.ref_dict(),
        harness_root=None if harness is None else str(harness.path),
        port_base=int(block["port_base"]), port_span=int(block["port_span"]),
    )
    if chosen is not None:
        record_target(record, chosen)   # before the session: a failed one is still reachable
    workspace = make_workspace(campaign, record)
    env_file = write_env_file(workspace, env)
    # The engine's own targets are staged as its skills; a harness target
    # arrives with the harness's deployment module.
    staged_targets = [chosen] if chosen is not None else list(offered)
    staged = stage_harness(
        harness, DEPLOYMENT_MODULE, session_dir=workspace, campaign_dir=campaign,
        session_id=record.data["session"], host=host,
        engine_skills=[candidate.path for candidate in staged_targets if candidate.origin == BUNDLED],
    )
    offered_lines = [{"name": candidate.name, "ready": readiness(candidate)} for candidate in staged_targets]
    record.update(offered=offered_lines)
    target_dir = None if chosen is None else skills_dir(workspace, host) / chosen.name
    prompt = _prompt(solution, record, chosen, offered, host, workspace, env, env_file, options, staged.guidance, offered_lines)
    own = {
        "timeout": role.timeout,
        "env_overrides": _variables(record, workspace, env_file, options, target_dir),
        "allowed_tools": [*ALLOWED_TOOLS, *staged.allowed_tools],
        "streaming": True,
    }
    agent = agent_factory.create(agent_factory.session_config(role, own, staged))
    agent.initialize(str(workspace))
    started = time.monotonic()
    software = Software(record, call_timeout=block["call_timeout"], script_timeout=block["script_timeout"])

    def fail(reason: str) -> DeployError:
        agent.cleanup()
        software.close()
        if record.data["target"] is None:
            _record_partial_target(record, workspace, offered)
        record.update(finished=now())
        record.set_state(FAILED, error=reason)
        if record.data["target"] is None:
            way = (
                "its session named no target the engine could read, so there is nothing for `--stop` to run: "
                "end by hand what the session may have started"
            )
        else:
            way = f"`kapso deploy {campaign} --stop --number {record.number}` ends what its deploy.sh may have started"
        return DeployError(f"deployment {record.number} of {campaign} failed: {reason} (the record: {record.path}; {way})")

    result = agent.generate_code(prompt)
    bad = _ended_badly(result)
    if bad is not None:
        raise fail(bad)
    finding = _check(record, workspace, chosen, offered, software)
    if finding is not None:
        remaining = None if role.timeout is None else role.timeout - (time.monotonic() - started)
        if remaining is not None and remaining <= 0:
            raise fail(f"{finding}; no room left under the session's deadline for a resume")
        cli_session_id = result.metadata.get("cli_session_id")
        if not cli_session_id:
            raise fail(f"{finding}; the session recorded no handle to resume")
        software.close()   # the resumed session may rewrite the module: the next check starts a worker afresh
        result = agent.resume(
            cli_session_id,
            f"The engine checked the deployment and found: {finding}. Fix it and write {workspace / RESULT_FILE} again.",
            timeout_seconds=remaining, reason=RESUME_REASON,
        )
        bad = _ended_badly(result)
        if bad is not None:
            raise fail(f"{bad} (resumed for: {finding})")
        finding = _check(record, workspace, chosen, offered, software)
        if finding is not None:
            raise fail(f"after the resume: {finding}")
    agent.cleanup()
    record.update(finished=now())
    return software
