"""`Software`: the handle on a deployment, built from its record
(docs/plans/redesign/wp5-deploy.md, "The call" and "Run kinds"). It runs
the deployed software by the record's run kind (a worker process for a
function, an HTTP call, the target's `scripts/run.py`), and stops, starts,
checks and reads it through the target's scripts with the deployment's
values in their environment, writing the state back into the record so
another process sees it. `Software.open(campaign, number)` is the same
handle from any later process. What the engine does to a deployment is
noted in `engine.log` beside the record, which `logs()` repeats."""

import concurrent.futures
import http.client
import json
import os
import select
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlsplit

import git

from kapso.core.config import load_config
from kapso.deployment.errors import DeployError
from kapso.deployment.record import DeploymentRecord, RUNNING, STOPPED, now, open_record
from kapso.deployment.targets import FUNCTION, HTTP, RUN_SCRIPT, SCRIPTS_DIR

START, STOP, HEALTH, LOGS = "start.sh", "stop.sh", "health.sh", "logs.sh"
SOURCE_REMOTE = "origin"
# A lifecycle script's stdout lines that write back into the record.
ENDPOINT_LINE = "KAPSO_ENDPOINT="
VALUE_LINE = "KAPSO_VALUE_"
WORKER_LOG = "worker.log"
ENGINE_LOG = "engine.log"
ENV_FILE = ".env"
# A dead worker's error is the end of its log: enough for a traceback.
_LOG_TAIL_LINES = 30


def source_remote(campaign: Path) -> Optional[str]:
    """Where the solution's code came from: the campaign repository's
    `origin`, followed through clones on this machine to the first URL
    that is not one (a campaign is often a clone of a working copy whose
    own origin is the host); None when the campaign has no origin."""
    where, url, seen = Path(campaign).resolve(), None, set()
    while (where / ".git").exists() and where not in seen:
        seen.add(where)
        remotes = {remote.name: remote.url for remote in git.Repo(where).remotes}
        if SOURCE_REMOTE not in remotes:
            break
        url = remotes[SOURCE_REMOTE]
        local = Path(url) if Path(url).is_absolute() else where / url   # git reads a relative URL from the repository
        if not local.is_dir():
            break
        where = local.resolve()
    return url


def provenance(record: DeploymentRecord) -> Dict[str, str]:
    """What a deployment was made from, as its session and its scripts
    see it: `KAPSO_SOURCE_REMOTE`, where the solution's code came from,
    and `KAPSO_VERIFICATION_DIR`, what verification wrote about the
    delivered attempt (the record's `verification`); each only when
    there is one."""
    variables: Dict[str, str] = {}
    remote = source_remote(record.campaign)
    if remote is not None:
        variables["KAPSO_SOURCE_REMOTE"] = remote
    if record.data["verification"] is not None:
        variables["KAPSO_VERIFICATION_DIR"] = str((record.directory / record.data["verification"]).resolve())
    return variables


def _end_group(process: subprocess.Popen) -> None:
    """The process ended with everything it started: every child the
    engine runs is its own session, so one signal reaches the group."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _communicate(process: subprocess.Popen, payload: Optional[str], deadline_seconds: float) -> Optional[Tuple[str, str]]:
    """The process's stdout and stderr once it ends, or None when it is
    still running at the deadline, killed then with its children. Pumped
    in a thread so a large output never blocks it, and bounded without
    an exception."""
    collected: Dict[str, str] = {}

    def pump() -> None:
        collected["out"], collected["err"] = process.communicate(payload)

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    thread.join(timeout=deadline_seconds)
    if thread.is_alive():
        _end_group(process)
        thread.join()
        return None
    return collected["out"], collected["err"]


class Software:
    """A deployment, running or not: `run(inputs)`, `stop()`, `start()`,
    `is_healthy()`, `logs()`, and the record's facts."""

    def __init__(self, record: DeploymentRecord, *, call_timeout: float, script_timeout: float):
        self._record = record
        self._call_timeout = float(call_timeout)
        self._script_timeout = float(script_timeout)
        self._worker: Optional[subprocess.Popen] = None

    @classmethod
    def open(cls, campaign: Union[str, Path], number: Optional[int] = None, config: Optional[str] = None) -> "Software":
        """The campaign's deployment of that number, or its highest; the
        timeouts from the config file (the packaged defaults without one)."""
        deployment = load_config(config)["deployment"]
        software = cls(
            open_record(Path(campaign), number),
            call_timeout=deployment["call_timeout"], script_timeout=deployment["script_timeout"],
        )
        software._check_files()
        return software

    # ------------------------------------------------------------- facts

    @property
    def number(self) -> int:
        return self._record.number

    @property
    def campaign(self) -> Path:
        return self._record.campaign

    @property
    def state(self) -> str:
        """Read from the record each time: a stop from another process shows."""
        self._record.reload()
        return self._record.state

    @property
    def target(self) -> Optional[str]:
        """The target's name; None while a session under `auto` has not chosen."""
        return self._record.data["target"]

    @property
    def endpoint(self) -> Optional[str]:
        return self._record.data["endpoint"]

    @property
    def workspace(self) -> Path:
        return self._record.workspace

    @property
    def record(self) -> Path:
        return self._record.path

    @property
    def target_dir(self) -> Path:
        """The target's skill directory copied beside the record: the scripts every call runs."""
        return self._record.target_dir

    @property
    def values(self) -> Dict[str, str]:
        return dict(self._record.data["values"])

    @property
    def run_kind(self) -> Optional[str]:
        """The result's run kind; None before a session wrote an acceptable result."""
        result = self._record.data["result"]
        return None if result is None else result["run"]["kind"]

    @property
    def proof(self) -> Optional[Dict[str, Any]]:
        """The call the session proved: what `run()` takes, and what it answered."""
        result = self._record.data["result"]
        return None if result is None else result["proof"]

    @property
    def observed(self) -> Any:
        """What the engine's own proof call answered at the acceptance."""
        return self._record.data["observed"]

    def __repr__(self) -> str:
        return f"Software(deployment {self.number} of {self.campaign}, {self.target}, {self.state})"

    # -------------------------------------------------------------- calls

    def run(self, inputs: Union[Dict[str, Any], str, bytes]) -> Dict[str, Any]:
        """One call: `{"status": "success", "output": …}` or
        `{"status": "error", "error": …}`; refused while the deployment is
        not running."""
        state = self.state
        if state != RUNNING:
            way_out = "start() it" if state == STOPPED else (self._record.data["error"] or "it never ran")
            raise DeployError(f"deployment {self.number} is {state}; {way_out}")
        return self._call(inputs)

    def run_batch(self, inputs_list: List[Any]) -> List[Dict[str, Any]]:
        return [self.run(inputs) for inputs in inputs_list]

    def __call__(self, inputs: Union[Dict[str, Any], str, bytes]) -> Dict[str, Any]:
        return self.run(inputs)

    def _call(self, inputs: Any) -> Dict[str, Any]:
        """One call by the record's run kind, the state not looked at: the
        acceptance proves a deployment this way before it is `running`."""
        kind = self.run_kind
        if kind is None:
            raise DeployError(f"deployment {self.number} recorded no result to call")
        if kind == FUNCTION:
            return self._run_function(inputs)
        if kind == HTTP:
            return self._run_http(inputs)
        if not self._script(RUN_SCRIPT):
            raise DeployError(f"the {self.target} target has no {SCRIPTS_DIR}/{RUN_SCRIPT}; see logs()")
        return self._run_script(inputs)

    def _run_http(self, inputs: Any) -> Dict[str, Any]:
        """`POST <url><path>`; a reply that is not 2xx, or no reply at all
        (refused, reset, timed out: the exception the request ended with,
        read off its future), is the call's error."""
        run = self._record.data["result"]["run"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._post, run["url"], run["path"], inputs)
        failure = future.exception()
        if failure is not None:
            return {"status": "error", "error": f"POST {run['url']}{run['path']}: {type(failure).__name__}: {failure}"}
        status, body = future.result()
        if 200 <= status < 300:
            return {"status": "success", "output": json.loads(body)}
        return {"status": "error", "error": f"{status} from {run['url']}{run['path']}: {body}"}

    def _post(self, base: str, path: str, inputs: Any) -> Tuple[int, str]:
        url = urlsplit(base)
        connection_class = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        connection = connection_class(url.hostname, url.port, timeout=self._call_timeout)
        connection.request(
            "POST", url.path.rstrip("/") + path, body=json.dumps(inputs), headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        connection.close()
        return response.status, body

    def _run_script(self, inputs: Any) -> Dict[str, Any]:
        process = subprocess.Popen(
            [self._record.data["interpreter"], str(self._script(RUN_SCRIPT))],
            cwd=self.workspace, env=self._environment(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
        ended = _communicate(process, json.dumps(inputs), self._call_timeout)
        if ended is None:
            return {"status": "error", "error": f"{RUN_SCRIPT} did not answer within {self._call_timeout:g} s and was killed"}
        out, err = ended
        if process.returncode != 0:
            return {"status": "error", "error": err.strip() or f"{RUN_SCRIPT} exited {process.returncode}"}
        return {"status": "success", "output": json.loads(out)}

    # ------------------------------------------------------------- worker

    def _worker_log(self) -> Path:
        return self._record.directory / WORKER_LOG

    def _ensure_worker(self) -> Optional[str]:
        """This handle's worker answering a ping, started when it has
        none; the reason (its log's tail) when it does not."""
        if self._worker is None or self._worker.poll() is not None:
            self._end_worker()
            run = self._record.data["result"]["run"]
            command = [
                self._record.data["interpreter"], "-m", "kapso.deployment.worker",
                str(self.workspace), run["module"], run["callable"],
            ]
            env_file = self.workspace / ENV_FILE
            if env_file.is_file():
                command.append(str(env_file))
            with self._worker_log().open("a", encoding="utf-8") as log:
                self._worker = subprocess.Popen(
                    command, cwd=self.workspace, env=self._environment(), stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=log, text=True, start_new_session=True,
                )
        reply = self._ask_worker({"ping": True})
        return None if reply == {"pong": True} else reply["error"]

    def _ask_worker(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """One request line to the worker and its reply line, bounded by
        the call timeout; a worker that ends (its function raised, its
        import failed) answers with the error from its log, which the
        closed reply channel announces."""
        worker = self._worker
        if worker.poll() is not None:
            self._end_worker()
            return {"error": self._worker_log_tail()}
        worker.stdin.write(json.dumps(request) + "\n")
        worker.stdin.flush()
        readable, _, _ = select.select([worker.stdout], [], [], self._call_timeout)
        if not readable:
            self._end_worker()
            return {"error": f"the function did not answer within {self._call_timeout:g} s; its worker was ended"}
        line = worker.stdout.readline()
        if line:
            return json.loads(line)
        self._end_worker()
        return {"error": self._worker_log_tail()}

    def _worker_log_tail(self) -> str:
        log = self._worker_log()
        lines = log.read_text(encoding="utf-8").splitlines() if log.is_file() else []
        return "\n".join(lines[-_LOG_TAIL_LINES:]) or "the worker ended without a message"

    def _end_worker(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        _end_group(worker)
        worker.stdin.close()
        worker.stdout.close()

    def _run_function(self, inputs: Any) -> Dict[str, Any]:
        failure = self._ensure_worker()
        if failure is not None:
            return {"status": "error", "error": failure}
        reply = self._ask_worker({"inputs": inputs})
        if "error" in reply:
            return {"status": "error", "error": reply["error"]}
        return {"status": "success", "output": reply["output"]}

    # ---------------------------------------------------------- lifecycle

    def stop(self) -> List[str]:
        """The target's stop script (its failure noted, never raised), this
        handle's worker ended, the state `stopped` (a failed deployment
        keeps its error); the lines noted, as `logs()` repeats them."""
        noted = []
        if self._script(STOP):
            code, out, err = self._lifecycle(STOP)
            noted.extend(self._script_lines(STOP, out))
            if code != 0:
                noted.append(self._note(f"{STOP} exited {code}: {err.strip()}"))
        else:
            noted.append(self._note(self._nothing_to("stop")))
        self._end_worker()
        self._record.set_state(STOPPED, error=self._record.data["error"])
        noted.append(self._note(f"stopped deployment {self.number}"))
        return noted

    def start(self) -> List[str]:
        """The target's start script (its failure raised), the worker
        started, the state `running`; the lines noted."""
        noted = []
        if self._script(START):
            code, out, err = self._lifecycle(START)
            if code != 0:
                raise DeployError(f"deployment {self.number}: {START} exited {code}: {err.strip()}")
            noted.extend(self._script_lines(START, out))
        else:
            noted.append(self._note(self._nothing_to("start")))
        if self.run_kind == FUNCTION:
            failure = self._ensure_worker()
            if failure is not None:
                raise DeployError(f"deployment {self.number}: the function's worker did not start: {failure}")
        self._record.set_state(RUNNING)
        noted.append(self._note(f"started deployment {self.number}"))
        return noted

    def is_healthy(self) -> bool:
        """Running by the record, and answering."""
        return self.state == RUNNING and self._answers()

    def _answers(self) -> bool:
        """The target's health script exits 0 (nothing to ask without one)
        and, for a function, this handle's worker answers a ping."""
        if self._script(HEALTH):
            code, _out, _err = self._lifecycle(HEALTH)
            if code != 0:
                return False
        if self.run_kind == FUNCTION:
            return self._ensure_worker() is None
        return True

    def logs(self) -> str:
        """The engine's own lines, the target's logs script's output, and a
        function's worker log."""
        parts = [self._engine_log().read_text(encoding="utf-8").rstrip("\n")] if self._engine_log().is_file() else []
        if self._script(LOGS):
            code, out, err = self._lifecycle(LOGS)
            parts.append(out.rstrip("\n") if code == 0 else f"{LOGS} exited {code}: {err.strip()}")
        if self.run_kind == FUNCTION and self._worker_log().is_file():
            parts.append(self._worker_log().read_text(encoding="utf-8").rstrip("\n"))
        return "\n".join(part for part in parts if part)

    def close(self) -> None:
        """This handle's worker ended, the deployment's state untouched: a
        function deployment keeps running for the next handle, which
        starts a worker of its own."""
        self._end_worker()

    def __enter__(self) -> "Software":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    # ------------------------------------------------------------ scripts

    def _script(self, name: str) -> Optional[Path]:
        path = self._record.target_dir / SCRIPTS_DIR / name
        return path if path.is_file() else None

    def _nothing_to(self, verb: str) -> str:
        if self.target is None:
            return (
                f"deployment {self.number} recorded no target (its session chose none the engine could read): "
                f"nothing to {verb}; end by hand what the session may have started"
            )
        return f"the {self.target} target has nothing to {verb}"

    def _engine_log(self) -> Path:
        return self._record.directory / ENGINE_LOG

    def _note(self, line: str) -> str:
        """One line of what the engine did, kept beside the record."""
        with self._engine_log().open("a", encoding="utf-8") as log:
            log.write(f"{now()} {line}\n")
        return line

    def _environment(self) -> Dict[str, str]:
        """The deployment's values, as every script and the worker see them."""
        data = self._record.data
        env = {
            **os.environ,
            "KAPSO_DEPLOYMENT_NAME": data["deployment_name"],
            "KAPSO_PORT": str(data["port"]),
            "KAPSO_NUMBER": str(data["number"]),
            "KAPSO_WORKSPACE": str(self.workspace),
            "KAPSO_RECORD_DIR": str(self._record.directory),
            "KAPSO_TARGET_DIR": str(self._record.target_dir),
            "KAPSO_INTERPRETER": data["interpreter"],
            **provenance(self._record),
        }
        if data["endpoint"]:
            env["KAPSO_ENDPOINT"] = data["endpoint"]
        if (self.workspace / ENV_FILE).is_file():
            env["KAPSO_ENV_FILE"] = str(self.workspace / ENV_FILE)
        if data["harness_root"]:
            env["KAPSO_HARNESS_ROOT"] = data["harness_root"]
        for name, value in data["options"].items():
            env[f"KAPSO_OPTION_{name.upper()}"] = str(value)
        for name, value in data["values"].items():
            env[f"{VALUE_LINE}{name.upper()}"] = str(value)
        return env

    def _lifecycle(self, name: str) -> Tuple[int, str, str]:
        """One lifecycle script from the record's target directory, the
        workspace as its working directory; its update lines written into
        the record when it succeeds. A script still running at the
        timeout is killed with what it started, and counts as failed."""
        process = subprocess.Popen(
            ["bash", str(self._script(name))], cwd=self.workspace, env=self._environment(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
        ended = _communicate(process, None, self._script_timeout)
        if ended is None:
            return 1, "", f"{name} did not end within {self._script_timeout:g} s and was killed, with what it started"
        out, err = ended
        if process.returncode == 0:
            self._take_updates(out)
        return process.returncode, out, err

    def _script_lines(self, name: str, out: str) -> List[str]:
        """What the script said, its update lines aside, noted."""
        return [
            self._note(f"{name}: {line}") for line in out.splitlines()
            if line.strip() and not line.startswith((ENDPOINT_LINE, VALUE_LINE))
        ]

    def _take_updates(self, out: str) -> None:
        """`KAPSO_ENDPOINT=` and `KAPSO_VALUE_<KEY>=` lines into the record;
        a new endpoint is where an `http` deployment's calls go."""
        endpoint = None
        values: Dict[str, str] = {}
        for line in out.splitlines():
            if line.startswith(ENDPOINT_LINE):
                endpoint = line[len(ENDPOINT_LINE):].rstrip("\r")
            elif line.startswith(VALUE_LINE) and "=" in line:
                key, value = line[len(VALUE_LINE):].split("=", 1)
                values[key.lower()] = value.rstrip("\r")
        if endpoint is None and not values:
            return
        self._record.reload()
        changes: Dict[str, Any] = {}
        if endpoint is not None:
            changes["endpoint"] = endpoint
            result = self._record.data["result"]
            if result is not None and result["run"]["kind"] == HTTP:
                changes["result"] = dict(result, run=dict(result["run"], url=endpoint))
        if values:
            changes["values"] = {**self._record.data["values"], **values}
        self._record.update(**changes)

    def _check_files(self) -> None:
        if not self.workspace.is_dir():
            raise DeployError(f"deployment {self.number} of {self.campaign}: its workspace is gone ({self.workspace})")
        if self.target is not None and not self.target_dir.is_dir():
            raise DeployError(f"deployment {self.number} of {self.campaign}: its target directory is gone ({self.target_dir})")
        root = self._record.data["harness_root"]
        if root and not Path(root).is_dir():
            raise DeployError(f"deployment {self.number} of {self.campaign}: its harness is gone ({root})")
