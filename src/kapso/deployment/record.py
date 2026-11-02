"""The deployment record: `<campaign>/.kapso/deployments/<n>/deployment.json`
(docs/plans/redesign/wp5-deploy.md, "The deployment record" and
"Identity"). Written with `state: deploying` before the session starts,
completed at its end, and the whole handle afterwards: `Software.open`
reads nothing else. Paths inside it are relative to its directory, so a
moved campaign still opens. Every write replaces the file whole and
atomically, and every change re-reads it first, so two handles on one
deployment never undo each other's fields."""

import hashlib
import json
import os
import re
import socket
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from kapso.deployment.errors import DeployError

DEPLOYMENTS_DIR = Path(".kapso") / "deployments"
RECORD_FILE = "deployment.json"
WORKSPACE_DIR = "workspace"
TARGET_DIR = "target"
DEPLOYING, RUNNING, STOPPED, FAILED = "deploying", "running", "stopped", "failed"
STATES = (DEPLOYING, RUNNING, STOPPED, FAILED)
# Two deploys in one process mint under the lock; two processes rely on
# the atomic mkdir.
_MINT_LOCK = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def campaign_hash(campaign: Path) -> str:
    """Six hex digits of the campaign's absolute path: what keeps two
    campaigns of one name apart in a deployment's name and port."""
    return hashlib.sha1(str(Path(campaign).resolve()).encode()).hexdigest()[:6]


def deployment_name(campaign: Path, number: int) -> str:
    """`<slug>-<hash6>-<n>`: the directory's name slugged, the hash, the
    number; what the targets name their containers, apps and services
    after (`kapso-<name>`)."""
    slug = re.sub(r"[^a-z0-9]+", "-", Path(campaign).resolve().name.lower()).strip("-")[:24] or "solution"
    return f"{slug}-{campaign_hash(campaign)}-{number}"


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def allocate_port(campaign: Path, number: int, *, base: int, span: int) -> int:
    """The first port nothing listens on from `base + (hash + number) %
    span`, upwards within the span and wrapping; refused when the whole
    span is taken. Probed, not held: the target's `deploy.sh` binds it
    later and fails naming it if something took it since."""
    start = (int(campaign_hash(campaign), 16) + number) % span
    for offset in range(span):
        port = base + (start + offset) % span
        if _port_is_free(port):
            return port
    raise DeployError(f"no free port in {base}..{base + span - 1} for the deployment; set deployment.port_base or port_span")


def _read(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise DeployError(f"no deployment record at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("state") not in STATES:
        raise DeployError(f"{path}: not a deployment record")
    return data


@dataclass
class DeploymentRecord:
    directory: Path   # absolute: the workspace and the target paths resolve against it from any cwd
    data: Dict[str, Any]

    @property
    def path(self) -> Path:
        return self.directory / RECORD_FILE

    @property
    def number(self) -> int:
        return int(self.data["number"])

    @property
    def campaign(self) -> Path:
        """The campaign the deployment belongs to: `.kapso/deployments/<n>` up."""
        return self.directory.parents[2]

    @property
    def state(self) -> str:
        return self.data["state"]

    @property
    def workspace(self) -> Path:
        return self.directory / self.data["workspace"]

    @property
    def target_dir(self) -> Path:
        return self.directory / self.data["target_dir"]

    def write(self) -> None:
        """The file replaced whole: a reader never sees a half-written record."""
        staging = self.directory / (RECORD_FILE + ".tmp")
        staging.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, self.path)

    def reload(self) -> None:
        """What is on disk now: a stop from another process shows."""
        self.data = _read(self.path)

    def update(self, **changes: Any) -> None:
        """These fields changed on the record as it is on disk now, so a
        handle that read it earlier changes nothing but what it means to."""
        self.reload()
        self.data.update(changes)
        self.write()

    def set_state(self, state: str, *, error: Optional[str] = None) -> None:
        if state not in STATES:
            raise ValueError(f"unknown deployment state {state!r}; the states are {', '.join(STATES)}")
        self.update(state=state, state_at=now(), error=error)


def _mint(deployments: Path) -> Path:
    """The next number's directory, made atomically: a number another
    process took in the meantime is skipped."""
    deployments.mkdir(parents=True, exist_ok=True)
    taken = [int(child.name) for child in deployments.iterdir() if child.name.isdigit()]
    number = max(taken, default=0) + 1
    while True:
        candidate = deployments / str(number)
        if not candidate.exists():
            candidate.mkdir(exist_ok=False)
            return candidate
        number += 1


def start_record(
    campaign: Path, *, target_requested: str, host: str, interpreter: str, env_names: List[str],
    options: Dict[str, str], commit: str, branch: Optional[str], chosen_attempt: Optional[Dict[str, Any]], harness_built: Optional[Dict[str, str]],
    harness: Optional[Dict[str, str]], harness_root: Optional[str], port_base: int, port_span: int,
) -> DeploymentRecord:
    """A deployment's record at its first second: the number minted, the
    identity allocated, the state `deploying`; the session has not run."""
    campaign = Path(campaign).resolve()
    with _MINT_LOCK:
        directory = _mint(campaign / DEPLOYMENTS_DIR)
        number = int(directory.name)
        port = allocate_port(campaign, number, base=port_base, span=port_span)
    record = DeploymentRecord(directory, {
        "number": number,
        "state": DEPLOYING,
        "state_at": now(),
        "target_requested": target_requested,
        "target": None,
        "reasoning": None,
        "offered": [],
        "commit": commit,
        "branch": branch,
        # Set when a person named the attempt (`Solution.chosen_attempt`);
        # null is the engine's deliverable.
        "chosen_attempt": chosen_attempt,
        "harness_built": harness_built,
        "harness": harness,
        "harness_root": harness_root,
        "host": host,
        "interpreter": interpreter,
        "deployment_name": deployment_name(campaign, number),
        "port": port,
        "workspace": WORKSPACE_DIR,
        "target_dir": TARGET_DIR,
        "env_names": list(env_names),
        "options": dict(options),
        "result": None,
        "observed": None,
        "endpoint": None,
        "values": {},
        "session": f"deployment/{number}",
        "started": now(),
        "finished": None,
        "error": None,
    })
    record.write()
    return record


def read_record(directory: Path) -> DeploymentRecord:
    directory = Path(directory).resolve()
    return DeploymentRecord(directory, _read(directory / RECORD_FILE))


def records(campaign: Path) -> List[DeploymentRecord]:
    """Every deployment of the campaign, by number."""
    deployments = Path(campaign).resolve() / DEPLOYMENTS_DIR
    if not deployments.is_dir():
        return []
    numbered = sorted((int(child.name), child) for child in deployments.iterdir() if child.name.isdigit() and (child / RECORD_FILE).is_file())
    return [read_record(directory) for _number, directory in numbered]


def open_record(campaign: Path, number: Optional[int] = None) -> DeploymentRecord:
    """The numbered deployment, or the highest number; refused when the
    campaign has none, naming the directory looked at."""
    deployments = Path(campaign).resolve() / DEPLOYMENTS_DIR
    if number is not None:
        directory = deployments / str(number)
        if not (directory / RECORD_FILE).is_file():
            raise DeployError(f"{campaign} has no deployment {number} (looked at {directory})")
        return read_record(directory)
    found = records(campaign)
    if not found:
        raise DeployError(f"{campaign} has no deployment (looked at {deployments})")
    return found[-1]
