# gcp_ephemeral — the codify run's placement target (CD§3, D8).
#
# The dev box is not a run host: each evaluation provisions a preemptible
# instance, bootstraps the standard env, ships the staged workspace, runs
# the registered evaluation under the iteration timeout, pulls the outcome,
# and tears down UNCONDITIONALLY — teardown is sequenced before any raise,
# never guarded by exception handling (Rule 2). A preemption is a rerun: a
# codify iteration is short and idempotent.

import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict

CODIFY_EVAL_CMD = (
    "cd /tmp/codify-workspace && {bootstrap} python replay/eval.py "
    "> replay-stdout.txt 2>&1; echo exit=$?"
)


class GcpEphemeralExecutor:
    """Run one registered evaluation on an ephemeral GCP instance."""

    def __init__(self, config: Dict[str, Any]):
        codify = config["learning"]["codify"]
        self.gcp = codify["gcp"]
        self.machine_type_gpu = codify["machine_type"]
        self.machine_type_cpu = self.gcp["cpu_machine_type"]
        self.timeout = codify["iteration_timeout_minutes"] * 60

    def _gcloud(self, args, timeout=300):
        return subprocess.run(
            ["gcloud", "--project", self.gcp["project"], *args],
            capture_output=True, text=True, timeout=timeout,
        )

    def run_evaluation(self, workspace: Path, gpu: bool = False) -> Path:
        workspace = Path(workspace)
        name = f"kapso-codify-{uuid.uuid4().hex[:8]}"
        zone = self.gcp["zone"]
        machine_type = self.machine_type_gpu if gpu else self.machine_type_cpu
        create_args = [
            "compute", "instances", "create", name,
            "--zone", zone, "--machine-type", machine_type,
            "--provisioning-model", "SPOT",
            "--image-family", self.gcp["image_family"],
            "--image-project", self.gcp["image_project"],
            "--boot-disk-size", self.gcp["boot_disk_gb"],
        ]
        if gpu:
            create_args += ["--accelerator", self.gcp["accelerator"],
                            "--maintenance-policy", "TERMINATE"]
        created = self._gcloud(create_args, timeout=600)
        if created.returncode != 0:
            raise RuntimeError(
                f"gcp_ephemeral: provisioning {name} failed: {created.stderr}"
            )

        # From here on, every path — success or failure — reaches teardown
        # before any raise (sequential rc capture, no exception guards).
        failure = ""
        pushed = self._gcloud([
            "compute", "scp", "--recurse", "--zone", zone,
            str(workspace), f"{name}:/tmp/codify-workspace",
        ], timeout=900)
        if pushed.returncode != 0:
            failure = f"workspace push failed: {pushed.stderr}"
        if not failure:
            ran = self._gcloud([
                "compute", "ssh", name, "--zone", zone, "--command",
                CODIFY_EVAL_CMD.format(bootstrap=self.gcp["bootstrap"]),
            ], timeout=self.timeout)
            if ran.returncode != 0:
                failure = f"evaluation ssh failed: {ran.stderr}"
        if not failure:
            for artifact in ("outcome.yaml", "replay-stdout.txt"):
                pulled = self._gcloud([
                    "compute", "scp", "--zone", zone,
                    f"{name}:/tmp/codify-workspace/{artifact}",
                    str(workspace / artifact),
                ], timeout=300)
                if pulled.returncode != 0 and artifact == "outcome.yaml":
                    # The evaluation may have died before writing it — the
                    # gates will name every miss on the empty outcome.
                    (workspace / "outcome.yaml").write_text("{}\n")
            # Gate artifacts are produced on the instance under outputs/ —
            # pull them back so the artifact gates judge real files. A
            # missing directory is not an error here: the gates name every
            # absent artifact.
            self._gcloud([
                "compute", "scp", "--recurse", "--zone", zone,
                f"{name}:/tmp/codify-workspace/outputs",
                str(workspace),
            ], timeout=600)

        deleted = self._gcloud([
            "compute", "instances", "delete", name, "--zone", zone, "--quiet",
        ], timeout=600)
        if deleted.returncode != 0:
            raise RuntimeError(
                f"gcp_ephemeral: TEARDOWN FAILED for {name} — delete it "
                f"manually: {deleted.stderr}"
            )
        if failure:
            raise RuntimeError(f"gcp_ephemeral: {failure}")
        outcome_path = workspace / "outcome.yaml"
        if not outcome_path.is_file():
            outcome_path.write_text("{}\n")
        return outcome_path
