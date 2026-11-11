"""The engine's own errors at the command line (wp6-observability.md,
change 10): their messages are written to be read alone and name their
fix, so the CLI prints the message, on stderr, and exits 1; any other
exception is a bug and keeps its traceback. The detached child installs
the same hook through `main()`, so its log shows the message."""
import sys
from typing import Type

from kapso.core.config import ConfigError
from kapso.core.git_reference import GitReferenceError
from kapso.core.preflight import PreflightError
from kapso.deployment.errors import DeployError
from kapso.execution.campaign_compare import CampaignCompareError
from kapso.execution.evaluation_integrity import EvaluationIntegrityError
from kapso.execution.experiment_workspace.experiment_workspace import WorkspaceCheckoutError
from kapso.execution.inbox import CampaignRunningError, InboxOpenError
from kapso.execution.run_checkpoint import RunCheckpointError
from kapso.experience.ingest import IngestError
from kapso.experience.reference import ExperienceReferenceError
from kapso.experience.sources import ExperienceError
from kapso.gated_mcp.presets import GateCapabilityError
from kapso.handles import BriefError, SolutionError
from kapso.harness.export import HarnessExportError
from kapso.harness.load import HarnessError
from kapso.harness.manifest import ManifestError
from kapso.harness.reference import HarnessReferenceError

ENGINE_ERRORS = (
    BriefError,
    SolutionError,
    ConfigError,
    HarnessError,
    HarnessExportError,
    HarnessReferenceError,
    ManifestError,
    ExperienceError,
    ExperienceReferenceError,
    IngestError,
    EvaluationIntegrityError,
    PreflightError,
    GitReferenceError,
    CampaignCompareError,
    CampaignRunningError,
    InboxOpenError,
    RunCheckpointError,
    WorkspaceCheckoutError,
    GateCapabilityError,
    DeployError,
)


def report_engine_error(exc_type: Type[BaseException], exc: BaseException, traceback) -> None:
    """The CLI's excepthook: the message alone for the engine's own error
    classes (however many lines it spans; the preflight's is a report),
    the traceback for everything else."""
    if isinstance(exc, ENGINE_ERRORS):
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.__excepthook__(exc_type, exc, traceback)


def install_error_hook() -> None:
    sys.excepthook = report_engine_error
