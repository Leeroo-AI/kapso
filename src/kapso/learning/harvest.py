# Campaign harvest — the evolve->learn bridge for live campaigns (P1.G).
#
# Design: learn-from-trajectories-design.md §3.4 (save_trajectory IS the
# harvest step) + §3.4.1 (the bundle-contract additions: workspace .kapso
# files, living documents, sessions, and the ideation/selector artifacts that
# previously died in temp worktrees). This module owns the framework-side
# gather conventions (where evolve keeps its workspace artifacts); the caller
# — a benchmark's campaign driver — supplies its own paths (work dir, log,
# living documents) and stays the authority on what its shared space holds.

from pathlib import Path
from typing import Dict, Optional

from kapso.learning.trajectory_store import TrajectoryStore, save_trajectory

# Workspace-side artifacts (evolve's conventions, framework-owned):
# required by the strict contract...
WORKSPACE_KAPSO_FILES = ("lens_plan_history.jsonl", "experiment_history.json")
# ...and gathered when present (session forensics; the ideation candidate
# pool + selector artifacts — the §3.4.1 bundle-contract addition).
WORKSPACE_KAPSO_DIRS = ("sessions", "ideation")


def harvest_campaign(
    store: TrajectoryStore,
    trajectory_id: str,
    work_dir: str,
    campaign_log: str,
    workspace_dir: str,
    living_documents: Optional[Dict[str, str]] = None,
    work_dir_exclude: tuple = (),
    kapso_commit: Optional[str] = None,
    bank_head: Optional[str] = None,
    upload: Optional[bool] = None,
) -> str:
    """Harvest a finished campaign into the store under the strict contract.

    Gathers the work dir (minus `work_dir_exclude`), the campaign log, the
    workspace `.kapso` artifacts (files required, dirs when present), and the
    caller's living documents (bundle-relative name -> source path; required
    strict parts among them are enforced by the contract check, so a missing
    features_history.md raises — no thin saves).
    """
    kapso_dir = Path(workspace_dir).expanduser() / ".kapso"
    if not kapso_dir.is_dir():
        raise FileNotFoundError(f"workspace {workspace_dir} has no .kapso directory")

    extra: Dict[str, str] = dict(living_documents or {})
    for name in WORKSPACE_KAPSO_FILES:
        extra[name] = str(kapso_dir / name)
    for name in WORKSPACE_KAPSO_DIRS:
        source = kapso_dir / name
        if source.is_dir():
            extra[name] = str(source)

    return save_trajectory(
        store,
        trajectory_id,
        work_dir=work_dir,
        campaign_log=campaign_log,
        extra_files=extra,
        work_dir_exclude=work_dir_exclude,
        contract="strict",
        kapso_commit=kapso_commit,
        bank_head=bank_head,
        upload=upload,
    )
