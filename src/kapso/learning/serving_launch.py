# Campaign-launch serving (design §5.3, P5): pin the bank, compile the push
# brief, stage the pull-tool parameters — all frame-side, before any session
# exists. The network is never on the campaign path: v1 serves from the
# durable local home (`learning.bank.local_path`); the remote arrives with
# D3 and changes only where the home syncs from, never this function.
#
# Everything staged here lands inside the campaign work dir (.kapso/serving/)
# so the harvested trajectory carries the exact served state: the pinned
# checkout, the push record, and the sessions' pull log.

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from kapso.learning.bank import Bank
from kapso.learning.retriever import compile_brief


def prepare_campaign_serving(
    config: Dict[str, Any],
    task_coords: Dict[str, str],
    work_dir,
) -> Optional[Dict[str, Any]]:
    """Stage serving for one campaign launch; None when serving is off.

    Returns {brief, bank_head, record_path, bank_serving} — `brief` is the
    stamped markdown for the problem context, `bank_serving` is the KAPSO_*
    env mapping the bank gate resolves on (ideation + implementation
    sessions only; the feedback judge never receives it).
    """
    serving_config = config["learning"]["serving"]
    if not serving_config["enabled"]:
        return None
    home = Path(config["learning"]["bank"]["local_path"]).expanduser()
    if not home.exists():
        raise FileNotFoundError(
            f"serving is enabled but the bank home {home} does not exist"
        )

    serve_dir = Path(work_dir).expanduser() / ".kapso" / "serving"
    checkout = serve_dir / "bank"
    if checkout.exists():
        shutil.rmtree(checkout)
    serve_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--quiet", str(home), str(checkout)],
        check=True, capture_output=True,
    )
    bank_head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    result = compile_brief(
        Bank(str(checkout)), task_coords, bank_head,
        config["learning"]["retriever"],
    )
    record_path = serve_dir / "serving-record.yaml"
    with open(record_path, "w") as handle:
        yaml.safe_dump(result["record"], handle, sort_keys=False)

    stamp = (
        f"Served from the knowledge bank at head `{bank_head}` "
        f"({len(result['record']['served'])} cards; pinned for this whole "
        f"campaign). Cards are measured practice, not constraints; cite "
        f"load-bearing use as [card:<name>]."
    )
    return {
        "brief": stamp + "\n\n" + result["brief"],
        "bank_head": bank_head,
        "record_path": str(record_path),
        "bank_serving": {
            "KAPSO_BANK_DIR": str(checkout),
            "KAPSO_BANK_HEAD": bank_head,
            "KAPSO_SERVING_PULL_LOG": str(serve_dir / "serving-pull.jsonl"),
            "KAPSO_TASK_FAMILY": task_coords["family"],
            **({"KAPSO_TASK_DATASET": task_coords["dataset"]}
               if task_coords.get("dataset") else {}),
        },
    }
