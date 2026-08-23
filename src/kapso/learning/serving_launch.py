# Campaign-launch serving (serving-agentic-redesign.md, v2): pin the bank,
# compile the intro, stage the tool parameters — all frame-side, before any
# session exists. The network is never on the campaign path: serving reads
# the durable local home (`learning.bank.local_path`).
#
# Everything staged here lands inside the campaign work dir (.kapso/serving/)
# so the harvested trajectory carries the exact served state: the pinned
# checkout, the launch record, and the sessions' pull log (the exposure
# ladder's source).

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from kapso.learning.bank import Bank
from kapso.learning.retriever import compile_intro


def prepare_campaign_serving(
    config: Dict[str, Any],
    task_coords: Dict[str, str],
    work_dir,
) -> Optional[Dict[str, Any]]:
    """Stage serving for one campaign launch; None when serving is off.

    Returns {intro, bank_head, record_path, bank_serving} — `intro` is the
    knowledge-bank introduction appended after the static context notes,
    `bank_serving` is the KAPSO_* env mapping the bank gate resolves on
    (ideation + implementation sessions only; the feedback judge never
    receives it).
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

    result = compile_intro(Bank(str(checkout)), task_coords, bank_head)
    record_path = serve_dir / "serving-record.yaml"
    with open(record_path, "w") as handle:
        yaml.safe_dump(result["record"], handle, sort_keys=False)

    return {
        "intro": result["intro"],
        "bank_head": bank_head,
        "record_path": str(record_path),
        "bank_serving": {
            "KAPSO_BANK_DIR": str(checkout),
            "KAPSO_BANK_HEAD": bank_head,
            "KAPSO_SERVING_PULL_LOG": str(serve_dir / "serving-pull.jsonl"),
            "KAPSO_TASK_FAMILY": task_coords["family"],
            "KAPSO_PROBE_BUDGET": str(
                config["learning"]["retriever"]["probe_budget"]
            ),
            **({"KAPSO_TASK_DATASET": task_coords["dataset"]}
               if task_coords.get("dataset") else {}),
        },
    }
