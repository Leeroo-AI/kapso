from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from kapso_datasets.common import is_debug, load_task
from relbench_solution import run


# Entrypoint

def main() -> None:
    started = time.time()
    debug = is_debug()
    print(f"[main] start debug={debug}")
    context = load_task()
    print(f"[main] database loaded elapsed={time.time() - started:.1f}s")
    run(context, debug)
    print(f"[main] finished elapsed={time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
