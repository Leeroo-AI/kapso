import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from kapso_datasets.common import is_debug, run_data_dir, shared_cache_dir
from sales_ranker import SalesGroupPipeline


def main() -> None:
    started = time.time()
    debug = is_debug()
    output = run_data_dir()
    cache = shared_cache_dir()
    pipeline = SalesGroupPipeline(output, cache, debug)
    diagnostics = pipeline.run()
    with (output / "metrics.json").open("w") as handle:
        json.dump(diagnostics, handle, indent=2)
    elapsed = time.time() - started
    print(f"[complete] elapsed_seconds={elapsed:.1f} debug={debug}")


if __name__ == "__main__":
    main()
