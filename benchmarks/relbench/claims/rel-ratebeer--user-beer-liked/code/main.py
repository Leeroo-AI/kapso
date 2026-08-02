import os
import sys
from pathlib import Path

from relbench.tasks import get_task

from ratebeer_recommender import run


def main():
    debug = "--debug" in sys.argv
    task = get_task(os.environ["RELBENCH_DATASET"], os.environ["RELBENCH_TASK"], download=False)
    output = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "./output_data_generic_exp_0"))
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(os.environ.get("KAPSO_SHARED_CACHE_DIR", "./shared_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    run(debug, task, output, cache)


if __name__ == "__main__":
    main()
