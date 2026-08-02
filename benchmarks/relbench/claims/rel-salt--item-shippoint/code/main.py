import os
import sys
import time

os.environ.setdefault("PYTHONWARNINGS", "ignore")

from graph_pipeline import run


if __name__ == "__main__":
    started = time.time()
    try:
        run("--debug" in sys.argv)
    except Exception as error:
        print(f"[fallback] {type(error).__name__}: {str(error).replace(chr(10), ' ')[:400]}")
        if not os.path.exists(os.path.join(os.environ["KAPSO_RUN_DATA_DIR"], "val_predictions.npy")):
            raise
    print(f"[phase] total elapsed={time.time() - started:.2f}s")
