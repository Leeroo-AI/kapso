import os
import sys

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from ensemble_pipeline import run


if __name__ == "__main__":
    run("--debug" in sys.argv)
