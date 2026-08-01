import sys

from document_drift_model import run


if __name__ == "__main__":
    run("--debug" in sys.argv)
