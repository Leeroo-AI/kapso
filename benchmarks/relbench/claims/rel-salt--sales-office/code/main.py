import sys

from sales_office_pipeline import run


if __name__ == "__main__":
    run("--debug" in sys.argv)
