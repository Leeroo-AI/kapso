import os
import sys
from pathlib import Path

import numpy as np

from item_sales_pipeline import run_pipeline
from kapso_datasets.common import load_task


def run_rolling(ctx) -> None:
    test_df = ctx.test.df
    transactions = ctx.db.table_dict["transactions"].df
    predictions = np.zeros(len(test_df), dtype=np.float64)
    for timestamp in test_df["timestamp"].drop_duplicates():
        mask = (transactions["t_dat"] > timestamp - np.timedelta64(7, "D")) & (
            transactions["t_dat"] <= timestamp
        )
        sales = transactions.loc[mask].groupby("article_id", sort=False)["price"].sum()
        rows = test_df["timestamp"].to_numpy() == np.datetime64(timestamp)
        predictions[rows] = (
            test_df.loc[rows, "article_id"].map(sales).fillna(0).to_numpy(np.float64)
        )
    out = Path(os.environ["KAPSO_RUN_DATA_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "test_predictions.npy", predictions)
    print(f"[pipeline] rolling fallback test{predictions.shape}")


def main() -> None:
    ctx = load_task()
    if len(ctx.val) == 0:
        run_rolling(ctx)
        return
    run_pipeline(ctx, debug="--debug" in sys.argv)


if __name__ == "__main__":
    main()
