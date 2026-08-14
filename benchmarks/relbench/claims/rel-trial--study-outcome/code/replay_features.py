# Imports

from __future__ import annotations

import numpy as np
import pandas as pd


# Replay labels

def generate_july_replay(db: object, start_year: int = 2008, end_year: int = 2018) -> pd.DataFrame:
    outcomes = db.table_dict["outcomes"].df[["id", "nct_id", "outcome_type"]]
    analyses = db.table_dict["outcome_analyses"].df.merge(outcomes, left_on="outcome_id", right_on="id", suffixes=("", "_outcome"))
    studies = db.table_dict["studies"].df[["nct_id", "start_date"]]
    qualifying = analyses[
        analyses["outcome_type"].eq("Primary")
        & analyses["p_value"].between(0, 1, inclusive="both")
        & (analyses["p_value_modifier"].isna() | analyses["p_value_modifier"].ne(">"))
    ][["nct_id", "date", "p_value"]].merge(studies, on="nct_id", how="left")
    rows = []
    for year in range(start_year, end_year + 1):
        timestamp = pd.Timestamp(year=year, month=7, day=1)
        current = qualifying[
            qualifying["start_date"].le(timestamp)
            & qualifying["date"].gt(timestamp)
            & qualifying["date"].le(timestamp + pd.Timedelta(days=365))
        ]
        labels = current.groupby("nct_id", as_index=False)["p_value"].min()
        labels["timestamp"] = timestamp
        labels["outcome"] = labels["p_value"].le(0.05).astype(np.int32)
        rows.append(labels[["timestamp", "nct_id", "outcome"]])
    return pd.concat(rows, ignore_index=True).sort_values(["timestamp", "nct_id"]).reset_index(drop=True)
