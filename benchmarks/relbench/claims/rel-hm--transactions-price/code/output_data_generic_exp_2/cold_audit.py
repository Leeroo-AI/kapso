import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from price_panel import (
    DataAccess,
    FeatureBuilder,
    LAUNCH_HISTORY_FEATURES,
    PricePanelPipeline,
    numeric_frame,
)

p = PricePanelPipeline(False)
a = DataAccess(("train",), 11)
w = a.weekly_panel(p.shared / "lane2_weekly_tasklabels_train_v1.parquet")
b = FeatureBuilder(w, p.article)
all_launch = a.launch_examples(pd.Timestamp("2020-09-07"), 60).merge(
    b.meta, on="article_id", how="left"
)
launch = all_launch[
    pd.to_datetime(all_launch.first_date) >= pd.Timestamp("2020-09-07") - pd.Timedelta(weeks=26)
].copy()
launch_features = LAUNCH_HISTORY_FEATURES
pieces = []
for origin, group in launch.groupby("launch_origin", sort=True):
    frame = group.copy()
    frame = p.attach_snapshot_activity(frame, a, pd.Timestamp(origin))
    frame = b.attach_recent_priors(frame, pd.Timestamp(origin), weeks=4)
    frame = p.attach_launch_history_priors(frame, all_launch, pd.Timestamp(origin))
    frame["week_of_year"] = int(pd.Timestamp(origin).isocalendar().week)
    pieces.append(frame)
d = p.launch_feature_frame(pd.concat(pieces, ignore_index=True))
order = [
    "product_channel_state",
    "product_state",
    "product_type_no_state",
    "department_no_state",
    "section_no_state",
    "garment_group_no_state",
    "index_group_no_state",
    "global_recent_state",
]
base = pd.Series(np.nan, index=d.index)
for column in order:
    base = base.fillna(d[column])
d["cascade_state"] = base
launch_order = [
    "product_channel_launch_state",
    "product_launch_state",
    "product_channel_state",
    "product_state",
    "product_type_no_launch_state",
    "department_no_launch_state",
    "product_type_no_state",
    "department_no_state",
    "section_no_launch_state",
    "garment_group_no_launch_state",
    "index_group_no_launch_state",
    "global_recent_state",
]
launch_base = pd.Series(np.nan, index=d.index)
for column in launch_order:
    launch_base = launch_base.fillna(d[column])
d["launch_cascade_state"] = launch_base
d["dual_cascade_state"] = 0.5 * (d["cascade_state"] + d["launch_cascade_state"])
columns = p.launch_feature_columns()
snapshot_columns = [
    "origin_day_panel_count_log",
    "origin_day_article_count_log",
    "origin_day_channel_share",
    "metadata_article_age_weeks",
    "metadata_total_activity_log",
]
no_snapshot_columns = [column for column in columns if column not in snapshot_columns]
residual_columns = columns
wide_columns = residual_columns
launch_cascade_columns = wide_columns + ["launch_cascade_state"]
dual_cascade_columns = wide_columns + ["dual_cascade_state"]
weeks = sorted(pd.to_datetime(d.launch_origin).unique())
folds = [
    (weeks[:13], weeks[14:18]),
    (weeks[:17], weeks[18:22]),
    (weeks[:21], weeks[22:26]),
]
results = []
for fold, (train_weeks, valid_weeks) in enumerate(folds):
    train = d[pd.to_datetime(d.launch_origin).isin(train_weeks)]
    valid = d[pd.to_datetime(d.launch_origin).isin(valid_weeks)]
    recency = (
        pd.Timestamp(max(train_weeks))
        + pd.Timedelta(days=7)
        - pd.to_datetime(train.launch_origin)
    ).dt.days / 7
    weight = train.target_count.to_numpy() * np.exp(-recency.to_numpy() / 13)
    parameters = dict(
        objective="regression_l2",
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=30,
        colsample_bytree=0.85,
        reg_lambda=10,
        random_state=1559 + fold,
        n_jobs=11,
        verbosity=-1,
    )
    raw = lgb.LGBMRegressor(**parameters).fit(
        numeric_frame(train, columns), train.target_mean, sample_weight=weight
    )
    raw_cascade = lgb.LGBMRegressor(**parameters).fit(
        numeric_frame(train, residual_columns), train.target_mean, sample_weight=weight
    )
    residual = lgb.LGBMRegressor(**parameters).fit(
        numeric_frame(train, residual_columns),
        train.target_mean - train.cascade_state,
        sample_weight=weight,
    )
    no_snapshot_residual = lgb.LGBMRegressor(**parameters).fit(
        numeric_frame(train, no_snapshot_columns),
        train.target_mean - train.cascade_state,
        sample_weight=weight,
    )
    wide_residual = lgb.LGBMRegressor(**parameters).fit(
        numeric_frame(train, wide_columns),
        train.target_mean - train.cascade_state,
        sample_weight=weight,
    )
    launch_cascade_residual = lgb.LGBMRegressor(**parameters).fit(
        numeric_frame(train, launch_cascade_columns),
        train.target_mean - train.launch_cascade_state,
        sample_weight=weight,
    )
    dual_cascade_residual = lgb.LGBMRegressor(**parameters).fit(
        numeric_frame(train, dual_cascade_columns),
        train.target_mean - train.dual_cascade_state,
        sample_weight=weight,
    )
    raw_prediction = raw.predict(numeric_frame(valid, columns))
    raw_cascade_prediction = raw_cascade.predict(numeric_frame(valid, residual_columns))
    cascade_prediction = valid.cascade_state.to_numpy()
    residual_prediction = cascade_prediction + residual.predict(
        numeric_frame(valid, residual_columns)
    )
    no_snapshot_prediction = cascade_prediction + no_snapshot_residual.predict(
        numeric_frame(valid, no_snapshot_columns)
    )
    wide_residual_prediction = cascade_prediction + wide_residual.predict(
        numeric_frame(valid, wide_columns)
    )
    launch_cascade_prediction = valid.launch_cascade_state.to_numpy()
    launch_cascade_residual_prediction = launch_cascade_prediction + launch_cascade_residual.predict(
        numeric_frame(valid, launch_cascade_columns)
    )
    dual_residual_prediction = 0.5 * (
        wide_residual_prediction + launch_cascade_residual_prediction
    )
    dual_cascade_prediction = valid.dual_cascade_state.to_numpy()
    dual_cascade_single_prediction = dual_cascade_prediction + dual_cascade_residual.predict(
        numeric_frame(valid, dual_cascade_columns)
    )
    target = valid.target_mean.to_numpy()
    valid_weight = valid.target_count.to_numpy()
    baseline_delta = launch_cascade_prediction - cascade_prediction
    current_alpha = float(
        np.clip(
            np.average((target - wide_residual_prediction) * baseline_delta, weights=valid_weight)
            / max(np.average(baseline_delta * baseline_delta, weights=valid_weight), 1e-12),
            -1,
            1,
        )
    )
    corrected_current_prediction = wide_residual_prediction + current_alpha * baseline_delta
    def mse(prediction):
        return float(np.average((target - prediction) ** 2, weights=valid_weight))

    delta = raw_prediction - cascade_prediction
    blend = float(
        np.clip(
            np.average((target - cascade_prediction) * delta, weights=valid_weight)
            / max(np.average(delta * delta, weights=valid_weight), 1e-12),
            0,
            1,
        )
    )
    blend_prediction = cascade_prediction + blend * delta
    results.append(
        {
            "fold": fold,
            "train": len(train),
            "valid": len(valid),
            "cascade_mse": mse(cascade_prediction),
            "raw_mse": mse(raw_prediction),
            "raw_cascade_mse": mse(raw_cascade_prediction),
            "residual_mse": mse(residual_prediction),
            "no_snapshot_mse": mse(no_snapshot_prediction),
            "wide_residual_mse": mse(wide_residual_prediction),
            "launch_cascade_mse": mse(launch_cascade_prediction),
            "launch_cascade_residual_mse": mse(launch_cascade_residual_prediction),
            "dual_residual_mse": mse(dual_residual_prediction),
            "dual_cascade_single_mse": mse(dual_cascade_single_prediction),
            "current_alpha": current_alpha,
            "corrected_current_mse": mse(corrected_current_prediction),
            "blend_weight": blend,
            "blend_mse": mse(blend_prediction),
        }
    )
summary = {
    "folds": results,
    "mean": {
        key: float(np.mean([row[key] for row in results]))
        for key in [
            "cascade_mse",
            "raw_mse",
            "raw_cascade_mse",
            "residual_mse",
            "no_snapshot_mse",
            "wide_residual_mse",
            "launch_cascade_mse",
            "launch_cascade_residual_mse",
            "dual_residual_mse",
            "dual_cascade_single_mse",
            "current_alpha",
            "corrected_current_mse",
            "blend_mse",
            "blend_weight",
        ]
    },
}
print(json.dumps(summary, indent=2))
a.close()
