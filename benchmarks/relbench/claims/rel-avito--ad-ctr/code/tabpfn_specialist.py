import gc
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor

os.environ.setdefault(
    "TABPFN_MODEL_CACHE_DIR",
    str(Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / "tabpfn_model_cache_v2"),
)

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion


TABPFN_FIELDS = (
    "ad_category",
    "is_context",
    "category_level",
    "parent_category",
    "subcategory",
    "log_price",
    "price_abs_band",
    "category_price_percentile",
    "category_price_band",
    "category_log_price_delta",
    "category_price_median",
    "title_char_length",
    "title_token_count",
    "title_digit_share",
    "title_cyrillic_share",
    "title_latin_share",
    "title_punctuation_share",
    "title_group_size",
    "title_svd_0",
    "title_svd_1",
    "title_svd_2",
    "title_svd_3",
    "title_svd_4",
    "title_svd_5",
    "title_svd_6",
    "title_svd_7",
    "title_svd_8",
    "title_svd_9",
    "title_svd_10",
    "title_svd_11",
    "origin_day",
    "origin_weekday_sin",
    "origin_weekday_cos",
    "future_weekday_5",
    "future_weekday_6",
    "n_all_6h",
    "n_labeled_6h",
    "clicks_6h",
    "ctr_official_6h",
    "hist_mean_6h",
    "position_mean_6h",
    "n_all_1d",
    "n_labeled_1d",
    "clicks_1d",
    "ctr_official_1d",
    "hist_mean_1d",
    "position_mean_1d",
    "n_all_2d",
    "n_labeled_2d",
    "clicks_2d",
    "ctr_official_2d",
    "hist_mean_2d",
    "position_mean_2d",
    "n_all_4d",
    "n_labeled_4d",
    "clicks_4d",
    "ctr_official_4d",
    "hist_mean_4d",
    "position_mean_4d",
    "n_all_8d",
    "n_labeled_8d",
    "clicks_8d",
    "ctr_official_8d",
    "hist_mean_8d",
    "position_mean_8d",
    "n_all_all",
    "n_labeled_all",
    "clicks_all",
    "ctr_official_all",
    "hist_mean_all",
    "position_mean_all",
    "hist_std_4d",
    "hist_last_4d",
    "hist_weighted_4d",
    "position_std_4d",
    "rank_1_share_4d",
    "rank_2_4_share_4d",
    "rank_5_8_share_4d",
    "hist_std_all",
    "hist_last_all",
    "hist_weighted_all",
    "position_std_all",
    "rank_1_share_all",
    "rank_2_4_share_all",
    "rank_5_8_share_all",
    "n_all_prev4",
    "ctr_official_prev4",
    "volume_trend_1d_8d",
    "volume_trend_4d_prev4",
    "ctr_trend_1d_8d",
    "ctr_trend_4d_prev4",
    "hist_trend_1d_all",
    "position_trend_1d_all",
    "first_seen_age_days",
    "last_impression_age_days",
    "last_click_age_days",
    "unique_users_all",
    "unique_ips_all",
    "logged_share_all",
    "query_share_all",
    "category_mismatch_all",
    "audience_locations_all",
    "audience_regions_all",
    "audience_cities_all",
    "agent_diversity_all",
    "os_diversity_all",
    "device_diversity_all",
    "family_diversity_all",
    "active_hours_all",
    "eb_global_20",
    "eb_category_20",
    "eb_crp_20",
    "eb_audience_20",
    "eb_title_20",
    "eb_ad_20",
    "visit_cp_events",
    "visit_cr_events",
    "phone_cp_events",
    "phone_cr_events",
    "phone_visit_ratio_cp",
)


class TabPFNSpecialist:
    bands = ("0", "1-10", "11-100", "101-1000", "1000+")
    band_edges = (-1, 0, 10, 100, 1000, np.inf)
    fold_dates = tuple(pd.date_range("2015-04-30", "2015-05-04", freq="D"))
    champion_iterations = {"raw": 408, "log": 408, "cat": 423}
    champion_weights = {"raw": 0.6, "log": 0.2, "cat": 0.2}
    expected_hashes = {
        "val": "cf5c2326c3914caf26cc6e4926f182fb004280ccf4f2eb44b26c16c233e1c181",
        "test": "acf6280841a741b8fe2a7f7debf25456428b5c533384a35c581d97371c2bbc4e",
    }

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.debug = pipeline.debug
        self.seed = pipeline.seed
        self.threads = pipeline.threads
        self.device = f"cuda:{int(os.environ.get('CUDA_DEVICE', '0'))}"
        self.context_cap = 500 if self.debug else 10000
        self.cache_root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / "lane1_tabpfn_v2_specialist"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.bank_root = (
            Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
            / "banked_run_0001_generic_exp_5_lane1"
        )
        self.feature_fields = []
        self.oof = pd.DataFrame()
        self.selection_metrics = {}
        self.context_records = []
        self.final_records = {}
        self.failures = []
        self.bank_hashes = self.verify_bank()
        if len(TABPFN_FIELDS) > 120:
            raise RuntimeError(f"TabPFN field cap exceeded: {len(TABPFN_FIELDS)}")
        if not torch.cuda.is_available():
            self.failures.append("CUDA unavailable; pure champion fallback")
        self.register_static_artifacts()

    def verify_bank(self) -> dict:
        hashes = {}
        expected_sizes = {"val": 1766, "test": 1816}
        for split, expected_hash in self.expected_hashes.items():
            path = self.bank_root / f"{split}_predictions.npy"
            if not path.exists():
                raise FileNotFoundError(f"banked run_0001 artifact missing: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_hash:
                raise RuntimeError(f"banked {split} hash mismatch: {digest}")
            values = np.load(path, allow_pickle=False)
            if values.shape != (expected_sizes[split],) or not np.isfinite(values).all():
                raise RuntimeError(f"banked {split} prediction contract mismatch")
            hashes[split] = digest
        print(
            f"[bank] verified val={hashes['val'][:12]} test={hashes['test'][:12]} "
            f"path={self.bank_root.name}"
        )
        return hashes

    def register_static_artifacts(self) -> None:
        for split in ("val", "test"):
            self.pipeline.register_artifact(
                self.bank_root / f"{split}_predictions.npy",
                f"Byte-verified run_0001 {split} champion predictions and fallback",
                f"rel-avito-ad-ctr-run0001-banked-{split}-generic-exp5-lane1",
            )
        checkpoint = Path(os.environ["TABPFN_MODEL_CACHE_DIR"]) / "tabpfn-v2-regressor.ckpt"
        if checkpoint.exists():
            self.pipeline.register_artifact(
                checkpoint,
                "TabPFN v2 regression checkpoint used for local CUDA inference",
                "tabpfn-v2-regressor-checkpoint",
            )

    def bank(self, split: str) -> np.ndarray:
        return np.asarray(
            np.load(self.bank_root / f"{split}_predictions.npy", allow_pickle=False),
            dtype=np.float64,
        )

    def history_band(self, frame: pd.DataFrame) -> pd.Series:
        return pd.cut(
            frame["n_all_all"].fillna(0),
            self.band_edges,
            labels=self.bands,
        ).astype(str)

    def tabpfn_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        fields = [field for field in TABPFN_FIELDS if field in frame.columns]
        if not self.feature_fields:
            self.feature_fields = fields
        if len(fields) > 120:
            raise RuntimeError(f"TabPFN field cap exceeded after routing: {len(fields)}")
        result = frame[fields].copy()
        for field in fields:
            result[field] = pd.to_numeric(result[field], errors="coerce")
        return result.replace([np.inf, -np.inf], np.nan).astype("float32")

    def balanced_take(
        self,
        frame: pd.DataFrame,
        cap: int,
        group_columns: list[str],
    ) -> pd.DataFrame:
        if len(frame) <= cap:
            return frame.copy()
        ordered = frame.copy()
        ordered["_priority"] = pd.util.hash_pandas_object(
            ordered[["AdID", "timestamp"]],
            index=False,
        ).to_numpy(dtype="uint64")
        ordered["_group_rank"] = (
            ordered.sort_values(["_priority", "AdID"])
            .groupby(group_columns, observed=True)
            .cumcount()
        )
        ordered = ordered.sort_values(
            ["_group_rank", *group_columns, "_priority", "AdID"]
        ).head(cap)
        return ordered.drop(columns=["_priority", "_group_rank"])

    def build_context(self, eligible: pd.DataFrame) -> pd.DataFrame:
        frame = eligible.sort_values(["timestamp", "AdID"]).reset_index(drop=True).copy()
        frame["_history_band"] = self.history_band(frame)
        low = frame[frame["_history_band"].isin(["0", "1-10"])]
        if len(low) >= self.context_cap:
            selected = self.balanced_take(
                low,
                self.context_cap,
                ["timestamp", "_history_band"],
            )
        else:
            warm = frame[~frame.index.isin(low.index)]
            anchors = self.balanced_take(
                warm,
                self.context_cap - len(low),
                ["timestamp"],
            )
            selected = pd.concat([low, anchors], ignore_index=True)
        return (
            selected.drop(columns=["_history_band"])
            .sort_values(["timestamp", "AdID"])
            .reset_index(drop=True)
        )

    def create_regressor(self):
        if not torch.cuda.is_available():
            raise RuntimeError("TabPFN requires the configured CUDA device")
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V2,
            device=self.device,
            n_estimators=8,
            fit_mode="fit_with_cache",
            show_progress_bar=False,
            random_state=self.seed,
        )

    def fit_specialists(
        self,
        context: pd.DataFrame,
        predict_frame: pd.DataFrame,
        stage: str,
    ) -> dict:
        x_fit = self.tabpfn_matrix(context)
        x_predict = self.tabpfn_matrix(predict_frame)
        y = np.clip(context["target"].to_numpy(dtype=float), 2e-4, 1.0)
        output = {}
        started = time.time()
        model = self.create_regressor()
        model.fit(x_fit, y)
        raw = model.predict(
            x_predict,
            output_type="main",
            quantiles=[0.25, 0.75],
        )
        output["raw"] = np.clip(np.asarray(raw["median"], dtype=float), 2e-4, 1.0)
        output["raw_q25"] = np.clip(np.asarray(raw["quantiles"][0], dtype=float), 2e-4, 1.0)
        output["raw_q75"] = np.clip(np.asarray(raw["quantiles"][1], dtype=float), 2e-4, 1.0)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = self.create_regressor()
        logit_y = np.log(
            np.clip(y, 2e-4, 1 - 2e-4)
            / (1 - np.clip(y, 2e-4, 1 - 2e-4))
        )
        model.fit(x_fit, logit_y)
        logit = model.predict(
            x_predict,
            output_type="main",
            quantiles=[0.25, 0.75],
        )
        for name, values in (
            ("logit", logit["median"]),
            ("logit_q25", logit["quantiles"][0]),
            ("logit_q75", logit["quantiles"][1]),
        ):
            values = np.asarray(values, dtype=float)
            output[name] = np.clip(1 / (1 + np.exp(-values)), 2e-4, 1.0)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        elapsed = time.time() - started
        record = {
            "stage": stage,
            "eligible_rows": int(len(context)),
            "predict_rows": int(len(predict_frame)),
            "features": int(x_fit.shape[1]),
            "seconds": float(elapsed),
        }
        self.context_records.append(record)
        print(
            f"[tabpfn] stage={stage} context={len(context)} predict={len(predict_frame)} "
            f"features={x_fit.shape[1]} seconds={elapsed:.2f}"
        )
        return output

    def fit_champion(
        self,
        fit: pd.DataFrame,
        predict_frame: pd.DataFrame,
    ) -> np.ndarray:
        fit = fit.sort_values(["timestamp", "AdID"]).reset_index(drop=True)
        columns, categorical = self.pipeline.model_columns(fit, 20)
        x_fit = self.pipeline.numerical_matrix(fit, columns)
        x_predict = self.pipeline.numerical_matrix(predict_frame, columns)
        y = fit["target"].to_numpy(dtype=float)
        raw = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=self.champion_iterations["raw"],
            learning_rate=0.025,
            num_leaves=47,
            min_child_samples=40,
            colsample_bytree=0.75,
            subsample=0.8,
            subsample_freq=1,
            reg_lambda=5,
            random_state=self.seed,
            n_jobs=self.threads,
            verbosity=-1,
        )
        log_model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=self.champion_iterations["log"],
            learning_rate=0.025,
            num_leaves=47,
            min_child_samples=40,
            colsample_bytree=0.75,
            subsample=0.8,
            subsample_freq=1,
            reg_lambda=5,
            random_state=self.seed + 1,
            n_jobs=self.threads,
            verbosity=-1,
        )
        callbacks = [lgb.log_evaluation(0)]
        raw.fit(x_fit, y, callbacks=callbacks)
        log_model.fit(
            x_fit,
            np.log(np.maximum(y, 2e-4)),
            callbacks=callbacks,
        )
        x_fit_cat = self.pipeline.cat_matrix(fit, columns, categorical)
        x_predict_cat = self.pipeline.cat_matrix(predict_frame, columns, categorical)
        cat = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            boosting_type="Ordered",
            depth=7,
            learning_rate=0.035,
            iterations=self.champion_iterations["cat"],
            l2_leaf_reg=8,
            random_seed=self.seed + 2,
            has_time=True,
            allow_writing_files=False,
            verbose=False,
            thread_count=self.threads,
        )
        cat.fit(x_fit_cat, y, cat_features=categorical)
        prediction = (
            self.champion_weights["raw"] * raw.predict(x_predict)
            + self.champion_weights["log"] * np.exp(log_model.predict(x_predict))
            + self.champion_weights["cat"] * cat.predict(x_predict_cat)
        )
        return np.clip(np.asarray(prediction, dtype=float), 2e-4, 1.0)

    def build_oof(self, train: pd.DataFrame) -> pd.DataFrame:
        records = []
        for fold_date in self.fold_dates:
            started = time.time()
            valid = train[train["timestamp"] == fold_date].copy()
            eligible = train[
                train["timestamp"] + pd.Timedelta(days=4) <= fold_date
            ].copy()
            if len(valid) == 0 or len(eligible) == 0:
                raise RuntimeError(f"empty forward fold at {fold_date.date()}")
            champion = self.fit_champion(eligible, valid)
            context = self.build_context(eligible)
            try:
                specialist = self.fit_specialists(
                    context,
                    valid,
                    f"oof_{fold_date.date()}",
                )
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                self.failures.append(f"{fold_date.date()} {message}")
                specialist = {
                    "raw": champion.copy(),
                    "raw_q25": champion.copy(),
                    "raw_q75": champion.copy(),
                    "logit": champion.copy(),
                    "logit_q25": champion.copy(),
                    "logit_q75": champion.copy(),
                }
            record = valid[
                ["AdID", "timestamp", "target", "n_all_all", "ad_category"]
            ].copy()
            record["fold"] = str(fold_date.date())
            record["champion"] = champion
            for name, values in specialist.items():
                record[name] = values
            records.append(record)
            print(
                f"[forward_fold] date={fold_date.date()} eligible={len(eligible)} "
                f"valid={len(valid)} total_seconds={time.time() - started:.2f}"
            )
        return pd.concat(records, ignore_index=True)

    def fold_summary(self, oof: pd.DataFrame, prediction: np.ndarray) -> dict:
        values = oof[["fold", "target"]].copy()
        values["error"] = np.abs(values["target"].to_numpy(dtype=float) - prediction)
        fold_mae = values.groupby("fold", observed=True)["error"].mean()
        return {
            "mean": float(fold_mae.mean()),
            "worst": float(fold_mae.max()),
            "se": float(fold_mae.std(ddof=1) / math.sqrt(len(fold_mae))),
            "folds": {str(key): float(value) for key, value in fold_mae.items()},
        }

    def band_mae(
        self,
        oof: pd.DataFrame,
        prediction: np.ndarray,
        band: str,
        excluded_fold: str | None = None,
    ) -> float:
        mask = oof["history_band"].eq(band)
        if excluded_fold is not None:
            mask &= oof["fold"].ne(excluded_fold)
        subset = oof.loc[mask, ["fold", "target"]].copy()
        if len(subset) == 0:
            return math.inf
        subset["prediction"] = prediction[mask.to_numpy()]
        fold_mae = subset.assign(
            error=np.abs(subset["target"] - subset["prediction"])
        ).groupby("fold", observed=True)["error"].mean()
        return float(fold_mae.mean())

    def candidate_prediction(
        self,
        oof: pd.DataFrame,
        head: str,
        weight: float,
    ) -> np.ndarray:
        champion = oof["champion"].to_numpy(dtype=float)
        if head == "champion" or weight == 0:
            return champion
        return np.clip(
            (1 - weight) * champion + weight * oof[head].to_numpy(dtype=float),
            2e-4,
            1.0,
        )

    def choose_band(
        self,
        oof: pd.DataFrame,
        band: str,
        excluded_fold: str | None,
    ) -> dict:
        choices = [("champion", 0.0)]
        for head in ("raw", "logit"):
            for weight in (0.25, 0.5, 0.75, 1.0):
                choices.append((head, weight))
        scored = []
        for order, (head, weight) in enumerate(choices):
            prediction = self.candidate_prediction(oof, head, weight)
            score = self.band_mae(oof, prediction, band, excluded_fold)
            scored.append((score, order, head, weight))
        score, _, head, weight = min(scored)
        return {"head": head, "weight": float(weight), "mae": float(score)}

    def bootstrap_probability(
        self,
        oof: pd.DataFrame,
        baseline: np.ndarray,
        candidate: np.ndarray,
        offset: int,
    ) -> float:
        values = oof[["fold", "target"]].copy()
        target = values["target"].to_numpy(dtype=float)
        values["delta"] = np.abs(target - candidate) - np.abs(target - baseline)
        delta = values.groupby("fold", observed=True)["delta"].mean().to_numpy()
        rng = np.random.default_rng(self.seed + offset)
        samples = rng.integers(0, len(delta), size=(5000, len(delta)))
        return float((delta[samples].mean(axis=1) < 0).mean())

    def select_bands(self, oof: pd.DataFrame) -> dict:
        oof = oof.copy()
        oof["history_band"] = self.history_band(oof)
        baseline = oof["champion"].to_numpy(dtype=float)
        selected_meta = baseline.copy()
        baseline_summary = self.fold_summary(oof, baseline)
        records = {}
        frozen = {}
        for offset, band in enumerate(self.bands):
            proposed = selected_meta.copy()
            crossfit = {}
            band_mask = oof["history_band"].eq(band).to_numpy()
            for fold in sorted(oof["fold"].unique()):
                choice = self.choose_band(oof, band, fold)
                crossfit[fold] = choice
                held = band_mask & oof["fold"].eq(fold).to_numpy()
                option = self.candidate_prediction(
                    oof,
                    choice["head"],
                    choice["weight"],
                )
                proposed[held] = option[held]
            band_base = self.band_mae(oof, baseline, band)
            band_candidate = self.band_mae(oof, proposed, band)
            summary = self.fold_summary(oof, proposed)
            probability = self.bootstrap_probability(
                oof,
                baseline,
                proposed,
                offset,
            )
            accepted = bool(
                band_candidate < band_base
                and summary["mean"] < baseline_summary["mean"]
                and summary["worst"] < baseline_summary["worst"]
                and probability >= 0.9
            )
            final_choice = (
                self.choose_band(oof, band, None)
                if accepted
                else {"head": "champion", "weight": 0.0, "mae": band_base}
            )
            if final_choice["weight"] == 0:
                accepted = False
            if accepted:
                selected_meta = proposed
                frozen[band] = {
                    "head": final_choice["head"],
                    "weight": float(final_choice["weight"]),
                }
            else:
                frozen[band] = {"head": "champion", "weight": 0.0}
            records[band] = {
                "count": int(band_mask.sum()),
                "champion_mae": float(band_base),
                "crossfit_mae": float(band_candidate),
                "bootstrap_probability": probability,
                "accepted": accepted,
                "crossfit_choices": crossfit,
                "frozen_choice": frozen[band],
                "overall": summary,
            }
        selected_summary = self.fold_summary(oof, selected_meta)
        target = oof["target"].to_numpy(dtype=float)
        specialist_diagnostics = {}
        for head in ("raw", "logit"):
            error_a = baseline - target
            error_b = oof[head].to_numpy(dtype=float) - target
            specialist_diagnostics[head] = {
                "mae": float(np.mean(np.abs(error_b))),
                "error_correlation": float(np.corrcoef(error_a, error_b)[0, 1]),
                "absolute_error_covariance": float(
                    np.cov(np.abs(error_a), np.abs(error_b), ddof=1)[0, 1]
                ),
                "mean_absolute_disagreement": float(
                    np.mean(np.abs(oof[head].to_numpy(dtype=float) - baseline))
                ),
                "q25_q75_coverage": float(
                    np.mean(
                        (target >= oof[f"{head}_q25"].to_numpy(dtype=float))
                        & (target <= oof[f"{head}_q75"].to_numpy(dtype=float))
                    )
                ),
                "q25_q75_width": float(
                    np.mean(
                        oof[f"{head}_q75"].to_numpy(dtype=float)
                        - oof[f"{head}_q25"].to_numpy(dtype=float)
                    )
                ),
            }
        self.oof = oof
        self.selection_metrics = {
            "champion": baseline_summary,
            "meta_crossfit": selected_summary,
            "bands": records,
            "specialists": specialist_diagnostics,
        }
        return {
            "bands": frozen,
            "feature_fields": list(self.feature_fields),
            "champion_iterations": self.champion_iterations,
            "champion_weights": self.champion_weights,
        }

    def forward_select(self, prepared: dict) -> dict:
        if self.debug:
            fields = [
                field for field in TABPFN_FIELDS if field in prepared["train_a"].columns
            ]
            self.feature_fields = fields
            return {
                "bands": {
                    band: {"head": "champion", "weight": 0.0}
                    for band in self.bands
                },
                "feature_fields": fields,
                "champion_iterations": self.champion_iterations,
                "champion_weights": self.champion_weights,
            }
        oof_path = self.cache_root / "five_fold_oof_v1.parquet"
        cache_valid = False
        if oof_path.exists():
            oof = pd.read_parquet(oof_path)
            required = {
                "fold",
                "champion",
                "raw",
                "logit",
                "raw_q25",
                "raw_q75",
                "logit_q25",
                "logit_q75",
            }
            if required.issubset(oof.columns) and set(oof["fold"].unique()) == {
                str(date.date()) for date in self.fold_dates
            }:
                cache_valid = True
                self.feature_fields = [
                    field
                    for field in TABPFN_FIELDS
                    if field in prepared["train_a"].columns
                ]
                print(f"[cache] five_fold_oof hit rows={len(oof)}")
            else:
                oof = self.build_oof(
                    prepared["train_a"]
                    .sort_values(["timestamp", "AdID"])
                    .reset_index(drop=True)
                )
        else:
            oof = self.build_oof(
                prepared["train_a"]
                .sort_values(["timestamp", "AdID"])
                .reset_index(drop=True)
            )
        if not cache_valid:
            temporary = oof_path.with_name(f"{oof_path.name}.{uuid.uuid4().hex}.tmp")
            oof.to_parquet(temporary, index=False)
            os.replace(temporary, oof_path)
        self.pipeline.register_artifact(
            oof_path,
            "Common five-fold run_0001 and TabPFN-v2 predictive-median OOF predictions",
            "rel-avito-ad-ctr-tabpfn-v2-five-fold-oof-v1",
        )
        selection = self.select_bands(oof)
        selection_path = self.cache_root / "five_fold_selection_v1.json"
        temporary = selection_path.with_name(
            f"{selection_path.name}.{uuid.uuid4().hex}.tmp"
        )
        temporary.write_text(
            json.dumps(
                {
                    "selection": selection,
                    "metrics": self.selection_metrics,
                },
                indent=2,
            )
        )
        os.replace(temporary, selection_path)
        self.pipeline.register_artifact(
            selection_path,
            "Leave-one-origin-out volume-band TabPFN selection with bootstrap gates",
            "rel-avito-ad-ctr-tabpfn-v2-five-fold-selection-v1",
        )
        print(
            f"[selection] champion={self.selection_metrics['champion']} "
            f"meta={self.selection_metrics['meta_crossfit']} "
            f"bands={selection['bands']}"
        )
        return selection

    def apply_selection(
        self,
        champion: np.ndarray,
        specialists: dict,
        frame: pd.DataFrame,
        selection: dict,
    ) -> np.ndarray:
        prediction = champion.copy()
        volume = self.history_band(frame).to_numpy()
        for band, choice in selection["bands"].items():
            weight = float(choice["weight"])
            head = choice["head"]
            if weight <= 0 or head == "champion":
                continue
            mask = volume == band
            prediction[mask] = (
                (1 - weight) * champion[mask]
                + weight * np.asarray(specialists[head], dtype=float)[mask]
            )
        return np.clip(prediction, 2e-4, 1.0).astype("float64")

    def fit_model_a(self, prepared: dict, selection: dict) -> np.ndarray:
        champion = self.bank("val")
        context = self.build_context(prepared["train_a"])
        try:
            specialist = self.fit_specialists(context, prepared["val"], "final_a")
            self.final_records["model_a"] = {
                "context_rows": int(len(context)),
                "raw_q25_q75_mean_width": float(
                    np.mean(specialist["raw_q75"] - specialist["raw_q25"])
                ),
                "logit_q25_q75_mean_width": float(
                    np.mean(specialist["logit_q75"] - specialist["logit_q25"])
                ),
            }
            if self.debug:
                return champion
            return self.apply_selection(
                champion,
                specialist,
                prepared["val"],
                selection,
            )
        except Exception as error:
            self.failures.append(f"model_a {type(error).__name__}: {error}")
            return champion

    def model_b_training(self, prepared: dict) -> pd.DataFrame:
        val_labels = prepared["task"].get_table("val").df[
            ["AdID", "timestamp", prepared["task"].target_col]
        ].rename(columns={prepared["task"].target_col: "target"})
        labels = pd.concat(
            [
                prepared["train_b"][["AdID", "timestamp", "target"]],
                val_labels,
            ],
            ignore_index=True,
        ).drop_duplicates(["AdID", "timestamp"], keep="last")
        return prepared["train_b"].drop(columns=["target"]).merge(
            labels,
            on=["AdID", "timestamp"],
            how="left",
            validate="one_to_one",
        )

    def fit_model_b(self, prepared: dict, selection: dict) -> np.ndarray:
        champion = self.bank("test")
        train_b = self.model_b_training(prepared)
        context = self.build_context(train_b)
        try:
            specialist = self.fit_specialists(context, prepared["test"], "final_b")
            self.final_records["model_b"] = {
                "context_rows": int(len(context)),
                "raw_q25_q75_mean_width": float(
                    np.mean(specialist["raw_q75"] - specialist["raw_q25"])
                ),
                "logit_q25_q75_mean_width": float(
                    np.mean(specialist["logit_q75"] - specialist["logit_q25"])
                ),
            }
            if self.debug:
                return champion
            return self.apply_selection(
                champion,
                specialist,
                prepared["test"],
                selection,
            )
        except Exception as error:
            self.failures.append(f"model_b {type(error).__name__}: {error}")
            return champion

    def diagnostics(self, prepared: dict, selection: dict) -> dict:
        output = {
            "debug": self.debug,
            "method": "run_0001_volume_banded_tabpfn_v2_predictive_median",
            "bank_hashes": self.bank_hashes,
            "replay_a_rows": int(len(prepared["train_a"])),
            "replay_b_rows": int(len(prepared["train_b"])),
            "context_cap": self.context_cap,
            "tabpfn_version": "V2",
            "tabpfn_estimators": 8,
            "tabpfn_device": self.device,
            "tabpfn_fit_mode": "fit_with_cache",
            "feature_count": int(len(selection["feature_fields"])),
            "feature_fields": selection["feature_fields"],
            "band_selection": selection["bands"],
            "context_records": self.context_records,
            "final_records": self.final_records,
            "failures": self.failures,
        }
        if self.selection_metrics:
            output["forward"] = self.selection_metrics
            strata = {}
            for axis in ("fold", "history_band"):
                grouped = self.oof.groupby(axis, observed=True)
                strata[axis] = {}
                for value, group in grouped:
                    strata[axis][str(value)] = {
                        "count": int(len(group)),
                        "champion_mae": float(
                            np.mean(np.abs(group["target"] - group["champion"]))
                        ),
                        "raw_tabpfn_mae": float(
                            np.mean(np.abs(group["target"] - group["raw"]))
                        ),
                        "logit_tabpfn_mae": float(
                            np.mean(np.abs(group["target"] - group["logit"]))
                        ),
                    }
            output["forward_strata"] = strata
        return output

    def record_campaign(self, metrics: dict) -> None:
        if self.debug:
            return
        marker = "lane1-tabpfn-v2-volume-specialist"
        content = f"""

### {marker}
- run/experiment: generic_exp_5 lane 1 | status: TESTED-KEPT
- what: compact 120-field TabPFN-v2 raw/logit predictive medians, common five-fold frozen run_0001 OOF, and leave-one-origin-out five-band blend gates.
- outcome: champion {json.dumps(metrics.get("forward", {}).get("champion", {}), sort_keys=True)}; meta-crossfit {json.dumps(metrics.get("forward", {}).get("meta_crossfit", {}), sort_keys=True)}; frozen bands {json.dumps(metrics.get("band_selection", {}), sort_keys=True)}.
- takeaway: retain only bands passing band MAE, overall mean/worst, and 5,000-replicate day-block bootstrap probability gates; all rejected rows stay byte-identical to run_0001.
"""
        self.pipeline.append_once(
            Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / "features_history.md",
            marker,
            content,
        )
