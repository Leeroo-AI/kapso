import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from kapso_datasets.common import load_task


def slice_metrics(labels: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    current_labels = labels[mask]
    current_prediction = prediction[mask]
    result = {
        "count": int(mask.sum()),
        "label_rate": float(current_labels.mean()) if len(current_labels) else float("nan"),
        "average_precision": float(average_precision_score(current_labels, current_prediction)) if len(np.unique(current_labels)) == 2 else float("nan"),
        "roc_auc": float(roc_auc_score(current_labels, current_prediction)) if len(np.unique(current_labels)) == 2 else float("nan"),
    }
    return result


def main() -> None:
    cache = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    with np.load(cache / "predictions" / "generic_exp_2_snapshot_direct_v1.npz", allow_pickle=False) as payload:
        prediction = payload["val"].astype(np.float64)
        routed = payload["validation_routed"].astype(bool)
    run0005 = np.load(Path(os.environ["RELBENCH_WORK_DIR"]) / "runs" / "run_0005" / "val_predictions.npy").astype(np.float64)
    context = load_task()
    labels = context.val.df[context.target_col].to_numpy(dtype=np.int32)
    rng = np.random.default_rng(1337)
    bootstrap = []
    paired_delta = []
    for _ in range(2000):
        indices = rng.integers(0, len(labels), len(labels))
        sampled = labels[indices]
        if len(np.unique(sampled)) == 2:
            bootstrap.append(roc_auc_score(sampled, prediction[indices]))
            paired_delta.append(roc_auc_score(sampled, prediction[indices]) - roc_auc_score(sampled, run0005[indices]))
    bootstrap_array = np.asarray(bootstrap, dtype=np.float64)
    paired_array = np.asarray(paired_delta, dtype=np.float64)
    correlations = {"run_0005": float(spearmanr(prediction, run0005).statistic)}
    candidate_vectors = [run0005, prediction]
    for name in ["generic_exp_0_literature_v3.npz", "generic_exp_1_tfidf_v1.npz", "generic_exp_1_modernbert_v1.npz"]:
        path = cache / "predictions" / name
        if path.exists():
            with np.load(path, allow_pickle=False) as payload:
                correlations[name] = float(spearmanr(prediction, payload["val"]).statistic)
                candidate_vectors.append(payload["val"].astype(np.float64))
    pairwise = []
    for left in range(len(candidate_vectors)):
        for right in range(left + 1, len(candidate_vectors)):
            pairwise.append(float(spearmanr(candidate_vectors[left], candidate_vectors[right]).statistic))
    result = {
        "validation": {
            "count": int(len(labels)),
            "label_rate": float(labels.mean()),
            "roc_auc": float(roc_auc_score(labels, prediction)),
            "bootstrap_draws": int(len(bootstrap_array)),
            "bootstrap_standard_error": float(bootstrap_array.std(ddof=1)),
            "bootstrap_lower_10": float(np.quantile(bootstrap_array, 0.10)),
            "bootstrap_upper_90": float(np.quantile(bootstrap_array, 0.90)),
            "run_0005_roc_auc": float(roc_auc_score(labels, run0005)),
            "paired_delta": float(roc_auc_score(labels, prediction) - roc_auc_score(labels, run0005)),
            "paired_delta_standard_error": float(paired_array.std(ddof=1)),
            "paired_probability_positive": float((paired_array > 0).mean()),
        },
        "slices": {
            "snapshot_direct_routed": slice_metrics(labels, prediction, routed),
            "exact_run_0005_fallback": slice_metrics(labels, prediction, ~routed),
        },
        "prediction_rank_correlations": correlations,
        "mean_pairwise_rank_correlation": float(np.mean(pairwise)),
        "representativeness": {
            "official_2018": {"count": 1128, "label_rate": 0.6108156028368794},
            "official_2019": {"count": 1093, "label_rate": 0.6029277218664227},
            "validation_2020": {"count": int(len(labels)), "label_rate": float(labels.mean())},
            "test_2021": {"count": 825, "label_rate": None},
        },
    }
    output = Path("output_data_generic_exp_2") / "post_eval_diagnostics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
