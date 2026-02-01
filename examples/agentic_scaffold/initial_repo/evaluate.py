"""
Evaluation script for VLM chart data extraction.

This script:
1. Loads the prepared dataset from subset_line_100/
2. Calls VLMExtractor.image_to_csv() for each chart image
3. Compares predicted CSV tables to ground truth using a similarity metric
4. Reports accuracy metric for Kapso optimization

Usage:
    python evaluate.py [--max-samples N] [--num-workers N]

The script outputs an accuracy score that Kapso uses for optimization guidance.
"""

import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import optimize  # the file Kapso mutates


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DEFAULT_MAX_SAMPLES = 100
DEFAULT_NUM_WORKERS = 4
LOG_EVERY = 5
MODEL_TO_USE = "gpt-4o-mini"
COST_CAP_PER_QUERY = 0.02  # Max average cost per query in USD
# ---------------------------------------------------------------------


def parse_csv_content(csv_text: str) -> list[list[str]]:
    """
    Parse CSV text into a list of rows.
    
    Args:
        csv_text: CSV formatted string.
    
    Returns:
        List of rows, where each row is a list of cell values.
    """
    rows = []
    for line in csv_text.strip().split("\n"):
        if line.strip():
            # Simple CSV parsing (handles basic cases)
            cells = [cell.strip() for cell in line.split(",")]
            rows.append(cells)
    return rows


def compute_similarity(predicted_csv: str, ground_truth_csv: str) -> float:
    """
    Compute similarity between predicted and ground truth CSV tables.
    
    The metric combines:
    - Header match (20% weight): Exact match of column headers
    - Content similarity (80% weight): Jaccard-based similarity of data rows
    
    Args:
        predicted_csv: The predicted CSV string.
        ground_truth_csv: The ground truth CSV string.
    
    Returns:
        Similarity score between 0.0 and 1.0.
    """
    pred_rows = parse_csv_content(predicted_csv)
    gt_rows = parse_csv_content(ground_truth_csv)
    
    if not pred_rows or not gt_rows:
        return 0.0
    
    # Header match (20% weight)
    pred_header = pred_rows[0] if pred_rows else []
    gt_header = gt_rows[0] if gt_rows else []
    
    header_score = 0.0
    if pred_header and gt_header:
        # Normalize headers for comparison
        pred_header_norm = [h.lower().strip() for h in pred_header]
        gt_header_norm = [h.lower().strip() for h in gt_header]
        
        if pred_header_norm == gt_header_norm:
            header_score = 1.0
        else:
            # Partial match based on overlap
            common = set(pred_header_norm) & set(gt_header_norm)
            total = set(pred_header_norm) | set(gt_header_norm)
            header_score = len(common) / len(total) if total else 0.0
    
    # Content similarity (80% weight)
    pred_data = pred_rows[1:] if len(pred_rows) > 1 else []
    gt_data = gt_rows[1:] if len(gt_rows) > 1 else []
    
    content_score = 0.0
    if pred_data and gt_data:
        # Compare row by row using SMAPE for numeric values
        total_cells = 0
        matching_cells = 0
        
        for i, gt_row in enumerate(gt_data):
            if i < len(pred_data):
                pred_row = pred_data[i]
                for j, gt_val in enumerate(gt_row):
                    total_cells += 1
                    if j < len(pred_row):
                        pred_val = pred_row[j]
                        # Try numeric comparison
                        try:
                            gt_num = float(gt_val.replace(",", ""))
                            pred_num = float(pred_val.replace(",", ""))
                            # SMAPE-based similarity
                            if gt_num == 0 and pred_num == 0:
                                matching_cells += 1
                            else:
                                smape = abs(pred_num - gt_num) / (abs(pred_num) + abs(gt_num) + 1e-10)
                                matching_cells += max(0, 1 - smape)
                        except ValueError:
                            # String comparison
                            if gt_val.strip().lower() == pred_val.strip().lower():
                                matching_cells += 1
            else:
                # Missing rows in prediction
                total_cells += len(gt_row)
        
        # Account for extra rows in prediction
        for i in range(len(gt_data), len(pred_data)):
            total_cells += len(pred_data[i])
        
        content_score = matching_cells / total_cells if total_cells > 0 else 0.0
    
    # Weighted combination
    final_score = 0.2 * header_score + 0.8 * content_score
    return final_score


def evaluate_sample(extractor, image_path: Path, gt_csv_path: Path) -> tuple[float, str]:
    """
    Evaluate a single sample.
    
    Args:
        extractor: The VLMExtractor instance.
        image_path: Path to the chart image.
        gt_csv_path: Path to the ground truth CSV.
    
    Returns:
        Tuple of (similarity_score, predicted_csv).
    """
    # Read ground truth
    with open(gt_csv_path, "r") as f:
        gt_csv = f.read()
    
    # Get prediction
    predicted_csv = extractor.image_to_csv(image_path)
    
    # Compute similarity
    score = compute_similarity(predicted_csv, gt_csv)
    
    return score, predicted_csv


def run_evaluation(max_samples: int, num_workers: int) -> float:
    """
    Run the full evaluation.
    
    Args:
        max_samples: Maximum number of samples to evaluate.
        num_workers: Number of parallel workers.
    
    Returns:
        Average accuracy score.
    """
    data_dir = Path("subset_line_100")
    index_path = data_dir / "index.csv"
    predictions_dir = Path("predictions")
    
    # Check if data exists
    if not index_path.exists():
        print("[error] Data not found. Please run prepare_data.py first.")
        return 0.0
    
    # Create predictions directory
    predictions_dir.mkdir(exist_ok=True)
    
    # Load index
    with open(index_path, "r") as f:
        reader = csv.DictReader(f)
        samples = list(reader)[:max_samples]
    
    print(f"[setup] evaluating {len(samples)} samples using {MODEL_TO_USE} …", flush=True)
    
    # Initialize extractor
    extractor = optimize.VLMExtractor(model=MODEL_TO_USE)
    
    scores = []
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {}
        for sample in samples:
            image_path = data_dir / sample["image"]
            gt_path = data_dir / sample["table"]
            future = pool.submit(evaluate_sample, extractor, image_path, gt_path)
            futures[future] = sample
        
        for idx, future in enumerate(as_completed(futures), 1):
            sample = futures[future]
            try:
                score, predicted_csv = future.result(timeout=120)
                scores.append(score)
                
                # Save prediction
                pred_path = predictions_dir / f"{sample['id']}.csv"
                with open(pred_path, "w") as f:
                    f.write(predicted_csv)
                    
            except Exception as exc:
                print(f"[error] Sample {sample['id']} failed: {exc}")
                scores.append(0.0)
            
            if idx % LOG_EVERY == 0 or idx == len(samples):
                elapsed = time.time() - start
                avg_score = sum(scores) / len(scores) if scores else 0
                print(
                    f"[progress] {idx}/{len(samples)} done, "
                    f"avg score: {avg_score:.4f}, elapsed {elapsed:.1f}s",
                    flush=True,
                )
    
    # Calculate final accuracy
    final_accuracy = sum(scores) / len(scores) if scores else 0.0
    return final_accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM chart extraction")
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=DEFAULT_MAX_SAMPLES,
        help=f"Number of samples to evaluate (default: {DEFAULT_MAX_SAMPLES})"
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_NUM_WORKERS})"
    )
    args = parser.parse_args()
    
    accuracy = run_evaluation(args.max_samples, args.num_workers)
    
    # Kapso parses this exact line format
    print(f"accuracy: {accuracy:.4f}")
    print(f"\n__SCORE__: {accuracy}")


if __name__ == "__main__":
    main()
