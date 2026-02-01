"""
Data preparation script for the chart extraction example.

Downloads a subset of line charts from the ChartQA dataset for evaluation.

Usage:
    python prepare_data.py

This creates a subset_line_100/ directory with:
- index.csv: mapping of example IDs to image and ground truth table paths
- images/: chart images (PNG/JPEG)
- tables/: ground truth CSV tables
"""

import os
import csv
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download


def prepare_data(output_dir: str = "subset_line_100", num_samples: int = 100):
    """
    Download and prepare a subset of line charts from ChartQA.
    
    Args:
        output_dir: Directory to store the prepared data.
        num_samples: Number of samples to include in the subset.
    """
    output_path = Path(output_dir)
    
    # Clean up existing directory
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Create directory structure
    output_path.mkdir(parents=True)
    (output_path / "images").mkdir()
    (output_path / "tables").mkdir()
    
    print(f"Downloading ChartQA dataset...")
    
    # Download the dataset snapshot
    # Note: This uses the ChartQA dataset from HuggingFace
    try:
        dataset_path = snapshot_download(
            repo_id="ahmed-masry/ChartQA",
            repo_type="dataset",
            cache_dir=".cache",
            allow_patterns=["*.png", "*.csv", "*.json"],
        )
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have huggingface_hub installed and internet access.")
        return
    
    dataset_path = Path(dataset_path)
    
    # Find line chart images and their corresponding tables
    # ChartQA structure varies, so we'll look for matching pairs
    index_rows = []
    sample_count = 0
    
    # Look for image-table pairs in the dataset
    for img_file in sorted(dataset_path.rglob("*.png")):
        if sample_count >= num_samples:
            break
            
        # Try to find corresponding CSV table
        table_name = img_file.stem + ".csv"
        table_candidates = list(dataset_path.rglob(table_name))
        
        if table_candidates:
            table_file = table_candidates[0]
            
            # Copy files to output directory
            new_img_name = f"{sample_count:04d}.png"
            new_table_name = f"{sample_count:04d}.csv"
            
            shutil.copy(img_file, output_path / "images" / new_img_name)
            shutil.copy(table_file, output_path / "tables" / new_table_name)
            
            index_rows.append({
                "id": sample_count,
                "image": f"images/{new_img_name}",
                "table": f"tables/{new_table_name}",
            })
            
            sample_count += 1
    
    # Write index file
    with open(output_path / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "image", "table"])
        writer.writeheader()
        writer.writerows(index_rows)
    
    print(f"Prepared {sample_count} samples in {output_dir}/")
    print(f"  - index.csv: mapping file")
    print(f"  - images/: {sample_count} chart images")
    print(f"  - tables/: {sample_count} ground truth CSV tables")


if __name__ == "__main__":
    prepare_data()
