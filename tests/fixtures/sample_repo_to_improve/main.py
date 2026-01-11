"""
Simple data processing pipeline.

This script:
1. Loads data from a CSV file
2. Computes basic statistics
3. Prints a summary report

Usage:
    python main.py
"""

import csv
import os
from typing import Dict, List

from stats import compute_stats
from report import format_report


def load_data(filepath: str) -> List[Dict[str, float]]:
    """
    Load numeric data from a CSV file.
    
    Args:
        filepath: Path to CSV file with header row
        
    Returns:
        List of row dicts with float values
    """
    if not os.path.exists(filepath):
        return []
    
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            numeric_row = {}
            for k, v in row.items():
                try:
                    numeric_row[k] = float(v)
                except ValueError:
                    numeric_row[k] = 0.0
            rows.append(numeric_row)
    return rows


def main():
    """Main entry point."""
    # Try to load data, fall back to sample data
    data = load_data("data.csv")
    
    if not data:
        # Sample data for testing
        data = [
            {"value": 10.0, "count": 5.0},
            {"value": 20.0, "count": 3.0},
            {"value": 15.0, "count": 8.0},
            {"value": 25.0, "count": 2.0},
        ]
    
    # Compute statistics
    stats = compute_stats(data)
    
    # Generate and print report
    report = format_report(stats)
    print(report)
    
    return stats


if __name__ == "__main__":
    result = main()
    print(f"\nSUCCESS: Processed {result.get('row_count', 0)} rows")
