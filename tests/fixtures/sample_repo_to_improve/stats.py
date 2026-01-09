"""
Statistics computation module.

Provides functions to compute basic descriptive statistics.
"""

from typing import Dict, List


def compute_stats(data: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Compute basic statistics from data rows.
    
    Args:
        data: List of row dicts with numeric values
        
    Returns:
        Dict with computed statistics
    """
    if not data:
        return {"row_count": 0, "mean": 0.0, "total": 0.0}
    
    # Get all numeric values from "value" column (if present)
    values = [row.get("value", 0.0) for row in data]
    
    row_count = len(data)
    total = sum(values)
    mean = total / row_count if row_count > 0 else 0.0
    
    # Min/max
    min_val = min(values) if values else 0.0
    max_val = max(values) if values else 0.0
    
    return {
        "row_count": row_count,
        "total": total,
        "mean": mean,
        "min": min_val,
        "max": max_val,
    }
