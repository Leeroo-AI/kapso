"""
Report formatting module.

Converts statistics dict into human-readable report.
"""

from typing import Dict


def format_report(stats: Dict[str, float]) -> str:
    """
    Format statistics into a readable report.
    
    Args:
        stats: Dict with computed statistics
        
    Returns:
        Formatted report string
    """
    lines = [
        "=" * 40,
        "DATA PROCESSING REPORT",
        "=" * 40,
        f"Rows processed: {int(stats.get('row_count', 0))}",
        f"Total:          {stats.get('total', 0.0):.2f}",
        f"Mean:           {stats.get('mean', 0.0):.2f}",
        f"Min:            {stats.get('min', 0.0):.2f}",
        f"Max:            {stats.get('max', 0.0):.2f}",
        "=" * 40,
    ]
    return "\n".join(lines)
