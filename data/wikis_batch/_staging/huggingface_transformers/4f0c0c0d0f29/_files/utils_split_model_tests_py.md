# File: `utils/split_model_tests.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 88 |
| Imports | argparse, ast, os |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Divides model test directories into balanced slices for distributed GitHub Actions testing.

**Mechanism:** Lists all model folders under tests/models/ plus top-level test directories, optionally filters to a pre-computed subset, then splits the sorted list into N equal slices ensuring each slice has roughly the same number of folders. Outputs a nested list where each inner list represents one matrix slice.

**Significance:** Critical CI infrastructure that enables bypassing GitHub Actions' 256 job matrix limit by pre-splitting test folders into manageable chunks. Used in self-scheduled.yml workflow to run tests across hundreds of models in parallel, balancing load across CI runners for optimal throughput.
