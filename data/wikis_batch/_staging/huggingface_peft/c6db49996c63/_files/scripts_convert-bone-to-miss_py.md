# File: `scripts/convert-bone-to-miss.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 70 |
| Functions | `convert_bone_to_miss`, `main` |
| Imports | argparse, json, os, pathlib, peft, safetensors |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Checkpoint conversion utility that migrates saved Bone PEFT method checkpoints to the MiSS format by updating configuration metadata and renaming weight tensor keys.

**Mechanism:** The convert_bone_to_miss() function reads the Bone adapter_config.json, changes peft_type from "BONE" to "MISS", and writes it to the output directory. Then uses safetensors to load the Bone weights, replaces all ".bone_" substrings in tensor keys with ".miss_", and saves to the new location. The main() function provides CLI argument parsing for input/output directories.

**Significance:** Migration utility that enables users to convert between related PEFT methods (Bone and MiSS). Useful when methods share similar architectures but have different names or when one method supersedes another. Demonstrates the checkpoint format compatibility between these two PEFT approaches.
