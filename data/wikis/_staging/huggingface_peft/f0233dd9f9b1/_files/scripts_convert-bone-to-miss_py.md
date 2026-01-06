# File: `scripts/convert-bone-to-miss.py`

**Category:** migration utility script

| Property | Value |
|----------|-------|
| Lines | 71 |
| Functions | `convert_bone_to_miss`, `main` |
| Imports | argparse, json, os, pathlib, safetensors.safe_open, safetensors.torch.save_file, peft.utils |
| Executable | Yes (`__main__`, shebang) |

## Understanding

**Status:** Explored

**Purpose:** Converts PEFT checkpoints from the legacy "Bone" format to the newer "MiSS" (Mixed Subspace Search) format. This is a migration tool for updating saved adapter weights.

**Mechanism:**
- `convert_bone_to_miss()`: Main conversion function that:
  - Loads the config.json from Bone checkpoint directory
  - Updates `peft_type` field from "BONE" to "MISS"
  - Saves modified config to MiSS directory
  - Loads safetensors weights from Bone checkpoint
  - Renames all weight keys by replacing ".bone_" prefix with ".miss_"
  - Saves converted weights to MiSS directory using safetensors format

- Uses PEFT constants:
  - `CONFIG_NAME`: Standard config filename ("adapter_config.json")
  - `SAFETENSORS_WEIGHTS_NAME`: Standard weights filename ("adapter_model.safetensors")

Command-line arguments:
- `bone_dir`: Path to directory containing Bone checkpoint files
- `miss_dir`: Path to output directory for converted MiSS checkpoint

**Significance:** Migration utility for users upgrading from the deprecated Bone adapter method to the newer MiSS method. Bone and MiSS appear to be structurally similar PEFT methods with identical weight structures but different naming conventions. This script ensures backward compatibility by allowing users to convert their saved checkpoints without retraining.
