# huggingface_transformers_ModelsToDeprecate

## Metadata

| Attribute | Value |
|-----------|-------|
| Source | `/utils/models_to_deprecate.py` |
| Repository | huggingface/transformers |
| Commit | f9f6619c2cf7 |
| Domains | model-lifecycle, analytics, repository-management, hub-integration |
| Last Updated | 2025-12-18 |

## Overview

The `models_to_deprecate.py` script identifies model architectures in the Transformers library that are candidates for deprecation based on download statistics from the Hugging Face Hub and model age. It provides data-driven insights to inform maintenance decisions and resource allocation.

## Description

This implementation performs comprehensive analysis to identify underutilized model architectures that may be candidates for deprecation. The script combines local repository analysis with Hugging Face Hub API queries to evaluate model usage and age.

**Analysis Methodology**:

1. **Repository Scanning**: Identifies all active model implementations by scanning the codebase
2. **Temporal Filtering**: Excludes recently added models (default: less than 1 year old) to allow adoption time
3. **Hub Integration**: Queries Hugging Face Hub for download statistics across all model checkpoints
4. **Tag Mapping**: Handles mismatches between local folder names and Hub model tags
5. **Multi-Tag Support**: Aggregates downloads across multiple tags for models with variants (e.g., VLMs with text-only versions)
6. **Threshold-Based Filtering**: Identifies models below download thresholds as deprecation candidates
7. **Validation**: Cross-references results against expected model mappings for completeness

**Key Design Features**:

- **Caching Support**: Can save/load model metadata to avoid repeated Git operations and Hub queries
- **Flexible Thresholds**: Configurable download and date thresholds for different deprecation criteria
- **Comprehensive Mapping**: Handles special cases where folder names don't match Hub tags
- **Progress Tracking**: Uses tqdm for long-running Hub queries
- **Error Handling**: Continues processing even if individual Hub queries fail
- **Validation Checks**: Ensures all non-deprecated models are accounted for in the analysis

The script is designed for periodic execution to maintain model lifecycle hygiene and inform deprecation decisions based on objective usage metrics.

## Usage

### Command Line Interface

```bash
python utils/models_to_deprecate.py \
    [--thresh_num_downloads <threshold>] \
    [--thresh_date <YYYY-MM-DD>] \
    [--save_model_info] \
    [--use_cache] \
    [--max_num_models <count>]
```

### Arguments

- `--thresh_num_downloads`: Download threshold below which models are flagged (default: 5000)
- `--thresh_date`: Consider models added after this date (format: YYYY-MM-DD, default: 1 year ago)
- `--save_model_info`: Save retrieved model information to models_info.json
- `--use_cache`: Load cached model info instead of querying Git/Hub
- `--max_num_models`: Limit analysis to first N models for testing (default: -1 = all models)

### Example

```bash
# Basic usage with defaults (5000 downloads, 1 year old)
python utils/models_to_deprecate.py

# Higher threshold for conservative deprecation
python utils/models_to_deprecate.py --thresh_num_downloads 10000

# Custom date threshold
python utils/models_to_deprecate.py --thresh_date 2023-01-01

# Save results for future analysis
python utils/models_to_deprecate.py --save_model_info

# Use cached data for faster re-runs with different thresholds
python utils/models_to_deprecate.py --use_cache --thresh_num_downloads 3000

# Test with limited models
python utils/models_to_deprecate.py --max_num_models 10 --save_model_info
```

## Code Reference

### Classes

#### `HubModelLister`

Iterator wrapper for Hub API model listing with error handling.

**Attributes**:
- `tags`: Model tag filter for Hub query
- `model_list`: HfApi list_models result iterator

**Methods**:
- `__init__(self, tags: str)`: Initialize with model tag filter
- `__iter__(self)`: Iterate over models, catching and logging exceptions

**Purpose**: Prevents single Hub query failures from crashing entire analysis

---

### Main Functions

#### `_extract_commit_hash(commits: list[str]) -> str`

Extracts commit hash from git log output.

**Parameters**:
- `commits`: Lines from git log command output

**Returns**: Commit hash string, or empty string if not found

**Implementation Details**:
- Searches for lines starting with "commit "
- Returns hash from first matching line

---

#### `get_list_of_repo_model_paths(models_dir: Path) -> list[str]`

Identifies all active model implementation files in the repository.

**Parameters**:
- `models_dir`: Path to src/transformers/models directory

**Returns**: List of paths to modeling_*.py files for active models

**Implementation Details**:
- Globs for all modeling_*.py files
- Excludes deprecated model implementations
- Excludes auto/ directory implementations
- Filters out symlinks to deprecated models

**Example Return**:
```python
[
    "/path/to/src/transformers/models/bert/modeling_bert.py",
    "/path/to/src/transformers/models/gpt2/modeling_gpt2.py",
    ...
]
```

---

#### `get_list_of_models_to_deprecate(thresh_num_downloads: int = 5_000, thresh_date: str = None, use_cache: bool = False, save_model_info: bool = False, max_num_models: int = -1) -> None`

Main analysis function that identifies deprecation candidates.

**Parameters**:
- `thresh_num_downloads`: Minimum download count to avoid deprecation (default: 5000)
- `thresh_date`: ISO format date string for minimum model age (default: 1 year ago)
- `use_cache`: Load from models_info.json instead of querying (default: False)
- `save_model_info`: Save collected data to models_info.json (default: False)
- `max_num_models`: Limit processing to N models for testing (default: -1 = all)

**Returns**: None (prints results to stdout)

**Workflow**:

1. **Date Processing**:
   - Parse thresh_date or default to 1 year ago
   - Convert to timezone-aware datetime

2. **Repository Analysis**:
   - Get all active model paths via get_list_of_repo_model_paths()
   - Build models_info dictionary with:
     - Model folder name as key
     - commit_hash: First commit introducing the model
     - first_commit_datetime: Timestamp of first commit
     - model_path: Path to modeling file
     - downloads: Initialize to 0
     - tags: List of Hub tags for this model

3. **Tag Mapping** (if not using cache):
   - Apply MODEL_FOLDER_NAME_TO_TAG_MAPPING for folder-to-tag mismatches
   - Apply EXTRA_TAGS_MAPPING for models with multiple Hub tags
   - Validate completeness against MODEL_NAMES_MAPPING

4. **Temporal Filtering**:
   - Remove models added after thresh_date
   - Allows time for model adoption before consideration

5. **Download Aggregation**:
   - For each model and its tags:
     - Query Hub API with tag filter
     - Sum downloads across all public checkpoints
     - Stop early if threshold exceeded (optimization)

6. **Candidate Identification**:
   - Filter to models with downloads < thresh_num_downloads
   - Sort by download count (lowest first)

7. **Reporting**:
   - Print each candidate with downloads and commit date
   - Print summary list and count
   - Include verification reminder

**Output Format**:
```
Building a dictionary of basic model info...
100%|████████████| 150/150 [00:30<00:00,  5.00it/s]

Making calls to the hub to find models below the threshold number of downloads...
1/120: getting hub downloads for model='transfo-xl' (tags=['transfo-xl'])
2/120: getting hub downloads for model='ctrl' (tags=['ctrl'])
...

Finding models to deprecate:

Model: transfo-xl
Downloads: 2543
Date: 2020-03-15 10:23:45+00:00

Model: ctrl
Downloads: 3821
Date: 2020-04-22 14:35:12+00:00

Models to deprecate:
transfo-xl
ctrl

Number of models to deprecate: 2
Before deprecating make sure to verify the models, including if they're used as a module in other models.
```

**Cache Structure** (models_info.json):
```json
{
    "bert": {
        "commit_hash": "abc123...",
        "first_commit_datetime": "2019-08-15T09:23:45+00:00",
        "model_path": "/path/to/modeling_bert.py",
        "downloads": 15234567,
        "tags": ["bert", "bert-japanese", "bertweet"]
    },
    "transfo-xl": {
        "commit_hash": "def456...",
        "first_commit_datetime": "2020-03-15T10:23:45+00:00",
        "model_path": "/path/to/modeling_transfo_xl.py",
        "downloads": 2543,
        "tags": ["transfo-xl"]
    }
}
```

---

### Constants

#### `MODEL_FOLDER_NAME_TO_TAG_MAPPING`

Dictionary mapping local folder names to Hub model tags when they differ.

**Type**: `dict[str, str]`

**Example Entries**:
```python
{
    "openai": "openai-gpt",
    "xlm_roberta": "xlm-roberta",
    "blip_2": "blip-2",
    ...
}
```

**Purpose**: Handles naming inconsistencies between local codebase and Hub taxonomy

---

#### `EXTRA_TAGS_MAPPING`

Dictionary mapping base model tags to additional Hub tags for download aggregation.

**Type**: `dict[str, list[str]]`

**Example Entries**:
```python
{
    "bert": ["bert-japanese", "bertweet", "herbert", "phobert"],
    "clip": ["clip_text_model", "clip_vision_model"],
    "llama": ["code_llama", "falcon3", "llama2", "llama3"],
    ...
}
```

**Purpose**: Ensures download counts include all variants/sub-architectures

---

#### `DEPRECATED_MODELS_TAGS`

Set of model tags for models already deprecated where tag differs from folder name.

**Type**: `set[str]`

**Example**:
```python
{"gptsan-japanese", "open-llama", "transfo-xl", "xlm-prophetnet"}
```

**Purpose**: Used in validation to account for already-deprecated models with non-standard tags

---

## I/O Contract

### Inputs

| Input Type | Description | Format | Example |
|------------|-------------|--------|---------|
| Command Line Arguments | Analysis configuration | Argparse flags | `--thresh_num_downloads 5000` |
| File System | Model implementations | Python modules | src/transformers/models/*/modeling_*.py |
| File System | Deprecated model directory | Directory structure | src/transformers/models/deprecated/* |
| Git Repository | Commit history | Git log output | commit hash and timestamp |
| Python Module | Model configuration mapping | CONFIG_MAPPING dict | transformers.models.auto.configuration_auto |
| Python Module | Deprecated models list | DEPRECATED_MODELS list | transformers.models.auto.configuration_auto |
| Python Module | Model name mapping | MODEL_NAMES_MAPPING dict | transformers.models.auto.configuration_auto |
| Hugging Face Hub | Model metadata | API JSON responses | download counts per model |
| Cache File (optional) | Saved model information | models_info.json | Serialized analysis results |

### Outputs

| Output Type | Description | Format | Location |
|-------------|-------------|--------|----------|
| Console Output | Analysis progress | Text with tqdm progress bars | stdout |
| Console Output | Deprecation candidates | Formatted text listing | stdout |
| Console Output | Summary statistics | Text summary | stdout |
| JSON File (optional) | Complete model metadata | JSON object | ./models_info.json |

### Side Effects

- Makes numerous HTTP requests to Hugging Face Hub API (rate limiting may apply)
- Executes git log commands on repository
- May create models_info.json file if --save_model_info used
- Long execution time for full repository analysis (minutes to hours depending on model count)

## Usage Examples

### Basic Analysis

```bash
# Run with default settings (5000 downloads, 1 year old)
python utils/models_to_deprecate.py

# Output shows:
# - Progress bars for repo scanning and Hub queries
# - List of models below threshold with details
# - Summary count and verification reminder
```

### Conservative Analysis (Higher Threshold)

```bash
# Flag models only if they have very low usage
python utils/models_to_deprecate.py --thresh_num_downloads 10000

# Fewer models will be flagged as candidates
```

### Aggressive Analysis (Recent Models Included)

```bash
# Include models added in last 6 months
python utils/models_to_deprecate.py --thresh_date 2024-06-01

# More models will be considered (less time for adoption)
```

### Caching Workflow

```bash
# First run: collect all data and save
python utils/models_to_deprecate.py \
    --save_model_info \
    --thresh_num_downloads 10000

# Creates models_info.json with all model metadata

# Subsequent runs: use cache for different thresholds
python utils/models_to_deprecate.py \
    --use_cache \
    --thresh_num_downloads 5000

# Much faster, no Git/Hub queries needed

python utils/models_to_deprecate.py \
    --use_cache \
    --thresh_num_downloads 3000

# Test different thresholds quickly
```

### Testing Mode

```bash
# Test script logic on small subset
python utils/models_to_deprecate.py \
    --max_num_models 10 \
    --save_model_info

# Processes only first 10 models for rapid testing
```

### Programmatic Usage

```python
from models_to_deprecate import (
    get_list_of_repo_model_paths,
    get_list_of_models_to_deprecate,
    HubModelLister,
    MODEL_FOLDER_NAME_TO_TAG_MAPPING,
    EXTRA_TAGS_MAPPING
)
from pathlib import Path

# Get all active model paths
models_dir = Path("src/transformers/models")
model_paths = get_list_of_repo_model_paths(models_dir)
print(f"Found {len(model_paths)} active models")

# Run analysis programmatically
get_list_of_models_to_deprecate(
    thresh_num_downloads=5000,
    thresh_date="2023-01-01",
    use_cache=False,
    save_model_info=True,
    max_num_models=-1
)

# Results printed to stdout
```

### Working with Model Tags

```python
# Check if model needs tag mapping
model_folder = "xlm_roberta"
if model_folder in MODEL_FOLDER_NAME_TO_TAG_MAPPING:
    hub_tag = MODEL_FOLDER_NAME_TO_TAG_MAPPING[model_folder]
    print(f"Folder '{model_folder}' uses Hub tag '{hub_tag}'")
else:
    hub_tag = model_folder
    print(f"Folder '{model_folder}' uses same Hub tag")

# Check for additional tags
if hub_tag in EXTRA_TAGS_MAPPING:
    extra_tags = EXTRA_TAGS_MAPPING[hub_tag]
    all_tags = [hub_tag] + extra_tags
    print(f"Model has {len(all_tags)} tags total: {all_tags}")
```

### Querying Hub with Error Handling

```python
from models_to_deprecate import HubModelLister

# Safe iteration over Hub models
tag = "bert"
total_downloads = 0

model_list = HubModelLister(tags=tag)
for model in model_list:
    if not model.private:
        total_downloads += model.downloads
        print(f"{model.id}: {model.downloads:,} downloads")

print(f"Total downloads for tag '{tag}': {total_downloads:,}")

# If Hub query fails, HubModelLister prints error but doesn't crash
```

### Loading Cached Data

```python
import json
from datetime import datetime

# Load cached model info
with open("models_info.json", "r") as f:
    models_info = json.load(f)

# Convert ISO strings back to datetime
for model, info in models_info.items():
    info["first_commit_datetime"] = datetime.fromisoformat(
        info["first_commit_datetime"]
    )

# Analyze cached data
low_download_models = {
    model: info
    for model, info in models_info.items()
    if info["downloads"] < 5000
}

print(f"Models below 5000 downloads: {len(low_download_models)}")
for model, info in sorted(
    low_download_models.items(),
    key=lambda x: x[1]["downloads"]
):
    print(f"  {model}: {info['downloads']} downloads")
```

### Custom Threshold Analysis

```python
from models_to_deprecate import get_list_of_models_to_deprecate

# Multiple threshold analysis
thresholds = [1000, 2500, 5000, 10000]

for threshold in thresholds:
    print(f"\n{'='*60}")
    print(f"Analysis with {threshold} download threshold")
    print('='*60)

    get_list_of_models_to_deprecate(
        thresh_num_downloads=threshold,
        use_cache=True  # Use cache after first run
    )
```

## Related Pages

- [Related implementation pages will be listed here]
