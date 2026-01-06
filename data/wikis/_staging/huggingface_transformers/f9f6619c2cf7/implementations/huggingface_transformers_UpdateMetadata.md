# UpdateMetadata

## Metadata
| Key | Value |
|-----|-------|
| **Source** | `utils/update_metadata.py` |
| **Repository** | huggingface/transformers |
| **Domains** | CI/CD, Metadata Management, Model Registry |
| **Last Updated** | 2025-12-18 |

## Overview

The UpdateMetadata module maintains the model metadata repository (`huggingface/transformers-metadata`) with current information about supported models, frameworks, processors, and pipeline mappings. It automatically syncs metadata whenever the Transformers library is updated, ensuring the Hub and other services have accurate information about model capabilities and compatibility.

## Description

This implementation provides automated metadata synchronization for the Transformers ecosystem:

**Core Capabilities:**
- Generates framework support tables (PyTorch) for all model types
- Maps models to their appropriate auto-classes and pipeline tags
- Identifies correct processor classes (AutoProcessor, AutoTokenizer, etc.)
- Uploads metadata to the `huggingface/transformers-metadata` dataset on the Hub
- Validates that all pipeline tasks have proper metadata definitions
- Supports incremental updates without removing historical entries

**Key Components:**
- `get_frameworks_table()`: Scans transformers module to detect model support
- `update_pipeline_and_auto_class_table()`: Maps models to pipelines and auto-classes
- `update_metadata()`: Main orchestration function for Hub updates
- `check_pipeline_tags()`: Validates completeness of pipeline definitions
- `camel_case_split()`: Utility for parsing model class names

**Integration Points:**
- HuggingFace Hub API via `huggingface_hub`
- Transformers auto-model mappings (configuration, modeling, processing)
- `PIPELINE_TAGS_AND_AUTO_MODELS` constant defining all supported pipelines
- GitHub Actions workflow for automated updates

## Usage

### Command Line Usage

```bash
# Update metadata on the Hub (used in CI)
python utils/update_metadata.py \
  --token $HF_TOKEN \
  --commit_sha $GITHUB_SHA

# Check that all pipelines are properly defined (used in make repo-consistency)
python utils/update_metadata.py --check-only
```

### GitHub Actions Integration

```yaml
# Example workflow step
- name: Update metadata
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
  run: |
    python utils/update_metadata.py \
      --token $HF_TOKEN \
      --commit_sha ${{ github.sha }}
```

## Code Reference

### Main Functions

```python
def get_frameworks_table() -> pd.DataFrame:
    """
    Generate dataframe with framework support for each model type.

    Scans all classes in transformers module to detect which models
    are supported by PyTorch.

    Returns:
        DataFrame with columns: model_type, pytorch, processor
    """

def update_pipeline_and_auto_class_table(
    table: dict[str, tuple[str, str]]
) -> dict[str, tuple[str, str]]:
    """
    Update pipeline/auto-class mappings without removing old entries.

    Args:
        table: Existing mapping of model_class -> (pipeline_tag, auto_class)

    Returns:
        Updated table with new mappings added
    """

def update_metadata(token: str, commit_sha: str):
    """
    Update metadata in huggingface/transformers-metadata repository.

    Args:
        token: HuggingFace Hub token with write access
        commit_sha: Git commit SHA for this update

    Side Effects:
        - Downloads current metadata from Hub
        - Generates updated metadata files
        - Uploads to Hub if changes detected
    """

def check_pipeline_tags():
    """
    Validate all pipelines are defined in PIPELINE_TAGS_AND_AUTO_MODELS.

    Raises:
        ValueError: If any pipeline tags are missing from the constant
    """

def camel_case_split(identifier: str) -> list[str]:
    """
    Split camel-cased identifier into words.

    Args:
        identifier: CamelCased string to split

    Returns:
        List of words extracted from identifier

    Example:
        >>> camel_case_split("BertForSequenceClassification")
        ["Bert", "For", "Sequence", "Classification"]
    """
```

### Constants

```python
# Path to transformers source
TRANSFORMERS_PATH = "src/transformers"

# Regex for matching PyTorch model classes
_re_pt_models = re.compile(r"(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration|ForRetrieval)")

# Complete mapping of pipeline tags to model mappings and auto-classes
PIPELINE_TAGS_AND_AUTO_MODELS = [
    ("pretraining", "MODEL_FOR_PRETRAINING_MAPPING_NAMES", "AutoModelForPreTraining"),
    ("feature-extraction", "MODEL_MAPPING_NAMES", "AutoModel"),
    ("image-feature-extraction", "MODEL_FOR_IMAGE_MAPPING_NAMES", "AutoModel"),
    ("audio-classification", "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES", "AutoModelForAudioClassification"),
    ("text-generation", "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", "AutoModelForCausalLM"),
    ("automatic-speech-recognition", "MODEL_FOR_CTC_MAPPING_NAMES", "AutoModelForCTC"),
    ("image-classification", "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES", "AutoModelForImageClassification"),
    ("image-segmentation", "MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES", "AutoModelForImageSegmentation"),
    ("image-text-to-text", "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES", "AutoModelForImageTextToText"),
    ("any-to-any", "MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES", "AutoModelForMultimodalLM"),
    ("image-to-image", "MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES", "AutoModelForImageToImage"),
    ("fill-mask", "MODEL_FOR_MASKED_LM_MAPPING_NAMES", "AutoModelForMaskedLM"),
    ("object-detection", "MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES", "AutoModelForObjectDetection"),
    ("zero-shot-object-detection", "MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES", "AutoModelForZeroShotObjectDetection"),
    ("question-answering", "MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES", "AutoModelForQuestionAnswering"),
    ("text2text-generation", "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES", "AutoModelForSeq2SeqLM"),
    ("text-classification", "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES", "AutoModelForSequenceClassification"),
    ("automatic-speech-recognition", "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES", "AutoModelForSpeechSeq2Seq"),
    ("table-question-answering", "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES", "AutoModelForTableQuestionAnswering"),
    ("token-classification", "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES", "AutoModelForTokenClassification"),
    ("multiple-choice", "MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES", "AutoModelForMultipleChoice"),
    ("next-sentence-prediction", "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES", "AutoModelForNextSentencePrediction"),
    ("audio-frame-classification", "MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES", "AutoModelForAudioFrameClassification"),
    ("audio-xvector", "MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES", "AutoModelForAudioXVector"),
    ("document-question-answering", "MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES", "AutoModelForDocumentQuestionAnswering"),
    ("visual-question-answering", "MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES", "AutoModelForVisualQuestionAnswering"),
    ("image-to-text", "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES", "AutoModelForVision2Seq"),
    ("zero-shot-image-classification", "MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES", "AutoModelForZeroShotImageClassification"),
    ("depth-estimation", "MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES", "AutoModelForDepthEstimation"),
    ("video-classification", "MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES", "AutoModelForVideoClassification"),
    ("mask-generation", "MODEL_FOR_MASK_GENERATION_MAPPING_NAMES", "AutoModelForMaskGeneration"),
    ("text-to-audio", "MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES", "AutoModelForTextToSpectrogram"),
    ("text-to-audio", "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES", "AutoModelForTextToWaveform"),
    ("keypoint-matching", "MODEL_FOR_KEYPOINT_MATCHING_MAPPING_NAMES", "AutoModelForKeypointMatching"),
]
```

### Key Imports

```python
import argparse
import collections
import os
import re
import tempfile

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download, upload_folder

from transformers.utils import direct_transformers_import
```

## I/O Contract

### Inputs

| Input | Type | Source | Description |
|-------|------|--------|-------------|
| `--token` | CLI Argument | Command Line | HuggingFace Hub authentication token |
| `--commit_sha` | CLI Argument | Command Line | Git commit SHA for update tracking |
| `--check-only` | CLI Flag | Command Line | Flag to only validate pipeline definitions |
| Transformers Module | Python Package | src/transformers/ | Source of truth for model capabilities |
| `CONFIG_MAPPING_NAMES` | Python Dict | transformers.models.auto | Model type to config mapping |
| `MODEL_FOR_*_MAPPING_NAMES` | Python Dicts | transformers.models.auto.modeling_auto | Task-specific model mappings |
| `PROCESSOR_MAPPING_NAMES` | Python Dict | transformers.models.auto.processing_auto | Model to processor mappings |
| `TOKENIZER_MAPPING_NAMES` | Python Dict | transformers.models.auto.tokenization_auto | Model to tokenizer mappings |
| Hub `frameworks.json` | JSON File | huggingface/transformers-metadata | Current framework support table |
| Hub `pipeline_tags.json` | JSON File | huggingface/transformers-metadata | Current pipeline tag mappings |

### Outputs

| Output | Type | Destination | Description |
|--------|------|-------------|-------------|
| `frameworks.json` | JSON File | huggingface/transformers-metadata | Updated framework support table |
| `pipeline_tags.json` | JSON File | huggingface/transformers-metadata | Updated pipeline/auto-class mappings |
| Console Output | stdout | Terminal/CI Logs | Update status and validation results |
| Hub Commit | Git Commit | transformers-metadata repo | Committed metadata changes with SHA reference |

### Side Effects

- Downloads files from HuggingFace Hub
- Creates temporary directory for file generation
- Uploads folder to Hub if changes detected
- Creates commit in transformers-metadata repository
- Validates pipeline configuration (--check-only mode)

## Usage Examples

### Example 1: Generating Frameworks Table

```python
from update_metadata import get_frameworks_table

# Generate complete framework support table
frameworks_df = get_frameworks_table()

print(frameworks_df.head())
# Output:
#   model_type  pytorch           processor
# 0       bert     True       AutoTokenizer
# 1      gpt2     True       AutoTokenizer
# 2       t5      True       AutoTokenizer
# 3       vit     True  AutoImageProcessor
# 4      clip     True       AutoProcessor

# Check which models have processors
models_with_processors = frameworks_df[
    frameworks_df["processor"] == "AutoProcessor"
]["model_type"].tolist()
```

### Example 2: Updating Pipeline Mappings

```python
from update_metadata import update_pipeline_and_auto_class_table

# Start with existing mappings
existing_table = {
    "BertForSequenceClassification": ("text-classification", "AutoModelForSequenceClassification"),
    "GPT2LMHeadModel": ("text-generation", "AutoModelForCausalLM"),
}

# Update with current transformers definitions
updated_table = update_pipeline_and_auto_class_table(existing_table)

# New models are added, old ones preserved
print(updated_table["BertForSequenceClassification"])
# ("text-classification", "AutoModelForSequenceClassification")

print(updated_table.get("NewModelForTextGeneration"))
# ("text-generation", "AutoModelForCausalLM")  # if model exists
```

### Example 3: Splitting Camel Case Names

```python
from update_metadata import camel_case_split

# Parse model class names
parts = camel_case_split("BertForSequenceClassification")
print(parts)
# ["Bert", "For", "Sequence", "Classification"]

# Extract model prefix
parts = camel_case_split("RobertaForMaskedLM")
print(parts)
# ["Roberta", "For", "Masked", "LM"]

# Works with acronyms
parts = camel_case_split("GPT2LMHeadModel")
print(parts)
# ["GPT", "2", "LM", "Head", "Model"]
```

### Example 4: Complete Metadata Update

```python
from update_metadata import update_metadata

# Typical usage in CI/CD
update_metadata(
    token="hf_xxxxxxxxxxxxxxxxxxxx",
    commit_sha="a1b2c3d4e5f6g7h8i9j0"
)

# Output:
# - Downloads current metadata from Hub
# - Generates new frameworks.json
# - Generates new pipeline_tags.json
# - Compares with Hub versions
# - Uploads if changes detected with commit message:
#   "Update with commit a1b2c3d4e5f6g7h8i9j0
#    See: https://github.com/huggingface/transformers/commit/a1b2c3d4e5f6g7h8i9j0"
```

### Example 5: Validating Pipeline Tags

```python
from update_metadata import check_pipeline_tags

# Validate all pipelines are defined (used in CI checks)
try:
    check_pipeline_tags()
    print("All pipeline tags are properly defined")
except ValueError as e:
    print(f"Missing pipeline tags: {e}")
    # Example error:
    # "The following pipeline tags are not present in the
    # `PIPELINE_TAGS_AND_AUTO_MODELS` constant inside
    # `utils/update_metadata.py`: new-task-type. Please add them!"
```

### Example 6: Checking Model Framework Support

```python
from update_metadata import get_frameworks_table, camel_case_split
import transformers

# Get frameworks table
frameworks_df = get_frameworks_table()

# Check if a specific model supports PyTorch
bert_support = frameworks_df[frameworks_df["model_type"] == "bert"]
print(f"BERT PyTorch support: {bert_support['pytorch'].values[0]}")

# Find all PyTorch-only models
pt_models = frameworks_df[frameworks_df["pytorch"] == True]["model_type"].tolist()
print(f"PyTorch models: {len(pt_models)}")

# Get processor type for a model
bert_processor = frameworks_df[frameworks_df["model_type"] == "bert"]["processor"].values[0]
print(f"BERT uses: {bert_processor}")  # AutoTokenizer
```

### Example 7: Extracting Model Capabilities

```python
import transformers.models.auto.modeling_auto as modeling_auto
from update_metadata import PIPELINE_TAGS_AND_AUTO_MODELS

# Find all tasks a model supports
def get_model_tasks(model_name: str) -> list[str]:
    tasks = []
    for pipeline_tag, model_mapping, auto_class in PIPELINE_TAGS_AND_AUTO_MODELS:
        if not hasattr(modeling_auto, model_mapping):
            continue
        mapping = getattr(modeling_auto, model_mapping)
        for names in mapping.values():
            if isinstance(names, str):
                names = [names]
            if model_name in names:
                tasks.append(pipeline_tag)
                break
    return tasks

# Check BERT's capabilities
bert_tasks = get_model_tasks("BertForSequenceClassification")
print(bert_tasks)  # ["text-classification"]

bert_mlm_tasks = get_model_tasks("BertForMaskedLM")
print(bert_mlm_tasks)  # ["fill-mask"]
```

### Example 8: Manual Metadata Check

```python
# Simulate what check_pipeline_tags does
import transformers

in_table = {tag: cls for tag, _, cls in PIPELINE_TAGS_AND_AUTO_MODELS}
pipeline_tasks = transformers.pipelines.SUPPORTED_TASKS

missing = []
for key in pipeline_tasks:
    if key not in in_table:
        model = pipeline_tasks[key]["pt"]
        if isinstance(model, (list, tuple)):
            model = model[0]
        model = model.__name__
        if model not in in_table.values():
            missing.append(key)

if missing:
    print(f"Missing pipeline tags: {', '.join(missing)}")
else:
    print("All pipeline tags are defined")
```

### Example 9: Determining Processor Type Logic

```python
from update_metadata import get_frameworks_table
import transformers.models.auto as auto_module

# Logic for determining processor type
def determine_processor(model_type: str) -> str:
    """Determine appropriate processor for a model type."""
    if model_type in auto_module.processing_auto.PROCESSOR_MAPPING_NAMES:
        return "AutoProcessor"
    elif model_type in auto_module.tokenization_auto.TOKENIZER_MAPPING_NAMES:
        return "AutoTokenizer"
    elif model_type in auto_module.image_processing_auto.IMAGE_PROCESSOR_MAPPING_NAMES:
        return "AutoImageProcessor"
    elif model_type in auto_module.feature_extraction_auto.FEATURE_EXTRACTOR_MAPPING_NAMES:
        return "AutoFeatureExtractor"
    else:
        # Default for backward compatibility
        return "AutoTokenizer"

# Test it
print(determine_processor("bert"))  # AutoTokenizer
print(determine_processor("clip"))  # AutoProcessor
print(determine_processor("vit"))   # AutoImageProcessor
print(determine_processor("wav2vec2"))  # AutoFeatureExtractor
```

### Example 10: CI Integration Script

```bash
#!/bin/bash
# Complete CI script for metadata updates

set -e

echo "Checking pipeline tags consistency..."
python utils/update_metadata.py --check-only

if [ "$GITHUB_EVENT_NAME" = "push" ] && [ "$GITHUB_REF" = "refs/heads/main" ]; then
    echo "Main branch push detected, updating metadata..."
    python utils/update_metadata.py \
        --token "$HF_TOKEN" \
        --commit_sha "$GITHUB_SHA"
    echo "Metadata update complete"
else
    echo "Not main branch, skipping metadata update"
fi
```

## Related Pages

- [Notification Service Doc Tests Implementation](/wikis/huggingface_transformers_NotificationServiceDocTests.md)
- [Tests Fetcher Implementation](/wikis/huggingface_transformers_TestsFetcher.md)
