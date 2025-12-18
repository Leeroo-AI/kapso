# create_dummy_models.py - Tiny Model Checkpoint Generator

| Metadata | Value |
|----------|-------|
| **Type** | Implementation |
| **Source File** | `utils/create_dummy_models.py` |
| **Repository** | huggingface/transformers |
| **Domains** | Testing Infrastructure, Model Generation, CI/CD Automation |
| **Last Updated** | 2025-12-18 |
| **Lines of Code** | 1,479 |

---

## Overview

The `create_dummy_models.py` script is an automated utility that creates tiny (minimal parameter) model checkpoints for testing purposes. These tiny models enable fast unit testing, continuous integration, and pipeline validation without requiring full-sized pre-trained models, significantly reducing test execution time and resource consumption.

---

## Description

### Purpose

This script generates lightweight model checkpoints that:
- Have minimal vocabulary sizes (target: 1,024 tokens)
- Use tiny architectural configurations (small hidden sizes, few layers)
- Include all necessary processors (tokenizers, feature extractors, image processors)
- Are functionally equivalent to full models but execute much faster
- Can be uploaded to the Hugging Face Hub for CI/CD usage

### Key Features

1. **Automated Model Generation**: Creates tiny versions of all Transformers model architectures
2. **Processor Conversion**: Reduces tokenizer vocabulary size and adapts feature extractors
3. **Multi-Architecture Support**: Handles 200+ different model architectures
4. **Multiprocessing**: Supports parallel model generation with configurable workers
5. **Hub Integration**: Automatically uploads models to Hugging Face Hub
6. **Comprehensive Reporting**: Generates detailed JSON and text reports on success/failure
7. **Composite Model Support**: Handles encoder-decoder and vision-text dual models

### Architecture

The script follows a pipeline approach:

1. **Discovery**: Find all model configurations and their processors
2. **Configuration**: Extract tiny configs from model testers
3. **Processor Building**: Create or download necessary processors
4. **Processor Conversion**: Reduce vocabulary size and adapt parameters
5. **Model Building**: Instantiate tiny models with converted processors
6. **Serialization**: Save models and processors to disk
7. **Upload**: Push to Hugging Face Hub (optional)
8. **Reporting**: Generate comprehensive success/failure reports

---

## Code Reference

### Main Entry Point

```python
def create_tiny_models(
    output_path,
    all,
    model_types,
    models_to_skip,
    no_check,
    upload,
    organization,
    token,
    num_workers=1,
):
    """Create tiny models for testing purposes.

    Args:
        output_path: Directory to store generated models
        all: Whether to create models for all architectures
        model_types: Specific model types to create (comma-separated)
        models_to_skip: Model classes to skip
        no_check: Skip architecture validation
        upload: Whether to upload to Hub
        organization: Hub organization name
        token: HuggingFace Hub authentication token
        num_workers: Number of parallel workers
    """
```

### Core Functions

#### Configuration Generation

```python
def get_tiny_config(config_class, model_class=None, **model_tester_kwargs):
    """Retrieve a tiny configuration from `config_class` using each model's `ModelTester`.

    Args:
        config_class: Subclass of `PreTrainedConfig`.
        model_class: Optional specific model class
        **model_tester_kwargs: Additional arguments for model tester

    Returns:
        An instance of `config_class` with tiny hyperparameters
    """
    model_type = config_class.model_type

    # Find the model tester class
    module_name = model_type_to_module_name(model_type)
    test_file = os.path.join("tests", "models", module_name, f"test_modeling_{modeling_name}.py")
    models_to_model_testers = get_model_to_tester_mapping(test_file)

    # Instantiate tester and get config
    model_tester = model_tester_class(parent=None, **model_tester_kwargs)

    if hasattr(model_tester, "get_pipeline_config"):
        config = model_tester.get_pipeline_config()
    elif hasattr(model_tester, "prepare_config_and_inputs"):
        config = model_tester.prepare_config_and_inputs()[0]
    elif hasattr(model_tester, "get_config"):
        config = model_tester.get_config()

    return config
```

#### Processor Building

```python
def build_processor(config_class, processor_class, allow_no_checkpoint=False):
    """Create a processor for `processor_class`.

    Args:
        config_class: Configuration class to determine checkpoint
        processor_class: Processor class to instantiate
        allow_no_checkpoint: Whether to create processor without checkpoint

    Returns:
        Processor instance or None if failed
    """
    checkpoint = get_checkpoint_from_config_class(config_class)

    processor = None
    try:
        processor = processor_class.from_pretrained(checkpoint)
    except Exception as e:
        logger.error(f"{e.__class__.__name__}: {e}")

    # Try alternative strategies if direct loading fails
    if processor is None:
        # Attempt 1: Get processor class from config
        config = AutoConfig.from_pretrained(checkpoint)
        tokenizer_class = config.tokenizer_class
        new_processor_class = getattr(transformers_module, tokenizer_class)
        processor = build_processor(config_class, new_processor_class)

    # Attempt 2: Build from components for ProcessorMixin
    if processor is None and issubclass(processor_class, ProcessorMixin):
        attrs = {}
        for attr_name in processor_class.get_attributes():
            attr_class_names = getattr(processor_class, f"{attr_name}_class")
            # Build each component recursively

    return processor
```

#### Processor Conversion

```python
def convert_tokenizer(tokenizer_fast: PreTrainedTokenizerFast):
    """Train a new tokenizer with reduced vocabulary size.

    Args:
        tokenizer_fast: Fast tokenizer to convert

    Returns:
        New tokenizer with vocabulary size ~1024
    """
    new_tokenizer = tokenizer_fast.train_new_from_iterator(
        data["training_ds"]["text"], TARGET_VOCAB_SIZE, show_progress=False
    )

    # Validation: ensure it works
    if not isinstance(new_tokenizer, LayoutLMv3TokenizerFast):
        new_tokenizer(data["testing_ds"]["text"])

    return new_tokenizer

def convert_feature_extractor(feature_extractor, tiny_config):
    """Adapt feature extractor to use smaller image sizes.

    Args:
        feature_extractor: Original feature extractor
        tiny_config: Tiny configuration with smaller parameters

    Returns:
        Updated feature extractor
    """
    to_convert = False
    kwargs = {}

    if hasattr(tiny_config, "image_size"):
        kwargs["size"] = tiny_config.image_size
        kwargs["crop_size"] = tiny_config.image_size
        to_convert = True

    if to_convert:
        feature_extractor = feature_extractor.__class__(**kwargs)

    # Validate image size is not too large (max 64x64)
    if isinstance(feature_extractor, BaseImageProcessor):
        largest_image_size = max(feature_extractor.size.values())
        if largest_image_size > 64:
            raise ValueError(f"Image size too large: {largest_image_size}")

    return feature_extractor

def convert_processors(processors, tiny_config, output_folder, result):
    """Change processors to work with smaller inputs.

    Args:
        processors: List of processors to convert
        tiny_config: Tiny configuration
        output_folder: Where to save converted processors
        result: Dictionary to store warnings/errors

    Returns:
        List of converted processors
    """
```

#### Model Building

```python
def build_model(model_arch, tiny_config, output_dir):
    """Create and save a model for `model_arch`.

    Args:
        model_arch: Model architecture class
        tiny_config: Tiny configuration
        output_dir: Output directory

    Returns:
        Created model instance
    """
    checkpoint_dir = get_checkpoint_dir(output_dir, model_arch)

    # Copy processors to model directory
    processor_output_dir = os.path.join(output_dir, "processors")
    if os.path.isdir(processor_output_dir):
        shutil.copytree(processor_output_dir, checkpoint_dir, dirs_exist_ok=True)

    tiny_config = copy.deepcopy(tiny_config)

    # Special handling for causal LMs
    if any(model_arch.__name__.endswith(x) for x in ["ForCausalLM", "LMHeadModel"]):
        tiny_config.is_encoder_decoder = False
        tiny_config.is_decoder = True

    model = model_arch(config=tiny_config)
    model.save_pretrained(checkpoint_dir)
    model.from_pretrained(checkpoint_dir)

    return model
```

#### Main Build Pipeline

```python
def build(config_class, models_to_create, output_dir):
    """Create all models for a certain model type.

    Args:
        config_class: PreTrainedConfig subclass
        models_to_create: Dict of processor/model classes to create
        output_dir: Directory to save checkpoints

    Returns:
        Dictionary with results for processors and models
    """
    # Load training dataset
    if data["training_ds"] is None or data["testing_ds"] is None:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        data["training_ds"] = ds["train"]
        data["testing_ds"] = ds["test"]

    result = {k: {} for k in models_to_create}
    result["error"] = None
    result["warnings"] = []

    # Step 1: Build processors
    processor_classes = models_to_create["processor"]
    for processor_class in processor_classes:
        processor = build_processor(config_class, processor_class, allow_no_checkpoint=True)
        if processor is not None:
            result["processor"][processor_class] = processor

    # Step 2: Get tiny config
    tiny_config = get_tiny_config(config_class)

    # Step 3: Convert processors
    processors = list(result["processor"].values())
    processor_output_folder = os.path.join(output_dir, "processors")
    processors = convert_processors(processors, tiny_config, processor_output_folder, result)

    # Step 4: Update config with processor info
    config_overrides = get_config_overrides(config_class, processors)
    for k, v in config_overrides.items():
        if hasattr(tiny_config, k):
            setattr(tiny_config, k, v)

    # Step 5: Build models
    for pytorch_arch in models_to_create["pytorch"]:
        model = build_model(pytorch_arch, tiny_config, output_dir=output_dir)
        result["pytorch"][pytorch_arch.__name__]["model"] = model.__class__.__name__
        result["pytorch"][pytorch_arch.__name__]["checkpoint"] = get_checkpoint_dir(output_dir, pytorch_arch)

    return result
```

#### Composite Models

```python
def build_composite_models(config_class, output_dir):
    """Build encoder-decoder and vision-text dual encoder models.

    Supports:
    - EncoderDecoderModel
    - VisionEncoderDecoderModel
    - SpeechEncoderDecoderModel
    - VisionTextDualEncoderModel
    """
    # Build encoder and decoder separately
    encoder_output_dir = os.path.join(tmpdir, "encoder")
    build(encoder_config_class, models_to_create, encoder_output_dir)

    decoder_output_dir = os.path.join(tmpdir, "decoder")
    build(decoder_config_class, models_to_create, decoder_output_dir)

    # Combine into composite model
    model = model_class.from_encoder_decoder_pretrained(
        encoder_path, decoder_path, decoder_config=decoder_config
    )

    model.save_pretrained(model_path)
```

#### Hub Upload

```python
def upload_model(model_dir, organization, token):
    """Upload the tiny models to Hugging Face Hub.

    Args:
        model_dir: Local directory containing model
        organization: Hub organization name
        token: Authentication token
    """
    arch_name = model_dir.split(os.path.sep)[-1]
    repo_name = f"tiny-random-{arch_name}"
    repo_id = f"{organization}/{repo_name}"

    # Create or update repo
    create_repo(repo_id=repo_id, exist_ok=False, repo_type="model", token=token)

    # Upload folder
    create_pr = repo_exist  # Open PR on existing repo
    commit = upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Update tiny models for {arch_name}",
        create_pr=create_pr,
        token=token,
    )
```

#### Reporting

```python
def build_tiny_model_summary(results, organization=None, token=None):
    """Build a summary of created models.

    Returns:
        Dictionary mapping architecture name to:
        - tokenizer_classes: List of tokenizer types
        - processor_classes: List of processor types
        - model_classes: List of model types
        - sha: Commit hash on Hub (if uploaded)
    """

def build_failed_report(results, include_warning=True):
    """Extract failures and warnings from results."""

def build_simple_report(results):
    """Build simple text report: 'ArchName: OK' or 'ArchName: Error'."""
```

### Important Constants

```python
TARGET_VOCAB_SIZE = 1024  # Target vocabulary size for converted tokenizers

# Composite model mappings
COMPOSITE_MODELS = {
    "EncoderDecoderModel": "EncoderDecoderModel-bert-bert",
    "SpeechEncoderDecoderModel": "SpeechEncoderDecoderModel-wav2vec2-bert",
    "VisionEncoderDecoderModel": "VisionEncoderDecoderModel-vit-gpt2",
    "VisionTextDualEncoderModel": "VisionTextDualEncoderModel-vit-bert",
}

# Models that cannot be converted to tiny versions
UNCONVERTIBLE_MODEL_ARCHITECTURES = {
    "BertGenerationEncoder",
    "BertGenerationDecoder",
    "MT5Model",
    "RetriBertModel",
    # ... (40+ entries)
}
```

---

## I/O Contract

### Input Specifications

| Input Type | Description | Format | Required |
|------------|-------------|--------|----------|
| Model Types | Comma-separated list of model types | String | Conditional* |
| Output Path | Directory for generated models | Path | Yes |
| Organization | Hub organization name | String | If uploading |
| Token | HuggingFace authentication token | String | If uploading |
| Workers | Number of parallel workers | Integer | No (default: 1) |

*Either `--all` or `--model_types` must be specified

### Output Specifications

| Output Type | Description | Format | Location |
|-------------|-------------|--------|----------|
| Model Checkpoints | Tiny model weights and config | PyTorch .bin/.safetensors | `{output_path}/{model_type}/{architecture}/` |
| Processors | Tokenizers, feature extractors | JSON, vocab files | `{output_path}/{model_type}/{architecture}/` |
| Summary Report | JSON with all created models | JSON | `{output_path}/reports/tiny_model_summary.json` |
| Detailed Report | Full creation results | JSON | `{output_path}/reports/tiny_model_creation_report.json` |
| Failed Report | Errors and warnings | JSON | `{output_path}/reports/failed_report.json` |
| Simple Report | One-line status per model | Text | `{output_path}/reports/simple_report.txt` |
| Upload Failures | Failed uploads | JSON | `{output_path}/reports/failed_uploads.json` |

### Report Structure

**tiny_model_summary.json**:
```json
{
  "BertModel": {
    "tokenizer_classes": ["BertTokenizer", "BertTokenizerFast"],
    "processor_classes": [],
    "model_classes": ["BertModel", "BertForMaskedLM", "BertForSequenceClassification"],
    "sha": "abc123..."
  }
}
```

**simple_report.txt**:
```
BertModel: OK
BertForMaskedLM: OK
GPT2Model: OK
T5Model: Failed to create: No processor could be built
```

---

## Usage Examples

### Create Tiny Models for All Architectures

```bash
python utils/create_dummy_models.py \
    --all \
    ./output \
    --num_workers 4
```

### Create Specific Model Types

```bash
python utils/create_dummy_models.py \
    --model_types bert,gpt2,t5 \
    ./output
```

### Create and Upload to Hub

```bash
python utils/create_dummy_models.py \
    --all \
    --upload \
    --organization hf-internal-testing \
    --token hf_xxx \
    ./output
```

### Skip Specific Models

```bash
python utils/create_dummy_models.py \
    --model_types bert \
    --models_to_skip BertForQuestionAnswering,BertForTokenClassification \
    ./output
```

### Parallel Processing

```bash
# Use 8 workers for faster generation
python utils/create_dummy_models.py \
    --all \
    --num_workers 8 \
    ./output
```

### Check Reports After Generation

```python
import json

# Load summary
with open("./output/reports/tiny_model_summary.json") as f:
    summary = json.load(f)

print(f"Total architectures created: {len(summary)}")
for arch_name, info in summary.items():
    print(f"{arch_name}: {len(info['model_classes'])} models")

# Check failures
with open("./output/reports/failed_report.json") as f:
    failures = json.load(f)

if failures:
    print(f"Failed configs: {len(failures)}")
    for config_name, error_info in failures.items():
        print(f"  {config_name}: {error_info.get('error', 'Unknown error')}")
```

### Use Generated Models in Tests

```python
from transformers import AutoModel, AutoTokenizer

# Use tiny model instead of full model
model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-BertModel")
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-BertModel")

# Fast inference for testing
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)

print(f"Model parameters: {model.num_parameters():,}")  # Much smaller!
print(f"Vocab size: {tokenizer.vocab_size}")  # ~1024
```

### Update Tiny Model Summary File

After generating new models, update the test utilities:

```bash
# This updates tests/utils/tiny_model_summary.json
python -c "
from utils.create_dummy_models import update_tiny_model_summary_file
update_tiny_model_summary_file('./output/reports')
"

# Then copy to tests directory
cp ./output/reports/updated_tiny_model_summary.json tests/utils/tiny_model_summary.json
```

---

## Related Pages

<!-- Links to related documentation pages -->

---

## Implementation Notes

### Vocabulary Reduction Strategy

The script uses the fast tokenizer's `train_new_from_iterator` method to create a new tokenizer with reduced vocabulary:

```python
new_tokenizer = tokenizer_fast.train_new_from_iterator(
    data["training_ds"]["text"],  # WikiText-2 dataset
    TARGET_VOCAB_SIZE,            # 1024 tokens
    show_progress=False
)
```

This ensures:
- Vocabulary is trained on real text data
- Special tokens are preserved
- Tokenizer remains functional for text processing

### Image Size Constraints

Feature extractors are validated to ensure image sizes don't exceed 64x64 (with exceptions for specific models):

```python
largest_image_size = max(feature_extractor.size.values())
if largest_image_size > 64:
    allowed_models = ("deformable_detr", "flava", "grounding_dino")
    if not any(model_name in tiny_config.model_type for model_name in allowed_models):
        raise ValueError(f"Image size too large: {largest_image_size}")
```

This prevents slow CI tests due to large image processing.

### Config Override Mechanism

The script updates tiny configs with tokenizer-specific values:

```python
config_overrides = get_config_overrides(config_class, processors)
# Example overrides:
# - vocab_size: len(tokenizer)
# - bos_token_id: tokenizer.bos_token_id
# - eos_token_id: tokenizer.eos_token_id
# - pad_token_id: tokenizer.pad_token_id
```

### Multiprocessing Safety

Uses `spawn` method to avoid hanging:

```python
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
```

And disables tokenizer parallelism:

```python
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

### Error Handling Strategy

The script implements comprehensive error handling:
- Collects warnings instead of failing immediately
- Attempts multiple strategies to build processors
- Falls back to original processors if conversion fails
- Generates detailed error reports for debugging

### Hub Repository Naming

Tiny models are uploaded with consistent naming:
- Standard: `tiny-random-{ArchitectureName}`
- Composite: `tiny-random-{CompositeType}-{encoder}-{decoder}`

Examples:
- `tiny-random-BertModel`
- `tiny-random-EncoderDecoderModel-bert-bert`
- `tiny-random-VisionEncoderDecoderModel-vit-gpt2`
