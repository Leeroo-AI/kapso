# Implementation: peft_types.py

## File Information
- **Path**: `/tmp/praxium_repo_35tl5_4u/src/peft/utils/peft_types.py`
- **Size**: 183 lines
- **Module**: `peft.utils.peft_types`
- **Description**: Core type enumerations for PEFT adapters and tasks

## Overview

This module defines the fundamental enumerations that classify PEFT (Parameter-Efficient Fine-Tuning) methods and task types. It provides a centralized type system that enables dynamic adapter registration, configuration validation, and model routing throughout the PEFT library.

## Core Enumerations

### PeftType

**Purpose**: Enumeration of all supported PEFT adapter types

```python
class PeftType(str, enum.Enum):
    """
    Enum class for the different types of adapters in PEFT.

    Inherits from both str and enum.Enum for:
    - String comparisons
    - JSON serialization
    - Type safety
    """
```

#### Supported PEFT Types (30+ methods)

**Prompt-Based Methods**:
- `PROMPT_TUNING`: Prefix-based prompt tuning
- `MULTITASK_PROMPT_TUNING`: Multi-task prompt tuning with task-specific parameters
- `P_TUNING`: P-tuning v2 (prompt encoder)
- `PREFIX_TUNING`: Prefix tuning for key/value injection
- `ADAPTION_PROMPT`: LLaMA-Adapter style gated attention prompts

**Low-Rank Adaptation Methods**:
- `LORA`: Low-Rank Adaptation (most popular)
- `ADALORA`: Adaptive LoRA with dynamic rank allocation
- `XLORA`: Mixture of LoRA experts
- `VBLORA`: Vector Bank LoRA
- `DELORA`: Delta-based LoRA
- `GRALORA`: Gradient-based LoRA

**Matrix Factorization Methods**:
- `LOHA`: Low-Rank Hadamard Product Adaptation
- `LOKR`: Low-Rank Kronecker Product Adaptation
- `POLY`: Polynomial adaptation

**Orthogonal Methods**:
- `OFT`: Orthogonal Finetuning
- `BOFT`: Butterfly Orthogonal Finetuning

**Activation-Based Methods**:
- `IA3`: Infused Adapter by Inhibiting and Amplifying Inner Activations

**Layer Normalization Methods**:
- `LN_TUNING`: Layer Normalization tuning

**Frequency-Domain Methods**:
- `FOURIERFT`: Fourier Transform-based finetuning
- `WAVEFT`: Wavelet Transform-based finetuning

**Shared Parameter Methods**:
- `VERA`: Vector-based Random Matrix Adaptation
- `SHIRA`: Shared Inference-time Random Adaptation

**Hybrid Methods**:
- `HRA`: Hadamard Rotation Adaptation
- `BONE`: Block-wise Orthonormal Adaptation
- `MISS`: Mixed Integer Structured Sparse Adaptation
- `RANDLORA`: Randomized LoRA
- `C3A`: Coordinate-wise Convolutional Adaptation
- `ROAD`: Robust Adaptation
- `OSF`: Orthogonal Subspace Factorization
- `CPT`: Context-aware Prompt Tuning
- `TRAINABLE_TOKENS`: Direct token embedding tuning

### TaskType

**Purpose**: Enumeration of supported downstream task types

```python
class TaskType(str, enum.Enum):
    """
    Enum class for the different types of tasks supported by PEFT.

    Determines:
    - Model head architecture
    - Forward pass behavior
    - Loss computation strategy
    """
```

#### Supported Task Types

**Text Classification**:
```python
SEQ_CLS = "SEQ_CLS"
# Sequence-level classification
# Examples: Sentiment analysis, topic classification
# Head: Linear layer on [CLS] token or pooled output
```

**Sequence-to-Sequence**:
```python
SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
# Encoder-decoder language modeling
# Examples: Translation, summarization
# Architecture: T5, BART, mT5
```

**Causal Language Modeling**:
```python
CAUSAL_LM = "CAUSAL_LM"
# Autoregressive generation
# Examples: Text generation, completion
# Architecture: GPT, LLaMA, Mistral
```

**Token Classification**:
```python
TOKEN_CLS = "TOKEN_CLS"
# Token-level classification
# Examples: NER, POS tagging
# Head: Linear layer per token
```

**Question Answering**:
```python
QUESTION_ANS = "QUESTION_ANS"
# Span extraction from context
# Examples: SQuAD, extractive QA
# Head: Start/end position classifiers
```

**Feature Extraction**:
```python
FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
# Provides hidden states as embeddings
# Examples: Sentence embeddings, retrieval
# Output: Raw hidden states without task-specific head
```

## Dynamic Registration System

### register_peft_method()

**Purpose**: Dynamically registers new PEFT methods at runtime

```python
def register_peft_method(
    *,
    name: str,                              # Unique identifier (lowercase)
    config_cls,                             # Configuration class
    model_cls,                              # Model/tuner class
    prefix: Optional[str] = None,           # State dict prefix (default: name + "_")
    is_mixed_compatible=False               # PeftMixedModel compatibility
) -> None:
    """
    Registers a finetuning method to be available in PEFT.

    Validation:
    - Name must be lowercase
    - Name must not end with '_'
    - Name must exist in PeftType enum
    - No duplicate names/prefixes allowed
    - Model prefix must match provided prefix
    """
```

#### Registration Process

**1. Validation**:
```python
# Name format checks
if name.endswith("_"):
    raise ValueError(f"Name cannot end with '_', got {name}")

if not name.islower():
    raise ValueError(f"Name must be lowercase, got {name}")

# Enum existence check
if name.upper() not in list(PeftType):
    raise ValueError(f"Unknown PEFT type {name.upper()}, add to PeftType enum first")
```

**2. Prefix Assignment**:
```python
# Default prefix: name + "_"
if prefix is None:
    prefix = name + "_"

# Consistency check
model_cls_prefix = getattr(model_cls, "prefix", None)
if model_cls_prefix and model_cls_prefix != prefix:
    raise ValueError(f"Inconsistent prefixes: '{prefix}' vs '{model_cls_prefix}'")
```

**3. Mapping Registration**:
```python
from peft.mapping import (
    PEFT_TYPE_TO_CONFIG_MAPPING,
    PEFT_TYPE_TO_MIXED_MODEL_MAPPING,
    PEFT_TYPE_TO_PREFIX_MAPPING,
    PEFT_TYPE_TO_TUNER_MAPPING,
)

peft_type = getattr(PeftType, name.upper())

# Register in global mappings
PEFT_TYPE_TO_PREFIX_MAPPING[peft_type] = prefix
PEFT_TYPE_TO_CONFIG_MAPPING[peft_type] = config_cls
PEFT_TYPE_TO_TUNER_MAPPING[peft_type] = model_cls

if is_mixed_compatible:
    PEFT_TYPE_TO_MIXED_MODEL_MAPPING[peft_type] = model_cls
```

#### Global Mappings Updated

**PEFT_TYPE_TO_CONFIG_MAPPING**:
- Maps PeftType → Configuration class
- Used by `PeftConfig.from_pretrained()`
- Example: `PeftType.LORA → LoraConfig`

**PEFT_TYPE_TO_TUNER_MAPPING**:
- Maps PeftType → Model/Tuner class
- Used by `get_peft_model()`
- Example: `PeftType.LORA → LoraModel`

**PEFT_TYPE_TO_PREFIX_MAPPING**:
- Maps PeftType → State dict prefix
- Used for adapter weight isolation
- Example: `PeftType.LORA → "lora_"`

**PEFT_TYPE_TO_MIXED_MODEL_MAPPING**:
- Maps PeftType → Mixed model compatible tuners
- Only applicable for tuners supporting multi-adapter inference
- Example: `PeftType.LORA → LoraModel`

## Usage Patterns

### Configuration Type Detection

```python
from peft import PeftConfig

# Automatic type detection from config
config = PeftConfig.from_pretrained("adapter_checkpoint")
if config.peft_type == PeftType.LORA:
    print("LoRA adapter detected")
elif config.peft_type == PeftType.PREFIX_TUNING:
    print("Prefix tuning adapter detected")
```

### Task-Specific Model Creation

```python
from peft import LoraConfig, TaskType, get_peft_model

# Causal LM configuration
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(base_model, config)
```

### Custom PEFT Method Registration

```python
from peft.utils import register_peft_method, PeftType
from peft.config import PeftConfig

# 1. Add enum entry (one-time, usually in library)
# PeftType.MY_METHOD = "MY_METHOD"  # Already defined

# 2. Define config and model classes
class MyMethodConfig(PeftConfig):
    def __post_init__(self):
        self.peft_type = PeftType.MY_METHOD

class MyMethodModel(BaseTuner):
    prefix: str = "my_method_"
    # Implementation...

# 3. Register
register_peft_method(
    name="my_method",
    config_cls=MyMethodConfig,
    model_cls=MyMethodModel,
    is_mixed_compatible=True
)

# 4. Now usable like built-in methods
config = MyMethodConfig(...)
model = get_peft_model(base_model, config)
```

## Design Principles

### String Enum Hybrid

```python
class PeftType(str, enum.Enum):
    LORA = "LORA"
```

**Benefits**:
1. **Type Safety**: Enum provides compile-time checking
2. **String Compatibility**: Can be compared to strings
3. **JSON Serialization**: Automatically serializes to string
4. **Readability**: `PeftType.LORA` more explicit than `"LORA"`

### Centralized Type System

**Advantages**:
- Single source of truth for supported methods
- Easy to add new methods (extend enum)
- Type checking across entire codebase
- Self-documenting (all methods listed)

**Trade-offs**:
- Requires enum update for new methods
- Tight coupling with mapping system
- Cannot add methods without code changes

### Dynamic Registration

**Use Cases**:
1. **Third-party Extensions**: External packages can register custom methods
2. **Research Prototypes**: Quick experimentation without PR
3. **Proprietary Methods**: Private implementations without forking

**Safety Measures**:
- Validation ensures consistency
- No duplicate names/prefixes
- Requires enum entry (prevents typos)

## Configuration Integration

### PeftConfig Base Class Integration

```python
@dataclass
class LoraConfig(PeftConfig):
    # ... config fields ...

    def __post_init__(self):
        self.peft_type = PeftType.LORA  # Sets type from enum
```

### Auto-detection from Saved Config

```python
# adapter_config.json:
{
    "peft_type": "LORA",
    "r": 16,
    ...
}

# Loading:
config = PeftConfig.from_pretrained("path")
# config.peft_type automatically converted to PeftType.LORA enum
```

## Validation and Error Handling

### Common Validation Errors

```python
# Uppercase name
register_peft_method(name="MyMethod", ...)
# ValueError: Name must be lowercase, got MyMethod

# Name with trailing underscore
register_peft_method(name="my_method_", ...)
# ValueError: Name cannot end with '_', got my_method_

# Unregistered enum
register_peft_method(name="unknown_method", ...)
# ValueError: Unknown PEFT type UNKNOWN_METHOD, add to PeftType enum first

# Duplicate registration
register_peft_method(name="lora", ...)
# KeyError: There is already PEFT method called 'lora'

# Prefix conflict
register_peft_method(name="my_method", prefix="lora_", ...)
# KeyError: There is already a prefix called 'lora_'
```

## Task Type Usage

### Model Head Selection

```python
def get_model_for_task(base_model, config):
    if config.task_type == TaskType.SEQ_CLS:
        return AutoModelForSequenceClassification(base_model)
    elif config.task_type == TaskType.CAUSAL_LM:
        return AutoModelForCausalLM(base_model)
    elif config.task_type == TaskType.SEQ_2_SEQ_LM:
        return AutoModelForSeq2SeqLM(base_model)
    # ...
```

### Prompt Tuning Compatibility

```python
if config.task_type == TaskType.SEQ_2_SEQ_LM:
    # Need prompts for both encoder and decoder
    num_transformer_submodules = 2
else:
    # Only decoder needs prompts
    num_transformer_submodules = 1
```

## Extensibility Patterns

### Plugin Architecture

```python
# In external package
from peft.utils import PeftType, register_peft_method

# Extend enum (if contributing to PEFT)
# Or use existing enum value if already added

register_peft_method(
    name="custom_adapter",
    config_cls=CustomAdapterConfig,
    model_cls=CustomAdapterModel,
    prefix="custom_",
    is_mixed_compatible=False
)
```

### Compatibility Flags

**is_mixed_compatible**:
- `True`: Supports multiple adapters simultaneously
- `False`: Single adapter only
- Determines PeftMixedModel eligibility

## Cross-References

- **Used By**: All PEFT config classes, `get_peft_model()`, `PeftConfig.from_pretrained()`
- **Related Modules**: `peft.mapping`, `peft.config`, `peft.peft_model`
- **Extended By**: Custom PEFT method implementations via `register_peft_method()`
