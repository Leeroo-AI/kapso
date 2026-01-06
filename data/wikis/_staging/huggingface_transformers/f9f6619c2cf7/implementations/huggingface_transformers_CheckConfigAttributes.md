{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Testing]], [[domain::Quality Assurance]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Validation utility that ensures all configuration class attributes defined in `__init__` are actually used in the corresponding modeling files.

=== Description ===
The check_config_attributes.py script is a 548-line quality assurance tool designed to prevent dead code in model configurations. It performs static code analysis to verify that every parameter defined in a configuration class's `__init__` method is referenced somewhere in the model's implementation files.

The script works by:
* Extracting all parameters from configuration class `__init__` signatures
* Scanning all modeling files in the same directory for usage of those parameters
* Detecting patterns like `config.attribute`, `getattr(config, "attribute")`, or `config.get_text_config().attribute`
* Accounting for attribute name mappings via `attribute_map`
* Allowing special cases for common attributes and model-specific exceptions
* Reporting any unused attributes as validation errors

The script includes an extensive allowlist (`SPECIAL_CASES_TO_ALLOW`) for legitimate cases where attributes may not appear directly in modeling files, such as:
* Attributes used only during training or generation
* Attributes consumed by parent classes or used internally in the config
* Legacy attributes maintained for backward compatibility
* Attributes passed to external libraries (like timm, DETR, etc.)

This ensures that configuration files remain clean and all parameters serve a documented purpose in the model implementation.

=== Usage ===
Use this script when:
* Adding new parameters to configuration classes (validate they're used)
* Refactoring model implementations (ensure removed code doesn't leave orphaned configs)
* Running CI checks to maintain code quality
* Reviewing pull requests that modify configuration classes
* Cleaning up legacy code to remove unused parameters
* Ensuring consistency between configuration and implementation

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers huggingface_transformers]
* '''File:''' utils/check_config_attributes.py

=== Signature ===
<syntaxhighlight lang="python">
# Main validation function
def check_config_attributes()

# Core checking logic
def check_config_attributes_being_used(config_class) -> list[str]
def check_attribute_being_used(config_class, attributes, default_value,
                                source_strings) -> bool

# Constants
CONFIG_MAPPING: dict  # Mapping of model types to config classes
SPECIAL_CASES_TO_ALLOW: dict  # Allowed exceptions by config class
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# This is a standalone script, typically run from command line
# But functions can be imported if needed:
from utils.check_config_attributes import (
    check_config_attributes,
    check_config_attributes_being_used,
    check_attribute_being_used,
)
</syntaxhighlight>

== I/O Contract ==

=== Input Sources ===
{| class="wikitable"
! Source !! Type !! Description
|-
| CONFIG_MAPPING || dict || Mapping from transformers.models.auto.configuration_auto
|-
| Configuration files || Python || Files named configuration_*.py in model directories
|-
| Modeling files || Python || Files named modeling_*.py in model directories
|}

=== Detection Patterns ===
{| class="wikitable"
! Pattern !! Example !! Description
|-
| Direct access || config.hidden_size || Direct attribute access
|-
| getattr || getattr(config, "hidden_size") || Dynamic attribute access with getattr
|-
| Self config || getattr(self.config, "hidden_size") || Access via self.config
|-
| Text config || config.get_text_config().hidden_size || VLM text config access
|-
| Multiline getattr || getattr(\n    config,\n    "hidden_size"\n) || getattr spanning lines
|}

=== Output Format ===
{| class="wikitable"
! Output !! Type !! Description
|-
| Error message || str || List of configs with unused attributes
|-
| Exit code || int || 0 for success, raises ValueError on failure
|}

=== Allowed Attributes (Always Pass) ===
{| class="wikitable"
! Attribute !! Category !! Reason
|-
| initializer_range || Common || Model weight initialization
|-
| is_encoder_decoder || Special || Architecture flag (when True)
|-
| tie_word_embeddings || Special || Embedding tying (when False)
|-
| use_cache || Common || KV cache control
|-
| rope_theta || Common || RoPE parameter (may be used indirectly)
|-
| *_token_id || Pattern || All token ID attributes
|-
| image_token_id || VLM || Vision-language model tokens
|-
| backbone* || Backbone || Backbone loading parameters
|}

== Usage Examples ==

=== Running from Command Line ===
<syntaxhighlight lang="python">
# Run validation on all configuration classes
python utils/check_config_attributes.py

# If validation fails, output will show:
# ValueError: The following configuration classes contain unused attributes
# in the corresponding modeling files:
# BertConfig: ['some_unused_param']
# GPT2Config: ['another_unused_param']
</syntaxhighlight>

=== Integration in CI Pipeline ===
<syntaxhighlight lang="python">
# In GitHub Actions or CircleCI
- name: Check config attributes
  run: |
    python utils/check_config_attributes.py
  # This will fail the build if unused attributes are found
</syntaxhighlight>

=== Programmatic Usage ===
<syntaxhighlight lang="python">
from transformers import BertConfig
from utils.check_config_attributes import check_config_attributes_being_used

# Check a specific configuration class
unused_attrs = check_config_attributes_being_used(BertConfig)

if unused_attrs:
    print(f"BertConfig has unused attributes: {unused_attrs}")
else:
    print("All BertConfig attributes are used!")
</syntaxhighlight>

=== Checking Attribute Usage Directly ===
<syntaxhighlight lang="python">
import os
from transformers import GPT2Config
from utils.check_config_attributes import check_attribute_being_used

# Get modeling file sources
model_dir = os.path.dirname(GPT2Config.__module__)
modeling_files = [f for f in os.listdir(model_dir) if f.startswith("modeling_")]

# Read source code
sources = []
for f in modeling_files:
    with open(os.path.join(model_dir, f)) as fp:
        sources.append(fp.read())

# Check if specific attribute is used
is_used = check_attribute_being_used(
    config_class=GPT2Config,
    attributes=["n_embd"],  # or attribute variants
    default_value=768,
    source_strings=sources
)

print(f"'n_embd' is used: {is_used}")
</syntaxhighlight>

=== Adding Special Cases ===
<syntaxhighlight lang="python">
# When adding a new model with legitimate unused attributes
# Update SPECIAL_CASES_TO_ALLOW in the script:

SPECIAL_CASES_TO_ALLOW = {
    "MyNewModelConfig": [
        "special_param",  # Used internally to compute other params
        "legacy_param",   # Kept for backward compatibility
    ],
    # Or allow all attributes for a config:
    "ComplexModelConfig": True,
}
</syntaxhighlight>

=== Understanding Attribute Maps ===
<syntaxhighlight lang="python">
# Some configs use attribute_map for aliasing
class MyConfig(PretrainedConfig):
    attribute_map = {
        "hidden_size": "n_embd",  # n_embd is the actual parameter
        "num_attention_heads": "n_head",
    }

    def __init__(self, n_embd=768, n_head=12, **kwargs):
        # Either config.hidden_size or config.n_embd can be used in modeling files
        super().__init__(**kwargs)
        self.n_embd = n_embd
        self.n_head = n_head

# The checker will accept if EITHER name is found in modeling files
</syntaxhighlight>

=== Example Validation Pass ===
<syntaxhighlight lang="python">
# modeling_bert.py
class BertModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        # Uses config.hidden_size ✓
        self.encoder = BertEncoder(config)
        # Uses config.num_hidden_layers ✓

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        self.word_embeddings = nn.Embedding(
            config.vocab_size,      # ✓
            config.hidden_size,     # ✓
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # ✓

# All config attributes referenced in modeling file → validation passes
</syntaxhighlight>

=== Example Validation Failure ===
<syntaxhighlight lang="python">
# configuration_mymodel.py
class MyModelConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=768,
        num_layers=12,
        unused_param=10,  # ❌ Never used in modeling files
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.unused_param = unused_param

# modeling_mymodel.py
class MyModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = nn.Linear(config.hidden_size, config.hidden_size)
        # config.unused_param is never referenced ❌

# Running check_config_attributes.py will raise:
# ValueError: The following configuration classes contain unused attributes:
# MyModelConfig: ['unused_param']
</syntaxhighlight>

=== Handling Complex Cases ===
<syntaxhighlight lang="python">
# Case 1: Attribute used to derive other attributes
class MyConfig(PretrainedConfig):
    def __init__(self, expand=2, hidden_size=768, **kwargs):
        super().__init__(**kwargs)
        self.expand = expand
        # Used internally to set intermediate_size
        self.intermediate_size = hidden_size * expand

# Add to SPECIAL_CASES_TO_ALLOW:
# "MyConfig": ["expand"]

# Case 2: Attribute used only in training (not in modeling)
class TrainingConfig(PretrainedConfig):
    def __init__(self, ignore_value=-100, **kwargs):
        super().__init__(**kwargs)
        self.ignore_value = ignore_value  # Used in loss computation

# Add to SPECIAL_CASES_TO_ALLOW:
# "TrainingConfig": ["ignore_value"]
</syntaxhighlight>

=== Running for Specific Model ===
<syntaxhighlight lang="python">
# To check only a specific model during development
from transformers import AutoConfig
from utils.check_config_attributes import check_config_attributes_being_used

config_class = AutoConfig.from_pretrained("bert-base-uncased").__class__
unused = check_config_attributes_being_used(config_class)

if unused:
    print(f"Found unused attributes in {config_class.__name__}: {unused}")
    print("Consider removing these or adding to SPECIAL_CASES_TO_ALLOW")
</syntaxhighlight>

== Related Pages ==
* (Empty)
