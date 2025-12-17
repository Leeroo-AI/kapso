# Implementation: huggingface_transformers_modular_model_converter

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Blog|Modular Transformers|https://huggingface.co/blog/modular-transformers]]
|-
! Domains
| [[domain::Code_Generation]], [[domain::Model_Architecture]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Code generation system that converts modular model definitions into traditional single-file model implementations, enabling code reuse while maintaining backward compatibility.

=== Description ===

The `modular_model_converter` module (1920 lines) transforms modular model files (`modular_*.py`) into traditional modeling files. Modular files define only the differences from a base model using Python inheritance. The converter:
- Parses modular files using libcst AST
- Resolves inheritance chains and import dependencies
- Generates complete, standalone modeling files with auto-generated headers
- Handles model name casing transformations (camelCase, snake_case)

This enables ~70% code reduction for new models while maintaining the traditional file structure users expect.

=== Usage ===

Run after creating/modifying `modular_*.py` files to regenerate traditional model files. CI enforces that generated files match modular definitions.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/utils/modular_model_converter.py utils/modular_model_converter.py]
* '''Lines:''' 1-1920

=== Signature ===
<syntaxhighlight lang="python">
def get_module_source_from_name(module_name: str) -> str:
    """Load source code from module name."""

def preserve_case_replace(text: str, patterns: dict, default_name: str) -> str:
    """Replace text preserving case patterns (MyModel, MY_MODEL, my_model)."""

def get_cased_name(lowercase_name: str) -> str:
    """Convert my_model to MyModel using CONFIG_MAPPING_NAMES."""

class ModularFileConverter:
    """Main converter class."""

    def convert(self, modular_path: str) -> str:
        """Convert modular file to traditional modeling file."""

def main():
    """
    CLI entry point.

    Args:
        --model: Specific model to convert
        --check: Verify generated files match
        --all: Convert all modular files
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
# Convert specific model
python utils/modular_model_converter.py --model llama

# Convert all modular files
python utils/modular_model_converter.py --all

# Check without modifying (CI mode)
python utils/modular_model_converter.py --check
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| modular_*.py || File || Yes || Modular model definition file
|-
| --model || str || No || Model name to convert
|-
| --check || Flag || No || Verify mode (no writes)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| modeling_*.py || File || Generated traditional model file
|-
| configuration_*.py || File || Generated config file (if modular)
|-
| tokenization_*.py || File || Generated tokenizer file (if modular)
|}

== Usage Examples ==

=== Creating a New Model ===
<syntaxhighlight lang="python">
# src/transformers/models/my_model/modular_my_model.py
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM

class MyModelModel(LlamaModel):
    """Only define what's different from Llama."""
    def __init__(self, config):
        super().__init__(config)
        # Custom modifications only

class MyModelForCausalLM(LlamaForCausalLM):
    pass  # Inherits everything from Llama
</syntaxhighlight>

<syntaxhighlight lang="bash">
# Generate traditional files
python utils/modular_model_converter.py --model my_model

# Creates:
# - modeling_my_model.py (complete standalone file)
# - With auto-generated header warning not to edit
</syntaxhighlight>

== Related Pages ==
