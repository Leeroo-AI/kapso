# Implementation: huggingface_transformers_check_config_attributes

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Code_Quality]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Configuration validation tool ensuring model config attributes are properly used and documented across modeling implementations.

=== Description ===

The `utils/check_config_attributes.py` module (548 lines) validates consistency between config classes and their usage. It:
- Parses config class attributes
- Scans modeling files for config attribute access
- Reports unused config attributes
- Identifies undocumented config usage

=== Usage ===

Run as part of CI quality checks to catch config/model mismatches early.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/utils/check_config_attributes.py utils/check_config_attributes.py]
* '''Lines:''' 1-548

=== Signature ===
<syntaxhighlight lang="python">
def get_config_attributes(config_class: type) -> set[str]:
    """Extract attributes from config class."""

def get_used_attributes(model_file: str) -> set[str]:
    """Find config attributes used in model file."""

def check_config_model_consistency(model_type: str) -> list[str]:
    """Report mismatches between config and model."""

def main():
    """Validate all model configs."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="bash">
python utils/check_config_attributes.py
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| Config files || Files || Yes || configuration_*.py files
|-
| Model files || Files || Yes || modeling_*.py files
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Validation report || stdout || Unused/undocumented attributes
|}

== Usage Examples ==

=== Check Config Consistency ===
<syntaxhighlight lang="bash">
python utils/check_config_attributes.py

# Output:
# bert: OK
# llama: WARN - unused config: rope_scaling
# mistral: ERROR - undocumented: sliding_window
</syntaxhighlight>

== Related Pages ==
