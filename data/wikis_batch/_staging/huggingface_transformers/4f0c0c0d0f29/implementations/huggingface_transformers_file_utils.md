# Implementation: huggingface_transformers_file_utils

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Backward_Compatibility]], [[domain::Utilities]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Backward compatibility import shim re-exporting utilities that were moved to `transformers.utils`, allowing legacy code to continue working.

=== Description ===

The `file_utils` module exists solely for backward compatibility. Originally containing file utilities for downloading and caching models, all functionality has been moved to `transformers.utils`. This module re-exports ~80 objects including constants (CONFIG_NAME, WEIGHTS_NAME), type checking utilities (is_torch_available, is_vision_available), and helper classes (ModelOutput, TensorType).

=== Usage ===

For new code, import directly from `transformers.utils`. Use `file_utils` only when maintaining legacy code that depends on the old import paths.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' [https://github.com/huggingface/transformers/blob/main/src/transformers/file_utils.py src/transformers/file_utils.py]
* '''Lines:''' 1-107

=== Signature ===
<syntaxhighlight lang="python">
# Re-exported constants
CONFIG_NAME: str
WEIGHTS_NAME: str
WEIGHTS_INDEX_NAME: str
FEATURE_EXTRACTOR_NAME: str
MODEL_CARD_NAME: str

# Re-exported backend checks
def is_torch_available() -> bool: ...
def is_tokenizers_available() -> bool: ...
def is_vision_available() -> bool: ...
# ... 50+ availability checks

# Re-exported classes
class ModelOutput: ...
class TensorType: ...
class PaddingStrategy: ...
class PushToHubMixin: ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Legacy import (still works)
from transformers.file_utils import is_torch_available, CONFIG_NAME

# Preferred modern import
from transformers.utils import is_torch_available, CONFIG_NAME
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| N/A || N/A || N/A || Module is a pure re-export shim
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| Various || Various || Re-exports ~80 objects from transformers.utils
|}

== Usage Examples ==

=== Legacy Code Compatibility ===
<syntaxhighlight lang="python">
# Old code that still works:
from transformers.file_utils import (
    is_torch_available,
    is_tokenizers_available,
    CONFIG_NAME,
    WEIGHTS_NAME,
)

if is_torch_available():
    import torch
    print(f"Config file: {CONFIG_NAME}")  # "config.json"
</syntaxhighlight>

=== Modern Equivalent ===
<syntaxhighlight lang="python">
# Recommended approach for new code:
from transformers.utils import (
    is_torch_available,
    is_tokenizers_available,
    CONFIG_NAME,
    WEIGHTS_NAME,
)

# Same functionality, proper import path
if is_torch_available():
    import torch
</syntaxhighlight>

== Related Pages ==
