{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Code_Quality]], [[domain::CI_CD]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A comprehensive utility script that validates and automatically fixes docstrings to ensure they match function/class signatures, with specialized support for the @auto_docstring decorator system used throughout the transformers library.

=== Description ===
The check_docstrings.py script provides two main validation systems:

'''Traditional Docstring Checking:'''
* '''match_docstring_with_signature()''': Compares a callable's docstring arguments section with its actual signature
* '''fix_docstring()''': Automatically updates docstrings to match signatures
* '''replace_default_in_arg_description()''': Updates default value descriptions in docstrings
* '''get_default_description()''': Generates default descriptions for undocumented parameters

'''@auto_docstring Decorator System:'''
* '''check_auto_docstrings()''': Validates all functions/classes decorated with @auto_docstring
* '''_build_ast_indexes()''': Parses source code using AST to find all @auto_docstring decorators
* '''generate_new_docstring_for_function()''': Generates docstrings for decorated functions
* '''generate_new_docstring_for_class()''': Generates docstrings for decorated classes (including __init__ and ModelOutput classes)
* '''update_file_with_new_docstrings()''': Writes updated docstrings back to source files

The @auto_docstring system allows automatic docstring generation and validation for model classes, pulling common argument descriptions from centralized sources (ModelArgs, ImageProcessorArgs, ModelOutputArgs) while allowing custom arguments to be specified.

The script supports mathematical expressions in default values, handles optional parameters, and can parse complex type annotations. It integrates with the repository's CI/CD pipeline through make commands.

=== Usage ===
Use check_docstrings.py when:
* Running CI/CD checks to ensure documentation quality (make repo-consistency)
* Automatically fixing docstring inconsistencies (make fix-copies)
* Adding new functions/classes that need documentation
* Using @auto_docstring decorators in model implementations
* Validating that parameter documentation matches actual signatures
* Before committing changes to ensure documentation is up-to-date

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers huggingface_transformers]
* '''File:''' utils/check_docstrings.py

=== Signature ===
<syntaxhighlight lang="python">
def check_docstrings(overwrite: bool = False, check_all: bool = False):
    """Check docstrings of all public objects that are callables and are documented."""

def check_auto_docstrings(overwrite: bool = False, check_all: bool = False):
    """Check docstrings of all objects decorated with @auto_docstring."""

def match_docstring_with_signature(obj: Any) -> tuple[str, str] | None:
    """Matches the docstring of an object with its signature."""

def fix_docstring(obj: Any, old_doc_args: str, new_doc_args: str):
    """Fixes the docstring of an object by replacing its arguments documentation."""

def replace_default_in_arg_description(description: str, default: Any) -> str:
    """Catches the default value in the description and replaces it."""

@dataclass
class DecoratedItem:
    """Information about a single @auto_docstring decorated function or class."""
    decorator_line: int
    def_line: int
    kind: str  # 'function' or 'class'
    body_start_line: int
    args: list[str]
    custom_args_text: str | None
    has_init: bool
    init_def_line: int | None
    is_model_output: bool
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Run from repository root
# Check only (errors on inconsistencies)
python utils/check_docstrings.py

# Auto-fix inconsistencies and generate docstring templates
python utils/check_docstrings.py --fix_and_overwrite

# Check all files (not just diff)
python utils/check_docstrings.py --check_all

# Both fix and check all
python utils/check_docstrings.py --fix_and_overwrite --check_all
</syntaxhighlight>

== I/O Contract ==

{| class="wikitable"
! Field !! Type !! Description
|-
| colspan="3" style="text-align:center;" | '''Input: check_docstrings()'''
|-
| overwrite || bool || Whether to automatically fix docstrings (default: False)
|-
| check_all || bool || Whether to check all files or just diff (default: False)
|-
| colspan="3" style="text-align:center;" | '''Output: check_docstrings()'''
|-
| (raises ValueError) || ValueError || Raised if inconsistencies found and overwrite=False
|-
| (prints) || str || Messages indicating files being fixed
|-
| colspan="3" style="text-align:center;" | '''Input: match_docstring_with_signature()'''
|-
| obj || Any || The callable object to validate
|-
| colspan="3" style="text-align:center;" | '''Output: match_docstring_with_signature()'''
|-
| old_doc_arg || str || Current parameter documentation from docstring
|-
| new_doc_arg || str || Corrected parameter documentation matching signature
|-
| (or None) || None || If no docstring or no Args section found
|-
| colspan="3" style="text-align:center;" | '''Input: check_auto_docstrings()'''
|-
| overwrite || bool || Whether to automatically fix docstrings (default: False)
|-
| check_all || bool || Whether to check all files or just diff (default: False)
|-
| colspan="3" style="text-align:center;" | '''Output: check_auto_docstrings()'''
|-
| (raises ValueError) || ValueError || Raised if missing/incomplete docstrings found
|-
| (prints) || str || Error messages with line numbers for issues found
|}

== Usage Examples ==
<syntaxhighlight lang="python">
# Example 1: Check all modified files for docstring consistency
from utils.check_docstrings import check_docstrings

try:
    check_docstrings(overwrite=False, check_all=False)
    print("All docstrings are consistent!")
except ValueError as e:
    print(f"Docstring issues found: {e}")

# Example 2: Auto-fix all docstrings
check_docstrings(overwrite=True, check_all=True)

# Example 3: Use @auto_docstring decorator in your code
from transformers.utils.auto_docstring import auto_docstring

@auto_docstring
class MyModel(nn.Module):
    """
    My custom model implementation.
    """
    def __init__(self, hidden_size: int, num_layers: int = 2):
        # Docstring will be auto-generated for __init__ parameters
        pass

# Example 4: Use @auto_docstring with custom args
CUSTOM_ARGS = r"""
    special_param (`int`):
        A special parameter unique to this model.
"""

@auto_docstring(custom_args=CUSTOM_ARGS)
class MySpecialModel(nn.Module):
    def __init__(self, hidden_size: int, special_param: int):
        # Common args from ModelArgs + custom_args will be merged
        pass

# Example 5: Match docstring manually
from utils.check_docstrings import match_docstring_with_signature

def my_function(arg1: int, arg2: str = "default"):
    """
    Args:
        arg1 (`int`): First argument
        arg2 (`str`): Second argument
    """
    pass

result = match_docstring_with_signature(my_function)
if result:
    old_doc, new_doc = result
    print(f"Current: {old_doc}")
    print(f"Should be: {new_doc}")

# Example 6: Check for specific patterns
from utils.check_docstrings import replace_default_in_arg_description

desc = "`int`, *optional*"
new_desc = replace_default_in_arg_description(desc, 42)
# Returns: "`int`, *optional*, defaults to 42"
</syntaxhighlight>

== Related Pages ==
* [[huggingface_transformers_CheckCopies]]
* [[huggingface_transformers_CheckInits]]
