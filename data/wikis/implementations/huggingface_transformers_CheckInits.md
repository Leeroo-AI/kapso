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
A validation utility that ensures __init__.py files in the transformers library maintain consistency between _import_structure definitions and TYPE_CHECKING blocks, and that all submodules are properly registered.

=== Description ===
The check_inits.py script validates the custom initialization pattern used throughout the transformers library to implement lazy imports. The library uses a two-part initialization system:

'''Part 1: _import_structure dictionary''' - Maps module names to lists of objects that should be imported from them. This is used at runtime for lazy importing.

'''Part 2: TYPE_CHECKING block''' - Contains traditional imports that are only processed by type checkers, not at runtime. This enables proper IDE support and type checking.

Key functions:

* '''parse_init()''': Parses an __init__.py file to extract both _import_structure objects and TYPE_CHECKING objects, organizing them by backend (e.g., "torch", "tf", "flax") or "none" for backend-independent imports
* '''analyze_results()''': Compares the two dictionaries to find mismatches, including duplicate imports, missing imports in either section, and backend inconsistencies
* '''check_submodules()''': Validates that all submodules in the repository are registered in the main __init__.py's _import_structure
* '''find_backend()''': Identifies conditional imports based on backend availability checks (e.g., is_torch_available())
* '''get_transformers_submodules()''': Scans the repository to discover all Python submodules

The script uses regular expressions to parse the special syntax used in these init files, handling backend-specific conditional blocks and various import patterns.

=== Usage ===
Use check_inits.py when:
* Running CI/CD checks to ensure import consistency (make repo-consistency)
* Adding new models or modules to ensure they're properly registered
* Modifying __init__.py files to maintain consistency
* Validating that lazy imports are correctly defined
* Ensuring all submodules are discoverable through the main transformers module
* Debugging import issues or missing module errors

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers huggingface_transformers]
* '''File:''' utils/check_inits.py

=== Signature ===
<syntaxhighlight lang="python">
def parse_init(init_file) -> tuple[dict[str, list[str]], dict[str, list[str]]] | None:
    """
    Read an init_file and parse (per backend) the `_import_structure` objects
    defined and the `TYPE_CHECKING` objects defined.

    Returns:
        Tuple of two dictionaries mapping backends to list of imported objects,
        or None if the init is not a custom init.
    """

def analyze_results(
    import_dict_objects: dict[str, list[str]],
    type_hint_objects: dict[str, list[str]]
) -> list[str]:
    """
    Analyze the differences between _import_structure objects and TYPE_CHECKING objects.

    Returns:
        List of error messages describing mismatches.
    """

def find_backend(line: str) -> str | None:
    """
    Find one (or multiple) backend in a code line of the init.

    Returns:
        Backend name or None if no backend check found.
    """

def check_submodules():
    """
    Check all submodules of Transformers are properly registered in the main init.
    """

def get_transformers_submodules() -> list[str]:
    """
    Returns the list of Transformers submodules.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Run from repository root
python utils/check_inits.py

# Or import and use functions
from utils.check_inits import parse_init, analyze_results

# Parse an init file
init_file = "src/transformers/models/bert/__init__.py"
result = parse_init(init_file)
if result:
    import_structure, type_checking = result
    errors = analyze_results(import_structure, type_checking)
    if errors:
        print("\n".join(errors))
</syntaxhighlight>

== I/O Contract ==

{| class="wikitable"
! Field !! Type !! Description
|-
| colspan="3" style="text-align:center;" | '''Input: parse_init()'''
|-
| init_file || str || Path to __init__.py file to parse
|-
| colspan="3" style="text-align:center;" | '''Output: parse_init()'''
|-
| import_dict_objects || dict[str, list[str]] || Maps backend names to _import_structure objects
|-
| type_hint_objects || dict[str, list[str]] || Maps backend names to TYPE_CHECKING objects
|-
| (or None) || None || If file uses traditional imports (not lazy loading)
|-
| colspan="3" style="text-align:center;" | '''Input: analyze_results()'''
|-
| import_dict_objects || dict[str, list[str]] || _import_structure objects by backend
|-
| type_hint_objects || dict[str, list[str]] || TYPE_CHECKING objects by backend
|-
| colspan="3" style="text-align:center;" | '''Output: analyze_results()'''
|-
| errors || list[str] || List of error messages describing inconsistencies
|-
| colspan="3" style="text-align:center;" | '''Input: find_backend()'''
|-
| line || str || A line of code from an __init__.py file
|-
| colspan="3" style="text-align:center;" | '''Output: find_backend()'''
|-
| backend || str or None || Backend name (e.g., "torch", "torch_and_vision") or None
|-
| colspan="3" style="text-align:center;" | '''Output: check_submodules()'''
|-
| (raises ValueError) || ValueError || If unregistered submodules found
|}

== Usage Examples ==
<syntaxhighlight lang="python">
# Example 1: Validate a specific __init__.py file
from utils.check_inits import parse_init, analyze_results

init_file = "src/transformers/models/bert/__init__.py"
result = parse_init(init_file)

if result is None:
    print(f"{init_file} uses traditional imports")
else:
    import_structure, type_checking = result
    errors = analyze_results(import_structure, type_checking)

    if errors:
        print(f"Errors found in {init_file}:")
        for error in errors:
            print(f"  - {error}")
    else:
        print(f"{init_file} is consistent!")

# Example 2: Check all submodules are registered
from utils.check_inits import check_submodules

try:
    check_submodules()
    print("All submodules properly registered!")
except ValueError as e:
    print(f"Unregistered submodules found:\n{e}")

# Example 3: Proper __init__.py structure for lazy loading
# In your __init__.py file:

_import_structure = {
    "configuration_bert": ["BertConfig"],
    "tokenization_bert": ["BertTokenizer"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_bert"] = [
        "BertModel",
        "BertForSequenceClassification",
    ]

if TYPE_CHECKING:
    from .configuration_bert import BertConfig
    from .tokenization_bert import BertTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_bert import (
            BertModel,
            BertForSequenceClassification,
        )

# Example 4: Find backend from conditional line
from utils.check_inits import find_backend

line1 = "    if not is_torch_available():"
backend1 = find_backend(line1)  # Returns "torch"

line2 = "    if not is_torch_available() and is_vision_available():"
backend2 = find_backend(line2)  # Returns "torch_and_vision"

line3 = "    regular_import()"
backend3 = find_backend(line3)  # Returns None

# Example 5: Get all submodules
from utils.check_inits import get_transformers_submodules

submodules = get_transformers_submodules()
print(f"Found {len(submodules)} submodules")
for submodule in sorted(submodules)[:5]:
    print(f"  - {submodule}")
# Output:
#   - models.bert
#   - models.gpt2
#   - tokenization_utils
#   - trainer
#   - ...
</syntaxhighlight>

== Related Pages ==
* [[huggingface_transformers_CheckCopies]]
* [[huggingface_transformers_CheckDocstrings]]
* [[huggingface_transformers_LazyImportSystem]]
