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
A utility script that validates code synchronization by checking that copied code blocks marked with "Copied from" comments match their original source, ensuring consistency across the transformers repository.

=== Description ===
The check_copies.py script provides a comprehensive system for maintaining code copy consistency across the HuggingFace Transformers repository. The main functionality is implemented through several key functions:

* '''check_copies()''': Main entry point that validates all Python files in the repository for copy consistency
* '''is_copy_consistent()''': Validates a single file by comparing code blocks marked with "# Copied from" comments against their source
* '''find_code_in_transformers()''': Locates and retrieves the source code of an object by name
* '''split_code_into_blocks()''': Splits class/function code into manageable blocks for comparison
* '''replace_code()''': Applies transformation patterns (e.g., with X1->X2, Y1->Y2) to copied code
* '''check_codes_match()''': Compares two code versions to detect differences

The script also validates that model lists in localized README files match the main README, using functions like get_model_list() and convert_to_localized_md().

The tool uses regular expressions to parse "Copied from" comments, handles code transformations specified in the comments, and can automatically fix inconsistencies when run with the --fix_and_overwrite flag.

=== Usage ===
Use check_copies.py when:
* Running CI/CD checks to ensure code consistency (make repo-consistency)
* Automatically fixing copy inconsistencies (make fix-copies)
* Validating that copied code blocks remain synchronized with their source
* Ensuring README model lists are consistent across localizations
* Before committing changes that might affect copied code sections

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers huggingface_transformers]
* '''File:''' utils/check_copies.py

=== Signature ===
<syntaxhighlight lang="python">
def check_copies(overwrite: bool = False, file: str | None = None):
    """Check every file is copy-consistent with the original."""

def is_copy_consistent(
    filename: str, overwrite: bool = False, buffer: dict | None = None
) -> list[tuple[str, int]] | None:
    """Check if the code commented as a copy in a file matches the original."""

def find_code_in_transformers(
    object_name: str, base_path: str | None = None, return_indices: bool = False
) -> str | tuple[list[str], int, int]:
    """Find and return the source code of an object."""

def split_code_into_blocks(
    lines: list[str], start_index: int, end_index: int, indent: int, backtrace: bool = False
) -> list[tuple[str, int, int]]:
    """Split the class/func block into inner blocks."""

def replace_code(code: str, replace_pattern: str) -> str:
    """Replace code by a pattern of the form `with X1->X2,Y1->Y2,Z1->Z2`."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Run from repository root
# Check only (errors on inconsistencies)
python utils/check_copies.py

# Auto-fix inconsistencies
python utils/check_copies.py --fix_and_overwrite

# Check specific file
python utils/check_copies.py --file path/to/file.py
</syntaxhighlight>

== I/O Contract ==

{| class="wikitable"
! Field !! Type !! Description
|-
| colspan="3" style="text-align:center;" | '''Input: check_copies()'''
|-
| overwrite || bool || Whether to automatically fix inconsistencies (default: False)
|-
| file || str, optional || Path to specific file to check (default: checks all files)
|-
| colspan="3" style="text-align:center;" | '''Output: check_copies()'''
|-
| (raises Exception) || Exception || Raised if inconsistencies found and overwrite=False
|-
| (prints) || str || Messages indicating files being rewritten (if overwrite=True)
|-
| colspan="3" style="text-align:center;" | '''Input: is_copy_consistent()'''
|-
| filename || str || Path to file to check
|-
| overwrite || bool || Whether to fix inconsistencies
|-
| buffer || dict, optional || Cache for previously retrieved code
|-
| colspan="3" style="text-align:center;" | '''Output: is_copy_consistent()'''
|-
| diffs || list[tuple[str, int]] || List of (object_name, line_number) tuples with differences
|}

== Usage Examples ==
<syntaxhighlight lang="python">
# Example 1: Check all files for copy consistency
from utils.check_copies import check_copies

try:
    check_copies(overwrite=False)
    print("All copies are consistent!")
except Exception as e:
    print(f"Copy inconsistencies found: {e}")

# Example 2: Auto-fix copies in all files
check_copies(overwrite=True)

# Example 3: Check specific file
check_copies(overwrite=False, file="src/transformers/models/bert/modeling_bert.py")

# Example 4: Find source code for an object
from utils.check_copies import find_code_in_transformers

code = find_code_in_transformers("transformers.models.bert.modeling_bert.BertAttention")
print(code)

# Example 5: Use in code with "Copied from" comment
# In your Python file:
# class MyNewAttention(nn.Module):
#     # Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->MyNew
#     def forward(self, hidden_states):
#         # This code will be validated against BertAttention
#         pass
</syntaxhighlight>

== Related Pages ==
* [[huggingface_transformers_CheckDocstrings]]
* [[huggingface_transformers_CheckInits]]
