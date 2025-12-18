# custom_init_isort.py - Custom Import Sorter for Transformers

| Metadata | Value |
|----------|-------|
| **Type** | Implementation |
| **Source File** | `utils/custom_init_isort.py` |
| **Repository** | huggingface/transformers |
| **Domains** | Code Quality, Import Management, Style Enforcement |
| **Last Updated** | 2025-12-18 |
| **Lines of Code** | 331 |

---

## Overview

The `custom_init_isort.py` script is a specialized utility that sorts imports in Transformers' custom `__init__.py` files. These init files use a delayed import pattern via `_import_structure` dictionaries to avoid loading all models at import time, improving the `import transformers` performance. While `isort` and `ruff` handle standard imports in the `TYPE_CHECKING` block, this script sorts the custom `_import_structure` dictionaries.

---

## Description

### Purpose

This script addresses a unique challenge in the Transformers library: maintaining sorted imports in a custom import structure that standard tools like `isort` or `ruff` cannot handle. The library uses a two-part initialization pattern:

1. **Runtime Imports**: A `_import_structure` dictionary that maps module names to object lists, enabling lazy loading
2. **Type-Checking Imports**: Standard imports in a `TYPE_CHECKING` block for static type checkers

Standard import sorters only handle part 2, so this script handles part 1.

### Key Features

1. **Custom Import Structure Sorting**: Sorts keys and values in `_import_structure` dictionaries
2. **Indentation-Aware Parsing**: Respects Python's indentation-based block structure
3. **isort-Compatible Rules**: Follows the same sorting rules as `isort` (uppercase, CamelCase, lowercase)
4. **Multi-Format Support**: Handles single-line, multi-line, and bracket-wrapped import lists
5. **Check-Only Mode**: Can validate without modifying files (for CI/CD)
6. **Recursive Processing**: Walks entire source tree to find and sort all init files

### Sorting Rules

The script follows isort's conventions:
1. **Constants** (all uppercase) come first: `BERT_CONFIG`, `GPT2_MODEL`
2. **Classes** (CamelCase) come second: `BertModel`, `BertConfig`
3. **Functions** (lowercase) come last: `load_model`, `save_pretrained`

Within each category, items are sorted alphabetically, ignoring underscores and case.

### Architecture

The script operates in several phases:

1. **File Discovery**: Walk the source tree to find all `__init__.py` files
2. **Block Splitting**: Split each file into indented blocks at the `_import_structure` level
3. **Import Extraction**: Extract module keys and their associated import lists
4. **Sorting**: Sort both keys (modules) and values (imports) using isort rules
5. **Reconstruction**: Rebuild the file with sorted imports
6. **Validation/Writing**: Either check for changes or write back to file

---

## Code Reference

### Main Entry Point

```python
def sort_imports_in_all_inits(check_only=True):
    """
    Sort the imports defined in the `_import_structure` of all inits in the repo.

    Args:
        check_only: Whether or not to just check (and not auto-fix) the init.

    Raises:
        ValueError: If check_only=True and files need sorting
    """
    failures = []
    for root, _, files in os.walk(PATH_TO_TRANSFORMERS):
        if "__init__.py" in files:
            result = sort_imports(os.path.join(root, "__init__.py"), check_only=check_only)
            if result:
                failures = [os.path.join(root, "__init__.py")]
    if len(failures) > 0:
        raise ValueError(f"Would overwrite {len(failures)} files, run `make style`.")
```

### Core Functions

#### Block Splitting

```python
def split_code_in_indented_blocks(
    code: str,
    indent_level: str = "",
    start_prompt: str | None = None,
    end_prompt: str | None = None
) -> list[str]:
    """
    Split some code into its indented blocks, starting at a given level.

    Args:
        code: The code to split.
        indent_level: The indent level (as string) to use for identifying blocks.
        start_prompt: If provided, only starts splitting at this line.
        end_prompt: If provided, stops splitting at this line.

    Warning:
        The text before start_prompt or after end_prompt is not ignored, just not split.
        The input code can be retrieved by joining the result.

    Returns:
        The list of blocks.
    """
    index = 0
    lines = code.split("\n")

    # Move to start_prompt if provided
    if start_prompt is not None:
        while not lines[index].startswith(start_prompt):
            index += 1
        blocks = ["\n".join(lines[:index])]
    else:
        blocks = []

    current_block = [lines[index]]
    index += 1

    # Split into blocks until end_prompt or EOF
    while index < len(lines) and (end_prompt is None or not lines[index].startswith(end_prompt)):
        # New block starts when line has proper indent
        if len(lines[index]) > 0 and get_indent(lines[index]) == indent_level:
            # Check if line is part of current block or new block
            if len(current_block) > 0 and get_indent(current_block[-1]).startswith(indent_level + " "):
                # Part of current block
                current_block.append(lines[index])
                blocks.append("\n".join(current_block))
                if index < len(lines) - 1:
                    current_block = [lines[index + 1]]
                    index += 1
                else:
                    current_block = []
            else:
                # New block
                blocks.append("\n".join(current_block))
                current_block = [lines[index]]
        else:
            current_block.append(lines[index])
        index += 1

    # Add final blocks
    if len(current_block) > 0:
        blocks.append("\n".join(current_block))
    if end_prompt is not None and index < len(lines):
        blocks.append("\n".join(lines[index:]))

    return blocks
```

#### Utility Functions

```python
def get_indent(line: str) -> str:
    """Returns the indent in given line (as string)."""
    search = _re_indent.search(line)
    return "" if search is None else search.groups()[0]

def ignore_underscore_and_lowercase(key: Callable[[Any], str]) -> Callable[[Any], str]:
    """Wraps a key function to lowercase and ignore underscores."""
    def _inner(x):
        return key(x).lower().replace("_", "")
    return _inner
```

#### Sorting Logic

```python
def sort_objects(objects: list[Any], key: Callable[[Any], str] | None = None) -> list[Any]:
    """
    Sort a list of objects following the rules of isort (all uppercased first,
    camel-cased second and lower-cased last).

    Args:
        objects: The list of objects to sort.
        key: A function taking an object as input and returning a string.
             If not provided, defaults to noop.

    Returns:
        The sorted list with the same elements as in the inputs
    """
    def noop(x):
        return x

    if key is None:
        key = noop

    # Separate into categories
    constants = [obj for obj in objects if key(obj).isupper()]
    classes = [obj for obj in objects if key(obj)[0].isupper() and not key(obj).isupper()]
    functions = [obj for obj in objects if not key(obj)[0].isupper()]

    # Sort each group
    key1 = ignore_underscore_and_lowercase(key)
    return sorted(constants, key=key1) + sorted(classes, key=key1) + sorted(functions, key=key1)
```

#### Import Statement Sorting

```python
def sort_objects_in_import(import_statement: str) -> str:
    """
    Sorts the imports in a single import statement.

    Args:
        import_statement: The import statement in which to sort the imports.

    Returns:
        The same as the input, but with objects properly sorted.
    """
    def _replace(match):
        """Inner function to sort imports between [ ]."""
        imports = match.groups()[0]
        if "," not in imports:
            return f"[{imports}]"
        keys = [part.strip().replace('"', "") for part in imports.split(",")]
        if len(keys[-1]) == 0:
            keys = keys[:-1]
        return "[" + ", ".join([f'"{k}"' for k in sort_objects(keys)]) + "]"

    lines = import_statement.split("\n")

    if len(lines) > 3:
        # Multi-line format (one import per line)
        # key: [
        #     "object1",
        #     "object2",
        # ]
        idx = 2 if lines[1].strip() == "[" else 1
        keys_to_sort = [(i, _re_strip_line.search(line).groups()[0])
                        for i, line in enumerate(lines[idx:-idx])]
        sorted_indices = sort_objects(keys_to_sort, key=lambda x: x[1])
        sorted_lines = [lines[x[0] + idx] for x in sorted_indices]
        return "\n".join(lines[:idx] + sorted_lines + lines[-idx:])

    elif len(lines) == 3:
        # Single line with brackets on separate lines
        # key: [
        #     "object1", "object2"
        # ]
        if _re_bracket_content.search(lines[1]) is not None:
            lines[1] = _re_bracket_content.sub(_replace, lines[1])
        else:
            keys = [part.strip().replace('"', "") for part in lines[1].split(",")]
            if len(keys[-1]) == 0:
                keys = keys[:-1]
            lines[1] = get_indent(lines[1]) + ", ".join([f'"{k}"' for k in sort_objects(keys)])
        return "\n".join(lines)

    else:
        # One-line format: key: ["object1", "object2"]
        import_statement = _re_bracket_content.sub(_replace, import_statement)
        return import_statement
```

#### Main Sorting Function

```python
def sort_imports(file: str, check_only: bool = True):
    """
    Sort the imports defined in the `_import_structure` of a given init.

    Args:
        file: The path to the init to check/fix.
        check_only: Whether or not to just check (and not auto-fix) the init.
    """
    with open(file, encoding="utf-8") as f:
        code = f.read()

    # Skip if not a custom init
    if "_import_structure" not in code or "define_import_structure" in code:
        return

    # Split into blocks at indent level 0
    main_blocks = split_code_in_indented_blocks(
        code, start_prompt="_import_structure = {", end_prompt="if TYPE_CHECKING:"
    )

    # Process each block between start and end prompts
    for block_idx in range(1, len(main_blocks) - 1):
        block = main_blocks[block_idx]
        block_lines = block.split("\n")

        # Find start of imports
        line_idx = 0
        while line_idx < len(block_lines) and "_import_structure" not in block_lines[line_idx]:
            if "import dummy" in block_lines[line_idx]:
                line_idx = len(block_lines)
            else:
                line_idx += 1

        if line_idx >= len(block_lines):
            continue

        # Split internal block into import statements
        internal_block_code = "\n".join(block_lines[line_idx:-1])
        indent = get_indent(block_lines[1])
        internal_blocks = split_code_in_indented_blocks(internal_block_code, indent_level=indent)

        # Extract keys (module names)
        pattern = _re_direct_key if "_import_structure = {" in block_lines[0] else _re_indirect_key
        keys = [(pattern.search(b).groups()[0] if pattern.search(b) is not None else None)
                for b in internal_blocks]

        # Sort blocks by key
        keys_to_sort = [(i, key) for i, key in enumerate(keys) if key is not None]
        sorted_indices = [x[0] for x in sorted(keys_to_sort, key=lambda x: x[1])]

        # Reorder blocks
        count = 0
        reordered_blocks = []
        for i in range(len(internal_blocks)):
            if keys[i] is None:
                reordered_blocks.append(internal_blocks[i])
            else:
                block = sort_objects_in_import(internal_blocks[sorted_indices[count]])
                reordered_blocks.append(block)
                count += 1

        # Reconstruct block
        main_blocks[block_idx] = "\n".join(block_lines[:line_idx] + reordered_blocks + [block_lines[-1]])

    # Check if changes are needed
    if code != "\n".join(main_blocks):
        if check_only:
            return True
        else:
            print(f"Overwriting {file}.")
            with open(file, "w", encoding="utf-8") as f:
                f.write("\n".join(main_blocks))
```

### Regular Expression Patterns

```python
# Pattern that looks at the indentation in a line.
_re_indent = re.compile(r"^(\s*)\S")

# Pattern that matches `"key":` and puts `key` in group 0.
_re_direct_key = re.compile(r'^\s*"([^"]+)":')

# Pattern that matches `_import_structure["key"]` and puts `key` in group 0.
_re_indirect_key = re.compile(r'^\s*_import_structure\["([^"]+)"\]')

# Pattern that matches `"key",` and puts `key` in group 0.
_re_strip_line = re.compile(r'^\s*"([^"]+)",\s*$')

# Pattern that matches any `[stuff]` and puts `stuff` in group 0.
_re_bracket_content = re.compile(r"\[([^\]]+)\]")
```

### Constants

```python
# Path is defined with the intent you should run this script from the root of the repo.
PATH_TO_TRANSFORMERS = "src/transformers"
```

---

## I/O Contract

### Input Specifications

| Input Type | Description | Format | Required |
|------------|-------------|--------|----------|
| Source Files | `__init__.py` files in transformers | Python files | Yes |
| Check Mode | Whether to check only or fix | Boolean flag | No (default: check) |

### Output Specifications

| Output Type | Description | Format | Conditions |
|-------------|-------------|--------|------------|
| Modified Files | Sorted init files | Python files | If `--check_only` not set |
| Console Output | Which files were modified | Text | If files modified |
| Exit Code | Success or failure status | Integer | Always |
| Exception | List of files needing sorting | ValueError | If check fails |

### Exit Codes

- **0**: All imports are sorted (or successfully sorted)
- **1**: Files need sorting (check mode) or other error

---

## Usage Examples

### Check Import Sorting (CI/CD)

Run in check-only mode to verify imports are sorted:

```bash
python utils/custom_init_isort.py --check_only
```

**Output if sorted**:
```
(no output, exit code 0)
```

**Output if unsorted**:
```
ValueError: Would overwrite 3 files, run `make style`.
```

### Auto-Fix Import Sorting

Run without `--check_only` to automatically fix sorting:

```bash
python utils/custom_init_isort.py
```

**Output**:
```
Overwriting src/transformers/models/bert/__init__.py.
Overwriting src/transformers/models/gpt2/__init__.py.
```

### Integration with Makefile

The script is integrated into the repository's style checks:

```bash
# Check quality (includes import sorting check)
make quality

# Fix style (includes import sorting)
make style
```

### Example: Before and After

**Before (unsorted)**:

```python
_import_structure = {
    "configuration_bert": [
        "BertConfig",
        "BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BertOnnxConfig",
    ],
    "tokenization_bert": [
        "BasicTokenizer",
        "BertTokenizer",
        "WordpieceTokenizer",
    ],
    "modeling_bert": [
        "BertModel",
        "BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BertForMaskedLM",
        "BertForPreTraining",
        "load_tf_weights_in_bert",
        "BertForSequenceClassification",
    ],
}
```

**After (sorted)**:

```python
_import_structure = {
    "configuration_bert": [
        "BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # Constants first
        "BertConfig",                           # Classes second
        "BertOnnxConfig",                       # Sorted alphabetically
    ],
    "modeling_bert": [
        "BERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # Constants first
        "BertForMaskedLM",                     # Classes sorted
        "BertForPreTraining",
        "BertForSequenceClassification",
        "BertModel",
        "load_tf_weights_in_bert",             # Functions last
    ],
    "tokenization_bert": [
        "BasicTokenizer",                      # Classes first
        "BertTokenizer",
        "WordpieceTokenizer",
    ],
}
```

### Example: Multi-Line Format

**Before**:

```python
_import_structure = {
    "modeling_bert": [
        "BertModel",
        "BertForTokenClassification",
        "BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BertForMaskedLM",
    ]
}
```

**After**:

```python
_import_structure = {
    "modeling_bert": [
        "BERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # Constant moved to top
        "BertForMaskedLM",                     # Classes sorted
        "BertForTokenClassification",
        "BertModel",
    ]
}
```

### Programmatic Usage

```python
from utils.custom_init_isort import sort_imports

# Check if file needs sorting
needs_sorting = sort_imports("src/transformers/__init__.py", check_only=True)

if needs_sorting:
    print("File needs sorting!")
    # Auto-fix
    sort_imports("src/transformers/__init__.py", check_only=False)
else:
    print("File is already sorted!")
```

### Testing the Sorting Logic

```python
from utils.custom_init_isort import sort_objects

# Test constant/class/function sorting
objects = [
    "load_model",              # function (lowercase)
    "BertModel",              # class (CamelCase)
    "BERT_CONFIG",            # constant (UPPERCASE)
    "save_checkpoint",        # function
    "BertTokenizer",          # class
    "GPT2_PRETRAINED_LIST",   # constant
]

sorted_objects = sort_objects(objects)
print(sorted_objects)
# Output:
# ['BERT_CONFIG', 'GPT2_PRETRAINED_LIST',  # Constants
#  'BertModel', 'BertTokenizer',           # Classes
#  'load_model', 'save_checkpoint']        # Functions
```

---

## Related Pages

<!-- Links to related documentation pages -->

---

## Implementation Notes

### Why Custom Sorting is Needed

The Transformers library uses a delayed import pattern to improve `import transformers` speed:

```python
# Standard imports (TYPE_CHECKING block) - handled by isort/ruff
if TYPE_CHECKING:
    from .configuration_bert import BertConfig
    from .modeling_bert import BertModel

# Runtime imports (lazy loading) - requires custom sorting
_import_structure = {
    "configuration_bert": ["BertConfig"],
    "modeling_bert": ["BertModel"],
}
```

At runtime, objects are imported on-demand via `__getattr__`:

```python
def __getattr__(name):
    if name in _import_structure:
        # Lazy import
        return importlib.import_module(f".{name}", __name__)
```

This dramatically reduces import time (from ~10s to <1s) when users have all optional dependencies installed.

### Indent-Based Parsing

The script respects Python's indentation structure without using `ast.parse()`:

```python
def get_indent(line: str) -> str:
    """Extract indent as string (spaces/tabs)."""
    search = _re_indent.search(line)
    return "" if search is None else search.groups()[0]
```

This allows parsing incomplete code blocks and preserving exact formatting.

### Sorting Algorithm

The three-tier sorting (constants, classes, functions) matches `isort`'s behavior:

```python
# 1. Separate by category
constants = [obj for obj in objects if key(obj).isupper()]
classes = [obj for obj in objects if key(obj)[0].isupper() and not key(obj).isupper()]
functions = [obj for obj in objects if not key(obj)[0].isupper()]

# 2. Sort each category (ignoring underscores and case)
key_fn = lambda x: x.lower().replace("_", "")
return sorted(constants, key=key_fn) + sorted(classes, key=key_fn) + sorted(functions, key=key_fn)
```

This ensures:
- `BERT_CONFIG` comes before `BertModel`
- `BertModel` comes before `load_bert`
- `_private_helper` is sorted as `privatehelper`

### Block Reconstruction

The script carefully preserves:
- Comments between import blocks
- Empty lines
- Exact indentation
- Everything before `_import_structure = {` and after `if TYPE_CHECKING:`

```python
# Preserve structure
blocks = split_code_in_indented_blocks(
    code,
    start_prompt="_import_structure = {",  # Start here
    end_prompt="if TYPE_CHECKING:"          # Stop here
)
# blocks[0] = everything before _import_structure
# blocks[1:-1] = import blocks to sort
# blocks[-1] = everything after TYPE_CHECKING
```

### Multiple Format Support

The script handles three import list formats:

**Format 1: Single line**
```python
"module": ["Class1", "Class2", "func1"]
```

**Format 2: Bracket on separate lines, one-line list**
```python
"module": [
    "Class1", "Class2", "func1"
]
```

**Format 3: Multi-line (one import per line)**
```python
"module": [
    "Class1",
    "Class2",
    "func1",
]
```

All three are detected and sorted appropriately.

### Edge Cases Handled

1. **Empty or comment-only lines**: Preserved in original position
2. **Dummy imports**: Skipped entirely (e.g., `import dummy_pt_objects`)
3. **Dynamic import structures**: Files with `define_import_structure` are skipped
4. **Trailing commas**: Preserved
5. **Mixed quote styles**: Normalized to double quotes
