# utils/check_copies.py

## Understanding

### Purpose
Enforces code copy consistency

### Mechanism
The script validates that code marked with "# Copied from" comments matches its source. It parses Python code into blocks (functions, classes, methods), applies specified transformations (find-replace patterns), and compares the result against the actual code. It also synchronizes model lists across localized README files and supports automatic fixing of inconsistencies. The tool uses AST inspection and regex parsing to handle complex code structures including multi-line definitions, decorators, and nested blocks.

### Significance
Maintains code consistency across similar model implementations by enforcing that copied code stays synchronized with its source. This reduces maintenance burden when fixing bugs or making improvements, as the fixes can be automatically propagated to all copies, while still allowing necessary customizations through replacement patterns.
