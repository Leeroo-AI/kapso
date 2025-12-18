# utils/check_docstrings.py

## Understanding

### Purpose
Validates and fixes docstring consistency

### Mechanism
The script performs two types of docstring validation: (1) for standard callable objects, it matches function/class signatures with their documented parameters, ensuring defaults are accurate and all parameters are documented; (2) for `@auto_docstring` decorated items, it parses AST to find decorated functions/classes, extracts or generates docstrings from signature and custom args, removes redundant documentation already in shared templates, and generates properly formatted docstrings. It handles complex cases like inherited arguments, dataclasses, ModelOutput classes, and multi-line signatures.

### Significance
Ensures API documentation stays synchronized with code implementations, preventing misleading documentation about parameter defaults or missing parameter descriptions. The auto-docstring feature reduces boilerplate by generating documentation from signatures while allowing custom descriptions where needed, maintaining documentation quality at scale.
