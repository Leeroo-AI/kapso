# utils/check_config_attributes.py

## Understanding

### Purpose
Validates configuration attribute usage

### Mechanism
The script inspects all configuration classes and their `__init__` parameters, then searches through the corresponding modeling files to verify each attribute is actually used in the code. It handles attribute name mappings, checks for common patterns like `config.attribute` and `getattr(config, "attribute")`, and maintains a list of special cases for attributes that are valid but don't appear directly in modeling code (e.g., used internally, for generation, or framework compatibility).

### Significance
Prevents configuration bloat by ensuring all declared config attributes serve a purpose in the actual model implementation. This catches unused parameters that might have been left behind during refactoring, improving code maintainability and reducing confusion for users and contributors.
