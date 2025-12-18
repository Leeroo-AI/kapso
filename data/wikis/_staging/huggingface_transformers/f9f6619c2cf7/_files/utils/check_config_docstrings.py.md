# utils/check_config_docstrings.py

## Understanding

### Purpose
Validates configuration class checkpoint links

### Mechanism
The script extracts and validates checkpoint links from configuration class docstrings, ensuring each config class references at least one valid HuggingFace model checkpoint. It uses regex patterns to find markdown-formatted links in docstrings and verifies that the checkpoint name matches the URL. A whitelist handles special cases like composite models or those without associated papers.

### Significance
Ensures users can easily find example models for each architecture by requiring every config class to document at least one working checkpoint. This improves the discoverability of models and maintains high-quality documentation standards across the library.
