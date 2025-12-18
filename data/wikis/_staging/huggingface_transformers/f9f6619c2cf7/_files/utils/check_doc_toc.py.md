# utils/check_doc_toc.py

## Understanding

### Purpose
Maintains documentation table of contents

### Mechanism
The script reads the YAML table of contents file (`_toctree.yml`), navigates to the model documentation section, and ensures all model entries within each modality are deduplicated and sorted alphabetically by title. It validates that entries with the same path have consistent titles and can automatically fix ordering issues.

### Significance
Keeps the documentation navigation organized and predictable, making it easier for users to find specific models. The alphabetical ordering and duplicate removal ensure a clean, professional documentation structure that's maintained automatically through CI checks.
