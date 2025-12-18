# utils/add_pipeline_model_mapping_to_test.py

## Understanding

### Purpose
Adds pipeline mappings to tests

### Mechanism
The script automatically generates and adds the `pipeline_model_mapping` attribute to model test files. It introspects test classes to find their model configurations, matches them against the pipeline test mappings defined in the library, and generates the appropriate mapping dictionary showing which model classes support which pipeline tasks. The script can process individual test files or all model test files at once.

### Significance
Ensures that pipeline tests are correctly configured for each model, enabling automated testing of models with various pipeline tasks. This reduces manual work when adding new models and maintains consistency in how models are tested across different pipeline types.
