# Repository Map: huggingface_transformers

> **Compact index** of repository files.
> Each file has a detail page in `_files/` with Understanding to fill.
> Mark files as ✅ explored in the table below as you complete them.

| Property | Value |
|----------|-------|
| Repository | https://github.com/huggingface/transformers |
| Branch | main |
| Generated | 2025-12-18 12:30 |
| Python Files | 200 |
| Total Lines | 114,291 |
| Explored | 200/200 |

## Structure

📦 **Packages:** benchmark, benchmark_v2, utils
📝 **Examples:** examples, notebooks, scripts
🧪 **Tests:** tests

📖 README: `README.md`
⚙️ Setup: `pyproject.toml`

---

## 📦 Package Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `benchmark/__init__.py` | 0 | Package initializer for benchmark module | — | [→](./_files/benchmark___init___py.md) |
| ✅ | `benchmark/benchmark.py` | 324 | Multi-commit benchmark orchestrator for performance regression testing | Impl:huggingface_transformers_Benchmark | [→](./_files/benchmark_benchmark_py.md) |
| ✅ | `benchmark/benchmarks_entrypoint.py` | 502 | Main benchmarking entrypoint for CI/CD with metrics recording | Impl:huggingface_transformers_MetricsRecorder | [→](./_files/benchmark_benchmarks_entrypoint_py.md) |
| ✅ | `benchmark/optimum_benchmark_wrapper.py` | 20 | Minimal wrapper invoking optimum-benchmark CLI | — | [→](./_files/benchmark_optimum_benchmark_wrapper_py.md) |
| ✅ | `benchmark_v2/run_benchmarks.py` | 128 | Next-gen benchmark orchestration with Hub integration | — | [→](./_files/benchmark_v2_run_benchmarks_py.md) |
| ✅ | `utils/add_dates.py` | 427 | Adds dates to model deprecation notices | — | [→](./_files/utils_add_dates_py.md) |
| ✅ | `utils/add_pipeline_model_mapping_to_test.py` | 308 | Adds pipeline model mappings to test files | — | [→](./_files/utils_add_pipeline_model_mapping_to_test_py.md) |
| ✅ | `utils/check_bad_commit.py` | 280 | Identifies commits that introduced test failures | — | [→](./_files/utils_check_bad_commit_py.md) |
| ✅ | `utils/check_config_attributes.py` | 548 | Validates model config attributes match documentation | — | [→](./_files/utils_check_config_attributes_py.md) |
| ✅ | `utils/check_config_docstrings.py` | 102 | Validates config class docstrings | — | [→](./_files/utils_check_config_docstrings_py.md) |
| ✅ | `utils/check_copies.py` | 1044 | Ensures copied code blocks stay synchronized | — | [→](./_files/utils_check_copies_py.md) |
| ✅ | `utils/check_doc_toc.py` | 133 | Validates documentation table of contents | — | [→](./_files/utils_check_doc_toc_py.md) |
| ✅ | `utils/check_docstrings.py` | 1559 | Validates docstring formatting and completeness | — | [→](./_files/utils_check_docstrings_py.md) |
| ✅ | `utils/check_doctest_list.py` | 86 | Validates doctest example lists | — | [→](./_files/utils_check_doctest_list_py.md) |
| ✅ | `utils/check_dummies.py` | 255 | Generates dummy objects for optional dependencies | — | [→](./_files/utils_check_dummies_py.md) |
| ✅ | `utils/check_inits.py` | 353 | Validates __init__.py exports match implementations | — | [→](./_files/utils_check_inits_py.md) |
| ✅ | `utils/check_model_tester.py` | 59 | Validates model tester implementations | — | [→](./_files/utils_check_model_tester_py.md) |
| ✅ | `utils/check_modeling_structure.py` | 150 | Validates modeling file structure conventions | — | [→](./_files/utils_check_modeling_structure_py.md) |
| ✅ | `utils/check_modular_conversion.py` | 247 | Validates modular model conversion outputs | — | [→](./_files/utils_check_modular_conversion_py.md) |
| ✅ | `utils/check_pipeline_typing.py` | 93 | Validates pipeline type annotations | — | [→](./_files/utils_check_pipeline_typing_py.md) |
| ✅ | `utils/check_repo.py` | 1309 | Automated repository health and consistency checker | — | [→](./_files/utils_check_repo_py.md) |
| ✅ | `utils/check_self_hosted_runner.py` | 57 | Validates self-hosted runner configuration | — | [→](./_files/utils_check_self_hosted_runner_py.md) |
| ✅ | `utils/collated_reports.py` | 217 | Aggregates and formats test reports | — | [→](./_files/utils_collated_reports_py.md) |
| ✅ | `utils/compare_test_runs.py` | 91 | Compares test results between runs | — | [→](./_files/utils_compare_test_runs_py.md) |
| ✅ | `utils/create_dependency_mapping.py` | 116 | Generates dependency graphs for documentation | — | [→](./_files/utils_create_dependency_mapping_py.md) |
| ✅ | `utils/create_dummy_models.py` | 1479 | Creates tiny model checkpoints for testing | — | [→](./_files/utils_create_dummy_models_py.md) |
| ✅ | `utils/custom_init_isort.py` | 331 | Custom isort configuration for transformers imports | — | [→](./_files/utils_custom_init_isort_py.md) |
| ✅ | `utils/deprecate_models.py` | 377 | Automates model deprecation workflow | — | [→](./_files/utils_deprecate_models_py.md) |
| ✅ | `utils/download_glue_data.py` | 160 | Downloads GLUE benchmark datasets | — | [→](./_files/utils_download_glue_data_py.md) |
| ✅ | `utils/extract_pr_number_from_circleci.py` | 31 | Extracts PR numbers from CircleCI context | — | [→](./_files/utils_extract_pr_number_from_circleci_py.md) |
| ✅ | `utils/extract_warnings.py` | 134 | Extracts and categorizes warnings from logs | — | [→](./_files/utils_extract_warnings_py.md) |
| ✅ | `utils/fetch_hub_objects_for_ci.py` | 216 | Pre-fetches Hub objects for CI testing | — | [→](./_files/utils_fetch_hub_objects_for_ci_py.md) |
| ✅ | `utils/get_ci_error_statistics.py` | 305 | Analyzes CI error patterns and statistics | — | [→](./_files/utils_get_ci_error_statistics_py.md) |
| ✅ | `utils/get_github_job_time.py` | 71 | Retrieves GitHub Actions job timing data | — | [→](./_files/utils_get_github_job_time_py.md) |
| ✅ | `utils/get_modified_files.py` | 36 | Lists files modified in git commits | — | [→](./_files/utils_get_modified_files_py.md) |
| ✅ | `utils/get_pr_run_slow_jobs.py` | 133 | Identifies slow CI jobs for PRs | — | [→](./_files/utils_get_pr_run_slow_jobs_py.md) |
| ✅ | `utils/get_previous_daily_ci.py` | 159 | Retrieves previous daily CI results for comparison | — | [→](./_files/utils_get_previous_daily_ci_py.md) |
| ✅ | `utils/get_test_info.py` | 197 | Extracts test metadata and statistics | — | [→](./_files/utils_get_test_info_py.md) |
| ✅ | `utils/get_test_reports.py` | 271 | Downloads and parses test reports | — | [→](./_files/utils_get_test_reports_py.md) |
| ✅ | `utils/important_files.py` | 29 | Defines list of critical repository files | — | [→](./_files/utils_important_files_py.md) |
| ✅ | `utils/models_to_deprecate.py` | 335 | Identifies models eligible for deprecation | — | [→](./_files/utils_models_to_deprecate_py.md) |
| ✅ | `utils/modular_integrations.py` | 184 | Handles modular model integrations | — | [→](./_files/utils_modular_integrations_py.md) |
| ✅ | `utils/modular_model_converter.py` | 1920 | Converts models to modular architecture | — | [→](./_files/utils_modular_model_converter_py.md) |
| ✅ | `utils/modular_model_detector.py` | 913 | Detects modular model patterns | — | [→](./_files/utils_modular_model_detector_py.md) |
| ✅ | `utils/notification_service.py` | 1622 | Posts CI/CD results to Slack and GitHub | — | [→](./_files/utils_notification_service_py.md) |
| ✅ | `utils/notification_service_doc_tests.py` | 384 | Documentation test notifications | — | [→](./_files/utils_notification_service_doc_tests_py.md) |
| ✅ | `utils/patch_helper.py` | 156 | Utilities for applying code patches | — | [→](./_files/utils_patch_helper_py.md) |
| ✅ | `utils/pr_slow_ci_models.py` | 175 | Identifies models causing slow CI | — | [→](./_files/utils_pr_slow_ci_models_py.md) |
| ✅ | `utils/print_env.py` | 76 | Prints environment information for debugging | — | [→](./_files/utils_print_env_py.md) |
| ✅ | `utils/process_bad_commit_report.py` | 141 | Processes bad commit detection reports | — | [→](./_files/utils_process_bad_commit_report_py.md) |
| ✅ | `utils/process_circleci_workflow_test_reports.py` | 146 | Processes CircleCI test reports | — | [→](./_files/utils_process_circleci_workflow_test_reports_py.md) |
| ✅ | `utils/process_test_artifacts.py` | 75 | Processes test artifacts from CI | — | [→](./_files/utils_process_test_artifacts_py.md) |
| ✅ | `utils/release.py` | 227 | Automates release process tasks | — | [→](./_files/utils_release_py.md) |
| ✅ | `utils/scan_skipped_tests.py` | 199 | Identifies and reports skipped tests | — | [→](./_files/utils_scan_skipped_tests_py.md) |
| ✅ | `utils/set_cuda_devices_for_ci.py` | 26 | Configures CUDA devices for CI environments | — | [→](./_files/utils_set_cuda_devices_for_ci_py.md) |
| ✅ | `utils/sort_auto_mappings.py` | 124 | Sorts auto-mapping dictionaries alphabetically | — | [→](./_files/utils_sort_auto_mappings_py.md) |
| ✅ | `utils/split_doctest_jobs.py` | 98 | Splits doctest jobs for parallel execution | — | [→](./_files/utils_split_doctest_jobs_py.md) |
| ✅ | `utils/split_model_tests.py` | 88 | Splits model tests for parallel CI execution | — | [→](./_files/utils_split_model_tests_py.md) |
| ✅ | `utils/tests_fetcher.py` | 1187 | Determines which tests to run based on changes | — | [→](./_files/utils_tests_fetcher_py.md) |
| ✅ | `utils/update_metadata.py` | 350 | Updates model metadata files | — | [→](./_files/utils_update_metadata_py.md) |
| ✅ | `utils/update_tiny_models.py` | 171 | Updates tiny model checkpoints | — | [→](./_files/utils_update_tiny_models_py.md) |

## 📝 Example Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `examples/3D_parallel.py` | 434 | Demonstrates 3D parallelism training (TP, DP, CP) | Workflow: huggingface_transformers_Distributed_Training_3D_Parallelism | [→](./_files/examples_3D_parallel_py.md) |
| ✅ | `examples/run_on_remote.py` | 70 | Runs example scripts on remote GPU clusters | — | [→](./_files/examples_run_on_remote_py.md) |
| ✅ | `scripts/check_tokenizers.py` | 179 | Validates fast/slow tokenizer equivalence | Workflow: huggingface_transformers_Tokenization_Pipeline | [→](./_files/scripts_check_tokenizers_py.md) |
| ✅ | `scripts/stale.py` | 76 | Manages stale GitHub issues | — | [→](./_files/scripts_stale_py.md) |

## 🧪 Test Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `tests/__init__.py` | 0 | Package initializer for tests | — | [→](./_files/tests___init___py.md) |
| ✅ | `tests/causal_lm_tester.py` | 667 | Reusable testing infrastructure for causal LMs | — | [→](./_files/tests_causal_lm_tester_py.md) |
| ✅ | `tests/test_backbone_common.py` | 226 | Test mixin for backbone vision models | — | [→](./_files/tests_test_backbone_common_py.md) |
| ✅ | `tests/test_configuration_common.py` | 243 | Test framework for model configuration classes | — | [→](./_files/tests_test_configuration_common_py.md) |
| ✅ | `tests/test_executorch.py` | 129 | Tests ExecuTorch export for decoder models | — | [→](./_files/tests_test_executorch_py.md) |
| ✅ | `tests/test_feature_extraction_common.py` | 54 | Test mixin for feature extractor save/load | — | [→](./_files/tests_test_feature_extraction_common_py.md) |
| ✅ | `tests/test_image_processing_common.py` | 805 | Comprehensive test framework for image processors | — | [→](./_files/tests_test_image_processing_common_py.md) |
| ✅ | `tests/test_image_transforms.py` | 647 | Unit tests for image transformation functions | — | [→](./_files/tests_test_image_transforms_py.md) |
| ✅ | `tests/test_modeling_common.py` | 4372 | Core ModelTesterMixin for all model architectures | Workflow: huggingface_transformers_Model_Loading | [→](./_files/tests_test_modeling_common_py.md) |
| ✅ | `tests/test_pipeline_mixin.py` | 981 | Automated pipeline testing infrastructure | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/tests_test_pipeline_mixin_py.md) |
| ✅ | `tests/test_processing_common.py` | 1880 | Multimodal processor testing framework | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/tests_test_processing_common_py.md) |
| ✅ | `tests/test_sentencepiece_backend_mixin.py` | 391 | Test mixin for SentencePiece tokenizers | Workflow: huggingface_transformers_Tokenization_Pipeline | [→](./_files/tests_test_sentencepiece_backend_mixin_py.md) |
| ✅ | `tests/test_sequence_feature_extraction_common.py` | 392 | Test mixin for audio/sequence feature extraction | — | [→](./_files/tests_test_sequence_feature_extraction_common_py.md) |
| ✅ | `tests/test_tokenization_common.py` | 2829 | Comprehensive tokenizer testing framework | Workflow: huggingface_transformers_Tokenization_Pipeline | [→](./_files/tests_test_tokenization_common_py.md) |
| ✅ | `tests/test_tokenization_mistral_common.py` | 2132 | Test suite for Mistral tokenizer implementations | Workflow: huggingface_transformers_Tokenization_Pipeline | [→](./_files/tests_test_tokenization_mistral_common_py.md) |
| ✅ | `tests/test_tokenizers_backend_mixin.py` | 460 | Test mixin for fast tokenizer backends | Workflow: huggingface_transformers_Tokenization_Pipeline | [→](./_files/tests_test_tokenizers_backend_mixin_py.md) |
| ✅ | `tests/test_training_args.py` | 67 | Unit tests for TrainingArguments validation | Workflow: huggingface_transformers_Model_Training_Trainer | [→](./_files/tests_test_training_args_py.md) |
| ✅ | `tests/test_training_mixin.py` | 419 | Training overfit test mixin for model verification | Workflow: huggingface_transformers_Model_Training_Trainer | [→](./_files/tests_test_training_mixin_py.md) |
| ✅ | `tests/test_video_processing_common.py` | 527 | Video processor testing framework | — | [→](./_files/tests_test_video_processing_common_py.md) |

## 📄 Other Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `.circleci/create_circleci_config.py` | 412 | Generates CircleCI configuration YAML | Impl:huggingface_transformers_CircleCIJob | [→](./_files/_circleci_create_circleci_config_py.md) |
| ✅ | `.circleci/parse_test_outputs.py` | 71 | Parses pytest output for human-readable summaries | — | [→](./_files/_circleci_parse_test_outputs_py.md) |
| ✅ | `conftest.py` | 152 | Pytest configuration for transformers test suite | — | [→](./_files/conftest_py.md) |
| ✅ | `setup.py` | 428 | Package installation configuration for pip/PyPI | Impl:huggingface_transformers_PackageSetup | [→](./_files/setup_py.md) |
| ✅ | `src/transformers/__init__.py` | 832 | Main package init with lazy loading | Impl:huggingface_transformers_LazyImportSystem | [→](./_files/src_transformers___init___py.md) |
| ✅ | `src/transformers/activations.py` | 360 | Collection of activation functions (GELU, SiLU, etc.) | Impl:huggingface_transformers_ActivationFunctions | [→](./_files/src_transformers_activations_py.md) |
| ✅ | `src/transformers/audio_utils.py` | 1238 | Pure numpy audio processing utilities | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_audio_utils_py.md) |
| ✅ | `src/transformers/cache_utils.py` | 1402 | KV cache management for generation | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_cache_utils_py.md) |
| ✅ | `src/transformers/configuration_utils.py` | 1228 | Base configuration class for all models | Workflow: huggingface_transformers_Model_Loading | [→](./_files/src_transformers_configuration_utils_py.md) |
| ✅ | `src/transformers/conversion_mapping.py` | 274 | Registry for checkpoint weight conversion mappings | Workflow: huggingface_transformers_Model_Loading | [→](./_files/src_transformers_conversion_mapping_py.md) |
| ✅ | `src/transformers/convert_slow_tokenizer.py` | 2083 | Converts slow tokenizers to fast tokenizers | Workflow: huggingface_transformers_Tokenization_Pipeline | [→](./_files/src_transformers_convert_slow_tokenizer_py.md) |
| ✅ | `src/transformers/convert_slow_tokenizers_checkpoints_to_fast.py` | 149 | CLI for batch tokenizer conversion | Workflow: huggingface_transformers_Tokenization_Pipeline | [→](./_files/src_transformers_convert_slow_tokenizers_checkpoints_to_fast_py.md) |
| ✅ | `src/transformers/core_model_loading.py` | 1031 | Core checkpoint loading with weight transformations | Workflow: huggingface_transformers_Model_Loading | [→](./_files/src_transformers_core_model_loading_py.md) |
| ✅ | `src/transformers/data/__init__.py` | 46 | Package init for data processing utilities | — | [→](./_files/src_transformers_data___init___py.md) |
| ✅ | `src/transformers/data/data_collator.py` | 1462 | Data collators for batching and padding | Workflow: huggingface_transformers_Model_Training_Trainer | [→](./_files/src_transformers_data_data_collator_py.md) |
| ✅ | `src/transformers/debug_utils.py` | 346 | Debugging utilities for numerical instability | Impl:huggingface_transformers_DebugUnderflowOverflow | [→](./_files/src_transformers_debug_utils_py.md) |
| ✅ | `src/transformers/dependency_versions_check.py` | 63 | Validates package dependency versions | Impl:huggingface_transformers_DependencyVersionsCheck | [→](./_files/src_transformers_dependency_versions_check_py.md) |
| ✅ | `src/transformers/dependency_versions_table.py` | 95 | Centralized dependency version requirements | Impl:huggingface_transformers_DependencyVersionsTable | [→](./_files/src_transformers_dependency_versions_table_py.md) |
| ✅ | `src/transformers/dynamic_module_utils.py` | 810 | Loads custom modules from Hub repos | Workflow: huggingface_transformers_Model_Loading | [→](./_files/src_transformers_dynamic_module_utils_py.md) |
| ✅ | `src/transformers/feature_extraction_sequence_utils.py` | 386 | Base class for audio/sequence feature extraction | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_feature_extraction_sequence_utils_py.md) |
| ✅ | `src/transformers/feature_extraction_utils.py` | 668 | Core feature extractor infrastructure | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_feature_extraction_utils_py.md) |
| ✅ | `src/transformers/file_utils.py` | 107 | Backward compatibility shim for utils | — | [→](./_files/src_transformers_file_utils_py.md) |
| ✅ | `src/transformers/hf_argparser.py` | 430 | Enhanced argument parser from dataclasses | Impl:huggingface_transformers_HfArgumentParser | [→](./_files/src_transformers_hf_argparser_py.md) |
| ✅ | `src/transformers/hyperparameter_search.py` | 124 | Unified hyperparameter optimization interface | Workflow: huggingface_transformers_Model_Training_Trainer | [→](./_files/src_transformers_hyperparameter_search_py.md) |
| ✅ | `src/transformers/image_processing_base.py` | 486 | Foundational mixin for image processor loading | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_image_processing_base_py.md) |
| ✅ | `src/transformers/image_processing_utils.py` | 320 | Base image processor with standard operations | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_image_processing_utils_py.md) |
| ✅ | `src/transformers/image_processing_utils_fast.py` | 953 | PyTorch/TorchVision-accelerated image processing | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_image_processing_utils_fast_py.md) |
| ✅ | `src/transformers/image_transforms.py` | 1001 | Core image transformation functions | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_image_transforms_py.md) |
| ✅ | `src/transformers/image_utils.py` | 959 | Central image utility module | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_image_utils_py.md) |
| ✅ | `src/transformers/initialization.py` | 208 | Guarded tensor initialization functions | Impl:huggingface_transformers_TensorInitialization | [→](./_files/src_transformers_initialization_py.md) |
| ✅ | `src/transformers/masking_utils.py` | 1381 | Unified attention mask creation system | Workflow: huggingface_transformers_Model_Loading, huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_masking_utils_py.md) |
| ✅ | `src/transformers/model_debugging_utils.py` | 456 | Developer tool for forward pass debugging | Impl:huggingface_transformers_ModelDebuggingUtils | [→](./_files/src_transformers_model_debugging_utils_py.md) |
| ✅ | `src/transformers/modelcard.py` | 767 | Automated model card generation | Impl:huggingface_transformers_ModelCard ⚠️DEPRECATED | [→](./_files/src_transformers_modelcard_py.md) |
| ✅ | `src/transformers/modeling_attn_mask_utils.py` | 485 | Legacy attention mask utilities (deprecated) | Impl:huggingface_transformers_AttentionMaskUtils | [→](./_files/src_transformers_modeling_attn_mask_utils_py.md) |
| ✅ | `src/transformers/modeling_flash_attention_utils.py` | 706 | Flash Attention integration utilities | Workflow: huggingface_transformers_Model_Loading | [→](./_files/src_transformers_modeling_flash_attention_utils_py.md) |
| ✅ | `src/transformers/modeling_gguf_pytorch_utils.py` | 587 | GGUF quantized weight loading for PyTorch | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_modeling_gguf_pytorch_utils_py.md) |
| ✅ | `src/transformers/modeling_layers.py` | 289 | Reusable base layers and task-specific heads | Impl:huggingface_transformers_ModelingLayers | [→](./_files/src_transformers_modeling_layers_py.md) |
| ✅ | `src/transformers/modeling_outputs.py` | 1717 | Standardized output dataclasses for all models | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_modeling_outputs_py.md) |
| ✅ | `src/transformers/modeling_rope_utils.py` | 939 | Rotary Position Embedding utilities | Impl:huggingface_transformers_RoPEUtils | [→](./_files/src_transformers_modeling_rope_utils_py.md) |
| ✅ | `src/transformers/modeling_utils.py` | 4671 | Core PreTrainedModel base class | Workflow: huggingface_transformers_Model_Loading, huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_modeling_utils_py.md) |
| ✅ | `src/transformers/optimization.py` | 972 | Learning rate schedulers and Adafactor optimizer | Workflow: huggingface_transformers_Model_Training_Trainer | [→](./_files/src_transformers_optimization_py.md) |
| ✅ | `src/transformers/pipelines/__init__.py` | 1086 | Central pipeline registry and factory | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines___init___py.md) |
| ✅ | `src/transformers/pipelines/any_to_any.py` | 505 | Multimodal any-to-any pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_any_to_any_py.md) |
| ✅ | `src/transformers/pipelines/audio_classification.py` | 259 | Audio classification pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_audio_classification_py.md) |
| ✅ | `src/transformers/pipelines/audio_utils.py` | 296 | Audio I/O utilities for pipelines | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_audio_utils_py.md) |
| ✅ | `src/transformers/pipelines/automatic_speech_recognition.py` | 684 | Speech-to-text transcription pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_automatic_speech_recognition_py.md) |
| ✅ | `src/transformers/pipelines/base.py` | 1394 | Foundational Pipeline base class | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_base_py.md) |
| ✅ | `src/transformers/pipelines/depth_estimation.py` | 145 | Monocular depth estimation pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_depth_estimation_py.md) |
| ✅ | `src/transformers/pipelines/document_question_answering.py` | 546 | Document QA with visual layout understanding | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_document_question_answering_py.md) |
| ✅ | `src/transformers/pipelines/feature_extraction.py` | 88 | Raw embedding extraction pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_feature_extraction_py.md) |
| ✅ | `src/transformers/pipelines/fill_mask.py` | 259 | Masked token prediction pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_fill_mask_py.md) |
| ✅ | `src/transformers/pipelines/image_classification.py` | 229 | Image classification pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_image_classification_py.md) |
| ✅ | `src/transformers/pipelines/image_feature_extraction.py` | 115 | Vision model feature extraction pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_image_feature_extraction_py.md) |
| ✅ | `src/transformers/pipelines/image_segmentation.py` | 223 | Unified image segmentation pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_image_segmentation_py.md) |
| ✅ | `src/transformers/pipelines/image_text_to_text.py` | 455 | Multimodal text generation from images | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_image_text_to_text_py.md) |
| ✅ | `src/transformers/pipelines/image_to_image.py` | 145 | Image transformation pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_image_to_image_py.md) |
| ✅ | `src/transformers/pipelines/image_to_text.py` | 229 | Image captioning pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_image_to_text_py.md) |
| ✅ | `src/transformers/pipelines/keypoint_matching.py` | 176 | Keypoint detection and matching pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_keypoint_matching_py.md) |
| ✅ | `src/transformers/pipelines/mask_generation.py` | 335 | Automatic mask generation (SAM) pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_mask_generation_py.md) |
| ✅ | `src/transformers/pipelines/object_detection.py` | 197 | Object detection pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_object_detection_py.md) |
| ✅ | `src/transformers/pipelines/pt_utils.py` | 323 | PyTorch-specific pipeline utilities | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_pt_utils_py.md) |
| ✅ | `src/transformers/pipelines/question_answering.py` | 685 | Extractive QA pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_question_answering_py.md) |
| ✅ | `src/transformers/pipelines/table_question_answering.py` | 382 | Table QA pipeline (TAPAS/seq2seq) | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_table_question_answering_py.md) |
| ✅ | `src/transformers/pipelines/text_classification.py` | 235 | Text classification pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_text_classification_py.md) |
| ✅ | `src/transformers/pipelines/text_generation.py` | 500 | Autoregressive text generation pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_text_generation_py.md) |
| ✅ | `src/transformers/pipelines/text_to_audio.py` | 311 | Text-to-speech/music generation pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_text_to_audio_py.md) |
| ✅ | `src/transformers/pipelines/token_classification.py` | 646 | Token classification (NER) pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_token_classification_py.md) |
| ✅ | `src/transformers/pipelines/video_classification.py` | 191 | Video classification pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_video_classification_py.md) |
| ✅ | `src/transformers/pipelines/visual_question_answering.py` | 212 | Visual QA pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_visual_question_answering_py.md) |
| ✅ | `src/transformers/pipelines/zero_shot_audio_classification.py` | 161 | Zero-shot audio classification (CLAP) | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_zero_shot_audio_classification_py.md) |
| ✅ | `src/transformers/pipelines/zero_shot_classification.py` | 267 | Zero-shot text classification (NLI) | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_zero_shot_classification_py.md) |
| ✅ | `src/transformers/pipelines/zero_shot_image_classification.py` | 202 | Zero-shot image classification (CLIP) | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_zero_shot_image_classification_py.md) |
| ✅ | `src/transformers/pipelines/zero_shot_object_detection.py` | 242 | Zero-shot object detection pipeline | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_pipelines_zero_shot_object_detection_py.md) |
| ✅ | `src/transformers/processing_utils.py` | 1922 | ProcessorMixin base class for multimodal inputs | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_processing_utils_py.md) |
| ✅ | `src/transformers/pytorch_utils.py` | 284 | PyTorch utility functions and custom layers | Impl:huggingface_transformers_PyTorchUtils | [→](./_files/src_transformers_pytorch_utils_py.md) |
| ✅ | `src/transformers/quantizers/__init__.py` | 16 | Quantizers module public API entry point | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers___init___py.md) |
| ✅ | `src/transformers/quantizers/auto.py` | 338 | Automatic quantizer dispatching system | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_auto_py.md) |
| ✅ | `src/transformers/quantizers/base.py` | 354 | Abstract base class for all quantizers | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_base_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_aqlm.py` | 73 | AQLM (Additive Quantization) support | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_aqlm_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_auto_round.py` | 71 | AutoRound data-driven quantization | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_auto_round_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_awq.py` | 95 | AWQ 4-bit activation-aware quantization | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_awq_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_bitnet.py` | 109 | BitNet 1.58-bit ternary quantization | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_bitnet_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_bnb_8bit.py` | 172 | bitsandbytes 8-bit INT8 quantization | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_bnb_8bit_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_compressed_tensors.py` | 111 | compressed-tensors library integration | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_compressed_tensors_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_eetq.py` | 108 | EETQ 8-bit GPU-accelerated quantization | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_eetq_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_fbgemm_fp8.py` | 187 | FBGEMM FP8 quantization for H100+ GPUs | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_fbgemm_fp8_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_finegrained_fp8.py` | 162 | Fine-grained FP8 with MoE support | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_finegrained_fp8_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_fp_quant.py` | 150 | FP-Quant for Blackwell GPUs | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_fp_quant_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_gptq.py` | 104 | GPTQ post-training quantization | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_gptq_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_higgs.py` | 176 | HIGGS Hadamard-based quantization | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_higgs_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_hqq.py` | 262 | HQQ half-quadratic quantization | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_hqq_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_mxfp4.py` | 292 | MXFP4 microscaling FP4 quantization | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_mxfp4_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_quanto.py` | 119 | optimum-quanto library integration | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_quanto_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_quark.py` | 115 | AMD Quark quantization library support | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_quark_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_spqr.py` | 79 | SpQR sparse quantization | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_spqr_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_torchao.py` | 356 | PyTorch torchao quantization integration | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_torchao_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_vptq.py` | 73 | VPTQ vector post-training quantization | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizer_vptq_py.md) |
| ✅ | `src/transformers/quantizers/quantizers_utils.py` | 41 | Shared quantizer utility functions | Workflow: huggingface_transformers_Model_Quantization | [→](./_files/src_transformers_quantizers_quantizers_utils_py.md) |
| ✅ | `src/transformers/safetensors_conversion.py` | 110 | On-the-fly .bin to safetensors conversion | Workflow: huggingface_transformers_Model_Loading | [→](./_files/src_transformers_safetensors_conversion_py.md) |
| ✅ | `src/transformers/testing_utils.py` | 4366 | Comprehensive testing utilities and decorators | Impl:huggingface_transformers_TestingUtils | [→](./_files/src_transformers_testing_utils_py.md) |
| ✅ | `src/transformers/time_series_utils.py` | 225 | Probability distributions for time series | Impl:huggingface_transformers_TimeSeriesUtils | [→](./_files/src_transformers_time_series_utils_py.md) |
| ✅ | `src/transformers/tokenization_mistral_common.py` | 1992 | Mistral tokenizer wrapper (SPM/Tekken) | Workflow: huggingface_transformers_Tokenization_Pipeline | [→](./_files/src_transformers_tokenization_mistral_common_py.md) |
| ✅ | `src/transformers/tokenization_python.py` | 1400 | Base class for Python-based slow tokenizers | Workflow: huggingface_transformers_Tokenization_Pipeline | [→](./_files/src_transformers_tokenization_python_py.md) |
| ✅ | `src/transformers/tokenization_utils_base.py` | 3639 | Base interface for all tokenizers | Workflow: huggingface_transformers_Tokenization_Pipeline | [→](./_files/src_transformers_tokenization_utils_base_py.md) |
| ✅ | `src/transformers/tokenization_utils_sentencepiece.py` | 316 | SentencePiece tokenizer backend | Workflow: huggingface_transformers_Tokenization_Pipeline | [→](./_files/src_transformers_tokenization_utils_sentencepiece_py.md) |
| ✅ | `src/transformers/tokenization_utils_tokenizers.py` | 1249 | Fast tokenizer backend (Rust-based) | Workflow: huggingface_transformers_Tokenization_Pipeline | [→](./_files/src_transformers_tokenization_utils_tokenizers_py.md) |
| ✅ | `src/transformers/trainer.py` | 5324 | Main Trainer class for PyTorch training | Workflow: huggingface_transformers_Model_Training_Trainer | [→](./_files/src_transformers_trainer_py.md) |
| ✅ | `src/transformers/trainer_callback.py` | 776 | Callback system for Trainer customization | Workflow: huggingface_transformers_Model_Training_Trainer | [→](./_files/src_transformers_trainer_callback_py.md) |
| ✅ | `src/transformers/trainer_jit_checkpoint.py` | 126 | JIT checkpointing for cloud preemption | Workflow: huggingface_transformers_Model_Training_Trainer | [→](./_files/src_transformers_trainer_jit_checkpoint_py.md) |
| ✅ | `src/transformers/trainer_pt_utils.py` | 1242 | PyTorch-specific Trainer utilities | Workflow: huggingface_transformers_Model_Training_Trainer | [→](./_files/src_transformers_trainer_pt_utils_py.md) |
| ✅ | `src/transformers/trainer_seq2seq.py` | 390 | Seq2seq-specific Trainer extensions | Workflow: huggingface_transformers_Model_Training_Trainer | [→](./_files/src_transformers_trainer_seq2seq_py.md) |
| ✅ | `src/transformers/trainer_utils.py` | 957 | Shared training utilities and data structures | Workflow: huggingface_transformers_Model_Training_Trainer | [→](./_files/src_transformers_trainer_utils_py.md) |
| ✅ | `src/transformers/training_args.py` | 2809 | TrainingArguments configuration dataclass | Workflow: huggingface_transformers_Model_Training_Trainer | [→](./_files/src_transformers_training_args_py.md) |
| ✅ | `src/transformers/training_args_seq2seq.py` | 89 | Seq2seq-specific training arguments | Workflow: huggingface_transformers_Model_Training_Trainer | [→](./_files/src_transformers_training_args_seq2seq_py.md) |
| ✅ | `src/transformers/video_processing_utils.py` | 888 | BaseVideoProcessor for video preprocessing | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_video_processing_utils_py.md) |
| ✅ | `src/transformers/video_utils.py` | 893 | Video data utilities and loaders | Workflow: huggingface_transformers_Pipeline_Inference | [→](./_files/src_transformers_video_utils_py.md) |

---

## Page Indexes

Each page type has its own index file for tracking and integrity checking:

| Index | Description |
|-------|-------------|
| [Workflows](./_WorkflowIndex.md) | Workflow pages with step connections |
| [Principles](./_PrincipleIndex.md) | Principle pages with implementations |
| [Implementations](./_ImplementationIndex.md) | Implementation pages with source locations |
| [Environments](./_EnvironmentIndex.md) | Environment requirement pages |
| [Heuristics](./_HeuristicIndex.md) | Heuristic/tips pages |
