# Repository Map: huggingface_transformers

> **Compact index** of repository files.
> Each file has a detail page in `_files/` with Understanding to fill.
> Mark files as ✅ explored in the table below as you complete them.

| Property | Value |
|----------|-------|
| Repository | https://github.com/huggingface/transformers |
| Branch | main |
| Generated | 2025-12-17 18:59 |
| Python Files | 200 |
| Total Lines | 114,314 |
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
| ✅ | `benchmark/__init__.py` | 0 | Package initialization marker | — | [→](./_files/benchmark___init___py.md) |
| ✅ | `benchmark/benchmark.py` | 324 | Multi-commit performance orchestrator | Impl: benchmark | [→](./_files/benchmark_benchmark_py.md) |
| ✅ | `benchmark/benchmarks_entrypoint.py` | 502 | Automated metrics collection system | Impl: benchmarks_entrypoint | [→](./_files/benchmark_benchmarks_entrypoint_py.md) |
| ✅ | `benchmark/optimum_benchmark_wrapper.py` | 20 | Optimum-benchmark CLI wrapper | — | [→](./_files/benchmark_optimum_benchmark_wrapper_py.md) |
| ✅ | `benchmark_v2/run_benchmarks.py` | 128 | Orchestrates performance benchmark execution | — | [→](./_files/benchmark_v2_run_benchmarks_py.md) |
| ✅ | `utils/add_dates.py` | 427 | Manages model documentation dates | Impl: add_dates | [→](./_files/utils_add_dates_py.md) |
| ✅ | `utils/add_pipeline_model_mapping_to_test.py` | 308 | Generates pipeline test mappings | — | [→](./_files/utils_add_pipeline_model_mapping_to_test_py.md) |
| ✅ | `utils/check_bad_commit.py` | 280 | Identifies test-breaking commits | — | [→](./_files/utils_check_bad_commit_py.md) |
| ✅ | `utils/check_config_attributes.py` | 548 | Validates config parameter usage | Impl: check_config_attributes | [→](./_files/utils_check_config_attributes_py.md) |
| ✅ | `utils/check_config_docstrings.py` | 102 | Ensures config checkpoint links | — | [→](./_files/utils_check_config_docstrings_py.md) |
| ✅ | `utils/check_copies.py` | 1044 | Enforces code duplication consistency | Impl: check_copies | [→](./_files/utils_check_copies_py.md) |
| ✅ | `utils/check_doc_toc.py` | 133 | Maintains documentation table contents | — | [→](./_files/utils_check_doc_toc_py.md) |
| ✅ | `utils/check_docstrings.py` | 1559 | Validates signature-docstring matching | Impl: check_docstrings | [→](./_files/utils_check_docstrings_py.md) |
| ✅ | `utils/check_doctest_list.py` | 86 | Maintains doctest configuration lists | — | [→](./_files/utils_check_doctest_list_py.md) |
| ✅ | `utils/check_dummies.py` | 255 | Generates backend-specific dummy objects | — | [→](./_files/utils_check_dummies_py.md) |
| ✅ | `utils/check_inits.py` | 353 | Validates delayed import system | Impl: check_inits | [→](./_files/utils_check_inits_py.md) |
| ✅ | `utils/check_model_tester.py` | 59 | Validates test configuration sizes | — | [→](./_files/utils_check_model_tester_py.md) |
| ✅ | `utils/check_modeling_structure.py` | 150 | Enforces modeling file conventions | — | [→](./_files/utils_check_modeling_structure_py.md) |
| ✅ | `utils/check_modular_conversion.py` | 247 | Validates modular-to-traditional conversion | — | [→](./_files/utils_check_modular_conversion_py.md) |
| ✅ | `utils/check_pipeline_typing.py` | 93 | Generates pipeline type overloads | — | [→](./_files/utils_check_pipeline_typing_py.md) |
| ✅ | `utils/check_repo.py` | 1309 | Repository consistency validation | Impl: check_repo | [→](./_files/utils_check_repo_py.md) |
| ✅ | `utils/check_self_hosted_runner.py` | 57 | Runner availability monitoring | — | [→](./_files/utils_check_self_hosted_runner_py.md) |
| ✅ | `utils/collated_reports.py` | 217 | Test result aggregation | — | [→](./_files/utils_collated_reports_py.md) |
| ✅ | `utils/compare_test_runs.py` | 91 | CI run comparison | — | [→](./_files/utils_compare_test_runs_py.md) |
| ✅ | `utils/create_dependency_mapping.py` | 116 | Model dependency analysis | — | [→](./_files/utils_create_dependency_mapping_py.md) |
| ✅ | `utils/create_dummy_models.py` | 1479 | Tiny model generation | Impl: create_dummy_models | [→](./_files/utils_create_dummy_models_py.md) |
| ✅ | `utils/custom_init_isort.py` | 331 | Import sorting automation | Impl: custom_init_isort | [→](./_files/utils_custom_init_isort_py.md) |
| ✅ | `utils/deprecate_models.py` | 377 | Model deprecation automation | Impl: deprecate_models | [→](./_files/utils_deprecate_models_py.md) |
| ✅ | `utils/download_glue_data.py` | 160 | GLUE benchmark downloader | Impl: download_glue_data | [→](./_files/utils_download_glue_data_py.md) |
| ✅ | `utils/extract_pr_number_from_circleci.py` | 31 | CircleCI PR extraction | — | [→](./_files/utils_extract_pr_number_from_circleci_py.md) |
| ✅ | `utils/extract_warnings.py` | 134 | Warning extraction tool | — | [→](./_files/utils_extract_warnings_py.md) |
| ✅ | `utils/fetch_hub_objects_for_ci.py` | 216 | CI asset pre-fetcher | — | [→](./_files/utils_fetch_hub_objects_for_ci_py.md) |
| ✅ | `utils/get_ci_error_statistics.py` | 305 | CI failure analyzer | Impl: get_ci_error_statistics | [→](./_files/utils_get_ci_error_statistics_py.md) |
| ✅ | `utils/get_github_job_time.py` | 71 | Job duration analyzer | — | [→](./_files/utils_get_github_job_time_py.md) |
| ✅ | `utils/get_modified_files.py` | 36 | Modified file detector | — | [→](./_files/utils_get_modified_files_py.md) |
| ✅ | `utils/get_pr_run_slow_jobs.py` | 133 | Determines PR slow CI jobs | — | [→](./_files/utils_get_pr_run_slow_jobs_py.md) |
| ✅ | `utils/get_previous_daily_ci.py` | 159 | Retrieves previous CI artifacts | — | [→](./_files/utils_get_previous_daily_ci_py.md) |
| ✅ | `utils/get_test_info.py` | 197 | Introspects model test relationships | — | [→](./_files/utils_get_test_info_py.md) |
| ✅ | `utils/get_test_reports.py` | 271 | Runs tests locally mirroring CI | — | [→](./_files/utils_get_test_reports_py.md) |
| ✅ | `utils/important_files.py` | 29 | Defines priority model list | — | [→](./_files/utils_important_files_py.md) |
| ✅ | `utils/models_to_deprecate.py` | 335 | Identifies low-usage deprecation candidates | Impl: models_to_deprecate | [→](./_files/utils_models_to_deprecate_py.md) |
| ✅ | `utils/modular_integrations.py` | 184 | Converts import statements AST | — | [→](./_files/utils_modular_integrations_py.md) |
| ✅ | `utils/modular_model_converter.py` | 1920 | Generates models from modular definitions | Impl: modular_model_converter | [→](./_files/utils_modular_model_converter_py.md) |
| ✅ | `utils/modular_model_detector.py` | 913 | Detects code similarity opportunities | Impl: modular_model_detector | [→](./_files/utils_modular_model_detector_py.md) |
| ✅ | `utils/notification_service.py` | 1622 | Posts CI reports to Slack | Impl: notification_service | [→](./_files/utils_notification_service_py.md) |
| ✅ | `utils/notification_service_doc_tests.py` | 384 | Reports doctest failures to Slack | Impl: notification_service_doc_tests | [→](./_files/utils_notification_service_doc_tests_py.md) |
| ✅ | `utils/patch_helper.py` | 156 | Automates release branch cherry-picking | — | [→](./_files/utils_patch_helper_py.md) |
| ✅ | `utils/pr_slow_ci_models.py` | 175 | Detects models for slow CI | — | [→](./_files/utils_pr_slow_ci_models_py.md) |
| ✅ | `utils/print_env.py` | 76 | Prints environment diagnostic info | — | [→](./_files/utils_print_env_py.md) |
| ✅ | `utils/process_bad_commit_report.py` | 141 | Attributes failures to authors | — | [→](./_files/utils_process_bad_commit_report_py.md) |
| ✅ | `utils/process_circleci_workflow_test_reports.py` | 146 | Aggregates CircleCI test results | — | [→](./_files/utils_process_circleci_workflow_test_reports_py.md) |
| ✅ | `utils/process_test_artifacts.py` | 75 | Calculates CI parallelism levels | — | [→](./_files/utils_process_test_artifacts_py.md) |
| ✅ | `utils/release.py` | 227 | Automates version and release management | — | [→](./_files/utils_release_py.md) |
| ✅ | `utils/scan_skipped_tests.py` | 199 | Analyzes test coverage gaps | — | [→](./_files/utils_scan_skipped_tests_py.md) |
| ✅ | `utils/set_cuda_devices_for_ci.py` | 26 | Configures GPU device visibility | — | [→](./_files/utils_set_cuda_devices_for_ci_py.md) |
| ✅ | `utils/sort_auto_mappings.py` | 124 | Enforces alphabetical model ordering | — | [→](./_files/utils_sort_auto_mappings_py.md) |
| ✅ | `utils/split_doctest_jobs.py` | 98 | Organizes parallel doctest execution | — | [→](./_files/utils_split_doctest_jobs_py.md) |
| ✅ | `utils/split_model_tests.py` | 88 | Divides tests into slices | — | [→](./_files/utils_split_model_tests_py.md) |
| ✅ | `utils/tests_fetcher.py` | 1187 | Intelligently selects impacted tests | Impl: tests_fetcher | [→](./_files/utils_tests_fetcher_py.md) |
| ✅ | `utils/update_metadata.py` | 350 | Maintains Hub metadata synchronization | Impl: update_metadata | [→](./_files/utils_update_metadata_py.md) |
| ✅ | `utils/update_tiny_models.py` | 171 | Orchestrates tiny model creation | — | [→](./_files/utils_update_tiny_models_py.md) |

## 📝 Example Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `examples/3D_parallel.py` | 434 | 3D parallelism training demonstration | Workflow: Training | [→](./_files/examples_3D_parallel_py.md) |
| ✅ | `examples/run_on_remote.py` | 70 | Remote execution wrapper script | — | [→](./_files/examples_run_on_remote_py.md) |
| ✅ | `scripts/check_tokenizers.py` | 179 | Tokenizer implementation validation | Workflow: Tokenization | [→](./_files/scripts_check_tokenizers_py.md) |
| ✅ | `scripts/stale.py` | 76 | Automated issue lifecycle management | — | [→](./_files/scripts_stale_py.md) |

## 🧪 Test Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `tests/__init__.py` | 0 | Empty Python package marker | — | [→](./_files/tests___init___py.md) |
| ✅ | `tests/causal_lm_tester.py` | 667 | Unified causal LM test framework | — | [→](./_files/tests_causal_lm_tester_py.md) |
| ✅ | `tests/test_backbone_common.py` | 226 | Vision backbone test mixin | — | [→](./_files/tests_test_backbone_common_py.md) |
| ✅ | `tests/test_configuration_common.py` | 243 | Configuration testing framework | Workflow: Model_Loading | [→](./_files/tests_test_configuration_common_py.md) |
| ✅ | `tests/test_executorch.py` | 129 | ExecuTorch export validation tests | — | [→](./_files/tests_test_executorch_py.md) |
| ✅ | `tests/test_feature_extraction_common.py` | 54 | Feature extractor serialization tests | — | [→](./_files/tests_test_feature_extraction_common_py.md) |
| ✅ | `tests/test_image_processing_common.py` | 805 | Image processor equivalence testing | Workflow: Pipeline_Inference | [→](./_files/tests_test_image_processing_common_py.md) |
| ✅ | `tests/test_image_transforms.py` | 647 | Image transformation unit tests | — | [→](./_files/tests_test_image_transforms_py.md) |
| ✅ | `tests/test_modeling_common.py` | 4372 | Massive model testing suite | Workflow: Model_Loading | [→](./_files/tests_test_modeling_common_py.md) |
| ✅ | `tests/test_pipeline_mixin.py` | 981 | Automated pipeline testing framework | Workflow: Pipeline_Inference | [→](./_files/tests_test_pipeline_mixin_py.md) |
| ✅ | `tests/test_processing_common.py` | 1880 | Multimodal processor integration testing | Workflow: Pipeline_Inference | [→](./_files/tests_test_processing_common_py.md) |
| ✅ | `tests/test_sentencepiece_backend_mixin.py` | 391 | SentencePiece tokenizer backend validation | Workflow: Tokenization | [→](./_files/tests_test_sentencepiece_backend_mixin_py.md) |
| ✅ | `tests/test_sequence_feature_extraction_common.py` | 392 | Audio feature extraction testing | Workflow: Pipeline_Inference | [→](./_files/tests_test_sequence_feature_extraction_common_py.md) |
| ✅ | `tests/test_tokenization_common.py` | 2829 | Universal tokenizer behavior verification | Workflow: Tokenization | [→](./_files/tests_test_tokenization_common_py.md) |
| ✅ | `tests/test_tokenization_mistral_common.py` | 2132 | Mistral tokenizer implementation testing | Workflow: Tokenization | [→](./_files/tests_test_tokenization_mistral_common_py.md) |
| ✅ | `tests/test_tokenizers_backend_mixin.py` | 460 | Rust tokenizers alignment methods | Workflow: Tokenization | [→](./_files/tests_test_tokenizers_backend_mixin_py.md) |
| ✅ | `tests/test_training_args.py` | 67 | Training configuration validation | Workflow: Training | [→](./_files/tests_test_training_args_py.md) |
| ✅ | `tests/test_training_mixin.py` | 419 | Model training sanity checks | Workflow: Training | [→](./_files/tests_test_training_mixin_py.md) |
| ✅ | `tests/test_video_processing_common.py` | 527 | Video processor functionality testing | Workflow: Pipeline_Inference | [→](./_files/tests_test_video_processing_common_py.md) |

## 📄 Other Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `.circleci/create_circleci_config.py` | 412 | Generates CircleCI configuration dynamically | Impl: create_circleci_config | [→](./_files/_circleci_create_circleci_config_py.md) |
| ✅ | `.circleci/parse_test_outputs.py` | 71 | Parses pytest outputs for CI | — | [→](./_files/_circleci_parse_test_outputs_py.md) |
| ✅ | `conftest.py` | 152 | Configures pytest behavior and markers | — | [→](./_files/conftest_py.md) |
| ✅ | `setup.py` | 428 | Standard Python package installation | Impl: setup_py | [→](./_files/setup_py.md) |
| ✅ | `src/transformers/__init__.py` | 832 | Lazy loading public API | Workflow: Model_Loading, Pipeline_Inference, Tokenization | [→](./_files/src_transformers___init___py.md) |
| ✅ | `src/transformers/activations.py` | 360 | Neural network activation functions | Workflow: Model_Loading | [→](./_files/src_transformers_activations_py.md) |
| ✅ | `src/transformers/audio_utils.py` | 1238 | Audio processing and spectrograms | Workflow: Pipeline_Inference | [→](./_files/src_transformers_audio_utils_py.md) |
| ✅ | `src/transformers/cache_utils.py` | 1402 | Attention KV cache management | Workflow: Model_Loading, Pipeline_Inference | [→](./_files/src_transformers_cache_utils_py.md) |
| ✅ | `src/transformers/configuration_utils.py` | 1270 | Model configuration base class | Workflow: Model_Loading | [→](./_files/src_transformers_configuration_utils_py.md) |
| ✅ | `src/transformers/conversion_mapping.py` | 274 | Checkpoint weight conversion mappings | Workflow: Model_Loading | [→](./_files/src_transformers_conversion_mapping_py.md) |
| ✅ | `src/transformers/convert_slow_tokenizer.py` | 2083 | Slow to fast tokenizer converter | Workflow: Tokenization | [→](./_files/src_transformers_convert_slow_tokenizer_py.md) |
| ✅ | `src/transformers/convert_slow_tokenizers_checkpoints_to_fast.py` | 149 | Batch tokenizer conversion utility | Workflow: Tokenization | [→](./_files/src_transformers_convert_slow_tokenizers_checkpoints_to_fast_py.md) |
| ✅ | `src/transformers/core_model_loading.py` | 1029 | Checkpoint loading infrastructure core | Workflow: Model_Loading | [→](./_files/src_transformers_core_model_loading_py.md) |
| ✅ | `src/transformers/data/__init__.py` | 46 | Data utilities package initialization | Workflow: Training | [→](./_files/src_transformers_data___init___py.md) |
| ✅ | `src/transformers/data/data_collator.py` | 1462 | Training batch collation utilities | Workflow: Training | [→](./_files/src_transformers_data_data_collator_py.md) |
| ✅ | `src/transformers/debug_utils.py` | 346 | Numerical instability debugging tools | Workflow: Training | [→](./_files/src_transformers_debug_utils_py.md) |
| ✅ | `src/transformers/dependency_versions_check.py` | 63 | Runtime dependency version validation | Impl: dependency_versions_check | [→](./_files/src_transformers_dependency_versions_check_py.md) |
| ✅ | `src/transformers/dependency_versions_table.py` | 95 | Centralized dependency version specs | Impl: dependency_versions_table | [→](./_files/src_transformers_dependency_versions_table_py.md) |
| ✅ | `src/transformers/dynamic_module_utils.py` | 810 | Hub custom code loader | Workflow: Model_Loading | [→](./_files/src_transformers_dynamic_module_utils_py.md) |
| ✅ | `src/transformers/feature_extraction_sequence_utils.py` | 386 | Audio sequence feature extraction | Workflow: Pipeline_Inference | [→](./_files/src_transformers_feature_extraction_sequence_utils_py.md) |
| ✅ | `src/transformers/feature_extraction_utils.py` | 655 | Feature extractor serialization infrastructure | Workflow: Pipeline_Inference | [→](./_files/src_transformers_feature_extraction_utils_py.md) |
| ✅ | `src/transformers/file_utils.py` | 107 | Backward compatibility import shim | Impl: file_utils | [→](./_files/src_transformers_file_utils_py.md) |
| ✅ | `src/transformers/hf_argparser.py` | 430 | Dataclass-driven CLI argument parser | Workflow: Training | [→](./_files/src_transformers_hf_argparser_py.md) |
| ✅ | `src/transformers/hyperparameter_search.py` | 124 | Multi-backend hyperparameter optimization | Workflow: Training | [→](./_files/src_transformers_hyperparameter_search_py.md) |
| ✅ | `src/transformers/image_processing_base.py` | 486 | Image processor serialization infrastructure | Workflow: Pipeline_Inference | [→](./_files/src_transformers_image_processing_base_py.md) |
| ✅ | `src/transformers/image_processing_utils.py` | 320 | NumPy-based image processor base | Workflow: Pipeline_Inference | [→](./_files/src_transformers_image_processing_utils_py.md) |
| ✅ | `src/transformers/image_processing_utils_fast.py` | 953 | PyTorch-accelerated image processor | Workflow: Pipeline_Inference | [→](./_files/src_transformers_image_processing_utils_fast_py.md) |
| ✅ | `src/transformers/image_transforms.py` | 1001 | Image transformation primitives library | Workflow: Pipeline_Inference | [→](./_files/src_transformers_image_transforms_py.md) |
| ✅ | `src/transformers/image_utils.py` | 959 | Image processing and format conversion | Workflow: Pipeline_Inference | [→](./_files/src_transformers_image_utils_py.md) |
| ✅ | `src/transformers/initialization.py` | 208 | Guarded weight initialization system | Workflow: Model_Loading | [→](./_files/src_transformers_initialization_py.md) |
| ✅ | `src/transformers/masking_utils.py` | 1381 | Unified attention masking framework | Workflow: Model_Loading | [→](./_files/src_transformers_masking_utils_py.md) |
| ✅ | `src/transformers/model_debugging_utils.py` | 456 | Forward pass tracing debugger | Workflow: Training | [→](./_files/src_transformers_model_debugging_utils_py.md) |
| ✅ | `src/transformers/modelcard.py` | 767 | Model documentation and metadata | Workflow: Training | [→](./_files/src_transformers_modelcard_py.md) |
| ✅ | `src/transformers/modeling_attn_mask_utils.py` | 485 | Legacy attention mask utilities | Workflow: Model_Loading | [→](./_files/src_transformers_modeling_attn_mask_utils_py.md) |
| ✅ | `src/transformers/modeling_flash_attention_utils.py` | 706 | Flash Attention backend integration | Workflow: Model_Loading | [→](./_files/src_transformers_modeling_flash_attention_utils_py.md) |
| ✅ | `src/transformers/modeling_gguf_pytorch_utils.py` | 587 | GGUF quantized model loader | Workflow: Model_Loading, Quantization | [→](./_files/src_transformers_modeling_gguf_pytorch_utils_py.md) |
| ✅ | `src/transformers/modeling_layers.py` | 289 | Reusable model components and heads | Workflow: Model_Loading | [→](./_files/src_transformers_modeling_layers_py.md) |
| ✅ | `src/transformers/modeling_outputs.py` | 1717 | Standardized model output dataclasses | Workflow: Model_Loading, Pipeline_Inference | [→](./_files/src_transformers_modeling_outputs_py.md) |
| ✅ | `src/transformers/modeling_rope_utils.py` | 939 | Rotary position embedding variants | Workflow: Model_Loading | [→](./_files/src_transformers_modeling_rope_utils_py.md) |
| ✅ | `src/transformers/modeling_utils.py` | 4671 | Core model loading infrastructure | Workflow: Model_Loading | [→](./_files/src_transformers_modeling_utils_py.md) |
| ✅ | `src/transformers/optimization.py` | 972 | Learning rate scheduling strategies | Workflow: Training | [→](./_files/src_transformers_optimization_py.md) |
| ✅ | `src/transformers/pipelines/__init__.py` | 1086 | Pipeline factory and task registry | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines___init___py.md) |
| ✅ | `src/transformers/pipelines/any_to_any.py` | 505 | Multimodal generation pipeline | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_any_to_any_py.md) |
| ✅ | `src/transformers/pipelines/audio_classification.py` | 259 | Audio category classification pipeline | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_audio_classification_py.md) |
| ✅ | `src/transformers/pipelines/audio_utils.py` | 296 | Audio processing FFmpeg utilities | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_audio_utils_py.md) |
| ✅ | `src/transformers/pipelines/automatic_speech_recognition.py` | 684 | Speech-to-text transcription pipeline | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_automatic_speech_recognition_py.md) |
| ✅ | `src/transformers/pipelines/base.py` | 1394 | Pipeline base class infrastructure | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_base_py.md) |
| ✅ | `src/transformers/pipelines/depth_estimation.py` | 145 | Image depth prediction pipeline | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_depth_estimation_py.md) |
| ✅ | `src/transformers/pipelines/document_question_answering.py` | 546 | Document QA with OCR | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_document_question_answering_py.md) |
| ✅ | `src/transformers/pipelines/feature_extraction.py` | 88 | Text embedding extraction pipeline | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_feature_extraction_py.md) |
| ✅ | `src/transformers/pipelines/fill_mask.py` | 259 | Masked token prediction pipeline | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_fill_mask_py.md) |
| ✅ | `src/transformers/pipelines/image_classification.py` | 229 | Image category classification pipeline | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_image_classification_py.md) |
| ✅ | `src/transformers/pipelines/image_feature_extraction.py` | 115 | Visual embedding extraction pipeline | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_image_feature_extraction_py.md) |
| ✅ | `src/transformers/pipelines/image_segmentation.py` | 223 | Image segmentation with masks | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_image_segmentation_py.md) |
| ✅ | `src/transformers/pipelines/image_text_to_text.py` | 455 | Multimodal text generation | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_image_text_to_text_py.md) |
| ✅ | `src/transformers/pipelines/image_to_image.py` | 145 | Image transformation/enhancement | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_image_to_image_py.md) |
| ✅ | `src/transformers/pipelines/image_to_text.py` | 229 | Image captioning generation | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_image_to_text_py.md) |
| ✅ | `src/transformers/pipelines/keypoint_matching.py` | 176 | Image correspondence matching | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_keypoint_matching_py.md) |
| ✅ | `src/transformers/pipelines/mask_generation.py` | 335 | SAM automatic segmentation | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_mask_generation_py.md) |
| ✅ | `src/transformers/pipelines/object_detection.py` | 197 | Object detection with boxes | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_object_detection_py.md) |
| ✅ | `src/transformers/pipelines/pt_utils.py` | 323 | PyTorch batching utilities | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_pt_utils_py.md) |
| ✅ | `src/transformers/pipelines/question_answering.py` | 685 | Extractive QA from text | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_question_answering_py.md) |
| ✅ | `src/transformers/pipelines/table_question_answering.py` | 382 | QA on tabular data | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_table_question_answering_py.md) |
| ✅ | `src/transformers/pipelines/text_classification.py` | 235 | Text categorization tasks | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_text_classification_py.md) |
| ✅ | `src/transformers/pipelines/text_generation.py` | 500 | Autoregressive text generation | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_text_generation_py.md) |
| ✅ | `src/transformers/pipelines/text_to_audio.py` | 311 | Text-to-speech and audio generation | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_text_to_audio_py.md) |
| ✅ | `src/transformers/pipelines/token_classification.py` | 646 | Named entity recognition and tagging | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_token_classification_py.md) |
| ✅ | `src/transformers/pipelines/video_classification.py` | 191 | Video content classification | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_video_classification_py.md) |
| ✅ | `src/transformers/pipelines/visual_question_answering.py` | 212 | Answer questions about images | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_visual_question_answering_py.md) |
| ✅ | `src/transformers/pipelines/zero_shot_audio_classification.py` | 161 | Classify audio without training | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_zero_shot_audio_classification_py.md) |
| ✅ | `src/transformers/pipelines/zero_shot_classification.py` | 267 | Classify text without training | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_zero_shot_classification_py.md) |
| ✅ | `src/transformers/pipelines/zero_shot_image_classification.py` | 202 | Classify images without training | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_zero_shot_image_classification_py.md) |
| ✅ | `src/transformers/pipelines/zero_shot_object_detection.py` | 242 | Detect objects without training | Workflow: Pipeline_Inference | [→](./_files/src_transformers_pipelines_zero_shot_object_detection_py.md) |
| ✅ | `src/transformers/processing_utils.py` | 1922 | Unified multimodal processing interface | Workflow: Pipeline_Inference | [→](./_files/src_transformers_processing_utils_py.md) |
| ✅ | `src/transformers/pytorch_utils.py` | 284 | PyTorch compatibility and utilities | Workflow: Model_Loading, Training | [→](./_files/src_transformers_pytorch_utils_py.md) |
| ✅ | `src/transformers/quantizers/__init__.py` | 16 | Public quantization API entry | Workflow: Quantization | [→](./_files/src_transformers_quantizers___init___py.md) |
| ✅ | `src/transformers/quantizers/auto.py` | 338 | Auto-dispatch quantization factory | Workflow: Quantization, Model_Loading | [→](./_files/src_transformers_quantizers_auto_py.md) |
| ✅ | `src/transformers/quantizers/base.py` | 354 | Abstract quantizer base class | Workflow: Quantization | [→](./_files/src_transformers_quantizers_base_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_aqlm.py` | 73 | AQLM additive quantization support | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_aqlm_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_auto_round.py` | 71 | AutoRound adaptive rounding quantization | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_auto_round_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_awq.py` | 95 | AWQ 4-bit activation-aware quantization | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_awq_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_bitnet.py` | 109 | BitNet 1.58-bit extreme quantization | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_bitnet_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_bnb_8bit.py` | 172 | BitsAndBytes INT8 quantization | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_bnb_8bit_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_compressed_tensors.py` | 111 | Compressed-tensors unified compression | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_compressed_tensors_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_eetq.py` | 108 | EETQ efficient INT8 quantization | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_eetq_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_fbgemm_fp8.py` | 187 | FBGEMM FP8 H100 quantization | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_fbgemm_fp8_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_finegrained_fp8.py` | 162 | Fine-grained FP8 consumer GPU | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_finegrained_fp8_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_fp_quant.py` | 150 | FP-Quant pseudo/real quantization | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_fp_quant_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_gptq.py` | 104 | GPTQ weight-only quantization | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_gptq_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_higgs.py` | 176 | FLUTE kernel hardware-aware quantization | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_higgs_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_hqq.py` | 262 | Half-Quadratic trainable quantization | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_hqq_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_mxfp4.py` | 292 | Microscaling FP4 fbgemm quantization | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_mxfp4_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_quanto.py` | 119 | Multi-precision quanto integration | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_quanto_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_quark.py` | 115 | AMD Quark framework integration | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_quark_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_spqr.py` | 79 | Sparse quantization representation | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_spqr_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_torchao.py` | 356 | PyTorch official quantization toolkit | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_torchao_py.md) |
| ✅ | `src/transformers/quantizers/quantizer_vptq.py` | 73 | Vector post-training quantization | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizer_vptq_py.md) |
| ✅ | `src/transformers/quantizers/quantizers_utils.py` | 41 | Module navigation and conversion | Workflow: Quantization | [→](./_files/src_transformers_quantizers_quantizers_utils_py.md) |
| ✅ | `src/transformers/safetensors_conversion.py` | 110 | Automatic safetensors model conversion | Workflow: Model_Loading | [→](./_files/src_transformers_safetensors_conversion_py.md) |
| ✅ | `src/transformers/testing_utils.py` | 4366 | Comprehensive testing infrastructure | Impl: testing_utils | [→](./_files/src_transformers_testing_utils_py.md) |
| ✅ | `src/transformers/time_series_utils.py` | 225 | Probabilistic forecasting distribution outputs | Impl: time_series_utils | [→](./_files/src_transformers_time_series_utils_py.md) |
| ✅ | `src/transformers/tokenization_mistral_common.py` | 1992 | Mistral AI official tokenizer | Workflow: Tokenization | [→](./_files/src_transformers_tokenization_mistral_common_py.md) |
| ✅ | `src/transformers/tokenization_python.py` | 1400 | Pure Python tokenizer base | Workflow: Tokenization | [→](./_files/src_transformers_tokenization_python_py.md) |
| ✅ | `src/transformers/tokenization_utils_base.py` | 3639 | Core tokenizer abstraction layer | Workflow: Tokenization | [→](./_files/src_transformers_tokenization_utils_base_py.md) |
| ✅ | `src/transformers/tokenization_utils_sentencepiece.py` | 316 | SentencePiece tokenizer integration | Workflow: Tokenization | [→](./_files/src_transformers_tokenization_utils_sentencepiece_py.md) |
| ✅ | `src/transformers/tokenization_utils_tokenizers.py` | 1249 | Fast Rust tokenizer backend | Workflow: Tokenization | [→](./_files/src_transformers_tokenization_utils_tokenizers_py.md) |
| ✅ | `src/transformers/trainer.py` | 5324 | Complete model training orchestration | Workflow: Training | [→](./_files/src_transformers_trainer_py.md) |
| ✅ | `src/transformers/trainer_callback.py` | 776 | Event-driven training loop hooks | Workflow: Training | [→](./_files/src_transformers_trainer_callback_py.md) |
| ✅ | `src/transformers/trainer_jit_checkpoint.py` | 126 | Preemptible infrastructure checkpoint | Workflow: Training | [→](./_files/src_transformers_trainer_jit_checkpoint_py.md) |
| ✅ | `src/transformers/trainer_pt_utils.py` | 1242 | PyTorch training infrastructure utilities | Workflow: Training | [→](./_files/src_transformers_trainer_pt_utils_py.md) |
| ✅ | `src/transformers/trainer_seq2seq.py` | 386 | Sequence-to-sequence trainer specialization | Workflow: Training | [→](./_files/src_transformers_trainer_seq2seq_py.md) |
| ✅ | `src/transformers/trainer_utils.py` | 957 | Framework-agnostic training utilities | Workflow: Training | [→](./_files/src_transformers_trainer_utils_py.md) |
| ✅ | `src/transformers/training_args.py` | 2809 | Comprehensive training configuration | Workflow: Training | [→](./_files/src_transformers_training_args_py.md) |
| ✅ | `src/transformers/training_args_seq2seq.py` | 89 | Seq2seq training configuration extension | Workflow: Training | [→](./_files/src_transformers_training_args_seq2seq_py.md) |
| ✅ | `src/transformers/video_processing_utils.py` | 888 | Video preprocessing pipeline base | Workflow: Pipeline_Inference | [→](./_files/src_transformers_video_processing_utils_py.md) |
| ✅ | `src/transformers/video_utils.py` | 893 | Multi-backend video loading utilities | Workflow: Pipeline_Inference | [→](./_files/src_transformers_video_utils_py.md) |

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
