# Phase 6b: Orphan Review Report

## Summary
- MANUAL_REVIEW files evaluated: 49
- Approved: 7
- Rejected: 42

## Decisions

| File | Decision | Reasoning |
|------|----------|-----------|
| `.circleci/parse_test_outputs.py` | REJECTED | Internal CI script, no public API |
| `benchmark_v2/run_benchmarks.py` | REJECTED | Internal CI script, CLI entry point only |
| `conftest.py` | REJECTED | Pytest config, internal testing infrastructure |
| `examples/run_on_remote.py` | REJECTED | Example script, no public API |
| `src/transformers/dependency_versions_check.py` | APPROVED | Public API `dep_version_check`, user-facing |
| `src/transformers/dependency_versions_table.py` | APPROVED | Public `deps` dict used across codebase |
| `src/transformers/file_utils.py` | REJECTED | Deprecated shim, re-exports only |
| `src/transformers/initialization.py` | APPROVED | Public init funcs, used in model loading |
| `src/transformers/modeling_layers.py` | APPROVED | Public classes for sequence/token classification |
| `src/transformers/pytorch_utils.py` | APPROVED | Public Conv1D, pruning utils, user-facing |
| `src/transformers/time_series_utils.py` | APPROVED | Public distribution classes for time series |
| `utils/check_bad_commit.py` | REJECTED | Internal CI script, no public API |
| `utils/check_config_docstrings.py` | REJECTED | Internal repo validation script |
| `utils/check_doc_toc.py` | REJECTED | Internal repo validation script |
| `utils/check_doctest_list.py` | REJECTED | Internal repo validation script |
| `utils/check_dummies.py` | REJECTED | Internal repo maintenance script |
| `utils/check_model_tester.py` | REJECTED | Internal CI validation, small script |
| `utils/check_modeling_structure.py` | REJECTED | Internal repo validation script |
| `utils/check_modular_conversion.py` | REJECTED | Internal CI validation script |
| `utils/check_pipeline_typing.py` | REJECTED | Internal repo validation script |
| `utils/check_self_hosted_runner.py` | REJECTED | Internal CI script, small |
| `utils/collated_reports.py` | REJECTED | Internal CI reporting script |
| `utils/compare_test_runs.py` | REJECTED | Internal CI utility, no public API |
| `utils/create_dependency_mapping.py` | REJECTED | Internal modular converter helper |
| `utils/download_glue_data.py` | REJECTED | Data download script, no public API |
| `utils/extract_pr_number_from_circleci.py` | REJECTED | Internal CI script, trivial |
| `utils/extract_warnings.py` | REJECTED | Internal CI log processing script |
| `utils/fetch_hub_objects_for_ci.py` | REJECTED | Internal CI preparation script |
| `utils/get_github_job_time.py` | REJECTED | Internal CI utility, no public API |
| `utils/get_modified_files.py` | REJECTED | Internal CI utility, trivial script |
| `utils/get_pr_run_slow_jobs.py` | REJECTED | Internal CI utility script |
| `utils/get_previous_daily_ci.py` | REJECTED | Internal CI utility script |
| `utils/get_test_info.py` | REJECTED | Internal test introspection utility |
| `utils/get_test_reports.py` | REJECTED | Internal CI test runner script |
| `utils/important_files.py` | REJECTED | Trivial constant list, internal use |
| `utils/modular_integrations.py` | REJECTED | Internal modular converter helper |
| `utils/patch_helper.py` | REJECTED | Internal release utility script |
| `utils/pr_slow_ci_models.py` | REJECTED | Internal CI utility script |
| `utils/print_env.py` | REJECTED | Internal debug utility, no public API |
| `utils/process_bad_commit_report.py` | REJECTED | Internal CI reporting script |
| `utils/process_circleci_workflow_test_reports.py` | REJECTED | Internal CI reporting script |
| `utils/process_test_artifacts.py` | REJECTED | Internal CI utility, small script |
| `utils/release.py` | REJECTED | Internal release automation script |
| `utils/scan_skipped_tests.py` | REJECTED | Internal test scanning script |
| `utils/set_cuda_devices_for_ci.py` | REJECTED | Internal CI script, trivial |
| `utils/sort_auto_mappings.py` | REJECTED | Internal repo style script |
| `utils/split_doctest_jobs.py` | REJECTED | Internal CI job splitting script |
| `utils/split_model_tests.py` | REJECTED | Internal CI job splitting script |
| `utils/update_tiny_models.py` | REJECTED | Internal CI maintenance script |

## Notes

### Patterns Observed
- **utils/ directory**: Dominated by CI/CD infrastructure scripts (38 files rejected). These are internal tooling for CircleCI, GitHub Actions, test management, and release automation. None expose public APIs.
- **src/transformers/ directory**: Contains the only approved files. These are genuine library utilities with public APIs that users would import.

### Approved Files (7 total)
All approved files are in `src/transformers/` and share common traits:
1. **Public API exposure**: Classes and functions without underscore prefixes
2. **User-facing functionality**: Would be imported by library users
3. **Distinct algorithms/utilities**: Implement meaningful functionality (distributions, layers, initialization)

Approved files:
- `dependency_versions_check.py` - Version validation utilities
- `dependency_versions_table.py` - Centralized version requirements
- `initialization.py` - Tensor initialization functions for model loading
- `modeling_layers.py` - Reusable base layers (sequence/token classification heads)
- `pytorch_utils.py` - PyTorch utilities including Conv1D layer
- `time_series_utils.py` - Probability distribution classes for time series models

### Borderline Files
- `src/transformers/file_utils.py` (REJECTED): Has public imports but is a deprecated backward compatibility shim that only re-exports from `utils`. Not valuable for documentation.
- `utils/download_glue_data.py` (REJECTED): Could be user-facing for data download, but lacks any public API - it's a standalone script with `if __name__ == "__main__"` only.

### Recommendation
The 7 approved files should have wiki pages created. The 42 rejected files are internal infrastructure and would not benefit end users.
