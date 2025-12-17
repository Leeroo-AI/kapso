# Phase 6b: Orphan Review Report

## Summary
- MANUAL_REVIEW files evaluated: 46
- Approved: 5
- Rejected: 41

## Decisions

| File | Decision | Reasoning |
|------|----------|-----------|
| `.circleci/parse_test_outputs.py` | REJECTED | Internal CI script, no public API |
| `benchmark_v2/run_benchmarks.py` | REJECTED | Internal benchmark runner, main-only script |
| `conftest.py` | REJECTED | Internal pytest config, no public API |
| `examples/run_on_remote.py` | REJECTED | Script wrapper, not user-facing library code |
| `src/transformers/dependency_versions_check.py` | APPROVED | Public API (dep_version_check), user-facing |
| `src/transformers/dependency_versions_table.py` | APPROVED | Public deps dict, user-facing configuration |
| `src/transformers/file_utils.py` | APPROVED | Public API re-exports, user-facing compat layer |
| `src/transformers/time_series_utils.py` | APPROVED | Public classes for time series, distinct algo |
| `utils/check_bad_commit.py` | REJECTED | Internal CI tool, no public API |
| `utils/check_config_docstrings.py` | REJECTED | Internal validation script, no public API |
| `utils/check_doc_toc.py` | REJECTED | Internal doc maintenance, no public API |
| `utils/check_doctest_list.py` | REJECTED | Internal test config maintenance |
| `utils/check_dummies.py` | REJECTED | Internal code generation script |
| `utils/check_model_tester.py` | REJECTED | Internal test validation, no public API |
| `utils/check_modeling_structure.py` | REJECTED | Internal lint/check script, no public API |
| `utils/check_modular_conversion.py` | REJECTED | Internal CI validation tool |
| `utils/check_pipeline_typing.py` | REJECTED | Internal code generation, no public API |
| `utils/check_self_hosted_runner.py` | REJECTED | Internal CI monitoring, no public API |
| `utils/collated_reports.py` | REJECTED | Internal CI reporting tool |
| `utils/compare_test_runs.py` | REJECTED | Internal CI comparison utility |
| `utils/create_dependency_mapping.py` | REJECTED | Internal dependency analysis tool |
| `utils/download_glue_data.py` | APPROVED | User-facing utility for GLUE dataset download |
| `utils/extract_pr_number_from_circleci.py` | REJECTED | Internal CI helper, trivial script |
| `utils/extract_warnings.py` | REJECTED | Internal CI analysis tool |
| `utils/fetch_hub_objects_for_ci.py` | REJECTED | Internal CI preparation script |
| `utils/get_github_job_time.py` | REJECTED | Internal CI metrics tool |
| `utils/get_modified_files.py` | REJECTED | Internal git utility, trivial script |
| `utils/get_pr_run_slow_jobs.py` | REJECTED | Internal CI job selection tool |
| `utils/get_previous_daily_ci.py` | REJECTED | Internal CI artifact retrieval |
| `utils/get_test_info.py` | REJECTED | Internal test introspection utility |
| `utils/get_test_reports.py` | REJECTED | Internal test runner script |
| `utils/important_files.py` | REJECTED | Internal config, just a list constant |
| `utils/modular_integrations.py` | REJECTED | Internal AST transform utilities |
| `utils/patch_helper.py` | REJECTED | Internal release automation tool |
| `utils/pr_slow_ci_models.py` | REJECTED | Internal CI model detection |
| `utils/print_env.py` | REJECTED | Internal diagnostic script |
| `utils/process_bad_commit_report.py` | REJECTED | Internal CI reporting tool |
| `utils/process_circleci_workflow_test_reports.py` | REJECTED | Internal CI report aggregation |
| `utils/process_test_artifacts.py` | REJECTED | Internal CI configuration tool |
| `utils/release.py` | REJECTED | Internal release automation, no public API |
| `utils/scan_skipped_tests.py` | REJECTED | Internal test coverage analysis |
| `utils/set_cuda_devices_for_ci.py` | REJECTED | Internal CI config, trivial script |
| `utils/sort_auto_mappings.py` | REJECTED | Internal code formatting utility |
| `utils/split_doctest_jobs.py` | REJECTED | Internal CI job splitting tool |
| `utils/split_model_tests.py` | REJECTED | Internal CI test splitting utility |
| `utils/update_tiny_models.py` | REJECTED | Internal CI model maintenance |

## Notes

### Patterns Observed
- **CI/Build infrastructure dominates utils/**: The vast majority (41 out of 46) of MANUAL_REVIEW files are internal CI/CD scripts, test utilities, and build infrastructure. These have no public API and are not user-facing.
- **src/transformers/ files are user-facing**: All 4 files in `src/transformers/` were approved because they contain public APIs that users may import or interact with.
- **One user-facing utility in utils/**: `download_glue_data.py` is the only utils file approved because it provides a user-facing command-line utility for downloading GLUE benchmark datasets.

### Approved Files Summary
1. **`dependency_versions_check.py`**: Provides `dep_version_check()` function for runtime dependency validation
2. **`dependency_versions_table.py`**: Exports `deps` dictionary with package version requirements
3. **`file_utils.py`**: Backward compatibility layer re-exporting symbols from utils module
4. **`time_series_utils.py`**: Contains `NormalOutput`, `StudentTOutput`, `NegativeBinomialOutput` classes for probabilistic forecasting
5. **`download_glue_data.py`**: User-facing CLI script for downloading GLUE benchmark data

### Borderline Cases
- **`utils/release.py`**: Contains version management functions but is intended for internal use by maintainers only
- **`utils/get_test_info.py`**: Has public functions but serves internal test introspection needs only
- **`conftest.py`**: While pytest uses this publicly, it's internal configuration not intended for user import
