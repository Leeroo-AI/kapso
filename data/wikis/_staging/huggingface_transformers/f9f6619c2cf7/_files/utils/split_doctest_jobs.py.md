# split_doctest_jobs.py

**Path:** `/tmp/praxium_repo_d5p6fp4d/utils/split_doctest_jobs.py`

## Understanding

### Purpose
Distributes doctest files across CI jobs.

### Mechanism
This script organizes doctest files into groups for parallel execution in GitHub Actions workflows. It retrieves all doctest files using `tests_fetcher.get_all_doctest_files()` and groups them by directory path. Files in `docs/source/en/model_doc` and `docs/source/en/tasks` are kept as individual entries (not grouped with other files in the same directory) to enable independent CI job execution. The script can output either a dictionary mapping directory paths to file lists or a nested list of path groups split into N slices (using `--num_splits`), which allows GitHub Actions to bypass the 256 job matrix limit.

### Significance
Enables efficient parallel testing of documentation examples by distributing doctest workload across multiple GitHub Actions jobs, reducing overall CI execution time while providing granular control over model documentation testing. The special treatment of model and task documentation files ensures they can be tested independently, which is particularly important given the large number of models in the transformers library.
