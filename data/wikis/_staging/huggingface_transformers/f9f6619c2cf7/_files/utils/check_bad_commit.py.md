# utils/check_bad_commit.py

## Understanding

### Purpose
Identifies commits introducing test failures

### Mechanism
The script uses git bisect to find the first commit between two revisions where a test starts failing. It creates a temporary Python script that installs dependencies and runs the target test with flake-finder, then automates the bisect process. The tool filters out flaky tests by verifying failures are reproducible and enriches results with GitHub PR information including commit SHA, PR number, author, and merger details.

### Significance
Critical for CI/CD workflows to automatically identify which specific code changes introduced test regressions. This dramatically speeds up debugging by pinpointing the exact commit responsible for failures, allowing developers to quickly fix or revert problematic changes.
