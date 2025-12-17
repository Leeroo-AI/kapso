# File: `libs/langchain_v1/tests/integration_tests/conftest.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 34 |
| Functions | `test_dir`, `vcr_cassette_dir` |
| Imports | pathlib, pytest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Pytest configuration for integration tests, providing shared fixtures and environment setup.

**Mechanism:** Loads environment variables from .env file at test startup. Provides test_dir fixture returning path to integration_tests directory and vcr_cassette_dir fixture for VCR (record/replay) cassette organization by test module.

**Significance:** Centralizes test configuration for integration tests. VCR cassettes enable recording and replaying API interactions for faster, more reliable tests. Environment variable loading ensures API keys are available without committing them to source control.
