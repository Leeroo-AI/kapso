# File: `libs/langchain_v1/tests/integration_tests/conftest.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 34 |
| Functions | `test_dir`, `vcr_cassette_dir` |
| Imports | pathlib, pytest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configures pytest fixtures and environment setup for integration tests in langchain_v1.

**Mechanism:** Automatically loads environment variables from tests/integration_tests/.env at import time using dotenv. Provides two fixtures: test_dir (returns path to integration_tests directory) and vcr_cassette_dir (generates cassette directory path for VCR.py HTTP recording based on test module location).

**Significance:** Central configuration that enables integration tests to access API credentials from .env files and supports HTTP recording/playback via VCR cassettes for deterministic integration testing without hitting live APIs repeatedly.
