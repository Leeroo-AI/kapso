# File: `libs/langchain_v1/tests/unit_tests/conftest.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 127 |
| Functions | `remove_request_headers`, `remove_response_headers`, `vcr_config`, `pytest_recording_configure`, `pytest_addoption`, `pytest_collection_modifyitems` |
| Imports | collections, importlib, langchain_tests, pytest, typing, vcr |

## Understanding

**Status:** ✅ Explored

**Purpose:** Pytest configuration file providing test infrastructure including VCR cassette recording, custom test markers (`requires`), and test filtering options (`--only-extended`, `--only-core`).

**Mechanism:** Configures VCR (Video Cassette Recorder) to record/replay HTTP interactions with sensitive header redaction. Implements custom `requires` marker to skip tests when specified packages are not installed by checking with `importlib.util.find_spec`. Adds command-line options to filter between core and extended tests. Registers custom YAML.gz serializer and persister for cassette files.

**Significance:** Foundational test infrastructure that enables: (1) safe HTTP recording without leaking credentials, (2) selective test execution based on installed dependencies, (3) separation between quick core tests and slower extended tests, and (4) reproducible test results through cassette replay. Essential for maintainability in a monorepo with optional dependencies.
