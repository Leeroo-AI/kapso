# File: `libs/langchain_v1/tests/unit_tests/conftest.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 127 |
| Functions | `remove_request_headers`, `remove_response_headers`, `vcr_config`, `pytest_recording_configure`, `pytest_addoption`, `pytest_collection_modifyitems` |
| Imports | collections, importlib, langchain_tests, pytest, typing, vcr |

## Understanding

**Status:** ✅ Explored

**Purpose:** Pytest configuration for unit tests in langchain_v1. Provides VCR cassette recording configuration, custom pytest options (--only-extended, --only-core), and custom marker handling for dependency requirements.

**Mechanism:** Extends base VCR config from langchain_tests with additional header filtering and custom serialization (yaml.gz). Implements pytest hooks: pytest_addoption for CLI flags, pytest_collection_modifyitems for processing @pytest.mark.requires markers, and pytest_recording_configure for VCR setup. Removes sensitive headers from HTTP requests/responses during recording.

**Significance:** Central test infrastructure configuration ensuring secure cassette recording, selective test execution based on dependency availability, and separation of core vs extended tests. Critical for test isolation and preventing accidental network calls in unit tests.
