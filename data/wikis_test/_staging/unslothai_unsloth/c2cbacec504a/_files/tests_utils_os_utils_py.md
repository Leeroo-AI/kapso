# File: `tests/utils/os_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 128 |
| Functions | `detect_package_manager`, `check_package_installed`, `require_package`, `require_python_package` |
| Imports | importlib, os, shutil, subprocess, sys |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides cross-platform system package management utilities for tests, detecting package managers and installing required dependencies dynamically during test execution.

**Mechanism:** Detects available package managers (apt, yum, brew, etc.) using shutil.which, checks if system packages are installed, automatically installs missing packages with appropriate manager, and validates Python packages can be imported after installation.

**Significance:** Enables tests to be self-sufficient by automatically installing system dependencies (like audio codecs for speech tests), improving test portability across different Linux distributions and macOS environments without manual setup.
