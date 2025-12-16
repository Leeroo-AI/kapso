# File: `tests/utils/os_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 128 |
| Functions | `detect_package_manager`, `check_package_installed`, `require_package`, `require_python_package` |
| Imports | importlib, os, shutil, subprocess, sys |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides cross-platform utilities for detecting system package managers and verifying that required system and Python packages are installed. Helps ensure test environments have necessary dependencies before running tests.

**Mechanism:** The file implements several layers of dependency checking:
- `detect_package_manager()` identifies the available package manager (apt, yum, dnf, pacman, zypper) by checking for executable paths
- `check_package_installed()` verifies if a system package is installed using the appropriate query command for each package manager (dpkg, rpm, pacman, zypper)
- `require_package()` enforces that a system package is installed, first checking if the executable is in PATH, then using the package manager; exits with helpful installation instructions if missing
- `require_python_package()` checks if a Python package is importable using `importlib.util.find_spec()` and exits with pip/conda installation instructions if missing

All requirement functions provide user-friendly error messages with installation commands for multiple package managers.

**Significance:** Essential for test environment validation, particularly for tests that depend on external system tools (like ffmpeg for video processing) or optional Python packages. This prevents cryptic failures by catching missing dependencies early with clear installation guidance.
