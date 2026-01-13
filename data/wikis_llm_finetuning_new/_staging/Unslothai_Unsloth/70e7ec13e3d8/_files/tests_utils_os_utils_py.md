# File: `tests/utils/os_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 128 |
| Functions | `detect_package_manager`, `check_package_installed`, `require_package`, `require_python_package` |
| Imports | importlib, os, shutil, subprocess, sys |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides cross-platform package detection and dependency requirement utilities for system and Python packages.

**Mechanism:** `detect_package_manager()` checks for apt/yum/dnf/pacman/zypper binaries in /usr/bin to identify the Linux distribution's package manager. `check_package_installed()` queries installation status via dpkg/rpm/pacman/zypper depending on detected manager. `require_package()` first checks PATH for executables via `shutil.which()`, then falls back to package manager queries, exiting with helpful installation instructions (including conda alternatives) if not found. `require_python_package()` uses `importlib.util.find_spec()` to verify Python package availability, providing pip/conda install guidance on failure.

**Significance:** Test prerequisite utility ensuring external dependencies (like ffmpeg) and Python packages are available before running tests, improving test environment setup and error messaging for developers.
