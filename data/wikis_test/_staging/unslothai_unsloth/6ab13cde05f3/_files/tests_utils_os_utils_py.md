# File: `tests/utils/os_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 128 |
| Functions | `detect_package_manager`, `check_package_installed`, `require_package`, `require_python_package` |
| Imports | importlib, os, shutil, subprocess, sys |

## Understanding

**Status:** ✅ Documented

**Purpose:** System package manager detection and dependency checking

**Mechanism:** Detects apt/yum/dnf/pacman package managers, checks package installation

**Significance:** Ensures test dependencies are available across different Linux distributions
