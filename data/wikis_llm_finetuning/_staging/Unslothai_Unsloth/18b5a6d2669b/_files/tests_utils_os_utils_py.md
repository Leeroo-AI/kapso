# File: `tests/utils/os_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 128 |
| Functions | `detect_package_manager`, `check_package_installed`, `require_package`, `require_python_package` |
| Imports | importlib, os, shutil, subprocess, sys |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides system-level package management utilities for detecting and validating OS-level and Python package dependencies across different Linux distributions.

**Mechanism:** Detects available package managers (apt, yum, dnf, pacman, zypper) by checking for their executables, then provides functions to check if packages are installed and enforce package requirements with helpful installation instructions. Also includes Python package validation using importlib to ensure required dependencies are present.

**Significance:** Essential test utility that ensures the testing environment has all required system dependencies (like ffmpeg) and Python packages before running tests, providing clear installation guidance across different platforms and package managers.
