# File: `tools/install_nixl_from_source_ubuntu.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 254 |
| Functions | `get_latest_nixl_version`, `run_command`, `is_pip_package_installed`, `find_nixl_wheel_in_cache`, `install_system_dependencies`, `build_and_install_prerequisites` |
| Imports | argparse, glob, json, os, subprocess, sys, urllib |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** NIXL dependency installer with UCX compilation

**Mechanism:** Complex installation pipeline: (1) checks for cached NIXL wheels to skip rebuilding, (2) fetches latest NIXL version from GitHub releases API, (3) installs system dependencies (build tools, meson, libtool) via apt if root, (4) clones and builds UCX v1.19.x from source with specific flags (verbs, CMA, multi-threading), (5) builds NIXL wheel with UCX dependencies using PKG_CONFIG_PATH and RPATH, (6) uses auditwheel to create self-contained wheels bundling UCX libraries, (7) caches wheels for future use. Handles S3 credentials and wheel management.

**Significance:** Automates installation of NIXL (Network Interface for X-Large models), a high-performance networking library for distributed inference. Critical for enabling efficient multi-node tensor parallelism in vLLM. The self-contained wheel approach with bundled UCX ensures deployment consistency across systems without requiring UCX preinstallation.
