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

**Purpose:** Builds and installs NIXL (networking library) from source with UCX dependencies for vLLM's distributed communication.

**Mechanism:** Three-stage process: (1) builds UCX v1.19.x from source with specific configuration flags (enable-cma, enable-mt, with-verbs), (2) builds NIXL Python wheel linking against custom UCX, (3) uses auditwheel to repair wheel by bundling UCX libraries. Implements caching to skip rebuild if wheel already exists. Checks for required system packages (patchelf, build tools) and attempts apt-get install if run as root.

**Significance:** Enables vLLM to use optimized RDMA/InfiniBand communication for multi-node deployments. The self-contained wheel approach (via auditwheel) ensures UCX dependencies are bundled, avoiding system library conflicts. Essential for high-performance distributed inference in cluster environments.
