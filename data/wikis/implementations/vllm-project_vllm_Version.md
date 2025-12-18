{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Version Management]], [[domain::Compatibility]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
The version module provides version information and utilities for checking version compatibility in vLLM.

=== Description ===
This module manages vLLM version information and provides utilities for version checking. Key features include:

* '''Version variables:''' Exposes __version__ and __version_tuple__ for runtime checks
* '''Graceful fallback:''' Sets version to "dev" if _version module unavailable
* '''Previous version checking:''' Helper to check if a version matches previous minor release
* '''Metrics compatibility:''' Used by --show-hidden-metrics-for-version flag
* '''Build metadata:''' Imports from auto-generated _version.py during build

The module is designed to work in both installed packages (where _version.py exists) and development trees (where it may not). It provides a simple API for version comparisons needed for backwards compatibility features.

=== Usage ===
Use this module when you need to:
* Check the current vLLM version at runtime
* Implement version-specific behavior or compatibility shims
* Display version information to users
* Validate compatibility with other components
* Enable hidden metrics for previous versions during migration

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/vllm/version.py vllm/version.py]

=== Signature ===
<syntaxhighlight lang="python">
# Version information (imported or defaulted)
__version__: str
__version_tuple__: tuple[int, int, str]

# Check if version string matches previous minor version
def _prev_minor_version_was(version_str: str) -> bool

# Get previous minor version string (for testing)
def _prev_minor_version() -> str
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.version import (
    __version__,
    __version_tuple__,
    _prev_minor_version_was,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| version_str || str || Version string to check (e.g., "0.6")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| __version__ || str || Version string (e.g., "0.7.4" or "dev")
|-
| __version_tuple__ || tuple[int, int, str] || Version tuple (e.g., (0, 7, "4"))
|-
| is_prev_version || bool || Whether version_str matches previous minor version
|}

== Usage Examples ==

=== Get Current Version ===
<syntaxhighlight lang="python">
from vllm.version import __version__, __version_tuple__

print(f"vLLM version: {__version__}")
# Output: "vLLM version: 0.7.4"

print(f"Version tuple: {__version_tuple__}")
# Output: "Version tuple: (0, 7, '4')"

# Check if development version
if __version__ == "dev":
    print("Running in development mode")
</syntaxhighlight>

=== Check Version Compatibility ===
<syntaxhighlight lang="python">
from vllm.version import __version_tuple__

# Require minimum version
MIN_VERSION = (0, 6, 0)

current = __version_tuple__[:3]
if current < MIN_VERSION:
    raise RuntimeError(
        f"vLLM {current} is too old. "
        f"Minimum required version: {MIN_VERSION}"
    )
</syntaxhighlight>

=== Check Previous Minor Version ===
<syntaxhighlight lang="python">
from vllm.version import _prev_minor_version_was

# Show hidden metrics for migration from 0.6 to 0.7
if _prev_minor_version_was("0.6"):
    print("Enabling 0.6 compatibility metrics")
    enable_legacy_metrics()

# This is used by --show-hidden-metrics-for-version CLI flag
# Example: vllm serve --show-hidden-metrics-for-version 0.6
</syntaxhighlight>

=== Display Version in CLI ===
<syntaxhighlight lang="python">
import argparse
from vllm.version import __version__

parser = argparse.ArgumentParser(description="vLLM CLI Tool")
parser.add_argument(
    "--version",
    action="version",
    version=f"vLLM {__version__}"
)

args = parser.parse_args()
</syntaxhighlight>

=== Version-Specific Feature Flags ===
<syntaxhighlight lang="python">
from vllm.version import __version_tuple__

def get_default_config():
    """Get configuration with version-specific defaults."""
    config = {}

    # Feature introduced in 0.7.0
    if __version_tuple__[:2] >= (0, 7):
        config["enable_chunked_prefill"] = True
    else:
        config["enable_chunked_prefill"] = False

    # Feature changed in 0.8.0
    if __version_tuple__[:2] >= (0, 8):
        config["default_dtype"] = "bfloat16"
    else:
        config["default_dtype"] = "float16"

    return config
</syntaxhighlight>

=== Development vs Release Detection ===
<syntaxhighlight lang="python">
from vllm.version import __version__, __version_tuple__

def is_dev_version() -> bool:
    """Check if running development version."""
    return __version__ == "dev" or __version_tuple__[:2] == (0, 0)

if is_dev_version():
    print("Running development build - enabling debug features")
    enable_debug_logging()
    disable_telemetry()
else:
    print(f"Running release version {__version__}")
</syntaxhighlight>

=== Logging and Diagnostics ===
<syntaxhighlight lang="python">
import logging
from vllm.version import __version__, __version_tuple__

logger = logging.getLogger(__name__)

def log_startup_info():
    """Log version information at startup."""
    logger.info(f"Starting vLLM {__version__}")
    logger.debug(f"Version tuple: {__version_tuple__}")

    if __version__ == "dev":
        logger.warning("Running development version - not for production use")

log_startup_info()
</syntaxhighlight>

=== API Response Headers ===
<syntaxhighlight lang="python">
from vllm.version import __version__
from fastapi import FastAPI
from fastapi.responses import Response

app = FastAPI()

@app.middleware("http")
async def add_version_header(request, call_next):
    response = await call_next(request)
    response.headers["X-vLLM-Version"] = __version__
    return response

@app.get("/version")
async def get_version():
    return {"version": __version__}
</syntaxhighlight>

=== Conditional Imports Based on Version ===
<syntaxhighlight lang="python">
from vllm.version import __version_tuple__

# Import version-specific modules
if __version_tuple__[:2] >= (0, 7):
    from vllm.new_feature import AdvancedScheduler as Scheduler
else:
    from vllm.legacy import LegacyScheduler as Scheduler

scheduler = Scheduler()
</syntaxhighlight>

=== Test Version Checking Logic ===
<syntaxhighlight lang="python">
from vllm.version import _prev_minor_version, _prev_minor_version_was

# Get previous version string
prev = _prev_minor_version()
print(f"Previous minor version: {prev}")
# If current is 0.7.x, outputs: "Previous minor version: 0.6"

# Test version matching
assert _prev_minor_version_was(prev)
assert not _prev_minor_version_was("0.1")
</syntaxhighlight>

=== Build Information Display ===
<syntaxhighlight lang="python">
import sys
from vllm.version import __version__, __version_tuple__

def print_build_info():
    """Print comprehensive build information."""
    print("=" * 60)
    print("vLLM Build Information")
    print("=" * 60)
    print(f"Version:        {__version__}")
    print(f"Version Tuple:  {__version_tuple__}")
    print(f"Python:         {sys.version.split()[0]}")
    print(f"Is Dev Build:   {__version__ == 'dev'}")
    print("=" * 60)

if __name__ == "__main__":
    print_build_info()
</syntaxhighlight>

== Version String Format ==

=== Release Versions ===
* Format: "MAJOR.MINOR.PATCH"
* Example: "0.7.4"
* Tuple: (0, 7, "4")

=== Development Versions ===
* Format: "dev"
* Tuple: (0, 0, "dev")
* Used when _version.py unavailable

=== Pre-release Versions ===
* May include suffixes: "0.7.0rc1", "0.7.0.dev0"
* Tuple third element contains full suffix

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[Configuration Management]]
* [[CLI Arguments]]
* [[Backwards Compatibility]]
* [[Metrics and Monitoring]]
