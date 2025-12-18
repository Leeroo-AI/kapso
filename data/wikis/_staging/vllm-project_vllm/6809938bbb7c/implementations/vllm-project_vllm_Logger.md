{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Logging]], [[domain::Diagnostics]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Custom logging configuration and utilities for vLLM with support for colored output, distributed logging, and function tracing.

=== Description ===
The logger.py module is a 303-line custom logging system that extends Python's standard logging framework with vLLM-specific features. It provides centralized logging configuration, colored output support, distributed-aware logging scopes, one-time logging methods, and optional function call tracing for debugging.

Key features include: (1) Configurable logging via environment variables (VLLM_LOGGING_LEVEL, VLLM_LOGGING_STREAM, VLLM_LOGGING_COLOR, VLLM_LOGGING_CONFIG_PATH); (2) Custom formatters (NewLineFormatter, ColoredFormatter) that add file location and line numbers; (3) Extended logger methods (debug_once, info_once, warning_once) that only print a message once even if called multiple times, with support for process/global/local scopes for distributed systems; (4) Function tracing capability (enable_trace_function_call) that logs every function entry/exit for debugging hangs or crashes; (5) Automatic httpx logging suppression to reduce verbosity from Hugging Face Hub access.

The logging configuration is initialized when the module is imported, using a thread-safe singleton pattern. The module supports both programmatic configuration via DEFAULT_LOGGING_CONFIG and file-based configuration via JSON. It integrates with vLLM's distributed execution to provide scope-aware logging that can filter messages to only the first rank globally or locally.

=== Usage ===
Use init_logger() to create loggers throughout vLLM codebase. Logging is automatically configured on import.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/vllm/logger.py vllm/logger.py]
* '''Lines:''' 1-303

=== Signature ===
<syntaxhighlight lang="python">
# Logger initialization
def init_logger(name: str) -> _VllmLogger

# Extended logger methods
class _VllmLogger(Logger):
    def debug_once(
        self,
        msg: str,
        *args: Hashable,
        scope: LogScope = "process"
    ) -> None

    def info_once(
        self,
        msg: str,
        *args: Hashable,
        scope: LogScope = "process"
    ) -> None

    def warning_once(
        self,
        msg: str,
        *args: Hashable,
        scope: LogScope = "process"
    ) -> None

# Scope types
LogScope = Literal["process", "global", "local"]

# Utility functions
@contextmanager
def suppress_logging(level: int = logging.INFO) -> Generator[None, Any, None]

def enable_trace_function_call(
    log_file_path: str,
    root_dir: str | None = None
) -> None

# Configuration
DEFAULT_LOGGING_CONFIG: dict[str, Any]
_FORMAT: str
_DATE_FORMAT: str
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.logger import init_logger

# Create a logger for your module
logger = init_logger(__name__)

# Use standard logging methods
logger.info("This is an info message")
logger.warning("This is a warning")

# Use once methods
logger.info_once("This message appears only once")
</syntaxhighlight>

== I/O Contract ==

=== Key Exports ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| init_logger || Function || Initialize a vLLM logger instance
|-
| _VllmLogger || Class || Extended logger with once methods and scope support
|-
| LogScope || TypeAlias || Literal type for logging scopes
|-
| suppress_logging || ContextManager || Temporarily suppress logging output
|-
| enable_trace_function_call || Function || Enable function call tracing
|-
| DEFAULT_LOGGING_CONFIG || dict || Default logging configuration
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Example 1: Basic logger usage
from vllm.logger import init_logger

logger = init_logger(__name__)

logger.debug("Debug information")
logger.info("Informational message")
logger.warning("Warning message")
logger.error("Error message")

# Example 2: One-time logging
# Useful for messages in loops that should only appear once
for i in range(1000):
    logger.info_once("This appears only once despite 1000 iterations")
    # Regular logging still works
    logger.debug(f"Processing item {i}")

# Example 3: Distributed logging with scopes
from vllm.logger import init_logger

logger = init_logger(__name__)

# Process scope (default) - logs on every process
logger.info("This logs on all processes", scope="process")

# Global scope - logs only on global rank 0
logger.info("This logs only on global rank 0", scope="global")

# Local scope - logs only on local rank 0 of each node
logger.info("This logs only on local rank 0", scope="local")

# Example 4: Suppressing logging temporarily
from vllm.logger import init_logger, suppress_logging
import logging

logger = init_logger(__name__)

logger.info("This will be logged")

with suppress_logging(logging.WARNING):
    # INFO and DEBUG are suppressed
    logger.info("This will NOT be logged")
    logger.debug("This will NOT be logged")
    logger.warning("This will be logged")
    logger.error("This will be logged")

logger.info("This will be logged again")

# Example 5: Function tracing for debugging
from vllm.logger import enable_trace_function_call

# Enable tracing (only for current thread)
enable_trace_function_call(
    log_file_path="/tmp/vllm_trace.log",
    root_dir="/path/to/vllm"  # Only trace functions in this directory
)

# All function calls in vllm will now be logged
# Useful for debugging hangs or crashes

# Example 6: Custom logging configuration
import os
import json

# Via environment variables
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["VLLM_LOGGING_COLOR"] = "1"  # Force colored output

# Via JSON config file
config = {
    "version": 1,
    "formatters": {
        "custom": {
            "format": "%(asctime)s - %(name)s - %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": "vllm.log",
            "formatter": "custom"
        }
    },
    "loggers": {
        "vllm": {
            "handlers": ["file"],
            "level": "INFO"
        }
    }
}

with open("/tmp/logging_config.json", "w") as f:
    json.dump(config, f)

os.environ["VLLM_LOGGING_CONFIG_PATH"] = "/tmp/logging_config.json"

# Now import vLLM to apply configuration
import vllm

# Example 7: Checking log configuration
import vllm.envs as envs

print(f"Log level: {envs.VLLM_LOGGING_LEVEL}")
print(f"Log stream: {envs.VLLM_LOGGING_STREAM}")
print(f"Log color: {envs.VLLM_LOGGING_COLOR}")
print(f"Log prefix: {envs.VLLM_LOGGING_PREFIX}")
</syntaxhighlight>

== Related Pages ==
* [[uses::Module:vllm-project_vllm_Environment_Variables]]
* [[implements::Pattern:Singleton_Logger]]
* [[related::Module:vllm-project_vllm_Distributed_Parallel_State]]
