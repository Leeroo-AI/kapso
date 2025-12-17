# Logger - Logging Infrastructure Configuration

## Overview

**File:** `/tmp/praxium_repo_583nq7ea/vllm/logger.py` (303 lines)

**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

**Purpose:** Configures vLLM's logging system with custom formatters, distributed logging support, log deduplication, and optional function call tracing for debugging. Provides consistent logging infrastructure across all vLLM components.

## Core Architecture

### Configuration Constants

**Lines:** 21-25

```python
_FORMAT = (
    f"{envs.VLLM_LOGGING_PREFIX}%(levelname)s %(asctime)s "
    "[%(fileinfo)s:%(lineno)d] %(message)s"
)
_DATE_FORMAT = "%m-%d %H:%M:%S"
```

### Color Detection

**Lines:** 28-37

```python
def _use_color() -> bool:
    if envs.NO_COLOR or envs.VLLM_LOGGING_COLOR == "0":
        return False
    if envs.VLLM_LOGGING_COLOR == "1":
        return True
    if envs.VLLM_LOGGING_STREAM == "ext://sys.stdout":
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    elif envs.VLLM_LOGGING_STREAM == "ext://sys.stderr":
        return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    return False
```

**Environment Variables:**
- `NO_COLOR`: Disables color (standard)
- `VLLM_LOGGING_COLOR`: Explicit color control ("0" or "1")
- `VLLM_LOGGING_STREAM`: Output stream selection

### Default Logging Configuration

**Lines:** 40-71

```python
DEFAULT_LOGGING_CONFIG = {
    "formatters": {
        "vllm": {
            "class": "vllm.logging_utils.NewLineFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
        "vllm_color": {
            "class": "vllm.logging_utils.ColoredFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
    },
    "handlers": {
        "vllm": {
            "class": "logging.StreamHandler",
            "formatter": "vllm_color" if _use_color() else "vllm",
            "level": envs.VLLM_LOGGING_LEVEL,
            "stream": envs.VLLM_LOGGING_STREAM,
        },
    },
    "loggers": {
        "vllm": {
            "handlers": ["vllm"],
            "level": envs.VLLM_LOGGING_LEVEL,
            "propagate": False,
        },
    },
    "version": 1,
    "disable_existing_loggers": False,
}
```

## Log Deduplication

### Once-Only Logging Functions

**Lines:** 74-89

```python
@lru_cache
def _print_debug_once(logger: Logger, msg: str, *args: Hashable) -> None:
    # stacklevel=3 to print original caller's line info
    logger.debug(msg, *args, stacklevel=3)

@lru_cache
def _print_info_once(logger: Logger, msg: str, *args: Hashable) -> None:
    logger.info(msg, *args, stacklevel=3)

@lru_cache
def _print_warning_once(logger: Logger, msg: str, *args: Hashable) -> None:
    logger.warning(msg, *args, stacklevel=3)
```

**Mechanism:** Uses `@lru_cache` on logger+message+args tuple to ensure each unique message prints only once per process.

### Distributed Logging Scope

**Lines:** 92-106

```python
LogScope = Literal["process", "global", "local"]

def _should_log_with_scope(scope: LogScope) -> bool:
    if scope == "global":
        from vllm.distributed.parallel_state import is_global_first_rank
        return is_global_first_rank()
    if scope == "local":
        from vllm.distributed.parallel_state import is_local_first_rank
        return is_local_first_rank()
    # default "process" scope: always log
    return True
```

**Scopes:**
- **process:** Always log (default)
- **global:** Only first rank globally
- **local:** Only first rank per node

## Enhanced Logger Class

**Lines:** 109-155

```python
class _VllmLogger(Logger):
    def debug_once(self, msg: str, *args: Hashable, scope: LogScope = "process") -> None:
        if not _should_log_with_scope(scope):
            return
        _print_debug_once(self, msg, *args)

    def info_once(self, msg: str, *args: Hashable, scope: LogScope = "process") -> None:
        if not _should_log_with_scope(scope):
            return
        _print_info_once(self, msg, *args)

    def warning_once(self, msg: str, *args: Hashable, scope: LogScope = "process") -> None:
        if not _should_log_with_scope(scope):
            return
        _print_warning_once(self, msg, *args)

# Pre-defined methods mapping
_METHODS_TO_PATCH = {
    "debug_once": _VllmLogger.debug_once,
    "info_once": _VllmLogger.info_once,
    "warning_once": _VllmLogger.warning_once,
}
```

**Note:** This is a type hint class; actual patching happens in `init_logger()`.

## Logger Configuration

### Root Logger Setup

**Lines:** 158-204

```python
def _configure_vllm_root_logger() -> None:
    logging_config = dict[str, Any]()

    if not envs.VLLM_CONFIGURE_LOGGING and envs.VLLM_LOGGING_CONFIG_PATH:
        raise RuntimeError(
            "VLLM_LOGGING_CONFIG_PATH implies VLLM_CONFIGURE_LOGGING. "
            "Please enable VLLM_CONFIGURE_LOGGING or unset VLLM_LOGGING_CONFIG_PATH."
        )

    if envs.VLLM_CONFIGURE_LOGGING:
        logging_config = DEFAULT_LOGGING_CONFIG
        # Refresh values in case env vars changed
        vllm_handler = logging_config["handlers"]["vllm"]
        vllm_handler["level"] = envs.VLLM_LOGGING_LEVEL
        vllm_handler["stream"] = envs.VLLM_LOGGING_STREAM
        vllm_handler["formatter"] = "vllm_color" if _use_color() else "vllm"

        vllm_loggers = logging_config["loggers"]["vllm"]
        vllm_loggers["level"] = envs.VLLM_LOGGING_LEVEL

    if envs.VLLM_LOGGING_CONFIG_PATH:
        if not path.exists(envs.VLLM_LOGGING_CONFIG_PATH):
            raise RuntimeError(f"Logging config file not found: {envs.VLLM_LOGGING_CONFIG_PATH}")
        with open(envs.VLLM_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            custom_config = json.loads(file.read())
        if not isinstance(custom_config, dict):
            raise ValueError(f"Invalid logging config. Expected dict, got {type(custom_config).__name__}")
        logging_config = custom_config

    # Backwards compatibility for #10134
    for formatter in logging_config.get("formatters", {}).values():
        if formatter.get("class") == "vllm.logging.NewLineFormatter":
            formatter["class"] = "vllm.logging_utils.NewLineFormatter"

    if logging_config:
        dictConfig(logging_config)
```

### Logger Initialization

**Lines:** 206-216

```python
def init_logger(name: str) -> _VllmLogger:
    logger = logging.getLogger(name)

    # Patch methods onto logger instance
    for method_name, method in _METHODS_TO_PATCH.items():
        setattr(logger, method_name, MethodType(method, logger))

    return cast(_VllmLogger, logger)
```

**Pattern:** Dynamically adds `*_once` methods to avoid conflicts with other libraries (e.g., `intel_extension_for_pytorch`).

### Context Manager

**Lines:** 219-224

```python
@contextmanager
def suppress_logging(level: int = logging.INFO) -> Generator[None, Any, None]:
    current_level = logging.root.manager.disable
    logging.disable(level)
    yield
    logging.disable(current_level)
```

## Function Call Tracing

**Lines:** 240-303

```python
def _trace_calls(log_path, root_dir, frame, event, arg=None):
    if event in ["call", "return"]:
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name

        if not filename.startswith(root_dir):
            return  # Only log functions in vllm root_dir

        try:
            last_frame = frame.f_back
            if last_frame is not None:
                last_filename = last_frame.f_code.co_filename
                last_lineno = last_frame.f_lineno
                last_func_name = last_frame.f_code.co_name
            else:
                # Initial frame
                last_filename = ""
                last_lineno = 0
                last_func_name = ""

            with open(log_path, "a") as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                if event == "call":
                    f.write(f"{ts} Call to {func_name} in {filename}:{lineno} "
                           f"from {last_func_name} in {last_filename}:{last_lineno}\n")
                else:
                    f.write(f"{ts} Return from {func_name} in {filename}:{lineno} "
                           f"to {last_func_name} in {last_filename}:{last_lineno}\n")
        except NameError:
            pass  # Modules deleted during shutdown

    return partial(_trace_calls, log_path, root_dir)

def enable_trace_function_call(log_file_path: str, root_dir: str | None = None):
    logger.warning(
        "VLLM_TRACE_FUNCTION is enabled. It will record every function "
        "executed by Python. This will slow down the code significantly."
    )
    logger.info("Trace frame log is saved to %s", log_file_path)

    if root_dir is None:
        root_dir = os.path.dirname(os.path.dirname(__file__))  # vllm root

    sys.settrace(partial(_trace_calls, log_file_path, root_dir))
```

**Usage:** For debugging hangs or crashes by logging all function calls/returns.

## Module Initialization

**Lines:** 227-237

```python
# Root logger configured at module import time
_configure_vllm_root_logger()

# Set httpx logging to WARNING when vLLM is INFO
if envs.VLLM_LOGGING_LEVEL == "INFO":
    logging.getLogger("httpx").setLevel(logging.WARNING)

logger = init_logger(__name__)
```

**Thread Safety:** Protected by Python's GIL and module import lock.

## Usage Patterns

### Basic Usage

```python
from vllm.logger import init_logger

logger = init_logger(__name__)
logger.info("Standard log message")
logger.info_once("This will only print once")
logger.info_once("This will only print once", scope="global")  # Only rank 0
```

### Distributed Logging

```python
# Log only on first rank globally
logger.info_once("Model loaded", scope="global")

# Log only on first rank per node
logger.warning_once("GPU memory low", scope="local")

# Log on every process
logger.debug_once("Processing batch", scope="process")
```

### Suppressing Logs

```python
from vllm.logger import suppress_logging

with suppress_logging(logging.WARNING):
    # Only ERROR and CRITICAL messages printed here
    some_noisy_function()
```

### Function Tracing

```python
from vllm.logger import enable_trace_function_call

# Enable tracing for debugging
enable_trace_function_call("/tmp/vllm_trace.log", root_dir="/path/to/vllm")
# All function calls in vllm will be logged to file
```

## Environment Variables

### Core Configuration

- `VLLM_CONFIGURE_LOGGING`: Enable/disable vLLM logging config
- `VLLM_LOGGING_LEVEL`: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `VLLM_LOGGING_STREAM`: Output stream (ext://sys.stdout or ext://sys.stderr)
- `VLLM_LOGGING_CONFIG_PATH`: Path to custom logging config JSON

### Formatting

- `VLLM_LOGGING_PREFIX`: Prefix for all log messages
- `VLLM_LOGGING_COLOR`: Force color on/off ("0" or "1")
- `NO_COLOR`: Standard environment variable to disable color

## Integration Points

### vllm.envs

Provides all environment variable values used by logger.

### vllm.logging_utils

- `NewLineFormatter`: Handles multi-line messages
- `ColoredFormatter`: ANSI color formatting

### vllm.distributed.parallel_state

- `is_global_first_rank()`: Global rank 0 check
- `is_local_first_rank()`: Node-local rank 0 check

### All vLLM Modules

Every module uses `init_logger(__name__)` to obtain a configured logger.

## Custom Formatters

### NewLineFormatter

Handles multi-line log messages by indenting continuation lines.

### ColoredFormatter

Adds ANSI color codes based on log level:
- DEBUG: Cyan
- INFO: Green
- WARNING: Yellow
- ERROR: Red
- CRITICAL: Bold red

## Performance Considerations

### Benefits

1. **Deduplication:** `*_once` methods prevent log spam in loops
2. **Distributed Aware:** Avoids redundant logs from all ranks
3. **Lazy Import:** Distributed functions imported only when needed

### Overhead

1. **LRU Cache:** Small memory overhead for cached messages
2. **Scope Checks:** Minimal rank checking overhead
3. **Function Tracing:** Significant performance impact (debugging only)

## Related Components

- **vllm/envs.py:** Environment variable definitions
- **vllm/logging_utils.py:** Custom formatter implementations
- **vllm/distributed/parallel_state.py:** Distributed rank information

## Technical Significance

This module is foundational infrastructure:
- **Consistency:** All vLLM logs have uniform format
- **Debuggability:** Function tracing helps diagnose complex issues
- **Distributed-First:** Scope awareness prevents log explosion in multi-GPU setups
- **Configurability:** Environment variables + JSON config enable flexible deployment
- **Compatibility:** Avoids conflicts with other logging frameworks

The `*_once` pattern is particularly valuable in distributed settings where the same message could be logged hundreds of times (once per GPU), and the scope filtering ensures operational logs come only from rank 0 while allowing per-rank debugging when needed.
