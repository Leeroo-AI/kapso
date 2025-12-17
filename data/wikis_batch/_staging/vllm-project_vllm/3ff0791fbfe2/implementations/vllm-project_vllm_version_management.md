# Implementation: Version Management and Compatibility

**File:** `/tmp/praxium_repo_583nq7ea/vllm/version.py` (39 lines)
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

## Overview

The `version.py` module provides version tracking and compatibility checking for vLLM. It imports version information from an auto-generated module with fallback handling, and provides utilities for comparing against previous versions to support feature deprecation and backward compatibility.

**Key Components:**
- `__version__`: String version identifier (e.g., "0.6.3")
- `__version_tuple__`: Tuple version for programmatic comparison (e.g., (0, 6, 3))
- `_prev_minor_version_was()`: Check if given version matches previous minor
- `_prev_minor_version()`: Get previous minor version string

## Implementation Details

### Version Import with Fallback

```python
try:
    from ._version import __version__, __version_tuple__
except Exception as e:
    import warnings

    warnings.warn(f"Failed to read commit hash:\n{e}", RuntimeWarning, stacklevel=2)

    __version__ = "dev"
    __version_tuple__ = (0, 0, __version__)
```

**Import Strategy:**

1. **Try Import**: Attempt to import from `_version` module
   - Generated at build time by version control systems (e.g., setuptools-scm)
   - Contains actual version from git tags/commits

2. **Fallback on Failure**: Set development defaults
   - `__version__ = "dev"`: String identifier for dev builds
   - `__version_tuple__ = (0, 0, "dev")`: Tuple with special sentinel

3. **Warning**: Issue RuntimeWarning with exception details
   - `stacklevel=2`: Points to caller, not this module
   - Helps debugging installation issues

**When Import Fails:**
- Source installation without proper build
- Editable/development installation
- Missing _version.py file
- Build system issues

### Version Tuple Structure

```python
__version_tuple__ = (major, minor, patch_or_dev)

# Examples:
(0, 6, 3)      # Release version 0.6.3
(0, 6, "3.1")  # Release version 0.6.3.1
(0, 0, "dev")  # Development version
```

**Components:**
- **major**: Major version number (currently always 0)
- **minor**: Minor version number (increments with features)
- **patch_or_dev**: Patch number or "dev" string

### Previous Minor Version Check

```python
def _prev_minor_version_was(version_str):
    """Check whether a given version matches the previous minor version.

    Return True if version_str matches the previous minor version.

    For example - return True if the current version if 0.7.4 and the
    supplied version_str is '0.6'.

    Used for --show-hidden-metrics-for-version.
    """
    # Match anything if this is a dev tree
    if __version_tuple__[0:2] == (0, 0):
        return True

    # Note - this won't do the right thing when we release 1.0!
    assert __version_tuple__[0] == 0
    assert isinstance(__version_tuple__[1], int)
    return version_str == f"{__version_tuple__[0]}.{__version_tuple__[1] - 1}"
```

**Logic Breakdown:**

#### 1. Development Build Handling
```python
if __version_tuple__[0:2] == (0, 0):
    return True
```
- In dev builds, always return True
- Allows testing deprecated features without version checks
- Prevents test failures in development environment

#### 2. Current Version (Pre-1.0) Handling
```python
assert __version_tuple__[0] == 0
assert isinstance(__version_tuple__[1], int)
return version_str == f"{__version_tuple__[0]}.{__version_tuple__[1] - 1}"
```

**Example:**
- Current version: "0.7.4" → `__version_tuple__ = (0, 7, 4)`
- Previous minor: "0.6"
- Check: `version_str == "0.6"` → `"0.6" == f"{0}.{7 - 1}"` → `"0.6" == "0.6"` → True

**Assertions:**
- `assert __version_tuple__[0] == 0`: Assumes pre-1.0 (TODO for 1.0+ handling)
- `assert isinstance(__version_tuple__[1], int)`: Ensures numeric minor version

### Get Previous Minor Version

```python
def _prev_minor_version():
    """For the purpose of testing, return a previous minor version number."""
    # In dev tree, this will return "0.-1", but that will work fine"
    assert isinstance(__version_tuple__[1], int)
    return f"{__version_tuple__[0]}.{__version_tuple__[1] - 1}"
```

**Purpose:**
- Returns previous minor version string
- Used for testing version comparison logic
- Comments acknowledge odd behavior in dev ("0.-1")

**Examples:**
- Current: "0.7.4" → Returns: "0.6"
- Current: "0.6.0" → Returns: "0.5"
- Current: "dev" (0.0) → Returns: "0.-1" (acknowledged quirk)

## Usage Patterns

### Version Display

```python
from vllm.version import __version__

print(f"vLLM version: {__version__}")
# Output: "vLLM version: 0.6.3" or "vLLM version: dev"
```

### Programmatic Version Comparison

```python
from vllm.version import __version_tuple__

if __version_tuple__ >= (0, 6, 0):
    # Use feature introduced in 0.6.0
    enable_new_feature()
else:
    # Fall back for older versions
    use_legacy_feature()
```

### Deprecation Warning with Version Check

```python
from vllm.version import _prev_minor_version_was
import warnings

def deprecated_function(show_hidden_metrics_for_version=None):
    if show_hidden_metrics_for_version:
        if _prev_minor_version_was(show_hidden_metrics_for_version):
            # Show hidden metrics for specified previous version
            return legacy_metrics()
        else:
            warnings.warn(
                f"Version {show_hidden_metrics_for_version} is not "
                f"the previous minor version",
                DeprecationWarning
            )
    # Normal behavior
    return current_metrics()
```

### Command-Line Version Checking

```python
# CLI argument: --show-hidden-metrics-for-version 0.6

from vllm.version import _prev_minor_version_was

if args.show_hidden_metrics_for_version:
    if _prev_minor_version_was(args.show_hidden_metrics_for_version):
        # Enable deprecated metrics from previous version
        enable_legacy_metrics = True
    else:
        print(f"Warning: {args.show_hidden_metrics_for_version} is not "
              f"the previous minor version")
        enable_legacy_metrics = False
```

### Testing Version Logic

```python
from vllm.version import _prev_minor_version

def test_version_comparison():
    # Get what should be the previous version
    prev_version = _prev_minor_version()

    # Test that it's recognized as previous
    assert _prev_minor_version_was(prev_version)

    # Test that current version is not previous
    from vllm.version import __version__
    assert not _prev_minor_version_was(__version__)
```

### User-Agent Construction

```python
from vllm.version import __version__

# In HTTP client
headers = {
    "User-Agent": f"vLLM/{__version__}",
    # ...
}
```

### API Version Validation

```python
from vllm.version import __version_tuple__

MIN_API_VERSION = (0, 5, 0)

if __version_tuple__ < MIN_API_VERSION:
    raise RuntimeError(
        f"This script requires vLLM >= {'.'.join(map(str, MIN_API_VERSION))}, "
        f"but found vLLM {'.'.join(map(str, __version_tuple__))}"
    )
```

## Integration Points

### Command-Line Interface

```python
# vllm serve --help
import argparse
from vllm.version import __version__

parser = argparse.ArgumentParser(
    description=f"vLLM server (version {__version__})"
)
parser.add_argument(
    "--version",
    action="version",
    version=f"vLLM {__version__}"
)
parser.add_argument(
    "--show-hidden-metrics-for-version",
    type=str,
    help="Show hidden metrics for previous version (e.g., '0.6')"
)
```

### Metrics System

```python
from vllm.version import _prev_minor_version_was

class MetricsRegistry:
    def register_metric(self, name, hidden_since_version=None):
        if hidden_since_version:
            # Only show if user explicitly requested previous version
            if not (args.show_hidden_metrics_for_version and
                    _prev_minor_version_was(args.show_hidden_metrics_for_version)):
                return  # Skip hidden metric

        # Register metric
        self._metrics[name] = create_metric(name)
```

### Build System Integration

**setup.py / pyproject.toml:**
```python
# Uses setuptools-scm to generate _version.py
[tool.setuptools_scm]
write_to = "vllm/_version.py"
version_scheme = "post-release"
local_scheme = "no-local-version"
```

**Generated _version.py:**
```python
# AUTO-GENERATED by setuptools-scm
__version__ = "0.6.3"
__version_tuple__ = (0, 6, 3)
```

### Compatibility Checking

```python
def check_compatible_versions(client_version, server_version):
    """Check if client and server versions are compatible."""
    client_tuple = parse_version_tuple(client_version)
    server_tuple = parse_version_tuple(server_version)

    # Compatible if major and minor match
    return client_tuple[0:2] == server_tuple[0:2]

def parse_version_tuple(version_str):
    if version_str == "dev":
        return (0, 0, "dev")
    parts = version_str.split(".")
    return (int(parts[0]), int(parts[1]), parts[2] if len(parts) > 2 else 0)
```

## Version Scheme

### Semantic Versioning (Modified)

vLLM follows a pre-1.0 semantic versioning scheme:

**Format:** `major.minor.patch`

**Current State (0.x.y):**
- **major = 0**: Pre-1.0, API not stable
- **minor**: Incremented for new features, may include breaking changes
- **patch**: Bug fixes, backward compatible

**Version Examples:**
- `0.6.0`: Minor version 6, initial release
- `0.6.1`: Patch release with bug fixes
- `0.6.2`: Another patch release
- `0.7.0`: New minor version with features

### Development Versions

**Format:** `dev` (string)

**Characteristics:**
- Source installations without proper build
- Editable installations (`pip install -e .`)
- Development branches without tags
- Recognized as `(0, 0, "dev")` tuple

## Design Rationale

### Why Tuple and String?

**String Version (`__version__`):**
- Human-readable
- Used in logs, error messages, CLI
- Standard for Python packages

**Tuple Version (`__version_tuple__`):**
- Programmatic comparison
- No string parsing needed
- Enables `<`, `>`, `==` comparisons

### Why Fallback to "dev"?

**Alternative:** Raise error if _version not found
**Problem:** Breaks development installations

**Chosen:** Fallback to "dev"
**Benefits:**
- Development installations work
- Clear indication of non-release build
- Tests can still run

### Why _prev_minor_version_was()?

**Use Case:** Gradual deprecation
- Remove metric in 0.7.0
- Still show it in 0.7.x if user runs `--show-hidden-metrics-for-version 0.6`
- Prevents immediate breakage of monitoring dashboards
- Gives users time to update

**Alternative:** Hard removal
**Problem:** Sudden monitoring breakage

### Why Dev Build Always Returns True?

```python
if __version_tuple__[0:2] == (0, 0):
    return True
```

**Rationale:**
- Tests run in dev environment
- Don't want version checks to fail tests
- Developers should see all features/metrics

## Known Limitations

### TODO: 1.0 Handling

```python
# Note - this won't do the right thing when we release 1.0!
assert __version_tuple__[0] == 0
```

**Current Behavior:**
- Assumes major version is always 0
- `_prev_minor_version_was("0.6")` works for version 0.7.x

**Post-1.0 Behavior (needs fix):**
- Version 1.0.0 should have previous minor 0.last
- Version 1.1.0 should have previous minor 1.0
- Current logic doesn't handle this transition

**Future Fix:**
```python
def _prev_minor_version_was(version_str):
    if __version_tuple__[0:2] == (0, 0):
        return True

    major, minor = __version_tuple__[0:2]

    # Handle major version boundaries
    if minor == 0:
        # First minor of new major, previous is last minor of prev major
        # This requires tracking what the last minor was
        return version_str == f"{major - 1}.X"  # TODO: determine X

    # Normal case within same major
    return version_str == f"{major}.{minor - 1}"
```

### Dev Version Quirk

```python
# In dev tree, this will return "0.-1", but that will work fine
_prev_minor_version() → "0.-1"
```

**Issue:** Technically invalid version string

**Why It's OK:**
- Only used for testing
- Dev builds return True for all version checks anyway
- Not user-facing

## Testing Considerations

### Mock Version

```python
from unittest.mock import patch
import vllm.version

def test_feature_with_old_version():
    with patch.object(vllm.version, '__version_tuple__', (0, 5, 0)):
        # Test behavior with old version
        result = version_dependent_function()
        assert result == "old_behavior"

def test_feature_with_new_version():
    with patch.object(vllm.version, '__version_tuple__', (0, 7, 0)):
        result = version_dependent_function()
        assert result == "new_behavior"
```

### Test Previous Version Logic

```python
def test_prev_minor_version_check():
    from vllm.version import _prev_minor_version_was, __version_tuple__

    if __version_tuple__[0:2] == (0, 0):
        # Dev version, skip test
        pytest.skip("Dev version, version checks not meaningful")

    major, minor, _ = __version_tuple__

    # Previous minor should be recognized
    prev_minor = f"{major}.{minor - 1}"
    assert _prev_minor_version_was(prev_minor)

    # Current minor should not be recognized as previous
    current_minor = f"{major}.{minor}"
    assert not _prev_minor_version_was(current_minor)

    # Future minor should not be recognized
    future_minor = f"{major}.{minor + 1}"
    assert not _prev_minor_version_was(future_minor)
```

### Test Version Import Fallback

```python
def test_version_import_failure():
    # Simulate import failure
    with patch('vllm.version._version', side_effect=ImportError):
        # Re-import module
        import importlib
        import vllm.version
        importlib.reload(vllm.version)

        # Should fall back to dev
        assert vllm.version.__version__ == "dev"
        assert vllm.version.__version_tuple__ == (0, 0, "dev")
```

## Related Components

- **vllm.connections**: Uses `__version__` for User-Agent header
- **vllm.engine.metrics**: Uses `_prev_minor_version_was()` for deprecation
- **vllm.entrypoints.openai**: Exposes version in API responses
- **Build System (setuptools-scm)**: Generates _version.py

## Future Enhancements

1. **1.0+ Support**: Fix `_prev_minor_version_was()` for post-1.0 versions
2. **Git Metadata**: Include git commit hash in version string
3. **Build Metadata**: Include build date, Python version in version info
4. **Compatibility Matrix**: Track which versions are compatible
5. **Deprecation Registry**: Centralized tracking of deprecated features by version

## Summary

The `version.py` module provides essential version management for vLLM with a focus on graceful degradation and backward compatibility. Its fallback mechanism ensures development installations work smoothly, while the previous version checking functions enable gradual deprecation of features. The dual representation (string and tuple) balances human readability with programmatic convenience, making it a robust foundation for version-dependent behavior throughout the codebase.
