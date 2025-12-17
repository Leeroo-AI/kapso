# Implementation: Backward Compatibility Linter Decorators

**File:** `/tmp/praxium_repo_583nq7ea/vllm/_bc_linter.py`
**Type:** Development Tooling
**Lines of Code:** 54
**Last Updated:** 2025-12-17

## Overview

The `_bc_linter.py` module provides decorator functions for controlling backward compatibility (BC) linting in the vLLM codebase. These decorators allow developers to explicitly mark code elements as exempt from or included in backward compatibility checks, enabling fine-grained control over API stability enforcement.

### Purpose

Facilitates systematic backward compatibility management by providing decorators to mark functions, classes, and methods for inclusion or exclusion from BC linting tools.

### Key Features

- **No-Op Decorators**: Zero runtime overhead
- **Flexible Application**: Can decorate with or without parameters
- **Type-Safe**: Includes type hints and overloads
- **Dual Control**: Both skip and include directives
- **Optional Reasoning**: Can document why BC checking is modified

## Architecture

### Decorator Design Pattern

```
┌────────────────────────────────────┐
│   BC Linter Tool                   │
│   (External Analysis)              │
└────────────┬───────────────────────┘
             │
             │ Scans for decorators
             │
             ▼
┌────────────────────────────────────┐
│   Code Elements                    │
├────────────────────────────────────┤
│  @bc_linter_skip                   │
│  def legacy_api(): ...             │  ← Excluded from BC checks
│                                    │
│  @bc_linter_include                │
│  def public_api(): ...             │  ← Explicitly included
│                                    │
│  def internal_function(): ...      │  ← Default behavior
└────────────────────────────────────┘
```

### No-Op Decorator Pattern

```python
# Runtime: No-op (returns input unchanged)
@bc_linter_skip
def function():
    pass

# Static Analysis: Detects decorator and modifies checks
# Runtime: Function executes normally without overhead
```

## Implementation Details

### Type System Setup

```python
from collections.abc import Callable
from typing import Any, TypeVar, overload

T = TypeVar("T")
```

**Type Variable:**
- `T`: Generic type for decorated objects
- Preserves type information through decoration
- Enables type checking in IDEs and mypy

### bc_linter_skip Decorator

#### Type Overloads

```python
@overload
def bc_linter_skip(obj: T) -> T: ...

@overload
def bc_linter_skip(*, reason: str | None = ...) -> Callable[[T], T]: ...
```

**Overload Purpose:**
- First overload: Direct decoration without parameters
- Second overload: Parameterized decoration with reason

#### Implementation

```python
def bc_linter_skip(obj: Any = None, *, reason: str | None = None):
    """
    No-op decorator to mark symbols/files for BC-linter suppression.

    Usage:
        @bc_linter_skip
        def legacy_api(...): ...
    """

    def _wrap(x: T) -> T:
        return x

    return _wrap if obj is None else obj
```

**Decorator Logic:**
1. If `obj` is None: Called with parameters, return wrapper function
2. If `obj` is provided: Direct decoration, return object unchanged
3. `_wrap` function: Identity function (returns input)
4. `reason` parameter: Documentation only (not used at runtime)

### bc_linter_include Decorator

#### Type Overloads

```python
@overload
def bc_linter_include(obj: T) -> T: ...

@overload
def bc_linter_include(*, reason: str | None = ...) -> Callable[[T], T]: ...
```

**Same Pattern:** Consistent interface with `bc_linter_skip`

#### Implementation

```python
def bc_linter_include(obj: Any = None, *, reason: str | None = None):
    """
    Usage:
        @bc_linter_include
        def public_api(...): ...
    """

    def _wrap(x: T) -> T:
        return x

    return _wrap if obj is None else obj
```

**Identical Structure:** Same no-op pattern as skip decorator

### Module Exports

```python
__all__ = ["bc_linter_skip", "bc_linter_include"]
```

**Public API:** Only decorators are exported, hiding implementation details.

## Usage Patterns

### Direct Decoration (No Parameters)

```python
from vllm._bc_linter import bc_linter_skip

@bc_linter_skip
def legacy_function():
    """Old API that's being deprecated."""
    pass

@bc_linter_skip
class LegacyClass:
    """Legacy class with unstable API."""
    pass
```

**Syntax:** Decorator applied directly without parentheses

### Parameterized Decoration (With Reason)

```python
from vllm._bc_linter import bc_linter_skip

@bc_linter_skip(reason="Internal implementation detail, not public API")
def _internal_helper():
    pass

@bc_linter_skip(reason="Experimental feature under development")
def experimental_api():
    pass
```

**Syntax:** Decorator called with keyword argument

### Explicit Inclusion

```python
from vllm._bc_linter import bc_linter_include

@bc_linter_include
def stable_public_api():
    """This API must remain backward compatible."""
    pass

@bc_linter_include(reason="Core user-facing API")
class PublicModel:
    """Public model class with stability guarantees."""
    pass
```

**Use Case:** Make BC checking explicit for critical APIs

### Method Decoration

```python
class MyClass:
    @bc_linter_skip
    def _private_method(self):
        """Internal method."""
        pass

    @bc_linter_include
    def public_method(self):
        """Public API method."""
        pass

    def normal_method(self):
        """Default BC checking applies."""
        pass
```

### Multiple Decorators

```python
@some_other_decorator
@bc_linter_skip
def function():
    pass
```

**Stacking:** BC linter decorators can be combined with other decorators

## Integration with BC Linter Tool

### Static Analysis Detection

The BC linter tool (external to this module) scans the codebase for these decorators:

```python
# Pseudocode for BC linter tool
def should_check_bc(symbol):
    if has_decorator(symbol, "bc_linter_skip"):
        return False
    if has_decorator(symbol, "bc_linter_include"):
        return True
    return default_policy(symbol)
```

### Reason Extraction

```python
# BC linter can extract reason for documentation
decorator = get_decorator(symbol, "bc_linter_skip")
if decorator.has_reason():
    reason = decorator.get_reason()
    # Log or report reason for suppression
```

## Design Rationale

### Why No-Op at Runtime?

**Zero Overhead:**
```python
# Without decorator
def function():
    pass

# With BC linter decorator (identical performance)
@bc_linter_skip
def function():
    pass
```

**Performance:** No runtime penalty for BC management infrastructure

### Why Not Use Comments?

**Structured Approach:**
```python
# Bad: Comment-based approach
# BC_LINTER_SKIP: reason goes here
def function():
    pass

# Good: Decorator-based approach
@bc_linter_skip(reason="reason goes here")
def function():
    pass
```

**Advantages of Decorators:**
- Programmatically detectable
- Type-safe
- IDE-supported
- Consistent syntax
- Harder to forget or mistype

### Overload Pattern Benefits

```python
# Both syntaxes work correctly with type checking
@bc_linter_skip          # ← Type checker knows this returns T
def func1(): pass

@bc_linter_skip(reason="...") # ← Type checker knows this returns Callable[[T], T]
def func2(): pass
```

**Type Safety:** Mypy and IDE tools understand both usage patterns

## Use Cases

### 1. Deprecating APIs

```python
@bc_linter_skip(reason="Deprecated, will be removed in v1.0")
def old_api():
    warnings.warn("This API is deprecated", DeprecationWarning)
    # ... implementation
```

### 2. Internal Implementation Details

```python
@bc_linter_skip(reason="Internal helper, not part of public API")
def _compute_internal_state():
    # Complex internal logic
    pass
```

### 3. Experimental Features

```python
@bc_linter_skip(reason="Experimental, API may change")
def experimental_feature():
    # New feature under development
    pass
```

### 4. Explicitly Stable APIs

```python
@bc_linter_include(reason="Core public API with stability guarantee")
class LLM:
    """Main user-facing class."""
    pass
```

### 5. Private Module Members

```python
# In a module with mixed public/private APIs
@bc_linter_skip
def _internal_function():
    pass

@bc_linter_include
def public_function():
    pass
```

## Best Practices

### 1. Always Provide Reason for Skips

```python
# Bad: No explanation
@bc_linter_skip
def function():
    pass

# Good: Clear reasoning
@bc_linter_skip(reason="Legacy API scheduled for removal in v2.0")
def function():
    pass
```

### 2. Use Include for Critical APIs

```python
# Explicitly mark stability guarantees
@bc_linter_include(reason="Stable API used by all major clients")
def critical_function():
    pass
```

### 3. Apply at Appropriate Scope

```python
# Skip entire class if all methods are internal
@bc_linter_skip(reason="Internal implementation class")
class _InternalHelper:
    def method1(self): pass
    def method2(self): pass

# Or skip individual methods if mixed
class PublicClass:
    @bc_linter_skip(reason="Internal helper")
    def _helper(self): pass

    @bc_linter_include
    def public_method(self): pass
```

### 4. Document BC Policies

```python
"""
Module: vllm.public_api

All functions in this module are subject to BC checks unless
explicitly marked with @bc_linter_skip.
"""

@bc_linter_skip(reason="Under development")
def beta_feature():
    pass
```

## Testing Considerations

### Unit Testing

```python
import pytest
from vllm._bc_linter import bc_linter_skip, bc_linter_include

def test_bc_linter_skip_no_op():
    """Verify decorator doesn't modify function."""
    def original_function():
        return 42

    decorated = bc_linter_skip(original_function)
    assert decorated is original_function
    assert decorated() == 42

def test_bc_linter_skip_with_reason():
    """Verify parameterized decorator works."""
    @bc_linter_skip(reason="test reason")
    def function():
        return "test"

    assert function() == "test"
```

### Integration Testing

```python
def test_decorator_detection():
    """Test that BC linter tool can detect decorators."""
    import inspect

    @bc_linter_skip
    def test_function():
        pass

    # BC linter tool should be able to inspect this
    # (actual implementation in separate BC linter tool)
```

## Relationship to BC Linter Infrastructure

### Component Architecture

```
┌──────────────────────────────────────────┐
│   BC Linter Tool (Separate Component)   │
│   - Scans codebase                       │
│   - Detects decorators                   │
│   - Analyzes API changes                 │
│   - Reports BC violations                │
└────────────┬─────────────────────────────┘
             │
             │ Uses decorators from
             │
             ▼
┌──────────────────────────────────────────┐
│   _bc_linter.py (This Module)            │
│   - Provides decorators                  │
│   - No runtime logic                     │
│   - Type-safe interface                  │
└──────────────────────────────────────────┘
             │
             │ Used by developers in
             │
             ▼
┌──────────────────────────────────────────┐
│   vLLM Codebase                          │
│   - Apply decorators to code             │
│   - Mark skip/include directives         │
│   - Document BC policies                 │
└──────────────────────────────────────────┘
```

### Workflow

1. **Development**: Developer applies decorators to code
2. **CI/CD**: BC linter tool runs during testing
3. **Analysis**: Tool detects decorators and adjusts checks
4. **Reporting**: Violations reported (respecting skip/include)

## Error Handling

### Invalid Usage Detection

```python
# BC linter tool can detect misuse
@bc_linter_skip(invalid_param="value")  # ← Tool warns about invalid param
def function():
    pass
```

### Type Checking

```python
# Type checker catches incorrect usage
@bc_linter_skip(reason=123)  # ← mypy error: reason should be str
def function():
    pass
```

## Performance Impact

### Runtime Overhead

```python
import timeit

def plain_function():
    return 42

@bc_linter_skip
def decorated_function():
    return 42

# Both have identical performance (no overhead)
print(timeit.timeit(plain_function))        # e.g., 0.05s
print(timeit.timeit(decorated_function))    # e.g., 0.05s (same)
```

### Import Time

```python
# Minimal import cost (simple module)
import time
start = time.time()
from vllm._bc_linter import bc_linter_skip
print(f"Import time: {time.time() - start:.6f}s")  # ~0.0001s
```

## Future Enhancements

### Potential Extensions

1. **Version Tracking**: Track when APIs were marked for skip
2. **Automatic Reporting**: Generate BC documentation from decorators
3. **Graduated Levels**: Skip for minor versions, enforce for major
4. **Team Ownership**: Associate skips with team/developer

### Example Future API

```python
# Hypothetical future enhancement
@bc_linter_skip(
    reason="Experimental",
    since_version="0.5.0",
    planned_stable="1.0.0",
    owner="ml-team"
)
def experimental_feature():
    pass
```

## Related Components

- **BC Linter Tool**: External tool that uses these decorators
- **API Documentation**: Generated docs may reflect BC status
- **CI/CD Pipeline**: Integrates BC checking in testing
- **Version Management**: BC policies tied to release strategy

## References

- **Source File**: `vllm/_bc_linter.py`
- **Type Hints**: Python 3.10+ style (PEP 604 unions with `|`)
- **Decorator Pattern**: PEP 318 decorators
- **Overloading**: PEP 484 typing.overload
- **Repository**: https://github.com/vllm-project/vllm

## Summary

The `_bc_linter.py` module provides a lightweight, type-safe interface for managing backward compatibility checking in the vLLM codebase. By using no-op decorators, it achieves zero runtime overhead while enabling sophisticated static analysis for API stability enforcement. The dual decorator approach (`bc_linter_skip` and `bc_linter_include`) gives developers fine-grained control over which code elements are subject to BC checks, supporting a mature API governance strategy.

---

*This module is part of vLLM's development tooling infrastructure for maintaining API backward compatibility and stability guarantees.*
