# ScalarType - Sub-Byte Numeric Type System

## Overview

**File:** `/tmp/praxium_repo_583nq7ea/vllm/scalar_type.py` (355 lines)

**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

**Purpose:** Implements a comprehensive type system for sub-byte and custom numeric types beyond PyTorch's standard dtypes, enabling support for diverse quantization schemes (GPTQ, AWQ, FP8, FP6, FP4) with Python/C++ interoperability.

## Core Architecture

### NanRepr Enum

**Lines:** 13-16

```python
class NanRepr(Enum):
    NONE = 0               # NaNs not supported
    IEEE_754 = 1           # Standard: exp all 1s, mantissa not all 0s
    EXTD_RANGE_MAX_MIN = 2 # Extended range: exp all 1s, mantissa all 1s
```

### ScalarType Dataclass

**Lines:** 22-260

```python
@dataclass(frozen=True)
class ScalarType:
    exponent: int        # Bits in exponent (0 for integer types)
    mantissa: int        # Bits in mantissa/integer (excluding sign)
    signed: bool         # Has sign bit
    bias: int           # Value offset (stored_value = value + bias)
    _finite_values_only: bool = False  # Infs not supported
    nan_repr: NanRepr = NanRepr.IEEE_754

    @property
    def size_bits(self) -> int:
        return self.exponent + self.mantissa + int(self.signed)
```

**Key Concept:** `bias` enables representing negative values with unsigned storage:
- Example: `uint4b8` stores values as unsigned 4-bit with bias of 8
- Value 0 is stored as 8, value -1 as 7, value 1 as 9

## Type Algebra

### Min/Max Calculation

**Lines:** 71-134

#### Floating Point Max

```python
def _floating_point_max_int(self) -> int:
    # Calculate max representable value as IEEE double raw bits
    max_mantissa = (1 << self.mantissa) - 1
    if self.nan_repr == NanRepr.EXTD_RANGE_MAX_MIN:
        max_mantissa -= 1  # Reserve all-1s for NaN

    max_exponent = (1 << self.exponent) - 2
    if self.nan_repr in [NanRepr.EXTD_RANGE_MAX_MIN, NanRepr.NONE]:
        max_exponent += 1  # Use full exponent range

    # Adjust exponent bias to IEEE double format
    exponent_bias = (1 << (self.exponent - 1)) - 1
    exponent_bias_double = (1 << 10) - 1  # double has 11 exp bits
    max_exponent_double = max_exponent - exponent_bias + exponent_bias_double

    # Pack into IEEE double format
    return (max_mantissa << (52 - self.mantissa)) | (max_exponent_double << 52)

def _floating_point_max(self) -> float:
    double_raw = self._floating_point_max_int()
    return struct.unpack("!d", struct.pack("!Q", double_raw))[0]
```

**Clever Technique:** Converts custom FP format to IEEE double by bit manipulation, then unpacks as float.

#### Integer Max

```python
def _raw_max(self) -> int | float:
    if self.is_floating_point():
        return self._floating_point_max()
    else:
        assert self.size_bits < 64 or (self.size_bits == 64 and self.is_signed())
        return (1 << self.mantissa) - 1
```

#### Min Calculation

```python
def _raw_min(self) -> int | float:
    if self.is_floating_point():
        # Set sign bit on max value
        sign_bit_double = 1 << 63
        max_raw = self._floating_point_max_int()
        min_raw = max_raw | sign_bit_double
        return struct.unpack("!d", struct.pack("!Q", min_raw))[0]
    else:
        if self.is_signed():
            return -(1 << (self.size_bits - 1))
        else:
            return 0
```

### Public Min/Max API

**Lines:** 170-182

```python
def min(self) -> int | float:
    """Min representable value (accounting for bias)"""
    return self._raw_min() - self.bias

def max(self) -> int | float:
    """Max representable value (accounting for bias)"""
    return self._raw_max() - self.bias
```

## Python/C++ Interoperability

### ID Encoding

**Lines:** 136-164

```python
@functools.cached_property
def id(self) -> int:
    """
    Encode ScalarType as int64 for passing to C++ kernels.
    Layout must match C++ ScalarType::from_id() method.
    """
    val = 0
    offset = 0

    def or_and_advance(member, bit_width):
        nonlocal val, offset
        bit_mask = (1 << bit_width) - 1
        val = val | (int(member) & bit_mask) << offset
        offset = offset + bit_width

    or_and_advance(self.exponent, 8)
    or_and_advance(self.mantissa, 8)
    or_and_advance(self.signed, 1)
    or_and_advance(self.bias, 32)
    or_and_advance(self._finite_values_only, 1)
    or_and_advance(self.nan_repr.value, 8)

    assert offset <= 64, f"ScalarType fields too big {offset} to fit into int64"

    _SCALAR_TYPES_ID_MAP[val] = self  # Cache for from_id()
    return val
```

**Bit Layout:**
- Bits 0-7: exponent (8 bits)
- Bits 8-15: mantissa (8 bits)
- Bit 16: signed (1 bit)
- Bits 17-48: bias (32 bits)
- Bit 49: finite_values_only (1 bit)
- Bits 50-57: nan_repr (8 bits)

### From ID

**Lines:** 308-312

```python
@classmethod
def from_id(cls, scalar_type_id: int):
    if scalar_type_id not in _SCALAR_TYPES_ID_MAP:
        raise ValueError(f"scalar_type_id {scalar_type_id} doesn't exist.")
    return _SCALAR_TYPES_ID_MAP[scalar_type_id]
```

## Type Introspection

**Lines:** 184-216

```python
def is_signed(self) -> bool:
    return self.signed

def is_floating_point(self) -> bool:
    return self.exponent != 0

def is_integer(self) -> bool:
    return self.exponent == 0

def has_bias(self) -> bool:
    return self.bias != 0

def has_infs(self) -> bool:
    return not self._finite_values_only

def has_nans(self) -> bool:
    return self.nan_repr != NanRepr.NONE.value

def is_ieee_754(self) -> bool:
    return self.nan_repr == NanRepr.IEEE_754.value and not self._finite_values_only
```

## String Representation

**Lines:** 218-255

```python
def __str__(self) -> str:
    """
    Naming follows https://github.com/jax-ml/ml_dtypes

    Floating point: float<size>_e<exp>m<mantissa>[flags]
      Flags: f (finite only), n (non-standard NaN)
      No flags = IEEE 754

    Integer: [u]int<size>[b<bias>]
    """
    if self.is_floating_point():
        ret = f"float{self.size_bits}_e{self.exponent}m{self.mantissa}"
        if not self.is_ieee_754():
            if self._finite_values_only:
                ret += "f"
            if self.nan_repr != NanRepr.NONE:
                ret += "n"
        return ret
    else:
        ret = ("int" if self.is_signed() else "uint") + str(self.size_bits)
        if self.has_bias():
            ret += "b" + str(self.bias)
        return ret
```

**Examples:**
- `float8_e4m3fn`: 8-bit float, 4 exp bits, 3 mantissa, finite-only, non-standard NaN
- `uint4b8`: 4-bit unsigned int with bias of 8
- `float16_e5m10`: Standard IEEE 754 half-precision (no flags)

## Convenience Constructors

**Lines:** 262-306

```python
@classmethod
def int_(cls, size_bits: int, bias: int | None) -> "ScalarType":
    """Signed integer (size_bits includes sign bit)"""
    ret = cls(0, size_bits - 1, True, bias if bias else 0)
    ret.id  # Cache the ID
    return ret

@classmethod
def uint(cls, size_bits: int, bias: int | None) -> "ScalarType":
    """Unsigned integer"""
    ret = cls(0, size_bits, False, bias if bias else 0)
    ret.id
    return ret

@classmethod
def float_IEEE754(cls, exponent: int, mantissa: int) -> "ScalarType":
    """Standard IEEE 754 floating point"""
    assert mantissa > 0 and exponent > 0
    ret = cls(exponent, mantissa, True, 0)
    ret.id
    return ret

@classmethod
def float_(cls, exponent: int, mantissa: int, finite_values_only: bool,
           nan_repr: NanRepr) -> "ScalarType":
    """Non-standard floating point"""
    assert mantissa > 0 and exponent > 0
    assert nan_repr != NanRepr.IEEE_754
    ret = cls(exponent, mantissa, True, 0, finite_values_only, nan_repr)
    ret.id
    return ret
```

## Predefined Types

**Lines:** 327-355

```python
class scalar_types:
    # Integer types
    int4 = ScalarType.int_(4, None)
    uint4 = ScalarType.uint(4, None)
    int8 = ScalarType.int_(8, None)
    uint8 = ScalarType.uint(8, None)

    # FP8 types
    float8_e4m3fn = ScalarType.float_(4, 3, True, NanRepr.EXTD_RANGE_MAX_MIN)
    float8_e5m2 = ScalarType.float_IEEE754(5, 2)
    float8_e8m0fnu = ScalarType(8, 0, False, 0, True, NanRepr.EXTD_RANGE_MAX_MIN)

    # FP16 types
    float16_e8m7 = ScalarType.float_IEEE754(8, 7)  # bfloat16
    float16_e5m10 = ScalarType.float_IEEE754(5, 10)  # standard float16

    # FP6 types (fp6_llm project)
    float6_e3m2f = ScalarType.float_(3, 2, True, NanRepr.NONE)
    float6_e2m3f = ScalarType.float_(2, 3, True, NanRepr.NONE)

    # FP4 types (OCP microscaling formats)
    float4_e2m1f = ScalarType.float_(2, 1, True, NanRepr.NONE)

    # GPTQ types with bias
    uint2b2 = ScalarType.uint(2, 2)
    uint3b4 = ScalarType.uint(3, 4)
    uint4b8 = ScalarType.uint(4, 8)
    uint8b128 = ScalarType.uint(8, 128)

    # Colloquial names
    bfloat16 = float16_e8m7
    float16 = float16_e5m10
```

## Usage Patterns

### Creating Custom Types

```python
# 3-bit unsigned integer with bias of 4
custom_type = ScalarType.uint(3, 4)
print(custom_type)  # "uint3b4"
print(custom_type.min())  # -4 (accounting for bias)
print(custom_type.max())  # 3

# 6-bit float with 3 exp bits, 2 mantissa, finite only
fp6 = ScalarType.float_(3, 2, finite_values_only=True, nan_repr=NanRepr.NONE)
print(fp6)  # "float6_e3m2f"
```

### Passing to C++ Kernels

```python
import torch
from vllm.scalar_type import scalar_types

# Get scalar type for GPTQ 4-bit weights
stype = scalar_types.uint4b8
stype_id = stype.id  # Encode as int64

# Pass to custom CUDA kernel
torch.ops.vllm.quantized_matmul(
    input, weight, scale, stype_id  # Kernel can decode type info
)
```

### Type Checking

```python
stype = scalar_types.float8_e4m3fn

if stype.is_floating_point():
    print(f"FP type with {stype.exponent} exp bits")

if stype.has_nans():
    print(f"Supports NaN via {stype.nan_repr}")

if not stype.has_infs():
    print("Extended value range (no infinity)")
```

### Weight Quantization

```python
def quantize_weights(weights, target_type):
    min_val = target_type.min()
    max_val = target_type.max()

    # Clamp and scale
    scaled = torch.clamp(weights, min_val, max_val)

    if target_type.has_bias():
        # Add bias before storing
        scaled = scaled + target_type.bias

    return scaled.to(torch.uint8 if target_type.size_bits <= 8 else torch.int16)
```

## Integration Points

### C++ Kernels

**csrc/core/scalar_type.hpp:** Mirror implementation with `from_id()` decoder.

### Quantization Systems

- **GPTQ:** Uses `uint4b8`, `uint3b4`, `uint2b2`
- **AWQ:** Uses `uint4`, `int4`
- **FP8:** Uses `float8_e4m3fn`, `float8_e5m2`
- **FP6/FP4:** Uses `float6_e3m2f`, `float4_e2m1f`

### CUTLASS Integration

**csrc/cutlass_extensions/vllm_cutlass_library_extension.py:** Maps vLLM scalar types to CUTLASS types.

## Technical Challenges

### Sub-Byte Representation

PyTorch doesn't natively support sub-byte types, so:
- 4-bit values packed into int8/uint8 storage
- Custom unpacking logic in CUDA kernels
- Careful attention to alignment and padding

### IEEE 754 Conversion

Converting custom FP formats to doubles requires:
1. Bit-level manipulation of exponent/mantissa
2. Exponent bias adjustment
3. Careful handling of special values (NaN, inf)

### Type Safety

The frozen dataclass ensures:
- Immutable types (can be dictionary keys)
- Cached IDs for efficient lookup
- Type checking via `isinstance()`

## Performance Considerations

### Benefits

1. **Memory Efficiency:** Sub-byte types dramatically reduce model size
2. **Compute Efficiency:** FP8/FP6 enable faster matrix operations
3. **Cached IDs:** `@cached_property` avoids repeated encoding

### Overhead

1. **Bit Packing/Unpacking:** Small overhead in kernels
2. **Range Mapping:** Bias adds/subtracts in conversions

## Related Components

- **csrc/core/scalar_type.hpp:** C++ mirror implementation
- **csrc/quantization/:** Quantization kernels using scalar types
- **csrc/cutlass_extensions/:** CUTLASS integration
- **vllm/model_executor/layers/quantization/:** Python quantization logic

## Technical Significance

This module is foundational for quantization:
- **Expressiveness:** Can represent virtually any numeric format
- **Extensibility:** Easy to add new types for future quantization schemes
- **Interoperability:** Seamless Python/C++ type sharing
- **Correctness:** Precise min/max/bias calculations ensure numeric stability
- **Standards-Based:** Follows ml_dtypes naming conventions

The bias feature is particularly innovative, enabling unsigned storage of signed values, which is common in symmetric quantization schemes where the zero point is offset from true zero.
