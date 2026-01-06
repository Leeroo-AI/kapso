{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Quantization]], [[domain::Type_System]], [[domain::Sub_Byte_Types]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Sub-byte scalar type representation system supporting custom floating-point and integer types with bias for quantization schemes.

=== Description ===
The scalar_type.py module is a 355-line type system that enables vLLM to represent and work with sub-byte data types, including exotic floating-point formats and biased integer types used in various quantization schemes. This is a Python mirror of the C++ ScalarType implementation in csrc/core/scalar_type.hpp, designed to bridge the gap until PyTorch inductor fully supports custom C++ classes.

The ScalarType class can represent: (1) Standard and custom floating-point types with configurable exponent and mantissa bits (FP8, FP6, FP4 variants); (2) Integer types from 2 to 64 bits, signed or unsigned; (3) Biased types where stored_value = actual_value + bias, commonly used in quantization (e.g., GPTQ 4-bit uses bias of 8); (4) Non-standard floating-point formats with finite values only (no infinities) or custom NaN representations. The NanRepr enum specifies how NaNs are encoded: IEEE 754 standard, extended range with max/min as NaN, or no NaN support.

The module includes helper methods to compute min/max representable values accounting for bias, check type properties (is_floating_point, is_signed, has_bias), and convert to/from integer IDs for passing to PyTorch custom ops. Pre-defined types in the scalar_types class cover common quantization formats including FP8 variants (e4m3fn, e5m2), FP6 (e3m2f, e2m3f), FP4 (e2m1f), and GPTQ bias types (uint2b2, uint3b4, uint4b8, uint8b128).

=== Usage ===
Used by quantization layers and custom ops to specify data types for quantized weights and activations.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/vllm/scalar_type.py vllm/scalar_type.py]
* '''Lines:''' 1-355

=== Signature ===
<syntaxhighlight lang="python">
# NaN representation enumeration
class NanRepr(Enum):
    NONE = 0  # No NaN support
    IEEE_754 = 1  # Standard IEEE 754 NaN encoding
    EXTD_RANGE_MAX_MIN = 2  # Extended range, NaN at max/min

# Main scalar type class
@dataclass(frozen=True)
class ScalarType:
    exponent: int  # Exponent bits (0 for integer)
    mantissa: int  # Mantissa bits (or integer bits excluding sign)
    signed: bool  # Has sign bit
    bias: int  # Value bias (stored = actual + bias)
    _finite_values_only: bool = False
    nan_repr: NanRepr = NanRepr.IEEE_754

    # Properties
    @property
    def size_bits(self) -> int
    @functools.cached_property
    def id(self) -> int

    # Value bounds
    def min(self) -> int | float
    def max(self) -> int | float

    # Type checks
    def is_signed(self) -> bool
    def is_floating_point(self) -> bool
    def is_integer(self) -> bool
    def has_bias(self) -> bool
    def has_infs(self) -> bool
    def has_nans(self) -> bool
    def is_ieee_754(self) -> bool

    # Constructors
    @classmethod
    def int_(cls, size_bits: int, bias: int | None) -> "ScalarType"

    @classmethod
    def uint(cls, size_bits: int, bias: int | None) -> "ScalarType"

    @classmethod
    def float_IEEE754(cls, exponent: int, mantissa: int) -> "ScalarType"

    @classmethod
    def float_(
        cls, exponent: int, mantissa: int,
        finite_values_only: bool, nan_repr: NanRepr
    ) -> "ScalarType"

    @classmethod
    def from_id(cls, scalar_type_id: int) -> "ScalarType"

# Pre-defined scalar types
class scalar_types:
    # Integer types
    int4: ScalarType
    uint4: ScalarType
    int8: ScalarType
    uint8: ScalarType

    # FP8 types
    float8_e4m3fn: ScalarType  # NVIDIA format
    float8_e5m2: ScalarType  # IEEE-like format
    float8_e8m0fnu: ScalarType  # Exponent-only format

    # FP16 variants
    float16_e8m7: ScalarType  # bfloat16
    float16_e5m10: ScalarType  # Standard float16

    # FP6 types
    float6_e3m2f: ScalarType
    float6_e2m3f: ScalarType

    # FP4 type
    float4_e2m1f: ScalarType

    # GPTQ biased types
    uint2b2: ScalarType
    uint3b4: ScalarType
    uint4b8: ScalarType
    uint8b128: ScalarType

    # Aliases
    bfloat16: ScalarType
    float16: ScalarType
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.scalar_type import ScalarType, scalar_types, NanRepr

# Use pre-defined types
fp8_type = scalar_types.float8_e4m3fn
gptq_type = scalar_types.uint4b8

# Create custom types
custom_int = ScalarType.int_(size_bits=5, bias=16)
custom_float = ScalarType.float_IEEE754(exponent=5, mantissa=3)
</syntaxhighlight>

== I/O Contract ==

=== Key Exports ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| NanRepr || Enum || NaN representation strategies
|-
| ScalarType || Class || Scalar type descriptor with bias support
|-
| scalar_types || Class || Collection of pre-defined scalar types
|-
| scalar_types.float8_e4m3fn || ScalarType || FP8 E4M3 finite format
|-
| scalar_types.float8_e5m2 || ScalarType || FP8 E5M2 IEEE-like format
|-
| scalar_types.uint4b8 || ScalarType || 4-bit unsigned int with bias 8 (GPTQ)
|-
| scalar_types.bfloat16 || ScalarType || BFloat16 (E8M7)
|-
| scalar_types.float16 || ScalarType || IEEE Float16 (E5M10)
|}

== Usage Examples ==

<syntaxhighlight lang="python">
from vllm.scalar_type import ScalarType, scalar_types, NanRepr

# Example 1: Using pre-defined types
fp8_type = scalar_types.float8_e4m3fn
print(f"FP8 type: {fp8_type}")
print(f"Size: {fp8_type.size_bits} bits")
print(f"Min: {fp8_type.min()}, Max: {fp8_type.max()}")
print(f"Has NaNs: {fp8_type.has_nans()}")
print(f"Has Infs: {fp8_type.has_infs()}")

# Example 2: GPTQ quantization types
gptq_4bit = scalar_types.uint4b8
print(f"GPTQ 4-bit type: {gptq_4bit}")
print(f"Bias: {gptq_4bit.bias}")
print(f"Range: [{gptq_4bit.min()}, {gptq_4bit.max()}]")
# Stored 0 represents value -8
# Stored 15 represents value 7

# Example 3: Creating custom integer types
custom_int = ScalarType.int_(size_bits=5, bias=None)
print(f"5-bit signed int: {custom_int}")
print(f"Range: [{custom_int.min()}, {custom_int.max()}]")  # [-16, 15]

# With bias
biased_int = ScalarType.int_(size_bits=5, bias=10)
print(f"5-bit signed int with bias 10: {biased_int}")
print(f"Range: [{biased_int.min()}, {biased_int.max()}]")  # [-26, 5]

# Example 4: Creating custom float types
# Standard IEEE 754-like
custom_fp16 = ScalarType.float_IEEE754(exponent=5, mantissa=10)
print(f"Custom FP16: {custom_fp16}")
print(f"Is IEEE 754: {custom_fp16.is_ieee_754()}")

# Non-standard with finite values only
custom_fp8 = ScalarType.float_(
    exponent=4,
    mantissa=3,
    finite_values_only=True,
    nan_repr=NanRepr.EXTD_RANGE_MAX_MIN
)
print(f"Custom FP8: {custom_fp8}")
print(f"Has infinities: {custom_fp8.has_infs()}")  # False

# Example 5: Converting to ID for custom ops
fp8_id = scalar_types.float8_e4m3fn.id
print(f"FP8 E4M3 ID: {fp8_id}")

# Pass to custom op
import torch
# In actual usage:
# torch.ops.vllm.quantize_to_scalar_type(tensor, fp8_id)

# Recover from ID
recovered_type = ScalarType.from_id(fp8_id)
assert recovered_type == scalar_types.float8_e4m3fn

# Example 6: Type property checks
def analyze_type(dtype: ScalarType):
    print(f"Type: {dtype}")
    print(f"  Floating point: {dtype.is_floating_point()}")
    print(f"  Integer: {dtype.is_integer()}")
    print(f"  Signed: {dtype.is_signed()}")
    print(f"  Has bias: {dtype.has_bias()}")
    print(f"  Size: {dtype.size_bits} bits")
    print(f"  Range: [{dtype.min()}, {dtype.max()}]")

analyze_type(scalar_types.float8_e4m3fn)
analyze_type(scalar_types.uint4b8)
analyze_type(scalar_types.bfloat16)

# Example 7: Using in quantization
class QuantizedLinear:
    def __init__(self, weight_dtype: ScalarType):
        self.weight_dtype = weight_dtype
        self.weight_dtype_id = weight_dtype.id

    def forward(self, x):
        # Use weight_dtype_id in custom ops
        return torch.ops.vllm.quantized_linear(
            x, self.weights, self.weight_dtype_id
        )

# Create layer with FP8 weights
layer = QuantizedLinear(scalar_types.float8_e4m3fn)
</syntaxhighlight>

== Related Pages ==
* [[used_by::Module:vllm-project_vllm_Quantization_Layers]]
* [[implements::Concept:Sub_Byte_Data_Types]]
* [[synced_with::CPP:vllm_core_scalar_type]]
* [[related::Module:vllm-project_vllm_Custom_Ops]]
