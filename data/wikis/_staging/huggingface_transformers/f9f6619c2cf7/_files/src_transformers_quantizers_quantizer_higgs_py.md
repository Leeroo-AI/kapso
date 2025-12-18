# File: `src/transformers/quantizers/quantizer_higgs.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 176 |
| Classes | `HiggsHfQuantizer` |
| Imports | base, quantizers_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements HIGGS (Hadamard-based Int Group Quantization Scheme) method using FLUTE kernels for low-bit quantization with Hadamard transformations.

**Mechanism:** Extends HfQuantizer requiring GPU-only operation with FLUTE kernel and fast_hadamard_transform libraries. Replaces Linear layers with HiggsLinear modules before loading. After weight loading, performs critical post-processing: creates reusable FLUTE workspaces per device, loads tune_metadata for each module, and repacks weights for the target GPU's streaming multiprocessor configuration via maybe_tune_and_repack. Enforces float16/bfloat16 dtypes and strict CUDA device mapping (no CPU/disk).

**Significance:** Research-oriented quantizer implementing structured quantization with Hadamard rotations for improved accuracy at low bitwidths. Non-trainable but serializable with dequantization support. Unique in requiring device-specific weight repacking based on SM count, optimizing for specific GPU architectures.
