# File: `src/peft/tuners/oft/eetq.py`

**Category:** Quantization Backend

| Property | Value |
|----------|-------|
| Lines | 116 |
| Classes | `EetqOFTLinear` |
| Functions | `dispatch_eetq` |
| Imports | peft, torch, typing |

## Understanding

**Status:** Fully explored

**Purpose:** Implements OFT for EETQ (Efficient Integer Quantization) Linear layers.

**Mechanism:**

### EetqOFTLinear
Applies OFT to EETQ EetqLinear layers:

**Forward**: OFT transformation before EETQ layer
**Merge/Unmerge**: Raises AttributeError (not supported)

### dispatch_eetq()
- Checks for eetq.EetqLinear base layer
- Creates EetqOFTLinear wrapper

**Significance:** EETQ provides efficient integer quantization. OFT support enables fine-tuning with explicit merge rejection (cleaner than silent failures).

## Key Features

- **EETQ Support**: Works with eetq library
- **Explicit No-Merge**: Raises clear error messages
- **Runtime Transform**: Applies OFT at forward time
