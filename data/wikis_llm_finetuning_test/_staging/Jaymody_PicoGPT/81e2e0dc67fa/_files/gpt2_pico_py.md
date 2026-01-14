# File: `gpt2_pico.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 62 |
| Functions | `gelu`, `softmax`, `layer_norm`, `linear`, `ffn`, `attention`, `mha`, `transformer_block`, `... +3 more` |
| Imports | numpy |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Minimalist "pico" version of GPT-2 - same logic in ~60 lines.

**Mechanism:** Implements identical GPT-2 functionality as `gpt2.py` but in a highly condensed format (~60 lines vs ~120 lines). Key differences: removes all comments, uses single-line function definitions where possible (e.g., `ffn` is one line), and removes intermediate variable assignments. The transformer architecture is identical: token embeddings → positional embeddings → N transformer blocks (with multi-head attention + FFN) → layer norm → logit projection. Same `main()` CLI interface.

**Significance:** Demonstrates that a working GPT-2 can fit in ~60 lines of NumPy code. Serves as a "code golf" showcase of minimal implementation while maintaining full functionality. Both versions produce identical outputs given the same inputs.
