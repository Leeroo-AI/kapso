# Phase 3: Enrichment Report

## Summary

This phase enriched the Unsloth wiki by mining environment constraints and heuristics (tribal knowledge) from the source code. The extraction focused on GPU requirements, dependency version constraints, and optimization tips found in code comments and warnings.

---

## Environments Created

| Environment | Required By | Description |
|-------------|-------------|-------------|
| unslothai_unsloth_CUDA | All implementations (18 pages) | GPU acceleration environment supporting NVIDIA CUDA, AMD ROCm (HIP), and Intel XPU |

### Environment Details: unslothai_unsloth_CUDA

**Key Requirements Extracted:**
- Python 3.9 - 3.13
- PyTorch >= 2.2.0 (>= 2.6.0 for Intel XPU)
- Triton >= 3.0.0
- transformers >= 4.51.3
- bitsandbytes >= 0.43.3 (>= 0.48.3 for AMD)
- peft >= 0.7.1
- trl >= 0.18.2
- NVIDIA Ampere+ GPU (SM 8.0+) for BFloat16 and Flash Attention

**Code Evidence Locations:**
- `device_type.py:37-59` - GPU detection logic
- `kernels/utils.py:41-44` - Intel XPU version check
- `kernels/utils.py:62-76` - Triton version handling
- `_utils.py:659-664` - BFloat16 support detection
- `__init__.py:198-260` - CUDA library linking

---

## Heuristics Created

| Heuristic | Applies To | Summary |
|-----------|------------|---------|
| unslothai_unsloth_Import_Order | import_unsloth, QLoRA_Finetuning, GRPO_RL | **Critical:** Import `unsloth` BEFORE `transformers/trl/peft` |
| unslothai_unsloth_Gradient_Checkpointing | get_peft_model, from_pretrained, QLoRA, GRPO | Use `"unsloth"` mode for 30% VRAM reduction |
| unslothai_unsloth_Sample_Packing | SFTTrainer_usage, trainer_train, QLoRA | Enable `packing=True` for >2x faster training |
| unslothai_unsloth_AMD_GPU_Limitations | from_pretrained, Model_Loading | AMD ROCm blocksize (128 vs 64) limits pre-quantized models |
| unslothai_unsloth_Flash_Attention_Gemma2 | from_pretrained, Model_Loading | Install `flash-attn>=2.6.3` for Gemma 2 softcapping |

### Heuristic Details

#### 1. Import_Order
- **Source:** `__init__.py:24-57`
- **Rule:** Import `unsloth` as first import, before HuggingFace libraries
- **Consequence if ignored:** 2-5x slower training, potential OOM errors

#### 2. Gradient_Checkpointing
- **Source:** `_utils.py` (smart checkpointing), `llama.py:L2578-3100`
- **Rule:** Set `use_gradient_checkpointing="unsloth"` (not `True`)
- **Benefit:** ~30% VRAM reduction with ~5-10% speed overhead

#### 3. Sample_Packing
- **Source:** `trainer.py:56-59, 394-396`
- **Rule:** Enable `packing=True` in SFTConfig
- **Benefit:** >2x training speedup by eliminating padding waste
- **Blocklist:** gemma2, gpt_oss (incompatible architectures)

#### 4. AMD_GPU_Limitations
- **Source:** `device_type.py:81-98`, `loader.py`
- **Rule:** Use base models, not pre-quantized `-bnb-4bit` models on AMD
- **Reason:** Blocksize mismatch (AMD=128, NVIDIA=64)

#### 5. Flash_Attention_Gemma2
- **Source:** `loader.py:442-454`, `_utils.py:676-681`
- **Rule:** Install `flash-attn>=2.6.3` for Gemma 2 models
- **Benefit:** Native softcapping support, 20-30% faster attention

---

## Links Added

### Environment Links
- Environment links added to Implementation Index: **18 implementations**
- All `⬜Env:unslothai_unsloth_CUDA` changed to `✅Env:unslothai_unsloth_CUDA`

### Heuristic Links
- `unslothai_unsloth_Import_Order`: 1 implementation
- `unslothai_unsloth_Gradient_Checkpointing`: 2 implementations
- `unslothai_unsloth_Sample_Packing`: 2 implementations
- `unslothai_unsloth_AMD_GPU_Limitations`: 1 implementation
- `unslothai_unsloth_Flash_Attention_Gemma2`: 1 implementation

**Total heuristic links added:** 7 to implementations, plus workflow/principle cross-references

---

## Index Updates

| Index | Updates Made |
|-------|--------------|
| `_EnvironmentIndex.md` | Added 1 environment entry with 11+ implementation connections |
| `_HeuristicIndex.md` | Added 5 heuristic entries with implementation/workflow/principle connections |
| `_ImplementationIndex.md` | Changed all `⬜Env:` to `✅Env:`, added `✅Heuristic:` references |

---

## Notes for Audit Phase

### Verification Needed
1. **Environment page completeness:** The CUDA environment covers all three platforms (NVIDIA/AMD/Intel) - verify code evidence is accurate
2. **Heuristic accuracy:** Sample packing blocklist should be verified against latest TRL/Unsloth versions
3. **Version constraints:** pyproject.toml versions should be cross-checked with code evidence

### Potential Broken Links
- None identified - all created pages have valid cross-references

### Pages That May Need Review
1. `unslothai_unsloth_AMD_GPU_Limitations` - AMD support is evolving rapidly
2. `unslothai_unsloth_Flash_Attention_Gemma2` - Flash Attention versions change frequently
3. Environment page may need updates as Intel XPU support matures

### Additional Heuristics Found (Not Implemented)
The following patterns were observed but not extracted as full heuristic pages:
- `torch.compile` optimization flags (`UNSLOTH_COMPILE_DEBUG`, `UNSLOTH_COMPILE_MAXIMUM`)
- Multi-GPU limitations (currently single-GPU only)
- Colab/Kaggle disk space management
- xformers version compatibility with torch versions

These could be added in future enrichment passes if needed.

---

## File Inventory

### Created Files

| File | Type | Path |
|------|------|------|
| Environment: CUDA | Environment | `environments/unslothai_unsloth_CUDA.md` |
| Heuristic: Import_Order | Heuristic | `heuristics/unslothai_unsloth_Import_Order.md` |
| Heuristic: Gradient_Checkpointing | Heuristic | `heuristics/unslothai_unsloth_Gradient_Checkpointing.md` |
| Heuristic: Sample_Packing | Heuristic | `heuristics/unslothai_unsloth_Sample_Packing.md` |
| Heuristic: AMD_GPU_Limitations | Heuristic | `heuristics/unslothai_unsloth_AMD_GPU_Limitations.md` |
| Heuristic: Flash_Attention_Gemma2 | Heuristic | `heuristics/unslothai_unsloth_Flash_Attention_Gemma2.md` |

### Updated Files

| File | Changes |
|------|---------|
| `_EnvironmentIndex.md` | Added CUDA environment entry |
| `_HeuristicIndex.md` | Added 5 heuristic entries |
| `_ImplementationIndex.md` | Updated all Env/Heuristic connections to ✅ |

---

*Generated: 2025-12-17*
