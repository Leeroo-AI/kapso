# Phase 4: Enrichment Report

## Summary

This phase mined the Unsloth codebase for Environment constraints and Heuristics (tribal knowledge), creating 11 new wiki pages and updating all indexes with bi-directional links.

---

## Environments Created

| Environment | Required By | Key Requirements |
|-------------|-------------|------------------|
| unslothai_unsloth_CUDA_Compute | FastLanguageModel, get_peft_model | NVIDIA/AMD GPU, CUDA 11.8+, PyTorch 2.0+, bitsandbytes |
| unslothai_unsloth_llama_cpp | save_pretrained_gguf | llama.cpp binaries, cmake, gcc/clang |
| unslothai_unsloth_Storage | save_pretrained_merged | Disk space (3x model size), 20GB limit on Kaggle |
| unslothai_unsloth_vLLM | FastLanguageModel (fast_inference), GRPO_Training | vLLM package, Linux, NVIDIA GPU |

### Environment Evidence from Code

**CUDA Detection (`device_type.py:37-59`):**
- `torch.cuda.is_available()` for NVIDIA
- `torch.xpu.is_available()` for Intel
- HIP detection for AMD ROCm

**AMD Compatibility (`device_type.py:81-98`):**
- Blocksize check for pre-quantized models (64 vs 128)
- bitsandbytes version gating (>0.48.2.dev0)

**Kaggle Limitations (`save.py:1979-1987`):**
- 20GB disk limit detection
- Recommendation to use /tmp directory

---

## Heuristics Created

| Heuristic | Applies To | Key Advice |
|-----------|------------|------------|
| unslothai_unsloth_LoRA_Rank_Selection | get_peft_model, QLoRA_Finetuning, LoRA_Configuration | r=16 default; r=8-16 simple, r=32-64 complex |
| unslothai_unsloth_Quantization_Method_Selection | save_pretrained_gguf, GGUF_Export, GGUF_Conversion | q4_k_m recommended; presets: not_quantized→f16, fast_quantized→q8_0, quantized→q4_k_m |
| unslothai_unsloth_Gradient_Checkpointing | FastLanguageModel, QLoRA_Finetuning, Environment_Setup | use_gradient_checkpointing="unsloth" for 30% VRAM savings |
| unslothai_unsloth_Memory_Management | save_pretrained_merged, FastLanguageModel, QLoRA_Finetuning | maximum_memory_usage=0.9; layer-by-layer processing; CPU offload for embeddings |
| unslothai_unsloth_LoRA_Dropout_Bias | get_peft_model, LoRA_Configuration | lora_dropout=0, bias="none" for fast patching |
| unslothai_unsloth_Sample_Packing | QLoRA_Finetuning, SFT_Training | packing=True for >2x speedup; blocklist: gemma2, gpt_oss |
| unslothai_unsloth_RL_Learning_Rate | GRPO_Training | 5e-6 for RL vs 2e-4 for SFT; prevents policy collapse |

### Heuristic Evidence from Code

**LoRA Fast Patching Warnings (`llama.py:2767-2777`):**
```python
if lora_dropout != 0:
    logger.warning_once(
        f"Unsloth: Dropout = 0 is supported for fast patching..."
    )
```

**Sample Packing Blocklist (`trainer.py:56-59`):**
```python
PADDING_FREE_BLOCKLIST = {
    "gemma2",  # slow_attention_softcapping issues
    "gpt_oss",  # Flex Attention incompatibility
}
```

**Quantization Presets (`save.py:1954-1964`):**
```python
if quant_method == "not_quantized":
    quant_method = "f16"
elif quant_method == "fast_quantized":
    quant_method = "q8_0"
elif quant_method == "quantized":
    quant_method = "q4_k_m"
```

---

## Links Added

### Environment Links
- **_ImplementationIndex.md**: Changed 4 `⬜Env:` to `✅Env:`, added 4 new `✅Heuristic:` entries
- **_WorkflowIndex.md**: Added 1 `✅Env:` and 5 `✅Heuristic:` entries
- **_PrincipleIndex.md**: Added 7 `✅Heuristic:` entries

### Summary Statistics
- Environment links added: 5
- Heuristic links added: 16
- Total new pages: 11 (4 Environments + 7 Heuristics)

---

## Index Updates

| Index | Updates |
|-------|---------|
| _EnvironmentIndex.md | Created with 4 environment entries |
| _HeuristicIndex.md | Created with 7 heuristic entries |
| _ImplementationIndex.md | Updated all environment references to ✅, added heuristic links |
| _WorkflowIndex.md | Added environment and heuristic links |
| _PrincipleIndex.md | Added heuristic links |

---

## Notes for Audit Phase

### Potential Issues
1. **vLLM Workflow Reference**: `unslothai_unsloth_GRPO_Training` workflow exists and is linked
2. **save_pretrained_merged Implementation**: References this implementation which exists
3. **save_pretrained_gguf Implementation**: References this implementation which exists

### Cross-Reference Verification Needed
- Verify all `✅Impl:`, `✅Workflow:`, `✅Principle:` targets in Heuristic pages exist
- Verify bi-directional links are consistent (if A→B, then B should reference A)

### Files with Most Tribal Knowledge
1. `unsloth/save.py` - Quantization methods, memory management, Kaggle limits
2. `unsloth/models/llama.py` - LoRA configuration warnings, embedding handling
3. `unsloth/trainer.py` - Sample packing blocklist, padding-free batching
4. `unsloth/device_type.py` - Hardware detection, AMD compatibility

### Suggested Future Heuristics
- **Tesla T4 float32 workaround** (`llama.py:2716-2717`) - T4 must use float32 not float16 for embeddings
- **Transformers version requirements** (`qwen3.py:41-45`, `falcon_h1.py:46-50`) - Model-specific minimum versions
- **vLLM LoRA limitations** (`vision.py:384-386, 985-997`) - Vision layers not supported with fast_inference
