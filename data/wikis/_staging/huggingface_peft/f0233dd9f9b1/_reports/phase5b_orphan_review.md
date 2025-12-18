# Phase 6b: Orphan Review Report

## Summary
- MANUAL_REVIEW files evaluated: 82
- Approved: 73
- Rejected: 9

## Decisions

| File | Decision | Reasoning |
|------|----------|-----------|
| `method_comparison/processing.py` | REJECTED | Internal benchmark tool, not user-facing |
| `method_comparison/sanitizer.py` | REJECTED | Internal benchmark utility |
| `setup.py` | REJECTED | Build metadata, not library code |
| `src/peft/functional.py` | REJECTED | Small re-export module, no new logic |
| `src/peft/tuners/_buffer_dict.py` | REJECTED | Private helper (underscore prefix) |
| `src/peft/tuners/adalora/bnb.py` | APPROVED | Public quantized layer classes |
| `src/peft/tuners/adalora/config.py` | APPROVED | Public config class AdaLoraConfig |
| `src/peft/tuners/adalora/gptq.py` | APPROVED | Public LoraGPTQQuantizer layer |
| `src/peft/tuners/adaption_prompt/config.py` | APPROVED | Public AdaptionPromptConfig class |
| `src/peft/tuners/adaption_prompt/layer.py` | APPROVED | Public AdaptedAttention layer |
| `src/peft/tuners/adaption_prompt/model.py` | APPROVED | Public AdaptionPromptModel class |
| `src/peft/tuners/adaption_prompt/utils.py` | REJECTED | Internal utility functions |
| `src/peft/tuners/boft/config.py` | APPROVED | Public BOFTConfig class |
| `src/peft/tuners/boft/model.py` | APPROVED | Public BOFTModel class |
| `src/peft/tuners/bone/config.py` | APPROVED | Public BoneConfig class |
| `src/peft/tuners/bone/model.py` | APPROVED | Public BoneModel class |
| `src/peft/tuners/c3a/config.py` | APPROVED | Public C3AConfig class |
| `src/peft/tuners/c3a/layer.py` | APPROVED | Public C3A layer classes |
| `src/peft/tuners/c3a/model.py` | APPROVED | Public C3AModel class |
| `src/peft/tuners/c3a/utils.py` | REJECTED | Internal utility, small helper |
| `src/peft/tuners/cpt/config.py` | APPROVED | Public CPTConfig class |
| `src/peft/tuners/fourierft/config.py` | APPROVED | Public FourierFTConfig class |
| `src/peft/tuners/fourierft/layer.py` | APPROVED | Public FourierFT layer classes |
| `src/peft/tuners/fourierft/model.py` | APPROVED | Public FourierFTModel class |
| `src/peft/tuners/gralora/config.py` | APPROVED | Public GraLoRAConfig class |
| `src/peft/tuners/gralora/model.py` | APPROVED | Public GraLoRAModel class |
| `src/peft/tuners/hra/config.py` | APPROVED | Public HRAConfig class |
| `src/peft/tuners/hra/model.py` | APPROVED | Public HRAModel class |
| `src/peft/tuners/ia3/bnb.py` | APPROVED | Public quantized IA3 layers |
| `src/peft/tuners/ia3/config.py` | APPROVED | Public IA3Config class |
| `src/peft/tuners/ln_tuning/config.py` | APPROVED | Public LNTuningConfig class |
| `src/peft/tuners/ln_tuning/layer.py` | APPROVED | Public LNTuning layer class |
| `src/peft/tuners/ln_tuning/model.py` | APPROVED | Public LNTuningModel class |
| `src/peft/tuners/loha/config.py` | APPROVED | Public LoHaConfig class |
| `src/peft/tuners/loha/model.py` | APPROVED | Public LoHaModel class |
| `src/peft/tuners/lokr/config.py` | APPROVED | Public LoKrConfig class |
| `src/peft/tuners/lokr/model.py` | APPROVED | Public LoKrModel class |
| `src/peft/tuners/lora/aqlm.py` | APPROVED | Public LoRA AQLM quantized layer |
| `src/peft/tuners/lora/inc.py` | APPROVED | Public Intel FP8 layer class |
| `src/peft/tuners/lora/torchao.py` | APPROVED | Public TorchAO quantized layer |
| `src/peft/tuners/lycoris_utils.py` | APPROVED | Public LyCORIS tuner base class |
| `src/peft/tuners/miss/config.py` | APPROVED | Public MissConfig class |
| `src/peft/tuners/miss/model.py` | APPROVED | Public MissModel class |
| `src/peft/tuners/multitask_prompt_tuning/config.py` | APPROVED | Public MPT config class |
| `src/peft/tuners/multitask_prompt_tuning/model.py` | APPROVED | Public MPT embedding model |
| `src/peft/tuners/oft/aqlm.py` | APPROVED | Public OFT AQLM quantized layer |
| `src/peft/tuners/oft/awq.py` | APPROVED | Public OFT AWQ quantized layer |
| `src/peft/tuners/oft/config.py` | APPROVED | Public OFTConfig class |
| `src/peft/tuners/oft/eetq.py` | APPROVED | Public OFT EETQ quantized layer |
| `src/peft/tuners/oft/gptq.py` | APPROVED | Public OFT GPTQ quantized layer |
| `src/peft/tuners/oft/hqq.py` | APPROVED | Public OFT HQQ quantized layer |
| `src/peft/tuners/oft/inc.py` | APPROVED | Public OFT Intel FP8 layer |
| `src/peft/tuners/oft/model.py` | APPROVED | Public OFTModel class |
| `src/peft/tuners/p_tuning/config.py` | APPROVED | Public PromptEncoderConfig class |
| `src/peft/tuners/p_tuning/model.py` | APPROVED | Public PromptEncoder model |
| `src/peft/tuners/poly/config.py` | APPROVED | Public PolyConfig class |
| `src/peft/tuners/poly/layer.py` | APPROVED | Public Poly layer classes |
| `src/peft/tuners/poly/model.py` | APPROVED | Public PolyModel class |
| `src/peft/tuners/poly/router.py` | APPROVED | Public router component |
| `src/peft/tuners/prefix_tuning/config.py` | APPROVED | Public PrefixTuningConfig class |
| `src/peft/tuners/prefix_tuning/model.py` | APPROVED | Public PrefixEncoder model |
| `src/peft/tuners/prompt_tuning/config.py` | APPROVED | Public PromptTuningConfig class |
| `src/peft/tuners/prompt_tuning/model.py` | APPROVED | Public PromptEmbedding model |
| `src/peft/tuners/randlora/config.py` | APPROVED | Public RandLoraConfig class |
| `src/peft/tuners/road/config.py` | APPROVED | Public RoAdConfig class |
| `src/peft/tuners/road/model.py` | APPROVED | Public RoAdModel class |
| `src/peft/tuners/shira/config.py` | APPROVED | Public ShiraConfig class |
| `src/peft/tuners/shira/layer.py` | APPROVED | Public ShiraLayer classes |
| `src/peft/tuners/shira/mask_functions.py` | REJECTED | Internal utility functions |
| `src/peft/tuners/shira/model.py` | APPROVED | Public ShiraModel class |
| `src/peft/tuners/trainable_tokens/config.py` | APPROVED | Public TrainableTokensConfig |
| `src/peft/tuners/trainable_tokens/layer.py` | APPROVED | Public TrainableTokens layer |
| `src/peft/tuners/trainable_tokens/model.py` | APPROVED | Public TrainableTokensModel |
| `src/peft/tuners/vblora/config.py` | APPROVED | Public VBLoRAConfig class |
| `src/peft/tuners/vblora/layer.py` | APPROVED | Public VBLoRALayer classes |
| `src/peft/tuners/vblora/model.py` | APPROVED | Public VBLoRAModel class |
| `src/peft/tuners/vera/config.py` | APPROVED | Public VeraConfig class |
| `src/peft/tuners/vera/layer.py` | APPROVED | Public VeraLayer classes |
| `src/peft/tuners/vera/model.py` | APPROVED | Public VeraModel class |
| `src/peft/tuners/xlora/classifier.py` | APPROVED | Public XLoraClassifier class |
| `src/peft/tuners/xlora/config.py` | APPROVED | Public XLoraConfig class |
| `src/peft/tuners/xlora/layer.py` | APPROVED | Public XLora layer classes |

## Notes

### Patterns Observed
- **PEFT tuner modules** have a consistent structure: config.py, model.py, layer.py for each tuner type
- Most tuner files contain public APIs (config classes, model classes, layer classes) that are user-facing
- Utility files (`utils.py`, `mask_functions.py`) are typically internal and not user-facing

### Files That Were Borderline
- `src/peft/tuners/poly/router.py` (81 lines) - Approved as it contains a public router class used by the Poly architecture
- `src/peft/tuners/ln_tuning/layer.py` (68 lines) - Approved as it provides the public layer class for LayerNorm tuning
- `src/peft/tuners/trainable_tokens/layer.py` (65 lines) - Approved as it's a core user-facing layer class

### Rejection Criteria Applied
1. **Internal benchmarks**: `method_comparison/` files are developer tools, not library API
2. **Build infrastructure**: `setup.py` is package metadata, not documentation-worthy
3. **Private helpers**: Files with underscore prefix (`_buffer_dict.py`) are internal
4. **Small utilities**: Internal utility files with helper functions only

### Approval Rate
- 73 out of 82 files (89%) were approved
- High approval rate reflects that most tuner modules contain public, user-facing APIs
