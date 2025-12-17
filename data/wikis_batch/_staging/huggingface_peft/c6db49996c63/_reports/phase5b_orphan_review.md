# Phase 6b: Orphan Review Report

## Summary
- MANUAL_REVIEW files evaluated: 77
- Approved: 64
- Rejected: 13

## Decisions

| File | Decision | Reasoning |
|------|----------|-----------|
| method_comparison/processing.py | REJECTED | Demo app helper, not core library |
| method_comparison/sanitizer.py | REJECTED | Demo app helper, not core library |
| setup.py | REJECTED | Build config, not PEFT algorithm |
| src/peft/functional.py | APPROVED | Public API for integrations |
| src/peft/helpers.py | APPROVED | User-facing helper functions |
| src/peft/import_utils.py | REJECTED | Internal utility, no public API |
| src/peft/optimizers/lorafa.py | APPROVED | Public optimizer class |
| src/peft/optimizers/loraplus.py | APPROVED | Public optimizer function |
| src/peft/tuners/__init__.py | REJECTED | Just re-exports, no new code |
| src/peft/tuners/_buffer_dict.py | REJECTED | Internal helper, underscore prefix |
| src/peft/tuners/adalora/bnb.py | APPROVED | Implements quantized AdaLoRA layers |
| src/peft/tuners/adalora/config.py | APPROVED | User-facing config class |
| src/peft/tuners/adalora/gptq.py | APPROVED | Implements GPTQ quantization |
| src/peft/tuners/adaption_prompt/config.py | APPROVED | User-facing config class |
| src/peft/tuners/adaption_prompt/layer.py | APPROVED | Public adapter layer classes |
| src/peft/tuners/adaption_prompt/model.py | APPROVED | Public model class |
| src/peft/tuners/adaption_prompt/utils.py | REJECTED | Internal helpers, no public API |
| src/peft/tuners/boft/config.py | APPROVED | User-facing config class |
| src/peft/tuners/boft/model.py | APPROVED | Public model class |
| src/peft/tuners/bone/config.py | APPROVED | User-facing config class |
| src/peft/tuners/bone/model.py | APPROVED | Public model class |
| src/peft/tuners/c3a/config.py | APPROVED | User-facing config class |
| src/peft/tuners/c3a/layer.py | APPROVED | Public layer classes |
| src/peft/tuners/c3a/model.py | APPROVED | Public model class |
| src/peft/tuners/c3a/utils.py | REJECTED | Internal FFT helpers |
| src/peft/tuners/cpt/config.py | APPROVED | User-facing config class |
| src/peft/tuners/fourierft/config.py | APPROVED | User-facing config class |
| src/peft/tuners/fourierft/layer.py | APPROVED | Public layer classes |
| src/peft/tuners/fourierft/model.py | APPROVED | Public model class |
| src/peft/tuners/gralora/config.py | APPROVED | User-facing config class |
| src/peft/tuners/gralora/model.py | APPROVED | Public model class |
| src/peft/tuners/hra/config.py | APPROVED | User-facing config class |
| src/peft/tuners/hra/model.py | APPROVED | Public model class |
| src/peft/tuners/ia3/bnb.py | APPROVED | Implements quantized IA3 layers |
| src/peft/tuners/ia3/config.py | APPROVED | User-facing config class |
| src/peft/tuners/loha/config.py | APPROVED | User-facing config class |
| src/peft/tuners/loha/model.py | APPROVED | Public model class |
| src/peft/tuners/lora/aqlm.py | APPROVED | Implements AQLM quantization |
| src/peft/tuners/lora/awq.py | APPROVED | Implements AWQ quantization |
| src/peft/tuners/lora/dora.py | APPROVED | Public DoRA layer classes |
| src/peft/tuners/lora/eetq.py | APPROVED | Implements EETQ quantization |
| src/peft/tuners/lora/gptq.py | APPROVED | Implements GPTQ quantization |
| src/peft/tuners/lora/hqq.py | APPROVED | Implements HQQ quantization |
| src/peft/tuners/lora/inc.py | APPROVED | Implements INC quantization |
| src/peft/tuners/lora/torchao.py | APPROVED | Implements TorchAO quantization |
| src/peft/tuners/lycoris_utils.py | APPROVED | Public base classes for LyCORIS |
| src/peft/tuners/miss/config.py | APPROVED | User-facing config class |
| src/peft/tuners/miss/model.py | APPROVED | Public model class |
| src/peft/tuners/multitask_prompt_tuning/config.py | APPROVED | User-facing config class |
| src/peft/tuners/multitask_prompt_tuning/model.py | APPROVED | Public model class |
| src/peft/tuners/oft/aqlm.py | APPROVED | Implements AQLM for OFT |
| src/peft/tuners/oft/awq.py | APPROVED | Implements AWQ for OFT |
| src/peft/tuners/oft/config.py | APPROVED | User-facing config class |
| src/peft/tuners/oft/eetq.py | APPROVED | Implements EETQ for OFT |
| src/peft/tuners/oft/gptq.py | APPROVED | Implements GPTQ for OFT |
| src/peft/tuners/oft/hqq.py | APPROVED | Implements HQQ for OFT |
| src/peft/tuners/oft/inc.py | APPROVED | Implements INC for OFT |
| src/peft/tuners/oft/model.py | APPROVED | Public model class |
| src/peft/tuners/poly/config.py | APPROVED | User-facing config class |
| src/peft/tuners/poly/layer.py | APPROVED | Public layer classes |
| src/peft/tuners/poly/model.py | APPROVED | Public model class |
| src/peft/tuners/poly/router.py | REJECTED | Internal routing helper |
| src/peft/tuners/randlora/config.py | APPROVED | User-facing config class |
| src/peft/tuners/road/config.py | APPROVED | User-facing config class |
| src/peft/tuners/road/model.py | APPROVED | Public model class |
| src/peft/tuners/shira/config.py | APPROVED | User-facing config class |
| src/peft/tuners/shira/layer.py | APPROVED | Public layer classes |
| src/peft/tuners/shira/mask_functions.py | REJECTED | Internal mask helpers |
| src/peft/tuners/shira/model.py | APPROVED | Public model class |
| src/peft/tuners/vblora/config.py | APPROVED | User-facing config class |
| src/peft/tuners/vblora/layer.py | APPROVED | Public layer classes |
| src/peft/tuners/vblora/model.py | APPROVED | Public model class |
| src/peft/tuners/vera/config.py | APPROVED | User-facing config class |
| src/peft/tuners/vera/layer.py | APPROVED | Public layer classes |
| src/peft/tuners/vera/model.py | APPROVED | Public model class |
| src/peft/utils/__init__.py | REJECTED | Just re-exports, no new code |
| src/peft/utils/peft_types.py | APPROVED | Public PeftType/TaskType enums |

## Notes

### Approval Patterns Identified
1. **Config classes** (e.g., `*Config`) - All user-facing configuration dataclasses were approved since users import and configure these directly.
2. **Model classes** (e.g., `*Model`) - All public model orchestration classes were approved as they coordinate adapter injection.
3. **Layer classes** (e.g., `*Layer`, `*Linear`) - Public adapter layer implementations were approved as they contain documented algorithm implementations.
4. **Quantization adapters** (e.g., `bnb.py`, `gptq.py`, `awq.py`) - These implement distinct quantization-specific adapter variants and were approved.
5. **Optimizer implementations** - Public optimizer classes/functions were approved.
6. **Core utilities** (e.g., `peft_types.py`, `helpers.py`, `functional.py`) - User-facing utilities with public API were approved.

### Rejection Patterns Identified
1. **Internal helpers** - Files with only `_private` functions or that serve as implementation details were rejected.
2. **Re-export modules** - `__init__.py` files that only aggregate exports without implementing new functionality were rejected.
3. **Build configuration** - `setup.py` and similar build artifacts were rejected.
4. **Demo/application code** - `method_comparison/*.py` files that are demo apps rather than library code were rejected.
5. **Import utilities** - Internal dependency detection code was rejected.

### Borderline Cases
- **Quantization files**: While these could be considered internal implementation details, they were approved because they implement distinct quantization algorithms and have public classes that users may need to understand for debugging or customization.
- **lycoris_utils.py**: Contains public base classes (`LycorisConfig`, `LycorisLayer`, `LycorisTuner`) that are documented and serve as extension points for the LyCORIS adapter family.

## Evaluation Methodology

Each file was evaluated against three criteria:

1. **Public API presence**: Does the file export classes or functions without underscore prefix?
2. **User-facing nature**: Would a PEFT user directly import or interact with this code?
3. **Algorithm implementation**: Does the code implement a distinct PEFT algorithm or variant?

Files meeting at least two of these criteria were approved. Files that are purely internal glue code, build artifacts, or application-specific code were rejected.
