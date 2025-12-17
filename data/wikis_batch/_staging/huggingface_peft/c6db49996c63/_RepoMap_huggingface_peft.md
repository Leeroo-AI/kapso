# Repository Map: huggingface_peft

> **Compact index** of repository files.
> Each file has a detail page in `_files/` with Understanding to fill.
> Mark files as ✅ explored in the table below as you complete them.

| Property | Value |
|----------|-------|
| Repository | https://github.com/huggingface/peft |
| Branch | main |
| Generated | 2025-12-17 18:59 |
| Python Files | 200 |
| Total Lines | 78,061 |
| Explored | 200/200 |

## Structure

📦 **Packages:** method_comparison
📝 **Examples:** examples, scripts
🧪 **Tests:** tests

📖 README: `README.md`
⚙️ Setup: `pyproject.toml`

---

## 📦 Package Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `method_comparison/__init__.py` | 0 | Empty package initialization | — | [→](./_files/method_comparison___init___py.md) |
| ✅ | `method_comparison/app.py` | 385 | Gradio visualization with Pareto | — | [→](./_files/method_comparison_app_py.md) |
| ✅ | `method_comparison/processing.py` | 150 | Data pipeline for results | — | [→](./_files/method_comparison_processing_py.md) |
| ✅ | `method_comparison/sanitizer.py` | 100 | Secure DataFrame filtering | — | [→](./_files/method_comparison_sanitizer_py.md) |
| ✅ | `method_comparison/test_sanitizer.py` | 38 | Security and correctness tests | — | [→](./_files/method_comparison_test_sanitizer_py.md) |

## 📝 Example Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `scripts/ci_clean_cache.py` | 67 | HuggingFace cache cleanup | — | [→](./_files/scripts_ci_clean_cache_py.md) |
| ✅ | `scripts/convert-bone-to-miss.py` | 70 | Bone-to-MiSS checkpoint converter | — | [→](./_files/scripts_convert-bone-to-miss_py.md) |
| ✅ | `scripts/launch_notebook_mp.py` | 47 | Multiprocessing CUDA regression | — | [→](./_files/scripts_launch_notebook_mp_py.md) |
| ✅ | `scripts/log_reports.py` | 144 | CI test results Slack reporter | — | [→](./_files/scripts_log_reports_py.md) |
| ✅ | `scripts/stale.py` | 65 | Automated GitHub issue closer | — | [→](./_files/scripts_stale_py.md) |
| ✅ | `scripts/train_memory.py` | 276 | Memory profiling and benchmarking | — | [→](./_files/scripts_train_memory_py.md) |

## 🧪 Test Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `tests/__init__.py` | 0 | Empty package initialization | — | [→](./_files/tests___init___py.md) |
| ✅ | `tests/conftest.py` | 86 | Pytest configuration and hooks | — | [→](./_files/tests_conftest_py.md) |
| ✅ | `tests/test_adaption_prompt.py` | 416 | Tests AdaptionPrompt functionality | — | [→](./_files/tests_test_adaption_prompt_py.md) |
| ✅ | `tests/test_arrow.py` | 509 | Tests Arrow routing mechanism | — | [→](./_files/tests_test_arrow_py.md) |
| ✅ | `tests/test_auto.py` | 225 | Tests AutoPeftModel loading | — | [→](./_files/tests_test_auto_py.md) |
| ✅ | `tests/test_boft.py` | 84 | Tests BOFT checkpoint format | — | [→](./_files/tests_test_boft_py.md) |
| ✅ | `tests/test_bufferdict.py` | 48 | Tests BufferDict utility class | — | [→](./_files/tests_test_bufferdict_py.md) |
| ✅ | `tests/test_common_gpu.py` | 2185 | Tests GPU ops and quantization | — | [→](./_files/tests_test_common_gpu_py.md) |
| ✅ | `tests/test_config.py` | 599 | Tests all configuration classes | — | [→](./_files/tests_test_config_py.md) |
| ✅ | `tests/test_cpt.py` | 305 | Tests Context-aware Prompt Tuning | — | [→](./_files/tests_test_cpt_py.md) |
| ✅ | `tests/test_custom_models.py` | 6350 | Tests PEFT with custom models | — | [→](./_files/tests_test_custom_models_py.md) |
| ✅ | `tests/test_decoder_models.py` | 1001 | Tests decoder-only architectures | — | [→](./_files/tests_test_decoder_models_py.md) |
| ✅ | `tests/test_encoder_decoder_models.py` | 434 | Tests encoder-decoder adapters | — | [→](./_files/tests_test_encoder_decoder_models_py.md) |
| ✅ | `tests/test_feature_extraction_models.py` | 379 | Tests feature extraction adapters | — | [→](./_files/tests_test_feature_extraction_models_py.md) |
| ✅ | `tests/test_gptqmodel.py` | 563 | Tests GPTQ quantization adapters | — | [→](./_files/tests_test_gptqmodel_py.md) |
| ✅ | `tests/test_gpu_examples.py` | 5682 | Tests GPU-specific functionality | — | [→](./_files/tests_test_gpu_examples_py.md) |
| ✅ | `tests/test_helpers.py` | 473 | Tests PEFT helper utilities | — | [→](./_files/tests_test_helpers_py.md) |
| ✅ | `tests/test_hub_features.py` | 234 | Tests HuggingFace Hub integration | — | [→](./_files/tests_test_hub_features_py.md) |
| ✅ | `tests/test_incremental_pca.py` | 188 | Tests incremental PCA utility | — | [→](./_files/tests_test_incremental_pca_py.md) |
| ✅ | `tests/test_initialization.py` | 5029 | Tests adapter initialization | — | [→](./_files/tests_test_initialization_py.md) |
| ✅ | `tests/test_integrations.py` | 97 | Tests integration utilities | — | [→](./_files/tests_test_integrations_py.md) |
| ✅ | `tests/test_lora_megatron.py` | 171 | Tests Megatron-LM integration | — | [→](./_files/tests_test_lora_megatron_py.md) |
| ✅ | `tests/test_lora_variants.py` | 316 | Tests DoRA and aLoRA variants | — | [→](./_files/tests_test_lora_variants_py.md) |
| ✅ | `tests/test_lorafa.py` | 152 | Tests LoRA-FA optimizer | — | [→](./_files/tests_test_lorafa_py.md) |
| ✅ | `tests/test_loraplus.py` | 99 | Tests LoRA+ optimizer | — | [→](./_files/tests_test_loraplus_py.md) |
| ✅ | `tests/test_low_level_api.py` | 658 | Tests low-level PEFT APIs | — | [→](./_files/tests_test_low_level_api_py.md) |
| ✅ | `tests/test_mapping.py` | 55 | Tests model reloading behavior | — | [→](./_files/tests_test_mapping_py.md) |
| ✅ | `tests/test_mixed.py` | 791 | Tests mixed adapter combinations | — | [→](./_files/tests_test_mixed_py.md) |
| ✅ | `tests/test_multitask_prompt_tuning.py` | 288 | Tests multi-task prompt tuning | — | [→](./_files/tests_test_multitask_prompt_tuning_py.md) |
| ✅ | `tests/test_osf.py` | 72 | Tests Orthogonal Subspace Finetuning | — | [→](./_files/tests_test_osf_py.md) |
| ✅ | `tests/test_other.py` | 624 | Tests miscellaneous edge cases | — | [→](./_files/tests_test_other_py.md) |
| ✅ | `tests/test_poly.py` | 100 | Tests Polytropon multi-task | — | [→](./_files/tests_test_poly_py.md) |
| ✅ | `tests/test_randlora.py` | 301 | Tests RandLora shared projections | — | [→](./_files/tests_test_randlora_py.md) |
| ✅ | `tests/test_seq_classifier.py` | 320 | Tests sequence classification | — | [→](./_files/tests_test_seq_classifier_py.md) |
| ✅ | `tests/test_shira.py` | 278 | Tests sparse high-rank adaptation | — | [→](./_files/tests_test_shira_py.md) |
| ✅ | `tests/test_stablediffusion.py` | 387 | Tests diffusion model adapters | — | [→](./_files/tests_test_stablediffusion_py.md) |
| ✅ | `tests/test_target_parameters.py` | 546 | Tests direct parameter targeting | — | [→](./_files/tests_test_target_parameters_py.md) |
| ✅ | `tests/test_torch_compile.py` | 599 | Tests torch.compile compatibility | — | [→](./_files/tests_test_torch_compile_py.md) |
| ✅ | `tests/test_trainable_tokens.py` | 1018 | Tests token embedding fine-tuning | — | [→](./_files/tests_test_trainable_tokens_py.md) |
| ✅ | `tests/test_tuners_utils.py` | 2182 | Tests tuner utility infrastructure | — | [→](./_files/tests_test_tuners_utils_py.md) |
| ✅ | `tests/test_vblora.py` | 269 | Tests vector bank LoRA | — | [→](./_files/tests_test_vblora_py.md) |
| ✅ | `tests/test_vera.py` | 298 | Tests VeRA random adaptation | — | [→](./_files/tests_test_vera_py.md) |
| ✅ | `tests/test_vision_models.py` | 160 | Tests vision model adapters | — | [→](./_files/tests_test_vision_models_py.md) |
| ✅ | `tests/test_xlora.py` | 473 | Tests mixture of LoRA experts | — | [→](./_files/tests_test_xlora_py.md) |
| ✅ | `tests/testing_common.py` | 1829 | Shared test infrastructure base | — | [→](./_files/tests_testing_common_py.md) |
| ✅ | `tests/testing_utils.py` | 305 | Test utilities and decorators | — | [→](./_files/tests_testing_utils_py.md) |

## 📄 Other Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `setup.py` | 110 | PyPI package configuration | — | [→](./_files/setup_py.md) |
| ✅ | `src/peft/__init__.py` | 250 | Package API entry point | Workflow: LoRA_Finetuning, QLoRA_Training, Adapter_Inference, Multi_Adapter_Management | [→](./_files/src_peft___init___py.md) |
| ✅ | `src/peft/auto.py` | 184 | Automatic model class selection | Workflow: Adapter_Inference | [→](./_files/src_peft_auto_py.md) |
| ✅ | `src/peft/config.py` | 408 | Configuration base classes | Workflow: LoRA_Finetuning, QLoRA_Training | [→](./_files/src_peft_config_py.md) |
| ✅ | `src/peft/functional.py` | 34 | Functional API for integrations | — | [→](./_files/src_peft_functional_py.md) |
| ✅ | `src/peft/helpers.py` | 251 | Usability and runtime utilities | — | [→](./_files/src_peft_helpers_py.md) |
| ✅ | `src/peft/import_utils.py` | 172 | Optional dependency detection | — | [→](./_files/src_peft_import_utils_py.md) |
| ✅ | `src/peft/mapping.py` | 92 | Registry and adapter injection | Workflow: LoRA_Finetuning, QLoRA_Training | [→](./_files/src_peft_mapping_py.md) |
| ✅ | `src/peft/mapping_func.py` | 128 | Primary PEFT model factory | Workflow: LoRA_Finetuning, QLoRA_Training | [→](./_files/src_peft_mapping_func_py.md) |
| ✅ | `src/peft/mixed_model.py` | 473 | Multi-adapter type support | Workflow: Multi_Adapter_Management | [→](./_files/src_peft_mixed_model_py.md) |
| ✅ | `src/peft/optimizers/__init__.py` | 19 | Specialized optimizer exports | — | [→](./_files/src_peft_optimizers___init___py.md) |
| ✅ | `src/peft/optimizers/lorafa.py` | 256 | LoRA-FA optimizer implementation | — | [→](./_files/src_peft_optimizers_lorafa_py.md) |
| ✅ | `src/peft/optimizers/loraplus.py` | 121 | LoRA+ learning rate scheduling | — | [→](./_files/src_peft_optimizers_loraplus_py.md) |
| ✅ | `src/peft/peft_model.py` | 3387 | Core model wrapper classes | Workflow: LoRA_Finetuning, QLoRA_Training, Adapter_Inference, Multi_Adapter_Management | [→](./_files/src_peft_peft_model_py.md) |
| ✅ | `src/peft/tuners/__init__.py` | 135 | Tuners module API aggregator | — | [→](./_files/src_peft_tuners___init___py.md) |
| ✅ | `src/peft/tuners/_buffer_dict.py` | 159 | Ordered buffer dictionary | — | [→](./_files/src_peft_tuners__buffer_dict_py.md) |
| ✅ | `src/peft/tuners/adalora/__init__.py` | 43 | AdaLoRA package initialization | — | [→](./_files/src_peft_tuners_adalora___init___py.md) |
| ✅ | `src/peft/tuners/adalora/bnb.py` | 143 | Quantized AdaLoRA layers | — | [→](./_files/src_peft_tuners_adalora_bnb_py.md) |
| ✅ | `src/peft/tuners/adalora/config.py` | 108 | Three-phase adaptive config | — | [→](./_files/src_peft_tuners_adalora_config_py.md) |
| ✅ | `src/peft/tuners/adalora/gptq.py` | 71 | GPTQ-quantized AdaLoRA | — | [→](./_files/src_peft_tuners_adalora_gptq_py.md) |
| ✅ | `src/peft/tuners/adalora/layer.py` | 360 | SVD-based layers and allocation | — | [→](./_files/src_peft_tuners_adalora_layer_py.md) |
| ✅ | `src/peft/tuners/adalora/model.py` | 346 | AdaLoRA model orchestration | — | [→](./_files/src_peft_tuners_adalora_model_py.md) |
| ✅ | `src/peft/tuners/adaption_prompt/__init__.py` | 23 | Adaption Prompt initialization | — | [→](./_files/src_peft_tuners_adaption_prompt___init___py.md) |
| ✅ | `src/peft/tuners/adaption_prompt/config.py` | 88 | Model-specific prompt mappings | — | [→](./_files/src_peft_tuners_adaption_prompt_config_py.md) |
| ✅ | `src/peft/tuners/adaption_prompt/layer.py` | 236 | Gated attention with prompts | — | [→](./_files/src_peft_tuners_adaption_prompt_layer_py.md) |
| ✅ | `src/peft/tuners/adaption_prompt/model.py` | 169 | Multi-adapter prompt management | — | [→](./_files/src_peft_tuners_adaption_prompt_model_py.md) |
| ✅ | `src/peft/tuners/adaption_prompt/utils.py` | 158 | Query state recomputation | — | [→](./_files/src_peft_tuners_adaption_prompt_utils_py.md) |
| ✅ | `src/peft/tuners/boft/__init__.py` | 24 | BOFT module initialization | — | [→](./_files/src_peft_tuners_boft___init___py.md) |
| ✅ | `src/peft/tuners/boft/config.py` | 160 | BOFT butterfly parameters | — | [→](./_files/src_peft_tuners_boft_config_py.md) |
| ✅ | `src/peft/tuners/boft/layer.py` | 1011 | Butterfly orthogonal layers | — | [→](./_files/src_peft_tuners_boft_layer_py.md) |
| ✅ | `src/peft/tuners/boft/model.py` | 131 | BOFT model orchestration | — | [→](./_files/src_peft_tuners_boft_model_py.md) |
| ✅ | `src/peft/tuners/bone/__init__.py` | 24 | BONE module (deprecated) | — | [→](./_files/src_peft_tuners_bone___init___py.md) |
| ✅ | `src/peft/tuners/bone/config.py` | 129 | BONE block affine config | — | [→](./_files/src_peft_tuners_bone_config_py.md) |
| ✅ | `src/peft/tuners/bone/layer.py` | 352 | Block-wise affine transform | — | [→](./_files/src_peft_tuners_bone_layer_py.md) |
| ✅ | `src/peft/tuners/bone/model.py` | 126 | BONE model adapter wrapper | — | [→](./_files/src_peft_tuners_bone_model_py.md) |
| ✅ | `src/peft/tuners/c3a/__init__.py` | 23 | C3A registration and exports | — | [→](./_files/src_peft_tuners_c3a___init___py.md) |
| ✅ | `src/peft/tuners/c3a/config.py` | 137 | Block circulant config | — | [→](./_files/src_peft_tuners_c3a_config_py.md) |
| ✅ | `src/peft/tuners/c3a/layer.py` | 202 | FFT-based circulant layers | — | [→](./_files/src_peft_tuners_c3a_layer_py.md) |
| ✅ | `src/peft/tuners/c3a/model.py` | 101 | C3A model orchestration | — | [→](./_files/src_peft_tuners_c3a_model_py.md) |
| ✅ | `src/peft/tuners/c3a/utils.py` | 48 | Circulant matrix FFT utilities | — | [→](./_files/src_peft_tuners_c3a_utils_py.md) |
| ✅ | `src/peft/tuners/cpt/__init__.py` | 24 | CPT registration and exports | — | [→](./_files/src_peft_tuners_cpt___init___py.md) |
| ✅ | `src/peft/tuners/cpt/config.py` | 99 | Context-aware prompt config | — | [→](./_files/src_peft_tuners_cpt_config_py.md) |
| ✅ | `src/peft/tuners/fourierft/__init__.py` | 24 | FourierFT registration | — | [→](./_files/src_peft_tuners_fourierft___init___py.md) |
| ✅ | `src/peft/tuners/fourierft/config.py` | 206 | Frequency-domain tuning config | — | [→](./_files/src_peft_tuners_fourierft_config_py.md) |
| ✅ | `src/peft/tuners/fourierft/layer.py` | 193 | Sparse spectral learning layers | — | [→](./_files/src_peft_tuners_fourierft_layer_py.md) |
| ✅ | `src/peft/tuners/fourierft/model.py` | 128 | FourierFT model orchestration | — | [→](./_files/src_peft_tuners_fourierft_model_py.md) |
| ✅ | `src/peft/tuners/gralora/__init__.py` | 24 | GraLoRA registration | — | [→](./_files/src_peft_tuners_gralora___init___py.md) |
| ✅ | `src/peft/tuners/gralora/config.py` | 182 | Block-structured LoRA config | — | [→](./_files/src_peft_tuners_gralora_config_py.md) |
| ✅ | `src/peft/tuners/gralora/layer.py` | 392 | Block-wise low-rank layers | — | [→](./_files/src_peft_tuners_gralora_layer_py.md) |
| ✅ | `src/peft/tuners/gralora/model.py` | 142 | GraLoRA model orchestration | — | [→](./_files/src_peft_tuners_gralora_model_py.md) |
| ✅ | `src/peft/tuners/hra/__init__.py` | 24 | HRA registration and exports | — | [→](./_files/src_peft_tuners_hra___init___py.md) |
| ✅ | `src/peft/tuners/hra/config.py` | 133 | Householder reflection config | — | [→](./_files/src_peft_tuners_hra_config_py.md) |
| ✅ | `src/peft/tuners/hra/layer.py` | 461 | Orthogonal transformation layers | — | [→](./_files/src_peft_tuners_hra_layer_py.md) |
| ✅ | `src/peft/tuners/hra/model.py` | 131 | HRA model orchestration | — | [→](./_files/src_peft_tuners_hra_model_py.md) |
| ✅ | `src/peft/tuners/ia3/__init__.py` | 39 | IA3 package initialization | — | [→](./_files/src_peft_tuners_ia3___init___py.md) |
| ✅ | `src/peft/tuners/ia3/bnb.py` | 129 | Quantized IA3 layers | — | [→](./_files/src_peft_tuners_ia3_bnb_py.md) |
| ✅ | `src/peft/tuners/ia3/config.py` | 112 | IA3 feedforward/attention config | — | [→](./_files/src_peft_tuners_ia3_config_py.md) |
| ✅ | `src/peft/tuners/ia3/layer.py` | 330 | Activation rescaling layers | — | [→](./_files/src_peft_tuners_ia3_layer_py.md) |
| ✅ | `src/peft/tuners/ia3/model.py` | 315 | IA3 adapter injection | — | [→](./_files/src_peft_tuners_ia3_model_py.md) |
| ✅ | `src/peft/tuners/loha/__init__.py` | 24 | LoHa package registration | — | [→](./_files/src_peft_tuners_loha___init___py.md) |
| ✅ | `src/peft/tuners/loha/config.py` | 143 | LoHa Hadamard parameters | — | [→](./_files/src_peft_tuners_loha_config_py.md) |
| ✅ | `src/peft/tuners/loha/layer.py` | 444 | Hadamard product layers | — | [→](./_files/src_peft_tuners_loha_layer_py.md) |
| ✅ | `src/peft/tuners/loha/model.py` | 116 | LoHa model wrapper | — | [→](./_files/src_peft_tuners_loha_model_py.md) |
| ✅ | `src/peft/tuners/lora/__init__.py` | 65 | LoRA module public API | Workflow: LoRA_Finetuning, QLoRA_Training | [→](./_files/src_peft_tuners_lora___init___py.md) |
| ✅ | `src/peft/tuners/lora/aqlm.py` | 114 | AQLM quantization adapter | — | [→](./_files/src_peft_tuners_lora_aqlm_py.md) |
| ✅ | `src/peft/tuners/lora/arrow.py` | 476 | MoE adaptive routing LoRA | — | [→](./_files/src_peft_tuners_lora_arrow_py.md) |
| ✅ | `src/peft/tuners/lora/awq.py` | 121 | AWQ quantization adapter | — | [→](./_files/src_peft_tuners_lora_awq_py.md) |
| ✅ | `src/peft/tuners/lora/bnb.py` | 611 | BitsAndBytes 4/8-bit LoRA | Workflow: QLoRA_Training | [→](./_files/src_peft_tuners_lora_bnb_py.md) |
| ✅ | `src/peft/tuners/lora/config.py` | 879 | LoRA configuration dataclasses | Workflow: LoRA_Finetuning, QLoRA_Training | [→](./_files/src_peft_tuners_lora_config_py.md) |
| ✅ | `src/peft/tuners/lora/corda.py` | 360 | Correlation-aware initialization | — | [→](./_files/src_peft_tuners_lora_corda_py.md) |
| ✅ | `src/peft/tuners/lora/dora.py` | 203 | Weight-decomposed LoRA layers | — | [→](./_files/src_peft_tuners_lora_dora_py.md) |
| ✅ | `src/peft/tuners/lora/eetq.py` | 118 | EETQ quantization adapter | — | [→](./_files/src_peft_tuners_lora_eetq_py.md) |
| ✅ | `src/peft/tuners/lora/eva.py` | 739 | Eigenvalue activation-aware init | — | [→](./_files/src_peft_tuners_lora_eva_py.md) |
| ✅ | `src/peft/tuners/lora/gptq.py` | 154 | GPTQ quantization adapter | — | [→](./_files/src_peft_tuners_lora_gptq_py.md) |
| ✅ | `src/peft/tuners/lora/hqq.py` | 251 | Half-quadratic quantization | — | [→](./_files/src_peft_tuners_lora_hqq_py.md) |
| ✅ | `src/peft/tuners/lora/inc.py` | 78 | Intel Neural Compressor LoRA | — | [→](./_files/src_peft_tuners_lora_inc_py.md) |
| ✅ | `src/peft/tuners/lora/layer.py` | 2304 | Core LoRA layer implementations | Workflow: LoRA_Finetuning, QLoRA_Training | [→](./_files/src_peft_tuners_lora_layer_py.md) |
| ✅ | `src/peft/tuners/lora/model.py` | 872 | LoRA model orchestration | Workflow: LoRA_Finetuning, QLoRA_Training, Multi_Adapter_Management | [→](./_files/src_peft_tuners_lora_model_py.md) |
| ✅ | `src/peft/tuners/lora/torchao.py` | 156 | PyTorch AO quantization | — | [→](./_files/src_peft_tuners_lora_torchao_py.md) |
| ✅ | `src/peft/tuners/lora/tp_layer.py` | 350 | Megatron tensor-parallel LoRA | — | [→](./_files/src_peft_tuners_lora_tp_layer_py.md) |
| ✅ | `src/peft/tuners/lora/variants.py` | 926 | Advanced LoRA variants | — | [→](./_files/src_peft_tuners_lora_variants_py.md) |
| ✅ | `src/peft/tuners/lycoris_utils.py` | 263 | LyCORIS-family base classes | — | [→](./_files/src_peft_tuners_lycoris_utils_py.md) |
| ✅ | `src/peft/tuners/miss/__init__.py` | 24 | MISS package registration | — | [→](./_files/src_peft_tuners_miss___init___py.md) |
| ✅ | `src/peft/tuners/miss/config.py` | 140 | MISS three variants config | — | [→](./_files/src_peft_tuners_miss_config_py.md) |
| ✅ | `src/peft/tuners/miss/layer.py` | 393 | Householder reflection layers | — | [→](./_files/src_peft_tuners_miss_layer_py.md) |
| ✅ | `src/peft/tuners/miss/model.py` | 130 | MISS model wrapper | — | [→](./_files/src_peft_tuners_miss_model_py.md) |
| ✅ | `src/peft/tuners/mixed/__init__.py` | 18 | Mixed adapter type exports | — | [→](./_files/src_peft_tuners_mixed___init___py.md) |
| ✅ | `src/peft/tuners/mixed/model.py` | 309 | Multi-adapter combination | Workflow: Multi_Adapter_Management | [→](./_files/src_peft_tuners_mixed_model_py.md) |
| ✅ | `src/peft/tuners/multitask_prompt_tuning/__init__.py` | 25 | Multitask prompt registration | — | [→](./_files/src_peft_tuners_multitask_prompt_tuning___init___py.md) |
| ✅ | `src/peft/tuners/multitask_prompt_tuning/config.py` | 62 | Multi-task initialization modes | — | [→](./_files/src_peft_tuners_multitask_prompt_tuning_config_py.md) |
| ✅ | `src/peft/tuners/multitask_prompt_tuning/model.py` | 120 | Factorized prompt embeddings | — | [→](./_files/src_peft_tuners_multitask_prompt_tuning_model_py.md) |
| ✅ | `src/peft/tuners/oft/__init__.py` | 52 | OFT with quantization support | — | [→](./_files/src_peft_tuners_oft___init___py.md) |
| ✅ | `src/peft/tuners/oft/aqlm.py` | 105 | OFT for AQLM quantization | — | [→](./_files/src_peft_tuners_oft_aqlm_py.md) |
| ✅ | `src/peft/tuners/oft/awq.py` | 119 | OFT for AWQ quantization | — | [→](./_files/src_peft_tuners_oft_awq_py.md) |
| ✅ | `src/peft/tuners/oft/bnb.py` | 388 | OFT for bitsandbytes | — | [→](./_files/src_peft_tuners_oft_bnb_py.md) |
| ✅ | `src/peft/tuners/oft/config.py` | 213 | OFT orthogonal config | — | [→](./_files/src_peft_tuners_oft_config_py.md) |
| ✅ | `src/peft/tuners/oft/eetq.py` | 116 | OFT for EETQ quantization | — | [→](./_files/src_peft_tuners_oft_eetq_py.md) |
| ✅ | `src/peft/tuners/oft/gptq.py` | 118 | OFT for GPTQ quantization | — | [→](./_files/src_peft_tuners_oft_gptq_py.md) |
| ✅ | `src/peft/tuners/oft/hqq.py` | 186 | OFT for HQQ quantization | — | [→](./_files/src_peft_tuners_oft_hqq_py.md) |
| ✅ | `src/peft/tuners/oft/inc.py` | 78 | OFT for Intel Neural Compressor | — | [→](./_files/src_peft_tuners_oft_inc_py.md) |
| ✅ | `src/peft/tuners/oft/layer.py` | 950 | Orthogonal rotation layers | — | [→](./_files/src_peft_tuners_oft_layer_py.md) |
| ✅ | `src/peft/tuners/oft/model.py` | 199 | OFT model with dispatching | — | [→](./_files/src_peft_tuners_oft_model_py.md) |
| ✅ | `src/peft/tuners/poly/__init__.py` | 24 | Poly module initialization | — | [→](./_files/src_peft_tuners_poly___init___py.md) |
| ✅ | `src/peft/tuners/poly/config.py` | 103 | Poly multi-task config | — | [→](./_files/src_peft_tuners_poly_config_py.md) |
| ✅ | `src/peft/tuners/poly/layer.py` | 165 | Multi-skill LoRA layers | — | [→](./_files/src_peft_tuners_poly_layer_py.md) |
| ✅ | `src/peft/tuners/poly/model.py` | 104 | Poly model orchestration | — | [→](./_files/src_peft_tuners_poly_model_py.md) |
| ✅ | `src/peft/tuners/poly/router.py` | 81 | Task-specific skill routing | — | [→](./_files/src_peft_tuners_poly_router_py.md) |
| ✅ | `src/peft/tuners/randlora/__init__.py` | 40 | RandLoRA package with lazy imports | — | [→](./_files/src_peft_tuners_randlora___init___py.md) |
| ✅ | `src/peft/tuners/randlora/bnb.py` | 456 | Quantized RandLoRA layers | — | [→](./_files/src_peft_tuners_randlora_bnb_py.md) |
| ✅ | `src/peft/tuners/randlora/config.py` | 199 | Shared random basis config | — | [→](./_files/src_peft_tuners_randlora_config_py.md) |
| ✅ | `src/peft/tuners/randlora/layer.py` | 350 | Shared projection layers | — | [→](./_files/src_peft_tuners_randlora_layer_py.md) |
| ✅ | `src/peft/tuners/randlora/model.py` | 356 | RandLoRA with shared bases | — | [→](./_files/src_peft_tuners_randlora_model_py.md) |
| ✅ | `src/peft/tuners/road/__init__.py` | 47 | RoAd package with lazy imports | — | [→](./_files/src_peft_tuners_road___init___py.md) |
| ✅ | `src/peft/tuners/road/bnb.py` | 407 | Quantized RoAd layers | — | [→](./_files/src_peft_tuners_road_bnb_py.md) |
| ✅ | `src/peft/tuners/road/config.py` | 126 | Rotation variant config | — | [→](./_files/src_peft_tuners_road_config_py.md) |
| ✅ | `src/peft/tuners/road/layer.py` | 418 | 2D rotation adaptation layers | — | [→](./_files/src_peft_tuners_road_layer_py.md) |
| ✅ | `src/peft/tuners/road/model.py` | 163 | RoAd with mixed batching | — | [→](./_files/src_peft_tuners_road_model_py.md) |
| ✅ | `src/peft/tuners/shira/__init__.py` | 27 | SHiRA package registration | — | [→](./_files/src_peft_tuners_shira___init___py.md) |
| ✅ | `src/peft/tuners/shira/config.py` | 129 | Sparse mask config | — | [→](./_files/src_peft_tuners_shira_config_py.md) |
| ✅ | `src/peft/tuners/shira/layer.py` | 217 | Sparse high-rank layers | — | [→](./_files/src_peft_tuners_shira_layer_py.md) |
| ✅ | `src/peft/tuners/shira/mask_functions.py` | 72 | Sparsity mask generation | — | [→](./_files/src_peft_tuners_shira_mask_functions_py.md) |
| ✅ | `src/peft/tuners/shira/model.py` | 142 | SHiRA with mask generation | — | [→](./_files/src_peft_tuners_shira_model_py.md) |
| ✅ | `src/peft/tuners/tuners_utils.py` | 2041 | Base tuner classes and infra | Workflow: LoRA_Finetuning | [→](./_files/src_peft_tuners_tuners_utils_py.md) |
| ✅ | `src/peft/tuners/vblora/__init__.py` | 24 | VBLoRA module initialization | — | [→](./_files/src_peft_tuners_vblora___init___py.md) |
| ✅ | `src/peft/tuners/vblora/config.py` | 196 | VBLoRA configuration | — | [→](./_files/src_peft_tuners_vblora_config_py.md) |
| ✅ | `src/peft/tuners/vblora/layer.py` | 251 | Vector bank top-k layers | — | [→](./_files/src_peft_tuners_vblora_layer_py.md) |
| ✅ | `src/peft/tuners/vblora/model.py` | 209 | VBLoRA model and vector bank | — | [→](./_files/src_peft_tuners_vblora_model_py.md) |
| ✅ | `src/peft/tuners/vera/__init__.py` | 40 | VeRA module initialization | — | [→](./_files/src_peft_tuners_vera___init___py.md) |
| ✅ | `src/peft/tuners/vera/bnb.py` | 411 | Quantized VeRA layers | — | [→](./_files/src_peft_tuners_vera_bnb_py.md) |
| ✅ | `src/peft/tuners/vera/config.py` | 162 | VeRA configuration | — | [→](./_files/src_peft_tuners_vera_config_py.md) |
| ✅ | `src/peft/tuners/vera/layer.py` | 291 | VeRA adapter layers | — | [→](./_files/src_peft_tuners_vera_layer_py.md) |
| ✅ | `src/peft/tuners/vera/model.py` | 294 | VeRA model orchestration | — | [→](./_files/src_peft_tuners_vera_model_py.md) |
| ✅ | `src/peft/utils/__init__.py` | 133 | Utils module API aggregator | — | [→](./_files/src_peft_utils___init___py.md) |
| ✅ | `src/peft/utils/constants.py` | 362 | Model architecture constants | — | [→](./_files/src_peft_utils_constants_py.md) |
| ✅ | `src/peft/utils/hotswap.py` | 630 | Rapid adapter switching | Workflow: Adapter_Hotswapping | [→](./_files/src_peft_utils_hotswap_py.md) |
| ✅ | `src/peft/utils/incremental_pca.py` | 338 | Memory-efficient incremental PCA | — | [→](./_files/src_peft_utils_incremental_pca_py.md) |
| ✅ | `src/peft/utils/integrations.py` | 291 | External framework integration | Workflow: QLoRA_Training | [→](./_files/src_peft_utils_integrations_py.md) |
| ✅ | `src/peft/utils/loftq_utils.py` | 410 | LoftQ quantization-aware init | — | [→](./_files/src_peft_utils_loftq_utils_py.md) |
| ✅ | `src/peft/utils/merge_utils.py` | 268 | Multi-adapter merging | Workflow: Multi_Adapter_Management, Adapter_Inference | [→](./_files/src_peft_utils_merge_utils_py.md) |
| ✅ | `src/peft/utils/other.py` | 1648 | Miscellaneous helpers | Workflow: QLoRA_Training | [→](./_files/src_peft_utils_other_py.md) |
| ✅ | `src/peft/utils/peft_types.py` | 183 | Core type enumerations | — | [→](./_files/src_peft_utils_peft_types_py.md) |
| ✅ | `src/peft/utils/save_and_load.py` | 724 | Adapter serialization and I/O | Workflow: LoRA_Finetuning, QLoRA_Training, Adapter_Inference | [→](./_files/src_peft_utils_save_and_load_py.md) |
| ✅ | `src/peft/utils/warning.py` | 17 | Custom PEFT warning class | — | [→](./_files/src_peft_utils_warning_py.md) |

---

## Page Indexes

Each page type has its own index file for tracking and integrity checking:

| Index | Description |
|-------|-------------|
| [Workflows](./_WorkflowIndex.md) | Workflow pages with step connections |
| [Principles](./_PrincipleIndex.md) | Principle pages with implementations |
| [Implementations](./_ImplementationIndex.md) | Implementation pages with source locations |
| [Environments](./_EnvironmentIndex.md) | Environment requirement pages |
| [Heuristics](./_HeuristicIndex.md) | Heuristic/tips pages |
