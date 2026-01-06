# Repository Map: huggingface_peft

> **Compact index** of repository files.
> Each file has a detail page in `_files/` with Understanding to fill.
> Mark files as ✅ explored in the table below as you complete them.

| Property | Value |
|----------|-------|
| Repository | https://github.com/huggingface/peft |
| Branch | main |
| Generated | 2025-12-18 12:30 |
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
| ✅ | `method_comparison/__init__.py` | 0 | Empty package init | — | [→](./_files/method_comparison___init___py.md) |
| ✅ | `method_comparison/app.py` | 385 | Gradio benchmark app | — | [→](./_files/method_comparison_app_py.md) |
| ✅ | `method_comparison/processing.py` | 150 | Benchmark data processing | — | [→](./_files/method_comparison_processing_py.md) |
| ✅ | `method_comparison/sanitizer.py` | 100 | AST-safe filter utility | — | [→](./_files/method_comparison_sanitizer_py.md) |
| ✅ | `method_comparison/test_sanitizer.py` | 38 | Sanitizer test cases | — | [→](./_files/method_comparison_test_sanitizer_py.md) |

## 📝 Example Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `scripts/ci_clean_cache.py` | 67 | CI cache cleaner | — | [→](./_files/scripts_ci_clean_cache_py.md) |
| ✅ | `scripts/convert-bone-to-miss.py` | 70 | Bone to MiSS converter | — | [→](./_files/scripts_convert-bone-to-miss_py.md) |
| ✅ | `scripts/launch_notebook_mp.py` | 47 | Multiprocess notebook test | — | [→](./_files/scripts_launch_notebook_mp_py.md) |
| ✅ | `scripts/log_reports.py` | 144 | CI Slack reporter | — | [→](./_files/scripts_log_reports_py.md) |
| ✅ | `scripts/stale.py` | 65 | Issue stale automation | — | [→](./_files/scripts_stale_py.md) |
| ✅ | `scripts/train_memory.py` | 276 | Memory benchmark tool | — | [→](./_files/scripts_train_memory_py.md) |

## 🧪 Test Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `tests/__init__.py` | 0 | Empty test package | — | [→](./_files/tests___init___py.md) |
| ✅ | `tests/conftest.py` | 86 | Pytest configuration | — | [→](./_files/tests_conftest_py.md) |
| ✅ | `tests/test_adaption_prompt.py` | 416 | Tests AdaptionPrompt adapter | — | [→](./_files/tests_test_adaption_prompt_py.md) |
| ✅ | `tests/test_arrow.py` | 509 | Tests Arrow routing | — | [→](./_files/tests_test_arrow_py.md) |
| ✅ | `tests/test_auto.py` | 225 | Tests AutoPeftModel | Workflow: Adapter_Loading_Inference | [→](./_files/tests_test_auto_py.md) |
| ✅ | `tests/test_boft.py` | 84 | Tests BOFT adapter | — | [→](./_files/tests_test_boft_py.md) |
| ✅ | `tests/test_bufferdict.py` | 48 | Tests BufferDict utility | — | [→](./_files/tests_test_bufferdict_py.md) |
| ✅ | `tests/test_common_gpu.py` | 2185 | GPU-specific operations | Workflow: LoRA_Fine_Tuning, QLoRA_Training | [→](./_files/tests_test_common_gpu_py.md) |
| ✅ | `tests/test_config.py` | 599 | Tests config serialization | Workflow: LoRA_Fine_Tuning | [→](./_files/tests_test_config_py.md) |
| ✅ | `tests/test_cpt.py` | 305 | Tests CPT adapter | — | [→](./_files/tests_test_cpt_py.md) |
| ✅ | `tests/test_custom_models.py` | 6350 | Custom architecture tests | Workflow: LoRA_Fine_Tuning | [→](./_files/tests_test_custom_models_py.md) |
| ✅ | `tests/test_decoder_models.py` | 1001 | Decoder model tests | Workflow: LoRA_Fine_Tuning | [→](./_files/tests_test_decoder_models_py.md) |
| ✅ | `tests/test_encoder_decoder_models.py` | 434 | Seq2seq model tests | Workflow: LoRA_Fine_Tuning | [→](./_files/tests_test_encoder_decoder_models_py.md) |
| ✅ | `tests/test_feature_extraction_models.py` | 379 | Encoder model tests | Workflow: LoRA_Fine_Tuning | [→](./_files/tests_test_feature_extraction_models_py.md) |
| ✅ | `tests/test_gptqmodel.py` | 563 | Tests GPTQ quantization | Workflow: QLoRA_Training | [→](./_files/tests_test_gptqmodel_py.md) |
| ✅ | `tests/test_gpu_examples.py` | 5682 | Advanced GPU features | Workflow: LoRA_Fine_Tuning, Adapter_Merging | [→](./_files/tests_test_gpu_examples_py.md) |
| ✅ | `tests/test_helpers.py` | 473 | Tests helper functions | — | [→](./_files/tests_test_helpers_py.md) |
| ✅ | `tests/test_hub_features.py` | 234 | Tests Hub integration | Workflow: Adapter_Loading_Inference | [→](./_files/tests_test_hub_features_py.md) |
| ✅ | `tests/test_incremental_pca.py` | 188 | Tests IncrementalPCA | — | [→](./_files/tests_test_incremental_pca_py.md) |
| ✅ | `tests/test_initialization.py` | 5029 | Tests init strategies | Workflow: LoRA_Fine_Tuning | [→](./_files/tests_test_initialization_py.md) |
| ✅ | `tests/test_integrations.py` | 97 | Tests library integrations | Workflow: QLoRA_Training | [→](./_files/tests_test_integrations_py.md) |
| ✅ | `tests/test_lora_megatron.py` | 171 | Tests Megatron integration | — | [→](./_files/tests_test_lora_megatron_py.md) |
| ✅ | `tests/test_lora_variants.py` | 316 | Tests DoRA, aLoRA | Workflow: LoRA_Fine_Tuning | [→](./_files/tests_test_lora_variants_py.md) |
| ✅ | `tests/test_lorafa.py` | 152 | Tests LoRA-FA optimizer | Workflow: LoRA_Fine_Tuning | [→](./_files/tests_test_lorafa_py.md) |
| ✅ | `tests/test_loraplus.py` | 99 | Tests LoRA+ optimizer | Workflow: LoRA_Fine_Tuning | [→](./_files/tests_test_loraplus_py.md) |
| ✅ | `tests/test_low_level_api.py` | 658 | Tests low-level API | — | [→](./_files/tests_test_low_level_api_py.md) |
| ✅ | `tests/test_mapping.py` | 55 | Tests get_peft_model | Workflow: LoRA_Fine_Tuning | [→](./_files/tests_test_mapping_py.md) |
| ✅ | `tests/test_mixed.py` | 791 | Tests mixed adapters | Workflow: Multi_Adapter_Management | [→](./_files/tests_test_mixed_py.md) |
| ✅ | `tests/test_multitask_prompt_tuning.py` | 288 | Tests multitask prompt | — | [→](./_files/tests_test_multitask_prompt_tuning_py.md) |
| ✅ | `tests/test_osf.py` | 72 | Tests OSF adapter | — | [→](./_files/tests_test_osf_py.md) |
| ✅ | `tests/test_other.py` | 624 | Misc edge cases | — | [→](./_files/tests_test_other_py.md) |
| ✅ | `tests/test_poly.py` | 100 | Tests Poly adapter | — | [→](./_files/tests_test_poly_py.md) |
| ✅ | `tests/test_randlora.py` | 301 | Tests RandLoRA | — | [→](./_files/tests_test_randlora_py.md) |
| ✅ | `tests/test_seq_classifier.py` | 320 | Tests classifier models | Workflow: LoRA_Fine_Tuning | [→](./_files/tests_test_seq_classifier_py.md) |
| ✅ | `tests/test_shira.py` | 278 | Tests SHiRA adapter | — | [→](./_files/tests_test_shira_py.md) |
| ✅ | `tests/test_stablediffusion.py` | 387 | Tests diffusers integration | — | [→](./_files/tests_test_stablediffusion_py.md) |
| ✅ | `tests/test_target_parameters.py` | 546 | Tests param targeting | Workflow: LoRA_Fine_Tuning | [→](./_files/tests_test_target_parameters_py.md) |
| ✅ | `tests/test_torch_compile.py` | 599 | Tests PyTorch compile | Workflow: Adapter_Loading_Inference | [→](./_files/tests_test_torch_compile_py.md) |
| ✅ | `tests/test_trainable_tokens.py` | 1018 | Tests trainable tokens | — | [→](./_files/tests_test_trainable_tokens_py.md) |
| ✅ | `tests/test_tuners_utils.py` | 2182 | Tests tuner utilities | Workflow: LoRA_Fine_Tuning | [→](./_files/tests_test_tuners_utils_py.md) |
| ✅ | `tests/test_vblora.py` | 269 | Tests VBLoRA | — | [→](./_files/tests_test_vblora_py.md) |
| ✅ | `tests/test_vera.py` | 298 | Tests VeRA adapter | — | [→](./_files/tests_test_vera_py.md) |
| ✅ | `tests/test_vision_models.py` | 160 | Tests vision models | Workflow: LoRA_Fine_Tuning | [→](./_files/tests_test_vision_models_py.md) |
| ✅ | `tests/test_xlora.py` | 473 | Tests X-LoRA | — | [→](./_files/tests_test_xlora_py.md) |
| ✅ | `tests/testing_common.py` | 1829 | Common test infrastructure | — | [→](./_files/tests_testing_common_py.md) |
| ✅ | `tests/testing_utils.py` | 305 | Test utility decorators | — | [→](./_files/tests_testing_utils_py.md) |

## 📄 Other Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `setup.py` | 110 | Package setup/metadata | — | [→](./_files/setup_py.md) |
| ✅ | `src/peft/__init__.py` | 250 | Main package entry | Workflow: LoRA_Fine_Tuning, QLoRA_Training, Adapter_Loading_Inference, Adapter_Merging, Multi_Adapter_Management | [→](./_files/src_peft___init___py.md) |
| ✅ | `src/peft/auto.py` | 184 | Auto model loaders | Workflow: Adapter_Loading_Inference | [→](./_files/src_peft_auto_py.md) |
| ✅ | `src/peft/config.py` | 408 | Base config classes | Workflow: LoRA_Fine_Tuning, QLoRA_Training | [→](./_files/src_peft_config_py.md) |
| ✅ | `src/peft/functional.py` | 34 | Functional API exports | — | [→](./_files/src_peft_functional_py.md) |
| ✅ | `src/peft/helpers.py` | 251 | Helper utilities | Workflow: QLoRA_Training | [→](./_files/src_peft_helpers_py.md) |
| ✅ | `src/peft/import_utils.py` | 172 | Optional dep checks | Workflow: QLoRA_Training | [→](./_files/src_peft_import_utils_py.md) |
| ✅ | `src/peft/mapping.py` | 92 | Registry mappings | Workflow: LoRA_Fine_Tuning | [→](./_files/src_peft_mapping_py.md) |
| ✅ | `src/peft/mapping_func.py` | 128 | get_peft_model factory | Workflow: LoRA_Fine_Tuning, QLoRA_Training | [→](./_files/src_peft_mapping_func_py.md) |
| ✅ | `src/peft/mixed_model.py` | 473 | Mixed adapter model | Workflow: Multi_Adapter_Management | [→](./_files/src_peft_mixed_model_py.md) |
| ✅ | `src/peft/optimizers/__init__.py` | 19 | Optimizer exports | Workflow: LoRA_Fine_Tuning | [→](./_files/src_peft_optimizers___init___py.md) |
| ✅ | `src/peft/optimizers/lorafa.py` | 256 | LoRA-FA optimizer | Workflow: LoRA_Fine_Tuning | [→](./_files/src_peft_optimizers_lorafa_py.md) |
| ✅ | `src/peft/optimizers/loraplus.py` | 121 | LoRA+ optimizer | Workflow: LoRA_Fine_Tuning | [→](./_files/src_peft_optimizers_loraplus_py.md) |
| ✅ | `src/peft/peft_model.py` | 3387 | Core PeftModel wrapper | Workflow: LoRA_Fine_Tuning, QLoRA_Training, Adapter_Loading_Inference, Adapter_Merging, Multi_Adapter_Management | [→](./_files/src_peft_peft_model_py.md) |
| ✅ | `src/peft/tuners/__init__.py` | 135 | Tuners package exports | Workflow: LoRA_Fine_Tuning | [→](./_files/src_peft_tuners___init___py.md) |
| ✅ | `src/peft/tuners/_buffer_dict.py` | 159 | BufferDict utility | — | [→](./_files/src_peft_tuners__buffer_dict_py.md) |
| ✅ | `src/peft/tuners/adalora/__init__.py` | 43 | AdaLoRA exports | — | [→](./_files/src_peft_tuners_adalora___init___py.md) |
| ✅ | `src/peft/tuners/adalora/bnb.py` | 143 | AdaLoRA 8/4-bit layers | — | [→](./_files/src_peft_tuners_adalora_bnb_py.md) |
| ✅ | `src/peft/tuners/adalora/config.py` | 108 | AdaLoRA configuration | — | [→](./_files/src_peft_tuners_adalora_config_py.md) |
| ✅ | `src/peft/tuners/adalora/gptq.py` | 71 | AdaLoRA GPTQ layer | — | [→](./_files/src_peft_tuners_adalora_gptq_py.md) |
| ✅ | `src/peft/tuners/adalora/layer.py` | 360 | AdaLoRA SVD layers | — | [→](./_files/src_peft_tuners_adalora_layer_py.md) |
| ✅ | `src/peft/tuners/adalora/model.py` | 346 | AdaLoRA model class | — | [→](./_files/src_peft_tuners_adalora_model_py.md) |
| ✅ | `src/peft/tuners/adaption_prompt/__init__.py` | 23 | AdaptionPrompt exports | — | [→](./_files/src_peft_tuners_adaption_prompt___init___py.md) |
| ✅ | `src/peft/tuners/adaption_prompt/config.py` | 88 | AdaptionPrompt config | — | [→](./_files/src_peft_tuners_adaption_prompt_config_py.md) |
| ✅ | `src/peft/tuners/adaption_prompt/layer.py` | 236 | AdaptedAttention layer | — | [→](./_files/src_peft_tuners_adaption_prompt_layer_py.md) |
| ✅ | `src/peft/tuners/adaption_prompt/model.py` | 169 | AdaptionPrompt model | — | [→](./_files/src_peft_tuners_adaption_prompt_model_py.md) |
| ✅ | `src/peft/tuners/adaption_prompt/utils.py` | 158 | RoPE query utils | — | [→](./_files/src_peft_tuners_adaption_prompt_utils_py.md) |
| ✅ | `src/peft/tuners/boft/__init__.py` | 24 | BOFT exports | — | [→](./_files/src_peft_tuners_boft___init___py.md) |
| ✅ | `src/peft/tuners/boft/config.py` | 160 | BOFT configuration | — | [→](./_files/src_peft_tuners_boft_config_py.md) |
| ✅ | `src/peft/tuners/boft/layer.py` | 1011 | BOFT Cayley layers | — | [→](./_files/src_peft_tuners_boft_layer_py.md) |
| ✅ | `src/peft/tuners/boft/model.py` | 131 | BOFT model class | — | [→](./_files/src_peft_tuners_boft_model_py.md) |
| ✅ | `src/peft/tuners/bone/__init__.py` | 24 | BONE exports (deprecated) | — | [→](./_files/src_peft_tuners_bone___init___py.md) |
| ✅ | `src/peft/tuners/bone/config.py` | 129 | BONE configuration | — | [→](./_files/src_peft_tuners_bone_config_py.md) |
| ✅ | `src/peft/tuners/bone/layer.py` | 352 | BONE Householder layers | — | [→](./_files/src_peft_tuners_bone_layer_py.md) |
| ✅ | `src/peft/tuners/bone/model.py` | 126 | BONE model class | — | [→](./_files/src_peft_tuners_bone_model_py.md) |
| ✅ | `src/peft/tuners/c3a/__init__.py` | 23 | C3A exports | — | [→](./_files/src_peft_tuners_c3a___init___py.md) |
| ✅ | `src/peft/tuners/c3a/config.py` | 137 | C3A configuration | — | [→](./_files/src_peft_tuners_c3a_config_py.md) |
| ✅ | `src/peft/tuners/c3a/layer.py` | 202 | C3A FFT layers | — | [→](./_files/src_peft_tuners_c3a_layer_py.md) |
| ✅ | `src/peft/tuners/c3a/model.py` | 101 | C3A model class | — | [→](./_files/src_peft_tuners_c3a_model_py.md) |
| ✅ | `src/peft/tuners/c3a/utils.py` | 48 | FFT convolution utils | — | [→](./_files/src_peft_tuners_c3a_utils_py.md) |
| ✅ | `src/peft/tuners/cpt/__init__.py` | 24 | CPT exports | — | [→](./_files/src_peft_tuners_cpt___init___py.md) |
| ✅ | `src/peft/tuners/cpt/config.py` | 99 | CPT configuration | — | [→](./_files/src_peft_tuners_cpt_config_py.md) |
| ✅ | `src/peft/tuners/fourierft/__init__.py` | 24 | FourierFT exports | — | [→](./_files/src_peft_tuners_fourierft___init___py.md) |
| ✅ | `src/peft/tuners/fourierft/config.py` | 206 | FourierFT configuration | — | [→](./_files/src_peft_tuners_fourierft_config_py.md) |
| ✅ | `src/peft/tuners/fourierft/layer.py` | 193 | FourierFT sparse layers | — | [→](./_files/src_peft_tuners_fourierft_layer_py.md) |
| ✅ | `src/peft/tuners/fourierft/model.py` | 128 | FourierFT model class | — | [→](./_files/src_peft_tuners_fourierft_model_py.md) |
| ✅ | `src/peft/tuners/gralora/__init__.py` | 24 | GraLoRA exports | — | [→](./_files/src_peft_tuners_gralora___init___py.md) |
| ✅ | `src/peft/tuners/gralora/config.py` | 182 | GraLoRA configuration | — | [→](./_files/src_peft_tuners_gralora_config_py.md) |
| ✅ | `src/peft/tuners/gralora/layer.py` | 392 | GraLoRA block layers | — | [→](./_files/src_peft_tuners_gralora_layer_py.md) |
| ✅ | `src/peft/tuners/gralora/model.py` | 142 | GraLoRA model class | — | [→](./_files/src_peft_tuners_gralora_model_py.md) |
| ✅ | `src/peft/tuners/hra/__init__.py` | 24 | HRA exports | — | [→](./_files/src_peft_tuners_hra___init___py.md) |
| ✅ | `src/peft/tuners/hra/config.py` | 133 | HRA configuration | — | [→](./_files/src_peft_tuners_hra_config_py.md) |
| ✅ | `src/peft/tuners/hra/layer.py` | 461 | HRA Householder layers | — | [→](./_files/src_peft_tuners_hra_layer_py.md) |
| ✅ | `src/peft/tuners/hra/model.py` | 131 | HRA model class | — | [→](./_files/src_peft_tuners_hra_model_py.md) |
| ✅ | `src/peft/tuners/ia3/__init__.py` | 39 | IA3 exports | — | [→](./_files/src_peft_tuners_ia3___init___py.md) |
| ✅ | `src/peft/tuners/ia3/bnb.py` | 129 | IA3 quantized layers | — | [→](./_files/src_peft_tuners_ia3_bnb_py.md) |
| ✅ | `src/peft/tuners/ia3/config.py` | 112 | IA3 configuration | — | [→](./_files/src_peft_tuners_ia3_config_py.md) |
| ✅ | `src/peft/tuners/ia3/layer.py` | 330 | IA3 scaling layers | — | [→](./_files/src_peft_tuners_ia3_layer_py.md) |
| ✅ | `src/peft/tuners/ia3/model.py` | 315 | IA3 model class | — | [→](./_files/src_peft_tuners_ia3_model_py.md) |
| ✅ | `src/peft/tuners/ln_tuning/__init__.py` | 24 | LN Tuning exports | — | [→](./_files/src_peft_tuners_ln_tuning___init___py.md) |
| ✅ | `src/peft/tuners/ln_tuning/config.py` | 52 | LN Tuning config | — | [→](./_files/src_peft_tuners_ln_tuning_config_py.md) |
| ✅ | `src/peft/tuners/ln_tuning/layer.py` | 68 | LN Tuning layers | — | [→](./_files/src_peft_tuners_ln_tuning_layer_py.md) |
| ✅ | `src/peft/tuners/ln_tuning/model.py` | 149 | LN Tuning model | — | [→](./_files/src_peft_tuners_ln_tuning_model_py.md) |
| ✅ | `src/peft/tuners/loha/__init__.py` | 24 | LoHa exports | — | [→](./_files/src_peft_tuners_loha___init___py.md) |
| ✅ | `src/peft/tuners/loha/config.py` | 143 | LoHa configuration | — | [→](./_files/src_peft_tuners_loha_config_py.md) |
| ✅ | `src/peft/tuners/loha/layer.py` | 444 | LoHa Hadamard layers | — | [→](./_files/src_peft_tuners_loha_layer_py.md) |
| ✅ | `src/peft/tuners/loha/model.py` | 116 | LoHa model class | — | [→](./_files/src_peft_tuners_loha_model_py.md) |
| ✅ | `src/peft/tuners/lokr/__init__.py` | 24 | LoKr exports | — | [→](./_files/src_peft_tuners_lokr___init___py.md) |
| ✅ | `src/peft/tuners/lokr/config.py` | 175 | LoKr configuration | — | [→](./_files/src_peft_tuners_lokr_config_py.md) |
| ✅ | `src/peft/tuners/lokr/layer.py` | 502 | LoKr Kronecker layers | — | [→](./_files/src_peft_tuners_lokr_layer_py.md) |
| ✅ | `src/peft/tuners/lokr/model.py` | 116 | LoKr model class | — | [→](./_files/src_peft_tuners_lokr_model_py.md) |
| ✅ | `src/peft/tuners/lora/__init__.py` | 65 | LoRA module exports | Workflow: LoRA_Fine_Tuning, QLoRA_Training | [→](./_files/src_peft_tuners_lora___init___py.md) |
| ✅ | `src/peft/tuners/lora/aqlm.py` | 114 | LoRA AQLM layer | — | [→](./_files/src_peft_tuners_lora_aqlm_py.md) |
| ✅ | `src/peft/tuners/lora/arrow.py` | 476 | Arrow MoE routing | — | [→](./_files/src_peft_tuners_lora_arrow_py.md) |
| ✅ | `src/peft/tuners/lora/awq.py` | 121 | LoRA AWQ layer | Workflow: QLoRA_Training | [→](./_files/src_peft_tuners_lora_awq_py.md) |
| ✅ | `src/peft/tuners/lora/bnb.py` | 611 | LoRA 8/4-bit layers | Workflow: QLoRA_Training | [→](./_files/src_peft_tuners_lora_bnb_py.md) |
| ✅ | `src/peft/tuners/lora/config.py` | 879 | LoRA configuration | Workflow: LoRA_Fine_Tuning, QLoRA_Training | [→](./_files/src_peft_tuners_lora_config_py.md) |
| ✅ | `src/peft/tuners/lora/corda.py` | 360 | CorDA covariance init | — | [→](./_files/src_peft_tuners_lora_corda_py.md) |
| ✅ | `src/peft/tuners/lora/dora.py` | 203 | DoRA helpers | Workflow: LoRA_Fine_Tuning | [→](./_files/src_peft_tuners_lora_dora_py.md) |
| ✅ | `src/peft/tuners/lora/eetq.py` | 118 | LoRA EETQ layer | Workflow: QLoRA_Training | [→](./_files/src_peft_tuners_lora_eetq_py.md) |
| ✅ | `src/peft/tuners/lora/eva.py` | 739 | EVA eigenvalue init | Workflow: LoRA_Fine_Tuning | [→](./_files/src_peft_tuners_lora_eva_py.md) |
| ✅ | `src/peft/tuners/lora/gptq.py` | 154 | LoRA GPTQ layer | Workflow: QLoRA_Training | [→](./_files/src_peft_tuners_lora_gptq_py.md) |
| ✅ | `src/peft/tuners/lora/hqq.py` | 251 | LoRA HQQ layer | Workflow: QLoRA_Training | [→](./_files/src_peft_tuners_lora_hqq_py.md) |
| ✅ | `src/peft/tuners/lora/inc.py` | 78 | LoRA Intel FP8 | — | [→](./_files/src_peft_tuners_lora_inc_py.md) |
| ✅ | `src/peft/tuners/lora/layer.py` | 2304 | Core LoRA layers | Workflow: LoRA_Fine_Tuning, QLoRA_Training | [→](./_files/src_peft_tuners_lora_layer_py.md) |
| ✅ | `src/peft/tuners/lora/model.py` | 872 | LoRA model class | Workflow: LoRA_Fine_Tuning, QLoRA_Training, Adapter_Merging | [→](./_files/src_peft_tuners_lora_model_py.md) |
| ✅ | `src/peft/tuners/lora/torchao.py` | 156 | LoRA TorchAO layer | — | [→](./_files/src_peft_tuners_lora_torchao_py.md) |
| ✅ | `src/peft/tuners/lora/tp_layer.py` | 350 | LoRA tensor parallel | — | [→](./_files/src_peft_tuners_lora_tp_layer_py.md) |
| ✅ | `src/peft/tuners/lora/variants.py` | 926 | DoRA, aLoRA variants | Workflow: LoRA_Fine_Tuning | [→](./_files/src_peft_tuners_lora_variants_py.md) |
| ✅ | `src/peft/tuners/lycoris_utils.py` | 263 | LyCORIS base class | — | [→](./_files/src_peft_tuners_lycoris_utils_py.md) |
| ✅ | `src/peft/tuners/miss/__init__.py` | 24 | MiSS exports | — | [→](./_files/src_peft_tuners_miss___init___py.md) |
| ✅ | `src/peft/tuners/miss/config.py` | 140 | MiSS configuration | — | [→](./_files/src_peft_tuners_miss_config_py.md) |
| ✅ | `src/peft/tuners/miss/layer.py` | 393 | MiSS Householder layers | — | [→](./_files/src_peft_tuners_miss_layer_py.md) |
| ✅ | `src/peft/tuners/miss/model.py` | 130 | MiSS model class | — | [→](./_files/src_peft_tuners_miss_model_py.md) |
| ✅ | `src/peft/tuners/mixed/__init__.py` | 18 | Mixed adapter exports | Workflow: Multi_Adapter_Management | [→](./_files/src_peft_tuners_mixed___init___py.md) |
| ✅ | `src/peft/tuners/mixed/model.py` | 309 | Mixed adapter model | Workflow: Multi_Adapter_Management | [→](./_files/src_peft_tuners_mixed_model_py.md) |
| ✅ | `src/peft/tuners/multitask_prompt_tuning/__init__.py` | 25 | MPT exports | — | [→](./_files/src_peft_tuners_multitask_prompt_tuning___init___py.md) |
| ✅ | `src/peft/tuners/multitask_prompt_tuning/config.py` | 62 | MPT configuration | — | [→](./_files/src_peft_tuners_multitask_prompt_tuning_config_py.md) |
| ✅ | `src/peft/tuners/multitask_prompt_tuning/model.py` | 120 | MPT model class | — | [→](./_files/src_peft_tuners_multitask_prompt_tuning_model_py.md) |
| ✅ | `src/peft/tuners/oft/__init__.py` | 52 | OFT exports | — | [→](./_files/src_peft_tuners_oft___init___py.md) |
| ✅ | `src/peft/tuners/oft/aqlm.py` | 105 | OFT AQLM layer | — | [→](./_files/src_peft_tuners_oft_aqlm_py.md) |
| ✅ | `src/peft/tuners/oft/awq.py` | 119 | OFT AWQ layer | — | [→](./_files/src_peft_tuners_oft_awq_py.md) |
| ✅ | `src/peft/tuners/oft/bnb.py` | 388 | OFT 8/4-bit layers | — | [→](./_files/src_peft_tuners_oft_bnb_py.md) |
| ✅ | `src/peft/tuners/oft/config.py` | 213 | OFT configuration | — | [→](./_files/src_peft_tuners_oft_config_py.md) |
| ✅ | `src/peft/tuners/oft/eetq.py` | 116 | OFT EETQ layer | — | [→](./_files/src_peft_tuners_oft_eetq_py.md) |
| ✅ | `src/peft/tuners/oft/gptq.py` | 118 | OFT GPTQ layer | — | [→](./_files/src_peft_tuners_oft_gptq_py.md) |
| ✅ | `src/peft/tuners/oft/hqq.py` | 186 | OFT HQQ layer | — | [→](./_files/src_peft_tuners_oft_hqq_py.md) |
| ✅ | `src/peft/tuners/oft/inc.py` | 78 | OFT Intel FP8 | — | [→](./_files/src_peft_tuners_oft_inc_py.md) |
| ✅ | `src/peft/tuners/oft/layer.py` | 950 | OFT orthogonal layers | — | [→](./_files/src_peft_tuners_oft_layer_py.md) |
| ✅ | `src/peft/tuners/oft/model.py` | 199 | OFT model class | — | [→](./_files/src_peft_tuners_oft_model_py.md) |
| ✅ | `src/peft/tuners/p_tuning/__init__.py` | 24 | P-Tuning exports | — | [→](./_files/src_peft_tuners_p_tuning___init___py.md) |
| ✅ | `src/peft/tuners/p_tuning/config.py` | 61 | P-Tuning config | — | [→](./_files/src_peft_tuners_p_tuning_config_py.md) |
| ✅ | `src/peft/tuners/p_tuning/model.py` | 131 | P-Tuning encoder | — | [→](./_files/src_peft_tuners_p_tuning_model_py.md) |
| ✅ | `src/peft/tuners/poly/__init__.py` | 24 | Poly exports | — | [→](./_files/src_peft_tuners_poly___init___py.md) |
| ✅ | `src/peft/tuners/poly/config.py` | 103 | Poly configuration | — | [→](./_files/src_peft_tuners_poly_config_py.md) |
| ✅ | `src/peft/tuners/poly/layer.py` | 165 | Poly skill layers | — | [→](./_files/src_peft_tuners_poly_layer_py.md) |
| ✅ | `src/peft/tuners/poly/model.py` | 104 | Poly model class | — | [→](./_files/src_peft_tuners_poly_model_py.md) |
| ✅ | `src/peft/tuners/poly/router.py` | 81 | Poly MLP router | — | [→](./_files/src_peft_tuners_poly_router_py.md) |
| ✅ | `src/peft/tuners/prefix_tuning/__init__.py` | 24 | Prefix Tuning exports | — | [→](./_files/src_peft_tuners_prefix_tuning___init___py.md) |
| ✅ | `src/peft/tuners/prefix_tuning/config.py` | 43 | Prefix Tuning config | — | [→](./_files/src_peft_tuners_prefix_tuning_config_py.md) |
| ✅ | `src/peft/tuners/prefix_tuning/model.py` | 81 | Prefix encoder | — | [→](./_files/src_peft_tuners_prefix_tuning_model_py.md) |
| ✅ | `src/peft/tuners/prompt_tuning/__init__.py` | 24 | Prompt Tuning exports | — | [→](./_files/src_peft_tuners_prompt_tuning___init___py.md) |
| ✅ | `src/peft/tuners/prompt_tuning/config.py` | 92 | Prompt Tuning config | — | [→](./_files/src_peft_tuners_prompt_tuning_config_py.md) |
| ✅ | `src/peft/tuners/prompt_tuning/model.py` | 106 | Prompt embedding | — | [→](./_files/src_peft_tuners_prompt_tuning_model_py.md) |
| ✅ | `src/peft/tuners/randlora/__init__.py` | 40 | RandLoRA exports | — | [→](./_files/src_peft_tuners_randlora___init___py.md) |
| ✅ | `src/peft/tuners/randlora/bnb.py` | 456 | RandLoRA quant layers | — | [→](./_files/src_peft_tuners_randlora_bnb_py.md) |
| ✅ | `src/peft/tuners/randlora/config.py` | 199 | RandLoRA configuration | — | [→](./_files/src_peft_tuners_randlora_config_py.md) |
| ✅ | `src/peft/tuners/randlora/layer.py` | 350 | RandLoRA random layers | — | [→](./_files/src_peft_tuners_randlora_layer_py.md) |
| ✅ | `src/peft/tuners/randlora/model.py` | 356 | RandLoRA model class | — | [→](./_files/src_peft_tuners_randlora_model_py.md) |
| ✅ | `src/peft/tuners/road/__init__.py` | 47 | RoAd exports | — | [→](./_files/src_peft_tuners_road___init___py.md) |
| ✅ | `src/peft/tuners/road/bnb.py` | 407 | RoAd quant layers | — | [→](./_files/src_peft_tuners_road_bnb_py.md) |
| ✅ | `src/peft/tuners/road/config.py` | 126 | RoAd configuration | — | [→](./_files/src_peft_tuners_road_config_py.md) |
| ✅ | `src/peft/tuners/road/layer.py` | 418 | RoAd rotation layers | — | [→](./_files/src_peft_tuners_road_layer_py.md) |
| ✅ | `src/peft/tuners/road/model.py` | 163 | RoAd model class | — | [→](./_files/src_peft_tuners_road_model_py.md) |
| ✅ | `src/peft/tuners/shira/__init__.py` | 27 | SHiRA exports | — | [→](./_files/src_peft_tuners_shira___init___py.md) |
| ✅ | `src/peft/tuners/shira/config.py` | 129 | SHiRA configuration | — | [→](./_files/src_peft_tuners_shira_config_py.md) |
| ✅ | `src/peft/tuners/shira/layer.py` | 217 | SHiRA sparse layers | — | [→](./_files/src_peft_tuners_shira_layer_py.md) |
| ✅ | `src/peft/tuners/shira/mask_functions.py` | 72 | SHiRA mask utils | — | [→](./_files/src_peft_tuners_shira_mask_functions_py.md) |
| ✅ | `src/peft/tuners/shira/model.py` | 142 | SHiRA model class | — | [→](./_files/src_peft_tuners_shira_model_py.md) |
| ✅ | `src/peft/tuners/trainable_tokens/__init__.py` | 24 | Trainable Tokens exports | — | [→](./_files/src_peft_tuners_trainable_tokens___init___py.md) |
| ✅ | `src/peft/tuners/trainable_tokens/config.py` | 45 | Trainable Tokens config | — | [→](./_files/src_peft_tuners_trainable_tokens_config_py.md) |
| ✅ | `src/peft/tuners/trainable_tokens/layer.py` | 65 | Trainable Tokens layer | — | [→](./_files/src_peft_tuners_trainable_tokens_layer_py.md) |
| ✅ | `src/peft/tuners/trainable_tokens/model.py` | 127 | Trainable Tokens model | — | [→](./_files/src_peft_tuners_trainable_tokens_model_py.md) |
| ✅ | `src/peft/tuners/tuners_utils.py` | 2041 | Base tuner utilities | Workflow: LoRA_Fine_Tuning | [→](./_files/src_peft_tuners_tuners_utils_py.md) |
| ✅ | `src/peft/tuners/vblora/__init__.py` | 24 | VBLoRA exports | — | [→](./_files/src_peft_tuners_vblora___init___py.md) |
| ✅ | `src/peft/tuners/vblora/config.py` | 196 | VBLoRA configuration | — | [→](./_files/src_peft_tuners_vblora_config_py.md) |
| ✅ | `src/peft/tuners/vblora/layer.py` | 251 | VBLoRA bank layers | — | [→](./_files/src_peft_tuners_vblora_layer_py.md) |
| ✅ | `src/peft/tuners/vblora/model.py` | 209 | VBLoRA model class | — | [→](./_files/src_peft_tuners_vblora_model_py.md) |
| ✅ | `src/peft/tuners/vera/__init__.py` | 40 | VeRA exports | — | [→](./_files/src_peft_tuners_vera___init___py.md) |
| ✅ | `src/peft/tuners/vera/bnb.py` | 411 | VeRA quant layers | — | [→](./_files/src_peft_tuners_vera_bnb_py.md) |
| ✅ | `src/peft/tuners/vera/config.py` | 162 | VeRA configuration | — | [→](./_files/src_peft_tuners_vera_config_py.md) |
| ✅ | `src/peft/tuners/vera/layer.py` | 291 | VeRA diagonal layers | — | [→](./_files/src_peft_tuners_vera_layer_py.md) |
| ✅ | `src/peft/tuners/vera/model.py` | 294 | VeRA model class | — | [→](./_files/src_peft_tuners_vera_model_py.md) |
| ✅ | `src/peft/tuners/xlora/__init__.py` | 24 | X-LoRA exports | — | [→](./_files/src_peft_tuners_xlora___init___py.md) |
| ✅ | `src/peft/tuners/xlora/classifier.py` | 189 | X-LoRA MLP classifier | — | [→](./_files/src_peft_tuners_xlora_classifier_py.md) |
| ✅ | `src/peft/tuners/xlora/config.py` | 120 | X-LoRA configuration | — | [→](./_files/src_peft_tuners_xlora_config_py.md) |
| ✅ | `src/peft/tuners/xlora/layer.py` | 189 | X-LoRA weighted layers | — | [→](./_files/src_peft_tuners_xlora_layer_py.md) |
| ✅ | `src/peft/tuners/xlora/model.py` | 371 | X-LoRA model class | — | [→](./_files/src_peft_tuners_xlora_model_py.md) |
| ✅ | `src/peft/utils/__init__.py` | 133 | Utils package exports | Workflow: LoRA_Fine_Tuning, QLoRA_Training | [→](./_files/src_peft_utils___init___py.md) |
| ✅ | `src/peft/utils/constants.py` | 362 | Default target modules | Workflow: LoRA_Fine_Tuning | [→](./_files/src_peft_utils_constants_py.md) |
| ✅ | `src/peft/utils/hotswap.py` | 630 | Runtime adapter swap | Workflow: Multi_Adapter_Management | [→](./_files/src_peft_utils_hotswap_py.md) |
| ✅ | `src/peft/utils/incremental_pca.py` | 338 | GPU IncrementalPCA | — | [→](./_files/src_peft_utils_incremental_pca_py.md) |
| ✅ | `src/peft/utils/integrations.py` | 291 | External lib compat | Workflow: QLoRA_Training | [→](./_files/src_peft_utils_integrations_py.md) |
| ✅ | `src/peft/utils/loftq_utils.py` | 410 | LoftQ quant init | Workflow: QLoRA_Training | [→](./_files/src_peft_utils_loftq_utils_py.md) |
| ✅ | `src/peft/utils/merge_utils.py` | 268 | Adapter merge algos | Workflow: Adapter_Merging | [→](./_files/src_peft_utils_merge_utils_py.md) |
| ✅ | `src/peft/utils/other.py` | 1648 | Core utilities | Workflow: LoRA_Fine_Tuning, QLoRA_Training | [→](./_files/src_peft_utils_other_py.md) |
| ✅ | `src/peft/utils/peft_types.py` | 183 | PEFT type enums | Workflow: LoRA_Fine_Tuning | [→](./_files/src_peft_utils_peft_types_py.md) |
| ✅ | `src/peft/utils/save_and_load.py` | 724 | Adapter serialization | Workflow: LoRA_Fine_Tuning, Adapter_Loading_Inference | [→](./_files/src_peft_utils_save_and_load_py.md) |
| ✅ | `src/peft/utils/warning.py` | 17 | Custom warning class | — | [→](./_files/src_peft_utils_warning_py.md) |

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
