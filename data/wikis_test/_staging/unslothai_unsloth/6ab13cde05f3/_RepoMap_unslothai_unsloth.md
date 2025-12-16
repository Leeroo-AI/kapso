# Repository Map: unslothai_unsloth

> **Compact index** of repository files.
> Each file has a detail page in `_files/` with Understanding to fill.
> Mark files as ✅ explored in the table below as you complete them.

| Property | Value |
|----------|-------|
| Repository | https://github.com/unslothai/unsloth |
| Branch | main |
| Generated | 2025-12-16 17:30 |
| Python Files | 116 |
| Total Lines | 50,480 |
| Explored | 116/116 |

## Structure

📦 **Packages:** unsloth
📝 **Examples:** scripts
🧪 **Tests:** tests

📖 README: `README.md`
⚙️ Setup: `pyproject.toml`

---

## 📦 Package Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `unsloth/__init__.py` | 280 | Package init and import orchestration | Workflow: unslothai_unsloth_QLoRA_Finetuning, unslothai_unsloth_Vision_Language_Model_Finetuning, unslothai_unsloth_GRPO_Reinforcement_Learning | [→](./_files/unsloth___init___py.md) |
| ✅ | `unsloth/_auto_install.py` | 41 | Installation script for torch deps | — | [→](./_files/unsloth__auto_install_py.md) |
| ✅ | `unsloth/chat_templates.py` | 3159 | Chat template formats for models | Workflow: unslothai_unsloth_QLoRA_Finetuning | [→](./_files/unsloth_chat_templates_py.md) |
| ✅ | `unsloth/dataprep/__init__.py` | 15 | Data preparation module init | — | [→](./_files/unsloth_dataprep___init___py.md) |
| ✅ | `unsloth/dataprep/synthetic.py` | 465 | Synthetic data generation framework | — | [→](./_files/unsloth_dataprep_synthetic_py.md) |
| ✅ | `unsloth/dataprep/synthetic_configs.py` | 111 | Synthetic data generation config | — | [→](./_files/unsloth_dataprep_synthetic_configs_py.md) |
| ✅ | `unsloth/device_type.py` | 98 | Device type detection and capability | — | [→](./_files/unsloth_device_type_py.md) |
| ✅ | `unsloth/import_fixes.py` | 426 | Runtime compatibility patches | — | [→](./_files/unsloth_import_fixes_py.md) |
| ✅ | `unsloth/kernels/__init__.py` | 73 | Module exports and initialization | — | [→](./_files/unsloth_kernels___init___py.md) |
| ✅ | `unsloth/kernels/cross_entropy_loss.py` | 459 | Optimized cross-entropy loss | Workflow: unslothai_unsloth_QLoRA_Finetuning | [→](./_files/unsloth_kernels_cross_entropy_loss_py.md) |
| ✅ | `unsloth/kernels/fast_lora.py` | 717 | Efficient LoRA forward/backward | Workflow: unslothai_unsloth_QLoRA_Finetuning, unslothai_unsloth_Vision_Language_Model_Finetuning | [→](./_files/unsloth_kernels_fast_lora_py.md) |
| ✅ | `unsloth/kernels/flex_attention.py` | 187 | Flexible attention with softcapping | — | [→](./_files/unsloth_kernels_flex_attention_py.md) |
| ✅ | `unsloth/kernels/fp8.py` | 599 | FP8 quantization and inference | — | [→](./_files/unsloth_kernels_fp8_py.md) |
| ✅ | `unsloth/kernels/geglu.py` | 290 | GeGLU activation function kernels | — | [→](./_files/unsloth_kernels_geglu_py.md) |
| ✅ | `unsloth/kernels/layernorm.py` | 225 | Optimized LayerNorm computation | — | [→](./_files/unsloth_kernels_layernorm_py.md) |
| ✅ | `unsloth/kernels/moe/__init__.py` | 0 | Package initialization marker | — | [→](./_files/unsloth_kernels_moe___init___py.md) |
| ✅ | `unsloth/kernels/moe/benchmark/benchmark_fused_moe.py` | 399 | Performance benchmarking for MoE | — | [→](./_files/unsloth_kernels_moe_benchmark_benchmark_fused_moe_py.md) |
| ✅ | `unsloth/kernels/moe/benchmark/utils.py` | 228 | Benchmark utilities and processing | — | [→](./_files/unsloth_kernels_moe_benchmark_utils_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/__init__.py` | 0 | Package initialization marker | — | [→](./_files/unsloth_kernels_moe_grouped_gemm___init___py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/interface.py` | 968 | High-level grouped GEMM API | — | [→](./_files/unsloth_kernels_moe_grouped_gemm_interface_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/kernels/__init__.py` | 0 | Package initialization marker | — | [→](./_files/unsloth_kernels_moe_grouped_gemm_kernels___init___py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/kernels/autotuning.py` | 396 | Triton config generation/pruning | — | [→](./_files/unsloth_kernels_moe_grouped_gemm_kernels_autotuning_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/kernels/backward.py` | 502 | Backward pass GEMM kernels | — | [→](./_files/unsloth_kernels_moe_grouped_gemm_kernels_backward_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/kernels/forward.py` | 265 | Forward pass GEMM kernel | — | [→](./_files/unsloth_kernels_moe_grouped_gemm_kernels_forward_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/kernels/tuning.py` | 277 | Kernel configuration dataclasses | — | [→](./_files/unsloth_kernels_moe_grouped_gemm_kernels_tuning_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/reference/__init__.py` | 0 | Package initialization marker | — | [→](./_files/unsloth_kernels_moe_grouped_gemm_reference___init___py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/reference/layers/llama4_moe.py` | 437 | Llama4 MoE block implementations | — | [→](./_files/unsloth_kernels_moe_grouped_gemm_reference_layers_llama4_moe_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/reference/layers/qwen3_moe.py` | 348 | Qwen3 MoE block implementations | — | [→](./_files/unsloth_kernels_moe_grouped_gemm_reference_layers_qwen3_moe_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/reference/moe_block.py` | 161 | Base MoE block utilities | — | [→](./_files/unsloth_kernels_moe_grouped_gemm_reference_moe_block_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/reference/moe_ops.py` | 151 | Low-level MoE operations | — | [→](./_files/unsloth_kernels_moe_grouped_gemm_reference_moe_ops_py.md) |
| ✅ | `unsloth/kernels/moe/tests/__init__.py` | 0 | Test package marker | — | [→](./_files/unsloth_kernels_moe_tests___init___py.md) |
| ✅ | `unsloth/kernels/moe/tests/common.py` | 336 | Test utilities and configurations | — | [→](./_files/unsloth_kernels_moe_tests_common_py.md) |
| ✅ | `unsloth/kernels/moe/tests/moe_utils.py` | 507 | Qwen3 MoE testing utilities | — | [→](./_files/unsloth_kernels_moe_tests_moe_utils_py.md) |
| ✅ | `unsloth/kernels/moe/tests/test_grouped_gemm.py` | 1213 | Comprehensive grouped GEMM tests | — | [→](./_files/unsloth_kernels_moe_tests_test_grouped_gemm_py.md) |
| ✅ | `unsloth/kernels/moe/tests/test_llama4_moe.py` | 262 | Llama4 MoE validation tests | — | [→](./_files/unsloth_kernels_moe_tests_test_llama4_moe_py.md) |
| ✅ | `unsloth/kernels/moe/tests/test_qwen3_moe.py` | 273 | Qwen3 MoE validation tests | — | [→](./_files/unsloth_kernels_moe_tests_test_qwen3_moe_py.md) |
| ✅ | `unsloth/kernels/rms_layernorm.py` | 335 | RMS LayerNorm for transformers | — | [→](./_files/unsloth_kernels_rms_layernorm_py.md) |
| ✅ | `unsloth/kernels/rope_embedding.py` | 465 | Rotary position embeddings | — | [→](./_files/unsloth_kernels_rope_embedding_py.md) |
| ✅ | `unsloth/kernels/swiglu.py` | 143 | SwiGLU activation kernels | — | [→](./_files/unsloth_kernels_swiglu_py.md) |
| ✅ | `unsloth/kernels/utils.py` | 1034 | Shared utilities and helpers | — | [→](./_files/unsloth_kernels_utils_py.md) |
| ✅ | `unsloth/models/__init__.py` | 30 | Model package exports | Workflow: unslothai_unsloth_QLoRA_Finetuning | [→](./_files/unsloth_models___init___py.md) |
| ✅ | `unsloth/models/_utils.py` | 2356 | Model utilities and optimization | — | [→](./_files/unsloth_models__utils_py.md) |
| ✅ | `unsloth/models/cohere.py` | 528 | Cohere model with QK norm | — | [→](./_files/unsloth_models_cohere_py.md) |
| ✅ | `unsloth/models/dpo.py` | 26 | DPO trainer compatibility stub | — | [→](./_files/unsloth_models_dpo_py.md) |
| ✅ | `unsloth/models/falcon_h1.py` | 764 | FalconH1 hybrid attention-SSM | — | [→](./_files/unsloth_models_falcon_h1_py.md) |
| ✅ | `unsloth/models/gemma.py` | 476 | Gemma model optimization | — | [→](./_files/unsloth_models_gemma_py.md) |
| ✅ | `unsloth/models/gemma2.py` | 656 | Gemma2 with logit softcapping | — | [→](./_files/unsloth_models_gemma2_py.md) |
| ✅ | `unsloth/models/granite.py` | 612 | Granite model with residual scaling | — | [→](./_files/unsloth_models_granite_py.md) |
| ✅ | `unsloth/models/llama.py` | 3400 | Core attention/transformer optimization | Workflow: unslothai_unsloth_QLoRA_Finetuning | [→](./_files/unsloth_models_llama_py.md) |
| ✅ | `unsloth/models/llama4.py` | 16 | Llama4 placeholder stub | — | [→](./_files/unsloth_models_llama4_py.md) |
| ✅ | `unsloth/models/loader.py` | 1262 | Model loading and optimization | Workflow: unslothai_unsloth_QLoRA_Finetuning, unslothai_unsloth_GRPO_Reinforcement_Learning | [→](./_files/unsloth_models_loader_py.md) |
| ✅ | `unsloth/models/loader_utils.py` | 423 | Model loading/distributed setup | — | [→](./_files/unsloth_models_loader_utils_py.md) |
| ✅ | `unsloth/models/mapper.py` | 1324 | Model name mapping for quantization | — | [→](./_files/unsloth_models_mapper_py.md) |
| ✅ | `unsloth/models/mistral.py` | 467 | Mistral with sliding windows | — | [→](./_files/unsloth_models_mistral_py.md) |
| ✅ | `unsloth/models/qwen2.py` | 101 | Qwen2 lightweight wrapper | — | [→](./_files/unsloth_models_qwen2_py.md) |
| ✅ | `unsloth/models/qwen3.py` | 457 | Qwen3 with QK normalization | — | [→](./_files/unsloth_models_qwen3_py.md) |
| ✅ | `unsloth/models/qwen3_moe.py` | 243 | Qwen3 mixture-of-experts | — | [→](./_files/unsloth_models_qwen3_moe_py.md) |
| ✅ | `unsloth/models/rl.py` | 1349 | RL trainer patches | Workflow: unslothai_unsloth_GRPO_Reinforcement_Learning | [→](./_files/unsloth_models_rl_py.md) |
| ✅ | `unsloth/models/rl_replacements.py` | 985 | RL trainer function replacements | Workflow: unslothai_unsloth_GRPO_Reinforcement_Learning | [→](./_files/unsloth_models_rl_replacements_py.md) |
| ✅ | `unsloth/models/vision.py` | 1263 | Vision language model framework | Workflow: unslothai_unsloth_Vision_Language_Model_Finetuning | [→](./_files/unsloth_models_vision_py.md) |
| ✅ | `unsloth/ollama_template_mappers.py` | 2192 | Ollama template mappings | Workflow: unslothai_unsloth_GGUF_Export | [→](./_files/unsloth_ollama_template_mappers_py.md) |
| ✅ | `unsloth/registry/__init__.py` | 78 | Model registry init and search | — | [→](./_files/unsloth_registry___init___py.md) |
| ✅ | `unsloth/registry/_deepseek.py` | 206 | Deepseek model registry | — | [→](./_files/unsloth_registry__deepseek_py.md) |
| ✅ | `unsloth/registry/_gemma.py` | 74 | Gemma model registry | — | [→](./_files/unsloth_registry__gemma_py.md) |
| ✅ | `unsloth/registry/_llama.py` | 125 | Llama model registry | — | [→](./_files/unsloth_registry__llama_py.md) |
| ✅ | `unsloth/registry/_mistral.py` | 88 | Mistral model registry | — | [→](./_files/unsloth_registry__mistral_py.md) |
| ✅ | `unsloth/registry/_phi.py` | 74 | Phi model registry | — | [→](./_files/unsloth_registry__phi_py.md) |
| ✅ | `unsloth/registry/_qwen.py` | 136 | Qwen model registry | — | [→](./_files/unsloth_registry__qwen_py.md) |
| ✅ | `unsloth/registry/registry.py` | 191 | Core model registry infrastructure | — | [→](./_files/unsloth_registry_registry_py.md) |
| ✅ | `unsloth/save.py` | 3068 | Model saving and deployment | Workflow: unslothai_unsloth_QLoRA_Finetuning, unslothai_unsloth_GGUF_Export | [→](./_files/unsloth_save_py.md) |
| ✅ | `unsloth/tokenizer_utils.py` | 1105 | Tokenizer conversion/validation | Workflow: unslothai_unsloth_GGUF_Export | [→](./_files/unsloth_tokenizer_utils_py.md) |
| ✅ | `unsloth/trainer.py` | 437 | Custom training arguments/trainer | Workflow: unslothai_unsloth_QLoRA_Finetuning, unslothai_unsloth_Vision_Language_Model_Finetuning | [→](./_files/unsloth_trainer_py.md) |
| ✅ | `unsloth/utils/__init__.py` | 48 | Utilities module exports | — | [→](./_files/unsloth_utils___init___py.md) |
| ✅ | `unsloth/utils/attention_dispatch.py` | 283 | Attention backend selection | — | [→](./_files/unsloth_utils_attention_dispatch_py.md) |
| ✅ | `unsloth/utils/hf_hub.py` | 78 | HuggingFace Hub discovery | — | [→](./_files/unsloth_utils_hf_hub_py.md) |
| ✅ | `unsloth/utils/packing.py` | 344 | Padding-free sequence packing | Workflow: unslothai_unsloth_QLoRA_Finetuning | [→](./_files/unsloth_utils_packing_py.md) |

## 📝 Example Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `scripts/enforce_kwargs_spacing.py` | 179 | Enforce kwarg spacing rules | — | [→](./_files/scripts_enforce_kwargs_spacing_py.md) |
| ✅ | `scripts/run_ruff_format.py` | 30 | Execute ruff formatter | — | [→](./_files/scripts_run_ruff_format_py.md) |

## 🧪 Test Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `tests/__init__.py` | 0 | Empty test package init | — | [→](./_files/tests___init___py.md) |
| ✅ | `tests/qlora/test_hf_qlora_train_and_merge.py` | 159 | HuggingFace QLoRA test | Workflow: unslothai_unsloth_QLoRA_Finetuning | [→](./_files/tests_qlora_test_hf_qlora_train_and_merge_py.md) |
| ✅ | `tests/qlora/test_unsloth_qlora_train_and_merge.py` | 211 | Unsloth QLoRA test | Workflow: unslothai_unsloth_QLoRA_Finetuning | [→](./_files/tests_qlora_test_unsloth_qlora_train_and_merge_py.md) |
| ✅ | `tests/saving/gpt-oss-merge/test_merged_model.py` | 60 | Merged model inference test | Workflow: unslothai_unsloth_GGUF_Export | [→](./_files/tests_saving_gpt-oss-merge_test_merged_model_py.md) |
| ✅ | `tests/saving/gpt-oss-merge/train_and_merge.py` | 102 | Train and merge workflow | Workflow: unslothai_unsloth_QLoRA_Finetuning, unslothai_unsloth_GGUF_Export | [→](./_files/tests_saving_gpt-oss-merge_train_and_merge_py.md) |
| ✅ | `tests/saving/language_models/test_merge_4bit_validation.py` | 248 | 4-bit merge validation | Workflow: unslothai_unsloth_GGUF_Export | [→](./_files/tests_saving_language_models_test_merge_4bit_validation_py.md) |
| ✅ | `tests/saving/language_models/test_merge_model_perplexity_llama-3.2.py` | 259 | Llama-3.2 perplexity test | Workflow: unslothai_unsloth_QLoRA_Finetuning | [→](./_files/tests_saving_language_models_test_merge_model_perplexity_llama-3_2_py.md) |
| ✅ | `tests/saving/language_models/test_merge_model_perplexity_mistral.py` | 318 | Mistral perplexity test | Workflow: unslothai_unsloth_QLoRA_Finetuning | [→](./_files/tests_saving_language_models_test_merge_model_perplexity_mistral_py.md) |
| ✅ | `tests/saving/language_models/test_merge_model_perplexity_phi_4.py` | 259 | Phi-4 perplexity test | Workflow: unslothai_unsloth_QLoRA_Finetuning | [→](./_files/tests_saving_language_models_test_merge_model_perplexity_phi_4_py.md) |
| ✅ | `tests/saving/language_models/test_merged_model_perplexity_llama-3.1-8b.py` | 263 | Llama-3.1-8B perplexity test | Workflow: unslothai_unsloth_QLoRA_Finetuning | [→](./_files/tests_saving_language_models_test_merged_model_perplexity_llama-3_1-8b_py.md) |
| ✅ | `tests/saving/language_models/test_merged_model_perplexity_qwen_2.5.py` | 311 | Qwen2.5 perplexity test | Workflow: unslothai_unsloth_QLoRA_Finetuning | [→](./_files/tests_saving_language_models_test_merged_model_perplexity_qwen_2_5_py.md) |
| ✅ | `tests/saving/language_models/test_push_to_hub_merged.py` | 204 | Push merged model to Hub | Workflow: unslothai_unsloth_GGUF_Export | [→](./_files/tests_saving_language_models_test_push_to_hub_merged_py.md) |
| ✅ | `tests/saving/language_models/test_push_to_hub_merged_sharded_index_file.py` | 223 | Push sharded model to Hub | Workflow: unslothai_unsloth_GGUF_Export | [→](./_files/tests_saving_language_models_test_push_to_hub_merged_sharded_index_file_py.md) |
| ✅ | `tests/saving/language_models/test_save_merged_grpo_model.py` | 825 | GRPO model save and eval | Workflow: unslothai_unsloth_GRPO_Reinforcement_Learning | [→](./_files/tests_saving_language_models_test_save_merged_grpo_model_py.md) |
| ✅ | `tests/saving/non_peft/test_mistral_non_peft.py` | 65 | Non-PEFT Mistral save test | — | [→](./_files/tests_saving_non_peft_test_mistral_non_peft_py.md) |
| ✅ | `tests/saving/non_peft/test_whisper_non_peft.py` | 65 | Non-PEFT Whisper save test | — | [→](./_files/tests_saving_non_peft_test_whisper_non_peft_py.md) |
| ✅ | `tests/saving/test_unsloth_save.py` | 401 | Model save methods test suite | Workflow: unslothai_unsloth_GGUF_Export | [→](./_files/tests_saving_test_unsloth_save_py.md) |
| ✅ | `tests/saving/text_to_speech_models/test_csm.py` | 168 | CSM TTS model test | — | [→](./_files/tests_saving_text_to_speech_models_test_csm_py.md) |
| ✅ | `tests/saving/text_to_speech_models/test_lasa.py` | 220 | Llasa TTS model test | — | [→](./_files/tests_saving_text_to_speech_models_test_lasa_py.md) |
| ✅ | `tests/saving/text_to_speech_models/test_orpheus.py` | 282 | Orpheus TTS model test | — | [→](./_files/tests_saving_text_to_speech_models_test_orpheus_py.md) |
| ✅ | `tests/saving/text_to_speech_models/test_whisper.py` | 195 | Whisper STT model test | — | [→](./_files/tests_saving_text_to_speech_models_test_whisper_py.md) |
| ✅ | `tests/saving/vision_models/test_index_file_sharded_model.py` | 293 | Vision model sharded save | Workflow: unslothai_unsloth_Vision_Language_Model_Finetuning | [→](./_files/tests_saving_vision_models_test_index_file_sharded_model_py.md) |
| ✅ | `tests/saving/vision_models/test_push_to_hub_merged.py` | 273 | Vision model Hub upload | Workflow: unslothai_unsloth_Vision_Language_Model_Finetuning | [→](./_files/tests_saving_vision_models_test_push_to_hub_merged_py.md) |
| ✅ | `tests/saving/vision_models/test_save_merge_qwen2.5vl32B_model_ocr_benchmark.py` | 287 | Large VLM OCR benchmark | Workflow: unslothai_unsloth_Vision_Language_Model_Finetuning | [→](./_files/tests_saving_vision_models_test_save_merge_qwen2_5vl32B_model_ocr_benchmark_py.md) |
| ✅ | `tests/saving/vision_models/test_save_merge_vision_model_ocr_benchmark.py` | 287 | Vision model OCR eval | Workflow: unslothai_unsloth_Vision_Language_Model_Finetuning | [→](./_files/tests_saving_vision_models_test_save_merge_vision_model_ocr_benchmark_py.md) |
| ✅ | `tests/test_model_registry.py` | 92 | Model registry validation | — | [→](./_files/tests_test_model_registry_py.md) |
| ✅ | `tests/utils/__init__.py` | 33 | Test utilities module | — | [→](./_files/tests_utils___init___py.md) |
| ✅ | `tests/utils/aime_eval.py` | 545 | AIME dataset evaluation | Workflow: unslothai_unsloth_GRPO_Reinforcement_Learning | [→](./_files/tests_utils_aime_eval_py.md) |
| ✅ | `tests/utils/cleanup_utils.py` | 226 | Memory and directory cleanup | — | [→](./_files/tests_utils_cleanup_utils_py.md) |
| ✅ | `tests/utils/data_utils.py` | 153 | Dataset and model analysis | Workflow: unslothai_unsloth_QLoRA_Finetuning | [→](./_files/tests_utils_data_utils_py.md) |
| ✅ | `tests/utils/hf_utils.py` | 291 | HuggingFace model utilities | — | [→](./_files/tests_utils_hf_utils_py.md) |
| ✅ | `tests/utils/ocr_eval.py` | 374 | OCR model evaluation | Workflow: unslothai_unsloth_Vision_Language_Model_Finetuning | [→](./_files/tests_utils_ocr_eval_py.md) |
| ✅ | `tests/utils/os_utils.py` | 128 | System package detection | — | [→](./_files/tests_utils_os_utils_py.md) |
| ✅ | `tests/utils/perplexity_eval.py` | 81 | Perplexity evaluation | Workflow: unslothai_unsloth_QLoRA_Finetuning | [→](./_files/tests_utils_perplexity_eval_py.md) |
| ✅ | `tests/utils/test_attention_masks.py` | 272 | Packed attention mask tests | — | [→](./_files/tests_utils_test_attention_masks_py.md) |
| ✅ | `tests/utils/test_packing.py` | 391 | Sequence packing tests | Workflow: unslothai_unsloth_QLoRA_Finetuning | [→](./_files/tests_utils_test_packing_py.md) |
| ✅ | `tests/utils/test_qat.py` | 156 | Quantization-aware training tests | — | [→](./_files/tests_utils_test_qat_py.md) |

## 📄 Other Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `unsloth-cli.py` | 441 | Fine-tuning CLI script | Workflow: unslothai_unsloth_QLoRA_Finetuning | [→](./_files/unsloth-cli_py.md) |

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
