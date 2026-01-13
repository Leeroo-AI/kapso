# Repository Map: Unslothai_Unsloth

> **Compact index** of repository files.
> Each file has a detail page in `_files/` with Understanding to fill.
> Mark files as ✅ explored in the table below as you complete them.

| Property | Value |
|----------|-------|
| Repository | https://github.com/unslothai/unsloth |
| Branch | main |
| Generated | 2026-01-12 19:36 |
| Python Files | 118 |
| Total Lines | 51,822 |
| Explored | 118/118 |

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
| ✅ | `unsloth/__init__.py` | 295 | Package entry point exports | Workflow: QLoRA_Finetuning, GRPO_Reinforcement_Learning, Vision_Model_Finetuning | [→](./_files/unsloth___init___py.md) |
| ✅ | `unsloth/_auto_install.py` | 41 | Auto dependency installer | — | [→](./_files/unsloth__auto_install_py.md) |
| ✅ | `unsloth/chat_templates.py` | 3159 | Chat template management | Workflow: QLoRA_Finetuning, GRPO_Reinforcement_Learning | [→](./_files/unsloth_chat_templates_py.md) |
| ✅ | `unsloth/dataprep/__init__.py` | 16 | Dataprep package init | — | [→](./_files/unsloth_dataprep___init___py.md) |
| ✅ | `unsloth/dataprep/raw_text.py` | 348 | Raw text data loader | Impl: RawTextDataLoader | [→](./_files/unsloth_dataprep_raw_text_py.md) |
| ✅ | `unsloth/dataprep/synthetic.py` | 465 | Synthetic data generator | Impl: SyntheticDataKit | [→](./_files/unsloth_dataprep_synthetic_py.md) |
| ✅ | `unsloth/dataprep/synthetic_configs.py` | 111 | Synthetic data configs | — | [→](./_files/unsloth_dataprep_synthetic_configs_py.md) |
| ✅ | `unsloth/device_type.py` | 127 | Hardware device detection | Impl: Device_Type | [→](./_files/unsloth_device_type_py.md) |
| ✅ | `unsloth/import_fixes.py` | 695 | Library compatibility patches | Impl: Import_Fixes | [→](./_files/unsloth_import_fixes_py.md) |
| ✅ | `unsloth/kernels/__init__.py` | 73 | Kernels package init | — | [→](./_files/unsloth_kernels___init___py.md) |
| ✅ | `unsloth/kernels/cross_entropy_loss.py` | 459 | Triton cross-entropy kernel | Workflow: QLoRA_Finetuning, GRPO_Reinforcement_Learning | [→](./_files/unsloth_kernels_cross_entropy_loss_py.md) |
| ✅ | `unsloth/kernels/fast_lora.py` | 730 | Fused LoRA operations | Workflow: QLoRA_Finetuning, GRPO_Reinforcement_Learning, Vision_Model_Finetuning | [→](./_files/unsloth_kernels_fast_lora_py.md) |
| ✅ | `unsloth/kernels/flex_attention.py` | 187 | Softcapping attention impl | Impl: Flex_Attention | [→](./_files/unsloth_kernels_flex_attention_py.md) |
| ✅ | `unsloth/kernels/fp8.py` | 615 | FP8 quantization kernels | Impl: FP8_Kernels | [→](./_files/unsloth_kernels_fp8_py.md) |
| ✅ | `unsloth/kernels/geglu.py` | 290 | GEGLU activation kernel | Impl: GEGLU_Kernels | [→](./_files/unsloth_kernels_geglu_py.md) |
| ✅ | `unsloth/kernels/layernorm.py` | 225 | LayerNorm Triton kernel | Impl: LayerNorm_Kernel | [→](./_files/unsloth_kernels_layernorm_py.md) |
| ✅ | `unsloth/kernels/moe/__init__.py` | 0 | MoE kernels package init | — | [→](./_files/unsloth_kernels_moe___init___py.md) |
| ✅ | `unsloth/kernels/moe/benchmark/benchmark_fused_moe.py` | 399 | MoE kernel benchmark | — | [→](./_files/unsloth_kernels_moe_benchmark_benchmark_fused_moe_py.md) |
| ✅ | `unsloth/kernels/moe/benchmark/utils.py` | 228 | Benchmark utilities | — | [→](./_files/unsloth_kernels_moe_benchmark_utils_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/__init__.py` | 0 | Grouped GEMM package init | — | [→](./_files/unsloth_kernels_moe_grouped_gemm___init___py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/interface.py` | 968 | Grouped GEMM main API | Impl: Grouped_GEMM_Interface | [→](./_files/unsloth_kernels_moe_grouped_gemm_interface_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/kernels/__init__.py` | 0 | GEMM kernels package init | — | [→](./_files/unsloth_kernels_moe_grouped_gemm_kernels___init___py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/kernels/autotuning.py` | 396 | GEMM kernel autotuning | Impl: GEMM_Autotuning | [→](./_files/unsloth_kernels_moe_grouped_gemm_kernels_autotuning_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/kernels/backward.py` | 502 | GEMM backward kernels | Impl: GEMM_Backward | [→](./_files/unsloth_kernels_moe_grouped_gemm_kernels_backward_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/kernels/forward.py` | 265 | GEMM forward kernel | Impl: GEMM_Forward | [→](./_files/unsloth_kernels_moe_grouped_gemm_kernels_forward_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/kernels/tuning.py` | 277 | Manual kernel tuning | Impl: GEMM_Tuning | [→](./_files/unsloth_kernels_moe_grouped_gemm_kernels_tuning_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/reference/__init__.py` | 0 | Reference impl package init | — | [→](./_files/unsloth_kernels_moe_grouped_gemm_reference___init___py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/reference/layers/llama4_moe.py` | 437 | Llama4 MoE reference | Impl: Llama4_MoE_Layer | [→](./_files/unsloth_kernels_moe_grouped_gemm_reference_layers_llama4_moe_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/reference/layers/qwen3_moe.py` | 348 | Qwen3 MoE reference | Impl: Qwen3_MoE_Layer | [→](./_files/unsloth_kernels_moe_grouped_gemm_reference_layers_qwen3_moe_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/reference/moe_block.py` | 161 | Triton MoE block impl | Impl: MoE_Block | [→](./_files/unsloth_kernels_moe_grouped_gemm_reference_moe_block_py.md) |
| ✅ | `unsloth/kernels/moe/grouped_gemm/reference/moe_ops.py` | 151 | Common MoE operations | Impl: MoE_Ops | [→](./_files/unsloth_kernels_moe_grouped_gemm_reference_moe_ops_py.md) |
| ✅ | `unsloth/kernels/moe/tests/__init__.py` | 0 | MoE tests package init | — | [→](./_files/unsloth_kernels_moe_tests___init___py.md) |
| ✅ | `unsloth/kernels/moe/tests/common.py` | 336 | MoE test infrastructure | — | [→](./_files/unsloth_kernels_moe_tests_common_py.md) |
| ✅ | `unsloth/kernels/moe/tests/moe_utils.py` | 507 | MoE test utilities | — | [→](./_files/unsloth_kernels_moe_tests_moe_utils_py.md) |
| ✅ | `unsloth/kernels/moe/tests/test_grouped_gemm.py` | 1213 | Grouped GEMM tests | — | [→](./_files/unsloth_kernels_moe_tests_test_grouped_gemm_py.md) |
| ✅ | `unsloth/kernels/moe/tests/test_llama4_moe.py` | 262 | Llama4 MoE tests | — | [→](./_files/unsloth_kernels_moe_tests_test_llama4_moe_py.md) |
| ✅ | `unsloth/kernels/moe/tests/test_qwen3_moe.py` | 273 | Qwen3 MoE tests | — | [→](./_files/unsloth_kernels_moe_tests_test_qwen3_moe_py.md) |
| ✅ | `unsloth/kernels/rms_layernorm.py` | 335 | RMSNorm Triton kernel | Impl: RMSNorm_Kernel | [→](./_files/unsloth_kernels_rms_layernorm_py.md) |
| ✅ | `unsloth/kernels/rope_embedding.py` | 465 | RoPE embedding kernel | Impl: RoPE_Kernel | [→](./_files/unsloth_kernels_rope_embedding_py.md) |
| ✅ | `unsloth/kernels/swiglu.py` | 143 | SwiGLU activation kernel | Impl: SwiGLU_Kernel | [→](./_files/unsloth_kernels_swiglu_py.md) |
| ✅ | `unsloth/kernels/utils.py` | 1034 | Kernel utilities | Impl: Kernel_Utils | [→](./_files/unsloth_kernels_utils_py.md) |
| ✅ | `unsloth/models/__init__.py` | 30 | Models package init | — | [→](./_files/unsloth_models___init___py.md) |
| ✅ | `unsloth/models/_utils.py` | 2453 | Core utilities module | Workflow: QLoRA_Finetuning, GRPO_Reinforcement_Learning, Vision_Model_Finetuning | [→](./_files/unsloth_models__utils_py.md) |
| ✅ | `unsloth/models/cohere.py` | 526 | Cohere model support | Impl: Cohere_Model | [→](./_files/unsloth_models_cohere_py.md) |
| ✅ | `unsloth/models/dpo.py` | 26 | DPO trainer stub | — | [→](./_files/unsloth_models_dpo_py.md) |
| ✅ | `unsloth/models/falcon_h1.py` | 764 | Falcon H1 hybrid model | Impl: Falcon_H1_Model | [→](./_files/unsloth_models_falcon_h1_py.md) |
| ✅ | `unsloth/models/gemma.py` | 474 | Gemma model support | Workflow: QLoRA_Finetuning | [→](./_files/unsloth_models_gemma_py.md) |
| ✅ | `unsloth/models/gemma2.py` | 654 | Gemma2 softcapping support | Workflow: QLoRA_Finetuning | [→](./_files/unsloth_models_gemma2_py.md) |
| ✅ | `unsloth/models/granite.py` | 610 | Granite model support | Impl: Granite_Model | [→](./_files/unsloth_models_granite_py.md) |
| ✅ | `unsloth/models/llama.py` | 3475 | Core FastLlamaModel impl | Workflow: QLoRA_Finetuning, GRPO_Reinforcement_Learning | [→](./_files/unsloth_models_llama_py.md) |
| ✅ | `unsloth/models/llama4.py` | 16 | Llama4 placeholder | — | [→](./_files/unsloth_models_llama4_py.md) |
| ✅ | `unsloth/models/loader.py` | 1374 | Main model loader | Workflow: QLoRA_Finetuning, GRPO_Reinforcement_Learning, Vision_Model_Finetuning | [→](./_files/unsloth_models_loader_py.md) |
| ✅ | `unsloth/models/loader_utils.py` | 427 | Loader utilities | Workflow: QLoRA_Finetuning, GRPO_Reinforcement_Learning | [→](./_files/unsloth_models_loader_utils_py.md) |
| ✅ | `unsloth/models/mapper.py` | 1329 | Model name mappings | Workflow: QLoRA_Finetuning | [→](./_files/unsloth_models_mapper_py.md) |
| ✅ | `unsloth/models/mistral.py` | 469 | Mistral model support | Workflow: QLoRA_Finetuning | [→](./_files/unsloth_models_mistral_py.md) |
| ✅ | `unsloth/models/qwen2.py` | 101 | Qwen2 model support | Workflow: QLoRA_Finetuning | [→](./_files/unsloth_models_qwen2_py.md) |
| ✅ | `unsloth/models/qwen3.py` | 457 | Qwen3 QK-norm support | Workflow: QLoRA_Finetuning | [→](./_files/unsloth_models_qwen3_py.md) |
| ✅ | `unsloth/models/qwen3_moe.py` | 243 | Qwen3 MoE support | Impl: Qwen3_MoE_Model | [→](./_files/unsloth_models_qwen3_moe_py.md) |
| ✅ | `unsloth/models/rl.py` | 1443 | TRL trainer patching | Workflow: GRPO_Reinforcement_Learning | [→](./_files/unsloth_models_rl_py.md) |
| ✅ | `unsloth/models/rl_replacements.py` | 995 | RL trainer patches | Workflow: GRPO_Reinforcement_Learning | [→](./_files/unsloth_models_rl_replacements_py.md) |
| ✅ | `unsloth/models/vision.py` | 1292 | Vision model base | Workflow: Vision_Model_Finetuning | [→](./_files/unsloth_models_vision_py.md) |
| ✅ | `unsloth/ollama_template_mappers.py` | 2192 | Ollama template mappings | Workflow: GGUF_Export | [→](./_files/unsloth_ollama_template_mappers_py.md) |
| ✅ | `unsloth/registry/__init__.py` | 78 | Registry package init | — | [→](./_files/unsloth_registry___init___py.md) |
| ✅ | `unsloth/registry/_deepseek.py` | 206 | DeepSeek model registry | Impl: DeepSeek_Registry | [→](./_files/unsloth_registry__deepseek_py.md) |
| ✅ | `unsloth/registry/_gemma.py` | 74 | Gemma model registry | — | [→](./_files/unsloth_registry__gemma_py.md) |
| ✅ | `unsloth/registry/_llama.py` | 125 | Llama model registry | — | [→](./_files/unsloth_registry__llama_py.md) |
| ✅ | `unsloth/registry/_mistral.py` | 88 | Mistral model registry | — | [→](./_files/unsloth_registry__mistral_py.md) |
| ✅ | `unsloth/registry/_phi.py` | 74 | Phi model registry | — | [→](./_files/unsloth_registry__phi_py.md) |
| ✅ | `unsloth/registry/_qwen.py` | 136 | Qwen model registry | — | [→](./_files/unsloth_registry__qwen_py.md) |
| ✅ | `unsloth/registry/registry.py` | 191 | Core registry module | Impl: Model_Registry | [→](./_files/unsloth_registry_registry_py.md) |
| ✅ | `unsloth/save.py` | 3100 | Model saving/export | Workflow: QLoRA_Finetuning, GRPO_Reinforcement_Learning, Vision_Model_Finetuning, GGUF_Export | [→](./_files/unsloth_save_py.md) |
| ✅ | `unsloth/tokenizer_utils.py` | 1106 | Tokenizer utilities | Workflow: QLoRA_Finetuning, GRPO_Reinforcement_Learning, GGUF_Export | [→](./_files/unsloth_tokenizer_utils_py.md) |
| ✅ | `unsloth/trainer.py` | 438 | UnslothTrainer impl | Workflow: QLoRA_Finetuning, Vision_Model_Finetuning | [→](./_files/unsloth_trainer_py.md) |
| ✅ | `unsloth/utils/__init__.py` | 48 | Utils package init | — | [→](./_files/unsloth_utils___init___py.md) |
| ✅ | `unsloth/utils/attention_dispatch.py` | 274 | Attention backend dispatch | Impl: Attention_Dispatch | [→](./_files/unsloth_utils_attention_dispatch_py.md) |
| ✅ | `unsloth/utils/hf_hub.py` | 80 | HuggingFace Hub utils | — | [→](./_files/unsloth_utils_hf_hub_py.md) |
| ✅ | `unsloth/utils/packing.py` | 344 | Sequence packing utils | Workflow: QLoRA_Finetuning | [→](./_files/unsloth_utils_packing_py.md) |

## 📝 Example Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `scripts/enforce_kwargs_spacing.py` | 179 | Code formatting script | — | [→](./_files/scripts_enforce_kwargs_spacing_py.md) |
| ✅ | `scripts/run_ruff_format.py` | 30 | Ruff formatter wrapper | — | [→](./_files/scripts_run_ruff_format_py.md) |

## 🧪 Test Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `tests/__init__.py` | 0 | Tests package init | — | [→](./_files/tests___init___py.md) |
| ✅ | `tests/qlora/test_hf_qlora_train_and_merge.py` | 159 | HF QLoRA baseline test | Workflow: QLoRA_Finetuning | [→](./_files/tests_qlora_test_hf_qlora_train_and_merge_py.md) |
| ✅ | `tests/qlora/test_unsloth_qlora_train_and_merge.py` | 211 | Unsloth QLoRA test | Workflow: QLoRA_Finetuning | [→](./_files/tests_qlora_test_unsloth_qlora_train_and_merge_py.md) |
| ✅ | `tests/saving/gpt-oss-merge/test_merged_model.py` | 60 | GPT-OSS inference test | — | [→](./_files/tests_saving_gpt-oss-merge_test_merged_model_py.md) |
| ✅ | `tests/saving/gpt-oss-merge/train_and_merge.py` | 102 | GPT-OSS training test | — | [→](./_files/tests_saving_gpt-oss-merge_train_and_merge_py.md) |
| ✅ | `tests/saving/language_models/test_merge_4bit_validation.py` | 248 | 4-bit merge validation | Workflow: QLoRA_Finetuning | [→](./_files/tests_saving_language_models_test_merge_4bit_validation_py.md) |
| ✅ | `tests/saving/language_models/test_merge_model_perplexity_llama-3.2.py` | 259 | Llama-3.2 perplexity test | Workflow: QLoRA_Finetuning | [→](./_files/tests_saving_language_models_test_merge_model_perplexity_llama-3_2_py.md) |
| ✅ | `tests/saving/language_models/test_merge_model_perplexity_mistral.py` | 318 | Mistral perplexity test | Workflow: QLoRA_Finetuning | [→](./_files/tests_saving_language_models_test_merge_model_perplexity_mistral_py.md) |
| ✅ | `tests/saving/language_models/test_merge_model_perplexity_phi_4.py` | 259 | Phi-4 perplexity test | Workflow: QLoRA_Finetuning | [→](./_files/tests_saving_language_models_test_merge_model_perplexity_phi_4_py.md) |
| ✅ | `tests/saving/language_models/test_merged_model_perplexity_llama-3.1-8b.py` | 263 | Llama-3.1-8B perplexity test | Workflow: QLoRA_Finetuning | [→](./_files/tests_saving_language_models_test_merged_model_perplexity_llama-3_1-8b_py.md) |
| ✅ | `tests/saving/language_models/test_merged_model_perplexity_qwen_2.5.py` | 311 | Qwen2.5 perplexity test | Workflow: QLoRA_Finetuning | [→](./_files/tests_saving_language_models_test_merged_model_perplexity_qwen_2_5_py.md) |
| ✅ | `tests/saving/language_models/test_push_to_hub_merged.py` | 204 | Hub push test | Workflow: QLoRA_Finetuning | [→](./_files/tests_saving_language_models_test_push_to_hub_merged_py.md) |
| ✅ | `tests/saving/language_models/test_push_to_hub_merged_sharded_index_file.py` | 223 | Sharded upload test | Workflow: QLoRA_Finetuning | [→](./_files/tests_saving_language_models_test_push_to_hub_merged_sharded_index_file_py.md) |
| ✅ | `tests/saving/language_models/test_save_merged_grpo_model.py` | 825 | GRPO training test | Workflow: GRPO_Reinforcement_Learning | [→](./_files/tests_saving_language_models_test_save_merged_grpo_model_py.md) |
| ✅ | `tests/saving/non_peft/test_mistral_non_peft.py` | 65 | Non-PEFT save warning | — | [→](./_files/tests_saving_non_peft_test_mistral_non_peft_py.md) |
| ✅ | `tests/saving/non_peft/test_whisper_non_peft.py` | 65 | Whisper non-PEFT test | — | [→](./_files/tests_saving_non_peft_test_whisper_non_peft_py.md) |
| ✅ | `tests/saving/test_unsloth_save.py` | 401 | Save functionality tests | Workflow: QLoRA_Finetuning, GGUF_Export | [→](./_files/tests_saving_test_unsloth_save_py.md) |
| ✅ | `tests/saving/text_to_speech_models/test_csm.py` | 168 | CSM TTS test | — | [→](./_files/tests_saving_text_to_speech_models_test_csm_py.md) |
| ✅ | `tests/saving/text_to_speech_models/test_lasa.py` | 220 | Llasa TTS test | — | [→](./_files/tests_saving_text_to_speech_models_test_lasa_py.md) |
| ✅ | `tests/saving/text_to_speech_models/test_orpheus.py` | 282 | Orpheus TTS test | — | [→](./_files/tests_saving_text_to_speech_models_test_orpheus_py.md) |
| ✅ | `tests/saving/text_to_speech_models/test_whisper.py` | 195 | Whisper STT test | — | [→](./_files/tests_saving_text_to_speech_models_test_whisper_py.md) |
| ✅ | `tests/saving/vision_models/test_index_file_sharded_model.py` | 293 | Vision sharded save test | Workflow: Vision_Model_Finetuning | [→](./_files/tests_saving_vision_models_test_index_file_sharded_model_py.md) |
| ✅ | `tests/saving/vision_models/test_push_to_hub_merged.py` | 273 | Vision Hub push test | Workflow: Vision_Model_Finetuning | [→](./_files/tests_saving_vision_models_test_push_to_hub_merged_py.md) |
| ✅ | `tests/saving/vision_models/test_save_merge_qwen2.5vl32B_model_ocr_benchmark.py` | 287 | Qwen2.5-VL 32B OCR test | Workflow: Vision_Model_Finetuning | [→](./_files/tests_saving_vision_models_test_save_merge_qwen2_5vl32B_model_ocr_benchmark_py.md) |
| ✅ | `tests/saving/vision_models/test_save_merge_vision_model_ocr_benchmark.py` | 287 | Vision OCR benchmark | Workflow: Vision_Model_Finetuning | [→](./_files/tests_saving_vision_models_test_save_merge_vision_model_ocr_benchmark_py.md) |
| ✅ | `tests/test_model_registry.py` | 92 | Model registry tests | — | [→](./_files/tests_test_model_registry_py.md) |
| ✅ | `tests/test_raw_text.py` | 172 | Raw text loader test | — | [→](./_files/tests_test_raw_text_py.md) |
| ✅ | `tests/utils/__init__.py` | 33 | Test utils init | — | [→](./_files/tests_utils___init___py.md) |
| ✅ | `tests/utils/aime_eval.py` | 545 | AIME benchmark eval | Workflow: GRPO_Reinforcement_Learning | [→](./_files/tests_utils_aime_eval_py.md) |
| ✅ | `tests/utils/cleanup_utils.py` | 226 | Memory cleanup utils | — | [→](./_files/tests_utils_cleanup_utils_py.md) |
| ✅ | `tests/utils/data_utils.py` | 153 | Test data utilities | Workflow: QLoRA_Finetuning | [→](./_files/tests_utils_data_utils_py.md) |
| ✅ | `tests/utils/hf_utils.py` | 291 | HF integration utils | Workflow: QLoRA_Finetuning | [→](./_files/tests_utils_hf_utils_py.md) |
| ✅ | `tests/utils/ocr_eval.py` | 374 | OCR evaluation utils | Workflow: Vision_Model_Finetuning | [→](./_files/tests_utils_ocr_eval_py.md) |
| ✅ | `tests/utils/os_utils.py` | 128 | OS package utils | — | [→](./_files/tests_utils_os_utils_py.md) |
| ✅ | `tests/utils/perplexity_eval.py` | 81 | Perplexity evaluation | Workflow: QLoRA_Finetuning | [→](./_files/tests_utils_perplexity_eval_py.md) |
| ✅ | `tests/utils/test_attention_masks.py` | 272 | Attention mask tests | — | [→](./_files/tests_utils_test_attention_masks_py.md) |
| ✅ | `tests/utils/test_packing.py` | 391 | Packing functionality tests | Workflow: QLoRA_Finetuning | [→](./_files/tests_utils_test_packing_py.md) |
| ✅ | `tests/utils/test_qat.py` | 156 | QAT functionality tests | — | [→](./_files/tests_utils_test_qat_py.md) |

## 📄 Other Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `unsloth-cli.py` | 473 | CLI fine-tuning tool | Workflow: QLoRA_Finetuning | [→](./_files/unsloth-cli_py.md) |

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
