# Workflow Index: unslothai_unsloth

> Comprehensive index of Workflows and their implementation context.
> This index bridges Phase 1 (Anchoring) and Phase 2 (Excavation).
> **Update IMMEDIATELY** after creating or modifying a Workflow page.

---

## Summary

| Workflow | Steps | Principles | Implementation APIs |
|----------|-------|------------|---------------------|
| QLoRA_Finetuning | 7 | 7 | FastLanguageModel.from_pretrained, get_peft_model, SFTTrainer, save_pretrained_merged |
| GRPO_Reinforcement_Learning | 8 | 8 | FastLanguageModel.from_pretrained, get_peft_model, get_chat_template, GRPOTrainer |
| Model_Export | 8 | 8 | save_pretrained, save_pretrained_merged, save_pretrained_gguf, push_to_hub |

---

## Workflow: unslothai_unsloth_QLoRA_Finetuning

**File:** [→](./workflows/unslothai_unsloth_QLoRA_Finetuning.md)
**Description:** End-to-end parameter-efficient fine-tuning using 4-bit QLoRA with Unsloth optimizations.

### Steps Overview

| # | Step Name | Principle | Implementation | Status |
|---|-----------|-----------|----------------|--------|
| 1 | Environment Setup | Environment_Initialization | `import unsloth` | ✅ |
| 2 | Model Loading | Model_Loading | `FastLanguageModel.from_pretrained` | ✅ |
| 3 | LoRA Injection | LoRA_Configuration | `FastLanguageModel.get_peft_model` | ✅ |
| 4 | Dataset Preparation | Data_Formatting | `get_chat_template`, `apply_chat_template` | ✅ |
| 5 | Trainer Configuration | Training_Configuration | `SFTConfig`, `SFTTrainer` | ✅ |
| 6 | Training Execution | SFT_Training | `trainer.train()` | ✅ |
| 7 | Model Saving | Model_Saving | `save_pretrained_merged` | ✅ |

### Step 1: Environment_Initialization

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_Environment_Initialization` |
| **Implementation** | `unslothai_unsloth_import_unsloth` |
| **API Call** | `import unsloth` (before transformers/trl) |
| **Source Location** | `unsloth/__init__.py:L1-100` |
| **External Dependencies** | `torch`, `triton`, `bitsandbytes`, `transformers`, `peft`, `trl` |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | N/A (import statement) |
| **Inputs** | None |
| **Outputs** | Patched transformers/trl libraries |

### Step 2: Model_Loading

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_Model_Loading` |
| **Implementation** | `unslothai_unsloth_FastLanguageModel_from_pretrained` |
| **API Call** | `FastLanguageModel.from_pretrained(model_name, max_seq_length, load_in_4bit, dtype)` |
| **Source Location** | `unsloth/models/loader.py:L120-620` |
| **External Dependencies** | `transformers`, `bitsandbytes`, `huggingface_hub` |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | `model_name: str`, `max_seq_length: int`, `load_in_4bit: bool`, `dtype: Optional[torch.dtype]` |
| **Inputs** | Model name/path (HuggingFace ID or local path) |
| **Outputs** | `Tuple[PeftModel, PreTrainedTokenizer]` |

### Step 3: LoRA_Configuration

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_LoRA_Configuration` |
| **Implementation** | `unslothai_unsloth_get_peft_model` |
| **API Call** | `FastLanguageModel.get_peft_model(model, r, lora_alpha, target_modules, use_gradient_checkpointing, ...)` |
| **Source Location** | `unsloth/models/llama.py:L2578-3100` |
| **External Dependencies** | `peft` |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | `r: int`, `lora_alpha: int`, `target_modules: List[str]`, `use_gradient_checkpointing: str` |
| **Inputs** | Model from Step 2 |
| **Outputs** | `PeftModel` with LoRA adapters injected |

### Step 4: Data_Formatting

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_Data_Formatting` |
| **Implementation** | `unslothai_unsloth_get_chat_template` |
| **API Call** | `get_chat_template(tokenizer, chat_template)`, `tokenizer.apply_chat_template(...)` |
| **Source Location** | `unsloth/chat_templates.py:L50-500` |
| **External Dependencies** | `datasets` |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | `chat_template: str`, `add_generation_prompt: bool` |
| **Inputs** | Raw dataset, tokenizer |
| **Outputs** | Formatted dataset with chat template applied |

### Step 5: Training_Configuration

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_Training_Configuration` |
| **Implementation** | `unslothai_unsloth_SFTTrainer_usage` |
| **API Call** | `SFTTrainer(model, tokenizer, train_dataset, args=SFTConfig(...))` |
| **Source Location** | TRL library (external), patched in `unsloth/trainer.py:L1-437` |
| **External Dependencies** | `trl.SFTTrainer`, `trl.SFTConfig` |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `max_steps`, `optim` |
| **Inputs** | Model, tokenizer, formatted dataset |
| **Outputs** | Configured SFTTrainer instance |

### Step 6: SFT_Training

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_SFT_Training` |
| **Implementation** | `unslothai_unsloth_trainer_train` |
| **API Call** | `trainer.train()` |
| **Source Location** | `unsloth/trainer.py:L100-437` (optimizations), TRL trainer core |
| **External Dependencies** | `trl`, `transformers.Trainer` |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | None (uses trainer config) |
| **Inputs** | Configured SFTTrainer |
| **Outputs** | TrainOutput with training metrics |

### Step 7: Model_Saving

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_Model_Saving` |
| **Implementation** | `unslothai_unsloth_save_pretrained_merged` |
| **API Call** | `model.save_pretrained_merged(save_directory, tokenizer, save_method)` |
| **Source Location** | `unsloth/save.py:L200-800` |
| **External Dependencies** | `safetensors`, `huggingface_hub` |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | `save_directory: str`, `save_method: str` ("merged_16bit", "lora", etc.) |
| **Inputs** | Trained model, tokenizer |
| **Outputs** | Saved model files (safetensors, config, tokenizer) |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| Environment_Initialization | `import_unsloth` | `import unsloth` | `__init__.py` | Pattern Doc |
| Model_Loading | `FastLanguageModel_from_pretrained` | `from_pretrained` | `loader.py` | API Doc |
| LoRA_Configuration | `get_peft_model` | `get_peft_model` | `llama.py` | API Doc |
| Data_Formatting | `get_chat_template` | `get_chat_template` | `chat_templates.py` | API Doc |
| Training_Configuration | `SFTTrainer_usage` | `SFTTrainer` | TRL (external) | Wrapper Doc |
| SFT_Training | `trainer_train` | `train()` | TRL (external) | Wrapper Doc |
| Model_Saving | `save_pretrained_merged` | `save_pretrained_merged` | `save.py` | API Doc |

---

## Workflow: unslothai_unsloth_GRPO_Reinforcement_Learning

**File:** [→](./workflows/unslothai_unsloth_GRPO_Reinforcement_Learning.md)
**Description:** Two-stage training pipeline combining optional SFT warmup with GRPO reinforcement learning for reasoning tasks.

### Steps Overview

| # | Step Name | Principle | Implementation | Status |
|---|-----------|-----------|----------------|--------|
| 1 | Model Loading + vLLM | RL_Model_Loading | `FastLanguageModel.from_pretrained(fast_inference=True)` | ✅ |
| 2 | LoRA Setup | LoRA_Configuration | `FastLanguageModel.get_peft_model` | ✅ |
| 3 | Chat Template Config | Chat_Template_Setup | `get_chat_template` | ✅ |
| 4 | SFT Warmup | SFT_Training | `SFTTrainer.train()` | ✅ |
| 5 | Reward Definition | Reward_Function_Interface | User-defined reward functions | ✅ |
| 6 | GRPO Config | GRPO_Configuration | `GRPOConfig` | ✅ |
| 7 | GRPO Training | GRPO_Training | `GRPOTrainer.train()` | ✅ |
| 8 | Save & Evaluate | Model_Saving | `save_pretrained_merged` | ✅ |

### Step 1: RL_Model_Loading

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_RL_Model_Loading` |
| **Implementation** | `unslothai_unsloth_FastLanguageModel_from_pretrained_vllm` |
| **API Call** | `FastLanguageModel.from_pretrained(model_name, fast_inference=True, gpu_memory_utilization, max_lora_rank)` |
| **Source Location** | `unsloth/models/loader.py:L120-620`, vLLM integration in `_utils.py` |
| **External Dependencies** | `vllm`, `transformers`, `bitsandbytes` |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | `fast_inference: bool=True`, `gpu_memory_utilization: float`, `max_lora_rank: int` |
| **Inputs** | Model name/path |
| **Outputs** | `Tuple[PeftModel, PreTrainedTokenizer]` with vLLM backend |

### Step 2: LoRA_Configuration

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_LoRA_Configuration` |
| **Implementation** | `unslothai_unsloth_get_peft_model` |
| **API Call** | `FastLanguageModel.get_peft_model(model, r, lora_alpha, target_modules, use_gradient_checkpointing="unsloth")` |
| **Source Location** | `unsloth/models/llama.py:L2578-3100` |
| **External Dependencies** | `peft` |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | `r: int` (typically 64 for RL), `use_gradient_checkpointing: str="unsloth"` |
| **Inputs** | Model from Step 1 |
| **Outputs** | `PeftModel` with LoRA adapters |

### Step 3: Chat_Template_Setup

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_Chat_Template_Setup` |
| **Implementation** | `unslothai_unsloth_get_chat_template` |
| **API Call** | `get_chat_template(tokenizer, chat_template="llama-3.1")` |
| **Source Location** | `unsloth/chat_templates.py:L50-500` |
| **External Dependencies** | None |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | `chat_template: str` |
| **Inputs** | Tokenizer |
| **Outputs** | Tokenizer with chat template configured |

### Step 4: SFT_Training (Optional Warmup)

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_SFT_Training` |
| **Implementation** | `unslothai_unsloth_SFTTrainer_usage` |
| **API Call** | `SFTTrainer(model, tokenizer, train_dataset, ...).train()` |
| **Source Location** | TRL library, `unsloth/trainer.py` |
| **External Dependencies** | `trl.SFTTrainer` |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | Standard SFT parameters |
| **Inputs** | Model, tokenizer, demonstration dataset (e.g., LIMO) |
| **Outputs** | Model with initial reasoning patterns |

### Step 5: Reward_Function_Interface

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_Reward_Function_Interface` |
| **Implementation** | `unslothai_unsloth_reward_function_pattern` |
| **API Call** | `def reward_func(prompts, completions, **kwargs) -> List[float]` |
| **Source Location** | User-defined (pattern documented in `rl.py` and examples) |
| **External Dependencies** | None |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | `prompts: List`, `completions: List`, `answer: List` (optional) |
| **Inputs** | Batch of prompts and completions |
| **Outputs** | `List[float]` - reward scores for each completion |

### Step 6: GRPO_Configuration

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_GRPO_Configuration` |
| **Implementation** | `unslothai_unsloth_GRPOConfig` |
| **API Call** | `GRPOConfig(learning_rate, num_generations, max_prompt_length, max_completion_length, ...)` |
| **Source Location** | TRL library, patched in `unsloth/models/rl.py:L1-500` |
| **External Dependencies** | `trl.GRPOConfig` |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | `num_generations: int`, `max_prompt_length: int`, `max_completion_length: int`, `learning_rate: float` |
| **Inputs** | Training hyperparameters |
| **Outputs** | GRPOConfig instance |

### Step 7: GRPO_Training

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_GRPO_Training` |
| **Implementation** | `unslothai_unsloth_GRPOTrainer_train` |
| **API Call** | `GRPOTrainer(model, processing_class, reward_funcs, args, train_dataset).train()` |
| **Source Location** | `unsloth/models/rl.py:L500-1349`, `rl_replacements.py` |
| **External Dependencies** | `trl.GRPOTrainer` |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | `reward_funcs: List[Callable]` |
| **Inputs** | Model, tokenizer, reward functions, config, dataset |
| **Outputs** | TrainOutput with RL metrics |

### Step 8: Model_Saving

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_Model_Saving` |
| **Implementation** | `unslothai_unsloth_save_pretrained_merged` |
| **API Call** | `model.save_pretrained_merged(save_directory, tokenizer, save_method="merged_16bit")` |
| **Source Location** | `unsloth/save.py:L200-800` |
| **External Dependencies** | `safetensors` |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | `save_directory: str`, `save_method: str` |
| **Inputs** | Trained model, tokenizer |
| **Outputs** | Saved model files |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| RL_Model_Loading | `FastLanguageModel_from_pretrained_vllm` | `from_pretrained` | `loader.py` | API Doc |
| LoRA_Configuration | `get_peft_model` | `get_peft_model` | `llama.py` | API Doc |
| Chat_Template_Setup | `get_chat_template` | `get_chat_template` | `chat_templates.py` | API Doc |
| SFT_Training | `SFTTrainer_usage` | `SFTTrainer` | TRL (external) | Wrapper Doc |
| Reward_Function_Interface | `reward_function_pattern` | N/A | User code | Pattern Doc |
| GRPO_Configuration | `GRPOConfig` | `GRPOConfig` | TRL (external) | Wrapper Doc |
| GRPO_Training | `GRPOTrainer_train` | `GRPOTrainer.train` | `rl.py` | Wrapper Doc |
| Model_Saving | `save_pretrained_merged` | `save_pretrained_merged` | `save.py` | API Doc |

---

## Workflow: unslothai_unsloth_Model_Export

**File:** [→](./workflows/unslothai_unsloth_Model_Export.md)
**Description:** Export trained models to various deployment formats including HuggingFace Hub, GGUF, and Ollama.

### Steps Overview

| # | Step Name | Principle | Implementation | Status |
|---|-----------|-----------|----------------|--------|
| 1 | Training Verification | Training_Verification | `model.generate()` | ✅ |
| 2 | Format Selection | Export_Format_Selection | N/A (decision) | ✅ |
| 3 | LoRA Export | LoRA_Export | `model.save_pretrained()` | ✅ |
| 4 | Merged Export | Merged_Export | `model.save_pretrained_merged()` | ✅ |
| 5 | GGUF Conversion | GGUF_Conversion | `model.save_pretrained_gguf()` | ✅ |
| 6 | Ollama Package | Ollama_Export | GGUF + Modelfile generation | ✅ |
| 7 | Hub Upload | Hub_Upload | `model.push_to_hub()` | ✅ |
| 8 | Validation | Export_Validation | Load and test | ✅ |

### Step 1: Training_Verification

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_Training_Verification` |
| **Implementation** | `unslothai_unsloth_model_generate` |
| **API Call** | `model.generate(input_ids, max_new_tokens, ...)` |
| **Source Location** | `unsloth/models/llama.py:L2500-2550` (patched generate) |
| **External Dependencies** | None |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | `max_new_tokens: int`, `temperature: float` |
| **Inputs** | Test prompts |
| **Outputs** | Generated text for quality verification |

### Step 2: Export_Format_Selection

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_Export_Format_Selection` |
| **Implementation** | N/A (decision point) |
| **API Call** | N/A |
| **Source Location** | N/A |
| **External Dependencies** | None |
| **Environment** | N/A |
| **Key Parameters** | N/A |
| **Inputs** | Deployment requirements |
| **Outputs** | Chosen export format |

### Step 3: LoRA_Export

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_LoRA_Export` |
| **Implementation** | `unslothai_unsloth_save_pretrained_lora` |
| **API Call** | `model.save_pretrained(save_directory)` |
| **Source Location** | `unsloth/save.py:L100-200` |
| **External Dependencies** | `peft`, `safetensors` |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | `save_directory: str` |
| **Inputs** | Trained model |
| **Outputs** | LoRA adapter files (adapter_model.safetensors, adapter_config.json) |

### Step 4: Merged_Export

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_Merged_Export` |
| **Implementation** | `unslothai_unsloth_save_pretrained_merged` |
| **API Call** | `model.save_pretrained_merged(save_directory, tokenizer, save_method="merged_16bit")` |
| **Source Location** | `unsloth/save.py:L200-800` |
| **External Dependencies** | `safetensors`, `bitsandbytes` (for dequantization) |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | `save_directory: str`, `save_method: str` |
| **Inputs** | Trained model, tokenizer |
| **Outputs** | Merged model files in safetensors format |

### Step 5: GGUF_Conversion

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_GGUF_Conversion` |
| **Implementation** | `unslothai_unsloth_save_pretrained_gguf` |
| **API Call** | `model.save_pretrained_gguf(save_directory, tokenizer, quantization_method)` |
| **Source Location** | `unsloth/save.py:L800-1500` |
| **External Dependencies** | `llama.cpp` (convert_hf_to_gguf.py, llama-quantize) |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | `quantization_method: str` (q4_k_m, q8_0, f16, etc.) |
| **Inputs** | Trained model, tokenizer |
| **Outputs** | GGUF file |

### Step 6: Ollama_Export

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_Ollama_Export` |
| **Implementation** | `unslothai_unsloth_ollama_modelfile` |
| **API Call** | GGUF conversion + Modelfile generation |
| **Source Location** | `unsloth/ollama_template_mappers.py:L1-2192` |
| **External Dependencies** | `llama.cpp` |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | Model family (for template selection) |
| **Inputs** | GGUF model |
| **Outputs** | GGUF file + Modelfile for Ollama |

### Step 7: Hub_Upload

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_Hub_Upload` |
| **Implementation** | `unslothai_unsloth_push_to_hub` |
| **API Call** | `model.push_to_hub(repo_id, tokenizer, save_method, ...)` |
| **Source Location** | `unsloth/save.py:L1500-2000` |
| **External Dependencies** | `huggingface_hub` |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | `repo_id: str`, `private: bool`, `token: str` |
| **Inputs** | Trained model, tokenizer |
| **Outputs** | HuggingFace repository URL |

### Step 8: Export_Validation

| Attribute | Value |
|-----------|-------|
| **Principle** | `unslothai_unsloth_Export_Validation` |
| **Implementation** | `unslothai_unsloth_load_and_validate` |
| **API Call** | Load exported model and run inference |
| **Source Location** | Various (depends on format) |
| **External Dependencies** | Depends on format (transformers, llama.cpp, Ollama) |
| **Environment** | `unslothai_unsloth_CUDA` |
| **Key Parameters** | Varies by format |
| **Inputs** | Exported model path |
| **Outputs** | Validation results |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| Training_Verification | `model_generate` | `generate` | `llama.py` | API Doc |
| Export_Format_Selection | N/A | N/A | N/A | Pattern Doc |
| LoRA_Export | `save_pretrained_lora` | `save_pretrained` | `save.py` | API Doc |
| Merged_Export | `save_pretrained_merged` | `save_pretrained_merged` | `save.py` | API Doc |
| GGUF_Conversion | `save_pretrained_gguf` | `save_pretrained_gguf` | `save.py` | API Doc |
| Ollama_Export | `ollama_modelfile` | Modelfile generation | `ollama_template_mappers.py` | API Doc |
| Hub_Upload | `push_to_hub` | `push_to_hub` | `save.py` | API Doc |
| Export_Validation | `load_and_validate` | Various | Various | Pattern Doc |

---

## External Dependencies Summary

| Library | Version | Used By Workflows |
|---------|---------|-------------------|
| `transformers` | >=4.37 | All |
| `peft` | >=0.7.2 | QLoRA, GRPO |
| `trl` | latest | QLoRA, GRPO |
| `bitsandbytes` | latest | QLoRA, GRPO |
| `vllm` | optional | GRPO (fast_inference) |
| `llama.cpp` | latest | Model_Export (GGUF) |
| `huggingface_hub` | latest | All (Hub operations) |
| `safetensors` | latest | All (model saving) |
| `triton` | >=3.0.0 | All (kernels) |

---

**Legend:** `✅` = Page exists (Implementation + Principle created) | `⬜` = Page needs creation
