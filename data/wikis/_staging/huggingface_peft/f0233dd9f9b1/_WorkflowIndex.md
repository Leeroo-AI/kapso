# Workflow Index: huggingface_peft

> Comprehensive index of Workflows and their implementation context.
> This index bridges Phase 1 (Anchoring) and Phase 2 (Excavation).
> **Update IMMEDIATELY** after creating or modifying a Workflow page.

---

## Summary

| Workflow | Steps | Principles | Key APIs |
|----------|-------|------------|----------|
| huggingface_peft_LoRA_Fine_Tuning | 6 | 6 | `LoraConfig`, `get_peft_model`, `save_pretrained` |
| huggingface_peft_QLoRA_Training | 7 | 7 | `BitsAndBytesConfig`, `prepare_model_for_kbit_training`, `get_peft_model` |
| huggingface_peft_Adapter_Loading_Inference | 5 | 5 | `PeftModel.from_pretrained`, `merge_and_unload`, `generate` |
| huggingface_peft_Adapter_Merging | 7 | 7 | `load_adapter`, `add_weighted_adapter`, `ties`, `dare_ties` |
| huggingface_peft_Multi_Adapter_Management | 6 | 6 | `load_adapter`, `set_adapter`, `disable_adapter`, `delete_adapter` |

---

## Workflow: huggingface_peft_LoRA_Fine_Tuning

**File:** [→](./workflows/huggingface_peft_LoRA_Fine_Tuning.md)
**Description:** Standard workflow for parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA).

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Load Base Model | Base_Model_Loading | `AutoModelForCausalLM.from_pretrained()` | auto.py |
| 2 | Configure LoRA | LoRA_Configuration | `LoraConfig(r, lora_alpha, target_modules, ...)` | tuners/lora/config.py |
| 3 | Create PEFT Model | PEFT_Model_Creation | `get_peft_model(model, config)` | mapping_func.py |
| 4 | Prepare Training | Training_Preparation | `model.train()`, optimizer setup | peft_model.py |
| 5 | Execute Training | Training_Execution | Forward/backward pass | External (Trainer) |
| 6 | Save Adapter | Adapter_Serialization | `model.save_pretrained(path)` | utils/save_and_load.py |

### Step 1: Base_Model_Loading

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Base_Model_Loading` |
| **Implementation** | `huggingface_peft_AutoModelForCausalLM_from_pretrained` |
| **API Call** | `AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path: str, torch_dtype: torch.dtype = None, device_map: str = None, **kwargs) -> PreTrainedModel` |
| **Source Location** | `transformers` (external library) |
| **External Dependencies** | `transformers`, `torch`, `safetensors` |
| **Environment** | `huggingface_peft_Base_Model_Environment` |
| **Key Parameters** | `pretrained_model_name_or_path: str` - HuggingFace Hub ID or local path, `torch_dtype: torch.dtype` - model precision (torch.float16, torch.bfloat16), `device_map: str` - device placement ("auto", "cuda", "cpu") |
| **Inputs** | Model identifier (HuggingFace Hub ID or local path) |
| **Outputs** | PreTrainedModel instance ready for adaptation |

### Step 2: LoRA_Configuration

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_LoRA_Configuration` |
| **Implementation** | `huggingface_peft_LoraConfig_init` |
| **API Call** | `LoraConfig(r: int = 8, lora_alpha: int = 8, target_modules: Union[list[str], str] = None, lora_dropout: float = 0.0, bias: str = "none", task_type: TaskType = None, use_rslora: bool = False, use_dora: bool = False, **kwargs) -> LoraConfig` |
| **Source Location** | `src/peft/tuners/lora/config.py:L47-300` |
| **External Dependencies** | `dataclasses`, `peft.utils.peft_types` |
| **Environment** | `huggingface_peft_Config_Environment` |
| **Key Parameters** | `r: int` - LoRA rank (typical: 8-64), `lora_alpha: int` - scaling factor (typical: 16-32), `target_modules: Union[list[str], str]` - modules to adapt (e.g., ["q_proj", "v_proj"], "all-linear"), `lora_dropout: float` - dropout probability (0.0-0.1), `use_rslora: bool` - rank-stabilized LoRA, `use_dora: bool` - enable DoRA variant |
| **Inputs** | LoRA hyperparameters |
| **Outputs** | LoraConfig instance defining adapter architecture |

### Step 3: PEFT_Model_Creation

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_PEFT_Model_Creation` |
| **Implementation** | `huggingface_peft_get_peft_model` |
| **API Call** | `get_peft_model(model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default", mixed: bool = False, autocast_adapter_dtype: bool = True, revision: str = None, low_cpu_mem_usage: bool = False) -> PeftModel` |
| **Source Location** | `src/peft/mapping_func.py:L58-169` |
| **External Dependencies** | `torch`, `transformers` |
| **Environment** | `huggingface_peft_PEFT_Model_Environment` |
| **Key Parameters** | `model: PreTrainedModel` - base model to adapt, `peft_config: PeftConfig` - LoRA/adapter configuration, `adapter_name: str` - identifier for adapter (default "default"), `mixed: bool` - allow mixing adapter types, `autocast_adapter_dtype: bool` - auto-cast adapter weights dtype |
| **Inputs** | Base model instance, LoraConfig instance |
| **Outputs** | PeftModel with LoRA adapters injected and ready for training |

### Step 4: Training_Preparation

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Training_Preparation` |
| **Implementation** | `huggingface_peft_model_train_mode` |
| **API Call** | `model.train() -> PeftModel` |
| **Source Location** | `torch.nn.Module` (PyTorch base) |
| **External Dependencies** | `torch` |
| **Environment** | `huggingface_peft_Training_Environment` |
| **Key Parameters** | None (method call sets training mode) |
| **Inputs** | PeftModel with adapters injected |
| **Outputs** | Model in training mode (dropout enabled, gradients tracked) |

### Step 5: Training_Execution

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Training_Execution` |
| **Implementation** | `huggingface_peft_Trainer_train` |
| **API Call** | `Trainer.train(resume_from_checkpoint: Union[str, bool] = None, trial: optuna.Trial = None, **kwargs) -> TrainOutput` |
| **Source Location** | `transformers.Trainer` (external library) |
| **External Dependencies** | `transformers`, `torch`, `datasets`, `accelerate` |
| **Environment** | `huggingface_peft_Training_Environment` |
| **Key Parameters** | `resume_from_checkpoint: Union[str, bool]` - checkpoint path or bool to resume, TrainingArguments: `learning_rate`, `num_train_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps` |
| **Inputs** | PeftModel, training dataset, TrainingArguments |
| **Outputs** | Trained model with updated LoRA weights, training metrics |

### Step 6: Adapter_Serialization

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Adapter_Serialization` |
| **Implementation** | `huggingface_peft_PeftModel_save_pretrained` |
| **API Call** | `PeftModel.save_pretrained(save_directory: str, safe_serialization: bool = True, selected_adapters: list[str] = None, save_embedding_layers: Union[str, bool] = "auto", is_main_process: bool = True, **kwargs) -> None` |
| **Source Location** | `src/peft/peft_model.py:L311-459` |
| **External Dependencies** | `safetensors`, `huggingface_hub`, `json` |
| **Environment** | `huggingface_peft_Save_Environment` |
| **Key Parameters** | `save_directory: str` - output directory path, `safe_serialization: bool` - use safetensors format (recommended), `selected_adapters: list[str]` - specific adapters to save (None = all), `save_embedding_layers: Union[str, bool]` - save embedding layers ("auto" checks if targeted) |
| **Inputs** | Trained PeftModel, save directory path |
| **Outputs** | `adapter_model.safetensors` (weights), `adapter_config.json` (config) |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| Base_Model_Loading | `AutoModelForCausalLM_from_pretrained` | `from_pretrained` | transformers (external) | Wrapper Doc |
| LoRA_Configuration | `LoraConfig_init` | `LoraConfig` | `src/peft/tuners/lora/config.py:L47-300` | API Doc |
| PEFT_Model_Creation | `get_peft_model` | `get_peft_model` | `src/peft/mapping_func.py:L58-169` | API Doc |
| Training_Preparation | `model_train_mode` | `train` | torch.nn.Module (external) | Wrapper Doc |
| Training_Execution | `Trainer_train` | `Trainer.train` | transformers (external) | Wrapper Doc |
| Adapter_Serialization | `PeftModel_save_pretrained` | `save_pretrained` | `src/peft/peft_model.py:L311-459` | API Doc |

---

## Workflow: huggingface_peft_QLoRA_Training

**File:** [→](./workflows/huggingface_peft_QLoRA_Training.md)
**Description:** Fine-tuning large models on consumer GPUs using 4-bit quantization with LoRA.

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Configure Quantization | Quantization_Configuration | `BitsAndBytesConfig(load_in_4bit=True, ...)` | External (transformers) |
| 2 | Load Quantized Model | Quantized_Model_Loading | `AutoModel.from_pretrained(..., quantization_config)` | External (transformers) |
| 3 | Prepare K-bit Training | Kbit_Training_Preparation | `prepare_model_for_kbit_training(model)` | helpers.py |
| 4 | Configure QLoRA | QLoRA_Configuration | `LoraConfig(target_modules="all-linear", ...)` | tuners/lora/config.py |
| 5 | Create QLoRA Model | PEFT_Model_Creation | `get_peft_model(model, config)` | mapping_func.py |
| 6 | Execute Training | QLoRA_Training_Execution | Training loop with gradient accumulation | External (Trainer) |
| 7 | Save Adapter | Adapter_Serialization | `model.save_pretrained(path)` | utils/save_and_load.py |

### Step 1: Quantization_Configuration

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Quantization_Configuration` |
| **Implementation** | `huggingface_peft_BitsAndBytesConfig_4bit` |
| **API Call** | `BitsAndBytesConfig(load_in_4bit: bool = False, load_in_8bit: bool = False, bnb_4bit_compute_dtype: torch.dtype = None, bnb_4bit_quant_type: str = "fp4", bnb_4bit_use_double_quant: bool = False, **kwargs) -> BitsAndBytesConfig` |
| **Source Location** | `transformers.BitsAndBytesConfig` (external) |
| **External Dependencies** | `transformers`, `bitsandbytes`, `torch` |
| **Environment** | `huggingface_peft_Quantization_Environment` |
| **Key Parameters** | `load_in_4bit: bool` - enable 4-bit quantization, `bnb_4bit_compute_dtype: torch.dtype` - compute dtype (torch.float16 or torch.bfloat16), `bnb_4bit_quant_type: str` - quantization type ("nf4" recommended, or "fp4"), `bnb_4bit_use_double_quant: bool` - nested quantization for extra memory savings |
| **Inputs** | Quantization configuration parameters |
| **Outputs** | BitsAndBytesConfig for model loading |

### Step 2: Quantized_Model_Loading

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Quantized_Model_Loading` |
| **Implementation** | `huggingface_peft_AutoModel_from_pretrained_quantized` |
| **API Call** | `AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path: str, quantization_config: BitsAndBytesConfig = None, device_map: str = "auto", **kwargs) -> PreTrainedModel` |
| **Source Location** | `transformers` (external library) |
| **External Dependencies** | `transformers`, `bitsandbytes`, `accelerate` |
| **Environment** | `huggingface_peft_Quantization_Environment` |
| **Key Parameters** | `pretrained_model_name_or_path: str` - model ID, `quantization_config: BitsAndBytesConfig` - 4-bit config from Step 1, `device_map: str` - "auto" required for quantization |
| **Inputs** | Model identifier, BitsAndBytesConfig |
| **Outputs** | Quantized PreTrainedModel with Linear4bit layers |

### Step 3: Kbit_Training_Preparation

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Kbit_Training_Preparation` |
| **Implementation** | `huggingface_peft_prepare_model_for_kbit_training` |
| **API Call** | `prepare_model_for_kbit_training(model: PreTrainedModel, use_gradient_checkpointing: bool = True, gradient_checkpointing_kwargs: dict = None) -> PreTrainedModel` |
| **Source Location** | `src/peft/utils/other.py:L250-320` |
| **External Dependencies** | `torch`, `transformers` |
| **Environment** | `huggingface_peft_QLoRA_Preparation_Environment` |
| **Key Parameters** | `model: PreTrainedModel` - quantized model to prepare, `use_gradient_checkpointing: bool` - enable gradient checkpointing (recommended, reduces VRAM), `gradient_checkpointing_kwargs: dict` - additional checkpointing options |
| **Inputs** | Quantized model from Step 2 |
| **Outputs** | Model prepared for k-bit training (gradients enabled on inputs, norms cast to float32) |

### Step 4: QLoRA_Configuration

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_QLoRA_Configuration` |
| **Implementation** | `huggingface_peft_LoraConfig_for_qlora` |
| **API Call** | `LoraConfig(r: int = 8, lora_alpha: int = 8, target_modules: Union[list[str], str] = None, lora_dropout: float = 0.0, bias: str = "none", task_type: TaskType = None, **kwargs) -> LoraConfig` |
| **Source Location** | `src/peft/tuners/lora/config.py:L47-300` |
| **External Dependencies** | `peft.utils.peft_types` |
| **Environment** | `huggingface_peft_Config_Environment` |
| **Key Parameters** | `r: int` - LoRA rank, `lora_alpha: int` - scaling factor, `target_modules: Union[list[str], str]` - modules to adapt (use "all-linear" for QLoRA), `task_type: TaskType` - CAUSAL_LM for decoder models |
| **Inputs** | LoRA hyperparameters for QLoRA training |
| **Outputs** | LoraConfig optimized for quantized training |

### Step 5: PEFT_Model_Creation

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_PEFT_Model_Creation_QLoRA` |
| **Implementation** | `huggingface_peft_get_peft_model_qlora` |
| **API Call** | `get_peft_model(model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default", **kwargs) -> PeftModel` |
| **Source Location** | `src/peft/mapping_func.py:L58-169` |
| **External Dependencies** | `torch`, `bitsandbytes` |
| **Environment** | `huggingface_peft_QLoRA_Model_Environment` |
| **Key Parameters** | `model: PreTrainedModel` - prepared quantized model, `peft_config: PeftConfig` - LoRA configuration |
| **Inputs** | Prepared quantized model from Step 3, LoraConfig from Step 4 |
| **Outputs** | PeftModel with LoRA adapters on quantized base (Linear4bit/Linear8bitLt layers wrapped) |

### Step 6: QLoRA_Training_Execution

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_QLoRA_Training_Execution` |
| **Implementation** | `huggingface_peft_Trainer_train_qlora` |
| **API Call** | `Trainer.train(resume_from_checkpoint: Union[str, bool] = None, **kwargs) -> TrainOutput` |
| **Source Location** | `transformers.Trainer` (external library) |
| **External Dependencies** | `transformers`, `torch`, `bitsandbytes`, `accelerate` |
| **Environment** | `huggingface_peft_QLoRA_Training_Environment` |
| **Key Parameters** | TrainingArguments: `gradient_accumulation_steps` (important for memory), `fp16` or `bf16` (mixed precision), `optim="paged_adamw_8bit"` (memory-efficient optimizer) |
| **Inputs** | QLoRA PeftModel, training dataset, TrainingArguments with gradient checkpointing |
| **Outputs** | Trained QLoRA model, training metrics |

### Step 7: Adapter_Serialization

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_QLoRA_Adapter_Serialization` |
| **Implementation** | `huggingface_peft_PeftModel_save_pretrained_qlora` |
| **API Call** | `PeftModel.save_pretrained(save_directory: str, safe_serialization: bool = True, **kwargs) -> None` |
| **Source Location** | `src/peft/peft_model.py:L311-459` |
| **External Dependencies** | `safetensors`, `huggingface_hub` |
| **Environment** | `huggingface_peft_Save_Environment` |
| **Key Parameters** | `save_directory: str` - output path, `safe_serialization: bool` - use safetensors |
| **Inputs** | Trained QLoRA PeftModel, save directory |
| **Outputs** | Saved adapter weights (only LoRA parameters, not quantized base weights) |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| Quantization_Configuration | `BitsAndBytesConfig_4bit` | `BitsAndBytesConfig` | transformers (external) | Wrapper Doc |
| Quantized_Model_Loading | `AutoModel_from_pretrained_quantized` | `from_pretrained` | transformers (external) | Wrapper Doc |
| Kbit_Training_Preparation | `prepare_model_for_kbit_training` | `prepare_model_for_kbit_training` | `src/peft/utils/other.py:L250-320` | API Doc |
| QLoRA_Configuration | `LoraConfig_for_qlora` | `LoraConfig` | `src/peft/tuners/lora/config.py:L47-300` | API Doc |
| PEFT_Model_Creation_QLoRA | `get_peft_model_qlora` | `get_peft_model` | `src/peft/mapping_func.py:L58-169` | API Doc |
| QLoRA_Training_Execution | `Trainer_train_qlora` | `Trainer.train` | transformers (external) | Wrapper Doc |
| QLoRA_Adapter_Serialization | `PeftModel_save_pretrained_qlora` | `save_pretrained` | `src/peft/peft_model.py:L311-459` | API Doc |

---

## Workflow: huggingface_peft_Adapter_Loading_Inference

**File:** [→](./workflows/huggingface_peft_Adapter_Loading_Inference.md)
**Description:** Loading pretrained PEFT adapters for inference, with optional merging.

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Load Base Model | Base_Model_Loading | `AutoModel.from_pretrained(base_model_id)` | External (transformers) |
| 2 | Load PEFT Adapter | Adapter_Loading | `PeftModel.from_pretrained(model, adapter_path)` | peft_model.py |
| 3 | Configure Inference | Inference_Configuration | `model.eval()`, `torch.no_grad()` | peft_model.py |
| 4 | Run Inference | Inference_Execution | `model.generate()` or forward pass | peft_model.py |
| 5 | Merge Adapter (Optional) | Adapter_Merging_Into_Base | `model.merge_and_unload()` | peft_model.py |

### Step 1: Base_Model_Loading

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Base_Model_Loading_Inference` |
| **Implementation** | `huggingface_peft_AutoModel_from_pretrained_inference` |
| **API Call** | `AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path: str, torch_dtype: torch.dtype = None, device_map: str = None, **kwargs) -> PreTrainedModel` |
| **Source Location** | `transformers` (external library) |
| **External Dependencies** | `transformers`, `torch`, `safetensors` |
| **Environment** | `huggingface_peft_Inference_Environment` |
| **Key Parameters** | `pretrained_model_name_or_path: str` - base model ID (must match adapter's base), `torch_dtype: torch.dtype` - model precision, `device_map: str` - device placement |
| **Inputs** | Base model identifier (from adapter config's `base_model_name_or_path`) |
| **Outputs** | PreTrainedModel instance |

### Step 2: Adapter_Loading

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Adapter_Loading` |
| **Implementation** | `huggingface_peft_PeftModel_from_pretrained` |
| **API Call** | `PeftModel.from_pretrained(model: PreTrainedModel, model_id: str, adapter_name: str = "default", is_trainable: bool = False, config: PeftConfig = None, revision: str = None, **kwargs) -> PeftModel` |
| **Source Location** | `src/peft/peft_model.py:L461-620` |
| **External Dependencies** | `transformers`, `safetensors`, `huggingface_hub` |
| **Environment** | `huggingface_peft_Inference_Environment` |
| **Key Parameters** | `model: PreTrainedModel` - base model from Step 1, `model_id: str` - adapter path (local or HF Hub), `adapter_name: str` - name for loaded adapter (default "default"), `is_trainable: bool` - load in training mode (False for inference), `config: PeftConfig` - override adapter config |
| **Inputs** | Base model instance, adapter checkpoint path/ID |
| **Outputs** | PeftModel with loaded adapter weights |

### Step 3: Inference_Configuration

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Inference_Configuration` |
| **Implementation** | `huggingface_peft_model_eval` |
| **API Call** | `model.eval() -> PeftModel` |
| **Source Location** | `torch.nn.Module` (PyTorch base) |
| **External Dependencies** | `torch` |
| **Environment** | `huggingface_peft_Inference_Environment` |
| **Key Parameters** | None (method call sets evaluation mode) |
| **Inputs** | PeftModel with loaded adapter |
| **Outputs** | Model in evaluation mode (dropout disabled, BatchNorm in eval mode) |

### Step 4: Inference_Execution

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Inference_Execution` |
| **Implementation** | `huggingface_peft_model_generate` |
| **API Call** | `model.generate(input_ids: torch.Tensor, attention_mask: torch.Tensor = None, max_new_tokens: int = None, do_sample: bool = False, temperature: float = 1.0, top_p: float = 1.0, **kwargs) -> torch.Tensor` |
| **Source Location** | `transformers.GenerationMixin` (external) |
| **External Dependencies** | `transformers`, `torch` |
| **Environment** | `huggingface_peft_Inference_Environment` |
| **Key Parameters** | `input_ids: torch.Tensor` - tokenized input, `max_new_tokens: int` - maximum tokens to generate, `do_sample: bool` - use sampling vs greedy, `temperature: float` - sampling temperature, `top_p: float` - nucleus sampling threshold, `top_k: int` - top-k sampling |
| **Inputs** | Tokenized input, generation parameters |
| **Outputs** | Generated token IDs |

### Step 5: Adapter_Merging_Into_Base

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Adapter_Merging_Into_Base` |
| **Implementation** | `huggingface_peft_merge_and_unload` |
| **API Call** | `PeftModel.merge_and_unload(progressbar: bool = False, safe_merge: bool = False, adapter_names: list[str] = None) -> torch.nn.Module` |
| **Source Location** | `src/peft/tuners/tuners_utils.py:L611-647` |
| **External Dependencies** | `torch`, `tqdm` |
| **Environment** | `huggingface_peft_Merge_Environment` |
| **Key Parameters** | `progressbar: bool` - show merge progress bar, `safe_merge: bool` - check for NaNs during merge (slower but safer), `adapter_names: list[str]` - specific adapters to merge (None = all active) |
| **Inputs** | PeftModel with adapter(s) |
| **Outputs** | Base model with adapter weights merged (no PEFT wrapper, standard transformers model) |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| Base_Model_Loading_Inference | `AutoModel_from_pretrained_inference` | `from_pretrained` | transformers (external) | Wrapper Doc |
| Adapter_Loading | `PeftModel_from_pretrained` | `from_pretrained` | `src/peft/peft_model.py:L461-620` | API Doc |
| Inference_Configuration | `model_eval` | `eval` | torch.nn.Module (external) | Wrapper Doc |
| Inference_Execution | `model_generate` | `generate` | transformers.GenerationMixin (external) | Wrapper Doc |
| Adapter_Merging_Into_Base | `merge_and_unload` | `merge_and_unload` | `src/peft/tuners/tuners_utils.py:L611-647` | API Doc |

---

## Workflow: huggingface_peft_Adapter_Merging

**File:** [→](./workflows/huggingface_peft_Adapter_Merging.md)
**Description:** Combining multiple trained adapters using TIES, DARE, or task arithmetic.

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Load Base Model | Base_Model_Loading | `AutoModel.from_pretrained()` | External (transformers) |
| 2 | Load Primary Adapter | Adapter_Loading | `PeftModel.from_pretrained(model, path)` | peft_model.py |
| 3 | Load Additional Adapters | Multi_Adapter_Loading | `model.load_adapter(path, adapter_name)` | peft_model.py |
| 4 | Configure Merge Strategy | Merge_Strategy_Configuration | Select TIES/DARE/linear, set weights, density | utils/merge_utils.py |
| 5 | Execute Merge | Adapter_Merge_Execution | `model.add_weighted_adapter(adapters, weights, ...)` | tuners/lora/model.py |
| 6 | Evaluate Merged | Merge_Evaluation | Run inference on test sets | External |
| 7 | Save Merged Adapter | Adapter_Serialization | `model.save_pretrained(path)` | utils/save_and_load.py |

### Step 1: Base_Model_Loading

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Base_Model_Loading_Merging` |
| **Implementation** | `huggingface_peft_AutoModel_from_pretrained_merging` |
| **API Call** | `AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path: str, **kwargs) -> PreTrainedModel` |
| **Source Location** | `transformers` (external library) |
| **External Dependencies** | `transformers`, `torch` |
| **Environment** | `huggingface_peft_Merge_Environment` |
| **Key Parameters** | `pretrained_model_name_or_path: str` - base model (must match all adapters' base) |
| **Inputs** | Base model identifier |
| **Outputs** | PreTrainedModel instance |

### Step 2: Adapter_Loading

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Primary_Adapter_Loading` |
| **Implementation** | `huggingface_peft_PeftModel_from_pretrained_primary` |
| **API Call** | `PeftModel.from_pretrained(model: PreTrainedModel, model_id: str, adapter_name: str = "default", **kwargs) -> PeftModel` |
| **Source Location** | `src/peft/peft_model.py:L461-620` |
| **External Dependencies** | `transformers`, `safetensors`, `huggingface_hub` |
| **Environment** | `huggingface_peft_Merge_Environment` |
| **Key Parameters** | `model: PreTrainedModel` - base model, `model_id: str` - first adapter path, `adapter_name: str` - name for first adapter |
| **Inputs** | Base model, primary adapter path |
| **Outputs** | PeftModel with first adapter loaded |

### Step 3: Multi_Adapter_Loading

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Multi_Adapter_Loading` |
| **Implementation** | `huggingface_peft_load_adapter` |
| **API Call** | `PeftModel.load_adapter(model_id: str, adapter_name: str, is_trainable: bool = False, torch_device: str = None, **kwargs) -> None` |
| **Source Location** | `src/peft/peft_model.py:L621-750` |
| **External Dependencies** | `safetensors`, `huggingface_hub` |
| **Environment** | `huggingface_peft_Multi_Adapter_Environment` |
| **Key Parameters** | `model_id: str` - additional adapter path, `adapter_name: str` - unique name for this adapter, `is_trainable: bool` - load with gradients (False for merging) |
| **Inputs** | Additional adapter paths, unique names |
| **Outputs** | Model with multiple adapters loaded |

### Step 4: Merge_Strategy_Configuration

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Merge_Strategy_Configuration` |
| **Implementation** | `huggingface_peft_merge_strategy_selection` |
| **API Call** | N/A (parameter configuration for Step 5) |
| **Source Location** | `src/peft/utils/merge_utils.py:L1-269` |
| **External Dependencies** | `torch` |
| **Environment** | `huggingface_peft_Merge_Environment` |
| **Key Parameters** | `combination_type: str` - merge method ("svd", "linear", "cat", "ties", "ties_svd", "dare_ties", "dare_linear", "dare_ties_svd", "dare_linear_svd", "magnitude_prune", "magnitude_prune_svd"), `weights: list[float]` - per-adapter weights, `density: float` - pruning density for TIES/DARE (0.0-1.0), `majority_sign_method: str` - "total" or "frequency" for TIES |
| **Inputs** | Merge strategy choice, weights, density |
| **Outputs** | Configuration for add_weighted_adapter call |

### Step 5: Adapter_Merge_Execution

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Adapter_Merge_Execution` |
| **Implementation** | `huggingface_peft_add_weighted_adapter` |
| **API Call** | `LoraModel.add_weighted_adapter(adapters: list[str], weights: list[float], adapter_name: str, combination_type: str = "svd", svd_rank: int = None, svd_clamp: int = None, svd_full_matrices: bool = True, svd_driver: str = None, density: float = None, majority_sign_method: str = "total") -> None` |
| **Source Location** | `src/peft/tuners/lora/model.py:L573-708` |
| **External Dependencies** | `torch`, `peft.utils.merge_utils` |
| **Environment** | `huggingface_peft_Merge_Environment` |
| **Key Parameters** | `adapters: list[str]` - adapter names to combine, `weights: list[float]` - per-adapter weights (can be negative for subtraction), `adapter_name: str` - name for merged adapter, `combination_type: str` - merge method, `density: float` - pruning density for TIES/DARE, `svd_rank: int` - rank for SVD methods, `majority_sign_method: str` - sign election for TIES |
| **Inputs** | List of adapter names, weights, combination parameters |
| **Outputs** | New merged adapter with combined weights |

### Step 6: Merge_Evaluation

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Merge_Evaluation` |
| **Implementation** | `huggingface_peft_merged_adapter_evaluation` |
| **API Call** | `model.set_adapter(merged_adapter_name)` then `model.generate()` or evaluation loop |
| **Source Location** | User-defined evaluation |
| **External Dependencies** | `torch`, `datasets`, evaluation libraries |
| **Environment** | `huggingface_peft_Evaluation_Environment` |
| **Key Parameters** | Evaluation metrics, test datasets |
| **Inputs** | Model with merged adapter active, test data |
| **Outputs** | Evaluation metrics (accuracy, perplexity, etc.) |

### Step 7: Adapter_Serialization

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Merged_Adapter_Serialization` |
| **Implementation** | `huggingface_peft_save_pretrained_merged` |
| **API Call** | `PeftModel.save_pretrained(save_directory: str, selected_adapters: list[str] = None, **kwargs) -> None` |
| **Source Location** | `src/peft/peft_model.py:L311-459` |
| **External Dependencies** | `safetensors`, `huggingface_hub` |
| **Environment** | `huggingface_peft_Save_Environment` |
| **Key Parameters** | `save_directory: str` - output path, `selected_adapters: list[str]` - specify merged adapter name |
| **Inputs** | PeftModel with merged adapter, save path |
| **Outputs** | Saved merged adapter weights and config |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| Base_Model_Loading_Merging | `AutoModel_from_pretrained_merging` | `from_pretrained` | transformers (external) | Wrapper Doc |
| Primary_Adapter_Loading | `PeftModel_from_pretrained_primary` | `from_pretrained` | `src/peft/peft_model.py:L461-620` | API Doc |
| Multi_Adapter_Loading | `load_adapter` | `load_adapter` | `src/peft/peft_model.py:L621-750` | API Doc |
| Merge_Strategy_Configuration | `merge_strategy_selection` | `ties`, `dare_ties`, `dare_linear`, `magnitude_prune` | `src/peft/utils/merge_utils.py:L144-269` | API Doc |
| Adapter_Merge_Execution | `add_weighted_adapter` | `add_weighted_adapter` | `src/peft/tuners/lora/model.py:L573-708` | API Doc |
| Merge_Evaluation | `merged_adapter_evaluation` | User-defined | N/A | Pattern Doc |
| Merged_Adapter_Serialization | `save_pretrained_merged` | `save_pretrained` | `src/peft/peft_model.py:L311-459` | API Doc |

---

## Workflow: huggingface_peft_Multi_Adapter_Management

**File:** [→](./workflows/huggingface_peft_Multi_Adapter_Management.md)
**Description:** Managing multiple adapters on a single model for multi-task inference.

### Steps Overview

| # | Step Name | Principle | Rough API | Related Files |
|---|-----------|-----------|-----------|---------------|
| 1 | Load First Adapter | Adapter_Loading | `PeftModel.from_pretrained(model, path)` | peft_model.py |
| 2 | Load Additional Adapters | Multi_Adapter_Loading | `model.load_adapter(path, adapter_name)` | peft_model.py |
| 3 | Switch Active Adapter | Adapter_Switching | `model.set_adapter(adapter_name)` | peft_model.py |
| 4 | Disable/Enable Adapters | Adapter_Enable_Disable | `model.disable_adapter()` context | peft_model.py |
| 5 | Delete Adapters | Adapter_Deletion | `model.delete_adapter(adapter_name)` | peft_model.py |
| 6 | Query State | Adapter_State_Query | `model.active_adapter`, `model.peft_config` | peft_model.py |

### Step 1: Adapter_Loading

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_First_Adapter_Loading` |
| **Implementation** | `huggingface_peft_PeftModel_from_pretrained_first` |
| **API Call** | `PeftModel.from_pretrained(model: PreTrainedModel, model_id: str, adapter_name: str = "default", is_trainable: bool = False, **kwargs) -> PeftModel` |
| **Source Location** | `src/peft/peft_model.py:L461-620` |
| **External Dependencies** | `transformers`, `safetensors`, `huggingface_hub` |
| **Environment** | `huggingface_peft_Multi_Adapter_Environment` |
| **Key Parameters** | `model: PreTrainedModel` - base model, `model_id: str` - first adapter path, `adapter_name: str` - name for first adapter, `is_trainable: bool` - load with gradients |
| **Inputs** | Base model, first adapter path |
| **Outputs** | PeftModel with first adapter loaded and active |

### Step 2: Multi_Adapter_Loading

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Additional_Adapter_Loading` |
| **Implementation** | `huggingface_peft_load_adapter_additional` |
| **API Call** | `PeftModel.load_adapter(model_id: str, adapter_name: str, is_trainable: bool = False, torch_device: str = None, **kwargs) -> None` |
| **Source Location** | `src/peft/peft_model.py:L621-750` |
| **External Dependencies** | `safetensors`, `huggingface_hub` |
| **Environment** | `huggingface_peft_Multi_Adapter_Environment` |
| **Key Parameters** | `model_id: str` - adapter path or Hub ID, `adapter_name: str` - unique name (must not conflict), `is_trainable: bool` - load with gradients, `torch_device: str` - target device |
| **Inputs** | Additional adapter checkpoint path, unique adapter name |
| **Outputs** | Model with additional adapter loaded (not active by default) |

### Step 3: Adapter_Switching

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Adapter_Switching` |
| **Implementation** | `huggingface_peft_set_adapter` |
| **API Call** | `PeftModel.set_adapter(adapter_name: Union[str, list[str]]) -> None` |
| **Source Location** | `src/peft/peft_model.py:L751-830` |
| **External Dependencies** | `torch` |
| **Environment** | `huggingface_peft_Multi_Adapter_Environment` |
| **Key Parameters** | `adapter_name: Union[str, list[str]]` - single adapter name or list for combining multiple adapters |
| **Inputs** | Adapter name(s) to activate |
| **Outputs** | Model with specified adapter(s) active (outputs summed if multiple) |

### Step 4: Adapter_Enable_Disable

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Adapter_Enable_Disable` |
| **Implementation** | `huggingface_peft_disable_adapter_context` |
| **API Call** | `with model.disable_adapter(): ...` (context manager) |
| **Source Location** | `src/peft/peft_model.py:L831-900` |
| **External Dependencies** | `torch` |
| **Environment** | `huggingface_peft_Multi_Adapter_Environment` |
| **Key Parameters** | None (context manager temporarily disables all adapters) |
| **Inputs** | PeftModel with adapters |
| **Outputs** | Within context: base model behavior (no adapter effect). After context: adapters re-enabled |

### Step 5: Adapter_Deletion

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Adapter_Deletion` |
| **Implementation** | `huggingface_peft_delete_adapter` |
| **API Call** | `PeftModel.delete_adapter(adapter_name: str) -> None` |
| **Source Location** | `src/peft/peft_model.py:L901-950` |
| **External Dependencies** | `torch` |
| **Environment** | `huggingface_peft_Multi_Adapter_Environment` |
| **Key Parameters** | `adapter_name: str` - name of adapter to remove (cannot delete active adapter) |
| **Inputs** | Adapter name to delete |
| **Outputs** | Model without the specified adapter (memory freed) |

### Step 6: Adapter_State_Query

| Attribute | Value |
|-----------|-------|
| **Principle** | `huggingface_peft_Adapter_State_Query` |
| **Implementation** | `huggingface_peft_query_adapter_state` |
| **API Call** | `model.active_adapter` (property), `model.peft_config` (dict), `model.get_model_status()` |
| **Source Location** | `src/peft/peft_model.py:L180-250` |
| **External Dependencies** | None |
| **Environment** | `huggingface_peft_Multi_Adapter_Environment` |
| **Key Parameters** | `model.active_adapter: Union[str, list[str]]` - currently active adapter(s), `model.peft_config: dict[str, PeftConfig]` - all adapter configs |
| **Inputs** | PeftModel instance |
| **Outputs** | Active adapter name(s), configuration dict, model status |

### Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| First_Adapter_Loading | `PeftModel_from_pretrained_first` | `from_pretrained` | `src/peft/peft_model.py:L461-620` | API Doc |
| Additional_Adapter_Loading | `load_adapter_additional` | `load_adapter` | `src/peft/peft_model.py:L621-750` | API Doc |
| Adapter_Switching | `set_adapter` | `set_adapter` | `src/peft/peft_model.py:L751-830` | API Doc |
| Adapter_Enable_Disable | `disable_adapter_context` | `disable_adapter` | `src/peft/peft_model.py:L831-900` | API Doc |
| Adapter_Deletion | `delete_adapter` | `delete_adapter` | `src/peft/peft_model.py:L901-950` | API Doc |
| Adapter_State_Query | `query_adapter_state` | `active_adapter`, `peft_config` | `src/peft/peft_model.py:L180-250` | API Doc |

---

## Global Implementation Extraction Guide

| Principle | Implementation | API | Source | Type |
|-----------|----------------|-----|--------|------|
| Base_Model_Loading | `AutoModel_from_pretrained` | `AutoModel.from_pretrained` | transformers (external) | Wrapper Doc |
| LoRA_Configuration | `LoraConfig_init` | `LoraConfig()` | `src/peft/tuners/lora/config.py:L47-300` | API Doc |
| PEFT_Model_Creation | `get_peft_model` | `get_peft_model()` | `src/peft/mapping_func.py:L58-169` | API Doc |
| Adapter_Serialization | `PeftModel_save_pretrained` | `PeftModel.save_pretrained()` | `src/peft/peft_model.py:L311-459` | API Doc |
| Adapter_Loading | `PeftModel_from_pretrained` | `PeftModel.from_pretrained()` | `src/peft/peft_model.py:L461-620` | API Doc |
| Multi_Adapter_Loading | `load_adapter` | `PeftModel.load_adapter()` | `src/peft/peft_model.py:L621-750` | API Doc |
| Adapter_Switching | `set_adapter` | `PeftModel.set_adapter()` | `src/peft/peft_model.py:L751-830` | API Doc |
| Adapter_Merge_Execution | `add_weighted_adapter` | `LoraModel.add_weighted_adapter()` | `src/peft/tuners/lora/model.py:L573-708` | API Doc |
| Merge_Strategy_Configuration | `merge_utils` | `ties()`, `dare_ties()`, `dare_linear()`, `magnitude_prune()` | `src/peft/utils/merge_utils.py:L144-269` | API Doc |
| Kbit_Training_Preparation | `prepare_model_for_kbit_training` | `prepare_model_for_kbit_training()` | `src/peft/utils/other.py:L250-320` | API Doc |
| Quantization_Configuration | `BitsAndBytesConfig` | `BitsAndBytesConfig()` | transformers (external) | Wrapper Doc |
| Layer_Merge | `Linear_merge` | `Linear.merge()` | `src/peft/tuners/lora/layer.py:L667-732` | API Doc |
| Layer_Unmerge | `Linear_unmerge` | `Linear.unmerge()` | `src/peft/tuners/lora/layer.py:L734-756` | API Doc |
| Merge_And_Unload | `merge_and_unload` | `BaseTuner.merge_and_unload()` | `src/peft/tuners/tuners_utils.py:L611-647` | API Doc |
| State_Dict_Get | `get_peft_model_state_dict` | `get_peft_model_state_dict()` | `src/peft/utils/save_and_load.py:L57-353` | API Doc |
| State_Dict_Set | `set_peft_model_state_dict` | `set_peft_model_state_dict()` | `src/peft/utils/save_and_load.py:L405-588` | API Doc |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation

**Implementation Types:**
- **API Doc:** Function/class defined in this repo with full implementation
- **Wrapper Doc:** External library with repo-specific usage patterns
- **Pattern Doc:** User-defined interface or pattern
- **External Tool Doc:** CLI or external tool
