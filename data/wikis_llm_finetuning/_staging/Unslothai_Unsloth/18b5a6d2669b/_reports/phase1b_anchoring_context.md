# Phase 1b: WorkflowIndex Enrichment Report

## Summary
- Workflows enriched: 4
- Steps with detailed tables: 28
- Source files traced: 8

## Enrichment Details

| Workflow | Steps Enriched | APIs Traced | Line Numbers Found |
|----------|----------------|-------------|-------------------|
| Unslothai_Unsloth_QLoRA_Finetuning | 6 | 6 | Yes |
| Unslothai_Unsloth_GRPO_Training | 9 | 9 | Yes |
| Unslothai_Unsloth_Vision_Finetuning | 7 | 7 | Yes |
| Unslothai_Unsloth_GGUF_Export | 6 | 6 | Yes |

## Implementation Types Found

| Type | Count | Examples |
|------|-------|----------|
| API Doc | 16 | `FastLanguageModel.from_pretrained`, `get_peft_model`, `get_chat_template`, `save_pretrained_merged`, `save_to_gguf`, `FastVisionModel.from_pretrained`, `FastBaseModel.get_peft_model` |
| Wrapper Doc | 6 | `SFTTrainer`, `UnslothTrainingArguments`, `UnslothGRPOTrainer`, `UnslothGRPOConfig`, `UnslothVisionDataCollator`, `SFTTrainer_vision` |
| Pattern Doc | 5 | `reward_function_pattern`, `dataset_mapping_pattern`, `multimodal_dataset_pattern`, `OLLAMA_TEMPLATES` |
| External Tool Doc | 1 | `llama-cli` (llama.cpp GGUF validation) |

## Source Files Traced

| File | Lines | APIs Extracted |
|------|-------|----------------|
| `unsloth/models/loader.py` | L123-700, L702-900 | `FastLanguageModel.from_pretrained`, `FastVisionModel.from_pretrained` |
| `unsloth/models/llama.py` | L2577-3100 | `FastLanguageModel.get_peft_model` |
| `unsloth/models/vision.py` | L321-918, L921-1076, L1190-1250 | `FastBaseModel.from_pretrained`, `FastBaseModel.get_peft_model`, `for_training`, `for_inference` |
| `unsloth/chat_templates.py` | L2123-2400 | `get_chat_template` |
| `unsloth/save.py` | L104-131, L235-860, L1070-1300, L1800-2000 | `unsloth_save_model`, `save_to_gguf`, `ALLOWED_QUANTS`, `push_to_hub_gguf` |
| `unsloth/trainer.py` | L133-137, L182-408 | `UnslothTrainingArguments`, `UnslothTrainer` |
| `unsloth/models/rl.py` | L240-700, L309-334 | `UnslothGRPOTrainer`, `UnslothGRPOConfig` |
| `unsloth/ollama_template_mappers.py` | L1-500 | `OLLAMA_TEMPLATES` mapping |

## API Signatures Extracted

### FastLanguageModel.from_pretrained (loader.py:L123-700)
```python
def from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
    load_in_8bit = False,
    load_in_16bit = False,
    full_finetuning = False,
    token = None,
    device_map = "sequential",
    rope_scaling = None,
    fix_tokenizer = True,
    trust_remote_code = False,
    use_gradient_checkpointing = "unsloth",
    resize_model_vocab = None,
    revision = None,
    use_exact_model_name = False,
    offload_embedding = False,
    float32_mixed_precision = None,
    fast_inference = False,
    gpu_memory_utilization = 0.5,
    float8_kv_cache = False,
    random_state = 3407,
    max_lora_rank = 64,
    disable_log_stats = True,
    qat_scheme = None,
    load_in_fp8 = False,
    unsloth_tiled_mlp = False,
    **kwargs,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]
```

### FastLanguageModel.get_peft_model (llama.py:L2577-3100)
```python
def get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0.0,
    bias = "none",
    layers_to_transform = None,
    layers_pattern = None,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    max_seq_length = 2048,
    use_rslora = False,
    modules_to_save = None,
    init_lora_weights = True,
    loftq_config = {},
    temporary_location = "_unsloth_temporary_saved_buffers",
    qat_scheme = None,
    ensure_weight_tying = False,
    **kwargs,
) -> PeftModelForCausalLM
```

### get_chat_template (chat_templates.py:L2123-2400)
```python
def get_chat_template(
    tokenizer,
    chat_template = "chatml",
    mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    map_eos_token = True,
    system_message = None,
) -> PreTrainedTokenizer
```

### save_to_gguf (save.py:L1070-1300)
```python
def save_to_gguf(
    model_name: str,
    model_type: str,
    model_dtype: str,
    is_sentencepiece: bool = False,
    model_directory: str = "unsloth_finetuned_model",
    quantization_method = "fast_quantized",
    first_conversion: str = None,
    is_vlm: bool = False,
    is_gpt_oss: bool = False,
) -> str
```

### FastBaseModel.get_peft_model (vision.py:L921-1076)
```python
def get_peft_model(
    model,
    r = 16,
    target_modules = None,
    lora_alpha = 16,
    lora_dropout = 0.0,
    bias = "none",
    finetune_vision_layers = True,
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    layers_to_transform = None,
    layers_pattern = None,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    max_seq_length = 2048,
    use_rslora = False,
    modules_to_save = None,
    init_lora_weights = True,
    loftq_config = {},
    task_type = TaskType.CAUSAL_LM,
    temporary_location = "_unsloth_temporary_saved_buffers",
    qat_scheme = None,
    ensure_weight_tying = False,
    **kwargs,
) -> PeftModelForCausalLM
```

## Issues Found
- `UnslothGRPOTrainer` and `UnslothGRPOConfig` are dynamically generated via template string execution in `rl.py`, making exact line numbers approximate
- `UnslothVisionDataCollator` is imported from `unsloth_zoo.vision_utils` (external dependency)
- `train_on_responses_only` is imported from `unsloth_zoo.dataset_utils` (external dependency)
- `push_to_hub_gguf` line numbers are approximate (L1800-2000) due to interleaved functions in save.py

## External Dependencies Identified

| Dependency | Usage |
|------------|-------|
| `transformers` | Base model loading, tokenizers, training arguments |
| `peft` | LoRA adapters, PeftModelForCausalLM |
| `trl` | SFTTrainer, GRPOTrainer base classes |
| `bitsandbytes` | 4-bit/8-bit quantization |
| `vllm` | Fast inference for GRPO training |
| `llama.cpp` | GGUF conversion and quantization |
| `unsloth_zoo` | Dataset utilities, vision utilities |

## Ready for Phase 2
- [x] All Step tables complete
- [x] All source locations verified
- [x] Implementation Extraction Guides complete
- [x] No `<!-- ENRICHMENT NEEDED -->` comments remain
- [x] Cross-workflow dependencies documented
- [x] Environment requirements documented
