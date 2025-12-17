# Principle Index: huggingface_transformers

> Index of Principle pages for the huggingface_transformers wiki.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| huggingface_transformers_Configuration_Loading | [→](./principles/huggingface_transformers_Configuration_Loading.md) | ✅Impl:huggingface_transformers_AutoConfig_from_pretrained, ✅Workflow:huggingface_transformers_Model_Loading | Load model configuration from Hub |
| huggingface_transformers_Checkpoint_Discovery | [→](./principles/huggingface_transformers_Checkpoint_Discovery.md) | ✅Impl:huggingface_transformers_get_checkpoint_shard_files, ✅Workflow:huggingface_transformers_Model_Loading | Locate sharded checkpoint files |
| huggingface_transformers_Quantization_Configuration | [→](./principles/huggingface_transformers_Quantization_Configuration.md) | ✅Impl:huggingface_transformers_get_hf_quantizer, ✅Workflow:huggingface_transformers_Model_Loading | Set up quantization method |
| huggingface_transformers_Model_Instantiation | [→](./principles/huggingface_transformers_Model_Instantiation.md) | ✅Impl:huggingface_transformers_PreTrainedModel_from_config, ✅Workflow:huggingface_transformers_Model_Loading | Create model on meta device |
| huggingface_transformers_Weight_Loading | [→](./principles/huggingface_transformers_Weight_Loading.md) | ✅Impl:huggingface_transformers_load_state_dict_in_model, ✅Workflow:huggingface_transformers_Model_Loading | Load and materialize weights |
| huggingface_transformers_Model_Post_Processing | [→](./principles/huggingface_transformers_Model_Post_Processing.md) | ✅Impl:huggingface_transformers_tie_weights, ✅Workflow:huggingface_transformers_Model_Loading | Finalize model (tying, adapters) |
| huggingface_transformers_Training_Arguments | [→](./principles/huggingface_transformers_Training_Arguments.md) | ✅Impl:huggingface_transformers_TrainingArguments, ✅Workflow:huggingface_transformers_Training | Hyperparameter configuration |
| huggingface_transformers_Dataset_Preparation | [→](./principles/huggingface_transformers_Dataset_Preparation.md) | ✅Impl:huggingface_transformers_Dataset_Tokenization, ✅Workflow:huggingface_transformers_Training | Tokenize and format datasets |
| huggingface_transformers_Data_Collation | [→](./principles/huggingface_transformers_Data_Collation.md) | ✅Impl:huggingface_transformers_DataCollatorWithPadding, ✅Workflow:huggingface_transformers_Training | Batch assembly with padding |
| huggingface_transformers_Trainer_Initialization | [→](./principles/huggingface_transformers_Trainer_Initialization.md) | ✅Impl:huggingface_transformers_Trainer_init, ✅Workflow:huggingface_transformers_Training | Set up training orchestrator |
| huggingface_transformers_Training_Loop | [→](./principles/huggingface_transformers_Training_Loop.md) | ✅Impl:huggingface_transformers_Trainer_train, ✅Workflow:huggingface_transformers_Training | Execute forward/backward pass |
| huggingface_transformers_Evaluation_Checkpointing | [→](./principles/huggingface_transformers_Evaluation_Checkpointing.md) | ✅Impl:huggingface_transformers_Trainer_evaluate, ✅Workflow:huggingface_transformers_Training | Measure model performance |
| huggingface_transformers_Model_Export | [→](./principles/huggingface_transformers_Model_Export.md) | ✅Impl:huggingface_transformers_Trainer_save_model, ✅Workflow:huggingface_transformers_Training | Serialize trained model |
| huggingface_transformers_Task_Resolution | [→](./principles/huggingface_transformers_Task_Resolution.md) | ✅Impl:huggingface_transformers_check_task, ✅Workflow:huggingface_transformers_Pipeline_Inference | Map task string to pipeline class |
| huggingface_transformers_Pipeline_Component_Loading | [→](./principles/huggingface_transformers_Pipeline_Component_Loading.md) | ✅Impl:huggingface_transformers_pipeline_load_model, ✅Workflow:huggingface_transformers_Pipeline_Inference | Load model and processors |
| huggingface_transformers_Pipeline_Instantiation | [→](./principles/huggingface_transformers_Pipeline_Instantiation.md) | ✅Impl:huggingface_transformers_pipeline_factory, ✅Workflow:huggingface_transformers_Pipeline_Inference | Create configured pipeline |
| huggingface_transformers_Pipeline_Preprocessing | [→](./principles/huggingface_transformers_Pipeline_Preprocessing.md) | ✅Impl:huggingface_transformers_Pipeline_preprocess, ✅Workflow:huggingface_transformers_Pipeline_Inference | Transform inputs to tensors |
| huggingface_transformers_Pipeline_Model_Forward | [→](./principles/huggingface_transformers_Pipeline_Model_Forward.md) | ✅Impl:huggingface_transformers_Pipeline_forward, ✅Workflow:huggingface_transformers_Pipeline_Inference | Execute model inference |
| huggingface_transformers_Pipeline_Postprocessing | [→](./principles/huggingface_transformers_Pipeline_Postprocessing.md) | ✅Impl:huggingface_transformers_Pipeline_postprocess, ✅Workflow:huggingface_transformers_Pipeline_Inference | Format outputs for users |
| huggingface_transformers_Tokenizer_Loading | [→](./principles/huggingface_transformers_Tokenizer_Loading.md) | ✅Impl:huggingface_transformers_AutoTokenizer_from_pretrained, ✅Workflow:huggingface_transformers_Tokenization | Load pre-trained tokenizer |
| huggingface_transformers_Special_Tokens | [→](./principles/huggingface_transformers_Special_Tokens.md) | ✅Impl:huggingface_transformers_add_special_tokens, ✅Workflow:huggingface_transformers_Tokenization | Add control tokens to vocab |
| huggingface_transformers_Text_Encoding | [→](./principles/huggingface_transformers_Text_Encoding.md) | ✅Impl:huggingface_transformers_tokenizer_call, ✅Workflow:huggingface_transformers_Tokenization | Convert text to token IDs |
| huggingface_transformers_Padding_Truncation | [→](./principles/huggingface_transformers_Padding_Truncation.md) | ✅Impl:huggingface_transformers_pad_truncate, ✅Workflow:huggingface_transformers_Tokenization | Uniform sequence lengths |
| huggingface_transformers_Chat_Templates | [→](./principles/huggingface_transformers_Chat_Templates.md) | ✅Impl:huggingface_transformers_apply_chat_template, ✅Workflow:huggingface_transformers_Tokenization | Format conversations |
| huggingface_transformers_Text_Decoding | [→](./principles/huggingface_transformers_Text_Decoding.md) | ✅Impl:huggingface_transformers_tokenizer_decode, ✅Workflow:huggingface_transformers_Tokenization | Convert tokens to text |
| huggingface_transformers_Quantization_Method_Selection | [→](./principles/huggingface_transformers_Quantization_Method_Selection.md) | ✅Impl:huggingface_transformers_QuantizationMethod, ✅Workflow:huggingface_transformers_Quantization | Choose quantization method |
| huggingface_transformers_Quantization_Config_Setup | [→](./principles/huggingface_transformers_Quantization_Config_Setup.md) | ✅Impl:huggingface_transformers_BitsAndBytesConfig, ✅Workflow:huggingface_transformers_Quantization | Configure quant parameters |
| huggingface_transformers_Quantizer_Initialization | [→](./principles/huggingface_transformers_Quantizer_Initialization.md) | ✅Impl:huggingface_transformers_get_hf_quantizer_init, ✅Workflow:huggingface_transformers_Quantization | Instantiate quantizer class |
| huggingface_transformers_Quantized_Model_Preparation | [→](./principles/huggingface_transformers_Quantized_Model_Preparation.md) | ✅Impl:huggingface_transformers_quantizer_preprocess_model, ✅Workflow:huggingface_transformers_Quantization | Replace layers for quantization |
| huggingface_transformers_Quantized_Weight_Loading | [→](./principles/huggingface_transformers_Quantized_Weight_Loading.md) | ✅Impl:huggingface_transformers_quantizer_postprocess_model, ✅Workflow:huggingface_transformers_Quantization | Load quantized weights |
| huggingface_transformers_Quantized_Runtime_Optimization | [→](./principles/huggingface_transformers_Quantized_Runtime_Optimization.md) | ✅Impl:huggingface_transformers_quantizer_runtime_config, ✅Workflow:huggingface_transformers_Quantization | Configure optimized kernels |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
