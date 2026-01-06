# Environment Index: huggingface_transformers

> Tracks Environment pages and which pages require them.
> **Update IMMEDIATELY** after creating or modifying a Environment page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| huggingface_transformers_Pipeline_Environment | [→](./environments/huggingface_transformers_Pipeline_Environment.md) | ✅Impl:huggingface_transformers_Pipeline_factory_function, ✅Impl:huggingface_transformers_AutoProcessor_initialization, ✅Impl:huggingface_transformers_Pipeline_model_initialization, ✅Impl:huggingface_transformers_Pipeline_preprocess, ✅Impl:huggingface_transformers_Pipeline_forward_pass, ✅Impl:huggingface_transformers_Pipeline_postprocess | Python 3.10+, PyTorch 2.2+, tokenizers for pipeline inference |
| huggingface_transformers_Training_Environment | [→](./environments/huggingface_transformers_Training_Environment.md) | ✅Impl:huggingface_transformers_TrainingArguments_setup, ✅Impl:huggingface_transformers_DataCollator_usage, ✅Impl:huggingface_transformers_Trainer_init, ✅Impl:huggingface_transformers_Optimizer_creation, ✅Impl:huggingface_transformers_Training_execution, ✅Impl:huggingface_transformers_Evaluate, ✅Impl:huggingface_transformers_Model_saving | GPU training with accelerate 1.1.0+, optional DeepSpeed/FSDP |
| huggingface_transformers_Loading_Environment | [→](./environments/huggingface_transformers_Loading_Environment.md) | ✅Impl:huggingface_transformers_PretrainedConfig_from_pretrained, ✅Impl:huggingface_transformers_Checkpoint_file_resolution, ✅Impl:huggingface_transformers_Quantizer_setup, ✅Impl:huggingface_transformers_Model_initialization, ✅Impl:huggingface_transformers_Weight_loading, ✅Impl:huggingface_transformers_Accelerate_dispatch, ✅Impl:huggingface_transformers_Post_init_processing | Model loading with safetensors, device_map, quantization |
| huggingface_transformers_Tokenization_Environment | [→](./environments/huggingface_transformers_Tokenization_Environment.md) | ✅Impl:huggingface_transformers_PreTrainedTokenizerBase_from_pretrained, ✅Impl:huggingface_transformers_Vocab_file_loading, ✅Impl:huggingface_transformers_Normalizer_application, ✅Impl:huggingface_transformers_PreTokenizer_application, ✅Impl:huggingface_transformers_Tokenizer_encode, ✅Impl:huggingface_transformers_Convert_tokens_to_ids, ✅Impl:huggingface_transformers_Batch_padding, ✅Impl:huggingface_transformers_BatchEncoding_creation | Fast tokenizers 0.22.0+, optional sentencepiece/tiktoken |
| huggingface_transformers_Distributed_Environment | [→](./environments/huggingface_transformers_Distributed_Environment.md) | ✅Impl:huggingface_transformers_Process_group_initialization, ✅Impl:huggingface_transformers_TensorParallel_from_pretrained, ✅Impl:huggingface_transformers_FSDP_wrapping, ✅Impl:huggingface_transformers_DistributedSampler_usage, ✅Impl:huggingface_transformers_Context_parallel_execution, ✅Impl:huggingface_transformers_AllReduce_gradients, ✅Impl:huggingface_transformers_Optimizer_step, ✅Impl:huggingface_transformers_DCP_save | Multi-GPU with NCCL, DeviceMesh, TP/DP/CP parallelism |
| huggingface_transformers_Quantization_Environment | [→](./environments/huggingface_transformers_Quantization_Environment.md) | ✅Impl:huggingface_transformers_BitsAndBytesConfig_setup, ✅Impl:huggingface_transformers_AutoHfQuantizer_dispatch, ✅Impl:huggingface_transformers_Quantizer_validate_environment, ✅Impl:huggingface_transformers_Quantizer_preprocess, ✅Impl:huggingface_transformers_Quantizer_convert_weights, ✅Impl:huggingface_transformers_Skip_modules_handling, ✅Impl:huggingface_transformers_Quantizer_postprocess | 4-bit/8-bit quantization with bitsandbytes, GPTQ, AWQ |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
