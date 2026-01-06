# Principle Index: huggingface_transformers

> Tracks Principle pages and their connections to Implementations, Workflows, etc.
> **Update IMMEDIATELY** after creating or modifying a Principle page.

## Summary

| Workflow | Principles | Status |
|----------|------------|--------|
| Pipeline_Inference | 6 | ✅ Complete |
| Model_Training_Trainer | 7 | ✅ Complete |
| Model_Loading | 7 | ✅ Complete |
| Tokenization_Pipeline | 8 | ✅ Complete |
| Distributed_Training_3D_Parallelism | 8 | ✅ Complete |
| Model_Quantization | 7 | ✅ Complete |
| **Total** | **43** | ✅ |

---

## Pipeline_Inference Principles

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| Task_Model_Resolution | [→](./principles/huggingface_transformers_Task_Model_Resolution.md) | ✅Impl:Pipeline_factory_function | Resolves task to model/pipeline |
| Processor_Loading | [→](./principles/huggingface_transformers_Processor_Loading.md) | ✅Impl:AutoProcessor_initialization | Loads tokenizer/image processor |
| Pipeline_Model_Loading | [→](./principles/huggingface_transformers_Pipeline_Model_Loading.md) | ✅Impl:Pipeline_model_initialization | Device placement in pipelines |
| Pipeline_Preprocessing | [→](./principles/huggingface_transformers_Pipeline_Preprocessing.md) | ✅Impl:Pipeline_preprocess | Input preprocessing pattern |
| Pipeline_Forward | [→](./principles/huggingface_transformers_Pipeline_Forward.md) | ✅Impl:Pipeline_forward_pass | Model forward pass pattern |
| Pipeline_Postprocessing | [→](./principles/huggingface_transformers_Pipeline_Postprocessing.md) | ✅Impl:Pipeline_postprocess | Output postprocessing pattern |

---

## Model_Training_Trainer Principles

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| TrainingArguments_Configuration | [→](./principles/huggingface_transformers_TrainingArguments_Configuration.md) | ✅Impl:TrainingArguments_setup | Training hyperparameters |
| Dataset_Preparation | [→](./principles/huggingface_transformers_Dataset_Preparation.md) | ✅Impl:DataCollator_usage | Data collation and batching |
| Trainer_Initialization | [→](./principles/huggingface_transformers_Trainer_Initialization.md) | ✅Impl:Trainer_init | Trainer setup |
| Optimizer_Scheduler_Setup | [→](./principles/huggingface_transformers_Optimizer_Scheduler_Setup.md) | ✅Impl:Optimizer_creation | Optimizer and LR scheduler |
| Training_Loop | [→](./principles/huggingface_transformers_Training_Loop.md) | ✅Impl:Training_execution | Main training loop |
| Evaluation_Loop | [→](./principles/huggingface_transformers_Evaluation_Loop.md) | ✅Impl:Evaluate | Evaluation loop |
| Checkpoint_Saving | [→](./principles/huggingface_transformers_Checkpoint_Saving.md) | ✅Impl:Model_saving | Checkpoint saving |

---

## Model_Loading Principles

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| Configuration_Resolution | [→](./principles/huggingface_transformers_Configuration_Resolution.md) | ✅Impl:PretrainedConfig_from_pretrained | Config loading |
| Checkpoint_Discovery | [→](./principles/huggingface_transformers_Checkpoint_Discovery.md) | ✅Impl:Checkpoint_file_resolution | File discovery |
| Quantization_Configuration | [→](./principles/huggingface_transformers_Quantization_Configuration.md) | ✅Impl:Quantizer_setup | Quantization config |
| Model_Instantiation | [→](./principles/huggingface_transformers_Model_Instantiation.md) | ✅Impl:Model_initialization | Model construction |
| State_Dict_Loading | [→](./principles/huggingface_transformers_State_Dict_Loading.md) | ✅Impl:Weight_loading | Weight loading |
| Device_Placement | [→](./principles/huggingface_transformers_Device_Placement.md) | ✅Impl:Accelerate_dispatch | Device mapping |
| Post_Loading_Hooks | [→](./principles/huggingface_transformers_Post_Loading_Hooks.md) | ✅Impl:Post_init_processing | Post-load operations |

---

## Tokenization_Pipeline Principles

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| Tokenizer_Loading | [→](./principles/huggingface_transformers_Tokenizer_Loading.md) | ✅Impl:PreTrainedTokenizerBase_from_pretrained | Tokenizer loading |
| Vocabulary_Initialization | [→](./principles/huggingface_transformers_Vocabulary_Initialization.md) | ✅Impl:Vocab_file_loading | Vocab setup |
| Text_Normalization | [→](./principles/huggingface_transformers_Text_Normalization.md) | ✅Impl:Normalizer_application | Text normalization |
| Pre_Tokenization | [→](./principles/huggingface_transformers_Pre_Tokenization.md) | ✅Impl:PreTokenizer_application | Pre-tokenization |
| Subword_Tokenization | [→](./principles/huggingface_transformers_Subword_Tokenization.md) | ✅Impl:Tokenizer_encode | Subword encoding |
| Token_ID_Conversion | [→](./principles/huggingface_transformers_Token_ID_Conversion.md) | ✅Impl:Convert_tokens_to_ids | Token-ID mapping |
| Padding_Truncation | [→](./principles/huggingface_transformers_Padding_Truncation.md) | ✅Impl:Batch_padding | Padding/truncation |
| Encoding_Creation | [→](./principles/huggingface_transformers_Encoding_Creation.md) | ✅Impl:BatchEncoding_creation | BatchEncoding output |

---

## Distributed_Training_3D_Parallelism Principles

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| Distributed_Init | [→](./principles/huggingface_transformers_Distributed_Init.md) | ✅Impl:Process_group_initialization | Process group init |
| TP_Model_Loading | [→](./principles/huggingface_transformers_TP_Model_Loading.md) | ✅Impl:TensorParallel_from_pretrained | Tensor parallel loading |
| Data_Parallelism_Setup | [→](./principles/huggingface_transformers_Data_Parallelism_Setup.md) | ✅Impl:FSDP_wrapping | FSDP setup |
| Distributed_Dataset | [→](./principles/huggingface_transformers_Distributed_Dataset.md) | ✅Impl:DistributedSampler_usage | Distributed data loading |
| Context_Parallelism | [→](./principles/huggingface_transformers_Context_Parallelism.md) | ✅Impl:Context_parallel_execution | Context parallelism |
| Gradient_Synchronization | [→](./principles/huggingface_transformers_Gradient_Synchronization.md) | ✅Impl:AllReduce_gradients | Gradient sync |
| Distributed_Optimizer_Step | [→](./principles/huggingface_transformers_Distributed_Optimizer_Step.md) | ✅Impl:Optimizer_step | Distributed optimizer |
| Distributed_Checkpointing | [→](./principles/huggingface_transformers_Distributed_Checkpointing.md) | ✅Impl:DCP_save | Distributed checkpoint |

---

## Model_Quantization Principles

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| Quantization_Config | [→](./principles/huggingface_transformers_Quantization_Config.md) | ✅Impl:BitsAndBytesConfig_setup | Quantization config |
| Quantizer_Selection | [→](./principles/huggingface_transformers_Quantizer_Selection.md) | ✅Impl:AutoHfQuantizer_dispatch | Quantizer dispatch |
| Quantization_Validation | [→](./principles/huggingface_transformers_Quantization_Validation.md) | ✅Impl:Quantizer_validate_environment | Environment validation |
| Weight_Quantization | [→](./principles/huggingface_transformers_Weight_Quantization.md) | ✅Impl:Quantizer_preprocess | Weight preprocessing |
| Linear_Layer_Replacement | [→](./principles/huggingface_transformers_Linear_Layer_Replacement.md) | ✅Impl:Quantizer_convert_weights | Layer replacement |
| Module_Targeting | [→](./principles/huggingface_transformers_Module_Targeting.md) | ✅Impl:Skip_modules_handling | Module targeting |
| Post_Quantization_Setup | [→](./principles/huggingface_transformers_Post_Quantization_Setup.md) | ✅Impl:Quantizer_postprocess | Post-quant setup |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
