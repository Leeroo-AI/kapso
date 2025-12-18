# Implementation Index: huggingface_peft

> Tracks Implementation pages and their connections to Principles, Environments, etc.
> **Update IMMEDIATELY** after creating or modifying an Implementation page.

## Pages

| Page | File | Connections | Source | Notes |
|------|------|-------------|--------|-------|
| huggingface_peft_AutoModelForCausalLM_from_pretrained | [→](./implementations/huggingface_peft_AutoModelForCausalLM_from_pretrained.md) | ✅Principle:Base_Model_Loading, ✅Env:huggingface_peft_Core_Environment | transformers (external) | Wrapper Doc - Base model loading for LoRA |
| huggingface_peft_LoraConfig_init | [→](./implementations/huggingface_peft_LoraConfig_init.md) | ✅Principle:LoRA_Configuration, ✅Env:huggingface_peft_Core_Environment, ✅Env:huggingface_peft_LoftQ_Environment, ✅Heuristic:huggingface_peft_LoRA_Rank_Selection, ✅Heuristic:huggingface_peft_Target_Module_Selection, ✅Heuristic:huggingface_peft_DoRA_Overhead | config.py:L321-879 | API Doc - LoRA hyperparameter configuration |
| huggingface_peft_get_peft_model | [→](./implementations/huggingface_peft_get_peft_model.md) | ✅Principle:PEFT_Model_Creation, ✅Env:huggingface_peft_Core_Environment, ✅Env:huggingface_peft_GPTQ_Environment | mapping_func.py:L30-128 | API Doc - Adapter injection factory |
| huggingface_peft_model_train_mode | [→](./implementations/huggingface_peft_model_train_mode.md) | ✅Principle:Training_Preparation, ✅Env:huggingface_peft_Core_Environment | torch.nn.Module | Wrapper Doc - Training mode setup |
| huggingface_peft_Trainer_train | [→](./implementations/huggingface_peft_Trainer_train.md) | ✅Principle:Training_Execution, ✅Env:huggingface_peft_Core_Environment, ✅Heuristic:huggingface_peft_Gradient_Checkpointing | transformers (external) | Wrapper Doc - HuggingFace Trainer integration |
| huggingface_peft_PeftModel_save_pretrained | [→](./implementations/huggingface_peft_PeftModel_save_pretrained.md) | ✅Principle:Adapter_Serialization, ✅Env:huggingface_peft_Core_Environment | peft_model.py:L190-386 | API Doc - Adapter checkpoint saving |
| huggingface_peft_BitsAndBytesConfig_4bit | [→](./implementations/huggingface_peft_BitsAndBytesConfig_4bit.md) | ✅Principle:Quantization_Configuration, ✅Env:huggingface_peft_Quantization_Environment, ✅Heuristic:huggingface_peft_Quantized_Merge_Warning | transformers (external) | Wrapper Doc - 4-bit quantization setup |
| huggingface_peft_prepare_model_for_kbit_training | [→](./implementations/huggingface_peft_prepare_model_for_kbit_training.md) | ✅Principle:Kbit_Training_Preparation, ✅Env:huggingface_peft_Quantization_Environment, ✅Heuristic:huggingface_peft_Gradient_Checkpointing | other.py:L130-215 | API Doc - QLoRA model preparation |
| huggingface_peft_PeftModel_from_pretrained | [→](./implementations/huggingface_peft_PeftModel_from_pretrained.md) | ✅Principle:Adapter_Loading, ✅Env:huggingface_peft_Core_Environment | peft_model.py:L388-604 | API Doc - Adapter loading for inference |
| huggingface_peft_merge_and_unload | [→](./implementations/huggingface_peft_merge_and_unload.md) | ✅Principle:Adapter_Merging_Into_Base, ✅Env:huggingface_peft_Core_Environment, ✅Heuristic:huggingface_peft_Quantized_Merge_Warning | tuners_utils.py:L611-647 | API Doc - Merge adapter into base model |
| huggingface_peft_load_adapter | [→](./implementations/huggingface_peft_load_adapter.md) | ✅Principle:Multi_Adapter_Loading, ✅Env:huggingface_peft_Core_Environment | peft_model.py:L1309-1475 | API Doc - Multi-adapter loading |
| huggingface_peft_add_weighted_adapter | [→](./implementations/huggingface_peft_add_weighted_adapter.md) | ✅Principle:Adapter_Merge_Execution, ✅Env:huggingface_peft_Core_Environment | lora/model.py:L573-708 | API Doc - TIES/DARE adapter merging |
| huggingface_peft_set_adapter | [→](./implementations/huggingface_peft_set_adapter.md) | ✅Principle:Adapter_Switching, ✅Env:huggingface_peft_Core_Environment | peft_model.py:L1477-1504 | API Doc - Active adapter selection |
| huggingface_peft_disable_adapter_context | [→](./implementations/huggingface_peft_disable_adapter_context.md) | ✅Principle:Adapter_Enable_Disable, ✅Env:huggingface_peft_Core_Environment | peft_model.py:L940-992 | API Doc - Temporary adapter bypass |
| huggingface_peft_delete_adapter | [→](./implementations/huggingface_peft_delete_adapter.md) | ✅Principle:Adapter_Deletion, ✅Env:huggingface_peft_Core_Environment | peft_model.py:L1083-1101 | API Doc - Adapter removal |
| huggingface_peft_query_adapter_state | [→](./implementations/huggingface_peft_query_adapter_state.md) | ✅Principle:Adapter_State_Query, ✅Env:huggingface_peft_Core_Environment | peft_model.py:L180-250 | API Doc - Adapter state introspection |
| huggingface_peft_model_eval | [→](./implementations/huggingface_peft_model_eval.md) | ✅Principle:Inference_Configuration, ✅Env:huggingface_peft_Core_Environment | torch.nn.Module | Wrapper Doc - Eval mode setup |
| huggingface_peft_model_generate | [→](./implementations/huggingface_peft_model_generate.md) | ✅Principle:Inference_Execution, ✅Env:huggingface_peft_Core_Environment | transformers (external) | Wrapper Doc - Text generation |
| huggingface_peft_merged_adapter_evaluation | [→](./implementations/huggingface_peft_merged_adapter_evaluation.md) | ✅Principle:Merge_Evaluation, ✅Env:huggingface_peft_Core_Environment | Pattern | Pattern Doc - Merged adapter evaluation |
| huggingface_peft_merge_strategy_selection | [→](./implementations/huggingface_peft_merge_strategy_selection.md) | ✅Principle:Merge_Strategy_Configuration, ✅Env:huggingface_peft_Core_Environment | merge_utils.py:L144-269 | API Doc - Merge algorithm config |
| huggingface_peft_LoraConfig_for_qlora | [→](./implementations/huggingface_peft_LoraConfig_for_qlora.md) | ✅Principle:QLoRA_Configuration, ✅Env:huggingface_peft_Core_Environment | config.py:L47-300 | API Doc - QLoRA-optimized config |
| huggingface_peft_Trainer_train_qlora | [→](./implementations/huggingface_peft_Trainer_train_qlora.md) | ✅Principle:QLoRA_Training_Execution, ✅Env:huggingface_peft_Quantization_Environment | transformers (external) | Wrapper Doc - QLoRA training |
| huggingface_peft_AutoModel_from_pretrained_quantized | [→](./implementations/huggingface_peft_AutoModel_from_pretrained_quantized.md) | ✅Principle:Quantized_Model_Loading, ✅Env:huggingface_peft_Quantization_Environment | transformers (external) | Wrapper Doc - Quantized model loading |
| huggingface_peft_AdaLoraConfig | [→](./implementations/huggingface_peft_AdaLoraConfig.md) | ✅Principle:AdaLoRA_Configuration, ✅Env:huggingface_peft_Core_Environment | adalora/config.py | API Doc - AdaLoRA configuration |
| huggingface_peft_AdaLoraGPTQ | [→](./implementations/huggingface_peft_AdaLoraGPTQ.md) | ✅Principle:AdaLoRA_Quantization, ✅Env:huggingface_peft_GPTQ_Environment | adalora/gptq.py | API Doc - AdaLoRA GPTQ layer |
| huggingface_peft_AdaLoraLayer | [→](./implementations/huggingface_peft_AdaLoraLayer.md) | ✅Principle:AdaLoRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | adalora/layer.py | API Doc - AdaLoRA layer |
| huggingface_peft_AdaLoraModel | [→](./implementations/huggingface_peft_AdaLoraModel.md) | ✅Principle:AdaLoRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | adalora/model.py | API Doc - AdaLoRA model |
| huggingface_peft_AdaLoraQuantized | [→](./implementations/huggingface_peft_AdaLoraQuantized.md) | ✅Principle:AdaLoRA_Quantization, ✅Env:huggingface_peft_Quantization_Environment | adalora/bnb.py | API Doc - AdaLoRA 8/4-bit layers |
| huggingface_peft_AdaptedAttention | [→](./implementations/huggingface_peft_AdaptedAttention.md) | ✅Principle:Adaption_Prompt, ✅Env:huggingface_peft_Core_Environment | adaption_prompt/layer.py | API Doc - Adapted attention layer |
| huggingface_peft_AdaptionPromptConfig | [→](./implementations/huggingface_peft_AdaptionPromptConfig.md) | ✅Principle:Adaption_Prompt, ✅Env:huggingface_peft_Core_Environment | adaption_prompt/config.py | API Doc - Adaption prompt config |
| huggingface_peft_AdaptionPromptModel | [→](./implementations/huggingface_peft_AdaptionPromptModel.md) | ✅Principle:Adaption_Prompt, ✅Env:huggingface_peft_Core_Environment | adaption_prompt/model.py | API Doc - Adaption prompt model |
| huggingface_peft_ArrowLoraLinearLayer | [→](./implementations/huggingface_peft_ArrowLoraLinearLayer.md) | ✅Principle:Arrow_LoRA, ✅Env:huggingface_peft_Core_Environment | lora/arrow.py | API Doc - Arrow LoRA layer |
| huggingface_peft_BOFTConfig | [→](./implementations/huggingface_peft_BOFTConfig.md) | ✅Principle:BOFT_Adaptation, ✅Env:huggingface_peft_Core_Environment | boft/config.py | API Doc - BOFT configuration |
| huggingface_peft_BOFTLayer | [→](./implementations/huggingface_peft_BOFTLayer.md) | ✅Principle:BOFT_Adaptation, ✅Env:huggingface_peft_Core_Environment | boft/layer.py | API Doc - BOFT layer |
| huggingface_peft_BOFTModel | [→](./implementations/huggingface_peft_BOFTModel.md) | ✅Principle:BOFT_Adaptation, ✅Env:huggingface_peft_Core_Environment | boft/model.py | API Doc - BOFT model |
| huggingface_peft_BoneConfig | [→](./implementations/huggingface_peft_BoneConfig.md) | ✅Principle:Bone_Adaptation, ✅Env:huggingface_peft_Core_Environment | bone/config.py | API Doc - Bone configuration |
| huggingface_peft_BoneLayer | [→](./implementations/huggingface_peft_BoneLayer.md) | ✅Principle:Bone_Adaptation, ✅Env:huggingface_peft_Core_Environment | bone/layer.py | API Doc - Bone layer |
| huggingface_peft_BoneModel | [→](./implementations/huggingface_peft_BoneModel.md) | ✅Principle:Bone_Adaptation, ✅Env:huggingface_peft_Core_Environment | bone/model.py | API Doc - Bone model |
| huggingface_peft_C3AConfig | [→](./implementations/huggingface_peft_C3AConfig.md) | ✅Principle:Circulant_Adaptation, ✅Env:huggingface_peft_Core_Environment | c3a/config.py | API Doc - C3A configuration |
| huggingface_peft_C3ALayer | [→](./implementations/huggingface_peft_C3ALayer.md) | ✅Principle:Circulant_Adaptation, ✅Env:huggingface_peft_Core_Environment | c3a/layer.py | API Doc - C3A FFT layer |
| huggingface_peft_C3AModel | [→](./implementations/huggingface_peft_C3AModel.md) | ✅Principle:Circulant_Adaptation, ✅Env:huggingface_peft_Core_Environment | c3a/model.py | API Doc - C3A model |
| huggingface_peft_CPTConfig | [→](./implementations/huggingface_peft_CPTConfig.md) | ✅Principle:Context_Prompt_Tuning, ✅Env:huggingface_peft_Core_Environment | cpt/config.py | API Doc - CPT configuration |
| huggingface_peft_CorDA | [→](./implementations/huggingface_peft_CorDA.md) | ✅Principle:CorDA_Decomposition, ✅Env:huggingface_peft_Core_Environment | lora/corda.py | API Doc - CorDA decomposition |
| huggingface_peft_FourierFTConfig | [→](./implementations/huggingface_peft_FourierFTConfig.md) | ✅Principle:Fourier_Adaptation, ✅Env:huggingface_peft_Core_Environment | fourierft/config.py | API Doc - FourierFT configuration |
| huggingface_peft_FourierFTLayer | [→](./implementations/huggingface_peft_FourierFTLayer.md) | ✅Principle:Fourier_Adaptation, ✅Env:huggingface_peft_Core_Environment | fourierft/layer.py | API Doc - FourierFT layer |
| huggingface_peft_FourierFTModel | [→](./implementations/huggingface_peft_FourierFTModel.md) | ✅Principle:Fourier_Adaptation, ✅Env:huggingface_peft_Core_Environment | fourierft/model.py | API Doc - FourierFT model |
| huggingface_peft_GraLoRAConfig | [→](./implementations/huggingface_peft_GraLoRAConfig.md) | ✅Principle:GraLoRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | gralora/config.py | API Doc - GraLoRA configuration |
| huggingface_peft_GraLoRALayer | [→](./implementations/huggingface_peft_GraLoRALayer.md) | ✅Principle:GraLoRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | gralora/layer.py | API Doc - GraLoRA layer |
| huggingface_peft_GraLoRAModel | [→](./implementations/huggingface_peft_GraLoRAModel.md) | ✅Principle:GraLoRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | gralora/model.py | API Doc - GraLoRA model |
| huggingface_peft_HRAConfig | [→](./implementations/huggingface_peft_HRAConfig.md) | ✅Principle:HRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | hra/config.py | API Doc - HRA configuration |
| huggingface_peft_HRALayer | [→](./implementations/huggingface_peft_HRALayer.md) | ✅Principle:HRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | hra/layer.py | API Doc - HRA layer |
| huggingface_peft_HRAModel | [→](./implementations/huggingface_peft_HRAModel.md) | ✅Principle:HRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | hra/model.py | API Doc - HRA model |
| huggingface_peft_IA3Config | [→](./implementations/huggingface_peft_IA3Config.md) | ✅Principle:IA3_Adaptation, ✅Env:huggingface_peft_Core_Environment | ia3/config.py | API Doc - IA3 configuration |
| huggingface_peft_IA3Layer | [→](./implementations/huggingface_peft_IA3Layer.md) | ✅Principle:IA3_Adaptation, ✅Env:huggingface_peft_Core_Environment | ia3/layer.py | API Doc - IA3 layer |
| huggingface_peft_IA3Model | [→](./implementations/huggingface_peft_IA3Model.md) | ✅Principle:IA3_Adaptation, ✅Env:huggingface_peft_Core_Environment | ia3/model.py | API Doc - IA3 model |
| huggingface_peft_IA3Quantized | [→](./implementations/huggingface_peft_IA3Quantized.md) | ✅Principle:IA3_Quantization, ✅Env:huggingface_peft_Quantization_Environment | ia3/bnb.py | API Doc - IA3 8/4-bit layers |
| huggingface_peft_IncrementalPCA | [→](./implementations/huggingface_peft_IncrementalPCA.md) | ✅Principle:Incremental_PCA, ✅Env:huggingface_peft_Core_Environment | utils/incremental_pca.py | API Doc - Incremental PCA utility |
| huggingface_peft_LNTuningConfig | [→](./implementations/huggingface_peft_LNTuningConfig.md) | ✅Principle:LN_Tuning, ✅Env:huggingface_peft_Core_Environment | ln_tuning/config.py | API Doc - LN Tuning configuration |
| huggingface_peft_LNTuningLayer | [→](./implementations/huggingface_peft_LNTuningLayer.md) | ✅Principle:LN_Tuning, ✅Env:huggingface_peft_Core_Environment | ln_tuning/layer.py | API Doc - LN Tuning layer |
| huggingface_peft_LNTuningModel | [→](./implementations/huggingface_peft_LNTuningModel.md) | ✅Principle:LN_Tuning, ✅Env:huggingface_peft_Core_Environment | ln_tuning/model.py | API Doc - LN Tuning model |
| huggingface_peft_LoHaConfig | [→](./implementations/huggingface_peft_LoHaConfig.md) | ✅Principle:LoHa_Adaptation, ✅Env:huggingface_peft_Core_Environment | loha/config.py | API Doc - LoHa configuration |
| huggingface_peft_LoHaLayer | [→](./implementations/huggingface_peft_LoHaLayer.md) | ✅Principle:LoHa_Adaptation, ✅Env:huggingface_peft_Core_Environment | loha/layer.py | API Doc - LoHa layer |
| huggingface_peft_LoHaModel | [→](./implementations/huggingface_peft_LoHaModel.md) | ✅Principle:LoHa_Adaptation, ✅Env:huggingface_peft_Core_Environment | loha/model.py | API Doc - LoHa model |
| huggingface_peft_LoKrConfig | [→](./implementations/huggingface_peft_LoKrConfig.md) | ✅Principle:LoKr_Adaptation, ✅Env:huggingface_peft_Core_Environment | lokr/config.py | API Doc - LoKr configuration |
| huggingface_peft_LoKrLayer | [→](./implementations/huggingface_peft_LoKrLayer.md) | ✅Principle:LoKr_Adaptation, ✅Env:huggingface_peft_Core_Environment | lokr/layer.py | API Doc - LoKr layer |
| huggingface_peft_LoKrModel | [→](./implementations/huggingface_peft_LoKrModel.md) | ✅Principle:LoKr_Adaptation, ✅Env:huggingface_peft_Core_Environment | lokr/model.py | API Doc - LoKr model |
| huggingface_peft_LoraAQLM | [→](./implementations/huggingface_peft_LoraAQLM.md) | ✅Principle:LoRA_Quantization, ✅Env:huggingface_peft_AQLM_Environment | lora/aqlm.py | API Doc - LoRA AQLM layer |
| huggingface_peft_LoraIntelFP8 | [→](./implementations/huggingface_peft_LoraIntelFP8.md) | ✅Principle:LoRA_Quantization, ✅Env:huggingface_peft_Intel_Environment | lora/inc.py | API Doc - LoRA Intel FP8 layer |
| huggingface_peft_LoraParallelLinear | [→](./implementations/huggingface_peft_LoraParallelLinear.md) | ✅Principle:Tensor_Parallel_LoRA, ✅Env:huggingface_peft_Megatron_Environment | lora/tp_layer.py | API Doc - LoRA tensor parallel layer |
| huggingface_peft_LoraTorchAO | [→](./implementations/huggingface_peft_LoraTorchAO.md) | ✅Principle:LoRA_Quantization, ✅Env:huggingface_peft_TorchAO_Environment | lora/torchao.py | API Doc - LoRA TorchAO layer |
| huggingface_peft_LyCORISUtils | [→](./implementations/huggingface_peft_LyCORISUtils.md) | ✅Principle:LyCORIS_Adaptation, ✅Env:huggingface_peft_Core_Environment | lycoris_utils.py | API Doc - LyCORIS base class |
| huggingface_peft_MissConfig | [→](./implementations/huggingface_peft_MissConfig.md) | ✅Principle:MiSS_Adaptation, ✅Env:huggingface_peft_Core_Environment | miss/config.py | API Doc - MiSS configuration |
| huggingface_peft_MissLayer | [→](./implementations/huggingface_peft_MissLayer.md) | ✅Principle:MiSS_Adaptation, ✅Env:huggingface_peft_Core_Environment | miss/layer.py | API Doc - MiSS layer |
| huggingface_peft_MissModel | [→](./implementations/huggingface_peft_MissModel.md) | ✅Principle:MiSS_Adaptation, ✅Env:huggingface_peft_Core_Environment | miss/model.py | API Doc - MiSS model |
| huggingface_peft_MultitaskPromptTuningConfig | [→](./implementations/huggingface_peft_MultitaskPromptTuningConfig.md) | ✅Principle:Multitask_Prompt_Tuning, ✅Env:huggingface_peft_Core_Environment | multitask_prompt_tuning/config.py | API Doc - MPT configuration |
| huggingface_peft_MultitaskPromptTuningModel | [→](./implementations/huggingface_peft_MultitaskPromptTuningModel.md) | ✅Principle:Multitask_Prompt_Tuning, ✅Env:huggingface_peft_Core_Environment | multitask_prompt_tuning/model.py | API Doc - MPT model |
| huggingface_peft_OFTConfig | [→](./implementations/huggingface_peft_OFTConfig.md) | ✅Principle:OFT_Adaptation, ✅Env:huggingface_peft_Core_Environment | oft/config.py | API Doc - OFT configuration |
| huggingface_peft_OFTLayer | [→](./implementations/huggingface_peft_OFTLayer.md) | ✅Principle:OFT_Adaptation, ✅Env:huggingface_peft_Core_Environment | oft/layer.py | API Doc - OFT layer |
| huggingface_peft_OFTModel | [→](./implementations/huggingface_peft_OFTModel.md) | ✅Principle:OFT_Adaptation, ✅Env:huggingface_peft_Core_Environment | oft/model.py | API Doc - OFT model |
| huggingface_peft_OFTQuantized | [→](./implementations/huggingface_peft_OFTQuantized.md) | ✅Principle:OFT_Quantization, ✅Env:huggingface_peft_Quantization_Environment | oft/bnb.py | API Doc - OFT 8/4-bit layers |
| huggingface_peft_OFT_AQLM | [→](./implementations/huggingface_peft_OFT_AQLM.md) | ✅Principle:OFT_Quantization, ✅Env:huggingface_peft_AQLM_Environment | oft/aqlm.py | API Doc - OFT AQLM layer |
| huggingface_peft_OFT_AWQ | [→](./implementations/huggingface_peft_OFT_AWQ.md) | ✅Principle:OFT_Quantization, ✅Env:huggingface_peft_AWQ_Environment | oft/awq.py | API Doc - OFT AWQ layer |
| huggingface_peft_OFT_EETQ | [→](./implementations/huggingface_peft_OFT_EETQ.md) | ✅Principle:OFT_Quantization, ✅Env:huggingface_peft_EETQ_Environment | oft/eetq.py | API Doc - OFT EETQ layer |
| huggingface_peft_OFT_GPTQ | [→](./implementations/huggingface_peft_OFT_GPTQ.md) | ✅Principle:OFT_Quantization, ✅Env:huggingface_peft_GPTQ_Environment | oft/gptq.py | API Doc - OFT GPTQ layer |
| huggingface_peft_OFT_HQQ | [→](./implementations/huggingface_peft_OFT_HQQ.md) | ✅Principle:OFT_Quantization, ✅Env:huggingface_peft_HQQ_Environment | oft/hqq.py | API Doc - OFT HQQ layer |
| huggingface_peft_OFT_IntelFP8 | [→](./implementations/huggingface_peft_OFT_IntelFP8.md) | ✅Principle:OFT_Quantization, ✅Env:huggingface_peft_Intel_Environment | oft/inc.py | API Doc - OFT Intel FP8 layer |
| huggingface_peft_PolyConfig | [→](./implementations/huggingface_peft_PolyConfig.md) | ✅Principle:Poly_Adaptation, ✅Env:huggingface_peft_Core_Environment | poly/config.py | API Doc - Poly configuration |
| huggingface_peft_PolyLayer | [→](./implementations/huggingface_peft_PolyLayer.md) | ✅Principle:Poly_Adaptation, ✅Env:huggingface_peft_Core_Environment | poly/layer.py | API Doc - Poly layer |
| huggingface_peft_PolyModel | [→](./implementations/huggingface_peft_PolyModel.md) | ✅Principle:Poly_Adaptation, ✅Env:huggingface_peft_Core_Environment | poly/model.py | API Doc - Poly model |
| huggingface_peft_PolyRouter | [→](./implementations/huggingface_peft_PolyRouter.md) | ✅Principle:Poly_Adaptation, ✅Env:huggingface_peft_Core_Environment | poly/router.py | API Doc - Poly router |
| huggingface_peft_PrefixEncoder | [→](./implementations/huggingface_peft_PrefixEncoder.md) | ✅Principle:Prefix_Tuning, ✅Env:huggingface_peft_Core_Environment | prefix_tuning/model.py | API Doc - Prefix encoder model |
| huggingface_peft_PrefixTuningConfig | [→](./implementations/huggingface_peft_PrefixTuningConfig.md) | ✅Principle:Prefix_Tuning, ✅Env:huggingface_peft_Core_Environment | prefix_tuning/config.py | API Doc - Prefix tuning config |
| huggingface_peft_PromptEmbedding | [→](./implementations/huggingface_peft_PromptEmbedding.md) | ✅Principle:Prompt_Tuning, ✅Env:huggingface_peft_Core_Environment | prompt_tuning/model.py | API Doc - Prompt embedding model |
| huggingface_peft_PromptEncoder | [→](./implementations/huggingface_peft_PromptEncoder.md) | ✅Principle:P_Tuning, ✅Env:huggingface_peft_Core_Environment | p_tuning/model.py | API Doc - P-Tuning encoder |
| huggingface_peft_PromptEncoderConfig | [→](./implementations/huggingface_peft_PromptEncoderConfig.md) | ✅Principle:P_Tuning, ✅Env:huggingface_peft_Core_Environment | p_tuning/config.py | API Doc - P-Tuning config |
| huggingface_peft_PromptTuningConfig | [→](./implementations/huggingface_peft_PromptTuningConfig.md) | ✅Principle:Prompt_Tuning, ✅Env:huggingface_peft_Core_Environment | prompt_tuning/config.py | API Doc - Prompt tuning config |
| huggingface_peft_RandLoraConfig | [→](./implementations/huggingface_peft_RandLoraConfig.md) | ✅Principle:RandLoRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | randlora/config.py | API Doc - RandLoRA configuration |
| huggingface_peft_RandLoraLayer | [→](./implementations/huggingface_peft_RandLoraLayer.md) | ✅Principle:RandLoRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | randlora/layer.py | API Doc - RandLoRA layer |
| huggingface_peft_RandLoraModel | [→](./implementations/huggingface_peft_RandLoraModel.md) | ✅Principle:RandLoRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | randlora/model.py | API Doc - RandLoRA model |
| huggingface_peft_RandLoraQuantized | [→](./implementations/huggingface_peft_RandLoraQuantized.md) | ✅Principle:RandLoRA_Quantization, ✅Env:huggingface_peft_Quantization_Environment | randlora/bnb.py | API Doc - RandLoRA 8/4-bit layers |
| huggingface_peft_RoadConfig | [→](./implementations/huggingface_peft_RoadConfig.md) | ✅Principle:RoAd_Adaptation, ✅Env:huggingface_peft_Core_Environment | road/config.py | API Doc - RoAd configuration |
| huggingface_peft_RoadLayer | [→](./implementations/huggingface_peft_RoadLayer.md) | ✅Principle:RoAd_Adaptation, ✅Env:huggingface_peft_Core_Environment | road/layer.py | API Doc - RoAd layer |
| huggingface_peft_RoadModel | [→](./implementations/huggingface_peft_RoadModel.md) | ✅Principle:RoAd_Adaptation, ✅Env:huggingface_peft_Core_Environment | road/model.py | API Doc - RoAd model |
| huggingface_peft_RoadQuantized | [→](./implementations/huggingface_peft_RoadQuantized.md) | ✅Principle:RoAd_Quantization, ✅Env:huggingface_peft_Quantization_Environment | road/bnb.py | API Doc - RoAd 8/4-bit layers |
| huggingface_peft_ShiraConfig | [→](./implementations/huggingface_peft_ShiraConfig.md) | ✅Principle:SHiRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | shira/config.py | API Doc - SHiRA configuration |
| huggingface_peft_ShiraLayer | [→](./implementations/huggingface_peft_ShiraLayer.md) | ✅Principle:SHiRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | shira/layer.py | API Doc - SHiRA layer |
| huggingface_peft_ShiraModel | [→](./implementations/huggingface_peft_ShiraModel.md) | ✅Principle:SHiRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | shira/model.py | API Doc - SHiRA model |
| huggingface_peft_TrainableTokensConfig | [→](./implementations/huggingface_peft_TrainableTokensConfig.md) | ✅Principle:Trainable_Tokens, ✅Env:huggingface_peft_Core_Environment | trainable_tokens/config.py | API Doc - Trainable Tokens config |
| huggingface_peft_TrainableTokensLayer | [→](./implementations/huggingface_peft_TrainableTokensLayer.md) | ✅Principle:Trainable_Tokens, ✅Env:huggingface_peft_Core_Environment | trainable_tokens/layer.py | API Doc - Trainable Tokens layer |
| huggingface_peft_TrainableTokensModel | [→](./implementations/huggingface_peft_TrainableTokensModel.md) | ✅Principle:Trainable_Tokens, ✅Env:huggingface_peft_Core_Environment | trainable_tokens/model.py | API Doc - Trainable Tokens model |
| huggingface_peft_VBLoRAConfig | [→](./implementations/huggingface_peft_VBLoRAConfig.md) | ✅Principle:VBLoRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | vblora/config.py | API Doc - VBLoRA configuration |
| huggingface_peft_VBLoRALayer | [→](./implementations/huggingface_peft_VBLoRALayer.md) | ✅Principle:VBLoRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | vblora/layer.py | API Doc - VBLoRA layer |
| huggingface_peft_VBLoRAModel | [→](./implementations/huggingface_peft_VBLoRAModel.md) | ✅Principle:VBLoRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | vblora/model.py | API Doc - VBLoRA model |
| huggingface_peft_VeraConfig | [→](./implementations/huggingface_peft_VeraConfig.md) | ✅Principle:VeRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | vera/config.py | API Doc - VeRA configuration |
| huggingface_peft_VeraLayer | [→](./implementations/huggingface_peft_VeraLayer.md) | ✅Principle:VeRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | vera/layer.py | API Doc - VeRA layer |
| huggingface_peft_VeraModel | [→](./implementations/huggingface_peft_VeraModel.md) | ✅Principle:VeRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | vera/model.py | API Doc - VeRA model |
| huggingface_peft_VeraQuantized | [→](./implementations/huggingface_peft_VeraQuantized.md) | ✅Principle:VeRA_Quantization, ✅Env:huggingface_peft_Quantization_Environment | vera/bnb.py | API Doc - VeRA 8/4-bit layers |
| huggingface_peft_XLoraClassifier | [→](./implementations/huggingface_peft_XLoraClassifier.md) | ✅Principle:XLoRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | xlora/classifier.py | API Doc - X-LoRA classifier |
| huggingface_peft_XLoraConfig | [→](./implementations/huggingface_peft_XLoraConfig.md) | ✅Principle:XLoRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | xlora/config.py | API Doc - X-LoRA configuration |
| huggingface_peft_XLoraLayer | [→](./implementations/huggingface_peft_XLoraLayer.md) | ✅Principle:XLoRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | xlora/layer.py | API Doc - X-LoRA layer |
| huggingface_peft_XLoraModel | [→](./implementations/huggingface_peft_XLoraModel.md) | ✅Principle:XLoRA_Adaptation, ✅Env:huggingface_peft_Core_Environment | xlora/model.py | API Doc - X-LoRA model |

---

## Summary

- **Total Implementation Pages:** 121
- **API Docs:** 112 (PEFT native APIs)
- **Wrapper Docs:** 8 (external library integrations)
- **Pattern Docs:** 1
- **External Tool Docs:** 0

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
