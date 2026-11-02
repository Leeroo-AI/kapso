# Implementation Index

| Page | Summary | Link |
|------|---------|------|
| AutoGPTQ_AutoGPTQ_Quantize | AutoGPTQForCausalLM is the main class of the AutoGPTQ library, and its quantize  | [→](./implementations/AutoGPTQ_AutoGPTQ_Quantize.md) |
| NVIDIA_Megatron-LM_ColumnParallelLinear | Linear layer in Megatron-LM whose weight matrix is divided along its output dime | [→](./implementations/NVIDIA_Megatron-LM_ColumnParallelLinear.md) |
| casper-hansen_AutoAWQ_Quantize | AutoAWQForCausalLM is the entry class of the AutoAWQ library, and its quantize m | [→](./implementations/casper-hansen_AutoAWQ_Quantize.md) |
| huggingface_peft_LoraConfig | LoraConfig is the configuration dataclass in the PEFT library that holds every s | [→](./implementations/huggingface_peft_LoraConfig.md) |
| huggingface_peft_PromptTuningConfig | Configuration dataclass for prompt tuning in PEFT; num_virtual_tokens sets the l | [→](./implementations/huggingface_peft_PromptTuningConfig.md) |
| huggingface_transformers_BitsAndBytes_4bit_Config | BitsAndBytesConfig in transformers selects 4-bit loading through the bitsandbyte | [→](./implementations/huggingface_transformers_BitsAndBytes_4bit_Config.md) |
| huggingface_transformers_Distillation_Loss_KL | Student loss for knowledge distillation computed with torch.nn.functional.kl_div | [→](./implementations/huggingface_transformers_Distillation_Loss_KL.md) |
| huggingface_transformers_Generate_Use_Cache | The use_cache argument of generate in transformers controls whether past attenti | [→](./implementations/huggingface_transformers_Generate_Use_Cache.md) |
| pytorch_pytorch_Autocast_GradScaler | The torch.autocast context manager and the torch.amp.GradScaler class are the tw | [→](./implementations/pytorch_pytorch_Autocast_GradScaler.md) |
| pytorch_pytorch_Checkpoint_Sequential | Function in torch.utils.checkpoint that runs a sequence of modules in segments a | [→](./implementations/pytorch_pytorch_Checkpoint_Sequential.md) |
| pytorch_pytorch_DDP_Wrapper | DistributedDataParallel wraps a module so that gradients are averaged across pro | [→](./implementations/pytorch_pytorch_DDP_Wrapper.md) |
| pytorch_pytorch_FSDP_Wrapper | FullyShardedDataParallel is the PyTorch wrapper class that distributes a module' | [→](./implementations/pytorch_pytorch_FSDP_Wrapper.md) |
| pytorch_pytorch_L1_Unstructured_Prune | Function torch.nn.utils.prune.l1_unstructured, which zeroes a given fraction of  | [→](./implementations/pytorch_pytorch_L1_Unstructured_Prune.md) |
| pytorch_pytorch_Pipeline_Schedule | Schedule classes in torch.distributed.pipelining, such as ScheduleGPipe and Sche | [→](./implementations/pytorch_pytorch_Pipeline_Schedule.md) |
| pytorch_pytorch_Save_On_CPU_Hook | Context manager in torch.autograd.graph that parks tensors saved for backward in | [→](./implementations/pytorch_pytorch_Save_On_CPU_Hook.md) |
| vllm-project_vllm_LLMEngine | LLMEngine is the core class of vLLM: it accepts generation requests, schedules t | [→](./implementations/vllm-project_vllm_LLMEngine.md) |
| vllm-project_vllm_SpeculativeConfig | Configuration object in vLLM that names the draft model or method used for specu | [→](./implementations/vllm-project_vllm_SpeculativeConfig.md) |
