# Principle Index

| Page | Summary | Link |
|------|---------|------|
| AutoGPTQ_AutoGPTQ_GPTQ_Post_Training_Quantization | Post-training method that rounds weights layer by layer to a low bit width, usin | [→](./principles/AutoGPTQ_AutoGPTQ_GPTQ_Post_Training_Quantization.md) |
| NVIDIA_Megatron-LM_Tensor_Parallelism | Technique that divides the weight matrices of individual layers among devices, s | [→](./principles/NVIDIA_Megatron-LM_Tensor_Parallelism.md) |
| casper-hansen_AutoAWQ_Activation_Aware_Weight_Quantization | Weight-only compression that finds a small set of salient channels from activati | [→](./principles/casper-hansen_AutoAWQ_Activation_Aware_Weight_Quantization.md) |
| huggingface_peft_DoRA_Weight_Decomposed_Adaptation | Extension of low-rank adaptation that splits each pretrained weight into a magni | [→](./principles/huggingface_peft_DoRA_Weight_Decomposed_Adaptation.md) |
| huggingface_peft_LoRA_Low_Rank_Adaptation | Parameter-efficient method that freezes pretrained weights and learns a low-rank | [→](./principles/huggingface_peft_LoRA_Low_Rank_Adaptation.md) |
| huggingface_peft_Prompt_Tuning | Adaptation method that keeps every model weight frozen and learns a small matrix | [→](./principles/huggingface_peft_Prompt_Tuning.md) |
| huggingface_peft_QLoRA_Quantized_Low_Rank_Adaptation | Adapter training on top of a base model stored in 4-bit NormalFloat, with double | [→](./principles/huggingface_peft_QLoRA_Quantized_Low_Rank_Adaptation.md) |
| huggingface_transformers_KV_Cache | Stored key and value tensors of the tokens already processed, reused at every de | [→](./principles/huggingface_transformers_KV_Cache.md) |
| huggingface_transformers_Knowledge_Distillation | Training a compact student model to match the softened output distribution of a  | [→](./principles/huggingface_transformers_Knowledge_Distillation.md) |
| pytorch_pytorch_Activation_Offloading | Technique that copies tensors kept for backpropagation to pinned CPU memory duri | [→](./principles/pytorch_pytorch_Activation_Offloading.md) |
| pytorch_pytorch_Data_Parallelism | Scheme that keeps a complete copy of the model on every device, feeds each devic | [→](./principles/pytorch_pytorch_Data_Parallelism.md) |
| pytorch_pytorch_Fully_Sharded_Data_Parallel | Variant of data parallelism that partitions parameters, gradients and optimizer  | [→](./principles/pytorch_pytorch_Fully_Sharded_Data_Parallel.md) |
| pytorch_pytorch_Gradient_Checkpointing | Training technique that discards selected intermediate tensors after the forward | [→](./principles/pytorch_pytorch_Gradient_Checkpointing.md) |
| pytorch_pytorch_Magnitude_Pruning | Compression technique that removes the connections judged least important by wei | [→](./principles/pytorch_pytorch_Magnitude_Pruning.md) |
| pytorch_pytorch_Mixed_Precision_Training | Training scheme that performs most arithmetic in 16-bit floating point while hol | [→](./principles/pytorch_pytorch_Mixed_Precision_Training.md) |
| pytorch_pytorch_Pipeline_Parallelism | Technique that assigns consecutive groups of layers to different devices and str | [→](./principles/pytorch_pytorch_Pipeline_Parallelism.md) |
| vllm-project_vllm_Continuous_Batching | Scheduling policy that admits new requests and retires finished ones at every de | [→](./principles/vllm-project_vllm_Continuous_Batching.md) |
| vllm-project_vllm_Paged_Attention | Attention kernel and memory manager that keeps the key-value cache in fixed-size | [→](./principles/vllm-project_vllm_Paged_Attention.md) |
| vllm-project_vllm_Speculative_Decoding | Generation method in which a cheap draft model proposes several tokens that the  | [→](./principles/vllm-project_vllm_Speculative_Decoding.md) |
