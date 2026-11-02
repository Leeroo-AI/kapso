# Heuristic Index

| Page | Summary | Link |
|------|---------|------|
| AutoGPTQ_AutoGPTQ_Calibration_Set_Size | Guidance on how many calibration sequences and tokens a post-training compressio | [→](./heuristics/AutoGPTQ_AutoGPTQ_Calibration_Set_Size.md) |
| huggingface_peft_LoRA_Rank_And_Alpha_Selection | Guidance for choosing the rank of low-rank adapters and their scaling factor, wi | [→](./heuristics/huggingface_peft_LoRA_Rank_And_Alpha_Selection.md) |
| pytorch_pytorch_Checkpoint_Every_Transformer_Block | Apply recomputation at the granularity of whole transformer blocks, which keeps  | [→](./heuristics/pytorch_pytorch_Checkpoint_Every_Transformer_Block.md) |
| pytorch_pytorch_Gradient_Accumulation_For_Small_Batches | Sum gradients over several micro-batches before each optimizer step so the effec | [→](./heuristics/pytorch_pytorch_Gradient_Accumulation_For_Small_Batches.md) |
| vllm-project_vllm_GPU_Memory_Utilization_Setting | Guidance for choosing the gpu_memory_utilization fraction, which caps the share  | [→](./heuristics/vllm-project_vllm_GPU_Memory_Utilization_Setting.md) |
