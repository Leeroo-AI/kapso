{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Distributed Training|https://huggingface.co/docs/transformers/parallelism]]
* [[source::Doc|FSDP Guide|https://huggingface.co/docs/transformers/fsdp]]
|-
! Domains
| [[domain::Distributed_Training]], [[domain::Tensor_Parallelism]], [[domain::Data_Parallelism]], [[domain::Large_Scale_Training]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
Advanced distributed training workflow combining Tensor Parallelism (TP), Data Parallelism (DP), and Context Parallelism (CP) for efficient training of large language models across multiple GPUs.

=== Description ===
3D Parallelism enables training of models that exceed single-GPU memory by combining multiple parallelism strategies:

* **Tensor Parallelism (TP)**: Splits individual layers across GPUs, reducing per-GPU memory by partitioning weight matrices
* **Data Parallelism (DP)**: Replicates the model across GPUs, each processing different data batches with gradient synchronization
* **Context Parallelism (CP)**: Splits long sequences across GPUs, enabling training with extended context lengths

This workflow demonstrates the full setup for training with PyTorch's native distributed primitives (DeviceMesh, FSDP, DTensor) combined with HuggingFace Transformers' model loading with tensor parallel plans.

=== Usage ===
Execute this workflow when you need to:
* Train models too large for single-GPU memory
* Scale training across multiple GPUs or nodes
* Process sequences longer than single-GPU memory allows
* Maximize training throughput with large batch sizes
* Leverage the latest PyTorch distributed features

Prerequisites:
* Multi-GPU environment (CUDA GPUs recommended)
* PyTorch 2.1+ with distributed support
* NCCL backend for communication
* Model with tensor parallel support (tp_plan="auto")

== Execution Steps ==

=== Step 1: Distributed Environment Initialization ===
[[step::Principle:huggingface_transformers_Distributed_Init]]

Initialize the distributed process group and configure the device mesh. The mesh defines how GPUs are organized for different parallelism dimensions.

'''Initialization components:'''
* NCCL process group initialization
* Local rank and world size configuration
* DeviceMesh creation with (dp, tp, cp) dimensions
* CUDA device assignment per rank

=== Step 2: Tokenizer and Model Loading ===
[[step::Principle:huggingface_transformers_TP_Model_Loading]]

Load the model with tensor parallelism enabled. The model is loaded with automatic tensor parallel plan that shards attention and MLP layers across the TP mesh dimension.

'''TP loading features:'''
* device_mesh parameter for TP distribution
* tp_plan="auto" for automatic sharding
* dtype specification (bf16 recommended)
* Automatic weight distribution

=== Step 3: Data Parallelism Setup ===
[[step::Principle:huggingface_transformers_Data_Parallelism_Setup]]

Wrap the model with FSDP or DDP for data parallelism. Configure sharding strategy and synchronization across the DP mesh dimension.

'''DP configuration options:'''
* FSDP with NO_SHARD for gradient sync only
* Full sharding for memory optimization
* DistributedSampler for data distribution
* Gradient synchronization strategy

=== Step 4: Dataset Preparation and DataLoader ===
[[step::Principle:huggingface_transformers_Distributed_Dataset]]

Prepare the dataset for distributed training. Create packed sequences, configure the DistributedSampler for DP sharding, and set up the DataLoader.

'''Dataset setup:'''
* Tokenize and pack sequences
* Configure local batch size (global_batch / dp_size)
* DistributedSampler with DP rank
* Efficient collation for batching

=== Step 5: Context Parallelism Execution ===
[[step::Principle:huggingface_transformers_Context_Parallelism]]

Execute the forward pass with context parallelism for sequence dimension sharding. Use the context_parallel context manager to automatically shard input tensors along the sequence dimension.

'''CP execution flow:'''
* Buffer registration for input tensors
* Automatic sequence dimension sharding
* Ring attention communication pattern
* Loss computation with sharded labels

=== Step 6: Gradient Synchronization ===
[[step::Principle:huggingface_transformers_Gradient_Synchronization]]

Synchronize gradients across all parallelism dimensions. Handle the complexity of DTensor gradients and cross-mesh communication.

'''Gradient sync operations:'''
* All-reduce across DP dimension
* All-reduce across CP dimension
* Handle DTensor local gradient extraction
* Gradient clipping across distributed tensors

=== Step 7: Optimizer Step and Logging ===
[[step::Principle:huggingface_transformers_Distributed_Optimizer_Step]]

Execute the optimizer step and log metrics. All-reduce loss for consistent reporting across all ranks.

'''Optimizer and logging:'''
* Gradient clipping before optimizer step
* AdamW with weight decay
* Loss all-reduce for logging
* WandB integration (rank 0 only)

=== Step 8: Distributed Checkpointing ===
[[step::Principle:huggingface_transformers_Distributed_Checkpointing]]

Save checkpoints using PyTorch's distributed checkpoint API (DCP). Handle the complexity of sharded state dicts across TP and DP dimensions.

'''DCP checkpointing:'''
* Unified checkpoint format across ranks
* Automatic resharding on load
* Optimizer state checkpointing
* Fault-tolerant saving

== Execution Diagram ==
{{#mermaid:graph TD
    A[Distributed Environment Init] --> B[TP Model Loading]
    B --> C[Data Parallelism Setup]
    C --> D[Dataset & DataLoader]
    D --> E[Context Parallelism Execution]
    E --> F[Gradient Synchronization]
    F --> G[Optimizer Step & Logging]
    G --> H[Distributed Checkpointing]
    G -->|Next Step| E
}}

== Related Pages ==
* [[step::Principle:huggingface_transformers_Distributed_Init]]
* [[step::Principle:huggingface_transformers_TP_Model_Loading]]
* [[step::Principle:huggingface_transformers_Data_Parallelism_Setup]]
* [[step::Principle:huggingface_transformers_Distributed_Dataset]]
* [[step::Principle:huggingface_transformers_Context_Parallelism]]
* [[step::Principle:huggingface_transformers_Gradient_Synchronization]]
* [[step::Principle:huggingface_transformers_Distributed_Optimizer_Step]]
* [[step::Principle:huggingface_transformers_Distributed_Checkpointing]]
