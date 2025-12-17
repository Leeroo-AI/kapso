# Implementation Index: vllm-project_vllm

> Tracks Implementation pages and their connections to Principles, Environments, etc.
> **Update IMMEDIATELY** after creating or modifying an Implementation page.

## Summary

| Workflow | Implementations | Type Distribution |
|----------|-----------------|-------------------|
| Basic_Offline_LLM_Inference | 6 | 6 API Doc |
| Online_API_Serving | 6 | 2 API Doc, 2 Wrapper Doc, 2 Pattern Doc |
| Vision_Language_Multimodal_Inference | 6 | 5 API Doc, 1 Pattern Doc |
| LoRA_Adapter_Inference | 6 | 5 API Doc, 1 Wrapper Doc |
| Speculative_Decoding | 6 | 5 API Doc, 1 Pattern Doc |
| Distributed_Data_Parallel_Inference | 6 | 3 API Doc, 3 Pattern Doc |
| Orphan (Standalone) | 87 | 87 API Doc |
| **Total** | **123** | **113 API, 3 Wrapper, 7 Pattern** |

---

## Pages

### Basic_Offline_LLM_Inference Implementations

| Page | File | Connections | Source | Type |
|------|------|-------------|--------|------|
| vllm-project_vllm_EngineArgs | [→](./implementations/vllm-project_vllm_EngineArgs.md) | ✅Principle:Engine_Configuration, ✅Env:vllm-project_vllm_CUDA_Environment, ✅Heuristic:vllm-project_vllm_GPU_Memory_Utilization, ✅Heuristic:vllm-project_vllm_Tensor_Parallelism, ✅Heuristic:vllm-project_vllm_Max_Model_Length, ✅Heuristic:vllm-project_vllm_Enforce_Eager_Mode | engine/arg_utils.py | API Doc |
| vllm-project_vllm_SamplingParams | [→](./implementations/vllm-project_vllm_SamplingParams.md) | ✅Principle:Sampling_Configuration, ✅Env:vllm-project_vllm_CUDA_Environment, ✅Heuristic:vllm-project_vllm_Temperature_Sampling | sampling_params.py | API Doc |
| vllm-project_vllm_LLM_init | [→](./implementations/vllm-project_vllm_LLM_init.md) | ✅Principle:Model_Loading, ✅Env:vllm-project_vllm_CUDA_Environment, ✅Heuristic:vllm-project_vllm_GPU_Memory_Utilization, ✅Heuristic:vllm-project_vllm_Max_Model_Length | entrypoints/llm.py | API Doc |
| vllm-project_vllm_PromptType | [→](./implementations/vllm-project_vllm_PromptType.md) | ✅Principle:Input_Formatting | inputs/data.py | API Doc |
| vllm-project_vllm_LLM_generate | [→](./implementations/vllm-project_vllm_LLM_generate.md) | ✅Principle:Batch_Generation, ✅Env:vllm-project_vllm_CUDA_Environment | entrypoints/llm.py | API Doc |
| vllm-project_vllm_RequestOutput | [→](./implementations/vllm-project_vllm_RequestOutput.md) | ✅Principle:Output_Processing | outputs.py | API Doc |

### Online_API_Serving Implementations

| Page | File | Connections | Source | Type |
|------|------|-------------|--------|------|
| vllm-project_vllm_vllm_serve | [→](./implementations/vllm-project_vllm_vllm_serve.md) | ✅Principle:Server_Configuration, ✅Env:vllm-project_vllm_CUDA_Environment, ✅Heuristic:vllm-project_vllm_Tensor_Parallelism, ✅Heuristic:vllm-project_vllm_Enforce_Eager_Mode | entrypoints/openai/ | API Doc |
| vllm-project_vllm_vllm_serve_startup | [→](./implementations/vllm-project_vllm_vllm_serve_startup.md) | ✅Principle:Server_Startup | entrypoints/openai/ | API Doc |
| vllm-project_vllm_OpenAI_Client | [→](./implementations/vllm-project_vllm_OpenAI_Client.md) | ✅Principle:API_Client_Setup | External (openai) | Wrapper Doc |
| vllm-project_vllm_chat_message_format | [→](./implementations/vllm-project_vllm_chat_message_format.md) | ✅Principle:Chat_Formatting | User code | Pattern Doc |
| vllm-project_vllm_chat_completions_create | [→](./implementations/vllm-project_vllm_chat_completions_create.md) | ✅Principle:API_Request_Processing, ✅Heuristic:vllm-project_vllm_Temperature_Sampling | External (openai) | Wrapper Doc |
| vllm-project_vllm_sse_streaming | [→](./implementations/vllm-project_vllm_sse_streaming.md) | ✅Principle:Streaming_Response | User code | Pattern Doc |

### Vision_Language_Multimodal_Inference Implementations

| Page | File | Connections | Source | Type |
|------|------|-------------|--------|------|
| vllm-project_vllm_EngineArgs_Multimodal_API | [→](./implementations/vllm-project_vllm_EngineArgs_Multimodal_API.md) | ✅Principle:VLM_Configuration, ✅Env:vllm-project_vllm_CUDA_Environment | engine/arg_utils.py | API Doc |
| vllm-project_vllm_Image_Loading_API | [→](./implementations/vllm-project_vllm_Image_Loading_API.md) | ✅Principle:Multimodal_Input_Preparation | multimodal/utils.py | API Doc |
| vllm-project_vllm_VLM_Prompt_Templates_Pattern | [→](./implementations/vllm-project_vllm_VLM_Prompt_Templates_Pattern.md) | ✅Principle:Multimodal_Prompt_Formatting | examples/ | Pattern Doc |
| vllm-project_vllm_LLM_Multimodal_Initialization_API | [→](./implementations/vllm-project_vllm_LLM_Multimodal_Initialization_API.md) | ✅Principle:VLM_Engine_Initialization, ✅Env:vllm-project_vllm_CUDA_Environment | entrypoints/llm.py | API Doc |
| vllm-project_vllm_LLM_Generate_Multimodal_API | [→](./implementations/vllm-project_vllm_LLM_Generate_Multimodal_API.md) | ✅Principle:Multimodal_Generation | entrypoints/llm.py | API Doc |
| vllm-project_vllm_RequestOutput_VLM_API | [→](./implementations/vllm-project_vllm_RequestOutput_VLM_API.md) | ✅Principle:VLM_Output_Processing | outputs.py | API Doc |

### LoRA_Adapter_Inference Implementations

| Page | File | Connections | Source | Type |
|------|------|-------------|--------|------|
| vllm-project_vllm_EngineArgs_lora | [→](./implementations/vllm-project_vllm_EngineArgs_lora.md) | ✅Principle:LoRA_Engine_Configuration, ✅Env:vllm-project_vllm_CUDA_Environment | engine/arg_utils.py | API Doc |
| vllm-project_vllm_LLMEngine_from_engine_args | [→](./implementations/vllm-project_vllm_LLMEngine_from_engine_args.md) | ✅Principle:LoRA_Base_Model_Loading | engine/llm_engine.py | API Doc |
| vllm-project_vllm_snapshot_download_lora | [→](./implementations/vllm-project_vllm_snapshot_download_lora.md) | ✅Principle:LoRA_Adapter_Loading | External (huggingface_hub) | Wrapper Doc |
| vllm-project_vllm_LoRARequest | [→](./implementations/vllm-project_vllm_LoRARequest.md) | ✅Principle:LoRA_Request_Creation | lora/request.py | API Doc |
| vllm-project_vllm_LLMEngine_add_request | [→](./implementations/vllm-project_vllm_LLMEngine_add_request.md) | ✅Principle:MultiLoRA_Inference | engine/llm_engine.py | API Doc |
| vllm-project_vllm_RequestOutput_lora | [→](./implementations/vllm-project_vllm_RequestOutput_lora.md) | ✅Principle:LoRA_Output_Processing | outputs.py | API Doc |

### Speculative_Decoding Implementations

| Page | File | Connections | Source | Type |
|------|------|-------------|--------|------|
| vllm-project_vllm_SpeculativeConfig | [→](./implementations/vllm-project_vllm_SpeculativeConfig.md) | ✅Principle:vllm-project_vllm_spec_method_selection | config/speculative.py | API Doc |
| vllm-project_vllm_LLM_speculative | [→](./implementations/vllm-project_vllm_LLM_speculative.md) | ✅Principle:vllm-project_vllm_speculative_engine_init, ✅Env:vllm-project_vllm_CUDA_Environment | entrypoints/llm.py | API Doc |
| vllm-project_vllm_TokensPrompt_spec | [→](./implementations/vllm-project_vllm_TokensPrompt_spec.md) | ✅Principle:vllm-project_vllm_speculative_prompt_prep | inputs.py | API Doc |
| vllm-project_vllm_LLM_generate_spec | [→](./implementations/vllm-project_vllm_LLM_generate_spec.md) | ✅Principle:vllm-project_vllm_speculative_generation | entrypoints/llm.py | API Doc |
| vllm-project_vllm_get_metrics | [→](./implementations/vllm-project_vllm_get_metrics.md) | ✅Principle:vllm-project_vllm_speculative_metrics | entrypoints/llm.py | API Doc |

### Distributed_Data_Parallel_Inference Implementations

| Page | File | Connections | Source | Type |
|------|------|-------------|--------|------|
| vllm-project_vllm_ParallelConfig | [→](./implementations/vllm-project_vllm_ParallelConfig.md) | ✅Principle:vllm-project_vllm_strategy_planning, ✅Heuristic:vllm-project_vllm_Tensor_Parallelism | config/parallel.py | API Doc |
| vllm-project_vllm_process_launcher | [→](./implementations/vllm-project_vllm_process_launcher.md) | ✅Principle:vllm-project_vllm_dp_env_vars | User code | Pattern Doc |
| vllm-project_vllm_LLM_class | [→](./implementations/vllm-project_vllm_LLM_class.md) | ✅Principle:vllm-project_vllm_LLM_distributed, ✅Env:vllm-project_vllm_CUDA_Environment | entrypoints/llm.py | API Doc |
| vllm-project_vllm_data_partition_impl | [→](./implementations/vllm-project_vllm_data_partition_impl.md) | ✅Principle:vllm-project_vllm_prompt_partitioning | User code | Pattern Doc |
| vllm-project_vllm_generate_method | [→](./implementations/vllm-project_vllm_generate_method.md) | ✅Principle:vllm-project_vllm_LLM_generate_dp | entrypoints/llm.py | API Doc |
| vllm-project_vllm_result_collector | [→](./implementations/vllm-project_vllm_result_collector.md) | ✅Principle:vllm-project_vllm_result_aggregation | User code | Pattern Doc |

---

## Orphan Implementations (Standalone)

These implementations were created from orphan files and are not yet linked to workflows.

### Benchmarks - Core

| Page | File | Source | Type |
|------|------|--------|------|
| vllm-project_vllm_BackendRequestFunc | [→](./implementations/vllm-project_vllm_BackendRequestFunc.md) | benchmarks/backend_request_func.py | API Doc |
| vllm-project_vllm_BatchInvarianceBenchmark | [→](./implementations/vllm-project_vllm_BatchInvarianceBenchmark.md) | benchmarks/benchmark_batch_invariance.py | API Doc |
| vllm-project_vllm_LongDocumentQABenchmark | [→](./implementations/vllm-project_vllm_LongDocumentQABenchmark.md) | benchmarks/benchmark_long_document_qa_throughput.py | API Doc |
| vllm-project_vllm_PrefixCachingBenchmark | [→](./implementations/vllm-project_vllm_PrefixCachingBenchmark.md) | benchmarks/benchmark_prefix_caching.py | API Doc |
| vllm-project_vllm_PrioritizationBenchmark | [→](./implementations/vllm-project_vllm_PrioritizationBenchmark.md) | benchmarks/benchmark_prioritization.py | API Doc |
| vllm-project_vllm_BenchmarkUtils | [→](./implementations/vllm-project_vllm_BenchmarkUtils.md) | benchmarks/benchmark_utils.py | API Doc |

### Benchmarks - CUTLASS

| Page | File | Source | Type |
|------|------|--------|------|
| vllm-project_vllm_SparseGEMMBenchmark | [→](./implementations/vllm-project_vllm_SparseGEMMBenchmark.md) | benchmarks/cutlass_benchmarks/sparse_benchmarks.py | API Doc |
| vllm-project_vllm_W8A8Benchmark | [→](./implementations/vllm-project_vllm_W8A8Benchmark.md) | benchmarks/cutlass_benchmarks/w8a8_benchmarks.py | API Doc |

### Benchmarks - Disaggregated

| Page | File | Source | Type |
|------|------|--------|------|
| vllm-project_vllm_DisaggPrefillProxy | [→](./implementations/vllm-project_vllm_DisaggPrefillProxy.md) | benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py | API Doc |

### Benchmarks - Fused Kernels

| Page | File | Source | Type |
|------|------|--------|------|
| vllm-project_vllm_RMSNormQuantBenchmark | [→](./implementations/vllm-project_vllm_RMSNormQuantBenchmark.md) | benchmarks/fused_kernels/layernorm_rms_benchmarks.py | API Doc |

### Benchmarks - Kernels (Quantization)

| Page | File | Source | Type |
|------|------|--------|------|
| vllm-project_vllm_BlockFP8Benchmark | [→](./implementations/vllm-project_vllm_BlockFP8Benchmark.md) | benchmarks/kernels/bench_block_fp8_gemm.py | API Doc |
| vllm-project_vllm_FP8GEMMBenchmark | [→](./implementations/vllm-project_vllm_FP8GEMMBenchmark.md) | benchmarks/kernels/bench_fp8_gemm.py | API Doc |
| vllm-project_vllm_INT8GEMMBenchmark | [→](./implementations/vllm-project_vllm_INT8GEMMBenchmark.md) | benchmarks/kernels/bench_int8_gemm.py | API Doc |
| vllm-project_vllm_MXFP4Benchmark | [→](./implementations/vllm-project_vllm_MXFP4Benchmark.md) | benchmarks/kernels/bench_mxfp4_qutlass.py | API Doc |
| vllm-project_vllm_NVFP4Benchmark | [→](./implementations/vllm-project_vllm_NVFP4Benchmark.md) | benchmarks/kernels/bench_nvfp4_gemm.py | API Doc |
| vllm-project_vllm_NVFP4HadamardBenchmark | [→](./implementations/vllm-project_vllm_NVFP4HadamardBenchmark.md) | benchmarks/kernels/bench_nvfp4_qutlass.py | API Doc |
| vllm-project_vllm_PerTokenFP8QuantBenchmark | [→](./implementations/vllm-project_vllm_PerTokenFP8QuantBenchmark.md) | benchmarks/kernels/bench_per_token_quant_fp8.py | API Doc |
| vllm-project_vllm_benchmark_quant_kernels | [→](./implementations/vllm-project_vllm_benchmark_quant_kernels.md) | benchmarks/kernels/benchmark_quant.py | API Doc |
| vllm-project_vllm_benchmark_per_token_group_quant | [→](./implementations/vllm-project_vllm_benchmark_per_token_group_quant.md) | benchmarks/kernels/benchmark_per_token_group_quant.py | API Doc |
| vllm-project_vllm_w8a8_block_fp8_tuner | [→](./implementations/vllm-project_vllm_w8a8_block_fp8_tuner.md) | benchmarks/kernels/benchmark_w8a8_block_fp8.py | API Doc |

### Benchmarks - Kernels (Activation & Attention)

| Page | File | Source | Type |
|------|------|--------|------|
| vllm-project_vllm_SiLUMulFP8QuantBenchmark | [→](./implementations/vllm-project_vllm_SiLUMulFP8QuantBenchmark.md) | benchmarks/kernels/benchmark_2d_silu_mul_fp8_quant.py | API Doc |
| vllm-project_vllm_ActivationBenchmark | [→](./implementations/vllm-project_vllm_ActivationBenchmark.md) | benchmarks/kernels/benchmark_activation.py | API Doc |
| vllm-project_vllm_silu_mul_fp8_quant_benchmark | [→](./implementations/vllm-project_vllm_silu_mul_fp8_quant_benchmark.md) | benchmarks/kernels/benchmark_silu_mul_fp8_quant.py | API Doc |
| vllm-project_vllm_trtllm_decode_attention_benchmark | [→](./implementations/vllm-project_vllm_trtllm_decode_attention_benchmark.md) | benchmarks/kernels/benchmark_trtllm_decode_attention.py | API Doc |
| vllm-project_vllm_trtllm_prefill_attention_benchmark | [→](./implementations/vllm-project_vllm_trtllm_prefill_attention_benchmark.md) | benchmarks/kernels/benchmark_trtllm_prefill_attention.py | API Doc |

### Benchmarks - Kernels (MOE)

| Page | File | Source | Type |
|------|------|--------|------|
| vllm-project_vllm_NVFP4MOEBenchmark | [→](./implementations/vllm-project_vllm_NVFP4MOEBenchmark.md) | benchmarks/kernels/benchmark_cutlass_fp4_moe.py | API Doc |
| vllm-project_vllm_CUTLASSFP8MOEBenchmark | [→](./implementations/vllm-project_vllm_CUTLASSFP8MOEBenchmark.md) | benchmarks/kernels/benchmark_cutlass_moe_fp8.py | API Doc |
| vllm-project_vllm_benchmark_moe_kernels | [→](./implementations/vllm-project_vllm_benchmark_moe_kernels.md) | benchmarks/kernels/benchmark_moe.py | API Doc |
| vllm-project_vllm_benchmark_moe_permute_unpermute | [→](./implementations/vllm-project_vllm_benchmark_moe_permute_unpermute.md) | benchmarks/kernels/benchmark_moe_permute_unpermute.py | API Doc |

### Benchmarks - Kernels (GEMM & Collective)

| Page | File | Source | Type |
|------|------|--------|------|
| vllm-project_vllm_BitBLASBenchmark | [→](./implementations/vllm-project_vllm_BitBLASBenchmark.md) | benchmarks/kernels/benchmark_bitblas.py | API Doc |
| vllm-project_vllm_FusedCollectiveBenchmark | [→](./implementations/vllm-project_vllm_FusedCollectiveBenchmark.md) | benchmarks/kernels/benchmark_fused_collective.py | API Doc |
| vllm-project_vllm_CUTLASSGroupedGEMMBenchmark | [→](./implementations/vllm-project_vllm_CUTLASSGroupedGEMMBenchmark.md) | benchmarks/kernels/benchmark_grouped_gemm_cutlass.py | API Doc |
| vllm-project_vllm_MacheteBenchmark | [→](./implementations/vllm-project_vllm_MacheteBenchmark.md) | benchmarks/kernels/benchmark_machete.py | API Doc |
| vllm-project_vllm_benchmark_marlin_quantized_gemm | [→](./implementations/vllm-project_vllm_benchmark_marlin_quantized_gemm.md) | benchmarks/kernels/benchmark_marlin.py | API Doc |
| vllm-project_vllm_deepgemm_fp8_block_benchmark | [→](./implementations/vllm-project_vllm_deepgemm_fp8_block_benchmark.md) | benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py | API Doc |

### Benchmarks - Kernels (RoPE, Norm, Cache)

| Page | File | Source | Type |
|------|------|--------|------|
| vllm-project_vllm_benchmark_mla_k_concat | [→](./implementations/vllm-project_vllm_benchmark_mla_k_concat.md) | benchmarks/kernels/benchmark_mla_k_concat.py | API Doc |
| vllm-project_vllm_benchmark_reshape_and_cache | [→](./implementations/vllm-project_vllm_benchmark_reshape_and_cache.md) | benchmarks/kernels/benchmark_reshape_and_cache.py | API Doc |
| vllm-project_vllm_benchmark_reshape_and_cache_flash | [→](./implementations/vllm-project_vllm_benchmark_reshape_and_cache_flash.md) | benchmarks/kernels/benchmark_reshape_and_cache_flash.py | API Doc |
| vllm-project_vllm_benchmark_rmsnorm | [→](./implementations/vllm-project_vllm_benchmark_rmsnorm.md) | benchmarks/kernels/benchmark_rmsnorm.py | API Doc |
| vllm-project_vllm_benchmark_rope | [→](./implementations/vllm-project_vllm_benchmark_rope.md) | benchmarks/kernels/benchmark_rope.py | API Doc |

### Benchmarks - Utilities

| Page | File | Source | Type |
|------|------|--------|------|
| vllm-project_vllm_benchmark_utils | [→](./implementations/vllm-project_vllm_benchmark_utils.md) | benchmarks/kernels/utils.py | API Doc |
| vllm-project_vllm_weight_shapes_config | [→](./implementations/vllm-project_vllm_weight_shapes_config.md) | benchmarks/kernels/weight_shapes.py | API Doc |

### Benchmarks - Multi-turn

| Page | File | Source | Type |
|------|------|--------|------|
| vllm-project_vllm_multi_turn_dataset_generator | [→](./implementations/vllm-project_vllm_multi_turn_dataset_generator.md) | benchmarks/multi_turn/bench_dataset.py | API Doc |
| vllm-project_vllm_sharegpt_to_openai_converter | [→](./implementations/vllm-project_vllm_sharegpt_to_openai_converter.md) | benchmarks/multi_turn/convert_sharegpt_to_openai.py | API Doc |

### Kernel Generators

| Page | File | Source | Type |
|------|------|--------|------|
| vllm-project_vllm_marlin_moe_kernel_generator | [→](./implementations/vllm-project_vllm_marlin_moe_kernel_generator.md) | csrc/moe/marlin_moe_wna16/generate_kernels.py | API Doc |
| vllm-project_vllm_GPTQMarlinKernelGenerator | [→](./implementations/vllm-project_vllm_GPTQMarlinKernelGenerator.md) | csrc/quantization/gptq_marlin/generate_kernels.py | API Doc |
| vllm-project_vllm_MacheteKernelGenerator | [→](./implementations/vllm-project_vllm_MacheteKernelGenerator.md) | csrc/quantization/machete/generate.py | API Doc |
| vllm-project_vllm_CutlassLibraryExtension | [→](./implementations/vllm-project_vllm_CutlassLibraryExtension.md) | csrc/cutlass_extensions/vllm_cutlass_library_extension.py | API Doc |

### Examples - RLHF

| Page | File | Source | Type |
|------|------|--------|------|
| vllm-project_vllm_rlhf_training_inference_separation | [→](./implementations/vllm-project_vllm_rlhf_training_inference_separation.md) | examples/offline_inference/rlhf.py | API Doc |
| vllm-project_vllm_rlhf_colocated_training_inference | [→](./implementations/vllm-project_vllm_rlhf_colocated_training_inference.md) | examples/offline_inference/rlhf_colocate.py | API Doc |
| vllm-project_vllm_rlhf_with_online_quantization | [→](./implementations/vllm-project_vllm_rlhf_with_online_quantization.md) | examples/offline_inference/rlhf_online_quant.py | API Doc |
| vllm-project_vllm_rlhf_utilities | [→](./implementations/vllm-project_vllm_rlhf_utilities.md) | examples/offline_inference/rlhf_utils.py | API Doc |

### Examples - Other

| Page | File | Source | Type |
|------|------|--------|------|
| vllm-project_vllm_TensorizeModel | [→](./implementations/vllm-project_vllm_TensorizeModel.md) | examples/others/tensorize_vllm_model.py | API Doc |
| vllm-project_vllm_kv_cache_event_subscriber | [→](./implementations/vllm-project_vllm_kv_cache_event_subscriber.md) | examples/online_serving/kv_events_subscriber.py | API Doc |

### Examples - Pooling

| Page | File | Source | Type |
|------|------|--------|------|
| vllm-project_vllm_openai_classification_client | [→](./implementations/vllm-project_vllm_openai_classification_client.md) | examples/pooling/classify/openai_classification_client.py | API Doc |
| vllm-project_vllm_PrithviGeospatialMAE | [→](./implementations/vllm-project_vllm_PrithviGeospatialMAE.md) | examples/pooling/plugin/prithvi_geospatial_mae_offline.py | API Doc |
| vllm-project_vllm_geospatial_segmentation_online_client | [→](./implementations/vllm-project_vllm_geospatial_segmentation_online_client.md) | examples/pooling/plugin/prithvi_geospatial_mae_client.py | API Doc |
| vllm-project_vllm_geospatial_segmentation_offline_processor | [→](./implementations/vllm-project_vllm_geospatial_segmentation_offline_processor.md) | examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py | API Doc |
| vllm-project_vllm_openai_pooling_client | [→](./implementations/vllm-project_vllm_openai_pooling_client.md) | examples/pooling/pooling/openai_pooling_client.py | API Doc |
| vllm-project_vllm_cohere_rerank_client | [→](./implementations/vllm-project_vllm_cohere_rerank_client.md) | examples/pooling/score/cohere_rerank_client.py | API Doc |
| vllm-project_vllm_convert_model_to_seq_cls_tool | [→](./implementations/vllm-project_vllm_convert_model_to_seq_cls_tool.md) | examples/pooling/score/convert_model_to_seq_cls.py | API Doc |
| vllm-project_vllm_cross_encoder_score_client | [→](./implementations/vllm-project_vllm_cross_encoder_score_client.md) | examples/pooling/score/openai_cross_encoder_score.py | API Doc |
| vllm-project_vllm_openai_reranker_client | [→](./implementations/vllm-project_vllm_openai_reranker_client.md) | examples/pooling/score/openai_reranker.py | API Doc |
| vllm-project_vllm_named_entity_recognition_offline | [→](./implementations/vllm-project_vllm_named_entity_recognition_offline.md) | examples/pooling/token_classify/ner.py | API Doc |
| vllm-project_vllm_named_entity_recognition_client | [→](./implementations/vllm-project_vllm_named_entity_recognition_client.md) | examples/pooling/token_classify/ner_client.py | API Doc |
| vllm-project_vllm_jina_multimodal_embeddings | [→](./implementations/vllm-project_vllm_jina_multimodal_embeddings.md) | examples/pooling/token_embed/jina_embeddings_v4.py | API Doc |
| vllm-project_vllm_multi_vector_retrieval_offline | [→](./implementations/vllm-project_vllm_multi_vector_retrieval_offline.md) | examples/pooling/token_embed/multi_vector_retrieval.py | API Doc |
| vllm-project_vllm_multi_vector_retrieval_client | [→](./implementations/vllm-project_vllm_multi_vector_retrieval_client.md) | examples/pooling/token_embed/multi_vector_retrieval_client.py | API Doc |

### Build & Tools

| Page | File | Source | Type |
|------|------|--------|------|
| vllm-project_vllm_SetupPy | [→](./implementations/vllm-project_vllm_SetupPy.md) | setup.py | API Doc |
| vllm-project_vllm_BuildTimeReporter | [→](./implementations/vllm-project_vllm_BuildTimeReporter.md) | tools/report_build_time_ninja.py | API Doc |

### Core Infrastructure

| Page | File | Source | Type |
|------|------|--------|------|
| vllm-project_vllm_AiterOps | [→](./implementations/vllm-project_vllm_AiterOps.md) | vllm/_aiter_ops.py | API Doc |
| vllm-project_vllm_CustomOps | [→](./implementations/vllm-project_vllm_CustomOps.md) | vllm/_custom_ops.py | API Doc |
| vllm-project_vllm_IPEXOps | [→](./implementations/vllm-project_vllm_IPEXOps.md) | vllm/_ipex_ops.py | API Doc |
| vllm-project_vllm_backward_compatibility_linter | [→](./implementations/vllm-project_vllm_backward_compatibility_linter.md) | vllm/_bc_linter.py | API Doc |
| vllm-project_vllm_beam_search_algorithm | [→](./implementations/vllm-project_vllm_beam_search_algorithm.md) | vllm/beam_search.py | API Doc |
| vllm-project_vllm_CollectEnv | [→](./implementations/vllm-project_vllm_CollectEnv.md) | vllm/collect_env.py | API Doc |
| vllm-project_vllm_http_connection_utilities | [→](./implementations/vllm-project_vllm_http_connection_utilities.md) | vllm/connections.py | API Doc |
| vllm-project_vllm_EnvOverride | [→](./implementations/vllm-project_vllm_EnvOverride.md) | vllm/env_override.py | API Doc |
| vllm-project_vllm_ForwardContext | [→](./implementations/vllm-project_vllm_ForwardContext.md) | vllm/forward_context.py | API Doc |
| vllm-project_vllm_Logger | [→](./implementations/vllm-project_vllm_Logger.md) | vllm/logger.py | API Doc |
| vllm-project_vllm_logits_processor | [→](./implementations/vllm-project_vllm_logits_processor.md) | vllm/logits_process.py | API Doc |
| vllm-project_vllm_logprobs_data_structures | [→](./implementations/vllm-project_vllm_logprobs_data_structures.md) | vllm/logprobs.py | API Doc |
| vllm-project_vllm_pooling_params | [→](./implementations/vllm-project_vllm_pooling_params.md) | vllm/pooling_params.py | API Doc |
| vllm-project_vllm_ScalarType | [→](./implementations/vllm-project_vllm_ScalarType.md) | vllm/scalar_type.py | API Doc |
| vllm-project_vllm_request_metrics_and_tensors | [→](./implementations/vllm-project_vllm_request_metrics_and_tensors.md) | vllm/sequence.py | API Doc |
| vllm-project_vllm_opentelemetry_tracing | [→](./implementations/vllm-project_vllm_opentelemetry_tracing.md) | vllm/tracing.py | API Doc |
| vllm-project_vllm_version_management | [→](./implementations/vllm-project_vllm_version_management.md) | vllm/version.py | API Doc |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
