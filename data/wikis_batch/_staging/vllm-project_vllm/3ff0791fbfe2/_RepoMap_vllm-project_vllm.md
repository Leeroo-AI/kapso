# Repository Map: vllm-project_vllm

> **Compact index** of repository files.
> Each file has a detail page in `_files/` with Understanding to fill.
> Mark files as ✅ explored in the table below as you complete them.

| Property | Value |
|----------|-------|
| Repository | https://github.com/vllm-project/vllm |
| Branch | main |
| Generated | 2025-12-17 18:59 |
| Python Files | 200 |
| Total Lines | 55,196 |
| Explored | 200/200 |

## Structure

📦 **Packages:** benchmarks, cmake, tools, vllm
📝 **Examples:** examples
🧪 **Tests:** tests

📖 README: `README.md`
⚙️ Setup: `pyproject.toml`

---

## 📦 Package Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `benchmarks/backend_request_func.py` | 657 | Unified async HTTP handlers | — | [→](./_files/benchmarks_backend_request_func_py.md) |
| ✅ | `benchmarks/benchmark_batch_invariance.py` | 380 | Measure batch-invariant mode overhead | — | [→](./_files/benchmarks_benchmark_batch_invariance_py.md) |
| ✅ | `benchmarks/benchmark_block_pool.py` | 74 | Performance test BlockPool operations | — | [→](./_files/benchmarks_benchmark_block_pool_py.md) |
| ✅ | `benchmarks/benchmark_hash.py` | 120 | Compare hash function performance | — | [→](./_files/benchmarks_benchmark_hash_py.md) |
| ✅ | `benchmarks/benchmark_latency.py` | 17 | Deprecation notice for latency | — | [→](./_files/benchmarks_benchmark_latency_py.md) |
| ✅ | `benchmarks/benchmark_long_document_qa_throughput.py` | 202 | Test prefix caching throughput | — | [→](./_files/benchmarks_benchmark_long_document_qa_throughput_py.md) |
| ✅ | `benchmarks/benchmark_ngram_proposer.py` | 215 | Benchmark ngram speculative decoding | Workflow: vllm-project_vllm_Speculative_Decoding | [→](./_files/benchmarks_benchmark_ngram_proposer_py.md) |
| ✅ | `benchmarks/benchmark_prefix_block_hash.py` | 110 | Compare prefix cache hashing | — | [→](./_files/benchmarks_benchmark_prefix_block_hash_py.md) |
| ✅ | `benchmarks/benchmark_prefix_caching.py` | 277 | Benchmark automatic prefix caching | — | [→](./_files/benchmarks_benchmark_prefix_caching_py.md) |
| ✅ | `benchmarks/benchmark_prioritization.py` | 221 | Test request prioritization throughput | — | [→](./_files/benchmarks_benchmark_prioritization_py.md) |
| ✅ | `benchmarks/benchmark_serving.py` | 17 | Deprecation notice for serving | — | [→](./_files/benchmarks_benchmark_serving_py.md) |
| ✅ | `benchmarks/benchmark_serving_structured_output.py` | 1040 | Benchmark serving structured outputs | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/benchmarks_benchmark_serving_structured_output_py.md) |
| ✅ | `benchmarks/benchmark_throughput.py` | 17 | Deprecation notice for throughput | — | [→](./_files/benchmarks_benchmark_throughput_py.md) |
| ✅ | `benchmarks/benchmark_utils.py` | 125 | Shared utilities for benchmarks | — | [→](./_files/benchmarks_benchmark_utils_py.md) |
| ✅ | `benchmarks/cutlass_benchmarks/sparse_benchmarks.py` | 515 | Benchmark 2:4 sparse GEMM | — | [→](./_files/benchmarks_cutlass_benchmarks_sparse_benchmarks_py.md) |
| ✅ | `benchmarks/cutlass_benchmarks/utils.py` | 100 | Tensor utilities for CUTLASS | — | [→](./_files/benchmarks_cutlass_benchmarks_utils_py.md) |
| ✅ | `benchmarks/cutlass_benchmarks/w8a8_benchmarks.py` | 372 | Benchmark W8A8 quantized GEMM | — | [→](./_files/benchmarks_cutlass_benchmarks_w8a8_benchmarks_py.md) |
| ✅ | `benchmarks/cutlass_benchmarks/weight_shapes.py` | 46 | Reference model GEMM shapes | — | [→](./_files/benchmarks_cutlass_benchmarks_weight_shapes_py.md) |
| ✅ | `benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py` | 260 | Proxy for disaggregated prefill/decode | — | [→](./_files/benchmarks_disagg_benchmarks_disagg_prefill_proxy_server_py.md) |
| ✅ | `benchmarks/disagg_benchmarks/rate_limiter.py` | 45 | Token bucket rate limiter | — | [→](./_files/benchmarks_disagg_benchmarks_rate_limiter_py.md) |
| ✅ | `benchmarks/disagg_benchmarks/request_queue.py` | 39 | Async request queue manager | — | [→](./_files/benchmarks_disagg_benchmarks_request_queue_py.md) |
| ✅ | `benchmarks/disagg_benchmarks/round_robin_proxy.py` | 63 | Simple round-robin load balancer | — | [→](./_files/benchmarks_disagg_benchmarks_round_robin_proxy_py.md) |
| ✅ | `benchmarks/disagg_benchmarks/visualize_benchmark_results.py` | 47 | Visualize disaggregation benchmark results | — | [→](./_files/benchmarks_disagg_benchmarks_visualize_benchmark_results_py.md) |
| ✅ | `benchmarks/fused_kernels/layernorm_rms_benchmarks.py` | 310 | Benchmark fused RMSNorm+quantization | — | [→](./_files/benchmarks_fused_kernels_layernorm_rms_benchmarks_py.md) |
| ✅ | `benchmarks/kernels/bench_block_fp8_gemm.py` | 160 | Benchmark W8A8 block FP8 | — | [→](./_files/benchmarks_kernels_bench_block_fp8_gemm_py.md) |
| ✅ | `benchmarks/kernels/bench_fp8_gemm.py` | 159 | Benchmark FP8 GEMM variants | — | [→](./_files/benchmarks_kernels_bench_fp8_gemm_py.md) |
| ✅ | `benchmarks/kernels/bench_int8_gemm.py` | 169 | Benchmark INT8 GEMM variants | — | [→](./_files/benchmarks_kernels_bench_int8_gemm_py.md) |
| ✅ | `benchmarks/kernels/bench_mxfp4_qutlass.py` | 191 | Benchmark MXFP4 quantized GEMM | — | [→](./_files/benchmarks_kernels_bench_mxfp4_qutlass_py.md) |
| ✅ | `benchmarks/kernels/bench_nvfp4_gemm.py` | 198 | Benchmark NVFP4 quantized GEMM | — | [→](./_files/benchmarks_kernels_bench_nvfp4_gemm_py.md) |
| ✅ | `benchmarks/kernels/bench_nvfp4_qutlass.py` | 207 | Benchmark NVFP4 with Hadamard | — | [→](./_files/benchmarks_kernels_bench_nvfp4_qutlass_py.md) |
| ✅ | `benchmarks/kernels/bench_per_token_quant_fp8.py` | 270 | Benchmark per-token FP8 quantization | — | [→](./_files/benchmarks_kernels_bench_per_token_quant_fp8_py.md) |
| ✅ | `benchmarks/kernels/benchmark_2d_silu_mul_fp8_quant.py` | 244 | Benchmark fused SiLU-mul-quant | — | [→](./_files/benchmarks_kernels_benchmark_2d_silu_mul_fp8_quant_py.md) |
| ✅ | `benchmarks/kernels/benchmark_activation.py` | 105 | Benchmark activation function kernels | — | [→](./_files/benchmarks_kernels_benchmark_activation_py.md) |
| ✅ | `benchmarks/kernels/benchmark_bitblas.py` | 244 | Benchmark BitBLAS quantized kernels | — | [→](./_files/benchmarks_kernels_benchmark_bitblas_py.md) |
| ✅ | `benchmarks/kernels/benchmark_cutlass_fp4_moe.py` | 504 | Benchmark NVFP4 MOE kernels | — | [→](./_files/benchmarks_kernels_benchmark_cutlass_fp4_moe_py.md) |
| ✅ | `benchmarks/kernels/benchmark_cutlass_moe_fp8.py` | 406 | Benchmarks CUTLASS FP8 MoE | — | [→](./_files/benchmarks_kernels_benchmark_cutlass_moe_fp8_py.md) |
| ✅ | `benchmarks/kernels/benchmark_device_communicators.py` | 508 | Benchmarks distributed communication backends | Workflow: vllm-project_vllm_Distributed_Data_Parallel_Inference | [→](./_files/benchmarks_kernels_benchmark_device_communicators_py.md) |
| ✅ | `benchmarks/kernels/benchmark_fused_collective.py` | 1129 | Benchmarks FlashInfer fused operations | — | [→](./_files/benchmarks_kernels_benchmark_fused_collective_py.md) |
| ✅ | `benchmarks/kernels/benchmark_grouped_gemm_cutlass.py` | 427 | Benchmarks CUTLASS grouped GEMM | — | [→](./_files/benchmarks_kernels_benchmark_grouped_gemm_cutlass_py.md) |
| ✅ | `benchmarks/kernels/benchmark_layernorm.py` | 94 | Benchmarks RMSNorm kernel performance | — | [→](./_files/benchmarks_kernels_benchmark_layernorm_py.md) |
| ✅ | `benchmarks/kernels/benchmark_lora.py` | 1488 | Comprehensive LoRA kernel benchmarking | Workflow: vllm-project_vllm_LoRA_Adapter_Inference | [→](./_files/benchmarks_kernels_benchmark_lora_py.md) |
| ✅ | `benchmarks/kernels/benchmark_machete.py` | 745 | Benchmarks Machete quantized GEMM | — | [→](./_files/benchmarks_kernels_benchmark_machete_py.md) |
| ✅ | `benchmarks/kernels/benchmark_marlin.py` | 413 | Benchmarks Marlin quantized GEMM | — | [→](./_files/benchmarks_kernels_benchmark_marlin_py.md) |
| ✅ | `benchmarks/kernels/benchmark_mla_k_concat.py` | 150 | Benchmarks k_nope/k_pe concatenation | — | [→](./_files/benchmarks_kernels_benchmark_mla_k_concat_py.md) |
| ✅ | `benchmarks/kernels/benchmark_moe.py` | 790 | Tunes and benchmarks MoE | — | [→](./_files/benchmarks_kernels_benchmark_moe_py.md) |
| ✅ | `benchmarks/kernels/benchmark_moe_align_block_size.py` | 87 | Benchmarks MoE block alignment | — | [→](./_files/benchmarks_kernels_benchmark_moe_align_block_size_py.md) |
| ✅ | `benchmarks/kernels/benchmark_moe_permute_unpermute.py` | 428 | Benchmarks MoE permute/unpermute operations | — | [→](./_files/benchmarks_kernels_benchmark_moe_permute_unpermute_py.md) |
| ✅ | `benchmarks/kernels/benchmark_mrope.py` | 322 | Benchmarks mRoPE for multimodal | Workflow: vllm-project_vllm_Vision_Language_Multimodal_Inference | [→](./_files/benchmarks_kernels_benchmark_mrope_py.md) |
| ✅ | `benchmarks/kernels/benchmark_paged_attention.py` | 250 | Benchmarks legacy paged attention | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/benchmarks_kernels_benchmark_paged_attention_py.md) |
| ✅ | `benchmarks/kernels/benchmark_per_token_group_quant.py` | 159 | Compares CUDA vs Triton | — | [→](./_files/benchmarks_kernels_benchmark_per_token_group_quant_py.md) |
| ✅ | `benchmarks/kernels/benchmark_quant.py` | 109 | Benchmarks FP8/INT8 quantization kernels | — | [→](./_files/benchmarks_kernels_benchmark_quant_py.md) |
| ✅ | `benchmarks/kernels/benchmark_reshape_and_cache.py` | 172 | Benchmarks reshape_and_cache KV operation | — | [→](./_files/benchmarks_kernels_benchmark_reshape_and_cache_py.md) |
| ✅ | `benchmarks/kernels/benchmark_reshape_and_cache_flash.py` | 210 | Benchmarks FlashInfer append_paged_kv_cache | — | [→](./_files/benchmarks_kernels_benchmark_reshape_and_cache_flash_py.md) |
| ✅ | `benchmarks/kernels/benchmark_rmsnorm.py` | 255 | Compares RMSNorm implementations | — | [→](./_files/benchmarks_kernels_benchmark_rmsnorm_py.md) |
| ✅ | `benchmarks/kernels/benchmark_rope.py` | 106 | Compares RoPE implementations | — | [→](./_files/benchmarks_kernels_benchmark_rope_py.md) |
| ✅ | `benchmarks/kernels/benchmark_shapes.py` | 94 | Defines weight matrix shapes | — | [→](./_files/benchmarks_kernels_benchmark_shapes_py.md) |
| ✅ | `benchmarks/kernels/benchmark_silu_mul_fp8_quant.py` | 720 | Comprehensive SiLU+Mul+FP8Quant benchmark | — | [→](./_files/benchmarks_kernels_benchmark_silu_mul_fp8_quant_py.md) |
| ✅ | `benchmarks/kernels/benchmark_trtllm_decode_attention.py` | 290 | Benchmarks TRT-LLM decode attention | — | [→](./_files/benchmarks_kernels_benchmark_trtllm_decode_attention_py.md) |
| ✅ | `benchmarks/kernels/benchmark_trtllm_prefill_attention.py` | 305 | Benchmarks TRT-LLM prefill attention | — | [→](./_files/benchmarks_kernels_benchmark_trtllm_prefill_attention_py.md) |
| ✅ | `benchmarks/kernels/benchmark_w8a8_block_fp8.py` | 415 | Tunes W8A8 block FP8 | — | [→](./_files/benchmarks_kernels_benchmark_w8a8_block_fp8_py.md) |
| ✅ | `benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py` | 435 | Benchmarks DeepGEMM FP8 block | — | [→](./_files/benchmarks_kernels_deepgemm_benchmark_fp8_block_dense_gemm_py.md) |
| ✅ | `benchmarks/kernels/graph_machete_bench.py` | 64 | Generates Machete visualization graphs | — | [→](./_files/benchmarks_kernels_graph_machete_bench_py.md) |
| ✅ | `benchmarks/kernels/utils.py` | 214 | Benchmark utility classes | — | [→](./_files/benchmarks_kernels_utils_py.md) |
| ✅ | `benchmarks/kernels/weight_shapes.py` | 104 | Defines weight shapes with TP | — | [→](./_files/benchmarks_kernels_weight_shapes_py.md) |
| ✅ | `benchmarks/multi_turn/bench_dataset.py` | 600 | Generates synthetic conversation datasets | — | [→](./_files/benchmarks_multi_turn_bench_dataset_py.md) |
| ✅ | `benchmarks/multi_turn/bench_utils.py` | 28 | Shared utilities for benchmarks | — | [→](./_files/benchmarks_multi_turn_bench_utils_py.md) |
| ✅ | `benchmarks/multi_turn/benchmark_serving_multi_turn.py` | 1666 | Comprehensive multi-turn serving framework | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/benchmarks_multi_turn_benchmark_serving_multi_turn_py.md) |
| ✅ | `benchmarks/multi_turn/convert_sharegpt_to_openai.py` | 354 | Converts ShareGPT to OpenAI | — | [→](./_files/benchmarks_multi_turn_convert_sharegpt_to_openai_py.md) |
| ✅ | `benchmarks/overheads/benchmark_hashing.py` | 64 | Profiles block hashing overhead | — | [→](./_files/benchmarks_overheads_benchmark_hashing_py.md) |
| ✅ | `cmake/hipify.py` | 80 | Converts CUDA to HIP | — | [→](./_files/cmake_hipify_py.md) |
| ✅ | `tools/generate_cmake_presets.py` | 180 | Auto-generates CMake configuration | — | [→](./_files/tools_generate_cmake_presets_py.md) |
| ✅ | `tools/install_nixl_from_source_ubuntu.py` | 254 | Builds NIXL networking library | — | [→](./_files/tools_install_nixl_from_source_ubuntu_py.md) |
| ✅ | `tools/report_build_time_ninja.py` | 325 | Analyzes build performance bottlenecks | — | [→](./_files/tools_report_build_time_ninja_py.md) |
| ✅ | `vllm/__init__.py` | 107 | Package initialization and public API | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/vllm___init___py.md) |
| ✅ | `vllm/_aiter_ops.py` | 1333 | ROCm AITER operations integration | — | [→](./_files/vllm__aiter_ops_py.md) |
| ✅ | `vllm/_bc_linter.py` | 54 | Backward compatibility linting decorators | — | [→](./_files/vllm__bc_linter_py.md) |
| ✅ | `vllm/_custom_ops.py` | 3080 | Custom PyTorch operations registry | — | [→](./_files/vllm__custom_ops_py.md) |
| ✅ | `vllm/_ipex_ops.py` | 457 | Intel Extension PyTorch operations | — | [→](./_files/vllm__ipex_ops_py.md) |
| ✅ | `vllm/beam_search.py` | 88 | Beam search algorithm implementation | — | [→](./_files/vllm_beam_search_py.md) |
| ✅ | `vllm/collect_env.py` | 857 | Environment diagnostic collection | — | [→](./_files/vllm_collect_env_py.md) |
| ✅ | `vllm/connections.py` | 189 | HTTP connection utilities | — | [→](./_files/vllm_connections_py.md) |
| ✅ | `vllm/env_override.py` | 378 | PyTorch compilation overrides | — | [→](./_files/vllm_env_override_py.md) |
| ✅ | `vllm/envs.py` | 1745 | Environment variable management | Workflow: vllm-project_vllm_Distributed_Data_Parallel_Inference | [→](./_files/vllm_envs_py.md) |
| ✅ | `vllm/forward_context.py` | 358 | Forward pass context management | — | [→](./_files/vllm_forward_context_py.md) |
| ✅ | `vllm/logger.py` | 303 | Logging infrastructure configuration | — | [→](./_files/vllm_logger_py.md) |
| ✅ | `vllm/logits_process.py` | 121 | Logits processing for bad words | — | [→](./_files/vllm_logits_process_py.md) |
| ✅ | `vllm/logprobs.py` | 206 | Log probability data structures | — | [→](./_files/vllm_logprobs_py.md) |
| ✅ | `vllm/outputs.py` | 345 | Output data structures | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/vllm_outputs_py.md) |
| ✅ | `vllm/pooling_params.py` | 230 | Pooling model parameters | — | [→](./_files/vllm_pooling_params_py.md) |
| ✅ | `vllm/sampling_params.py` | 597 | Text generation parameters | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/vllm_sampling_params_py.md) |
| ✅ | `vllm/scalar_type.py` | 355 | Sub-byte numeric type system | — | [→](./_files/vllm_scalar_type_py.md) |
| ✅ | `vllm/scripts.py` | 17 | Deprecated CLI entry point | — | [→](./_files/vllm_scripts_py.md) |
| ✅ | `vllm/sequence.py` | 98 | Request metrics and tensors | — | [→](./_files/vllm_sequence_py.md) |
| ✅ | `vllm/tasks.py` | 13 | Task type definitions | — | [→](./_files/vllm_tasks_py.md) |
| ✅ | `vllm/tracing.py` | 135 | OpenTelemetry tracing integration | — | [→](./_files/vllm_tracing_py.md) |
| ✅ | `vllm/version.py` | 39 | Version management and compatibility | — | [→](./_files/vllm_version_py.md) |

## 📝 Example Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `examples/offline_inference/async_llm_streaming.py` | 111 | Streaming offline inference | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/examples_offline_inference_async_llm_streaming_py.md) |
| ✅ | `examples/offline_inference/audio_language.py` | 540 | Audio language models | Workflow: vllm-project_vllm_Vision_Language_Multimodal_Inference | [→](./_files/examples_offline_inference_audio_language_py.md) |
| ✅ | `examples/offline_inference/automatic_prefix_caching.py` | 103 | Automatic prefix caching | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/examples_offline_inference_automatic_prefix_caching_py.md) |
| ✅ | `examples/offline_inference/batch_llm_inference.py` | 93 | Ray Data batch inference | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/examples_offline_inference_batch_llm_inference_py.md) |
| ✅ | `examples/offline_inference/chat_with_tools.py` | 147 | Function calling demonstration | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_offline_inference_chat_with_tools_py.md) |
| ✅ | `examples/offline_inference/context_extension.py` | 68 | Context length extension | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/examples_offline_inference_context_extension_py.md) |
| ✅ | `examples/offline_inference/data_parallel.py` | 268 | Data parallel inference | Workflow: vllm-project_vllm_Distributed_Data_Parallel_Inference | [→](./_files/examples_offline_inference_data_parallel_py.md) |
| ✅ | `examples/offline_inference/disaggregated_prefill.py` | 127 | Disaggregated prefill pattern | Workflow: vllm-project_vllm_Distributed_Data_Parallel_Inference | [→](./_files/examples_offline_inference_disaggregated_prefill_py.md) |
| ✅ | `examples/offline_inference/encoder_decoder_multimodal.py` | 133 | Encoder-decoder multimodal models | Workflow: vllm-project_vllm_Vision_Language_Multimodal_Inference | [→](./_files/examples_offline_inference_encoder_decoder_multimodal_py.md) |
| ✅ | `examples/offline_inference/llm_engine_example.py` | 74 | LLMEngine API usage | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/examples_offline_inference_llm_engine_example_py.md) |
| ✅ | `examples/offline_inference/llm_engine_reset_kv.py` | 98 | Prefix cache resetting | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/examples_offline_inference_llm_engine_reset_kv_py.md) |
| ✅ | `examples/offline_inference/load_sharded_state.py` | 94 | Load sharded models | Workflow: vllm-project_vllm_Distributed_Data_Parallel_Inference | [→](./_files/examples_offline_inference_load_sharded_state_py.md) |
| ✅ | `examples/offline_inference/lora_with_quantization_inference.py` | 127 | LoRA with quantization | Workflow: vllm-project_vllm_LoRA_Adapter_Inference | [→](./_files/examples_offline_inference_lora_with_quantization_inference_py.md) |
| ✅ | `examples/offline_inference/metrics.py` | 50 | Metrics collection demonstration | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/examples_offline_inference_metrics_py.md) |
| ✅ | `examples/offline_inference/mistral-small.py` | 186 | Mistral-Small multimodal inference | Workflow: vllm-project_vllm_Vision_Language_Multimodal_Inference | [→](./_files/examples_offline_inference_mistral-small_py.md) |
| ✅ | `examples/offline_inference/mlpspeculator.py` | 72 | MLP speculative decoding | Workflow: vllm-project_vllm_Speculative_Decoding | [→](./_files/examples_offline_inference_mlpspeculator_py.md) |
| ✅ | `examples/offline_inference/multilora_inference.py` | 106 | Multiple LoRA adapters | Workflow: vllm-project_vllm_LoRA_Adapter_Inference | [→](./_files/examples_offline_inference_multilora_inference_py.md) |
| ✅ | `examples/offline_inference/prefix_caching.py` | 98 | Manual prefix caching | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/examples_offline_inference_prefix_caching_py.md) |
| ✅ | `examples/offline_inference/prompt_embed_inference.py` | 97 | Prompt embedding inputs | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/examples_offline_inference_prompt_embed_inference_py.md) |
| ✅ | `examples/offline_inference/qwen_1m.py` | 70 | 1M context length | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/examples_offline_inference_qwen_1m_py.md) |
| ✅ | `examples/offline_inference/reproducibility.py` | 46 | Reproducibility configuration | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/examples_offline_inference_reproducibility_py.md) |
| ✅ | `examples/offline_inference/rlhf.py` | 147 | RLHF training-inference separation | — | [→](./_files/examples_offline_inference_rlhf_py.md) |
| ✅ | `examples/offline_inference/rlhf_colocate.py` | 251 | Co-located RLHF | — | [→](./_files/examples_offline_inference_rlhf_colocate_py.md) |
| ✅ | `examples/offline_inference/rlhf_online_quant.py` | 162 | RLHF with quantization | — | [→](./_files/examples_offline_inference_rlhf_online_quant_py.md) |
| ✅ | `examples/offline_inference/rlhf_utils.py` | 168 | RLHF utilities | — | [→](./_files/examples_offline_inference_rlhf_utils_py.md) |
| ✅ | `examples/offline_inference/save_sharded_state.py` | 87 | Save sharded models | Workflow: vllm-project_vllm_Distributed_Data_Parallel_Inference | [→](./_files/examples_offline_inference_save_sharded_state_py.md) |
| ✅ | `examples/offline_inference/simple_profiling.py` | 52 | Torch profiler integration | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/examples_offline_inference_simple_profiling_py.md) |
| ✅ | `examples/offline_inference/skip_loading_weights_in_engine_init.py` | 53 | Deferred weight loading | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/examples_offline_inference_skip_loading_weights_in_engine_init_py.md) |
| ✅ | `examples/offline_inference/spec_decode.py` | 234 | Speculative decoding methods | Workflow: vllm-project_vllm_Speculative_Decoding | [→](./_files/examples_offline_inference_spec_decode_py.md) |
| ✅ | `examples/offline_inference/structured_outputs.py` | 113 | Structured output constraints | Workflow: vllm-project_vllm_Basic_Offline_LLM_Inference | [→](./_files/examples_offline_inference_structured_outputs_py.md) |
| ✅ | `examples/offline_inference/torchrun_dp_example.py` | 151 | Torchrun data parallelism | Workflow: vllm-project_vllm_Distributed_Data_Parallel_Inference | [→](./_files/examples_offline_inference_torchrun_dp_example_py.md) |
| ✅ | `examples/offline_inference/torchrun_example.py` | 76 | Torchrun tensor parallelism | Workflow: vllm-project_vllm_Distributed_Data_Parallel_Inference | [→](./_files/examples_offline_inference_torchrun_example_py.md) |
| ✅ | `examples/offline_inference/vision_language.py` | 2243 | Vision-language models | Workflow: vllm-project_vllm_Vision_Language_Multimodal_Inference | [→](./_files/examples_offline_inference_vision_language_py.md) |
| ✅ | `examples/offline_inference/vision_language_multi_image.py` | 1542 | Multi-image inference | Workflow: vllm-project_vllm_Vision_Language_Multimodal_Inference | [→](./_files/examples_offline_inference_vision_language_multi_image_py.md) |
| ✅ | `examples/online_serving/api_client.py` | 93 | Demo HTTP client for API | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_api_client_py.md) |
| ✅ | `examples/online_serving/gradio_openai_chatbot_webserver.py` | 112 | Gradio chatbot UI | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_gradio_openai_chatbot_webserver_py.md) |
| ✅ | `examples/online_serving/gradio_webserver.py` | 75 | Gradio UI for legacy API | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_gradio_webserver_py.md) |
| ✅ | `examples/online_serving/kv_events_subscriber.py` | 117 | KV cache event monitoring | — | [→](./_files/examples_online_serving_kv_events_subscriber_py.md) |
| ✅ | `examples/online_serving/multi_instance_data_parallel.py` | 87 | Data parallel multi-instance | Workflow: vllm-project_vllm_Distributed_Data_Parallel_Inference | [→](./_files/examples_online_serving_multi_instance_data_parallel_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_client.py` | 64 | Basic OpenAI chat client | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_openai_chat_completion_client_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_client_for_multimodal.py` | 353 | Multimodal input client | Workflow: vllm-project_vllm_Vision_Language_Multimodal_Inference, vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_openai_chat_completion_client_for_multimodal_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_client_with_tools.py` | 195 | Function calling with tools | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_openai_chat_completion_client_with_tools_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_client_with_tools_required.py` | 130 | Required tool choice demo | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_openai_chat_completion_client_with_tools_required_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_client_with_tools_xlam.py` | 245 | xLAM function calling | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_openai_chat_completion_client_with_tools_xlam_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_client_with_tools_xlam_streaming.py` | 273 | xLAM streaming function calling | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_openai_chat_completion_client_with_tools_xlam_streaming_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_tool_calls_with_reasoning.py` | 170 | Reasoning models with tools | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_openai_chat_completion_tool_calls_with_reasoning_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_with_reasoning.py` | 65 | Reasoning model completions | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_openai_chat_completion_with_reasoning_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_with_reasoning_streaming.py` | 73 | Streaming reasoning responses | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_openai_chat_completion_with_reasoning_streaming_py.md) |
| ✅ | `examples/online_serving/openai_completion_client.py` | 53 | OpenAI completions API client | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_openai_completion_client_py.md) |
| ✅ | `examples/online_serving/openai_responses_client.py` | 44 | OpenAI Responses API example | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_openai_responses_client_py.md) |
| ✅ | `examples/online_serving/openai_responses_client_with_mcp_tools.py` | 184 | MCP tools integration | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_openai_responses_client_with_mcp_tools_py.md) |
| ✅ | `examples/online_serving/openai_responses_client_with_tools.py` | 83 | Responses API with tools | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_openai_responses_client_with_tools_py.md) |
| ✅ | `examples/online_serving/openai_transcription_client.py` | 97 | Audio transcription API client | Workflow: vllm-project_vllm_Vision_Language_Multimodal_Inference | [→](./_files/examples_online_serving_openai_transcription_client_py.md) |
| ✅ | `examples/online_serving/openai_translation_client.py` | 75 | Audio translation API client | Workflow: vllm-project_vllm_Vision_Language_Multimodal_Inference | [→](./_files/examples_online_serving_openai_translation_client_py.md) |
| ✅ | `examples/online_serving/prompt_embed_inference_with_openai_client.py` | 79 | Prompt embeddings inference | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_prompt_embed_inference_with_openai_client_py.md) |
| ✅ | `examples/online_serving/ray_serve_deepseek.py` | 55 | Ray Serve deployment | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_ray_serve_deepseek_py.md) |
| ✅ | `examples/online_serving/retrieval_augmented_generation_with_langchain.py` | 257 | RAG with LangChain | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_retrieval_augmented_generation_with_langchain_py.md) |
| ✅ | `examples/online_serving/retrieval_augmented_generation_with_llamaindex.py` | 225 | RAG with LlamaIndex | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_retrieval_augmented_generation_with_llamaindex_py.md) |
| ✅ | `examples/online_serving/streamlit_openai_chatbot_webserver.py` | 311 | Streamlit chatbot with reasoning | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_streamlit_openai_chatbot_webserver_py.md) |
| ✅ | `examples/online_serving/token_generation_client.py` | 49 | Token-level generation API | Workflow: vllm-project_vllm_Online_API_Serving | [→](./_files/examples_online_serving_token_generation_client_py.md) |
| ✅ | `examples/online_serving/utils.py` | 26 | Shared utility functions | — | [→](./_files/examples_online_serving_utils_py.md) |
| ✅ | `examples/others/tensorize_vllm_model.py` | 392 | Serializes models for fast loading | — | [→](./_files/examples_others_tensorize_vllm_model_py.md) |
| ✅ | `examples/pooling/classify/openai_classification_client.py` | 53 | OpenAI classification API client | — | [→](./_files/examples_pooling_classify_openai_classification_client_py.md) |
| ✅ | `examples/pooling/plugin/prithvi_geospatial_mae_client.py` | 56 | Geospatial segmentation client | — | [→](./_files/examples_pooling_plugin_prithvi_geospatial_mae_client_py.md) |
| ✅ | `examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py` | 58 | Geospatial I/O processor | — | [→](./_files/examples_pooling_plugin_prithvi_geospatial_mae_io_processor_py.md) |
| ✅ | `examples/pooling/plugin/prithvi_geospatial_mae_offline.py` | 419 | Geospatial flood segmentation | — | [→](./_files/examples_pooling_plugin_prithvi_geospatial_mae_offline_py.md) |
| ✅ | `examples/pooling/pooling/openai_pooling_client.py` | 63 | Generic pooling API client | — | [→](./_files/examples_pooling_pooling_openai_pooling_client_py.md) |
| ✅ | `examples/pooling/pooling/vision_language_pooling.py` | 410 | Vision-language multimodal pooling | Workflow: vllm-project_vllm_Vision_Language_Multimodal_Inference | [→](./_files/examples_pooling_pooling_vision_language_pooling_py.md) |
| ✅ | `examples/pooling/score/cohere_rerank_client.py` | 47 | Cohere SDK reranking | — | [→](./_files/examples_pooling_score_cohere_rerank_client_py.md) |
| ✅ | `examples/pooling/score/convert_model_to_seq_cls.py` | 134 | Convert LM to classifier | — | [→](./_files/examples_pooling_score_convert_model_to_seq_cls_py.md) |
| ✅ | `examples/pooling/score/openai_cross_encoder_score.py` | 63 | Cross-encoder scoring client | — | [→](./_files/examples_pooling_score_openai_cross_encoder_score_py.md) |
| ✅ | `examples/pooling/score/openai_reranker.py` | 42 | OpenAI-compatible reranking | — | [→](./_files/examples_pooling_score_openai_reranker_py.md) |
| ✅ | `examples/pooling/token_classify/ner.py` | 54 | Named entity recognition offline | — | [→](./_files/examples_pooling_token_classify_ner_py.md) |
| ✅ | `examples/pooling/token_classify/ner_client.py` | 71 | Named entity recognition client | — | [→](./_files/examples_pooling_token_classify_ner_client_py.md) |
| ✅ | `examples/pooling/token_embed/jina_embeddings_v4.py` | 71 | Jina multimodal token embeddings | — | [→](./_files/examples_pooling_token_embed_jina_embeddings_v4_py.md) |
| ✅ | `examples/pooling/token_embed/multi_vector_retrieval.py` | 56 | Multi-vector retrieval offline | — | [→](./_files/examples_pooling_token_embed_multi_vector_retrieval_py.md) |
| ✅ | `examples/pooling/token_embed/multi_vector_retrieval_client.py` | 54 | Multi-vector retrieval client | — | [→](./_files/examples_pooling_token_embed_multi_vector_retrieval_client_py.md) |

## 🧪 Test Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `tests/__init__.py` | 0 | Package initialization marker | — | [→](./_files/tests___init___py.md) |
| ✅ | `tests/ci_envs.py` | 52 | CI-specific environment variables | — | [→](./_files/tests_ci_envs_py.md) |
| ✅ | `tests/conftest.py` | 1517 | Pytest configuration and fixtures | — | [→](./_files/tests_conftest_py.md) |
| ✅ | `tests/test_config.py` | 1052 | Configuration system validation | — | [→](./_files/tests_test_config_py.md) |
| ✅ | `tests/test_embedded_commit.py` | 11 | Version metadata validation | — | [→](./_files/tests_test_embedded_commit_py.md) |
| ✅ | `tests/test_envs.py` | 456 | Environment variable system testing | — | [→](./_files/tests_test_envs_py.md) |
| ✅ | `tests/test_inputs.py` | 125 | Input parsing and preprocessing | — | [→](./_files/tests_test_inputs_py.md) |
| ✅ | `tests/test_logger.py` | 557 | Logging system comprehensive testing | — | [→](./_files/tests_test_logger_py.md) |
| ✅ | `tests/test_logprobs.py` | 210 | Logprobs data structure testing | — | [→](./_files/tests_test_logprobs_py.md) |
| ✅ | `tests/test_outputs.py` | 21 | Output dataclass compatibility | — | [→](./_files/tests_test_outputs_py.md) |
| ✅ | `tests/test_pooling_params.py` | 156 | PoolingParams validation testing | — | [→](./_files/tests_test_pooling_params_py.md) |
| ✅ | `tests/test_regression.py` | 79 | User-reported bug prevention | — | [→](./_files/tests_test_regression_py.md) |
| ✅ | `tests/test_routing_simulator.py` | 199 | MoE routing simulation | — | [→](./_files/tests_test_routing_simulator_py.md) |
| ✅ | `tests/test_scalartype.py` | 43 | Scalar type min/max validation | — | [→](./_files/tests_test_scalartype_py.md) |
| ✅ | `tests/test_seed_behavior.py` | 25 | Random seed reproducibility | — | [→](./_files/tests_test_seed_behavior_py.md) |
| ✅ | `tests/test_sequence.py` | 49 | IntermediateTensors equality testing | — | [→](./_files/tests_test_sequence_py.md) |
| ✅ | `tests/test_triton_utils.py` | 94 | Triton placeholder testing | — | [→](./_files/tests_test_triton_utils_py.md) |
| ✅ | `tests/test_version.py` | 38 | Version utilities testing | — | [→](./_files/tests_test_version_py.md) |
| ✅ | `tests/test_vllm_port.py` | 39 | VLLM_PORT environment validation | — | [→](./_files/tests_test_vllm_port_py.md) |
| ✅ | `tests/utils.py` | 1312 | Comprehensive test utilities library | — | [→](./_files/tests_utils_py.md) |

## 📄 Other Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `.buildkite/check-wheel-size.py` | 53 | Validates wheel size limits | — | [→](./_files/_buildkite_check-wheel-size_py.md) |
| ✅ | `csrc/cutlass_extensions/vllm_cutlass_library_extension.py` | 76 | Extends CUTLASS type system | — | [→](./_files/csrc_cutlass_extensions_vllm_cutlass_library_extension_py.md) |
| ✅ | `csrc/moe/marlin_moe_wna16/generate_kernels.py` | 306 | Generates Marlin MoE kernels | — | [→](./_files/csrc_moe_marlin_moe_wna16_generate_kernels_py.md) |
| ✅ | `csrc/quantization/gptq_marlin/generate_kernels.py` | 316 | Generates GPTQ quantization kernels | — | [→](./_files/csrc_quantization_gptq_marlin_generate_kernels_py.md) |
| ✅ | `csrc/quantization/machete/generate.py` | 694 | Generates optimized Machete kernels | — | [→](./_files/csrc_quantization_machete_generate_py.md) |
| ✅ | `setup.py` | 813 | Multi-platform build orchestration | — | [→](./_files/setup_py.md) |
| ✅ | `use_existing_torch.py` | 18 | Removes PyTorch dependency declarations | — | [→](./_files/use_existing_torch_py.md) |

---

## Page Indexes

Each page type has its own index file for tracking and integrity checking:

| Index | Description |
|-------|-------------|
| [Workflows](./_WorkflowIndex.md) | Workflow pages with step connections |
| [Principles](./_PrincipleIndex.md) | Principle pages with implementations |
| [Implementations](./_ImplementationIndex.md) | Implementation pages with source locations |
| [Environments](./_EnvironmentIndex.md) | Environment requirement pages |
| [Heuristics](./_HeuristicIndex.md) | Heuristic/tips pages |
