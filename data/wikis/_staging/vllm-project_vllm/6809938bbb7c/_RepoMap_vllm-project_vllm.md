# Repository Map: vllm-project_vllm

> **Compact index** of repository files.
> Each file has a detail page in `_files/` with Understanding to fill.
> Mark files as ✅ explored in the table below as you complete them.

| Property | Value |
|----------|-------|
| Repository | https://github.com/vllm-project/vllm |
| Branch | main |
| Generated | 2025-12-18 12:30 |
| Python Files | 200 |
| Total Lines | 55,243 |
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
| ✅ | `benchmarks/backend_request_func.py` | 657 | Async request handlers backends | — | [→](./_files/benchmarks_backend_request_func_py.md) |
| ✅ | `benchmarks/benchmark_batch_invariance.py` | 380 | Batch invariance overhead test | — | [→](./_files/benchmarks_benchmark_batch_invariance_py.md) |
| ✅ | `benchmarks/benchmark_block_pool.py` | 74 | KV cache block allocation | — | [→](./_files/benchmarks_benchmark_block_pool_py.md) |
| ✅ | `benchmarks/benchmark_hash.py` | 120 | Hash function comparison | — | [→](./_files/benchmarks_benchmark_hash_py.md) |
| ✅ | `benchmarks/benchmark_latency.py` | 17 | Deprecated latency CLI shim | — | [→](./_files/benchmarks_benchmark_latency_py.md) |
| ✅ | `benchmarks/benchmark_long_document_qa_throughput.py` | 202 | Long-context prefix caching | — | [→](./_files/benchmarks_benchmark_long_document_qa_throughput_py.md) |
| ✅ | `benchmarks/benchmark_ngram_proposer.py` | 215 | N-gram speculative decoding | Workflow: vllm-project_vllm_Speculative_Decoding | [→](./_files/benchmarks_benchmark_ngram_proposer_py.md) |
| ✅ | `benchmarks/benchmark_prefix_block_hash.py` | 110 | Prefix cache hashing perf | — | [→](./_files/benchmarks_benchmark_prefix_block_hash_py.md) |
| ✅ | `benchmarks/benchmark_prefix_caching.py` | 277 | Prefix caching efficiency | — | [→](./_files/benchmarks_benchmark_prefix_caching_py.md) |
| ✅ | `benchmarks/benchmark_prioritization.py` | 221 | Priority scheduling throughput | — | [→](./_files/benchmarks_benchmark_prioritization_py.md) |
| ✅ | `benchmarks/benchmark_serving.py` | 17 | Deprecated serving CLI shim | — | [→](./_files/benchmarks_benchmark_serving_py.md) |
| ✅ | `benchmarks/benchmark_serving_structured_output.py` | 1040 | Structured output constraints | Workflow: vllm-project_vllm_Structured_Output_Generation | [→](./_files/benchmarks_benchmark_serving_structured_output_py.md) |
| ✅ | `benchmarks/benchmark_throughput.py` | 17 | Deprecated throughput CLI shim | — | [→](./_files/benchmarks_benchmark_throughput_py.md) |
| ✅ | `benchmarks/benchmark_utils.py` | 125 | Shared benchmark utilities | — | [→](./_files/benchmarks_benchmark_utils_py.md) |
| ✅ | `benchmarks/cutlass_benchmarks/sparse_benchmarks.py` | 515 | 2:4 structured sparsity GEMM | — | [→](./_files/benchmarks_cutlass_benchmarks_sparse_benchmarks_py.md) |
| ✅ | `benchmarks/cutlass_benchmarks/utils.py` | 100 | Tensor generation utilities | — | [→](./_files/benchmarks_cutlass_benchmarks_utils_py.md) |
| ✅ | `benchmarks/cutlass_benchmarks/w8a8_benchmarks.py` | 372 | W8A8 quantized GEMM bench | — | [→](./_files/benchmarks_cutlass_benchmarks_w8a8_benchmarks_py.md) |
| ✅ | `benchmarks/cutlass_benchmarks/weight_shapes.py` | 46 | Model weight dimensions ref | — | [→](./_files/benchmarks_cutlass_benchmarks_weight_shapes_py.md) |
| ✅ | `benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py` | 260 | Prefill/decode split proxy | — | [→](./_files/benchmarks_disagg_benchmarks_disagg_prefill_proxy_server_py.md) |
| ✅ | `benchmarks/disagg_benchmarks/rate_limiter.py` | 45 | Token bucket rate limiter | — | [→](./_files/benchmarks_disagg_benchmarks_rate_limiter_py.md) |
| ✅ | `benchmarks/disagg_benchmarks/request_queue.py` | 39 | Async queue concurrency ctrl | — | [→](./_files/benchmarks_disagg_benchmarks_request_queue_py.md) |
| ✅ | `benchmarks/disagg_benchmarks/round_robin_proxy.py` | 63 | Load balancer proxy | — | [→](./_files/benchmarks_disagg_benchmarks_round_robin_proxy_py.md) |
| ✅ | `benchmarks/disagg_benchmarks/visualize_benchmark_results.py` | 47 | Results visualization | — | [→](./_files/benchmarks_disagg_benchmarks_visualize_benchmark_results_py.md) |
| ✅ | `benchmarks/fused_kernels/layernorm_rms_benchmarks.py` | 310 | RMSNorm layer performance | — | [→](./_files/benchmarks_fused_kernels_layernorm_rms_benchmarks_py.md) |
| ✅ | `benchmarks/kernels/bench_block_fp8_gemm.py` | 160 | Block FP8 GEMM benchmark | — | [→](./_files/benchmarks_kernels_bench_block_fp8_gemm_py.md) |
| ✅ | `benchmarks/kernels/bench_fp8_gemm.py` | 159 | FP8 GEMM benchmark | — | [→](./_files/benchmarks_kernels_bench_fp8_gemm_py.md) |
| ✅ | `benchmarks/kernels/bench_int8_gemm.py` | 169 | INT8 GEMM benchmark | — | [→](./_files/benchmarks_kernels_bench_int8_gemm_py.md) |
| ✅ | `benchmarks/kernels/bench_mxfp4_qutlass.py` | 191 | MXFP4 CUTLASS benchmark | — | [→](./_files/benchmarks_kernels_bench_mxfp4_qutlass_py.md) |
| ✅ | `benchmarks/kernels/bench_nvfp4_gemm.py` | 198 | NVFP4 GEMM benchmark | — | [→](./_files/benchmarks_kernels_bench_nvfp4_gemm_py.md) |
| ✅ | `benchmarks/kernels/bench_nvfp4_qutlass.py` | 207 | NVFP4 CUTLASS benchmark | — | [→](./_files/benchmarks_kernels_bench_nvfp4_qutlass_py.md) |
| ✅ | `benchmarks/kernels/bench_per_token_quant_fp8.py` | 270 | Per-token FP8 quantization | — | [→](./_files/benchmarks_kernels_bench_per_token_quant_fp8_py.md) |
| ✅ | `benchmarks/kernels/benchmark_2d_silu_mul_fp8_quant.py` | 244 | 2D SiLU-mul FP8 fusion | — | [→](./_files/benchmarks_kernels_benchmark_2d_silu_mul_fp8_quant_py.md) |
| ✅ | `benchmarks/kernels/benchmark_activation.py` | 105 | Activation function bench | — | [→](./_files/benchmarks_kernels_benchmark_activation_py.md) |
| ✅ | `benchmarks/kernels/benchmark_bitblas.py` | 244 | BitBLAS kernel benchmark | — | [→](./_files/benchmarks_kernels_benchmark_bitblas_py.md) |
| ✅ | `benchmarks/kernels/benchmark_cutlass_fp4_moe.py` | 504 | CUTLASS FP4 MoE benchmark | — | [→](./_files/benchmarks_kernels_benchmark_cutlass_fp4_moe_py.md) |
| ✅ | `benchmarks/kernels/benchmark_cutlass_moe_fp8.py` | 406 | CUTLASS FP8 MoE performance | — | [→](./_files/benchmarks_kernels_benchmark_cutlass_moe_fp8_py.md) |
| ✅ | `benchmarks/kernels/benchmark_device_communicators.py` | 508 | Inter-GPU communication bench | — | [→](./_files/benchmarks_kernels_benchmark_device_communicators_py.md) |
| ✅ | `benchmarks/kernels/benchmark_fused_collective.py` | 1129 | Fused all-reduce operations | — | [→](./_files/benchmarks_kernels_benchmark_fused_collective_py.md) |
| ✅ | `benchmarks/kernels/benchmark_grouped_gemm_cutlass.py` | 427 | CUTLASS grouped GEMM MoE | — | [→](./_files/benchmarks_kernels_benchmark_grouped_gemm_cutlass_py.md) |
| ✅ | `benchmarks/kernels/benchmark_layernorm.py` | 94 | LayerNorm performance | — | [→](./_files/benchmarks_kernels_benchmark_layernorm_py.md) |
| ✅ | `benchmarks/kernels/benchmark_lora.py` | 1488 | LoRA adapter operations | Workflow: vllm-project_vllm_Multi_LoRA_Inference | [→](./_files/benchmarks_kernels_benchmark_lora_py.md) |
| ✅ | `benchmarks/kernels/benchmark_machete.py` | 745 | Machete mixed-precision GEMM | — | [→](./_files/benchmarks_kernels_benchmark_machete_py.md) |
| ✅ | `benchmarks/kernels/benchmark_marlin.py` | 413 | Marlin quantized GEMM | — | [→](./_files/benchmarks_kernels_benchmark_marlin_py.md) |
| ✅ | `benchmarks/kernels/benchmark_mla_k_concat.py` | 150 | MLA tensor concatenation | — | [→](./_files/benchmarks_kernels_benchmark_mla_k_concat_py.md) |
| ✅ | `benchmarks/kernels/benchmark_moe.py` | 790 | MoE kernel tuning | — | [→](./_files/benchmarks_kernels_benchmark_moe_py.md) |
| ✅ | `benchmarks/kernels/benchmark_moe_align_block_size.py` | 87 | MoE block alignment | — | [→](./_files/benchmarks_kernels_benchmark_moe_align_block_size_py.md) |
| ✅ | `benchmarks/kernels/benchmark_moe_permute_unpermute.py` | 428 | MoE token routing | — | [→](./_files/benchmarks_kernels_benchmark_moe_permute_unpermute_py.md) |
| ✅ | `benchmarks/kernels/benchmark_mrope.py` | 322 | Multi-dimensional RoPE | — | [→](./_files/benchmarks_kernels_benchmark_mrope_py.md) |
| ✅ | `benchmarks/kernels/benchmark_paged_attention.py` | 250 | Paged attention kernels | Workflow: vllm-project_vllm_Basic_Offline_Inference | [→](./_files/benchmarks_kernels_benchmark_paged_attention_py.md) |
| ✅ | `benchmarks/kernels/benchmark_per_token_group_quant.py` | 159 | Dynamic FP8 quantization | — | [→](./_files/benchmarks_kernels_benchmark_per_token_group_quant_py.md) |
| ✅ | `benchmarks/kernels/benchmark_quant.py` | 109 | Scaled quantization ops | — | [→](./_files/benchmarks_kernels_benchmark_quant_py.md) |
| ✅ | `benchmarks/kernels/benchmark_reshape_and_cache.py` | 172 | KV cache storage | — | [→](./_files/benchmarks_kernels_benchmark_reshape_and_cache_py.md) |
| ✅ | `benchmarks/kernels/benchmark_reshape_and_cache_flash.py` | 210 | FlashInfer KV caching | — | [→](./_files/benchmarks_kernels_benchmark_reshape_and_cache_flash_py.md) |
| ✅ | `benchmarks/kernels/benchmark_rmsnorm.py` | 255 | RMSNorm optimization | — | [→](./_files/benchmarks_kernels_benchmark_rmsnorm_py.md) |
| ✅ | `benchmarks/kernels/benchmark_rope.py` | 106 | RoPE implementations compare | — | [→](./_files/benchmarks_kernels_benchmark_rope_py.md) |
| ✅ | `benchmarks/kernels/benchmark_shapes.py` | 94 | Model layer dimensions | — | [→](./_files/benchmarks_kernels_benchmark_shapes_py.md) |
| ✅ | `benchmarks/kernels/benchmark_silu_mul_fp8_quant.py` | 720 | Fused activation-quant | — | [→](./_files/benchmarks_kernels_benchmark_silu_mul_fp8_quant_py.md) |
| ✅ | `benchmarks/kernels/benchmark_trtllm_decode_attention.py` | 290 | TRT-LLM decode attention | — | [→](./_files/benchmarks_kernels_benchmark_trtllm_decode_attention_py.md) |
| ✅ | `benchmarks/kernels/benchmark_trtllm_prefill_attention.py` | 305 | TRT-LLM prefill attention | — | [→](./_files/benchmarks_kernels_benchmark_trtllm_prefill_attention_py.md) |
| ✅ | `benchmarks/kernels/benchmark_w8a8_block_fp8.py` | 415 | Block-wise FP8 GEMM | — | [→](./_files/benchmarks_kernels_benchmark_w8a8_block_fp8_py.md) |
| ✅ | `benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py` | 435 | DeepGEMM FP8 kernels | — | [→](./_files/benchmarks_kernels_deepgemm_benchmark_fp8_block_dense_gemm_py.md) |
| ✅ | `benchmarks/kernels/graph_machete_bench.py` | 64 | Benchmark visualization | — | [→](./_files/benchmarks_kernels_graph_machete_bench_py.md) |
| ✅ | `benchmarks/kernels/utils.py` | 214 | Shared benchmark utilities | — | [→](./_files/benchmarks_kernels_utils_py.md) |
| ✅ | `benchmarks/kernels/weight_shapes.py` | 104 | TP-aware model shapes | — | [→](./_files/benchmarks_kernels_weight_shapes_py.md) |
| ✅ | `benchmarks/multi_turn/bench_dataset.py` | 600 | Conversation dataset loading | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/benchmarks_multi_turn_bench_dataset_py.md) |
| ✅ | `benchmarks/multi_turn/bench_utils.py` | 28 | Multi-turn utilities | — | [→](./_files/benchmarks_multi_turn_bench_utils_py.md) |
| ✅ | `benchmarks/multi_turn/benchmark_serving_multi_turn.py` | 1666 | Conversational serving perf | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/benchmarks_multi_turn_benchmark_serving_multi_turn_py.md) |
| ✅ | `benchmarks/multi_turn/convert_sharegpt_to_openai.py` | 354 | Dataset format conversion | — | [→](./_files/benchmarks_multi_turn_convert_sharegpt_to_openai_py.md) |
| ✅ | `benchmarks/overheads/benchmark_hashing.py` | 64 | Prefix cache hashing | — | [→](./_files/benchmarks_overheads_benchmark_hashing_py.md) |
| ✅ | `cmake/hipify.py` | 80 | CUDA-to-ROCm converter | — | [→](./_files/cmake_hipify_py.md) |
| ✅ | `tools/generate_cmake_presets.py` | 180 | CMake preset generator | — | [→](./_files/tools_generate_cmake_presets_py.md) |
| ✅ | `tools/install_nixl_from_source_ubuntu.py` | 254 | NIXL/UCX build automation | — | [→](./_files/tools_install_nixl_from_source_ubuntu_py.md) |
| ✅ | `tools/report_build_time_ninja.py` | 325 | Ninja build time reporter | — | [→](./_files/tools_report_build_time_ninja_py.md) |
| ✅ | `vllm/__init__.py` | 107 | Main entry point API | Workflow: vllm-project_vllm_Basic_Offline_Inference | [→](./_files/vllm___init___py.md) |
| ✅ | `vllm/_aiter_ops.py` | 1339 | ROCm/AMD GPU optimizations | — | [→](./_files/vllm__aiter_ops_py.md) |
| ✅ | `vllm/_bc_linter.py` | 54 | API stability decorators | — | [→](./_files/vllm__bc_linter_py.md) |
| ✅ | `vllm/_custom_ops.py` | 3116 | High-performance GPU kernels | — | [→](./_files/vllm__custom_ops_py.md) |
| ✅ | `vllm/_ipex_ops.py` | 457 | Intel CPU optimizations | — | [→](./_files/vllm__ipex_ops_py.md) |
| ✅ | `vllm/beam_search.py` | 88 | Beam search decoding | — | [→](./_files/vllm_beam_search_py.md) |
| ✅ | `vllm/collect_env.py` | 857 | System diagnostics collector | — | [→](./_files/vllm_collect_env_py.md) |
| ✅ | `vllm/connections.py` | 189 | HTTP client utilities | — | [→](./_files/vllm_connections_py.md) |
| ✅ | `vllm/env_override.py` | 378 | PyTorch monkey patches | — | [→](./_files/vllm_env_override_py.md) |
| ✅ | `vllm/envs.py` | 1750 | Centralized config management | — | [→](./_files/vllm_envs_py.md) |
| ✅ | `vllm/forward_context.py` | 358 | Forward pass context mgmt | — | [→](./_files/vllm_forward_context_py.md) |
| ✅ | `vllm/logger.py` | 303 | Custom logging utilities | — | [→](./_files/vllm_logger_py.md) |
| ✅ | `vllm/logits_process.py` | 121 | Bad words filtering | Workflow: vllm-project_vllm_Structured_Output_Generation | [→](./_files/vllm_logits_process_py.md) |
| ✅ | `vllm/logprobs.py` | 206 | Token probability tracking | — | [→](./_files/vllm_logprobs_py.md) |
| ✅ | `vllm/outputs.py` | 345 | Generation output structures | Workflow: vllm-project_vllm_Basic_Offline_Inference | [→](./_files/vllm_outputs_py.md) |
| ✅ | `vllm/pooling_params.py` | 230 | Embedding/classification params | — | [→](./_files/vllm_pooling_params_py.md) |
| ✅ | `vllm/sampling_params.py` | 597 | Comprehensive sampling control | Workflow: vllm-project_vllm_Basic_Offline_Inference, vllm-project_vllm_Structured_Output_Generation | [→](./_files/vllm_sampling_params_py.md) |
| ✅ | `vllm/scalar_type.py` | 355 | Sub-byte type representation | — | [→](./_files/vllm_scalar_type_py.md) |
| ✅ | `vllm/scripts.py` | 17 | Deprecated CLI shim | — | [→](./_files/vllm_scripts_py.md) |
| ✅ | `vllm/sequence.py` | 98 | Request tracking structures | — | [→](./_files/vllm_sequence_py.md) |
| ✅ | `vllm/tasks.py` | 13 | Task type definitions | — | [→](./_files/vllm_tasks_py.md) |
| ✅ | `vllm/tracing.py` | 135 | OpenTelemetry integration | — | [→](./_files/vllm_tracing_py.md) |
| ✅ | `vllm/version.py` | 39 | Version management | — | [→](./_files/vllm_version_py.md) |

## 📝 Example Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `examples/offline_inference/async_llm_streaming.py` | 111 | Async streaming inference | Workflow: vllm-project_vllm_Basic_Offline_Inference | [→](./_files/examples_offline_inference_async_llm_streaming_py.md) |
| ✅ | `examples/offline_inference/audio_language.py` | 540 | Audio model inference | — | [→](./_files/examples_offline_inference_audio_language_py.md) |
| ✅ | `examples/offline_inference/automatic_prefix_caching.py` | 103 | Automatic prefix caching | Workflow: vllm-project_vllm_Basic_Offline_Inference | [→](./_files/examples_offline_inference_automatic_prefix_caching_py.md) |
| ✅ | `examples/offline_inference/batch_llm_inference.py` | 93 | Large-scale batch processing | Workflow: vllm-project_vllm_Basic_Offline_Inference | [→](./_files/examples_offline_inference_batch_llm_inference_py.md) |
| ✅ | `examples/offline_inference/chat_with_tools.py` | 147 | Function calling example | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/examples_offline_inference_chat_with_tools_py.md) |
| ✅ | `examples/offline_inference/context_extension.py` | 68 | Extended context windows | — | [→](./_files/examples_offline_inference_context_extension_py.md) |
| ✅ | `examples/offline_inference/data_parallel.py` | 268 | Native data parallelism | — | [→](./_files/examples_offline_inference_data_parallel_py.md) |
| ✅ | `examples/offline_inference/disaggregated_prefill.py` | 127 | Disaggregated architecture | — | [→](./_files/examples_offline_inference_disaggregated_prefill_py.md) |
| ✅ | `examples/offline_inference/encoder_decoder_multimodal.py` | 133 | Encoder-decoder models | Workflow: vllm-project_vllm_Vision_Language_Inference | [→](./_files/examples_offline_inference_encoder_decoder_multimodal_py.md) |
| ✅ | `examples/offline_inference/llm_engine_example.py` | 74 | Low-level engine API | Workflow: vllm-project_vllm_Basic_Offline_Inference | [→](./_files/examples_offline_inference_llm_engine_example_py.md) |
| ✅ | `examples/offline_inference/llm_engine_reset_kv.py` | 98 | KV cache management | — | [→](./_files/examples_offline_inference_llm_engine_reset_kv_py.md) |
| ✅ | `examples/offline_inference/load_sharded_state.py` | 94 | Fast weight loading | — | [→](./_files/examples_offline_inference_load_sharded_state_py.md) |
| ✅ | `examples/offline_inference/lora_with_quantization_inference.py` | 127 | LoRA on quantized models | Workflow: vllm-project_vllm_Multi_LoRA_Inference | [→](./_files/examples_offline_inference_lora_with_quantization_inference_py.md) |
| ✅ | `examples/offline_inference/metrics.py` | 50 | Performance monitoring | — | [→](./_files/examples_offline_inference_metrics_py.md) |
| ✅ | `examples/offline_inference/mistral-small.py` | 186 | Mistral-Small reference | Workflow: vllm-project_vllm_Basic_Offline_Inference | [→](./_files/examples_offline_inference_mistral-small_py.md) |
| ✅ | `examples/offline_inference/mlpspeculator.py` | 72 | MLP speculative decoding | Workflow: vllm-project_vllm_Speculative_Decoding | [→](./_files/examples_offline_inference_mlpspeculator_py.md) |
| ✅ | `examples/offline_inference/multilora_inference.py` | 106 | Multi-adapter serving | Workflow: vllm-project_vllm_Multi_LoRA_Inference | [→](./_files/examples_offline_inference_multilora_inference_py.md) |
| ✅ | `examples/offline_inference/prefix_caching.py` | 98 | Manual prefix caching | Workflow: vllm-project_vllm_Basic_Offline_Inference | [→](./_files/examples_offline_inference_prefix_caching_py.md) |
| ✅ | `examples/offline_inference/prompt_embed_inference.py` | 97 | Custom embeddings input | — | [→](./_files/examples_offline_inference_prompt_embed_inference_py.md) |
| ✅ | `examples/offline_inference/qwen_1m.py` | 70 | Million-token contexts | — | [→](./_files/examples_offline_inference_qwen_1m_py.md) |
| ✅ | `examples/offline_inference/reproducibility.py` | 46 | Deterministic outputs | Workflow: vllm-project_vllm_Basic_Offline_Inference | [→](./_files/examples_offline_inference_reproducibility_py.md) |
| ✅ | `examples/offline_inference/rlhf.py` | 147 | RLHF integration | — | [→](./_files/examples_offline_inference_rlhf_py.md) |
| ✅ | `examples/offline_inference/rlhf_colocate.py` | 251 | Memory-efficient RLHF | — | [→](./_files/examples_offline_inference_rlhf_colocate_py.md) |
| ✅ | `examples/offline_inference/rlhf_online_quant.py` | 162 | RLHF with quantization | — | [→](./_files/examples_offline_inference_rlhf_online_quant_py.md) |
| ✅ | `examples/offline_inference/rlhf_utils.py` | 168 | RLHF utilities | — | [→](./_files/examples_offline_inference_rlhf_utils_py.md) |
| ✅ | `examples/offline_inference/save_sharded_state.py` | 87 | Checkpoint optimization | — | [→](./_files/examples_offline_inference_save_sharded_state_py.md) |
| ✅ | `examples/offline_inference/simple_profiling.py` | 52 | Basic benchmarking | — | [→](./_files/examples_offline_inference_simple_profiling_py.md) |
| ✅ | `examples/offline_inference/skip_loading_weights_in_engine_init.py` | 53 | Custom weight loading | — | [→](./_files/examples_offline_inference_skip_loading_weights_in_engine_init_py.md) |
| ✅ | `examples/offline_inference/spec_decode.py` | 234 | Speculative decoding | Workflow: vllm-project_vllm_Speculative_Decoding | [→](./_files/examples_offline_inference_spec_decode_py.md) |
| ✅ | `examples/offline_inference/structured_outputs.py` | 113 | JSON schema constraints | Workflow: vllm-project_vllm_Structured_Output_Generation | [→](./_files/examples_offline_inference_structured_outputs_py.md) |
| ✅ | `examples/offline_inference/torchrun_dp_example.py` | 151 | Torchrun data parallelism | — | [→](./_files/examples_offline_inference_torchrun_dp_example_py.md) |
| ✅ | `examples/offline_inference/torchrun_example.py` | 76 | Torchrun tensor parallelism | — | [→](./_files/examples_offline_inference_torchrun_example_py.md) |
| ✅ | `examples/offline_inference/vision_language.py` | 2243 | VLM comprehensive reference | Workflow: vllm-project_vllm_Vision_Language_Inference | [→](./_files/examples_offline_inference_vision_language_py.md) |
| ✅ | `examples/offline_inference/vision_language_multi_image.py` | 1542 | Multi-image VLM reference | Workflow: vllm-project_vllm_Vision_Language_Inference | [→](./_files/examples_offline_inference_vision_language_multi_image_py.md) |
| ✅ | `examples/online_serving/api_client.py` | 93 | Legacy demo API client | — | [→](./_files/examples_online_serving_api_client_py.md) |
| ✅ | `examples/online_serving/gradio_openai_chatbot_webserver.py` | 112 | Gradio chatbot OpenAI API | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/examples_online_serving_gradio_openai_chatbot_webserver_py.md) |
| ✅ | `examples/online_serving/gradio_webserver.py` | 75 | Legacy Gradio interface | — | [→](./_files/examples_online_serving_gradio_webserver_py.md) |
| ✅ | `examples/online_serving/kv_events_subscriber.py` | 117 | KV cache monitoring ZMQ | — | [→](./_files/examples_online_serving_kv_events_subscriber_py.md) |
| ✅ | `examples/online_serving/multi_instance_data_parallel.py` | 87 | Multi-instance DP setup | — | [→](./_files/examples_online_serving_multi_instance_data_parallel_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_client.py` | 64 | Basic OpenAI chat client | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/examples_online_serving_openai_chat_completion_client_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_client_for_multimodal.py` | 353 | Multimodal inputs client | Workflow: vllm-project_vllm_Vision_Language_Inference, vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/examples_online_serving_openai_chat_completion_client_for_multimodal_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_client_with_tools.py` | 195 | Tool calling streaming | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/examples_online_serving_openai_chat_completion_client_with_tools_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_client_with_tools_required.py` | 130 | Required tool calling | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/examples_online_serving_openai_chat_completion_client_with_tools_required_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_client_with_tools_xlam.py` | 245 | xLAM-2 tool calling | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/examples_online_serving_openai_chat_completion_client_with_tools_xlam_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_client_with_tools_xlam_streaming.py` | 273 | xLAM streaming tools | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/examples_online_serving_openai_chat_completion_client_with_tools_xlam_streaming_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_tool_calls_with_reasoning.py` | 170 | Reasoning + tool calling | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/examples_online_serving_openai_chat_completion_tool_calls_with_reasoning_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_with_reasoning.py` | 65 | Basic reasoning models | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/examples_online_serving_openai_chat_completion_with_reasoning_py.md) |
| ✅ | `examples/online_serving/openai_chat_completion_with_reasoning_streaming.py` | 73 | Streaming reasoning output | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/examples_online_serving_openai_chat_completion_with_reasoning_streaming_py.md) |
| ✅ | `examples/online_serving/openai_completion_client.py` | 53 | Completions API non-chat | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/examples_online_serving_openai_completion_client_py.md) |
| ✅ | `examples/online_serving/openai_responses_client.py` | 44 | Responses API reasoning | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/examples_online_serving_openai_responses_client_py.md) |
| ✅ | `examples/online_serving/openai_responses_client_with_mcp_tools.py` | 184 | MCP protocol integration | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/examples_online_serving_openai_responses_client_with_mcp_tools_py.md) |
| ✅ | `examples/online_serving/openai_responses_client_with_tools.py` | 83 | Responses API with tools | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/examples_online_serving_openai_responses_client_with_tools_py.md) |
| ✅ | `examples/online_serving/openai_transcription_client.py` | 97 | Whisper audio transcription | — | [→](./_files/examples_online_serving_openai_transcription_client_py.md) |
| ✅ | `examples/online_serving/openai_translation_client.py` | 75 | Whisper audio translation | — | [→](./_files/examples_online_serving_openai_translation_client_py.md) |
| ✅ | `examples/online_serving/prompt_embed_inference_with_openai_client.py` | 79 | Pre-computed embeddings | — | [→](./_files/examples_online_serving_prompt_embed_inference_with_openai_client_py.md) |
| ✅ | `examples/online_serving/ray_serve_deepseek.py` | 55 | Ray Serve deployment | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/examples_online_serving_ray_serve_deepseek_py.md) |
| ✅ | `examples/online_serving/retrieval_augmented_generation_with_langchain.py` | 257 | RAG with LangChain | — | [→](./_files/examples_online_serving_retrieval_augmented_generation_with_langchain_py.md) |
| ✅ | `examples/online_serving/retrieval_augmented_generation_with_llamaindex.py` | 225 | RAG with LlamaIndex | — | [→](./_files/examples_online_serving_retrieval_augmented_generation_with_llamaindex_py.md) |
| ✅ | `examples/online_serving/streamlit_openai_chatbot_webserver.py` | 311 | Advanced Streamlit chatbot | Workflow: vllm-project_vllm_OpenAI_Compatible_Serving | [→](./_files/examples_online_serving_streamlit_openai_chatbot_webserver_py.md) |
| ✅ | `examples/online_serving/token_generation_client.py` | 49 | Direct token ID generation | — | [→](./_files/examples_online_serving_token_generation_client_py.md) |
| ✅ | `examples/online_serving/utils.py` | 26 | Shared utility functions | — | [→](./_files/examples_online_serving_utils_py.md) |
| ✅ | `examples/others/tensorize_vllm_model.py` | 392 | Fast GPU model loading | — | [→](./_files/examples_others_tensorize_vllm_model_py.md) |
| ✅ | `examples/pooling/classify/openai_classification_client.py` | 53 | Classification API client | — | [→](./_files/examples_pooling_classify_openai_classification_client_py.md) |
| ✅ | `examples/pooling/plugin/prithvi_geospatial_mae_client.py` | 56 | Geospatial TIFF client | — | [→](./_files/examples_pooling_plugin_prithvi_geospatial_mae_client_py.md) |
| ✅ | `examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py` | 58 | Offline geospatial example | — | [→](./_files/examples_pooling_plugin_prithvi_geospatial_mae_io_processor_py.md) |
| ✅ | `examples/pooling/plugin/prithvi_geospatial_mae_offline.py` | 419 | Satellite image segmentation | — | [→](./_files/examples_pooling_plugin_prithvi_geospatial_mae_offline_py.md) |
| ✅ | `examples/pooling/pooling/openai_pooling_client.py` | 63 | Dual-format pooling client | — | [→](./_files/examples_pooling_pooling_openai_pooling_client_py.md) |
| ✅ | `examples/pooling/pooling/vision_language_pooling.py` | 410 | Vision-language embeddings | Workflow: vllm-project_vllm_Vision_Language_Inference | [→](./_files/examples_pooling_pooling_vision_language_pooling_py.md) |
| ✅ | `examples/pooling/score/cohere_rerank_client.py` | 47 | Cohere-compatible reranking | — | [→](./_files/examples_pooling_score_cohere_rerank_client_py.md) |
| ✅ | `examples/pooling/score/convert_model_to_seq_cls.py` | 134 | CausalLM to SeqCls converter | — | [→](./_files/examples_pooling_score_convert_model_to_seq_cls_py.md) |
| ✅ | `examples/pooling/score/openai_cross_encoder_score.py` | 63 | Cross-encoder scoring | — | [→](./_files/examples_pooling_score_openai_cross_encoder_score_py.md) |
| ✅ | `examples/pooling/score/openai_reranker.py` | 42 | Raw HTTP rerank endpoint | — | [→](./_files/examples_pooling_score_openai_reranker_py.md) |
| ✅ | `examples/pooling/token_classify/ner.py` | 54 | Offline NER example | — | [→](./_files/examples_pooling_token_classify_ner_py.md) |
| ✅ | `examples/pooling/token_classify/ner_client.py` | 71 | Online NER via API | — | [→](./_files/examples_pooling_token_classify_ner_client_py.md) |
| ✅ | `examples/pooling/token_embed/jina_embeddings_v4.py` | 71 | Jina multimodal embeddings | — | [→](./_files/examples_pooling_token_embed_jina_embeddings_v4_py.md) |
| ✅ | `examples/pooling/token_embed/multi_vector_retrieval.py` | 56 | Multi-vector embeddings | — | [→](./_files/examples_pooling_token_embed_multi_vector_retrieval_py.md) |
| ✅ | `examples/pooling/token_embed/multi_vector_retrieval_client.py` | 54 | Multi-vector API client | — | [→](./_files/examples_pooling_token_embed_multi_vector_retrieval_client_py.md) |

## 🧪 Test Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `tests/__init__.py` | 0 | Package initialization marker | — | [→](./_files/tests___init___py.md) |
| ✅ | `tests/ci_envs.py` | 52 | CI environment management | — | [→](./_files/tests_ci_envs_py.md) |
| ✅ | `tests/conftest.py` | 1517 | Pytest config and fixtures | — | [→](./_files/tests_conftest_py.md) |
| ✅ | `tests/test_config.py` | 1052 | Configuration system tests | — | [→](./_files/tests_test_config_py.md) |
| ✅ | `tests/test_embedded_commit.py` | 11 | Version info validation | — | [→](./_files/tests_test_embedded_commit_py.md) |
| ✅ | `tests/test_envs.py` | 456 | Environment variable tests | — | [→](./_files/tests_test_envs_py.md) |
| ✅ | `tests/test_inputs.py` | 125 | Input parsing validation | — | [→](./_files/tests_test_inputs_py.md) |
| ✅ | `tests/test_logger.py` | 557 | Logging system tests | — | [→](./_files/tests_test_logger_py.md) |
| ✅ | `tests/test_logprobs.py` | 210 | Logprobs data structure tests | — | [→](./_files/tests_test_logprobs_py.md) |
| ✅ | `tests/test_outputs.py` | 21 | Output forward compatibility | — | [→](./_files/tests_test_outputs_py.md) |
| ✅ | `tests/test_pooling_params.py` | 156 | PoolingParams validation | — | [→](./_files/tests_test_pooling_params_py.md) |
| ✅ | `tests/test_regression.py` | 79 | Regression tests for bugs | — | [→](./_files/tests_test_regression_py.md) |
| ✅ | `tests/test_routing_simulator.py` | 199 | MoE routing simulator tests | — | [→](./_files/tests_test_routing_simulator_py.md) |
| ✅ | `tests/test_scalartype.py` | 43 | Scalar type validation | — | [→](./_files/tests_test_scalartype_py.md) |
| ✅ | `tests/test_seed_behavior.py` | 25 | Random seed reproducibility | — | [→](./_files/tests_test_seed_behavior_py.md) |
| ✅ | `tests/test_sequence.py` | 49 | IntermediateTensors tests | — | [→](./_files/tests_test_sequence_py.md) |
| ✅ | `tests/test_triton_utils.py` | 94 | Triton placeholder fallback | — | [→](./_files/tests_test_triton_utils_py.md) |
| ✅ | `tests/test_version.py` | 38 | Version utility tests | — | [→](./_files/tests_test_version_py.md) |
| ✅ | `tests/test_vllm_port.py` | 39 | VLLM_PORT parsing tests | — | [→](./_files/tests_test_vllm_port_py.md) |
| ✅ | `tests/utils.py` | 1312 | Shared test utilities | — | [→](./_files/tests_utils_py.md) |

## 📄 Other Files

| Status | File | Lines | Purpose | Coverage | Details |
|--------|------|-------|---------|----------|---------|
| ✅ | `.buildkite/check-wheel-size.py` | 53 | CI wheel size validation | — | [→](./_files/_buildkite_check-wheel-size_py.md) |
| ✅ | `csrc/cutlass_extensions/vllm_cutlass_library_extension.py` | 76 | CUTLASS custom data types | — | [→](./_files/csrc_cutlass_extensions_vllm_cutlass_library_extension_py.md) |
| ✅ | `csrc/moe/marlin_moe_wna16/generate_kernels.py` | 306 | Marlin MoE kernel codegen | — | [→](./_files/csrc_moe_marlin_moe_wna16_generate_kernels_py.md) |
| ✅ | `csrc/quantization/gptq_marlin/generate_kernels.py` | 316 | GPTQ-Marlin kernel codegen | — | [→](./_files/csrc_quantization_gptq_marlin_generate_kernels_py.md) |
| ✅ | `csrc/quantization/machete/generate.py` | 694 | Machete GEMM kernel codegen | — | [→](./_files/csrc_quantization_machete_generate_py.md) |
| ✅ | `setup.py` | 813 | Multi-platform build system | — | [→](./_files/setup_py.md) |
| ✅ | `use_existing_torch.py` | 18 | Torch dependency removal | — | [→](./_files/use_existing_torch_py.md) |

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
