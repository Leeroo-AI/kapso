# Phase 6b: Orphan Review Report

## Summary
- MANUAL_REVIEW files evaluated: 74
- Approved: 48
- Rejected: 26

## Decisions

| File | Decision | Reasoning |
|------|----------|-----------|
| `.buildkite/check-wheel-size.py` | REJECTED | CI/build tooling, not user-facing |
| `benchmarks/benchmark_block_pool.py` | REJECTED | Internal benchmark, no public API |
| `benchmarks/benchmark_hash.py` | REJECTED | Internal benchmark, no public API |
| `benchmarks/benchmark_long_document_qa_throughput.py` | REJECTED | Internal benchmark, no public API |
| `benchmarks/benchmark_prefix_block_hash.py` | REJECTED | Internal benchmark, no public API |
| `benchmarks/benchmark_prefix_caching.py` | REJECTED | Internal benchmark, no public API |
| `benchmarks/benchmark_prioritization.py` | REJECTED | Internal benchmark, no public API |
| `benchmarks/benchmark_utils.py` | REJECTED | Internal helper, no public API |
| `benchmarks/cutlass_benchmarks/utils.py` | REJECTED | Internal benchmark helper |
| `benchmarks/cutlass_benchmarks/weight_shapes.py` | REJECTED | Internal data constants |
| `benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py` | REJECTED | Internal benchmark tooling |
| `benchmarks/disagg_benchmarks/rate_limiter.py` | REJECTED | Internal benchmark utility |
| `benchmarks/disagg_benchmarks/request_queue.py` | REJECTED | Internal benchmark utility |
| `benchmarks/disagg_benchmarks/round_robin_proxy.py` | REJECTED | Internal benchmark utility |
| `benchmarks/disagg_benchmarks/visualize_benchmark_results.py` | REJECTED | Internal benchmark visualization |
| `benchmarks/kernels/benchmark_layernorm.py` | REJECTED | Internal kernel benchmark |
| `benchmarks/kernels/benchmark_moe_align_block_size.py` | REJECTED | Internal kernel benchmark |
| `benchmarks/kernels/benchmark_shapes.py` | REJECTED | Internal data constants |
| `benchmarks/kernels/graph_machete_bench.py` | REJECTED | Internal benchmark visualization |
| `benchmarks/multi_turn/bench_utils.py` | REJECTED | Internal benchmark utility |
| `benchmarks/overheads/benchmark_hashing.py` | REJECTED | Internal benchmark, no public API |
| `cmake/hipify.py` | REJECTED | Build tooling, not user-facing |
| `csrc/cutlass_extensions/vllm_cutlass_library_extension.py` | REJECTED | Build/compile-time extension |
| `examples/offline_inference/context_extension.py` | APPROVED | User-facing example, public API |
| `examples/offline_inference/data_parallel.py` | APPROVED | User-facing example, public API |
| `examples/offline_inference/disaggregated_prefill.py` | APPROVED | User-facing example, public API |
| `examples/offline_inference/llm_engine_reset_kv.py` | APPROVED | User-facing example, public API |
| `examples/offline_inference/load_sharded_state.py` | APPROVED | User-facing example, public API |
| `examples/offline_inference/metrics.py` | APPROVED | User-facing example, public API |
| `examples/offline_inference/prompt_embed_inference.py` | APPROVED | User-facing example, public API |
| `examples/offline_inference/qwen_1m.py` | APPROVED | User-facing example, public API |
| `examples/offline_inference/rlhf.py` | APPROVED | User-facing example, public API |
| `examples/offline_inference/rlhf_colocate.py` | APPROVED | User-facing example, public API |
| `examples/offline_inference/rlhf_online_quant.py` | APPROVED | User-facing example, public API |
| `examples/offline_inference/rlhf_utils.py` | APPROVED | Reusable utility for user examples |
| `examples/offline_inference/save_sharded_state.py` | APPROVED | User-facing example, public API |
| `examples/offline_inference/simple_profiling.py` | APPROVED | User-facing example, public API |
| `examples/offline_inference/skip_loading_weights_in_engine_init.py` | APPROVED | User-facing example, public API |
| `examples/offline_inference/torchrun_dp_example.py` | APPROVED | User-facing example, public API |
| `examples/offline_inference/torchrun_example.py` | APPROVED | User-facing example, public API |
| `examples/online_serving/api_client.py` | APPROVED | User-facing example, public API |
| `examples/online_serving/gradio_webserver.py` | APPROVED | User-facing example, public API |
| `examples/online_serving/kv_events_subscriber.py` | APPROVED | User-facing example, public API |
| `examples/online_serving/multi_instance_data_parallel.py` | APPROVED | User-facing example, public API |
| `examples/online_serving/openai_transcription_client.py` | APPROVED | User-facing example, public API |
| `examples/online_serving/openai_translation_client.py` | APPROVED | User-facing example, public API |
| `examples/online_serving/prompt_embed_inference_with_openai_client.py` | APPROVED | User-facing example, public API |
| `examples/online_serving/retrieval_augmented_generation_with_langchain.py` | APPROVED | User-facing example, public API |
| `examples/online_serving/retrieval_augmented_generation_with_llamaindex.py` | APPROVED | User-facing example, public API |
| `examples/online_serving/token_generation_client.py` | APPROVED | User-facing example, public API |
| `examples/online_serving/utils.py` | REJECTED | Small helper, no distinct algorithm |
| `examples/pooling/classify/openai_classification_client.py` | APPROVED | User-facing example, public API |
| `examples/pooling/plugin/prithvi_geospatial_mae_client.py` | APPROVED | User-facing example, public API |
| `examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py` | APPROVED | User-facing example, public API |
| `examples/pooling/pooling/openai_pooling_client.py` | APPROVED | User-facing example, public API |
| `examples/pooling/score/cohere_rerank_client.py` | APPROVED | User-facing example, public API |
| `examples/pooling/score/convert_model_to_seq_cls.py` | APPROVED | User-facing utility, public API |
| `examples/pooling/score/openai_cross_encoder_score.py` | APPROVED | User-facing example, public API |
| `examples/pooling/score/openai_reranker.py` | APPROVED | User-facing example, public API |
| `examples/pooling/token_classify/ner.py` | APPROVED | User-facing example, public API |
| `examples/pooling/token_classify/ner_client.py` | APPROVED | User-facing example, public API |
| `examples/pooling/token_embed/jina_embeddings_v4.py` | APPROVED | User-facing example, public API |
| `examples/pooling/token_embed/multi_vector_retrieval.py` | APPROVED | User-facing example, public API |
| `examples/pooling/token_embed/multi_vector_retrieval_client.py` | APPROVED | User-facing example, public API |
| `tools/generate_cmake_presets.py` | REJECTED | Build tooling, not user-facing |
| `tools/install_nixl_from_source_ubuntu.py` | REJECTED | Build/install tooling |
| `vllm/_bc_linter.py` | REJECTED | Internal linting, no public API |
| `vllm/beam_search.py` | APPROVED | Public API, core algorithm |
| `vllm/connections.py` | APPROVED | Public API, user-importable |
| `vllm/logprobs.py` | APPROVED | Public API, core data structure |
| `vllm/pooling_params.py` | APPROVED | Public API, user-facing config |
| `vllm/sequence.py` | APPROVED | Public API, core data structures |
| `vllm/tracing.py` | APPROVED | Public API, user-facing feature |
| `vllm/version.py` | APPROVED | Public API, user-importable |

## Notes

### Approval Patterns
- **User-facing examples**: All files in `examples/` directory (except tiny utils) were approved as they demonstrate public vLLM APIs for users
- **Core modules**: Files in `vllm/` with public classes/functions that users might import were approved

### Rejection Patterns
- **Internal benchmarks**: 21 files in `benchmarks/` were rejected as internal performance measurement tools without public APIs
- **Build tooling**: 4 files for CI/build/install scripts were rejected as not user-facing
- **Internal utilities**: Small helper files with no distinct algorithms or public APIs

### Borderline Cases
- `examples/online_serving/utils.py` - Rejected despite being in examples, as it only contains a single 27-line helper function (`get_first_model`) that doesn't warrant its own wiki page
- `vllm/_bc_linter.py` - Rejected due to underscore prefix indicating internal-only usage
