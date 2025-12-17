# Phase 6b: Orphan Review Report

## Summary
- MANUAL_REVIEW files evaluated: 53
- Approved: 32
- Rejected: 21

## Decisions

| File | Decision | Reasoning |
|------|----------|-----------|
| `.buildkite/check-wheel-size.py` | REJECTED | CI tooling, no public API |
| `benchmarks/benchmark_block_pool.py` | REJECTED | Internal benchmark script |
| `benchmarks/benchmark_hash.py` | REJECTED | Internal benchmark, no API |
| `benchmarks/benchmark_long_document_qa_throughput.py` | APPROVED | User-facing benchmark with docs |
| `benchmarks/benchmark_prefix_block_hash.py` | REJECTED | Internal benchmark script |
| `benchmarks/benchmark_prefix_caching.py` | APPROVED | User-facing feature benchmark |
| `benchmarks/benchmark_prioritization.py` | APPROVED | Demonstrates priority feature |
| `benchmarks/benchmark_utils.py` | APPROVED | Public TimeCollector class |
| `benchmarks/cutlass_benchmarks/utils.py` | REJECTED | Internal utilities, no API |
| `benchmarks/cutlass_benchmarks/weight_shapes.py` | REJECTED | Data-only config, no API |
| `benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py` | APPROVED | Reference implementation |
| `benchmarks/disagg_benchmarks/rate_limiter.py` | REJECTED | Small internal utility |
| `benchmarks/disagg_benchmarks/request_queue.py` | REJECTED | Small internal utility |
| `benchmarks/disagg_benchmarks/round_robin_proxy.py` | REJECTED | Simple example, not unique |
| `benchmarks/disagg_benchmarks/visualize_benchmark_results.py` | REJECTED | Simple plotting script |
| `benchmarks/kernels/benchmark_layernorm.py` | REJECTED | Internal kernel benchmark |
| `benchmarks/kernels/benchmark_moe_align_block_size.py` | REJECTED | Internal kernel benchmark |
| `benchmarks/kernels/benchmark_shapes.py` | REJECTED | Data-only config file |
| `benchmarks/kernels/graph_machete_bench.py` | REJECTED | Internal visualization script |
| `benchmarks/multi_turn/bench_utils.py` | REJECTED | Trivial utility (logging) |
| `benchmarks/overheads/benchmark_hashing.py` | REJECTED | Internal profiling script |
| `cmake/hipify.py` | REJECTED | Build tooling, no API |
| `csrc/cutlass_extensions/vllm_cutlass_library_extension.py` | APPROVED | Public type definitions |
| `examples/offline_inference/rlhf.py` | APPROVED | User-facing RLHF example |
| `examples/offline_inference/rlhf_colocate.py` | APPROVED | User-facing RLHF colocate example |
| `examples/offline_inference/rlhf_online_quant.py` | APPROVED | User-facing RLHF+quant example |
| `examples/offline_inference/rlhf_utils.py` | APPROVED | Public WorkerExtension classes |
| `examples/online_serving/kv_events_subscriber.py` | APPROVED | User-facing KV events example |
| `examples/online_serving/utils.py` | REJECTED | Trivial helper function |
| `examples/pooling/classify/openai_classification_client.py` | APPROVED | User-facing API example |
| `examples/pooling/plugin/prithvi_geospatial_mae_client.py` | APPROVED | User-facing plugin example |
| `examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py` | APPROVED | User-facing plugin example |
| `examples/pooling/pooling/openai_pooling_client.py` | APPROVED | User-facing pooling example |
| `examples/pooling/score/cohere_rerank_client.py` | APPROVED | User-facing rerank example |
| `examples/pooling/score/convert_model_to_seq_cls.py` | APPROVED | User-facing conversion tool |
| `examples/pooling/score/openai_cross_encoder_score.py` | APPROVED | User-facing score example |
| `examples/pooling/score/openai_reranker.py` | APPROVED | User-facing rerank example |
| `examples/pooling/token_classify/ner.py` | APPROVED | User-facing NER example |
| `examples/pooling/token_classify/ner_client.py` | APPROVED | User-facing NER client |
| `examples/pooling/token_embed/jina_embeddings_v4.py` | APPROVED | User-facing embedding example |
| `examples/pooling/token_embed/multi_vector_retrieval.py` | APPROVED | User-facing retrieval example |
| `examples/pooling/token_embed/multi_vector_retrieval_client.py` | APPROVED | User-facing retrieval client |
| `tools/generate_cmake_presets.py` | REJECTED | Build tooling, no API |
| `tools/install_nixl_from_source_ubuntu.py` | REJECTED | Build/install script |
| `vllm/_bc_linter.py` | APPROVED | Public decorators for API |
| `vllm/beam_search.py` | APPROVED | Public BeamSearch classes |
| `vllm/connections.py` | APPROVED | Public HTTPConnection class |
| `vllm/logits_process.py` | APPROVED | Public LogitsProcessor API |
| `vllm/logprobs.py` | APPROVED | Public Logprob/FlatLogprobs |
| `vllm/pooling_params.py` | APPROVED | Public PoolingParams class |
| `vllm/sequence.py` | APPROVED | Public RequestMetrics, IntermediateTensors |
| `vllm/tracing.py` | APPROVED | Public tracing API |
| `vllm/version.py` | APPROVED | Public __version__ API |

## Notes

### Patterns Observed

1. **Examples files strongly approved**: Most files in the `examples/` directory were approved because they demonstrate user-facing functionality and serve as reference implementations for users.

2. **Internal benchmarks rejected**: Benchmark files testing internal components (like hash functions, block pools, kernel performance) were rejected as they don't expose public APIs and are primarily for internal performance testing.

3. **Build/CI tooling rejected**: Files in `.buildkite/`, `cmake/`, and `tools/` that support the build process were rejected as they don't provide user-facing functionality.

4. **Core vllm modules approved**: Files in the main `vllm/` directory that define public classes and APIs were approved, including data structures (Logprob, PoolingParams), utilities (HTTPConnection), and algorithms (BeamSearch).

### Borderline Decisions

- `benchmarks/benchmark_utils.py`: Approved because it exports a public `TimeCollector` class that could be useful for users running their own benchmarks.

- `vllm/_bc_linter.py`: Approved despite the `_` prefix in the filename because it exports public decorators (`bc_linter_skip`, `bc_linter_include`) intended for use by external code.

- `benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py`: Approved as a reference implementation for users implementing disaggregated serving.

### Category Breakdown

| Category | Approved | Rejected |
|----------|----------|----------|
| Benchmarks | 5 | 16 |
| Examples | 19 | 1 |
| Tools/Build | 1 | 4 |
| Core vllm | 9 | 0 |
