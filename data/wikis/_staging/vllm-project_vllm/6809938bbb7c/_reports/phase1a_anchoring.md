# Phase 1a: Anchoring Report

## Summary
- Workflows created: 6
- Total steps documented: 30

## Workflows Created

| Workflow | Source Files | Steps | Rough APIs |
|----------|--------------|-------|------------|
| Basic_Offline_Inference | vllm/__init__.py, sampling_params.py, outputs.py, examples/offline_inference/*.py | 5 | LLM, SamplingParams, generate() |
| OpenAI_Compatible_Serving | examples/online_serving/openai_*.py, gradio_*.py, streamlit_*.py | 5 | vllm serve, OpenAI client, chat.completions |
| Multi_LoRA_Inference | examples/offline_inference/multilora_inference.py, lora_with_quantization_inference.py | 5 | LLMEngine, LoRARequest, enable_lora |
| Vision_Language_Inference | examples/offline_inference/vision_language.py, vision_language_multi_image.py | 5 | LLM, EngineArgs, limit_mm_per_prompt |
| Speculative_Decoding | examples/offline_inference/spec_decode.py, mlpspeculator.py | 5 | LLM, speculative_config, EAGLE/ngram |
| Structured_Output_Generation | examples/offline_inference/structured_outputs.py, vllm/sampling_params.py | 5 | StructuredOutputsParams, SamplingParams |

## Coverage Summary
- Source files covered: ~40 files mapped to workflows
- Example files documented: 35+ examples covered by workflows

## Source Files Identified Per Workflow

### vllm-project_vllm_Basic_Offline_Inference
- `vllm/__init__.py` - Main entry point API
- `vllm/sampling_params.py` - Comprehensive sampling control (597 lines)
- `vllm/outputs.py` - Generation output structures (345 lines)
- `examples/offline_inference/batch_llm_inference.py` - Large-scale batch processing with Ray Data
- `examples/offline_inference/automatic_prefix_caching.py` - Automatic prefix caching demo
- `examples/offline_inference/llm_engine_example.py` - Low-level engine API
- `examples/offline_inference/prefix_caching.py` - Manual prefix caching
- `examples/offline_inference/reproducibility.py` - Deterministic outputs
- `examples/offline_inference/async_llm_streaming.py` - Async streaming inference

### vllm-project_vllm_OpenAI_Compatible_Serving
- `examples/online_serving/openai_chat_completion_client.py` - Basic OpenAI chat client (64 lines)
- `examples/online_serving/openai_chat_completion_client_with_tools.py` - Tool calling streaming (195 lines)
- `examples/online_serving/openai_chat_completion_client_for_multimodal.py` - Multimodal inputs (353 lines)
- `examples/online_serving/gradio_openai_chatbot_webserver.py` - Gradio chatbot (112 lines)
- `examples/online_serving/streamlit_openai_chatbot_webserver.py` - Streamlit chatbot (311 lines)
- `examples/online_serving/ray_serve_deepseek.py` - Ray Serve deployment
- `benchmarks/multi_turn/benchmark_serving_multi_turn.py` - Multi-turn serving benchmark (1666 lines)

### vllm-project_vllm_Multi_LoRA_Inference
- `examples/offline_inference/multilora_inference.py` - Multi-adapter serving (106 lines)
- `examples/offline_inference/lora_with_quantization_inference.py` - LoRA on quantized models (127 lines)
- `benchmarks/kernels/benchmark_lora.py` - LoRA adapter operations benchmark (1488 lines)

### vllm-project_vllm_Vision_Language_Inference
- `examples/offline_inference/vision_language.py` - VLM comprehensive reference (2243 lines, 60+ models)
- `examples/offline_inference/vision_language_multi_image.py` - Multi-image VLM reference (1542 lines)
- `examples/offline_inference/encoder_decoder_multimodal.py` - Encoder-decoder models (133 lines)
- `examples/pooling/pooling/vision_language_pooling.py` - Vision-language embeddings (410 lines)
- `examples/online_serving/openai_chat_completion_client_for_multimodal.py` - Multimodal API client

### vllm-project_vllm_Speculative_Decoding
- `examples/offline_inference/spec_decode.py` - Speculative decoding example (234 lines)
- `examples/offline_inference/mlpspeculator.py` - MLP speculative decoding (72 lines)
- `benchmarks/benchmark_ngram_proposer.py` - N-gram speculative decoding benchmark (215 lines)

### vllm-project_vllm_Structured_Output_Generation
- `examples/offline_inference/structured_outputs.py` - JSON schema constraints example (113 lines)
- `vllm/sampling_params.py` - StructuredOutputsParams definition (lines 33-99)
- `benchmarks/benchmark_serving_structured_output.py` - Structured output constraints benchmark (1040 lines)

## Notes for Phase 1b (Enrichment)

### Files that need line-by-line tracing
1. `vllm/entrypoints/llm.py` - LLM class and generate() method (not in sample, but referenced)
2. `vllm/engine/llm_engine.py` - LLMEngine core logic (not in sample)
3. `vllm/lora/request.py` - LoRARequest structure (not in sample)
4. `vllm/config.py` - speculative_config handling (not in sample)

### External APIs to document
1. **OpenAI Python client** - Wrapper documentation for `openai.OpenAI`, `chat.completions.create()`
2. **Ray Data** - `ray.data.llm.build_llm_processor`, `vLLMEngineProcessorConfig`
3. **HuggingFace Hub** - `snapshot_download` for LoRA adapters
4. **PIL/Pillow** - Image handling for VLM

### Any unclear mappings
1. Server-side implementation of OpenAI-compatible API (vllm/entrypoints/openai/) - not in sampled files
2. Scheduler internals for LoRA adapter management - referenced but not detailed
3. Multimodal preprocessing pipeline - vllm/multimodal/ not fully visible

## Workflow Principle Mappings (30 total)

### Basic_Offline_Inference (5 Principles)
- ⬜ Principle:vllm-project_vllm_LLM_Class_Initialization
- ⬜ Principle:vllm-project_vllm_Sampling_Parameters
- ⬜ Principle:vllm-project_vllm_Prompt_Formatting
- ⬜ Principle:vllm-project_vllm_Batch_Generation
- ⬜ Principle:vllm-project_vllm_Output_Processing

### OpenAI_Compatible_Serving (5 Principles)
- ⬜ Principle:vllm-project_vllm_Server_Configuration
- ⬜ Principle:vllm-project_vllm_Server_Launch
- ⬜ Principle:vllm-project_vllm_OpenAI_Client_Setup
- ⬜ Principle:vllm-project_vllm_Chat_Completion_API
- ⬜ Principle:vllm-project_vllm_Response_Handling

### Multi_LoRA_Inference (5 Principles)
- ⬜ Principle:vllm-project_vllm_LoRA_Engine_Configuration
- ⬜ Principle:vllm-project_vllm_LoRA_Adapter_Registration
- ⬜ Principle:vllm-project_vllm_LoRA_Request_Submission
- ⬜ Principle:vllm-project_vllm_LoRA_Scheduling
- ⬜ Principle:vllm-project_vllm_LoRA_Output_Processing

### Vision_Language_Inference (5 Principles)
- ⬜ Principle:vllm-project_vllm_VLM_Model_Configuration
- ⬜ Principle:vllm-project_vllm_Image_Input_Preparation
- ⬜ Principle:vllm-project_vllm_VLM_Prompt_Construction
- ⬜ Principle:vllm-project_vllm_Multimodal_Generation
- ⬜ Principle:vllm-project_vllm_VLM_Output_Processing

### Speculative_Decoding (5 Principles)
- ⬜ Principle:vllm-project_vllm_Speculative_Method_Selection
- ⬜ Principle:vllm-project_vllm_Speculative_Configuration
- ⬜ Principle:vllm-project_vllm_Speculative_Engine_Init
- ⬜ Principle:vllm-project_vllm_Speculative_Generation
- ⬜ Principle:vllm-project_vllm_Speculation_Metrics

### Structured_Output_Generation (5 Principles)
- ⬜ Principle:vllm-project_vllm_Constraint_Definition
- ⬜ Principle:vllm-project_vllm_StructuredOutputsParams
- ⬜ Principle:vllm-project_vllm_Structured_SamplingParams
- ⬜ Principle:vllm-project_vllm_Constrained_Generation
- ⬜ Principle:vllm-project_vllm_Structured_Output_Parsing

## Files Created

1. `workflows/vllm-project_vllm_Basic_Offline_Inference.md`
2. `workflows/vllm-project_vllm_OpenAI_Compatible_Serving.md`
3. `workflows/vllm-project_vllm_Multi_LoRA_Inference.md`
4. `workflows/vllm-project_vllm_Vision_Language_Inference.md`
5. `workflows/vllm-project_vllm_Speculative_Decoding.md`
6. `workflows/vllm-project_vllm_Structured_Output_Generation.md`
7. `_WorkflowIndex.md` (updated)
8. `_RepoMap_vllm-project_vllm.md` (coverage column updated)
