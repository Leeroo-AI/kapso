# API Request Processing

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|OpenAI Chat API|https://platform.openai.com/docs/api-reference/chat/create]]
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Serving]], [[domain::API]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Principle for executing inference requests through vLLM's OpenAI-compatible chat completions API endpoint.

=== Description ===

API Request Processing handles the server-side execution of inference requests:

1. **Request validation**: Check message format, parameters
2. **Template application**: Convert chat messages to model prompt
3. **Scheduling**: Add request to continuous batching queue
4. **Generation**: Execute token-by-token generation
5. **Response formatting**: Return OpenAI-compatible response

The API supports both synchronous (blocking) and streaming modes.

=== Usage ===

Execute API requests when:
* Building chat applications with vLLM backend
* Integrating vLLM into existing OpenAI-based systems
* Processing inference requests at scale
* Implementing conversational AI systems

== Theoretical Basis ==

'''Request Flow:'''
<syntaxhighlight lang="python">
# Abstract request processing flow
async def chat_completion(request):
    # 1. Validate and parse request
    messages = request.messages
    sampling_params = extract_sampling_params(request)

    # 2. Apply chat template
    prompt = tokenizer.apply_chat_template(messages)

    # 3. Submit to engine
    request_id = generate_request_id()
    await engine.add_request(request_id, prompt, sampling_params)

    # 4. Wait for completion (or stream)
    async for output in engine.generate(request_id):
        if request.stream:
            yield format_chunk(output)
        else:
            final_output = output

    # 5. Return formatted response
    return format_response(final_output)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_chat_completions_create]]
