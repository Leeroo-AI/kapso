{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Embeddings]], [[domain::Custom Input]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
This example demonstrates how to generate prompt embeddings externally and pass them to vLLM for inference.

=== Description ===
The prompt embed inference example shows how to bypass vLLM's tokenization and embedding layers by providing pre-computed prompt embeddings directly. This enables advanced use cases like custom embedding modifications, multi-modal inputs, or integration with external embedding models. The example uses Hugging Face Transformers to generate embeddings from the same model, but you could compute embeddings from different models, apply transformations, or inject custom representations. This pattern is particularly useful for research, prompt engineering, or building hybrid systems.

=== Usage ===
Use this pattern when you need to modify embeddings before generation, integrate with external embedding models, implement custom prompt engineering techniques, process multi-modal inputs, or experiment with embedding-space interventions. The model must support enable_prompt_embeds=True.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/prompt_embed_inference.py examples/offline_inference/prompt_embed_inference.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
python examples/offline_inference/prompt_embed_inference.py
</syntaxhighlight>

Note: This example uses meta-llama/Llama-3.2-1B-Instruct, which is gated on Hugging Face. You must request access before running.

== Key Concepts ==

=== Enabling Prompt Embeddings ===
The LLM must be configured to accept embeddings as input:
* '''enable_prompt_embeds=True''': Allow embedding input instead of text
* Bypasses tokenizer and embedding layer
* Input must be torch.Tensor with shape [seq_len, hidden_dim]
* Compatible with standard sampling parameters

=== External Embedding Generation ===
The example shows embedding generation via Transformers:
* Load the same model with AutoModelForCausalLM
* Extract embedding layer: model.get_input_embeddings()
* Tokenize text: tokenizer.apply_chat_template()
* Compute embeddings: embedding_layer(token_ids)
* Results in tensor ready for vLLM input

=== Input Format ===
Embeddings are passed via dictionary format:
* Key: "prompt_embeds"
* Value: torch.Tensor of shape [seq_len, hidden_dim]
* Single request: {"prompt_embeds": tensor}
* Batch requests: [{"prompt_embeds": tensor1}, {"prompt_embeds": tensor2}, ...]

== Usage Examples ==

=== Initialization ===
<syntaxhighlight lang="python">
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

def init_tokenizer_and_llm(model_name: str):
    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load Transformers model for embedding generation
    transformers_model = AutoModelForCausalLM.from_pretrained(model_name)
    embedding_layer = transformers_model.get_input_embeddings()

    # Create vLLM instance with embedding support
    llm = LLM(model=model_name, enable_prompt_embeds=True)

    return tokenizer, embedding_layer, llm

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer, embedding_layer, llm = init_tokenizer_and_llm(model_name)
</syntaxhighlight>

=== Generating Prompt Embeddings ===
<syntaxhighlight lang="python">
def get_prompt_embeds(
    chat: list[dict[str, str]],
    tokenizer,
    embedding_layer: torch.nn.Module,
):
    # Tokenize using chat template
    token_ids = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # Generate embeddings
    prompt_embeds = embedding_layer(token_ids).squeeze(0)

    return prompt_embeds

chat = [{"role": "user", "content": "Please tell me about the capital of France."}]
prompt_embeds = get_prompt_embeds(chat, tokenizer, embedding_layer)

print(f"Embedding shape: {prompt_embeds.shape}")
# Output: torch.Size([seq_len, hidden_dim])
</syntaxhighlight>

=== Single Prompt Inference ===
<syntaxhighlight lang="python">
def single_prompt_inference(llm, tokenizer, embedding_layer):
    chat = [{"role": "user", "content": "Please tell me about the capital of France."}]
    prompt_embeds = get_prompt_embeds(chat, tokenizer, embedding_layer)

    # Pass embeddings to vLLM
    outputs = llm.generate(
        {
            "prompt_embeds": prompt_embeds,
        }
    )

    print("\n[Single Inference Output]")
    print("-" * 30)
    for o in outputs:
        print(o.outputs[0].text)
    print("-" * 30)

single_prompt_inference(llm, tokenizer, embedding_layer)
</syntaxhighlight>

=== Batch Prompt Inference ===
<syntaxhighlight lang="python">
def batch_prompt_inference(llm, tokenizer, embedding_layer):
    chats = [
        [{"role": "user", "content": "Please tell me about the capital of France."}],
        [{"role": "user", "content": "When is the day longest during the year?"}],
        [{"role": "user", "content": "Where is bigger, the moon or the sun?"}],
    ]

    # Generate embeddings for all prompts
    prompt_embeds_list = [
        get_prompt_embeds(chat, tokenizer, embedding_layer)
        for chat in chats
    ]

    # Batch inference with embeddings
    outputs = llm.generate([
        {"prompt_embeds": embeds}
        for embeds in prompt_embeds_list
    ])

    print("\n[Batch Inference Outputs]")
    print("-" * 30)
    for i, o in enumerate(outputs):
        print(f"Q{i + 1}: {chats[i][0]['content']}")
        print(f"A{i + 1}: {o.outputs[0].text}\n")
    print("-" * 30)

batch_prompt_inference(llm, tokenizer, embedding_layer)
</syntaxhighlight>

=== Custom Embedding Modifications ===
<syntaxhighlight lang="python">
def get_modified_embeds(chat, tokenizer, embedding_layer, scale_factor=1.5):
    """Generate embeddings and apply custom modifications."""
    prompt_embeds = get_prompt_embeds(chat, tokenizer, embedding_layer)

    # Example: Scale embeddings to increase/decrease attention
    modified_embeds = prompt_embeds * scale_factor

    # Example: Add noise for exploration
    # noise = torch.randn_like(prompt_embeds) * 0.01
    # modified_embeds = prompt_embeds + noise

    return modified_embeds

# Use modified embeddings
chat = [{"role": "user", "content": "Tell me a creative story."}]
modified_embeds = get_modified_embeds(chat, tokenizer, embedding_layer, scale_factor=1.2)

outputs = llm.generate({"prompt_embeds": modified_embeds})
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
