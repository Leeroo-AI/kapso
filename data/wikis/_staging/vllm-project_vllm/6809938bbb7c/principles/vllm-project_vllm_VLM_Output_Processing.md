{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
* [[source::Paper|Visual Instruction Tuning|https://arxiv.org/abs/2304.08485]]
|-
! Domains
| [[domain::Vision]], [[domain::NLP]], [[domain::Output_Processing]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The practice of extracting, validating, and utilizing text outputs generated from vision-language model inference.

=== Description ===

VLM Output Processing involves handling the text generated in response to multimodal inputs. Key considerations:

1. **Caption Extraction:** Getting the generated text description
2. **Quality Assessment:** Checking for complete, coherent responses
3. **Truncation Handling:** Managing responses cut off by max_tokens
4. **Confidence Evaluation:** Assessing response reliability
5. **Structured Parsing:** Extracting structured data when applicable

=== Usage ===

Process VLM outputs when:
- Building image captioning pipelines
- Creating visual QA systems
- Extracting structured information from images
- Quality checking VLM responses

== Theoretical Basis ==

'''Output Quality Indicators:'''

<syntaxhighlight lang="python">
def assess_vlm_output(output):
    """Assess quality of VLM output."""
    completion = output.outputs[0]

    quality = {
        "complete": completion.finish_reason == "stop",
        "length": len(completion.token_ids),
        "has_content": len(completion.text.strip()) > 0,
    }

    # Warn on potential issues
    if not quality["complete"]:
        print("Warning: Response may be truncated")

    if quality["length"] < 5:
        print("Warning: Very short response")

    return quality
</syntaxhighlight>

'''Common Output Patterns:'''

<syntaxhighlight lang="python">
# Pattern 1: Direct caption
# "A golden retriever playing in a park"

# Pattern 2: Descriptive paragraph
# "This image shows a golden retriever..."

# Pattern 3: Structured answer
# "Type: Dog\nBreed: Golden Retriever\nActivity: Playing"

# Pattern 4: Conversational
# "I can see a beautiful golden retriever..."
</syntaxhighlight>

'''Post-Processing Strategies:'''

<syntaxhighlight lang="python">
def post_process_vlm_output(text, task_type):
    """Clean and format VLM output based on task."""

    # Remove common artifacts
    text = text.strip()
    text = text.replace("ASSISTANT:", "").strip()

    if task_type == "caption":
        # First sentence only
        return text.split(".")[0] + "."

    elif task_type == "qa":
        # Full response
        return text

    elif task_type == "structured":
        # Try to parse as structured data
        try:
            return parse_structured(text)
        except:
            return text

    return text
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_RequestOutput_vlm]]
