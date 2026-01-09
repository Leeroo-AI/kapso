# Principle: GGUF_Verification

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|llama.cpp CLI|https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md]]
|-
! Domains
| [[domain::Testing]], [[domain::GGUF]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Verification of GGUF model exports by running inference with llama.cpp CLI tools.

=== Description ===

GGUF Verification ensures exported models work correctly:

1. **Load Test**: Verify GGUF file loads without errors
2. **Generation Test**: Run sample prompts and check output quality
3. **Tokenizer Test**: Verify tokenization matches expected behavior
4. **Performance Test**: Measure inference speed (optional)

=== Usage ===

Run after GGUF export before deployment to catch conversion errors early.

== Theoretical Basis ==

=== Verification Checklist ===

| Test | Command | Expected Result |
|------|---------|-----------------|
| Load | `llama-cli -m model.gguf -p "" -n 1` | No errors |
| Generate | `llama-cli -m model.gguf -p "Hello" -n 50` | Coherent output |
| Template | `llama-cli -m model.gguf --chat` | Correct formatting |

=== Common Issues ===

* **Tokenizer mismatch**: BOS/EOS token handling differs
* **Architecture unsupported**: Some architectures not in llama.cpp
* **Quantization errors**: Rare numerical issues in low-bit quants

=== VLM Verification ===

For vision-language models:
```bash
llama-mtmd-cli -m model.gguf --mmproj mmproj.gguf
/image test.jpg
Describe this image.
```

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_llama_cli_validation]]

