# Implementation: llama_cli_validation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|llama.cpp|https://github.com/ggerganov/llama.cpp]]
|-
! Domains
| [[domain::Testing]], [[domain::GGUF]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

External tool documentation for validating GGUF exports using llama.cpp CLI.

=== Description ===

`llama-cli` (formerly `main`) is the llama.cpp command-line inference tool. Use it to verify GGUF exports before deployment.

=== Usage ===

Run after GGUF export to validate model functionality.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/ggerganov/llama.cpp llama.cpp]
* '''File:''' examples/main/main.cpp
* '''Binary:''' llama-cli (built from source)

=== Installation ===
<syntaxhighlight lang="bash">
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j

# Or with CUDA support
make -j LLAMA_CUDA=1
</syntaxhighlight>

== I/O Contract ==

=== CLI Arguments ===
{| class="wikitable"
|-
! Argument !! Description
|-
| -m, --model || Path to GGUF model file
|-
| -p, --prompt || Input prompt text
|-
| -n, --n-predict || Number of tokens to generate
|-
| --chat || Enable chat mode with template
|-
| -c, --ctx-size || Context size (default: 512)
|-
| -t, --threads || Number of CPU threads
|-
| --gpu-layers || Layers to offload to GPU
|}

== Usage Examples ==

=== Basic Generation Test ===
<syntaxhighlight lang="bash">
# Test text generation
llama-cli \
  --model ./model-Q4_K_M.gguf \
  --prompt "The capital of France is" \
  --n-predict 50

# Expected: Coherent continuation about Paris
</syntaxhighlight>

=== Chat Mode Test ===
<syntaxhighlight lang="bash">
# Test chat template
llama-cli \
  --model ./model-Q4_K_M.gguf \
  --chat \
  --prompt "Hello, how are you?"

# Should use model's chat template
</syntaxhighlight>

=== VLM Test ===
<syntaxhighlight lang="bash">
# Test vision-language model
llama-mtmd-cli \
  --model ./model-Q4_K_M.gguf \
  --mmproj ./mmproj-Q4_K_M.gguf

# Interactive mode:
# /image path/to/image.jpg
# What do you see in this image?
</syntaxhighlight>

=== Ollama Test ===
<syntaxhighlight lang="bash">
# Register with Ollama
ollama create my-model -f ./Modelfile

# Test with Ollama
ollama run my-model "What is 2+2?"
</syntaxhighlight>

=== Quick Validation Script ===
<syntaxhighlight lang="python">
import subprocess
import sys

def validate_gguf(model_path: str) -> bool:
    """Quick validation of GGUF export."""
    try:
        result = subprocess.run(
            [
                "llama-cli",
                "-m", model_path,
                "-p", "Hello",
                "-n", "10",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            print(f"✓ Model loaded successfully")
            print(f"Output: {result.stdout[:200]}...")
            return True
        else:
            print(f"✗ Error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("✗ llama-cli not found. Install llama.cpp first.")
        return False
    except subprocess.TimeoutExpired:
        print("✗ Timeout during inference")
        return False

if __name__ == "__main__":
    validate_gguf(sys.argv[1])
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_GGUF_Verification]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_llama_cpp_Environment]]

