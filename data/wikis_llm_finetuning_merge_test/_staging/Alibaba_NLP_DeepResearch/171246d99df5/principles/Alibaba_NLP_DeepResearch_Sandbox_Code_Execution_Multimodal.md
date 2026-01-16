# Principle: Sandbox_Code_Execution_Multimodal

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::File|sandbox_module.py|WebAgent/WebWatcher/infer/scripts_eval/mmrag_r1/code/sandbox_module.py]]
|-
! Domains
| [[domain::Code_Execution]], [[domain::Security]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

HTTP-based code sandbox for multimodal agent workflows.

=== Description ===

Sandbox Code Execution provides secure Python code execution capabilities for multimodal agents. When the vision-language model generates code (e.g., for calculations, data processing, or image analysis), it is executed in an isolated remote environment rather than on the host system.

Key security properties:

1. **Process Isolation** - Code runs in a separate container/process
2. **Network Isolation** - Limited or no network access from sandbox
3. **Resource Limits** - CPU, memory, and time constraints
4. **Stateless Execution** - Each execution starts fresh

The sandbox enables:
- Mathematical calculations based on visual data
- Data transformations and formatting
- Simple algorithms for problem-solving
- Safe experimentation with code generation

=== Usage ===

Use sandbox code execution when:
- The agent needs to perform calculations
- Processing or transforming extracted data
- Implementing algorithmic solutions
- Running any LLM-generated code safely

The sandbox is the code_interpreter tool in the multimodal agent's toolkit.

== Theoretical Basis ==

The sandbox execution pipeline:

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# HTTP-based sandbox execution
def run_code_in_sandbox(code: str, timeout: int = 50) -> Tuple[str, Optional[Dict]]:
    # Step 1: Prepare request
    endpoint = os.getenv("SANDBOX_FUSION_ENDPOINT")
    payload = {
        "code": code,
        "timeout": timeout
    }

    # Step 2: Send to sandbox service
    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=timeout + 10  # Extra time for network
        )
        result = response.json()
    except Exception as e:
        return f"Execution error: {str(e)}", None

    # Step 3: Extract output
    if "stdout" in result:
        output = result["stdout"]
    elif "stderr" in result:
        output = f"Error: {result['stderr']}"
    else:
        output = str(result)

    return output, result
</syntaxhighlight>

The sandbox service typically provides:
- '''Python runtime''' - Standard Python with common libraries
- '''stdout/stderr capture''' - All print output returned
- '''Timeout enforcement''' - Prevents infinite loops
- '''Error handling''' - Exceptions captured and returned

Security considerations:
- No file system persistence between calls
- No access to host environment variables (except allowed ones)
- No subprocess spawning or shell commands
- Import restrictions for dangerous modules

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_run_code_in_sandbox]]
