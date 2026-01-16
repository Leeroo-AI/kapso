# Implementation: run_code_in_sandbox

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

Function that executes Python code in a remote HTTP-based sandbox and returns stdout/stderr output.

=== Description ===

The `run_code_in_sandbox()` function provides secure code execution by sending Python code to a remote sandbox service via HTTP POST. It handles the request/response cycle, extracts output from the response, and returns both formatted output and raw response data.

The implementation features:
- Configurable timeout (default 50 seconds)
- Endpoint configuration via SANDBOX_FUSION_ENDPOINT environment variable
- Error handling for network and execution failures
- Both stdout and stderr capture
- Raw response preservation for debugging

=== Usage ===

Use `run_code_in_sandbox()` when:
- Executing LLM-generated Python code safely
- Performing calculations in agent workflows
- Running any untrusted code that needs isolation
- Processing data transformations

The function is called by the code_interpreter tool in the agent pipeline.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' WebAgent/WebWatcher/infer/scripts_eval/mmrag_r1/code/sandbox_module.py
* '''Lines:''' 4-36

=== Signature ===
<syntaxhighlight lang="python">
def run_code_in_sandbox(code: str, timeout: int = 50) -> Tuple[str, Optional[Dict]]:
    """
    Execute Python code in a remote sandbox service.

    Args:
        code: str - Python code to execute
        timeout: int - Maximum execution time in seconds (default 50)

    Returns:
        Tuple[str, Optional[Dict]]:
            - str: stdout/stderr output or error message
            - Optional[Dict]: Raw response dictionary (None on error)

    Environment:
        SANDBOX_FUSION_ENDPOINT: URL of the sandbox service

    Note:
        - Code runs in isolated environment
        - No persistence between executions
        - Common Python libraries available (numpy, pandas, etc.)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from mmrag_r1.code.sandbox_module import run_code_in_sandbox

output, raw = run_code_in_sandbox("print(2 + 2)")
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| code || str || Yes || Python code to execute
|-
| timeout || int || No || Max execution time in seconds (default: 50)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| output || str || stdout/stderr text or error message
|-
| raw_response || Optional[Dict] || Full response from sandbox (None on failure)
|}

'''Raw Response Structure:'''
<syntaxhighlight lang="json">
{
    "stdout": "output text\n",
    "stderr": "",
    "exit_code": 0,
    "execution_time": 0.123
}
</syntaxhighlight>

== Usage Examples ==

=== Basic Code Execution ===
<syntaxhighlight lang="python">
from mmrag_r1.code.sandbox_module import run_code_in_sandbox

# Simple calculation
code = """
x = 42
y = 17
print(f"Sum: {x + y}")
print(f"Product: {x * y}")
"""

output, raw = run_code_in_sandbox(code)
print(output)
# Output:
# Sum: 59
# Product: 714
</syntaxhighlight>

=== Mathematical Computation ===
<syntaxhighlight lang="python">
from mmrag_r1.code.sandbox_module import run_code_in_sandbox

# Calculate based on extracted visual data
code = """
import math

# Dimensions from image analysis
width = 1920
height = 1080

# Calculate aspect ratio and diagonal
aspect_ratio = width / height
diagonal = math.sqrt(width**2 + height**2)

print(f"Aspect ratio: {aspect_ratio:.2f}")
print(f"Diagonal: {diagonal:.2f} pixels")
"""

output, raw = run_code_in_sandbox(code)
print(output)
</syntaxhighlight>

=== With Custom Timeout ===
<syntaxhighlight lang="python">
from mmrag_r1.code.sandbox_module import run_code_in_sandbox

# Longer computation needs more time
code = """
import time
# Simulate longer computation
total = sum(range(10000000))
print(f"Sum: {total}")
"""

output, raw = run_code_in_sandbox(code, timeout=120)
print(output)
</syntaxhighlight>

=== Error Handling ===
<syntaxhighlight lang="python">
from mmrag_r1.code.sandbox_module import run_code_in_sandbox

# Code with intentional error
code = """
x = 1 / 0  # Division by zero
"""

output, raw = run_code_in_sandbox(code)
print(output)
# Output contains: ZeroDivisionError: division by zero

# Check if execution succeeded
if raw and raw.get("exit_code") == 0:
    print("Execution successful")
else:
    print("Execution failed")
</syntaxhighlight>

=== Agent Integration ===
<syntaxhighlight lang="python">
from mmrag_r1.code.sandbox_module import run_code_in_sandbox, extract_code_from_response

# Extract code from LLM response
llm_response = """
To calculate the area, I'll use Python:
<code>
import math
radius = 5.5
area = math.pi * radius ** 2
print(f"Area: {area:.2f} square units")
</code>
"""

# Parse and execute
code = extract_code_from_response(llm_response)
if code:
    output, raw = run_code_in_sandbox(code)
    print(f"Result: {output}")
</syntaxhighlight>

=== Data Processing ===
<syntaxhighlight lang="python">
from mmrag_r1.code.sandbox_module import run_code_in_sandbox

# Process data extracted from image/webpage
code = """
data = [
    {"name": "Item A", "price": 29.99},
    {"name": "Item B", "price": 49.99},
    {"name": "Item C", "price": 19.99},
]

total = sum(item["price"] for item in data)
average = total / len(data)

print(f"Total: ${total:.2f}")
print(f"Average: ${average:.2f}")
"""

output, raw = run_code_in_sandbox(code)
print(output)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Sandbox_Code_Execution_Multimodal]]
