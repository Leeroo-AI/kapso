# Environment: Sandbox_Execution_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Doc|SandboxFusion|https://github.com/bytedance/SandboxFusion]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Code_Execution]], [[domain::Security]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==
Sandboxed Python code execution environment using SandboxFusion for secure agent-generated code execution.

=== Description ===
DeepResearch agents can generate and execute Python code to perform calculations, data processing, and analysis. This code runs in a sandboxed environment provided by SandboxFusion to prevent security risks. The sandbox supports multiple endpoints for load balancing and fault tolerance.

=== Usage ===
Use this environment when the ReAct agent or multimodal agent needs to execute Python code. The sandbox is required for the PythonInterpreter tool and run_code_in_sandbox functionality. Without a configured sandbox, code execution features will fail.

== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
| SandboxFusion || Running instance || Deploy via Docker or Kubernetes
|-
| Network || HTTP access to sandbox || Default port 8080
|-
| Memory || 2GB+ per sandbox instance || Depends on code workload
|}

== Dependencies ==

=== System Packages ===
* `docker` (for SandboxFusion deployment)
* Network access to sandbox endpoints

=== Python Packages ===
* `requests` (HTTP client for sandbox API)

== Credentials ==
The following environment variables must be set:

* `SANDBOX_FUSION_ENDPOINT`: Comma-separated list of SandboxFusion HTTP endpoints (e.g., `http://host1:8080,http://host2:8080`)

== Quick Install ==
<syntaxhighlight lang="bash">
# Deploy SandboxFusion (see https://github.com/bytedance/SandboxFusion)
docker pull sandboxfusion/sandbox:latest
docker run -d -p 8080:8080 sandboxfusion/sandbox:latest

# Configure endpoint in .env
export SANDBOX_FUSION_ENDPOINT="http://localhost:8080"

# For multiple endpoints (load balancing):
export SANDBOX_FUSION_ENDPOINT="http://host1:8080,http://host2:8080,http://host3:8080"
</syntaxhighlight>

== Code Evidence ==

Sandbox endpoint configuration from `tool_python.py:25`:
<syntaxhighlight lang="python">
SANDBOX_FUSION_ENDPOINTS = os.environ['SANDBOX_FUSION_ENDPOINT'].split(',')
</syntaxhighlight>

Sandbox HTTP API call from `sandbox_module.py:4-36`:
<syntaxhighlight lang="python">
def run_code_in_sandbox(code):
    sandbox_url = os.environ.get('SANDBOX_URL', 'http://localhost:8080')

    try:
        response = requests.post(
            f"{sandbox_url}/run_code",
            json={"code": code},
            timeout=30
        )
        result = response.json()
        return result.get("output", "No output")
    except Exception as e:
        return f"Sandbox error: {str(e)}"
</syntaxhighlight>

Multiple endpoint support with random selection from `tool_python.py`:
<syntaxhighlight lang="python">
# Endpoints are split by comma and randomly selected for load balancing
endpoint = random.choice(SANDBOX_FUSION_ENDPOINTS)
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `KeyError: 'SANDBOX_FUSION_ENDPOINT'` || Environment variable not set || Set SANDBOX_FUSION_ENDPOINT in .env
|-
|| `Sandbox error: Connection refused` || Sandbox service not running || Start SandboxFusion container
|-
|| `Sandbox error: timeout` || Code execution timeout || Increase timeout or simplify code
|-
|| `[Python Interpreter Error]: Formatting error.` || Invalid code format || Check code syntax before execution
|}

== Compatibility Notes ==

* **Multiple Endpoints:** Supports comma-separated endpoints for distributed execution
* **Load Balancing:** Randomly selects from available endpoints
* **Timeout:** Default 30-second timeout for code execution
* **Security:** Code runs in isolated container with limited system access

== Related Pages ==
* [[required_by::Implementation:Alibaba_NLP_DeepResearch_PythonInterpreter_call]]
* [[required_by::Implementation:Alibaba_NLP_DeepResearch_run_code_in_sandbox]]
