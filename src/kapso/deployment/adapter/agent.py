# Adapter Agent
#
# Uses coding agents to adapt a solution for a specific deployment target.
# Loads instructions from the strategies/ registry.
#
# Usage:
#     adapter = AdapterAgent(coding_agent_type="claude_code", model="...")
#     result = adapter.adapt(solution, setting, env_vars={"API_KEY": "..."})

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kapso.deployment.base import DeploymentSetting, AdaptationResult
from kapso.deployment.strategies import StrategyRegistry
from kapso.deployment.strategies.base import deployment_context
from kapso.deployment.adapter.validator import AdaptationValidator
from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.execution.solution import SolutionResult

# Left behind when the solution is copied into the adapted workspace: version
# control and caches are never part of a deployment, and .git alone can be
# larger than the code. Data and model files are copied — the deployed code
# may need them.
ADAPTED_COPY_IGNORE = (
    ".git", "__pycache__", "*.pyc", "*.pyo", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".venv", "venv", "node_modules", ".DS_Store",
)

ENV_FILE = ".env"


def write_env_file(workspace: str, env_vars: Optional[Dict[str, str]]) -> Optional[str]:
    """Write the caller's environment variables to `.env` in the workspace.

    Existing lines are kept and the caller's keys win, so a solution that
    ships its own `.env` keeps it. Returns the file's path, or None when
    there was nothing to write. Owner-only permissions: the file holds
    secrets, and the workspace is a sibling of the solution the user owns.
    """
    if not env_vars:
        return None
    path = Path(workspace) / ENV_FILE
    kept: List[str] = []
    if path.exists():
        for line in path.read_text().splitlines():
            key = line.split("=", 1)[0].strip()
            if not line.strip() or line.lstrip().startswith("#") or key not in env_vars:
                kept.append(line)
    lines = kept + [f"{key}={value}" for key, value in env_vars.items()]
    path.write_text("\n".join(lines) + "\n")
    os.chmod(path, 0o600)
    return str(path)


class AdapterAgent:
    """
    Uses coding agents to adapt a solution for deployment.
    
    Flow:
    1. Load instructions from strategies registry
    2. Create coding agent (Claude Code, Codex, etc.)
    3. Generate adaptation prompt
    4. Execute coding agent to transform the code and deploy
    5. Validate the adaptation; on failure, re-run the agent with the
       validator's finding, up to max_retries times
    6. Return result with run interface and adapted path
    """
    
    def __init__(
        self,
        coding_agent_type: str,
        model: str,
        fallback_agent_type: str = "gemini",
        max_retries: int = 2,
    ):
        """
        Initialize the adapter agent.
        
        Args:
            coding_agent_type: Primary coding agent (claude_code, codex, ...)
            model: LLM model for the primary coding agent
            fallback_agent_type: Secondary coding agent if primary fails
            max_retries: Re-runs allowed after a failed validation
        """
        self.coding_agent_type = coding_agent_type
        self.model = model
        self.fallback_agent_type = fallback_agent_type
        self.max_retries = max_retries
        self.registry = StrategyRegistry.get()
        self.validator = AdaptationValidator()
        
        # Path to the adaptation prompt template
        self.adaptation_prompt_path = Path(__file__).parent / "adaptation_prompt.txt"
    
    def _create_adapted_workspace(self, original_path: str, strategy: str) -> str:
        """
        Create a copy of the solution for adaptation.
        
        The original solution is never modified. All adaptation happens
        in the new workspace at {original_path}_adapted_{strategy}.
        
        Args:
            original_path: Path to the original solution
            strategy: Deployment strategy name (used in directory name)
            
        Returns:
            Path to the adapted workspace
        """
        original = Path(original_path)
        adapted = Path(f"{original_path}_adapted_{strategy}")
        
        # Remove existing adapted workspace if it exists
        if adapted.exists():
            shutil.rmtree(adapted)
        
        # Copy the original solution to the adapted workspace
        shutil.copytree(
            original, adapted,
            ignore=shutil.ignore_patterns(*ADAPTED_COPY_IGNORE),
            symlinks=True,
        )
        
        return str(adapted)
    
    def adapt(
        self,
        solution: SolutionResult,
        setting: DeploymentSetting,
        allowed_strategies: Optional[List[str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> AdaptationResult:
        """
        Adapt a solution for the specified deployment setting.
        
        Creates a copy of the solution at {code_path}_adapted_{strategy} and
        performs adaptation there. The original solution is never modified.
        
        Args:
            solution: The SolutionResult from Kapso.evolve()
            setting: Selected deployment configuration
            allowed_strategies: Optional list of allowed strategies
            env_vars: Variables the deployed software needs at runtime;
                written to `.env` in the workspace and named in the prompt
            
        Returns:
            AdaptationResult with run interface and adapted path
        """
        print(f"[Adapter] Adapting for {setting.strategy} deployment")
        
        # Extract from solution
        original_path = solution.code_path
        goal = solution.goal
        
        # Validate strategy is available
        available = self.registry.list_strategies(allowed=allowed_strategies)
        if setting.strategy not in available:
            return AdaptationResult(
                success=False,
                adapted_path=original_path,
                run_interface={},
                error=f"Strategy '{setting.strategy}' not available. Options: {available}",
            )
        
        # 1. Create adapted workspace (copy original, don't modify it)
        adapted_path = self._create_adapted_workspace(original_path, setting.strategy)
        print(f"[Adapter] Created adapted workspace: {adapted_path}")
        
        # 2. The caller's environment variables, and the names/port this
        #    solution deploys under
        env_file = write_env_file(adapted_path, env_vars)
        context = self._build_context(original_path, env_vars, env_file)
        
        # 3. Load target-specific instructions from registry
        target_instructions = self.registry.get_adapter_instruction(setting.strategy)
        
        # 4. Run the coding agent; validate; retry with the finding
        previous_error: Optional[str] = None
        files_changed: List[str] = []
        agent_output = ""
        for attempt in range(1, self.max_retries + 2):
            prompt = self._build_adaptation_prompt(
                goal=goal,
                setting=setting,
                target_instructions=target_instructions,
                context=context,
                previous_error=previous_error,
            )
            outcome = self._run_with_fallback(adapted_path, prompt)
            if outcome is None:
                return AdaptationResult(
                    success=False,
                    adapted_path=adapted_path,
                    run_interface={},
                    error="Both primary and fallback agents failed",
                )
            files_changed, agent_output = outcome
            print(f"[Adapter] Files changed: {len(files_changed)}")
            
            validation = self.validator.validate(adapted_path, setting)
            if validation.success:
                break
            previous_error = validation.error or "validation failed"
            print(f"[Adapter] Validation failed (attempt {attempt}): {previous_error}")
            if attempt > self.max_retries:
                return AdaptationResult(
                    success=False,
                    adapted_path=adapted_path,
                    run_interface={},
                    files_changed=files_changed,
                    error=f"Adaptation failed validation after {attempt} attempt(s): {previous_error}",
                )
        
        # 5. Read what the agent reported
        run_interface_from_agent = self._extract_run_interface_from_output(agent_output)
        endpoint = self._extract_endpoint_from_output(agent_output)
        if run_interface_from_agent:
            print(f"[Adapter] Run interface from agent: {run_interface_from_agent}")
        if endpoint:
            print(f"[Adapter] Endpoint: {endpoint}")
        
        # 6. Build run interface (how to call the deployed software)
        run_interface = self._build_run_interface(
            strategy=setting.strategy,
            endpoint=endpoint,
            agent_run_interface=run_interface_from_agent,
            context=context,
        )
        
        print(f"[Adapter] Complete: {adapted_path}")
        
        return AdaptationResult(
            success=True,
            adapted_path=adapted_path,
            run_interface=run_interface,
            files_changed=files_changed,
        )
    
    # -------------------------------------------------------------------------
    # Agent sessions
    # -------------------------------------------------------------------------
    
    def _run_with_fallback(self, adapted_path: str, prompt: str) -> Optional[Tuple[List[str], str]]:
        """Run the primary agent on the workspace; on any failure, the fallback.
        Returns (files_changed, output), or None when both failed."""
        try:
            return self._run_agent(self.coding_agent_type, self.model, adapted_path, prompt)
        except (ImportError, ValueError) as e:
            print(f"[Adapter] Primary agent not available: {e}")
        except Exception as e:
            print(f"[Adapter] Primary agent error: {e}")
        
        print(f"[Adapter] Trying fallback agent: {self.fallback_agent_type}")
        try:
            return self._run_agent(self.fallback_agent_type, None, adapted_path, prompt)
        except Exception as e:
            print(f"[Adapter] Fallback agent also failed: {e}")
            return None
    
    def _run_agent(
        self,
        agent_type: str,
        model: Optional[str],
        adapted_path: str,
        prompt: str,
    ) -> Tuple[List[str], str]:
        """One coding-agent session over the adapted workspace."""
        kwargs = {"agent_type": agent_type, "workspace": adapted_path}
        if model:
            kwargs["model"] = model
        config = CodingAgentFactory.build_config(**kwargs)
        agent = CodingAgentFactory.create(config)
        agent.initialize(adapted_path)
        
        print(f"[Adapter] Running {agent_type} agent...")
        # The agent also runs the deploy command via its shell tool
        result = agent.generate_code(prompt)
        if not result.success:
            raise RuntimeError(result.error or "Coding agent failed")
        agent.cleanup()
        
        files_changed = result.files_changed if isinstance(result.files_changed, list) else []
        return files_changed, result.output or ""
    
    # -------------------------------------------------------------------------
    # Prompt and interface
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _build_context(
        original_path: str,
        env_vars: Optional[Dict[str, str]],
        env_file: Optional[str],
    ) -> Dict[str, Any]:
        """Values the prompt, the instructions and the run-interface defaults
        are templated with. Variable names only — values stay in the file."""
        names = sorted(env_vars) if env_vars else []
        context = deployment_context(original_path)
        context.update({
            "env_var_names": ", ".join(names) if names else "none",
            "env_file": ENV_FILE if env_file else "none",
            "env_file_flag": f"--env-file {ENV_FILE}" if env_file else "",
        })
        return context
    
    def _build_adaptation_prompt(
        self,
        goal: str,
        setting: DeploymentSetting,
        target_instructions: str,
        context: Optional[Dict[str, Any]] = None,
        previous_error: Optional[str] = None,
    ) -> str:
        """
        Build the prompt for the coding agent.
        
        Loads template from adaptation_prompt.txt and fills in placeholders.
        
        Note: Uses .replace() instead of .format() because the template and
        target_instructions contain Python code examples with dictionary literals
        like {"status": "success"}, which .format() would incorrectly interpret
        as format placeholders.
        """
        template = self.adaptation_prompt_path.read_text()
        
        # Use .replace() instead of .format() to avoid interpreting
        # dictionary literals in code examples as format placeholders
        result = template
        result = result.replace("{goal}", goal)
        result = result.replace("{strategy}", setting.strategy)
        result = result.replace("{provider}", setting.provider or "N/A")
        result = result.replace("{interface}", setting.interface)
        result = result.replace("{resources}", str(setting.resources))
        result = result.replace("{target_instructions}", target_instructions)
        # The target instructions carry the same placeholders, so the context
        # is applied after they are inserted
        for key, value in (context or {}).items():
            result = result.replace("{" + key + "}", str(value))
        
        if previous_error:
            result += (
                "\n\n---\n\n## The previous attempt failed validation\n\n"
                f"{previous_error}\n\n"
                "Fix this in the workspace, re-run the DEPLOY COMMAND, and "
                "report the run interface again."
            )
        
        return result
    
    def _build_run_interface(
        self,
        strategy: str,
        endpoint: Optional[str],
        agent_run_interface: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Build the run interface for the deployed software.
        
        Priority:
        1. Use run_interface from agent output (if provided)
        2. Fall back to the strategy's config.yaml defaults, templated with
           the per-solution context ({deployment_name}, {port})
        
        Args:
            strategy: Deployment strategy name
            endpoint: Endpoint URL extracted from agent output (if any)
            agent_run_interface: Run interface JSON from agent output (if any)
            context: Per-solution values for the templates
            
        Returns:
            Interface dict for the Runner
        """
        if agent_run_interface:
            interface = agent_run_interface.copy()
        else:
            interface = self._render(self.registry.get_default_run_interface(strategy), context or {})
        
        # The Docker instruction once said `path` where the runner reads
        # `predict_path`; accept both so an agent trained on either works
        if "path" in interface and "predict_path" not in interface:
            interface["predict_path"] = interface.pop("path")
        
        # Ensure we have at least a type (safe fallback)
        if "type" not in interface:
            interface["type"] = "function"
        
        # Add endpoint if available (from agent's deployment output)
        if endpoint:
            interface["endpoint"] = endpoint
            interface["deployment_url"] = endpoint
        
        return interface
    
    @staticmethod
    def _render(values: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Fill `{key}` templates in config.yaml values; a value that becomes
        all digits (the port) is returned as an int."""
        rendered = {}
        for key, value in values.items():
            if isinstance(value, str):
                for name, replacement in context.items():
                    value = value.replace("{" + name + "}", str(replacement))
                if value.isdigit():
                    value = int(value)
            rendered[key] = value
        return rendered
    
    def _extract_run_interface_from_output(self, output: str) -> Optional[Dict[str, Any]]:
        """
        Extract run_interface JSON from coding agent output.
        
        The agent is instructed to output the run interface in XML-style tags:
        <run_interface>{"type": "function", "module": "main", ...}</run_interface>
        
        Args:
            output: Full output from the coding agent
            
        Returns:
            Parsed run_interface dict, or None if not found/invalid
        """
        if not output:
            return None
        
        # Extract JSON from <run_interface>...</run_interface> tags
        match = re.search(
            r'<run_interface>\s*(\{[^<]+\})\s*</run_interface>',
            output,
            re.DOTALL
        )
        
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError as e:
                print(f"[Adapter] Warning: Invalid run_interface JSON: {e}")
                return None
        
        return None
    
    def _extract_endpoint_from_output(self, output: str) -> Optional[str]:
        """
        Extract deployment endpoint URL from coding agent output.
        
        The agent is instructed to output the endpoint in XML-style tags:
        <endpoint_url>https://...</endpoint_url>
        """
        if not output:
            return None
        
        match = re.search(r'<endpoint_url>\s*(https?://[^\s<>]+)\s*</endpoint_url>', output)
        if match:
            return match.group(1).strip()
        
        return None
