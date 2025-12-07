# Deployed Software Implementation
#
# The unified Software implementation that wraps any runner.
# This is the ONLY class users get back from deploy().
#
# Usage:
#     software = solution.deploy()  # Returns DeployedSoftware
#     result = software.run({"input": "data"})  # Unified interface
#     software.stop()

from typing import Any, Dict, List, Optional, Union

from src.deployment.base import Software, DeployConfig, DeploymentInfo
from src.deployment.strategies.base import Runner


class DeployedSoftware(Software):
    """
    Unified Software implementation that wraps any runner.
    
    This is the ONLY Software class users get back from deploy().
    It provides a consistent interface regardless of:
    - Whether it's running locally or remotely
    - Whether it's using HTTP, function calls, or subprocess
    - Whether it's on Docker, Modal, or bare metal
    
    All infrastructure complexity is hidden inside the runner.
    
    Response Format (always consistent):
        {"status": "success", "output": <result>}
        {"status": "error", "error": <message>}
    """
    
    def __init__(
        self,
        config: DeployConfig,
        runner: Runner,
        info: DeploymentInfo,
    ):
        """
        Initialize deployed software.
        
        Args:
            config: Deployment configuration
            runner: Infrastructure-specific runner
            info: Deployment metadata
        """
        super().__init__(config)
        self._runner = runner
        self._info = info
        self._logs: List[str] = []
        self._running = True
        
        self._logs.append(f"Initialized {info.strategy} deployment")
    
    @property
    def name(self) -> str:
        """Return the deployment strategy name."""
        return self._info.strategy
    
    def run(self, inputs: Union[Dict, str, bytes]) -> Dict[str, Any]:
        """
        Execute with unified response format.
        
        No matter what runner is used, response is ALWAYS:
        {
            "status": "success" | "error",
            "output": <result>,      # if success
            "error": <message>,      # if error
        }
        
        Args:
            inputs: Input data (dict, string, or bytes)
            
        Returns:
            Unified response dictionary
        """
        if not self._running:
            return {
                "status": "error",
                "error": "Software has been stopped",
            }
        
        # Log the call (truncate long inputs)
        input_preview = str(inputs)[:100] + "..." if len(str(inputs)) > 100 else str(inputs)
        self._logs.append(f"run() called with: {input_preview}")
        
        try:
            # Runner handles all infrastructure details
            result = self._runner.run(inputs)
            
            # Normalize response format
            normalized = self._normalize_response(result)
            
            self._logs.append(f"run() completed: status={normalized['status']}")
            return normalized
            
        except Exception as e:
            self._logs.append(f"run() failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
            }
    
    def _normalize_response(self, result: Any) -> Dict[str, Any]:
        """
        Normalize any result to the unified format.
        
        Args:
            result: Raw result from runner
            
        Returns:
            Normalized response dict
        """
        if isinstance(result, dict):
            # If already has status, return as-is (but ensure required fields)
            if "status" in result:
                if result["status"] == "success" and "output" not in result:
                    # Move all other keys to output
                    output = {k: v for k, v in result.items() if k != "status"}
                    return {"status": "success", "output": output}
                return result
            else:
                # Wrap dict in output
                return {"status": "success", "output": result}
        else:
            # Wrap non-dict in output
            return {"status": "success", "output": result}
    
    def stop(self) -> None:
        """Stop the software and cleanup."""
        if self._running:
            self._runner.stop()
            self._running = False
            self._logs.append("Stopped")
    
    def logs(self) -> str:
        """Get execution logs from both software and runner."""
        runner_logs = self._runner.get_logs() if hasattr(self._runner, 'get_logs') else ""
        all_logs = self._logs.copy()
        if runner_logs:
            all_logs.append("--- Runner Logs ---")
            all_logs.append(runner_logs)
        return "\n".join(all_logs)
    
    def is_healthy(self) -> bool:
        """Check if software is healthy."""
        if not self._running:
            return False
        return self._runner.is_healthy()
    
    # =========================================================================
    # DEPLOYMENT INFO (for advanced users / debugging)
    # =========================================================================
    
    def get_deploy_command(self) -> str:
        """
        Get the command to manually deploy this software.
        
        Useful for debugging or deploying outside this system.
        
        Returns:
            Deploy command string (e.g., "docker build -t solution .")
        """
        return self._info.deploy_command
    
    def get_endpoint(self) -> Optional[str]:
        """
        Get the HTTP endpoint if applicable.
        
        Returns:
            Endpoint URL or None if not HTTP-based
        """
        return self._info.endpoint
    
    def get_deployment_info(self) -> Dict[str, Any]:
        """
        Get full deployment metadata.
        
        Returns:
            Dictionary with all deployment details
        """
        return {
            "strategy": self._info.strategy,
            "provider": self._info.provider,
            "deploy_command": self._info.deploy_command,
            "endpoint": self._info.endpoint,
            "adapted_files": self._info.adapted_files,
            "resources": self._info.resources,
        }
    
    def get_strategy(self) -> str:
        """Get the deployment strategy name."""
        return self._info.strategy
    
    def get_provider(self) -> Optional[str]:
        """Get the cloud provider name if applicable."""
        return self._info.provider

