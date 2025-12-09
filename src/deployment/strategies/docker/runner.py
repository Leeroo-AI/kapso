# Docker Runner
#
# Executes software by making HTTP requests to a Docker container.
# Used for DOCKER deployment with HTTP-based API.
#
# Usage:
#     runner = DockerRunner(endpoint="http://localhost:8000")
#     result = runner.run({"input": "data"})

import time
from typing import Any, Dict, List, Optional, Union

from src.deployment.strategies.base import Runner


class DockerRunner(Runner):
    """
    Runs by making HTTP requests to a Docker container.
    
    Used for DOCKER deployment with HTTP-based API.
    Expects the endpoint to have:
    - POST /predict - for running predictions
    - GET /health - for health checks (optional)
    """
    
    def __init__(
        self,
        endpoint: str = "http://localhost:8000",
        predict_path: str = "/predict",
        health_path: str = "/health",
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None,
        code_path: str = None,
        **kwargs,  # Accept extra params from run_interface
    ):
        """
        Initialize the Docker runner.
        
        Args:
            endpoint: Base URL of the deployed service
            predict_path: Path for prediction endpoint
            health_path: Path for health check endpoint
            timeout: Request timeout in seconds
            headers: Additional headers to send with requests
            code_path: Path to the code directory (for deploy command)
            **kwargs: Additional parameters (ignored)
        """
        self.endpoint = endpoint.rstrip("/")
        self.predict_path = predict_path
        self.health_path = health_path
        self.timeout = timeout
        self.headers = headers or {}
        self.code_path = code_path
        self._logs: List[str] = []
        
        self._logs.append(f"Initialized Docker runner for {endpoint}")
    
    def run(self, inputs: Union[Dict, str, bytes]) -> Any:
        """
        Make HTTP POST request to prediction endpoint.
        
        Args:
            inputs: Input data (sent as JSON body)
            
        Returns:
            JSON response from the endpoint
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")
        
        url = f"{self.endpoint}{self.predict_path}"
        
        # Prepare request body
        if isinstance(inputs, dict):
            json_body = inputs
        elif isinstance(inputs, bytes):
            json_body = {"data": inputs.decode('utf-8')}
        else:
            json_body = {"data": str(inputs)}
        
        self._logs.append(f"POST {url}")
        
        response = requests.post(
            url,
            json=json_body,
            headers=self.headers,
            timeout=self.timeout,
        )
        
        self._logs.append(f"Response: {response.status_code}")
        
        response.raise_for_status()
        return response.json()
    
    def stop(self) -> None:
        """Docker runner doesn't manage the container lifecycle."""
        self._logs.append("Stopped (Docker runner)")
    
    def is_healthy(self) -> bool:
        """Check if the endpoint is responding."""
        try:
            import requests
        except ImportError:
            return False
        
        try:
            response = requests.get(
                f"{self.endpoint}{self.health_path}",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_logs(self) -> str:
        """Return accumulated logs."""
        return "\n".join(self._logs)
    
    def wait_for_ready(self, timeout: int = 60, interval: int = 2) -> bool:
        """
        Wait for the container to become healthy.
        
        Args:
            timeout: Maximum seconds to wait
            interval: Seconds between health checks
            
        Returns:
            True if healthy, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_healthy():
                self._logs.append("Container is ready")
                return True
            self._logs.append(f"Waiting for container... ({int(time.time() - start_time)}s)")
            time.sleep(interval)
        
        self._logs.append("Timeout waiting for container")
        return False
    
    def get_deploy_command(self) -> str:
        """Get the command to build and run the Docker container."""
        if self.code_path:
            return f"cd {self.code_path} && docker build -t solution . && docker run -p 8000:8000 solution"
        return "docker build -t solution . && docker run -p 8000:8000 solution"

