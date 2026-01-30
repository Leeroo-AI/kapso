# Docker Runner
#
# Executes software by making HTTP requests to a Docker container.
# Manages container lifecycle using docker-py SDK.
#
# Usage:
#     runner = DockerRunner(endpoint="http://localhost:8000", container_name="my-app")
#     result = runner.run({"input": "data"})
#     runner.stop()  # Stops and removes container
#     runner.start()  # Restarts container

import time
from typing import Any, Dict, List, Optional, Union

from kapso.deployment.strategies.base import Runner


class DockerRunner(Runner):
    """
    Runs by making HTTP requests to a Docker container.
    
    Manages container lifecycle using docker-py SDK:
    - start(): Start or restart the container
    - stop(): Stop and remove the container
    - run(): Make HTTP requests to the container's API
    
    Expects the container to expose:
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
        container_name: str = None,
        image_name: str = None,
        port: int = 8000,
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
            code_path: Path to the code directory (for building image)
            container_name: Name of the Docker container to manage
            image_name: Name of the Docker image to use
            port: Port to expose (default: 8000)
            **kwargs: Additional parameters (ignored)
        """
        self.endpoint = endpoint.rstrip("/")
        self.predict_path = predict_path
        self.health_path = health_path
        self.timeout = timeout
        self.headers = headers or {}
        self.code_path = code_path
        self.container_name = container_name or "kapso-app"
        self.image_name = image_name or "kapso-solution"
        self.port = port
        self._container = None
        self._docker_client = None
        self._logs: List[str] = []
        
        # Initialize Docker client
        self._init_docker_client()
        self._logs.append(f"Initialized Docker runner for {endpoint}")
    
    def _init_docker_client(self) -> None:
        """Initialize the Docker client."""
        try:
            import docker
            self._docker_client = docker.from_env()
            self._logs.append("Docker client initialized")
        except ImportError:
            self._logs.append("docker package not installed. Run: pip install docker")
            self._docker_client = None
        except Exception as e:
            self._logs.append(f"Failed to connect to Docker daemon: {e}")
            self._docker_client = None
    
    def _get_container(self):
        """Get the container by name if it exists."""
        if not self._docker_client:
            return None
        
        try:
            return self._docker_client.containers.get(self.container_name)
        except Exception:
            return None
    
    def start(self) -> None:
        """
        Start or restart the Docker container.
        
        If container exists and is stopped, starts it.
        If container doesn't exist, creates and starts it.
        """
        if not self._docker_client:
            self._init_docker_client()
            if not self._docker_client:
                self._logs.append("Cannot start: Docker client not available")
                return
        
        # Check if container already exists
        container = self._get_container()
        
        if container:
            status = container.status
            if status == "running":
                self._logs.append(f"Container {self.container_name} is already running")
                self._container = container
                return
            elif status in ("exited", "created", "paused"):
                # Start the existing container
                self._logs.append(f"Starting existing container {self.container_name}...")
                container.start()
                self._container = container
                self._logs.append(f"Container {self.container_name} started")
                return
        
        # Create and start new container
        self._logs.append(f"Creating new container {self.container_name} from image {self.image_name}...")
        try:
            self._container = self._docker_client.containers.run(
                self.image_name,
                name=self.container_name,
                ports={f"{self.port}/tcp": self.port},
                detach=True,
            )
            self._logs.append(f"Container {self.container_name} created and started")
        except Exception as e:
            self._logs.append(f"Failed to create container: {e}")
            raise
    
    def stop(self) -> None:
        """
        Stop and remove the Docker container.
        
        Gracefully stops the container then removes it.
        Can be restarted with start().
        """
        if not self._docker_client:
            self._logs.append("Docker client not available")
            return
        
        # Get current container reference
        container = self._get_container()
        
        if not container:
            self._logs.append(f"Container {self.container_name} not found")
            self._container = None
            return
        
        try:
            # Stop the container if running
            if container.status == "running":
                self._logs.append(f"Stopping container {self.container_name}...")
                container.stop(timeout=10)
                self._logs.append(f"Container {self.container_name} stopped")
            
            # Remove the container
            self._logs.append(f"Removing container {self.container_name}...")
            container.remove()
            self._logs.append(f"Container {self.container_name} removed")
            
        except Exception as e:
            self._logs.append(f"Error stopping/removing container: {e}")
        
        self._container = None
    
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
    
    def is_healthy(self) -> bool:
        """Check if the container is running and endpoint is responding."""
        # First check if container is running
        container = self._get_container()
        if container and container.status != "running":
            return False
        
        # Then check HTTP health endpoint
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
        """Return accumulated logs including container logs."""
        all_logs = self._logs.copy()
        
        # Try to get container logs
        container = self._get_container()
        if container:
            try:
                container_logs = container.logs(tail=50).decode('utf-8')
                all_logs.append("--- Container Logs ---")
                all_logs.append(container_logs)
            except Exception:
                pass
        
        return "\n".join(all_logs)
    
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
            return (
                f"cd {self.code_path} && "
                f"docker build -t {self.image_name} . && "
                f"docker run -d --name {self.container_name} -p {self.port}:{self.port} {self.image_name}"
            )
        return (
            f"docker build -t {self.image_name} . && "
            f"docker run -d --name {self.container_name} -p {self.port}:{self.port} {self.image_name}"
        )
