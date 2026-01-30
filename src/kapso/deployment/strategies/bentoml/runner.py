# BentoML Runner
#
# Executes software by calling BentoCloud deployed endpoints.
# Manages BentoCloud deployment lifecycle using bentoml CLI.
#
# Usage:
#     runner = BentoMLRunner(deployment_name="my-service", code_path="/path/to/adapted/repo")
#     result = runner.run({"input": "data"})
#     runner.stop()  # Terminates the BentoCloud deployment
#     runner.start()  # Re-deploys from code_path

import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Union

from kapso.deployment.strategies.base import Runner


class BentoMLRunner(Runner):
    """
    Runs by calling a BentoCloud deployed service.
    
    Manages BentoCloud deployment lifecycle using CLI commands:
    - start(): Re-deploy from code_path using deploy script
    - stop(): Terminate deployment via `bentoml deployment terminate`
    - run(): Call the service endpoint via HTTP
    
    Usage:
        runner = BentoMLRunner(
            deployment_name="text-classifier",
            endpoint="https://text-classifier.bentoml.ai",
            code_path="/path/to/adapted/repo"
        )
        result = runner.run({"text": "hello"})
        runner.stop()   # Terminates deployment
        runner.start()  # Re-deploys and gets new endpoint
    """
    
    def __init__(
        self,
        deployment_name: str = None,
        endpoint: str = None,
        predict_path: str = "/predict",
        code_path: str = None,
        **kwargs,  # Accept extra params from run_interface
    ):
        """
        Initialize the BentoCloud runner.
        
        Args:
            deployment_name: Name of the BentoCloud deployment
            endpoint: BentoCloud endpoint URL
            predict_path: Path for prediction endpoint
            code_path: Path to the adapted code directory (for re-deployment)
            **kwargs: Additional parameters (ignored)
        """
        self.deployment_name = deployment_name
        self.predict_path = predict_path
        self.code_path = code_path
        self._endpoint = endpoint
        self._api_key = os.environ.get("BENTO_CLOUD_API_KEY")
        self._deployed = False
        self._logs: List[str] = []
        
        # If endpoint provided, mark as deployed
        if self._endpoint:
            self._deployed = True
            self._logs.append(f"Connected to BentoCloud: {self._endpoint}")
        else:
            self._logs.append("No endpoint provided - call start() to deploy")
    
    def start(self) -> None:
        """
        Start or restart the BentoCloud deployment.
        
        Runs the deploy script from code_path to create a new deployment.
        Parses the output to extract the new endpoint URL.
        """
        self._logs.append(f"Starting BentoCloud deployment from {self.code_path}...")
        
        if not self.code_path:
            self._logs.append("No code_path specified, cannot deploy")
            return
        
        if not os.path.exists(self.code_path):
            self._logs.append(f"Code path does not exist: {self.code_path}")
            return
        
        # Look for deploy script
        deploy_script = os.path.join(self.code_path, "deploy.py")
        if not os.path.exists(deploy_script):
            self._logs.append(f"Deploy script not found: {deploy_script}")
            self._logs.append("Trying bentoml deploy command...")
            self._deploy_with_bentoml_cli()
            return
        
        try:
            self._logs.append(f"Running: python deploy.py in {self.code_path}")
            
            result = subprocess.run(
                ["python", "deploy.py"],
                cwd=self.code_path,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout for deployment
            )
            
            output = result.stdout + result.stderr
            self._logs.append(f"Deploy output:\n{output[-500:]}")  # Last 500 chars
            
            if result.returncode == 0:
                # Try to extract deployment name from output
                deploy_name = self._extract_deployment_name(output)
                if deploy_name:
                    self.deployment_name = deploy_name
                    self._logs.append(f"New deployment: {deploy_name}")
                
                # Fetch endpoint from BentoCloud API
                endpoint = self._fetch_endpoint_from_deployment()
                if endpoint:
                    self._endpoint = endpoint
                    self._deployed = True
                    self._logs.append(f"Deployment successful! Endpoint: {self._endpoint}")
                else:
                    # Fallback: try to extract from output
                    endpoint = self._extract_endpoint(output)
                    if endpoint:
                        self._endpoint = endpoint
                        self._deployed = True
                        self._logs.append(f"Deployment successful! Endpoint: {self._endpoint}")
                    else:
                        self._logs.append("Deployment completed but couldn't extract endpoint")
                        self._deployed = True
            else:
                self._logs.append(f"Deploy failed with code {result.returncode}")
                    
        except FileNotFoundError:
            self._logs.append("Python not found")
        except subprocess.TimeoutExpired:
            self._logs.append("Timeout waiting for deployment (5 min)")
        except Exception as e:
            self._logs.append(f"Error during deployment: {e}")
    
    def _deploy_with_bentoml_cli(self) -> None:
        """Fallback: Deploy using bentoml deploy command."""
        try:
            self._logs.append(f"Running: bentoml deploy in {self.code_path}")
            
            result = subprocess.run(
                ["bentoml", "deploy", "."],
                cwd=self.code_path,
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            output = result.stdout + result.stderr
            self._logs.append(f"Deploy output:\n{output[-500:]}")
            
            if result.returncode == 0:
                endpoint = self._extract_endpoint(output)
                if endpoint:
                    self._endpoint = endpoint
                    self._deployed = True
                    self._logs.append(f"Deployment successful! Endpoint: {self._endpoint}")
                else:
                    self._deployed = True
                    self._logs.append("Deployment completed but couldn't extract endpoint")
            else:
                self._logs.append(f"bentoml deploy failed with code {result.returncode}")
                
        except FileNotFoundError:
            self._logs.append("bentoml CLI not found. Install with: pip install bentoml")
        except subprocess.TimeoutExpired:
            self._logs.append("Timeout waiting for bentoml deploy")
        except Exception as e:
            self._logs.append(f"Error during bentoml deploy: {e}")
    
    def _extract_deployment_name(self, output: str) -> Optional[str]:
        """Extract deployment name from deploy output."""
        # Look for patterns like 'Created deployment "qa-service-87iv"'
        patterns = [
            r'[Cc]reated deployment ["\']([^"\']+)["\']',
            r'deployment ["\']([^"\']+)["\']',
            r'qa-service-[a-z0-9]+',  # Common naming pattern
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1) if match.lastindex else match.group(0)
        
        return None
    
    def _fetch_endpoint_from_deployment(self) -> Optional[str]:
        """Fetch endpoint URL from BentoCloud deployment info."""
        if not self.deployment_name:
            return None
        
        try:
            result = subprocess.run(
                ["bentoml", "deployment", "get", self.deployment_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode == 0:
                # Look for endpoint_urls in the output
                match = re.search(r'https://[a-zA-Z0-9-]+\S*\.bentoml\.ai', result.stdout)
                if match:
                    return match.group(0)
        except Exception as e:
            self._logs.append(f"Could not fetch endpoint: {e}")
        
        return None
    
    def _extract_endpoint(self, output: str) -> Optional[str]:
        """Extract endpoint URL from deployment output."""
        # Common patterns for BentoCloud endpoint URLs
        patterns = [
            r'https://[a-zA-Z0-9-]+[a-zA-Z0-9-]*\.mt-[a-z0-9]+\.bentoml\.ai',  # Most specific
            r'https://[a-zA-Z0-9-]+\..*\.bentoml\.ai',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                endpoint = match.group(0)
                endpoint = endpoint.rstrip('.,;:')
                return endpoint
        
        return None
    
    def stop(self) -> None:
        """
        Terminate the BentoCloud deployment.
        
        Uses `bentoml deployment terminate <name>` to stop the deployment.
        Can be restarted later with start() which will re-deploy.
        """
        self._logs.append(f"Terminating BentoCloud deployment: {self.deployment_name}...")
        
        # Clear local state
        self._deployed = False
        self._endpoint = None
        
        if not self.deployment_name:
            self._logs.append("No deployment name specified, skipping termination")
            return
        
        try:
            # Use bentoml CLI to terminate the deployment
            result = subprocess.run(
                ["bentoml", "deployment", "terminate", self.deployment_name],
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            if result.returncode == 0:
                self._logs.append(f"BentoCloud deployment {self.deployment_name} terminated successfully")
                if result.stdout:
                    self._logs.append(f"Output: {result.stdout.strip()}")
            else:
                self._logs.append(f"Terminate returned code {result.returncode}")
                if result.stderr:
                    self._logs.append(f"Error: {result.stderr.strip()}")
                    
        except FileNotFoundError:
            self._logs.append("bentoml CLI not found. Install with: pip install bentoml")
        except subprocess.TimeoutExpired:
            self._logs.append("Timeout waiting for bentoml deployment terminate")
        except Exception as e:
            self._logs.append(f"Error terminating deployment: {e}")
    
    def run(self, inputs: Union[Dict, str, bytes]) -> Any:
        """
        Call the BentoCloud service.
        
        Args:
            inputs: Input data for the service
            
        Returns:
            Service output
        """
        if not self._deployed or not self._endpoint:
            return {
                "error": "BentoCloud service not deployed or not connected",
                "instructions": [
                    "1. Set BENTO_CLOUD_API_KEY environment variable",
                    f"2. Call runner.start() to deploy from {self.code_path}",
                    f"3. Deployment name: {self.deployment_name}",
                ],
                "start_hint": "Call runner.start() to deploy",
            }
        
        try:
            import requests
            
            url = f"{self._endpoint.rstrip('/')}{self.predict_path}"
            headers = {"Content-Type": "application/json"}
            
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            
            # Format input for BentoML endpoint
            if isinstance(inputs, dict):
                json_body = inputs  # Send directly, let BentoML handle it
            elif isinstance(inputs, bytes):
                json_body = {"data": inputs.decode('utf-8')}
            else:
                json_body = {"text": str(inputs)}
            
            self._logs.append(f"POST {url}")
            
            response = requests.post(url, json=json_body, headers=headers, timeout=60)
            
            self._logs.append(f"Response: {response.status_code}")
            response.raise_for_status()
            
            return response.json()
            
        except ImportError:
            return {"error": "requests package required. Run: pip install requests"}
        except Exception as e:
            self._logs.append(f"BentoCloud call error: {e}")
            return {"error": str(e)}
    
    def is_healthy(self) -> bool:
        """Check if the BentoCloud service is available."""
        if not self._deployed or not self._endpoint:
            return False
        
        try:
            import requests
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            response = requests.get(f"{self._endpoint.rstrip('/')}/healthz", headers=headers, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_logs(self) -> str:
        """Get runner logs."""
        return "\n".join(self._logs)
    
    def get_deploy_command(self) -> str:
        """Get the command to deploy to BentoCloud."""
        if self.code_path:
            return f"cd {self.code_path} && python deploy.py"
        return "python deploy.py"
