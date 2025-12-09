# BentoML Runner
#
# Executes software by calling BentoCloud deployed endpoints.
# For BENTOML deployment strategy with cloud hosting.
#
# Usage:
#     runner = BentoMLRunner(deployment_name="my-service")
#     result = runner.run({"input": "data"})

import os
from typing import Any, Dict, List, Union

from src.deployment.strategies.base import Runner


class BentoMLRunner(Runner):
    """
    Runs by calling a BentoCloud deployed service.
    
    This runner provides the interface to BentoCloud deployments.
    It handles authentication and API calls to the deployed service.
    
    Usage:
        runner = BentoMLRunner(deployment_name="text-classifier")
        result = runner.run({"text": "hello"})
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
            endpoint: BentoCloud endpoint URL (auto-detected if not provided)
            predict_path: Path for prediction endpoint
            code_path: Path to the code directory (for deploy command)
            **kwargs: Additional parameters (ignored)
        """
        self.deployment_name = deployment_name
        self.predict_path = predict_path
        self.code_path = code_path
        self._endpoint = endpoint
        self._api_key = os.environ.get("BENTO_CLOUD_API_KEY")
        self._api_endpoint = os.environ.get("BENTO_CLOUD_API_ENDPOINT", "https://cloud.bentoml.com")
        self._deployed = False
        self._logs: List[str] = []
        
        self._connect()
    
    def _connect(self) -> None:
        """Try to connect to the BentoCloud deployment."""
        if not self._api_key:
            self._logs.append("BENTO_CLOUD_API_KEY not set")
            self._logs.append(f"To deploy: cd {self.code_path} && python deploy.py")
            return
        
        if not self._endpoint:
            try:
                self._endpoint = self._discover_endpoint()
            except Exception as e:
                self._logs.append(f"Could not discover endpoint: {e}")
                return
        
        if self._endpoint:
            self._deployed = True
            self._logs.append(f"Connected to BentoCloud: {self._endpoint}")
    
    def _discover_endpoint(self) -> str:
        """Discover the deployment endpoint from BentoCloud."""
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            
            api_url = f"{self._api_endpoint}/api/v1/deployments/{self.deployment_name}"
            response = requests.get(api_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("endpoint") or data.get("url")
            
            self._logs.append(f"Deployment not found: {self.deployment_name}")
            return None
            
        except ImportError:
            self._logs.append("requests package required. Run: pip install requests")
            return None
        except Exception as e:
            self._logs.append(f"API error: {e}")
            return None
    
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
                "error": "BentoCloud service not deployed or not authenticated",
                "instructions": [
                    "1. Set BENTO_CLOUD_API_KEY environment variable",
                    f"2. Deploy: cd {self.code_path} && python deploy.py",
                    f"3. Deployment name: {self.deployment_name}",
                ],
                "deploy_command": f"cd {self.code_path} && python deploy.py",
            }
        
        try:
            import requests
            
            url = f"{self._endpoint.rstrip('/')}{self.predict_path}"
            headers = {"Content-Type": "application/json"}
            
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            
            if isinstance(inputs, dict):
                json_body = {"inputs": inputs}
            elif isinstance(inputs, bytes):
                json_body = {"inputs": {"data": inputs.decode('utf-8')}}
            else:
                json_body = {"inputs": {"text": str(inputs)}}
            
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
    
    def stop(self) -> None:
        """No-op for BentoCloud - services are managed by BentoCloud."""
        self._logs.append("Disconnected from BentoCloud")
    
    def is_healthy(self) -> bool:
        """Check if the BentoCloud service is available."""
        if not self._deployed or not self._endpoint:
            return False
        
        try:
            import requests
            headers = {"Authorization": f"Bearer {self._api_key}"}
            response = requests.get(f"{self._endpoint.rstrip('/')}/health", headers=headers, timeout=10)
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

