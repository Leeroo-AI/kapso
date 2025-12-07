# Modal Runner
#
# Executes software by calling Modal functions remotely.
# For MODAL deployment strategy.
#
# Usage:
#     runner = ModalRunner(app_name="my-app", function_name="predict")
#     result = runner.run({"input": "data"})

import os
from typing import Any, Dict, List, Union

from src.deployment.strategies.base import Runner


class ModalRunner(Runner):
    """
    Runs by calling a Modal function remotely.
    
    This runner provides the interface to deployed Modal functions.
    It requires Modal to be installed and configured for remote calls.
    
    Usage:
        runner = ModalRunner(app_name="text-embeddings", function_name="predict")
        result = runner.run({"text": "hello"})
    """
    
    def __init__(
        self,
        app_name: str,
        function_name: str = "predict",
        code_path: str = None,
    ):
        """
        Initialize the Modal runner.
        
        Args:
            app_name: Name of the Modal app
            function_name: Name of the function to call
            code_path: Path to the code directory (for deploy command)
        """
        self.app_name = app_name
        self.function_name = function_name
        self.code_path = code_path
        self._fn = None
        self._deployed = False
        self._logs: List[str] = []
        
        # Try to load Modal function if Modal is available
        self._load()
    
    def _load(self) -> None:
        """Try to load the Modal function."""
        try:
            import modal
            
            # Check for Modal token
            if not os.environ.get("MODAL_TOKEN_ID"):
                self._logs.append("Modal not authenticated. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.")
                self._logs.append(f"To deploy: cd {self.code_path} && modal deploy modal_app.py")
                return
            
            # Try to look up the deployed function
            try:
                self._fn = modal.Function.lookup(self.app_name, self.function_name)
                self._deployed = True
                self._logs.append(f"Connected to Modal function: {self.app_name}/{self.function_name}")
            except Exception as e:
                self._logs.append(f"Modal function not deployed: {e}")
                self._logs.append(f"To deploy: cd {self.code_path} && modal deploy modal_app.py")
                
        except ImportError:
            self._logs.append("Modal not installed. Run: pip install modal")
            self._logs.append("Then authenticate: modal token new")
    
    def run(self, inputs: Union[Dict, str, bytes]) -> Any:
        """
        Call the Modal function remotely.
        
        Args:
            inputs: Input data for the function
            
        Returns:
            Function output
        """
        if not self._deployed or self._fn is None:
            return {
                "error": "Modal function not deployed or not authenticated",
                "instructions": [
                    "1. Install Modal: pip install modal",
                    "2. Authenticate: modal token new",
                    "3. Deploy: modal deploy modal_app.py",
                    f"4. App name: {self.app_name}",
                ],
                "deploy_command": f"cd {self.code_path} && modal deploy modal_app.py",
            }
        
        try:
            self._logs.append(f"Calling Modal function with: {str(inputs)[:100]}...")
            result = self._fn.remote(inputs)
            self._logs.append("Modal function returned successfully")
            return result
        except Exception as e:
            self._logs.append(f"Modal call error: {e}")
            return {"error": str(e)}
    
    def stop(self) -> None:
        """No-op for Modal - functions are serverless."""
        self._fn = None
    
    def is_healthy(self) -> bool:
        """Check if the Modal function is available."""
        return self._deployed
    
    def get_logs(self) -> str:
        """Get runner logs."""
        return "\n".join(self._logs)
    
    def get_deploy_command(self) -> str:
        """Get the command to deploy to Modal."""
        if self.code_path:
            return f"cd {self.code_path} && modal deploy modal_app.py"
        return "modal deploy modal_app.py"

