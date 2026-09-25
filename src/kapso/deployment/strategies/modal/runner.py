# Modal Runner
#
# Executes software by calling Modal functions remotely.
# Manages Modal app lifecycle including deployment termination.
#
# Usage:
#     runner = ModalRunner(app_name="my-app", function_name="predict")
#     result = runner.run({"input": "data"})
#     runner.stop()  # Terminates the Modal app
#     runner.start()  # Re-connects to the function

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Union

from kapso.deployment.strategies.base import Runner


class ModalRunner(Runner):
    """
    Runs by calling a Modal function remotely.
    
    Manages Modal app lifecycle:
    - start(): Re-lookup and connect to the Modal function
    - stop(): Terminate the Modal app deployment
    - run(): Call the function remotely
    
    Requires Modal to be installed and authenticated.
    
    Usage:
        runner = ModalRunner(app_name="text-embeddings", function_name="predict")
        result = runner.run({"text": "hello"})
        runner.stop()  # Terminates deployment
    """
    
    def __init__(
        self,
        app_name: str = None,
        function_name: str = "predict",
        code_path: str = None,
        **kwargs,  # Accept extra params from run_interface
    ):
        """
        Initialize the Modal runner.
        
        Args:
            app_name: Name of the Modal app
            function_name: Name of the function to call
            code_path: Path to the code directory (for deploy command)
            **kwargs: Additional parameters (ignored)
        """
        self.app_name = app_name
        self.function_name = function_name
        self.code_path = code_path
        self._fn = None
        self._deployed = False
        self._logs: List[str] = []
        
        # Try to load Modal function if Modal is available
        self._load()
    
    @staticmethod
    def _authenticated() -> bool:
        """`modal token new` writes ~/.modal.toml; CI sets the env pair."""
        return bool(os.environ.get("MODAL_TOKEN_ID")) or (Path.home() / ".modal.toml").exists()
    
    def _load(self) -> None:
        """Try to load the Modal function."""
        try:
            import modal
            
            if not self._authenticated():
                self._logs.append("Modal not authenticated. Run `modal token new`, or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.")
                self._logs.append(f"To deploy: cd {self.code_path} && modal deploy modal_app.py")
                return
            
            if not self.app_name:
                self._logs.append("No app name specified; cannot look up the function")
                return
            
            # Look up the deployed function. `Function.from_name` is the
            # current API (lazy: it resolves on first use, so hydrate to learn
            # now whether the app is deployed); `lookup` serves older SDKs.
            try:
                from_name = getattr(modal.Function, "from_name", None)
                fn = from_name(self.app_name, self.function_name) if from_name else modal.Function.lookup(self.app_name, self.function_name)
                hydrate = getattr(fn, "hydrate", None)
                if callable(hydrate):
                    hydrate()
                self._fn = fn
                self._deployed = True
                self._logs.append(f"Connected to Modal function: {self.app_name}/{self.function_name}")
            except Exception as e:
                self._logs.append(f"Modal function not deployed: {e}")
                self._logs.append(f"To deploy: cd {self.code_path} && modal deploy modal_app.py")
                
        except ImportError:
            self._logs.append("Modal not installed. Run: pip install modal")
            self._logs.append("Then authenticate: modal token new")
    
    def start(self) -> None:
        """
        Start or restart the Modal runner.
        
        Looks the function up again; when the app is not deployed (stop()
        removes it) and the adapted workspace holds `modal_app.py`, runs
        `modal deploy` there first, so stop() → start() is a round trip.
        """
        self._logs.append("Starting Modal runner...")
        self._fn = None
        self._deployed = False
        self._load()
        
        if not self._deployed and self._authenticated():
            self._deploy()
            if not self._deployed:
                self._load()
        
        if self._deployed:
            self._logs.append("Modal runner started - connected to function")
        else:
            self._logs.append("Modal runner started - function not yet deployed")
    
    def _deploy(self) -> None:
        """Run `modal deploy modal_app.py` in the adapted workspace."""
        app_file = Path(self.code_path or "") / "modal_app.py"
        if not self.code_path or not app_file.exists():
            self._logs.append("No modal_app.py in the workspace; cannot deploy")
            return
        try:
            self._logs.append(f"Running: modal deploy modal_app.py in {self.code_path}")
            result = subprocess.run(
                ["modal", "deploy", "modal_app.py"],
                cwd=self.code_path, capture_output=True, text=True, timeout=600,
            )
            output = (result.stdout or "") + (result.stderr or "")
            self._logs.append(f"Deploy output:\n{output[-500:]}")
            if result.returncode != 0:
                self._logs.append(f"modal deploy failed with code {result.returncode}")
        except FileNotFoundError:
            self._logs.append("Modal CLI not found. Install with: pip install modal")
        except subprocess.TimeoutExpired:
            self._logs.append("Timeout waiting for modal deploy (10 min)")
        except Exception as e:
            self._logs.append(f"Error during modal deploy: {e}")
    
    def stop(self) -> None:
        """
        Stop the Modal app deployment.
        
        Terminates the Modal app using `modal app stop`.
        This will stop all running containers and release resources.
        Can be restarted with `modal deploy` followed by start().
        """
        self._logs.append(f"Stopping Modal app: {self.app_name}...")
        
        # Clear local references first
        self._fn = None
        self._deployed = False
        
        if not self.app_name:
            self._logs.append("No app name specified, skipping Modal app stop")
            return
        
        try:
            # --yes: the CLI asks for confirmation and aborts without a TTY,
            # and a runner never has one
            result = subprocess.run(
                ["modal", "app", "stop", self.app_name, "--yes"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode == 0:
                self._logs.append(f"Modal app {self.app_name} stopped successfully")
                if result.stdout:
                    self._logs.append(f"Output: {result.stdout.strip()}")
            else:
                # Modal app stop might fail if app doesn't exist or is already stopped
                self._logs.append(f"Modal app stop returned: {result.returncode}")
                if result.stderr:
                    self._logs.append(f"Error: {result.stderr.strip()}")
                    
        except FileNotFoundError:
            self._logs.append("Modal CLI not found. Install with: pip install modal")
        except subprocess.TimeoutExpired:
            self._logs.append("Timeout waiting for modal app stop")
        except Exception as e:
            self._logs.append(f"Error stopping Modal app: {e}")
    
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
    
    def is_healthy(self) -> bool:
        """Check if the Modal function is available."""
        return self._deployed and self._fn is not None
    
    def get_logs(self) -> str:
        """Get runner logs."""
        return "\n".join(self._logs)
    
    def get_deploy_command(self) -> str:
        """Get the command to deploy to Modal."""
        if self.code_path:
            return f"cd {self.code_path} && modal deploy modal_app.py"
        return "modal deploy modal_app.py"
