# Local Runner
#
# Executes software by importing and calling a Python function.
# Best for LOCAL deployment with function-based interface.
#
# Usage:
#     runner = LocalRunner(code_path, module="main", callable="predict")
#     result = runner.run({"input": "data"})

import os
import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Union

from kapso.deployment.strategies.base import Runner


class LocalRunner(Runner):
    """
    Runs by importing and calling a Python function.
    
    Used for LOCAL deployment with function interface.
    The function is imported once and called for each run().
    
    Expected function signature:
        def predict(inputs: dict) -> dict:
            # Process inputs
            return {"result": ...}
    """
    
    def __init__(
        self, 
        code_path: str, 
        module: str = "main", 
        callable: str = "predict",
        **kwargs,  # Accept extra params from run_interface
    ):
        """
        Initialize the local runner.
        
        Args:
            code_path: Path to the solution directory
            module: Module name to import (without .py)
            callable: Function name to call
            **kwargs: Additional parameters (ignored)
        """
        self.code_path = code_path
        self.module_name = module
        self.callable_name = callable
        self._fn = None
        self._module = None
        self._logs: List[str] = []
        self._load()
    
    def _load(self) -> None:
        """Import the module and get the callable."""
        # Add code path to Python path (use absolute path)
        abs_code_path = os.path.abspath(self.code_path)
        if abs_code_path not in sys.path:
            sys.path.insert(0, abs_code_path)
        
        # Find the module file - try multiple locations
        module_paths = [
            Path(self.code_path) / f"{self.module_name}.py",
            Path(self.code_path) / "src" / f"{self.module_name}.py",
            Path(self.code_path) / self.module_name / "__init__.py",
        ]
        
        module_path = None
        for path in module_paths:
            if path.exists():
                module_path = path
                break
        
        if module_path is None:
            raise FileNotFoundError(
                f"Module {self.module_name}.py not found in {self.code_path}. "
                f"Tried: {[str(p) for p in module_paths]}"
            )
        
        self._logs.append(f"Loading module from {module_path}")
        
        # Import the module
        spec = importlib.util.spec_from_file_location(self.module_name, module_path)
        self._module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = self._module
        spec.loader.exec_module(self._module)
        
        # Get the callable - try function first, then class
        if hasattr(self._module, self.callable_name):
            self._fn = getattr(self._module, self.callable_name)
            self._logs.append(f"Loaded function {self.callable_name}")
        else:
            # Try to find a class with predict/run method
            for attr_name in dir(self._module):
                attr = getattr(self._module, attr_name)
                if isinstance(attr, type):
                    if hasattr(attr, self.callable_name):
                        instance = attr()
                        self._fn = getattr(instance, self.callable_name)
                        self._logs.append(f"Loaded class method {attr_name}.{self.callable_name}")
                        break
            
            if self._fn is None:
                raise AttributeError(
                    f"Module '{self.module_name}' has no '{self.callable_name}' function or class method. "
                    f"Available: {[n for n in dir(self._module) if not n.startswith('_')]}"
                )
    
    def run(self, inputs: Union[Dict, str, bytes]) -> Any:
        """
        Call the imported function with inputs.
        
        Args:
            inputs: Input data for the function
            
        Returns:
            Result from the function
        """
        if self._fn is None:
            raise RuntimeError("Function not loaded. Call start() to restart.")
        
        self._logs.append(f"Calling {self.callable_name} with inputs")
        return self._fn(inputs)
    
    def start(self) -> None:
        """
        Start or restart the runner.
        
        Re-loads the module and function. Can be called after stop()
        to restart the runner, or to reload the module if it changed.
        """
        self._logs.append("Starting/restarting runner...")
        self._load()
        self._logs.append("Runner started")
    
    def stop(self) -> None:
        """
        Stop and cleanup the loaded module.
        
        Unloads the module from sys.modules and clears references.
        Can be restarted with start().
        """
        self._fn = None
        self._module = None
        if self.module_name in sys.modules:
            del sys.modules[self.module_name]
        self._logs.append("Stopped")
    
    def is_healthy(self) -> bool:
        """Check if function is loaded and callable."""
        return self._fn is not None and callable(self._fn)
    
    def get_logs(self) -> str:
        """Return accumulated logs."""
        return "\n".join(self._logs)

