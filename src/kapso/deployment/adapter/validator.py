# Adaptation Validator
#
# Validates that a solution has been correctly adapted for deployment.
# Runs a series of checks based on the target deployment strategy.
#
# Usage:
#     validator = AdaptationValidator()
#     result = validator.validate(path, setting)

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from kapso.deployment.base import DeploymentSetting


@dataclass
class ValidationResult:
    """Result of validation checks."""
    success: bool
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)


class AdaptationValidator:
    """
    Validates that a solution is correctly adapted for deployment.
    
    Validation tiers:
    1. Syntax: Python files are syntactically correct
    2. Import: Main module can be imported without error
    3. Structure: Required files exist for the deployment strategy
    4. Execution: Basic execution test (optional)
    """
    
    def validate(
        self,
        code_path: str,
        setting: DeploymentSetting,
        run_execution_test: bool = False,
    ) -> ValidationResult:
        """
        Validate an adapted solution.
        
        Args:
            code_path: Path to the adapted solution
            setting: Deployment setting to validate against
            run_execution_test: Whether to run execution tests
            
        Returns:
            ValidationResult with success status and details
        """
        result = ValidationResult(success=True)
        path = Path(code_path)
        
        # Tier 1: Syntax check
        syntax_ok = self._check_syntax(path, result)
        if not syntax_ok:
            result.success = False
            return result
        
        # Tier 2: Import check
        import_ok = self._check_import(path, result)
        if not import_ok:
            result.success = False
            return result
        
        # Tier 3: Structure check (strategy-specific)
        structure_ok = self._check_structure(path, setting, result)
        if not structure_ok:
            result.success = False
            return result
        
        # Tier 4: Execution test (optional)
        if run_execution_test:
            exec_ok = self._check_execution(path, setting, result)
            if not exec_ok:
                result.success = False
                return result
        
        result.logs.append("All validation checks passed")
        return result
    
    def _check_syntax(self, path: Path, result: ValidationResult) -> bool:
        """Check that all Python files are syntactically correct."""
        result.logs.append("Checking syntax...")
        
        for py_file in path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    compile(f.read(), py_file, 'exec')
                result.checks_passed.append(f"syntax:{py_file.name}")
            except SyntaxError as e:
                result.checks_failed.append(f"syntax:{py_file.name}")
                result.error = f"Syntax error in {py_file.name}: {e}"
                return False
        
        result.logs.append("Syntax check passed")
        return True
    
    def _check_import(self, path: Path, result: ValidationResult) -> bool:
        """Check that the main module can be imported."""
        result.logs.append("Checking import...")
        
        # Find main module
        main_files = ["main.py", "app.py"]
        for name in main_files:
            main_file = path / name
            if main_file.exists():
                # Try to import using subprocess to avoid polluting current namespace
                cmd = [
                    sys.executable, "-c",
                    f"import sys; sys.path.insert(0, '{path}'); "
                    f"import {name[:-3]}; print('OK')"
                ]
                try:
                    proc = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if proc.returncode == 0 and "OK" in proc.stdout:
                        result.checks_passed.append(f"import:{name}")
                        result.logs.append(f"Import check passed for {name}")
                        return True
                    else:
                        result.checks_failed.append(f"import:{name}")
                        result.error = f"Import failed for {name}: {proc.stderr}"
                        return False
                except subprocess.TimeoutExpired:
                    result.checks_failed.append(f"import:{name}")
                    result.error = f"Import timed out for {name}"
                    return False
        
        # No main file found - still OK if other .py files exist
        result.logs.append("No main.py or app.py found, skipping import check")
        return True
    
    def _check_structure(
        self,
        path: Path,
        setting: DeploymentSetting,
        result: ValidationResult,
    ) -> bool:
        """Check that required files exist for the deployment strategy."""
        result.logs.append(f"Checking structure for {setting.strategy}...")
        
        # Strategy-specific required files
        required_files = {
            "local": ["main.py"],  # Or src/main.py
            "docker": ["Dockerfile", "requirements.txt"],
            "modal": ["modal_app.py", "requirements.txt"],
            "bentoml": ["service.py", "bentofile.yaml"],
        }
        
        # Get required files for this strategy
        required = required_files.get(setting.strategy, ["main.py"])
        
        all_found = True
        for filename in required:
            file_path = path / filename
            alt_path = path / "src" / filename  # Also check src/
            
            if file_path.exists() or alt_path.exists():
                result.checks_passed.append(f"structure:{filename}")
            else:
                # Some files are optional depending on interface
                if filename == "Dockerfile" and setting.interface == "function":
                    result.logs.append(f"Skipping {filename} (function interface)")
                    continue
                if filename in ["modal_app.py", "service.py", "bentofile.yaml"]:
                    # These are generated by adapter, warn but don't fail
                    result.logs.append(f"Warning: {filename} not found")
                    continue
                
                result.checks_failed.append(f"structure:{filename}")
                result.error = f"Required file not found: {filename}"
                all_found = False
        
        if all_found:
            result.logs.append("Structure check passed")
        
        return all_found
    
    def _check_execution(
        self,
        path: Path,
        setting: DeploymentSetting,
        result: ValidationResult,
    ) -> bool:
        """Run basic execution test."""
        result.logs.append("Running execution test...")
        
        # Try to run main.py with test input
        main_file = path / "main.py"
        if not main_file.exists():
            main_file = path / "src" / "main.py"
        
        if not main_file.exists():
            result.logs.append("No main.py found, skipping execution test")
            return True
        
        # Run with test input
        test_input = '{"test": true}'
        cmd = [sys.executable, str(main_file)]
        
        try:
            proc = subprocess.run(
                cmd,
                input=test_input,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(path),
            )
            
            # Check if it ran without crashing (exit code 0 or output produced)
            if proc.returncode == 0 or proc.stdout:
                result.checks_passed.append("execution:basic")
                result.logs.append("Execution test passed")
                return True
            else:
                result.checks_failed.append("execution:basic")
                result.error = f"Execution failed: {proc.stderr}"
                return False
                
        except subprocess.TimeoutExpired:
            result.checks_failed.append("execution:timeout")
            result.error = "Execution timed out"
            return False
        except Exception as e:
            result.checks_failed.append("execution:error")
            result.error = f"Execution error: {e}"
            return False

