# =============================================================================
# Objective - Complete Problem Specification for Expert Agent
# =============================================================================
#
# An Objective captures EVERYTHING needed to solve a problem:
# - What to achieve (goal description)
# - What data is available (data files, context files)
# - How to evaluate (success criteria, evaluator)
# - Constraints (time, resources, etc.)
# - Source and metadata (benchmark name, problem ID)
#
# This is the single source of truth for a problem specification.
#
# =============================================================================

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum


class ObjectiveType(Enum):
    """Categories of objectives for better KG retrieval and decision-making."""
    ML_TRAINING = "ml_training"          # Fine-tuning, training models
    ML_INFERENCE = "ml_inference"        # Running inference, predictions
    DATA_PROCESSING = "data_processing"  # ETL, data cleaning, transformation
    CODE_GENERATION = "code_generation"  # Writing new code
    BUG_FIX = "bug_fix"                  # Fixing existing code
    OPTIMIZATION = "optimization"        # Performance tuning
    RESEARCH = "research"                # Exploring, experimenting
    BENCHMARK = "benchmark"              # From MLE/ALE benchmark
    GENERIC = "generic"                  # Default/unknown


@dataclass
class DataFile:
    """
    Reference to a data file needed for the objective.
    
    Attributes:
        path: Path to the file (absolute or relative to working_dir)
        description: What this file contains and how to use it
        file_type: Type hint (csv, json, parquet, txt, etc.)
        is_input: True if input file, False if expected output
    """
    path: str
    description: str = ""
    file_type: str = ""
    is_input: bool = True
    
    def exists(self, working_dir: Optional[str] = None) -> bool:
        """Check if file exists."""
        p = Path(self.path)
        if not p.is_absolute() and working_dir:
            p = Path(working_dir) / p
        return p.exists()
    
    def read_preview(self, working_dir: Optional[str] = None, max_lines: int = 10) -> str:
        """Read first N lines for context."""
        p = Path(self.path)
        if not p.is_absolute() and working_dir:
            p = Path(working_dir) / p
        
        if not p.exists():
            return f"[File not found: {p}]"
        
        try:
            with open(p, 'r') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"... ({p.stat().st_size} bytes total)")
                        break
                    lines.append(line.rstrip())
                return "\n".join(lines)
        except Exception as e:
            return f"[Error reading file: {e}]"


@dataclass
class Objective:
    """
    Complete problem specification for the Expert agent.
    
    This is NOT just a goal string - it's the full context:
    - What to achieve
    - What data is available
    - How to evaluate success
    - What constraints apply
    
    Attributes:
        description: Natural language description of what to achieve
        objective_type: Category for KG retrieval (ML_TRAINING, BUG_FIX, etc.)
        
        data_files: List of data files (inputs and expected outputs)
        context_files: Additional context files (README, docs, examples)
        working_dir: Working directory for the problem
        
        success_criteria: How to know if we succeeded (for LLM evaluator)
        expected_output_format: Expected output format description
        
        constraints: Resource/time constraints
        additional_context: Extra instructions, tips, domain knowledge
        
        source: Where this objective came from (user, benchmark, automated)
        metadata: Extra info (benchmark name, problem ID, competition, etc.)
        
        logs_dir: Where to save experiment logs
    """
    # === Core ===
    description: str
    objective_type: ObjectiveType = ObjectiveType.GENERIC
    
    # === Data Context ===
    data_files: List[DataFile] = field(default_factory=list)
    context_files: List[str] = field(default_factory=list)
    working_dir: str = ""
    
    # === Evaluation ===
    success_criteria: str = ""
    expected_output_format: str = ""
    evaluator: str = "no_score"
    evaluator_params: Dict[str, Any] = field(default_factory=dict)
    
    # === Constraints ===
    constraints: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 300
    max_iterations: int = 10
    
    # === Extra Context ===
    additional_context: str = ""
    
    # === Source & Metadata ===
    source: str = "user"  # "user", "benchmark", "mle", "ale", "automated"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # === Logging ===
    logs_dir: str = ""
    
    # === Internal ===
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __str__(self) -> str:
        """Return description for backward compatibility."""
        return self.description
    
    @classmethod
    def from_string(cls, description: str) -> "Objective":
        """Create minimal Objective from a plain string."""
        # Infer type from keywords
        desc_lower = description.lower()
        
        obj_type = ObjectiveType.GENERIC
        if any(kw in desc_lower for kw in ["train", "fine-tune", "finetune", "lora"]):
            obj_type = ObjectiveType.ML_TRAINING
        elif any(kw in desc_lower for kw in ["predict", "inference", "classify"]):
            obj_type = ObjectiveType.ML_INFERENCE
        elif any(kw in desc_lower for kw in ["fix", "bug", "error", "issue"]):
            obj_type = ObjectiveType.BUG_FIX
        elif any(kw in desc_lower for kw in ["optimize", "performance", "speed"]):
            obj_type = ObjectiveType.OPTIMIZATION
        elif any(kw in desc_lower for kw in ["data", "process", "etl", "clean"]):
            obj_type = ObjectiveType.DATA_PROCESSING
        
        return cls(description=description, objective_type=obj_type)
    
    @classmethod
    def from_mle_problem(
        cls,
        problem_id: str,
        description: str,
        data_dir: str,
        **kwargs
    ) -> "Objective":
        """
        Create Objective from MLE benchmark problem.
        
        Automatically discovers data files in data_dir.
        """
        data_files = []
        
        if data_dir and os.path.exists(data_dir):
            for fname in os.listdir(data_dir):
                fpath = os.path.join(data_dir, fname)
                if os.path.isfile(fpath):
                    ext = os.path.splitext(fname)[1].lower()
                    is_input = not fname.startswith("expected") and not fname.startswith("output")
                    data_files.append(DataFile(
                        path=fpath,
                        description=f"{'Input' if is_input else 'Expected output'}: {fname}",
                        file_type=ext.lstrip('.'),
                        is_input=is_input,
                    ))
        
        return cls(
            description=description,
            objective_type=ObjectiveType.BENCHMARK,
            data_files=data_files,
            working_dir=data_dir,
            source="mle",
            metadata={"problem_id": problem_id, "benchmark": "mle", **kwargs},
            logs_dir=os.path.join(data_dir, "logs") if data_dir else "",
        )
    
    def to_kg_query(self) -> str:
        """Format objective for KG search query."""
        type_context = f"[{self.objective_type.value}] " if self.objective_type != ObjectiveType.GENERIC else ""
        return f"{type_context}{self.description}"
    
    def get_data_context(self, max_lines_per_file: int = 5) -> str:
        """
        Generate data context string for the agent.
        
        Includes file list with previews.
        """
        if not self.data_files:
            return ""
        
        lines = ["## Data Files"]
        
        for df in self.data_files:
            role = "INPUT" if df.is_input else "OUTPUT"
            lines.append(f"\n### {df.path} [{role}]")
            if df.description:
                lines.append(f"{df.description}")
            if df.file_type:
                lines.append(f"Type: {df.file_type}")
            
            # Preview for input files
            if df.is_input and max_lines_per_file > 0:
                preview = df.read_preview(self.working_dir, max_lines_per_file)
                lines.append(f"Preview:\n```\n{preview}\n```")
        
        return "\n".join(lines)
    
    def render(self) -> str:
        """
        Render complete objective as text for agent context.
        
        This is what the agent sees.
        """
        lines = []
        
        # Goal
        lines.append("# OBJECTIVE")
        lines.append(self.description)
        lines.append("")
        
        # Type and source
        lines.append(f"**Type:** {self.objective_type.value}")
        lines.append(f"**Source:** {self.source}")
        if self.metadata:
            lines.append(f"**Metadata:** {self.metadata}")
        lines.append("")
        
        # Data files
        if self.data_files:
            lines.append(self.get_data_context())
            lines.append("")
        
        # Success criteria
        if self.success_criteria:
            lines.append("## Success Criteria")
            lines.append(self.success_criteria)
            lines.append("")
        
        # Expected output
        if self.expected_output_format:
            lines.append("## Expected Output")
            lines.append(self.expected_output_format)
            lines.append("")
        
        # Constraints
        if self.constraints:
            lines.append("## Constraints")
            for k, v in self.constraints.items():
                lines.append(f"- {k}: {v}")
            lines.append("")
        
        # Additional context
        if self.additional_context:
            lines.append("## Additional Context")
            lines.append(self.additional_context)
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "description": self.description,
            "objective_type": self.objective_type.value,
            "data_files": [
                {"path": df.path, "description": df.description, "file_type": df.file_type, "is_input": df.is_input}
                for df in self.data_files
            ],
            "context_files": self.context_files,
            "working_dir": self.working_dir,
            "success_criteria": self.success_criteria,
            "expected_output_format": self.expected_output_format,
            "evaluator": self.evaluator,
            "evaluator_params": self.evaluator_params,
            "constraints": self.constraints,
            "timeout": self.timeout,
            "max_iterations": self.max_iterations,
            "additional_context": self.additional_context,
            "source": self.source,
            "metadata": self.metadata,
            "logs_dir": self.logs_dir,
            "created_at": self.created_at.isoformat(),
        }
    
    def get_logs_path(self, experiment_id: str) -> str:
        """Get path for experiment logs."""
        if not self.logs_dir:
            return ""
        
        os.makedirs(self.logs_dir, exist_ok=True)
        return os.path.join(self.logs_dir, f"{experiment_id}.log")


# =============================================================================
# Factory for creating Objectives from various sources
# =============================================================================

class ObjectiveFactory:
    """Factory for creating Objective objects from various sources."""
    
    @staticmethod
    def from_cli(
        goal: str,
        data_dir: Optional[str] = None,
        context_files: Optional[List[str]] = None,
        evaluator: str = "no_score",
        evaluator_params: Optional[Dict[str, Any]] = None,
        timeout: int = 300,
        max_iterations: int = 10,
        logs_dir: Optional[str] = None,
    ) -> Objective:
        """Create Objective from CLI arguments."""
        data_files = []
        
        if data_dir and os.path.exists(data_dir):
            for fname in os.listdir(data_dir):
                fpath = os.path.join(data_dir, fname)
                if os.path.isfile(fpath):
                    ext = os.path.splitext(fname)[1].lower()
                    data_files.append(DataFile(
                        path=fpath,
                        file_type=ext.lstrip('.'),
                        is_input=True,
                    ))
        
        return Objective(
            description=goal,
            data_files=data_files,
            context_files=context_files or [],
            working_dir=data_dir or "",
            evaluator=evaluator,
            evaluator_params=evaluator_params or {},
            timeout=timeout,
            max_iterations=max_iterations,
            source="user",
            logs_dir=logs_dir or "",
        )
    
    @staticmethod
    def from_benchmark(
        problem_id: str,
        description: str,
        data_dir: str,
        benchmark_name: str = "generic",
        evaluator: str = "no_score",
        evaluator_params: Optional[Dict[str, Any]] = None,
        **metadata
    ) -> Objective:
        """Create Objective from benchmark problem."""
        return Objective.from_mle_problem(
            problem_id=problem_id,
            description=description,
            data_dir=data_dir,
            benchmark=benchmark_name,
            evaluator=evaluator,
            evaluator_params=evaluator_params or {},
            **metadata,
        )

