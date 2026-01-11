"""
Tests for RepoMemory (using real LLM calls).

These tests validate that:
1. RepoMemory can infer a semantic model from a real repository
2. All claims are evidence-backed (quotes exist in actual files)
3. Memory updates correctly after code changes
4. Quality metrics are accurate

NOTE: These tests make real LLM API calls and cost money.
Run with: pytest tests/test_repo_memory.py -v -s
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import git
import pytest
from dotenv import load_dotenv

# Load environment variables (API keys) from .env file
load_dotenv()

from src.core.llm import LLMBackend
from src.repo_memory import RepoMemoryManager
from src.repo_memory.builders import build_repo_map, validate_evidence


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def llm():
    """Provide real LLM backend for tests."""
    return LLMBackend()


@pytest.fixture
def sample_repo(tmp_path):
    """
    Create a small but realistic test repository with actual code.
    
    The repo has:
    - A README describing what it does
    - A main entrypoint (main.py)
    - A core algorithm module (algorithm.py)
    - A config file (config.yaml)
    """
    repo_dir = tmp_path / "sample_repo"
    repo_dir.mkdir()
    
    # Initialize git repo
    repo = git.Repo.init(repo_dir)
    
    # Create README.md
    readme = repo_dir / "README.md"
    readme.write_text("""# Sample ML Pipeline

This repository implements a simple machine learning pipeline for classification.

## Features
- Data loading from CSV files
- Feature preprocessing with normalization
- Logistic regression classifier
- Model evaluation with accuracy metrics

## Usage
```bash
python main.py --data input.csv --output model.pkl
```

## Architecture
The pipeline follows a modular design:
1. `data_loader.py` - handles data ingestion
2. `preprocessor.py` - feature engineering
3. `classifier.py` - model training and prediction
4. `evaluator.py` - metrics computation
""")
    
    # Create main.py
    main_py = repo_dir / "main.py"
    main_py.write_text('''"""
Main entrypoint for the ML pipeline.

Usage:
    python main.py --data input.csv --output model.pkl
"""

import argparse
from pathlib import Path

from classifier import LogisticClassifier
from data_loader import load_csv_data
from evaluator import compute_accuracy
from preprocessor import normalize_features


def main():
    parser = argparse.ArgumentParser(description="ML Pipeline")
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to save model")
    args = parser.parse_args()
    
    # Load and preprocess data
    X, y = load_csv_data(args.data)
    X_normalized = normalize_features(X)
    
    # Train classifier
    model = LogisticClassifier(learning_rate=0.01, max_iter=1000)
    model.fit(X_normalized, y)
    
    # Evaluate
    predictions = model.predict(X_normalized)
    accuracy = compute_accuracy(y, predictions)
    print(f"Training accuracy: {accuracy:.4f}")
    
    # Save model
    model.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
''')
    
    # Create classifier.py with actual algorithm
    classifier_py = repo_dir / "classifier.py"
    classifier_py.write_text('''"""
Logistic regression classifier implementation.

Uses gradient descent optimization with configurable learning rate.
"""

import pickle
from typing import List

import numpy as np


class LogisticClassifier:
    """
    Binary logistic regression classifier.
    
    Uses sigmoid activation and cross-entropy loss.
    Optimized via batch gradient descent.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticClassifier":
        """
        Train the classifier using gradient descent.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
        
        Returns:
            self for method chaining
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for _ in range(self.max_iter):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_pred)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        linear_pred = np.dot(X, self.weights) + self.bias
        probabilities = self._sigmoid(linear_pred)
        return (probabilities >= 0.5).astype(int)
    
    def save(self, path: str) -> None:
        """Save model to pickle file."""
        with open(path, "wb") as f:
            pickle.dump({"weights": self.weights, "bias": self.bias}, f)
    
    def load(self, path: str) -> "LogisticClassifier":
        """Load model from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.weights = data["weights"]
        self.bias = data["bias"]
        return self
''')
    
    # Create data_loader.py
    data_loader_py = repo_dir / "data_loader.py"
    data_loader_py.write_text('''"""
Data loading utilities for CSV files.
"""

import csv
from typing import Tuple

import numpy as np


def load_csv_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from a CSV file.
    
    Assumes last column is the target variable.
    
    Args:
        path: Path to CSV file
        
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        data = [row for row in reader]
    
    data = np.array(data, dtype=float)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y
''')
    
    # Create preprocessor.py
    preprocessor_py = repo_dir / "preprocessor.py"
    preprocessor_py.write_text('''"""
Feature preprocessing utilities.
"""

import numpy as np


def normalize_features(X: np.ndarray) -> np.ndarray:
    """
    Normalize features to zero mean and unit variance.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        
    Returns:
        Normalized feature matrix
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    return (X - mean) / std
''')
    
    # Create evaluator.py
    evaluator_py = repo_dir / "evaluator.py"
    evaluator_py.write_text('''"""
Model evaluation utilities.
"""

import numpy as np


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy as a float in [0, 1]
    """
    return np.mean(y_true == y_pred)
''')
    
    # Create config.yaml
    config_yaml = repo_dir / "config.yaml"
    config_yaml.write_text("""# ML Pipeline Configuration

model:
  type: logistic_regression
  learning_rate: 0.01
  max_iterations: 1000

preprocessing:
  normalize: true
  
evaluation:
  metric: accuracy
""")
    
    # Commit all files
    repo.git.add("-A")
    repo.git.commit("-m", "Initial commit: ML pipeline implementation")
    
    return repo_dir


# ---------------------------------------------------------------------------
# Test: RepoMap (deterministic, no LLM)
# ---------------------------------------------------------------------------

def test_build_repo_map_deterministic(sample_repo):
    """Test that build_repo_map works without LLM and produces correct structure."""
    repo_map = build_repo_map(str(sample_repo))
    
    # Should have correct structure
    assert "repo_root" in repo_map
    assert "file_count" in repo_map
    assert "files" in repo_map
    assert "languages_by_extension" in repo_map
    assert "key_files" in repo_map
    assert "entrypoints" in repo_map
    
    # Should find our files
    assert repo_map["file_count"] >= 6
    assert "README.md" in repo_map["key_files"]
    assert "main.py" in repo_map["entrypoints"]
    
    # Should detect Python as primary language
    assert ".py" in repo_map["languages_by_extension"]


# ---------------------------------------------------------------------------
# Test: Bootstrap baseline model (real LLM)
# ---------------------------------------------------------------------------

def test_bootstrap_baseline_model_with_real_llm(sample_repo, llm):
    """
    Test that bootstrap_baseline_model produces evidence-backed memory.
    
    This test:
    1. Calls the real LLM to infer repo structure
    2. Validates that all claims have evidence in actual files
    3. Checks that summary and claims are non-empty
    """
    # Bootstrap the memory
    RepoMemoryManager.bootstrap_baseline_model(
        repo_root=str(sample_repo),
        llm=llm,
        seed_repo_path=str(sample_repo),
    )
    
    # Load and verify
    doc = RepoMemoryManager.load_from_worktree(str(sample_repo))
    assert doc is not None, "Memory file should exist"
    
    # Check structure
    assert doc.get("schema_version") == 1
    assert "repo_map" in doc
    assert "repo_model" in doc
    assert "quality" in doc
    
    # Check quality - evidence must pass
    quality = doc["quality"]
    assert quality["evidence_ok"] is True, f"Evidence validation failed: {quality.get('missing_evidence')}"
    assert quality["claim_count"] >= 1, "Should have at least one claim"
    
    # Check repo_model has content
    repo_model = doc["repo_model"]
    assert repo_model.get("summary"), "Summary should not be empty"
    assert len(repo_model.get("claims", [])) >= 1, "Should have at least one claim"
    
    # Verify evidence validation independently
    check = validate_evidence(str(sample_repo), repo_model)
    assert check.ok, f"Independent evidence check failed: {check.missing}"
    
    # Print summary for inspection
    print("\n=== Generated RepoMemory ===")
    print(f"Summary: {repo_model.get('summary', '')[:500]}")
    print(f"Claims: {len(repo_model.get('claims', []))}")
    for claim in repo_model.get("claims", [])[:5]:
        print(f"  - [{claim.get('kind')}] {claim.get('statement', '')[:100]}")


# ---------------------------------------------------------------------------
# Test: Update after experiment (real LLM)
# ---------------------------------------------------------------------------

def test_update_after_experiment_with_real_llm(sample_repo, llm):
    """
    Test that update_after_experiment correctly updates memory after code changes.
    
    This test:
    1. Bootstraps baseline memory
    2. Makes a code change (add new feature)
    3. Calls update_after_experiment
    4. Verifies updated memory is still evidence-backed
    5. Verifies experiment delta is recorded
    """
    repo = git.Repo(sample_repo)
    
    # Bootstrap baseline
    RepoMemoryManager.bootstrap_baseline_model(
        repo_root=str(sample_repo),
        llm=llm,
        seed_repo_path=str(sample_repo),
    )
    repo.git.add("-A")
    repo.git.commit("-m", "Add baseline memory")
    base_commit = repo.head.commit.hexsha
    
    # Create experiment branch and make changes
    repo.git.checkout("-b", "experiment-001")
    
    # Add a new feature file
    new_file = sample_repo / "cross_validator.py"
    new_file.write_text('''"""
Cross-validation utilities for model evaluation.
"""

import numpy as np
from typing import List, Tuple


def k_fold_split(X: np.ndarray, y: np.ndarray, k: int = 5) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split data into k folds for cross-validation.
    
    Args:
        X: Feature matrix
        y: Target labels
        k: Number of folds
        
    Returns:
        List of (X_train, X_val, y_train, y_val) tuples
    """
    n_samples = len(X)
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n_samples
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        
        folds.append((X[train_idx], X[val_idx], y[train_idx], y[val_idx]))
    
    return folds
''')
    
    # Modify main.py to use cross-validation
    main_py = sample_repo / "main.py"
    old_content = main_py.read_text()
    new_content = old_content.replace(
        "from evaluator import compute_accuracy",
        "from evaluator import compute_accuracy\nfrom cross_validator import k_fold_split"
    )
    main_py.write_text(new_content)
    
    repo.git.add("-A")
    repo.git.commit("-m", "Add cross-validation support")
    
    # Update memory
    RepoMemoryManager.update_after_experiment(
        repo_root=str(sample_repo),
        llm=llm,
        branch_name="experiment-001",
        parent_branch_name="main",
        base_commit_sha=base_commit,
        solution_spec="Add k-fold cross-validation for better model evaluation",
        run_result={"score": 0.85, "run_had_error": False},
    )
    
    # Load and verify
    doc = RepoMemoryManager.load_from_worktree(str(sample_repo))
    assert doc is not None
    
    # Evidence must still be valid
    quality = doc["quality"]
    assert quality["evidence_ok"] is True, f"Evidence validation failed: {quality.get('missing_evidence')}"
    
    # Should have experiment recorded
    experiments = doc.get("experiments", [])
    assert len(experiments) >= 1, "Should have at least one experiment recorded"
    
    exp = experiments[-1]
    assert exp["branch"] == "experiment-001"
    assert exp["parent_branch"] == "main"
    assert "cross_validator.py" in exp["changed_files"]
    
    # Verify evidence independently
    repo_model = doc["repo_model"]
    check = validate_evidence(str(sample_repo), repo_model)
    assert check.ok, f"Independent evidence check failed: {check.missing}"
    
    print("\n=== Updated RepoMemory ===")
    print(f"Summary: {repo_model.get('summary', '')[:500]}")
    print(f"Claims: {len(repo_model.get('claims', []))}")
    print(f"Experiments recorded: {len(experiments)}")


# ---------------------------------------------------------------------------
# Test: Evidence validation catches hallucinations
# ---------------------------------------------------------------------------

def test_validate_evidence_catches_invalid_quotes():
    """Test that evidence validation correctly rejects hallucinated quotes."""
    with tempfile.TemporaryDirectory() as tmp:
        # Create a simple file
        Path(tmp, "foo.py").write_text("def hello():\n    return 'world'\n")
        
        # Valid claim
        valid_model = {
            "claims": [{
                "kind": "algorithm",
                "statement": "Has a hello function",
                "evidence": [{"path": "foo.py", "quote": "def hello():"}]
            }]
        }
        check = validate_evidence(tmp, valid_model)
        assert check.ok, "Valid quote should pass"
        
        # Invalid claim (hallucinated quote)
        invalid_model = {
            "claims": [{
                "kind": "algorithm",
                "statement": "Has a goodbye function",
                "evidence": [{"path": "foo.py", "quote": "def goodbye():"}]
            }]
        }
        check = validate_evidence(tmp, invalid_model)
        assert not check.ok, "Hallucinated quote should fail"
        assert len(check.missing) == 1
        
        # Invalid claim (file doesn't exist)
        missing_file_model = {
            "claims": [{
                "kind": "algorithm",
                "statement": "Has bar module",
                "evidence": [{"path": "bar.py", "quote": "anything"}]
            }]
        }
        check = validate_evidence(tmp, missing_file_model)
        assert not check.ok, "Missing file should fail"


# ---------------------------------------------------------------------------
# Test: Render brief for prompts
# ---------------------------------------------------------------------------

def test_render_brief_produces_usable_prompt(sample_repo, llm):
    """Test that render_brief produces a usable prompt summary."""
    # Bootstrap
    RepoMemoryManager.bootstrap_baseline_model(
        repo_root=str(sample_repo),
        llm=llm,
        seed_repo_path=str(sample_repo),
    )
    
    doc = RepoMemoryManager.load_from_worktree(str(sample_repo))
    brief = RepoMemoryManager.render_brief(doc)
    
    # Should have key sections
    assert "# Repo Memory" in brief
    assert "## Repo Summary" in brief
    assert "## Entrypoints" in brief
    assert "## Where to edit" in brief
    assert "## Key claims" in brief
    
    # Should be bounded
    assert len(brief) <= 10000, "Brief should be bounded for prompt context"
    
    print("\n=== Rendered Brief ===")
    print(brief[:2000])


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
