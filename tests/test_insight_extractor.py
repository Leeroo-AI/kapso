"""
Tests for Insight Extractor
===========================

Tests the LLM-based insight extraction from errors and successes.
"""

import pytest
from unittest.mock import Mock, patch

from kapso.execution.memories.experiment_memory.insight_extractor import (
    InsightExtractor,
    ExtractedInsight,
    InsightType,
)


class TestExtractedInsight:
    """Tests for ExtractedInsight dataclass."""
    
    def test_to_formatted_string(self):
        """Test formatting insight as string."""
        insight = ExtractedInsight(
            lesson="Always install dependencies before running",
            trigger_conditions="When importing external libraries",
            suggested_fix="Run pip install -r requirements.txt",
            confidence=0.9,
            insight_type=InsightType.CRITICAL_ERROR,
            original_text="ModuleNotFoundError: No module named 'peft'",
            tags=["dependency", "import"],
        )
        
        formatted = insight.to_formatted_string()
        assert "Always install dependencies" in formatted
        assert "When importing external libraries" in formatted
        assert "pip install" in formatted


class TestInsightExtractor:
    """Tests for InsightExtractor class."""
    
    def test_fallback_error_insight_module_not_found(self):
        """Test fallback insight for ModuleNotFoundError."""
        extractor = InsightExtractor()
        
        insight = extractor._fallback_error_insight(
            "ModuleNotFoundError: No module named 'peft'"
        )
        
        assert insight.insight_type == InsightType.CRITICAL_ERROR
        assert "missing_module" in insight.tags
        assert insight.confidence == 0.3
    
    def test_fallback_error_insight_cuda(self):
        """Test fallback insight for CUDA errors."""
        extractor = InsightExtractor()
        
        insight = extractor._fallback_error_insight(
            "CUDA error: device-side assert triggered"
        )
        
        assert insight.insight_type == InsightType.CRITICAL_ERROR
        assert "cuda_error" in insight.tags
    
    def test_fallback_error_insight_oom(self):
        """Test fallback insight for OOM errors."""
        extractor = InsightExtractor()
        
        insight = extractor._fallback_error_insight(
            "RuntimeError: out of memory"
        )
        
        assert insight.insight_type == InsightType.CRITICAL_ERROR
        assert "memory_error" in insight.tags
    
    def test_fallback_success_insight(self):
        """Test fallback insight for success."""
        extractor = InsightExtractor()
        
        insight = extractor._fallback_success_insight(
            "Great job! The model achieved 95% accuracy.",
            score=0.95,
        )
        
        assert insight.insight_type == InsightType.BEST_PRACTICE
        assert "success" in insight.tags
        assert insight.confidence == 0.95 * 0.5
    
    def test_fallback_truncates_long_messages(self):
        """Test that fallback truncates long error messages."""
        extractor = InsightExtractor()
        
        long_error = "Error: " + "x" * 1000
        insight = extractor._fallback_error_insight(long_error)
        
        # Lesson should be truncated to 500 chars
        assert len(insight.lesson) <= 510  # "Error: " + 500 chars
    
    @patch.object(InsightExtractor, '_call_llm')
    def test_extract_from_error_with_mock_llm(self, mock_llm):
        """Test error extraction with mocked LLM."""
        mock_llm.return_value = '''
        {
            "lesson": "The peft library must be installed for LoRA operations",
            "trigger_conditions": "When using LoraConfig or get_peft_model",
            "suggested_fix": "Run pip install peft before running the script",
            "confidence": 0.9,
            "tags": ["peft", "lora", "dependency"]
        }
        '''
        
        extractor = InsightExtractor()
        insight = extractor.extract_from_error(
            error_message="ModuleNotFoundError: No module named 'peft'",
            goal="Fine-tune LLaMA with LoRA",
        )
        
        assert insight.insight_type == InsightType.CRITICAL_ERROR
        assert "peft" in insight.lesson.lower()
        assert insight.confidence == 0.9
        assert "peft" in insight.tags
    
    @patch.object(InsightExtractor, '_call_llm')
    def test_extract_from_success_with_mock_llm(self, mock_llm):
        """Test success extraction with mocked LLM."""
        mock_llm.return_value = '''
        {
            "lesson": "Using gradient checkpointing reduces memory usage significantly",
            "trigger_conditions": "When training large models with limited GPU memory",
            "suggested_fix": "Enable gradient_checkpointing=True in training config",
            "confidence": 0.85,
            "tags": ["memory", "optimization", "training"]
        }
        '''
        
        extractor = InsightExtractor()
        insight = extractor.extract_from_success(
            feedback="Excellent! Model trained successfully with gradient checkpointing.",
            score=0.92,
            goal="Train large model on single GPU",
        )
        
        assert insight.insight_type == InsightType.BEST_PRACTICE
        assert "gradient checkpointing" in insight.lesson.lower()
        assert insight.confidence == 0.85
    
    @patch.object(InsightExtractor, '_call_llm')
    def test_extract_handles_llm_failure(self, mock_llm):
        """Test that extraction falls back gracefully on LLM failure."""
        mock_llm.side_effect = Exception("LLM API error")
        
        extractor = InsightExtractor()
        insight = extractor.extract_from_error(
            error_message="SyntaxError: invalid syntax",
            goal="Fix code",
        )
        
        # Should return fallback insight
        assert insight.insight_type == InsightType.CRITICAL_ERROR
        assert "syntax_error" in insight.tags
        assert insight.confidence == 0.3
    
    @patch.object(InsightExtractor, '_call_llm')
    def test_extract_handles_invalid_json(self, mock_llm):
        """Test that extraction handles invalid JSON response."""
        mock_llm.return_value = "This is not valid JSON"
        
        extractor = InsightExtractor()
        insight = extractor.extract_from_error(
            error_message="TypeError: unsupported operand",
            goal="Fix type error",
        )
        
        # Should return fallback insight
        assert insight.insight_type == InsightType.CRITICAL_ERROR
        assert "type_error" in insight.tags


class TestExperimentHistoryStoreWithInsights:
    """Tests for ExperimentHistoryStore insight integration."""
    
    def test_store_creates_record_with_insight_fields(self, tmp_path):
        """Test that store creates records with insight fields."""
        from kapso.execution.memories.experiment_memory import ExperimentHistoryStore
        
        json_path = str(tmp_path / "experiments.json")
        store = ExperimentHistoryStore(
            json_path=json_path,
            goal="Test goal",
            enable_insights=False,  # Disable for this test
        )
        
        # Create mock node
        node = Mock()
        node.node_id = 1
        node.solution = "print('hello')"
        node.score = 0.8
        node.feedback = "Good job"
        node.branch_name = "exp-1"
        node.had_error = False
        node.error_message = ""
        
        store.add_experiment(node)
        
        assert len(store.experiments) == 1
        record = store.experiments[0]
        assert record.node_id == 1
        assert record.insight is None  # Insights disabled
    
    def test_store_backward_compatibility(self, tmp_path):
        """Test that store loads old records without insight fields."""
        import json
        from kapso.execution.memories.experiment_memory import ExperimentHistoryStore
        
        json_path = str(tmp_path / "experiments.json")
        
        # Write old-format record (without insight fields)
        old_record = {
            "node_id": 1,
            "solution": "print('hello')",
            "score": 0.8,
            "feedback": "Good job",
            "branch_name": "exp-1",
            "had_error": False,
            "error_message": "",
            "timestamp": "2024-01-01T00:00:00",
        }
        with open(json_path, 'w') as f:
            json.dump([old_record], f)
        
        # Load store
        store = ExperimentHistoryStore(
            json_path=json_path,
            enable_insights=False,
        )
        
        assert len(store.experiments) == 1
        record = store.experiments[0]
        assert record.node_id == 1
        assert record.insight is None
        assert record.insight_tags == []
    
    def test_get_experiments_with_insights(self, tmp_path):
        """Test filtering experiments by insights."""
        from kapso.execution.memories.experiment_memory import ExperimentHistoryStore, ExperimentRecord
        
        json_path = str(tmp_path / "experiments.json")
        store = ExperimentHistoryStore(
            json_path=json_path,
            enable_insights=False,
        )
        
        # Add records directly
        store.experiments = [
            ExperimentRecord(
                node_id=1, solution="s1", score=0.5, feedback="f1",
                branch_name="b1", had_error=False, error_message="",
                timestamp="t1", insight=None,
            ),
            ExperimentRecord(
                node_id=2, solution="s2", score=0.8, feedback="f2",
                branch_name="b2", had_error=False, error_message="",
                timestamp="t2", insight="Lesson 2", insight_type="best_practice",
                insight_confidence=0.9, insight_tags=["tag1"],
            ),
            ExperimentRecord(
                node_id=3, solution="s3", score=None, feedback="f3",
                branch_name="b3", had_error=True, error_message="Error",
                timestamp="t3", insight="Lesson 3", insight_type="critical_error",
                insight_confidence=0.7, insight_tags=["tag2"],
            ),
        ]
        
        # Get all insights
        with_insights = store.get_experiments_with_insights(k=10)
        assert len(with_insights) == 2
        
        # Get only best practices
        best_practices = store.get_experiments_with_insights(k=10, insight_type="best_practice")
        assert len(best_practices) == 1
        assert best_practices[0].node_id == 2
        
        # Get only errors
        errors = store.get_experiments_with_insights(k=10, insight_type="critical_error")
        assert len(errors) == 1
        assert errors[0].node_id == 3
