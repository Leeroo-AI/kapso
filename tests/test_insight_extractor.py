"""Tests for the single-path insight extractor and its store integration.

The extraction contract: one `extract()` call per finished node — no score
threshold, no success/error branching — sourced from the implementor's
technical_difficulties + the judge's feedback, with the insight_type
classified by the model in its JSON output.
"""

from unittest.mock import Mock, patch

from kapso.execution.memories.experiment_memory.insight_extractor import (
    InsightExtractor,
    InsightType,
    ExtractedInsight,
)


class TestExtractedInsight:
    def test_to_formatted_string(self):
        insight = ExtractedInsight(
            lesson="Always pin dependency versions",
            trigger_conditions="When importing external libraries",
            suggested_fix="Use a lockfile",
            confidence=0.8,
            insight_type=InsightType.BEST_PRACTICE,
            original_text="original",
            tags=["dependency", "import"],
        )
        formatted = insight.to_formatted_string()
        assert "Always pin dependency versions" in formatted
        assert "When importing external libraries" in formatted


class TestInsightExtractor:
    @patch.object(InsightExtractor, "_call_llm")
    def test_extract_takes_type_from_model_json(self, mock_llm):
        mock_llm.return_value = """
        {
            "lesson": "Base models have untrained special-token embeddings",
            "trigger_conditions": "When fine-tuning a base model for tool calling",
            "suggested_fix": "Train embedding rows or re-init special tokens",
            "confidence": 0.9,
            "insight_type": "critical_error",
            "tags": ["special-tokens", "lora", "base-model"]
        }
        """
        extractor = InsightExtractor()
        insight = extractor.extract(
            technical_difficulties="LoRA froze embeddings; control tokens garbage",
            feedback="Recovered via full FT",
            score=0.96,
            goal="Post-train for BFCL",
        )
        assert insight.insight_type == InsightType.CRITICAL_ERROR
        assert "special-token" in insight.lesson.lower()
        assert insight.confidence == 0.9

    @patch.object(InsightExtractor, "_call_llm")
    def test_extract_runs_for_low_and_missing_scores(self, mock_llm):
        # No threshold: a failed, unscored node still extracts.
        mock_llm.return_value = """
        {"lesson": "L", "trigger_conditions": "T", "suggested_fix": "F",
         "confidence": 0.5, "insight_type": "best_practice", "tags": []}
        """
        extractor = InsightExtractor()
        insight = extractor.extract(
            technical_difficulties="died at step 100",
            feedback="",
            score=None,
            goal="g",
        )
        assert insight.insight_type == InsightType.BEST_PRACTICE
        assert mock_llm.called

    @patch.object(InsightExtractor, "_call_llm")
    def test_prompt_carries_difficulties_and_feedback(self, mock_llm):
        mock_llm.return_value = (
            '{"lesson": "L", "trigger_conditions": "T", "suggested_fix": "F",'
            ' "confidence": 0.5, "insight_type": "best_practice", "tags": []}'
        )
        extractor = InsightExtractor()
        extractor.extract(
            technical_difficulties="OOM at batch 16",
            feedback="promoted at 0.92",
            score=0.92,
            goal="post-train",
        )
        prompt = mock_llm.call_args[0][0]
        assert "OOM at batch 16" in prompt
        assert "promoted at 0.92" in prompt

    @patch.object(InsightExtractor, "_call_llm")
    def test_llm_failure_falls_back_to_source_head(self, mock_llm):
        mock_llm.side_effect = Exception("LLM API error")
        extractor = InsightExtractor()
        insight = extractor.extract(
            technical_difficulties="SyntaxError: invalid syntax in train.py",
            feedback="",
            score=None,
            goal="g",
        )
        assert "SyntaxError" in insight.lesson
        assert insight.confidence == 0.3
        assert "fallback" in insight.tags

    @patch.object(InsightExtractor, "_call_llm")
    def test_invalid_json_falls_back(self, mock_llm):
        mock_llm.return_value = "This is not valid JSON"
        extractor = InsightExtractor()
        insight = extractor.extract(
            technical_difficulties="difficulty text",
            feedback="",
            score=0.5,
            goal="g",
        )
        assert insight.lesson == "difficulty text"
        assert "fallback" in insight.tags

    @patch.object(InsightExtractor, "_call_llm")
    def test_unknown_insight_type_falls_back(self, mock_llm):
        mock_llm.return_value = (
            '{"lesson": "L", "trigger_conditions": "T", "suggested_fix": "F",'
            ' "confidence": 0.5, "insight_type": "nonsense", "tags": []}'
        )
        extractor = InsightExtractor()
        insight = extractor.extract(
            technical_difficulties="src",
            feedback="",
            score=0.5,
            goal="g",
        )
        assert "fallback" in insight.tags


class TestExperimentHistoryStoreWithInsights:
    def _node(self, **overrides):
        node = Mock()
        node.node_id = overrides.get("node_id", 1)
        node.solution = overrides.get("solution", "print('hello')")
        node.score = overrides.get("score", 0.8)
        node.feedback = overrides.get("feedback", "Good job")
        node.branch_name = overrides.get("branch_name", "exp-1")
        node.had_error = overrides.get("had_error", False)
        node.error_message = overrides.get("error_message", "")
        node.technical_difficulties = overrides.get(
            "technical_difficulties", "hit an OOM, fixed with batch 8"
        )
        return node

    def test_record_carries_technical_difficulties(self, tmp_path):
        from kapso.execution.memories.experiment_memory import (
            ExperimentHistoryStore,
        )

        store = ExperimentHistoryStore(
            json_path=str(tmp_path / "experiments.json"),
            goal="Test goal",
            enable_insights=False,
        )
        store.add_experiment(self._node())
        record = store.experiments[0]
        assert record.technical_difficulties == "hit an OOM, fixed with batch 8"

    def test_extraction_runs_unconditionally_when_enabled(self, tmp_path):
        from kapso.execution.memories.experiment_memory import (
            ExperimentHistoryStore,
        )

        store = ExperimentHistoryStore(
            json_path=str(tmp_path / "experiments.json"),
            goal="Test goal",
            enable_insights=True,
        )
        fake = ExtractedInsight(
            lesson="L",
            trigger_conditions="T",
            suggested_fix="F",
            confidence=0.9,
            insight_type=InsightType.CRITICAL_ERROR,
            original_text="o",
            tags=["t"],
        )
        with patch.object(
            InsightExtractor, "extract", return_value=fake
        ) as mock_extract:
            # A LOW-scoring node still extracts (old threshold was 0.7).
            store.add_experiment(self._node(node_id=1, score=0.1))
            # An unscored node still extracts.
            store.add_experiment(self._node(node_id=2, score=None))
        assert mock_extract.call_count == 2
        assert store.experiments[0].insight_type == "critical_error"
        kwargs = mock_extract.call_args.kwargs
        assert kwargs["technical_difficulties"] == (
            "hit an OOM, fixed with batch 8"
        )

    def test_get_experiments_with_insights_filters_by_type(self, tmp_path):
        from kapso.execution.memories.experiment_memory import (
            ExperimentHistoryStore,
            ExperimentRecord,
        )

        store = ExperimentHistoryStore(
            json_path=str(tmp_path / "experiments.json"),
            enable_insights=False,
        )
        store.experiments = [
            ExperimentRecord(
                node_id=1, solution="s1", score=0.5, feedback="f1",
                branch_name="b1", had_error=False, error_message="",
                timestamp="t1", insight=None,
            ),
            ExperimentRecord(
                node_id=2, solution="s2", score=0.8, feedback="f2",
                branch_name="b2", had_error=False, error_message="",
                timestamp="t2", insight="Lesson 2",
                insight_type="best_practice",
                insight_confidence=0.9, insight_tags=["tag1"],
            ),
            ExperimentRecord(
                node_id=3, solution="s3", score=None, feedback="f3",
                branch_name="b3", had_error=True, error_message="Error",
                timestamp="t3", insight="Lesson 3",
                insight_type="critical_error",
                insight_confidence=0.7, insight_tags=["tag2"],
            ),
        ]
        assert len(store.get_experiments_with_insights(k=10)) == 2
        best = store.get_experiments_with_insights(
            k=10, insight_type="best_practice"
        )
        assert [r.node_id for r in best] == [2]
        errors = store.get_experiments_with_insights(
            k=10, insight_type="critical_error"
        )
        assert [r.node_id for r in errors] == [3]
