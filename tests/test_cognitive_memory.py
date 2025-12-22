import unittest
from unittest.mock import MagicMock, patch
import shutil
import os
import json

from src.memory.types import WorkingMemory, Insight, InsightType
from src.memory.controller import CognitiveController
from src.memory.episodic import EpisodicStore

class TestCognitiveMemory(unittest.TestCase):
    def setUp(self):
        self.test_store_path = ".test_memory_store.json"
        # Clean up start
        if os.path.exists(self.test_store_path):
            os.remove(self.test_store_path)
            
    def tearDown(self):
        # Clean up end
        if os.path.exists(self.test_store_path):
            os.remove(self.test_store_path)

    def test_episodic_store(self):
        store = EpisodicStore(self.test_store_path)
        
        insight = Insight(
            content="Use pip install -r requirements.txt",
            insight_type=InsightType.BEST_PRACTICE,
            confidence=0.9,
            source_experiment_id="exp_1"
        )
        
        store.add_insight(insight)
        
        # Verify save
        self.assertTrue(os.path.exists(self.test_store_path))
        
        # Verify retrieve
        results = store.retrieve_relevant("how to install requirements")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, insight.content)
        
        # Verify persistence
        store2 = EpisodicStore(self.test_store_path)
        self.assertEqual(len(store2.insights), 1)

    @patch('src.memory.controller.LLMBackend')
    def test_controller_briefing(self, MockLLM):
        # Setup mocks
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.llm_completion.return_value = "how to fix import error"
        
        controller = CognitiveController(
            episodic_store_path=self.test_store_path,
            knowledge_search=None
        )
        
        # Add a relevant insight to store
        insight = Insight(
            content="Use absolute imports in src module",
            insight_type=InsightType.CRITICAL_ERROR,
            confidence=1.0,
            source_experiment_id="exp_prev"
        )
        controller.episodic.add_insight(insight)
        
        wm = WorkingMemory(
            current_goal="Fix imports",
            active_plan=["1. Check file"]
        )
        
        briefing = controller.prepare_briefing(wm, last_error="ImportError: No module named x")
        
        # Verify flow
        self.assertEqual(briefing.goal, "Fix imports")
        self.assertIn("Use absolute imports", briefing.insights[0])
        
        # Verify Meta-Cognition call
        mock_llm_instance.llm_completion.assert_called()

    @patch('src.memory.controller.LLMBackend')
    def test_insight_extraction(self, MockLLM):
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.llm_completion.return_value = json.dumps({
            "rule": "Always check numpy version",
            "type": "critical_error",
            "confidence": 0.85
        })
        
        controller = CognitiveController(episodic_store_path=self.test_store_path)
        
        # Mock result object
        mock_result = MagicMock()
        mock_result.run_had_error = True
        mock_result.error_details = "Numpy version mismatch"
        
        wm = WorkingMemory()
        
        new_wm, new_insight = controller.process_result(mock_result, wm)
        
        self.assertIsNotNone(new_insight)
        self.assertEqual(new_insight.content, "Always check numpy version")
        self.assertEqual(new_insight.insight_type, InsightType.CRITICAL_ERROR)
        
        # Verify it was added to store
        self.assertEqual(len(controller.episodic.insights), 1)

if __name__ == '__main__':
    unittest.main()

