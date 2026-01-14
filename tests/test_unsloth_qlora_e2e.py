"""
E2E Test - Unsloth QLoRA Fine-Tuning

Tests that Tinkerer.evolve() can fine-tune LLMs using QLoRA with Unsloth,
retrieving Workflow:Unslothai_Unsloth_QLoRA_Finetuning from the KG.

Uses Claude Code via AWS Bedrock for code generation.

Prerequisites:
    ./start_infra.sh --wiki-dir data/wikis_llm_finetuning
    export OPENAI_API_KEY=...            # For LLM Judge + embeddings
    export AWS_BEARER_TOKEN_BEDROCK=...  # For Claude Code via Bedrock

To run:
    PYTHONPATH=. python tests/test_unsloth_qlora_e2e.py
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-5s │ %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

for noisy in ['httpx', 'httpcore', 'urllib3', 'openai', 'LiteLLM', 'litellm', 'neo4j', 'weaviate']:
    logging.getLogger(noisy).setLevel(logging.WARNING)


# =============================================================================
# Test Configuration
# =============================================================================

TEST_CONFIG = {
    "goal": """
Fine-tune a language model using QLoRA with Unsloth.
Create a Python script that:
1. Loads a pre-trained model using FastLanguageModel.from_pretrained() with 4-bit quantization
2. Configures LoRA adapters with get_peft_model() targeting attention and MLP layers
3. Formats a sample dataset with the appropriate chat template
4. Sets up SFTTrainer with training arguments
5. Runs the training loop
6. Saves the trained LoRA adapter

Use 'unsloth/Llama-3.2-1B' model. Create a minimal dummy dataset.
Print status messages after each step.
""",
    "evaluator_criteria": """
Code implements Unsloth QLoRA fine-tuning with:
- FastLanguageModel.from_pretrained() with load_in_4bit=True
- get_peft_model() with LoRA config (r, lora_alpha, target_modules)
- Dataset formatted with chat template
- SFTTrainer with TrainingArguments
- trainer.train() called
- Model saved with save_pretrained()
""",
    "max_iterations": 6,
    "coding_agent": "claude_code",
    "language": "python",
    "main_file": "main.py",
    "timeout": 300,
    "mode": "COGNITIVE",
    "evaluator": "llm_judge",
    "evaluator_params": {
        "criteria": """Code implements Unsloth QLoRA fine-tuning with FastLanguageModel, get_peft_model, SFTTrainer, and save_pretrained.""",
        "model": "gpt-4o-mini",
        "scale": 10,
    },
    "stop_condition": "threshold",
    "stop_condition_params": {"threshold": 0.7},
}

# Bedrock model config
BEDROCK_MODEL = "us.anthropic.claude-opus-4-5-20251101-v1:0"


# =============================================================================
# Bedrock Config Patch
# =============================================================================

def patch_claude_code_for_bedrock():
    """
    Patch CodingAgentFactory to use Bedrock for claude_code agent.
    
    This modifies the agent_specific defaults to enable Bedrock mode.
    """
    from src.execution.coding_agents.factory import CodingAgentFactory
    
    # Get current claude_code config
    if "claude_code" in CodingAgentFactory._agent_configs:
        config = CodingAgentFactory._agent_configs["claude_code"]
        
        # Update model to Bedrock model ID
        config["default_model"] = BEDROCK_MODEL
        config["default_debug_model"] = BEDROCK_MODEL
        
        # Add Bedrock settings to agent_specific
        config["agent_specific"]["use_bedrock"] = True
        config["agent_specific"]["aws_region"] = os.environ.get("AWS_REGION", "us-east-1")
        
        logger.info(f"Patched claude_code for Bedrock: {BEDROCK_MODEL}")


# =============================================================================
# Main Test
# =============================================================================

def run_test():
    """Run Unsloth QLoRA test using Tinkerer.evolve()."""
    from src.tinkerer import Tinkerer
    from src.knowledge.search import KnowledgeSearchFactory
    
    logger.info("=" * 60)
    logger.info("UNSLOTH QLORA TEST (Tinkerer.evolve + Bedrock)")
    logger.info("=" * 60)
    
    # Patch claude_code to use Bedrock
    patch_claude_code_for_bedrock()
    
    output_dir = tempfile.mkdtemp(prefix="unsloth_qlora_")
    logger.info(f"Output: {output_dir}")
    
    try:
        # Initialize Tinkerer with KG
        tinkerer = Tinkerer(domain="llm_finetuning")
        tinkerer.knowledge_search = KnowledgeSearchFactory.create("kg_graph_search")
        logger.info(f"KG enabled: {tinkerer.knowledge_search.is_enabled()}")
        
        # Run Tinkerer.evolve()
        logger.info("Running Tinkerer.evolve()...")
        solution = tinkerer.evolve(
            goal=TEST_CONFIG["goal"],
            output_path=output_dir,
            max_iterations=TEST_CONFIG["max_iterations"],
            mode=TEST_CONFIG["mode"],
            coding_agent=TEST_CONFIG["coding_agent"],
            language=TEST_CONFIG["language"],
            main_file=TEST_CONFIG["main_file"],
            timeout=TEST_CONFIG["timeout"],
            evaluator=TEST_CONFIG["evaluator"],
            evaluator_params=TEST_CONFIG["evaluator_params"],
            stop_condition=TEST_CONFIG["stop_condition"],
            stop_condition_params=TEST_CONFIG["stop_condition_params"],
        )
        
        # Results
        logger.info("")
        logger.info("RESULTS:")
        logger.info(f"  Experiments: {len(solution.experiment_logs)}")
        logger.info(f"  Code: {solution.code_path}")
        
        # Check generated code for Unsloth patterns
        main_file = Path(solution.code_path) / "main.py"
        if main_file.exists():
            code = main_file.read_text()
            patterns = ["FastLanguageModel", "get_peft_model", "SFTTrainer", "save_pretrained"]
            found = 0
            for p in patterns:
                if p in code:
                    logger.info(f"  ✓ {p}")
                    found += 1
                else:
                    logger.info(f"  ✗ {p}")
            logger.info(f"  Patterns: {found}/{len(patterns)}")
        
        logger.info("")
        logger.info("✅ TEST COMPLETED")
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if tinkerer.knowledge_search.is_enabled():
            tinkerer.knowledge_search.close()


def main():
    # Check prerequisites
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set")
        return False
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        logger.error("AWS_BEARER_TOKEN_BEDROCK not set")
        return False
    
    return run_test()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
