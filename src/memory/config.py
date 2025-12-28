# Cognitive Memory Configuration Loader
#
# Loads and merges configuration from cognitive_memory.yaml
# Supports presets and environment variable overrides.

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default config file location
CONFIG_FILE = Path(__file__).parent / "cognitive_memory.yaml"


@dataclass
class EpisodicConfig:
    """Configuration for EpisodicStore."""
    collection_name: str = "EpisodicInsights"
    embedding_model: str = "text-embedding-3-small"
    weaviate_host: str = "localhost"
    weaviate_port: int = 8080
    retrieval_top_k: int = 5
    min_confidence: float = 0.5
    max_insights: int = 1000
    persist_path: str = ".memory_store.json"


@dataclass
class ControllerConfig:
    """Configuration for CognitiveController."""
    llm_model: str = "gpt-4o-mini"
    fallback_models: list = field(default_factory=lambda: ["gpt-4.1-mini"])
    state_file_path: str = ".praxium_state.md"
    max_error_length: int = 1000
    max_fact_length: int = 200


@dataclass
class InsightExtractionConfig:
    """Configuration for insight extraction."""
    enabled: bool = True
    min_error_length: int = 50
    max_insight_length: int = 500
    default_confidence: float = 0.8
    auto_tags: list = field(default_factory=lambda: ["auto-generated"])


@dataclass
class BriefingConfig:
    """
    Configuration for briefing generation.
    
    Context size is controlled implicitly through KG graph structure:
    - ALL heuristics linked to steps (via USES_HEURISTIC edges)
    - ALL implementations linked to steps (via IMPLEMENTED_BY edges)  
    - ALL environments linked to implementations (via REQUIRES_ENV edges)
    
    Well-curated KG = well-bounded context. No arbitrary truncation.
    Only episodic insights are limited by retrieval top_k.
    """
    include_plan: bool = True
    include_error_history: bool = True
    
    # Only episodic insights are limited (everything else comes from graph)
    max_episodic_insights: int = 5


@dataclass
class CognitiveMemoryConfig:
    """
    Complete cognitive memory configuration.
    
    Usage:
        config = CognitiveMemoryConfig.load()
        config = CognitiveMemoryConfig.load(preset="high_quality")
        
        # Access settings
        config.episodic.embedding_model
        config.controller.llm_model
    """
    episodic: EpisodicConfig = field(default_factory=EpisodicConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    insight_extraction: InsightExtractionConfig = field(default_factory=InsightExtractionConfig)
    briefing: BriefingConfig = field(default_factory=BriefingConfig)
    
    @classmethod
    def load(
        cls, 
        config_path: Optional[str] = None,
        preset: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> "CognitiveMemoryConfig":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file (default: cognitive_memory.yaml)
            preset: Name of preset to apply (e.g., "minimal", "high_quality")
            overrides: Dict of overrides to apply on top of preset
            
        Returns:
            CognitiveMemoryConfig instance
        """
        path = Path(config_path) if config_path else CONFIG_FILE
        
        # Start with defaults
        config_dict = {
            "episodic": {},
            "controller": {},
            "insight_extraction": {},
            "briefing": {}
        }
        
        # Load from YAML if exists
        if path.exists():
            try:
                with open(path) as f:
                    yaml_data = yaml.safe_load(f) or {}
                
                # Merge defaults
                if "defaults" in yaml_data:
                    for section in config_dict:
                        if section in yaml_data["defaults"]:
                            config_dict[section].update(yaml_data["defaults"][section])
                
                # Apply preset if specified
                if preset and "presets" in yaml_data and preset in yaml_data["presets"]:
                    preset_data = yaml_data["presets"][preset]
                    for section in config_dict:
                        if section in preset_data:
                            config_dict[section].update(preset_data[section])
                    logger.info(f"Applied cognitive memory preset: {preset}")
                            
            except Exception as e:
                logger.warning(f"Failed to load cognitive memory config from {path}: {e}")
        
        # Apply overrides
        if overrides:
            for section, values in overrides.items():
                if section in config_dict and isinstance(values, dict):
                    config_dict[section].update(values)
        
        # Apply environment variable overrides
        config_dict = cls._apply_env_overrides(config_dict)
        
        # Build config objects
        return cls(
            episodic=EpisodicConfig(**config_dict["episodic"]),
            controller=ControllerConfig(**config_dict["controller"]),
            insight_extraction=InsightExtractionConfig(**config_dict["insight_extraction"]),
            briefing=BriefingConfig(**config_dict["briefing"])
        )
    
    @staticmethod
    def _apply_env_overrides(config_dict: Dict) -> Dict:
        """
        Apply environment variable overrides.
        
        Environment variables follow the pattern:
            COGNITIVE_MEMORY_<SECTION>_<KEY>=value
            
        Examples:
            COGNITIVE_MEMORY_EPISODIC_EMBEDDING_MODEL=text-embedding-3-large
            COGNITIVE_MEMORY_CONTROLLER_LLM_MODEL=gpt-4-turbo
        """
        prefix = "COGNITIVE_MEMORY_"
        
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
                
            # Parse: COGNITIVE_MEMORY_EPISODIC_EMBEDDING_MODEL -> episodic.embedding_model
            parts = key[len(prefix):].lower().split("_", 1)
            if len(parts) != 2:
                continue
                
            section, param = parts
            if section not in config_dict:
                continue
            
            # Type conversion
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "").isdigit():
                value = float(value)
                
            config_dict[section][param] = value
            logger.debug(f"Applied env override: {section}.{param} = {value}")
        
        return config_dict
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "episodic": {
                "collection_name": self.episodic.collection_name,
                "embedding_model": self.episodic.embedding_model,
                "weaviate_host": self.episodic.weaviate_host,
                "weaviate_port": self.episodic.weaviate_port,
                "retrieval_top_k": self.episodic.retrieval_top_k,
                "min_confidence": self.episodic.min_confidence,
                "max_insights": self.episodic.max_insights,
                "persist_path": self.episodic.persist_path,
            },
            "controller": {
                "llm_model": self.controller.llm_model,
                "fallback_models": self.controller.fallback_models,
                "state_file_path": self.controller.state_file_path,
                "max_error_length": self.controller.max_error_length,
                "max_fact_length": self.controller.max_fact_length,
            },
            "insight_extraction": {
                "enabled": self.insight_extraction.enabled,
                "min_error_length": self.insight_extraction.min_error_length,
                "max_insight_length": self.insight_extraction.max_insight_length,
                "default_confidence": self.insight_extraction.default_confidence,
                "auto_tags": self.insight_extraction.auto_tags,
            },
            "briefing": {
                "include_plan": self.briefing.include_plan,
                "include_error_history": self.briefing.include_error_history,
                "max_episodic_insights": self.briefing.max_episodic_insights,
            }
        }


# Convenience function
def get_config(preset: str = None) -> CognitiveMemoryConfig:
    """Get cognitive memory configuration with optional preset."""
    return CognitiveMemoryConfig.load(preset=preset)
