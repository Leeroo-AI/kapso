# Knowledge Sources
#
# Re-exports from src.knowledge_base.types for convenience.
#
# Usage:
#     from src.knowledge_base.learners import Source
#     
#     pipeline.run(Source.Repo("https://github.com/user/repo"))
#     pipeline.run(Source.Idea(query="...", source="...", content="..."))

from src.knowledge_base.types import Source

__all__ = ["Source"]
