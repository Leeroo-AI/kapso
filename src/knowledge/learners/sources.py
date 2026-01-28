# Knowledge Sources
#
# Re-exports from src.knowledge.types for convenience.
#
# Usage:
#     from src.knowledge.learners import Source
#     
#     pipeline.run(Source.Repo("https://github.com/user/repo"))
#     pipeline.run(Source.Idea(query="...", source="...", content="..."))

from src.knowledge.types import Source

__all__ = ["Source"]
