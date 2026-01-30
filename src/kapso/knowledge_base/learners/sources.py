# Knowledge Sources
#
# Re-exports from kapso.knowledge_base.types for convenience.
#
# Usage:
#     from kapso.knowledge_base.learners import Source
#     
#     pipeline.run(Source.Repo("https://github.com/user/repo"))
#     pipeline.run(Source.Idea(query="...", source="...", content="..."))

from kapso.knowledge_base.types import Source

__all__ = ["Source"]
