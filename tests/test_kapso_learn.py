from __future__ import annotations

import pytest

from kapso.kapso import Kapso
from kapso.knowledge_base.types import Source


def test_learn_rejects_url_wiki_directory_before_pipeline_work():
    source = Source.Idea(
        query="test",
        source="https://example.com/research",
        content="Complete synthetic idea.",
    )

    with pytest.raises(ValueError, match="local filesystem path, not a URL"):
        Kapso(config_path="src/kapso/config.yaml").learn(
            source,
            wiki_dir="https://knowledge.example.com/wiki",
        )
