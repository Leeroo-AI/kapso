"""
Live Smoke Test: Web research prompt quality

This script is meant to be run manually (it makes real OpenAI API calls).

It checks that the prompt instructions in `src/knowledge/web_research/prompts/`
are being followed at a basic structural level:
- Required sections exist
- Raw URL citations appear
- Implementation mode includes a ranked "Top sources" section (and typically GitHub stars)

Run (example):
  conda run -n tinkerer_conda python tests/test_web_research_prompts_live.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _require_env() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Put it in your `.env` file or export it in your shell."
        )


def _run_case(tinkerer, *, objective: str, mode: str, depth: str, must_contain: list[str]) -> None:
    print("\n" + "=" * 80)
    print(f"CASE: mode={mode} depth={depth}")
    print(f"Objective: {objective}")
    print("=" * 80)

    research = tinkerer.research(objective, mode=mode, depth=depth)
    text = (research.report_markdown or "").strip()

    assert text, "Expected non-empty report_markdown"
    # The model is instructed to wrap output in <research_result> tags, but the code
    # should parse and return ONLY the inner content (no tags in the final report).
    assert "<research_result>" not in text and "</research_result>" not in text, (
        "Expected <research_result> tags to be stripped by the parser"
    )
    assert "(https://" in text or "(http://" in text, "Expected at least one raw URL citation like (https://...)"

    for needle in must_contain:
        assert needle in text, f"Expected section/token not found: {needle!r}"

    print("\n--- Report preview (first 2000 chars) ---")
    print(text)
    print("\nOK")


def main() -> None:
    _require_env()

    # Import here so the env is loaded first.
    from src.tinkerer import Tinkerer

    tinkerer = Tinkerer()

    # Keep this small and fast-ish. Add more cases as needed.
    cases = [
        {
            "objective": "unsloth FastLanguageModel example",
            "mode": "implementation",
            "depth": "light",
            "must_contain": [
                "## Top sources (ranked)",
                "stars",  # should appear if a GitHub repo is found
            ],
        },
        {
            "objective": "What is LoRA in LLM fine-tuning?",
            "mode": "idea",
            "depth": "light",
            "must_contain": [
                "## Key sources (ranked)",
                "## Core concepts",
                "## Trade-offs",
            ],
        },
        {
            "objective": "LoRA rank selection best practices",
            "mode": "both",
            "depth": "light",
            "must_contain": [
                "## Top sources (ranked)",
                "## Core concepts",
                "## Recommended approach",
            ],
        },
    ]

    for c in cases:
        _run_case(
            tinkerer,
            objective=c["objective"],
            mode=c["mode"],
            depth=c["depth"],
            must_contain=c["must_contain"],
        )

    print("\n" + "=" * 80)
    print("All web research prompt smoke tests passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()

