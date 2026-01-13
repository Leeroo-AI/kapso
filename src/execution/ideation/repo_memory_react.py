"""
RepoMemory ReAct loop (engine-mediated)
======================================

This module implements an engine-mediated loop for ideation:
- The ideation LLM starts with RepoMemory Summary+TOC.
- It may request specific sections (by ID) to ground its proposal.
- The engine (this code) fetches the section content and returns it.
- The loop ends when the model returns a final solution.

Why do this in the engine?
- Most LLM backends used for ideation are plain text-only (no native tool calling).
- It keeps section retrieval auditable and deterministic on the engine side.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

import git

from src.core.llm import LLMBackend
from src.core.prompt_loader import load_prompt, render_prompt
from src.repo_memory import RepoMemoryManager


def _extract_json_obj(text: str) -> Dict[str, Any]:
    """
    Parse a JSON object from model output.

    We prefer strict parsing, but also support a fallback of extracting the first
    {...} block (models sometimes add stray whitespace).
    """
    raw = (text or "").strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        raise ValueError("Expected a JSON object")
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError("Model did not return a JSON object")
    data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def ideate_solution_with_repo_memory_react(
    *,
    llm: LLMBackend,
    model: str,
    repo: git.Repo,
    base_branch: str,
    problem: str,
    workflow_guidance: str = "",
    history_summary: str = "",
    additional_knowledge: str = "",
    output_requirements: str = "",
    max_rounds: int = 6,
    section_max_chars: int = 8000,
    toc_max_chars: int = 3000,
) -> Tuple[str, List[str]]:
    """
    Generate a solution spec via a RepoMemory-aware ReAct loop.

    Returns:
        (solution_text, sections_consulted)
    """
    doc = RepoMemoryManager.load_from_git_branch(repo, base_branch) or {}
    repo_memory_summary_toc = (
        RepoMemoryManager.render_summary_and_toc(doc, max_chars=toc_max_chars) if doc else ""
    )

    template = load_prompt("execution/prompts/ideation_solution_react.md")
    initial_prompt = render_prompt(
        template,
        {
            "problem": problem or "",
            "base_branch": base_branch or "",
            "repo_memory_summary_toc": repo_memory_summary_toc or "(No repo memory available.)",
            "workflow_guidance": workflow_guidance or "",
            "history_summary": history_summary or "",
            "additional_knowledge": additional_knowledge or "",
            "output_requirements": output_requirements or "",
        },
    )

    messages = [{"role": "user", "content": initial_prompt}]
    consulted: List[str] = []

    for _ in range(max_rounds):
        # Deterministic: we're running a structured JSON protocol loop.
        # Lower randomness reduces flakiness and keeps the agent on-format.
        raw = llm.llm_completion(model=model, messages=messages, temperature=0)
        try:
            action = _extract_json_obj(raw)
        except Exception as e:
            # Ask the model to retry with strictly valid JSON.
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your last message was not valid JSON.\n"
                        "Reply again with exactly ONE JSON object, matching the protocol.\n"
                        f"Parsing error: {e}"
                    ),
                }
            )
            continue

        kind = (action or {}).get("action", "")
        if kind == "get_section":
            section_id = str((action or {}).get("section_id", "")).strip()
            if not section_id:
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Missing `section_id`.\n"
                            "Reply with: {\"action\":\"get_section\",\"section_id\":\"<id>\"} or {\"action\":\"final\",\"solution\":\"...\"}"
                        ),
                    }
                )
                continue

            consulted.append(section_id)
            section_text = RepoMemoryManager.get_section(doc, section_id, max_chars=section_max_chars)
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"RepoMemory section '{section_id}':\n\n"
                        f"{section_text}\n\n"
                        "Continue. Reply with ONE JSON object only."
                    ),
                }
            )
            continue

        if kind == "final":
            solution = (action or {}).get("solution", "")
            if not isinstance(solution, str):
                solution = str(solution)
            return solution, _dedupe_preserve_order(consulted)

        # Unknown action -> retry.
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Unknown action '{kind}'.\n"
                    "Valid actions are: get_section, final.\n"
                    "Reply with exactly ONE JSON object."
                ),
            }
        )

    # If we hit the limit, return best-effort: ask the model for a final answer once.
    messages.append(
        {
            "role": "user",
            "content": "You are out of tool rounds. Reply now with {\"action\":\"final\",\"solution\":\"...\"} only.",
        }
    )
    raw = llm.llm_completion(model=model, messages=messages, temperature=0)
    action = _extract_json_obj(raw)
    if (action or {}).get("action") == "final":
        return str((action or {}).get("solution", "")), _dedupe_preserve_order(consulted)
    return str(raw or ""), _dedupe_preserve_order(consulted)

