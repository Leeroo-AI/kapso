"""The "Domain guidance" block in every module's prompt: both pages whole,
the attachments by name, placed before the section that follows the
module's context; and without a harness every prompt is byte-identical to
one rendered with no guidance at all. The block's size per module is
printed, so the proof run can record what a harness adds to a prompt."""

import shutil
from pathlib import Path

import pytest

from kapso.core.config import load_config
from kapso.core.prompt_loader import load_prompt, optional_block, render_prompt
from kapso.execution.search_strategies.generic.feedback_generator.feedback_generator import FeedbackGenerator
from kapso.execution.search_strategies.generic.ideation import build_ideation_prompt
from kapso.execution.search_strategies.generic.implementation import (
    CLOSING_LINE_WITHOUT_INBOX,
    build_implementation_prompt,
)
from kapso.harness import GUIDANCE_HEADING, load_harness, stage_harness

TRADING = Path(__file__).resolve().parent.parent / "examples" / "harnesses" / "trading"
GENERIC = "execution/search_strategies/generic/prompts/"


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("harness")
    copy = tmp / "trading"
    shutil.copytree(TRADING, copy)
    return load_harness(str(copy), settings=dict(load_config()["harness"], store=str(tmp / "store")), say=lambda line: None)


def guidance_for(harness, module, tmp_path) -> str:
    campaign = tmp_path / "campaign"
    session = campaign / "s"
    session.mkdir(parents=True, exist_ok=True)
    return stage_harness(harness, module, session_dir=session, campaign_dir=campaign, session_id="s", host="claude_code").guidance


def page(harness, name) -> str:
    return (harness.path / "instructions" / name).read_text().strip()


def ideation_prompt(guidance: str) -> str:
    return build_ideation_prompt(
        "the problem", "the memory", budget_status="b", shared_artifacts_brief="", domain_guidance=guidance,
    )


def implementation_prompt(guidance: str) -> str:
    return build_implementation_prompt(
        solution="s", problem="p", branch_name="b", repo_memory_brief="m",
        repo_memory_detail_access_instructions="i", previous_errors="", budget_status="b",
        evaluation_instructions="e", shared_artifacts_brief="", domain_guidance=guidance,
    )


def judge_prompt(guidance: str) -> str:
    generator = FeedbackGenerator.__new__(FeedbackGenerator)
    return generator._build_prompt(
        goal="g", idea="i", code_changes_summary="c", base_branch="main", head_branch="x",
        commit_message="m", evaluation_script_path="e.py", evaluation_result="r", workspace_dir="/w",
        domain_guidance=guidance,
    )


def test_ideation_carries_both_pages_whole_and_the_attachments_before_its_process(harness, tmp_path):
    guidance = guidance_for(harness, "ideation", tmp_path)
    prompt = ideation_prompt(guidance)
    assert page(harness, "guardrails.md") in prompt and page(harness, "ideation.md") in prompt
    assert "- strategy-families — " in prompt and "- market-structure — " in prompt
    assert prompt.index(GUIDANCE_HEADING) < prompt.index("## Your Process")
    assert prompt.index("### Goal") < prompt.index(GUIDANCE_HEADING)
    print(f"guidance bytes ideation: {len(guidance.encode())}")


def test_implementation_carries_its_module_and_still_ends_with_the_closing_line(harness, tmp_path):
    guidance = guidance_for(harness, "implementation", tmp_path)
    prompt = implementation_prompt(guidance)
    assert page(harness, "implementation.md") in prompt and page(harness, "guardrails.md") in prompt
    assert "- lookahead-auditor — " in prompt and "- market-data — its tools are named mcp__market-data__<tool>" in prompt
    assert prompt.index(GUIDANCE_HEADING) < prompt.index("## Problem")
    assert prompt.rstrip().endswith(CLOSING_LINE_WITHOUT_INBOX.rstrip())
    print(f"guidance bytes implementation: {len(guidance.encode())}")


def test_the_judge_carries_the_feedback_page_before_its_task(harness, tmp_path):
    guidance = guidance_for(harness, "feedback", tmp_path)
    prompt = judge_prompt(guidance)
    assert page(harness, "feedback.md") in prompt and "- risk-reviewer — " in prompt
    assert prompt.index(GUIDANCE_HEADING) < prompt.index("## Your Task")
    print(f"guidance bytes feedback: {len(guidance.encode())}")


@pytest.mark.parametrize("template, variables", [
    (GENERIC + "ideation_selector.md", {"problem": "p", "repo_memory_brief": "m", "campaign_state": "c", "candidates": "x",
                                        "campaign_notes": ""}),
    (GENERIC + "ideation_lens_planner.md", {"problem": "p", "member_roster": "r", "lens_count": "2", "shared_artifacts_brief": "", "axes": "a",
                                            "experience_intro": "", "campaign_notes": ""}),
    (GENERIC + "ideation_lens_replanner.md", {"problem": "p", "member_roster": "r", "lens_count": "2", "shared_artifacts_brief": "", "axes": "a",
                                              "campaign_state": "c", "plan_iteration": "1", "previous_lenses": "l", "previous_sources": "s",
                                              "previous_rationale": "r", "recent_feedback": "f", "champion_solution": "w", "experience_intro": "",
                                              "campaign_notes": ""}),
    ("execution/evaluation_maintainer/prompts/setup_provided.md", {"goal": "g", "entrypoint_name": "e", "manifest_marker": "m", "fast_fraction": "0.1", "subsample_seed": "7"}),
    ("execution/evaluation_maintainer/prompts/setup_build.md", {"goal": "g", "data_dir": "d", "entrypoint_name": "e", "manifest_marker": "m", "fast_fraction": "0.1", "subsample_seed": "7"}),
    ("execution/evaluation_maintainer/prompts/change_request.md", {"requested_by": "x", "summary": "s", "evidence": "e", "provided_logic_immutable": "true", "entrypoint_name": "e", "manifest_marker": "m"}),
])
def test_every_other_module_template_takes_the_block_and_renders_clean_without_it(harness, tmp_path, template, variables):
    text = load_prompt(template)
    assert "{{domain_guidance}}" in text
    guidance = guidance_for(harness, "setup" if "setup" in template else "verification" if "change" in template else "ideation", tmp_path)
    with_block = render_prompt(text, {**variables, "domain_guidance": optional_block(guidance)})
    without = render_prompt(text, {**variables, "domain_guidance": optional_block("")})
    assert GUIDANCE_HEADING in with_block and page(harness, "guardrails.md") in with_block
    assert GUIDANCE_HEADING not in without and "{{" not in without
    assert without == render_prompt(text, {**variables, "domain_guidance": ""})


def test_without_a_harness_every_prompt_is_byte_identical_to_a_render_with_no_block():
    for prompt in (ideation_prompt(""), implementation_prompt(""), judge_prompt("")):
        assert GUIDANCE_HEADING not in prompt and "{{domain_guidance}}" not in prompt
    assert implementation_prompt("").rstrip().endswith(CLOSING_LINE_WITHOUT_INBOX.rstrip())


def test_a_long_page_reaches_the_prompt_whole(harness, tmp_path):
    copy = tmp_path / "long"
    shutil.copytree(harness.path, copy)
    long_page = copy / "instructions" / "guardrails.md"
    long_page.write_text(long_page.read_text() + "\n" + "\n".join(f"- rule number {i} of the domain" for i in range(600)) + "\n")
    settings = dict(load_config()["harness"], store=str(tmp_path / "store"))
    long_harness = load_harness(str(copy), settings=settings, say=lambda line: None)
    guidance = guidance_for(long_harness, "implementation", tmp_path / "x")
    assert len(long_page.read_text().encode()) > 20_000
    assert long_page.read_text().strip() in implementation_prompt(guidance)
