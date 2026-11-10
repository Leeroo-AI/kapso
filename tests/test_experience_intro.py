"""The experience intro's seam (wp3-experience.md, change 14): the intro
rides its own slot, `{{experience_intro}}`, in the ideation, lens-planner
and implementation prompts, after the harness guidance and before the
section that follows; the judge, setup, verification and maintainer
prompts never take it; without a source every prompt is byte-identical to
a render with no block. The strategy reads the staged intro lazily from
the serving directory, so a seeded workspace stays empty at construction."""

from contextlib import contextmanager
from types import SimpleNamespace

from kapso.execution.coding_agents.base import CodingAgentConfig
from strategy_support import generic_params, packaged_hosts

import pytest

from kapso.core.prompt_loader import load_prompt, optional_block, render_prompt
from kapso.execution.search_strategies.generic.feedback_generator.feedback_generator import FeedbackGenerator
from kapso.execution.search_strategies.generic.ideation import build_ideation_prompt
from kapso.execution.search_strategies.generic.implementation import build_implementation_prompt
from kapso.execution.search_strategies.generic.strategy import GenericSearch
from kapso.harness import GUIDANCE_HEADING
from kapso.learning.serving_launch import INTRO_HEADING

GENERIC = "execution/search_strategies/generic/prompts/"
INTRO = INTRO_HEADING + "\n\nYou are served:\n- 3 cards measured by you, under the trading harness\n"
GUIDANCE = GUIDANCE_HEADING + "\n\nthe domain's rules\n"


def ideation_prompt(intro: str, guidance: str = "") -> str:
    return build_ideation_prompt(
        "the problem", "the memory", budget_status="b", shared_artifacts_brief="",
        domain_guidance=guidance, experience_intro=intro,
    )


def implementation_prompt(intro: str, guidance: str = "") -> str:
    return build_implementation_prompt(
        solution="s", problem="p", branch_name="b", repo_memory_brief="m",
        repo_memory_detail_access_instructions="i", previous_errors="", budget_status="b",
        evaluation_instructions="e", shared_artifacts_brief="", domain_guidance=guidance,
        experience_intro=intro,
    )


def test_the_intro_rides_after_the_guidance_and_before_the_module_section():
    prompt = ideation_prompt(INTRO, GUIDANCE)
    assert INTRO.strip() in prompt
    assert prompt.index(GUIDANCE_HEADING) < prompt.index(INTRO_HEADING) < prompt.index("## Your Process")
    prompt = implementation_prompt(INTRO, GUIDANCE)
    assert INTRO.strip() in prompt
    assert prompt.index(GUIDANCE_HEADING) < prompt.index(INTRO_HEADING) < prompt.index("## Problem")


@pytest.mark.parametrize("template, variables", [
    (GENERIC + "ideation_lens_planner.md", {"problem": "p", "member_roster": "r", "lens_count": "2", "shared_artifacts_brief": "", "axes": "a",
                                            "campaign_notes": ""}),
    (GENERIC + "ideation_lens_replanner.md", {"problem": "p", "member_roster": "r", "lens_count": "2", "shared_artifacts_brief": "", "axes": "a",
                                              "campaign_state": "c", "plan_iteration": "1", "previous_lenses": "l", "previous_sources": "s",
                                              "previous_rationale": "r", "recent_feedback": "f", "champion_solution": "w",
                                              "campaign_notes": ""}),
])
def test_the_lens_planner_templates_take_the_block_and_render_clean_without_it(template, variables):
    text = load_prompt(template)
    assert "{{domain_guidance}}{{experience_intro}}" in text
    with_block = render_prompt(text, {**variables, "domain_guidance": "", "experience_intro": optional_block(INTRO)})
    without = render_prompt(text, {**variables, "domain_guidance": "", "experience_intro": optional_block("")})
    assert INTRO_HEADING in with_block and INTRO_HEADING not in without and "{{" not in without


def test_the_judge_setup_verification_and_maintainer_prompts_never_take_the_intro():
    for template in (
        GENERIC + "verification.md", GENERIC + "harness_setup.md", GENERIC + "ideation_selector.md",
        "execution/evaluation_maintainer/prompts/setup_provided.md",
        "execution/evaluation_maintainer/prompts/setup_build.md",
        "execution/evaluation_maintainer/prompts/change_request.md",
    ):
        assert "experience_intro" not in load_prompt(template)
    generator = FeedbackGenerator.__new__(FeedbackGenerator)
    judge = generator._build_prompt(
        goal="g", idea="i", code_changes_summary="c", base_branch="main", head_branch="x",
        commit_message="m", evaluation_script_path="e.py", evaluation_result="r", workspace_dir="/w",
        domain_guidance=GUIDANCE,
    )
    assert INTRO_HEADING not in judge and "experience_intro" not in judge


def test_without_a_source_every_prompt_is_byte_identical_to_a_render_with_no_block():
    for prompt in (ideation_prompt(""), implementation_prompt("")):
        assert INTRO_HEADING not in prompt and "{{experience_intro}}" not in prompt
    assert ideation_prompt("", GUIDANCE) == build_ideation_prompt(
        "the problem", "the memory", budget_status="b", shared_artifacts_brief="", domain_guidance=GUIDANCE,
    )


@contextmanager
def _patched_super_init(workspace_dir):
    from kapso.execution.search_strategies.base import SearchStrategy

    original = SearchStrategy.__init__

    def fake_init(self, config, wd=None, import_from_checkpoint=False):
        self.params = config.params or {}
        self.workspace_dir = workspace_dir
        self.feedback_generator = None
        self.budget_snapshot = None
        self.budget_snapshot_monotonic = None

    SearchStrategy.__init__ = fake_init
    yield
    SearchStrategy.__init__ = original


def test_the_strategy_reads_the_staged_intro_lazily_and_fails_loud_when_it_is_missing(tmp_path):
    # The intro is staged after orchestrator construction (a seeded
    # workspace must be empty then), so the strategy carries its path and
    # reads it when a prompt is built; the judge never gets the bank gate.
    intro_path = tmp_path / ".kapso" / "serving" / "intro.md"
    params = generic_params(bank_serving={"KAPSO_BANKS": "[]"}, experience_intro_path=str(intro_path))
    with _patched_super_init(str(tmp_path)):
        strategy = GenericSearch(SimpleNamespace(params=params, hosts=packaged_hosts()), str(tmp_path))
    strategy.shared_artifacts_brief = ""
    strategy._render_budget_status = lambda: "b"
    strategy._render_inbox_ideation = lambda: ""
    with pytest.raises(FileNotFoundError):
        strategy._build_ideation_prompt("p", "m")
    intro_path.parent.mkdir(parents=True)
    intro_path.write_text(INTRO)
    assert INTRO.strip() in strategy._build_ideation_prompt("p", "m")
    assert "bank" in strategy.ideation_gates and "bank" in strategy.implementation_gates
    # The judge is a session with no gate list at all: its agent config
    # mounts no server, so no bank can reach it.
    judge = CodingAgentConfig(agent_type="claude_code", model="claude-opus-5", debug_model="claude-opus-5", agent_specific={})
    assert "mcp_servers" not in FeedbackGenerator(judge).coding_agent_config.agent_specific
    with _patched_super_init(str(tmp_path)):
        plain = GenericSearch(SimpleNamespace(params=generic_params(), hosts=packaged_hosts()), str(tmp_path))
    assert plain.experience_intro() == ""
