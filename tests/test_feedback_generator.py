"""Hermetic tests for the FeedbackGenerator's judge contract.

Pins the fixes from the live track's feedback fragility findings: a
tagless response earns exactly one retry and then an explicit failure
result (never a fabricated verdict), the raw output is never truncated on
its way into the next iteration's prompt, and the budget clamp reaches
the call through the per-call timeout.
"""

from types import SimpleNamespace

from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.search_strategies.generic.feedback_generator.feedback_generator import (
    FeedbackGenerator,
)


class ScriptedFeedbackAgent:
    """Returns queued outputs; records prompts and per-call timeouts."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.cost = 0.0

    def initialize(self, workspace):
        pass

    def generate_code(self, prompt, debug_mode=False, timeout_seconds=None):
        self.calls.append(
            {"prompt": prompt, "timeout_seconds": timeout_seconds}
        )
        self.cost += 0.1
        return SimpleNamespace(output=self.outputs.pop(0))

    def get_cumulative_cost(self):
        return self.cost


TAGGED = (
    "<stop>false</stop>\n<evaluation_valid>true</evaluation_valid>\n"
    "<score>0.8</score>\n<feedback>keep going</feedback>"
)


def make_generator(agent) -> FeedbackGenerator:
    generator = FeedbackGenerator.__new__(FeedbackGenerator)
    generator.coding_agent_config = CodingAgentConfig(
        agent_type="stub",
        model="stub",
        debug_model="stub",
        agent_specific={"timeout": 90},
    )
    generator.agent = agent
    return generator


def generate(generator, timeout_seconds=None):
    return generator.generate(
        goal="g",
        idea="i",
        code_changes_summary="c",
        base_branch="main",
        head_branch="exp",
        evaluation_script_path="kapso_evaluation/kapso_eval.py",
        evaluation_result="out",
        workspace_dir="/tmp/nowhere",
        timeout_seconds=timeout_seconds,
    )


def patch_prompt(monkeypatch):
    import kapso.execution.search_strategies.generic.feedback_generator.feedback_generator as module

    monkeypatch.setattr(module, "load_prompt", lambda path: "PROMPT")
    monkeypatch.setattr(
        module, "render_prompt", lambda template, values: "PROMPT"
    )


def test_tagless_response_retries_once_then_fails_explicitly(
    monkeypatch,
):
    patch_prompt(monkeypatch)
    raw = "I analyzed everything but forgot the tags entirely. " * 40
    agent = ScriptedFeedbackAgent([raw, raw])
    generator = make_generator(agent)
    monkeypatch.setattr(
        generator, "_get_commit_message", lambda *args: "msg"
    )

    result = generate(generator)

    assert len(agent.calls) == 2
    assert "required tags" in agent.calls[1]["prompt"]
    assert result.score is None
    assert result.stop is False
    assert "Feedback generation failed" in result.feedback
    # The raw output flows into the next iteration's prompt: never
    # truncated (the old path clipped it to 500 characters).
    assert raw in result.feedback


def test_retry_recovers_a_tagged_response(monkeypatch):
    patch_prompt(monkeypatch)
    agent = ScriptedFeedbackAgent(["no tags here", TAGGED])
    generator = make_generator(agent)
    monkeypatch.setattr(
        generator, "_get_commit_message", lambda *args: "msg"
    )

    result = generate(generator)

    assert len(agent.calls) == 2
    assert result.score == 0.8
    assert result.feedback == "keep going"
    # Cost telemetry covers both calls of the transaction.
    assert result.cost_usd > 0.15


def test_timeout_threads_to_every_call(monkeypatch):
    patch_prompt(monkeypatch)
    agent = ScriptedFeedbackAgent(["no tags", TAGGED])
    generator = make_generator(agent)
    monkeypatch.setattr(
        generator, "_get_commit_message", lambda *args: "msg"
    )

    generate(generator, timeout_seconds=42.0)

    assert [call["timeout_seconds"] for call in agent.calls] == [42.0, 42.0]
    assert generator.configured_timeout_seconds == 90.0


def test_prompt_forbids_per_sample_eval_content_in_feedback():
    """R10-P2-1: the judge quoted eval samples + gold answers into feedback.

    The invariants section must bind the JUDGE to the data rules and ban
    per-sample eval content; losing this text reopens the leak.
    """
    template = open(
        "src/kapso/execution/search_strategies/generic/feedback_generator/"
        "prompts/feedback_generator.md"
    ).read()
    section = template.split("## Invariant rules (highest priority)")[1]
    for required in (
        "PER-SAMPLE evaluation content",
        "no gold/expected answers",
        "IMMUTABLE across iterations",
        "only aggregate results",
    ):
        assert required in section


def test_return_economics_and_scoped_invariants_in_prompt():
    """R11+ return shaping: a stable tie far from the bar is a budget LOSS;
    approach closures are strong priors with recorded reopening conditions,
    never blanket invariants."""
    from kapso.core.prompt_loader import load_prompt

    prompt = load_prompt(
        "execution/search_strategies/generic/feedback_generator/prompts/"
        "feedback_generator.md"
    )
    assert "Return economics" in prompt
    assert "is a LOSS" in prompt
    assert "STRONG PRIOR, not an invariant" in prompt
    assert 'blanket "do NOT reopen' in prompt


def test_axis_frontier_duty_and_design_axes_render():
    """Anti-freeze contract (user-directed 2026-07-27): the feedback prompt
    must carry the per-axis frontier duty (headroom evidence + untried moves
    per non-saturated axis) and render the strategy-supplied axis map."""
    from kapso.core.prompt_loader import load_prompt

    from kapso.execution.search_strategies.generic.feedback_generator.feedback_generator import (
        FeedbackGenerator,
    )

    prompt = load_prompt(FeedbackGenerator.PROMPT_PATH)
    assert "Report the axis frontier" in prompt
    assert "Design axes of the solution space" in prompt
    assert "saturation claim needs cited" in prompt.lower() or (
        "saturation claim needs cited measurement" in prompt
    )

    generator = FeedbackGenerator.__new__(FeedbackGenerator)
    rendered = FeedbackGenerator._build_prompt(
        generator,
        goal="g",
        idea="i",
        code_changes_summary="c",
        base_branch="b",
        head_branch="h",
        commit_message="m",
        evaluation_script_path="e",
        evaluation_result="r",
        workspace_dir="w",
        design_axes="1. input representation — features",
    )
    assert "1. input representation — features" in rendered
    rendered_default = FeedbackGenerator._build_prompt(
        generator,
        goal="g",
        idea="i",
        code_changes_summary="c",
        base_branch="b",
        head_branch="h",
        commit_message="m",
        evaluation_script_path="e",
        evaluation_result="r",
        workspace_dir="w",
    )
    assert "(none declared)" in rendered_default
