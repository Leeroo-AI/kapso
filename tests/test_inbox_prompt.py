"""The inbox in the prompts (design v4 Appendix A).

What must hold: with the inbox off the implementation and ideation
prompts render exactly as before — no tool line, no section, the old
closing line, no template variable left behind; with it on the section,
the tool line, the swapped closing line and the checklist note appear,
and the campaign's requests are listed; the follow-up a resumed session
reads carries every request with its reply and the recorded next steps,
and refuses an unanswered request; ideation sees what was answered.
"""

import pytest

from kapso.execution.inbox import Request
from kapso.execution.search_strategies.generic.ideation import build_ideation_prompt
from kapso.execution.search_strategies.generic.implementation import (
    CHECKLIST_NOTE_WITH_INBOX,
    CLOSING_LINE_WITH_INBOX,
    CLOSING_LINE_WITHOUT_INBOX,
    build_implementation_prompt,
    render_follow_up,
    render_inbox_ideation,
    render_inbox_section,
    render_inbox_state,
)


def request(request_id, key, reply=None, next_steps="embed, re-rank, evaluate"):
    return Request(
        id=request_id, node=3, session="s", key=key,
        hit="openai.AuthenticationError", tried="OPENAI_KEY unset too; no .env",
        fix="add OPENAI_API_KEY=sk-... to .env", next_steps=next_steps,
        requested_at="now", reply=reply,
    )


def implementation_prompt(**overrides):
    kwargs = dict(
        solution="do the thing", problem="the goal", branch_name="generic_exp_3",
        repo_memory_brief="brief", repo_memory_detail_access_instructions="use the tool",
        previous_errors="", budget_status="42/120 min", evaluation_instructions="evaluate",
        shared_artifacts_brief="none",
    )
    kwargs.update(overrides)
    return build_implementation_prompt(**kwargs)


def test_implementation_prompt_is_unchanged_with_the_inbox_off():
    prompt = implementation_prompt()
    assert "{{" not in prompt
    assert "request_from_user" not in prompt
    assert "When you are blocked" not in prompt
    assert CHECKLIST_NOTE_WITH_INBOX not in prompt
    assert prompt.rstrip().endswith(CLOSING_LINE_WITHOUT_INBOX)


def test_implementation_prompt_with_the_inbox_on():
    requests = {1: request(1, "env:OPENAI_API_KEY", reply="added the key"), 2: request(2, "data/x.csv")}
    prompt = implementation_prompt(inbox_section=render_inbox_section(requests))
    assert "{{" not in prompt
    assert "### Asking the person (MCP tool)" in prompt
    assert "## When you are blocked on something only a person can provide" in prompt
    assert "Prove it before you ask" in prompt
    assert "### Requests already in this campaign's inbox" in prompt
    assert "#1 env:OPENAI_API_KEY — answered (node 3): 'added the key'" in prompt
    assert "#2 data/x.csv — open, no reply yet: treat as ABSENT" in prompt
    assert prompt.rstrip().endswith(CLOSING_LINE_WITH_INBOX)
    assert "LAST thing in your response**" + CHECKLIST_NOTE_WITH_INBOX in prompt
    # The section sits after the runtime-discipline rules, before the budget.
    assert prompt.index("## Session Runtime Discipline") < prompt.index("## When you are blocked") < prompt.index("## Budget")


def test_inbox_state_is_empty_on_a_fresh_campaign():
    assert render_inbox_state({}) == ""
    section = render_inbox_section({})
    assert section.startswith("## When you are blocked")
    assert "Requests already" not in section


def test_inbox_section_names_where_values_go():
    """A fix must point at the campaign's .env, never at the session
    folder (live L1 run 2 sent the person there)."""
    with_env = render_inbox_section({}, dotenv_path="/home/me/churn/.env")
    assert "The campaign's `.env` is `/home/me/churn/.env`" in with_env
    assert "never point the person here" in with_env and "{{" not in with_env
    without = render_inbox_section({})
    assert "No `.env` was found for this campaign" in without and "{{" not in without


def test_follow_up_carries_every_reply_and_the_next_steps():
    requests = [
        request(1, "env:OPENAI_API_KEY", reply="added the key"),
        request(2, "data/x.csv", reply="", next_steps="load the csv, then embed"),
    ]
    text = render_follow_up(requests)
    assert "Request #1 — env:OPENAI_API_KEY" in text
    assert 'their reply: "added the key"' in text
    assert "Request #2 — data/x.csv" in text and "their reply: (done)" in text
    assert "you tried: OPENAI_KEY unset too; no .env" in text
    assert "  embed, re-rank, evaluate\n  load the csv, then embed" in text
    assert "end with the XML result tags" in text
    with pytest.raises(ValueError, match="no reply yet"):
        render_follow_up([request(3, "tool:docker")])
    with pytest.raises(ValueError, match="at least one"):
        render_follow_up([])


def test_ideation_prompt_carries_the_rule_and_the_answered_requests():
    off = build_ideation_prompt("goal", "brief", budget_status="b", shared_artifacts_brief="none")
    assert "{{" not in off and "only a person can provide" not in off
    rule_only = render_inbox_ideation({1: request(1, "env:X")})
    assert "### Things only a person can provide" in rule_only
    assert "Needs from the person:" in rule_only and "already answered" not in rule_only
    block = render_inbox_ideation({
        1: request(1, "env:X", reply="not available, use bge-large"),
        2: request(2, "data/y.csv", reply=""),
    })
    on = build_ideation_prompt("goal", "brief", budget_status="b", shared_artifacts_brief="none", inbox_ideation=block)
    assert "### Things only a person can provide" in on
    assert "What the person has already answered about resources:" in on
    assert "- env:X: 'not available, use bge-large'" in on
    assert "- data/y.csv: provided" in on
    assert "{{" not in on
