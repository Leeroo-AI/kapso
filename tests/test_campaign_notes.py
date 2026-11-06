"""The campaign's notes: what the person tells a running campaign unasked.

What must hold: a note is one line of `.kapso/notes.jsonl` — its number,
a UTC time and the text exactly as given; numbers only go up, whoever
writes at once; a write that never finished is not a note and does not
make the file unreadable, a complete line that is malformed raises, a
missing file is "no notes"; an empty note and a credential-shaped one are
refused before anything is written, the second unless the caller says it
knows. Every ideation, implementation and judge prompt built while notes
exist carries all of them whole, oldest first, in one section before the
role's own task, and a prompt built with none is the prompt as it was;
the strategy reads the file each time it builds a prompt, with the inbox
off as well as on; no other session's template takes the section.
"""

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import kapso.execution.coding_agents.factory as factory_module
from kapso.core.prompt_loader import optional_block
from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.inbox import Note, leave_note, notes_path, read_notes
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.feedback_generator.feedback_generator import FeedbackGenerator
from kapso.execution.search_strategies.generic.ideation import build_ideation_prompt
from kapso.execution.search_strategies.generic.implementation import (
    build_implementation_prompt,
    render_campaign_notes,
)
from kapso.execution.search_strategies.generic.strategy import GenericSearch

SOURCE = Path(__file__).resolve().parent.parent / "src" / "kapso"
HEADING = "## Notes from the person running this campaign"
TOKEN = "sk-abcdefghijklmnopqrstuvwxyz0123"

# Long, many-lined and awkward on purpose: a cap, a first-line cut, a
# strip or a second template substitution on the way to a prompt would
# change it. The line separator and the form feed are line boundaries to
# `str.splitlines`, which must not split the note's line of the file.
LONG_NOTE = (
    "The staging warehouse was rebuilt at 10:40 UTC.\n"
    "\n"
    "  1. Use {{ ref('orders') }}, never the raw table; the literal strings {{problem}}, {{goal}} and {{solution}} stay as written.\n"
    "\t2. Tickets 14 and 15 are fixed upstream: skip them.\n"
    + "\n".join(f"  - model {index}: keep the nightly build under its row budget" for index in range(1, 201))
    + "\nÉcrit à 10h40 — 完了 \u2028 after a line separator, \x0c after a form feed.  \n"
)
SECOND_NOTE = "Ignore the row budget in note 1: the limit was lifted this morning."


# =============================================================================
# THE FILE
# =============================================================================

def test_notes_are_numbered_from_one_and_kept_exactly_as_given(tmp_path):
    path = notes_path(tmp_path)
    first = leave_note(path, LONG_NOTE)
    second = leave_note(path, SECOND_NOTE)

    assert (first.id, second.id) == (1, 2)
    assert read_notes(path) == [first, second]
    assert read_notes(path)[0].text == LONG_NOTE
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, "one note, one line"
    stored = json.loads(lines[0])
    assert sorted(stored) == ["id", "text", "ts"]
    assert stored["id"] == 1 and stored["text"] == LONG_NOTE
    assert datetime.fromisoformat(stored["ts"]).utcoffset() == timedelta(0)


def test_notes_left_at_once_each_get_their_own_number(tmp_path):
    path = notes_path(tmp_path)
    texts = [f"note from writer {index}" for index in range(40)]
    threads = [threading.Thread(target=leave_note, args=(path, text)) for text in texts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    notes = read_notes(path)
    assert [note.id for note in notes] == list(range(1, 41))
    assert sorted(note.text for note in notes) == sorted(texts)


def test_an_unfinished_write_is_not_a_note_and_the_next_writer_removes_it(tmp_path):
    path = notes_path(tmp_path)
    leave_note(path, "first")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"id": 2, "ts": "2026-11-06T10:00:00+00:00", "text": "cut off he')

    assert [note.text for note in read_notes(path)] == ["first"]
    assert leave_note(path, "second").id == 2
    assert [note.text for note in read_notes(path)] == ["first", "second"]
    assert all(json.loads(line) for line in path.read_text().splitlines())


@pytest.mark.parametrize("bad_line, error, message", [
    ("not json", json.JSONDecodeError, None),
    ('["a", "list"]', ValueError, "must be an object"),
    ('{"id": 1, "ts": "now", "text": "the first number again"}', ValueError, "note id must be an integer above 1"),
    ('{"id": 2, "ts": "now"}', ValueError, "non-empty text"),
])
def test_a_malformed_complete_line_raises_for_the_reader_and_the_writer(tmp_path, bad_line, error, message):
    path = notes_path(tmp_path)
    leave_note(path, "first")
    corrupt = path.read_text() + bad_line + "\n"
    path.write_text(corrupt)

    with pytest.raises(error, match=message):
        read_notes(path)
    with pytest.raises(error, match=message):
        leave_note(path, "after the bad line")
    assert path.read_text() == corrupt


def test_a_missing_file_is_no_notes(tmp_path):
    assert read_notes(notes_path(tmp_path)) == []
    assert not (tmp_path / ".kapso").exists()


def test_an_empty_note_and_a_credential_are_refused_before_anything_is_written(tmp_path):
    """The guard holds for every caller, not only the CLI's --yes check;
    the way out exists because the shapes also match hyphenated names."""
    path = notes_path(tmp_path)
    for empty in ("", "  \n\t"):
        with pytest.raises(ValueError, match="a note needs text"):
            leave_note(path, empty)
        with pytest.raises(ValueError, match="a note needs text"):
            leave_note(path, empty, allow_secret=True)
    for shaped in (f"the key is {TOKEN}", "start from mask-rcnn-resnet50-fpn-coco"):
        with pytest.raises(ValueError, match="looks like a credential"):
            leave_note(path, shaped)
    assert not path.exists()
    assert leave_note(path, "start from mask-rcnn-resnet50-fpn-coco", allow_secret=True).id == 1


# =============================================================================
# THE PROMPTS
# =============================================================================

def notes(*texts):
    return [
        Note(id=number, written_at=f"2026-11-06T10:{number:02d}:00+00:00", text=text)
        for number, text in enumerate(texts, start=1)
    ]


def ideation_prompt(campaign_notes: str = "") -> str:
    return build_ideation_prompt(
        "the problem", "the memory", budget_status="b", shared_artifacts_brief="none",
        campaign_notes=campaign_notes,
    )


def implementation_prompt(campaign_notes: str = "") -> str:
    return build_implementation_prompt(
        solution="s", problem="p", branch_name="b", repo_memory_brief="m",
        repo_memory_detail_access_instructions="i", previous_errors="", budget_status="b",
        evaluation_instructions="e", shared_artifacts_brief="none", campaign_notes=campaign_notes,
    )


def judge_prompt(campaign_notes: str = "") -> str:
    return FeedbackGenerator.__new__(FeedbackGenerator)._build_prompt(
        goal="g", idea="i", code_changes_summary="c", base_branch="main", head_branch="x",
        commit_message="m", evaluation_script_path="e.py", evaluation_result="r", workspace_dir="/w",
        campaign_notes=campaign_notes,
    )


def test_the_section_frames_the_notes_and_is_nothing_without_them():
    assert render_campaign_notes([]) == ""
    section = render_campaign_notes(notes(LONG_NOTE, SECOND_NOTE))
    assert section.startswith(HEADING + "\n")
    framing = " ".join(section[: section.index("<note")].split())
    assert "wrote these notes while it ran, oldest first" in framing
    assert "the later note overrides the earlier" in framing
    assert "never changes the evaluation or the task's own rules" in framing
    assert f'<note number="1" written="2026-11-06T10:01:00+00:00">\n{LONG_NOTE}\n</note>' in section
    assert f'<note number="2" written="2026-11-06T10:02:00+00:00">\n{SECOND_NOTE}\n</note>' in section
    assert section.count("<note ") == 2 and "{{notes}}" not in section


@pytest.mark.parametrize("build, own_task", [
    (ideation_prompt, "## Your Process"),
    (implementation_prompt, "## Problem"),
    (judge_prompt, "## Your Task"),
], ids=["ideation", "implementation", "judge"])
def test_a_role_prompt_carries_every_note_whole_and_in_order(build, own_task):
    section = render_campaign_notes(notes(LONG_NOTE, SECOND_NOTE))
    prompt = build(section)
    assert prompt.count(HEADING) == 1
    assert prompt.index(HEADING) < prompt.index(LONG_NOTE) < prompt.index(SECOND_NOTE) < prompt.index(own_task)

    # With no notes the prompt is the prompt as it was: the section is the
    # only difference, and no slot is left behind.
    without = build()
    assert HEADING not in without and "{{" not in without
    assert prompt.replace(optional_block(section), "") == without


def test_the_slot_is_in_the_role_templates_only_and_adds_nothing_when_empty():
    # The members, the selector and the lens planner are the ideation
    # role's sessions; verification, the harness setup, the evaluation
    # maintainer and deployment never read a note.
    generic = "execution/search_strategies/generic/"
    taking = {
        path.relative_to(SOURCE).as_posix(): path.read_text(encoding="utf-8")
        for path in SOURCE.rglob("*.md")
        if "{{campaign_notes}}" in path.read_text(encoding="utf-8")
    }
    assert sorted(taking) == [
        generic + "feedback_generator/prompts/feedback_generator.md",
        generic + "prompts/ideation_claude_code.md",
        generic + "prompts/ideation_lens_planner.md",
        generic + "prompts/ideation_lens_replanner.md",
        generic + "prompts/ideation_selector.md",
        generic + "prompts/implementation_claude_code.md",
    ]
    # Glued between the optional block before it and the heading after it:
    # with no notes the slot renders empty and the prompt keeps every byte
    # it had before the slot existed.
    for template, text in taking.items():
        assert text.count("{{campaign_notes}}") == 1 and "}}{{campaign_notes}}## " in text, template


# =============================================================================
# THE STRATEGY
# =============================================================================

class RecordingJudgeAgent:
    """The judge's session: keeps the prompt it was given, answers with tags."""

    def __init__(self):
        self.prompts = []

    def initialize(self, workspace):
        pass

    def generate_code(self, prompt, debug_mode=False, timeout_seconds=None):
        self.prompts.append(prompt)
        return SimpleNamespace(
            success=True, error=None, metadata={"cli_session_id": "judge-1"},
            output="<stop>false</stop><evaluation_valid>true</evaluation_valid><score>0.5</score><feedback>go on</feedback>",
        )

    def get_cumulative_cost(self):
        return 0.0


def test_the_strategy_reads_the_notes_each_time_it_builds_a_prompt(tmp_path, monkeypatch):
    """A note left while the campaign runs reaches the next prompt of every
    role, through the real judge as well, and needs no request tool: the
    inbox is off here."""
    judge_agent = RecordingJudgeAgent()
    monkeypatch.setattr(factory_module.CodingAgentFactory, "create", classmethod(lambda cls, config: judge_agent))
    monkeypatch.setattr(GenericSearch, "_evaluation_instructions", lambda self: "evaluate")
    judge = FeedbackGenerator.__new__(FeedbackGenerator)
    judge.coding_agent_config = CodingAgentConfig(agent_type="claude_code", model="judge", debug_model="judge", agent_specific={})
    judge._own_agent_specific = {}

    strategy = GenericSearch.__new__(GenericSearch)
    strategy.harness = None
    strategy.workspace_dir = str(tmp_path)
    strategy.inbox_settings = {"enabled": False}
    strategy.experience_intro_path = None
    strategy.hosted = None
    strategy.shared_artifacts_brief = "none"
    strategy.iteration_count = 1
    strategy.goal = "the goal"
    strategy.budget_snapshot = None
    strategy.budget_snapshot_monotonic = None
    strategy.registered_evaluation_command = ""
    strategy.feedback_generator = judge

    def prompts_built_now():
        strategy._generate_feedback(
            SearchNode(node_id=0, branch_name="candidate", parent_branch_name="main", workspace_dir=str(tmp_path))
        )
        return {
            "ideation": strategy._build_ideation_prompt("the problem", "the memory"),
            "implementation": strategy._build_implementation_prompt(
                solution="s", problem="p", branch_name="b", repo_memory_brief="m",
                repo_memory_detail_access_instructions="i", previous_errors="",
            ),
            "judge": judge_agent.prompts[-1],
        }

    for role, prompt in prompts_built_now().items():
        assert HEADING not in prompt, role

    leave_note(notes_path(tmp_path), LONG_NOTE)
    leave_note(notes_path(tmp_path), SECOND_NOTE)

    for role, prompt in prompts_built_now().items():
        assert prompt.count(HEADING) == 1, role
        assert prompt.index(HEADING) < prompt.index(LONG_NOTE) < prompt.index(SECOND_NOTE), role
