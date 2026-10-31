"""The judge (docs/plans/redesign/wp9-semantic-suite.md, change 5): the
rubrics carry the design's dimensions by name (a drift check); a step is
rendered as the user's messages and each turn's final text, on both
hosts' real row shapes, with numbered lines; the reply must be one JSON
object with exactly the rubric's dimensions, integer scores in the scale,
and at least one evidence entry whose quote is on the line it names in the
material's own file, else a judge failure that keeps the raw reply and
never passes; a conversation step is judged with all of its transcripts;
a missing one raises; two passes, each straddling item adjudicated; the
pending block is non-passing until the judging completes; the preflight
reads one JSON object. The model call is the one mock."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("judge", ROOT / "scripts" / "redesign" / "judge.py")
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)
SUITE = judge.load_suite()
FIXTURES = ROOT / "tests" / "fixtures" / "semantic_transcripts"
DIMENSIONS = SUITE["tests"]["S2"]["judge"]["dimensions"]
WATCH = "judge/watch.rendered.md"


def test_the_rubrics_carry_the_designs_dimensions_by_name():
    for test, rubric in (("S2", "s2"), ("S4", "s4"), ("S1", "s1"), ("author", "author")):
        text = (judge.RUBRICS / f"{rubric}.md").read_text()
        for dimension in SUITE["tests"][test]["judge"]["dimensions"]:
            assert f"**{dimension}**" in text, (rubric, dimension)
    assert "findings the user did not know, with evidence" in (judge.RUBRICS / "s2.md").read_text()
    # The not-annoying clause reads the step's own line budget, which the judge's material states (gather_materials).
    assert "no message longer than the step's own line budget" in (judge.RUBRICS / "s2.md").read_text()
    assert "a loader that ingests" in (judge.RUBRICS / "s1.md").read_text()


def test_a_step_is_rendered_as_the_users_messages_and_every_message_the_host_printed_on_both_hosts():
    claude = judge.final_texts(judge.transcript_rows(FIXTURES / "claude_code-watch.jsonl"))
    assert [turn["role"] for turn in claude] == ["user", "assistant", "user"]
    assert claude[0]["text"] == "/kapso:watch ../churn-campaign" and claude[1]["text"].startswith("**Running**") and claude[2]["text"].startswith("thanks")
    codex = judge.final_texts(judge.transcript_rows(FIXTURES / "codex-watch.jsonl"))
    assert [turn["role"] for turn in codex] == ["user", "assistant", "assistant"] * 4, "every agent message of a turn is rendered, as the user read them, not the last alone"
    assert codex[1]["text"] != codex[2]["text"], "two messages, two blocks"
    rendered = judge.render_step("watch", judge.transcript_rows(FIXTURES / "claude_code-watch.jsonl"))
    assert rendered.startswith("L1: # step watch\n") and "L3: [user]\nL4: /kapso:watch ../churn-campaign\n" in rendered


def test_the_reply_must_be_one_json_object_with_the_rubrics_dimensions_and_evidence_on_its_line(tmp_path):
    rendered = tmp_path / "judge" / "watch.rendered.md"
    rendered.parent.mkdir()
    rendered.write_text("L1: # step watch\nL2: \nL3: [assistant]\nL4: Running, iteration 1 in ideation; no score yet.\n")
    good = {"scores": {d: {"score": 4, "evidence": [{"where": f"{WATCH}:L4", "quote": "no score yet"}]} for d in DIMENSIONS}}
    assert judge.validate(judge.extract_json("```json\n" + json.dumps(good) + "\n```"), DIMENSIONS, [1, 5], tmp_path, [WATCH])["clear"]["score"] == 4
    with pytest.raises(judge.JudgeFailure, match="not one JSON object"):
        judge.extract_json("Here are my scores: {}")
    with pytest.raises(json.JSONDecodeError):   # model output that is not JSON raises as it is (Rule 2); the pending block on disk is non-passing
        judge.extract_json("{not json,}")
    with pytest.raises(judge.JudgeFailure, match="the dimensions are"):
        judge.validate({"scores": {"clear": {"score": 4, "evidence": [{"where": f"{WATCH}:L4", "quote": "no score yet"}]}}}, DIMENSIONS, [1, 5], tmp_path, [WATCH])
    with pytest.raises(judge.JudgeFailure, match="evidence is empty"):
        judge.validate({"scores": {d: {"score": 4, "evidence": []} for d in DIMENSIONS}}, DIMENSIONS, [1, 5], tmp_path, [WATCH])
    wrong = json.loads(json.dumps(good))
    wrong["scores"]["clear"]["score"] = 6
    with pytest.raises(judge.JudgeFailure, match=r"clear: the score 6 is not an integer in \[1, 5\]"):
        judge.validate(wrong, DIMENSIONS, [1, 5], tmp_path, [WATCH])
    wrong = json.loads(json.dumps(good))
    wrong["scores"]["clear"]["evidence"] = [{"where": f"{WATCH}:L4", "quote": "the best score is 0.9"}]
    with pytest.raises(judge.JudgeFailure, match="is not on"):
        judge.validate(wrong, DIMENSIONS, [1, 5], tmp_path, [WATCH])
    wrong = json.loads(json.dumps(good))
    wrong["scores"]["clear"]["evidence"] = [{"where": "judge/elsewhere.md:L1", "quote": "x"}]
    with pytest.raises(judge.JudgeFailure, match="not the material"):
        judge.validate(wrong, DIMENSIONS, [1, 5], tmp_path, [WATCH])


def _record_dir(tmp_path, extra_transcripts=()):
    record_dir = tmp_path / "run"
    (record_dir / "transcripts").mkdir(parents=True)
    (record_dir / "transcripts" / "watch.jsonl").write_text((FIXTURES / "claude_code-watch.jsonl").read_text())
    for name in extra_transcripts:
        (record_dir / "transcripts" / name).write_text((FIXTURES / "claude_code-watch.jsonl").read_text())
    (record_dir / "record.json").write_text(json.dumps({
        "test": "S2", "host": "claude_code", "steps": [{"name": "watch", "kind": "conversation", "completed": True}, {"name": "triggers", "kind": "shell", "completed": True}],
    }))
    return record_dir


def _reply(scores):
    return json.dumps({"scores": {d: {"score": score, "evidence": [{"where": f"{WATCH}:L4", "quote": "/kapso:watch"}]} for d, score in zip(DIMENSIONS, scores)}})


def test_two_passes_judge_every_conversation_step_with_all_its_transcripts_and_a_straddle_is_adjudicated(tmp_path, monkeypatch):
    record_dir = _record_dir(tmp_path, extra_transcripts=("watch-2.jsonl",))
    replies = [_reply([5, 3, 5, 5, 5]), _reply([5, 5, 5, 5, 5]), _reply([5, 4, 5, 5, 5])]
    prompts = []
    monkeypatch.setattr(judge, "ask", lambda prompt, settings: (prompts.append(prompt), replies.pop(0))[1])
    block = judge.judge_record(record_dir, "s2", None, SUITE)
    assert block["failure"] is None and len(block["passes"]) == 2
    assert [p["items"]["watch"]["clear"]["score"] for p in block["passes"]] == [3, 5]
    assert block["adjudication"] == {"clear|watch": {"score": 4, "evidence": [{"where": f"{WATCH}:L4", "quote": "/kapso:watch"}]}}
    assert len(prompts) == 3 and "five dimensions" in prompts[0].lower() and "L4: /kapso:watch ../churn-campaign" in prompts[0]
    # The follow-up transcript was folded into the same rendered material.
    assert "watch-2" not in block["passes"][0]["items"] and (record_dir / "judge" / "watch.rendered.md").read_text().count("/kapso:watch ../churn-campaign") == 2
    written = json.loads((record_dir / "record.json").read_text())["judge"]
    assert written["model"] == SUITE["judge"]["model"] and (record_dir / "judge" / "raw-watch-pass1.txt").is_file()
    assert "triggers" not in written["passes"][0]["items"], "a shell step is not a conversation"


def test_a_malformed_reply_is_a_judge_failure_with_the_raw_reply_kept_never_a_pass(tmp_path, monkeypatch):
    record_dir = _record_dir(tmp_path)
    monkeypatch.setattr(judge, "ask", lambda prompt, settings: "I would rate this 4/5 overall.")
    with pytest.raises(judge.JudgeFailure, match="not one JSON object"):
        judge.judge_record(record_dir, "s2", None, SUITE)
    # The raw reply is kept and the on-disk record carries a non-passing
    # pending block, never a stale pass.
    assert (record_dir / "judge" / "raw-watch-pass1.txt").read_text() == "I would rate this 4/5 overall."
    assert json.loads((record_dir / "record.json").read_text())["judge"]["failure"] == "judging did not complete"
    with pytest.raises(ValueError, match="is judged by the rubric 's2', not 's4'"):
        judge.judge_record(record_dir, "s4", None, SUITE)


def test_a_conversation_step_with_no_transcript_raises(tmp_path):
    record_dir = tmp_path / "run"
    (record_dir / "transcripts").mkdir(parents=True)
    (record_dir / "record.json").write_text(json.dumps({"test": "S2", "host": "claude_code", "steps": [{"name": "watch", "kind": "conversation", "completed": True}]}))
    with pytest.raises(judge.JudgeFailure, match="watch has no transcript"):
        judge.gather_materials(record_dir, json.loads((record_dir / "record.json").read_text()), None, record_dir / "judge")


def test_main_prints_the_per_item_summary_without_crashing(tmp_path, monkeypatch, capsys):
    # main() runs the judge then prints a per-item, per-dimension summary over
    # the block it wrote; the summary must read the block's `items` shape, not
    # the model reply's `scores` key (a crash there aborts the runner's grade).
    record_dir = _record_dir(tmp_path)
    replies = [_reply([5, 5, 5, 5, 5]), _reply([5, 5, 5, 5, 5])]
    monkeypatch.setattr(judge, "ask", lambda prompt, settings: replies.pop(0))
    assert judge.main([str(record_dir), "--rubric", "s2"]) == 0
    out = capsys.readouterr().out
    assert f"watch / {DIMENSIONS[0]}: [5, 5]" in out


def test_the_preflight_reads_one_json_object(monkeypatch):
    monkeypatch.setattr(judge, "ask", lambda prompt, settings: '{"ok": true}')
    assert judge.preflight(SUITE).startswith("the judge's route answers")
    monkeypatch.setattr(judge, "ask", lambda prompt, settings: "ok")
    with pytest.raises(judge.JudgeFailure):
        judge.preflight(SUITE)


def test_the_personas_reply_is_rendered_once_and_an_unsent_follow_up_is_marked():
    # The driver writes the persona's reply as `persona_reply` and again as
    # the `proof_prompt` the host receives: one message. A reply the host
    # never received (the turn cap) is marked as never sent, so the judge
    # does not read it as a question the host ignored; DONE alone is the
    # persona satisfied.
    rows = [
        {"proof_prompt": "how is the campaign going?"},
        {"type": "result", "result": "Running, attempt 2 of 3."},
        {"persona_reply": "what is attempt 2 trying?", "done": False},
        {"proof_prompt": "what is attempt 2 trying?"},
        {"type": "result", "result": "A gradient-boosted model."},
        {"persona_reply": "and attempt 3?", "done": False},
    ]
    turns = judge.final_texts(rows)
    assert [(turn["role"], turn["text"]) for turn in turns] == [
        ("user", "how is the campaign going?"), ("assistant", "Running, attempt 2 of 3."),
        ("user", "what is attempt 2 trying?"), ("assistant", "A gradient-boosted model."),
        ("user", "and attempt 3?"),
    ]
    assert turns[-1]["unsent"] is True and "unsent" not in turns[2]
    rendered = judge.render_step("watch", rows)
    assert "[user, never sent: the step's turn cap ended the exchange here]" in rendered and rendered.count("[user]") == 2
    satisfied = judge.final_texts(rows[:5] + [{"persona_reply": "DONE", "done": True}])
    assert satisfied[-1] == {"role": "user", "text": "(the user is satisfied and ends the exchange: DONE)", "done": True}


def test_a_steps_transcripts_are_read_in_the_order_the_exchange_ran(tmp_path):
    # A lexical sort put `review-2.jsonl` before `review.jsonl` and the
    # judge read the follow-up first; a step whose name merely begins with
    # this one is not its follow-up.
    folder = tmp_path / "transcripts"
    folder.mkdir()
    for name in ("review-2.jsonl", "review.jsonl", "review-10.jsonl", "reviewer.jsonl", "review-x.jsonl"):
        (folder / name).write_text("{}\n", encoding="utf-8")
    assert [path.name for path in judge.step_transcripts(tmp_path, "review")] == ["review.jsonl", "review-2.jsonl", "review-10.jsonl"]


def test_a_quote_keeps_a_tokens_own_underscore_and_may_run_on_over_many_lines():
    lines = ["L1: the key `keep_me` is read", "L2: one", "L3: two", "L4: three", "L5: four", "L6: five"]
    assert judge.quote_on(lines, "1", "the key keep_me is read")
    assert not judge.quote_on(lines, "1", "the key keepme is read"), "an underscore inside a token is not markdown"
    assert judge.quote_on(["L1: _emphasised_ words"], "1", "emphasised words")
    assert judge.quote_on(lines, "2", "one two three four five"), "a quotation that starts on the cited line may run on"
    # It must start on the cited line, or the one before or after it: a judge
    # counting past a fence or a brace cites one off (rel-s4-codex cited the
    # `{` line for the `"branch"` line below it, twice); two lines off is not
    # the citation.
    assert judge.quote_on(lines, "3", "one two three"), "one line before the citation is the material's"
    assert judge.quote_on(lines, "1", "one two"), "one line after the citation is the material's"
    assert not judge.quote_on(lines, "4", "one two three"), "two lines off is not the citation"
    assert not judge.quote_on(lines, "6", "three four"), "nor two lines the other way"
    # A line that is a JSON string carries its escapes; the judge quotes the
    # words it reads (rel-s4-claude's grading died on `—` against an em dash).
    escaped = ['L1:     "feedback": "## Verdict: 0.8471 vs target 0.85 \\u2014 gap 0.0029. \\"Not achieved.\\"\\nNext line",']
    assert judge.quote_on(escaped, "1", "Verdict: 0.8471 vs target 0.85 — gap 0.0029. \"Not achieved.\" Next line")
    assert judge.quote_on(escaped, "1", "0.85 \\u2014 gap"), "a quote carrying the escape itself reads the same"


def test_a_test_whose_rubric_rules_on_several_inputs_at_once_gets_them_as_one_item(tmp_path):
    # S1's judge ruled on the flows file alone and scored the docs dimension
    # 0 for want of the answers it had been promised: a combined-inputs test
    # hands every input in one item, each file citable.
    for name, text in (("artifacts/comprehension-answers.txt", "the answers\nthe key"), ("artifacts/flows.md", "# flows\nscript a.py (3 code lines)")):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")
    record = {"test": "S1", "steps": []}
    materials = judge.gather_materials(tmp_path, record, ["artifacts/comprehension-answers.txt", "artifacts/flows.md"], tmp_path / "judge", SUITE)
    assert list(materials) == ["inputs"] and materials["inputs"]["files"] == ["artifacts/comprehension-answers.txt", "artifacts/flows.md"]
    assert "(file artifacts/comprehension-answers.txt)\nL1: the answers" in materials["inputs"]["text"] and "(file artifacts/flows.md)\nL1: # flows" in materials["inputs"]["text"]
    separate = judge.gather_materials(tmp_path, {"test": "S4", "steps": []}, ["artifacts/flows.md"], tmp_path / "judge", SUITE)
    assert list(separate) == ["flows"] and separate["flows"]["files"] == ["artifacts/flows.md"], "a test without combined_inputs judges each input alone"


def test_a_quote_that_is_only_a_fence_is_on_its_line_when_the_line_is_that_fence():
    # The judge cites the fenced brief as "a brief the host quotes back"
    # by quoting the fence line itself; folded to nothing it matched
    # nothing, and three re-judges died on the second slip.
    lines = ["L1: Here is the brief:", "L2: ```", "L3: objective: churn", "L4: ```"]
    assert judge.quote_on(lines, "2", "```")
    assert not judge.quote_on(lines, "1", "```"), "a fence quoted against a prose line is a misquote"
    assert not judge.quote_on(lines, "2", ""), "an empty quote is evidence of nothing"
