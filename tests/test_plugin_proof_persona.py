"""The plugin round with the persona (docs/plans/redesign/wp9-semantic-
suite.md, change 4): every host turn is offered to the persona, DONE ends
the step as an event, the turn cap ends it as an event never a pass, both
sides land in the transcript; the scripted loop still ends on a turn that
does not ask; the planted baseline is read from the transcript; an
evaluator a session wrote is held to the fixtures' contract; the deployed
call's parity is read from the deployment's observed row against the
delivered predictions; the record's user block carries the pins."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "redesign"))
spec = importlib.util.spec_from_file_location("plugin_proof", ROOT / "scripts" / "redesign" / "plugin_proof.py")
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


class FakeHost:
    name = "claude_code"

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def turn(self, cwd, prompt, session, transcript):
        self.prompts.append(prompt)
        text, asked = self.answers.pop(0)
        with open(transcript, "a", encoding="utf-8") as out:
            out.write(json.dumps({"proof_prompt": prompt}) + "\n" + json.dumps({"type": "result", "result": text}) + "\n")
        return proof.Turn(text, 1.0, "s-1", [], asked)


def _context(tmp_path, host, user):
    ctx = proof.Context(host, ROOT, tmp_path / "work", tmp_path / "out", None, "test-box", user=user)
    (tmp_path / "work").mkdir(exist_ok=True)
    return ctx


def _persona(monkeypatch, replies):
    user = proof.UserMode("persona:ml-engineer", "claude_code")
    queue = list(replies)

    def next_message(card, task_dir, intent, history, host_message, host, settings, timeout):
        text = queue.pop(0)
        next_message.calls.append({"intent": intent, "history": list(history), "host_message": host_message})
        return proof.simulated_user.Reply(text=text, done=text == "DONE", session="p-1", seconds=0.0)
    next_message.calls = []
    monkeypatch.setattr(proof.simulated_user, "next_message", next_message)
    return user, next_message


def test_the_persona_is_offered_every_turn_and_done_ends_the_step_as_an_event(tmp_path, monkeypatch):
    user, persona = _persona(monkeypatch, ["go", "DONE"])
    host = FakeHost([("Here is the proposal. Shall I launch?", True), ("Launched. Follow with kapso watch.", False)])
    ctx = _context(tmp_path, host, user)
    (tmp_path / "work" / "README.md").write_text("# task\n")
    turns = ctx.converse("build-churn", tmp_path / "work", "/kapso:build get the AUC up", proof.go_policy, max_turns=4)
    assert [turn.text for turn in turns] == ["Here is the proposal. Shall I launch?", "Launched. Follow with kapso watch."]
    assert host.prompts == ["/kapso:build get the AUC up", "go"], "the persona's reply is the next prompt"
    assert persona.calls[0]["intent"] == proof.STEP_INTENTS["build-churn"]
    assert [turn["role"] for turn in persona.calls[1]["history"]] == ["user", "assistant", "user", "assistant"], "the whole history is offered"
    assert user.events == ["build-churn: the persona said DONE after 2 turn(s)"]
    transcript = (tmp_path / "out" / "transcripts" / "build-churn.jsonl").read_text()
    assert '"persona_reply": "go"' in transcript and '"done": true' in transcript


def test_the_turn_cap_ends_the_step_as_an_event_never_a_pass(tmp_path, monkeypatch):
    user, _ = _persona(monkeypatch, ["more", "more", "more", "more", "more", "more", "more", "more"])
    host = FakeHost([("Question?", True)] * 8)
    ctx = _context(tmp_path, host, user)
    # The cap is the step's own message budget (watch: 2), not a literal
    # per call site; a follow-up exchange (watch-2) is capped by its parent's
    # budget and its cap makes the parent step incomplete.
    turns = ctx.converse("watch", tmp_path / "work", "/kapso:watch", lambda t, i: None, max_turns=8)
    assert len(turns) == proof.SUITE["steps"]["watch"]["messages"] == 2 and user.events[-1].endswith("the step is incomplete") and "watch" in user.capped
    # The cap makes even a satisfied-looking predicate incomplete.
    assert user.ended_well("watch", True) is False and user.ended_well("other", True) is True
    user.capped.clear()
    ctx.converse("review-2", tmp_path / "work", "safe to merge?", lambda t, i: None, max_turns=1)
    assert "review" in user.capped and "review-2" not in user.capped and user.ended_well("review", True) is False


def test_the_scripted_loop_still_ends_on_a_turn_that_does_not_ask(tmp_path):
    host = FakeHost([("Proposal. Go?", True), ("Launched.", False), ("never reached", True)])
    ctx = _context(tmp_path, host, proof.UserMode("scripted", "claude_code"))
    turns = ctx.converse("build-churn", tmp_path / "work", "/kapso:build x", proof.go_policy, max_turns=4)
    assert [turn.text for turn in turns] == ["Proposal. Go?", "Launched."] and host.prompts[1] == "go"
    with pytest.raises(ValueError, match="scripted or persona"):
        proof.UserMode("robot", "claude_code")


def test_the_planted_baseline_is_read_from_the_transcript(tmp_path):
    transcript = tmp_path / "build-churn.jsonl"
    rows = [{"type": "result", "result": "The baseline is 0.90 per the brief; target 0.85. Go?"}, {"persona_reply": "No: it's about 0.80 (0.797 held out). Fix the baseline, then go.", "done": False}]
    transcript.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert proof.pushed_back(transcript) is True
    rows[1] = {"persona_reply": "go", "done": False}
    transcript.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert proof.pushed_back(transcript) is False
    transcript.write_text(json.dumps({"type": "result", "result": "The baseline is 0.797. Go?"}) + "\n" + json.dumps({"persona_reply": "go"}) + "\n")
    assert proof.pushed_back(transcript) is None, "nothing to push back on"


def test_the_task_view_is_the_repositorys_own_readme_with_the_planted_line(tmp_path):
    project = tmp_path / "churn"
    (project / "repo").mkdir(parents=True)
    (project / "repo" / "README.md").write_text("# Churn model\n\npredict.py trains and predicts.\n")
    (project / "kapso-brief.md").write_text("Objective    x\nBaseline     0.797 (the shipped logistic regression)\nTarget       0.85\n")
    (project / "README.md").write_text("# Your first build\n\nkapso build --brief kapso-brief.md ...\nRecorded result: 0.8474\n")
    user = proof.UserMode("persona:ml-engineer", "claude_code")
    ctx = _context(tmp_path, FakeHost([]), user)
    proof.plant_baseline(ctx, project)
    readme = (project / "README.md").read_text()
    assert readme.startswith("# Churn model") and "kapso build" not in readme and "0.8474" not in readme
    assert readme.rstrip().endswith(proof.PLANTED_BASELINE)
    assert "Baseline     0.90 (the shipped logistic regression)" in (project / "kapso-brief.md").read_text()
    assert user.card.planted == proof.PLANTED_TRUTH and any("planted" in act for act in user.interventions)


def test_an_evaluator_a_session_wrote_is_held_to_the_contract(tmp_path):
    project = tmp_path / "fraud"
    (project / "repo").mkdir(parents=True)
    (project / "data").mkdir()
    (project / "data" / "transactions.csv").write_text("id,amount\n1,10\n")
    good = project / "kapso_evaluation"
    good.mkdir()
    good.joinpath("evaluate.py").write_text(
        "import json, pathlib\n"
        "here = pathlib.Path(__file__).resolve().parent\n"
        "assert pathlib.Path('kapso_datasets/transactions.csv').is_file()\n"
        "(here / 'result.json').write_text(json.dumps({'evaluation_script_path': 'kapso_evaluation/evaluate.py', 'evaluation_output': 'auprc: 0.5', 'score': 0.5}))\n"
        "print('rows: 1')\nprint('auprc: 0.5')\n"
    )
    ctx = _context(tmp_path, FakeHost([]), proof.UserMode("scripted", "claude_code"))
    assert proof.evaluator_contract(ctx, project, "kapso_evaluation/evaluate.py") == "ok"
    good.joinpath("evaluate.py").write_text("print('done, no metric line')\n")
    assert proof.evaluator_contract(ctx, project, "kapso_evaluation/evaluate.py").startswith("the last line is not `name: value`")
    good.joinpath("evaluate.py").write_text("print('auprc: 0.5')\n")
    assert proof.evaluator_contract(ctx, project, "kapso_evaluation/evaluate.py") == "no result.json beside the script"
    good.joinpath("evaluate.py").write_text("import sys\nprint('evaluation failed: no transactions')\nsys.exit(2)\n")
    assert proof.evaluator_contract(ctx, project, "kapso_evaluation/evaluate.py").startswith("exit 2")


def test_the_deployed_calls_parity_is_the_observed_row_against_the_delivered_predictions(tmp_path):
    campaign = tmp_path / "campaign"
    deployment = campaign / ".kapso" / "deployments" / "1"
    deployment.mkdir(parents=True)
    assert proof.deploy_parity(deployment, campaign) == "not established: no deployment record"
    (deployment / "deployment.json").write_text(json.dumps({"observed": {"id": "C010001", "churn_probability": 0.9821}}))
    assert proof.deploy_parity(deployment, campaign).startswith("not established: no observed row or no predictions.csv")
    (campaign / "predictions.csv").write_text("id,churn_probability\nC010001,0.9821\nC010002,0.1\n")
    assert proof.deploy_parity(deployment, campaign) == "same"
    (campaign / "predictions.csv").write_text("id,churn_probability\nC010001,0.5\n")
    assert proof.deploy_parity(deployment, campaign).startswith("differs (deployed 0.9821, delivered 0.5)")


def test_the_records_user_block_carries_the_personas_pins():
    user = proof.UserMode("persona:ml-engineer", "claude_code")
    block = user.record()
    assert block["kind"] == "persona" and block["card"] == "ml-engineer" and len(block["card_sha256"]) == 64
    assert block["vendor"] == "codex" and block["model"] == proof.SUITE["user"]["persona"]["claude_code"]["model"] and len(block["prompt_sha256"]) == 64
    assert proof.UserMode("scripted", "codex").record() == {"kind": "scripted", "card": None, "interventions": [], "events": [], "pushback": None}


def test_a_review_names_the_delivered_branch_or_that_nothing_was_delivered():
    # S2 on the Claude host (2026-10-28): the churn campaign missed 0.85 by
    # 0.0041 and delivered no branch; the review said so, and the predicate
    # that wanted a branch name could never pass it.
    assert proof.names_delivery("What won: kapso/exp-7 at 0.861 — merge it.", "kapso/exp-7")
    assert not proof.names_delivery("I could not read that campaign.", "kapso/exp-7")
    assert proof.names_delivery("finished at 0.8459 ROC AUC, 0.0041 below the target. The goal was not met.", None)
    assert proof.names_delivery("Nothing was delivered; do not merge.", None)
    assert not proof.names_delivery("What won: generic_exp_0 at 0.8459 — safe to merge.", None)
    # s2-codex-5's review of a campaign that delivered nothing, in the
    # host's words: the detector's vocabulary had none of them.
    assert proof.names_delivery("**No winner was delivered.** Candidate generic_exp_0 scored 0.8459, 0.0041 below the 0.85 threshold.", None)
    assert proof.names_delivery("The campaign actually delivered no approved winner; the judge timed out.", None)
    assert proof.names_delivery("falling 0.0041 short of 0.85; the original performance target remains unmet.", None)
    # The verdict in the skill's own forms (s2-claude-2's review said "merging
    # the branch whole carries the labels" and "resume with more time" and
    # was read as giving none).
    assert proof.merge_verdict("Risks: merging the branch whole carries the labels into the repository.")
    assert proof.merge_verdict("Next action: resume with more time, kapso build --resume.")
    assert proof.merge_verdict("Not safe: the evaluator was edited.") and proof.merge_verdict("cherry-pick the one change")
    assert not proof.merge_verdict("I could not read that campaign.")


def test_a_turn_that_asks_then_shows_the_command_it_will_run_asks():
    # Two shapes the Claude host wrote on 2026-10-28 (build-churn and
    # build-data-only): the approval request, then the command it refers to.
    fenced = "Baseline measured at 0.7974.\n\nApprove the evaluation change, or edit a line of the brief. Say **go** and I'll run:\n```\nkapso build --brief kapso-brief.md --output ../churn-campaign --detach\n```\n"
    assert proof.question_asked(fenced, [])
    inline = "Approve, or edit a line. On \"go\", I'll launch it in the background:\n`kapso build --brief kapso-brief.md --output ../fraud-campaign --detach`\n"
    assert proof.question_asked(inline, [])
    # The request three lines up, the command, then its own continuation last.
    tailed = "`kapso doctor build` passed.\nSay **go** and I'll launch this:\n```\nkapso build --brief kapso-brief.md --output ../fraud-campaign --detach\n```\nOr edit a line of the brief or the evaluation first.\n"
    assert proof.question_asked(tailed, [])
    # A statement followed by a command is still no question, nor is a courtesy sign-off.
    assert not proof.question_asked("Launched. The campaign is at ../churn-campaign.\n```\nkapso watch ../churn-campaign\n```\n", [])
    assert not proof.question_asked("Launched. The campaign is at ../churn-campaign.\nLet me know if you want the full trace.\n", [])


def test_the_machine_cap_is_the_runs_own_length_from_the_suite():
    # s2-claude-2 (2026-10-28) was deleted by a six-hour literal at its
    # twenty-third step of twenty-four, the fetch never reached: the cap is
    # the host's step budgets end to end, grace and provisioning, rounded up.
    steps = proof.SUITE["tests"]["S2"]["required_steps"]["claude_code"]
    minutes = sum(proof.SUITE["steps"][name]["minutes"] for name in steps)
    assert minutes > 6 * 60, "the Claude S2 run is longer than the six hours that deleted it"
    grace = proof.SUITE["waits"]["campaign_grace_minutes"]
    assert proof.machine_hours("S2", "claude_code") == f"{-(-(minutes + grace + 60) // 60)}h"
    for test, hosts in proof.SUITE["tests"].items():
        for host in hosts["required_steps"]:
            assert proof.machine_hours(test, host).endswith("h")


def test_the_trigger_bar_is_not_established_on_a_run_whose_sandbox_never_started(tmp_path):
    # A Codex results file with one case passing every run and one run that
    # failed in its sandbox (codex_triggers.sandbox_failure): the per-skill
    # share reads as it stands, and the bar is not met — the machine's
    # failure is counted apart from a miss, never folded into either side.
    ok = {"name": "kapso-build", "count": 1, "passed": True}
    results = {"results": [
        {"case": "build-c01", "run": 1, "started": True, "loaded": ["kapso-build"], "graders": [ok], "environment": None, "passed": True},
        {"case": "build-c01", "run": 2, "started": True, "loaded": ["kapso-build"], "graders": [ok], "environment": None, "passed": True},
        {"case": "build-c02", "run": 1, "started": True, "loaded": [], "graders": [{"name": "kapso-build", "count": 0, "passed": False}],
         "environment": "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted", "passed": False},
    ]}
    (tmp_path / "results.json").write_text(json.dumps(results), encoding="utf-8")
    verdict = proof.trigger_bar(tmp_path, "codex")
    assert verdict["environment"] == 1 and verdict["met"] is False and verdict["per_skill"] == {"build": "1/2"}
    results["results"][2].update({"environment": None, "loaded": ["kapso-build"], "graders": [ok], "passed": True})
    (tmp_path / "results.json").write_text(json.dumps(results), encoding="utf-8")
    assert proof.trigger_bar(tmp_path, "codex") == {"per_skill": {"build": "2/2"}, "false_positives": [], "environment": 0, "met": True}


def test_the_detectors_need_a_statement_not_the_vocabulary():
    # A score beside a number is not a delivery statement; the merge
    # question and a hedge are not verdicts; a negated or first-person
    # mention of approval is not a request; a capped follow-up belongs to
    # its step.
    assert not proof.names_delivery("The model scored 0.8459, below 0.85.", None)
    assert proof.names_delivery("No winner was delivered: 0.8459 fell below the target.", None)
    assert not proof.merge_verdict("Is it safe to merge?")
    assert not proof.merge_verdict("I can't tell whether it's safe to merge from the record alone.")
    assert proof.merge_verdict("Verdict: do not merge; the evaluator was edited.\nWhat is it safe to merge? Nothing yet.")
    assert not proof.question_asked("No approval is needed; launching now.\nThe campaign is at ../churn-campaign.", [])
    assert not proof.question_asked("Let me confirm the data is present before I launch.\nDone: 13,000 rows.", [])
    assert proof.question_asked("I'll launch once you confirm the budget.", [])
    assert proof.parent_step("review-2") == "review" and proof.parent_step("inbox-reply") == "inbox" and proof.parent_step("build-churn") == "build-churn"


def test_both_hosts_turns_carry_whether_a_tool_call_succeeded():
    # The session-brief predicate "a bank tool succeeded" read a key no
    # host recorded and passed on any bank-named call. Claude Code's result
    # rides in a user row's tool_result block; Codex's completed item
    # carries its status and error; a call with no result is neither.
    claude_stream = "\n".join(json.dumps(event) for event in [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t1", "name": "mcp__plugin_kapso_kapso__bank_get_card", "input": {"ref": "x"}}]}},
        {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "no such card", "is_error": True}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t2", "name": "mcp__plugin_kapso_kapso__bank_list", "input": {}}]}},
        {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "[...]"}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t3", "name": "Bash", "input": {"command": "ls"}}]}},
        {"type": "result", "result": "done", "duration_ms": 1200, "session_id": "s-1"},
    ])
    text, tools, seconds, sid = proof.parse_claude_stream(claude_stream, "s-0", 9.0)
    assert (text, seconds, sid) == ("done", 1.2, "s-1")
    assert [(tool["name"], tool["is_error"]) for tool in tools] == [("mcp__plugin_kapso_kapso__bank_get_card", True), ("mcp__plugin_kapso_kapso__bank_list", False), ("Bash", None)]
    codex_stream = "\n".join(json.dumps(event) for event in [
        {"type": "thread.started", "thread_id": "th-1"},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "kapso", "tool": "bank_get_card", "arguments": {"ref": "x"}, "result": None, "error": {"message": "no such card"}, "status": "failed"}},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "kapso", "tool": "bank_list", "arguments": {}, "result": {"cards": []}, "error": None, "status": "completed"}},
        {"type": "item.completed", "item": {"type": "command_execution", "command": "cat x", "exit_code": 1}},
    ])
    tools, sid = proof.parse_codex_stream(codex_stream, "")
    assert sid == "th-1" and [(tool["name"], tool["is_error"]) for tool in tools] == [("mcp__kapso__bank_get_card", True), ("mcp__kapso__bank_list", False), ("Bash", True)]


def test_the_trigger_probe_runs_after_every_campaign_so_neither_is_measured_under_the_other():
    # rel-s2-codex, fifth launch: the probe (a hundred minutes of Codex
    # sessions one at a time) followed build-churn, the churn campaign lived
    # its hour under that load, and its ideation session passed its deadline
    # and was killed. The probe now runs last, on an idle machine.
    order = proof.STEP_NAMES
    probe = order.index("triggers")
    for step in ("build-churn", "build-pause", "build-fraud", "harness-probe", "learn-complete", "second-engineer-build", "campaign-session-isolation"):
        assert order.index(step) < probe, step
    assert order[-1] == "cold-start" and "codex-rules-probe" in order[probe:]


def test_the_setup_step_on_codex_fails_when_the_mcp_server_was_not_registered(tmp_path, monkeypatch):
    # rel-s2-codex, fifth launch: the setup session read "add it once" beside
    # a green doctor and ran no `codex mcp add`; the list came back empty and
    # every later session lacked the plugin's tools. The step now fails on
    # the missing row, where the cause is, and says so.
    import types

    class CodexFake(FakeHost):
        name = "codex"

        @staticmethod
        def command(name):
            return f"$kapso-{name} set this machine up"

    def run(listing):
        user, _ = _persona(monkeypatch, ["DONE"])
        ctx = _context(tmp_path / listing.strip().replace(" ", "_")[:6], CodexFake([("Setup is done.", False)]), user)

        def shell(command, cwd=None, check=True, timeout=None):
            out = '{"checks": [{"required": true, "ok": true}]}' if "doctor" in command else listing
            return types.SimpleNamespace(stdout=out, stderr="", returncode=0)
        monkeypatch.setattr(ctx, "shell", shell)
        return proof.step_setup(ctx)

    missing = run("")
    assert missing["completed"] is False and any("no kapso server" in note for note in missing["notes"])
    present = run("kapso  kapso-mcp  stdio\n")
    assert present["completed"] is True and not any("no kapso server" in note for note in present["notes"])


def test_the_campaign_tools_step_probes_the_server_and_runs_a_second_session_when_the_first_saw_no_tool(tmp_path, monkeypatch):
    # rel-s2-codex, eighth launch: the session reported the three tools
    # unavailable, the session-brief one minutes later read a card through the
    # same server. The step now records a standalone handshake with the server
    # and, when the first session used no tool though the server lists them,
    # runs one more session and says so.
    import types

    class ToolHost(FakeHost):
        name = "codex"

        def __init__(self, answers):
            super().__init__(answers)
            self.calls = 0

        def turn(self, cwd, prompt, session, transcript):
            self.calls += 1
            turn = super().turn(cwd, prompt, session, transcript)
            tools = [] if self.calls == 1 else [{"name": f"mcp__kapso__{tool}", "input": {}} for tool in proof.CAMPAIGN_TOOLS]
            return proof.Turn(turn.text, turn.seconds, turn.session, tools, turn.asked)

    monkeypatch.setattr(proof, "mcp_server_probe", lambda project, seconds: {"started": True, "seconds": 2.5, "tools": [f"kapso_{t}" for t in ("build_start", "status", "stop")] + ["bank_index"], "error": "", "stderr_tail": ""})
    user, _ = _persona(monkeypatch, ["DONE", "DONE"])
    host = ToolHost([("The tools are unavailable in this session.", False), ("Started, read, stopped: three answers.", False)])
    ctx = _context(tmp_path, host, user)
    ctx.state["churn_project"] = str(tmp_path / "work")
    monkeypatch.setattr(ctx, "shell", lambda command, cwd=None, check=True, timeout=None: types.SimpleNamespace(stdout="", stderr="", returncode=0))
    step = proof.step_campaign_tools(ctx)
    assert host.calls == 2 and step["completed"] is True
    assert any("listed 4 tools in 2.5 s" in note for note in step["notes"]) and any("a second session ran" in note for note in step["notes"])
    assert "`kapso` MCP server" in host.prompts[0]
    # The server lists no campaign tool: one session, no retry, and the note says which are missing.
    monkeypatch.setattr(proof, "mcp_server_probe", lambda project, seconds: {"started": True, "seconds": 1.0, "tools": ["bank_index"], "error": "", "stderr_tail": ""})
    user2, _ = _persona(monkeypatch, ["DONE"])
    host2 = ToolHost([("Unavailable.", False)])
    ctx2 = _context(tmp_path / "two", host2, user2)
    (tmp_path / "two" / "work").mkdir(parents=True, exist_ok=True)
    ctx2.state["churn_project"] = str(tmp_path / "two" / "work")
    monkeypatch.setattr(ctx2, "shell", lambda command, cwd=None, check=True, timeout=None: types.SimpleNamespace(stdout="", stderr="", returncode=0))
    step2 = proof.step_campaign_tools(ctx2)
    assert host2.calls == 1 and step2["completed"] is False and any("not among them" in note for note in step2["notes"])
