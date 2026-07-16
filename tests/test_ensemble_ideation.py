"""Hermetic tests for ensemble ideation in the generic strategy.

Pins the fan-out contract: parallel CLI members pool <solution> candidates,
a selector-critic chooses one, failures degrade softly, and — critically —
omitting the config keeps the single-session path byte-identical.
"""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import kapso.execution.coding_agents.adapters.claude_code_agent as claude_module
import kapso.execution.search_strategies.generic.codex_ideation as codex_module
import kapso.gated_mcp as gated_mcp_module
from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.execution.search_strategies.generic.strategy import (
    GenericSearch,
    normalize_ensemble_member,
    normalize_ideation_ensemble,
)

CODEX_MEMBER = {"cli": "codex", "model": "gpt-5.6-sol", "effort": "xhigh", "lens": "data"}
CLAUDE_MEMBER = {"cli": "claude_code", "model": "claude-fable-5", "effort": "xhigh", "lens": "recipe"}
SELECTOR = {"cli": "claude_code", "model": "claude-fable-5", "effort": "xhigh"}

# Real candidates are plans, not phrases; keep test candidates above the
# degenerate-artifact floor.
def _plan(name):
    return (f"# Core Idea\n{name}: " + "concrete codable step. " * 5).strip()



# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad",
    [
        [],
        "not-a-list",
        [{"cli": "gemini", "model": "x"}],
        [{"cli": "codex"}],
        [{"cli": "codex", "model": " "}],
        [{"cli": "codex", "model": "m", "unknown_key": 1}],
    ],
)
def test_invalid_ensemble_configs_raise(bad):
    with pytest.raises(ValueError):
        normalize_ideation_ensemble(bad)


def test_ensemble_requires_selector_and_selector_must_be_claude():
    with _patched_super_init():
        with pytest.raises(ValueError, match="ideation_selector"):
            GenericSearch(
                SimpleNamespace(
                    params={"ideation_ensemble": [dict(CODEX_MEMBER)]}
                )
            )
        with pytest.raises(ValueError, match="claude_code"):
            GenericSearch(
                SimpleNamespace(
                    params={
                        "ideation_ensemble": [dict(CLAUDE_MEMBER)],
                        "ideation_selector": {"cli": "codex", "model": "m"},
                    }
                )
            )


@contextmanager
def _patched_super_init():
    from kapso.execution.search_strategies.base import SearchStrategy

    original = SearchStrategy.__init__

    def fake_init(self, config, workspace_dir=None, import_from_checkpoint=False):
        self.params = config.params or {}

    SearchStrategy.__init__ = fake_init
    yield
    SearchStrategy.__init__ = original


# ---------------------------------------------------------------------------
# Fan-out harness (mirrors test_parent_selection's detached-view scaffolding)
# ---------------------------------------------------------------------------

def make_ensemble_strategy(tmp_path, monkeypatch, *, ensemble, selector,
                           claude_output, selector_output, codex_output,
                           codex_timed_out=False, claude_success=True,
                           selector_success=True):
    events = {"claude_prompts": [], "codex_calls": [], "configs": []}
    selected_dir = str(tmp_path / "selected-parent")

    class FakeWorkspace:
        repo = object()

        @contextmanager
        def materialize_ref(self, ref):
            Path(selected_dir).mkdir(exist_ok=True)
            yield selected_dir

    class FakeAgent:
        def __init__(self, config):
            events["configs"].append(config)
            self._model = config.model
            # Member sessions carry the gate servers (possibly empty dict);
            # the selector config has no mcp_servers key at all.
            self._is_selector = "mcp_servers" not in config.agent_specific

        def initialize(self, workspace):
            pass

        def generate_code(self, prompt):
            events["claude_prompts"].append((self._is_selector, prompt))
            if self._is_selector:
                return SimpleNamespace(
                    success=selector_success,
                    output=selector_output if selector_success else "",
                    error=None if selector_success else "boom",
                    metadata={},
                )
            return SimpleNamespace(
                success=claude_success,
                output=claude_output if claude_success else "",
                error=None if claude_success else "CLI exited with code 1",
                metadata={},
            )

        def get_cumulative_cost(self):
            return 1.0

        def cleanup(self):
            pass

    def fake_codex(prompt, model, cwd, timeout_seconds, effort=None, artifacts_dir=None):
        events["codex_calls"].append(
            {"model": model, "cwd": cwd, "timeout": timeout_seconds,
             "effort": effort, "artifacts_dir": artifacts_dir, "prompt": prompt}
        )
        meta = {"last_message_empty": not codex_output, "stream_tail": "",
                "stream_path": None, "last_path": None}
        return codex_output, codex_timed_out, 1.0, meta

    monkeypatch.setattr(claude_module, "ClaudeCodeCodingAgent", FakeAgent)
    monkeypatch.setattr(codex_module, "run_codex_ideation", fake_codex)
    monkeypatch.setattr(
        gated_mcp_module, "get_mcp_config", lambda **kw: ({}, [])
    )
    monkeypatch.setattr(
        RepoMemoryManager, "load_from_git_branch",
        classmethod(lambda cls, repo, branch: {}),
    )
    monkeypatch.setattr(
        RepoMemoryManager, "render_summary_and_toc",
        classmethod(lambda cls, doc, max_chars=2500: "memory"),
    )

    strategy = GenericSearch.__new__(GenericSearch)
    strategy.workspace = FakeWorkspace()
    strategy.workspace_dir = str(tmp_path / "root")
    strategy.experiment_history_path = str(tmp_path / "history.json")
    strategy.ideation_gates = []
    strategy.gate_failure_policy = "skip"
    strategy.idea_generation_model = "unused-single-path-model"
    strategy._claude_auth_settings = {"auth_mode": "oauth"}
    strategy.aws_region = "us-east-1"
    strategy.ideation_timeout = 600
    strategy.budget_snapshot = None
    strategy.iteration_count = 0
    strategy.session_effort = None
    strategy.env_strip = []
    strategy.ideation_ensemble = ensemble
    strategy.ideation_selector = selector
    return strategy, events


def test_fanout_pools_candidates_and_selector_choice_wins(tmp_path, monkeypatch):
    strategy, events = make_ensemble_strategy(
        tmp_path, monkeypatch,
        ensemble=[dict(CODEX_MEMBER), dict(CLAUDE_MEMBER)],
        selector=dict(SELECTOR),
        claude_output=f"<solution>{_plan('claude A')}</solution>"
                      f"<solution>{_plan('claude B')}</solution>",
        codex_output=f"noise <solution>{_plan('codex A')}</solution> noise",
        selector_output=(
            "<selection_reasoning>codex A is time-fit</selection_reasoning>"
            "<solution>the synthesized winner</solution>"
        ),
    )
    solution, sections, telemetry = strategy._generate_solution("problem", "main")

    assert solution == "the synthesized winner"
    # selector prompt carried every pooled candidate
    selector_prompts = [p for is_sel, p in events["claude_prompts"] if is_sel]
    assert len(selector_prompts) == 1
    for text in (_plan("codex A"), _plan("claude A"), _plan("claude B")):
        assert text in selector_prompts[0]
    # member + selector costs both counted
    assert telemetry["cost_usd"] == pytest.approx(2.0)
    # codex ran in the materialized worktree with its own model/effort
    assert events["codex_calls"][0]["model"] == "gpt-5.6-sol"
    assert events["codex_calls"][0]["effort"] == "xhigh"


def test_selector_failure_falls_back_to_first_claude_candidate(tmp_path, monkeypatch):
    strategy, _ = make_ensemble_strategy(
        tmp_path, monkeypatch,
        ensemble=[dict(CODEX_MEMBER), dict(CLAUDE_MEMBER)],
        selector=dict(SELECTOR),
        claude_output=f"<solution>{_plan('claude first')}</solution>",
        codex_output=f"<solution>{_plan('codex first')}</solution>",
        selector_output="",
        selector_success=False,
    )
    solution, _, _ = strategy._generate_solution("problem", "main")
    assert solution == _plan("claude first")


def test_all_members_failing_falls_back_to_template(tmp_path, monkeypatch):
    strategy, _ = make_ensemble_strategy(
        tmp_path, monkeypatch,
        ensemble=[dict(CODEX_MEMBER), dict(CLAUDE_MEMBER)],
        selector=dict(SELECTOR),
        claude_output="",
        claude_success=False,
        codex_output="short",  # below salvage floor, no tags
        codex_timed_out=True,
        selector_output="unused",
    )
    solution, _, _ = strategy._generate_solution("problem", "main")
    assert "Fallback solution due to ideation failure" in solution


def test_single_candidate_skips_selector(tmp_path, monkeypatch):
    strategy, events = make_ensemble_strategy(
        tmp_path, monkeypatch,
        ensemble=[dict(CODEX_MEMBER)],
        selector=dict(SELECTOR),
        claude_output="unused",
        codex_output=f"<solution>{_plan('only codex')}</solution>",
        selector_output="unused",
    )
    solution, _, _ = strategy._generate_solution("problem", "main")
    assert solution == _plan("only codex")
    assert not [p for is_sel, p in events["claude_prompts"] if is_sel]


def test_codex_timeout_salvages_substantive_output(tmp_path, monkeypatch):
    long_untagged = "research notes about datasets " * 20
    strategy, _ = make_ensemble_strategy(
        tmp_path, monkeypatch,
        ensemble=[dict(CODEX_MEMBER)],
        selector=dict(SELECTOR),
        claude_output="unused",
        codex_output=long_untagged,
        codex_timed_out=True,
        selector_output="unused",
    )
    solution, _, _ = strategy._generate_solution("problem", "main")
    assert "Salvaged from a deadline-terminated ideation session" in solution
    assert "research notes" in solution


def test_no_ensemble_config_keeps_single_session_path(tmp_path, monkeypatch):
    strategy, events = make_ensemble_strategy(
        tmp_path, monkeypatch,
        ensemble=None,
        selector=None,
        claude_output="<solution>single path</solution>",
        codex_output="never called",
        selector_output="never called",
    )
    solution, _, _ = strategy._generate_solution("problem", "main")
    assert solution == "single path"
    assert events["codex_calls"] == []
    assert len(events["claude_prompts"]) == 1


# ---------------------------------------------------------------------------
# Codex runner unit
# ---------------------------------------------------------------------------

def test_codex_runner_builds_command_and_strips_openai_key(tmp_path, monkeypatch):
    captured = {}

    class FakeProcess:
        pid = 4242

        def poll(self):
            return 0

        def wait(self):
            return 0

    class FakeStdin:
        def __init__(self):
            self.data = ""
            self.closed = False

        def write(self, text):
            self.data += text

        def close(self):
            self.closed = True

    def fake_popen(cmd, cwd, env, stdin, stdout, stderr, text, start_new_session):
        captured.update(cmd=cmd, cwd=cwd, env=env, start_new_session=start_new_session)
        stdout.write("transcript echo of the prompt, then duplicates")
        last_path = cmd[cmd.index("--output-last-message") + 1]
        with open(last_path, "w") as fh:
            fh.write("<solution>from codex</solution>")
        proc = FakeProcess()
        proc.stdin = FakeStdin()
        captured["stdin_obj"] = proc.stdin
        return proc

    monkeypatch.setattr(codex_module.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setattr(codex_module.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("OPENAI_API_KEY", "leak-me-not")

    output, timed_out, duration, meta = codex_module.run_codex_ideation(
        prompt="the prompt",
        model="gpt-5.6-sol",
        cwd=str(tmp_path),
        timeout_seconds=5,
        effort="xhigh",
    )

    assert output == "<solution>from codex</solution>"
    assert timed_out is False
    assert duration >= 0
    assert meta["last_message_empty"] is False
    assert captured["cmd"][:6] == [
        "codex", "--search", "exec", "--sandbox", "read-only",
        "--skip-git-repo-check",
    ]
    assert "--output-last-message" in captured["cmd"]
    assert "gpt-5.6-sol" in captured["cmd"]
    assert 'model_reasoning_effort="xhigh"' in captured["cmd"]
    # the prompt must NEVER appear in argv (self-pkill hazard, run #8):
    assert "the prompt" not in captured["cmd"]
    assert captured["stdin_obj"].data == "the prompt"
    assert captured["stdin_obj"].closed is True
    assert "OPENAI_API_KEY" not in captured["env"]
    assert captured["cwd"] == str(tmp_path)
    assert captured["start_new_session"] is True


def test_codex_runner_missing_cli_fails_loud(monkeypatch, tmp_path):
    monkeypatch.setattr(codex_module.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Codex CLI not found"):
        codex_module.run_codex_ideation(
            prompt="p", model="m", cwd=str(tmp_path), timeout_seconds=1
        )


def test_pool_hygiene_drops_degenerate_and_duplicate_candidates(tmp_path, monkeypatch):
    real_plan = "# Core Idea\nswap the dataset for in-domain CoT " + "x" * 80
    strategy, events = make_ensemble_strategy(
        tmp_path, monkeypatch,
        ensemble=[dict(CODEX_MEMBER), dict(CLAUDE_MEMBER)],
        selector=dict(SELECTOR),
        # codex echo artifacts: a tag-phrase fragment and a duplicate of the
        # claude candidate must both be dropped before the selector runs.
        codex_output=(
            "<solution> and </solution>"
            f"<solution>{real_plan}</solution>"
            f"<solution>{real_plan}</solution>"
        ),
        claude_output=f"<solution>{real_plan}</solution>",
        selector_output="unused",
    )
    solution, _, _ = strategy._generate_solution("problem", "main")
    # after hygiene only ONE candidate remains -> selector skipped entirely
    assert solution == real_plan
    assert not [p for is_sel, p in events["claude_prompts"] if is_sel]


def test_codex_zero_candidates_retries_once(tmp_path, monkeypatch):
    calls = {"n": 0}
    plan = _plan("second try")

    strategy, events = make_ensemble_strategy(
        tmp_path, monkeypatch,
        ensemble=[dict(CODEX_MEMBER)],
        selector=dict(SELECTOR),
        claude_output="unused",
        codex_output="unused",
        selector_output="unused",
    )

    def flaky_codex(prompt, model, cwd, timeout_seconds, effort=None, artifacts_dir=None):
        calls["n"] += 1
        meta = {"last_message_empty": calls["n"] == 1, "stream_tail": "boom",
                "stream_path": None, "last_path": None}
        if calls["n"] == 1:
            return "", False, 1.0, meta
        return f"<solution>{plan}</solution>", False, 1.0, meta

    monkeypatch.setattr(codex_module, "run_codex_ideation", flaky_codex)
    solution, _, _ = strategy._generate_solution("problem", "main")
    assert calls["n"] == 2
    assert solution == plan


def test_codex_timeout_does_not_retry(tmp_path, monkeypatch):
    calls = {"n": 0}
    strategy, _ = make_ensemble_strategy(
        tmp_path, monkeypatch,
        ensemble=[dict(CODEX_MEMBER)],
        selector=dict(SELECTOR),
        claude_output="unused",
        codex_output="unused",
        selector_output="unused",
    )

    def timing_out_codex(prompt, model, cwd, timeout_seconds, effort=None, artifacts_dir=None):
        calls["n"] += 1
        return "short", True, 1.0, {"last_message_empty": True, "stream_tail": "",
                                    "stream_path": None, "last_path": None}

    monkeypatch.setattr(codex_module, "run_codex_ideation", timing_out_codex)
    solution, _, _ = strategy._generate_solution("problem", "main")
    assert calls["n"] == 1
    assert "Fallback solution due to ideation failure" in solution


def test_prompt_echo_candidates_are_dropped(tmp_path, monkeypatch):
    """A candidate that is verbatim part of OUR prompt is a transcript echo
    (run #8's 'blank template'), never a model contribution."""
    strategy, events = make_ensemble_strategy(
        tmp_path, monkeypatch,
        ensemble=[dict(CODEX_MEMBER)],
        selector=dict(SELECTOR),
        claude_output="unused",
        codex_output="PLACEHOLDER",
        selector_output="unused",
    )
    real = _plan("genuine")

    def echoing_codex(prompt, model, cwd, timeout_seconds, effort=None, artifacts_dir=None):
        # echo a large verbatim chunk of the prompt inside solution tags,
        # plus one genuine candidate
        echo = prompt[100:600]
        out = f"<solution>{echo}</solution><solution>{real}</solution>"
        return out, False, 1.0, {"last_message_empty": False, "stream_tail": "",
                                 "stream_path": None, "last_path": None}

    monkeypatch.setattr(codex_module, "run_codex_ideation", echoing_codex)
    solution, _, _ = strategy._generate_solution("problem", "main")
    assert solution == real  # echo dropped -> single candidate -> selector skipped


def test_skeleton_candidates_never_reach_selector(tmp_path, monkeypatch):
    from kapso.execution.search_strategies.generic.strategy import (
        is_degenerate_ensemble_candidate,
    )
    skeleton = (
        "# Why This Approach\n[How this builds on previous experiments]\n"
        "# Solution Steps\n1. [First step with specific details]\n"
        "# Hyperparameters\n- param1: value1\n# Rationale\n[Why this works]"
    )
    assert is_degenerate_ensemble_candidate(skeleton)
    assert not is_degenerate_ensemble_candidate(_plan("real thing"))


def test_codex_artifacts_persist_when_dir_given(tmp_path, monkeypatch):
    class FakeStdin:
        def write(self, t): pass
        def close(self): pass

    class FakeProcess:
        pid = 4242
        stdin = FakeStdin()
        def poll(self): return 0
        def wait(self): return 0

    def fake_popen(cmd, cwd, env, stdin, stdout, stderr, text, start_new_session):
        stdout.write("stream contents")
        last_path = cmd[cmd.index("--output-last-message") + 1]
        with open(last_path, "w") as fh:
            fh.write("<solution>persisted</solution>")
        return FakeProcess()

    monkeypatch.setattr(codex_module.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setattr(codex_module.subprocess, "Popen", fake_popen)
    art = str(tmp_path / "ideation")
    output, timed_out, duration, meta = codex_module.run_codex_ideation(
        prompt="p", model="gpt-5.6-sol", cwd=str(tmp_path),
        timeout_seconds=5, artifacts_dir=art,
    )
    assert output == "<solution>persisted</solution>"
    assert meta["stream_path"] and meta["last_path"]
    assert open(meta["stream_path"]).read() == "stream contents"
    assert open(meta["last_path"]).read() == "<solution>persisted</solution>"


def test_env_strip_reaches_member_and_selector_session_configs(tmp_path, monkeypatch):
    strategy, events = make_ensemble_strategy(
        tmp_path, monkeypatch,
        ensemble=[dict(CODEX_MEMBER), dict(CLAUDE_MEMBER)],
        selector=dict(SELECTOR),
        claude_output=f"<solution>{_plan('claude only')}</solution>",
        codex_output=f"<solution>{_plan('codex only')}</solution>",
        selector_output=(
            "<selection_reasoning>fine</selection_reasoning>"
            "<solution>winner</solution>"
        ),
    )
    strategy.env_strip = ["OPENAI_API_KEY"]

    strategy._generate_solution("problem", "main")

    # Containment boundary: every Claude session the strategy spawns (member
    # AND selector) must carry the strip list, or an agent inherits the
    # orchestrator's own LLM credential on official non-judge runs.
    assert events["configs"], "harness captured no session configs"
    for config in events["configs"]:
        assert config.agent_specific["env_strip"] == ["OPENAI_API_KEY"]
