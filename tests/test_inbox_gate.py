"""The inbox gate's one tool, request_from_user (design v4 Appendix A.2).

What must hold: the schema requires the five fields; a call writes one
requested event per entry with the node and session from the injected
environment; a malformed call raises (the server turns it into a tool
error, never into prose); a post on an open key joins it; the result
text tells the session to stop and shows the previous reply when there
is one; the gate resolves only with all three variables and they are
forwarded into the bundled server's environment.
"""

import asyncio

import pytest

from kapso.execution.inbox import REQUEST_FIELDS, load_requests, record_reply
from kapso.gated_mcp import GATES, get_mcp_config
from kapso.gated_mcp.gates.inbox_gate import REQUEST_FROM_USER_TOOL, InboxGate
from kapso.gated_mcp.server import GATE_CLASSES

ENTRY = {
    "key": "env:OPENAI_API_KEY",
    "hit": "openai.AuthenticationError",
    "tried": "OPENAI_KEY unset too; README says export it",
    "fix": "add OPENAI_API_KEY=sk-... to /home/me/churn/.env",
    "next_steps": "embed, re-rank, evaluate",
}


@pytest.fixture
def gate_env(tmp_path, monkeypatch):
    path = tmp_path / ".kapso" / "inbox.jsonl"
    monkeypatch.setenv("KAPSO_INBOX_PATH", str(path))
    monkeypatch.setenv("KAPSO_SESSION_ID", "550e8400-e29b-41d4-a716-446655440000")
    monkeypatch.setenv("KAPSO_NODE_ID", "3")
    return path


def call(arguments):
    return asyncio.run(InboxGate().handle_call(REQUEST_FROM_USER_TOOL, arguments))


def test_tool_schema_requires_every_field():
    (tool,) = InboxGate().get_tools()
    assert tool.name == REQUEST_FROM_USER_TOOL
    items = tool.inputSchema["properties"]["requests"]["items"]
    assert items["required"] == list(REQUEST_FIELDS)
    assert tool.inputSchema["required"] == ["requests"]
    assert "STOPS your session" in tool.description
    assert GATE_CLASSES["inbox"] is InboxGate
    assert GATES["inbox"].tools == [REQUEST_FROM_USER_TOOL]


def test_call_files_requests_from_the_injected_environment(gate_env):
    (content,) = call({"requests": [ENTRY, {**ENTRY, "key": "data/x.csv"}]})
    assert content.text.startswith("Recorded as requests #1, #2.")
    requests = load_requests(gate_env)
    assert [r.key for r in requests.values()] == ["env:OPENAI_API_KEY", "data/x.csv"]
    assert requests[1].node == 3
    assert requests[1].session == "550e8400-e29b-41d4-a716-446655440000"
    assert requests[1].tried == ENTRY["tried"]


def test_malformed_call_raises_and_writes_nothing(gate_env):
    with pytest.raises(ValueError, match="missing"):
        call({"requests": [{"key": "k"}]})
    with pytest.raises(ValueError, match="non-empty list"):
        call({"requests": []})
    assert not gate_env.exists()


def test_open_key_joins_and_answered_key_shows_previous_reply(gate_env):
    call({"requests": [ENTRY]})
    (joined,) = call({"requests": [ENTRY]})
    assert joined.text.startswith("Recorded as request #1.")
    assert len(load_requests(gate_env)) == 1
    record_reply(gate_env, 1, "added the key")
    (again,) = call({"requests": [ENTRY]})
    assert again.text.startswith("Recorded as request #2.")
    assert "answered — 'added the key'" in again.text


def test_other_tools_are_not_handled(gate_env):
    assert asyncio.run(InboxGate().handle_call("bank_index", {})) is None


def test_gate_resolves_only_with_all_three_variables_and_forwards_them(tmp_path, monkeypatch):
    for name in ("KAPSO_INBOX_PATH", "KAPSO_SESSION_ID", "KAPSO_NODE_ID"):
        monkeypatch.delenv(name, raising=False)
    context = {
        "KAPSO_INBOX_PATH": str(tmp_path / "inbox.jsonl"),
        "KAPSO_SESSION_ID": "sid",
        "KAPSO_NODE_ID": "0",
    }
    servers, tools = get_mcp_config(["inbox"], inbox=context, include_base_tools=False)
    assert tools == ["mcp__gated-knowledge__request_from_user"]
    env = servers["gated-knowledge"]["env"]
    assert env["MCP_ENABLED_GATES"] == "inbox"
    assert {k: env[k] for k in context} == context

    partial = dict(context)
    del partial["KAPSO_NODE_ID"]
    servers, tools = get_mcp_config(["inbox"], inbox=partial, include_base_tools=False, gate_failure_policy="skip")
    assert tools == [] and servers == {}
