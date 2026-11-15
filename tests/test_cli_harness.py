"""`kapso harness`: every subcommand's flags are accounted for; `new`
writes the two files, prints the check line, and the result passes check;
`check` exits 1 on a failure and 0 with warnings, printing the summary
line; `list` prints the empty-store line and the two sections; `add`
fetches and never fails on what the machine lacks; `remove` refuses a
path; `--harness` on `kapso build` and `kapso deploy` are accounted for."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pytest

import kapso.cli.build as build_cli
import kapso.cli.deploy as deploy_cli
import kapso.cli.harness as harness_cli
from kapso.cli import main
from kapso.harness import BUNDLED_ROOT
from tests.handles_support import git

TRADING = Path(__file__).resolve().parent.parent / "examples" / "harnesses" / "trading"


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["kapso", "harness", *argv])
    main()


@pytest.fixture
def settings(monkeypatch, tmp_path):
    values = {"store": str(tmp_path / "store"), "page_warning_lines": 40, "setup_minutes": 20,
              "verification_minutes": 10, "verification_attempts": 2, "server_tool_timeout_seconds": 600}
    monkeypatch.setattr(harness_cli, "load_config", lambda config=None: {"harness": dict(values)})
    monkeypatch.chdir(tmp_path)
    return values


def _subparsers():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    harness_cli.add_parser(subparsers)
    harness_parser = subparsers.choices["harness"]
    sub = next(action for action in harness_parser._actions if isinstance(action, argparse._SubParsersAction))
    return {name: {a.dest for a in p._actions if a.dest != "help"} for name, p in sub.choices.items()}


def test_every_subcommand_and_flag_is_accounted_for():
    assert _subparsers() == {
        "new": {"name", "dir", "config"},
        "check": {"ref", "config"},
        "add": {"ref", "config"},
        "update": {"ref", "config"},
        "remove": {"ref", "config"},
        "list": {"json", "config"},
        "export": {"ref", "host", "to", "config"},
    }


def test_new_writes_the_two_files_that_pass_check(settings, monkeypatch, capsys):
    _run(monkeypatch, "new", "lending")
    root = Path.cwd() / "lending"
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) == [
        "harness.yaml", "instructions/guardrails.md",
    ]
    assert "name: lending" in (root / "harness.yaml").read_text()
    assert "# Lending guardrails (every module)" in (root / "instructions" / "guardrails.md").read_text()
    out = capsys.readouterr().out
    assert f"created {root}" in out and "kapso harness check ./lending" in out

    _run(monkeypatch, "check", "./lending")
    out = capsys.readouterr().out
    assert out.startswith("check ok (0 skills, 0 agents, 0 servers, 0 hooks)")
    assert "name     lending 0.1.0" in out


def test_new_refuses_a_bad_name_and_an_existing_directory(settings, monkeypatch, capsys):
    with pytest.raises(SystemExit) as caught:
        _run(monkeypatch, "new", "Lending")
    assert caught.value.code == 1 and "matches [a-z0-9][a-z0-9-]*" in capsys.readouterr().err
    (Path.cwd() / "taken").mkdir()
    with pytest.raises(SystemExit):
        _run(monkeypatch, "new", "taken")
    assert "exists; choose another name" in capsys.readouterr().err


def test_check_exits_one_on_a_failure_and_zero_with_warnings(settings, monkeypatch, capsys):
    copy = Path.cwd() / "trading"
    shutil.copytree(TRADING, copy)
    _run(monkeypatch, "check", "./trading")
    assert capsys.readouterr().out.startswith("check ok (7 skills, 2 agents, 2 servers, 7 hooks)")

    (copy / "hooks" / "extra").mkdir()
    _run(monkeypatch, "check", "./trading")
    out = capsys.readouterr().out
    assert out.startswith("check ok (7 skills, 2 agents, 2 servers, 8 hooks)") and "warn   hooks/extra/" in out

    (copy / "agents" / "risk-reviewer.md").unlink()
    with pytest.raises(SystemExit) as caught:
        _run(monkeypatch, "check", "./trading")
    assert caught.value.code == 1
    out = capsys.readouterr().out
    assert out.startswith("check failed: 1 failure(s)") and "lists agent `risk-reviewer`" in out


def test_list_has_an_empty_state_and_the_two_sections(settings, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(harness_cli, "BUNDLED_ROOT", tmp_path / "no-bundled")
    _run(monkeypatch, "list")
    assert capsys.readouterr().out.startswith("no harnesses yet: write one with `kapso harness new <name>`")

    bundled = tmp_path / "bundled"
    shutil.copytree(TRADING, bundled / "trading")
    monkeypatch.setattr(harness_cli, "BUNDLED_ROOT", bundled)
    _run(monkeypatch, "list")
    out = capsys.readouterr().out
    assert out.startswith("bundled\n  trading                  0.1.0")
    assert "fetched" not in out


def test_list_json_gives_every_harness_its_state(settings, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(harness_cli, "BUNDLED_ROOT", tmp_path / "no-bundled")
    _run(monkeypatch, "list", "--json")
    assert json.loads(capsys.readouterr().out) == []

    bundled = tmp_path / "bundled"
    _run(monkeypatch, "new", "lending", "--dir", str(bundled))
    shutil.copytree(TRADING, bundled / "trading")
    shutil.copytree(TRADING, bundled / "broken")
    shutil.rmtree(bundled / "broken" / "instructions")
    for name in ("trading", "broken"):
        manifest = bundled / name / "harness.yaml"
        manifest.write_text(manifest.read_text().replace(
            "environment:\n  python: requirements.txt\n", "environment:\n  python: requirements.txt\n  commands: [kapso-test-missing-tool]\n",
        ))
    monkeypatch.setattr(harness_cli, "BUNDLED_ROOT", bundled)
    capsys.readouterr()

    _run(monkeypatch, "list", "--json")
    rows = {row["path"].rsplit("/", 1)[1]: row for row in json.loads(capsys.readouterr().out)}

    assert rows["lending"] == {
        "name": "lending", "version": rows["lending"]["version"], "source": "bundled", "reference": None,
        "path": str(bundled / "lending"), "state": "ready", "needs": [], "failures": 0,
    }
    assert (rows["trading"]["name"], rows["trading"]["state"]) == ("trading", "needs")
    assert "kapso-test-missing-tool (command)" in rows["trading"]["needs"]
    assert (rows["broken"]["state"], rows["broken"]["needs"], rows["broken"]["failures"]) == ("fails", [], 1)

    _run(monkeypatch, "list")
    text = capsys.readouterr().out
    assert "0.1.0      fails check (1 failure(s))\n" in text
    assert "0.1.0      ready\n" in text
    assert "0.1.0      needs kapso-test-missing-tool (command)" in text


def test_list_json_marks_a_fetched_harness_with_its_reference(settings, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(harness_cli, "BUNDLED_ROOT", tmp_path / "no-bundled")
    work = tmp_path / "work"
    work.mkdir()
    git(work, "init", "-q", "-b", "main")
    shutil.copytree(TRADING, work / "trading")
    git(work, "add", ".")
    git(work, "commit", "-q", "-m", "trading")
    bare = tmp_path / "acme-harnesses.git"
    git(tmp_path, "clone", "-q", "--bare", str(work), str(bare))
    _run(monkeypatch, "add", f"git+file://{bare}@main#trading")
    capsys.readouterr()

    _run(monkeypatch, "list", "--json")

    [row] = json.loads(capsys.readouterr().out)
    assert (row["name"], row["source"]) == ("trading", "fetched")
    assert row["reference"].startswith(f"{bare.as_uri()}@") and row["reference"].endswith("#trading")


def test_add_fetches_checks_and_never_fails_on_needs(settings, monkeypatch, capsys, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    git(work, "init", "-q", "-b", "main")
    shutil.copytree(TRADING, work / "trading")
    manifest = work / "trading" / "harness.yaml"
    manifest.write_text(manifest.read_text().replace(
        "environment:\n  python: requirements.txt\n", "environment:\n  python: requirements.txt\n  commands: [kapso-test-missing-tool]\n",
    ))
    git(work, "add", ".")
    git(work, "commit", "-q", "-m", "trading")
    bare = tmp_path / "acme-harnesses.git"
    git(tmp_path, "clone", "-q", "--bare", str(work), str(bare))

    _run(monkeypatch, "add", f"git+file://{bare}@main#trading")
    out = capsys.readouterr().out
    assert out.startswith("fetched ") and "check ok (7 skills" in out
    assert "needs  kapso-test-missing-tool (command): install kapso-test-missing-tool" in out

    _run(monkeypatch, "list")
    out = capsys.readouterr().out
    assert "fetched\n  trading                  0.1.0" in out and "needs kapso-test-missing-tool (command)" in out

    _run(monkeypatch, "update", f"git+file://{bare}@main#trading")
    assert ": a tag or a commit never moves" in capsys.readouterr().out

    _run(monkeypatch, "remove", f"git+file://{bare}@main#trading")
    assert capsys.readouterr().out.startswith("removed ")
    with pytest.raises(Exception, match="is a local path; nothing was fetched"):
        _run(monkeypatch, "remove", str(TRADING))


def test_the_build_and_deploy_parsers_take_the_harness_flag():
    for module in (build_cli, deploy_cli):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        module.add_parser(subparsers)
        dests = {a.dest for a in subparsers.choices[module.NAME]._actions}
        assert "harness" in dests
