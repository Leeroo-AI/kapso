"""The deployment record and the deployment's identity
(docs/plans/redesign/wp5-deploy.md, change 2): numbers minted atomically,
a name and a port that carry the number and the campaign, a record that
reads back after the campaign moved, the states."""

import shutil
import socket
import threading

import pytest

from kapso.deployment.errors import DeployError
from kapso.deployment.record import (
    DEPLOYING,
    FAILED,
    RUNNING,
    STOPPED,
    allocate_port,
    campaign_hash,
    deployment_name,
    open_record,
    read_record,
    records,
    start_record,
)

PORTS = {"port_base": 48000, "port_span": 50}


def started(campaign, **overrides):
    fields = dict(
        target_requested="auto", host="claude_code:claude-opus-5", interpreter="/usr/bin/python3", env_names=["API_KEY"],
        options={}, commit="1a102cc2c8dc", branch="generic_exp_0", chosen_attempt=None, harness_built=None, harness=None, harness_root=None,
        **PORTS,
    )
    fields.update(overrides)
    return start_record(campaign, **fields)


def test_numbers_are_minted_once_each_under_contention(tmp_path):
    campaign = tmp_path / "Churn Build"
    campaign.mkdir()
    together = threading.Barrier(6)
    taken = []

    def mint():
        together.wait()
        taken.append(started(campaign).number)

    threads = [threading.Thread(target=mint) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(taken) == [1, 2, 3, 4, 5, 6]
    assert [record.number for record in records(campaign)] == [1, 2, 3, 4, 5, 6]


def test_the_identity_carries_the_campaign_and_the_number(tmp_path):
    first = tmp_path / "a" / "churn-build"
    second = tmp_path / "b" / "churn-build"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    one = started(first)
    two = started(first)
    other = started(second)
    assert one.data["deployment_name"] == f"churn-build-{campaign_hash(first)}-1"
    assert two.data["deployment_name"] == f"churn-build-{campaign_hash(first)}-2"
    assert deployment_name(first, 3) != deployment_name(second, 3)   # the same name, another path
    # Two deployments of one campaign never share a port: the number is in
    # the port's seed. Two campaigns may, since nothing listens on a port
    # until a deploy script binds it; the first to bind keeps it.
    assert one.data["port"] != two.data["port"]
    assert all(PORTS["port_base"] <= record.data["port"] < PORTS["port_base"] + PORTS["port_span"] for record in (one, two, other))


def test_a_taken_port_is_skipped_and_a_full_span_is_refused(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    wanted = allocate_port(campaign, 1, base=48100, span=4)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", wanted))
        holder.listen(1)
        taken = allocate_port(campaign, 1, base=48100, span=4)
        assert taken != wanted and 48100 <= taken < 48104
        holders = []
        for port in range(48100, 48104):
            if port == wanted:
                continue
            other = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            other.bind(("127.0.0.1", port))
            other.listen(1)
            holders.append(other)
        with pytest.raises(DeployError, match="no free port in 48100..48103"):
            allocate_port(campaign, 1, base=48100, span=4)
        for other in holders:
            other.close()


def test_the_record_reads_back_after_the_campaign_moved(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    record = started(campaign, options={"schedule": "*/5 * * * *"})
    assert record.state == DEPLOYING and record.data["session"] == "deployment/1" and record.data["finished"] is None
    assert record.workspace == record.directory / "workspace" and record.target_dir == record.directory / "target"
    moved = tmp_path / "elsewhere" / "campaign"
    shutil.move(str(campaign), str(moved))
    again = open_record(moved)
    assert again.number == 1 and again.workspace == moved / ".kapso" / "deployments" / "1" / "workspace"
    assert again.data["options"] == {"schedule": "*/5 * * * *"} and again.data["env_names"] == ["API_KEY"]


def test_a_relative_campaign_path_opens_with_absolute_paths(tmp_path, monkeypatch):
    """`Software.open("campaign", 1)` from the campaign's parent: the
    record's paths are absolute, so the worker's working directory and
    the scripts' do not depend on the caller's."""
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    monkeypatch.chdir(tmp_path)
    started("campaign")
    record = open_record("campaign", 1)
    assert record.directory.is_absolute() and record.workspace == campaign / ".kapso" / "deployments" / "1" / "workspace"
    assert record.campaign == campaign and records("campaign")[0].directory.is_absolute()


def test_the_states_and_the_opening_by_number(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    with pytest.raises(DeployError, match=r"has no deployment \(looked at"):
        open_record(campaign)
    one = started(campaign)
    two = started(campaign)
    one.set_state(RUNNING)
    two.set_state(FAILED, error="the session was cut at its deadline")
    assert open_record(campaign).number == 2 and open_record(campaign).state == FAILED
    assert open_record(campaign, 1).state == RUNNING and read_record(two.directory).data["error"] == "the session was cut at its deadline"
    one.set_state(STOPPED)
    assert open_record(campaign, 1).data["state_at"] and open_record(campaign, 1).state == STOPPED
    with pytest.raises(DeployError, match="has no deployment 3"):
        open_record(campaign, 3)
    with pytest.raises(ValueError, match="unknown deployment state"):
        one.set_state("paused")


def test_a_change_re_reads_the_record_so_two_handles_never_undo_each_other(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    mine = started(campaign)
    theirs = open_record(campaign)
    theirs.update(endpoint="http://new:1", values={"image": "img:2"})
    mine.set_state(RUNNING)
    assert mine.data["endpoint"] == "http://new:1" and mine.data["values"] == {"image": "img:2"}
    on_disk = read_record(mine.directory).data
    assert on_disk["state"] == RUNNING and on_disk["endpoint"] == "http://new:1"
    # The file is replaced whole: no staging file is left beside it.
    assert sorted(p.name for p in mine.directory.iterdir()) == ["deployment.json"]
