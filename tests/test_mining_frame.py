# Mining frame tests — the P2 contract (docs/plans/learning/p2-mine-corpus.md).
#
# The provider boundary is the coding agent: a fake lead writes mined/ trees,
# and the tests assert on the frame's real behavior — staging, the mechanical
# validators (format, coverage arithmetic, ref resolution/re-grep, index
# consistency, raw immutability), the repair loop, and the derived-layer
# commit. Each test names the regression it catches (Rule 9).

from pathlib import Path

import pytest

from kapso.learning.mining import (
    FORMAT_DOC_NAME,
    MinedViewValidator,
    MiningFrame,
)
from kapso.learning.trajectory_store import TrajectoryStore, save_trajectory
from tests.test_trajectory_store import TRAJECTORY_ID, build_work_dir

MINING_CONFIG = {
    "lead": {"cli": "claude_code", "model": "m", "effort": "xhigh", "auth_mode": "oauth"},
    "flow_writer_model": "writer-model",
    "critic_model": "critic-model",
    "timeout_minutes": 1,
    "repair_rounds": 1,
}


class FakeResult:
    def __init__(self, output):
        self.success = True
        self.output = output


class FakeLead:
    """Stands at the provider boundary: 'runs a session' by invoking a writer
    callback against the bundle dir, returning a canned mining report."""

    def __init__(self, writers):
        self.writers = list(writers)
        self.prompts = []
        self.workspace = None

    def initialize(self, workspace):
        self.workspace = workspace

    def generate_code(self, prompt, debug_mode=False, timeout_seconds=None):
        self.prompts.append(prompt)
        writer = self.writers.pop(0)
        return FakeResult(writer(self.workspace))


class FakeFactory:
    def __init__(self, lead):
        self.lead = lead
        self.configs = []

    def create(self, config):
        self.configs.append(config)
        return self.lead


def make_frame(tmp_path, writers):
    store = TrajectoryStore(local=str(tmp_path / "store"))
    work_dir, log = build_work_dir(tmp_path)
    save_trajectory(
        store, TRAJECTORY_ID, work_dir=str(work_dir), campaign_log=str(log)
    )
    lead = FakeLead(writers)
    frame = MiningFrame(store, MINING_CONFIG, agent_factory=FakeFactory(lead))
    return frame, store, lead


def write_valid_mined(workspace):
    """A minimal contract-conformant mined view over build_work_dir's bundle
    (one run: run_0001)."""
    mined = Path(workspace) / "mined"
    (mined / "it-1").mkdir(parents=True, exist_ok=True)
    (mined / "index.md").write_text(
        "# Campaign\n- [it-1](it-1/index.md) — the only round\n"
    )
    (mined / "strategy.md").write_text("## Lens 1\nbelief -> evidence -> re-aim\n")
    (mined / "operations.md").write_text("No incidents.\n")
    (mined / "artifacts.md").write_text("No shared-space artifacts.\n")
    (mined / "it-1" / "index.md").write_text(
        "# it-1\n- [flow-1](flow-1.md) — the winning flow\n"
    )
    (mined / "it-1" / "flow-1.md").write_text(
        "---\n"
        "flow: it-1/flow-1\n"
        "status: judged\n"
        "runs: [run_0001]\n"
        "sources:\n"
        "  idea: campaign.log#\"campaign started\"\n"
        "  evaluation: runs/run_0001/metrics.json\n"
        "---\n"
        "## Idea\nVerbatim idea [campaign.log#L1].\n"
        "## Evaluation\nScored 0.7136 [runs/run_0001/metrics.json].\n"
    )
    return "MINING REPORT: 1 flow written; no gaps."


def test_mine_end_to_end_commits_derived_layer(tmp_path):
    # Regression: the full frame path — staging, launch, validation green,
    # report filed, staging droppings removed, derived block recorded.
    frame, store, lead = make_frame(tmp_path, [write_valid_mined])
    mined_dir = frame.mine(TRAJECTORY_ID)
    assert (mined_dir / "mining-report.md").read_text().startswith("MINING REPORT")
    manifest = store.manifest(TRAJECTORY_ID)
    assert manifest["derived"]["mined"]["files"] == 7
    bundle_dir = store.resolve(TRAJECTORY_ID)
    assert not (bundle_dir / FORMAT_DOC_NAME).exists()
    assert not (bundle_dir / ".claude").exists()
    # idempotent: a second mine() does not re-launch a session
    frame.mine(TRAJECTORY_ID)
    assert len(lead.prompts) == 1


def test_staging_places_contract_and_templated_agents(tmp_path):
    # Regression: the crew must find the contract at the root and agent
    # definitions with the config's models substituted.
    captured = {}

    def snoop(workspace):
        root = Path(workspace)
        captured["format"] = (root / FORMAT_DOC_NAME).is_file()
        captured["writer"] = (root / ".claude" / "agents" / "flow-writer.md").read_text()
        captured["critic"] = (root / ".claude" / "agents" / "critic.md").read_text()
        return write_valid_mined(workspace)

    frame, _, lead = make_frame(tmp_path, [snoop])
    frame.mine(TRAJECTORY_ID)
    assert captured["format"]
    assert "model: writer-model" in captured["writer"]
    assert "model: critic-model" in captured["critic"]
    assert "{{" not in captured["writer"]
    # the launch prompt carries the trajectory id, not the placeholder
    assert TRAJECTORY_ID in lead.prompts[0]


def test_repair_loop_fixes_then_commits(tmp_path):
    # Regression: findings go back to the lead by name exactly once; a
    # repaired view commits.
    def write_broken(workspace):
        report = write_valid_mined(workspace)
        flow = Path(workspace) / "mined" / "it-1" / "flow-1.md"
        flow.write_text(flow.read_text().replace("status: judged", "status: bogus"))
        return report

    def repair(workspace):
        flow = Path(workspace) / "mined" / "it-1" / "flow-1.md"
        flow.write_text(flow.read_text().replace("status: bogus", "status: judged"))
        return "MINING REPORT after repair."

    frame, store, lead = make_frame(tmp_path, [write_broken, repair])
    frame.mine(TRAJECTORY_ID)
    assert "status: 'bogus'" in lead.prompts[1] or "bogus" in lead.prompts[1]
    assert store.manifest(TRAJECTORY_ID)["derived"]["mined"]


def test_unrepaired_findings_fail_loud(tmp_path):
    # Regression: one repair round then RuntimeError — never a silent commit
    # of a violating view.
    def write_broken(workspace):
        report = write_valid_mined(workspace)
        (Path(workspace) / "mined" / "strategy.md").unlink()
        return report

    frame, store, _ = make_frame(tmp_path, [write_broken, write_broken])
    with pytest.raises(RuntimeError, match="strategy.md"):
        frame.mine(TRAJECTORY_ID)
    assert "derived" not in store.manifest(TRAJECTORY_ID)


def test_raw_mutation_is_detected(tmp_path):
    # Regression: the raw bundle is read-only to the crew — a doctored raw
    # file must surface as a named finding via the manifest hashes.
    def write_and_mutate(workspace):
        report = write_valid_mined(workspace)
        (Path(workspace) / "campaign_meta.json").write_text('{"doctored": true}')
        return report

    frame, _, _ = make_frame(tmp_path, [write_and_mutate, write_valid_mined])
    with pytest.raises(RuntimeError, match="campaign_meta.json was modified"):
        frame.mine(TRAJECTORY_ID)


def make_validator(tmp_path, mutate):
    """Build a saved bundle + valid mined view, apply `mutate`, validate."""
    store = TrajectoryStore(local=str(tmp_path / "store"))
    work_dir, log = build_work_dir(tmp_path)
    save_trajectory(
        store, TRAJECTORY_ID, work_dir=str(work_dir), campaign_log=str(log)
    )
    bundle_dir = store.resolve(TRAJECTORY_ID)
    write_valid_mined(bundle_dir)
    (bundle_dir / "mined" / "mining-report.md").write_text("report: none skipped")
    mutate(bundle_dir)
    return MinedViewValidator(bundle_dir, store.manifest(TRAJECTORY_ID)).validate(
        verify_hashes=False
    )


def test_validator_catches_fabricated_quote(tmp_path):
    # Regression: a quote that does not re-grep is a rejected document — the
    # anti-fabrication check.
    def mutate(bundle_dir):
        flow = bundle_dir / "mined" / "it-1" / "flow-1.md"
        flow.write_text(
            flow.read_text().replace('campaign.log#"campaign started"',
                                     'campaign.log#"never was written"')
        )

    findings = make_validator(tmp_path, mutate)
    assert any("does not re-grep" in f for f in findings)


def test_validator_catches_unresolvable_ref(tmp_path):
    # Regression: refs must point into the raw inventory.
    def mutate(bundle_dir):
        flow = bundle_dir / "mined" / "it-1" / "flow-1.md"
        flow.write_text(
            flow.read_text().replace("runs/run_0001/metrics.json",
                                     "runs/run_0099/metrics.json")
        )

    findings = make_validator(tmp_path, mutate)
    assert any("run_0099" in f and "not in the raw inventory" in f for f in findings)


def test_validator_coverage_arithmetic(tmp_path):
    # Regression: a bundle run neither claimed by a flow nor named in the
    # report as skipped is a coverage hole; a run claimed twice is a clash.
    def unclaim(bundle_dir):
        flow = bundle_dir / "mined" / "it-1" / "flow-1.md"
        flow.write_text(flow.read_text().replace("runs: [run_0001]", "runs: []"))
        (bundle_dir / "mined" / "mining-report.md").write_text("says nothing")

    findings = make_validator(tmp_path / "unclaimed", unclaim)
    assert any("run_0001 is neither claimed" in f for f in findings)

    def double_claim(bundle_dir):
        it_dir = bundle_dir / "mined" / "it-1"
        original = (it_dir / "flow-1.md").read_text()
        (it_dir / "flow-2.md").write_text(original.replace("it-1/flow-1", "it-1/flow-2"))
        index = it_dir / "index.md"
        index.write_text(index.read_text() + "- [flow-2](flow-2.md) — twin\n")

    findings = make_validator(tmp_path / "doubled", double_claim)
    assert any("claimed by both" in f for f in findings)


def test_validator_index_listing_mismatch(tmp_path):
    # Regression: an index must list exactly the flow files present.
    def mutate(bundle_dir):
        index = bundle_dir / "mined" / "it-1" / "index.md"
        index.write_text("# it-1\n- [ghost](flow-9.md) — does not exist\n")

    findings = make_validator(tmp_path, mutate)
    assert any("does not list flow-1.md" in f for f in findings)
    assert any("lists flow-9.md which does not exist" in f for f in findings)


def test_validator_skipped_run_named_in_report_passes(tmp_path):
    # Regression: the explicitly-skipped escape hatch — a run named in the
    # mining report is accounted for.
    def mutate(bundle_dir):
        flow = bundle_dir / "mined" / "it-1" / "flow-1.md"
        flow.write_text(flow.read_text().replace("runs: [run_0001]", "runs: []"))
        (bundle_dir / "mined" / "mining-report.md").write_text(
            "skipped run_0001: infra-dead lane, no flow"
        )

    findings = make_validator(tmp_path, mutate)
    assert not findings
