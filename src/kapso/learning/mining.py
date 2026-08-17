# Mining frame — stage / launch / check / commit around the mining crew.
#
# Design: learn-from-trajectories-design.md §3.4.1 + §4.2; the crew's
# instruction material lives in crews/mining/ (source of truth:
# docs/research/learn-from-trajectories-mining-prompts.md). The frame never
# orchestrates agents — it stages the contract and agent definitions, launches
# ONE Claude lead session in the bundle root (self-organization is the CLI's
# native Task tool), then mechanically verifies what came back: schema,
# coverage arithmetic on stable identities, ref resolution with quote re-grep,
# index consistency, and raw immutability against the manifest hashes. One
# repair round, then fail loud. The mined view is a derived layer inside the
# trajectory: regenerable, recorded in the manifest's `derived:` block.

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.learning.trajectory_store import MANIFEST_NAME, TrajectoryStore, _hash_file

CREW_DIR = Path(__file__).parent / "crews" / "mining"
MINED_DIR_NAME = "mined"
MINING_REPORT_NAME = "mining-report.md"
FORMAT_DOC_NAME = "mined-format.md"

FLOW_STATUSES = (
    "ideated", "selected-unbuilt", "build-failed", "evaluated", "judged", "champion"
)
CAMPAIGN_GRAIN_DOCS = ("index.md", "strategy.md", "operations.md", "artifacts.md")

_FLOW_FILE_PATTERN = re.compile(r"^it-\d+/flow-\d+\.md$")
_RUN_ID_PATTERN = re.compile(r"^runs/(run_\d+)/")
_MD_LINK_PATTERN = re.compile(r"\]\(([^)\s]+)\)")
_LINE_FRAGMENT_PATTERN = re.compile(r"^L\d+(-L?\d+)?$")


def _split_frontmatter(text: str) -> Optional[Dict[str, Any]]:
    """Parse a mined document's YAML frontmatter.

    Structural absence (no block, no terminator, not a mapping) returns None —
    a repairable finding for the crew. Genuinely malformed YAML raises
    (Rule 2: model output that cannot parse fails loud, it is not skipped).
    """
    if not text.startswith("---\n") or "\n---" not in text[4:]:
        return None
    end = text.index("\n---", 4)
    frontmatter = yaml.safe_load(text[4:end])
    return frontmatter if isinstance(frontmatter, dict) else None


def _ref_findings(ref: str, label: str, bundle_dir: Path, inventory: Dict[str, str]) -> List[str]:
    """Verify one ref into the raw bundle: path resolves; L-anchors and quoted
    fragments verify mechanically; other fragments are path-existence only."""
    path_part, _, fragment = str(ref).partition("#")
    if path_part not in inventory:
        return [f"{label}: ref path {path_part!r} is not in the raw inventory"]
    if not fragment:
        return []
    if _LINE_FRAGMENT_PATTERN.match(fragment):
        line_count = sum(1 for _ in open(bundle_dir / path_part, errors="replace"))
        first_line = int(fragment.split("-")[0][1:])
        if first_line > line_count:
            return [f"{label}: ref {ref!r} anchors line {first_line} but "
                    f"{path_part} has {line_count} lines"]
        return []
    if fragment.startswith('"') and fragment.endswith('"') and len(fragment) > 2:
        snippet = fragment[1:-1]
        if snippet not in open(bundle_dir / path_part, errors="replace").read():
            return [f"{label}: quote {snippet[:60]!r} does not re-grep in {path_part}"]
    return []


class MinedViewValidator:
    """The mechanical checks over a written mined/ tree. Pure functions of the
    bundle on disk + its manifest; every violation is a named finding."""

    def __init__(self, bundle_dir: Path, manifest: Dict[str, Any]):
        self.bundle_dir = bundle_dir
        self.mined_dir = bundle_dir / MINED_DIR_NAME
        self.inventory: Dict[str, str] = manifest["inventory"]["sha256"]
        self.bundle_runs = {
            match.group(1)
            for rel in self.inventory
            if (match := _RUN_ID_PATTERN.match(rel))
        }

    def validate(self, verify_hashes: bool = True) -> List[str]:
        findings: List[str] = []
        findings += self._check_layout()
        if findings:
            return findings  # nothing else is checkable without the layout
        flows = self._load_flows(findings)
        findings += self._check_coverage(flows)
        findings += self._check_refs(flows)
        findings += self._check_indexes()
        if verify_hashes:
            findings += self._check_raw_immutability()
        return findings

    # ------------------------------------------------------------ layout

    def _check_layout(self) -> List[str]:
        findings = []
        if not self.mined_dir.is_dir():
            return [f"{MINED_DIR_NAME}/ was not written"]
        for name in CAMPAIGN_GRAIN_DOCS:
            if not (self.mined_dir / name).is_file():
                findings.append(f"mined/{name} is missing")
        if not (self.mined_dir / MINING_REPORT_NAME).is_file():
            findings.append(f"mined/{MINING_REPORT_NAME} is missing")
        for it_dir in sorted(self.mined_dir.glob("it-*")):
            if it_dir.is_dir() and not (it_dir / "index.md").is_file():
                findings.append(f"mined/{it_dir.name}/index.md is missing")
        return findings

    # ------------------------------------------------------------- flows

    def _flow_files(self) -> List[Path]:
        return sorted(self.mined_dir.glob("it-*/flow-*.md"))

    def _load_flows(self, findings: List[str]) -> List[Dict[str, Any]]:
        flows = []
        for path in self._flow_files():
            rel = path.relative_to(self.mined_dir).as_posix()
            frontmatter = _split_frontmatter(path.read_text())
            if frontmatter is None:
                findings.append(f"mined/{rel}: missing or non-mapping frontmatter block")
                continue
            for key in ("flow", "status", "sources"):
                if key not in frontmatter:
                    findings.append(f"mined/{rel}: frontmatter missing `{key}`")
            if frontmatter.get("flow") != rel[:-3]:
                findings.append(
                    f"mined/{rel}: `flow: {frontmatter.get('flow')}` does not "
                    f"match its path"
                )
            if frontmatter.get("status") not in FLOW_STATUSES:
                findings.append(
                    f"mined/{rel}: status {frontmatter.get('status')!r} is not "
                    f"one of {FLOW_STATUSES}"
                )
            sources = frontmatter.get("sources")
            if "sources" in frontmatter and (
                not isinstance(sources, dict) or not sources
            ):
                findings.append(f"mined/{rel}: `sources` must be a non-empty mapping")
            runs = frontmatter.get("runs", [])
            if runs and not (
                isinstance(runs, list)
                and all(isinstance(r, str) and r.startswith("run_") for r in runs)
            ):
                findings.append(f"mined/{rel}: `runs` must be a list of run_NNNN ids")
            flows.append({"rel": rel, "path": path, "frontmatter": frontmatter})
        return flows

    # ---------------------------------------------------------- coverage

    def _check_coverage(self, flows: List[Dict[str, Any]]) -> List[str]:
        """Every bundle run is claimed by exactly one flow, or named in the
        mining report as skipped — the stable-identity arithmetic."""
        findings = []
        claimed: Dict[str, str] = {}
        for flow in flows:
            for run in flow["frontmatter"].get("runs") or []:
                if run in claimed:
                    findings.append(
                        f"run {run} is claimed by both {claimed[run]} and "
                        f"{flow['rel']}"
                    )
                claimed[run] = flow["rel"]
        report_path = self.mined_dir / MINING_REPORT_NAME
        report_text = report_path.read_text() if report_path.is_file() else ""
        for run in sorted(self.bundle_runs - set(claimed)):
            if run not in report_text:
                findings.append(
                    f"run {run} is neither claimed by a flow nor named in the "
                    f"mining report as skipped"
                )
        return findings

    # -------------------------------------------------------------- refs

    def _check_refs(self, flows: List[Dict[str, Any]]) -> List[str]:
        """`sources` refs and body links into the raw bundle must resolve;
        line anchors and quoted fragments verify mechanically."""
        findings = []
        for flow in flows:
            sources = flow["frontmatter"].get("sources")
            if isinstance(sources, dict):
                for section, ref in sources.items():
                    findings += _ref_findings(
                        str(ref), f"mined/{flow['rel']} sources.{section}",
                        self.bundle_dir, self.inventory,
                    )
            for link in _MD_LINK_PATTERN.findall(flow["path"].read_text()):
                if link.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path_part = link.partition("#")[0]
                if not path_part:
                    continue
                if path_part in self.inventory:
                    # A raw-bundle ref: fragment anchors/quotes verify.
                    findings += _ref_findings(
                        link, f"mined/{flow['rel']} body link",
                        self.bundle_dir, self.inventory,
                    )
                    continue
                # Otherwise it must resolve as a mined-internal link (relative
                # to the flow doc, the mined root, or the bundle root).
                resolvable = (
                    (flow["path"].parent / path_part).exists()
                    or (self.mined_dir / path_part).exists()
                    or (self.bundle_dir / path_part).exists()
                )
                if not resolvable:
                    findings.append(
                        f"mined/{flow['rel']}: link {link!r} resolves to "
                        f"neither the raw inventory nor mined/"
                    )
        return findings

    # ------------------------------------------------------------ indexes

    def _check_indexes(self) -> List[str]:
        """Each it-N/index.md lists exactly the flow files present."""
        findings = []
        for it_dir in sorted(self.mined_dir.glob("it-*")):
            if not it_dir.is_dir():
                continue
            index_path = it_dir / "index.md"
            if not index_path.is_file():
                continue  # already a layout finding
            index_text = index_path.read_text()
            listed = {
                link.partition("#")[0].removeprefix("./")
                for link in _MD_LINK_PATTERN.findall(index_text)
            }
            present = {p.name for p in it_dir.glob("flow-*.md")}
            for name in sorted(present - listed):
                findings.append(f"mined/{it_dir.name}/index.md does not list {name}")
            for name in sorted(l for l in listed - present if l.startswith("flow-")):
                findings.append(
                    f"mined/{it_dir.name}/index.md lists {name} which does not exist"
                )
        return findings

    # --------------------------------------------------- raw immutability

    def _check_raw_immutability(self) -> List[str]:
        """Re-hash every inventory file against the manifest — the raw bundle
        is read-only to the crew, and this is the proof."""
        findings = []
        for rel, expected in self.inventory.items():
            path = self.bundle_dir / rel
            if not path.is_file():
                findings.append(f"raw file {rel} disappeared during mining")
            elif _hash_file(path) != expected:
                findings.append(f"raw file {rel} was modified during mining")
        return findings


class MiningFrame:
    """Stage -> launch -> check -> commit for one trajectory's mined view."""

    def __init__(
        self,
        store: TrajectoryStore,
        mining_config: Dict[str, Any],
        agent_factory=CodingAgentFactory,
    ):
        self.store = store
        self.config = mining_config
        self.agent_factory = agent_factory

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MiningFrame":
        learning = config.get("learning")
        if not isinstance(learning, dict) or "mining" not in learning:
            raise KeyError("config has no `learning.mining` block")
        return cls(TrajectoryStore.from_config(config), learning["mining"])

    # ------------------------------------------------------------- public

    def mine(self, trajectory_id: str, force: bool = False) -> Path:
        """Produce <bundle>/mined/ for one trajectory; returns the mined dir.

        Idempotent: an already-mined trajectory is skipped unless `force`
        (the mined view is a derived layer — regeneration replaces it whole).
        """
        manifest = self.store.manifest(trajectory_id)
        bundle_dir = self.store.resolve(trajectory_id)
        mined_dir = bundle_dir / MINED_DIR_NAME
        if manifest.get("derived", {}).get("mined") and mined_dir.is_dir() and not force:
            return mined_dir
        if mined_dir.exists():
            shutil.rmtree(mined_dir)

        self._stage(bundle_dir, trajectory_id)
        agent = self._build_lead(bundle_dir)
        timeout = self.config["timeout_minutes"] * 60
        prompt = (CREW_DIR / "lead_prompt.md").read_text().replace(
            "{{trajectory_id}}", trajectory_id
        )
        result = agent.generate_code(prompt, timeout_seconds=timeout)
        self._write_report(mined_dir, result.output)

        findings = MinedViewValidator(bundle_dir, manifest).validate()
        repair_rounds = self.config["repair_rounds"]
        rounds_used = 0
        while findings and rounds_used < repair_rounds:
            rounds_used += 1
            repair_prompt = self._repair_prompt(findings)
            result = agent.generate_code(repair_prompt, timeout_seconds=timeout)
            self._write_report(mined_dir, result.output)
            findings = MinedViewValidator(bundle_dir, manifest).validate()
        self._unstage(bundle_dir)
        if findings:
            raise RuntimeError(
                f"mined view for {trajectory_id} failed validation after "
                f"{rounds_used} repair round(s): " + "; ".join(findings)
            )

        self._commit(trajectory_id, bundle_dir, manifest)
        return mined_dir

    # ------------------------------------------------------------ staging

    def _stage(self, bundle_dir: Path, trajectory_id: str) -> None:
        """Place the contract and agent definitions at the checkout root.

        A crashed earlier run leaves our own staging behind; that exact
        leftover (byte-identical contract doc) is removed and re-staged. Any
        OTHER occupant of these names is not ours to delete — refuse loudly.
        """
        format_target = bundle_dir / FORMAT_DOC_NAME
        contract_text = (CREW_DIR / "mined_format.md").read_text()
        if format_target.exists():
            if format_target.read_text() != contract_text:
                raise FileExistsError(
                    f"bundle {trajectory_id} contains a foreign {FORMAT_DOC_NAME}; "
                    "refusing to stage over it"
                )
            format_target.unlink()
        claude_dir = bundle_dir / ".claude"
        if claude_dir.exists():
            leftover_names = sorted(
                p.name for p in (claude_dir / "agents").glob("*.md")
            ) if (claude_dir / "agents").is_dir() else []
            our_names = sorted(p.name for p in (CREW_DIR / "agents").glob("*.md"))
            if leftover_names != our_names:
                raise FileExistsError(
                    f"bundle {trajectory_id} contains a foreign .claude dir; "
                    "refusing to stage over it"
                )
            shutil.rmtree(claude_dir)
        shutil.copy2(CREW_DIR / "mined_format.md", format_target)
        agents_dir = bundle_dir / ".claude" / "agents"
        agents_dir.mkdir(parents=True)
        substitutions = {
            "{{flow_writer_model}}": self.config["flow_writer_model"],
            "{{critic_model}}": self.config["critic_model"],
        }
        for definition in (CREW_DIR / "agents").glob("*.md"):
            text = definition.read_text()
            for placeholder, value in substitutions.items():
                text = text.replace(placeholder, value)
            (agents_dir / definition.name).write_text(text)

    def _unstage(self, bundle_dir: Path) -> None:
        (bundle_dir / FORMAT_DOC_NAME).unlink()
        shutil.rmtree(bundle_dir / ".claude")

    def _build_lead(self, bundle_dir: Path):
        lead = self.config["lead"]
        agent_config = CodingAgentConfig(
            agent_type=lead["cli"],
            model=lead["model"],
            debug_model=lead["model"],
            agent_specific={
                "auth_mode": lead["auth_mode"],
                "effort": lead["effort"],
                "append_system_prompt":
                    (CREW_DIR / "append_system_prompt.md").read_text().strip(),
                "streaming": True,
            },
        )
        agent = self.agent_factory.create(agent_config)
        agent.initialize(str(bundle_dir))
        return agent

    # ----------------------------------------------------------- outputs

    def _write_report(self, mined_dir: Path, report_text: Optional[str]) -> None:
        """The lead's final message IS the mining report; the frame files it."""
        if not report_text or not report_text.strip():
            raise RuntimeError("mining lead produced no final report")
        mined_dir.mkdir(exist_ok=True)
        (mined_dir / MINING_REPORT_NAME).write_text(report_text)

    def _repair_prompt(self, findings: List[str]) -> str:
        numbered = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(findings))
        return (
            "The frame's mechanical verification of mined/ found the following "
            "violations. Fix each one in place (mined/ only — the raw bundle "
            "stays read-only), then finish with the updated mining report as "
            "your final message.\n\n" + numbered
        )

    def _commit(
        self, trajectory_id: str, bundle_dir: Path, manifest: Dict[str, Any]
    ) -> None:
        """Record the derived layer in the manifest (regenerable, never part
        of the raw inventory) and mirror it to the remote when configured."""
        mined_files = [
            p for p in (bundle_dir / MINED_DIR_NAME).rglob("*") if p.is_file()
        ]
        manifest["derived"] = {
            "mined": {
                "generated": datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                "files": len(mined_files),
                "report": f"{MINED_DIR_NAME}/{MINING_REPORT_NAME}",
            }
        }
        with open(bundle_dir / MANIFEST_NAME, "w") as handle:
            yaml.safe_dump(manifest, handle, sort_keys=False)
        if self.store.remote is not None:
            self.store.upload_derived(trajectory_id, MINED_DIR_NAME)
