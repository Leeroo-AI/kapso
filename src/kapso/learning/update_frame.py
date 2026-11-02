# Update-run frame — stage / launch / validate / commit around the update crew.
#
# Design: learn-from-trajectories-update-crew.md (the run shape, in-flight
# schema, frame contract) + main doc §4.3. The frame does everything
# mechanical and nothing judgmental: it seeds the worksheet (batch rows from
# hindcast reports, docket rows from the pre-run bank), stages the role
# prompts and the run-role.sh worker launcher (D4: codex workers driven by
# the Claude lead; the critic is the lead's native subagent), launches ONE
# lead session, then validates the transaction — surface, diff invariants,
# evidence admission, coverage arithmetic, score bounds — with one repair
# bounce before failing loud. A green transaction commits as one reviewed
# commit, tagged lr_<id>, pushed to the bank home; the learner report is
# assembled frame-side.

import re
import shlex
import shutil
import subprocess
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from kapso.execution.coding_agents.adapters.codex_agent import DEFAULT_SANDBOX
from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.execution.coding_agents.session_record import EXIT_ERROR, EXIT_SUCCESS, append_run, run_entry
from kapso.execution.coding_agents.transcripts import codex_threads, digest_codex
from kapso.harness.load import Harness
from kapso.harness.staging import CODEX_HOST, StagedHarness, host_for, stage_harness
from kapso.learning.bank import DECOY_REGISTRY_NAME, Bank
from kapso.learning.bank_invariants import (
    BankTransactionValidator,
    body_contract_gaps,
    evidence_admission_findings,
)
from kapso.learning.graders.hindcast import HindcastReport
from kapso.learning.refs import extract_refs
from kapso.learning.retriever import compile_probe_queue, exposure_ladder
from kapso.learning.trajectory_store import MANIFEST_NAME, TrajectoryStore

CREW_DIR = Path(__file__).parent / "crews" / "update"
# The two sessions a run stages the harness's learning module into, as
# recorded under `<run_dir>/.kapso/sessions/`: the Claude lead, and the
# `codex exec` workers run-role.sh launches, which read the module's
# guidance from a file beside the launcher.
LEAD_SESSION = "lead"
WORKERS_SESSION = "workers"
WORKERS_GUIDANCE_FILE = "harness-guidance.md"
# The campaign's pull log inside its archived bundle: the exposure the
# evidence check credits usage claims against.
PULL_LOG_REL = ".kapso/serving/serving-pull.jsonl"
# NOTE is the acknowledgment verdict for serving-feedback rows (journal-only,
# no bank edit) — a small vocabulary extension over the update-crew doc,
# discovered at implementation: feedback rows need a verdict that is neither a
# card decision nor a PASS (whose rebuttal contract does not fit them).
JOURNAL_VERDICTS = (
    "ATTACH", "SPAWN", "SIGHTING", "PASS", "NOTE",
    "MERGE", "GENERALIZE", "RESOLVE", "EXPIRE", "CODIFY",
)
_ROW_PATTERN = re.compile(r"^- \*\*(obs-\d+|dk-\d+)\*\*", re.MULTILINE)
_JOURNAL_PATTERN = re.compile(
    r"^- \*\*(obs-\d+|dk-\d+) → ([A-Z-]+)\*\*", re.MULTILINE
)
_WORD_PATTERN = re.compile(r"[a-z]{3,}")
# A bank's commits are the engine's, on a machine that may have no git identity.
BANK_COMMITTER = ("-c", "user.name=Kapso", "-c", "user.email=kapso@leeroo.com")


def init_bank(home_path: str) -> None:
    """Create the bank home (a bare repo) with the founding skeleton. It is
    built aside and renamed into place last, so a failure leaves no home."""
    home = Path(home_path).expanduser()
    if home.exists():
        raise FileExistsError(f"bank home {home_path} already exists")
    home.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=home.parent, prefix=f".init-{home.name}-") as scratch:
        building = Path(scratch) / "bank.git"
        seed = Path(scratch) / "seed"
        subprocess.run(["git", "init", "--bare", "-b", "main", str(building)], check=True)
        subprocess.run(["git", "clone", str(building), str(seed)], check=True)
        (seed / "insights").mkdir()
        (seed / "procedures").mkdir()
        (seed / "index.md").write_text(
            "# kapso knowledge bank\n\n- [insights](insights/index.md) — declarative memory\n"
            "- [procedures](procedures/index.md) — procedural memory\n"
        )
        (seed / "log.md").write_text("# Learner-run journal\n")
        (seed / "sightings.md").write_text("# Sightings — pre-card observations\n")
        (seed / "insights" / "index.md").write_text("# Insights\n")
        (seed / "procedures" / "index.md").write_text("# Procedures\n")
        subprocess.run(["git", "-C", str(seed), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(seed), *BANK_COMMITTER, "commit", "-m", "bank founding skeleton"],
            check=True,
        )
        subprocess.run(["git", "-C", str(seed), "push", "origin", "main"], check=True)
        building.rename(home)


def _tokens(text: str) -> set:
    return set(_WORD_PATTERN.findall(text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class UpdateFrame:
    """One learning run: batch + docket -> one bank transaction + report."""

    def __init__(
        self,
        store: TrajectoryStore,
        config: Dict[str, Any],
        agent_factory=CodingAgentFactory,
        bank_home: Optional[Path] = None,
    ):
        """``bank_home`` is the bank the run clones, edits and pushes back
        to: the facade's resolved home (`Kapso(bank=)` may have moved it
        off the config); the config's `learning.bank.local_path` when the
        caller passes none."""
        self.store = store
        self.learning_config = config["learning"]
        self.crew_config = self.learning_config["update_crew"]
        self.bank_config = self.learning_config["bank"]
        self.bank_home = (
            Path(bank_home).expanduser() if bank_home is not None
            else Path(self.bank_config["local_path"]).expanduser()
        )
        self.score_band = self.learning_config["graders"]["score_band"]
        self.codify_config = self.learning_config["codify"]
        self.agent_factory = agent_factory

    # ------------------------------------------------------------------ run

    def run_update(
        self,
        batch: List[Dict[str, str]],
        run_root: str,
        learner_version: str,
        harness: Optional[Harness] = None,
    ) -> Path:
        """batch: [{trajectory, hindcast_report}] — empty list is a
        docket-only consolidation run. Returns the run dir (report inside).
        ``harness`` is the one the lesson runs with: its learning module is
        staged into the run directory for the lead's session and for the
        workers' (the pages in their prompts; the skills, servers, delegates
        and hooks in each host's own format); none without one."""
        lr_id = "lr_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_dir = Path(run_root).expanduser() / lr_id
        (run_dir / "work").mkdir(parents=True)

        subprocess.run(
            ["git", "clone", str(self.bank_home), str(run_dir / "bank")],
            check=True, capture_output=True,
        )
        before_dir = run_dir / "bank-before"
        shutil.copytree(run_dir / "bank", before_dir)
        before = Bank(str(run_dir / "bank"))

        self._stage_codify_runs(run_dir, before)
        self._seed_worksheet(run_dir, batch, before)
        lead = self.crew_config["lead"]
        staged_lead = stage_harness(
            harness, "learning", session_dir=run_dir, campaign_dir=run_dir, session_id=LEAD_SESSION,
            host=host_for(lead["cli"]) if harness is not None else lead["cli"],
        )
        staged_workers = stage_harness(
            harness, "learning", session_dir=run_dir, campaign_dir=run_dir, session_id=WORKERS_SESSION,
            host=CODEX_HOST,
        )
        self._stage_crew(run_dir, lr_id, batch, learner_version, staged_workers)

        agent = self._build_lead(run_dir, staged_lead)
        timeout = self.crew_config["timeout_minutes"] * 60
        prompt = (CREW_DIR / "lead_prompt.md").read_text()
        if staged_lead.guidance:
            prompt += "\n\n" + staged_lead.guidance
        # The workers' invocations are recorded however the lead's session
        # ends: the launcher runs under the lead, so no adapter saw them.
        try:
            agent.generate_code(prompt, timeout_seconds=timeout)

            findings = self._validate(run_dir, before_dir, batch)
            repair_rounds = self.crew_config["repair_rounds"]
            rounds_used = 0
            while findings and rounds_used < repair_rounds:
                rounds_used += 1
                numbered = "\n".join(f"{i+1}. {f}" for i, f in enumerate(findings))
                agent.generate_code(
                    prompt + "\n\nThe frame's validation found the following "
                    "violations. Fix each in place (bank/ and work/ only), then "
                    "stop:\n" + numbered,
                    timeout_seconds=timeout,
                )
                findings = self._validate(run_dir, before_dir, batch)
            agent.cleanup()
        finally:
            self._record_workers(staged_workers, self.crew_config["worker"]["model"])
        if findings:
            raise RuntimeError(
                f"update run {lr_id} failed validation after {rounds_used} "
                f"repair round(s): " + "; ".join(findings)
            )

        # A lesson that admits nothing moves nothing: no card, sighting or
        # quarantine changed, so there is no commit, no tag and no push —
        # the report says so, and the bank stays at the head it was.
        if self._bank_changed(run_dir):
            self._rebuild_derived(run_dir, lr_id)
            self._commit(run_dir, lr_id, batch)
        self._assemble_report(run_dir, lr_id, batch, learner_version, before_dir)
        return run_dir

    def _record_workers(self, workers: StagedHarness, model: str) -> None:
        """The workers' invocations, one run entry per `codex exec` thread
        the launcher appended to the crew's stream, recorded once the
        lead's session is over (the launcher runs under the lead, so no
        adapter saw them end). A thread the launcher never closed, or one
        that completed no turn, ended in error."""
        stream = Path(workers.agent_specific["stream_artifact_path"])
        if not stream.exists():
            return
        with open(stream, encoding="utf-8", errors="replace") as handle:
            threads = codex_threads(handle)
        for thread in threads:
            digest = digest_codex(thread["lines"])
            if thread["code"] is None:
                error: Optional[str] = "the launcher recorded no exit: the lead's session ended first"
            elif not thread["lines"]:
                error = f"codex exec exited with code {thread['code']} before starting a thread"
            elif thread["code"] != 0:
                error = f"codex exec exited with code {thread['code']}"
            elif not digest["turns"]:
                error = "codex exec completed no turn"
            else:
                error = None
            append_run(workers.record_path, run_entry(
                digest, model=model, cli_session_id=digest["cli_session_id"],
                exit=EXIT_SUCCESS if error is None else EXIT_ERROR, error=error,
                files_changed=[], started=None, ended=None, duration_seconds=None,
            ))

    def _bank_changed(self, run_dir: Path) -> bool:
        """Whether the crew changed the bank's content: the cards, the
        retired cards, the sightings, the quarantine registry — not the
        derived index or the log, which the frame rebuilds."""
        status = subprocess.run(
            ["git", "-C", str(run_dir / "bank"), "status", "--porcelain", "--",
             "insights", "procedures", "retired", "sightings.md", DECOY_REGISTRY_NAME],
            check=True, capture_output=True, text=True,
        )
        return bool(status.stdout.strip())

    # ------------------------------------------------------------- staging

    def _stage_codify_runs(self, run_dir: Path, bank: Bank) -> None:
        """CD§2: a green codify verdict is folded by the NEXT transaction.
        Stage each still-text card's newest green run — verdict plus the
        workspace artifacts — into work/codify-runs/<card>/ so the crew can
        commit the representation flip with the verdict in-transaction."""
        codify_root = Path(self.crew_config["run_root"]).expanduser() / "codify"
        if not codify_root.is_dir():
            return
        for run in sorted(codify_root.iterdir()):  # sorted: newest last wins
            verdict_path = run / "verdict.yaml"
            if not run.is_dir() or not verdict_path.is_file():
                continue
            card_name = run.name.rsplit("-", 1)[0]
            card = bank.cards.get(card_name)
            if card is None or card.representation == "code":
                continue
            verdict = yaml.safe_load(verdict_path.read_text())
            if not isinstance(verdict, dict) or verdict.get("status") != "green":
                continue
            target = run_dir / "work" / "codify-runs" / card_name
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)
            shutil.copy2(verdict_path, target / "verdict.yaml")
            for artifact in ("code", "replay"):
                source = run / "workspace" / artifact
                if source.is_dir():
                    shutil.copytree(source, target / artifact)

    def _seed_worksheet(
        self, run_dir: Path, batch: List[Dict[str, str]], bank: Bank
    ) -> None:
        """Batch rows from the hindcast reports (lift / card-candidate /
        serving-feedback), docket rows from the pre-run bank (tension /
        dup-merge / generalize nominations — similarity nominates, mechanism
        decides). Every row gets exactly one journal verdict; the frame
        counts."""
        rows: List[str] = []
        counter = 0

        def add(prefix: str, kind: str, text: str) -> None:
            nonlocal counter
            counter += 1
            flat = " ".join(text.split())
            rows.append(f"- **{prefix}-{counter:02d}** [{kind}] — {flat}")

        for item in batch:
            report = HindcastReport.parse(Path(item["hindcast_report"]).read_text())
            trajectory = item["trajectory"]
            for entry in report.sections["claims"]:
                if entry["marker"] in ("AGREED", "CONTRADICTED"):
                    target = entry["card_refs"][0][1] if entry["card_refs"] else "?"
                    add("obs", f"seed: lift → {target}",
                        f"({trajectory}) {entry['text']}")
            for entry in report.sections["extraction"]:
                if entry["marker"] in ("MISS-UNCARDED", "MISS-NOVEL"):
                    add("obs", "seed: card-candidate",
                        f"({trajectory}) {entry['text']}")
            for entry in report.sections["serving"]:
                # Serving outcomes are card evidence, not commentary: a used
                # card's payoff (or its absence) and an ignored card's
                # uptake failure both reach the writer as rows, so the
                # card's usefulness claim gets settled like any other claim.
                if entry["marker"] in ("UPTAKE-FAIL", "SERVED-USED"):
                    add("obs", "seed: serving-feedback",
                        f"({trajectory}) {entry['text']}")

        seen_pairs = set()
        for name, card in sorted(bank.cards.items()):
            for other in card.frontmatter.get("contradicts") or []:
                pair = tuple(sorted((name, str(other))))
                if pair not in seen_pairs and str(other) in bank.cards:
                    seen_pairs.add(pair)
                    add("dk", "tension",
                        f"cards {pair[0]} and {pair[1]} are both active with a "
                        f"standing contradicts edge")
        # Expiry rows — lapsed clocks for the sweeper. (a) Sightings: age in
        # batches = bank log.md entries newer than the sighting's date; at
        # the configured age the sweeper resolves recurred-or-drop. (b) Code
        # freshness (CD§4): representation:code cards whose last_replayed
        # exceeds the max age (or was never stamped) re-earn their replay.
        expiry_batches = self.crew_config["sightings_expiry_batches"]
        log_dates = [
            line.split()[1][3:11]
            for line in bank.log_text().splitlines()
            if line.startswith("- lr_")
        ]
        for line in bank.sightings_text().splitlines():
            if not line.startswith("- ") or "·" not in line:
                continue
            sighting_date = line[2:].split("·")[0].strip().replace("-", "")
            age = sum(1 for stamp in log_dates if stamp > sighting_date)
            if age >= expiry_batches:
                add("dk", "expiry",
                    f"sighting aged {age} batches without recurrence: "
                    f"{line[2:].strip()} — resolve (recurred -> spawn, else drop)")
        max_age_days = self.codify_config["replay_max_age_days"]
        today = datetime.now(timezone.utc).date()
        for name in sorted(bank.cards):
            card = bank.cards[name]
            if card.representation != "code":
                continue
            last_replayed = card.frontmatter.get("last_replayed")
            if last_replayed is None:
                add("dk", "expiry",
                    f"code card {name} carries no last_replayed stamp — "
                    f"replay freshness unproven")
                continue
            age_days = (today - date.fromisoformat(str(last_replayed))).days
            if age_days > max_age_days:
                add("dk", "expiry",
                    f"code card {name} last replayed {age_days} days ago "
                    f"(max {max_age_days}) — freshness re-run due")

        # Codify nominations (CD§1, layer 1): pure ledger arithmetic —
        # executed-verdict entries only, closure through founding references,
        # recurrence as distinct source campaigns. The specialist's
        # compatibility gate is layer 2; the frame never reads code.
        min_recurrence = self.codify_config["min_recurrence"]
        for name in sorted(bank.cards):
            card = bank.cards[name]
            if card.type != "procedure" or card.representation != "text":
                continue
            if card.state != "active" or card.frontmatter.get("contradicts"):
                continue
            if (run_dir / "work" / "codify-runs" / name).is_dir():
                continue  # green run staged — the flip row below owns it
            if bank.codify_blocked_by_failed_attempt(card):
                continue
            recurrence = bank.codify_recurrence(card)
            if recurrence >= min_recurrence:
                add("dk", "codify",
                    f"procedure {name} has {recurrence} executed source "
                    f"campaigns in its closure — nominate for codification "
                    f"(compatibility is the specialist's judgment)")
        staged_flips = run_dir / "work" / "codify-runs"
        if staged_flips.is_dir():
            for flip_dir in sorted(staged_flips.iterdir()):
                add("dk", "codify",
                    f"procedure {flip_dir.name} holds a GREEN codify run in "
                    f"this transaction (work/codify-runs/{flip_dir.name}) — "
                    f"commit the representation flip: move its code/ and "
                    f"replay/ into the card directory (entrypoint inside "
                    f"code/, gates+notes inside replay/), set "
                    f"representation: code, entrypoint: code/<file>, and "
                    f"last_replayed to the run date")

        # Reader-contract migration (user decision 2026-08-21: iterate on a
        # small prioritized slice, not the whole bank at once): seed rewrite
        # rows for live non-conforming card bodies, highest reliability
        # first, capped per run. The invariant enforces the contract on any
        # card a transaction touches; these rows drive the backlog down.
        rewrite_cap = self.crew_config["rewrite_rows_per_run"]
        floors = self.crew_config["body_floors"]
        decoys = bank.decoy_names
        nonconforming = [
            (card.score or 0.0, name)
            for name, card in bank.cards.items()
            if name not in decoys
            and body_contract_gaps(card.body, floors)
        ]
        for _, name in sorted(nonconforming, reverse=True)[:rewrite_cap]:
            gaps = body_contract_gaps(bank.cards[name].body, floors)
            add("dk", "rewrite",
                f"card {name} misses the card template "
                f"({'; '.join(gaps[:3])}{'; …' if len(gaps) > 3 else ''}) — "
                f"rewrite the body to the engineer-facing template in the "
                f"card-writer role doc (# title / **Rule:** / Is this your "
                f"situation? / What to do / Why believe this / "
                f"**Confidence:**), claim meaning unchanged, one version "
                f"bump with one log entry")

        nominate_threshold = self.crew_config["dup_nominate_jaccard"]
        names = sorted(bank.cards)
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                card_a, card_b = bank.cards[name_a], bank.cards[name_b]
                if card_a.type != card_b.type:
                    continue
                similarity = _jaccard(
                    _tokens(card_a.hero + " " + card_a.body),
                    _tokens(card_b.hero + " " + card_b.body),
                )
                if similarity >= nominate_threshold:
                    if card_a.scope != "domain" and card_b.scope != "domain" \
                            and card_a.scope != card_b.scope:
                        add("dk", "generalize",
                            f"cards {name_a} and {name_b} may state one "
                            f"mechanism across sibling scopes "
                            f"(lexical similarity {similarity:.2f})")
                    else:
                        add("dk", "dup-merge",
                            f"cards {name_a} and {name_b} nominated as twins "
                            f"(lexical similarity {similarity:.2f}; similarity "
                            f"nominates, mechanism decides)")

        (run_dir / "work" / "observations.md").write_text(
            "# Routing worksheet — frame-seeded; the lead completes it\n\n"
            + "\n".join(rows) + ("\n" if rows else "")
        )
        (run_dir / "work" / "journal.md").write_text(
            "# Routing journal — one verdict per worksheet row\n\n"
        )

    def _stage_crew(
        self, run_dir: Path, lr_id: str, batch: List[Dict[str, str]],
        learner_version: str, workers: StagedHarness,
    ) -> None:
        """The role prompts, the critic, the inputs, and the worker launcher:
        a worker reads its role, the harness's guidance when there is one,
        then the assignment, and runs with the staged servers, delegates and
        hook trust (`StagedHarness.codex_exec_args`) and variables."""
        roles_dir = run_dir / "role-prompts"
        roles_dir.mkdir()
        for role_file in (CREW_DIR / "roles").glob("*.md"):
            shutil.copy2(role_file, roles_dir / role_file.name)
        agents_dir = run_dir / ".claude" / "agents"
        agents_dir.mkdir(parents=True)
        critic = (CREW_DIR / "agents" / "critic.md").read_text().replace(
            "{{critic_model}}", self.crew_config["critic"]["model"]
        )
        (agents_dir / "critic.md").write_text(critic)

        worker = self.crew_config["worker"]
        guidance_file = ""
        if workers.guidance:
            (run_dir / WORKERS_GUIDANCE_FILE).write_text(workers.guidance)
            guidance_file = f" {WORKERS_GUIDANCE_FILE}"
        exports = "".join(
            f"export {name}={shlex.quote(value)}\n" for name, value in workers.env.items()
        )
        harness_args = workers.codex_exec_args()
        extra = (" \\\n  " + " ".join(shlex.quote(arg) for arg in harness_args)) if harness_args else ""
        stream = shlex.quote(workers.agent_specific["stream_artifact_path"])
        # The worker runs codex exec directly, outside the adapter: its
        # sandbox is the role's when the config names one, else the
        # adapter's own default, so the two never drift apart.
        sandbox = worker.get("sandbox", DEFAULT_SANDBOX)
        launcher = f"""#!/bin/bash
# Generated by the update frame — worker-role invocations (config-threaded).
# Every invocation's event stream is appended to the workers' stream, one
# thread each, closed by its exit code; the frame records them after the
# lead's session.
set -euo pipefail
ROLE="$1"; ASSIGNMENT="$2"; OUT="$3"
STREAM={stream}
mkdir -p "$(dirname "$STREAM")"
status=0
{exports}cat "role-prompts/$ROLE.md"{guidance_file} "$ASSIGNMENT" | codex exec --json \\
  --sandbox {sandbox} --skip-git-repo-check --color never \\
  --output-last-message "$OUT" -m "{worker['model']}" \\
  -c 'model_reasoning_effort="{worker['effort']}"'{extra} >> "$STREAM" || status=$?
printf '{{"type":"kapso.exit","role":"%s","code":%s}}\\n' "$ROLE" "$status" >> "$STREAM"
# The lead reads the worker's final message, not its event stream.
cat "$OUT" 2>/dev/null
exit "$status"
"""
        launcher_path = run_dir / "run-role.sh"
        launcher_path.write_text(launcher)
        launcher_path.chmod(0o755)

        inputs = {
            "lr_id": lr_id,
            "learner_version": learner_version,
            "batch": [
                {
                    "trajectory": item["trajectory"],
                    "mined_view": str(self.store.local / item["trajectory"] / "mined"),
                    "hindcast_report": str(item["hindcast_report"]),
                }
                for item in batch
            ],
            "previous_report": self._previous_report_path(run_dir.parent),
        }
        with open(run_dir / "inputs.yaml", "w") as handle:
            yaml.safe_dump(inputs, handle, sort_keys=False)

    def _previous_report_path(self, run_root: Path) -> Optional[str]:
        """Latest prior report under the run root THIS run was invoked with —
        a development run's batches chain within their own `updates/` root,
        never through the global production run dir (the current run's own
        dir has no report.md yet, so the glob naturally excludes it)."""
        if not run_root.is_dir():
            return None
        reports = sorted(run_root.glob("lr_*/report.md"))
        return str(reports[-1]) if reports else None

    def _build_lead(self, run_dir: Path, staged: StagedHarness):
        lead = self.crew_config["lead"]
        agent = self.agent_factory.create(CodingAgentConfig(
            agent_type=lead["cli"],
            model=lead["model"],
            debug_model=lead["model"],
            agent_specific=staged.merged_agent_specific({
                "auth_mode": lead["auth_mode"],
                "effort": lead["effort"],
                "append_system_prompt":
                    (CREW_DIR / "append_system_prompt.md").read_text().strip(),
                "streaming": True,
            }),
        ))
        agent.initialize(str(run_dir))
        return agent

    # ---------------------------------------------------------- validation

    def _validate(
        self, run_dir: Path, before_dir: Path, batch: List[Dict[str, str]]
    ) -> List[str]:
        findings: List[str] = []
        before = Bank(str(before_dir))
        after = Bank(str(run_dir / "bank"))

        findings += BankTransactionValidator(
            before, after,
            body_floors=self.crew_config["body_floors"],
        ).validate()
        findings += self._check_coverage(run_dir)
        findings += self._check_evidence(before, after, batch)
        findings += self._check_score_bounds(after)
        findings += self._check_codify_flips(run_dir, before, after)
        findings += self._check_reassessment(before, after)
        for name in ("headline.md", "closing.md"):
            if not (run_dir / "work" / name).is_file():
                findings.append(f"work/{name} was not written")
        return findings

    def _check_reassessment(self, before: Bank, after: Bank) -> List[str]:
        """Step D is not optional (B6 live finding): a card that takes a new
        OUTCOME verdict (confirm/weaken/refute) changes the ledger its
        scores rest on — the transaction must show the reassessment as a
        claim-layer event (version bump; the rationale citing the ledger),
        even when the score lands unchanged."""
        findings: List[str] = []
        for name, card in after.cards.items():
            previous = before.cards.get(name)
            if previous is None:
                continue  # new cards are born assessed (conformance owns shape)
            new_entries = card.evidence[len(previous.evidence):]
            outcome_entries = [
                entry for entry in new_entries
                if entry.get("verdict") in ("confirm", "weaken", "refute")
            ]
            if not outcome_entries:
                continue
            before_version = (previous.frontmatter.get("provenance") or {}).get("version")
            after_version = (card.frontmatter.get("provenance") or {}).get("version")
            if before_version == after_version:
                findings.append(
                    f"{name}: took {len(outcome_entries)} outcome verdict(s) "
                    f"but reliability was never reassessed (version "
                    f"{after_version} unchanged) — settlements must move or "
                    f"re-affirm the scores, never freeze them"
                )
        return findings

    def _check_codify_flips(
        self, run_dir: Path, before: Bank, after: Bank
    ) -> List[str]:
        """CD§2's transaction rule: a representation flip (text -> code)
        commits only inside a transaction holding a GREEN codify run, and
        the flipped card carries its artifacts — code/, replay/, entrypoint.
        No green run, no flip."""
        findings: List[str] = []
        for name, card in after.cards.items():
            previous = before.cards.get(name)
            flipped = card.representation == "code" and (
                previous is None or previous.representation != "code"
            )
            if not flipped:
                continue
            verdict_path = run_dir / "work" / "codify-runs" / name / "verdict.yaml"
            if not verdict_path.is_file():
                findings.append(
                    f"{name}: representation flip without a codify-run "
                    f"verdict in this transaction — no green run, no flip"
                )
            else:
                verdict = yaml.safe_load(verdict_path.read_text())
                if not isinstance(verdict, dict) or verdict.get("status") != "green":
                    findings.append(
                        f"{name}: representation flip on a non-green codify "
                        f"run ({(verdict or {}).get('status')!r})"
                    )
            card_dir = run_dir / "bank" / "procedures" / name
            for required in ("code", "replay"):
                if not (card_dir / required).is_dir():
                    findings.append(
                        f"{name}: representation code without {required}/"
                    )
            # Layout law (user decision 2026-08-21): everything runnable
            # lives in code/ (entrypoint included), everything reproduction
            # lives in replay/ (eval + gates + notes).
            entrypoint = str(card.frontmatter.get("entrypoint") or "").strip()
            if not entrypoint:
                findings.append(f"{name}: representation code without an entrypoint")
            elif not entrypoint.startswith("code/"):
                findings.append(
                    f"{name}: entrypoint {entrypoint!r} must live inside "
                    f"code/ (e.g. code/main.py)"
                )
            elif not (card_dir / entrypoint).is_file():
                findings.append(
                    f"{name}: entrypoint {entrypoint!r} does not exist in "
                    f"the card directory"
                )
            if not (card_dir / "replay" / "gates.yaml").is_file():
                findings.append(
                    f"{name}: representation code without replay/gates.yaml "
                    f"— the reproduction contract ships with the card"
                )
        return findings

    def _check_coverage(self, run_dir: Path) -> List[str]:
        findings = []
        worksheet = (run_dir / "work" / "observations.md").read_text()
        journal_path = run_dir / "work" / "journal.md"
        journal = journal_path.read_text() if journal_path.is_file() else ""
        rows = _ROW_PATTERN.findall(worksheet)
        verdicts: Dict[str, List[str]] = {}
        for row_id, verdict in _JOURNAL_PATTERN.findall(journal):
            verdicts.setdefault(row_id, []).append(verdict)
        for row_id in rows:
            row_verdicts = verdicts.get(row_id, [])
            if not row_verdicts:
                findings.append(f"worksheet row {row_id} has no journal verdict")
            elif len(row_verdicts) > 1:
                findings.append(f"worksheet row {row_id} has {len(row_verdicts)} verdicts")
            for verdict in row_verdicts:
                if verdict not in JOURNAL_VERDICTS:
                    findings.append(
                        f"journal verdict {verdict!r} on {row_id} is not in "
                        f"{JOURNAL_VERDICTS}"
                    )
        for row_id in verdicts:
            if row_id not in rows:
                findings.append(f"journal names {row_id} which is not a worksheet row")
        for line in journal.splitlines():
            match = _JOURNAL_PATTERN.match(line)
            if match and match.group(2) == "PASS":
                block = self._entry_block(journal, line)
                if "rebuttal:" not in block:
                    findings.append(
                        f"PASS on {match.group(1)} carries no `rebuttal:` — "
                        f"passing needs the surviving rebuttal"
                    )
        return findings

    def _entry_block(self, journal: str, start_line: str) -> str:
        lines = journal.splitlines()
        index = lines.index(start_line)
        block = [start_line]
        for line in lines[index + 1:]:
            if line.startswith("- **"):
                break
            block.append(line)
        return "\n".join(block)

    def _check_evidence(
        self, before: Bank, after: Bank, batch: List[Dict[str, str]]
    ) -> List[str]:
        # Exposure is the campaign's own pull log, archived in its bundle
        # (the hindcast's serving record is availability, not exposure);
        # a campaign that ran unserved has none, so a usage claim on it
        # is a finding.
        exposures: Dict[str, Dict[str, Dict[str, str]]] = {}
        for item in batch:
            # Through the store's resolver when the bundle is here or a
            # remote store can bring it: a served campaign is never read
            # as unserved for want of a local copy. A trap batch the
            # gauntlet names without a bundle has no exposure.
            trajectory = item["trajectory"]
            if not (self.store.local / trajectory / MANIFEST_NAME).is_file() and self.store.remote is None:
                continue
            if PULL_LOG_REL in self.store.manifest(trajectory)["inventory"]["sha256"]:
                exposures[trajectory] = exposure_ladder(self.store.resolve(trajectory, PULL_LOG_REL))
        findings: List[str] = []
        for name, card in after.cards.items():
            previous = before.cards.get(name)
            new_entries = card.evidence[len(previous.evidence) if previous else 0:]
            if new_entries:
                findings += evidence_admission_findings(
                    card, new_entries, self.store, exposures
                )
        return findings

    def _check_score_bounds(self, after: Bank) -> List[str]:
        """The assessor's validity is frame-bounded against the ledger's
        verdict arithmetic (v1 bound: confirms over outcome verdicts ± the
        graders band). Decoys are exempt: crews may not touch them, so a
        corridor miss there would deadlock every transaction (seen live at
        the clean-room founding)."""
        findings = []
        decoys = after.decoy_names
        for name, card in after.cards.items():
            if name in decoys:
                continue
            confirms = sum(1 for e in card.evidence if e.get("verdict") == "confirm")
            outcomes = sum(
                1 for e in card.evidence
                if e.get("verdict") in ("confirm", "weaken", "refute")
            )
            validity = card.reliability.get("validity")
            if outcomes and isinstance(validity, (int, float)):
                center = confirms / outcomes
                if abs(validity - center) > self.score_band + 1e-9:
                    findings.append(
                        f"{name}: validity {validity} is outside the ledger "
                        f"corridor ({center:.2f} ± {self.score_band})"
                    )
        return findings

    # ------------------------------------------------------------- closing

    def _rebuild_derived(self, run_dir: Path, lr_id: str) -> None:
        """The frame, not agents, recompiles the derived index (edges from
        typed fields + body refs) and appends the log.md journal line."""
        bank = Bank(str(run_dir / "bank"))
        edges = []
        for name, card in sorted(bank.cards.items()):
            for other in card.frontmatter.get("contradicts") or []:
                edges.append({"from": name, "to": str(other), "type": "contradicts"})
            supersedes = card.frontmatter.get("supersedes")
            for parent in supersedes if isinstance(supersedes, list) else []:
                edges.append({"from": name, "to": str(parent), "type": "supersedes"})
            for ref in extract_refs(card.body):
                if ref.startswith(("insights/", "procedures/", "/insights/", "/procedures/")):
                    edges.append({"from": name, "to": ref.lstrip("/"),
                                  "type": "reference"})
        index_dir = run_dir / "bank" / "index"
        index_dir.mkdir(exist_ok=True)
        with open(index_dir / "edges.yaml", "w") as handle:
            yaml.safe_dump(edges, handle, sort_keys=False)
        # The probe queue (§5.1): VoI-ranked open probes, recompiled each
        # batch — the push rider serves from it under the probe budget.
        (index_dir / "probe-queue.md").write_text(compile_probe_queue(bank))

        headline_path = run_dir / "work" / "headline.md"
        headline = headline_path.read_text().strip().splitlines()
        first_line = headline[0] if headline else "(no headline)"
        log_path = run_dir / "bank" / "log.md"
        log_path.write_text(log_path.read_text() + f"- {lr_id} — {first_line}\n")

    def _commit(self, run_dir: Path, lr_id: str, batch: List[Dict[str, str]]) -> None:
        bank_dir = run_dir / "bank"
        headline = (run_dir / "work" / "headline.md").read_text().strip().splitlines()
        message = (headline[0] if headline else lr_id) + "\n\nbatch:\n" + "\n".join(
            f"- {item['trajectory']}" for item in batch
        )
        subprocess.run(["git", "-C", str(bank_dir), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(bank_dir), *BANK_COMMITTER, "commit", "-m", message],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(bank_dir), "tag", lr_id], check=True
        )
        subprocess.run(
            ["git", "-C", str(bank_dir), "push", "origin", "main", lr_id],
            check=True, capture_output=True,
        )

    def _assemble_report(
        self, run_dir: Path, lr_id: str, batch: List[Dict[str, str]],
        learner_version: str, before_dir: Path,
    ) -> None:
        after = Bank(str(run_dir / "bank"))
        states: Dict[str, int] = {}
        for card in after.cards.values():
            states[str(card.state)] = states.get(str(card.state), 0) + 1
        for card in after.retired_cards.values():
            states[str(card.state)] = states.get(str(card.state), 0) + 1
        tensions = sum(
            len(card.frontmatter.get("contradicts") or [])
            for card in after.cards.values()
        ) // 2
        sightings_entries = sum(
            1 for line in after.sightings_text().splitlines()
            if line.startswith("- ")
        )
        head = subprocess.run(
            ["git", "-C", str(run_dir / "bank"), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        before_head = subprocess.run(
            ["git", "-C", str(before_dir), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        frontmatter = {
            "run": lr_id,
            "learner_version": learner_version,
            "batch": [item["trajectory"] for item in batch],
            "bank": {"before": before_head, "after": head},
            "health": {
                "cards": states,
                "contradictions_open": tensions,
                "sightings_entries": sightings_entries,
            },
        }
        work = run_dir / "work"

        def section(name: str, path: Path) -> str:
            body = path.read_text().strip() if path.is_file() else "(absent)"
            return f"## {name}\n\n{body}"

        report = (
            "---\n" + yaml.safe_dump(frontmatter, sort_keys=False) + "---\n\n"
            + section("Headline", work / "headline.md") + "\n\n"
            + section("Routing journal", work / "journal.md") + "\n\n"
            + section("Rejections and critic findings", work / "critic-findings.md")
            + "\n\n"
            + section("Closing assessment", work / "closing.md") + "\n"
        )
        (run_dir / "report.md").write_text(report)
