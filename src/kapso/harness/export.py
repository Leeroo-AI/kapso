"""`kapso harness export`: a harness's skills, delegates and servers
written as the files an ordinary session of one host discovers
(docs/plans/redesign/wp7-plugin.md, change 7). The union over the
harness's modules by `Manifest.listed()`, each attachment once in
first-occurrence order. Hooks stay behind (campaign-bound: a Stop gate or
a prepared-directory deny would block an ordinary session), as do the
module pages and the routing per module, which only a campaign
reproduces; the standing rules travel as a page the person includes from
their own instructions file, which the export never writes. Every
destination is checked before anything is written; one that exists
refuses the whole export by name."""

import json
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from kapso.core.front_matter import read_front_matter
from kapso.execution.coding_agents.adapters.codex_agent import toml_string
from kapso.harness.hooks import expand_plugin_root
from kapso.harness.load import HarnessError
from kapso.harness.manifest import MODULES, Manifest, read_manifest
from kapso.harness.staging import CLAUDE_HOST, CODEX_HOST, HARNESS_ROOT_VARIABLE, is_reference

HOSTS = (CLAUDE_HOST, CODEX_HOST)
MCP_FILE = "mcp.json"
GUARDRAILS_PAGE = Path("instructions") / "guardrails.md"
CLAUDE_SERVERS_FILE = Path(".mcp.json")
_SKILLS_DIR = {CLAUDE_HOST: Path(".claude") / "skills", CODEX_HOST: Path(".agents") / "skills"}
_AGENTS_DIR = {CLAUDE_HOST: Path(".claude") / "agents", CODEX_HOST: Path(".codex") / "agents"}
_INSTRUCTIONS_FILE = {CLAUDE_HOST: "CLAUDE.md", CODEX_HOST: "AGENTS.md"}
HOOKS_LEFT_BEHIND = (
    "campaign-bound: a Stop gate or a prepared-directory deny would block an ordinary session"
)
PAGES_LEFT_BEHIND = "only a campaign routes a module's page to its sessions"


class HarnessExportError(HarnessError):
    """An export refused: a destination exists, or the host is unknown."""


@dataclass
class HarnessExport:
    """What an export wrote (paths relative to the directory, in the order
    written), what the person does next (the include line for their own
    instructions file; on Codex the config lines and the server lines),
    and what stayed behind."""

    harness: str
    version: str
    host: str
    into: Path
    written: List[Path] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    left_behind: List[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"exported {self.harness} {self.version} for {self.host} into {self.into}"]
        lines += [f"  {path.as_posix()}{'/' if (self.into / path).is_dir() else ''}" for path in self.written]
        for item in self.left_behind:
            lines.append(f"left behind: {item}")
        lines.append("next:")
        lines += [f"  {step}" for step in self.next_steps]
        return "\n".join(lines)


def export_harness(root: Path, host: str, into: Path) -> HarnessExport:
    """Write the harness at `root` for `host` into `into`; the report says
    what was written and what to do next."""
    if host not in HOSTS:
        raise HarnessExportError(f"unknown host {host!r}; the hosts are {', '.join(HOSTS)}")
    root = root.resolve()
    into = into.resolve()
    manifest = read_manifest(root)
    export = HarnessExport(manifest.name, manifest.version, host, into)
    writes: List[Tuple[Path, Callable[[], None]]] = []
    for name in manifest.listed("skills"):
        writes.append(_copy(root / "skills" / name, _SKILLS_DIR[host] / name, into))
    for name in manifest.listed("agents"):
        writes.append(_agent(root / "agents" / f"{name}.md", name, host, into, export))
    referring = [name for name in manifest.listed("skills") if _mentions_harness_root(root / "skills" / name)]
    if referring:
        export.next_steps.append(
            f"export KAPSO_HARNESS_ROOT={root} in the shell these skills run from ({', '.join(referring)} refer to it; "
            "a campaign sets it, a session of your own does not)"
        )
    servers = _servers(root, manifest)
    if servers and host == CLAUDE_HOST:
        file = into / CLAUDE_SERVERS_FILE
        writes.append((CLAUDE_SERVERS_FILE, lambda: file.write_text(json.dumps({"mcpServers": servers}, indent=2) + "\n", encoding="utf-8")))
    for name, entry in servers.items():
        if host == CODEX_HOST:
            export.next_steps += _codex_server_lines(name, entry)
    page = Path(f"kapso-harness-{manifest.name}.md")
    writes.append(_copy(root / GUARDRAILS_PAGE, page, into))
    instructions = _INSTRUCTIONS_FILE[host]
    include = f"@{page.as_posix()}" if host == CLAUDE_HOST else f"Read {page.as_posix()} before any work in this domain."
    export.next_steps.insert(0, f"add to {instructions} (never written by the export): {include}")
    hooks = manifest.listed("hooks")
    if hooks:
        export.left_behind.append(f"{len(hooks)} hook(s) ({', '.join(hooks)}): {HOOKS_LEFT_BEHIND}")
    pages = [module for module in MODULES if (root / "instructions" / f"{module}.md").is_file()]
    if pages:
        export.left_behind.append(f"the module page(s) ({', '.join(pages)}) and the routing per module: {PAGES_LEFT_BEHIND}")
    in_the_way = [path for path, _ in writes if _in_the_way(into, path)]
    if in_the_way:
        raise HarnessExportError(
            "nothing written: " + ", ".join(path.as_posix() for path in in_the_way) + f" in the way under {into} "
            "(something at the path, a dangling link included, or a file where a directory on the way to it must be); "
            "move them, or export into an empty directory with --to and merge by hand"
        )
    for path, write in writes:
        (into / path).parent.mkdir(parents=True, exist_ok=True)
        write()
        export.written.append(path)
    return export


def _in_the_way(into: Path, relative: Path) -> bool:
    """Whether writing `relative` under `into` would meet something: the
    destination itself in any form, or a file where a directory on the way
    to it must be."""
    target = into / relative
    if target.exists() or target.is_symlink():
        return True
    for parent in target.parents:
        if parent == into:
            return False
        if parent.exists() and not parent.is_dir():
            return True
    return False


def _mentions_harness_root(directory: Path) -> bool:
    return any("KAPSO_HARNESS_ROOT" in path.read_text(encoding="utf-8", errors="replace") for path in directory.rglob("*") if path.is_file())


def _toml_table(mapping: Dict[str, str]) -> str:
    """An inline TOML table, `{ "X-Key" = "value" }` (JSON's colon is not TOML's)."""
    return "{ " + ", ".join(f"{json.dumps(key)} = {json.dumps(value)}" for key, value in mapping.items()) + " }"


def _copy(source: Path, relative: Path, into: Path) -> Tuple[Path, Callable[[], None]]:
    target = into / relative
    if source.is_dir():
        return relative, lambda: shutil.copytree(source, target)
    return relative, lambda: shutil.copy2(source, target)


def _agent(source: Path, name: str, host: str, into: Path, export: HarnessExport) -> Tuple[Path, Callable[[], None]]:
    """A delegate: on Claude Code the file itself (its front matter is the
    agent file's form); on Codex a TOML config file whose developer
    instructions are the body, and the `[agents.<name>]` lines the user's
    config needs, printed."""
    if host == CLAUDE_HOST:
        return _copy(source, _AGENTS_DIR[host] / f"{name}.md", into)
    data, body = read_front_matter(source)
    relative = _AGENTS_DIR[host] / f"{name}.toml"
    export.next_steps.append(
        "add to ~/.codex/config.toml:\n"
        f"    [agents.{name}]\n"
        f"    description = {toml_string(str(data['description']))}\n"
        f"    config_file = {toml_string(str(into / relative))}"
    )
    return relative, lambda: (into / relative).write_text(
        f"developer_instructions = {toml_string(body.strip() + chr(10))}\n", encoding="utf-8",
    )


def _servers(root: Path, manifest: Manifest) -> Dict[str, Dict[str, Any]]:
    """The listed servers' entries, `${PLUGIN_ROOT}` expanded to the
    harness's directory and the harness root in each stdio server's
    environment, as the staging mounts them."""
    listed = manifest.listed("mcp_servers")
    if not listed:
        return {}
    defined = json.loads((root / MCP_FILE).read_text(encoding="utf-8"))["mcpServers"]
    servers: Dict[str, Dict[str, Any]] = {}
    for name in listed:
        entry = expand_plugin_root(defined[name], root)
        if "command" in entry:
            env = dict(entry.get("env") or {})
            env[HARNESS_ROOT_VARIABLE] = str(root)
            entry["env"] = env
        servers[name] = entry
    return servers


def _codex_server_lines(name: str, entry: Dict[str, Any]) -> List[str]:
    """The `codex mcp add` line for one server; a value referenced by
    variable name (`${NAME}`) is forwarded by name through the config
    keys the Codex adapter uses, never baked in by the shell."""
    if "url" in entry:
        lines = [shlex.join(["codex", "mcp", "add", name, "--url", entry["url"]])]
        headers = entry.get("headers") or {}
        literal = {k: v for k, v in headers.items() if not is_reference(v)}
        referenced = {k: v[2:-1] for k, v in headers.items() if is_reference(v)}
        if literal:
            lines.append(f"then under [mcp_servers.{name}] in ~/.codex/config.toml: http_headers = {_toml_table(literal)}")
        if referenced:
            lines.append(f"then under [mcp_servers.{name}] in ~/.codex/config.toml: env_http_headers = {_toml_table(referenced)}")
        return lines
    env = entry["env"]
    literal = [f"{k}={v}" for k, v in env.items() if not is_reference(v)]
    referenced = [v[2:-1] for v in env.values() if is_reference(v)]
    argv = ["codex", "mcp", "add", name]
    for pair in literal:
        argv += ["--env", pair]
    argv += ["--", entry["command"], *entry.get("args", [])]
    lines = [shlex.join(argv)]
    if referenced:
        lines.append(f"then under [mcp_servers.{name}] in ~/.codex/config.toml: env_vars = {json.dumps(referenced)}")
    return lines
