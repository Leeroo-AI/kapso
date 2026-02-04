"""
CLI for wiki sync management.

Provides commands for initial sync, status checking, and conflict resolution.
"""

import argparse
import json
import sys
from pathlib import Path

from .config import SyncConfig
from .state_manager import StateManager
from .sync_engine import SyncEngine
from .transforms import path_to_title, title_to_path


def cmd_initial_sync(args) -> int:
    """Perform initial synchronization."""
    config = SyncConfig(
        wiki_url=args.wiki_url,
        wiki_dir=Path(args.wiki_dir),
        sync_state=Path(args.state),
        mw_user=args.user,
        mw_pass=args.password,
    )

    state = StateManager(config.sync_state)
    engine = SyncEngine(config, state)

    print(f"Initial sync mode: {args.mode}")
    print(f"Wiki URL: {config.wiki_url}")
    print(f"Wiki dir: {config.wiki_dir}")
    print()

    stats = engine.initial_sync(mode=args.mode)

    print()
    print("=" * 50)
    print(f"Synced from local: {stats['synced_local']}")
    print(f"Synced from wiki:  {stats['synced_wiki']}")
    print(f"Conflicts:         {stats['conflicts']}")
    print(f"Errors:            {stats['errors']}")
    print("=" * 50)

    return 0 if stats["errors"] == 0 else 1


def cmd_status(args) -> int:
    """Show sync status."""
    config = SyncConfig(
        wiki_dir=Path(args.wiki_dir),
        sync_state=Path(args.state),
    )

    state = StateManager(config.sync_state)
    sync_state = state.load()

    print("Wiki Sync Status")
    print("=" * 50)
    print(f"State file: {config.sync_state}")
    print(f"Version: {sync_state.version}")
    print(f"Last RC timestamp: {sync_state.last_rc_timestamp or 'Never'}")
    print(f"Last RC ID: {sync_state.last_rc_id or 'None'}")
    print(f"Tracked files: {len(sync_state.files)}")
    print(f"Pending changes: {len(sync_state.pending)}")
    print(f"Active conflicts: {len(sync_state.conflicts)}")
    print("=" * 50)

    if args.verbose:
        print("\nTracked files:")
        for rel_path, file_state in sorted(sync_state.files.items()):
            print(f"  {rel_path}")
            print(f"    Wiki: {file_state.wiki_title} (rev {file_state.wiki_revid})")
            print(f"    Synced: {file_state.synced_at}")

    return 0


def cmd_conflicts(args) -> int:
    """List current conflicts."""
    config = SyncConfig(
        wiki_dir=Path(args.wiki_dir),
        sync_state=Path(args.state),
    )

    state = StateManager(config.sync_state)
    conflicts = state.get_conflicts()

    if not conflicts:
        print("No active conflicts.")
        return 0

    print(f"Active conflicts: {len(conflicts)}")
    print("=" * 50)

    for conflict in conflicts:
        print(f"\nFile: {conflict.file_path}")
        print(f"  Wiki title: {conflict.wiki_title}")
        print(f"  Wiki revid: {conflict.wiki_revid}")
        print(f"  Detected: {conflict.detected_at}")
        if conflict.local_conflict_file:
            print(f"  Local version: {conflict.local_conflict_file}")
        if conflict.wiki_conflict_file:
            print(f"  Wiki version: {conflict.wiki_conflict_file}")

    print()
    print("To resolve, use: sync resolve <file_path> --keep local|wiki")

    return 0


def cmd_resolve(args) -> int:
    """Resolve a conflict."""
    config = SyncConfig(
        wiki_url=args.wiki_url,
        wiki_dir=Path(args.wiki_dir),
        sync_state=Path(args.state),
        mw_user=args.user,
        mw_pass=args.password,
    )

    state = StateManager(config.sync_state)
    engine = SyncEngine(config, state)

    # Find the conflict
    conflicts = state.get_conflicts()
    conflict = None
    for c in conflicts:
        if c.file_path == args.file_path:
            conflict = c
            break

    if not conflict:
        print(f"No conflict found for: {args.file_path}")
        return 1

    file_path = config.wiki_dir / conflict.file_path

    if args.keep == "local":
        # Read local conflict file and push to wiki
        if conflict.local_conflict_file and Path(conflict.local_conflict_file).exists():
            content = Path(conflict.local_conflict_file).read_text(encoding="utf-8")
        elif file_path.exists():
            content = file_path.read_text(encoding="utf-8")
        else:
            print(f"Local file not found: {file_path}")
            return 1

        # Write to main file and sync to wiki
        file_path.write_text(content, encoding="utf-8")
        if engine.sync_local_to_wiki(file_path):
            print(f"Resolved conflict (kept local): {args.file_path}")
            state.remove_conflict(args.file_path)
            _cleanup_conflict_files(conflict)
            return 0
        else:
            print("Failed to sync to wiki")
            return 1

    elif args.keep == "wiki":
        # Read wiki conflict file and write to local
        if conflict.wiki_conflict_file and Path(conflict.wiki_conflict_file).exists():
            content = Path(conflict.wiki_conflict_file).read_text(encoding="utf-8")
        else:
            # Fetch from wiki
            info = engine.get_wiki_page_info(conflict.wiki_title)
            if not info:
                print(f"Wiki page not found: {conflict.wiki_title}")
                return 1
            content = info["content"]

        # Write to local file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

        # Update state
        from .state_manager import compute_content_hash
        info = engine.get_wiki_page_info(conflict.wiki_title)
        if info:
            state.update_file_state(
                rel_path=conflict.file_path,
                content_hash=compute_content_hash(content),
                wiki_title=conflict.wiki_title,
                wiki_revid=info["revid"],
            )

        print(f"Resolved conflict (kept wiki): {args.file_path}")
        state.remove_conflict(args.file_path)
        _cleanup_conflict_files(conflict)
        return 0

    return 1


def _cleanup_conflict_files(conflict) -> None:
    """Remove conflict files after resolution."""
    for path_str in [conflict.local_conflict_file, conflict.wiki_conflict_file]:
        if path_str:
            path = Path(path_str)
            if path.exists():
                path.unlink()

    # Also remove the .CONFLICT.json file
    if conflict.local_conflict_file:
        info_file = Path(conflict.local_conflict_file).with_suffix("")
        info_file = info_file.parent / info_file.name.replace(".LOCAL", ".CONFLICT")
        info_file = Path(str(info_file) + ".json")
        if info_file.exists():
            info_file.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Wiki sync CLI",
        prog="python -m sync.cli",
    )
    parser.add_argument(
        "--wiki-url",
        default="http://localhost:8090",
        help="MediaWiki base URL",
    )
    parser.add_argument(
        "--wiki-dir",
        default="/wikis",
        help="Local wiki files directory",
    )
    parser.add_argument(
        "--state",
        default="/state/sync.json",
        help="State file path",
    )
    parser.add_argument(
        "--user",
        default="agent",
        help="MediaWiki username",
    )
    parser.add_argument(
        "--password",
        default="",
        help="MediaWiki password",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # initial-sync command
    p_init = subparsers.add_parser(
        "initial-sync",
        help="Perform initial synchronization",
    )
    p_init.add_argument(
        "--mode",
        choices=["detect_only", "wiki_wins", "local_wins"],
        default="detect_only",
        help="Sync mode: detect_only (build state), wiki_wins, local_wins",
    )
    p_init.set_defaults(func=cmd_initial_sync)

    # status command
    p_status = subparsers.add_parser(
        "status",
        help="Show sync status",
    )
    p_status.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed file list",
    )
    p_status.set_defaults(func=cmd_status)

    # conflicts command
    p_conflicts = subparsers.add_parser(
        "conflicts",
        help="List current conflicts",
    )
    p_conflicts.set_defaults(func=cmd_conflicts)

    # resolve command
    p_resolve = subparsers.add_parser(
        "resolve",
        help="Resolve a conflict",
    )
    p_resolve.add_argument(
        "file_path",
        help="Relative path of the conflicting file",
    )
    p_resolve.add_argument(
        "--keep",
        choices=["local", "wiki"],
        required=True,
        help="Which version to keep",
    )
    p_resolve.set_defaults(func=cmd_resolve)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
