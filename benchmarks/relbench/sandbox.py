"""Sanitized RelBench cache builder.

Builds a per-task copy of the RelBench cache from which the information a
solution is not allowed to see has been *physically removed*, so the coding
agent's process cannot leak labels even by accident:

- Forecasting / recommendation tasks: the database is truncated at the test
  timestamp before saving (the pristine on-disk db cache contains rows *after*
  the test cutoff, from which future labels could be reconstructed).
- Recommendation tasks: the default relbench masking keeps foreign-key columns,
  and the ground-truth destination list is registered as a foreign key — so it
  survives `get_table("test")`. The sanitized test table keeps only the source
  entity id and the seed timestamp.
- Autocomplete tasks: the database keeps rows after the test cutoff (test
  entities are created after it), but the target column values of rows after
  the test cutoff are blanked. `db.table_dict[...].removed_cols` in the agent
  process therefore contains no test labels.
- Entity tasks: the test task table keeps only the entity id and timestamp.
- No `db.zip` / task zips are copied and the whole tree is made read-only, so
  `download=True` in the agent process fails loudly instead of silently
  restoring pristine (label-bearing) files. Solutions must use
  `download=False`; the sanitized cache is complete so nothing needs downloading.

Train and validation task tables are copied unmodified (labels included):
tuning on validation is part of the RelBench protocol.

Run as a script in a *fresh* interpreter so relbench's in-memory caches cannot
carry pristine data across:

    python -m benchmarks.relbench.sandbox --dataset rel-f1 --task driver-position \
        --dest /path/to/sanitized_cache [--source ~/.cache/relbench]
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from pathlib import Path


def _default_source_cache() -> str:
    return os.environ.get(
        "RELBENCH_PRISTINE_CACHE_DIR",
        os.path.expanduser(os.path.join("~", ".cache", "relbench")),
    )


def _copy_tree_with_hardlinks(src: Path, dst: Path) -> None:
    """Copy a directory, hardlinking file content to avoid duplicating big DBs."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.unlink()
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)


def _rewrite_file(path: Path, write_fn) -> None:
    """Replace a possibly-hardlinked file with fresh content (break the link)."""
    # Keep the .parquet suffix: relbench's Table.save asserts on it.
    tmp = path.with_name(f".tmp_{path.name}")
    write_fn(tmp)
    if path.exists():
        path.unlink()
    tmp.rename(path)


def _make_read_only(root: Path) -> None:
    dir_mode = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    file_mode = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    for item in sorted(root.rglob("*"), reverse=True):
        if item.is_dir():
            os.chmod(item, dir_mode)
        elif item.stat().st_nlink == 1:
            # Hardlinked files share their inode (and mode) with the pristine
            # cache — leave those untouched. Read-only directories already
            # block file creation/replacement, which is what stops pooch from
            # restoring pristine (label-bearing) files on download=True.
            os.chmod(item, file_mode)
    os.chmod(root, dir_mode)


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    os.chmod(root, 0o755)
    for item in root.rglob("*"):
        os.chmod(item, 0o755 if item.is_dir() else 0o644)


def build_sanitized_cache(
    dataset_name: str,
    task_name: str,
    dest_cache_dir: str,
    source_cache_dir: str | None = None,
) -> str:
    """Build the sanitized cache. Returns the destination cache dir.

    Must run with the pristine cache already populated (the handler downloads
    dataset and task tables before calling this).
    """
    # Imports are deferred so the module can be inspected without relbench.
    import pandas as pd

    from relbench.base import AutoCompleteTask, Table
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task, task_registry

    from benchmarks.relbench.task_specs import coerce_boolean_target

    source = Path(source_cache_dir or _default_source_cache())
    dest = Path(dest_cache_dir)

    src_ds_dir = source / dataset_name
    if not (src_ds_dir / "db").exists():
        raise FileNotFoundError(
            f"Pristine cache for {dataset_name} not found at {src_ds_dir}/db. "
            "Download the dataset first (handler does this automatically)."
        )

    if dest.exists():
        _make_writable(dest)
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # The task object gives us the family, key columns and timestamps.
    # NOTE: this loads from the pristine cache (RELBENCH_CACHE_DIR must NOT
    # point at `dest` while building).
    dataset = get_dataset(dataset_name, download=False)
    task = get_task(dataset_name, task_name, download=False)
    is_autocomplete = isinstance(task, AutoCompleteTask)

    dest_ds_dir = dest / dataset_name
    dest_db_dir = dest_ds_dir / "db"
    dest_task_dir = dest_ds_dir / "tasks" / task_name
    dest_task_dir.mkdir(parents=True)

    # ------------------------------------------------------------------
    # 1. Database
    # ------------------------------------------------------------------
    _copy_tree_with_hardlinks(src_ds_dir / "db", dest_db_dir)

    if is_autocomplete:
        # Keep post-test rows (test entities exist only there), but blank the
        # target column for rows after the test cutoff in the entity table.
        cls, args, kwargs = task_registry[dataset_name][task_name]
        entity_table = kwargs["entity_table"]
        target_col = kwargs["target_col"]
        entity_pq = dest_db_dir / f"{entity_table}.parquet"
        # Go through the Table API so pkey/fkey/time-col schema metadata
        # survives the rewrite (raw to_parquet would drop it).
        entity_tbl = Table.load(str(entity_pq))
        time_col = task.time_col
        if time_col is None or time_col not in entity_tbl.df.columns:
            raise RuntimeError(
                f"Autocomplete entity table {entity_table} has no usable time column"
            )
        mask = entity_tbl.df[time_col] > dataset.test_timestamp
        entity_tbl.df.loc[mask, target_col] = None
        _rewrite_file(entity_pq, lambda p, t=entity_tbl: t.save(str(p)))
        print(
            f"[sandbox] blanked {int(mask.sum())} post-test '{target_col}' values "
            f"in {entity_table}"
        )
    else:
        # Forecasting / recommendation: physically truncate every table at the
        # test timestamp. `Table.load` keeps schema metadata; rewrite via the
        # relbench Table API so pkey/fkey/time metadata is preserved.
        db = dataset.get_db(upto_test_timestamp=True)
        for table_name, table in db.table_dict.items():
            table_pq = dest_db_dir / f"{table_name}.parquet"
            if not table_pq.exists():
                raise FileNotFoundError(f"Expected db table file missing: {table_pq}")
            pristine = pd.read_parquet(table_pq)
            if len(pristine) != len(table.df):
                _rewrite_file(table_pq, lambda p, t=table: t.save(str(p)))
                print(
                    f"[sandbox] truncated {table_name}: "
                    f"{len(pristine)} -> {len(table.df)} rows"
                )

    # ------------------------------------------------------------------
    # 2. Task tables: train/val verbatim, test masked to (time, entity) only.
    # ------------------------------------------------------------------
    src_task_dir = src_ds_dir / "tasks" / task_name
    for split in ("train", "val"):
        src_pq = src_task_dir / f"{split}.parquet"
        if not src_pq.exists():
            raise FileNotFoundError(
                f"Pristine task table missing: {src_pq}. "
                "Download the task first (handler does this automatically)."
            )
        table = task.get_table(split, mask_input_cols=False)
        if not hasattr(task, "src_entity_col"):  # EntityTask / AutoCompleteTask
            # A text boolean target ('t'/'f') must reach the candidate as 0/1:
            # relbench's own metrics reject string labels, so the in-loop grader
            # would fail on every run (observed live on rel-trial/studies-has_dmc).
            # RecommendationTask has no scalar target_col — nothing to coerce.
            table = coerce_boolean_target(table, task.target_col)
        table.save(str(dest_task_dir / f"{split}.parquet"))

    test_table = task.get_table("test", mask_input_cols=False)
    if hasattr(task, "src_entity_col"):  # RecommendationTask
        keep_cols = [test_table.time_col, task.src_entity_col]
        fkeys = {task.src_entity_col: task.src_entity_table}
    else:  # EntityTask / AutoCompleteTask
        keep_cols = [test_table.time_col, task.entity_col]
        fkeys = {task.entity_col: task.entity_table}
    masked = Table(
        # Row order must be preserved: predictions align positionally.
        df=test_table.df[keep_cols].reset_index(drop=True),
        fkey_col_to_pkey_table=fkeys,
        pkey_col=None,
        time_col=test_table.time_col,
    )
    masked.save(str(dest_task_dir / "test.parquet"))
    print(f"[sandbox] masked test table: cols={keep_cols} rows={len(masked)}")

    # ------------------------------------------------------------------
    # 3. Lock it down. No zips are copied: `download=True` now fails loudly
    #    (read-only tree) instead of silently restoring pristine files.
    # ------------------------------------------------------------------
    _make_read_only(dest)
    print(f"[sandbox] sanitized cache ready at {dest}")
    return str(dest)


def build_rolling_caches(
    dataset_name: str,
    task_name: str,
    dest_root: str,
    source_cache_dir: str | None = None,
) -> str:
    """Per-tick snapshot cascade for regime-sensitive windowed tasks
    (EVALUATION_PROTOCOL.md §"Rolling harness").

    One sanitized mini-cache per eval tick (val + test). Leak-proof invariant:
    a snapshot truncated at tick t cannot contain t's label window (t, t+Δ].
    Each snapshot's task dir holds `train.parquet` = every window CLOSED by the
    tick, relabeled by the task's OWN make_table SQL over the truncated db
    (never from pristine labels), `test.parquet` = that tick's rows with input
    cols only, and `indices.npy` = the rows' positions in the official split
    table (for canonical-order assembly by the grader).
    """
    import numpy as np
    import pandas as pd

    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    from relbench.base import Table

    source = Path(source_cache_dir or _default_source_cache())
    if not (source / dataset_name / "db").exists():
        raise FileNotFoundError(f"Pristine cache for {dataset_name} not found at {source}")
    dest = Path(dest_root)
    if dest.exists():
        _make_writable(dest)
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    dataset = get_dataset(dataset_name, download=False)
    task = get_task(dataset_name, task_name, download=False)
    if getattr(task, "num_eval_timestamps", 1) <= 1:
        raise ValueError(f"{dataset_name}/{task_name} is not a rolling task")
    td = task.timedelta
    time_col, entity_col = task.time_col, task.entity_col
    db_full = dataset.get_db(upto_test_timestamp=False)

    split_tables = {s: task.get_table(s, mask_input_cols=False) for s in ("train", "val", "test")}
    all_starts = pd.Series(
        pd.concat([t.df[time_col] for t in split_tables.values()]).unique()
    ).sort_values(ignore_index=True)

    n_snaps = 0
    for split in ("val", "test"):
        df = split_tables[split].df.reset_index(drop=True)
        for i, tick in enumerate(sorted(df[time_col].unique())):
            tick = pd.Timestamp(tick)
            snap = dest / f"{split}_{i:03d}_{tick.date()}"
            db_dir = snap / dataset_name / "db"
            task_dir = snap / dataset_name / "tasks" / task_name
            db_dir.mkdir(parents=True)
            task_dir.mkdir(parents=True)

            db_snap = db_full.upto(tick)
            for name, table in db_snap.table_dict.items():
                table.save(str(db_dir / f"{name}.parquet"))

            closed = all_starts[all_starts + td <= tick]
            train_out = task.make_table(db_snap, pd.Series(closed.to_numpy()))
            if len(train_out.df) and (train_out.df[time_col] + td > tick).any():
                raise RuntimeError(f"open window leaked into {snap.name} train table")
            train_out.save(str(task_dir / "train.parquet"))

            rows = df[df[time_col] == tick]
            masked = Table(
                df=rows[[time_col, entity_col]].reset_index(drop=True),
                fkey_col_to_pkey_table={entity_col: task.entity_table},
                pkey_col=None,
                time_col=time_col,
            )
            masked.save(str(task_dir / "test.parquet"))
            # Empty schema-correct val table: relbench's get_table("val") on a
            # cache miss REGENERATES and SAVES — which the read-only tree turns
            # into a deep pyarrow PermissionError for any candidate (or helper)
            # that touches val. Shipping an empty table makes every val path
            # load 0 rows instead of crashing; per-tick evaluation ignores it.
            empty_val = Table(
                df=split_tables["val"].df.iloc[0:0].reset_index(drop=True),
                fkey_col_to_pkey_table={entity_col: task.entity_table},
                pkey_col=None,
                time_col=time_col,
            )
            empty_val.save(str(task_dir / "val.parquet"))
            np.save(task_dir / "indices.npy", rows.index.to_numpy())
            n_snaps += 1

    _make_read_only(dest)
    print(f"[sandbox] rolling cascade ready at {dest}: {n_snaps} snapshots "
          f"(val={split_tables['val'].df[time_col].nunique()}, "
          f"test={split_tables['test'].df[time_col].nunique()})")
    return str(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--source", default=None)
    parser.add_argument("--rolling", action="store_true",
                        help="build the per-tick snapshot cascade instead of the flat cache")
    args = parser.parse_args()

    if os.path.abspath(os.environ.get("RELBENCH_CACHE_DIR", "")) == os.path.abspath(args.dest):
        print(
            "RELBENCH_CACHE_DIR must not point at --dest while building",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.rolling:
        build_rolling_caches(args.dataset, args.task, args.dest, args.source)
    else:
        build_sanitized_cache(args.dataset, args.task, args.dest, args.source)


if __name__ == "__main__":
    main()
