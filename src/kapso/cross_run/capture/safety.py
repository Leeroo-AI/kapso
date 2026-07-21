"""Shared path-policy and immutable regular-file reads for capture boundaries."""

from __future__ import annotations

import os
import re
import stat
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Iterable


def path_matches_denied_pattern(
    relative_path: str,
    denied_patterns: Iterable[str],
) -> bool:
    components = tuple(part.casefold() for part in PurePosixPath(relative_path).parts)
    for raw_pattern in denied_patterns:
        pattern = raw_pattern.casefold()
        token = pattern.removeprefix("token:")
        for component in components:
            if not pattern.startswith("token:") and component == pattern:
                return True
            if pattern == ".git" and (
                component == ".gitconfig" or component.startswith(".git-")
            ):
                return True
            if pattern.startswith("token:"):
                component_tokens = tuple(
                    value for value in re.split(r"[._-]+", component) if value
                )
                if token in component_tokens:
                    return True
    return False


def read_restricted_regular_file(
    root: Path,
    relative_path: str,
    error_type: type[Exception],
    *,
    require_restricted: bool = True,
    maximum_bytes: int | None = None,
) -> bytes:
    if maximum_bytes is not None and maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive")
    normalized = _require_safe_relative_path(relative_path, error_type)
    _reject_existing_symlink_components(root, normalized, error_type)
    with ExitStack() as descriptors:
        parent_descriptor = _open_pinned_parent(
            root,
            normalized,
            error_type,
            descriptors,
        )
        descriptor = os.open(
            normalized.parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        handle = descriptors.enter_context(os.fdopen(descriptor, "rb"))
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise error_type("artifact is not an independent regular file")
        if require_restricted and metadata.st_mode & 0o077:
            raise error_type("artifact is not access restricted")
        if maximum_bytes is not None and metadata.st_size > maximum_bytes:
            raise error_type("artifact exceeds configured size limit")
        payload = handle.read(-1 if maximum_bytes is None else maximum_bytes + 1)
        if maximum_bytes is not None and len(payload) > maximum_bytes:
            raise error_type("artifact exceeds configured size limit")
        return payload


def remove_restricted_directory(
    root: Path,
    relative_path: str,
    expected_identity: tuple[int, int],
    error_type: type[Exception],
) -> None:
    """Remove one relative directory through pinned, no-follow parent descriptors."""

    normalized = _require_safe_relative_path(relative_path, error_type)
    _reject_existing_symlink_components(root, normalized, error_type)
    with ExitStack() as descriptors:
        parent_descriptor = _open_pinned_parent(
            root,
            normalized,
            error_type,
            descriptors,
        )
        target_descriptor = os.open(
            normalized.parts[-1],
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        descriptors.callback(os.close, target_descriptor)
        metadata = os.fstat(target_descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != expected_identity
        ):
            raise error_type("removal target identity changed")
        _remove_pinned_directory_contents(target_descriptor, error_type)
        current = os.stat(
            normalized.parts[-1],
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != expected_identity
        ):
            raise error_type("removal target was replaced")
        os.rmdir(normalized.parts[-1], dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)


def restricted_directory_identity(
    root: Path,
    relative_path: str,
    error_type: type[Exception],
) -> tuple[int, int]:
    normalized = _require_safe_relative_path(relative_path, error_type)
    _reject_existing_symlink_components(root, normalized, error_type)
    with ExitStack() as descriptors:
        parent_descriptor = _open_pinned_parent(
            root,
            normalized,
            error_type,
            descriptors,
        )
        target_descriptor = os.open(
            normalized.parts[-1],
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        descriptors.callback(os.close, target_descriptor)
        metadata = os.fstat(target_descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise error_type("directory identity target is not a real directory")
        current = os.stat(
            normalized.parts[-1],
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise error_type("directory identity target was replaced")
        return metadata.st_dev, metadata.st_ino


def _remove_pinned_directory_contents(
    descriptor: int,
    error_type: type[Exception],
) -> None:
    with os.scandir(descriptor) as iterator:
        entries = tuple(
            (entry.name, entry.stat(follow_symlinks=False)) for entry in iterator
        )
    for name, expected in entries:
        current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
            raise error_type("removal entry was replaced")
        if stat.S_ISDIR(expected.st_mode):
            with ExitStack() as child_descriptors:
                child_descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                child_descriptors.callback(os.close, child_descriptor)
                opened = os.fstat(child_descriptor)
                if (opened.st_dev, opened.st_ino) != (
                    expected.st_dev,
                    expected.st_ino,
                ):
                    raise error_type("removal directory was replaced")
                _remove_pinned_directory_contents(child_descriptor, error_type)
            current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (
                expected.st_dev,
                expected.st_ino,
            ):
                raise error_type("removal directory was replaced")
            os.rmdir(name, dir_fd=descriptor)
        elif stat.S_ISREG(expected.st_mode) or stat.S_ISLNK(expected.st_mode):
            os.unlink(name, dir_fd=descriptor)
        else:
            raise error_type("removal target contains a special file")
    os.fsync(descriptor)


def _require_safe_relative_path(
    relative_path: str,
    error_type: type[Exception],
) -> PurePosixPath:
    normalized = PurePosixPath(relative_path)
    if (
        not normalized.parts
        or normalized == PurePosixPath(".")
        or normalized.is_absolute()
        or ".." in normalized.parts
        or normalized.as_posix() != relative_path
    ):
        raise error_type("artifact path is unsafe")
    return normalized


def _reject_existing_symlink_components(
    root: Path,
    relative_path: PurePosixPath,
    error_type: type[Exception],
) -> None:
    if not root.is_dir() or root.is_symlink():
        raise error_type("artifact root is not a real directory")
    candidate = root
    for name in relative_path.parts:
        candidate /= name
        if candidate.is_symlink():
            raise error_type("artifact path contains a symlink")


def _open_pinned_parent(
    root: Path,
    relative_path: PurePosixPath,
    error_type: type[Exception],
    descriptors: ExitStack,
) -> int:
    absolute_root = Path(os.path.abspath(root))
    descriptor = os.open(
        absolute_root.anchor,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    descriptors.callback(os.close, descriptor)
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        raise error_type("artifact root is not a real directory")
    for name in absolute_root.parts[1:]:
        child_descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        descriptors.callback(os.close, child_descriptor)
        if not stat.S_ISDIR(os.fstat(child_descriptor).st_mode):
            raise error_type("artifact root parent is not a real directory")
        descriptor = child_descriptor
    for name in relative_path.parts[:-1]:
        child_descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        descriptors.callback(os.close, child_descriptor)
        if not stat.S_ISDIR(os.fstat(child_descriptor).st_mode):
            raise error_type("artifact path parent is not a real directory")
        descriptor = child_descriptor
    return descriptor
