"""Bounded canonical source-archive inspection shared by trusted consumers."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tarfile
import tempfile
from io import DEFAULT_BUFFER_SIZE
from pathlib import Path, PurePosixPath
from typing import TypeVar

import zstandard

from kapso.cross_run.canonical import source_tree_digest
from kapso.cross_run.contracts import SourceFileDescriptor

_TAR_BLOCK_SIZE = 512
_TAR_ZERO_BLOCK = b"\0" * _TAR_BLOCK_SIZE
_CANONICAL_TAR_TYPES = {b"\0", b"0", b"5"}
_ZSTD_MAX_BLOCK_SIZE = 128 * 1024
_ZSTD_MAX_FRAME_HEADER_SIZE = 18

_ArchiveError = TypeVar("_ArchiveError", bound=RuntimeError)


class SourceArchiveError(RuntimeError):
    """A source archive is unsafe, noncanonical, or outside configured bounds."""


class SourceArchiveExtractor:
    """Extract canonical tar assets without links, ambiguity, or hidden state."""

    def __init__(
        self,
        *,
        zstd_window_size_bytes: int,
        error_type: type[_ArchiveError] = SourceArchiveError,
    ) -> None:
        if zstd_window_size_bytes <= 0:
            raise ValueError("zstd_window_size_bytes must be positive")
        self.zstd_window_size_bytes = zstd_window_size_bytes
        self.error_type = error_type

    def extract(
        self,
        *,
        archive: Path,
        destination: Path,
        extracted_paths: dict[str, str],
        maximum_bytes: int,
        maximum_entries: int,
    ) -> tuple[int, int]:
        if maximum_bytes <= 0 or maximum_entries <= 0:
            raise self.error_type("source archive extraction bound is exhausted")
        if archive.is_symlink() or not archive.is_file():
            raise self.error_type("source archive must be a regular file")
        if not archive.name.endswith((".tar", ".tar.zst")):
            raise self.error_type("source archive format is unsupported")
        tar_path = archive
        if archive.name.endswith(".tar.zst"):
            expected_content_size = self._validate_single_zstd_frame(
                archive,
                maximum_bytes,
            )
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".decompressed-",
                suffix=".tar",
                dir=destination.parent,
            )
            os.close(descriptor)
            tar_path = Path(temporary_name)
            with archive.open("rb") as compressed, tar_path.open("wb") as decompressed:
                decompressor = zstandard.ZstdDecompressor(
                    max_window_size=self.zstd_window_size_bytes
                )
                with decompressor.stream_reader(compressed) as reader:
                    decompressed_bytes = 0
                    while True:
                        chunk = reader.read(DEFAULT_BUFFER_SIZE)
                        if not chunk:
                            break
                        decompressed_bytes += len(chunk)
                        if decompressed_bytes > maximum_bytes:
                            raise self.error_type(
                                "decompressed archive exceeds configured size limit"
                            )
                        decompressed.write(chunk)
            if (
                expected_content_size is not None
                and decompressed_bytes != expected_content_size
            ):
                raise self.error_type(
                    "decompressed archive size differs from its zstd frame"
                )
        result = self._extract_tar(
            tar_path,
            destination,
            extracted_paths,
            maximum_bytes,
            maximum_entries,
        )
        if tar_path != archive:
            tar_path.unlink()
        return result

    def source_tree_files(self, source_tree: Path) -> tuple[SourceFileDescriptor, ...]:
        source_paths = tuple(sorted(source_tree.rglob("*")))
        relative_entries = {
            path: path.relative_to(source_tree).as_posix() for path in source_paths
        }
        for path, relative_path in relative_entries.items():
            self.safe_archive_path(relative_path, path.is_dir())
        if any(
            path.is_symlink() or (not path.is_file() and not path.is_dir())
            for path in source_paths
        ):
            raise self.error_type("source tree contains an invalid entry")
        if any(
            path.stat(follow_symlinks=False).st_nlink != 1
            for path in source_paths
            if path.is_file()
        ):
            raise self.error_type("source tree files must be independent")
        source_files = tuple(
            SourceFileDescriptor(
                relative_path=relative_entries[path],
                digest=self.file_digest(path),
                mode="100755" if path.stat().st_mode & 0o111 else "100644",
                size=path.stat().st_size,
            )
            for path in source_paths
            if path.is_file()
        )
        if not source_files:
            raise self.error_type("source archive tree is empty")
        observed_directories = {
            relative_entries[path] for path in source_paths if path.is_dir()
        }
        implied_directories = {
            parent.as_posix()
            for source_file in source_files
            for parent in PurePosixPath(source_file.relative_path).parents
            if parent != PurePosixPath(".")
        }
        if observed_directories != implied_directories:
            raise self.error_type("source tree contains undeclared empty directories")
        return source_files

    @staticmethod
    def file_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file_handle:
            while True:
                chunk = file_handle.read(DEFAULT_BUFFER_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    def safe_archive_path(self, name: str, is_directory: bool) -> str | None:
        normalized_name = name[:-1] if is_directory and name.endswith("/") else name
        if not normalized_name or normalized_name == ".":
            return None
        path = PurePosixPath(normalized_name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in normalized_name
            or path.as_posix() != normalized_name
            or ".git" in path.parts
            or normalized_name == ".gitmodules"
        ):
            raise self.error_type("archive contains an unsafe path")
        return normalized_name

    @staticmethod
    def tree_hash(source_files: tuple[SourceFileDescriptor, ...]) -> str:
        return source_tree_digest(
            {
                item.relative_path: (item.digest, item.mode, item.size)
                for item in source_files
            }
        )

    def _extract_tar(
        self,
        archive: Path,
        destination: Path,
        extracted_paths: dict[str, str],
        maximum_bytes: int,
        maximum_entries: int,
    ) -> tuple[int, int]:
        physical_members = self._validate_canonical_tar(archive, maximum_entries)
        total_bytes = 0
        implicit_directories = 0
        with tarfile.open(archive, mode="r:") as package:
            for member in package:
                relative = self.safe_archive_path(member.name, member.isdir())
                if relative is None:
                    if not member.isdir():
                        raise self.error_type(
                            "tar contains a hidden regular-file member"
                        )
                    continue
                if not member.isdir() and not member.isfile():
                    raise self.error_type("tar links and special files are forbidden")
                implicit_directories += self._reserve_archive_path(
                    relative,
                    member.isdir(),
                    extracted_paths,
                )
                if physical_members + implicit_directories > maximum_entries:
                    raise self.error_type("archive exceeds configured entry limit")
                target = destination / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                total_bytes += member.size
                if total_bytes > maximum_bytes:
                    raise self.error_type("archive exceeds configured size limit")
                source = package.extractfile(member)
                if source is None:
                    raise self.error_type("tar file entry has no content")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
        return total_bytes, physical_members + implicit_directories

    def _validate_canonical_tar(self, archive: Path, maximum_entries: int) -> int:
        archive_size = archive.stat().st_size
        if archive_size % _TAR_BLOCK_SIZE != 0:
            raise self.error_type("tar archive has an incomplete block")
        physical_members = 0
        terminated = False
        with archive.open("rb") as file_handle:
            while file_handle.tell() < archive_size:
                header = file_handle.read(_TAR_BLOCK_SIZE)
                if header == _TAR_ZERO_BLOCK:
                    second_zero = file_handle.read(_TAR_BLOCK_SIZE)
                    if second_zero != _TAR_ZERO_BLOCK:
                        raise self.error_type("tar archive has an invalid end marker")
                    while file_handle.tell() < archive_size:
                        padding = file_handle.read(DEFAULT_BUFFER_SIZE)
                        if padding.strip(b"\0"):
                            raise self.error_type(
                                "tar archive has data after its end marker"
                            )
                    terminated = True
                    break
                physical_members += 1
                if physical_members > maximum_entries:
                    raise self.error_type("archive exceeds configured entry limit")
                member_type = header[156:157]
                if member_type not in _CANONICAL_TAR_TYPES:
                    raise self.error_type(
                        "tar extension headers and special files are forbidden"
                    )
                raw_size = header[124:136].strip(b"\0 ")
                if raw_size and re.fullmatch(rb"[0-7]+", raw_size) is None:
                    raise self.error_type("tar member size is not canonical octal")
                member_size = int(raw_size, 8) if raw_size else 0
                if member_type == b"5" and member_size != 0:
                    raise self.error_type("tar directory contains payload bytes")
                padded_size = (
                    (member_size + _TAR_BLOCK_SIZE - 1) // _TAR_BLOCK_SIZE
                ) * _TAR_BLOCK_SIZE
                next_header = file_handle.tell() + padded_size
                if next_header > archive_size:
                    raise self.error_type("tar member exceeds archive bounds")
                file_handle.seek(padded_size, os.SEEK_CUR)
        if not terminated:
            raise self.error_type("tar archive has no canonical end marker")
        return physical_members

    def _validate_single_zstd_frame(
        self,
        archive: Path,
        maximum_bytes: int,
    ) -> int | None:
        archive_size = archive.stat().st_size
        with archive.open("rb") as file_handle:
            frame_header = file_handle.read(_ZSTD_MAX_FRAME_HEADER_SIZE)
            header_size = zstandard.frame_header_size(frame_header)
            parameters = zstandard.get_frame_parameters(frame_header)
            if parameters.window_size > self.zstd_window_size_bytes:
                raise self.error_type("zstd frame window exceeds configured size limit")
            expected_content_size = (
                None
                if parameters.content_size == zstandard.CONTENTSIZE_UNKNOWN
                else parameters.content_size
            )
            if (
                expected_content_size is not None
                and expected_content_size > maximum_bytes
            ):
                raise self.error_type(
                    "decompressed archive exceeds configured size limit"
                )
            file_handle.seek(header_size)
            last_block = False
            while not last_block:
                block_header = file_handle.read(3)
                if len(block_header) != 3:
                    raise self.error_type("zstd frame has an incomplete block")
                descriptor = int.from_bytes(block_header, "little")
                last_block = bool(descriptor & 1)
                block_type = (descriptor >> 1) & 3
                block_size = descriptor >> 3
                if block_type == 3 or block_size > _ZSTD_MAX_BLOCK_SIZE:
                    raise self.error_type("zstd frame block is invalid")
                stored_size = 1 if block_type == 1 else block_size
                next_block = file_handle.tell() + stored_size
                if next_block > archive_size:
                    raise self.error_type("zstd frame block exceeds archive bounds")
                file_handle.seek(stored_size, os.SEEK_CUR)
            if parameters.has_checksum:
                checksum = file_handle.read(4)
                if len(checksum) != 4:
                    raise self.error_type("zstd frame checksum is incomplete")
            if file_handle.tell() != archive_size:
                raise self.error_type(
                    "zstd archive must contain exactly one canonical frame"
                )
        return expected_content_size

    def _reserve_archive_path(
        self,
        path: str,
        is_directory: bool,
        extracted_paths: dict[str, str],
    ) -> int:
        parts = PurePosixPath(path).parts
        implicit_directories = 0
        for depth in range(1, len(parts)):
            parent = PurePosixPath(*parts[:depth]).as_posix()
            state = extracted_paths.get(parent)
            if state == "file":
                raise self.error_type("archive path descends through a regular file")
            if state is None:
                extracted_paths[parent] = "implicit-directory"
                implicit_directories += 1
        state = extracted_paths.get(path)
        if is_directory and state == "implicit-directory":
            extracted_paths[path] = "directory"
            return implicit_directories
        if state is not None:
            raise self.error_type("archive packages contain a duplicate path")
        extracted_paths[path] = "directory" if is_directory else "file"
        return implicit_directories
