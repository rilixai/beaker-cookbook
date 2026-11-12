"""Linux filesystem confinement for the agent and all of its MCP descendants.

Landlock is required: an unsupported kernel must fail closed, never run with
the trusted benchmark cache exposed. Network access remains available to tools.
See https://docs.kernel.org/userspace-api/landlock.html.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path


class _Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def restrict_filesystem(*, read_paths: list[Path], write_paths: list[Path]) -> None:
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("Toolathlon onboarding requires Linux x86_64/aarch64 with Landlock ABI >= 3")
    libc = ctypes.CDLL(None, use_errno=True)

    def checked(value: int) -> int:
        if value < 0:
            error = ctypes.get_errno()
            raise OSError(error, f"Landlock sandbox unavailable: {os.strerror(error)}")
        return value

    abi = checked(libc.syscall(444, 0, 0, 1))  # landlock_create_ruleset(VERSION)
    if abi < 3:
        raise RuntimeError("Toolathlon onboarding requires Landlock ABI >= 3 (including TRUNCATE)")
    # ABI 3 covers execute, read, write, create, remove, rename and truncate.
    all_access = (1 << 15) - 1
    read_access = (1 << 0) | (1 << 2) | (1 << 3)
    write_access = all_access & ~((1 << 6) | (1 << 11))  # Never create device nodes.
    ruleset = _Ruleset(all_access)
    ruleset_fd = checked(libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0))
    try:
        for paths, access in ((read_paths, read_access), (write_paths, write_access)):
            for path in paths:
                path = path.resolve(strict=True)
                path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
                try:
                    # Directory-only rights cannot be granted to individual files.
                    rights = access if path.is_dir() else access & ((1 << 0) | (1 << 1) | (1 << 2) | (1 << 14))
                    rule = _PathRule(rights, path_fd)
                    checked(libc.syscall(445, ruleset_fd, 1, ctypes.byref(rule), 0))
                finally:
                    os.close(path_fd)
        checked(libc.prctl(38, 1, 0, 0, 0))  # PR_SET_NO_NEW_PRIVS
        checked(libc.syscall(446, ruleset_fd, 0))  # landlock_restrict_self
    finally:
        os.close(ruleset_fd)


if __name__ == "__main__":
    import json

    if sys.argv[1] == "--check":
        restrict_filesystem(read_paths=[], write_paths=[])
    else:
        policy = json.loads(Path(sys.argv[1]).read_text())
        restrict_filesystem(
            read_paths=[Path(path) for path in policy["read"]],
            write_paths=[Path(path) for path in policy["write"]],
        )
        os.execv(sys.argv[2], sys.argv[2:])
