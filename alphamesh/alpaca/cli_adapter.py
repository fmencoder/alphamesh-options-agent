"""Alpaca CLI adapter.

A thin, honest wrapper around an installed Alpaca CLI binary. It is an
*operational* path - for a human or an operator agent to inspect the paper
account, the clock and open orders without going through the Python runtime.

If no CLI is installed, :meth:`available` returns False and every call raises.
Nothing here simulates CLI output.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

DEFAULT_TIMEOUT = 30.0

READ_ONLY_COMMANDS: dict[str, list[str]] = {
    "account": ["account", "get"],
    "clock": ["clock", "get"],
    "positions": ["positions", "list"],
    "orders": ["orders", "list"],
}
"""Commands the adapter is willing to run. Order-placing verbs are absent by
design: the CLI path is for inspection, not for autonomous execution."""


class CliUnavailableError(RuntimeError):
    """No Alpaca CLI binary is installed or it is not executable."""


@dataclass(frozen=True)
class CliResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def json(self) -> Any:
        return json.loads(self.stdout)


class AlpacaCliAdapter:
    def __init__(self, binary: str = "alpaca", paper: bool = True) -> None:
        self.binary = binary
        self.paper = paper

    def resolved_path(self) -> str | None:
        return shutil.which(self.binary)

    def available(self) -> bool:
        return self.resolved_path() is not None

    def _run(self, args: list[str], timeout: float = DEFAULT_TIMEOUT) -> CliResult:
        path = self.resolved_path()
        if path is None:
            raise CliUnavailableError(
                f"Alpaca CLI {self.binary!r} is not on PATH; "
                "install it or unset ALPACA_CLI_PATH"
            )
        command = [path, *args]
        if self.paper:
            command.append("--paper")
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CliResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def run_named(self, name: str) -> CliResult:
        """Run one of the allowlisted read-only commands."""
        if name not in READ_ONLY_COMMANDS:
            raise ValueError(
                f"{name!r} is not an allowlisted read-only CLI command; "
                f"choose from {sorted(READ_ONLY_COMMANDS)}"
            )
        return self._run(READ_ONLY_COMMANDS[name])

    def status(self) -> dict[str, Any]:
        """Report what the CLI path can actually do right now, without guessing."""
        path = self.resolved_path()
        if path is None:
            return {
                "available": False,
                "binary": self.binary,
                "detail": "not found on PATH",
                "commands": sorted(READ_ONLY_COMMANDS),
            }
        return {
            "available": True,
            "binary": path,
            "paper": self.paper,
            "commands": sorted(READ_ONLY_COMMANDS),
        }


__all__ = [
    "READ_ONLY_COMMANDS",
    "AlpacaCliAdapter",
    "CliResult",
    "CliUnavailableError",
]
