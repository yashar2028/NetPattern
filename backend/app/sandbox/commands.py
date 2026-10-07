"""Small async helper for running host commands (nix) with a timeout."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandError(RuntimeError):
    pass


async def run_command(
    *args: str, cwd: Path | None = None, timeout: float = 3600, log_file: Path | None = None
) -> CommandResult:
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError as error:
        process.kill()
        await process.communicate()
        raise CommandError(f"{args[0]} timed out after {timeout:.0f}s") from error
    result = CommandResult(
        process.returncode or 0,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )
    if log_file is not None:
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"$ {' '.join(args)}\n{result.stderr}\n")
    return result


def tail(text: str, lines: int = 15) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])
