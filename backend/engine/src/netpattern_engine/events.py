"""Events the engine prints as JSON lines on stdout.

The worker reads stdout line by line while the process runs, so each event is
exactly one JSON object per line and the stream is flushed after every event.
The runtime binds the real stdout here and points sys.stdout at stderr, so stray
prints from libraries can never corrupt the event stream.
"""

import json
import sys
import time
from typing import Any, TextIO

_stream: TextIO | None = None


def bind(stream: TextIO) -> None:
    global _stream
    _stream = stream


def emit(event_type: str, *, stream: TextIO | None = None, **payload: Any) -> None:
    record = {"type": event_type, "ts": round(time.time(), 3), **payload}
    out = stream or _stream or sys.stdout
    out.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
    out.flush()
