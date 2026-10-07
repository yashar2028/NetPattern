"""Events the engine prints as JSON lines on stdout.

The worker reads stdout line by line while the process runs, so each event is
exactly one JSON object per line and stdout is flushed after every event.
Human-readable logs belong on stderr.
"""

import json
import sys
import time
from typing import Any, TextIO


def emit(event_type: str, *, stream: TextIO | None = None, **payload: Any) -> None:
    record = {"type": event_type, "ts": time.time(), **payload}
    out = stream if stream is not None else sys.stdout
    out.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
    out.flush()
