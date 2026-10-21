"""Helpers for the inference API: input files and a simple rate limit."""

import base64
import binascii
import time
from collections import defaultdict
from pathlib import Path

# Magic bytes → file suffix. The engine picks its reader from the suffix, so the format is
# taken from the content, never from a client-supplied name.
SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"II*\x00", ".tif"),
    (b"MM\x00*", ".tif"),
    (b"BM", ".bmp"),
)
ACCEPTED_FORMATS = ["png", "jpeg", "tiff", "bmp", "webp", "dicom"]


class InputError(ValueError):
    pass


def sniff_suffix(data: bytes) -> str:
    for signature, suffix in SIGNATURES:
        if data.startswith(signature):
            return suffix
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[128:132] == b"DICM":
        return ".dcm"
    raise InputError(f"unsupported image format (accepted: {', '.join(ACCEPTED_FORMATS)})")


def decode_base64(value: str) -> bytes:
    if "," in value[:64] and value.startswith("data:"):
        value = value.split(",", 1)[1]
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise InputError("image_base64 is not valid base64") from error


def write_inputs(folder: Path, images: list[bytes]) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for number, data in enumerate(images):
        path = folder / f"{number}{sniff_suffix(data)}"
        path.write_bytes(data)
        paths.append(str(path))
    return paths


class RateLimiter:
    """Fixed one-minute windows per key, in this process (enough for one backend instance)."""

    def __init__(self) -> None:
        self.windows: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))

    def allow(self, key: str, limit: int) -> bool:
        window = int(time.time() // 60)
        start, count = self.windows[key]
        if start != window:
            start, count = window, 0
        if count >= limit:
            return False
        self.windows[key] = (start, count + 1)
        return True


limiter = RateLimiter()
