"""Password hashing, access tokens and API keys (no third-party crypto, as in mcpquick)."""

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

from app.core.config import settings

PASSWORD_ALGORITHM = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 210_000
PASSWORD_SALT_BYTES = 16

API_KEY_PREFIX = "np_"
API_KEY_LOOKUP_LENGTH = 12  # characters stored in clear to find the key


class TokenError(ValueError):
    """Raised when an access token is invalid or expired."""


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


# --------------------------------------------------------------------------- passwords


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(PASSWORD_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return (
        f"{PASSWORD_ALGORITHM}${PASSWORD_ITERATIONS}$"
        f"{_b64url_encode(salt)}${_b64url_encode(digest)}"
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations_text, salt_text, digest_text = password_hash.split("$")
        if algorithm != PASSWORD_ALGORITHM:
            return False
        salt = _b64url_decode(salt_text)
        expected = _b64url_decode(digest_text)
        iterations = int(iterations_text)
    except (ValueError, TypeError):
        return False
    computed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(computed, expected)


# --------------------------------------------------------------------------- access tokens


def _sign(signing_input: bytes) -> bytes:
    return hmac.new(
        settings.AUTH_JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()


def create_access_token(user_id: str) -> str:
    """A signed HS256 JWT with subject and expiry."""
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + settings.AUTH_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }
    segments = [
        _b64url_encode(json.dumps(part, separators=(",", ":")).encode("utf-8"))
        for part in (header, payload)
    ]
    signing_input = ".".join(segments).encode("ascii")
    return f"{segments[0]}.{segments[1]}.{_b64url_encode(_sign(signing_input))}"


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
        signature = _b64url_decode(signature_segment)
    except ValueError as error:
        raise TokenError("token format is invalid") from error
    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    if not hmac.compare_digest(signature, _sign(signing_input)):
        raise TokenError("token signature is invalid")
    try:
        payload = json.loads(_b64url_decode(payload_segment))
    except ValueError as error:
        raise TokenError("token payload is invalid") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("sub"), str):
        raise TokenError("token payload is invalid")
    if not isinstance(payload.get("exp"), int) or payload["exp"] < int(time.time()):
        raise TokenError("token has expired")
    return payload


# --------------------------------------------------------------------------- API keys


def generate_api_key() -> str:
    return API_KEY_PREFIX + secrets.token_urlsafe(32)


def api_key_lookup(key: str) -> str:
    return key[:API_KEY_LOOKUP_LENGTH]


def hash_api_key(key: str) -> str:
    """API keys are long random strings, so a plain SHA-256 is enough to store them."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def is_api_key(credential: str) -> bool:
    return credential.startswith(API_KEY_PREFIX)
