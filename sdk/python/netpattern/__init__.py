"""Minimal client for the NetPattern API (standard library only).

    from netpattern import Client

    client = Client(api_key="np_…", base_url="http://localhost:8000")
    result = client.predict("dep_…", ["part_0142.png"], top_k=3)
    print(result["outputs"][0]["label"])
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

__all__ = ["Client", "ApiError"]
__version__ = "0.1.0"


class ApiError(RuntimeError):
    """A problem+json error from the API (status, code, title, detail)."""

    def __init__(self, status: int, body: dict[str, Any]) -> None:
        self.status = status
        self.code = body.get("code")
        self.detail = body.get("detail")
        super().__init__(f"{status} {body.get('title', 'error')}: {self.detail or ''}".strip())


class Client:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120,
    ) -> None:
        self.api_key = api_key or os.environ.get("NETPATTERN_API_KEY")
        if not self.api_key:
            raise ValueError("pass api_key or set NETPATTERN_API_KEY")
        self.base_url = (base_url or os.environ.get("NETPATTERN_URL") or "http://localhost:8000").rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.base_url + path, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self.api_key}")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read() or b"null")
        except urllib.error.HTTPError as error:
            try:
                payload = json.loads(error.read() or b"{}")
            except json.JSONDecodeError:
                payload = {}
            raise ApiError(error.code, payload) from error

    def schema(self, deployment_id: str) -> dict[str, Any]:
        """What the deployment accepts and returns (task, classes, input size, limits)."""
        return self._request("GET", f"/v1/deployments/{deployment_id}/schema")

    def predict(
        self, deployment_id: str, images: list[str | Path | bytes], **options: Any
    ) -> dict[str, Any]:
        """Predict one or more images (file paths or raw bytes). Options: top_k, threshold."""
        inputs = []
        for image in images:
            data = image if isinstance(image, bytes) else Path(image).read_bytes()
            inputs.append({"image_base64": base64.b64encode(data).decode()})
        return self._request(
            "POST", f"/v1/deployments/{deployment_id}/predict", {"inputs": inputs, "options": options}
        )
