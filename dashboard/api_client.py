"""HTTP transport for the NetLift dashboard."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_URL = os.getenv("NETLIFT_API_URL", "http://127.0.0.1:8000").rstrip("/")


def post(path: str, payload: bytes, *, content_type: str, accept: str) -> bytes:
    """Send a request and surface the API's structured validation messages."""
    request = Request(
        f"{API_URL}{path}",
        data=payload,
        headers={"Content-Type": content_type, "Accept": accept},
        method="POST",
    )
    return _read_response(request)


def get_json(path: str) -> dict:
    """Read a JSON object from a dashboard endpoint."""
    request = Request(f"{API_URL}{path}", headers={"Accept": "application/json"})
    payload = _read_response(request)
    try:
        result = json.loads(payload)
    except ValueError as exc:
        raise RuntimeError("The NetLift API returned invalid JSON.") from exc
    if not isinstance(result, dict):
        raise TypeError("The NetLift API returned an unexpected JSON response.")
    return result


def _read_response(request: Request) -> bytes:
    try:
        with urlopen(request, timeout=60) as response:
            return response.read()
    except HTTPError as exc:
        with exc:
            try:
                error = json.load(exc)["error"]
                message = error["message"]
                details = [
                    detail["message"]
                    for detail in error.get("details", [])
                    if "message" in detail
                ]
                if details:
                    message = f"{message} {' '.join(details[:3])}"
            except (KeyError, ValueError, TypeError, AttributeError):
                message = f"The NetLift API returned HTTP {exc.code}."
        raise RuntimeError(message) from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach the NetLift API at {API_URL}.") from exc
