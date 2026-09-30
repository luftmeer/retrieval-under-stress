"""Minimal HTTP client for the retrieval API."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ApiClientError(RuntimeError):
    pass


def request_json(
    url: str,
    *,
    payload: dict | None = None,
    timeout: float = 60,
) -> dict:
    body = None
    headers = {}

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

        headers["Content-Type"] = "application/json"

    request = Request(
        url,
        data=body,
        headers=headers,
        method=("POST" if payload is not None else "GET"),
    )

    try:
        with urlopen(
            request,
            timeout=timeout,
        ) as response:
            return json.load(response)

    except HTTPError as exc:
        try:
            detail = json.load(exc)
        except Exception:
            detail = exc.reason

        raise ApiClientError(f"API returned HTTP {exc.code}: {detail}") from exc

    except URLError as exc:
        raise ApiClientError(f"Cannot reach retrieval API: {exc.reason}") from exc


def health(base_url: str) -> dict:
    return request_json(
        f"{base_url.rstrip('/')}/health",
        timeout=10,
    )


def search(
    base_url: str,
    *,
    query: str,
    method: str,
    top_k: int,
) -> dict:
    return request_json(
        f"{base_url.rstrip('/')}/search",
        payload={
            "query": query,
            "method": method,
            "top_k": top_k,
        },
    )
