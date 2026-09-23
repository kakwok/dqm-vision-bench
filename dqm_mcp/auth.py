"""
Static bearer-token authentication for the streamable-http transport.

Every HTTP request must carry ``Authorization: Bearer <token>``, where the
token is a shared secret from ``DQM_MCP_TOKEN`` (or the file named by
``DQM_MCP_TOKEN_FILE``). stdio opens no port and does not use this.

The middleware is plain ASGI, so it wraps the SDK's Starlette app without
depending on which mcp version built it. Lifespan events pass straight
through: the app's lifespan is what starts the SDK session manager.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

MIN_TOKEN_LENGTH = 32
GENERATE_HINT = 'python -c "import secrets; print(secrets.token_urlsafe(32))"'


class TokenError(RuntimeError):
    """Raised when no usable token is configured."""


def load_token() -> str | None:
    """DQM_MCP_TOKEN if set, else the contents of DQM_MCP_TOKEN_FILE, else None."""
    token = os.environ.get("DQM_MCP_TOKEN", "").strip()
    if token:
        return token
    path = os.environ.get("DQM_MCP_TOKEN_FILE", "").strip()
    if path:
        p = Path(path).expanduser()
        try:
            return p.read_text().strip() or None
        except OSError as e:
            raise TokenError(f"DQM_MCP_TOKEN_FILE={path!r} could not be read: {e}") from e
    return None


def validate_token(token: str | None) -> str:
    if not token:
        raise TokenError(
            "streamable-http requires authentication: set DQM_MCP_TOKEN (or "
            f"DQM_MCP_TOKEN_FILE) in .env. Generate one with: {GENERATE_HINT}"
        )
    if len(token) < MIN_TOKEN_LENGTH:
        raise TokenError(
            f"DQM_MCP_TOKEN is {len(token)} characters; use at least {MIN_TOKEN_LENGTH}. "
            f"Generate one with: {GENERATE_HINT}"
        )
    return token


class BearerAuthMiddleware:
    """Reject any HTTP request whose Authorization header is not ``Bearer <token>``."""

    def __init__(self, app, token: str):
        self.app = app
        self._expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        presented = b""
        for name, value in scope.get("headers") or ():
            if name == b"authorization":
                presented = value
                break

        if hmac.compare_digest(presented, self._expected):
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        log.warning("Rejected unauthenticated %s %s from %s",
                    scope.get("method"), scope.get("path"),
                    f"{client[0]}:{client[1]}" if client else "unknown")
        body = json.dumps({"error": "unauthorized"}).encode()
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b"Bearer"),
            ],
        })
        await send({"type": "http.response.body", "body": body})
