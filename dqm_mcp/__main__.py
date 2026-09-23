"""
Launch the DQM MCP server.

    python -m dqm_mcp                                   # stdio (default)
    python -m dqm_mcp --transport streamable-http --host 127.0.0.1 --port 8000

streamable-http requires a bearer token (DQM_MCP_TOKEN or DQM_MCP_TOKEN_FILE in
.env) and binds to loopback only unless --allow-remote is given.

Runs from the repository root regardless of the caller's cwd, loads .env from
there, and logs to stderr only (stdout is the protocol channel under stdio).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from .config import REPO_ROOT

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m dqm_mcp", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="streamable-http only")
    parser.add_argument("--port", type=int, default=8000, help="streamable-http only")
    parser.add_argument("--allow-remote", action="store_true",
                        help="streamable-http only: permit a non-loopback --host")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    args = parser.parse_args(argv)

    # The flat repo modules import each other by bare name and use relative
    # data paths, so run from the root no matter where the client started us.
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    logging.basicConfig(level=args.log_level, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.transport == "stdio":
        from .server import build_app
        build_app().run(transport="stdio")
        return

    # Check auth and bind address before importing the pipeline, so a
    # misconfigured launch fails fast and never opens a port.
    from .auth import BearerAuthMiddleware, TokenError, load_token, validate_token
    try:
        token = validate_token(load_token())
    except TokenError as e:
        parser.exit(2, f"error: {e}\n")
    if args.host not in LOOPBACK_HOSTS:
        if not args.allow_remote:
            parser.exit(2, f"error: --host {args.host} is not loopback. Prefer an SSH tunnel "
                           "(ssh -L 8000:127.0.0.1:8000 <host>); pass --allow-remote to bind anyway.\n")
        logging.warning("Binding to %s: traffic is plain HTTP, so the bearer token is visible "
                        "on the network. Use a TLS proxy or an SSH tunnel.", args.host)

    import uvicorn

    from .server import build_app
    # Passing host keeps the SDK's automatic DNS-rebinding protection on loopback.
    starlette_app = build_app().streamable_http_app(host=args.host)
    uvicorn.run(BearerAuthMiddleware(starlette_app, token),
                host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
