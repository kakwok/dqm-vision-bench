"""
Launch the DQM MCP server.

    python -m dqm_mcp                                   # stdio (default)
    python -m dqm_mcp --transport streamable-http --host 0.0.0.0 --port 8000

Runs from the repository root regardless of the caller's cwd, loads .env from
there, and logs to stderr only (stdout is the protocol channel under stdio).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from .config import REPO_ROOT


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m dqm_mcp", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="streamable-http only")
    parser.add_argument("--port", type=int, default=8000, help="streamable-http only")
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

    from .server import build_app
    app = build_app()
    if args.transport == "stdio":
        app.run(transport="stdio")
    else:
        app.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
