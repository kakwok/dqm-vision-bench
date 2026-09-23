"""
Launch the DQM MCP server.

    python -m dqm_mcp --model google/gemma4-31b --provider litellm      # stdio (default)
    python -m dqm_mcp --model google/gemma4-31b --provider litellm \
        --transport streamable-http --host 127.0.0.1 --port 8000

The model and provider used by query_plot are fixed at startup and required:
pass --model/--provider or set OWUI_MODEL/DEFAULT_PROVIDER in .env. Clients
cannot change them.

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

from .config import REPO_ROOT, get_settings

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m dqm_mcp", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="model for query_plot (default: $OWUI_MODEL)")
    parser.add_argument("--provider", help="provider for query_plot, a key of owui_client.PROVIDERS "
                                           "(default: $DEFAULT_PROVIDER)")
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

    try:
        model, provider = _choose_model(args.model, args.provider)
        settings = get_settings()  # also refuses output dirs that overlap testing data
    except (ValueError, EnvironmentError) as e:
        parser.exit(2, f"error: {e}\n")
    logging.info("query model=%s provider=%s", model, provider)
    logging.info("data dir: %s", settings.data_dir)

    if args.transport == "stdio":
        from .server import build_app
        build_app().run(transport="stdio")
        return

    # Check auth and bind address before opening a port, so a misconfigured
    # launch fails fast.
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


def _choose_model(model: str | None, provider: str | None) -> tuple[str, str]:
    """
    Settle the query model and provider for the life of the process, or raise.

    The choice is written back to the environment before owui_client is first
    imported, so its module-level defaults and Settings.from_env agree.
    """
    model = (model or os.environ.get("OWUI_MODEL", "")).strip()
    provider = (provider or os.environ.get("DEFAULT_PROVIDER", "")).strip()
    if not model or not provider:
        raise ValueError(
            "choose a model and provider: --model <name> --provider {litellm,owui,nrp} "
            "(or OWUI_MODEL / DEFAULT_PROVIDER in .env)"
        )
    os.environ["OWUI_MODEL"] = model
    os.environ["DEFAULT_PROVIDER"] = provider

    import owui_client  # raises EnvironmentError without OWUI_API_KEY
    owui_client.get_provider(provider)  # unknown or unconfigured provider raises
    return model, provider


if __name__ == "__main__":
    main()
