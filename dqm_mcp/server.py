"""
MCP tool definitions. This is the only module that imports the MCP SDK.

Works with the mcp Python SDK 2.x (``MCPServer``) and 1.x (``FastMCP``);
the decorator, ``Image`` and ``ToolError`` are the same in both.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager, redirect_stdout
from typing import Any, Literal

try:  # mcp >= 2
    from mcp.server.mcpserver import Image, MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer, Image
    from mcp.server.fastmcp.exceptions import ToolError

from . import __version__, context, fetch, parse, query, resolve, results
from .budget import BudgetError, get_budget
from .config import get_settings

EventType = Literal["collisions", "cosmics", "circulating"]

INSTRUCTIONS = """\
Tools over the CMS DQM vision bench: fetch a shift-layout plot from the ONLINE DQM GUI
for a run and ask a vision model whether it looks GOOD or BAD, using the shifter
instructions for that plot type.

How to use:
- Plot numbers are the leading token of the shift-layout title, usually zero-padded
  ('00', '02'); unpadded digits are accepted. Call list_subsystems / list_plots if unsure.
- A plot with several panels (n_panels > 1) is fetched and judged as ONE plot with one verdict.
- query_plot needs event_type from the operator (collisions | cosmics | circulating); the
  server does not look it up.
- The model and provider default to the server's environment (OWUI_MODEL, DEFAULT_PROVIDER)
  and can be overridden per call.
- Fetches from the DQM GUI are rate-limited (per call and per rolling hour); cached plots
  are always served. See server_status for the remaining budget and proxy state.
"""


@contextmanager
def _guarded():
    """
    Map internal errors to ToolError and keep stdout clean.

    Library code occasionally print()s (proxy expiry warnings, RAG debug);
    on the stdio transport stdout is the protocol channel, so any such
    output is redirected to stderr for the duration of the call.
    """
    try:
        with redirect_stdout(sys.stderr):
            yield
    except (ValueError, RuntimeError, BudgetError, FileNotFoundError) as e:
        raise ToolError(str(e)) from e


def _images_of(result: dict) -> list[str]:
    if result.get("images"):
        return list(result["images"])
    raw = result.get("image") or ""
    return [p for p in raw.split(";") if p]


def _assemble(group: resolve.PlotGroup, run: str, result: dict, *, cached: bool,
              fetched: list[fetch.FetchResult] | None, event_type: str | None) -> dict[str, Any]:
    structured = parse.structure_response(result.get("response"))
    instr = context.instructions_for(group)
    warnings = [
        f"{f.stem}: {f.size} B, possibly a 'not found' render"
        for f in (fetched or []) if f.placeholder
    ]
    return {
        "run": int(run),
        "subsystem": group.subsystem,
        "plot_number": group.plot_number,
        "folder": group.folder,
        "title": group.title,
        "n_panels": group.n_panels,
        "stems": group.stems,
        "me_paths": [f.me_path for f in fetched] if fetched else group.me_paths,
        "image_urls": [f.url for f in fetched] if fetched else None,
        "image_paths": _images_of(result),
        "image_sources": [f.source for f in fetched] if fetched else None,
        "placeholder_warnings": warnings,
        "instructions_found": instr["found"],
        "instructions": result.get("rag_text") or instr["instruction"],
        "reference_images": result.get("reference_images") or [],
        "event_type": event_type,
        "model": result.get("model"),
        "provider": result.get("provider"),
        "model_used": result.get("model_used"),
        "cached": cached,
        "result_file": result.get("result_file"),
        "response": result.get("response"),
        "sections": structured["sections"],
        "verdict": structured["verdict"],
        "latency_s": result.get("latency_s"),
        "load_latency_s": result.get("load_latency_s"),
        "error": result.get("error"),
        "budget": get_budget().status(),
    }


def build_app() -> MCPServer:
    app = MCPServer("dqm-vision-bench", instructions=INSTRUCTIONS, version=__version__)

    @app.tool()
    def list_subsystems() -> list[dict[str, Any]]:
        """List DQM subsystems from the shift layouts, with plot counts and whether curated instructions exist."""
        with _guarded():
            return resolve.subsystems()

    @app.tool()
    def list_plots(subsystem: str) -> list[dict[str, Any]]:
        """List the plots of one subsystem: plot_number, title, folder, n_panels, monitor-element paths."""
        with _guarded():
            return resolve.plots(subsystem)

    @app.tool()
    def get_plot_instructions(subsystem: str, plot_number: str, folder: str | None = None) -> dict[str, Any]:
        """Shifter instructions and known issues for one plot type. No image, no model call."""
        with _guarded():
            return context.instructions_for(resolve.group_for(subsystem, plot_number, folder))

    @app.tool()
    def fetch_plot_image(run: int, subsystem: str, plot_number: str,
                         folder: str | None = None, overwrite: bool = False):
        """
        Fetch every panel of a plot for a run from the online DQM GUI and return the PNGs
        followed by a metadata dict (urls, paths, sizes, sources, placeholder flags, budget).
        Subject to the fetch guardrail.
        """
        with _guarded():
            group = resolve.group_for(subsystem, plot_number, folder)
            fetched = fetch.fetch_group(run, group, overwrite=overwrite)
            meta = {
                "run": int(fetch.run_display(run)),
                "subsystem": group.subsystem,
                "plot_number": group.plot_number,
                "folder": group.folder,
                "title": group.title,
                "n_panels": group.n_panels,
                "panels": [f.meta() for f in fetched],
                "budget": get_budget().status(),
            }
            return [Image(data=f.path.read_bytes(), format="png") for f in fetched] + [meta]

    @app.tool()
    def query_plot(run: int, subsystem: str, plot_number: str, event_type: EventType,
                   folder: str | None = None, model: str | None = None,
                   provider: str | None = None, use_cache: bool = True,
                   overwrite_image: bool = False) -> dict[str, Any]:
        """
        End to end: fetch the plot (all panels) for a run, attach the shifter instructions,
        ask the vision model, and return the raw response plus parsed sections and a
        GOOD/BAD verdict. event_type is required and comes from the operator. Results are
        saved under results/<run_id>/<folder>/ and reused when use_cache is true.
        """
        with _guarded():
            group = resolve.group_for(subsystem, plot_number, folder)
            run_text = fetch.run_display(run)
            fetched = fetch.fetch_group(run_text, group, overwrite=overwrite_image)
            model_name = query.resolve_model(model)

            cached = results.cached_response(group, run_text, model_name) if use_cache else None
            if cached is not None and not cached.get("error"):
                return _assemble(group, run_text, cached, cached=True, fetched=fetched,
                                 event_type=event_type)

            result = query.ask_model(
                [f.path for f in fetched], [f.stem for f in fetched],
                run=run_text, event_type=event_type, model=model_name, provider=provider,
            )
            result["result_file"] = str(results.save_response(group, run_text, result))
            return _assemble(group, run_text, result, cached=False, fetched=fetched,
                             event_type=event_type)

    @app.tool()
    def get_cached_response(run: int, subsystem: str, plot_number: str,
                            folder: str | None = None, model: str | None = None) -> dict[str, Any]:
        """Return a previously saved query_plot answer for this run/plot/model, without fetching or querying."""
        with _guarded():
            group = resolve.group_for(subsystem, plot_number, folder)
            run_text = fetch.run_display(run)
            model_name = query.resolve_model(model)
            cached = results.cached_response(group, run_text, model_name)
            if cached is None:
                return {"found": False, "looked_in": str(results.result_file(group, run_text, model_name))}
            out = _assemble(group, run_text, cached, cached=True, fetched=None, event_type=None)
            out["found"] = True
            return out

    @app.tool()
    def server_status() -> dict[str, Any]:
        """Proxy state, default model/provider, fetch budget, and configured paths."""
        with _guarded():
            import os
            try:
                import mcp
                from importlib.metadata import version as _v
                sdk = _v("mcp")
            except Exception:
                sdk = None
            return {
                "server_version": __version__,
                "mcp_sdk_version": sdk,
                "proxy": fetch.proxy_status(),
                "model_client_configured": bool(os.environ.get("OWUI_API_KEY")),
                "settings": get_settings().public(),
                "budget": get_budget().status(),
            }

    return app
