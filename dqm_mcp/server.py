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
- The model and provider are fixed when the server starts; server_status shows them.
  Do not try to choose a model.
- get_plot_images returns the input panels and the reference images, each preceded by a
  text label saying which is which; query_plot returns only their paths. Some clients
  cannot display images.
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
    def get_plot_images(run: int | None, subsystem: str, plot_number: str,
                        folder: str | None = None, include_input: bool = True,
                        include_references: bool = True, overwrite: bool = False):
        """
        The images behind a verdict: the plot's input panels for a run (fetched from the
        online DQM GUI if not already stored) and the reference images for the plot type.
        Each image is preceded by a text label ("Input panel i of N" / "Reference k of M"),
        and a metadata dict (urls, paths, sources, placeholder flags, budget) comes last.
        run is required when include_input is true. Only input panels that hit the network
        count against the fetch guardrail; references are local and free.
        """
        with _guarded():
            group = resolve.group_for(subsystem, plot_number, folder)
            content: list[Any] = []
            meta: dict[str, Any] = {
                "run": None,
                "subsystem": group.subsystem,
                "plot_number": group.plot_number,
                "folder": group.folder,
                "title": group.title,
                "n_panels": group.n_panels,
            }

            if include_input:
                if run is None:
                    raise ValueError("run is required when include_input is true.")
                run_text = fetch.run_display(run)
                fetched = fetch.fetch_group(run_text, group, overwrite=overwrite)
                meta["run"] = int(run_text)
                meta["panels"] = [f.meta() for f in fetched]
                for i, f in enumerate(fetched, start=1):
                    content.append(f"Input panel {i} of {len(fetched)}: {f.stem} "
                                   f"(run {run_text}, source {f.source})")
                    content.append(Image(data=f.path.read_bytes(), format="png"))

            if include_references:
                # Keyed on the first panel's stem; the file need not exist.
                refs = query.references_for(fetch.image_path_for(group, group.stems[0], run or 0))
                meta["reference_images"] = [str(r) for r in refs]
                if not refs:
                    content.append(f"No reference images for {group.folder} "
                                   f"under {get_settings().ref_dir}.")
                for k, r in enumerate(refs, start=1):
                    content.append(f"Reference {k} of {len(refs)}: {r.name}")
                    content.append(Image(data=r.read_bytes(), format="png"))

            meta["budget"] = get_budget().status()
            return content + [meta]

    @app.tool()
    def query_plot(run: int, subsystem: str, plot_number: str, event_type: EventType,
                   folder: str | None = None, use_cache: bool = True,
                   overwrite_image: bool = False) -> dict[str, Any]:
        """
        End to end: fetch the plot (all panels) for a run, attach the shifter instructions,
        ask the vision model, and return the raw response plus parsed sections and a
        GOOD/BAD verdict. event_type is required and comes from the operator. The model
        and provider are configured on the server. To see the images that were judged,
        call get_plot_images with the same run and plot. Results are saved under
        dqm_mcp_data/results/<run_id>/<folder>/ and reused when use_cache is true.
        """
        with _guarded():
            group = resolve.group_for(subsystem, plot_number, folder)
            run_text = fetch.run_display(run)
            fetched = fetch.fetch_group(run_text, group, overwrite=overwrite_image)
            model_name = query.resolve_model()

            cached = results.cached_response(group, run_text, model_name) if use_cache else None
            if cached is not None and not cached.get("error"):
                return _assemble(group, run_text, cached, cached=True, fetched=fetched,
                                 event_type=event_type)

            result = query.ask_model(
                [f.path for f in fetched], [f.stem for f in fetched],
                run=run_text, event_type=event_type, model=model_name,
            )
            result["result_file"] = str(results.save_response(group, run_text, result))
            return _assemble(group, run_text, result, cached=False, fetched=fetched,
                             event_type=event_type)

    @app.tool()
    def get_cached_response(run: int, subsystem: str, plot_number: str,
                            folder: str | None = None) -> dict[str, Any]:
        """Return a previously saved query_plot answer for this run/plot (server's model), without fetching or querying."""
        with _guarded():
            group = resolve.group_for(subsystem, plot_number, folder)
            run_text = fetch.run_display(run)
            model_name = query.resolve_model()
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
