"""
Read and write query results in the same ``.txt`` format the batch pipeline
uses, so answers produced through MCP sit next to batch answers under
``<results_root>/<run_id>/<folder>/`` and evaluate.py can load them.

A multi-panel plot is one result; its file is named after the folder rather
than any single panel stem.
"""
from __future__ import annotations

from pathlib import Path

from .config import get_settings
from .resolve import PlotGroup


def _owui():
    import owui_client  # lazy: raises EnvironmentError without OWUI_API_KEY
    return owui_client


def image_key(group: PlotGroup, run: int | str) -> Path:
    """Synthetic image path whose stem names the whole plot for one run."""
    return Path(f"{group.folder}_run{str(run).zfill(6)}.png")


def result_file(group: PlotGroup, run: int | str, model: str) -> Path:
    o = _owui()
    s = get_settings()
    out_dir = o.resolve_output_dir(s.results_root, group.folder, s.run_id)
    return o.resolve_output_file(out_dir, image_key(group, run), model)


def cached_response(group: PlotGroup, run: int | str, model: str) -> dict | None:
    path = result_file(group, run, model)
    if not path.exists():
        return None
    result = _owui()._parse_result_file(path, group.folder)
    result["result_file"] = str(path)
    return result


def save_response(group: PlotGroup, run: int | str, result: dict) -> Path:
    path = result_file(group, run, result["model"])
    path.parent.mkdir(parents=True, exist_ok=True)
    _owui()._write_result_file(path, result, group.folder, get_settings().run_id)
    return path
