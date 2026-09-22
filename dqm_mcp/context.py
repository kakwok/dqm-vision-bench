"""
Shifter instructions for a plot, from plot_instructions/<subsystem>.yaml.

Keyed by the plot's ``folder`` (the stem without any ``_grpN``), which is how
``rag_backends`` keys the YAML and how the images are laid out on disk.
"""
from __future__ import annotations

from rag_backends import _load_yaml_doc, lookup_instruction

from .config import get_settings
from .resolve import PlotGroup


def instructions_for(group: PlotGroup) -> dict:
    store_dir = get_settings().store_dir
    text = lookup_instruction(group.folder, store_dir=store_dir)
    doc = _load_yaml_doc(store_dir, group.folder.split("_")[0]) or {}
    entry = (doc.get("plots") or {}).get(group.folder) or {}
    return {
        "subsystem": group.subsystem,
        "plot_number": group.plot_number,
        "folder": group.folder,
        "title": entry.get("title") or group.title,
        "found": bool(text),
        "instruction": text,
        "known_issues": entry.get("known_issues") or [],
        "last_updated": str(entry["last_updated"]) if entry.get("last_updated") else None,
        "layout_description": group.specs[0]["description"],
        "source": str(store_dir / f"{group.folder.split('_')[0]}.yaml") if entry else None,
    }
