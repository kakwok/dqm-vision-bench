"""
Helpers for reading shift_layouts.json and building PLOT_CONFIG blocks
for use in produce_images.ipynb.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_LAYOUTS_FILE = Path(__file__).parent / "shift_layouts.json"


def _load() -> list[dict]:
    with open(_LAYOUTS_FILE) as f:
        return json.load(f)


def _sanitize(name: str) -> str:
    """Convert a path component to a filesystem-safe, dict-key-friendly string."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return name.strip("_")


def _unique_keys(plots: list[dict]) -> list[str]:
    """
    Assign a unique dict key to each plot.

    Starts with the last path component and adds more leading components
    until all keys are distinct. Appends an index suffix as a last resort
    when sanitization collapses two distinct paths to the same string.
    """
    if not plots:
        return []

    def key_from_parts(parts: list[str], n: int) -> str:
        return _sanitize("_".join(parts[-n:]))

    raw = [p["path"].split("/") for p in plots]
    keys = [key_from_parts(parts, 1) for parts in raw]
    max_depth = max(len(p) for p in raw)

    depth = 1
    while depth < max_depth:
        seen: dict[str, list[int]] = {}
        for i, k in enumerate(keys):
            seen.setdefault(k, []).append(i)
        if all(len(v) == 1 for v in seen.values()):
            break
        depth += 1
        for indices in seen.values():
            if len(indices) > 1:
                for i in indices:
                    keys[i] = key_from_parts(raw[i], depth)

    # Last resort: append a counter for any still-colliding keys
    seen_final: dict[str, int] = {}
    for i, k in enumerate(keys):
        if k in seen_final:
            seen_final[k] += 1
            keys[i] = f"{k}_{seen_final[k]}"
        else:
            seen_final[k] = 0

    return keys


def _to_root_path(json_path: str) -> str:
    """
    Convert a shift_layouts.json path to a ROOT file path template.

    JSON path:  L1T/L1TStage2CaloLayer1/ecalOccRecdEtWgt
    ROOT path:  DQMData/Run {run}/L1T/Run summary/L1TStage2CaloLayer1/ecalOccRecdEtWgt
    """
    parts = json_path.split("/")
    return "DQMData/Run {run}/" + parts[0] + "/Run summary/" + "/".join(parts[1:])


def list_subsystems() -> list[str]:
    """Return all subsystem names found in shift_layouts.json."""
    return [entry["subsystem"] for entry in _load()]


def list_plots(subsystem: str, with_descriptions: bool = False) -> None:
    """
    Print the shift-layout plots for *subsystem*.

    Parameters
    ----------
    subsystem : str
        Must match the 'subsystem' field in shift_layouts.json exactly.
        Call list_subsystems() to see available names.
    with_descriptions : bool
        When True, also print each plot's description and internal path.
    """
    for entry in _load():
        if entry["subsystem"] != subsystem:
            continue
        print(f"Subsystem: {subsystem}  ({len(entry['plots'])} plots)\n")
        keys = _unique_keys(entry["plots"])
        for key, plot in zip(keys, entry["plots"]):
            print(f"  [{key}]  {plot['title']}")
            if with_descriptions:
                desc = re.sub(r"<[^>]+>", "", plot["description"]).strip()
                print(f"    path : {plot['path']}")
                print(f"    desc : {desc}")
        return
    raise ValueError(
        f"Subsystem '{subsystem}' not found.\n"
        f"Available: {list_subsystems()}"
    )


def build_plot_config(subsystem: str) -> dict[str, str]:
    """
    Return a PLOT_CONFIG dict for all plots in *subsystem*.

    Keys are sanitized histogram names (last path component, or last two if
    needed for uniqueness). Values are ROOT path templates with a {run}
    placeholder, ready for use in produce_images.ipynb.
    """
    for entry in _load():
        if entry["subsystem"] != subsystem:
            continue
        keys = _unique_keys(entry["plots"])
        return {
            key: _to_root_path(plot["path"])
            for key, plot in zip(keys, entry["plots"])
        }
    raise ValueError(
        f"Subsystem '{subsystem}' not found.\n"
        f"Available: {list_subsystems()}"
    )


def format_plot_config(subsystem: str) -> None:
    """
    Print a PLOT_CONFIG block for *subsystem*, ready to paste into
    produce_images.ipynb.

    Example output
    --------------
    PLOT_CONFIG = {
        # 00 - CaloLayer1 ECAL occupancy
        "ecalOccRecdEtWgt": (
            "DQMData/Run {run}/L1T/Run summary/L1TStage2CaloLayer1/ecalOccRecdEtWgt"
        ),
        ...
    }
    """
    for entry in _load():
        if entry["subsystem"] != subsystem:
            continue
        keys = _unique_keys(entry["plots"])
        lines = ["PLOT_CONFIG = {"]
        for key, plot in zip(keys, entry["plots"]):
            lines.append(f'    # {plot["title"]}')
            lines.append(f'    "{key}": (')
            lines.append(f'        "{_to_root_path(plot["path"])}"')
            lines.append(f'    ),')
        lines.append("}")
        print("\n".join(lines))
        return
    raise ValueError(
        f"Subsystem '{subsystem}' not found.\n"
        f"Available: {list_subsystems()}"
    )
