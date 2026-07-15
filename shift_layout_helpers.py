"""
Helpers for reading shift_layouts.json and building image configs
for use in produce_images.ipynb.

Image naming convention
-----------------------
    <subsystem>_<plotNumber>[_<subplot>]_run<XXXXXX>.png

  - subsystem   : sanitized subsystem name (e.g. "L1T", "P5_Cosmics")
  - plotNumber  : leading token from the plot title (e.g. "11", "A")
  - subplot     : last path component (or title suffix) when multiple plots
                  share the same plotNumber; omitted when the plot is unique
  - runNumber   : zero-padded 6-digit run number

Folder layout
-------------
    images/
        <subsystem>_<plotNumber>/        ← one folder per plot slot
            <stem>_run<XXXXXX>.png       ← one file per subplot × run
        ref/
            <stem>_run<XXXXXX>.png       ← reference / "good example" images
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

_LAYOUTS_FILE = Path(__file__).parent / "shift_layouts.json"


def _load() -> list[dict]:
    with open(_LAYOUTS_FILE) as f:
        return json.load(f)


def _sanitize(name: str) -> str:
    """Convert a path component to a filesystem-safe, dict-key-friendly string."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return name.strip("_")


def _plot_number(title: str) -> str:
    """Extract the leading number/letter prefix ('11 - uGMT...' → '11', '07 OnTrack...' → '07')."""
    m = re.match(r"^\s*(\S+)", title)
    return m.group(1) if m else _sanitize(title)[:8]


def _title_slug(title: str) -> str:
    """
    Strip the leading number/letter prefix from a title and remove all
    non-alphanumeric characters (including spaces) to produce a compact slug.

    '05 RecHit Energy'                 -> 'RecHitEnergy'
    '11 - uGMT MUON P_{T}'            -> 'uGMTMUONPT'
    '07 OnTrackCluster'                -> 'OnTrackCluster'
    '04 - fitted x0, sigma(x0) vs LS' -> 'fittedx0sigmax0vsLS'
    'B - muX vs lumi'                  -> 'muXvslumi'
    """
    suffix = re.sub(r"^\s*\S+\s*[-–]?\s*", "", title).strip()
    return re.sub(r"[^A-Za-z0-9]+", "", suffix)


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


# ---------------------------------------------------------------------------
# New image naming convention
# ---------------------------------------------------------------------------

def build_image_config(subsystem: str) -> list[dict]:
    """
    Return a list of ImageSpec dicts for all plots in *subsystem*.

    Naming convention
    -----------------
    stem   : <subsystem>_<plotNumber>_<titleSlug>[_grpN]
    folder : <subsystem>_<plotNumber>_<titleSlug>   (same for all grp siblings)

    grpN is added only when multiple plots share both the same plotNumber AND
    the same title (i.e. the same titleSlug); otherwise the title already
    makes each stem unique.

    Examples
    --------
    ECAL "05 RecHit Energy" × 3 paths:
        stem   ECAL_05_RecHitEnergy_grp0  (folder ECAL_05_RecHitEnergy)
        stem   ECAL_05_RecHitEnergy_grp1
        stem   ECAL_05_RecHitEnergy_grp2

    L1T "11 - uGMT MUON P_{T}" (unique):
        stem   L1T_11_uGMTMUONPT          (folder L1T_11_uGMTMUONPT)

    BeamPixel "B - muX vs lumi" / "B - muY vs lumi" (different titles):
        stem   BeamPixel_B_muXvslumi       (folder BeamPixel_B_muXvslumi)
        stem   BeamPixel_B_muYvslumi       (folder BeamPixel_B_muYvslumi)

    Each dict contains:
        stem        — filename stem
        folder      — image subdirectory (stem without grpN suffix)
        root_path   — ROOT path template with {run} placeholder
        path        — original JSON path (for reverse lookup)
        description — plain-text description (HTML tags stripped)
        subsystem   — subsystem name as stored in shift_layouts.json
        plot_number — leading number/letter from the title
        subplot     — 'grpN' string or None

    Use with produce_images():
        results = produce_images(FILE_PATTERNS, build_image_config("ECAL"), ...)

    Reference images live under images/ref/ with the same stem:
        images/ref/<stem>_run<XXXXXX>.png
    """
    for entry in _load():
        if entry["subsystem"] != subsystem:
            continue

        plots   = entry["plots"]
        sys_safe = _sanitize(subsystem)
        numbers = [_plot_number(p["title"]) for p in plots]
        slugs   = [_title_slug(p["title"]) for p in plots]

        # base stem = folder name (no grpN yet)
        base_stems = [f"{sys_safe}_{num}_{slug}" for num, slug in zip(numbers, slugs)]

        # Group by plot_number; add grpN only where base stems collide
        groups: dict[str, list[int]] = OrderedDict()
        for i, num in enumerate(numbers):
            groups.setdefault(num, []).append(i)

        final_stems: list[str]       = list(base_stems)
        subplots:    list[str | None] = [None] * len(plots)

        for indices in groups.values():
            if len(indices) == 1:
                continue
            group_bases = [base_stems[i] for i in indices]
            if len(set(group_bases)) < len(group_bases):
                # Collision (same title → same slug) → add grpN to ALL in group
                for rank, idx in enumerate(indices):
                    grp = f"grp{rank}"
                    final_stems[idx] = f"{base_stems[idx]}_{grp}"
                    subplots[idx] = grp

        result = []
        for i, plot in enumerate(plots):
            result.append({
                "stem":        final_stems[i],
                "folder":      base_stems[i],
                "root_path":   _to_root_path(plot["path"]),
                "path":        plot["path"],
                "description": re.sub(r"<[^>]+>", "", plot.get("description") or "").strip(),
                "subsystem":   subsystem,
                "plot_number": numbers[i],
                "subplot":     subplots[i],
                "style":       plot.get("style"),
            })
        return result

    raise ValueError(
        f"Subsystem '{subsystem}' not found.\n"
        f"Available: {list_subsystems()}"
    )


def build_stem_map() -> dict[str, dict]:
    """
    Return a dict mapping every image stem → its ImageSpec across all subsystems.

    Useful for reverse lookup: given a filename like
    'ECAL_05_RecHitEnergy_grp1_run398185.png', strip the run suffix and
    extension to get the stem 'ECAL_05_RecHitEnergy_grp1', then call
    lookup_by_stem().
    """
    result: dict[str, dict] = {}
    for entry in _load():
        for spec in build_image_config(entry["subsystem"]):
            result[spec["stem"]] = spec
    return result


def lookup_by_stem(stem: str) -> dict:
    """
    Return path and description for an image identified by its filename stem.

    Parameters
    ----------
    stem : Filename stem without run suffix or extension.
           e.g. 'ECAL_05_RecHitEnergy_grp1'  (from 'ECAL_05_RecHitEnergy_grp1_run398185.png')

    Returns
    -------
    dict with keys: stem, folder, root_path, path, description, subsystem,
                    plot_number, subplot
    """
    mapping = build_stem_map()
    if stem not in mapping:
        raise KeyError(
            f"Unknown stem {stem!r}.\n"
            f"Tip: strip '_run<XXXXXX>' and the extension, then call lookup_by_stem()."
        )
    return mapping[stem]


def format_image_config(subsystem: str) -> None:
    """
    Print a summary table of the new-convention image stems for *subsystem*.

    Example output
    --------------
    STEM                                       FOLDER            PATH
    L1T_00_ecalOccRecdEtWgt                   L1T_00            L1T/L1TStage2CaloLayer1/ecalOccRecdEtWgt
    L1T_01_caloLayer2CenJets                   L1T_01            L1T/L1TStage2CaloLayer2/caloLayer2CenJets
    ...
    """
    specs = build_image_config(subsystem)
    col = max(len(s["stem"]) for s in specs) + 2
    fol = max(len(s["folder"]) for s in specs) + 2
    print(f"{'STEM':<{col}} {'FOLDER':<{fol}} PATH")
    print("-" * (col + fol + 60))
    for s in specs:
        print(f"{s['stem']:<{col}} {s['folder']:<{fol}} {s['path']}")
