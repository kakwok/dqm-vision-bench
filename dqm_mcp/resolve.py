"""
Resolve (subsystem, plot number) to the shift-layout specs behind it.

A "plot" in the shifter's sense can be several monitor elements that share a
number and a title — ``build_image_config`` gives those ``_grpN`` stems and a
common ``folder``. This module groups them back into one PlotGroup so the rest
of the server treats them as a single plot with N panels.

Layouts are parsed once per process; ``shift_layout_helpers`` re-reads the
145 KB JSON on every call otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from shift_layout_helpers import _load as _load_layouts  # same file, same order
from shift_layout_helpers import build_image_config, list_subsystems

from .config import get_settings


@dataclass(frozen=True)
class PlotGroup:
    subsystem: str
    plot_number: str
    folder: str
    title: str
    specs: tuple[dict, ...]   # ordered grp0..N (or a single spec)

    @property
    def stems(self) -> list[str]:
        return [s["stem"] for s in self.specs]

    @property
    def n_panels(self) -> int:
        return len(self.specs)

    @property
    def me_paths(self) -> list[str]:
        return [s["path"] for s in self.specs]


# ---------------------------------------------------------------------------
# Cached layout access
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _subsystem_names() -> tuple[str, ...]:
    # shift_layouts.json lists 'Ecal' and 'Unknown' twice; build_image_config
    # returns the first block, so dedupe the same way.
    return tuple(dict.fromkeys(list_subsystems()))


@lru_cache(maxsize=None)
def _groups(subsystem: str) -> tuple[PlotGroup, ...]:
    specs = build_image_config(subsystem)
    # build_image_config walks entry["plots"] in order; zip to recover titles,
    # which the ImageSpec dicts do not carry.
    entry = next(e for e in _load_layouts() if e["subsystem"] == subsystem)
    titles = [p["title"] for p in entry["plots"]]

    by_folder: dict[str, list[tuple[dict, str]]] = {}
    for spec, title in zip(specs, titles):
        by_folder.setdefault(spec["folder"], []).append((spec, title))

    groups = []
    for folder, items in by_folder.items():
        first_spec, first_title = items[0]
        groups.append(PlotGroup(
            subsystem=subsystem,
            plot_number=first_spec["plot_number"],
            folder=folder,
            title=first_title,
            specs=tuple(s for s, _ in items),
        ))
    return tuple(groups)


def _has_instructions(subsystem: str) -> bool:
    return (get_settings().store_dir / f"{subsystem}.yaml").is_file()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def canonical_subsystem(name: str) -> str:
    """Return the subsystem name as spelled in shift_layouts.json (case-insensitive match)."""
    names = _subsystem_names()
    if name in names:
        return name
    lowered = {n.lower(): n for n in names}
    hit = lowered.get(name.strip().lower())
    if hit:
        return hit
    raise ValueError(
        f"Unknown subsystem {name!r}. Available: {', '.join(names)}"
    )


def subsystems() -> list[dict]:
    out = []
    for name in _subsystem_names():
        groups = _groups(name)
        out.append({
            "name": name,
            "n_plots": len(groups),
            "n_panels": sum(g.n_panels for g in groups),
            "has_instructions": _has_instructions(name),
        })
    return out


def plots(subsystem: str) -> list[dict]:
    name = canonical_subsystem(subsystem)
    has_yaml = _has_instructions(name)
    return [
        {
            "plot_number": g.plot_number,
            "title": g.title,
            "folder": g.folder,
            "n_panels": g.n_panels,
            "stems": g.stems,
            "me_paths": g.me_paths,
            "description": g.specs[0]["description"],
            "has_instructions": has_yaml,
        }
        for g in _groups(name)
    ]


def group_for(subsystem: str, plot_number: str, folder: str | None = None) -> PlotGroup:
    """
    Resolve one plot. ``plot_number`` may be given unpadded ('2' → '02') when
    the padded form exists. If several distinct plots share a number (same
    number, different titles), ``folder`` disambiguates; the error lists them.
    """
    name = canonical_subsystem(subsystem)
    groups = _groups(name)
    wanted = str(plot_number).strip()

    candidates = [g for g in groups if g.plot_number == wanted]
    if not candidates and wanted.isdigit():
        padded = wanted.zfill(2)
        candidates = [g for g in groups if g.plot_number == padded]
    if not candidates:
        numbers = list(dict.fromkeys(g.plot_number for g in groups))
        raise ValueError(
            f"No plot {plot_number!r} in subsystem {name!r}. "
            f"Available plot numbers: {', '.join(numbers) or '(none)'}"
        )
    if folder:
        candidates = [g for g in candidates if g.folder == folder]
        if not candidates:
            raise ValueError(f"No plot {wanted!r} in {name!r} with folder {folder!r}.")
    if len(candidates) > 1:
        listing = "; ".join(f"{g.folder} ({g.title!r}, {g.n_panels} panel(s))" for g in candidates)
        raise ValueError(
            f"Plot number {wanted!r} in {name!r} matches {len(candidates)} distinct plots. "
            f"Pass folder=<one of> to choose: {listing}"
        )
    return candidates[0]
