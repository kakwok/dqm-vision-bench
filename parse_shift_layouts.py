#!/usr/bin/env python3
"""
Parse CMS DQM shift layout Python files by executing them in a controlled
namespace, intercepting DQMItem and dqmitems to capture all plot definitions.

Usage:
    python3 parse_layouts.py ./shift_*_layout.py -o shift_layouts.json
"""

import re
import glob
import json
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Shims that stand in for the GUI framework objects
# ---------------------------------------------------------------------------

class DQMItem:
    """Captures the row/cell dicts passed as layout=rows."""
    def __init__(self, layout=None):
        self.layout = layout or []


class DQMItemDict(dict):
    """
    Drop-in for the `dqmitems` dict.
    Intercepts  dqmitems["00 Shift/SubSystem/Title"] = DQMItem(layout=rows)
    and records (key, item) pairs in insertion order.
    """
    def __setitem__(self, key, value):
        super().__setitem__(key, value)


# ---------------------------------------------------------------------------
# Parse one file
# ---------------------------------------------------------------------------

# Extract subsystem from the layout key, e.g. "00 Shift/BeamMonitor/00 - Title"
KEY_RE = re.compile(r'^00 Shift/([^/]+)/(.+)$')


def parse_file(filepath: Path) -> dict:
    """
    Execute the layout file in a sandboxed namespace and collect every
    dqmitems entry into a nested structure:

        {
            "subsystem":   str,
            "source_file": str,
            "plots": [
                {"title": str, "path": str, "description": str},
                ...
            ]
        }
    """
    source = filepath.read_text(encoding='utf-8')

    dqmitems = DQMItemDict()
    namespace = {
        "dqmitems": dqmitems,
        "DQMItem":  DQMItem,
    }

    exec(compile(source, str(filepath), 'exec'), namespace)  # noqa: S102

    # Derive subsystem from the first key that matches the expected pattern
    subsystem = "Unknown"
    for key in dqmitems:
        m = KEY_RE.match(key)
        if m:
            subsystem = m.group(1)
            break

    plots = []
    for key, item in dqmitems.items():
        m = KEY_RE.match(key)
        if not m:
            continue
        title = m.group(2)

        # Each row in item.layout is a list of cell dicts
        for row in item.layout:
            if row is None:
                continue
            # row may be a single dict or a list of dicts
            cells = row if isinstance(row, list) else [row]
            for cell in cells:
                if not isinstance(cell, dict) or 'path' not in cell:
                    continue
                plots.append({
                    "title":       title,
                    "path":        cell.get("path"),
                    "description": cell.get("description"),
                })

    return {
        "subsystem":   subsystem,
        "source_file": filepath.name,
        "plots":       plots,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse DQM shift layout files into JSON.")
    parser.add_argument("patterns", nargs="+",
                        help="Glob pattern(s) for layout .py files, e.g. ./shift_*_layout.py")
    parser.add_argument("-o", "--output", type=Path, default=Path("shift_layouts.json"),
                        help="Output JSON file (default: shift_layouts.json)")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent level")
    args = parser.parse_args()

    seen = set()
    files = []
    for pattern in args.patterns:
        for match in sorted(glob.glob(pattern)):
            if match not in seen:
                seen.add(match)
                files.append(Path(match))

    if not files:
        print(f"[ERROR] No files matched: {args.patterns}")
        return

    output = []
    for f in files:
        entry = parse_file(f)
        output.append(entry)
        print(f"  {f.name:40s}  {len(entry['plots']):4d} plots  "
              f"(subsystem: {entry['subsystem']})")

    args.output.write_text(
        json.dumps(output, indent=args.indent, ensure_ascii=False),
        encoding='utf-8'
    )
    print(f"\nWrote {len(output)} subsystems → {args.output}")


if __name__ == "__main__":
    main()
