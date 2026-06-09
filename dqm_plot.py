#!/usr/bin/env python3
"""
dqm_plot.py — Extract DQM histograms from ROOT file(s) and save as PNG.

Designed to be imported by a Jupyter notebook (produce_images.ipynb) which
supplies the file list and PLOT_CONFIG. Can also be run from the command line.

CLI — single file:
    python dqm_plot.py \\
        "root://cmseos.fnal.gov//store/user/kkwok/DQM/DQM_V0001_L1T_R000398185.root" \\
        "DQMData/Run 398185/L1T/Run summary/L1TStage2CaloLayer1/ecalOccRecdEtWgt;1"

CLI — batch (uses PLOT_CONFIG defined in this file):
    python dqm_plot.py --batch \\
        --files "root://cmseos.fnal.gov//store/user/kkwok/DQM/DQM_V0001_L1T_*.root"

Output structure:
    images/
        <plotName>/
            <plotName>_run<XXXXXX>.png
"""

import argparse
import fnmatch
import os
import re
import subprocess
import sys

# ── ROOT import ────────────────────────────────────────────────────────────────
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
except ImportError:
    # Raise instead of sys.exit() so notebook import fails gracefully
    raise ImportError(
        "PyROOT not available. "
        "Source a CMSSW environment or activate a conda env with ROOT."
    )

# ==============================================================================
# Default PLOT_CONFIG — overridden by the notebook at runtime.
# Map: plot_name → internal ROOT object path template.
# Use {run} as a placeholder; it is zero-padded to 6 digits at render time.
# ==============================================================================
PLOT_CONFIG: dict[str, str] = {
    "ecalOccRecdEtWgt": (
        "DQMData/Run {run}/L1T/Run summary/L1TStage2CaloLayer1/ecalOccRecdEtWgt"
    ),
    "caloLayer2CenJets": (
        "DQMData/Run {run}/L1T/Run summary/L1TStage2CaloLayer2/caloLayer2CenJets"
    ),
}
# ==============================================================================


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def expand_files(patterns: list[str]) -> list[str]:
    """
    Expand a list of file path strings or glob patterns into concrete paths.

    Supports:
      - Exact local paths        /path/to/file.root
      - Local glob patterns      /path/to/DQM_*.root
      - XRootD exact URLs        root://cmseos.fnal.gov//store/.../file.root
      - XRootD glob patterns     root://cmseos.fnal.gov//store/.../DQM_*.root
        (resolved via `xrdfs ls` + fnmatch)
    """
    import glob as _glob

    result = []
    for pattern in patterns:
        if pattern.startswith("root://"):
            m = re.match(r"(root://[^/]+/)(/.+)", pattern)
            # Capture server prefix exactly as given (preserving double slash)
            # root://cmseos.fnal.gov//store/...
            #  group(1) = "root://cmseos.fnal.gov/"   (server, one trailing slash)
            #  group(2) = "/store/..."                 (remote path, leading slash)
            # Prefix = group(1) + "/" so we restore the double slash on output.
            if not m:
                print(f"WARNING: Cannot parse XRootD URL: {pattern}")
                result.append(pattern)
                continue
            server_base = m.group(1).rstrip("/")   # root://cmseos.fnal.gov
            remote_path = m.group(2)               # /store/user/...
            # Prefix used to reconstruct full URLs (preserves //)
            url_prefix  = f"{server_base}/"        # root://cmseos.fnal.gov/

            if any(c in remote_path for c in ("*", "?", "[")):
                parent          = os.path.dirname(remote_path)
                basename_pat    = os.path.basename(remote_path)
                try:
                    out = subprocess.check_output(
                        ["xrdfs", server_base, "ls", parent],
                        text=True, stderr=subprocess.DEVNULL,
                    )
                    matches = sorted(
                        f"{url_prefix}{p}"
                        for p in out.splitlines()
                        if fnmatch.fnmatch(os.path.basename(p), basename_pat)
                    )
                    if not matches:
                        print(f"WARNING: No XRootD files matched: {pattern}")
                    result.extend(matches)
                except Exception as e:
                    print(f"WARNING: xrdfs expansion failed for {pattern}: {e}")
                    result.append(pattern)
            else:
                result.append(pattern)
        else:
            expanded = sorted(_glob.glob(pattern))
            if not expanded:
                print(f"WARNING: No local files matched: {pattern}")
            result.extend(expanded)

    return result


def infer_run_number(root_path: str) -> str | None:
    """Extract run number from filename (R000XXXXXX pattern)."""
    m = re.search(r"R0*(\d+)", os.path.basename(root_path))
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# ROOT helpers
# ---------------------------------------------------------------------------

def open_file(root_path: str) -> ROOT.TFile:
    tf = ROOT.TFile.Open(root_path)
    if not tf or tf.IsZombie():
        raise OSError(f"Cannot open ROOT file: {root_path}")
    return tf


def get_object(tf: ROOT.TFile, obj_path: str):
    """
    Retrieve a histogram by internal path, stripping any cycle number (;N).
    Returns None (with a warning) if not found rather than exiting.
    """
    clean = re.sub(r";[0-9]+$", "", obj_path)
    obj = tf.Get(clean)
    if not obj:
        print(f"  WARNING: Object not found: {clean}")
        return None
    return obj


def style_canvas(canvas: ROOT.TCanvas):
    ROOT.gStyle.SetPalette(ROOT.kRainBow)
    ROOT.gStyle.SetNumberContours(255)
    ROOT.gStyle.SetFrameBorderMode(0)
    ROOT.gStyle.SetCanvasBorderMode(0)
    ROOT.gStyle.SetPadBorderMode(0)
    ROOT.gStyle.SetPadColor(0)
    ROOT.gStyle.SetCanvasColor(0)
    ROOT.gStyle.SetTitleFillColor(0)
    ROOT.gStyle.SetTitleBorderSize(0)
    ROOT.gStyle.SetPadTopMargin(0.08)
    ROOT.gStyle.SetPadBottomMargin(0.12)
    ROOT.gStyle.SetPadLeftMargin(0.14)
    ROOT.gStyle.SetPadRightMargin(0.14)
    canvas.SetFillColor(0)
    canvas.SetBorderMode(0)


def draw_option(obj) -> str:
    cls = obj.ClassName()
    if any(cls.startswith(p) for p in ("TH2", "TProfile2D")):
        return "COLZ"
    if cls.startswith("TH3"):
        return "BOX"
    return ""


# ---------------------------------------------------------------------------
# Core: save one histogram to PNG
# ---------------------------------------------------------------------------

def save_plot(
    obj,
    plot_name: str,
    run: str,
    outdir: str = "images",
    width: int = 900,
    height: int = 700,
) -> str:
    """
    Draw a ROOT histogram and save it as PNG.

    Parameters
    ----------
    obj       : ROOT histogram object (already detached from file).
    plot_name : Name used for the subdirectory and filename stem.
    run       : Run number string (will be zero-padded to 6 digits).
    outdir    : Top-level output directory.
    width     : Canvas width in pixels.
    height    : Canvas height in pixels.

    Returns
    -------
    str : Path to the saved PNG.
    """
    out_dir = os.path.join(outdir, plot_name)
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{plot_name}_run{run.zfill(6)}.png")

    canvas = ROOT.TCanvas("c1", plot_name, width, height)
    style_canvas(canvas)

    draw_opt = draw_option(obj)
    ROOT.gStyle.SetOptStat(10 if draw_opt in ("COLZ", "BOX") else 1111)

    obj.Draw(draw_opt)
    obj.GetXaxis().SetTitleSize(0.045)
    obj.GetYaxis().SetTitleSize(0.045)
    obj.GetXaxis().SetLabelSize(0.035)
    obj.GetYaxis().SetLabelSize(0.035)
    if draw_opt == "COLZ":
        obj.GetZaxis().SetTitleSize(0.040)
        obj.GetZaxis().SetLabelSize(0.030)
        ROOT.gPad.SetRightMargin(0.18)

    canvas.Update()
    canvas.SaveAs(out_png)
    canvas.Close()
    return out_png


# ---------------------------------------------------------------------------
# Batch entry point (called by notebook or CLI)
# ---------------------------------------------------------------------------

def produce_images(
    file_patterns: list[str],
    plot_config: dict[str, str],
    outdir: str = "images",
    run_override: str | None = None,
    width: int = 900,
    height: int = 700,
    verbose: bool = True,
) -> list[dict]:
    """
    For every file × every plot in plot_config, extract the histogram and
    save it as a PNG.

    Parameters
    ----------
    file_patterns : List of file paths or glob patterns (local or XRootD).
    plot_config   : Dict mapping plot_name → object path template with {run}.
    outdir        : Top-level output directory.
    run_override  : Force a specific run number for all files.
    width/height  : Canvas dimensions in pixels.
    verbose       : Print progress.

    Returns
    -------
    List of result dicts with keys:
        root_file, run, plot_name, out_png, error
    """
    files = expand_files(file_patterns)
    if not files:
        raise FileNotFoundError("No files found matching the provided patterns.")
    if not plot_config:
        raise ValueError("plot_config is empty.")

    total   = len(files) * len(plot_config)
    results = []
    n       = 0

    if verbose:
        print(f"{len(files)} file(s) × {len(plot_config)} plot(s) = {total} total\n")

    for root_path in files:
        run = run_override or infer_run_number(root_path)
        if run is None:
            print(f"SKIP (no run number): {root_path}")
            for plot_name in plot_config:
                results.append({"root_file": root_path, "run": None,
                                 "plot_name": plot_name, "out_png": None,
                                 "error": "run number not found"})
            continue

        run_display = run.lstrip("0") or "0"
        if verbose:
            print(f"── Run {run_display}  {os.path.basename(root_path)}")

        try:
            tf = open_file(root_path)
        except OSError as e:
            print(f"  ERROR opening file: {e}")
            for plot_name in plot_config:
                results.append({"root_file": root_path, "run": run_display,
                                 "plot_name": plot_name, "out_png": None,
                                 "error": str(e)})
            continue

        for plot_name, obj_template in plot_config.items():
            n += 1
            obj_path = obj_template.format(run=run_display.zfill(6))
            obj = get_object(tf, obj_path)

            if obj is None:
                if verbose:
                    print(f"  [{n}/{total}] SKIP {plot_name} — object not found")
                results.append({"root_file": root_path, "run": run_display,
                                 "plot_name": plot_name, "out_png": None,
                                 "error": "object not found"})
                continue

            obj.SetDirectory(0)
            try:
                out_png = save_plot(obj, plot_name, run_display, outdir, width, height)
                if verbose:
                    print(f"  [{n}/{total}] {plot_name} → {out_png}")
                results.append({"root_file": root_path, "run": run_display,
                                 "plot_name": plot_name, "out_png": out_png,
                                 "error": None})
            except Exception as e:
                print(f"  [{n}/{total}] ERROR {plot_name}: {e}")
                results.append({"root_file": root_path, "run": run_display,
                                 "plot_name": plot_name, "out_png": None,
                                 "error": str(e)})

        tf.Close()

    ok      = sum(1 for r in results if r["error"] is None)
    skipped = sum(1 for r in results if r["error"] is not None)
    if verbose:
        print(f"\nDone. {ok} saved, {skipped} skipped/errored.")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(
        description="Extract DQM histograms from ROOT file(s) and save as PNG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--batch", action="store_true",
                      help="Batch mode: iterate files × PLOT_CONFIG.")

    parser.add_argument("root_file",   nargs="?",
                        help="[single] Path/URL to one ROOT file.")
    parser.add_argument("object_path", nargs="?",
                        help="[single] Internal histogram path.")
    parser.add_argument("--files", nargs="+", metavar="PATTERN",
                        help="[batch] File paths or glob patterns.")
    parser.add_argument("--outdir",  default="images")
    parser.add_argument("--run",     default=None)
    parser.add_argument("--width",   type=int, default=900)
    parser.add_argument("--height",  type=int, default=700)
    args = parser.parse_args()

    if args.batch:
        if not args.files:
            parser.error("--batch requires --files")
        produce_images(
            args.files, PLOT_CONFIG,
            outdir=args.outdir, run_override=args.run,
            width=args.width, height=args.height,
        )
    else:
        if not args.root_file or not args.object_path:
            parser.error("Single mode requires: root_file object_path")
        run = args.run or infer_run_number(args.root_file)
        if run is None:
            parser.error("Cannot infer run number. Use --run XXXXXX.")
        run = run.lstrip("0") or "0"
        obj_path  = re.sub(r";[0-9]+$", "", args.object_path)
        plot_name = obj_path.rstrip("/").split("/")[-1]
        tf  = open_file(args.root_file)
        obj = get_object(tf, obj_path)
        if obj is None:
            sys.exit(1)
        obj.SetDirectory(0)
        tf.Close()
        out = save_plot(obj, plot_name, run, args.outdir, args.width, args.height)
        print(f"Saved → {out}")


if __name__ == "__main__":
    _cli()
