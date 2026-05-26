#!/usr/bin/env python3
"""
dqm_plot.py — Extract a DQM histogram from a ROOT file and save as PNG.

Usage:
    python dqm_plot.py <root_file_path> <internal_object_path> [options]

Examples:
    python dqm_plot.py \
        "root://cmseos.fnal.gov//store/user/kkwok/DQM/DQM_V0001_L1T_R000398185.root" \
        "DQMData/Run 398185/L1T/Run summary/L1TStage2CaloLayer1/ecalOccRecdEtWgt;1"

    # Custom output directory and run number override
    python dqm_plot.py \
        "root://cmseos.fnal.gov//store/user/kkwok/DQM/DQM_V0001_L1T_R000398185.root" \
        "DQMData/Run 398185/L1T/Run summary/L1TStage2CaloLayer1/ecalOccRecdEtWgt;1" \
        --outdir my_images \
        --run 398185

Output structure:
    images/
        <plotName>/
            <plotName>_run<XXXXXX>.png
"""

import argparse
import os
import re
import sys

# ── ROOT import ────────────────────────────────────────────────────────────────
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)          # never open an X display
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
except ImportError:
    sys.exit("ERROR: PyROOT is not available. "
             "Source a CMSSW environment or activate a conda env with ROOT.")


# ── helpers ────────────────────────────────────────────────────────────────────

def infer_run_number(root_path: str):
    """Pull the run number from the file name (R000XXXXXX pattern)."""
    m = re.search(r"R0*(\d+)", os.path.basename(root_path))
    return m.group(1) if m else None


def open_file(root_path: str) -> ROOT.TFile:
    tf = ROOT.TFile.Open(root_path)
    if not tf or tf.IsZombie():
        sys.exit(f"ERROR: Cannot open file: {root_path}")
    return tf


def get_object(tf: ROOT.TFile, obj_path: str):
    """Retrieve a ROOT object by its internal path, stripping cycle number."""
    # Strip trailing ;N cycle number so TFile::Get works reliably
    clean = re.sub(r";[0-9]+$", "", obj_path)
    obj = tf.Get(clean)
    if not obj:
        # list top-level keys to help diagnose
        print("ERROR: Object not found:", clean)
        print("Top-level keys in file:")
        for key in tf.GetListOfKeys():
            print("  ", key.GetName())
        sys.exit(1)
    return obj


def style_canvas(canvas: ROOT.TCanvas):
    """Apply a clean CMS-ish style."""
    # OptStat set per-histogram in main() based on dimension
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
    """Choose a sensible Draw() option based on the histogram dimension."""
    cls = obj.ClassName()
    if any(cls.startswith(p) for p in ("TH2", "TProfile2D")):
        return "COLZ"
    if cls.startswith("TH3"):
        return "BOX"
    return ""          # 1-D default


def add_run_label(run: str, canvas: ROOT.TCanvas):
    """Add a small 'Run XXXXXX' watermark in the top-left corner."""
    label = ROOT.TLatex()
    label.SetNDC(True)
    label.SetTextFont(42)
    label.SetTextSize(0.035)
    label.SetTextColor(ROOT.kGray + 2)
    label.DrawLatex(0.16, 0.935, f"Run {run}")
    canvas.Update()
    # keep the label alive for the duration of the canvas lifetime
    canvas._run_label = label


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract a DQM histogram from a ROOT file and save as PNG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("root_file",  help="Path to the ROOT file (local or xrootd URL)")
    parser.add_argument("object_path", help="Internal path to the histogram inside the file")
    parser.add_argument("--outdir",  default="images",
                        help="Top-level output directory (default: images/)")
    parser.add_argument("--run",     default=None,
                        help="Run number override (auto-detected from filename if omitted)")
    parser.add_argument("--width",   type=int, default=900,  help="Canvas width  in pixels")
    parser.add_argument("--height",  type=int, default=700,  help="Canvas height in pixels")
    args = parser.parse_args()

    # ── run number ──────────────────────────────────────────────────────────
    run = args.run or infer_run_number(args.root_file)
    if run is None:
        sys.exit("ERROR: Cannot infer run number from filename. "
                 "Please supply --run XXXXXX explicitly.")
    run = run.lstrip("0") or "0"          # strip leading zeros for display

    # ── derive plot name from the last component of the path ────────────────
    # Strip cycle number and take the last '/'-separated token
    obj_clean = re.sub(r";[0-9]+$", "", args.object_path)
    plot_name = obj_clean.rstrip("/").split("/")[-1]

    # ── output paths ────────────────────────────────────────────────────────
    out_dir = os.path.join(args.outdir, plot_name)
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{plot_name}_run{run.zfill(6)}.png")

    # ── open file and fetch object ───────────────────────────────────────────
    print(f"Opening: {args.root_file}")
    tf = open_file(args.root_file)

    print(f"Fetching: {obj_clean}")
    obj = get_object(tf, obj_clean)
    obj.SetDirectory(0)   # detach from file so it survives tf.Close()
    tf.Close()

    # ── draw ─────────────────────────────────────────────────────────────────
    canvas = ROOT.TCanvas("c1", plot_name, args.width, args.height)
    style_canvas(canvas)

    draw_opt = draw_option(obj)

    # Stat box: name+entries+mean+rms for 1D; entries only for 2D/3D
    if draw_opt in ("COLZ", "BOX"):
        ROOT.gStyle.SetOptStat(10)    # entries only (no name row)
    else:
        ROOT.gStyle.SetOptStat(1111)  # name + entries + mean + rms

    obj.Draw(draw_opt)

    # Axis cosmetics
    obj.GetXaxis().SetTitleSize(0.045)
    obj.GetYaxis().SetTitleSize(0.045)
    obj.GetXaxis().SetLabelSize(0.035)
    obj.GetYaxis().SetLabelSize(0.035)
    if draw_opt == "COLZ":
        obj.GetZaxis().SetTitleSize(0.040)
        obj.GetZaxis().SetLabelSize(0.030)
        # widen right margin so palette + stat box don't overlap
        ROOT.gPad.SetRightMargin(0.18)

    canvas.Update()
    #add_run_label(run, canvas)

    # ── save ──────────────────────────────────────────────────────────────────
    canvas.SaveAs(out_png)
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
