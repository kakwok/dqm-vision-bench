"""
DQM render plugins — PyROOT ports of dmwm/deployment/dqmgui/style/*.cc
Provides pre_draw / post_draw styling hooks for each subsystem.
"""
from __future__ import annotations
import array
import ROOT

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _int_palette(colors: list[int]) -> None:
    arr = array.array('i', colors)
    ROOT.gStyle.SetPalette(len(arr), arr)


def _gradient_palette(n: int, stops: list, r: list, g: list, b: list) -> None:
    s  = array.array('d', stops)
    rv = array.array('d', r)
    gv = array.array('d', g)
    bv = array.array('d', b)
    ROOT.TColor.CreateGradientColorTable(len(stops), s, rv, gv, bv, n)
    ROOT.gStyle.SetNumberContours(n)


def report_summary_map_palette(obj) -> None:
    """Approximate port of dqm::utils::reportSummaryMapPalette.
    Continuous red→yellow→green gradient used on efficiency/quality summary maps."""
    _gradient_palette(
        100,
        [0.00, 0.12, 0.40, 0.60, 0.88, 1.00],
        [0.85, 1.00, 1.00, 0.80, 0.00, 0.00],
        [0.00, 0.20, 1.00, 1.00, 0.90, 0.80],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    )
    obj.SetMinimum(-1e-15)
    obj.SetMaximum(1.0)


# Lazy-built palettes
_ecal_quality_palette: array.array | None = None
_ecal_tcolor_refs: list = []  # Keep Python refs so ROOT TColor objects aren't GC'd


def _ecal_quality_colors() -> array.array:
    """7-color discrete palette for ECAL quality flag maps (integer values 0–6)."""
    global _ecal_quality_palette, _ecal_tcolor_refs
    if _ecal_quality_palette is not None:
        return _ecal_quality_palette
    rgb = [
        (1.00, 0.00, 0.00),  # 0 → red
        (0.00, 1.00, 0.00),  # 1 → green (GOOD)
        (1.00, 0.96, 0.00),  # 2 → yellow
        (0.50, 0.00, 0.00),  # 3 → dark red
        (0.00, 1.00, 0.00),  # 4 → green
        (0.80, 0.80, 0.00),  # 5 → dark yellow
        (1.00, 1.00, 1.00),  # 6 → white
    ]
    idxs: list[int] = []
    for r, g, b in rgb:
        idx = ROOT.TColor.GetFreeColorIndex()
        tc  = ROOT.TColor(idx, r, g, b)
        ROOT.SetOwnership(tc, False)  # Transfer ownership to ROOT; prevents C++ delete on Python GC
        _ecal_tcolor_refs.append(tc)  # Keep Python ref anyway as a safety net
        idxs.append(idx)
    _ecal_quality_palette = array.array('i', idxs)
    return _ecal_quality_palette


# ---------------------------------------------------------------------------
# ECAL
# ---------------------------------------------------------------------------

def pre_draw_ecal(canvas, obj, path: str) -> str | None:
    cls   = obj.ClassName()
    is_2d = cls.startswith(("TH2", "TProfile2D"))
    name  = path.lower()

    ROOT.TGaxis.SetMaxDigits(3)

    if is_2d:
        ROOT.gStyle.SetOptStat(0)
        ROOT.gPad.SetLogy(False)
        ROOT.gPad.SetRightMargin(0.15)
        ROOT.gPad.SetGrid(True, True)
        obj.SetStats(False)
        obj.SetContour(50)
        draw_opt = "COLZ"

        if any(k in name for k in ("quality", "global summary", "status summary")):
            obj.SetMinimum(-1e-8)
            obj.SetMaximum(7.0)
            p = _ecal_quality_colors()
            ROOT.gStyle.SetPalette(len(p), p)
            ROOT.gStyle.SetNumberContours(7)  # override style_canvas()'s 255
            obj.SetContour(7)                  # one bin per quality flag 0-6
            draw_opt = "COL"
        elif "summarymap" in name:
            report_summary_map_palette(obj)
        elif "masking" in name:
            obj.SetMinimum(0)
            obj.SetMaximum(1.0)
            ROOT.gStyle.SetPalette(2)
            draw_opt = "COL"
        elif "timing" in name and "vs" not in name and "bx" not in name:
            obj.GetZaxis().SetRangeUser(-5., 5.)
            _gradient_palette(
                50,
                [0.00, 0.42, 0.50, 0.58, 1.00],
                [0.142, 0.740, 0.500, 1.000, 0.650],
                [0.000, 0.900, 1.000, 0.900, 0.000],
                [0.850, 1.000, 0.500, 0.610, 0.130],
            )
        elif "pedestal" in name:
            _gradient_palette(
                30,
                [0.00, 0.33, 0.50, 0.67, 1.00],
                [0.142, 0.740, 0.500, 1.000, 0.650],
                [0.000, 0.900, 1.000, 0.900, 0.000],
                [0.850, 1.000, 0.500, 0.610, 0.130],
            )
            zmax = 260. if "EcalEndcap" in path else 240.
            obj.GetZaxis().SetRangeUser(160., zmax)
        else:
            obj.SetMinimum(0.)
            _gradient_palette(50, [0.0, 1.0],
                              [0.70, 0.00], [0.90, 0.10], [0.90, 0.90])

        return draw_opt

    if cls.startswith("TProfile"):
        ROOT.gStyle.SetOptStat("ourme")
        obj.SetMarkerStyle(8)
        return "P"

    if cls.startswith("TH1"):
        ROOT.gStyle.SetOptStat("ourme")
        return "HIST"

    return None


def post_draw_ecal(canvas, obj, path: str) -> None:
    if "EcalBarrel" not in path:
        return
    name = path.lower()
    if not any(k in name for k in ("quality", "global summary", "status summary", "summarymap")):
        return
    cls = obj.ClassName()
    if not cls.startswith(("TH2", "TProfile2D")):
        return

    ymin = obj.GetYaxis().GetXmin()
    ymax = obj.GetYaxis().GetXmax()
    xmin = obj.GetXaxis().GetXmin()
    xmax = obj.GetXaxis().GetXmax()

    # 17 vertical lines marking the 18 SM phi boundaries (every 20 iphi)
    line = ROOT.TLine()
    line.SetLineColor(ROOT.kBlack)
    line.SetLineWidth(1)
    for i in range(1, 18):
        line.DrawLine(i * 20, ymin, i * 20, ymax)
    # Horizontal separator between EB+ and EB- (ieta = 0)
    line.DrawLine(xmin, 0, xmax, 0)

    # SM number labels: "+NN" on EB+ side, "-NN" on EB- side
    t = ROOT.TLatex()
    t.SetTextSize(0.035)
    t.SetTextAlign(22)  # centre-centre
    y_pos  =  ymax * 0.65  # ~+55 for the standard -85/+85 axis
    y_neg  =  ymin * 0.65  # ~-55
    for i in range(18):
        iphi_c = 10 + i * 20  # SM phi centres at 10, 30, 50, ..., 350
        t.DrawLatex(iphi_c, y_pos, f"+{i+1:02d}")
        t.DrawLatex(iphi_c, y_neg, f"-{i+1:02d}")


# ---------------------------------------------------------------------------
# HCAL
# ---------------------------------------------------------------------------

_HCAL_STATUS_COLORS  = [ROOT.kWhite, ROOT.kGray, ROOT.kWhite,
                         ROOT.kGreen, ROOT.kYellow, ROOT.kRed, ROOT.kBlack]
_HCAL_CONTOUR_BOUNDS = array.array('d', [0, 1, 2, 3, 4, 5, 6, 7])


def pre_draw_hcal(canvas, obj, path: str) -> str | None:
    ROOT.gStyle.SetOptStat(False)
    ROOT.TGaxis.SetMaxDigits(4)

    cls  = obj.ClassName()
    name = path

    if cls.startswith(("TH2", "TProfile2D")):
        obj.SetStats(False)
        ROOT.gStyle.SetPalette(1)
        obj.GetZaxis().SetRangeUser(obj.GetMinimum(), obj.GetMaximum())
        draw_opt = "COLZ"

        if "Summary" in name:
            draw_opt = "COL"
            if "runSummary" not in name:
                ROOT.gPad.SetGrid(True, True)
            _int_palette(_HCAL_STATUS_COLORS)
            n = len(_HCAL_CONTOUR_BOUNDS) - 1
            obj.SetContour(n, _HCAL_CONTOUR_BOUNDS)

        return draw_opt

    if cls.startswith("TProfile"):
        obj.SetMarkerStyle(20)
        return "P"

    return None


# ---------------------------------------------------------------------------
# L1T (covers L1TStage2*, uGMT, uGT)
# ---------------------------------------------------------------------------

def pre_draw_l1t(canvas, obj, path: str) -> str | None:
    cls  = obj.ClassName()
    name = path
    base = name.split("/")[-1]

    if cls.startswith(("TH2", "TProfile2D")):
        ROOT.gStyle.SetPalette(1)
        ROOT.gStyle.SetNumberContours(100)
        ROOT.gStyle.SetOptStat(11)
        draw_opt = "COLZ"

        if "reportSummaryMap" in name:
            obj.SetStats(False)
            report_summary_map_palette(obj)
            obj.GetXaxis().SetLabelSize(0.1)
            obj.GetXaxis().CenterLabels()
            obj.GetYaxis().CenterLabels()
        elif base.startswith("Ratio_") or "/Ratio_" in name:
            obj.GetZaxis().SetRangeUser(0., 1.)
        elif any(k in base for k in ("ugmtMuonBXvs", "muColl1TrkAddr", "muColl2TrkAddr")):
            draw_opt = "TEXT COLZ"
        elif base.startswith("rescaled_") or "_rescaled_" in base:
            draw_opt = "COLZ TEXT"

        return draw_opt

    if cls.startswith("TH1"):
        ROOT.gStyle.SetOptStat(11)
        if any(k in base for k in ("hwPt", "MuonPt", "muHwPt",
                                    "ugmtBMTFhwPt", "ugmtOMTFhwPt", "ugmtEMTFhwPt")):
            ROOT.gPad.SetLogy(True)
        if "mismatchRatio" in base.lower():
            obj.GetYaxis().SetRangeUser(0., 1.05)
        if any(base.startswith(k) for k in ("errorSummary", "mismatchRatio",
                                              "summary", "zeroSuppVal")):
            return "TEXT HIST"
        if base.startswith("Ratio_"):
            obj.GetYaxis().SetRangeUser(0., 1.05)

    return None


# ---------------------------------------------------------------------------
# CSC
# ---------------------------------------------------------------------------

def pre_draw_csc(canvas, obj, path: str) -> str | None:
    ROOT.gStyle.SetPalette(1, 0)
    obj.SetFillColor(45)

    cls  = obj.ClassName()
    name = path

    if cls.startswith("TH2"):
        if any(k in name for k in (
            "Physics_EMU", "Physics_ME1", "Physics_ME2",
            "Physics_ME3", "Physics_ME4",
        )):
            obj.SetStats(False)
            ROOT.gStyle.SetOptStat("e")
            return "COL"

        if any(k in name for k in (
            "CSC_STATS_", "CSC_Reporting", "CSC_Unpacked", "CSC_wo_",
            "CSC_Format_", "CSC_standby", "CSC_L1A_", "CSC_DMB_",
            "CSC_ALCT0_", "CSC_CLCT0_", "DMB_Reporting", "DMB_Unpacked",
            "reportSummaryMap",
        )):
            obj.SetStats(False)
            ROOT.gStyle.SetOptStat("e")
            ROOT.gPad.SetGridx()
            ROOT.gPad.SetGridy()
            if "Fract" in name:
                obj.SetMaximum(1.0)
            return "COLZ"

        if "FEDBufferSize" in name:
            obj.SetStats(False)
            ROOT.gStyle.SetOptStat("e")
            return "COLZ"

    elif cls.startswith("TH1"):
        if any(k in name for k in ("FEDEntries", "FEDFatal", "FEDNonFatal",
                                    "FED_DDU_L1A_mismatch")):
            obj.SetStats(False)
            obj.SetMinimum(0.)
            if "fract" in name.lower():
                obj.SetMaximum(1.)
            return "BAR1 TEXT"

        if any(k in name for k in ("FEDTotalEventSize", "FEDTotalSize", "DCCBufferSize")):
            ROOT.gStyle.SetOptStat("emro")
            obj.SetFillColor(45)
            ROOT.gPad.SetLogy()

    return None


# ---------------------------------------------------------------------------
# RPC
# ---------------------------------------------------------------------------

def pre_draw_rpc(canvas, obj, path: str) -> str | None:
    ROOT.gStyle.SetCanvasBorderMode(0)
    ROOT.gStyle.SetCanvasColor(ROOT.kWhite)
    ROOT.gStyle.SetPadBorderMode(0)
    ROOT.gStyle.SetPadBorderSize(0)
    ROOT.gStyle.SetPalette(1)

    cls  = obj.ClassName()
    name = path

    if cls.startswith("TH2"):
        obj.SetStats(False)
        ROOT.gStyle.SetOptStat(10)
        obj.GetXaxis().SetNdivisions(-510)
        obj.GetYaxis().SetNdivisions(-510)
        obj.GetXaxis().SetLabelSize(0.05)
        obj.GetYaxis().SetLabelSize(0.045)
        obj.GetXaxis().CenterLabels()
        obj.GetYaxis().CenterLabels()
        ROOT.gPad.SetGridx()
        ROOT.gPad.SetGridy()
        draw_opt = "COLZ"

        if "reportSummaryMap" in name:
            report_summary_map_palette(obj)
            ROOT.gStyle.SetPaintTextFormat(".3f")
            draw_opt = "COLZ TEXT"
        elif "noisySummaryMap" in name:
            _int_palette([400, 807, 807, 632, 632])
            obj.SetMinimum(0.0)
            obj.SetMaximum(3.0)
            ROOT.gStyle.SetPaintTextFormat(".3f")
            draw_opt = "COLZ TEXT"
        elif "SummaryMap" in name:
            report_summary_map_palette(obj)
        elif "Occupancy_for_" in name or "SummaryHistograms" in name:
            draw_opt = "COLZ TEXT"
        elif "RPC_chamberEff_Barrel_W" in name:
            report_summary_map_palette(obj)
            obj.GetXaxis().SetNdivisions(-21, True)
            obj.GetYaxis().SetNdivisions(-12, True)
            obj.SetMinimum(0.)
            ROOT.gStyle.SetPaintTextFormat("1.2f")
            ROOT.gPad.SetGrid(True, True)
            obj.SetMarkerSize(2)
            draw_opt = "TEXT COLZ"
        elif "RPC_chamberEff_Endcap_Sta" in name:
            report_summary_map_palette(obj)
            obj.GetXaxis().SetNdivisions(-6, True)
            obj.GetYaxis().SetNdivisions(36, True)
            obj.SetMinimum(0.)
            ROOT.gStyle.SetPaintTextFormat("1.2f")
            ROOT.gPad.SetGrid(True, True)
            obj.SetMarkerSize(2)
            draw_opt = "TEXT45 COLZ"
        elif "ClusterSizeMean" in name:
            _int_palette([400, 416, 416, 807, 632])
            obj.SetMinimum(0.0)
            obj.SetMaximum(5.0)
        elif "AsymmetryLeftRight" in name:
            _int_palette([416, 416, 416, 400, 400, 807, 807, 632, 632, 632])
            obj.SetMinimum(-1e-15)
            obj.SetMaximum(1.0)
        elif "DeadChannelFraction" in name:
            _int_palette([416, 416, 416, 400, 400, 400, 807, 807, 632, 632])
            obj.SetMinimum(-1e-15)
            obj.SetMaximum(1.0)
        elif "rpcHVStatus" in name:
            _int_palette([632, 416])
            obj.SetMinimum(-0.5)
            obj.SetMaximum(1.5)

        return draw_opt

    if cls.startswith("TH1"):
        if "RPC_chamberEff_" in name:
            obj.SetMinimum(0.)
            ROOT.gPad.SetGrid(True, True)

    return None


def post_draw_rpc(canvas, obj, path: str) -> None:
    if "SummaryMap" not in path:
        return
    line = ROOT.TLine()
    line.SetLineWidth(1)
    for x1, y1, x2, y2 in [
        (-3.5,  0.5, -3.5,  6.5),
        (-7.5,  6.5, -3.5,  6.5),
        (-2.5,  0.5, -2.5, 12.5),
        ( 2.5,  0.5,  2.5, 12.5),
        (-2.5, 12.5,  2.5, 12.5),
        ( 3.5,  0.5,  3.5,  6.5),
        ( 3.5,  6.5,  7.5,  6.5),
        ( 7.5,  0.5,  7.5,  6.5),
    ]:
        line.DrawLine(x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# DT
# ---------------------------------------------------------------------------

def pre_draw_dt(canvas, obj, path: str) -> str | None:
    cls  = obj.ClassName()
    name = path
    ROOT.gStyle.SetPalette(1)

    if cls.startswith(("TH2", "TProfile2D")):
        ROOT.gStyle.SetOptStat(0)
        obj.SetStats(False)
        obj.GetXaxis().SetLabelSize(0.05)
        obj.GetYaxis().SetLabelSize(0.05)
        draw_opt = "COLZ"

        def _wheel_axes():
            obj.GetXaxis().SetNdivisions(13, True)
            obj.GetYaxis().SetNdivisions(5, True)
            obj.GetXaxis().CenterLabels()
            obj.GetYaxis().CenterLabels()
            ROOT.gPad.SetGrid(True, True)
            canvas.SetBottomMargin(0.1)
            canvas.SetLeftMargin(0.12)
            canvas.SetRightMargin(0.12)

        def _global_axes():
            obj.GetXaxis().SetNdivisions(13, True)
            obj.GetYaxis().SetNdivisions(6, True)
            obj.GetXaxis().CenterLabels()
            obj.GetYaxis().CenterLabels()
            ROOT.gPad.SetGrid(True, True)

        if "reportSummaryMap" in name:
            _int_palette([632, 810, 800, 400, 416])
            _global_axes()

        elif "CertificationSummaryMap" in name or "DAQSummaryMap" in name:
            report_summary_map_palette(obj)
            _global_axes()

        elif "SegmentGlbSummary" in name:
            _int_palette([632, 810, 800, 400, 416])
            _global_axes()
            canvas.SetBottomMargin(0.1)
            canvas.SetLeftMargin(0.12)
            canvas.SetRightMargin(0.12)
            obj.SetMinimum(0.)
            obj.SetMaximum(1.25)

        elif "EfficiencyGlbSummary" in name:
            _int_palette([632, 628, 810, 807, 797, 800, 400, 406, 407, 416])
            _global_axes()
            canvas.SetBottomMargin(0.1)
            canvas.SetLeftMargin(0.12)
            canvas.SetRightMargin(0.12)
            obj.SetMinimum(0.)
            obj.SetMaximum(1.0)

        elif "GlbSummary" in name or "DataIntegritySummary" in name:
            report_summary_map_palette(obj)
            _global_axes()
            obj.SetMarkerSize(2)

        elif "OccupancySummary" in name:
            _int_palette([416, 400, 800, 625, 632])
            _wheel_axes()
            obj.SetMinimum(-1e-8)
            obj.SetMaximum(5.0)

        elif "NoiseSummary" in name:
            _global_axes()
            obj.SetMaximum(20 if "_W" in name else 50)

        elif "DataIntegrityTDCSummary" in name:
            _int_palette([416, 594, 632])
            _global_axes()
            canvas.SetBottomMargin(0.1)
            canvas.SetLeftMargin(0.15)
            canvas.SetRightMargin(0.12)
            obj.SetMinimum(-1e-8)
            obj.SetMaximum(3.0)

        elif "SynchNoiseSummary" in name:
            _int_palette([416, 632])
            _global_axes()
            canvas.SetBottomMargin(0.1)
            canvas.SetLeftMargin(0.15)
            canvas.SetRightMargin(0.12)
            obj.SetMinimum(-1e-5)
            obj.SetMaximum(2.0)

        elif "EfficiencyMap" in name:
            obj.SetMinimum(0.0)
            obj.SetMaximum(1.0)
            obj.GetXaxis().SetNdivisions(15, True)
            obj.GetYaxis().SetNdivisions(5, True)
            obj.GetXaxis().CenterLabels()
            obj.GetYaxis().CenterLabels()
            ROOT.gPad.SetGrid(True, True)

        elif "BXDiff" in name:
            _wheel_axes()
            obj.SetMinimum(0.)
            obj.SetMaximum(20.)
            obj.SetMarkerSize(1.5)
            ROOT.gStyle.SetPaintTextFormat("2.1f")
            draw_opt = "TEXT COLZ"

        elif "ROChannel" in name or "TimeBoxSummary" in name:
            report_summary_map_palette(obj)
            _global_axes()
            obj.SetMarkerSize(2)
            if "ROChannel" in name:
                obj.SetMaximum(1.0)

        elif "segmentSummary" in name:
            _int_palette([416, 400, 632])
            _wheel_axes() if "_W" in name else _global_axes()
            obj.SetMinimum(-1e-8)
            obj.SetMaximum(3.0)

        elif "DT_chamberEff_W" in name:
            report_summary_map_palette(obj)
            obj.GetXaxis().SetNdivisions(-14, True)
            obj.GetYaxis().SetNdivisions(-4, True)
            obj.GetXaxis().CenterLabels()
            obj.GetYaxis().CenterLabels()
            obj.SetMinimum(0.)
            obj.SetMarkerSize(2)
            ROOT.gStyle.SetPaintTextFormat("1.2f")
            ROOT.gPad.SetGrid(True, True)
            draw_opt = "TEXT COLZ"

        return draw_opt

    if cls.startswith("TH1"):
        obj.SetStats(False)
        obj.GetXaxis().SetLabelSize(0.05)
        obj.GetYaxis().SetLabelSize(0.05)
        base = name.split("/")[-1]

        if "MeanDistr" in base or "SigmaDistr" in base:
            ROOT.gStyle.SetOptStat(1111111)
            obj.SetStats(True)
        elif "EventLength" in base or "ROSEventLength" in base:
            ROOT.gStyle.SetOptStat(1111111)
            obj.SetStats(True)
            if obj.GetEntries() > 0:
                ROOT.gPad.SetLogy(True)
        elif "NoiseRateSummary" in base:
            if obj.GetEntries() > 0:
                ROOT.gPad.SetLogy(True)
        elif "FEDIntegrity" in base:
            obj.GetXaxis().SetNdivisions(11, True)
            obj.GetXaxis().CenterLabels()
            ROOT.gPad.SetGrid(True, False)

    return None


# ---------------------------------------------------------------------------
# SiStrip
# ---------------------------------------------------------------------------

def pre_draw_sistrip(canvas, obj, path: str) -> str | None:
    cls  = obj.ClassName()
    name = path

    if cls.startswith("TH2"):
        for ax in (obj.GetXaxis(), obj.GetYaxis()):
            ax.SetTitleOffset(0.7)
            ax.SetTitleSize(0.05)
            ax.SetLabelSize(0.04)
        draw_opt = "COLZ"

        if "TkHMap" in name:
            obj.SetStats(False)
            ROOT.gStyle.SetPalette(1, 0)
            if "FractionOfBadChannels" in name:
                obj.SetMinimum(0.0001)
                obj.SetMaximum(1.0)
        elif any(k in name for k in ("reportSummaryMap", "detFractionReportMap",
                                      "sToNReportMap", "DataPresentInLastLS")):
            obj.SetStats(False)
            report_summary_map_palette(obj)
            draw_opt = "COLZ TEXT"
        elif "SummaryOfCabling" in name:
            obj.SetStats(False)
            draw_opt = "TEXT"
        elif "ClusterWidths_vs_Amplitudes" in name or "TrackEtaPhi" in name:
            obj.SetStats(False)
            ROOT.gStyle.SetPalette(1, 0)
            ROOT.gPad.SetLogz(True)
        elif "FEDErrorsVsId" in name or "ApvIdVsFedId" in name:
            obj.SetStats(False)
            ROOT.gStyle.SetPalette(1, 0)
            ROOT.gPad.SetGrid()
            ROOT.gPad.SetLeftMargin(0.2)
            obj.GetYaxis().SetTitle("")
        elif "ErrorsVsModules" in name:
            ROOT.gStyle.SetPalette(1, 0)
            ROOT.gStyle.SetOptStat(10)
            ROOT.gPad.SetLeftMargin(0.2)
            ROOT.gPad.SetBottomMargin(0.2)
        elif "DataPresentInLS" in name:
            obj.SetStats(False)
            report_summary_map_palette(obj)
        elif "StripClusVsPixClus" in name or "SeedsVsClusters" in name:
            obj.SetStats(False)
            ROOT.gStyle.SetPalette(1, 0)
        else:
            ROOT.gStyle.SetPalette(1, 0)
            obj.SetStats(False)

        return draw_opt

    if cls.startswith("TH1"):
        ROOT.gStyle.SetOptStat(1110)
        base = name.split("/")[-1]
        if any(k in base for k in ("NumberOfTracks_", "Chi2oNDF_", "TrackPt_")):
            if obj.GetEntries() > 10:
                ROOT.gPad.SetLogy(True)
            ROOT.gPad.SetGridy()

    return None


def post_draw_sistrip(canvas, obj, path: str) -> None:
    cls  = obj.ClassName()
    name = path

    if cls.startswith("TH1") and any(
        k in name for k in ("/ReadoutView/FE/VsId/", "/ReadoutView/FED/VsId/",
                            "/ReadoutView/Fiber/VsId/", "/ReadoutView/DataPresent")
    ):
        ymax = max(obj.GetMaximum() * 1.1, 1.0)
        for x, col, width in [(134.0, 922, 1), (164.0, 922, 2),
                               (260.0, 922, 2), (356.0, 922, 2)]:
            line = ROOT.TLine()
            line.SetLineColor(col)
            line.SetLineWidth(width)
            line.SetLineStyle(7)
            line.DrawLine(x, 0, x, ymax)

        t = ROOT.TLatex()
        t.SetTextSize(0.04)
        t.SetTextColor(15)
        for text, ndcx, ndcy in [("TIB/D", 0.18, 0.91), ("TEC-", 0.38, 0.91),
                                   ("TEC+", 0.55, 0.91), ("TOB",  0.72, 0.91)]:
            t.DrawLatexNDC(ndcx, ndcy, text)


# ---------------------------------------------------------------------------
# PixelPhase1
# ---------------------------------------------------------------------------

def pre_draw_pixelphase1(canvas, obj, path: str) -> str | None:
    if "OnlineBlock" not in path:
        return None
    cls = obj.ClassName()
    if cls.startswith("TH2"):
        obj.SetStats(True)
        ROOT.gStyle.SetOptStat(111)
        # Original plugin renders overlaid TH1 slices per time block.
        # We fall back to COLZ to show the 2D distribution.
        return "COLZ"
    return None


# ---------------------------------------------------------------------------
# Plugin routing
# ---------------------------------------------------------------------------

_PLUGINS: dict[str, tuple] = {
    'EcalBarrel':   (pre_draw_ecal,        post_draw_ecal),
    'EcalEndcap':   (pre_draw_ecal,        post_draw_ecal),
    'Hcal':         (pre_draw_hcal,        None),
    'L1T':          (pre_draw_l1t,         None),
    'CSC':          (pre_draw_csc,         None),
    'RPC':          (pre_draw_rpc,         post_draw_rpc),
    'DT':           (pre_draw_dt,          None),
    'SiStrip':      (pre_draw_sistrip,     post_draw_sistrip),
    'PixelPhase1':  (pre_draw_pixelphase1, None),
}

_SUBSYSTEM_MARKERS = [
    ('EcalBarrel',  '/EcalBarrel/'),
    ('EcalEndcap',  '/EcalEndcap/'),
    ('Hcal',        '/Hcal/'),
    ('Hcal',        '/HcalCalib/'),
    ('Hcal',        '/Hcal2/'),
    ('Hcal',        '/HcalReco/'),
    ('L1T',         '/L1T/'),
    ('L1T',         '/L1TEMU/'),
    ('CSC',         '/CSC/'),
    ('RPC',         '/RPC/'),
    ('DT',          '/DT/'),
    ('SiStrip',     '/SiStrip/'),
    ('SiStrip',     '/Tracking/'),
    ('PixelPhase1', '/PixelPhase1/'),
]


def get_plugin(root_path: str) -> tuple | None:
    """
    Return (pre_draw_fn, post_draw_fn) for the subsystem in root_path.
    Either function in the tuple may be None. Returns None if no plugin matches.
    """
    if not root_path:
        return None
    for key, marker in _SUBSYSTEM_MARKERS:
        if marker in root_path:
            return _PLUGINS.get(key)
    return None
