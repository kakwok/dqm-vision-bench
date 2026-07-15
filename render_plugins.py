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


# ---------------------------------------------------------------------------
# ECAL endcap geometry constants
# ---------------------------------------------------------------------------

# SM label positions on the 10×10 label grid (0–100 data coordinates).
# Derived from iEESM[] in EcalRenderPlugin.cc: bin (ix,iy) → center (ix*10-5, iy*10-5).
_EEP_SM_LABELS = [          # (sm_number, x_center, y_center) for EE+
    (1, 35, 85), (2, 15, 65), (3, 15, 45), (4, 25, 25), (5, 55, 15),
    (6, 75, 25), (7, 85, 45), (8, 85, 65), (9, 65, 85),
]

# DCC sector boundary horizontal lines for one EE disk: [y, x1, x2]
_EE_DCC_H = [
    [0,  40,  60], [3,  35,  40], [3,  60,  65], [5,  25,  35], [5,  65,  75],
    [8,  20,  25], [8,  75,  80], [13, 15,  20], [13, 80,  85], [15, 13,  15],
    [15, 35,  40], [15, 60,  65], [15, 85,  87], [20, 8,   13], [20, 87,  92],
    [25, 5,   10], [25, 90,  95], [30, 10,  20], [30, 40,  45], [30, 55,  60],
    [30, 80,  90], [35, 3,   5],  [35, 20,  30], [35, 70,  80], [35, 95,  97],
    [39, 45,  55], [40, 0,   3],  [40, 30,  35], [40, 43,  45], [40, 55,  57],
    [40, 65,  70], [40, 97, 100], [41, 42,  43], [41, 57,  58], [42, 41,  42],
    [42, 58,  59], [43, 40,  41], [43, 59,  60], [45, 35,  40], [45, 60,  65],
    [50, 35,  39], [50, 61,  65], [55, 10,  35], [55, 39,  40], [55, 60,  61],
    [55, 65,  90], [57, 40,  41], [57, 59,  60], [58, 41,  42], [58, 58,  59],
    [59, 42,  43], [59, 57,  58], [60, 0,   10], [60, 40,  45], [60, 55,  60],
    [60, 90, 100], [61, 45,  55], [65, 3,   5],  [65, 35,  40], [65, 60,  65],
    [65, 95,  97], [70, 30,  35], [70, 65,  70], [75, 5,   8],  [75, 25,  30],
    [75, 70,  75], [75, 92,  95], [80, 8,   13], [80, 87,  92], [85, 13,  15],
    [85, 20,  25], [85, 75,  80], [85, 85,  87], [87, 15,  20], [87, 80,  85],
    [92, 20,  25], [92, 75,  80], [95, 25,  35], [95, 65,  75], [97, 35,  40],
    [97, 60,  65], [100, 40, 60],
]

# TCC sector boundary horizontal lines for one EE disk (EE+ coords): [y, x1, x2]
# EE+ draws at (x1,y)→(x2,y); EE- mirrors as (100-x2,y)→(100-x1,y). Style 2, width 2.
_EE_TCC_H = [
    [6, 33, 34], [6, 66, 67], [9, 34, 35], [9, 65, 66], [12, 35, 37], [12, 63, 65],
    [13, 36, 37], [13, 49, 51], [13, 63, 64], [14, 36, 37], [14, 46, 49], [14, 51, 54],
    [14, 63, 64], [15, 20, 21], [15, 37, 46], [15, 54, 63], [15, 79, 80], [16, 37, 38],
    [16, 62, 63], [17, 21, 23], [17, 35, 38], [17, 62, 65], [17, 77, 79], [18, 23, 25],
    [18, 34, 35], [18, 65, 66], [18, 75, 77], [19, 31, 34], [19, 38, 39], [19, 61, 62],
    [19, 66, 69], [20, 29, 31], [20, 39, 40], [20, 60, 61], [20, 69, 71], [21, 25, 27],
    [21, 28, 29], [21, 71, 72], [21, 73, 75], [22, 27, 28], [22, 72, 73], [23, 26, 27],
    [23, 73, 74], [24, 25, 26], [24, 27, 29], [24, 71, 73], [24, 74, 75], [25, 8, 10],
    [25, 24, 25], [25, 29, 30], [25, 40, 41], [25, 59, 60], [25, 70, 71], [25, 75, 76],
    [25, 90, 92], [26, 23, 24], [26, 41, 42], [26, 58, 59], [26, 76, 77], [27, 10, 11],
    [27, 22, 23], [27, 30, 31], [27, 69, 70], [27, 77, 78], [27, 89, 90], [28, 11, 14],
    [28, 21, 22], [28, 31, 32], [28, 68, 69], [28, 78, 79], [28, 86, 89], [29, 14, 15],
    [29, 20, 21], [29, 42, 43], [29, 57, 58], [29, 79, 80], [29, 85, 86], [30, 15, 16],
    [30, 32, 35], [30, 65, 68], [30, 84, 85], [31, 16, 18], [31, 19, 20], [31, 80, 81],
    [31, 82, 84], [32, 18, 19], [32, 20, 21], [32, 43, 44], [32, 56, 57], [32, 79, 80],
    [32, 81, 82], [33, 21, 22], [33, 35, 37], [33, 44, 45], [33, 55, 56], [33, 63, 65],
    [33, 78, 79], [34, 18, 19], [34, 37, 38], [34, 62, 63], [34, 81, 82], [35, 17, 18],
    [35, 22, 25], [35, 38, 40], [35, 75, 78], [35, 82, 83], [36, 25, 27], [36, 73, 75],
    [37, 16, 17], [37, 27, 28], [37, 61, 62], [37, 72, 73], [37, 83, 84], [38, 15, 16],
    [38, 28, 30], [38, 60, 61], [38, 70, 72], [38, 84, 85], [39, 45, 55], [40, 30, 35],
    [40, 40, 42], [40, 43, 45], [40, 55, 60], [40, 65, 70], [41, 0, 5], [41, 42, 43],
    [41, 57, 58], [41, 95, 100], [42, 5, 11], [42, 35, 37], [42, 41, 42], [42, 58, 59],
    [42, 63, 65], [42, 89, 95], [43, 11, 14], [43, 15, 17], [43, 39, 41], [43, 59, 61],
    [43, 83, 85], [43, 86, 89], [44, 14, 15], [44, 17, 18], [44, 37, 39], [44, 61, 63],
    [44, 82, 83], [44, 85, 86], [45, 18, 25], [45, 29, 30], [45, 39, 40], [45, 60, 61],
    [45, 70, 71], [45, 75, 82], [46, 14, 15], [46, 25, 27], [46, 28, 29], [46, 32, 33],
    [46, 67, 68], [46, 71, 72], [46, 73, 75], [46, 85, 86], [47, 27, 28], [47, 30, 32],
    [47, 33, 34], [47, 35, 37], [47, 63, 65], [47, 66, 67], [47, 68, 70], [47, 72, 73],
    [48, 34, 35], [48, 37, 39], [48, 61, 63], [48, 65, 66], [49, 13, 14], [49, 86, 87],
    [51, 13, 14], [51, 86, 87], [52, 34, 35], [52, 37, 39], [52, 61, 63], [52, 65, 66],
    [53, 27, 28], [53, 30, 32], [53, 33, 34], [53, 35, 37], [53, 63, 65], [53, 66, 67],
    [53, 68, 70], [53, 72, 73], [54, 14, 15], [54, 25, 27], [54, 28, 29], [54, 32, 33],
    [54, 67, 68], [54, 71, 72], [54, 73, 75], [54, 85, 86], [55, 18, 25], [55, 29, 30],
    [55, 70, 71], [55, 75, 82], [56, 14, 15], [56, 17, 18], [56, 37, 39], [56, 61, 63],
    [56, 82, 83], [56, 85, 86], [57, 11, 14], [57, 15, 17], [57, 39, 40], [57, 60, 61],
    [57, 83, 85], [57, 86, 89], [58, 5, 11], [58, 35, 37], [58, 63, 65], [58, 89, 95],
    [59, 0, 5], [59, 95, 100], [60, 0, 3], [60, 30, 35], [60, 40, 43], [60, 58, 60],
    [60, 65, 70], [60, 97, 99], [62, 15, 16], [62, 28, 30], [62, 39, 40], [62, 70, 72],
    [62, 84, 85], [63, 16, 17], [63, 27, 28], [63, 38, 39], [63, 72, 73], [63, 83, 84],
    [64, 25, 27], [64, 73, 75], [65, 3, 5], [65, 17, 18], [65, 22, 25], [65, 60, 62],
    [65, 75, 78], [65, 82, 83], [65, 95, 97], [66, 18, 19], [66, 37, 38], [66, 62, 63],
    [66, 81, 82], [67, 21, 22], [67, 35, 37], [67, 44, 45], [67, 55, 56], [67, 63, 65],
    [67, 78, 79], [68, 18, 19], [68, 20, 21], [68, 43, 44], [68, 56, 57], [68, 79, 80],
    [68, 81, 82], [69, 16, 18], [69, 19, 20], [69, 80, 81], [69, 82, 84], [70, 15, 16],
    [70, 32, 35], [70, 65, 68], [70, 84, 85], [71, 14, 15], [71, 20, 21], [71, 42, 43],
    [71, 57, 58], [71, 79, 80], [71, 85, 86], [72, 11, 14], [72, 21, 22], [72, 31, 32],
    [72, 68, 69], [72, 78, 79], [72, 86, 89], [73, 10, 11], [73, 22, 23], [73, 30, 31],
    [73, 69, 70], [73, 77, 78], [73, 89, 90], [74, 23, 24], [74, 41, 42], [74, 58, 59],
    [74, 76, 77], [75, 5, 10], [75, 24, 25], [75, 29, 30], [75, 40, 41], [75, 59, 60],
    [75, 70, 71], [75, 75, 76], [75, 90, 95], [76, 25, 26], [76, 27, 29], [76, 71, 73],
    [76, 74, 75], [77, 26, 27], [77, 73, 74], [78, 27, 28], [78, 72, 73], [79, 25, 27],
    [79, 28, 29], [79, 71, 72], [79, 73, 75], [80, 8, 13], [80, 29, 31], [80, 39, 40],
    [80, 60, 61], [80, 69, 71], [80, 87, 92], [81, 31, 34], [81, 38, 39], [81, 61, 62],
    [81, 66, 69], [82, 23, 25], [82, 34, 35], [82, 65, 66], [82, 75, 77], [83, 21, 23],
    [83, 35, 38], [83, 62, 65], [83, 77, 79], [84, 37, 38], [84, 62, 63], [85, 13, 15],
    [85, 20, 21], [85, 37, 46], [85, 54, 63], [85, 79, 80], [85, 85, 87], [86, 36, 37],
    [86, 46, 49], [86, 51, 54], [86, 63, 64], [87, 15, 20], [87, 36, 37], [87, 49, 51],
    [87, 63, 64], [87, 80, 85], [88, 35, 37], [88, 63, 65], [91, 34, 35], [91, 65, 66],
    [92, 20, 25], [92, 75, 80], [94, 33, 34], [94, 66, 67], [95, 25, 35], [95, 65, 75],
    [97, 35, 40], [97, 60, 65], [100, 40, 60],
]

# TCC sector boundary vertical lines for one EE disk (EE+ coords): [x, y1, y2]
# EE+ draws at (x,y1)→(x,y2); EE- mirrors as (100-x,y1)→(100-x,y2). Style 2, width 2.
_EE_TCC_V = [
    [5, 41, 42], [5, 58, 59], [10, 25, 27], [10, 73, 75], [11, 27, 28], [11, 42, 43],
    [11, 57, 58], [11, 72, 73], [13, 49, 51], [14, 28, 29], [14, 43, 44], [14, 46, 49],
    [14, 51, 54], [14, 56, 57], [14, 71, 72], [15, 29, 30], [15, 38, 46], [15, 54, 62],
    [15, 70, 71], [16, 30, 31], [16, 37, 38], [16, 62, 63], [16, 69, 70], [17, 35, 37],
    [17, 43, 44], [17, 56, 57], [17, 63, 65], [18, 31, 32], [18, 34, 35], [18, 44, 45],
    [18, 55, 56], [18, 65, 66], [18, 68, 69], [19, 31, 34], [19, 66, 69], [20, 13, 15],
    [20, 29, 32], [20, 68, 71], [20, 85, 87], [21, 15, 17], [21, 28, 29], [21, 32, 33],
    [21, 67, 68], [21, 71, 72], [21, 83, 85], [22, 27, 28], [22, 33, 35], [22, 65, 67],
    [22, 72, 73], [23, 17, 18], [23, 26, 27], [23, 73, 74], [23, 82, 83], [24, 25, 26],
    [24, 74, 75], [25, 18, 21], [25, 24, 25], [25, 35, 36], [25, 45, 46], [25, 54, 55],
    [25, 64, 65], [25, 75, 76], [25, 79, 82], [26, 23, 24], [26, 76, 77], [27, 21, 24],
    [27, 36, 37], [27, 46, 47], [27, 53, 54], [27, 63, 64], [27, 76, 79], [28, 21, 22],
    [28, 37, 38], [28, 46, 47], [28, 53, 54], [28, 62, 63], [28, 78, 79], [29, 20, 21],
    [29, 24, 25], [29, 45, 46], [29, 54, 55], [29, 75, 76], [29, 79, 80], [30, 25, 27],
    [30, 38, 40], [30, 45, 47], [30, 53, 55], [30, 60, 62], [30, 73, 75], [31, 19, 20],
    [31, 27, 28], [31, 72, 73], [31, 80, 81], [32, 28, 30], [32, 46, 47], [32, 53, 54],
    [32, 70, 72], [33, 5, 6], [33, 46, 47], [33, 53, 54], [33, 94, 95], [34, 6, 9],
    [34, 18, 19], [34, 47, 48], [34, 52, 53], [34, 81, 82], [34, 91, 94], [35, 9, 12],
    [35, 17, 18], [35, 30, 33], [35, 40, 42], [35, 47, 48], [35, 52, 53], [35, 58, 60],
    [35, 67, 70], [35, 82, 83], [35, 88, 91], [36, 13, 14], [36, 86, 87], [37, 12, 13],
    [37, 14, 15], [37, 16, 17], [37, 33, 34], [37, 42, 44], [37, 47, 48], [37, 52, 53],
    [37, 56, 58], [37, 66, 67], [37, 83, 84], [37, 85, 86], [37, 87, 88], [38, 15, 16],
    [38, 17, 19], [38, 34, 35], [38, 63, 66], [38, 81, 83], [38, 84, 85], [39, 19, 20],
    [39, 43, 44], [39, 45, 55], [39, 56, 57], [39, 62, 63], [39, 80, 81], [40, 20, 25],
    [40, 35, 40], [40, 43, 45], [40, 55, 57], [40, 60, 62], [40, 75, 80], [41, 25, 26],
    [41, 42, 43], [41, 57, 58], [41, 74, 75], [42, 26, 29], [42, 40, 42], [42, 58, 59],
    [42, 71, 74], [43, 29, 32], [43, 40, 41], [43, 59, 60], [43, 68, 71], [44, 32, 33],
    [44, 67, 68], [45, 33, 40], [45, 60, 67], [46, 14, 15], [46, 85, 86], [49, 13, 14],
    [49, 86, 87], [50, 0, 39], [50, 61, 100], [51, 13, 14], [51, 86, 87], [54, 14, 15],
    [54, 85, 86], [55, 33, 39], [55, 61, 67], [56, 32, 33], [56, 67, 68], [57, 29, 32],
    [57, 68, 71], [58, 26, 29], [58, 59, 60], [58, 71, 74], [59, 25, 26], [59, 74, 75],
    [60, 0, 3], [60, 20, 25], [60, 38, 40], [60, 60, 65], [60, 75, 80], [60, 97, 99],
    [61, 19, 20], [61, 37, 38], [61, 43, 44], [61, 56, 57], [61, 80, 81], [62, 15, 16],
    [62, 17, 19], [62, 34, 37], [62, 65, 66], [62, 81, 83], [62, 84, 85], [63, 12, 13],
    [63, 14, 15], [63, 16, 17], [63, 33, 34], [63, 42, 44], [63, 47, 48], [63, 52, 53],
    [63, 56, 58], [63, 66, 67], [63, 83, 84], [63, 85, 86], [63, 87, 88], [64, 13, 14],
    [64, 86, 87], [65, 3, 5], [65, 9, 12], [65, 17, 18], [65, 30, 33], [65, 40, 42],
    [65, 47, 48], [65, 52, 53], [65, 58, 60], [65, 67, 70], [65, 82, 83], [65, 88, 91],
    [65, 95, 97], [66, 6, 9], [66, 18, 19], [66, 47, 48], [66, 52, 53], [66, 81, 82],
    [66, 91, 94], [67, 5, 6], [67, 46, 47], [67, 53, 54], [67, 94, 95], [68, 28, 30],
    [68, 46, 47], [68, 53, 54], [68, 70, 72], [69, 19, 20], [69, 27, 28], [69, 72, 73],
    [69, 80, 81], [70, 25, 27], [70, 38, 40], [70, 45, 47], [70, 53, 55], [70, 60, 62],
    [70, 73, 75], [71, 20, 21], [71, 24, 25], [71, 45, 46], [71, 54, 55], [71, 75, 76],
    [71, 79, 80], [72, 21, 22], [72, 37, 38], [72, 46, 47], [72, 53, 54], [72, 62, 63],
    [72, 78, 79], [73, 21, 24], [73, 36, 37], [73, 46, 47], [73, 53, 54], [73, 63, 64],
    [73, 76, 79], [74, 23, 24], [74, 76, 77], [75, 5, 8], [75, 18, 21], [75, 24, 25],
    [75, 35, 36], [75, 45, 46], [75, 54, 55], [75, 64, 65], [75, 75, 76], [75, 79, 82],
    [75, 92, 95], [76, 25, 26], [76, 74, 75], [77, 17, 18], [77, 26, 27], [77, 73, 74],
    [77, 82, 83], [78, 27, 28], [78, 33, 35], [78, 65, 67], [78, 72, 73], [79, 15, 17],
    [79, 28, 29], [79, 32, 33], [79, 67, 68], [79, 71, 72], [79, 83, 85], [80, 8, 15],
    [80, 29, 32], [80, 68, 71], [80, 85, 92], [81, 31, 34], [81, 66, 69], [82, 31, 32],
    [82, 34, 35], [82, 44, 45], [82, 55, 56], [82, 65, 66], [82, 68, 69], [83, 35, 37],
    [83, 43, 44], [83, 56, 57], [83, 63, 65], [84, 30, 31], [84, 37, 38], [84, 62, 63],
    [84, 69, 70], [85, 13, 15], [85, 29, 30], [85, 38, 46], [85, 54, 62], [85, 70, 71],
    [85, 85, 87], [86, 28, 29], [86, 43, 44], [86, 46, 49], [86, 51, 54], [86, 56, 57],
    [86, 71, 72], [87, 15, 20], [87, 49, 51], [87, 80, 85], [89, 27, 28], [89, 42, 43],
    [89, 57, 58], [89, 72, 73], [90, 25, 27], [90, 73, 75], [92, 20, 25], [92, 75, 80],
    [95, 25, 35], [95, 41, 42], [95, 58, 59], [95, 65, 75], [97, 35, 40], [97, 60, 65],
    [99, 59, 60], [100, 40, 59],
]

# DCC sector boundary vertical lines for one EE disk: [x, y1, y2]
_EE_DCC_V = [
    [0,   40,  60], [3,   35,  40], [3,   60,  65], [5,   25,  35], [5,   65,  75],
    [8,   20,  25], [8,   75,  80], [10,  25,  30], [10,  55,  60], [13,  15,  20],
    [13,  80,  85], [15,  13,  15], [15,  85,  87], [20,  8,   13], [20,  30,  35],
    [20,  85,  92], [25,  5,   8],  [25,  75,  85], [25,  92,  95], [30,  35,  40],
    [30,  70,  75], [35,  3,   15], [35,  40,  45], [35,  50,  55], [35,  65,  70],
    [35,  95,  97], [39,  45,  55], [40,  0,   3],  [40,  15,  30], [40,  43,  45],
    [40,  55,  57], [40,  60,  65], [40,  97, 100], [41,  42,  43], [41,  57,  58],
    [42,  41,  42], [42,  58,  59], [43,  40,  41], [43,  59,  60], [45,  30,  40],
    [45,  60,  61], [50,  61, 100], [55,  30,  40], [55,  60,  61], [57,  40,  41],
    [57,  59,  60], [58,  41,  42], [58,  58,  59], [59,  42,  43], [59,  57,  58],
    [60,  0,   3],  [60,  15,  30], [60,  43,  45], [60,  55,  57], [60,  60,  65],
    [60,  97, 100], [61,  45,  55], [65,  3,   15], [65,  40,  45], [65,  50,  55],
    [65,  65,  70], [65,  95,  97], [70,  35,  40], [70,  70,  75], [75,  5,   8],
    [75,  75,  85], [75,  92,  95], [80,  8,   13], [80,  30,  35], [80,  85,  92],
    [85,  13,  15], [85,  85,  87], [87,  15,  20], [87,  80,  85], [90,  25,  30],
    [90,  55,  60], [92,  20,  25], [92,  75,  80], [95,  25,  35], [95,  65,  75],
    [97,  35,  40], [97,  60,  65], [100, 40,  60],
]

# ---------------------------------------------------------------------------
# 7-color discrete palette for ECAL quality flag maps (values 0–6).
# Uses ROOT built-in color constants — no TColor allocation, no GC risk.
_ECAL_QUALITY_COLORS = array.array('i', [
    ROOT.kRed,     # 0 → bad / no data
    ROOT.kGreen,   # 1 → GOOD
    ROOT.kYellow,  # 2 → warning
    ROOT.kRed + 2, # 3 → dark red variant
    ROOT.kGreen,   # 4 → GOOD (same as flag 1)
    ROOT.kOrange,  # 5 → orange / dark yellow
    ROOT.kWhite,   # 6 → empty
])


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
            ROOT.gStyle.SetPalette(len(_ECAL_QUALITY_COLORS), _ECAL_QUALITY_COLORS)
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
    name = path.lower()
    if not any(k in name for k in ("quality", "global summary", "status summary", "summarymap")):
        return
    cls = obj.ClassName()
    if not cls.startswith(("TH2", "TProfile2D")):
        return

    xmin = obj.GetXaxis().GetXmin()
    xmax = obj.GetXaxis().GetXmax()
    ymin = obj.GetYaxis().GetXmin()
    ymax = obj.GetYaxis().GetXmax()

    line = ROOT.TLine()
    line.SetLineColor(ROOT.kBlack)
    line.SetLineWidth(1)

    t = ROOT.TLatex()
    t.SetTextAlign(22)  # centre-centre

    if "EcalBarrel" in path:
        # 17 vertical lines → 18 SM phi boundaries (every 20 iphi)
        for i in range(1, 18):
            line.DrawLine(i * 20, ymin, i * 20, ymax)
        # Horizontal EB+/EB- separator at ieta = 0
        line.DrawLine(xmin, 0, xmax, 0)
        # SM labels: "+NN" upper half, "-NN" lower half
        t.SetTextSize(0.035)
        y_pos = ymax * 0.65   # ~+55 for the standard ±85 axis
        y_neg = ymin * 0.65   # ~-55
        for i in range(18):
            iphi_c = 10 + i * 20  # SM phi centres: 10, 30, …, 350
            t.DrawLatex(iphi_c, y_pos, f"+{i+1:02d}")
            t.DrawLatex(iphi_c, y_neg, f"-{i+1:02d}")

    elif "EcalEndcap" in path:
        nx = obj.GetNbinsX()
        t.SetTextSize(0.03)
        leaf = path.split("/")[-1]
        is_tt = "TT " in leaf   # triggers TCC overlay (ECAL 07 TriggerPrimitives)

        def _draw_dcc_lines(x_offset=0, mirror=False):
            for y, x1, x2 in _EE_DCC_H:
                if mirror:
                    line.DrawLine(100 - x2 + x_offset, y, 100 - x1 + x_offset, y)
                else:
                    line.DrawLine(x1 + x_offset, y, x2 + x_offset, y)
            for x, y1, y2 in _EE_DCC_V:
                xd = 100 - x if mirror else x
                line.DrawLine(xd + x_offset, y1, xd + x_offset, y2)

        def _draw_tcc_lines(x_offset=0, mirror=False):
            tcc = ROOT.TLine()
            tcc.SetLineColor(ROOT.kBlack)
            tcc.SetLineStyle(2)
            tcc.SetLineWidth(2)
            for y, x1, x2 in _EE_TCC_H:
                if mirror:
                    tcc.DrawLine(100 - x2 + x_offset, y, 100 - x1 + x_offset, y)
                else:
                    tcc.DrawLine(x1 + x_offset, y, x2 + x_offset, y)
            for x, y1, y2 in _EE_TCC_V:
                xd = 100 - x if mirror else x
                tcc.DrawLine(xd + x_offset, y1, xd + x_offset, y2)

        def _draw_sm_labels(sm_list, x_offset=0):
            for sm_num, sx, sy in sm_list:
                t.DrawLatex(sx + x_offset, sy, str(sm_num))

        if nx == 200:
            # kEE: combined 200×100 — EE- left (0-100), EE+ right (100-200).
            # DCC geometry is the same for both disks (no mirror needed).
            _draw_dcc_lines(x_offset=0, mirror=False)    # EE- side at 0-100
            _draw_dcc_lines(x_offset=100, mirror=False)  # EE+ side at 100-200
            if is_tt:
                # C++ uses eepTCCArray for both halves in the combined case.
                _draw_tcc_lines(x_offset=0, mirror=False)    # left half
                _draw_tcc_lines(x_offset=100, mirror=False)  # right half
            eem_labels = [(-n, sx, sy) for (n, sx, sy) in _EEP_SM_LABELS]
            eep_labels = list(_EEP_SM_LABELS)
            _draw_sm_labels(eem_labels, x_offset=0)
            _draw_sm_labels(eep_labels, x_offset=100)
        else:
            # kEEp / kEEm: single 100×100 disk.
            # DCC geometry is identical for EE+ and EE- in ix-iy coords (no mirror).
            # TCC geometry is mirrored for EE- (eemTCCArray vs eepTCCArray in C++).
            is_minus = ("EE -" in path or "EEM" in path or "/EEMinus" in path)
            _draw_dcc_lines(mirror=False)
            if is_tt:
                _draw_tcc_lines(mirror=is_minus)
            if is_minus:
                labels = [(-n, sx, sy) for (n, sx, sy) in _EEP_SM_LABELS]
            else:
                labels = list(_EEP_SM_LABELS)
            _draw_sm_labels(labels)


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
