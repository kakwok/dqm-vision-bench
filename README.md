# dqm-vision-bench
Benchmarking vision-language models on CMS DQM plots

## Shift Workspace Plot Index
 The CMS Online DQM GUI organises its **"00 Shift"** workspace through Python layout
files (`shift_*_layout.py`) that map human-readable titles to ROOT histogram paths
inside the DQM files.  These files are maintained in `/data/srv/current/config/dqmgui/layouts` in the P5 online DQM machine 
 
`parse_shift_layouts.py` ingests those layout files and produces a structured index
(`shift_layouts.json`) that the benchmarking pipeline uses to look up the correct
histogram path and description for each plot it queries.


**Usage**
 
```bash
python3 parse_shift_layouts.py ./shift_*_layout.py -o shift_layouts.json
```
 
| Argument | Description |
|---|---|
| `patterns` | One or more glob patterns matching layout files |
| `-o / --output` | Output JSON file (default: `shift_layouts.json`) |
| `--indent` | JSON indent level (default: `2`) |
 
### `shift_layouts.json`
 
A JSON array with one entry per layout file.  Each entry groups all plots under
their subsystem and source file:
 
```json
[
  {
    "subsystem": "BeamMonitor",
    "source_file": "shift_beam_layout.py",
    "plots": [
      {
        "title": "00 - BeamMonitorHLT ReportSummary",
        "path": "BeamMonitorHLT/EventInfo/reportSummaryMap",
        "description": "BeamSpot summary map"
      },
      ...
    ]
  },
  ...
]
