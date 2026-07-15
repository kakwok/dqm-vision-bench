# dqm-vision-bench
Benchmarking vision-language models on CMS DQM plots

## Setup

### Notebook output stripping (nbstripout)

Jupyter autosave writes cell outputs and execution counts back into `.ipynb` files,
making them perpetually dirty in git. `nbstripout` strips outputs before staging so
only source changes are committed.

**First time on a new machine (or the repo's first setup) — also creates `.gitattributes`:**

```bash
pip install nbstripout
nbstripout --install --attributes .gitattributes
git add .gitattributes
git add *.ipynb      # apply filter to any notebooks already dirtied by Jupyter
```

Commit `.gitattributes` so every clone picks up the filter mapping automatically.

**On each subsequent clone (after `.gitattributes` has been committed and pulled):**

```bash
pip install nbstripout
nbstripout --install  # registers the filter in .git/config; .gitattributes already present
git add *.ipynb       # fix notebooks already dirtied before the filter was active
```

> **Why two commands?** `.gitattributes` tells git *which* files to filter (committed,
> shared across clones). `nbstripout --install` tells git *how* to run the filter
> (written to `.git/config`, local to each clone). Both are needed on every machine.

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
```

## Producing Images

`produce_images.ipynb` renders DQM histograms from ROOT files as PNG images, organised by plot name and run number. The images are the primary input for VLM evaluation.

**Output layout**
```
images/
    <plotName>/
        <plotName>_run<XXXXXX>.png
```

### Workflow

1. **Set `FILE_PATTERNS`** — XRootD URLs, XRootD globs, or local paths/globs.  `expand_files()` resolves them and prints what was found.

2. **Browse available plots** — Use `shift_layout_helpers` to explore `shift_layouts.json`:
   ```python
   list_subsystems()                        # all subsystem names
   list_plots("L1T")                        # plot titles for L1T
   list_plots("L1T", with_descriptions=True)  # + path and description
   format_plot_config("L1T")               # print copy-pasteable PLOT_CONFIG block
   ```

3. **Configure `PLOT_CONFIG`** — two options:
   - **Hand-pick** individual plots (paste entries from `format_plot_config()` and remove what you don't need):
     ```python
     PLOT_CONFIG = {
         "ecalOccRecdEtWgt": (
             "DQMData/Run {run}/L1T/Run summary/L1TStage2CaloLayer1/ecalOccRecdEtWgt"
         ),
     }
     ```
   - **Use all shift-layout plots** for a subsystem at once:
     ```python
     PLOT_CONFIG = build_plot_config("L1T")
     ```

4. **Run `produce_images()`** — iterates over all resolved files × all configured plots and writes PNGs to `OUTDIR` (default `images/`).

### CLI usage

`dqm_plot.py` can also be run directly from the command line (requires ROOT/CMSSW environment):

```bash
# Produce specific plots for one or more subsystems
python3 dqm_plot.py --batch \
    --files '/eos/cms/store/group/comm_dqm/DQMGUI_data/Run2024/.../*.root' \
    --subsystem L1T --plot 08 09 24 \
    --outdir images
```

Use `--plot` to select individual plot numbers within a subsystem. Omit `--plot` to produce all plots for that subsystem. Run `--list-plots <SUBSYSTEM>` to see available plot numbers.

### Key files

| File | Purpose |
|---|---|
| `produce_images.ipynb` | Driver notebook — all configuration lives here |
| `dqm_plot.py` | Core logic: ROOT rendering, file resolution, path expansion |
| `shift_layout_helpers.py` | Helpers to browse `shift_layouts.json` and generate `PLOT_CONFIG` |
| `shift_layouts.json` | Structured index of all shift-workspace plots (paths + descriptions) |

### `PLOT_CONFIG` format

Keys become the subdirectory name and PNG filename stem. Values are ROOT object path templates with `{run}` replaced at runtime by the zero-padded 6-digit run number:

```
DQMData/Run {run}/{Subsystem}/Run summary/{...path...}/{plotName}
```

The `{run}` value is auto-detected from the filename (`R000XXXXXX`) or overridden via `RUN_OVERRIDE`.

## RAG Details

Retrieval-Augmented Generation (RAG) injects relevant shift instructions into each query so the model has the specific rules for the plot it is evaluating. The knowledge base is sourced from [archi](https://github.com/archi-physics/archi) and exported as `document_chunks.csv` and `documents.csv`.

### Embedding model

Chunks are embedded at ingestion time using `sentence-transformers/all-MiniLM-L6-v2` (via `langchain-huggingface`) with `normalize_embeddings=True`. The same model must be used at query time — mixing models produces vectors in incompatible spaces and makes similarity scores meaningless.

The query embedding is computed once per call (~10 ms on CPU). Chunk embeddings are loaded from the CSV and cached in memory after the first call.

### Query message structure

Each query is assembled in four ordered parts:

1. **Reference image** *(optional)* — a known-good example image shown before the instructions
2. **RAG context** — the retrieved chunk text injected as `Relevant instructions: ...`
3. **Input image** — the plot being evaluated
4. **Prompt** — the text instruction

### Retrieval methods

Three strategies are available via the `method` parameter in `retrieve_chunks`, `query`, and `batch_query_images`:

| `method` | Description | Key parameters |
|---|---|---|
| `"hybrid"` *(default)* | Combines BM25 keyword scores and vector cosine scores with a weighted sum, then returns the top `top_k` results. BM25 reliably catches exact plot names and thresholds; vector search catches paraphrased or conceptually related rules. | `top_k`, `alpha` |
| `"top_k"` | Vector cosine similarity only; returns the `top_k` highest-scoring chunks. | `top_k` |
| `"threshold"` | Vector cosine similarity only; returns all chunks scoring at or above `score_threshold`, capped at `top_k` to prevent context-window overflow. | `score_threshold`, `top_k` |

### Parameter defaults and rationale

| Parameter | Default | Rationale |
|---|---|---|
| `top_k` | `5` | OpenWebUI defaults to 3; 5 provides slightly more context without significantly increasing prompt length. |
| `alpha` | `0.5` | Equal weight between vector and BM25. Tune toward lower values (more BM25) if exact plot-name matching is more important than semantic recall. |
| `score_threshold` | `0.35` | Values below ~0.35 on `all-MiniLM-L6-v2` are typically noise. A threshold of 0.0 would return virtually all chunks. Tune based on observed score distributions for your corpus. |
