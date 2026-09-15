# dqm-vision-bench
Benchmarking vision-language models on CMS DQM plots

## Quick start

### 1. Install the environment

The environment is managed by [pixi](https://pixi.sh) and defined in `pixi.toml`.

```bash
curl -fsSL https://pixi.sh/install.sh | sh   # install pixi (once per user)
source ~/.bashrc                               # reload PATH
pixi install                                   # create env from pixi.toml
```

Then register the kernel (on LPC):

```bash
pixi run python -m ipykernel install --user --name dqm-vision-bench --display-name "dqm-vision-bench"
```

### Environment contents

| Package | Purpose |
|---|---|
| `sentence-transformers` | Embeds query text for vector search (local RAG) |
| `langchain-huggingface` | HuggingFace backend for sentence-transformers |
| `rank-bm25` | BM25 keyword search (hybrid RAG) |
| `pandas`, `numpy` | Data handling |
| `requests` | HTTP calls to OWUI / LiteLLM |
| `python-dotenv` | Loads `.env` credentials |
| `pyyaml` | YAML instruction lookup backend |
| `ipykernel` | Jupyter kernel |

### 2. Configure credentials

Create a `.env` file in the repo root (never commit this file — it is in `.gitignore`):

```bash
# .env
OWUI_API_KEY=<your Bearer token from OpenWebUI Settings → Account>
OWUI_URL=https://openwebui.fnal.gov
LITELLM_API_KEY=<your LiteLLM key>
LITELLM_URL=<LiteLLM base URL>

# Optional — sets the default model when none is specified
# OWUI_MODEL=qwen2.5vl:latest
# OWUI_TIMEOUT=400
```

`owui_client.py` loads this file automatically via `python-dotenv` on import.

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
| `dqm_gui_client.py` | DQM GUI HTTP client (X.509 auth, PNG fetch) — see "Fetching from the DQM GUI" |
| `fetch_gui_cli.py` | CLI driver for fetching plots from the DQM GUI |

### `PLOT_CONFIG` format

Keys become the subdirectory name and PNG filename stem. Values are ROOT object path templates with `{run}` replaced at runtime by the zero-padded 6-digit run number:

```
DQMData/Run {run}/{Subsystem}/Run summary/{...path...}/{plotName}
```

The `{run}` value is auto-detected from the filename (`R000XXXXXX`) or overridden via `RUN_OVERRIDE`.

## Fetching from the DQM GUI

`dqm_plot.py` needs a ROOT file staged on disk or XRootD. `fetch_gui_cli.py` is the
network alternative: it asks the CMS DQM GUI to render a monitor element and saves the
PNG a shifter would see, into the **same `images/<folder>/<stem>_run<XXXXXX>.png` layout**
`produce_images()` uses. Everything downstream (`run_batch_cli.py`, `evaluate_cli.py`,
reference-image lookup) is unaffected by which source produced the images.

It does not import ROOT, so it runs in the plain pixi environment without CMSSW.

### Authentication — X.509, not SSO

`cmsweb.cern.ch/dqm/*` is gated by an **X.509 client certificate**, not by CERN SSO: it
answers unauthenticated requests with a bare `401`, with no `WWW-Authenticate` header and
no redirect to `auth.cern.ch` even for a browser. So an SSO token — including one from
`tsgauth` — will not open it; a CMS grid proxy will.

```bash
voms-proxy-init -voms cms -valid 24:00
voms-proxy-info -all | grep -E 'subject|timeleft|VO'   # VO must be cms
```

Optional `.env` keys (all are *paths*, not secrets — the proxy file is the credential):

```bash
DQM_WORKSPACE=offline                            # offline | online
X509_USER_PROXY=/tmp/x509up_u1000                # default: /tmp/x509up_u<uid>
DQM_CA_BUNDLE=/etc/grid-security/certificates    # CERN Grid CA
```

`DQM_CA_BUNDLE` matters: cmsweb is signed by the CERN Grid CA, which is **not** in the
default `certifi` trust store, so verification fails with `unable to get local issuer
certificate` unless you point at the grid CA directory.

The client never creates, renews or destroys a proxy. It reads the one you have, warns
when under two hours remain, and turns cmsweb's bare `401` into a message saying whether
the proxy expired mid-run or was rejected outright.

### Usage

```bash
# Preview the URLs — no proxy needed, no network call made
python3 fetch_gui_cli.py --runs 398185 --subsystem L1T --plot 00 --dry-run \
    --dataset '/ZeroBias/Run2024C-PromptReco-v1/DQMIO'

# Check proxy status
python3 fetch_gui_cli.py --check-proxy

# Fetch
python3 fetch_gui_cli.py --runs 398185 398186 --subsystem L1T \
    --dataset '/ZeroBias/Run2024C-PromptReco-v1/DQMIO' --outdir images

# Online workspace takes no dataset
python3 fetch_gui_cli.py --runs 398185 --subsystem L1T --workspace online
```

Start with `--dry-run`. It needs no credential and prints exactly what would be requested,
which is the cheapest way to catch a wrong dataset or plot selection.

| Flag | Description |
|---|---|
| `--runs` | One or more run numbers |
| `--subsystem` / `--plot` | Plot selection, same semantics as `dqm_plot.py` |
| `--dataset` | Required for `offline`; implicit for `online` |
| `--workspace` | `offline` (default) or `online` |
| `--overwrite` | Re-fetch even if the PNG exists (default: skip) |
| `--no-cache` | Bypass the `.dqm_cache/` response cache |
| `--dry-run` | Print URLs and exit; no proxy, no network |
| `--check-proxy` | Report proxy status and exit |

### Workspace path mapping

The two workspaces address monitor elements differently, handled by
`shift_layout_helpers.gui_path()`:

```
shift_layouts.json   L1T/L1TStage2CaloLayer1/ecalOccRecdEtWgt
online               L1T/L1TStage2CaloLayer1/ecalOccRecdEtWgt
offline              L1T/Run summary/L1TStage2CaloLayer1/ecalOccRecdEtWgt
ROOT file            DQMData/Run {run}/L1T/Run summary/L1TStage2CaloLayer1/ecalOccRecdEtWgt
```

### Caveats

- **The `online` plotfairy URL is confirmed** against live cmsweb (2026-09-15): a direct
  fetch of run 398185's `L1T/L1TStage2CaloLayer1/ecalOccRecdEtWgt` returned the same PNG
  shown in the browser. **`offline` is not yet independently confirmed** — it follows the
  same documented convention and online's success makes it likely correct, but if a fetch
  404s there, the templates are at the top of `dqm_gui_client.py` in one block — that is
  the only place to edit.
- **`list_samples()` does not work for the `online` workspace — confirmed, not just
  suspected.** `data/json/samples?match=<run>` returned `{"samples": []}` for a run
  actively being written to online DQM. The online GUI browses runs through a stateful
  session (open → `select` → `setFocus`), which this client does not implement, since the
  online dataset is fixed (`/Global/Online/ALL`) and needs no lookup. Calling
  `list_samples()` on the online workspace raises `DQMGUIError` rather than returning a
  misleading empty result.
- **A missing monitor element may return a valid PNG**, not an error: the GUI renders a
  "not found" placeholder. `looks_like_placeholder()` flags suspiciously small images in
  the results, but it is a size heuristic and wants calibrating against a known-missing ME
  now that a real plot's size is known to be well above the current 6000-byte threshold.
- **Proxies expire (~24 h).** A long campaign needs a renewal; the client reports this
  rather than failing opaquely.
- Responses are cached under `.dqm_cache/` keyed by URL. Delete it to force refetching.

## Batch Querying

`run_batch_cli.py` drives `owui_client.run_batch()` — it sends every image under
`images/<plotName>/` to each configured model and writes one `.txt` response per
`(model, image)` pair under `results/<run_id>/<plotName>/`.

```bash
# Run a named preset (see PRESETS in run_batch_cli.py)
python3 run_batch_cli.py --preset yaml

# Sanity-check without querying (model validity, reference coverage, output paths)
python3 run_batch_cli.py --preset yaml --dry-run

# Re-query even if output files already exist
python3 run_batch_cli.py --preset yaml --overwrite

# Load a BatchConfig from JSON instead of a preset (e.g. a scoped/variant config)
python3 run_batch_cli.py --config test_configs/YAML_direct.json
```

| Flag | Description |
|---|---|
| `--preset NAME` | Named `BatchConfig` from `PRESETS` in `run_batch_cli.py` |
| `--config PATH` | Load a `BatchConfig` from a JSON file (overrides `--preset`) |
| `--image-mode {direct,ref_captioned,both_captioned}` | How reference/input images are sent — as-is, or captioned to text first |
| `--caption-model NAME` | Model used for captioning instead of self-caption (only with a captioned `--image-mode`) |
| `--provider {litellm,owui,nrp}` | Inference provider to query all models through |
| `--overwrite` | Re-query `(model, image)` pairs even if an output `.txt` already exists |
| `--retry` | Retry any failed queries after the batch completes |
| `--all-plots` | Clear `plot_filter` (run every plot under `image_root`) |
| `--dry-run` | Print the sanity-check summary and exit without querying |

**Scoping a run to specific plots or image modes:** there's no `--plot` flag —
`plot_filter` is set in the preset or a `--config` JSON file. `test_configs/` holds an
example: three configs cloned from the `"yaml"` preset, one per `image_mode`, each with
`plot_filter` restricted to a single plot and a distinct `run_id` (`YAML_direct`,
`YAML_ref_captioned`, `YAML_both_captioned` — outputs must use different `run_id`s per
image_mode, since output filenames don't otherwise distinguish them). Rerun all three
with `bash test_configs/run.sh` (forwards flags like `--overwrite`/`--dry-run`).

## Evaluation

`evaluate_cli.py` drives `evaluate.run_evaluations()` — the LLM-judge scoring step. It
loads `.txt` responses from one or more `run_id`s under `results/`, matches each to its
ground-truth file under `truth/`, scores it against a rubric with a judge model, and
appends the scores to a cached CSV (`eval_output/eval_scores.csv` by default) so
interrupted or repeated runs resume without re-judging. This is the headless/CLI
equivalent of the judge pipeline in `evaluate.ipynb`; the notebook is still the place
for exploratory reporting and plots (`report_scores`, `plot_section_heatmap`,
`browse_responses`, etc. — all importable from `evaluate.py`).

```bash
# Run a named preset (see PRESETS in evaluate_cli.py)
python3 evaluate_cli.py --preset yaml_image_modes

# Or pass run_ids directly
python3 evaluate_cli.py --run-ids YAML_direct YAML_ref_captioned YAML_both_captioned

# Sanity-check only — truth coverage, latency, error count — no judge calls
python3 evaluate_cli.py --preset yaml_image_modes --dry-run

# Restrict to specific evaluated models
python3 evaluate_cli.py --preset yaml_image_modes --models google/gemma4-31b qwen/qwen3.6
```

| Flag | Description |
|---|---|
| `--preset NAME` | Named `run_id` group from `PRESETS` in `evaluate_cli.py` |
| `--run-ids ID [ID ...]` | `run_id`(s) under `--output-root` to evaluate (overrides `--preset`) |
| `--models NAME [NAME ...]` | Restrict to these evaluated model names (default: all found) |
| `--judge-model NAME` | Judge model to score responses with (default: `openai/gpt-oss-120b`) |
| `--delay SECONDS` | Delay between judge calls (default: `1.0`) |
| `--output-root PATH` | Root of `<run_id>/` result dirs (default: `results`) |
| `--truth-root PATH` | Root of ground-truth `.txt` files (default: `truth`) |
| `--eval-csv PATH` | Cached score CSV to append to/resume from (default: `eval_output/eval_scores.csv`) |
| `--group-by COL [COL ...]` | Columns to group the final score report by (default: `run_id model_short`) |
| `--dry-run` | Load results and print coverage/latency/error summary; exit without judging |

Since the score CSV is a shared, append-only cache keyed on `(file, judge_model)`,
re-running the same `run_id`s is cheap — already-scored responses are skipped.

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
