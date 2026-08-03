"""
evaluate.py
-----------
Core evaluation library for DQM vision bench.
All experiment config (run IDs, models, judge, paths) lives in evaluate.ipynb.
Import this module; do not run it directly.
"""

import re
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from owui_client import query, ModelConfig
from rag_backends import NoContext


# ── Constants ──────────────────────────────────────────────────────────────────

SCORE_COLS = [
    's1_instruction_quote', 's2_plot_description',
    's3_comparison', 's4_decision', 'overall',
]
SECTION_LABELS = [
    'S1 instruction', 'S2 description',
    'S3 comparison',  'S4 decision',  'Overall',
]

IMAGE_ROOT = Path('images')

JUDGE_SYSTEM = """\
You are an expert evaluator for CMS detector quality monitoring (DQM).
Score the LLM response against the ground truth on each of the 4 sections using the rubrics below.
Return scores as integers 1–5 only. Do not interpolate (no 3.5 etc.).

SECTION 1 — Instruction quote (did it cite the right rule?):
  1 = Wrong event type cited (e.g. cosmic rules for a collision plot), or no quote at all
  2 = Correct event type but quotes irrelevant or tangential instruction
  3 = Correct event type, partially relevant quote but misses the key quality criterion
  4 = Correct event type and relevant quote, minor omission
  5 = Correct event type, quotes the exact relevant quality criterion verbatim
  NOTE: citing the wrong event type is an automatic score of 1 regardless of other quality.

SECTION 2 — Plot description (is the description physically accurate?):
  1 = Misidentifies axes, color bar, or plot type entirely
  2 = Identifies plot type but makes significant errors on axes, ranges, or color scale
  3 = Mostly correct description with one notable inaccuracy
  4 = Accurate description with only minor omissions
  5 = Complete and accurate: axes, labels, color bar range, and occupancy pattern all correct

SECTION 3 — Comparison (does reasoning connect plot features to the instruction?):
  1 = Reasoning uses wrong event type instructions, or no comparison made
  2 = Uses correct instructions but fails to connect them to specific plot features
  3 = Connects instructions to plot features but reasoning contains errors or gaps
  4 = Sound reasoning with minor gap or imprecision
  5 = Correctly identifies specific plot features and maps them to the instruction criteria

SECTION 4 — Final decision (does the Good/Bad verdict match truth?):
  1 = Wrong verdict with no justification
  2 = Wrong verdict but with some reasoning
  3 = Correct verdict but with weak or missing justification
  4 = Correct verdict with sound justification, minor gap
  5 = Correct verdict with clear reasoning referencing both the instruction and the plot

OVERALL — Holistic quality across all four sections:
  1 = Fails on multiple sections, not usable
  2 = Gets the verdict right but reasoning is largely wrong
  3 = Partially useful — correct on some sections but unreliable
  4 = Solid response, correct verdict and mostly sound reasoning
  5 = Matches truth across all four sections\
"""

JUDGE_PROMPT_TEMPLATE = """\
## Ground truth (human expert):
{truth}

## LLM response to evaluate:
{response}

Score the LLM response on each section using the rubric in your instructions.
Scores must be integers 1–5 only. Return ONLY valid JSON with no other text:
{{
  "s1_instruction_quote": {{"score": <1-5>, "comment": "<one sentence>"}},
  "s2_plot_description":  {{"score": <1-5>, "comment": "<one sentence>"}},
  "s3_comparison":        {{"score": <1-5>, "comment": "<one sentence>"}},
  "s4_decision":          {{"score": <1-5>, "comment": "<one sentence>"}},
  "overall":              {{"score": <1-5>, "comment": "<one sentence>"}}
}}
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def short_name(model: str) -> str:
    """Shorten a model ID for axis/legend labels."""
    return model.split(':')[0].split('/')[-1]

try:
    display
except NameError:
    def display(df):
        print(df.to_string())
        print()


# ── Result discovery and loading ───────────────────────────────────────────────

def parse_result_file(path: Path) -> dict:
    """Parse a batch_query .txt result file into a flat dict."""
    text = path.read_text()
    sep  = '-' * 60
    header_raw, response = (text.split(sep, 1) if sep in text else (text, ''))

    meta = {}
    for line in header_raw.strip().splitlines():
        if ':' in line:
            k, _, v = line.partition(':')
            meta[k.strip().lower().replace(' ', '_')] = v.strip()

    def _lat(s):
        try:    return float(s.rstrip('s'))
        except: return None

    plot       = meta.get('plot', '')
    image_str  = meta.get('image', '')
    image_name = Path(image_str).name if image_str else ''

    return {
        'model':                meta.get('model', ''),
        'plot_name':            plot,
        'image':                image_str,
        'image_name':           image_name,
        'run_id':               meta.get('run_id', ''),
        'prompt':               meta.get('prompt', ''),
        'load_latency_s':       _lat(meta.get('load_latency', '')),
        'generation_latency_s': _lat(meta.get('generation_latency', '')),
        'latency_s':            _lat(meta.get('total_latency', '')),
        'response':             response.strip(),
        'error':                meta.get('error') or None,
    }


def discover_runs(output_root: Path) -> pd.DataFrame:
    """
    Scan output_root/ and return a summary DataFrame of all available run
    directories. Reads config_<run_id>.json where present; falls back to
    sampling .txt files for model names.

    Columns: run_id, n_responses, n_plot_types, models, context,
             plot_filter, ref_dir, has_config
    """
    rows = []
    for run_dir in sorted(Path(output_root).iterdir()):
        if not run_dir.is_dir():
            continue

        cfg = {}
        config_files = list(run_dir.glob('config_*.json'))
        if config_files:
            with open(config_files[0]) as f:
                cfg = json.load(f)

        txt_files = list(run_dir.rglob('*.txt'))
        plot_dirs = [d.name for d in run_dir.iterdir() if d.is_dir()]

        if cfg.get('models'):
            models = [m['name'] if isinstance(m, dict) else m
                      for m in cfg['models']]
        else:
            # Sample up to 20 files to discover model names without full load
            models_seen: set[str] = set()
            for txt in txt_files[:20]:
                try:
                    models_seen.add(parse_result_file(txt)['model'])
                except Exception:
                    pass
            models = sorted(models_seen)

        rows.append({
            'run_id':        run_dir.name,
            'n_responses':   len(txt_files),
            'n_plot_types':  len(plot_dirs),
            'models':        ', '.join(short_name(m) for m in models),
            'context':       cfg.get('context', {}).get('type', '?') if cfg else '?',
            'plot_filter':   str(cfg.get('plot_filter') or '') if cfg else '',
            'ref_dir':       str(cfg.get('ref_dir') or '') if cfg else '',
            'has_config':    bool(config_files),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['run_id', 'n_responses', 'n_plot_types', 'models',
                 'context', 'plot_filter', 'ref_dir', 'has_config'])


def load_results(
    output_root: Path,
    run_ids: 'list[str]',
    models: 'list[str] | None' = None,
) -> pd.DataFrame:
    """
    Load all .txt result files for the given run_ids from output_root/<run_id>/.
    Tags each row with run_id. Deduplicates by (run_id, model, image), keeping
    the most recently written file.
    """
    rows = []
    for run_id in run_ids:
        run_dir = Path(output_root) / run_id
        if not run_dir.exists():
            print(f'  WARNING: {run_dir} not found — skipping')
            continue
        for txt in sorted(run_dir.rglob('*.txt')):
            try:
                parsed = parse_result_file(txt)
            except Exception as e:
                print(f'  WARNING: could not parse {txt.name}: {e}')
                continue
            if models and parsed['model'] not in models:
                continue
            parsed['run_id'] = run_id
            parsed['file']   = str(txt)
            parsed['mtime']  = txt.stat().st_mtime
            rows.append(parsed)

    if not rows:
        print(f'No result files found for run_ids={run_ids}')
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    before = len(df)
    df = (
        df.sort_values('mtime')
          .drop_duplicates(subset=['run_id', 'model', 'image'], keep='last')
          .drop(columns='mtime')
          .reset_index(drop=True)
    )
    if len(df) < before:
        print(f'  Deduplicated {before} → {len(df)} '
              f'(dropped {before - len(df)} older duplicates)')
    print(f'Loaded {len(df)} responses — '
          f'{df["run_id"].nunique()} run(s), '
          f'{df["model"].nunique()} model(s), '
          f'{df["plot_name"].nunique()} plot type(s)')
    return df


# ── Truth ──────────────────────────────────────────────────────────────────────

def find_truth_file(
    truth_root: Path, plot: str, image_str: str
) -> 'Path | None':
    """
    Map an image filename to its truth file.

    Good images:  *_run<N>.png              → truth/<plot>/run<N>.txt
    Bad images:   *_run<N>_bad.png          → truth/<plot>/truth_run<N>_bad.txt
                  *_run<N>_bad_small.png    → truth/<plot>/truth_run<N>_bad_small.txt
                  *_run<N>_bad_medium.png   → truth/<plot>/truth_run<N>_bad_medium.txt
    """
    stem  = Path(image_str).stem
    match = re.search(r'run\d+', stem)
    run   = match.group() if match else stem

    if   '_bad_small'  in stem: candidate = truth_root / plot / f'truth_{run}_bad_small.txt'
    elif '_bad_medium' in stem: candidate = truth_root / plot / f'truth_{run}_bad_medium.txt'
    elif '_bad'        in stem: candidate = truth_root / plot / f'truth_{run}_bad.txt'
    else:                       candidate = truth_root / plot / f'{run}.txt'

    return candidate if candidate.exists() else None


def truth_coverage(df: pd.DataFrame, truth_root: Path) -> pd.DataFrame:
    """Return a DataFrame showing which (plot, image) pairs have truth files."""
    rows = []
    for _, row in df[['plot_name', 'image', 'image_name']].drop_duplicates().iterrows():
        tf = find_truth_file(truth_root, row['plot_name'], row['image'])
        rows.append({
            'plot_name': row['plot_name'],
            'image':     row['image_name'],
            'has_truth': tf is not None,
        })
    cov = pd.DataFrame(rows).sort_values(['plot_name', 'image']).reset_index(drop=True)
    n   = cov['has_truth'].sum()
    print(f'Truth coverage: {n}/{len(cov)} images have truth files')
    return cov


# ── Judge ──────────────────────────────────────────────────────────────────────

def judge(
    truth_text: str,
    response_text: str,
    model: str,
    judge_system: str = JUDGE_SYSTEM,
    judge_prompt_template: str = JUDGE_PROMPT_TEMPLATE,
) -> dict:
    """Call the judge model. Returns a scores dict or {'error': ...}."""
    prompt = judge_prompt_template.format(truth=truth_text, response=response_text)
    result = query(prompt, model=ModelConfig(name=model),
                   system=judge_system, context=NoContext())
    if result.get('error'):
        return {'error': f"API error: {result['error']}"}
    raw = result.get('response') or ''
    if not raw:
        return {'error': 'empty response from judge'}
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(m.group()) if m else {'error': f'no JSON in: {raw[:200]}'}
    except Exception as e:
        return {'error': str(e), 'raw': raw}


def run_evaluations(
    df_results: pd.DataFrame,
    truth_root: Path,
    judge_model_for: 'dict[str, list[str]]',
    eval_csv: Path,
    delay: float = 1.0,
    judge_system: str = JUDGE_SYSTEM,
    judge_prompt_template: str = JUDGE_PROMPT_TEMPLATE,
) -> pd.DataFrame:
    """
    Run LLM judge over df_results, appending new scores to eval_csv immediately
    so interrupted runs resume from where they left off.

    judge_model_for: {evaluated_model_name: [judge_model, ...]}
    Returns the full scored DataFrame (cached + new rows).
    """
    Path(eval_csv).parent.mkdir(parents=True, exist_ok=True)

    if Path(eval_csv).exists():
        df_eval = pd.read_csv(eval_csv)
        done    = set(zip(df_eval['file'], df_eval['judge_model']))
        print(f'Loaded {len(df_eval)} cached scores — resuming')
    else:
        df_eval = pd.DataFrame()
        done    = set()

    new_rows: list[dict] = []

    for _, row in df_results.iterrows():
        if not row.get('response') or not row.get('plot_name'):
            continue

        truth_file = find_truth_file(truth_root, row['plot_name'], row['image'])
        if not truth_file:
            continue
        truth_text = truth_file.read_text().strip()

        for judge_model in judge_model_for.get(row['model'], []):
            if judge_model == row['model']:
                continue
            if (row['file'], judge_model) in done:
                continue

            print(
                f"  [{short_name(row['model']):18s}]"
                f" run={row['run_id']:10s}"
                f" img={Path(row['image']).name}"
                f" judge={short_name(judge_model)} ...",
                end=' ', flush=True,
            )
            scores = judge(truth_text, row['response'], judge_model,
                           judge_system, judge_prompt_template)
            if 'error' in scores:
                print(f"ERROR: {scores['error']}")
                continue
            print('ok')

            new_row = {
                'file':                 row['file'],
                'run_id':               row['run_id'],
                'plot_name':            row['plot_name'],
                'image':                row['image_name'],
                'model':                row['model'],
                'judge_model':          judge_model,
                'generation_latency_s': row.get('generation_latency_s'),
                'latency_s':            row.get('latency_s'),
                **{k:              v['score']   if isinstance(v, dict) else None
                   for k, v in scores.items()},
                **{f'{k}_comment': v['comment'] if isinstance(v, dict) else None
                   for k, v in scores.items()},
            }
            new_rows.append(new_row)
            done.add((row['file'], judge_model))

            pd.DataFrame([new_row]).to_csv(
                eval_csv, mode='a', header=not Path(eval_csv).exists(), index=False,
            )
            time.sleep(delay)

    if new_rows:
        df_eval = pd.concat([df_eval, pd.DataFrame(new_rows)], ignore_index=True)
        print(f'\nSaved {len(new_rows)} new score(s) → {eval_csv}  '
              f'(total {len(df_eval)})')
    elif df_eval.empty:
        print('No scores produced. Check truth files exist and model names are correct.')
        return df_eval

    df_eval['s4_correct'] = (df_eval['s4_decision'] >= 3)
    df_eval['image_quality'] = np.where(
        df_eval['image'].astype(str).apply(lambda s: '_bad' in Path(s).stem),
        'bad', 'good',
    )
    return df_eval


# ── Reports (text) ─────────────────────────────────────────────────────────────

def report_latency(df: pd.DataFrame, group_col: str = 'model') -> None:
    """Print latency statistics from the raw results DataFrame."""
    print(f'=== Generation latency by {group_col} ===')
    display(
        df.groupby(group_col)['generation_latency_s']
        .agg(['mean', 'median', 'min', 'max', 'count'])
        .round(2)
        .rename(columns={'mean': 'mean_s', 'median': 'med_s',
                         'min': 'min_s', 'max': 'max_s'})
    )
    if 'run_id' in df.columns and group_col != 'run_id':
        print(f'\n=== Generation latency — run_id × {group_col} ===')
        display(
            df.groupby(['run_id', group_col])['generation_latency_s']
            .mean().round(2).rename('mean gen_s')
            .unstack(group_col)
        )


def report_scores(df_eval: pd.DataFrame, group_col: str = 'model') -> None:
    """Print section score tables from the scored DataFrame."""
    print(f'=== Mean section scores by {group_col} ===')
    display(df_eval.groupby(group_col)[SCORE_COLS].mean().round(2))

    if 'run_id' in df_eval.columns and group_col != 'run_id':
        print(f'\n=== Mean scores by run_id × {group_col} (overall only) ===')
        display(
            df_eval.groupby(['run_id', group_col])['overall']
            .mean().round(2).unstack(group_col)
        )

    print(f'\n=== S4 decision accuracy (% correct) by {group_col} ===')
    display(
        df_eval.groupby(group_col)['s4_correct']
        .mean().mul(100).round(1).rename('% correct').to_frame()
    )


# ── Plots ──────────────────────────────────────────────────────────────────────

def _save(fig, out_path: Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_section_heatmap(
    df_eval: pd.DataFrame,
    row_col: str,
    title: str,
    out_path: Path,
) -> None:
    """Heatmap: rows = row_col values, columns = sections, cells = mean score."""
    row_vals = sorted(df_eval[row_col].dropna().unique())
    pivot    = df_eval.groupby(row_col)[SCORE_COLS].mean().reindex(row_vals)

    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.55 * len(row_vals) + 1.5)))
    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=1, vmax=5, aspect='auto')
    ax.set_xticks(range(len(SCORE_COLS)))
    ax.set_xticklabels(SECTION_LABELS, rotation=20, ha='right', fontsize=9)
    ax.set_yticks(range(len(row_vals)))
    ax.set_yticklabels([short_name(str(v)) for v in row_vals], fontsize=9)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label='mean score (1–5)')
    for i in range(len(row_vals)):
        for j in range(len(SCORE_COLS)):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                        fontsize=9, fontweight='bold')
    plt.tight_layout()
    _save(fig, out_path)


def plot_s4_accuracy(
    df_eval: pd.DataFrame,
    row_col: str,
    title: str,
    out_path: Path,
) -> None:
    """Bar chart of S4 (Good/Bad) decision accuracy per row_col value."""
    row_vals = sorted(df_eval[row_col].dropna().unique())
    acc      = df_eval.groupby(row_col)['s4_correct'].mean().mul(100).reindex(row_vals)
    colors   = plt.cm.Set1(np.linspace(0, 1, len(row_vals)))

    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(row_vals)), 4))
    ax.bar(range(len(row_vals)), acc.values, color=colors)
    ax.set_xticks(range(len(row_vals)))
    ax.set_xticklabels([short_name(str(v)) for v in row_vals],
                       rotation=20, ha='right', fontsize=9)
    ax.set_ylim(0, 110)
    ax.axhline(100, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('% correct  (S4 score ≥ 3)')
    for i, v in enumerate(acc.values):
        if pd.notna(v):
            ax.text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=9)
    plt.tight_layout()
    _save(fig, out_path)


def plot_latency(
    df: pd.DataFrame,
    row_col: str,
    title: str,
    out_path: Path,
) -> None:
    """Bar chart of mean generation latency per row_col value."""
    row_vals = sorted(df[row_col].dropna().unique())
    lat      = df.groupby(row_col)['generation_latency_s'].mean().reindex(row_vals)
    colors   = plt.cm.Set1(np.linspace(0, 1, len(row_vals)))

    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(row_vals)), 4))
    ax.bar(range(len(row_vals)), lat.values, color=colors)
    ax.set_xticks(range(len(row_vals)))
    ax.set_xticklabels([short_name(str(v)) for v in row_vals],
                       rotation=20, ha='right', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('Mean generation latency (s)')
    max_lat = float(np.nanmax(lat.values.astype(float))) if lat.notna().any() else 1.0
    for i, v in enumerate(lat.values):
        if pd.notna(v):
            ax.text(i, v + max_lat * 0.02, f'{v:.1f}s', ha='center', fontsize=9)
    plt.tight_layout()
    _save(fig, out_path)


def plot_comparison(
    df_eval: pd.DataFrame,
    compare_col: str,
    row_col: str,
    title: str,
    out_path: Path,
) -> None:
    """
    Grouped bar chart comparing compare_col values across sections.
    One subplot per row_col value (e.g. one per model).
    Within each subplot: x = sections, bar groups = compare_col values.

    Example: compare_col='run_id', row_col='model'
      → for each model, shows localRAG vs YAML bars side-by-side per section.
    """
    row_vals     = sorted(df_eval[row_col].dropna().unique())
    compare_vals = sorted(df_eval[compare_col].dropna().unique())
    colors       = plt.cm.Set2(np.linspace(0, 1, len(compare_vals)))

    x     = np.arange(len(SCORE_COLS))
    width = 0.8 / len(compare_vals)

    fig, axes = plt.subplots(
        len(row_vals), 1,
        figsize=(10, 3.5 * len(row_vals)),
        squeeze=False,
    )
    for ri, row_val in enumerate(row_vals):
        ax  = axes[ri, 0]
        sub = df_eval[df_eval[row_col] == row_val]

        for ci, cval in enumerate(compare_vals):
            scores = (
                sub[sub[compare_col] == cval][SCORE_COLS]
                .mean().values.astype(float)
            )
            bars = ax.bar(x + ci * width, scores, width,
                          label=str(cval), color=colors[ci])
            for bar, v in zip(bars, scores):
                if pd.notna(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.05,
                            f'{v:.1f}', ha='center', fontsize=7)

        ax.set_title(f'{row_col} = {short_name(str(row_val))}', fontsize=9)
        ax.set_xticks(x + width * (len(compare_vals) - 1) / 2)
        ax.set_xticklabels(SECTION_LABELS, rotation=15, ha='right', fontsize=9)
        ax.set_ylim(1, 5.8)
        ax.set_ylabel('Mean score (1–5)')
        ax.legend(title=compare_col, fontsize=8, loc='lower right')

    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    _save(fig, out_path)
