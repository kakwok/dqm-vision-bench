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
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import mplhep as hep
    _HAS_MPLHEP = True
except ImportError:
    hep = None
    _HAS_MPLHEP = False

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
    df = _add_run_subsystem_cols(df)
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

    if 'model_short' in df.columns:
        by_model = (
            df[['model_short', 'plot_name', 'image_name']]
            .merge(cov, left_on=['plot_name', 'image_name'], right_on=['plot_name', 'image'])
            .groupby('model_short')['has_truth']
            .agg(has_truth='sum', n_images='count')
        )
        print('\n=== Truth coverage by model ===')
        display(by_model)

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
                **{k:              scores.get(k, {}).get('score')
                   if isinstance(scores.get(k), dict) else None
                   for k in SCORE_COLS},
                **{f'{k}_comment': scores.get(k, {}).get('comment')
                   if isinstance(scores.get(k), dict) else None
                   for k in SCORE_COLS},
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

    return _add_derived_cols(df_eval)


def load_eval_csv(eval_csv: Path) -> pd.DataFrame:
    """Load a saved eval_scores.csv and recompute derived columns."""
    df = pd.read_csv(eval_csv)
    return _add_derived_cols(df)


def _extract_run_number(image_str: str) -> 'int | None':
    """Pull the numeric DQM run number out of an image filename (e.g. 'run398185' -> 398185)."""
    m = re.search(r'run(\d+)', Path(str(image_str)).stem)
    return int(m.group(1)) if m else None


def _add_run_subsystem_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'model_short' (provider-stripped model id), 'run_number' (int, from image
    filename), and 'subsystem' (from plot_name prefix)."""
    df = df.copy()
    df['model_short'] = df['model'].astype(str).apply(short_name)
    df['run_number']  = df['image'].apply(_extract_run_number)
    df['subsystem']   = df['plot_name'].astype(str).apply(lambda s: s.split('_')[0])
    return df


def _add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['s4_correct'] = (df['s4_decision'] >= 3)
    df['image_quality'] = np.where(
        df['image'].astype(str).apply(lambda s: '_bad' in Path(s).stem),
        'bad', 'good',
    )
    df = _add_run_subsystem_cols(df)
    return df


# ── Reports (text) ─────────────────────────────────────────────────────────────

def report_latency(df: pd.DataFrame, group_col: str = 'model_short') -> None:
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


# Section markers the response/truth text is structured around. The model is
# instructed (see batch_query.ipynb SYSTEM_PROMPT) to answer in exactly these
# 4 sections; truth files follow a matching fixed template. 'overall' has no
# corresponding section in either text — it's a holistic judge score.
_RESPONSE_SECTION_MARKERS = {
    's1_instruction_quote': 'Instructions',
    's2_plot_description':  'Observations',
    's3_comparison':        'Assessment',
    's4_decision':          'Verdict',
}
_TRUTH_SECTION_MARKERS = {
    's1_instruction_quote': 'Quote',
    's2_plot_description':  'Describe',
    's3_comparison':        'Compare',
    's4_decision':          'Decide',
}


def _split_sections(text: 'str | None', markers: dict, keep_same_line: bool = True) -> dict:
    """
    Split free-form text into {score_col: content} based on lines that open a
    new section — a line that, after stripping markdown decoration (#, *, -,
    >, whitespace), starts with one of `markers`' values (case-insensitive).

    keep_same_line: if True, text following the marker on its own line (e.g.
      "Verdict: GOOD") is kept as the start of that section's content; if
      False (used for truth files, whose marker line is itself boilerplate
      instruction text, not ground truth), that remainder is discarded and
      content is only collected from subsequent lines.
    """
    cols = list(markers.keys())
    if not text:
        return {c: '' for c in cols}

    label_to_col = {v.lower(): k for k, v in markers.items()}
    pattern = re.compile(
        r'^[#>\-\*\s]*(' + '|'.join(re.escape(v) for v in markers.values()) + r')\b[:\s\*#_]*',
        re.IGNORECASE,
    )
    collected = {c: [] for c in cols}
    current = None
    for line in str(text).splitlines():
        m = pattern.match(line)
        if m:
            current = label_to_col[m.group(1).lower()]
            if keep_same_line:
                rest = line[m.end():].strip()
                if rest:
                    collected[current].append(rest)
            continue
        if current is not None:
            collected[current].append(line)
    return {c: '\n'.join(v).strip() for c, v in collected.items()}


def _wrap_cell(text: 'str | None', width: int) -> 'list[str]':
    if not text:
        return ['']
    lines = []
    for para in str(text).splitlines() or ['']:
        lines.extend(textwrap.wrap(para, width=width) or [''])
    return lines or ['']


def _print_section_table(rows: 'list[dict]', columns: 'list[tuple[str, int]]') -> None:
    """Print rows as a fixed-width, word-wrapped text table (no pandas repr)."""
    rule = '+' + '+'.join('-' * (w + 2) for _, w in columns) + '+'
    print(rule)
    print('| ' + ' | '.join(h.ljust(w) for h, w in columns) + ' |')
    print(rule)
    for row in rows:
        wrapped = {h: _wrap_cell(row.get(h, ''), w) for h, w in columns}
        for i in range(max(len(v) for v in wrapped.values())):
            cells = [
                (wrapped[h][i] if i < len(wrapped[h]) else '').ljust(w)
                for h, w in columns
            ]
            print('| ' + ' | '.join(cells) + ' |')
        print(rule)


def check_df_eval_consistency(df: pd.DataFrame, df_eval: pd.DataFrame) -> None:
    """
    Warn about mismatches between raw responses (df) and judge scores (df_eval),
    which browse_responses left-joins on 'file'. df_eval is built from df by
    run_evaluations, so its files should always be a subset of df's:
      - 'missing' (df files absent from df_eval) is expected whenever some
        responses haven't been judged yet — those rows show blank scores in
        browse_responses instead of erroring, so it's worth surfacing.
      - 'extra' (df_eval files absent from df) should never happen — it means
        df_eval is stale, e.g. loaded from a different RUN_IDS/MODELS
        selection than the currently loaded df.
    """
    df_files      = set(df['file'])
    df_eval_files = set(df_eval['file'])
    missing       = df_files - df_eval_files
    extra         = df_eval_files - df_files

    if missing:
        print(f'WARNING: {len(missing)}/{len(df_files)} responses in df have no df_eval match '
              f'(scores/comments will be blank for these in browse_responses):')
        display(
            df[df['file'].isin(missing)]
            .groupby(['run_id', 'model_short'])['image_name'].count()
            .rename('n_missing_from_df_eval')
        )
    else:
        print(f'OK: all {len(df_files)} responses in df have a matching df_eval row')

    if extra:
        print(f'ALERT: {len(extra)} df_eval rows have no matching file in df — df_eval should '
              f'always be derived from df, so this means df_eval is stale (e.g. loaded from a '
              f'different RUN_IDS/MODELS selection than the currently loaded df):')
        display(
            df_eval[df_eval['file'].isin(extra)]
            .groupby(['run_id', 'model_short'])['file'].count()
            .rename('n_extra_in_df_eval')
        )


def browse_responses(
    df: pd.DataFrame,
    df_eval: 'pd.DataFrame | None' = None,
    truth_root: 'Path | None' = None,
    *,
    run_id=None,
    plot_name=None,
    model=None,
    image_name=None,
    judge_model=None,
    run_number=None,
    n: 'int | None' = 5,
    width: int = 40,
) -> pd.DataFrame:
    """
    Debugging printout of individual responses. For each matching response,
    prints a metadata header followed by a table with one row per section
    (s1_instruction_quote, s2_plot_description, s3_comparison, s4_decision,
    overall) and columns: score, response (that section's slice of the raw
    response text), truth (that section's slice of the truth text, if
    truth_root is given), and the judge's comment (if df_eval is given).

    run_id, plot_name, model, image_name, judge_model, run_number — filter
      controls, same convention as the plot functions: None (default)
      includes everything, a single value is an exact match, and a list
      restricts to those values. judge_model only has an effect when df_eval
      is given — a single raw response can match more than one judge_model,
      producing one printout each.
    n — cap the number of responses shown (None = no cap; default 5).
    width — wrap width, in characters, for the response/truth/comment columns.

    Returns the flat (non-split) matches as a DataFrame for further use —
    printing is a side effect.
    """
    if df_eval is not None:
        check_df_eval_consistency(df, df_eval)
        eval_cols = ['file', 'judge_model'] + SCORE_COLS + [f'{c}_comment' for c in SCORE_COLS]
        merged = df.merge(df_eval[eval_cols], on='file', how='left')
    else:
        merged = df.copy()
        merged['judge_model'] = None

    sub_all = _apply_filters(
        merged, run_id=run_id, plot_name=plot_name, model=model,
        image_name=image_name, judge_model=judge_model, run_number=run_number,
    )
    sub = sub_all.head(n) if n is not None else sub_all

    print(f'{len(sub)} of {len(sub_all)} matching response(s) shown\n')

    columns = [('section', 15), ('score', 5), ('response', width),
               ('truth', width), ('comment', width)]

    for _, row in sub.iterrows():
        print('=' * 100)
        print(f"run_id={row['run_id']}  model={row.get('model_short', row['model'])}  "
              f"plot={row['plot_name']}  image={row['image_name']}  "
              f"judge={row.get('judge_model') or '—'}  latency={row.get('latency_s')}s")
        if row.get('error'):
            print(f"ERROR: {row['error']}")

        truth_text = None
        if truth_root is not None:
            tf = find_truth_file(truth_root, row['plot_name'], row['image'])
            truth_text = tf.read_text().strip() if tf else None

        response_parts = _split_sections(row.get('response'), _RESPONSE_SECTION_MARKERS)
        truth_parts     = _split_sections(truth_text, _TRUTH_SECTION_MARKERS, keep_same_line=False)

        table_rows = []
        for score_col, label in zip(SCORE_COLS, SECTION_LABELS):
            table_rows.append({
                'section':  label,
                'score':    row.get(score_col, ''),
                'response': response_parts.get(score_col, ''),
                'truth':    truth_parts.get(score_col, ''),
                'comment':  row.get(f'{score_col}_comment', ''),
            })
        _print_section_table(table_rows, columns)
        print()

    return sub_all.reset_index(drop=True)


# ── Plots ──────────────────────────────────────────────────────────────────────

def _save(fig, out_path: 'Path | None' = None) -> None:
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {out_path}')
    try:
        get_ipython()
    except NameError:
        pass
    else:
        from IPython.display import display as ipy_display
        ipy_display(fig)
    plt.close(fig)


def _apply_filters(df: pd.DataFrame, **filters) -> pd.DataFrame:
    """
    Apply exact/list/default filters to df columns.

    For each `col=val` kwarg:
      val is None                  -> no filtering (include all, the default)
      val is a scalar               -> exact match
      val is a list/tuple/set       -> restrict to the given values

    The `model` filter matches against both the full `model` column and the
    provider-stripped `model_short` column, so callers can pass either form
    (or a mix of both) without needing to know which one a given value is.
    """
    for col, val in filters.items():
        if val is None:
            continue
        vals = val if isinstance(val, (list, tuple, set)) else [val]
        if col == 'model' and 'model_short' in df.columns:
            df = df[df[col].isin(vals) | df['model_short'].isin(vals)]
        else:
            df = df[df[col].isin(vals)]
    return df


def plot_section_heatmap(
    df_eval: pd.DataFrame,
    row_col: str,
    title: str,
    out_path: 'Path | None' = None,
    *,
    subplot_col: 'str | None' = None,
    run_id=None,
    plot_name=None,
    model=None,
    agg: str = 'mean',
):
    """
    Heatmap: rows = row_col values, columns = sections, cells = agg(score).

    subplot_col: optional — if given, draws one heatmap panel per
      subplot_col value (e.g. one per model), stacked as separate Axes in a
      single figure, each with its own row_col rows (e.g. run_id). Mirrors
      plot_comparison's "one subplot per row_col" behavior, but rendered as
      heatmaps instead of grouped bars.

    Dimensions:
      run_id, plot_name, model — filter controls, each either None (include
        all, default), an exact value, or a list of values to restrict to.
        row_col (and subplot_col, if used) is typically one of these three
        (the axis/axes being compared).
      run_number — never filtered; always collapsed via `agg`
        ('mean' default, 'median', or 'std').

    Returns the Axes (single panel) or a list of Axes (one per subplot_col
    value) for further styling.
    """
    sub_all = _apply_filters(df_eval, run_id=run_id, plot_name=plot_name, model=model)
    panels  = (
        sorted(sub_all[subplot_col].dropna().unique(), key=str)
        if subplot_col is not None else [None]
    )

    if agg == 'std':
        all_vals   = sub_all[SCORE_COLS].values
        vmin, vmax = 0, float(np.nanmax(all_vals)) if np.isfinite(all_vals).any() else 1
        cmap, cbar_label = 'YlOrRd', 'std dev'
    else:
        vmin, vmax = 1, 5
        cmap, cbar_label = 'RdYlGn', f'{agg} score (1–5)'

    panel_data = []
    for panel_val in panels:
        panel_sub = sub_all if panel_val is None else sub_all[sub_all[subplot_col] == panel_val]
        row_vals  = sorted(panel_sub[row_col].dropna().unique(), key=str)
        pivot     = panel_sub.groupby(row_col)[SCORE_COLS].agg(agg).reindex(row_vals)
        panel_data.append((panel_val, row_vals, pivot))

    heights = [max(2.5, 0.55 * len(row_vals) + 1.5) for _, row_vals, _ in panel_data]
    fig, axes = plt.subplots(
        len(panel_data), 1, figsize=(9, sum(heights)),
        gridspec_kw={'height_ratios': heights}, squeeze=False,
    )

    for i, (panel_val, row_vals, pivot) in enumerate(panel_data):
        ax = axes[i, 0]
        im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(SCORE_COLS)))
        ax.set_xticklabels(SECTION_LABELS, rotation=20, ha='right', fontsize=9)
        ax.set_yticks(range(len(row_vals)))
        ax.set_yticklabels([short_name(str(v)) for v in row_vals], fontsize=9)
        panel_title = title if panel_val is None else f'{title}  |  {short_name(str(panel_val))}'
        ax.set_title(panel_title, fontsize=10)
        plt.colorbar(im, ax=ax, label=cbar_label)
        for r in range(len(row_vals)):
            for c in range(len(SCORE_COLS)):
                v = pivot.values[r, c]
                if pd.notna(v):
                    ax.text(c, r, f'{v:.1f}', ha='center', va='center',
                            fontsize=9, fontweight='bold')

    plt.tight_layout()
    _save(fig, out_path)
    return axes[0, 0] if subplot_col is None else list(axes[:, 0])


def plot_s4_accuracy(
    df_eval: pd.DataFrame,
    row_col: str,
    title: str,
    out_path: 'Path | None' = None,
    *,
    run_id=None,
    plot_name=None,
    model=None,
):
    """
    Bar chart of S4 (Good/Bad) decision accuracy per row_col value.

    run_id, plot_name, model — filter controls (None = all, exact value, or a
      list to restrict to), same convention as plot_section_heatmap.

    Returns the Axes for further styling.
    """
    sub      = _apply_filters(df_eval, run_id=run_id, plot_name=plot_name, model=model)
    row_vals = sorted(sub[row_col].dropna().unique(), key=str)
    acc      = sub.groupby(row_col)['s4_correct'].mean().mul(100).reindex(row_vals)
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
    return ax


def plot_histogram(
    df: pd.DataFrame,
    row_col: str,
    value_col: str,
    title: str,
    out_path: 'Path | None' = None,
    *,
    xlabel: 'str | None' = None,
    run_id=None,
    plot_name=None,
    model=None,
    agg: str = 'mean',
    bins: int = 20,
    overlay: bool = False,
):
    """
    Histogram of value_col (1D output, e.g. latency) split by row_col value.
    Rendered with mplhep in CMS style when mplhep is available (axis styling
    only — no "CMS ..." experiment label is drawn).

    xlabel: x-axis label; defaults to value_col if not given.

    Same dimensions and controls as plot_section_heatmap:
      run_id, plot_name, model — filter controls (None = all, exact value,
        or a list to restrict to).
      run_number — never filtered; contributes its full distribution to the
        histogram. `agg` ('mean' default, 'median', or 'std') is drawn as a
        reference line/label per row_col value rather than collapsing the data.

    overlay: False (default) draws one subplot per row_col value (filled bars).
      True draws every row_col value as an outlined step histogram on a single
      shared Axes — use this to compare several models/run_ids on one plot
      instead of scrolling through subplots.

    Returns (fig, Axes) — a single Axes for overlay or a single row_col value,
    or (fig, array of Axes) one per row_col value otherwise.
    """
    sub      = _apply_filters(df, run_id=run_id, plot_name=plot_name, model=model)
    row_vals = sorted(sub[row_col].dropna().unique(), key=str)
    colors   = plt.cm.Set1(np.linspace(0, 1, len(row_vals)))

    all_vals  = sub[value_col].dropna()
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins) if len(all_vals) else bins

    n_axes = 1 if overlay else len(row_vals)
    style_ctx = plt.style.context(hep.style.CMS) if _HAS_MPLHEP else plt.style.context({})
    with style_ctx:
        fig, axes = plt.subplots(n_axes, 1, figsize=(8, 6 * n_axes), squeeze=False)
        for i, val in enumerate(row_vals):
            ax   = axes[0, 0] if overlay else axes[i, 0]
            vals = sub.loc[sub[row_col] == val, value_col].dropna()

            if len(vals):
                reduced = vals.agg(agg)
                label   = f'{short_name(str(val))} ({agg}={reduced:.2f})'
            else:
                reduced, label = None, short_name(str(val))

            color = mcolors.to_hex(colors[i])
            if _HAS_MPLHEP:
                counts, edges = np.histogram(vals, bins=bin_edges)
                hep.histplot(counts, edges, ax=ax,
                            histtype='step' if overlay else 'fill',
                            color=color, edgecolor=color if overlay else 'black',
                            linewidth=2 if overlay else 1,
                            alpha=0.9 if overlay else 0.85, label=label)
            else:
                ax.hist(vals, bins=bin_edges, color=color,
                        histtype='step' if overlay else 'bar',
                        edgecolor=None if overlay else 'black',
                        linewidth=2 if overlay else 1,
                        alpha=0.9 if overlay else 0.85, label=label)

            if reduced is not None:
                ax.axvline(reduced, color=color if overlay else 'black',
                          linestyle='--', linewidth=1.2)

            if not overlay:
                ax.set_xlabel(xlabel or value_col)
                ax.set_ylabel('count')
                ax.legend(fontsize=9, loc='upper right')

        if overlay:
            ax = axes[0, 0]
            ax.set_xlabel(xlabel or value_col)
            ax.set_ylabel('count')
            ax.legend(fontsize=9, loc='upper right')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        _save(fig, out_path)

    if overlay or len(row_vals) <= 1:
        return fig, axes[0, 0]
    return fig, axes


def plot_latency(
    df: pd.DataFrame,
    row_col: str,
    title: str,
    out_path: 'Path | None' = None,
    *,
    run_id=None,
    plot_name=None,
    model=None,
    agg: str = 'mean',
    bins: int = 20,
    overlay: bool = False,
):
    """
    Histogram of generation latency per row_col value. See plot_histogram.
    Returns (fig, Axes).
    """
    return plot_histogram(
        df, row_col, 'generation_latency_s', title, out_path,
        xlabel='Generation latency (s)',
        run_id=run_id, plot_name=plot_name, model=model, agg=agg, bins=bins,
        overlay=overlay,
    )


def plot_comparison(
    df_eval: pd.DataFrame,
    compare_col: str,
    row_col: str,
    title: str,
    out_path: 'Path | None' = None,
    *,
    run_id=None,
    plot_name=None,
    model=None,
    agg: str = 'mean',
):
    """
    Grouped bar chart comparing compare_col values across sections.
    One subplot per row_col value (e.g. one per model).
    Within each subplot: x = sections, bar groups = compare_col values.

    Example: compare_col='run_id', row_col='model'
      → for each model, shows localRAG vs YAML bars side-by-side per section.

    run_id, plot_name, model — filter controls (None = all, exact value, or a
      list to restrict to), same convention as plot_section_heatmap. Useful to
      narrow down further when compare_col/row_col are a different pair of
      dimensions (e.g. restrict plot_name while comparing run_id × model).
    run_number — never filtered; always collapsed via `agg`
      ('mean' default, 'median', or 'std').

    Returns the array of Axes for further styling.
    """
    sub_all      = _apply_filters(df_eval, run_id=run_id, plot_name=plot_name, model=model)
    row_vals     = sorted(sub_all[row_col].dropna().unique(), key=str)
    compare_vals = sorted(sub_all[compare_col].dropna().unique(), key=str)
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
        sub = sub_all[sub_all[row_col] == row_val]

        for ci, cval in enumerate(compare_vals):
            scores = (
                sub[sub[compare_col] == cval][SCORE_COLS]
                .agg(agg).values.astype(float)
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
        if agg != 'std':
            ax.set_ylim(1, 5.8)
        ax.set_ylabel(f'{agg} score (1–5)')
        ax.legend(title=compare_col, fontsize=8, loc='lower right')

    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    _save(fig, out_path)
    return axes if len(row_vals) > 1 else axes[0, 0]
