# evaluate.py
# -----------
# Runs LLM-as-judge evaluation on batch_query outputs and produces
# score tables and plots.
#
# Usage: run cell-by-cell in Jupyter, or `python evaluate.py`
#
# Expected directory layout:
#   results/
#       <run_id>_no_ref/<plot>/<image_stem>_<model>.txt
#       <run_id>_with_ref/...
#       <run_id>_with_goodref/...
#       <run_id>_with_refs/...
#     run_id is one of RUN_IDS, e.g. "goodtest" (clean images) or
#     "badtest" (known-bad images: <..>_bad / _bad_small / _bad_medium)
#   truth/
#       <plot>/run<XXXXXX>.txt                     (good images)
#       <plot>/truth_run<XXXXXX>_bad.txt            (bad images)
#       <plot>/truth_run<XXXXXX>_bad_small.txt
#       <plot>/truth_run<XXXXXX>_bad_medium.txt

# %% ── Imports & config ───────────────────────────────────────────────────────
import re
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from owui_client import query, ModelConfig, RAGConfig

# display() is built-in in Jupyter; fall back to print when running as a script
try:
    display
except NameError:
    def display(df): print(df.to_string()); print()

# ── CLI argument: --judge overrides the default judge per model ───────────────
# Usage: python evaluate.py --judge qwen3-vl:latest
# In Jupyter, JUDGE_OVERRIDE stays None and JUDGE_MODEL_FOR defaults are used.
try:
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--judge', default=None,
                         help='Override judge model for all evaluated models')
    _args, _ = _parser.parse_known_args()
    JUDGE_OVERRIDE = _args.judge
except SystemExit:
    JUDGE_OVERRIDE = None

# ── Paths ──────────────────────────────────────────────────────────────────────
IMAGE_ROOT       = Path('images')          # source images (mirrors batch_query.ipynb's IMAGE_ROOT)
OUTPUT_ROOT      = Path('results')         # responses from the LLMs we're judging (per batch_query.ipynb runs)
TRUTH_ROOT       = Path('truth')           # human-authored ground truth
EVAL_OUTPUT_ROOT = Path('eval_output')     # this script's own output: score CSV + figures
EVAL_CSV         = EVAL_OUTPUT_ROOT / 'eval_scores.csv'  # cached judge results; re-run resumes from here

EVAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── Experiment config ──────────────────────────────────────────────────────────
RUN_IDS  = ['goodtest', 'badtest']   # goodtest = normal images, badtest = known-bad images
RUN_ID   = '_'.join(RUN_IDS)         # combined label — used in titles/output filenames below
# VARIANTS = ['no_ref', 'with_ref', 'with_goodref', 'with_refs']
VARIANTS = ['no_ref', 'with_ref', 'with_refs']
MODELS   = [
    'qwen2.5vl:latest',
    'qwen3-vl-longctx',
    'qwen3-vl-longctx32b',
    'gemma3:latest',
    'litellm-ow.google/gemma4-31b',
]
DELAY = 1.0   # seconds between judge calls

# Single independent judge for all models (text-only task, no self-judging risk)
JUDGE_MODEL = 'vllm.gpt-oss:120b'
JUDGE_MODEL_FOR = {m: [JUDGE_MODEL] for m in MODELS}

# Pairwise comparisons of interest:
#   version upgrade : qwen2.5vl vs qwen3-vl-longctx      (same size, next gen)
#   size scaling    : qwen3-vl-longctx vs qwen3-vl-longctx32b
#   architecture    : qwen3-vl-longctx vs gemma3:latest  (same size, different arch)
#   generational    : gemma3:latest vs gemma4-31b        (same family, next gen)
#   architecture    : qwen3-vl-longctx32b vs gemma4-31b  (similar size, ~31-32B)
PAIRWISE = [
    ('qwen2.5vl:latest',    'qwen3-vl-longctx',              'version upgrade (same size)'),
    ('qwen3-vl-longctx',    'qwen3-vl-longctx32b',           'size scaling (same arch)'),
    ('qwen3-vl-longctx',    'gemma3:latest',                 'architecture (same size)'),
    ('gemma3:latest',       'litellm-ow.google/gemma4-31b',  'generational upgrade (Gemma family)'),
    ('qwen3-vl-longctx32b', 'litellm-ow.google/gemma4-31b',  'architecture (similar size, ~31-32B)'),
]

SCORE_COLS      = ['s1_instruction_quote', 's2_plot_description',
                   's3_comparison', 's4_decision', 'overall']
SECTION_LABELS  = ['S1 instruction', 'S2 description',
                   'S3 comparison',  'S4 decision', 'Overall']

# %% ── Parse helpers ──────────────────────────────────────────────────────────
def parse_result_file(path: Path) -> dict:
    """Parse a batch_query result txt into metadata + response."""
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
    image_name = Path(meta['image']).name if meta.get('image') else ''
    image_path = (IMAGE_ROOT / plot / image_name) if image_name else None

    return {
        'model':                meta.get('model', ''),
        'plot':                 plot,
        'image':                meta.get('image', ''),        # raw string as recorded by batch_query
        'image_path':           str(image_path) if image_path else None,  # resolved via IMAGE_ROOT
        'run_id':               meta.get('run_id', ''),
        'prompt':               meta.get('prompt', ''),
        'load_latency_s':       _lat(meta.get('load_latency', '')),
        'generation_latency_s': _lat(meta.get('generation_latency', '')),
        'total_latency_s':      _lat(meta.get('total_latency', '')),
        'response':             response.strip(),
    }


def load_all_results(output_root: Path, run_ids: list, variants: list,
                     models: list | None = None) -> pd.DataFrame:
    """Load result txt files across run_ids × variants into a DataFrame.

    Parameters
    ----------
    run_ids : list of run-id prefixes, e.g. ['goodtest', 'badtest']. Each is
              combined with every variant as '<run_id>_<variant>'.
    models : if given, only files whose parsed model field is in this list
             are kept. Filters out old/retired model names.
    """
    rows = []
    for run_id in run_ids:
        for variant in variants:
            run_dir = output_root / f'{run_id}_{variant}'
            if not run_dir.exists():
                print(f'  WARNING: {run_dir} not found — skipping')
                continue

            for txt in sorted(run_dir.rglob('*.txt')):
                parsed = parse_result_file(txt)
                if models and parsed['model'] not in models:
                    continue  # file belongs to an old/retired model — skip
                parsed['run_id']  = run_id  # authoritative — which batch this came from
                parsed['variant'] = variant
                parsed['file']    = str(txt)
                parsed['mtime']   = txt.stat().st_mtime
                rows.append(parsed)

    df = pd.DataFrame(rows)
    if df.empty:
        print(f'  No matching result files found under {output_root}/<{"|".join(run_ids)}>_<variant>/ '
              f'for models={models}. Check that the directories exist and that the '
              f"'Model:' header in those .txt files matches an entry in MODELS.")
        return df

    # Deduplicate: same (model, variant, image) may exist from multiple batch
    # runs. Keep the most recently modified file.
    before = len(df)
    df = (
        df.sort_values('mtime')
          .drop_duplicates(subset=['model', 'variant', 'image'], keep='last')
          .drop(columns='mtime')
          .reset_index(drop=True)
    )
    if len(df) < before:
        print(f'  Deduplicated {before} → {len(df)} files '
              f'(dropped {before - len(df)} older duplicates)')

    print(f'Loaded {len(df)} result files across {df["variant"].nunique()} variants')
    print(f'  Model names found in files: {sorted(df["model"].unique())}')
    return df


def find_truth_file(truth_root: Path, plot: str, image_str: str) -> Path | None:
    """Find the truth file for a given image.

    Good images:  <...>_run<XXXXXX>.png            -> truth/<plot>/run<XXXXXX>.txt
    Bad images:   <...>_run<XXXXXX>_bad.png         -> truth/<plot>/truth_run<XXXXXX>_bad.txt
                  <...>_run<XXXXXX>_bad_small.png   -> truth/<plot>/truth_run<XXXXXX>_bad_small.txt
                  <...>_run<XXXXXX>_bad_medium.png  -> truth/<plot>/truth_run<XXXXXX>_bad_medium.txt

    Truth files are keyed by run number (+ bad-image tag), not by the full
    image filename — multiple image variants from the same run share one
    truth file. Note the good/bad naming is asymmetric (bad files carry a
    "truth_" prefix, good files don't) — that's the actual naming in use.
    "_bad_small"/"_bad_medium" are checked before plain "_bad" since both
    contain "_bad" as a substring.
    """
    stem  = Path(image_str).stem
    match = re.search(r'run\d+', stem)
    run   = match.group() if match else stem

    if '_bad_small' in stem:
        candidate = truth_root / plot / f'truth_{run}_bad_small.txt'
    elif '_bad_medium' in stem:
        candidate = truth_root / plot / f'truth_{run}_bad_medium.txt'
    elif '_bad' in stem:
        candidate = truth_root / plot / f'truth_{run}_bad.txt'
    else:
        candidate = truth_root / plot / f'{run}.txt'

    return candidate if candidate.exists() else None


# %% ── Judge setup ────────────────────────────────────────────────────────────
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


def judge(truth_text: str, response_text: str, model: str) -> dict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(truth=truth_text, response=response_text)
    # Judge is text-only: no image, no RAG — use an empty RAGConfig to skip retrieval.
    result = query(prompt, model=ModelConfig(name=model), system=JUDGE_SYSTEM,
                   rag=RAGConfig(backend="local"))
    if result.get('error'):
        return {'error': f"API error: {result['error']}"}
    response_raw = result.get('response') or ''
    if not response_raw:
        return {'error': 'empty response from judge model'}
    try:
        match = re.search(r'\{.*\}', response_raw, re.DOTALL)
        return json.loads(match.group()) if match else {'error': f'no JSON found in: {response_raw[:200]}'}
    except Exception as e:
        return {'error': str(e), 'raw': response_raw}


# %% ── Run evaluations (cached to EVAL_CSV) ───────────────────────────────────
df_results = load_all_results(OUTPUT_ROOT, RUN_IDS, VARIANTS, models=MODELS)

# Load existing cache so interrupted runs resume cleanly
if EVAL_CSV.exists():
    df_eval = pd.read_csv(EVAL_CSV)
    done    = set(zip(df_eval['file'], df_eval['judge_model']))
    print(f'Loaded {len(df_eval)} cached scores — resuming from where we left off')
else:
    df_eval = pd.DataFrame()
    done    = set()

eval_rows = []

for _, row in df_results.iterrows():
    if not row['response'] or not row['plot']:
        continue

    truth_file = find_truth_file(TRUTH_ROOT, row['plot'], row['image'])
    if not truth_file:
        print(f"  No truth for {row['plot']}/{Path(row['image']).name} — skipping")
        continue
    truth_text = truth_file.read_text().strip()

    # Resolve judge list: CLI override → default list → fallback to first model
    if JUDGE_OVERRIDE:
        judge_models = [JUDGE_OVERRIDE]
    else:
        judge_models = JUDGE_MODEL_FOR.get(row['model'], [MODELS[0]])

    for judge_model in judge_models:
        if judge_model == row['model']:
            continue  # never self-judge

        if (row['file'], judge_model) in done:
            continue  # already scored in a previous run

        print(
            f"  Judging: {row['model']:25s} | {row['variant']:12s} | "
            f"{Path(row['image']).name} | judge={judge_model} ...",
            end=' ', flush=True,
        )
        scores = judge(truth_text, row['response'], model=judge_model)

        if 'error' in scores:
            print(f"ERROR: {scores['error']}")
            continue
        print('ok')

        new_row = {
            'file':                 row['file'],
            'plot':                 row['plot'],
            'image':                Path(row['image']).name,
            'model':                row['model'],
            'variant':              row['variant'],
            'prompt':               row['prompt'],
            'judge_model':          judge_model,
            'generation_latency_s': row['generation_latency_s'],
            'total_latency_s':      row['total_latency_s'],
            **{k:              v['score']   if isinstance(v, dict) else None for k, v in scores.items()},
            **{f'{k}_comment': v['comment'] if isinstance(v, dict) else None for k, v in scores.items()},
        }
        eval_rows.append(new_row)
        done.add((row['file'], judge_model))  # guard against re-judging within this same run

        # Persist immediately — don't wait for the whole loop to finish. If a
        # later file/network error interrupts the run, everything judged so
        # far is already durably on disk and won't need to be re-judged
        # (and re-billed/re-called) on the next run.
        pd.DataFrame([new_row]).to_csv(
            EVAL_CSV, mode='a', header=not EVAL_CSV.exists(), index=False
        )
        time.sleep(DELAY)

# Fold this run's new rows into df_eval for the analysis below.
# (Already persisted to EVAL_CSV row-by-row above — no bulk rewrite needed.)
if eval_rows:
    df_eval = pd.concat([df_eval, pd.DataFrame(eval_rows)], ignore_index=True)
    print(f'\nSaved {len(eval_rows)} new score(s) → {EVAL_CSV} (total {len(df_eval)})')

if df_eval.empty:
    raise RuntimeError(
        'No evaluation scores produced. Check that:\n'
        f'  1. Result files exist under results/<{"|".join(RUN_IDS)}>_<variant>/\n'
        '  2. Truth files exist under truth/<plot>/run<XXXXXX>.txt (good) or\n'
        '     truth/<plot>/truth_run<XXXXXX>_bad[_small|_medium].txt (bad)\n'
        '  3. Judge model names are correct (run list_models() to verify)\n'
        f'  4. JUDGE_OVERRIDE={JUDGE_OVERRIDE!r}, JUDGE_MODEL_FOR={JUDGE_MODEL_FOR}'
    )

# Drop rows scored by a judge that's no longer in use. EVAL_CSV is append-only
# (the cache key is (file, judge_model)), so switching JUDGE_MODEL doesn't
# remove old scores — it just adds new ones alongside them. Without this
# filter, a stale judge's rows would silently blend into every groupby('model')
# average below and spuriously trigger the inter-judge agreement section.
active_judges = {m for models in JUDGE_MODEL_FOR.values() for m in models}
stale = df_eval[~df_eval['judge_model'].isin(active_judges)]
if not stale.empty:
    print(
        f"Ignoring {len(stale)} cached score(s) from judge(s) no longer in use: "
        f"{sorted(stale['judge_model'].unique())} (still in {EVAL_CSV}, just excluded from this analysis)"
    )
df_eval = df_eval[df_eval['judge_model'].isin(active_judges)].reset_index(drop=True)

df_eval['variant'] = pd.Categorical(df_eval['variant'], categories=VARIANTS, ordered=True)
df_eval['s4_correct'] = (df_eval['s4_decision'] >= 3)  # rubric: 3-5 = correct verdict, 1-2 = wrong verdict
# 'good' vs 'bad' is read off the image filename itself (contains "_bad" for
# any of the bad/bad_small/bad_medium variants), not off run_id/variant, so
# it's correct regardless of which directory a file happened to load from.
df_eval['image_quality'] = np.where(
    df_eval['image'].astype(str).apply(lambda s: '_bad' in Path(s).stem),
    'bad', 'good',
)
judge_models = sorted(df_eval['judge_model'].unique())
display(df_eval[['model', 'variant', 'image', 'image_quality'] + SCORE_COLS])


# %% ── Reference-variant comparison: find the optimal prompt variant ─────────
DECISION_COL   = 's1_instruction_quote'
DECISION_LABEL = 'Section 1 (instruction quote)'

print('=' * 70)
print(f'PROMPT-VARIANT COMPARISON: {" vs ".join(VARIANTS)}')
print(f'(decision based on {DECISION_LABEL})')
print('=' * 70)

REF_VARIANTS = VARIANTS
have_all = all(v in df_eval['variant'].unique() for v in REF_VARIANTS)

if not have_all:
    found = sorted(str(v) for v in df_eval['variant'].dropna().unique())
    print(f'  Need all of {REF_VARIANTS} to compare — found: {found}.')
    print('  Falling back to using all available variants for benchmarking.')
    BENCH_VARIANT = None
else:
    ref_pivot = (
        df_eval[df_eval['variant'].isin(REF_VARIANTS)]
        .groupby(['model', 'image', 'variant'])[DECISION_COL].mean()
        .unstack('variant')
        .reindex(columns=REF_VARIANTS)
        .dropna(subset=REF_VARIANTS)
    )
    if not ref_pivot.empty:
        ref_pivot['best_variant'] = ref_pivot[REF_VARIANTS].idxmax(axis=1)
        n_total    = len(ref_pivot)
        win_counts = ref_pivot['best_variant'].value_counts().reindex(REF_VARIANTS, fill_value=0)
        print(f'  Best variant per model×image, on {DECISION_LABEL} ({n_total} combinations):')
        for v in REF_VARIANTS:
            print(f'    {v}: wins {win_counts[v]}/{n_total} ({win_counts[v] / n_total * 100:.0f}%)')
        display(ref_pivot.round(2))

    per_model = (
        df_eval[df_eval['variant'].isin(REF_VARIANTS)]
        .groupby(['model', 'variant'])[DECISION_COL].mean()
        .unstack('variant').reindex(index=MODELS, columns=REF_VARIANTS)
    )
    print(f'\n  Per-model average "{DECISION_COL}" score:')
    display(per_model.round(2))

    decision_avg = (
        df_eval[df_eval['variant'].isin(REF_VARIANTS)]
        .groupby('variant')[DECISION_COL].mean()
        .reindex(REF_VARIANTS)
    )
    print(f'\n  Average {DECISION_LABEL} score by variant:')
    for v in REF_VARIANTS:
        val = decision_avg.get(v, float('nan'))
        print(f'    {v}: {val:.3f}' if pd.notna(val) else f'    {v}: (no data)')

    if decision_avg.notna().any():
        BENCH_VARIANT = decision_avg.idxmax()
        print(f'\n✓ "{BENCH_VARIANT}" has the highest average {DECISION_LABEL} score — using it for benchmarking.')
    else:
        BENCH_VARIANT = None
        print(f'\n  No variant has {DECISION_LABEL} data — falling back to all variants for benchmarking.')

df_bench = (
    df_eval[df_eval['variant'] == BENCH_VARIANT].copy()
    if BENCH_VARIANT else df_eval.copy()
)
bench_label = f'prompt: {BENCH_VARIANT}' if BENCH_VARIANT else 'all variants'
print(f'\nVariant used for benchmarking: {bench_label}')
print('=' * 70)

# Bad-image responses test a different capability (correctly catching a known
# problem) than good-image responses (correctly clearing a normal plot), so
# every combined/pairwise model-comparison plot below is produced for both
# scopes: all images (good+bad, unchanged filenames/behavior) and good images
# only (new "_good"-suffixed files). Note BENCH_VARIANT itself is decided once
# above from all images — it is not re-decided per scope.
ANALYSIS_SCOPES = {
    'all_images': df_bench,
    'good_only':  df_bench[df_bench['image_quality'] == 'good'],
}


# %% ── Score tables ───────────────────────────────────────────────────────────
print('\n=== Mean scores by model (' + bench_label + ') ===')
display(
    df_bench.groupby('model')[SCORE_COLS]
    .mean().round(2).reindex(MODELS)
)

print('\n=== Mean scores by model × variant (full view) ===')
display(
    df_eval.groupby(['model', 'variant'])[SCORE_COLS]
    .mean().round(2)
)

print('\n=== S4 decision accuracy — benchmarking prompt ===')
display(
    df_bench.groupby('model')['s4_correct']
    .mean().mul(100).round(1).rename('% correct')
    .reindex(MODELS).to_frame()
)

print('\n=== Generation latency (s) by model — benchmarking prompt ===')
display(
    df_bench.groupby('model')['generation_latency_s']
    .mean().round(2).rename('gen latency (s)')
    .reindex(MODELS).to_frame()
)

# %% ── Pairwise model comparisons (data prep) ────────────────────────────────
# Uses df_bench (the chosen benchmarking variant). Built once per ANALYSIS_SCOPE
# (all images, good images only) so pairwise plots can show both.

_PAIR_COLUMNS = ['comparison', 'model_a', 'model_b', 'section', 'score_a', 'score_b', 'diff']

def build_pair_rows(df_subset, verbose=False):
    rows = []
    for model_a, model_b, label in PAIRWISE:
        a = df_subset[df_subset['model'] == model_a][SCORE_COLS].mean()
        b = df_subset[df_subset['model'] == model_b][SCORE_COLS].mean()
        if a.isna().all() or b.isna().all():
            if verbose:
                print(f'  Skipping "{label}" — one or both models have no scores yet')
            continue
        for col in SCORE_COLS:
            rows.append({
                'comparison': label, 'model_a': model_a, 'model_b': model_b,
                'section': col, 'score_a': a[col], 'score_b': b[col], 'diff': b[col] - a[col],
            })
    return pd.DataFrame(rows, columns=_PAIR_COLUMNS)  # explicit columns so an
    # all-skipped run (e.g. too few models present) doesn't crash downstream

df_pairs_by_scope = {}
for scope_name, df_scope in ANALYSIS_SCOPES.items():
    df_pairs_by_scope[scope_name] = build_pair_rows(df_scope, verbose=(scope_name == 'all_images'))
df_pairs = df_pairs_by_scope['all_images']  # kept for backward compatibility with anything referencing df_pairs directly

# %% ── Inter-judge agreement ──────────────────────────────────────────────────
if len(judge_models) > 1:
    print('=== Inter-judge agreement (mean absolute score difference per section) ===')

    agree_rows = []
    for (f, m, v), grp in df_eval.groupby(['file', 'model', 'variant']):
        if grp['judge_model'].nunique() < 2:
            continue
        for col in SCORE_COLS:
            scores_by_judge = grp.set_index('judge_model')[col].dropna()
            if len(scores_by_judge) < 2:
                continue
            pairs = [
                abs(scores_by_judge.iloc[i] - scores_by_judge.iloc[j])
                for i in range(len(scores_by_judge))
                for j in range(i + 1, len(scores_by_judge))
            ]
            agree_rows.append({'model': m, 'variant': v, 'section': col,
                                'mean_abs_diff': np.mean(pairs)})

    if agree_rows:
        df_agree = pd.DataFrame(agree_rows)
        display(
            df_agree.groupby('section')['mean_abs_diff']
            .mean().round(2).rename('mean |Δscore| across judges')
            .reindex(SCORE_COLS)
        )

        print('\n=== S4 judge agreement: cases where judges disagree on Good/Bad ===')
        s4_pivot = (
            df_eval[df_eval['s4_decision'].notna()]
            .groupby(['file', 'model', 'variant', 'judge_model'])['s4_decision']
            .first().unstack('judge_model')
        )
        disagree = s4_pivot[s4_pivot.nunique(axis=1) > 1]
        if disagree.empty:
            print('  All judges agree on S4 for every response.')
        else:
            print(f'  {len(disagree)} response(s) with S4 disagreement:')
            display(disagree.reset_index()[['model', 'variant', 'file'] + judge_models])
    else:
        print('  Not enough overlapping judge scores to compute agreement.')


# %% ── Plotting helpers ───────────────────────────────────────────────────────
def short_name(model: str) -> str:
    """Shorten a model id for use as an axis/legend label."""
    name = model.split(':')[0].replace('qwen3-vl-longctx', 'qwen3-longctx')
    return name.split('/')[-1]  # drop any "litellm-ow.xxx/" provider prefix


def plot_section_scores(df_subset, models, title, out_path):
    """Heatmap of mean S1-S4/overall scores, one row per model."""
    pivot = df_subset.groupby('model')[SCORE_COLS].mean().reindex(models)
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(models) + 1.5)))
    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=1, vmax=5, aspect='auto')
    ax.set_xticks(range(len(SCORE_COLS)))
    ax.set_xticklabels(SECTION_LABELS, rotation=20, ha='right', fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short_name(m) for m in models], fontsize=9)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax)
    for i in range(len(models)):
        for j in range(len(SCORE_COLS)):
            val = pivot.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_s4_accuracy(df_subset, models, title, out_path):
    """Bar chart of S4 (Good/Bad) decision accuracy per model."""
    acc = df_subset.groupby('model')['s4_correct'].mean().mul(100).reindex(models)
    colors_m = plt.cm.Set1(np.linspace(0, 1, len(models)))
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(models)), 4))
    ax.bar(range(len(models)), acc.values, color=colors_m)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([short_name(m) for m in models], rotation=20, ha='right', fontsize=8)
    ax.set_ylim(0, 110)
    ax.axhline(100, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('% correct')
    for i, v in enumerate(acc.values):
        if pd.notna(v):
            ax.text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_generation_latency(df_subset, models, title, out_path):
    """Bar chart of mean generation latency (s) per model."""
    lat = df_subset.groupby('model')['generation_latency_s'].mean().reindex(models)
    colors_m = plt.cm.Set1(np.linspace(0, 1, len(models)))
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(models)), 4))
    ax.bar(range(len(models)), lat.values, color=colors_m)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([short_name(m) for m in models], rotation=20, ha='right', fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('seconds')
    max_lat = np.nanmax(lat.values) if np.any(~np.isnan(lat.values.astype(float))) else 1.0
    for i, v in enumerate(lat.values):
        if pd.notna(v):
            ax.text(i, v + max_lat * 0.02, f'{v:.1f}s', ha='center', fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# %% ── Visualisations ─────────────────────────────────────────────────────────
COMBINED_DIR  = EVAL_OUTPUT_ROOT / 'combined'
PER_IMAGE_DIR = EVAL_OUTPUT_ROOT / 'per_image'
PAIRWISE_DIR  = COMBINED_DIR / 'pairwise'
COMBINED_DIR.mkdir(parents=True, exist_ok=True)
PER_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
PAIRWISE_DIR.mkdir(parents=True, exist_ok=True)

if len(judge_models) > 1:
    fig_j, ax_j = plt.subplots(figsize=(10, 4))
    judge_pivot = (
        df_eval.groupby(['judge_model', 'variant'])['overall']
        .mean().unstack('variant').reindex(columns=VARIANTS)
    )
    judge_pivot.T.plot(kind='bar', ax=ax_j, colormap='tab10', width=0.7)
    ax_j.set_title('Overall score by judge model × variant')
    ax_j.set_ylim(1, 5)
    ax_j.set_xlabel('')
    ax_j.tick_params(axis='x', rotation=20)
    ax_j.legend(title='judge', fontsize=8)
    plt.tight_layout()
    out_judge = COMBINED_DIR / f'eval_{RUN_ID}_by_judge.png'
    plt.savefig(out_judge, dpi=150, bbox_inches='tight')
    plt.close(fig_j)
    print(f'Saved: {out_judge}')

# BENCH_VARIANT itself was decided once, from all images (see above) — these
# figures don't redecide it per scope, they just show what the underlying
# scores look like when bad images are excluded vs included.
def plot_prompt_comparison(df_subset, title_suffix, out_path):
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

    pivot = (
        df_subset.groupby(['model', 'variant'])[DECISION_COL]
        .mean().unstack('variant').reindex(columns=VARIANTS, index=MODELS)
    )
    im = axes1[0].imshow(pivot.values, cmap='RdYlGn', vmin=1, vmax=5, aspect='auto')
    axes1[0].set_xticks(range(len(VARIANTS)))
    axes1[0].set_xticklabels(VARIANTS, rotation=20, ha='right', fontsize=9)
    axes1[0].set_yticks(range(len(MODELS)))
    axes1[0].set_yticklabels([short_name(m) for m in MODELS], fontsize=9)
    axes1[0].set_title(f'{DECISION_LABEL} score by model × prompt variant\n(this is what the blue-box decision is based on)')
    plt.colorbar(im, ax=axes1[0])
    for i in range(len(MODELS)):
        for j in range(len(VARIANTS)):
            val = pivot.values[i, j]
            if pd.notna(val):
                axes1[0].text(j, i, f'{val:.1f}', ha='center', va='center',
                              fontweight='bold', fontsize=11)
    # Highlight the variant chosen for benchmarking, across every model row
    if BENCH_VARIANT in VARIANTS:
        j_bench = VARIANTS.index(BENCH_VARIANT)
        for i in range(len(MODELS)):
            axes1[0].add_patch(plt.Rectangle((j_bench - 0.5, i - 0.5), 1, 1,
                               fill=False, edgecolor='blue', linewidth=2))

    ax_r = fig1.add_subplot(1, 2, 2, polar=True)
    section_cols = SCORE_COLS[:-1]
    N      = len(section_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    colors_v = plt.cm.Set2(np.linspace(0, 1, len(VARIANTS)))
    for vi, variant in enumerate(VARIANTS):
        vals = df_subset[df_subset['variant'] == variant][section_cols].mean().tolist()
        vals += vals[:1]
        ax_r.plot(angles, vals, 'o-', label=variant, color=colors_v[vi], linewidth=2)
        ax_r.fill(angles, vals, alpha=0.07, color=colors_v[vi])
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(['S1\ninstruction', 'S2\ndescription',
                           'S3\ncomparison',  'S4\ndecision'], fontsize=9)
    ax_r.set_ylim(1, 5)
    ax_r.set_title('Per-section scores by variant\n(avg across models)', pad=15, fontsize=9)
    ax_r.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15), fontsize=8)

    fig1.suptitle(f'DQM Prompt Comparison — {RUN_ID}{title_suffix} (blue box = variant used for benchmarking)',
                  fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f'Saved: {out_path}')

plot_prompt_comparison(df_eval, '', COMBINED_DIR / f'eval_{RUN_ID}_prompts.png')
plot_prompt_comparison(df_eval[df_eval['image_quality'] == 'good'], ' (good images only)',
                        COMBINED_DIR / f'eval_{RUN_ID}_prompts_good.png')

plot_section_scores(ANALYSIS_SCOPES['all_images'], MODELS,
                     f'Section scores by model — all images ({bench_label})',
                     COMBINED_DIR / f'section_scores_{RUN_ID}.png')
plot_section_scores(ANALYSIS_SCOPES['good_only'], MODELS,
                     f'Section scores by model — good images only ({bench_label})',
                     COMBINED_DIR / f'section_scores_{RUN_ID}_good.png')

plot_generation_latency(ANALYSIS_SCOPES['all_images'], MODELS,
                         f'Generation latency — all images ({bench_label})',
                         COMBINED_DIR / f'generation_latency_{RUN_ID}.png')
plot_generation_latency(ANALYSIS_SCOPES['good_only'], MODELS,
                         f'Generation latency — good images only ({bench_label})',
                         COMBINED_DIR / f'generation_latency_{RUN_ID}_good.png')

# S4 accuracy: good images test "correctly say good", bad images test
# "correctly catch the problem" — these are different capabilities, so each
# gets its own plot in addition to the combined view.
plot_s4_accuracy(df_bench, MODELS,
                  f'S4 decision accuracy — all images, good+bad combined ({bench_label})',
                  COMBINED_DIR / f's4_accuracy_{RUN_ID}.png')
plot_s4_accuracy(df_bench[df_bench['image_quality'] == 'good'], MODELS,
                  f'S4 decision accuracy — good images only ({bench_label})',
                  COMBINED_DIR / f's4_accuracy_{RUN_ID}_good.png')
plot_s4_accuracy(df_bench[df_bench['image_quality'] == 'bad'], MODELS,
                  f'S4 decision accuracy — bad images only ({bench_label})',
                  COMBINED_DIR / f's4_accuracy_{RUN_ID}_bad.png')
print(f'Saved combined section-score / S4-accuracy (combined+good+bad) / latency plots to {COMBINED_DIR}/')

image_names = sorted(df_bench['image'].dropna().unique())
for img_name in image_names:
    sub = df_bench[df_bench['image'] == img_name]
    img_dir = PER_IMAGE_DIR / Path(img_name).stem
    plot_section_scores(sub, MODELS, f'Section scores by model — {img_name} ({bench_label})',
                         img_dir / 'section_scores.png')
    plot_s4_accuracy(sub, MODELS, f'S4 decision accuracy — {img_name} ({bench_label})',
                      img_dir / 's4_accuracy.png')
    plot_generation_latency(sub, MODELS, f'Generation latency — {img_name} ({bench_label})',
                             img_dir / 'generation_latency.png')
print(f'Saved {len(image_names)} per-image plot sets to {PER_IMAGE_DIR}/<image>/')

# %% ── Pairwise comparison plot (redesigned) ──────────────────────────────────
# Grouped bars of each model's actual scores per section — easier to read than
# a raw score-difference chart — plus a plain-language verdict printed to the
# console and drawn under the plot itself. Produced for both scopes: all
# images (unchanged filename) and good images only ("_good"-suffixed file).
def plot_pairwise(df_pairs_subset, filename_suffix, title_suffix):
    for model_a, model_b, label in PAIRWISE:
        sub = df_pairs_subset[df_pairs_subset['comparison'] == label]
        if sub.empty:
            continue  # already reported as skipped during data prep above

        sub = sub.set_index('section').reindex(SCORE_COLS)
        a_scores = sub['score_a'].values.astype(float)
        b_scores = sub['score_b'].values.astype(float)

        x = np.arange(len(SCORE_COLS))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 5.5))
        bars_a = ax.bar(x - width / 2, a_scores, width, label=short_name(model_a), color='#4C72B0')
        bars_b = ax.bar(x + width / 2, b_scores, width, label=short_name(model_b), color='#DD8452')
        ax.set_xticks(x)
        ax.set_xticklabels(SECTION_LABELS, rotation=15, ha='right', fontsize=9)
        ax.set_ylim(0, 5.8)
        ax.set_ylabel('Mean score (1–5)')
        ax.legend()
        for bars in (bars_a, bars_b):
            for rect in bars:
                h = rect.get_height()
                if pd.notna(h):
                    ax.text(rect.get_x() + rect.get_width() / 2, h + 0.08, f'{h:.1f}',
                            ha='center', fontsize=8)

        overall_a = sub.loc['overall', 'score_a']
        overall_b = sub.loc['overall', 'score_b']
        wins_b = int((sub['score_b'] > sub['score_a']).sum())
        wins_a = int((sub['score_a'] > sub['score_b']).sum())

        if pd.notna(overall_a) and pd.notna(overall_b) and overall_b > overall_a:
            verdict = (f'{short_name(model_b)} performs better overall '
                       f'(+{overall_b - overall_a:.2f} on "overall"; wins {wins_b}/{len(SCORE_COLS)} sections)')
        elif pd.notna(overall_a) and pd.notna(overall_b) and overall_a > overall_b:
            verdict = (f'{short_name(model_a)} performs better overall '
                       f'(+{overall_a - overall_b:.2f} on "overall"; wins {wins_a}/{len(SCORE_COLS)} sections)')
        else:
            verdict = 'Models tie on overall score'

        ax.set_title(f'{label}{title_suffix}\nA={short_name(model_a)}  B={short_name(model_b)}',
                      fontsize=10, fontweight='bold')
        fig.text(0.5, -0.04, verdict, ha='center', fontsize=9.5, style='italic')

        print(f'  {label}{title_suffix}: {verdict}')

        plt.tight_layout()
        safe_label = re.sub(r'[^A-Za-z0-9_-]+', '_', label.replace(' ', '_')).strip('_')
        out_pair = PAIRWISE_DIR / f'{safe_label}{filename_suffix}.png'
        plt.savefig(out_pair, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out_pair}')

print('=== Pairwise comparisons — all images ===')
plot_pairwise(df_pairs_by_scope['all_images'], '', '')
print('=== Pairwise comparisons — good images only ===')
plot_pairwise(df_pairs_by_scope['good_only'], '_good', ' (good images only)')
