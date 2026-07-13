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
#   truth/
#       <plot>/<image_stem>_truth.txt

# %% ── Imports & config ───────────────────────────────────────────────────────
import re
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from owui_client import query

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
OUTPUT_ROOT = Path('results')
TRUTH_ROOT  = Path('results/truth')
EVAL_CSV    = Path('eval_scores.csv')   # cached judge results; re-run resumes from here

# ── Experiment config ──────────────────────────────────────────────────────────
RUN_ID   = 'baseline'
VARIANTS = ['no_ref', 'with_ref', 'with_goodref', 'with_refs']
MODELS   = [
    'qwen2.5vl:latest',
    'qwen3-vl-longctx',
    'qwen3-vl-longctx32b',
    'gemma3:latest',
]
DELAY = 1.0   # seconds between judge calls

# Single independent judge for all models (text-only task, no self-judging risk)
JUDGE_MODEL = 'hopper.openai/gpt-oss-120b'
JUDGE_MODEL_FOR = {m: [JUDGE_MODEL] for m in MODELS}

# Pairwise comparisons of interest:
#   version upgrade : qwen2.5vl vs qwen3-vl-longctx  (same size, next gen)
#   size scaling    : qwen3-vl-longctx vs qwen3-vl-longctx32b
#   architecture    : qwen3-vl-longctx vs gemma3:latest  (same size, different arch)
PAIRWISE = [
    ('qwen2.5vl:latest',  'qwen3-vl-longctx',    'version upgrade (same size)'),
    ('qwen3-vl-longctx',  'qwen3-vl-longctx32b', 'size scaling (same arch)'),
    ('qwen3-vl-longctx',  'gemma3:latest',        'architecture (same size)'),
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

    return {
        'model':                meta.get('model', ''),
        'plot':                 meta.get('plot', ''),
        'image':                meta.get('image', ''),
        'run_id':               meta.get('run_id', ''),
        'prompt':               meta.get('prompt', ''),
        'load_latency_s':       _lat(meta.get('load_latency', '')),
        'generation_latency_s': _lat(meta.get('generation_latency', '')),
        'total_latency_s':      _lat(meta.get('total_latency', '')),
        'response':             response.strip(),
    }


def load_all_results(output_root: Path, run_id: str, variants: list,
                     models: list | None = None) -> pd.DataFrame:
    """Load result txt files across variants into a DataFrame.

    Parameters
    ----------
    models : if given, only files whose parsed model field is in this list
             are kept. Filters out old/retired model names.
    """
    rows = []
    for variant in variants:
        run_dir = output_root / f'{run_id}_{variant}'
        if not run_dir.exists():
            print(f'  WARNING: {run_dir} not found — skipping')
            continue
        # ── WORKAROUND START ──────────────────────────────────────────────────
        # OpenWebUI returns a normalised model name in the SSE stream (e.g.
        # "qwen3-vl:latest") that differs from the requested name stored in
        # the filename (e.g. "qwen3-vl-longctx"). Until owui_client.py is
        # fixed to write the requested model name to the header, we derive
        # the model from the filename instead of the Model: header.
        # Remove this block (and the model_from_file references below) once
        # owui_client.py writes `model` (not `model_used`) to the txt header.
        safe_to_model = {
            m.replace('/', '_').replace(':', '_'): m
            for m in (models or [])
        }
        # ── WORKAROUND END ────────────────────────────────────────────────────

        for txt in sorted(run_dir.rglob('*.txt')):
            parsed = parse_result_file(txt)

            # ── WORKAROUND START ──────────────────────────────────────────────
            model_from_file = None
            for safe, original in safe_to_model.items():
                if txt.stem.endswith('_' + safe):
                    model_from_file = original
                    break
            # ── WORKAROUND END ────────────────────────────────────────────────

            if models:
                if model_from_file is None:
                    continue  # file belongs to an old/retired model — skip
                parsed['model'] = model_from_file  # WORKAROUND: remove once owui_client.py is fixed
            parsed['variant'] = variant
            parsed['file']    = str(txt)
            parsed['mtime']   = txt.stat().st_mtime
            rows.append(parsed)

    df = pd.DataFrame(rows)

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
    """Find truth/<plot>/<image_stem>_truth.txt"""
    stem      = Path(image_str).stem
    candidate = truth_root / plot / f'{stem}_truth.txt'
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
    result = query(prompt, model=model, system=JUDGE_SYSTEM)
    if result.get('error'):
        return {'error': f"API error: {result['error']}"}
    response_text = result.get('response') or ''
    if not response_text:
        return {'error': 'empty response from judge model'}
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        return json.loads(match.group()) if match else {'error': f'no JSON found in: {response_text[:200]}'}
    except Exception as e:
        return {'error': str(e), 'raw': response_text}


# %% ── Run evaluations (cached to EVAL_CSV) ───────────────────────────────────
df_results = load_all_results(OUTPUT_ROOT, RUN_ID, VARIANTS, models=MODELS)

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

        eval_rows.append({
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
        })
        time.sleep(DELAY)

# Merge with cache and persist
if eval_rows:
    df_eval = pd.concat([df_eval, pd.DataFrame(eval_rows)], ignore_index=True)
    df_eval.to_csv(EVAL_CSV, index=False)
    print(f'\nSaved {len(df_eval)} total scores → {EVAL_CSV}')

if df_eval.empty:
    raise RuntimeError(
        'No evaluation scores produced. Check that:\n'
        '  1. Result files exist under results/<run_id>_<variant>/\n'
        '  2. Truth files exist under truth/<plot>/<image_stem>_truth.txt\n'
        '  3. Judge model names are correct (run list_models() to verify)\n'
        f'  4. JUDGE_OVERRIDE={JUDGE_OVERRIDE!r}, JUDGE_MODEL_FOR={JUDGE_MODEL_FOR}'
    )

df_eval['variant'] = pd.Categorical(df_eval['variant'], categories=VARIANTS, ordered=True)
df_eval['s4_correct'] = (df_eval['s4_decision'] == 5)
judge_models = sorted(df_eval['judge_model'].unique())
display(df_eval[['model', 'variant', 'image'] + SCORE_COLS])


# %% ── Prompt performance analysis ───────────────────────────────────────────
# For each model, find which prompt variant produces the highest overall score.
# If one variant wins across all models, use it for all subsequent analysis.

print('=' * 70)
print('PROMPT PERFORMANCE ANALYSIS')
print('=' * 70)

variant_scores = (
    df_eval.groupby(['model', 'variant'])['overall']
    .mean().round(2).unstack('variant').reindex(columns=VARIANTS)
)
display(variant_scores)

best_per_model = {}
for model in MODELS:
    if model not in variant_scores.index:
        continue
    row = variant_scores.loc[model].dropna()
    if row.empty:
        continue
    best_variant = row.idxmax()
    best_score   = row.max()
    worst_variant = row.idxmin()
    worst_score   = row.min()
    best_per_model[model] = best_variant
    print(
        f'  {model:30s}  best={best_variant} ({best_score:.2f})'
        f'  worst={worst_variant} ({worst_score:.2f})'
        f'  Δ={best_score - worst_score:.2f}'
    )

# Check if one variant is best across all models
best_counts = {}
for v in best_per_model.values():
    best_counts[v] = best_counts.get(v, 0) + 1

consensus_variant = None
if best_counts:
    top_variant, top_count = max(best_counts.items(), key=lambda x: x[1])
    if top_count == len(best_per_model):
        consensus_variant = top_variant
        print(f'\n✓ Consensus: "{consensus_variant}" is the best prompt for ALL models.')
        print(  f'  Using only "{consensus_variant}" for subsequent benchmarking.\n')
    else:
        print(f'\n✗ No single best prompt across all models.')
        print(  f'  Best counts: {best_counts}')
        print(  f'  Proceeding with all variants for subsequent benchmarking.\n')

# Filter df_eval for downstream analysis if there is a consensus
df_bench = (
    df_eval[df_eval['variant'] == consensus_variant].copy()
    if consensus_variant else df_eval.copy()
)
BENCH_VARIANTS = [consensus_variant] if consensus_variant else VARIANTS
print(f'Variants used for benchmarking: {BENCH_VARIANTS}')
print('=' * 70)


# %% ── Score tables ───────────────────────────────────────────────────────────
print('\n=== Mean scores by model (prompt: ' + (consensus_variant or 'all variants') + ') ===')
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

# %% ── Pairwise model comparisons ────────────────────────────────────────────
# Uses df_bench (consensus prompt if found, else all variants).

print('=== Pairwise score differences (B − A, positive = B is better) ===\n')
pair_rows = []

for model_a, model_b, label in PAIRWISE:
    a = df_bench[df_bench['model'] == model_a].groupby('variant')[SCORE_COLS].mean()
    b = df_bench[df_bench['model'] == model_b].groupby('variant')[SCORE_COLS].mean()

    if a.empty or b.empty:
        print(f'  Skipping "{label}" — one or both models have no scores yet')
        continue

    diff = (b - a).reindex(VARIANTS)
    mean_diff = diff.mean()

    print(f'  {label}')
    print(f'  A = {model_a}')
    print(f'  B = {model_b}')
    display(diff.round(2))
    print(f'  Mean across variants: {mean_diff.round(2).to_dict()}\n')

    for variant in VARIANTS:
        if variant in diff.index:
            for col in SCORE_COLS:
                pair_rows.append({
                    'comparison': label,
                    'model_a':    model_a,
                    'model_b':    model_b,
                    'variant':    variant,
                    'section':    col,
                    'diff':       diff.loc[variant, col],
                })

df_pairs = pd.DataFrame(pair_rows)

# %% ── Inter-judge agreement ──────────────────────────────────────────────────
# Only meaningful when more than one judge scored the same response.
if len(judge_models) > 1:
    print('=== Inter-judge agreement (mean absolute score difference per section) ===')

    # Pivot so each row is one (file, model, variant) and columns are judge scores
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


# %% ── Visualisations ─────────────────────────────────────────────────────────
models = df_eval['model'].unique()
colors = plt.cm.Set2(np.linspace(0, 1, len(VARIANTS)))

# ── Extra figure: overall score by judge model ───────────────────────────────
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
    out_judge = f'eval_{RUN_ID}_by_judge.png'
    plt.savefig(out_judge, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {out_judge}')

# ── Figure 1: Prompt comparison (always uses all variants) ───────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

# (A) Overall score heatmap: model × variant
pivot = (
    df_eval.groupby(['model', 'variant'])['overall']
    .mean().unstack('variant').reindex(columns=VARIANTS, index=MODELS)
)
im = axes1[0].imshow(pivot.values, cmap='RdYlGn', vmin=1, vmax=5, aspect='auto')
axes1[0].set_xticks(range(len(VARIANTS)))
axes1[0].set_xticklabels(VARIANTS, rotation=20, ha='right', fontsize=9)
axes1[0].set_yticks(range(len(MODELS)))
axes1[0].set_yticklabels(MODELS, fontsize=9)
axes1[0].set_title('Overall score by model × prompt variant')
plt.colorbar(im, ax=axes1[0])
for i in range(len(MODELS)):
    for j in range(len(VARIANTS)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            axes1[0].text(j, i, f'{val:.1f}', ha='center', va='center',
                          fontweight='bold', fontsize=11)
        # Highlight best variant per model
        if not np.isnan(val) and MODELS[i] in best_per_model and VARIANTS[j] == best_per_model[MODELS[i]]:
            axes1[0].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                               fill=False, edgecolor='blue', linewidth=2))

# (B) Radar chart: per-section by variant
ax_r = axes1[1]
ax_r = fig1.add_subplot(1, 2, 2, polar=True)
section_cols = SCORE_COLS[:-1]
N      = len(section_cols)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]
colors_v = plt.cm.Set2(np.linspace(0, 1, len(VARIANTS)))
for vi, variant in enumerate(VARIANTS):
    vals = df_eval[df_eval['variant'] == variant][section_cols].mean().tolist()
    vals += vals[:1]
    ax_r.plot(angles, vals, 'o-', label=variant, color=colors_v[vi], linewidth=2)
    ax_r.fill(angles, vals, alpha=0.07, color=colors_v[vi])
ax_r.set_xticks(angles[:-1])
ax_r.set_xticklabels(['S1\ninstruction', 'S2\ndescription',
                       'S3\ncomparison',  'S4\ndecision'], fontsize=9)
ax_r.set_ylim(1, 5)
ax_r.set_title('Per-section scores by variant\n(avg across models)', pad=15, fontsize=9)
ax_r.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15), fontsize=8)

bench_label = f'prompt: {consensus_variant}' if consensus_variant else 'all variants'
fig1.suptitle(f'DQM Prompt Comparison — {RUN_ID}', fontsize=13, fontweight='bold')
plt.tight_layout()
out_prompt = f'eval_{RUN_ID}_prompts.png'
plt.savefig(out_prompt, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {out_prompt}')

# ── Figure 2: Model benchmarking (uses df_bench = consensus prompt or all) ───
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

# (A) Per-section heatmap: model × section, using bench prompt
pivot_bench = (
    df_bench.groupby('model')[SCORE_COLS]
    .mean().reindex(MODELS)
)
im2 = axes2[0].imshow(pivot_bench.values, cmap='RdYlGn', vmin=1, vmax=5, aspect='auto')
axes2[0].set_xticks(range(len(SCORE_COLS)))
axes2[0].set_xticklabels(SECTION_LABELS, rotation=20, ha='right', fontsize=8)
axes2[0].set_yticks(range(len(MODELS)))
axes2[0].set_yticklabels(MODELS, fontsize=9)
axes2[0].set_title(f'Section scores by model\n({bench_label})')
plt.colorbar(im2, ax=axes2[0])
for i in range(len(MODELS)):
    for j in range(len(SCORE_COLS)):
        val = pivot_bench.values[i, j]
        if not np.isnan(val):
            axes2[0].text(j, i, f'{val:.1f}', ha='center', va='center',
                          fontsize=9, fontweight='bold')

# (B) S4 decision accuracy
acc_bench = (
    df_bench.groupby('model')['s4_correct']
    .mean().mul(100).reindex(MODELS)
)
colors_m = plt.cm.Set1(np.linspace(0, 1, len(MODELS)))
axes2[1].bar(range(len(MODELS)), acc_bench.values, color=colors_m)
axes2[1].set_xticks(range(len(MODELS)))
axes2[1].set_xticklabels(MODELS, rotation=20, ha='right', fontsize=8)
axes2[1].set_ylim(0, 110)
axes2[1].axhline(100, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
axes2[1].set_title(f'S4 decision accuracy (%)\n({bench_label})')
axes2[1].set_ylabel('% correct')
for i, v in enumerate(acc_bench.values):
    if not np.isnan(v):
        axes2[1].text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=9)

# (C) Generation latency
lat_bench = df_bench.groupby('model')['generation_latency_s'].mean().reindex(MODELS)
axes2[2].bar(range(len(MODELS)), lat_bench.values, color=colors_m)
axes2[2].set_xticks(range(len(MODELS)))
axes2[2].set_xticklabels(MODELS, rotation=20, ha='right', fontsize=8)
axes2[2].set_title(f'Generation latency (s)\n({bench_label})')
axes2[2].set_ylabel('seconds')
for i, v in enumerate(lat_bench.values):
    if not np.isnan(v):
        axes2[2].text(i, v + 0.2, f'{v:.1f}s', ha='center', fontsize=9)

fig2.suptitle(f'DQM Model Benchmarking — {RUN_ID} ({bench_label})',
              fontsize=13, fontweight='bold')
plt.tight_layout()
out_png = f'eval_{RUN_ID}_benchmark.png'
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {out_png}')

# %% ── Pairwise comparison plot ───────────────────────────────────────────────
if not df_pairs.empty:
    valid_pairs = [(a, b, lbl) for a, b, lbl in PAIRWISE
                   if not df_eval[df_eval['model'] == a].empty
                   and not df_eval[df_eval['model'] == b].empty]

    if valid_pairs:
        fig_p, axes_p = plt.subplots(1, len(valid_pairs),
                                     figsize=(7 * len(valid_pairs), 5),
                                     sharey=True)
        if len(valid_pairs) == 1:
            axes_p = [axes_p]

        for ax, (model_a, model_b, label) in zip(axes_p, valid_pairs):
            sub = df_pairs[df_pairs['comparison'] == label]
            # Mean diff per section, averaged across variants
            sec_diff = sub.groupby('section')['diff'].mean().reindex(SCORE_COLS)

            bar_colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in sec_diff]
            ax.barh(SECTION_LABELS, sec_diff.values, color=bar_colors)
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_title(label, fontsize=9, fontweight='bold')
            ax.set_xlabel('Score difference (B − A)')
            short_a = model_a.split(':')[0].replace('qwen3-vl-longctx', 'qwen3-longctx')
            short_b = model_b.split(':')[0].replace('qwen3-vl-longctx', 'qwen3-longctx')
            ax.set_title(f'{label}\nA={short_a}  B={short_b}', fontsize=8)

        fig_p.suptitle('Pairwise score differences (green = B better, red = A better)',
                        fontsize=11, fontweight='bold')
        plt.tight_layout()
        out_pair = f'eval_{RUN_ID}_pairwise.png'
        plt.savefig(out_pair, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'Saved: {out_pair}')
