"""
evaluate_cli.py
----------------
Command-line driver for evaluate.run_evaluations() (LLM-judge scoring).
Headless equivalent of the judge pipeline in evaluate.ipynb.

Two ways to select which results to evaluate:
  --preset NAME     pick one of the example run_id groups defined below
  --run-ids ID...   pass run_id(s) directly (overrides --preset)

Examples
--------
  python3 evaluate_cli.py --preset yaml_image_modes
  python3 evaluate_cli.py --run-ids YAML_direct YAML_ref_captioned YAML_both_captioned
  python3 evaluate_cli.py --preset yaml_image_modes --dry-run
  python3 evaluate_cli.py --preset yaml_image_modes --models google/gemma4-31b qwen/qwen3.6
  python3 evaluate_cli.py --run-ids YAML --plots L1T_00_CaloLayer1ECALoccupancy \
      --models asksage-overflow/claude-opus-5 asksage-overflow/claude-sonnet-4-6
"""
import argparse
import sys
from pathlib import Path

from evaluate import (
    load_results,
    truth_coverage,
    run_evaluations,
    report_latency,
    report_scores,
)

# ---------------------------------------------------------------------------
# Example presets — add your own entries here, or drive everything from
# --run-ids instead.
# ---------------------------------------------------------------------------
PRESETS: dict[str, list[str]] = {
    "yaml_image_modes": ["YAML_direct", "YAML_ref_captioned", "YAML_both_captioned"],
}

DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive evaluate.run_evaluations() from the command line.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None,
                         help="Named run_id group from PRESETS.")
    parser.add_argument("--run-ids", nargs="+", default=None,
                         help="One or more run_id(s) under --output-root. Overrides --preset.")
    parser.add_argument("--models", nargs="+", default=None,
                         help="Restrict to these evaluated model names. Default: all models found.")
    parser.add_argument("--plots", nargs="+", default=None,
                         help="Restrict to these plot_name(s). Default: all plots in the run_id.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                         help=f"Judge model to score responses with. Default: {DEFAULT_JUDGE_MODEL}")
    parser.add_argument("--delay", type=float, default=1.0,
                         help="Seconds to sleep between judge calls. Default: 1.0")
    parser.add_argument("--output-root", type=Path, default=Path("results"),
                         help="Root folder containing <run_id>/ result dirs. Default: results")
    parser.add_argument("--truth-root", type=Path, default=Path("truth"),
                         help="Root folder of ground-truth .txt files. Default: truth")
    parser.add_argument("--eval-csv", type=Path, default=None,
                         help="Cached judge-score CSV to append to/resume from. "
                              "Default: eval_output/eval_scores.csv")
    parser.add_argument("--claim-csv", type=Path, default=None,
                         help="Per-claim verdict CSV to append to (claims-mode only). "
                              "Default: eval_output/claim_scores.csv")
    parser.add_argument("--group-by", nargs="+", default=["run_id", "model_short"],
                         help="Columns to group the final score reports by. Default: run_id model_short")
    parser.add_argument("--dry-run", action="store_true",
                         help="Load results and print coverage/latency/error summary; "
                              "exit without calling the judge model.")
    parser.add_argument("--no-claims", action="store_true",
                         help="Force holistic grading even if claims/<plot>.yaml exists.")
    args = parser.parse_args()

    if args.run_ids:
        run_ids = args.run_ids
    elif args.preset:
        run_ids = PRESETS[args.preset]
    else:
        parser.error("Pass --preset NAME or --run-ids ID [ID ...].")

    eval_csv = args.eval_csv or (Path("eval_output") / "eval_scores.csv")
    eval_csv.parent.mkdir(parents=True, exist_ok=True)
    claim_csv = args.claim_csv or (Path("eval_output") / "claim_scores.csv")
    claim_csv.parent.mkdir(parents=True, exist_ok=True)

    df = load_results(args.output_root, run_ids, models=args.models)
    if args.plots:
        df = df[df["plot_name"].isin(args.plots)]
    if df.empty:
        sys.exit(1)

    print()
    truth_coverage(df, args.truth_root)

    print()
    report_latency(df, group_col="model_short")

    errors = df[df["error"].notna()]
    print(f"\n{len(errors)} errors" if not errors.empty else "\nNo errors.")

    if args.dry_run:
        return

    judge_model_for = {m: [args.judge_model] for m in df["model"].unique()}
    print(f"\nJudge model: {args.judge_model}")
    print(f"Evaluated models: {sorted(judge_model_for)}")

    df_eval = run_evaluations(df, args.truth_root, judge_model_for, eval_csv, delay=args.delay,
                               no_claims=args.no_claims, claim_csv=claim_csv)
    print(f"\n{len(df_eval)} scored rows in df_eval")

    df_eval = df_eval[df_eval["run_id"].isin(run_ids)]

    for group_col in args.group_by:
        print(f"\n=== By {group_col} ===")
        report_scores(df_eval, group_col=group_col)


if __name__ == "__main__":
    main()
