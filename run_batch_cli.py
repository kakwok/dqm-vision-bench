"""
run_batch_cli.py
-----------------
Command-line driver for owui_client.run_batch().

Two ways to supply a BatchConfig:
  --preset NAME   pick one of the example configs defined below
  --config PATH   load a BatchConfig from a JSON file (e.g. a previous run's
                   results/<run_id>/config_<run_id>.json, or a hand-written one),
                   via BatchConfig.from_json(). Takes precedence over --preset.

Examples
--------
  python3 run_batch_cli.py --preset yaml
  python3 run_batch_cli.py --preset yaml --overwrite
  python3 run_batch_cli.py --config results/YAML/config_YAML.json --retry
  python3 run_batch_cli.py --preset yaml --dry-run
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from owui_client import (
    PROVIDERS,
    BatchConfig,
    CaptionedBoth,
    CaptionedReferences,
    DirectImages,
    ModelConfig,
    RunMetadata,
    run_batch,
    retry_failed,
    sanity_check,
)
from rag_backends import YAMLContext, LocalRAG

IMAGE_MODES = {
    "direct": DirectImages,
    "ref_captioned": CaptionedReferences,
    "both_captioned": CaptionedBoth,
}

# ---------------------------------------------------------------------------
# Example presets — mirrors the cfg cell in batch_query.ipynb.
# Add your own entries here, or drive everything from --config instead.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an assistant to CMS detector operations shifters, helping evaluate DQM monitoring plots.

Assess whether the input plot indicates a detector problem requiring action.

Respond in 4 sections:

Instructions: Quote the shift instructions most relevant to this plot type.

Observations: Describe what you see in the input plot (color distribution, notable features, anomalies). If a reference plot is shown, describe it and note how the input differs.

Assessment: Compare the input plot against the quoted instructions. If a reference is provided, use it to calibrate what normal looks like.

Verdict: State GOOD or BAD. If BAD, specify the required action from the instructions (e.g. "contact ECAL DOC", "add to elog only").\
"""

PRESETS: dict[str, BatchConfig] = {
    "yaml": BatchConfig(
        run_id="YAML",
        image_root=Path("images"),
        output_root=Path("results"),
        ref_dir=Path("ref_images"),
        plot_filter=None,
        models=[
            "qwen/qwen3.6",
            "google/gemma3-27b",
            "google/gemma4-31b",
        ],
        context=YAMLContext(),
        system_prompt=SYSTEM_PROMPT,
        prompt="",
        delay=1.5,
        run_metadata=RunMetadata(event_type_map={
            398185: "collisions",
            398186: "cosmics",
            398187: "circulating",
            398188: "collisions",
            398189: "collisions",
            398191: "collisions",
            398194: "cosmics",
            398199: "cosmics",
        }),
    ),
    # Next campaign: input images fetched from the live DQM GUI API
    # (fetch_gui_cli.py) instead of locally-rendered ROOT plots — the GUI's
    # own rendering is what shifters actually see and may resolve color-scale
    # discrepancies the local renderer introduces. Scoped to
    # L1T_00_CaloLayer1ECALoccupancy, the only plot with a claims/*.yaml
    # checklist, so claims-v3 grading applies. Populate images_api/ first:
    #   python3 fetch_gui_cli.py --runs 398185 398186 398187 398188 398189 \
    #       398191 398194 398199 --subsystem L1T --plot 00 \
    #       --dataset '/ZeroBias/Run2024C-PromptReco-v1/DQMIO' --outdir images_api
    # then run_batch_cli.py --preset yaml_context_api, then evaluate:
    #   python3 evaluate_cli.py --run-ids YAMLCONtext \
    #       --plots L1T_00_CaloLayer1ECALoccupancy \
    #       --judge-model openai/gpt-oss-120b
    "yaml_context_api": BatchConfig(
        run_id="YAMLCONtext",
        image_root=Path("images_api"),
        output_root=Path("results"),
        ref_dir=Path("ref_images"),
        plot_filter=["L1T_00_CaloLayer1ECALoccupancy"],
        models=[
            "qwen/qwen3.6",
            "google/gemma3-27b",
            "google/gemma4-31b",
            "asksage-overflow/claude-haiku-4-5",
            "asksage-overflow/claude-opus-4-8",
            "asksage-overflow/claude-opus-5",
            "asksage-overflow/claude-sonnet-4-6",
            "asksage-overflow/gpt-5.6-luna",
            "asksage-overflow/gpt-5.6-sol",
            "asksage-overflow/gpt-5.6-terra",
        ],
        context=YAMLContext(),
        system_prompt=SYSTEM_PROMPT,
        prompt="",
        delay=1.5,
        run_metadata=RunMetadata(event_type_map={
            398185: "collisions",
            398186: "cosmics",
            398187: "circulating",
            398188: "collisions",
            398189: "collisions",
            398191: "collisions",
            398194: "cosmics",
            398199: "cosmics",
        }),
    ),
    "localrag": BatchConfig(
        run_id="localRAG",
        image_root=Path("images"),
        output_root=Path("results"),
        ref_dir=Path("ref_images"),
        plot_filter=None,
        models=[
            "qwen/qwen3.6",
            "google/gemma3-27b",
            "google/gemma4-31b",
        ],
        context=LocalRAG(csv_path=Path("document_chunks.csv")),  # new defaults: alpha=0.8, structured query
        system_prompt=SYSTEM_PROMPT,
        prompt="",
        delay=1.5,
        run_metadata=RunMetadata(event_type_map={
            398185: "collisions",
            398186: "cosmics",
            398187: "circulating",
            398188: "collisions",
            398189: "collisions",
            398191: "collisions",
            398194: "cosmics",
            398199: "cosmics",
        }),
    ),
}


def save_summary_csv(cfg: BatchConfig, results: list[dict]) -> Path:
    """Append this run's results to the run's summary CSV (does not overwrite prior rows)."""
    df = pd.DataFrame(results)
    df["image_name"] = df["image"].apply(lambda p: Path(p).name if p else None)

    if cfg.run_id:
        csv_path = Path(cfg.output_root) / cfg.run_id / f"summary_{cfg.run_id}.csv"
    else:
        csv_path = Path(cfg.output_root) / "summary.csv"

    if csv_path.exists():
        df = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)

    df.to_csv(csv_path, index=False)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive owui_client.run_batch() from the command line.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None,
                         help="Named BatchConfig from PRESETS.")
    parser.add_argument("--config", type=Path, default=None,
                         help="Load a BatchConfig from a JSON file (overrides --preset).")
    parser.add_argument("--overwrite", action="store_true",
                         help="Re-query (model, image) pairs even if an output .txt already exists.")
    parser.add_argument("--retry", action="store_true",
                         help="Retry any failed queries after the batch completes.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the sanity-check summary and exit without querying.")
    parser.add_argument("--all-plots", action="store_true",
                         help="Clear plot_filter (run every plot under image_root), "
                              "overriding whatever the preset/config file set.")
    parser.add_argument("--image-mode", choices=sorted(IMAGE_MODES), default=None,
                         help="direct (default): send images as-is. ref_captioned: caption "
                              "reference images, send input directly. both_captioned: caption "
                              "both. Overrides whatever the preset/config file set. Captioning "
                              "defaults to self-caption (the model under test captions its own "
                              "images) unless --caption-model is given.")
    parser.add_argument("--caption-model", default=None,
                         help="Model to use for captioning instead of self-caption. "
                              "Only applies with --image-mode ref_captioned/both_captioned.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=None,
                         help="Inference provider to query all models through (see owui_client.PROVIDERS). "
                              "Overrides whatever the preset/config file set, and is saved into "
                              "config_<run_id>.json so the run stays reproducible.")
    args = parser.parse_args()

    if args.config:
        cfg = BatchConfig.from_json(args.config)
    elif args.preset:
        cfg = PRESETS[args.preset]
    else:
        parser.error("Pass --preset NAME or --config PATH.")

    if args.all_plots:
        cfg.plot_filter = None

    if args.image_mode:
        mode_cls = IMAGE_MODES[args.image_mode]
        cfg.image_mode = mode_cls() if mode_cls is DirectImages else mode_cls(caption_model=args.caption_model)

    if args.provider:
        cfg.models = [
            ModelConfig(**{**vars(m), "provider": args.provider}) if isinstance(m, ModelConfig)
            else ModelConfig(name=m, provider=args.provider)
            for m in cfg.models
        ]

    pairs = cfg.build_pairs()
    sanity_check(cfg, pairs)
    if args.dry_run:
        return

    results = run_batch(cfg, pairs, overwrite=args.overwrite)

    if args.retry:
        results = retry_failed(cfg, results)

    csv_path = save_summary_csv(cfg, results)
    print(f"\nSaved: {csv_path}")

    n_errors = sum(1 for r in results if r["error"])
    print(f"Done. {len(results)} queries, {n_errors} still failing.")
    sys.exit(1 if n_errors else 0)


if __name__ == "__main__":
    main()
