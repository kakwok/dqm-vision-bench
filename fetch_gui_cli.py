#!/usr/bin/env python3
"""
fetch_gui_cli.py
----------------
Command-line driver for dqm_gui_client.produce_images_from_gui().

Fetches shifter-view PNGs from the CMS DQM GUI and writes them into the same
images/<folder>/<stem>_run<XXXXXX>.png layout that dqm_plot.py produces, so the
rest of the pipeline (run_batch_cli.py, evaluate_cli.py) is unaffected.

Requires a CMS VOMS proxy for anything that touches the network:

    voms-proxy-init -voms cms -valid 24:00

Examples
--------
  # Preview the URLs that would be fetched — needs NO proxy and makes no request
  python3 fetch_gui_cli.py --runs 398185 --subsystem L1T --plot 00 --dry-run \
      --dataset '/ZeroBias/Run2024C-PromptReco-v1/DQMIO'

  # Check the proxy before committing to a long run
  python3 fetch_gui_cli.py --check-proxy

  # Real fetch
  python3 fetch_gui_cli.py --runs 398185 398186 --subsystem L1T \
      --dataset '/ZeroBias/Run2024C-PromptReco-v1/DQMIO' --outdir images

  # Online workspace needs no dataset
  python3 fetch_gui_cli.py --runs 398185 --subsystem L1T --workspace online
"""
from __future__ import annotations

import argparse
import os
import sys

from dqm_gui_client import (
    DQMGUISession,
    ProxyError,
    describe_proxy,
    produce_images_from_gui,
)
from shift_layout_helpers import build_image_config, gui_path, list_subsystems


def _select_specs(subsystem: str, plots: list[str] | None, parser) -> list[dict]:
    try:
        specs = build_image_config(subsystem)
    except ValueError as e:
        parser.error(str(e))
    if plots:
        keep = set(plots)
        specs = [s for s in specs if s["plot_number"] in keep]
        if not specs:
            parser.error(
                f"No plots matched --plot {plots} in subsystem '{subsystem}'. "
                f"Run 'python3 dqm_plot.py --list-plots {subsystem}' to see what exists."
            )
    return specs


def main():
    parser = argparse.ArgumentParser(
        description="Fetch DQM plots from the CMS DQM GUI as PNGs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--runs", nargs="+", metavar="RUN",
                        help="Run numbers to fetch.")
    parser.add_argument("--subsystem", metavar="NAME",
                        help="Subsystem from shift_layouts.json (e.g. L1T, Ecal).")
    parser.add_argument("--plot", nargs="+", metavar="NUMBER",
                        help="Restrict to these plot numbers (e.g. 00 08). "
                             "Requires --subsystem.")
    parser.add_argument("--dataset", default=None,
                        help="Offline dataset, e.g. "
                             "'/ZeroBias/Run2024C-PromptReco-v1/DQMIO'. "
                             "Not needed for --workspace online.")
    parser.add_argument("--workspace", default=os.environ.get("DQM_WORKSPACE", "offline"),
                        choices=("offline", "online"))
    parser.add_argument("--outdir", default="images")
    parser.add_argument("--width",  type=int, default=900)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-fetch even when the output PNG already exists.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass the on-disk response cache.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the URLs that would be fetched and exit. "
                             "Makes no network call and needs no proxy.")
    parser.add_argument("--check-proxy", action="store_true",
                        help="Report X.509 proxy status and exit.")
    parser.add_argument("--list-subsystems", action="store_true",
                        help="Print subsystem names and exit.")
    args = parser.parse_args()

    if args.list_subsystems:
        for s in dict.fromkeys(list_subsystems()):
            print(s)
        return

    if args.check_proxy:
        print(describe_proxy())
        return

    if not args.runs:
        parser.error("--runs is required (or use --check-proxy / --list-subsystems)")
    if args.plot and not args.subsystem:
        parser.error("--plot requires --subsystem")
    if not args.subsystem:
        parser.error("--subsystem is required")

    specs = _select_specs(args.subsystem, args.plot, parser)

    if args.dry_run:
        # No proxy check: previewing URLs must not require a credential.
        sess = DQMGUISession(workspace=args.workspace, cache_dir=None,
                             check_proxy=False)
        # Validate the dataset before emitting anything, so a missing one is
        # reported cleanly rather than halfway through the listing.
        try:
            sess.resolve_dataset(args.dataset)
        except ValueError as e:
            parser.error(str(e))

        print(f"{len(args.runs)} run(s) x {len(specs)} plot(s) "
              f"= {len(args.runs) * len(specs)} request(s)\n")
        for run in args.runs:
            for spec in specs:
                url = sess.plot_url(run, gui_path(spec["path"], args.workspace),
                                    args.dataset, args.width, args.height)
                print(f"{spec['stem']}\n  -> {url}")
        print(f"\n(dry run — nothing fetched; {describe_proxy()})")
        return

    try:
        sess = DQMGUISession(
            workspace=args.workspace,
            cache_dir=None if args.no_cache else ".dqm_cache",
        )
        results = produce_images_from_gui(
            args.runs, specs,
            dataset=args.dataset,
            outdir=args.outdir,
            width=args.width, height=args.height,
            session=sess, overwrite=args.overwrite,
        )
    except ProxyError as e:
        print(f"\nAuthentication failed.\n{e}", file=sys.stderr)
        sys.exit(2)

    failed = [r for r in results if r["error"]]
    if failed:
        print(f"\n{len(failed)} plot(s) failed or were flagged:", file=sys.stderr)
        for r in failed[:20]:
            print(f"  {r['plot_name']}: {r['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
