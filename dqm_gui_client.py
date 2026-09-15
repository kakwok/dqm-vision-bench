#!/usr/bin/env python3
"""
dqm_gui_client.py — Fetch rendered DQM plots from the CMS DQM GUI over HTTP.

This is the network-sourced counterpart to dqm_plot.py: instead of opening a
ROOT file and rendering a histogram locally, it asks the DQM GUI to render the
monitor element and returns the PNG the shifter would see in the browser.

It deliberately does NOT import ROOT, so it runs in the plain pixi environment
without a CMSSW release.

Authentication
--------------
cmsweb is gated by an X.509 client certificate, not by CERN SSO. Supply a CMS
VOMS proxy:

    voms-proxy-init -voms cms -valid 24:00

This module never creates, renews or destroys a proxy — it only reads the one
you already have and reports clearly when it is missing or expired.

Usage
-----
    from dqm_gui_client import DQMGUISession, produce_images_from_gui
    from shift_layout_helpers import build_image_config

    sess = DQMGUISession(workspace="offline")
    png  = sess.fetch_png(398185, "/ZeroBias/Run2024C-PromptReco-v1/DQMIO",
                          "L1T/Run summary/L1TStage2CaloLayer1/ecalOccRecdEtWgt")

See fetch_gui_cli.py for the command-line driver.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover - urllib3 always ships with requests
    Retry = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional for this module
    pass

from shift_layout_helpers import gui_path

# =============================================================================
# URL templates
#
# plotfairy: CONFIRMED against live cmsweb (2026-09-15), online workspace,
# run 398185, ME "L1T/L1TStage2CaloLayer1/ecalOccRecdEtWgt". A GET to
#   {base}/online/plotfairy/archive/398185/Global/Online/ALL/L1T/...?w=900;h=700
# returned HTTP 200 with a real PNG matching the browser's rendering. The
# GUI's own session state for that ME independently reported
#   'location': "archive/398185/Global/Online/ALL"
# i.e. exactly "archive/{run}{dataset}", confirming both this template and
# gui_path()'s online branch (no "Run summary" infix) are correct.
#
# The offline workspace template is NOT yet independently confirmed — it
# follows the same documented convention and online's confirmation makes it
# likely correct, but it has not been tested against live cmsweb. If a fetch
# 404s there, THIS is the block to revisit.
#
# samples: CONFIRMED WRONG for the online workspace. A live run (398185, with
# data actively being written) still returns {"samples": []} via
# data/json/samples?match=398185 — the online GUI does not expose run
# discovery this way; it is browsed through a stateful session (open session
# -> "select" -> "setFocus"), which this client deliberately does not
# implement (see list_samples() below). Do not use list_samples() to decide
# whether an online run exists — a negative result is not meaningful.
# Never independently confirmed working for offline either.
# =============================================================================
BASE_URL = os.environ.get("DQM_BASE_URL", "https://cmsweb.cern.ch/dqm").rstrip("/")

#   {base}/{workspace}/plotfairy/archive/{run}{dataset}/{me_path}?w=..;h=..
#   `dataset` begins with a slash, so it abuts `run` without a separator.
PLOTFAIRY_TMPL = "{base}/{ws}/plotfairy/archive/{run}{dataset}/{me}"

#   {base}/{workspace}/data/json/samples?match=...
#   Works for discovering OFFLINE datasets by run (unconfirmed but plausible).
#   Confirmed NOT to reflect ONLINE run availability — see note above.
SAMPLES_TMPL = "{base}/{ws}/data/json/samples"

# Online DQM has a single implicit dataset; offline requires a real one.
# Confirmed correct: the GUI's own session state reports
# 'dataset': "/Global/Online/ALL" for the online workspace.
ONLINE_DATASET = "/Global/Online/ALL"

WORKSPACES = ("offline", "online")

# cmsweb only mints its legacy cms-auth cookie for user agents matching
# (?:Prod|WM)Agent|visDQMUpload|[Pp]ython|curl — "python-requests/..." matches.
USER_AGENT = "python-requests/dqm-vision-bench"


class ProxyError(RuntimeError):
    """The X.509 proxy is missing, unreadable or expired."""


class DQMGUIError(RuntimeError):
    """The GUI returned something we could not use."""


# ---------------------------------------------------------------------------
# Proxy inspection — read-only. This module never runs voms-proxy-init.
# ---------------------------------------------------------------------------

def default_proxy_path() -> str:
    """Return $X509_USER_PROXY, else the conventional /tmp/x509up_u<uid>."""
    env = os.environ.get("X509_USER_PROXY")
    if env:
        return env
    return f"/tmp/x509up_u{os.getuid()}"


def default_ca_bundle() -> str | bool:
    """
    Return the CA bundle to verify cmsweb against.

    cmsweb is signed by the CERN Grid CA, which is not in certifi's default
    trust store, so plain verification fails with "unable to get local issuer
    certificate". requests accepts either a bundle file or a hashed CA
    directory; /etc/grid-security/certificates is the latter.
    """
    env = os.environ.get("DQM_CA_BUNDLE")
    if env:
        return env
    grid = "/etc/grid-security/certificates"
    if os.path.isdir(grid):
        return grid
    return True  # fall back to certifi; will likely fail against cmsweb


def proxy_seconds_left(proxy_path: str) -> int | None:
    """
    Seconds of validity remaining on the proxy, or None if it cannot be read.

    Uses `openssl x509 -noout -enddate`, which reads the first certificate in
    the PEM — for a VOMS proxy that is the short-lived proxy certificate
    itself, which is exactly the lifetime we care about.

    Returns None (rather than raising) when openssl is unavailable or the
    output is unparseable, so that a missing openssl degrades to "cannot
    check" rather than blocking the request.
    """
    try:
        out = subprocess.check_output(
            ["openssl", "x509", "-noout", "-enddate", "-in", proxy_path],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
    except Exception:
        return None

    m = re.search(r"notAfter=(.+)", out.strip())
    if not m:
        return None
    for fmt in ("%b %d %H:%M:%S %Y %Z", "%b %d %H:%M:%S %Y"):
        try:
            end = datetime.strptime(m.group(1).strip(), fmt)
            end = end.replace(tzinfo=timezone.utc)
            return int((end - datetime.now(timezone.utc)).total_seconds())
        except ValueError:
            continue
    return None


def describe_proxy(proxy_path: str | None = None) -> str:
    """Human-readable one-line proxy status — handy for CLI diagnostics."""
    path = proxy_path or default_proxy_path()
    if not os.path.exists(path):
        return f"proxy: ABSENT ({path})"
    left = proxy_seconds_left(path)
    if left is None:
        return f"proxy: present at {path} (lifetime unknown — openssl unavailable)"
    if left <= 0:
        return f"proxy: EXPIRED {-left // 3600}h ago ({path})"
    return f"proxy: valid for {left // 3600}h{(left % 3600) // 60:02d}m ({path})"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class DQMGUISession:
    """
    An authenticated connection to one DQM GUI workspace.

    Parameters
    ----------
    workspace    : 'offline' or 'online'.
    proxy        : Path to the VOMS proxy. Defaults to $X509_USER_PROXY, else
                   /tmp/x509up_u<uid>.
    ca_bundle    : CA bundle file or hashed CA directory. Defaults to
                   $DQM_CA_BUNDLE, else /etc/grid-security/certificates.
    cache_dir    : Directory for cached responses. None disables caching.
    min_lifetime : Warn when the proxy has less than this many seconds left.
    timeout      : Per-request timeout in seconds.
    check_proxy  : Validate the proxy up front. Set False only for URL-building
                   previews (--dry-run), which make no network call.

    Credentials are resolved lazily here, not at import time, so that importing
    this module without a proxy present never breaks an unrelated notebook.
    """

    def __init__(
        self,
        workspace: str = "offline",
        proxy: str | None = None,
        ca_bundle: str | bool | None = None,
        cache_dir: str | Path | None = ".dqm_cache",
        min_lifetime: int = 7200,
        timeout: int = 120,
        base_url: str | None = None,
        check_proxy: bool = True,
    ):
        if workspace not in WORKSPACES:
            raise ValueError(
                f"Unknown workspace {workspace!r}; expected one of {WORKSPACES}."
            )

        self.workspace    = workspace
        self.proxy        = proxy or default_proxy_path()
        self.ca_bundle    = default_ca_bundle() if ca_bundle is None else ca_bundle
        self.cache_dir    = Path(cache_dir) if cache_dir else None
        self.timeout      = timeout
        self.base_url     = (base_url or BASE_URL).rstrip("/")
        self.min_lifetime = min_lifetime

        # Tracks whether any request has ever succeeded, so that a later 401
        # can be attributed to an expired proxy rather than a bad path.
        self._had_success = False

        # check_proxy=False supports --dry-run: building URLs needs no
        # credential, so a preview must not require one.
        if check_proxy:
            self._check_proxy()

        self.session = requests.Session()
        self.session.cert    = (self.proxy, self.proxy)  # VOMS proxy: one PEM, both roles
        self.session.verify  = self.ca_bundle
        self.session.headers.update({"User-Agent": USER_AGENT})

        if Retry is not None:
            retry = Retry(
                total=4, backoff_factor=1.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset(["GET"]),
                raise_on_status=False,
            )
            self.session.mount("https://", HTTPAdapter(max_retries=retry))

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- auth ---------------------------------------------------------------

    def _check_proxy(self) -> None:
        """
        Fail fast and legibly on a missing or expired proxy.

        This only ever *reads* the proxy. Renewing it is the user's job —
        the fix is printed, never performed.
        """
        if not os.path.exists(self.proxy):
            raise ProxyError(
                f"No X.509 proxy at {self.proxy}.\n"
                f"Create one with:  voms-proxy-init -voms cms -valid 24:00\n"
                f"(or point X509_USER_PROXY / the proxy= argument at an existing one)"
            )

        left = proxy_seconds_left(self.proxy)
        if left is None:
            return  # cannot determine; let the request itself be the test
        if left <= 0:
            raise ProxyError(
                f"X.509 proxy at {self.proxy} expired {-left // 3600}h ago.\n"
                f"Renew it with:  voms-proxy-init -voms cms -valid 24:00"
            )
        if left < self.min_lifetime:
            print(
                f"WARNING: X.509 proxy expires in {left // 3600}h{(left % 3600) // 60:02d}m. "
                f"Renew with 'voms-proxy-init -voms cms -valid 24:00' before a long run."
            )

    def _raise_for_auth(self, resp: requests.Response, url: str) -> None:
        """Turn cmsweb's bare 401 into something a human can act on."""
        if resp.status_code != 401:
            return
        left = proxy_seconds_left(self.proxy)
        if self._had_success:
            hint = ("The proxy worked earlier in this session, so it has most likely "
                    "expired mid-run. Renew it with 'voms-proxy-init -voms cms'.")
        elif left is not None and left <= 0:
            hint = "The proxy is expired. Renew with 'voms-proxy-init -voms cms'."
        else:
            hint = ("cmsweb rejected the certificate. Check that the proxy carries the "
                    "CMS VO ('voms-proxy-info -all' should show VO cms) and that your "
                    "grid certificate is registered with CMS.")
        raise ProxyError(f"HTTP 401 from {url}\n{hint}")

    # -- transport ----------------------------------------------------------

    def _cache_file(self, url: str, suffix: str) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha1(url.encode()).hexdigest()
        return self.cache_dir / f"{key}{suffix}"

    def _get(self, url: str, suffix: str = "", use_cache: bool = True) -> bytes:
        """GET *url*, returning raw bytes, with an on-disk cache."""
        cache = self._cache_file(url, suffix) if use_cache else None
        if cache and cache.exists():
            return cache.read_bytes()

        try:
            resp = self.session.get(url, timeout=self.timeout)
        except requests.exceptions.SSLError as e:
            raise DQMGUIError(
                f"TLS failure contacting {url}: {e}\n"
                f"cmsweb is signed by the CERN Grid CA; set DQM_CA_BUNDLE to a CA "
                f"directory (e.g. /etc/grid-security/certificates) or a bundle PEM."
            ) from e

        self._raise_for_auth(resp, url)
        if resp.status_code == 404:
            raise DQMGUIError(f"HTTP 404 — no such run/dataset/monitor element:\n  {url}")
        resp.raise_for_status()

        self._had_success = True
        if cache:
            cache.write_bytes(resp.content)
        return resp.content

    # -- public API ---------------------------------------------------------

    def resolve_dataset(self, dataset: str | None) -> str:
        """
        Return the dataset to query.

        Online DQM has a single implicit dataset, so it may be omitted. Offline
        has one per primary dataset / processing, so it cannot be guessed —
        omitting it there is an error rather than a silent default.
        """
        if dataset:
            return dataset
        if self.workspace == "online":
            return ONLINE_DATASET
        raise ValueError(
            "A dataset is required for the offline workspace "
            "(e.g. '/ZeroBias/Run2024C-PromptReco-v1/DQMIO'). "
            "Use list_samples(run) to see what exists for a run."
        )

    def samples_url(self, match: str | None = None) -> str:
        url = SAMPLES_TMPL.format(base=self.base_url, ws=self.workspace)
        if match:
            url += f"?match={quote(match)}"
        return url

    def list_samples(self, match: str | None = None) -> dict:
        """
        Return the GUI's sample index as parsed JSON.

        `match` is a substring filter applied by the server, e.g. a run number
        or a dataset fragment. The response shape for the offline workspace is
        a GUI implementation detail that has not been independently confirmed;
        treat it as opaque.

        Raises DQMGUIError immediately for the online workspace: confirmed
        live (2026-09-15) that this endpoint returns {"samples": []} for a
        run that is actively being written to online DQM. It does not reflect
        online run availability at all — the online GUI's own frontend
        browses runs through a stateful session (open -> "select" ->
        "setFocus"), not this endpoint. Calling it here to check "does this
        run exist online" would silently return a false negative, so it is
        refused rather than left to mislead a caller.
        """
        if self.workspace == "online":
            raise DQMGUIError(
                "list_samples() does not work for the online workspace — "
                "confirmed empirically that it returns {'samples': []} even "
                "for a run currently live in online DQM. The online dataset "
                "is fixed (ONLINE_DATASET) and needs no lookup: just call "
                "fetch_png(run, me_path) directly. If you need to confirm a "
                "run is actually present online, check via the GUI itself."
            )
        import json
        raw = self._get(self.samples_url(match), suffix=".json")
        try:
            return json.loads(raw)
        except ValueError as e:
            raise DQMGUIError(
                f"Sample index was not JSON — the endpoint shape is probably wrong.\n"
                f"  url: {self.samples_url(match)}\n"
                f"  first bytes: {raw[:200]!r}"
            ) from e

    def plot_url(
        self,
        run: int | str,
        me_path: str,
        dataset: str | None = None,
        width: int = 900,
        height: int = 700,
    ) -> str:
        """Build the plotfairy URL for one monitor element."""
        ds = self.resolve_dataset(dataset)
        if not ds.startswith("/"):
            ds = "/" + ds
        # ME paths contain spaces ("Run summary"); encode them but keep the
        # path separators intact.
        url = PLOTFAIRY_TMPL.format(
            base=self.base_url,
            ws=self.workspace,
            run=str(run).lstrip("0") or "0",
            dataset=quote(ds, safe="/"),
            me=quote(me_path, safe="/"),
        )
        return f"{url}?w={width};h={height}"

    def fetch_png(
        self,
        run: int | str,
        me_path: str,
        dataset: str | None = None,
        width: int = 900,
        height: int = 700,
        use_cache: bool = True,
    ) -> bytes:
        """
        Return the rendered PNG bytes for one monitor element.

        Raises DQMGUIError if the response is not a PNG. Note that a *valid*
        PNG saying "no such monitor element" is still returned — see
        looks_like_placeholder().
        """
        url = self.plot_url(run, me_path, dataset, width, height)
        data = self._get(url, suffix=".png", use_cache=use_cache)
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise DQMGUIError(
                f"Response was not a PNG (first bytes {data[:32]!r}).\n  url: {url}"
            )
        return data


# ---------------------------------------------------------------------------
# Placeholder detection
# ---------------------------------------------------------------------------

# A "monitor element not found" render is a small, nearly blank image. Real
# DQM plots — axes, labels, stat box — compress to far more than this.
#
# Confirmed data point (2026-09-15): a real plot (online, run 398185, L1T
# ecalOccRecdEtWgt, 900x700) is ~29 KB — well above this threshold, so a real
# plot of this kind will not false-positive. The GUI's actual "not found"
# placeholder size is still unconfirmed, so the threshold is a conservative
# guess on the low side rather than one calibrated against a real placeholder.
PLACEHOLDER_MAX_BYTES = 6000


def looks_like_placeholder(png: bytes, threshold: int = PLACEHOLDER_MAX_BYTES) -> bool:
    """
    Heuristic: does this PNG look like the GUI's "not found" placeholder?

    This is a size heuristic, not a content check, and it is deliberately
    conservative — it is meant to *flag* suspicious results for a human, not to
    discard them. A real plot is confirmed to be ~29 KB (see above); the
    placeholder itself has not been observed, so `threshold` is not yet
    calibrated against one. Fetch a monitor element known to be absent for a
    given run to get that reference point and tighten this.
    """
    return len(png) < threshold


# ---------------------------------------------------------------------------
# Batch entry point — mirrors dqm_plot.produce_images()
# ---------------------------------------------------------------------------

def produce_images_from_gui(
    runs: list[str | int],
    image_config: list[dict] | dict[str, str],
    dataset: str | None = None,
    workspace: str = "offline",
    outdir: str = "images",
    width: int = 900,
    height: int = 700,
    session: DQMGUISession | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """
    Fetch every plot in *image_config* for every run in *runs* from the DQM GUI.

    Writes PNGs to exactly the paths dqm_plot.produce_images() would use:

        <outdir>/<folder>/<stem>_run<XXXXXX>.png

    and returns result dicts with the same keys, so everything downstream
    (owui_client.find_reference_images, evaluate.py) works unchanged. The only
    difference is that `root_file` holds the source URL instead of a file path.

    Parameters
    ----------
    runs         : Run numbers.
    image_config : Either a list of ImageSpec dicts from
                   shift_layout_helpers.build_image_config(), or a legacy
                   {stem: path} mapping. ImageSpec entries carry the shift
                   layout `path`, which is what the GUI needs.
    dataset      : Offline dataset. Optional for the online workspace.
    overwrite    : Re-fetch even when the output PNG already exists.
    """
    if isinstance(image_config, dict):
        specs = [{"stem": k, "folder": k, "path": v} for k, v in image_config.items()]
    else:
        specs = list(image_config)

    if not runs:
        raise ValueError("No runs given.")
    if not specs:
        raise ValueError("image_config is empty.")

    sess = session or DQMGUISession(workspace=workspace)

    total   = len(runs) * len(specs)
    results: list[dict] = []
    n = 0

    if verbose:
        print(f"{len(runs)} run(s) x {len(specs)} plot(s) = {total} total")
        print(f"{describe_proxy(sess.proxy)}\n")

    for run in runs:
        run_display = str(run).lstrip("0") or "0"
        if verbose:
            print(f"-- Run {run_display}")

        for spec in specs:
            n += 1
            stem   = spec["stem"]
            folder = spec.get("folder", stem)

            # ImageSpec carries the raw shift-layout path; translate it to the
            # workspace's ME path. A legacy mapping may already hold one.
            json_path = spec.get("path")
            if not json_path:
                results.append(_result(None, run_display, stem, folder, None,
                                       "spec has no 'path' key"))
                continue
            me_path = gui_path(json_path, sess.workspace)

            out_dir = Path(outdir) / folder
            out_png = out_dir / f"{stem}_run{run_display.zfill(6)}.png"

            if out_png.exists() and not overwrite:
                if verbose:
                    print(f"  [{n}/{total}] SKIP {stem} — exists")
                results.append(_result(None, run_display, stem, folder,
                                       str(out_png), None))
                continue

            url = sess.plot_url(run_display, me_path, dataset, width, height)
            try:
                png = sess.fetch_png(run_display, me_path, dataset, width, height)
            except ProxyError:
                raise  # auth problems are fatal, not per-plot failures
            except Exception as e:
                if verbose:
                    print(f"  [{n}/{total}] ERROR {stem}: {e}")
                results.append(_result(url, run_display, stem, folder, None, str(e)))
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            out_png.write_bytes(png)

            if looks_like_placeholder(png):
                note = f"suspiciously small ({len(png)} B) — possibly a 'not found' render"
                if verbose:
                    print(f"  [{n}/{total}] WARN {stem} -> {out_png} ({note})")
                results.append(_result(url, run_display, stem, folder,
                                       str(out_png), note))
                continue

            if verbose:
                print(f"  [{n}/{total}] {stem} -> {out_png}")
            results.append(_result(url, run_display, stem, folder, str(out_png), None))

    ok      = sum(1 for r in results if r["error"] is None)
    errored = len(results) - ok
    if verbose:
        print(f"\nDone. {ok} saved, {errored} skipped/errored/flagged.")
    return results


def _result(url, run, stem, folder, out_png, error) -> dict:
    """Result dict shaped exactly like dqm_plot.produce_images() emits."""
    return {
        "root_file": url,      # source URL, in the slot produce_images uses for the path
        "run":       run,
        "plot_name": stem,
        "folder":    folder,
        "out_png":   out_png,
        "error":     error,
    }
