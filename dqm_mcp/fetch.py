"""
Fetch every panel of a plot from the online DQM GUI, honouring the guardrail.

Panels are written to ``<image_root>/<folder>/<stem>_run<RUN>.png`` — the
layout ``produce_images_from_gui`` uses — because the instruction lookup and
reference-image lookup both key off the image's parent directory.

Sources, cheapest first:
  disk     the PNG already exists under image_root (free)
  cache    the GUI response is in .dqm_cache (free, no proxy needed)
  network  a real request to cmsweb (counts against the budget, needs a proxy)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dqm_gui_client import DQMGUISession, ProxyError, describe_proxy, looks_like_placeholder
from shift_layout_helpers import gui_path

from .budget import get_budget
from .config import get_settings
from .resolve import PlotGroup


@dataclass
class FetchResult:
    stem: str
    me_path: str
    url: str
    path: Path
    size: int
    placeholder: bool
    source: str   # "disk" | "cache" | "network"

    def meta(self) -> dict:
        return {
            "stem": self.stem,
            "me_path": self.me_path,
            "url": self.url,
            "path": str(self.path),
            "bytes": self.size,
            "placeholder_suspected": self.placeholder,
            "source": self.source,
        }


_session: DQMGUISession | None = None


def get_session() -> DQMGUISession:
    """
    One session per process. Built with check_proxy=False so that URL building
    and cache hits work without a credential; the proxy is validated right
    before the first network request instead (see _ensure_proxy).
    """
    global _session
    if _session is None:
        s = get_settings()
        _session = DQMGUISession(
            workspace=s.workspace,
            cache_dir=s.cache_dir,
            check_proxy=False,
        )
    return _session


def _ensure_proxy(sess: DQMGUISession) -> None:
    try:
        sess._check_proxy()  # the client's own check; gives the actionable message
    except ProxyError as e:
        raise RuntimeError(f"DQM GUI authentication unavailable: {e}") from e


def image_path_for(group: PlotGroup, stem: str, run: int | str) -> Path:
    return get_settings().image_root / group.folder / f"{stem}_run{str(run).zfill(6)}.png"


def run_display(run: int | str) -> str:
    """Run number as the GUI wants it: digits, no leading zeros."""
    text = str(run).strip().lstrip("0")
    if not text.isdigit():
        raise ValueError(f"run must be a positive integer, got {run!r}")
    return text


def fetch_group(run: int | str, group: PlotGroup, *, overwrite: bool = False) -> list[FetchResult]:
    s = get_settings()
    budget = get_budget()
    sess = get_session()
    run_text = run_display(run)

    budget.check_call(group.n_panels)

    # Plan first, so the budget is reserved once, atomically, for exactly the
    # panels that will actually hit the network.
    plan = []
    for spec in group.specs:
        me_path = gui_path(spec["path"], s.workspace)
        url = sess.plot_url(run_text, me_path, None, s.width, s.height)
        path = image_path_for(group, spec["stem"], run_text)
        if path.exists() and not overwrite:
            source = "disk"
        elif (c := sess._cache_file(url, ".png")) is not None and c.exists():
            source = "cache"
        else:
            source = "network"
        plan.append((spec, me_path, url, path, source))

    n_network = sum(1 for *_, src in plan if src == "network")
    if n_network:
        _ensure_proxy(sess)
        budget.reserve(n_network)

    results = []
    for spec, me_path, url, path, source in plan:
        if source == "disk":
            png = path.read_bytes()
        else:
            try:
                png = sess.fetch_png(run_text, me_path, None, s.width, s.height)
            except ProxyError as e:
                raise RuntimeError(f"DQM GUI authentication failed: {e}") from e
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(png)
        results.append(FetchResult(
            stem=spec["stem"],
            me_path=me_path,
            url=url,
            path=path,
            size=len(png),
            placeholder=looks_like_placeholder(png),
            source=source,
        ))
    return results


def proxy_status() -> str:
    return describe_proxy()
