"""
Runtime settings for the MCP server, read once from the environment.

All paths are resolved relative to the repository root so the server behaves
the same regardless of the client's working directory. Every knob has a
default; nothing here is required for the server to start.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from e


def _env_path(name: str, default: str) -> Path:
    raw = os.environ.get(name, "").strip() or default
    p = Path(raw).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass(frozen=True)
class Settings:
    workspace: str
    image_root: Path
    results_root: Path
    run_id: str
    ref_dir: Path
    store_dir: Path
    cache_dir: Path
    width: int
    height: int
    model: str
    provider: str
    max_images_per_call: int
    max_fetches_per_hour: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            # Online is the only workspace verified against cmsweb; offline
            # needs a dataset string and is deliberately not exposed.
            workspace="online",
            image_root=_env_path("DQM_MCP_IMAGE_ROOT", "images"),
            results_root=_env_path("DQM_MCP_RESULTS_ROOT", "results"),
            run_id=os.environ.get("DQM_MCP_RUN_ID", "MCP").strip() or "MCP",
            ref_dir=_env_path("DQM_MCP_REF_DIR", "ref_images"),
            store_dir=_env_path("DQM_MCP_STORE_DIR", "plot_instructions"),
            cache_dir=_env_path("DQM_MCP_CACHE_DIR", ".dqm_cache"),
            width=_env_int("DQM_MCP_WIDTH", 900),
            height=_env_int("DQM_MCP_HEIGHT", 700),
            model=os.environ.get("OWUI_MODEL", "").strip(),
            provider=os.environ.get("DEFAULT_PROVIDER", "").strip(),
            max_images_per_call=_env_int("DQM_MCP_MAX_IMAGES_PER_CALL", 12),
            max_fetches_per_hour=_env_int("DQM_MCP_MAX_FETCHES_PER_HOUR", 60),
        )

    def public(self) -> dict:
        """Settings safe to show to a client (paths and limits, no secrets)."""
        return {
            "workspace": self.workspace,
            "image_root": str(self.image_root),
            "results_root": str(self.results_root),
            "run_id": self.run_id,
            "ref_dir": str(self.ref_dir),
            "ref_dir_exists": self.ref_dir.is_dir(),
            "store_dir": str(self.store_dir),
            "cache_dir": str(self.cache_dir),
            "image_size": [self.width, self.height],
            "default_model": self.model or None,
            "default_provider": self.provider or None,
            "max_images_per_call": self.max_images_per_call,
            "max_fetches_per_hour": self.max_fetches_per_hour,
        }


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reset_settings() -> None:
    """Forget the cached settings (tests, or after changing the environment)."""
    global _settings
    _settings = None
