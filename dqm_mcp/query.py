"""
Ask a vision model about one plot (all of its panels at once).

Builds on ``owui_client.build_messages`` / ``send_query`` so the message
layout, provider registry and result shape are identical to the batch
pipeline. The only addition is splicing extra panels into the user message
when a plot has several.

``owui_client`` is imported lazily: it raises at import time when
OWUI_API_KEY is unset, and that should surface as a tool error, not prevent
the server from starting.
"""
from __future__ import annotations

from pathlib import Path

from .config import get_settings

PANEL_INTRO_SINGLE = "The following is the image to be judged:"
PANEL_INTRO_MULTI = (
    "The following are the {n} panels of the plot to be judged. "
    "They are parts of one shift-layout plot; assess them together and give one verdict."
)


def _owui():
    try:
        import owui_client
    except EnvironmentError as e:
        raise RuntimeError(
            f"Model client not configured: {e} "
            "Put OWUI_API_KEY plus the provider URL/key (e.g. LITELLM_URL, LITELLM_API_KEY) in .env."
        ) from e
    return owui_client


def resolve_model() -> str:
    """The query model fixed at server startup (python -m dqm_mcp --model / OWUI_MODEL)."""
    name = get_settings().model
    if not name:
        raise RuntimeError(
            "No model configured on the server: start it with --model or set OWUI_MODEL in .env."
        )
    return name


def resolve_provider() -> str | None:
    """The query provider fixed at server startup (--provider / DEFAULT_PROVIDER)."""
    return get_settings().provider or None


def references_for(image_path: Path) -> list[Path]:
    """Reference images (good/bad/cosmics) for this plot, if a ref_dir is provided."""
    ref_dir = get_settings().ref_dir
    if not ref_dir.is_dir():
        return []
    return _owui().find_reference_images(image_path, ref_dir)


def build_group_spec(
    paths: list[Path],
    stems: list[str],
    *,
    run: int | str,
    event_type: str,
) -> dict:
    """
    build_messages() for the first panel, then splice the remaining panels
    into the same user message so the model sees the whole plot.
    """
    o = _owui()
    from run_batch_cli import SYSTEM_PROMPT  # lazy: pulls pandas + owui_client

    primary = paths[0]
    meta = o.RunMetadata(event_type_map={int(str(run).lstrip("0") or 0): event_type})
    prompt = o.resolve_prompt("", primary, meta)
    spec = o.build_messages(
        prompt,
        system=SYSTEM_PROMPT,
        reference_images=references_for(primary),
        image_path=primary,
        context=o.YAMLContext(store_dir=get_settings().store_dir),
    )

    if len(paths) > 1:
        content = spec["messages"][-1]["content"]
        idx = next(
            i for i, block in enumerate(content)
            if block.get("type") == "text" and block.get("text") == PANEL_INTRO_SINGLE
        )
        content[idx]["text"] = PANEL_INTRO_MULTI.format(n=len(paths))
        # content[idx + 1] is the first panel's image block
        extra = []
        extra.append({"type": "text", "text": f"Panel 1 of {len(paths)}: {stems[0]}"})
        extra.append(content[idx + 1])
        for i, (p, stem) in enumerate(zip(paths[1:], stems[1:]), start=2):
            extra.append({"type": "text", "text": f"Panel {i} of {len(paths)}: {stem}"})
            extra.append(o._image_content_block(p))
        content[idx + 1:idx + 2] = extra
    return spec


def ask_model(
    paths: list[Path],
    stems: list[str],
    *,
    run: int | str,
    event_type: str,
    model: str,
) -> dict:
    o = _owui()
    spec = build_group_spec(paths, stems, run=run, event_type=event_type)
    cfg = o.ModelConfig(name=model, provider=resolve_provider())
    result = o.send_query(spec, model=cfg)
    result["image"] = ";".join(str(p) for p in paths)   # record every panel in the result file
    result["images"] = [str(p) for p in paths]
    result["rag_text"] = spec["rag_text"]
    result["provider"] = cfg.provider
    return result
