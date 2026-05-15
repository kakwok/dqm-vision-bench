"""
owui_client.py
--------------
Core query functions for the FNAL OpenWebUI instance.
Import this from a notebook or script; do not run directly.

Requires in environment (or .env):
    OWUI_API_KEY   - Bearer token from OpenWebUI Settings > Account
    OWUI_URL       - Base URL of the OpenWebUI instance
    OWUI_MODEL     - (optional) default model ID

Expected image layout:
    images/
        <plotName>/
            <plotName>_run<XXXXXX>.png
            ...

Output layout (default, overwrite mode):
    results/
        <plotName>/
            <safe_model_name>/
                <plotName>_run<XXXXXX>.txt
                ...
        summary.csv

Output layout (with run_id):
    results/
        <plotName>/
            <safe_model_name>/
                <run_id>/
                    <plotName>_run<XXXXXX>.txt
                    ...
        summary_<run_id>.csv
"""

import base64
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

API_KEY  = os.environ.get("OWUI_API_KEY", "")
OWUI_URL = os.environ.get("OWUI_URL", "https://openwebui.fnal.gov/").rstrip("/")
MODEL    = os.environ.get("OWUI_MODEL", "")
TIMEOUT  = int(os.environ.get("OWUI_TIMEOUT", "120"))

if not API_KEY:
    raise EnvironmentError("OWUI_API_KEY not set. Check your .env file.")

_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

_MEDIA_TYPES = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
}


def encode_image(image_path: str | Path) -> tuple[str, str]:
    """
    Read an image file and return (base64_string, media_type).
    Raises FileNotFoundError if path does not exist.
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    media_type = _MEDIA_TYPES.get(p.suffix.lower(), "image/png")
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), media_type


def _image_content_block(image_path: str | Path) -> dict:
    b64, media_type = encode_image(image_path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{b64}"},
    }


# ---------------------------------------------------------------------------
# Model / knowledge helpers
# ---------------------------------------------------------------------------

def list_models() -> list[str]:
    """Return all model IDs available on this OpenWebUI instance."""
    r = requests.get(f"{OWUI_URL}/api/models", headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return [m["id"] for m in r.json()["data"]]


def get_knowledge_map() -> dict[str, str]:
    """Return {name: id} for all knowledge collections visible to this user."""
    r = requests.get(f"{OWUI_URL}/api/v1/knowledge/", headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return {kb["name"]: kb["id"] for kb in r.json()["items"]}


# ---------------------------------------------------------------------------
# Core query
# ---------------------------------------------------------------------------

def query(
    prompt: str,
    *,
    model: str = "",
    system: str = "",
    image_path: str | Path | None = None,
    collection_ids: list[str] | None = None,
) -> dict:
    """
    Send a single query to OpenWebUI and return a result dict.

    Parameters
    ----------
    prompt          : User message text.
    model           : Model ID. Falls back to OWUI_MODEL env var, then server default.
    system          : Optional system prompt.
    image_path      : Optional path to an image file (requires a vision model).
    collection_ids  : Optional list of RAG knowledge collection UUIDs.

    Returns
    -------
    dict with keys: prompt, image, model_used, response, latency_s, error
    """
    model = model or MODEL

    # Build message content
    if image_path:
        content = [
            _image_content_block(image_path),
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    payload: dict = {"messages": messages}
    if model:
        payload["model"] = model
    if collection_ids:
        payload["files"] = [{"type": "collection", "id": cid} for cid in collection_ids]

    t0 = time.time()
    try:
        r = requests.post(
            f"{OWUI_URL}/api/chat/completions",
            headers=_HEADERS,
            json=payload,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        return {
            "prompt":     prompt,
            "image":      str(image_path) if image_path else None,
            "model_used": data.get("model", model),
            "response":   data["choices"][0]["message"]["content"],
            "latency_s":  round(time.time() - t0, 2),
            "error":      None,
        }
    except Exception as e:
        return {
            "prompt":     prompt,
            "image":      str(image_path) if image_path else None,
            "model_used": model,
            "response":   None,
            "latency_s":  round(time.time() - t0, 2),
            "error":      str(e),
        }


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _safe_model_name(model: str) -> str:
    """Sanitise a model ID for use as a directory name."""
    return model.replace("/", "_").replace(":", "_")


def _collect_images(
    image_root: Path,
    extensions: tuple[str, ...],
) -> list[tuple[str, Path]]:
    """
    Walk <image_root>/<plotName>/<plotName>_run<XXXXXX>.ext and return
    a list of (plot_name, image_path) tuples, sorted by plot_name then filename.

    Falls back to flat layout (image_root/*.ext) if no subdirectories are found.
    """
    pairs: list[tuple[str, Path]] = []

    subdirs = sorted(d for d in image_root.iterdir() if d.is_dir())
    if subdirs:
        for plot_dir in subdirs:
            plot_name = plot_dir.name
            for img in sorted(
                p for p in plot_dir.iterdir()
                if p.is_file() and p.suffix.lower() in extensions
            ):
                pairs.append((plot_name, img))
    else:
        # Flat fallback: treat all images as belonging to a single "default" plot
        for img in sorted(
            p for p in image_root.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ):
            pairs.append((img.stem, img))

    return pairs


def resolve_output_dir(
    output_root: Path,
    plot_name: str,
    model: str,
    run_id: str | None,
) -> Path:
    """
    Return the output directory for a given plot / model / run_id combination.

    With run_id    → <output_root>/<plot_name>/<safe_model>/<run_id>/
    Without run_id → <output_root>/<plot_name>/<safe_model>/   (overwrite)
    """
    base = output_root / plot_name / _safe_model_name(model)
    return base / run_id if run_id else base


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def batch_query_images(
    prompt: str,
    image_root: str | Path,
    models: list[str],
    output_root: str | Path,
    *,
    run_id: str | None = None,
    system: str = "",
    collection_ids: list[str] | None = None,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
    delay: float = 1.0,
    verbose: bool = True,
) -> list[dict]:
    """
    Iterate over all images in <image_root>/<plotName>/ × each model and
    write responses to <output_root>/<plotName>/<safe_model>[/<run_id>]/.

    Parameters
    ----------
    prompt          : Prompt sent with every image.
    image_root      : Root folder containing per-plot subdirectories.
    models          : List of model IDs to iterate over.
    output_root     : Root folder for results (mirrors image_root structure).
    run_id          : Optional string tag (e.g. 'v1', 'baseline').
                      If given, outputs go into a subdirectory of that name
                      so previous results are preserved.
                      If None (default), outputs overwrite previous results.
    system          : Optional system prompt applied to all queries.
    collection_ids  : Optional RAG collection UUIDs attached to every query.
    extensions      : Image file extensions to include.
    delay           : Seconds to sleep between API calls.
    verbose         : Print progress to stdout.

    Returns
    -------
    List of result dicts, one per (plot_name, image, model) combination.
    Each dict has keys: plot_name, image, model_used, response, latency_s, error.
    """
    image_root  = Path(image_root)
    output_root = Path(output_root)

    pairs = _collect_images(image_root, extensions)
    if not pairs:
        raise FileNotFoundError(f"No images found under {image_root}")

    total = len(models) * len(pairs)
    results: list[dict] = []
    n = 0

    for model in models:
        for plot_name, image_path in pairs:
            n += 1
            if verbose:
                print(
                    f"[{n}/{total}] model={model}  "
                    f"plot={plot_name}  image={image_path.name} ...",
                    end=" ", flush=True,
                )

            result = query(
                prompt,
                model=model,
                system=system,
                image_path=image_path,
                collection_ids=collection_ids,
            )
            result["plot_name"] = plot_name
            results.append(result)

            # Resolve output path
            out_dir = resolve_output_dir(output_root, plot_name, model, run_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / (image_path.stem + ".txt")

            with open(out_file, "w") as f:
                f.write(f"Model:    {result['model_used']}\n")
                f.write(f"Plot:     {plot_name}\n")
                f.write(f"Image:    {result['image']}\n")
                f.write(f"Run ID:   {run_id or '(overwrite)'}\n")
                f.write(f"Latency:  {result['latency_s']}s\n")
                f.write(f"Prompt:   {result['prompt']}\n")
                f.write("-" * 60 + "\n")
                if result["error"]:
                    f.write(f"ERROR: {result['error']}\n")
                else:
                    f.write(result["response"])
                    f.write("\n")

            if verbose:
                status = "ERROR" if result["error"] else f"{result['latency_s']}s → {out_file}"
                print(status)

            time.sleep(delay)

    return results
