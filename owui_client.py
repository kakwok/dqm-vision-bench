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

Output layout (with run_id — preserves previous runs):
    results/
        <run_id>/
            <plotName>/
                <plotName>_<safe_model>_run<XXXXXX>.txt
            summary_<run_id>.csv

Output layout (no run_id — overwrites):
    results/
        <plotName>/
            <plotName>_<safe_model>_run<XXXXXX>.txt
        summary.csv
"""

import base64
import json
import os
import re
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import requests
from dotenv import load_dotenv

from rag_backends import (
    ContextBackend,
    LocalRAG,
    NoContext,
    OWUIContext,
    YAMLContext,
    retrieve_context,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

OWUI_API_KEY  = os.environ.get("OWUI_API_KEY", "")
OWUI_URL = os.environ.get("OWUI_URL", "https://openwebui.fnal.gov/").rstrip("/")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")
LITELLM_URL     = os.environ.get("LITELLM_URL", "").rstrip("/")
MODEL    = os.environ.get("OWUI_MODEL", "")
TIMEOUT  = int(os.environ.get("OWUI_TIMEOUT", "400"))

if not OWUI_API_KEY:
    raise EnvironmentError("OWUI_API_KEY not set -  needed for knowledge/RAG access.")
if not LITELLM_API_KEY:
    raise EnvironmentError("LITELLM_API_KEY not set - needed for model inference.")

# Knowledge/RAG calls: authenticated with OWUI key
_OWUI_HEADERS = {
    "Authorization": f"Bearer {OWUI_API_KEY}",
    "Content-Type": "application/json",
}
# Model inference + listing: authenticated with LiteLLM key
_LITELLM_HEADERS = {
    "Authorization": f"Bearer {LITELLM_API_KEY}",
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
def list_models(detail: bool = False) -> list:
    """
    Return available models — authenticated via LiteLLM key.

    Parameters
    ----------
    detail : False (default) → list of model ID strings
             True            → list of full model dicts from the API
    """
    r = requests.get(f"{LITELLM_URL}/v1/models", headers=_LITELLM_HEADERS, timeout=30)
    r.raise_for_status()
    models = r.json()["data"]
    return models if detail else [m["id"] for m in models]


def validate_models(models: "list[str | ModelConfig]") -> bool:
    """
    Check that every model in *models* is available on the LiteLLM server.
    Prints a summary and returns True if all are valid, False otherwise.
    OWUIContext models are not checked (server-side resolution).
    """
    available = set(list_models())
    ok = True
    for m in models:
        if isinstance(m, ModelConfig):
            name = m.name
            extras = [
                f"{f.name}={getattr(m, f.name)}"
                for f in fields(m)
                if f.name != "name" and getattr(m, f.name) is not None
            ]
            detail = f"  ({', '.join(extras)})" if extras else ""
        else:
            name = m
            detail = ""
        status = "OK" if name in available else "!! UNKNOWN"
        if name not in available:
            ok = False
        print(f"  {status}  {name}{detail}")
    return ok


_stem_map_cache: dict | None = None


def _get_stem_map() -> dict:
    global _stem_map_cache
    if _stem_map_cache is None:
        try:
            from shift_layout_helpers import build_stem_map
            _stem_map_cache = build_stem_map()
        except Exception:
            _stem_map_cache = {}
    return _stem_map_cache


def find_reference_images(image_path: str | Path, ref_dir: str | Path) -> list[Path]:
    """
    Return all reference images for *image_path* from *ref_dir*.

    Looks up the plot's subsystem and folder via the shift_layout stem map, then
    returns every .png whose name starts with the plot's base stem (grpN stripped).
    Typical results: good, bad, cosmics variants — all sorted alphabetically.

    Returns an empty list when the stem is unrecognised or the folder has no matches.
    """
    stem = re.sub(r"_run\d+$", "", Path(image_path).stem)
    ref_dir = Path(ref_dir)

    info = _get_stem_map().get(stem)
    if not info:
        return []

    subpath = ref_dir / info["subsystem"] / info["folder"]
    if not subpath.exists():
        return []

    base_stem = re.sub(r"_grp\d+$", "", stem)
    return sorted(subpath.glob(f"{base_stem}*.png"))


def find_reference_image(image_path: str | Path, ref_dir: str | Path) -> Path | None:
    """Return the first reference image for *image_path*, or None. See find_reference_images()."""
    refs = find_reference_images(image_path, ref_dir)
    return refs[0] if refs else None


def get_knowledge_map() -> dict[str, str]:
    """Return {name: id} for all knowledge collections visible to this user."""
    r = requests.get(f"{OWUI_URL}/api/v1/knowledge/", headers=_OWUI_HEADERS, timeout=30)
    r.raise_for_status()
    return {kb["name"]: kb["id"] for kb in r.json()["items"]}


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Model selection and inference settings."""
    name: str = ""
    image_token_budget: int | None = None  # Gemma 4 only; valid values: 70,140,280,560,1120


@dataclass
class RunMetadata:
    """
    Per-run context injected into each query prompt.

    Current fields
    --------------
    event_type_map : {run_number: event_type} — appends
                     'The plot is from a "<event_type>" event.' to the prompt.
                     Typical values: 'collisions', 'cosmics', 'circulating'.

    Planned fields
    --------------
    run_duration   : {run_number: float} — run duration in seconds; useful for
                     flagging plots from very short runs as potentially unreliable.
    fill_number    : {run_number: int} — LHC fill number; enables grouping runs
                     by fill for trend analysis.
    lumi_recorded  : {run_number: float} — recorded luminosity in pb^-1.
    """
    event_type_map: dict[int, str] = field(default_factory=dict)


def _resolve_prompt(base_prompt: str, image_path: Path, run_metadata: "RunMetadata | None") -> str:
    """Append event type to the prompt when run_metadata provides a mapping for this image's run."""
    if not run_metadata or not run_metadata.event_type_map:
        return base_prompt
    m = re.search(r"run(\d+)", image_path.stem)
    if not m:
        return base_prompt
    event = run_metadata.event_type_map.get(int(m.group(1)))
    if not event:
        return base_prompt
    suffix = f'The plot is from a "{event}" event.'
    return f"{base_prompt}\n{suffix}".strip() if base_prompt else suffix


# ---------------------------------------------------------------------------
# Core query — two-stage: build then send
# ---------------------------------------------------------------------------

def build_messages(
    prompt: str,
    *,
    system: str = "",
    reference_images: "list[str | Path] | str | Path | None" = None,
    image_path: str | Path | None = None,
    context: "ContextBackend | None" = None,
) -> dict:
    """
    Assemble the API message list without sending anything.

    Returns a spec dict with keys:
        prompt          — original prompt text
        system          — system prompt (may be empty)
        messages        — list ready for the API payload
        ref_list        — resolved reference image paths (list[Path])
        img_path        — resolved input image path (Path | None)
        rag_text        — retrieved context string; empty string if none
        context         — resolved ContextBackend (NoContext() if None was passed)

    Use send_query() to send the spec, or inspect it first:
        spec = build_messages(PROMPT, image_path=img, context=CONTEXT)
        print(spec['rag_text'])
        result = send_query(spec, model=ModelConfig(name='...'))
    """
    context = context or NoContext()

    if reference_images is None:
        ref_list: list[Path] = []
    elif isinstance(reference_images, (str, Path)):
        ref_list = [Path(reference_images)]
    else:
        ref_list = [Path(r) for r in reference_images]

    img_path = Path(image_path) if image_path else None

    rag_text = retrieve_context(context, image_path=img_path, prompt=prompt)

    content = []
    if ref_list:
        content.append({"type": "text", "text": "The following are reference images for this plot type:"})
        for ref in ref_list:
            content.append(_image_content_block(ref))
    if rag_text:
        label = "Instructions" if isinstance(context, YAMLContext) else "Relevant instructions"
        content.append({"type": "text", "text": f"{label}:\n\n{rag_text}"})
    if img_path:
        content.append({"type": "text", "text": "The following is the image to be judged:"})
        content.append(_image_content_block(img_path))
    content.append({"type": "text", "text": prompt})

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    return {
        "prompt":   prompt,
        "system":   system,
        "messages": messages,
        "ref_list": ref_list,
        "img_path": img_path,
        "rag_text": rag_text,
        "context":  context,
    }


def send_query(spec: dict, *, model: "ModelConfig | None" = None) -> dict:
    """
    Send a pre-built message spec and return a result dict.

    Parameters
    ----------
    spec  : Dict returned by build_messages().
    model : ModelConfig with .name and .image_token_budget.

    Returns
    -------
    dict with keys: prompt, image, reference_images, rag_backend,
                    model (requested), model_used (echoed by backend),
                    response, usage, load_latency_s, generation_latency_s,
                    latency_s, error
    """
    model = model or ModelConfig()

    context    = spec["context"]
    messages   = spec["messages"]
    ref_list   = spec["ref_list"]
    img_path   = spec["img_path"]
    prompt     = spec["prompt"]
    model_name = model.name or MODEL

    payload: dict = {"messages": messages, "stream": True}
    if model_name:
        payload["model"] = model_name
    if model.image_token_budget is not None and "gemma-4" in model_name.lower():
        payload["images_config"] = {"max_soft_tokens": model.image_token_budget}
    if isinstance(context, OWUIContext) and context.collection_ids:
        payload["files"] = [{"type": "collection", "id": cid} for cid in context.collection_ids]

    if isinstance(context, OWUIContext):
        url, headers = f"{OWUI_URL}/api/chat/completions", _OWUI_HEADERS
    else:
        url, headers = f"{LITELLM_URL}/v1/chat/completions", _LITELLM_HEADERS

    t0 = time.time()
    t_first_token = None
    full_response = ""
    usage = None

    try:
        with requests.post(url, headers=headers, json=payload,
                           timeout=TIMEOUT, stream=True) as r:
            r.raise_for_status()
            model_used = model_name
            for line in r.iter_lines():
                if not line:
                    continue
                if not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if t_first_token is None:
                    t_first_token = time.time()
                # The backend may echo a different model name than requested
                # (e.g. Ollama resolving a Modelfile alias to its base tag).
                # Keep this for diagnostics only; use model_name for filenames.
                model_used = chunk.get("model", model_used)
                if "usage" in chunk:
                    usage = chunk["usage"]
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {}).get("content", "")
                    if delta:
                        full_response += delta

        t_end = time.time()
        ttft = round(t_first_token - t0, 2) if t_first_token else None
        return {
            "prompt":               prompt,
            "image":                str(img_path) if img_path else None,
            "reference_images":     [str(r) for r in ref_list],
            "rag_backend":          type(context).__name__,
            "model":                model_name,
            "model_used":           model_used,
            "response":             full_response,
            "usage":                usage,
            "load_latency_s":       ttft,
            "generation_latency_s": round(t_end - t_first_token, 2) if t_first_token else None,
            "latency_s":            round(t_end - t0, 2),
            "error":                None,
        }
    except Exception as e:
        return {
            "prompt":               prompt,
            "image":                str(img_path) if img_path else None,
            "reference_images":     [str(r) for r in ref_list],
            "rag_backend":          type(context).__name__,
            "model":                model_name,
            "model_used":           model_name,
            "response":             None,
            "usage":                None,
            "load_latency_s":       None,
            "generation_latency_s": None,
            "latency_s":            round(time.time() - t0, 2),
            "error":                str(e),
        }


def query(
    prompt: str,
    *,
    system: str = "",
    reference_images: "list[str | Path] | str | Path | None" = None,
    image_path: str | Path | None = None,
    context: "ContextBackend | None" = None,
    model: "ModelConfig | None" = None,
) -> dict:
    """Convenience wrapper: build_messages() then send_query()."""
    spec = build_messages(
        prompt,
        system=system,
        reference_images=reference_images,
        image_path=image_path,
        context=context,
    )
    return send_query(spec, model=model)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _safe_model_name(model: str) -> str:
    """Sanitise a model ID for use as part of a filename or directory name."""
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

    subdirs = sorted(d for d in image_root.iterdir() if d.is_dir() and d.name != "ref")
    if subdirs:
        for plot_dir in subdirs:
            plot_name = plot_dir.name
            for img in sorted(
                p for p in plot_dir.iterdir()
                if p.is_file() and p.suffix.lower() in extensions
            ):
                pairs.append((plot_name, img))
    else:
        for img in sorted(
            p for p in image_root.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ):
            pairs.append((img.stem, img))

    return pairs


def resolve_output_dir(
    output_root: Path,
    plot_name: str,
    run_id: str | None,
) -> Path:
    """
    Return the output directory for a given plot / run_id combination.

    With run_id    → <output_root>/<run_id>/<plot_name>/
    Without run_id → <output_root>/<plot_name>/   (overwrite)
    """
    if run_id:
        return output_root / run_id / plot_name
    return output_root / plot_name


def resolve_output_file(out_dir: Path, image_path: Path, model: str) -> Path:
    """
    Build the output filename: <image_stem>_<safe_model>.txt
    e.g. ecal_occupancy_run386593_llama3.2-vision_latest.txt
    """
    return out_dir / f"{image_path.stem}_{_safe_model_name(model)}.txt"


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def batch_query_images(
    prompt: str,
    image_root: str | Path,
    models: "list[str | ModelConfig]",
    output_root: str | Path,
    *,
    run_id: str | None = None,
    system: str = "",
    ref_dir: str | Path | None = None,
    context: "ContextBackend | None" = None,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
    pairs: "list[tuple[str, Path]] | None" = None,
    delay: float = 1.0,
    verbose: bool = True,
    run_metadata: "RunMetadata | None" = None,
) -> list[dict]:
    """
    Iterate over all images in <image_root>/<plotName>/ × each model and
    write responses following the flat output layout.

    Parameters
    ----------
    prompt       : Prompt sent with every image.
    image_root   : Root folder containing per-plot subdirectories.
    models       : List of model IDs (str) or ModelConfig objects.
    output_root  : Root folder for results.
    run_id       : Optional string tag (e.g. 'v1', 'baseline').
                   If given, outputs land under <output_root>/<run_id>/.
                   If None (default), outputs overwrite previous results.
    system       : Optional system prompt applied to all queries.
    ref_dir      : Optional ref_images root. find_reference_images() returns all
                   matching reference files. Leave None to send no references.
    context      : ContextBackend controlling retrieval (LocalRAG, YAMLContext,
                   OWUIContext, NoContext). Defaults to NoContext.
    extensions   : Image file extensions to include.
    pairs        : Pre-built (plot_name, image_path) pairs; skips image_root scan.
    delay        : Seconds to sleep between API calls.
    verbose      : Print progress to stdout.
    run_metadata : Optional RunMetadata with event_type_map {run_number: event_type}.
                   Appends the event type to the prompt for each image.

    Returns
    -------
    List of result dicts, one per (model, plot_name, image) combination.
    """
    image_root  = Path(image_root)
    output_root = Path(output_root)
    ref_dir     = Path(ref_dir) if ref_dir else None
    context     = context or NoContext()

    if pairs is None:
        pairs = _collect_images(image_root, extensions)
    if not pairs:
        raise FileNotFoundError(f"No images found under {image_root}")

    # Normalise models to ModelConfig objects
    model_configs = [
        m if isinstance(m, ModelConfig) else ModelConfig(name=m)
        for m in models
    ]

    total = len(model_configs) * len(pairs)
    results: list[dict] = []
    n = 0

    for model_cfg in model_configs:
        for plot_name, image_path in pairs:
            n += 1

            refs = find_reference_images(image_path, ref_dir) if ref_dir is not None else []
            resolved_prompt = _resolve_prompt(prompt, image_path, run_metadata)

            if verbose:
                ref_label = ", ".join(r.name for r in refs) if refs else "none"
                extras = [
                    f"{f.name}={getattr(model_cfg, f.name)}"
                    for f in fields(model_cfg)
                    if f.name != "name" and getattr(model_cfg, f.name) is not None
                ]
                model_label = model_cfg.name
                if extras:
                    model_label += f"[{', '.join(extras)}]"
                print(
                    f"[{n}/{total}] model={model_label}  "
                    f"plot={plot_name}  image={image_path.name}  refs=[{ref_label}] ...",
                    end=" ", flush=True,
                )

            result = query(
                resolved_prompt,
                model=model_cfg,
                system=system,
                reference_images=refs,
                context=context,
                image_path=image_path,
            )
            result["plot_name"] = plot_name
            results.append(result)

            out_dir = resolve_output_dir(output_root, plot_name, run_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = resolve_output_file(out_dir, image_path, model_cfg.name)

            with open(out_file, "w") as f:
                f.write(f"Model:      {result['model']}\n")
                f.write(f"Plot:       {plot_name}\n")
                f.write(f"Image:      {result['image']}\n")
                refs_str = ", ".join(Path(r).name for r in result["reference_images"])
                f.write(f"References: {refs_str or '(none)'}\n")
                f.write(f"Run ID:     {run_id or '(overwrite)'}\n")
                f.write(f"Load latency:       {result['load_latency_s']}s\n")
                f.write(f"Generation latency: {result['generation_latency_s']}s\n")
                f.write(f"Total latency:      {result['latency_s']}s\n")
                f.write(f"Prompt:     {result['prompt']}\n")
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

