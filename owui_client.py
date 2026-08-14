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
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Union

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
class DirectImages:
    """Send both reference and input images directly to the model (default)."""
    pass


@dataclass
class CaptionedReferences:
    """Caption reference images as text; input image still sent directly."""
    caption_model: str | None = None  # None -> self-caption with the model under test


@dataclass
class CaptionedBoth:
    """Caption both reference and input images as text; no raw images sent."""
    caption_model: str | None = None  # None -> self-caption with the model under test


ImageMode = Union[DirectImages, CaptionedReferences, CaptionedBoth]


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


@dataclass
class BatchConfig:
    """Complete configuration for one batch query run."""
    run_id: str
    image_root: Path
    models: list                          # list[str | ModelConfig]
    context: "ContextBackend" = field(default_factory=lambda: NoContext())
    image_mode: "ImageMode" = field(default_factory=lambda: DirectImages())
    system_prompt: str = ""
    prompt: str = ""
    output_root: Path = field(default_factory=lambda: Path("results"))
    ref_dir: "Path | None" = None
    plot_filter: "list[str] | None" = None
    delay: float = 1.5
    run_metadata: "RunMetadata | None" = None

    def build_pairs(self) -> "list[tuple[str, Path]]":
        """Collect and filter (plot_name, image_path) pairs from image_root."""
        all_pairs = _collect_images(
            Path(self.image_root), (".png", ".jpg", ".jpeg", ".webp")
        )
        if not self.plot_filter:
            return all_pairs
        return [(p, img) for p, img in all_pairs
                if any(f in p for f in self.plot_filter)]

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of this config."""
        def _tagged(obj) -> dict:
            d: dict = {"type": type(obj).__name__}
            for f in fields(obj):
                v = getattr(obj, f.name)
                d[f.name] = str(v) if isinstance(v, Path) else v
            return d

        def _model(m):
            if isinstance(m, ModelConfig):
                return {f.name: getattr(m, f.name) for f in fields(m)}
            return m

        return {
            "run_id":        self.run_id,
            "image_root":    str(self.image_root),
            "output_root":   str(self.output_root),
            "ref_dir":       str(self.ref_dir) if self.ref_dir else None,
            "plot_filter":   self.plot_filter,
            "models":        [_model(m) for m in self.models],
            "context":       _tagged(self.context),
            "image_mode":    _tagged(self.image_mode),
            "system_prompt": self.system_prompt,
            "prompt":        self.prompt,
            "delay":         self.delay,
            "run_metadata":  asdict(self.run_metadata) if self.run_metadata else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BatchConfig":
        """Reconstruct a BatchConfig from the dict produced by to_dict()."""
        context_classes = {
            "LocalRAG": LocalRAG,
            "YAMLContext": YAMLContext,
            "OWUIContext": OWUIContext,
            "NoContext": NoContext,
        }
        image_mode_classes = {
            "DirectImages": DirectImages,
            "CaptionedReferences": CaptionedReferences,
            "CaptionedBoth": CaptionedBoth,
        }

        def _model(m):
            return ModelConfig(**m) if isinstance(m, dict) else m

        def _context(d_ctx: "dict | None"):
            if not d_ctx:
                return NoContext()
            kwargs = {k: v for k, v in d_ctx.items() if k != "type"}
            return context_classes[d_ctx["type"]](**kwargs)

        def _image_mode(d_mode: "dict | None"):
            if not d_mode:
                return DirectImages()
            kwargs = {k: v for k, v in d_mode.items() if k != "type"}
            return image_mode_classes[d_mode["type"]](**kwargs)

        run_metadata = None
        if d.get("run_metadata"):
            event_type_map = {
                int(k): v for k, v in d["run_metadata"].get("event_type_map", {}).items()
            }
            run_metadata = RunMetadata(event_type_map=event_type_map)

        return cls(
            run_id=d["run_id"],
            image_root=Path(d["image_root"]),
            models=[_model(m) for m in d["models"]],
            context=_context(d.get("context")),
            image_mode=_image_mode(d.get("image_mode")),
            system_prompt=d.get("system_prompt", ""),
            prompt=d.get("prompt", ""),
            output_root=Path(d.get("output_root", "results")),
            ref_dir=Path(d["ref_dir"]) if d.get("ref_dir") else None,
            plot_filter=d.get("plot_filter"),
            delay=d.get("delay", 1.5),
            run_metadata=run_metadata,
        )

    @classmethod
    def from_json(cls, path: "str | Path") -> "BatchConfig":
        """Load a BatchConfig from a JSON file written by to_dict() (e.g. the config_*.json run_batch() saves)."""
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


def resolve_prompt(base_prompt: str, image_path: Path, run_metadata: "RunMetadata | None") -> str:
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
    reference_captions: "list[str] | None" = None,
    image_path: str | Path | None = None,
    image_caption: "str | None" = None,
    context: "ContextBackend | None" = None,
) -> dict:
    """
    Assemble the API message list without sending anything.

    reference_captions / image_caption, when given, are sent as text in place
    of the corresponding image(s) — see resolve_image_inputs(). ref_list /
    img_path are still resolved and returned either way, so the spec always
    records which image(s) were involved even when captions were sent instead.

    Returns a spec dict with keys:
        prompt              — original prompt text
        system              — system prompt (may be empty)
        messages            — list ready for the API payload
        ref_list            — resolved reference image paths (list[Path])
        img_path            — resolved input image path (Path | None)
        reference_captions  — reference captions actually sent (list[str])
        image_caption       — input image caption actually sent (str | None)
        rag_text            — retrieved context string; empty string if none
        context             — resolved ContextBackend (NoContext() if None was passed)

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
    reference_captions = reference_captions or []

    rag_text = retrieve_context(context, image_path=img_path, prompt=prompt)

    content = []
    if reference_captions:
        content.append({"type": "text", "text": "The following are captions describing reference images for this plot type:"})
        for cap in reference_captions:
            content.append({"type": "text", "text": cap})
    elif ref_list:
        content.append({"type": "text", "text": "The following are reference images for this plot type:"})
        for ref in ref_list:
            content.append(_image_content_block(ref))
    if rag_text:
        label = "Instructions" if isinstance(context, YAMLContext) else "Relevant instructions"
        content.append({"type": "text", "text": f"{label}:\n\n{rag_text}"})
    if image_caption:
        content.append({"type": "text", "text": f"The following is a caption describing the image to be judged:\n\n{image_caption}"})
    elif img_path:
        content.append({"type": "text", "text": "The following is the image to be judged:"})
        content.append(_image_content_block(img_path))
    content.append({"type": "text", "text": prompt})

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    return {
        "prompt":             prompt,
        "system":             system,
        "messages":           messages,
        "ref_list":           ref_list,
        "img_path":           img_path,
        "reference_captions": reference_captions,
        "image_caption":      image_caption,
        "rag_text":           rag_text,
        "context":            context,
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

    context             = spec["context"]
    messages            = spec["messages"]
    ref_list            = spec["ref_list"]
    img_path            = spec["img_path"]
    reference_captions  = spec.get("reference_captions", [])
    image_caption       = spec.get("image_caption")
    prompt              = spec["prompt"]
    model_name          = model.name or MODEL

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
            "reference_captions":   reference_captions,
            "image_caption":        image_caption,
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
            "reference_captions":   reference_captions,
            "image_caption":        image_caption,
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
    reference_captions: "list[str] | None" = None,
    image_path: str | Path | None = None,
    image_caption: "str | None" = None,
    context: "ContextBackend | None" = None,
    model: "ModelConfig | None" = None,
) -> dict:
    """Convenience wrapper: build_messages() then send_query()."""
    spec = build_messages(
        prompt,
        system=system,
        reference_images=reference_images,
        reference_captions=reference_captions,
        image_path=image_path,
        image_caption=image_caption,
        context=context,
    )
    return send_query(spec, model=model)


# ---------------------------------------------------------------------------
# Captioning
# ---------------------------------------------------------------------------

DEFAULT_CAPTION_PROMPT = (
    "Describe this DQM plot in detail: the axis labels, ranges, the overall "
    "shape of the distribution, and any visible anomalies, empty regions, or "
    "outliers. Be objective and factual — do not judge whether the plot looks "
    "good or bad."
)


def caption_image(
    image_path: str | Path,
    *,
    model: ModelConfig,
    prompt: str = DEFAULT_CAPTION_PROMPT,
) -> str:
    """Caption one image using *model*. Raises RuntimeError on API error."""
    result = query(prompt, model=model, image_path=image_path, context=NoContext())
    if result["error"]:
        raise RuntimeError(f"Captioning failed for {image_path}: {result['error']}")
    return result["response"]


def resolve_image_inputs(
    image_mode: "ImageMode",
    *,
    image_path: "Path | None",
    reference_images: list[Path],
    eval_model: ModelConfig,
) -> dict:
    """
    Resolve *image_mode* into caption text for build_messages()/query().

    Returns {"image_caption": str | None, "reference_captions": list[str]}.
    DirectImages returns both empty/None — raw images should be sent as-is.
    For the captioned modes, *eval_model* is used as the captioner unless
    image_mode.caption_model overrides it (self-caption by default: whichever
    model is under test also captions its own images).
    """
    if isinstance(image_mode, DirectImages):
        return {"image_caption": None, "reference_captions": []}

    cap_model = (
        ModelConfig(name=image_mode.caption_model) if image_mode.caption_model else eval_model
    )
    reference_captions = [caption_image(r, model=cap_model) for r in reference_images]

    if isinstance(image_mode, CaptionedReferences):
        return {"image_caption": None, "reference_captions": reference_captions}
    elif isinstance(image_mode, CaptionedBoth):
        image_caption = caption_image(image_path, model=cap_model) if image_path else None
        return {"image_caption": image_caption, "reference_captions": reference_captions}
    else:
        raise TypeError(f"Unhandled image mode: {type(image_mode).__name__}")


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


def _write_result_file(out_file: Path, result: dict, plot_name: str, run_id: "str | None") -> None:
    """Write a single query result to its .txt output file."""
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
        if result.get("reference_captions"):
            f.write(f"Reference captions: {json.dumps(result['reference_captions'])}\n")
        if result.get("image_caption"):
            f.write(f"Image caption: {json.dumps(result['image_caption'])}\n")
        f.write("-" * 60 + "\n")
        if result["error"]:
            f.write(f"ERROR: {result['error']}\n")
        else:
            f.write(result["response"] + "\n")


def _parse_result_file(path: Path, plot_name: str) -> dict:
    """Reconstruct a result dict from a previously written .txt output file."""
    header, _, body = path.read_text().partition("-" * 60 + "\n")
    meta = {}
    for line in header.splitlines():
        key, sep, val = line.partition(":")
        if sep:
            meta[key.strip()] = val.strip()

    def _num(s: "str | None") -> "float | None":
        return float(s.rstrip("s")) if s and s != "None" else None

    error, response = None, None
    if body.startswith("ERROR: "):
        error = body[len("ERROR: "):].rstrip("\n")
    else:
        response = body.rstrip("\n") if body else None

    refs = meta.get("References", "")
    reference_captions = json.loads(meta["Reference captions"]) if "Reference captions" in meta else []
    image_caption = json.loads(meta["Image caption"]) if "Image caption" in meta else None
    return {
        "prompt":               meta.get("Prompt", ""),
        "image":                meta.get("Image"),
        "reference_images":     [] if refs in ("", "(none)") else [r.strip() for r in refs.split(",")],
        "reference_captions":   reference_captions,
        "image_caption":        image_caption,
        "rag_backend":          None,
        "model":                meta.get("Model"),
        "model_used":           meta.get("Model"),
        "response":             response,
        "usage":                None,
        "load_latency_s":       _num(meta.get("Load latency")),
        "generation_latency_s": _num(meta.get("Generation latency")),
        "latency_s":            _num(meta.get("Total latency")),
        "error":                error,
        "plot_name":            plot_name,
    }


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
    image_mode: "ImageMode | None" = None,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
    pairs: "list[tuple[str, Path]] | None" = None,
    delay: float = 1.0,
    verbose: bool = True,
    run_metadata: "RunMetadata | None" = None,
    overwrite: bool = False,
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
    image_mode   : ImageMode controlling whether reference/input images are sent
                   directly or captioned first (DirectImages, CaptionedReferences,
                   CaptionedBoth). Defaults to DirectImages. Captioning defaults to
                   self-caption (the model under test captions its own images);
                   override via image_mode.caption_model. Not cached separately —
                   the caption used is written into the .txt result file, and the
                   existing skip-if-exists behavior avoids re-captioning on reruns.
    extensions   : Image file extensions to include.
    pairs        : Pre-built (plot_name, image_path) pairs; skips image_root scan.
    delay        : Seconds to sleep between API calls.
    verbose      : Print progress to stdout.
    run_metadata : Optional RunMetadata with event_type_map {run_number: event_type}.
                   Appends the event type to the prompt for each image.
    overwrite    : If False (default), a (model, image) pair whose output .txt
                   already exists is skipped — the existing file is parsed back
                   into a result instead of re-querying. If True, always re-query.

    Returns
    -------
    List of result dicts, one per (model, plot_name, image) combination.
    """
    image_root  = Path(image_root)
    output_root = Path(output_root)
    ref_dir     = Path(ref_dir) if ref_dir else None
    context     = context or NoContext()
    image_mode  = image_mode or DirectImages()

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

            out_dir = resolve_output_dir(output_root, plot_name, run_id)
            out_file = resolve_output_file(out_dir, image_path, model_cfg.name)

            if not overwrite and out_file.exists():
                result = _parse_result_file(out_file, plot_name)
                results.append(result)
                if verbose:
                    print(
                        f"[{n}/{total}] model={model_cfg.name}  "
                        f"plot={plot_name}  image={image_path.name} ... skip (exists) → {out_file}"
                    )
                continue

            refs = find_reference_images(image_path, ref_dir) if ref_dir is not None else []
            resolved_prompt = resolve_prompt(prompt, image_path, run_metadata)

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
                mode_label = "" if isinstance(image_mode, DirectImages) else f"  mode={type(image_mode).__name__}"
                print(
                    f"[{n}/{total}] model={model_label}  "
                    f"plot={plot_name}  image={image_path.name}  refs=[{ref_label}]{mode_label} ...",
                    end=" ", flush=True,
                )

            resolved_images = resolve_image_inputs(
                image_mode, image_path=image_path, reference_images=refs, eval_model=model_cfg
            )

            result = query(
                resolved_prompt,
                model=model_cfg,
                system=system,
                reference_images=refs,
                reference_captions=resolved_images["reference_captions"],
                context=context,
                image_path=image_path,
                image_caption=resolved_images["image_caption"],
            )
            result["plot_name"] = plot_name
            results.append(result)

            out_dir.mkdir(parents=True, exist_ok=True)
            _write_result_file(out_file, result, plot_name, run_id)

            if verbose:
                status = "ERROR" if result["error"] else f"{result['latency_s']}s → {out_file}"
                print(status)

            time.sleep(delay)

    return results


# ---------------------------------------------------------------------------
# BatchConfig helpers
# ---------------------------------------------------------------------------

def _merge_unique(old: list, new: list) -> list:
    """Union of two lists, de-duplicated and order-preserving. Items may be dicts (unhashable)."""
    def _key(item):
        return json.dumps(item, sort_keys=True) if isinstance(item, dict) else item

    seen = set()
    merged = []
    for item in old + new:
        k = _key(item)
        if k not in seen:
            seen.add(k)
            merged.append(item)
    return merged


def run_batch(config: BatchConfig, pairs: "list | None" = None, *, overwrite: bool = False) -> list[dict]:
    """
    Run a full batch query from a BatchConfig. Builds pairs from config if not provided.

    overwrite : If False (default), (model, image) pairs whose output .txt already
                exists are skipped and read back from disk instead of re-queried.
    """
    resolved_pairs = pairs if pairs is not None else config.build_pairs()

    out_root = Path(config.output_root)
    run_dir = out_root / config.run_id if config.run_id else out_root
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / f"config_{config.run_id or 'run'}.json"
    new_dict = config.to_dict()

    if config_path.exists():
        old_dict = json.loads(config_path.read_text())

        # plot_filter and models describe what's been run under this run_id so far —
        # merge rather than replace, so narrowing either to target new plots/models
        # doesn't erase the record of what earlier calls already covered.
        old_filter, new_filter = old_dict.get("plot_filter"), new_dict.get("plot_filter")
        if old_filter is None or new_filter is None:
            new_dict["plot_filter"] = None  # None means "all plots"
        else:
            new_dict["plot_filter"] = _merge_unique(old_filter, new_filter)

        new_dict["models"] = _merge_unique(old_dict.get("models", []), new_dict.get("models", []))

        drifted = [
            k for k in ("system_prompt", "prompt", "context", "image_mode", "image_root", "ref_dir")
            if old_dict.get(k) != new_dict.get(k)
        ]
        if drifted:
            print(
                f"WARNING: config_{config.run_id or 'run'}.json already exists with a different "
                f"{', '.join(drifted)} — outputs under this run_id may now reflect mixed settings."
            )

    with open(config_path, "w") as fh:
        json.dump(new_dict, fh, indent=2)
    print(f"Config saved: {config_path}")

    return batch_query_images(
        config.prompt,
        image_root=config.image_root,
        models=config.models,
        output_root=config.output_root,
        run_id=config.run_id,
        system=config.system_prompt,
        ref_dir=config.ref_dir,
        context=config.context,
        image_mode=config.image_mode,
        pairs=resolved_pairs,
        delay=config.delay,
        verbose=True,
        run_metadata=config.run_metadata,
        overwrite=overwrite,
    )


def retry_failed(
    config: BatchConfig,
    results: list[dict],
    *,
    delay: "float | None" = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Re-run any queries in `results` whose 'error' field is set.

    Retried results overwrite their .txt output file and are merged back into
    the returned list, keyed on (model, image); everything else is passed through
    unchanged.
    """
    failed = [(r["model"], r["image"]) for r in results if r["error"] is not None]
    if not failed:
        if verbose:
            print("No errors in results — nothing to retry.")
        return results

    delay = config.delay if delay is None else delay
    ref_dir = Path(config.ref_dir) if config.ref_dir else None

    if verbose:
        print(f'Retrying {len(failed)} failed quer{"y" if len(failed) == 1 else "ies"}...')

    retry_results = []
    for i, (model_name, image_str) in enumerate(failed):
        image_path = Path(image_str)
        plot_name = image_path.parent.name
        if verbose:
            print(f"  [{i+1}/{len(failed)}] model={model_name}  image={image_path.name} ...", end=" ", flush=True)

        refs = find_reference_images(image_path, ref_dir) if ref_dir is not None else []
        resolved_prompt = resolve_prompt(config.prompt, image_path, config.run_metadata)
        model_cfg = ModelConfig(name=model_name)
        resolved_images = resolve_image_inputs(
            config.image_mode, image_path=image_path, reference_images=refs, eval_model=model_cfg
        )

        result = query(
            resolved_prompt,
            model=model_cfg,
            system=config.system_prompt,
            reference_images=refs,
            reference_captions=resolved_images["reference_captions"],
            image_path=image_path,
            image_caption=resolved_images["image_caption"],
            context=config.context,
        )
        result["plot_name"] = plot_name
        retry_results.append(result)

        out_dir = resolve_output_dir(Path(config.output_root), plot_name, config.run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = resolve_output_file(out_dir, image_path, model_name)
        _write_result_file(out_file, result, plot_name, config.run_id)

        if verbose:
            status = "ERROR" if result["error"] else f"{result['latency_s']}s → {out_file}"
            print(status)
        time.sleep(delay)

    retry_index = {(r["model"], r["image"]): r for r in retry_results}
    merged = [retry_index.get((r["model"], r["image"]), r) for r in results]

    still_failing = sum(1 for r in retry_results if r["error"])
    if verbose:
        print(f"\nDone. {len(retry_results) - still_failing}/{len(retry_results)} recovered.")
        if still_failing:
            print("Still failing:")
            for r in retry_results:
                if r["error"]:
                    print(f"  {r['model']}  {Path(r['image']).name}  → {r['error']}")

    return merged


def sanity_check(config: BatchConfig, pairs: list) -> None:
    """Print a pre-flight summary for a BatchConfig."""
    ref_dir = Path(config.ref_dir) if config.ref_dir else None
    plot_names = sorted(set(p for p, _ in pairs))

    print(f"Run ID       : {config.run_id or '(none — overwrite mode)'}")
    print(f"Image root   : {config.image_root}")
    print(f"Output root  : {config.output_root}")
    print(f"Context      : {config.context!r}")
    print(f"Image mode   : {config.image_mode!r}")
    print(f"Run metadata : {config.run_metadata!r}")

    print(f"\nPlots found  : {len(plot_names)}")
    for pn in plot_names:
        imgs = [img for p, img in pairs if p == pn]
        print(f"  {pn}/  ({len(imgs)} images)")
        for img in imgs:
            refs = find_reference_images(img, ref_dir) if ref_dir else []
            if refs:
                for r in refs:
                    print(f"    {img.name}  ← {r.name}")
            else:
                print(f"    {img.name}  ← (no reference)")

    if ref_dir and ref_dir.exists():
        covered = sum(1 for _, img in pairs if find_reference_images(img, ref_dir))
        print(f"\nReference dir: {ref_dir}/  ({covered}/{len(pairs)} images have references)")
    else:
        print(f"\nReference dir: {ref_dir} — not found, reference images will be skipped")

    print(f"\nModels ({len(config.models)}):")
    validate_models(config.models)

    print(f"\nTotal queries: {len(pairs) * len(config.models)}")

    if pairs:
        print("\nExample output paths:")
        for model in config.models:
            model_name = model.name if isinstance(model, ModelConfig) else model
            pn, img = pairs[0]
            d = resolve_output_dir(Path(config.output_root), pn, config.run_id)
            f = resolve_output_file(d, img, model_name)
            print(f"  {f}")
