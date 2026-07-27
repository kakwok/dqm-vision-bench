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

import ast
import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

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
    #r = requests.get(f'{OWUI_URL}/api/models', headers=_OWUI_HEADERS, timeout=30)
    r.raise_for_status()
    models = r.json()["data"]
    return models if detail else [m["id"] for m in models]


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
# Local RAG from pre-chunked CSV
# ---------------------------------------------------------------------------

_chunks_cache: dict[str, pd.DataFrame] = {}
_bm25_cache:   dict[str, object] = {}

_NOISE_RE = re.compile(
    r"EditAttachPDF|Edit \| Attach|TWiki\? use Discourse|"
    r"Copyright &|If you are experiencing TWiki instability",
)
_FILE_LISTING_RE = re.compile(
    r"^\s*(png|jpg|gif)\s*$", re.MULTILINE,
)


def _is_noise(text: str) -> bool:
    if len(text.strip()) < 50:
        return True
    if _NOISE_RE.search(text):
        return True
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    manage_lines = sum(1 for ln in lines if ln == "manage" or ln == "r1")
    if manage_lines >= 3 and manage_lines / max(len(lines), 1) > 0.15:
        return True
    return False


def _load_chunks(csv_path: str | Path, filter_noise: bool = True) -> tuple[pd.DataFrame, object]:
    key = f"{csv_path}::{filter_noise}"
    if key not in _chunks_cache:
        from rank_bm25 import BM25Okapi
        df = pd.read_csv(csv_path)
        if filter_noise:
            keep = ~df["chunk_text"].apply(_is_noise)
            df = df[keep].reset_index(drop=True)
        df["_emb"] = df["embedding"].apply(
            lambda s: np.array(ast.literal_eval(s), dtype=np.float32)
        )
        tokenized = [t.lower().split() for t in df["chunk_text"].tolist()]
        _chunks_cache[key] = df
        _bm25_cache[key]   = BM25Okapi(tokenized)
    return _chunks_cache[key], _bm25_cache[key]


_embedder = None

def _embed_query(text: str) -> np.ndarray:
    global _embedder
    if _embedder is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return np.array(_embedder.embed_query(text), dtype=np.float32)


def _score_chunks(
    query_text: str,
    df: pd.DataFrame,
    bm25: object,
    method: str = "hybrid",
    alpha: float = 0.5,
    score_threshold: float = 0.35,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (top_idx, scores) arrays for the best-matching chunks."""
    q = _embed_query(query_text)
    emb_matrix = np.stack(df["_emb"].values)
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    row_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-10
    vec_scores = (emb_matrix / row_norms) @ q_norm

    if method == "threshold":
        mask = vec_scores >= score_threshold
        idx = np.where(mask)[0]
        top_idx = idx[np.argsort(vec_scores[idx])[::-1]][:top_k]
        return top_idx, vec_scores[top_idx]

    elif method == "top_k":
        top_idx = np.argsort(vec_scores)[::-1][:top_k]
        return top_idx, vec_scores[top_idx]

    else:  # hybrid
        tokens = query_text.lower().split()
        raw_bm25 = np.array(bm25.get_scores(tokens), dtype=np.float32)
        bm25_scores = raw_bm25 / (raw_bm25.max() + 1e-10)
        scores = alpha * vec_scores + (1 - alpha) * bm25_scores
        top_idx = np.argsort(scores)[::-1][:top_k]
        return top_idx, scores[top_idx]


def _chunk_source_label(row: pd.Series) -> str:
    """Extract a human-readable source label from chunk metadata."""
    try:
        meta = json.loads(row["metadata"].replace("'", '"'))
    except Exception:
        return ""
    title = meta.get("title", "")
    title = re.sub(r"\s*<\s*CMS\s*<\s*TWiki\s*$", "", title).strip()
    return title


def retrieve_chunks(
    query_text: str,
    csv_path: str | Path = "document_chunks.csv",
    top_k: int = 5,
    method: str = "hybrid",
    alpha: float = 0.5,
    score_threshold: float = 0.35,
) -> str:
    """
    Retrieve relevant chunks from csv_path and return them as a formatted
    context string.

    Parameters
    ----------
    query_text       : The query to search for.
    csv_path         : Path to document_chunks.csv.
    top_k            : Maximum number of chunks to return (used by "top_k" and "hybrid").
    method           : Retrieval strategy:
                         "hybrid"    — BM25 + vector scores combined, then top_k (default)
                         "top_k"     — vector similarity, top_k results
                         "threshold" — vector similarity, chunks above score_threshold,
                                       capped at top_k to avoid flooding the context window
    alpha            : Hybrid only. Weight for vector scores; (1-alpha) goes to BM25.
    score_threshold  : Threshold only. Minimum cosine similarity to include a chunk
                       (0.35 is a reasonable floor for all-MiniLM-L6-v2).
    """
    df, bm25 = _load_chunks(csv_path)
    top_idx, _ = _score_chunks(query_text, df, bm25, method, alpha, score_threshold, top_k)

    parts = []
    for i in top_idx:
        row = df.iloc[i]
        label = _chunk_source_label(row)
        header = f"[From: {label}]\n" if label else ""
        parts.append(f"{header}{row['chunk_text']}")
    return "\n\n---\n\n".join(parts)


def retrieve_and_inspect(
    query_text: str,
    csv_path: str | Path = "document_chunks.csv",
    top_k: int = 5,
    method: str = "hybrid",
    alpha: float = 0.5,
    score_threshold: float = 0.35,
) -> list[dict]:
    """
    Debug helper: retrieve chunks and return structured metadata for each.

    Returns a list of dicts with keys:
        rank, chunk_index, document_id, source, score, text_preview, text_length
    """
    df, bm25 = _load_chunks(csv_path)
    top_idx, top_scores = _score_chunks(query_text, df, bm25, method, alpha, score_threshold, top_k)

    results = []
    for rank, (i, score) in enumerate(zip(top_idx, top_scores), 1):
        row = df.iloc[i]
        text = row["chunk_text"]
        results.append({
            "rank": rank,
            "chunk_index": int(row["chunk_index"]),
            "document_id": int(row["document_id"]),
            "source": _chunk_source_label(row),
            "score": round(float(score), 4),
            "text_preview": text[:200].replace("\n", " "),
            "text_length": len(text),
        })
    return results


# ---------------------------------------------------------------------------
# YAML instruction store lookup
# ---------------------------------------------------------------------------

_yaml_cache: dict[str, dict] = {}

def lookup_instruction(stem: str, store_dir: str | Path = "plot_instructions") -> str:
    """
    Return the instruction text for a plot stem from plot_instructions/<subsystem>.yaml.

    Renders: description + quality_criteria + any non-expired known_issues.
    Returns empty string if the YAML file or the stem entry does not exist.
    """
    subsystem = stem.split("_")[0]
    yaml_path = Path(store_dir) / f"{subsystem}.yaml"
    if not yaml_path.exists():
        return ""

    cache_key = str(yaml_path.resolve())
    if cache_key not in _yaml_cache:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            _yaml_cache[cache_key] = yaml.safe_load(f)

    entry = _yaml_cache.get(cache_key, {}).get("plots", {}).get(stem)
    if not entry:
        return ""

    from datetime import date
    today = date.today().isoformat()

    parts: list[str] = []
    if entry.get("description"):
        parts.append(entry["description"].strip())
    if entry.get("quality_criteria"):
        parts.append(entry["quality_criteria"].strip())
    for issue in entry.get("known_issues", []):
        expires = issue.get("expires")
        if expires and str(expires) < today:
            continue
        parts.append(f"Known issue: {issue['text']}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    """Retrieval-Augmented Generation settings."""
    backend: str = "local"
    csv_path: str | Path | None = None
    top_k: int = 5
    method: str = "hybrid"
    alpha: float = 0.5
    score_threshold: float = 0.35
    collection_ids: list[str] | None = field(default=None)


@dataclass
class ModelConfig:
    """Model selection and inference settings."""
    name: str = ""
    image_token_budget: int | None = None  # Gemma 4 only; valid values: 70,140,280,560,1120


# ---------------------------------------------------------------------------
# Core query
# ---------------------------------------------------------------------------

def query(
    prompt: str,
    *,
    system: str = "",
    reference_images: "list[str | Path] | str | Path | None" = None,
    image_path: str | Path | None = None,
    rag: RAGConfig | None = None,
    model: ModelConfig | None = None,
    no_think: bool = False,
) -> dict:
    """
    Send a single query and return a result dict.

    The user message is built in four ordered parts:
      1. reference_images — zero or more reference images shown before context
      2. RAG context      — chunks from local CSV or OWUI knowledge collection
      3. image_path       — the input image to evaluate
      4. prompt           — the text instruction

    Parameters
    ----------
    prompt           : User message text.
    model            : ModelConfig with .name and .image_token_budget.
    system           : Optional system prompt.
    reference_images : Optional reference image(s) shown before RAG context.
    image_path       : Path to the input image to evaluate.
    rag              : RAGConfig controlling retrieval backend and parameters.
    no_think         : Append /no_think to prompt and system (Qwen3 thinking models).

    Returns
    -------
    dict with keys: prompt, image, reference_images, rag_backend,
                    model (requested), model_used (echoed by backend),
                    response, usage, load_latency_s, generation_latency_s,
                    latency_s, error
    """
    rag   = rag   or RAGConfig()
    model = model or ModelConfig()

    if rag.csv_path and rag.collection_ids:
        raise ValueError(
            "RAGConfig.csv_path and collection_ids are mutually exclusive. "
            "Set backend='local' with csv_path, or backend='owui' with collection_ids."
        )

    model_name = model.name or MODEL

    # Must run before building content so /no_think reaches the payload.
    if no_think:
        prompt = f"{prompt}\n/no_think"
        if system:
            system = f"{system}\n/no_think"

    # Normalise reference_images to a list of Path objects
    if reference_images is None:
        ref_list: list[Path] = []
    elif isinstance(reference_images, (str, Path)):
        ref_list = [Path(reference_images)]
    else:
        ref_list = [Path(r) for r in reference_images]

    content = []

    for ref in ref_list:
        content.append(_image_content_block(ref))

    if rag.backend == "local" and rag.csv_path:
        rag_query = Path(image_path).parent.name if image_path else prompt
        rag_context = retrieve_chunks(
            rag_query, csv_path=rag.csv_path, top_k=rag.top_k,
            method=rag.method, alpha=rag.alpha, score_threshold=rag.score_threshold,
        )
        content.append({"type": "text", "text": f"Relevant instructions:\n\n{rag_context}"})

    elif rag.backend == "yaml" and image_path:
        stem = Path(image_path).parent.name
        instruction = lookup_instruction(stem)
        if instruction:
            content.append({"type": "text", "text": f"Instructions:\n\n{instruction}"})

    if image_path:
        content.append(_image_content_block(image_path))

    content.append({"type": "text", "text": prompt})

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    payload: dict = {"messages": messages, "stream": True}
    if model_name:
        payload["model"] = model_name
    if model.image_token_budget is not None and "gemma-4" in model_name.lower():
        payload["images_config"] = {"max_soft_tokens": model.image_token_budget}
    if rag.backend == "owui" and rag.collection_ids:
        payload["files"] = [{"type": "collection", "id": cid} for cid in rag.collection_ids]

    if rag.backend == "owui":
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
            "image":                str(image_path) if image_path else None,
            "reference_images":     [str(r) for r in ref_list],
            "rag_backend":          rag.backend,
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
            "image":                str(image_path) if image_path else None,
            "reference_images":     [str(r) for r in ref_list],
            "rag_backend":          rag.backend,
            "model":                model_name,
            "model_used":           model_name,
            "response":             None,
            "usage":                None,
            "load_latency_s":       None,
            "generation_latency_s": None,
            "latency_s":            round(time.time() - t0, 2),
            "error":                str(e),
        }


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
    rag: RAGConfig | None = None,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
    pairs: "list[tuple[str, Path]] | None" = None,
    delay: float = 1.0,
    verbose: bool = True,
    prompt_map: "dict[Path, str] | None" = None,
) -> list[dict]:
    """
    Iterate over all images in <image_root>/<plotName>/ × each model and
    write responses following the flat output layout.

    Parameters
    ----------
    prompt     : Prompt sent with every image.
    image_root : Root folder containing per-plot subdirectories.
    models     : List of model IDs (str) or ModelConfig objects.
    output_root: Root folder for results.
    run_id     : Optional string tag (e.g. 'v1', 'baseline').
                 If given, outputs land under <output_root>/<run_id>/.
                 If None (default), outputs overwrite previous results.
    system     : Optional system prompt applied to all queries.
    ref_dir    : Optional ref_images root. find_reference_images() returns all
                 matching reference files. Leave None to send no references.
    rag        : RAGConfig controlling retrieval backend and parameters.
    extensions : Image file extensions to include.
    pairs      : Pre-built (plot_name, image_path) pairs; skips image_root scan.
    delay      : Seconds to sleep between API calls.
    verbose    : Print progress to stdout.
    prompt_map : Optional {image_path: prompt} dict for per-image prompts
                 (e.g. built from run number → event type). Takes precedence
                 over the static `prompt` argument.

    Returns
    -------
    List of result dicts, one per (model, plot_name, image) combination.
    """
    image_root  = Path(image_root)
    output_root = Path(output_root)
    ref_dir     = Path(ref_dir) if ref_dir else None
    rag         = rag or RAGConfig()

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
            resolved_prompt = prompt_map[image_path] if prompt_map and image_path in prompt_map else prompt

            if verbose:
                ref_label = ", ".join(r.name for r in refs) if refs else "none"
                print(
                    f"[{n}/{total}] model={model_cfg.name}  "
                    f"plot={plot_name}  image={image_path.name}  refs=[{ref_label}] ...",
                    end=" ", flush=True,
                )

            result = query(
                resolved_prompt,
                model=model_cfg,
                system=system,
                reference_images=refs,
                rag=rag,
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

