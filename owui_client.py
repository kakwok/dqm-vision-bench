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
TIMEOUT  = int(os.environ.get("OWUI_TIMEOUT", "120"))

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


def find_reference_image(image_path: str | Path, ref_dir: str | Path) -> Path | None:
    """
    Look up the reference image for *image_path* inside *ref_dir*.

    Expected ref_dir layout:
        <ref_dir>/<subsystem>/<folder>/<stem>[_run<XXXXXX>].png

    Example:
        image_path : images/Ecal_05_RecHitEnergy/Ecal_05_RecHitEnergy_grp0_run398185.png
        ref_dir    : ref_images
        returns    : ref_images/Ecal/Ecal_05_RecHitEnergy/Ecal_05_RecHitEnergy_grp0_run398185.png

    Subsystem and folder are resolved via the shift_layout stem map.
    Returns None if no matching file is found or the stem is not recognised.
    """
    stem = re.sub(r"_run\d+$", "", Path(image_path).stem)
    ref_dir = Path(ref_dir)

    info = _get_stem_map().get(stem)
    if not info:
        return None

    subpath = ref_dir / info["subsystem"] / info["folder"]
    matches = sorted(subpath.glob(f"{stem}_run*.png")) or sorted(subpath.glob(f"{stem}.png"))
    return matches[0] if matches else None


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
# Core query
# ---------------------------------------------------------------------------

def query(
    prompt: str,
    *,
    model: str = "",
    system: str = "",
    reference_image: str | Path | None = None,
    rag_backend: str = "local",
    csv_path: str | Path | None = None,
    top_k: int = 5,
    method: str = "hybrid",
    alpha: float = 0.5,
    score_threshold: float = 0.35,
    image_path: str | Path | None = None,
    collection_ids: list[str] | None = None,
) -> dict:
    """
    Send a single query to OpenWebUI and return a result dict.

    The user message is built in four ordered parts:
      1. reference_image  — optional "good example" image shown before context
      2. RAG context      — chunks from local CSV or OWUI knowledge collection
      3. image_path       — the input image to evaluate
      4. prompt           — the text instruction

    Parameters
    ----------
    prompt          : User message text.
    model           : Model ID. Falls back to OWUI_MODEL env var, then server default.
    system          : Optional system prompt.
    reference_image : Optional path to a reference/example image.
    rag_backend     : "local" (default) — retrieve from csv_path using retrieve_chunks.
                      "owui"            — delegate retrieval to OWUI via collection_ids.
    csv_path        : Path to document_chunks.csv. Required when rag_backend="local".
    top_k           : Max chunks to retrieve (used by "hybrid" and "top_k" methods).
    method          : Retrieval strategy — "hybrid" (default), "top_k", or "threshold".
    alpha           : Hybrid only. Weight for vector scores (1-alpha to BM25).
    score_threshold : Threshold only. Minimum cosine similarity to include a chunk.
    image_path      : Path to the input image to evaluate (requires a vision model).
    collection_ids  : OWUI knowledge collection UUIDs. Required when rag_backend="owui".

    Returns
    -------
    dict with keys: prompt, image, reference_image, rag_backend, model_used, response, latency_s, error
    """
    if csv_path and collection_ids:
        raise ValueError(
            "csv_path and collection_ids are mutually exclusive. "
            "Set rag_backend='local' with csv_path, or rag_backend='owui' with collection_ids."
        )

    model = model or MODEL

    content = []

    if reference_image:
        content.append(_image_content_block(reference_image))

    if rag_backend == "local" and csv_path:
        rag_context = retrieve_chunks(
            prompt, csv_path=csv_path, top_k=top_k,
            method=method, alpha=alpha, score_threshold=score_threshold,
        )
        content.append({"type": "text", "text": f"Relevant instructions:\n\n{rag_context}"})

    if image_path:
        content.append(_image_content_block(image_path))

    content.append({"type": "text", "text": prompt})

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    payload: dict = {"messages": messages}
    if model:
        payload["model"] = model
    if rag_backend == "owui" and collection_ids:
        payload["files"] = [{"type": "collection", "id": cid} for cid in collection_ids]

    if rag_backend == "owui":
        url, headers = f"{OWUI_URL}/api/chat/completions", _OWUI_HEADERS
    else:
        url, headers = f"{LITELLM_URL}/v1/chat/completions", _LITELLM_HEADERS

    t0 = time.time()
    try:
        r = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        return {
            "prompt":          prompt,
            "image":           str(image_path) if image_path else None,
            "reference_image": str(reference_image) if reference_image else None,
            "rag_backend":     rag_backend,
            "model_used":      data.get("model", model),
            "response":        data["choices"][0]["message"]["content"],
            "latency_s":       round(time.time() - t0, 2),
            "error":           None,
        }
    except Exception as e:
        return {
            "prompt":          prompt,
            "image":           str(image_path) if image_path else None,
            "reference_image": str(reference_image) if reference_image else None,
            "rag_backend":     rag_backend,
            "model_used":      model,
            "response":        None,
            "latency_s":       round(time.time() - t0, 2),
            "error":           str(e),
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
    models: list[str],
    output_root: str | Path,
    *,
    run_id: str | None = None,
    system: str = "",
    reference_image: str | Path | None = None,
    ref_dir: str | Path | None = None,
    rag_backend: str = "local",
    csv_path: str | Path | None = None,
    top_k: int = 5,
    method: str = "hybrid",
    alpha: float = 0.5,
    score_threshold: float = 0.35,
    collection_ids: list[str] | None = None,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
    delay: float = 1.0,
    verbose: bool = True,
) -> list[dict]:
    """
    Iterate over all images in <image_root>/<plotName>/ × each model and
    write responses following the flat output layout.

    Parameters
    ----------
    prompt          : Prompt sent with every image.
    image_root      : Root folder containing per-plot subdirectories.
    models          : List of model IDs to iterate over.
    output_root     : Root folder for results.
    run_id          : Optional string tag (e.g. 'v1', 'baseline').
                      If given, all outputs land under <output_root>/<run_id>/
                      so previous runs are preserved.
                      If None (default), outputs overwrite previous results.
    system          : Optional system prompt applied to all queries.
    reference_image : Optional single reference image sent for every query.
                      Ignored when ref_dir is set.
    ref_dir         : Optional directory of per-plot reference images (images/ref/).
                      For each input image, find_reference_image() looks for a
                      matching file by stem. Takes precedence over reference_image.
    rag_backend     : "local" (default) — use csv_path with retrieve_chunks.
                      "owui"            — delegate to OWUI via collection_ids.
    csv_path        : Path to document_chunks.csv. Required when rag_backend="local".
    top_k           : Number of chunks to retrieve from csv_path.
    method          : Retrieval strategy — "hybrid" (default), "top_k", or "threshold".
    alpha           : Hybrid only. Weight for vector scores (1-alpha to BM25).
    score_threshold : Threshold only. Minimum cosine similarity to include a chunk.
    collection_ids  : OWUI knowledge collection UUIDs. Required when rag_backend="owui".
    extensions      : Image file extensions to include.
    delay           : Seconds to sleep between API calls.
    verbose         : Print progress to stdout.

    Returns
    -------
    List of result dicts, one per (model, plot_name, image) combination.
    Each dict has keys: plot_name, image, reference_image, model_used, response, latency_s, error.
    """
    image_root  = Path(image_root)
    output_root = Path(output_root)
    ref_dir     = Path(ref_dir) if ref_dir else None

    pairs = _collect_images(image_root, extensions)
    if not pairs:
        raise FileNotFoundError(f"No images found under {image_root}")

    total = len(models) * len(pairs)
    results: list[dict] = []
    n = 0

    for model in models:
        for plot_name, image_path in pairs:
            n += 1

            # Resolve reference image: per-image lookup takes precedence
            if ref_dir is not None:
                ref_img = find_reference_image(image_path, ref_dir)
            else:
                ref_img = Path(reference_image) if reference_image else None

            if verbose:
                ref_label = ref_img.name if ref_img else "none"
                print(
                    f"[{n}/{total}] model={model}  "
                    f"plot={plot_name}  image={image_path.name}  ref={ref_label} ...",
                    end=" ", flush=True,
                )

            result = query(
                prompt,
                model=model,
                system=system,
                reference_image=ref_img,
                rag_backend=rag_backend,
                csv_path=csv_path,
                top_k=top_k,
                method=method,
                alpha=alpha,
                score_threshold=score_threshold,
                image_path=image_path,
                collection_ids=collection_ids,
            )
            result["plot_name"] = plot_name
            results.append(result)

            out_dir = resolve_output_dir(output_root, plot_name, run_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = resolve_output_file(out_dir, image_path, model)

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

