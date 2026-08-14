"""
rag_backends.py
---------------
Context backend configs and retrieval implementations.

Each backend is a plain dataclass — no logic, no lazy state, safe to serialize
and construct freely in notebooks or test fixtures.

retrieve_context() dispatches on type with assert_never for exhaustiveness:
adding a new backend without handling it here is a static type error.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Backend config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LocalRAG:
    """Retrieve context from a pre-chunked CSV using BM25 + vector search.

    The query text sent for image-driven lookups is a structured sentence built
    from plot_instructions metadata — "Find information about plot {title}
    numbered {plot_number} in subsystem {subsystem}" — rather than the raw
    image folder name, which retrieval-quality testing showed nearly doubles
    recall@5 (see rag_retrieval_investigation.ipynb, Round 3). Falls back to
    the raw stem when no matching plot_instructions entry exists.
    """
    csv_path: str | Path
    top_k: int = 5
    method: str = "hybrid"   # "hybrid" | "top_k" | "threshold"
    alpha: float = 0.8
    score_threshold: float = 0.35
    store_dir: str | Path = "plot_instructions"  # metadata source for the structured query


@dataclass
class YAMLContext:
    """Retrieve context from a YAML instruction store keyed by plot stem."""
    store_dir: str | Path = "plot_instructions"


@dataclass
class OWUIContext:
    """RAG handled server-side by OpenWebUI knowledge collections."""
    collection_ids: list[str] = field(default_factory=list)


@dataclass
class NoContext:
    """Send no RAG context — prompt and image only."""
    pass


ContextBackend = Union[LocalRAG, YAMLContext, OWUIContext, NoContext]


# ---------------------------------------------------------------------------
# Local RAG — CSV chunk retrieval
# ---------------------------------------------------------------------------

_chunks_cache: dict[str, pd.DataFrame] = {}
_bm25_cache:   dict[str, object] = {}

_NOISE_RE = re.compile(
    r"EditAttachPDF|Edit \| Attach|TWiki\? use Discourse|"
    r"Copyright &|If you are experiencing TWiki instability",
)
_FILE_LISTING_RE = re.compile(r"^\s*(png|jpg|gif)\s*$", re.MULTILINE)


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
    top_k            : Maximum number of chunks to return.
    method           : "hybrid" (BM25 + vector), "top_k" (vector only),
                       or "threshold" (vector above score_threshold, capped at top_k).
    alpha            : Hybrid only. Weight for vector scores; (1-alpha) for BM25.
    score_threshold  : Threshold only. Minimum cosine similarity to include a chunk.
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


def _load_yaml_doc(store_dir: str | Path, subsystem: str) -> "dict | None":
    """Load (and cache) plot_instructions/<subsystem>.yaml. None if it doesn't exist."""
    yaml_path = Path(store_dir) / f"{subsystem}.yaml"
    if not yaml_path.exists():
        return None
    cache_key = str(yaml_path.resolve())
    if cache_key not in _yaml_cache:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            _yaml_cache[cache_key] = yaml.safe_load(f)
    return _yaml_cache[cache_key]


def lookup_instruction(stem: str, store_dir: str | Path = "plot_instructions") -> str:
    """
    Return the instruction text for a plot stem from plot_instructions/<subsystem>.yaml.

    Renders: description + quality_criteria + any non-expired known_issues.
    Returns empty string if the YAML file or the stem entry does not exist.
    """
    doc = _load_yaml_doc(store_dir, stem.split("_")[0])
    entry = (doc or {}).get("plots", {}).get(stem)
    if not entry:
        return ""

    from datetime import date
    today = date.today().isoformat()

    parts: list[str] = []
    if entry.get("instruction"):
        parts.append(entry["instruction"].strip())
    for issue in entry.get("known_issues", []):
        expires = issue.get("expires")
        if expires and str(expires) < today:
            continue
        parts.append(f"Known issue: {issue['text']}")

    return "\n\n".join(parts)


def _build_structured_query(stem: str, store_dir: str | Path = "plot_instructions") -> "str | None":
    """
    Build a structured retrieval query from plot_instructions metadata:
    "Find information about plot {title} numbered {plot_number} in subsystem {subsystem}".

    Returns None if no plot_instructions entry exists for *stem* (e.g. a subsystem
    without curated instructions yet) — callers should fall back to the raw stem.
    """
    subsystem_guess = stem.split("_")[0]
    doc = _load_yaml_doc(store_dir, subsystem_guess)
    entry = (doc or {}).get("plots", {}).get(stem)
    if not entry or not entry.get("title"):
        return None
    title = entry["title"]
    plot_number = entry.get("plot_number", "")
    subsystem = doc.get("subsystem", subsystem_guess)
    return f"Find information about plot {title} numbered {plot_number} in subsystem {subsystem}"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def retrieve_context(
    context: ContextBackend,
    *,
    image_path: Path | None = None,
    prompt: str = "",
) -> str:
    """
    Return the RAG context string for the given backend.

    OWUIContext returns "" — retrieval is handled server-side via payload files.
    """
    if isinstance(context, NoContext):
        return ""
    elif isinstance(context, LocalRAG):
        query_text = prompt
        if image_path:
            stem = image_path.parent.name
            query_text = _build_structured_query(stem, context.store_dir) or stem
        return retrieve_chunks(
            query_text,
            csv_path=context.csv_path,
            top_k=context.top_k,
            method=context.method,
            alpha=context.alpha,
            score_threshold=context.score_threshold,
        )
    elif isinstance(context, YAMLContext):
        if not image_path:
            return ""
        stem = image_path.parent.name
        return lookup_instruction(stem, store_dir=context.store_dir)
    elif isinstance(context, OWUIContext):
        return ""
    else:
        raise TypeError(f"Unhandled context backend: {type(context).__name__}")
