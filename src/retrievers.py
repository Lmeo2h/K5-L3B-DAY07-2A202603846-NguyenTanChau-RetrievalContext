"""Retrieval methods for Day 7: Lexical TF-IDF, Dense (ChromaDB + BAAI/bge-m3), Hybrid RRF, and improvements."""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "starter") not in sys.path:
    sys.path.insert(0, str(ROOT / "starter"))

from baseline import TfidfIndex, pack_context  # noqa: E402

DEFAULT_DAY6_CHROMA_DIR = Path(r"S:\Lab6\K5-L3-Day06-DataIngestionEngineering\data\chroma_db")
DEFAULT_COLLECTION = "strategy_b"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

KNOWN_SECTIONS = (
    "Introduction",
    "LLM Output Configuration",
    "Prompting Techniques",
    "Code Prompting",
    "Best Practices",
    "Summary",
)


def matches_filters(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Check whether a chunk's metadata satisfies all key-value pairs in filters."""
    if not filters:
        return True
    meta = chunk.get("metadata") or {}
    for key, val in filters.items():
        if key == "page" and chunk.get("page") == val:
            continue
        if meta.get(key) != val:
            return False
    return True


def to_chroma_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert a flat filter dict into a ChromaDB `where` clause."""
    if not filters:
        return None
    items = [{k: v} for k, v in filters.items() if v is not None]
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    return {"$and": items}


def infer_section_filter(query: str) -> dict[str, Any] | None:
    """Infer a section filter only when the query explicitly names a known section."""
    q_low = unicodedata.normalize("NFC", query).casefold()
    for section in KNOWN_SECTIONS:
        if section.casefold() in q_low:
            return {"section": section}
    return None


def _normalize_chunk(chunk_id: str, text: str, meta: dict[str, Any], score: float, strategy: str) -> dict[str, Any]:
    return {
        "id": str(chunk_id),
        "source_id": str(meta.get("document_id", "")),
        "source": str(meta.get("source", "")),
        "page": int(meta["page"]),
        "text": re.sub(r"\s+", " ", str(text)).strip(),
        "metadata": dict(meta),
        "score": float(score),
        "strategy": strategy,
    }


class LexicalTfidfRetriever:
    """Method 1: TF-IDF lexical retriever over exported chunks."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.index = TfidfIndex(chunks)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        raw = self.index.search(query, top_k=len(self.chunks))
        filtered = [c for c in raw if matches_filters(c, filters)]
        return filtered[:top_k]


class DenseChromaRetriever:
    """Method 2: Dense retriever using Day 6 ChromaDB and BAAI/bge-m3."""

    def __init__(
        self,
        chroma_dir: Path = DEFAULT_DAY6_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        infer_filter: bool = False,
        embedder: SentenceTransformer | None = None,
    ) -> None:
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_collection(name=collection_name)
        self.embedder = embedder or SentenceTransformer(embedding_model, local_files_only=True)
        self.infer_filter = infer_filter

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        effective_filters = filters
        if effective_filters is None and self.infer_filter:
            effective_filters = infer_section_filter(query)
        where = to_chroma_where(effective_filters)
        q_vec = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
        kwargs: dict[str, Any] = {
            "query_embeddings": [q_vec],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where
        res = self.collection.query(**kwargs)
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[dict[str, Any]] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            score = 1.0 - float(dist)
            out.append(_normalize_chunk(cid, doc, meta or {}, score, "dense"))
        return out


class HybridRrfRetriever:
    """Method 3: Hybrid retriever combining Dense and Lexical via Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense: DenseChromaRetriever,
        lexical: LexicalTfidfRetriever,
        rrf_k: int = 60,
        candidate_multiplier: int = 5,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        cand_k = max(top_k * self.candidate_multiplier, 15)
        dense_hits = self.dense.search(query, filters=filters, top_k=cand_k)
        lex_hits = self.lexical.search(query, filters=filters, top_k=cand_k)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}
        for rank, item in enumerate(dense_hits, start=1):
            cid = item["id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, item in enumerate(lex_hits, start=1):
            cid = item["id"]
            chunk_map.setdefault(cid, item)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked_ids = sorted(rrf_scores.keys(), key=lambda cid: (-rrf_scores[cid], cid))[:top_k]
        return [{**chunk_map[cid], "score": rrf_scores[cid], "strategy": "hybrid_rrf"} for cid in ranked_ids]


class QueryRewriteRetriever:
    """Wrapper that rewrites Vietnamese/user queries into English technical search queries via Gemini."""

    def __init__(self, base_retriever: Any, model: str = "gemini-3.5-flash-lite") -> None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in environment/.env")
        self.client = genai.Client(api_key=api_key)
        self.base = base_retriever
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def rewrite(self, query: str) -> str:
        prompt = (
            "Rewrite the following question into a concise English technical search query "
            "(keywords and key phrases only, 6-15 words) to search a Google Prompt Engineering PDF. "
            "Output ONLY the English search query, no explanation.\n\n"
            f"Question: {query}"
        )
        resp = self.client.models.generate_content(model=self.model, contents=prompt)
        usage = getattr(resp, "usage_metadata", None)
        if usage is not None:
            self.total_input_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
            self.total_output_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)
        rewritten = (resp.text or "").strip()
        return rewritten if rewritten else query

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        rewritten = self.rewrite(query)
        results = self.base.search(rewritten, filters=filters, top_k=top_k)
        return [{**r, "rewritten_query": rewritten} for r in results]


def expand_with_neighbors(
    results: list[dict[str, Any]],
    all_chunks: list[dict[str, Any]],
    max_total: int = 6,
) -> list[dict[str, Any]]:
    """Expand retrieved chunks with adjacent chunks (+/-1 chunk_index) on the same page."""
    by_page_idx: dict[tuple[str, int, int], dict[str, Any]] = {}
    for c in all_chunks:
        idx = (c.get("metadata") or {}).get("chunk_index")
        if idx is not None:
            by_page_idx[(c["source_id"], int(c["page"]), int(idx))] = c

    expanded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in results:
        if r["id"] not in seen:
            expanded.append(r)
            seen.add(r["id"])
        meta = r.get("metadata") or {}
        idx = meta.get("chunk_index")
        if idx is None:
            continue
        for delta in (1, -1):
            if len(expanded) >= max_total:
                break
            nb = by_page_idx.get((r["source_id"], int(r["page"]), int(idx) + delta))
            if nb and nb["id"] not in seen:
                expanded.append({**nb, "score": r.get("score", 0.0) * 0.9, "strategy": f"{r.get('strategy', 'base')}+neighbor"})
                seen.add(nb["id"])
    return expanded[:max_total]


def pack_with_budget(
    results: list[dict[str, Any]],
    budget: int = 2400,
    all_chunks: list[dict[str, Any]] | None = None,
    neighbor_expand: bool = False,
) -> tuple[str, list[str]]:
    """Pack retrieved chunks (optionally neighbor-expanded) within character budget."""
    candidates = expand_with_neighbors(results, all_chunks) if (neighbor_expand and all_chunks) else results
    return pack_context(candidates, budget=budget)
