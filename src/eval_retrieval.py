"""Evaluate retrieval methods and improvement configs on a specified split (dev/test)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "src"))

from baseline import hit, validate  # noqa: E402
from retrievers import (  # noqa: E402
    DenseChromaRetriever,
    HybridRrfRetriever,
    LexicalTfidfRetriever,
    QueryRewriteRetriever,
    expand_with_neighbors,
    pack_with_budget,
)

# Gemini 2.5 Flash standard pricing (USD per 1M tokens) for cost estimation
GEMINI_FLASH_INPUT_PER_1M = 0.30
GEMINI_FLASH_OUTPUT_PER_1M = 2.50


def evaluate_config(
    name: str,
    retriever: Any,
    chunks: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    top_k: int = 3,
    budget: int = 800,
    neighbor_expand: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    for q in questions:
        filters = q.get("filter")
        start = time.perf_counter()
        results = retriever.search(q["question"], filters=filters, top_k=top_k)
        # Always measure Hit@1 and Hit@3 on the top-3 retrieved items
        top3 = results[:3]
        context, ids = pack_with_budget(
            results,
            budget=budget,
            all_chunks=chunks,
            neighbor_expand=neighbor_expand,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        marked_top3 = [{**c, "hit": hit(c, q["gold"])} for c in top3]
        packed_source = expand_with_neighbors(results, chunks) if neighbor_expand else results
        packed_by_id = {c["id"]: c for c in packed_source}
        context_hit = any(hit(packed_by_id[cid], q["gold"]) for cid in ids if cid in packed_by_id)

        rewritten = results[0].get("rewritten_query") if results else None
        traces.append(
            {
                "query_id": q["id"],
                "split": q["split"],
                "question": q["question"],
                "rewritten_query": rewritten,
                "filter": filters,
                "config": name,
                "top_k": top_k,
                "budget_chars": budget,
                "neighbor_expand": neighbor_expand,
                "gold": q["gold"],
                "results_top3": marked_top3,
                "context": context,
                "context_ids": ids,
                "context_chars": len(context),
                "elapsed_ms": elapsed_ms,
                "hit1": bool(marked_top3 and marked_top3[0]["hit"]),
                "hit3": any(c["hit"] for c in marked_top3),
                "context_hit": context_hit,
            }
        )

    in_tok = getattr(retriever, "total_input_tokens", 0)
    out_tok = getattr(retriever, "total_output_tokens", 0)
    cost_usd = (in_tok / 1_000_000) * GEMINI_FLASH_INPUT_PER_1M + (out_tok / 1_000_000) * GEMINI_FLASH_OUTPUT_PER_1M
    n = len(traces)
    summary = {
        "config": name,
        "split": questions[0]["split"],
        "n": n,
        "top_k": top_k,
        "budget_chars": budget,
        "neighbor_expand": neighbor_expand,
        "hit1": sum(t["hit1"] for t in traces),
        "hit1_rate": round(sum(t["hit1"] for t in traces) / n, 4),
        "hit3": sum(t["hit3"] for t in traces),
        "hit3_rate": round(sum(t["hit3"] for t in traces) / n, 4),
        "context_hit": sum(t["context_hit"] for t in traces),
        "context_hit_rate": round(sum(t["context_hit"] for t in traces) / n, 4),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2),
        "total_seconds": round(sum(t["elapsed_ms"] for t in traces) / 1000.0, 3),
        "llm_input_tokens": in_tok,
        "llm_output_tokens": out_tok,
        "service_cost_usd": round(cost_usd, 6),
    }
    return summary, traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval configs on dev/test split")
    parser.add_argument("--chunks", type=Path, default=Path("data/day6/chunks.json"))
    parser.add_argument("--questions", type=Path, default=Path("data/day6/questions.json"))
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--output-dir", type=Path, default=Path("results/retrieval"))
    parser.add_argument("--skip-rewrite", action="store_true")
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = [q for q in all_questions if q.get("split") == args.split]
    validate(chunks, selected)

    lexical = LexicalTfidfRetriever(chunks)
    dense = DenseChromaRetriever(infer_filter=True)
    hybrid = HybridRrfRetriever(dense, lexical)

    configs: list[tuple[str, Any, dict[str, Any]]] = [
        ("1_tfidf_baseline", lexical, {"top_k": 3, "budget": 800, "neighbor_expand": False}),
        ("2_dense_bge_m3", dense, {"top_k": 3, "budget": 800, "neighbor_expand": False}),
        ("3_hybrid_rrf", hybrid, {"top_k": 3, "budget": 800, "neighbor_expand": False}),
        ("4_hybrid_budget_neighbor", hybrid, {"top_k": 3, "budget": 2400, "neighbor_expand": True}),
    ]
    if not args.skip_rewrite:
        rewrite_hybrid = QueryRewriteRetriever(hybrid, model="gemini-3.5-flash-lite")
        configs.append(("5_hybrid_rewrite_budget", rewrite_hybrid, {"top_k": 3, "budget": 2400, "neighbor_expand": True}))

    summaries: list[dict[str, Any]] = []
    out_split_dir = args.output_dir / args.split
    out_split_dir.mkdir(parents=True, exist_ok=True)
    for name, ret, kw in configs:
        summary, traces = evaluate_config(name, ret, chunks, selected, **kw)
        summaries.append(summary)
        cfg_dir = out_split_dir / name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (cfg_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (out_split_dir / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
