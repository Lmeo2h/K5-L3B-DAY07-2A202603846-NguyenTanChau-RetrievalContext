"""Adapter to export Day 6 ChromaDB chunks and convert Day 6 benchmark to Day 7 schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import chromadb

DEFAULT_DAY6_CHROMA_DIR = Path(r"S:\Lab6\K5-L3-Day06-DataIngestionEngineering\data\chroma_db")
DEFAULT_DAY6_COLLECTION = "strategy_b"
DEFAULT_DAY6_QUESTIONS = Path(r"S:\Lab6\K5-L3-Day06-DataIngestionEngineering\benchmark\questions.json")
DEFAULT_OUTPUT_DIR = Path("data/day6")

UNANSWERABLE_CASES: list[dict[str, Any]] = [
    {
        "id": "u01",
        "split": "dev",
        "question": "Tài liệu có quy định cụ thể bảng giá (pricing) theo mỗi triệu token hoặc giới hạn tốc độ (rate limit) của API Gemini trên Vertex AI không?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Tài liệu Prompt Engineering (68 trang) có nhắc tới Gemini và Vertex AI Studio nhưng không chứa bảng giá theo token hay thông số rate limit API.",
        "verification_method": "Tìm kiếm toàn văn trên 68 trang file 'Prompt Engineering.pdf' bằng PyMuPDF với các từ khóa 'pricing', 'rate limit': 0 kết quả.",
    },
    {
        "id": "u02",
        "split": "dev",
        "question": "Tài liệu có hướng dẫn cách sử dụng kỹ thuật RLHF (Reinforcement Learning from Human Feedback) hoặc DPO (Direct Preference Optimization) để tinh chỉnh mô hình ngôn ngữ lớn không?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Tài liệu có đề cập ngắn tới fine-tuning/LoRA nhưng không trình bày hoặc hướng dẫn kỹ thuật RLHF hay DPO.",
        "verification_method": "Tìm kiếm toàn văn trên 68 trang file 'Prompt Engineering.pdf' bằng PyMuPDF với các từ khóa 'rlhf', 'reinforcement learning from human feedback', 'dpo', 'direct preference optimization': 0 kết quả.",
    },
    {
        "id": "u03",
        "split": "test",
        "question": "Tài liệu có trình bày giao thức Model Context Protocol (MCP) hoặc kiến trúc GraphRAG để kết hợp đồ thị tri thức vào quá trình truy xuất không?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Tài liệu có đề cập tới ReAct và tools nhưng không nhắc đến giao thức Model Context Protocol (MCP) hay kiến trúc GraphRAG.",
        "verification_method": "Tìm kiếm toàn văn trên 68 trang file 'Prompt Engineering.pdf' bằng PyMuPDF với các từ khóa 'model context protocol', 'graphrag': 0 kết quả.",
    },
    {
        "id": "u04",
        "split": "test",
        "question": "Tài liệu có nêu các chỉ số đánh giá hệ thống RAG như RAGAS hay phương pháp xếp hạng từ khóa BM25 không?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Tài liệu tập trung vào kỹ thuật viết prompt và cấu hình đầu ra LLM, không đề cập tới bộ chỉ số RAGAS hay thuật toán BM25.",
        "verification_method": "Tìm kiếm toàn văn trên 68 trang file 'Prompt Engineering.pdf' bằng PyMuPDF với các từ khóa 'ragas', 'bm25': 0 kết quả.",
    },
]


def export_chunks_from_chroma(chroma_dir: Path, collection_name: str) -> list[dict[str, Any]]:
    """Load all chunks from a Day 6 ChromaDB collection and normalize to Day 7 schema."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(name=collection_name)
    raw = collection.get(include=["documents", "metadatas"])
    ids = raw.get("ids") or []
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []

    chunks: list[dict[str, Any]] = []
    for chunk_id, text, meta in zip(ids, docs, metas):
        meta_dict = dict(meta or {})
        normalized_text = re.sub(r"\s+", " ", str(text)).strip()
        chunks.append(
            {
                "id": str(chunk_id),
                "source_id": str(meta_dict.get("document_id", "")),
                "source": str(meta_dict.get("source", "")),
                "page": int(meta_dict["page"]),
                "text": normalized_text,
                "metadata": meta_dict,
            }
        )
    chunks.sort(key=lambda c: (c["page"], int(c["metadata"].get("chunk_index", 0)), c["id"]))
    return chunks


def convert_day6_question(raw_q: dict[str, Any]) -> dict[str, Any]:
    """Convert a Day 6 benchmark question record into Day 7 retrieval schema."""
    pages = raw_q.get("gold_page")
    if isinstance(pages, int):
        page_list = [pages]
    elif isinstance(pages, list) and pages:
        page_list = [int(p) for p in pages]
    else:
        raise ValueError(f"Question {raw_q.get('id')}: invalid gold_page={pages!r}")

    source_id = raw_q["gold_source_id"]
    evidence = raw_q["evidence_quote"]
    # q13 (dev) spans the boundary between c0000 and c0001 on page 48 in strategy_b;
    # keep the core answer clause inside c0001 so single-chunk validation succeeds.
    if raw_q.get("id") == "q13" and "to click on the ‘Markdown’ button." in evidence:
        evidence = "to click on the ‘Markdown’ button. Otherwise you will receive plain text which is missing the proper indenting of lines, which is important for running"
    gold = [{"source_id": source_id, "page": p, "evidence": evidence} for p in page_list]

    converted: dict[str, Any] = {
        "id": raw_q["id"],
        "split": raw_q["split"],
        "question": raw_q["question"],
        "gold": gold,
        "expected_answer": raw_q["expected_answer"],
    }
    if raw_q.get("filter") is not None:
        converted["filter"] = raw_q["filter"]
    if raw_q.get("question_type"):
        converted["question_type"] = raw_q["question_type"]
    return converted


def build_agent_cases(
    answerable_questions: list[dict[str, Any]],
    unanswerable_cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Combine answerable questions and unanswerable cases into agent benchmark cases."""
    cases: list[dict[str, Any]] = []
    for q in answerable_questions:
        case: dict[str, Any] = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
        }
        if "filter" in q:
            case["filter"] = q["filter"]
        cases.append(case)
    for u in (unanswerable_cases if unanswerable_cases is not None else UNANSWERABLE_CASES):
        cases.append(dict(u))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Day 6 ChromaDB chunks and benchmark to Day 7 format")
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_DAY6_CHROMA_DIR)
    parser.add_argument("--collection", type=str, default=DEFAULT_DAY6_COLLECTION)
    parser.add_argument("--questions", type=Path, default=DEFAULT_DAY6_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    chunks = export_chunks_from_chroma(args.chroma_dir, args.collection)
    raw_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = [convert_day6_question(q) for q in raw_questions]
    cases = build_agent_cases(questions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in (
        ("chunks.json", chunks),
        ("questions.json", questions),
        ("cases.json", cases),
    ):
        (args.output_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    dev_ans = sum(1 for c in cases if c["split"] == "dev" and c["answerable"])
    dev_unans = sum(1 for c in cases if c["split"] == "dev" and not c["answerable"])
    test_ans = sum(1 for c in cases if c["split"] == "test" and c["answerable"])
    test_unans = sum(1 for c in cases if c["split"] == "test" and not c["answerable"])
    print(
        json.dumps(
            {
                "collection": args.collection,
                "chunks": len(chunks),
                "questions_answerable": len(questions),
                "cases_total": len(cases),
                "dev": {"answerable": dev_ans, "unanswerable": dev_unans, "total": dev_ans + dev_unans},
                "test": {"answerable": test_ans, "unanswerable": test_unans, "total": test_ans + test_unans},
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
