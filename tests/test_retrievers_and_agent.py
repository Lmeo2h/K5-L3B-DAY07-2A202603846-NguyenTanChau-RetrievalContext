"""Offline unit tests for Day 7 retrievers, RRF fusion, filters, and neighbor expansion."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_runner import AgentRetrieverAdapter  # noqa: E402
from retrievers import (  # noqa: E402
    HybridRrfRetriever,
    LexicalTfidfRetriever,
    expand_with_neighbors,
    infer_section_filter,
    to_chroma_where,
)


def _chunk(cid: str, page: int, idx: int, text: str, section: str = "Introduction") -> dict:
    return {
        "id": cid,
        "source_id": "doc",
        "source": "doc.pdf",
        "page": page,
        "text": text,
        "metadata": {"section": section, "chunk_index": idx, "page": page, "document_id": "doc"},
    }


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            _chunk("c1", 1, 0, "Temperature zero selects the highest probability token.", "LLM Output Configuration"),
            _chunk("c2", 1, 1, "Top-K and top-P become irrelevant when temperature is zero.", "LLM Output Configuration"),
            _chunk("c3", 2, 0, "Code prompting helps translate Bash scripts to Python.", "Code Prompting"),
        ]

    def test_filters_and_chroma_where(self):
        self.assertIsNone(to_chroma_where(None))
        self.assertEqual(to_chroma_where({"section": "Code Prompting"}), {"section": "Code Prompting"})
        self.assertEqual(
            to_chroma_where({"section": "Code Prompting", "year": 2025}),
            {"$and": [{"section": "Code Prompting"}, {"year": 2025}]},
        )
        self.assertEqual(infer_section_filter("Giải thích về Code Prompting"), {"section": "Code Prompting"})
        self.assertIsNone(infer_section_filter("Câu hỏi chung không nêu chương"))

        lex = LexicalTfidfRetriever(self.chunks)
        res = lex.search("prompting", filters={"section": "Code Prompting"}, top_k=3)
        self.assertEqual([r["id"] for r in res], ["c3"])

    def test_hybrid_rrf_combines_ranks(self):
        class FakeDense:
            def search(self, query, *, filters=None, top_k=3):
                return [{**self.chunks[1], "score": 0.9}, {**self.chunks[0], "score": 0.8}]

        FakeDense.chunks = self.chunks
        lex = LexicalTfidfRetriever(self.chunks)
        hybrid = HybridRrfRetriever(FakeDense(), lex, rrf_k=60)
        res = hybrid.search("temperature zero", filters=None, top_k=2)
        self.assertEqual(len(res), 2)
        self.assertTrue(all(r["strategy"] == "hybrid_rrf" for r in res))
        self.assertEqual(len({r["id"] for r in res}), 2)

    def test_neighbor_expansion_and_adapter_top_k(self):
        expanded = expand_with_neighbors([self.chunks[0]], self.chunks, max_total=3)
        self.assertEqual([c["id"] for c in expanded], ["c1", "c2"])

        lex = LexicalTfidfRetriever(self.chunks)
        adapter = AgentRetrieverAdapter(lex, self.chunks, neighbor_expand=True)
        res = adapter.search("temperature", filters=None, top_k=3)
        self.assertLessEqual(len(res), 3)
        self.assertEqual(len({r["id"] for r in res}), len(res))


if __name__ == "__main__":
    unittest.main()
