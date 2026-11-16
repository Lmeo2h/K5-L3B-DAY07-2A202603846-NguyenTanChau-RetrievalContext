"""Unit tests for the Day 6 to Day 7 adapter."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "starter"))

from baseline import validate  # noqa: E402
from day6_adapter import UNANSWERABLE_CASES, build_agent_cases, convert_day6_question  # noqa: E402


class Day6AdapterTests(unittest.TestCase):
    def test_convert_day6_question_list_page_and_filter(self):
        raw_q = {
            "id": "q04",
            "split": "dev",
            "question_type": "dieu_kien_ngoai_le",
            "question": "Khi temperature = 0 thì sao?",
            "expected_answer": "Top-K và top-P không còn tác dụng.",
            "gold_source_id": "google-prompt-engineering-2025",
            "gold_page": [12],
            "evidence_quote": "If you set temperature to 0, top-K and top-P become irrelevant",
            "filter": {"section": "LLM Output Configuration"},
        }
        converted = convert_day6_question(raw_q)
        self.assertEqual(converted["gold"], [{"source_id": "google-prompt-engineering-2025", "page": 12, "evidence": raw_q["evidence_quote"]}])
        self.assertEqual(converted["filter"], {"section": "LLM Output Configuration"})

        chunk = {
            "id": "c1",
            "source_id": "google-prompt-engineering-2025",
            "source": "data/raw/Prompt Engineering.pdf",
            "page": 12,
            "text": "If you set temperature to 0, top-K and top-P become irrelevant in practice.",
            "metadata": {"section": "LLM Output Configuration"},
        }
        validate([chunk], [converted])

    def test_build_agent_cases_counts(self):
        ans_q = [{"id": "q01", "split": "dev", "question": "Q1"}]
        cases = build_agent_cases(ans_q)
        self.assertEqual(len(cases), 1 + len(UNANSWERABLE_CASES))
        self.assertTrue(cases[0]["answerable"])
        self.assertEqual(sum(1 for c in cases if not c["answerable"] and c["split"] == "dev"), 2)
        self.assertEqual(sum(1 for c in cases if not c["answerable"] and c["split"] == "test"), 2)


if __name__ == "__main__":
    unittest.main()
