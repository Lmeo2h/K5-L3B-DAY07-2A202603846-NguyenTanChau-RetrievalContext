"""Inspect dev traces in full for manual audit (never touches test)."""

import json
from pathlib import Path

qs = {q["id"]: q for q in json.loads(Path("data/day6/questions.json").read_text(encoding="utf-8")) if q["split"] == "dev"}

for cfg in ("baseline_tfidf", "improved_hybrid_neighbor", "improved_dense_neighbor"):
    ts = json.loads(Path(f"results/agent/dev/{cfg}/traces.json").read_text(encoding="utf-8"))
    print(f"\n================ {cfg} ================")
    for t in ts:
        qid = t["query_id"]
        if cfg == "improved_dense_neighbor" and qid != "q03":
            continue
        if cfg == "baseline_tfidf" and t["status"] != "answered":
            continue
        exp = qs[qid]["expected_answer"] if qid in qs else "(UNANSWERABLE)"
        print(f"[{qid}] status={t['status']} cites={t['citations']}")
        print(f"  EXP: {exp}")
        print(f"  ANS: {t['answer']}")
