"""Inspect all 7 test traces and retrieval contexts for manual audit."""

import json
from pathlib import Path

qs = {q["id"]: q for q in json.loads(Path("data/day6/questions.json").read_text(encoding="utf-8")) if q["split"] == "test"}

for cfg in ("baseline_tfidf", "improved_hybrid_neighbor"):
    print(f"\n==================== {cfg} (TEST) ====================")
    ts = json.loads(Path(f"results/agent/test/{cfg}/traces.json").read_text(encoding="utf-8"))
    for t in ts:
        qid = t["query_id"]
        exp = qs[qid]["expected_answer"] if qid in qs else "(UNANSWERABLE)"
        gold = qs[qid]["gold"] if qid in qs else []
        print(f"\n[{qid}] answerable={t['answerable']} status={t['status']} calls={t['search_calls']} cites={t['citations']}")
        print(f"  Q   : {t['question']}")
        print(f"  EXP : {exp}")
        if gold:
            print(f"  GOLD: page={gold[0]['page']} | {gold[0]['evidence']}")
        for ev in t["events"]:
            if ev.get("event") == "search":
                print(f"  SEARCH query={ev['query']!r} filter={ev['filter']} ctx_ids={ev['context_ids']}")
                print(f"  CTX_PREVIEW: {ev['context'][:500]}...")
        print(f"  ANS : {t['answer']}")
