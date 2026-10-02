import json
import sys

import yaml
from sentence_transformers import CrossEncoder

from retrieval import load_index, search, RERANK_MODEL
from ask import answer

RERANK = "--no-rerank" not in sys.argv
OUT = "eval/answers_rerank.json" if RERANK else "eval/answers_norerank.json"

questions = yaml.safe_load(open("eval/questions.yaml", encoding="utf-8"))
chunks, embeddings, model = load_index()
reranker = CrossEncoder(RERANK_MODEL) if RERANK else None

records = []
for q in questions:
    text, results = answer(q["question"], chunks, embeddings, model,
                           reranker, rerank=RERANK)
    refused = text.startswith("Not found")
    records.append({
        "id": q["id"],
        "question": q["question"],
        "gold": q["primary"] + q.get("acceptable", []),
        "answer": text,
        "sources": [f"{c['path']}:{c['start_line']}-{c['end_line']}" for c in results],
        "refused": refused,
        "correct": None,
    })
    print(f"{'REF' if refused else 'OK '}  {q['id']}")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

answered = sum(1 for r in records if not r["refused"])
print(f"\nrerank={RERANK} — {answered}/{len(records)} answered → {OUT}")