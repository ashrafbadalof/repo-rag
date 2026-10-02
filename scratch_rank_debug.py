import yaml, numpy as np
from retrieval import load_index

chunks, embeddings, model = load_index()
questions = yaml.safe_load(open("eval/questions.yaml", encoding="utf-8"))

for q in questions:
    gold = set(q["primary"]) | set(q.get("acceptable", []))
    scores = embeddings @ model.encode(q["question"], normalize_embeddings=True)
    order = np.argsort(scores)[::-1]
    ranks = [(r, chunks[i]["path"], chunks[i].get("symbol"))
             for r, i in enumerate(order[:500], 1)
             if chunks[i]["path"] in gold]
    print(f"\n{q['id']}: {len(ranks)} gold chunks in top 500")
    for r, path, sym in ranks[:5]:
        print(f"  rank {r}: {sym} ({path})")