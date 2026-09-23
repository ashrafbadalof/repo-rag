# evaluate.py
import yaml
from sentence_transformers import CrossEncoder

from retrieval import load_index, search, RERANK_MODEL

RUN_RERANK = True
FINAL_K = 5
K_VALUES = [1, 5, 10, 20, 50]

with open("eval/questions.yaml", encoding="utf-8") as f:
    questions = yaml.safe_load(f)

chunks, embeddings, model = load_index("ast")
reranker = CrossEncoder(RERANK_MODEL) if RUN_RERANK else None


def is_hit(retrieved_paths, entry):
    key = set(entry['primary']) | set(entry.get('acceptable', []))
    return bool(set(retrieved_paths) & key)


for k in K_VALUES:
    hits = 0
    for entry in questions:
        results = search(entry['question'], chunks, embeddings, model,
                         k=k, rerank=False)
        if is_hit([c['path'] for c in results], entry):
            hits += 1
    print(f"{hits}/{len(questions)} at k={k}")

baseline_hits = 0
reranked_hits = 0
reranked_misses = []

for entry in questions:
    baseline = search(entry['question'], chunks, embeddings, model,
                      k=FINAL_K, rerank=False)
    if is_hit([c['path'] for c in baseline], entry):
        baseline_hits += 1

    if RUN_RERANK:
        reranked = search(entry['question'], chunks, embeddings, model,
                          k=FINAL_K, rerank=True, reranker=reranker)
        reranked_paths = [c['path'] for c in reranked]
        if is_hit(reranked_paths, entry):
            reranked_hits += 1
        else:
            reranked_misses.append((entry['id'], sorted(set(reranked_paths))))

print(f"\nbaseline: {baseline_hits}/{len(questions)} at k={FINAL_K}")
if RUN_RERANK:
    print(f"reranked: {reranked_hits}/{len(questions)} at k={FINAL_K}")

for qid, paths in reranked_misses:
    print(f'\n{qid}')
    for p in paths:
        print(" ", p)