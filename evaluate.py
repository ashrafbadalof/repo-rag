import numpy as np
import json
import yaml
from sentence_transformers import SentenceTransformer, CrossEncoder

embeddings = np.load("embeddings_ast.npy")
with open("chunks_ast.json", encoding="utf-8") as f:
    chunks = json.load(f)

with open("eval/questions.yaml", encoding="utf-8") as f:
    questions = yaml.safe_load(f)

question_texts = [q['question'] for q in questions]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
q_emb = model.encode(question_texts, normalize_embeddings=True)
similarity = embeddings @ q_emb.T

RUN_RERANK = True
RETRIEVE_N = 50
FINAL_K = 5
K_VALUES = [1, 5, 10, 20, 50, 100]

if RUN_RERANK:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def is_hit(retrieved_paths, entry):
    key = set(entry['primary']) | set(entry.get('acceptable', []))
    return bool(set(retrieved_paths) & key)

for k in K_VALUES:
    hits = 0
    for j, entry in enumerate(questions):
        top_k = np.argsort(similarity[:, j])[::-1][:k]
        if is_hit([chunks[i]['path'] for i in top_k], entry):
            hits += 1
    print(f"{hits}/{len(questions)} at k={k}")

baseline_hits = 0
reranked_hits = 0
reranked_misses = []
baseline_misses = []

for j, entry in enumerate(questions):
    scores = similarity[:, j]
    candidates = np.argsort(scores)[::-1][:RETRIEVE_N]
    retrieved_paths = [chunks[i]['path'] for i in candidates[:FINAL_K]]

    if is_hit(retrieved_paths, entry):
        baseline_hits += 1
    else:
        baseline_misses.append((entry['id'], sorted(set(retrieved_paths))))

    if RUN_RERANK:
        pairs = [[question_texts[j], chunks[i]['text']] for i in candidates]
        reranked_scores = reranker.predict(pairs)
        reranked = candidates[np.argsort(reranked_scores)[::-1][:FINAL_K]]
        reranked_paths = [chunks[i]["path"] for i in reranked]

        if is_hit(reranked_paths, entry):
            reranked_hits += 1
        else:
            reranked_misses.append((entry['id'], sorted(set(reranked_paths))))

print(f"\nbaseline: {baseline_hits}/{len(questions)} at k={FINAL_K}")
if RUN_RERANK:
    print(f"reranked: {reranked_hits}/{len(questions)} at k={FINAL_K} (from {RETRIEVE_N} candidates)")

for qid, paths in reranked_misses:
    print(f'\n{qid}')
    for p in paths:
        print(" ", p)