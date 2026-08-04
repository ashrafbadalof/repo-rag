import numpy as np
import json
import yaml
from sentence_transformers import SentenceTransformer

embeddings = np.load("embeddings.npy")
with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

with open("eval/questions.yaml", encoding="utf-8") as f:
    questions = yaml.safe_load(f)

question_texts = [q['question'] for q in questions]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
q_emb = model.encode(question_texts, normalize_embeddings=True)
similarity = embeddings @ q_emb.T

k_vals =[1, 5, 10, 20, 50, 100]
results = {}
for k in k_vals:
    misses = []
    hits = 0

    for j, entry in enumerate(questions):
        key = set(entry['primary']) | set(entry.get('acceptable', []))

        scores = similarity[:, j]

        top_k = np.argsort(scores)[::-1][:k]
        retrieved = {chunks[i]['path'] for i in top_k}

        if retrieved & key:
            hits +=1
        else:
            misses.append((entry['id'], sorted(retrieved)))
    results[k] = (hits, misses)
    print(f'{hits}/{len(questions)} at k={k}')
    
for qid, paths in results[1][1]:
    print(f'\n{qid}')
    for p in paths:
        print(" ", p)    
