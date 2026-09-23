import json
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def load_index(strategy="ast"):
    suffix = "_ast" if strategy == "ast" else ""
    embeddings = np.load(f"embeddings{suffix}.npy")
    with open(f"chunks{suffix}.json", encoding="utf-8") as f:
        chunks = json.load(f)
    model = SentenceTransformer(EMBED_MODEL)
    return chunks, embeddings, model

def search(query, chunks, embeddings, model, k=5, rerank=True,
           reranker=None, retrieve_n=50):
    q_emb = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ q_emb
    candidates = np.argsort(scores)[::-1][:retrieve_n]

    if rerank:
        pairs = [[query, chunks[i]["text"]] for i in candidates]
        rr = reranker.predict(pairs)
        candidates = candidates[np.argsort(rr)[::-1]]

    out, seen = [], set()
    for i in candidates:
        path = chunks[i]["path"]
        if path in seen:
            continue
        seen.add(path)
        out.append(chunks[i])
        if len(out) == k:
            break
    return out