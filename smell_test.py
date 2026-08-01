import numpy as np
import json
import yaml
from sentence_transformers import SentenceTransformer

embeddings = np.load("embeddings.npy")
with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

with open("eval/questions.yaml", encoding="utf-8") as f:
    questions = yaml.safe_load(f)

q = questions[3]['question']

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
q_emb = model.encode(q, normalize_embeddings=True)
similarity = embeddings @ q_emb

first_five = np.argsort(similarity)[::-1][:5]
for i in first_five:
    ch = chunks[i]
    print(ch['path'], round(float(similarity[i]), 3))

order = np.argsort(similarity)[::-1]
ranks = [r for r, i in enumerate(order) if chunks[i]["path"] == "django/utils/text.py"]
print(ranks[:5])