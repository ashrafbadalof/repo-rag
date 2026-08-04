from pathlib import Path
import yaml
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import ast

CLONE_ROOT = Path('django')
PACKAGE_DIR = CLONE_ROOT / "django"
EXCLUDE_DIRS = {'tests'}

files = list(PACKAGE_DIR.glob('**/*.py'))
# print(len(files)) --> 908

MIN_FILE_BYTES = 100  # empty packages returns empty chunks that pollute search
def should_index(path):
    if not EXCLUDE_DIRS.isdisjoint(path.parts):
        return False
    if path.stat().st_size < MIN_FILE_BYTES:
        return False
    return True

files_to_index = [i for i in files if should_index(i)]
# print(len(files_to_index)) --> 750

def rel_path(path):
    return path.relative_to(CLONE_ROOT).as_posix()

with open('eval/questions.yaml') as f:
    questions = yaml.safe_load(f)

indexed = {rel_path(p) for p in files_to_index}
missing = []
for q in questions:
    for path in q.get("primary", []) + q.get("acceptable", []):
        if path not in indexed:
            missing.append((q["id"], path))
# print(f"{len(questions)} questions, {len(missing)} missing paths") --> 20 questions, 0 missing paths

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
def chunk(disk_path):
    text = disk_path.read_text(encoding='utf-8', errors='replace')
    meta_path = rel_path(disk_path)

    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        body = text[start:end]

        start_line = text.count('\n', 0, start) + 1
        end_line = start_line + body.count('\n')

        chunks.append({
            'text': body,
            'path': meta_path,
            'start_line': start_line,
            'end_line': end_line
        })

        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

CHUNKABLE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
def node_start(node):
    """Real start line"""
    if getattr(node, "decorator_list", None):
        return node.decorator_list[0].lineno
    return node.lineno

def chunk_file_ast(disk_path):
    text = disk_path.read_text(encoding='utf-8', errors='replace')
    meta_path = rel_path(disk_path)
    lines = text.splitlines()
    tree = ast.parse(text)

    preamble_lines = []
    chunks = []
    preamble_start = None
    preamble_end = None

    for node in tree.body:
        if isinstance(node, CHUNKABLE):
            start = node_start(node)
            node_lines = lines[start-1 : node.end_lineno]
            body = "\n".join(node_lines)
            chunks.append({
                'text': body,
                'path': meta_path,
                'start_line': start,
                'end_line': node.end_lineno,
            })
        else:
            preamble_lines.extend(lines[node.lineno-1 : node.end_lineno])
            if preamble_start is None:
                preamble_start = node.lineno
            preamble_end = node.end_lineno
    
    if preamble_lines:
        chunks.insert(0, {
            'text': "\n".join(preamble_lines),
            'path': meta_path,
            'start_line': preamble_start,
            'end_line': preamble_end,
        })

    return chunks

CHUNK_STRATEGY = "ast"
chunker = chunk_file_ast if CHUNK_STRATEGY == 'ast' else chunk

all_chunks = []
for f in files_to_index:
    all_chunks.extend(chunker(f))

print(len(all_chunks))

# sanity check 1
# checking for how many chunks each file contributes; large files might over-appear in retrieval results

# ff = {}
# for f in files_to_index:
#     x = len(chunk(f))
#     ff[rel_path(f)] = x
# print(sorted(ff.items(), key=lambda kv: kv[1])[-5:])

# sanity check 2
# x = 0
# for ch in all_chunks:
#     if len(ch['text']) < 100:
#         x+=1
# print(x)        -->  92 chunks are shorter than 100 characters

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
chunks = [f['text'] for f in all_chunks]
embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
np.save("embeddings_ast.npy", embeddings)

with open("chunks_ast.json", "w", encoding='utf-8') as f:
    json.dump(all_chunks, f)

print(embeddings.shape)