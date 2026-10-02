from pathlib import Path
import yaml
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import ast

CLONE_ROOT = Path('django')
PACKAGE_DIR = CLONE_ROOT / "django"
EXCLUDE_DIRS = {'tests'}
DJANGO_COMMIT = "957d0cee71"      # main-branch snapshot, not a release tag

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
if missing:
    raise SystemExit(f"Gold paths not in index: {missing}")

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
            'end_line': end_line,
            'symbol': None,
        })

        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

IMPORT_NODES = (ast.Import, ast.ImportFrom)
MIN_PREAMBLE_CHARS = 120
MAX_GAP = 2
CHUNKABLE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
METHOD_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
CLASS_SPLIT_LINES = 60        # classes longer than this are split per method

def node_start(node):
    """Real start line"""
    if getattr(node, "decorator_list", None):
        return node.decorator_list[0].lineno
    return node.lineno


def make_chunk(lines, meta_path, start, end, symbol):
    return {
        "text": "\n".join(lines[start - 1:end]),
        "path": meta_path,
        "start_line": start,
        "end_line": end,
        "symbol": symbol,
    }


def is_docstring(node):
    return (isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


def class_skeleton(cls, lines, meta_path):
    """Class line + docstring + method signatures, so the class is still
    findable as a concept after its methods are split out."""
    head_start = node_start(cls)
    head_end = cls.body[0].end_lineno if (cls.body and is_docstring(cls.body[0])) \
        else head_start

    text = "\n".join(lines[head_start - 1:head_end])
    sigs = [lines[node_start(m) - 1].strip()
            for m in cls.body if isinstance(m, METHOD_NODES)]
    if sigs:
        text += "\n    " + "\n    ".join(sigs)

    return {"text": text, "path": meta_path,
            "start_line": head_start, "end_line": head_end,
            "symbol": f"{cls.name} (skeleton)",
            "synthetic": True}


def chunk_file_ast(disk_path, split_methods=False):
    text = disk_path.read_text(encoding="utf-8", errors="replace")
    meta_path = rel_path(disk_path)
    lines = text.splitlines()
    tree = ast.parse(text)

    chunks = []
    run = []

    def flush():
        if not run:
            return
        if all(isinstance(n, IMPORT_NODES) for n in run):
            return                                   # import-only: drop
        start, end = run[0].lineno, run[-1].end_lineno
        body = "\n".join(lines[start - 1:end])
        if len(body) < MIN_PREAMBLE_CHARS:
            return                                   # too small to be useful
        chunks.append({"text": body, "path": meta_path,
                       "start_line": start, "end_line": end,
                       "symbol": None})

    for node in tree.body:
        if isinstance(node, CHUNKABLE):
            flush()
            run.clear()
            start = node_start(node)
            span = node.end_lineno - start + 1

            if (split_methods and isinstance(node, ast.ClassDef)
                    and span > CLASS_SPLIT_LINES):
                chunks.append(class_skeleton(node, lines, meta_path))
                for m in node.body:
                    if isinstance(m, METHOD_NODES):
                        chunks.append(make_chunk(
                            lines, meta_path, node_start(m), m.end_lineno,
                            f"{node.name}.{m.name}"))
            else:
                chunks.append(make_chunk(lines, meta_path, start,
                                         node.end_lineno, node.name))
        else:
            if run and node.lineno - run[-1].end_lineno - 1 > MAX_GAP:
                flush()
                run.clear()
            run.append(node)

    flush()
    return chunks


CHUNK_STRATEGY = "method"       # "method" | "ast" | "fixed"

CHUNKERS = {
    "method": lambda p: chunk_file_ast(p, split_methods=True),
    "ast": lambda p: chunk_file_ast(p, split_methods=False),
    "fixed": chunk,
}
chunker = CHUNKERS[CHUNK_STRATEGY]

all_chunks = []
for f in files_to_index:
    all_chunks.extend(chunker(f))

print(f"strategy={CHUNK_STRATEGY}  chunks={len(all_chunks)}")

suffix = "" if CHUNK_STRATEGY == "fixed" else f"_{CHUNK_STRATEGY}"

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
texts = [c['text'] for c in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
np.save(f"embeddings{suffix}.npy", embeddings)

with open(f"chunks{suffix}.json", "w", encoding='utf-8') as f:
    json.dump(all_chunks, f)

print(embeddings.shape)