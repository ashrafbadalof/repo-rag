# repo-rag

Ask questions about a large unfamiliar Python codebase, get answers with
file-and-line citations. Built against the Django source (9,241 chunks), 
evaluated on 20 hand-labelled questions.

The point of this repo isn't the score — it's that every design decision was
measured, and three of them turned out to be wrong.

## Example
$ python ask.py "Where does Django compress a response before sending it?"

Django compresses a response in GZipMiddleware.process_response
(django/middleware/gzip.py:9-57). For streaming responses it uses
compress_sequence() ... (django/middleware/gzip.py:24-32)

Sources:
django/middleware/gzip.py:9-66
django/http/init.py:1-52


If the retrieved code doesn't contain the answer, it says so instead of guessing.

## Pipeline

AST chunking (one chunk per function; classes over 60 lines split per method)
→ MiniLM embeddings → top-50 dense retrieval → optional cross-encoder rerank
→ top-5 chunks → Claude Haiku with a citation-and-refusal prompt.

## Results (20 questions, Django)

| config | recall@5 | answered | correct | wrong |
|---|---|---|---|---|
| AST chunks, dense | 9/20 | — | — | — |
| AST chunks + rerank | 15/20 | 13 | 7 | 3 |
| + preamble fix, dense | 14/20 | 12 | 9 | 1 |
| + method-level chunks, dense | **14/20** | **13** | **9** | **1** |
| + 2 chunks/file, outlines demoted | 12/20 | 14 | 8 | 2 |

"correct" is hand-graded: does the answer name the right code *and* does the
citation point at it. Graded answers are in `eval/answers_*.json`.

## Three things that were wrong

**The reranker was compensating for a bug.** `ms-marco-MiniLM-L-6-v2` appeared
to lift recall@5 from 9/20 to 15/20. After fixing the chunker it cost a recall
point, and end-to-end it turned two refusals into confidently wrong answers. It
had been rescuing a ranking full of junk chunks. It's trained on web prose, not
code. Dense-only is the default; `--rerank` opts in.

**The chunker's line numbers lied.** Every top-level non-function line in a file
was collected into one chunk labelled from the first to the last of them — so a
chunk holding ~30 scattered lines claimed to be `validators.py:1-376`. Short and
keyword-dense, so it retrieved well and was useless to the generator. Splitting
into contiguous runs and dropping import-only runs: recall@5 9→14, correct 7→9.

**A fix became a bottleneck at a different granularity.** One chunk per file was
right when chunks were whole classes. Once classes were split per method, a
file's docstring could block the method that answered the question. But relaxing
it to two per file and demoting class outlines made things worse — the outlines
were sometimes the only hit.

## Honest metrics

File-level recall was blind to two real improvements, because a chunk containing
only imports *from the right file* counted as a hit. At one point retrieval
scored 15/20 while only 7/20 answers were correct.

## Limitations

- **No held-out set** — all configs evaluated on the same 20 questions, so the
  numbers are optimistic.
- Gold labels are file-level, so a question with two defensible answers
  (`cache-expiry`) scores wrong for picking the other one.
- Single-hop only; questions spanning two files are out of scope.

## What the evidence says to do next

`scratch/rank_debug.py` shows where the needed chunk actually ranks:
`EmailBackend._send` at 6, `LazySettings._setup` at 7,
`EmailValidator.__call__` at 11, `LocMemCache._has_expired` at 20,
`slugify` at 230, `URLResolver.resolve` not in the top 500.

So the remaining failures aren't a chunking problem:

1. **Hybrid BM25 + dense (RRF)** — `slugify` ranks 230 for a question containing
   "slug"; exact identifier matching is what embeddings are worst at.
2. **A code-trained reranker** — promoting a rank-6-to-20 chunk out of a
   50-candidate pool is exactly a reranker's job; the MS MARCO one can't.

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/django/django && git -C django checkout 957d0cee71
python index.py                 # chunks + embeddings, ~2 min on CPU
python evaluate.py              # retrieval metrics
python ask.py "your question"   # --rerank to enable the cross-encoder
```
