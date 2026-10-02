## Preamble chunking fix (Django @ 957d0cee71, 20 questions)

| config | recall@5 | answered | correct | wrong |
|---|---|---|---|---|
| AST chunking, dense        | 9/20  | —     | —    | —   |
| AST chunking, + rerank     | 15/20 | 13/20 | 7/20 | 3/20 |
| preamble fix, dense        | 14/20 | 12/20 | 9/20 | 1/20 |
| preamble fix, + rerank     | 13/20 | 13/20 | 9/20 | 3/20 |

Reranking looked worth +6 recall before the chunker was fixed; after the fix
it costs 1 recall point and adds 2 wrong answers. It was compensating for an
indexing bug. Default is now dense-only.

No held-out set: all configs evaluated on the same 20 questions.