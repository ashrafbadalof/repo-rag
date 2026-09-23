import argparse
import os

from anthropic import Anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

from retrieval import load_index, search, RERANK_MODEL

load_dotenv()
client = Anthropic()

MODEL = 'claude-haiku-4-5'
MAX_TOKENS = 1000
MAX_CHUNK_CHARS = 2000

PROMPT = """You are answering questions about the Django codebase.

Use ONLY the code below. Do not rely on anything else you know about Django.
After each claim, cite the source as (path:start-end) using the exact line
numbers from the headers below. 
If the code below does not contain the answer, say "Not found in the retrieved
code." and do not guess.
Do not describe code that is not shown in full; an import or a reference is not an implementation.

<code>
{context}
</code>

Question: {question}"""

def build_context(chunks):
    blocks = []
    for c in chunks:
        text = c["text"]
        start = c["start_line"]
        end = c["end_line"]

        if len(text) > MAX_CHUNK_CHARS:
            lines = text[:MAX_CHUNK_CHARS].split("\n")[:-1]
            text = "\n".join(lines)
            end = start + len(lines) - 1
            text += "\n     # ... truncated"

        blocks.append(f"--- {c['path']}:{start}-{end} ---\n{text}")

    return "\n\n".join(blocks)

def answer(question, chunks, embeddings, model, reranker, k=5, rerank=True, results=None):
    if results is None:
        results = search(question, chunks, embeddings, model, k=5, rerank=rerank, reranker=reranker)

    if not results:
        return "No code retrieved for this question", []

    context = build_context(results)

    response = client.messages.create(
        model = MODEL,
        max_tokens = MAX_TOKENS,
        temperature=0,
        messages=[{
            "role":"user",
            "content": PROMPT.format(context = context, question=question)
        }]
    )

    return response.content[0].text, results

def print_sources(chunks):
    print("\nSources:")
    for c in chunks:
        print(f"  {c['path']}:{c['start_line']}-{c['end_line']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--verbose", action="store_true",
                        help="Print retrieved chunks before answering.")
    args = parser.parse_args()

    rerank = not args.no_rerank

    chunks, embeddings, model = load_index("ast")
    reranker = CrossEncoder(RERANK_MODEL) if rerank else None

    results = search(args.question, chunks, embeddings, model,
                     k=args.k, rerank=rerank, reranker=reranker)

    if args.verbose:
        print(f"Retrieved {len(results)} chunks:")
        for c in results:
            preview = c["text"].split("\n")[0][:80]
            print(f"  {c['path']}:{c['start_line']}-{c['end_line']}  {preview}")
        print()

    text, results = answer(args.question, chunks, embeddings, model,
                           reranker, k=args.k, rerank=rerank,
                           results=results)

    print(text)
    print_sources(results)


if __name__ == "__main__":
    main()