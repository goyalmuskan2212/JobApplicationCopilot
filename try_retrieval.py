import sys

from retrieval import load_chunks, search

chunks = load_chunks()
sources = {c.source for c in chunks}
print(f"Loaded {len(chunks)} chunks from {len(sources)} notes: {sorted(sources)}\n")

# Queries come from the command line, or fall back to these examples.
queries = sys.argv[1:] or ["Python", "REST API", "machine learning"]

for query in queries:
    print(f"QUERY: {query}")
    results = search(query, chunks=chunks)
    if not results:
        print("  (no evidence found)")
    for score, chunk in results:
        preview = chunk.text.replace("\n", " ")[:90]
        print(f"  {score:.2f} {chunk.citation} {preview}")
    print()