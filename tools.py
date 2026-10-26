from langchain_core.tools import tool

from retrieval import load_chunks, search

CHUNKS = load_chunks()


@tool
def search_notes(query: str) -> str:
    """Search the candidate's project notes for evidence of a skill or experience.
    Use short keyword queries such as 'Docker' or 'REST API'.
    Returns matching note excerpts with citations, or says no evidence was found."""
    results = search(query, k=3, chunks=CHUNKS)
    if not results:
        return "No evidence found."
    return "\n\n".join(f"{chunk.citation}\n{chunk.text}" for _, chunk in results)