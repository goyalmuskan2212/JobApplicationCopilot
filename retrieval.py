import re
from dataclasses import dataclass
from pathlib import Path

NOTES_DIR = Path(__file__).parent / "profile"

STOPWORDS = {
    "a", "an", "and", "the", "of", "to", "in", "on", "for", "with", "is",
    "are", "was", "be", "as", "at", "by", "or", "it", "this", "that", "i",
}


@dataclass
class Chunk:
    source: str  # which file the text came from
    index: int   # which block of that file
    text: str

    @property
    def citation(self) -> str:
        return f"[{self.source}#{self.index}]"


def load_chunks(notes_dir: Path = NOTES_DIR) -> list[Chunk]:
    """Read every note and split it into chunks at blank lines."""
    chunks: list[Chunk] = []
    for path in sorted(notes_dir.glob("*.md")):
        if path.name.startswith("_"):  # skip the template
            continue
        text = path.read_text(encoding="utf-8")
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        for i, block in enumerate(blocks):
            chunks.append(Chunk(source=path.name, index=i, text=block))
    return chunks


def tokenize(text: str) -> set[str]:
    """Lowercase words, keeping things like node.js, c++ and c#."""
    words = re.findall(r"[a-z0-9.+#]+", text.lower())
    cleaned = {w.strip(".") for w in words}
    return {w for w in cleaned if len(w) > 1 and w not in STOPWORDS}


def search(query: str, k: int = 3, chunks: list[Chunk] | None = None) -> list[tuple[float, Chunk]]:
    """Return the top k chunks, scored by the fraction of query words they contain."""
    if chunks is None:
        chunks = load_chunks()
    query_words = tokenize(query)
    if not query_words:
        return []
    scored = []
    for chunk in chunks:
        overlap = query_words & tokenize(chunk.text)
        if overlap:
            scored.append((len(overlap) / len(query_words), chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:k]