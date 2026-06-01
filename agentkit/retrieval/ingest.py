"""Ingest CLI: chunk and index .txt/.md files from a directory.

Usage:
    python -m agentkit.retrieval.ingest data/docs
    python -m agentkit.retrieval.ingest data/docs --reset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentkit.retrieval.chunking import Document, chunk_document
from agentkit.retrieval.store import get_retriever

_SUFFIXES = {".txt", ".md"}


def ingest_path(root: str, *, reset: bool = False) -> int:
    base = Path(root)
    files = sorted(p for p in base.rglob("*") if p.suffix.lower() in _SUFFIXES)
    if not files:
        print(f"no .txt/.md files under {base}", file=sys.stderr)
        return 0

    retriever = get_retriever()
    if reset:
        retriever.clear()

    total = 0
    for path in files:
        doc = Document(source=str(path.relative_to(base)), content=path.read_text("utf-8"))
        chunks = chunk_document(doc)
        n = retriever.ingest(chunks)
        total += n
        print(f"  {doc.source}: {n} chunks")
    print(f"ingested {total} chunks from {len(files)} files")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the retrieval store")
    parser.add_argument("path", help="directory containing .txt/.md files")
    parser.add_argument("--reset", action="store_true", help="truncate the store before ingest")
    args = parser.parse_args()
    ingest_path(args.path, reset=args.reset)


if __name__ == "__main__":
    main()
