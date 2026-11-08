"""
Text chunking for ingestion. Pure Python, no dependencies.

Strategy: split into paragraph/code blocks (blank-line separated), then pack
blocks greedily into chunks of at most `size` characters. Blocks larger than
`size` are split on line boundaries; single lines larger than `size` are hard
split. Consecutive chunks share a small overlap (the tail of the previous chunk,
aligned to a line or word boundary) so a thought cut at a boundary is not lost.
"""
from __future__ import annotations

import re

_BLANK_LINES = re.compile(r"\n\s*\n")


def _split_long(block: str, size: int) -> list[str]:
    """Split one oversized block on line boundaries (hard-split overlong lines)."""
    parts: list[str] = []
    cur = ""
    for line in block.split("\n"):
        while len(line) > size:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(line[:size])
            line = line[size:]
        if cur and len(cur) + 1 + len(line) > size:
            parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        parts.append(cur)
    return [p for p in parts if p.strip()]


def _tail(chunk: str, n: int) -> str:
    """Last ~n characters of `chunk`, starting at a line (or word) boundary."""
    if n <= 0 or len(chunk) <= n:
        return ""
    t = chunk[-n:]
    i = t.find("\n")
    if 0 <= i < len(t) - 1:
        t = t[i + 1:]
    else:
        j = t.find(" ")
        if 0 <= j < len(t) - 1:
            t = t[j + 1:]
    return t.strip()


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    """Return a list of chunks, each <= `size` characters (overlap included)."""
    size = max(int(size), 100)
    overlap = max(0, min(int(overlap), size // 3))

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    blocks: list[str] = []
    for raw in _BLANK_LINES.split(text):
        block = raw.strip("\n")
        if not block.strip():
            continue
        blocks.extend(_split_long(block, size) if len(block) > size else [block])

    chunks: list[str] = []
    cur = ""
    for block in blocks:
        if not cur:
            cur = block
        elif len(cur) + 2 + len(block) <= size:
            cur += "\n\n" + block
        else:
            chunks.append(cur)
            tail = _tail(cur, overlap)
            if tail and len(tail) + 2 + len(block) <= size:
                cur = tail + "\n\n" + block
            else:
                cur = block
    if cur:
        chunks.append(cur)
    return chunks
