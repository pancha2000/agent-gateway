"""Builds the prompt-ready `context` strings returned by the API."""
from __future__ import annotations

from typing import Any

from .websearch import UNTRUSTED_NOTE


def build_hint_context(hits: list[dict[str, Any]], errors: list[dict[str, Any]], budget: int) -> str:
    """Format memory hits + past mistakes into one prompt block, never exceeding `budget` chars."""
    if not hits and not errors:
        return ""
    parts: list[str] = [UNTRUSTED_NOTE]
    if hits:
        parts.append("### Relevant code & notes from memory")
        for i, h in enumerate(hits, 1):
            tag = ", ".join(x for x in (h.get("kind"), h.get("language"), h.get("source")) if x)
            head = f"[{i}] #{h['id']} ({tag}; similarity {h['similarity']:.2f})"
            if h.get("title"):
                head += f" {h['title']}"
            parts.append(f"{head}\n{h['content']}")
    if errors:
        parts.append("### Past mistakes to avoid")
        for e in errors:
            line = f"- Task: {e['task']}\n  Error: {e['error_message']}"
            if e.get("lesson"):
                line += f"\n  Lesson: {e['lesson']}"
            if e.get("fix"):
                line += f"\n  Fix: {e['fix']}"
            line += f"\n  Status: {'resolved' if e.get('resolved') else 'UNRESOLVED'} (attempts: {e.get('attempts', 1)})"
            if e["occurrences"] > 1:
                line += f"\n  (repeated {e['occurrences']} times!)"
            parts.append(line)
    return "\n\n".join(parts)[:budget]


def build_pages_context(pages: list[dict[str, Any]], others: list[dict[str, Any]], budget: int) -> str:
    """Context from freshly fetched pages (first chunks of each) + remaining search results."""
    if not pages and not others:
        return ""
    parts: list[str] = [UNTRUSTED_NOTE]
    for i, pg in enumerate(pages, 1):
        body = "\n\n".join(c["text"] for c in pg["chunks"][:3])
        parts.append(f"[{i}] {pg['title']}\n{pg['url']}\n--- page text ---\n{body}")
    for j, r in enumerate(others, len(pages) + 1):
        parts.append(f"[{j}] {r['title']}\n{r['url']}\n{r['snippet']}")
    return "\n\n".join(parts)[:budget]
