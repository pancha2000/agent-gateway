"""
PDF -> text, done with `pypdf` (pure Python, small). It is imported lazily so the
gateway still starts if pypdf is missing; PDFs are then simply skipped.

RAM guard: the caller already refuses PDFs above PDF_MAX_BYTES; here we also cap
the number of pages and the amount of text we collect.
"""
from __future__ import annotations

import io
import logging

log = logging.getLogger("gateway.pdf")


def pdf_to_text(data: bytes, *, max_pages: int, max_chars: int) -> tuple[str, str | None] | None:
    """
    Returns (text, title) or None if the PDF cannot be read (encrypted, scanned
    images only, corrupt, pypdf missing). `title` comes from the PDF metadata.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf is not installed; PDF skipped (pip install pypdf)")
        return None

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    return None
            except Exception:                      # noqa: BLE001 - pypdf raises many types
                return None

        parts: list[str] = []
        total = 0
        for i, page in enumerate(reader.pages):
            if i >= max_pages or total >= max_chars:
                break
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(text)
                total += len(text)

        body = "\n\n".join(parts).strip()
        if not body:
            return None                            # scanned PDF without a text layer

        title = None
        try:
            meta = reader.metadata
            if meta and meta.title:
                title = " ".join(str(meta.title).split())[:300] or None
        except Exception:                          # noqa: BLE001
            title = None
        return body[:max_chars], title
    except Exception as exc:                       # noqa: BLE001 - corrupt PDFs raise anything
        log.info("PDF parse failed: %s", exc)
        return None
