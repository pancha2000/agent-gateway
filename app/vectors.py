"""
Vector helpers. Deliberately NO numpy: with only 1 GB RAM we avoid it (~30 MB
import cost). The gateway never does math on vectors itself, because Turso
computes cosine distance with its native `vector_distance_cos()` function.
We only validate and serialise them.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

from fastapi import HTTPException


def validate(vec: Sequence[float], dim: int, field: str = "vector") -> None:
    """422 if the vector has the wrong size or contains NaN/inf."""
    if len(vec) != dim:
        raise HTTPException(422, f"{field} has {len(vec)} dimensions, expected {dim} (EMBED_DIM)")
    if not all(math.isfinite(x) for x in vec):
        raise HTTPException(422, f"{field} contains NaN or infinity")


def to_text(vec: Sequence[float]) -> str:
    """
    '[0.12345678,-0.5,...]' - the text form accepted by Turso's vector32().
    Fixed-point formatting avoids scientific notation (1e-05) which some
    vector parsers reject; 8 decimals is far finer than float32 needs.
    """
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
