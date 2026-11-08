"""POST /escalation_package - turn a stuck problem into a self-contained brief for a stronger model (or a .txt)."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from .. import repo, vectors
from ..config import get_settings
from ..schemas import EscalationIn, EscalationOut
from ..security import require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/escalation_package", response_model=EscalationOut)
async def escalation_package(body: EscalationIn, request: Request) -> EscalationOut:
    """
    After the local model has failed its retries, call this to get ONE markdown document
    containing the task, the error, the code, what was already tried and the most relevant
    notes / past mistakes from the vector database. Send `markdown` to Claude/Gemini, or
    save it as the .txt you hand to the human when every cloud model failed too.
    Set include_memory=false if the database content must not leave your machine.
    """
    s = get_settings()
    hits: list[dict] = []
    errors: list[dict] = []
    if body.include_memory and body.vector is not None:
        vectors.validate(body.vector, s.embed_dim)
        hits, errors = await repo.get_hint(
            request.app.state.turso, s, vector=body.vector, top_k=body.top_k, min_similarity=0.35,
            kind=None, language=None, include_errors=True,
            scope=body.scope, all_scopes=body.all_scopes, project_id=body.project_id)

    now = datetime.now(timezone.utc)
    headline = body.task.strip().splitlines()[0][:100]

    head = [f"# Help request: {headline}", f"Generated (UTC): {now:%Y-%m-%d %H:%M}", "",
            "## Task", body.task.strip(), "",
            "## Error / symptom", "```", body.error_message.strip(), "```"]
    if body.code:
        head += ["", "## Code involved", "```", body.code.strip("\n"), "```"]
    if body.attempts:
        head += ["", "## What was already tried (do not just repeat these)"]
        for a in body.attempts:
            head.append(f"{a.attempt}. {a.approach.strip()}")
            if a.result:
                head.append(f"   Result: {a.result.strip()}")

    ask = ["", "## What I need",
           "Find the root cause and give a complete, working fix (full corrected code, not a diff). "
           "Explain briefly why the earlier attempts failed."]

    base = "\n".join(head + ask)
    budget = max(0, body.max_chars - len(base))

    mem: list[str] = []
    if hits:
        mem += ["", "## Possibly relevant notes from my local knowledge base (reference data, may be imperfect)"]
        for i, h in enumerate(hits, 1):
            src = f" ({h['source']})" if h.get("source") else ""
            mem.append(f"[{i}]{src}\n{h['content'][:1500]}")
    if errors:
        mem += ["", "## Similar mistakes I made before"]
        for e in errors:
            line = f"- Task: {e['task']}\n  Error: {e['error_message']}"
            if e.get("lesson"):
                line += f"\n  Lesson: {e['lesson']}"
            if e.get("fix"):
                line += f"\n  Fix: {e['fix']}"
            mem.append(line)
    mem_text = "\n".join(mem)[:budget] if budget else ""

    markdown = "\n".join(head) + ("\n" + mem_text if mem_text else "") + "\n" + "\n".join(ask) + "\n"
    return EscalationOut(
        markdown=markdown, filename=f"escalation_{now:%Y%m%d-%H%M%S}.txt",
        memory_hits=len(hits) + len(errors))
