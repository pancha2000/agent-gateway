"""POST /error_log and POST /error_log/{id}/resolve - Reflexion tracking so the AI doesn't repeat mistakes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import repo, vectors
from ..config import get_settings
from ..schemas import ErrorLogIn, ErrorLogOut, ErrorResolveIn
from ..security import require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/error_log", response_model=ErrorLogOut)
async def error_log(body: ErrorLogIn, request: Request) -> ErrorLogOut:
    """
    Log a mistake. The same task+error logged again increments `occurrences`
    instead of creating a duplicate; `is_repeat=True` tells your orchestrator the
    lesson isn't sticking. `attempt` records which retry this was (the highest value
    is kept). If you send `vector`, /get_hint will surface this mistake whenever a
    similar task comes up.
    """
    if body.vector is not None:
        vectors.validate(body.vector, get_settings().embed_dim)

    r = await repo.log_error(
        request.app.state.turso,
        task=body.task, error_message=body.error_message, root_cause=body.root_cause,
        lesson=body.lesson, fix=body.fix, agent=body.agent, tags=body.tags, vector=body.vector,
        attempt=body.attempt, resolved=body.resolved,
    )
    return ErrorLogOut(
        id=r["id"], occurrences=r["occurrences"], is_repeat=r["occurrences"] > 1,
        attempts=r["attempts"], resolved=r["resolved"])


@router.post("/error_log/{error_id}/resolve", response_model=ErrorLogOut)
async def resolve_error(error_id: int, body: ErrorResolveIn, request: Request) -> ErrorLogOut:
    """Mark a logged mistake as fixed and store the working fix / lesson."""
    r = await repo.resolve_error(
        request.app.state.turso, error_id, fix=body.fix, lesson=body.lesson)
    if r is None:
        raise HTTPException(404, "error_log entry not found")
    return ErrorLogOut(
        id=r["id"], occurrences=r["occurrences"], is_repeat=r["occurrences"] > 1,
        attempts=r["attempts"], resolved=r["resolved"])
