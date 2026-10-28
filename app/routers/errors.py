"""POST /error_log - Reflexion tracking so the AI doesn't repeat mistakes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from .. import repo, vectors
from ..config import get_settings
from ..schemas import ErrorLogIn, ErrorLogOut
from ..security import require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/error_log", response_model=ErrorLogOut)
async def error_log(body: ErrorLogIn, request: Request) -> ErrorLogOut:
    """
    Log a mistake. The same task+error logged again increments `occurrences`
    instead of creating a duplicate; `is_repeat=True` tells your orchestrator the
    lesson isn't sticking. If you send `vector`, /get_hint will surface this
    mistake whenever a similar task comes up.
    """
    if body.vector is not None:
        vectors.validate(body.vector, get_settings().embed_dim)

    error_id, occurrences = await repo.log_error(
        request.app.state.turso,
        task=body.task, error_message=body.error_message, root_cause=body.root_cause,
        lesson=body.lesson, fix=body.fix, agent=body.agent, tags=body.tags, vector=body.vector,
    )
    return ErrorLogOut(id=error_id, occurrences=occurrences, is_repeat=occurrences > 1)
