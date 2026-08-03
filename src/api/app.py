"""FastAPI application for the research workbench."""
from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api.manager import RunManager
from src.api.models import RunCreate, RunDetail, RunList, RunSummary
from src.paths import PROCESSED_ROOT, PROJECT_ROOT


def _max_concurrent_runs() -> int:
    raw = os.getenv("API_MAX_CONCURRENT_RUNS", "2").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("API_MAX_CONCURRENT_RUNS must be an integer") from exc
    if value <= 0:
        raise ValueError("API_MAX_CONCURRENT_RUNS must be greater than zero")
    return value


def _sse(event: str, data: dict, *, event_id: int | None = None) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    lines.append(f"data: {payload}")
    return "\n".join(lines) + "\n\n"


def _resolve_visual(run: RunDetail, evidence_id: str, block_id: str) -> Path:
    if run.result is None:
        raise FileNotFoundError("Run has no persisted result")
    evidence = next(
        (item for item in run.result.evidence if item.evidence_id == evidence_id), None
    )
    if evidence is None:
        raise FileNotFoundError(f"Evidence not found: {evidence_id}")
    visual = next(
        (item for item in evidence.visual_assets if item.block_id == block_id), None
    )
    if visual is None or not visual.image_crop:
        raise FileNotFoundError(f"Visual not found: {block_id}")
    candidate = Path(visual.image_crop)
    path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    resolved = path.resolve()
    processed = PROCESSED_ROOT.resolve()
    if processed not in resolved.parents or resolved.suffix.lower() not in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }:
        raise PermissionError("Visual path is outside the processed asset directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"Visual file not found: {block_id}")
    return resolved


def create_app(manager: RunManager | None = None) -> FastAPI:
    resolved_manager = manager or RunManager(max_concurrent_runs=_max_concurrent_runs())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        resolved_manager.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="Evidence Research API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.run_manager = resolved_manager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/runs", response_model=RunSummary, status_code=202)
    def create_run(payload: RunCreate) -> RunSummary:
        return resolved_manager.create(payload.request)

    @app.get("/api/runs", response_model=RunList)
    def list_runs(limit: int = Query(default=50, ge=1, le=200)) -> RunList:
        return RunList(items=resolved_manager.list(limit=limit))

    @app.get("/api/runs/{run_id}", response_model=RunDetail)
    def get_run(run_id: str) -> RunDetail:
        try:
            return resolved_manager.get(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/cancel", response_model=RunSummary)
    def cancel_run(run_id: str) -> RunSummary:
        try:
            return resolved_manager.cancel(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/events")
    async def run_events(
        request: Request,
        run_id: str,
        after: int = Query(default=0, ge=0),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        if last_event_id and last_event_id.isdigit():
            after = max(after, int(last_event_id))
        try:
            resolved_manager.events_after(run_id, after)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async def generate() -> AsyncIterator[str]:
            sequence = after
            while True:
                events, terminal = resolved_manager.events_after(run_id, sequence)
                for event in events:
                    sequence = event.sequence
                    yield _sse(
                        event.type,
                        event.model_dump(mode="json"),
                        event_id=event.sequence,
                    )
                if terminal:
                    break
                if await request.is_disconnected():
                    break
                await asyncio.to_thread(
                    resolved_manager.wait_for_events, run_id, sequence, 12
                )
                latest, terminal = resolved_manager.events_after(run_id, sequence)
                if not latest and not terminal:
                    yield ": keep-alive\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get(
        "/api/runs/{run_id}/evidence/{evidence_id}/visuals/{block_id}",
        response_class=FileResponse,
    )
    def evidence_visual(run_id: str, evidence_id: str, block_id: str) -> FileResponse:
        try:
            path = _resolve_visual(
                resolved_manager.get(run_id), evidence_id, block_id
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return FileResponse(path)

    web_dist = PROJECT_ROOT / "web" / "dist"
    if web_dist.is_dir():
        assets = web_dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="web-assets")

        @app.get("/{path:path}", include_in_schema=False, response_model=None)
        def web_app(path: str):
            candidate = (web_dist / path).resolve()
            if web_dist.resolve() in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            index = web_dist / "index.html"
            if index.is_file():
                return FileResponse(index)
            return JSONResponse({"detail": "Web build not found"}, status_code=404)

    return app


app = create_app()
