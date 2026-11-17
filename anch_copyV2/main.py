"""FastAPI server for クリスタルLM (CrystalLM)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent_core import AgentHarness, DEFAULT_LM_STUDIO_URL, check_lm_studio

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="クリスタルLM (CrystalLM)", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

harness = AgentHarness(BASE_DIR)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1)
    max_steps: Optional[int] = Field(default=None, ge=0)
    model: Optional[str] = None
    supervisor_model: Optional[str] = None
    coder_model: Optional[str] = None
    standard_model: Optional[str] = None
    base_url: Optional[str] = DEFAULT_LM_STUDIO_URL
    mode: Optional[str] = None


class MemoryUpdateRequest(BaseModel):
    content: str


class CreateSessionRequest(BaseModel):
    title: str = "新しいチャット"
    mode: str = "normal"


class ModelSelectRequest(BaseModel):
    model: str = ""
    base_url: Optional[str] = DEFAULT_LM_STUDIO_URL
    parallel: int = Field(default=1, ge=1, le=1)
    role: str = "main"
    previous_model: Optional[str] = None
    keep_models: Optional[list[str]] = None


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/models")
async def get_models(base_url: str = Query(default=DEFAULT_LM_STUDIO_URL)) -> dict:
    return await check_lm_studio(base_url)


@app.post("/api/models/select")
async def select_model(req: ModelSelectRequest) -> dict:
    result = await harness.select_model(
        req.model,
        req.base_url,
        parallel=1,
        role=req.role,
        previous_model=req.previous_model,
        keep_models=req.keep_models,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=result.get("message", "モデル切り替えに失敗しました"))
    return result


def _annotate_sessions_with_running(sessions: list[dict]) -> list[dict]:
    running_id = harness.state.session_id if harness.is_running() else None
    return [{**s, "is_running": bool(running_id and s.get("id") == running_id)} for s in sessions]


@app.get("/api/live")
async def get_live_state(session_id: Optional[str] = Query(default=None)) -> dict:
    return harness.get_session_live_status(session_id)


@app.get("/api/status")
async def get_status(base_url: str = Query(default=DEFAULT_LM_STUDIO_URL)) -> dict:
    lm_status = await check_lm_studio(base_url)
    return {
        "agent": harness._status_payload(),
        "lm_studio": lm_status,
        "memory": harness.memory.read_all(),
        "sessions": _annotate_sessions_with_running(harness.sessions.list_sessions()),
        "files": harness.list_workspace_files(),
    }


@app.get("/api/sessions")
async def list_sessions() -> dict:
    return {"sessions": _annotate_sessions_with_running(harness.sessions.list_sessions())}


@app.post("/api/sessions")
async def create_session(req: CreateSessionRequest) -> dict:
    session = harness.sessions.create_session(title=req.title, mode=req.mode)
    return {"session": session}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    try:
        session = harness.sessions.get_session(session_id)
        return {
            "session": session,
            "live": harness.get_session_live_status(session_id),
            "tree": harness.sessions.get_session_tree(session_id),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/tree")
async def get_session_tree(session_id: str) -> dict:
    try:
        harness.sessions.get_session(session_id)
        return harness.sessions.get_session_tree(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/files")
async def list_session_files(session_id: str) -> dict:
    """List all folders and files (relative paths) under sessions/<session_id>/."""
    try:
        harness.sessions.get_session(session_id)
        tree_data = harness.sessions.get_session_tree(session_id)
        return {
            "session_id": tree_data["session_id"],
            "tree": tree_data["tree"],
            "files": tree_data["flat_files"],
            "flat_files": tree_data["flat_files"],
            "ultrathink": tree_data.get("ultrathink"),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/file_content")
async def read_session_file_content(session_id: str, path: str = Query(..., min_length=1)) -> dict:
    """Read UTF-8 content of a file inside sessions/<session_id>/ with strict path-traversal protection."""
    try:
        return harness.sessions.read_session_file(session_id, path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/file")
async def read_session_file(session_id: str, path: str = Query(..., min_length=1)) -> dict:
    try:
        return harness.sessions.read_session_file(session_id, path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/sessions/{session_id}/file")
async def write_session_file(session_id: str, req: MemoryUpdateRequest, path: str = Query(..., min_length=1)) -> dict:
    try:
        saved = harness.sessions.write_session_file(session_id, path, req.content)
        if saved["path"] in ("plan.md", "longthink/shared_workspace/master_plan.md", "ultrathink/GLOBAL_INDEX.md"):
            await harness.broker.publish(
                "memory_update",
                {"filename": "plan.md", "content": req.content, "session_id": session_id},
            )
        await harness.broker.publish(
            "files_update",
            {"session_id": session_id, "tree": harness.sessions.get_session_tree(session_id)},
        )
        return {"ok": True, **saved}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    try:
        if harness.is_running() and harness.state.session_id == session_id:
            await harness.stop_task()
            await asyncio.sleep(0.05)
        deleted = harness.sessions.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="セッションが見つかりません。")
        return {"ok": True}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/memory")
async def get_memory() -> dict:
    return {"memory": harness.memory.read_all()}


@app.put("/api/memory/{filename}")
async def update_memory_file(filename: str, req: MemoryUpdateRequest) -> dict:
    try:
        saved = harness.memory.write_file(filename, req.content)
        await harness.broker.publish(
            "memory_update",
            {"filename": filename, "content": saved},
        )
        return {"ok": True, "filename": filename, "content": saved}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/chat")
async def chat_turn(req: ChatRequest) -> dict:
    try:
        result = await harness.start_chat_turn(
            session_id=req.session_id,
            user_message=req.message.strip(),
            max_steps=req.max_steps,
            model=req.model,
            supervisor_model=req.supervisor_model,
            coder_model=req.coder_model,
            standard_model=req.standard_model,
            base_url=req.base_url,
            mode=req.mode,
        )
        return {"ok": True, **result}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/stop")
async def stop_agent() -> dict:
    status = await harness.stop_task()
    return {"ok": True, "agent": status}


@app.get("/api/files")
async def list_files() -> dict:
    return {"files": harness.list_workspace_files()}


@app.get("/api/files/content")
async def read_workspace_file(path: str = Query(..., min_length=1)) -> dict:
    workspace_root = harness.workspace_root.resolve()
    cleaned = path.strip().lstrip("/\\")
    target = (workspace_root / cleaned).resolve()
    if target != workspace_root and workspace_root not in target.parents:
        raise HTTPException(status_code=403, detail="workspace外のファイルにはアクセスできません。")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="ファイルが見つかりません。")
    content = target.read_text(encoding="utf-8", errors="replace")
    return {
        "path": target.relative_to(workspace_root).as_posix(),
        "content": content,
    }


@app.get("/api/stream")
async def stream_events() -> StreamingResponse:
    queue = harness.broker.subscribe()

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    payload = json.dumps(event, ensure_ascii=False)
                    yield f"event: {event['type']}\ndata: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            harness.broker.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
