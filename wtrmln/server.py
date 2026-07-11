"""FastAPI backend: connector catalog, connection sessions, SSE event stream,
live browser screen, and user-input forwarding for the login handoff.

Run: uvicorn wtrmln.server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import db
from .agent import ConnectionSession, load_playbooks
from .browser import VIEWPORT

app = FastAPI(title="wtrmln data platform")

SESSIONS: dict[str, ConnectionSession] = {}
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/connectors")
async def connectors():
    books = load_playbooks()
    return [
        {k: v for k, v in b.items() if k != "body"}
        for b in books.values()
    ]


@app.get("/api/connections")
async def connections():
    live = {s.connection_id: s.id for s in SESSIONS.values()}
    rows = db.list_connections()
    for r in rows:
        r["session_id"] = live.get(r["id"])
        if r.get("sync_config"):
            r["sync_config"] = json.loads(r["sync_config"])
    return rows


class ConnectRequest(BaseModel):
    connector: str


@app.post("/api/connections")
async def start_connection(req: ConnectRequest):
    books = load_playbooks()
    if req.connector not in books:
        raise HTTPException(404, f"Unknown connector '{req.connector}'")
    session = ConnectionSession(req.connector, books[req.connector])
    SESSIONS[session.id] = session
    asyncio.create_task(session.run())
    return {"session_id": session.id, "connection_id": session.connection_id}


def _session(session_id: str) -> ConnectionSession:
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, "Unknown session")
    return s


@app.get("/api/sessions/{session_id}/events")
async def events(session_id: str):
    session = _session(session_id)

    async def stream():
        q = session.subscribe()
        try:
            while True:
                event = await q.get()
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "status" and event.get("status") in (
                    "connected", "failed", "blocked", "aborted",
                ):
                    break
        finally:
            session.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/api/sessions/{session_id}/screen")
async def screen(session_id: str):
    session = _session(session_id)
    if not session.latest_screenshot_b64:
        raise HTTPException(404, "No screenshot yet")
    return Response(base64.b64decode(session.latest_screenshot_b64),
                    media_type="image/png",
                    headers={"Cache-Control": "no-store"})


class UserInput(BaseModel):
    kind: str  # click | type | key
    x: float | None = None
    y: float | None = None
    text: str | None = None


@app.post("/api/sessions/{session_id}/input")
async def user_input(session_id: str, inp: UserInput):
    session = _session(session_id)
    if session.status != "awaiting_user":
        raise HTTPException(409, "Browser input is only forwarded during the login handoff")
    if inp.kind == "click" and inp.x is not None and inp.y is not None:
        await session.browser.user_click(
            min(max(inp.x, 0), VIEWPORT["width"]),
            min(max(inp.y, 0), VIEWPORT["height"]),
        )
    elif inp.kind == "type" and inp.text:
        await session.browser.user_type(inp.text)
    elif inp.kind == "key" and inp.text:
        await session.browser.user_key(inp.text)
    else:
        raise HTTPException(400, "Bad input")
    await session._refresh_screen()
    return {"ok": True}


@app.post("/api/sessions/{session_id}/resume")
async def resume(session_id: str):
    session = _session(session_id)
    if session.status != "awaiting_user":
        raise HTTPException(409, "Session is not waiting for login")
    session.resume_from_login()
    return {"ok": True}


@app.post("/api/sessions/{session_id}/abort")
async def abort(session_id: str):
    _session(session_id).abort()
    return {"ok": True}
