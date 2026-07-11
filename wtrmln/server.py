"""FastAPI backend: connector catalog, connection sessions, SSE event stream,
live browser screen, and user-input forwarding for the login handoff.

Run: uvicorn wtrmln.server:app --host 0.0.0.0 --port 8000

Security model (v0, single-operator):
- Fails closed: outside dev mode (WTRMLN_DEV_MODE=1) the process refuses to
  start unless WTRMLN_BASIC_AUTH and WTRMLN_VAULT_KEY are both set.
- HTTP Basic auth covers every route.
- Each session has a bearer token issued at creation; all session endpoints
  (watch, screen, input, resume, abort) require it. Knowing a session id is
  not enough to observe or control a session.
"""

import asyncio
import base64
import json
import os
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db
from .agent import TERMINAL_STATUSES, ConnectionSession, load_playbooks
from .browser import VIEWPORT

DEV_MODE = os.environ.get("WTRMLN_DEV_MODE") == "1"
MAX_ACTIVE_SESSIONS = int(os.environ.get("WTRMLN_MAX_SESSIONS", "3"))
FINISHED_SESSION_TTL = 3600  # seconds a finished session stays queryable


def _enforce_secure_config():
    if DEV_MODE:
        return
    missing = [v for v in ("WTRMLN_BASIC_AUTH", "WTRMLN_VAULT_KEY")
               if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            f"Refusing to start: {', '.join(missing)} not set. "
            "Set them (see .env.example) or export WTRMLN_DEV_MODE=1 for "
            "local development only."
        )


_enforce_secure_config()

app = FastAPI(title="wtrmln data platform")

SESSIONS: dict[str, ConnectionSession] = {}
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    expected = os.environ.get("WTRMLN_BASIC_AUTH")
    if expected:
        header = request.headers.get("authorization", "")
        supplied = ""
        if header.startswith("Basic "):
            try:
                supplied = base64.b64decode(header[6:]).decode()
            except Exception:
                supplied = ""
        if not secrets.compare_digest(supplied, expected):
            return Response(status_code=401, content="Authentication required",
                            headers={"WWW-Authenticate": 'Basic realm="wtrmln"'})
    return await call_next(request)


def _prune_sessions():
    now = time.time()
    for sid, s in list(SESSIONS.items()):
        if s.status in TERMINAL_STATUSES and now - s.finished_at > FINISHED_SESSION_TTL:
            del SESSIONS[sid]


def _active_count() -> int:
    return sum(1 for s in SESSIONS.values() if s.status not in TERMINAL_STATUSES)


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/connectors")
async def connectors():
    books = load_playbooks()
    return [{k: v for k, v in b.items() if k != "body"} for b in books.values()]


@app.get("/api/connections")
async def connections():
    live = {s.connection_id: s.id for s in SESSIONS.values()}
    rows = db.list_connections()
    for r in rows:
        r["session_id"] = live.get(r["id"])  # id only; control requires the token
        if r.get("sync_config"):
            r["sync_config"] = json.loads(r["sync_config"])
    return rows


class ConnectRequest(BaseModel):
    connector: str


@app.post("/api/connections")
async def start_connection(req: ConnectRequest):
    _prune_sessions()
    if _active_count() >= MAX_ACTIVE_SESSIONS:
        raise HTTPException(429, f"Too many active sessions (max {MAX_ACTIVE_SESSIONS}); "
                                 "finish or cancel one first")
    books = load_playbooks()
    if req.connector not in books:
        raise HTTPException(404, f"Unknown connector '{req.connector}'")
    session = ConnectionSession(req.connector, books[req.connector])
    SESSIONS[session.id] = session
    asyncio.create_task(session.run())
    # The token is returned once, to the creator only, and never persisted
    # or listed. It is required for every session endpoint below.
    return {"session_id": session.id, "connection_id": session.connection_id,
            "session_token": session.token}


def _session(session_id: str, request: Request) -> ConnectionSession:
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(404, "Unknown session")
    supplied = (request.headers.get("x-session-token")
                or request.query_params.get("token") or "")
    if not secrets.compare_digest(supplied, s.token):
        raise HTTPException(403, "Missing or invalid session token")
    return s


@app.get("/api/sessions/{session_id}/events")
async def events(session_id: str, request: Request):
    session = _session(session_id, request)

    async def stream():
        q = session.subscribe()
        try:
            while True:
                event = await q.get()
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "status" and event.get("status") in TERMINAL_STATUSES:
                    break
        finally:
            session.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/api/sessions/{session_id}/screen")
async def screen(session_id: str, request: Request):
    session = _session(session_id, request)
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
async def user_input(session_id: str, inp: UserInput, request: Request):
    session = _session(session_id, request)
    if session.status != "awaiting_user":
        raise HTTPException(409, "Browser input is only forwarded during the login handoff")
    if inp.kind == "click" and inp.x is not None and inp.y is not None:
        await session.browser.user_click(
            min(max(inp.x, 0), VIEWPORT["width"]),
            min(max(inp.y, 0), VIEWPORT["height"]),
        )
    elif inp.kind == "type" and inp.text:
        await session.browser.user_type(inp.text[:512])
    elif inp.kind == "key" and inp.text:
        await session.browser.user_key(inp.text[:32])
    else:
        raise HTTPException(400, "Bad input")
    await session.refresh_screen()
    return {"ok": True}


@app.post("/api/sessions/{session_id}/resume")
async def resume(session_id: str, request: Request):
    session = _session(session_id, request)
    if session.status != "awaiting_user":
        raise HTTPException(409, "Session is not waiting for login")
    session.resume_from_login()
    return {"ok": True}


@app.post("/api/sessions/{session_id}/abort")
async def abort(session_id: str, request: Request):
    _session(session_id, request).abort()
    return {"ok": True}
