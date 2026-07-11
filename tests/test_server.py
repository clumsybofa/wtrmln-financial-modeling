import base64
import importlib
import os

import pytest
from fastapi.testclient import TestClient


def _load_server():
    import wtrmln.server as server
    return importlib.reload(server)


def _auth(userpass: str) -> dict:
    return {"Authorization": "Basic " + base64.b64encode(userpass.encode()).decode()}


def test_fails_closed_without_secrets(monkeypatch):
    monkeypatch.delenv("WTRMLN_DEV_MODE", raising=False)
    monkeypatch.delenv("WTRMLN_BASIC_AUTH", raising=False)
    monkeypatch.delenv("WTRMLN_VAULT_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Refusing to start"):
        _load_server()


def test_starts_with_secrets_set(monkeypatch):
    monkeypatch.delenv("WTRMLN_DEV_MODE", raising=False)
    monkeypatch.setenv("WTRMLN_BASIC_AUTH", "u:p")
    monkeypatch.setenv("WTRMLN_VAULT_KEY", "x" * 43 + "=")
    server = _load_server()
    assert server.app is not None


def test_basic_auth_enforced(monkeypatch):
    monkeypatch.delenv("WTRMLN_DEV_MODE", raising=False)
    monkeypatch.setenv("WTRMLN_BASIC_AUTH", "team:pw")
    monkeypatch.setenv("WTRMLN_VAULT_KEY", "x" * 43 + "=")
    server = _load_server()
    client = TestClient(server.app)
    assert client.get("/api/connectors").status_code == 401
    assert client.get("/api/connectors", headers=_auth("team:wrong")).status_code == 401
    assert client.get("/api/connectors", headers=_auth("team:pw")).status_code == 200


def test_session_endpoints_require_token(monkeypatch):
    monkeypatch.setenv("WTRMLN_DEV_MODE", "1")
    monkeypatch.delenv("WTRMLN_BASIC_AUTH", raising=False)
    server = _load_server()
    from wtrmln.agent import ConnectionSession, load_playbooks

    books = load_playbooks()
    session = ConnectionSession("xero", books["xero"])
    session.latest_screenshot_b64 = base64.b64encode(b"png").decode()
    server.SESSIONS[session.id] = session

    client = TestClient(server.app)
    base = f"/api/sessions/{session.id}"

    # no token / wrong token -> 403 on every session endpoint
    assert client.get(f"{base}/screen").status_code == 403
    assert client.get(f"{base}/screen?token=wrong").status_code == 403
    assert client.post(f"{base}/abort").status_code == 403
    assert client.post(f"{base}/resume",
                       headers={"X-Session-Token": "nope"}).status_code == 403

    # correct token works
    assert client.get(f"{base}/screen?token={session.token}").status_code == 200
    r = client.post(f"{base}/abort", headers={"X-Session-Token": session.token})
    assert r.status_code == 200

    # unknown session id -> 404
    assert client.get("/api/sessions/doesnotexist/screen").status_code == 404


def test_connection_listing_never_leaks_tokens(monkeypatch):
    monkeypatch.setenv("WTRMLN_DEV_MODE", "1")
    monkeypatch.delenv("WTRMLN_BASIC_AUTH", raising=False)
    server = _load_server()
    from wtrmln.agent import ConnectionSession, load_playbooks

    session = ConnectionSession("xero", load_playbooks()["xero"])
    server.SESSIONS[session.id] = session
    client = TestClient(server.app)
    body = client.get("/api/connections").text
    assert session.token not in body
