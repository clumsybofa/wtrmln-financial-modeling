"""The connection agent: a Claude computer-use loop that sets up a data
integration by driving a real browser, with a human handoff for login.

Flow per session:
  starting -> agent_working -> awaiting_user (login handoff) -> agent_working
  -> connected | failed | aborted

Security model for the handoff: while status is awaiting_user the loop is
parked inside the request_human_login tool call. The user's clicks and
keystrokes are forwarded directly to the Playwright page by the server — they
never enter the model conversation. When the agent resumes, it only sees the
post-login screen.
"""

import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path

from . import db, vault
from .browser import VIEWPORT, VirtualBrowser

MODEL = os.environ.get("WTRMLN_AGENT_MODEL", "claude-opus-4-8")
COMPUTER_USE_BETA = "computer-use-2025-11-24"
MAX_ITERATIONS = 80

PLAYBOOK_DIR = Path(__file__).resolve().parent / "playbooks"

SYSTEM_PROMPT = """You are the wtrmln connection agent. wtrmln is an FP&A \
platform; your job is to connect a customer's SaaS system (accounting, CRM, \
billing) to wtrmln by operating a real web browser on their behalf, so the \
customer never has to learn the provider's admin console.

You are given a connector playbook below. Follow its goal and verification \
steps, adapting to whatever the provider's UI actually shows — playbooks \
describe intent, not pixel-exact steps.

Rules:
- You control a browser via the computer tool. Take a screenshot first to see \
the current state before acting.
- NEVER ask the user for credentials and never type credentials yourself. \
When you reach any login, SSO, MFA, or CAPTCHA wall, call request_human_login \
and wait. The user completes it directly in the browser; you resume afterward.
- When the provider's UI gives you a secret to copy (an API key, OAuth client \
id/secret, webhook signing secret), read it from the screen (use zoom if it \
is small) and persist it with save_credential immediately — many providers \
show secrets only once.
- When setup is verified per the playbook, call mark_connected with a plain-\
language summary the customer can understand and a sync_config describing \
what data will sync and how.
- If you are blocked and cannot proceed (permissions missing, plan too low, \
provider outage), call report_blocked with a clear explanation of what the \
customer needs to do.
- Narrate briefly before tool calls so the customer can follow along; one \
short sentence is enough.
"""


def load_playbooks() -> dict[str, dict]:
    """Parse playbook markdown files with a simple `---` front-matter block."""
    books = {}
    for path in sorted(PLAYBOOK_DIR.glob("*.md")):
        text = path.read_text()
        meta, body = {}, text
        m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
        if m:
            for line in m.group(1).splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            body = m.group(2)
        meta.setdefault("slug", path.stem)
        meta["body"] = body.strip()
        books[meta["slug"]] = meta
    return books


class ConnectionSession:
    """One live connection attempt: a browser + an agent task + an event log."""

    def __init__(self, connector: str, playbook: dict):
        self.id = uuid.uuid4().hex[:12]
        self.connector = connector
        self.playbook = playbook
        self.connection_id = db.create_connection(connector)
        self.status = "starting"
        self.browser = VirtualBrowser()
        self.events: list[dict] = []
        self._subscribers: list[asyncio.Queue] = []
        self._resume_event = asyncio.Event()
        self._abort = False
        self.latest_screenshot_b64: str | None = None
        self._seq = 0

    # --- event plumbing (SSE) ------------------------------------------------

    def emit(self, type_: str, **data):
        event = {"seq": self._seq, "type": type_, "ts": time.time(), **data}
        self._seq += 1
        self.events.append(event)
        db.log_event(self.id, event["seq"], type_, data)
        for q in list(self._subscribers):
            q.put_nowait(event)

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue()
        for e in self.events:  # replay history so late joiners see everything
            q.put_nowait(e)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        if q in self._subscribers:
            self._subscribers.remove(q)

    def set_status(self, status: str, **extra):
        self.status = status
        db.update_connection(self.connection_id, status, extra.get("summary"),
                             extra.get("sync_config"))
        self.emit("status", status=status, **extra)

    def resume_from_login(self):
        self._resume_event.set()

    def abort(self):
        self._abort = True
        self._resume_event.set()  # unblock a parked handoff

    async def _refresh_screen(self):
        try:
            self.latest_screenshot_b64 = await self.browser.screenshot_b64()
            self.emit("screen")  # UI re-fetches /screen on this signal
        except Exception:
            pass

    # --- the agent loop -------------------------------------------------------

    async def run(self):
        try:
            await self.browser.start(self.playbook.get("login_url", "about:blank"))
            await self._refresh_screen()
            if os.environ.get("WTRMLN_MOCK_AGENT") == "1":
                await self._run_mock()
            else:
                await self._run_agent()
        except Exception as e:
            self.set_status("failed", summary=f"Unexpected error: {e}")
        finally:
            await self.browser.stop()

    def _tools(self):
        return [
            {
                "type": "computer_20251124",
                "name": "computer",
                "display_width_px": VIEWPORT["width"],
                "display_height_px": VIEWPORT["height"],
                "enable_zoom": True,
            },
            {
                "name": "request_human_login",
                "description": (
                    "Pause and hand the browser to the customer so they can log "
                    "in (credentials, SSO, MFA, CAPTCHA). Their input goes "
                    "directly to the browser and is never shown to you. Returns "
                    "when they signal completion; verify with a screenshot."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "What the customer should do, e.g. 'Log in to Xero, including 2FA.'"}
                    },
                    "required": ["reason"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "save_credential",
                "description": (
                    "Persist a secret you provisioned in the provider's UI "
                    "(API key, OAuth client id/secret, export URL) into "
                    "wtrmln's encrypted vault."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Identifier, e.g. 'xero_client_secret'"},
                        "value": {"type": "string"},
                    },
                    "required": ["name", "value"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "mark_connected",
                "description": "Declare the integration verified and finished.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "Plain-language summary for the customer."},
                        "sync_config": {
                            "type": "object",
                            "properties": {
                                "datasets": {"type": "array", "items": {"type": "string"}},
                                "method": {"type": "string", "description": "How data will be pulled, e.g. 'oauth_api', 'report_export'"},
                                "frequency": {"type": "string"},
                            },
                            "required": ["datasets", "method", "frequency"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["summary", "sync_config"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "report_blocked",
                "description": "Stop because setup cannot proceed; tell the customer what they must do first.",
                "input_schema": {
                    "type": "object",
                    "properties": {"explanation": {"type": "string"}},
                    "required": ["explanation"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ]

    async def _run_agent(self):
        import anthropic

        client = anthropic.AsyncAnthropic()
        self.set_status("agent_working")
        self.emit("log", text=f"Starting {self.playbook.get('name', self.connector)} setup…")

        messages = [{
            "role": "user",
            "content": (
                f"Connect this customer's {self.playbook.get('name', self.connector)} "
                f"account to wtrmln. Connector playbook follows.\n\n{self.playbook['body']}\n\n"
                f"The browser is already open at {self.playbook.get('login_url', 'about:blank')}. Begin."
            ),
        }]

        for _ in range(MAX_ITERATIONS):
            if self._abort:
                self.set_status("aborted", summary="Cancelled by user.")
                return

            response = await client.beta.messages.create(
                model=MODEL,
                max_tokens=8192,
                betas=[COMPUTER_USE_BETA],
                thinking={"type": "adaptive"},
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                tools=self._tools(),
                messages=messages,
            )

            for block in response.content:
                if block.type == "text" and block.text.strip():
                    self.emit("agent_message", text=block.text)

            if response.stop_reason == "refusal":
                self.set_status("failed", summary="The agent declined to continue this task.")
                return

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                if response.stop_reason == "pause_turn":
                    messages.append({"role": "assistant", "content": response.content})
                    continue
                self.set_status("failed", summary="Agent stopped without completing setup.")
                return

            messages.append({"role": "assistant", "content": response.content})
            results = []
            done = False
            for tu in tool_uses:
                result_content, finished = await self._handle_tool(tu.name, tu.input or {})
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result_content,
                })
                done = done or finished
            messages.append({"role": "user", "content": results})
            if done:
                return

        self.set_status("failed", summary="Setup exceeded the step limit without finishing.")

    async def _handle_tool(self, name: str, tool_input: dict) -> tuple[list | str, bool]:
        """Execute one tool call. Returns (tool_result content, session_finished)."""
        if name == "computer":
            self.emit("action", action=tool_input.get("action"),
                      detail={k: v for k, v in tool_input.items() if k != "action"})
            out = await self.browser.execute(tool_input)
            await self._refresh_screen()
            if "image_b64" in out:
                return [{
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png",
                               "data": out["image_b64"]},
                }], False
            return out.get("text", "ok"), False

        if name == "request_human_login":
            reason = tool_input.get("reason", "Please log in.")
            self._resume_event.clear()
            self.set_status("awaiting_user", reason=reason)
            self.emit("log", text="Handing the browser to you for login. "
                                  "Your keystrokes go straight to the site — the AI never sees them.")
            while True:
                try:
                    await asyncio.wait_for(self._resume_event.wait(), timeout=1.0)
                    break
                except asyncio.TimeoutError:
                    await self._refresh_screen()  # keep the user's live view fresh
            if self._abort:
                self.set_status("aborted", summary="Cancelled by user.")
                return "aborted", True
            self.set_status("agent_working")
            self.emit("log", text="Login confirmed — resuming setup.")
            return ("The customer reports login is complete. Take a screenshot "
                    "to verify you are signed in before proceeding."), False

        if name == "save_credential":
            vault.store_credential(self.connection_id, tool_input["name"], tool_input["value"])
            self.emit("credential_saved", name=tool_input["name"])  # value never emitted
            return f"Stored '{tool_input['name']}' in the encrypted vault.", False

        if name == "mark_connected":
            self.set_status("connected", summary=tool_input.get("summary"),
                            sync_config=tool_input.get("sync_config"))
            return "Connection recorded. You are done.", True

        if name == "report_blocked":
            self.set_status("blocked", summary=tool_input.get("explanation"))
            return "Blocker recorded. You are done.", True

        return f"Unknown tool {name}", False

    # --- mock mode (no API key needed) ---------------------------------------

    async def _run_mock(self):
        """Scripted walk-through of the full state machine for demos/tests."""
        self.set_status("agent_working")
        name = self.playbook.get("name", self.connector)
        self.emit("agent_message", text=f"I'll set up your {name} connection. Let me look at the login page first.")
        await self.browser.execute({"action": "screenshot"})
        await self._refresh_screen()
        await asyncio.sleep(1)

        _, done = await self._handle_tool("request_human_login",
                                          {"reason": f"Log in to {name}, including any 2FA."})
        if done:
            return

        self.emit("agent_message", text="You're signed in. I'm creating wtrmln's API access now…")
        await self.browser.execute({"action": "screenshot"})
        await self._refresh_screen()
        await asyncio.sleep(1.5)
        await self._handle_tool("save_credential",
                                {"name": f"{self.connector}_demo_key", "value": "mock-" + uuid.uuid4().hex})
        await asyncio.sleep(1)
        await self._handle_tool("mark_connected", {
            "summary": f"{name} is connected (demo mode). In a real run I would have "
                       "provisioned API access and verified a data pull.",
            "sync_config": {"datasets": ["demo"], "method": "mock", "frequency": "daily"},
        })
