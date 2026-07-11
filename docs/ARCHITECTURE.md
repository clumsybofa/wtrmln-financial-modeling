# wtrmln — Agent-First Data Platform Architecture

## The thesis

FP&A tools (Aleph, Datarails, etc.) die in onboarding. The customer has to
learn the vendor's tool *and* wire it into their own stack — map their chart
of accounts, configure OAuth apps in five different admin consoles, learn a
proprietary modeling language. wtrmln's differentiator: **the customer never
does integration work.** An AI agent (Claude, driving a real browser via
computer use) does it for them, with the customer only supplying the one
thing an agent must never touch — their credentials.

## The core interaction

```
Customer clicks "Connect Xero"
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ ConnectionSession                                          │
│                                                            │
│  Claude (computer_20251124 tool) ──► Playwright Chromium   │
│        ▲                                    │              │
│        │  screenshots / tool_results        │ screenshots  │
│        │                                    ▼              │
│        │                            Web UI (live view)     │
│        │                                    ▲              │
│  request_human_login ──► pause loop         │ clicks/keys  │
│        (agent parked)                Customer logs in ─────┘
│        ◄── resume ── customer clicks "I've logged in"
│                                                            │
│  save_credential ──► encrypted vault (Fernet, local)       │
│  mark_connected  ──► connection record + sync config       │
└───────────────────────────────────────────────────────────┘
```

## Key design decisions

### 1. Connectors are playbooks, not code

A connector is a markdown file (`wtrmln/playbooks/*.md`) with front-matter
(name, login URL, icon) and a body describing **intent**: the goal, the
rough approach, verification criteria, and what sync config to record. The
agent adapts to whatever the provider's UI actually looks like. Adding a
connector = writing a page of prose. No SDKs, no API version chasing, no
brittle selectors.

### 2. Credentials never pass through the model

The single hard security rule. When the agent hits a login/SSO/MFA/CAPTCHA
wall it calls `request_human_login` and the loop **parks**:

- Session status flips to `awaiting_user`.
- The UI overlays an interactive mode: the customer's clicks and keystrokes
  are POSTed to `/api/sessions/{id}/input` and forwarded straight to the
  same Playwright page. Nothing the customer types is added to the model
  conversation, and no screenshots are sent to the model while parked.
- The customer clicks "I've logged in" → the tool call returns a plain text
  result ("login complete, verify with a screenshot") and the agent resumes
  in the now-authenticated session.

Secrets the agent *provisions* (OAuth client secrets, API keys it creates in
the provider's dev console — which is the point of setup) are persisted via
the `save_credential` tool into a Fernet-encrypted vault on disk, and are
never echoed into the event feed.

### 3. The agent loop is plain Messages API, not a framework

`wtrmln/agent.py` runs a manual tool-use loop against
`client.beta.messages.create` with:

- `computer_20251124` (beta `computer-use-2025-11-24`), 1280×800 display,
  `enable_zoom: true` so the agent can read small secrets on screen
- Model: `claude-opus-4-8` (override with `WTRMLN_AGENT_MODEL`)
- Adaptive thinking, cached system prompt
- Custom tools: `request_human_login`, `save_credential`, `mark_connected`,
  `report_blocked` — the state machine is enforced by tools, not prose
- Iteration cap (80) as a runaway backstop

### 4. Session state machine

`starting → agent_working ⇄ awaiting_user → connected | blocked | failed | aborted`

`blocked` is a first-class outcome: "you need admin rights in Salesforce" is
a successful agent run with a clear next step for the customer, not an error.

### 5. Observability is the product

Watching the agent work *is* the onboarding UX. Every session emits an event
stream (SSE): agent narration, each computer action, credential saves
(name only), status transitions, plus a `screen` signal that makes the UI
re-fetch the latest screenshot. Events are also persisted to SQLite for
audit.

## What exists today (phase 1)

- FastAPI backend (`wtrmln/server.py`) + single-page UI (`wtrmln/static/`)
- The computer-use agent loop with human-login handoff (`wtrmln/agent.py`)
- Playwright virtual browser implementing the full `computer_20251124`
  action set (`wtrmln/browser.py`)
- Playbooks: Xero, QuickBooks Online, Salesforce, HubSpot
- Encrypted credential vault, SQLite persistence
- Mock mode (`WTRMLN_MOCK_AGENT=1`) exercising the whole state machine
  without an API key
- The original Streamlit modeling app (`app.py`) — the eventual consumer of
  synced data

## Phase 2 (next)

1. **Sync engine.** Connections record a `sync_config` (datasets, method,
   frequency). Implement pulls with the credentials in the vault: OAuth API
   where available; where APIs are gated or terrible, the same computer-use
   agent runs scheduled "export report → download → land as parquet" jobs.
   Data lands in `data/{connection_id}/` and feeds the modeling layer.
2. **OAuth callback.** Host `wtrmln.app/oauth/callback/*` so the client
   ids/secrets the agent provisions complete a real token exchange.
3. **Model mapping agent.** Second agent type: reads the synced chart of
   accounts and maps it to the customer's model (the other half of what
   makes Aleph painful).
4. **Multi-provider.** The agent runner is provider-agnostic by design —
   an OpenAI computer-use backend is a second implementation of the same
   `_run_agent` contract against the same browser/tool surface.
5. **Hardening.** Per-session browser contexts are already isolated;
   add domain allowlists per playbook, session recording, and human
   approval gates for destructive actions.
