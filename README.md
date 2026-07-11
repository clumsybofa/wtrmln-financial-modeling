# 🍉 wtrmln

An **agent-first FP&A data platform**. Same category as Aleph/Datarails, one
structural difference: the customer never does integration work. They click
**Connect Xero** (or QuickBooks, Salesforce, HubSpot…), a Claude
computer-use agent opens a real browser, walks to the login page, hands the
browser to the customer for credentials/MFA — **which never pass through the
model** — then configures API access, saves the secrets into an encrypted
vault, and records what data will sync. The customer watches it happen live.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design.

## Repo layout

| Path | What it is |
|---|---|
| `wtrmln/server.py` | FastAPI backend: connectors, sessions, SSE, live screen, input forwarding |
| `wtrmln/agent.py` | Claude computer-use agent loop + human-login handoff |
| `wtrmln/browser.py` | Playwright browser implementing the `computer_20251124` action set |
| `wtrmln/playbooks/` | Connectors as prose playbooks (Xero, QuickBooks, Salesforce, HubSpot) |
| `wtrmln/vault.py` | Encrypted-at-rest storage for agent-provisioned secrets |
| `wtrmln/static/index.html` | The web UI |
| `app.py` | The original Streamlit financial modeling app (XNPV/XIRR, Monte Carlo) |

## Quickstart

```bash
pip install -r requirements.txt
playwright install chromium   # skip if Chromium is already available

# Demo mode — full flow, no API key:
WTRMLN_MOCK_AGENT=1 uvicorn wtrmln.server:app --port 8000

# Real agent:
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn wtrmln.server:app --port 8000
```

Open http://localhost:8000, click **Connect** on a source, log in when the
banner asks you to, and watch the agent finish the setup.

The modeling app still runs separately: `streamlit run app.py`.

## How the login handoff works

1. The agent drives the browser to the provider's login page and calls its
   `request_human_login` tool. The agent loop **parks**.
2. The UI switches to interactive mode — your clicks and keystrokes are
   forwarded directly to the same browser session. Nothing you type enters
   the model conversation, and no screenshots are sent to the model while
   you're logging in.
3. You click **"I've logged in — continue"**. The agent takes a fresh
   screenshot of the now-authenticated session and finishes the setup:
   creating the OAuth app / API key in the provider's console, saving
   secrets to the encrypted vault, and verifying per the playbook.

## Adding a connector

Write one markdown file in `wtrmln/playbooks/` — front-matter (name, login
URL, icon) plus a prose description of the goal, approach, verification
steps, and sync config. No code.
