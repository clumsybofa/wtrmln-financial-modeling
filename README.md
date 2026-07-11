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

# Demo mode — full flow, no API key (dev mode: auth + vault checks relaxed):
WTRMLN_DEV_MODE=1 WTRMLN_MOCK_AGENT=1 uvicorn wtrmln.server:app --port 8000

# Real agent, locally:
export ANTHROPIC_API_KEY=sk-ant-...
WTRMLN_DEV_MODE=1 uvicorn wtrmln.server:app --port 8000
```

Outside dev mode the server **fails closed**: it refuses to start unless
`WTRMLN_BASIC_AUTH` and `WTRMLN_VAULT_KEY` are set (see `.env.example` and
`SECURITY.md`).

Open http://localhost:8000, click **Connect** on a source, log in when the
banner asks you to, and watch the agent finish the setup.

The modeling app still runs separately: `streamlit run app.py`.

## Deploying to the web (v0)

The whole platform is one long-running process (FastAPI + in-process
Chromium), so it deploys as a single Docker image. Serverless hosts
(Vercel/Netlify functions) won't work — the browser and SSE streams need a
persistent machine.

**Fly.io** (config included):

```bash
fly launch --copy-config --no-deploy
fly volumes create wtrmln_data --size 1
fly secrets set ANTHROPIC_API_KEY=sk-ant-... WTRMLN_BASIC_AUTH=team:strongpassword
fly deploy
```

**Railway / Render / any Docker host:** point it at the `Dockerfile`, expose
port 8080, mount a volume at `/app/data`, and set the same two environment
variables.

`WTRMLN_BASIC_AUTH=user:pass` puts HTTP Basic auth in front of everything —
required for anything internet-facing, since sessions display customer
data. Size the machine ~4 GB RAM: each concurrent onboarding session runs
its own Chromium (~300–500 MB). Real multi-tenant auth, per-tenant vaults,
and a remote browser pool are the v1/v2 steps (see
`docs/ARCHITECTURE.md`).

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
