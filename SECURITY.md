# Security

## Reporting

Report vulnerabilities privately via GitHub security advisories on this
repository (Security → Report a vulnerability). Please do not open public
issues for security reports.

## Security model (v0)

- **Fail closed.** The server refuses to start unless `WTRMLN_BASIC_AUTH`
  and `WTRMLN_VAULT_KEY` are set. `WTRMLN_DEV_MODE=1` bypasses this for
  local development only and must never be set on an internet-facing deploy.
- **Session authorization.** Every session endpoint (events, screen, input,
  resume, abort) requires the per-session bearer token issued at creation.
  Session ids alone grant nothing.
- **Login handoff.** User-entered credentials go directly from the
  operator's browser to the provider page via input forwarding. The model
  receives no screenshots and no text while a session is `awaiting_user`.
- **Agent-provisioned secrets DO pass through the model.** API keys and
  OAuth client secrets that the agent creates in a provider's console are
  read off the screen by the model and echoed into the `save_credential`
  tool call. This is inherent to agent-driven setup. They are stored
  Fernet-encrypted at rest and never emitted to the event feed or API.
- **Vault key separation.** The Fernet key comes from `WTRMLN_VAULT_KEY`
  (a secrets manager in hosted deploys). Auto-generated colocated keys are
  dev-mode only.
- **Browser containment.** Each session runs an isolated browser context.
  Top-level navigation is restricted to the connector playbook's
  `allowed_domains`. TLS verification is on; `WTRMLN_PROXY_INSECURE_TLS=1`
  is an explicit dev-sandbox escape hatch only.
- **Resource limits.** Concurrent sessions are capped
  (`WTRMLN_MAX_SESSIONS`, default 3), sessions time out
  (`WTRMLN_SESSION_TIMEOUT`, default 30 min), in-memory event logs are
  capped, and finished sessions are pruned.
- **UI.** All model/provider-derived strings render as text (no innerHTML
  sinks); a restrictive CSP is set.

## Known v0 limitations

- Single-operator HTTP Basic auth — no per-user accounts or tenant
  separation yet (planned: v1, Supabase auth + RLS).
- "Connected" records the agent's observed verification evidence
  (`verification_evidence` in the sync config) but is not yet validated by
  a provider API call (`verified_by_sync: false` until the sync engine
  lands).
- SQLite/local-disk persistence; no KMS-backed key rotation yet.
