---
name: OGsys
slug: ogsys
icon: 🛢️
color: #1B3A5C
login_url: https://ondemand.ogsys.com/sign_in
description: Oil & gas accounting (Quorum On Demand) — GL, JIB, revenue distribution, AFE, severance tax
---

# Goal

Connect the customer's OGsys instance (now sold by Quorum as "On Demand
Accounting") to wtrmln so upstream oil & gas accounting data — general ledger, joint
interest billing, revenue distribution, AFE spend, AP/AR — feeds wtrmln's
partnership and project models.

# Context

This is vertical O&G SaaS, not a mainstream platform like Xero: there is no
self-service developer portal, and API access ("secure REST APIs" per Quorum)
is typically enabled per-account by an administrator. Expect to work with
what the customer's login can actually see, and prefer the fallback ladder
below over getting stuck.

# Approach

1. The browser opens on the OGsys On Demand sign-in page
   (ondemand.ogsys.com). Hand off to the customer for login.
2. After login, note the company/instance name shown in the app header and
   mention it in your narration.
3. Try the integration paths in order — stop at the first one that works:

   **Path A — API access.** Look for an administration or settings area with
   API, integration, or web-services options (menus vary by version; check
   Utilities, Company, or Administration menus). If you can create an API
   key or service credential for wtrmln, do so and save it with
   save_credential as `quorum_api_key` (plus `quorum_company_id` if the UI
   shows an instance/company identifier). Record method "rest_api".

   **Path B — scheduled report exports.** OGsys has a strong report writer.
   If API access is not available to this user, configure (or verify the
   customer can run) recurring exports of: trial balance / GL detail,
   JIB summary, revenue distribution, and AFE status — CSV or Excel format.
   Save any export URLs or report names you set up as credentials/notes
   (e.g. `quorum_export_reports`). Record method "report_export".

   **Path C — blocked.** If the login lacks permissions for both paths,
   call report_blocked telling the customer exactly what to ask their
   OGsys administrator or Quorum support for (API access for a wtrmln
   service user, or report-writer permissions).

4. Do not change any accounting data, posting periods, or company settings —
   this connection is read-only by intent.

# Verification

- Path A: an API credential exists and is saved to the vault.
- Path B: the four core reports exist and can be exported; names recorded.
- Either way: you confirmed which company/instance is connected.

# Sync config to record

- datasets: ["general_ledger", "joint_interest_billing", "revenue_distribution", "afe_spend", "ap_aging", "ar_aging"]
- method: "rest_api" or "report_export" (whichever path succeeded)
- frequency: "daily"
