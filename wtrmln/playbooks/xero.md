---
name: Xero
slug: xero
icon: 📘
color: #13B5EA
allowed_domains: xero.com, login.xero.com
login_url: https://login.xero.com/identity/user/login
description: Accounting — GL, P&L, balance sheet, invoices, bank transactions
---

# Goal

Connect the customer's Xero organisation to wtrmln so accounting data
(chart of accounts, P&L, balance sheet, invoices, bank transactions) syncs
automatically.

# Approach

1. The browser opens on the Xero login page. Hand off to the customer for
   login (Xero commonly enforces MFA).
2. Once signed in, confirm which organisation is active (top-left org menu).
   If several orgs exist, use the one the customer most recently used and say
   so in your narration.
3. Go to the Xero developer portal at https://developer.xero.com/app/manage
   (same session) and create a new app named "wtrmln" of type
   "Web app". Set:
   - Company or application URL: https://wtrmln.app
   - Redirect URI: https://wtrmln.app/oauth/callback/xero
4. Copy the Client ID and generate a Client Secret. Save both with
   save_credential as `xero_client_id` and `xero_client_secret` —
   the secret is shown only once.
5. Verify the app appears in the app list.

# Verification

- App "wtrmln" exists in the developer portal with the correct redirect URI.
- Both credentials saved.

# Sync config to record

- datasets: ["chart_of_accounts", "profit_and_loss", "balance_sheet", "invoices", "bank_transactions"]
- method: "oauth_api"
- frequency: "hourly"
