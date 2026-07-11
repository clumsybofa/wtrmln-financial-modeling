---
name: QuickBooks Online
slug: quickbooks
icon: 📗
color: #2CA01C
login_url: https://accounts.intuit.com/app/sign-in
description: Accounting — GL, P&L, balance sheet, AR/AP aging
---

# Goal

Connect the customer's QuickBooks Online company to wtrmln for accounting
data sync (chart of accounts, P&L, balance sheet, AR/AP aging, invoices).

# Approach

1. Hand off for Intuit login (expect MFA / verification codes).
2. Confirm the active company after login.
3. Open the Intuit developer portal at https://developer.intuit.com/app/developer/dashboard
   in the same session. Create a new app named "wtrmln" using the
   "QuickBooks Online and Payments" API. Scope: com.intuit.quickbooks.accounting.
4. From the app's Keys & credentials page (use Production keys if available,
   otherwise Development), copy the Client ID and Client Secret and save them
   as `quickbooks_client_id` / `quickbooks_client_secret`.
5. Add redirect URI: https://wtrmln.app/oauth/callback/quickbooks

# Verification

- App exists with the accounting scope and the wtrmln redirect URI.
- Both credentials saved.

# Sync config to record

- datasets: ["chart_of_accounts", "profit_and_loss", "balance_sheet", "ar_aging", "ap_aging", "invoices"]
- method: "oauth_api"
- frequency: "hourly"
