---
name: Salesforce
slug: salesforce
icon: ☁️
color: #00A1E0
login_url: https://login.salesforce.com
description: CRM — pipeline, opportunities, accounts, closed-won bookings
---

# Goal

Connect the customer's Salesforce org to wtrmln for pipeline and bookings
data (opportunities, accounts, opportunity history).

# Approach

1. Hand off for Salesforce login (SSO and MFA are common).
2. After login, open Setup (gear icon, top right → Setup).
3. In Setup, create a Connected App:
   Quick Find → "App Manager" → "New Connected App". Name it "wtrmln",
   contact email admin@wtrmln.app. Enable OAuth settings:
   - Callback URL: https://wtrmln.app/oauth/callback/salesforce
   - Scopes: "Access and manage your data (api)" and
     "Perform requests on your behalf at any time (refresh_token, offline_access)"
4. Save. Salesforce may say the app takes a few minutes to activate — that is
   fine; continue.
5. From the app's "Manage Consumer Details" page (may require a verification
   code — hand off to the customer if so), copy the Consumer Key and Consumer
   Secret; save as `salesforce_consumer_key` / `salesforce_consumer_secret`.

# Verification

- Connected App "wtrmln" exists with the correct callback URL and scopes.
- Both credentials saved.

# Sync config to record

- datasets: ["opportunities", "accounts", "opportunity_field_history"]
- method: "oauth_api"
- frequency: "hourly"
