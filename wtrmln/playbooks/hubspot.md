---
name: HubSpot
slug: hubspot
icon: 🧡
color: #FF7A59
allowed_domains: hubspot.com, hubapi.com
login_url: https://app.hubspot.com/login
description: CRM — deals, pipeline, companies, contacts
---

# Goal

Connect the customer's HubSpot portal to wtrmln for deal and pipeline data.

# Approach

1. Hand off for HubSpot login.
2. After login, note which portal (account) is active.
3. Create a Private App: Settings (gear icon) → Integrations → Private Apps →
   "Create a private app". Name it "wtrmln".
4. On the Scopes tab, grant read scopes: crm.objects.deals.read,
   crm.objects.companies.read, crm.objects.contacts.read,
   crm.schemas.deals.read.
5. Create the app and copy the access token; save as `hubspot_access_token`.

# Verification

- Private app "wtrmln" exists with the read scopes above.
- Access token saved.

# Sync config to record

- datasets: ["deals", "companies", "contacts", "pipelines"]
- method: "private_app_token"
- frequency: "hourly"
