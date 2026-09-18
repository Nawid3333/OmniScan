# OmniScan extract.pics webhook relay

Cloudflare Worker + SQLite-backed Durable Object that receives webhooks from
[extract.pics](https://extract.pics), stores them durably and idempotently, and
pushes them to the local OmniScan client over a WebSocket with replay.

## Deploy

1. Create a Cloudflare API token with **Edit Cloudflare Workers** permission at
   https://dash.cloudflare.com/profile/api-tokens.
2. Add GitHub repository secrets:
   - `CLOUDFLARE_API_TOKEN`
   - `CLOUDFLARE_ACCOUNT_ID`
   - `HOOK_SECRET` — generate with `openssl rand -hex 32`
   - `CLIENT_TOKEN` — generate with `openssl rand -hex 32`
3. Push this repo to `main`. The workflow at `.github/workflows/relay-deploy.yml`
   runs `npm run typecheck` and `npm test`, then deploys the Worker and sets the
   secrets.

The resulting webhook URL is:

```
https://omniscan-relay.<subdomain>.workers.dev/hooks/extractpics/<HOOK_SECRET>
```

Enter that URL in extract.pics as the webhook destination for your project.

## Local development

```bash
cd relay
npm install
npm run typecheck
npm test
```

Do **not** run `wrangler deploy` or `wrangler login` from this worktree; deploy
is handled by GitHub Actions.

## Client API

All requests are routed through a single Durable Object instance.

- `GET /health` — returns `200 ok`, no auth.
- `POST /hooks/extractpics/:secret` — receives the extract.pics webhook.
  - Wrong `:secret` → `404`.
  - Body > 5 MiB → `413`.
  - Invalid JSON → `400`.
  - Unknown `type` or missing `id`/`batch_id` → `422`.
  - Valid request → `200 {"ok":true,"duplicate":<bool>,"seq":<n>}`.
- `GET /events?after=<seq>&limit=<n>` — list stored events with `seq > after`.
  Requires `Authorization: Bearer <CLIENT_TOKEN>`. Default limit `100`, max `500`.
  Response: `{"events":[...],"next":<last seq or after>}`.
- `POST /ack` — body `{"upto":<seq>}` deletes events with `seq <= upto`.
  Requires `Authorization: Bearer <CLIENT_TOKEN>`.
  Response: `{"deleted":n}`.
- `GET /ws` — WebSocket upgrade. Requires `Authorization: Bearer <CLIENT_TOKEN>`.
  After connecting send `{"type":"hello","after":<seq>}` to receive every stored
  event with `seq > after`, then live events as they arrive.
