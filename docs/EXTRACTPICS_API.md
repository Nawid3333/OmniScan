# extract.pics API — as observed on 2026-09-20 (live probe, 1 basic credit spent on a CC-BY Pepper&Carrot page)

The docs site is a JavaScript app that no fetch tool can read; everything below was measured against `https://api.extract.pics`. Builders test against fakes shaped like this.

- **Auth:** header `Authorization: <api key>` — the raw key, **no `Bearer` prefix**. Wrong key → `401 {"message":"Unauthenticated."}`. The key lives in `~/.config/omniscan/secrets.env` (`EXTRACTPICS_API_KEY`, loaded by `core.config.Secrets`); never print it.
- **Create:** `POST /v0/extractions`, JSON body `{"url": "<page url>", "mode": "basic" | "advanced"}` → `201`
  `{"data": {"id": "<uuid>", "status": "pending", "url": "<canonicalised, lower-cased url>", "images": [], "created_at": "...", "project_id": "<uuid>"}}`
  Invalid url → `422 {"message": "The url format is invalid.", "errors": {"url": ["..."]}}`.
- **Poll:** `GET /v0/extractions/<id>` → `200` same envelope; `status` goes `pending` → `running` → `done` (took ~2–4 s for a basic extraction). `images` is filled when `done`: a list of `{"id": "<uuid>", "url": "<absolute url>"}` — **no width/height**. A failed extraction is expected to report another status (`failed`/`error`) — treat every status other than `pending|running|done` as failure and keep the raw body in the error message. Unknown id → `404 {"message": "No query results for model [...]"}`.
- **`GET /v0/extractions` (list) is not allowed** (`405`); there is no way to list past extractions.
- **Rate limit:** headers `x-ratelimit-limit: 25`, `x-ratelimit-remaining: <n>` on every response (25 requests per minute as observed). A `429` is expected when exceeded; honour `Retry-After` if present.
- **Cost (owner's docs summary):** 1 credit per basic extraction, 2 per advanced (JS + scrolling); free plan 100 credits/month; failed extractions cost nothing. The response carries no credit information, so the client counts credits itself.
- **The image list is a dump of every `<img>`/CSS image on the page in document order**, not only the chapter pages. The probe page returned 43 URLs: 29 SVG/PNG interface icons (`/core/img/*.svg`, favicon), 2 thumbnails (`/cache/…_120x120px_….jpg`), 1 Open-Graph style URL without extension, 1 hi-res "gfx-only" artwork, and the 11 chapter pages `…/low-res/en_Pepper-and-Carrot_by-David-Revoy_E06P00.jpg … P10.jpg` in the middle. So the acquire step must filter (SVG, icons, thumbnails, tiny images) and pick the run of pages — see cards B33b/B33c.
- Image hosts may check the `Referer`: send the chapter page URL when downloading (the downloader's `ImageRef.referer`).
