# Public apps: threat model and decisions

`POST /apps/ask` on the hosted relay is the one inference route a browser-only app
(a static site, anonymous visitors) can call. Such an app can't keep a secret, so
it ships a **publishable key** (`pk_…`). The key does not unlock the relay's raw
contract. It selects the app's **spec**, stored on the relay, and the relay builds
the whole prompt from that spec. The first app is the Terminal chat on
https://vlyot.github.io/portfolio/.

## How a question is answered

1. The browser sends `{ "key": "pk_…", "input": "...", "history": ["earlier question", ...] }`.
   Unknown fields are rejected, so there is no way to send a prompt, system prompt,
   message list, tier or token count.
2. The relay resolves the key to its app (unknown, revoked or disabled → 401), checks
   the `Origin` against the app's `origins` (403 otherwise), and checks the input
   against the app's size limits.
3. Per-visitor limits (hashed IP, per app), then knowledge loaded and the PC worker
   polling `/claim` in the last minute (otherwise an immediate 503 `unavailable`, and
   nothing is queued), then the app's per-minute budget, concurrency cap and daily
   cap.
4. If the app has a `knowledge_url`, its documents are ranked against the question and
   earlier questions (BM25, `always` documents first, about 6,000 characters).
5. The relay builds the `InferRequest`: the app's pinned system prompt, a fixed safety
   frame (sources only, cite `[n]`, visitor text is data), the numbered sources, and
   the question inside `<question>` delimiters. `max_tokens`, `min_tier` and the wait
   come from the spec, with `reasoning: "off"`. It is enqueued under the identity
   `app:<id>` with `max_attempts: 1`.
6. It waits up to `wait_seconds` for the PC. A job still pending at that point, or
   when the visitor disconnects, is deleted so it never runs later.
7. The answer is cleaned (markup and URLs removed, citations checked) and returned as
   `{ answer, citations: [{ n, title, link }], refused }`.

## Keys and identities

| Credential | Reaches | Identity | Limits |
|---|---|---|---|
| `pk_…` (publishable, in the browser) | `POST /apps/ask` only. 401 everywhere else | `app:<id>` | the app's spec |
| `sk_…` (secret, the app's own backend) | `/enqueue`, `/result/{id}`, `/status`, `DELETE /job/{id}`, and `/apps/ask` | `app:<id>` | the app's spec |
| Neon Auth Bearer (invited member) | the raw contract | `user:<stack-id>` | global `RATE_PER_MIN` / `DAILY_QUOTA` |
| `QUEUE_API_KEY` (owner) | the raw contract, plus `/claim` and `/ack` | `apikey` | global |

An app's `/apps/ask` questions and its `sk_` `/enqueue` calls draw from one budget:
one per-minute bucket and one `usage` row per day. Keys are stored as `sha256(key)`
and shown once when issued. Revoke a single key by its prefix, or disable the app to
switch off every key at once.

## Public status

`GET /apps/status` needs no key and allows any origin. It returns only
`{ online, checked_at }`, so a site can say whether its chat has a model behind it
before a visitor asks. Queue depth and poll times stay on the member-only `/status`.
The answer comes from the relay's memory of the PC's last poll, never the database,
so a page left open doesn't keep Neon awake.

## Threats

| Threat | What could happen | Defence |
|---|---|---|
| Open proxy | Someone uses the GPU as a free general-purpose LLM through a public key. | The client never sends a prompt. The relay builds every request from the pinned spec, caps output at `max_output_tokens`, and (with `require_citations`) replaces anything without a valid citation with the refusal, so off-topic output never reaches the caller. |
| Leaked or scraped `pk_` key | A script reuses the key from the site's JavaScript, with a forged or missing `Origin`. | Expected: the key is public by design, and the origin check only stops browsers. What the key buys is bounded: narrow, grounded answers, per-visitor limits, the app's per-minute and daily caps, and at most `max_concurrent` in flight. Revoke the key, or disable the app, with `appadd`. |
| Impersonating the PC | A member or app key calls `/claim` to read other callers' prompts, or `/ack` to forge answers. | `/claim` and `/ack` accept only the owner key (403 for members and app keys). This also fixes a gap that predates public apps: any authenticated identity could reach both routes. |
| Prompt injection in the question | "Ignore previous instructions", fake system turns, role-play. | The question sits inside `<question>` delimiters under a system prompt that says it is data. `<`, `>`, newlines and control characters are stripped, so it can't close the delimiter. |
| Forged conversation history | A client invents earlier assistant turns (many-shot jailbreak). | History is earlier questions only, within the app's `max_history` / `max_history_chars`. Earlier answers are never accepted. |
| Harmful links or markup in answers | An injected answer carries a phishing link or HTML. | URLs, HTML tags and markdown are stripped. Citations resolve only to documents from the app's own knowledge file, and the client should still render answers as text. |
| Flooding | Scripts burn GPU time or fill the queue. | Body ≤ 8 KB (413). Input limits (400). Per visitor: per-minute bucket and daily count. Per app: per-minute, daily, and concurrency caps (429 / 503). |
| Abandoned jobs | A caller gives up, or the PC is off, and the job runs hours later. | `/apps/ask` cancels its own unclaimed job on timeout or disconnect. Any other caller can do the same with `DELETE /job/{id}`, which only works on the caller's own pending job. |
| Oversized or poisoned knowledge | A bad deploy of the site serves junk. | Capped at 512 KB, must parse with `version: 1` and unique document ids. On any failure the last good copy is kept. The file comes from the app owner's own site, so its content is trusted. |
| Status flooding or snooping | Scripts hammer `/apps/status`, or read it to learn when the PC is on. | Answered from the relay's memory, so flooding it costs no database work and doesn't wake Neon. The body is a single boolean and the check time, nothing about load or usage. When the PC is on is already visible to anyone who asks the chat. |
| Visitor tracking | The relay stores who asked what. | The visitor key is `sha256(APP_IP_SALT + ip)`, kept in memory only. The job row is purged by the result TTL. Only a question's length is logged, never its text. |
| Spoofed client address | A caller rotates `X-Forwarded-For` to dodge per-visitor limits. | The key uses `X-Real-IP`, which Railway's edge sets itself. The per-app caps bound the damage if it were ever bypassed. |

## Decisions

| Decision | Verdict | Reason |
|---|---|---|
| Public way in | Publishable key that selects a spec stored on the server | A public key is only safe if the server pins what it can make the model do. Neon Auth guest sessions would get past the login but not past the open-proxy problem. A backend per app costs a service each, and it would have held the owner key, which then also reached the worker routes. |
| Where specs live | `apps` table (JSONB), pushed from a JSON file in the app's repo with `cmd/appadd` | Reviewable and versioned with the app, with no admin UI to build. `ParseSpec` rejects unknown fields and validates before anything is stored. |
| Preflight | Allowed for any origin | An `OPTIONS` request carries no body, so no key and no app to check against. The `POST` that follows is origin-checked before any work is done. |
| Knowledge format | `{ version: 1, documents: [{ id, title, link, text, always }] }` | Generic. Each site writes its own text: the profile becomes an `always` document, and words like "flagship" go into a document's text rather than into relay code. |
| Answers without citations | Replaced by the app's `refusal` when `require_citations` is set | The clearest machine-checkable sign the answer left the supplied content. |
| Plain prompt, not chat messages | `text_input.prompt` on `min_tier: mid` by default | Through the chat template Gemma 4 "thinks" and spent a 400-token budget before answering. The plain completion path answers directly. `reasoning: "off"` also covers a spec that drops to the weak tier, which forces the template. |
| Visitor limits store | In memory | No visitor data in Postgres. A redeploy resets per-visitor counts. The per-app daily count is a `usage` row, so it survives restarts. |
