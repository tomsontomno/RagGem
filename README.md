# RagGem

A small, hardened **Retrieval-Augmented Generation (RAG) API server**. You give
it documents; it answers questions **only** from those documents, in the user's
language, and returns short **source links** that actually back the answer. If
the answer is not in your data, it says so - and returns **no** sources rather
than guessing.

- **Models:** Google Gemini (`gemini-flash-lite` for answers, `gemini-embedding-2`
  for embeddings). No LangChain.
- **Vector store:** ChromaDB, embedded - zero external services to run.
- **Interfaces:** HTTP API (for a website), a CLI, and an optional MCP server.
- **Isolation:** each knowledge base is a separate **"brain"**; they never mix.

> The application layer (the website/UI) is **not** part of this repo. This is
> the backend: it receives a question and returns `{answer, sources, metadata}`.
> How that is presented is up to the consuming application.

---

## 1. Quick start with Docker (plug-and-play)

```bash
cp .env.example .env          # then edit: set GOOGLE_API_KEY and API_ADMIN_KEY
docker compose up -d --build  # starts the API on http://localhost:8100
curl http://localhost:8100/health
```

That is the whole server. Brains persist in the `raggem-data` volume across
restarts. Without compose:

```bash
docker build -t raggem .
docker run -p 8100:8100 --env-file .env -v raggem-data:/app/data raggem
```

## 2. Quick start without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install .                 # or: pip install -e '.[dev]' for development
raggem serve                  # API on http://localhost:8100
```

---

## 3. Configuration (`.env`)

| Variable | Required | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | **yes** | Gemini API key (embeddings + answers) |
| `API_ADMIN_KEY` | **yes** | full access (upload, delete, rebuild, query). The server refuses to start without it. |
| `API_PUBLIC_KEY` | no | query-only key, safe(r) to expose to a website |
| `MODEL_NAME` | no | answer model (default `models/gemini-flash-lite-latest`) |
| `SITE_NAME` | no | the name the assistant identifies as |
| `CORS_ORIGINS` | no | comma-separated allowlist (default `*`; set to your domain in production) |

All keys are read from the environment - none are baked into the image.

---

## 4. Adding your knowledge

A **brain** is an isolated knowledge base. Pick any name (`a-z 0-9 _ -`), e.g.
`handbook`. Files added to it are embedded immediately. Supported file types:
**`.pdf`, `.txt`, `.md`, `.json`**.

### Via the HTTP API (works against the running container)

```bash
# one file
curl -X POST http://localhost:8100/api/v1/brains/handbook/files \
  -H "X-API-Key: $API_ADMIN_KEY" \
  -F "file=@./manual.pdf"

# several files / a whole folder
for f in ./docs/*; do
  curl -X POST http://localhost:8100/api/v1/brains/handbook/files \
    -H "X-API-Key: $API_ADMIN_KEY" -F "file=@$f"
done
```

### Via the CLI (local install)

```bash
raggem upload handbook ./manual.pdf                  # one file
raggem upload handbook ./a.md ./b.txt ./catalog.json # several at once
raggem upload handbook ./docs/*.md                   # a folder via shell glob
raggem list handbook                                 # what's in the brain
raggem rebuild handbook                              # re-embed everything
```

### Structured data (JSON)

A JSON **array** becomes **one embedding per element** automatically (no
chunking) - ideal for catalogs (products, tracks, articles...). Put a `url` and a
`title` field on each record and they come back as the citation link + label.

```json
[
  { "title": "Winter Tire X1", "url": "https://shop.example/x1",
    "price_eur": 89.9, "season": "winter", "description": "Studless winter tire ..." }
]
```

---

## 5. Asking questions

### HTTP API

```bash
curl -X POST http://localhost:8100/api/v1/brains/handbook/query \
  -H "X-API-Key: $API_ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"question": "Which winter tires do you have under 100 euros?"}'
```

Response - the answer plus only the sources that actually support it:

```json
{
  "answer": "We have the Winter Tire X1 for 89.90 EUR.",
  "sources": [
    { "source_id": 1, "title": "Winter Tire X1",
      "url": "https://shop.example/x1", "distance": 0.41 }
  ],
  "metadata": { "grounded": true, "retrieved_chunks": 6, "cited_sources": 1 }
}
```

If nothing in the brain supports an answer, `answer` is a polite "I don't have
that information" and `sources` is `[]` - never a list of unrelated guesses.

### CLI

```bash
raggem query handbook "Which winter tires do you have under 100 euros?" --show-sources
```

---

## 6. HTTP API reference

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | none | liveness |
| `POST` | `/api/v1/brains/{brain}/query` | public or admin | ask a question |
| `GET` | `/api/v1/brains` | admin | list brains |
| `GET` | `/api/v1/brains/{brain}/files` | admin | list files in a brain |
| `POST` | `/api/v1/brains/{brain}/files` | admin | upload + index a file |
| `DELETE` | `/api/v1/brains/{brain}/files/{name}` | admin | remove a file |
| `POST` | `/api/v1/brains/{brain}/rebuild` | admin | re-embed everything |
| `DELETE` | `/api/v1/brains/{brain}` | admin | erase a brain |

Interactive docs at `http://localhost:8100/docs` while the server runs.

---

## 7. How it works

`question -> retrieve (top-k vectors) -> grounding gate -> generate (grounded
answer) -> re-challenge (keep only sources that actually support the answer)`.

Two safety properties:
1. **Grounding gate** - if the closest match is too far, it refuses before the
   LLM even runs. No hallucination.
2. **Re-challenge** - a second pass re-reads the answer against the retrieved
   sources and keeps only the ones that genuinely support it. This is why a
   "not found" answer carries zero sources.

---

## 8. Security & abuse (deployment notes)

The server ships with: two-tier API-key auth, strict input-size limits, prompt-
injection-hardened prompts, and a CORS allowlist. The public key is query-only.

A stolen public key being `curl`-spammed from rotating proxies is an
**operations/edge concern**, handled where you terminate traffic - not inside
this service. For production:

- Put RagGem **behind a reverse proxy / API gateway** (nginx, Cloudflare, an API
  gateway) and enforce **rate limiting + bot protection** there.
- Set `CORS_ORIGINS` to your exact domain (not `*`).
- Treat `API_PUBLIC_KEY` as rotatable; for per-user limits, proxy requests
  through your own backend instead of calling RagGem directly from the browser.

---

## 9. Project layout

```
src/raggem/         the package (config, cli, server, prompts, security, core/)
tests/              test suite
Dockerfile          single-image server
docker-compose.yml  plug-and-play run
.env.example        configuration template
```

License: MIT (see `LICENSE`).
