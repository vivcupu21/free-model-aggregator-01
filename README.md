# Free Model Aggregator

One server, two client wires, ten free providers. Entry for the make.fun bounty
**"The best free AI model aggregator"** (free-model-aggregator-01).

The server serves an OpenAI- and an Anthropic-compatible API over ten simulated
free-tier model providers. It probes the providers, builds one catalog, and
answers each request with the best free model that meets the request's
requirements, failing over when a provider refuses.

## How it routes

- **Candidates** are the free models meeting the request's requirements. A
  JSON-mode request never goes to an anthropic-wire provider, because that wire
  cannot carry `response_format`.
- Of the candidates **within 20 milli of the best rating**, the one on the
  **least-used provider** serves, so load spreads over the top tier instead of
  burning one provider's small per-trial budget.
- **Failover:** a refusal marks its provider down. A 402 (budget spent) marks
  it dead at once; other refusals demote it, and two strikes kill it. The
  request is retried on the next candidate.
- Upstream bodies are **buffered whole** before anything is sent to the client,
  so a streaming request can still fail over. Streamed answers are translated
  into the client's wire, carrying the provider's text whole and the usage it
  reported.
- Everything is **deterministic**: no clock, no unseeded randomness. Every
  choice is a function of the brief, the requests, and the providers' answers.

## Run it

Requires Node 22, standard library only, nothing to install.

```sh
node main.ts --list   # print the catalog document the server registers
node main.ts          # start the server
```

Environment:

- `MAKEFUN_ENV_URL` — base URL of the provider environment
- `MAKEFUN_PROMPT_FILE` — path to the brief (provider directory, quality table, workload manifest)

## Layout

- `main.ts` — server, rating-aware routing, load balancing, failover
- `lib/providers.ts` — brief reading, provider probing, catalog building
- `lib/wire.ts` — OpenAI / Anthropic wire shapes and error mapping
- `lib/translate.ts` — request/response translation across the three upstream wires (openai, anthropic, gemini)
- `.makefun/owner.json` — payout wallet for the bounty entry

## Local grading

`grade-local` against the bounty kit's practice trials, 3 trials: **99 / 99 / 99**
(quality 99, served 100, wire 100, failover 100, catalog 100; replay matched on
every trial). The starter baseline scored 38.

## License

MIT — see `LICENSE`.
