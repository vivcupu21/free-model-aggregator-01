// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The brief gives a provider directory, a sealed quality table and a workload
// manifest. This server probes the providers, builds one catalog, registers with
// the env, and answers the client's twenty requests with the best free model that
// meets each request's requirements, on whichever of the three upstream wires
// (openai, anthropic, gemini) that provider speaks, failing over when a provider
// refuses. Everything is deterministic: no clock, no unseeded randomness; every
// choice is a function of the brief, the requests and the providers' answers.
//
// Routing: candidates are the free models meeting the request's requirements (a
// JSON-mode request never goes to an anthropic-wire provider: that wire cannot
// carry response_format, so the provider would echo it absent). Of the candidates
// within 20 milli of the best rating, the one on the least-used provider serves,
// so the load spreads over the top tier instead of burning one provider's small
// per-trial budget. A refusal marks its provider down (a 402, budget spent, marks
// it dead at once); the request is retried on the next candidate. Upstream bodies
// are buffered whole before anything is sent to the client, so a streaming
// request can still fail over. Streamed answers are translated into the client's
// wire, carrying the provider's text whole and the usage it reported.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicSseToOpenAI,
  anthropicToGeminiRequest,
  anthropicToOpenAIJson,
  geminiSseToAnthropic,
  geminiSseToOpenAI,
  geminiToAnthropicJson,
  geminiToOpenAIJson,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 15000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** How far below the best rating a candidate may sit and still be preferred when its provider is least used. */
const RATING_BAND_MILLI = 20;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/** Per-provider call state: how many calls it has taken, and whether it has refused. */
interface ProviderState {
  calls: number;
  refused: boolean;
  dead: boolean;
}
const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, refused: false, dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve one request, best rating first. A JSON-mode
 * request never goes to an anthropic-wire provider: that wire cannot carry
 * response_format, so the provider would echo it absent and the echo check fails.
 */
function candidatesFor(requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (requires.json === true && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : 1) || byId(a.model, b.model));
  return out;
}

/** The effective load of a provider for ordering: its calls, with refusals pushed far down and dead ones out. */
function loadOf(providerId: string): number {
  const state = stateOf(providerId);
  if (state.dead) return 1_000_000;
  return state.calls + (state.refused ? 100 : 0);
}

/**
 * The candidates in serving order for one request: of those within the rating
 * band of the best, the least-loaded provider first (ties by rating, then by
 * provider and model id, so the order is deterministic); the rest follow by
 * rating. Dead providers sit last and are only used when nothing else is left.
 */
function orderedCandidates(requires: Requirement): Candidate[] {
  const all = candidatesFor(requires);
  if (all.length === 0) return all;
  const best = all[0].rating;
  const near = all.filter((c) => c.rating >= best - RATING_BAND_MILLI);
  const rest = all.filter((c) => c.rating < best - RATING_BAND_MILLI);
  near.sort((a, b) => loadOf(a.probe.provider.id) - loadOf(b.probe.provider.id) || b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : 1) || byId(a.model, b.model));
  return [...near, ...rest];
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamResult {
  ok: boolean;
  status: number;
  /** The upstream body, whole, when the call answered. */
  text: string;
}

/** Build the upstream request body for one candidate in its own wire. */
function upstreamBody(clientWire: 'openai' | 'anthropic', candidate: Candidate, body: Json, requires: Requirement): Json {
  const wire = candidate.probe.provider.wire;
  if (wire === 'openai') {
    if (clientWire === 'openai') {
      const out: Json = { ...body, model: candidate.model.id };
      if (body.stream === true) out.stream_options = { include_usage: true };
      return out;
    }
    const out = anthropicToOpenAI(body, candidate.model.id);
    if (requires.json === true) out.response_format = { type: 'json_object' };
    return out;
  }
  if (wire === 'anthropic') {
    if (clientWire === 'anthropic') return { ...body, model: candidate.model.id };
    return openAIToAnthropicRequest(body, candidate.model.id);
  }
  // Gemini wire.
  if (clientWire === 'openai') return openAIToGeminiRequest(body, candidate.model.id, requires.json === true);
  return anthropicToGeminiRequest(body, candidate.model.id, requires.json === true);
}

/** The upstream URL for one candidate's inference call. */
function upstreamUrl(candidate: Candidate, stream: boolean): string {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  const wire = candidate.probe.provider.wire;
  if (wire === 'openai') return `${base}/v1/chat/completions`;
  if (wire === 'anthropic') return `${base}/v1/messages`;
  return stream
    ? `${base}/v1beta/models/${candidate.model.id}:streamGenerateContent?alt=sse`
    : `${base}/v1beta/models/${candidate.model.id}:generateContent`;
}

/** One call to one candidate's provider, buffered whole. */
async function callCandidate(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json, requires: Requirement): Promise<UpstreamResult> {
  const state = stateOf(candidate.probe.provider.id);
  state.calls += 1;
  const stream = body.stream === true;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(candidate.probe.provider.auth, candidate.probe.provider.key),
  };
  if (candidate.probe.provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream) headers.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(upstreamUrl(candidate, stream), {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody(clientWire, candidate, body, requires)),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, status: 0, text: (err as Error).message };
  }
  const text = await res.text();
  return { ok: res.ok, status: res.status, text };
}

/** Record a refusal: a 402 means the budget is spent, so the provider is dead; other refusals mark it down, twice dead. */
function noteRefusal(candidate: Candidate, status: number): void {
  const state = stateOf(candidate.probe.provider.id);
  if (status === 402) {
    state.dead = true;
    return;
  }
  if (state.refused) state.dead = true;
  else state.refused = true;
}

/**
 * The provider's answer translated into the client's wire. `stream` is what the
 * client asked for; the upstream body is already buffered whole.
 */
async function translateAnswer(
  clientWire: 'openai' | 'anthropic',
  candidate: Candidate,
  stream: boolean,
  upstreamText: string,
): Promise<{ json: Json | null; sse: string | null }> {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (!stream) {
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(upstreamText);
    } catch {
      return { json: null, sse: null };
    }
    if (!isRecord(parsed)) return { json: null, sse: null };
    if (wire === 'openai') {
      return { json: clientWire === 'openai' ? parsed : openAIToAnthropic(parsed, model), sse: null };
    }
    if (wire === 'anthropic') {
      return { json: clientWire === 'anthropic' ? parsed : anthropicToOpenAIJson(parsed, model), sse: null };
    }
    return { json: clientWire === 'openai' ? geminiToOpenAIJson(parsed, model) : geminiToAnthropicJson(parsed, model), sse: null };
  }
  // Streamed: translate the buffered upstream events into the client's wire.
  if (wire === 'openai') {
    if (clientWire === 'openai') return { json: null, sse: upstreamText };
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(upstreamText));
        controller.close();
      },
    });
    let out = '';
    for await (const event of anthropicEventsFromOpenAI(stream, model)) out += event;
    return { json: null, sse: out };
  }
  if (wire === 'anthropic') {
    if (clientWire === 'anthropic') return { json: null, sse: upstreamText };
    return { json: null, sse: anthropicSseToOpenAI(upstreamText, model) };
  }
  return { json: null, sse: clientWire === 'openai' ? geminiSseToOpenAI(upstreamText, model) : geminiSseToAnthropic(upstreamText, model) };
}

/**
 * Serve one client request: try the ordered candidates in turn until one answers,
 * then translate its answer into the client's wire. Returns false when no
 * candidate answered, so the handler can send its error.
 */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, requires: Requirement, res: ServerResponse): Promise<boolean> {
  const ordered = orderedCandidates(requires);
  const stream = body.stream === true;
  const tried = new Set<string>();
  for (const candidate of ordered) {
    const key = `${candidate.probe.provider.id}/${candidate.model.id}`;
    if (tried.has(key)) continue;
    tried.add(key);
    const result = await callCandidate(candidate, clientWire, body, requires);
    if (!result.ok) {
      noteRefusal(candidate, result.status);
      log(`refused ${key} HTTP ${result.status}, failing over`);
      continue;
    }
    const answer = await translateAnswer(clientWire, candidate, stream, result.text);
    if (answer.json !== null) {
      sendJson(res, 200, answer.json);
      return true;
    }
    if (answer.sse !== null) {
      sendSse(res, answer.sse);
      return true;
    }
    // The provider answered 200 with a body this server could not read: treat it
    // as a refusal and try the next candidate.
    noteRefusal(candidate, result.status);
    log(`unreadable answer from ${key}, failing over`);
  }
  return false;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by any provider wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  if (orderedCandidates(requires).length === 0) {
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  }
  const served = await serve('openai', body, requires, res);
  if (!served) return sendError(res, 'openai', 503, 'no_free_model', 'no provider on the directory answered this request');
}

/** POST /v1/messages: an Anthropic-shaped client, answered by any provider wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  if (orderedCandidates(requires).length === 0) {
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  }
  const served = await serve('anthropic', body, requires, res);
  if (!served) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no provider on the directory answered this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
