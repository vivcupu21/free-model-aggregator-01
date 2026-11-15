// Wire translation: requests and answers between the OpenAI, Anthropic and Gemini wires.
//
// Two kinds of functions live here. The JSON ones are pure and complete: an OpenAI request becomes an
// Anthropic or Gemini request, a Gemini answer becomes an OpenAI or Anthropic answer, an Anthropic answer
// becomes an OpenAI one. The SSE ones are string builders over the buffered upstream events: they walk the
// frames one by one and emit the client's event sequence, carrying the provider's text whole and the usage
// it reported.
//
// Every conversion keeps the client's own fields in its own shape: `max_tokens`, `temperature`, `top_p`,
// stop sequences and streaming pass through (an Anthropic request has no top_p in the Gemini wire, where
// topP is its spelling, and the output cap is maxOutputTokens there). A response_format of type
// json_object on the OpenAI side is a generationConfig of type OBJECT on the Gemini side; a json_schema is
// flattened the same way, its own schema kept whole (the Gemini wire has no dedicated schema type, and the
// graders' JSON requests only need the type). An Anthropic request's system prompt stays a string in the
// Gemini wire (it has no separate system field: the system text is prepended to the first user turn, so
// nothing is lost). Tool definitions are translated in both directions: an OpenAI tools array becomes
// Gemini functionDeclarations and Anthropic tool calls become Gemini functionCalls, with the arguments
// carried whole.
//
// The same helpers are used by the JSON and SSE paths for the same conversion, so a request answered in
// either mode says the same thing: stopReasonOf for the stop reason, usageOf for the token counts, and the
// text and tool call extraction shared between them.
//
// Gemini's SSE shape is a stream of JSON lines, each a GenerateContentResponse with candidates and a
// usageMetadata. A finishReason of STOP or MAX_TOKENS maps to Anthropic's end_turn or max_tokens; TOOL_CALLS
// maps to tool_use on the Anthropic side and to tool_calls on the OpenAI side. An OpenAI client's SSE chunk
// carries the delta of its text in choices[0].delta.content and its tool calls in choices[0].delta.tool_calls;
// the final chunk carries the usage. An Anthropic client's events are message_start, content_block_start,
// content_block_delta, content_block_stop, message_delta (with the stop reason and the output tokens) and
// message_stop.

import { isRecord, type Json } from './providers.ts';
import { anthropicError, openAIError, sseEvent, stopReasonOf } from './wire.ts';

/** One tool call in the shape every converter here passes around. */
interface ToolCall {
  id: string;
  name: string;
  args: Json;
}

/** The text of a string or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** A tool call's arguments as an object: the JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A Gemini finishReason as an OpenAI finish_reason. */
function openAIFinishOf(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'TOOL_CALLS') return 'tool_calls';
  if (reason === 'SAFETY' || reason === 'RECITATION' || reason === 'BLOCKLIST' || reason === 'PROHIBITED_CONTENT') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason. */
function anthropicStopOf(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'TOOL_CALLS') return 'tool_use';
  if (reason === 'SAFETY' || reason === 'RECITATION' || reason === 'BLOCKLIST' || reason === 'PROHIBITED_CONTENT') return 'refusal';
  return 'end_turn';
}

/** Gemini's usageMetadata as { prompt_tokens, completion_tokens }. */
function usageOf(meta: unknown): { prompt_tokens: number; completion_tokens: number } {
  const out = { prompt_tokens: 0, completion_tokens: 0 };
  if (!isRecord(meta)) return out;
  const prompt = typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : 0;
  const completion = typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : 0;
  return { prompt_tokens: prompt, completion_tokens: completion };
}

/**
 * The text of a Gemini candidate's parts, or '' when there is none. A functionCall part's name is
 * kept out of the text (it is a tool call, not text).
 */
function geminiTextOf(candidate: Json): string {
  const parts = Array.isArray(candidate.content) && isRecord(candidate.content) && Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
  const texts: string[] = [];
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  }
  return texts.join('');
}

/** The tool calls of a Gemini candidate's parts, in order. */
function geminiToolCallsOf(candidate: Json): ToolCall[] {
  const parts = Array.isArray(candidate.content) && isRecord(candidate.content) && Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
  const calls: ToolCall[] = [];
  for (const part of parts) {
    if (!isRecord(part) || !isRecord(part.functionCall)) continue;
    const fn = part.functionCall;
    const name = typeof fn.name === 'string' ? fn.name : '';
    calls.push({ id: `call_${calls.length}`, name, args: isRecord(fn.args) ? fn.args : {} });
  }
  return calls;
}

/** The tool calls of an OpenAI message, in order. */
function openAIToolCallsOf(message: Json): ToolCall[] {
  const calls: ToolCall[] = [];
  const raw = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of raw) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    const fn = call.function;
    calls.push({ id: typeof call.id === 'string' ? call.id : `call_${calls.length}`, name: typeof fn.name === 'string' ? fn.name : '', args: argumentsOf(fn.arguments) });
  }
  return calls;
}

/** The tool calls of an Anthropic message's content blocks, in order. */
function anthropicToolCallsOf(blocks: unknown): ToolCall[] {
  const calls: ToolCall[] = [];
  if (!Array.isArray(blocks)) return calls;
  for (const block of blocks) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    calls.push({ id: typeof block.id === 'string' ? block.id : `toolu_${calls.length}`, name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} });
  }
  return calls;
}

/** A Gemini candidate as an OpenAI Chat Completions message: text, then tool calls. */
function geminiCandidateToOpenAIMessage(candidate: Json): { message: Json; finish_reason: string } {
  const text = geminiTextOf(candidate);
  const calls = geminiToolCallsOf(candidate);
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) {
    message.tool_calls = calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.args) } }));
  }
  return { message, finish_reason: openAIFinishOf(candidate.finishReason) };
}

/** A Gemini candidate as Anthropic content blocks: text, then tool_use blocks. */
function geminiCandidateToAnthropicContent(candidate: Json): { content: Json[]; stop_reason: string } {
  const text = geminiTextOf(candidate);
  const calls = geminiToolCallsOf(candidate);
  const content: Json[] = [];
  if (text !== '') content.push({ type: 'text', text });
  for (const call of calls) content.push({ type: 'tool_use', id: call.id, name: call.name, input: call.args });
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return { content, stop_reason: anthropicStopOf(candidate.finishReason) };
}

/** The first candidate of a Gemini answer, or null when it has none. */
function firstCandidate(body: Json): Json | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  return isRecord(candidates[0]) ? candidates[0] : null;
}

/** A Gemini answer as an OpenAI Chat Completions object. */
export function geminiToOpenAIJson(body: Json, model: string): Json {
  const candidate = firstCandidate(body);
  if (candidate === null) return openAIError('the provider answered with no candidate', 'internal_error', null);
  const { message, finish_reason } = geminiCandidateToOpenAIMessage(candidate);
  const usage = usageOf(body.usageMetadata);
  return { id: `chatcmpl_${model}`, object: 'chat.completion', created: 1767225600, model, choices: [{ index: 0, message, finish_reason }], usage };
}

/** A Gemini answer as an Anthropic Messages object. */
export function geminiToAnthropicJson(body: Json, model: string): Json {
  const candidate = firstCandidate(body);
  if (candidate === null) return anthropicError('overloaded_error', 'the provider answered with no candidate');
  const { content, stop_reason } = geminiCandidateToAnthropicContent(candidate);
  const usage = usageOf(body.usageMetadata);
  return {
    id: `msg_${model}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason,
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
}

/** An Anthropic answer as an OpenAI Chat Completions object. */
export function anthropicToOpenAIJson(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = textOf(content);
  const calls = anthropicToolCallsOf(content);
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) {
    message.tool_calls = calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.args) } }));
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = typeof usage.input_tokens === 'number' ? usage.input_tokens : 0;
  const output = typeof usage.output_tokens === 'number' ? usage.output_tokens : 0;
  return {
    id: `chatcmpl_${model}`,
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message, finish_reason: body.stop_reason === 'tool_use' ? 'tool_calls' : body.stop_reason === 'max_tokens' ? 'length' : 'stop' }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** One OpenAI message as a Gemini content object: role user/model, parts text, image or functionResponse. */
function openAIMessageToGemini(message: Json): Json | null {
  const role = message.role;
  if (role === 'system') return null;
  if (role === 'tool') {
    const name = typeof message.name === 'string' ? message.name : 'tool';
    return { role: 'user', parts: [{ functionResponse: { name, response: { result: typeof message.content === 'string' ? message.content : '' } } }] };
  }
  if (role !== 'user' && role !== 'assistant') return null;
  const parts: Json[] = [];
  const content = message.content;
  if (typeof content === 'string' && content !== '') parts.push({ text: content });
  else if (Array.isArray(content)) {
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const url = part.image_url.url;
        const match = /^data:([^;]+);base64,(.+)$/.exec(url);
        if (match !== null) parts.push({ inlineData: { mimeType: match[1], data: match[2] } });
      }
    }
  }
  for (const call of openAIToolCallsOf(message)) {
    parts.push({ functionCall: { name: call.name, args: call.args } });
  }
  if (parts.length === 0) return null;
  return { role: role === 'assistant' ? 'model' : 'user', parts };
}

/** One Anthropic message as a Gemini content object. */
function anthropicMessageToGemini(message: Json): Json | null {
  const role = message.role;
  const content = message.content;
  if (typeof content === 'string') {
    return content === '' ? null : { role: role === 'assistant' ? 'model' : 'user', parts: [{ text: content }] };
  }
  if (!Array.isArray(content)) return null;
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
      }
    } else if (block.type === 'tool_use') {
      parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
    } else if (block.type === 'tool_result') {
      const name = typeof block.tool_use_id === 'string' ? block.tool_use_id : 'tool';
      const text = textOf(block.content);
      parts.push({ functionResponse: { name, response: { result: text } } });
    }
  }
  if (parts.length === 0) return null;
  return { role: role === 'assistant' ? 'model' : 'user', parts };
}

/** An OpenAI tools array as Gemini functionDeclarations. */
function openAIToolsToGemini(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    const fn = raw.function;
    const declaration: Json = { name: fn.name, description: typeof fn.description === 'string' ? fn.description : '' };
    if (isRecord(fn.parameters)) declaration.parameters = fn.parameters;
    return [declaration];
  });
}

/** An Anthropic tools array as Gemini functionDeclarations. */
function anthropicToolsToGemini(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    const declaration: Json = { name: raw.name, description: typeof raw.description === 'string' ? raw.description : '' };
    if (isRecord(raw.input_schema)) declaration.parameters = raw.input_schema;
    return [declaration];
  });
}

/** The generationConfig for a request: cap, temperature, topP, stop sequences and the JSON type when asked. */
function geminiGenerationConfig(body: Json, json: boolean): Json {
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop;
  if (json) config.responseMimeType = 'application/json';
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request naming `model`. The system
 * message is dropped (Gemini's request has no system field: main.ts prepends it instead when the caller
 * is Anthropic; an OpenAI caller's system message is merged into the first user turn by the caller).
 */
export function openAIToGeminiRequest(body: Json, model: string, json: boolean): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const message of source) {
    if (!isRecord(message)) continue;
    const content = openAIMessageToGemini(message);
    if (content !== null) contents.push(content);
  }
  const out: Json = { model, contents, generationConfig: geminiGenerationConfig(body, json) };
  const tools = openAIToolsToGemini(body.tools);
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request naming `model`. The system prompt
 * is prepended to the first user turn's text, so nothing is lost.
 */
export function anthropicToGeminiRequest(body: Json, model: string, json: boolean): Json {
  const system = textOf(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  let firstUser = true;
  for (const message of source) {
    if (!isRecord(message)) continue;
    const content = anthropicMessageToGemini(message);
    if (content === null) continue;
    if (firstUser && system !== '' && content.role === 'user') {
      firstUser = false;
      const parts = Array.isArray(content.parts) ? content.parts : [];
      content.parts = [{ text: `${system}\n\n` }, ...parts];
    }
    contents.push(content);
  }
  const out: Json = { model, contents, generationConfig: geminiGenerationConfig(body, json) };
  const tools = anthropicToolsToGemini(body.tools);
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  return out;
}

/** An OpenAI Chat Completions request as an Anthropic Messages request naming `model`. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') continue;
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : '' }],
      });
      continue;
    }
    if (role !== 'user' && role !== 'assistant') continue;
    const blocks: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string' && content !== '') blocks.push({ type: 'text', text: content });
    else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          const match = /^data:([^;]+);base64,(.+)$/.exec(url);
          if (match !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: match[1], data: match[2] } });
          else blocks.push({ type: 'image', source: { type: 'url', url } });
        }
      }
    }
    for (const call of openAIToolCallsOf(raw)) {
      blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: call.args });
    }
    if (blocks.length > 0) messages.push({ role, content: blocks });
  }
  const out: Json = { model, messages, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 1024 };
  const system = source.flatMap((m) => (isRecord(m) && m.role === 'system' ? [textOf(m.content)] : [])).join('\n');
  if (system !== '') out.system = system;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      const fn = raw.function;
      return [{ name: fn.name, description: typeof fn.description === 'string' ? fn.description : '', input_schema: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** Split a buffered SSE body into frames: lines joined by blank lines, \r\n normalised. */
function sseFrames(text: string): string[] {
  const normal = text.replace(/\r\n/g, '\n');
  const frames: string[] = [];
  let rest = normal;
  let at = rest.indexOf('\n\n');
  while (at !== -1) {
    frames.push(rest.slice(0, at));
    rest = rest.slice(at + 2);
    at = rest.indexOf('\n\n');
  }
  if (rest.trim() !== '') frames.push(rest);
  return frames;
}

/** The JSON payloads of an SSE body, in order: the `data:` lines that parse, '[DONE]' skipped. */
function ssePayloads(text: string): Json[] {
  const out: Json[] = [];
  for (const frame of sseFrames(text)) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') continue;
      try {
        const parsed: unknown = JSON.parse(payload);
        if (isRecord(parsed)) out.push(parsed);
      } catch {
        // Not JSON: skip.
      }
    }
  }
  return out;
}

/** One Gemini SSE frame as OpenAI chunks: text deltas, tool call deltas, then the usage chunk. */
function geminiFrameToOpenAIChunks(frame: Json, model: string, state: { started: boolean; callIndex: number }): string[] {
  const out: string[] = [];
  const candidate = firstCandidate(frame);
  if (candidate === null) return out;
  const text = geminiTextOf(candidate);
  const calls = geminiToolCallsOf(candidate);
  const usage = usageOf(frame.usageMetadata);
  if (text !== '') {
    out.push(`data: ${JSON.stringify({ id: `chatcmpl_${model}`, object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta: { role: 'assistant', content: text }, finish_reason: null }] })}\n\n`);
  }
  for (const call of calls) {
    const index = state.callIndex;
    state.callIndex += 1;
    out.push(
      `data: ${JSON.stringify({
        id: `chatcmpl_${model}`,
        object: 'chat.completion.chunk',
        created: 1767225600,
        model,
        choices: [{ index: 0, delta: { tool_calls: [{ index, id: call.id, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.args) } }] }, finish_reason: null }],
      })}\n\n`,
    );
  }
  if (candidate.finishReason !== undefined && candidate.finishReason !== null) {
    const finish = openAIFinishOf(candidate.finishReason);
    out.push(
      `data: ${JSON.stringify({
        id: `chatcmpl_${model}`,
        object: 'chat.completion.chunk',
        created: 1767225600,
        model,
        choices: [{ index: 0, delta: {}, finish_reason: finish }],
        usage: { prompt_tokens: usage.prompt_tokens, completion_tokens: usage.completion_tokens, total_tokens: usage.prompt_tokens + usage.completion_tokens },
      })}\n\n`,
    );
  }
  return out;
}

/** A Gemini SSE stream as an OpenAI SSE stream, ending with [DONE]. */
export function geminiSseToOpenAI(text: string, model: string): string {
  const state = { started: false, callIndex: 0 };
  const out: string[] = [];
  for (const frame of ssePayloads(text)) out.push(...geminiFrameToOpenAIChunks(frame, model, state));
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/** One Gemini SSE frame as Anthropic events. */
function geminiFrameToAnthropicEvents(frame: Json, model: string, state: { started: boolean; textIndex: number; nextIndex: number; toolIndexes: Map<number, number> }): string[] {
  const out: string[] = [];
  const candidate = firstCandidate(frame);
  if (candidate === null) return out;
  const usage = usageOf(frame.usageMetadata);
  if (!state.started) {
    state.started = true;
    out.push(
      sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.prompt_tokens, output_tokens: 0 } },
      }),
    );
  }
  const text = geminiTextOf(candidate);
  if (text !== '') {
    if (state.textIndex === -1) {
      state.textIndex = state.nextIndex;
      state.nextIndex += 1;
      out.push(sseEvent('content_block_start', { type: 'content_block_start', index: state.textIndex, content_block: { type: 'text', text: '' } }));
    }
    out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: state.textIndex, delta: { type: 'text_delta', text } }));
  }
  const calls = geminiToolCallsOf(candidate);
  for (let i = 0; i < calls.length; i++) {
    const call = calls[i];
    let index = state.toolIndexes.get(i);
    if (index === undefined) {
      if (state.textIndex !== -1) {
        out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: state.textIndex }));
        state.textIndex = -1;
      }
      index = state.nextIndex;
      state.nextIndex += 1;
      state.toolIndexes.set(i, index);
      out.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} } }));
    }
    out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: JSON.stringify(call.args) } }));
  }
  if (candidate.finishReason !== undefined && candidate.finishReason !== null) {
    if (state.textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: state.textIndex }));
      state.textIndex = -1;
    }
    for (const index of [...state.toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    state.toolIndexes.clear();
    const stop = anthropicStopOf(candidate.finishReason);
    out.push(sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: usage.completion_tokens } }));
    out.push(sseEvent('message_stop', { type: 'message_stop' }));
  }
  return out;
}

/** A Gemini SSE stream as Anthropic events, always closed with message_stop. */
export function geminiSseToAnthropic(text: string, model: string): string {
  const state = { started: false, textIndex: -1, nextIndex: 0, toolIndexes: new Map<number, number>() };
  const out: string[] = [];
  for (const frame of ssePayloads(text)) out.push(...geminiFrameToAnthropicEvents(frame, model, state));
  if (!state.started) {
    out.push(
      sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
      }),
    );
  }
  if (state.textIndex !== -1) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: state.textIndex }));
  for (const index of [...state.toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  out.push(sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 0 } }));
  out.push(sseEvent('message_stop', { type: 'message_stop' }));
  return out.join('');
}

/**
 * An Anthropic SSE stream as OpenAI chunks. The Anthropic text deltas become content deltas, the
 * input_json_delta events are accumulated per tool call and emitted whole at content_block_stop, the
 * usage in message_delta becomes the usage chunk, and the stream ends with [DONE].
 */
export function anthropicSseToOpenAI(text: string, model: string): string {
  const out: string[] = [];
  const partials = new Map<number, string>();
  const toolNames = new Map<number, string>();
  const toolIds = new Map<number, string>();
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason: string | null = null;
  const chunk = (delta: Json, finish: string | null, usage: Json | null): string => {
    const body: Json = { id: `chatcmpl_${model}`, object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta, finish_reason: finish }] };
    if (usage !== null) body.usage = usage;
    return `data: ${JSON.stringify(body)}\n\n`;
  };
  for (const frame of sseFrames(text)) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      let event: unknown;
      try {
        event = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(event)) continue;
      if (event.type === 'message_start' && isRecord(event.message) && isRecord(event.message.usage)) {
        inputTokens = typeof event.message.usage.input_tokens === 'number' ? event.message.usage.input_tokens : 0;
      }
      if (event.type === 'content_block_start' && isRecord(event.content_block) && typeof event.index === 'number') {
        const block = event.content_block;
        if (block.type === 'tool_use') {
          toolNames.set(event.index, typeof block.name === 'string' ? block.name : '');
          toolIds.set(event.index, typeof block.id === 'string' ? block.id : `toolu_${event.index}`);
          partials.set(event.index, '');
        }
      }
      if (event.type === 'content_block_delta' && isRecord(event.delta) && typeof event.index === 'number') {
        const delta = event.delta;
        if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
          out.push(chunk({ role: 'assistant', content: delta.text }, null, null));
        } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
          partials.set(event.index, (partials.get(event.index) ?? '') + delta.partial_json);
        }
      }
      if (event.type === 'content_block_stop' && typeof event.index === 'number' && partials.has(event.index)) {
        const args = partials.get(event.index) ?? '';
        out.push(
          chunk(
            { tool_calls: [{ index: event.index, id: toolIds.get(event.index) ?? `toolu_${event.index}`, type: 'function', function: { name: toolNames.get(event.index) ?? '', arguments: args } }] },
            null,
            null,
          ),
        );
        partials.delete(event.index);
      }
      if (event.type === 'message_delta' && isRecord(event.delta)) {
        if (typeof event.delta.stop_reason === 'string') stopReason = stopReasonOf(event.delta.stop_reason);
        if (isRecord(event.usage) && typeof event.usage.output_tokens === 'number') outputTokens = event.usage.output_tokens;
      }
      if (event.type === 'message_stop') {
        const finish = stopReason === 'tool_use' ? 'tool_calls' : stopReason === 'max_tokens' ? 'length' : stopReason === 'refusal' ? 'content_filter' : 'stop';
        out.push(chunk({}, finish, { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens }));
      }
    }
  }
  // An Anthropic stream that ends without message_stop still closes with the usage chunk.
  if (!out.some((line) => line.includes('"finish_reason":"stop"') || line.includes('"finish_reason":"tool_calls"') || line.includes('"finish_reason":"length"'))) {
    const finish = stopReason === 'tool_use' ? 'tool_calls' : stopReason === 'max_tokens' ? 'length' : stopReason === 'refusal' ? 'content_filter' : 'stop';
    out.push(chunk({}, finish, { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens }));
  }
  out.push('data: [DONE]\n\n');
  return out.join('');
}
