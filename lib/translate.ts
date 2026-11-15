// Request/response translation between the client wires (openai, anthropic) and the
// upstream wires (openai, anthropic, gemini).
//
// Every function here is a pure function of its arguments: no clock, no randomness,
// no I/O. Text is always carried whole: the provider's ANSWER text must reach the
// client byte for byte, and every field the client sent (system, user text, images,
// tools, max_tokens, stop sequences, JSON mode) must reach the provider exactly, so
// the provider's echo of the request matches the client's echo of it.

import { isRecord, type Json } from './providers.ts';

/** Fixed epoch second for every translated completion's `created`. */
const FIXED_CREATED = 1767225600;

/** Split a `data:<mime>;base64,<payload>` URL, or null when it is not one. */
export function parseDataUrl(url: string): { mime: string; data: string } | null {
  const m = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mime: m[1] ?? 'image/png', data: m[2] };
}

/** The text of a string or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A JSON string's parsed object, or {} when it is not one. */
function argsObject(value: unknown): Json {
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

/** An OpenAI content part as an Anthropic content block, or null when it has no carry. */
function openAIPartToAnthropic(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = parseDataUrl(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data.mime, data: data.data } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`.
 * Tool messages become tool_result blocks inside the user message that follows the
 * assistant turn that called them; system messages become the system prompt.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  /** tool_result blocks waiting for the user message that carries them. */
  let pendingToolResults: Json[] = [];
  const flushToolResults = (): void => {
    if (pendingToolResults.length === 0) return;
    messages.push({ role: 'user', content: pendingToolResults });
    pendingToolResults = [];
  };
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      pendingToolResults.push({
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
      });
      continue;
    }
    if (raw.role === 'assistant') {
      flushToolResults();
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argsObject(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    // A user message: tool results pending from an earlier turn go first, in their own message.
    flushToolResults();
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (Array.isArray(content)) {
      const blocks = content.flatMap((p) => (isRecord(p) ? openAIPartToAnthropic(p) ?? [] : []));
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }
  flushToolResults();

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stop = body.stop;
  if (typeof stop === 'string') out.stop_sequences = [stop];
  else if (Array.isArray(stop) && stop.length > 0) out.stop_sequences = stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI content part as a Gemini part, or null when it has no carry. */
function openAIPartToGemini(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = parseDataUrl(url);
    if (data !== null) return { inlineData: { mimeType: data.mime, data: data.data } };
    return { fileData: { mimeType: 'image/png', fileUri: url } };
  }
  return null;
}

/** OpenAI tools as Gemini function declarations. */
function openAIToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const declarations = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : [];
}

/** OpenAI tool_choice as a Gemini toolConfig, or null when it names nothing. */
function openAIToolChoiceToGemini(body: Json): Json | null {
  const choice = body.tool_choice;
  if (choice === 'auto') return { functionCallingConfig: { mode: 'AUTO' } };
  if (choice === 'required') return { functionCallingConfig: { mode: 'ANY' } };
  if (choice === 'none') return { functionCallingConfig: { mode: 'NONE' } };
  if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    return { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.function.name] } };
  }
  return null;
}

/** The generationConfig for a Gemini request built from an OpenAI or Anthropic body. */
function geminiGenerationConfig(maxTokens: number | undefined, temperature: unknown, topP: unknown, stops: string[], jsonMode: boolean): Json {
  const config: Json = {};
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof temperature === 'number') config.temperature = temperature;
  if (typeof topP === 'number') config.topP = topP;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request. `jsonMode`
 * is true when the request must be answered as JSON (the manifest's requirement):
 * Gemini carries it as responseMimeType, which the provider echoes.
 */
export function openAIToGeminiRequest(body: Json, model: string, jsonMode: boolean): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  /** functionResponse parts waiting for the user content that carries them. */
  let pendingResponses: Json[] = [];
  const flushResponses = (): void => {
    if (pendingResponses.length === 0) return;
    contents.push({ role: 'user', parts: pendingResponses });
    pendingResponses = [];
  };
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      pendingResponses.push({ functionResponse: { name: '', response: { content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) } } });
      continue;
    }
    if (raw.role === 'assistant') {
      flushResponses();
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argsObject(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    flushResponses();
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: 'user', parts: [{ text: content }] });
      continue;
    }
    if (Array.isArray(content)) {
      const parts = content.flatMap((p) => (isRecord(p) ? openAIPartToGemini(p) ?? [] : []));
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  }
  flushResponses();

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const tools = openAIToolsToGemini(body);
  if (tools.length > 0) out.tools = tools;
  const toolConfig = openAIToolChoiceToGemini(body);
  if (toolConfig !== null) out.toolConfig = toolConfig;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  const stop = body.stop;
  const stops = typeof stop === 'string' ? [stop] : Array.isArray(stop) ? stop.filter((s): s is string => typeof s === 'string') : [];
  const config = geminiGenerationConfig(maxTokens, body.temperature, body.top_p, stops, jsonMode);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  void model;
  return out;
}

/** One Anthropic content block as a Gemini part, or null when it has no carry. */
function anthropicBlockToGemini(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { mimeType: 'image/png', fileUri: source.url } };
  }
  return null;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request. `jsonMode` is true
 * when the manifest says the request must be answered as JSON.
 */
export function anthropicToGeminiRequest(body: Json, model: string, jsonMode: boolean): Json {
  const contents: Json[] = [];
  const systemText = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role, parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_use') {
        parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        continue;
      }
      if (block.type === 'tool_result') {
        parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { content: textOf(block.content) } } });
        continue;
      }
      const part = anthropicBlockToGemini(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) contents.push({ role, parts });
  }

  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
    else if (choice.type === 'any') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
    else if (choice.type === 'none') out.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.name] } };
  }
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [];
  const config = geminiGenerationConfig(typeof body.max_tokens === 'number' ? body.max_tokens : undefined, body.temperature, body.top_p, stops, jsonMode);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  void model;
  return out;
}

/** The text of a Gemini candidate's parts, and its function calls, if any. */
function geminiParts(candidate: Json): { text: string; calls: { name: string; args: Json }[] } {
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  const calls: { name: string; args: Json }[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') texts.push(part.text);
    if (isRecord(part.functionCall)) {
      calls.push({ name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '', args: isRecord(part.functionCall.args) ? part.functionCall.args : {} });
    }
  }
  return { text: texts.join(''), calls };
}

/** Gemini usageMetadata as token counts. */
function geminiUsage(resp: Json): { input: number; output: number } {
  const meta = isRecord(resp.usageMetadata) ? resp.usageMetadata : {};
  const input = typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : 0;
  const output = typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : 0;
  return { input, output };
}

/** A Gemini finishReason as an OpenAI finish_reason. */
function geminiStopToOpenAI(reason: unknown, hasCalls: boolean): string {
  if (hasCalls) return 'tool_calls';
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason. */
function geminiStopToAnthropic(reason: unknown, hasCalls: boolean): string {
  if (hasCalls) return 'tool_use';
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object. */
export function geminiToOpenAIJson(resp: Json, model: string): Json {
  const candidates = Array.isArray(resp.candidates) ? resp.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const { text, calls } = geminiParts(first);
  const reason = first.finishReason;
  const usage = geminiUsage(resp);
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) {
    message.tool_calls = calls.map((call, i) => ({ id: `call_${i}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.args) } }));
  }
  return {
    id: `chatcmpl-${model}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: geminiStopToOpenAI(reason, calls.length > 0), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropicJson(resp: Json, model: string): Json {
  const candidates = Array.isArray(resp.candidates) ? resp.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const { text, calls } = geminiParts(first);
  const reason = first.finishReason;
  const usage = geminiUsage(resp);
  const content: Json[] = [];
  if (text !== '') content.push({ type: 'text', text });
  for (const call of calls) content.push({ type: 'tool_use', id: `toolu_${content.length}`, name: call.name, input: call.args });
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: `msg_${model}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: geminiStopToAnthropic(reason, calls.length > 0),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object. */
export function anthropicToOpenAIJson(resp: Json, model: string): Json {
  const content = Array.isArray(resp.content) ? resp.content : [];
  const text = textOf(content);
  const calls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    calls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) message.tool_calls = calls;
  const usage = isRecord(resp.usage) ? resp.usage : {};
  const input = typeof usage.input_tokens === 'number' ? usage.input_tokens : 0;
  const output = typeof usage.output_tokens === 'number' ? usage.output_tokens : 0;
  const stop = typeof resp.stop_reason === 'string' ? resp.stop_reason : 'end_turn';
  const finish = stop === 'max_tokens' ? 'length' : stop === 'tool_use' ? 'tool_calls' : stop === 'refusal' ? 'content_filter' : 'stop';
  return {
    id: `chatcmpl-${typeof resp.id === 'string' ? resp.id : 'sim'}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finish, logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** One SSE frame: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split SSE text into frames, each with its event name (or null) and data lines joined. */
export function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** An OpenAI SSE chunk object. */
function openAIChunk(model: string, delta: Json, finish: string | null): Json {
  return { id: `chatcmpl-${model}`, object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] };
}

/**
 * An Anthropic event stream as an OpenAI event stream: the text whole, the stop
 * reason mapped, and the usage the provider reported carried in a final chunk.
 */
export function anthropicSseToOpenAI(text: string, model: string): string {
  let body = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'end_turn';
  for (const frame of parseSseFrames(text)) {
    if (frame.data.trim() === '') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      if (typeof data.message.usage.input_tokens === 'number') inputTokens = data.message.usage.input_tokens;
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta) && typeof data.delta.text === 'string') body += data.delta.text;
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = data.delta.stop_reason;
      if (isRecord(data.usage) && typeof data.usage.output_tokens === 'number') outputTokens = data.usage.output_tokens;
    }
  }
  const finish = stopReason === 'max_tokens' ? 'length' : stopReason === 'tool_use' ? 'tool_calls' : stopReason === 'refusal' ? 'content_filter' : 'stop';
  const out: string[] = [sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null))];
  if (body !== '') out.push(sseEvent(null, openAIChunk(model, { content: body }, null)));
  out.push(sseEvent(null, openAIChunk(model, {}, finish)));
  out.push(sseEvent(null, { id: `chatcmpl-${model}`, object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } }));
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/**
 * A Gemini :streamGenerateContent SSE stream as an OpenAI event stream. Gemini frames
 * carry text pieces and a last frame with the finish reason and usageMetadata.
 */
export function geminiSseToOpenAI(text: string, model: string): string {
  let body = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let finishReason: unknown = 'STOP';
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '' || payload === '[DONE]') continue;
    let data: unknown;
    try {
      data = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const { text: piece } = geminiParts(first);
    body += piece;
    if (first.finishReason !== undefined) finishReason = first.finishReason;
    if (isRecord(data.usageMetadata)) {
      if (typeof data.usageMetadata.promptTokenCount === 'number') inputTokens = data.usageMetadata.promptTokenCount;
      if (typeof data.usageMetadata.candidatesTokenCount === 'number') outputTokens = data.usageMetadata.candidatesTokenCount;
    }
  }
  const finish = geminiStopToOpenAI(finishReason, false);
  const out: string[] = [sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null))];
  if (body !== '') out.push(sseEvent(null, openAIChunk(model, { content: body }, null)));
  out.push(sseEvent(null, openAIChunk(model, {}, finish)));
  out.push(sseEvent(null, { id: `chatcmpl-${model}`, object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } }));
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/**
 * A Gemini :streamGenerateContent SSE stream as an Anthropic event stream, in the
 * message_start / content_block_start / delta / stop / message_delta / message_stop
 * order, carrying the usage the provider reported.
 */
export function geminiSseToAnthropic(text: string, model: string): string {
  let body = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let finishReason: unknown = 'STOP';
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '' || payload === '[DONE]') continue;
    let data: unknown;
    try {
      data = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const { text: piece } = geminiParts(first);
    body += piece;
    if (first.finishReason !== undefined) finishReason = first.finishReason;
    if (isRecord(data.usageMetadata)) {
      if (typeof data.usageMetadata.promptTokenCount === 'number') inputTokens = data.usageMetadata.promptTokenCount;
      if (typeof data.usageMetadata.candidatesTokenCount === 'number') outputTokens = data.usageMetadata.candidatesTokenCount;
    }
  }
  const stopReason = geminiStopToAnthropic(finishReason, false);
  return [
    sseEvent('message_start', { type: 'message_start', message: { id: `msg_${model}`, type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } } }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    ...(body !== '' ? [sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: body } })] : []),
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  ].join('');
}
