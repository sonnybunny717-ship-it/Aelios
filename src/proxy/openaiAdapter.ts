import type { AssembledPrompt, SystemBlock } from "../assembler/types";
import { assembledToOpenAIChatMessages, assembledToOpenAIMessages } from "../assembler/toOpenAI";
import type { Env, OpenAIChatMessage, OpenAIChatRequest, TokenUsage } from "../types";

interface OpenRouterTextPart {
  type: "text";
  text: string;
  cache_control?: {
    type: "ephemeral";
    ttl?: "5m" | "1h";
  };
}

function stripClaudeNativeThinkingFields(req: OpenAIChatRequest): OpenAIChatRequest {
  const cleaned: OpenAIChatRequest = { ...req };
  delete cleaned.thinking;
  delete cleaned.post_user_instructions;
  delete cleaned.context_epoch;
  delete cleaned.context_compaction;
  return cleaned;
}

function getPostUserInstructions(req: OpenAIChatRequest): string | null {
  return typeof req.post_user_instructions === "string"
    ? req.post_user_instructions.trim() || null
    : null;
}

export function isOpenRouterAnthropicModel(model: string): boolean {
  return model.toLowerCase().startsWith("openrouter/anthropic/");
}

function buildOpenRouterCacheControl(
  env: Env,
  model: string
): OpenRouterTextPart["cache_control"] | undefined {
  const ttl = getOpenRouterAnthropicCacheTtl(env, model);
  if (!ttl) return undefined;
  return ttl === "1h" ? { type: "ephemeral", ttl } : { type: "ephemeral" };
}

export function getOpenRouterAnthropicCacheTtl(env: Env, model: string): "5m" | "1h" | null {
  if (env.ANTHROPIC_CACHE_ENABLED === "false") return null;
  if (model.toLowerCase() === "openrouter/anthropic/claude-fable-5") return "5m";
  return env.ANTHROPIC_CACHE_TTL === "1h" ? "1h" : "5m";
}

export function getOpenRouterAnthropicCacheMode(env: Env): string | null {
  if (env.ANTHROPIC_CACHE_ENABLED === "false") return null;
  return env.ANTHROPIC_ROLLING_CACHE_ENABLED === "false"
    ? "openrouter_anthropic_explicit"
    : "openrouter_anthropic_explicit_rolling";
}

function applyOpenRouterSessionId(req: OpenAIChatRequest): OpenAIChatRequest {
  const result = { ...req };
  const conversationId = typeof result.conversation_id === "string" ? result.conversation_id.trim() : "";
  if (conversationId && typeof result.session_id !== "string") result.session_id = conversationId;
  delete result.conversation_id;
  return result;
}

function splitOpenRouterDynamicBlocks(assembled: AssembledPrompt): {
  systemBlocks: SystemBlock[];
  volatileContext: string | null;
  dynamicMemoryPatch: string | null;
} {
  const volatileIndex = assembled.meta.block_ids.indexOf("client_volatile_context");
  const memoryIndex = assembled.meta.block_ids.indexOf("dynamic_memory_patch");
  const removed = new Set([volatileIndex, memoryIndex].filter((index) => index >= 0));
  return {
    systemBlocks: assembled.system_blocks.filter((_, index) => !removed.has(index)),
    volatileContext: volatileIndex >= 0 ? assembled.system_blocks[volatileIndex]?.text ?? null : null,
    dynamicMemoryPatch: memoryIndex >= 0 ? assembled.system_blocks[memoryIndex]?.text ?? null : null
  };
}

function cloneContentParts(content: OpenAIChatMessage["content"]): Array<unknown> {
  if (typeof content === "string") return [{ type: "text", text: content }];
  if (!Array.isArray(content)) return [{ type: "text", text: "" }];
  return content.map((part) =>
    part && typeof part === "object" && !Array.isArray(part) ? { ...(part as Record<string, unknown>) } : part
  );
}

function applyOpenRouterRollingCache(
  messages: OpenAIChatMessage[],
  cacheControl: OpenRouterTextPart["cache_control"],
  env: Env
): void {
  if (!cacheControl || env.ANTHROPIC_ROLLING_CACHE_ENABLED === "false") return;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== "user") continue;
    const parts = cloneContentParts(message.content);
    for (let partIndex = parts.length - 1; partIndex >= 0; partIndex -= 1) {
      const part = parts[partIndex];
      if (!part || typeof part !== "object" || Array.isArray(part)) continue;
      if ((part as Record<string, unknown>).type !== "text") continue;
      parts[partIndex] = { ...(part as Record<string, unknown>), cache_control: cacheControl };
      message.content = parts;
      return;
    }
    parts.push({ type: "text", text: "", cache_control: cacheControl });
    message.content = parts;
    return;
  }
}

function appendOpenRouterUncachedContext(messages: OpenAIChatMessage[], text: string | null): void {
  const trimmed = text?.trim();
  if (!trimmed) return;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== "user") continue;
    const parts = cloneContentParts(message.content);
    parts.push({ type: "text", text: trimmed });
    message.content = parts;
    return;
  }
  messages.push({ role: "user", content: [{ type: "text", text: trimmed }] });
}

function buildOpenRouterAnthropicMessages(
  assembled: AssembledPrompt,
  env: Env,
  targetModel: string,
  postUserInstructions: string | null
): OpenAIChatMessage[] {
  const cacheControl = buildOpenRouterCacheControl(env, targetModel);
  const { systemBlocks, volatileContext, dynamicMemoryPatch } = splitOpenRouterDynamicBlocks(assembled);
  const messages: OpenAIChatMessage[] = [];

  if (systemBlocks.length > 0) {
    const content: OpenRouterTextPart[] = systemBlocks.map((block) => ({
      type: "text",
      text: block.text,
      ...(block.cache_control && cacheControl ? { cache_control: cacheControl } : {})
    }));
    messages.push({ role: "system", content });
  }

  messages.push(...assembledToOpenAIMessages(assembled.messages).map((message) => ({
    ...message,
    content: Array.isArray(message.content) ? cloneContentParts(message.content) : message.content
  })));
  applyOpenRouterRollingCache(messages, cacheControl, env);
  appendOpenRouterUncachedContext(messages, postUserInstructions);
  appendOpenRouterUncachedContext(messages, volatileContext);
  appendOpenRouterUncachedContext(messages, dynamicMemoryPatch);
  return messages;
}

export function buildOpenAICompatRequest(req: OpenAIChatRequest, targetModel: string, env?: Env): OpenAIChatRequest {
  const postUserInstructions = getPostUserInstructions(req);
  const cleaned = stripClaudeNativeThinkingFields(req);
  const messages = cleaned.messages.map((message) => ({
    ...message,
    content: Array.isArray(message.content) ? cloneContentParts(message.content) : message.content
  }));
  appendOpenRouterUncachedContext(messages, postUserInstructions);
  let result: OpenAIChatRequest = {
    ...cleaned,
    messages,
    model: targetModel,
    stream: Boolean(cleaned.stream)
  };
  if (isOpenRouterAnthropicModel(targetModel)) {
    result = applyOpenRouterSessionId(result);
    const cacheControl = env ? buildOpenRouterCacheControl(env, targetModel) : undefined;
    if (cacheControl) result.cache_control = cacheControl;
  }
  return result;
}

/**
 * Build an OpenAI-compatible request from an AssembledPrompt.
 * System blocks are merged into one system message; conversation messages
 * (including image_url) are preserved as-is.
 */
export function buildOpenAIRequestFromAssembled(
  req: OpenAIChatRequest,
  targetModel: string,
  assembled: AssembledPrompt,
  env?: Env
): OpenAIChatRequest {
  if (isOpenRouterAnthropicModel(targetModel) && env) {
    const postUserInstructions = getPostUserInstructions(req);
    const cleaned = applyOpenRouterSessionId(stripClaudeNativeThinkingFields(req));
    return {
      ...cleaned,
      messages: buildOpenRouterAnthropicMessages(assembled, env, targetModel, postUserInstructions),
      model: targetModel,
      stream: Boolean(cleaned.stream)
    };
  }
  const messages = assembledToOpenAIChatMessages(assembled);
  return buildOpenAICompatRequest({ ...req, messages }, targetModel);
}

export function normalizeOpenAIUsage(usage: TokenUsage | undefined): TokenUsage | undefined {
  if (!usage) return undefined;
  const details = usage.prompt_tokens_details;
  const promptDetails = details && typeof details === "object" && !Array.isArray(details)
    ? details as Record<string, unknown>
    : null;
  const cachedTokens = typeof promptDetails?.cached_tokens === "number" ? promptDetails.cached_tokens : undefined;
  const cacheWriteTokens = typeof promptDetails?.cache_write_tokens === "number"
    ? promptDetails.cache_write_tokens
    : undefined;
  return {
    ...usage,
    cache_read_input_tokens: usage.cache_read_input_tokens ?? cachedTokens,
    cache_creation_input_tokens: usage.cache_creation_input_tokens ?? cacheWriteTokens
  };
}

export function getOpenAICompatUrl(env: Env): string {
  return `${normalizeAiGatewayBaseUrl(env)}/compat/chat/completions`;
}

export function normalizeAiGatewayBaseUrl(env: Env): string {
  const base = env.AI_GATEWAY_BASE_URL;
  if (!base) {
    throw new Error("Missing AI_GATEWAY_BASE_URL");
  }

  return base
    .replace(/\/+$/, "")
    .replace(/\/compat$/i, "")
    .replace(/\/compat\/chat\/completions$/i, "")
    .replace(/\/compat\/embeddings$/i, "")
    .replace(/\/anthropic\/v1\/messages$/i, "");
}

export function buildOpenAICompatHeaders(env: Env): Headers {
  const headers = new Headers({
    "content-type": "application/json"
  });

  if (env.CF_AIG_TOKEN) {
    headers.set("cf-aig-authorization", `Bearer ${env.CF_AIG_TOKEN}`);
  }

  return headers;
}

export async function callOpenAICompat(env: Env, body: OpenAIChatRequest): Promise<Response> {
  return fetch(getOpenAICompatUrl(env), {
    method: "POST",
    headers: buildOpenAICompatHeaders(env),
    body: JSON.stringify(body)
  });
}

export async function callOpenAICompatEmbeddings(
  env: Env,
  body: { model: string; input: string | string[]; dimensions?: number }
): Promise<Response> {
  const headers = buildOpenAICompatHeaders(env);
  if (body.model.startsWith("workers-ai/") && env.CLOUDFLARE_API_TOKEN) {
    headers.set("authorization", `Bearer ${env.CLOUDFLARE_API_TOKEN}`);
  }

  return fetch(`${normalizeAiGatewayBaseUrl(env)}/compat/embeddings`, {
    method: "POST",
    headers,
    body: JSON.stringify(body)
  });
}
