import { buildStableMemoryPack } from "../memory/stablePack";
import type { AssembledPrompt } from "../assembler/types";
import { assembledToAnthropicMessages, assembledToAnthropicSystem } from "../assembler/toAnthropic";
import type { Env, MemoryApiRecord, OpenAIChatMessage, OpenAIChatRequest, OpenAIChatResponse, TokenUsage } from "../types";
import { formatMemoryPatch } from "../memory/inject";
import { normalizeAiGatewayBaseUrl } from "./openaiAdapter";

interface AnthropicTextBlock {
  type: "text";
  text: string;
  cache_control?: {
    type: "ephemeral";
    ttl?: "5m" | "1h";
  };
}

interface AnthropicMessage {
  role: "user" | "assistant";
  content: AnthropicTextBlock[];
}

type AdaptiveEffort = "low" | "medium" | "high" | "xhigh" | "max";

interface AnthropicRequest {
  model: string;
  max_tokens: number;
  metadata?: {
    user_id: string;
  };
  cache_control?: {
    type: "ephemeral";
    ttl?: "5m" | "1h";
  };
  temperature?: number;
  stream?: boolean;
  thinking?:
    | {
        type: "enabled";
        budget_tokens: number;
        display?: "summarized" | "omitted";
      }
    | {
        type: "adaptive";
        display?: "summarized" | "omitted";
      }
    | {
        type: "disabled";
      };
  output_config?: {
    effort: AdaptiveEffort;
  };
  system: AnthropicTextBlock[];
  messages: AnthropicMessage[];
}

interface CloudflareFableRequest {
  max_tokens: number;
  metadata?: AnthropicRequest["metadata"];
  stream?: boolean;
  thinking: {
    type: "adaptive";
  };
  output_config: {
    effort: "high";
  };
  system?: string;
  messages: Array<{
    role: AnthropicMessage["role"];
    content: AnthropicTextBlock[];
  }>;
}

interface AnthropicResponse {
  id?: string;
  model?: string;
  role?: string;
  content?: Array<{ type?: string; text?: string; thinking?: string }>;
  stop_reason?: string | null;
  usage?: TokenUsage;
}

function contentToText(content: OpenAIChatMessage["content"]): string {
  if (typeof content === "string") return content;
  if (content == null) return "";
  return JSON.stringify(content);
}

function stripAnthropicProviderPrefix(model: string): string {
  return model.replace(/^anthropic\//i, "");
}

function isCloudflareFableModel(model: string): boolean {
  return model.toLowerCase() === "anthropic/claude-fable-5";
}

function isClaudeOpus5Model(model: string): boolean {
  return model.toLowerCase().replace(/^anthropic\//, "") === "claude-opus-5";
}

function getCloudflareAiGatewayId(env: Env): string {
  if (env.AI_GATEWAY_ID) return env.AI_GATEWAY_ID;
  try {
    const url = new URL(env.AI_GATEWAY_BASE_URL || "");
    const parts = url.pathname.split("/").filter(Boolean);
    if (url.hostname === "gateway.ai.cloudflare.com" && parts[0] === "v1") {
      return parts[2] || "";
    }
  } catch {}
  return "";
}

function joinAnthropicTextBlocks(blocks: AnthropicTextBlock[]): string {
  return blocks.map((block) => block.text).filter(Boolean).join("\n\n");
}

function buildCloudflareFableRequest(body: AnthropicRequest): CloudflareFableRequest {
  const system = joinAnthropicTextBlocks(body.system);
  return {
    max_tokens: body.max_tokens,
    metadata: body.metadata,
    stream: body.stream,
    thinking: { type: "adaptive" },
    output_config: { effort: "high" },
    system: system || undefined,
    messages: body.messages.map((message) => ({
      role: message.role,
      content: message.content.map((block) => ({
        ...block,
        ...(block.cache_control ? { cache_control: { type: "ephemeral" as const } } : {})
      }))
    }))
  };
}

function normalizeAdaptiveEffort(value: unknown): AdaptiveEffort | null {
  if (typeof value !== "string") return null;
  const normalized = value.trim().toLowerCase();
  if (normalized === "minimal") return "low";
  if (normalized === "auto") return "high";
  if (normalized === "extra_high") return "xhigh";
  return ["low", "medium", "high", "xhigh", "max"].includes(normalized)
    ? normalized as AdaptiveEffort
    : null;
}

function getAdaptiveEffort(req: OpenAIChatRequest): AdaptiveEffort {
  const sources = [
    req,
    isRecord(req.extra_body) ? req.extra_body : null,
    isRecord(req.extraBody) ? req.extraBody : null,
  ];
  for (const source of sources) {
    if (!source) continue;
    const direct = normalizeAdaptiveEffort(source.reasoning_effort);
    if (direct) return direct;
    if (isRecord(source.reasoning)) {
      const nested = normalizeAdaptiveEffort(source.reasoning.effort);
      if (nested) return nested;
    }
  }
  return "high";
}

function applyModelThinkingMode(
  body: AnthropicRequest,
  req: OpenAIChatRequest,
  targetModel: string
): AnthropicRequest {
  if (!isClaudeOpus5Model(targetModel)) return body;
  const directive = getRequestThinkingDirective(req);
  if (directive.enabled === false) {
    return {
      ...body,
      temperature: undefined,
      thinking: { type: "disabled" },
      output_config: undefined,
    };
  }
  return {
    ...body,
    temperature: undefined,
    thinking: { type: "adaptive", display: "summarized" },
    output_config: { effort: getAdaptiveEffort(req) },
  };
}

function parseCustomProviderModel(model: string): { slug: string; model: string } | null {
  const match = model.match(/^custom-([a-z0-9-]+)\/(.+)$/i);
  if (!match) return null;
  return {
    slug: match[1],
    model: match[2]
  };
}

function stripAnthropicModelPrefix(model: string): string {
  return parseCustomProviderModel(model)?.model || stripAnthropicProviderPrefix(model);
}

function getCustomAnthropicMessagesPath(env: Env): string {
  return (env.CUSTOM_ANTHROPIC_MESSAGES_PATH || "messages").replace(/^\/+/, "");
}

export function getAnthropicCacheTtl(env: Env, model?: string): "5m" | "1h" | null {
  if (env.ANTHROPIC_CACHE_ENABLED === "false") return null;
  if (model && isCloudflareFableModel(model)) return "5m";
  return env.ANTHROPIC_CACHE_TTL === "1h" ? "1h" : "5m";
}

function buildCacheControl(env: Env, model?: string): AnthropicTextBlock["cache_control"] | undefined {
  if (env.ANTHROPIC_CACHE_ENABLED === "false") return undefined;
  const ttl = getAnthropicCacheTtl(env, model);
  if (!ttl) return undefined;
  return ttl === "1h" ? { type: "ephemeral", ttl } : { type: "ephemeral" };
}

function buildAutomaticCacheControl(env: Env, model?: string): AnthropicRequest["cache_control"] | undefined {
  if (env.ANTHROPIC_CACHE_ENABLED === "false") return undefined;
  if (env.ANTHROPIC_AUTO_CACHE_ENABLED !== "true") return undefined;
  return buildCacheControl(env, model);
}

function buildCacheMetadata(env: Env): AnthropicRequest["metadata"] | undefined {
  if (env.ANTHROPIC_CACHE_ENABLED === "false") return undefined;
  return { user_id: env.ANTHROPIC_CACHE_USER_ID || "aelios-sticky-stable" };
}

export function getAnthropicCacheMode(env: Env): string | null {
  if (env.ANTHROPIC_CACHE_ENABLED === "false") return null;
  const parts = ["anthropic"];
  parts.push("explicit");
  if (env.ANTHROPIC_AUTO_CACHE_ENABLED === "true") parts.push("auto");
  if (env.ANTHROPIC_ROLLING_CACHE_ENABLED !== "false") parts.push("rolling");
  return parts.join("_");
}

function applyRollingMessageCache(messages: AnthropicMessage[], env: Env, model?: string): void {
  const cacheControl = buildCacheControl(env, model);
  if (!cacheControl) return;
  if (env.ANTHROPIC_ROLLING_CACHE_ENABLED === "false") return;

  // Rolling breakpoint on the last user message: each turn the boundary
  // advances and only the delta since the previous hit is written (1.25x on
  // the increment, 0.1x reads on everything before it). Volatile context is
  // appended AFTER this marked block, so it never enters the cached prefix.
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (message.role !== "user" || message.content.length === 0) continue;
    message.content[message.content.length - 1].cache_control = cacheControl;
    return;
  }
}

function appendUncachedUserContext(messages: AnthropicMessage[], text: string | null | undefined): void {
  const trimmed = text?.trim();
  if (!trimmed) return;

  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (message.role !== "user") continue;
    message.content.push({ type: "text", text: trimmed });
    return;
  }

  messages.push({ role: "user", content: [{ type: "text", text: trimmed }] });
}

function getPostUserInstructions(req: OpenAIChatRequest): string | null {
  return typeof req.post_user_instructions === "string"
    ? req.post_user_instructions.trim() || null
    : null;
}

function splitDynamicSystemBlocks(
  assembled: AssembledPrompt
): { systemBlocks: AssembledPrompt["system_blocks"]; dynamicMemoryPatch: string | null; volatileContext: string | null } {
  const memIdx = assembled.meta.block_ids.indexOf("dynamic_memory_patch");
  const volIdx = assembled.meta.block_ids.indexOf("client_volatile_context");

  const removeSet = new Set<number>();
  let dynamicMemoryPatch: string | null = null;
  let volatileContext: string | null = null;

  if (memIdx >= 0 && memIdx < assembled.system_blocks.length) {
    dynamicMemoryPatch = assembled.system_blocks[memIdx].text;
    removeSet.add(memIdx);
  }
  if (volIdx >= 0 && volIdx < assembled.system_blocks.length) {
    volatileContext = assembled.system_blocks[volIdx].text;
    removeSet.add(volIdx);
  }

  if (removeSet.size === 0) {
    return { systemBlocks: assembled.system_blocks, dynamicMemoryPatch: null, volatileContext: null };
  }

  return {
    systemBlocks: assembled.system_blocks.filter((_, i) => !removeSet.has(i)),
    dynamicMemoryPatch,
    volatileContext,
  };
}

function getMaxTokens(req: OpenAIChatRequest): number {
  const value = typeof req.max_tokens === "number" ? req.max_tokens : 1024;
  return Math.max(Math.floor(value), 1);
}

function clampThinkingBudget(value: unknown): number | null {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
  if (!Number.isFinite(numeric)) return null;
  return Math.min(Math.max(Math.floor(numeric), 1024), 32000);
}

function getEnvThinkingBudget(env: Env): number {
  const value = clampThinkingBudget(env.ANTHROPIC_THINKING_BUDGET);
  return value ?? 1024;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function parseBooleanLike(value: unknown): boolean | null {
  if (typeof value === "boolean") return value;
  if (typeof value !== "string") return null;
  const normalized = value.trim().toLowerCase();
  if (["true", "1", "yes", "on", "enabled"].includes(normalized)) return true;
  if (["false", "0", "no", "off", "disabled", "none"].includes(normalized)) return false;
  return null;
}

function budgetFromReasoningEffort(value: unknown): number | null {
  if (typeof value !== "string") return null;
  const normalized = value.trim().toLowerCase();
  if (["none", "off", "disabled", "disable"].includes(normalized)) return 0;
  if (["minimal", "low"].includes(normalized)) return 1024;
  if (["medium", "auto"].includes(normalized)) return 2048;
  if (normalized === "high") return 4096;
  if (["xhigh", "extra_high"].includes(normalized)) return 8192;
  return null;
}

function readThinkingDirective(source: Record<string, unknown>): { enabled?: boolean; budget?: number } {
  const effortBudget = budgetFromReasoningEffort(source.reasoning_effort);
  if (effortBudget === 0) return { enabled: false };
  if (effortBudget && effortBudget > 0) return { enabled: true, budget: effortBudget };

  const enableThinking = parseBooleanLike(source.enable_thinking);
  if (enableThinking !== null) {
    return {
      enabled: enableThinking,
      budget: clampThinkingBudget(source.thinking_budget ?? source.reasoning_budget ?? source.budget_tokens) ?? undefined
    };
  }

  const thinking = source.thinking;
  if (parseBooleanLike(thinking) !== null) {
    const enabled = parseBooleanLike(thinking);
    return {
      enabled: enabled ?? undefined,
      budget: clampThinkingBudget(source.thinking_budget ?? source.reasoning_budget ?? source.budget_tokens) ?? undefined
    };
  }

  if (isRecord(thinking)) {
    const type = typeof thinking.type === "string" ? thinking.type.trim().toLowerCase() : "";
    if (["disabled", "off", "none"].includes(type)) return { enabled: false };
    const budget = clampThinkingBudget(thinking.budget_tokens ?? thinking.budget ?? source.thinking_budget);
    if (type === "enabled" || budget) return { enabled: true, budget: budget ?? undefined };
  }

  const reasoning = source.reasoning;
  if (parseBooleanLike(reasoning) !== null) {
    const enabled = parseBooleanLike(reasoning);
    return {
      enabled: enabled ?? undefined,
      budget: clampThinkingBudget(source.reasoning_budget ?? source.budget_tokens) ?? undefined
    };
  }

  if (isRecord(reasoning)) {
    const enabled = parseBooleanLike(reasoning.enabled);
    if (enabled === false) return { enabled: false };
    const budget =
      clampThinkingBudget(reasoning.budget_tokens ?? reasoning.budget ?? source.reasoning_budget) ??
      budgetFromReasoningEffort(reasoning.effort);
    if (enabled === true || (budget && budget > 0)) return { enabled: true, budget: budget ?? undefined };
  }

  const budget = clampThinkingBudget(source.thinking_budget ?? source.reasoning_budget ?? source.budget_tokens);
  if (budget) return { enabled: true, budget };

  return {};
}

function getRequestThinkingDirective(req: OpenAIChatRequest): { enabled?: boolean; budget?: number } {
  for (const source of [req, isRecord(req.extra_body) ? req.extra_body : null, isRecord(req.extraBody) ? req.extraBody : null]) {
    if (!source) continue;
    const directive = readThinkingDirective(source);
    if (directive.enabled !== undefined || directive.budget !== undefined) return directive;
  }

  return {};
}

function buildThinkingConfig(env: Env, req: OpenAIChatRequest): AnthropicRequest["thinking"] | undefined {
  const requestDirective = getRequestThinkingDirective(req);
  if (requestDirective.enabled === false) return undefined;

  if (requestDirective.enabled === true || requestDirective.budget) {
    return {
      type: "enabled",
      budget_tokens: requestDirective.budget ?? getEnvThinkingBudget(env),
      display: "summarized"
    };
  }

  if (env.ANTHROPIC_THINKING_ENABLED !== "true") return undefined;
  return {
    type: "enabled",
    budget_tokens: getEnvThinkingBudget(env),
    display: "summarized"
  };
}

function getAnthropicMaxTokens(
  req: OpenAIChatRequest,
  env: Env,
  thinking: AnthropicRequest["thinking"] | undefined
): number {
  const maxTokens = getMaxTokens(req);
  if (!thinking || thinking.type !== "enabled") return maxTokens;
  return Math.max(maxTokens, thinking.budget_tokens + Math.min(Math.max(maxTokens, 256), 4096));
}

function extractSystemBlocks(messages: OpenAIChatMessage[]): AnthropicTextBlock[] {
  return messages
    .filter((message) => message.role === "system")
    .map((message) => contentToText(message.content).trim())
    .filter(Boolean)
    .map((text) => ({ type: "text", text }));
}

function convertMessages(messages: OpenAIChatMessage[]): AnthropicMessage[] {
  const result: AnthropicMessage[] = [];

  for (const message of messages) {
    if (message.role === "system") continue;
    const role = message.role === "assistant" ? "assistant" : "user";
    const text = contentToText(message.content);
    if (!text) continue;

    const previous = result[result.length - 1];
    if (previous?.role === role) {
      previous.content.push({ type: "text", text });
      continue;
    }

    result.push({
      role,
      content: [{ type: "text", text }]
    });
  }

  if (result.length === 0) {
    result.push({ role: "user", content: [{ type: "text", text: "" }] });
  }

  return result;
}

export function getAnthropicNativeUrl(env: Env): string {
  return `${normalizeAiGatewayBaseUrl(env)}/anthropic/v1/messages`;
}

export function getAnthropicUrlForModel(env: Env, targetModel: string): string {
  const customProvider = parseCustomProviderModel(targetModel);
  if (!customProvider) return getAnthropicNativeUrl(env);
  return `${normalizeAiGatewayBaseUrl(env)}/custom-${customProvider.slug}/${getCustomAnthropicMessagesPath(env)}`;
}

export function buildAnthropicHeaders(env: Env): Headers {
  const headers = new Headers({
    "content-type": "application/json",
    "anthropic-version": "2023-06-01",
    "cf-aig-skip-cache": "true"
  });

  if (env.CF_AIG_TOKEN) {
    headers.set("cf-aig-authorization", `Bearer ${env.CF_AIG_TOKEN}`);
  }

  return headers;
}

export async function buildAnthropicNativeRequest(
  req: OpenAIChatRequest,
  input: { env: Env; targetModel: string; namespace: string; memories: MemoryApiRecord[] }
): Promise<AnthropicRequest> {
  const thinking = buildThinkingConfig(input.env, req);
  const stableMemoryPack = await buildStableMemoryPack(input.env, input.namespace);
  const stableBlock: AnthropicTextBlock = {
    type: "text",
    text: stableMemoryPack
  };

  if (input.env.ANTHROPIC_CACHE_STABLE_SYSTEM !== "false") {
    stableBlock.cache_control = buildCacheControl(input.env, input.targetModel);
  }

  const dynamicMemoryPatch = formatMemoryPatch(input.memories);
  const system: AnthropicTextBlock[] = [
    ...extractSystemBlocks(req.messages),
    {
      type: "text",
      text: [
        "以下长期记忆来自代理层。",
        "你可以自然使用它们，但不要提到记忆系统、数据库、RAG、代理层。",
        "如果记忆与当前用户消息无关，不要强行提起。"
      ].join("\n")
    },
    stableBlock
  ];

  const messages = convertMessages(req.messages);
  applyRollingMessageCache(messages, input.env, input.targetModel);
  appendUncachedUserContext(messages, getPostUserInstructions(req));
  appendUncachedUserContext(messages, dynamicMemoryPatch);

  return applyModelThinkingMode({
    model: stripAnthropicModelPrefix(input.targetModel),
    max_tokens: getAnthropicMaxTokens(req, input.env, thinking),
    metadata: buildCacheMetadata(input.env),
    cache_control: buildAutomaticCacheControl(input.env, input.targetModel),
    temperature: thinking ? undefined : typeof req.temperature === "number" ? req.temperature : undefined,
    stream: Boolean(req.stream),
    thinking,
    system,
    messages
  }, req, input.targetModel);
}

/**
 * Build an Anthropic native request from an AssembledPrompt.
 *
 * - System blocks are converted via assembledToAnthropicSystem
 * - Messages via assembledToAnthropicMessages
 *   (structured content like image_url is JSON.stringify'd — temporary fallback)
 * - dynamic_memory_patch is moved out of system and appended after the
 *   rolling cache point, so changing RAG hits do not poison cached prefixes
 * - cache_control is applied to the client_system anchor block and the
 *   rolling user/window block, respecting ANTHROPIC_CACHE_ENABLED and
 *   ANTHROPIC_CACHE_TTL
 */
export function buildAnthropicRequestFromAssembled(
  req: OpenAIChatRequest,
  targetModel: string,
  assembled: AssembledPrompt,
  env: Env
): AnthropicRequest {
  const thinking = buildThinkingConfig(env, req);
  const { systemBlocks, dynamicMemoryPatch, volatileContext } = splitDynamicSystemBlocks(assembled);
  const system = assembledToAnthropicSystem(systemBlocks);
  const messages = assembledToAnthropicMessages(assembled.messages);
  applyCacheOverrides(system, env, targetModel);
  applyRollingMessageCache(messages, env, targetModel);
  appendUncachedUserContext(messages, getPostUserInstructions(req));
  appendUncachedUserContext(messages, volatileContext);
  appendUncachedUserContext(messages, dynamicMemoryPatch);

  return applyModelThinkingMode({
    model: stripAnthropicModelPrefix(targetModel),
    max_tokens: getAnthropicMaxTokens(req, env, thinking),
    metadata: buildCacheMetadata(env),
    cache_control: buildAutomaticCacheControl(env, targetModel),
    temperature: thinking ? undefined : typeof req.temperature === "number" ? req.temperature : undefined,
    stream: Boolean(req.stream),
    thinking,
    system,
    messages,
  }, req, targetModel);
}

function applyCacheOverrides(systemBlocks: AnthropicTextBlock[], env: Env, model?: string): void {
  // Layered anchors: client_system and long_term_summary can each carry
  // cache_control — override TTL (or strip) on every one of them.
  for (const block of systemBlocks) {
    if (!block.cache_control) continue;

    if (env.ANTHROPIC_CACHE_ENABLED === "false") {
      delete block.cache_control;
      continue;
    }

    const ttl = getAnthropicCacheTtl(env, model);
    block.cache_control = ttl ? { type: "ephemeral", ttl } : undefined;
  }
}

export async function callAnthropicNative(env: Env, body: AnthropicRequest, targetModel?: string): Promise<Response> {
  const resolvedModel = targetModel || body.model;
  if (isCloudflareFableModel(resolvedModel)) {
    if (!env.AI) throw new Error("Missing Cloudflare AI binding");

    const gatewayId = getCloudflareAiGatewayId(env);
    const output: unknown = await env.AI.run(
      resolvedModel as Parameters<Ai["run"]>[0],
      buildCloudflareFableRequest(body) as unknown as Parameters<Ai["run"]>[1],
      gatewayId ? { gateway: { id: gatewayId } } : undefined
    );
    const logId = (env.AI as Ai & { aiGatewayLogId?: string }).aiGatewayLogId;
    if (output instanceof Response) {
      if (!logId || output.headers.has("cf-aig-log-id")) return output;
      const headers = new Headers(output.headers);
      headers.set("cf-aig-log-id", logId);
      return new Response(output.body, {
        status: output.status,
        statusText: output.statusText,
        headers
      });
    }
    if (output instanceof ReadableStream) {
      return new Response(output, {
        headers: {
          "content-type": "text/event-stream; charset=utf-8",
          ...(logId ? { "cf-aig-log-id": logId } : {})
        }
      });
    }
    return Response.json(output, {
      headers: logId ? { "cf-aig-log-id": logId } : undefined
    });
  }

  return fetch(getAnthropicUrlForModel(env, targetModel || body.model), {
    method: "POST",
    headers: buildAnthropicHeaders(env),
    body: JSON.stringify(body)
  });
}

export function parseAnthropicNonStream(response: AnthropicResponse): {
  openai: OpenAIChatResponse;
  content: string;
  finishReason: string | null;
  usage?: TokenUsage;
} {
  const content = (response.content ?? [])
    .filter((block) => block.type === "text" && typeof block.text === "string")
    .map((block) => block.text)
    .join("");
  const reasoningContent = (response.content ?? [])
    .filter((block) => block.type === "thinking" && typeof block.thinking === "string")
    .map((block) => block.thinking)
    .join("");

  const usage = normalizeAnthropicUsage(response.usage);

  return {
    content,
    finishReason: response.stop_reason ?? null,
    usage,
    openai: {
      id: response.id,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: response.model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content,
            ...(reasoningContent ? { reasoning_content: reasoningContent } : {})
          },
          finish_reason: response.stop_reason ?? null
        }
      ],
      usage
    }
  };
}

export function normalizeAnthropicUsage(usage: TokenUsage | undefined): TokenUsage | undefined {
  if (!usage) return undefined;

  const input = usage.input_tokens ?? usage.prompt_tokens;
  const output = usage.output_tokens ?? usage.completion_tokens;

  return {
    ...usage,
    prompt_tokens: input,
    completion_tokens: output,
    total_tokens: typeof input === "number" && typeof output === "number" ? input + output : usage.total_tokens
  };
}
