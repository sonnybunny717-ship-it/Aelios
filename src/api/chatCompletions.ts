import { authenticate } from "../auth/apiKey";
import { requireScope } from "../auth/scopes";
import {
  getOrCreateConversation,
} from "../db/conversations";
import { listMemories } from "../db/memories";
import { saveAssistantMessage, saveUserMessages } from "../db/messages";
import { getLatestSummary } from "../db/summaries";
import { saveUsageLog } from "../db/usageLogs";
import { extractLastUserText, injectMemoryPatchAsSystemMessage, selectMemoriesForInjection } from "../memory/inject";
import { toMemoryApiRecord } from "../memory/search";
import { assemble } from "../assembler/assemble";
import { PERSONA_MEMORY_TYPES, type AssembledPrompt } from "../assembler/types";
import { enqueueMemoryMaintenanceIfNeeded, enqueueRetentionIfNeeded } from "../queue/producer";
import {
  buildAnthropicNativeRequest,
  buildAnthropicRequestFromAssembled,
  callAnthropicNative,
  getAnthropicCacheMode,
  getAnthropicCacheTtl,
  parseAnthropicNonStream
} from "../proxy/anthropicAdapter";
import {
  buildOpenAICompatRequest,
  buildOpenAIRequestFromAssembled,
  callOpenAICompat,
  getOpenRouterAnthropicCacheMode,
  getOpenRouterAnthropicCacheTtl,
  isOpenRouterAnthropicModel,
  normalizeOpenAIUsage
} from "../proxy/openaiAdapter";
import { classifyProvider, resolveTargetModel } from "../proxy/resolveModel";
import { streamAnthropicToOpenAI } from "../proxy/streamAnthropic";
import { streamOpenAIWithTee } from "../proxy/streamOpenAI";
import { CONTENT_RULES } from "../preset/regexRules";
import { applyRegexRules } from "../preset/regexPipeline";
import type { Env, MemoryApiRecord, OpenAIChatRequest, OpenAIChatResponse } from "../types";
import { openAiError } from "../utils/json";
import { hasImageContent } from "../utils/messages";
import { addCloudflareCost } from "../billing/cloudflare";
import {
  prepareConversationContext,
  type PreparedConversationContext,
} from "../memory/windowContext";

function extractAssistantText(response: OpenAIChatResponse): string {
  const message = response.choices?.[0]?.message;
  if (!message) return "";

  if (typeof message.content === "string") return message.content;
  if (message.content == null) return "";
  return JSON.stringify(message.content);
}

function requestConversationId(body: OpenAIChatRequest, namespace: string): string | undefined {
  const raw = typeof body.conversation_id === "string" ? body.conversation_id.trim() : "";
  if (!raw) return undefined;
  const normalized = raw.slice(0, 256).replace(/[^A-Za-z0-9._:-]/g, "_");
  return normalized.startsWith(`${namespace}:`) ? normalized : `${namespace}:${normalized}`;
}

function fingerprint(value: string): string {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function countAnthropicMessageBlocks(messages: AssembledPrompt["messages"]): number {
  let blocks = 0;
  let previousRole: "user" | "assistant" | null = null;
  for (const message of messages) {
    if (message.role === "assistant" && previousRole === "assistant") continue;
    blocks += 1;
    previousRole = message.role;
  }
  return blocks;
}

function buildCacheDiagnostics(
  conversationId: string,
  assembled: AssembledPrompt,
  conversationContext: PreparedConversationContext,
): string {
  const anchorHashes: Record<string, string> = {};
  let cumulativeSystem = "";
  for (let index = 0; index < assembled.system_blocks.length; index += 1) {
    const block = assembled.system_blocks[index];
    const id = assembled.meta.block_ids[index] || `system_${index}`;
    cumulativeSystem += `${id.length}:${id}${block.text.length}:${block.text}`;
    if (block.cache_control) anchorHashes[id] = fingerprint(cumulativeSystem);
  }

  return JSON.stringify({
    version: 1,
    conversation_id_hash: fingerprint(conversationId),
    context_epoch: conversationContext.epoch,
    anchor_hashes: anchorHashes,
    summary_hash: fingerprint(conversationContext.summaryEntry?.content || ""),
    summary_source_updated_at: conversationContext.summarySnapshot.sourceUpdatedAt,
    message_rows: assembled.messages.length,
    message_blocks: countAnthropicMessageBlocks(assembled.messages),
    rolling_prefix_hash: fingerprint(JSON.stringify(assembled.messages)),
  });
}

export function hasToolContent(body: OpenAIChatRequest): boolean {
  return body.messages.some(
    (m) => m.role === "tool" || (m.role === "assistant" && m.tool_calls != null)
  );
}

/**
 * Fetch pinned memories whose type is "persona" or "identity" from D1.
 * Returns MemoryApiRecord[] for the assembler's persona_pinned block.
 * Deterministic sort is applied later by the assembler itself.
 */
async function fetchPinnedPersonaMemories(
  db: D1Database,
  namespace: string
): Promise<MemoryApiRecord[]> {
  const records = await listMemories(db, {
    namespace,
    status: "active",
    limit: 100,
  });

  return records
    .filter((r) => r.pinned && PERSONA_MEMORY_TYPES.includes(r.type))
    .map((r) => toMemoryApiRecord(r));
}

export async function handleChatCompletions(
  request: Request,
  env: Env,
  ctx: ExecutionContext
): Promise<Response> {
  const auth = await authenticate(request, env);
  if (!auth.ok) return openAiError("Unauthorized", 401, "authentication_error");

  const scopeError = requireScope(auth.profile, "chat:proxy");
  if (scopeError) return scopeError;

  let body: OpenAIChatRequest;
  try {
    body = (await request.json()) as OpenAIChatRequest;
  } catch {
    return openAiError("Request body must be valid JSON", 400);
  }

  if (!Array.isArray(body.messages)) {
    return openAiError("messages must be an array", 400);
  }

  let targetModel: string;
  try {
    targetModel = resolveTargetModel(body.model, auth.profile, env);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to resolve target model";
    return openAiError(message, 500);
  }

  if (hasImageContent(body)) {
    if (!env.VISION_MODEL) return openAiError("Missing VISION_MODEL", 500);
    targetModel = env.VISION_MODEL;
  }

  const provider = classifyProvider(targetModel);
  const openRouterAnthropic = isOpenRouterAnthropicModel(targetModel);
  const openRouterCacheMode = openRouterAnthropic ? getOpenRouterAnthropicCacheMode(env) : null;
  const openRouterCacheTtl = openRouterCacheMode ? getOpenRouterAnthropicCacheTtl(env, targetModel) : null;

  const conversation = await getOrCreateConversation(env.DB, {
    namespace: auth.profile.namespace,
    id: requestConversationId(body, auth.profile.namespace)
  });

  const savedUserMessageIds = await saveUserMessages(env.DB, {
    conversationId: conversation.id,
    namespace: auth.profile.namespace,
    source: auth.profile.source,
    messages: body.messages,
    requestModel: body.model,
    upstreamModel: targetModel,
    upstreamProvider: provider,
    stream: Boolean(body.stream)
  });
  const latestUserMessageId = savedUserMessageIds[savedUserMessageIds.length - 1];

  const memories = await selectMemoriesForInjection(env, {
    profile: auth.profile,
    query: extractLastUserText(body.messages)
  });

  const pinnedPersonaMemories = await fetchPinnedPersonaMemories(env.DB, auth.profile.namespace);
  const latestSummary = await getLatestSummary(env.DB, auth.profile.namespace);
  let conversationContext: PreparedConversationContext;
  try {
    conversationContext = await prepareConversationContext(env, {
      conversationId: conversation.id,
      namespace: auth.profile.namespace,
      request: body,
      latestSummary,
      currentPinnedPersonaMemories: pinnedPersonaMemories,
    });
  } catch (error) {
    console.error("conversation context preparation failed", error);
    return openAiError("Failed to prepare conversation context", 502);
  }

  let upstream: Response;
  let clientSystemHash: string | null = null;
  let cacheAnchorBlock: string | null = null;
  let cacheDiagnosticsJson: string | null = null;
  try {
    if (provider === "anthropic") {
      if (hasToolContent(body)) {
        // Tool messages / tool_calls not yet supported by assembler — fall back
        const anthropicRequest = await buildAnthropicNativeRequest(body, {
          env,
          targetModel,
          namespace: auth.profile.namespace,
          memories
        });
        upstream = await callAnthropicNative(env, anthropicRequest, targetModel);
      } else {
        const assembled = assemble({
          request: body,
          pinnedPersonaMemories: conversationContext.pinnedPersonaMemories,
          summaryEntry: conversationContext.summaryEntry,
          ragMemories: memories,
          visionOutput: null,
        });
        clientSystemHash = assembled.meta.client_system_hash;
        cacheAnchorBlock = assembled.meta.anchor_index >= 0 ? "client_system" : null;
        cacheDiagnosticsJson = buildCacheDiagnostics(conversation.id, assembled, conversationContext);
        // NOTE: Anthropic adapter stringifies structured content (image_url etc.)
        // as a temporary fallback; native Anthropic image support will be added
        // when the vision pipeline is wired in.
        upstream = await callAnthropicNative(env, buildAnthropicRequestFromAssembled(body, targetModel, assembled, env), targetModel);
      }
    } else {
      if (hasToolContent(body)) {
        // Tool messages / tool_calls not yet supported by assembler — fall back
        const patchedBody = injectMemoryPatchAsSystemMessage(body, memories);
        const upstreamRequest = buildOpenAICompatRequest(patchedBody, targetModel, env);
        upstream = await callOpenAICompat(env, upstreamRequest);
      } else {
        const assembled = assemble({
          request: body,
          pinnedPersonaMemories: conversationContext.pinnedPersonaMemories,
          summaryEntry: conversationContext.summaryEntry,
          ragMemories: memories,
          visionOutput: null,
        });
        clientSystemHash = assembled.meta.client_system_hash;
        if (openRouterAnthropic && assembled.meta.anchor_index >= 0) cacheAnchorBlock = "client_system";
        if (openRouterAnthropic) {
          cacheDiagnosticsJson = buildCacheDiagnostics(conversation.id, assembled, conversationContext);
        }
        upstream = await callOpenAICompat(env, buildOpenAIRequestFromAssembled(body, targetModel, assembled, env));
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to call upstream";
    return openAiError(message, 502);
  }

  if (!upstream.ok) {
    const errorText = await upstream.text();
    return new Response(errorText, {
      status: upstream.status,
      headers: {
        "content-type": upstream.headers.get("content-type") || "application/json; charset=utf-8"
      }
    });
  }

  const aiGatewayLogId = upstream.headers.get("cf-aig-log-id");

  if (body.stream) {
    if (provider === "anthropic") {
      return streamAnthropicToOpenAI(upstream, {
        env,
        ctx,
        profile: auth.profile,
        conversationId: conversation.id,
        fromMessageId: latestUserMessageId,
        requestModel: body.model,
        upstreamModel: targetModel,
        provider,
        clientSystemHash,
        cacheAnchorBlock,
        cacheDiagnosticsJson
      });
    }

    return streamOpenAIWithTee(upstream, {
      env,
      ctx,
      profile: auth.profile,
      conversationId: conversation.id,
      fromMessageId: latestUserMessageId,
      requestModel: body.model,
      upstreamModel: targetModel,
      provider,
      clientSystemHash,
      cacheAnchorBlock,
      cacheDiagnosticsJson,
      cacheMode: openRouterCacheMode,
      cacheTtl: openRouterCacheTtl
    });
  }

  const responseText = await upstream.text();

  if (provider === "anthropic") {
    let anthropicParsed: unknown;
    try {
      anthropicParsed = JSON.parse(responseText) as unknown;
    } catch {
      return openAiError("Upstream returned invalid JSON", 502);
    }

    const parsed = parseAnthropicNonStream(anthropicParsed as never);
    parsed.usage = await addCloudflareCost(parsed.usage, env, aiGatewayLogId);
    parsed.openai.usage = parsed.usage;
    parsed.openai.aelios_context_epoch = conversationContext.epoch;
    const anthropicCacheMode = getAnthropicCacheMode(env);
    // Filter visible content only — reasoning_content is preserved upstream.
    const filteredContent = applyRegexRules(parsed.content, CONTENT_RULES);
    if (parsed.openai.choices?.[0]?.message) {
      parsed.openai.choices[0].message.content = filteredContent;
    }
    const assistantMessageId = await saveAssistantMessage(env.DB, {
      conversationId: conversation.id,
      namespace: auth.profile.namespace,
      source: auth.profile.source,
      content: filteredContent,
      requestModel: body.model,
      upstreamModel: targetModel,
      provider,
      stream: false,
      finishReason: parsed.finishReason,
      usage: parsed.usage,
      cacheMode: anthropicCacheMode,
      cacheTtl: getAnthropicCacheTtl(env, targetModel)
    });

    ctx.waitUntil(
      Promise.all([
        saveUsageLog(env.DB, {
          messageId: assistantMessageId,
          namespace: auth.profile.namespace,
          provider,
          model: targetModel,
          usage: parsed.usage,
          cacheMode: anthropicCacheMode,
          cacheTtl: getAnthropicCacheTtl(env, targetModel),
          clientSystemHash,
          cacheAnchorBlock,
          cacheDiagnosticsJson
        }),
        enqueueMemoryMaintenanceIfNeeded(env, {
          namespace: auth.profile.namespace,
          conversationId: conversation.id,
          fromMessageId: latestUserMessageId,
          toMessageId: assistantMessageId,
          source: auth.profile.source
        }),
        enqueueRetentionIfNeeded(env, auth.profile.namespace)
      ])
    );

    return new Response(JSON.stringify(parsed.openai), {
      status: 200,
      headers: {
        "content-type": "application/json; charset=utf-8"
      }
    });
  }

  let parsed: OpenAIChatResponse;
  try {
    parsed = JSON.parse(responseText) as OpenAIChatResponse;
  } catch {
    return openAiError("Upstream returned invalid JSON", 502);
  }
  parsed.usage = normalizeOpenAIUsage(parsed.usage);
  parsed.usage = await addCloudflareCost(parsed.usage, env, aiGatewayLogId);
  parsed.aelios_context_epoch = conversationContext.epoch;

  const assistantContent = extractAssistantText(parsed);
  const filteredContent = applyRegexRules(assistantContent, CONTENT_RULES);
  // Patch the response that goes back to the client.
  if (parsed.choices?.[0]?.message) {
    parsed.choices[0].message.content = filteredContent;
  }
  const assistantMessageId = await saveAssistantMessage(env.DB, {
    conversationId: conversation.id,
    namespace: auth.profile.namespace,
    source: auth.profile.source,
    content: filteredContent,
    requestModel: body.model,
    upstreamModel: targetModel,
    provider,
    stream: false,
    finishReason: parsed.choices?.[0]?.finish_reason,
    usage: parsed.usage,
    cacheMode: openRouterCacheMode,
    cacheTtl: openRouterCacheTtl
  });

  ctx.waitUntil(
    Promise.all([
      saveUsageLog(env.DB, {
        messageId: assistantMessageId,
        namespace: auth.profile.namespace,
        provider,
        model: targetModel,
        usage: parsed.usage,
        cacheMode: openRouterCacheMode,
        cacheTtl: openRouterCacheTtl,
        clientSystemHash,
        cacheAnchorBlock,
        cacheDiagnosticsJson
      }),
      enqueueMemoryMaintenanceIfNeeded(env, {
        namespace: auth.profile.namespace,
        conversationId: conversation.id,
        fromMessageId: latestUserMessageId,
        toMessageId: assistantMessageId,
        source: auth.profile.source
      }),
      enqueueRetentionIfNeeded(env, auth.profile.namespace)
    ])
  );

  return new Response(JSON.stringify(parsed), {
    status: 200,
    headers: {
      "content-type": "application/json; charset=utf-8"
    }
  });
}
