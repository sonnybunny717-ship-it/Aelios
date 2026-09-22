import { listMemories } from "../db/memories";
import type { Env, InjectionMode, KeyProfile, MemoryApiRecord, OpenAIChatMessage, OpenAIChatRequest } from "../types";
import { filterAndCompressMemories } from "./filter";
import { searchMemories, toMemoryApiRecord } from "./search";
import { listVectorMemories, searchVectorMemories } from "./vectorStore";

function contentToText(content: OpenAIChatMessage["content"]): string {
  if (typeof content === "string") return content;
  if (content == null) return "";
  return content
    .flatMap((part) => {
      if (!part || typeof part !== "object" || Array.isArray(part)) return [];
      const value = part as { type?: unknown; text?: unknown };
      return value.type === "text" && typeof value.text === "string" ? [value.text] : [];
    })
    .join("\n");
}

export function extractLastUserText(messages: OpenAIChatMessage[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "user") return contentToText(message.content);
  }

  return "";
}

const RECALL_CONTEXT_MESSAGE_CHARS = 700;

function recallContextText(content: OpenAIChatMessage["content"]): string {
  const text = contentToText(content).trim();
  if (text.length <= RECALL_CONTEXT_MESSAGE_CHARS) return text;
  return text.slice(-RECALL_CONTEXT_MESSAGE_CHARS);
}

/**
 * Build a fallback search query from the current message and the immediately
 * preceding user/assistant exchange. This query is used only when the normal
 * current-message retrieval produces no usable memories.
 */
export function buildContextualMemoryQuery(
  messages: OpenAIChatMessage[],
  currentQuery: string
): string | null {
  const current = currentQuery.trim();
  if (!current) return null;

  let currentUserIndex = -1;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "user") {
      currentUserIndex = index;
      break;
    }
  }
  if (currentUserIndex <= 0) return null;

  const previous: string[] = [];
  for (let index = currentUserIndex - 1; index >= 0 && previous.length < 2; index -= 1) {
    const message = messages[index];
    if (message.role !== "user" && message.role !== "assistant") continue;
    const text = recallContextText(message.content);
    if (text) previous.unshift(text);
  }
  if (previous.length === 0) return null;

  return [...previous, current].join("\n\n");
}

function resolveInjectionMode(profile: KeyProfile, env: Env): InjectionMode {
  const mode = env.INJECTION_MODE || profile.injectionMode;
  if (mode === "full" || mode === "hybrid" || mode === "none") return mode;
  return "rag";
}

function getTopK(env: Env): number {
  const value = Number(env.MEMORY_TOP_K || 50);
  return Number.isFinite(value) ? Math.min(Math.max(value, 1), 200) : 50;
}

async function searchMemoriesForInjection(
  env: Env,
  input: { namespace: string; query: string; topK: number }
): Promise<MemoryApiRecord[]> {
  try {
    if (env.MEMORY_BACKEND === "d1") {
      return await searchMemories(env, {
        namespace: input.namespace,
        query: input.query,
        topK: input.topK
      });
    }

    return await searchVectorMemories(env, {
      namespace: input.namespace,
      query: input.query,
      topK: input.topK
    });
  } catch (error) {
    console.error("memory injection search failed", error);
    return [];
  }
}

async function listMemoriesForInjection(
  env: Env,
  input: { namespace: string; limit: number }
): Promise<MemoryApiRecord[]> {
  if (env.MEMORY_BACKEND === "d1") {
    const records = await listMemories(env.DB, {
      namespace: input.namespace,
      status: "active",
      limit: input.limit
    });
    return records.map((record) => toMemoryApiRecord(record));
  }

  try {
    const page = await listVectorMemories(env, {
      namespace: input.namespace,
      count: Math.min(input.limit, 1000)
    });
    return page.data;
  } catch (error) {
    console.error("memory injection list failed", error);
    return [];
  }
}

function dedupeMemories(memories: MemoryApiRecord[]): MemoryApiRecord[] {
  const seen = new Set<string>();
  const result: MemoryApiRecord[] = [];

  for (const memory of memories) {
    if (seen.has(memory.id)) continue;
    seen.add(memory.id);
    result.push(memory);
  }

  return result;
}

function sanitizeMemoryContent(text: string): string {
  return text
    .replace(/debug-test/gi, "")
    .replace(/记忆系统/g, "")
    .replace(/自动记忆测试口令/g, "口令")
    .replace(/测试口令/g, "口令")
    .replace(/标签为?[^，。；\s]+/g, "")
    .replace(/标签[:：]?[^，。；\s]+/g, "")
    .replace(/[，,；;：:]\s*([。.!！?？])/g, "$1")
    .replace(/\s{2,}/g, " ")
    .replace(/^[，,；;：:\s]+|[，,；;：:\s]+$/g, "")
    .trim();
}

export async function selectMemoriesForInjection(
  env: Env,
  input: { profile: KeyProfile; query: string; messages?: OpenAIChatMessage[] }
): Promise<MemoryApiRecord[]> {
  const mode = resolveInjectionMode(input.profile, env);
  if (mode === "none") return [];

  const namespace = input.profile.namespace;
  const contextualQuery = buildContextualMemoryQuery(input.messages ?? [], input.query);

  if (mode === "full") {
    const memories = await listMemoriesForInjection(env, {
      namespace,
      limit: 500
    });

    const primary = await filterAndCompressMemories(env, {
      query: input.query,
      memories
    });
    if (primary.length > 0 || !contextualQuery) return primary;
    return filterAndCompressMemories(env, {
      query: contextualQuery,
      memories
    });
  }

  const ragMemories = input.query.trim()
    ? await searchMemoriesForInjection(env, {
        namespace,
        query: input.query,
        topK: getTopK(env)
      })
    : [];

  if (mode === "rag") {
    const primary = await filterAndCompressMemories(env, {
      query: input.query,
      memories: ragMemories
    });
    if (primary.length > 0 || !contextualQuery) return primary;

    const contextualMemories = await searchMemoriesForInjection(env, {
      namespace,
      query: contextualQuery,
      topK: getTopK(env)
    });
    return filterAndCompressMemories(env, {
      query: contextualQuery,
      memories: contextualMemories
    });
  }

  const records = await listMemoriesForInjection(env, {
    namespace,
    limit: 500
  });
  const pinned = records.filter((record) => record.pinned);

  const primary = await filterAndCompressMemories(env, {
    query: input.query,
    memories: dedupeMemories([...pinned, ...ragMemories])
  });
  if (primary.length > 0 || !contextualQuery) return primary;

  const contextualMemories = await searchMemoriesForInjection(env, {
    namespace,
    query: contextualQuery,
    topK: getTopK(env)
  });
  return filterAndCompressMemories(env, {
    query: contextualQuery,
    memories: dedupeMemories([...pinned, ...contextualMemories])
  });
}

export function formatMemoryPatch(memories: MemoryApiRecord[]): string {
  if (memories.length === 0) return "";

  const lines = memories.flatMap((memory) => {
    const content = sanitizeMemoryContent(memory.content);
    if (!content) return [];
    const importance = memory.importance.toFixed(2);
    const pinned = memory.pinned ? "[pinned]" : "";
    const sourceMessageIds = memory.source_message_ids.filter((id) => /^\d+$/.test(id));
    const source = sourceMessageIds.length > 0 ? ` [source=${sourceMessageIds.join(",")}]` : "";
    return [`- [${memory.type}][importance=${importance}]${pinned} ${content}${source}`];
  });

  if (lines.length === 0) return "";

  return [
    "以下是你自然记得的长期记忆。只有在相关时使用，不要机械复述。",
    "不要说“根据记忆库”“系统记录”或暴露任何代理层实现。",
    "",
    "<memories>",
    ...lines,
    "</memories>"
  ].join("\n");
}

export function injectMemoryPatchAsSystemMessage(
  request: OpenAIChatRequest,
  memories: MemoryApiRecord[]
): OpenAIChatRequest {
  const patch = formatMemoryPatch(memories);
  if (!patch) return request;

  const memoryMessage: OpenAIChatMessage = {
    role: "system",
    content: patch
  };

  const messages = [...request.messages];
  let insertAt = 0;

  while (insertAt < messages.length && messages[insertAt].role === "system") {
    insertAt += 1;
  }

  messages.splice(insertAt, 0, memoryMessage);

  return {
    ...request,
    messages
  };
}
