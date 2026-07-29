import { authenticate } from "../auth/apiKey";
import { requireScope } from "../auth/scopes";
import { readCursor, writeCursor } from "../db/retention";
import {
  listVectorMemorySources,
  upsertVectorMemorySources,
  type VectorMemorySourceRecord,
} from "../db/vectorMemorySources";
import { listVectorMemories } from "../memory/vectorStore";
import type { Env } from "../types";
import { json, openAiError } from "../utils/json";

type MessageRow = {
  id: string;
  created_at: string;
};

type DerivedMemoryRow = {
  id: string;
  type: string;
  vector_id: string | null;
  source_message_ids: string | null;
};

type SummaryRow = {
  id: string;
  conversation_id: string | null;
  from_message_id: string | null;
  to_message_id: string | null;
  vector_id: string | null;
  from_created_at: string | null;
  to_created_at: string | null;
};

const DELETE_BATCH_SIZE = 100;
const VECTOR_SOURCE_INDEX_CURSOR_PREFIX = "vector_memory_sources_backfill:v1";

function parseStringArray(value: string | null | undefined): string[] {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : [];
  } catch {
    return [];
  }
}

export function normalizeConversationId(rawId: string, namespace: string): string | null {
  let decoded: string;
  try {
    decoded = decodeURIComponent(rawId).trim();
  } catch {
    return null;
  }
  if (!decoded || decoded.includes("/")) return null;
  return decoded.startsWith(`${namespace}:`) ? decoded : `${namespace}:${decoded}`;
}

export function sourceIdsOverlap(sourceIds: string[], deletedIds: Set<string>): boolean {
  return sourceIds.some((id) => deletedIds.has(id));
}

function chunks<T>(values: T[], size = DELETE_BATCH_SIZE): T[][] {
  const result: T[][] = [];
  for (let index = 0; index < values.length; index += size) {
    result.push(values.slice(index, index + size));
  }
  return result;
}

function deleteByIds(
  db: D1Database,
  table: "memories" | "memory_events" | "summaries",
  column: "id" | "memory_id",
  namespace: string,
  ids: string[],
): D1PreparedStatement[] {
  return chunks(ids).map((batch) => {
    const placeholders = batch.map(() => "?").join(", ");
    return db
      .prepare(`DELETE FROM ${table} WHERE namespace = ? AND ${column} IN (${placeholders})`)
      .bind(namespace, ...batch);
  });
}

function summaryContainsDeletedMessage(
  summary: SummaryRow,
  conversationId: string,
  deletedIds: Set<string>,
  deletedTimes: string[],
): boolean {
  if (summary.conversation_id === conversationId) return true;
  if (summary.from_message_id && deletedIds.has(summary.from_message_id)) return true;
  if (summary.to_message_id && deletedIds.has(summary.to_message_id)) return true;
  if (!summary.from_created_at || !summary.to_created_at) return false;
  return deletedTimes.some(
    (createdAt) => createdAt >= summary.from_created_at! && createdAt <= summary.to_created_at!,
  );
}

async function ensureVectorMemorySourceIndex(
  env: Env,
  namespace: string,
): Promise<void> {
  if (!env.VECTORIZE) return;
  const cursorName = `${VECTOR_SOURCE_INDEX_CURSOR_PREFIX}:${namespace}`;
  if (await readCursor(env.DB, cursorName)) return;

  let cursor: string | undefined;
  do {
    const page = await listVectorMemories(env, {
      namespace,
      count: 1000,
      ...(cursor ? { cursor } : {}),
    });
    await upsertVectorMemorySources(env.DB, page.data.map((memory) => ({
      namespace: memory.namespace,
      memory_id: memory.id,
      vector_id: memory.vector_id || "",
      type: memory.type,
      source_message_ids: JSON.stringify(memory.source_message_ids),
    })).filter((record) => record.vector_id));
    cursor = page.hasMore && page.cursor ? page.cursor : undefined;
  } while (cursor);

  await writeCursor(env.DB, cursorName, new Date().toISOString());
}

async function listDerivedVectorMemories(
  env: Env,
  namespace: string,
  deletedIds: Set<string>,
): Promise<VectorMemorySourceRecord[]> {
  if (!env.VECTORIZE || deletedIds.size === 0) return [];
  await ensureVectorMemorySourceIndex(env, namespace);
  return (await listVectorMemorySources(env.DB, namespace)).filter((memory) =>
    sourceIdsOverlap(parseStringArray(memory.source_message_ids), deletedIds),
  );
}

async function deleteVectors(env: Env, ids: string[]): Promise<void> {
  if (!env.VECTORIZE) return;
  const uniqueIds = [...new Set(ids.filter(Boolean))];
  for (const batch of chunks(uniqueIds)) {
    await env.VECTORIZE.deleteByIds(batch);
  }
}

export async function handleDeleteConversation(request: Request, env: Env): Promise<Response> {
  const auth = await authenticate(request, env);
  if (!auth.ok) return openAiError("Unauthorized", 401, "authentication_error");

  const scopeError = requireScope(auth.profile, "memory:write");
  if (scopeError) return scopeError;

  const path = new URL(request.url).pathname;
  const rawId = path.slice("/v1/conversations/".length);
  const conversationId = normalizeConversationId(rawId, auth.profile.namespace);
  if (!conversationId) return openAiError("Invalid conversation id", 400);

  const conversation = await env.DB
    .prepare("SELECT id FROM conversations WHERE id = ? AND namespace = ?")
    .bind(conversationId, auth.profile.namespace)
    .first<{ id: string }>();

  // 删除接口保持幂等：从未向 Aelios 发过消息的空会话也可以在 Garden 正常删除。
  if (!conversation) {
    return json({ data: { id: conversationId, deleted: false, already_absent: true } });
  }

  const messageResult = await env.DB
    .prepare(
      `SELECT id, created_at
       FROM messages
       WHERE conversation_id = ? AND namespace = ?
       ORDER BY created_at ASC`,
    )
    .bind(conversationId, auth.profile.namespace)
    .all<MessageRow>();
  const messages = messageResult.results ?? [];
  const messageIds = messages.map((message) => message.id);
  const deletedIds = new Set(messageIds);
  const deletedTimes = messages.map((message) => message.created_at);

  const memoryResult = await env.DB
    .prepare(
      `SELECT id, type, vector_id, source_message_ids
       FROM memories
       WHERE namespace = ? AND source_message_ids IS NOT NULL`,
    )
    .bind(auth.profile.namespace)
    .all<DerivedMemoryRow>();
  const d1Memories = (memoryResult.results ?? []).filter((memory) =>
    sourceIdsOverlap(parseStringArray(memory.source_message_ids), deletedIds),
  );

  const summaryResult = await env.DB
    .prepare(
      `SELECT s.id, s.conversation_id, s.from_message_id, s.to_message_id, s.vector_id,
              fm.created_at AS from_created_at, tm.created_at AS to_created_at
       FROM summaries s
       LEFT JOIN messages fm ON fm.id = s.from_message_id
       LEFT JOIN messages tm ON tm.id = s.to_message_id
       WHERE s.namespace = ?`,
    )
    .bind(auth.profile.namespace)
    .all<SummaryRow>();
  const summaries = (summaryResult.results ?? []).filter((summary) =>
    summaryContainsDeletedMessage(summary, conversationId, deletedIds, deletedTimes),
  );

  let vectorMemories: VectorMemorySourceRecord[];
  try {
    vectorMemories = await listDerivedVectorMemories(env, auth.profile.namespace, deletedIds);
    await deleteVectors(env, [
      ...d1Memories.map((memory) => memory.vector_id || ""),
      ...vectorMemories.map((memory) => memory.vector_id || ""),
      ...summaries.map((summary) => summary.vector_id || ""),
    ]);
  } catch {
    return openAiError("Failed to delete derived vector memories", 503, "memory_delete_error");
  }

  const d1MemoryIds = d1Memories.map((memory) => memory.id);
  const allMemoryIds = [...new Set([
    ...d1MemoryIds,
    ...vectorMemories.map((memory) => memory.memory_id),
  ])];
  const summaryIds = summaries.map((summary) => summary.id);
  const statements: D1PreparedStatement[] = [
    env.DB
      .prepare(
        `DELETE FROM usage_logs
         WHERE namespace = ?
           AND message_id IN (
             SELECT id FROM messages WHERE conversation_id = ? AND namespace = ?
           )`,
      )
      .bind(auth.profile.namespace, conversationId, auth.profile.namespace),
    ...deleteByIds(env.DB, "memory_events", "memory_id", auth.profile.namespace, allMemoryIds),
    ...deleteByIds(env.DB, "memories", "id", auth.profile.namespace, d1MemoryIds),
    ...deleteByIds(env.DB, "summaries", "id", auth.profile.namespace, summaryIds),
  ];
  statements.push(...chunks(vectorMemories.map((memory) => memory.memory_id)).map((batch) => {
    const placeholders = batch.map(() => "?").join(", ");
    return env.DB
      .prepare(
        `DELETE FROM vector_memory_sources
         WHERE namespace = ? AND memory_id IN (${placeholders})`,
      )
      .bind(auth.profile.namespace, ...batch);
  }));

  if (summaries.length > 0) {
    // 已冻结到其他窗口的旧长期摘要也必须失效，避免被删除的内容继续出现在提示词中。
    statements.push(
      env.DB
        .prepare(
          `UPDATE conversations
           SET summary_snapshot = NULL, summary_snapshot_source_updated_at = NULL
           WHERE namespace = ? AND id <> ?`,
        )
        .bind(auth.profile.namespace, conversationId),
    );
  }

  if (d1Memories.some((memory) => memory.type === "identity" || memory.type === "persona")
      || vectorMemories.some((memory) => memory.type === "identity" || memory.type === "persona")) {
    statements.push(
      env.DB
        .prepare(
          `UPDATE conversations
           SET persona_snapshot_json = NULL
           WHERE namespace = ? AND id <> ?`,
        )
        .bind(auth.profile.namespace, conversationId),
    );
  }

  statements.push(
    env.DB
      .prepare("DELETE FROM messages WHERE conversation_id = ? AND namespace = ?")
      .bind(conversationId, auth.profile.namespace),
    env.DB
      .prepare("DELETE FROM conversations WHERE id = ? AND namespace = ?")
      .bind(conversationId, auth.profile.namespace),
  );
  await env.DB.batch(statements);

  return json({
    data: {
      id: conversationId,
      deleted: true,
      messages: messageIds.length,
      memories: allMemoryIds.length,
      summaries: summaryIds.length,
      vectors: new Set([
        ...d1Memories.map((memory) => memory.vector_id).filter(Boolean),
        ...vectorMemories.map((memory) => memory.vector_id).filter(Boolean),
        ...summaries.map((summary) => summary.vector_id).filter(Boolean),
      ]).size,
    },
  });
}
