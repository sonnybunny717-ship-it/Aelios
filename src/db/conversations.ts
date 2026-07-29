import type { Conversation } from "../types";
import { newId } from "../utils/ids";
import { nowIso } from "../utils/time";

// ── Processing cursor (per-namespace memory extraction checkpoint) ──

export interface ProcessingCursor {
  name: string;
  value: string;
  updated_at: string;
}

export interface ConversationContextState {
  context_epoch: number;
  summary_snapshot: string | null;
  summary_snapshot_source_updated_at: string | null;
  window_summary: string | null;
  persona_snapshot_json: string | null;
}

export async function getProcessingCursor(
  db: D1Database,
  namespace: string
): Promise<ProcessingCursor | null> {
  const name = `memory_extract_cursor:${namespace}`;
  const row = await db
    .prepare("SELECT name, value, updated_at FROM processing_cursors WHERE name = ?")
    .bind(name)
    .first<ProcessingCursor>();
  return row ?? null;
}

export async function setProcessingCursor(
  db: D1Database,
  namespace: string,
  value: string
): Promise<void> {
  const name = `memory_extract_cursor:${namespace}`;
  const now = nowIso();
  await db
    .prepare(
      `INSERT INTO processing_cursors (name, value, updated_at) VALUES (?, ?, ?)
       ON CONFLICT(name) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at`
    )
    .bind(name, value, now)
    .run();
}

export async function getOrCreateConversation(
  db: D1Database,
  input: { namespace: string; id?: string }
): Promise<Conversation> {
  const id = input.id || `${input.namespace}:default`;
  const existing = await db
    .prepare(
      `SELECT id, namespace, summary_snapshot, summary_snapshot_source_updated_at,
              context_epoch, window_summary, persona_snapshot_json,
              created_at, updated_at
       FROM conversations
       WHERE id = ?`
    )
    .bind(id)
    .first<Conversation>();

  if (existing) return existing;

  const now = nowIso();
  const conversation: Conversation = {
    id,
    namespace: input.namespace,
    summary_snapshot: null,
    summary_snapshot_source_updated_at: null,
    context_epoch: 0,
    window_summary: null,
    persona_snapshot_json: null,
    created_at: now,
    updated_at: now
  };

  await db
    .prepare("INSERT INTO conversations (id, namespace, created_at, updated_at) VALUES (?, ?, ?, ?)")
    .bind(conversation.id, conversation.namespace, conversation.created_at, conversation.updated_at)
    .run();

  return conversation;
}

export async function initializeConversationContextState(
  db: D1Database,
  input: {
    conversationId: string;
    namespace: string;
    summarySnapshot: string;
    summarySnapshotSourceUpdatedAt: string | null;
    personaSnapshotJson: string;
  }
): Promise<void> {
  const now = nowIso();
  await db
    .prepare(
      `UPDATE conversations
       SET summary_snapshot = ?,
           summary_snapshot_source_updated_at = ?,
           persona_snapshot_json = ?,
           updated_at = ?
       WHERE id = ? AND namespace = ? AND persona_snapshot_json IS NULL`
    )
    .bind(
      input.summarySnapshot,
      input.summarySnapshotSourceUpdatedAt,
      input.personaSnapshotJson,
      now,
      input.conversationId,
      input.namespace
    )
    .run();
}

export async function getConversationContextState(
  db: D1Database,
  conversationId: string,
  namespace: string,
): Promise<ConversationContextState | null> {
  const row = await db
    .prepare(
      `SELECT context_epoch, summary_snapshot, summary_snapshot_source_updated_at,
              window_summary, persona_snapshot_json
       FROM conversations
       WHERE id = ? AND namespace = ?`
    )
    .bind(conversationId, namespace)
    .first<ConversationContextState>();
  return row ?? null;
}

export async function advanceConversationContextState(
  db: D1Database,
  input: {
    conversationId: string;
    namespace: string;
    epoch: number;
    windowSummary: string;
    summarySnapshot: string;
    summarySnapshotSourceUpdatedAt: string | null;
    personaSnapshotJson: string;
  },
): Promise<void> {
  await db
    .prepare(
      `UPDATE conversations
       SET context_epoch = ?,
           window_summary = ?,
           summary_snapshot = ?,
           summary_snapshot_source_updated_at = ?,
           persona_snapshot_json = ?,
           updated_at = ?
       WHERE id = ? AND namespace = ? AND context_epoch < ?`
    )
    .bind(
      input.epoch,
      input.windowSummary,
      input.summarySnapshot,
      input.summarySnapshotSourceUpdatedAt,
      input.personaSnapshotJson,
      nowIso(),
      input.conversationId,
      input.namespace,
      input.epoch,
    )
    .run();
}
