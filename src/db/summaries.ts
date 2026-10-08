import type { MessageRecord, SummaryRecord } from "../types";
import { newId } from "../utils/ids";
import { nowIso } from "../utils/time";

interface ClassifiedSummaryRecord extends SummaryRecord {
  kind: "daily_handoff" | "long_term" | string;
  summary_date: string | null;
}

const SUMMARY_COLUMNS = `id, namespace, conversation_id, content, from_message_id, to_message_id,
                         message_count, vector_id, created_at, updated_at, kind, summary_date`;

// ---------------------------------------------------------------------------
// Get the latest summary for a namespace
// ---------------------------------------------------------------------------

export async function getLatestSummary(
  db: D1Database,
  namespace: string
): Promise<ClassifiedSummaryRecord | null> {
  const row = await db
    .prepare(
      `SELECT ${SUMMARY_COLUMNS}
       FROM summaries
       WHERE namespace = ?
       ORDER BY updated_at DESC
       LIMIT 1`
    )
    .bind(namespace)
    .first<ClassifiedSummaryRecord>();
  return row ?? null;
}

// ---------------------------------------------------------------------------
// Daily handoffs — one row per date; new conversations receive the latest two
// ---------------------------------------------------------------------------

export async function getLatestDailySummaryBundle(
  db: D1Database,
  namespace: string,
  limit = 2
): Promise<{ content: string; updated_at: string } | null> {
  const result = await db
    .prepare(
      `SELECT ${SUMMARY_COLUMNS}
       FROM summaries
       WHERE namespace = ?
         AND kind = 'daily_handoff'
         AND TRIM(content) <> ''
       ORDER BY summary_date DESC, updated_at DESC, id DESC
       LIMIT ?`
    )
    .bind(namespace, limit)
    .all<ClassifiedSummaryRecord>();
  const summaries = result.results ?? [];
  if (summaries.length === 0) return null;

  const latest = summaries[0];
  return {
    content: [...summaries].reverse().map((summary) => summary.content.trim()).join("\n\n"),
    updated_at: latest.updated_at,
  };
}

export async function upsertDailySummary(
  db: D1Database,
  input: {
    namespace: string;
    dateLabel: string;
    content: string;
    fromMessageId?: string | null;
    toMessageId?: string | null;
    messageCount?: number;
  }
): Promise<SummaryRecord> {
  const now = nowIso();
  const existing = await db
    .prepare(
      `SELECT ${SUMMARY_COLUMNS}
       FROM summaries
       WHERE namespace = ? AND kind = 'daily_handoff' AND summary_date = ?
       LIMIT 1`
    )
    .bind(input.namespace, input.dateLabel)
    .first<ClassifiedSummaryRecord>();

  if (existing) {
    await db
      .prepare(
        `UPDATE summaries
         SET content = ?, from_message_id = ?, to_message_id = ?,
             message_count = ?, updated_at = ?
         WHERE id = ?`
      )
      .bind(
        input.content,
        input.fromMessageId ?? null,
        input.toMessageId ?? null,
        input.messageCount ?? 0,
        now,
        existing.id
      )
      .run();

    return {
      ...existing,
      content: input.content,
      from_message_id: input.fromMessageId ?? null,
      to_message_id: input.toMessageId ?? null,
      message_count: input.messageCount ?? 0,
      updated_at: now,
    };
  }

  const id = newId("sum");
  const record: SummaryRecord = {
    id,
    namespace: input.namespace,
    conversation_id: null,
    content: input.content,
    from_message_id: input.fromMessageId ?? null,
    to_message_id: input.toMessageId ?? null,
    message_count: input.messageCount ?? 0,
    vector_id: null,
    created_at: now,
    updated_at: now,
  };

  await db
    .prepare(
      `INSERT INTO summaries (id, namespace, conversation_id, content, from_message_id,
                              to_message_id, message_count, vector_id, created_at, updated_at,
                              kind, summary_date)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'daily_handoff', ?)`
    )
    .bind(
      record.id,
      record.namespace,
      record.conversation_id,
      record.content,
      record.from_message_id,
      record.to_message_id,
      record.message_count,
      record.vector_id,
      record.created_at,
      record.updated_at,
      input.dateLabel
    )
    .run();

  return record;
}

// ---------------------------------------------------------------------------
// Legacy rolling long-term summary — kept separate from daily handoffs
// ---------------------------------------------------------------------------

export async function getLatestLongTermSummary(
  db: D1Database,
  namespace: string
): Promise<ClassifiedSummaryRecord | null> {
  const row = await db
    .prepare(
      `SELECT ${SUMMARY_COLUMNS}
       FROM summaries
       WHERE namespace = ? AND kind = 'long_term'
       ORDER BY updated_at DESC
       LIMIT 1`
    )
    .bind(namespace)
    .first<ClassifiedSummaryRecord>();
  return row ?? null;
}

export async function upsertLongTermSummary(
  db: D1Database,
  input: {
    namespace: string;
    content: string;
    fromMessageId?: string | null;
    toMessageId?: string | null;
    messageCount?: number;
  }
): Promise<SummaryRecord> {
  const now = nowIso();
  const existing = await getLatestLongTermSummary(db, input.namespace);

  if (existing) {
    await db
      .prepare(
        `UPDATE summaries
         SET content = ?, from_message_id = ?, to_message_id = ?,
             message_count = ?, updated_at = ?
         WHERE id = ?`
      )
      .bind(
        input.content,
        input.fromMessageId ?? null,
        input.toMessageId ?? null,
        input.messageCount ?? 0,
        now,
        existing.id
      )
      .run();

    return {
      ...existing,
      content: input.content,
      from_message_id: input.fromMessageId ?? null,
      to_message_id: input.toMessageId ?? null,
      message_count: input.messageCount ?? 0,
      updated_at: now,
    };
  }

  const id = newId("sum");
  const record: SummaryRecord = {
    id,
    namespace: input.namespace,
    conversation_id: null,
    content: input.content,
    from_message_id: input.fromMessageId ?? null,
    to_message_id: input.toMessageId ?? null,
    message_count: input.messageCount ?? 0,
    vector_id: null,
    created_at: now,
    updated_at: now,
  };

  await db
    .prepare(
      `INSERT INTO summaries (id, namespace, conversation_id, content, from_message_id,
                              to_message_id, message_count, vector_id, created_at, updated_at,
                              kind, summary_date)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'long_term', NULL)`
    )
    .bind(
      record.id,
      record.namespace,
      record.conversation_id,
      record.content,
      record.from_message_id,
      record.to_message_id,
      record.message_count,
      record.vector_id,
      record.created_at,
      record.updated_at
    )
    .run();

  return record;
}

// ---------------------------------------------------------------------------
// Count user/assistant messages after a given timestamp or message id
// ---------------------------------------------------------------------------

export async function countMessagesAfter(
  db: D1Database,
  namespace: string,
  afterCreatedAt: string | null
): Promise<number> {
  if (!afterCreatedAt) {
    // No previous summary — count all user/assistant messages
    const row = await db
      .prepare(
        `SELECT COUNT(*) as cnt FROM messages
         WHERE namespace = ? AND memory_active = 1 AND role IN ('user', 'assistant')`
      )
      .bind(namespace)
      .first<{ cnt: number }>();
    return row?.cnt ?? 0;
  }

  const row = await db
    .prepare(
      `SELECT COUNT(*) as cnt FROM messages
       WHERE namespace = ? AND memory_active = 1 AND role IN ('user', 'assistant') AND created_at > ?`
    )
    .bind(namespace, afterCreatedAt)
    .first<{ cnt: number }>();
  return row?.cnt ?? 0;
}

// ---------------------------------------------------------------------------
// Get a message's created_at by id (used to resolve summary cursor)
// ---------------------------------------------------------------------------

export async function getMessageCreatedAt(
  db: D1Database,
  namespace: string,
  messageId: string
): Promise<string | null> {
  const row = await db
    .prepare(
      `SELECT created_at FROM messages WHERE namespace = ? AND id = ?`
    )
    .bind(namespace, messageId)
    .first<{ created_at: string }>();
  return row?.created_at ?? null;
}

// ---------------------------------------------------------------------------
// List recent user/assistant messages for summary generation
// ---------------------------------------------------------------------------

export async function listRecentMessagesForSummary(
  db: D1Database,
  namespace: string,
  limit: number
): Promise<MessageRecord[]> {
  const result = await db
    .prepare(
      `SELECT id, conversation_id, namespace, role, content, source, created_at
       FROM messages
       WHERE namespace = ? AND memory_active = 1 AND role IN ('user', 'assistant')
       ORDER BY created_at DESC
       LIMIT ?`
    )
    .bind(namespace, limit)
    .all<MessageRecord>();

  // Reverse so oldest first
  return (result.results ?? []).reverse();
}
