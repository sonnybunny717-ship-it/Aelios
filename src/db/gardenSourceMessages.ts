export interface GardenSourceMessageEvent {
  sourceMessageId: string;
  conversationId: string;
  turnId: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  deliveryStatus: string;
  active: boolean;
  historical: boolean;
}

export interface GardenSourceMessageRecord {
  namespace: string;
  source_message_id: string;
  conversation_id: string;
  turn_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  delivery_status: string;
  active: number;
  updated_at: string;
}

export interface GardenSourceTurnRecord {
  namespace: string;
  conversation_id: string;
  turn_id: string;
  first_message_at: string;
  revision: number;
  processed_revision: number;
  updated_at: string;
}

export interface GardenSourceTurnBatchItem extends GardenSourceTurnRecord {
  messages: GardenSourceMessageRecord[];
  serializedChars: number;
}

const SQL_BATCH_SIZE = 50;

function turnKey(conversationId: string, turnId: string): string {
  return `${conversationId}\u0000${turnId}`;
}

async function runStatements(db: D1Database, statements: D1PreparedStatement[]): Promise<void> {
  for (let offset = 0; offset < statements.length; offset += SQL_BATCH_SIZE) {
    await db.batch(statements.slice(offset, offset + SQL_BATCH_SIZE));
  }
}

export async function syncGardenSourceMessages(
  db: D1Database,
  namespace: string,
  events: GardenSourceMessageEvent[]
): Promise<{ synced: number; touchedTurns: number }> {
  if (events.length === 0) return { synced: 0, touchedTurns: 0 };

  const sourceIds = [...new Set(events.map((event) => event.sourceMessageId))];
  const existing = new Map<string, Pick<GardenSourceMessageRecord, "source_message_id" | "conversation_id" | "turn_id">>();
  for (let offset = 0; offset < sourceIds.length; offset += SQL_BATCH_SIZE) {
    const ids = sourceIds.slice(offset, offset + SQL_BATCH_SIZE);
    const placeholders = ids.map(() => "?").join(", ");
    const result = await db
      .prepare(
        `SELECT source_message_id, conversation_id, turn_id
         FROM garden_source_messages
         WHERE namespace = ? AND source_message_id IN (${placeholders})`
      )
      .bind(namespace, ...ids)
      .all<Pick<GardenSourceMessageRecord, "source_message_id" | "conversation_id" | "turn_id">>();
    for (const row of result.results ?? []) existing.set(row.source_message_id, row);
  }

  const touchedTurns = new Map<string, {
    conversationId: string;
    turnId: string;
    firstMessageAt: string;
    dirty: boolean;
  }>();
  const touch = (conversationId: string, turnId: string, firstMessageAt: string, dirty: boolean) => {
    const key = turnKey(conversationId, turnId);
    const current = touchedTurns.get(key);
    touchedTurns.set(key, {
      conversationId,
      turnId,
      firstMessageAt: current && current.firstMessageAt < firstMessageAt ? current.firstMessageAt : firstMessageAt,
      dirty: Boolean(current?.dirty || dirty)
    });
  };

  const now = new Date().toISOString();
  const upserts: D1PreparedStatement[] = [];
  for (const event of events) {
    const old = existing.get(event.sourceMessageId);
    if (old && (old.conversation_id !== event.conversationId || old.turn_id !== event.turnId)) {
      touch(old.conversation_id, old.turn_id, event.createdAt, !event.historical);
    }
    touch(event.conversationId, event.turnId, event.createdAt, !event.historical);
    upserts.push(
      db.prepare(
        `INSERT INTO garden_source_messages (
           namespace, source_message_id, conversation_id, turn_id, role, content,
           created_at, delivery_status, active, updated_at
         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(namespace, source_message_id) DO UPDATE SET
           conversation_id = excluded.conversation_id,
           turn_id = excluded.turn_id,
           role = excluded.role,
           content = excluded.content,
           created_at = excluded.created_at,
           delivery_status = excluded.delivery_status,
           active = excluded.active,
           updated_at = excluded.updated_at`
      ).bind(
        namespace,
        event.sourceMessageId,
        event.conversationId,
        event.turnId,
        event.role,
        event.content,
        event.createdAt,
        event.deliveryStatus,
        event.active ? 1 : 0,
        now
      )
    );
  }
  await runStatements(db, upserts);

  const turnStatements: D1PreparedStatement[] = [];
  for (const turn of touchedTurns.values()) {
    if (turn.dirty) {
      turnStatements.push(
        db.prepare(
          `INSERT INTO garden_source_turns (
             namespace, conversation_id, turn_id, first_message_at,
             revision, processed_revision, updated_at
           ) VALUES (?, ?, ?, ?, 1, 0, ?)
           ON CONFLICT(namespace, conversation_id, turn_id) DO UPDATE SET
             first_message_at = MIN(garden_source_turns.first_message_at, excluded.first_message_at),
             revision = garden_source_turns.revision + 1,
             updated_at = excluded.updated_at`
        ).bind(namespace, turn.conversationId, turn.turnId, turn.firstMessageAt, now)
      );
    } else {
      turnStatements.push(
        db.prepare(
          `INSERT INTO garden_source_turns (
             namespace, conversation_id, turn_id, first_message_at,
             revision, processed_revision, updated_at
           ) VALUES (?, ?, ?, ?, 1, 1, ?)
           ON CONFLICT(namespace, conversation_id, turn_id) DO UPDATE SET
             first_message_at = MIN(garden_source_turns.first_message_at, excluded.first_message_at),
             updated_at = excluded.updated_at`
        ).bind(namespace, turn.conversationId, turn.turnId, turn.firstMessageAt, now)
      );
    }
  }
  await runStatements(db, turnStatements);

  return { synced: events.length, touchedTurns: touchedTurns.size };
}

export async function listPendingCompleteGardenTurns(
  db: D1Database,
  input: { namespace: string; limit: number }
): Promise<GardenSourceTurnRecord[]> {
  const result = await db
    .prepare(
      `SELECT t.*
       FROM garden_source_turns t
       WHERE t.namespace = ?
         AND t.processed_revision < t.revision
         AND EXISTS (
           SELECT 1 FROM garden_source_messages m
           WHERE m.namespace = t.namespace
             AND m.conversation_id = t.conversation_id
             AND m.turn_id = t.turn_id
             AND m.active = 1
             AND m.role = 'user'
         )
         AND EXISTS (
           SELECT 1 FROM garden_source_messages m
           WHERE m.namespace = t.namespace
             AND m.conversation_id = t.conversation_id
             AND m.turn_id = t.turn_id
             AND m.active = 1
             AND m.role = 'assistant'
             AND m.delivery_status IN ('complete', 'interrupted', 'truncated')
         )
         AND NOT EXISTS (
           SELECT 1 FROM garden_source_messages m
           WHERE m.namespace = t.namespace
             AND m.conversation_id = t.conversation_id
             AND m.turn_id = t.turn_id
             AND m.active = 1
             AND m.role = 'user'
             AND m.delivery_status IN ('pending', 'generating')
         )
       ORDER BY t.first_message_at ASC, t.conversation_id ASC, t.turn_id ASC
       LIMIT ?`
    )
    .bind(input.namespace, input.limit)
    .all<GardenSourceTurnRecord>();
  return result.results ?? [];
}

export async function listGardenTurnMessages(
  db: D1Database,
  input: { namespace: string; conversationId: string; turnId: string }
): Promise<GardenSourceMessageRecord[]> {
  const result = await db
    .prepare(
      `SELECT * FROM garden_source_messages
       WHERE namespace = ? AND conversation_id = ? AND turn_id = ? AND active = 1
         AND role IN ('user', 'assistant')
       ORDER BY created_at ASC, CAST(source_message_id AS INTEGER) ASC`
    )
    .bind(input.namespace, input.conversationId, input.turnId)
    .all<GardenSourceMessageRecord>();
  return result.results ?? [];
}

export async function listGardenSourceMessagesInRange(
  db: D1Database,
  input: {
    namespace: string;
    startCreatedAt: string;
    endCreatedAt: string;
    afterCreatedAt?: string | null;
    afterSourceMessageId?: string | null;
    limit: number;
  }
): Promise<GardenSourceMessageRecord[]> {
  let sql = `SELECT * FROM garden_source_messages
             WHERE namespace = ? AND active = 1
               AND role IN ('user', 'assistant')
               AND created_at >= ? AND created_at < ?`;
  const binds: unknown[] = [input.namespace, input.startCreatedAt, input.endCreatedAt];
  if (input.afterCreatedAt && input.afterSourceMessageId) {
    sql += ` AND (created_at > ? OR (created_at = ? AND CAST(source_message_id AS INTEGER) > ?))`;
    binds.push(input.afterCreatedAt, input.afterCreatedAt, Number(input.afterSourceMessageId));
  }
  sql += ` ORDER BY created_at ASC, CAST(source_message_id AS INTEGER) ASC LIMIT ?`;
  binds.push(input.limit);
  const result = await db.prepare(sql).bind(...binds).all<GardenSourceMessageRecord>();
  return result.results ?? [];
}

export async function areGardenTurnsCurrent(
  db: D1Database,
  namespace: string,
  turns: Array<Pick<GardenSourceTurnRecord, "conversation_id" | "turn_id" | "revision">>
): Promise<boolean> {
  for (const turn of turns) {
    const row = await db
      .prepare(
        `SELECT revision FROM garden_source_turns
         WHERE namespace = ? AND conversation_id = ? AND turn_id = ?`
      )
      .bind(namespace, turn.conversation_id, turn.turn_id)
      .first<{ revision: number }>();
    if (!row || row.revision !== turn.revision) return false;
  }
  return true;
}

export async function markGardenTurnsProcessed(
  db: D1Database,
  namespace: string,
  turns: Array<Pick<GardenSourceTurnRecord, "conversation_id" | "turn_id" | "revision">>
): Promise<boolean> {
  const now = new Date().toISOString();
  const statements = turns.map((turn) => db.prepare(
    `UPDATE garden_source_turns
     SET processed_revision = ?, updated_at = ?
     WHERE namespace = ? AND conversation_id = ? AND turn_id = ? AND revision = ?`
  ).bind(turn.revision, now, namespace, turn.conversation_id, turn.turn_id, turn.revision));
  for (let offset = 0; offset < statements.length; offset += SQL_BATCH_SIZE) {
    const results = await db.batch(statements.slice(offset, offset + SQL_BATCH_SIZE));
    if (results.some((result) => (result.meta.changes ?? 0) !== 1)) return false;
  }
  return true;
}

export async function areGardenSourceMessagesActive(
  db: D1Database,
  namespace: string,
  sourceMessageIds: string[]
): Promise<boolean> {
  const ids = [...new Set(sourceMessageIds)];
  let count = 0;
  for (let offset = 0; offset < ids.length; offset += SQL_BATCH_SIZE) {
    const batch = ids.slice(offset, offset + SQL_BATCH_SIZE);
    const placeholders = batch.map(() => "?").join(", ");
    const row = await db.prepare(
      `SELECT COUNT(*) AS count FROM garden_source_messages
       WHERE namespace = ? AND active = 1 AND source_message_id IN (${placeholders})`
    ).bind(namespace, ...batch).first<{ count: number }>();
    count += row?.count ?? 0;
  }
  return count === ids.length;
}
