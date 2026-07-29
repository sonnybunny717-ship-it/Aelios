import { nowIso } from "../utils/time";

export interface VectorMemorySourceRecord {
  namespace: string;
  memory_id: string;
  vector_id: string;
  type: string;
  source_message_ids: string;
}

const WRITE_BATCH_SIZE = 100;

export async function upsertVectorMemorySources(
  db: D1Database,
  records: VectorMemorySourceRecord[],
): Promise<void> {
  for (let index = 0; index < records.length; index += WRITE_BATCH_SIZE) {
    const batch = records.slice(index, index + WRITE_BATCH_SIZE).map((record) =>
      db
        .prepare(
          `INSERT INTO vector_memory_sources (
             namespace, memory_id, vector_id, type, source_message_ids, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(namespace, memory_id) DO UPDATE SET
             vector_id = excluded.vector_id,
             type = excluded.type,
             source_message_ids = excluded.source_message_ids,
             updated_at = excluded.updated_at`,
        )
        .bind(
          record.namespace,
          record.memory_id,
          record.vector_id,
          record.type,
          record.source_message_ids,
          nowIso(),
        ),
    );
    if (batch.length > 0) await db.batch(batch);
  }
}

export async function upsertVectorMemorySource(
  db: D1Database,
  record: VectorMemorySourceRecord,
): Promise<void> {
  await upsertVectorMemorySources(db, [record]);
}

export async function deleteVectorMemorySource(
  db: D1Database,
  namespace: string,
  memoryId: string,
): Promise<void> {
  await db
    .prepare("DELETE FROM vector_memory_sources WHERE namespace = ? AND memory_id = ?")
    .bind(namespace, memoryId)
    .run();
}

export async function listVectorMemorySources(
  db: D1Database,
  namespace: string,
): Promise<VectorMemorySourceRecord[]> {
  const result = await db
    .prepare(
      `SELECT namespace, memory_id, vector_id, type, source_message_ids
       FROM vector_memory_sources
       WHERE namespace = ?`,
    )
    .bind(namespace)
    .all<VectorMemorySourceRecord>();
  return result.results ?? [];
}
