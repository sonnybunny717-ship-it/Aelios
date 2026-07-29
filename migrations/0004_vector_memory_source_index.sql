CREATE TABLE IF NOT EXISTS vector_memory_sources (
  namespace TEXT NOT NULL,
  memory_id TEXT NOT NULL,
  vector_id TEXT NOT NULL,
  type TEXT NOT NULL,
  source_message_ids TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (namespace, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_vector_memory_sources_namespace
ON vector_memory_sources(namespace);

CREATE INDEX IF NOT EXISTS idx_vector_memory_sources_vector
ON vector_memory_sources(vector_id);
