CREATE TABLE IF NOT EXISTS garden_source_messages (
  namespace TEXT NOT NULL,
  source_message_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  delivery_status TEXT NOT NULL DEFAULT 'complete',
  active INTEGER NOT NULL DEFAULT 1,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (namespace, source_message_id)
);

CREATE INDEX IF NOT EXISTS idx_garden_source_messages_turn
  ON garden_source_messages(namespace, conversation_id, turn_id, active, created_at, source_message_id);

CREATE INDEX IF NOT EXISTS idx_garden_source_messages_time
  ON garden_source_messages(namespace, active, created_at, source_message_id);

CREATE TABLE IF NOT EXISTS garden_source_turns (
  namespace TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  first_message_at TEXT NOT NULL,
  revision INTEGER NOT NULL DEFAULT 1,
  processed_revision INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (namespace, conversation_id, turn_id)
);

CREATE INDEX IF NOT EXISTS idx_garden_source_turns_pending
  ON garden_source_turns(namespace, processed_revision, revision, first_message_at);
