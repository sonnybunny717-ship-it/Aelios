ALTER TABLE messages ADD COLUMN client_turn_id TEXT;
ALTER TABLE messages ADD COLUMN client_variant_id TEXT;
ALTER TABLE messages ADD COLUMN memory_active INTEGER NOT NULL DEFAULT 1;
CREATE INDEX IF NOT EXISTS idx_messages_client_variant
  ON messages(namespace, conversation_id, client_turn_id, client_variant_id);
CREATE TABLE IF NOT EXISTS reply_selections (
  namespace TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  variant_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  finalized INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(namespace, conversation_id, turn_id)
);
