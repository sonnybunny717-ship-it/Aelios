CREATE TABLE IF NOT EXISTS reply_edit_archives (
  namespace TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  archive_id TEXT NOT NULL,
  source_turn_id TEXT NOT NULL,
  selected_variant_id TEXT NOT NULL,
  selection_revision INTEGER NOT NULL,
  messages_json TEXT NOT NULL,
  draft_content TEXT NOT NULL,
  discarded_turns_json TEXT NOT NULL DEFAULT '[]',
  applied INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (namespace, conversation_id, archive_id)
);

CREATE INDEX IF NOT EXISTS idx_reply_edit_archives_conversation
  ON reply_edit_archives(namespace, conversation_id, created_at);
