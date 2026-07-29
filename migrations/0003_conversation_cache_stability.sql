-- Freeze the long-term summary per client conversation so background summary
-- refreshes do not invalidate a live chat's rolling prompt-cache prefix.
ALTER TABLE conversations ADD COLUMN summary_snapshot TEXT;
ALTER TABLE conversations ADD COLUMN summary_snapshot_source_updated_at TEXT;
ALTER TABLE conversations ADD COLUMN context_epoch INTEGER NOT NULL DEFAULT 0;
ALTER TABLE conversations ADD COLUMN window_summary TEXT;
ALTER TABLE conversations ADD COLUMN persona_snapshot_json TEXT;

-- Hash-only cache diagnostics: no prompt or message content is stored here.
ALTER TABLE usage_logs ADD COLUMN cache_diagnostics_json TEXT;
