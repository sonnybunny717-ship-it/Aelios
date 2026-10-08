ALTER TABLE summaries ADD COLUMN kind TEXT NOT NULL DEFAULT 'long_term';
ALTER TABLE summaries ADD COLUMN summary_date TEXT;

UPDATE summaries
SET kind = 'daily_handoff',
    summary_date = substr(content, 2, 10)
WHERE content GLOB '【[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]】*';

CREATE INDEX IF NOT EXISTS idx_summaries_namespace_kind_updated
ON summaries(namespace, kind, updated_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_summaries_daily_handoff_date
ON summaries(namespace, summary_date)
WHERE kind = 'daily_handoff' AND summary_date IS NOT NULL;
