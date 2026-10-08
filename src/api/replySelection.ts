import { authenticate } from "../auth/apiKey";
import { requireScope } from "../auth/scopes";
import { parseReplyVariant } from "../db/replyVariants";
import { normalizeConversationId, invalidateMessageDerivatives } from "./conversations";
import { newId } from "../utils/ids";
import type { Env } from "../types";
import { json, openAiError } from "../utils/json";

const ARCHIVE_ID = /^[a-zA-Z0-9_-]{1,100}$/;

function editArchiveId(value: unknown): string | null {
  return typeof value === "string" && ARCHIVE_ID.test(value) ? value : null;
}

function discardedTurnIds(value: unknown, sourceTurnId: string): string[] | null {
  if (!Array.isArray(value) || value.length > 1000) return null;
  const ids = [...new Set(value)];
  return ids.every(id => typeof id === "string" && ARCHIVE_ID.test(id) && id !== sourceTurnId)
    ? ids as string[]
    : null;
}

async function persistMissingAssistantCandidate(
  body: Record<string, unknown>,
  env: Env,
  namespace: string,
  conversationId: string,
  turnId: string,
  variantId: string,
): Promise<boolean> {
  if (typeof body.assistantContent !== "string") return false;
  const assistantContent = body.assistantContent;
  const assistantReasoning = typeof body.assistantReasoning === "string" ? body.assistantReasoning : "";
  if (!assistantContent.trim() && !assistantReasoning.trim()) return false;
  const finishReason = typeof body.deliveryStatus === "string" && body.deliveryStatus
    ? body.deliveryStatus
    : null;
  const inserted = await env.DB.prepare(`INSERT INTO messages (
      id, conversation_id, namespace, role, content, reasoning_content, source,
      upstream_model, upstream_provider, request_model, stream, finish_reason,
      created_at, client_turn_id, client_variant_id, memory_active
    ) SELECT ?, candidate.conversation_id, candidate.namespace, 'assistant', ?, ?, candidate.source,
      candidate.upstream_model, candidate.upstream_provider, candidate.request_model, candidate.stream, ?,
      ?, candidate.client_turn_id, candidate.client_variant_id, 0
    FROM messages candidate
    WHERE candidate.namespace = ? AND candidate.conversation_id = ?
      AND candidate.client_turn_id = ? AND candidate.client_variant_id = ? AND candidate.role = 'user'
      AND NOT EXISTS (SELECT 1 FROM messages existing
        WHERE existing.namespace = candidate.namespace
          AND existing.conversation_id = candidate.conversation_id
          AND existing.client_turn_id = candidate.client_turn_id
          AND existing.client_variant_id = candidate.client_variant_id
          AND existing.role = 'assistant')
    ORDER BY candidate.created_at DESC LIMIT 1`)
    .bind(
      newId("msg"), assistantContent, assistantReasoning || null, finishReason,
      new Date().toISOString(), namespace, conversationId, turnId, variantId,
    ).run();
  if (inserted.meta.changes === 1) return true;
  const existing = await env.DB.prepare(`SELECT id FROM messages
    WHERE namespace = ? AND conversation_id = ? AND client_turn_id = ?
      AND client_variant_id = ? AND role = 'assistant' LIMIT 1`)
    .bind(namespace, conversationId, turnId, variantId).first<{ id: string }>();
  return Boolean(existing);
}

async function applyArchivedEdit(
  env: Env,
  namespace: string,
  conversationId: string,
  sourceTurnId: string,
  discardedTurns: string[],
): Promise<void> {
  const turnIds = [sourceTurnId, ...discardedTurns];
  const obsolete: Array<{ id: string; created_at: string }> = [];
  for (const turnId of turnIds) {
    const rows = await env.DB.prepare(`SELECT id, created_at FROM messages
      WHERE namespace = ? AND conversation_id = ? AND client_turn_id = ?`)
      .bind(namespace, conversationId, turnId).all<{ id: string; created_at: string }>();
    obsolete.push(...(rows.results ?? []));
  }
  const statements = await invalidateMessageDerivatives(env, namespace, conversationId, obsolete);
  for (const turnId of turnIds) {
    statements.push(
      env.DB.prepare(`DELETE FROM usage_logs WHERE namespace = ? AND message_id IN (
        SELECT id FROM messages WHERE namespace = ? AND conversation_id = ? AND client_turn_id = ?)`)
        .bind(namespace, namespace, conversationId, turnId),
      env.DB.prepare(`DELETE FROM messages WHERE namespace = ? AND conversation_id = ? AND client_turn_id = ?`)
        .bind(namespace, conversationId, turnId),
      env.DB.prepare(`DELETE FROM reply_selections WHERE namespace = ? AND conversation_id = ? AND turn_id = ?`)
        .bind(namespace, conversationId, turnId),
    );
  }
  await env.DB.batch(statements);
}

async function handleEditArchiveAction(
  body: Record<string, unknown>,
  env: Env,
  namespace: string,
  conversationId: string,
): Promise<Response | null> {
  const action = String(body.action || "");
  if (!['archive_edit', 'update_edit_draft', 'delete_edit_archive'].includes(action)) return null;
  const archiveId = editArchiveId(body.archiveId);
  if (!archiveId) return openAiError("Invalid edit archive", 400);

  if (action === 'delete_edit_archive' || action === 'update_edit_draft') {
    return json({ ok: true, archiveId });
  }

  const draftContent = typeof body.draftContent === 'string' ? body.draftContent.trim() : '';
  if (!draftContent) return openAiError("Invalid edited prompt", 400);

  let variant;
  try { variant = parseReplyVariant(body); }
  catch { return openAiError("Invalid reply variant", 400); }
  if (!variant) return openAiError("Missing reply variant", 400);
  const revision = Number(body.revision);
  if (!Number.isSafeInteger(revision) || revision < 1) return openAiError("Invalid revision", 400);
  const discardedTurns = discardedTurnIds(body.discardedTurns ?? [], variant.turnId);
  if (!discardedTurns) return openAiError("Invalid discarded turns", 400);

  const current = await env.DB.prepare(`SELECT revision, variant_id FROM reply_selections
    WHERE namespace = ? AND conversation_id = ? AND turn_id = ?`)
    .bind(namespace, conversationId, variant.turnId)
    .first<{ revision: number; variant_id: string }>();
  if (!current) {
    const remaining = await env.DB.prepare(`SELECT id FROM messages
      WHERE namespace = ? AND conversation_id = ? AND client_turn_id = ? LIMIT 1`)
      .bind(namespace, conversationId, variant.turnId).first<{ id: string }>();
    if (!remaining) return json({ ok: true, archiveId });
    return openAiError("Stale reply selection", 409);
  }
  if (current.revision !== revision || current.variant_id !== variant.variantId) {
    return openAiError("Stale reply selection", 409);
  }
  const assistant = await env.DB.prepare(`SELECT id FROM messages
    WHERE namespace = ? AND conversation_id = ? AND client_turn_id = ? AND role = 'assistant' LIMIT 1`)
    .bind(namespace, conversationId, variant.turnId).first<{ id: string }>();
  if (!assistant) return openAiError("Reply is not persisted yet", 409);

  await applyArchivedEdit(env, namespace, conversationId, variant.turnId, discardedTurns);
  return json({ ok: true, archiveId });
}

export async function handleReplySelection(request: Request, env: Env): Promise<Response> {
  const auth = await authenticate(request, env);
  if (!auth.ok) return openAiError("Unauthorized", 401, "authentication_error");
  const denied = requireScope(auth.profile, "memory:write");
  if (denied) return denied;
  const namespace = auth.profile.namespace;
  const rawId = new URL(request.url).pathname.slice("/v1/conversations/".length, -"/reply-selection".length);
  const conversationId = normalizeConversationId(rawId, namespace);
  if (!conversationId) return openAiError("Invalid conversation id", 400);
  if (request.method === "GET") {
    const url = new URL(request.url);
    const turnId = url.searchParams.get("turnId");
    const variantId = url.searchParams.get("variantId");
    if (turnId == null && variantId == null) return json({ version: 2, editArchives: true, completedCandidate: true });
    let variant;
    try { variant = parseReplyVariant({ turnId, variantId }); }
    catch { return openAiError("Invalid reply variant", 400); }
    if (!variant) return openAiError("Invalid reply variant", 400);
    const candidate = await env.DB.prepare(`SELECT
      id, content, reasoning_content, finish_reason, upstream_model, upstream_provider,
      request_model, token_input, token_output, cache_read_tokens,
      cache_creation_tokens, raw_usage_json, created_at
      FROM messages WHERE namespace = ? AND conversation_id = ?
      AND client_turn_id = ? AND client_variant_id = ? AND role = 'assistant'
      ORDER BY created_at DESC LIMIT 1`)
      .bind(namespace, conversationId, variant.turnId, variant.variantId)
      .first<{
        id: string;
        content: string;
        reasoning_content: string | null;
        finish_reason: string | null;
        upstream_model: string | null;
        upstream_provider: string | null;
        request_model: string | null;
        token_input: number | null;
        token_output: number | null;
        cache_read_tokens: number | null;
        cache_creation_tokens: number | null;
        raw_usage_json: string | null;
        created_at: string;
      }>();
    if (!candidate) return json({ version: 2, editArchives: true, completedCandidate: true, candidate: null });
    let rawUsage: Record<string, unknown> = {};
    try { rawUsage = JSON.parse(candidate.raw_usage_json || "{}") as Record<string, unknown>; }
    catch {}
    return json({
      version: 2,
      editArchives: true,
      completedCandidate: true,
      candidate: {
        messageId: candidate.id,
        turnId: variant.turnId,
        variantId: variant.variantId,
        content: candidate.content,
        reasoningContent: candidate.reasoning_content || "",
        finishReason: candidate.finish_reason,
        model: candidate.request_model || candidate.upstream_model || "",
        provider: candidate.upstream_provider || "",
        createdAt: candidate.created_at,
        usage: {
          ...rawUsage,
          prompt_tokens: candidate.token_input ?? rawUsage.prompt_tokens ?? rawUsage.input_tokens ?? 0,
          completion_tokens: candidate.token_output ?? rawUsage.completion_tokens ?? rawUsage.output_tokens ?? 0,
          cache_read_input_tokens: candidate.cache_read_tokens ?? rawUsage.cache_read_input_tokens ?? 0,
          cache_creation_input_tokens: candidate.cache_creation_tokens ?? rawUsage.cache_creation_input_tokens ?? 0,
        },
      },
    });
  }
  let body: Record<string, unknown>;
  try { body = await request.json() as Record<string, unknown>; }
  catch { return openAiError("Invalid JSON", 400); }
  const editArchiveResponse = await handleEditArchiveAction(body, env, namespace, conversationId);
  if (editArchiveResponse) return editArchiveResponse;
  let variant;
  try { variant = parseReplyVariant(body); }
  catch { return openAiError("Invalid reply variant", 400); }
  if (!variant) return openAiError("Missing reply variant", 400);
  const { turnId, variantId } = variant;
  const revision = Number(body.revision);
  if (!Number.isSafeInteger(revision) || revision < 1) return openAiError("Invalid revision", 400);
  const currentQuery = env.DB.prepare(`SELECT revision, variant_id, finalized FROM reply_selections
    WHERE namespace = ? AND conversation_id = ? AND turn_id = ?`)
    .bind(namespace, conversationId, turnId);
  const candidateQuery = env.DB.prepare(`SELECT id FROM messages WHERE namespace = ? AND conversation_id = ?
    AND client_turn_id = ? AND client_variant_id = ? AND role = 'assistant' LIMIT 1`)
    .bind(namespace, conversationId, turnId, variantId);
  // Selection reads share one D1 round trip; all revision checks still precede writes.
  const selectionReads = body.action === 'select' ? await env.DB.batch([currentQuery, candidateQuery]) : null;
  const current = selectionReads
    ? selectionReads[0].results[0] as { revision: number; variant_id: string; finalized: number } | undefined
    : await currentQuery.first<{ revision: number; variant_id: string; finalized: number }>();
  if (current && (current.revision > revision || (current.revision === revision && current.variant_id !== variantId))) {
    return openAiError("Stale reply selection", 409);
  }
  if (current?.revision === revision) {
    if (body.action === 'amend') {
      if (!['user', 'assistant'].includes(String(body.role)) || (body.content !== null && typeof body.content !== 'string')) return openAiError('Invalid amendment', 400);
      const rows = (await env.DB.prepare(`SELECT content FROM messages WHERE namespace = ? AND conversation_id = ?
        AND client_turn_id = ? AND client_variant_id = ? AND role = ?`)
        .bind(namespace, conversationId, turnId, variantId, body.role).all<{ content: string }>()).results;
      if (body.content === null ? rows.length !== 0 : rows.length !== 1 || rows[0].content !== body.content) return openAiError('Conflicting amendment', 409);
    }
    if (body.action === 'select' && Boolean(current.finalized) !== (body.finalized === true)) return openAiError("Conflicting reply selection", 409);
    return json({ ok: true, revision });
  }

  if (body.action === 'amend') {
    if (!current || current.variant_id !== variantId || revision !== current.revision + 1) return openAiError('Stale amendment', 409);
    if (!['user', 'assistant'].includes(String(body.role)) || (body.content !== null && (typeof body.content !== 'string' || !body.content.trim()))) return openAiError('Invalid amendment', 400);
    const rows = (await env.DB.prepare(`SELECT id, created_at FROM messages WHERE namespace = ? AND conversation_id = ?
      AND client_turn_id = ? AND role = ? AND (? = 'user' OR client_variant_id = ?)`)
      .bind(namespace, conversationId, turnId, body.role, body.role, variantId).all<{ id: string; created_at: string }>()).results;
    if (!rows.length && body.content !== null) return openAiError('Message not found', 409);
    const statements = await invalidateMessageDerivatives(env, namespace, conversationId, rows);
    // A new source ID makes in-flight extraction of the old text fail its source recheck.
    for (const row of rows) {
      const replacementId = newId('msg');
      const guard = `EXISTS (SELECT 1 FROM reply_selections WHERE namespace = ? AND conversation_id = ? AND turn_id = ? AND revision = ? AND variant_id = ?)`;
      const guardArgs = [namespace, conversationId, turnId, current.revision, variantId];
      statements.push(body.content === null
        ? env.DB.prepare(`DELETE FROM usage_logs WHERE namespace = ? AND message_id = ? AND ${guard}`).bind(namespace, row.id, ...guardArgs)
        : env.DB.prepare(`UPDATE usage_logs SET message_id = ? WHERE namespace = ? AND message_id = ? AND ${guard}`).bind(replacementId, namespace, row.id, ...guardArgs));
      statements.push(body.content === null
        ? env.DB.prepare(`DELETE FROM messages WHERE namespace = ? AND id = ? AND ${guard}`).bind(namespace, row.id, ...guardArgs)
        : env.DB.prepare(`UPDATE messages SET id = ?, content = ? WHERE namespace = ? AND id = ? AND ${guard}`).bind(replacementId, body.content, namespace, row.id, ...guardArgs));
    }
    statements.push(env.DB.prepare(`UPDATE reply_selections SET revision = ? WHERE namespace = ? AND conversation_id = ?
      AND turn_id = ? AND revision = ? AND variant_id = ?`).bind(revision, namespace, conversationId, turnId, current.revision, variantId));
    const changed = await env.DB.batch(statements);
    if (changed.at(-1)?.meta.changes !== 1) return openAiError('Stale amendment', 409);
    return json({ ok: true, revision });
  }

  if (body.action === "adopt") {
    if (current) return openAiError("Turn already tracked", 409);
    if (typeof body.userContent !== "string" || typeof body.assistantContent !== "string") {
      return openAiError("Missing legacy content", 400);
    }
    // Both sides of the exact pair must match uniquely inside this conversation.
    const candidates = await env.DB.prepare(`SELECT a.id, u.id AS user_id FROM messages a
      JOIN messages u ON u.rowid = (SELECT MAX(p.rowid) FROM messages p
        WHERE p.namespace = a.namespace AND p.conversation_id = a.conversation_id
          AND p.rowid < a.rowid AND p.role = 'user')
      WHERE a.namespace = ? AND a.conversation_id = ? AND a.role = 'assistant'
        AND a.client_turn_id IS NULL AND u.client_turn_id IS NULL
        AND a.content = ? AND u.content = ? LIMIT 2`)
      .bind(namespace, conversationId, body.assistantContent, body.userContent)
      .all<{ id: string; user_id: string }>();
    if (candidates.results?.length !== 1) return openAiError("Legacy reply could not be matched uniquely", 409, "legacy_reply_unresolved");
    const match = candidates.results[0];
    const adopted = await env.DB.batch([
      env.DB.prepare(`INSERT INTO reply_selections SELECT ?, ?, ?, ?, ?, 0
        WHERE (SELECT COUNT(*) FROM messages WHERE namespace = ? AND conversation_id = ?
          AND id IN (?, ?) AND client_turn_id IS NULL) = 2`)
        .bind(namespace, conversationId, turnId, variantId, revision, namespace, conversationId, match.id, match.user_id),
      env.DB.prepare(`UPDATE messages SET client_turn_id = ?, client_variant_id = ?
        WHERE namespace = ? AND conversation_id = ? AND id IN (?, ?) AND client_turn_id IS NULL`)
        .bind(turnId, variantId, namespace, conversationId, match.id, match.user_id),
    ]);
    if (adopted[0].meta.changes !== 1 || adopted[1].meta.changes !== 2) return openAiError("Legacy reply changed during adoption", 409);
    return json({ ok: true, revision });
  }
  if (body.action !== "select") return openAiError("Invalid action", 400);
  let exists = selectionReads?.[1].results[0];
  if (!exists && await persistMissingAssistantCandidate(
    body, env, namespace, conversationId, turnId, variantId,
  )) {
    exists = { id: variantId };
  }
  if (!exists && !(body.finalized === true && current?.variant_id === variantId)) return openAiError("Reply is not persisted yet", 409, "reply_not_persisted");
  const discardedTurns = Array.isArray(body.discardedTurns) ? body.discardedTurns : [];
  if (discardedTurns.length > 1000 || discardedTurns.some(id => typeof id !== "string" || !/^[a-zA-Z0-9_-]{1,100}$/.test(id) || id === turnId)) {
    return openAiError("Invalid discarded turns", 400);
  }
  const statements = [
    env.DB.prepare(`INSERT INTO reply_selections VALUES (?, ?, ?, ?, ?, ?)
      ON CONFLICT(namespace, conversation_id, turn_id) DO UPDATE SET
      variant_id = excluded.variant_id, revision = excluded.revision, finalized = excluded.finalized
      WHERE reply_selections.revision < excluded.revision`)
      .bind(namespace, conversationId, turnId, variantId, revision, body.finalized === true ? 1 : 0),
    env.DB.prepare(`UPDATE messages SET memory_active = CASE WHEN client_variant_id = ? THEN 1 ELSE 0 END
      WHERE namespace = ? AND conversation_id = ? AND client_turn_id = ?
      AND EXISTS (SELECT 1 FROM reply_selections s WHERE s.namespace = ? AND s.conversation_id = ?
        AND s.turn_id = ? AND s.revision = ? AND s.variant_id = ?)`)
      .bind(variantId, namespace, conversationId, turnId, namespace, conversationId, turnId, revision, variantId),
  ];
  if (body.finalized === true) {
    const obsolete = (await env.DB.prepare(`SELECT id, created_at FROM messages WHERE namespace = ? AND conversation_id = ?
      AND client_turn_id = ? AND client_variant_id <> ?`).bind(namespace, conversationId, turnId, variantId)
      .all<{ id: string; created_at: string }>()).results;
    statements.push(...await invalidateMessageDerivatives(env, namespace, conversationId, obsolete));
    statements.push(env.DB.prepare(`DELETE FROM usage_logs WHERE namespace = ? AND message_id IN (
      SELECT id FROM messages WHERE namespace = ? AND conversation_id = ? AND client_turn_id = ? AND client_variant_id <> ?)
      AND EXISTS (SELECT 1 FROM reply_selections WHERE namespace = ? AND conversation_id = ? AND turn_id = ? AND revision = ? AND variant_id = ?)`)
      .bind(namespace, namespace, conversationId, turnId, variantId, namespace, conversationId, turnId, revision, variantId));
    statements.push(env.DB.prepare(`DELETE FROM messages WHERE namespace = ? AND conversation_id = ?
      AND client_turn_id = ? AND client_variant_id <> ? AND EXISTS (
        SELECT 1 FROM reply_selections WHERE namespace = ? AND conversation_id = ? AND turn_id = ?
        AND revision = ? AND variant_id = ? AND finalized = 1)`)
      .bind(namespace, conversationId, turnId, variantId, namespace, conversationId, turnId, revision, variantId));
  }
  // Old branches have explicit turn identities; never delete by content or timestamp.
  for (const discarded of discardedTurns) {
    const obsolete = (await env.DB.prepare(`SELECT id, created_at FROM messages WHERE namespace = ? AND conversation_id = ? AND client_turn_id = ?`)
      .bind(namespace, conversationId, discarded).all<{ id: string; created_at: string }>()).results;
    statements.push(...await invalidateMessageDerivatives(env, namespace, conversationId, obsolete));
    statements.push(env.DB.prepare(`DELETE FROM usage_logs WHERE namespace = ? AND message_id IN (
      SELECT id FROM messages WHERE namespace = ? AND conversation_id = ? AND client_turn_id = ?)
      AND EXISTS (SELECT 1 FROM reply_selections WHERE namespace = ? AND conversation_id = ? AND turn_id = ? AND revision = ?)`)
      .bind(namespace, namespace, conversationId, discarded, namespace, conversationId, turnId, revision));
    statements.push(env.DB.prepare(`DELETE FROM messages WHERE namespace = ? AND conversation_id = ? AND client_turn_id = ?
      AND EXISTS (SELECT 1 FROM reply_selections WHERE namespace = ? AND conversation_id = ? AND turn_id = ? AND revision = ?)`)
      .bind(namespace, conversationId, discarded, namespace, conversationId, turnId, revision));
  }
  statements.push(env.DB.prepare(`SELECT revision, variant_id FROM reply_selections
    WHERE namespace = ? AND conversation_id = ? AND turn_id = ?`)
    .bind(namespace, conversationId, turnId));
  const committed = await env.DB.batch(statements);
  const saved = committed.at(-1)?.results[0] as { revision: number; variant_id: string } | undefined;
  if (saved?.revision !== revision || saved.variant_id !== variantId) return openAiError("Stale reply selection", 409);
  return json({ ok: true, revision });
}
