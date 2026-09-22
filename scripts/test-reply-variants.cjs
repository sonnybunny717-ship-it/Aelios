const assert = require('node:assert/strict');
const { readFileSync, readdirSync } = require('node:fs');
const { resolve } = require('node:path');
const test = require('node:test');
const ts = require('typescript');
const Database = require('../../garden-adapter/node_modules/better-sqlite3');
require.extensions['.ts'] = (module, filename) => module._compile(ts.transpileModule(readFileSync(filename, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText, filename);
const { saveUserMessages, saveAssistantMessage, listMessagesByNamespace, getMessagesByIds } = require('../src/db/messages.ts');
const { listRecentMessagesForSummary } = require('../src/db/summaries.ts');
const { handleReplySelection } = require('../src/api/replySelection.ts');
const { streamOpenAIWithTee } = require('../src/proxy/streamOpenAI.ts');
const { streamAnthropicToOpenAI } = require('../src/proxy/streamAnthropic.ts');
const { KEY_PROFILES } = require('../src/config/keyProfiles.ts');

function fixture() {
  const sqlite = new Database(':memory:');
  for (const file of readdirSync(resolve(__dirname, '../migrations')).filter(name => name.endsWith('.sql')).sort()) {
    sqlite.exec(readFileSync(resolve(__dirname, '../migrations', file), 'utf8'));
  }
  function prepared(sql, values = []) {
    return {
      bind: (...args) => prepared(sql, args),
      first: async () => sqlite.prepare(sql).get(...values) || null,
      all: async () => ({ results: sqlite.prepare(sql).all(...values) }),
      run: async () => execute(),
      execute,
    };
    function execute() {
      const statement = sqlite.prepare(sql);
      if (statement.reader) return { results: statement.all(...values), meta: { changes: 0 } };
      const result = statement.run(...values);
      return { results: [], meta: { changes: result.changes } };
    }
  }
  const DB = { prepare: prepared, batch: async statements => sqlite.transaction(() => statements.map(statement => statement.execute()))() };
  const env = { DB, CHATBOX_API_KEY: 'test', ENABLE_INCREMENTAL_MEMORY: 'false', MEMORY_QUEUE: { send: async () => {} } };
  const select = (body, conversation = 'chat', key = 'test') => handleReplySelection(new Request(`https://test/v1/conversations/${conversation}/reply-selection`, {
    method: 'POST', headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }), env);
  const pair = async (turnId, variantId, content = variantId, namespace = 'default', conversation = 'chat') => {
    const input = { conversationId: `${namespace}:${conversation}`, namespace, source: 'test', requestModel: 'test', upstreamModel: 'test',
      upstreamProvider: 'test', provider: 'test', stream: false,
      ...(turnId ? { replyVariant: { turnId, variantId } } : {}) };
    const users = await saveUserMessages(DB, { ...input, messages: [{ role: 'user', content: 'question' }] });
    const assistant = await saveAssistantMessage(DB, { ...input, content });
    return [...users, assistant];
  };
  return { sqlite, DB, env, pair, select };
}

test('only selected pair enters Dream and summary; finalize removes alternatives; revisions fence retries', async () => {
  const { sqlite, DB, pair, select } = fixture();
  try {
    const first = await pair('turn1', 'v1');
    const second = await pair('turn1', 'v2');
    await pair('turn2', 'v3', 'same content');
    await pair('turn1', 'v1', 'foreign', 'foreign');
    assert.equal((await listMessagesByNamespace(DB, 'default', null, 100)).length, 0);
    assert.equal((await select({ action: 'select', turnId: 'turn1', variantId: 'v1', revision: 1 })).status, 200);
    assert.deepEqual((await listMessagesByNamespace(DB, 'default', null, 100)).map(row => row.id), first);
    assert.equal((await select({ action: 'select', turnId: 'turn1', variantId: 'v2', revision: 2 })).status, 200);
    assert.equal((await getMessagesByIds(DB, { namespace: 'default', ids: first })).length, 0);
    assert.deepEqual((await listRecentMessagesForSummary(DB, 'default', 100)).map(row => row.id), second);
    assert.equal((await select({ action: 'select', turnId: 'turn1', variantId: 'v1', revision: 1 })).status, 409);
    const final = { action: 'select', turnId: 'turn1', variantId: 'v2', revision: 3, finalized: true };
    assert.equal((await select(final)).status, 200);
    assert.equal((await select(final)).status, 200);
    assert.equal(sqlite.prepare('SELECT count(*) n FROM messages WHERE id IN (?, ?)').get(...first).n, 0);
    assert.equal(sqlite.prepare("SELECT count(*) n FROM messages WHERE namespace = 'foreign'").get().n, 2);
    assert.equal(sqlite.prepare("SELECT count(*) n FROM messages WHERE client_turn_id = 'turn2'").get().n, 2);
    assert.equal((await select({ ...final, revision: 4 }, 'other')).status, 409);
    assert.equal((await select(final, 'chat', 'wrong')).status, 401);
  } finally { sqlite.close(); }
});

test('batched selection preserves replay, missing-candidate and concurrent-revision guards', async () => {
  const { sqlite, DB, pair, select } = fixture();
  try {
    await pair('turn', 'v1'); await pair('turn', 'v2');
    const batch = DB.batch;
    let calls = 0;
    DB.batch = async statements => { calls++; return batch(statements); };
    const first = { action: 'select', turnId: 'turn', variantId: 'v1', revision: 1 };
    assert.equal((await select(first)).status, 200);
    assert.equal(calls, 2);
    assert.equal((await select(first)).status, 200);
    assert.equal((await select({ ...first, variantId: 'v2' })).status, 409);
    assert.equal((await select({ ...first, revision: 2, variantId: 'missing' })).status, 409);
    assert.equal((await select({ ...first, finalized: true })).status, 409);
    let injected = false;
    DB.batch = async statements => {
      const result = await batch(statements);
      if (!injected) {
        injected = true;
        sqlite.prepare('UPDATE reply_selections SET revision = 3').run();
      }
      return result;
    };
    assert.equal((await select({ ...first, variantId: 'v2', revision: 2 })).status, 409);
    assert.equal(sqlite.prepare('SELECT variant_id FROM reply_selections').get().variant_id, 'v1');
    assert.deepEqual(sqlite.prepare('SELECT DISTINCT client_variant_id FROM messages WHERE memory_active=1').all(), [{ client_variant_id: 'v1' }]);
  } finally { sqlite.close(); }
});

test('selection restores an interrupted assistant candidate when the stream ended before persistence', async () => {
  const { sqlite, DB, select } = fixture();
  try {
    await saveUserMessages(DB, {
      conversationId: 'default:chat', namespace: 'default', source: 'test',
      requestModel: 'request-model', upstreamModel: 'upstream-model', upstreamProvider: 'test',
      stream: true, replyVariant: { turnId: 'turn-interrupted', variantId: 'v-partial' },
      messages: [{ role: 'user', content: 'question' }],
    });
    const selection = {
      action: 'select', turnId: 'turn-interrupted', variantId: 'v-partial', revision: 1,
      assistantContent: 'partial answer', assistantReasoning: 'paid reasoning', deliveryStatus: 'interrupted',
    };
    assert.equal((await select(selection)).status, 200);
    assert.equal((await select(selection)).status, 200);
    const assistant = sqlite.prepare(`SELECT content, reasoning_content, finish_reason, memory_active
      FROM messages WHERE client_variant_id = 'v-partial' AND role = 'assistant'`).get();
    assert.deepEqual(assistant, {
      content: 'partial answer', reasoning_content: 'paid reasoning',
      finish_reason: 'interrupted', memory_active: 1,
    });
    assert.equal(sqlite.prepare(`SELECT COUNT(*) AS n FROM messages
      WHERE client_variant_id = 'v-partial' AND role = 'assistant'`).get().n, 1);
    assert.equal((await select({ ...selection, turnId: 'missing-turn', variantId: 'missing', revision: 2 })).status, 409);
  } finally { sqlite.close(); }
});

test('edited prompt removes remote variants without retaining an archive row', async () => {
  const { sqlite, pair, select } = fixture();
  try {
    await pair('turn-edit', 'v1', 'first answer');
    await pair('turn-edit', 'v2', 'second answer');
    assert.equal((await select({ action: 'select', turnId: 'turn-edit', variantId: 'v1', revision: 1 })).status, 200);
    assert.equal((await select({ action: 'select', turnId: 'turn-edit', variantId: 'v2', revision: 2 })).status, 200);

    const archive = {
      action: 'archive_edit', archiveId: 'archive-one', turnId: 'turn-edit', variantId: 'v2', revision: 2,
      draftContent: 'edited question', discardedTurns: [],
    };
    assert.equal((await select(archive)).status, 200);
    assert.equal(sqlite.prepare("SELECT count(*) n FROM messages WHERE client_turn_id = 'turn-edit'").get().n, 0);
    assert.equal(sqlite.prepare("SELECT count(*) n FROM reply_selections WHERE turn_id = 'turn-edit'").get().n, 0);
    assert.equal(sqlite.prepare("SELECT count(*) n FROM reply_edit_archives").get().n, 0);

    assert.equal((await select({ ...archive, draftContent: 'edited question twice' })).status, 200);
    assert.equal((await select({ action: 'update_edit_draft', archiveId: 'archive-one', draftContent: 'final wording' })).status, 200);
    assert.equal((await select({ action: 'delete_edit_archive', archiveId: 'archive-one' })).status, 200);
    assert.equal((await select({ action: 'delete_edit_archive', archiveId: 'archive-one' })).status, 200);
    assert.equal(sqlite.prepare("SELECT count(*) n FROM reply_edit_archives").get().n, 0);
  } finally { sqlite.close(); }
});

test('legacy adoption requires a unique exact pair; ambiguous history is untouched', async () => {
  const { sqlite, pair, select } = fixture();
  try {
    await pair(null, null, 'legacy');
    const adopt = { action: 'adopt', turnId: 'old', variantId: 'v1', revision: 1, userContent: 'question', assistantContent: 'legacy' };
    assert.equal((await select(adopt)).status, 200);
    assert.equal((await select(adopt)).status, 200);
    await pair(null, null, 'duplicate'); await pair(null, null, 'duplicate');
    assert.equal((await select({ ...adopt, turnId: 'ambiguous', assistantContent: 'duplicate' })).status, 409);
    assert.equal(sqlite.prepare('SELECT count(*) n FROM messages WHERE client_turn_id IS NULL AND memory_active = 1').get().n, 4);
  } finally { sqlite.close(); }
});

for (const provider of ['openai', 'anthropic']) test(`${provider} managed stream is persisted before DONE`, async () => {
  const { sqlite, env } = fixture();
  try {
    const event = value => `data: ${JSON.stringify(value)}\n\n`;
    const data = provider === 'openai'
      ? event({ choices: [{ delta: { content: 'answer' }, finish_reason: 'stop' }] }) + 'data: [DONE]\n\n'
      : event({ type: 'content_block_delta', delta: { type: 'text_delta', text: 'answer' } })
        + event({ type: 'message_delta', delta: { stop_reason: 'end_turn' } });
    const upstream = new Response(data);
    const stream = provider === 'openai' ? streamOpenAIWithTee : streamAnthropicToOpenAI;
    const response = stream(upstream, { env, ctx: { waitUntil() {} }, profile: KEY_PROFILES.chatbox,
      conversationId: 'default:chat', requestModel: 'test', upstreamModel: 'test', provider,
      replyVariant: { turnId: 'turn', variantId: 'candidate' } });
    const reader = response.body.getReader(); let text = '';
    for (;;) {
      const { done, value } = await reader.read(); if (done) break;
      text += new TextDecoder().decode(value);
      if (text.includes('[DONE]')) {
        const row = sqlite.prepare("SELECT * FROM messages WHERE client_variant_id = 'candidate'").get();
        assert.equal(row.content, 'answer'); assert.equal(row.memory_active, 0);
      }
    }
    assert.ok(text.includes('[DONE]'));
  } finally { sqlite.close(); }
});

test('openai managed stream persists visible answer routed through reasoning_content on truncation', async () => {
  const { sqlite, env } = fixture();
  try {
    const event = value => `data: ${JSON.stringify(value)}\n\n`;
    const data = event({ choices: [{ delta: { reasoning_content: '<thinking>内部推理</thin' }, finish_reason: null }] })
      + event({ choices: [{ delta: { reasoning_content: 'king>截断前仍然产生的正文' }, finish_reason: null }] })
      + event({ choices: [{ delta: { content: ' ' }, finish_reason: 'length' }] })
      + 'data: [DONE]\n\n';
    const response = streamOpenAIWithTee(new Response(data), {
      env, ctx: { waitUntil() {} }, profile: KEY_PROFILES.chatbox,
      conversationId: 'default:chat', requestModel: 'test', upstreamModel: 'test', provider: 'openai',
      replyVariant: { turnId: 'turn', variantId: 'truncated' },
    });
    assert.ok((await response.text()).includes('[DONE]'));
    const row = sqlite.prepare("SELECT content, finish_reason FROM messages WHERE client_variant_id = 'truncated'").get();
    assert.equal(row.content, '截断前仍然产生的正文 ');
    assert.equal(row.finish_reason, 'length');
  } finally { sqlite.close(); }
});

for (const provider of ['openai', 'anthropic']) test(`${provider} interrupted stream with visible text becomes an inactive candidate`, async () => {
  const { sqlite, env } = fixture();
  try {
    const event = provider === 'openai'
      ? { choices: [{ delta: { content: 'partial' }, finish_reason: null }] }
      : { type: 'content_block_delta', delta: { type: 'text_delta', text: 'partial' } };
    const stream = provider === 'openai' ? streamOpenAIWithTee : streamAnthropicToOpenAI;
    const response = stream(new Response(`data: ${JSON.stringify(event)}\n\n`), {
      env, ctx: { waitUntil() {} }, profile: KEY_PROFILES.chatbox,
      conversationId: 'default:chat', requestModel: 'test', upstreamModel: 'test', provider,
      replyVariant: { turnId: 'turn', variantId: 'interrupted' },
    });
    await assert.rejects(response.text(), /finish reason/);
    const row = sqlite.prepare("SELECT content, finish_reason, memory_active FROM messages WHERE client_variant_id = 'interrupted'").get();
    assert.equal(row.content, 'partial');
    assert.equal(row.finish_reason, 'interrupted');
    assert.equal(row.memory_active, 0);
  } finally { sqlite.close(); }
});

for (const provider of ['openai', 'anthropic']) test(`${provider} interrupted stream without visible text is not persisted`, async () => {
  const { sqlite, env } = fixture();
  try {
    const stream = provider === 'openai' ? streamOpenAIWithTee : streamAnthropicToOpenAI;
    const response = stream(new Response(''), {
      env, ctx: { waitUntil() {} }, profile: KEY_PROFILES.chatbox,
      conversationId: 'default:chat', requestModel: 'test', upstreamModel: 'test', provider,
      replyVariant: { turnId: 'turn', variantId: 'empty-interrupted' },
    });
    await assert.rejects(response.text(), /finish reason/);
    assert.equal(sqlite.prepare("SELECT count(*) n FROM messages WHERE client_variant_id = 'empty-interrupted'").get().n, 0);
  } finally { sqlite.close(); }
});

for (const provider of ['openai', 'anthropic']) test(`${provider} interrupted stream with only reasoning becomes an inactive candidate`, async () => {
  const { sqlite, env } = fixture();
  try {
    const event = provider === 'openai'
      ? { choices: [{ delta: { reasoning_content: '花掉的思考' }, finish_reason: null }] }
      : { type: 'content_block_delta', delta: { type: 'thinking_delta', thinking: '花掉的思考' } };
    const stream = provider === 'openai' ? streamOpenAIWithTee : streamAnthropicToOpenAI;
    const response = stream(new Response(`data: ${JSON.stringify(event)}\n\n`), {
      env, ctx: { waitUntil() {} }, profile: KEY_PROFILES.chatbox,
      conversationId: 'default:chat', requestModel: 'test', upstreamModel: 'test', provider,
      replyVariant: { turnId: 'turn', variantId: 'reasoning-only' },
    });
    await assert.rejects(response.text(), /finish reason/);
    const row = sqlite.prepare("SELECT content, reasoning_content, finish_reason, memory_active FROM messages WHERE client_variant_id = 'reasoning-only'").get();
    assert.equal(row.content, '');
    assert.equal(row.reasoning_content, '花掉的思考');
    assert.equal(row.finish_reason, 'interrupted');
    assert.equal(row.memory_active, 0);
  } finally { sqlite.close(); }
});

test('adapter and Aelios complete generation, switching, recovery, and finalization together', async () => {
  const { mkdtempSync, rmSync } = require('node:fs');
  const { tmpdir } = require('node:os');
  const { SessionStore } = await import('../../garden-adapter/session-store.js');
  const { ReplyVersions, syncReplySelection } = await import('../../garden-adapter/reply-versions.js');
  const { sqlite, DB, env, pair } = fixture();
  const directory = mkdtempSync(resolve(tmpdir(), 'reply-integration-'));
  const store = new SessionStore({ dbPath: resolve(directory, 'chat.db') });
  const session = store.createSession('panpan', 'integration', { engine: 'api' });
  let offline = false;
  const versions = new ReplyVersions(store, (session, payload) => syncReplySelection({
    baseUrl: 'https://test', apiKey: 'test', conversationId: session.id, payload,
    fetchImpl: async (url, options) => {
      if (offline) throw new Error('offline');
      return handleReplySelection(new Request(url, options), env);
    },
  }));
  try {
    const staged = store.stageApiUserMessage(session.id, { text: 'question' });
    let prepared = store.beginApiGeneration(session.id, { turnId: staged.turnId });
    await versions.prepare(session, prepared, 'question', '');
    await pair(prepared.turnId, prepared.variantId, 'first', 'default', session.id);
    const first = await versions.finish(session, prepared, [{ text: 'first', ts: Date.now() }], {});
    prepared = store.beginApiGeneration(session.id, { fromAssistantMessageId: first.messageIds[0] });
    await versions.prepare(session, prepared, 'question', 'first');
    await pair(prepared.turnId, prepared.variantId, 'second', 'default', session.id);
    offline = true;
    await assert.rejects(versions.finish(session, prepared, [{ text: 'second', ts: Date.now() }], {}), /offline/);
    assert.equal((await listMessagesByNamespace(DB, 'default', null, 100)).at(-1).content, 'first');
    assert.equal(store.getMessages(session.id).at(-1).text, 'first');
    offline = false;
    await versions.flush(session);
    assert.equal((await listMessagesByNamespace(DB, 'default', null, 100)).at(-1).content, 'second');
    const state = versions.state(session.id)[0];
    await versions.select(session, { turnId: state.turnId, variantId: state.variants[0], revision: state.revision });
    assert.equal(store.getMessages(session.id).at(-1).text, 'first');
    assert.equal((await listMessagesByNamespace(DB, 'default', null, 100)).at(-1).content, 'first');
    await versions.finalize(session);
    assert.equal(sqlite.prepare("SELECT count(*) n FROM messages WHERE content = 'second'").get().n, 0);
    assert.deepEqual(versions.state(session.id), []);
  } finally { store.stop(); sqlite.close(); rmSync(directory, { recursive: true, force: true }); }
});

test('amend removes obsolete sources and derived memories, retries are idempotent, last bubble can finalize', async () => {
  const { sqlite, DB, env, pair, select } = fixture();
  try {
    const ids = await pair('turn', 'v1', 'old text');
    await pair('turn', 'v2', 'alternative');
    await select({ action: 'select', turnId: 'turn', variantId: 'v1', revision: 1 });
    sqlite.prepare(`INSERT INTO memories (id, namespace, type, content, source_message_ids, created_at, updated_at) VALUES ('derived', 'default', 'note', 'old fact', ?, 'now', 'now')`).run(JSON.stringify([ids[1]]));
    const { writeCursor } = require('../src/db/retention.ts');
    await writeCursor(DB, 'vector_memory_sources_backfill:v1:default', 'done');
    sqlite.prepare(`INSERT INTO vector_memory_sources (namespace, memory_id, vector_id, type, source_message_ids, updated_at) VALUES ('default', 'vector-derived', 'vec-derived', 'note', ?, 'now')`).run(JSON.stringify([ids[1]]));
    const deletedVectors = [];
    let vectorFailure = true;
    env.VECTORIZE = { deleteByIds: async ids => { if (vectorFailure) throw new Error('vector unavailable'); deletedVectors.push(...ids); } };
    const edit = { action: 'amend', turnId: 'turn', variantId: 'v1', revision: 2, role: 'assistant', content: 'corrected text' };
    await assert.rejects(select(edit), /vector unavailable/);
    assert.equal(sqlite.prepare('SELECT content FROM messages WHERE id = ?').get(ids[1]).content, 'old text');
    vectorFailure = false;
    assert.equal((await select(edit)).status, 200);
    assert.deepEqual(deletedVectors, ['vec-derived']);
    assert.equal(sqlite.prepare("SELECT count(*) n FROM vector_memory_sources WHERE memory_id = 'vector-derived'").get().n, 0);
    assert.equal((await getMessagesByIds(DB, { namespace: 'default', ids: [ids[1]] })).length, 0);
    assert.equal(sqlite.prepare("SELECT count(*) n FROM memories WHERE id = 'derived'").get().n, 0);
    assert.equal(sqlite.prepare("SELECT content FROM messages WHERE client_variant_id = 'v1' AND role = 'assistant'").get().content, 'corrected text');
    assert.equal((await select(edit)).status, 200);
    assert.equal((await select({ ...edit, content: 'conflict' })).status, 409);
    assert.equal((await select({ ...edit, revision: 3, role: 'user', content: 'edited question' })).status, 200);
    assert.equal(sqlite.prepare("SELECT count(*) n FROM messages WHERE role = 'user' AND content = 'edited question'").get().n, 2);
    assert.equal((await select({ ...edit, revision: 4, content: null })).status, 200);
    assert.equal((await select({ action: 'select', turnId: 'turn', variantId: 'v1', revision: 5, finalized: true })).status, 200);
    assert.equal(sqlite.prepare("SELECT count(*) n FROM messages WHERE role = 'assistant'").get().n, 0);
  } finally { sqlite.close(); }
});
