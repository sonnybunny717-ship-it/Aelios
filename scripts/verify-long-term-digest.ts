import assert from "node:assert/strict";

import {
  buildLongTermDigestPrompt,
  normalizeLongTermDigestResult,
  shouldIncludeCompleteTurn
} from "../src/memory/longTermDigest";
import { formatMemoryPatch } from "../src/memory/inject";
import type { MemoryApiRecord } from "../src/types";

function memory(input: Partial<MemoryApiRecord> & Pick<MemoryApiRecord, "id" | "type" | "content">): MemoryApiRecord {
  return {
    id: input.id,
    namespace: "default",
    type: input.type,
    content: input.content,
    summary: null,
    importance: input.importance ?? 0.8,
    confidence: input.confidence ?? 0.9,
    status: "active",
    pinned: false,
    tags: input.tags ?? [],
    source: "dream",
    source_message_ids: input.source_message_ids ?? [],
    vector_id: null,
    last_recalled_at: null,
    recall_count: 0,
    created_at: "2026-09-21T00:00:00.000Z",
    updated_at: "2026-09-21T00:00:00.000Z",
    expires_at: null
  };
}

const existing = [
  memory({ id: "mem_project", type: "project", content: "旧项目记忆", source_message_ids: ["90"] }),
  memory({ id: "mem_theater", type: "theater", content: "【小剧场｜猫猫公司】旧剧情" })
];
const normalized = normalizeLongTermDigestResult({
  memories_to_update: [
    {
      target_id: "mem_project",
      type: "project",
      content: "【2026-09-21】更新后的项目记忆",
      source_message_ids: [101, 102]
    },
    {
      target_id: "mem_theater",
      type: "theater",
      content: "【小剧场｜猫猫公司】累计剧情",
      source_message_ids: [999]
    }
  ],
  memories_to_add: [
    {
      type: "project",
      content: "不应重复新增的项目记忆",
      source_message_ids: [102, 101]
    },
    {
      type: "theater",
      content: "【小剧场｜猫猫公司】不应重复新增的剧情",
      source_message_ids: []
    },
    {
      type: "relationship",
      content: "【2026-09-21】留下来的关系记忆",
      source_message_ids: [103, 999]
    },
    {
      type: "note",
      content: "没有来源，不应保存",
      source_message_ids: []
    }
  ],
  memories_to_delete: [{ target_id: "mem_project" }]
}, {
  allowedSourceIds: new Set(["101", "102", "103"]),
  existingMemories: existing
});

assert.equal(normalized.memories_to_update.length, 2);
assert.equal(normalized.memories_to_add.length, 1);
assert.deepEqual(normalized.memories_to_update[1].source_message_ids, []);
assert.deepEqual(normalized.memories_to_add[0].source_message_ids, ["103"]);
assert.equal("memories_to_delete" in normalized, false);

const prompt = buildLongTermDigestPrompt({
  existingMemories: existing,
  transcript: '<turn id="turn-1">\n[101][2026-09-21][盼盼] 测试\n</turn>'
});
assert.match(prompt, /先完成全部 memories_to_update/);
assert.match(prompt, /不得输出 memories_to_delete/);
assert.match(prompt, /最多 16 条/);

const patch = formatMemoryPatch([
  memory({
    id: "mem_recall",
    type: "relationship",
    content: "【2026-09-21】我们记住了这件事",
    source_message_ids: ["101", "msg_old", "103"]
  })
]);
assert.match(patch, /我们记住了这件事 \[source=101,103\]/);
assert.doesNotMatch(patch, /msg_old/);

assert.equal(shouldIncludeCompleteTurn(0, 0, 35_000), true);
assert.equal(shouldIncludeCompleteTurn(1, 29_000, 999), false);
assert.equal(shouldIncludeCompleteTurn(19, 10_000, 1_000), true);
assert.equal(shouldIncludeCompleteTurn(20, 10_000, 1), false);

console.log("long-term digest verification passed");
