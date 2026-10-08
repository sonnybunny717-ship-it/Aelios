import type { SummaryEntry } from "../assembler/types";
import {
  advanceConversationContextState,
  getConversationContextState,
  initializeConversationContextState,
  type ConversationContextState,
} from "../db/conversations";
import { callOpenAICompat } from "../proxy/openaiAdapter";
import type {
  Env,
  MemoryApiRecord,
  OpenAIChatRequest,
  OpenAIChatResponse,
} from "../types";

const MAX_COMPACTION_MESSAGES = 60;
const MAX_COMPACTION_INPUT_CHARS = 80_000;
const WINDOW_SUMMARY_MAX_CHARS = 1_200;
const LONG_TERM_STATE_MAX_CHARS = 2_000;

interface ContextCompactionMessage {
  role: "user" | "assistant";
  content: string;
}

interface ContextCompaction {
  epoch: number;
  messages: ContextCompactionMessage[];
}

export interface PreparedConversationContext {
  epoch: number;
  pinnedPersonaMemories: MemoryApiRecord[];
  summaryEntry: SummaryEntry | null;
  summarySnapshot: {
    content: string | null;
    sourceUpdatedAt: string | null;
  };
}

function integerEpoch(value: unknown): number | null {
  return Number.isSafeInteger(value) && Number(value) >= 0 ? Number(value) : null;
}

export function parseContextCompaction(value: unknown): ContextCompaction | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const raw = value as Record<string, unknown>;
  const epoch = integerEpoch(raw.epoch);
  if (epoch === null || epoch < 1 || !Array.isArray(raw.messages)) return null;

  const messages: ContextCompactionMessage[] = [];
  let chars = 0;
  for (const item of raw.messages.slice(0, MAX_COMPACTION_MESSAGES)) {
    if (!item || typeof item !== "object" || Array.isArray(item)) continue;
    const record = item as Record<string, unknown>;
    if (record.role !== "user" && record.role !== "assistant") continue;
    const content = typeof record.content === "string" ? record.content.trim() : "";
    if (!content) continue;
    chars += content.length;
    if (chars > MAX_COMPACTION_INPUT_CHARS) {
      throw new Error("context_compaction is too large");
    }
    messages.push({ role: record.role, content });
  }
  if (messages.length === 0) return null;
  return { epoch, messages };
}

function parsePersonaSnapshot(value: string | null): MemoryApiRecord[] {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed as MemoryApiRecord[] : [];
  } catch {
    return [];
  }
}

function truncate(text: string, limit: number): string {
  const normalized = text.trim();
  if (normalized.length <= limit) return normalized;
  return `${normalized.slice(0, limit - 3)}...`;
}

export function buildConversationStateSummary(
  longTermSummary: string | null,
  windowSummary: string | null,
): SummaryEntry | null {
  const parts: string[] = [];
  if (longTermSummary?.trim()) {
    parts.push(`[每日交接]\n${truncate(longTermSummary, LONG_TERM_STATE_MAX_CHARS)}`);
  }
  if (windowSummary?.trim()) {
    parts.push(`[此前窗口]\n${truncate(windowSummary, WINDOW_SUMMARY_MAX_CHARS)}`);
  }
  return parts.length > 0 ? { content: parts.join("\n\n") } : null;
}

function extractJsonObject(text: string): Record<string, unknown> | null {
  try {
    const value = JSON.parse(text);
    return value && typeof value === "object" && !Array.isArray(value)
      ? value as Record<string, unknown>
      : null;
  } catch {}
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  try {
    const value = JSON.parse(text.slice(start, end + 1));
    return value && typeof value === "object" && !Array.isArray(value)
      ? value as Record<string, unknown>
      : null;
  } catch {
    return null;
  }
}

function buildWindowSummaryPrompt(
  previousSummary: string | null,
  messages: ContextCompactionMessage[],
): string {
  const transcript = messages
    .map((message) => `[${message.role === "assistant" ? "我" : "盼盼"}] ${message.content}`)
    .join("\n\n");
  return [
    "请把即将移出短期上下文的对话压缩成一份连续窗口备忘，供后续对话直接使用。",
    "只输出 JSON，不要 markdown，不要解释。",
    "称对方为“盼盼”，称助手为“我”，共同决定称“我们”；不要写成用户/助手报告。",
    "保留正在进行的话题、明确决定、待办、重要事实、情绪关系变化、承诺和必要的因果。",
    "忽略寒暄、重复表达、文风/格式要求、无关调试噪声和已被后续结论替代的实现尝试；保留仍影响当前任务的关键技术决定。",
    `content 不超过 ${WINDOW_SUMMARY_MAX_CHARS} 字。`,
    "",
    previousSummary?.trim()
      ? `上一份窗口备忘：\n${previousSummary.trim()}\n`
      : "上一份窗口备忘：无\n",
    "本次移出的对话：",
    transcript,
    "",
    '输出格式：{ "content": "合并后的窗口备忘" }',
  ].join("\n");
}

async function summarizeCompaction(
  env: Env,
  previousSummary: string | null,
  messages: ContextCompactionMessage[],
): Promise<string> {
  const model = env.WINDOW_SUMMARY_MODEL || env.DREAM_MODEL || env.SUMMARY_MODEL;
  if (!model) throw new Error("Missing WINDOW_SUMMARY_MODEL or DREAM_MODEL");

  const request: OpenAIChatRequest = {
    model,
    messages: [
      { role: "system", content: "你是严格的 JSON 生成器，只输出 JSON。" },
      { role: "user", content: buildWindowSummaryPrompt(previousSummary, messages) },
    ],
    temperature: 0,
    max_tokens: 8192,
    stream: false,
  };
  const response = await callOpenAICompat(env, request);
  if (!response.ok) throw new Error(`window compaction model returned ${response.status}`);

  let parsed: OpenAIChatResponse;
  try {
    parsed = await response.json() as OpenAIChatResponse;
  } catch {
    throw new Error("window compaction model returned invalid JSON response");
  }
  const raw = parsed.choices?.[0]?.message?.content;
  const json = typeof raw === "string" ? extractJsonObject(raw) : null;
  const content = typeof json?.content === "string" ? json.content.trim() : "";
  if (!content) throw new Error("window compaction model returned empty content");
  return truncate(content, WINDOW_SUMMARY_MAX_CHARS);
}

function preparedFromState(state: ConversationContextState): PreparedConversationContext {
  return {
    epoch: state.context_epoch,
    pinnedPersonaMemories: parsePersonaSnapshot(state.persona_snapshot_json),
    summaryEntry: buildConversationStateSummary(state.summary_snapshot, state.window_summary),
    summarySnapshot: {
      content: state.summary_snapshot,
      sourceUpdatedAt: state.summary_snapshot_source_updated_at,
    },
  };
}

export async function prepareConversationContext(
  env: Env,
  input: {
    conversationId: string;
    namespace: string;
    request: OpenAIChatRequest;
    latestSummary: { content: string; updated_at: string } | null;
    currentPinnedPersonaMemories: MemoryApiRecord[];
  },
): Promise<PreparedConversationContext> {
  const personaSnapshotJson = JSON.stringify(input.currentPinnedPersonaMemories);
  await initializeConversationContextState(env.DB, {
    conversationId: input.conversationId,
    namespace: input.namespace,
    summarySnapshot: input.latestSummary?.content ?? "",
    summarySnapshotSourceUpdatedAt: input.latestSummary?.updated_at ?? null,
    personaSnapshotJson,
  });

  let state = await getConversationContextState(
    env.DB,
    input.conversationId,
    input.namespace,
  );
  if (!state) throw new Error("Conversation context state not found");

  const requestedEpoch = integerEpoch(input.request.context_epoch) ?? state.context_epoch;
  const compaction = parseContextCompaction(input.request.context_compaction);
  if (compaction && compaction.epoch !== requestedEpoch) {
    throw new Error("context_compaction epoch does not match context_epoch");
  }

  if (requestedEpoch > state.context_epoch) {
    if (!compaction) throw new Error("context compaction payload is required for a new epoch");
    const windowSummary = await summarizeCompaction(
      env,
      state.window_summary,
      compaction.messages,
    );
    await advanceConversationContextState(env.DB, {
      conversationId: input.conversationId,
      namespace: input.namespace,
      epoch: requestedEpoch,
      windowSummary,
      summarySnapshot: input.latestSummary?.content ?? "",
      summarySnapshotSourceUpdatedAt: input.latestSummary?.updated_at ?? null,
      personaSnapshotJson,
    });
    state = await getConversationContextState(
      env.DB,
      input.conversationId,
      input.namespace,
    );
    if (!state) throw new Error("Conversation context state disappeared");
  }

  return preparedFromState(state);
}
